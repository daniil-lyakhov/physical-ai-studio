# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the XR0 Lightning policy wrapper.

Fast, self-contained tests with no external dependencies (no HuggingFace model
downloads). The full model / preprocessor pipeline (which loads Qwen3-VL-4B) is
covered separately; these tests exercise the lazy-init path, config wiring,
hyperparameter capture, error handling, and the policy factory.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from physicalai.data import Observation
from physicalai.data.observation import ACTION, IMAGES, STATE, TASK
from physicalai.export import ExportBackend
from physicalai.export.backends import TorchExportParameters
from physicalai.inference.data import InferenceFeatureType
from physicalai.policies import get_physicalai_policy_class, get_policy
from physicalai.policies.xr0 import XR0, XR0Config


def _minimal_export_stats() -> dict[str, dict[str, Any]]:
    """Return minimal dataset statistics for exercising the export hooks."""
    return {
        "observation.state": {
            "name": "observation.state",
            "shape": (8,),
            "mean": [0.0] * 8,
            "std": [1.0] * 8,
            "type": "STATE",
        },
        "observation.images.base": {
            "name": "observation.images.base",
            "shape": (3, 256, 256),
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "type": "VISUAL",
        },
        "action": {
            "name": "action",
            "shape": (6,),
            "mean": [0.0] * 6,
            "std": [1.0] * 6,
            "type": "ACTION",
        },
    }


class TestXR0Config:
    """Config resolution through the policy constructor."""

    def test_lazy_initialization(self) -> None:
        policy = XR0()
        assert policy.model is None
        assert policy._preprocessor is None
        assert policy._postprocessor is None
        assert policy._dataset_stats is None

    def test_config_wiring(self) -> None:
        policy = XR0(chunk_size=30, n_action_steps=15, optimizer_lr=1e-4, dtype="float32")
        assert isinstance(policy.config, XR0Config)
        assert policy.config.chunk_size == 30
        assert policy.config.n_action_steps == 15
        assert policy.config.optimizer_lr == 1e-4
        assert policy.config.dtype == "float32"
        assert policy._n_action_steps == 15

    def test_hyperparameters_saved(self) -> None:
        policy = XR0(chunk_size=30, optimizer_lr=1e-4, freeze_vision_encoder=True)
        assert policy.hparams.chunk_size == 30
        assert policy.hparams.optimizer_lr == 1e-4
        assert policy.hparams.freeze_vision_encoder is True
        assert "config" in policy.hparams
        assert policy.hparams["config"]["chunk_size"] == 30


class TestXR0Policy:
    """Policy behaviour without an initialized model."""

    @pytest.mark.parametrize("method", ["forward", "predict_action_chunk"])
    def test_methods_raise_without_model(self, method: str) -> None:
        policy = XR0()
        obs = Observation(state=torch.randn(1, 8))
        with pytest.raises(ValueError, match="not initialized"):
            getattr(policy, method)(obs)

    def test_eval_forward_dispatches_to_predict(self) -> None:
        policy = XR0().eval()
        obs = Observation(state=torch.randn(1, 8))
        # eval forward routes to predict_action_chunk, which raises without a model
        with pytest.raises(ValueError, match="not initialized"):
            policy(obs)


class TestXR0Factory:
    """Policy factory registration."""

    def test_factory_class(self) -> None:
        assert get_physicalai_policy_class("xr0") is XR0

    def test_get_policy(self) -> None:
        policy = get_policy("xr0", chunk_size=30, optimizer_lr=1e-4)
        assert isinstance(policy, XR0)
        assert policy.model is None


class TestXR0Export:
    """Torch export hooks (no model download)."""

    def test_supported_backends_torch_only(self) -> None:
        assert XR0.get_supported_export_backends() == [ExportBackend.TORCH]

    def test_schemas_none_before_init(self) -> None:
        policy = XR0()
        assert policy.inputs_schema is None
        assert policy.outputs_schema is None

    def test_extra_export_args_torch_only(self) -> None:
        policy = XR0()
        extra = policy.extra_export_args
        assert set(extra) == {"torch"}
        assert isinstance(extra["torch"], TorchExportParameters)

    def test_extra_export_args_trims_when_chunk_differs(self) -> None:
        trimmed = XR0(chunk_size=30, n_action_steps=15).extra_export_args["torch"]
        assert any(spec.type == "action_chunk_trimmer" for spec in trimmed.postprocessors_specs)

        untrimmed = XR0(chunk_size=30, n_action_steps=30).extra_export_args["torch"]
        assert all(spec.type != "action_chunk_trimmer" for spec in untrimmed.postprocessors_specs)

    def test_inputs_schema_from_stats(self) -> None:
        policy = XR0(chunk_size=30)
        policy.model = object()  # type: ignore[assignment]  # sentinel to bypass lazy-init guard
        policy._dataset_stats = _minimal_export_stats()

        schema = policy.inputs_schema
        assert schema is not None
        by_name = {feature.name: feature for feature in schema}
        assert set(by_name) == {STATE, IMAGES, TASK}
        assert by_name[STATE].ftype is InferenceFeatureType.STATE
        assert by_name[STATE].shape == (8,)
        assert by_name[IMAGES].ftype is InferenceFeatureType.VISUAL
        assert by_name[TASK].ftype is InferenceFeatureType.LANGUAGE

    def test_outputs_schema_from_stats(self) -> None:
        policy = XR0(chunk_size=30)
        policy.model = object()  # type: ignore[assignment]  # sentinel to bypass lazy-init guard
        policy._dataset_stats = _minimal_export_stats()

        schema = policy.outputs_schema
        assert schema is not None
        assert len(schema) == 1
        assert schema[0].name == ACTION
        assert schema[0].ftype is InferenceFeatureType.ACTION
        assert schema[0].shape == (30, 6)
