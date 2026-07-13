# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the XR0 Lightning policy wrapper.

Fast, self-contained tests with no external dependencies (no HuggingFace model
downloads). The full model / preprocessor pipeline (which loads Qwen3-VL-4B) is
covered separately; these tests exercise the lazy-init path, config wiring,
hyperparameter capture, error handling, and the policy factory.
"""

from __future__ import annotations

import pytest
import torch

from physicalai.data import Observation
from physicalai.policies import get_physicalai_policy_class, get_policy
from physicalai.policies.xr0 import XR0, XR0Config


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
