# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the XR0 Lightning policy wrapper.

Fast, self-contained tests with no external dependencies (no HuggingFace model
downloads). The full model / preprocessor pipeline (which loads Qwen3-VL-4B) is
covered separately; these tests exercise the lazy-init path, config wiring,
hyperparameter capture, error handling, and the policy factory.
"""

from __future__ import annotations

import types
from typing import Any

import pytest
import torch

from physicalai.data import Feature, FeatureType, Observation
from physicalai.data.observation import ACTION, IMAGES, STATE, TASK
from physicalai.export import ExportBackend
from physicalai.export.backends import TorchExportParameters
from physicalai.inference.data import InferenceFeatureType
from physicalai.policies import get_physicalai_policy_class, get_policy
from physicalai.policies.xr0 import XR0, XR0Config
from physicalai.policies.xr0.export_openvino import ov_friendly_rmsnorm_forward
from physicalai.policies.xr0.vlm import (
    export_add_deepstack_embeds,
    export_build_additive_causal_mask,
    export_fast_pos_embed_interpolate,
    export_rot_pos_emb,
    export_scatter_visual_embeds,
    export_vision_attn_forward,
)


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


class TestXR0Features:
    """Explicit input/output feature schema handling (no model download)."""

    def test_requires_both_features(self) -> None:
        state = Feature(name="state", ftype=FeatureType.STATE, shape=(8,))
        with pytest.raises(ValueError, match="both input and output features"):
            XR0(input_features=[state], output_features=None)

    def test_feature_properties_raise_before_init(self) -> None:
        policy = XR0()
        with pytest.raises(ValueError, match="no input features"):
            _ = policy.input_features
        with pytest.raises(ValueError, match="no output features"):
            _ = policy.output_features

    def test_features_stats_roundtrip(self) -> None:
        inputs = [
            Feature(name="state", ftype=FeatureType.STATE, shape=(8,)),
            Feature(name="base", ftype=FeatureType.VISUAL, shape=(3, 256, 256)),
        ]
        outputs = [Feature(name=ACTION, ftype=FeatureType.ACTION, shape=(6,))]

        stats = XR0._features_to_stats(inputs, outputs)
        assert set(stats) == {"observation.state", "observation.base", ACTION}

        recon_inputs, recon_outputs = XR0._stats_to_features(stats)
        assert {f.name for f in recon_inputs} == {"state", "base"}
        assert {f.ftype for f in recon_inputs} == {FeatureType.STATE, FeatureType.VISUAL}
        assert [f.name for f in recon_outputs] == [ACTION]
        assert recon_outputs[0].ftype is FeatureType.ACTION
        assert recon_outputs[0].shape == (6,)


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

    def test_inputs_schema_from_features(self) -> None:
        policy = XR0(chunk_size=30)
        policy.model = object()  # type: ignore[assignment]  # sentinel to bypass lazy-init guard
        # Set the features directly rather than via the constructor: passing
        # input/output features to ``XR0(...)`` derives dataset stats from them,
        # which triggers the eager model build (Qwen3-VL-4B download). Here we
        # mimic that reconstructed schema without the build.
        policy._input_features, policy._output_features = XR0._stats_to_features(_minimal_export_stats())

        schema = policy.inputs_schema
        assert schema is not None
        by_name = {feature.name: feature for feature in schema}
        assert set(by_name) == {STATE, IMAGES, TASK}
        assert by_name[STATE].ftype is InferenceFeatureType.STATE
        assert by_name[STATE].shape == (8,)
        assert by_name[IMAGES].ftype is InferenceFeatureType.VISUAL
        assert by_name[TASK].ftype is InferenceFeatureType.LANGUAGE

    def test_outputs_schema_from_features(self) -> None:
        policy = XR0(chunk_size=30)
        policy.model = object()  # type: ignore[assignment]  # sentinel to bypass lazy-init guard
        # Set features directly: passing them to ``XR0(...)`` would build dataset
        # stats and trigger the eager model build (Qwen3-VL-4B download).
        policy._input_features, policy._output_features = XR0._stats_to_features(_minimal_export_stats())

        schema = policy.outputs_schema
        assert schema is not None
        assert len(schema) == 1
        assert schema[0].name == ACTION
        assert schema[0].ftype is InferenceFeatureType.ACTION
        assert schema[0].shape == (30, 6)

    def test_multi_camera_inputs_schema_names_views(self) -> None:
        policy = XR0(chunk_size=30)
        policy.model = object()  # type: ignore[assignment]  # sentinel to bypass lazy-init guard
        # Set features directly: passing them to ``XR0(...)`` would build dataset
        # stats and trigger the eager model build (Qwen3-VL-4B download).
        policy._input_features = [
            Feature(name="state", ftype=FeatureType.STATE, shape=(8,)),
            Feature(name="base", ftype=FeatureType.VISUAL, shape=(3, 256, 256)),
            Feature(name="wrist_left", ftype=FeatureType.VISUAL, shape=(3, 256, 256)),
        ]
        policy._output_features = [Feature(name=ACTION, ftype=FeatureType.ACTION, shape=(6,))]

        names = {feature.name for feature in policy.inputs_schema or []}
        # Two cameras -> per-view names (no single-camera IMAGES collapse).
        assert names == {STATE, f"{IMAGES}.base", f"{IMAGES}.wrist_left", TASK}


class TestExportPatchParity:
    """Numerical parity of the export-friendly VLM ops against stock Qwen3-VL.

    ``XR0Qwen3VL._ensure_export_patch`` swaps a handful of stock Qwen3-VL ops for
    OpenVINO-friendly reimplementations (see the ``export_*`` module-level
    functions). These tests check each replacement independently, comparing it to
    the stock ``transformers`` op on small reference tensors. They deliberately
    build tiny weight-free / small-module fixtures instead of loading the 4B model
    so they stay fast and download-free; the export patch is numerically identical
    to stock, so the outputs must match to floating-point tolerance.
    """

    def test_rot_pos_emb_matches_stock(self) -> None:
        """``export_rot_pos_emb`` matches stock ``rot_pos_emb`` (weight-free)."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionModel,
            Qwen3VLVisionRotaryEmbedding,
        )

        torch.manual_seed(0)
        merge_size = 2
        # rotary emb is a deterministic ``inv_freq`` buffer -> no learned weights.
        visual = types.SimpleNamespace(
            spatial_merge_size=merge_size,
            rotary_pos_emb=Qwen3VLVisionRotaryEmbedding(8),
        )
        # Two images (multi-frame on the second) to exercise the repeat path.
        grid_thw = torch.tensor([[1, 4, 6], [2, 2, 8]])

        stock = Qwen3VLVisionModel.rot_pos_emb(visual, grid_thw)
        exported = export_rot_pos_emb(visual, grid_thw.tolist())

        assert exported.shape == stock.shape
        assert torch.equal(exported, stock)

    def test_fast_pos_embed_interpolate_matches_stock(self) -> None:
        """``export_fast_pos_embed_interpolate`` matches the stock interpolation."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        torch.manual_seed(0)
        num_grid_per_side = 6
        hidden = 16
        merge_size = 2
        pos_embed = torch.nn.Embedding(num_grid_per_side * num_grid_per_side, hidden)
        visual = types.SimpleNamespace(
            num_grid_per_side=num_grid_per_side,
            pos_embed=pos_embed,
            config=types.SimpleNamespace(spatial_merge_size=merge_size),
        )
        grid_thw = torch.tensor([[1, 4, 6], [1, 2, 8]])

        stock = Qwen3VLVisionModel.fast_pos_embed_interpolate(visual, grid_thw)
        exported = export_fast_pos_embed_interpolate(visual, grid_thw.tolist())

        assert exported.shape == stock.shape
        assert torch.allclose(exported, stock, atol=1e-6)

    def test_vision_attn_forward_matches_stock(self) -> None:
        """``export_vision_attn_forward`` matches stock attention (SDPA path)."""
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionAttention

        torch.manual_seed(0)
        config = Qwen3VLVisionConfig(hidden_size=32, num_heads=4)
        config._attn_implementation = "sdpa"
        attn = Qwen3VLVisionAttention(config).eval()
        head_dim = config.hidden_size // config.num_heads

        # Two attention windows of 6 and 4 tokens -> seq_len 10.
        split_sizes = [6, 4]
        seq_len = sum(split_sizes)
        cu_seqlens = torch.tensor([0, 6, 10])
        hidden_states = torch.randn(seq_len, config.hidden_size)
        cos = torch.randn(seq_len, head_dim)
        sin = torch.randn(seq_len, head_dim)
        position_embeddings = (cos, sin)

        with torch.no_grad():
            stock = attn.forward(
                hidden_states,
                cu_seqlens,
                position_embeddings=position_embeddings,
            )
            exported = export_vision_attn_forward(
                attn,
                split_sizes,
                hidden_states,
                position_embeddings,
            )

        assert exported.shape == stock.shape
        assert torch.allclose(exported, stock, atol=1e-5)

    def test_scatter_visual_embeds_matches_masked_scatter(self) -> None:
        """``export_scatter_visual_embeds`` matches stock ``masked_scatter`` merge."""
        torch.manual_seed(0)
        seq_len, hidden, num_visual = 8, 4, 3
        inputs_embeds = torch.randn(1, seq_len, hidden)
        image_token_indices = torch.tensor([2, 4, 5])
        image_embeds = torch.randn(num_visual, hidden)

        # Stock merge: broadcast a boolean mask and ``masked_scatter``.
        image_mask = torch.zeros(1, seq_len, dtype=torch.bool)
        image_mask[0, image_token_indices] = True
        image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        stock = inputs_embeds.masked_scatter(image_mask, image_embeds)

        exported = export_scatter_visual_embeds(inputs_embeds, image_token_indices, image_embeds)

        assert exported.shape == stock.shape
        assert torch.equal(exported, stock)

    def test_add_deepstack_embeds_matches_stock(self) -> None:
        """``export_add_deepstack_embeds`` matches stock ``_deepstack_process``."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

        torch.manual_seed(0)
        seq_len, hidden, num_visual = 8, 4, 3
        hidden_states = torch.randn(1, seq_len, hidden)
        image_token_indices = torch.tensor([1, 3, 6])
        visual_embeds = torch.randn(num_visual, hidden)

        # Stock deepstack uses a boolean mask over the flattened (batch, seq) grid.
        visual_pos_masks = torch.zeros(1, seq_len, dtype=torch.bool)
        visual_pos_masks[0, image_token_indices] = True
        stock = Qwen3VLTextModel._deepstack_process(
            types.SimpleNamespace(),
            hidden_states,
            visual_pos_masks,
            visual_embeds,
        )

        exported = export_add_deepstack_embeds(hidden_states, image_token_indices, visual_embeds)

        assert exported.shape == stock.shape
        assert torch.allclose(exported, stock, atol=1e-6)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize(
        "attention_mask",
        [
            [[1, 1, 1, 1, 1]],  # no padding -> pure causal
            [[1, 1, 1, 0, 0]],  # right padding
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]],  # batched, mixed padding
        ],
    )
    def test_build_additive_causal_mask_matches_stock(
        self,
        attention_mask: list[list[int]],
        dtype: torch.dtype,
    ) -> None:
        """``export_build_additive_causal_mask`` matches stock ``eager_mask``."""
        from transformers.masking_utils import eager_mask

        mask = torch.tensor(attention_mask, dtype=torch.long)
        batch, seq_len = mask.shape

        stock = eager_mask(
            batch_size=batch,
            q_length=seq_len,
            kv_length=seq_len,
            attention_mask=mask.to(torch.bool),
            dtype=dtype,
        )
        exported = export_build_additive_causal_mask(mask, dtype)

        assert exported.shape == (batch, 1, seq_len, seq_len)
        assert exported.shape == stock.shape
        assert exported.dtype == stock.dtype
        assert torch.equal(exported, stock)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("shape", [(2, 16), (2, 5, 16), (1, 3, 4, 16)])
    def test_ov_friendly_rmsnorm_matches_stock(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        """``ov_friendly_rmsnorm_forward`` matches stock ``Qwen3VLTextRMSNorm``."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm

        torch.manual_seed(0)
        hidden = shape[-1]
        norm = Qwen3VLTextRMSNorm(hidden).eval()
        with torch.no_grad():
            # Randomize the weight so the weight-scaling path is exercised.
            norm.weight.copy_(torch.randn(hidden))
        x = torch.randn(*shape, dtype=dtype)

        with torch.no_grad():
            # Clone per call: the export forward reduces over ``dim() - 1`` in
            # place on its float32 copy, which for a float32 input would alias x.
            stock = norm(x.clone())
            exported = ov_friendly_rmsnorm_forward(norm, x.clone())

        assert exported.shape == stock.shape
        assert exported.dtype == stock.dtype
        assert torch.equal(exported, stock)
