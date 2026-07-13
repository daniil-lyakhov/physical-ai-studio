# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the assembled XR0 VLA model (``vla``).

Fast, self-contained tests with no external dependencies (no HuggingFace model
downloads). A tiny synthetic Qwen3-VL shim is injected so the full
VLM -> MRoPE-continuation -> DiT -> rectified-flow pipeline runs end to end on
CPU in fp32. The tests exercise the framework ``Model`` contract (training loss,
inference chunk, delta-index properties) and pin the inference output against
reference data. The VLM numerics and the flow/DiT math are proven separately by
the ``vlm`` tests and the golden fixtures.
"""

from __future__ import annotations

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

from physicalai.policies.xr0.vla import XR0Model
from physicalai.policies.xr0.vlm import XR0Qwen3VL

IMAGE_TOKEN_ID = 151
VIDEO_TOKEN_ID = 152
VISION_START_TOKEN_ID = 150
IMAGE_GRID = (2, 4, 4)
SPATIAL_MERGE = 2
N_IMAGE_TOKENS = (IMAGE_GRID[0] * IMAGE_GRID[1] * IMAGE_GRID[2]) // SPATIAL_MERGE**2

# Tiny model dims. The DiT head_dim / kv_heads / layer count must match the VLM
# so the DiT can consume the VLM KV-cache.
STATE_LEN = 1
STATE_DIM = 8
ACTION_LEN = 4
ACTION_DIM = 8
DIT_HIDDEN = 64
DIT_HEAD_DIM = 16
DIT_KV_HEADS = 2
DIT_LAYERS = 2
NUM_STEPS = 3

# Reference first action token of the denoised chunk for the seeded tiny model.
REFERENCE_ACTION = torch.tensor(
    [0.41849822, 2.12935758, -1.08118093, 0.15135361, 0.33237639, -0.34592688, 1.10643888, 0.60246569]
)


def _config() -> Qwen3VLConfig:
    """Tiny Qwen3-VL config whose head_dim / kv_heads match the DiT."""
    vision = Qwen3VLVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        depth=2,
        out_hidden_size=64,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=SPATIAL_MERGE,
        in_channels=3,
        deepstack_visual_indexes=[0],
    )
    text = Qwen3VLTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=DIT_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=DIT_KV_HEADS,
        head_dim=DIT_HEAD_DIM,
        vocab_size=200,
        rope_scaling={"type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": False},
    )
    return Qwen3VLConfig(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
    )


def _build_model() -> XR0Model:
    """Build the assembled model on a tiny injected VLM (no download)."""
    torch.manual_seed(0)
    vlm = XR0Qwen3VL(_config())
    return XR0Model(
        vlm=vlm,
        state_shape=(STATE_LEN, STATE_DIM),
        action_shape=(ACTION_LEN, ACTION_DIM),
        dit_num_layers=DIT_LAYERS,
        dit_hidden_size=DIT_HIDDEN,
        dit_head_dim=DIT_HEAD_DIM,
        dit_kv_heads=DIT_KV_HEADS,
        num_steps=NUM_STEPS,
        training_repeat=1,
        dtype=torch.float32,
    )


def _batch() -> dict:
    """Build a deterministic multimodal batch with action / state targets."""
    grid = torch.tensor([list(IMAGE_GRID)])
    num_patches = int(grid.prod(-1).item())
    patch_dim = 3 * 2 * 16 * 16
    torch.manual_seed(0)
    pixel_values = torch.randn(num_patches, patch_dim)
    input_ids = torch.tensor([[5, 6, VISION_START_TOKEN_ID, *([IMAGE_TOKEN_ID] * N_IMAGE_TOKENS), 7, 8, 9]])
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": grid,
        "action": torch.randn(1, ACTION_LEN, ACTION_DIM),
        "action_mask": torch.ones(1, ACTION_LEN, ACTION_DIM, dtype=torch.int32),
        "state": torch.randn(1, STATE_LEN, STATE_DIM),
    }


class TestDeltaIndices:
    """Framework Model delta-index properties."""

    def test_indices(self) -> None:
        model = _build_model()
        assert model.reward_delta_indices is None
        assert model.observation_delta_indices is None
        assert model.action_delta_indices == list(range(ACTION_LEN))


class TestCompanionLoss:
    """Training path returns a differentiable flow-matching loss."""

    def test_compute_loss_shapes(self) -> None:
        model = _build_model().train()
        torch.manual_seed(0)
        loss, loss_dict = model.compute_loss(_batch())
        assert loss.ndim == 0
        assert loss.requires_grad
        assert set(loss_dict) == {"loss", "loss_mse", "loss_freq"}
        assert all(isinstance(v, float) for v in loss_dict.values())

    def test_loss_backward_populates_grads(self) -> None:
        model = _build_model().train()
        torch.manual_seed(0)
        loss, _ = model.compute_loss(_batch())
        loss.backward()
        grads = [p.grad for p in model.flow.dit.parameters() if p.requires_grad]
        assert any(g is not None and torch.any(g != 0) for g in grads)


class TestInference:
    """Eval path denoises an action chunk of the right shape and value."""

    def test_predict_action_chunk_shape(self) -> None:
        model = _build_model().eval()
        torch.manual_seed(0)
        with torch.no_grad():
            out = model.predict_action_chunk(_batch())
        assert out.shape == (1, ACTION_LEN, ACTION_DIM)

    def test_forward_eval_dispatches_to_predict(self) -> None:
        model = _build_model().eval()
        torch.manual_seed(0)
        with torch.no_grad():
            out = model(_batch())
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, ACTION_LEN, ACTION_DIM)

    def test_predict_reference(self) -> None:
        model = _build_model().eval()
        torch.manual_seed(0)
        with torch.no_grad():
            out = model.predict_action_chunk(_batch())[0, 0]
        assert torch.allclose(out, REFERENCE_ACTION, atol=1e-4, rtol=1e-4), out.tolist()
