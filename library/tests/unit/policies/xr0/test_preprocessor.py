# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the XR0 preprocessor / postprocessor.

These exercise the framework-observation -> XR0-batch mapping, mirroring the
source repository's Qwen3-VL multi-view prompt + ``io`` state/action helpers.
The prompt/vision tests require the Qwen3-VL processor to be available locally
and are skipped otherwise; the normalization round-trip test does not.
"""

from __future__ import annotations

import pytest
import torch

from physicalai.data.observation import ACTION, STATE, TASK
from physicalai.policies.xr0.preprocessor import (
    XR0Postprocessor,
    XR0Preprocessor,
    make_xr0_preprocessors,
)

STATE_DIM = 8
ACTION_DIM = 7
HORIZON = 30


def _stats() -> dict:
    return {
        "observation.state": {"name": "observation.state", "shape": (STATE_DIM,), "mean": [0.0] * STATE_DIM, "std": [1.0] * STATE_DIM},
        "action": {"name": "action", "shape": (ACTION_DIM,), "mean": [0.1] * ACTION_DIM, "std": [2.0] * ACTION_DIM},
    }


def _batch(batch_size: int = 2) -> dict:
    return {
        "images.base": torch.rand(batch_size, 3, 64, 64),
        "images.wrist_left": torch.rand(batch_size, 3, 64, 64),
        STATE: torch.rand(batch_size, STATE_DIM),
        TASK: ["pick up the cube", "open the drawer"][:batch_size],
        ACTION: torch.randn(batch_size, HORIZON, ACTION_DIM),
    }


def _processor_available() -> bool:
    try:
        XR0Preprocessor(camera_views=("base",)).processor  # noqa: B018
    except Exception:  # noqa: BLE001
        return False
    return True


requires_processor = pytest.mark.skipif(
    not _processor_available(),
    reason="Qwen3-VL processor not available locally",
)


class TestNormalization:
    """Action normalization round-trips without the processor."""

    def test_action_stats_padded(self) -> None:
        pre, post = make_xr0_preprocessors(stats=_stats(), max_action_dim=32)
        assert pre.action_mean.shape == (32,)
        # real dims take the stats, padding dims stay mean 0 / std 1
        assert torch.allclose(pre.action_mean[:ACTION_DIM], torch.full((ACTION_DIM,), 0.1))
        assert torch.allclose(pre.action_mean[ACTION_DIM:], torch.zeros(32 - ACTION_DIM))
        assert torch.allclose(pre.action_std[:ACTION_DIM], torch.full((ACTION_DIM,), 2.0))
        assert post.action_dim == ACTION_DIM

    def test_prepare_action_roundtrip(self) -> None:
        pre, post = make_xr0_preprocessors(stats=_stats(), max_action_dim=32)
        action = torch.randn(2, HORIZON, ACTION_DIM)
        normalized, mask = pre._prepare_action(action, torch.device("cpu"))  # noqa: SLF001
        assert normalized.shape == (2, HORIZON, 32)
        assert mask[..., :ACTION_DIM].all()
        assert not mask[..., ACTION_DIM:].any()
        recovered = post({ACTION: normalized})[ACTION]
        assert recovered.shape == (2, HORIZON, ACTION_DIM)
        assert torch.allclose(recovered, action, atol=1e-5)

    def test_identity_without_stats(self) -> None:
        pre = XR0Preprocessor(max_action_dim=32, features=None)
        assert torch.allclose(pre.action_mean, torch.zeros(32))
        assert torch.allclose(pre.action_std, torch.ones(32))


@requires_processor
class TestVisionPrompt:
    """Prompt + vision pipeline through the real Qwen3-VL processor."""

    def test_forward_keys_and_shapes(self) -> None:
        pre, _ = make_xr0_preprocessors(camera_views=("base", "wrist_left"), stats=_stats())
        out = pre(_batch(2))
        assert set(out) == {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "state", ACTION, "action_mask"}
        assert out["input_ids"].shape[0] == 2
        assert out["state"].shape == (2, 1, 32)
        assert out[ACTION].shape == (2, HORIZON, 32)
        # one image_grid_thw row per (sample, view)
        assert out["image_grid_thw"].shape[0] == 2 * 2

    def test_inference_batch_without_action(self) -> None:
        pre, _ = make_xr0_preprocessors(camera_views=("base", "wrist_left"), stats=_stats())
        batch = _batch(1)
        batch.pop(ACTION)
        out = pre(batch)
        assert ACTION not in out
        assert "action_mask" not in out
        assert out["state"].shape == (1, 1, 32)
