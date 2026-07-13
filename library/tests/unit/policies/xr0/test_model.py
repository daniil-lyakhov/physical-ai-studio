# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the XR0 rectified-flow action model (``model``).

Fast, self-contained tests with no external dependencies (no HuggingFace model
downloads). A small ``XR0FlowModel`` exercises the rectified-flow math, the
``dit_forward`` orchestration, and the Euler ``_flow_generate`` loop for
structural invariants and a pinned reference output. Exact numerical parity
against the source implementation is proven separately by the golden fixtures
(``G20``, ``G21``, ``G30``, ``G31``).
"""

from __future__ import annotations

import pytest
import torch

from physicalai.policies.xr0.model import XR0FlowModel

# Small structural config shared across tests.
HIDDEN = 128
HEAD_DIM = 128  # DiT default; hidden must be divisible by it.
KV_HEADS = 1
LAYERS = 2
STATE_LEN = 1
ACTION_LEN = 4
STATE_DIM = 8
ACTION_DIM = 8
BATCH = 2
CACHE = 3
NUM_STEPS = 3

# Tolerance for pinned reference outputs.
TOL = {"atol": 1e-4, "rtol": 1e-4}


def _build_model() -> XR0FlowModel:
    """Build a small fp32 flow model matching the shared config."""
    return XR0FlowModel(
        state_shape=(STATE_LEN, STATE_DIM),
        action_shape=(ACTION_LEN, ACTION_DIM),
        dit_num_layers=LAYERS,
        dit_hidden_size=HIDDEN,
        dit_kv_heads=KV_HEADS,
        num_steps=NUM_STEPS,
        dtype=torch.float32,
    )


def _dit_inputs() -> dict:
    """Build small inputs for ``dit_forward`` / ``_flow_generate``."""
    q_len = 1 + STATE_LEN + ACTION_LEN  # sink + state + action
    ang = torch.randn(BATCH, q_len, HEAD_DIM)
    q_causal = torch.tril(torch.ones(q_len, q_len, dtype=torch.bool))
    cache_ones = torch.ones(q_len, CACHE, dtype=torch.bool)
    mask = torch.cat([cache_ones, q_causal], dim=-1)[None, None].expand(BATCH, 1, q_len, CACHE + q_len)
    return {
        "noisy_action": torch.randn(BATCH, ACTION_LEN, ACTION_DIM),
        "t": torch.ones(BATCH, 1, 1) * 0.3,
        "action_mask": torch.ones(BATCH, ACTION_LEN, ACTION_DIM),
        "state_embed": torch.randn(BATCH, STATE_LEN, HIDDEN),
        "cos": torch.cos(ang),
        "sin": torch.sin(ang),
        "past_key_values": [
            (torch.randn(BATCH, KV_HEADS, CACHE, HEAD_DIM), torch.randn(BATCH, KV_HEADS, CACHE, HEAD_DIM))
            for _ in range(LAYERS)
        ],
        "attn_mask": mask.contiguous(),
    }


def _dit_kwargs(i: dict) -> dict:
    """Assemble the ``dit_forward`` keyword args from an input bundle."""
    return {
        "action_mask": i["action_mask"],
        "state_embed": i["state_embed"],
        "position_embeds": (i["cos"], i["sin"]),
        "past_key_values": i["past_key_values"],
        "attn_mask": i["attn_mask"],
        "prefix_length": 0,
    }


# ============================================================================ #
# Rectified flow math                                                          #
# ============================================================================ #


class TestFlowMath:
    """Tests for the pure rectified-flow helpers."""

    def test_interpolate_endpoints(self) -> None:
        """t=0 returns x0 and t=1 returns x1."""
        model = _build_model()
        x0 = torch.randn(BATCH, ACTION_LEN, ACTION_DIM)
        x1 = torch.randn(BATCH, ACTION_LEN, ACTION_DIM)
        torch.testing.assert_close(model._flow_interpolate(x0, x1, torch.zeros(BATCH, 1, 1)), x0)
        torch.testing.assert_close(model._flow_interpolate(x0, x1, torch.ones(BATCH, 1, 1)), x1)

    @pytest.mark.parametrize(
        "reference",
        [torch.tensor([0.39629608, -0.72900736, -0.29863209, -0.36008668, 0.27356121])],
    )
    def test_interpolate_formula(self, reference: torch.Tensor) -> None:
        """A seeded interpolation pins a slice of z_t = (1 - t)*x0 + t*x1."""
        torch.manual_seed(0)
        model = _build_model()
        x0 = torch.randn(BATCH, ACTION_LEN, ACTION_DIM)
        x1 = torch.randn(BATCH, ACTION_LEN, ACTION_DIM)
        t = torch.rand(BATCH, 1, 1)
        out = model._flow_interpolate(x0, x1, t)[0, 0, :5]
        assert torch.allclose(out, reference, **TOL), out.tolist()

    @pytest.mark.parametrize(
        "reference",
        [torch.tensor([0.48739055, -0.00960857, -0.06220679, -1.28131175, 2.19818282])],
    )
    def test_velocity_target(self, reference: torch.Tensor) -> None:
        """A seeded velocity target pins a slice of v = x1 - x0."""
        torch.manual_seed(0)
        model = _build_model()
        x0 = torch.randn(BATCH, ACTION_LEN, ACTION_DIM)
        x1 = torch.randn(BATCH, ACTION_LEN, ACTION_DIM)
        out = model._flow_velocity_target(x0, x1)[0, 0, :5]
        assert torch.allclose(out, reference, **TOL), out.tolist()

    def test_sample_timestep_beta_range(self) -> None:
        """Beta sampling yields shape (batch,) values in (0, 0.999]."""
        t = _build_model()._sample_timestep(16, dtype=torch.float32)
        assert t.shape == (16,)
        assert torch.all(t > 0) and torch.all(t <= 0.999)

    def test_sample_timestep_uniform_fallback(self) -> None:
        """A non-beta/logit sampler falls back to Uniform(0, 1)."""
        model = _build_model()
        model.flow_sampling = "uniform"
        t = model._sample_timestep(16, dtype=torch.float32)
        assert torch.all(t >= 0) and torch.all(t < 1)


# ============================================================================ #
# dit_forward                                                                  #
# ============================================================================ #


class TestDitForward:
    """Tests for a single DiT velocity prediction."""

    def test_output_shape(self) -> None:
        """dit_forward returns (B, action_len, action_dim)."""
        model = _build_model()
        i = _dit_inputs()
        out = model.dit_forward(i["noisy_action"], i["t"], **_dit_kwargs(i))
        assert out.shape == (BATCH, ACTION_LEN, ACTION_DIM)

    def test_prefix_length_zeroes_leading_actions(self) -> None:
        """A positive prefix_length forces the leading action tokens to zero."""
        model = _build_model()
        i = _dit_inputs()
        kwargs = _dit_kwargs(i)
        kwargs["prefix_length"] = 2
        out = model.dit_forward(i["noisy_action"], i["t"], **kwargs)
        torch.testing.assert_close(out[:, :2], torch.zeros(BATCH, 2, ACTION_DIM))

    @pytest.mark.parametrize(
        "reference",
        [torch.tensor([0.15467618, 0.23847848, 0.09961469, -0.11562672, 0.05315172])],
    )
    def test_reference(self, reference: torch.Tensor) -> None:
        """A seeded dit_forward pins a slice of its velocity output."""
        torch.manual_seed(0)
        model = _build_model()
        i = _dit_inputs()
        out = model.dit_forward(i["noisy_action"], i["t"], **_dit_kwargs(i))[0, 0, :5]
        assert torch.allclose(out, reference, **TOL), out.tolist()


# ============================================================================ #
# _flow_generate                                                               #
# ============================================================================ #


class TestFlowGenerate:
    """Tests for the Euler integration inference loop."""

    @pytest.mark.parametrize(
        "reference",
        [torch.tensor([-1.96027327, -0.91721338, 0.38879472, -0.24504544, 0.8044914])],
    )
    def test_reference(self, reference: torch.Tensor) -> None:
        """A seeded _flow_generate pins a slice of its Euler-integrated output."""
        torch.manual_seed(0)
        model = _build_model()
        i = _dit_inputs()
        x0 = torch.randn(BATCH, ACTION_LEN, ACTION_DIM)
        out = model._flow_generate(x0, _dit_kwargs(i))[0, 0, :5]
        assert torch.allclose(out, reference, **TOL), out.tolist()
