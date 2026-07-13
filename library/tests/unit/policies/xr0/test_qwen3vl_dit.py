# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the XR0 DiT action head (``qwen3vl_dit``).

Fast, self-contained tests with no external dependencies (no HuggingFace model
downloads). Small module configs exercise the tensor helpers and DiT decoder
stack for structural invariants and pinned reference outputs. Weights and inputs
are drawn from a fixed seed, so a chosen slice of each forward output is
reproducible and guards against silent behavioural regressions. Exact numerical
parity against the source implementation is proven separately by the golden
fixtures (``G10``-``G15``).
"""

from __future__ import annotations

import pytest
import torch

from physicalai.policies.xr0.qwen3vl_dit import (
    DiT,
    DecoderLayer,
    DiTAttention,
    DiTMLP,
    MLPProjector,
    TimestepEmbedder,
    apply_rotary_pos_emb,
    modulate,
    repeat_kv,
)

# Small structural config shared across module tests.
HIDDEN = 64
HEAD_DIM = 16
NUM_HEADS = HIDDEN // HEAD_DIM  # 4
KV_HEADS = 2
BATCH = 2
SEQ = 5
CACHE = 3

# Tolerance for pinned reference outputs.
TOL = {"atol": 1e-4, "rtol": 1e-4}


def _attn_inputs(seq: int = SEQ, cache: int = CACHE):
    """Build small inputs for a DiTAttention / DecoderLayer forward."""
    hidden = torch.randn(BATCH, seq, HIDDEN)
    past_key = torch.randn(BATCH, KV_HEADS, cache, HEAD_DIM)
    past_value = torch.randn(BATCH, KV_HEADS, cache, HEAD_DIM)
    cos = torch.randn(BATCH, seq, HEAD_DIM)
    sin = torch.randn(BATCH, seq, HEAD_DIM)
    # Boolean mask over [cache | causal-query] keys.
    mask = torch.ones(BATCH, 1, seq, cache + seq, dtype=torch.bool)
    return hidden, (past_key, past_value), (cos, sin), mask


# ============================================================================ #
# Helper Functions                                                             #
# ============================================================================ #


class TestHelpers:
    """Tests for the DiT tensor helper functions."""

    @pytest.mark.parametrize(
        "reference",
        [torch.tensor([-0.21684796, 0.37115434, 1.17266655, 0.0787273, 1.57226944])],
    )
    def test_modulate_reference(self, reference: torch.Tensor) -> None:
        """A seeded modulate pins a slice of x * (1 + scale) + shift."""
        torch.manual_seed(0)
        x = torch.randn(2, 3, HIDDEN)
        shift = torch.randn(2, 1, HIDDEN)
        scale = torch.randn(2, 1, HIDDEN)
        out = modulate(x, shift, scale)[0, 0, :5]
        assert torch.allclose(out, reference, **TOL), out.tolist()

    def test_repeat_kv_identity(self) -> None:
        """n_rep=1 is a no-op."""
        kv = torch.randn(BATCH, KV_HEADS, CACHE, HEAD_DIM)
        torch.testing.assert_close(repeat_kv(kv, 1), kv)

    def test_repeat_kv_interleaves(self) -> None:
        """Repeated heads are interleaved (head i -> positions i*n_rep .. )."""
        kv = torch.randn(1, 2, CACHE, HEAD_DIM)
        out = repeat_kv(kv, 2)
        torch.testing.assert_close(out[:, 0], kv[:, 0])
        torch.testing.assert_close(out[:, 1], kv[:, 0])
        torch.testing.assert_close(out[:, 2], kv[:, 1])
        torch.testing.assert_close(out[:, 3], kv[:, 1])

    @pytest.mark.parametrize(
        "reference",
        [torch.tensor([0.64037418, 0.58257341, -0.39190835, -0.41302168, -0.85603261])],
    )
    def test_apply_rotary_reference(self, reference: torch.Tensor) -> None:
        """A seeded rotary embedding pins a slice of the rotated query."""
        torch.manual_seed(0)
        q = torch.randn(BATCH, NUM_HEADS, SEQ, HEAD_DIM)
        k = torch.randn(BATCH, NUM_HEADS, SEQ, HEAD_DIM)
        cos = torch.randn(BATCH, SEQ, HEAD_DIM)
        sin = torch.randn(BATCH, SEQ, HEAD_DIM)
        q_out, _ = apply_rotary_pos_emb(q, k, cos, sin)
        out = q_out[0, 0, 0, :5]
        assert torch.allclose(out, reference, **TOL), out.tolist()


# ============================================================================ #
# Projectors & Embedders                                                       #
# ============================================================================ #


class TestMLPProjector:
    """Tests for MLPProjector."""

    def test_single_layer_is_linear(self) -> None:
        """num_layers=1, bias=False is a bare linear projection."""
        proj = MLPProjector(8, 4, num_layers=1, bias=False)
        assert len(proj.layers) == 1
        x = torch.randn(3, 8)
        torch.testing.assert_close(proj(x), x @ proj.layers[0].weight.t())

    def test_bias_flag(self) -> None:
        """bias=True adds bias to the linear layers."""
        assert MLPProjector(8, 4, num_layers=1, bias=True).layers[0].bias is not None
        assert MLPProjector(8, 4, num_layers=1, bias=False).layers[0].bias is None

    @pytest.mark.parametrize(
        "reference",
        [torch.tensor([0.06443175, -0.08235656, -0.01462553, -0.19436540, -0.07179737])],
    )
    def test_reference(self, reference: torch.Tensor) -> None:
        """A seeded 2-layer projector pins a slice of its output."""
        torch.manual_seed(0)
        proj = MLPProjector(32, HIDDEN, num_layers=2)
        out = proj(torch.randn(BATCH, SEQ, 32))[0, 0, :5]
        assert torch.allclose(out, reference, **TOL), out.tolist()


class TestTimestepEmbedder:
    """Tests for TimestepEmbedder."""

    def test_dtype_respected(self) -> None:
        """The frequency embedding is cast to the configured dtype."""
        emb = TimestepEmbedder(HIDDEN, dtype=torch.float32)
        assert emb(torch.tensor([1.0])).dtype == torch.float32

    @pytest.mark.parametrize(
        "reference",
        [torch.tensor([0.34102309, 0.21420282, 0.16651681, 0.06136062])],
    )
    def test_reference(self, reference: torch.Tensor) -> None:
        """A seeded embedder pins the first channel across timesteps."""
        torch.manual_seed(0)
        emb = TimestepEmbedder(HIDDEN, dtype=torch.float32)
        out = emb(torch.tensor([0.0, 200.0, 500.0, 999.0]))[:, 0, 0]
        assert torch.allclose(out, reference, **TOL), out.tolist()


# ============================================================================ #
# DiT Components                                                               #
# ============================================================================ #


class TestDiTMLP:
    """Tests for DiTMLP (SwiGLU)."""

    def test_intermediate_size(self) -> None:
        """Intermediate size is 4x the hidden size."""
        assert DiTMLP(HIDDEN).intermediate_size == 4 * HIDDEN

    @pytest.mark.parametrize(
        "reference",
        [torch.tensor([0.14671412, 0.22900799, 0.19239995, 0.04836469, 0.22137016])],
    )
    def test_reference(self, reference: torch.Tensor) -> None:
        """A seeded SwiGLU MLP pins a slice of its output."""
        torch.manual_seed(0)
        mlp = DiTMLP(HIDDEN)
        out = mlp(torch.randn(BATCH, SEQ, HIDDEN))[0, 0, :5]
        assert torch.allclose(out, reference, **TOL), out.tolist()


class TestDiTAttention:
    """Tests for DiTAttention."""

    def test_head_configuration(self) -> None:
        """num_heads and kv_group derive from hidden/head_dim/kv_heads."""
        attn = DiTAttention(hidden_size=HIDDEN, head_dim=HEAD_DIM, kv_heads=KV_HEADS)
        assert attn.num_heads == NUM_HEADS
        assert attn.kv_group == NUM_HEADS // KV_HEADS

    @pytest.mark.parametrize(
        "reference",
        [torch.tensor([0.13715474, -0.13632900, 0.11728425, 0.05819567, -0.16545057])],
    )
    def test_reference(self, reference: torch.Tensor) -> None:
        """A seeded attention forward pins a slice of its output."""
        torch.manual_seed(0)
        attn = DiTAttention(hidden_size=HIDDEN, head_dim=HEAD_DIM, kv_heads=KV_HEADS)
        hidden, kv, pos, mask = _attn_inputs()
        out = attn(hidden, kv, pos, attn_mask=mask)[0, 0, :5]
        assert torch.allclose(out, reference, **TOL), out.tolist()


class TestDecoderLayer:
    """Tests for DecoderLayer."""

    def test_adaln_table_shape(self) -> None:
        """AdaLN table produces 6 modulation vectors of hidden size."""
        layer = DecoderLayer(hidden_size=HIDDEN, head_dim=HEAD_DIM, kv_heads=KV_HEADS)
        assert layer.adaln_table.shape == (6, HIDDEN)

    @pytest.mark.parametrize(
        "reference",
        [torch.tensor([0.26495734, -0.48088220, 1.40348279, -0.17601027, -1.48331332])],
    )
    def test_reference(self, reference: torch.Tensor) -> None:
        """A seeded decoder-layer forward pins a slice of its output."""
        torch.manual_seed(0)
        layer = DecoderLayer(hidden_size=HIDDEN, head_dim=HEAD_DIM, kv_heads=KV_HEADS)
        hidden, kv, pos, mask = _attn_inputs()
        t_embeds = torch.randn(BATCH, 6, HIDDEN)
        out = layer(hidden, kv, pos, t_embeds, attn_mask=mask)[0, 0, :5]
        assert torch.allclose(out, reference, **TOL), out.tolist()


class TestDiT:
    """Tests for the DiT decoder stack."""

    def test_tail_alignment_with_longer_cache(self) -> None:
        """A KV-cache longer than layer_num is consumed from its tail."""
        dit = DiT(hidden_size=HIDDEN, layer_num=2, head_dim=HEAD_DIM, kv_heads=KV_HEADS)
        assert len(dit.layers) == 2
        hidden, kv, pos, mask = _attn_inputs()
        # 4-entry cache with only 2 DiT layers -> start_idx = 2 (uses last two).
        past_key_values = [(kv[0].clone(), kv[1].clone()) for _ in range(4)]
        t_embeds = torch.randn(BATCH, 6, HIDDEN)
        out = dit(hidden, past_key_values, mask, pos, t_embeds)
        assert out.shape == (BATCH, SEQ, HIDDEN)

    @pytest.mark.parametrize(
        "reference",
        [torch.tensor([1.01393712, 0.45370287, -0.99828362, -0.52321780, -0.47535068])],
    )
    def test_reference(self, reference: torch.Tensor) -> None:
        """A seeded DiT forward pins a slice of its output."""
        torch.manual_seed(0)
        dit = DiT(hidden_size=HIDDEN, layer_num=2, head_dim=HEAD_DIM, kv_heads=KV_HEADS)
        hidden, kv, pos, mask = _attn_inputs()
        past_key_values = [kv, (kv[0].clone(), kv[1].clone())]
        t_embeds = torch.randn(BATCH, 6, HIDDEN)
        out = dit(hidden, past_key_values, mask, pos, t_embeds)[0, 0, :5]
        assert torch.allclose(out, reference, **TOL), out.tolist()
