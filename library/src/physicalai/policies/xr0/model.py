# Copyright (C) 2026 Xiaomi Corporation.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Rectified-flow action model for the XR0 Vision-Language-Action policy.

This module owns the DiT action expert (the DiT decoder
stack plus its projectors, timestep embedder, and sink token) and the
rectified-flow orchestration that drives it:

* ``_sample_timestep``    -- training-time timestep sampling (beta / logit-normal).
* ``_flow_interpolate``   -- ``z_t = (1 - t) * x0 + t * x1``.
* ``_flow_velocity_target`` -- ``v = x1 - x0``.
* ``dit_forward``         -- a single DiT velocity prediction given VLM features.
* ``_flow_generate``      -- ``num_steps`` Euler integration for inference.

It is deliberately **VLM-independent**: the Qwen3-VL backbone outputs
(``state_embed``, ``past_key_values``, ``position_embeds``, ``attn_mask``) are
passed in as arguments rather than computed here, so this unit can be proven in
isolation.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.distributions import Beta, LogisticNormal

from .qwen3vl_dit import DiT, MLPProjector, TimestepEmbedder


class XR0FlowModel(nn.Module):
    """DiT action expert with rectified-flow generation (VLM features passed in)."""

    def __init__(
        self,
        state_shape: tuple[int, int] = (1, 32),
        action_shape: tuple[int, int] = (30, 32),
        dit_num_layers: int = 16,
        dit_hidden_size: int = 1024,
        dit_head_dim: int = 128,
        dit_kv_heads: int = 8,
        num_steps: int = 5,
        flow_sampling: str = "beta",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.dit_num_layers = dit_num_layers
        self.dit_hidden_size = dit_hidden_size
        self.num_steps = num_steps

        # Rectified flow timestep sampling distributions.
        self.flow_sampling = flow_sampling
        self.logistic_normal = LogisticNormal(0.0, 1.0)
        self.beta = Beta(1.5, 1.0)

        # DiT policy head.
        self.dit = DiT(
            hidden_size=dit_hidden_size,
            kv_heads=dit_kv_heads,
            layer_num=dit_num_layers,
            head_dim=dit_head_dim,
        )

        # State / action projectors.
        self.state_projector = MLPProjector(
            input_dim=state_shape[-1],
            output_dim=dit_hidden_size,
            num_layers=2,
        )
        self.action_projector = MLPProjector(
            input_dim=action_shape[-1],
            output_dim=dit_hidden_size,
            num_layers=2,
        )
        self.action_output_layer = MLPProjector(
            input_dim=dit_hidden_size,
            output_dim=action_shape[-1],
            num_layers=2,
        )

        # Timestep embedding for the diffusion t.
        self.t_embedder = TimestepEmbedder(dit_hidden_size, dtype=dtype)
        self.t_projector = MLPProjector(input_dim=dit_hidden_size, output_dim=6 * dit_hidden_size, bias=True)

        # Sink token prepended to the DiT input.
        self.sink = nn.Embedding(1, dit_hidden_size)

        self.to(dtype)

    # --------------------------------------------------------
    # Rectified flow methods
    # --------------------------------------------------------

    @torch.no_grad()
    def _sample_timestep(
        self,
        batch_size: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Sample random timesteps for rectified flow training.

        The distribution is controlled by ``self.flow_sampling``:
        - ``"logit_normal"``: LogisticNormal(0, 1)
        - ``"beta"``: Beta(1.5, 1.0) rescaled to (0, 0.999)
        - otherwise: Uniform(0, 1)

        Args:
            batch_size: Number of timesteps to sample.
            dtype: Output tensor dtype.
            device: Output tensor device.

        Returns:
            Timestep tensor of shape ``(batch_size,)``.
        """
        if self.flow_sampling == "logit_normal":
            u = self.logistic_normal.sample((batch_size,))[:, 0].to(device)
        elif self.flow_sampling == "beta":
            u = self.beta.sample((batch_size,)).to(device)
            u = (1 - u) * 0.999
        else:
            u = torch.rand(size=(batch_size,), device=device)
        return u.to(dtype)

    @torch.no_grad()
    def _flow_interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between noise and data: ``z_t = (1-t)*x0 + t*x1``.

        Args:
            x0: Noise sample (source distribution).
            x1: Data sample (target distribution).
            t: Interpolation coefficient in [0, 1].

        Returns:
            Interpolated tensor with the same shape as x0/x1.
        """
        return (1 - t) * x0 + t * x1

    @torch.no_grad()
    def _flow_velocity_target(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Velocity target for rectified flow: ``v = x1 - x0``.

        Args:
            x0: Noise sample.
            x1: Data sample.

        Returns:
            Velocity tensor of the same shape.
        """
        return x1 - x0

    @torch.no_grad()
    def _flow_generate(self, x0: torch.Tensor, dit_kwargs: dict[str, Any]) -> torch.Tensor:
        """Euler integration: generate action from noise over ``num_steps`` steps.

        Args:
            x0: Initial noise tensor of shape ``(B, action_len, action_dim)``.
            dit_kwargs: Keyword arguments forwarded to ``dit_forward``.

        Returns:
            Denoised action prediction.
        """
        dt = 1.0 / self.num_steps
        z = x0.clone()
        for step in range(self.num_steps):
            t = torch.ones((z.shape[0], 1, 1), device=z.device, dtype=z.dtype) * step / self.num_steps
            v = self.dit_forward(z, t, **dit_kwargs)
            z = z + v * dt
        return z

    # --------------------------------------------------------
    # DiT forward
    # --------------------------------------------------------

    def dit_forward(
        self,
        noisy_action: torch.Tensor,
        t: torch.Tensor,
        action_mask: torch.Tensor,
        state_embed: torch.Tensor,
        position_embeds: tuple[torch.Tensor, torch.Tensor],
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        attn_mask: torch.Tensor,
        prefix_length: int = 0,
    ) -> torch.Tensor:
        """Single forward pass of DiT.

        1. Embed timestep *t* -> 6 AdaLN modulation parameters per layer.
        2. Project noisy action to hidden dim.
        3. Prepend [sink, state] tokens.
        4. Run DiT decoder layers (cross-attending to VLM KV-cache).
        5. Extract action tokens and project back to action dim.

        Args:
            noisy_action: Noisy action tensor ``(B, action_len, action_dim)``.
            t: Timestep ``(B, 1, 1)``.
            action_mask: Binary mask ``(B, action_len, action_dim)``.
            state_embed: Projected state ``(B, state_len, D)``.
            position_embeds: (cos, sin) rotary embeddings for DiT tokens.
            past_key_values: Per-layer VLM KV-cache.
            attn_mask: Boolean attention mask.
            prefix_length: Number of leading action tokens forced to zero.

        Returns:
            Predicted velocity (training) or action (inference) of shape
            ``(B, action_len, action_dim)``.
        """
        # Timestep conditioning: embed t -> 6 modulation parameters per layer.
        t_embeds = self.t_embedder(t[:, 0, 0] * 1000)
        t_embeds = self.t_projector(t_embeds).view(t_embeds.shape[0], 6, -1)

        # Project noisy action to DiT hidden dim.
        noisy_action = noisy_action * action_mask
        noisy_action = self.action_projector(noisy_action)

        # Concatenate: [sink, state, noisy_action].
        sink = self.sink.weight[None].repeat(state_embed.shape[0], 1, 1)
        hidden_states = torch.cat([sink, state_embed, noisy_action], dim=1).contiguous()

        # DiT forward with VLM KV-cache (joint self-attention).
        hidden_states = self.dit(hidden_states, past_key_values, attn_mask, position_embeds, t_embeds)

        # Extract action tokens and project back to action dim.
        hidden_states = hidden_states[:, -noisy_action.shape[1] :, :]
        output = self.action_output_layer(hidden_states)
        if prefix_length > 0:
            output[:, :prefix_length] = 0.0

        return output
