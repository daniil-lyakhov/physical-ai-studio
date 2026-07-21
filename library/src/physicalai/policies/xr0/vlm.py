# Copyright (C) 2026 Xiaomi Corporation.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL backbone shim for the XR0 Vision-Language-Action policy.

The XR0 source ships a machine-generated verbatim copy of the stock
``transformers`` Qwen3-VL model (``xr0/mibot/models/VLM/qwen3vl.py``). The only
*functional* difference from the upstream model is that the copy surfaces the 3D
MRoPE ``position_ids`` (and the ``attention_mask``) on its output dataclasses --
``XR0.forward`` consumes ``vlm_outputs.position_ids.max(dim=-1)`` to continue the
MRoPE sequence into the DiT action head, plus ``vlm_outputs.past_key_values``.

Rather than vendor ~1500 lines of upstream model code (which is version-locked to
the transformers release it was generated from), this module subclasses the
installed stock :class:`~transformers.Qwen3VLForConditionalGeneration` and adds
back only that one behaviour: it computes the 3D position ids with the model's
own :meth:`compute_3d_position_ids` and attaches them to the returned output. All
VLM numerics are inherited unchanged from stock ``transformers``.
"""

from __future__ import annotations

import torch
from transformers import Qwen3VLForConditionalGeneration


class XR0Qwen3VL(Qwen3VLForConditionalGeneration):
    """Stock Qwen3-VL that also exposes the 3D MRoPE ``position_ids``.

    Stock ``transformers`` computes the 3D position ids internally but discards
    them (only ``rope_deltas`` is returned). XR0's action head needs the full
    ``(3, batch, seq)`` grid, so this shim computes it up front via
    :meth:`~transformers.Qwen3VLModel.compute_3d_position_ids`, passes it into the
    stock forward (so the backbone uses exactly the exposed ids), and attaches it
    to the output as ``outputs.position_ids``.

    When ``mm_token_type_ids`` is not supplied (the Qwen3-VL processor normally
    provides it) it is derived from ``input_ids`` using the configured image and
    video token ids so the MRoPE index can still be built.
    """

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,  # noqa: ANN001
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ):
        """Run the stock forward and attach the 3D MRoPE ``position_ids``."""
        if (
            mm_token_type_ids is None
            and input_ids is not None
            and (image_grid_thw is not None or video_grid_thw is not None)
        ):
            mm_token_type_ids = torch.zeros_like(input_ids)
            mm_token_type_ids[input_ids == self.config.image_token_id] = 1
            mm_token_type_ids[input_ids == self.config.video_token_id] = 2

        if position_ids is None:
            position_ids = self.model.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        outputs.position_ids = position_ids
        return outputs
