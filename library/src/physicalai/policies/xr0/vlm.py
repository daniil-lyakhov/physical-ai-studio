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

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import torch
from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast

if TYPE_CHECKING:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast


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

    def build_3d_position_ids(
        self,
        input_ids: torch.LongTensor | None,
        attention_mask: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
    ) -> torch.Tensor:
        """Compute the 3D MRoPE ``position_ids`` for the given inputs.

        This wraps the stock :meth:`~transformers.Qwen3VLModel.compute_3d_position_ids`
        (deriving ``mm_token_type_ids`` from ``input_ids`` when absent). The
        underlying ``get_rope_index`` uses data-dependent Python control flow
        (``tensor.tolist()`` / :func:`itertools.groupby`), so this **must** run
        eagerly on concrete tensors -- it cannot be captured by ``torch.export``.
        Callers that trace the model (e.g. ONNX/OpenVINO export) should compute
        the ids up front (in the preprocessor) and pass them into
        :meth:`forward` so the data-dependent path is skipped.

        Returns:
            The 3D MRoPE ``position_ids`` tensor (shape ``(3, batch, seq)``).
        """
        if (
            mm_token_type_ids is None
            and input_ids is not None
            and (image_grid_thw is not None or video_grid_thw is not None)
        ):
            derived_ids = torch.zeros_like(input_ids)
            derived_ids[input_ids == self.config.image_token_id] = 1
            derived_ids[input_ids == self.config.video_token_id] = 2
            mm_token_type_ids = cast("torch.IntTensor", derived_ids)

        return self.model.compute_3d_position_ids(
            input_ids=input_ids,
            inputs_embeds=None,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=mm_token_type_ids,
        )

    @torch.no_grad()
    def encode_vision(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the vision tower eagerly and return its precomputed embeddings.

        The Qwen3-VL vision tower derives token geometry from ``image_grid_thw``
        contents (``tensor.tolist()`` -> ``linspace`` / ``split`` sizes), which
        ``torch.export`` cannot capture. Running it eagerly here (in the
        preprocessor) and passing the result into :meth:`forward` as
        ``image_embeds`` / ``deepstack_embeds`` lets the exported model graph skip
        the vision tower entirely while staying numerically identical.

        Returns:
            A ``(image_embeds, deepstack_embeds)`` tuple where ``image_embeds`` is
            ``(num_visual_tokens, hidden)`` and ``deepstack_embeds`` is
            ``(num_deepstack_layers, num_visual_tokens, hidden)``.
        """
        # Call the vision tower directly (not via ``get_image_features``, which
        # ``forward`` patches) so this stays correct across repeated inference.
        vision_output = self.model.visual(
            pixel_values.type(self.model.visual.dtype),
            grid_thw=image_grid_thw,
            return_dict=True,
        )
        image_embeds = vision_output.pooler_output
        deepstack_embeds = torch.stack(tuple(vision_output.deepstack_features), dim=0)
        return image_embeds, deepstack_embeds

    def image_token_positions(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Return the integer sequence positions of the image tokens.

        The stock Qwen3-VL merge / deepstack code scatters the visual embeddings
        into the language-model hidden states with *boolean-mask* assignment
        (``masked_scatter`` / ``hidden[mask] = ...``). Under ``torch.export`` those
        lower to an OpenVINO-hostile ``Where`` whose operand shapes disagree
        (``[1, seq, hidden]`` vs ``[num_visual, hidden]``). The export path instead
        scatters by *integer index* (:func:`torch.Tensor.index_copy` ->
        ``ScatterND``, which OpenVINO supports); those indices are data-dependent
        (:func:`torch.nonzero`), so they are computed here eagerly (in the
        preprocessor) and passed into :meth:`forward`.

        Returns:
            A ``(num_visual_tokens,)`` long tensor of image-token positions in the
            (single-batch) sequence.
        """
        return (input_ids[0] == self.config.image_token_id).nonzero(as_tuple=True)[0]

    def _ensure_export_patch(self) -> None:
        """Install export-friendly (integer-index) visual-scatter overrides.

        Replaces the two boolean-mask scatters that OpenVINO cannot convert -- the
        image/text merge in ``Qwen3VLModel.forward`` and the deepstack injection in
        ``Qwen3VLTextModel._deepstack_process`` -- with numerically-identical
        ``index_copy`` / ``index_select`` variants that lower to ``ScatterND`` /
        ``Gather``. Both overrides use the eagerly-precomputed image-token
        positions stashed on this shim and fall back to the stock implementation
        when those positions are absent (normal, non-export inference).
        """
        if getattr(self, "_export_patched", False):
            return
        shim = self
        inner = self.model
        text_model = inner.language_model
        orig_model_forward = inner.forward
        orig_deepstack_process = text_model._deepstack_process  # noqa: SLF001

        def _patched_model_forward(
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: object | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            pixel_values: torch.Tensor | None = None,
            pixel_values_videos: torch.FloatTensor | None = None,
            image_grid_thw: torch.LongTensor | None = None,
            video_grid_thw: torch.LongTensor | None = None,
            mm_token_type_ids: torch.IntTensor | None = None,
            **kwargs: object,
        ) -> Qwen3VLModelOutputWithPast:
            idx = shim._image_token_indices  # noqa: SLF001
            if idx is None or pixel_values is None or pixel_values_videos is not None:
                return orig_model_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    mm_token_type_ids=mm_token_type_ids,
                    **kwargs,
                )
            if inputs_embeds is None:
                inputs_embeds = inner.get_input_embeddings()(input_ids)
            image_outputs = inner.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            deepstack_image_embeds = image_outputs.deepstack_features
            # OpenVINO-friendly merge: integer-index scatter (``index_copy`` ->
            # ``ScatterND``) instead of ``masked_scatter`` (-> unconvertible
            # ``Where``). Numerically identical for a single-batch sequence.
            merged = inputs_embeds[0].index_copy(0, idx, image_embeds)
            inputs_embeds = merged.unsqueeze(0)
            visual_pos_masks = input_ids == inner.config.image_token_id
            if position_ids is None:
                position_ids = inner.compute_3d_position_ids(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    mm_token_type_ids=mm_token_type_ids,
                )
            outputs = inner.language_model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_image_embeds,
                **kwargs,
            )
            return Qwen3VLModelOutputWithPast(**outputs, rope_deltas=inner.rope_deltas)

        def _patched_deepstack_process(
            hidden_states: torch.Tensor,
            visual_pos_masks: torch.Tensor,
            visual_embeds: torch.Tensor,
        ) -> torch.Tensor:
            idx = shim._image_token_indices  # noqa: SLF001
            if idx is None:
                return orig_deepstack_process(hidden_states, visual_pos_masks, visual_embeds)
            visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
            row = hidden_states[0]
            updated = row.index_copy(0, idx, row.index_select(0, idx) + visual_embeds)
            return updated.unsqueeze(0)

        inner.forward = _patched_model_forward
        text_model._deepstack_process = _patched_deepstack_process  # noqa: SLF001
        self._export_patched = True

    def _ensure_vision_patch(self) -> None:
        """Install an export-friendly ``get_image_features`` that returns stashed embeds.

        Replaces the underlying model's vision-tower call with one that returns the
        precomputed ``image_embeds`` / ``deepstack_embeds`` stashed on this shim,
        so the stock (traceable) merge / deepstack / language-model code runs
        unchanged during export.
        """
        if getattr(self, "_vision_patched", False):
            return
        shim = self

        def _patched_get_image_features(
            pixel_values: torch.Tensor | None = None,  # noqa: ARG001
            image_grid_thw: torch.LongTensor | None = None,  # noqa: ARG001
            return_dict: bool = True,  # noqa: ARG001, FBT001, FBT002
            **_kwargs: object,
        ) -> SimpleNamespace:
            image_embeds, deepstack_embeds = shim._precomputed_vision  # noqa: SLF001
            return SimpleNamespace(
                pooler_output=(image_embeds,),
                deepstack_features=list(deepstack_embeds.unbind(0)),
            )

        self.model.get_image_features = _patched_get_image_features
        self._vision_patched = True

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
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
        image_embeds: torch.Tensor | None = None,
        deepstack_embeds: torch.Tensor | None = None,
        image_token_indices: torch.Tensor | None = None,
        **kwargs: object,
    ) -> Qwen3VLCausalLMOutputWithPast:
        """Run the stock forward and attach the 3D MRoPE ``position_ids``.

        When ``image_embeds`` / ``deepstack_embeds`` are supplied (precomputed by
        :meth:`encode_vision`), the vision tower is bypassed: the stashed
        embeddings feed a patched ``get_image_features`` so the stock merge and
        deepstack code runs without the non-traceable vision tower. This is the
        path used for ``torch.export`` (ONNX / OpenVINO).

        When ``image_token_indices`` are also supplied, the boolean-mask visual
        scatters (merge + deepstack) are replaced with OpenVINO-friendly
        integer-index scatters (see :meth:`_ensure_export_patch`).

        Returns:
            The stock Qwen3-VL output with the 3D MRoPE ``position_ids`` attached.
        """
        use_precomputed_vision = image_embeds is not None
        if use_precomputed_vision:
            self._ensure_vision_patch()
            self._precomputed_vision = (image_embeds, deepstack_embeds)
            self._image_token_indices = image_token_indices
            if image_token_indices is not None:
                self._ensure_export_patch()
            # A non-``None`` ``pixel_values`` triggers the stock image branch; the
            # patched ``get_image_features`` ignores its value and returns the
            # stashed embeddings instead.
            pixel_values = image_embeds
            image_grid_thw = None

        if (
            mm_token_type_ids is None
            and input_ids is not None
            and (image_grid_thw is not None or video_grid_thw is not None)
        ):
            derived_ids = torch.zeros_like(input_ids)
            derived_ids[input_ids == self.config.image_token_id] = 1
            derived_ids[input_ids == self.config.video_token_id] = 2
            mm_token_type_ids = cast("torch.IntTensor", derived_ids)

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

        try:
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
        finally:
            if use_precomputed_vision:
                self._precomputed_vision = None
                self._image_token_indices = None
        outputs.position_ids = position_ids
        return outputs
