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
    def prepare_ingraph_export(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        image_grid_thw: torch.LongTensor,
    ) -> None:
        """Bake the fixed image geometry as constants for a self-contained export.

        For a fixed image size and prompt layout the 3D MRoPE ``position_ids``, the
        ``image_grid_thw`` geometry and the image-token positions are all
        deterministic. This precomputes them once from a representative
        (right-padded) sample and stores them as non-persistent constant buffers,
        then enables in-graph export mode. During :meth:`forward` (export mode) the
        real vision tower runs on ``pixel_values`` with the constant
        ``image_grid_thw`` (so its ``tensor.tolist()`` geometry folds), the
        constant ``position_ids`` are injected (skipping the non-traceable
        rope-index builder), and the merge / deepstack scatters use the constant
        integer index (``index_copy`` -> ``ScatterND``) instead of the
        OpenVINO-hostile ``masked_scatter``.

        Args:
            input_ids: Token ids of the representative padded prompt ``(1, L)``.
            attention_mask: Attention mask of the same prompt ``(1, L)``.
            image_grid_thw: The fixed vision geometry ``(num_images, 3)``.
        """
        position_ids = self.build_3d_position_ids(
            input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
        )
        image_token_indices = self.image_token_positions(input_ids)
        for name, tensor in (
            ("_export_image_grid_thw", image_grid_thw),
            ("_export_position_ids", position_ids),
            ("_export_image_token_indices", image_token_indices),
        ):
            if hasattr(self, name):
                delattr(self, name)
            self.register_buffer(name, tensor.detach().clone(), persistent=False)
        # Keep the vision geometry as a *Python* constant too. ``torch.export``
        # lifts registered buffers as tensor inputs, so ``grid_thw.tolist()`` in
        # the vision tower would yield unbacked symints; the export-time tower
        # patch consumes these concrete ints instead (see :meth:`_ensure_export_patch`).
        self._export_grid_list = [[int(dim) for dim in row] for row in image_grid_thw.tolist()]
        # Per-window token counts for the vision attention. Stock builds these
        # from ``cu_seqlens`` and calls ``lengths.tolist()`` (unbacked symints
        # under export); the attention patch splits by these constant ints instead.
        self._export_vision_seqlens = [h * w for t, h, w in self._export_grid_list for _ in range(t)]
        self._ingraph_export = True

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

        It also rewrites the vision tower's two geometry helpers
        (``rot_pos_emb`` / ``fast_pos_embed_interpolate``) to read the *Python*
        constant ``_export_grid_list`` instead of ``grid_thw.tolist()``. Under
        ``torch.export`` the baked ``grid_thw`` buffer is a lifted tensor input, so
        ``.tolist()`` yields unbacked symints and ``torch.linspace(0, N, h)`` fails
        with a data-dependent guard; the constant ints make those shapes concrete.
        The results are numerically identical to stock.
        """
        if getattr(self, "_export_patched", False):
            return
        shim = self
        inner = self.model
        visual = inner.visual
        text_model = inner.language_model
        orig_model_forward = inner.forward
        orig_deepstack_process = text_model._deepstack_process  # noqa: SLF001

        from transformers.models.qwen3_vl import modeling_qwen3_vl as _mqv

        def _make_vision_attn_forward(attn: torch.nn.Module) -> object:
            """Build an export-friendly ``forward`` for one vision attention block.

            Stock ``Qwen3VLVisionAttention.forward`` splits the per-image attention
            windows with ``lengths.tolist()`` (derived from ``cu_seqlens``), which
            yields unbacked symints under ``torch.export``. This variant splits by
            the constant Python window sizes (``_export_vision_seqlens``) and drops
            the flash-attention branch (XR0 uses SDPA); it is numerically identical.
            """

            def _forward(
                hidden_states: torch.Tensor,
                cu_seqlens: torch.Tensor,  # noqa: ARG001
                rotary_pos_emb: torch.Tensor | None = None,  # noqa: ARG001
                position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
                **kwargs: object,
            ) -> torch.Tensor:
                seq_length = hidden_states.shape[0]
                query_states, key_states, value_states = (
                    attn.qkv(hidden_states).reshape(seq_length, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
                )
                cos, sin = position_embeddings
                query_states, key_states = _mqv.apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
                query_states = query_states.transpose(0, 1).unsqueeze(0)
                key_states = key_states.transpose(0, 1).unsqueeze(0)
                value_states = value_states.transpose(0, 1).unsqueeze(0)

                split_sizes = shim._export_vision_seqlens  # noqa: SLF001
                splits = [
                    torch.split(tensor, split_sizes, dim=2)
                    for tensor in (query_states, key_states, value_states)
                ]
                # Call SDPA directly (not the shared attention interface): the
                # vision tower has equal q/kv heads, but the interface passes
                # ``enable_gqa=True`` which the ONNX exporter rejects unless
                # q_heads > kv_heads. A plain SDPA is numerically identical here.
                attn_outputs = [
                    torch.nn.functional.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=False,
                        scale=attn.scaling,
                    ).transpose(1, 2)
                    for q, k, v in zip(*splits)
                ]
                attn_output = torch.cat(attn_outputs, dim=1)
                attn_output = attn_output.reshape(seq_length, -1).contiguous()
                return attn.proj(attn_output)

            return _forward

        def _patched_rot_pos_emb(grid_thw: torch.Tensor) -> torch.Tensor:
            merge_size = visual.spatial_merge_size
            grid_thw_list = shim._export_grid_list  # noqa: SLF001

            max_hw = max(max(h, w) for _, h, w in grid_thw_list)
            freq_table = visual.rotary_pos_emb(max_hw)
            device = freq_table.device

            total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
            pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

            offset = 0
            for num_frames, height, width in grid_thw_list:
                merged_h, merged_w = height // merge_size, width // merge_size
                block_rows = torch.arange(merged_h, device=device)
                block_cols = torch.arange(merged_w, device=device)
                intra_row = torch.arange(merge_size, device=device)
                intra_col = torch.arange(merge_size, device=device)
                row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
                col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
                row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
                col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
                coords = torch.stack((row_idx, col_idx), dim=-1)
                if num_frames > 1:
                    coords = coords.repeat(num_frames, 1)
                num_tokens = coords.shape[0]
                pos_ids[offset : offset + num_tokens] = coords
                offset += num_tokens

            embeddings = freq_table[pos_ids]
            return embeddings.flatten(1)

        def _patched_fast_pos_embed_interpolate(grid_thw: torch.Tensor) -> torch.Tensor:
            grid_thw_list = shim._export_grid_list  # noqa: SLF001
            grid_ts = [row[0] for row in grid_thw_list]
            grid_hs = [row[1] for row in grid_thw_list]
            grid_ws = [row[2] for row in grid_thw_list]
            device = visual.pos_embed.weight.device

            idx_list: list[list[float]] = [[] for _ in range(4)]
            weight_list: list[list[float]] = [[] for _ in range(4)]

            for _t, h, w in grid_thw_list:
                h_idxs = torch.linspace(0, visual.num_grid_per_side - 1, h)
                w_idxs = torch.linspace(0, visual.num_grid_per_side - 1, w)
                h_idxs_floor = h_idxs.int()
                w_idxs_floor = w_idxs.int()
                h_idxs_ceil = (h_idxs.int() + 1).clip(max=visual.num_grid_per_side - 1)
                w_idxs_ceil = (w_idxs.int() + 1).clip(max=visual.num_grid_per_side - 1)
                dh = h_idxs - h_idxs_floor
                dw = w_idxs - w_idxs_floor
                base_h = h_idxs_floor * visual.num_grid_per_side
                base_h_ceil = h_idxs_ceil * visual.num_grid_per_side
                indices = [
                    (base_h[None].T + w_idxs_floor[None]).flatten(),
                    (base_h[None].T + w_idxs_ceil[None]).flatten(),
                    (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                    (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
                ]
                weights = [
                    ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                    ((1 - dh)[None].T * dw[None]).flatten(),
                    (dh[None].T * (1 - dw)[None]).flatten(),
                    (dh[None].T * dw[None]).flatten(),
                ]
                for i in range(4):
                    idx_list[i].extend(indices[i].tolist())
                    weight_list[i].extend(weights[i].tolist())

            idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
            weight_tensor = torch.tensor(weight_list, dtype=visual.pos_embed.weight.dtype, device=device)
            pos_embeds = visual.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
            patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
            patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

            patch_pos_embeds_permute = []
            merge_size = visual.config.spatial_merge_size
            for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
                pos_embed = pos_embed.repeat(t, 1)
                pos_embed = (
                    pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                    .permute(0, 1, 3, 2, 4, 5)
                    .flatten(0, 4)
                )
                patch_pos_embeds_permute.append(pos_embed)
            return torch.cat(patch_pos_embeds_permute)

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
            # Run the tower directly (not ``get_image_features``) so its
            # ``split_sizes.tolist()`` -- another unbacked-symint source -- is
            # skipped; the two patched geometry helpers above already consume the
            # constant grid, and a freshly built *constant* grid tensor keeps the
            # tower's inline ``cu_seqlens`` (tensor ops on ``grid_thw``) concrete.
            grid_const = torch.tensor(
                shim._export_grid_list,  # noqa: SLF001
                dtype=torch.long,
                device=pixel_values.device,
            )
            vision_output = visual(
                pixel_values.type(visual.dtype),
                grid_thw=grid_const,
                return_dict=True,
            )
            image_embeds = vision_output.pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)
            deepstack_image_embeds = vision_output.deepstack_features
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
        visual.rot_pos_emb = _patched_rot_pos_emb
        visual.fast_pos_embed_interpolate = _patched_fast_pos_embed_interpolate
        for block in visual.blocks:
            block.attn.forward = _make_vision_attn_forward(block.attn)
        self._export_patched = True

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
        **kwargs: object,
    ) -> Qwen3VLCausalLMOutputWithPast:
        """Run the stock forward and attach the 3D MRoPE ``position_ids``.

        Normal (eager) inference runs the fully stock path: the vision tower, the
        3D MRoPE ``position_ids`` (:meth:`~transformers.Qwen3VLModel.compute_3d_position_ids`)
        and the boolean-mask visual scatter all run unchanged.

        In-graph export mode (enabled by :meth:`prepare_ingraph_export`) keeps all
        three inside the traced graph while making them OpenVINO-convertible: the
        real vision tower runs on ``pixel_values`` with the baked-constant
        ``image_grid_thw`` (so its ``tensor.tolist()`` geometry folds), the
        baked-constant ``position_ids`` are injected (skipping the non-traceable
        rope-index builder), and the merge / deepstack scatters use the
        baked-constant integer index (``index_copy`` -> ``ScatterND``) instead of
        ``masked_scatter`` (see :meth:`_ensure_export_patch`).

        Returns:
            The stock Qwen3-VL output with the 3D MRoPE ``position_ids`` attached.
        """
        if getattr(self, "_ingraph_export", False):
            image_grid_thw = self._export_image_grid_thw
            position_ids = self._export_position_ids
            self._image_token_indices = self._export_image_token_indices
            self._ensure_export_patch()
            if mm_token_type_ids is None and input_ids is not None:
                # Image tokens -> 1 (XR0 has no video). A pure elementwise cast, so
                # it traces without the boolean-scatter ``Where`` of the masked
                # assignment used on the eager path below.
                mm_token_type_ids = cast(
                    "torch.IntTensor",
                    (input_ids == self.config.image_token_id).to(torch.int32),
                )
        elif (
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
