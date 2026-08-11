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


# --------------------------------------------------------------------------- #
# Export-friendly reimplementations of the stock Qwen3-VL ops.                 #
#                                                                             #
# Each of these is numerically identical to a stock ``transformers`` op but   #
# expressed so ``torch.export`` / OpenVINO can convert it. They are           #
# module-level (not closures) so each can be unit-tested in isolation against #
# its stock counterpart; ``XR0Qwen3VL._ensure_export_patch`` installs them.   #
# --------------------------------------------------------------------------- #


def export_rot_pos_emb(visual: torch.nn.Module, grid_thw_list: list[list[int]]) -> torch.Tensor:  # noqa: PLR0914
    """Vision rotary position embeddings driven by a *Python* grid list.

    Numerically identical to stock ``Qwen3VLVisionModel.rot_pos_emb``, but it
    iterates over the Python constant ``grid_thw_list`` instead of
    ``grid_thw.tolist()``. Under ``torch.export`` the baked ``grid_thw`` buffer is
    a lifted tensor input, so ``.tolist()`` would yield unbacked symints and the
    per-image ``arange`` / ``reshape`` shapes could not be resolved; the constant
    ints keep every shape concrete. For each image it enumerates the (row, col)
    patch coordinates in ``spatial_merge_size`` blocks, gathers their rotary
    frequencies from the shared table and flattens them.

    Args:
        visual: The vision tower (reads ``spatial_merge_size`` and ``rotary_pos_emb``).
        grid_thw_list: Per-image ``[t, h, w]`` geometry as plain Python ints.

    Returns:
        The flattened rotary position embeddings for all vision patches.
    """
    merge_size = visual.spatial_merge_size

    max_hw = max(max(h, w) for _, h, w in grid_thw_list)
    freq_table = visual.rotary_pos_emb(max_hw)
    device = freq_table.device

    total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

    offset = 0
    for num_frames, height, width in grid_thw_list:
        # Patch coordinates are laid out in spatial_merge_size x spatial_merge_size
        # blocks (matching how the merger later folds neighbouring patches
        # together), so build row/col indices as block-offset + intra-block-offset.
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
            # Temporal frames share the same spatial grid, so repeat it.
            coords = coords.repeat(num_frames, 1)
        num_tokens = coords.shape[0]
        pos_ids[offset : offset + num_tokens] = coords
        offset += num_tokens

    embeddings = freq_table[pos_ids]
    return embeddings.flatten(1)


def export_fast_pos_embed_interpolate(visual: torch.nn.Module, grid_thw_list: list[list[int]]) -> torch.Tensor:  # noqa: PLR0914
    """Bilinearly interpolate the learned position embeddings to the grid.

    Numerically identical to stock ``Qwen3VLVisionModel.fast_pos_embed_interpolate``,
    but driven by the Python constant ``grid_thw_list`` so the per-image
    ``torch.linspace(0, num_grid_per_side - 1, h)`` calls take concrete sizes.
    Under ``torch.export`` those ``linspace`` bounds come from ``grid_thw`` (a
    lifted tensor input via ``.tolist()``), which triggers a data-dependent guard;
    the constant ints avoid it. For each image it maps the target ``h x w`` grid
    onto the learned ``num_grid_per_side`` grid, gathers the four surrounding
    embeddings and blends them with the fractional ``(dh, dw)`` bilinear weights,
    then reorders the patches into ``spatial_merge_size`` blocks.

    Args:
        visual: The vision tower (reads ``num_grid_per_side``, ``pos_embed`` and
            ``config.spatial_merge_size``).
        grid_thw_list: Per-image ``[t, h, w]`` geometry as plain Python ints.

    Returns:
        The interpolated position embeddings for all vision patches.
    """
    grid_ts = [row[0] for row in grid_thw_list]
    grid_hs = [row[1] for row in grid_thw_list]
    grid_ws = [row[2] for row in grid_thw_list]
    device = visual.pos_embed.weight.device

    # Four accumulators = the four bilinear corners (floor/ceil x floor/ceil) of
    # the learned-grid neighbours for every target patch.
    idx_list: list[list[float]] = [[] for _ in range(4)]
    weight_list: list[list[float]] = [[] for _ in range(4)]

    for _t, h, w in grid_thw_list:
        # Sample positions on the learned grid for the target h/w axes.
        h_idxs = torch.linspace(0, visual.num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, visual.num_grid_per_side - 1, w)
        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clip(max=visual.num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clip(max=visual.num_grid_per_side - 1)
        # Fractional distances -> bilinear interpolation weights.
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
    # Blend the four gathered corners with their bilinear weights.
    pos_embeds = visual.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
    patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws, strict=False)])

    # Reorder each image's patches into spatial_merge_size blocks so they line up
    # with the tower's merged token ordering.
    patch_pos_embeds_permute = []
    merge_size = visual.config.spatial_merge_size
    for patch_pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws, strict=False):
        pos_embed = patch_pos_embed.repeat(t, 1)
        pos_embed = (
            pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        patch_pos_embeds_permute.append(pos_embed)
    return torch.cat(patch_pos_embeds_permute)


def export_vision_attn_forward(
    attn: torch.nn.Module,
    split_sizes: list[int],
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Export-friendly vision attention for one block.

    Numerically identical to stock ``Qwen3VLVisionAttention.forward`` (non-flash
    path), but it splits the per-image attention windows by the constant Python
    ``split_sizes`` instead of ``lengths.tolist()`` (derived from ``cu_seqlens``,
    which yields unbacked symints under ``torch.export``) and calls SDPA directly.
    The shared attention interface passes ``enable_gqa=True``, which the ONNX
    exporter rejects unless ``q_heads > kv_heads``; the vision tower has equal
    q/kv heads, so a plain SDPA is numerically identical.

    Args:
        attn: The vision attention module (reads ``qkv``, ``proj``, ``num_heads``,
            ``scaling``).
        split_sizes: Per-window token counts summing to the sequence length.
        hidden_states: ``(seq_len, dim)`` input hidden states.
        position_embeddings: The ``(cos, sin)`` rotary embeddings.

    Returns:
        The ``(seq_len, dim)`` attention output.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision  # noqa: PLC0415

    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        attn.qkv(hidden_states).reshape(seq_length, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    splits = [torch.split(tensor, split_sizes, dim=2) for tensor in (query_states, key_states, value_states)]
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
        for q, k, v in zip(*splits, strict=False)
    ]
    attn_output = torch.cat(attn_outputs, dim=1)
    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    return attn.proj(attn_output)


def export_scatter_visual_embeds(
    inputs_embeds: torch.Tensor,
    image_token_indices: torch.Tensor,
    image_embeds: torch.Tensor,
) -> torch.Tensor:
    """Merge visual embeds into token embeddings by integer index.

    OpenVINO-friendly replacement for the stock image/text merge, which uses
    ``masked_scatter`` (-> an unconvertible ``Where`` whose operand shapes
    disagree). The integer-index ``index_copy`` (-> ``ScatterND``) is numerically
    identical for a single-batch sequence.

    Args:
        inputs_embeds: ``(1, seq_len, hidden)`` token embeddings.
        image_token_indices: ``(num_visual,)`` positions of the image tokens.
        image_embeds: ``(num_visual, hidden)`` visual embeddings.

    Returns:
        ``(1, seq_len, hidden)`` embeddings with the image slots replaced.
    """
    merged = inputs_embeds[0].index_copy(0, image_token_indices, image_embeds)
    return merged.unsqueeze(0)


def export_add_deepstack_embeds(
    hidden_states: torch.Tensor,
    image_token_indices: torch.Tensor,
    visual_embeds: torch.Tensor,
) -> torch.Tensor:
    """Add deepstack visual features at the image-token positions by index.

    OpenVINO-friendly replacement for the stock ``_deepstack_process``, which adds
    ``visual_embeds`` into ``hidden_states`` via boolean-mask assignment (-> an
    unconvertible ``Where``). The ``index_select`` + ``index_copy`` variant
    (-> ``Gather`` / ``ScatterND``) is numerically identical for a single-batch
    sequence.

    Args:
        hidden_states: ``(1, seq_len, hidden)`` decoder hidden states.
        image_token_indices: ``(num_visual,)`` positions of the image tokens.
        visual_embeds: ``(num_visual, hidden)`` deepstack features to add.

    Returns:
        ``(1, seq_len, hidden)`` hidden states with the features added.
    """
    visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
    row = hidden_states[0]
    updated = row.index_copy(0, image_token_indices, row.index_select(0, image_token_indices) + visual_embeds)
    return updated.unsqueeze(0)


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

        For a fixed image size the ``image_grid_thw`` geometry, the image-token
        positions and the MRoPE ``position_ids`` of the fixed prefix (system
        prompt + image grid) are deterministic. This precomputes them once from a
        representative (right-padded) sample and stores them as non-persistent
        constant buffers, then enables in-graph export mode. During :meth:`forward`
        (export mode) the real vision tower runs on ``pixel_values`` with the
        constant ``image_grid_thw`` (so its ``tensor.tolist()`` geometry folds),
        the ``position_ids`` are rebuilt for the runtime prompt length (the fixed
        prefix reused verbatim, the variable-length post-image task text
        recomputed from ``attention_mask`` -- see
        :meth:`_runtime_export_position_ids` -- skipping the non-traceable
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
        # Everything up to (and including) the last image token -- the fixed
        # system-prompt text and the fixed image grid -- has deterministic MRoPE
        # positions, so it stays baked. The *task text* after the image varies in
        # length between prompts, so its positions must be recomputed at inference
        # from the runtime ``attention_mask`` (see :meth:`_runtime_export_position_ids`);
        # otherwise a prompt longer than this sample would leave its trailing
        # tokens with the sample's padding position id (0), corrupting RoPE. The
        # post-image text is plain 1D sequential (all three MRoPE axes equal), so
        # we only need where it starts and its first position value.
        post_image_start = int(image_token_indices.max().item()) + 1
        self._export_post_image_start = post_image_start
        self._export_post_image_base = int(position_ids[0, 0, post_image_start].item())
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

    def _runtime_export_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Rebuild the export MRoPE ``position_ids`` for the runtime prompt length.

        The baked ``_export_position_ids`` are only correct for prompts whose
        valid length matches the export sample: a longer prompt's trailing task
        tokens would inherit the sample's padding position id (0). The prefix
        (system-prompt text + fixed image grid, up to ``_export_post_image_start``)
        is identical for every prompt, so it is reused verbatim; the post-image
        task text is plain 1D sequential (all three MRoPE axes equal), so its
        positions are recomputed as ``base + cumulative_valid_index`` from the
        runtime ``attention_mask``. All ops are trace-friendly (``ScatterND``-free
        elementwise / ``cumsum`` / ``where``), unlike the data-dependent
        ``get_rope_index`` builder.

        Args:
            attention_mask: The runtime attention mask ``(1, L)``.

        Returns:
            The 3D MRoPE ``position_ids`` tensor ``(3, 1, L)`` for this prompt.
        """
        baked = self._export_position_ids
        seq_len = baked.shape[-1]
        mask = attention_mask.reshape(-1)[:seq_len].to(torch.long)
        seq_index = torch.arange(seq_len, device=baked.device)
        in_tail = seq_index >= self._export_post_image_start
        valid_tail = mask * in_tail.to(torch.long)
        # 0-based running index among the valid post-image tokens.
        tail_index = torch.cumsum(valid_tail, dim=0) - 1
        tail_pos = (self._export_post_image_base + tail_index).clamp_min(0)
        use_tail = (in_tail & (mask > 0)).reshape(1, 1, seq_len).expand_as(baked)
        tail_pos = tail_pos.reshape(1, 1, seq_len).expand_as(baked)
        return torch.where(use_tail, tail_pos, baked)

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
        """Swap the stock Qwen3-VL ops for their export-friendly equivalents.

        Installs the module-level ``export_*`` reimplementations onto the vision
        tower and language model (vision attention / rotary / position-embed
        geometry, and the image-merge / deepstack scatters). Each is numerically
        identical to stock but OpenVINO-convertible; see those functions for why.
        Idempotent, and only takes effect once export constants are baked.
        """
        if getattr(self, "_export_patched", False):
            return
        shim = self
        inner = self.model
        visual = inner.visual
        text_model = inner.language_model
        orig_model_forward = inner.forward
        orig_deepstack_process = text_model._deepstack_process  # noqa: SLF001

        def _make_vision_attn_forward(attn: torch.nn.Module) -> object:
            """Build an export-friendly ``forward`` for one vision attention block.

            Thin wrapper binding one attention module and the baked per-window
            token counts to :func:`export_vision_attn_forward`.

            Returns:
                The export-friendly ``forward`` callable for the block.
            """

            def _forward(
                hidden_states: torch.Tensor,
                cu_seqlens: torch.Tensor | None = None,  # noqa: ARG001
                rotary_pos_emb: torch.Tensor | None = None,  # noqa: ARG001
                position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
                **kwargs: object,  # noqa: ARG001
            ) -> torch.Tensor:
                return export_vision_attn_forward(
                    attn,
                    shim._export_vision_seqlens,  # noqa: SLF001
                    hidden_states,
                    cast("tuple[torch.Tensor, torch.Tensor]", position_embeddings),
                )

            return _forward

        def _patched_rot_pos_emb(grid_thw: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
            return export_rot_pos_emb(visual, shim._export_grid_list)  # noqa: SLF001

        def _patched_fast_pos_embed_interpolate(grid_thw: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
            return export_fast_pos_embed_interpolate(visual, shim._export_grid_list)  # noqa: SLF001

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
            inputs_embeds = export_scatter_visual_embeds(inputs_embeds, idx, image_embeds)
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
            """Add the deepstack visual features at the image-token positions.

            Thin wrapper over :func:`export_add_deepstack_embeds` using the baked
            image-token indices; falls back to stock when they are absent (normal,
            non-export inference).

            Returns:
                ``hidden_states`` with the deepstack features added.
            """
            idx = shim._image_token_indices  # noqa: SLF001
            if idx is None:
                return orig_deepstack_process(hidden_states, visual_pos_masks, visual_embeds)
            return export_add_deepstack_embeds(hidden_states, idx, visual_embeds)

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

        Returns:
            The stock Qwen3-VL output with the 3D MRoPE ``position_ids`` attached.
        """
        if getattr(self, "_ingraph_export", False):
            image_grid_thw = self._export_image_grid_thw
            position_ids = self._runtime_export_position_ids(attention_mask)
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
