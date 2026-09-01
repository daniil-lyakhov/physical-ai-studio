# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torch-free Qwen3-VL chat-prompt rendering for XR0.

Reproduces, in pure Python, the exact string the Qwen3-VL processor's
``apply_chat_template`` emits for the XR0 multi-view message -- with each
``<|image_pad|>`` placeholder already expanded to the number of image tokens the
processor would insert. Tokenizing the rendered string with a bare tokenizer
(``add_special_tokens=False``) therefore reproduces the processor's ``input_ids``
bit-for-bit, so inference does not need to load the full ``Qwen3VLProcessor`` (an
exported OpenVINO tokenizer or a plain HF tokenizer suffices).

The parity with the real processor is asserted in
``tests/unit/policies/xr0/test_prompt.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

# --- prompt text (shared with the training preprocessor) -------------------
_MULTI_VIEW_HEADER = "The following observations are captured from multiple views.\n"
_TASK_TEMPLATE = "Generate robot actions for the task:\n{instruction} /no_cot"
_ASSISTANT_PRIMER = "<cot></cot>"

# View titles the model was trained with (Xiaomi reference server prompt in
# deploy/server.py), e.g. "wrist_left" -> "Left-Wrist" so the prompt reads
# "# Left-Wrist View". A plain capitalize would wrongly yield "Wrist Left".
_VIEW_TITLES = {
    "base": "Base",
    "wrist_left": "Left-Wrist",
    "wrist_right": "Right-Wrist",
}

# Qwen3-VL chat special tokens rendered as literal text (a bare tokenizer maps
# each to its dedicated special-token id).
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_VISION_START = "<|vision_start|>"
_VISION_END = "<|vision_end|>"
_IMAGE_PAD = "<|image_pad|>"


def view_title(view: str) -> str:
    """Human-readable view title matching the reference prompt.

    Known views use the reference eval's exact titles (e.g. ``"wrist_left"`` ->
    ``"Left-Wrist"``); unknown views fall back to a capitalized join.

    Returns:
        The human-readable view title.
    """
    key = view.replace("-", "_")
    if key in _VIEW_TITLES:
        return _VIEW_TITLES[key]
    return " ".join(word.capitalize() for word in key.split("_"))


def image_pad_count(grid_t: int, grid_h: int, grid_w: int, merge_size: int) -> int:
    """Number of ``<|image_pad|>`` tokens the processor expands one image into.

    Mirrors the Qwen2/Qwen3-VL image-token count
    ``prod(grid_thw) // merge_size**2``.

    Returns:
        The image-pad token count for a single image.
    """
    return (grid_t * grid_h * grid_w) // (merge_size * merge_size)


def render_chat_prompt(views: Sequence[str], pad_counts: Sequence[int], instruction: str) -> str:
    """Render the XR0 Qwen3-VL chat prompt as a raw string.

    Reproduces ``processor.apply_chat_template(..., tokenize=False)`` for the XR0
    multi-view message, with each image's ``<|image_pad|>`` already expanded to
    ``pad_counts[i]`` copies. Tokenizing the result with ``add_special_tokens=
    False`` yields the same ``input_ids`` as the processor.

    Args:
        views: Ordered camera view names (one per image).
        pad_counts: Per-view ``<|image_pad|>`` token counts (see
            :func:`image_pad_count`), aligned with ``views``.
        instruction: The task instruction text.

    Returns:
        The fully-rendered chat prompt string.

    Raises:
        ValueError: If ``views`` and ``pad_counts`` have different lengths.
    """
    if len(views) != len(pad_counts):
        msg = f"views ({len(views)}) and pad_counts ({len(pad_counts)}) must have the same length"
        raise ValueError(msg)

    parts: list[str] = [_MULTI_VIEW_HEADER]
    for view, count in zip(views, pad_counts, strict=True):
        parts.append(f"# {view_title(view)} View\n")
        parts.append(_VISION_START + _IMAGE_PAD * count + _VISION_END)
        parts.append("\n")
    parts.append(_TASK_TEMPLATE.format(instruction=instruction))
    user = "".join(parts)
    return f"{_IM_START}user\n{user}{_IM_END}\n{_IM_START}assistant\n{_ASSISTANT_PRIMER}{_IM_END}\n"
