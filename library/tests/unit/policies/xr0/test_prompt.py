# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for the torch-free XR0 chat-prompt renderer.

Asserts that the pure-Python :func:`~physicalai.policies.xr0.prompt.render_chat_prompt`,
tokenized by a bare tokenizer with ``add_special_tokens=False``, reproduces the
``input_ids`` the real ``Qwen3VLProcessor.apply_chat_template`` produces. This is
the gate that lets XR0 inference drop the full processor in favour of an exported
OpenVINO tokenizer (or a plain HF tokenizer) fed a rendered string.

The processor is loaded offline from the local HuggingFace cache; the test skips
when it is not available (no network in CI).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

from physicalai.policies.xr0.prompt import (
    _ASSISTANT_PRIMER,
    _MULTI_VIEW_HEADER,
    _TASK_TEMPLATE,
    image_pad_count,
    render_chat_prompt,
    view_title,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_PROCESSOR_NAME = "Qwen/Qwen3-VL-4B-Instruct"


@pytest.fixture(scope="module")
def processor() -> object:
    """Load the Qwen3-VL processor offline, skipping when it is unavailable.

    Returns:
        The loaded ``Qwen3VLProcessor``.
    """
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoProcessor.from_pretrained(_PROCESSOR_NAME)
    except Exception as exc:  # noqa: BLE001 - any load failure => skip
        pytest.skip(f"Qwen3-VL processor unavailable offline: {exc}")


def _build_reference_messages(
    views: Sequence[str],
    images: Sequence[Image.Image],
    instruction: str,
) -> list[list[dict[str, object]]]:
    """Build the XR0 chat message exactly as ``XR0Preprocessor._build_message`` does.

    Returns:
        The single-sample batch of Qwen3-VL chat messages.
    """
    content: list[dict[str, object]] = [{"type": "text", "text": _MULTI_VIEW_HEADER}]
    for view, image in zip(views, images, strict=True):
        content.extend((
            {"type": "text", "text": f"# {view_title(view)} View\n"},
            {"type": "image", "image": image},
            {"type": "text", "text": "\n"},
        ))
    content.append({"type": "text", "text": _TASK_TEMPLATE.format(instruction=instruction)})
    return [
        [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": _ASSISTANT_PRIMER}]},
        ],
    ]


def _pad_counts(images: Sequence[Image.Image], patch_size: int, merge_size: int) -> list[int]:
    """Per-image ``<|image_pad|>`` counts from the (already patch-aligned) image sizes.

    Returns:
        The image-pad token count for each image.
    """
    counts: list[int] = []
    for image in images:
        width, height = image.size
        counts.append(image_pad_count(1, height // patch_size, width // patch_size, merge_size))
    return counts


@pytest.mark.parametrize(
    ("views", "sizes", "instruction"),
    [
        (("base", "wrist_left"), [(64, 64), (64, 64)], "pick the ball"),
        (("base", "wrist_left"), [(256, 256), (256, 256)], "put the cube in the box"),
        (("base", "wrist_left"), [(128, 96), (96, 128)], "close the drawer"),
        (("base",), [(160, 160)], "wave"),
        (("base", "wrist_left", "wrist_right"), [(64, 64), (96, 64), (64, 96)], "stack the blocks"),
    ],
)
def test_render_prompt_matches_processor_input_ids(
    processor: object,
    views: tuple[str, ...],
    sizes: list[tuple[int, int]],
    instruction: str,
) -> None:
    """Rendered-string tokens equal the processor's ``apply_chat_template`` ids.

    For several view counts and image geometries (incl. non-square), the bare
    tokenizer applied to :func:`render_chat_prompt` must reproduce, bit-for-bit,
    the ids the full Qwen3-VL processor emits with images.
    """
    tokenizer = processor.tokenizer  # type: ignore[attr-defined]
    image_processor = processor.image_processor  # type: ignore[attr-defined]
    patch_size = image_processor.patch_size
    merge_size = image_processor.merge_size

    images = [Image.fromarray(np.zeros((height, width, 3), np.uint8)) for width, height in sizes]

    ground_truth = processor.apply_chat_template(  # type: ignore[attr-defined]
        _build_reference_messages(views, images, instruction),
        tokenize=True,
        return_dict=True,
        return_tensors="np",
        processor_kwargs={"padding": True, "images_kwargs": {"do_resize": False}},
    )["input_ids"][0]

    prompt = render_chat_prompt(views, _pad_counts(images, patch_size, merge_size), instruction)
    rendered_ids = np.asarray(tokenizer(prompt, add_special_tokens=False, return_tensors="np")["input_ids"][0])

    assert rendered_ids.shape == ground_truth.shape
    np.testing.assert_array_equal(rendered_ids, ground_truth)


def test_image_pad_count_matches_merge_formula() -> None:
    """``image_pad_count`` equals ``prod(grid_thw) // merge_size**2``."""
    assert image_pad_count(1, 4, 4, 2) == 4
    assert image_pad_count(1, 16, 16, 2) == 64
    assert image_pad_count(1, 6, 4, 2) == 6
    assert image_pad_count(2, 4, 4, 2) == 8


def test_render_prompt_length_mismatch_raises() -> None:
    """Mismatched ``views`` / ``pad_counts`` lengths raise ``ValueError``."""
    with pytest.raises(ValueError, match="same length"):
        render_chat_prompt(("base", "wrist_left"), [4], "do it")
