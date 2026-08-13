# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the baked Qwen3-VL image patchify and NumPy pixel-grid builder.

Fast, self-contained tests with no model downloads. The torch
:func:`patchify_image_grid` (baked into the exported graph) is pinned against a
plain-NumPy reimplementation of the transformers ``Qwen2VLImageProcessor``
patchify block, and the full NumPy image path
(:func:`~physicalai.policies.xr0.io.build_pixel_grid` + patchify) is checked for
parity against the *real* HuggingFace image processor (instantiated offline). That
parity is what lets the native pipeline build the pre-patchify grid directly in
NumPy instead of un-patchifying the processor's output.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from physicalai.policies.xr0.io import build_pixel_grid
from physicalai.policies.xr0.patchify import patchify_image_grid

# Qwen3-VL geometry (Qwen3-VL reuses the Qwen2-VL image processor with patch_size=16).
TEMPORAL_PATCH_SIZE = 2
PATCH_SIZE = 16
MERGE_SIZE = 2
CHANNELS = 3


def _reference_patchify(
    images: np.ndarray,
    grid_thw: list[list[int]],
    temporal_patch_size: int,
    patch_size: int,
    merge_size: int,
) -> np.ndarray:
    """Plain-NumPy reference mirroring the transformers Qwen3-VL patchify.

    Returns:
        The flat ``pixel_values`` array.
    """
    flattened = []
    for index, (grid_t, grid_h, grid_w) in enumerate(grid_thw):
        # Single-frame image temporally duplicated to ``temporal_patch_size`` frames.
        patches = np.repeat(images[index][np.newaxis], temporal_patch_size, axis=0)
        channel = patches.shape[1]
        patches = patches.reshape(
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flattened.append(
            patches.reshape(
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            ),
        )
    return np.concatenate(flattened, axis=0)


def _make_grid(grid_thw: list[list[int]]) -> torch.Tensor:
    """Build a random normalized image grid matching ``grid_thw``.

    Returns:
        A ``(num_images, C, H, W)`` float32 tensor.
    """
    generator = torch.Generator().manual_seed(0)
    height = grid_thw[0][1] * PATCH_SIZE
    width = grid_thw[0][2] * PATCH_SIZE
    return torch.randn(len(grid_thw), CHANNELS, height, width, generator=generator, dtype=torch.float32)


def test_patchify_matches_transformers_reference() -> None:
    """The torch patchify equals the plain-NumPy transformers reference."""
    grid_thw = [[1, 16, 16], [1, 16, 16]]
    images = _make_grid(grid_thw)

    result = patchify_image_grid(
        images,
        grid_thw,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        patch_size=PATCH_SIZE,
        merge_size=MERGE_SIZE,
    )
    reference = _reference_patchify(
        images.numpy(),
        grid_thw,
        TEMPORAL_PATCH_SIZE,
        PATCH_SIZE,
        MERGE_SIZE,
    )

    assert result.shape == (2 * 16 * 16, CHANNELS * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE)
    np.testing.assert_array_equal(result.numpy(), reference)


def test_numpy_grid_plus_patchify_matches_image_processor() -> None:
    """``build_pixel_grid`` + baked patchify equals the real image processor output.

    Instantiates the actual Qwen2-VL image processor Qwen3-VL uses (offline, no
    download) and checks that the NumPy grid built by ``build_pixel_grid``, once
    patchified by the baked graph op, reproduces the processor's ``pixel_values``
    (and that the geometry matches ``image_grid_thw``). This pins the native NumPy
    image path against the HuggingFace processor it replaces.
    """
    image_processor = Qwen2VLImageProcessor(
        patch_size=PATCH_SIZE,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        merge_size=MERGE_SIZE,
    )
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 256, (256, 256, CHANNELS), dtype=np.uint8) for _ in range(2)]

    # Reference: the real processor (images are already patch-aligned -> do_resize=False).
    processed = image_processor(images=images, do_resize=False, return_tensors="np")
    grid_thw = [[int(dim) for dim in row] for row in processed["image_grid_thw"].tolist()]

    grid = build_pixel_grid(
        images,
        image_processor.image_mean,
        image_processor.image_std,
        image_processor.rescale_factor,
    )
    flat = patchify_image_grid(
        torch.from_numpy(grid),
        grid_thw,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        patch_size=PATCH_SIZE,
        merge_size=MERGE_SIZE,
    ).numpy()

    assert grid.shape == (2, CHANNELS, 256, 256)
    assert flat.shape == processed["pixel_values"].shape
    np.testing.assert_allclose(flat, processed["pixel_values"], rtol=0, atol=1e-5)
