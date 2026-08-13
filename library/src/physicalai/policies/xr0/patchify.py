# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Reversible Qwen3-VL image patchify used to bake the reshape/transpose into the graph.

The Qwen3-VL image processor turns a normalized image grid ``(num_images, C, H, W)``
into a flat ``pixel_values`` tensor ``(sum(grid_t*grid_h*grid_w), C*tp*ps*ps)`` via a
static temporal duplication + reshape/transpose (see the transformers
``Qwen2VLImageProcessor._preprocess`` patchify block). Because the grid geometry is
fixed at export time, that op is a constant reshape/transpose and can be *baked into*
the exported OpenVINO graph.

This module provides the tensor op used to do that:

* :func:`patchify_image_grid` reproduces the transformers patchify. It runs *inside*
  the exported graph so the graph's ``pixel_values`` input is the pre-patchify
  normalized image grid instead of the flat patch tensor.

The inverse direction is not needed: the Runtime preprocessor builds the normalized
image grid directly in NumPy (see
:func:`physicalai.policies.xr0.io.build_pixel_grid`) rather than un-patchifying the
HuggingFace image processor's output.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

# Transposition transformers applies after the 9-D reshape.
_PATCHIFY_PERM = (0, 3, 6, 4, 7, 2, 1, 5, 8)


def patchify_image_grid(
    images: torch.Tensor,
    grid_thw: Sequence[Sequence[int]],
    *,
    temporal_patch_size: int,
    patch_size: int,
    merge_size: int,
) -> torch.Tensor:
    """Patchify a normalized image grid exactly like the Qwen3-VL image processor.

    Mirrors the transformers ``Qwen2VLImageProcessor._preprocess`` patchify: each
    single-frame image is temporally duplicated to ``temporal_patch_size`` frames,
    reshaped into merge-size patch blocks and transposed, then flattened to the flat
    ``pixel_values`` layout the vision tower consumes.

    Args:
        images: Normalized image grid of shape ``(num_images, C, H, W)`` where
            ``H == grid_h * patch_size`` and ``W == grid_w * patch_size`` for the
            matching ``grid_thw`` row.
        grid_thw: One ``(grid_t, grid_h, grid_w)`` triple per image (``grid_t`` is the
            temporal group count, ``1`` for a still image).
        temporal_patch_size: Number of frames grouped per temporal patch.
        patch_size: Spatial patch side length in pixels.
        merge_size: Spatial merge block size.

    Returns:
        The flat ``pixel_values`` tensor of shape
        ``(sum(grid_t * grid_h * grid_w), C * temporal_patch_size * patch_size ** 2)``.
    """
    flattened: list[torch.Tensor] = []
    for index, (grid_t, grid_h, grid_w) in enumerate(grid_thw):
        image = images[index : index + 1]  # (1, C, H, W)
        channel = image.shape[1]
        patches = image.repeat(temporal_patch_size, 1, 1, 1)  # (tp, C, H, W)
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
        patches = patches.permute(*_PATCHIFY_PERM)
        flattened.append(
            patches.reshape(
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            ),
        )
    return torch.cat(flattened, dim=0)

