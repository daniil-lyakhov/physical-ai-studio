# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compare the training-side XR0 preprocessor against the exported runtime one.

The exported IR bakes the image-token geometry and the MRoPE prefix from the
export sample, so the deployed NumPy preprocessor + OpenVINO tokenizer must
reproduce the HuggingFace ``apply_chat_template`` tokens and the Qwen3-VL
``pixel_values`` layout. Any token shift silently misplaces the image block and
corrupts the conditioning, which shows up as a large action error even though
the graph itself is fine.

Feeds one dataset frame through both paths and diffs ``input_ids`` /
``pixel_values`` / ``state``.

Usage:
    python probe_xr0_preproc_parity.py
"""

from __future__ import annotations

import numpy as np
import torch

from physicalai.inference import InferenceModel
from physicalai.inference.constants import IMAGES, STATE, TASK
from physicalai.policies.xr0.preprocessor import XR0Preprocessor

MODEL_PATH = "/home/devuser/dlyakhov/physical-ai-studio/library/export/xr0_30_09"
DATASET_ROOT = "/home/devuser/dlyakhov/datasets/Put-different-balls-to-the-box"
EPISODE = 3
SEED = 0

IMAGE_KEY_VIEW_MAP = {
    "images.top_camera": "ego",
    "images.pov_black_follower_camera": "wrist_left",
}


def _report(name: str, a: np.ndarray, b: np.ndarray) -> None:
    """Print a shape/value diff between the training and runtime tensors."""
    print(f"\n[{name}] train {a.shape} {a.dtype} | runtime {b.shape} {b.dtype}")
    if a.shape != b.shape:
        print("  SHAPE MISMATCH")
        return
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    print(f"  max_abs {diff.max():.6g}  mean_abs {diff.mean():.6g}")


def main() -> None:
    """Run one frame through both preprocessors and diff their outputs."""
    from physicalai.data import LeRobotDataModule

    dm = LeRobotDataModule(
        root=DATASET_ROOT,
        train_batch_size=1,
        episodes=[EPISODE],
        val_split=0.0,
        data_format="physicalai",
    )
    dm.setup("fit")
    torch.manual_seed(SEED)
    batch = next(iter(dm.train_dataloader()))

    # Training path: HuggingFace processor + chat template.
    train_preproc = XR0Preprocessor(image_key_view_map=IMAGE_KEY_VIEW_MAP, normalize_state=False)
    train_batch = {STATE: batch.state, TASK: batch.task}
    train_batch.update({f"{IMAGES}.{view}": img for view, img in batch.images.items()})
    train_out = train_preproc(train_batch)

    # Deploy path: NumPy preprocessor + OpenVINO tokenizer from the exported manifest.
    observation: dict[str, object] = {
        IMAGES: {view: img[0].cpu().numpy() for view, img in batch.images.items()},
        STATE: batch.state[0].cpu().numpy(),
        TASK: batch.task,
    }
    model = InferenceModel(MODEL_PATH, device="CPU")
    runtime_out: dict[str, object] = observation
    for preprocessor in model.preprocessors:
        runtime_out = preprocessor(runtime_out)

    train_ids = train_out["input_ids"][0].cpu().numpy()
    runtime_ids = np.asarray(runtime_out["tokenized_prompt"]).reshape(-1)
    # The exported tokenizer right-pads to the graph length; compare valid tokens.
    valid = int(np.asarray(runtime_out["tokenized_prompt_mask"]).reshape(-1).sum())
    print(f"input_ids: train len {train_ids.size}, runtime valid len {valid}")
    common = min(train_ids.size, valid)
    mismatch = np.nonzero(train_ids[:common] != runtime_ids[:common])[0]
    if mismatch.size:
        first = int(mismatch[0])
        print(f"  FIRST TOKEN MISMATCH at index {first}: train {train_ids[first]} vs runtime {runtime_ids[first]}")
        print(f"  total mismatched tokens: {mismatch.size}")
    else:
        print("  tokens identical over the common prefix")

    _report("pixel_values", train_out["pixel_values"].cpu().numpy(), np.asarray(runtime_out["pixel_values"]))
    _report("state", train_out["state"].cpu().numpy(), np.asarray(runtime_out["state"]))

    # The training resize is antialiased bicubic (PIL-like); the runtime uses
    # cv2.INTER_CUBIC, which does not prefilter. Re-run the deploy preprocessor
    # with INTER_AREA (which does) to see how much of the pixel gap that closes.
    import cv2
    from physicalai.inference.preprocessors import xr0 as runtime_xr0

    original_resize = runtime_xr0._resize_image  # noqa: SLF001

    def _area_resize(image: np.ndarray, factor: int, max_pixels: int) -> np.ndarray:
        resized = original_resize(image, factor=factor, max_pixels=max_pixels)
        return cv2.resize(image, (resized.shape[1], resized.shape[0]), interpolation=cv2.INTER_AREA)

    runtime_xr0._resize_image = _area_resize  # noqa: SLF001
    try:
        area_out: dict[str, object] = observation
        for preprocessor in model.preprocessors:
            area_out = preprocessor(area_out)
    finally:
        runtime_xr0._resize_image = original_resize  # noqa: SLF001
    _report("pixel_values (INTER_AREA)", train_out["pixel_values"].cpu().numpy(), np.asarray(area_out["pixel_values"]))


if __name__ == "__main__":
    main()
