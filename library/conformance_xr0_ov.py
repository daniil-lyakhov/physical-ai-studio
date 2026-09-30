# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Conformance check for the *exported* XR0 OpenVINO IR.

Same idea as ``conformance_xr0.py`` (run one dataset frame, compare predicted vs
target actions) but instead of the Torch checkpoint it drives the exported
OpenVINO IR through the Runtime ``InferenceModel`` -- i.e. the deploy path from
``local_inf.py``. The IR's own preprocessor / OV tokenizer / delta postprocessor
run inside ``predict_action_chunk``, so we only feed it a raw observation.
"""

from __future__ import annotations

import numpy as np
import torch
from lerobot.datasets.feature_utils import check_delta_timestamps, get_delta_indices

from physicalai.data import LeRobotDataModule
from physicalai.inference import InferenceModel
from physicalai.inference.constants import IMAGES, STATE, TASK

MODEL_PATH = "/home/devuser/dlyakhov/physical-ai-studio/xr0_put_balls_to_box_irs_long_train"
MODEL_PATH = "/home/devuser/dlyakhov/physical-ai-studio/xr0_put_balls_to_box_irs_long_train_correct_export"
MODEL_PATH = "/home/devuser/dlyakhov/physical-ai-studio/library/export/xr0_30_09"

DATASET_ROOT = "/home/devuser/dlyakhov/datasets/Put-different-balls-to-the-box"
EPISODE = 3
ACTION_DIM = 6
# Predicted action-chunk length the IR emits (XR0 action_shape[-2]); used to
# expand the dataset target into a full chunk without loading the checkpoint.
CHUNK_SIZE = 30
DEVICE = "CPU"  # "GPU" + bf16 mirrors deployment; "CPU" is portable
PRECISION_HINT = "f32"
# The runtime preprocessor downscales with cv2.INTER_CUBIC (no prefilter) while
# training used antialiased bicubic, so the deployed model sees aliased images.
# INTER_AREA prefilters on downscale; flip this to measure the action impact.
USE_AREA_RESIZE = True
# Must match conformance_xr0.py so both scripts select the same shuffled frame
# (identical reference target) for a fair comparison.
SEED = 0


def _set_action_chunk(dm: LeRobotDataModule, chunk_size: int) -> None:
    """Expand the dataset ``action`` feature into a ``chunk_size`` chunk.

    Mirrors ``reformat_dataset_to_match_policy`` for the action key only
    (``action_delta_indices == range(chunk_size)`` for XR0) so the target is a
    full chunk -- without needing to load the multi-GB checkpoint just to read
    that one property.
    """
    dataset = dm.train_dataset
    fps = dataset.fps
    delta_timestamps = {"action": [i / fps for i in range(chunk_size)]}
    check_delta_timestamps(delta_timestamps, fps, dataset.tolerance_s)
    dataset.delta_indices = get_delta_indices(delta_timestamps, fps)


def _patch_area_resize() -> None:
    """Swap the runtime preprocessor's cv2.INTER_CUBIC downscale for INTER_AREA."""
    import cv2
    from physicalai.inference.preprocessors import xr0 as runtime_xr0

    original_resize = runtime_xr0._resize_image  # noqa: SLF001

    def _area_resize(image, factor, max_pixels):  # noqa: ANN001, ANN202
        resized = original_resize(image, factor=factor, max_pixels=max_pixels)
        return cv2.resize(image, (resized.shape[1], resized.shape[0]), interpolation=cv2.INTER_AREA)

    runtime_xr0._resize_image = _area_resize  # noqa: SLF001


def main() -> None:
    dm = LeRobotDataModule(
        root=DATASET_ROOT,
        train_batch_size=1,
        episodes=[EPISODE],
        val_split=0.0,
        data_format="physicalai",
    )
    dm.setup("fit")
    # Expand the action target into a full chunk directly (no checkpoint needed;
    # inference is the OV IR, not a Torch policy).
    _set_action_chunk(dm, CHUNK_SIZE)

    # Seed the shuffled sampler so we pull the same frame as conformance_xr0.py.
    torch.manual_seed(SEED)
    batch = next(iter(dm.train_dataloader()))
    target = batch.action[..., :ACTION_DIM].reshape(-1, ACTION_DIM).cpu().numpy()

    observation = {
        IMAGES: {view: img[0].cpu().numpy() for view, img in batch.images.items()},
        STATE: batch.state[0].cpu().numpy(),
        TASK: batch.task,
    }

    if USE_AREA_RESIZE:
        _patch_area_resize()
    model = InferenceModel(MODEL_PATH, device=DEVICE, INFERENCE_PRECISION_HINT=PRECISION_HINT)
    pred = model.predict_action_chunk(observation)[..., :ACTION_DIM]

    err = np.abs(pred - target)
    print(f"pred shape {pred.shape}  target shape {target.shape}")
    print(f"per-dim MAE: {np.round(err.mean(0), 4)}")
    print(f"overall MAE: {err.mean():.4f}   max abs err: {err.max():.4f}")
    for t in (0, target.shape[0] // 2, target.shape[0] - 1):
        print(f"\nt={t}")
        print(f"  target: {np.round(target[t], 4)}")
        print(f"  pred  : {np.round(pred[t], 4)}")


if __name__ == "__main__":
    main()
