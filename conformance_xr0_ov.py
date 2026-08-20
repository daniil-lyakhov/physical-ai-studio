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

from physicalai.data import LeRobotDataModule
from physicalai.inference import InferenceModel
from physicalai.inference.constants import IMAGES, STATE, TASK
from physicalai.policies import XR0
from physicalai.train.utils import reformat_dataset_to_match_policy

CKPT = "experiments/xr0_Put-the-yellow-ball-to-the-black-box/checkpoints/last.ckpt"
MODEL_PATH = "xr0_export_with_ov_tok"
DATASET_ROOT = "/home/dlyakhov/datasets/Put-the-yellow-ball-to-the-black-box"
EPISODE = 3
ACTION_DIM = 6
DEVICE = "CPU"  # "GPU" + bf16 mirrors deployment; "CPU" is portable
PRECISION_HINT = "f32"
# Must match conformance_xr0.py so both scripts select the same shuffled frame
# (identical reference target) for a fair comparison.
SEED = 0


def main() -> None:
    dm = LeRobotDataModule(
        root=DATASET_ROOT,
        train_batch_size=1,
        episodes=[EPISODE],
        val_split=0.0,
        data_format="physicalai",
    )
    dm.setup("fit")
    # Load the checkpoint on CPU only to expand the action target into a full
    # chunk (reformat reads the policy's action_delta_indices); inference is the
    # OV IR, not this policy.
    policy = XR0.load_from_checkpoint(CKPT, dtype="float32", vlm_attn_implementation="sdpa", map_location="cpu")
    reformat_dataset_to_match_policy(policy, dm)

    # Seed the shuffled sampler so we pull the same frame as conformance_xr0.py.
    torch.manual_seed(SEED)
    batch = next(iter(dm.train_dataloader()))
    target = batch.action[..., :ACTION_DIM].reshape(-1, ACTION_DIM).cpu().numpy()

    observation = {
        IMAGES: {view: img[0].cpu().numpy() for view, img in batch.images.items()},
        STATE: batch.state[0].cpu().numpy(),
        TASK: batch.task,
    }

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
