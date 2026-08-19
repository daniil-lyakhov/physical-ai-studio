# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Conformance check: run the fine-tuned XR0 checkpoint on one dataset frame and
compare predicted vs target actions (sanity that it's not noisy shaking)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from physicalai.data import LeRobotDataModule
from physicalai.policies import XR0
from physicalai.train.utils import reformat_dataset_to_match_policy

CKPT = "experiments/xr0_Put-the-yellow-ball-to-the-black-box/checkpoints/last.ckpt"
DATASET_ROOT = "/home/dlyakhov/datasets/Put-the-yellow-ball-to-the-black-box"
EPISODE = 3
ACTION_DIM = 6


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = XR0.load_from_checkpoint(CKPT, dtype="float32", vlm_attn_implementation="sdpa")
    policy.to(device).eval()

    dm = LeRobotDataModule(
        root=DATASET_ROOT,
        train_batch_size=1,
        episodes=[EPISODE],
        val_split=0.0,
        data_format="physicalai",
    )
    dm.setup("fit")
    reformat_dataset_to_match_policy(policy, dm)

    batch = next(iter(dm.train_dataloader()))
    target = batch.action[..., :ACTION_DIM].reshape(-1, ACTION_DIM).cpu().numpy()

    with torch.no_grad():
        pred = policy.predict_action_chunk(batch)
    pred = pred[..., :ACTION_DIM].reshape(-1, ACTION_DIM).cpu().numpy()

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
