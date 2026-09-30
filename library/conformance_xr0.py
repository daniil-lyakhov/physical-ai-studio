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
CHECKPOINT = "/home/devuser/dlyakhov/physical-ai-studio/library/experiments/Put_different_box_10/checkpoints/last.ckpt"
N_ACTION_STEPS = 30

DATASET_ROOT = "/home/devuser/dlyakhov/datasets/Put-different-balls-to-the-box"
EPISODE = 3
ACTION_DIM = 6
# Shared with conformance_xr0_ov.py so both pick the *same* shuffled frame
# (same reference target) and the same rectified-flow noise -> fair 1-to-1.
SEED = 0


IMAGE_KEY_VIEW_MAP = {
    "images.top_camera": "ego",
    "images.pov_black_follower_camera": "wrist_left",
}
def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    datamodule = LeRobotDataModule(root=str(DATASET_ROOT), data_format="physicalai")
    datamodule.setup("fit")

    hparams = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["hyper_parameters"]

    policy = XR0(
        pretrained_name_or_path=str(CHECKPOINT),
        dataset_stats=datamodule.train_dataset.stats,
        vlm_attn_implementation="sdpa",
        normalization_mode="MEAN_STD",
        normalize_state=False,
        action_mode="delta",
        n_action_steps=N_ACTION_STEPS,
        image_key_view_map=IMAGE_KEY_VIEW_MAP,
        action_mean=hparams["action_mean"],
        action_std=hparams["action_std"],
    )
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

    # Seed the shuffled sampler so we always pull the same frame (identical to
    # conformance_xr0_ov.py).
    torch.manual_seed(SEED)
    batch = next(iter(dm.train_dataloader()))
    target = batch.action[..., :ACTION_DIM].reshape(-1, ACTION_DIM).cpu().numpy()

    with torch.no_grad():
        # Seed again so the flow noise (torch.randn inside _sample_noise) is
        # reproducible run-to-run.
        torch.manual_seed(SEED)
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
