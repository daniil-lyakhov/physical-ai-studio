# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Train / fine-tune the XR0 policy on a local LeRobot dataset.

Fine-tunes XR0 (Qwen3-VL-4B backbone + DiT action expert) starting from the
pretrained LIBERO checkpoint on the local dataset at
``/home/dlyakhov/datasets/Put-the-yellow-ball-to-the-black-box``.

XR0 is a large VLA model, so this uses a small batch size, gradient
checkpointing, and bf16 mixed precision to fit on a single GPU.

Usage:
    python train_local_xr0.py
"""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint

from physicalai.data import LeRobotDataModule
from physicalai.policies import XR0
from physicalai.train import Trainer

# Local LeRobot dataset (the folder that contains meta/, data/, videos/).
DATASET_ROOT = Path("/home/dlyakhov/datasets/Put-the-yellow-ball-to-the-black-box")

# Pretrained XR0 checkpoint to fine-tune from.
CHECKPOINT = "XiaomiRobotics/Xiaomi-Robotics-0-LIBERO"

# Training hyperparameters. XR0 is far larger than ACT, so keep the batch small.
MAX_STEPS = 40_000
BATCH_SIZE = 1

EXPERIMENT_DIR = Path("experiments") / "xr0_Put-the-yellow-ball-to-the-black-box"


def main() -> None:
    """Run XR0 fine-tuning on the local dataset."""

    datamodule = LeRobotDataModule(
        # `repo_id` is only used as a name when `root` points at a local dataset.
        root=str(DATASET_ROOT),
        train_batch_size=BATCH_SIZE,
        val_split=0.1,
        data_format="physicalai",
    )

    # Fine-tune from the pretrained LIBERO checkpoint. `sdpa` avoids the hard
    # dependency on flash-attention; gradient checkpointing keeps memory in check.
    # `MEAN_STD` matches the state/action normalization used by the original
    # Xiaomi XR0 training pipeline. `normalize_state=True` normalizes the raw
    # proprioceptive state (this dataset stores joint positions in degrees, which
    # is off the scale the pretrained checkpoint expects) so the DiT conditioning
    # stays well-scaled during fine-tuning.
    policy = XR0(
        pretrained_name_or_path=CHECKPOINT,
        vlm_attn_implementation="sdpa",
        gradient_checkpointing=True,
        normalization_mode="MEAN_STD",
        normalize_state=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=EXPERIMENT_DIR / "checkpoints",
        filename="xr0-{step:06d}",
        save_top_k=1,
        save_last=True,
        monitor="val/loss",
        mode="min",
        every_n_train_steps=20_000,
    )

    trainer = Trainer(
        max_steps=MAX_STEPS,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        callbacks=[checkpoint_callback],
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
    )

    trainer.fit(model=policy, datamodule=datamodule)


if __name__ == "__main__":
    main()
