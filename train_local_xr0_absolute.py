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
from physicalai.train.utils import reformat_dataset_to_match_policy

# Local LeRobot dataset (the folder that contains meta/, data/, videos/).
DATASET_ROOT = Path("/home/dlyakhov/datasets/Put-the-yellow-ball-to-the-black-box")

# Pretrained XR0 checkpoint to fine-tune from.
CHECKPOINT = "XiaomiRobotics/Xiaomi-Robotics-0-Pretrain"

# Smoke test: overfit a few batches for a handful of steps to confirm the
# training loop runs end-to-end and the loss decreases before committing to a
# full run. Flip to False for real fine-tuning.
SMOKE_TEST = True

# Training hyperparameters. XR0 is far larger than ACT, so keep the batch small.
MAX_STEPS = 300 if SMOKE_TEST else 40_000
BATCH_SIZE = 16 if SMOKE_TEST else 16
WARMUP_STEPS = 20 if SMOKE_TEST else 2_000

# Smoke-test knobs: cap the train/val loop to a few batches so the sanity run
# stays fast without iterating the whole dataset. ``None`` means "use
# everything" for a full run.
LIMIT_TRAIN_BATCHES = 4 if SMOKE_TEST else None
LIMIT_VAL_BATCHES = 2 if SMOKE_TEST else None
LOG_EVERY_N_STEPS = 1 if SMOKE_TEST else 50

# Single episode to overfit during the smoke test.
OVERFIT_EPISODE = 3

EXPERIMENT_DIR = Path("experiments") / "xr0_Put-the-yellow-ball-to-the-black-box_absolute"


def main() -> None:
    """Run XR0 fine-tuning on the local dataset."""

    datamodule = LeRobotDataModule(
        # `repo_id` is only used as a name when `root` points at a local dataset.
        root=str(DATASET_ROOT),
        train_batch_size=BATCH_SIZE,
        episodes=[OVERFIT_EPISODE] if SMOKE_TEST else None,
        val_split=0.0 if SMOKE_TEST else 0.1,
        data_format="physicalai",
    )

    # Fine-tune from the pretrained LIBERO checkpoint. `sdpa` avoids the hard
    # dependency on flash-attention; gradient checkpointing keeps memory in check.
    # This is the *absolute*-action counterpart of ``train_local_xr0.py`` (which
    # trains in the delta space). `action_mode="absolute"` regresses the raw
    # absolute joint targets, which *contradicts* the delta-native pretrained
    # flow head, so `normalize_state=True` is required to keep the DiT
    # conditioning well-scaled (absolute + raw state exploded the loss to the
    # millions in earlier runs). `enable_freq=False` matches the delta run so the
    # only deliberate difference is the action space. `optimizer_lr=5e-5` is a
    # little higher than the delta run's 2.5e-5 because absolute targets fight the
    # pretrained prior and need a slightly larger step to adapt.
    policy = XR0(
        pretrained_name_or_path=CHECKPOINT,
        vlm_attn_implementation="sdpa",
        gradient_checkpointing=True,
        normalization_mode="MEAN_STD",
        normalize_state=True,
        action_mode="absolute",
        enable_freq=False,
        optimizer_lr=5e-5,
        # Align the cosine decay horizon with the full training length (the
        # config default decays over 30k steps, which under-decays a 40k run).
        scheduler_decay_steps=MAX_STEPS,
        scheduler_warmup_steps=WARMUP_STEPS,
    )

    # Absolute mode uses the dataset's absolute action stats (computed during
    # ``fit``); only the action-chunk indexing needs to be aligned to the policy
    # here so the dataset yields 30-step chunks. No delta-action stats are needed.
    datamodule.setup("fit")
    reformat_dataset_to_match_policy(policy, datamodule)

    checkpoint_callback = ModelCheckpoint(
        dirpath=EXPERIMENT_DIR / "checkpoints",
        filename="xr0-{step:06d}",
        save_top_k=1,
        save_last=True,
        monitor=None if SMOKE_TEST else "val/loss",
        mode="min",
        every_n_train_steps=20_000,
    )

    trainer = Trainer(
        max_steps=MAX_STEPS,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        callbacks=[checkpoint_callback],
        log_every_n_steps=LOG_EVERY_N_STEPS,
        check_val_every_n_epoch=1,
        limit_train_batches=LIMIT_TRAIN_BATCHES,
        limit_val_batches=LIMIT_VAL_BATCHES,
    )

    trainer.fit(model=policy, datamodule=datamodule)


if __name__ == "__main__":
    main()
