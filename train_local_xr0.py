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
from physicalai.policies.xr0 import compute_delta_action_stats
from physicalai.train import Trainer
from physicalai.train.utils import reformat_dataset_to_match_policy

# Local LeRobot dataset (the folder that contains meta/, data/, videos/).
DATASET_ROOT = Path("/home/dlyakhov/datasets/Put-the-yellow-ball-to-the-black-box")

# Pretrained XR0 checkpoint to fine-tune from.
CHECKPOINT = "XiaomiRobotics/Xiaomi-Robotics-0-Pretrain"

# Training hyperparameters. XR0 is far larger than ACT, so keep the batch small.
MAX_STEPS = 40_000
BATCH_SIZE = 16

# Action-chunk geometry. XR0 predicts a fixed 30-step chunk; the local SO-101
# arm exposes 6 joint targets. Both are hardcoded here to compute the
# per-timestep delta-action statistics before training.
CHUNK_SIZE = 30
ACTION_DIM = 6

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
    # Xiaomi XR0 training pipeline. The pretrained flow head is *delta-native*
    # (it predicts `action[t] - state`), so `action_mode="delta"` trains in that
    # same space instead of fighting the prior with absolute joint targets.
    # `normalize_state=False` keeps the raw proprioceptive state that the delta
    # postprocessor re-adds at inference/export time.
    policy = XR0(
        pretrained_name_or_path=CHECKPOINT,
        vlm_attn_implementation="sdpa",
        gradient_checkpointing=True,
        normalization_mode="MEAN_STD",
        normalize_state=False,
        action_mode="delta",
        # Align the cosine decay horizon with the full training length (the
        # config default decays over 30k steps, which under-decays a 40k run).
        scheduler_decay_steps=MAX_STEPS,
    )

    # Estimate the per-timestep delta-action mean/std from the fine-tuning data.
    # The pretrained model was built eagerly above, so its action delta indices
    # are available to configure the dataset's action chunking; stats are then
    # accumulated over the chunked deltas and fed back into the delta
    # normalization used by the pre/post-processors during `fit`.
    datamodule.setup("fit")
    reformat_dataset_to_match_policy(policy, datamodule)
    delta_mean, delta_std = compute_delta_action_stats(
        datamodule,
        chunk_size=CHUNK_SIZE,
        action_dim=ACTION_DIM,
        max_action_dim=policy.config.max_action_dim,
        setup_stage=None,
    )
    policy._action_delta_mean = delta_mean  # noqa: SLF001
    policy._action_delta_std = delta_std  # noqa: SLF001
    policy.hparams["action_delta_mean"] = delta_mean.tolist()
    policy.hparams["action_delta_std"] = delta_std.tolist()

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
