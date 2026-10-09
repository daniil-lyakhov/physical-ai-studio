# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Train / fine-tune the XR0 policy on a local SO-101 LeRobot dataset.

Fine-tunes XR0 (Qwen3-VL-4B backbone + 16-layer DiT action expert) from the
Xiaomi pretrained checkpoint on a single-task SO-101 dataset of roughly
150 minutes at 30 fps (~270k frames), with 6-DoF joint-target actions and a
6-DoF proprioceptive state.

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
from physicalai.policies.xr0.pretrained_utils import compute_action_chunk_stats
from physicalai.train import Trainer
from physicalai.train.utils import reformat_dataset_to_match_policy

# Local LeRobot dataset (the folder that contains meta/, data/, videos/).
DATASET_ROOT = Path("/home/devuser/dlyakhov/datasets/Put-different-balls-to-the-box")

# Pretrained XR0 checkpoint to fine-tune from.
CHECKPOINT = "XiaomiRobotics/Xiaomi-Robotics-0-Pretrain"

# Smoke test: overfit a few batches for a handful of steps to confirm the
# training loop runs end-to-end and the loss decreases before committing to a
# full run. Flip to False for real fine-tuning.
SMOKE_TEST = False

# Training hyperparameters. XR0 is far larger than ACT, so keep the micro-batch
# small and recover the effective batch through gradient accumulation:
#   8 samples/forward x 4 accumulation steps = effective batch 32.
# The small micro-batch is what bounds activation memory, which matters here
# because the fp32 weights below already roughly double the parameter and
# gradient footprint.
BATCH_SIZE = 8
ACCUMULATE_GRAD_BATCHES = 4

# Schedule sizing: ~150 min x 60 s x 30 fps ~= 270k frames, so one epoch is
# 270k / 32 ~= 8.4k optimizer steps. 30k steps is therefore ~3.6 epochs, a
# reasonable budget for fine-tuning a 4.7B VLA on a single task (and the same
# horizon the original XR0 recipe uses). Raise it if val/loss is still falling
# at the end; best-checkpoint selection on a held-out split makes that safe.
MAX_STEPS = 300 if SMOKE_TEST else 30_000
WARMUP_STEPS = 20 if SMOKE_TEST else 2_000

# Action-chunk geometry. The flow head always predicts ``chunk_size`` timesteps;
# only the first ``N_ACTION_STEPS`` are executed before replanning. These MUST
# differ, otherwise ``extra_export_args`` emits no ``action_chunk_trimmer`` and
# the exported policy runs a full 1 s chunk open-loop on the robot.
# At 30 fps: 30 predicted = 1.0 s, 10 executed = 0.33 s between replans.
CHUNK_SIZE = 30
N_ACTION_STEPS = 10

# The local SO-101 arm exposes 6 joint targets, which sizes the per-timestep
# action statistics computed before training.
ACTION_DIM = 6

# Cap the delta-stats pass. The train loader shuffles, so this is a random
# sample: 3000 x 8 = 24k chunks is far more than enough for a per-timestep
# mean/std, and avoids decoding every video frame in the dataset an extra time.
STATS_MAX_BATCHES = 8 if SMOKE_TEST else 3_000

# An epoch is ~8.4k optimizer steps, so epoch-granular validation would run only
# ~3 times in the whole job. Validate on a step cadence instead (~30 times
# total), capped to a fixed number of batches so the eval loop does not
# dominate.
#
# NB: Lightning counts ``val_check_interval`` in *dataloader batches*, not
# optimizer steps, so it has to be scaled by the accumulation factor.
VAL_CHECK_INTERVAL_STEPS = 1_000
VAL_CHECK_INTERVAL = VAL_CHECK_INTERVAL_STEPS * ACCUMULATE_GRAD_BATCHES
LIMIT_TRAIN_BATCHES = 4 if SMOKE_TEST else None
LIMIT_VAL_BATCHES = 2 if SMOKE_TEST else 100
LOG_EVERY_N_STEPS = 1 if SMOKE_TEST else 50

# Single episode to overfit during the smoke test.
OVERFIT_EPISODE = 3

# Checkpoints are tens of GB at this model size, so keep them off the repo
# volume. `Trainer` also defaults `default_root_dir` to a relative
# "experiments", hence passing EXPERIMENTS_ROOT explicitly below so the
# TensorBoard logs land next to the checkpoints.
EXPERIMENTS_ROOT = Path("/mnt/data/experiments")
EXPERIMENT_NAME = "xr0_so101_put_balls"
EXPERIMENT_DIR = EXPERIMENTS_ROOT / EXPERIMENT_NAME

# The dataset's camera keys are rig-specific and mean nothing to the pretrained
# checkpoint, whose prompt is built from the canonical view names "ego",
# "base", "wrist_left" and "wrist_right". Mapping them makes the fine-tuning
# prompt read "# Ego View" / "# Left-Wrist View", exactly like the data
# Xiaomi-Robotics-0-Pretrain was trained on.
IMAGE_KEY_VIEW_MAP = {
    "images.top_camera": "ego",
    "images.pov_black_follower_camera": "wrist_left",
}


def main() -> None:
    """Run XR0 fine-tuning on the local dataset."""
    datamodule = LeRobotDataModule(
        # `repo_id` is only used as a name when `root` points at a local dataset.
        root=str(DATASET_ROOT),
        train_batch_size=BATCH_SIZE,
        episodes=[OVERFIT_EPISODE] if SMOKE_TEST else None,
        # 5% of ~270k frames is ~13.5k held-out frames, plenty to pick a
        # checkpoint without spending a tenth of the data on it.
        val_split=0.0 if SMOKE_TEST else 0.05,
        val_split_seed=42,
        data_format="physicalai",
    )

    # Fine-tune from the pretrained base checkpoint. `sdpa` avoids the hard
    # dependency on flash-attention; gradient checkpointing keeps memory in check.
    # `MEAN_STD` matches the state/action normalization used by the original
    # Xiaomi XR0 training pipeline. `normalize_state=False` keeps the raw
    # proprioceptive state that the delta postprocessor re-adds at
    # inference/export time.
    policy = XR0(
        pretrained_name_or_path=CHECKPOINT,
        vlm_attn_implementation="sdpa",
        gradient_checkpointing=True,
        # Keep parameters in fp32. The default "bfloat16" stores weights in
        # bf16, and Lightning's bf16-mixed autocasts operations without
        # creating fp32 master weights, so any AdamW update smaller than half a
        # bf16 ULP rounds away entirely. Measured on a bf16 run here: every
        # tensor with magnitude >= 1.0 came out 100% bit-identical, and
        # corr(log10 magnitude, unchanged_fraction) was +0.92. In fp32 that
        # correlation drops to ~0.00.
        #
        # This roughly doubles weight + gradient memory. If it does not fit,
        # the alternative is dtype="bfloat16" with the default
        # optimizer_type="adamw4bit", whose torchao bf16_stochastic_round makes
        # sub-ULP updates land probabilistically instead of being discarded.
        dtype="float32",
        chunk_size=CHUNK_SIZE,
        n_action_steps=N_ACTION_STEPS,
        normalization_mode="MEAN_STD",
        normalize_state=False,
        augment_images=True,
        # SO-101 actions are absolute joint targets in the same units as the
        # proprioceptive state, so `action[t] - state` is a well-defined
        # residual and matches the pretrained flow head's delta prior. (This
        # would not hold for an embodiment whose actions are end-effector
        # velocity commands, where the subtraction mixes incompatible spaces.)
        action_mode="delta",
        image_key_view_map=IMAGE_KEY_VIEW_MAP,
        # Fine-tuning, not pretraining: half the config's default LR and a
        # tenth of its weight decay, so the pretrained features get nudged
        # rather than overwritten.
        optimizer_lr=5e-5,
        optimizer_weight_decay=0.01,
        # Decay over the whole run; cutting the cosine short (or long) leaves
        # the final steps at the wrong LR.
        scheduler_warmup_steps=WARMUP_STEPS,
        scheduler_decay_steps=MAX_STEPS,
    )

    # Estimate the per-timestep action mean/std from the fine-tuning data.
    # The pretrained model was built eagerly above, so its action delta indices
    # are available to configure the dataset's action chunking; stats are then
    # accumulated over the chunked targets and installed into the normalization
    # used by the pre/post-processors during `fit`.
    datamodule.setup("fit")
    reformat_dataset_to_match_policy(policy, datamodule)
    action_mean, action_std = compute_action_chunk_stats(
        datamodule,
        chunk_size=policy.config.chunk_size,
        action_dim=ACTION_DIM,
        max_action_dim=policy.config.max_action_dim,
        action_mode=policy.config.action_mode,
        max_batches=STATS_MAX_BATCHES,
        setup_stage=None,
    )
    policy.set_action_stats(action_mean, action_std)

    checkpoint_callback = ModelCheckpoint(
        dirpath=EXPERIMENT_DIR / "checkpoints",
        filename="xr0-{step:06d}",
        # Written at every validation, i.e. every VAL_CHECK_INTERVAL_STEPS
        # optimizer steps. Monitoring ``step`` with ``mode="max"`` makes
        # ``save_top_k`` a rolling window over the most recent checkpoints
        # rather than a best-of selection, so only the newest 3 survive
        # (plus ``last.ckpt`` for resuming). ``val/loss`` is still logged for
        # the training curves, it just no longer drives retention.
        save_top_k=1 if SMOKE_TEST else 3,
        save_last=True,
        monitor="step",
        mode="max",
    )

    trainer = Trainer(
        experiment_name=EXPERIMENT_NAME,
        default_root_dir=str(EXPERIMENTS_ROOT),
        max_steps=MAX_STEPS,
        accelerator="gpu",
        devices=[3],
        precision="bf16-mixed",
        # Speeds up the matmuls that stay in fp32 under bf16-mixed.
        allow_tf32=True,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        callbacks=[checkpoint_callback],
        log_every_n_steps=LOG_EVERY_N_STEPS,
        val_check_interval=None if SMOKE_TEST else VAL_CHECK_INTERVAL,
        limit_train_batches=LIMIT_TRAIN_BATCHES,
        limit_val_batches=LIMIT_VAL_BATCHES,
    )

    trainer.fit(model=policy, datamodule=datamodule)


if __name__ == "__main__":
    main()
