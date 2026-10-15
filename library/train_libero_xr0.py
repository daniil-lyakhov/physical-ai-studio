# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fine-tune XR0 on the full local LIBERO-10 dataset.

Run on the training server with a local dataset directory and fine-tune the
Xiaomi-Robotics-0-Pretrain base checkpoint from Hugging Face. XR0 loads its
default Qwen3-VL backbone without a separate VLM path.
The lerobot/libero_10 dataset has two image views (image/wrist_image), an
8-D state, a 7-D action, and LIBERO task instructions. The dataset is never
downloaded by this script. The gym emits image2 instead of wrist_image;
rename that gym view before evaluating this checkpoint.
This run uses every episode and has no validation split. Compare checkpoints
on separate evaluation data: training loss alone cannot measure generalization.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint

from physicalai.data import FeatureType, LeRobotDataModule
from physicalai.policies import XR0
from physicalai.train import Trainer

_LIBERO_ACTION_DIM = 7
_LIBERO_STATE_DIM = 8
CHECKPOINT = "XiaomiRobotics/Xiaomi-Robotics-0-Pretrain"


def main() -> None:
    """Fine-tune on all LIBERO-10 demonstrations and save checkpoints.

    Raises:
        ValueError: If the dataset does not expose flat LIBERO actions.
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Local LeRobot dataset directory (contains meta/).",
    )
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        parser.error(f"Directory does not exist: {data_root}")

    datamodule = LeRobotDataModule(
        root=data_root,
        data_format="physicalai",
        train_batch_size=16,
        val_split=0.0,
        num_workers=0,
    )
    policy = XR0(
        pretrained_name_or_path=CHECKPOINT,
        vlm_attn_implementation="sdpa",
        # Keep parameters in fp32. The default "bfloat16" stores weights in bf16, and
        # Lightning's bf16-mixed autocasts operations without creating fp32 master
        # weights, so AdamW updates smaller than half a bf16 ULP round away entirely.
        dtype="float32",
        n_action_steps=10,
        image_key_view_map={"images.image": "ego", "images.wrist_image": "wrist_right"},
        normalization_mode="MEAN_STD",
        normalize_state=False,
        # MUST stay "absolute" for LIBERO. Delta mode trains on
        # `action - state[:min(action_dim, state_dim)]`, which only means
        # something when the action and state share a coordinate system. Here
        # they do not: LIBERO actions are *relative* OSC commands in [-1, 1]
        # (LiberoGym control_mode="relative"), while the 8-dim state is an
        # *absolute* eef pose -- position in metres (z ~ 1.0 on the table) plus
        # axis-angle orientation in radians (+-pi) plus gripper qpos.
        # Subtracting one from the other makes the regression target dominated
        # by -state rather than by the command. The round trip is still
        # self-consistent (inference re-adds state), but the model spends its
        # capacity echoing a negated state it already gets as an input, and a
        # small relative error on that dominant term lands as a large absolute
        # error in an action whose whole range is [-1, 1]. Symptom: rollouts
        # look correct for ~2 s near the fixed home pose, then go random, at
        # every checkpoint regardless of training length.
        # (Delta IS correct for SO-101, where actions are joint targets in the
        # same units as the joint-position state -- see train_local_xr0.py.)
        action_mode="absolute",
        augment_images=True,
        freeze_vision_encoder=False,
        optimizer_lr=5e-5,
        optimizer_weight_decay=0.01,
        scheduler_warmup_steps=1_200,
        scheduler_decay_steps=None,
    )

    # Validate the local LIBERO schema before loading video batches in fit().
    datamodule.setup("fit")
    observation_features = datamodule.train_dataset.observation_features
    image_keys = {key for key, feature in observation_features.items() if feature.ftype == FeatureType.VISUAL}
    state_feature = observation_features.get("state")
    if (
        image_keys != {"image", "wrist_image"}
        or state_feature is None
        or tuple(state_feature.shape) != (_LIBERO_STATE_DIM,)
    ):
        msg = "Expected LIBERO observations with image, wrist_image, and an 8-D state"
        raise ValueError(msg)
    action_feature = datamodule.train_dataset.action_features.get("action")
    if action_feature is None or tuple(action_feature.shape) != (_LIBERO_ACTION_DIM,):
        msg = "Expected a flat 7-D LIBERO action feature named 'action'"
        raise ValueError(msg)
    trainer = Trainer(
        experiment_name="xr0_libero_10_full",
        max_steps=24_000,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        accumulate_grad_batches=4,
        limit_val_batches=0,
        log_every_n_steps=10,
        callbacks=[
            ModelCheckpoint(
                monitor=None,
                save_top_k=-1,
                save_last=True,
                every_n_train_steps=5_000,
                filename="step-{step:06d}",
            ),
        ],
    )
    trainer.fit(model=policy, datamodule=datamodule)


if __name__ == "__main__":
    main()

