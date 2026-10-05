# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fine-tune XR0 on LIBERO-10 gym task 1 from local LeRobot demonstrations.

Run on the training server with a local dataset directory and fine-tune the
Xiaomi-Robotics-0-Pretrain base checkpoint from Hugging Face. XR0 loads its
default Qwen3-VL backbone without a separate VLM path.
The lerobot/libero_10 dataset has two image views (image/wrist_image), an
8-D state, a 7-D action, and LIBERO task instructions. The dataset is never
downloaded by this script. The gym emits image2 instead of wrist_image;
rename that gym view before evaluating this checkpoint.
Gym task 1 is "put both the cream cheese box and the butter in the basket";
its LeRobot task_index is not the same as the gym task ID.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from lerobot.datasets.io_utils import load_episodes
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from physicalai.data import FeatureType, LeRobotDataModule
from physicalai.policies import XR0
from physicalai.policies.xr0.pretrained_utils import compute_action_chunk_stats
from physicalai.train import Trainer
from physicalai.train.utils import reformat_dataset_to_match_policy

_LIBERO_ACTION_DIM = 7
_LIBERO_STATE_DIM = 8
_TASK_INSTRUCTION = "put both the cream cheese box and the butter in the basket"
CHECKPOINT = "XiaomiRobotics/Xiaomi-Robotics-0-Pretrain"


def _task_episodes(data_root: Path) -> list[int]:
    """Select only episodes for LIBERO-10 gym task 1 from local metadata.

    Returns:
        Sorted episode indices for the selected task.

    Raises:
        ValueError: If an episode mixes tasks or there are too few episodes to split.
    """
    selected = []
    for episode in load_episodes(data_root):
        tasks = {str(task).strip().rstrip(".").casefold() for task in episode["tasks"]}
        if _TASK_INSTRUCTION not in tasks:
            continue
        if tasks != {_TASK_INSTRUCTION}:
            msg = "LIBERO-10 task 1 shares an episode with other tasks; episode filtering cannot isolate its frames"
            raise ValueError(msg)
        selected.append(int(episode["episode_index"]))

    if len(selected) <= 1:
        msg = f"Need at least two LIBERO-10 task 1 episodes for train/validation, found {len(selected)}"
        raise ValueError(msg)
    return sorted(selected)


def main() -> None:
    """Fine-tune on demonstrations, selecting the checkpoint with the best held-out loss.

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
        episodes=_task_episodes(data_root),
        data_format="physicalai",
        train_batch_size=16,
        val_split=0.1,
        val_split_seed=42,
        num_workers=0,
    )
    policy = XR0(
        pretrained_name_or_path=CHECKPOINT,
        vlm_attn_implementation="sdpa",
        n_action_steps=10,
        image_key_view_map={"images.image": "ego", "images.wrist_image": "wrist_right"},
        normalization_mode="MEAN_STD",
        normalize_state=False,
        action_mode="delta",
        augment_images=True,
        freeze_vision_encoder=True,
        optimizer_lr=2.5e-5,
        scheduler_warmup_steps=500,
        scheduler_decay_steps=None,
    )

    # LeRobot's frame-level action stats cannot normalize XR0's 30-step chunks.
    datamodule.setup("fit")
    reformat_dataset_to_match_policy(policy, datamodule)
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
    mean, std = compute_action_chunk_stats(
        datamodule,
        chunk_size=policy.config.chunk_size,
        action_dim=_LIBERO_ACTION_DIM,
        max_action_dim=policy.config.max_action_dim,
        action_mode=policy.config.action_mode,
        setup_stage=None,
        device="cpu",
    )
    policy.set_action_stats(mean, std)

    trainer = Trainer(
        experiment_name="xr0_libero_task_1",
        max_steps=10_000,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        accumulate_grad_batches=1,
        val_check_interval=min(1_000, len(datamodule.train_dataloader())),
        check_val_every_n_epoch=None,
        callbacks=[
            ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, save_last=True),
            EarlyStopping(monitor="val/loss", mode="min", patience=5),
        ],
    )
    trainer.fit(model=policy, datamodule=datamodule)


if __name__ == "__main__":
    main()
