# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export an XR0 checkpoint trained by ``train_local_xr0.py`` to OpenVINO.

Rebuilds the policy with the same configuration used for fine-tuning (delta
action mode, MEAN_STD normalization, raw state, camera view mapping), restores
the action-chunk statistics saved in the checkpoint hyper-parameters, and writes
an OpenVINO IR plus ``manifest.json`` that Runtime can load with
``InferenceModel(...)``.

Usage:
    python export_xr0_local.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from physicalai.data import LeRobotDataModule
from physicalai.policies import XR0

DATASET_ROOT = Path("/home/devuser/dlyakhov/datasets/Put-different-balls-to-the-box")
CHECKPOINT = Path("experiments") / "Put_different_box_09_26_long" / "checkpoints" / "last.ckpt"
EXPORT_DIR = Path("export") / "xr0_put_different_box_09_26_long"

# Execute only 10 of the 30 predicted actions before replanning. Keeping
# ``chunk_size (30) != n_action_steps (10)`` makes the export emit an
# ``action_chunk_trimmer`` into the manifest; running all 30 open-loop overshoots.
N_ACTION_STEPS = 10

# Must match ``IMAGE_KEY_VIEW_MAP`` in ``train_local_xr0.py``.
IMAGE_KEY_VIEW_MAP = {
    "images.top_camera": "ego",
    "images.pov_black_follower_camera": "wrist_left",
}


def main() -> None:
    """Export the fine-tuned XR0 checkpoint to OpenVINO."""
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

    policy.to_openvino(str(EXPORT_DIR))


if __name__ == "__main__":
    main()
