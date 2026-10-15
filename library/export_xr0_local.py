# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export an XR0 checkpoint trained by ``train_local_xr0.py`` to OpenVINO.

Lightning restores the training configuration, dataset and action-chunk stats,
and model weights from the checkpoint. Override only the execution horizon.
Writes an OpenVINO IR plus ``manifest.json`` for ``InferenceModel(...)``.

Usage:
    python export_xr0_local.py
"""

from __future__ import annotations

from pathlib import Path

from physicalai.policies import XR0

CHECKPOINT = Path("experiments") / "Put_different_box_09_26_long" / "checkpoints" / "last.ckpt"
EXPORT_DIR = Path("export") / "xr0_put_different_box_09_26_long"

# Execute only 10 of the 30 predicted actions before replanning. Keeping
# ``chunk_size (30) != n_action_steps (10)`` makes the export emit an
# ``action_chunk_trimmer`` into the manifest; running all 30 open-loop overshoots.
N_ACTION_STEPS = 10


def main() -> None:
    """Export the fine-tuned XR0 checkpoint to OpenVINO."""
    # Lightning checkpoints contain Feature objects; only load checkpoints
    # produced locally by a trusted training run, never an untrusted file.
    policy = XR0.load_from_checkpoint(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
        n_action_steps=N_ACTION_STEPS,
    )

    policy.to_openvino(str(EXPORT_DIR))


if __name__ == "__main__":
    main()
