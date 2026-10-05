# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Evaluate an XR0 checkpoint from train_libero_xr0.py in LIBERO-10.

Run from library/: uv run --no-sync python eval_libero_xr0.py \
    --checkpoint experiments/xr0_libero/version_0/checkpoints/last.ckpt
Defaults to one episode of task 0; pass --all-tasks --episodes 20 for the full suite.
Install the optional xr0 and libero extras on the evaluation server.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import torch

from physicalai.benchmark.gyms import LiberoBenchmark
from physicalai.data.observation import Feature, FeatureType, NormalizationParameters
from physicalai.devices import get_available_device
from physicalai.policies import XR0

_LIBERO_10_TASKS = 10


def main() -> None:
    """Load the trained policy and evaluate it with LIBERO-10 observations."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local Lightning .ckpt from train_libero_xr0.py")
    parser.add_argument("--task-id", type=int, default=0, help="LIBERO-10 task index (0-9); default: 0")
    parser.add_argument("--all-tasks", action="store_true", help="Evaluate all ten LIBERO-10 tasks")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per task; default: 1")
    parser.add_argument("--output-dir", type=Path, default=Path("results/xr0_libero_10"))
    parser.add_argument("--video-dir", type=Path, help="Record rollout videos here if specified")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.suffix != ".ckpt":
        parser.error(f"Expected an existing .ckpt file: {checkpoint}")
    if not 0 <= args.task_id < _LIBERO_10_TASKS:
        parser.error("--task-id must be between 0 and 9")
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    # Training uses wrist_image, while LiberoGym calls the same camera image2.
    # Override only the observation-key mapping, not the learned weights/stats.
    view_map = {"images.image": "ego", "images.image2": "wrist_right"}
    # Lightning saves first-party Feature objects in checkpoint hyperparameters.
    # Allowlist those classes instead of enabling unrestricted pickle loading.
    with torch.serialization.safe_globals([Feature, FeatureType, NormalizationParameters]):
        policy = XR0.load_from_checkpoint(
            str(checkpoint), map_location="cpu", weights_only=True, image_key_view_map=view_map,
        )
    policy.to(get_available_device()).eval()

    benchmark = LiberoBenchmark(
        task_suite="libero_10",
        task_ids=None if args.all_tasks else [args.task_id],
        num_episodes=args.episodes,
        video_dir=args.video_dir,
        record_mode="all" if args.video_dir else "none",
    )
    try:
        results = benchmark.evaluate(policy, continue_on_error=False)
    finally:
        for gym in benchmark.gyms:
            gym.close()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_json(output_dir / "results.json")
    results.to_csv(output_dir / "results.csv")
    print(results.summary())  # noqa: T201


if __name__ == "__main__":
    main()
