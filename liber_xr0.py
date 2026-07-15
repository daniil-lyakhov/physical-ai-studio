#!/usr/bin/env python
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run LIBERO validation on the pretrained XR0 (Xiaomi-Robotics-0) policy.

This loads the released ``XiaomiRobotics/Xiaomi-Robotics-0-LIBERO`` checkpoint
into the framework-native :class:`physicalai.policies.xr0.XR0` policy and
evaluates it on one or more LIBERO task suites through
:class:`physicalai.benchmark.gyms.libero.LiberoBenchmark`.

Requirements (see ``library/pyproject.toml``):
    * base ``physicalai-train`` package
    * ``[libero]`` extra  -> hf-libero (LIBERO + robosuite + MuJoCo)
    * ``[smolvla]`` (or ``[pi0]``) extra -> pinned ``transformers`` for Qwen3-VL

Run with the LIBERO env python, e.g.::

    # quick smoke test (1 task, 2 episodes)
    env_libero/bin/python liber_xr0.py --task-suite libero_10 --task-ids 0 --num-episodes 2

    # full LIBERO-10 sweep
    env_libero/bin/python liber_xr0.py --task-suite libero_10 --num-episodes 20 \
        --video-dir ./videos --results ./xr0_libero_results.json
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch

logger = logging.getLogger("liber_xr0")

DEFAULT_CHECKPOINT = "XiaomiRobotics/Xiaomi-Robotics-0-LIBERO"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate the pretrained XR0 policy on LIBERO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Local path or HF repo id of the pretrained XR0 checkpoint.",
    )
    parser.add_argument(
        "--task-suite",
        default="libero_10",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO task suite to evaluate.",
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="*",
        default=[0],
        help="Task ids within the suite (space separated). Empty -> all tasks.",
    )
    parser.add_argument("--num-episodes", type=int, default=2, help="Episodes per task.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max steps per episode (default: LIBERO suite default).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed.")
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=10,
        help=(
            "Actions executed per predicted chunk before replanning (open-loop horizon). "
            "Matches the reference LIBERO eval's replan_steps=10; the full chunk is 30 but "
            "executing all 30 open-loop overshoots the target and collapses success to ~0%."
        ),
    )
    parser.add_argument(
        "--attn",
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="VLM attention backend. Use 'sdpa' unless flash-attn is installed.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float32"],
        help="Model dtype. 'auto' -> bfloat16 on CUDA, float32 on CPU.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device ('auto', 'cuda', 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--obs-size",
        type=int,
        default=256,
        help="Observation image height/width fed to the gym.",
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        help="Directory to save episode videos. None disables recording.",
    )
    parser.add_argument(
        "--record-mode",
        default="failures",
        choices=["all", "successes", "failures", "none"],
        help="Which episodes to record when --video-dir is set.",
    )
    parser.add_argument(
        "--results",
        default=None,
        help="Optional path to write the aggregated results as JSON.",
    )
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    """Resolve the requested device string to a concrete torch.device."""
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def resolve_dtype(choice: str, device: torch.device) -> str:
    """Resolve the model dtype string ('bfloat16' / 'float32')."""
    if choice != "auto":
        return choice
    return "bfloat16" if device.type == "cuda" else "float32"


def build_policy(args: argparse.Namespace, device: torch.device) -> object:
    """Instantiate the XR0 policy from the pretrained checkpoint."""
    from physicalai.policies.xr0 import XR0  # noqa: PLC0415

    dtype = resolve_dtype(args.dtype, device)
    logger.info(
        "Building XR0 from %s (attn=%s, dtype=%s, device=%s, n_action_steps=%d)",
        args.checkpoint,
        args.attn,
        dtype,
        device,
        args.n_action_steps,
    )
    policy = XR0(
        pretrained_name_or_path=args.checkpoint,
        vlm_attn_implementation=args.attn,
        dtype=dtype,
        n_action_steps=args.n_action_steps,
    )
    policy.to(device)
    policy.eval()
    return policy


def build_benchmark(args: argparse.Namespace) -> object:
    """Create the LiberoBenchmark for the requested suite/tasks."""
    from physicalai.benchmark.gyms.libero import LiberoBenchmark  # noqa: PLC0415

    task_ids = args.task_ids if args.task_ids else None
    logger.info(
        "LiberoBenchmark(task_suite=%s, task_ids=%s, num_episodes=%d)",
        args.task_suite,
        task_ids,
        args.num_episodes,
    )
    return LiberoBenchmark(
        #task_suite=args.task_suite,
        #task_ids=task_ids,
        num_episodes=args.num_episodes,
        #max_steps=args.max_steps,
        seed=args.seed,
        observation_height=args.obs_size,
        observation_width=args.obs_size,
        video_dir=args.video_dir,
        record_mode=args.record_mode if args.video_dir else "none",
    )


def main() -> int:
    """Entry point: build policy, run the LIBERO benchmark, print the summary."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA requested but not available.")
        return 1

    torch.manual_seed(args.seed)

    policy = build_policy(args, device)
    benchmark = build_benchmark(args)

    logger.info("Starting evaluation...")
    results = benchmark.evaluate(policy)

    print("\n" + results.summary())
    print(f"\nOverall success rate: {results.overall_success_rate:.1f}%")

    if args.results:
        path = results.to_json(args.results)
        logger.info("Wrote results to %s", path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
