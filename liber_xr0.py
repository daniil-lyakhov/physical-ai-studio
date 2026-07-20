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

    # evaluate a whole suite (all tasks)
    env_libero/bin/python liber_xr0.py --task-suite libero_spatial --num-episodes 20

    # full LIBERO-10 sweep with videos + JSON results
    env_libero/bin/python liber_xr0.py --task-suite libero_10 --num-episodes 20 \
        --video-dir ./videos --results ./xr0_libero_results.json

The eval is aligned with the Xiaomi reference (``eval_libero/main.py``): instructions
are capitalized with a trailing period, the LIBERO init state is advanced per episode,
and the rectified-flow noise is seeded deterministically per observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys

import numpy as np
import torch
from PIL import Image

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
    parser.add_argument("--num-episodes", type=int, default=20, help="Episodes per task.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max steps per episode (default: LIBERO suite default).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Evaluation seed. Matches the Xiaomi reference (eval_libero args.seed=7).",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=10,
        help=(
            "Actions executed per predicted chunk before replanning (open-loop horizon). "
            "Matches the reference LIBERO eval's replan_steps=10; the full chunk is 30 but "
            "executing all 30 open-loop overshoots the target and collapses success to ~0%%."
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


def format_instruction(text: str) -> str:
    """Match the reference eval's instruction formatting: capitalize + trailing period."""
    return str(text).capitalize() + "."


def _hash_data_to_seed(data: dict, max_bytes: int = 4) -> int:
    """Byte-for-byte replica of the Xiaomi reference ``hash_data_to_seed``.

    Serialises the dict (numpy arrays + PIL images) to canonical JSON and hashes
    it with SHA256, so the resulting seed matches the reference eval exactly for
    identical inputs.
    """

    def custom_encoder(obj: object) -> object:
        if isinstance(obj, np.ndarray):
            return {"__type__": "numpy", "dtype": str(obj.dtype), "shape": obj.shape, "data": obj.tobytes().hex()}
        if isinstance(obj, Image.Image):
            img_hash = hashlib.md5(obj.tobytes()).hexdigest()  # noqa: S324
            return {"__type__": "PIL.Image", "mode": obj.mode, "size": obj.size, "content_hash": img_hash}
        if isinstance(obj, set):
            return sorted(obj)
        msg = f"Type {type(obj)} is not JSON serializable"
        raise TypeError(msg)

    json_str = json.dumps(
        data,
        default=custom_encoder,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    seed_int = int(hashlib.sha256(json_str.encode("utf-8")).hexdigest(), 16)
    if max_bytes > 0:
        seed_int = seed_int % (2 ** (8 * max_bytes))
    return seed_int


def _obs_to_pil(img: object) -> Image.Image | None:
    """Recover the reference's uint8 HWC PIL image from a framework image tensor.

    The framework gym stores images as float32 CHW in ``[0, 1]`` (normalised from
    uint8/255); the reference hashes the original uint8 HWC PIL image. Rounding
    ``x * 255`` recovers the exact uint8 values, so the PIL bytes match.
    """
    if img is None:
        return None
    arr = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
    if arr.ndim == 4:  # (1, C, H, W) -> (C, H, W)
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 3:  # CHW -> HWC  # noqa: PLR2004
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(arr))


def observation_seed(observation: object, max_bytes: int = 4) -> int:
    """Derive a deterministic seed matching the Xiaomi reference eval.

    Reconstructs the reference ``model_inputs`` dict (``task_id``, 32-dim padded
    ``state``, ``base``/``wrist_left`` PIL images, capitalised ``language`` before
    the trailing period) from the framework observation, then applies the exact
    same SHA256-over-canonical-JSON hashing as ``eval_libero/main.py``.

    Note: the reference builds ``state`` in float64 while the framework gym
    downcasts it to float32 in ``to_observation`` (widened back here), so the
    state bytes match in value/layout but not in the low mantissa bits.
    """
    model_inputs: dict[str, object] = {"task_id": "libero_all"}

    state = getattr(observation, "state", None)
    if state is not None:
        arr = state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else np.asarray(state)
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        if arr.shape[0] < 32:  # noqa: PLR2004
            arr = np.concatenate([arr, np.zeros(32 - arr.shape[0], dtype=np.float64)])
        model_inputs["state"] = arr

    images = getattr(observation, "images", None) or {}
    if isinstance(images, dict):
        base = _obs_to_pil(images.get("image"))
        wrist_left = _obs_to_pil(images.get("image2"))
        if base is not None:
            model_inputs["base"] = base
        if wrist_left is not None:
            model_inputs["wrist_left"] = wrist_left

    task = getattr(observation, "task", None)
    if isinstance(task, (list, tuple)):
        task = task[0] if task else ""
    language = str(task)
    if language.endswith("."):  # reference hashes capitalize() BEFORE appending the period
        language = language[:-1]
    model_inputs["language"] = language

    return _hash_data_to_seed(model_inputs, max_bytes)


def _wrap_reset_align(gym: object, seed: int) -> None:
    """Match the Xiaomi reference per-episode reset behaviour.

    - Advance the LIBERO init state each episode (reference eval iterates init states).
    - Pin the env RNG to a constant ``seed`` every episode: the reference calls
      ``env.seed(args.seed)`` with the SAME seed for every trial, whereas the
      framework increments the seed per episode (``start_seed + rollout_idx``).
    """
    original_reset = gym.reset
    init_states = getattr(gym, "_init_states", None)
    n = len(init_states) if init_states is not None else 0
    counter = {"i": 0}

    def reset(*args: object, **kwargs: object) -> object:
        if n:
            gym._init_state_id = counter["i"] % n  # noqa: SLF001
            counter["i"] += 1
        kwargs["seed"] = seed  # constant env seed across episodes (Xiaomi env.seed(args.seed))
        return original_reset(*args, **kwargs)

    gym.reset = reset


def _wrap_predict_seed(policy: object) -> None:
    """Seed rectified-flow noise deterministically per observation before each chunk."""
    original = policy.predict_action_chunk

    def predict_action_chunk(batch: object, *args: object, **kwargs: object) -> object:
        torch.manual_seed(observation_seed(batch))
        return original(batch, *args, **kwargs)

    policy.predict_action_chunk = predict_action_chunk


def apply_xiaomi_alignment(benchmark: object, policy: object, seed: int) -> None:
    """Align the framework eval with the Xiaomi reference protocol (in-place).

    - Format instructions (capitalize + trailing period).
    - Cycle the LIBERO init state per episode and pin the env seed (constant per trial).
    - Seed the rectified-flow noise deterministically per observation.
    """
    for gym in benchmark.gyms:
        gym.task_description = format_instruction(gym.task_description)
        _wrap_reset_align(gym, seed)
    _wrap_predict_seed(policy)
    logger.info(
        "Xiaomi alignment applied: instruction e.g. %r; per-episode init states; per-obs flow seed.",
        benchmark.gyms[0].task_description,
    )


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
    """Create the LiberoBenchmark for the requested suite (all tasks)."""
    from physicalai.benchmark.gyms.libero import LiberoBenchmark  # noqa: PLC0415

    max_steps = args.max_steps
    logger.info(
        "LiberoBenchmark(task_suite=%s, all tasks, num_episodes=%d, max_steps=%s)",
        args.task_suite,
        args.num_episodes,
        max_steps if max_steps is not None else "suite-default",
    )
    return LiberoBenchmark(
        task_suite=args.task_suite,
        num_episodes=args.num_episodes,
        max_steps=max_steps,
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
    np.random.seed(args.seed)

    policy = build_policy(args, device)
    benchmark = build_benchmark(args)

    apply_xiaomi_alignment(benchmark, policy, args.seed)

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
