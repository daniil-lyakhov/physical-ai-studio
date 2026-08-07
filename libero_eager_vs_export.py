#!/usr/bin/env python
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Trajectory parity: eager XR0 vs exported OpenVINO IR over a LIBERO rollout.

The single-sample parity check (``xr0_orig_vs_export.py``) only compares one
observation. This script instead drives a *real* LIBERO episode and, at every
step, feeds the **same** observation to both models, comparing the action each
returns. This surfaces divergence that only shows up along a trajectory
(closed-loop drift, per-frame precision sensitivity).

Protocol
--------
* One shared gym, reset once. Both policies reset once.
* Each step, both ``select_action(observation)`` on the identical observation.
* The environment is advanced with the **eager** action, so the trajectory is
  the known-good reference path; the OpenVINO action is measured *against* it on
  that path (it does not steer the env).
* Both policies replan on the same schedule (same ``n_action_steps`` queue),
  reset together and stepped in lockstep, so their action queues stay aligned.

Caveat -- rectified-flow noise
------------------------------
Each chunk starts from random Gaussian noise. The eager model draws it with
``torch.randn``; the exported IR draws its own internally (a baked
``RandomUniform``), and the two RNG streams cannot be synchronised through
``select_action``. So on replan steps the per-step diff includes an irreducible
noise term, not just precision. Watch the *mean over the run* and whether the
diff stays bounded rather than growing -- a real export bug moves it far more
than the noise floor.

Run with the LIBERO env python::

    env_libero/bin/python libero_eager_vs_export.py
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from physicalai.benchmark.gyms import LiberoBenchmark
from physicalai.benchmark.gyms.benchmark import _wrap_policy
from physicalai.inference import InferenceModel
from physicalai.policies.xr0 import XR0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("libero_eager_vs_export")

# --- Configuration ----------------------------------------------------------
CHECKPOINT = "XiaomiRobotics/Xiaomi-Robotics-0-LIBERO"
EXPORT_DIR = "xr0_ir"
# Eager reference: float32 on CPU is the most accurate baseline to compare against.
EAGER_DTYPE = "float32"
EAGER_DEVICE = "cpu"
# OpenVINO device for the exported IR. "GPU" runs the Intel iGPU path.
OV_DEVICE = "CPU"
# Intel GPU compute precision (ignored on CPU). "bf16" matches the eager/training
# precision; "f32" is the safe fallback. Both dodge the f16-only RoPE OpenCL kernel.
GPU_PRECISION_HINT = "bf16"

TASK_SUITE = "libero_10"
TASK_ID = 0
N_STEPS = 50
SEED = 42
N_ACTION_STEPS = 10  # open-loop horizon per predicted chunk (matches the reference eval)


def build_eager() -> XR0:
    """Build the eager XR0 policy from the published checkpoint.

    Returns:
        The XR0 policy in eval mode on ``EAGER_DEVICE``.
    """
    policy = XR0(
        pretrained_name_or_path=CHECKPOINT,
        vlm_attn_implementation="sdpa",
        dtype=EAGER_DTYPE,
        n_action_steps=N_ACTION_STEPS,
    )
    policy.to(torch.device(EAGER_DEVICE))
    policy.eval()
    return policy


def build_export() -> object:
    """Load the exported OpenVINO IR through the Runtime InferenceModel.

    Returns:
        A Policy-wrapped InferenceModel exposing ``select_action(observation)``.
    """
    adapter_kwargs = {"INFERENCE_PRECISION_HINT": GPU_PRECISION_HINT} if OV_DEVICE == "GPU" else {}
    inf_model = InferenceModel(EXPORT_DIR, device=OV_DEVICE, **adapter_kwargs)
    return _wrap_policy(inf_model)


def build_gym() -> object:
    """Build a single LIBERO gym for the configured task.

    Returns:
        The first gym of a one-task LiberoBenchmark.
    """
    benchmark = LiberoBenchmark(
        task_suite=TASK_SUITE,
        task_ids=[TASK_ID],
        num_episodes=1,
        seed=SEED,
    )
    return benchmark.gyms[0]


def _as_1d(action: torch.Tensor) -> np.ndarray:
    """Flatten a (B, action_dim) or (action_dim,) action to 1D float32 numpy.

    Returns:
        The executed action as a 1D numpy array.
    """
    arr = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else np.asarray(action)
    return arr.reshape(-1).astype(np.float32)


def main() -> None:
    """Roll out one LIBERO episode and report eager-vs-export per-step action diff."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    logger.info("Building eager XR0 (%s, %s) ...", EAGER_DTYPE, EAGER_DEVICE)
    eager = build_eager()
    logger.info("Loading exported OpenVINO IR from %s (device=%s) ...", EXPORT_DIR, OV_DEVICE)
    export = build_export()
    logger.info("Building LIBERO gym %s task %d ...", TASK_SUITE, TASK_ID)
    gym = build_gym()

    observation, _ = gym.reset(seed=SEED)
    eager.reset()
    export.reset()

    abs_diffs: list[float] = []
    header = f"{'step':>4} | {'max abs':>10} | {'mean abs':>10} | {'eager[0]':>10} | {'export[0]':>10}"
    print(header)
    print("-" * len(header))

    for step in range(N_STEPS):
        with torch.inference_mode():
            a_eager = eager.select_action(observation)
            a_export = export.select_action(observation)

        ve = _as_1d(a_eager)
        vo = _as_1d(a_export)
        diff = np.abs(ve - vo)
        abs_diffs.append(float(diff.max()))
        print(f"{step:>4} | {diff.max():>10.3e} | {diff.mean():>10.3e} | {ve[0]:>10.4f} | {vo[0]:>10.4f}")

        # Advance the env with the eager (reference) action.
        step_action = a_eager.squeeze(0) if a_eager.ndim > 1 else a_eager
        observation, _reward, terminated, truncated, _info = gym.step(step_action)
        if bool(terminated) or bool(truncated):
            logger.info("Episode ended at step %d.", step)
            break

    arr = np.asarray(abs_diffs)
    print("\n=== eager vs exported OpenVINO -- trajectory action parity ===")
    print(f"steps compared      : {arr.size}")
    print(f"per-step max abs    : mean {arr.mean():.3e} | max {arr.max():.3e} | min {arr.min():.3e}")
    print(f"OV device / precision: {OV_DEVICE} / {GPU_PRECISION_HINT if OV_DEVICE == 'GPU' else 'native'}")


if __name__ == "__main__":
    main()
