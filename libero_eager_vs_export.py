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
import shutil
from pathlib import Path

import numpy as np
import openvino as ov
import openvino.opset13 as ops
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

# Fully deterministic parity: replace the rectified-flow sampler with a single
# fixed noise draw in BOTH models -- baked as a Constant into the exported IR and
# monkeypatched into the eager sampler -- so the only remaining difference is
# numerical precision, not the RNG stream.
CONSTANT_NOISE = True
CONST_NOISE_DIR = "xr0_ir_constnoise"
CONST_NOISE_SEED = 0

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


def _find_noise_node(model: ov.Model) -> ov.Node:
    """Locate the Gaussian-noise (``randn``) node in the exported IR.

    ``torch.randn`` lowers to a ``RandomUniform`` followed by a Box-Muller
    transform: ``sqrt(-2*log(u1)) * cos(2*pi*u2)``. The final ``Multiply`` of that
    ``Sqrt`` and ``Cos`` branch is the rectified-flow starting noise.

    Returns:
        The ``Multiply`` node producing the Gaussian noise tensor.

    Raises:
        RuntimeError: If the Box-Muller ``Multiply`` cannot be found.
    """
    for op in model.get_ops():
        if op.get_type_name() != "Multiply":
            continue
        parents = {op.input_value(i).get_node().get_type_name() for i in range(len(op.inputs()))}
        if {"Sqrt", "Cos"} <= parents:
            return op
    msg = "Could not locate the Box-Muller noise node (Sqrt*Cos) in the IR."
    raise RuntimeError(msg)


def bake_constant_noise(src_dir: str, dst_dir: str, seed: int) -> np.ndarray:
    """Write a copy of the IR whose noise node is replaced by a fixed Constant.

    Reads ``src_dir``'s IR, finds the Box-Muller Gaussian-noise node, and rewires
    all its consumers to a Constant filled with a single deterministic standard-
    normal draw. The ``manifest.json`` is copied verbatim so ``InferenceModel``
    can load the result exactly like the original export.

    Args:
        src_dir: Directory holding the original ``xr0.xml`` / ``xr0.bin`` / manifest.
        dst_dir: Directory to write the constant-noise IR into.
        seed: RNG seed for the fixed noise draw.

    Returns:
        The baked noise as a float32 numpy array (to feed the eager model too).
    """
    src = Path(src_dir)
    dst = Path(dst_dir)
    core = ov.Core()
    model = core.read_model(str(src / "xr0.xml"))
    noise_node = _find_noise_node(model)

    shape = list(noise_node.get_output_partial_shape(0).to_shape())
    elem = noise_node.get_output_element_type(0)
    noise_np = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)

    const = ops.constant(noise_np, dtype=ov.Type.f32)
    source = const if elem == ov.Type.f32 else ops.convert(const, destination_type=elem)
    for target in list(noise_node.output(0).get_target_inputs()):
        target.replace_source_output(source.output(0))
    model.validate_nodes_and_infer_types()

    dst.mkdir(parents=True, exist_ok=True)
    ov.save_model(model, str(dst / "xr0.xml"), compress_to_fp16=False)
    del model  # release the source .bin mmap before returning
    shutil.copy2(src / "manifest.json", dst / "manifest.json")
    return noise_np


def patch_eager_noise(policy: XR0, noise_np: np.ndarray) -> None:
    """Force the eager rectified-flow sampler to return the fixed baked noise.

    The IR's noise node output shares the eager action-noise layout, so the same
    array feeds both sides element-for-element.
    """
    fixed = torch.from_numpy(noise_np)

    def _sample_noise(action: torch.Tensor, seed: object) -> torch.Tensor:  # noqa: ARG001
        return fixed.to(action.device, action.dtype).reshape(action.shape)

    policy.model._sample_noise = _sample_noise  # noqa: SLF001


def build_export(export_dir: str) -> object:
    """Load the exported OpenVINO IR through the Runtime InferenceModel.

    Args:
        export_dir: Directory of the IR to load (original or constant-noise copy).

    Returns:
        A Policy-wrapped InferenceModel exposing ``select_action(observation)``.
    """
    adapter_kwargs = {"INFERENCE_PRECISION_HINT": GPU_PRECISION_HINT} if OV_DEVICE == "GPU" else {}
    inf_model = InferenceModel(export_dir, device=OV_DEVICE, **adapter_kwargs)
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

    export_dir = EXPORT_DIR
    if CONSTANT_NOISE:
        logger.info("Baking constant noise into IR -> %s ...", CONST_NOISE_DIR)
        noise_np = bake_constant_noise(EXPORT_DIR, CONST_NOISE_DIR, CONST_NOISE_SEED)
        patch_eager_noise(eager, noise_np)
        export_dir = CONST_NOISE_DIR

    logger.info("Loading exported OpenVINO IR from %s (device=%s) ...", export_dir, OV_DEVICE)
    export = build_export(export_dir)
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
