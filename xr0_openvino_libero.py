#!/usr/bin/env python
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test the exported XR0 OpenVINO model on LIBERO (CPU, one short task).

This loads the OpenVINO IR produced by ``xr0_to_openvino.py`` (in ``xr0_ir/``)
*natively* through the Runtime :class:`~physicalai.inference.model.InferenceModel`
and runs it inside the LIBERO benchmark for a single short task, to verify the
exported model produces usable actions end-to-end -- without the Torch policy.

How it works
------------
The exported OpenVINO graph is self-contained: it runs the whole heavy XR0 stack
-- the Qwen3-VL vision tower, the language model and the DiT action head -- inside
the graph. Its inputs are just the tokenised prompt and the raw pixels/state
(``input_ids`` / ``attention_mask`` / ``pixel_values`` / ``state``); the fixed
image geometry, the 3D MRoPE ``position_ids`` and the image-token scatter
positions were baked in as constants at export time, and the ``action`` output
was baked to f32.

The XR0 tokenisation (the Runtime ``xr0`` preprocessor: Qwen3-VL chat template +
image resize + rendered ``task`` prompt) and the action denormalization (the
Runtime ``xr0_denormalize`` postprocessor) are declared in the exported
``manifest.json`` as registered component types, so ``InferenceModel("xr0_ir")``
reconstructs the full pipeline and ``benchmark.evaluate(model)`` runs it natively
-- exactly the path a deployed Runtime would take.

Run with the LIBERO env python::

    env_libero/bin/python xr0_openvino_libero.py

To run on an Intel GPU instead of CPU, set ``DEVICE = "GPU"`` below.
"""

from __future__ import annotations

import logging

from physicalai.benchmark.gyms import LiberoBenchmark
from physicalai.inference import InferenceModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("xr0_openvino_libero")

# --- Configuration (hardcoded, no CLI) --------------------------------------
EXPORT_DIR = "xr0_ir"
DEVICE = "CPU"  # set to "GPU" to run on an Intel GPU
# Intel GPU compute precision. "bf16" matches the eager/training precision the
# model was validated at (best parity); "f32" is the safe fallback. Both dodge
# the f16-only RoPE OpenCL kernel that fails to build on the GPU plugin.
GPU_PRECISION_HINT = "bf16"

TASK_SUITE = "libero_10"
TASK_IDS = [0]  # a single, short task
NUM_EPISODES = 1
MAX_STEPS = 120


def main() -> None:
    """Load the exported OpenVINO XR0 model and evaluate it on one LIBERO task."""
    logger.info("Loading exported model from %s (device=%s) ...", EXPORT_DIR, DEVICE)
    # GPU precision workaround: the DiT action head's rotary-embedding block lowers
    # to a fused eltwise cone whose *f16* OpenCL kernel fails to build on the Intel
    # GPU plugin ("clBuildProgram, CL_BUILD_PROGRAM_FAILURE -11"); forcing a
    # non-f16 compute precision makes the build succeed. Apply it at load/compile
    # time -- the adapter forwards **adapter_kwargs straight into
    # ``compile_model(config=...)`` -- so the exported IR stays untouched. CPU
    # ignores the hint.
    adapter_kwargs = {"INFERENCE_PRECISION_HINT": GPU_PRECISION_HINT} if DEVICE == "GPU" else {}
    model = InferenceModel(EXPORT_DIR, device=DEVICE, **adapter_kwargs)

    benchmark = LiberoBenchmark(
        task_suite=TASK_SUITE,
        task_ids=TASK_IDS,
        num_episodes=NUM_EPISODES,
        #max_steps=MAX_STEPS,
        seed=42,
        video_dir="xr0_openvino_libero_videos",
    )

    results = benchmark.evaluate(model)

    print("\n" + results.summary())
    print(f"\nOverall success rate: {results.overall_success_rate:.1f}%")


if __name__ == "__main__":
    main()
