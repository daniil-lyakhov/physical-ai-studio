#!/usr/bin/env python
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Numerical parity check: XR0 eager forward vs in-graph-export forward.

The OpenVINO export path swaps several stock Qwen3-VL ops for
export-friendly (but *numerically identical*) variants -- the vision tower
geometry helpers read a baked Python grid instead of ``grid_thw.tolist()``, the
vision attention calls SDPA directly, the 3D MRoPE ``position_ids`` are injected
as a baked constant, and the image/text merge + deepstack injection scatter by
integer index (``index_copy``) instead of ``masked_scatter`` (see
``xr0/vlm.py``). This script confirms those wrappers introduce **no deviation**
in the model output.

It runs the *same* padded observation through ``model.predict_action_chunk``
twice on one policy instance:

1. First in normal **eager** mode (all stock ops), capturing the action chunk.
2. Then after ``prepare_ingraph_export(...)`` flips the shim into **in-graph
   export mode** (the wrapper ops above), capturing the action chunk again.

The rectified-flow sampler draws Gaussian starting noise, so both runs pin the
same ``seed`` in the batch (``XR0Vla._sample_noise`` is deterministic when a seed
is provided), making the comparison exact up to op-ordering. The two chunks are
then compared and per-element diff statistics are printed.

Run with the project env python::

    ./env/bin/python xr0_orig_vs_export.py
"""

from __future__ import annotations

import numpy as np
import openvino as ov
import torch
import torch.nn.functional as F  # noqa: N812
from openvino.preprocess import PrePostProcessor

from physicalai.policies import XR0
from physicalai.policies.xr0.pretrained_utils import extract_xr0_dataset_stats

CHECKPOINT = "XiaomiRobotics/Xiaomi-Robotics-0-LIBERO"
# Directory / IR produced by ``xr0_to_openvino.py``.
EXPORT_XML = "xr0_ir/xr0.xml"
# Device the exported IR is compiled on for the parity run. Set to "GPU" to
# compare against the same Intel GPU path used at inference (with the precision
# hint applied at compile time, exactly like the InferenceModel load).
OV_DEVICE = "GPU"
# Intel GPU compute precision. "bf16" matches the eager/training precision (best
# parity); "f32" is the safe fallback. Both dodge the f16-only RoPE OpenCL kernel.
GPU_PRECISION_HINT = "bf16"
# Fixed rectified-flow noise seed so eager and export draw identical noise.
SEED = 42
# Max abs diff above which we treat the wrappers as having changed the numerics.
# The model head runs in bf16, so a tiny op-ordering epsilon is expected; a real
# wrapper bug would move the output far more than this.
TOLERANCE = 1e-2
# The exported IR runs its DiT head in bf16 on CPU, so the OpenVINO action will
# differ from the eager (also bf16) action by a small kernel/accumulation epsilon
# even when fed identical noise; treat anything under this as a match.
OV_TOLERANCE = 1e-1


def build_policy() -> XR0:
    """Build the XR0 policy from the published LIBERO checkpoint.

    Mirrors ``xr0_to_openvino.py``: the checkpoint only carries action
    normalization stats, so the observation schema (state + two camera views) is
    added so ``sample_input`` / the preprocessor have everything they need.

    Returns:
        The initialized XR0 policy in eval mode.
    """
    stats = extract_xr0_dataset_stats(CHECKPOINT) or {}
    stats["observation.state"] = {"name": "state", "type": "STATE", "shape": (8,)}
    stats["observation.images.base"] = {"name": "images.base", "type": "VISUAL", "shape": (3, 256, 256)}
    stats["observation.images.wrist_left"] = {
        "name": "images.wrist_left",
        "type": "VISUAL",
        "shape": (3, 256, 256),
    }
    policy = XR0(
        pretrained_name_or_path=CHECKPOINT,
        dataset_stats=stats,
        vlm_attn_implementation="sdpa",
    )
    policy.eval()
    return policy


def build_processed(policy: XR0) -> dict[str, torch.Tensor]:
    """Preprocess the sample observation and right-pad it to the baked length.

    Returns:
        The preprocessor output with ``input_ids`` / ``attention_mask``
        right-padded to ``config.tokenizer_max_length`` (the fixed length the
        export graph is baked for).
    """
    processed = policy._preprocessor(policy.sample_input)  # noqa: SLF001
    seq_len = policy.config.tokenizer_max_length
    pad_id = policy._preprocessor.processor.tokenizer.pad_token_id or 0  # noqa: SLF001
    cur_len = processed["input_ids"].shape[1]
    if cur_len > seq_len:
        msg = f"Sample prompt ({cur_len} tokens) exceeds SEQ_LEN={seq_len}."
        raise ValueError(msg)
    pad = seq_len - cur_len
    if pad:
        processed["input_ids"] = F.pad(processed["input_ids"], (0, pad), value=pad_id)
        processed["attention_mask"] = F.pad(processed["attention_mask"], (0, pad), value=0)
    return processed


@torch.no_grad()
def run_forward(policy: XR0, processed: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run ``model.predict_action_chunk`` on a fresh copy of the batch.

    ``predict_action_chunk`` pops keys from the batch, so the input is cloned to
    keep ``processed`` reusable across the eager and export runs. A fixed ``seed``
    is injected so the rectified-flow starting noise is deterministic.

    Returns:
        The predicted (still normalized) action chunk as a CPU float32 tensor.
    """
    batch: dict[str, object] = {
        key: (value.clone() if torch.is_tensor(value) else value) for key, value in processed.items()
    }
    batch["seed"] = SEED
    actions = policy.model.predict_action_chunk(batch)
    return actions.float().cpu()


@torch.no_grad()
def run_forward_with_noise(policy: XR0, processed: dict[str, torch.Tensor], noise: torch.Tensor) -> torch.Tensor:
    """Run ``predict_action_chunk`` forcing a specific rectified-flow ``noise``.

    The exported IR samples its starting noise internally (a ``RandomUniform``
    with seed 0), so to compare it against the eager model we must feed the eager
    model the *same* noise the IR drew. This temporarily overrides
    ``XR0Model._sample_noise`` so the flow starts from the supplied ``noise``.

    Returns:
        The predicted (still normalized) action chunk as a CPU float32 tensor.
    """
    batch: dict[str, object] = {
        key: (value.clone() if torch.is_tensor(value) else value) for key, value in processed.items()
    }
    model = policy.model
    original = model._sample_noise  # noqa: SLF001
    model._sample_noise = lambda action, seed: noise.to(action.device, action.dtype)  # noqa: SLF001, ARG005
    try:
        actions = model.predict_action_chunk(batch)
    finally:
        model._sample_noise = original  # noqa: SLF001
    return actions.float().cpu()


def _ov_config() -> dict[str, str]:
    """Compile config for the OpenVINO parity run.

    On the Intel GPU the DiT action head's rotary-embedding block has an f16
    OpenCL kernel that fails to build (``clBuildProgram, CL_BUILD_PROGRAM_FAILURE
    -11``); forcing a non-f16 compute precision via ``INFERENCE_PRECISION_HINT``
    makes it build. This mirrors the hint the ``InferenceModel`` load passes to
    ``compile_model``. ``bf16`` additionally matches the eager compute precision.

    Returns:
        The compile config dict (empty for non-GPU devices).
    """
    return {"INFERENCE_PRECISION_HINT": GPU_PRECISION_HINT} if OV_DEVICE == "GPU" else {}


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


def run_openvino(processed: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the exported IR, returning both its action and the noise it drew.

    The IR's ``RandomUniform`` (seed 0) draws fresh noise every inference, so the
    noise node is exposed as an extra output and read back from the *same* run as
    the action -- guaranteeing the returned ``action`` and ``noise`` are
    consistent. The noise can then be replayed through the eager model for an
    apples-to-apples comparison.

    Returns:
        Tuple of ``(action, noise)`` as CPU float32 tensors.
    """
    core = ov.Core()
    model = core.read_model(EXPORT_XML)

    # Expose the internal Gaussian noise as a second output (cast to f32 so NumPy
    # can read the otherwise-bf16 tensor) without disturbing the action output.
    noise_node = _find_noise_node(model)
    model.add_outputs(noise_node.output(0))
    ppp = PrePostProcessor(model)
    ppp.output(1).tensor().set_element_type(ov.Type.f32)
    model = ppp.build()

    compiled = core.compile_model(model, OV_DEVICE, _ov_config())
    feed = {
        "input_ids": processed["input_ids"].cpu().numpy().astype(np.int64),
        "attention_mask": processed["attention_mask"].cpu().numpy().astype(np.int64),
        "pixel_values": processed["pixel_values"].float().cpu().numpy().astype(np.float32),
        "state": processed["state"].float().cpu().numpy().astype(np.float32),
    }
    result = compiled(feed)
    action = torch.from_numpy(np.asarray(result[compiled.output(0)])).float()
    noise = torch.from_numpy(np.asarray(result[compiled.output(1)])).float()
    return action, noise


def _report(name: str, reference: torch.Tensor, other: torch.Tensor, tolerance: float) -> bool:
    """Print per-element diff statistics for two action chunks.

    Returns:
        ``True`` when the max abs diff is within ``tolerance``.
    """
    diff = (reference - other).abs()
    denom = reference.abs().clamp_min(1e-6)
    max_abs = diff.max().item()
    within = max_abs <= tolerance
    print(f"\n=== {name} ===")
    print(f"output shape        : {tuple(reference.shape)}")
    print(f"max  abs diff       : {max_abs:.3e}")
    print(f"mean abs diff       : {diff.mean().item():.3e}")
    print(f"max  rel diff       : {(diff / denom).max().item():.3e}")
    print(f"reference range     : [{reference.min().item():.4f}, {reference.max().item():.4f}]")
    print(f"other     range     : [{other.min().item():.4f}, {other.max().item():.4f}]")
    print(f"within(atol={tolerance:g})   : {within}")
    return within


def main() -> None:
    """Compare eager vs in-graph-export action outputs and report the diff."""
    torch.manual_seed(0)
    policy = build_policy()
    processed = build_processed(policy)

    print("Running EAGER forward (stock ops) ...")
    eager = run_forward(policy, processed)

    print("Enabling in-graph export mode (baking constants + wrapper ops) ...")
    policy.prepare_ingraph_export(processed)

    print("Running EXPORT-mode forward (wrapper ops) ...")
    export = run_forward(policy, processed)

    if eager.shape != export.shape:
        msg = f"Shape mismatch: eager {tuple(eager.shape)} vs export {tuple(export.shape)}"
        raise AssertionError(msg)

    diff = (eager - export).abs()
    denom = eager.abs().clamp_min(1e-6)
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = (diff / denom).max().item()
    allclose = torch.allclose(eager, export, rtol=1e-3, atol=TOLERANCE)

    print("\n=== eager vs in-graph-export action parity ===")
    print(f"output shape        : {tuple(eager.shape)}")
    print(f"max  abs diff       : {max_abs:.3e}")
    print(f"mean abs diff       : {mean_abs:.3e}")
    print(f"max  rel diff       : {max_rel:.3e}")
    print(f"eager  range        : [{eager.min().item():.4f}, {eager.max().item():.4f}]")
    print(f"export range        : [{export.min().item():.4f}, {export.max().item():.4f}]")
    print(f"allclose(atol={TOLERANCE:g}) : {allclose}")

    if max_abs <= TOLERANCE:
        print(f"\nPASS: wrappers add no meaningful deviation (max abs diff {max_abs:.3e} <= {TOLERANCE:g}).")
    else:
        print(f"\nFAIL: wrappers changed the output (max abs diff {max_abs:.3e} > {TOLERANCE:g}).")
        raise SystemExit(1)

    # ------------------------------------------------------------------ #
    # Exported OpenVINO IR vs eager (in-graph-export mode), same noise.   #
    # ------------------------------------------------------------------ #
    print("\nRunning EXPORTED OpenVINO IR ...")
    ov_action, ov_noise = run_openvino(processed)

    print("Replaying the IR's noise through the eager model ...")
    eager_ov = run_forward_with_noise(policy, processed, ov_noise)

    if ov_action.shape != eager_ov.shape:
        msg = f"Shape mismatch: OV {tuple(ov_action.shape)} vs eager {tuple(eager_ov.shape)}"
        raise AssertionError(msg)

    within = _report("OpenVINO IR vs eager (same noise) parity", eager_ov, ov_action, OV_TOLERANCE)
    if within:
        print(f"\nPASS: OpenVINO IR matches the eager model (max abs diff <= {OV_TOLERANCE:g}).")
    else:
        print(f"\nFAIL: OpenVINO IR diverges from the eager model (max abs diff > {OV_TOLERANCE:g}).")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
