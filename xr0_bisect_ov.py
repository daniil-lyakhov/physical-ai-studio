#!/usr/bin/env python
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Localize the eager-vs-OpenVINO divergence inside the XR0 DiT action head.

The trajectory parity check (``libero_eager_vs_export.py``) proved the exported
IR diverges from the eager model by ~100x the bf16 floor -- a real graph bug, not
precision. This probe splits the head to find *where* the divergence starts, on a
single observation with identical rectified-flow noise (the IR draws it; the
eager model replays the same array).

Three cut points, from prefix to output:

* ``state_embed``  -- ``flow.state_projector`` output. VLM-independent, computed
  once in the prefix. If this diverges the bug is in the state input path.
* ``vlm_K[0]``     -- the VLM KV-cache key feeding DiT layer 0's joint attention.
  Computed once by the Qwen3-VL backbone. If this diverges the bug is in the VLM.
* ``velocity[0]``  -- DiT ``action_output_layer`` output at Euler step 0. If the
  two above match but this diverges the bug is inside the DiT body (attention /
  RoPE application / RMSNorm / MLP).

The OpenVINO graph unrolls the 5 Euler steps, so the step-0 instance is selected
structurally: the noise node ``x0`` feeds the first ``z += v*dt`` update, whose
velocity input is step 0.

Run with the project env python::

    ./env/bin/python xr0_bisect_ov.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import openvino as ov
import torch
import torch.nn.functional as F  # noqa: N812
from openvino.preprocess import PrePostProcessor

from physicalai.data.observation import IMAGES, STATE, TASK
from physicalai.inference.constants import TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.policies import XR0
from physicalai.policies.xr0.patchify import patchify_image_grid
from physicalai.policies.xr0.pretrained_utils import extract_xr0_dataset_stats

CHECKPOINT = "XiaomiRobotics/Xiaomi-Robotics-0-LIBERO"
EXPORT_DIR = "xr0_ir"
EXPORT_XML = "xr0_ir/xr0.xml"
OV_DEVICE = "CPU"
N_DIT_LAYERS = 16
TEMPORAL_PATCH_SIZE = 2
OV_RANDOM_UNIFORM_GLOBAL_SEED = 42
OV_RANDOM_UNIFORM_OP_SEED = 7
# Input source. False = the baked `sample_input` (the exact layout the graph was
# exported for). True = a real LIBERO observation (step 0 of a rollout), which is
# what the failing deployment actually feeds -- surfaces observation-dependent
# divergence that `sample_input` cannot.
USE_LIBERO_OBS = True
TASK_SUITE = "libero_10"
TASK_ID = 0
SEED = 42


def build_policy(dtype: str = "float32") -> XR0:
    """Build the XR0 policy from the published LIBERO checkpoint.

    Args:
        dtype: Torch dtype for the model weights/compute ("float32" or
            "bfloat16"). The exported IR carries bf16 weights, so a bf16 eager
            build isolates a real graph bug from bf16-weight quantization.

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
        dtype=dtype,
    )
    policy.eval()
    return policy


def build_processed(policy: XR0) -> dict[str, torch.Tensor]:
    """Preprocess the sample observation and right-pad to the baked length.

    Returns:
        The preprocessor output padded to ``config.tokenizer_max_length``.
    """
    processed = policy._preprocessor(policy.sample_input)  # noqa: SLF001
    seq_len = policy.config.tokenizer_max_length
    pad_id = policy._preprocessor.processor.tokenizer.pad_token_id or 0  # noqa: SLF001
    cur_len = processed["input_ids"].shape[1]
    pad = seq_len - cur_len
    if pad > 0:
        processed["input_ids"] = F.pad(processed["input_ids"], (0, pad), value=pad_id)
        processed["attention_mask"] = F.pad(processed["attention_mask"], (0, pad), value=0)
    return processed


def _pad_processed(policy: XR0, processed: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Right-pad ``input_ids`` / ``attention_mask`` to the baked graph length.

    Returns:
        The same dict with token tensors padded to ``tokenizer_max_length``.
    """
    seq_len = policy.config.tokenizer_max_length
    pad_id = policy._preprocessor.processor.tokenizer.pad_token_id or 0  # noqa: SLF001
    cur_len = processed["input_ids"].shape[1]
    pad = seq_len - cur_len
    if pad > 0:
        processed["input_ids"] = F.pad(processed["input_ids"], (0, pad), value=pad_id)
        processed["attention_mask"] = F.pad(processed["attention_mask"], (0, pad), value=0)
    return processed


def build_processed_libero(policy: XR0) -> dict[str, torch.Tensor]:
    """Capture the processed batch the eager model builds for a real LIBERO obs.

    Resets a LIBERO gym, feeds step-0's observation through the eager
    ``select_action`` path, and intercepts the exact batch handed to
    ``predict_action_chunk`` -- the real-deployment input the export graph must
    handle. The batch is then right-padded to the baked graph length.

    Returns:
        The processed batch (input_ids/attention_mask/pixel_values/state, ...).
    """
    from physicalai.benchmark.gyms import LiberoBenchmark  # noqa: PLC0415

    gym = LiberoBenchmark(
        task_suite=TASK_SUITE,
        task_ids=[TASK_ID],
        num_episodes=1,
        seed=SEED,
    ).gyms[0]
    observation, _ = gym.reset(seed=SEED)

    captured: dict[str, dict] = {}
    model = policy.model
    orig = model.predict_action_chunk

    def _capture(batch: dict[str, object]) -> object:
        captured["batch"] = {
            key: (value.clone() if torch.is_tensor(value) else value) for key, value in batch.items()
        }
        return orig(batch)

    model.predict_action_chunk = _capture
    try:
        policy.reset()
        with torch.inference_mode():
            policy.select_action(observation)
    finally:
        model.predict_action_chunk = orig

    if "batch" not in captured:
        msg = "select_action did not call predict_action_chunk (empty action queue?)."
        raise RuntimeError(msg)
    return _pad_processed(policy, captured["batch"])


# --------------------------------------------------------------------------- #
# OpenVINO graph analysis                                                      #
# --------------------------------------------------------------------------- #
def _ancestors(node: ov.Node) -> set[str]:
    """Collect the friendly names of all ancestor nodes (transitive inputs).

    Returns:
        Set of ancestor node friendly names (including ``node`` itself).
    """
    seen: set[str] = set()
    stack = [node]
    while stack:
        cur = stack.pop()
        name = cur.get_friendly_name()
        if name in seen:
            continue
        seen.add(name)
        for i in range(len(cur.inputs())):
            stack.append(cur.input_value(i).get_node())
    return seen


def _find_noise_node(model: ov.Model) -> ov.Node:
    """Locate the Box-Muller Gaussian-noise node (``x0``) in the IR.

    Returns:
        The ``Multiply`` node producing the rectified-flow starting noise.

    Raises:
        RuntimeError: If the node cannot be found.
    """
    for op in model.get_ops():
        if op.get_type_name() != "Multiply":
            continue
        parents = {op.input_value(i).get_node().get_type_name() for i in range(len(op.inputs()))}
        if {"Sqrt", "Cos"} <= parents:
            return op
    msg = "Could not locate the Box-Muller noise node (Sqrt*Cos)."
    raise RuntimeError(msg)


def _first_update_add(noise_node: ov.Node) -> tuple[ov.Node, ov.Node]:
    """Find the first ``z = z + v0*dt`` Add and its noise-side ``z`` branch.

    The step-0 update is ``masked_noise + v0*dt``; both Add inputs descend from
    the noise node (the velocity is ``DiT(masked_noise)``), so the ``z`` branch is
    identified as the specific node BFS traversed through to reach the Add.

    Returns:
        Tuple of ``(update_add, z_branch_node)``.

    Raises:
        RuntimeError: If no such Add is found.
    """
    shape = noise_node.output(0).get_partial_shape()
    frontier = [noise_node]
    seen: set[str] = set()
    while frontier:
        nxt: list[ov.Node] = []
        for node in frontier:
            for tgt in node.output(0).get_target_inputs():
                consumer = tgt.get_node()
                name = consumer.get_friendly_name()
                if name in seen:
                    continue
                seen.add(name)
                if consumer.get_type_name() == "Add" and consumer.output(0).get_partial_shape() == shape:
                    return consumer, node
                nxt.append(consumer)
        frontier = nxt
    msg = "Could not locate the first z-update Add off the noise node."
    raise RuntimeError(msg)


def _velocity0(update_add: ov.Node, z_branch: ov.Node) -> ov.Output:
    """Return the step-0 velocity output feeding the first state-update Add.

    The Add is ``z + v0*dt`` where ``z`` is ``z_branch``; the other input is the
    velocity branch, a ``Multiply`` by the ``dt`` constant whose non-constant
    input is the ``action_output_layer`` velocity.

    Returns:
        The ``ov.Output`` of the step-0 velocity tensor.

    Raises:
        RuntimeError: If the velocity cannot be resolved.
    """
    z_name = z_branch.get_friendly_name()
    for i in range(len(update_add.inputs())):
        cand = update_add.input_value(i)
        if cand.get_node().get_friendly_name() == z_name:
            continue  # this is the z (noise) branch
        node = cand.get_node()
        if node.get_type_name() == "Multiply":
            for j in range(len(node.inputs())):
                inner = node.input_value(j)
                if inner.get_node().get_type_name() != "Constant":
                    return inner
        return cand
    msg = "Could not resolve the step-0 velocity from the update Add."
    raise RuntimeError(msg)


def _matmul_output(model: ov.Model, weight_name: str, allowed: set[str] | None) -> ov.Output | None:
    """Find the output of the MatMul consuming ``weight_name``.

    Args:
        model: The IR model.
        weight_name: Friendly name of the weight constant.
        allowed: If given, only accept a MatMul whose name is in this set
            (used to pick the step-0 instance among unrolled copies).

    Returns:
        The MatMul output, or ``None`` if not found.
    """
    for op in model.get_ops():
        if op.get_type_name() != "MatMul":
            continue
        names = {op.input_value(i).get_node().get_friendly_name() for i in range(len(op.inputs()))}
        if weight_name not in names:
            continue
        if allowed is None or op.get_friendly_name() in allowed:
            return op.output(0)
    return None


def _vlm_kv_layer0(model: ov.Model, step0: set[str]) -> list[ov.Output]:
    """Find the VLM KV-cache tensors prepended in DiT layer 0's attention.

    Layer 0's attention concatenates the DiT-derived K/V (descended from
    ``flow.dit.layers.0.attn.qkv_proj``) with the VLM cache. The VLM branch is the
    Concat input whose ancestry does *not* include that qkv_proj weight.

    Returns:
        The VLM-branch outputs (K and V) of layer-0 attention Concats.
    """
    qkv = "flow.dit.layers.0.attn.qkv_proj.weight"
    found: list[ov.Output] = []
    for op in model.get_ops():
        if op.get_type_name() != "Concat" or op.get_friendly_name() not in step0:
            continue
        inputs = [op.input_value(i) for i in range(len(op.inputs()))]
        if len(inputs) != 2:  # noqa: PLR2004
            continue
        anc = [_ancestors(v.get_node()) for v in inputs]
        dit_side = [qkv in a for a in anc]
        # Exactly one branch is DiT-derived; the other is the pure-VLM cache.
        if dit_side.count(True) == 1:
            vlm_out = inputs[dit_side.index(False)]
            found.append(vlm_out)
    return found


def _velocity_ladder(noise_node: ov.Node) -> list[tuple[ov.Output, ov.Output]]:
    """Walk the unrolled Euler chain, returning ``(velocity, z_state)`` per step.

    Starting from the noise node, each ``z = z + v*dt`` update is found in turn;
    the next step's search starts from that update's output. Returns one entry per
    Euler step in order (step 0 first).

    Returns:
        List of ``(velocity_output, z_state_output)`` per step.
    """
    steps: list[tuple[ov.Output, ov.Output]] = []
    cur: ov.Node = noise_node
    for _ in range(8):  # safety cap; the graph unrolls 5 steps
        try:
            add, z_branch = _first_update_add(cur)
        except RuntimeError:
            break
        steps.append((_velocity0(add, z_branch), add.output(0)))
        cur = add
    return steps


def _pin_random_uniform_seed(model: ov.Model) -> None:
    """Force the IR's ``RandomUniform`` onto a fixed seed for reproducible noise.

    The exported graph bakes ``global_seed=0`` / ``op_seed=0``, which OpenVINO
    treats as non-deterministic (a fresh seed every execution). Overriding both
    seeds with fixed non-zero values makes a fresh compile's first inference draw
    identical starting noise on every run.

    Raises:
        RuntimeError: If no ``RandomUniform`` node is present in the IR.
    """
    for op in model.get_ops():
        if op.get_type_name() == "RandomUniform":
            op.set_attribute("global_seed", OV_RANDOM_UNIFORM_GLOBAL_SEED)
            op.set_attribute("op_seed", OV_RANDOM_UNIFORM_OP_SEED)
            return
    msg = "Could not locate a RandomUniform node to pin the noise seed in the IR."
    raise RuntimeError(msg)


def _load_numpy_preprocessor(manifest: dict[str, Any]) -> Any:  # noqa: ANN401
    """Reconstruct the exported XR0 NumPy inference preprocessor from the manifest.

    Returns:
        The Runtime ``XR0Preprocessor`` resolved from the ``type="xr0"`` spec.

    Raises:
        RuntimeError: If the preprocessor spec is missing from the manifest.
    """
    from physicalai.inference.component_factory import instantiate_component  # noqa: PLC0415
    from physicalai.inference.manifest import ComponentSpec  # noqa: PLC0415

    for spec in manifest["model"]["preprocessors"]:
        if spec.get("type") == "xr0":
            return instantiate_component(ComponentSpec.model_validate(spec))
    msg = "xr0 preprocessor spec not found in the export manifest"
    raise RuntimeError(msg)


def _raw_observation(policy: XR0) -> dict[str, object]:
    """Convert the policy's torch ``sample_input`` into a raw NumPy observation.

    Returns:
        The observation dict with flattened ``images.*`` arrays, a ``state`` array
        and a ``task`` string, as the NumPy inference preprocessor consumes.
    """
    observation: dict[str, object] = {}
    for key, value in policy.sample_input.items():
        if isinstance(key, str) and key.startswith(f"{IMAGES}."):
            observation[key] = value.detach().cpu().numpy()
        elif key == STATE:
            observation[STATE] = value.detach().cpu().numpy()
        elif key == TASK:
            observation[TASK] = value
    return observation


def build_inputs(policy: XR0) -> tuple[dict[str, np.ndarray], dict[str, torch.Tensor]]:
    """Build the OpenVINO feed and the matching eager batch, exactly as the test.

    Mirrors ``test_xr0_openvino_parity``: the NumPy inference preprocessor turns
    the sample observation into the raw pixel grid + state the graph consumes on
    its renamed ``tokenized_prompt`` / ``pixel_values`` / ``state`` ports, and the
    same grid is patchified for the eager model so both backends see identical
    inputs.

    Returns:
        Tuple of ``(graph_inputs, eager_processed)`` -- the NumPy feed dict keyed
        by the graph's input names, and the torch batch (patchified grid + shared
        ids/mask/state) for the eager replay.
    """
    manifest = json.loads((Path(EXPORT_DIR) / "manifest.json").read_text())
    preprocessor = _load_numpy_preprocessor(manifest)
    np_out = preprocessor(_raw_observation(policy))
    pixel_grid = np.ascontiguousarray(np_out["pixel_values"], dtype=np.float32)
    state = np.ascontiguousarray(np_out["state"], dtype=np.float32)

    spec = next(s for s in manifest["model"]["preprocessors"] if s.get("type") == "xr0")
    patch_size = int(spec["patch_size"])
    merge_size = int(spec["merge_size"])

    processed = build_processed(policy)
    graph_inputs = {
        TOKENIZED_PROMPT: processed["input_ids"].cpu().numpy().astype(np.int64),
        TOKENIZED_PROMPT_MASK: processed["attention_mask"].cpu().numpy().astype(np.int64),
        "pixel_values": pixel_grid,
        "state": state,
    }

    num_images, _, height, width = pixel_grid.shape
    grid_thw = [[1, height // patch_size, width // patch_size]] * num_images
    eager_pixels = patchify_image_grid(
        torch.from_numpy(pixel_grid),
        grid_thw,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        patch_size=patch_size,
        merge_size=merge_size,
    )
    eager_processed = {
        "input_ids": processed["input_ids"],
        "attention_mask": processed["attention_mask"],
        "pixel_values": eager_pixels,
        "image_grid_thw": torch.tensor(grid_thw, dtype=torch.int64),
        "state": torch.from_numpy(state),
    }
    return graph_inputs, eager_processed


def expose_and_run(graph_inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compile the IR with extra outputs for the bisection cut points and run it.

    Returns:
        Mapping of cut-point name to its numpy value (plus ``noise``).
    """
    core = ov.Core()
    model = core.read_model(EXPORT_XML)
    _pin_random_uniform_seed(model)

    noise_node = _find_noise_node(model)

    extra: dict[str, ov.Output] = {"noise": noise_node.output(0)}
    ladder = _velocity_ladder(noise_node)
    for i, (vel, z_state) in enumerate(ladder):
        extra[f"velocity{i}"] = vel
        extra[f"z{i}"] = z_state

    # Step-0 subgraph: pick the DiT sublayer instances used by the first update.
    first_add, _ = _first_update_add(noise_node)
    step0 = _ancestors(first_add)
    ap = _matmul_output(model, "flow.action_projector.layers.2.weight", step0)
    if ap is not None:
        extra["action_proj"] = ap
    sp = _matmul_output(model, "flow.state_projector.layers.2.weight", None)
    if sp is not None:
        extra["state_proj"] = sp
    for layer in range(N_DIT_LAYERS):
        attn = _matmul_output(model, f"flow.dit.layers.{layer}.attn.o_proj.weight", step0)
        if attn is not None:
            extra[f"L{layer}.attn"] = attn
        mlp = _matmul_output(model, f"flow.dit.layers.{layer}.mlp.down_proj.weight", step0)
        if mlp is not None:
            extra[f"L{layer}.mlp"] = mlp
        if layer == 0:
            gate = _matmul_output(model, f"flow.dit.layers.{layer}.mlp.gate_proj.weight", step0)
            if gate is not None:
                extra[f"L{layer}.gate"] = gate
            up = _matmul_output(model, f"flow.dit.layers.{layer}.mlp.up_proj.weight", step0)
            if up is not None:
                extra[f"L{layer}.up"] = up

    names = list(extra)
    model.add_outputs([extra[n] for n in names])

    # Cast the appended outputs to f32 so NumPy can read otherwise-bf16 tensors.
    ppp = PrePostProcessor(model)
    n_orig = len(model.outputs) - len(names)
    for i in range(n_orig, len(model.outputs)):
        ppp.output(i).tensor().set_element_type(ov.Type.f32)
    model = ppp.build()

    compiled = core.compile_model(model, OV_DEVICE, {"INFERENCE_PRECISION_HINT": "f32"})
    result = compiled(graph_inputs)
    out: dict[str, np.ndarray] = {}
    base = len(compiled.outputs) - len(names)
    for i, name in enumerate(names):
        out[name] = np.asarray(result[compiled.output(base + i)])
    return out


# --------------------------------------------------------------------------- #
# Eager capture                                                                #
# --------------------------------------------------------------------------- #
def run_eager(policy: XR0, processed: dict[str, torch.Tensor], noise: np.ndarray) -> dict[str, np.ndarray]:
    """Run the eager head with the IR's noise, capturing the cut-point tensors.

    Returns:
        Mapping of cut-point name to its numpy value.
    """
    model = policy.model
    flow = model.flow
    captured: dict[str, np.ndarray] = {}
    velocities: list[np.ndarray] = []

    def vel_hook(_m: torch.nn.Module, _inp: tuple, out: torch.Tensor) -> None:
        velocities.append(out.detach().float().cpu().numpy())

    def first_call(name: str):  # noqa: ANN202
        def hook(_m: torch.nn.Module, _inp: tuple, out: torch.Tensor) -> None:
            if name not in captured:  # keep the FIRST call (Euler step 0)
                captured[name] = out.detach().float().cpu().numpy()

        return hook

    handles = [flow.action_output_layer.register_forward_hook(vel_hook)]
    for layer in range(N_DIT_LAYERS):
        blk = flow.dit.layers[layer]
        handles.append(blk.attn.o_proj.register_forward_hook(first_call(f"L{layer}.attn")))
        handles.append(blk.mlp.down_proj.register_forward_hook(first_call(f"L{layer}.mlp")))
    blk0 = flow.dit.layers[0]
    handles.append(blk0.mlp.gate_proj.register_forward_hook(first_call("L0.gate")))
    handles.append(blk0.mlp.up_proj.register_forward_hook(first_call("L0.up")))
    handles.append(flow.action_projector.register_forward_hook(first_call("action_proj")))
    handles.append(flow.state_projector.register_forward_hook(first_call("state_proj")))

    fixed = torch.from_numpy(noise)
    orig_noise = model._sample_noise  # noqa: SLF001
    model._sample_noise = lambda action, seed: fixed.to(action.device, action.dtype).reshape(action.shape)  # noqa: SLF001, ARG005, E501
    try:
        batch = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in processed.items()}
        with torch.no_grad():
            model.predict_action_chunk(batch)
    finally:
        model._sample_noise = orig_noise  # noqa: SLF001
        for h in handles:
            h.remove()
    for i, vel in enumerate(velocities):
        captured[f"velocity{i}"] = vel
    return captured


def _report(name: str, a: np.ndarray, b: np.ndarray) -> None:
    """Print max/mean abs diff plus the argmax location and both values there."""
    if a.shape != b.shape:
        print(f"{name:>14} : SHAPE MISMATCH eager {a.shape} vs ov {b.shape}")
        return
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    idx = np.unravel_index(int(diff.argmax()), diff.shape)
    print(
        f"{name:>14} : max {diff.max():.3e} | mean {diff.mean():.3e} | "
        f"at {idx} eager {a[idx]:+.3f} ov {b[idx]:+.3f} | shape {tuple(a.shape)}"
    )


def _report3(name: str, f32: np.ndarray, bf16: np.ndarray, ov: np.ndarray) -> None:
    """Print OV's max abs diff against both the f32 and bf16 eager references.

    If ``ov`` tracks the ``bf16`` reference (``ov~bf16`` small) the divergence is
    bf16-weight precision; if ``ov`` diverges from *both* it is a real graph bug.
    """
    if not (f32.shape == bf16.shape == ov.shape):
        print(f"{name:>10} : SHAPE MISMATCH")
        return
    d_f32 = np.abs(f32.astype(np.float32) - ov.astype(np.float32)).max()
    d_bf16 = np.abs(bf16.astype(np.float32) - ov.astype(np.float32)).max()
    d_floor = np.abs(f32.astype(np.float32) - bf16.astype(np.float32)).max()
    flag = "  <-- graph bug" if d_bf16 > 3 * max(d_floor, 1e-3) else ""
    print(f"{name:>10} : ov-vs-f32 {d_f32:.3e} | ov-vs-bf16 {d_bf16:.3e} | bf16-floor {d_floor:.3e}{flag}")


def main() -> None:
    """Run the eager-vs-OpenVINO bisection on one observation and report diffs."""
    policy = build_policy("float32")
    graph_inputs, eager_processed = build_inputs(policy)

    print("Running exported IR with exposed cut points ...")
    ov_out = expose_and_run(graph_inputs)

    noise = ov_out["noise"].astype(np.float32)
    print("Replaying the IR noise through the f32 eager head ...")
    eager = run_eager(policy, eager_processed, noise)
    del policy

    print("Replaying the IR noise through the bf16 eager head ...")
    policy_bf16 = build_policy("bfloat16")
    eager_bf16 = run_eager(policy_bf16, eager_processed, noise)

    n_steps = sum(1 for k in ov_out if k.startswith("velocity"))
    print("\n=== velocity ladder: OV vs f32 eager vs bf16 eager ===")
    for i in range(n_steps):
        key = f"velocity{i}"
        if key in eager and key in eager_bf16 and key in ov_out:
            _report3(key, eager[key], eager_bf16[key], ov_out[key])

    print("\n=== step-0 DiT per-layer: OV vs f32 vs bf16 (attn o_proj / mlp down_proj) ===")
    for layer in range(N_DIT_LAYERS):
        for sub in ("attn", "mlp"):
            key = f"L{layer}.{sub}"
            if key in eager and key in eager_bf16 and key in ov_out:
                _report3(key, eager[key], eager_bf16[key], ov_out[key])
    print("\n=== DiT input path: OV vs f32 vs bf16 (projectors) ===")
    for key in ("state_proj", "action_proj"):
        if key in eager and key in eager_bf16 and key in ov_out:
            _report3(key, eager[key], eager_bf16[key], ov_out[key])
    print("\n=== L0 SwiGLU sub-ops: OV vs f32 vs bf16 (gate / up / down=L0.mlp) ===")
    for sub in ("gate", "up", "mlp"):
        key = f"L0.{sub}"
        if key in eager and key in eager_bf16 and key in ov_out:
            _report3(key, eager[key], eager_bf16[key], ov_out[key])
    print("\nInterpret: rows flagged '<-- graph bug' diverge from BOTH the f32 and "
          "bf16 eager refs, isolating a real OpenVINO graph error from bf16 precision.")


if __name__ == "__main__":
    main()
