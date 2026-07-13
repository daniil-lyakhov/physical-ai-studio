#!/usr/bin/env python
"""Bisect the subgraph that breaks GPU compilation of an XR0 IR.

Reuses the GPU-compile helper ``compile_out`` (and, transitively,
``ancestor_parameters``) from ``probe_region_f32_gpu.py`` instead of duplicating
OpenVINO logic.

Strategy: a monotone cut-frontier BISECT. For a topological frontier ``k`` every
tensor produced upstream of ``k`` but consumed downstream is rewired to a fresh
``Parameter``, so only the ops at index >= ``k`` are compiled. Cutting past the
failure's fusion cluster makes the GPU build succeed, so "still BUILD-FAILs" is
monotone in ``k`` -- binary search finds the LARGEST failing ``k`` in ~log2(N)
GPU compiles (vs a node-by-node walk that recompiles the whole VLM each step).
``ordered[k]`` is the op whose presence triggers the ``clBuildProgram`` abort.

    ./env/bin/python xr0_gpu_compile_probe.py [xr0_ir/xr0.xml]

Run on the machine with the Intel GPU.
"""

from __future__ import annotations

import sys

import openvino as ov
import openvino.opset13 as opset

from probe_region_f32_gpu import compile_out

_BUILD_FAIL = "BUILD-FAIL"


def _target_tensor(model: ov.Model) -> ov.Output:
    """Return the tensor FEEDING the first Result (never the Result output)."""
    out = model.outputs[0]
    node = out.get_node()
    return node.input_value(0) if node.get_type_name() == "Result" else out


def _cut_frontier(model: ov.Model, order: dict[str, int], k: int) -> tuple[ov.Output, int]:
    """Clone ``model`` and replace every tensor crossing topo-frontier ``k``.

    Any tensor produced by an op with topo-index < ``k`` but consumed by an op
    with index >= ``k`` is rewired to a fresh ``Parameter`` of the same shape and
    dtype. The returned target tensor's ancestor cone is then bounded to ops at
    index >= ``k`` (plus surviving original Parameters), so compiling it only
    builds the downstream region -- fast, and monotone in ``k``.

    Returns the target tensor and the number of tensors that were cut.
    """
    clone = model.clone()
    target = _target_tensor(clone)
    n_cuts = 0
    for op in clone.get_ordered_ops():
        if order[op.get_friendly_name()] >= k:
            continue
        for out in op.outputs():
            crossing = [
                c for c in out.get_target_inputs()
                if order[c.get_node().get_friendly_name()] >= k
            ]
            if not crossing:
                continue
            # Parameters accept a PartialShape, so dynamic-dim tensors are fine.
            param = opset.parameter(out.get_partial_shape(), out.get_element_type())
            param.set_friendly_name(f"cut::{op.get_friendly_name()}")
            for inp in crossing:
                inp.replace_source_output(param.output(0))
            n_cuts += 1
    return target, n_cuts


def _fails(core: ov.Core, model: ov.Model, order: dict[str, int], k: int,
           memo: dict[int, str]) -> bool:
    """True iff the frontier-``k`` cut model BUILD-FAILs on GPU (memoized)."""
    status = memo.get(k)
    if status is None:
        target, n_cuts = _cut_frontier(model, order, k)
        status = compile_out(core, target, f"k{k}", device="GPU")
        memo[k] = status
        print(f"    frontier k={k:>6}  cuts={n_cuts:>5}  ->  {status.split(':')[0]}", flush=True)
    return status == _BUILD_FAIL


def bisect(core: ov.Core, model: ov.Model, ordered: list[ov.Node]) -> ov.Node:
    """Binary-search the topo frontier for the op critical to the GPU failure.

    Finds the LARGEST ``k`` whose cut model still BUILD-FAILs. Cutting one step
    further (``k+1``) removes ``ordered[k]`` from the downstream region and the
    build succeeds -- so ``ordered[k]`` is the op whose presence triggers the
    ``clBuildProgram`` abort (the fusion sink / culprit).
    """
    order = {op.get_friendly_name(): i for i, op in enumerate(ordered)}
    memo: dict[int, str] = {}

    # Bound the search by the target's own topo index. At k == idx_target the
    # target's inputs are all cut to Parameters (trivial cone -> OK); at k == 0
    # nothing is cut (full graph -> FAIL). fails() is monotone True->False here.
    target = _target_tensor(model)
    idx_target = order[target.get_node().get_friendly_name()]

    print("[bisect] confirming full graph (k=0) fails on GPU ...", flush=True)
    if not _fails(core, model, order, 0, memo):
        raise RuntimeError("full graph did not BUILD-FAIL on GPU; nothing to bisect")

    lo, hi = 0, idx_target  # invariant: fails(lo) == True, fails(hi) == False
    if _fails(core, model, order, hi, memo):
        return ordered[hi]  # even the target's immediate cone fails
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _fails(core, model, order, mid, memo):
            lo = mid
        else:
            hi = mid
    return ordered[lo]


def _describe(node: ov.Node, ordered: list[ov.Node], idx: int) -> None:
    print(f"\n>>> CRITICAL NODE [topo #{idx}]: {node.get_friendly_name()}  ({node.get_type_name()})")
    for i in range(len(node.inputs())):
        src = node.input_value(i).get_node()
        shape = node.input_value(i).get_partial_shape()
        print(f"    in[{i}] <- {src.get_friendly_name()}  ({src.get_type_name()})  shape={shape}")
    lo = max(0, idx - 3)
    hi = min(len(ordered), idx + 4)
    print("    topo window:")
    for j in range(lo, hi):
        mark = "  <== critical" if j == idx else ""
        op = ordered[j]
        print(f"      #{j} {op.get_type_name():<24} {op.get_friendly_name()}{mark}")


def main() -> int:
    xml = sys.argv[1] if len(sys.argv) > 1 else "xr0_ir/xr0.xml"

    print(f"[info] openvino {ov.__version__}")
    core = ov.Core()
    print(f"[info] devices: {core.available_devices}")
    if not any("GPU" in d for d in core.available_devices):
        print("[FATAL] no GPU device visible to OpenVINO.")
        return 2

    model = core.read_model(xml)
    ordered = model.get_ordered_ops()
    print(f"[info] read {xml}: {len(model.outputs)} output(s), {len(ordered)} ops")

    culprit = bisect(core, model, ordered)
    idx = {op.get_friendly_name(): i for i, op in enumerate(ordered)}[culprit.get_friendly_name()]
    _describe(culprit, ordered, idx)

    print("\n[next] drill into this node with the region probe:")
    print(f"    ./env/bin/python probe_region_f32_gpu.py {xml} {culprit.get_friendly_name()}")
    print("[next] classify: GatherND -> gathernd_bool bug, Transpose -> bf16_transpose bug,")
    print("       Multiply/Add -> rope_fusion bug (see openvino_gpu_*_bug.md).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
