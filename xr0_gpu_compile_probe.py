#!/usr/bin/env python
"""Localize the subgraph that breaks GPU compilation of an XR0 IR.

Reuses the GPU-compile helpers from ``probe_region_f32_gpu.py`` (``compile_out``,
``ancestor_parameters``, ``build_fail``) instead of duplicating them.

The GPU failure is a *fusion* bug: per ``openvino_gpu_rope_fusion_bug.md`` the
isolated block builds fine and the ``clBuildProgram`` abort only appears at the
sink node once the surrounding cone is present. So we localize TOP-DOWN from a
failing model output: descend into whichever input branch still BUILD-FAILs, and
stop at the node whose own cone fails but ALL of whose inputs build -- that node
is the fusion sink to feed into ``probe_region_f32_gpu.py``.

    ./env/bin/python xr0_gpu_compile_probe.py [xr0_ir/xr0.xml]

Run on the machine with the Intel GPU.
"""

from __future__ import annotations

import sys

import openvino as ov

from probe_region_f32_gpu import ancestor_parameters, build_fail, compile_out

_BUILD_FAIL = "BUILD-FAIL"


def _fails_on_gpu(core: ov.Core, out: ov.Output, cache: dict[str, str]) -> bool:
    """Return True iff the ancestor cone of ``out`` hits a GPU build failure.

    Results are memoized by the producing node's friendly name so the top-down
    descent never recompiles the same cone twice.
    """
    key = out.get_node().get_friendly_name()
    status = cache.get(key)
    if status is None:
        status = compile_out(core, out, key)
        cache[key] = status
    return status == _BUILD_FAIL


def localize(core: ov.Core, out: ov.Output, cache: dict[str, str]) -> ov.Node:
    """Descend from a failing output to the minimal BUILD-FAIL sink node."""
    node = out.get_node()
    while True:
        next_branch: ov.Output | None = None
        for i in range(len(node.inputs())):
            src = node.input_value(i)
            if src.get_node().get_type_name() == "Constant":
                continue
            if _fails_on_gpu(core, src, cache):
                next_branch = src
                break
        if next_branch is None:
            return node  # cone fails but every input builds -> this is the sink
        node = next_branch.get_node()


def _describe(node: ov.Node) -> None:
    print(f"\n>>> BROKEN SINK NODE: {node.get_friendly_name()}  ({node.get_type_name()})")
    for i in range(len(node.inputs())):
        src = node.input_value(i).get_node()
        shape = node.input_value(i).get_partial_shape()
        print(f"    in[{i}] <- {src.get_friendly_name()}  ({src.get_type_name()})  shape={shape}")
    n_params = len(ancestor_parameters(node.output(0)))
    print(f"    ancestor Parameters in cone: {n_params}")


def main() -> int:
    xml = sys.argv[1] if len(sys.argv) > 1 else "xr0_ir/xr0.xml"

    print(f"[info] openvino {ov.__version__}")
    core = ov.Core()
    print(f"[info] devices: {core.available_devices}")
    if not any("GPU" in d for d in core.available_devices):
        print("[FATAL] no GPU device visible to OpenVINO.")
        return 2

    model = core.read_model(xml)
    print(f"[info] read {xml}: {len(model.outputs)} output(s), {len(model.get_ops())} ops")

    cache: dict[str, str] = {}
    culprit: ov.Node | None = None
    for idx, out in enumerate(model.outputs):
        status = compile_out(core, out, f"output_{idx}")
        cache[out.get_node().get_friendly_name()] = status
        print(f"[probe] output[{idx}] {out.get_node().get_friendly_name()}: {status}")
        if status == _BUILD_FAIL:
            culprit = localize(core, out, cache)
            break

    if culprit is None:
        print("\n[result] No BUILD-FAIL reproduced from the model outputs on this GPU.")
        return 0

    _describe(culprit)
    print("\n[next] drill into this node with the region probe:")
    print(f"    ./env/bin/python probe_region_f32_gpu.py {xml} {culprit.get_friendly_name()}")
    print("[next] classify: GatherND -> gathernd_bool bug, Transpose -> bf16_transpose bug,")
    print("       Multiply/Add -> rope_fusion bug (see openvino_gpu_*_bug.md).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
