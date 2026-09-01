#!/usr/bin/env python
"""Slice the *real* failing subgraph out of the exported XR0 IR and compile it
standalone on GPU/CPU.

Why
---
The minimal, hand-written RoPE-on-transposed-q block compiles fine on the Intel
GPU in fp16 (see ``openvino_gpu_rope_transpose_repro.py``). So the clBuildProgram
failure at the culprit node (``node_mul_605``, found by ``bisect_gpu_build.py``)
does NOT come from that op in isolation -- it comes from the *fused surrounding
subgraph* the plugin actually builds. To reproduce the true bug we cut the exact
ancestor cone of the culprit straight out of ``xr0_ir/xr0.xml`` and compile it.

Because the exported prefix through the culprit fails to build, the culprit's
ancestor cone is the smallest self-contained subgraph that still reproduces the
failure. Optionally extend the slice a few levels downstream (``--downstream N``)
so the plugin's fusion of the culprit with its consumers (e.g. the ``Add`` in
``(q*cos) + (rotate_half(q)*sin)``) is preserved.

Run on the Intel GPU machine:
    ./env/bin/python extract_failing_cone_gpu.py --xml xr0_ir/xr0.xml --node node_mul_605
    ./env/bin/python extract_failing_cone_gpu.py --node node_mul_605 --downstream 2 --save cone_ir/cone.xml
"""

from __future__ import annotations

import argparse

import openvino as ov
import openvino.opset13 as ops

BUILD_FAIL_MARKERS = ("clBuildProgram", "ProgramBuilder build failed")


def is_build_failure(exc: Exception) -> bool:
    return any(m in str(exc) for m in BUILD_FAIL_MARKERS)


def find_node(model: ov.Model, name: str):
    for op in model.get_ops():
        if op.get_friendly_name() == name:
            return op
    raise SystemExit(f"[FATAL] node {name!r} not found in the IR")


def ancestors(target) -> dict:
    """Return {id: node} for the full ancestor cone of ``target`` (inclusive)."""
    seen: dict = {}
    stack = [target]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen[id(node)] = node
        for inp in node.inputs():
            src = inp.get_source_output().get_node()
            if id(src) not in seen:
                stack.append(src)
    return seen


def extend_downstream(target, levels: int) -> list:
    """Return consumer nodes up to ``levels`` hops below ``target``."""
    frontier = [target]
    collected: dict = {id(target): target}
    for _ in range(levels):
        nxt = []
        for node in frontier:
            for out in node.outputs():
                for ti in out.get_target_inputs():
                    c = ti.get_node()
                    if id(c) not in collected:
                        collected[id(c)] = c
                        nxt.append(c)
        frontier = nxt
    return list(collected.values())


def build_submodel(nodes: dict, name: str) -> ov.Model:
    """Assemble a standalone ov.Model from the given {id: node} set.

    Parameters are the Parameter ops inside the set; results are every output
    that leaves the set (consumed outside, or unconsumed). Non-float frontier
    outputs are converted to f32 so they are valid GPU model outputs.
    """
    node_ids = set(nodes.keys())
    parameters = [n for n in nodes.values() if n.get_type_name() == "Parameter"]

    results = []
    for node in nodes.values():
        if node.get_type_name() in ("Result", "Parameter", "Constant"):
            continue
        for out in node.outputs():
            consumers = list(out.get_target_inputs())
            leaves = (not consumers) or any(id(ti.get_node()) not in node_ids for ti in consumers)
            if not leaves:
                continue
            src = out
            et = str(out.get_element_type())
            if et not in ("f16", "f32", "bf16"):
                src = ops.convert(out, "f32").output(0)
            results.append(ops.result(src))

    if not results:
        raise SystemExit("[FATAL] slice has no frontier outputs")
    return ov.Model(results, parameters, name)


def try_device(core: ov.Core, submodel: ov.Model, device: str) -> None:
    try:
        core.compile_model(submodel, device)
    except Exception as e:  # noqa: BLE001
        kind = "BUILD-FAIL (clBuildProgram)" if is_build_failure(e) else "other error"
        print(f"  [{device}] compile FAILED -- {kind}:\n      {repr(e)[:600]}")
        return
    print(f"  [{device}] compile OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="xr0_ir/xr0.xml")
    ap.add_argument("--node", default="node_mul_605", help="culprit node friendly name")
    ap.add_argument("--downstream", type=int, default=0,
                    help="also include N levels of consumers (and their ancestors)")
    ap.add_argument("--devices", nargs="+", default=["GPU", "CPU"])
    ap.add_argument("--save", default="", help="optional path to save the sliced IR")
    args = ap.parse_args()

    core = ov.Core()
    print(f"[info] openvino {ov.__version__}")
    print(f"[info] devices : {core.available_devices}")
    for d in args.devices:
        if d in core.available_devices:
            try:
                print(f"[info]   {d}: {core.get_property(d, 'FULL_DEVICE_NAME')}")
            except Exception:  # noqa: BLE001
                pass

    model = core.read_model(args.xml)
    target = find_node(model, args.node)

    nodes = dict(ancestors(target))
    if args.downstream > 0:
        for succ in extend_downstream(target, args.downstream):
            nodes.update(ancestors(succ))

    print(f"[info] culprit : {target.get_friendly_name()} "
          f"({target.get_type_name()}, out {target.output(0).get_element_type()} "
          f"{target.output(0).get_partial_shape()})")
    print(f"[info] slice   : {len(nodes)} ops "
          f"(downstream levels = {args.downstream})")
    op_hist: dict[str, int] = {}
    for n in nodes.values():
        op_hist[n.get_type_name()] = op_hist.get(n.get_type_name(), 0) + 1
    print(f"[info] op mix  : {dict(sorted(op_hist.items(), key=lambda kv: -kv[1]))}")

    submodel = build_submodel(nodes, f"cone_{args.node}")
    submodel.validate_nodes_and_infer_types()

    if args.save:
        ov.save_model(submodel, args.save, compress_to_fp16=False)
        print(f"[info] saved sliced IR -> {args.save}")

    print("\n[compile] building the sliced cone on each device:")
    for device in args.devices:
        if device not in core.available_devices:
            print(f"  [{device}] not available, skipping")
            continue
        try_device(core, submodel, device)

    print("\nIf GPU shows BUILD-FAIL while CPU shows OK, this sliced cone is a")
    print("faithful, self-contained reproducer of the real clBuildProgram bug.")
    print("Shrink it further with a smaller --node or grow it with --downstream.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
