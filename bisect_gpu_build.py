#!/usr/bin/env python
"""Binary-search the exact node whose kernel fails to build on the Intel GPU.

Run on the machine WITH the Intel GPU:

    ./env/bin/python bisect_gpu_build.py /path/to/xr0.xml

Idea
----
Topologically order the graph. A "prefix" = the first k ops. Because the prefix
is ancestor-closed, it contains the culprit node C iff k > index(C). So compiling
a prefix fails with clBuildProgram iff the prefix includes C. That makes failure
monotonic in k, so we binary-search the smallest failing k. The node newly added
at that k is the offender.

For each prefix we expose its "frontier" outputs as model results (converting
non-float outputs to f32 so they are always valid GPU outputs), then compile.
"""

import sys
import openvino as ov
import openvino.opset13 as ops

BUILD_FAIL_MARKERS = ("clBuildProgram", "ProgramBuilder build failed")


def is_build_failure(exc: Exception) -> bool:
    msg = str(exc)
    return any(m in msg for m in BUILD_FAIL_MARKERS)


def build_prefix_model(ordered, k):
    """Build an ov.Model containing exactly the first k topological ops."""
    prefix = ordered[:k]
    prefix_set = set(id(n) for n in prefix)

    parameters = [n for n in prefix if n.get_type_name() == "Parameter"]

    results = []
    seen = set()
    for node in prefix:
        for out in node.outputs():
            # An output is on the frontier if any consumer is outside the prefix,
            # or it has no consumers at all (dangling), or it is a graph result.
            consumers = list(out.get_target_inputs())
            outside = any(id(ti.get_node()) not in prefix_set for ti in consumers)
            frontier = outside or len(consumers) == 0
            if not frontier:
                continue
            key = (id(node), out.get_index())
            if key in seen:
                continue
            seen.add(key)
            src = out
            # Normalize non-float outputs to f32 so they are valid GPU model outputs.
            et = str(out.get_element_type())
            if et not in ("f16", "f32", "bf16"):
                src = ops.convert(out, "f32").output(0)
            results.append(ops.result(src))

    if not results:
        raise RuntimeError("no frontier outputs for this prefix")
    return ov.Model(results, parameters, f"prefix_{k}")


def compile_prefix(core, ordered, k):
    """Return (status, detail). status in {'ok','build_fail','other'}."""
    try:
        model = build_prefix_model(ordered, k)
    except Exception as e:  # noqa: BLE001
        return "other", f"assemble error: {e!r}"
    try:
        core.compile_model(model, "GPU")
    except Exception as e:  # noqa: BLE001
        if is_build_failure(e):
            return "build_fail", str(e)
        return "other", repr(e)
    return "ok", ""


def describe(node) -> str:
    ins = ", ".join(
        f"{i.get_node().get_type_name()}[{i.get_element_type()}]"
        for i in node.inputs()
    )
    outs = ", ".join(
        f"{o.get_element_type()}:{o.get_partial_shape()}" for o in node.outputs()
    )
    return (f"  name : {node.get_friendly_name()}\n"
            f"  type : {node.get_type_name()}\n"
            f"  in   : {ins or '(none)'}\n"
            f"  out  : {outs}")


def main() -> int:
    xml_path = sys.argv[1] if len(sys.argv) > 1 else "xr0_ir/xr0.xml"
    print(f"[info] openvino {ov.__version__}")
    core = ov.Core()
    print(f"[info] devices: {core.available_devices}")
    if not any("GPU" in d for d in core.available_devices):
        print("[FATAL] no GPU visible to OpenVINO on this machine.")
        return 2

    model = core.read_model(xml_path)
    ordered = model.get_ordered_ops()
    n = len(ordered)
    print(f"[info] total ops: {n}")

    # Sanity: full graph must fail, else there is nothing to bisect.
    print("[info] compiling FULL graph (expected to fail) ...")
    status, detail = compile_prefix(core, ordered, n)
    print(f"[info] full graph -> {status}")
    if status != "build_fail":
        print("[FATAL] full graph did not reproduce the clBuildProgram failure "
              f"({status}). Detail:\n{detail}")
        return 3

    # Binary search: smallest k in [1, n] whose prefix fails to build.
    lo, hi = 1, n
    first_fail = n
    while lo <= hi:
        mid = (lo + hi) // 2
        status, detail = compile_prefix(core, ordered, mid)
        tag = {"ok": "OK", "build_fail": "BUILD-FAIL", "other": "skip"}[status]
        print(f"[bisect] k={mid:6d}  {ordered[mid-1].get_type_name():<18} -> {tag}")
        if status == "build_fail":
            first_fail = mid
            hi = mid - 1
        elif status == "ok":
            lo = mid + 1
        else:
            # 'other' error is ambiguous for this k; nudge upward past it.
            # (Rare — happens if a frontier output type is unsupported as an output.)
            lo = mid + 1

    culprit = ordered[first_fail - 1]
    print("\n==================== CULPRIT ====================")
    print(f"[result] first failing prefix k = {first_fail}")
    print(describe(culprit))
    print("\n[context] producers feeding this node:")
    for inp in culprit.inputs():
        src = inp.get_source_output().get_node()
        print(f"    <- {src.get_type_name():<18} {src.get_friendly_name()} "
              f"[{inp.get_element_type()}]")
    print("[context] consumers of this node:")
    for out in culprit.outputs():
        for ti in out.get_target_inputs():
            dst = ti.get_node()
            print(f"    -> {dst.get_type_name():<18} {dst.get_friendly_name()}")
    print("================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
