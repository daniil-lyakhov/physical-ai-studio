#!/usr/bin/env python
"""Test whether keeping the transpose AND its consuming RoPE multiply in one
contiguous f32 region builds on the Intel GPU (i.e. a region wrap that survives
the plugin's convert-folding / eltwise fusion).

    ./env/bin/python probe_region_f32_gpu.py /path/to/xr0.xml node_mul_605

Run against the ORIGINAL (un-wrapped) IR.
"""

import sys
import openvino as ov
import openvino.opset13 as ops

MARK = ("clBuildProgram", "ProgramBuilder build failed", "CL_BUILD_PROGRAM_FAILURE")


def build_fail(e):
    return any(m in str(e) for m in MARK)


def ancestor_parameters(out):
    # Dedup by the STABLE per-node instance id. Python's id() is unreliable here
    # because get_node() returns a throwaway wrapper each call and CPython can
    # recycle a freed wrapper's id(), falsely pruning a whole ancestor branch
    # (which drops Parameters and yields "undeclared parameters" on assemble).
    seen, params, stack = set(), [], [out.get_node()]
    while stack:
        node = stack.pop()
        iid = node.get_instance_id()
        if iid in seen:
            continue
        seen.add(iid)
        if node.get_type_name() == "Parameter":
            params.append(node)
            continue
        for inp in node.inputs():
            stack.append(inp.get_source_output().get_node())
    return params


def compile_out(core, out, name, device="GPU"):
    try:
        model = ov.Model([ops.result(out)], ancestor_parameters(out), name)
    except Exception as e:  # noqa: BLE001
        return f"assemble-error: {e!r}"
    try:
        core.compile_model(model, device)
    except Exception as e:  # noqa: BLE001
        return "BUILD-FAIL" if build_fail(e) else f"other: {repr(e)[:160]}"
    return "OK"


def main() -> int:
    xml = sys.argv[1] if len(sys.argv) > 1 else "xr0_ir/xr0.xml"
    mul_name = sys.argv[2] if len(sys.argv) > 2 else "node_mul_605"

    print(f"[info] openvino {ov.__version__}")
    core = ov.Core()
    print(f"[info] devices: {core.available_devices}")
    model = core.read_model(xml)

    mul = next((n for n in model.get_ops() if n.get_friendly_name() == mul_name), None)
    if mul is None or mul.get_type_name() != "Multiply":
        print(f"[FATAL] '{mul_name}' not found or not a Multiply.")
        return 2

    # Identify the transpose input vs the constant input of the RoPE multiply.
    in_tr = in_const = None
    for iv in (mul.input_value(0), mul.input_value(1)):
        if iv.get_node().get_type_name() == "Transpose":
            in_tr = iv
        else:
            in_const = iv
    if in_tr is None or in_const is None:
        print("[FATAL] expected one Transpose input and one other input.")
        print(f"  in0={mul.input_value(0).get_node().get_type_name()} "
              f"in1={mul.input_value(1).get_node().get_type_name()}")
        return 3

    tr = in_tr.get_node()
    data = tr.input_value(0)      # bf16 tensor feeding the transpose
    perm = tr.input_value(1)      # perm constant

    results = {}

    # Baseline: the real bf16 region as-is.
    results["0_baseline_bf16"] = compile_out(core, mul.output(0), "baseline")

    # Region f32: convert BEFORE the transpose, keep f32 through the multiply,
    # convert back to bf16 only after. Converts bracket real f32 compute, so the
    # plugin cannot fold them away.
    d32 = ops.convert(data, "f32")
    t32 = ops.transpose(d32, perm)
    c32 = ops.convert(in_const, "f32")
    m32 = ops.multiply(t32, c32)
    out = ops.convert(m32, "bf16")
    results["1_region_f32_transpose_mul"] = compile_out(core, out.output(0), "region")

    print("\n==================== REGION PROBE ====================")
    for k in sorted(results):
        print(f"  {k:<30} : {results[k]}")
    print("=====================================================")
    print("\nInterpretation:")
    print("  1 OK, 0 FAIL -> an f32 region across transpose+multiply is the fix")
    print("                  (implement a region wrap at export).")
    print("  1 FAIL       -> even an f32 region fails; only a full f16 export works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
