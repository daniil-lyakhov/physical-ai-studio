#!/usr/bin/env python
"""Pin down the failing Transpose kernel on the Intel GPU.

    ./env/bin/python probe_transpose_gpu.py /path/to/xr0.xml node_transpose_290
"""

import sys
import openvino as ov
import openvino.opset13 as ops

MARK = ("clBuildProgram", "ProgramBuilder build failed")


def build_fail(e):
    return any(m in str(e) for m in MARK)


def ancestor_parameters(out):
    seen, params, stack = set(), [], [out.get_node()]
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        if n.get_type_name() == "Parameter":
            params.append(n)
            continue
        for inp in n.inputs():
            stack.append(inp.get_source_output().get_node())
    return params


def compile_out(core, out, name):
    try:
        model = ov.Model([ops.result(out)], ancestor_parameters(out), name)
    except Exception as e:  # noqa: BLE001
        return f"assemble-error: {e!r}"
    try:
        core.compile_model(model, "GPU")
    except Exception as e:  # noqa: BLE001
        return "BUILD-FAIL" if build_fail(e) else f"other: {repr(e)[:150]}"
    return "OK"


def main() -> int:
    xml_path = sys.argv[1] if len(sys.argv) > 1 else "xr0_ir/xr0.xml"
    tname = sys.argv[2] if len(sys.argv) > 2 else "node_transpose_290"

    print(f"[info] openvino {ov.__version__}")
    core = ov.Core()
    print(f"[info] devices: {core.available_devices}")

    model = core.read_model(xml_path)
    tr = next((n for n in model.get_ops() if n.get_friendly_name() == tname), None)
    if tr is None or tr.get_type_name() != "Transpose":
        print(f"[FATAL] '{tname}' not found or not a Transpose "
              f"(got {tr.get_type_name() if tr else None}).")
        return 2

    data = tr.input_value(0)
    dsrc = data.get_node()
    perm_node = tr.input_value(1).get_node()
    try:
        perm = list(perm_node.get_vector()) if perm_node.get_type_name() == "Constant" else "?"
    except Exception:  # noqa: BLE001
        perm = "?"
    dshape = data.get_partial_shape()
    dtype = data.get_element_type()
    print(f"[info] Transpose '{tname}'")
    print(f"        data  <- {dsrc.get_type_name()} '{dsrc.get_friendly_name()}' "
          f"{dtype} {dshape}")
    print(f"        perm   = {perm}")
    print(f"        out    = {tr.output(0).get_element_type()} "
          f"{tr.output(0).get_partial_shape()}")

    def static_shape():
        return ov.PartialShape([d.get_length() if d.is_static else -1
                                for d in dshape])

    results = {}

    # A) Baseline: the real transpose in its real context.
    results["A_baseline"] = compile_out(core, tr.output(0), "A")

    # B) Transpose fed by a fresh Parameter (same shape/dtype) -> transpose ALONE.
    try:
        p = ops.parameter(static_shape(), dtype)
        t_only = ops.transpose(p, tr.input_value(1))
        results["B_transpose_only_bf16"] = compile_out(core, t_only.output(0), "B")
    except Exception as e:  # noqa: BLE001
        results["B_transpose_only_bf16"] = f"assemble-error: {e!r}"

    # C) Real cone but transpose done in f32 (upcast input before transpose).
    try:
        t_f32 = ops.transpose(ops.convert(data, "f32"), tr.input_value(1))
        results["C_transpose_f32_realcone"] = compile_out(core, t_f32.output(0), "C")
    except Exception as e:  # noqa: BLE001
        results["C_transpose_f32_realcone"] = f"assemble-error: {e!r}"

    # D) Transpose ALONE but in f32 (Parameter -> f32 -> transpose).
    try:
        p2 = ops.parameter(static_shape(), dtype)
        t_only_f32 = ops.transpose(ops.convert(p2, "f32"), tr.input_value(1))
        results["D_transpose_only_f32"] = compile_out(core, t_only_f32.output(0), "D")
    except Exception as e:  # noqa: BLE001
        results["D_transpose_only_f32"] = f"assemble-error: {e!r}"

    print("\n==================== TRANSPOSE PROBE ====================")
    for k in sorted(results):
        print(f"  {k:<28} : {results[k]}")
    print("========================================================")
    print("\nInterpretation:")
    print("  B FAIL           -> the Transpose PRIMITIVE (this shape/perm/dtype) is broken.")
    print("  B OK, A FAIL     -> failure is UPSTREAM of the transpose, not the transpose.")
    print("  D OK, B FAIL     -> doing the transpose in f32 fixes it (export-time fix).")
    print("  C OK, A FAIL     -> f32 transpose fixes it even in the real cone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
