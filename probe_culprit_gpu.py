#!/usr/bin/env python
"""Isolate WHY a specific node's kernel fails to build on the Intel GPU.

Run on the machine WITH the Intel GPU:

    ./env/bin/python probe_culprit_gpu.py /path/to/xr0.xml node_mul_605

It compiles a series of tiny sub-models, each testing one hypothesis about the
culprit node, and reports which variant builds and which fails. This tells us
whether the problem is the bf16 dtype, the feeding Transpose, or the fusion.
"""

import sys
import openvino as ov
import openvino.opset13 as ops

BUILD_FAIL_MARKERS = ("clBuildProgram", "ProgramBuilder build failed")


def is_build_failure(exc: Exception) -> bool:
    return any(m in str(exc) for m in BUILD_FAIL_MARKERS)


def ancestor_parameters(out):
    """All Parameter nodes reachable backward from `out`."""
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


def compile_output(core, out, name):
    """Build a sub-model producing `out` and compile it on GPU."""
    try:
        params = ancestor_parameters(out)
        model = ov.Model([ops.result(out)], params, name)
    except Exception as e:  # noqa: BLE001
        return f"assemble-error: {e!r}"
    try:
        core.compile_model(model, "GPU")
    except Exception as e:  # noqa: BLE001
        return "BUILD-FAIL" if is_build_failure(e) else f"other: {repr(e)[:160]}"
    return "OK"


def main() -> int:
    xml_path = sys.argv[1] if len(sys.argv) > 1 else "xr0_ir/xr0.xml"
    target_name = sys.argv[2] if len(sys.argv) > 2 else "node_mul_605"

    print(f"[info] openvino {ov.__version__}")
    core = ov.Core()
    print(f"[info] devices: {core.available_devices}")

    model = core.read_model(xml_path)
    target = next((n for n in model.get_ops()
                   if n.get_friendly_name() == target_name), None)
    if target is None:
        print(f"[FATAL] node '{target_name}' not found.")
        return 2
    if target.get_type_name() != "Multiply":
        print(f"[warn] target type is {target.get_type_name()}, expected Multiply. "
              "Some probes may not apply.")

    in0 = target.input_value(0)
    in1 = target.input_value(1)
    n0, n1 = in0.get_node(), in1.get_node()
    print(f"[info] target {target_name} = Multiply(")
    print(f"          in0 = {n0.get_type_name()} '{n0.get_friendly_name()}' "
          f"{in0.get_element_type()} {in0.get_partial_shape()}")
    print(f"          in1 = {n1.get_type_name()} '{n1.get_friendly_name()}' "
          f"{in1.get_element_type()} {in1.get_partial_shape()} )")

    results = {}

    # 1) Baseline: the culprit exactly as-is.
    results["1_baseline_bf16"] = compile_output(core, target.output(0), "baseline")

    # 2) Same multiply but computed in f32 (inputs upcast, result downcast).
    m32 = ops.multiply(ops.convert(in0, "f32"), ops.convert(in1, "f32"))
    out32 = ops.convert(m32, "bf16")
    results["2_multiply_in_f32"] = compile_output(core, out32.output(0), "mul_f32")

    # 3) Only the Transpose producer (if present) — does that kernel build alone?
    transpose_in = None
    for iv in (in0, in1):
        if iv.get_node().get_type_name() == "Transpose":
            transpose_in = iv
            break
    if transpose_in is not None:
        results["3_transpose_alone"] = compile_output(
            core, transpose_in, "transpose_alone")
        # 4) Replace the Transpose with a fresh Parameter, keep bf16 multiply.
        shape = transpose_in.get_partial_shape()
        try:
            pshape = ov.PartialShape(shape)
            param = ops.parameter(pshape, ov.Type.bf16)
            other = in1 if transpose_in is in0 else in0
            m_np = ops.multiply(param, other)
            results["4_multiply_no_transpose"] = compile_output(
                core, m_np.output(0), "mul_no_transpose")
        except Exception as e:  # noqa: BLE001
            results["4_multiply_no_transpose"] = f"assemble-error: {e!r}"
    else:
        results["3_transpose_alone"] = "n/a (no Transpose input)"
        results["4_multiply_no_transpose"] = "n/a"

    print("\n==================== PROBE RESULTS ====================")
    for k in sorted(results):
        print(f"  {k:<26} : {results[k]}")
    print("======================================================")
    print("\nInterpretation:")
    print("  - 2 OK while 1 FAIL  -> bf16 eltwise kernel is the problem (fix: f32 RoPE).")
    print("  - 3 FAIL             -> the Transpose kernel itself won't build.")
    print("  - 4 OK while 1 FAIL  -> the Transpose->Multiply fusion is the problem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
