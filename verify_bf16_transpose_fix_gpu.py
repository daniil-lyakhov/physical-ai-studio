#!/usr/bin/env python
"""Validate the bf16-Transpose GPU workaround on an existing IR without re-export.

    ./env/bin/python verify_bf16_transpose_fix_gpu.py /path/to/xr0.xml

Wraps every bf16 Transpose as Convert(bf16->f32)->Transpose(f32)->Convert(f32->bf16)
in memory, then compiles on GPU. Prints whether the model now builds.
"""

import sys
import openvino as ov
import openvino.opset13 as ops


def wrap_bf16_transposes(model: ov.Model) -> int:
    count = 0
    for tr in [op for op in model.get_ops() if op.get_type_name() == "Transpose"]:
        tin = tr.input_value(0)
        if tin.get_element_type() != ov.Type.bf16:
            continue
        consumers = list(tr.output(0).get_target_inputs())
        f32 = ops.convert(tin, destination_type="f32")
        tr.input(0).replace_source_output(f32.output(0))
        tr.validate_and_infer_types()
        back = ops.convert(tr.output(0), destination_type="bf16")
        for ti in consumers:
            ti.replace_source_output(back.output(0))
        count += 1
    model.validate_nodes_and_infer_types()
    return count


def main() -> int:
    xml = sys.argv[1] if len(sys.argv) > 1 else "xr0_ir/xr0.xml"
    print(f"[info] openvino {ov.__version__}")
    core = ov.Core()
    print(f"[info] devices: {core.available_devices}")
    model = core.read_model(xml)

    n = wrap_bf16_transposes(model)
    print(f"[info] wrapped {n} bf16 Transpose node(s)")

    print("[info] compiling patched model on GPU ...")
    try:
        core.compile_model(model, "GPU")
    except Exception as e:  # noqa: BLE001
        print("[RESULT] STILL FAILS on GPU:")
        print(repr(e))
        return 1
    print("[RESULT] GPU COMPILE OK -- workaround is sufficient.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
