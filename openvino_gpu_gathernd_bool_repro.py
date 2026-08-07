#!/usr/bin/env python
"""Minimal reproducer: OpenVINO Intel GPU plugin cannot compile a `GatherND`
whose `data` input is boolean (u8) -- it reports no available layout/kernel for
`gathernd` with `data_type: u8`. The identical `GatherND` on i32/f16/f32 data
compiles fine.

This is the pattern emitted by the Qwen3-VL attention-mask builder:
    attention_mask (i64) -> Convert(boolean) -> GatherND -> LogicalAnd
which makes any such bf16/f16 model fail to compile on the Intel GPU (CPU has
the kernel and loads fine).

Usage:
    python openvino_gpu_gathernd_bool_repro.py

Requires only: openvino, numpy. No model file needed.
"""

import numpy as np
import openvino as ov
import openvino.opset13 as ops

# data[6, 6] gathered by indices[3, 2] -> output[3] of the data's element type.
DATA_SHAPE = [6, 6]
INDICES = np.array([[0, 0], [1, 2], [5, 3]], dtype=np.int32)

DTYPES = {
    "boolean": ov.Type.boolean,   # u8 under the hood -- the failing case
    "i32": ov.Type.i32,
    "f16": ov.Type.f16,
    "f32": ov.Type.f32,
}


def make_gathernd_model(dtype: ov.Type) -> ov.Model:
    data = ops.parameter(ov.PartialShape(DATA_SHAPE), dtype)
    indices = ops.constant(INDICES, ov.Type.i32)
    gnd = ops.gather_nd(data, indices, batch_dims=0)
    return ov.Model([ops.result(gnd)], [data], f"gathernd_{dtype}")


def try_compile(core: ov.Core, model: ov.Model) -> tuple[bool, str]:
    try:
        core.compile_model(model, "GPU")
    except Exception as e:  # noqa: BLE001
        return False, str(e).strip().replace("\n", " ")
    return True, ""


def main() -> int:
    print(f"OpenVINO version : {ov.__version__}")
    core = ov.Core()
    print(f"Available devices: {core.available_devices}")
    if not any("GPU" in d for d in core.available_devices):
        print("FATAL: no Intel GPU visible to OpenVINO.")
        return 2

    try:
        name = core.get_property("GPU", "FULL_DEVICE_NAME")
        print(f"GPU device       : {name}")
    except Exception as e:  # noqa: BLE001
        print(f"(could not query GPU device name: {e})")

    print(f"\nGatherND under test: data{DATA_SHAPE} gathered by indices"
          f"{list(INDICES.shape)} (batch_dims=0)\n")

    failures = 0
    for label, dtype in DTYPES.items():
        ok, err = try_compile(core, make_gathernd_model(dtype))
        print(f"  [{label:>7}] GatherND -> {'OK  ' if ok else 'FAIL'}")
        if not ok:
            failures += 1
            print(f"           {err[:200]}")

    print()
    if failures:
        print("REPRODUCED: GatherND fails to compile for at least one dtype on GPU.")
        print("Expected: boolean FAIL, i32/f16/f32 OK.")
    else:
        print("Not reproduced on this machine: all dtypes compiled.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
