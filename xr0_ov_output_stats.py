#!/usr/bin/env python
"""Run an exported XR0 IR on fixed input and print per-output stats.

Deterministic, so run under different OpenVINO versions and diff the output.

    ./env/bin/python xr0_ov_output_stats.py xr0_ir/xr0.xml
"""

import sys

import numpy as np
import openvino as ov

SEED = 0
VALID_PROMPT_LEN = 190


def build_inputs(model):
    rng = np.random.default_rng(SEED)
    inputs = {}
    for port in model.inputs:
        name = port.get_any_name()
        shape = tuple(port.get_partial_shape().to_shape())
        dtype = np.dtype(port.get_element_type().to_dtype())
        if np.issubdtype(dtype, np.integer):
            if "mask" in name.lower():
                arr = np.zeros(shape, dtype=dtype)
                arr.reshape(-1)[:VALID_PROMPT_LEN] = 1
            else:
                arr = rng.integers(0, 1000, size=shape).astype(dtype)
        else:
            arr = rng.standard_normal(size=shape).astype(dtype)
        inputs[name] = arr
    return inputs


def main():
    model_path = sys.argv[1]
    core = ov.Core()
    model = core.read_model(model_path)
    config = {"INFERENCE_NUM_THREADS": "1", "NUM_STREAMS": "1", "EXECUTION_MODE_HINT": "ACCURACY"}
    compiled = core.compile_model(model, device_name="CPU", config=config)

    result = compiled(build_inputs(model))

    print(f"openvino {ov.get_version()}  model {model_path}")
    for port in compiled.outputs:
        a = np.asarray(result[port]).astype(np.float64)
        print(
            f"{port.get_any_name()} shape={a.shape} "
            f"min={a.min():+.9e} max={a.max():+.9e} mean={a.mean():+.9e} std={a.std():+.9e}",
        )


if __name__ == "__main__":
    main()
