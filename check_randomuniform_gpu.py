#!/usr/bin/env python
"""Test whether RandomUniform is what makes the XR0 IR fail to compile on an Intel GPU.

Usage (on the machine WITH the Intel GPU):

    ./env/bin/python check_randomuniform_gpu.py /path/to/xr0.xml

It will:
  1. Read the IR you point it at.
  2. Rewire every consumer of RandomUniform to a fixed constant (RNG removed).
  3. Save the variant next to the original as <name>_norand.xml.
  4. Compile ONLY the variant on GPU and report success/failure.

Compile the ORIGINAL separately (it may segfault) — see the note printed at the end.
"""

import sys
import numpy as np
import openvino as ov
import openvino.opset13 as ops


def op_counts(model):
    counts = {}
    for n in model.get_ops():
        t = n.get_type_name()
        counts[t] = counts.get(t, 0) + 1
    return counts


def main() -> int:
    xml_path = sys.argv[1] if len(sys.argv) > 1 else "xr0_ir/xr0.xml"
    print(f"[info] openvino {ov.__version__}")
    print(f"[info] reading model: {xml_path}")

    core = ov.Core()
    print(f"[info] available devices: {core.available_devices}")
    if "GPU" not in "".join(core.available_devices):
        print("[FATAL] no GPU device visible to OpenVINO on this machine.")
        return 2

    model = core.read_model(xml_path)

    ru_nodes = [n for n in model.get_ops() if n.get_type_name() == "RandomUniform"]
    print(f"[info] RandomUniform nodes found: {len(ru_nodes)}")
    if not ru_nodes:
        print("[warn] no RandomUniform in this IR — nothing to rewire. "
              "Compile the model directly on GPU to check.")

    rng = np.random.default_rng(0)
    for ru in ru_nodes:
        out = ru.output(0)
        pshape = out.get_partial_shape()
        if pshape.is_dynamic:
            print(f"[FATAL] RandomUniform '{ru.get_friendly_name()}' has dynamic shape "
                  f"{pshape}; cannot substitute a static constant. Give it a fixed shape.")
            return 3
        shape = [d.get_length() for d in pshape]
        # Values kept strictly inside (0, 1) so any downstream Log()/Sqrt() stays finite.
        vals = rng.uniform(0.05, 0.95, size=shape).astype(np.float32)
        et = out.get_element_type()
        const = ops.constant(vals.astype(np.float32), ov.Type.f32)
        # Match the RandomUniform output dtype if it isn't f32.
        src = const.output(0)
        if str(et) != str(ov.Type.f32):
            src = ops.convert(const, et).output(0)
        targets = list(out.get_target_inputs())
        for ti in targets:
            ti.replace_source_output(src)
        print(f"[info] rewired '{ru.get_friendly_name()}' shape={shape} dtype={et} "
              f"-> constant ({len(targets)} consumers)")

    model.validate_nodes_and_infer_types()
    left = [n for n in model.get_ops() if n.get_type_name() == "RandomUniform"]
    print(f"[info] RandomUniform remaining after rewire: {len(left)}")

    variant_path = xml_path.rsplit(".xml", 1)[0] + "_norand.xml"
    ov.save_model(model, variant_path, compress_to_fp16=False)
    print(f"[info] saved variant: {variant_path}")

    print("[info] compiling NO-RANDOM variant on GPU ...")
    try:
        core.compile_model(model, "GPU")
    except Exception as e:  # noqa: BLE001 - we want the full message
        print("[RESULT] NO-RANDOM variant FAILED to compile on GPU:")
        print(repr(e))
        print("\n=> RandomUniform is NOT the (only) problem. Something else in the "
              "graph breaks clBuildProgram.")
        return 1

    print("[RESULT] NO-RANDOM variant COMPILED SUCCESSFULLY on GPU.")
    print("\n=> This confirms RandomUniform (the in-graph noise sampler) is what "
          "crashes the Intel GPU OpenCL compiler.")
    print("\nNext, confirm the ORIGINAL fails on THIS machine by running separately:\n"
          f"    ./env/bin/python -c \"import openvino as ov; c=ov.Core(); "
          f"c.compile_model(c.read_model('{xml_path}'),'GPU'); print('ORIG OK')\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
