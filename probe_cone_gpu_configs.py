#!/usr/bin/env python
"""Probe which GPU config lets the failing RoPE cone build.

The broadcast-materialization fix did NOT help (the plugin re-folds the constants
and the OpenCL kernel still fails: ``CL_BUILD_PROGRAM_FAILURE -11``). That points
at the *precision* of the fused region, not its shapes. This probe compiles the
same ``cone.xml`` on the GPU under several precision/fusion configs and reports
which one builds -- telling us the real lever for the export fix.

Run on the Intel GPU machine:
    ./env/bin/python probe_cone_gpu_configs.py --xml cone_ir/cone.xml
"""

from __future__ import annotations

import argparse

import openvino as ov

CONFIGS = [
    ("baseline (default f16 GPU)", {}),
    ("INFERENCE_PRECISION_HINT=f32", {"INFERENCE_PRECISION_HINT": "f32"}),
    ("EXECUTION_MODE_HINT=ACCURACY", {"EXECUTION_MODE_HINT": "ACCURACY"}),
    ("both f32 + ACCURACY", {"INFERENCE_PRECISION_HINT": "f32", "EXECUTION_MODE_HINT": "ACCURACY"}),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="cone_ir/cone.xml")
    ap.add_argument("--device", default="GPU")
    args = ap.parse_args()

    core = ov.Core()
    print(f"[info] openvino {ov.__version__}")
    print(f"[info] devices : {core.available_devices}")
    if args.device in core.available_devices:
        try:
            print(f"[info]   {args.device}: {core.get_property(args.device, 'FULL_DEVICE_NAME')}")
        except Exception:  # noqa: BLE001
            pass

    model = core.read_model(args.xml)

    print(f"\n[probe] compiling {args.xml} on {args.device} under each config:\n")
    results = []
    for label, cfg in CONFIGS:
        try:
            core.compile_model(model, args.device, cfg)
            print(f"  OK   : {label}")
            results.append((label, True))
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()
            tail = next((ln for ln in reversed(msg) if ln.strip()), "")
            print(f"  FAIL : {label}\n         {tail[:160]}")
            results.append((label, False))

    print("\n================ SUMMARY ================")
    for label, ok in results:
        print(f"  {'OK  ' if ok else 'FAIL'}  {label}")
    print("\nIf an f32/ACCURACY config builds while baseline fails, the fix is to keep")
    print("this fused RoPE region in f32 (precision island) -- NOT a shape rewrite.")
    print("If ALL fail, the bug is structural (fused kernel) and needs a fusion barrier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
