#!/usr/bin/env python
"""Candidate GPU fix: materialize the broadcasting RoPE ``cos``/``sin`` constants.

The GPU ``clBuildProgram`` failure comes from the plugin fusing a *broadcasting*
``Multiply`` (a transposed ``q`` of shape ``[1,8,32,128]`` times a ``cos``/``sin``
constant of shape ``[1,1,32,128]`` that broadcasts over the heads axis) into the
downstream ``Add`` -- the fused-eltwise shape inference then collapses to scalar
shapes and aborts (see ``openvino_gpu_rope_fusion_bug.md``).

This rewrite finds every ``Multiply`` whose inputs are a ``Transpose`` and a
``Constant`` that *broadcasts* against the transpose output, and replaces the
constant with a fully-materialized constant of the transpose output shape (tiled
along the broadcast axes). The multiply becomes plain elementwise -- numerically
identical -- so the broadcast never reaches the plugin's fused shape inference.

Run on the Intel GPU machine (proves the fix compiles where the original failed):
    ./env/bin/python try_rope_broadcast_fix_gpu.py --xml cone_ir/cone.xml --devices GPU CPU
    ./env/bin/python try_rope_broadcast_fix_gpu.py --xml xr0_ir/xr0.xml  --devices GPU CPU --save xr0_ir/xr0.fixed.xml
"""

from __future__ import annotations

import argparse

import numpy as np
import openvino as ov
import openvino.opset13 as ops

BUILD_FAIL_MARKERS = ("clBuildProgram", "ProgramBuilder build failed", "broadcast_merge_into")


def is_build_failure(exc: Exception) -> bool:
    return any(m in str(exc) for m in BUILD_FAIL_MARKERS)


def _broadcasts(const_shape: list[int], target_shape: list[int]) -> bool:
    """True if const_shape numpy-broadcasts to target_shape AND differs from it."""
    if len(const_shape) != len(target_shape):
        return False
    diff = False
    for c, t in zip(const_shape, target_shape):
        if c == t:
            continue
        if c == 1:
            diff = True
        else:
            return False
    return diff


def materialize_rope_constants(model: ov.Model) -> int:
    """Expand every broadcasting rank-4 ``Multiply(x, Constant)`` constant to the
    other input's shape. Targets the RoPE combine ``(q*cos) + (rot(q)*sin)``:
    both multiplies broadcast a ``[1,1,S,D]`` constant over the heads axis of a
    ``[1,H,S,D]`` tensor and feed the same ``Add``. The ``x`` side is a
    ``Transpose`` for ``q*cos`` and a ``Concat`` (``rotate_half``) for ``rot(q)*sin``,
    so we key on the *broadcast* + rank-4, not the producer type. Returns the
    number of constants rewritten."""
    rewritten = 0
    for op in list(model.get_ops()):
        if op.get_type_name() != "Multiply":
            continue
        types = [op.input_value(i).get_node().get_type_name() for i in range(op.get_input_size())]
        if types.count("Constant") != 1:
            continue
        c_idx = types.index("Constant")
        t_idx = 1 - c_idx
        const = op.input_value(c_idx).get_node()
        other = op.input_value(t_idx)
        target_shape = list(other.get_shape())
        const_shape = list(const.output(0).get_shape())
        # Restrict to rank-4 RoPE-style tensors to avoid touching unrelated
        # broadcasts (e.g. RMSNorm weight multiplies).
        if len(target_shape) != 4:
            continue
        # Skip pure scalars ([1,1,1,1]): scalar-eltwise fuses fine on the GPU.
        # The bug is specifically a *mid-axis* broadcast (heads/seq 1 -> N) with a
        # matching trailing dim, i.e. cos/sin of shape [1,1,S,D] or [1,1,1,D].
        if int(np.prod(const_shape)) == 1:
            continue
        if const_shape[-1] != target_shape[-1] or target_shape[-1] == 1:
            continue
        if not _broadcasts(const_shape, target_shape):
            continue

        et = const.output(0).get_element_type()
        # Read constant data (works for f16/f32; bf16 via the raw view if present).
        try:
            data = const.data  # numpy view; ml_dtypes provides bf16
        except Exception:  # noqa: BLE001
            print(f"  [skip] {const.get_friendly_name()}: cannot read constant data")
            continue
        expanded = np.ascontiguousarray(np.broadcast_to(data, target_shape))
        new_const = ops.constant(expanded, et)
        new_const.set_friendly_name(const.get_friendly_name() + "_full")
        op.input(c_idx).replace_source_output(new_const.output(0))
        op.validate_and_infer_types()
        rewritten += 1
        print(f"  [fix] {op.get_friendly_name()}: constant {const_shape} -> {target_shape}")
    if rewritten:
        model.validate_nodes_and_infer_types()
    return rewritten


def try_device(core: ov.Core, model: ov.Model, device: str) -> None:
    try:
        core.compile_model(model, device)
    except Exception as e:  # noqa: BLE001
        kind = "BUILD-FAIL" if is_build_failure(e) else "other error"
        print(f"  [{device}] compile FAILED -- {kind}:\n      {repr(e)[:400]}")
        return
    print(f"  [{device}] compile OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="cone_ir/cone.xml")
    ap.add_argument("--devices", nargs="+", default=["GPU", "CPU"])
    ap.add_argument("--save", default="", help="optional path to save the fixed IR")
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

    print("\n[before] compiling the unmodified IR:")
    for device in args.devices:
        if device not in core.available_devices:
            print(f"  [{device}] not available, skipping")
            continue
        try_device(core, model, device)

    print("\n[rewrite] materializing broadcasting RoPE constants:")
    n = materialize_rope_constants(model)
    print(f"[rewrite] {n} constant(s) expanded")

    if args.save and n:
        ov.save_model(model, args.save, compress_to_fp16=False)
        print(f"[info] saved fixed IR -> {args.save}")

    print("\n[after] compiling the rewritten IR:")
    for device in args.devices:
        if device not in core.available_devices:
            print(f"  [{device}] not available, skipping")
            continue
        try_device(core, model, device)

    print("\nIf GPU was BUILD-FAIL [before] and OK [after], the broadcast-materialization")
    print("fix works and can be folded into xr0_to_openvino.py's post-export pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
