#!/usr/bin/env python
"""Reproducer: RoPE (`apply_rotary_pos_emb`) on a *transposed* query tensor fails
to build on the Intel GPU OpenVINO plugin -- in **fp16 AND bf16**.

Background
----------
XR0's DiT action-head attention lays q/k out as ``[batch, seq, heads, head_dim]``
and does ``q.transpose(1, 2)`` -> ``[batch, heads, seq, head_dim]`` before RoPE:

    q = q_bshd.transpose(1, 2)                 # Transpose perm [0, 2, 1, 3]
    q_embed = (q * cos) + (rotate_half(q) * sin)

In the exported IR this is exactly the failing node found by graph bisection:

    node_mul_605  Multiply  in: Transpose[f16], Constant[f16]  out: f16 [1,8,32,128]
                  <- Transpose node_transpose_290
                  <- Constant  unsqueeze_282  (baked cos)
                  -> Add       node_add_426

The Intel GPU plugin fails to build the OpenCL kernel for an elementwise op whose
input is a permuted (transposed) tensor. This was first blamed on bf16, but it
reproduces identically in fp16 -- the plugin executes both as f16, so switching
the export precision bf16 -> f16 does NOT fix it.

What this script does
---------------------
Builds the minimal RoPE-on-transposed-q block and, for each requested precision
and export route, compiles it on the requested device(s), reporting OK/FAIL and
the verbatim error. Run it on the Intel GPU server.

Usage
-----
    # full matrix on GPU and CPU, both export routes, all precisions
    python openvino_gpu_rope_transpose_repro.py

    # just the headline case: fp16 on GPU via the ONNX route (XR0's real route)
    python openvino_gpu_rope_transpose_repro.py --devices GPU --dtypes float16 --routes onnx

Requires: openvino, torch. Needs an Intel GPU for the GPU rows.
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import openvino as ov
import torch
import torch.nn as nn

# Shapes taken straight from the failing node: out [1, 8, 32, 128].
BATCH, SEQ, HEADS, HEAD_DIM = 1, 32, 8, 128

_TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
_OV_DTYPE = {
    "float16": ov.Type.f16,
    "bfloat16": ov.Type.bf16,
    "float32": ov.Type.f32,
}


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Byte-for-byte the HF ``rotate_half``."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RoPEOnTransposedQ(nn.Module):
    """RoPE applied to a query that is produced by a ``transpose(1, 2)``.

    ``cos``/``sin`` are baked constants (buffers) exactly like XR0's in-graph
    export, so in the IR the elementwise multiply has one Transpose input and one
    Constant input -- reproducing ``node_mul_605``.
    """

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        pos = torch.arange(SEQ, dtype=torch.float32)
        inv = 1.0 / (10000.0 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
        freqs = torch.outer(pos, inv)  # [seq, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq, head_dim]
        cos = emb.cos()[None, None]  # [1, 1, seq, head_dim] -> broadcasts over heads
        sin = emb.sin()[None, None]
        self.register_buffer("cos", cos.to(dtype))
        self.register_buffer("sin", sin.to(dtype))

    def forward(self, q_bshd: torch.Tensor) -> torch.Tensor:
        # [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        q = q_bshd.transpose(1, 2)  # Transpose perm [0, 2, 1, 3]
        return (q * self.cos) + (rotate_half(q) * self.sin)


def _example(dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(BATCH, SEQ, HEADS, HEAD_DIM, dtype=dtype)


def export_native(dtype_name: str) -> ov.Model:
    """Export via the OpenVINO PyTorch frontend (handles bf16)."""
    dt = _TORCH_DTYPE[dtype_name]
    model = RoPEOnTransposedQ(dt).eval()
    # Lock the input to a fully static shape/precision so the whole subgraph
    # stays in the target type and rank (no spurious dynamic batch/head dims).
    ovm = ov.convert_model(
        model,
        example_input=_example(dt),
        input=[(ov.PartialShape([BATCH, SEQ, HEADS, HEAD_DIM]), _OV_DTYPE[dtype_name])],
    )
    ovm.validate_nodes_and_infer_types()
    return ovm


def export_onnx(dtype_name: str) -> ov.Model:
    """Export via ONNX (XR0 uses ``via_onnx=True``). bf16 is not supported by the
    ONNX exporter, so this route is fp16/fp32 only."""
    dt = _TORCH_DTYPE[dtype_name]
    model = RoPEOnTransposedQ(dt).eval()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    torch.onnx.export(
        model,
        (_example(dt),),
        onnx_path,
        opset_version=18,
        input_names=["q"],
        output_names=["q_embed"],
        dynamo=False,  # legacy TorchScript exporter: no onnxscript dependency
    )
    ovm = ov.convert_model(onnx_path)
    os.unlink(onnx_path)
    return ovm


ROUTES = {"native": export_native, "onnx": export_onnx}


def _mul_input_types(ovm: ov.Model) -> str:
    for op in ovm.get_ops():
        if op.get_type_name() != "Multiply":
            continue
        producers = [op.input_value(i).get_node().get_type_name() for i in range(op.get_input_size())]
        if "Transpose" in producers:
            ins = ", ".join(str(op.input_value(i).get_element_type()) for i in range(op.get_input_size()))
            out = str(op.output(0).get_element_type())
            shape = str(op.output(0).get_partial_shape())
            return f"Multiply(in=[{ins}]) -> {out} {shape}  producers={producers}"
    return "<no Multiply-fed-by-Transpose found>"


def try_device(ovm: ov.Model, device: str, dtype_name: str) -> bool:
    """Return the *compile* (clBuildProgram) verdict. Inference is reported as
    secondary info and never flips the verdict -- the GPU kernel-build failure
    we are hunting happens at compile time."""
    core = ov.Core()
    cfg = {}
    # Force the plugin to keep the target inference precision (don't let it
    # silently upcast f16 -> f32 and hide the kernel-build failure).
    if dtype_name in ("float16", "bfloat16"):
        cfg["INFERENCE_PRECISION_HINT"] = "f16" if dtype_name == "float16" else "bf16"
    try:
        compiled = core.compile_model(ovm, device, cfg)
    except Exception as e:  # noqa: BLE001
        print(f"    [{device}] compile FAILED: {repr(e)[:280]}")
        return False
    print(f"    [{device}] compile OK")
    try:
        q = np.random.randn(BATCH, SEQ, HEADS, HEAD_DIM).astype(np.float32)
        out = compiled(q)
        arr = next(iter(out.values()))
        print(f"    [{device}] infer OK, out shape {arr.shape}, dtype {arr.dtype}")
    except Exception as e:  # noqa: BLE001
        print(f"    [{device}] infer note (compile still OK): {repr(e)[:200]}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="+", default=["GPU", "CPU"])
    ap.add_argument("--dtypes", nargs="+", default=["float16", "bfloat16", "float32"])
    ap.add_argument("--routes", nargs="+", default=["onnx", "native"])
    args = ap.parse_args()

    core = ov.Core()
    print(f"OpenVINO : {ov.__version__}")
    print(f"torch    : {torch.__version__}")
    print(f"devices  : {core.available_devices}")
    for d in args.devices:
        if d in core.available_devices:
            try:
                print(f"  {d}: {core.get_property(d, 'FULL_DEVICE_NAME')}")
            except Exception:  # noqa: BLE001
                pass

    results: dict[tuple[str, str, str], bool] = {}
    for route in args.routes:
        for dtype_name in args.dtypes:
            if route == "onnx" and dtype_name == "bfloat16":
                print(f"\n### route={route} dtype={dtype_name}: SKIP (ONNX exporter has no bf16)")
                continue
            print(f"\n### route={route} dtype={dtype_name}")
            try:
                ovm = ROUTES[route](dtype_name)
            except Exception as e:  # noqa: BLE001
                print(f"    export FAILED: {repr(e)[:280]}")
                continue
            print(f"    culprit node: {_mul_input_types(ovm)}")
            for device in args.devices:
                if device not in core.available_devices:
                    print(f"    [{device}] not available, skipping")
                    continue
                results[(route, dtype_name, device)] = try_device(ovm, device, dtype_name)

    print("\n=========== COMPILE (clBuildProgram) SUMMARY ===========")
    for (route, dtype_name, device), ok in sorted(results.items()):
        print(f"  {route:6s} {dtype_name:9s} {device:4s} : {'OK' if ok else 'FAIL'}")
    print("\nExpected on the Intel GPU: fp16 AND bf16 FAIL to build (clBuildProgram),")
    print("fp32 builds, and CPU builds every precision. That proves the failure is")
    print("the transposed-input elementwise kernel, NOT the bf16 precision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
