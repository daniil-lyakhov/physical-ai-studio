import os

import openvino as ov
import torch

from physicalai.policies import XR0
from physicalai.export import ExportBackend
from physicalai.policies.xr0.pretrained_utils import extract_xr0_dataset_stats

CHECKPOINT = "XiaomiRobotics/Xiaomi-Robotics-0-LIBERO"
# Fixed, right-padded prompt length the exported graph is baked for. The image
# tokens are a constant prefix and only the task text varies, so the baked MRoPE
# ``position_ids`` / image-token positions stay valid for any prompt padded (on
# the right, masked by ``attention_mask``) to this length. Derived from the
# policy config below so it matches the ``seq_len`` the manifest's
# ``XR0InferencePreprocessor`` right-pads to at inference time.

# The checkpoint only carries *action* normalization stats, so the auto-generated
# export sample would be missing the ``state``/image tensors the preprocessor needs
# (KeyError: 'state'). Start from the checkpoint's action stats and add the
# observation schema so ``inputs_schema`` / ``sample_input`` are complete.
# NOTE: verify these against your robot/dataset (LIBERO defaults shown).
stats = extract_xr0_dataset_stats(CHECKPOINT) or {}
stats["observation.state"] = {"name": "state", "type": "STATE", "shape": (8,)}
stats["observation.images.base"] = {"name": "images.base", "type": "VISUAL", "shape": (3, 256, 256)}
stats["observation.images.wrist_left"] = {
    "name": "images.wrist_left",
    "type": "VISUAL",
    "shape": (3, 256, 256),
}

policy = XR0(
    pretrained_name_or_path=CHECKPOINT,
    dataset_stats=stats,
    vlm_attn_implementation="sdpa",
)
policy.get_supported_export_backends = lambda: [ExportBackend.TORCH, ExportBackend.OPENVINO]

# Pad to the same fixed length the manifest's ``XR0InferencePreprocessor`` uses at
# inference time, so the exported static graph and the native pipeline agree.
SEQ_LEN = policy.config.tokenizer_max_length

# Build a representative processed sample and right-pad it to SEQ_LEN. The whole
# vision tower, the 3D MRoPE ``position_ids`` and the image-token scatter now live
# INSIDE the exported graph; ``prepare_ingraph_export`` bakes the fixed image
# geometry as constants so those (otherwise non-traceable) paths convert cleanly.
processed = policy._preprocessor(policy.sample_input)

pad_id = policy._preprocessor.processor.tokenizer.pad_token_id or 0
cur_len = processed["input_ids"].shape[1]
if cur_len > SEQ_LEN:
    msg = f"Sample prompt ({cur_len} tokens) exceeds SEQ_LEN={SEQ_LEN}; increase SEQ_LEN."
    raise ValueError(msg)
pad = SEQ_LEN - cur_len
if pad:
    processed["input_ids"] = torch.nn.functional.pad(processed["input_ids"], (0, pad), value=pad_id)
    processed["attention_mask"] = torch.nn.functional.pad(processed["attention_mask"], (0, pad), value=0)

# Bake the fixed geometry into the VLM shim (enables in-graph export mode).
policy.prepare_ingraph_export(processed)

# The export sample EXCLUDES ``image_grid_thw``: the shim supplies the baked
# constant internally, and keeping it as a runtime input would reintroduce the
# non-traceable ``tensor.tolist()`` vision geometry. The self-contained graph
# consumes {input_ids, attention_mask, pixel_values, state} -> action.
input_sample = {
    name: tensor
    for name, tensor in processed.items()
    if name != "image_grid_thw" and isinstance(tensor, torch.Tensor)
}

# Every RMSNorm in the assembled model reduces with
# ``hidden_states.pow(2).mean(-1, keepdim=True)``. The OpenVINO PyTorch frontend
# mis-materializes that *negative* reduction axis into a garbage ReduceMean axis
# constant, so loading the exported IR fails with
# "Axis <huge> out of the tensor rank range". Patch every RMSNorm class present
# to reduce over a *positive, static* axis (``ndim - 1`` is a concrete int during
# tracing) so the frontend emits a valid axis constant. Numerically identical to
# the original. Both the DiT action head (``Qwen2RMSNorm``) and the Qwen3-VL text
# model (``Qwen3VLTextRMSNorm``) use this pattern, so both must be patched.
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm  # noqa: E402
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm  # noqa: E402


def _rmsnorm_forward_positive_axis(self: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    axis = hidden_states.dim() - 1  # concrete positive int -> clean ReduceMean axis
    variance = hidden_states.pow(2).mean(axis, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)


Qwen2RMSNorm.forward = _rmsnorm_forward_positive_axis
Qwen3VLTextRMSNorm.forward = _rmsnorm_forward_positive_axis

# The DiT action head runs in bf16, but ``XR0Model._run`` casts the ``action``
# output to f32 while in in-graph export mode, so ``to_openvino`` emits a single,
# self-consistent f32-output IR that the Runtime OpenVINO adapter can read with
# NumPy directly -- no fragile post-hoc re-save of the multi-GB ``.bin``.
policy.to_openvino("xr0_ir", input_sample=input_sample)

# GPU-friendliness pass: the Qwen3-VL attention-mask builder lowers to a GatherND
# on a *boolean* (u8) tensor -- ``attention_mask`` (i64) -> Convert(bool) ->
# GatherND -> LogicalAnd. The Intel GPU OpenVINO plugin has no u8 GatherND kernel
# ("No layout format available for gathernd ... data_type: u8"), so it fails to
# compile (CPU has the kernel and loads fine). Rewrite the gather to run on i32
# -- which the GPU supports -- and cast the result back to bool, leaving the
# downstream LogicalAnd untouched. Numerically identical: the mask is only 0/1.
import openvino.opset13 as _ops  # noqa: E402

_XML = "xr0_ir/xr0.xml"
_core = ov.Core()
_ovm = _core.read_model(_XML)
for _gnd in [op for op in _ovm.get_ops() if op.get_type_name() == "GatherND"]:
    _data = _gnd.input_value(0)
    if _data.get_element_type() != ov.Type.boolean:
        continue
    _consumers = list(_gnd.output(0).get_target_inputs())
    _d32 = _ops.convert(_data, destination_type="i32")
    _gnd.input(0).replace_source_output(_d32.output(0))
    _gnd.validate_and_infer_types()
    _back = _ops.convert(_gnd.output(0), destination_type="boolean")
    for _ti in _consumers:
        _ti.replace_source_output(_back.output(0))

# GPU-friendliness pass 2: the Intel GPU OpenVINO plugin cannot build the OpenCL
# kernel for a bf16 4-D ``Transpose`` -- the attention ``.transpose(1, 2)`` permute
# [0,2,1,3] on [1,32,8,128] -- ``clBuildProgram`` fails (and segfaults under
# ``OV_GPU_Verbose=1``). The identical permute in f16/f32 builds fine, so this is a
# plugin bug specific to the bf16 permute kernel (see
# ``openvino_gpu_bf16_transpose_bug.md`` and ``openvino_gpu_bf16_transpose_repro.py``).
# Mitigate ONLY that op by wrapping every bf16 Transpose as
# ``Convert(bf16->f32) -> Transpose(f32) -> Convert(f32->bf16)``. A transpose is pure
# data movement and every bf16 value is exactly representable in f32, so the
# round-trip is numerically identical -- the model stays bit-for-bit bf16, preserving
# parity with the Pi0.5 export while selecting the working f32 permute kernel.
for _tr in [op for op in _ovm.get_ops() if op.get_type_name() == "Transpose"]:
    _tin = _tr.input_value(0)
    if _tin.get_element_type() != ov.Type.bf16:
        continue
    _consumers = list(_tr.output(0).get_target_inputs())
    _f32 = _ops.convert(_tin, destination_type="f32")
    _tr.input(0).replace_source_output(_f32.output(0))
    _tr.validate_and_infer_types()
    _back = _ops.convert(_tr.output(0), destination_type="bf16")
    for _ti in _consumers:
        _ti.replace_source_output(_back.output(0))

_ovm.validate_nodes_and_infer_types()

# ``read_model`` memory-maps the multi-GB ``xr0.bin`` and ``save_model`` streams
# those weights from the live mapping while writing. Write to a temp path first,
# drop the mapping (``del``), then atomically replace the originals (the ``.bin``
# is located by the ``.xml`` basename, so both must be renamed together).
# ``compress_to_fp16=False`` preserves the exported weight precision as-is.
_TMP_XML = "xr0_ir/xr0.gpu.xml"
_TMP_BIN = "xr0_ir/xr0.gpu.bin"
ov.save_model(_ovm, _TMP_XML, compress_to_fp16=False)
del _ovm  # release the mmap on the source xr0.bin before overwriting it
os.replace(_TMP_XML, _XML)
os.replace(_TMP_BIN, "xr0_ir/xr0.bin")
