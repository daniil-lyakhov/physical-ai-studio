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
