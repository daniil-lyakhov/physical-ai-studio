import os

import openvino as ov
import torch
from openvino.preprocess import PrePostProcessor

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

policy.to_openvino("xr0_ir", input_sample=input_sample)

# The DiT action head runs in bf16, so the exported ``action`` output is bf16 --
# which NumPy cannot represent, so the Runtime OpenVINO adapter (``np.array(...)``)
# would fail. Bake an f32 cast into the saved IR so the native ``InferenceModel``
# reads the action back as plain f32.
_EXPORT_XML = "xr0_ir_gpu_friendly/xr0.xml"
_core = ov.Core()
_model = _core.read_model(_EXPORT_XML)
_ppp = PrePostProcessor(_model)
_ppp.output("action").tensor().set_element_type(ov.Type.f32)
_model = _ppp.build()

# ``read_model`` memory-maps the 9 GB source ``xr0.bin`` and ``save_model``
# streams the constant weights from that live mapping *while* it writes. Saving
# back to the SAME path would overwrite the file the mapping still points at and
# crash with a bus error (SIGBUS). Write to a temp path first, drop the mapping,
# then atomically replace the originals (the IR ``.bin`` is located by the
# ``.xml`` basename, so both must be renamed together).
_TMP_XML = "xr0_ir/xr0.f32.xml"
_TMP_BIN = "xr0_ir/xr0.f32.bin"
ov.save_model(_model, _TMP_XML)
del _model  # release the mmap on the source xr0.bin before overwriting it
os.replace(_TMP_XML, _EXPORT_XML)
os.replace(_TMP_BIN, "xr0_ir/xr0.bin")
