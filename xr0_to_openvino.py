from physicalai.policies import XR0
from physicalai.policies.xr0.pretrained_utils import extract_xr0_dataset_stats

CHECKPOINT = "XiaomiRobotics/Xiaomi-Robotics-0-LIBERO"

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
    dtype="bfloat16",
    # Execute only 10 of the 30 predicted actions before replanning. This makes
    # ``chunk_size (30) != n_action_steps (10)``, so the export emits an
    # ``action_chunk_trimmer`` into the manifest and the deployed Runtime replans
    # every 10 steps -- matching the working eager LIBERO eval (liber_xr0_.py).
    # Executing all 30 open-loop overshoots the target and collapses success ~0%.
    #n_action_steps=10,
    num_inference_steps=1
)

# All XR0 export preparation and workarounds live inside ``XR0.to_openvino``:
# right-padding the sample to the graph length, baking the vision geometry,
# rebuilding the MRoPE ``position_ids`` in-graph, installing the OpenVINO-friendly
# RMSNorm, and the post-export GPU-friendly ``GatherND`` rewrite.
policy.to_openvino("xr0_ir_1step")
