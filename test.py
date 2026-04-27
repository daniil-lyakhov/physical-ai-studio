import shutil
from pathlib import Path

import nncf
import numpy as np
import openvino
import torch

from physicalai.benchmark import LiberoBenchmark
from physicalai.data import Observation
from physicalai.inference import InferenceModel
from physicalai.inference.runners import ActionChunking, SinglePass
from physicalai.policies.pi05 import Pi05


class InferenceModelPolicyWrapper:
    def __init__(self, inf_model: InferenceModel) -> None:
        self._inf_model = inf_model

    def select_action(self, observation: Observation) -> torch.Tensor:
        np_obs = observation.to_numpy().to_dict(flatten=False)
        np_inputs = {k: v for k, v in np_obs.items() if v is not None}
        action = self._inf_model.select_action(np_inputs)
        return torch.from_numpy(action)

    def reset(self) -> None:
        """Reset policy state for new episode."""
        self._inf_model.reset()


def compress_partitioned_model(
    source_dir: str | Path,
    output_dir: str | Path,
    mode: nncf.CompressWeightsMode = nncf.CompressWeightsMode.INT8_SYM,
) -> Path:
    """Compress prefix_lm and gemma_expert_decoder with INT8 weight compression.

    Copies image_embedder and action_output_head unchanged.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    core = openvino.Core()

    models_to_compress = {"prefix_lm", "gemma_expert_decoder"}
    model_names = ["image_embedder", "prefix_lm", "gemma_expert_decoder", "action_output_head"]

    for name in model_names:
        xml_path = source_dir / f"{name}.xml"
        ov_model = core.read_model(str(xml_path))

        if name in models_to_compress:
            print(f"Compressing {name} with {mode.name}...")
            compressed = nncf.compress_weights(ov_model, mode=mode)
        else:
            print(f"Skipping compression for {name}")
            compressed = ov_model

        openvino.save_model(compressed, str(output_dir / f"{name}.xml"))

    # Copy non-model artifacts (manifest, preprocessor config, tokenizer, etc.)
    for item in source_dir.iterdir():
        dest = output_dir / item.name
        if item.suffix in (".xml", ".bin") or dest.exists():
            continue
        if item.is_dir():
            shutil.copytree(str(item), str(dest))
        else:
            shutil.copy2(str(item), str(dest))

    # Report size reduction
    fp32_size = sum(f.stat().st_size for f in source_dir.glob("*.bin"))
    int8_size = sum(f.stat().st_size for f in output_dir.glob("*.bin"))
    print(f"Original size: {fp32_size / 1e6:.1f} MB")
    print(f"Compressed size: {int8_size / 1e6:.1f} MB")
    print(f"Compression ratio: {fp32_size / int8_size:.2f}x")

    return output_dir


if __name__ == "__main__":
    FP32_EXPORT_DIR = "pi05_libero_finetuned_hf_ov"
    INT8_EXPORT_DIR = "pi05_libero_finetuned_hf_ov_int8"

    # Step 0: Export partitioned FP32 model (if needed)
    if not Path(FP32_EXPORT_DIR).exists():
        policy = Pi05(
            pretrained_name_or_path="lerobot/pi05_libero_finetuned",
            n_action_steps=10,
            compile_model=False,
            dtype="bfloat16",
        )
        policy.to_openvino_partitioned(FP32_EXPORT_DIR)

    # Step 1: Compress prefix_lm and gemma_expert_decoder to INT8
    compress_partitioned_model(
        source_dir=FP32_EXPORT_DIR,
        output_dir=INT8_EXPORT_DIR,
        mode=nncf.CompressWeightsMode.INT8_SYM,
    )

    # Step 2: Evaluate compressed model on LIBERO
    benchmark = LiberoBenchmark(
        task_suite="libero_10",
        task_ids=[6],
        num_episodes=1,
        max_steps=10,
    )

    ov_model = InferenceModel(INT8_EXPORT_DIR, device="CPU", runner=ActionChunking(SinglePass()))
    ov_policy = InferenceModelPolicyWrapper(ov_model)
    ov_results = benchmark.evaluate(ov_policy)
    print(ov_results.summary())