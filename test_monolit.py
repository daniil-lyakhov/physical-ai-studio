import gc
import random
import shutil
from pathlib import Path

import nncf
import numpy as np
import openvino
import torch
from tqdm import tqdm
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters, GroupSizeFallbackMode

from physicalai.benchmark import LiberoBenchmark
from physicalai.data import Observation
from physicalai.gyms import LiberoGym
from physicalai.inference import InferenceModel
from physicalai.inference.adapters.openvino_partitioned import PartitionedOpenVINOAdapter
from physicalai.inference.runners import ActionChunking, SinglePass

_CALIBRATION_TASK_SUITE = "libero_10"
_CALIBRATION_SAMPLES_PER_TASK = 32
_CALIBRATION_NUM_TASKS = 10
_CALIBRATION_SAMPLES = _CALIBRATION_SAMPLES_PER_TASK * _CALIBRATION_NUM_TASKS


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
    device: str = "CPU",
) -> Path:
    """Compress prefix_lm and gemma_expert_decoder with INT4 data-aware compression.

    Uses real calibration data from the LIBERO dataset to enable AWQ and
    scale estimation for better INT4 accuracy.

    Args:
        source_dir: Directory with FP32 partitioned models.
        output_dir: Directory for compressed output.
        device: OpenVINO device for calibration inference.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    core = openvino.Core()


    print("Copy tokenizer")
    ov_model = core.read_model(str(source_dir / "tokenizer.xml"))
    openvino.save_model(ov_model, str(output_dir / "tokenizer.xml"))
    print("Compressing pi05")
    ov_model = core.read_model(str(source_dir / "pi05.xml"))
    compressed = nncf.compress_weights(
        ov_model,
        mode=nncf.CompressWeightsMode.INT8_SYM,
        #group_size=128,
        #dataset=nncf.Dataset(prefix_lm_inputs),
        #awq=True,
        #scale_estimation=True,
        #subset_size=_CALIBRATION_SAMPLES,
        advanced_parameters=AdvancedCompressionParameters(
            calibration_device=device,
            group_size_fallback_mode=GroupSizeFallbackMode.ADJUST,
        ),
    )
    openvino.save_model(compressed, str(output_dir / "pi05.xml"))

    # --- gemma_expert_decoder: INT4 with AWQ + scale estimation ---
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
    int4_size = sum(f.stat().st_size for f in output_dir.glob("*.bin"))
    print(f"Original size: {fp32_size / 1e6:.1f} MB")
    print(f"Compressed size: {int4_size / 1e6:.1f} MB")
    print(f"Compression ratio: {fp32_size / int4_size:.2f}x")

    return output_dir

def print_stats(latencies: list) -> None:

    arr = np.array(latencies) * 1000  # ms
    print(f"\n{'=' * 60}")
    print(f"{'=' * 60}")
    print(f"  Total calls : {len(arr)}")
    print(f"  Total time  : {arr.sum() / 1000:.2f} s")
    print(f"  Mean        : {arr.mean():.2f} ms")
    print(f"  Std         : {arr.std():.2f} ms")
    print(f"  Min         : {arr.min():.2f} ms")
    print(f"  Max         : {arr.max():.2f} ms")
    print(f"  Median      : {np.median(arr):.2f} ms")
    print(f"  P95         : {np.percentile(arr, 95):.2f} ms")
    print(f"  P99         : {np.percentile(arr, 99):.2f} ms")
    print(f"  Throughput  : {1000 / arr.mean():.1f} actions/s")

if __name__ == "__main__":
    FP32_EXPORT_DIR = "pi05_libero_finetuned_hf_ov_monolit"
    INT4_EXPORT_DIR = "pi05_libero_finetuned_hf_ov_int4_monolit"
    INT8_EXPORT_DIR = "pi05_libero_finetuned_hf_ov_int8_monolit"
    current_export_dir = INT8_EXPORT_DIR 

    # Step 1: Export partitioned FP32 model (if needed)
    if not Path(FP32_EXPORT_DIR).exists():
        from physicalai.policies.pi05 import Pi05

        policy = Pi05(
            pretrained_name_or_path="lerobot/pi05_libero_finetuned",
            n_action_steps=10,
            compile_model=False,
            dtype="bfloat16",
        )
        policy.to_openvino(FP32_EXPORT_DIR)

    DEVICE = "GPU.0"

    # Step 2: Compress with INT4 data-aware algorithms
    if not Path(current_export_dir).exists():
        compress_partitioned_model(
            source_dir=FP32_EXPORT_DIR,
            output_dir=current_export_dir,
            device=DEVICE,
        )

    # Step 3: Evaluate compressed model on LIBERO

    for suite in ["libero_10" ]:
        benchmark = LiberoBenchmark(
            task_suite=suite,
            #task_ids=[6],
            num_episodes=5,
            max_steps=60,
        )
        runner = ActionChunking(SinglePass())
        ov_model = InferenceModel(current_export_dir, device=DEVICE, runner=runner)
        ov_policy = InferenceModelPolicyWrapper(ov_model)
        ov_results = benchmark.evaluate(ov_policy)
        print(ov_results.summary())
        print_stats(runner.elapsed_times)

        # Close all gym environments to release MuJoCo/rendering resources
        for gym in benchmark.gyms:
            gym.close()
        del ov_results, ov_policy, ov_model, benchmark
        gc.collect()
