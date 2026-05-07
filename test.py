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


def _collect_gym_observations(
    inf_model: InferenceModel,
    task_suite: str,
    num_tasks: int,
    samples_per_task: int,
) -> list[dict[str, np.ndarray]]:
    """Run the FP32 model on LIBERO gym tasks and collect observations.

    Runs 1 episode per task in closed-loop, then randomly samples
    observations from each task's rollout.

    Returns a list of preprocessed dicts matching the partitioned
    OV model's input format.
    """
    all_samples: list[dict[str, np.ndarray]] = []

    max_steps = 520  # libero_10 suite max episode length

    for task_id in tqdm(range(num_tasks), desc="Collecting gym observations"):
        gym = LiberoGym(task_suite=task_suite, task_id=task_id)
        obs, _ = gym.reset()
        inf_model.reset()

        task_observations: list[dict[str, np.ndarray]] = []

        for _ in range(max_steps):
            np_obs = obs.to_numpy().to_dict(flatten=False)
            inputs = {k: v for k, v in np_obs.items() if v is not None}
            task_observations.append(inputs)

            # Use a copy for inference since preprocessors mutate the dict
            action = inf_model.select_action(inputs.copy())
            obs, _, terminated, truncated, _ = gym.step(action.squeeze(0))
            if terminated or truncated:
                break

        gym.close()
        del gym
        gc.collect()
        print(f"Task {task_id}: collected {len(task_observations)} observations")

        # Uniformly sample: first, last, and evenly spaced in between
        n = len(task_observations)
        k = min(samples_per_task, n)
        indices = np.linspace(0, n - 1, k, dtype=int)
        task_samples = [task_observations[i] for i in indices]
        del task_observations

        # Run through preprocessors
        for raw_obs in task_samples:
            processed = raw_obs
            for preprocessor in inf_model.preprocessors:
                processed = preprocessor(processed)
            all_samples.append(processed)
        del task_samples

    #random.seed(42)
    #random.shuffle(all_samples)

    print(
        f"Collected {len(all_samples)} preprocessed samples "
        f"from {num_tasks} tasks ({samples_per_task} per task)"
    )
    return all_samples


def _embed_all_samples(
    adapter: PartitionedOpenVINOAdapter,
    preprocessed_samples: list[dict[str, np.ndarray]],
) -> list[tuple[dict[str, np.ndarray], np.ndarray, int]]:
    """Run image_embedder + language embedding on all samples once.

    Returns a list of (prefix_lm_input_dict, prefix_pad_masks, batch_size)
    tuples, reusing PartitionedOpenVINOAdapter._embed_inputs.
    """
    embedded = []
    for sample in preprocessed_samples:
        lm_inputs, pad_masks, batch_size = adapter._embed_inputs(sample)
        embedded.append((lm_inputs, pad_masks, batch_size))
    print(f"Embedded {len(embedded)} samples via adapter")
    return embedded


def _build_expert_decoder_calibration(
    adapter: PartitionedOpenVINOAdapter,
    embedded_samples: list[tuple[dict[str, np.ndarray], np.ndarray, int]],
    num_inference_steps: int,
    compressed_prefix_lm_path: str | Path,
    device: str = "CPU",
) -> nncf.Dataset:
    """Build calibration dataset for gemma_expert_decoder.

    Runs prefix_lm on pre-computed embeddings, then collects
    expert_decoder inputs at every denoising step.

    Args:
        adapter: Loaded partitioned adapter (used for expert advancement).
        embedded_samples: Pre-computed (plm_input, pad_masks, batch_size) tuples.
        num_inference_steps: Number of denoising steps.
        compressed_prefix_lm_path: Path to the compressed prefix_lm IR.
            This ensures calibration data reflects the already-compressed prefix_lm.
        device: OpenVINO device for running the compressed prefix_lm.
    """
    core = openvino.Core()
    prefix_lm_compiled = core.compile_model(
        core.read_model(str(compressed_prefix_lm_path)), device_name=device,
    )

    dt = -1.0 / num_inference_steps
    expert_inputs: list[dict[str, np.ndarray]] = []

    for plm_input, prefix_pad_masks, batch_size in embedded_samples:
        # Run prefix_lm -> KV cache
        prefix_result = prefix_lm_compiled(plm_input)
        cache_keys = prefix_result[0]
        cache_values = prefix_result[1]

        # Denoising loop — collect expert inputs at a random subset of steps
        x_t = adapter._sample_noise(batch_size)
        collect_steps = set(random.sample(range(num_inference_steps - 1), k=min(2, num_inference_steps)))
        collect_steps.add(0)
        collect_steps.add(num_inference_steps -1)

        for step in range(num_inference_steps):
            t = 1.0 + step * dt
            time_tensor = np.full((batch_size,), t, dtype=np.float32)

            expert_input = {
                "x_t": x_t.copy(),
                "timestep": time_tensor,
                "cache_keys": np.array(cache_keys),
                "cache_values": np.array(cache_values),
                "prefix_pad_masks": prefix_pad_masks.astype(bool),
            }

            if step in collect_steps:
                expert_inputs.append(expert_input)

            # Advance denoising so next step has realistic x_t
            x_t = adapter._apply_expert(expert_input, dt)

    print(f"Built {len(expert_inputs)} calibration samples for gemma_expert_decoder")
    return nncf.Dataset(expert_inputs)


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

    # Load InferenceModel (adapter + preprocessors) from FP32 export
    inf_model = InferenceModel(source_dir, device=device)
    adapter = inf_model.adapter

    # Collect calibration data from gym rollouts and embed once
    calibration_samples = _collect_gym_observations(
        inf_model,
        task_suite=_CALIBRATION_TASK_SUITE,
        num_tasks=_CALIBRATION_NUM_TASKS,
        samples_per_task=_CALIBRATION_SAMPLES_PER_TASK,
    )
    embedded_samples = _embed_all_samples(adapter, calibration_samples)
    num_inference_steps = adapter._num_inference_steps

    # --- image_embedder: copy unchanged (small model) ---
    print("Skipping compression for image_embedder")
    ov_model = core.read_model(str(source_dir / "image_embedder.xml"))
    openvino.save_model(ov_model, str(output_dir / "image_embedder.xml"))

    # --- prefix_lm: INT4 with AWQ + scale estimation ---
    print("Compressing prefix_lm with INT4_SYM (data-aware)...")
    ov_model = core.read_model(str(source_dir / "prefix_lm.xml"))
    prefix_lm_inputs = [plm_input for plm_input, _, _ in embedded_samples]
    print(f"Built {len(prefix_lm_inputs)} calibration samples for prefix_lm")
    compressed = nncf.compress_weights(
        ov_model,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        group_size=128,
        dataset=nncf.Dataset(prefix_lm_inputs),
        awq=True,
        scale_estimation=True,
        subset_size=_CALIBRATION_SAMPLES,
        advanced_parameters=AdvancedCompressionParameters(
            calibration_device=device,
            group_size_fallback_mode=GroupSizeFallbackMode.ADJUST,
        ),
    )
    openvino.save_model(compressed, str(output_dir / "prefix_lm.xml"))

    # --- gemma_expert_decoder: INT4 with AWQ + scale estimation ---
    print("Compressing gemma_expert_decoder with INT4_SYM (data-aware)...")
    ov_model = core.read_model(str(source_dir / "gemma_expert_decoder.xml"))
    expert_dataset = _build_expert_decoder_calibration(
        adapter, embedded_samples,
        num_inference_steps=num_inference_steps,
        compressed_prefix_lm_path=output_dir / "prefix_lm.xml",
        device=device,
    )
    compressed = nncf.compress_weights(
        ov_model,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        group_size=128,
        dataset=expert_dataset,
        awq=True,
        scale_estimation=True,
        subset_size=_CALIBRATION_SAMPLES * 4,
        advanced_parameters=AdvancedCompressionParameters(
            calibration_device=device,
            group_size_fallback_mode=GroupSizeFallbackMode.ADJUST,
        ),
    )
    openvino.save_model(compressed, str(output_dir / "gemma_expert_decoder.xml"))

    # --- action_output_head: copy unchanged (small model) ---
    print("Skipping compression for action_output_head")
    ov_model = core.read_model(str(source_dir / "action_output_head.xml"))
    openvino.save_model(ov_model, str(output_dir / "action_output_head.xml"))

    # Copy non-model artifacts
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


if __name__ == "__main__":
    FP32_EXPORT_DIR = "pi05_libero_finetuned_hf_ov"
    INT4_EXPORT_DIR = "pi05_libero_finetuned_hf_ov_int4"

    # Step 1: Export partitioned FP32 model (if needed)
    if not Path(FP32_EXPORT_DIR).exists():
        from physicalai.policies.pi05 import Pi05

        policy = Pi05(
            pretrained_name_or_path="lerobot/pi05_libero_finetuned",
            n_action_steps=10,
            compile_model=False,
            dtype="bfloat16",
        )
        policy.to_openvino_partitioned(FP32_EXPORT_DIR)
        del policy

    DEVICE = "CPU"

    # Step 2: Compress with INT4 data-aware algorithms
    #if not Path(INT4_EXPORT_DIR).exists():
    compress_partitioned_model(
        source_dir=FP32_EXPORT_DIR,
        output_dir=INT4_EXPORT_DIR,
        device=DEVICE,
    )

    # Step 3: Evaluate compressed model on LIBERO

    for suite in ["libero_spatial", "libero_object" , "libero_goal" , "libero_10" , "libero_90"]:
        benchmark = LiberoBenchmark(
            task_suite=suite,
            #task_ids=[6],
            num_episodes=20,
            #max_steps=10,
        )

        ov_model = InferenceModel(INT4_EXPORT_DIR, device=DEVICE, runner=ActionChunking(SinglePass()))
        ov_policy = InferenceModelPolicyWrapper(ov_model)
        ov_results = benchmark.evaluate(ov_policy)
        print(ov_results.summary())

        # Close all gym environments to release MuJoCo/rendering resources
        for gym in benchmark.gyms:
            gym.close()
        del ov_results, ov_policy, ov_model, benchmark
        gc.collect()
