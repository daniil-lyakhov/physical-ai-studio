"""Collect calibration datasets for prefix_lm and gemma_expert_decoder.

Loads the FP32 partitioned model, runs it on LIBERO gym tasks to collect
observations, then builds and saves calibration datasets for both LLMs.
"""

import argparse
import gc
import random
from pathlib import Path

import numpy as np
import openvino
from safetensors.numpy import save_file
from tqdm import tqdm

from physicalai.gyms import LiberoGym
from physicalai.inference import InferenceModel
from physicalai.inference.adapters.openvino_partitioned import PartitionedOpenVINOAdapter
from physicalai.inference.runners import ActionChunking, SinglePass

_CALIBRATION_TASK_SUITE = "libero_10"
_CALIBRATION_SAMPLES_PER_TASK = 32
_CALIBRATION_NUM_TASKS = 10


def _collect_gym_observations(
    inf_model: InferenceModel,
    task_suite: str,
    num_tasks: int,
    samples_per_task: int,
) -> list[dict[str, np.ndarray]]:
    """Run the FP32 model on LIBERO gym tasks and collect observations.

    Runs 1 episode per task in closed-loop, then uniformly samples
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
    tuples.
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
    prefix_lm_path: str | Path,
    device: str = "CPU",
) -> list[dict[str, np.ndarray]]:
    """Build calibration dataset for gemma_expert_decoder.

    Runs prefix_lm on pre-computed embeddings, then collects
    expert_decoder inputs at a subset of denoising steps.
    """
    core = openvino.Core()
    prefix_lm_compiled = core.compile_model(
        core.read_model(str(prefix_lm_path)), device_name=device,
    )

    dt = -1.0 / num_inference_steps
    expert_inputs: list[dict[str, np.ndarray]] = []

    for plm_input, prefix_pad_masks, batch_size in tqdm(embedded_samples, desc="Building expert calibration"):
        # Run prefix_lm -> KV cache
        prefix_result = prefix_lm_compiled(plm_input)
        cache_keys = prefix_result[0]
        cache_values = prefix_result[1]

        # Denoising loop — collect expert inputs at a random subset of steps
        x_t = adapter._sample_noise(batch_size)
        random.seed(42)
        collect_steps = set(random.sample(range(num_inference_steps - 1), k=min(2, num_inference_steps)))
        collect_steps.add(0)
        collect_steps.add(num_inference_steps - 1)

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
    return expert_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect calibration datasets for INT4 compression.")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to FP32 partitioned model directory")
    parser.add_argument("--output-dir", type=str, default="./calibration_data", help="Output directory for datasets")
    parser.add_argument("--task-suite", type=str, default=_CALIBRATION_TASK_SUITE, help="LIBERO task suite")
    parser.add_argument("--num-tasks", type=int, default=_CALIBRATION_NUM_TASKS, help="Number of tasks to collect from")
    parser.add_argument("--samples-per-task", type=int, default=_CALIBRATION_SAMPLES_PER_TASK, help="Samples per task")
    parser.add_argument("--device", type=str, default="CPU", help="OpenVINO device")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load FP32 partitioned model
    print(f"Loading model from {args.model_dir} on {args.device}...")
    inf_model = InferenceModel(args.model_dir, device=args.device, runner=ActionChunking(SinglePass()))
    adapter = inf_model.adapter

    # Step 1: Collect gym observations and preprocess
    calibration_samples = _collect_gym_observations(
        inf_model,
        task_suite=args.task_suite,
        num_tasks=args.num_tasks,
        samples_per_task=args.samples_per_task,
    )

    # Step 2: Embed samples (prefix_lm calibration data)
    embedded_samples = _embed_all_samples(adapter, calibration_samples)
    num_inference_steps = adapter._num_inference_steps

    # Save prefix_lm calibration data
    prefix_lm_inputs = [plm_input for plm_input, _, _ in embedded_samples]
    prefix_lm_path = output_dir / "prefix_lm_calibration.safetensors"
    flat = {f"{i:04d}_{k}": v for i, s in enumerate(prefix_lm_inputs) for k, v in s.items()}
    save_file(flat, prefix_lm_path)
    print(f"Saved prefix_lm calibration ({len(prefix_lm_inputs)} samples) to {prefix_lm_path}")

    # Step 3: Build expert decoder calibration data (using FP32 prefix_lm)
    source_prefix_lm = Path(args.model_dir) / "prefix_lm.xml"
    expert_inputs = _build_expert_decoder_calibration(
        adapter,
        embedded_samples,
        num_inference_steps=num_inference_steps,
        prefix_lm_path=source_prefix_lm,
        device=args.device,
    )

    # Save expert decoder calibration data
    expert_path = output_dir / "expert_decoder_calibration.safetensors"
    flat = {f"{i:04d}_{k}": v for i, s in enumerate(expert_inputs) for k, v in s.items()}
    save_file(flat, expert_path)
    print(f"Saved expert_decoder calibration ({len(expert_inputs)} samples) to {expert_path}")

    # Summary
    print(f"\nCalibration data saved to {output_dir}/")
    print(f"  prefix_lm_calibration.safetensors: {len(prefix_lm_inputs)} samples")
    print(f"  expert_decoder_calibration.safetensors: {len(expert_inputs)} samples")


if __name__ == "__main__":
    main()
