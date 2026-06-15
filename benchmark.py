import gc
import time
from pathlib import Path
import sys

import numpy as np
import torch

from physicalai.benchmark.gyms import LiberoBenchmark
from physicalai.data import Observation
from physicalai.inference import InferenceModel


class TimedPolicyWrapper:
    """Wraps InferenceModel and records per-call inference latency."""

    def __init__(self, inf_model: InferenceModel) -> None:
        self._inf_model = inf_model
        self._latencies: list[float] = []

    def select_action(self, observation: Observation) -> torch.Tensor:
        np_obs = observation.to_numpy().to_dict(flatten=False)
        np_inputs = {k: v for k, v in np_obs.items() if v is not None}

        t0 = time.perf_counter()
        action = self._inf_model.select_action(np_inputs)
        elapsed = time.perf_counter() - t0

        self._latencies.append(elapsed)
        return torch.from_numpy(action)

    def reset(self) -> None:
        self._inf_model.reset()

    def print_stats(self, suite_name: str = "") -> None:
        if not self._latencies:
            print("No latency data collected.")
            return

        arr = np.array(self._latencies) * 1000  # ms
        header = f"Inference Timing — {suite_name}" if suite_name else "Inference Timing"
        print(f"\n{'=' * 60}")
        print(f"  {header}")
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
    current_export_dir = sys.argv[1]


    DEVICE = "GPU.0"


    # Step 3: Evaluate compressed model on LIBERO

    for suite in ["libero_spatial",]:
        benchmark = LiberoBenchmark(
            task_suite=suite,
            #task_ids=[6],
            num_episodes=4,
            max_steps=80,
        )

        ov_model = InferenceModel(current_export_dir, device=DEVICE)
        ov_policy = TimedPolicyWrapper(ov_model)
        ov_results = benchmark.evaluate(ov_policy)
        print(ov_results.summary())
        ov_policy.print_stats(suite_name=suite)

        # Close all gym environments to release MuJoCo/rendering resources
        for gym in benchmark.gyms:
            gym.close()
        del ov_results, ov_policy, ov_model, benchmark
        gc.collect()
