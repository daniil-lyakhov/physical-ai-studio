"""Profile prefix_lm activations using NNCF activation profiler.

Loads calibration data from safetensors (output of collect_calib_datasets.py),
profiles the FP32 prefix_lm, optionally compares with INT4, and outputs
statistics tables (CSV) and distribution plots (PNG).
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import openvino
from safetensors.numpy import load_file

import nncf
from nncf.openvino.engine import calibration_device_context

# Add NNCF activation profiler to path
sys.path.insert(0, "/home/dlyakhov/Projects/nncf/tools/activation_profiler")
from profiler import NNCFProfiler  # noqa: E402


def _load_calibration_data(safetensors_path: str | Path) -> list[dict[str, np.ndarray]]:
    """Load calibration data from safetensors and reconstruct list of dicts."""
    flat = load_file(str(safetensors_path))

    # Reconstruct list of dicts from {idx:04d}_{key} format
    samples: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    for key, arr in flat.items():
        idx_str, tensor_name = key.split("_", 1)
        samples[int(idx_str)][tensor_name] = arr

    n_samples = max(samples.keys()) + 1
    result = [samples[i] for i in range(n_samples)]
    print(f"Loaded {len(result)} calibration samples from {safetensors_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile prefix_lm activations.")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to FP32 partitioned model directory")
    parser.add_argument("--compressed-model-dir", type=str, default=None, help="Path to INT4 model directory for comparison")
    parser.add_argument("--calibration-file", type=str, required=True, help="Path to prefix_lm_calibration.safetensors")
    parser.add_argument("--output-dir", type=str, default="./profiling_results", help="Output directory")
    parser.add_argument("--pattern", type=str, default=r".*MatMul", help="Regex pattern for layer selection")
    parser.add_argument("--num-samples", type=int, default=32, help="Number of samples for profiling")
    parser.add_argument("--device", type=str, default="GPU", help="OpenVINO device for statistics collection")
    parser.add_argument("--no-histograms", action="store_true", help="Skip per-layer histogram plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Load calibration data
    calibration_samples = _load_calibration_data(args.calibration_file)

    # Wrap as nncf.Dataset (identity transform — data already matches model inputs)
    num_samples = min(args.num_samples, len(calibration_samples))
    dataset = nncf.Dataset(calibration_samples[:num_samples])

    # Load FP32 prefix_lm
    core = openvino.Core()
    fp32_path = Path(args.model_dir) / "prefix_lm.xml"
    print(f"Loading FP32 model from {fp32_path}...")
    ov_model_fp = core.read_model(str(fp32_path))

    # Create profiler
    profiler = NNCFProfiler(pattern=args.pattern, dataset=dataset, num_samples=num_samples)

    # Collect FP32 activations
    print(f"Collecting FP32 activations (pattern: {args.pattern}, samples: {num_samples}, device: {args.device})...")
    with calibration_device_context(args.device):
        data_fp = profiler.collect_activations(ov_model_fp)
    print(f"  Found {len(data_fp)} matching layers")

    # Calculate FP32 statistics
    stats_fp = profiler.calculate_stats(
        data_fp, statistics=["min", "max", "mean", "std", "median", "percentile_95", "abs_mean"]
    )
    stats_path = output_dir / "fp32_stats.csv"
    stats_fp.to_csv(stats_path, index=False)
    print(f"Saved FP32 statistics to {stats_path}")
    print(stats_fp.to_string())

    # Compare with INT4 if provided
    if args.compressed_model_dir:
        int4_path = Path(args.compressed_model_dir) / "prefix_lm.xml"
        print(f"\nLoading INT4 model from {int4_path}...")
        ov_model_int4 = core.read_model(str(int4_path))

        print("Collecting INT4 activations...")
        with calibration_device_context(args.device):
            data_int4 = profiler.collect_activations(ov_model_int4)

        # Compare
        print("Comparing FP32 vs INT4...")
        comparison = profiler.compare_activations(
            data_fp, data_int4,
            metrics=["mean_diff", "std_diff", "relative_diff"],
            statistics=["min", "max", "mean", "std"],
        )
        comp_path = output_dir / "comparison_fp32_vs_int4.csv"
        comparison.to_csv(comp_path, index=False)
        print(f"Saved comparison to {comp_path}")
        print(comparison.to_string())

        # Generate plots
        print("Generating comparison plots...")
        all_figs, summary_figs = profiler.plot(
            "compare_detailed",
            data_fp, data_int4,
            data1_label="FP32",
            data2_label="INT4",
            show_histograms=not args.no_histograms,
            show_summary=True,
            display_figures=False,
        )

        # Save summary plots
        for act_type, fig in summary_figs.items():
            fig_path = figures_dir / f"summary_{act_type}.png"
            fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
            print(f"  Saved {fig_path}")

        # Save per-layer histograms
        if not args.no_histograms:
            for i, (layer_name, type_figs) in enumerate(all_figs.items()):
                for act_type, fig in type_figs.items():
                    fig_path = figures_dir / f"hist_{i:03d}_{act_type}.png"
                    fig.savefig(str(fig_path), dpi=100, bbox_inches="tight")
            print(f"  Saved {len(all_figs)} histogram figures to {figures_dir}/")

    print(f"\nDone. Results in {output_dir}/")


if __name__ == "__main__":
    main()
