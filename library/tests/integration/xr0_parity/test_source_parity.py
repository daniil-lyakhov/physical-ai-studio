# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end parity test: source XR0 vs. framework XR0 on the LIBERO checkpoint.

The two implementations pin incompatible ``transformers`` versions, so each runs
in its own repository ``env`` as a subprocess (see :mod:`runner`). Both consume
an identical batch of synthetic samples (:mod:`build_inputs`, default 8) and the
same LIBERO checkpoint, inject the same pinned per-sample rectified-flow noise,
and emit the raw predicted action chunks *plus* the VLM key/value cache the DiT
cross-attends to.

Everything runs in **float32**. At this precision the two Qwen3-VL ports
(transformers 4.57.1 vs 5.3.0) are numerically identical, so the comparison
splits cleanly:

* the **VLM KV cache** diff (float32) is the VLM-port residual -- essentially
  zero (rel mean < 0.1 %), confirming the vendored VLM shim matches the source;
* the **whole-model action** diff (float32) is therefore the DiT / flow-head
  port residual, since the VLM feeds both DiTs numerically-identical inputs.

This test compares the action chunks per-sample (cosine, percentile abs diff)
and as output distributions (per-dimension mean / std), and separately diffs the
VLM KV cache.

Opt-in and heavy: it loads Qwen3-VL-4B (~8 GB) twice on CPU. Enable with
``XR0_PARITY=1`` and run with ``-m slow`` (or ``-p no:cacheprovider`` as needed)::

    XR0_PARITY=1 env/bin/python -m pytest \\
        library/tests/integration/xr0_parity/test_source_parity.py -m slow -s
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from .build_inputs import build_inputs  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

CHECKPOINT_REPO = "XiaomiRobotics/Xiaomi-Robotics-0-LIBERO"

# Number of synthetic samples to run through both implementations. Overridable
# for a quicker smoke run (e.g. XR0_PARITY_SAMPLES=4).
NUM_SAMPLES = int(os.environ.get("XR0_PARITY_SAMPLES", "8"))

# physical-ai-studio repo root: .../library/tests/integration/xr0_parity/<this>.
_FRAMEWORK_REPO = Path(__file__).resolve().parents[4]
_PROJECTS_ROOT = _FRAMEWORK_REPO.parent

_FRAMEWORK_PYTHON = Path(
    os.environ.get("XR0_FRAMEWORK_PYTHON", _FRAMEWORK_REPO / "env" / "bin" / "python")
)
_SOURCE_PYTHON = Path(
    os.environ.get(
        "XR0_SOURCE_PYTHON",
        _PROJECTS_ROOT / "Xiaomi-Robotics-0" / "xr0" / "env" / "bin" / "python",
    )
)

_RUNNER = Path(__file__).resolve().parent / "runner.py"
_SUBPROCESS_TIMEOUT_S = 3600

# Parity thresholds (float32 end-to-end). In float32 the two Qwen3-VL ports are
# numerically identical, so the action residual is just the DiT / flow-head port
# plus float32 rounding through the rectified-flow ODE. Observed on the LIBERO
# checkpoint (8 samples): min per-sample cosine ~1.000000, mean_abs ~2.1e-5,
# p99 abs diff ~2.4e-4, max ~8.1e-3. The full abs-diff distribution
# (p50/p90/p99/p99.9 + max + mean) is reported; the max is asserted only as a
# gross-regression ceiling, while p99 / mean / cosine / distribution carry the
# tight parity bounds.
_MIN_COSINE = 0.9999
_MAX_MEAN_ABS_DIFF = 0.002
_MAX_P99_ABS_DIFF = 0.005
# Sanity ceiling on the single worst element (catches gross regressions only).
_MAX_ABS_DIFF = 0.1
# Per-dimension distribution statistics (mean / std over the sample batch).
_MAX_STAT_DIFF = 0.02
# VLM KV-cache parity (float32): the DiT's input. In float32 the two Qwen3-VL
# ports agree to float32 rounding -- observed rel mean ~0.01-0.02 %, max abs
# ~0.018 keys / ~0.028 values over the last 16 layers. (This requires restoring
# the source's rotary ``inv_freq`` to full float32 in the runner: mibot's
# ``XR0._build_model`` casts the whole model to bf16, quantizing ``inv_freq``,
# and a later ``model.float()`` cannot recover it -- leaving the RoPE
# frequencies ~9e-4 off and inflating the fp32 key cache diff to ~0.2.) The
# relative-mean bound is the parity gate; the max is a gross-regression ceiling
# (a real port bug would blow the max up to the value scale, |x| up to ~28, or
# NaN).
_MAX_VLM_REL_MEAN_DIFF = 0.001
_MAX_VLM_ABS_DIFF = 0.1


def _checkpoint_cached() -> bool:
    """Whether the LIBERO checkpoint snapshot is present in the local HF cache."""
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    folder = cache / f"models--{CHECKPOINT_REPO.replace('/', '--')}"
    return folder.exists() and any(folder.glob("snapshots/*/model.safetensors.index.json"))


_SKIP_REASON = (
    "XR0 parity test is opt-in and requires both repo envs + cached checkpoint. "
    "Set XR0_PARITY=1 and ensure env interpreters and the LIBERO checkpoint exist."
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("XR0_PARITY") != "1"
        or not _FRAMEWORK_PYTHON.exists()
        or not _SOURCE_PYTHON.exists()
        or not _checkpoint_cached(),
        reason=_SKIP_REASON,
    ),
]


def _resolve_checkpoint_dir() -> str:
    """Return the local checkpoint snapshot directory (offline, from cache)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from physicalai.policies.xr0.pretrained_utils import resolve_pretrained_path

    return str(resolve_pretrained_path(CHECKPOINT_REPO))


def _run_impl(python: Path, impl: str, inputs_path: Path, checkpoint: str, output: Path) -> None:
    """Invoke ``runner.py`` for one implementation in its own environment."""
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        str(python),
        str(_RUNNER),
        "--impl",
        impl,
        "--inputs",
        str(inputs_path),
        "--checkpoint",
        checkpoint,
        "--output",
        str(output),
    ]
    print(f"\n[{impl}] $ {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(  # noqa: S603
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_S,
        check=False,
    )
    print(f"[{impl}] stdout:\n{result.stdout}", file=sys.stderr)
    print(f"[{impl}] stderr:\n{result.stderr}", file=sys.stderr)
    if result.returncode != 0:
        pytest.fail(f"{impl} runner failed (exit {result.returncode}). See captured output above.")


@pytest.fixture(scope="module")
def predictions(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, torch.Tensor]]:
    """Run both implementations once and return their action chunks + VLM KV cache."""
    workdir = tmp_path_factory.mktemp("xr0_parity")
    inputs_path = workdir / "inputs.pt"
    source_out = workdir / "source_pred.pt"
    framework_out = workdir / "framework_pred.pt"

    torch.save(build_inputs(NUM_SAMPLES), inputs_path)
    checkpoint_dir = _resolve_checkpoint_dir()

    _run_impl(_FRAMEWORK_PYTHON, "framework", inputs_path, checkpoint_dir, framework_out)
    _run_impl(_SOURCE_PYTHON, "source", inputs_path, checkpoint_dir, source_out)

    return {
        "framework": torch.load(framework_out, map_location="cpu"),
        "source": torch.load(source_out, map_location="cpu"),
    }


def _abs_diff_report(label: str, source: torch.Tensor, framework: torch.Tensor) -> dict[str, float]:
    """Compute and print the fp32 abs-diff percentile distribution for two tensors.

    ``max`` / ``mean`` are exact over the full tensor; the percentiles are
    estimated on a deterministic random subsample so ``torch.quantile`` stays
    within its element-count limit for the large VLM KV tensors.
    """
    diff = (source.to(torch.float32) - framework.to(torch.float32)).abs()
    flat = diff.flatten()

    sample = flat
    max_quantile_elements = 1_000_000
    if flat.numel() > max_quantile_elements:
        generator = torch.Generator().manual_seed(0)
        index = torch.randperm(flat.numel(), generator=generator)[:max_quantile_elements]
        sample = flat[index]

    levels = [0.5, 0.9, 0.99, 0.999]
    percentiles = {
        level: value.item()
        for level, value in zip(levels, torch.quantile(sample, torch.tensor(levels)))
    }
    stats = {
        "max": diff.max().item(),
        "mean": diff.mean().item(),
        "p99": percentiles[0.99],
    }
    percentile_report = " ".join(f"p{level * 100:g}={percentiles[level]:.4e}" for level in levels)
    print(
        f"\n[{label}] elements={flat.numel()}"
        f"\n[{label}]   abs_diff: {percentile_report} "
        f"max={stats['max']:.4e} mean={stats['mean']:.4e}",
        file=sys.stderr,
    )
    return stats


def test_shapes_match(predictions: dict[str, dict[str, torch.Tensor]]) -> None:
    """Both implementations produce the same action-chunk batch shape."""
    source = predictions["source"]["action"]
    framework = predictions["framework"]["action"]
    assert source.shape == framework.shape
    assert tuple(framework.shape) == (NUM_SAMPLES, 30, 32)


def test_action_chunk_parity(predictions: dict[str, dict[str, torch.Tensor]]) -> None:
    """The predicted action chunks match within the float32 parity tolerance."""
    source = predictions["source"]["action"].to(torch.float32)
    framework = predictions["framework"]["action"].to(torch.float32)

    stats = _abs_diff_report("parity", source, framework)
    per_sample_cosine = torch.nn.functional.cosine_similarity(
        source.flatten(1), framework.flatten(1), dim=1
    )
    min_cosine = per_sample_cosine.min().item()
    mean_cosine = per_sample_cosine.mean().item()
    print(
        f"[parity]   cosine:   min={min_cosine:.6f} mean={mean_cosine:.6f}",
        file=sys.stderr,
    )

    assert min_cosine >= _MIN_COSINE, f"min per-sample cosine {min_cosine:.6f} < {_MIN_COSINE}"
    assert stats["mean"] <= _MAX_MEAN_ABS_DIFF, f"mean abs diff {stats['mean']:.4e} > {_MAX_MEAN_ABS_DIFF}"
    assert stats["p99"] <= _MAX_P99_ABS_DIFF, f"p99 abs diff {stats['p99']:.4e} > {_MAX_P99_ABS_DIFF}"
    assert stats["max"] <= _MAX_ABS_DIFF, f"max abs diff {stats['max']:.4e} > {_MAX_ABS_DIFF}"


def test_vlm_kv_cache_parity(predictions: dict[str, dict[str, torch.Tensor]]) -> None:
    """The VLM KV cache (the DiT's input) agrees upstream of the flow integration.

    In float32 the two Qwen3-VL ports (transformers 4.57.1 vs 5.3.0) are
    numerically identical (rel mean < 0.1 %), so this diff is the VLM-port
    residual -- and its near-zero value is what lets the whole-model action diff
    be attributed to the DiT / flow head. The relative-mean is the parity gate;
    the raw max (a cancellation-sensitive element where the cross-version
    reduction-order difference is relatively amplified) is asserted only as a
    max-abs gross-regression ceiling.
    """
    for name in ("vlm_keys", "vlm_values"):
        source = predictions["source"][name].to(torch.float32)
        framework = predictions["framework"][name].to(torch.float32)
        assert source.shape == framework.shape, f"{name} shape mismatch"
        stats = _abs_diff_report(f"parity-vlm:{name}", source, framework)
        magnitude_mean = source.abs().mean().item()
        rel_mean = stats["mean"] / magnitude_mean
        print(
            f"[parity-vlm:{name}]   magnitude_mean={magnitude_mean:.4f} "
            f"rel_mean={rel_mean:.4%}",
            file=sys.stderr,
        )
        # The absolute diff is dominated by massive-activation channels; the
        # relative mean is the honest port-parity metric.
        assert rel_mean <= _MAX_VLM_REL_MEAN_DIFF, (
            f"{name} relative mean diff {rel_mean:.4%} > {_MAX_VLM_REL_MEAN_DIFF:.4%}"
        )
        # Keep the raw max abs diff as a gross-regression ceiling.
        assert stats["max"] <= _MAX_VLM_ABS_DIFF, (
            f"{name} max abs diff {stats['max']:.4e} > {_MAX_VLM_ABS_DIFF}"
        )


def test_distribution_stats_match(predictions: dict[str, dict[str, torch.Tensor]]) -> None:
    """Per-dimension mean/std of the action distribution agree across implementations."""
    source = predictions["source"]["action"].to(torch.float32)
    framework = predictions["framework"]["action"].to(torch.float32)

    mean_diff = (source.mean(dim=0) - framework.mean(dim=0)).abs().max().item()
    std_diff = (source.std(dim=0) - framework.std(dim=0)).abs().max().item()

    print(
        f"\n[parity-stats] max_mean_diff={mean_diff:.4e} max_std_diff={std_diff:.4e}",
        file=sys.stderr,
    )

    assert mean_diff <= _MAX_STAT_DIFF, f"max per-dim mean diff {mean_diff:.4e} > {_MAX_STAT_DIFF}"
    assert std_diff <= _MAX_STAT_DIFF, f"max per-dim std diff {std_diff:.4e} > {_MAX_STAT_DIFF}"
