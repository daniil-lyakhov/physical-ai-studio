# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Training-iteration parity test: source XR0 vs. framework XR0 on the LIBERO ckpt.

Companion to :mod:`test_source_parity` (which checks fp32 *inference*). Where
that test diffs the generated action chunk and VLM KV cache, this one diffs a
full **training step**: one forward flow-matching loss followed by
``loss.backward()``. Each implementation runs in its own repository ``env`` as a
subprocess (the two pin incompatible ``transformers`` versions -- see
:mod:`training_runner`), consuming an identical batch of synthetic samples
(:mod:`build_inputs`) and the same LIBERO checkpoint.

Every training-time random draw is pinned so the step is bit-for-bit
reproducible across the two environments: the rectified-flow ``noise`` and
``timestep`` are baked into ``inputs.pt``, and both models are built with
``async_train=False`` / ``training_repeat=1`` (prefix-free, single-copy) with the
DiT's default zero dropout. Everything runs in float32 / eager attention, so any
residual isolates the DiT / flow-head port (the two Qwen3-VL backbones are
numerically identical at float32).

The test asserts parity of (a) the scalar loss components, (b) the predicted
velocity / target, and (c) the resulting gradients (per-parameter grad-norm
distribution + element-wise grads of the small flow-head params).

Opt-in and heavy: it loads Qwen3-VL-4B (~8 GB) twice on CPU. Enable with
``XR0_PARITY=1`` and run with ``-m slow``::

    XR0_PARITY=1 env/bin/python -m pytest \\
        library/tests/integration/xr0_parity/test_training_parity.py -m slow -s
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

_RUNNER = Path(__file__).resolve().parent / "training_runner.py"
_SUBPROCESS_TIMEOUT_S = 3600

# Parity thresholds (float32 end-to-end, one prefix-free training step). In
# float32 the two Qwen3-VL ports are numerically identical, so the residual is
# the DiT / flow-head port plus float32 rounding. Loss and predicted velocity
# carry the tight bounds; the gradient bounds are looser because a gradient
# accumulates the forward residual through the whole backward graph and the two
# transformers versions use different (numerically non-identical) attention /
# vision kernels. The single-worst-element checks are gross-regression ceilings.
_MAX_LOSS_REL_DIFF = 0.01
# Predicted velocity (pre-reduction DiT output).
_MAX_PRED_REL_MEAN_DIFF = 0.001
_MAX_PRED_ABS_DIFF = 0.1
# Per-parameter gradient-norm relative diff (median / p99 over all grad params).
_MAX_GRAD_NORM_REL_MEDIAN = 0.01
_MAX_GRAD_NORM_REL_P99 = 0.05
# Element-wise relative-mean diff on the small flow-head param gradients.
_MAX_FLOW_GRAD_REL_MEAN = 0.02


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
    """Invoke ``training_runner.py`` for one implementation in its own environment."""
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
def training_step(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, object]]:
    """Run one training step under both implementations and return their artifacts."""
    workdir = tmp_path_factory.mktemp("xr0_training_parity")
    inputs_path = workdir / "inputs.pt"
    source_out = workdir / "source_step.pt"
    framework_out = workdir / "framework_step.pt"

    torch.save(build_inputs(NUM_SAMPLES), inputs_path)
    checkpoint_dir = _resolve_checkpoint_dir()

    _run_impl(_FRAMEWORK_PYTHON, "framework", inputs_path, checkpoint_dir, framework_out)
    _run_impl(_SOURCE_PYTHON, "source", inputs_path, checkpoint_dir, source_out)

    return {
        "framework": torch.load(framework_out, map_location="cpu"),
        "source": torch.load(source_out, map_location="cpu"),
    }


def _rel_diff(source: float, framework: float) -> float:
    """Relative difference of two scalars, guarded against a zero denominator."""
    denom = max(abs(source), abs(framework), 1e-8)
    return abs(source - framework) / denom


def test_loss_matches(training_step: dict[str, dict[str, object]]) -> None:
    """The scalar flow-matching loss components match within tolerance."""
    source = training_step["source"]
    framework = training_step["framework"]

    for key in ("loss", "loss_mse", "loss_freq"):
        src_val = float(source[key])  # type: ignore[arg-type]
        fw_val = float(framework[key])  # type: ignore[arg-type]
        rel = _rel_diff(src_val, fw_val)
        print(
            f"\n[loss] {key}: source={src_val:.6e} framework={fw_val:.6e} rel={rel:.4e}",
            file=sys.stderr,
        )
        if key == "loss_freq" and src_val == 0.0 and fw_val == 0.0:
            continue  # frequency term disabled (enable_freq=False).
        assert rel <= _MAX_LOSS_REL_DIFF, f"{key} rel diff {rel:.4e} > {_MAX_LOSS_REL_DIFF}"


def test_velocity_pred_parity(training_step: dict[str, dict[str, object]]) -> None:
    """The predicted velocity (and target) match within the float32 parity bound."""
    source_pred = training_step["source"]["pred"].to(torch.float32)  # type: ignore[union-attr]
    framework_pred = training_step["framework"]["pred"].to(torch.float32)  # type: ignore[union-attr]
    source_target = training_step["source"]["target"].to(torch.float32)  # type: ignore[union-attr]
    framework_target = training_step["framework"]["target"].to(torch.float32)  # type: ignore[union-attr]

    assert source_pred.shape == framework_pred.shape
    # The velocity target is (action_target - noise), both pinned -> must match
    # to float32 rounding regardless of the model.
    target_max = (source_target - framework_target).abs().max().item()
    assert target_max <= _MAX_PRED_ABS_DIFF, f"velocity target diff {target_max:.4e}"

    diff = (source_pred - framework_pred).abs()
    rel_mean = diff.mean().item() / max(source_pred.abs().mean().item(), 1e-8)
    max_abs = diff.max().item()
    print(
        f"\n[pred] rel_mean={rel_mean:.4e} max_abs={max_abs:.4e} "
        f"target_max={target_max:.4e}",
        file=sys.stderr,
    )
    assert rel_mean <= _MAX_PRED_REL_MEAN_DIFF, f"pred rel mean {rel_mean:.4e}"
    assert max_abs <= _MAX_PRED_ABS_DIFF, f"pred max abs {max_abs:.4e}"


def test_gradient_parity(training_step: dict[str, dict[str, object]]) -> None:
    """Per-parameter gradient norms agree across the two implementations."""
    source_stats: dict[str, list[float]] = training_step["source"]["grad_stats"]  # type: ignore[assignment]
    framework_stats: dict[str, list[float]] = training_step["framework"]["grad_stats"]  # type: ignore[assignment]

    shared = sorted(set(source_stats) & set(framework_stats))
    assert shared, "no gradient parameters shared between the two implementations"

    rel_diffs = []
    worst_name, worst_rel = "", 0.0
    for name in shared:
        src_norm = source_stats[name][0]
        fw_norm = framework_stats[name][0]
        # Both grads must be non-null and non-zero (a real port bug that drops a
        # parameter from the backward graph would zero its grad norm).
        assert src_norm > 0.0, f"source grad norm for {name} is zero"
        assert fw_norm > 0.0, f"framework grad norm for {name} is zero"
        rel = _rel_diff(src_norm, fw_norm)
        rel_diffs.append(rel)
        if rel > worst_rel:
            worst_name, worst_rel = name, rel

    rel_tensor = torch.tensor(rel_diffs)
    median = rel_tensor.median().item()
    p99 = torch.quantile(rel_tensor, 0.99).item()
    print(
        f"\n[grad] params={len(shared)} grad-norm rel diff: "
        f"median={median:.4e} p99={p99:.4e} max={worst_rel:.4e} ({worst_name})",
        file=sys.stderr,
    )
    assert median <= _MAX_GRAD_NORM_REL_MEDIAN, f"grad-norm median rel {median:.4e}"
    assert p99 <= _MAX_GRAD_NORM_REL_P99, f"grad-norm p99 rel {p99:.4e}"


def test_flow_grad_elementwise(training_step: dict[str, dict[str, object]]) -> None:
    """Element-wise gradients of the small flow-head params match closely."""
    source_grads: dict[str, torch.Tensor] = training_step["source"]["flow_grads"]  # type: ignore[assignment]
    framework_grads: dict[str, torch.Tensor] = training_step["framework"]["flow_grads"]  # type: ignore[assignment]

    shared = sorted(set(source_grads) & set(framework_grads))
    assert shared, "no flow-head gradients shared between the two implementations"

    worst_name, worst_rel = "", 0.0
    for name in shared:
        src = source_grads[name].to(torch.float32)
        fw = framework_grads[name].to(torch.float32)
        assert src.shape == fw.shape, f"grad shape mismatch for {name}"
        rel_mean = (src - fw).abs().mean().item() / max(src.abs().mean().item(), 1e-8)
        if rel_mean > worst_rel:
            worst_name, worst_rel = name, rel_mean
        print(f"[flow_grad] {name}: rel_mean={rel_mean:.4e}", file=sys.stderr)
        assert rel_mean <= _MAX_FLOW_GRAD_REL_MEAN, (
            f"{name} grad rel mean {rel_mean:.4e} > {_MAX_FLOW_GRAD_REL_MEAN}"
        )
    print(f"\n[flow_grad] worst rel_mean={worst_rel:.4e} ({worst_name})", file=sys.stderr)
