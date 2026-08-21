# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Replay ONE recorded observation sequence through several models and compare
their closed-loop STABILITY, offline (no robot, no retrain).

Why: on the real robot XR0 shakes while pi05 works on the *identical* setup. The
export is deterministic (std=0), so the shaking is the model's real, repeatable
response to the observations. This script isolates whether that instability is
intrinsic to XR0 by feeding the SAME observations to both models and measuring
how much each model's plan changes between consecutive (1-step-apart) frames.

Metric — self-consistency:
  For consecutive observations o_k, o_{k+1} the model emits chunks C_k, C_{k+1}
  (each ``(chunk, action_dim)``). C_k predicts absolute steps k..k+H-1 and
  C_{k+1} predicts k+1..k+H, so they overlap on H-1 steps. A stable policy
  predicts nearly the SAME actions for those shared future steps:
      disagreement = mean_over_overlap | C_k[1:] - C_{k+1}[:-1] |   (per joint)
  Low  -> plan is stable frame-to-frame  (smooth robot).
  High -> tiny obs change -> large plan change (the shaking).

Interpretation:
  * XR0 disagreement >> pi05 disagreement  -> XR0 is intrinsically hypersensitive;
    representation/robustness is the real axis -> dig there before any retrain.
  * XR0 ~= pi05                              -> inter-chunk swing is normal; the
    shaking is NOT explained by model plan instability -> look elsewhere.

Usage: edit OBS_DIR and MODELS below, then ``python replay_compare.py``.
Record the obs sequence by running ``run_local.py`` (DiagRecorder now saves up to
``max_obs`` consecutive obs into ``diag_<model>/``).
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

from physicalai.inference import InferenceModel
from physicalai.inference.constants import IMAGES, STATE, TASK

# Directory of consecutive obs_*.npz recorded by run_local.py's DiagRecorder.
OBS_DIR = "diag_xr0_v1"
TASK_STR = "Put the green ball to the black box"
ACTION_DIM = 6  # real SO-101 joints to report on

# Each entry is replayed over the SAME obs sequence, one model at a time.
MODELS: list[dict] = [
    {
        "name": "xr0",
        "path": "/home/dupeljan/Projects/home_robot/models/xr0_v1/xr0_put_balls_to_box_irs_normalized_full",
        "device": "GPU.0",
        "precision": "f16",
    },
    {
        "name": "pi05",
        "path": "/home/dupeljan/Projects/home_robot/models/pi05_balls/export_openvino",
        "device": "GPU.0",
        "precision": "f16",
    },
]


def load_obs_sequence(obs_dir: str) -> list[dict]:
    """Load consecutive observations, sorted by step, as runtime-style dicts."""
    files = sorted(Path(obs_dir).glob("obs_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        raise SystemExit(f"no obs_*.npz found in {obs_dir!r} (run run_local.py first)")
    seq = []
    for f in files:
        d = np.load(f)
        images = {k[len("img__") :]: d[k] for k in d if k.startswith("img__")}
        obs: dict = {STATE: np.asarray(d["state"], dtype=np.float32)[np.newaxis]}
        if len(images) > 1:
            for name, arr in images.items():
                obs[f"{IMAGES}.{name}"] = arr[np.newaxis]
        else:
            obs[IMAGES] = next(iter(images.values()))[np.newaxis]
        obs[TASK] = [TASK_STR]
        seq.append(obs)
    print(f"loaded {len(seq)} consecutive observations from {obs_dir}")
    return seq


def run_model(cfg: dict, seq: list[dict]) -> np.ndarray:
    """Return stacked chunks ``(n_obs, chunk, ACTION_DIM)`` for one model."""
    adapter = {"INFERENCE_PRECISION_HINT": cfg["precision"]} if cfg["device"].startswith("GPU") else {}
    model = InferenceModel(cfg["path"], device=cfg["device"], **adapter)
    chunks = [np.asarray(model.predict_action_chunk(o), dtype=np.float32)[..., :ACTION_DIM] for o in seq]
    del model
    gc.collect()
    return np.stack(chunks)  # (n_obs, chunk, ACTION_DIM)


def summarize(name: str, chunks: np.ndarray) -> None:
    """Print per-joint stability metrics for one model."""
    # Within-chunk smoothness: mean |Δ/step| along the horizon.
    within = np.abs(np.diff(chunks, axis=1)).mean(axis=(0, 1))
    # Self-consistency: overlap disagreement between consecutive (1-step-apart) plans.
    a, b = chunks[:-1, 1:, :], chunks[1:, :-1, :]  # shared absolute steps
    overlap = np.abs(a - b).mean(axis=(0, 1))
    print(f"\n=== {name} ===  (n_obs={chunks.shape[0]}, chunk={chunks.shape[1]})")
    print(f"  within-chunk mean |Δ/step| per joint: {np.round(within, 3)}")
    print(f"  SELF-CONSISTENCY mean |disagreement| per joint: {np.round(overlap, 3)}")
    print(f"  SELF-CONSISTENCY overall mean: {overlap.mean():.3f}   (lower == more stable)")


def main() -> None:
    seq = load_obs_sequence(OBS_DIR)
    results = {}
    for cfg in MODELS:
        try:
            results[cfg["name"]] = run_model(cfg, seq)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {cfg['name']}: {exc}")
    for name, chunks in results.items():
        summarize(name, chunks)
    if len(results) > 1:
        print("\n--- verdict ---")
        cons = {n: float(np.abs(c[:-1, 1:, :] - c[1:, :-1, :]).mean()) for n, c in results.items()}
        for n in results:
            print(f"  {n:>6}: self-consistency mean |disagreement| = {cons[n]:.3f}")
        best = min(cons, key=cons.get)
        worst = max(cons, key=cons.get)
        ratio = cons[worst] / max(cons[best], 1e-6)
        print(f"  {worst} is {ratio:.1f}x less self-consistent than {best}.")
        print("  >> if xr0 is the worst by a large margin, its instability is intrinsic.")


if __name__ == "__main__":
    main()
