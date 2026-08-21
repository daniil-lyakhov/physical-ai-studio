# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Measure how *smooth* an eager XR0 checkpoint's predicted action chunks are.

Loads the eager XR0 policy from a Lightning checkpoint, runs it on frames drawn
from a single dataset episode, and reports smoothness metrics on the predicted
30-step action chunks -- compared against the ground-truth action chunks from
the same episode as the reference.

Smoothness is quantified *inside a single predicted chunk* (one forward pass),
so the numbers are independent of closed-loop drift / covariate shift. Metrics
(per joint, in the dataset's physical action units, e.g. degrees):

  * velocity      mean |a[t+1] - a[t]|            (1st difference)
  * acceleration  mean |a[t+1] - 2a[t] + a[t-1]|  (2nd difference)
  * jerk          mean |3rd difference|           (canonical smoothness metric)
  * hf_ratio      fraction of AC spectral power above ``HF_CUTOFF_FRAC`` of
                  Nyquist (0 = smooth, ->1 = high-frequency / jittery)

The ``pred / ground-truth`` ratio at the end is the headline number: 1.0 means
the model is as smooth as the data; >1 means it injects roughness the data does
not contain.

Usage:
    python measure_smoothness.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from physicalai.data import LeRobotDataModule
from physicalai.policies import XR0
from physicalai.train.utils import reformat_dataset_to_match_policy

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CKPT = "experiments/xr0_Put-the-yellow-ball-to-the-black-box/checkpoints/last.ckpt"
DATASET_ROOT = "/home/dlyakhov/datasets/Put-the-yellow-ball-to-the-black-box"
EPISODE = 3
ACTION_DIM = 6
# Number of frames (=chunks) to average over from the episode. None -> all.
MAX_FRAMES: int | None = 50
# Fraction of Nyquist above which spectral power counts as "high frequency".
HF_CUTOFF_FRAC = 0.25
# Reproducible frame sampling + rectified-flow noise.
SEED = 0

JOINT_NAMES = ("shoulder", "upper_arm", "forearm", "wrist_pitch", "wrist_roll", "gripper")


def nth_diff_l1(chunk: np.ndarray, n: int) -> np.ndarray:
    """Per-joint mean absolute n-th finite difference along time.

    Args:
        chunk: ``(T, D)`` action chunk.
        n: Difference order (1=velocity, 2=acceleration, 3=jerk).

    Returns:
        ``(D,)`` per-joint mean absolute n-th difference.
    """
    d = chunk
    for _ in range(n):
        d = np.diff(d, axis=0)
    return np.abs(d).mean(axis=0)


def high_freq_ratio(chunk: np.ndarray, cutoff_frac: float) -> np.ndarray:
    """Per-joint fraction of AC spectral power above ``cutoff_frac`` of Nyquist.

    Args:
        chunk: ``(T, D)`` action chunk.
        cutoff_frac: Cutoff as a fraction of Nyquist (0..0.5 in cycles/frame).

    Returns:
        ``(D,)`` per-joint high-frequency power ratio in ``[0, 1]``.
    """
    length = chunk.shape[0]
    spectrum = np.fft.rfft(chunk, axis=0)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(length)  # 0 .. 0.5 cycles/frame
    # Drop the DC bin so the ratio reflects oscillation, not the mean offset.
    ac_power = power[1:]
    ac_freqs = freqs[1:]
    high = ac_freqs > cutoff_frac
    total = ac_power.sum(axis=0)
    high_p = ac_power[high].sum(axis=0)
    return high_p / (total + 1e-8)


def _fmt(vec: np.ndarray) -> str:
    """Format a per-joint vector for printing."""
    return "[" + ", ".join(f"{v:6.3f}" for v in vec) + "]"


def main() -> None:  # noqa: PLR0914
    """Load the eager checkpoint and report predicted-vs-GT smoothness."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = XR0.load_from_checkpoint(CKPT, dtype="float32", vlm_attn_implementation="sdpa")
    policy.to(device).eval()

    dm = LeRobotDataModule(
        root=DATASET_ROOT,
        train_batch_size=1,
        episodes=[EPISODE],
        val_split=0.0,
        data_format="physicalai",
    )
    dm.setup("fit")
    reformat_dataset_to_match_policy(policy, dm)

    torch.manual_seed(SEED)

    pred_metrics: dict[str, list[np.ndarray]] = {"vel": [], "acc": [], "jerk": [], "hf": []}
    gt_metrics: dict[str, list[np.ndarray]] = {"vel": [], "acc": [], "jerk": [], "hf": []}

    n_frames = 0
    for batch in dm.train_dataloader():
        gt = batch.action[..., :ACTION_DIM].reshape(-1, ACTION_DIM).cpu().numpy()

        with torch.no_grad():
            pred_t = policy.predict_action_chunk(batch)
        pred = pred_t[..., :ACTION_DIM].reshape(-1, ACTION_DIM).cpu().numpy()

        for name, chunk in (("pred", pred), ("gt", gt)):
            store = pred_metrics if name == "pred" else gt_metrics
            store["vel"].append(nth_diff_l1(chunk, 1))
            store["acc"].append(nth_diff_l1(chunk, 2))
            store["jerk"].append(nth_diff_l1(chunk, 3))
            store["hf"].append(high_freq_ratio(chunk, HF_CUTOFF_FRAC))

        n_frames += 1
        if MAX_FRAMES is not None and n_frames >= MAX_FRAMES:
            break

    def agg(store: dict[str, list[np.ndarray]], key: str) -> np.ndarray:
        return np.mean(np.stack(store[key], axis=0), axis=0)

    print(f"checkpoint : {CKPT}")
    print(f"episode    : {EPISODE}   frames averaged: {n_frames}   chunk len: {pred.shape[0]}")
    print(f"joints     : {list(JOINT_NAMES)}")
    print(f"hf cutoff  : {HF_CUTOFF_FRAC:.2f} x Nyquist\n")

    header = f"{'metric':<12}{'source':<8}per-joint" + " " * 40 + "overall"
    print(header)
    print("-" * len(header))
    for key, label in (("vel", "velocity"), ("acc", "acceler."), ("jerk", "jerk"), ("hf", "hf_ratio")):
        p = agg(pred_metrics, key)
        g = agg(gt_metrics, key)
        print(f"{label:<12}{'pred':<8}{_fmt(p)}   {p.mean():7.3f}")
        print(f"{'':<12}{'gt':<8}{_fmt(g)}   {g.mean():7.3f}")
        ratio = p.mean() / (g.mean() + 1e-8)
        print(f"{'':<12}{'ratio':<8}{'pred / gt = ':>{len(_fmt(g)) + 3}}{ratio:7.2f}x\n")

    jerk_ratio = agg(pred_metrics, "jerk").mean() / (agg(gt_metrics, "jerk").mean() + 1e-8)
    print(f"HEADLINE: predicted jerk is {jerk_ratio:.2f}x the dataset jerk "
          f"({'smooth' if jerk_ratio < 1.5 else 'ROUGH'}).")


if __name__ == "__main__":
    main()
