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
# Zero-phase low-pass cutoff (fraction of Nyquist) applied to the *predicted*
# chunk before it is executed. The whole 30-step chunk is available before the
# robot moves, so a non-causal (zero-phase) filter is deployable per chunk. The
# measured roughness is near-Nyquist jitter on top of a correct low-frequency
# trajectory, so cutting above this keeps the motion and removes the shake.
SMOOTH_CUTOFF_FRAC = 0.15
# Width (fraction of Nyquist) of the raised-cosine rolloff, to avoid ringing.
SMOOTH_TRANSITION_FRAC = 0.10
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


def lowpass_chunk(chunk: np.ndarray, cutoff_frac: float, transition_frac: float) -> np.ndarray:
    """Zero-phase raised-cosine low-pass along the time axis of a chunk.

    Filtering happens on the whole predicted chunk before execution, so it is
    non-causal (zero phase = no lag) yet deployable per chunk.

    The endpoint-to-endpoint *linear trend* is removed before the FFT and added
    back afterwards. An action chunk is not periodic (the arm has moved, so
    ``chunk[0] != chunk[-1]``); FFT filtering the raw chunk treats that gap as a
    wrap-around discontinuity and produces Gibbs ringing that *inflates*
    velocity/acceleration. Detrending makes the residual start and end near zero
    so the periodic extension is smooth; the linear trend carries constant
    velocity and zero jerk, so re-adding it introduces no roughness.

    Args:
        chunk: ``(T, D)`` action chunk.
        cutoff_frac: Passband edge as a fraction of Nyquist (0..0.5).
        transition_frac: Rolloff width as a fraction of Nyquist.

    Returns:
        ``(T, D)`` smoothed chunk (same shape/units).
    """
    length = chunk.shape[0]
    if length < 3:  # noqa: PLR2004
        return chunk.copy()
    ramp = np.linspace(0.0, 1.0, length)[:, None]
    trend = chunk[:1] + (chunk[-1:] - chunk[:1]) * ramp
    residual = chunk - trend
    spectrum = np.fft.rfft(residual, axis=0)
    freqs = np.fft.rfftfreq(length)  # 0 .. 0.5 cycles/frame
    lo = cutoff_frac - transition_frac / 2.0
    hi = cutoff_frac + transition_frac / 2.0
    gain = np.ones_like(freqs)
    band = (freqs >= lo) & (freqs <= hi)
    gain[band] = 0.5 * (1.0 + np.cos(np.pi * (freqs[band] - lo) / max(hi - lo, 1e-8)))
    gain[freqs > hi] = 0.0
    smoothed = np.fft.irfft(spectrum * gain[:, None], n=length, axis=0)
    return smoothed + trend


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
    smooth_metrics: dict[str, list[np.ndarray]] = {"vel": [], "acc": [], "jerk": [], "hf": []}
    gt_metrics: dict[str, list[np.ndarray]] = {"vel": [], "acc": [], "jerk": [], "hf": []}
    # Per-joint |pred_smooth - pred|: how far the filter moves the trajectory.
    shift_acc: list[np.ndarray] = []
    # Per-joint per-chunk MAE vs GT, for raw and low-passed predictions.
    mae_pred_acc: list[np.ndarray] = []
    mae_smooth_acc: list[np.ndarray] = []

    n_frames = 0
    for batch in dm.train_dataloader():
        gt = batch.action[..., :ACTION_DIM].reshape(-1, ACTION_DIM).cpu().numpy()

        with torch.no_grad():
            pred_t = policy.predict_action_chunk(batch)
        pred = pred_t[..., :ACTION_DIM].reshape(-1, ACTION_DIM).cpu().numpy()
        pred_sm = lowpass_chunk(pred, SMOOTH_CUTOFF_FRAC, SMOOTH_TRANSITION_FRAC)

        for chunk, store in ((pred, pred_metrics), (pred_sm, smooth_metrics), (gt, gt_metrics)):
            store["vel"].append(nth_diff_l1(chunk, 1))
            store["acc"].append(nth_diff_l1(chunk, 2))
            store["jerk"].append(nth_diff_l1(chunk, 3))
            store["hf"].append(high_freq_ratio(chunk, HF_CUTOFF_FRAC))

        shift_acc.append(np.abs(pred_sm - pred).mean(axis=0))
        mae_pred_acc.append(np.abs(pred - gt).mean(axis=0))
        mae_smooth_acc.append(np.abs(pred_sm - gt).mean(axis=0))

        n_frames += 1
        if MAX_FRAMES is not None and n_frames >= MAX_FRAMES:
            break

    def agg(store: dict[str, list[np.ndarray]], key: str) -> np.ndarray:
        return np.mean(np.stack(store[key], axis=0), axis=0)

    print(f"checkpoint : {CKPT}")
    print(f"episode    : {EPISODE}   frames averaged: {n_frames}   chunk len: {pred.shape[0]}")
    print(f"joints     : {list(JOINT_NAMES)}")
    print(f"hf cutoff  : {HF_CUTOFF_FRAC:.2f} x Nyquist\n")

    print(f"smoothing  : zero-phase low-pass @ {SMOOTH_CUTOFF_FRAC:.2f}x Nyquist "
          f"(rolloff {SMOOTH_TRANSITION_FRAC:.2f})\n")

    header = f"{'metric':<12}{'source':<10}per-joint" + " " * 40 + "overall"
    print(header)
    print("-" * len(header))
    for key, label in (("vel", "velocity"), ("acc", "acceler."), ("jerk", "jerk"), ("hf", "hf_ratio")):
        p = agg(pred_metrics, key)
        s = agg(smooth_metrics, key)
        g = agg(gt_metrics, key)
        print(f"{label:<12}{'pred':<10}{_fmt(p)}   {p.mean():7.3f}")
        print(f"{'':<12}{'pred+lp':<10}{_fmt(s)}   {s.mean():7.3f}")
        print(f"{'':<12}{'gt':<10}{_fmt(g)}   {g.mean():7.3f}")
        pad = len(_fmt(g)) + 5
        print(f"{'':<12}{'ratio':<10}{'pred / gt   = ':>{pad}}{p.mean() / (g.mean() + 1e-8):7.2f}x")
        print(f"{'':<12}{'':<10}{'pred+lp / gt = ':>{pad}}{s.mean() / (g.mean() + 1e-8):7.2f}x\n")

    shift = np.mean(np.stack(shift_acc, axis=0), axis=0)
    print(f"filter shift |pred+lp - pred| per joint: {_fmt(shift)}   overall {shift.mean():7.3f}")
    print("(how many action units the low-pass moves the trajectory; small = safe)\n")

    mae_pred = np.mean(np.stack(mae_pred_acc, axis=0), axis=0)
    mae_smooth = np.mean(np.stack(mae_smooth_acc, axis=0), axis=0)
    print(f"{'MAE vs GT':<12}{'pred':<10}{_fmt(mae_pred)}   {mae_pred.mean():7.3f}")
    print(f"{'':<12}{'pred+lp':<10}{_fmt(mae_smooth)}   {mae_smooth.mean():7.3f}")
    print(f"{'':<12}{'delta':<10}{_fmt(mae_smooth - mae_pred)}   {mae_smooth.mean() - mae_pred.mean():+7.3f}")
    print("(per-chunk mean |pred - gt|; delta>0 means the filter cost precision)\n")

    raw_jerk = agg(pred_metrics, "jerk").mean() / (agg(gt_metrics, "jerk").mean() + 1e-8)
    lp_jerk = agg(smooth_metrics, "jerk").mean() / (agg(gt_metrics, "jerk").mean() + 1e-8)
    print(f"HEADLINE: jerk ratio  raw={raw_jerk:.2f}x  ->  with low-pass={lp_jerk:.2f}x "
          f"({'smooth' if lp_jerk < 1.5 else 'still ROUGH'}).")


if __name__ == "__main__":
    main()
