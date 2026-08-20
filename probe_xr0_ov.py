# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Probe an exported XR0 OpenVINO IR to find why delta reconstruction fails.

Run on the machine that has the export (and optionally the checkpoint):

    python probe_xr0_ov.py <export_dir> [checkpoint.ckpt]

It answers three questions without needing the dataset:
  1. What does the manifest bake?  -> xr0_denormalize.action_mode, action_dim,
     action_mean/std shape;  xr0 preprocessor.normalize_state / camera_views.
  2. What ports does the IR expose? -> is there a state pass-through output, and
     under what name.
  3. Empirically: feed a synthetic obs with a KNOWN state, capture the *raw*
     graph outputs (normalized delta + echoed state) and the *final* action, and
     check whether the current state was actually added back
     (final[t=0] ~= state  =>  delta re-added;  final[t=0] ~= 0  =>  it was not).
"""

from __future__ import annotations

import sys

import numpy as np

from physicalai.inference import InferenceModel
from physicalai.inference.constants import ACTION, IMAGES, STATE, TASK


def _fmt(value: object) -> str:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return repr(value)
    return f"<array shape={arr.shape} dtype={arr.dtype} min={arr.min():.4g} max={arr.max():.4g}>"


def _dump_specs(title: str, specs: list) -> None:  # noqa: ANN001
    print(f"\n== {title} ({len(specs)}) ==")
    for spec in specs:
        print(f"  type={getattr(spec, 'type', spec)!r}")
        init_args = dict(getattr(spec, "init_args", {}) or {})
        for key, val in init_args.items():
            print(f"      {key} = {_fmt(val)}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    export_dir = sys.argv[1]

    model = InferenceModel(export_dir, device="CPU")

    _dump_specs("PREPROCESSORS", model.manifest.model.preprocessors)
    _dump_specs("POSTPROCESSORS", model.manifest.model.postprocessors)

    print("\n== ADAPTER PORTS ==")
    print(f"  inputs : {model.adapter.input_names}")
    print(f"  outputs: {getattr(model.adapter, 'output_names', '<n/a>')}")

    # --- empirical delta check ------------------------------------------------
    # Distinctive per-joint state so a re-add is unmistakable in the output.
    state_vec = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float32)
    views = model.manifest.model.preprocessors  # camera_views come from init_args
    cam = ("base", "wrist_left")
    for spec in views:
        ca = (getattr(spec, "init_args", {}) or {}).get("camera_views")
        if ca:
            cam = tuple(ca)
            break
    obs = {
        IMAGES: {v: np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8) for v in cam},
        STATE: state_vec,
        TASK: "probe task",
    }

    # Replicate InferenceModel.__call__ but capture the RAW graph outputs.
    x: dict = obs
    for pre in model.preprocessors:
        x = pre(x)
    prepared = model._prepare_inputs(x)  # noqa: SLF001
    raw = model.runner.run(model.adapter, prepared)

    print("\n== RAW GRAPH OUTPUTS (pre-postproc) ==")
    for key, val in raw.items():
        print(f"  {key}: {_fmt(val)}")
        if np.asarray(val).size <= 64:
            print(f"      values={np.asarray(val).ravel()[:32]}")

    final = raw
    for post in model.postprocessors:
        final = post(final)
    action = np.asarray(final[ACTION])
    a0 = action.reshape(-1, action.shape[-1])[0]

    print("\n== FINAL ACTION ==")
    print(f"  shape={action.shape}")
    print(f"  state (input)   = {state_vec}")
    print(f"  action[t=0]     = {np.round(a0[: state_vec.shape[0]], 4)}")
    diff = a0[: state_vec.shape[0]] - state_vec
    print(f"  action[0]-state = {np.round(diff, 4)}")
    print(
        "\nVERDICT: "
        + (
            "state WAS added (delta ok)"
            if np.abs(diff).mean() < np.abs(state_vec).mean() * 0.5
            else "state NOT added -> postprocessor is effectively ABSOLUTE, "
            "or the echoed state is normalized (~0). Check action_mode above and "
            "the 'state' passthrough values in RAW GRAPH OUTPUTS."
        )
    )

    if len(sys.argv) > 2:
        import torch

        ckpt = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
        hp = ckpt.get("hyper_parameters", {}) if isinstance(ckpt, dict) else {}
        print("\n== CHECKPOINT HPARAMS ==")
        print(f"  action_mode        = {hp.get('action_mode')!r}")
        print(f"  normalize_state    = {hp.get('normalize_state')!r}")
        for key in ("action_delta_mean", "action_delta_std"):
            val = hp.get(key)
            print(f"  {key} present = {val is not None}" + (f"  {_fmt(val)}" if val is not None else ""))


if __name__ == "__main__":
    main()
