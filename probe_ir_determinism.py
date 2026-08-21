# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Is the exported XR0 IR deterministic per observation?

Hypothesis (from run_obs/chunks.json): consecutive plans disagree by up to
~66 units on the SAME future timesteps despite observations only 2 frames apart.
That is the signature of FRESH rectified-flow noise drawn every inference call.

This feeds ONE recorded real observation (run_obs/obs_00000.npz) through the SAME
deployed InferenceModel N times and reports how much the output changes:
  * diff == 0  -> IR noise is fixed/deterministic; the inter-chunk disagreement is
                  from observation change / delta re-anchoring (look elsewhere).
  * diff  > 0  -> IR draws fresh noise per call -> THIS is the shaking source.
                  Fix = bake a fixed/constant noise into the export (or seed it),
                  so consecutive plans for similar observations agree.
"""

from __future__ import annotations

import numpy as np

from physicalai.inference import InferenceModel
from physicalai.inference.constants import IMAGES, STATE, TASK  # noqa: F401

# Match run_local.py.
MODEL_PATH = "/home/dupeljan/Projects/home_robot/models/xr0_v1/xr0_put_balls_to_box_irs_normalized_full"
OBS_NPZ = "run_obs/obs_00000.npz"
TASK_STR = "Put the green ball to the black box"
DEVICE = "GPU.0"
PRECISION_HINT = "f16"
N_RUNS = 5
ACTION_DIM = 6


def main() -> None:
    d = np.load(OBS_NPZ)
    # Rebuild the observation exactly like the runtime's PolicySource does:
    # multi-camera -> ``images.<name>`` keys, each batched (1, H, W, 3); state (1, D).
    images = {k[len("img__") :]: d[k] for k in d if k.startswith("img__")}
    observation: dict = {STATE: np.asarray(d["state"], dtype=np.float32)[np.newaxis]}
    if len(images) > 1:
        for name, arr in images.items():
            observation[f"{IMAGES}.{name}"] = arr[np.newaxis]
    else:
        observation[IMAGES] = next(iter(images.values()))[np.newaxis]
    observation[TASK] = [TASK_STR]

    adapter_kwargs = {"INFERENCE_PRECISION_HINT": PRECISION_HINT} if DEVICE.startswith("GPU") else {}
    model = InferenceModel(MODEL_PATH, device=DEVICE, **adapter_kwargs)

    runs = np.stack([model.predict_action_chunk(observation)[..., :ACTION_DIM] for _ in range(N_RUNS)])
    print(f"ran {N_RUNS}x on the SAME observation; chunk shape {runs.shape[1:]}")
    # Spread across runs, per joint (0 == deterministic).
    spread = runs.std(axis=0)  # (chunk, joints)
    print("per-joint max std across runs:", np.round(spread.max(axis=0), 4))
    print("per-joint mean std across runs:", np.round(spread.mean(axis=0), 4))
    print("overall max abs diff run0 vs run1:", np.round(np.abs(runs[0] - runs[1]).max(), 4))
    print("chunk[0] run0:", np.round(runs[0, 0], 3))
    print("chunk[0] run1:", np.round(runs[1, 0], 3))


if __name__ == "__main__":
    main()
