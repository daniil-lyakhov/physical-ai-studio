# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Conformance check for the *exported* XR0 OpenVINO IR.

Same idea as ``conformance_xr0.py`` (run one dataset frame, compare predicted vs
target actions) but instead of the Torch checkpoint it drives the exported
OpenVINO IR through the Runtime ``InferenceModel`` -- i.e. the deploy path from
``local_inf.py``. The IR's own preprocessor / OV tokenizer / delta postprocessor
run inside ``predict_action_chunk``, so we only feed it a raw observation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from lerobot.datasets.feature_utils import check_delta_timestamps, get_delta_indices
from openvino.preprocess import PrePostProcessor

from physicalai.data import LeRobotDataModule
from physicalai.inference import InferenceModel
from physicalai.inference.constants import IMAGES, STATE, TASK

MODEL_PATH = "/home/devuser/dlyakhov/physical-ai-studio/xr0_put_balls_to_box_irs_long_train"
MODEL_PATH = "/home/devuser/dlyakhov/physical-ai-studio/xr0_put_balls_to_box_irs_long_train_correct_export"
MODEL_PATH = "/home/devuser/dlyakhov/physical-ai-studio/library/export/xr0_30_09"

DATASET_ROOT = "/home/devuser/dlyakhov/datasets/Put-different-balls-to-the-box"
EPISODE = 3
ACTION_DIM = 6
# Predicted action-chunk length the IR emits (XR0 action_shape[-2]); used to
# expand the dataset target into a full chunk without loading the checkpoint.
CHUNK_SIZE = 30
DEVICE = "CPU"  # "GPU" + bf16 mirrors deployment; "CPU" is portable
PRECISION_HINT = "f32"
# The runtime preprocessor downscales with cv2.INTER_CUBIC (no prefilter) while
# training used antialiased bicubic, so the deployed model sees aliased images.
# INTER_AREA prefilters on downscale; flip this to measure the action impact.
USE_AREA_RESIZE = True
# The IR draws its rectified-flow noise internally. It is dumped here so
# conformance_xr0.py can replay the exact same noise through the eager model,
# making the two runs directly comparable instead of two independent samples.
NOISE_PATH = Path("xr0_shared_noise.npy")
# The export bakes ``global_seed=0`` / ``op_seed=0``, which OpenVINO reads as
# "fresh seed every run". Non-zero seeds make the drawn noise reproducible.
OV_RANDOM_UNIFORM_GLOBAL_SEED = 42
OV_RANDOM_UNIFORM_OP_SEED = 7
# Must match conformance_xr0.py so both scripts select the same shuffled frame
# (identical reference target) for a fair comparison.
SEED = 0


def _set_action_chunk(dm: LeRobotDataModule, chunk_size: int) -> None:
    """Expand the dataset ``action`` feature into a ``chunk_size`` chunk.

    Mirrors ``reformat_dataset_to_match_policy`` for the action key only
    (``action_delta_indices == range(chunk_size)`` for XR0) so the target is a
    full chunk -- without needing to load the multi-GB checkpoint just to read
    that one property.
    """
    dataset = dm.train_dataset
    fps = dataset.fps
    delta_timestamps = {"action": [i / fps for i in range(chunk_size)]}
    check_delta_timestamps(delta_timestamps, fps, dataset.tolerance_s)
    dataset.delta_indices = get_delta_indices(delta_timestamps, fps)


def _patch_area_resize() -> None:
    """Swap the runtime preprocessor's cv2.INTER_CUBIC downscale for INTER_AREA."""
    import cv2
    from physicalai.inference.preprocessors import xr0 as runtime_xr0

    original_resize = runtime_xr0._resize_image  # noqa: SLF001

    def _area_resize(image, factor, max_pixels):  # noqa: ANN001, ANN202
        resized = original_resize(image, factor=factor, max_pixels=max_pixels)
        return cv2.resize(image, (resized.shape[1], resized.shape[0]), interpolation=cv2.INTER_AREA)

    runtime_xr0._resize_image = _area_resize  # noqa: SLF001


def _find_noise_node(model: ov.Model) -> ov.Node:
    """Locate the Box-Muller Gaussian-noise node in the exported IR.

    ``torch.randn`` lowers to a ``RandomUniform`` followed by
    ``sqrt(-2*log(u1)) * cos(2*pi*u2)``; the final ``Multiply`` is the noise.

    Returns:
        The ``Multiply`` node producing the rectified-flow starting noise.

    Raises:
        RuntimeError: If the Box-Muller ``Multiply`` cannot be found.
    """
    for op in model.get_ops():
        if op.get_type_name() != "Multiply":
            continue
        parents = {op.input_value(i).get_node().get_type_name() for i in range(len(op.inputs()))}
        if {"Sqrt", "Cos"} <= parents:
            return op
    msg = "Could not locate the Box-Muller noise node (Sqrt*Cos) in the IR."
    raise RuntimeError(msg)


def _run_ir_with_pinned_noise(ir_xml: Path, graph_inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run the IR with a pinned noise seed, returning its outputs plus the noise.

    Returns:
        The graph outputs keyed by name, with the drawn noise under ``"noise"``.

    Raises:
        RuntimeError: If the IR has no ``RandomUniform`` node to pin.
    """
    core = ov.Core()
    model = core.read_model(ir_xml)

    for op in model.get_ops():
        if op.get_type_name() == "RandomUniform":
            op.set_attribute("global_seed", OV_RANDOM_UNIFORM_GLOBAL_SEED)
            op.set_attribute("op_seed", OV_RANDOM_UNIFORM_OP_SEED)
            break
    else:
        msg = "Could not locate a RandomUniform node to pin the noise seed in the IR."
        raise RuntimeError(msg)

    output_names = [output.get_any_name() for output in model.outputs]
    feed = {name: graph_inputs[name] for name in (inp.get_any_name() for inp in model.inputs)}
    # Expose the noise as an extra output, cast to f32 so NumPy can read it.
    model.add_outputs(_find_noise_node(model).output(0))
    ppp = PrePostProcessor(model)
    ppp.output(len(output_names)).tensor().set_element_type(ov.Type.f32)
    model = ppp.build()

    compiled = core.compile_model(model, DEVICE, {"INFERENCE_PRECISION_HINT": PRECISION_HINT})
    result = compiled(feed)
    outputs = {name: np.asarray(result[compiled.output(i)]) for i, name in enumerate(output_names)}
    outputs["noise"] = np.asarray(result[compiled.output(len(output_names))])
    return outputs


def main() -> None:
    dm = LeRobotDataModule(
        root=DATASET_ROOT,
        train_batch_size=1,
        episodes=[EPISODE],
        val_split=0.0,
        data_format="physicalai",
    )
    dm.setup("fit")
    # Expand the action target into a full chunk directly (no checkpoint needed;
    # inference is the OV IR, not a Torch policy).
    _set_action_chunk(dm, CHUNK_SIZE)

    # Seed the shuffled sampler so we pull the same frame as conformance_xr0.py.
    torch.manual_seed(SEED)
    batch = next(iter(dm.train_dataloader()))
    target = batch.action[..., :ACTION_DIM].reshape(-1, ACTION_DIM).cpu().numpy()

    observation = {
        IMAGES: {view: img[0].cpu().numpy() for view, img in batch.images.items()},
        STATE: batch.state[0].cpu().numpy(),
        TASK: batch.task,
    }

    if USE_AREA_RESIZE:
        _patch_area_resize()
    model = InferenceModel(MODEL_PATH, device=DEVICE, INFERENCE_PRECISION_HINT=PRECISION_HINT)

    # Drive the exported pipeline by hand so the IR's internal noise can be
    # pinned and dumped: preprocessors -> IR (fixed seed) -> postprocessors.
    graph_inputs: dict = observation
    for preprocessor in model.preprocessors:
        graph_inputs = preprocessor(graph_inputs)

    manifest = json.loads((Path(MODEL_PATH) / "manifest.json").read_text())
    ir_xml = Path(MODEL_PATH) / manifest["model"]["artifacts"]["openvino"]
    outputs = _run_ir_with_pinned_noise(ir_xml, graph_inputs)
    np.save(NOISE_PATH, outputs.pop("noise"))
    print(f"noise dumped to {NOISE_PATH}")

    for postprocessor in model.postprocessors:
        outputs = postprocessor(outputs)
    pred = np.asarray(outputs["action"]).reshape(-1, outputs["action"].shape[-1])[..., :ACTION_DIM]

    target = target[: pred.shape[0]]
    err = np.abs(pred - target)
    print(f"pred shape {pred.shape}  target shape {target.shape}")
    print(f"per-dim MAE: {np.round(err.mean(0), 4)}")
    print(f"overall MAE: {err.mean():.4f}   max abs err: {err.max():.4f}")
    for t in (0, target.shape[0] // 2, target.shape[0] - 1):
        print(f"\nt={t}")
        print(f"  target: {np.round(target[t], 4)}")
        print(f"  pred  : {np.round(pred[t], 4)}")


if __name__ == "__main__":
    main()
