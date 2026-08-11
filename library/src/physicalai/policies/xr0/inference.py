# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runtime inference components for the exported XR0 OpenVINO model.

The exported XR0 OpenVINO graph is self-contained: the Qwen3-VL vision tower, the
language model, the 3D MRoPE ``position_ids`` and the image-token scatter all run
*inside* the graph. Its inputs are only the tokenized prompt plus the raw
pixels/state (``input_ids`` / ``attention_mask`` / ``pixel_values`` / ``state``)
and its output is the still-normalized, ``max_action_dim``-wide action chunk.

These two components let the Runtime :class:`~physicalai.inference.model.InferenceModel`
reconstruct the XR0 inference pipeline directly from ``manifest.json`` (via
``class_path`` + ``init_args``), so the exported model runs natively in the gym
without the Torch policy:

* :class:`XR0InferencePreprocessor` reuses the model-free training
  :class:`~physicalai.policies.xr0.preprocessor.XR0Preprocessor` (Qwen3-VL
  tokenization + image resize + state padding) to build the graph inputs, then
  right-pads the token sequence to the fixed length the graph was baked for and
  drops ``image_grid_thw`` (supplied as a baked constant inside the graph).
* :class:`XR0InferencePostprocessor` denormalizes the predicted action with the
  source mean/std convention and slices it back to the real action dimension.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from physicalai.inference.constants import ACTION
from physicalai.inference.postprocessors.base import Postprocessor
from physicalai.inference.preprocessors.base import Preprocessor

from physicalai.data.observation import IMAGES, STATE, TASK

from .io import ACTION_EPS
from .preprocessor import XR0Preprocessor

if TYPE_CHECKING:
    from collections.abc import Sequence

# Names of the exported OpenVINO graph inputs the preprocessor must emit.
_MODEL_INPUT_KEYS = ("input_ids", "attention_mask", "pixel_values", "state")
# Prompt inputs that are integer-typed (the rest are float32).
_INT_INPUT_KEYS = ("input_ids", "attention_mask")


def _as_tensor(value: object) -> torch.Tensor:
    """Convert a NumPy array / tensor / sequence into a CPU ``torch.Tensor``.

    Returns:
        The value as a CPU ``torch.Tensor``.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu")
    return torch.as_tensor(np.asarray(value))


class XR0InferencePreprocessor(Preprocessor):
    """Build the exported XR0 graph inputs from a raw observation dict.

    Wraps the model-free training :class:`XR0Preprocessor`: converts the Runtime
    observation (NumPy) into the batch layout it expects, runs it to obtain
    ``input_ids`` / ``attention_mask`` / ``pixel_values`` / ``state``, right-pads
    the prompt to ``seq_len`` (the baked graph length, masked by
    ``attention_mask``) and returns NumPy arrays with the exact dtypes the graph
    expects. ``image_grid_thw`` is intentionally dropped -- the graph carries it
    as a baked constant.

    Args:
        camera_views: Ordered view names embedded into the prompt (must match the
            views used at export time so the baked image geometry stays valid).
        max_state_dim: State dimension after padding.
        max_action_dim: Action dimension after padding.
        seq_len: Fixed, right-padded prompt length the graph was baked for.
        processor_name: HuggingFace id of the Qwen3-VL processor.
        image_factor: Patch-alignment factor for image resizing.
        image_max_pixels: Maximum image area for image resizing.
        normalize_state: Whether the exported model expects normalized state.
            Defaults to False (raw state), matching the training default.
        state_mean: Baked ``max_state_dim`` state mean (identity when disabled).
        state_std: Baked ``max_state_dim`` state std (identity when disabled).
    """

    def __init__(
        self,
        camera_views: Sequence[str] = ("base", "wrist_left"),
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        seq_len: int = 256,
        processor_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        image_factor: int = 32,
        image_max_pixels: int = 90000,
        *,
        normalize_state: bool = False,
        state_mean: Sequence[float] | None = None,
        state_std: Sequence[float] | None = None,
    ) -> None:
        """Initialize the XR0 inference preprocessor.

        Raises:
            ValueError: If ``camera_views`` is empty or ``seq_len`` is not positive.
        """
        super().__init__()
        camera_views = tuple(camera_views)
        if not camera_views:
            msg = "XR0InferencePreprocessor requires at least one camera view"
            raise ValueError(msg)
        if int(seq_len) <= 0:
            msg = f"seq_len must be a positive integer, got {seq_len!r}"
            raise ValueError(msg)

        self._camera_views = camera_views
        self._seq_len = int(seq_len)
        self._preprocessor = XR0Preprocessor(
            camera_views=camera_views,
            max_state_dim=int(max_state_dim),
            max_action_dim=int(max_action_dim),
            features=None,
            image_factor=int(image_factor),
            image_max_pixels=int(image_max_pixels),
            processor_name=str(processor_name),
            normalize_state=bool(normalize_state),
            state_mean=state_mean,
            state_std=state_std,
        )
        self._pad_id: int | None = None

    @property
    def pad_id(self) -> int:
        """Lazily resolve the tokenizer pad id used for right-padding.

        Returns:
            The tokenizer ``pad_token_id`` (or ``0`` when undefined).
        """
        if self._pad_id is None:
            self._pad_id = self._preprocessor.processor.tokenizer.pad_token_id or 0
        return self._pad_id

    @staticmethod
    def _to_batch(inputs: dict[str, object]) -> dict[str, object]:
        """Assemble the Torch batch the training preprocessor consumes.

        Bridges the Runtime observation (NumPy, nested ``images`` dict, string
        ``task``) into the ``STATE`` / ``TASK`` / flattened ``images.*`` layout
        expected by :class:`XR0Preprocessor`, adding a leading batch dim when the
        observation is a single unbatched frame.

        Returns:
            A batch dict with ``state``, ``task`` and one ``images.<view>`` tensor
            per camera view.

        Raises:
            ValueError: If the observation has no state entry.
        """
        state_value = inputs.get(STATE)
        if state_value is None:
            msg = "XR0 inference requires a 'state' observation"
            raise ValueError(msg)
        state = _as_tensor(state_value).to(torch.float32)
        if state.ndim == 1:  # (D,) -> (1, D)
            state = state.unsqueeze(0)

        batch: dict[str, object] = {STATE: state}

        images_value = inputs.get(IMAGES)
        if isinstance(images_value, dict):
            image_items = {f"{IMAGES}.{view}": array for view, array in images_value.items()}
        else:
            image_items = {
                key: value
                for key, value in inputs.items()
                if isinstance(key, str) and key.startswith(f"{IMAGES}.") and "is_pad" not in key
            }
        if not image_items:
            msg = "XR0 inference requires at least one image observation"
            raise ValueError(msg)
        for key, array in image_items.items():
            image = _as_tensor(array)
            if image.ndim == 3:  # (C, H, W) -> (1, C, H, W)  # noqa: PLR2004
                image = image.unsqueeze(0)
            batch[key] = image

        task = inputs.get(TASK)
        if task is None:
            batch[TASK] = [""]
        elif isinstance(task, str):
            batch[TASK] = [task]
        elif isinstance(task, np.ndarray):
            batch[TASK] = [str(entry) for entry in np.atleast_1d(task).tolist()]
        elif isinstance(task, (list, tuple)):
            batch[TASK] = [str(entry) for entry in task]
        else:
            batch[TASK] = [str(task)]

        return batch

    def __call__(self, inputs: dict[str, object]) -> dict[str, np.ndarray]:
        """Transform a raw observation into the exported graph inputs.

        Args:
            inputs: Observation dict with a ``state`` array, ``images`` (nested
                dict or flattened ``images.*`` keys) and a ``task`` string.

        Returns:
            Dict with ``input_ids`` / ``attention_mask`` (int64) and
            ``pixel_values`` / ``state`` (float32) NumPy arrays, with the prompt
            right-padded to ``seq_len``.

        Raises:
            ValueError: If the tokenized prompt is longer than ``seq_len``.
        """
        batch = self._to_batch(inputs)
        with torch.no_grad():
            processed = self._preprocessor(batch)

        input_ids = processed["input_ids"]
        attention_mask = processed["attention_mask"]
        cur_len = input_ids.shape[1]
        if cur_len > self._seq_len:
            msg = (
                f"Prompt ({cur_len} tokens) exceeds the baked seq_len={self._seq_len}; re-export with a larger length."
            )
            raise ValueError(msg)
        pad = self._seq_len - cur_len
        if pad:
            input_ids = F.pad(input_ids, (0, pad), value=self.pad_id)
            attention_mask = F.pad(attention_mask, (0, pad), value=0)

        return {
            "input_ids": np.ascontiguousarray(input_ids.cpu().numpy().astype(np.int64)),
            "attention_mask": np.ascontiguousarray(attention_mask.cpu().numpy().astype(np.int64)),
            "pixel_values": np.ascontiguousarray(processed["pixel_values"].cpu().numpy().astype(np.float32)),
            "state": np.ascontiguousarray(processed["state"].cpu().numpy().astype(np.float32)),
        }


class XR0InferencePostprocessor(Postprocessor):
    """Denormalize the exported XR0 graph's action output.

    Inverts the source action normalization (``action * (std + eps) + mean``) and
    slices the padded action back to its real dimension, mirroring the training
    :class:`~physicalai.policies.xr0.preprocessor.XR0Postprocessor`.

    Args:
        action_mean: Per-dimension action mean (padded to ``max_action_dim``).
        action_std: Per-dimension action std (padded to ``max_action_dim``).
        action_dim: Real (unpadded) action dimension; when set, the output is
            sliced to it. ``None`` keeps the padded width.
        action_eps: Numerical epsilon added to the std (matches training).
    """

    def __init__(
        self,
        action_mean: Sequence[float],
        action_std: Sequence[float],
        action_dim: int | None = None,
        action_eps: float = ACTION_EPS,
    ) -> None:
        """Initialize the XR0 inference postprocessor.

        Raises:
            ValueError: If ``action_mean`` and ``action_std`` shapes differ.
        """
        super().__init__()
        self._mean = np.asarray(action_mean, dtype=np.float32)
        self._std = np.asarray(action_std, dtype=np.float32)
        if self._mean.shape != self._std.shape:
            msg = f"action_mean {self._mean.shape} and action_std {self._std.shape} must have the same shape"
            raise ValueError(msg)
        self._action_dim = int(action_dim) if action_dim is not None else None
        self._eps = float(action_eps)

    def __call__(self, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Denormalize and unpad the predicted action chunk.

        Args:
            outputs: Runner output dict containing an ``action`` array.

        Returns:
            The outputs dict with the ``action`` denormalized (and sliced to the
            real action dimension when known).
        """
        action = outputs.get(ACTION)
        if action is None:
            return outputs
        action = np.asarray(action, dtype=np.float32)
        action = action * (self._std + self._eps) + self._mean
        if self._action_dim is not None:
            action = action[..., : self._action_dim]
        result = dict(outputs)
        result[ACTION] = action
        return result
