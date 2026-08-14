# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runtime inference components for the exported XR0 OpenVINO model.

The exported XR0 OpenVINO graph is self-contained: the Qwen3-VL vision tower, the
language model, the 3D MRoPE ``position_ids`` and the image-token scatter all run
*inside* the graph. Its inputs are the tokenized prompt plus the raw pixels/state
(``tokenized_prompt`` / ``tokenized_prompt_mask`` / ``pixel_values`` / ``state``)
and its output is the still-normalized, ``max_action_dim``-wide action chunk.

These two components let the Runtime :class:`~physicalai.inference.model.InferenceModel`
reconstruct the XR0 inference pipeline directly from ``manifest.json`` (via
``class_path`` + ``init_args``), so the exported model runs natively in the gym
without the Torch policy or the full Qwen3-VL processor:

* :class:`XR0InferencePreprocessor` is a lightweight, torch-free NumPy component:
  it resizes the camera views into the Qwen3-VL ``pixel_values`` grid, pads the
  ``state`` and renders the multi-view chat prompt as a plain ``task`` string.
  It does **not** tokenize -- a sibling OpenVINO tokenizer (``tokenizer.xml``,
  exported next to the graph) turns ``task`` into the graph's ``tokenized_prompt``
  / ``tokenized_prompt_mask`` inputs. The image geometry and normalization
  constants are baked at export time so no HuggingFace processor is loaded at
  inference; ``image_grid_thw`` is carried inside the graph as a baked constant.
* :class:`XR0InferencePostprocessor` denormalizes the predicted action with the
  source mean/std convention and slices it back to the real action dimension.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from physicalai.data.observation import IMAGES, STATE, TASK
from physicalai.inference.constants import ACTION
from physicalai.inference.postprocessors.base import Postprocessor
from physicalai.inference.preprocessors.base import Preprocessor
from PIL import Image

from .io import ACTION_EPS, build_pixel_grid, resize_image
from .prompt import image_pad_count, render_chat_prompt

if TYPE_CHECKING:
    from collections.abc import Sequence

# Qwen3-VL image-normalization + geometry constants. These are baked into the
# manifest ``init_args`` at export time from the source image processor; the
# defaults mirror ``Qwen/Qwen3-VL-4B-Instruct`` so the component is usable
# standalone.
_QWEN3VL_IMAGE_MEAN = (0.5, 0.5, 0.5)
_QWEN3VL_IMAGE_STD = (0.5, 0.5, 0.5)
_QWEN3VL_RESCALE_FACTOR = 1.0 / 255.0
_QWEN3VL_PATCH_SIZE = 16
_QWEN3VL_MERGE_SIZE = 2

_TEMPORAL_IMAGE_NDIM = 5
_BATCHED_IMAGE_NDIM = 4
_CHANNELS_FIRST_NDIM = 3
_TEMPORAL_STATE_NDIM = 3


def _to_pil(array: object) -> Image.Image:
    """Convert a NumPy image (``(H,W,C)`` / ``(C,H,W)`` / batched / temporal) to PIL.

    Mirrors the training :func:`~physicalai.policies.xr0.preprocessor._to_pil`
    convention (channels-first detection, ``[0, 1]`` float rescaling, grayscale
    expansion) so the resized geometry matches the baked export exactly.

    Returns:
        The image as an RGB PIL ``Image``.
    """
    arr = np.asarray(array)
    if arr.ndim == _TEMPORAL_IMAGE_NDIM:  # (B, T, C, H, W) -> last frame of first sample
        arr = arr[0, -1]
    elif arr.ndim == _BATCHED_IMAGE_NDIM:  # (T|B, C, H, W) -> last frame
        arr = arr[-1]
    if arr.ndim == _CHANNELS_FIRST_NDIM and arr.shape[0] in {1, 3}:  # channels-first
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return Image.fromarray(arr)


class XR0InferencePreprocessor(Preprocessor):
    """Build the exported XR0 graph inputs from a raw observation dict.

    Lightweight, torch-free NumPy preprocessor: resizes the camera views into the
    Qwen3-VL ``pixel_values`` grid, pads/normalizes the ``state`` and renders the
    multi-view chat prompt as a plain ``task`` string. It does **not** tokenize --
    a sibling OpenVINO tokenizer (``tokenizer.xml``) turns ``task`` into
    ``tokenized_prompt`` / ``tokenized_prompt_mask``. The image geometry and
    normalization constants are baked at export time so no HuggingFace processor
    is loaded at inference. ``image_grid_thw`` is carried inside the graph as a
    baked constant.

    Args:
        camera_views: Ordered view names embedded into the prompt (must match the
            views used at export time so the baked image geometry stays valid).
        max_state_dim: State dimension after padding.
        image_factor: Patch-alignment factor for image resizing.
        image_max_pixels: Maximum image area for image resizing.
        image_mean: Per-channel image mean (baked from the source image processor).
        image_std: Per-channel image std (baked from the source image processor).
        rescale_factor: Pixel rescale factor (``1/255`` for Qwen3-VL).
        patch_size: Vision patch size used to derive the ``<|image_pad|>`` count.
        merge_size: Spatial merge size used to derive the ``<|image_pad|>`` count.
        normalize_state: Whether the exported model expects normalized state.
            Defaults to False (raw state), matching the training default.
        state_mean: Baked ``max_state_dim`` state mean (identity when disabled).
        state_std: Baked ``max_state_dim`` state std (identity when disabled).
    """

    def __init__(
        self,
        camera_views: Sequence[str] = ("base", "wrist_left"),
        max_state_dim: int = 32,
        image_factor: int = 32,
        image_max_pixels: int = 90000,
        image_mean: Sequence[float] = _QWEN3VL_IMAGE_MEAN,
        image_std: Sequence[float] = _QWEN3VL_IMAGE_STD,
        rescale_factor: float = _QWEN3VL_RESCALE_FACTOR,
        patch_size: int = _QWEN3VL_PATCH_SIZE,
        merge_size: int = _QWEN3VL_MERGE_SIZE,
        *,
        normalize_state: bool = False,
        state_mean: Sequence[float] | None = None,
        state_std: Sequence[float] | None = None,
    ) -> None:
        """Initialize the XR0 inference preprocessor.

        Raises:
            ValueError: If ``camera_views`` is empty or ``patch_size`` / ``merge_size``
                is not positive.
        """
        super().__init__()
        camera_views = tuple(camera_views)
        if not camera_views:
            msg = "XR0InferencePreprocessor requires at least one camera view"
            raise ValueError(msg)
        if int(patch_size) <= 0 or int(merge_size) <= 0:
            msg = f"patch_size and merge_size must be positive, got {patch_size!r} / {merge_size!r}"
            raise ValueError(msg)

        self._camera_views = camera_views
        self._max_state_dim = int(max_state_dim)
        self._image_factor = int(image_factor)
        self._image_max_pixels = int(image_max_pixels)
        self._image_mean = tuple(float(v) for v in image_mean)
        self._image_std = tuple(float(v) for v in image_std)
        self._rescale_factor = float(rescale_factor)
        self._patch_size = int(patch_size)
        self._merge_size = int(merge_size)
        self._normalize_state = bool(normalize_state)

        # State normalization is opt-in; padded dims use identity stats (mean 0,
        # std 1) so they stay zero, mirroring the training XR0Preprocessor.
        if normalize_state and state_mean is not None and state_std is not None:
            self._state_mean = self._pad_state_stat(state_mean, 0.0)
            self._state_std = self._pad_state_stat(state_std, 1.0)
        else:
            self._state_mean = np.zeros(self._max_state_dim, dtype=np.float32)
            self._state_std = np.ones(self._max_state_dim, dtype=np.float32)

    def _pad_state_stat(self, values: Sequence[float], fill: float) -> np.ndarray:
        """Pad/truncate a state stat to ``max_state_dim`` (padded dims use ``fill``).

        Returns:
            The ``(max_state_dim,)`` float32 stat array.
        """
        arr = np.asarray(values, dtype=np.float32).flatten()
        out = np.full(self._max_state_dim, fill, dtype=np.float32)
        dim = min(self._max_state_dim, arr.shape[0])
        out[:dim] = arr[:dim]
        return out

    def _extract_images(self, inputs: dict[str, object]) -> list[Image.Image]:
        """Return the resized PIL views in ``camera_views`` (sorted-key) order.

        Returns:
            The list of resized PIL images (one per available camera view).

        Raises:
            ValueError: If the observation contains no image entry.
        """
        images_value = inputs.get(IMAGES)
        if isinstance(images_value, dict):
            image_items = {f"{IMAGES}.{view}": array for view, array in images_value.items()}
        else:
            image_items = {
                key: value
                for key, value in inputs.items()
                if isinstance(key, str) and key.startswith(f"{IMAGES}.") and "is_pad" not in key
            }
        keys = sorted(image_items)[: len(self._camera_views)]
        if not keys:
            msg = "XR0 inference requires at least one image observation"
            raise ValueError(msg)
        return [
            resize_image(_to_pil(image_items[key]), factor=self._image_factor, max_pixels=self._image_max_pixels)
            for key in keys
        ]

    def _prepare_state(self, inputs: dict[str, object]) -> np.ndarray:
        """Pad the state into ``(B, 1, max_state_dim)`` (optionally normalized).

        Returns:
            The padded ``(B, 1, max_state_dim)`` float32 state array.

        Raises:
            ValueError: If the observation has no state entry.
        """
        state_value = inputs.get(STATE)
        if state_value is None:
            msg = "XR0 inference requires a 'state' observation"
            raise ValueError(msg)
        state = np.asarray(state_value, dtype=np.float32)
        if state.ndim == 1:  # (D,) -> (1, D)
            state = state[None, :]
        if state.ndim == _TEMPORAL_STATE_NDIM:  # (B, T, D) -> last frame
            state = state[:, -1, :]
        dim = state.shape[-1]
        if dim < self._max_state_dim:
            state = np.pad(state, ((0, 0), (0, self._max_state_dim - dim)))
        state = state[:, : self._max_state_dim]
        if self._normalize_state:
            state = (state - self._state_mean) / (self._state_std + ACTION_EPS)
        return state[:, None, :].astype(np.float32)  # (B, 1, max_state_dim)

    @staticmethod
    def _instruction(inputs: dict[str, object]) -> str:
        """Extract the task instruction string from the observation.

        Returns:
            The (first) task instruction as a string (empty when absent).
        """
        task = inputs.get(TASK)
        if task is None:
            return ""
        if isinstance(task, str):
            return task
        if isinstance(task, np.ndarray):
            flat = np.atleast_1d(task).tolist()
            return str(flat[0]) if flat else ""
        if isinstance(task, (list, tuple)):
            return str(task[0]) if task else ""
        return str(task)

    def __call__(self, inputs: dict[str, object]) -> dict[str, object]:
        """Transform a raw observation into the exported graph inputs.

        Args:
            inputs: Observation dict with a ``state`` array, ``images`` (nested
                dict or flattened ``images.*`` keys) and a ``task`` string.

        Returns:
            Dict with ``pixel_values`` / ``state`` (float32 NumPy) and ``task``
            (a single-element list holding the rendered chat prompt string).
            ``pixel_values`` is the pre-patchify normalized image grid
            ``(num_images, C, H, W)`` -- the exported graph bakes the Qwen3-VL
            temporal-duplication + patchify reshape/transpose; the sibling
            OpenVINO tokenizer turns ``task`` into the graph's ``tokenized_prompt``
            / ``tokenized_prompt_mask`` inputs.
        """
        images = self._extract_images(inputs)
        pixel_values = build_pixel_grid(images, self._image_mean, self._image_std, self._rescale_factor)
        pad_counts = [
            image_pad_count(
                1,
                image.size[1] // self._patch_size,  # image.size == (width, height)
                image.size[0] // self._patch_size,
                self._merge_size,
            )
            for image in images
        ]
        views = self._camera_views[: len(images)]
        prompt = render_chat_prompt(views, pad_counts, self._instruction(inputs))
        state = self._prepare_state(inputs)
        return {
            "pixel_values": np.ascontiguousarray(pixel_values.astype(np.float32)),
            "state": np.ascontiguousarray(state),
            TASK: [prompt],
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
