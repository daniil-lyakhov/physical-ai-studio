# Copyright (C) 2026 Xiaomi Corporation.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor / postprocessor for the XR0 model.

* **Prompt & vision** -- a Qwen3-VL multi-view chat prompt is assembled (one
  ``<|vision_start|><|image_pad|><|vision_end|>`` block per configured camera
  view) and tokenized with the stock ``Qwen3VLProcessor`` (via
  ``AutoProcessor``) Images are resized with
  :func:`~physicalai.policies.xr0.io.resize_image` and passed
  to the processor with ``do_resize=False``.
* **State** -- padded into the 32-dim bimanual layout and shaped ``(B, 1, D)``,
  matching the source ``state.view(1, 1, -1)``.
* **Action** -- normalized with the source ``normalize_action`` mean/std
  convention (:func:`~physicalai.policies.xr0.io.normalize_action`), padded to
  ``max_action_dim``, with a validity ``action_mask``.

The postprocessor inverts the action normalization
(:func:`~physicalai.policies.xr0.io.denormalize_action`).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image

from physicalai.data import Feature, FeatureType
from physicalai.data.observation import ACTION, IMAGES, STATE, TASK, Observation

from .io import ACTION_EPS, resize_image

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

_MULTI_VIEW_HEADER = "The following observations are captured from multiple views.\n"
_TASK_TEMPLATE = "Generate robot actions for the task:\n{instruction} /no_cot"
_ASSISTANT_PRIMER = "<cot></cot>"
_TEMPORAL_STATE_NDIM = 3
_TEMPORAL_IMAGE_NDIM = 5

# Pinned Qwen3-VL processor revision for reproducible and secure downloads
# (commit on the "Qwen/Qwen3-VL-4B-Instruct" repo).
_QWEN3_VL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"

# View titles the model was trained with (Xiaomi reference server prompt in
# deploy/server.py), e.g. "wrist_left" -> "Left-Wrist" so the prompt reads
# "# Left-Wrist View". A plain capitalize would wrongly yield "Wrist Left".
_VIEW_TITLES = {
    "base": "Base",
    "wrist_left": "Left-Wrist",
    "wrist_right": "Right-Wrist",
}


def _view_title(view: str) -> str:
    """Human-readable view title matching the reference prompt.

    Known views use the reference eval's exact titles (e.g. ``"wrist_left"`` ->
    ``"Left-Wrist"``); unknown views fall back to a capitalized join.

    Returns:
        The human-readable view title.
    """
    key = view.replace("-", "_")
    if key in _VIEW_TITLES:
        return _VIEW_TITLES[key]
    return " ".join(word.capitalize() for word in key.split("_"))


def _to_pil(image: torch.Tensor) -> Image.Image:
    """Convert a single ``(C, H, W)`` or ``(H, W, C)`` image tensor to a PIL image.

    Returns:
        The image as a PIL ``Image``.
    """
    array = image.detach().cpu()
    if array.ndim == _TEMPORAL_STATE_NDIM and array.shape[0] in {1, 3}:  # channels-first
        array = array.permute(1, 2, 0)
    np_img = array.numpy()
    if np_img.dtype != np.uint8:
        np_img = np.clip(np_img, 0.0, 1.0) * 255.0
        np_img = np_img.round().astype(np.uint8)
    if np_img.shape[-1] == 1:
        np_img = np.repeat(np_img, 3, axis=-1)
    return Image.fromarray(np_img)


class XR0Preprocessor(torch.nn.Module):
    """Transform framework observations into the XR0 model batch.

    Produces the keys consumed by ``XR0Model`` (``input_ids``,
    ``attention_mask``, ``pixel_values``, ``image_grid_thw``, ``state`` and --
    during training -- ``action`` / ``action_mask``).

    Args:
        camera_views: Ordered view names embedded into the prompt. The framework
            ``IMAGES.*`` keys are mapped to these positionally (sorted).
        max_state_dim: State dimension after padding.
        max_action_dim: Action dimension after padding.
        features: Optional feature map (from dataset stats) used to normalize the
            action with the source mean/std convention. When ``None`` the action
            is passed through unnormalized.
        image_factor: Patch-alignment factor for :func:`resize_image`.
        image_max_pixels: Maximum image area for :func:`resize_image`.
        processor_name: HuggingFace id of the Qwen3-VL processor.
    """

    action_mean: torch.Tensor
    action_std: torch.Tensor

    def __init__(
        self,
        camera_views: Sequence[str] = ("base", "wrist_left"),
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        features: dict[str, Feature] | None = None,
        image_factor: int = 32,
        image_max_pixels: int = 90000,
        processor_name: str = "Qwen/Qwen3-VL-4B-Instruct",
    ) -> None:
        """Initialize the XR0 preprocessor."""
        super().__init__()
        self.camera_views = tuple(camera_views)
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.image_factor = image_factor
        self.image_max_pixels = image_max_pixels
        self.processor_name = processor_name
        self._processor: Any = None

        mean, std = self._action_stats(features)
        self.register_buffer("action_mean", mean, persistent=False)
        self.register_buffer("action_std", std, persistent=False)

    def _action_stats(self, features: dict[str, Feature] | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Build padded ``(max_action_dim,)`` mean/std buffers from action features.

        Returns:
            A ``(mean, std)`` tuple of ``(max_action_dim,)`` buffers.
        """
        mean = torch.zeros(self.max_action_dim)
        std = torch.ones(self.max_action_dim)
        if features is None:
            return mean, std
        for feature in features.values():
            if feature.ftype != FeatureType.ACTION or feature.normalization_data is None:
                continue
            norm = feature.normalization_data
            if norm.mean is None or norm.std is None:
                continue
            feat_mean = torch.as_tensor(norm.mean, dtype=torch.float32).flatten()
            feat_std = torch.as_tensor(norm.std, dtype=torch.float32).flatten()
            dim = min(self.max_action_dim, feat_mean.numel())
            mean[:dim] = feat_mean[:dim]
            std[:dim] = feat_std[:dim]
            break
        return mean, std

    @property
    def processor(self) -> Any:  # noqa: ANN401
        """Lazy-load the Qwen3-VL processor.

        Raises:
            ImportError: If transformers is not installed.
        """
        if self._processor is None:
            try:
                from transformers import AutoProcessor  # noqa: PLC0415
            except ImportError as exc:
                msg = "XR0 preprocessing requires transformers. Install with: uv pip install transformers"
                raise ImportError(msg) from exc
            # Revision pinned for reproducibility and security
            self._processor = AutoProcessor.from_pretrained(self.processor_name, revision=_QWEN3_VL_REVISION)
            self._processor.tokenizer.padding_side = "right"
        return self._processor

    def _build_message(self, instruction: str, images: list[Image.Image]) -> list[dict[str, Any]]:
        """Assemble the Qwen3-VL multi-view chat message for one sample.

        Returns:
            The Qwen3-VL chat message (user + assistant primer) for one sample.
        """
        content: list[dict[str, Any]] = [{"type": "text", "text": _MULTI_VIEW_HEADER}]
        for view, image in zip(self.camera_views, images, strict=False):
            content.extend((
                {"type": "text", "text": f"# {_view_title(view)} View\n"},
                {"type": "image", "image": image},
                {"type": "text", "text": "\n"},
            ))
        content.append({"type": "text", "text": _TASK_TEMPLATE.format(instruction=instruction)})
        return [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": _ASSISTANT_PRIMER}]},
        ]

    def _extract_view_images(self, batch: dict[str, Any]) -> list[list[Image.Image]]:
        """Return, per sample, the list of resized PIL images for each camera view.

        Returns:
            Per sample, the list of resized PIL images (one per camera view).

        Raises:
            ValueError: If the batch contains no image observation.
        """
        image_keys = [key for key in Observation.get_flattened_keys(batch, IMAGES) if "is_pad" not in key]
        image_keys = sorted(image_keys)[: len(self.camera_views)]
        if not image_keys:
            msg = "XR0Preprocessor requires at least one image observation"
            raise ValueError(msg)

        per_view: list[torch.Tensor] = []
        for key in image_keys:
            tensor = batch[key]
            if tensor.ndim == _TEMPORAL_IMAGE_NDIM:  # (B, T, C, H, W) -> last frame
                tensor = tensor[:, -1]
            per_view.append(tensor)

        batch_size = per_view[0].shape[0]
        images: list[list[Image.Image]] = []
        for sample in range(batch_size):
            sample_images = [
                resize_image(_to_pil(view[sample]), factor=self.image_factor, max_pixels=self.image_max_pixels)
                for view in per_view
            ]
            images.append(sample_images)
        return images

    def _prepare_state(self, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
        """Pad the state into ``(B, 1, max_state_dim)`` (source ``state.view(1, 1, -1)``).

        Returns:
            The padded state tensor of shape ``(B, 1, max_state_dim)``.
        """
        state = batch[STATE]
        if state.ndim == _TEMPORAL_STATE_NDIM:  # (B, T, D) -> last frame
            state = state[:, -1, :]
        state = state.to(torch.float32)
        state = F.pad(state, (0, max(0, self.max_state_dim - state.shape[-1])))[:, : self.max_state_dim]
        return state.unsqueeze(1).to(device)

    def _prepare_action(self, action: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize (source convention) + pad the action, and build its validity mask.

        Returns:
            A ``(action, mask)`` tuple of padded action and its validity mask.
        """
        action = action.to(torch.float32)
        real_dim = min(action.shape[-1], self.max_action_dim)
        action = F.pad(action, (0, max(0, self.max_action_dim - action.shape[-1])))[..., : self.max_action_dim]
        # Mirror io.normalize_action: (action - mean) / (std + eps).
        action = (action - self.action_mean) / (self.action_std + ACTION_EPS)

        mask = torch.zeros_like(action, dtype=torch.int32)
        mask[..., :real_dim] = 1
        return action.to(device), mask.to(device)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Process a batch into the XR0 model input.

        Args:
            batch: Dict with STATE, TASK, image keys and optionally ACTION.

        Returns:
            Dict with ``input_ids`` / ``attention_mask`` / ``pixel_values`` /
            ``image_grid_thw`` / ``state`` and optionally ``action`` /
            ``action_mask``.
        """
        batch = dict(batch)
        device = batch[STATE].device

        images = self._extract_view_images(batch)
        batch_size = len(images)

        task = batch.get(TASK)
        if task is None:
            task = [""] * batch_size
        elif isinstance(task, str):
            task = [task]

        messages = [self._build_message(str(task[i]).strip(), images[i]) for i in range(batch_size)]
        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True, "images_kwargs": {"do_resize": False}},
        )

        out: dict[str, torch.Tensor] = {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
            "pixel_values": encoded["pixel_values"].to(device),
            "image_grid_thw": encoded["image_grid_thw"].to(device),
            "state": self._prepare_state(batch, device),
        }

        if ACTION in batch and batch[ACTION] is not None:
            action, action_mask = self._prepare_action(batch[ACTION], device)
            out[ACTION] = action
            out["action_mask"] = action_mask

        return out


class XR0Postprocessor(torch.nn.Module):
    """Invert the XR0 action normalization.

    Denormalizes predicted actions with the source
    :func:`~physicalai.policies.xr0.io.denormalize_action` convention and slices
    back to the original (unpadded) action dimension when known.

    Args:
        max_action_dim: Padded action dimension used by the preprocessor.
        features: Optional feature map used to recover the action mean/std and
            the original action dimension.
    """

    action_mean: torch.Tensor
    action_std: torch.Tensor

    def __init__(
        self,
        max_action_dim: int = 32,
        features: dict[str, Feature] | None = None,
    ) -> None:
        """Initialize the XR0 postprocessor."""
        super().__init__()
        self.max_action_dim = max_action_dim
        self.action_dim: int | None = None

        mean = torch.zeros(max_action_dim)
        std = torch.ones(max_action_dim)
        if features is not None:
            for feature in features.values():
                if feature.ftype != FeatureType.ACTION or feature.normalization_data is None:
                    continue
                norm = feature.normalization_data
                if norm.mean is None or norm.std is None:
                    continue
                feat_mean = torch.as_tensor(norm.mean, dtype=torch.float32).flatten()
                feat_std = torch.as_tensor(norm.std, dtype=torch.float32).flatten()
                dim = min(max_action_dim, feat_mean.numel())
                mean[:dim] = feat_mean[:dim]
                std[:dim] = feat_std[:dim]
                self.action_dim = int(feat_mean.numel())
                break

        self.register_buffer("action_mean", mean, persistent=False)
        self.register_buffer("action_std", std, persistent=False)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Denormalize and unpad the predicted actions.

        Returns:
            Batch dict with the denormalized action.
        """
        batch = dict(batch)
        if ACTION in batch and batch[ACTION] is not None:
            action = batch[ACTION].to(torch.float32)
            mean = self.action_mean.to(action.device)
            std = self.action_std.to(action.device)
            # Mirror io.denormalize_action: action * (std + eps) + mean.
            action = action * (std + ACTION_EPS) + mean
            if self.action_dim is not None:
                action = action[..., : self.action_dim]
            batch[ACTION] = action
        return batch


def make_xr0_preprocessors(
    camera_views: Sequence[str] = ("base", "wrist_left"),
    max_state_dim: int = 32,
    max_action_dim: int = 32,
    stats: dict[str, dict[str, Any]] | None = None,
    *,
    image_factor: int = 32,
    image_max_pixels: int = 90000,
    processor_name: str = "Qwen/Qwen3-VL-4B-Instruct",
) -> tuple[XR0Preprocessor, XR0Postprocessor]:
    """Create the XR0 preprocessor / postprocessor pair from dataset stats.

    Args:
        camera_views: Ordered camera view names for the prompt.
        max_state_dim: Padded state dimension.
        max_action_dim: Padded action dimension.
        stats: Dataset statistics as nested dicts (LeRobot format).
        image_factor: Patch-alignment factor for image resizing.
        image_max_pixels: Maximum image area for image resizing.
        processor_name: HuggingFace id of the Qwen3-VL processor.

    Returns:
        Tuple of (preprocessor, postprocessor).
    """
    from physicalai.data import NormalizationParameters  # noqa: PLC0415

    features: dict[str, Feature] = {}
    if stats is not None:
        for key, stat in stats.items():
            if ACTION in key:
                feature_type = FeatureType.ACTION
            elif STATE in key:
                feature_type = FeatureType.STATE
            else:
                continue
            raw_name = str(stat.get("name", key))
            mapped_name = raw_name.rsplit("observation.", maxsplit=1)[-1] if "observation." in raw_name else raw_name
            features[mapped_name] = Feature(
                name=mapped_name,
                ftype=feature_type,
                shape=tuple(stat["shape"]),
                normalization_data=NormalizationParameters(
                    mean=stat.get("mean"),
                    std=stat.get("std"),
                    q01=stat.get("q01"),
                    q99=stat.get("q99"),
                ),
            )

    preprocessor = XR0Preprocessor(
        camera_views=camera_views,
        max_state_dim=max_state_dim,
        max_action_dim=max_action_dim,
        features=features,
        image_factor=image_factor,
        image_max_pixels=image_max_pixels,
        processor_name=processor_name,
    )
    postprocessor = XR0Postprocessor(max_action_dim=max_action_dim, features=features)
    return preprocessor, postprocessor
