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

from .io import ACTION_EPS, build_pixel_grid, resize_image
from .prompt import _ASSISTANT_PRIMER, _MULTI_VIEW_HEADER, _TASK_TEMPLATE
from .prompt import view_title as _view_title

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

_TEMPORAL_STATE_NDIM = 3
_TEMPORAL_IMAGE_NDIM = 5
_BATCHED_ACTION_NDIM = 2

# Pinned commit SHA for the default Qwen3-VL processor download. A concrete
# revision keeps the fetched tokenizer/processor reproducible and avoids the
# supply-chain risk of resolving to a moving HEAD (see library security rule 9).
_PROCESSOR_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


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
        max_token_len: Fixed prompt length the OpenVINO tokenizer pads to at export
            (matches the graph's baked ``tokenizer_max_length``).
        normalize_state: When True, normalize the state with per-dimension
            mean/std (from ``features`` or explicit ``state_mean`` / ``state_std``).
            Defaults to False (raw state, matching the upstream recipe).
        state_mean: Optional explicit ``max_state_dim`` state mean overriding the
            feature-derived value (used to reload the exported normalization).
        state_std: Optional explicit ``max_state_dim`` state std overriding the
            feature-derived value (used to reload the exported normalization).
    """

    action_mean: torch.Tensor
    action_std: torch.Tensor
    state_mean: torch.Tensor
    state_std: torch.Tensor

    def __init__(
        self,
        camera_views: Sequence[str] = ("base", "wrist_left"),
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        features: dict[str, Feature] | None = None,
        image_factor: int = 32,
        image_max_pixels: int = 90000,
        processor_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        max_token_len: int = 256,
        *,
        normalize_state: bool = False,
        state_mean: Sequence[float] | None = None,
        state_std: Sequence[float] | None = None,
        action_mode: str = "absolute",
        action_mean: Sequence[float] | torch.Tensor | None = None,
        action_std: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        """Initialize the XR0 preprocessor."""
        super().__init__()
        self.camera_views = tuple(camera_views)
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.image_factor = image_factor
        self.image_max_pixels = image_max_pixels
        self.processor_name = processor_name
        self.max_token_len = int(max_token_len)
        self.normalize_state = bool(normalize_state)
        self.action_mode = str(action_mode)
        self._processor: Any = None

        # Explicit ``action_mean`` / ``action_std`` (e.g. per-timestep delta
        # stats for ``action_mode="delta"``) take precedence over the
        # feature-derived absolute-action stats. They may be 1D ``(D,)`` or 2D
        # ``(T, D)`` and broadcast over the action chunk.
        if action_mean is not None and action_std is not None:
            mean = torch.as_tensor(action_mean, dtype=torch.float32)
            std = torch.as_tensor(action_std, dtype=torch.float32)
        else:
            mean, std = self._action_stats(features)
        self.register_buffer("action_mean", mean, persistent=False)
        self.register_buffer("action_std", std, persistent=False)

        # State normalization is opt-in and identity by default so raw-state
        # checkpoints (e.g. the upstream LIBERO / Pretrain releases) are
        # unaffected. Explicit ``state_mean`` / ``state_std`` (baked into the
        # exported manifest) take precedence over feature-derived stats so the
        # exported graph reproduces the training normalization exactly.
        if state_mean is not None and state_std is not None:
            s_mean = torch.as_tensor(state_mean, dtype=torch.float32).flatten()
            s_std = torch.as_tensor(state_std, dtype=torch.float32).flatten()
        else:
            s_mean, s_std = self._state_stats(features)
        self.register_buffer("state_mean", s_mean, persistent=False)
        self.register_buffer("state_std", s_std, persistent=False)

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

    def _state_stats(self, features: dict[str, Feature] | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Build padded ``(max_state_dim,)`` mean/std buffers from state features.

        Returns identity buffers (mean 0, std 1) when state normalization is
        disabled or no state stats are available, so ``_prepare_state`` is a
        no-op and raw-state checkpoints stay bit-for-bit unchanged.

        Returns:
            A ``(mean, std)`` tuple of ``(max_state_dim,)`` buffers.
        """
        mean = torch.zeros(self.max_state_dim)
        std = torch.ones(self.max_state_dim)
        if features is None or not self.normalize_state:
            return mean, std
        for feature in features.values():
            if feature.ftype != FeatureType.STATE or feature.normalization_data is None:
                continue
            norm = feature.normalization_data
            if norm.mean is None or norm.std is None:
                continue
            feat_mean = torch.as_tensor(norm.mean, dtype=torch.float32).flatten()
            feat_std = torch.as_tensor(norm.std, dtype=torch.float32).flatten()
            dim = min(self.max_state_dim, feat_mean.numel())
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
            self._processor = AutoProcessor.from_pretrained(self.processor_name, revision=_PROCESSOR_REVISION)
            self._processor.tokenizer.padding_side = "right"
        return self._processor

    @property
    def tokenizer(self) -> Any:  # noqa: ANN401
        """Return the Qwen3-VL tokenizer (used for the OpenVINO tokenizer export).

        Returns:
            The processor's underlying HuggingFace tokenizer.
        """
        return self.processor.tokenizer

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

    def image_grid(self, batch: dict[str, Any]) -> np.ndarray:
        """Build the pre-patchify normalized image grid the exported graph consumes.

        Reproduces the Qwen3-VL image path (resize + rescale + normalize) in NumPy,
        omitting the patchify -- the exported OpenVINO graph bakes the temporal
        duplication + patchify reshape/transpose. Used by the inference
        preprocessor and the export input sample so the native pipeline builds the
        grid directly instead of patchifying (via the processor) and un-patchifying.

        Returns:
            The ``(num_images, C, H, W)`` float32 grid, views concatenated
            sample-major to match the processor's ``pixel_values`` ordering.
        """
        images = self._extract_view_images(batch)
        image_processor = self.processor.image_processor
        flat_images = [image for sample in images for image in sample]
        return build_pixel_grid(
            flat_images,
            image_processor.image_mean,
            image_processor.image_std,
            image_processor.rescale_factor,
        )

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
        if self.normalize_state:
            # (state - mean) / (std + eps); padded dims use identity stats so
            # they stay zero. Mirrors the action normalization convention.
            mean = self.state_mean.to(state.device)
            std = self.state_std.to(state.device)
            state = (state - mean) / (std + ACTION_EPS)
        return state.unsqueeze(1).to(device)

    def _prepare_action(
        self,
        action: torch.Tensor,
        device: torch.device,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize (source convention) + pad the action, and build its validity mask.

        When ``action_mode == "delta"`` the raw current-frame ``state`` is
        subtracted from the action target (on the overlapping leading channels)
        before padding/normalization, so the flow head predicts
        ``action[t] - state`` against the per-timestep delta stats.

        Returns:
            A ``(action, mask)`` tuple of padded action and its validity mask.

        Raises:
            ValueError: If ``action_mode == 'delta'`` but no ``state`` is provided.
        """
        action = action.to(torch.float32).clone()  # Clone tensor to avoid mutating the input action
        if self.action_mode == "delta":
            if state is None:
                msg = "action_mode='delta' requires the current state to form the delta target."
                raise ValueError(msg)
            raw_state = state
            if raw_state.ndim == _TEMPORAL_STATE_NDIM:  # (B, T, D) -> current (last) frame
                raw_state = raw_state[:, -1, :]
            raw_state = raw_state.to(torch.float32)
            overlap = min(action.shape[-1], raw_state.shape[-1])
            current = raw_state[..., :overlap].unsqueeze(1)  # (B, 1, overlap)
            if overlap < action.shape[-1]:
                head = action[..., :overlap] - current
                action = torch.cat([head, action[..., overlap:]], dim=-1)
            else:
                action -= current
        real_dim = min(action.shape[-1], self.max_action_dim)
        action = F.pad(action, (0, max(0, self.max_action_dim - action.shape[-1])))[..., : self.max_action_dim]
        # Mirror io.normalize_action: (action - mean) / (std + eps). In delta mode
        # action_mean/action_std are per-timestep (T, D) and broadcast over batch.
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
            action, action_mask = self._prepare_action(batch[ACTION], device, state=batch[STATE])
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
        *,
        action_mode: str = "absolute",
        action_mean: Sequence[float] | torch.Tensor | None = None,
        action_std: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        """Initialize the XR0 postprocessor."""
        super().__init__()
        self.max_action_dim = max_action_dim
        self.action_dim: int | None = None
        self.action_mode = str(action_mode)

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

        # Explicit stats (per-timestep delta stats for ``action_mode="delta"``)
        # override the feature-derived denormalization mean/std; the unpadded
        # ``action_dim`` is still recovered from ``features`` for the final slice.
        if action_mean is not None and action_std is not None:
            mean = torch.as_tensor(action_mean, dtype=torch.float32)
            std = torch.as_tensor(action_std, dtype=torch.float32)

        self.register_buffer("action_mean", mean, persistent=False)
        self.register_buffer("action_std", std, persistent=False)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Denormalize and unpad the predicted actions.

        In ``action_mode="delta"`` the denormalized prediction is a delta and the
        current-frame ``state`` is re-added (on the overlapping leading channels)
        to recover the absolute action.

        Returns:
            Batch dict with the denormalized action.

        Raises:
            ValueError: If ``action_mode == 'delta'`` but no ``state`` is provided.
        """
        batch = dict(batch)
        if ACTION in batch and batch[ACTION] is not None:
            action = batch[ACTION].to(torch.float32)
            mean = self.action_mean.to(action.device)
            std = self.action_std.to(action.device)
            # Mirror io.denormalize_action: action * (std + eps) + mean.
            action = action * (std + ACTION_EPS) + mean
            if self.action_mode == "delta":
                state = batch.get(STATE)
                if state is None:
                    msg = "action_mode='delta' requires the current state to invert the delta prediction."
                    raise ValueError(
                        msg,
                    )
                current = state.to(torch.float32).to(action.device)
                if current.ndim == _TEMPORAL_STATE_NDIM:  # (B, T, D) -> current (last) frame
                    current = current[:, -1, :]
                overlap = min(action.shape[-1], current.shape[-1])
                current = current[..., :overlap].unsqueeze(1)  # (B, 1, overlap)
                if overlap < action.shape[-1]:
                    head = action[..., :overlap] + current
                    action = torch.cat([head, action[..., overlap:]], dim=-1)
                else:
                    action += current
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
    normalize_state: bool = False,
    action_mode: str = "absolute",
    action_delta_mean: Sequence[float] | torch.Tensor | None = None,
    action_delta_std: Sequence[float] | torch.Tensor | None = None,
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
        normalize_state: When True, normalize the state with the dataset's
            per-dimension mean/std. Defaults to False (raw state).
        action_mode: ``"absolute"`` (default) or ``"delta"``. In delta mode the
            action target/inverse use ``action_delta_mean`` / ``action_delta_std``.
        action_delta_mean: Per-timestep delta-action mean (``(chunk_size,
            max_action_dim)``), used only when ``action_mode="delta"``.
        action_delta_std: Per-timestep delta-action std, same shape as
            ``action_delta_mean``.

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

    override_mean: torch.Tensor | None = None
    override_std: torch.Tensor | None = None
    if action_mode == "delta" and action_delta_mean is not None and action_delta_std is not None:
        override_mean = torch.as_tensor(action_delta_mean, dtype=torch.float32)
        override_std = torch.as_tensor(action_delta_std, dtype=torch.float32)

    preprocessor = XR0Preprocessor(
        camera_views=camera_views,
        max_state_dim=max_state_dim,
        max_action_dim=max_action_dim,
        features=features,
        image_factor=image_factor,
        image_max_pixels=image_max_pixels,
        processor_name=processor_name,
        normalize_state=normalize_state,
        action_mode=action_mode,
        action_mean=override_mean,
        action_std=override_std,
    )
    postprocessor = XR0Postprocessor(
        max_action_dim=max_action_dim,
        features=features,
        action_mode=action_mode,
        action_mean=override_mean,
        action_std=override_std,
    )
    return preprocessor, postprocessor


def _delta_feature_tensor(value: Any) -> torch.Tensor:  # noqa: ANN401
    """Coerce an observation field (tensor or single-entry dict) to a tensor.

    Returns:
        The underlying tensor for the field.

    Raises:
        TypeError: If no tensor can be extracted from ``value``.
    """
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for sub in value.values():
            if isinstance(sub, torch.Tensor):
                return sub
    msg = f"expected a tensor (or dict containing one), got {type(value)!r}"
    raise TypeError(msg)


def compute_delta_action_stats(
    datamodule: Any,  # noqa: ANN401
    *,
    chunk_size: int,
    action_dim: int,
    max_action_dim: int = 32,
    max_batches: int | None = None,
    setup_stage: str | None = "fit",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-timestep delta-action mean/std over a training dataset.

    Iterates the datamodule's train dataloader once and, for every sample, forms
    the delta action ``action[t] - state`` (the current-frame state broadcast over
    the chunk, matching the XR0 delta target). Per-chunk-position mean and std are
    accumulated in float64 and returned as ``(chunk_size, max_action_dim)`` buffers,
    padded with identity stats (mean ``0``, std ``1``) on the unused columns so the
    downstream normalization is a no-op there.

    Args:
        datamodule: A Lightning datamodule exposing ``train_dataloader()`` (and an
            optional ``setup`` method) yielding batched observations with ``state``
            and ``action`` fields.
        chunk_size: Number of action steps per chunk (the temporal dimension).
        action_dim: True (unpadded) action dimension used to compute deltas.
        max_action_dim: Padded action dimension of the returned buffers.
        max_batches: Optional cap on the number of batches to consume (for a quick
            estimate). ``None`` consumes the full train set.
        setup_stage: Stage passed to ``datamodule.setup(...)`` before iterating.
            Pass ``None`` to skip setup (e.g. when already set up).

    Returns:
        A ``(mean, std)`` tuple of ``(chunk_size, max_action_dim)`` float32 tensors.

    Raises:
        ValueError: If the dataset yields no samples, or an action chunk whose
            temporal dimension does not match ``chunk_size``.
    """
    if setup_stage is not None and hasattr(datamodule, "setup"):
        datamodule.setup(setup_stage)
    loader = datamodule.train_dataloader()

    sum_1 = torch.zeros(chunk_size, action_dim, dtype=torch.float64)
    sum_2 = torch.zeros(chunk_size, action_dim, dtype=torch.float64)
    count = 0

    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break

        state_field = batch.state if hasattr(batch, "state") else batch[STATE]
        action_field = batch.action if hasattr(batch, "action") else batch[ACTION]
        state = _delta_feature_tensor(state_field).to(torch.float64)
        action = _delta_feature_tensor(action_field).to(torch.float64)

        if state.ndim == _TEMPORAL_STATE_NDIM:  # (B, T, D) -> current (last) frame
            state = state[:, -1, :]
        state = state[..., :action_dim]

        if action.ndim == _BATCHED_ACTION_NDIM:  # (B, D) -> (B, 1, D)
            action = action.unsqueeze(1)
        if action.shape[1] != chunk_size:
            msg = f"action chunk has temporal dim {action.shape[1]}, expected chunk_size={chunk_size}"
            raise ValueError(msg)
        action = action[..., :action_dim]

        delta = action - state.unsqueeze(1)  # (B, T, action_dim)
        sum_1 += delta.sum(dim=0)
        sum_2 += (delta * delta).sum(dim=0)
        count += delta.shape[0]

    if count == 0:
        msg = "no samples found while computing delta action stats"
        raise ValueError(msg)

    mean = sum_1 / count
    var = (sum_2 / count - mean * mean).clamp_min(0.0)
    std = var.sqrt()

    mean_full = torch.zeros(chunk_size, max_action_dim, dtype=torch.float32)
    std_full = torch.ones(chunk_size, max_action_dim, dtype=torch.float32)
    mean_full[:, :action_dim] = mean.to(torch.float32)
    std_full[:, :action_dim] = std.to(torch.float32)
    return mean_full, std_full
