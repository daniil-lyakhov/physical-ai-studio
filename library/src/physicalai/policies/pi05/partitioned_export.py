# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Partitioned export for Pi0.5 model.

Splits Pi05Model into three independently exportable sub-modules:
1. PaliGemmaEncoder — prefix encoding (vision + language) → KV cache
2. GemmaExpertDecoder — denoising step (suffix embedding + expert forward)
3. ActionOutputHead — action projection

This enables per-submodule quantization with different strategies.
"""

from __future__ import annotations

from physicalai.export.mixin_policy import _postprocess_openvino_model

import logging
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers.cache_utils import DynamicCache

from .model import (
    OPENPI_ATTENTION_MASK_VALUE,
    Pi05Model,
    _clone_kv_cache,
    _create_sinusoidal_pos_embedding,
    _make_att_2d_masks,
)

logger = logging.getLogger(__name__)


def _dynamic_cache_to_tensors(past_key_values: DynamicCache) -> tuple[Tensor, Tensor]:
    """Convert DynamicCache to stacked key/value tensors for export.

    Returns:
        Tuple of (all_keys, all_values) each with shape
        (num_layers, batch, num_heads, seq_len, head_dim).
    """
    keys = []
    values = []
    for layer in past_key_values.layers:
        if layer.keys is None or layer.values is None:
            continue
        keys.append(layer.keys)
        values.append(layer.values)
    return torch.stack(keys, dim=0), torch.stack(values, dim=0)


def _tensors_to_dynamic_cache(all_keys: Tensor, all_values: Tensor) -> DynamicCache:
    """Convert stacked key/value tensors back to DynamicCache.

    Args:
        all_keys: (num_layers, batch, num_heads, seq_len, head_dim)
        all_values: (num_layers, batch, num_heads, seq_len, head_dim)

    Returns:
        DynamicCache populated with the given key/value states.
    """
    cache = DynamicCache()
    num_layers = all_keys.shape[0]
    for layer_idx in range(num_layers):
        cache.update(all_keys[layer_idx], all_values[layer_idx], layer_idx)
    return cache


class PaliGemmaEncoder(nn.Module):
    """Wrapper for the PaliGemma prefix encoding stage.

    Takes a single image, image mask, tokens, and token masks, and returns
    the KV cache (as flat tensors) and prefix padding masks.

    Note: For simplicity this wrapper supports a single camera input.
    Multi-camera support can be added by extending the forward signature.
    """

    def __init__(self, pi05_model: Pi05Model) -> None:
        """Initialize from a Pi05Model instance."""
        super().__init__()
        self.paligemma_with_expert = pi05_model.paligemma_with_expert
        self._chunk_size = pi05_model._chunk_size
        self._max_action_dim = pi05_model._max_action_dim
        self._min_period = pi05_model._min_period
        self._max_period = pi05_model._max_period
        # Store reference to parent model for embed_prefix
        self._pi05_model = pi05_model

    def forward(
        self,
        image: Tensor,
        img_mask: Tensor,
        tokens: Tensor,
        masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode prefix (images + language tokens) and produce KV cache.

        Args:
            image: Single image tensor (batch, C, H, W).
            img_mask: Image mask (batch,) boolean.
            tokens: Tokenized prompt (batch, seq_len).
            masks: Token masks (batch, seq_len).

        Returns:
            Tuple of (cache_keys, cache_values, prefix_pad_masks).
            cache_keys/values shape: (num_layers, batch, num_heads, seq_len, head_dim)
        """
        # Wrap single image/mask as lists for embed_prefix
        images = [image]
        img_masks = [img_mask]

        prefix_embs, prefix_pad_masks, prefix_att_masks = self._pi05_model.embed_prefix(
            images, img_masks, tokens, masks
        )
        prefix_att_2d_masks = _make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        cache_keys, cache_values = _dynamic_cache_to_tensors(past_key_values)
        return cache_keys, cache_values, prefix_pad_masks

    def _prepare_attention_masks_4d(self, att_2d_masks: Tensor) -> Tensor:
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)


class GemmaExpertDecoder(nn.Module):
    """Wrapper for the Gemma expert denoising step.

    Takes noisy actions, timestep, KV cache (as tensors), and prefix masks,
    and returns the raw suffix output embeddings (before action_out_proj).
    """

    def __init__(self, pi05_model: Pi05Model) -> None:
        """Initialize from a Pi05Model instance."""
        super().__init__()
        self.paligemma_with_expert = pi05_model.paligemma_with_expert
        self.action_in_proj = pi05_model.action_in_proj
        self.time_mlp_in = pi05_model.time_mlp_in
        self.time_mlp_out = pi05_model.time_mlp_out
        self._chunk_size = pi05_model._chunk_size
        self._min_period = pi05_model._min_period
        self._max_period = pi05_model._max_period

    def forward(
        self,
        x_t: Tensor,
        timestep: Tensor,
        cache_keys: Tensor,
        cache_values: Tensor,
        prefix_pad_masks: Tensor,
    ) -> Tensor:
        """Run one denoising step through the Gemma expert.

        Args:
            x_t: Noisy actions (batch, chunk_size, max_action_dim)
            timestep: Time values (batch,)
            cache_keys: KV cache keys (num_layers, batch, num_heads, seq_len, head_dim)
            cache_values: KV cache values (num_layers, batch, num_heads, seq_len, head_dim)
            prefix_pad_masks: Prefix padding masks (batch, prefix_len)

        Returns:
            suffix_out: Raw expert output (batch, chunk_size, hidden_dim)
        """
        # Reconstruct DynamicCache from tensors
        past_key_values = _tensors_to_dynamic_cache(cache_keys, cache_values)

        # Embed suffix (actions + time)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self._embed_suffix(x_t, timestep)

        # Build attention masks
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = _make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        past_key_values = _clone_kv_cache(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self._chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return suffix_out

    def _embed_suffix(
        self,
        noisy_actions: Tensor,
        timestep: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Embed noisy actions and timestep for the expert."""
        time_emb = _create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self._min_period,
            max_period=self._max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        action_emb = self.action_in_proj(noisy_actions)

        x = self.time_mlp_in(time_emb)
        x = F.silu(x)
        x = self.time_mlp_out(x)
        time_emb = F.silu(x)

        action_time_emb = action_emb
        adarms_cond = time_emb

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)

        att_masks_list = [1] + ([0] * (self._chunk_size - 1))
        att_masks = torch.tensor(att_masks_list, dtype=action_time_emb.dtype, device=action_time_emb.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks_list))

        return action_time_emb, action_time_mask, att_masks, adarms_cond

    def _prepare_attention_masks_4d(self, att_2d_masks: Tensor) -> Tensor:
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)


class ActionOutputHead(nn.Module):
    """Wrapper for the action output projection.

    Takes raw suffix output embeddings and projects to action space.
    """

    def __init__(self, pi05_model: Pi05Model) -> None:
        """Initialize from a Pi05Model instance."""
        super().__init__()
        self.action_out_proj = pi05_model.action_out_proj

    def forward(self, suffix_out: Tensor) -> Tensor:
        """Project suffix output to action space.

        Args:
            suffix_out: (batch, chunk_size, hidden_dim)

        Returns:
            actions: (batch, chunk_size, max_action_dim)
        """
        return self.action_out_proj(suffix_out)


class Pi05PartitionedOVModel(nn.Module):
    """Inference module that runs 3 OpenVINO IR models underneath.

    Replaces the monolithic Pi05Model for inference by running the
    PaliGemmaEncoder, GemmaExpertDecoder, and ActionOutputHead
    as separate OpenVINO compiled models.
    """

    def __init__(
        self,
        paligemma_model_path: str,
        expert_model_path: str,
        head_model_path: str,
        num_inference_steps: int = 10,
        chunk_size: int = 50,
        max_action_dim: int = 32,
    ) -> None:
        """Initialize with paths to the 3 OpenVINO IR models.

        Args:
            paligemma_model_path: Path to paligemma encoder .xml file.
            expert_model_path: Path to gemma expert decoder .xml file.
            head_model_path: Path to action output head .xml file.
            num_inference_steps: Number of denoising steps.
            chunk_size: Action chunk size.
            max_action_dim: Maximum action dimension.
        """
        super().__init__()
        import openvino

        self._num_inference_steps = num_inference_steps
        self._chunk_size = chunk_size
        self._max_action_dim = max_action_dim

        core = openvino.Core()
        self._paligemma_compiled = core.compile_model(
            core.read_model(paligemma_model_path), "CPU"
        )
        self._expert_compiled = core.compile_model(
            core.read_model(expert_model_path), "CPU"
        )
        self._head_compiled = core.compile_model(
            core.read_model(head_model_path), "CPU"
        )

    def forward(self, batch: dict[str, Any]) -> Tensor:
        """Run full inference using the 3 partitioned IR models.

        Args:
            batch: Preprocessed batch dict containing IMAGES, IMAGE_MASKS,
                TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK.

        Returns:
            Denoised action tensor.
        """
        import numpy as np

        from physicalai.data.constants import IMAGE_MASKS, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
        from physicalai.data.observation import IMAGES

        images = batch[IMAGES]
        img_masks = batch[IMAGE_MASKS]
        tokens = batch[TOKENIZED_PROMPT]
        masks = batch[TOKENIZED_PROMPT_MASK]

        # Stage 1: Prefix encoding via PaliGemma IR (single camera)
        paligemma_inputs = {
            "images": images[0].numpy(),
            "image_masks": img_masks[0].numpy(),
            "tokenized_prompt": tokens.numpy(),
            "tokenized_prompt_mask": masks.numpy(),
        }

        paligemma_result = self._paligemma_compiled(paligemma_inputs)
        cache_keys = torch.from_numpy(paligemma_result[0])
        cache_values = torch.from_numpy(paligemma_result[1])
        prefix_pad_masks = torch.from_numpy(paligemma_result[2])

        # Stage 2: Iterative denoising via Expert IR
        bsize = tokens.shape[0]
        x_t = torch.zeros(bsize, self._chunk_size, self._max_action_dim, dtype=torch.float32)
        dt = -1.0 / self._num_inference_steps

        for step in range(self._num_inference_steps):
            time = 1.0 + step * dt
            time_tensor = torch.full((bsize,), time, dtype=torch.float32)

            expert_inputs = {
                "x_t": x_t.numpy(),
                "timestep": time_tensor.numpy(),
                "cache_keys": cache_keys.numpy(),
                "cache_values": cache_values.numpy(),
                "prefix_pad_masks": prefix_pad_masks.numpy(),
            }
            expert_result = self._expert_compiled(expert_inputs)
            suffix_out = torch.from_numpy(expert_result[0])

            # Stage 3: Action projection via Head IR
            head_result = self._head_compiled({"suffix_out": suffix_out.numpy()})
            v_t = torch.from_numpy(head_result[0])

            x_t = x_t + dt * v_t

        return x_t

    def select_action(self, observation: dict[str, Any]) -> "np.ndarray":
        """Select action for given observation (PolicyLike interface).

        Runs the full partitioned inference pipeline and returns
        the action chunk as a numpy array.

        Args:
            observation: Preprocessed observation dict with keys matching
                the batch format (IMAGES, IMAGE_MASKS, TOKENIZED_PROMPT, etc.)

        Returns:
            Action array of shape (batch, chunk_size, max_action_dim).
        """
        import numpy as np

        action_tensor = self.forward(observation)
        return action_tensor.numpy()

    def reset(self) -> None:
        """Reset policy state for a new episode (PolicyLike interface).

        No internal state to reset for this model.
        """


def export_partitioned_openvino(
    pi05_model: Pi05Model,
    output_dir: str,
    compress_to_fp16: bool = True,
    tokenizer: Any | None = None,
) -> dict[str, str]:
    """Export Pi05Model as 3 separate OpenVINO IR models.

    Args:
        pi05_model: The Pi05Model instance (in eval mode).
        output_dir: Directory where the 3 IR models will be saved.
        input_sample: Optional sample input for tracing. If None, uses model's sample_input.
        compress_to_fp16: Whether to compress weights to FP16.
        tokenizer: Optional pre-loaded tokenizer instance. If None, will attempt
            to load from HuggingFace.

    Returns:
        Dict mapping part names to their .xml file paths.
    """
    from pathlib import Path

    import openvino

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pi05_model.eval()
    device = next(pi05_model.parameters()).device

    # Create wrapper modules (they share parameters with pi05_model)
    paligemma_encoder = PaliGemmaEncoder(pi05_model)
    expert_decoder = GemmaExpertDecoder(pi05_model)
    action_head = ActionOutputHead(pi05_model)

    paligemma_encoder.eval()
    expert_decoder.eval()
    action_head.eval()

    # Generate sample inputs for tracing
    batch_size = 1
    chunk_size = pi05_model._chunk_size
    max_action_dim = pi05_model._max_action_dim
    expert_width = pi05_model.action_in_proj.out_features

    # --- Export PaliGemma Encoder ---
    logger.info("Exporting PaliGemma encoder...")
    # Always use the vision tower's expected resolution for tracing inputs.
    # Dataset stats may store the original (pre-resize) image shape, but the
    # SigLIP vision tower requires images at its configured resolution.
    vision_cfg = pi05_model.paligemma_with_expert.paligemma.config.vision_config
    img_size = vision_cfg.image_size
    num_channels = vision_cfg.num_channels
    image_shape = (num_channels, img_size, img_size)
    logger.info("Using vision config image shape for tracing: %s", image_shape)

    # Create sample inputs for encoder tracing (single image)
    sample_image = torch.randn(batch_size, *image_shape, device=device)
    sample_img_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    token_len = 200  # default max tokenizer length
    sample_tokens = torch.zeros(batch_size, token_len, dtype=torch.long, device=device)
    sample_masks = torch.ones(batch_size, token_len, dtype=torch.bool, device=device)

    # Run once to get output shapes for expert tracing
    with torch.no_grad():
        cache_keys, cache_values, prefix_pad_masks = paligemma_encoder(
            sample_image, sample_img_mask, sample_tokens, sample_masks
        )

    # Trace PaliGemma encoder via ONNX
    paligemma_onnx_path = output_path / "paligemma_encoder.onnx"
    torch.onnx.export(
        paligemma_encoder,
        args=(sample_image, sample_img_mask, sample_tokens, sample_masks),
        f=str(paligemma_onnx_path),
        input_names=["images", "image_masks", "tokenized_prompt", "tokenized_prompt_mask"],
        output_names=["cache_keys", "cache_values", "prefix_pad_masks"],
        dynamic_axes={
            "images": {0: "batch"},
            "image_masks": {0: "batch"},
            "tokenized_prompt": {0: "batch"},
            "tokenized_prompt_mask": {0: "batch"},
            "cache_keys": {1: "batch", 3: "seq"},
            "cache_values": {1: "batch", 3: "seq"},
            "prefix_pad_masks": {0: "batch", 1: "seq"},
        },
    )
    ov_paligemma = openvino.convert_model(str(paligemma_onnx_path))
    _postprocess_openvino_model(ov_paligemma, ["cache_keys", "cache_values", "prefix_pad_masks"])
    paligemma_xml = output_path / "paligemma_encoder.xml"
    openvino.save_model(ov_paligemma, str(paligemma_xml), compress_to_fp16=compress_to_fp16)
    paligemma_onnx_path.unlink()  # cleanup intermediate onnx
    logger.info("PaliGemma encoder exported to %s", paligemma_xml)

    # --- Export Gemma Expert Decoder ---
    logger.info("Exporting Gemma expert decoder...")
    sample_x_t = torch.randn(batch_size, chunk_size, max_action_dim, device=device)
    sample_timestep = torch.tensor([0.9], device=device)

    expert_onnx_path = output_path / "gemma_expert_decoder.onnx"
    torch.onnx.export(
        expert_decoder,
        args=(sample_x_t, sample_timestep, cache_keys, cache_values, prefix_pad_masks),
        f=str(expert_onnx_path),
        input_names=["x_t", "timestep", "cache_keys", "cache_values", "prefix_pad_masks"],
        output_names=["suffix_out"],
        dynamic_axes={
            "x_t": {0: "batch"},
            "timestep": {0: "batch"},
            "cache_keys": {1: "batch", 3: "seq"},
            "cache_values": {1: "batch", 3: "seq"},
            "prefix_pad_masks": {0: "batch", 1: "seq"},
            "suffix_out": {0: "batch"},
        },
    )
    ov_expert = openvino.convert_model(str(expert_onnx_path))
    _postprocess_openvino_model(ov_expert, ["suffix_out"])
    expert_xml = output_path / "gemma_expert_decoder.xml"
    openvino.save_model(ov_expert, str(expert_xml), compress_to_fp16=compress_to_fp16)
    expert_onnx_path.unlink()
    logger.info("Gemma expert decoder exported to %s", expert_xml)

    # --- Export Action Output Head ---
    logger.info("Exporting action output head...")
    sample_suffix_out = torch.randn(batch_size, chunk_size, expert_width, device=device)

    head_onnx_path = output_path / "action_output_head.onnx"
    torch.onnx.export(
        action_head,
        args=(sample_suffix_out,),
        f=str(head_onnx_path),
        input_names=["suffix_out"],
        output_names=["action"],
        dynamic_axes={
            "suffix_out": {0: "batch"},
            "action": {0: "batch"},
        },
    )
    ov_head = openvino.convert_model(str(head_onnx_path))
    _postprocess_openvino_model(ov_head, ["action"])
    head_xml = output_path / "action_output_head.xml"
    openvino.save_model(ov_head, str(head_xml), compress_to_fp16=compress_to_fp16)
    head_onnx_path.unlink()
    logger.info("Action output head exported to %s", head_xml)

    # --- Save preprocessor config + tokenizer ---
    _save_preprocessor_config(pi05_model, output_path)
    _save_tokenizer(pi05_model, output_path, tokenizer=tokenizer)
    logger.info("Preprocessor config saved to %s", output_path / "preprocessor_config.json")

    # --- Write manifest.json so InferenceModel.load() works ---
    # Derive actual action dimension from dataset stats
    action_dim = None
    if pi05_model._dataset_stats and "action" in pi05_model._dataset_stats:
        action_dim = pi05_model._dataset_stats["action"]["shape"][0]

    _write_partitioned_manifest(
        output_path,
        chunk_size=chunk_size,
        max_action_dim=max_action_dim,
        num_inference_steps=pi05_model._num_inference_steps,
        action_dim=action_dim,
        dataset_stats=pi05_model._dataset_stats,
        image_resolution=image_shape[1:] if image_shape is not None else (224, 224),
    )
    logger.info("Manifest written to %s", output_path / "manifest.json")

    return {
        "paligemma": str(paligemma_xml),
        "expert": str(expert_xml),
        "head": str(head_xml),
    }


def _save_preprocessor_config(pi05_model: Pi05Model, output_dir: Any) -> None:
    """Save preprocessor configuration for inference-time observation preprocessing.

    Writes ``preprocessor_config.json`` containing dataset stats, tokenizer name,
    image resolution, and other parameters needed by Pi05Preprocessor.

    Args:
        pi05_model: The Pi05Model with dataset stats.
        output_dir: Export directory.
    """
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from physicalai.data.observation import ACTION, STATE  # noqa: PLC0415

    stats = pi05_model._dataset_stats
    config: dict[str, Any] = {
        "tokenizer_name": "google/paligemma-3b-pt-224",
        "max_token_len": 200,
        "image_resolution": list(pi05_model.paligemma_with_expert.paligemma.config.vision_config.image_size
                                  if hasattr(pi05_model.paligemma_with_expert.paligemma.config.vision_config, 'image_size')
                                  and not isinstance(pi05_model.paligemma_with_expert.paligemma.config.vision_config.image_size, int)
                                  else [pi05_model.paligemma_with_expert.paligemma.config.vision_config.image_size] * 2),
        "empty_cameras": 0,
    }

    # Extract state normalization stats
    for key, stat in stats.items():
        if STATE in key:
            config["state_stats"] = {
                "mean": list(stat["mean"]),
                "std": list(stat["std"]),
            }
        elif ACTION in key:
            config["action_stats"] = {
                "mean": list(stat["mean"]),
                "std": list(stat["std"]),
                "shape": list(stat["shape"]),
            }

    config_path = Path(output_dir) / "preprocessor_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def _save_tokenizer(pi05_model: Pi05Model, output_dir: Any, tokenizer: Any | None = None) -> None:
    """Save the PaliGemma tokenizer from the model to the export directory.

    Uses the provided tokenizer if available, otherwise falls back to loading
    from HuggingFace. If loading fails (e.g. gated repo without access),
    the tokenizer is skipped — the inference preprocessor will load it at
    runtime from HuggingFace cache or the local ``tokenizer/`` directory.

    Args:
        pi05_model: The Pi05Model (unused when tokenizer is provided).
        output_dir: Export directory.
        tokenizer: Optional pre-loaded tokenizer instance.
    """
    from pathlib import Path  # noqa: PLC0415

    tokenizer_dir = Path(output_dir) / "tokenizer"

    if tokenizer is None:
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415

            logger.info("No tokenizer provided, attempting to load from HuggingFace cache.")
            tokenizer = AutoTokenizer.from_pretrained(
                "google/paligemma-3b-pt-224",
                revision="35e4f46485b4d07967e7e9935bc3786aad50687c",
                use_fast=True,
                local_files_only=True,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Could not load tokenizer (gated repo or not cached). "
                "Skipping tokenizer save. The inference preprocessor will "
                "need to load it at runtime."
            )
            return

    tokenizer.save_pretrained(str(tokenizer_dir))
    logger.info("Tokenizer saved to %s", tokenizer_dir)


def _stats_to_serializable(stats_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert stats values (tensors/arrays) to JSON-serializable lists.

    Args:
        stats_dict: Dict mapping stat names to tensor/array/list values.

    Returns:
        Dict with all values converted to plain Python lists.
    """
    result: dict[str, Any] = {}
    for key, value in stats_dict.items():
        if hasattr(value, "tolist"):
            result[key] = value.tolist()
        elif isinstance(value, (list, tuple)):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def _write_partitioned_manifest(
    output_dir: Any,
    chunk_size: int,
    max_action_dim: int,
    num_inference_steps: int,
    action_dim: int | None = None,
    dataset_stats: dict[str, Any] | None = None,
    image_resolution: tuple[int, int] = (224, 224),
    normalization_mode: str = "mean_std",
    tokenizer_max_length: int = 200,
) -> None:
    """Write a manifest.json for the partitioned export.

    The manifest includes an ``adapter`` ComponentSpec that tells
    InferenceModel to use PartitionedOpenVINOAdapter instead of the
    default monolithic OpenVINO adapter, and preprocessor/postprocessor
    specs for observation-to-tokenized-input conversion and action
    denormalization.

    Args:
        output_dir: Directory containing the exported IR files.
        chunk_size: Action chunk size.
        max_action_dim: Maximum action dimension.
        num_inference_steps: Number of denoising steps.
        dataset_stats: Dataset statistics dict for normalization.
        image_resolution: Target image resolution (H, W).
        normalization_mode: Normalization mode ('mean_std', 'min_max', etc.).
        tokenizer_max_length: Maximum tokenizer length.
    """
    from physicalai.data.observation import ACTION, STATE  # noqa: PLC0415
    from physicalai.inference.adapters.openvino_partitioned import PartitionedOpenVINOAdapter  # noqa: PLC0415
    from physicalai.inference.manifest import ComponentSpec, Manifest, ModelSpec, PolicySpec  # noqa: PLC0415
    from physicalai.inference.runners.action_chunking import ActionChunking  # noqa: PLC0415
    from physicalai.inference.runners.single_pass import SinglePass  # noqa: PLC0415

    adapter_spec = ComponentSpec.from_class(
        PartitionedOpenVINOAdapter,
        device="CPU",
        num_inference_steps=num_inference_steps,
        chunk_size=chunk_size,
        max_action_dim=max_action_dim,
        action_dim=action_dim,
    )

    runner_spec = ComponentSpec.from_class(
        ActionChunking,
        runner=ComponentSpec.from_class(SinglePass),
        chunk_size=chunk_size,
    )

    # Build preprocessor specs
    preprocessor_specs = [
        ComponentSpec(
            type="pi05",
            image_resolution=list(image_resolution),
            empty_cameras=0,
        ),
    ]

    postprocessor_specs = []

    if dataset_stats is not None:
        # State normalization preprocessor
        state_key = f"observation.{STATE}"
        if state_key in dataset_stats:
            state_stats = _stats_to_serializable(dataset_stats[state_key])
            preprocessor_specs.append(
                ComponentSpec(
                    type="normalize",
                    stats={STATE: state_stats},
                    mode=normalization_mode,
                ),
            )

        # HF tokenizer preprocessor (partitioned uses HF tokenizer, not OV)
        preprocessor_specs.append(
            ComponentSpec(
                type="hf_tokenizer",
                tokenizer_name="google/paligemma-3b-pt-224",
                revision="35e4f46485b4d07967e7e9935bc3786aad50687c",
                max_token_len=tokenizer_max_length,
            ),
        )

        # Action denormalization postprocessor
        if ACTION in dataset_stats:
            action_stats = _stats_to_serializable(dataset_stats[ACTION])
            postprocessor_specs.append(
                ComponentSpec(
                    type="denormalize",
                    stats={ACTION: action_stats},
                    mode=normalization_mode,
                ),
            )

    manifest = Manifest(
        policy=PolicySpec(
            name="pi05",
            source={"class_path": "physicalai.policies.pi05.policy.Pi05"},
        ),
        model=ModelSpec(
            artifacts={"openvino": "paligemma_encoder.xml"},
            adapter=adapter_spec,
            runner=runner_spec,
            preprocessors=preprocessor_specs,
            postprocessors=postprocessor_specs,
        ),
    )
    manifest.save(output_dir / "manifest.json")
