# Copyright (C) 2026 Xiaomi Corporation.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""XR0 Policy - Lightning wrapper for training and inference.

Wraps :class:`~physicalai.policies.xr0.vla.XR0Model` (Qwen3-VL-4B backbone + DiT
action expert) in the framework ``Policy`` contract, following the same
dual-path initialization as Pi0.5:

* **Lazy path**: ``XR0()`` + ``trainer.fit()`` -- the model and preprocessors are
  built in ``setup()`` from the training dataset statistics.
* **Eager path**: ``XR0(dataset_stats=...)`` -- built immediately.

Input/output adaptation (the Qwen3-VL multi-view prompt, ``io`` state/action
helpers, and action normalization) lives in
:mod:`~physicalai.policies.xr0.preprocessor`, mirroring the source repository.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import torch

from physicalai.data.dataset import Dataset
from physicalai.policies.base import Policy
from physicalai.train.schedulers import cosine_decay_with_warmup_scheduler
from physicalai.train.utils import reformat_dataset_to_match_policy

from .config import XR0Config
from .preprocessor import make_xr0_preprocessors
from .vla import XR0Model

if TYPE_CHECKING:
    from physicalai.data import Observation

    from .preprocessor import XR0Postprocessor, XR0Preprocessor

logger = logging.getLogger(__name__)

_DTYPES: dict[str, torch.dtype] = {"bfloat16": torch.bfloat16, "float32": torch.float32}


class XR0(Policy):
    """XR0 Policy - Xiaomi's flow-matching VLA model.

    Lightning wrapper for training and inference with :class:`XR0Model`.

    Args:
        vlm_model_id: HuggingFace id of the Qwen3-VL backbone.
        vlm_attn_implementation: Attention backend for the VLM.
        dtype: Model precision (``"bfloat16"`` or ``"float32"``).
        n_obs_steps: Number of observation steps.
        chunk_size: Number of action steps to predict.
        n_action_steps: Number of action steps to execute.
        max_state_dim: Padded state dimension.
        max_action_dim: Padded action dimension.
        state_len: Number of state tokens.
        dit_num_layers: DiT decoder layers.
        dit_hidden_size: DiT hidden width.
        dit_head_dim: DiT attention head dim.
        dit_kv_heads: DiT key/value heads.
        num_inference_steps: Euler integration steps for inference.
        flow_sampling: Training timestep distribution.
        local_window: Local-attention window for the action tokens.
        training_repeat: Per-sample training repeat factor.
        enable_freq: Add the frequency-domain loss term.
        prefix_mask_prob: Probability of masking a prefix token in training.
        async_train: Randomly condition on an action prefix in training.
        camera_views: Ordered camera view names for the prompt.
        image_resolution: Target image resolution (unused placeholder kept for
            config parity; the Qwen3-VL processor performs area-based resizing).
        tokenizer_max_length: Maximum tokenizer length.
        gradient_checkpointing: Enable gradient checkpointing.
        compile_model: Whether to use torch.compile.
        compile_mode: Torch compile mode.
        freeze_vision_encoder: Freeze the vision encoder.
        normalization_mode: Normalization method for state/action features.
        optimizer_lr: Learning rate.
        optimizer_betas: Adam beta coefficients.
        optimizer_eps: Optimizer epsilon.
        optimizer_weight_decay: Weight decay coefficient.
        optimizer_grad_clip_norm: Maximum gradient norm for clipping.
        scheduler_warmup_steps: Number of warmup steps.
        scheduler_decay_steps: Cosine decay horizon in steps (``None`` auto).
        scheduler_decay_lr: Final learning rate after decay.
        dataset_stats: Dataset stats for eager initialization.

    Example:
        Training:

        >>> policy = XR0(optimizer_lr=2.5e-5)
        >>> trainer = physicalai.train.Trainer(max_epochs=100)
        >>> trainer.fit(policy, datamodule)
    """

    def __init__(  # noqa: PLR0913
        self,
        vlm_model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        vlm_attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "flash_attention_2",
        dtype: Literal["bfloat16", "float32"] = "bfloat16",
        n_obs_steps: int = 1,
        chunk_size: int = 30,
        n_action_steps: int = 30,
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        state_len: int = 1,
        *,
        dit_num_layers: int = 16,
        dit_hidden_size: int = 1024,
        dit_head_dim: int = 128,
        dit_kv_heads: int = 8,
        num_inference_steps: int = 5,
        flow_sampling: Literal["beta", "logit_normal", "uniform"] = "beta",
        local_window: int = 4,
        training_repeat: int = 4,
        enable_freq: bool = False,
        prefix_mask_prob: float = 0.5,
        async_train: bool = False,
        camera_views: tuple[str, ...] = ("base", "wrist_left"),
        image_resolution: tuple[int, int] = (256, 256),
        tokenizer_max_length: int = 256,
        gradient_checkpointing: bool = True,
        compile_model: bool = False,
        compile_mode: str = "max-autotune",
        freeze_vision_encoder: bool = False,
        normalization_mode: Literal["MEAN_STD", "QUANTILES"] = "QUANTILES",
        optimizer_lr: float = 2.5e-5,
        optimizer_betas: tuple[float, float] = (0.9, 0.95),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 0.01,
        optimizer_grad_clip_norm: float = 1.0,
        scheduler_warmup_steps: int = 1_000,
        scheduler_decay_steps: int | None = 30_000,
        scheduler_decay_lr: float = 2.5e-6,
        dataset_stats: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the XR0 policy."""
        super().__init__(n_action_steps=n_action_steps)

        self.config = XR0Config(
            vlm_model_id=vlm_model_id,
            vlm_attn_implementation=vlm_attn_implementation,
            dtype=dtype,
            n_obs_steps=n_obs_steps,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            state_len=state_len,
            dit_num_layers=dit_num_layers,
            dit_hidden_size=dit_hidden_size,
            dit_head_dim=dit_head_dim,
            dit_kv_heads=dit_kv_heads,
            num_inference_steps=num_inference_steps,
            flow_sampling=flow_sampling,
            local_window=local_window,
            training_repeat=training_repeat,
            enable_freq=enable_freq,
            prefix_mask_prob=prefix_mask_prob,
            async_train=async_train,
            camera_views=camera_views,
            image_resolution=image_resolution,
            tokenizer_max_length=tokenizer_max_length,
            gradient_checkpointing=gradient_checkpointing,
            compile_model=compile_model,
            compile_mode=compile_mode,
            freeze_vision_encoder=freeze_vision_encoder,
            normalization_mode=normalization_mode,
            optimizer_lr=optimizer_lr,
            optimizer_betas=optimizer_betas,
            optimizer_eps=optimizer_eps,
            optimizer_weight_decay=optimizer_weight_decay,
            optimizer_grad_clip_norm=optimizer_grad_clip_norm,
            scheduler_warmup_steps=scheduler_warmup_steps,
            scheduler_decay_steps=scheduler_decay_steps,
            scheduler_decay_lr=scheduler_decay_lr,
        )

        self.save_hyperparameters(ignore=["config", "compile_model"])
        self._set_hparam_keys()

        self.model: XR0Model | None = None
        self._preprocessor: XR0Preprocessor | None = None
        self._postprocessor: XR0Postprocessor | None = None
        self._dataset_stats = dataset_stats

        if dataset_stats is not None:
            self._initialize_model(dataset_stats)

    def _set_hparam_keys(self) -> None:
        """Sync top-level checkpoint hparams from the resolved policy config."""
        for key, value in self.config.__dict__.items():
            if key == "compile_model" or key not in self.hparams:
                continue
            self.hparams[key] = value
        self.hparams["config"] = self.config.to_dict()

    def _initialize_model(self, dataset_stats: dict[str, dict[str, Any]]) -> None:
        """Build the model and preprocessors from dataset statistics."""
        cfg = self.config
        self.model = XR0Model(
            vlm_model_id=cfg.vlm_model_id,
            vlm_attn_implementation=cfg.vlm_attn_implementation,
            state_shape=(cfg.state_len, cfg.max_state_dim),
            action_shape=(cfg.chunk_size, cfg.max_action_dim),
            dit_num_layers=cfg.dit_num_layers,
            dit_hidden_size=cfg.dit_hidden_size,
            dit_head_dim=cfg.dit_head_dim,
            dit_kv_heads=cfg.dit_kv_heads,
            num_steps=cfg.num_inference_steps,
            flow_sampling=cfg.flow_sampling,
            local_window=cfg.local_window,
            training_repeat=cfg.training_repeat,
            enable_freq=cfg.enable_freq,
            prefix_mask_prob=cfg.prefix_mask_prob,
            async_train=cfg.async_train,
            dtype=_DTYPES[cfg.dtype],
        )

        self._preprocessor, self._postprocessor = make_xr0_preprocessors(
            camera_views=cfg.camera_views,
            max_state_dim=cfg.max_state_dim,
            max_action_dim=cfg.max_action_dim,
            stats=dataset_stats,
            processor_name=cfg.vlm_model_id,
        )
        self._dataset_stats = dataset_stats

    def setup(self, stage: str) -> None:
        """Build the model from the datamodule statistics (lazy path).

        Raises:
            TypeError: If the train dataset is not a physicalai Dataset.
        """
        del stage
        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        train_dataset = datamodule.train_dataset
        if not isinstance(train_dataset, Dataset):
            msg = f"Expected physicalai Dataset, got {type(train_dataset)}"
            raise TypeError(msg)

        stats_dict = train_dataset.stats
        if self.model is None:
            self.hparams["dataset_stats"] = stats_dict
            self._initialize_model(stats_dict)

        reformat_dataset_to_match_policy(self, datamodule)

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Forward pass: training loss (train) or action chunk (eval).

        Returns:
            Loss tuple in training mode, or action tensor in eval mode.

        Raises:
            ValueError: If the model is not initialized.
        """
        if self.training:
            if self.model is None or self._preprocessor is None:
                msg = "Model is not initialized"
                raise ValueError(msg)
            processed = self._preprocessor(batch.to_dict())
            return self.model(processed)
        return self.predict_action_chunk(batch)

    def compute_val_loss(self, batch: Observation) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the validation loss.

        Returns:
            Tuple of (loss tensor, loss dict).

        Raises:
            ValueError: If the model is not initialized.
        """
        if self.model is None or self._preprocessor is None:
            msg = "Model is not initialized"
            raise ValueError(msg)
        processed = self._preprocessor(batch.to_dict())
        return self.model.compute_val_loss(processed)

    @torch.no_grad()
    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict a chunk of actions from an observation.

        Returns:
            Denormalized action chunk tensor.

        Raises:
            ValueError: If the model is not initialized.
        """
        from physicalai.data.observation import ACTION  # noqa: PLC0415

        if self.model is None or self._preprocessor is None or self._postprocessor is None:
            msg = "Model is not initialized"
            raise ValueError(msg)
        processed = self._preprocessor(batch.to(self.device).to_dict())
        actions = self.model.predict_action_chunk(processed)
        return self._postprocessor({ACTION: actions})[ACTION]

    def training_step(self, batch: Observation, batch_idx: int) -> torch.Tensor:
        """Lightning training step.

        Returns:
            Training loss tensor.
        """
        del batch_idx
        loss, loss_dict = self(batch)
        self.log("train/loss", loss_dict["loss"], prog_bar=True)
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure the AdamW optimizer and cosine-decay-with-warmup scheduler.

        Returns:
            Dict with optimizer and lr_scheduler config.
        """
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.optimizer_lr,
            weight_decay=self.config.optimizer_weight_decay,
            betas=self.config.optimizer_betas,
            eps=self.config.optimizer_eps,
        )

        num_training_steps = self.trainer.estimated_stepping_batches
        num_decay_steps = self.config.scheduler_decay_steps
        if num_decay_steps is None:
            num_decay_steps = num_training_steps

        scheduler = cosine_decay_with_warmup_scheduler(
            optimizer,
            peak_lr=self.config.optimizer_lr,
            decay_lr=self.config.scheduler_decay_lr,
            num_warmup_steps=self.config.scheduler_warmup_steps,
            num_decay_steps=num_decay_steps,
            num_training_steps=num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Configure gradient clipping from the policy config."""
        clip_val = gradient_clip_val if gradient_clip_val is not None else self.config.optimizer_grad_clip_norm
        if clip_val and clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm or "norm",
            )
