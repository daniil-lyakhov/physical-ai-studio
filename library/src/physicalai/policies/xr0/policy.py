# Copyright (C) 2026 Xiaomi Corporation.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""XR0 Policy - Lightning wrapper for training and inference."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import torch
from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype, InferenceFeatureType
from physicalai.inference.manifest import ComponentSpec

from physicalai.data.dataset import Dataset
from physicalai.data.observation import ACTION, IMAGES, STATE, TASK, Feature, FeatureType, NormalizationParameters
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.export.backends import ExportParameters, TorchExportParameters
from physicalai.policies.base import Policy
from physicalai.train.schedulers import cosine_decay_with_warmup_scheduler
from physicalai.train.utils import reformat_dataset_to_match_policy

from .config import XR0Config
from .preprocessor import make_xr0_preprocessors
from .pretrained_utils import extract_xr0_dataset_stats, load_xr0_pretrained_weights, resolve_pretrained_path
from .vla import XR0Model

if TYPE_CHECKING:
    from pathlib import Path

    from physicalai.data import Observation

    from .preprocessor import XR0Postprocessor, XR0Preprocessor

logger = logging.getLogger(__name__)

_DTYPES: dict[str, torch.dtype] = {"bfloat16": torch.bfloat16, "float32": torch.float32}


class XR0(ExportablePolicyMixin, Policy):
    """XR0 Policy - Xiaomi's flow-matching VLA model.

    Lightning wrapper for training and inference with :class:`XR0Model`.

    Args:
        pretrained_name_or_path: Optional local path or HuggingFace repo id of a
            pretrained XR0 checkpoint (e.g.
            ``"XiaomiRobotics/Xiaomi-Robotics-0-LIBERO"``). When given, the
            weights are loaded into the model once it is built.
        input_features: Optional explicit observation feature schema
            (``list[Feature]``). When omitted, it is traced back from the
            training dataset in :meth:`setup`. Must be given together with
            ``output_features``.
        output_features: Optional explicit action feature schema
            (``list[Feature]``). Must be given together with ``input_features``.
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

        Fine-tuning from the pretrained LIBERO checkpoint:

        >>> policy = XR0(pretrained_name_or_path="XiaomiRobotics/Xiaomi-Robotics-0-LIBERO")
        >>> trainer.fit(policy, datamodule)
    """

    def __init__(  # noqa: PLR0913
        self,
        pretrained_name_or_path: str | Path | None = None,
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
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        dit_num_layers: int = 16,
        dit_hidden_size: int = 1024,
        dit_head_dim: int = 128,
        dit_kv_heads: int = 8,
        num_inference_steps: int = 5,
        flow_sampling: Literal["beta", "logit_normal", "uniform"] = "beta",
        local_window: int = 4,
        training_repeat: int = 4,
        enable_freq: bool = True,
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
        optimizer_lr: float = 1.0e-4,
        optimizer_betas: tuple[float, float] = (0.9, 0.95),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 0.1,
        optimizer_grad_clip_norm: float = 1.0,
        scheduler_warmup_steps: int = 2_000,
        scheduler_decay_steps: int | None = 30_000,
        scheduler_decay_lr: float = 5.0e-7,
        dataset_stats: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the XR0 policy.

        Raises:
            ValueError: If only one of ``input_features`` / ``output_features``
                is provided.
        """
        super().__init__(n_action_steps=n_action_steps)

        # Input/output features must be provided together (or both omitted and
        # traced back from the dataset in ``setup``), mirroring MolmoAct2.
        if bool(input_features) != bool(output_features):
            msg = f"Need both input and output features: input: {input_features} - output: {output_features}"
            raise ValueError(msg)

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
            input_features=input_features,
            output_features=output_features,
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

        self.save_hyperparameters(ignore=["config", "compile_model", "pretrained_name_or_path"])
        self._set_hparam_keys()

        self.model: XR0Model | None = None
        self._preprocessor: XR0Preprocessor | None = None
        self._postprocessor: XR0Postprocessor | None = None
        self._dataset_stats = dataset_stats
        self._input_features = input_features
        self._output_features = output_features

        # Resolve (download) the pretrained checkpoint now; load it into the
        # model once it is built (eager path here, or lazily in ``setup``).
        self._pretrained_path: Path | None = (
            resolve_pretrained_path(pretrained_name_or_path) if pretrained_name_or_path is not None else None
        )

        # When explicit input/output features are given without dataset stats,
        # derive the normalization stats from them so the model can be built
        # eagerly (no training dataset required).
        if dataset_stats is None and input_features is not None and output_features is not None:
            dataset_stats = self._features_to_stats(input_features, output_features)
            self._dataset_stats = dataset_stats

        # When a pretrained checkpoint is given without explicit dataset stats,
        # recover the action-normalization stats from the checkpoint so the
        # policy is usable for standalone inference (no training dataset).
        if dataset_stats is None and pretrained_name_or_path is not None:
            dataset_stats = extract_xr0_dataset_stats(pretrained_name_or_path)
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

        if self._pretrained_path is not None:
            self._load_pretrained_weights(self._pretrained_path)

        self._preprocessor, self._postprocessor = make_xr0_preprocessors(
            camera_views=cfg.camera_views,
            max_state_dim=cfg.max_state_dim,
            max_action_dim=cfg.max_action_dim,
            stats=dataset_stats,
            processor_name=cfg.vlm_model_id,
        )
        self._dataset_stats = dataset_stats

        # When features were not provided (or traced from a dataset) yet,
        # reconstruct the typed schema from the stats dict so the export
        # ``inputs_schema`` / ``outputs_schema`` are feature-driven.
        if self._input_features is None or self._output_features is None:
            self._input_features, self._output_features = self._stats_to_features(dataset_stats)

    def _load_pretrained_weights(self, pretrained_path: Path) -> None:
        """Load remapped pretrained weights into ``self.model`` (non-strict).

        Raises:
            ValueError: If the model has not been built yet.
        """
        if self.model is None:
            msg = "Cannot load pretrained weights before the model is initialized"
            raise ValueError(msg)

        state_dict = load_xr0_pretrained_weights(pretrained_path)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False, assign=True)
        self.model.to(_DTYPES[self.config.dtype])

        if missing:
            msg = f"Missing keys when loading pretrained XR0 weights: {len(missing)} keys"
            logger.warning(msg)
            for key in missing[:10]:
                logger.warning("  - %s", key)
        if unexpected:
            msg = f"Unexpected keys when loading pretrained XR0 weights: {len(unexpected)} keys"
            logger.warning(msg)
            for key in unexpected[:10]:
                logger.warning("  - %s", key)

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

        # Trace the input/output feature schema back from the dataset when it
        # was not provided explicitly at construction time.
        if self._input_features is None or self._output_features is None:
            input_features, output_features = self._dataset_features(train_dataset)
            self._input_features = input_features
            self._output_features = output_features
            self.hparams["input_features"] = input_features
            self.hparams["output_features"] = output_features

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

        num_training_steps = int(self.trainer.estimated_stepping_batches)
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

    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
        """Get the list of export backends supported by the policy.

        XR0 wraps a Qwen3-VL backbone, so only the tracing-free Torch backend is
        supported; graph-based backends (ONNX/OpenVINO/ExecuTorch) are not.

        Returns:
            list[str | ExportBackend]: The supported export backends.
        """
        return [ExportBackend.TORCH]

    @staticmethod
    def _coerce_dataset_feature(feature: Feature) -> Feature:
        """Return a defensive copy of a dataset feature for the schema.

        Returns:
            A new :class:`Feature` with copied normalization data and a concrete
            integer-tuple shape.
        """
        norm = feature.normalization_data
        copied_norm: NormalizationParameters | None = None
        if norm is not None:
            copied_norm = NormalizationParameters(
                mean=norm.mean,
                std=norm.std,
                min=norm.min,
                max=norm.max,
                q01=norm.q01,
                q99=norm.q99,
            )
        shape = tuple(int(dim) for dim in feature.shape) if feature.shape is not None else ()
        return Feature(
            name=str(feature.name),
            ftype=FeatureType(feature.ftype) if feature.ftype is not None else None,
            shape=shape,
            normalization_data=copied_norm,
        )

    @staticmethod
    def _dataset_features(train_dataset: Dataset) -> tuple[list[Feature], list[Feature]]:
        """Trace the input/output feature schema back from the dataset.

        Returns:
            A ``(input_features, output_features)`` tuple built from the
            dataset's observation and action features.
        """
        input_features = [XR0._coerce_dataset_feature(f) for f in train_dataset.observation_features.values()]
        output_features = [XR0._coerce_dataset_feature(f) for f in train_dataset.action_features.values()]
        return input_features, output_features

    @staticmethod
    def _feature_stat_entry(feature: Feature) -> dict[str, Any]:
        """Serialize one feature into a LeRobot-style stats dict entry.

        Returns:
            The per-feature stats entry (normalization values + metadata).
        """
        entry: dict[str, Any] = {}
        norm = feature.normalization_data
        if norm is not None:
            for stat in ("mean", "std", "min", "max", "q01", "q99"):
                value = getattr(norm, stat, None)
                if value is not None:
                    entry[stat] = value
        entry["type"] = feature.ftype.value if feature.ftype is not None else ""
        entry["name"] = feature.name if feature.name is not None else ""
        entry["shape"] = feature.shape if feature.shape is not None else ()
        return entry

    @staticmethod
    def _features_to_stats(
        input_features: list[Feature],
        output_features: list[Feature],
    ) -> dict[str, dict[str, Any]]:
        """Build the stats dict consumed by the preprocessor from typed features.

        Returns:
            A stats dict keyed like :attr:`Dataset.stats`
            (``observation.<name>`` for inputs, ``action`` for outputs).
        """
        stats: dict[str, dict[str, Any]] = {}
        for feature in input_features:
            stats[f"observation.{feature.name}"] = XR0._feature_stat_entry(feature)
        for feature in output_features:
            stats[str(feature.name)] = XR0._feature_stat_entry(feature)
        return stats

    @staticmethod
    def _stats_to_features(
        stats: dict[str, dict[str, Any]],
    ) -> tuple[list[Feature], list[Feature]]:
        """Reconstruct typed input/output features from a stats dict.

        Used when only a stats dict is available (e.g. a pretrained checkpoint),
        so :attr:`inputs_schema` / :attr:`outputs_schema` stay feature-driven.

        Returns:
            A ``(input_features, output_features)`` tuple.
        """
        input_features: list[Feature] = []
        output_features: list[Feature] = []
        for key, stat in stats.items():
            ftype_str = str(stat.get("type", ""))
            if str(FeatureType.ACTION) in ftype_str or ACTION in key:
                ftype = FeatureType.ACTION
            elif str(FeatureType.VISUAL) in ftype_str:
                ftype = FeatureType.VISUAL
            elif str(FeatureType.STATE) in ftype_str or STATE in key:
                ftype = FeatureType.STATE
            else:
                continue
            name = str(stat.get("name", key)).removeprefix("observation.")
            feature = Feature(
                name=name,
                ftype=ftype,
                shape=tuple(stat["shape"]) if stat.get("shape") else (),
                normalization_data=NormalizationParameters(
                    mean=stat.get("mean"),
                    std=stat.get("std"),
                    min=stat.get("min"),
                    max=stat.get("max"),
                    q01=stat.get("q01"),
                    q99=stat.get("q99"),
                ),
            )
            if ftype == FeatureType.ACTION:
                output_features.append(feature)
            else:
                input_features.append(feature)
        return input_features, output_features

    @property
    def input_features(self) -> list[Feature]:
        """Explicit observation feature schema.

        Returns:
            The list of input :class:`Feature` descriptors.

        Raises:
            ValueError: If the features have not been initialized yet.
        """
        if self._input_features is None:
            msg = "Model has not been initialized, no input features exist yet."
            raise ValueError(msg)
        return self._input_features

    @property
    def output_features(self) -> list[Feature]:
        """Explicit action feature schema.

        Returns:
            The list of output :class:`Feature` descriptors.

        Raises:
            ValueError: If the features have not been initialized yet.
        """
        if self._output_features is None:
            msg = "Model has not been initialized, no output features exist yet."
            raise ValueError(msg)
        return self._output_features

    @property
    def inputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the policy's expected model inputs for export.

        Derived from :attr:`input_features` (traced back from the dataset when
        not provided explicitly at construction time).

        Returns:
            A list of feature descriptors covering the robot state, one image
            feature per camera view, and the language task. Returns ``None`` if
            the model or the input features have not been initialized yet.

        Raises:
            ValueError: If an input feature is missing a concrete shape.
        """
        if self.model is None or self._input_features is None:
            return None

        num_image_features = sum(1 for feature in self._input_features if feature.ftype == FeatureType.VISUAL)

        schema: list[InferenceFeature] = []
        for feature in self._input_features:
            if feature.ftype not in {FeatureType.STATE, FeatureType.VISUAL}:
                continue
            if feature.shape is None:
                msg = "input feature missing concrete shape for export"
                raise ValueError(msg)
            if feature.ftype == FeatureType.STATE:
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.STATE,
                        shape=tuple(feature.shape),
                        name=STATE,
                        dtype=InferenceFeatureDtype.FLOAT32,
                    ),
                )
            else:
                feature_name = str(feature.name or "").removeprefix("observation.").removeprefix(f"{IMAGES}.")
                name = IMAGES if num_image_features == 1 else f"{IMAGES}.{feature_name}"
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.VISUAL,
                        shape=tuple(feature.shape),
                        name=name,
                        dtype=InferenceFeatureDtype.FLOAT32,
                    ),
                )

        schema.append(
            InferenceFeature(
                ftype=InferenceFeatureType.LANGUAGE,
                shape=(),
                name=TASK,
                dtype=InferenceFeatureDtype.STRING,
            ),
        )

        return schema

    @property
    def outputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the policy's model output for export.

        Derived from :attr:`output_features`. Returns ``None`` if the model or
        the output features have not been initialized yet.

        Returns:
            A list with a single ``action`` feature of shape
            ``(chunk_size, *action_dim)``, where ``action_dim`` comes from the
            action feature.

        Raises:
            ValueError: If the action feature is missing a concrete shape.
        """
        if self.model is None or self._output_features is None:
            return None

        action_feature = next(
            (feature for feature in self._output_features if feature.ftype == FeatureType.ACTION),
            None,
        )
        if action_feature is None:
            return None
        if action_feature.shape is None:
            msg = "output feature missing concrete shape for export"
            raise ValueError(msg)

        return [
            InferenceFeature(
                ftype=InferenceFeatureType.ACTION,
                shape=(self.config.chunk_size, *tuple(action_feature.shape)),
                name=ACTION,
                dtype=InferenceFeatureDtype.FLOAT32,
            ),
        ]

    @property
    def extra_export_args(self) -> dict[str, ExportParameters]:
        """Additional export arguments for model conversion.

        The reconstructed policy runs its own preprocessor/postprocessor during
        Torch inference, so only lightweight input casting and optional action
        trimming are declared here.

        Returns:
            dict[str, ExportParameters]: A mapping from backend name to its export
            parameters (Torch only).
        """
        torch_postproc_specs: list[ComponentSpec] = []
        if self.config.chunk_size != self.config.n_action_steps:
            torch_postproc_specs.append(
                ComponentSpec(
                    type="action_chunk_trimmer",
                    n_action_steps=self.config.n_action_steps,
                ),
            )

        return {
            "torch": TorchExportParameters(
                preprocessors_specs=[ComponentSpec(type="to_float_tensor")],
                postprocessors_specs=torch_postproc_specs,
            ),
        }
