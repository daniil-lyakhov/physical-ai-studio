# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Partitioned OpenVINO adapter for Pi0.5 inference.

Wraps 4 separate OpenVINO IR models (image embedder, prefix LM, gemma expert
decoder, action output head) plus a language embedding weight file behind the
standard RuntimeAdapter interface, enabling integration with InferenceModel
and the evaluation pipeline.

The image embedder is called once per camera, language embedding is done via
NumPy lookup, and the prefix LM fuses all embeddings into a KV cache — so
the same exported model works for any number of cameras.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from physicalai.inference.adapters.base import RuntimeAdapter

if TYPE_CHECKING:
    from pathlib import Path

    import openvino


class PartitionedOpenVINOAdapter(RuntimeAdapter):
    """OpenVINO adapter for partitioned Pi0.5 models.

    Loads 4 separately exported IR models and runs the full denoising
    inference pipeline (embedder × N cameras → prefix LM → N×expert → head)
    in a single ``predict()`` call, returning the action chunk.

    The denoising loop runs inside ``predict()`` so that from the
    runner's perspective, one call = one full action chunk — matching
    the behavior of the monolithic OpenVINO adapter.

    Examples:
        >>> adapter = PartitionedOpenVINOAdapter(
        ...     num_inference_steps=10, chunk_size=50, max_action_dim=32
        ... )
        >>> adapter.load(Path("/exports/partitioned"))
        >>> outputs = adapter.predict({"image": img, "img_mask": mask, ...})
        >>> outputs["action"].shape  # (1, 50, 32)
    """

    def __init__(
        self,
        device: str = "CPU",
        num_inference_steps: int = 10,
        chunk_size: int = 50,
        max_action_dim: int = 32,
        action_dim: int | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the partitioned adapter.

        Args:
            device: OpenVINO device ('CPU', 'GPU', 'NPU', 'AUTO').
            num_inference_steps: Number of denoising iterations.
            chunk_size: Number of action steps in a chunk.
            max_action_dim: Maximum action dimension (padded).
            action_dim: Actual action dimension (unpadded). If provided,
                output actions are trimmed to this size.
            **kwargs: Additional OpenVINO compile options.
        """
        super().__init__(device, **kwargs)
        self._num_inference_steps = num_inference_steps
        self._chunk_size = chunk_size
        self._max_action_dim = max_action_dim
        self._action_dim = action_dim

        self._embedder_compiled: openvino.CompiledModel | None = None
        self._prefix_lm_compiled: openvino.CompiledModel | None = None
        self._expert_compiled: openvino.CompiledModel | None = None
        self._head_compiled: openvino.CompiledModel | None = None
        self._lang_weight: np.ndarray | None = None  # (vocab, hidden) pre-scaled

        self._input_names: list[str] = []
        self._output_names: list[str] = ["action"]

    def load(self, model_path: Path) -> None:
        """Load the 4 partitioned IR models and language embedding from a directory.

        Expects the directory to contain:
        - image_embedder.xml / .bin
        - prefix_lm.xml / .bin
        - gemma_expert_decoder.xml / .bin
        - action_output_head.xml / .bin
        - language_embedding.npy

        Args:
            model_path: Path to directory containing the model files.
                Can also be a path to any .xml file (directory is inferred).

        Raises:
            ImportError: If OpenVINO is not installed.
            FileNotFoundError: If any model file is missing.
        """
        try:
            import openvino as ov  # noqa: PLC0415
        except ImportError as e:
            msg = "OpenVINO is not installed. Install with: uv pip install openvino"
            raise ImportError(msg) from e

        from pathlib import Path as _Path

        model_dir = _Path(model_path)
        if model_dir.is_file():
            model_dir = model_dir.parent

        embedder_path = model_dir / "image_embedder.xml"
        prefix_lm_path = model_dir / "prefix_lm.xml"
        expert_path = model_dir / "gemma_expert_decoder.xml"
        head_path = model_dir / "action_output_head.xml"
        lang_embed_path = model_dir / "language_embedding.npy"

        for path in (embedder_path, prefix_lm_path, expert_path, head_path, lang_embed_path):
            if not path.exists():
                msg = f"Model file not found: {path}"
                raise FileNotFoundError(msg)

        core = ov.Core()
        self._embedder_compiled = core.compile_model(
            core.read_model(str(embedder_path)), device_name=self.device, config=self.config,
        )
        self._prefix_lm_compiled = core.compile_model(
            core.read_model(str(prefix_lm_path)), device_name=self.device, config=self.config,
        )
        self._expert_compiled = core.compile_model(
            core.read_model(str(expert_path)), device_name=self.device, config=self.config,
        )
        self._head_compiled = core.compile_model(
            core.read_model(str(head_path)), device_name=self.device, config=self.config,
        )
        self._lang_weight = np.load(str(lang_embed_path))

        # NOTE: input_names is left empty so that InferenceModel._prepare_inputs
        # passes all preprocessor outputs through. The adapter's predict() method
        # picks the keys it needs (images, image_masks, tokenized_prompt, etc.).
        self._input_names = []

    def predict(
        self,
        inputs: dict[str, np.ndarray],
        *,
        noise: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Run the full denoising inference pipeline.

        Executes:
        1. Image embedder × N cameras → image embeddings
        2. Language embedding via NumPy lookup
        3. Prefix LM: concatenated embeddings → KV cache
        4. Expert decoder × num_inference_steps: denoising loop
        5. Action head: final action projection

        Args:
            inputs: Dict with keys: ``images`` (n_cameras, batch, C, H, W),
                ``image_masks`` (n_cameras, batch), ``tokenized_prompt``
                (batch, seq), ``tokenized_prompt_mask`` (batch, seq).
            noise: Optional initial noise tensor of shape
                (batch, chunk_size, max_action_dim). If ``None``,
                random noise is sampled.

        Returns:
            Dict with "action" key containing the denoised action chunk.

        Raises:
            RuntimeError: If models are not loaded.
        """
        if self._embedder_compiled is None:
            msg = "Models not loaded. Call load() first."
            raise RuntimeError(msg)

        # --- Parse inputs ---
        lm_inputs, prefix_pad_masks, batch_size = self._embed_inputs(inputs)

        # --- Stage 3: Prefix LM → KV cache ---
        prefix_result = self._prefix_lm_compiled(lm_inputs)
        cache_keys = prefix_result[0]
        cache_values = prefix_result[1]

        # --- Stage 4: Iterative denoising ---
        x_t = noise
        if noise is None:
            x_t = self._sample_noise(batch_size)
            
        dt = -1.0 / self._num_inference_steps

        for step in range(self._num_inference_steps):
            time = 1.0 + step * dt
            time_tensor = np.full((batch_size,), time, dtype=np.float32)

            expert_inputs = {
                "x_t": x_t,
                "timestep": time_tensor,
                "cache_keys": cache_keys,
                "cache_values": cache_values,
                "prefix_pad_masks": prefix_pad_masks,
            }

            x_t = self._apply_expert(expert_inputs, dt)

        # Trim padded action dimension to actual action size
        if self._action_dim is not None:
            x_t = x_t[:, :, :self._action_dim]

        return {"action": x_t}

    def _apply_expert(self, expert_inputs, dt):
        expert_result = self._expert_compiled(expert_inputs)
        suffix_out = expert_result[0]

        # Stage 5: Action projection
        head_result = self._head_compiled({"suffix_out": suffix_out})
        v_t = head_result[0]
        return  expert_inputs["x_t"] + dt * v_t

    def _sample_noise(self, batch_size):
        return np.random.randn(
                batch_size, self._chunk_size, self._max_action_dim,
            ).astype(np.float32)

    def _embed_inputs(self, inputs):
        paligemma_inputs = dict(inputs)
        images = paligemma_inputs.pop("images")
        image_masks = paligemma_inputs.pop("image_masks")
        tokens = paligemma_inputs["tokenized_prompt"]
        token_masks = paligemma_inputs["tokenized_prompt_mask"]

        if images.ndim == 4:  # noqa: PLR2004
            images = images[np.newaxis]
            image_masks = image_masks[np.newaxis]

        n_cameras = images.shape[0]
        batch_size = images.shape[1]

        # --- Stage 1: Image embedding (per camera) ---
        embs = []
        pad_masks = []
        att_masks_list: list[int] = []

        for cam_idx in range(n_cameras):
            result = self._embedder_compiled({"image": images[cam_idx]})
            img_emb = result[0]  # (batch, num_patches, hidden)
            num_patches = img_emb.shape[1]

            embs.append(img_emb)
            # Expand image mask (batch,) → (batch, num_patches)
            cam_mask = image_masks[cam_idx][:, np.newaxis]  # (batch, 1)
            pad_masks.append(np.broadcast_to(cam_mask, (batch_size, num_patches)))
            att_masks_list.extend([0] * num_patches)

        # --- Stage 2: Language embedding (NumPy lookup) ---
        lang_emb = self._lang_weight[tokens]  # (batch, token_len, hidden)
        embs.append(lang_emb)
        pad_masks.append(token_masks)
        att_masks_list.extend([0] * tokens.shape[1])

        # --- Concatenate and build masks ---
        prefix_embs = np.concatenate(embs, axis=1).astype(np.float32)
        prefix_pad_masks = np.concatenate(pad_masks, axis=1).astype(bool)
        prefix_att_masks = np.broadcast_to(
            np.array(att_masks_list, dtype=bool)[np.newaxis, :],
            (batch_size, len(att_masks_list)),
        ).copy()

        # Pre-compute 4D attention mask and position IDs in NumPy
        # (these ops were moved out of the exported PrefixLM IR).
        MASK_VALUE = -2.3819763e38  # OPENPI_ATTENTION_MASK_VALUE  # noqa: N806
        cumsum = np.cumsum(prefix_att_masks.astype(np.int64), axis=1)
        att_2d = (cumsum[:, np.newaxis, :] <= cumsum[:, :, np.newaxis])  # (B, seq, seq)
        pad_2d = (prefix_pad_masks[:, np.newaxis, :] & prefix_pad_masks[:, :, np.newaxis])
        att_2d = att_2d & pad_2d
        att_4d = att_2d[:, np.newaxis, :, :]  # (B, 1, seq, seq)
        prefix_att_2d_masks_4d = np.where(att_4d, 0.0, MASK_VALUE).astype(np.float32)
        prefix_position_ids = (np.cumsum(prefix_pad_masks.astype(np.int64), axis=1) - 1)
        return {
            "prefix_embs": prefix_embs,
            "prefix_att_2d_masks_4d": prefix_att_2d_masks_4d,
            "prefix_position_ids": prefix_position_ids,
        }, prefix_pad_masks, batch_size

    def default_device(self) -> str:  # noqa: PLR6301
        """Get default OpenVINO device.

        Returns:
            'CPU' (most compatible OpenVINO device)
        """
        return "CPU"

    @property
    def input_names(self) -> list[str]:
        """Get input tensor names (from paligemma encoder).

        Returns:
            List of input names.
        """
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        """Get output tensor names.

        Returns:
            List containing 'action'.
        """
        return self._output_names
