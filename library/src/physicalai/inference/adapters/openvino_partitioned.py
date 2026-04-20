# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Partitioned OpenVINO adapter for Pi0.5 inference.

Wraps 3 separate OpenVINO IR models (paligemma encoder, gemma expert decoder,
action output head) behind the standard RuntimeAdapter interface, enabling
integration with InferenceModel and the evaluation pipeline.
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

    Loads 3 separately exported IR models and runs the full denoising
    inference pipeline (paligemma → N×expert → head) in a single
    ``predict()`` call, returning the action chunk.

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
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the partitioned adapter.

        Args:
            device: OpenVINO device ('CPU', 'GPU', 'NPU', 'AUTO').
            num_inference_steps: Number of denoising iterations.
            chunk_size: Number of action steps in a chunk.
            max_action_dim: Maximum action dimension (padded).
            **kwargs: Additional OpenVINO compile options.
        """
        super().__init__(device, **kwargs)
        self._num_inference_steps = num_inference_steps
        self._chunk_size = chunk_size
        self._max_action_dim = max_action_dim

        self._paligemma_compiled: openvino.CompiledModel | None = None
        self._expert_compiled: openvino.CompiledModel | None = None
        self._head_compiled: openvino.CompiledModel | None = None

        self._input_names: list[str] = []
        self._output_names: list[str] = ["action"]

    def load(self, model_path: Path) -> None:
        """Load the 3 partitioned IR models from a directory.

        Expects the directory to contain:
        - paligemma_encoder.xml / .bin
        - gemma_expert_decoder.xml / .bin
        - action_output_head.xml / .bin

        Args:
            model_path: Path to directory containing the 3 model files.
                Can also be a path to the paligemma .xml file (directory
                is inferred).

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

        paligemma_path = model_dir / "paligemma_encoder.xml"
        expert_path = model_dir / "gemma_expert_decoder.xml"
        head_path = model_dir / "action_output_head.xml"

        for path in (paligemma_path, expert_path, head_path):
            if not path.exists():
                msg = f"Model file not found: {path}"
                raise FileNotFoundError(msg)

        core = ov.Core()
        self._paligemma_compiled = core.compile_model(
            core.read_model(str(paligemma_path)), device_name=self.device, config=self.config
        )
        self._expert_compiled = core.compile_model(
            core.read_model(str(expert_path)), device_name=self.device, config=self.config
        )
        self._head_compiled = core.compile_model(
            core.read_model(str(head_path)), device_name=self.device, config=self.config
        )

        # Input names come from the paligemma encoder (first model in the chain)
        self._input_names = [
            inp.any_name for inp in self._paligemma_compiled.inputs
        ]

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run the full 3-stage denoising inference pipeline.

        Executes:
        1. PaliGemma encoder: inputs → KV cache + prefix masks
        2. Expert decoder × N steps: denoising loop
        3. Action head: final projection

        Args:
            inputs: Dict with keys matching paligemma encoder inputs
                (e.g., "image", "img_mask", "tokens", "masks").

        Returns:
            Dict with "action" key containing the denoised action chunk
            of shape (batch, chunk_size, max_action_dim).

        Raises:
            RuntimeError: If models are not loaded.
        """
        if self._paligemma_compiled is None:
            msg = "Models not loaded. Call load() first."
            raise RuntimeError(msg)

        # Stage 1: Prefix encoding
        paligemma_result = self._paligemma_compiled(inputs)
        cache_keys = paligemma_result[0]
        cache_values = paligemma_result[1]
        prefix_pad_masks = paligemma_result[2]

        # Determine batch size from first input
        first_input = next(iter(inputs.values()))
        batch_size = first_input.shape[0]

        # Stage 2: Iterative denoising
        x_t = np.zeros(
            (batch_size, self._chunk_size, self._max_action_dim),
            dtype=np.float32,
        )
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
            expert_result = self._expert_compiled(expert_inputs)
            suffix_out = expert_result[0]

            # Stage 3: Action projection
            head_result = self._head_compiled({"suffix_out": suffix_out})
            v_t = head_result[0]

            x_t = x_t + dt * v_t

        return {"action": x_t}

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
