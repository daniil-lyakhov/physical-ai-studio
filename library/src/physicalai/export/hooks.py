# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Post-export hooks for model compression and optimization."""

import logging

logger = logging.getLogger(__name__)


def compress_weights_openvino_int8_sym(model_path: str) -> None:
    """Compress an exported OpenVINO model to INT8 symmetric weights.

    Reads the model from ``model_path``, applies ``nncf.compress_weights``
    with ``INT8_SYM`` mode, and overwrites the model file in-place.

    Args:
        model_path: Path to the exported OpenVINO IR model (``.xml`` file).

    Raises:
        ImportError: If ``nncf`` is not installed.
    """
    try:
        import nncf  # noqa: PLC0415
    except ImportError as e:
        msg = "nncf is required for weight compression hooks. Install with: pip install physicalai-train[nncf]"
        raise ImportError(msg) from e

    import openvino  # noqa: PLC0415

    logger.info("Compressing weights to INT8_SYM: %s", model_path)

    core = openvino.Core()
    ov_model = core.read_model(model_path)

    compressed_model = nncf.compress_weights(ov_model, mode=nncf.CompressWeightsMode.INT8_SYM)

    openvino.save_model(compressed_model, model_path)
    logger.info("INT8_SYM weight compression complete: %s", model_path)
