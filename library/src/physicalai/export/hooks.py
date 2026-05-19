# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Post-export hooks for model compression and optimization."""

import logging
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def compress_weights_openvino_int8_sym(model_path: str) -> None:
    """Compress an exported OpenVINO model to INT8 symmetric weights.

    Reads the model from ``model_path``, applies ``nncf.compress_weights``
    with ``INT8_SYM`` mode, and replaces the original model files.

    The compressed model is first saved to a temporary directory, then the
    original ``.xml`` and ``.bin`` files are replaced atomically to avoid
    issues with OpenVINO overwriting files it currently has open.

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

    xml_path = Path(model_path)
    bin_path = xml_path.with_suffix(".bin")

    core = openvino.Core()
    ov_model = core.read_model(str(xml_path))

    compressed_model = nncf.compress_weights(ov_model, mode=nncf.CompressWeightsMode.INT8_SYM)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_xml = Path(tmp_dir) / xml_path.name
        openvino.save_model(compressed_model, str(tmp_xml))

        # Release model references before touching original files —
        # OpenVINO memory-maps the .bin file, causing segfault if deleted while mapped.
        del compressed_model, ov_model, core

        # Replace original files with compressed ones
        xml_path.unlink()
        bin_path.unlink()
        tmp_bin = tmp_xml.with_suffix(".bin")
        shutil.move(str(tmp_xml), str(xml_path))
        shutil.move(str(tmp_bin), str(bin_path))

    logger.info("INT8_SYM weight compression complete: %s", model_path)
