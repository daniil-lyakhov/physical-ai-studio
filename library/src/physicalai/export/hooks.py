# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Post-export hooks for model compression and optimization."""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch.export.exported_program import ExportedProgram

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


def compress_weights_executorch_openvino_int8_sym(
    exported_program: "ExportedProgram",
    example_args: tuple[Any, ...],
) -> "ExportedProgram":
    """Compress an ExecuTorch/OpenVINO exported program to INT8 symmetric weights.

    Applies ``nncf.compress_weights`` with ``INT8_SYM`` mode to the FX module of
    ``exported_program`` and re-exports the compressed module.

    Unlike :func:`compress_weights_openvino_int8_sym`, this operates on the
    in-memory exported program rather than a serialized artifact. Weight
    compression must run before the program is lowered with the OpenVINO
    partitioner, since the resulting ``.pte`` file embeds an already-compiled
    OpenVINO blob that can no longer be re-compressed.

    Args:
        exported_program: The ATen-dialect exported program to compress.
        example_args: Example positional arguments used to re-export the
            compressed module.

    Returns:
        A new exported program with INT8_SYM compressed weights.

    Raises:
        ImportError: If ``nncf`` is not installed.
    """
    try:
        import nncf  # noqa: PLC0415
    except ImportError as e:
        msg = "nncf is required for weight compression hooks. Install with: pip install physicalai-train[nncf]"
        raise ImportError(msg) from e

    import torch  # noqa: PLC0415

    logger.info("Compressing ExecuTorch/OpenVINO weights to INT8_SYM")

    compressed_module = nncf.compress_weights(
        exported_program.module(),
        mode=nncf.CompressWeightsMode.INT8_SYM,
    )
    compressed_program = torch.export.export(compressed_module, example_args)

    logger.info("INT8_SYM weight compression complete")
    return compressed_program
