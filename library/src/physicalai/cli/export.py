# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001

"""Register the ``physicalai export`` subcommand."""

from __future__ import annotations

import logging
from pathlib import Path

from jsonargparse import ActionConfigFile, ArgumentParser, Namespace
from physicalai.cli._spec import SubcommandSpec  # noqa: PLC2701

from physicalai.cli._help import print_export_help  # noqa: PLC2701
from physicalai.cli._logging import configure_console_logging  # noqa: PLC2701
from physicalai.cli._policy import load_policy  # noqa: PLC2701

logger = logging.getLogger(__name__)

HELP = "Export a trained policy for deployment."


def build_parser() -> ArgumentParser:
    """Build the ``export`` parser.

    Returns:
        Parser for ``physicalai export``.
    """
    parser = ArgumentParser(prog="physicalai export", description=HELP)
    parser.add_argument("--config", action=ActionConfigFile, help="YAML/JSON config file.")
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Policy class path (e.g., physicalai.policies.ACT).",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to Lightning checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        help="Export backend: onnx, openvino, executorch, or torch.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where export artifacts are written.",
    )
    parser.add_argument(
        "--trust_local_checkpoint",
        action="store_true",
        help="Allow loading a trusted local XR0 Lightning checkpoint that contains Python Feature objects.",
    )
    return parser


def run(parser: ArgumentParser, cfg: Namespace) -> int:
    """Dispatch ``export`` by loading a policy and invoking its export contract.

    Args:
        parser: Parser used for the export subcommand.
        cfg: Parsed configuration namespace.

    Returns:
        Process exit code.

    Raises:
        TypeError: If the loaded policy does not implement ``export()``.
        ValueError: If XR0 OpenVINO export lacks a trusted local checkpoint opt-in.
    """
    del parser
    configure_console_logging()
    if (
        cfg.policy
        in {
            "physicalai.policies.XR0",
            "physicalai.policies.xr0.XR0",
            "physicalai.policies.xr0.policy.XR0",
        }
        and cfg.backend == "openvino"
    ):
        if not cfg.trust_local_checkpoint:
            msg = "XR0 OpenVINO export requires --trust_local_checkpoint for a checkpoint you created locally."
            raise ValueError(msg)
        checkpoint = Path(cfg.ckpt_path).resolve()
        if not checkpoint.is_relative_to(Path.cwd().resolve()) or not checkpoint.is_file():
            msg = "XR0 checkpoint must be an existing file inside the current working directory"
            raise ValueError(msg)
        from physicalai.policies import XR0  # noqa: PLC0415

        # Trusted local Lightning checkpoints embed Feature objects; unpickling
        # arbitrary user-supplied checkpoints is not safe.
        policy = XR0.load_from_checkpoint(checkpoint, map_location="cpu", weights_only=False, dtype="bfloat16")
        device = "cpu"
        policy.eval()
    else:
        policy, device = load_policy(cfg.policy, cfg.ckpt_path)
    if not hasattr(policy, "export"):
        msg = f"Policy '{cfg.policy}' does not support export()."
        raise TypeError(msg)

    logger.info("Policy: %s", type(policy).__name__)
    logger.info("Device: %s", device)
    policy.export(cfg.output_dir, backend=cfg.backend)
    logger.info("Export saved to %s", cfg.output_dir)
    return 0


def print_help(prog: str) -> None:
    """Print lightweight help without building the full parser."""
    print_export_help(prog, description=HELP)


def register() -> SubcommandSpec:
    """Return the ``export`` subcommand spec.

    Returns:
        Registered spec for the shared CLI host.
    """
    return SubcommandSpec(name="export", parser=build_parser(), dispatch=run, help=HELP)
