# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest configuration for export test scripts."""

from __future__ import annotations

from pathlib import Path

import pytest

_DEFAULT_EXPORT_CACHE_DIR = Path(__file__).resolve().parent / ".export_cache" / "partitioned"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom CLI flags for partitioned export tests."""
    parser.addoption(
        "--force-reexport",
        action="store_true",
        default=False,
        help="Force re-export of partitioned OV models even if cached.",
    )
    parser.addoption(
        "--export-cache-dir",
        type=str,
        default=str(_DEFAULT_EXPORT_CACHE_DIR),
        help="Directory to cache exported OV models between runs.",
    )
