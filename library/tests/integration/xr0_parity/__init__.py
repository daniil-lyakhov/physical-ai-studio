# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cross-environment parity tests for the vendored XR0 policy.

These modules load the *source* XR0 implementation (``mibot``, transformers
4.57.1, its own ``env``) and the *framework* XR0 implementation
(``physicalai``, transformers 5.3.0, its own ``env``), run both on identical
synthetic inputs with the real LIBERO checkpoint, and compare the predicted
action chunks. The two implementations cannot share a Python process (they pin
different ``transformers`` versions), so the runner is executed once per
environment as a subprocess and the saved outputs are compared.
"""
