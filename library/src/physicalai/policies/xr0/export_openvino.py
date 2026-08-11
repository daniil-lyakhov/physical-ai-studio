# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO export workarounds for the XR0 self-contained IR.

Two OpenVINO-specific fixes the XR0 export needs:

**RMSNorm (pre-export, :func:`install_ov_friendly_rmsnorm`).** Every RMSNorm in
the model reduces over a negative axis (``mean(-1)``). The OpenVINO PyTorch
frontend mis-materializes that into a garbage ``ReduceMean`` axis, so the IR
fails to load ("Axis <huge> out of the tensor rank range"). The fix reduces over
the concrete positive ``dim() - 1`` instead; the math is otherwise identical.
The installer rebinds ``forward`` on every RMSNorm *instance* in the module tree
(pass the top-level ``XR0Model`` to cover both the VLM and DiT head). Swapping a
subclass is not enough: the VLM's ``Qwen3VLTextRMSNorm`` modules are built inside
``transformers`` via ``from_pretrained``, so there is no call site to swap.

**GatherND (post-export, :func:`rewrite_openvino_gpu_friendly`).** The Qwen3-VL
attention-mask builder lowers to a ``GatherND`` on a boolean (``u8``) tensor,
which the Intel GPU plugin has no kernel for (CPU is fine). The rewrite runs the
gather on ``i32`` and casts back to ``boolean`` (numerically identical, mask is
0/1). It re-saves via a temp file because ``read_model`` mmaps the ``.bin`` and
writing over it directly would corrupt the source (SIGBUS).
"""

from __future__ import annotations

import types
from typing import Protocol

import torch

# Marker attribute used to make the RMSNorm install idempotent (re-running export
# prep in the same process must not double-wrap a module's forward).
_PATCHED_FLAG = "_ov_friendly_rmsnorm"


class _RMSNormLike(Protocol):
    """Structural type for the RMSNorm modules whose forward we swap."""

    weight: torch.Tensor
    variance_epsilon: float


def ov_friendly_rmsnorm_forward(self: _RMSNormLike, hidden_states: torch.Tensor) -> torch.Tensor:
    """RMSNorm forward that reduces over a positive, static axis.

    Drop-in replacement for the stock ``Qwen2RMSNorm`` / ``Qwen3VLTextRMSNorm``
    forward. Identical math, but the reduction axis is the concrete positive
    ``ndim - 1`` instead of ``-1`` so the OpenVINO PyTorch frontend emits a valid
    ``ReduceMean`` axis constant (a negative axis is mis-materialized and makes the
    exported IR fail to load).

    Args:
        self: The RMSNorm module (provides ``weight`` and ``variance_epsilon``).
        hidden_states: The input activations to normalize.

    Returns:
        The RMS-normalized, weight-scaled activations in the input dtype.
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    axis = hidden_states.dim() - 1  # concrete positive int -> clean ReduceMean axis
    variance = hidden_states.pow(2).mean(axis, keepdim=True)
    hidden_states *= torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)


def _is_rmsnorm(module: torch.nn.Module) -> bool:
    """Return whether ``module`` is an RMSNorm to patch.

    Identified structurally (has ``variance_epsilon`` and ``weight``) and by class
    name suffix, so it matches both the Qwen2 (DiT head) and Qwen3-VL text
    RMSNorm variants without importing the ``transformers`` classes.

    Returns:
        ``True`` if the module is an RMSNorm whose forward should be swapped.
    """
    return (
        type(module).__name__.endswith("RMSNorm") and hasattr(module, "variance_epsilon") and hasattr(module, "weight")
    )


def install_ov_friendly_rmsnorm(module: torch.nn.Module) -> int:
    """Swap every RMSNorm instance in ``module`` to the OpenVINO-friendly forward.

    Walks the whole submodule tree and, for each RMSNorm instance, rebinds its
    ``forward`` to :func:`ov_friendly_rmsnorm_forward`. Pass the top-level
    :class:`~physicalai.policies.xr0.vla.XR0Model` to cover both the Qwen3-VL text
    backbone and the DiT action head in a single call. Idempotent: modules already
    patched are skipped, so it is safe to call more than once.

    Args:
        module: The model (or subtree) whose RMSNorm modules should be patched.

    Returns:
        The number of RMSNorm modules that were patched by this call.
    """
    patched = 0
    for submodule in module.modules():
        if not _is_rmsnorm(submodule) or getattr(submodule, _PATCHED_FLAG, False):
            continue
        submodule.forward = types.MethodType(ov_friendly_rmsnorm_forward, submodule)
        submodule.__dict__[_PATCHED_FLAG] = True
        patched += 1
    return patched
