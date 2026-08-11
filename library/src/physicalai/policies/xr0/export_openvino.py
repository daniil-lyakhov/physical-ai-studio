# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO export workarounds for the XR0 self-contained IR.

This module bundles the two OpenVINO-specific workarounds the XR0 export needs:
a **pre-export** RMSNorm fix applied to the live model before tracing
(:func:`install_ov_friendly_rmsnorm`), and a **post-export** ``GatherND`` rewrite
applied to the written IR (:func:`rewrite_openvino_gpu_friendly`).

RMSNorm (pre-export)
--------------------
Every RMSNorm in the assembled XR0 model -- the Qwen3-VL text backbone
(``Qwen3VLTextRMSNorm``) *and* the DiT action head
(``Qwen2RMSNorm``; see :mod:`physicalai.policies.xr0.qwen3vl_dit`) -- normalizes
with a **negative** reduction axis::

    variance = hidden_states.pow(2).mean(-1, keepdim=True)

When such a model is converted with the OpenVINO PyTorch frontend, that ``-1``
axis is mis-materialized into a garbage ``ReduceMean`` axis constant, so loading
the exported IR later fails with::

    Axis <huge-number> out of the tensor rank range

The fix is to reduce over a **positive, static** axis. During tracing the tensor
rank is known, so ``hidden_states.dim() - 1`` is a concrete Python ``int`` and the
frontend emits a valid axis constant. The computation is otherwise identical to
stock RMSNorm (same float32 upcast, same ``rsqrt``, same weight scaling), so the
numerics are unchanged.

:func:`install_ov_friendly_rmsnorm` rebinds the ``forward`` of every RMSNorm
*instance* inside a given module tree. It is applied per-instance (not by patching
the ``transformers`` classes globally), so the change is scoped to the exact model
being exported: it never leaks into other models sharing the same process, needs
no import-time class surgery, and is trivially reversible by rebuilding the model.
Passing the top-level :class:`~physicalai.policies.xr0.vla.XR0Model` covers both
the VLM and the DiT head in one call, since ``module.modules()`` recurses into
every submodule.

Note that it is **not enough to just instantiate a new (OV-friendly) subclass**.
Only the DiT head's ``Qwen2RMSNorm`` modules are constructed in our own code and
could be swapped at their call sites. The VLM's ``Qwen3VLTextRMSNorm`` modules are
instantiated *inside* ``transformers`` when ``Qwen3VLForConditionalGeneration``
is built via ``from_pretrained`` -- we never call their constructors, so there is
no call site to swap. Rebinding ``forward`` on the already-built instances is the
only way to reach those, which is why this installer walks the live module tree
instead of relying on a subclass.

GatherND (post-export)
----------------------
The exported XR0 IR loads and runs on the OpenVINO CPU plugin as-is, but the
Intel GPU plugin rejects one construct: the Qwen3-VL attention-mask builder
lowers to a ``GatherND`` on a **boolean** (``u8``) tensor
(``attention_mask`` (i64) -> ``Convert(bool)`` -> ``GatherND`` -> ``LogicalAnd``).
The GPU plugin has no ``u8`` ``GatherND`` kernel ("No layout format available for
gathernd ... data_type: u8"), so model compilation fails there (CPU has the kernel
and loads fine).

:func:`rewrite_openvino_gpu_friendly` rewrites every boolean ``GatherND`` to run on
``i32`` -- which the GPU supports -- and casts the gathered result back to
``boolean`` so the downstream ``LogicalAnd`` is untouched. The mask only ever holds
0/1, so this is numerically identical.

The rewritten model is streamed back to the original ``.xml``/``.bin`` pair via a
temp file. ``read_model`` memory-maps the multi-GB ``.bin`` and ``save_model``
streams weights from that live mapping while writing, so writing directly over the
source would corrupt it (SIGBUS). Writing to a temp path first, dropping the
mapping, then atomically replacing both files avoids that.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import openvino as ov
import openvino.opset13 as ops
import torch

if TYPE_CHECKING:
    import os

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


def rewrite_openvino_gpu_friendly(xml_path: str | os.PathLike[str]) -> int:
    """Rewrite boolean ``GatherND`` ops in an exported IR to run on ``i32``.

    Reads the IR at ``xml_path``, converts every ``GatherND`` whose data input is
    boolean to gather on ``i32`` (casting the result back to boolean), and, if any
    op was rewritten, saves the model back over the original ``.xml``/``.bin`` pair
    using a temp file to avoid corrupting the memory-mapped source weights. A no-op
    (and no re-save) if the IR contains no boolean ``GatherND``.

    Args:
        xml_path: Path to the exported ``.xml`` (its sibling ``.bin`` is loaded and
            rewritten alongside it).

    Returns:
        The number of ``GatherND`` ops that were rewritten.
    """
    xml_path = Path(xml_path)

    core = ov.Core()
    model = core.read_model(str(xml_path))

    rewritten = 0
    for gnd in [op for op in model.get_ops() if op.get_type_name() == "GatherND"]:
        data = gnd.input_value(0)
        if data.get_element_type() != ov.Type.boolean:
            continue
        consumers = list(gnd.output(0).get_target_inputs())
        data_i32 = ops.convert(data, destination_type="i32")
        gnd.input(0).replace_source_output(data_i32.output(0))
        gnd.validate_and_infer_types()
        back_to_bool = ops.convert(gnd.output(0), destination_type="boolean")
        for target_input in consumers:
            target_input.replace_source_output(back_to_bool.output(0))
        rewritten += 1

    if rewritten == 0:
        return 0

    model.validate_nodes_and_infer_types()

    # Write to a temp path first, drop the mmap on the source ``.bin`` (``del``),
    # then atomically replace the originals. The ``.bin`` is located by the
    # ``.xml`` basename, so both must be renamed together.
    # ``compress_to_fp16=False`` preserves the exported weight precision as-is.
    tmp_xml = xml_path.with_suffix(".gpu.xml")
    tmp_bin = tmp_xml.with_suffix(".bin")
    bin_path = xml_path.with_suffix(".bin")
    ov.save_model(model, str(tmp_xml), compress_to_fp16=False)
    del model  # release the mmap on the source .bin before overwriting it
    Path(tmp_xml).replace(xml_path)
    Path(tmp_bin).replace(bin_path)

    return rewritten
