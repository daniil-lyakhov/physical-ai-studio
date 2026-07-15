# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Single-implementation runner for the XR0 *training-iteration* parity test.

Companion to :mod:`runner` (which checks fp32 *inference*). Executed once per
environment (each repo's ``env``) as a subprocess:

* ``--impl source``    -> ``mibot.models.VLA.XR0.XR0`` (transformers 4.57.1).
* ``--impl framework`` -> ``physicalai`` ``XR0Model`` (transformers 5.3.0).

Both load the *same* LIBERO checkpoint and the *same* synthetic inputs, run one
full training step -- forward flow-matching loss then ``loss.backward()`` -- and
save the loss components, the predicted velocity / target, and the resulting
gradients so the orchestrating test can diff a whole training iteration between
the two implementations.

Determinism across the two environments (whose different torch versions cannot
be seeded to draw identical samples) is achieved by pinning *every* random draw
to tensors baked into ``inputs.pt``:

* the rectified-flow ``noise`` -> injected via ``torch.randn_like`` (reused from
  :mod:`runner`),
* the rectified-flow ``timestep`` -> injected by monkeypatching the model's
  ``_sample_timestep``.

The remaining training-time randomness (async prefix conditioning, prefix
masking, the ``training_repeat`` batch expansion) is disabled by constructing
both models with ``async_train=False`` and ``training_repeat=1`` so the step is
a single deterministic prefix-free flow-matching update, and the DiT runs with
its default zero dropout. Everything runs in float32 / eager attention on CPU
(the source additionally restores its rotary ``inv_freq`` to full float32 -- see
:func:`runner._restore_rotary_fp32`).

The artifact is keyed by *framework-canonical* parameter names (source flat head
names are nested under ``flow.``) so the test can compare like-for-like.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable

import torch

# Reuse the inference runner's helpers (module import is safe under either env:
# ``runner`` only imports the stdlib + torch at module scope).
from runner import (
    _DTYPE,
    _install_fixed_noise,
    _load_safetensors_dir,
    _restore_rotary_fp32,
)

# Top-level submodules owned by the flow action-expert. On the source ``XR0``
# they are flat (``dit.*``, ``state_projector.*``, ...); the framework nests them
# under ``flow.``. Kept as a local literal so this module never imports
# ``physicalai`` in the source environment. Mirrors
# ``physicalai.policies.xr0.pretrained_utils._FLOW_SUBMODULES``.
_FLOW_SUBMODULES = frozenset(
    {
        "dit",
        "state_projector",
        "action_projector",
        "action_output_layer",
        "t_embedder",
        "t_projector",
        "sink",
    },
)

# Full per-element gradients are saved only for flow-head params at or below this
# size (the large DiT / t_projector weight matrices are covered by grad_stats).
_FLOW_GRAD_MAX_NUMEL = 2_000_000


def _build_train_batch(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Assemble the float32 training batch (VLM inputs + state + target action)."""
    return {
        "input_ids": inputs["input_ids"].to(torch.long),
        "attention_mask": inputs["attention_mask"].to(torch.long),
        "pixel_values": inputs["pixel_values"].to(_DTYPE),
        "image_grid_thw": inputs["image_grid_thw"].to(torch.long),
        "state": inputs["state"].to(_DTYPE),
        # The flow regresses toward the ground-truth action target.
        "action": inputs["action_target"].to(_DTYPE),
        "action_mask": inputs["action_mask"].to(torch.int32),
    }


def _install_fixed_timestep(owner: object, timestep: torch.Tensor) -> None:
    """Pin ``owner._sample_timestep`` to return the baked-in ``timestep``.

    Both implementations call ``_sample_timestep(batch_size, dtype=..., device=...)``;
    the source on the ``XR0`` module itself, the framework on its ``flow``
    submodule. Replacing it with a constant removes the sole training-time
    stochastic draw the two torch versions cannot align.
    """

    def patched(
        batch_size: int,
        dtype: torch.dtype = _DTYPE,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        return timestep[:batch_size].to(dtype=dtype, device=device)

    owner._sample_timestep = patched  # type: ignore[attr-defined]


def _install_pred_capture(model: object, method_name: str, store: dict[str, torch.Tensor]) -> None:
    """Wrap the loss method to capture its ``(pred, target)`` before reduction."""
    original: Callable[..., Any] = getattr(model, method_name)

    def wrapper(
        pred: torch.Tensor,
        target: torch.Tensor,
        action_mask: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> Any:  # noqa: ANN401
        store["pred"] = pred.detach().to(torch.float32).cpu()
        store["target"] = target.detach().to(torch.float32).cpu()
        return original(pred, target, action_mask, weight)

    setattr(model, method_name, wrapper)


def _to_framework_name(name: str) -> str:
    """Map a source flat parameter name into the framework namespace."""
    head = name.split(".", 1)[0]
    if head in _FLOW_SUBMODULES:
        return f"flow.{name}"
    return name


def _collect_gradients(
    model: torch.nn.Module, rename: Callable[[str], str]
) -> tuple[dict[str, list[float]], dict[str, torch.Tensor]]:
    """Gather per-parameter gradient stats (all) and full grads (small flow heads).

    Returns:
        Tuple of ``(grad_stats, flow_grads)`` keyed by framework-canonical name.
        ``grad_stats`` maps every grad-bearing param to ``[norm, mean, absmax]``;
        ``flow_grads`` holds the full float32 gradient for each small
        (``<= _FLOW_GRAD_MAX_NUMEL``) ``flow.*`` head param.
    """
    grad_stats: dict[str, list[float]] = {}
    flow_grads: dict[str, torch.Tensor] = {}
    for raw_name, param in model.named_parameters():
        if param.grad is None:
            continue
        name = rename(raw_name)
        grad = param.grad.detach().float()
        grad_stats[name] = [grad.norm().item(), grad.mean().item(), grad.abs().max().item()]
        if name.startswith("flow.") and grad.numel() <= _FLOW_GRAD_MAX_NUMEL:
            flow_grads[name] = grad.cpu()
    return grad_stats, flow_grads


def run_framework(
    checkpoint: str, batch: dict[str, torch.Tensor], timestep: torch.Tensor
) -> dict[str, Any]:
    """Build the framework ``XR0Model``, run one training step, and collect grads."""
    from physicalai.policies.xr0.pretrained_utils import load_xr0_pretrained_weights
    from physicalai.policies.xr0.vla import XR0Model

    model = XR0Model(
        vlm_model_id="Qwen/Qwen3-VL-4B-Instruct",
        vlm_attn_implementation="eager",
        dtype=_DTYPE,
        async_train=False,
        training_repeat=1,
        enable_freq=False,
    )
    state_dict = load_xr0_pretrained_weights(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    print(f"[framework] missing={len(missing)} unexpected={len(unexpected)}")
    model.to(_DTYPE).train()
    _install_fixed_timestep(model.flow, timestep)

    store: dict[str, torch.Tensor] = {}
    _install_pred_capture(model, "_flow_loss", store)

    loss_dict = model._run(batch, return_loss=True)  # noqa: SLF001
    loss_dict["loss"].backward()

    grad_stats, flow_grads = _collect_gradients(model, rename=lambda name: name)
    return _assemble_artifact(loss_dict, store, grad_stats, flow_grads)


def run_source(
    checkpoint: str, batch: dict[str, torch.Tensor], timestep: torch.Tensor
) -> dict[str, Any]:
    """Build the source ``XR0``, run one training step, and collect grads.

    The source is hard-wired to bfloat16, so this forces float32 the same four
    ways as :func:`runner.run_source` (``model.float()``,
    ``t_embedder.dtype``, :func:`runner._restore_rotary_fp32`, and bypassing the
    ``@auto_cast`` ``forward`` decorator) plus eager attention for CPU.
    """
    import mibot.models.VLA.XR0 as xr0_module

    original_from_pretrained = xr0_module.Qwen3VLForConditionalGeneration.from_pretrained

    def eager_from_pretrained(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        kwargs["attn_implementation"] = "eager"
        return original_from_pretrained(*args, **kwargs)

    xr0_module.Qwen3VLForConditionalGeneration.from_pretrained = eager_from_pretrained

    model = xr0_module.XR0(
        state_shape=(1, 32),
        action_shape=(30, 32),
        dit_num_layers=16,
        dit_hidden_size=1024,
        num_steps=5,
        async_train=False,
        training_repeat=1,
        enable_freq=False,
    )
    state_dict = _load_safetensors_dir(checkpoint)
    result = model.load_state_dict(state_dict, strict=False, assign=True)
    print(f"[source] missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")
    model.float().train()
    model.t_embedder.dtype = _DTYPE
    _restore_rotary_fp32(model)
    _install_fixed_timestep(model, timestep)

    store: dict[str, torch.Tensor] = {}
    _install_pred_capture(model, "compute_loss", store)

    # Bypass the @auto_cast decorator (it downcasts the float32 batch to bf16).
    raw_forward = type(model).forward.__wrapped__
    loss_dict = raw_forward(model, batch, return_loss=True)
    loss_dict["loss"].backward()

    grad_stats, flow_grads = _collect_gradients(model, rename=_to_framework_name)
    return _assemble_artifact(loss_dict, store, grad_stats, flow_grads)


def _assemble_artifact(
    loss_dict: dict[str, torch.Tensor],
    store: dict[str, torch.Tensor],
    grad_stats: dict[str, list[float]],
    flow_grads: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Pack the training-step outputs into the saved artifact dict."""
    return {
        "loss": float(loss_dict["loss"].detach()),
        "loss_mse": float(loss_dict["loss_mse"].detach()),
        "loss_freq": float(loss_dict["loss_freq"].detach()),
        "pred": store["pred"],
        "target": store["target"],
        "grad_stats": grad_stats,
        "flow_grads": flow_grads,
    }


def main() -> None:
    """CLI entry point: run one implementation's training step and save the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", required=True, choices=["source", "framework"])
    parser.add_argument("--inputs", required=True, help="Shared synthetic inputs .pt.")
    parser.add_argument("--checkpoint", required=True, help="Local checkpoint directory.")
    parser.add_argument("--output", required=True, help="Destination .pt for the training step.")
    args = parser.parse_args()

    torch.manual_seed(0)
    inputs = torch.load(args.inputs, map_location="cpu")
    _install_fixed_noise(inputs["noise"])
    batch = _build_train_batch(inputs)
    timestep = inputs["timestep"].to(_DTYPE)

    runner = run_source if args.impl == "source" else run_framework
    artifact = runner(args.checkpoint, batch, timestep)

    torch.save(artifact, args.output)
    print(
        f"[{args.impl}] saved loss={artifact['loss']:.6e} "
        f"pred {tuple(artifact['pred'].shape)} "
        f"grads={len(artifact['grad_stats'])} flow_grads={len(artifact['flow_grads'])} "
        f"-> {args.output}"
    )


if __name__ == "__main__":
    main()
