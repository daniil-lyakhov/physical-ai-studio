# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Single-implementation runner for the XR0 source-vs-framework parity test.

Executed once per environment (each repo's ``env``) as a subprocess:

* ``--impl source``    -> ``mibot.models.VLA.XR0.XR0`` (transformers 4.57.1).
* ``--impl framework`` -> ``physicalai`` ``XR0Model`` (transformers 5.3.0).

Both load the *same* LIBERO checkpoint and the *same* synthetic inputs, inject
the *same* pinned rectified-flow noise, run action generation in float32 / eager
attention on CPU, and save (as a dict) both the raw predicted action chunk and
the VLM key/value cache the DiT cross-attends to (tail layers) so the
orchestrating test can diff the VLM in isolation *and* the whole model output.

Running in float32 makes the comparison a genuine implementation-parity check:
the two VLMs are numerically identical in float32, so the whole-model float32
diff is the pure DiT / flow-head port difference (the VLM contributes ~0). The
source is hard-wired to bfloat16 (``_build_model`` ends with ``self.to(bf16)``,
``forward`` is ``@auto_cast`` which downcasts float32 batches, and
``TimestepEmbedder.dtype`` defaults to bf16), so the source branch overrides all
three to reach float32.

The implementation-specific imports live inside their branch so the script runs
under either environment (neither package is importable from the other's env).
"""

from __future__ import annotations

import argparse
import json
import os

import torch

# All parity is measured in float32: the two VLM ports are numerically identical
# at this precision, so any residual isolates the DiT / flow-head port.
_DTYPE = torch.float32

# Number of trailing VLM KV-cache layers to capture -- the DiT aligns its layers
# with the tail of the VLM cache, so these are the layers actually consumed.
_VLM_KV_LAYERS = 16


def _install_fixed_noise(noise: torch.Tensor) -> None:
    """Monkeypatch ``torch.randn_like`` to return ``noise`` for the action shape."""
    real_randn_like = torch.randn_like
    target_shape = tuple(noise.shape)

    def patched(tensor: torch.Tensor, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        if tuple(tensor.shape) == target_shape:
            return noise.to(dtype=tensor.dtype, device=tensor.device)
        return real_randn_like(tensor, *args, **kwargs)

    torch.randn_like = patched


def _build_batch(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Cast the shared inputs into the float32 model batch (VLM keys + state/action)."""
    return {
        "input_ids": inputs["input_ids"].to(torch.long),
        "attention_mask": inputs["attention_mask"].to(torch.long),
        "pixel_values": inputs["pixel_values"].to(_DTYPE),
        "image_grid_thw": inputs["image_grid_thw"].to(torch.long),
        "state": inputs["state"].to(_DTYPE),
        "action": inputs["action"].to(_DTYPE),
        "action_mask": inputs["action_mask"].to(torch.int32),
    }


def _load_safetensors_dir(checkpoint_dir: str) -> dict[str, torch.Tensor]:
    """Load a (possibly sharded) safetensors checkpoint from a local directory."""
    from safetensors.torch import load_file

    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        state_dict: dict[str, torch.Tensor] = {}
        for shard in sorted(set(weight_map.values())):
            state_dict.update(load_file(os.path.join(checkpoint_dir, shard)))
        return state_dict
    return load_file(os.path.join(checkpoint_dir, "model.safetensors"))


def _extract_kv_layers(
    past_key_values: object,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Normalize a KV cache into a per-layer list of ``(keys, values)`` tensors.

    Handles the transformers 5.x ``DynamicCache`` (``.layers`` with
    ``.keys``/``.values``), the 4.x ``.key_cache``/``.value_cache`` lists, and the
    legacy iterable-of-tuples form.
    """
    if hasattr(past_key_values, "layers"):
        return [(layer.keys, layer.values) for layer in past_key_values.layers]
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))
    return [(keys, values) for keys, values in past_key_values]  # type: ignore[union-attr]


def _run_with_vlm_capture(
    vlm: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    generate: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run ``generate`` while capturing the VLM's KV cache via a forward hook.

    Returns:
        Tuple of ``(prediction, vlm_keys, vlm_values)`` where the KV tensors are
        the last ``_VLM_KV_LAYERS`` layers stacked as ``(L, B, H, S, D)``.
    """
    captured: dict[str, object] = {}

    def hook(_module: object, _args: object, output: object) -> None:
        captured["past_key_values"] = output.past_key_values

    handle = vlm.register_forward_hook(hook)
    try:
        with torch.no_grad():
            prediction = generate(batch)  # type: ignore[operator]
    finally:
        handle.remove()

    layers = _extract_kv_layers(captured["past_key_values"])[-_VLM_KV_LAYERS:]
    keys = torch.stack([layer_keys for layer_keys, _ in layers], dim=0)
    values = torch.stack([layer_values for _, layer_values in layers], dim=0)
    return prediction, keys, values


def run_framework(
    checkpoint: str, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the framework ``XR0Model``, load the checkpoint, and generate actions."""
    from physicalai.policies.xr0.pretrained_utils import load_xr0_pretrained_weights
    from physicalai.policies.xr0.vla import XR0Model

    model = XR0Model(
        vlm_model_id="Qwen/Qwen3-VL-4B-Instruct",
        vlm_attn_implementation="eager",
        dtype=_DTYPE,
    )
    state_dict = load_xr0_pretrained_weights(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    print(f"[framework] missing={len(missing)} unexpected={len(unexpected)}")
    model.to(_DTYPE).eval()
    return _run_with_vlm_capture(model.vlm, batch, model)


def _restore_rotary_fp32(model: torch.nn.Module) -> None:
    """Recompute every rotary ``inv_freq`` buffer at full float32.

    ``XR0._build_model`` ends with ``self.to(torch.bfloat16)``, which quantizes
    the non-persistent ``inv_freq`` buffers of the rotary embeddings to
    bfloat16. A later ``model.float()`` widens the dtype back to float32 but
    cannot restore the lost mantissa bits, so the frequencies stay ~9e-4 off
    their true values. Rebuilding them from each module's own ``rope_init_fn``
    (or ``compute_default_rope_parameters``) and ``config`` yields the same
    full-precision ``inv_freq`` the framework uses, making the fp32 comparison a
    genuine algorithm-vs-algorithm check instead of an fp32-vs-bf16 one.
    """
    for module in model.modules():
        if type(module).__name__ not in {
            "Qwen3VLTextRotaryEmbedding",
            "Qwen3VLVisionRotaryEmbedding",
        }:
            continue
        init_fn = getattr(module, "rope_init_fn", None) or getattr(
            module, "compute_default_rope_parameters", None
        )
        inv_freq = getattr(module, "inv_freq", None)
        if init_fn is None or inv_freq is None or not hasattr(module, "config"):
            continue
        fresh, _scaling = init_fn(module.config, device=inv_freq.device)
        inv_freq.copy_(fresh.to(inv_freq.dtype))
        if isinstance(getattr(module, "original_inv_freq", None), torch.Tensor):
            module.original_inv_freq = inv_freq.clone()


def run_source(
    checkpoint: str, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the source ``XR0`` (eager attention), load the checkpoint, and generate.

    The source is hard-wired to bfloat16, so this forces float32 four ways:
    ``model.float()`` (overrides the ``self.to(bf16)`` in ``_build_model``),
    ``t_embedder.dtype = float32`` (its ``timestep_embedding`` casts to that
    attribute), recomputing each rotary-embedding ``inv_freq`` buffer at full
    float32 (``_build_model``'s ``self.to(bf16)`` quantizes it to bfloat16, and
    the later ``model.float()`` cannot recover the lost bits -- leaving the
    RoPE frequencies ~9e-4 off and skewing the fp32 key cache by up to ~0.2),
    and calling ``forward.__wrapped__`` to bypass the ``@auto_cast`` decorator
    that would otherwise downcast the float32 batch back to bf16.
    """
    import mibot.models.VLA.XR0 as xr0_module

    # Force eager attention: the source hard-codes flash_attention_2, which is
    # unavailable on CPU.
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
    )
    state_dict = _load_safetensors_dir(checkpoint)
    result = model.load_state_dict(state_dict, strict=False, assign=True)
    print(f"[source] missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")
    model.float().eval()
    model.t_embedder.dtype = _DTYPE
    _restore_rotary_fp32(model)

    # Bypass the @auto_cast decorator (it downcasts the float32 batch to bf16).
    raw_forward = type(model).forward.__wrapped__

    def generate(model_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return raw_forward(model, model_batch, return_loss=False)

    return _run_with_vlm_capture(model.vlm, batch, generate)


def main() -> None:
    """CLI entry point: run one implementation and save its action chunk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", required=True, choices=["source", "framework"])
    parser.add_argument("--inputs", required=True, help="Shared synthetic inputs .pt.")
    parser.add_argument("--checkpoint", required=True, help="Local checkpoint directory.")
    parser.add_argument("--output", required=True, help="Destination .pt for the action chunk.")
    args = parser.parse_args()

    torch.manual_seed(0)
    inputs = torch.load(args.inputs, map_location="cpu")
    _install_fixed_noise(inputs["noise"])
    batch = _build_batch(inputs)

    runner = run_source if args.impl == "source" else run_framework
    prediction, vlm_keys, vlm_values = runner(args.checkpoint, batch)

    artifact = {
        "action": prediction.detach().to(torch.float32).cpu(),
        "vlm_keys": vlm_keys.detach().to(torch.float32).cpu(),
        "vlm_values": vlm_values.detach().to(torch.float32).cpu(),
    }
    torch.save(artifact, args.output)
    print(
        f"[{args.impl}] saved action {tuple(artifact['action'].shape)} "
        f"vlm_keys {tuple(artifact['vlm_keys'].shape)} -> {args.output}"
    )


if __name__ == "__main__":
    main()
