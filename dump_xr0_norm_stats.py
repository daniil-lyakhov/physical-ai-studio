#!/usr/bin/env python
"""Dump the XR0 normalization stats baked at training time from a checkpoint.

Run this on the training server (in the physicalai env) against the Lightning
checkpoint you deployed. It reconstructs the *exact* ``state_mean`` /
``state_std`` / ``action_mean`` / ``action_std`` vectors the training-time
preprocessor used (same left-pack + padding as
``physicalai.policies.xr0.preprocessor.make_xr0_preprocessors``), so you can
paste them into a deployed ``manifest.json`` without re-exporting.

Usage
-----
Print the values (and a sanity summary):

    python dump_xr0_norm_stats.py --checkpoint experiments/xr0_Put-the-yellow-ball-to-the-black-box/checkpoints/last.ckpt

Also patch a manifest in place (writes a ``.bak`` first). Only do this if the
checkpoint was trained with ``normalize_state=True``:

    python dump_xr0_norm_stats.py \
        --checkpoint .../last.ckpt \
        --manifest xr0_ir/manifest.json \
        --patch-manifest

The script prints the state normalization exactly as the training preprocessor
would compute it (``normalize_state=True``). If the checkpoint carries no state
stats, the vectors stay identity (0/1) and a warning is emitted.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch


def _load_dataset_stats(checkpoint: Path) -> dict[str, dict[str, Any]]:
    """Load the training ``dataset_stats`` dict from a Lightning checkpoint."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters") or ckpt.get("hparams") or {}
    stats = hparams.get("dataset_stats")
    if not stats:
        msg = (
            f"No 'dataset_stats' found in {checkpoint} hyper_parameters. "
            f"Available keys: {sorted(hparams)}"
        )
        raise SystemExit(msg)
    return stats


def _real_dim(stats: dict[str, dict[str, Any]], kind: str, cap: int) -> int | None:
    """Recover the unpadded feature dim for ``state`` / ``action`` from the stats shape."""
    for key, stat in stats.items():
        is_action = "action" in key
        is_state = ("state" in key) and not is_action
        if (kind == "action" and is_action) or (kind == "state" and is_state):
            shape = stat.get("shape")
            if shape:
                return min(cap, int(shape[-1]))
    return None


def _summarize(name: str, mean: list[float], std: list[float], real_dim: int | None) -> None:
    """Print a compact sanity summary of a mean/std vector."""
    dim = real_dim if real_dim is not None else len(mean)
    real_mean = mean[:dim]
    real_std = std[:dim]
    print(f"  {name}: real_dim={dim}")
    print(f"    mean[:{dim}] = {[round(v, 4) for v in real_mean]}")
    print(f"    std [:{dim}] = {[round(v, 4) for v in real_std]}")
    if real_std:
        print(f"    std range = [{min(real_std):.4g}, {max(real_std):.4g}]")
        if max(real_std) < 1e-2:
            print("    !! WARNING: std looks tiny (<1e-2) -- likely the wrong (checkpoint/delta) stats.")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to the .ckpt you deployed.")
    parser.add_argument("--state-dim", type=int, default=32, help="Padded state dim (max_state_dim). Default 32.")
    parser.add_argument("--action-dim", type=int, default=32, help="Padded action dim (max_action_dim). Default 32.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional manifest.json to inspect/patch.")
    parser.add_argument(
        "--patch-manifest",
        action="store_true",
        help="Patch --manifest in place (sets normalize_state=true + state stats). Writes a .bak backup.",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    stats = _load_dataset_stats(args.checkpoint)

    print(f"Loaded dataset_stats from: {args.checkpoint}")
    print(f"  feature keys: {sorted(stats)}\n")

    # Reconstruct the *exact* training-time vectors via the library code path.
    # normalize_state=True so the state vectors reflect the fine-tuning stats
    # (left-packed into real dims, padding dims stay identity 0/1).
    from physicalai.policies.xr0.preprocessor import make_xr0_preprocessors  # noqa: PLC0415

    pre, post = make_xr0_preprocessors(
        max_state_dim=args.state_dim,
        max_action_dim=args.action_dim,
        stats=stats,
        normalize_state=True,
    )

    state_mean = pre.state_mean.tolist()
    state_std = pre.state_std.tolist()
    action_mean = post.action_mean.tolist()
    action_std = post.action_std.tolist()
    state_real_dim = _real_dim(stats, "state", args.state_dim)
    action_real_dim = post.action_dim

    if all(m == 0.0 for m in state_mean) and all(s == 1.0 for s in state_std):
        print(
            "!! WARNING: state stats are identity (0/1). The checkpoint carries no usable\n"
            "   state stats, so patching cannot help -- this checkpoint was NOT trained\n"
            "   with a normalized state, or the state key is missing from dataset_stats.\n"
        )

    print("Sanity summary (real dims only):")
    _summarize("state", state_mean, state_std, state_real_dim)
    _summarize("action", action_mean, action_std, action_real_dim)
    print()

    print("=== manifest 'xr0_normalize' fields (paste these) ===")
    print(json.dumps({"normalize_state": True, "state_mean": state_mean, "state_std": state_std}, indent=2))
    print()
    print("=== manifest 'xr0_denormalize' fields (for verification) ===")
    print(json.dumps({"action_mean": action_mean, "action_std": action_std}, indent=2))
    print()

    if args.manifest is not None:
        _handle_manifest(args, state_mean, state_std, action_mean, action_std)

    return 0


def _handle_manifest(
    args: argparse.Namespace,
    state_mean: list[float],
    state_std: list[float],
    action_mean: list[float],
    action_std: list[float],
) -> None:
    """Compare (and optionally patch) the manifest against the checkpoint stats."""
    manifest_path: Path = args.manifest
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    # Components live under manifest["model"] in current exports; fall back to
    # the top level for older/flat layouts.
    container = manifest.get("model", manifest)
    preprocessors = container.get("preprocessors", [])
    postprocessors = container.get("postprocessors", [])

    # The state-normalization fields live on the combined "xr0" preprocessor
    # (image + state), not a separate component. Match by the presence of the
    # "normalize_state" key, then by type == "xr0".
    norm_spec = next((c for c in preprocessors if "normalize_state" in c), None)
    if norm_spec is None:
        norm_spec = next((c for c in preprocessors if c.get("type") == "xr0"), None)
    denorm_spec = next((c for c in postprocessors if c.get("type") == "xr0_denormalize"), None)

    if norm_spec is None:
        print("!! No 'xr0' preprocessor component (with 'normalize_state') found in the manifest.")
    else:
        cur = norm_spec.get("normalize_state")
        print(f"manifest currently: normalize_state={cur}")
        if cur is False and any(s != 1.0 for s in state_std):
            print("   -> MISMATCH: checkpoint has non-identity state stats but manifest skips state normalization.")

    if denorm_spec is not None:
        if denorm_spec.get("action_std") == action_std and denorm_spec.get("action_mean") == action_mean:
            print("action stats: manifest MATCHES checkpoint. Good.")
        else:
            print("!! action stats: manifest DIFFERS from checkpoint -- denormalization may be wrong too.")

    if not args.patch_manifest:
        print("\n(Dry run -- pass --patch-manifest to write the state stats into the manifest.)")
        return

    if norm_spec is None:
        raise SystemExit("Cannot patch: no 'xr0' preprocessor component in the manifest.")

    backup = manifest_path.with_suffix(manifest_path.suffix + ".bak")
    shutil.copy2(manifest_path, backup)
    norm_spec["normalize_state"] = True
    norm_spec["state_mean"] = state_mean
    norm_spec["state_std"] = state_std
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nPatched {manifest_path} (backup at {backup}).")


if __name__ == "__main__":
    sys.exit(main())
