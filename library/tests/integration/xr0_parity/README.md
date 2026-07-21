# XR0 source-vs-framework parity test

End-to-end numerical parity check between the **source** XR0 (Xiaomi's `mibot`,
transformers 4.57.1) and the **framework** XR0 (`physicalai`, transformers
5.3.0) on the LIBERO checkpoint.

The two implementations pin incompatible `transformers` versions, so each runs
in its own repository `env` **as a subprocess**. Both consume an identical batch
of synthetic samples, load the same LIBERO checkpoint, inject the same pinned
per-sample rectified-flow noise, and emit the raw predicted action chunks _plus_
the VLM key/value cache the DiT cross-attends to.

Everything runs in **float32**. At this precision the two Qwen3-VL ports are
numerically equivalent, so the comparison splits cleanly:

- the **VLM KV cache** diff is the VLM-port residual (rel mean < 0.1 %),
  confirming the vendored VLM shim matches the source;
- the **whole-model action** diff is therefore the DiT / flow-head residual,
  since the VLM feeds both DiTs numerically-identical inputs.

Two test suites live here:

- **inference parity** (`test_source_parity.py`) — diffs the generated action
  chunk and the VLM KV cache (fp32, no gradients);
- **training parity** (`test_training_parity.py`) — diffs one full training
  step: the flow-matching loss, the predicted velocity, and the gradients from
  `loss.backward()`.

## Files

| File                      | Purpose                                                                                                                                                                                |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `build_inputs.py`         | Builds one shared `inputs.pt` (deterministic synthetic VLM inputs + state + pinned noise; plus `action_target` / `timestep` for the training step) consumed unchanged by both runners. |
| `runner.py`               | Inference: runs a single implementation (`--impl source\|framework`) and saves `{action, vlm_keys, vlm_values}`.                                                                       |
| `test_source_parity.py`   | Orchestrates both inference runners and asserts the parity thresholds.                                                                                                                 |
| `training_runner.py`      | Training: runs one forward+backward step for a single implementation and saves `{loss*, pred, target, grad_stats, flow_grads}`.                                                        |
| `test_training_parity.py` | Orchestrates both training runners and asserts loss / velocity / gradient parity.                                                                                                      |

## Running the pytest

Opt-in and heavy (loads Qwen3-VL-4B ~8 GB twice on CPU). Requires `XR0_PARITY=1`,
both env interpreters present, and the checkpoint cached; otherwise it skips.

```bash
cd /home/dlyakhov/Projects/physical-ai-studio
XR0_PARITY=1 HF_HUB_OFFLINE=1 env/bin/python -m pytest \
    library/tests/integration/xr0_parity/test_source_parity.py -m slow -s
```

For the **training-iteration** parity test, point pytest at the other file (same
env vars and skip guard):

```bash
cd /home/dlyakhov/Projects/physical-ai-studio
XR0_PARITY=1 HF_HUB_OFFLINE=1 env/bin/python -m pytest \
    library/tests/integration/xr0_parity/test_training_parity.py -m slow -s
```

Useful environment variables:

| Variable               | Default                                   | Meaning                           |
| ---------------------- | ----------------------------------------- | --------------------------------- |
| `XR0_PARITY`           | (unset)                                   | Must be `1` to enable the test.   |
| `XR0_PARITY_SAMPLES`   | `8`                                       | Number of synthetic samples.      |
| `XR0_FRAMEWORK_PYTHON` | `env/bin/python`                          | Framework interpreter (tf 5.3.0). |
| `XR0_SOURCE_PYTHON`    | `../Xiaomi-Robotics-0/xr0/env/bin/python` | Source interpreter (tf 4.57.1).   |

## Running the runners manually

Use this to inspect alignment directly. The two implementations cannot share a
process (incompatible `transformers`), so run each in its own env, then diff the
artifacts.

```bash
cd /home/dlyakhov/Projects/physical-ai-studio

# Resolve paths once
export CKPT=$(ls -d ~/.cache/huggingface/hub/models--XiaomiRobotics--Xiaomi-Robotics-0-LIBERO/snapshots/*/)
export SRC_PY=/home/dlyakhov/Projects/Xiaomi-Robotics-0/xr0/env/bin/python

# Step 0: build the shared inputs.pt once (4 samples is enough for a manual check)
HF_HUB_OFFLINE=1 env/bin/python library/tests/integration/xr0_parity/build_inputs.py \
    --out /tmp/xr0_inputs.pt --num-samples 4
```

**1. Framework runner** (this repo's env, transformers 5.3.0):

```bash
HF_HUB_OFFLINE=1 env/bin/python library/tests/integration/xr0_parity/runner.py \
    --impl framework \
    --inputs /tmp/xr0_inputs.pt \
    --checkpoint "$CKPT" \
    --output /tmp/xr0_framework.pt
```

**2. Source runner** (mibot env, transformers 4.57.1):

```bash
HF_HUB_OFFLINE=1 "$SRC_PY" library/tests/integration/xr0_parity/runner.py \
    --impl source \
    --inputs /tmp/xr0_inputs.pt \
    --checkpoint "$CKPT" \
    --output /tmp/xr0_source.pt
```

**Compare the two artifacts** (`action`, `vlm_keys`, `vlm_values`):

```bash
env/bin/python - <<'PY'
import torch
fw = torch.load("/tmp/xr0_framework.pt"); sr = torch.load("/tmp/xr0_source.pt")
for k in ("action", "vlm_keys", "vlm_values"):
    a, b = fw[k], sr[k]
    d = (a - b).abs()
    mag = b.abs().clamp_min(1e-6)
    print(f"{k:11s} shape={tuple(a.shape)} max_abs={d.max():.4e} "
          f"mean_abs={d.mean():.4e} rel_mean={d.mean()/mag.mean():.4%}")
af, as_ = fw["action"].flatten(1), sr["action"].flatten(1)
cos = torch.nn.functional.cosine_similarity(af, as_, dim=1)
print(f"action per-sample cosine: min={cos.min():.6f} mean={cos.mean():.6f}")
PY
```

Notes:

- `env/bin/python` is the framework interpreter (tf 5.3.0); `$SRC_PY` is the
  mibot interpreter (tf 4.57.1).
- Both runners force fp32 + eager attention on CPU and inject the same pinned
  noise, so the only difference measured is the two model implementations.
- Drop `HF_HUB_OFFLINE=1` if the Qwen3-VL-4B processor/weights are not cached yet
  and need downloading.

## Interpreting the numbers

Expected float32 agreement (8 samples):

| metric                         | typical value |
| ------------------------------ | ------------- |
| action per-sample cosine       | ~1.000000     |
| action mean abs diff           | ~2e-5         |
| action max abs diff            | ~8e-3         |
| vlm_keys / vlm_values rel mean | ~0.01–0.02 %  |

The residual VLM KV diff is dominated by large-magnitude "massive-activation"
channels (worst element ~0.3 % relative) and by cross-version fp32 attention /
vision-tower numerics accumulating through the 36-layer stack — not a port bug.
The **relative-mean** bound is the parity gate; the raw max-abs is only a
gross-regression ceiling.

> **float32 gotcha:** the source hard-wires bfloat16 (`XR0._build_model` ends
> with `self.to(torch.bfloat16)`), which quantizes the rotary `inv_freq` buffer.
> A later `model.float()` cannot recover those bits, leaving the RoPE
> frequencies ~9e-4 off and inflating the key-cache diff to ~0.2. The source
> runner therefore recomputes each rotary `inv_freq` at full float32
> (`_restore_rotary_fp32`) so the fp32 comparison is genuinely algorithm-vs-
> algorithm. This is a test-only correction; in a real bfloat16 deployment both
> sides quantize `inv_freq` identically and match.

## Training-iteration parity

`test_training_parity.py` / `training_runner.py` check a full **training step**
instead of inference: each runner sets the model to `train()`, runs the
flow-matching forward to get the loss, calls `loss.backward()`, and saves the
loss components, the predicted velocity / target, and the gradients.

Cross-environment determinism (the two torch versions cannot be seeded to draw
identical samples) is achieved by pinning **every** training-time random draw to
tensors baked into `inputs.pt`:

- the rectified-flow `noise` — injected via `torch.randn_like` (as in the
  inference runner);
- the rectified-flow `timestep` — injected by monkeypatching `_sample_timestep`.

Both models are built with `async_train=False`, `training_repeat=1` and
`enable_freq=False`, so the step is a single deterministic, prefix-free
flow-matching update with the DiT's default zero dropout — fully comparable.
Gradients are keyed by **framework-canonical** names (source flat head params
are nested under `flow.`) so like-for-like params line up.

### Running the training runners manually

```bash
cd /home/dlyakhov/Projects/physical-ai-studio
export CKPT=$(ls -d ~/.cache/huggingface/hub/models--XiaomiRobotics--Xiaomi-Robotics-0-LIBERO/snapshots/*/)
export SRC_PY=/home/dlyakhov/Projects/Xiaomi-Robotics-0/xr0/env/bin/python

# Shared inputs.pt (includes action_target + timestep used by the training step)
HF_HUB_OFFLINE=1 env/bin/python library/tests/integration/xr0_parity/build_inputs.py \
    --out /tmp/xr0_inputs.pt --num-samples 4
```

**1. Framework training runner** (this repo's env, transformers 5.3.0):

```bash
HF_HUB_OFFLINE=1 env/bin/python library/tests/integration/xr0_parity/training_runner.py \
    --impl framework --inputs /tmp/xr0_inputs.pt --checkpoint "$CKPT" \
    --output /tmp/xr0_framework_step.pt
```

**2. Source training runner** (mibot env, transformers 4.57.1):

```bash
HF_HUB_OFFLINE=1 "$SRC_PY" library/tests/integration/xr0_parity/training_runner.py \
    --impl source --inputs /tmp/xr0_inputs.pt --checkpoint "$CKPT" \
    --output /tmp/xr0_source_step.pt
```

**Compare the two training steps** (loss, predicted velocity, gradient norms):

```bash
env/bin/python - <<'PY'
import torch
fw = torch.load("/tmp/xr0_framework_step.pt"); sr = torch.load("/tmp/xr0_source_step.pt")
for k in ("loss", "loss_mse", "loss_freq"):
    a, b = fw[k], sr[k]
    denom = max(abs(a), abs(b), 1e-8)
    print(f"{k:9s} source={b:.6e} framework={a:.6e} rel={abs(a-b)/denom:.4e}")
d = (fw["pred"] - sr["pred"]).abs()
print(f"pred      rel_mean={d.mean()/sr['pred'].abs().mean():.4e} max_abs={d.max():.4e}")
shared = sorted(set(fw["grad_stats"]) & set(sr["grad_stats"]))
rel = torch.tensor([abs(fw["grad_stats"][n][0]-sr["grad_stats"][n][0])
                    / max(abs(sr["grad_stats"][n][0]), 1e-8) for n in shared])
print(f"grad-norm params={len(shared)} median={rel.median():.4e} "
      f"p99={torch.quantile(rel,0.99):.4e} max={rel.max():.4e}")
PY
```

### Interpreting the training numbers

Expected float32 agreement (8 samples):

| metric                                | typical value |
| ------------------------------------- | ------------- |
| loss / loss_mse rel diff              | ~2e-6         |
| predicted velocity rel mean           | ~3e-5         |
| predicted velocity max abs            | ~1e-3         |
| gradient-norm rel diff (median / p99) | ~1e-4 / ~7e-4 |
| flow-head grad rel mean (worst)       | ~7e-4         |

The loss and predicted velocity match to float32 rounding; gradients are looser
because each accumulates the forward residual through the whole backward graph
across two non-identical attention / vision kernels — still ~1e-4 relative, well
below a real port bug. Gradient norms are compared over the parameters present
(with a non-null grad) in **both** implementations; the source freezes the VLM
input embeddings and the framework may not, so that param is simply absent from
the shared set rather than a mismatch.
