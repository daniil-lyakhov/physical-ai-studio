# XR0 fine-tuning: divergences from the upstream recipe & convergence notes

Context: fine-tuning XR0 from a pretrained Xiaomi checkpoint
(`Xiaomi-Robotics-0-LIBERO` / `Xiaomi-Robotics-0-Pretrain`) on a **single-arm
SO-101** LeRobot dataset whose action/state are **absolute joint positions in
degrees** (e.g. `Put-the-yellow-ball-to-the-black-box`).

The upstream repo (`Xiaomi-Robotics-0`, `xr0/mibot`) only ships a **bimanual**,
**delta / relative-EEF**, **raw radian-scale-state** recipe (single data config
`earphone.yaml`). There is **no SO-100/SO-101 recipe upstream**, so the Studio
side defines the embodiment contract itself. Three divergences follow from that.

## Divergence 1 — 32-dim slot layout

- Upstream `compose_state` / `ACTION_PARTS` (`xr0/mibot/utils/io.py`) place
  proprio/action into **fixed structured slots** (state: `6` gripper, `7-12`
  joint, `20`, `21-26`; slots `0-5` are always zero). Action slots: `0-2`
  `left_ee_pos`, `3-5` `left_ee_aa`, `6` gripper, `7-12` joint, ...
- Studio `XR0Preprocessor._prepare_state` / `_prepare_action` **left-pack** the
  user's dims into slots `0..D-1`.

**Impact on convergence:** not fatal for fine-tuning. Because the same
preprocessor is used at train / `select_action` / export, the layout only needs
to be self-consistent, and the `state_projector` is retrained. It would only
matter for zero-shot reuse of a frozen projector. **No action required.**

## Divergence 2 — state units / scale (ROOT CAUSE of the loss explosion)

- Upstream feeds **raw** state, which works only because its joints are
  ~radian-scale (O(1-3)). This dataset's state is **joint positions in degrees**
  (O(100), per-dim std ~4-48).
- Fed unnormalized into the pretrained `state_projector` + DiT AdaLN modulation
  (16 layers), the out-of-scale state blows up the conditioning -> `pred` ~1e3-1e4
  while the (normalized) target is O(1) -> masked MSE in the **hundreds of
  millions**.

**Status: FIXED.** `XR0Preprocessor` now supports opt-in state normalization
(`normalize_state`, identity by default). Enabled via `XR0(normalize_state=True)`
in `train_local_xr0.py`, it maps state to zero-mean/unit-std (O(1)), matching
what the conditioning expects. The stats are persisted with the checkpoint and
baked into the exported manifest (`XR0InferencePreprocessor`), so train /
benchmark / export stay in lockstep. Raw-state checkpoints (LIBERO/Pretrain) are
unaffected because the default is identity.

**Expected effect:** first-batch loss should start in a sane range (single/double
digits) instead of ~1e8.

## Divergence 3 — action representation

- Upstream actions are **delta / relative-EEF** (axis-angle rotation + delta
  joints, small magnitude). This dataset uses **absolute joint positions**.
- The action is always normalized by the dataset's own mean/std, so the flow
  **target stays O(1)** regardless of representation. Therefore this divergence
  **does not cause the loss explosion.**

**Impact on convergence:** quality/speed only. The pretrained flow head's prior
is tuned for deltas; on absolute joints it must relearn its output semantics
during fine-tuning. Options:

1. **Absolute + normalized (current, recommended).** Simplest; for
   single-embodiment fine-tuning the head adapts. No code changes.
2. **Convert to deltas** to better reuse the pretrained prior (closer to
   upstream, potentially faster convergence). Bigger change: needs a delta
   transform in the preprocessor **and** its inverse at inference, and it
   changes the action contract for benchmarking/export. Only worth it if
   convergence stalls.

## Recommended path & what to watch

- Proceed with **absolute joints + `normalize_state=True`** (Divergence 2 fixed;
  1 and 3 are representation choices, not loss bugs).
- On the next run, confirm scale is sane on the first batch:
  `state.abs().max()` (~O(1) after norm), `pred.abs().max()` and
  `target.abs().max()` (~O(1)), and the `loss_mse` vs `loss_freq` split.
- If loss is sane but convergence later stalls, the **delta-action conversion**
  (Divergence 3) is the next lever.

## Unrelated memory / throughput note

The ~130 GB memory usage seen earlier is a separate issue: gradient
checkpointing is wired only to the vision tower
(`self.vlm.model.visual.gradient_checkpointing_enable()`), not the Qwen3 LM, so
LM activations are retained. Not a convergence problem, but relevant to fitting
larger batches.
