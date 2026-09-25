# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Per-timestep action normalization statistics for the XR0 policy.

XR0 normalizes its action chunk with per-timestep ``(chunk_size,
max_action_dim)`` statistics (the upstream ``validate_stats`` contract). The
dataset-level statistics published by LeRobot are per-dimension and therefore
cannot be used directly: :func:`compute_action_chunk_stats` accumulates the
statistics over the *chunked* targets the model actually sees.
"""

from __future__ import annotations

from typing import Any, Literal

import torch

from physicalai.data.constants import ACTION, STATE

_TEMPORAL_STATE_NDIM = 3
_BATCHED_ACTION_NDIM = 2


def _feature_tensor(value: Any) -> torch.Tensor:  # noqa: ANN401
    """Coerce an observation field (tensor or single-entry dict) to a tensor.

    Returns:
        The tensor held by ``value``.

    Raises:
        TypeError: If no tensor can be recovered from ``value``.
    """
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for sub in value.values():
            if isinstance(sub, torch.Tensor):
                return sub
    msg = f"expected a tensor (or dict containing one), got {type(value)!r}"
    raise TypeError(msg)


def compute_action_chunk_stats(
    datamodule: Any,  # noqa: ANN401
    *,
    chunk_size: int,
    action_dim: int,
    max_action_dim: int = 32,
    action_mode: Literal["absolute", "delta"] = "delta",
    max_batches: int | None = None,
    setup_stage: str | None = "fit",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-timestep action mean/std over a training dataset.

    Iterates the datamodule's train dataloader once and accumulates, per chunk
    position, the mean and std of the model's regression target: the raw action
    in ``action_mode="absolute"``, or ``action[t] - state`` (the current-frame
    state broadcast over the chunk) in ``action_mode="delta"``. Sums are kept in
    float64 and returned as ``(chunk_size, max_action_dim)`` buffers, padded with
    identity stats (mean ``0``, std ``1``) on the unused columns so the
    downstream normalization is a no-op there.

    The dataloader is used deliberately, so the statistics describe exactly what
    training consumes. This includes the chunks that LeRobot pads at the end of
    an episode (the trailing frames are clamped to the last one): those samples
    are part of the training distribution, so they are part of the statistics.

    ``datamodule`` must already be chunked for the policy -- call
    :func:`physicalai.train.utils.reformat_dataset_to_match_policy` first,
    otherwise the action field is a single ``(B, D)`` frame instead of a
    ``(B, chunk_size, D)`` chunk.

    Args:
        datamodule: A Lightning datamodule exposing ``train_dataloader()`` (and an
            optional ``setup`` method) yielding batched observations with ``state``
            and ``action`` fields.
        chunk_size: Number of action steps per chunk (the temporal dimension).
        action_dim: True (unpadded) action dimension of the dataset.
        max_action_dim: Padded action dimension of the returned buffers.
        action_mode: ``"delta"`` (default) subtracts the current state from the
            action target; ``"absolute"`` accumulates the raw action.
        max_batches: Optional cap on the number of batches to consume (for a quick
            estimate). ``None`` consumes the full train set.
        setup_stage: Stage passed to ``datamodule.setup(...)`` before iterating.
            Pass ``None`` to skip setup (e.g. when already set up).

    Returns:
        A ``(mean, std)`` tuple of ``(chunk_size, max_action_dim)`` float32 tensors.

    Raises:
        ValueError: If the dataset yields no samples, or an action chunk whose
            temporal dimension does not match ``chunk_size``.
    """
    if setup_stage is not None and hasattr(datamodule, "setup"):
        datamodule.setup(setup_stage)

    sum_1 = torch.zeros(chunk_size, action_dim, dtype=torch.float64)
    sum_2 = torch.zeros(chunk_size, action_dim, dtype=torch.float64)
    count = 0

    for index, batch in enumerate(datamodule.train_dataloader()):
        if max_batches is not None and index >= max_batches:
            break

        action_field = batch.action if hasattr(batch, "action") else batch[ACTION]
        action = _feature_tensor(action_field).to(torch.float64)
        if action.ndim == _BATCHED_ACTION_NDIM:  # (B, D) -> (B, 1, D)
            action = action.unsqueeze(1)
        if action.shape[1] != chunk_size:
            msg = (
                f"action chunk has temporal dim {action.shape[1]}, expected chunk_size={chunk_size}; "
                "call reformat_dataset_to_match_policy(policy, datamodule) first"
            )
            raise ValueError(msg)
        target = action[..., :action_dim]

        if action_mode == "delta":
            state_field = batch.state if hasattr(batch, "state") else batch[STATE]
            state = _feature_tensor(state_field).to(torch.float64)
            if state.ndim == _TEMPORAL_STATE_NDIM:  # (B, T, D) -> current (last) frame
                state = state[:, -1, :]
            # Not in-place: ``target`` is a view of the caller's action tensor.
            target = target - state[..., :action_dim].unsqueeze(1)  # noqa: PLR6104

        sum_1 += target.sum(dim=0)
        sum_2 += (target * target).sum(dim=0)
        count += target.shape[0]

    if count == 0:
        msg = "no samples found while computing action chunk stats"
        raise ValueError(msg)

    mean = sum_1 / count
    var = (sum_2 / count - mean * mean).clamp_min(0.0)
    std = var.sqrt()

    mean_full = torch.zeros(chunk_size, max_action_dim, dtype=torch.float32)
    std_full = torch.ones(chunk_size, max_action_dim, dtype=torch.float32)
    width = min(action_dim, max_action_dim)
    mean_full[:, :width] = mean[:, :width].to(torch.float32)
    std_full[:, :width] = std[:, :width].to(torch.float32)
    return mean_full, std_full
