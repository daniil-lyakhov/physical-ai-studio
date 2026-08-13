# Copyright (C) 2026 Xiaomi Corporation.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pure numpy/PIL adapters for the XR0 bimanual state/action space.

These are the dependency-free helpers that
compose/split the 32-dim bimanual state and action layout, convert delta actions to
absolute targets, normalize/denormalize actions, do axis-angle <-> rotation-matrix
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL import Image

ACTION_DIM = 32
STATE_DIM = 32
ACTION_EPS = 1e-6

# Reject images whose aspect ratio exceeds this (matches the source preprocessor).
_MAX_ASPECT_RATIO = 200
# Threshold below which an axis component is treated as zero when recovering the
# rotation axis of a near-180-degree rotation.
_AXIS_COMPONENT_EPS = 1e-8
# Minimum axis norm before it is considered degenerate and replaced by a default.
_MIN_AXIS_NORM = 1e-12
# Angle tolerance for the near-zero / near-pi rotation special cases.
_ANGLE_EPS = 1e-6

ACTION_PARTS = (
    ("left_ee_pos", slice(0, 3)),
    ("left_ee_aa", slice(3, 6)),
    ("left_gripper", slice(6, 7)),
    ("left_joint", slice(7, 13)),
    ("right_ee_pos", slice(14, 17)),
    ("right_ee_aa", slice(17, 20)),
    ("right_gripper", slice(20, 21)),
    ("right_joint", slice(21, 27)),
)


def get_value(data: object, path: str) -> object:
    """Look up a dotted ``path`` in a nested mapping, returning ``None`` if any key is missing.

    Returns:
        The value at ``path``, or ``None`` if any intermediate key is missing.
    """
    for key in path.split("."):
        if not isinstance(data, Mapping):
            return None
        data = data.get(key)
        if data is None:
            return None
    return data


def resize_image(
    image: Image.Image,
    factor: int = 32,
    min_pixels: int = 32 * 32,
    max_pixels: int = 90000,
) -> Image.Image:
    """Resize a PIL image to patch-aligned dimensions within an area budget.

    Both sides are rounded to multiples of ``factor`` and the area is kept within
    ``[min_pixels, max_pixels]``, preserving aspect ratio for the VLM vision encoder.

    Returns:
        The resized PIL image.

    Raises:
        ValueError: If the image aspect ratio exceeds ``_MAX_ASPECT_RATIO``.
    """
    width, height = image.size
    ratio = max(height, width) / min(height, width)
    if ratio > _MAX_ASPECT_RATIO:
        msg = f"absolute aspect ratio must be smaller than 200, got {ratio}"
        raise ValueError(msg)

    new_height = max(factor, round(height / factor) * factor)
    new_width = max(factor, round(width / factor) * factor)

    if new_height * new_width > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        new_height = max(factor, math.floor(height / scale / factor) * factor)
        new_width = max(factor, math.floor(width / scale / factor) * factor)
    elif new_height * new_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        new_height = max(factor, math.ceil(height * scale / factor) * factor)
        new_width = max(factor, math.ceil(width * scale / factor) * factor)

    return image.resize((new_width, new_height))


def build_pixel_grid(
    images: Sequence[Image.Image | np.ndarray],
    image_mean: Sequence[float],
    image_std: Sequence[float],
    rescale_factor: float,
) -> np.ndarray:
    """Rescale + normalize already-resized images into a Qwen3-VL pixel grid.

    Reproduces the Qwen3-VL image processor's rescale + normalize + channel-first
    steps in pure NumPy (the images must already be resized to patch-aligned
    dimensions) and stacks the views into the ``(num_images, C, H, W)`` normalized
    grid the exported graph patchifies. This is the NumPy replacement for calling
    the HuggingFace image processor and inverting its patchify.

    Args:
        images: Already-resized RGB images (PIL images or ``(H, W, C)`` arrays),
            one per camera view, all the same size.
        image_mean: Per-channel mean (the image processor's ``image_mean``).
        image_std: Per-channel std (the image processor's ``image_std``).
        rescale_factor: Pixel rescale factor (``1/255`` for Qwen3-VL).

    Returns:
        The normalized image grid of shape ``(num_images, C, H, W)`` as float32.
    """
    mean = np.asarray(image_mean, dtype=np.float32)
    std = np.asarray(image_std, dtype=np.float32)
    grid = [
        np.transpose((np.asarray(image, dtype=np.float32) * np.float32(rescale_factor) - mean) / std, (2, 0, 1))
        for image in images
    ]
    return np.stack(grid).astype(np.float32)


def _axis_from_pi(rotm: np.ndarray) -> np.ndarray:
    """Recover the unit rotation axis for a near-180-degree rotation.

    The standard skew-symmetric formula is degenerate at 180 degrees; the axis is
    instead picked from the largest diagonal term of the rotation matrix.

    Returns:
        The recovered unit rotation axis of shape ``(3,)``.
    """
    rot00, rot11, rot22 = rotm[0, 0], rotm[1, 1], rotm[2, 2]

    if rot00 >= rot11 and rot00 >= rot22:
        axis = np.array([np.sqrt(max((rot00 + 1.0) / 2.0, 0.0)), 0.0, 0.0], dtype=np.float32)
        if axis[0] > _AXIS_COMPONENT_EPS:
            axis[1] = rotm[0, 1] / (2.0 * axis[0])
            axis[2] = rotm[0, 2] / (2.0 * axis[0])
    elif rot11 >= rot22:
        axis = np.array([0.0, np.sqrt(max((rot11 + 1.0) / 2.0, 0.0)), 0.0], dtype=np.float32)
        if axis[1] > _AXIS_COMPONENT_EPS:
            axis[0] = rotm[0, 1] / (2.0 * axis[1])
            axis[2] = rotm[1, 2] / (2.0 * axis[1])
    else:
        axis = np.array([0.0, 0.0, np.sqrt(max((rot22 + 1.0) / 2.0, 0.0))], dtype=np.float32)
        if axis[2] > _AXIS_COMPONENT_EPS:
            axis[0] = rotm[0, 2] / (2.0 * axis[2])
            axis[1] = rotm[1, 2] / (2.0 * axis[2])

    norm = np.linalg.norm(axis)
    if norm < _MIN_AXIS_NORM:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return axis / norm


def rotm2aa_batch(rotms: np.ndarray) -> np.ndarray:
    """Convert a batch of ``(N,3,3)`` rotation matrices to ``(N,3)`` axis-angle vectors.

    Handles the near-zero and near-pi angle special cases.

    Returns:
        The ``(N, 3)`` axis-angle vectors.
    """
    rotms = np.asarray(rotms, dtype=np.float32)
    theta = np.arccos(np.clip((np.einsum("nii->n", rotms) - 1.0) / 2.0, -1.0, 1.0))

    axis_angle = np.zeros((rotms.shape[0], 3), dtype=np.float32)
    near_zero = theta <= _ANGLE_EPS
    near_pi = np.abs(theta - np.pi) <= _ANGLE_EPS
    normal = ~(near_zero | near_pi)

    if np.any(normal):
        axis = np.stack(
            [
                rotms[:, 2, 1] - rotms[:, 1, 2],
                rotms[:, 0, 2] - rotms[:, 2, 0],
                rotms[:, 1, 0] - rotms[:, 0, 1],
            ],
            axis=1,
        )
        axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-12
        axis_angle[normal] = axis[normal] * theta[normal, None]

    if np.any(near_pi):
        for i in np.where(near_pi)[0]:
            axis_angle[i] = _axis_from_pi(rotms[i]) * theta[i]

    return axis_angle


def aa2rotm(axis_angle: np.ndarray) -> np.ndarray:
    """Convert a single axis-angle vector to a ``(3,3)`` rotation matrix via Rodrigues' formula.

    Returns:
        The ``(3, 3)`` rotation matrix.
    """
    axis_angle = np.asarray(axis_angle, dtype=np.float32)
    angle = float(np.linalg.norm(axis_angle))
    axis = axis_angle / (angle + 1e-10)
    x, y, z = axis.tolist()
    axis_hat = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float32)
    eye = np.identity(3, dtype=np.float32)
    return eye + np.sin(angle) * axis_hat + (1.0 - np.cos(angle)) * axis_hat @ axis_hat


def validate_stats(
    mean: Sequence[Sequence[float]],
    std: Sequence[Sequence[float]],
    action_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce ``mean``/``std`` to float32 arrays and assert they match ``(action_length, ACTION_DIM)``.

    Returns:
        The ``(mean, std)`` float32 arrays.

    Raises:
        ValueError: If ``mean`` or ``std`` does not have shape ``(action_length, ACTION_DIM)``.
    """
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    if mean_arr.shape != (action_length, ACTION_DIM):
        msg = f"mean expected shape {(action_length, ACTION_DIM)}, got {mean_arr.shape}"
        raise ValueError(msg)
    if std_arr.shape != (action_length, ACTION_DIM):
        msg = f"std expected shape {(action_length, ACTION_DIM)}, got {std_arr.shape}"
        raise ValueError(msg)
    return mean_arr, std_arr


def build_action_mask(action_length: int, temporal_mask: np.ndarray | None = None) -> np.ndarray:
    """Build the ``(action_length, ACTION_DIM)`` binary mask marking valid bimanual columns.

    The mask is broadcast over an optional per-timestep ``temporal_mask`` (all ones
    if omitted).

    Returns:
        The ``(action_length, ACTION_DIM)`` binary column mask.
    """
    temporal = (
        np.ones(action_length, dtype=np.int32) if temporal_mask is None else np.asarray(temporal_mask, dtype=np.int32)
    )
    mask = np.zeros((action_length, ACTION_DIM), dtype=np.int32)
    for _, slc in ACTION_PARTS:
        mask[:, slc] = temporal[:, None]
    return mask


def compose_action(
    left_ee_pos: np.ndarray,
    left_ee_aa: np.ndarray,
    left_gripper: np.ndarray,
    left_joint: np.ndarray,
    right_ee_pos: np.ndarray,
    right_ee_aa: np.ndarray,
    right_gripper: np.ndarray,
    right_joint: np.ndarray,
    action_length: int,
) -> np.ndarray:
    """Pack the eight named bimanual parts into a single action array.

    The parts are placed following the fixed ``(action_length, ACTION_DIM)`` column
    layout; padding columns stay zero.

    Returns:
        The packed ``(action_length, ACTION_DIM)`` action array.
    """
    values = (left_ee_pos, left_ee_aa, left_gripper, left_joint, right_ee_pos, right_ee_aa, right_gripper, right_joint)
    action = np.zeros((action_length, ACTION_DIM), dtype=np.float32)
    for (_, slc), value in zip(ACTION_PARTS, values, strict=False):
        action[:, slc] = np.asarray(value, dtype=np.float32)
    return action


def compose_state(
    left_gripper: np.ndarray,
    left_joint: np.ndarray,
    right_gripper: np.ndarray,
    right_joint: np.ndarray,
) -> np.ndarray:
    """Assemble a ``(1, STATE_DIM)`` bimanual state vector.

    The left/right gripper and arm-joint values are placed into their fixed slots;
    all other slots stay zero.

    Returns:
        The ``(1, STATE_DIM)`` bimanual state vector.
    """
    state = np.zeros((1, STATE_DIM), dtype=np.float32)
    for slc, value in (
        (slice(6, 7), left_gripper),
        (slice(7, 13), left_joint),
        (slice(20, 21), right_gripper),
        (slice(21, 27), right_joint),
    ):
        state[:, slc] = np.asarray(value, dtype=np.float32).reshape(1, slc.stop - slc.start)
    return state


def normalize_action(action: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardize ``action`` to zero-mean/unit-scale using ``(action - mean) / (std + eps)``.

    Returns:
        The normalized action array.
    """
    return (action - mean) / (std + ACTION_EPS)


def denormalize_action(action: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Invert :func:`normalize_action`, mapping a normalized action back to raw units.

    Returns:
        The denormalized action array.
    """
    return action * (std + ACTION_EPS) + mean


def split_action(action: np.ndarray) -> dict[str, np.ndarray]:
    """Slice an action array into the eight named bimanual parts (inverse of :func:`compose_action`).

    Returns:
        A mapping of part name to its action slice.
    """
    action = np.asarray(action, dtype=np.float32)
    return {name: action[..., slc] for name, slc in ACTION_PARTS}


def recover_action(action: np.ndarray, robot_state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert per-step delta actions into absolute targets given the current ``robot_state``.

    End-effector position deltas are applied in the current EE frame, orientation deltas are
    composed as axis-angle rotations, and gripper/joint deltas add to the current values.

    Returns:
        A mapping of target name to the recovered absolute target array.
    """
    parts = split_action(action)
    targets = {}

    for side in ("left", "right"):
        rotm = np.asarray(robot_state[f"{side}_ee_rotm"], dtype=np.float32).reshape(3, 3)
        pos = np.asarray(robot_state[f"{side}_ee_pos"], dtype=np.float32).reshape(3)
        gripper = np.asarray(robot_state[f"{side}_gripper_pos"], dtype=np.float32).reshape(1)
        joint = np.asarray(robot_state[f"{side}_arm_joint"], dtype=np.float32).reshape(6)

        targets[f"{side}_ee_pos"] = (pos[None] + parts[f"{side}_ee_pos"] @ rotm.T).astype(np.float32)
        targets[f"{side}_ee_rotm"] = np.stack(
            [rotm @ aa2rotm(delta) for delta in parts[f"{side}_ee_aa"]],
            axis=0,
        ).astype(np.float32)
        targets[f"{side}_gripper_pos"] = (gripper[None] + parts[f"{side}_gripper"]).astype(np.float32)
        targets[f"{side}_arm_joint"] = (joint[None] + parts[f"{side}_joint"]).astype(np.float32)

    return targets
