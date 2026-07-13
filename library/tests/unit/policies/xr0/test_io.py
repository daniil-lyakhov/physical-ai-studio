# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - XR0 io adapters.

Lightweight, self-contained checks of the vendored XR0 bimanual state/action
helpers. Inputs are tiny and inline; correctness is asserted via structural
invariants and round-trips (no golden binaries). Numerical parity against the
source is proven separately by the golden fixtures (``G00``-``G08``).
"""

import numpy as np
from PIL import Image

from physicalai.policies.xr0 import io


class TestComposeSplit:
    """State/action packing into the 32-dim bimanual layout."""

    def test_compose_state_slots(self):
        state = io.compose_state(
            left_gripper=[0.1], left_joint=np.arange(1, 7, dtype=np.float32),
            right_gripper=[0.2], right_joint=np.arange(7, 13, dtype=np.float32),
        )
        assert state.shape == (1, io.STATE_DIM)
        assert state[0, 6] == 0.1
        assert state[0, 20] == 0.2
        np.testing.assert_array_equal(state[0, 7:13], np.arange(1, 7))
        np.testing.assert_array_equal(state[0, 21:27], np.arange(7, 13))
        # Untouched slots stay zero (e.g. ee_pos/ee_aa and padding columns).
        assert state[0, 0:6].sum() == 0.0
        assert state[0, 13] == 0.0
        assert state[0, 27:].sum() == 0.0

    def test_compose_split_roundtrip(self):
        rng = np.random.default_rng(0)
        parts = {name: rng.standard_normal((5, slc.stop - slc.start)).astype(np.float32)
                 for name, slc in io.ACTION_PARTS}
        action = io.compose_action(
            parts["left_ee_pos"], parts["left_ee_aa"], parts["left_gripper"], parts["left_joint"],
            parts["right_ee_pos"], parts["right_ee_aa"], parts["right_gripper"], parts["right_joint"],
            action_length=5,
        )
        assert action.shape == (5, io.ACTION_DIM)
        # Padding columns are never written.
        assert action[:, 13].sum() == 0.0
        assert action[:, 27:].sum() == 0.0
        recovered = io.split_action(action)
        for name in parts:
            np.testing.assert_array_equal(recovered[name], parts[name])

    def test_build_action_mask(self):
        temporal = np.array([1, 0, 1], dtype=np.int32)
        mask = io.build_action_mask(3, temporal)
        assert mask.shape == (3, io.ACTION_DIM)
        # Valid columns follow the layout; padding columns stay zero.
        for _, slc in io.ACTION_PARTS:
            np.testing.assert_array_equal(mask[:, slc], np.broadcast_to(temporal[:, None], (3, slc.stop - slc.start)))
        assert mask[:, 13].sum() == 0
        assert mask[:, 27:].sum() == 0


class TestNormalization:
    """Action normalize/denormalize round-trip."""

    def test_normalize_denormalize_roundtrip(self):
        rng = np.random.default_rng(1)
        action = rng.standard_normal((4, io.ACTION_DIM)).astype(np.float32)
        mean = rng.standard_normal((4, io.ACTION_DIM)).astype(np.float32)
        std = np.abs(rng.standard_normal((4, io.ACTION_DIM)).astype(np.float32)) + 0.1
        normalized = io.normalize_action(action, mean, std)
        roundtrip = io.denormalize_action(normalized, mean, std)
        np.testing.assert_allclose(roundtrip, action, atol=1e-5, rtol=1e-5)


class TestRotation:
    """Axis-angle <-> rotation-matrix math."""

    def test_aa2rotm_identity(self):
        np.testing.assert_allclose(io.aa2rotm([0.0, 0.0, 0.0]), np.eye(3), atol=1e-6)

    def test_aa2rotm_orthonormal(self):
        rotm = io.aa2rotm([0.3, -0.7, 1.1])
        np.testing.assert_allclose(rotm @ rotm.T, np.eye(3), atol=1e-5)
        np.testing.assert_allclose(np.linalg.det(rotm), 1.0, atol=1e-5)

    def test_aa_rotm_roundtrip(self):
        axis_angles = np.array(
            [[0.0, 0.0, 0.0], [0.2, -0.5, 0.9], [np.pi, 0.0, 0.0], [0.0, np.pi / 2, 0.0]],
            dtype=np.float32,
        )
        rotms = np.stack([io.aa2rotm(aa) for aa in axis_angles], axis=0)
        recovered = io.rotm2aa_batch(rotms)
        # Compare via the resulting rotation matrices (axis-angle sign is ambiguous at pi).
        back = np.stack([io.aa2rotm(aa) for aa in recovered], axis=0)
        np.testing.assert_allclose(back, rotms, atol=1e-5)


class TestRecoverAction:
    """Delta -> absolute target recovery."""

    def test_identity_frame_is_additive(self):
        # With identity rotation at the origin, deltas add directly.
        action = np.zeros((3, io.ACTION_DIM), dtype=np.float32)
        action[:, 0:3] = np.array([[0.1, 0.2, 0.3]] * 3)   # left_ee_pos delta
        action[:, 6] = 0.5                                  # left_gripper delta
        action[:, 7:13] = 0.01                              # left_joint delta
        robot_state = {}
        for side in ("left", "right"):
            robot_state[f"{side}_ee_rotm"] = np.eye(3, dtype=np.float32).reshape(9)
            robot_state[f"{side}_ee_pos"] = np.zeros(3, dtype=np.float32)
            robot_state[f"{side}_gripper_pos"] = np.zeros(1, dtype=np.float32)
            robot_state[f"{side}_arm_joint"] = np.zeros(6, dtype=np.float32)
        out = io.recover_action(action, robot_state)
        np.testing.assert_allclose(out["left_ee_pos"], action[:, 0:3], atol=1e-6)
        np.testing.assert_allclose(out["left_gripper_pos"][:, 0], action[:, 6], atol=1e-6)
        np.testing.assert_allclose(out["left_arm_joint"], action[:, 7:13], atol=1e-6)


class TestResizeImage:
    """VLM image resize keeps factor alignment within the pixel budget."""

    def test_factor_aligned_within_budget(self):
        img = Image.fromarray(np.zeros((200, 300, 3), dtype=np.uint8))
        out = io.resize_image(img, factor=32, min_pixels=1024, max_pixels=90000)
        w, h = out.size
        assert w % 32 == 0 and h % 32 == 0
        assert w * h <= 90000
