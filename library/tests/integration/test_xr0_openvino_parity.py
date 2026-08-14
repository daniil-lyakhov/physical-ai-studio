# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests: native PyTorch XR0 vs OpenVINO export numerical and closed-loop parity.

Loads the published ``XiaomiRobotics/Xiaomi-Robotics-0-LIBERO`` checkpoint via
``physicalai.policies.XR0(pretrained_name_or_path=...)``, exports it to a
self-contained OpenVINO IR, and validates that the export reproduces the native
model's behaviour:

  1. **Numerical**: ``model.predict_action_chunk`` max-abs-diff and cosine
     similarity on the representative sample observation. XR0's rectified-flow
     sampler draws Gaussian starting noise, so the exported IR's internal
     ``RandomUniform`` noise is exposed as an extra output and *replayed*
     through the eager model for an apples-to-apples comparison (mirrors
     ``xr0_orig_vs_export.py``).
  2. **Closed-loop**: LIBERO success-rate delta between the native policy and
     the exported ``InferenceModel`` on a single short task.
  3. **Tokenizer**: the exported OpenVINO tokenizer (``tokenizer.xml``) fed the
     NumPy preprocessor's rendered ``task`` prompt reproduces the full Qwen3-VL
     processor's ``input_ids`` (the graph's prompt ports are renamed to
     ``tokenized_prompt`` / ``tokenized_prompt_mask`` to consume it directly).

Both tests are marked ``@pytest.mark.slow`` because they require downloading a
multi-GB checkpoint, exporting a large VLM, and running many environment steps.
Run them explicitly with::

    pytest -m slow tests/integration/test_xr0_openvino_parity.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import openvino as ov
import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from openvino.preprocess import PrePostProcessor

os.environ.setdefault("MUJOCO_GL", "egl")

from physicalai.data.observation import IMAGES, STATE, TASK
from physicalai.inference import InferenceModel
from physicalai.inference.constants import TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.policies import XR0
from physicalai.policies.xr0.patchify import patchify_image_grid
from physicalai.policies.xr0.pretrained_utils import extract_xr0_dataset_stats

if TYPE_CHECKING:
    from physicalai.benchmark.gyms import LiberoBenchmark

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHECKPOINT = "XiaomiRobotics/Xiaomi-Robotics-0-LIBERO"
# Fixed rectified-flow noise seed so the eager replay is deterministic.
_SEED = 42
# The exported IR runs its DiT head in bf16, so the OpenVINO action differs from
# the eager (also bf16) action by a small kernel/accumulation epsilon even when
# fed identical noise; treat anything under this as a match.
_MAX_ABS_DIFF_TOLERANCE = 0.1
_MIN_COSINE_SIMILARITY = 0.99
# Qwen3-VL vision geometry, used to patchify the NumPy pixel grid for the eager
# path exactly as the exported graph patchifies it in-graph (matches
# ``tests/unit/policies/xr0/test_patchify.py``).
_TEMPORAL_PATCH_SIZE = 2
# Closed-loop LIBERO configuration. XR0 samples noise stochastically and the two
# backends cannot be forced onto the same noise through the benchmark, so the
# success-rate delta tolerance is generous.
_TASK_SUITE = "libero_10"
_TASK_IDS = [0]
_NUM_EPISODES = 5
_SUCCESS_RATE_DIFF_TOLERANCE_PCT = 40.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_dataset_stats() -> dict[str, Any]:
    """Build the XR0 dataset stats, adding the observation schema.

    The checkpoint only carries action normalization stats, so the observation
    schema (state + two camera views) is added so ``sample_input`` and the
    preprocessor have everything they need.

    Returns:
        The dataset stats dict augmented with the observation schema.
    """
    stats = extract_xr0_dataset_stats(_CHECKPOINT) or {}
    stats["observation.state"] = {"name": "state", "type": "STATE", "shape": (8,)}
    stats["observation.images.base"] = {"name": "images.base", "type": "VISUAL", "shape": (3, 256, 256)}
    stats["observation.images.wrist_left"] = {
        "name": "images.wrist_left",
        "type": "VISUAL",
        "shape": (3, 256, 256),
    }
    return stats


def _build_native_policy() -> XR0:
    """Build the XR0 policy from the published LIBERO checkpoint in eval mode.

    Returns:
        The initialized XR0 policy.
    """
    policy = XR0(
        pretrained_name_or_path=_CHECKPOINT,
        dataset_stats=_build_dataset_stats(),
        vlm_attn_implementation="sdpa",
    )
    policy.eval()
    return policy


def _build_processed(policy: XR0) -> dict[str, torch.Tensor]:
    """Preprocess the sample observation and right-pad it to the baked length.

    Returns:
        The preprocessor output with ``input_ids`` / ``attention_mask``
        right-padded to ``config.tokenizer_max_length`` (the fixed length the
        export graph is baked for).
    """
    processed = policy._preprocessor(policy.sample_input)
    seq_len = policy.config.tokenizer_max_length
    pad_id = policy._preprocessor.processor.tokenizer.pad_token_id or 0
    cur_len = processed["input_ids"].shape[1]
    if cur_len > seq_len:
        msg = f"Sample prompt ({cur_len} tokens) exceeds SEQ_LEN={seq_len}."
        raise ValueError(msg)
    pad = seq_len - cur_len
    if pad:
        processed["input_ids"] = F.pad(processed["input_ids"], (0, pad), value=pad_id)
        processed["attention_mask"] = F.pad(processed["attention_mask"], (0, pad), value=0)
    return processed


@torch.no_grad()
def _run_forward_with_noise(policy: XR0, processed: dict[str, torch.Tensor], noise: torch.Tensor) -> torch.Tensor:
    """Run ``predict_action_chunk`` forcing a specific rectified-flow ``noise``.

    The exported IR samples its starting noise internally (a ``RandomUniform``
    with seed 0), so to compare it against the eager model we feed the eager
    model the *same* noise the IR drew. This temporarily overrides
    ``XR0Model._sample_noise`` so the flow starts from the supplied ``noise``.

    Returns:
        The predicted (still normalized) action chunk as a CPU float32 tensor.
    """
    batch: dict[str, object] = {
        key: (value.clone() if torch.is_tensor(value) else value) for key, value in processed.items()
    }
    model = policy.model
    original = model._sample_noise
    model._sample_noise = lambda action, seed: noise.to(action.device, action.dtype)  # noqa: ARG005
    try:
        actions = model.predict_action_chunk(batch)
    finally:
        model._sample_noise = original
    return actions.float().cpu()


def _find_noise_node(model: ov.Model) -> ov.Node:
    """Locate the Gaussian-noise (``randn``) node in the exported IR.

    ``torch.randn`` lowers to a ``RandomUniform`` followed by a Box-Muller
    transform: ``sqrt(-2*log(u1)) * cos(2*pi*u2)``. The final ``Multiply`` of
    that ``Sqrt`` and ``Cos`` branch is the rectified-flow starting noise.

    Returns:
        The ``Multiply`` node producing the Gaussian noise tensor.

    Raises:
        RuntimeError: If the Box-Muller ``Multiply`` cannot be found.
    """
    for op in model.get_ops():
        if op.get_type_name() != "Multiply":
            continue
        parents = {op.input_value(i).get_node().get_type_name() for i in range(len(op.inputs()))}
        if {"Sqrt", "Cos"} <= parents:
            return op
    msg = "Could not locate the Box-Muller noise node (Sqrt*Cos) in the IR."
    raise RuntimeError(msg)


def _run_openvino_ir(ir_xml: Path, graph_inputs: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the exported IR, returning both its action and the noise it drew.

    The IR's ``RandomUniform`` draws its noise internally, so the noise node is
    exposed as an extra output and read back from the *same* run as the action --
    guaranteeing the returned ``action`` and ``noise`` are consistent. The noise
    can then be replayed through the eager model for an apples-to-apples
    comparison.

    Args:
        ir_xml: Path to the exported OpenVINO IR ``.xml``.
        graph_inputs: Feed dict already keyed by the graph's (renamed) input
            names -- ``tokenized_prompt`` / ``tokenized_prompt_mask`` /
            ``pixel_values`` / ``state`` -- with matching NumPy dtypes.

    Returns:
        Tuple of ``(action, noise)`` as CPU float32 tensors.
    """
    core = ov.Core()
    model = core.read_model(ir_xml)

    # Expose the internal Gaussian noise as a second output (cast to f32 so NumPy
    # can read the otherwise-bf16 tensor) without disturbing the action output.
    noise_node = _find_noise_node(model)
    model.add_outputs(noise_node.output(0))
    ppp = PrePostProcessor(model)
    ppp.output(1).tensor().set_element_type(ov.Type.f32)
    model = ppp.build()

    compiled = core.compile_model(model, "CPU")
    result = compiled(graph_inputs)
    action = torch.from_numpy(np.asarray(result[compiled.output(0)])).float()
    noise = torch.from_numpy(np.asarray(result[compiled.output(1)])).float()
    return action, noise


def _locate_ir_xml(export_dir: Path) -> Path:
    """Resolve the exported OpenVINO IR ``.xml`` path from the export manifest.

    Returns:
        The path to the exported OpenVINO IR ``.xml`` file.
    """
    manifest = json.loads((export_dir / "manifest.json").read_text())
    return export_dir / manifest["model"]["artifacts"]["openvino"]


def _load_numpy_preprocessor(manifest: dict[str, Any]) -> Any:  # noqa: ANN401
    """Reconstruct the exported XR0 NumPy inference preprocessor from the manifest.

    Returns:
        The Runtime ``XR0Preprocessor`` resolved from the ``type="xr0"`` spec via
        the component registry (so the baked image geometry / normalization
        constants are exercised through the real instantiation path).

    Raises:
        RuntimeError: If the preprocessor spec is missing from the manifest.
    """
    from physicalai.inference.component_factory import instantiate_component  # noqa: PLC0415
    from physicalai.inference.manifest import ComponentSpec  # noqa: PLC0415

    for spec in manifest["model"]["preprocessors"]:
        if spec.get("type") == "xr0":
            return instantiate_component(ComponentSpec.model_validate(spec))
    msg = "xr0 preprocessor spec not found in the export manifest"
    raise RuntimeError(msg)


def _raw_observation(policy: XR0) -> dict[str, object]:
    """Convert the policy's torch ``sample_input`` into a raw NumPy observation.

    Returns:
        The observation dict with flattened ``images.*`` arrays, a ``state`` array
        and a ``task`` string, as the NumPy inference preprocessor consumes.
    """
    observation: dict[str, object] = {}
    for key, value in policy.sample_input.items():
        if isinstance(key, str) and key.startswith(f"{IMAGES}."):
            observation[key] = value.detach().cpu().numpy()
        elif key == STATE:
            observation[STATE] = value.detach().cpu().numpy()
        elif key == TASK:
            observation[TASK] = value
    return observation


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def native_policy() -> XR0:
    """Load the native XR0 policy from the pretrained checkpoint once per module."""
    return _build_native_policy()


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Export a fresh XR0 policy to OpenVINO once per module and return the export directory.

    A separate policy instance is exported (and then discarded) because XR0's
    in-graph export bakes constants and rebinds module forwards in place, which
    would leave ``native_policy`` unusable for the eager and closed-loop runs.
    """
    export_path = tmp_path_factory.mktemp("xr0_openvino_export")
    policy = _build_native_policy()
    policy.export(export_path, backend="openvino")
    return export_path


@pytest.fixture(scope="module")
def exported_model(export_dir: Path) -> InferenceModel:
    """Load the exported OpenVINO XR0 model through the Runtime ``InferenceModel``."""
    return InferenceModel(str(export_dir), device="CPU")


@pytest.fixture(scope="module")
def libero_benchmark() -> LiberoBenchmark:
    """Create the LIBERO benchmark for a single short task once per module."""
    pytest.importorskip("libero", reason="LIBERO not installed")
    pytest.importorskip("robosuite", reason="robosuite not installed")

    from physicalai.benchmark.gyms import LiberoBenchmark

    return LiberoBenchmark(
        task_suite=_TASK_SUITE,
        task_ids=_TASK_IDS,
        num_episodes=_NUM_EPISODES,
        seed=_SEED,
    )


@pytest.fixture(scope="module")
def numerical_parity(native_policy: XR0, export_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Run the exported IR and replay its noise through the eager model once per module.

    The deployed graph consumes the NumPy inference preprocessor's *pixel grid*
    and patchifies it in-graph, so both backends are fed identical inputs derived
    from that preprocessor: the IR gets the grid directly, and the eager model
    gets the same grid patchified (plus the shared prompt ids/mask and state).

    Returns:
        Tuple of ``(eager_action, exported_action)`` as float32 NumPy arrays,
        computed from identical rectified-flow noise.
    """
    torch.manual_seed(_SEED)

    manifest = json.loads((export_dir / "manifest.json").read_text())
    preprocessor = _load_numpy_preprocessor(manifest)
    np_out = preprocessor(_raw_observation(native_policy))
    pixel_grid = np.ascontiguousarray(np_out["pixel_values"], dtype=np.float32)
    state = np.ascontiguousarray(np_out["state"], dtype=np.float32)

    spec = next(s for s in manifest["model"]["preprocessors"] if s.get("type") == "xr0")
    patch_size = int(spec["patch_size"])
    merge_size = int(spec["merge_size"])

    # The rendered NumPy prompt tokenizes to the same ids as the full processor
    # (see the tokenizer-parity test), so reuse the eager preprocessor's ids/mask
    # (already right-padded to the baked graph length).
    processed = _build_processed(native_policy)
    graph_inputs = {
        TOKENIZED_PROMPT: processed["input_ids"].cpu().numpy().astype(np.int64),
        TOKENIZED_PROMPT_MASK: processed["attention_mask"].cpu().numpy().astype(np.int64),
        "pixel_values": pixel_grid,
        "state": state,
    }
    ir_xml = _locate_ir_xml(export_dir)
    exported_action, exported_noise = _run_openvino_ir(ir_xml, graph_inputs)

    # Feed the eager model the *same* inputs: patchify the identical pixel grid so
    # the eager vision tower sees exactly what the in-graph patchify produces.
    num_images, _, height, width = pixel_grid.shape
    grid_thw = [[1, height // patch_size, width // patch_size]] * num_images
    eager_pixels = patchify_image_grid(
        torch.from_numpy(pixel_grid),
        grid_thw,
        temporal_patch_size=_TEMPORAL_PATCH_SIZE,
        patch_size=patch_size,
        merge_size=merge_size,
    )
    eager_processed = {
        "input_ids": processed["input_ids"],
        "attention_mask": processed["attention_mask"],
        "pixel_values": eager_pixels,
        "image_grid_thw": torch.tensor(grid_thw, dtype=torch.int64),
        "state": torch.from_numpy(state),
    }
    eager_action = _run_forward_with_noise(native_policy, eager_processed, exported_noise)
    return eager_action.numpy(), np.asarray(exported_action)


# ---------------------------------------------------------------------------
# Numerical parity test
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestXR0OpenVINONumericalParity:
    """Verify predict_action_chunk outputs are numerically close between backends."""

    def test_max_abs_diff_within_tolerance(
        self,
        numerical_parity: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Max absolute difference (same noise) must be below tolerance."""
        eager_action, exported_action = numerical_parity
        assert eager_action.shape == exported_action.shape, (
            f"Shape mismatch: eager {eager_action.shape} vs exported {exported_action.shape}"
        )
        max_abs = float(np.abs(eager_action - exported_action).max())
        assert max_abs <= _MAX_ABS_DIFF_TOLERANCE, (
            f"Max abs diff {max_abs:.6f} exceeds tolerance {_MAX_ABS_DIFF_TOLERANCE}"
        )

    def test_cosine_similarity_near_one(
        self,
        numerical_parity: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Cosine similarity (same noise) must be close to 1."""
        eager_action, exported_action = numerical_parity
        eager_flat = eager_action.flatten()
        exported_flat = exported_action.flatten()
        cosine = float(
            np.dot(eager_flat, exported_flat)
            / (np.linalg.norm(eager_flat) * np.linalg.norm(exported_flat) + 1e-12)
        )
        assert cosine >= _MIN_COSINE_SIMILARITY, (
            f"Cosine similarity {cosine:.6f} is below {_MIN_COSINE_SIMILARITY}"
        )


# ---------------------------------------------------------------------------
# Closed-loop parity test
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skip()
class TestXR0OpenVINOClosedLoopParity:
    """Verify closed-loop LIBERO success rates are comparable between backends."""

    def test_success_rate_difference_within_tolerance(
        self,
        native_policy: XR0,
        exported_model: InferenceModel,
        libero_benchmark: LiberoBenchmark,
    ) -> None:
        """Absolute success-rate difference must be within tolerance."""
        native_results = libero_benchmark.evaluate(native_policy)
        exported_results = libero_benchmark.evaluate(exported_model)

        native_rate = native_results.overall_success_rate
        exported_rate = exported_results.overall_success_rate
        diff = abs(native_rate - exported_rate)

        assert diff <= _SUCCESS_RATE_DIFF_TOLERANCE_PCT, (
            f"Success-rate diff {diff:.1f}pp exceeds tolerance {_SUCCESS_RATE_DIFF_TOLERANCE_PCT}pp "
            f"(native={native_rate:.1f}%, openvino={exported_rate:.1f}%)"
        )


# ---------------------------------------------------------------------------
# OpenVINO tokenizer parity test
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestXR0OVTokenizerParity:
    """Verify the exported ``tokenizer.xml`` reproduces the processor's token ids."""

    def test_ov_tokenizer_matches_processor(
        self,
        native_policy: XR0,
        export_dir: Path,
    ) -> None:
        """NumPy preprocessor + exported ``ov_tokenizer`` ids equal the processor ids.

        Reconstructs the exported NumPy preprocessor from the manifest, renders the
        sample observation's ``task`` prompt, tokenizes it with the exported
        OpenVINO tokenizer, and asserts the resulting ``tokenized_prompt`` (trimmed
        to the real length via its mask) matches, bit-for-bit, the ``input_ids`` the
        full Qwen3-VL processor produces for the same observation.
        """
        from physicalai.inference.preprocessors.ov_tokenizer import OVTokenizer

        manifest = json.loads((export_dir / "manifest.json").read_text())
        preprocessor = _load_numpy_preprocessor(manifest)
        preprocessed = preprocessor(_raw_observation(native_policy))

        tokenizer = OVTokenizer(export_dir / "tokenizer.xml")
        tokenized = tokenizer(dict(preprocessed))
        ov_ids = np.asarray(tokenized[TOKENIZED_PROMPT])[0]
        ov_mask = np.asarray(tokenized[TOKENIZED_PROMPT_MASK])[0].astype(bool)
        ov_real_ids = ov_ids[ov_mask]

        processed = native_policy._preprocessor(native_policy.sample_input)
        real_len = int(processed["attention_mask"][0].sum().item())
        reference_ids = processed["input_ids"][0, :real_len].cpu().numpy()

        assert ov_real_ids.shape == reference_ids.shape, (
            f"Token count mismatch: ov_tokenizer {ov_real_ids.shape} vs processor {reference_ids.shape}"
        )
        np.testing.assert_array_equal(ov_real_ids, reference_ids)

