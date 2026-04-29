# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pytest tests for partitioned Pi0.5 OpenVINO export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

_DEFAULT_EXPORT_CACHE_DIR = Path(__file__).resolve().parent / ".export_cache" / "partitioned"
_DEFAULT_MONOLITHIC_CACHE_DIR = Path(__file__).resolve().parent / ".export_cache" / "monolithic"
_MODEL_SEED = 42
_CHUNK_SIZE = 10
_MAX_ACTION_DIM = 32
_NUM_INFERENCE_STEPS = 3
_DEVICE = "cpu"

_DATASET_STATS = {
    "observation.state": {
        "shape": (14,),
        "mean": [0.0] * 14,
        "std": [1.0] * 14,
        "name": "state",
    },
    "observation.images.top": {
        "shape": (3, 224, 224),
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "name": "top",
    },
    "observation.images.wrist": {
        "shape": (3, 224, 224),
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "name": "wrist",
    },
    "action": {
        "shape": (14,),
        "mean": [0.0] * 14,
        "std": [1.0] * 14,
        "name": "action",
    },
}


@pytest.fixture(scope="module")
def pi05_model():
    """Create a standalone Pi05Model with dummy dataset stats.

    Uses a fixed seed so that random weights are reproducible across runs,
    which is required for reusing cached OV exports.
    """
    print("Start pi0.5 pytorch model init...")
    from physicalai.policies.pi05.model import Pi05Model

    torch.manual_seed(_MODEL_SEED)

    model = Pi05Model(
        _DATASET_STATS,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        dtype="float32",
        chunk_size=_CHUNK_SIZE,
        max_action_dim=_MAX_ACTION_DIM,
        num_inference_steps=_NUM_INFERENCE_STEPS,
        image_resolution=(224, 224),
        gradient_checkpointing=False,
    )
    model.eval()

    print("Done pi0.5 pytorch model init")
    return model


@pytest.fixture(scope="module")
def export_dir(request):
    """Export partitioned OV models to a directory, reusing cache if available.

    Use ``--force-reexport`` to force a fresh export even when cached artifacts exist.
    Use ``--export-cache-dir <path>`` to override the default cache location.
    """
    from physicalai.policies.pi05.partitioned_export import export_partitioned_openvino

    force = request.config.getoption("--force-reexport", default=False)
    out = Path(request.config.getoption("--export-cache-dir", default=str(_DEFAULT_EXPORT_CACHE_DIR)))

    marker_file = out / "export_done.marker"
    if not force and marker_file.exists():
        yield out
    else:
        if out.exists():
            import shutil

            shutil.rmtree(out)

        pi05_model = request.getfixturevalue("pi05_model")
        print("Start pi0.5 openvino partial model init...")
        export_partitioned_openvino(pi05_model, output_dir=str(out), compress_to_fp16=False)
        print("Done pi0.5 openvino partial model init")
        marker_file.touch()
        yield out


_REFERENCE_INPUT_SEED = 123


def _make_reference_inputs(
    batch_size: int = 2,
    n_cameras: int = 2,
) -> dict[str, np.ndarray]:
    """Create deterministic model inputs."""
    np.random.seed(_REFERENCE_INPUT_SEED)
    return {
        "images": np.random.randn(n_cameras, batch_size, 3, 224, 224).astype(np.float32),
        "image_masks": np.ones((n_cameras, batch_size), dtype=bool),
        "tokenized_prompt": np.zeros((batch_size, 200), dtype=np.int64),
        "tokenized_prompt_mask": np.ones((batch_size, 200), dtype=bool),
    }


@pytest.fixture(scope="module")
def pi05_reference(request):
    """Provide deterministic inputs and cached PyTorch reference outputs.

    If a cached .npz file exists, loads from disk without initializing the
    PyTorch model.  Otherwise, requests ``pi05_model``, runs inference,
    and saves the results for future runs.
    """
    cache_file = _DEFAULT_EXPORT_CACHE_DIR.parent / "pytorch_reference.npz"
    inputs = _make_reference_inputs()

    force = request.config.getoption("--force-reexport", default=False)
    if force or not cache_file.exists():
        print("Original model output regeneration starts...")
        pi05_model = request.getfixturevalue("pi05_model")
        batch_size = inputs["images"].shape[1]
        n_cameras = inputs["images"].shape[0]
        noise = torch.zeros(
            batch_size,
            _CHUNK_SIZE,
            _MAX_ACTION_DIM,
            dtype=torch.float32,
            device=_DEVICE,
        )
        with torch.no_grad():
            pytorch_actions = pi05_model.sample_actions(
                [torch.from_numpy(inputs["images"][i]).to(_DEVICE) for i in range(n_cameras)],
                [torch.from_numpy(inputs["image_masks"][i]).to(_DEVICE) for i in range(n_cameras)],
                torch.from_numpy(inputs["tokenized_prompt"]).to(_DEVICE),
                torch.from_numpy(inputs["tokenized_prompt_mask"]).to(_DEVICE),
                noise=noise,
            ).numpy()
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(cache_file), pytorch_actions=pytorch_actions)
        print(f"Original model output saved at {cache_file}")
    else:
        data = np.load(str(cache_file))
        print(f"Original model output loaded from {cache_file}")
        pytorch_actions = data["pytorch_actions"]

    return {"inputs": inputs, "pytorch_actions": pytorch_actions}


def _run_partitioned_wrappers(pi05_model, images, img_masks, tokens, masks, noise):
    """Run partitioned wrappers and return denoised actions.

    Uses ImageEmbedder per camera + language embedding + PrefixLM,
    matching the adapter's runtime flow.
    """
    import math  # noqa: PLC0415

    from physicalai.policies.pi05.partitioned_export import (
        ActionOutputHead,
        GemmaExpertDecoder,
        ImageEmbedder,
        PrefixLM,
    )

    device = noise.device
    batch_size = noise.shape[0]

    embedder = ImageEmbedder(pi05_model).eval()
    prefix_lm = PrefixLM(pi05_model).eval()
    decoder = GemmaExpertDecoder(pi05_model).eval()
    head = ActionOutputHead(pi05_model).eval()

    # Image embedding per camera
    embs = []
    pad_masks = []
    att_masks_list: list[int] = []

    for img, img_mask in zip(images, img_masks, strict=True):
        img_emb = embedder(img)
        bsize, num_patches = img_emb.shape[:2]
        embs.append(img_emb)
        pad_masks.append(img_mask[:, None].expand(bsize, num_patches))
        att_masks_list.extend([0] * num_patches)

    # Language embedding
    lang_emb = pi05_model.paligemma_with_expert.embed_language_tokens(tokens)
    lang_emb_dim = lang_emb.shape[-1]
    lang_emb = lang_emb * math.sqrt(lang_emb_dim)
    embs.append(lang_emb)
    pad_masks.append(masks)
    att_masks_list.extend([0] * lang_emb.shape[1])

    prefix_embs = torch.cat(embs, dim=1)
    prefix_pad_masks = torch.cat(pad_masks, dim=1)
    prefix_att_masks = torch.tensor(
        att_masks_list,
        dtype=torch.bool,
        device=device,
    )
    prefix_att_masks = prefix_att_masks[None, :].expand(batch_size, len(att_masks_list))

    # Pre-compute 4D attention mask and position IDs for PrefixLM
    from physicalai.policies.pi05.model import OPENPI_ATTENTION_MASK_VALUE, _make_att_2d_masks

    prefix_att_2d_masks = _make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = prefix_att_2d_masks[:, None, :, :]
    prefix_att_2d_masks_4d = torch.where(
        prefix_att_2d_masks_4d,
        0.0,
        OPENPI_ATTENTION_MASK_VALUE,
    )

    # Prefix LM → KV cache
    cache_keys, cache_values = prefix_lm(
        prefix_embs,
        prefix_att_2d_masks_4d,
        prefix_position_ids,
    )

    x_t = torch.zeros_like(noise)
    dt = -1.0 / _NUM_INFERENCE_STEPS

    for step in range(_NUM_INFERENCE_STEPS):
        t = 1.0 + step * dt
        time_tensor = torch.tensor(t, dtype=torch.float32, device=device).expand(batch_size)
        suffix_out = decoder(x_t, time_tensor, cache_keys, cache_values, prefix_pad_masks)
        v_t = head(suffix_out)
        x_t = x_t + dt * v_t

    return x_t


class TestPytorchWrappers:
    """Test 1: PyTorch wrapper modules reproduce original sample_actions."""

    @pytest.mark.parametrize(
        ("batch_size", "n_cameras"),
        [(1, 1), (2, 1), (1, 2), (2, 2)],
        ids=["bs1_cam1", "bs2_cam1", "bs1_cam2", "bs2_cam2"],
    )
    def test_wrappers_match_original(self, pi05_model, batch_size, n_cameras):
        device = _DEVICE

        images = [torch.randn(batch_size, 3, 224, 224, device=device) for _ in range(n_cameras)]
        img_masks = [torch.ones(batch_size, dtype=torch.bool, device=device) for _ in range(n_cameras)]
        tokens = torch.randint(0, 1000, (batch_size, 200), device=device)
        masks = torch.ones(batch_size, 200, dtype=torch.bool, device=device)
        noise = torch.zeros(batch_size, _CHUNK_SIZE, _MAX_ACTION_DIM, dtype=torch.float32, device=device)

        with torch.no_grad():
            original = pi05_model.sample_actions(images, img_masks, tokens, masks, noise=noise)
            wrapper_result = _run_partitioned_wrappers(
                pi05_model,
                images,
                img_masks,
                tokens,
                masks,
                noise,
            )

        max_diff = torch.max(torch.abs(original - wrapper_result)).item()
        assert max_diff < 1e-4, f"bs={batch_size}, cams={n_cameras}: max_diff={max_diff}"


class TestOpenVINOExport:
    """Test 2: OV export + adapter predict matches PyTorch."""

    def test_ov_adapter_matches_pytorch(self, pi05_reference, export_dir):
        from physicalai.inference.adapters.openvino_partitioned import PartitionedOpenVINOAdapter

        adapter = PartitionedOpenVINOAdapter(
            device="CPU",
            num_inference_steps=_NUM_INFERENCE_STEPS,
            chunk_size=_CHUNK_SIZE,
            max_action_dim=_MAX_ACTION_DIM,
        )
        adapter.load(export_dir)

        inputs = pi05_reference["inputs"]
        pytorch_actions = pi05_reference["pytorch_actions"]
        batch_size = inputs["images"].shape[1]

        noise = np.zeros(
            (batch_size, _CHUNK_SIZE, _MAX_ACTION_DIM),
            dtype=np.float32,
        )
        outputs = adapter.predict(inputs, noise=noise)
        assert "action" in outputs
        action = outputs["action"]
        assert action.shape == (batch_size, _CHUNK_SIZE, _MAX_ACTION_DIM)

        assert np.max(np.abs(action - pytorch_actions)) < 1e-3


@pytest.fixture(scope="module")
def pi05_policy():
    """Create a full Pi05 policy with preprocessor/postprocessor.

    Uses the same seed and config as ``pi05_model`` so weights are identical
    and the OV export cache can be reused.
    """
    from physicalai.policies.pi05.policy import Pi05

    torch.manual_seed(_MODEL_SEED)

    policy = Pi05(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        dtype="float32",
        chunk_size=_CHUNK_SIZE,
        n_action_steps=_CHUNK_SIZE,
        max_action_dim=_MAX_ACTION_DIM,
        num_inference_steps=_NUM_INFERENCE_STEPS,
        image_resolution=(224, 224),
        gradient_checkpointing=False,
        normalization_mode="MEAN_STD",
        use_random_input_noise=False,
        dataset_stats=_DATASET_STATS,
    )
    policy.eval()
    return policy


_POLICY_REF_SEED = 456


@pytest.fixture(scope="module")
def policy_reference(request):
    """Provide a raw Observation, its numpy form, and cached PyTorch reference actions.

    Caches ``pt_actions.npy`` and ``observation.npz`` inside *export_dir* so
    the heavy ``Pi05`` policy is only built when cached artifacts are missing.
    Use ``--force-reexport`` to regenerate.
    """
    from physicalai.data.observation import Observation

    export_dir = _DEFAULT_EXPORT_CACHE_DIR.parent
    cache_obs = export_dir / "policy_ref_observation.npz"
    cache_actions = export_dir / "policy_ref_actions.npy"

    force = request.config.getoption("--force-reexport", default=False)

    if not force and cache_obs.exists() and cache_actions.exists():
        print(f"Loading cached policy reference from {export_dir}")
        obs_data = dict(np.load(str(cache_obs), allow_pickle=False))
        pt_actions_np = np.load(str(cache_actions))

        # Rebuild high-level inputs dict (InferenceModel preprocessors
        # convert state/images/task → tokenized_prompt, image_masks, etc.)
        np_inputs = {
            "state": obs_data["state"],
            "images": {
                "top": obs_data["images_top"],
                "wrist": obs_data["images_wrist"],
            },
            "task": obs_data["task"].tolist(),
        }
    else:
        print("Generating policy reference outputs...")
        pi05_policy = request.getfixturevalue("pi05_policy")

        batch_size = 2
        device = _DEVICE

        torch.manual_seed(_POLICY_REF_SEED)
        obs = Observation(
            state=torch.randn(batch_size, 14, device=device),
            images={
                "top": torch.randn(batch_size, 3, 224, 224, device=device),
                "wrist": torch.randn(batch_size, 3, 224, 224, device=device),
            },
            task=["pick up the object", "pick up the object"],
        )

        with torch.no_grad():
            pt_actions = pi05_policy.predict_action_chunk(obs)
        pt_actions_np = pt_actions.cpu().numpy()

        # Convert observation to numpy for caching and OV inference
        np_obs = obs.to_numpy().to_dict(flatten=False)
        np_inputs = {k: v for k, v in np_obs.items() if v is not None}

        # Cache to disk
        export_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(cache_obs),
            state=np_inputs["state"],
            images_top=np_inputs["images"]["top"],
            images_wrist=np_inputs["images"]["wrist"],
            task=np.array(np_inputs["task"]),
        )
        np.save(str(cache_actions), pt_actions_np)
        print(f"Policy reference cached to {export_dir}")

    return {"np_inputs": np_inputs, "pt_actions_np": pt_actions_np}


class TestInferenceModelVsPolicy:
    """Test 5: InferenceModel (OV + pre/postprocessors) vs Pi05 predict_action_chunk."""

    def test_inference_model_matches_predict_action_chunk(self, policy_reference, export_dir):
        """Full InferenceModel pipeline should match Pi05.predict_action_chunk."""
        from unittest.mock import patch

        from physicalai.data.observation import ACTION
        from physicalai.inference.model import InferenceModel
        from physicalai.inference.runners.single_pass import SinglePass

        pt_actions_np = policy_reference["pt_actions_np"]
        np_inputs = policy_reference["np_inputs"]

        # --- OpenVINO InferenceModel path ---
        inf_model = InferenceModel(
            export_dir=export_dir,
            runner=SinglePass(),
        )

        # Patch _sample_noise -> zeros to match Pi05's use_random_input_noise=False
        def _zeros_noise(self, batch_size):
            return np.zeros((batch_size, self._chunk_size, self._max_action_dim), dtype=np.float32)

        with patch(
            "physicalai.inference.adapters.openvino_partitioned.PartitionedOpenVINOAdapter._sample_noise",
            _zeros_noise,
        ):
            ov_outputs = inf_model(np_inputs)

        ov_actions = ov_outputs[ACTION]

        # Align shapes: predict_action_chunk trims to original_action_dim (14)
        # and n_action_steps; OV adapter trims to action_dim from manifest.
        min_action_dim = min(pt_actions_np.shape[-1], ov_actions.shape[-1])
        min_chunk = min(pt_actions_np.shape[-2], ov_actions.shape[-2])
        pt_trimmed = pt_actions_np[:, :min_chunk, :min_action_dim]
        ov_trimmed = ov_actions[:, :min_chunk, :min_action_dim]

        max_diff = np.max(np.abs(pt_trimmed - ov_trimmed))
        assert max_diff < 1e-3, f"InferenceModel vs predict_action_chunk max diff: {max_diff}"
        print(f"InferenceModel vs predict_action_chunk max diff: {max_diff}")


class TestQuantizedPartitionedModel:
    """Test 4: Weight-compressed (INT8) partitioned OV model vs PyTorch."""

    @pytest.mark.skip(reason="NNCF is not integrated yet")
    def test_nncf_weight_compression_matches_pytorch(self, pi05_reference, export_dir):
        """Compress each partition to INT8 via NNCF and compare with PyTorch."""
        import shutil

        import nncf
        import openvino

        from physicalai.inference.adapters.openvino_partitioned import PartitionedOpenVINOAdapter

        core = openvino.Core()
        compressed_dir = export_dir.resolve().parent / "compressed"

        model_files = {
            "image_embedder": export_dir / "image_embedder.xml",
            "prefix_lm": export_dir / "prefix_lm.xml",
            "gemma_expert_decoder": export_dir / "gemma_expert_decoder.xml",
            "action_output_head": export_dir / "action_output_head.xml",
        }

        for name, xml_path in model_files.items():
            ov_model = core.read_model(str(xml_path))
            if name in ["action_output_head", "image_embedder"]:
                compressed_model = ov_model
            else:
                compressed_model = nncf.compress_weights(
                    ov_model,
                    mode=nncf.CompressWeightsMode.INT8_SYM,
                )
            out_xml = compressed_dir / f"{name}.xml"
            openvino.save_model(compressed_model, str(out_xml))

        # Copy non-model artifacts (manifest, preprocessor config, tokenizer)
        for item in export_dir.iterdir():
            if item.suffix not in (".xml", ".bin") and not (compressed_dir / item.name).exists():
                if item.is_dir():
                    shutil.copytree(str(item), str(compressed_dir / item.name))
                else:
                    shutil.copy2(str(item), str(compressed_dir / item.name))

        adapter = PartitionedOpenVINOAdapter(
            device="CPU",
            num_inference_steps=_NUM_INFERENCE_STEPS,
            chunk_size=_CHUNK_SIZE,
            max_action_dim=_MAX_ACTION_DIM,
        )
        adapter.load(compressed_dir)

        inputs = pi05_reference["inputs"]
        pytorch_actions = pi05_reference["pytorch_actions"]
        batch_size = inputs["images"].shape[1]

        noise = np.zeros(
            (batch_size, _CHUNK_SIZE, _MAX_ACTION_DIM),
            dtype=np.float32,
        )
        outputs = adapter.predict(inputs, noise=noise)
        assert "action" in outputs
        action = outputs["action"]
        assert action.shape == (batch_size, _CHUNK_SIZE, _MAX_ACTION_DIM)

        # INT8 compression introduces larger error than FP32 export
        max_diff = np.max(np.abs(action - pytorch_actions))
        assert max_diff < 1.0, f"INT8 compressed model diverged too much: max_diff={max_diff}"
        print(f"Max diff: {max_diff}")

        # Verify compression actually reduced model size
        # TODO(dlyakhov): check if it is correct
        fp32_size = sum(f.stat().st_size for f in export_dir.glob("*.bin"))
        int8_size = sum(f.stat().st_size for f in compressed_dir.glob("*.bin"))
        assert int8_size < fp32_size, f"Compressed size ({int8_size}) should be smaller than original ({fp32_size})"
        print(f"File compression ratio: {fp32_size / int8_size}")


@pytest.fixture(scope="module")
def monolithic_export_dir(request):
    """Export a monolithic (non-partitioned) OV model, reusing cache if available.

    Uses the same ``pi05_policy`` fixture so weights are identical to the
    partitioned export.  Use ``--force-reexport`` to force a fresh export.
    """
    force = request.config.getoption("--force-reexport", default=False)
    out = Path(request.config.getoption("--export-cache-dir", default=str(_DEFAULT_MONOLITHIC_CACHE_DIR)))

    marker_file = out / "export_done.marker"
    if not force and marker_file.exists():
        yield out
    else:
        if out.exists():
            import shutil

            shutil.rmtree(out)

        pi05_policy = request.getfixturevalue("pi05_policy")
        print("Start pi0.5 monolithic openvino export...")
        pi05_policy.to_openvino(out)
        print("Done pi0.5 monolithic openvino export")
        marker_file.touch()
        yield out


@pytest.mark.skip(reason="FP16 monolitic vs FP32 partitial export")
class TestPartitionedVsMonolithic:
    """Test: Partitioned OV export matches monolithic (non-partitioned) OV export."""

    def test_partitioned_matches_monolithic(self, policy_reference, export_dir, monolithic_export_dir):
        """Both export paths should produce the same actions for identical inputs."""
        from unittest.mock import patch

        from physicalai.data.observation import ACTION
        from physicalai.inference.model import InferenceModel
        from physicalai.inference.runners.single_pass import SinglePass

        np_inputs = policy_reference["np_inputs"]

        def _zeros_noise(self, batch_size):
            return np.zeros((batch_size, self._chunk_size, self._max_action_dim), dtype=np.float32)

        # Run partitioned adapter
        partitioned_model = InferenceModel(
            export_dir=export_dir,
            runner=SinglePass(),
        )
        with patch(
            "physicalai.inference.adapters.openvino_partitioned.PartitionedOpenVINOAdapter._sample_noise",
            _zeros_noise,
        ):
            partitioned_outputs = partitioned_model(np_inputs)
        partitioned_actions = partitioned_outputs[ACTION]

        # Run monolithic adapter
        monolithic_model = InferenceModel(
            export_dir=monolithic_export_dir,
            runner=SinglePass(),
        )
        with patch.object(np.random, "randn", side_effect=_zeros_like_randn):
            monolithic_outputs = monolithic_model(np_inputs)
        monolithic_actions = monolithic_outputs[ACTION]

        # Align shapes and compare
        min_action_dim = min(partitioned_actions.shape[-1], monolithic_actions.shape[-1])
        min_chunk = min(partitioned_actions.shape[-2], monolithic_actions.shape[-2])
        part_trimmed = partitioned_actions[:, :min_chunk, :min_action_dim]
        mono_trimmed = monolithic_actions[:, :min_chunk, :min_action_dim]

        max_diff = np.max(np.abs(part_trimmed - mono_trimmed))
        assert max_diff < 1e-3, f"Partitioned vs monolithic max diff: {max_diff}"
        print(f"Partitioned vs monolithic max diff: {max_diff}")
