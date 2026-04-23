# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pytest tests for partitioned Pi0.5 OpenVINO export.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

_DEFAULT_EXPORT_CACHE_DIR = Path(__file__).resolve().parent / ".export_cache" / "partitioned"
_MODEL_SEED = 42
_CHUNK_SIZE = 10
_MAX_ACTION_DIM = 32
_NUM_INFERENCE_STEPS = 3
_DEVICE = "cpu"


@pytest.fixture(scope="module")
def pi05_model():
    """Create a standalone Pi05Model with dummy dataset stats.

    Uses a fixed seed so that random weights are reproducible across runs,
    which is required for reusing cached OV exports.
    """
    print("Start pi0.5 pytorch model init...")
    from physicalai.policies.pi05.model import Pi05Model

    torch.manual_seed(_MODEL_SEED)

    dataset_stats = {
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
        "action": {
            "shape": (14,),
            "mean": [0.0] * 14,
            "std": [1.0] * 14,
            "name": "action",
        },
    }

    model = Pi05Model(
        dataset_stats,
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


def _make_reference_inputs(batch_size: int = 1) -> dict[str, np.ndarray]:
    """Create deterministic model inputs."""
    np.random.seed(_REFERENCE_INPUT_SEED)
    return {
        "images": np.random.randn(batch_size, 3, 224, 224).astype(np.float32),
        "image_masks": np.ones((batch_size,), dtype=bool),
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
    cache_file = _DEFAULT_EXPORT_CACHE_DIR / "pytorch_reference.npz"
    inputs = _make_reference_inputs()

    force = request.config.getoption("--force-reexport", default=False)
    if force or not cache_file.exists():
        print("Original model output regeneration starts...")
        pi05_model = request.getfixturevalue("pi05_model")
        noise = torch.zeros(
            1, _CHUNK_SIZE, _MAX_ACTION_DIM,
            dtype=torch.float32, device=_DEVICE,
        )
        with torch.no_grad():
            pytorch_actions = pi05_model.sample_actions(
                [torch.from_numpy(inputs["images"]).to(_DEVICE)],
                [torch.from_numpy(inputs["image_masks"]).to(_DEVICE)],
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


class TestPytorchWrappers:
    """Test 1: PyTorch wrapper modules reproduce original sample_actions."""

    def test_wrappers_match_original(self, pi05_model):
        from physicalai.policies.pi05.partitioned_export import (
            ActionOutputHead,
            GemmaExpertDecoder,
            PaliGemmaEncoder,
        )

        device = _DEVICE
        batch_size = 1

        image = torch.randn(batch_size, 3, 224, 224, device=device)
        img_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        tokens = torch.randint(0, 1000, (batch_size, 200), device=device)
        masks = torch.ones(batch_size, 200, dtype=torch.bool, device=device)
        noise = torch.zeros(batch_size, _CHUNK_SIZE, _MAX_ACTION_DIM,
                            dtype=torch.float32, device=device)

        with torch.no_grad():
            original = pi05_model.sample_actions([image], [img_mask], tokens, masks, noise=noise)

            encoder = PaliGemmaEncoder(pi05_model).eval()
            decoder = GemmaExpertDecoder(pi05_model).eval()
            head = ActionOutputHead(pi05_model).eval()

            cache_keys, cache_values, prefix_pad_masks = encoder(image, img_mask, tokens, masks)
            x_t = torch.zeros_like(noise)
            dt = -1.0 / _NUM_INFERENCE_STEPS

            for step in range(_NUM_INFERENCE_STEPS):
                t = 1.0 + step * dt
                time_tensor = torch.tensor(t, dtype=torch.float32, device=device).expand(batch_size)
                suffix_out = decoder(x_t, time_tensor, cache_keys, cache_values, prefix_pad_masks)
                v_t = head(suffix_out)
                x_t = x_t + dt * v_t

        assert torch.max(torch.abs(original - x_t)).item() < 1e-4


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
        batch_size = inputs["image"].shape[0]

        outputs = adapter.predict(inputs)
        assert "action" in outputs
        action = outputs["action"]
        assert action.shape == (batch_size, _CHUNK_SIZE, _MAX_ACTION_DIM)

        assert np.max(np.abs(action - pytorch_actions)) < 1e-3


class TestInferenceModelPipeline:
    """Test 3: InferenceModel + ActionChunking runner."""

    # Dlyakhov: not sure about this exact test
    def test_select_action_and_reset(self, pi05_reference, export_dir):
        from physicalai.inference.adapters.openvino_partitioned import PartitionedOpenVINOAdapter
        from physicalai.inference.model import InferenceModel
        from physicalai.inference.runners.action_chunking import ActionChunking
        from physicalai.inference.runners.single_pass import SinglePass

        adapter = PartitionedOpenVINOAdapter(
            device="CPU",
            num_inference_steps=_NUM_INFERENCE_STEPS,
            chunk_size=_CHUNK_SIZE,
            max_action_dim=_MAX_ACTION_DIM,
        )
        adapter.load(export_dir)

        model = InferenceModel.__new__(InferenceModel)
        model.adapter = adapter
        model.runner = ActionChunking(SinglePass(), chunk_size=_CHUNK_SIZE)
        model.preprocessors = []
        model.postprocessors = []
        model.callbacks = []
        model.metadata = {}

        obs = pi05_reference["inputs"]
        batch_size = obs["images"].shape[0]

        model.reset()
        action_1 = model.select_action(obs)
        assert action_1.shape == (batch_size, _MAX_ACTION_DIM)

        action_2 = model.select_action(obs)
        assert not np.allclose(action_1, action_2)

        for _ in range(_CHUNK_SIZE - 2):
            model.select_action(obs)

        model.reset()
        action_post_reset = model.select_action(obs)
        assert action_post_reset.shape == (batch_size, _MAX_ACTION_DIM)

class TestQuantizedPartitionedModel:
    """Test 4: Weight-compressed (INT8) partitioned OV model vs PyTorch."""

    def test_nncf_weight_compression_matches_pytorch(self, pi05_reference, export_dir):
        """Compress each partition to INT8 via NNCF and compare with PyTorch."""
        import shutil

        import nncf
        import openvino

        from physicalai.inference.adapters.openvino_partitioned import PartitionedOpenVINOAdapter

        core = openvino.Core()
        compressed_dir = export_dir.resolve().parent / "compressed"

        model_files = {
            "paligemma_encoder": export_dir / "paligemma_encoder.xml",
            "gemma_expert_decoder": export_dir / "gemma_expert_decoder.xml",
            "action_output_head": export_dir / "action_output_head.xml",
        }

        for name, xml_path in model_files.items():
            ov_model = core.read_model(str(xml_path))
            if name == "action_output_head":
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
        batch_size = inputs["images"].shape[0]

        outputs = adapter.predict(inputs)
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
        assert int8_size < fp32_size, (
            f"Compressed size ({int8_size}) should be smaller than original ({fp32_size})"
        )
        print(f"File compression ratio: {fp32_size / int8_size}")