# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Build shared synthetic inputs for the XR0 source-vs-framework parity test.

Run once (under either environment that has the Qwen3-VL processor cached) to
produce a single ``inputs.pt`` consumed *unchanged* by both runners. Building
the VLM inputs once guarantees the two implementations see bit-identical
``input_ids`` / ``pixel_values`` / ``image_grid_thw`` -- the only difference
then measured is the two model implementations themselves.

A batch of ``num_samples`` samples is produced (each with its own random image
pair, ``state`` and rectified-flow ``noise``) so the parity test can compare
output statistics across a distribution rather than a single point. The
per-sample initial noise is baked in here (``noise``) and injected into
``torch.randn_like`` by the runner, removing the sole source of nondeterminism
in the inference path. Every sample uses the same instruction and image size so
the tokenized sequences share a length and batch without padding.

Two extra fields feed the training-iteration parity runner and are ignored by
the inference runner: ``action_target`` (the ground-truth action the flow
matches against) and ``timestep`` (the pinned rectified-flow interpolation
coefficient in ``(0, 1)``). Baking these in makes the training forward/backward
fully deterministic across the two environments, whose different torch versions
cannot be seeded to draw identical timesteps.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

PROCESSOR_NAME = "Qwen/Qwen3-VL-4B-Instruct"
SEED = 0
IMAGE_SIZE = 256
NUM_VIEWS = 2
NUM_SAMPLES = 32
STATE_DIM = 32
STATE_LEN = 1
ACTION_DIM = 32
ACTION_LEN = 30
INSTRUCTION = "pick up the black bowl and place it on the plate"


def build_inputs(
    num_samples: int = NUM_SAMPLES, processor_name: str = PROCESSOR_NAME
) -> dict[str, torch.Tensor]:
    """Assemble a deterministic batch of synthetic XR0 model inputs (float32, CPU).

    Args:
        num_samples: Number of samples in the batch.
        processor_name: HuggingFace id of the Qwen3-VL processor.

    Returns:
        Dict with batched VLM inputs (``input_ids``, ``attention_mask``,
        ``pixel_values``, ``image_grid_thw``), per-sample synthetic ``state``, an
        ``action`` placeholder (shape-only), an all-ones ``action_mask``, the
        pinned per-sample rectified-flow ``noise``, the ground-truth
        ``action_target`` and the pinned ``timestep`` (both training-only).
    """
    from transformers import AutoProcessor

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    processor = AutoProcessor.from_pretrained(processor_name)

    conversations = []
    for _ in range(num_samples):
        images = [
            Image.fromarray(rng.integers(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
            for _ in range(NUM_VIEWS)
        ]
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": INSTRUCTION})
        conversations.append([{"role": "user", "content": content}])

    encoded = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )

    return {
        "input_ids": encoded["input_ids"].to(torch.long),
        "attention_mask": encoded["attention_mask"].to(torch.long),
        "pixel_values": encoded["pixel_values"].to(torch.float32),
        "image_grid_thw": encoded["image_grid_thw"].to(torch.long),
        "state": torch.randn(num_samples, STATE_LEN, STATE_DIM, dtype=torch.float32),
        "action": torch.zeros(num_samples, ACTION_LEN, ACTION_DIM, dtype=torch.float32),
        "action_mask": torch.ones(num_samples, ACTION_LEN, ACTION_DIM, dtype=torch.int32),
        "noise": torch.randn(num_samples, ACTION_LEN, ACTION_DIM, dtype=torch.float32),
        # Training-only: ground-truth action the flow regresses toward and the
        # pinned rectified-flow timestep (kept strictly inside (0, 1)).
        "action_target": torch.randn(num_samples, ACTION_LEN, ACTION_DIM, dtype=torch.float32),
        "timestep": torch.rand(num_samples, dtype=torch.float32).clamp(0.01, 0.99),
    }


def main() -> None:
    """CLI entry point: write the synthetic inputs to ``--out``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Destination .pt path.")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--processor", default=PROCESSOR_NAME)
    args = parser.parse_args()

    inputs = build_inputs(args.num_samples, args.processor)
    torch.save(inputs, args.out)
    shapes = {key: tuple(value.shape) for key, value in inputs.items()}
    print(f"[build_inputs] wrote {args.out} ({args.num_samples} samples)")
    print(f"[build_inputs] shapes: {shapes}")


if __name__ == "__main__":
    main()
