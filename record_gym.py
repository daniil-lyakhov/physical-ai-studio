"""Record LIBERO gym episodes with a partitioned OpenVINO model."""

import argparse
from pathlib import Path

import numpy as np
import torch

from physicalai.data import Observation
from physicalai.eval.rollout.functional import rollout
from physicalai.eval.video import VideoRecorder
from physicalai.gyms import LiberoGym
from physicalai.inference import InferenceModel
from physicalai.inference.runners import ActionChunking, SinglePass


class InferenceModelPolicyWrapper:
    def __init__(self, inf_model: InferenceModel) -> None:
        self._inf_model = inf_model

    def select_action(self, observation: Observation) -> torch.Tensor:
        np_obs = observation.to_numpy().to_dict(flatten=False)
        np_inputs = {k: v for k, v in np_obs.items() if v is not None}
        action = self._inf_model.select_action(np_inputs)
        return torch.from_numpy(action)

    def reset(self) -> None:
        self._inf_model.reset()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record LIBERO gym episodes as video.")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to partitioned OV model directory")
    parser.add_argument("--task-suite", type=str, default="libero_10", help="LIBERO task suite name")
    parser.add_argument("--task-id", type=int, default=0, help="Task ID within suite")
    parser.add_argument("--output-dir", type=str, default="./recordings", help="Video output directory")
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to record")
    parser.add_argument("--max-steps", type=int, default=520, help="Max steps per episode")
    parser.add_argument("--device", type=str, default="CPU", help="OpenVINO device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_dir} on {args.device}...")
    inf_model = InferenceModel(args.model_dir, device=args.device, runner=ActionChunking(SinglePass()))
    policy = InferenceModelPolicyWrapper(inf_model)

    # Create gym
    print(f"Creating gym: {args.task_suite} task {args.task_id}")
    gym = LiberoGym(task_suite=args.task_suite, task_id=args.task_id)

    # Create video recorder
    output_dir = Path(args.output_dir) / f"{args.task_suite}_task{args.task_id}"
    recorder = VideoRecorder(output_dir=output_dir, record_mode="all")

    # Run episodes
    successes = 0
    for ep in range(args.num_episodes):
        seed = args.seed + ep
        recorder.start_episode(f"episode_{ep:03d}")

        result = rollout(
            gym,
            policy,
            seed=seed,
            max_steps=args.max_steps,
            video_recorder=recorder,
        )

        success = bool(result.get("success", np.array([False])).item())
        video_path = recorder.finish_episode(success=success)
        successes += int(success)

        print(
            f"  Episode {ep}: {'SUCCESS' if success else 'FAILURE'} | "
            f"steps={result['episode_length']} | fps={result.get('fps', 0):.1f} | "
            f"saved: {video_path}"
        )

    # Summary
    gym.close()
    print(f"\nDone: {successes}/{args.num_episodes} successful ({100 * successes / args.num_episodes:.0f}%)")
    print(f"Videos saved to: {output_dir}")


if __name__ == "__main__":
    main()
