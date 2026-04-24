from physicalai.policies.pi05 import Pi05
from physicalai.benchmark import LiberoBenchmark
from physicalai.inference import InferenceModel
from physicalai.data import Observation
import torch
from physicalai.inference.runners import ActionChunking, SinglePass



class InferenceModelPolicyWrapper:
    def __init__(self, inf_model: InferenceModel) -> None:
        self._inf_model = inf_model

    def select_action(self, observation: Observation) -> torch.Tensor:
        np_obs = observation.to_numpy().to_dict(flatten=False)
        # Filter out None values and convert to format expected by InferenceModel
        np_inputs = {k: v for k, v in np_obs.items() if v is not None}
        action = self._inf_model.select_action(np_inputs)
        return torch.from_numpy(action)

    def reset(self) -> None:
        """Reset policy state for new episode."""
        self._inf_model.reset()


if __name__ == "__main__":
    # Load from HuggingFace (equivalent to your lerobot command)
    if False:
        policy = Pi05(
            pretrained_name_or_path="lerobot/pi05_libero_finetuned",
            n_action_steps=10,       # same as --policy.n_action_steps=10
            compile_model=False,
            dtype="bfloat16",
        )
        policy.to_openvino_partitioned("pi05_libero_finetuned_hf_ov")

    # Evaluate on LIBERO (equivalent to your lerobot eval)
    benchmark = LiberoBenchmark(
        task_suite="libero_10",   # --env.task=libero_10
        task_ids=[6],             # --env.task_ids=[6] — reproduces libero_10_6
        num_episodes=1,          # --env.num_episodes=10
        max_steps=500,            # match the 500 steps seen in the failing run
    )

    #policy.to("xpu")
    #results = benchmark.evaluate(policy)
    #policy.to("cpu")
    #print(results.summary())

    ov_model = InferenceModel("pi05_libero_finetuned_hf_ov", device="CPU", runner=ActionChunking(SinglePass()))
    ov_policy = InferenceModelPolicyWrapper(ov_model)
    ov_results = benchmark.evaluate(ov_policy)
    print(ov_results.summary())