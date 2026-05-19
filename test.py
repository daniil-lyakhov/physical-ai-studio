from physicalai.policies.pi05 import Pi05
from physicalai.benchmark import LiberoBenchmark
from physicalai.export.hooks import compress_weights_openvino_int8_sym
from physicalai.inference import InferenceModel


if __name__ == "__main__":
    policy = Pi05(
        pretrained_name_or_path="lerobot/pi05_libero_finetuned",
        n_action_steps=10,
        compile_model=False,
        dtype="float32",
    )
    export_dir = "pi05_libero_finetuned_int8_direct_export"
    policy.export(export_dir, backend="openvino", post_export_hooks=[compress_weights_openvino_int8_sym])

    benchmark = LiberoBenchmark(
        task_suite="libero_10",   # --env.task=libero_10
        #task_ids=[6],             # --env.task_ids=[6] — reproduces libero_10_6
        num_episodes=10,          # --env.num_episodes=10
        video_dir="fail_videos"
    )

    results = benchmark.evaluate(policy)
    print(results.summary())

    ov_model = InferenceModel(export_dir, device="GPU.0")
    ov_results = benchmark.evaluate(ov_model)
    print(ov_results.summary())
