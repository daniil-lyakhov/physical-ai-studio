# From robot demonstrations to edge inference with ExecuTorch and OpenVINO

Record a task, train a policy, and put it back on the robot: Physical AI Studio connects the entire workflow. ExecuTorch packages the trained model for edge inference; its OpenVINO delegate runs supported parts of the model on Intel hardware. Here is the path from a pick-and-place demonstration to a supervised robot trial using the **Pi0.5** vision-language-action model as a concrete example.

## 1. Capture the task in Studio

Follow the [Studio installation guide](application/docs/01-installation.md), connect a leader and follower arm and cameras, then create an environment and dataset. Teleoperate the pick-and-place task, return the robot home at the end of each episode, and keep only clean demonstrations. Pi0.5 is language-conditioned, so give the dataset a specific task string such as “pick up the brown cube and place it in the black box.” Start with around 50 consistent episodes; the [recording guide](application/docs/05-recording-datasets.md) shows how to review camera footage and joint traces before training.

## 2. Train a policy

In **Models → Train model**, select that dataset and **Pi0.5**, then follow the loss curve and logs. Pi0.5 pulls its backbone from Hugging Face Hub, so configure a read-scoped token in **Settings → General → Hugging Face** first. Save the checkpoint and training settings, and test the policy on held-out trials before exporting.

## 3. Export with the OpenVINO delegate

Studio can automatically produce a **portable ExecuTorch** export after training on installations with the optional dependency. To use **ExecuTorch *with the OpenVINO delegate***, export the checkpoint explicitly through the Python API:

```python
from physicalai.policies import Pi05

policy = Pi05.load_from_checkpoint("/path/to/last.ckpt")
policy.eval()
policy.export(
    "./pi05-executorch-openvino",
    backend="executorch",
    delegate="openvino",
    delegate_config={"device": "GPU"},
)
```

This produces `pi05.pte` and `manifest.json`, which carries the policy's input/output contract — including the language input Pi0.5 expects at inference. `device="GPU"` targets the Intel XPU; use `"CPU"` on a machine without an Intel GPU. The OpenVINO delegate needs `physicalai-train[xpu,executorch-openvino]` **and** an ExecuTorch build with the OpenVINO partitioner — a standard ExecuTorch installation alone is not enough. See the [backend comparison notebook](library/notebooks/export/backend_comparison.ipynb) for the build workflow. Compare the exported policy's actions against the checkpoint on recorded observations, and measure latency before driving motors.

## 4. Run the exported policy on the robot

ExecuTorch is a first-class format in Studio. Open the model, pick the **ExecuTorch** export, and hit **Run model**: choose the environment, enter the task string, and Studio wires the cameras and the follower arm to the `.pte` policy for you. Our demo runs on an Intel XPU machine, so the OpenVINO delegate executes the model on the integrated GPU while Studio streams observations and action chunks in real time. Keep an operator on the stop control, watch the predicted actions before actuation, then run at a safe speed.

The same artifact loads from Python — `InferenceModel` picks the ExecuTorch backend from the `.pte` extension and reads the input/output contract from `manifest.json`:

```python
from physicalai.inference import InferenceModel

policy = InferenceModel("./pi05-executorch-openvino")

obs, info = env.reset()
done = False
while not done:
    action = policy.select_action(obs | {"task": "pick up the brown cube and place it in the black box"})
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

That is the full loop: **Studio → Pi0.5 → ExecuTorch with OpenVINO → robot**, one workflow from demonstration to edge inference on real hardware.


