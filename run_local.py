#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import signal

from physicalai.inference import InferenceModel
from physicalai.runtime import (
    AsyncExecution,
    PolicySource,
    ChunkedActionQueue,
    LerpSmoother,
    RobotRuntime,
)

from physicalai.robot import SO101
from physicalai.capture import UVCCamera


MODEL_PATH = "/home/dupeljan/Projects/home_robot/models/act_hunh_40k"
MODEL_PATH  = "/home/dupeljan/Projects/home_robot/models/act_yellow_ball/export_openvino"
MODEL_PATH  = "/home/dupeljan/Projects/home_robot/models/xr0_put_balls_to_box_irs"
MODEL_PATH  = "/home/dupeljan/Projects/home_robot/models/pi05_balls/export_openvino"
MODEL_PATH  = "/home/dupeljan/Projects/home_robot/models/xr0_v1/xr0_put_balls_to_box_irs_normalized_full"


def main() -> None:
    # Force-exit on second Ctrl+C (Rerun's blocked channels prevent clean shutdown)
    def _handle_sigint(sig: int, frame: object) -> None:
        # Restore default handler so next Ctrl+C kills immediately via OS signal
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\nInterrupting... press Ctrl+C again to force kill.")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_sigint)

    # ── Load model ──

    dev = "GPU.0"

    adapter_kwargs = {"INFERENCE_PRECISION_HINT": "f16"} if dev == "GPU.0" else {}
    model = InferenceModel(MODEL_PATH, device=dev, **adapter_kwargs)
    #model = InferenceModel(MODEL_PATH, device=dev)

    # ── Build robot & cameras ──
    robot = SO101(port="/dev/ttyACM0", 
                  calibration="/home/dupeljan/.cache/huggingface/lerobot/calibration/robots/so_follower/black_follower_arm.json",
                  role="follower")


    cameras = {
        "top_camera": UVCCamera(device="/dev/video0"),
        "pov_black_follower_camera": UVCCamera(device="/dev/video2"),
    }

    # ── Run ──
    policy_source = PolicySource(
        model=model,
        execution=AsyncExecution(request_threshold=0.1), # SyncExecution(request_threshold=0.1)
        action_queue=ChunkedActionQueue(smoother=LerpSmoother(duration_frames=5)),
        task="Put the green ball to the black box",
    )
    runtime = RobotRuntime(
        robot=robot,
        action_source=policy_source,
        cameras=cameras,
        fps=30,
        #callbacks=callbacks,
    )

    duration_s = 120
    with runtime:
        # Do not re-hold (re-enable torque) when the runtime disconnects on exit,
        # so the torque-off we issue below stays in effect after Ctrl+C / episode end.
        # Warning: the arm will drop under gravity once torque is released.
        robot.torque_on_disconnect = False
        try:
            for name, cam in cameras.items():
                w = getattr(cam, "actual_width", None)
                h = getattr(cam, "actual_height", None)
                f = getattr(cam, "actual_fps", None)
                print(f"  {name}: {w}x{h} @ {f}fps" if w and h else f"  {name}: connected")
            print(f"Running at 30 fps for {duration_s}s...")
            stats = runtime.run(duration_s=duration_s)
            #print(f"\nDone — {stats.steps} steps, {stats.inference_count} inferences, {stats.total_holds} holds")
        finally:
            # Runs on normal completion and on Ctrl+C (KeyboardInterrupt) while the
            # robot is still connected, so torque is actually released on the servos.
            print("Disabling torque...")
            robot.set_torque(enabled=False)


if __name__ == "__main__":
    main()