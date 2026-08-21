#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import signal
from pathlib import Path

import numpy as np
from PIL import Image

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


class DiagRecorder:
    """Capture ground-truth deploy data to locate the shaking source.

    Records, with NO effect on control:
      * per-inference: the raw model action chunk (BEFORE smoothing) -> tells us
        whether the shaking is already in the model output or added by the
        queue/smoother.
      * per-tick: the live state fed to the model and the action actually SENT to
        the robot -> the commanded signal; high-frequency wiggle here == shaking.
      * the first few full observations (per-camera images + state) -> replay them
        offline through the SAME InferenceModel to prove the model behaves on REAL
        obs (and to eyeball which physical camera is which / orientation).

    Dump to ``out_dir`` on ``stop()``; analyze offline.
    """

    def __init__(self, out_dir: str = "xr0_diag", n_png: int = 3, max_obs: int = 250) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        # ``n_png``: how many observations to also dump as human-viewable PNGs.
        # ``max_obs``: how many CONSECUTIVE full observations (state + camera
        # images) to persist as ``obs_*.npz`` so they can be replayed OFFLINE
        # through several models. Bounded so disk stays sane (~1.8 MB/obs).
        self.n_png = n_png
        self.max_obs = max_obs
        self._saved = 0
        self.ticks: list[dict] = []
        self.chunks: list[dict] = []

    def on_inference(self, event: object) -> None:
        chunk = np.asarray(getattr(event, "chunk"), dtype=np.float32)
        self.chunks.append(
            {
                "t": float(getattr(event, "timestamp", 0.0)),
                "latency_s": float(getattr(event, "latency_s", 0.0)),
                "offset": int(getattr(event, "offset", 0)),
                "chunk": chunk.tolist(),
            }
        )

    def on_tick(self, event: object) -> None:
        state = np.asarray(getattr(event.robot_state, "state", []), dtype=np.float32)
        sent = getattr(event, "action_sent", None)
        sent = None if sent is None else np.asarray(sent, dtype=np.float32).tolist()
        self.ticks.append({"step": int(event.step), "state": state.tolist(), "action_sent": sent})

        if self._saved < self.max_obs:
            imgs = {name: np.asarray(frame.data) for name, frame in event.camera_frames.items()}
            np.savez(
                self.dir / f"obs_{event.step:05d}.npz",
                state=state,
                **{f"img__{k}": v for k, v in imgs.items()},
            )
            if self._saved < self.n_png:
                for name, arr in imgs.items():
                    try:
                        Image.fromarray(arr.astype(np.uint8)).save(self.dir / f"obs_{event.step:05d}__{name}.png")
                    except Exception:  # noqa: BLE001, S110
                        pass
            self._saved += 1

    def stop(self) -> None:
        np.savez(
            self.dir / "ticks.npz",
            steps=np.array([t["step"] for t in self.ticks], dtype=np.int64),
            states=np.array([t["state"] for t in self.ticks], dtype=np.float32),
            actions_sent=np.array(
                [t["action_sent"] if t["action_sent"] is not None else [np.nan] * 6 for t in self.ticks],
                dtype=np.float32,
            ),
        )
        (self.dir / "chunks.json").write_text(json.dumps(self.chunks))
        # Per-joint high-frequency wiggle of the COMMANDED signal (frame-to-frame
        # |Δ|). Big values on some joints == that joint shakes; this localizes the
        # bug (one joint => mapping/units; all joints => replan/timing/model).
        sent = np.array(
            [t["action_sent"] for t in self.ticks if t["action_sent"] is not None], dtype=np.float32
        )
        if len(sent) > 2:
            jitter = np.abs(np.diff(sent, axis=0)).mean(axis=0)
            print(f"[DIAG] mean |Δaction/frame| per joint: {np.round(jitter, 3)}")
        # How much consecutive PLANS disagree on their immediate target (model-side
        # oscillation, independent of the smoother).
        if len(self.chunks) > 1:
            firsts = np.array([c["chunk"][0][:6] for c in self.chunks], dtype=np.float32)
            print(f"[DIAG] std of chunk[0] across replans per joint: {np.round(firsts.std(axis=0), 3)}")
        print(f"[DIAG] wrote {len(self.ticks)} ticks, {len(self.chunks)} chunks, {self._saved} obs to {self.dir}/")



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
        callbacks=[recorder := DiagRecorder(out_dir=f"diag_{Path(MODEL_PATH).parent.name or Path(MODEL_PATH).name}")],
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
            recorder.stop()
            print("Disabling torque...")
            robot.set_torque(enabled=False)


if __name__ == "__main__":
    main()