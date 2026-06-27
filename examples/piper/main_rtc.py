from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from openpi_client import websocket_client_policy
import tyro

import main as piper_main

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    """Piper RTC control client arguments."""

    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None
    can_name: str = "can0"
    prompt: str = "抓起书本放到另外一个格子里"
    camera_backend: str = "realsense"
    head_camera_serial: str = "339322074804"
    wrist_camera_serial: str = "346522074547"
    head_camera: int | str = 0
    wrist_camera: int | str = 2
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    camera_timeout_ms: int = 1000
    camera_warmup_frames: int = 5
    show_cameras: bool = False
    camera_preview_window: str = "Piper cameras"
    camera_preview_scale: float = 1.5
    max_timesteps: int = 2000
    control_hz: float = piper_main.PIPER_CONTROL_FREQUENCY
    move_speed_percent: int = 30
    gripper_open_mm: float = 70.0
    gripper_closed_mm: float = 0.0
    gripper_threshold_mm: float = 35.0
    gripper_effort: int = 1000
    gripper_hold_close: bool = False
    gripper_close_trigger_mm: float = 25.0
    gripper_release_trigger_mm: float = 45.0
    gripper_hold_mm: float = 0.0
    image_size: int = 224
    enable_robot: bool = False
    dry_run: bool = True
    checkpoint: Path = piper_main.DEFAULT_CHECKPOINT


@dataclasses.dataclass(frozen=True)
class PolicyRequest:
    start_action_step: int
    start_time: float


class RealtimeActionStream:
    """Maintain an action stream while policy inference runs asynchronously.

    The dynamic RTC delay is the number of actions that were actually executed
    while a policy request was pending. This mirrors the real deployment
    meaning of inference delay: the robot does not pause for inference.
    """

    def __init__(self, initial_action_chunk: np.ndarray) -> None:
        self._current_actions = self._validate_chunk(initial_action_chunk)
        self._cursor = 0
        self.action_step = 0

    def start_request(self, *, action_step: int, start_time: float | None = None) -> PolicyRequest:
        return PolicyRequest(
            start_action_step=int(action_step),
            start_time=time.monotonic() if start_time is None else float(start_time),
        )

    def accept_response(self, request: PolicyRequest, action_chunk: np.ndarray) -> int:
        action_chunk = self._validate_chunk(action_chunk)
        delay_steps = self.action_step - request.start_action_step
        if delay_steps < 0:
            raise RuntimeError(
                f"Policy response start step {request.start_action_step} is ahead of action step {self.action_step}."
            )
        if delay_steps >= action_chunk.shape[0]:
            raise RuntimeError(
                f"Policy response arrived after {delay_steps} executed actions, but chunk length is "
                f"{action_chunk.shape[0]}."
            )
        self._current_actions = action_chunk
        self._cursor = delay_steps
        return delay_steps

    def next_action(self) -> np.ndarray:
        if self._cursor >= self._current_actions.shape[0]:
            raise RuntimeError(
                f"Ran out of actions at stream step {self.action_step}; policy inference is too slow for this chunk."
            )
        action = self._current_actions[self._cursor]
        self._cursor += 1
        self.action_step += 1
        return action

    @staticmethod
    def _validate_chunk(action_chunk: np.ndarray) -> np.ndarray:
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        if action_chunk.ndim != 2 or action_chunk.shape[1] != 7:
            raise ValueError(f"Expected action chunk shape (N, 7), got {action_chunk.shape}.")
        if action_chunk.shape[0] < 1:
            raise ValueError("Expected action chunk to contain at least one action.")
        return action_chunk


def _infer_action_chunk(
    policy_client: websocket_client_policy.WebsocketClientPolicy,
    request_data: dict[str, Any],
    *,
    expected_action_horizon: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    response = policy_client.infer(request_data)
    action_chunk = piper_main.validate_action_chunk(
        np.asarray(response["actions"], dtype=np.float32),
        expected_action_horizon=expected_action_horizon,
    )
    return action_chunk, response.get("server_timing", {})


def run(args: Args) -> None:
    if args.dry_run and args.enable_robot:
        raise ValueError("Use either dry_run=True or enable_robot=True; both together are ambiguous.")
    if args.gripper_hold_close and args.gripper_release_trigger_mm <= args.gripper_close_trigger_mm:
        raise ValueError("gripper_release_trigger_mm must be greater than gripper_close_trigger_mm.")
    if not min(args.gripper_open_mm, args.gripper_closed_mm) <= args.gripper_threshold_mm <= max(
        args.gripper_open_mm, args.gripper_closed_mm
    ):
        raise ValueError("gripper_threshold_mm must be between gripper_closed_mm and gripper_open_mm.")

    logger.info("Expected checkpoint: %s", args.checkpoint)
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port, api_key=args.api_key)
    server_metadata = policy_client.get_server_metadata()
    logger.info("Server metadata: %s", server_metadata)
    policy_action_horizon = piper_main.get_policy_action_horizon(server_metadata)
    if policy_action_horizon is not None:
        logger.info("Using policy action_horizon=%d from server metadata.", policy_action_horizon)
    else:
        logger.warning("Server metadata does not include action_horizon; will infer it from returned action chunks.")

    head_camera = piper_main.make_camera(
        backend=args.camera_backend,
        serial=args.head_camera_serial,
        opencv_id=args.head_camera,
        name="head",
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        timeout_ms=args.camera_timeout_ms,
        warmup_frames=args.camera_warmup_frames,
    )
    wrist_camera = piper_main.make_camera(
        backend=args.camera_backend,
        serial=args.wrist_camera_serial,
        opencv_id=args.wrist_camera,
        name="right wrist",
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        timeout_ms=args.camera_timeout_ms,
        warmup_frames=args.camera_warmup_frames,
    )
    visualizer = piper_main.CameraVisualizer(
        enabled=args.show_cameras,
        window_name=args.camera_preview_window,
        scale=args.camera_preview_scale,
    )
    arm = piper_main.PiperArm(
        args.can_name,
        enable_robot=args.enable_robot,
        dry_run=args.dry_run,
        move_speed_percent=args.move_speed_percent,
        gripper_open_mm=args.gripper_open_mm,
        gripper_closed_mm=args.gripper_closed_mm,
        gripper_threshold_mm=args.gripper_threshold_mm,
        gripper_effort=args.gripper_effort,
    )
    gripper_hold = piper_main.GripperHoldClose(
        enabled=args.gripper_hold_close,
        close_trigger_mm=args.gripper_close_trigger_mm,
        release_trigger_mm=args.gripper_release_trigger_mm,
        hold_mm=args.gripper_hold_mm,
    )
    control_period = 1.0 / args.control_hz

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pending_future: concurrent.futures.Future[tuple[np.ndarray, dict[str, Any]]] | None = None
    pending_request: PolicyRequest | None = None
    action_stream: RealtimeActionStream | None = None

    try:
        for step in range(args.max_timesteps):
            start = time.time()
            head_rgb = head_camera.read_rgb()
            wrist_rgb = wrist_camera.read_rgb()
            if not visualizer.show(head_rgb, wrist_rgb):
                logger.info("Camera preview requested shutdown.")
                break
            state = arm.read_state()
            request_data = piper_main.build_policy_observation(
                head_rgb,
                wrist_rgb,
                state,
                args.prompt,
                image_size=args.image_size,
            )

            if action_stream is None:
                with piper_main.prevent_keyboard_interrupt():
                    action_chunk, server_timing = _infer_action_chunk(
                        policy_client,
                        request_data,
                        expected_action_horizon=policy_action_horizon,
                    )
                if policy_action_horizon is None:
                    policy_action_horizon = int(action_chunk.shape[0])
                    logger.info("Inferred policy action_horizon=%d from returned action chunk.", policy_action_horizon)
                action_stream = RealtimeActionStream(action_chunk)
                logger.info("Received initial action chunk at step %d with shape %s timing=%s.", step, action_chunk.shape, server_timing)

            if pending_future is not None and pending_future.done():
                assert pending_request is not None
                action_chunk, server_timing = pending_future.result()
                inference_elapsed_ms = (time.monotonic() - pending_request.start_time) * 1000.0
                delay_steps = action_stream.accept_response(pending_request, action_chunk)
                logger.info(
                    "RTC accepted chunk at step %d: delay_steps=%d elapsed_ms=%.1f server_timing=%s.",
                    step,
                    delay_steps,
                    inference_elapsed_ms,
                    server_timing,
                )
                pending_future = None
                pending_request = None

            if pending_future is None:
                pending_request = action_stream.start_request(action_step=action_stream.action_step)
                pending_future = executor.submit(
                    _infer_action_chunk,
                    policy_client,
                    request_data,
                    expected_action_horizon=policy_action_horizon,
                )

            action = action_stream.next_action()
            action = gripper_hold.apply(action)
            arm.send_action(action)

            elapsed = time.time() - start
            if elapsed < control_period:
                time.sleep(control_period - elapsed)
    finally:
        if pending_future is not None:
            pending_future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        visualizer.close()
        arm.close()
        head_camera.close()
        wrist_camera.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    run(tyro.cli(Args))


if __name__ == "__main__":
    main()
