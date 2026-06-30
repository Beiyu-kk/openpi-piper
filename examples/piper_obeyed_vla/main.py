from __future__ import annotations

import dataclasses
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from openpi_client import websocket_client_policy
import tyro

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.piper import main as piper_main
from examples.piper_obeyed_vla import adapter

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args(piper_main.Args):
    """Piper control client with an OBEYED-VLA perception adapter."""

    perception_mode: str = "http"
    perception_endpoint: str = "http://127.0.0.1:18080/process"
    perception_timeout_s: float = 30.0
    perception_no_fallback: bool = False
    select_objects: str = ""
    exclude_objects: str = ""
    wrist_init_period: int = 15
    checkpoint: Path = piper_main.DEFAULT_CHECKPOINT


class GroundedObservationBuilder:
    def __init__(
        self,
        *,
        perception_client: adapter.PerceptionClient,
        prompt: str,
        image_size: int,
        select_objects: str,
        exclude_objects: str,
        wrist_init_period: int,
    ) -> None:
        self._perception_client = perception_client
        self._prompt = prompt
        self._image_size = int(image_size)
        self._select_objects = select_objects
        self._exclude_objects = exclude_objects
        self._wrist_init_period = max(1, int(wrist_init_period))
        self._cached_observation: dict[str, Any] | None = None
        self._has_initialized = False

    def build_if_replanning(
        self,
        *,
        should_replan: bool,
        step: int,
        head_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        state: np.ndarray,
    ) -> dict[str, Any]:
        if not should_replan and self._cached_observation is not None:
            return self._cached_observation

        perception_result = self._perception_client.process(
            select_objects=self._select_objects,
            exclude_objects=self._exclude_objects,
            is_base_init=not self._has_initialized,
            is_wrist_init=(not self._has_initialized or int(step) % self._wrist_init_period == 0),
            base_rgb=head_rgb,
            wrist_rgb=wrist_rgb,
        )
        if perception_result.used_fallback and perception_result.error:
            logger.warning("Using passthrough perception fallback: %s", perception_result.error)
        self._has_initialized = True
        self._cached_observation = piper_main.build_policy_observation(
            perception_result.base_rgb,
            perception_result.wrist_rgb,
            state,
            self._prompt,
            image_size=self._image_size,
        )
        return self._cached_observation


def run(args: Args) -> None:
    if args.open_loop_horizon < 1:
        raise ValueError("open_loop_horizon must be >= 1.")
    if args.dry_run and args.enable_robot:
        raise ValueError("Use either dry_run=True or enable_robot=True; both together are ambiguous.")

    perception_client = adapter.make_perception_client(
        mode=args.perception_mode,
        endpoint=args.perception_endpoint,
        timeout_s=args.perception_timeout_s,
        fallback_to_passthrough=not args.perception_no_fallback,
    )
    observation_builder = GroundedObservationBuilder(
        perception_client=perception_client,
        prompt=args.prompt,
        image_size=args.image_size,
        select_objects=args.select_objects,
        exclude_objects=args.exclude_objects,
        wrist_init_period=args.wrist_init_period,
    )

    logger.info("Expected checkpoint: %s", args.checkpoint)
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port, api_key=args.api_key)
    server_metadata = policy_client.get_server_metadata()
    logger.info("Server metadata: %s", server_metadata)
    policy_action_horizon = piper_main.get_policy_action_horizon(server_metadata)
    if policy_action_horizon is not None and args.open_loop_horizon > policy_action_horizon:
        raise ValueError(
            f"open_loop_horizon={args.open_loop_horizon} exceeds policy action_horizon={policy_action_horizon}."
        )

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

    actions_from_chunk_completed = 0
    pred_action_chunk: np.ndarray | None = None
    control_period = 1.0 / args.control_hz

    try:
        for step in range(args.max_timesteps):
            start = time.time()
            head_rgb = head_camera.read_rgb()
            wrist_rgb = wrist_camera.read_rgb()
            if not visualizer.show(head_rgb, wrist_rgb):
                logger.info("Camera preview requested shutdown.")
                break
            state = arm.read_state()

            should_replan = pred_action_chunk is None or actions_from_chunk_completed >= args.open_loop_horizon
            request_data = observation_builder.build_if_replanning(
                should_replan=should_replan,
                step=step,
                head_rgb=head_rgb,
                wrist_rgb=wrist_rgb,
                state=state,
            )

            if should_replan:
                actions_from_chunk_completed = 0
                with piper_main.prevent_keyboard_interrupt():
                    response = policy_client.infer(request_data)
                pred_action_chunk = piper_main.validate_action_chunk(
                    np.asarray(response["actions"], dtype=np.float32),
                    expected_action_horizon=policy_action_horizon,
                )
                if args.open_loop_horizon > pred_action_chunk.shape[0]:
                    raise ValueError(
                        f"open_loop_horizon={args.open_loop_horizon} exceeds returned action chunk length "
                        f"{pred_action_chunk.shape[0]}."
                    )
                if policy_action_horizon is None:
                    policy_action_horizon = int(pred_action_chunk.shape[0])
                    logger.info("Inferred policy action_horizon=%d from returned action chunk.", policy_action_horizon)
                logger.info("Received new grounded action chunk at step %d with shape %s.", step, pred_action_chunk.shape)

            action = pred_action_chunk[actions_from_chunk_completed]
            actions_from_chunk_completed += 1
            arm.send_action(gripper_hold.apply(action))

            elapsed = time.time() - start
            if elapsed < control_period:
                time.sleep(control_period - elapsed)
    finally:
        visualizer.close()
        arm.close()
        head_camera.close()
        wrist_camera.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    run(tyro.cli(Args))


if __name__ == "__main__":
    main()
