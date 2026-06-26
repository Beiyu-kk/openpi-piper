import math
import types
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import main  # noqa: E402


def test_camera_preview_scale_defaults_to_larger_window():
    assert main.Args().camera_preview_scale == 1.5


def test_policy_action_horizon_is_read_from_flat_server_metadata():
    assert main.get_policy_action_horizon({"action_horizon": 30}) == 30


def test_policy_action_horizon_is_read_from_nested_server_metadata():
    metadata = {"model": {"action_horizon": 15}}

    assert main.get_policy_action_horizon(metadata) == 15


def test_policy_action_horizon_is_unknown_when_metadata_does_not_include_it():
    assert main.get_policy_action_horizon({"reset_pose": [0.0]}) is None


def test_validate_action_chunk_accepts_metadata_horizon():
    chunk = np.zeros((30, 7), dtype=np.float32)

    validated = main.validate_action_chunk(chunk, expected_action_horizon=30)

    assert validated is chunk


def test_validate_action_chunk_accepts_unknown_horizon_from_returned_shape():
    chunk = np.zeros((15, 7), dtype=np.float32)

    validated = main.validate_action_chunk(chunk, expected_action_horizon=None)

    assert validated is chunk


def test_validate_action_chunk_rejects_chunk_that_differs_from_metadata_horizon():
    chunk = np.zeros((15, 7), dtype=np.float32)

    with pytest.raises(RuntimeError, match=r"Expected action chunk shape \(30, 7\), got \(15, 7\)"):
        main.validate_action_chunk(chunk, expected_action_horizon=30)


def test_joint_radians_to_sdk_units_converts_to_millidegrees():
    joints = np.array([0.0, math.pi / 2, -math.pi / 2, math.pi, -math.pi, 0.123], dtype=np.float32)

    units = main.joint_radians_to_sdk_units(joints)

    assert units == [0, 90_000, -90_000, 180_000, -180_000, 7047]


def test_gripper_action_to_sdk_units_preserves_continuous_meter_actions():
    assert main.gripper_action_to_sdk_units(0.0, open_mm=70.0, closed_mm=0.0) == 0
    assert main.gripper_action_to_sdk_units(0.035, open_mm=70.0, closed_mm=0.0) == 35_000
    assert main.gripper_action_to_sdk_units(0.07, open_mm=70.0, closed_mm=0.0) == 70_000


def test_gripper_action_to_sdk_units_clips_to_configured_mm_range():
    assert main.gripper_action_to_sdk_units(-0.01, open_mm=70.0, closed_mm=3.0) == 3_000
    assert main.gripper_action_to_sdk_units(0.09, open_mm=65.5, closed_mm=3.25) == 65_500


def test_gripper_sdk_units_to_meters_matches_training_state_units():
    assert main.gripper_sdk_units_to_meters(70_000) == pytest.approx(0.07)


def test_build_policy_observation_uses_piper_training_keys_and_float32_state():
    head = np.zeros((240, 320, 3), dtype=np.uint8)
    wrist = np.ones((240, 320, 3), dtype=np.uint8)
    state = np.arange(7, dtype=np.float64)

    observation = main.build_policy_observation(head, wrist, state, "抓起书本")

    assert set(observation) == {
        "observation/image",
        "observation/right_wrist_image",
        "observation/state",
        "prompt",
    }
    assert observation["observation/image"].shape == (224, 224, 3)
    assert observation["observation/right_wrist_image"].shape == (224, 224, 3)
    assert observation["observation/image"].dtype == np.uint8
    assert observation["observation/state"].dtype == np.float32
    np.testing.assert_array_equal(observation["observation/state"], np.arange(7, dtype=np.float32))
    assert observation["prompt"] == "抓起书本"


def test_camera_visualizer_disabled_is_noop():
    visualizer = main.CameraVisualizer(enabled=False, window_name="unused", scale=1.0)

    assert visualizer.show(np.zeros((1, 1, 3), dtype=np.uint8), np.zeros((1, 1, 3), dtype=np.uint8))
    visualizer.close()


def test_camera_visualizer_shows_two_rgb_views_side_by_side_as_bgr(monkeypatch):
    calls = []

    class FakeCV2:
        WINDOW_NORMAL = 0

        def namedWindow(self, *args):
            calls.append(("namedWindow", args))

        def resizeWindow(self, *args):
            calls.append(("resizeWindow", args))

        def resize(self, image, size):
            width, height = size
            calls.append(("resize", size))
            return np.zeros((height, width, 3), dtype=image.dtype)

        def imshow(self, *args):
            calls.append(("imshow", args))

        def waitKey(self, delay):
            calls.append(("waitKey", delay))
            return ord("q")

        def destroyWindow(self, *args):
            calls.append(("destroyWindow", args))

    monkeypatch.setitem(sys.modules, "cv2", FakeCV2())
    visualizer = main.CameraVisualizer(enabled=True, window_name="Piper cameras", scale=2.0)
    head_rgb = np.array([[[255, 0, 0]]], dtype=np.uint8)
    wrist_rgb = np.array([[[0, 255, 0]]], dtype=np.uint8)

    should_continue = visualizer.show(head_rgb, wrist_rgb)
    visualizer.close()

    assert not should_continue
    assert calls[0] == ("namedWindow", ("Piper cameras", 0))
    assert calls[1] == ("resize", (4, 2))
    assert calls[2] == ("resizeWindow", ("Piper cameras", 4, 2))
    window_name, preview_bgr = calls[3][1]
    assert window_name == "Piper cameras"
    np.testing.assert_array_equal(
        preview_bgr,
        np.zeros((2, 4, 3), dtype=np.uint8),
    )
    assert calls[4] == ("waitKey", 1)
    assert calls[5] == ("destroyWindow", ("Piper cameras",))


def test_camera_visualizer_resizes_window_only_when_preview_size_changes(monkeypatch):
    calls = []

    class FakeCV2:
        WINDOW_NORMAL = 0

        def namedWindow(self, *args):
            calls.append(("namedWindow", args))

        def resizeWindow(self, *args):
            calls.append(("resizeWindow", args))

        def resize(self, image, size):
            width, height = size
            calls.append(("resize", size))
            return np.zeros((height, width, 3), dtype=image.dtype)

        def imshow(self, *args):
            calls.append(("imshow", args))

        def waitKey(self, delay):
            calls.append(("waitKey", delay))
            return -1

        def destroyWindow(self, *args):
            calls.append(("destroyWindow", args))

    monkeypatch.setitem(sys.modules, "cv2", FakeCV2())
    visualizer = main.CameraVisualizer(enabled=True, window_name="Piper cameras", scale=2.0)
    head_rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    wrist_rgb = np.zeros((1, 1, 3), dtype=np.uint8)

    assert visualizer.show(head_rgb, wrist_rgb)
    assert visualizer.show(head_rgb, wrist_rgb)
    visualizer.close()

    resize_window_calls = [call for call in calls if call[0] == "resizeWindow"]
    assert resize_window_calls == [("resizeWindow", ("Piper cameras", 4, 2))]


def test_make_camera_uses_realsense_serials_when_backend_is_realsense(monkeypatch):
    created = []

    class FakeRealSenseCamera:
        def __init__(self, serial, **kwargs):
            created.append((serial, kwargs))

    monkeypatch.setattr(main, "RealSenseCamera", FakeRealSenseCamera)

    camera = main.make_camera(
        backend="realsense",
        serial="346522074547",
        opencv_id=2,
        name="right wrist",
        width=640,
        height=480,
        fps=30,
        timeout_ms=1000,
        warmup_frames=3,
    )

    assert isinstance(camera, FakeRealSenseCamera)
    assert created == [
        (
            "346522074547",
            {
                "name": "right wrist",
                "width": 640,
                "height": 480,
                "fps": 30,
                "timeout_ms": 1000,
                "warmup_frames": 3,
            },
        )
    ]


def test_realsense_camera_converts_bgr_color_frame_to_rgb(monkeypatch):
    class FakeColorFrame:
        def get_data(self):
            return np.array([[[1, 2, 3]]], dtype=np.uint8)

    class FakeFrameSet:
        def get_color_frame(self):
            return FakeColorFrame()

    class FakePipeline:
        def start(self, _config):
            return None

        def wait_for_frames(self, timeout_ms):
            assert timeout_ms == 123
            return FakeFrameSet()

        def stop(self):
            return None

    class FakeConfig:
        def enable_device(self, serial):
            assert serial == "339322074804"

        def enable_stream(self, *args):
            assert args == ("color", 640, 480, "bgr8", 30)

    fake_rs = types.SimpleNamespace(
        pipeline=FakePipeline,
        config=FakeConfig,
        stream=types.SimpleNamespace(color="color"),
        format=types.SimpleNamespace(bgr8="bgr8"),
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)

    camera = main.RealSenseCamera(
        "339322074804",
        name="head",
        width=640,
        height=480,
        fps=30,
        timeout_ms=123,
        warmup_frames=0,
    )

    np.testing.assert_array_equal(camera.read_rgb(), np.array([[[3, 2, 1]]], dtype=np.uint8))
