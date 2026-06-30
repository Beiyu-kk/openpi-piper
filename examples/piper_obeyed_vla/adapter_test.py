from __future__ import annotations

import io
import json
from pathlib import Path
from urllib.error import URLError

import cv2
import numpy as np
import pytest

from examples.piper_obeyed_vla import adapter
from examples.piper_obeyed_vla import convert_lerobot_dataset


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, body: bytes, headers: dict[str, str]) -> None:
        self._body = body
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_passthrough_client_returns_original_images() -> None:
    head = np.full((4, 5, 3), 11, dtype=np.uint8)
    wrist = np.full((4, 5, 3), 29, dtype=np.uint8)

    client = adapter.PassthroughPerceptionClient()
    result = client.process(
        select_objects="book",
        exclude_objects="",
        is_base_init=True,
        is_wrist_init=True,
        base_rgb=head,
        wrist_rgb=wrist,
    )

    assert np.array_equal(result.base_rgb, head)
    assert np.array_equal(result.wrist_rgb, wrist)
    assert result.used_fallback is True
    assert result.error is None


def test_http_client_posts_two_npy_files_and_parses_multipart_response() -> None:
    head = np.full((2, 3, 3), 7, dtype=np.uint8)
    wrist = np.full((2, 3, 3), 9, dtype=np.uint8)
    head_out = np.full((2, 3, 3), 17, dtype=np.uint8)
    wrist_out = np.full((2, 3, 3), 19, dtype=np.uint8)
    seen = {}

    boundary = "overlayboundary"
    body = (
        f"--{boundary}\r\n"
        "Content-Type: application/octet-stream\r\n"
        "Content-Disposition: attachment; filename=overlay_base.npy\r\n\r\n"
    ).encode() + _npy_bytes(head_out) + (
        f"\r\n--{boundary}\r\n"
        "Content-Type: application/octet-stream\r\n"
        "Content-Disposition: attachment; filename=overlay_wrist.npy\r\n\r\n"
    ).encode() + _npy_bytes(wrist_out) + f"\r\n--{boundary}--\r\n".encode()

    def opener(request, timeout):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        seen["body"] = request.data
        seen["content_type"] = request.headers["Content-type"]
        return _FakeResponse(body, {"Content-Type": f"multipart/mixed; boundary={boundary}"})

    client = adapter.HttpPerceptionClient(
        endpoint="http://127.0.0.1:18080/process",
        timeout_s=2.5,
        opener=opener,
    )
    result = client.process(
        select_objects="book",
        exclude_objects="shelf",
        is_base_init=True,
        is_wrist_init=False,
        base_rgb=head,
        wrist_rgb=wrist,
    )

    assert seen["url"] == "http://127.0.0.1:18080/process"
    assert seen["timeout"] == 2.5
    assert "multipart/form-data" in seen["content_type"]
    assert b'name="select_objects"' in seen["body"]
    assert b"book" in seen["body"]
    assert b'name="base_npy"; filename="base.npy"' in seen["body"]
    assert np.array_equal(result.base_rgb, head_out)
    assert np.array_equal(result.wrist_rgb, wrist_out)
    assert result.used_fallback is False


def test_http_client_replaces_black_overlay_background_with_dimmed_original_context() -> None:
    head = np.full((2, 3, 3), 100, dtype=np.uint8)
    wrist = np.full((2, 3, 3), 80, dtype=np.uint8)
    head_out = np.zeros((2, 3, 3), dtype=np.uint8)
    wrist_out = np.zeros((2, 3, 3), dtype=np.uint8)
    head_out[0, 1] = [0, 128, 255]
    wrist_out[1, 2] = [255, 64, 0]

    boundary = "overlayboundary"
    body = (
        f"--{boundary}\r\n"
        "Content-Type: application/octet-stream\r\n"
        "Content-Disposition: attachment; filename=overlay_base.npy\r\n\r\n"
    ).encode() + _npy_bytes(head_out) + (
        f"\r\n--{boundary}\r\n"
        "Content-Type: application/octet-stream\r\n"
        "Content-Disposition: attachment; filename=overlay_wrist.npy\r\n\r\n"
    ).encode() + _npy_bytes(wrist_out) + f"\r\n--{boundary}--\r\n".encode()

    def opener(_request, timeout):
        del timeout
        return _FakeResponse(body, {"Content-Type": f"multipart/mixed; boundary={boundary}"})

    client = adapter.HttpPerceptionClient(
        endpoint="http://127.0.0.1:18080/process",
        opener=opener,
    )
    result = client.process(
        select_objects="book",
        exclude_objects="",
        is_base_init=True,
        is_wrist_init=True,
        base_rgb=head,
        wrist_rgb=wrist,
    )

    assert np.array_equal(result.base_rgb[0, 0], [35, 35, 35])
    assert np.array_equal(result.wrist_rgb[0, 0], [28, 28, 28])
    assert np.array_equal(result.base_rgb[0, 1], [0, 128, 255])
    assert np.array_equal(result.wrist_rgb[1, 2], [255, 64, 0])


def test_fallback_client_returns_original_images_when_http_fails() -> None:
    head = np.full((2, 2, 3), 3, dtype=np.uint8)
    wrist = np.full((2, 2, 3), 5, dtype=np.uint8)

    def opener(_request, timeout):
        del timeout
        raise URLError("service down")

    client = adapter.FallbackPerceptionClient(
        primary=adapter.HttpPerceptionClient(
            endpoint="http://127.0.0.1:18080/process",
            opener=opener,
        ),
        fallback=adapter.PassthroughPerceptionClient(),
    )
    result = client.process(
        select_objects="book",
        exclude_objects="",
        is_base_init=False,
        is_wrist_init=False,
        base_rgb=head,
        wrist_rgb=wrist,
    )

    assert np.array_equal(result.base_rgb, head)
    assert np.array_equal(result.wrist_rgb, wrist)
    assert result.used_fallback is True
    assert "service down" in result.error


def test_lerobot_output_paths_match_v21_video_layout(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "data/chunk-000").mkdir(parents=True)
    (dataset / "videos/chunk-000/observation.images.top_head").mkdir(parents=True)
    (dataset / "videos/chunk-000/observation.images.hand_right").mkdir(parents=True)
    (dataset / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "fps": 30,
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            }
        )
    )

    pair = convert_lerobot_dataset.episode_video_pair(dataset, episode_index=12)

    assert pair.base == dataset / "videos/chunk-000/observation.images.top_head/episode_000012.mp4"
    assert pair.wrist == dataset / "videos/chunk-000/observation.images.hand_right/episode_000012.mp4"


def test_convert_episode_writes_processed_video_pair(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    (input_root / "meta").mkdir(parents=True)
    (input_root / "data/chunk-000").mkdir(parents=True)
    (input_root / "videos/chunk-000/observation.images.top_head").mkdir(parents=True)
    (input_root / "videos/chunk-000/observation.images.hand_right").mkdir(parents=True)
    (input_root / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "fps": 5,
                "chunks_size": 1000,
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            }
        )
    )
    (input_root / "data/chunk-000/episode_000000.parquet").write_bytes(b"not used")
    base_video = input_root / "videos/chunk-000/observation.images.top_head/episode_000000.mp4"
    wrist_video = input_root / "videos/chunk-000/observation.images.hand_right/episode_000000.mp4"
    _write_tiny_video(base_video, value=10)
    _write_tiny_video(wrist_video, value=20)
    convert_lerobot_dataset.prepare_output_dataset(input_root=input_root, output_root=output_root)

    class AddOneClient:
        def process(self, **kwargs):
            return adapter.PerceptionResult(
                base_rgb=np.full_like(kwargs["base_rgb"], 180),
                wrist_rgb=np.full_like(kwargs["wrist_rgb"], 200),
            )

    convert_lerobot_dataset.convert_episode(
        input_root=input_root,
        output_root=output_root,
        episode_index=0,
        client=AddOneClient(),
        select_objects="book",
        exclude_objects="",
    )

    pair = convert_lerobot_dataset.episode_video_pair(output_root, episode_index=0)
    assert pair.base.exists()
    assert pair.wrist.exists()
    assert _read_first_rgb(pair.base).mean() > _read_first_rgb(base_video).mean()
    assert _read_first_rgb(pair.wrist).mean() > _read_first_rgb(wrist_video).mean()


def _write_tiny_video(path: Path, *, value: int) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (8, 6),
    )
    assert writer.isOpened()
    for _ in range(2):
        rgb = np.full((6, 8, 3), value, dtype=np.uint8)
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()


def _read_first_rgb(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    try:
        ok, bgr = cap.read()
        assert ok
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
