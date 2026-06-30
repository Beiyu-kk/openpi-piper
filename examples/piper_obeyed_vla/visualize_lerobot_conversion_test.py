from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from examples.piper_obeyed_vla import visualize_lerobot_conversion


def test_export_preview_writes_labeled_comparison_frames(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    converted_root = tmp_path / "converted"
    for root in (input_root, converted_root):
        (root / "meta").mkdir(parents=True)
        (root / "videos/chunk-000/observation.images.top_head").mkdir(parents=True)
        (root / "videos/chunk-000/observation.images.hand_right").mkdir(parents=True)
        (root / "meta/info.json").write_text(
            json.dumps(
                {
                    "fps": 5,
                    "chunks_size": 1000,
                    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                }
            )
        )
    _write_video(
        input_root / "videos/chunk-000/observation.images.top_head/episode_000000.mp4",
        value=20,
    )
    _write_video(
        input_root / "videos/chunk-000/observation.images.hand_right/episode_000000.mp4",
        value=40,
    )
    _write_video(
        converted_root / "videos/chunk-000/observation.images.top_head/episode_000000.mp4",
        value=180,
    )
    _write_video(
        converted_root / "videos/chunk-000/observation.images.hand_right/episode_000000.mp4",
        value=220,
    )

    written = visualize_lerobot_conversion.export_preview(
        input_root=input_root,
        converted_root=converted_root,
        output_dir=tmp_path / "preview",
        episode_index=0,
        frame_indices=[0, 1],
    )

    assert len(written) == 4
    assert all(path.exists() for path in written)
    preview = cv2.imread(str(written[0]))
    assert preview is not None
    assert preview.shape[:2] == (12, 32)
    assert preview[:, :16].mean() < preview[:, 16:].mean()


def _write_video(path: Path, *, value: int) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (16, 12),
    )
    assert writer.isOpened()
    for _ in range(2):
        writer.write(np.full((12, 16, 3), value, dtype=np.uint8))
    writer.release()
