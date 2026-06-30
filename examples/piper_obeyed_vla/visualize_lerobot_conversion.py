from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.piper_obeyed_vla.convert_lerobot_dataset import (
    BASE_VIDEO_KEY,
    WRIST_VIDEO_KEY,
    episode_video_pair,
)


def _read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
        return frame
    finally:
        cap.release()


def _fit_width(frame: np.ndarray, width: int) -> np.ndarray:
    height, current_width = frame.shape[:2]
    if current_width == width:
        return frame
    scale = float(width) / float(current_width)
    return cv2.resize(frame, (width, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _side_by_side(left: np.ndarray, right: np.ndarray, *, left_label: str, right_label: str) -> np.ndarray:
    width = min(left.shape[1], right.shape[1])
    left = _fit_width(left, width)
    right = _fit_width(right, width)
    height = min(left.shape[0], right.shape[0])
    left = left[:height, :width]
    right = right[:height, :width]
    return np.concatenate([_label(left, left_label), _label(right, right_label)], axis=1)


def export_preview(
    *,
    input_root: Path,
    converted_root: Path,
    output_dir: Path,
    episode_index: int,
    frame_indices: list[int],
) -> list[Path]:
    raw_pair = episode_video_pair(input_root, episode_index=episode_index)
    converted_pair = episode_video_pair(converted_root, episode_index=episode_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for view_name, raw_path, converted_path in (
        (BASE_VIDEO_KEY, raw_pair.base, converted_pair.base),
        (WRIST_VIDEO_KEY, raw_pair.wrist, converted_pair.wrist),
    ):
        short_name = view_name.rsplit(".", maxsplit=1)[-1]
        for frame_index in frame_indices:
            raw_frame = _read_frame(raw_path, frame_index)
            converted_frame = _read_frame(converted_path, frame_index)
            preview = _side_by_side(
                raw_frame,
                converted_frame,
                left_label=f"raw {short_name} frame {frame_index}",
                right_label=f"OBEYED {short_name} frame {frame_index}",
            )
            out_path = output_dir / f"{short_name}_raw_vs_obeyed_episode_{episode_index:06d}_frame_{frame_index:06d}.jpg"
            if not cv2.imwrite(str(out_path), preview):
                raise RuntimeError(f"Failed to write preview image: {out_path}")
            written.append(out_path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export raw-vs-OBEYED LeRobot video comparison frames.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--converted-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 10, 19])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = export_preview(
        input_root=args.input_root,
        converted_root=args.converted_root,
        output_dir=args.output_dir,
        episode_index=args.episode_index,
        frame_indices=args.frames,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
