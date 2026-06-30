from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.piper_obeyed_vla import adapter


BASE_VIDEO_KEY = "observation.images.top_head"
WRIST_VIDEO_KEY = "observation.images.hand_right"


@dataclasses.dataclass(frozen=True)
class EpisodeVideoPair:
    base: Path
    wrist: Path


def episode_video_pair(dataset_root: Path, *, episode_index: int) -> EpisodeVideoPair:
    dataset_root = Path(dataset_root)
    info_path = dataset_root / "meta/info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    video_template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    episode_chunk = int(episode_index) // int(info.get("chunks_size", 1000))

    def render(video_key: str) -> Path:
        return dataset_root / video_template.format(
            episode_chunk=episode_chunk,
            video_key=video_key,
            episode_index=episode_index,
        )

    return EpisodeVideoPair(
        base=render(BASE_VIDEO_KEY),
        wrist=render(WRIST_VIDEO_KEY),
    )


def prepare_output_dataset(
    *,
    input_root: Path,
    output_root: Path,
    overwrite: bool = False,
) -> None:
    input_root = Path(input_root)
    output_root = Path(output_root)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists. Pass overwrite=True to recreate it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    for name in ("meta", "data"):
        shutil.copytree(input_root / name, output_root / name)
    norm_stats = input_root / "norm_stats.json"
    if norm_stats.exists():
        shutil.copy2(norm_stats, output_root / "norm_stats.json")
    for video_key in (BASE_VIDEO_KEY, WRIST_VIDEO_KEY):
        (output_root / "videos/chunk-000" / video_key).mkdir(parents=True, exist_ok=True)


def _read_info(dataset_root: Path) -> dict:
    with (dataset_root / "meta/info.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _open_reader(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return cap


def _make_writer(path: Path, *, width: int, height: int, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {path}")
    return writer


def convert_episode(
    *,
    input_root: Path,
    output_root: Path,
    episode_index: int,
    client: adapter.PerceptionClient,
    select_objects: str,
    exclude_objects: str,
    wrist_init_period: int = 15,
    limit_frames: int | None = None,
) -> None:
    input_pair = episode_video_pair(input_root, episode_index=episode_index)
    output_pair = episode_video_pair(output_root, episode_index=episode_index)
    base_cap = _open_reader(input_pair.base)
    wrist_cap = _open_reader(input_pair.wrist)
    try:
        base_count = int(base_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        wrist_count = int(wrist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = min(base_count, wrist_count)
        if limit_frames is not None:
            frame_count = min(frame_count, int(limit_frames))
        width = int(base_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(base_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = base_cap.get(cv2.CAP_PROP_FPS) or float(_read_info(input_root).get("fps", 30))
        base_writer = _make_writer(output_pair.base, width=width, height=height, fps=fps)
        wrist_writer = _make_writer(output_pair.wrist, width=width, height=height, fps=fps)
        try:
            for frame_idx in range(frame_count):
                ok_base, base_bgr = base_cap.read()
                ok_wrist, wrist_bgr = wrist_cap.read()
                if not ok_base or not ok_wrist:
                    break
                base_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
                wrist_rgb = cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB)
                result = client.process(
                    select_objects=select_objects,
                    exclude_objects=exclude_objects,
                    is_base_init=(frame_idx == 0),
                    is_wrist_init=(frame_idx == 0 or frame_idx % max(1, int(wrist_init_period)) == 0),
                    base_rgb=base_rgb,
                    wrist_rgb=wrist_rgb,
                )
                base_out = cv2.cvtColor(result.base_rgb, cv2.COLOR_RGB2BGR)
                wrist_out = cv2.cvtColor(result.wrist_rgb, cv2.COLOR_RGB2BGR)
                if base_out.shape[:2] != (height, width):
                    base_out = cv2.resize(base_out, (width, height), interpolation=cv2.INTER_LINEAR)
                if wrist_out.shape[:2] != (height, width):
                    wrist_out = cv2.resize(wrist_out, (width, height), interpolation=cv2.INTER_LINEAR)
                base_writer.write(base_out)
                wrist_writer.write(wrist_out)
        finally:
            base_writer.release()
            wrist_writer.release()
    finally:
        base_cap.release()
        wrist_cap.release()


def episode_indices(dataset_root: Path) -> list[int]:
    data_root = Path(dataset_root) / "data"
    indices: list[int] = []
    for path in sorted(data_root.glob("chunk-*/episode_*.parquet")):
        indices.append(int(path.stem.split("_")[-1]))
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an OBEYED-VLA-grounded LeRobot v2.1 Piper dataset.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--select-objects", default="")
    parser.add_argument("--exclude-objects", default="")
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080/process")
    parser.add_argument("--mode", choices=["http", "passthrough"], default="http")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--wrist-init-period", type=int, default=15)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--limit-frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_output_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )
    client = adapter.make_perception_client(
        mode=args.mode,
        endpoint=args.endpoint,
        timeout_s=args.timeout_s,
        fallback_to_passthrough=not args.no_fallback,
    )
    indices = episode_indices(args.input_root)
    if args.max_episodes > 0:
        indices = indices[: args.max_episodes]
    for episode_index in tqdm(indices, desc="OBEYED-VLA LeRobot conversion"):
        convert_episode(
            input_root=args.input_root,
            output_root=args.output_root,
            episode_index=episode_index,
            client=client,
            select_objects=args.select_objects,
            exclude_objects=args.exclude_objects,
            wrist_init_period=args.wrist_init_period,
            limit_frames=args.limit_frames or None,
        )


if __name__ == "__main__":
    main()
