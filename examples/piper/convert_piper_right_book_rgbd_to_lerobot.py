from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import cv2
from datasets import Dataset
import h5py
from lerobot.common.datasets.lerobot_dataset import compute_episode_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import tyro


PIPER_DATA_ROOT = Path("/mnt/c9dd2903-1a5c-4ec3-b146-9f8ee2434744/Dataset/piper_data/data")
DEFAULT_BASE_LEROBOT_DATASET = PIPER_DATA_ROOT / "lerobot_v21/piper_right_book_noRGBD"
DEFAULT_RAW_RGBD_DATASET = PIPER_DATA_ROOT / "piper_right_book_RGBD_V1_fixed"
DEFAULT_OUTPUT_DATASET = PIPER_DATA_ROOT / "lerobot_v21/piper_right_book_noRGBD_RGBD_merged"


def _episode_number(path: Path) -> int:
    match = re.search(r"episode_(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Could not parse episode number from {path}.")
    return int(match.group(1))


def _read_default_task(base_lerobot_dataset: Path) -> str:
    tasks_path = base_lerobot_dataset / "meta/tasks.jsonl"
    with tasks_path.open(encoding="utf-8") as f:
        first_task = json.loads(next(f))
    return str(first_task["task"])


def _read_rgb(cap: cv2.VideoCapture, *, path: Path, frame_index: int) -> np.ndarray:
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_index} from {path}.")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _append_rgbd_episode(
    dataset: LeRobotDataset,
    episode_path: Path,
    raw_rgbd_dataset: Path,
    *,
    task: str,
) -> None:
    with h5py.File(episode_path, "r") as episode:
        state = np.asarray(episode["observations/qpos"], dtype=np.float32)
        action = np.asarray(episode["action"], dtype=np.float32)

    episode_id = _episode_number(episode_path)
    top_video_path = raw_rgbd_dataset / "video/cam_high" / f"episode_{episode_id}.mp4"
    wrist_video_path = raw_rgbd_dataset / "video/cam_right_wrist" / f"episode_{episode_id}.mp4"
    top_video = cv2.VideoCapture(str(top_video_path))
    wrist_video = cv2.VideoCapture(str(wrist_video_path))
    if not top_video.isOpened():
        raise RuntimeError(f"Failed to open top-head RGB video: {top_video_path}")
    if not wrist_video.isOpened():
        raise RuntimeError(f"Failed to open right-wrist RGB video: {wrist_video_path}")

    try:
        top_frames = int(top_video.get(cv2.CAP_PROP_FRAME_COUNT))
        wrist_frames = int(wrist_video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = min(len(state), len(action), top_frames, wrist_frames)
        if frame_count <= 0:
            raise RuntimeError(f"Episode {episode_path} has no usable frames.")

        for frame_index in range(frame_count):
            dataset.add_frame(
                {
                    "observation.images.top_head": _read_rgb(
                        top_video, path=top_video_path, frame_index=frame_index
                    ),
                    "observation.images.hand_right": _read_rgb(
                        wrist_video, path=wrist_video_path, frame_index=frame_index
                    ),
                    "observation.state": state[frame_index],
                    "action": action[frame_index],
                    "task": task,
                }
            )
        dataset.save_episode()
    finally:
        top_video.release()
        wrist_video.release()


def _jsonable_stats(stats: dict) -> dict:
    return {key: value.tolist() if hasattr(value, "tolist") else value for key, value in stats.items()}


def _copy_rgbd_episode_fast(
    *,
    output_dataset: Path,
    episode_path: Path,
    raw_rgbd_dataset: Path,
    episode_index: int,
    global_frame_start: int,
    task_index: int,
    info: dict,
) -> tuple[int, dict]:
    with h5py.File(episode_path, "r") as episode:
        state = np.asarray(episode["observations/qpos"], dtype=np.float32)
        action = np.asarray(episode["action"], dtype=np.float32)

    raw_episode_id = _episode_number(episode_path)
    top_video_src = raw_rgbd_dataset / "video/cam_high" / f"episode_{raw_episode_id}.mp4"
    wrist_video_src = raw_rgbd_dataset / "video/cam_right_wrist" / f"episode_{raw_episode_id}.mp4"
    top_video = cv2.VideoCapture(str(top_video_src))
    wrist_video = cv2.VideoCapture(str(wrist_video_src))
    try:
        if not top_video.isOpened():
            raise RuntimeError(f"Failed to open top-head RGB video: {top_video_src}")
        if not wrist_video.isOpened():
            raise RuntimeError(f"Failed to open right-wrist RGB video: {wrist_video_src}")
        frame_count = min(
            len(state),
            len(action),
            int(top_video.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(wrist_video.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        top_video.release()
        wrist_video.release()

    if frame_count <= 0:
        raise RuntimeError(f"Episode {episode_path} has no usable frames.")

    episode_chunk = episode_index // int(info["chunks_size"])
    top_video_dst = output_dataset / info["video_path"].format(
        episode_chunk=episode_chunk,
        video_key="observation.images.top_head",
        episode_index=episode_index,
    )
    wrist_video_dst = output_dataset / info["video_path"].format(
        episode_chunk=episode_chunk,
        video_key="observation.images.hand_right",
        episode_index=episode_index,
    )
    top_video_dst.parent.mkdir(parents=True, exist_ok=True)
    wrist_video_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(top_video_src, top_video_dst)
    shutil.copy2(wrist_video_src, wrist_video_dst)

    frame_indices = np.arange(frame_count, dtype=np.int64)
    episode_indices = np.full(frame_count, episode_index, dtype=np.int64)
    global_indices = np.arange(global_frame_start, global_frame_start + frame_count, dtype=np.int64)
    timestamps = (frame_indices / float(info["fps"])).astype(np.float32)
    task_indices = np.full(frame_count, task_index, dtype=np.int64)
    data = {
        "observation.state": list(state[:frame_count]),
        "action": list(action[:frame_count]),
        "timestamp": timestamps,
        "frame_index": frame_indices,
        "episode_index": episode_indices,
        "index": global_indices,
        "task_index": task_indices,
    }
    data_path = output_dataset / info["data_path"].format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_pandas(pd.DataFrame(data), preserve_index=False).to_parquet(str(data_path))

    episode_stats = compute_episode_stats(
        {
            "observation.state": state[:frame_count],
            "action": action[:frame_count],
            "timestamp": timestamps,
            "frame_index": frame_indices,
            "episode_index": episode_indices,
            "index": global_indices,
            "task_index": task_indices,
        },
        info["features"],
    )
    return frame_count, episode_stats


def _append_rgbd_episodes_fast(
    *,
    output_dataset: Path,
    raw_rgbd_dataset: Path,
    episode_paths: list[Path],
    task: str,
) -> None:
    from lerobot.common.datasets.lerobot_dataset import aggregate_stats
    from lerobot.common.datasets.lerobot_dataset import load_episodes_stats
    from lerobot.common.datasets.lerobot_dataset import write_episode
    from lerobot.common.datasets.lerobot_dataset import write_episode_stats
    from lerobot.common.datasets.utils import write_json

    info_path = output_dataset / "meta/info.json"
    info = json.loads(info_path.read_text())
    next_episode_index = int(info["total_episodes"])
    next_global_frame = int(info["total_frames"])
    existing_stats = load_episodes_stats(output_dataset)
    task_index = 0

    for offset, episode_path in enumerate(tqdm(episode_paths, desc="Fast-copying RGBD RGB episodes")):
        episode_index = next_episode_index + offset
        frame_count, episode_stats = _copy_rgbd_episode_fast(
            output_dataset=output_dataset,
            episode_path=episode_path,
            raw_rgbd_dataset=raw_rgbd_dataset,
            episode_index=episode_index,
            global_frame_start=next_global_frame,
            task_index=task_index,
            info=info,
        )
        write_episode({"episode_index": episode_index, "tasks": [task], "length": frame_count}, output_dataset)
        write_episode_stats(episode_index, episode_stats, output_dataset)
        existing_stats[episode_index] = episode_stats
        next_global_frame += frame_count

    info["total_episodes"] = next_episode_index + len(episode_paths)
    info["total_frames"] = next_global_frame
    info["total_tasks"] = 1
    info["total_videos"] = info["total_episodes"] * 2
    info["total_chunks"] = max(1, (info["total_episodes"] + int(info["chunks_size"]) - 1) // int(info["chunks_size"]))
    info["splits"] = {"train": f"0:{info['total_episodes']}"}
    stats = aggregate_stats(list(existing_stats.values()))
    write_json(info, info_path)
    write_json({key: _jsonable_stats(value) for key, value in stats.items()}, output_dataset / "meta/stats.json")


def main(
    raw_rgbd_dataset: Path = DEFAULT_RAW_RGBD_DATASET,
    base_lerobot_dataset: Path = DEFAULT_BASE_LEROBOT_DATASET,
    output_dataset: Path = DEFAULT_OUTPUT_DATASET,
    *,
    overwrite: bool = False,
    max_episodes: int | None = None,
    fast_copy_videos: bool = True,
) -> None:
    """Create a merged Piper LeRobot v2.1 dataset from noRGBD plus RGB-only frames from RGBD_V1_fixed."""
    raw_rgbd_dataset = raw_rgbd_dataset.expanduser().resolve()
    base_lerobot_dataset = base_lerobot_dataset.expanduser().resolve()
    output_dataset = output_dataset.expanduser().resolve()

    if output_dataset.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dataset} already exists. Pass --overwrite to recreate it.")
        shutil.rmtree(output_dataset)

    shutil.copytree(base_lerobot_dataset, output_dataset)
    task = _read_default_task(base_lerobot_dataset)

    episode_paths = sorted(raw_rgbd_dataset.glob("episode_*.hdf5"), key=_episode_number)
    if max_episodes is not None:
        episode_paths = episode_paths[:max_episodes]

    if fast_copy_videos:
        _append_rgbd_episodes_fast(
            output_dataset=output_dataset,
            raw_rgbd_dataset=raw_rgbd_dataset,
            episode_paths=episode_paths,
            task=task,
        )
        dataset = LeRobotDataset(str(output_dataset), video_backend="pyav")
    else:
        dataset = LeRobotDataset(str(output_dataset), video_backend="pyav")
        for episode_path in tqdm(episode_paths, desc="Appending RGBD episodes as RGB-only LeRobot data"):
            _append_rgbd_episode(dataset, episode_path, raw_rgbd_dataset, task=task)

    print(f"Merged dataset written to {output_dataset}")
    print(f"Total episodes: {dataset.meta.total_episodes}")
    print(f"Total frames: {dataset.meta.total_frames}")


if __name__ == "__main__":
    tyro.cli(main)
