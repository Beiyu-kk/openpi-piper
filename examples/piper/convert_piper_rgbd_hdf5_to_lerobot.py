from __future__ import annotations

import re
import shutil
from pathlib import Path

import cv2
from datasets import Dataset
import h5py
from lerobot.common.datasets.lerobot_dataset import compute_episode_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np
import pandas as pd
from tqdm import tqdm
import tyro


PIPER_DISK_DATA_ROOT = Path("/mnt/disk/Dataset/piper_data/data")
DEFAULT_RAW_DATASET = PIPER_DISK_DATA_ROOT / "piper_right_book_RGBD_V1_fixed"
DEFAULT_OUTPUT_DATASET = PIPER_DISK_DATA_ROOT / "lerobot_v21/piper_right_book_RGBD_V1_fixed"
DEFAULT_TASK = "将“C和指针”这本书从书架中取出，并放置到左边黑色置书架从左往右数第2个格子中"


def make_features() -> dict:
    joint_names = [f"right_joint_{idx}" for idx in range(7)]
    return {
        "observation.images.top_head": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.hand_right": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": joint_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": joint_names,
        },
    }


def _episode_number(path: Path) -> int:
    match = re.search(r"episode_(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Could not parse episode number from {path}.")
    return int(match.group(1))


def _read_rgb(cap: cv2.VideoCapture, *, path: Path, frame_index: int) -> np.ndarray:
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_index} from {path}.")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _add_episode(dataset: LeRobotDataset, episode_path: Path, raw_dataset: Path, *, task: str) -> None:
    with h5py.File(episode_path, "r") as episode:
        state = np.asarray(episode["observations/qpos"], dtype=np.float32)
        action = np.asarray(episode["action"], dtype=np.float32)

    episode_id = _episode_number(episode_path)
    top_video_path = raw_dataset / "video/cam_high" / f"episode_{episode_id}.mp4"
    wrist_video_path = raw_dataset / "video/cam_right_wrist" / f"episode_{episode_id}.mp4"
    top_video = cv2.VideoCapture(str(top_video_path))
    wrist_video = cv2.VideoCapture(str(wrist_video_path))
    if not top_video.isOpened():
        raise RuntimeError(f"Failed to open top-head RGB video: {top_video_path}")
    if not wrist_video.isOpened():
        raise RuntimeError(f"Failed to open right-wrist RGB video: {wrist_video_path}")

    try:
        frame_count = min(
            len(state),
            len(action),
            int(top_video.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(wrist_video.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
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


def _copy_episode_fast(
    *,
    metadata: LeRobotDatasetMetadata,
    episode_path: Path,
    raw_dataset: Path,
    episode_index: int,
    global_frame_start: int,
    task_index: int,
) -> tuple[int, dict]:
    with h5py.File(episode_path, "r") as episode:
        state = np.asarray(episode["observations/qpos"], dtype=np.float32)
        action = np.asarray(episode["action"], dtype=np.float32)

    raw_episode_id = _episode_number(episode_path)
    top_video_src = raw_dataset / "video/cam_high" / f"episode_{raw_episode_id}.mp4"
    wrist_video_src = raw_dataset / "video/cam_right_wrist" / f"episode_{raw_episode_id}.mp4"
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

    top_video_dst = metadata.root / metadata.get_video_file_path(episode_index, "observation.images.top_head")
    wrist_video_dst = metadata.root / metadata.get_video_file_path(episode_index, "observation.images.hand_right")
    top_video_dst.parent.mkdir(parents=True, exist_ok=True)
    wrist_video_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(top_video_src, top_video_dst)
    shutil.copy2(wrist_video_src, wrist_video_dst)

    frame_indices = np.arange(frame_count, dtype=np.int64)
    episode_indices = np.full(frame_count, episode_index, dtype=np.int64)
    global_indices = np.arange(global_frame_start, global_frame_start + frame_count, dtype=np.int64)
    timestamps = (frame_indices / float(metadata.fps)).astype(np.float32)
    task_indices = np.full(frame_count, task_index, dtype=np.int64)

    data_path = metadata.root / metadata.get_data_file_path(episode_index)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_pandas(
        pd.DataFrame(
            {
                "observation.state": list(state[:frame_count]),
                "action": list(action[:frame_count]),
                "timestamp": timestamps,
                "frame_index": frame_indices,
                "episode_index": episode_indices,
                "index": global_indices,
                "task_index": task_indices,
            }
        ),
        preserve_index=False,
    ).to_parquet(str(data_path))

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
        metadata.features,
    )
    return frame_count, episode_stats


def _convert_fast(
    *,
    raw_dataset: Path,
    output_dataset: Path,
    episode_paths: list[Path],
    task: str,
) -> tuple[int, int]:
    metadata = LeRobotDatasetMetadata.create(
        repo_id=str(output_dataset),
        root=output_dataset,
        robot_type="agilex",
        fps=30,
        features=make_features(),
        use_videos=True,
    )
    metadata.add_task(task)
    task_index = metadata.get_task_index(task)
    assert task_index is not None

    total_frames = 0
    for episode_index, episode_path in enumerate(tqdm(episode_paths, desc="Fast-copying Piper RGBD RGB videos")):
        frame_count, episode_stats = _copy_episode_fast(
            metadata=metadata,
            episode_path=episode_path,
            raw_dataset=raw_dataset,
            episode_index=episode_index,
            global_frame_start=total_frames,
            task_index=task_index,
        )
        metadata.save_episode(episode_index, frame_count, [task], episode_stats)
        total_frames += frame_count

    return metadata.total_episodes, metadata.total_frames


def main(
    raw_dataset: Path = DEFAULT_RAW_DATASET,
    output_dataset: Path = DEFAULT_OUTPUT_DATASET,
    *,
    task: str = DEFAULT_TASK,
    overwrite: bool = False,
    max_episodes: int | None = None,
    fast_copy_videos: bool = True,
) -> None:
    """Convert Piper RGBD HDF5 episodes to LeRobot v2.1 using RGB streams only."""
    raw_dataset = raw_dataset.expanduser().resolve()
    output_dataset = output_dataset.expanduser().resolve()

    if output_dataset.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dataset} already exists. Pass --overwrite to recreate it.")
        shutil.rmtree(output_dataset)

    episode_paths = sorted(raw_dataset.glob("episode_*.hdf5"), key=_episode_number)
    if max_episodes is not None:
        episode_paths = episode_paths[:max_episodes]

    if fast_copy_videos:
        total_episodes, total_frames = _convert_fast(
            raw_dataset=raw_dataset,
            output_dataset=output_dataset,
            episode_paths=episode_paths,
            task=task,
        )
    else:
        dataset = LeRobotDataset.create(
            repo_id=str(output_dataset),
            root=output_dataset,
            robot_type="agilex",
            fps=30,
            features=make_features(),
            use_videos=True,
            image_writer_threads=8,
            video_backend="pyav",
        )
        for episode_path in tqdm(episode_paths, desc="Converting Piper RGBD HDF5 to LeRobot RGB-only"):
            _add_episode(dataset, episode_path, raw_dataset, task=task)
        total_episodes = dataset.meta.total_episodes
        total_frames = dataset.meta.total_frames

    print(f"LeRobot v2.1 dataset written to {output_dataset}")
    print(f"Task: {task}")
    print(f"Total episodes: {total_episodes}")
    print(f"Total frames: {total_frames}")


if __name__ == "__main__":
    tyro.cli(main)
