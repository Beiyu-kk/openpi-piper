import pathlib

from scripts import compute_norm_stats_local_assets


class _Config:
    name = "example_config"
    assets_dirs = pathlib.Path("/tmp/project/assets/example_config")


class _DataConfig:
    repo_id = "/mnt/disk/Dataset/piper_data/data/lerobot_v21/example_dataset"
    asset_id = "example_asset"


def test_output_paths_include_local_dataset_name_and_asset_path():
    paths = compute_norm_stats_local_assets._output_paths(_Config(), _DataConfig())

    assert paths == [
        pathlib.Path("/mnt/disk/Dataset/piper_data/data/lerobot_v21/example_dataset"),
        pathlib.Path("/tmp/project/assets/example_config/example_asset"),
        pathlib.Path("/tmp/project/assets/example_config/example_dataset"),
    ]

