from examples.piper import convert_piper_rgbd_hdf5_to_lerobot as converter


def test_default_paths_and_prompt_target_disk_dataset():
    assert converter.DEFAULT_RAW_DATASET.as_posix() == "/mnt/disk/Dataset/piper_data/data/piper_right_book_RGBD_V1_fixed"
    assert converter.DEFAULT_OUTPUT_DATASET.as_posix() == (
        "/mnt/disk/Dataset/piper_data/data/lerobot_v21/piper_right_book_RGBD_V1_fixed"
    )
    assert converter.DEFAULT_TASK == "将“C和指针”这本书从书架中取出，并放置到左边黑色置书架从左往右数第2个格子中"


def test_features_are_rgb_only_without_depth():
    features = converter.make_features()

    assert set(features) == {
        "observation.images.top_head",
        "observation.images.hand_right",
        "observation.state",
        "action",
    }
    assert features["observation.images.top_head"]["dtype"] == "video"
    assert features["observation.images.hand_right"]["dtype"] == "video"
    assert features["observation.state"]["shape"] == (7,)
    assert features["action"]["shape"] == (7,)
