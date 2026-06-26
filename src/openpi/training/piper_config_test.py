import dataclasses

from openpi import transforms
from openpi.models import pi0_config
from openpi.training import config as training_config


def _piper_delta_and_absolute_masks(data_config):
    delta_transforms = [
        transform for transform in data_config.data_transforms.inputs if isinstance(transform, transforms.DeltaActions)
    ]
    absolute_transforms = [
        transform
        for transform in data_config.data_transforms.outputs
        if isinstance(transform, transforms.AbsoluteActions)
    ]
    return delta_transforms[-1].mask, absolute_transforms[0].mask


def test_piper_config_keeps_gripper_absolute_for_existing_checkpoints(tmp_path):
    data_config = training_config.LeRobotPiperDataConfig(
        repo_id="/tmp/piper",
        assets=training_config.AssetsConfig(asset_id="piper_right_book_v5"),
    ).create(tmp_path, pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=15))

    delta_mask, absolute_mask = _piper_delta_and_absolute_masks(data_config)

    assert delta_mask == transforms.make_bool_mask(6, -1)
    assert absolute_mask == transforms.make_bool_mask(6, -1)


def test_piper_all_delta_config_uses_delta_actions_for_all_seven_action_dims(tmp_path):
    data_config = training_config.LeRobotPiperDataConfig(
        repo_id="/tmp/piper",
        assets=training_config.AssetsConfig(asset_id="piper_right_book_v5_all_delta"),
        delta_gripper_action=True,
    ).create(tmp_path, pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=15))

    delta_mask, absolute_mask = _piper_delta_and_absolute_masks(data_config)

    assert delta_mask == transforms.make_bool_mask(7)
    assert absolute_mask == transforms.make_bool_mask(7)


def test_piper_joint_delta_gripper_absolute_train_config_keeps_gripper_absolute(tmp_path):
    config = training_config.get_config("pi05_piper_right_book_v5_lora_joint_delta_gripper_absolute")
    config = dataclasses.replace(config, assets_base_dir=str(tmp_path))
    data_config = config.data.create(config.assets_dirs, config.model)

    delta_mask, absolute_mask = _piper_delta_and_absolute_masks(data_config)

    assert data_config.asset_id == "piper_right_book_v5_joint_delta_gripper_absolute"
    assert delta_mask == transforms.make_bool_mask(6, -1)
    assert absolute_mask == transforms.make_bool_mask(6, -1)
    assert data_config.action_delta_timestamps_start == 1


def test_piper_no_rgbd_train_config_uses_new_dataset_and_keeps_gripper_absolute(tmp_path):
    config = training_config.get_config("pi05_piper_right_book_noRGBD_lora_joint_delta_gripper_absolute")
    config = dataclasses.replace(config, assets_base_dir=str(tmp_path))
    data_config = config.data.create(config.assets_dirs, config.model)

    delta_mask, absolute_mask = _piper_delta_and_absolute_masks(data_config)

    assert config.data.repo_id == training_config.PIPER_LEROBOT_NO_RGBD_DATASET
    assert config.checkpoint_base_dir == training_config.PIPER_CHECKPOINT_ROOT
    assert data_config.asset_id == "piper_right_book_noRGBD_joint_delta_gripper_absolute"
    assert delta_mask == transforms.make_bool_mask(6, -1)
    assert absolute_mask == transforms.make_bool_mask(6, -1)
    assert data_config.action_delta_timestamps_start == 1
