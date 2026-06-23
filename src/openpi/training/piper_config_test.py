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
