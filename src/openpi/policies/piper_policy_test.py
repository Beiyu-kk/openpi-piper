import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import piper_policy


def test_piper_inputs_map_right_hand_camera_and_pad_left_wrist():
    base_image = np.full((4, 5, 3), 17, dtype=np.uint8)
    right_wrist_image = np.full((4, 5, 3), 29, dtype=np.uint8)
    data = {
        "observation/image": base_image,
        "observation/right_wrist_image": right_wrist_image,
        "observation/state": np.arange(7, dtype=np.float32),
        "actions": np.ones((2, 7), dtype=np.float32),
        "prompt": "抓起书本放到另外一个格子里",
    }

    transformed = piper_policy.PiperInputs(model_type=_model.ModelType.PI05)(data)

    assert np.array_equal(transformed["image"]["base_0_rgb"], base_image)
    assert np.array_equal(transformed["image"]["right_wrist_0_rgb"], right_wrist_image)
    assert np.array_equal(transformed["image"]["left_wrist_0_rgb"], np.zeros_like(base_image))
    assert transformed["image_mask"]["base_0_rgb"] == np.True_
    assert transformed["image_mask"]["right_wrist_0_rgb"] == np.True_
    assert transformed["image_mask"]["left_wrist_0_rgb"] == np.False_
    assert np.array_equal(transformed["state"], data["observation/state"])
    assert np.array_equal(transformed["actions"], data["actions"])
    assert transformed["prompt"] == data["prompt"]


def test_piper_delta_converts_all_action_dims_and_outputs_preserve_continuous_gripper():
    state = np.array([1, 2, 3, 4, 5, 6, 0.2], dtype=np.float32)
    actions = np.array(
        [
            [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 0.49],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.51],
        ],
        dtype=np.float32,
    )
    delta = transforms.DeltaActions(mask=transforms.make_bool_mask(7))(
        {"state": state.copy(), "actions": actions.copy()}
    )

    np.testing.assert_allclose(delta["actions"][..., :6], actions[..., :6] - state[:6])
    np.testing.assert_allclose(delta["actions"][..., 6], actions[..., 6] - state[6])

    absolute = transforms.AbsoluteActions(mask=transforms.make_bool_mask(7))(delta)
    outputs = piper_policy.PiperOutputs(binarize_gripper=False)(absolute)

    np.testing.assert_allclose(outputs["actions"][..., :6], actions[..., :6])
    np.testing.assert_allclose(outputs["actions"][..., 6], np.array([0.49, 0.51], dtype=np.float32))


def test_piper_existing_checkpoint_delta_keeps_gripper_absolute():
    state = np.array([1, 2, 3, 4, 5, 6, 0.2], dtype=np.float32)
    actions = np.array(
        [
            [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 0.49],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.51],
        ],
        dtype=np.float32,
    )

    delta = transforms.DeltaActions(mask=transforms.make_bool_mask(6, -1))(
        {"state": state.copy(), "actions": actions.copy()}
    )

    np.testing.assert_allclose(delta["actions"][..., :6], actions[..., :6] - state[:6])
    np.testing.assert_allclose(delta["actions"][..., 6], actions[..., 6])

    absolute = transforms.AbsoluteActions(mask=transforms.make_bool_mask(6, -1))(delta)
    outputs = piper_policy.PiperOutputs(binarize_gripper=False)(absolute)

    np.testing.assert_allclose(outputs["actions"], actions)
