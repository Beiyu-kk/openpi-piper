# Piper openpi Control Client

This example runs the Piper robot-side client for the `pi05_piper_right_book_v5_lora` policy.

On this machine, migrated Piper checkpoints live under:

```text
/mnt/c9dd2903-1a5c-4ec3-b146-9f8ee2434744/checkpoints/openpi
```

Start the policy server from the openpi repo:

```bash
cd <openpi_repo>
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --no-dev scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_piper_right_book_noRGBD_lora_joint_delta_gripper_absolute \
  --policy.dir=/mnt/c9dd2903-1a5c-4ec3-b146-9f8ee2434744/checkpoints/openpi/pi05_piper_right_book_noRGBD_lora_joint_delta_gripper_absolute/piper_right_book_noRGBD_joint_delta_gripper_absolute_bs32/5000
```

Then run the Piper client from the `piper-openpi` conda environment:

```bash
conda activate piper-openpi
python examples/piper/main.py \
  --host=127.0.0.1 \
  --port=8000 \
  --prompt="抓起书本放到另外一个格子里" \
  --camera-backend=realsense \
  --head-camera-serial=339322074804 \
  --wrist-camera-serial=346522074547 \
  --show-cameras \
  --head-camera=0 \
  --wrist-camera=2 \
  --can-name=can_right \
  --dry-run
```

Set `--no-dry-run --enable-robot` only after the camera, policy server, CAN interface, and workspace safety checks are ready. For the real robot, use tyro-style boolean flags without `=True` or `=False`:

```bash
python examples/piper/main.py \
  --host=127.0.0.1 \
  --port=8000 \
  --prompt="抓起书本放到另外一个格子里" \
  --camera-backend=realsense \
  --head-camera-serial=339322074804 \
  --wrist-camera-serial=346522074547 \
  --show-cameras \
  --can-name=can_right \
  --open-loop-horizon=12 \
  --control-hz=15 \
  --move-speed-percent=30 \
  --gripper-open-mm=70.0 \
  --gripper-closed-mm=0.0 \
  --gripper-threshold-mm=35.0 \
  --gripper-hold-close \
  --gripper-close-trigger-mm=25.0 \
  --gripper-release-trigger-mm=45.0 \
  --gripper-hold-mm=0.0 \
  --no-dry-run \
  --enable-robot
```

The gripper action is thresholded in deployment: policy targets at or below `--gripper-threshold-mm` are sent as `--gripper-closed-mm`, and targets above it are sent as `--gripper-open-mm`.
`--gripper-hold-close` latches the gripper closed once the policy target goes below `--gripper-close-trigger-mm`; it releases only when the policy target rises above `--gripper-release-trigger-mm`.
Use `--show-cameras` to display the head and wrist camera views side by side; press `q` in the preview window to stop the client.
