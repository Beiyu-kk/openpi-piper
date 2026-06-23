# Piper openpi Control Client

This example runs the Piper robot-side client for the `pi05_piper_right_book_v5_lora` policy.

Start the policy server from the openpi repo:

```bash
cd <openpi_repo>
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --no-dev scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_piper_right_book_v5_lora \
  --policy.dir=checkpoints/pi05_piper_right_book_v5_lora/piper_right_book_v5_lora_bs96/10000
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
  --gripper-hold-close \
  --gripper-close-trigger-mm=25.0 \
  --gripper-release-trigger-mm=45.0 \
  --gripper-hold-mm=0.0 \
  --no-dry-run \
  --enable-robot
```

The gripper action is interpreted as a continuous meter value, clipped by `--gripper-closed-mm` and `--gripper-open-mm`, then converted to Piper SDK `0.001mm` units.
`--gripper-hold-close` latches the gripper closed once the policy target goes below `--gripper-close-trigger-mm`; it releases only when the policy target rises above `--gripper-release-trigger-mm`.
Use `--show-cameras` to display the head and wrist camera views side by side; press `q` in the preview window to stop the client.
