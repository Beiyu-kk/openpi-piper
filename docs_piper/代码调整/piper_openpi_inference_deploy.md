# Piper openpi 推理部署记录

本文记录 Piper 右臂 `pi05_piper_right_book_v5_lora` checkpoint 的推理部署环境、SDK 接入、控制端代码和启动方式。

## 1. 本次接入目标

- 推理模型 checkpoint：
  `/home/ubun/project/VLA/PiPER/openpi/openpi/checkpoints/pi05_piper_right_book_v5_lora/piper_right_book_v5_lora_bs96/10000`
- 模型配置名：
  `pi05_piper_right_book_v5_lora`
- 控制端环境名：
  `piper-openpi`
- Piper SDK 来源：
  `/home/ubun/project/VLA/PiPER/ros2_ws/piper_sdk`
- 控制端代码位置：
  `examples/piper/main.py`

## 2. Conda 环境

已创建 conda 环境：

```bash
conda create -y -n piper-openpi python=3.10
```

已安装控制端运行依赖：

```bash
cd <openpi_repo>
conda activate piper-openpi
python -m pip install --no-deps -e packages/openpi-client
python -m pip install tyro dm-tree msgpack websockets opencv-python
python -m pip install -e third_party/piper_sdk
conda install -y pillow
```

其中 `<openpi_repo>` 是当前 openpi 仓库根目录，也就是包含 `pyproject.toml`、`scripts/`、`examples/`、`packages/` 的目录。当前机器上对应为 `/home/ubun/project/VLA/PiPER/openpi/openpi`。

当前验证到的关键包版本：

```text
dm-tree        0.1.10
msgpack        1.2.0
numpy          1.26.4
opencv-python  4.13.0.92
openpi-client  0.1.0  packages/openpi-client
pillow         12.2.0
piper_sdk      0.6.1  /home/ubun/project/VLA/PiPER/ros2_ws/piper_sdk
tyro           1.0.14
websockets     16.0
```

说明：推理服务端仍建议使用 openpi 仓库当前 `uv` 环境启动，因为服务端需要 JAX/CUDA、模型配置和 checkpoint 读取能力；`piper-openpi` 只用于机器人控制端。

## 3. Piper SDK 接入

已在 openpi 的 `third_party` 下加入 Piper SDK：

```bash
third_party/piper_sdk -> /home/ubun/project/VLA/PiPER/ros2_ws/piper_sdk
```

这里使用符号链接，目的是让 openpi 直接使用本机正在维护的 Piper SDK，而不是复制整个 ROS2 workspace。安装时使用：

```bash
cd <openpi_repo>
conda activate piper-openpi
python -m pip install -e third_party/piper_sdk
```

## 4. 控制端代码

新增目录：

```text
examples/piper/
  README.md
  __init__.py
  main.py
  piper_control_test.py
  requirements.txt
```

控制端使用：

- `openpi_client.websocket_client_policy.WebsocketClientPolicy` 连接 policy server。
- Piper SDK `C_PiperInterface_V2` 连接 `can_right`。
- OpenCV 读取头部相机和右腕相机。
- `openpi_client.image_tools.resize_with_pad` 将图像压到 `224x224` 后发送给服务端。

发送给模型的 request keys 和训练配置一致：

```python
{
    "observation/image": head_rgb_224,
    "observation/right_wrist_image": wrist_rgb_224,
    "observation/state": state_7d,
    "prompt": prompt,
}
```

其中 `state_7d` 为：

```text
[joint_1_rad, joint_2_rad, joint_3_rad, joint_4_rad, joint_5_rad, joint_6_rad, gripper_m]
```

## 5. 动作块和单位转换

服务端返回动作块：

```text
actions.shape == (15, 7)
```

动作含义：

- 前 6 维是模型输出的绝对关节目标，单位为弧度。
- 第 7 维是夹爪连续目标位置，单位为米；策略输出链路不再做阈值化。
- 旧配置 `pi05_piper_right_book_v5_lora` 训练时只对前 6 维关节使用 delta action，夹爪保持连续绝对值。
- 新配置 `pi05_piper_right_book_v5_lora_all_delta` 训练时 7 维动作全部使用 delta action。
- 两个配置的推理输出链路都会在发给控制端前转回绝对目标。

控制端发送 Piper SDK 前做如下转换：

- 关节：`rad -> degree * 1000`，对应 `JointCtrl` 的 `0.001°` 单位。
- 夹爪：连续米值先裁剪到 `[gripper_closed_mm, gripper_open_mm]` 对应的米制范围，再乘 `1_000_000`，对应 `GripperCtrl` 的 `0.001mm` 单位。
- 默认 `gripper_closed_mm=0.0`，`gripper_open_mm=70.0`，即夹爪输出范围约为 `0.0m` 到 `0.07m`。
- 默认 `move_speed_percent=30`，避免初次部署动作过快。

控制端也会将 6 个关节目标裁剪到 Piper SDK 文档给出的关节限制：

```text
j1 [-2.6179,  2.6179]
j2 [ 0.0000,  3.1400]
j3 [-2.9670,  0.0000]
j4 [-1.7450,  1.7450]
j5 [-1.2200,  1.2200]
j6 [-2.09439, 2.09439]
```

## 6. 启动 policy server

在 openpi 仓库目录启动：

```bash
cd <openpi_repo>
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --no-dev scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_piper_right_book_v5_lora \
  --policy.dir=checkpoints/pi05_piper_right_book_v5_lora/piper_right_book_v5_lora_bs96/10000
```

`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` 用于限制 JAX 预分配显存比例，避免服务端吃满整张显卡。

## 7. 启动 Piper 控制端

先 dry-run 验证相机、policy server 和动作块链路：

```bash
cd <openpi_repo>
conda activate piper-openpi
python examples/piper/main.py \
  --host=127.0.0.1 \
  --port=8000 \
  --prompt="抓起书本放到另外一个格子里" \
  --camera-backend=realsense \
  --head-camera-serial=339322074804 \
  --wrist-camera-serial=346522074547 \
  --show-cameras \
  --camera-preview-window="Piper cameras" \
  --camera-preview-scale=1.5 \
  --can-name=can_right \
  --open-loop-horizon=8 \
  --control-hz=15 \
  --dry-run
```

确认相机、CAN、机械臂工作空间和急停都准备好后，再允许真机发指令：

```bash
cd <openpi_repo>
conda activate piper-openpi
python examples/piper/main.py \
  --host=127.0.0.1 \
  --port=8000 \
  --prompt="抓起书本放到另外一个格子里" \
  --camera-backend=realsense \
  --head-camera-serial=339322074804 \
  --wrist-camera-serial=346522074547 \
  --show-cameras \
  --camera-preview-window="Piper cameras" \
  --camera-preview-scale=1.5 \
  --can-name=can_right \
  --open-loop-horizon=8 \
  --control-hz=15 \
  --move-speed-percent=30 \
  --gripper-open-mm=70.0 \
  --gripper-closed-mm=0.0 \
  --no-dry-run \
  --enable-robot
```

如果 RealSense 相机序列号不同，修改 `--head-camera-serial` 和 `--wrist-camera-serial`；如果 policy server 在另一台机器上，修改 `--host`。`--show-cameras` 会显示左主视角、右腕部相机的双画面预览，按 `q` 可退出；无图形界面时去掉该参数。

## 8. CAN 准备

启动真机前需要确认 `can_right` 已存在并处于 up 状态：

```bash
ip link show can_right
```

如果需要手动启用 CAN，按本机 Piper SDK/ROS2 workspace 的实际 CAN 配置执行，例如：

```bash
sudo ip link set can_right up type can bitrate 1000000
```

具体 bitrate 以当前 Piper 机械臂和 SDK 配置为准。

## 9. 已完成验证

已完成以下验证：

```bash
conda run -n piper-openpi python -c "import cv2, numpy, openpi_client, piper_sdk, tyro; from PIL import Image; print('imports-ok')"
```

结果：

```text
imports-ok
```

已验证控制端 CLI 可以正常解析：

```bash
conda run -n piper-openpi python examples/piper/main.py --help
```

已运行 Piper 控制端单测和策略 transform 单测：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-dev --with pytest --with pynvml pytest \
  examples/piper/piper_control_test.py \
  src/openpi/policies/piper_policy_test.py \
  -q
```

结果：

```text
9 passed
```

## 10. 注意事项

- 默认 `--dry-run`，不会向 Piper 发送 CAN 控制命令。
- 真机部署必须显式设置 `--no-dry-run --enable-robot`。
- `open-loop-horizon` 不能超过 15，因为该 checkpoint 的 `action_horizon=15`。
- 初次真机建议保留 `--move-speed-percent=30`，稳定后再逐步调整。
- 如果模型输出动作抖动，可以先降低 `--open-loop-horizon` 或 `--move-speed-percent`。
- 夹爪输出不再做 `0/1` 阈值化；如果夹爪行程不合适，优先调整 `--gripper-open-mm` 和 `--gripper-closed-mm` 的裁剪范围。
