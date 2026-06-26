# Piper pi0/pi05 推理启动步骤

本文件只写启动推理需要执行的命令。当前 Piper 微调模型使用 openpi 的 `pi05` 配置启动，属于 pi0 系列推理服务。

## 0. 动作块长度自动适配

现在 `examples/piper/main.py` 不再写死动作块长度为 15。

启动 `scripts/serve_policy.py` 时，服务端会根据 `--policy.config` 对应的训练配置自动把 `action_horizon` 写入 server metadata；控制端连接后会读取这个值，并按模型自己的动作块长度接收动作：

- `pi05_piper_right_book_v5_lora`：`action_horizon=15`
- `pi05_piper_right_book_v5_lora_all_delta`：`action_horizon=15`
- `pi05_piper_right_book_v5_lora_joint_delta_gripper_absolute`：`action_horizon=30`

所以控制端命令里不需要额外传模型动作块大小。只要服务端的 `--policy.config` 和 `--policy.dir` 对应正确 checkpoint，控制端会自动适配 15 或 30 的返回动作块。

注意：`--open-loop-horizon` 不是模型动作块长度，而是每次收到动作块后连续执行多少步再重新请求模型。它必须小于等于模型的 `action_horizon`。例如 30 步动作块的模型可以设置 `--open-loop-horizon=15`，表示每次只执行前 15 步后重新推理。

## 1.1 启动模型推理服务 piper_right_book_v5_lora_bs96

该配置是前六维关节相对动作 + 第七维夹爪绝对位置，但是norm stats计算使用的是绝对，存在action=state的问题

打开第一个终端：

```bash
cd <openpi_repo>
source .venv/bin/activate

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --no-dev scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_piper_right_book_v5_lora \
  --policy.dir=checkpoints/pi05_piper_right_book_v5_lora/piper_right_book_v5_lora_bs96/10000
```

说明：

- `<openpi_repo>` 是 openpi 仓库根目录，也就是包含 `pyproject.toml`、`scripts/`、`examples/`、`checkpoints/` 的目录。
- `--policy.config=pi05_piper_right_book_v5_lora` 指定 Piper 的 pi0/pi05 推理配置。
- `--policy.dir=.../10000` 指定使用训练到 10000 step 的 checkpoint。
- `--port=8000` 是控制端连接的端口。
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` 限制 JAX 最多预分配 95% 显存。
- `--port=8000` 必须写在 `policy:checkpoint` 前面，这是 `serve_policy.py` 的参数解析要求。

服务端启动后保持这个终端不要关闭。

## 1.2 启动模型推理服务 pi05_piper_right_book_v5_lora_all_delta

该配置是前六维关节相对位置 + 第七维夹爪绝对位置，norm stats计算使用的是相对位置，但是存在action=state的问题

```bash
cd <openpi_repo>
source .venv/bin/activate

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --no-dev scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_piper_right_book_v5_lora_all_delta \
  --policy.dir=checkpoints/pi05_piper_right_book_v5_lora_all_delta/piper_right_book_v5_lora_all_delta_bs32/10000
```

## 1.3 启动模型推理服务 pi05_piper_right_book_v5_lora_joint_delta_gripper_absolute

该配置是前六维关节相对位置 + 第七维夹爪绝对位置，norm stats计算使用的是相对位置，并通过 `action_delta_timestamps_start=1` 避开 `action_t == state_t` 的首帧问题。该模型的 `action_horizon=30`，控制端会从服务端 metadata 自动识别，不需要额外传动作块大小。

```bash
cd <openpi_repo>
source .venv/bin/activate

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --no-dev scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_piper_right_book_v5_lora_joint_delta_gripper_absolute \
  --policy.dir=checkpoints/pi05_piper_right_book_v5_lora_joint_delta_gripper_absolute/piper_right_book_joint_delta_gripper_absolute_bs32/10000
```

## 2. 真机启动

确认相机序列号、`can_right`、机械臂工作空间、急停和夹爪开合方向都正确后，再执行真机命令：

正常启动

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
  --camera-preview-scale=1.2 \
  --can-name=can_right \
  --open-loop-horizon=15 \
  --control-hz=15 \
  --move-speed-percent=30 \
  --gripper-open-mm=70.0 \
  --gripper-closed-mm=0.0 \
  --no-dry-run \
  --enable-robot
```

特殊化处理夹爪启动

如果模型输出夹爪目标 <= 25mm：
    进入“闭合保持”状态
    后续夹爪目标强制压到 0mm

在闭合保持状态下：
    只要模型输出夹爪目标 < 45mm
    就继续保持 0mm 闭合

只有当模型输出夹爪目标 >= 45mm：
    才退出闭合保持状态
    恢复按模型输出控制夹爪
    
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
  --camera-preview-scale=1.2 \
  --can-name=can_right \
  --open-loop-horizon=15 \
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

注意：`examples/piper/main.py` 使用 tyro 解析参数，布尔参数不要写成 `=True` 或 `=False`。启用某个开关直接写 `--gripper-hold-close`、`--enable-robot`，关闭默认开启的 dry-run 写 `--no-dry-run`。

## 3. 常用修改项

- 如果模型服务不在本机，把 `--host=127.0.0.1` 改成服务端机器 IP。
- 如果端口不是 8000，同时修改服务端 `--port` 和控制端 `--port`。
- 如果 RealSense 相机序列号不同，修改 `--head-camera-serial` 和 `--wrist-camera-serial`。
- 当前主视角相机序列号为 `339322074804`，右腕相机序列号为 `346522074547`。
- 当前右臂 CAN 口为 `can_right`；如果换机器或换机械臂，修改 `--can-name`。
- 如需部署时查看双相机画面，保留 `--show-cameras`；无显示器或远程无图形界面时去掉该参数。
- `--camera-preview-scale` 控制预览窗口缩放比例，默认推荐 `1.5`；例如 `0.5` 表示显示为原始画面的一半大小。
- 夹爪输出不再做阈值化：模型第 7 维按连续米值解释，控制端只用 `--gripper-open-mm` 和 `--gripper-closed-mm` 作为安全裁剪范围，再转成 Piper SDK 的 `0.001mm` 单位。
- `--gripper-hold-close` 是部署侧夹爪闭合保持策略：当模型夹爪目标低于 `--gripper-close-trigger-mm` 时进入闭合保持，将夹爪目标压到 `--gripper-hold-mm`；只有当模型目标高于 `--gripper-release-trigger-mm` 时才释放。

## 4. CAN 检查

真机启动前检查 CAN：

```bash
ip link show can_right
```

如果 `can_right` 没有启动，可按本机 Piper 配置启用，例如：

```bash
sudo ip link set can_right up type can bitrate 1000000
```

具体 bitrate 以当前 Piper 机械臂和 SDK 配置为准。
