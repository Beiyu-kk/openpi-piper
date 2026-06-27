# openpi-piper 环境搭建

本文档简洁说明两套环境的搭建方式：

- **openpi 训练/推理服务端环境**：使用 `uv`，用于训练、计算 norm stats、启动 `scripts/serve_policy.py`。
- **piper-openpi 控制端环境**：使用 conda，专门用于运行 `examples/piper/main.py` 控制 Piper 真机。

两套环境不要混用。服务端依赖 JAX/CUDA/checkpoint 读取能力，放在 openpi 的 `uv` 环境里；控制端依赖 Piper SDK、相机、CAN 控制，放在轻量 conda 环境里。

## 1. openpi 训练/推理服务端环境

进入 openpi 仓库根目录：

```bash
cd <openpi_repo>
```

如果还没有安装 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

初始化依赖：

```bash
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

验证：

```bash
uv run python -c "import jax, torch, openpi; print('openpi-env-ok')"
```

计算 Piper norm stats 示例：

```bash
uv run --no-dev scripts/compute_norm_stats.py \
  --config-name=pi05_piper_right_book_noRGBD_lora_joint_delta_gripper_absolute
```

启动训练示例：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
uv run --no-dev scripts/train.py pi05_piper_right_book_noRGBD_lora_joint_delta_gripper_absolute \
  --exp-name=piper_right_book_noRGBD_joint_delta_gripper_absolute_bs32 \
  --batch-size=32 \
  --overwrite
```

启动推理服务示例：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --no-dev scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_piper_right_book_noRGBD_lora_joint_delta_gripper_absolute \
  --policy.dir=/mnt/disk/checkpoints/openpi/pi05_piper_right_book_noRGBD_lora_joint_delta_gripper_absolute/piper_right_book_noRGBD_joint_delta_gripper_absolute_bs32/5000
```

## 2. piper-openpi 控制端环境

该环境只用于 Piper 控制端，不用于训练，也不用于启动模型服务。

创建 conda 环境：

```bash
conda create -y -n piper-openpi python=3.10
conda activate piper-openpi
```

安装控制端依赖：

```bash
cd <openpi_repo>
python -m pip install --no-deps -e packages/openpi-client
python -m pip install -r examples/piper/requirements.txt
conda install -y pillow
```

`examples/piper/requirements.txt` 已固定 `numpy==1.26.4` 和 `opencv-python==4.10.0.84`。不要单独执行 `python -m pip install opencv-python`，否则 pip 可能把 `numpy` 升级到 2.x，和 `openpi-client` 的 `numpy<2.0.0` 约束冲突。

当前控制端依赖固定为：

```text
numpy==1.26.4
opencv-python==4.10.0.84
pyrealsense2==2.58.2.10647
dm-tree==0.1.10
msgpack==1.2.0
websockets==16.0
pillow==12.2.0
tyro==1.0.14
```

这次补充了 `pyrealsense2`。如果使用 `--camera-backend=realsense`，缺少它会在启动时出现：

```text
ModuleNotFoundError: No module named 'pyrealsense2'
```

接入 Piper SDK。当前项目约定把本机 Piper SDK 放到：

```text
/home/server/project/piper/openpi/piper_sdk
```

如果仓库里还没有 `third_party/piper_sdk`，先建立软链接：

```bash
mkdir -p third_party
ln -s /home/server/project/piper/openpi/piper_sdk third_party/piper_sdk
```

安装 Piper SDK：

```bash
python -m pip install -e third_party/piper_sdk
```

验证控制端依赖：

```bash
conda run -n piper-openpi python -c "import cv2, numpy, pyrealsense2, openpi_client, piper_sdk, tyro; from PIL import Image; print('piper-client-env-ok')"
conda run -n piper-openpi python -m pip check
```

查看 Piper 控制端参数：

```bash
conda run -n piper-openpi python examples/piper/main.py --help
```

## 3. 启动顺序

先启动 openpi 推理服务端：

```bash
cd <openpi_repo>
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --no-dev scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_piper_right_book_noRGBD_lora_joint_delta_gripper_absolute \
  --policy.dir=/mnt/disk/checkpoints/openpi/pi05_piper_right_book_noRGBD_lora_joint_delta_gripper_absolute/piper_right_book_noRGBD_joint_delta_gripper_absolute_bs32/5000
```

再启动 Piper 控制端：

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
  --can-name=can_right \
  --open-loop-horizon=15 \
  --control-hz=15 \
  --move-speed-percent=30 \
  --gripper-open-mm=70.0 \
  --gripper-closed-mm=0.0 \
  --gripper-threshold-mm=35.0 \
  --no-dry-run \
  --enable-robot
```

真机启动前请先确认相机、CAN、机械臂工作空间和急停状态。布尔参数使用 tyro 写法：开启直接写 `--enable-robot`，关闭默认 dry-run 写 `--no-dry-run`，不要写 `=True` 或 `=False`。

## 4. 常见问题

- `uv: command not found`：执行 `export PATH="$HOME/.local/bin:$PATH"`，或重新打开 shell。
- `openpi_client` 导入失败：在 `piper-openpi` 环境里重新执行 `python -m pip install --no-deps -e packages/openpi-client`。
- `piper_sdk` 导入失败：确认 `third_party/piper_sdk` 软链接存在，并重新执行 `python -m pip install -e third_party/piper_sdk`。
- 推理服务 OOM：降低 `XLA_PYTHON_CLIENT_MEM_FRACTION`，例如改成 `0.85`。
- 控制端连接不上服务端：确认服务端端口和控制端 `--host/--port` 一致。
