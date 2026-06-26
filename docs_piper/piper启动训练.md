# Piper 启动训练

进入 openpi 仓库：

```bash
cd <openpi_repo>
```

当前新机器的数据集和 checkpoint 根目录：

```text
数据集：/mnt/c9dd2903-1a5c-4ec3-b146-9f8ee2434744/Dataset/piper_data/data
LeRobot v2.1 noRGBD：/mnt/c9dd2903-1a5c-4ec3-b146-9f8ee2434744/Dataset/piper_data/data/lerobot_v21/piper_right_book_noRGBD
checkpoint：/mnt/c9dd2903-1a5c-4ec3-b146-9f8ee2434744/checkpoints/openpi
```

## 推荐配置

当前推荐训练配置：

```text
pi05_piper_right_book_noRGBD_lora_joint_delta_gripper_absolute
```

动作含义：

```text
前 6 维关节：相对动作 delta
第 7 维夹爪：连续绝对开度
action chunk：从未来 1 帧开始取，避开 action_t == state_t 的零 delta 问题
```

## 1. 计算 norm stats

```bash
uv run --no-dev scripts/compute_norm_stats.py \
  --config-name=pi05_piper_right_book_noRGBD_lora_joint_delta_gripper_absolute
```

训练会读取这里的统计文件：

```text
assets/pi05_piper_right_book_noRGBD_lora_joint_delta_gripper_absolute/piper_right_book_noRGBD_joint_delta_gripper_absolute/norm_stats.json
```

## 2. 启动训练

前台启动：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
uv run --no-dev scripts/train.py pi05_piper_right_book_noRGBD_lora_joint_delta_gripper_absolute \
  --exp-name=piper_right_book_noRGBD_joint_delta_gripper_absolute_bs32 \
  --batch-size=32 \
  --overwrite
```

后台启动：

```bash
mkdir -p logs

WANDB_MODE=offline \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
nohup uv run --no-dev scripts/train.py pi05_piper_right_book_noRGBD_lora_joint_delta_gripper_absolute \
  --exp-name=piper_right_book_noRGBD_joint_delta_gripper_absolute_bs32 \
  --batch-size=32 \
  --overwrite \
  > logs/piper_right_book_noRGBD_joint_delta_gripper_absolute_bs32.log 2>&1 &
```

## 3. 查看训练

```bash
tail -f logs/piper_right_book_noRGBD_joint_delta_gripper_absolute_bs32.log
```

```bash
pgrep -af "scripts/train.py pi05_piper_right_book_noRGBD_lora_joint_delta_gripper_absolute"
```

checkpoint 输出位置：

```text
/mnt/c9dd2903-1a5c-4ec3-b146-9f8ee2434744/checkpoints/openpi/pi05_piper_right_book_noRGBD_lora_joint_delta_gripper_absolute/piper_right_book_noRGBD_joint_delta_gripper_absolute_bs32
```

## 4. 恢复或停止

恢复训练：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
uv run --no-dev scripts/train.py pi05_piper_right_book_noRGBD_lora_joint_delta_gripper_absolute \
  --exp-name=piper_right_book_noRGBD_joint_delta_gripper_absolute_bs32 \
  --batch-size=32 \
  --resume
```

停止训练：

```bash
pkill -f "scripts/train.py pi05_piper_right_book_noRGBD_lora_joint_delta_gripper_absolute"
```

恢复训练时不要加 `--overwrite`。
