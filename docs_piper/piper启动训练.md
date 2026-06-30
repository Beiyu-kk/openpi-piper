# Piper 启动训练

## 0. 进入仓库

```bash
cd /home/server/project/piper/openpi/openpi-piper
```

本机路径：

```text
数据根目录：/mnt/disk/Dataset/piper_data/data
checkpoint 根目录：/mnt/disk/checkpoints/openpi
```

## 1. 推荐训练配置

当前单独训练 RGBD_V1_fixed 转换后的 LeRobot 数据集，使用：

```text
pi05_piper_right_book_RGBD_V1_fixed_lora_joint_delta_gripper_absolute
```

对应数据集：

```text
/mnt/disk/Dataset/piper_data/data/lerobot_v21/piper_right_book_RGBD_V1_fixed
```

动作策略：

```text
前 6 维关节：相对位置 delta 训练
第 7 维夹爪：连续绝对位置训练
夹爪：不做阈值化
batch_size：32
num_workers：4
action_horizon：30
```

## 2. 确认数据集存在

```bash
test -d /mnt/disk/Dataset/piper_data/data/lerobot_v21/piper_right_book_RGBD_V1_fixed && echo "dataset ok"
```

如果目录不存在，先转换原始 hdf5 数据集：

```bash
.venv/bin/python examples/piper/convert_piper_rgbd_hdf5_to_lerobot.py
```

## 3. 计算 norm stats

训练前必须先计算统计文件：

```bash
.venv/bin/python scripts/compute_norm_stats_local_assets.py \
  --config-name=pi05_piper_right_book_RGBD_V1_fixed_lora_joint_delta_gripper_absolute
```

主要生成位置：

```text
assets/pi05_piper_right_book_RGBD_V1_fixed_lora_joint_delta_gripper_absolute/piper_right_book_RGBD_V1_fixed_joint_delta_gripper_absolute/norm_stats.json
```

这个路径是训练配置里的 `asset_id` 路径，训练启动时会从这里加载 norm stats。

脚本还会额外写一份到本地数据集名路径：

```text
assets/pi05_piper_right_book_RGBD_V1_fixed_lora_joint_delta_gripper_absolute/piper_right_book_RGBD_V1_fixed/norm_stats.json
```

确认：

```bash
test -f assets/pi05_piper_right_book_RGBD_V1_fixed_lora_joint_delta_gripper_absolute/piper_right_book_RGBD_V1_fixed_joint_delta_gripper_absolute/norm_stats.json && echo "norm stats ok"
```

## 4. 启动训练

前台启动：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
.venv/bin/python scripts/train.py pi05_piper_right_book_RGBD_V1_fixed_lora_joint_delta_gripper_absolute \
  --exp-name=piper_right_book_RGBD_V1_fixed_joint_delta_gripper_absolute_bs32 \
  --batch-size=32 \
  --overwrite
```

后台启动：

```bash
mkdir -p logs

WANDB_MODE=offline \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
nohup .venv/bin/python scripts/train.py pi05_piper_right_book_RGBD_V1_fixed_lora_joint_delta_gripper_absolute \
  --exp-name=piper_right_book_RGBD_V1_fixed_joint_delta_gripper_absolute_bs32 \
  --batch-size=32 \
  --overwrite \
  > logs/piper_right_book_RGBD_V1_fixed_joint_delta_gripper_absolute_bs32.log 2>&1 &
```

checkpoint 输出目录：

```text
/mnt/disk/checkpoints/openpi/pi05_piper_right_book_RGBD_V1_fixed_lora_joint_delta_gripper_absolute/piper_right_book_RGBD_V1_fixed_joint_delta_gripper_absolute_bs32
```

## 5. 查看训练

看日志：

```bash
tail -f logs/piper_right_book_RGBD_V1_fixed_joint_delta_gripper_absolute_bs32.log
```

看进程：

```bash
pgrep -af "scripts/train.py pi05_piper_right_book_RGBD_V1_fixed_lora_joint_delta_gripper_absolute"
```

看 checkpoint：

```bash
find /mnt/disk/checkpoints/openpi/pi05_piper_right_book_RGBD_V1_fixed_lora_joint_delta_gripper_absolute/piper_right_book_RGBD_V1_fixed_joint_delta_gripper_absolute_bs32 \
  -maxdepth 1 -type d | sort
```

## 6. 恢复训练

恢复训练时不要加 `--overwrite`，改用 `--resume`：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
.venv/bin/python scripts/train.py pi05_piper_right_book_RGBD_V1_fixed_lora_joint_delta_gripper_absolute \
  --exp-name=piper_right_book_RGBD_V1_fixed_joint_delta_gripper_absolute_bs32 \
  --batch-size=32 \
  --resume
```

## 7. 停止训练

```bash
pkill -f "scripts/train.py pi05_piper_right_book_RGBD_V1_fixed_lora_joint_delta_gripper_absolute"
```

## 8. 可选：合并数据集训练

如果要训练 noRGBD + RGBD 合并数据集，使用旧配置：

```text
pi05_piper_right_book_noRGBD_RGBD_lora_joint_delta_gripper_absolute
```

启动命令只需要把上面的 config 和 exp-name 换成：

```text
config：pi05_piper_right_book_noRGBD_RGBD_lora_joint_delta_gripper_absolute
exp-name：piper_right_book_noRGBD_RGBD_joint_delta_gripper_absolute_bs32
```
