# Piper Right Book V5 接入 openpi 微调记录

本文档记录将本地 LeRobot v2.1 数据集接入 openpi 并使用 `pi05_base` + LoRA 微调的完整步骤。

## 1. 数据集检查

数据集路径：

```bash
/mnt/c9dd2903-1a5c-4ec3-b146-9f8ee2434744/Dataset/piper_data/data/lerobot_v21/piper_right_book_noRGBD
```

已检查 `meta/info.json`、`meta/tasks.jsonl` 和 `meta/episodes.jsonl`：

- `codebase_version`: `v2.1`
- `robot_type`: `agilex`
- `total_episodes`: `100`
- `total_frames`: `41928`
- `fps`: `30`
- prompt 保存在 LeRobot task 中：`抓起书本放到另外一个格子里`
- 图像字段：
  - `observation.images.top_head`
  - `observation.images.hand_right`
- 状态字段：`observation.state`，7 维
- 动作字段：`action`，7 维

## 2. 新增 Piper 输入输出 transform

新增文件：

```bash
src/openpi/policies/piper_policy.py
```

映射关系：

```text
observation.images.top_head   -> image/base_0_rgb
observation.images.hand_right -> image/right_wrist_0_rgb
observation.state             -> state
action                        -> actions
task_index/tasks.jsonl        -> prompt
```

`pi05` 模型输入固定包含三路图像 key：`base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`。当前数据只有头部相机和右腕相机，所以：

- `base_0_rgb`: 使用 `top_head`
- `right_wrist_0_rgb`: 使用 `hand_right`
- `left_wrist_0_rgb`: 使用零图占位，`image_mask=False`

## 3. 动作与夹爪处理

该数据集 7 维动作解释为：

```text
前 6 维：右臂关节绝对位置
第 7 维：夹爪连续位置值
```

训练输入使用：

```python
_transforms.DeltaActions(_transforms.make_bool_mask(6, -1))
```

含义：

- 旧配置 `pi05_piper_right_book_v5_lora` 中，前 6 维关节从绝对位置转换为相对位移。
- 第 7 维夹爪保持连续绝对值，不做 delta，也不做 `0/1` 阈值化。
- 新配置 `pi05_piper_right_book_v5_lora_all_delta` 使用 `_transforms.make_bool_mask(7)`，7 维动作全部做 delta。

推理输出链路使用：

```python
_transforms.AbsoluteActions(_transforms.make_bool_mask(6, -1))
```

含义：

- 旧配置中，前 6 维相对动作转回绝对关节位置，第 7 维夹爪保持绝对连续米值。
- 新配置中，7 维相对动作全部转回绝对目标位置。
- `PiperOutputs(binarize_gripper=False)` 保留最终夹爪的连续输出，不再做 `0/1` 阈值化。

## 4. 新增训练配置

修改文件：

```bash
src/openpi/training/config.py
```

新增数据配置类：

```python
LeRobotPiperDataConfig
```

新增 train config：

```text
pi05_piper_right_book_v5_lora
```

关键参数：

```python
model=pi0_config.Pi0Config(
    pi05=True,
    action_dim=32,
    action_horizon=15,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
)
weight_loader=weight_loaders.CheckpointWeightLoader(
    "gs://openpi-assets/checkpoints/pi05_base/params"
)
ema_decay=None
num_train_steps=30_000
save_interval=5_000
keep_period=5_000
batch_size=128
num_workers=4
```

LoRA freeze filter 使用与模型配置一致的 `Pi0Config(...).get_freeze_filter()`，即冻结原始主干权重，只训练 LoRA 参数。

实际启动验证中，`batch_size=128` 在第一步编译时 OOM，XLA 估计 rematerialization 后仍需约 49-62 GiB，超过本机 48GB 显存。最终将默认 `batch_size` 调整为 `96`，并已确认能够完成第 0 步训练并持续推进。

## 5. Norm Stats

数据集目录已有 `norm_stats.json`，内容为 openpi 需要的格式，包含：

- `state`
- `actions`

并且已 pad 到 32 维。已复制到训练配置默认读取位置：

```bash
assets/pi05_piper_right_book_v5_lora/piper_right_book_v5/norm_stats.json
```

这样 `LeRobotPiperDataConfig(... assets=AssetsConfig(asset_id="piper_right_book_v5"))` 会直接加载该统计文件。

## 6. 测试与验证

新增测试：

```bash
src/openpi/policies/piper_policy_test.py
```

覆盖内容：

- `top_head` 映射到 `base_0_rgb`
- `hand_right` 映射到 `right_wrist_0_rgb`
- `left_wrist_0_rgb` 零图占位且 mask 为 `False`
- 旧配置前 6 维关节做 delta，夹爪保持绝对连续值
- 新配置 7 维动作全部做 delta
- 输出阶段按对应配置转回绝对位置
- 最终夹爪保持连续输出，部署端只做物理范围裁剪

首次运行测试时，`uv` 需要创建 `.venv` 并下载依赖；本机网络在下载 dev 依赖 `ruff==0.11.12` 时超时。后续启动训练使用 `--no-dev` 避免下载 dev 依赖。

数据加载 smoke test 已通过，确认：

- `video_backend` 为 `pyav`
- `actions` shape 为 `(batch, 15, 32)`
- `state` shape 为 `(batch, 32)`
- prompt token shape 为 `(batch, 200)`
- 图像 key 为 `base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`
- `left_wrist_0_rgb` mask 为 `False`
- `right_wrist_0_rgb` mask 为 `True`

本机缺少 `torchcodec` 所需的系统 FFmpeg 动态库，且无法无密码 sudo 安装系统包。因此 Piper 数据配置显式使用 LeRobot 的 `pyav` 视频解码后端，避免训练启动时触发 `torchcodec` 动态库错误。

## 7. 启动训练

工作目录：

```bash
cd /home/server/project/piper/openpi/openpi-piper
```

推荐启动命令：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
uv run --no-dev scripts/train.py pi05_piper_right_book_v5_lora \
  --exp-name=piper_right_book_v5_lora_bs96 \
  --overwrite
```

说明：

- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` 限制 JAX 最多预分配约 95% GPU 显存，避免系统完全卡死。
- 本机 GPU 是 RTX 4090 48GB，`batch_size=128` 已验证 OOM；`batch_size=96` 已验证可启动并持续训练。
- 若后续仍出现 OOM，继续降到 64：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
uv run --no-dev scripts/train.py pi05_piper_right_book_v5_lora \
  --exp-name=piper_right_book_v5_lora_bs64 \
  --batch-size=64 \
  --overwrite
```

后台启动命令：

```bash
WANDB_MODE=offline XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
nohup uv run --no-dev scripts/train.py pi05_piper_right_book_v5_lora \
  --exp-name=piper_right_book_v5_lora_bs96 \
  --overwrite \
  > logs/pi05_piper_right_book_v5_lora_bs96.log 2>&1 &
```

## 8. 输出位置

默认 checkpoint 目录：

```bash
/mnt/c9dd2903-1a5c-4ec3-b146-9f8ee2434744/checkpoints/openpi/pi05_piper_right_book_v5_lora_joint_delta_gripper_absolute/piper_right_book_joint_delta_gripper_absolute_bs32
```

每 5000 step 保存一次 checkpoint，训练总步数 30000。
