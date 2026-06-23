# Piper 启动训练流程

进入 openpi 仓库根目录：

```bash
cd <openpi_repo>
```

## 1. 训练配置选择

兼容旧 checkpoint 的配置：

```text
pi05_piper_right_book_v5_lora
```

动作策略：

```text
前 6 维关节：相对关节位移
第 7 维夹爪：连续绝对值，不做 delta
```

新实验配置：

```text
pi05_piper_right_book_v5_lora_all_delta
```

动作策略：

```text
7 维动作全部启用 delta action
前 6 维关节：相对关节位移
第 7 维夹爪：相对夹爪位移
```

两者夹爪都不启用阈值化，最终仍保持连续输出：

```python
binarize_gripper_outputs=False
```

如果要继续使用之前已经训练好的 checkpoint，policy server 必须继续使用：

```text
--policy.config=pi05_piper_right_book_v5_lora
```

如果要训练“夹爪也做相对动作”的新版模型，使用：

```text
pi05_piper_right_book_v5_lora_all_delta
```

## 2. 训练前计算 norm stats

训练旧配置前计算旧配置对应的 norm stats：

```bash
uv run --no-dev scripts/compute_norm_stats.py \
  --config-name=pi05_piper_right_book_v5_lora
```

输出位置：

```text
assets/pi05_piper_right_book_v5_lora/piper_right_book_v5/norm_stats.json
```

旧配置说明：

- `actions` 前 6 维是 delta 后的相对关节位移统计。
- 第 7 维夹爪保持连续绝对值统计。

训练新版全 delta 配置前，单独计算新版 norm stats：

```bash
uv run --no-dev scripts/compute_norm_stats.py \
  --config-name=pi05_piper_right_book_v5_lora_all_delta
```

输出位置：

```text
assets/pi05_piper_right_book_v5_lora_all_delta/piper_right_book_v5_all_delta/norm_stats.json
```

新版配置说明：

- `compute_norm_stats.py` 会走当前 config 的数据 transform。
- 因为新版 config 启用了 `DeltaActions(make_bool_mask(7))`，所以计算出来的 `actions` norm stats 应该是 7 维全部 delta 后的相对动作统计。
- 第 7 维夹爪也做 delta，所以夹爪统计是相对夹爪位移；推理输出链路会再转回绝对夹爪米值。
- 夹爪是否阈值化不影响 norm stats；当前阈值化关闭。

快速检查：

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("assets/pi05_piper_right_book_v5_lora/piper_right_book_v5/norm_stats.json")
data = json.loads(p.read_text())["norm_stats"]

print("state mean first7:", data["state"]["mean"][:7])
print("actions mean first7:", data["actions"]["mean"][:7])
print("actions std first7:", data["actions"]["std"][:7])
PY
```

## 3. 前台启动训练旧配置

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
uv run --no-dev scripts/train.py pi05_piper_right_book_v5_lora \
  --exp-name=piper_right_book_v5_lora_bs32 \
  --batch-size=32 \
  --overwrite
```

## 4. 后台启动训练旧配置

```bash
mkdir -p logs

WANDB_MODE=offline \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
nohup uv run --no-dev scripts/train.py pi05_piper_right_book_v5_lora \
  --exp-name=piper_right_book_v5_lora_bs32 \
  --batch-size=32 \
  --overwrite \
  > logs/pi05_piper_right_book_v5_lora_bs32.log 2>&1 &
```

## 5. 启动新版全 delta 训练

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
uv run --no-dev scripts/train.py pi05_piper_right_book_v5_lora_all_delta \
  --exp-name=piper_right_book_v5_lora_all_delta_bs32 \
  --batch-size=32 \
  --overwrite
```

后台启动：

```bash
mkdir -p logs

WANDB_MODE=offline \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
nohup uv run --no-dev scripts/train.py pi05_piper_right_book_v5_lora_all_delta \
  --exp-name=piper_right_book_v5_lora_all_delta_bs32 \
  --batch-size=32 \
  --overwrite \
  > logs/pi05_piper_right_book_v5_lora_all_delta_bs32.log 2>&1 &
```

## 6. 查看日志

```bash
tail -f logs/pi05_piper_right_book_v5_lora_bs32.log
```

新版日志：

```bash
tail -f logs/pi05_piper_right_book_v5_lora_all_delta_bs32.log
```

## 7. 恢复训练

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
uv run --no-dev scripts/train.py pi05_piper_right_book_v5_lora \
  --exp-name=piper_right_book_v5_lora_bs32 \
  --batch-size=32 \
  --resume
```

恢复训练时不要加 `--overwrite`。

## 8. OOM 时调 batch size

如果 `batch-size=32` 仍然 OOM，继续降低：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
uv run --no-dev scripts/train.py pi05_piper_right_book_v5_lora \
  --exp-name=piper_right_book_v5_lora_bs16 \
  --batch-size=16 \
  --overwrite
```

如果显存还有余量，可以提高 batch size，例如：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
uv run --no-dev scripts/train.py pi05_piper_right_book_v5_lora \
  --exp-name=piper_right_book_v5_lora_bs64 \
  --batch-size=64 \
  --overwrite
```

## 9. 查看训练进程

```bash
pgrep -af "scripts/train.py pi05_piper_right_book_v5_lora"
```

## 10. 停止训练

```bash
pkill -f "scripts/train.py pi05_piper_right_book_v5_lora"
```

## 11. Checkpoint 输出位置

```text
checkpoints/pi05_piper_right_book_v5_lora/<exp-name>
checkpoints/pi05_piper_right_book_v5_lora_all_delta/<exp-name>
```

例如：

```text
checkpoints/pi05_piper_right_book_v5_lora/piper_right_book_v5_lora_bs32
```
