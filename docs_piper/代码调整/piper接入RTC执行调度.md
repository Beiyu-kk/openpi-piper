# Piper 接入 RTC 推理执行说明

本文档记录 `openpi-piper-rtc` 中新的 Piper RTC 接入方式。

## 1. 接入结论

普通 Piper 控制端保持在：

```text
examples/piper/main.py
```

该文件已恢复为不包含 RTC 的 open-loop 推理执行逻辑，通过 `--open-loop-horizon` 控制每次收到 action chunk 后连续执行多少步。

RTC 使用独立脚本：

```text
examples/piper/main_rtc.py
```

不要再向 `examples/piper/main.py` 传 `--rtc-enabled`、`--rtc-execute-horizon` 或 `--rtc-inference-delay`。这些固定参数已经移除。

## 2. RTC 语义

参考 `real-time-chunking-kinetix` 的实时执行语义，真实部署中的 `inference_delay` 不应该是固定手填值，而应该表示：

```text
策略服务生成新 action chunk 期间，机器人已经实际执行了多少个旧 chunk 动作。
```

因此新的 RTC 脚本使用动态 delay：

1. 控制端先同步拿到第一段 action chunk，开始执行。
2. 后台线程异步请求下一段 action chunk。
3. 请求期间控制循环不暂停，继续按 `--control-hz` 执行动作。
4. 新 chunk 返回后，脚本统计这段时间里实际发送了多少个动作，得到 `delay_steps`。
5. 新 chunk 从 `new_chunk[delay_steps]` 开始接入。

日志示例：

```text
RTC accepted chunk at step 42: delay_steps=3 elapsed_ms=185.7 server_timing={'infer_ms': 170.2}
```

这里 `delay_steps=3` 表示新 chunk 推理期间，机械臂已经执行了 3 个旧动作，所以新 chunk 的前 3 个动作已经过期，需要跳过。

## 3. 代码改动

新增文件：

```text
examples/piper/main_rtc.py
examples/piper/piper_rtc_test.py
```

`main_rtc.py` 复用 `main.py` 中已有的：

- 相机创建和读取
- Piper 机械臂控制
- action chunk shape 校验
- policy observation 构造
- 夹爪阈值化和闭合保持

新增核心类：

```python
RealtimeActionStream
```

它维护当前 action stream，并用实际执行步数计算动态 delay：

```python
delay_steps = current_action_step - request.start_action_step
cursor = delay_steps
```

如果新 chunk 返回时 `delay_steps >= action_horizon`，说明策略推理太慢，旧 chunk 已经执行完，脚本会报错，而不是继续发送错位动作。

## 4. 启动方式

先启动 policy server：

```bash
cd <openpi_repo>
source .venv/bin/activate

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --no-dev scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_piper_right_book_noRGBD_lora_joint_delta_gripper_absolute \
  --policy.dir=/mnt/disk/checkpoints/openpi/pi05_piper_right_book_noRGBD_lora_joint_delta_gripper_absolute/piper_right_book_noRGBD_joint_delta_gripper_absolute_bs32/5000
```

再启动 RTC 控制端：

```bash
cd <openpi_repo>
conda activate piper-openpi

python examples/piper/main_rtc.py \
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
  --control-hz=15 \
  --move-speed-percent=30 \
  --gripper-open-mm=70.0 \
  --gripper-closed-mm=0.0 \
  --gripper-threshold-mm=35.0 \
  --no-dry-run \
  --enable-robot
```

真机前建议先去掉 `--no-dry-run` 和 `--enable-robot`，确认相机、服务端连接和 RTC 日志正常。

## 5. 测试

RTC 动态 delay 单测：

```bash
conda run -n pi python -m pytest examples/piper/piper_rtc_test.py -q
```

Piper 控制端完整相关测试：

```bash
conda run -n pi python -m pytest examples/piper/piper_control_test.py examples/piper/piper_rtc_test.py -q
```

已覆盖：

- 新 chunk 返回前已经执行 3 步时，从 `new_chunk[3]` 接入。
- 新 chunk 立即返回时，从 `new_chunk[0]` 接入。
- 新 chunk 返回过晚、旧 chunk 已执行完时直接报错。

## 6. 限制

当前仍然不是模型内 RTC guidance。openpi policy server 仍按原始方式生成 action chunk；`main_rtc.py` 只在客户端侧解决真实推理延迟导致的时间错位。

如果要复现 `FlowPolicy.realtime_action()` 的生成时 guidance，还需要继续改 openpi 模型采样路径，例如：

```text
src/openpi/models/pi0.py::sample_actions
src/openpi/models_pytorch/pi0_pytorch.py::sample_actions
```

并扩展 policy/websocket 请求，使服务端生成动作时能看到上一轮 `prev_action_chunk`。
