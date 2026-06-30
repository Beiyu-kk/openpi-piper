# Piper OBEYED-VLA Adapter

This directory contains a standalone middle layer between the existing Piper openpi code and the OBEYED-VLA perception grounding service.

It does not modify the original Piper training, inference, or policy transform files. The adapter converts the two existing Piper image inputs:

- `observation.images.top_head` / `observation/image`
- `observation.images.hand_right` / `observation/right_wrist_image`

into OBEYED-VLA grounded images, restores dimmed original-image context behind
the OBEYED mask overlay, then feeds them back into the same openpi input keys.

## Files

- `adapter.py`: HTTP client, passthrough fallback, and `.npy` multipart packing/parsing.
- `convert_lerobot_dataset.py`: offline LeRobot v2.1 dataset converter.
- `visualize_lerobot_conversion.py`: raw-vs-OBEYED frame comparison exporter.
- `main.py`: standalone Piper inference client with OBEYED-VLA perception before policy requests.
- `adapter_test.py`, `main_test.py`, `visualize_lerobot_conversion_test.py`: focused tests for the middle layer.

## Runtime Modes

### `http`

Default mode. The adapter sends two RGB frames to an already running OBEYED-VLA FastAPI service:

```text
Piper cameras / LeRobot videos
  -> adapter HTTP POST /process
  -> OBEYED-VLA service
  -> grounded base/wrist images with dimmed original context
  -> existing openpi policy input keys
```

This keeps OBEYED-VLA dependencies separate from the openpi environment.
The dimmed-context postprocess avoids pure black backgrounds hiding black target
areas while still emphasizing the selected masks and blue outlines.

### `passthrough`

Debug mode. The adapter returns the original RGB images unchanged. This is useful for testing the full control or dataset pipeline without running OBEYED-VLA.

### Fallback

By default, `http` mode falls back to passthrough if the perception service is unavailable or times out. Use `--no-fallback` or `--perception-no-fallback` to fail hard instead.

## Start OBEYED-VLA Service

Run this in the OBEYED-VLA repository:

```bash
cd /home/server/project/piper/openpi/OBEYED_VLA
MPLCONFIGDIR=/tmp/matplotlib YOLO_CONFIG_DIR=/tmp/Ultralytics \
  .venv/bin/python -m uvicorn perception_service_fastapi:app --host 127.0.0.1 --port 18080
```

The adapter defaults to:

```text
http://127.0.0.1:18080/process
```

The current OBEYED-VLA service is configured with `use_vlm=True`. It calls an
OpenAI-compatible VLM endpoint at `http://127.0.0.1:8080/v1`.

For a local smoke run without a downloaded Qwen-VL checkpoint, start the shim:

```bash
cd /home/server/project/piper/openpi/OBEYED_VLA
MPLCONFIGDIR=/tmp/matplotlib YOLO_CONFIG_DIR=/tmp/Ultralytics \
  VLM_SERVER_MODE=heuristic \
  .venv/bin/python -m uvicorn vlm_openai_server:app --host 127.0.0.1 --port 8080
```

This exercises the original VLM/Cutie code path, but the mask choice is
heuristic. For real semantic selection such as "the red book" or one specific
book among many, run the shim with `VLM_SERVER_MODE=qwen` and
`VLM_MODEL_PATH=/mnt/disk/checkpoints/obeyed_vla/vlm/<qwen-model>`.

## Offline LeRobot Conversion

Convert the current LeRobot v2.1 dataset into a new grounded dataset:

```bash
cd /home/server/project/piper/openpi/openpi-piper

.venv/bin/python examples/piper_obeyed_vla/convert_lerobot_dataset.py \
  --input-root /mnt/disk/Dataset/piper_data/data/lerobot_v21/piper_right_book_RGBD_V1_fixed \
  --output-root /mnt/disk/Dataset/piper_data/data/lerobot_v21/piper_right_book_RGBD_V1_fixed_obeyed_vla \
  --mode http \
  --endpoint http://127.0.0.1:18080/process \
  --select-objects "book" \
  --overwrite \
  --no-fallback
```

Smoke test without OBEYED-VLA:

```bash
.venv/bin/python examples/piper_obeyed_vla/convert_lerobot_dataset.py \
  --input-root /mnt/disk/Dataset/piper_data/data/lerobot_v21/piper_right_book_RGBD_V1_fixed \
  --output-root /mnt/disk/Dataset/piper_data/data/lerobot_v21/piper_right_book_RGBD_V1_fixed_obeyed_vla_smoke \
  --mode passthrough \
  --max-episodes 1 \
  --limit-frames 3 \
  --overwrite
```

The converter copies `meta/`, `data/`, and `norm_stats.json`, then writes processed videos under the same LeRobot v2.1 video layout.

Preview one episode with the VLM path:

```bash
.venv/bin/python examples/piper_obeyed_vla/convert_lerobot_dataset.py \
  --input-root /mnt/disk/Dataset/piper_data/data/lerobot_v21/piper_right_book_RGBD_V1_fixed \
  --output-root /mnt/disk/Dataset/piper_data/data/lerobot_v21/piper_right_book_RGBD_V1_fixed_obeyed_vlm_dim_preview \
  --mode http \
  --endpoint http://127.0.0.1:18080/process \
  --select-objects "book" \
  --max-episodes 1 \
  --limit-frames 20 \
  --overwrite \
  --no-fallback
```

Do not use this preview directory for LoRA training: `meta/` and `data/` are
copied in full, but only episode 0 has converted videos. For LoRA training, run
the conversion without `--max-episodes` and without `--limit-frames`.

Export raw-vs-OBEYED comparison images:

```bash
.venv/bin/python examples/piper_obeyed_vla/visualize_lerobot_conversion.py \
  --input-root /mnt/disk/Dataset/piper_data/data/lerobot_v21/piper_right_book_RGBD_V1_fixed \
  --converted-root /mnt/disk/Dataset/piper_data/data/lerobot_v21/piper_right_book_RGBD_V1_fixed_obeyed_vlm_dim_preview \
  --output-dir /mnt/disk/checkpoints/openpi/obeyed_vla_vlm_dim_preview_frames \
  --episode-index 0 \
  --frames 0 10 19
```

## Online Piper Inference

Use the standalone grounded inference entrypoint instead of `examples/piper/main.py`:

```bash
cd /home/server/project/piper/openpi/openpi-piper

python examples/piper_obeyed_vla/main.py \
  --host=127.0.0.1 \
  --port=8000 \
  --perception-mode=http \
  --perception-endpoint=http://127.0.0.1:18080/process \
  --select-objects="book" \
  --open-loop-horizon=15
```

The perception adapter runs only when a new action chunk is requested. It does not run every control step, which keeps control latency bounded by the existing open-loop action chunk behavior.

## Verification

Run the adapter tests:

```bash
cd /home/server/project/piper/openpi/openpi-piper
.venv/bin/python -m pytest \
  examples/piper_obeyed_vla/adapter_test.py \
  examples/piper_obeyed_vla/main_test.py \
  examples/piper_obeyed_vla/visualize_lerobot_conversion_test.py \
  -q
```
