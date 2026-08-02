#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

SERIAL_ARGS=()
if [[ -n "${SERIAL_PORT:-}" ]]; then
  SERIAL_ARGS=(--serial-port "$SERIAL_PORT" --serial-baud "${SERIAL_BAUD:-115200}")
fi

# IMX415 exposes ISP scaler 4 as /dev/video4. Requesting 960x540 there keeps the
# sensor/ISP at 1080p30 while avoiding single-core GStreamer scale/conversion.
# OpenCV converts NV12 to the color frame required by E25 and laser detection.
exec python3 -u main.py \
  --backend gstreamer \
  --device "${DEVICE:-/dev/video4}" \
  --set-subdev-format 0 \
  --size "${SIZE:-960x540}" \
  --output-size "${OUTPUT_SIZE:-960x540}" \
  --fps "${FPS:-30}" \
  --max-processing-fps "${MAX_PROCESSING_FPS:-30}" \
  --format NV12 \
  --capture-mode "${CAPTURE_MODE:-raw}" \
  --raw-detect-color "${RAW_DETECT_COLOR:-1}" \
  --awisp 1 \
  --largemode 0 \
  --detector rect \
  --detect-scale "${DETECT_SCALE:-0.5}" \
  --detect-multi-pass "${DETECT_MULTI_PASS:-1}" \
  --display 0 \
  --display-mode color \
  --stream-port "${STREAM_PORT:-8081}" \
  --stream-scale "${STREAM_SCALE:-1.0}" \
  --stream-every "${STREAM_EVERY:-1}" \
  --stream-quality "${STREAM_QUALITY:-75}" \
  --print-interval "${PRINT_INTERVAL:-1}" \
  --control "${CONTROL:-0}" \
  --invert-yaw "${INVERT_YAW:-1}" \
  --invert-pitch "${INVERT_PITCH:-0}" \
  "${SERIAL_ARGS[@]}" \
  "$@"
