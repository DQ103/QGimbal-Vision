#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

# The Camera 4K overlay exposes full 4K through /dev/video1 with largemode=1,
# but 1080p on /dev/video0 avoids the dual-stream ISP path and is appropriate
# for real-time OpenCV rectangle detection.
exec python3 -u main.py \
  --backend gstreamer \
  --device "${DEVICE:-/dev/video0}" \
  --set-subdev-format 0 \
  --size "${SIZE:-1920x1080}" \
  --output-size "${OUTPUT_SIZE:-960x540}" \
  --fps "${FPS:-30}" \
  --max-processing-fps "${MAX_PROCESSING_FPS:-30}" \
  --format NV12 \
  --capture-mode "${CAPTURE_MODE:-bgr}" \
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
  "$@"
