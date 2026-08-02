#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_ROOT=${MODEL_ROOT:-/home/radxa/awnpu_model_zoo}
YOLO_DEMO=${YOLO_DEMO:-$MODEL_ROOT/examples/yolov8/build_native/yolov8_demo_a733}
YOLO_NB=${YOLO_NB:-$MODEL_ROOT/examples/yolov8/model/yolov8n_6_uint8_a733.nb}
LD_DIR=${LD_DIR:-$MODEL_ROOT/common/npuruntime/lib_linux_aarch64/A733}
RUNTIME_LD_PATH="${LD_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$ROOT_DIR"

exec python3 -u main.py \
  --display "${DISPLAY_WINDOW:-0}" \
  --size "${SIZE:-1920x1080}" \
  --fps "${FPS:-30}" \
  --format NV12 \
  --capture-mode raw \
  --awisp "${AWISP:-0}" \
  --largemode "${LARGEMODE:-0}" \
  --detect-scale "${DETECT_SCALE:-0.25}" \
  --detector hybrid \
  --yolo-scale "${YOLO_SCALE:-0.33}" \
  --yolo-every "${YOLO_EVERY:-30}" \
  --yolo-timeout "${YOLO_TIMEOUT:-3.0}" \
  --yolo-min-confidence "${YOLO_MIN_CONFIDENCE:-0.4}" \
  --yolo-jpeg-quality "${YOLO_JPEG_QUALITY:-65}" \
  --yolo-command "python3 scripts/yolo_json_worker_cli_adapter.py --parser allwinner-yolo --timeout ${YOLO_WORKER_TIMEOUT:-3.0} --command \"env LD_LIBRARY_PATH=${RUNTIME_LD_PATH} ${YOLO_DEMO} -nb ${YOLO_NB} -i {image}\"" \
  --display-mode "${DISPLAY_MODE:-color}" \
  --stream-port "${STREAM_PORT:-8080}" \
  --stream-scale "${STREAM_SCALE:-0.33}" \
  --stream-every "${STREAM_EVERY:-8}" \
  --stream-quality "${STREAM_QUALITY:-65}" \
  --print-interval "${PRINT_INTERVAL:-1}" \
  --control "${CONTROL:-0}" \
  "$@"
