#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

exec "$ROOT_DIR/scripts/run_a7a_imx415_rect_web.sh" \
  --competition-mode 1 \
  --target-miss-frames "${TARGET_MISS_FRAMES:-3}" \
  --aim-enter-radius-ratio "${AIM_ENTER_RADIUS_RATIO:-0.085}" \
  --aim-exit-radius-ratio "${AIM_EXIT_RADIUS_RATIO:-0.12}" \
  --aim-confirm-frames "${AIM_CONFIRM_FRAMES:-3}" \
  --aim-offset-x-ratio "${AIM_OFFSET_X_RATIO:-0.0}" \
  --aim-offset-y-ratio "${AIM_OFFSET_Y_RATIO:-0.0}" \
  --laser-hold-frames "${LASER_HOLD_FRAMES:-2}" \
  --laser-min-luma "${LASER_MIN_LUMA:-165}" \
  --laser-fallback-min-luma "${LASER_FALLBACK_MIN_LUMA:-210}" \
  "$@"
