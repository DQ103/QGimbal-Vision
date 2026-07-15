#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

exec "$ROOT_DIR/scripts/run_a7a_imx415_competition_web.sh" \
  --a4-target 0 \
  --detect-multi-pass 1 \
  --rect-min-area-ratio "${RECT_MIN_AREA_RATIO:-0.020833}" \
  --rect-max-area-ratio "${RECT_MAX_AREA_RATIO:-0.85}" \
  --rect-max-aspect "${RECT_MAX_ASPECT:-5.0}" \
  --target-miss-frames "${TARGET_MISS_FRAMES:-3}" \
  --aim-enter-radius-ratio "${AIM_ENTER_RADIUS_RATIO:-0.018}" \
  --aim-exit-radius-ratio "${AIM_EXIT_RADIUS_RATIO:-0.028}" \
  --aim-confirm-frames "${AIM_CONFIRM_FRAMES:-1}" \
  --laser-hold-frames "${LASER_HOLD_FRAMES:-3}" \
  --laser-min-luma "${LASER_MIN_LUMA:-150}" \
  --laser-fallback-min-luma "${LASER_FALLBACK_MIN_LUMA:-255}" \
  --laser-require-violet 1 \
  --laser-strict-violet 1 \
  --laser-detect-after-ready-only 0 \
  --stream-every "${STREAM_EVERY:-1}" \
  "$@"
