#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

exec "$ROOT_DIR/scripts/run_a7a_imx415_competition_web.sh" \
  --a4-target 1 \
  --detect-multi-pass 0 \
  --a4-global-interval "${A4_GLOBAL_INTERVAL:-15}" \
  --a4-search-interval "${A4_SEARCH_INTERVAL:-12}" \
  --a4-local-validate-interval "${A4_LOCAL_VALIDATE_INTERVAL:-5}" \
  --a4-min-area-ratio "${A4_MIN_AREA_RATIO:-0.015}" \
  --a4-min-apparent-aspect "${A4_MIN_APPARENT_ASPECT:-1.20}" \
  --a4-acquire-confidence "${A4_ACQUIRE_CONFIDENCE:-0.72}" \
  --a4-track-confidence "${A4_TRACK_CONFIDENCE:-0.52}" \
  --a4-occlusion-frames "${A4_OCCLUSION_FRAMES:-20}" \
  --a4-require-red-rings "${A4_REQUIRE_RED_RINGS:-0}" \
  --stream-every "${A4_STREAM_EVERY:-1}" \
  "$@"
