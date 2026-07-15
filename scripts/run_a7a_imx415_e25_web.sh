#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

exec "$ROOT_DIR/scripts/run_a7a_imx415_competition_web.sh" \
  --e25-target 1 \
  --a4-target 0 \
  --detector rect \
  --detect-multi-pass 0 \
  --rect-max-area-ratio "${E25_MAX_AREA_RATIO:-0.70}" \
  --a4-global-interval "${E25_GLOBAL_INTERVAL:-15}" \
  --a4-search-interval "${E25_SEARCH_INTERVAL:-6}" \
  --a4-local-validate-interval "${E25_VALIDATE_INTERVAL:-5}" \
  --a4-min-area-ratio "${E25_MIN_AREA_RATIO:-0.015}" \
  --a4-min-apparent-aspect "${E25_MIN_APPARENT_ASPECT:-1.20}" \
  --a4-acquire-confidence "${E25_ACQUIRE_CONFIDENCE:-0.68}" \
  --a4-track-confidence "${E25_TRACK_CONFIDENCE:-0.48}" \
  --a4-occlusion-frames "${E25_OCCLUSION_FRAMES:-12}" \
  --a4-require-red-rings "${E25_REQUIRE_RED_RINGS:-0}" \
  --stream-scale "${E25_STREAM_SCALE:-0.75}" \
  --stream-every "${E25_STREAM_EVERY:-2}" \
  "$@"
