#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-/dev/video0}"
SUBDEV="${SUBDEV:-/dev/v4l-subdev0}"
WIDTH="${WIDTH:-1920}"
HEIGHT="${HEIGHT:-1080}"
FPS="${FPS:-30}"
FORMAT="${FORMAT:-NV12}"
AWISP="${AWISP:-1}"
LARGEMODE="${LARGEMODE:-0}"
SINK="${SINK:-xvimagesink}"
DISPLAY="${DISPLAY:-:0}"

usage() {
  cat <<'EOF'
Usage: scripts/vnc_imx219_preview.sh [options]

Options:
  --device PATH      V4L2 video device, default: /dev/video0
  --subdev PATH      V4L2 sensor subdev, default: /dev/v4l-subdev0
  --size WxH         Capture size, default: 1920x1080
  --fps N            Capture FPS, default: 30
  --format NAME      GStreamer raw format, default: NV12
                    Try NV12 or RGB for comparison.
  --awisp 0|1       v4l2src en-awisp, default: 1
  --largemode 0|1   v4l2src en-largemode, default: 0
  --sink NAME        GStreamer video sink, default: ximagesink
                    Try xvimagesink, autovideosink, or glimagesink if needed.
  -h, --help         Show this help.

Environment overrides are also supported: DEVICE, SUBDEV, WIDTH, HEIGHT, FPS,
FORMAT, AWISP, LARGEMODE, SINK, DISPLAY.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --subdev)
      SUBDEV="$2"
      shift 2
      ;;
    --size)
      IFS=x read -r WIDTH HEIGHT <<<"$2"
      shift 2
      ;;
    --fps)
      FPS="$2"
      shift 2
      ;;
    --format)
      FORMAT="$2"
      shift 2
      ;;
    --awisp)
      AWISP="$2"
      shift 2
      ;;
    --largemode)
      LARGEMODE="$2"
      shift 2
      ;;
    --sink)
      SINK="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing command: $1" >&2
    exit 127
  fi
}

need v4l2-ctl
need gst-launch-1.0

if [[ ! -e "$DEVICE" ]]; then
  echo "Video device not found: $DEVICE" >&2
  exit 1
fi

if [[ ! -e "$SUBDEV" ]]; then
  echo "Sensor subdev not found: $SUBDEV" >&2
  exit 1
fi

export DISPLAY

echo "DISPLAY=$DISPLAY"
echo "Setting $SUBDEV to ${WIDTH}x${HEIGHT} RAW10..."
v4l2-ctl -d "$SUBDEV" --set-subdev-fmt "pad=0,code=0x300f,width=${WIDTH},height=${HEIGHT}" >/dev/null
v4l2-ctl -d "$SUBDEV" --get-subdev-fmt pad=0

echo
echo "Starting preview from $DEVICE at ${WIDTH}x${HEIGHT}@${FPS}, format=$FORMAT, awisp=$AWISP, largemode=$LARGEMODE using $SINK"
echo "Close the video window or press Ctrl+C in this terminal to stop."

exec gst-launch-1.0 -v \
  v4l2src device="$DEVICE" en-awisp="$AWISP" en-largemode="$LARGEMODE" \
  ! "video/x-raw,format=${FORMAT},width=${WIDTH},height=${HEIGHT},framerate=${FPS}/1" \
  ! videoconvert \
  ! "$SINK" sync=false
