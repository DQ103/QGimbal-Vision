#!/usr/bin/env bash
set -euo pipefail

MODEL_ZOO_DIR="${1:-$HOME/awnpu_model_zoo_v0.9}"
YOLO_DIR="$MODEL_ZOO_DIR/examples/yolov5"
NPU_RT_DIR="$MODEL_ZOO_DIR/common/npuruntime"
NPU_LIB_DIR="$NPU_RT_DIR/lib_linux_aarch64/A733"
BUILD_DIR="$YOLO_DIR/build_native"

if ! command -v pkg-config >/dev/null 2>&1; then
  echo "pkg-config is required. Install with: sudo apt-get install -y pkg-config libopencv-dev" >&2
  exit 1
fi

if ! pkg-config --exists opencv4; then
  echo "OpenCV development files are required. Install with: sudo apt-get install -y libopencv-dev" >&2
  exit 1
fi

if [[ ! -d "$YOLO_DIR" || ! -d "$NPU_RT_DIR" || ! -d "$NPU_LIB_DIR" ]]; then
  echo "Cannot find Allwinner model zoo under: $MODEL_ZOO_DIR" >&2
  echo "Expected examples/yolov5 and common/npuruntime/lib_linux_aarch64/A733." >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

g++ -std=c++11 -O2 \
  -I.. -I"$NPU_RT_DIR" -I"$NPU_RT_DIR/include" \
  ../main.cpp ../yolov5_pre.cpp ../yolov5_post.cpp \
  "$NPU_RT_DIR/npulib.cpp" "$NPU_RT_DIR/npu_util.cpp" \
  $(pkg-config --cflags --libs opencv4) \
  -L"$NPU_LIB_DIR" -lVIPhal -lNBGlinker -lm -ldl \
  -Wl,-rpath,"$NPU_LIB_DIR" \
  -o yolov5_demo_a733

echo "Built: $BUILD_DIR/yolov5_demo_a733"
echo "Runtime:"
echo "  export LD_LIBRARY_PATH=$NPU_LIB_DIR:\$LD_LIBRARY_PATH"
