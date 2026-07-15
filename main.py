# 摄像头读取并显示画面
# 使用: python main.py --camera 0

"""
简单的摄像头预览脚本（使用 OpenCV）。
参数：
  --camera        摄像头索引（默认 0）
  --display       是否显示图形化窗口（0/1，默认 1）
                 - 1：显示图像（现有效果），并叠加矩形与 FPS
                 - 0：不显示窗口，在终端输出 FPS + 检测到的矩形中心点坐标/面积
  --print-interval 终端输出间隔秒数（仅 --display 0 时生效，默认 0.5）

GUI 模式按 'q' 或 ESC 退出；无窗口模式请按 Ctrl+C 退出。
"""

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import subprocess
import sys
import threading
import time
from typing import Optional
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from vision.a4_target import (
    a4_center_mm,
    A4TargetConfig,
    A4TargetTracker,
    detection_as_rect,
    find_black_band_candidates,
    map_image_point_to_a4,
    merge_candidates,
)
from vision.competition_vision import (
    AimReadyGate,
    HybridLaserConfig,
    HybridLaserTracker,
    TrackedTarget,
    TargetTracker,
    VisionStage,
    resolve_stage,
    stage_error,
)
from vision.e25_laser import E25LaserTracker
from vision.e25_pipeline import E25PipelineConfig, E25VisionPipeline
from vision.rect_detect import (
    DetectedRect,
    RectSelectionConfig,
    RectSelector,
    detect_rectangles,
    detect_rectangles_multi_pass,
    draw_detected_rect,
)
from vision.yolo_detect import AsyncYoloDetector

from control.config import ControlConfig
from control.serial_stub import GimbalSerialStub
from control.tracker_control import GimbalTracker

DEFAULT_CAMERA = 0  # 摄像头索引（legacy V4L2 backend）
DEFAULT_DEVICE = "/dev/video0"
DEFAULT_SUBDEV = "/dev/v4l-subdev0"
DEFAULT_SET_SUBDEV_FORMAT = 1
DEFAULT_WIDTH = 1920  # 期望宽度
DEFAULT_HEIGHT = 1080  # 期望高度
DEFAULT_FPS = 30  # 期望帧率
DEFAULT_FORMAT = "NV12"
DEFAULT_CAPTURE_MODE = "raw"
DEFAULT_AWISP = 0
DEFAULT_LARGEMODE = 0
DEFAULT_DETECT_SCALE = 0.25
DEFAULT_DETECT_MULTI_PASS = 1
DEFAULT_RECT_CENTER_WEIGHT = 0.25
DEFAULT_RECT_PREV_WEIGHT = 0.70
DEFAULT_RECT_MAX_ASPECT = 5.0
DEFAULT_RECT_MAX_AREA_RATIO = 0.5
DEFAULT_COMPETITION_MODE = 0
DEFAULT_TARGET_MISS_FRAMES = 3
DEFAULT_AIM_ENTER_RADIUS_RATIO = 0.085
DEFAULT_AIM_EXIT_RADIUS_RATIO = 0.12
DEFAULT_AIM_CONFIRM_FRAMES = 3
DEFAULT_LASER_HOLD_FRAMES = 2
DEFAULT_LASER_MIN_LUMA = 165
DEFAULT_LASER_FALLBACK_MIN_LUMA = 210
DEFAULT_A4_TARGET = 0
DEFAULT_A4_GLOBAL_INTERVAL = 10
DEFAULT_A4_SEARCH_INTERVAL = 6
DEFAULT_A4_LOCAL_VALIDATE_INTERVAL = 3
DEFAULT_A4_MIN_AREA_RATIO = 0.015
DEFAULT_A4_MIN_APPARENT_ASPECT = 1.08
DEFAULT_A4_ACQUIRE_CONFIDENCE = 0.72
DEFAULT_A4_TRACK_CONFIDENCE = 0.52
DEFAULT_A4_OCCLUSION_FRAMES = 20
DEFAULT_A4_REQUIRE_RED_RINGS = 1
DEFAULT_E25_TARGET = 0
DEFAULT_DETECTOR = "rect"
DEFAULT_YOLO_SCALE = 0.33
DEFAULT_YOLO_EVERY = 1
DEFAULT_YOLO_TIMEOUT = 0.05
DEFAULT_YOLO_MIN_CONFIDENCE = 0.25
DEFAULT_YOLO_JPEG_QUALITY = 70
DEFAULT_FLIP = 0
DEFAULT_DISPLAY_MODE = "gray"
DEFAULT_DISPLAY_SCALE = 0.5
DEFAULT_DISPLAY_EVERY = 1
DEFAULT_DISPLAY = 1
DEFAULT_PRINT_INTERVAL = 0.05
DEFAULT_STREAM_HOST = "0.0.0.0"
DEFAULT_STREAM_PORT = 0
DEFAULT_STREAM_SCALE = 0.5
DEFAULT_STREAM_EVERY = 2
DEFAULT_STREAM_QUALITY = 70
COLOR_DEFAULTS = {
    "brightness": 0.0,
    "contrast": 1.0,
    "saturation": 1.0,
    "hue": 0.0,
    "gamma": 1.0,
    "red": 1.0,
    "green": 1.0,
    "blue": 1.0,
    "auto_wb": 0.0,
    "auto_level": 0.0,
    "level_low": 0.5,
    "level_high": 99.0,
    "vibrance": 0.0,
    "clarity": 0.0,
    "sharpness": 0.0,
}
COLOR_RANGES = {
    "brightness": (-100.0, 100.0, 1.0),
    "contrast": (0.2, 3.0, 0.01),
    "saturation": (0.0, 3.0, 0.01),
    "hue": (-90.0, 90.0, 1.0),
    "gamma": (0.2, 3.0, 0.01),
    "red": (0.2, 3.0, 0.01),
    "green": (0.2, 3.0, 0.01),
    "blue": (0.2, 3.0, 0.01),
    "auto_wb": (0.0, 1.0, 0.01),
    "auto_level": (0.0, 1.0, 0.01),
    "level_low": (0.0, 10.0, 0.1),
    "level_high": (90.0, 100.0, 0.1),
    "vibrance": (-1.0, 2.0, 0.01),
    "clarity": (0.0, 1.0, 0.01),
    "sharpness": (0.0, 1.5, 0.01),
}
COLOR_PRESETS = {
    "neutral": COLOR_DEFAULTS.copy(),
    "a7a_realtime": {
        **COLOR_DEFAULTS,
        "brightness": 28.0,
        "contrast": 0.92,
        "saturation": 1.04,
        "gamma": 1.32,
        "red": 1.18,
        "green": 0.96,
        "blue": 0.90,
        "vibrance": 0.03,
    },
    "a7a_soft": {
        **COLOR_DEFAULTS,
        "brightness": 2.0,
        "contrast": 0.92,
        "saturation": 1.05,
        "gamma": 1.12,
        "auto_wb": 0.85,
        "auto_level": 0.55,
        "vibrance": 0.16,
        "clarity": 0.10,
        "sharpness": 0.10,
    },
    "a7a_balanced": {
        **COLOR_DEFAULTS,
        "brightness": 3.0,
        "contrast": 0.96,
        "saturation": 1.10,
        "gamma": 1.16,
        "red": 1.02,
        "green": 0.99,
        "blue": 0.98,
        "auto_wb": 0.85,
        "auto_level": 0.68,
        "vibrance": 0.24,
        "clarity": 0.16,
        "sharpness": 0.12,
    },
    "a7a_strong": {
        **COLOR_DEFAULTS,
        "brightness": 4.0,
        "contrast": 1.0,
        "saturation": 1.12,
        "gamma": 1.20,
        "red": 1.02,
        "blue": 0.98,
        "auto_wb": 0.75,
        "auto_level": 0.78,
        "level_low": 0.8,
        "level_high": 98.8,
        "vibrance": 0.30,
        "clarity": 0.20,
        "sharpness": 0.12,
    },
}

# 控制默认参数（可通过命令行覆盖）
DEFAULT_CONTROL_ENABLED = 1
DEFAULT_MAX_RPM = 20.0
DEFAULT_DEADBAND_PX = 0.0
DEFAULT_LOST_TIMEOUT_S = 0.4


def parse_args():
    p = argparse.ArgumentParser(description="OpenCV 摄像头显示示例")
    p.add_argument('--camera', type=int, default=DEFAULT_CAMERA, help=f'摄像头索引（默认 {DEFAULT_CAMERA}）')
    p.add_argument('--backend', choices=['gstreamer', 'v4l2'], default='gstreamer',
                   help='摄像头后端：gstreamer 会启用 AWISP，v4l2 为旧路径')
    p.add_argument('--device', type=str, default=DEFAULT_DEVICE,
                   help=f'V4L2 设备路径（默认 {DEFAULT_DEVICE}）')
    p.add_argument('--subdev', type=str, default=DEFAULT_SUBDEV,
                   help=f'传感器 subdev 路径（默认 {DEFAULT_SUBDEV}）')
    p.add_argument('--set-subdev-format', type=int, choices=[0, 1], default=DEFAULT_SET_SUBDEV_FORMAT,
                   help='启动前是否通过 v4l2-ctl 设置 subdev RAW10 格式；官方 IMX415 管线应设为 0')
    p.add_argument('--size', type=str, default=f'{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}',
                   help=f'采集尺寸 WxH（默认 {DEFAULT_WIDTH}x{DEFAULT_HEIGHT}）')
    p.add_argument('--output-size', type=str, default='',
                   help='BGR 模式下 GStreamer 输出尺寸 WxH；留空表示与采集尺寸相同')
    p.add_argument('--fps', type=int, default=DEFAULT_FPS,
                   help=f'采集帧率（默认 {DEFAULT_FPS}）')
    p.add_argument('--max-processing-fps', type=float, default=0.0,
                   help='限制检测和推流循环帧率；0 表示不限制')
    p.add_argument('--format', type=str, default=DEFAULT_FORMAT,
                   help=f'GStreamer v4l2src 输出格式（默认 {DEFAULT_FORMAT}，可试 NV12/RGB）')
    p.add_argument('--capture-mode', choices=['raw', 'bgr'], default=DEFAULT_CAPTURE_MODE,
                   help='raw 直接把 NV12 送入 OpenCV 并用 Y 平面检测；bgr 使用 videoconvert 转 BGR')
    p.add_argument('--awisp', type=int, choices=[0, 1], default=DEFAULT_AWISP,
                   help=f'是否启用 AWISP（0/1，默认 {DEFAULT_AWISP}；1080p 稳定 30 FPS 建议 0）')
    p.add_argument('--largemode', type=int, choices=[0, 1], default=DEFAULT_LARGEMODE,
                   help=f'v4l2src en-largemode 参数（0/1，默认 {DEFAULT_LARGEMODE}）')
    p.add_argument('--v4l2-ctrl', type=str, default='',
                   help='启动采集前写入 V4L2 控制项，例如 auto_exposure_bias=6,wide_dynamic_range=1')
    p.add_argument('--detect-scale', type=float, default=DEFAULT_DETECT_SCALE,
                   help=f'检测缩放比例，0-1；小于 1 可提高 FPS（默认 {DEFAULT_DETECT_SCALE}）')
    p.add_argument('--detect-multi-pass', type=int, choices=[0, 1], default=DEFAULT_DETECT_MULTI_PASS,
                   help=f'是否启用多轮矩形检测（0/1，默认 {DEFAULT_DETECT_MULTI_PASS}）')
    p.add_argument('--rect-center-weight', type=float, default=DEFAULT_RECT_CENTER_WEIGHT,
                   help=f'目标评分中心权重（默认 {DEFAULT_RECT_CENTER_WEIGHT}）')
    p.add_argument('--rect-prev-weight', type=float, default=DEFAULT_RECT_PREV_WEIGHT,
                   help=f'目标评分历史连续性权重（默认 {DEFAULT_RECT_PREV_WEIGHT}）')
    p.add_argument('--rect-max-aspect', type=float, default=DEFAULT_RECT_MAX_ASPECT,
                   help=f'候选矩形最大长宽比（默认 {DEFAULT_RECT_MAX_ASPECT}）')
    p.add_argument('--rect-max-area-ratio', type=float, default=DEFAULT_RECT_MAX_AREA_RATIO,
                   help=f'候选矩形最大面积占比（默认 {DEFAULT_RECT_MAX_AREA_RATIO}）')
    p.add_argument('--competition-mode', type=int, choices=[0, 1], default=DEFAULT_COMPETITION_MODE,
                   help='启用矩形对中、激光门控与融合激光跟踪实验（0/1）')
    p.add_argument('--target-miss-frames', type=int, default=DEFAULT_TARGET_MISS_FRAMES,
                   help=f'连续多少帧丢失后清除矩形跟踪（默认 {DEFAULT_TARGET_MISS_FRAMES}）')
    p.add_argument('--aim-enter-radius-ratio', type=float, default=DEFAULT_AIM_ENTER_RADIUS_RATIO,
                   help='目标进入就绪状态的半径，占画面短边比例')
    p.add_argument('--aim-exit-radius-ratio', type=float, default=DEFAULT_AIM_EXIT_RADIUS_RATIO,
                   help='目标退出就绪状态的迟滞半径，占画面短边比例')
    p.add_argument('--aim-confirm-frames', type=int, default=DEFAULT_AIM_CONFIRM_FRAMES,
                   help=f'目标连续对中确认帧数（默认 {DEFAULT_AIM_CONFIRM_FRAMES}）')
    p.add_argument('--aim-offset-x-ratio', type=float, default=0.0,
                   help='标定中心相对画面中心的水平偏移，占画面宽度比例')
    p.add_argument('--aim-offset-y-ratio', type=float, default=0.0,
                   help='标定中心相对画面中心的垂直偏移，占画面高度比例')
    p.add_argument('--laser-hold-frames', type=int, default=DEFAULT_LASER_HOLD_FRAMES,
                   help=f'激光漏检后仅用于显示和关联的保持帧数（默认 {DEFAULT_LASER_HOLD_FRAMES}）')
    p.add_argument('--laser-min-luma', type=int, default=DEFAULT_LASER_MIN_LUMA,
                   help=f'动态高亮检测最低灰度（默认 {DEFAULT_LASER_MIN_LUMA}）')
    p.add_argument('--laser-fallback-min-luma', type=int, default=DEFAULT_LASER_FALLBACK_MIN_LUMA,
                   help=f'没有紫色光晕时允许亮点候选的最低灰度（默认 {DEFAULT_LASER_FALLBACK_MIN_LUMA}）')
    p.add_argument('--a4-target', type=int, choices=[0, 1], default=DEFAULT_A4_TARGET,
                   help='启用无NPU A4靶纸结构验证和空间时域跟踪（0/1）')
    p.add_argument('--a4-global-interval', type=int, default=DEFAULT_A4_GLOBAL_INTERVAL,
                   help=f'稳定跟踪时每多少帧执行一次全局重检（默认 {DEFAULT_A4_GLOBAL_INTERVAL}）')
    p.add_argument('--a4-search-interval', type=int, default=DEFAULT_A4_SEARCH_INTERVAL,
                   help=f'未锁定时每多少帧执行一次全局搜索（默认 {DEFAULT_A4_SEARCH_INTERVAL}）')
    p.add_argument('--a4-local-validate-interval', type=int, default=DEFAULT_A4_LOCAL_VALIDATE_INTERVAL,
                   help=f'光流跟踪时每多少帧执行一次完整A4结构复核（默认 {DEFAULT_A4_LOCAL_VALIDATE_INTERVAL}）')
    p.add_argument('--a4-min-area-ratio', type=float, default=DEFAULT_A4_MIN_AREA_RATIO,
                   help=f'A4候选最小画面面积占比（默认 {DEFAULT_A4_MIN_AREA_RATIO}）')
    p.add_argument('--a4-min-apparent-aspect', type=float, default=DEFAULT_A4_MIN_APPARENT_ASPECT,
                   help=f'A4候选最小表观长宽比（默认 {DEFAULT_A4_MIN_APPARENT_ASPECT}）')
    p.add_argument('--a4-acquire-confidence', type=float, default=DEFAULT_A4_ACQUIRE_CONFIDENCE,
                   help=f'A4首次锁定置信度门限（默认 {DEFAULT_A4_ACQUIRE_CONFIDENCE}）')
    p.add_argument('--a4-track-confidence', type=float, default=DEFAULT_A4_TRACK_CONFIDENCE,
                   help=f'A4正常跟踪置信度门限（默认 {DEFAULT_A4_TRACK_CONFIDENCE}）')
    p.add_argument('--a4-occlusion-frames', type=int, default=DEFAULT_A4_OCCLUSION_FRAMES,
                   help=f'A4部分遮挡最大保持帧数（默认 {DEFAULT_A4_OCCLUSION_FRAMES}）')
    p.add_argument('--a4-require-red-rings', type=int, choices=[0, 1], default=DEFAULT_A4_REQUIRE_RED_RINGS,
                   help='A4首次锁定是否要求红色圆环结构（0/1）')
    p.add_argument('--e25-target', type=int, choices=[0, 1], default=DEFAULT_E25_TARGET,
                   help='启用E25模型化A4四边测量、动静预测和毫米坐标链路（0/1）')
    p.add_argument('--detector', choices=['rect', 'yolo', 'hybrid'], default=DEFAULT_DETECTOR,
                   help=f'检测器：rect 传统CV，yolo 外部NPU/YOLO，hybrid YOLO优先传统CV兜底（默认 {DEFAULT_DETECTOR}）')
    p.add_argument('--yolo-command', type=str, default='',
                   help='外部 YOLO/NPU worker 命令；--detector yolo/hybrid 时必填')
    p.add_argument('--yolo-scale', type=float, default=DEFAULT_YOLO_SCALE,
                   help=f'送入 YOLO worker 的缩放比例（默认 {DEFAULT_YOLO_SCALE}）')
    p.add_argument('--yolo-every', type=int, default=DEFAULT_YOLO_EVERY,
                   help=f'每 N 帧调用一次 YOLO worker（默认 {DEFAULT_YOLO_EVERY}）')
    p.add_argument('--yolo-timeout', type=float, default=DEFAULT_YOLO_TIMEOUT,
                   help=f'等待 YOLO worker 输出的超时时间秒（默认 {DEFAULT_YOLO_TIMEOUT}）')
    p.add_argument('--yolo-min-confidence', type=float, default=DEFAULT_YOLO_MIN_CONFIDENCE,
                   help=f'YOLO 检测框最低置信度（默认 {DEFAULT_YOLO_MIN_CONFIDENCE}）')
    p.add_argument('--yolo-jpeg-quality', type=int, default=DEFAULT_YOLO_JPEG_QUALITY,
                   help=f'传给 YOLO worker 的 JPEG 质量（默认 {DEFAULT_YOLO_JPEG_QUALITY}）')
    p.add_argument('--yolo-label', action='append', default=[],
                   help='只接受指定 label，可重复传多次；不传则接受所有类别')
    p.add_argument('--flip', type=int, choices=[0, 1], default=DEFAULT_FLIP,
                   help=f'是否把图像旋转 180 度（0/1，默认 {DEFAULT_FLIP}；开启会降低 FPS）')
    p.add_argument('--display-mode', choices=['gray', 'color'], default=DEFAULT_DISPLAY_MODE,
                   help=f'显示模式：gray 直接显示 Y 平面，color 转 BGR 显示彩色画面（默认 {DEFAULT_DISPLAY_MODE}）')
    p.add_argument('--display-scale', type=float, default=DEFAULT_DISPLAY_SCALE,
                   help=f'显示窗口缩放比例，0-1；只影响屏幕显示，不影响检测坐标（默认 {DEFAULT_DISPLAY_SCALE}）')
    p.add_argument('--display-every', type=int, default=DEFAULT_DISPLAY_EVERY,
                   help=f'每 N 帧刷新一次屏幕显示；只影响屏幕显示，不影响检测/控制（默认 {DEFAULT_DISPLAY_EVERY}）')
    p.add_argument('--display', type=int, choices=[0, 1], default=DEFAULT_DISPLAY,
                   help=f'是否显示图形化窗口（0/1，默认 {DEFAULT_DISPLAY}）')
    p.add_argument('--print-interval', type=float, default=DEFAULT_PRINT_INTERVAL,
                   help=f'终端输出间隔秒数（仅 --display 0 生效，默认 {DEFAULT_PRINT_INTERVAL}）')
    p.add_argument('--print-in-display', type=int, choices=[0, 1], default=0,
                   help='显示窗口时是否也在终端打印 FPS（0/1，默认 0）')
    p.add_argument('--stream-host', type=str, default=DEFAULT_STREAM_HOST,
                   help=f'MJPEG 预览监听地址（默认 {DEFAULT_STREAM_HOST}）')
    p.add_argument('--stream-port', type=int, default=DEFAULT_STREAM_PORT,
                   help=f'MJPEG 预览端口；0 表示关闭（默认 {DEFAULT_STREAM_PORT}）')
    p.add_argument('--stream-scale', type=float, default=DEFAULT_STREAM_SCALE,
                   help=f'MJPEG 预览缩放比例，0-1（默认 {DEFAULT_STREAM_SCALE}）')
    p.add_argument('--stream-every', type=int, default=DEFAULT_STREAM_EVERY,
                   help=f'每 N 帧推送一次 MJPEG 预览（默认 {DEFAULT_STREAM_EVERY}）')
    p.add_argument('--stream-quality', type=int, default=DEFAULT_STREAM_QUALITY,
                   help=f'MJPEG JPEG 质量，1-100（默认 {DEFAULT_STREAM_QUALITY}）')

    # 控制相关
    p.add_argument('--control', type=int, choices=[0, 1], default=DEFAULT_CONTROL_ENABLED,
                   help=f'是否启用 PID 控制输出（0/1，默认 {DEFAULT_CONTROL_ENABLED}）')
    p.add_argument('--max-rpm', type=float, default=DEFAULT_MAX_RPM,
                   help=f'最大转速输出（RPM，默认 {DEFAULT_MAX_RPM}）')
    p.add_argument('--deadband-px', type=float, default=DEFAULT_DEADBAND_PX,
                   help=f'像素死区（默认 {DEFAULT_DEADBAND_PX}）')
    p.add_argument('--lost-timeout', type=float, default=DEFAULT_LOST_TIMEOUT_S,
                   help=f'丢目标超时后复位控制器的时间（秒，默认 {DEFAULT_LOST_TIMEOUT_S}）')

    # 串口相关（协议在 control/serial_stub.py 内实现）
    p.add_argument('--serial-port', type=str, default=None, help='串口端口号，例如 COM3；不填则不发送')
    p.add_argument('--serial-baud', type=int, default=1152000, help='串口波特率（默认 1152000）')

    return p.parse_args()


def parse_size(size_text: str) -> tuple[int, int]:
    try:
        width_text, height_text = size_text.lower().split('x', 1)
        width = int(width_text)
        height = int(height_text)
    except ValueError as exc:
        raise SystemExit(f'无效尺寸: {size_text}, 应为 WxH，例如 1920x1080') from exc
    if width <= 0 or height <= 0:
        raise SystemExit(f'无效尺寸: {size_text}')
    return width, height


def set_sensor_format(subdev: str, width: int, height: int) -> None:
    cmd = [
        'v4l2-ctl',
        '-d',
        subdev,
        '--set-subdev-fmt',
        f'pad=0,code=0x300f,width={width},height={height}',
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    except FileNotFoundError as exc:
        raise SystemExit('缺少 v4l2-ctl，请先安装 v4l-utils') from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f'设置 {subdev} 为 {width}x{height} RAW10 失败: {exc}') from exc


def parse_v4l2_controls(ctrl_text: str) -> list[tuple[str, str]]:
    controls = []
    for item in ctrl_text.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise SystemExit(f'无效 V4L2 控制项: {item}, 应为 key=value')
        key, value = item.split('=', 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise SystemExit(f'无效 V4L2 控制项: {item}, 应为 key=value')
        controls.append((key, value))
    return controls


def apply_v4l2_controls(device: str, ctrl_text: str) -> None:
    controls = parse_v4l2_controls(ctrl_text)
    for key, value in controls:
        cmd = ['v4l2-ctl', '-d', device, '--set-ctrl', f'{key}={value}']
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        except FileNotFoundError as exc:
            raise SystemExit('缺少 v4l2-ctl，请先安装 v4l-utils') from exc
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f'设置 V4L2 控制项 {key}={value} 失败: {exc}') from exc


def build_gst_pipeline(
    device: str,
    width: int,
    height: int,
    output_width: int,
    output_height: int,
    fps: int,
    raw_format: str,
    capture_mode: str,
    awisp: int,
    largemode: int,
) -> str:
    src = (
        f'v4l2src device={device} en-awisp={int(awisp)} en-largemode={int(largemode)} ! '
        f'video/x-raw,format={raw_format},width={width},height={height},framerate={fps}/1 ! '
    )
    if capture_mode == 'raw':
        return src + 'queue max-size-buffers=1 leaky=downstream ! appsink drop=true max-buffers=1 sync=false'

    if (output_width, output_height) != (width, height):
        src += (
            'videoscale ! '
            f'video/x-raw,format={raw_format},width={output_width},height={output_height} ! '
        )
    return (
        src
        + 'videoconvert ! video/x-raw,format=BGR ! '
        + 'queue max-size-buffers=1 leaky=downstream ! appsink drop=true max-buffers=1 sync=false'
    )


def open_capture(args, width: int, height: int, output_width: int, output_height: int):
    if args.backend == 'gstreamer':
        if args.capture_mode == 'raw' and args.format.upper() != 'NV12':
            raise SystemExit('--capture-mode raw 目前只支持 --format NV12')
        if args.set_subdev_format:
            set_sensor_format(args.subdev, width, height)
        if args.v4l2_ctrl:
            apply_v4l2_controls(args.device, args.v4l2_ctrl)
        pipeline = build_gst_pipeline(
            args.device,
            width,
            height,
            output_width,
            output_height,
            args.fps,
            args.format,
            args.capture_mode,
            args.awisp,
            args.largemode,
        )
        print(f'Using GStreamer pipeline:\n{pipeline}')
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print('无法打开 GStreamer 摄像头管线，请检查设备是否被其它程序占用。')
            sys.exit(2)
        return cap

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"无法打开摄像头索引 {args.camera}. 请检查设备或更换索引。")
        sys.exit(2)

    if args.v4l2_ctrl:
        apply_v4l2_controls(args.device, args.v4l2_ctrl)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    return cap


def extract_frame(frame, width: int, height: int, args):
    flip = bool(args.flip)
    if args.backend == 'gstreamer' and args.capture_mode == 'raw':
        nv12_height = height + height // 2
        if frame.ndim != 2 or frame.shape[0] < nv12_height or frame.shape[1] < width:
            raise RuntimeError(f'期望 NV12 frame shape >= ({nv12_height}, {width})，实际为 {frame.shape}')

        gray = frame[:height, :width]
        if flip:
            gray = cv2.flip(gray, -1)
        return gray

    bgr = cv2.flip(frame, -1) if flip else frame
    return bgr


def scaled_size(width: int, height: int, scale: float) -> tuple[int, int]:
    if scale == 1.0:
        return width, height
    scaled_width = max(2, int(round(width * scale)))
    scaled_height = max(2, int(round(height * scale)))
    if scaled_width % 2:
        scaled_width -= 1
    if scaled_height % 2:
        scaled_height -= 1
    return scaled_width, scaled_height


def scale_detected_rect(rect, scale_x: float, scale_y: float):
    if rect is None:
        return None
    return DetectedRect(
        center=(rect.center[0] * scale_x, rect.center[1] * scale_y),
        box=rect.box * np.array([scale_x, scale_y], dtype=np.float32),
        area=rect.area * scale_x * scale_y,
        pass_index=rect.pass_index,
        score=rect.score,
    )


def scale_detected_rects(rects, scale_x: float, scale_y: float):
    return [scale_detected_rect(rect, scale_x, scale_y) for rect in rects]


def _scale_xywh(box, scale_x: float, scale_y: float):
    if box is None:
        return None
    x, y, w, h = box
    return (
        int(round(x * scale_x)),
        int(round(y * scale_y)),
        max(1, int(round(w * scale_x))),
        max(1, int(round(h * scale_y))),
    )


def make_competition_overlay(
    target,
    laser,
    aim_status,
    stage,
    error,
    frame_w: int,
    frame_h: int,
    aim_radius_ratio: float,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    a4_result=None,
    a4_require_red_rings=None,
):
    if aim_status is None or stage is None:
        return None
    overlay = {
        "target_current": bool(target is not None and target.current),
        "target_held": bool(target is not None and target.held),
        "laser_current": bool(laser is not None and laser.current),
        "laser_held": bool(laser is not None and laser.held),
        "laser_center": None if laser is None else (
            laser.center[0] * scale_x,
            laser.center[1] * scale_y,
        ),
        "laser_bbox": None if laser is None else _scale_xywh(laser.bbox, scale_x, scale_y),
        "laser_roi": None if laser is None else _scale_xywh(laser.roi, scale_x, scale_y),
        "laser_score": 0.0 if laser is None else laser.score,
        "laser_luma": 0 if laser is None else laser.max_luma,
        "laser_violet": 0 if laser is None else laser.violet_pixels,
        "aim_center": (
            aim_status.center[0] * scale_x,
            aim_status.center[1] * scale_y,
        ),
        "aim_radius": aim_radius_ratio * min(frame_w * scale_x, frame_h * scale_y),
        "ready": aim_status.ready,
        "confirm_count": aim_status.confirm_count,
        "stage": stage.value,
        "error": (error[0] * scale_x, error[1] * scale_y),
    }
    if a4_result is None:
        return overlay

    detection = a4_result.detection
    is_e25 = hasattr(a4_result, "confidence")
    overlay["target_model"] = "e25" if is_e25 else "a4"
    overlay["a4_state"] = a4_result.state.value
    overlay["a4_require_red_rings"] = bool(a4_require_red_rings)
    overlay["a4_current"] = a4_result.current
    overlay["a4_predicted"] = a4_result.predicted
    overlay["a4_miss_count"] = a4_result.miss_count
    overlay["a4_flow_inliers"] = a4_result.flow_inliers
    overlay["a4_confidence"] = 0.0 if detection is None else detection.confidence
    overlay["a4_structural"] = 0.0 if detection is None else detection.structural_confidence
    overlay["a4_edge"] = 0.0 if detection is None else detection.scores.edge
    overlay["a4_black"] = 0.0 if detection is None else detection.scores.black_band
    overlay["a4_red"] = 0.0 if detection is None else detection.scores.red_rings
    overlay["a4_temporal"] = 0.0 if detection is None else detection.scores.temporal
    overlay["a4_visible_sides"] = 0 if detection is None else detection.scores.visible_sides
    overlay["a4_edge_points"] = () if detection is None else tuple(
        (x * scale_x, y * scale_y, score) for x, y, score in detection.edge_points
    )
    overlay["a4_canonical"] = None if detection is None else detection.canonical
    overlay["a4_error_mm"] = None
    overlay["e25_identity"] = 0.0
    overlay["e25_measurement"] = 0.0
    overlay["e25_tracking"] = 0.0
    overlay["e25_control_valid"] = False
    if is_e25:
        overlay["e25_identity"] = a4_result.confidence.identity
        overlay["e25_measurement"] = a4_result.confidence.measurement
        overlay["e25_tracking"] = a4_result.confidence.tracking
        overlay["e25_control_valid"] = a4_result.confidence.control_valid
    if detection is not None and laser is not None and laser.current:
        laser_mm = map_image_point_to_a4(detection, laser.center)
        target_mm = a4_center_mm(detection)
        overlay["a4_error_mm"] = (
            target_mm[0] - laser_mm[0],
            target_mm[1] - laser_mm[1],
        )
    return overlay


def make_display_frame(raw_frame, detect_frame, width: int, height: int, args, output_scale: float = 1.0):
    output_width, output_height = scaled_size(width, height, output_scale)
    if args.backend == 'gstreamer' and args.capture_mode == 'raw':
        if args.display_mode == 'gray':
            if output_scale == 1.0:
                return detect_frame
            return cv2.resize(detect_frame, (output_width, output_height), interpolation=cv2.INTER_AREA)

        nv12_height = height + height // 2
        if output_scale == 1.0:
            nv12 = raw_frame[:nv12_height, :width]
        else:
            y_plane = raw_frame[:height, :width]
            uv_plane = raw_frame[height:nv12_height, :width]
            y_small = cv2.resize(y_plane, (output_width, output_height), interpolation=cv2.INTER_AREA)
            uv_small = cv2.resize(uv_plane, (output_width, output_height // 2), interpolation=cv2.INTER_AREA)
            nv12 = np.vstack((y_small, uv_small))
        display_frame = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
        if args.flip:
            display_frame = cv2.flip(display_frame, -1)
        return display_frame

    if args.display_mode == 'gray' and detect_frame.ndim == 3:
        display_frame = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
    else:
        display_frame = detect_frame
    if output_scale == 1.0:
        return display_frame
    return cv2.resize(display_frame, (output_width, output_height), interpolation=cv2.INTER_AREA)


def scale_display_frame(display_frame, display_scale: float):
    if display_scale == 1.0:
        return display_frame
    return cv2.resize(display_frame, (0, 0), fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)


def draw_overlay(
    display_frame,
    best,
    ctrl_out,
    fps: float,
    frame_w: int,
    frame_h: int,
    competition=None,
):
    is_gray_display = display_frame.ndim == 2
    rect_color = 255 if is_gray_display else (0, 255, 0)
    marker_color = 255 if is_gray_display else (255, 0, 0)
    text_color = 255 if is_gray_display else (0, 255, 0)
    control_text_color = 200 if is_gray_display else (0, 255, 255)

    if best is not None:
        if competition is not None and competition["target_held"]:
            target_color = 220 if is_gray_display else (0, 255, 255)
        else:
            target_color = rect_color
        draw_detected_rect(display_frame, best, color=target_color)

    cv2.drawMarker(
        display_frame,
        (frame_w // 2, frame_h // 2),
        marker_color,
        markerType=cv2.MARKER_CROSS,
        markerSize=18,
        thickness=2,
    )
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
    cv2.putText(
        display_frame,
        f"err(px)=({ctrl_out.err_x_px:.0f},{ctrl_out.err_y_px:.0f}) rpm=({ctrl_out.yaw_rpm:.1f},{ctrl_out.pitch_rpm:.1f})",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        control_text_color,
        2,
    )
    if best is not None:
        cv2.putText(
            display_frame,
            f"pass={best.pass_index} score={best.score:.2f}",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            control_text_color,
            2,
        )

    if competition is None:
        return

    aim_x, aim_y = competition["aim_center"]
    ready_color = 255 if is_gray_display else ((0, 255, 0) if competition["ready"] else (0, 165, 255))
    cv2.circle(
        display_frame,
        (int(round(aim_x)), int(round(aim_y))),
        max(4, int(round(competition["aim_radius"]))),
        ready_color,
        1,
    )

    laser_roi = competition["laser_roi"]
    if laser_roi is not None:
        cv2.rectangle(display_frame, laser_roi, 180 if is_gray_display else (0, 255, 255), 1)

    laser_bbox = competition["laser_bbox"]
    laser_center = competition["laser_center"]
    if laser_bbox is not None and laser_center is not None:
        laser_color = 255 if is_gray_display else (
            (255, 255, 255) if competition["laser_current"] else (0, 255, 255)
        )
        cv2.rectangle(display_frame, laser_bbox, laser_color, 2)
        cv2.drawMarker(
            display_frame,
            (int(round(laser_center[0])), int(round(laser_center[1]))),
            laser_color,
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=2,
        )

    err_x, err_y = competition["error"]
    cv2.putText(
        display_frame,
        f"stage={competition['stage']} ready={int(competition['ready'])} confirm={competition['confirm_count']}",
        (10, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        control_text_color,
        2,
    )
    cv2.putText(
        display_frame,
        f"vision_err=({err_x:.0f},{err_y:.0f}) laser_score={competition['laser_score']:.0f} "
        f"luma={competition['laser_luma']} violet={competition['laser_violet']}",
        (10, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        control_text_color,
        1,
    )

    if "a4_state" not in competition:
        return

    for x, y, score in competition["a4_edge_points"]:
        if is_gray_display:
            point_color = 255 if score >= 0.45 else 120
        else:
            if score >= 0.45:
                point_color = (0, 255, 0)
            elif score >= 0.30:
                point_color = (0, 215, 255)
            else:
                point_color = (0, 0, 255)
        cv2.circle(display_frame, (int(round(x)), int(round(y))), 2, point_color, -1)

    cv2.putText(
        display_frame,
        f"{competition.get('target_model', 'a4')}={competition['a4_state']} "
        f"mode={'full' if competition['a4_require_red_rings'] else 'frame'} "
        f"conf={competition['a4_confidence']:.2f} "
        f"struct={competition['a4_structural']:.2f} sides={competition['a4_visible_sides']} "
        f"flow={competition['a4_flow_inliers']}",
        (10, 178),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        control_text_color,
        1,
    )
    cv2.putText(
        display_frame,
        (
            f"id={competition['e25_identity']:.2f} meas={competition['e25_measurement']:.2f} "
            f"track={competition['e25_tracking']:.2f} ctl={int(competition['e25_control_valid'])}"
            if competition.get('target_model') == 'e25'
            else f"edge={competition['a4_edge']:.2f} black={competition['a4_black']:.2f} "
                 f"red={competition['a4_red']:.2f} temporal={competition['a4_temporal']:.2f}"
        ),
        (10, 202),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        control_text_color,
        1,
    )
    if competition["a4_error_mm"] is not None:
        error_mm_x, error_mm_y = competition["a4_error_mm"]
        cv2.putText(
            display_frame,
            f"laser_error_mm=({error_mm_x:.1f},{error_mm_y:.1f})",
            (10, 226),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            control_text_color,
            2,
        )

    canonical = competition["a4_canonical"]
    if canonical is not None and not is_gray_display:
        thumb_width = min(150, max(80, frame_w // 6))
        thumb_height = max(1, int(round(canonical.shape[0] * thumb_width / canonical.shape[1])))
        if thumb_height > frame_h // 3:
            thumb_height = frame_h // 3
            thumb_width = max(1, int(round(canonical.shape[1] * thumb_height / canonical.shape[0])))
        thumbnail = cv2.resize(canonical, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
        x1 = frame_w - thumb_width - 10
        y1 = 10
        display_frame[y1 : y1 + thumb_height, x1 : x1 + thumb_width] = thumbnail
        cv2.rectangle(display_frame, (x1, y1), (x1 + thumb_width, y1 + thumb_height), (0, 255, 255), 1)


class A4RuntimeSettings:
    def __init__(self, enabled: bool, require_red_rings: bool):
        self._lock = threading.Lock()
        self.enabled = bool(enabled)
        self._require_red_rings = bool(require_red_rings)

    def get_require_red_rings(self) -> bool:
        with self._lock:
            return self._require_red_rings

    def set_values(self, payload):
        with self._lock:
            if "require_red_rings" in payload:
                self._require_red_rings = bool(payload["require_red_rings"])
            return self.as_payload_unlocked()

    def as_payload(self):
        with self._lock:
            return self.as_payload_unlocked()

    def as_payload_unlocked(self):
        return {
            "enabled": self.enabled,
            "require_red_rings": self._require_red_rings,
        }


class ColorAdjust:
    def __init__(self):
        self._lock = threading.Lock()
        self._values = COLOR_DEFAULTS.copy()
        self._gamma_key = None
        self._gamma_table = None
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def get_values(self):
        with self._lock:
            return self._values.copy()

    def reset(self):
        with self._lock:
            self._values = COLOR_DEFAULTS.copy()
            return self._values.copy()

    def apply_preset(self, name: str):
        preset = COLOR_PRESETS.get(name)
        if preset is None:
            return None
        with self._lock:
            self._values = preset.copy()
            return self._values.copy()

    def set_values(self, updates):
        with self._lock:
            for key, value in updates.items():
                if key not in COLOR_RANGES:
                    continue
                lo, hi, _ = COLOR_RANGES[key]
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(number):
                    continue
                self._values[key] = min(max(number, lo), hi)
            return self._values.copy()

    def as_payload(self):
        return {
            "values": self.get_values(),
            "defaults": COLOR_DEFAULTS,
            "presets": COLOR_PRESETS,
            "ranges": {
                key: {"min": value[0], "max": value[1], "step": value[2]}
                for key, value in COLOR_RANGES.items()
            },
        }

    def apply(self, frame):
        values = self.get_values()
        if frame.ndim != 3:
            return self._apply_luma(frame, values)
        if all(abs(values[key] - COLOR_DEFAULTS[key]) < 1e-9 for key in COLOR_DEFAULTS):
            return frame

        adjusted = frame.astype(np.float32)
        adjusted = self._apply_auto_levels(adjusted, values)
        adjusted = self._apply_auto_white_balance(adjusted, values)
        adjusted[:, :, 0] *= values["blue"]
        adjusted[:, :, 1] *= values["green"]
        adjusted[:, :, 2] *= values["red"]
        adjusted = adjusted * values["contrast"] + values["brightness"]
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

        if (
            abs(values["saturation"] - 1.0) > 1e-9
            or abs(values["hue"]) > 1e-9
            or abs(values["vibrance"]) > 1e-9
        ):
            hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + values["hue"] / 2.0) % 180.0
            saturation = hsv[:, :, 1] * values["saturation"]
            vibrance = values["vibrance"]
            if vibrance >= 0:
                saturation += (255.0 - saturation) * (vibrance * 0.22)
            else:
                saturation *= 1.0 + vibrance * 0.5
            hsv[:, :, 1] = np.clip(saturation, 0, 255)
            adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        adjusted = self._apply_gamma(adjusted, values["gamma"])
        adjusted = self._apply_clarity(adjusted, values["clarity"])
        adjusted = self._apply_sharpness(adjusted, values["sharpness"])

        return adjusted

    def _apply_auto_levels(self, frame, values):
        strength = values["auto_level"]
        if strength <= 1e-9:
            return frame

        gray = cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        sample = gray[::4, ::4].reshape(-1)
        low = float(np.percentile(sample, values["level_low"]))
        high = float(np.percentile(sample, values["level_high"]))
        if high - low < 8.0:
            return frame

        leveled = (frame - low) * (255.0 / (high - low))
        return frame * (1.0 - strength) + leveled * strength

    @staticmethod
    def _apply_auto_white_balance(frame, values):
        strength = values["auto_wb"]
        if strength <= 1e-9:
            return frame

        sample = frame[::4, ::4]
        luma = sample[:, :, 2] * 0.299 + sample[:, :, 1] * 0.587 + sample[:, :, 0] * 0.114
        mask = (luma > 20.0) & (luma < 245.0)
        if int(mask.sum()) >= 100:
            pixels = sample[mask].reshape(-1, 3)
        else:
            pixels = sample.reshape(-1, 3)

        means = pixels.mean(axis=0)
        target = float(means.mean())
        gains = np.clip(target / np.maximum(means, 1.0), 0.45, 2.25)
        balanced = frame * gains.reshape(1, 1, 3)
        return frame * (1.0 - strength) + balanced * strength

    def _apply_gamma(self, frame, gamma: float):
        if abs(gamma - 1.0) <= 1e-9:
            return frame
        key = round(float(gamma), 4)
        if self._gamma_key != key:
            inv_gamma = 1.0 / gamma
            self._gamma_table = np.array(
                [((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)],
                dtype=np.uint8,
            )
            self._gamma_key = key
        return cv2.LUT(frame, self._gamma_table)

    def _apply_clarity(self, frame, clarity: float):
        if clarity <= 1e-9:
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        luma, a, b = cv2.split(lab)
        enhanced = self._clahe.apply(luma)
        luma = cv2.addWeighted(luma, 1.0 - clarity, enhanced, clarity, 0)
        return cv2.cvtColor(cv2.merge((luma, a, b)), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _apply_sharpness(frame, sharpness: float):
        if sharpness <= 1e-9:
            return frame
        blurred = cv2.GaussianBlur(frame, (0, 0), 1.0)
        return cv2.addWeighted(frame, 1.0 + sharpness, blurred, -sharpness, 0)

    @staticmethod
    def _apply_luma(frame, values):
        if (
            abs(values["brightness"]) < 1e-9
            and abs(values["contrast"] - 1.0) < 1e-9
            and abs(values["gamma"] - 1.0) < 1e-9
            and abs(values["auto_level"]) < 1e-9
        ):
            return frame
        adjusted = frame.astype(np.float32)
        strength = values["auto_level"]
        if strength > 1e-9:
            sample = adjusted[::4, ::4].reshape(-1)
            low = float(np.percentile(sample, values["level_low"]))
            high = float(np.percentile(sample, values["level_high"]))
            if high - low >= 8.0:
                leveled = (adjusted - low) * (255.0 / (high - low))
                adjusted = adjusted * (1.0 - strength) + leveled * strength

        adjusted = adjusted * values["contrast"] + values["brightness"]
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        gamma = values["gamma"]
        if abs(gamma - 1.0) > 1e-9:
            inv_gamma = 1.0 / gamma
            table = np.array(
                [((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)],
                dtype=np.uint8,
            )
            adjusted = cv2.LUT(adjusted, table)
        return adjusted


def build_preview_html():
    controls = [
        ("auto_wb", "Auto WB"),
        ("auto_level", "Auto Levels"),
        ("level_low", "Black point"),
        ("level_high", "White point"),
        ("brightness", "Brightness"),
        ("contrast", "Contrast"),
        ("saturation", "Saturation"),
        ("vibrance", "Vibrance"),
        ("hue", "Hue"),
        ("gamma", "Gamma"),
        ("red", "Red gain"),
        ("green", "Green gain"),
        ("blue", "Blue gain"),
        ("clarity", "Clarity"),
        ("sharpness", "Sharpness"),
    ]
    control_html = "\n".join(
        f'<label><span>{label}</span><input id="{key}" type="range">'
        f'<output id="{key}-value"></output></label>'
        for key, label in controls
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>QGimbal Vision Preview</title>
  <style>
    body {{ margin:0; background:#111; color:#eee; font-family:sans-serif; }}
    main {{ display:grid; grid-template-columns:1fr 340px; min-height:100vh; }}
    .preview {{ display:flex; align-items:center; justify-content:center; min-width:0; }}
    img {{ max-width:100%; max-height:100vh; display:block; }}
    aside {{ box-sizing:border-box; padding:16px; background:#1b1b1b; border-left:1px solid #333; }}
    h1 {{ font-size:18px; margin:0 0 14px; }}
    label {{ display:grid; grid-template-columns:100px 1fr 58px; gap:10px; align-items:center; margin:12px 0; }}
    input[type=range] {{ width:100%; }}
    output {{ text-align:right; color:#b7d7ff; font-variant-numeric:tabular-nums; }}
    .presets {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin:0 0 14px; }}
    .a4-settings {{ margin:0 0 18px; padding:0 0 16px; border-bottom:1px solid #333; }}
    .toggle {{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin:0; }}
    .toggle input {{ width:18px; height:18px; }}
    button {{ width:100%; margin-top:12px; padding:9px 10px; border:0; color:#111; background:#eee; cursor:pointer; }}
    .presets button {{ margin-top:0; }}
    pre {{ white-space:pre-wrap; color:#aaa; font-size:12px; line-height:1.4; }}
    @media (max-width: 900px) {{ main {{ grid-template-columns:1fr; }} aside {{ border-left:0; border-top:1px solid #333; }} }}
  </style>
</head>
<body>
  <main>
    <section class="preview"><img src="/stream.mjpg"></section>
    <aside>
      <section id="a4-settings" class="a4-settings" hidden>
        <h1>A4 Target</h1>
        <label class="toggle"><span>Require red rings</span><input id="require-red-rings" type="checkbox"></label>
      </section>
      <h1>Color Adjust</h1>
      <div class="presets">
        <button data-preset="a7a_realtime">Realtime</button>
        <button data-preset="a7a_soft">Soft</button>
        <button data-preset="a7a_balanced">Balanced</button>
        <button data-preset="a7a_strong">Strong</button>
      </div>
      {control_html}
      <button id="reset">Neutral</button>
      <button id="copy">Copy JSON</button>
      <pre id="status"></pre>
    </aside>
  </main>
  <script>
    const keys = {json.dumps([key for key, _ in controls])};
    let ranges = {{}};
    let sendTimer = null;

    async function loadA4Settings() {{
      const res = await fetch('/api/a4');
      const data = await res.json();
      if (!data.enabled) return;
      const section = document.getElementById('a4-settings');
      const toggle = document.getElementById('require-red-rings');
      section.hidden = false;
      toggle.checked = Boolean(data.require_red_rings);
      toggle.addEventListener('change', async () => {{
        const response = await fetch('/api/a4', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{require_red_rings: toggle.checked}}),
        }});
        const updated = await response.json();
        toggle.checked = Boolean(updated.require_red_rings);
      }});
    }}

    async function loadControls() {{
      const res = await fetch('/api/color');
      const data = await res.json();
      ranges = data.ranges;
      for (const key of keys) {{
        const input = document.getElementById(key);
        const output = document.getElementById(`${{key}}-value`);
        input.min = ranges[key].min;
        input.max = ranges[key].max;
        input.step = ranges[key].step;
        input.value = data.values[key];
        output.value = formatValue(key, data.values[key]);
        input.addEventListener('input', () => {{
          output.value = formatValue(key, input.value);
          scheduleSend();
        }});
      }}
      renderStatus(data.values);
    }}

    function formatValue(key, value) {{
      const step = Number(ranges[key]?.step ?? 0.01);
      const digits = step >= 1 ? 0 : step >= 0.1 ? 1 : 2;
      return Number(value).toFixed(digits);
    }}

    function currentValues() {{
      const values = {{}};
      for (const key of keys) values[key] = Number(document.getElementById(key).value);
      return values;
    }}

    function scheduleSend() {{
      clearTimeout(sendTimer);
      sendTimer = setTimeout(sendControls, 80);
    }}

    async function sendControls() {{
      const values = currentValues();
      const res = await fetch('/api/color', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(values),
      }});
      const data = await res.json();
      renderStatus(data.values);
    }}

    async function resetControls() {{
      const res = await fetch('/api/color/reset', {{method: 'POST'}});
      const data = await res.json();
      for (const key of keys) {{
        const input = document.getElementById(key);
        input.value = data.values[key];
        document.getElementById(`${{key}}-value`).value = formatValue(key, data.values[key]);
      }}
      renderStatus(data.values);
    }}

    async function applyPreset(name) {{
      const res = await fetch('/api/color/preset', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{name}}),
      }});
      const data = await res.json();
      for (const key of keys) {{
        const input = document.getElementById(key);
        input.value = data.values[key];
        document.getElementById(`${{key}}-value`).value = formatValue(key, data.values[key]);
      }}
      renderStatus(data.values);
    }}

    function renderStatus(values) {{
      document.getElementById('status').textContent = JSON.stringify(values, null, 2);
    }}

    document.getElementById('reset').addEventListener('click', resetControls);
    document.getElementById('copy').addEventListener('click', async () => {{
      await navigator.clipboard.writeText(JSON.stringify(currentValues()));
    }});
    for (const button of document.querySelectorAll('[data-preset]')) {{
      button.addEventListener('click', () => applyPreset(button.dataset.preset));
    }}
    loadControls();
    loadA4Settings();
  </script>
</body>
</html>"""


class MjpegStreamer:
    def __init__(
        self,
        host: str,
        port: int,
        quality: int,
        a4_settings: A4RuntimeSettings,
    ):
        self.host = host
        self.port = port
        self.quality = quality
        self.a4_settings = a4_settings
        self.color_adjust = ColorAdjust()
        self._condition = threading.Condition()
        self._jpeg: Optional[bytes] = None
        self._frame_id = 0
        self._encode_condition = threading.Condition()
        self._pending_frame = None
        self._pending_frame_id = 0
        self._stopped = False
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._encoder_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path in ('/', '/index.html'):
                    body = build_preview_html().encode('utf-8')
                    self.send_response(HTTPStatus.OK)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if parsed.path == '/api/a4':
                    self._send_json(streamer.a4_settings.as_payload())
                    return

                if parsed.path == '/api/color':
                    updates = {
                        key: values[-1]
                        for key, values in parse_qs(parsed.query).items()
                        if values
                    }
                    if updates:
                        streamer.color_adjust.set_values(updates)
                    self._send_json(streamer.color_adjust.as_payload())
                    return

                if parsed.path == '/snapshot.jpg':
                    with streamer._condition:
                        jpeg = streamer._jpeg
                    if jpeg is None:
                        self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, 'No frame yet')
                        return
                    self.send_response(HTTPStatus.OK)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg)
                    return

                if parsed.path != '/stream.mjpg':
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return

                self.send_response(HTTPStatus.OK)
                self.send_header('Age', '0')
                self.send_header('Cache-Control', 'no-cache, private')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()

                last_frame_id = -1
                try:
                    while True:
                        with streamer._condition:
                            streamer._condition.wait_for(lambda: streamer._frame_id != last_frame_id)
                            jpeg = streamer._jpeg
                            last_frame_id = streamer._frame_id
                        if jpeg is None:
                            continue
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n')
                        self.wfile.write(f'Content-Length: {len(jpeg)}\r\n\r\n'.encode('ascii'))
                        self.wfile.write(jpeg)
                        self.wfile.write(b'\r\n')
                except (BrokenPipeError, ConnectionResetError):
                    return

            def do_POST(self):
                parsed = urlparse(self.path)
                if parsed.path == '/api/a4':
                    length = int(self.headers.get('Content-Length', '0') or '0')
                    body = self.rfile.read(length) if length > 0 else b''
                    payload = {}
                    if body:
                        try:
                            payload = json.loads(body.decode('utf-8'))
                        except json.JSONDecodeError:
                            self.send_error(HTTPStatus.BAD_REQUEST, 'Invalid JSON')
                            return
                    self._send_json(streamer.a4_settings.set_values(payload))
                    return

                if parsed.path == '/api/color/reset':
                    streamer.color_adjust.reset()
                    self._send_json(streamer.color_adjust.as_payload())
                    return

                if parsed.path == '/api/color/preset':
                    length = int(self.headers.get('Content-Length', '0') or '0')
                    body = self.rfile.read(length) if length > 0 else b''
                    preset_name = parse_qs(parsed.query).get('name', [''])[0]
                    if body:
                        try:
                            payload = json.loads(body.decode('utf-8'))
                        except json.JSONDecodeError:
                            self.send_error(HTTPStatus.BAD_REQUEST, 'Invalid JSON')
                            return
                        preset_name = str(payload.get('name', preset_name))

                    if streamer.color_adjust.apply_preset(preset_name) is None:
                        self.send_error(HTTPStatus.BAD_REQUEST, 'Unknown preset')
                        return
                    self._send_json(streamer.color_adjust.as_payload())
                    return

                if parsed.path != '/api/color':
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return

                length = int(self.headers.get('Content-Length', '0') or '0')
                body = self.rfile.read(length) if length > 0 else b''
                updates = {}
                if body:
                    try:
                        updates = json.loads(body.decode('utf-8'))
                    except json.JSONDecodeError:
                        self.send_error(HTTPStatus.BAD_REQUEST, 'Invalid JSON')
                        return
                streamer.color_adjust.set_values(updates)
                self._send_json(streamer.color_adjust.as_payload())

            def _send_json(self, payload):
                body = json.dumps(payload).encode('utf-8')
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_port
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._encoder_thread = threading.Thread(target=self._encode_loop, daemon=True)
        self._encoder_thread.start()

    def update(self, frame) -> None:
        with self._encode_condition:
            self._pending_frame = frame.copy()
            self._pending_frame_id += 1
            self._encode_condition.notify()

    def _encode_loop(self) -> None:
        last_encoded_id = 0
        while True:
            with self._encode_condition:
                self._encode_condition.wait_for(
                    lambda: self._stopped or self._pending_frame_id != last_encoded_id
                )
                if self._stopped:
                    return
                frame = self._pending_frame
                last_encoded_id = self._pending_frame_id

            if frame is None:
                continue

            frame = self.color_adjust.apply(frame)
            self._encode_frame(frame)

    def _encode_frame(self, frame) -> None:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.quality)]
        ok, encoded = cv2.imencode('.jpg', frame, encode_params)
        if not ok:
            return
        with self._condition:
            self._jpeg = encoded.tobytes()
            self._frame_id += 1
            self._condition.notify_all()

    def stop(self) -> None:
        with self._encode_condition:
            self._stopped = True
            self._encode_condition.notify_all()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


def detect_with_scale(
    frame,
    detect_scale: float,
    multi_pass: bool,
    max_area_ratio: float,
):
    detect_func = detect_rectangles_multi_pass if multi_pass else detect_rectangles
    if detect_scale == 1.0:
        if multi_pass:
            return detect_func(frame, min_area_ratio=0.005, max_area_ratio=max_area_ratio)
        return detect_func(frame, min_area_ratio=0.005, max_area_ratio=max_area_ratio, angle_tol=25.0)

    small = cv2.resize(frame, (0, 0), fx=detect_scale, fy=detect_scale, interpolation=cv2.INTER_AREA)
    if multi_pass:
        rects = detect_func(small, min_area_ratio=0.005, max_area_ratio=max_area_ratio)
    else:
        rects = detect_func(small, min_area_ratio=0.005, max_area_ratio=max_area_ratio, angle_tol=25.0)
    inv = 1.0 / detect_scale
    return [
        DetectedRect(
            center=(rect.center[0] * inv, rect.center[1] * inv),
            box=rect.box * inv,
            area=rect.area * inv * inv,
            pass_index=rect.pass_index,
            score=rect.score,
        )
        for rect in rects
    ]


def detect_candidates(
    args,
    raw_frame,
    detect_frame,
    width: int,
    height: int,
    yolo_detector: Optional[AsyncYoloDetector],
    frame_index: int,
):
    detector = args.detector
    if detector in ('yolo', 'hybrid') and yolo_detector is not None:
        should_run_yolo = frame_index % int(args.yolo_every) == 0
        if should_run_yolo:
            yolo_frame = make_display_frame(raw_frame, detect_frame, width, height, args, float(args.yolo_scale))
            yolo_detector.submit(yolo_frame)

        yolo_result = yolo_detector.latest()
        if yolo_result is not None and yolo_result.rects:
            frame_h, frame_w = detect_frame.shape[:2]
            yolo_w, yolo_h = yolo_result.frame_size
            return scale_detected_rects(yolo_result.rects, frame_w / yolo_w, frame_h / yolo_h)
        if detector == 'yolo':
            return []

    return detect_with_scale(
        detect_frame,
        float(args.detect_scale),
        bool(args.detect_multi_pass),
        float(args.rect_max_area_ratio),
    )


def print_status(fps: float, best, ctrl_out, a4_result=None) -> None:
    a4_text = ""
    if a4_result is not None:
        confidence = 0.0 if a4_result.detection is None else a4_result.detection.confidence
        label = "e25" if hasattr(a4_result, "confidence") else "a4"
        a4_text = (
            f" {label}={a4_result.state.value} conf={confidence:.2f} "
            f"flow={a4_result.flow_inliers} miss={a4_result.miss_count}"
        )
    if best is None:
        print(f"fps={fps:.1f} rect=none rpm=({ctrl_out.yaw_rpm:.1f},{ctrl_out.pitch_rpm:.1f}){a4_text}")
    else:
        cx, cy = best.center
        area = best.area
        print(
            f"fps={fps:.1f} cx={cx:.1f} cy={cy:.1f} area={area:.0f} "
            f"pass={best.pass_index} score={best.score:.2f} "
            f"err=({ctrl_out.err_x_px:.0f},{ctrl_out.err_y_px:.0f}) "
            f"rpm=({ctrl_out.yaw_rpm:.1f},{ctrl_out.pitch_rpm:.1f}){a4_text}"
        )


def main():
    args = parse_args()
    width, height = parse_size(args.size)
    output_width, output_height = (
        parse_size(args.output_size) if args.output_size else (width, height)
    )
    if args.capture_mode == 'raw' and (output_width, output_height) != (width, height):
        raise SystemExit('--output-size 仅支持 --capture-mode bgr')
    if args.max_processing_fps < 0.0:
        raise SystemExit('--max-processing-fps 必须大于等于 0')
    if not 0.0 < args.detect_scale <= 1.0:
        raise SystemExit('--detect-scale 必须在 0 到 1 之间')
    if not 0.0 < args.yolo_scale <= 1.0:
        raise SystemExit('--yolo-scale 必须在 0 到 1 之间')
    if args.yolo_every < 1:
        raise SystemExit('--yolo-every 必须大于等于 1')
    if args.detector in ('yolo', 'hybrid') and not args.yolo_command:
        raise SystemExit('--detector yolo/hybrid 需要提供 --yolo-command')
    if args.yolo_timeout <= 0.0:
        raise SystemExit('--yolo-timeout 必须大于 0')
    if not 0.0 <= args.yolo_min_confidence <= 1.0:
        raise SystemExit('--yolo-min-confidence 必须在 0 到 1 之间')
    if not 1 <= args.yolo_jpeg_quality <= 100:
        raise SystemExit('--yolo-jpeg-quality 必须在 1 到 100 之间')
    if args.rect_center_weight < 0.0:
        raise SystemExit('--rect-center-weight 必须大于等于 0')
    if args.rect_prev_weight < 0.0:
        raise SystemExit('--rect-prev-weight 必须大于等于 0')
    if args.rect_max_aspect < 1.0:
        raise SystemExit('--rect-max-aspect 必须大于等于 1')
    if not 0.0 < args.rect_max_area_ratio <= 1.0:
        raise SystemExit('--rect-max-area-ratio 必须在 0 到 1 之间')
    if args.target_miss_frames < 1:
        raise SystemExit('--target-miss-frames 必须大于等于 1')
    if args.aim_enter_radius_ratio <= 0.0:
        raise SystemExit('--aim-enter-radius-ratio 必须大于 0')
    if args.aim_exit_radius_ratio < args.aim_enter_radius_ratio:
        raise SystemExit('--aim-exit-radius-ratio 必须大于等于 --aim-enter-radius-ratio')
    if args.aim_confirm_frames < 1:
        raise SystemExit('--aim-confirm-frames 必须大于等于 1')
    if args.laser_hold_frames < 0:
        raise SystemExit('--laser-hold-frames 必须大于等于 0')
    if not 0 <= args.laser_min_luma <= 255:
        raise SystemExit('--laser-min-luma 必须在 0 到 255 之间')
    if not args.laser_min_luma <= args.laser_fallback_min_luma <= 255:
        raise SystemExit('--laser-fallback-min-luma 必须不低于 --laser-min-luma 且不超过 255')
    if args.a4_global_interval < 1:
        raise SystemExit('--a4-global-interval 必须大于等于 1')
    if args.a4_search_interval < 1:
        raise SystemExit('--a4-search-interval 必须大于等于 1')
    if args.a4_local_validate_interval < 1:
        raise SystemExit('--a4-local-validate-interval 必须大于等于 1')
    if not 0.0 < args.a4_min_area_ratio < args.rect_max_area_ratio:
        raise SystemExit('--a4-min-area-ratio 必须大于 0 且小于 --rect-max-area-ratio')
    if not 1.0 <= args.a4_min_apparent_aspect < 2.3:
        raise SystemExit('--a4-min-apparent-aspect 必须在 1.0 到 2.3 之间')
    if not 0.0 <= args.a4_track_confidence <= args.a4_acquire_confidence <= 1.0:
        raise SystemExit('A4跟踪门限必须不高于首次锁定门限，且均在 0 到 1 之间')
    if args.a4_occlusion_frames < 1:
        raise SystemExit('--a4-occlusion-frames 必须大于等于 1')
    if args.a4_target and args.e25_target:
        raise SystemExit('--a4-target 与 --e25-target 不能同时启用')
    if (args.a4_target or args.e25_target) and args.detector != 'rect':
        raise SystemExit('A4/E25模型模式目前只支持 --detector rect')
    if not 0.0 < args.display_scale <= 1.0:
        raise SystemExit('--display-scale 必须在 0 到 1 之间')
    if args.display_every < 1:
        raise SystemExit('--display-every 必须大于等于 1')
    if not 0.0 < args.stream_scale <= 1.0:
        raise SystemExit('--stream-scale 必须在 0 到 1 之间')
    if args.stream_every < 1:
        raise SystemExit('--stream-every 必须大于等于 1')
    if not 1 <= args.stream_quality <= 100:
        raise SystemExit('--stream-quality 必须在 1 到 100 之间')
    if args.stream_port < 0:
        raise SystemExit('--stream-port 必须大于等于 0')

    cap = open_capture(args, width, height, output_width, output_height)

    display = bool(args.display)
    a4_settings = A4RuntimeSettings(
        enabled=bool(args.a4_target or args.e25_target),
        require_red_rings=bool(args.a4_require_red_rings),
    )
    streamer = None
    if args.stream_port > 0:
        streamer = MjpegStreamer(
            args.stream_host,
            args.stream_port,
            int(args.stream_quality),
            a4_settings,
        )
        streamer.start()
        print(f'MJPEG preview: http://<board-ip>:{streamer.port}/  stream=/stream.mjpg snapshot=/snapshot.jpg')

    # 控制器初始化（串口协议先留 stub，你后续替换 send_rpm 即可）
    ctrl_cfg = ControlConfig(
        enabled=bool(args.control),
        deadband_px=float(args.deadband_px),
        lost_timeout_s=float(args.lost_timeout),
        max_rpm_yaw=float(args.max_rpm),
        max_rpm_pitch=float(args.max_rpm),
    )
    tracker = GimbalTracker(ctrl_cfg)
    serial = GimbalSerialStub(port=args.serial_port, baudrate=int(args.serial_baud))
    serial.open()
    yolo_detector = None
    if args.detector in ('yolo', 'hybrid'):
        labels = set(args.yolo_label) if args.yolo_label else None
        yolo_detector = AsyncYoloDetector.from_shell_command(
            args.yolo_command,
            timeout_s=float(args.yolo_timeout),
            jpeg_quality=int(args.yolo_jpeg_quality),
            min_confidence=float(args.yolo_min_confidence),
            labels=labels,
        )
    selection_config = RectSelectionConfig(
        max_area_ratio=float(args.rect_max_area_ratio),
        max_aspect_ratio=float(args.rect_max_aspect),
        center_weight=float(args.rect_center_weight),
        previous_weight=float(args.rect_prev_weight),
    )
    a4_mode = bool(args.a4_target)
    e25_mode = bool(args.e25_target)
    model_target_mode = a4_mode or e25_mode
    competition_mode = bool(args.competition_mode or model_target_mode)
    rect_selector = None if competition_mode else RectSelector(selection_config)
    target_tracker = None
    a4_tracker = None
    e25_tracker = None
    aim_gate = None
    laser_tracker = None
    if competition_mode:
        if e25_mode:
            e25_tracker = E25VisionPipeline(
                E25PipelineConfig(
                    min_area_ratio=float(args.a4_min_area_ratio),
                    max_area_ratio=float(args.rect_max_area_ratio),
                    min_apparent_aspect=float(args.a4_min_apparent_aspect),
                    acquire_confidence=float(args.a4_acquire_confidence),
                    track_identity_confidence=float(args.a4_track_confidence),
                    occlusion_hold_frames=int(args.a4_occlusion_frames),
                    static_global_interval=int(args.a4_global_interval),
                    search_interval=int(args.a4_search_interval),
                    structural_validate_interval=int(args.a4_local_validate_interval),
                ),
                require_red_rings=a4_settings.get_require_red_rings(),
            )
        elif a4_mode:
            a4_tracker = A4TargetTracker(
                A4TargetConfig(
                    min_area_ratio=float(args.a4_min_area_ratio),
                    max_area_ratio=float(args.rect_max_area_ratio),
                    min_apparent_aspect=float(args.a4_min_apparent_aspect),
                    acquire_confidence=float(args.a4_acquire_confidence),
                    track_confidence=float(args.a4_track_confidence),
                    occlusion_hold_frames=int(args.a4_occlusion_frames),
                    global_interval=int(args.a4_global_interval),
                    search_interval=int(args.a4_search_interval),
                    local_validate_interval=int(args.a4_local_validate_interval),
                ),
                require_red_rings=a4_settings.get_require_red_rings(),
            )
        else:
            target_tracker = TargetTracker(
                selection_config,
                miss_confirm_frames=int(args.target_miss_frames),
            )
        aim_gate = AimReadyGate(
            enter_radius_ratio=float(args.aim_enter_radius_ratio),
            exit_radius_ratio=float(args.aim_exit_radius_ratio),
            confirm_frames=int(args.aim_confirm_frames),
            offset_x_ratio=float(args.aim_offset_x_ratio),
            offset_y_ratio=float(args.aim_offset_y_ratio),
        )
        if e25_mode:
            laser_tracker = E25LaserTracker()
        else:
            laser_tracker = HybridLaserTracker(
                HybridLaserConfig(
                    min_dynamic_luma=int(args.laser_min_luma),
                    fallback_min_luma=int(args.laser_fallback_min_luma),
                    hold_frames=int(args.laser_hold_frames),
                )
            )
        print(
            'Competition vision enabled: target hold, aim gate, '
            'LAB violet + dynamic bright-core laser tracking'
        )
        if a4_mode:
            print(
                'A4 target mode enabled: black-tape/red-ring structure, '
                '64 edge points, LK homography and occlusion state machine'
            )
        if e25_mode:
            print(
                'E25 model mode enabled: global identity, per-side normal scans, '
                'robust line fitting, motion prediction and IMX415 laser fusion'
            )

    win_name = f"Camera {args.device if args.backend == 'gstreamer' else args.camera}"
    if display:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # 无窗口模式：终端输出节流
    last_print = 0.0
    prev_time = time.time()
    fps = 0.0
    fps_window_start = prev_time
    fps_window_frames = 0
    frame_index = 0
    frame_period = 1.0 / args.max_processing_fps if args.max_processing_fps > 0.0 else 0.0
    next_frame_deadline = time.monotonic()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("无法从摄像头读取到帧，正在重试...")
                time.sleep(0.1)
                continue

            frame_index += 1
            detect_frame = extract_frame(frame, output_width, output_height, args)
            h, w = detect_frame.shape[:2]
            if model_target_mode:
                model_tracker = e25_tracker if e25_mode else a4_tracker
                model_tracker.set_require_red_rings(a4_settings.get_require_red_rings())
                a4_detection_cycle = model_tracker.needs_global_detection(frame_index)
                if a4_detection_cycle:
                    base_rects = detect_candidates(
                        args,
                        frame,
                        detect_frame,
                        output_width,
                        output_height,
                        yolo_detector,
                        frame_index,
                    )
                    black_rects = (
                        find_black_band_candidates(
                            detect_frame,
                            model_tracker.detector.config if e25_mode else model_tracker.config,
                        )
                        if e25_mode or len(base_rects) < 2
                        else []
                    )
                    rects = merge_candidates(base_rects, black_rects, limit=12)
                else:
                    rects = []
            else:
                rects = detect_candidates(
                    args,
                    frame,
                    detect_frame,
                    output_width,
                    output_height,
                    yolo_detector,
                    frame_index,
                )

            target_track = None
            a4_result = None
            laser_track = None
            aim_status = None
            vision_stage = None
            vision_error = (0.0, 0.0)
            if competition_mode:
                if model_target_mode:
                    if e25_mode:
                        a4_result = e25_tracker.update(
                            detect_frame,
                            rects,
                            detection_cycle=a4_detection_cycle,
                            dt=1.0 / max(float(args.fps), 1.0),
                        )
                    else:
                        a4_result = a4_tracker.update(
                            detect_frame,
                            rects,
                            detection_cycle=a4_detection_cycle,
                        )
                    if a4_result.detection is not None:
                        target_track = TrackedTarget(
                            rect=detection_as_rect(a4_result.detection),
                            current=a4_result.current,
                            held=not a4_result.current,
                            miss_count=a4_result.miss_count,
                        )
                else:
                    target_track = target_tracker.update(rects, w, h)
                best = target_track.rect if target_track is not None else None
                aim_status = aim_gate.update(target_track, w, h)
                laser_track = laser_tracker.update(
                    detect_frame,
                    target_track,
                    enabled=(
                        target_track is not None and target_track.current
                        if e25_mode
                        else aim_status.ready
                    ),
                )
                vision_stage = resolve_stage(target_track, aim_status, laser_track)
                vision_error = stage_error(vision_stage, target_track, aim_status, laser_track)
            else:
                best = rect_selector.update(rects, w, h)

            # 计算实际处理 FPS（1 秒窗口），避免启动阶段或显示阻塞造成长期误导。
            now = time.time()
            dt = now - prev_time
            prev_time = now
            fps_window_frames += 1
            fps_window_elapsed = now - fps_window_start
            if fps_window_elapsed >= 1.0:
                fps = fps_window_frames / fps_window_elapsed
                fps_window_frames = 0
                fps_window_start = now

            # PID 控制：将目标中心追踪到屏幕中心，输出 yaw/pitch rpm
            if competition_mode:
                target_center = None
                target_is_valid = (
                    target_track is not None
                    and target_track.current
                    and (
                        not e25_mode
                        or (a4_result is not None and a4_result.confidence.control_valid)
                    )
                )
                if target_is_valid:
                    target_center = target_track.rect.center
                    if (
                        e25_mode
                        and vision_stage == VisionStage.ALIGN_LASER
                        and laser_track is not None
                        and laser_track.current
                    ):
                        target_center = (
                            0.5 * w + vision_error[0],
                            0.5 * h + vision_error[1],
                        )
            else:
                target_center = best.center if best is not None else None
            ret, ctrl_out = tracker.update(frame_w=w, frame_h=h, target_center=target_center, dt=max(dt, 1e-6), now=now)
            if ret:
                serial.send_rpm(ctrl_out.yaw_rpm, ctrl_out.pitch_rpm)

            should_refresh_display = display and frame_index % int(args.display_every) == 0
            should_refresh_stream = streamer is not None and frame_index % int(args.stream_every) == 0
            if should_refresh_stream:
                stream_frame = make_display_frame(
                    frame,
                    detect_frame,
                    output_width,
                    output_height,
                    args,
                    float(args.stream_scale),
                )
                stream_h, stream_w = stream_frame.shape[:2]
                stream_best = scale_detected_rect(best, stream_w / w, stream_h / h)
                stream_competition = make_competition_overlay(
                    target_track,
                    laser_track,
                    aim_status,
                    vision_stage,
                    vision_error,
                    w,
                    h,
                    float(args.aim_enter_radius_ratio),
                    stream_w / w,
                    stream_h / h,
                    a4_result=a4_result,
                    a4_require_red_rings=a4_settings.get_require_red_rings(),
                )
                draw_overlay(
                    stream_frame,
                    stream_best,
                    ctrl_out,
                    fps,
                    stream_w,
                    stream_h,
                    competition=stream_competition,
                )
                streamer.update(stream_frame)

            if should_refresh_display:
                display_frame = make_display_frame(
                    frame, detect_frame, output_width, output_height, args
                )
                display_competition = make_competition_overlay(
                    target_track,
                    laser_track,
                    aim_status,
                    vision_stage,
                    vision_error,
                    w,
                    h,
                    float(args.aim_enter_radius_ratio),
                    a4_result=a4_result,
                    a4_require_red_rings=a4_settings.get_require_red_rings(),
                )
                draw_overlay(
                    display_frame,
                    best,
                    ctrl_out,
                    fps,
                    w,
                    h,
                    competition=display_competition,
                )
                window_frame = scale_display_frame(display_frame, float(args.display_scale))
                cv2.imshow(win_name, window_frame)
                if args.print_in_display and (args.print_interval <= 0 or (now - last_print) >= args.print_interval):
                    last_print = now
                    print_status(fps, best, ctrl_out, a4_result)
                key = cv2.waitKey(1) & 0xFF
                # 按 'q' 或 ESC 退出
                if key == ord('q') or key == 27:
                    break

            if not display:
                # 无窗口：终端输出 FPS + 检测结果（按间隔打印，避免刷屏）
                if args.print_interval <= 0 or (now - last_print) >= args.print_interval:
                    last_print = now
                    print_status(fps, best, ctrl_out, a4_result)

            if frame_period > 0.0:
                next_frame_deadline += frame_period
                delay = next_frame_deadline - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)
                else:
                    next_frame_deadline = time.monotonic()

    except KeyboardInterrupt:
        print('\n收到中断，退出...')
    finally:
        cap.release()
        serial.close()
        if yolo_detector is not None:
            yolo_detector.close()
        if streamer is not None:
            streamer.stop()
        if display:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
