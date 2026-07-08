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

from vision.rect_detect import (
    DetectedRect,
    RectSelectionConfig,
    RectSelector,
    detect_rectangles,
    detect_rectangles_multi_pass,
    draw_detected_rect,
)

from control.config import ControlConfig
from control.serial_stub import GimbalSerialStub
from control.tracker_control import GimbalTracker

DEFAULT_CAMERA = 0  # 摄像头索引（legacy V4L2 backend）
DEFAULT_DEVICE = "/dev/video0"
DEFAULT_SUBDEV = "/dev/v4l-subdev0"
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
    p.add_argument('--size', type=str, default=f'{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}',
                   help=f'采集尺寸 WxH（默认 {DEFAULT_WIDTH}x{DEFAULT_HEIGHT}）')
    p.add_argument('--fps', type=int, default=DEFAULT_FPS,
                   help=f'采集帧率（默认 {DEFAULT_FPS}）')
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

    return src + 'videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false'


def open_capture(args, width: int, height: int):
    if args.backend == 'gstreamer':
        if args.capture_mode == 'raw' and args.format.upper() != 'NV12':
            raise SystemExit('--capture-mode raw 目前只支持 --format NV12')
        set_sensor_format(args.subdev, width, height)
        if args.v4l2_ctrl:
            apply_v4l2_controls(args.device, args.v4l2_ctrl)
        pipeline = build_gst_pipeline(
            args.device,
            width,
            height,
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


def draw_overlay(display_frame, best, ctrl_out, fps: float, frame_w: int, frame_h: int):
    is_gray_display = display_frame.ndim == 2
    rect_color = 255 if is_gray_display else (0, 255, 0)
    marker_color = 255 if is_gray_display else (255, 0, 0)
    text_color = 255 if is_gray_display else (0, 255, 0)
    control_text_color = 200 if is_gray_display else (0, 255, 255)

    if best is not None:
        draw_detected_rect(display_frame, best, color=rect_color)

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
  </script>
</body>
</html>"""


class MjpegStreamer:
    def __init__(self, host: str, port: int, quality: int):
        self.host = host
        self.port = port
        self.quality = quality
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


def print_status(fps: float, best, ctrl_out) -> None:
    if best is None:
        print(f"fps={fps:.1f} rect=none rpm=({ctrl_out.yaw_rpm:.1f},{ctrl_out.pitch_rpm:.1f})")
    else:
        cx, cy = best.center
        area = best.area
        print(
            f"fps={fps:.1f} cx={cx:.1f} cy={cy:.1f} area={area:.0f} "
            f"pass={best.pass_index} score={best.score:.2f} "
            f"err=({ctrl_out.err_x_px:.0f},{ctrl_out.err_y_px:.0f}) rpm=({ctrl_out.yaw_rpm:.1f},{ctrl_out.pitch_rpm:.1f})"
        )


def main():
    args = parse_args()
    width, height = parse_size(args.size)
    if not 0.0 < args.detect_scale <= 1.0:
        raise SystemExit('--detect-scale 必须在 0 到 1 之间')
    if args.rect_center_weight < 0.0:
        raise SystemExit('--rect-center-weight 必须大于等于 0')
    if args.rect_prev_weight < 0.0:
        raise SystemExit('--rect-prev-weight 必须大于等于 0')
    if args.rect_max_aspect < 1.0:
        raise SystemExit('--rect-max-aspect 必须大于等于 1')
    if not 0.0 < args.rect_max_area_ratio <= 1.0:
        raise SystemExit('--rect-max-area-ratio 必须在 0 到 1 之间')
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

    cap = open_capture(args, width, height)

    display = bool(args.display)
    streamer = None
    if args.stream_port > 0:
        streamer = MjpegStreamer(args.stream_host, args.stream_port, int(args.stream_quality))
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
    rect_selector = RectSelector(
        RectSelectionConfig(
            max_area_ratio=float(args.rect_max_area_ratio),
            max_aspect_ratio=float(args.rect_max_aspect),
            center_weight=float(args.rect_center_weight),
            previous_weight=float(args.rect_prev_weight),
        )
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

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("无法从摄像头读取到帧，正在重试...")
                time.sleep(0.1)
                continue

            frame_index += 1
            detect_frame = extract_frame(frame, width, height, args)

            # 对每帧执行矩形检测
            rects = detect_with_scale(
                detect_frame,
                float(args.detect_scale),
                bool(args.detect_multi_pass),
                float(args.rect_max_area_ratio),
            )
            h, w = detect_frame.shape[:2]
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
            target_center = best.center if best is not None else None
            ret, ctrl_out = tracker.update(frame_w=w, frame_h=h, target_center=target_center, dt=max(dt, 1e-6), now=now)
            if ret:
                serial.send_rpm(ctrl_out.yaw_rpm, ctrl_out.pitch_rpm)

            should_refresh_display = display and frame_index % int(args.display_every) == 0
            should_refresh_stream = streamer is not None and frame_index % int(args.stream_every) == 0
            if should_refresh_stream:
                stream_frame = make_display_frame(frame, detect_frame, width, height, args, float(args.stream_scale))
                stream_h, stream_w = stream_frame.shape[:2]
                stream_best = scale_detected_rect(best, stream_w / w, stream_h / h)
                draw_overlay(stream_frame, stream_best, ctrl_out, fps, stream_w, stream_h)
                streamer.update(stream_frame)

            if should_refresh_display:
                display_frame = make_display_frame(frame, detect_frame, width, height, args)
                draw_overlay(display_frame, best, ctrl_out, fps, w, h)
                window_frame = scale_display_frame(display_frame, float(args.display_scale))
                cv2.imshow(win_name, window_frame)
                if args.print_in_display and (args.print_interval <= 0 or (now - last_print) >= args.print_interval):
                    last_print = now
                    print_status(fps, best, ctrl_out)
                key = cv2.waitKey(1) & 0xFF
                # 按 'q' 或 ESC 退出
                if key == ord('q') or key == 27:
                    break

            if not display:
                # 无窗口：终端输出 FPS + 检测结果（按间隔打印，避免刷屏）
                if args.print_interval <= 0 or (now - last_print) >= args.print_interval:
                    last_print = now
                    print_status(fps, best, ctrl_out)

    except KeyboardInterrupt:
        print('\n收到中断，退出...')
    finally:
        cap.release()
        serial.close()
        if streamer is not None:
            streamer.stop()
        if display:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
