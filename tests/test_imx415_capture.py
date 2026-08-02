from types import SimpleNamespace

import cv2
import numpy as np

from main import extract_frame, make_display_frame


def _nv12_frame(width: int, height: int) -> np.ndarray:
    bgr = np.zeros((height, width, 3), dtype=np.uint8)
    bgr[:, : width // 2] = (0, 0, 255)
    bgr[:, width // 2 :] = (0, 255, 0)
    i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    y_size = width * height
    uv_size = y_size // 4
    y = i420[:y_size]
    u = i420[y_size : y_size + uv_size]
    v = i420[y_size + uv_size :]
    uv = np.empty(uv_size * 2, dtype=np.uint8)
    uv[0::2] = u
    uv[1::2] = v
    return np.concatenate((y, uv)).reshape(height + height // 2, width)


def test_raw_nv12_can_feed_color_detection_without_duplicate_conversion() -> None:
    width, height = 8, 4
    args = SimpleNamespace(
        backend='gstreamer',
        capture_mode='raw',
        raw_detect_color=1,
        display_mode='color',
        flip=0,
    )
    raw = _nv12_frame(width, height)
    detected = extract_frame(raw, width, height, args)
    assert detected.shape == (height, width, 3)
    assert detected[:, : width // 2, 2].mean() > 200
    assert detected[:, width // 2 :, 1].mean() > 200

    displayed = make_display_frame(raw, detected, width, height, args)
    assert displayed is detected


def test_raw_nv12_grayscale_mode_remains_available() -> None:
    width, height = 8, 4
    args = SimpleNamespace(
        backend='gstreamer',
        capture_mode='raw',
        raw_detect_color=0,
        display_mode='gray',
        flip=0,
    )
    raw = _nv12_frame(width, height)
    detected = extract_frame(raw, width, height, args)
    assert detected.shape == (height, width)
