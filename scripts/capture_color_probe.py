#!/usr/bin/env python3
"""Capture one A7A NV12 frame and write color conversion probes."""

from __future__ import annotations

import argparse
import os
import subprocess
import time

import cv2
import numpy as np


def parse_size(text: str) -> tuple[int, int]:
    width_text, height_text = text.lower().split("x", 1)
    return int(width_text), int(height_text)


def set_sensor_format(subdev: str, width: int, height: int) -> None:
    subprocess.run(
        [
            "v4l2-ctl",
            "-d",
            subdev,
            "--set-subdev-fmt",
            f"pad=0,code=0x300f,width={width},height={height}",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--subdev", default="/dev/v4l-subdev0")
    parser.add_argument("--size", default="1920x1080")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--awisp", type=int, choices=(0, 1), default=1)
    parser.add_argument("--largemode", type=int, choices=(0, 1), default=0)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--out", default="/home/radxa/qgimbal_color_probe")
    parser.add_argument("--gst-convert", action="store_true", help="let GStreamer videoconvert output BGR")
    args = parser.parse_args()

    width, height = parse_size(args.size)
    set_sensor_format(args.subdev, width, height)
    pipeline = (
        f"v4l2src device={args.device} en-awisp={args.awisp} en-largemode={args.largemode} ! "
        f"video/x-raw,format=NV12,width={width},height={height},framerate={args.fps}/1 ! "
    )
    if args.gst_convert:
        pipeline += "videoconvert ! video/x-raw,format=BGR ! "
    pipeline += "queue max-size-buffers=1 leaky=downstream ! appsink drop=true max-buffers=1 sync=false"
    print(f"Using GStreamer pipeline:\n{pipeline}", flush=True)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    print(f"opened={cap.isOpened()}", flush=True)
    if not cap.isOpened():
        return 2

    frame = None
    for index in range(30):
        ok, candidate = cap.read()
        print(f"read {index}: ok={ok} shape={None if not ok else candidate.shape}", flush=True)
        if ok:
            frame = candidate
        time.sleep(0.03)
    cap.release()
    if frame is None:
        return 3

    os.makedirs(args.out, exist_ok=True)
    if args.gst_convert:
        image = frame
        if args.scale != 1.0:
            out_width = max(2, int(round(width * args.scale)))
            out_height = max(2, int(round(height * args.scale)))
            image = cv2.resize(image, (out_width, out_height), interpolation=cv2.INTER_AREA)
        path = os.path.join(args.out, "gst_videoconvert_bgr.jpg")
        cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        pixels = image.reshape(-1, 3)
        print("gst_videoconvert_bgr", "channel_mean", pixels.mean(axis=0), "channel_std", pixels.std(axis=0), flush=True)
        print("Wrote:", flush=True)
        print(path, flush=True)
        return 0

    nv12_height = height + height // 2
    nv12 = frame[:nv12_height, :width]
    y_plane = nv12[:height, :width]
    uv_plane = nv12[height:nv12_height, :width]

    print(
        "Y mean/std/min/max",
        float(y_plane.mean()),
        float(y_plane.std()),
        int(y_plane.min()),
        int(y_plane.max()),
        flush=True,
    )
    print(
        "UV mean/std/min/max",
        float(uv_plane.mean()),
        float(uv_plane.std()),
        int(uv_plane.min()),
        int(uv_plane.max()),
        flush=True,
    )
    print("U mean/std", float(uv_plane[:, 0::2].mean()), float(uv_plane[:, 0::2].std()), flush=True)
    print("V mean/std", float(uv_plane[:, 1::2].mean()), float(uv_plane[:, 1::2].std()), flush=True)

    out_width = max(2, int(round(width * args.scale)))
    out_height = max(2, int(round(height * args.scale)))
    out_width -= out_width % 2
    out_height -= out_height % 2
    y_small = cv2.resize(y_plane, (out_width, out_height), interpolation=cv2.INTER_AREA)
    uv_small = cv2.resize(uv_plane, (out_width, out_height // 2), interpolation=cv2.INTER_AREA)
    small = np.vstack([y_small, uv_small])

    variants = {
        "nv12_bgr": cv2.cvtColor(small, cv2.COLOR_YUV2BGR_NV12),
        "nv21_bgr": cv2.cvtColor(small, cv2.COLOR_YUV2BGR_NV21),
        "nv12_rgb_as_bgr": cv2.cvtColor(small, cv2.COLOR_YUV2RGB_NV12),
        "nv21_rgb_as_bgr": cv2.cvtColor(small, cv2.COLOR_YUV2RGB_NV21),
        "gray_y": y_small,
    }

    for name, image in variants.items():
        path = os.path.join(args.out, f"{name}.jpg")
        cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if image.ndim == 3:
            pixels = image.reshape(-1, 3)
            print(name, "channel_mean", pixels.mean(axis=0), "channel_std", pixels.std(axis=0), flush=True)
        else:
            print(name, "gray_mean", float(image.mean()), "gray_std", float(image.std()), flush=True)

    print("Wrote:", flush=True)
    for name in variants:
        print(os.path.join(args.out, f"{name}.jpg"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
