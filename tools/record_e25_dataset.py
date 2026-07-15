#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2


def build_pipeline(device: str, width: int, height: int, fps: int) -> str:
    return (
        f"v4l2src device={device} en-awisp=1 en-largemode=0 ! "
        f"video/x-raw,format=NV12,width=1920,height=1080,framerate={fps}/1 ! "
        f"videoscale ! video/x-raw,format=NV12,width={width},height={height} ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Record an E25 IMX415 replay dataset")
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(
        build_pipeline(args.device, args.width, args.height, args.fps),
        cv2.CAP_GSTREAMER,
    )
    if not capture.isOpened():
        raise SystemExit("failed to open IMX415 GStreamer pipeline")
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(args.fps),
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise SystemExit("failed to open MJPEG video writer")

    started = time.monotonic()
    timestamps = []
    frames = 0
    try:
        while time.monotonic() - started < args.seconds:
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            writer.write(frame)
            timestamps.append(time.monotonic() - started)
            frames += 1
    finally:
        capture.release()
        writer.release()

    metadata = {
        "video": args.output.name,
        "label": args.label,
        "device": args.device,
        "width": args.width,
        "height": args.height,
        "requested_fps": args.fps,
        "frames": frames,
        "duration_s": timestamps[-1] if timestamps else 0.0,
        "timestamps_s": timestamps,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in metadata.items() if key != "timestamps_s"}))


if __name__ == "__main__":
    main()
