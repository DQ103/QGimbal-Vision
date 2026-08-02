#!/usr/bin/env python3
"""Minimal A7A camera MJPEG preview.

This does not run detection, draw overlays, or apply color adjustment. The only
image conversion is NV12 to BGR/JPEG so a browser can display the stream.
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import subprocess
import sys
import threading
import time

import cv2


def parse_size(text: str) -> tuple[int, int]:
    try:
        width_text, height_text = text.lower().split("x", 1)
        width = int(width_text)
        height = int(height_text)
    except ValueError as exc:
        raise SystemExit(f"invalid --size {text!r}, expected WxH") from exc
    if width <= 0 or height <= 0:
        raise SystemExit(f"invalid --size {text!r}")
    return width, height


def set_sensor_format(subdev: str, width: int, height: int) -> None:
    cmd = [
        "v4l2-ctl",
        "-d",
        subdev,
        "--set-subdev-fmt",
        f"pad=0,code=0x300f,width={width},height={height}",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def build_pipeline(device: str, width: int, height: int, fps: int, awisp: int, largemode: int) -> str:
    return (
        f"v4l2src device={device} en-awisp={int(awisp)} en-largemode={int(largemode)} ! "
        f"video/x-raw,format=NV12,width={width},height={height},framerate={fps}/1 ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


class StreamState:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.jpeg: bytes | None = None
        self.frame_id = 0
        self.fps = 0.0

    def update(self, jpeg: bytes, fps: float) -> None:
        with self.condition:
            self.jpeg = jpeg
            self.fps = fps
            self.frame_id += 1
            self.condition.notify_all()


def make_handler(state: StreamState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = (
                    "<!doctype html><html><head><meta charset='utf-8'>"
                    "<title>Raw Camera Preview</title>"
                    "<style>body{margin:0;background:#111;color:#ddd;font-family:sans-serif}"
                    "main{height:100vh;display:flex;align-items:center;justify-content:center}"
                    "img{max-width:100%;max-height:100vh;display:block}"
                    ".hud{position:fixed;left:10px;top:8px;font-size:13px;color:#ddd;"
                    "background:rgba(0,0,0,.45);padding:4px 6px}</style></head>"
                    "<body><div class='hud'>raw preview</div>"
                    "<main><img src='/stream.mjpg'></main></body></html>"
                ).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/snapshot.jpg":
                with state.condition:
                    jpeg = state.jpeg
                if jpeg is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "No frame yet")
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                return

            if self.path != "/stream.mjpg":
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            self.send_response(HTTPStatus.OK)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            last_frame_id = -1
            try:
                while True:
                    with state.condition:
                        state.condition.wait_for(lambda: state.frame_id != last_frame_id)
                        jpeg = state.jpeg
                        last_frame_id = state.frame_id
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return

    return Handler


def parse_args():
    parser = argparse.ArgumentParser(description="A7A raw MJPEG camera preview")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--subdev", default="/dev/v4l-subdev0")
    parser.add_argument("--size", default="1920x1080")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--awisp", type=int, choices=(0, 1), default=0)
    parser.add_argument("--largemode", type=int, choices=(0, 1), default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--quality", type=int, default=75)
    parser.add_argument("--mode", choices=("color", "gray"), default="color")
    parser.add_argument("--gst-convert", action="store_true", help="let GStreamer videoconvert output BGR")
    parser.add_argument("--print-interval", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    width, height = parse_size(args.size)
    if not 0.0 < args.scale <= 1.0:
        raise SystemExit("--scale must be in (0, 1]")
    if not 1 <= args.quality <= 100:
        raise SystemExit("--quality must be in [1, 100]")

    set_sensor_format(args.subdev, width, height)
    if args.gst_convert:
        pipeline = (
            f"v4l2src device={args.device} en-awisp={int(args.awisp)} en-largemode={int(args.largemode)} ! "
            f"video/x-raw,format=NV12,width={width},height={height},framerate={args.fps}/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "queue max-size-buffers=1 leaky=downstream ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
    else:
        pipeline = build_pipeline(args.device, width, height, args.fps, args.awisp, args.largemode)
    print(f"Using GStreamer pipeline:\n{pipeline}", flush=True)

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("failed to open camera pipeline", file=sys.stderr)
        return 2

    state = StreamState()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Raw preview: http://<board-ip>:{server.server_port}/", flush=True)

    frame_count = 0
    fps_start = time.monotonic()
    fps = 0.0
    last_print = fps_start
    nv12_height = height + height // 2
    output_size = (max(2, int(width * args.scale)), max(2, int(height * args.scale)))
    output_size = (output_size[0] - output_size[0] % 2, output_size[1] - output_size[1] % 2)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if args.gst_convert:
                image = frame
                if args.scale != 1.0:
                    image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
            elif args.mode == "gray":
                raw = frame[:nv12_height, :width]
                image = raw[:height, :width]
                if args.scale != 1.0:
                    image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
            else:
                raw = frame[:nv12_height, :width]
                if args.scale == 1.0:
                    nv12 = raw
                else:
                    y_plane = raw[:height, :width]
                    uv_plane = raw[height:nv12_height, :width]
                    y_small = cv2.resize(y_plane, output_size, interpolation=cv2.INTER_AREA)
                    uv_small = cv2.resize(
                        uv_plane,
                        (output_size[0], output_size[1] // 2),
                        interpolation=cv2.INTER_AREA,
                    )
                    nv12 = cv2.vconcat([y_small, uv_small])
                image = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)

            ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
            if not ok:
                continue

            frame_count += 1
            now = time.monotonic()
            elapsed = now - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = now
            state.update(encoded.tobytes(), fps)
            if args.print_interval >= 0 and now - last_print >= args.print_interval:
                print(f"fps={fps:.1f}", flush=True)
                last_print = now
    finally:
        cap.release()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
