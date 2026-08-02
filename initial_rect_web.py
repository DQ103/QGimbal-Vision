import argparse
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from main import detect_rectangles, draw_detected_rect


class Preview:
    def __init__(self, port: int) -> None:
        self.port = port
        self.condition = threading.Condition()
        self.jpeg = None
        self.frame_id = 0

    def start(self) -> None:
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format, *_args):
                return

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = (
                        "<!doctype html><html><head><meta charset='utf-8'>"
                        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                        "<title>QGimbal initial rectangle detector</title>"
                        "<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif}"
                        "header{padding:10px 14px}img{display:block;width:100%;height:auto}</style>"
                        "</head><body><header>QGimbal 72db51d - initial rectangle detector</header>"
                        "<img src='/stream.mjpg'></body></html>"
                    ).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path == "/snapshot.jpg":
                    with preview.condition:
                        jpeg = preview.jpeg
                    if jpeg is None:
                        self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
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
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last_frame_id = -1
                try:
                    while True:
                        with preview.condition:
                            preview.condition.wait_for(lambda: preview.frame_id != last_frame_id)
                            jpeg = preview.jpeg
                            last_frame_id = preview.frame_id
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return

        server = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()

    def update(self, frame) -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            return
        with self.condition:
            self.jpeg = encoded.tobytes()
            self.frame_id += 1
            self.condition.notify_all()


def build_pipeline(device: str) -> str:
    return (
        f"v4l2src device={device} en-awisp=1 en-largemode=0 ! "
        "video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 ! "
        "videoscale ! video/x-raw,format=NV12,width=960,height=540 ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--flip", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()

    capture = cv2.VideoCapture(build_pipeline(args.device), cv2.CAP_GSTREAMER)
    if not capture.isOpened():
        raise SystemExit("failed to open IMX415 GStreamer pipeline")

    preview = Preview(args.port)
    preview.start()
    print(f"Initial QGimbal preview: http://<board-ip>:{args.port}/", flush=True)

    window_started = time.monotonic()
    window_frames = 0
    fps = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            if args.flip:
                frame = cv2.flip(frame, -1)

            rects = detect_rectangles(
                frame,
                min_area_ratio=0.005,
                max_area_ratio=0.5,
                angle_tol=25.0,
            )
            best = rects[0] if rects else None
            draw_detected_rect(frame, best)

            window_frames += 1
            elapsed = time.monotonic() - window_started
            if elapsed >= 1.0:
                fps = window_frames / elapsed
                window_frames = 0
                window_started = time.monotonic()
                if best is None:
                    print(f"fps={fps:.1f} rect=none", flush=True)
                else:
                    cx, cy = best["center"]
                    print(
                        f"fps={fps:.1f} cx={cx:.1f} cy={cy:.1f} area={best['area']:.0f}",
                        flush=True,
                    )
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            preview.update(frame)
    finally:
        capture.release()


if __name__ == "__main__":
    main()
