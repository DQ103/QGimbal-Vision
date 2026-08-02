#!/usr/bin/env python3
"""Camera MJPEG demo that periodically runs the Allwinner YOLOv8 NPU CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlparse

import cv2


DETECTION_RE = re.compile(
    r"^\s*(?P<class_id>\d+):\s*"
    r"(?P<percent>\d+(?:\.\d+)?)%\s*,\s*"
    r"\[\s*(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y1>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<x2>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y2>-?\d+(?:\.\d+)?)\s*\]\s*,\s*"
    r"(?P<label>[A-Za-z0-9_ -]+)\s*$"
)


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    class_id: int


@dataclass(frozen=True)
class NpuResult:
    detections: list[Detection]
    wall_time_ms: float
    npu_time_us: int | None
    frame_id: int
    timestamp: float
    error: str = ""


class NpuWorker:
    def __init__(
        self,
        demo: Path,
        model: Path,
        runtime_dir: Path,
        timeout_s: float,
        jpeg_quality: int,
        min_confidence: float,
    ) -> None:
        self.demo = demo
        self.model = model
        self.runtime_dir = runtime_dir
        self.timeout_s = timeout_s
        self.jpeg_quality = jpeg_quality
        self.min_confidence = min_confidence
        self._condition = threading.Condition()
        self._pending_frame = None
        self._pending_frame_id = 0
        self._latest: NpuResult | None = None
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame) -> int:
        with self._condition:
            self._pending_frame = frame.copy()
            self._pending_frame_id += 1
            frame_id = self._pending_frame_id
            self._condition.notify()
            return frame_id

    def latest(self) -> NpuResult | None:
        with self._condition:
            return self._latest

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        last_seen_id = 0
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stopped or self._pending_frame_id != last_seen_id
                )
                if self._stopped:
                    return
                frame = self._pending_frame.copy()
                frame_id = self._pending_frame_id
                last_seen_id = frame_id

            result = self._run_once(frame, frame_id)
            with self._condition:
                self._latest = result

    def _run_once(self, frame, frame_id: int) -> NpuResult:
        suffix = ".jpg"
        tmp_path = None
        start = time.monotonic()
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = Path(tmp.name)
            ok = cv2.imwrite(
                str(tmp_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
            )
            if not ok:
                raise RuntimeError("failed to encode temporary frame")

            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = f"{self.runtime_dir}:{env.get('LD_LIBRARY_PATH', '')}"
            proc = subprocess.run(
                [str(self.demo), "-nb", str(self.model), "-i", str(tmp_path)],
                cwd=str(self.demo.parent),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_s,
                check=False,
            )
            text = "\n".join(part for part in (proc.stdout.strip(), proc.stderr.strip()) if part)
            if proc.returncode != 0:
                raise RuntimeError(text or f"demo returned {proc.returncode}")

            detections = [
                det for det in parse_detections(text) if det.confidence >= self.min_confidence
            ]
            wall_time_ms = (time.monotonic() - start) * 1000.0
            return NpuResult(
                detections=detections,
                wall_time_ms=wall_time_ms,
                npu_time_us=parse_npu_time_us(text),
                frame_id=frame_id,
                timestamp=time.time(),
            )
        except Exception as exc:
            wall_time_ms = (time.monotonic() - start) * 1000.0
            return NpuResult(
                detections=[],
                wall_time_ms=wall_time_ms,
                npu_time_us=None,
                frame_id=frame_id,
                timestamp=time.time(),
                error=str(exc),
            )
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass


class StreamState:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.jpeg: bytes | None = None
        self.frame_id = 0
        self.status: dict[str, object] = {}

    def update(self, jpeg: bytes, status: dict[str, object]) -> None:
        with self.condition:
            self.jpeg = jpeg
            self.status = status
            self.frame_id += 1
            self.condition.notify_all()

    def snapshot(self) -> tuple[bytes | None, dict[str, object]]:
        with self.condition:
            return self.jpeg, dict(self.status)


def parse_size(text: str) -> tuple[int, int]:
    try:
        width_text, height_text = text.lower().split("x", 1)
        width = int(width_text)
        height = int(height_text)
    except ValueError as exc:
        raise SystemExit(f"invalid size {text!r}, expected WxH") from exc
    if width <= 0 or height <= 0:
        raise SystemExit(f"invalid size {text!r}")
    return width, height


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


def build_pipeline(
    device: str,
    width: int,
    height: int,
    fps: int,
    awisp: int,
    largemode: int,
) -> str:
    return (
        f"v4l2src device={device} en-awisp={int(awisp)} en-largemode={int(largemode)} ! "
        f"video/x-raw,format=NV12,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def parse_detections(text: str) -> list[Detection]:
    detections: list[Detection] = []
    for line in text.splitlines():
        match = DETECTION_RE.match(line)
        if not match:
            continue
        data = match.groupdict()
        detections.append(
            Detection(
                label=data["label"].strip(),
                confidence=float(data["percent"]) / 100.0,
                bbox=(
                    float(data["x1"]),
                    float(data["y1"]),
                    float(data["x2"]),
                    float(data["y2"]),
                ),
                class_id=int(data["class_id"]),
            )
        )
    return detections


def parse_npu_time_us(text: str) -> int | None:
    match = re.search(r"run time for this network\s+\d+:\s+(\d+)\s+us", text)
    if not match:
        return None
    return int(match.group(1))


def draw_detections(frame, detections: list[Detection]) -> None:
    colors = [(40, 220, 40), (40, 160, 255), (255, 180, 40), (220, 80, 220)]
    for index, det in enumerate(detections):
        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
        color = colors[index % len(colors)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det.label} {det.confidence * 100:.0f}%"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        label_y = max(0, y1 - text_h - baseline - 6)
        cv2.rectangle(frame, (x1, label_y), (x1 + text_w + 10, label_y + text_h + baseline + 6), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 5, label_y + text_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )


def draw_hud(frame, camera_fps: float, result: NpuResult | None) -> None:
    if result is None:
        text = f"camera {camera_fps:.1f} fps | NPU waiting"
    elif result.error:
        text = f"camera {camera_fps:.1f} fps | NPU error"
    else:
        npu_text = f"{result.npu_time_us / 1000.0:.1f} ms" if result.npu_time_us else "n/a"
        text = (
            f"camera {camera_fps:.1f} fps | "
            f"NPU {npu_text}, wall {result.wall_time_ms:.0f} ms | "
            f"{len(result.detections)} detections"
        )
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 2, cv2.LINE_AA)


def build_status(camera_fps: float, result: NpuResult | None) -> dict[str, object]:
    payload: dict[str, object] = {"camera_fps": camera_fps, "npu": None}
    if result is not None:
        payload["npu"] = {
            "frame_id": result.frame_id,
            "wall_time_ms": result.wall_time_ms,
            "npu_time_us": result.npu_time_us,
            "age_ms": (time.time() - result.timestamp) * 1000.0,
            "error": result.error,
            "detections": [
                {
                    "label": det.label,
                    "confidence": det.confidence,
                    "bbox": det.bbox,
                    "class_id": det.class_id,
                }
                for det in result.detections
            ],
        }
    return payload


def build_page() -> bytes:
    return b"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>A7A NPU Camera YOLOv8</title>
<style>
body{margin:0;background:#0f1115;color:#e8e8e8;font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{height:100vh;display:grid;grid-template-columns:minmax(0,1fr) 320px}
.stage{display:flex;align-items:center;justify-content:center;min-width:0;background:#090b0f}
img{max-width:100%;max-height:100vh;display:block}
aside{box-sizing:border-box;padding:14px;background:#171a21;border-left:1px solid #2a2f3a;overflow:auto}
h1{font-size:18px;margin:0 0 12px}
.kv{display:grid;grid-template-columns:110px 1fr;gap:6px 10px;font-size:13px;margin-bottom:12px}
.k{color:#aab2c0}.v{font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:6px 4px;border-bottom:1px solid #2a2f3a}
th{color:#aab2c0}
pre{white-space:pre-wrap;color:#aab2c0;font-size:12px}
@media(max-width:900px){main{grid-template-columns:1fr;grid-template-rows:minmax(0,1fr) auto}aside{border-left:0;border-top:1px solid #2a2f3a}}
</style>
</head>
<body>
<main>
<section class="stage"><img src="/stream.mjpg"></section>
<aside>
<h1>NPU YOLOv8 Camera</h1>
<div class="kv">
<div class="k">camera</div><div id="camera" class="v">-</div>
<div class="k">NPU runtime</div><div id="npu" class="v">-</div>
<div class="k">wall time</div><div id="wall" class="v">-</div>
<div class="k">detections</div><div id="count" class="v">-</div>
</div>
<table><thead><tr><th>label</th><th>conf</th><th>box</th></tr></thead><tbody id="rows"></tbody></table>
<pre id="error"></pre>
</aside>
</main>
<script>
async function refresh(){
  try{
    const data = await fetch('/api/status', {cache:'no-store'}).then(r=>r.json());
    document.getElementById('camera').textContent = `${Number(data.camera_fps||0).toFixed(1)} fps`;
    const npu = data.npu || {};
    document.getElementById('npu').textContent = npu.npu_time_us ? `${(npu.npu_time_us/1000).toFixed(1)} ms` : '-';
    document.getElementById('wall').textContent = npu.wall_time_ms ? `${Number(npu.wall_time_ms).toFixed(0)} ms` : '-';
    const dets = npu.detections || [];
    document.getElementById('count').textContent = dets.length;
    document.getElementById('rows').innerHTML = dets.map(d => `<tr><td>${d.label}</td><td>${(d.confidence*100).toFixed(0)}%</td><td>${d.bbox.map(v=>Number(v).toFixed(0)).join(', ')}</td></tr>`).join('');
    document.getElementById('error').textContent = npu.error || '';
  }catch(e){
    document.getElementById('error').textContent = String(e);
  }
}
setInterval(refresh, 500);
refresh();
</script>
</body>
</html>"""


def make_handler(state: StreamState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                body = build_page()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/api/status":
                _, status = state.snapshot()
                body = json.dumps(status).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/snapshot.jpg":
                jpeg, _ = state.snapshot()
                if jpeg is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "No frame yet")
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                return

            if parsed.path != "/stream.mjpg":
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--subdev", default="/dev/v4l-subdev0")
    parser.add_argument("--size", default="1920x1080")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--awisp", type=int, choices=(0, 1), default=1)
    parser.add_argument("--largemode", type=int, choices=(0, 1), default=0)
    parser.add_argument("--preview-size", default="960x540")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--quality", type=int, default=75)
    parser.add_argument("--infer-interval", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--npu-timeout", type=float, default=3.0)
    parser.add_argument("--npu-jpeg-quality", type=int, default=80)
    parser.add_argument(
        "--demo",
        default="/home/radxa/awnpu_model_zoo/examples/yolov8/build_native/yolov8_demo_a733",
    )
    parser.add_argument(
        "--model",
        default="/home/radxa/awnpu_model_zoo/examples/yolov8/model/yolov8n_6_uint8_a733.nb",
    )
    parser.add_argument(
        "--runtime-dir",
        default="/home/radxa/awnpu_model_zoo/common/npuruntime/lib_linux_aarch64/A733",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    width, height = parse_size(args.size)
    preview_w, preview_h = parse_size(args.preview_size)
    if args.infer_interval <= 0.0:
        raise SystemExit("--infer-interval must be positive")

    set_sensor_format(args.subdev, width, height)
    pipeline = build_pipeline(args.device, width, height, args.fps, args.awisp, args.largemode)
    print(f"Using GStreamer pipeline:\n{pipeline}", flush=True)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("failed to open camera pipeline")
        return 2

    worker = NpuWorker(
        demo=Path(args.demo),
        model=Path(args.model),
        runtime_dir=Path(args.runtime_dir),
        timeout_s=float(args.npu_timeout),
        jpeg_quality=int(args.npu_jpeg_quality),
        min_confidence=float(args.min_confidence),
    )
    state = StreamState()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"NPU camera demo: http://<board-ip>:{server.server_port}/", flush=True)

    frame_count = 0
    fps_start = time.monotonic()
    camera_fps = 0.0
    last_submit = 0.0
    last_print = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            display = cv2.resize(frame, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
            now = time.monotonic()
            if now - last_submit >= args.infer_interval:
                worker.submit(display)
                last_submit = now

            result = worker.latest()
            if result is not None:
                draw_detections(display, result.detections)
            draw_hud(display, camera_fps, result)

            ok, encoded = cv2.imencode(
                ".jpg",
                display,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(args.quality)],
            )
            if not ok:
                continue

            frame_count += 1
            elapsed = now - fps_start
            if elapsed >= 1.0:
                camera_fps = frame_count / elapsed
                frame_count = 0
                fps_start = now

            state.update(encoded.tobytes(), build_status(camera_fps, result))
            if now - last_print >= 1.0:
                npu_text = "waiting"
                if result is not None:
                    if result.error:
                        npu_text = f"error={result.error}"
                    else:
                        npu_ms = result.npu_time_us / 1000.0 if result.npu_time_us else 0.0
                        npu_text = (
                            f"npu={npu_ms:.1f}ms wall={result.wall_time_ms:.0f}ms "
                            f"det={len(result.detections)}"
                        )
                print(f"camera_fps={camera_fps:.1f} {npu_text}", flush=True)
                last_print = now
    finally:
        worker.stop()
        cap.release()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
