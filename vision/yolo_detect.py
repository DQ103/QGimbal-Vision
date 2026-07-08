from __future__ import annotations

import base64
import json
import select
import shlex
import subprocess
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set

import cv2
import numpy as np

from .rect_detect import DetectedRect


@dataclass(frozen=True)
class YoloBox:
    """One YOLO-style axis-aligned detection in image coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str = ""


def parse_yolo_response(line: str, min_confidence: float = 0.25, labels: Optional[Set[str]] = None) -> List[YoloBox]:
    payload = json.loads(line)
    boxes = []
    for item in payload.get("detections", []):
        label = str(item.get("label", ""))
        if labels and label not in labels:
            continue

        confidence = float(item.get("confidence", item.get("score", 0.0)))
        if confidence < min_confidence:
            continue

        if "bbox" in item:
            x1, y1, x2, y2 = item["bbox"]
        else:
            x1 = item["x1"]
            y1 = item["y1"]
            x2 = item["x2"]
            y2 = item["y2"]

        boxes.append(YoloBox(float(x1), float(y1), float(x2), float(y2), confidence, label))
    return boxes


def yolo_boxes_to_rects(boxes: Iterable[YoloBox], pass_index: int = 100) -> List[DetectedRect]:
    rects = []
    for box in boxes:
        x1 = min(box.x1, box.x2)
        y1 = min(box.y1, box.y2)
        x2 = max(box.x1, box.x2)
        y2 = max(box.y1, box.y2)
        width = x2 - x1
        height = y2 - y1
        if width <= 0.0 or height <= 0.0:
            continue
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        rects.append(
            DetectedRect(
                center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                box=pts,
                area=width * height,
                pass_index=pass_index,
                score=float(box.confidence),
            )
        )
    rects.sort(key=lambda r: r.score, reverse=True)
    return rects


class YoloSubprocessDetector:
    """JSON-lines bridge to an external YOLO/NPU worker.

    Protocol:
      stdin:  {"frame_id":N,"width":W,"height":H,"format":"jpg_b64","image":"..."}
      stdout: {"frame_id":N,"detections":[{"bbox":[x1,y1,x2,y2],"confidence":0.9,"label":"target"}]}

    The external process can be a C++ VIPLite/NPU program or a Python stub.
    """

    def __init__(
        self,
        command: Sequence[str],
        timeout_s: float = 0.05,
        jpeg_quality: int = 70,
        min_confidence: float = 0.25,
        labels: Optional[Set[str]] = None,
    ):
        if not command:
            raise ValueError("YOLO command is required")
        self.command = list(command)
        self.timeout_s = float(timeout_s)
        self.jpeg_quality = int(jpeg_quality)
        self.min_confidence = float(min_confidence)
        self.labels = labels
        self._proc: Optional[subprocess.Popen] = None
        self._frame_id = 0

    @classmethod
    def from_shell_command(
        cls,
        command: str,
        timeout_s: float = 0.05,
        jpeg_quality: int = 70,
        min_confidence: float = 0.25,
        labels: Optional[Set[str]] = None,
    ) -> "YoloSubprocessDetector":
        return cls(shlex.split(command), timeout_s, jpeg_quality, min_confidence, labels)

    def close(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=0.5)
        self._proc = None

    def detect(self, frame_bgr: np.ndarray) -> List[DetectedRect]:
        self._ensure_started()
        assert self._proc is not None
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None

        ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return []

        self._frame_id += 1
        request = {
            "frame_id": self._frame_id,
            "width": int(frame_bgr.shape[1]),
            "height": int(frame_bgr.shape[0]),
            "format": "jpg_b64",
            "image": base64.b64encode(encoded.tobytes()).decode("ascii"),
        }
        try:
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError:
            self.close()
            return []

        if not self._wait_for_stdout():
            return []

        line = self._proc.stdout.readline()
        if not line:
            self.close()
            return []

        boxes = parse_yolo_response(line, self.min_confidence, self.labels)
        return yolo_boxes_to_rects(boxes)

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def _wait_for_stdout(self) -> bool:
        assert self._proc is not None
        assert self._proc.stdout is not None
        readable, _, _ = select.select([self._proc.stdout], [], [], self.timeout_s)
        return bool(readable)
