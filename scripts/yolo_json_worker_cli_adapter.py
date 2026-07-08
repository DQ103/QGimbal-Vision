#!/usr/bin/env python3
"""Adapt an external NPU/YOLO CLI program to the JSONL worker protocol.

The command is run once per frame. Use this for early integration with vendor
demos. For production, prefer a persistent C++ worker to avoid process startup
cost and repeated model loading.

Expected external stdout:
  {"detections":[{"bbox":[x1,y1,x2,y2],"confidence":0.9,"label":"target"}]}
or:
  [{"bbox":[x1,y1,x2,y2],"confidence":0.9,"label":"target"}]

With --parser allwinner-yolo it also accepts Allwinner/Radxa demo output:
  16:  95%, [ 131,  220,  308,  541], dog
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--command",
        required=True,
        help="Command template. Placeholders: {image}, {width}, {height}, {frame_id}",
    )
    parser.add_argument(
        "--parser",
        choices=("auto", "json", "allwinner-yolo", "allwinner-yolov8"),
        default="auto",
        help="external output parser, default auto",
    )
    parser.add_argument("--suffix", default=".jpg", help="temporary image suffix, default .jpg")
    parser.add_argument("--timeout", type=float, default=1.0)
    return parser.parse_args()


ALLWINNER_DETECTION_RE = re.compile(
    r"^\s*(?P<class_id>\d+):\s*"
    r"(?P<percent>\d+(?:\.\d+)?)%\s*,\s*"
    r"\[\s*(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y1>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<x2>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y2>-?\d+(?:\.\d+)?)\s*\]\s*,\s*"
    r"(?P<label>[A-Za-z0-9_ -]+)\s*$"
)


def parse_json_response(stdout: str):
    text = stdout.strip()
    if not text:
        return []
    payload = json.loads(text.splitlines()[-1])
    if isinstance(payload, list):
        return payload
    return payload.get("detections", [])


def parse_allwinner_yolov8_response(text: str):
    detections = []
    for line in text.splitlines():
        match = ALLWINNER_DETECTION_RE.match(line)
        if not match:
            continue
        data = match.groupdict()
        detections.append(
            {
                "bbox": [
                    float(data["x1"]),
                    float(data["y1"]),
                    float(data["x2"]),
                    float(data["y2"]),
                ],
                "confidence": float(data["percent"]) / 100.0,
                "label": data["label"].strip(),
                "class_id": int(data["class_id"]),
            }
        )
    return detections


def normalize_response(stdout: str, stderr: str, frame_id, parser: str):
    text = "\n".join(part for part in (stdout.strip(), stderr.strip()) if part)
    if not text:
        return {"frame_id": frame_id, "detections": []}

    detections = []
    if parser in ("auto", "json"):
        try:
            detections = parse_json_response(stdout)
        except json.JSONDecodeError:
            if parser == "json":
                raise

    if not detections and parser in ("auto", "allwinner-yolo", "allwinner-yolov8"):
        detections = parse_allwinner_yolov8_response(text)

    return {"frame_id": frame_id, "detections": detections}


def run_command(command_template: str, image_path: Path, width: int, height: int, frame_id, timeout: float):
    command = command_template.format(
        image=str(image_path),
        width=int(width),
        height=int(height),
        frame_id=frame_id,
    )
    return subprocess.run(
        shlex.split(command),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def main() -> int:
    args = parse_args()
    for line in sys.stdin:
        if not line.strip():
            continue
        frame_id = None
        try:
            request = json.loads(line)
            frame_id = request.get("frame_id")
            if request.get("format") != "jpg_b64":
                raise ValueError(f"unsupported format: {request.get('format')}")
            width = int(request.get("width", 0))
            height = int(request.get("height", 0))
            image_bytes = base64.b64decode(request["image"])
            with tempfile.NamedTemporaryFile(suffix=args.suffix, delete=True) as tmp:
                tmp.write(image_bytes)
                tmp.flush()
                proc = run_command(args.command, Path(tmp.name), width, height, frame_id, args.timeout)
            if proc.returncode != 0:
                response = {"frame_id": frame_id, "detections": [], "error": proc.stderr.strip()}
            else:
                response = normalize_response(proc.stdout, proc.stderr, frame_id, args.parser)
        except Exception as exc:
            response = {"frame_id": frame_id, "detections": [], "error": str(exc)}
        print(json.dumps(response), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
