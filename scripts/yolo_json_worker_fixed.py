#!/usr/bin/env python3
"""JSONL YOLO worker that returns one fixed detection.

This is an integration-test worker for the NPU branch. It proves that YOLO
results can flow into `main.py` without requiring a real NBG model yet.
Coordinates can be provided as ratios of input width/height.
"""

from __future__ import annotations

import argparse
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", default="0.42,0.35,0.58,0.65", help="x1,y1,x2,y2 ratios by default")
    parser.add_argument("--pixels", action="store_true", help="treat --bbox as pixel coordinates")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--label", default="target")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bbox_values = [float(part) for part in args.bbox.split(",")]
    if len(bbox_values) != 4:
        raise SystemExit("--bbox must contain four comma-separated numbers")

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            width = float(payload.get("width", 1))
            height = float(payload.get("height", 1))
            if args.pixels:
                bbox = bbox_values
            else:
                x1, y1, x2, y2 = bbox_values
                bbox = [x1 * width, y1 * height, x2 * width, y2 * height]
            response = {
                "frame_id": payload.get("frame_id"),
                "detections": [
                    {
                        "bbox": bbox,
                        "confidence": args.confidence,
                        "label": args.label,
                    }
                ],
            }
        except Exception as exc:
            response = {"frame_id": None, "detections": [], "error": str(exc)}
        print(json.dumps(response), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
