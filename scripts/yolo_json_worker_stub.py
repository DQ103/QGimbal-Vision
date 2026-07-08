#!/usr/bin/env python3
"""Minimal JSONL YOLO worker stub.

It accepts the same stdin protocol expected by `vision.yolo_detect` and returns
no detections. Use it to verify `--detector yolo/hybrid` process plumbing before
replacing it with an NPU/VIPLite worker.
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"frame_id": None, "detections": [], "error": "invalid_json"}), flush=True)
            continue

        print(json.dumps({"frame_id": payload.get("frame_id"), "detections": []}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
