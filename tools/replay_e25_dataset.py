#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vision.a4_target import detection_as_rect, find_black_band_candidates, merge_candidates
from vision.competition_vision import TrackedTarget
from vision.e25_laser import E25LaserTracker
from vision.e25_pipeline import E25VisionPipeline
from vision.rect_detect import detect_rectangles_multi_pass


def draw_result(frame, result, laser) -> None:
    if result.detection is not None:
        color = (0, 255, 0) if result.current else (0, 255, 255)
        cv2.polylines(frame, [result.detection.quad.astype("int32")], True, color, 2)
        for x, y, score in result.detection.edge_points:
            point_color = (0, 255, 0) if score >= 0.45 else ((0, 215, 255) if score >= 0.28 else (0, 0, 255))
            cv2.circle(frame, (int(round(x)), int(round(y))), 2, point_color, -1)
    if laser is not None:
        x, y, width, height = laser.bbox
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)
        cv2.drawMarker(
            frame,
            (int(round(laser.center[0])), int(round(laser.center[1]))),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            14,
            2,
        )
    confidence = result.confidence
    cv2.putText(
        frame,
        f"{result.state.value} id={confidence.identity:.2f} meas={confidence.measurement:.2f} "
        f"track={confidence.tracking:.2f} ctl={int(confidence.control_valid)}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay an E25 dataset through the model tracker")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise SystemExit(f"failed to open {args.input}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if args.output is not None:
        writer = cv2.VideoWriter(
            str(args.output),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (width, height),
        )

    pipeline = E25VisionPipeline(require_red_rings=False)
    laser_tracker = E25LaserTracker()
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame_index += 1
            detection_cycle = pipeline.needs_global_detection(frame_index)
            if detection_cycle:
                base = detect_rectangles_multi_pass(frame, min_area_ratio=0.015, max_area_ratio=0.70)
                black = find_black_band_candidates(frame, pipeline.detector.config)
                candidates = merge_candidates(base, black, limit=12)
            else:
                candidates = []
            result = pipeline.update(frame, candidates, detection_cycle, dt=1.0 / fps)
            target = None
            if result.detection is not None:
                target = TrackedTarget(
                    rect=detection_as_rect(result.detection),
                    current=result.current,
                    held=not result.current,
                    miss_count=result.miss_count,
                )
            laser = laser_tracker.update(frame, target, enabled=target is not None and target.current)
            draw_result(frame, result, laser)
            if writer is not None:
                writer.write(frame)
            if args.show:
                cv2.imshow("E25 replay", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
