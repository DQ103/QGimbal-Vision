import cv2
import numpy as np

from vision.rect_detect import (
    DetectedRect,
    RectSelectionConfig,
    RectSelector,
    detect_rectangles_multi_pass,
    pick_best_rect,
)


def make_rect(center, size, area=None):
    cx, cy = center
    w, h = size
    box = np.array(
        [
            [cx - w / 2, cy - h / 2],
            [cx + w / 2, cy - h / 2],
            [cx + w / 2, cy + h / 2],
            [cx - w / 2, cy + h / 2],
        ],
        dtype=np.float32,
    )
    return DetectedRect(center=(float(cx), float(cy)), box=box, area=float(area if area is not None else w * h))


def test_multi_pass_detects_synthetic_rectangle() -> None:
    frame = np.zeros((240, 400), dtype=np.uint8)
    cv2.rectangle(frame, (110, 70), (290, 170), 255, thickness=3)

    rects = detect_rectangles_multi_pass(frame, min_area_ratio=0.001, max_area_ratio=0.8)

    assert rects
    best = rects[0]
    assert abs(best.center[0] - 200.0) < 5.0
    assert abs(best.center[1] - 120.0) < 5.0
    assert best.pass_index >= 0


def test_pick_best_rect_prefers_temporal_continuity_over_area_only() -> None:
    large_far = make_rect((540, 390), (170, 120))
    smaller_near_previous = make_rect((320, 240), (90, 70))
    cfg = RectSelectionConfig(max_area_ratio=0.8, center_weight=0.25, previous_weight=0.70)

    picked = pick_best_rect(
        [large_far, smaller_near_previous],
        frame_w=640,
        frame_h=480,
        last_center=(322.0, 238.0),
        config=cfg,
    )

    assert picked is not None
    assert picked.center == smaller_near_previous.center
    assert picked.score > 0.0


def test_rect_selector_resets_history_on_miss() -> None:
    selector = RectSelector()
    rect = make_rect((320, 240), (100, 80))

    assert selector.update([rect], 640, 480) is not None
    assert selector.last_center == rect.center

    assert selector.update([], 640, 480) is None
    assert selector.last_center is None
