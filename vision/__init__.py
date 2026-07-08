"""Vision modules for this project.

This package contains small, reusable computer vision utilities used by `main.py`.
"""

from .rect_detect import (
    DETECT_PASSES,
    DEFAULT_SELECTION_CONFIG,
    DetectedRect,
    DetectionPass,
    RectSelectionConfig,
    RectSelector,
    detect_rectangles,
    detect_rectangles_multi_pass,
    draw_detected_rect,
    pick_best_rect,
)

__all__ = [
    "DETECT_PASSES",
    "DEFAULT_SELECTION_CONFIG",
    "DetectedRect",
    "DetectionPass",
    "RectSelectionConfig",
    "RectSelector",
    "detect_rectangles",
    "detect_rectangles_multi_pass",
    "draw_detected_rect",
    "pick_best_rect",
]
