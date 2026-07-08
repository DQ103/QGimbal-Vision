from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectedRect:
    """A detected (possibly rotated) rectangle in image coordinates."""

    center: Tuple[float, float]
    box: np.ndarray  # shape: (4, 2), dtype: float32
    area: float
    pass_index: int = -1
    score: float = 0.0


@dataclass(frozen=True)
class DetectionPass:
    """One rectangle detection attempt with a specific preprocessing profile."""

    canny_th1: int
    canny_th2: int
    epsilon_factor: float
    angle_tol: float
    blur_ksize: int
    use_otsu: bool = False
    morph_ksize: int = 0
    erode_iterations: int = 0


@dataclass(frozen=True)
class RectSelectionConfig:
    """Weights and filters used to pick one target from candidate rectangles."""

    min_area_ratio: float = 0.005
    max_area_ratio: float = 0.5
    max_aspect_ratio: float = 5.0
    center_weight: float = 0.25
    previous_weight: float = 0.70


# First pass keeps the original Otsu+morphology path. Later passes follow the
# K230 script idea: progressively looser Canny/approximation profiles.
DEFAULT_DETECTION_PASS = DetectionPass(
    canny_th1=25,
    canny_th2=75,
    epsilon_factor=0.02,
    angle_tol=25.0,
    blur_ksize=3,
    use_otsu=True,
    morph_ksize=5,
    erode_iterations=1,
)
DETECT_PASSES: Tuple[DetectionPass, ...] = (
    DEFAULT_DETECTION_PASS,
    DetectionPass(80, 150, 0.04, 25.0, 5),
    DetectionPass(60, 140, 0.05, 35.0, 5),
    DetectionPass(40, 120, 0.06, 45.0, 3),
)
DEFAULT_SELECTION_CONFIG = RectSelectionConfig()


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return the angle (in degrees) between 2 vectors.

    If either vector is near-zero, returns 0.0.
    """
    dot = float(v1.dot(v2))
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 * n2 == 0:
        return 0.0
    cos = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos))


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def rect_aspect(rect: DetectedRect) -> float:
    dists = [float(np.linalg.norm(rect.box[i] - rect.box[(i + 1) % 4])) for i in range(4)]
    min_dist = min(dists)
    max_dist = max(dists)
    if min_dist <= 0.0:
        return float("inf")
    return max_dist / min_dist


def _as_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _normalized_blur_ksize(blur_ksize: int) -> int:
    if blur_ksize <= 1:
        return 1
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    return blur_ksize


def _preprocess(gray: np.ndarray, detect_pass: DetectionPass) -> np.ndarray:
    blur_ksize = _normalized_blur_ksize(detect_pass.blur_ksize)
    if blur_ksize > 1:
        work = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    else:
        work = gray

    if not detect_pass.use_otsu:
        return work

    _, work = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if detect_pass.morph_ksize > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (detect_pass.morph_ksize, detect_pass.morph_ksize))
        work = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel, iterations=1)
        if detect_pass.erode_iterations > 0:
            work = cv2.erode(work, kernel, iterations=detect_pass.erode_iterations)
    return work


def _quad_center(pts: np.ndarray) -> Tuple[float, float]:
    # For a perspective quadrilateral, the physical center is better estimated
    # by the intersection of image diagonals than by averaging corners.
    p1, p2, p3, p4 = pts
    x1, y1 = p1
    x2, y2 = p3
    x3, y3 = p2
    x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(float(den)) > 1e-6:
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
        return float(px), float(py)

    center_arr = np.mean(pts, axis=0)
    return float(center_arr[0]), float(center_arr[1])


def _is_rectangle_quad(pts: np.ndarray, angle_tol: float) -> bool:
    angles = []
    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i + 1) % 4]
        p2 = pts[(i + 2) % 4]
        angles.append(angle_between(p0 - p1, p2 - p1))
    return all(abs(a - 90.0) < angle_tol for a in angles)


def _detect_with_pass(
    gray: np.ndarray,
    detect_pass: DetectionPass,
    min_area: float,
    max_area: float,
    pass_index: int,
) -> List[DetectedRect]:
    work = _preprocess(gray, detect_pass)
    edges = cv2.Canny(work, int(detect_pass.canny_th1), int(detect_pass.canny_th2))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects: List[DetectedRect] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, float(detect_pass.epsilon_factor) * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        pts = approx.reshape(4, 2).astype(np.float32)
        if not _is_rectangle_quad(pts, float(detect_pass.angle_tol)):
            continue

        center = _quad_center(pts)
        rects.append(DetectedRect(center=center, box=pts, area=area, pass_index=pass_index))

    rects.sort(key=lambda r: r.area, reverse=True)
    return rects


def detect_rectangles(
    frame: np.ndarray,
    min_area_ratio: float = 0.005,
    max_area_ratio: float = 0.5,
    angle_tol: float = 25.0,
) -> List[DetectedRect]:
    """
    在输入 BGR 或灰度图像中检测矩形（包括旋转矩形）。返回矩形的 box points 和相关信息。
    - min_area_ratio: 与图像面积的最小比率（过小的轮廓会被丢弃）
    - angle_tol: 角度容忍度（判断为矩形时，四个角接近 90 度的容差）
    Returns rectangles sorted by area (descending).
    """
    h, w = frame.shape[:2]
    img_area = h * w
    min_area = img_area * float(min_area_ratio)
    max_area = img_area * float(max_area_ratio)

    detect_pass = DetectionPass(
        canny_th1=25,
        canny_th2=75,
        epsilon_factor=0.02,
        angle_tol=float(angle_tol),
        blur_ksize=3,
        use_otsu=True,
        morph_ksize=5,
        erode_iterations=1,
    )
    return _detect_with_pass(_as_gray(frame), detect_pass, min_area, max_area, pass_index=0)


def detect_rectangles_multi_pass(
    frame: np.ndarray,
    min_area_ratio: float = 0.005,
    max_area_ratio: float = 0.5,
    passes: Sequence[DetectionPass] = DETECT_PASSES,
) -> List[DetectedRect]:
    """Detect rectangles with progressively looser profiles.

    The first non-empty pass wins, matching the K230 script strategy. This keeps
    the fast/strict path preferred while still recovering difficult frames.
    """
    h, w = frame.shape[:2]
    img_area = h * w
    min_area = img_area * float(min_area_ratio)
    max_area = img_area * float(max_area_ratio)
    gray = _as_gray(frame)

    for pass_index, detect_pass in enumerate(passes):
        rects = _detect_with_pass(gray, detect_pass, min_area, max_area, pass_index)
        if rects:
            return rects
    return []


def with_rect_score(rect: DetectedRect, score: float) -> DetectedRect:
    return DetectedRect(
        center=rect.center,
        box=rect.box,
        area=rect.area,
        pass_index=rect.pass_index,
        score=score,
    )


def score_rect(
    rect: DetectedRect,
    frame_w: int,
    frame_h: int,
    last_center: Optional[Tuple[float, float]] = None,
    config: RectSelectionConfig = DEFAULT_SELECTION_CONFIG,
) -> float:
    img_area = float(frame_w * frame_h)
    max_area = img_area * float(config.max_area_ratio)
    max_center_dist = math.hypot(frame_w / 2.0, frame_h / 2.0)
    cx, cy = rect.center

    area_score = clamp(rect.area / max(max_area, 1.0), 0.0, 1.0)

    center_dist = math.hypot(cx - frame_w / 2.0, cy - frame_h / 2.0)
    center_score = 1.0 - clamp(center_dist / max(max_center_dist, 1.0), 0.0, 1.0)

    if last_center is None:
        previous_score = center_score
    else:
        previous_dist = math.hypot(cx - last_center[0], cy - last_center[1])
        previous_score = 1.0 - clamp(previous_dist / max(max_center_dist, 1.0), 0.0, 1.0)

    return area_score + config.center_weight * center_score + config.previous_weight * previous_score


def pick_best_rect(
    rects: Iterable[DetectedRect],
    frame_w: int,
    frame_h: int,
    last_center: Optional[Tuple[float, float]] = None,
    config: RectSelectionConfig = DEFAULT_SELECTION_CONFIG,
) -> Optional[DetectedRect]:
    img_area = float(frame_w * frame_h)
    min_area = img_area * float(config.min_area_ratio)
    max_area = img_area * float(config.max_area_ratio)
    best = None
    best_score = -1.0

    for rect in rects:
        if rect.area < min_area or rect.area > max_area:
            continue
        if rect_aspect(rect) > config.max_aspect_ratio:
            continue

        score = score_rect(rect, frame_w, frame_h, last_center, config)
        if score > best_score:
            best_score = score
            best = with_rect_score(rect, score)

    return best


class RectSelector:
    """Stateful target selector with temporal continuity."""

    def __init__(self, config: RectSelectionConfig = DEFAULT_SELECTION_CONFIG):
        self.config = config
        self.last_center: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        self.last_center = None

    def update(self, rects: Iterable[DetectedRect], frame_w: int, frame_h: int) -> Optional[DetectedRect]:
        best = pick_best_rect(rects, frame_w, frame_h, self.last_center, self.config)
        if best is None:
            self.reset()
            return None

        self.last_center = best.center
        return best


def draw_detected_rect(
    frame: np.ndarray,
    rect: Optional[DetectedRect],
    color: Union[int, Tuple[int, int, int]] = (0, 255, 0),
    thickness: int = 3,
    draw_center: bool = True,
    draw_text: bool = True,
) -> None:
    """Draw a detected rectangle in-place on a BGR frame."""
    if rect is None:
        return

    box = rect.box.astype(np.int32)
    cv2.polylines(frame, [box], isClosed=True, color=color, thickness=thickness)

    cx, cy = rect.center

    if draw_text:
        label = f"A:{int(rect.area)}"
        label_coord = f"X:{int(cx)} Y:{int(cy)}"
        cv2.putText(
            frame,
            label,
            (int(cx) - 80, int(cy) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        cv2.putText(
            frame,
            label_coord,
            (int(cx) - 80, int(cy) + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    if draw_center:
        center_color = 255 if frame.ndim == 2 else (0, 0, 255)
        cv2.circle(frame, (int(cx), int(cy)), 5, center_color, -1)
