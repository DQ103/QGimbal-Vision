from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .competition_vision import TrackedLaser, TrackedTarget


@dataclass(frozen=True)
class E25LaserConfig:
    min_luma: int = 150
    threshold_delta: int = 8
    centroid_delta: int = 6
    hold_frames: int = 2
    position_alpha: float = 0.78
    max_jump_ratio: float = 0.14
    min_violet_pixels_at_240p: float = 4.0
    ideal_pixels_at_240p: float = 45.0
    max_pixels_at_240p: float = 420.0
    max_side_at_240p: float = 34.0
    max_aspect_ratio: float = 2.5
    target_pad_ratio: float = 0.025
    use_active_difference: bool = False
    min_difference: int = 28


class E25LaserTracker:
    def __init__(self, config: E25LaserConfig = E25LaserConfig()) -> None:
        self.config = config
        self.last: Optional[TrackedLaser] = None
        self.last_pixels: Optional[int] = None
        self.last_area: Optional[int] = None
        self.miss_count = config.hold_frames
        self.off_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.last = None
        self.last_pixels = None
        self.last_area = None
        self.miss_count = self.config.hold_frames
        self.off_gray = None

    def set_laser_off_reference(self, frame: np.ndarray) -> None:
        self.off_gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.off_gray = self.off_gray.copy()

    def update(
        self,
        frame: np.ndarray,
        target: Optional[TrackedTarget],
        enabled: bool,
    ) -> Optional[TrackedLaser]:
        if not enabled or target is None or not target.current:
            self.reset()
            return None
        roi = self._target_roi(target, frame.shape[1], frame.shape[0])
        detected = self._detect(frame, roi)
        if detected is None:
            self.miss_count += 1
            if self.last is not None and self.miss_count <= self.config.hold_frames:
                held = TrackedLaser(
                    center=self.last.center,
                    bbox=self.last.bbox,
                    roi=roi,
                    score=self.last.score,
                    max_luma=self.last.max_luma,
                    threshold=self.last.threshold,
                    bright_pixels=self.last.bright_pixels,
                    violet_pixels=self.last.violet_pixels,
                    current=False,
                    held=True,
                    miss_count=self.miss_count,
                )
                self.last = held
                return held
            self.last = None
            self.last_pixels = None
            self.last_area = None
            return None

        cx, cy = detected.center
        if self.last is not None:
            alpha = self.config.position_alpha
            cx = alpha * cx + (1.0 - alpha) * self.last.center[0]
            cy = alpha * cy + (1.0 - alpha) * self.last.center[1]
        current = TrackedLaser(
            center=(cx, cy),
            bbox=detected.bbox,
            roi=detected.roi,
            score=detected.score,
            max_luma=detected.max_luma,
            threshold=detected.threshold,
            bright_pixels=detected.bright_pixels,
            violet_pixels=detected.violet_pixels,
            current=True,
            held=False,
            miss_count=0,
        )
        self.last = current
        self.last_pixels = current.bright_pixels
        self.last_area = current.bbox[2] * current.bbox[3]
        self.miss_count = 0
        return current

    def _target_roi(
        self,
        target: TrackedTarget,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[int, int, int, int]:
        x, y, width, height = cv2.boundingRect(target.rect.box.astype(np.float32))
        pad = max(2, int(round(min(frame_w, frame_h) * self.config.target_pad_ratio)))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame_w, x + width + pad)
        y2 = min(frame_h, y + height + pad)
        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

    def _detect(
        self,
        frame: np.ndarray,
        roi: Tuple[int, int, int, int],
    ) -> Optional[TrackedLaser]:
        rx, ry, rw, rh = roi
        roi_frame = frame[ry : ry + rh, rx : rx + rw]
        if roi_frame.size == 0:
            return None
        if roi_frame.ndim == 2:
            gray = roi_frame
            violet_mask = np.zeros_like(gray)
        else:
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2LAB)
            blue, green, red = cv2.split(roi_frame)
            blue_i16 = blue.astype(np.int16)
            blue_delta = np.where(
                (blue_i16 - red.astype(np.int16) >= 30)
                & (blue_i16 - green.astype(np.int16) >= 30),
                255,
                0,
            ).astype(np.uint8)
            hsv_mask = cv2.inRange(hsv, (112, 60, 175), (165, 255, 255))
            lab_mask = cv2.inRange(lab, (80, 150, 0), (255, 255, 112))
            violet_mask = cv2.bitwise_and(blue_delta, cv2.bitwise_or(hsv_mask, lab_mask))
            violet_mask = cv2.morphologyEx(
                violet_mask,
                cv2.MORPH_OPEN,
                np.ones((3, 3), dtype=np.uint8),
            )

        max_luma = int(gray.max())
        if max_luma < self.config.min_luma:
            return None
        threshold = min(254, max(self.config.min_luma, max_luma - self.config.threshold_delta))
        bright_mask = cv2.inRange(gray, threshold, 255)
        if self.config.use_active_difference and self.off_gray is not None:
            off_roi = self.off_gray[ry : ry + rh, rx : rx + rw]
            if off_roi.shape == gray.shape:
                difference = cv2.subtract(gray, off_roi)
                bright_mask = cv2.bitwise_and(
                    bright_mask,
                    cv2.inRange(difference, self.config.min_difference, 255),
                )
        merged = cv2.dilate(bright_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        scale = max(0.5, min(frame.shape[:2]) / 240.0)
        ideal_pixels = max(1.0, self.config.ideal_pixels_at_240p * scale * scale)
        max_pixels = max(4.0, self.config.max_pixels_at_240p * scale * scale)
        max_side = max(4.0, self.config.max_side_at_240p * scale)
        min_violet = max(2, int(round(self.config.min_violet_pixels_at_240p * scale * scale)))
        max_jump = self.config.max_jump_ratio * min(frame.shape[:2])

        best = None
        best_score = -1.0
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if width <= 0 or height <= 0 or width > max_side or height > max_side:
                continue
            aspect = max(width, height) / max(1.0, min(width, height))
            if aspect > self.config.max_aspect_ratio:
                continue
            local_bright = bright_mask[y : y + height, x : x + width]
            bright_pixels = int(cv2.countNonZero(local_bright))
            if bright_pixels < 1 or bright_pixels > max_pixels:
                continue
            local_gray = gray[y : y + height, x : x + width]
            local_max = int(local_gray.max())
            core_threshold = max(threshold, local_max - self.config.centroid_delta)
            core = cv2.inRange(local_gray, core_threshold, 255)
            ys, xs = np.nonzero(core)
            if xs.size == 0:
                continue
            weights = local_gray[ys, xs].astype(np.float64) - core_threshold + 1.0
            weight_sum = float(weights.sum())
            cx = rx + x + float(np.dot(xs, weights) / weight_sum)
            cy = ry + y + float(np.dot(ys, weights) / weight_sum)

            halo_pad = max(2, int(round(6.0 * scale)))
            hx1 = max(0, x - halo_pad)
            hy1 = max(0, y - halo_pad)
            hx2 = min(rw, x + width + halo_pad)
            hy2 = min(rh, y + height + halo_pad)
            violet_pixels = int(cv2.countNonZero(violet_mask[hy1:hy2, hx1:hx2]))
            if violet_pixels < min_violet:
                continue

            position_score = 0.0
            if self.last is not None:
                distance = math.hypot(cx - self.last.center[0], cy - self.last.center[1])
                if self.miss_count == 0 and distance > max_jump:
                    continue
                position_score = 1.0 - min(1.0, distance / max(max_jump, 1.0))
            size_score = _ideal_size_score(bright_pixels, ideal_pixels, max_pixels)
            round_score = min(width, height) / max(width, height)
            density_score = min(1.0, bright_pixels / float(width * height))
            bright_score = max(0.0, min(1.0, (local_max - self.config.min_luma) / (255 - self.config.min_luma)))
            violet_score = min(1.0, violet_pixels / max(float(min_violet) * 4.0, 1.0))
            temporal_score = 0.0
            if self.last_pixels is not None and self.last_area is not None:
                temporal_score = 0.6 * _ratio_similarity(bright_pixels, self.last_pixels)
                temporal_score += 0.4 * _ratio_similarity(width * height, self.last_area)
            score = 280.0 * position_score
            score += 240.0 * violet_score
            score += 220.0 * bright_score
            score += 160.0 * size_score
            score += 120.0 * round_score
            score += 100.0 * temporal_score
            score += 70.0 * density_score
            if score > best_score:
                best_score = score
                best = TrackedLaser(
                    center=(cx, cy),
                    bbox=(rx + x, ry + y, width, height),
                    roi=roi,
                    score=score,
                    max_luma=local_max,
                    threshold=threshold,
                    bright_pixels=bright_pixels,
                    violet_pixels=violet_pixels,
                    current=True,
                    held=False,
                    miss_count=0,
                )
        return best


def _ratio_similarity(first: float, second: float) -> float:
    if first <= 0.0 or second <= 0.0:
        return 0.0
    return min(first, second) / max(first, second)


def _ideal_size_score(value: float, ideal: float, maximum: float) -> float:
    if value <= ideal:
        return max(0.0, min(1.0, value / max(ideal, 1.0)))
    return 1.0 - max(0.0, min(1.0, (value - ideal) / max(maximum - ideal, 1.0)))
