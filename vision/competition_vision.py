from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np

from .rect_detect import (
    DEFAULT_SELECTION_CONFIG,
    DetectedRect,
    RectSelectionConfig,
    pick_best_rect,
)


@dataclass(frozen=True)
class TrackedTarget:
    rect: DetectedRect
    current: bool
    held: bool
    miss_count: int


class TargetTracker:
    """Keep a rectangle briefly across misses without treating it as measured."""

    def __init__(
        self,
        config: RectSelectionConfig = DEFAULT_SELECTION_CONFIG,
        miss_confirm_frames: int = 3,
    ) -> None:
        if miss_confirm_frames < 1:
            raise ValueError("miss_confirm_frames must be >= 1")
        self.config = config
        self.miss_confirm_frames = int(miss_confirm_frames)
        self.last_rect: Optional[DetectedRect] = None
        self.miss_count = self.miss_confirm_frames

    def reset(self) -> None:
        self.last_rect = None
        self.miss_count = self.miss_confirm_frames

    def update(
        self,
        rects: Iterable[DetectedRect],
        frame_w: int,
        frame_h: int,
    ) -> Optional[TrackedTarget]:
        last_center = self.last_rect.center if self.last_rect is not None else None
        best = pick_best_rect(rects, frame_w, frame_h, last_center, self.config)
        if best is not None:
            self.last_rect = best
            self.miss_count = 0
            return TrackedTarget(best, current=True, held=False, miss_count=0)

        self.miss_count = min(self.miss_count + 1, self.miss_confirm_frames)
        if self.last_rect is not None and self.miss_count < self.miss_confirm_frames:
            return TrackedTarget(
                self.last_rect,
                current=False,
                held=True,
                miss_count=self.miss_count,
            )

        self.reset()
        return None


@dataclass(frozen=True)
class AimStatus:
    ready: bool
    center: Tuple[float, float]
    distance: Optional[float]
    confirm_count: int


class AimReadyGate:
    """Require consecutive centered frames and use hysteresis after locking."""

    def __init__(
        self,
        enter_radius_ratio: float = 0.085,
        exit_radius_ratio: float = 0.12,
        confirm_frames: int = 3,
        offset_x_ratio: float = 0.0,
        offset_y_ratio: float = 0.0,
    ) -> None:
        if enter_radius_ratio <= 0.0:
            raise ValueError("enter_radius_ratio must be > 0")
        if exit_radius_ratio < enter_radius_ratio:
            raise ValueError("exit_radius_ratio must be >= enter_radius_ratio")
        if confirm_frames < 1:
            raise ValueError("confirm_frames must be >= 1")
        self.enter_radius_ratio = float(enter_radius_ratio)
        self.exit_radius_ratio = float(exit_radius_ratio)
        self.confirm_frames = int(confirm_frames)
        self.offset_x_ratio = float(offset_x_ratio)
        self.offset_y_ratio = float(offset_y_ratio)
        self.ready = False
        self.confirm_count = 0

    def reset(self) -> None:
        self.ready = False
        self.confirm_count = 0

    def update(
        self,
        target: Optional[TrackedTarget],
        frame_w: int,
        frame_h: int,
    ) -> AimStatus:
        aim_center = (
            frame_w * (0.5 + self.offset_x_ratio),
            frame_h * (0.5 + self.offset_y_ratio),
        )
        if target is None or not target.current:
            self.reset()
            return AimStatus(False, aim_center, None, 0)

        distance = math.hypot(
            target.rect.center[0] - aim_center[0],
            target.rect.center[1] - aim_center[1],
        )
        scale = float(min(frame_w, frame_h))
        enter_radius = self.enter_radius_ratio * scale
        exit_radius = self.exit_radius_ratio * scale

        if self.ready:
            if distance > exit_radius:
                self.reset()
        elif distance <= enter_radius:
            self.confirm_count += 1
            if self.confirm_count >= self.confirm_frames:
                self.ready = True
        else:
            self.confirm_count = 0

        return AimStatus(self.ready, aim_center, distance, self.confirm_count)


@dataclass(frozen=True)
class HybridLaserConfig:
    min_dynamic_luma: int = 165
    fallback_min_luma: int = 210
    threshold_delta_from_max: int = 8
    centroid_delta_from_max: int = 6
    target_pad_ratio: float = 0.025
    ideal_pixels_at_240p: float = 45.0
    max_pixels_at_240p: float = 420.0
    max_side_at_240p: float = 34.0
    max_aspect_ratio: float = 2.5
    max_jump_ratio: float = 0.14
    hold_frames: int = 2
    position_alpha: float = 0.75


@dataclass(frozen=True)
class TrackedLaser:
    center: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    roi: Tuple[int, int, int, int]
    score: float
    max_luma: int
    threshold: int
    bright_pixels: int
    violet_pixels: int
    current: bool
    held: bool
    miss_count: int


class VisionStage(Enum):
    SEARCH_TARGET = "search_target"
    ALIGN_TARGET = "align_target"
    SEARCH_LASER = "search_laser"
    ALIGN_LASER = "align_laser"


def resolve_stage(
    target: Optional[TrackedTarget],
    aim: AimStatus,
    laser: Optional[TrackedLaser],
) -> VisionStage:
    if target is None or not target.current:
        return VisionStage.SEARCH_TARGET
    if not aim.ready:
        return VisionStage.ALIGN_TARGET
    if laser is None or not laser.current:
        return VisionStage.SEARCH_LASER
    return VisionStage.ALIGN_LASER


def stage_error(
    stage: VisionStage,
    target: Optional[TrackedTarget],
    aim: AimStatus,
    laser: Optional[TrackedLaser],
) -> Tuple[float, float]:
    if target is None or not target.current:
        return 0.0, 0.0
    tx, ty = target.rect.center
    if stage == VisionStage.ALIGN_LASER and laser is not None and laser.current:
        return tx - laser.center[0], ty - laser.center[1]
    return tx - aim.center[0], ty - aim.center[1]


class HybridLaserTracker:
    """Track a saturated laser core with optional blue-violet halo support."""

    def __init__(self, config: HybridLaserConfig = HybridLaserConfig()) -> None:
        if config.hold_frames < 0:
            raise ValueError("hold_frames must be >= 0")
        if not 0.0 < config.position_alpha <= 1.0:
            raise ValueError("position_alpha must be in (0, 1]")
        self.config = config
        self.last: Optional[TrackedLaser] = None
        self.last_pixels: Optional[int] = None
        self.last_area: Optional[int] = None
        self.miss_count = config.hold_frames

    def reset(self) -> None:
        self.last = None
        self.last_pixels = None
        self.last_area = None
        self.miss_count = self.config.hold_frames

    def update(
        self,
        frame: np.ndarray,
        target: Optional[TrackedTarget],
        enabled: bool,
    ) -> Optional[TrackedLaser]:
        if not enabled or target is None or not target.current:
            self.reset()
            return None

        roi = self._target_roi(target.rect, frame.shape[1], frame.shape[0])
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

        raw_x, raw_y = detected.center
        if self.last is not None:
            alpha = self.config.position_alpha
            raw_x = alpha * raw_x + (1.0 - alpha) * self.last.center[0]
            raw_y = alpha * raw_y + (1.0 - alpha) * self.last.center[1]

        current = TrackedLaser(
            center=(raw_x, raw_y),
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
        rect: DetectedRect,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[int, int, int, int]:
        x, y, w, h = cv2.boundingRect(rect.box.astype(np.float32))
        pad = max(2, int(round(min(frame_w, frame_h) * self.config.target_pad_ratio)))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame_w, x + w + pad)
        y2 = min(frame_h, y + h + pad)
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
            lab = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2LAB)
            strict = cv2.inRange(lab, (89, 128, 0), (255, 255, 44))
            relaxed = cv2.inRange(lab, (38, 128, 0), (255, 255, 53))
            violet_mask = cv2.bitwise_or(strict, relaxed)

        max_luma = int(gray.max())
        if max_luma < self.config.min_dynamic_luma:
            return None
        threshold = min(
            254,
            max(self.config.min_dynamic_luma, max_luma - self.config.threshold_delta_from_max),
        )
        bright_mask = cv2.inRange(gray, threshold, 255)
        merged_mask = cv2.dilate(bright_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
        contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        scale = max(0.5, min(frame.shape[:2]) / 240.0)
        ideal_pixels = max(1.0, self.config.ideal_pixels_at_240p * scale * scale)
        max_pixels = max(4.0, self.config.max_pixels_at_240p * scale * scale)
        max_side = max(4.0, self.config.max_side_at_240p * scale)
        max_jump = self.config.max_jump_ratio * min(frame.shape[:2])

        best: Optional[TrackedLaser] = None
        best_score = -1.0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0 or w > max_side or h > max_side:
                continue
            aspect = max(w, h) / max(1.0, min(w, h))
            if aspect > self.config.max_aspect_ratio:
                continue

            local_bright = bright_mask[y : y + h, x : x + w]
            bright_pixels = int(cv2.countNonZero(local_bright))
            if bright_pixels < 1 or bright_pixels > max_pixels:
                continue

            local_gray = gray[y : y + h, x : x + w]
            local_max = int(local_gray.max())
            core_threshold = max(threshold, local_max - self.config.centroid_delta_from_max)
            core_mask = cv2.inRange(local_gray, core_threshold, 255)
            ys, xs = np.nonzero(core_mask)
            if xs.size == 0:
                continue
            weights = local_gray[ys, xs].astype(np.float64) - core_threshold + 1.0
            sum_weight = float(weights.sum())
            cx = rx + x + float(np.dot(xs, weights) / sum_weight)
            cy = ry + y + float(np.dot(ys, weights) / sum_weight)

            halo_pad = max(2, int(round(6.0 * scale)))
            hx1 = max(0, x - halo_pad)
            hy1 = max(0, y - halo_pad)
            hx2 = min(rw, x + w + halo_pad)
            hy2 = min(rh, y + h + halo_pad)
            violet_pixels = int(cv2.countNonZero(violet_mask[hy1:hy2, hx1:hx2]))
            if violet_pixels == 0 and local_max < self.config.fallback_min_luma:
                continue

            if self.last is not None:
                distance = math.hypot(cx - self.last.center[0], cy - self.last.center[1])
                if self.miss_count == 0 and distance > max_jump:
                    continue
                position_score = 1.0 - min(1.0, distance / max(max_jump, 1.0))
            else:
                position_score = 0.0

            size_score = _ideal_size_score(bright_pixels, ideal_pixels, max_pixels)
            round_score = min(w, h) / max(w, h)
            density_score = min(1.0, bright_pixels / float(w * h))
            bright_score = max(
                0.0,
                min(1.0, (local_max - self.config.min_dynamic_luma) / float(255 - self.config.min_dynamic_luma)),
            )
            violet_score = min(1.0, violet_pixels / max(2.0, ideal_pixels * 0.25))
            temporal_score = 0.0
            if self.last_pixels is not None and self.last_area is not None:
                temporal_score = 0.6 * _ratio_similarity(bright_pixels, self.last_pixels)
                temporal_score += 0.4 * _ratio_similarity(w * h, self.last_area)

            score = 260.0 * position_score
            score += 220.0 * bright_score
            score += 180.0 * size_score
            score += 150.0 * violet_score
            score += 120.0 * round_score
            score += 100.0 * temporal_score
            score += 70.0 * density_score

            if score > best_score:
                best_score = score
                best = TrackedLaser(
                    center=(cx, cy),
                    bbox=(rx + x, ry + y, w, h),
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


def _ratio_similarity(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return min(a, b) / max(a, b)


def _ideal_size_score(value: float, ideal: float, maximum: float) -> float:
    if value <= ideal:
        return max(0.0, min(1.0, value / max(ideal, 1.0)))
    return 1.0 - max(0.0, min(1.0, (value - ideal) / max(maximum - ideal, 1.0)))
