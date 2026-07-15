from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .a4_target import order_quad_points


@dataclass(frozen=True)
class CoarseMotionConfig:
    analysis_scale: float = 0.3
    max_features: int = 50
    quality_level: float = 0.01
    min_distance: float = 6.0
    roi_pad_ratio: float = 0.30
    min_inliers: int = 5
    max_forward_backward_error: float = 1.8
    max_scale_change: float = 0.28
    max_rotation_deg: float = 24.0
    max_corner_shift_ratio: float = 0.55
    motion_pixel_threshold: int = 10
    motion_ratio_threshold: float = 0.012


@dataclass(frozen=True)
class CoarseMotionEstimate:
    quad: Optional[np.ndarray]
    inliers: int
    tracked_points: int
    confidence: float
    shift: Tuple[float, float]
    scale: float
    rotation_deg: float


class E25CoarseMotionTracker:
    def __init__(self, config: CoarseMotionConfig = CoarseMotionConfig()) -> None:
        self.config = config
        self.previous_gray: Optional[np.ndarray] = None
        self.previous_quad: Optional[np.ndarray] = None

    def reset(
        self,
        frame: Optional[np.ndarray] = None,
        quad: Optional[np.ndarray] = None,
    ) -> None:
        if frame is None or quad is None:
            self.previous_gray = None
            self.previous_quad = None
            return
        gray = _analysis_gray(frame, self.config.analysis_scale)
        scaled_quad = order_quad_points(quad) * self.config.analysis_scale
        self._store_state(gray, scaled_quad)

    def align_quad(self, quad: np.ndarray) -> None:
        if self.previous_gray is None:
            return
        self.previous_quad = (
            order_quad_points(quad) * self.config.analysis_scale
        ).astype(np.float32)

    def estimate(
        self,
        frame: np.ndarray,
        reference_quad: np.ndarray,
    ) -> CoarseMotionEstimate:
        gray = _analysis_gray(frame, self.config.analysis_scale)
        reference_full = order_quad_points(reference_quad)
        reference_quad = reference_full * self.config.analysis_scale
        if self.previous_gray is None or self.previous_quad is None:
            self._store_state(gray, reference_quad)
            return _empty_estimate()

        motion_ratio = self._motion_ratio(
            self.previous_gray,
            gray,
            self.previous_quad,
        )
        if motion_ratio < self.config.motion_ratio_threshold:
            self._store_state(gray, reference_quad)
            return CoarseMotionEstimate(
                quad=reference_full.copy(),
                inliers=0,
                tracked_points=0,
                confidence=1.0,
                shift=(0.0, 0.0),
                scale=1.0,
                rotation_deg=0.0,
            )

        points = self._feature_points(self.previous_gray, self.previous_quad)
        if points is None or len(points) < self.config.min_inliers:
            self._store_state(gray, reference_quad)
            return _empty_estimate()

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            points,
            None,
            winSize=(25, 25),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        )
        if next_points is None or status is None:
            self._store_state(gray, reference_quad)
            return _empty_estimate()
        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            self.previous_gray,
            next_points,
            None,
            winSize=(25, 25),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        )
        if back_points is None or back_status is None:
            self._store_state(gray, reference_quad)
            return _empty_estimate()

        valid = (status.reshape(-1) > 0) & (back_status.reshape(-1) > 0)
        forward_backward = np.linalg.norm(
            points.reshape(-1, 2) - back_points.reshape(-1, 2),
            axis=1,
        )
        valid &= forward_backward <= self.config.max_forward_backward_error
        previous = points.reshape(-1, 2)[valid]
        current = next_points.reshape(-1, 2)[valid]
        if len(previous) < self.config.min_inliers:
            self._store_state(gray, reference_quad)
            return _empty_estimate(tracked_points=len(previous))

        transform, inlier_mask = cv2.estimateAffinePartial2D(
            previous,
            current,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=1000,
            confidence=0.995,
            refineIters=10,
        )
        if transform is None or inlier_mask is None:
            self._store_state(gray, reference_quad)
            return _empty_estimate(tracked_points=len(previous))

        inliers = int(np.count_nonzero(inlier_mask))
        if inliers < self.config.min_inliers:
            self._store_state(gray, reference_quad)
            return _empty_estimate(inliers=inliers, tracked_points=len(previous))

        a, b, tx = transform[0]
        c, d, ty = transform[1]
        scale = math.sqrt(max(0.0, float(a * a + c * c)))
        rotation_deg = math.degrees(math.atan2(float(c), float(a)))
        transformed_scaled = cv2.transform(reference_quad.reshape(1, -1, 2), transform)[0]
        transformed = transformed_scaled / self.config.analysis_scale
        corner_shift = np.linalg.norm(transformed - reference_full, axis=1)
        target_scale = max(math.sqrt(abs(float(cv2.contourArea(reference_full)))), 1.0)
        valid_transform = (
            abs(scale - 1.0) <= self.config.max_scale_change
            and abs(rotation_deg) <= self.config.max_rotation_deg
            and float(np.max(corner_shift))
            <= max(28.0, self.config.max_corner_shift_ratio * target_scale)
            and cv2.isContourConvex(transformed.astype(np.float32))
        )
        if not valid_transform:
            self._store_state(gray, reference_quad)
            return _empty_estimate(inliers=inliers, tracked_points=len(previous))

        inlier_ratio = inliers / max(float(len(previous)), 1.0)
        fb_score = 1.0 - min(
            1.0,
            float(np.median(forward_backward[valid]))
            / max(self.config.max_forward_backward_error, 1e-3),
        )
        confidence = max(0.0, min(1.0, 0.72 * inlier_ratio + 0.28 * fb_score))
        transformed = order_quad_points(transformed.astype(np.float32))
        self._store_state(gray, transformed_scaled)
        return CoarseMotionEstimate(
            quad=transformed,
            inliers=inliers,
            tracked_points=len(previous),
            confidence=confidence,
            shift=(
                float(tx) / self.config.analysis_scale,
                float(ty) / self.config.analysis_scale,
            ),
            scale=float(scale),
            rotation_deg=float(rotation_deg),
        )

    def _feature_points(
        self,
        gray: np.ndarray,
        quad: np.ndarray,
    ) -> Optional[np.ndarray]:
        frame_h, frame_w = gray.shape[:2]
        x, y, width, height = cv2.boundingRect(order_quad_points(quad))
        pad = int(round(max(width, height) * self.config.roi_pad_ratio))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame_w, x + width + pad)
        y2 = min(frame_h, y + height + pad)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        mask = np.zeros_like(gray)
        mask[y1:y2, x1:x2] = 255
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.config.max_features,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_distance,
            mask=mask,
            blockSize=5,
            useHarrisDetector=False,
        )

    def _store_state(self, gray: np.ndarray, scaled_quad: np.ndarray) -> None:
        self.previous_gray = gray.copy()
        self.previous_quad = order_quad_points(scaled_quad).copy()

    def _motion_ratio(
        self,
        previous_gray: np.ndarray,
        gray: np.ndarray,
        quad: np.ndarray,
    ) -> float:
        frame_h, frame_w = gray.shape[:2]
        x, y, width, height = cv2.boundingRect(order_quad_points(quad))
        pad = int(round(max(width, height) * self.config.roi_pad_ratio))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame_w, x + width + pad)
        y2 = min(frame_h, y + height + pad)
        if x2 <= x1 or y2 <= y1:
            return 1.0
        difference = cv2.absdiff(
            previous_gray[y1:y2, x1:x2],
            gray[y1:y2, x1:x2],
        )
        moving = cv2.compare(
            difference,
            self.config.motion_pixel_threshold,
            cv2.CMP_GT,
        )
        return cv2.countNonZero(moving) / max(float(moving.size), 1.0)


def _as_gray(frame: np.ndarray) -> np.ndarray:
    return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _analysis_gray(frame: np.ndarray, scale: float) -> np.ndarray:
    gray = _as_gray(frame)
    if scale == 1.0:
        return gray
    return cv2.resize(
        gray,
        (0, 0),
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


def _empty_estimate(
    inliers: int = 0,
    tracked_points: int = 0,
) -> CoarseMotionEstimate:
    return CoarseMotionEstimate(
        quad=None,
        inliers=int(inliers),
        tracked_points=int(tracked_points),
        confidence=0.0,
        shift=(0.0, 0.0),
        scale=1.0,
        rotation_deg=0.0,
    )
