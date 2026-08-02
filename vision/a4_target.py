from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .rect_detect import DetectedRect, clamp


@dataclass(frozen=True)
class A4TargetConfig:
    canonical_short: int = 210
    canonical_long: int = 297
    tape_width: int = 18
    edge_samples_per_side: int = 16
    max_candidates: int = 2
    min_area_ratio: float = 0.015
    max_area_ratio: float = 0.70
    acquire_confidence: float = 0.72
    track_confidence: float = 0.52
    occluded_confidence: float = 0.30
    acquire_confirm_frames: int = 3
    occlusion_hold_frames: int = 8
    switch_margin: float = 0.15
    switch_confirm_frames: int = 5
    global_interval: int = 10
    search_interval: int = 6
    local_validate_interval: int = 3
    min_flow_points: int = 12
    max_flow_points: int = 100
    max_apparent_aspect: float = 2.3


@dataclass(frozen=True)
class A4FeatureScores:
    edge: float
    black_band: float
    red_rings: float
    pose: float
    temporal: float
    side_scores: Tuple[float, float, float, float]
    ring_scores: Tuple[float, float, float, float, float]
    visible_sides: int


@dataclass(frozen=True)
class A4Detection:
    rect: DetectedRect
    quad: np.ndarray
    center: Tuple[float, float]
    area: float
    confidence: float
    structural_confidence: float
    homography: np.ndarray
    inverse_homography: np.ndarray
    canonical_size: Tuple[int, int]
    canonical: np.ndarray
    scores: A4FeatureScores
    edge_points: Tuple[Tuple[float, float, float], ...]


class A4TrackState(Enum):
    SEARCH = "search"
    ACQUIRED = "acquired"
    TRACKING = "tracking"
    OCCLUDED = "occluded"
    LOST = "lost"


@dataclass(frozen=True)
class A4TrackResult:
    detection: Optional[A4Detection]
    state: A4TrackState
    current: bool
    predicted: bool
    miss_count: int
    flow_inliers: int


def order_quad_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)
    edge_a = ordered[1] - ordered[0]
    edge_b = ordered[2] - ordered[1]
    if float(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0]) < 0:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered.astype(np.float32)


def quad_center(quad: np.ndarray) -> Tuple[float, float]:
    q = order_quad_points(quad)
    p1, p2, p3, p4 = q
    den = (p1[0] - p3[0]) * (p2[1] - p4[1]) - (p1[1] - p3[1]) * (p2[0] - p4[0])
    if abs(float(den)) < 1e-6:
        center = q.mean(axis=0)
        return float(center[0]), float(center[1])
    a = p1[0] * p3[1] - p1[1] * p3[0]
    b = p2[0] * p4[1] - p2[1] * p4[0]
    x = (a * (p2[0] - p4[0]) - (p1[0] - p3[0]) * b) / den
    y = (a * (p2[1] - p4[1]) - (p1[1] - p3[1]) * b) / den
    return float(x), float(y)


def detection_as_rect(detection: A4Detection) -> DetectedRect:
    return DetectedRect(
        center=detection.center,
        box=detection.quad.astype(np.float32),
        area=detection.area,
        pass_index=detection.rect.pass_index,
        score=detection.confidence,
    )


def map_image_point_to_a4(detection: A4Detection, point: Tuple[float, float]) -> Tuple[float, float]:
    src = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(src, detection.homography)[0, 0]
    pixels_per_mm = min(detection.canonical_size) / 210.0
    return float(mapped[0]) / pixels_per_mm, float(mapped[1]) / pixels_per_mm


def a4_center_mm(detection: A4Detection) -> Tuple[float, float]:
    pixels_per_mm = min(detection.canonical_size) / 210.0
    return (
        detection.canonical_size[0] / (2.0 * pixels_per_mm),
        detection.canonical_size[1] / (2.0 * pixels_per_mm),
    )


def find_black_band_candidates(
    frame: np.ndarray,
    config: A4TargetConfig = A4TargetConfig(),
) -> List[DetectedRect]:
    original_h, original_w = frame.shape[:2]
    analysis_scale = 0.5 if min(original_w, original_h) >= 480 else 1.0
    if analysis_scale < 1.0:
        analysis_w = max(2, int(round(original_w * analysis_scale)))
        analysis_h = max(2, int(round(original_h * analysis_scale)))
        analysis_frame = cv2.resize(frame, (analysis_w, analysis_h), interpolation=cv2.INTER_AREA)
    else:
        analysis_frame = frame
    gray = analysis_frame if analysis_frame.ndim == 2 else cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frame_h, frame_w = gray.shape[:2]
    frame_area = float(frame_w * frame_h)
    min_area = frame_area * config.min_area_ratio
    max_area = frame_area * config.max_area_ratio
    candidates: List[DetectedRect] = []

    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < min_area * 0.15 or contour_area > max_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(np.float32)
        else:
            rotated = cv2.minAreaRect(contour)
            quad = cv2.boxPoints(rotated).astype(np.float32)
        quad = order_quad_points(quad)
        area = float(abs(cv2.contourArea(quad)))
        if area < min_area or area > max_area:
            continue
        if analysis_scale < 1.0:
            inverse_scale = 1.0 / analysis_scale
            quad = quad * inverse_scale
            area = area * inverse_scale * inverse_scale
        center = quad_center(quad)
        candidates.append(DetectedRect(center=center, box=quad, area=area, pass_index=20))

    candidates.sort(key=lambda item: item.area, reverse=True)
    return candidates[: config.max_candidates * 2]


def merge_candidates(
    primary: Iterable[DetectedRect],
    secondary: Iterable[DetectedRect],
    limit: int = 12,
) -> List[DetectedRect]:
    merged: List[DetectedRect] = []
    for rect in list(primary) + list(secondary):
        quad = order_quad_points(rect.box)
        bbox = cv2.boundingRect(quad)
        duplicate = False
        for existing in merged:
            existing_bbox = cv2.boundingRect(order_quad_points(existing.box))
            if _bbox_iou(bbox, existing_bbox) >= 0.72:
                duplicate = True
                break
        if not duplicate:
            merged.append(rect)
    merged.sort(key=lambda item: item.area, reverse=True)
    return merged[:limit]


class A4TargetDetector:
    def __init__(self, config: A4TargetConfig = A4TargetConfig()) -> None:
        self.config = config

    def detect(
        self,
        frame: np.ndarray,
        candidates: Sequence[DetectedRect],
        previous: Optional[A4Detection] = None,
    ) -> List[A4Detection]:
        detections: List[A4Detection] = []
        ranked = sorted(candidates, key=_quick_candidate_score, reverse=True)
        for rect in ranked[: self.config.max_candidates]:
            detection = self.evaluate(frame, rect, previous)
            if detection is not None:
                detections.append(detection)
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections

    def evaluate(
        self,
        frame: np.ndarray,
        rect: DetectedRect,
        previous: Optional[A4Detection] = None,
    ) -> Optional[A4Detection]:
        frame_h, frame_w = frame.shape[:2]
        frame_area = float(frame_w * frame_h)
        if rect.area < frame_area * self.config.min_area_ratio:
            return None
        if rect.area > frame_area * self.config.max_area_ratio:
            return None

        original = order_quad_points(rect.box)
        if _apparent_aspect(original) > self.config.max_apparent_aspect:
            return None
        horizontal = 0.5 * (
            np.linalg.norm(original[1] - original[0]) + np.linalg.norm(original[2] - original[3])
        )
        vertical = 0.5 * (
            np.linalg.norm(original[2] - original[1]) + np.linalg.norm(original[3] - original[0])
        )
        inferred_size = (
            (self.config.canonical_long, self.config.canonical_short)
            if horizontal >= vertical
            else (self.config.canonical_short, self.config.canonical_long)
        )
        canonical_sizes = (previous.canonical_size,) if rect.pass_index == -3 and previous is not None else (inferred_size,)
        scale_variants = (
            ((1.0, 1.0),)
            if rect.pass_index == -3
            else ((1.0, 1.0), (1.16, 1.12))
        )
        best: Optional[A4Detection] = None
        for scale_x, scale_y in scale_variants:
            quad = _scale_quad(original, scale_x, scale_y)
            if not _quad_inside_reasonable_bounds(quad, frame_w, frame_h):
                continue
            for canonical_size in canonical_sizes:
                detection = self._evaluate_quad(frame, rect, quad, canonical_size, previous)
                if detection is not None and (best is None or detection.confidence > best.confidence):
                    best = detection
        return best

    def _evaluate_quad(
        self,
        frame: np.ndarray,
        rect: DetectedRect,
        quad: np.ndarray,
        canonical_size: Tuple[int, int],
        previous: Optional[A4Detection],
    ) -> Optional[A4Detection]:
        width, height = canonical_size
        destination = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(quad.astype(np.float32), destination)
        if not np.isfinite(homography).all():
            return None
        inverse = np.linalg.inv(homography)
        canonical = cv2.warpPerspective(
            frame,
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        edge_score, side_scores, edge_points = self._edge_support(canonical, inverse)
        black_score = self._black_band_score(canonical)
        red_score, ring_scores = self._red_ring_score(canonical)
        pose_score = _pose_score(quad)
        if pose_score < 0.30:
            return None
        temporal_score = _temporal_score(quad, previous, frame.shape[1], frame.shape[0])
        visible_sides = sum(score >= 0.32 for score in side_scores)

        structural = 0.31 * edge_score + 0.31 * black_score + 0.25 * red_score + 0.13 * pose_score
        confidence = 0.88 * structural + 0.12 * temporal_score
        area = float(abs(cv2.contourArea(quad)))
        center = quad_center(quad)
        scored_rect = DetectedRect(
            center=center,
            box=quad.astype(np.float32),
            area=area,
            pass_index=rect.pass_index,
            score=confidence,
        )
        scores = A4FeatureScores(
            edge=edge_score,
            black_band=black_score,
            red_rings=red_score,
            pose=pose_score,
            temporal=temporal_score,
            side_scores=side_scores,
            ring_scores=ring_scores,
            visible_sides=visible_sides,
        )
        return A4Detection(
            rect=scored_rect,
            quad=quad.astype(np.float32),
            center=center,
            area=area,
            confidence=confidence,
            structural_confidence=structural,
            homography=homography,
            inverse_homography=inverse,
            canonical_size=canonical_size,
            canonical=canonical,
            scores=scores,
            edge_points=edge_points,
        )

    def _edge_support(
        self,
        canonical: np.ndarray,
        inverse_homography: np.ndarray,
    ) -> Tuple[float, Tuple[float, float, float, float], Tuple[Tuple[float, float, float], ...]]:
        gray = canonical if canonical.ndim == 2 else cv2.cvtColor(canonical, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        tape = min(self.config.tape_width, min(width, height) // 5)
        band_pos = max(4, tape // 2)
        inner_pos = min(min(width, height) // 3, tape + max(10, tape // 2))
        fractions = np.linspace(0.10, 0.90, self.config.edge_samples_per_side)
        xs = np.rint(fractions * (width - 1)).astype(np.int32)
        ys = np.rint(fractions * (height - 1)).astype(np.int32)
        smooth = cv2.blur(gray, (7, 7))
        side_coordinates = (
            (xs, np.full_like(xs, band_pos), xs, np.full_like(xs, inner_pos)),
            (
                np.full_like(ys, width - 1 - band_pos),
                ys,
                np.full_like(ys, width - 1 - inner_pos),
                ys,
            ),
            (
                xs,
                np.full_like(xs, height - 1 - band_pos),
                xs,
                np.full_like(xs, height - 1 - inner_pos),
            ),
            (np.full_like(ys, band_pos), ys, np.full_like(ys, inner_pos), ys),
        )
        side_values: List[np.ndarray] = []
        canonical_points: List[Tuple[float, float, float]] = []
        for band_x, band_y, inner_x, inner_y in side_coordinates:
            band = smooth[band_y, band_x].astype(np.float32)
            inner = smooth[inner_y, inner_x].astype(np.float32)
            contrast_score = np.clip((inner - band - 8.0) / 55.0, 0.0, 1.0)
            darkness_score = np.clip((205.0 - band) / 120.0, 0.0, 1.0)
            support = 0.72 * contrast_score + 0.28 * darkness_score
            side_values.append(support)
            canonical_points.extend(
                (float(x), float(y), float(value))
                for x, y, value in zip(band_x, band_y, support)
            )

        side_scores = tuple(_robust_support(values.tolist()) for values in side_values)
        sorted_sides = sorted(side_scores, reverse=True)
        edge_score = 0.80 * float(np.mean(sorted_sides[:3])) + 0.20 * float(np.mean(sorted_sides))

        source = np.array([[[x, y] for x, y, _ in canonical_points]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(source, inverse_homography)[0]
        edge_points = tuple(
            (float(point[0]), float(point[1]), float(canonical_points[index][2]))
            for index, point in enumerate(mapped)
        )
        return edge_score, side_scores, edge_points

    def _black_band_score(self, canonical: np.ndarray) -> float:
        gray = canonical if canonical.ndim == 2 else cv2.cvtColor(canonical, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        tape = min(self.config.tape_width, min(width, height) // 5)
        margin = max(4, tape // 6)
        inset = min(min(width, height) // 3, tape + max(14, tape // 2))
        border_values = np.concatenate(
            (
                gray[margin:tape, margin : width - margin].reshape(-1),
                gray[height - tape : height - margin, margin : width - margin].reshape(-1),
                gray[tape : height - tape, margin:tape].reshape(-1),
                gray[tape : height - tape, width - tape : width - margin].reshape(-1),
            )
        )
        interior_values = gray[inset : height - inset, inset : width - inset].reshape(-1)
        if border_values.size == 0 or interior_values.size == 0:
            return 0.0
        border_median = float(np.median(border_values))
        interior_median = float(np.median(interior_values))
        contrast = clamp((interior_median - border_median - 10.0) / 65.0, 0.0, 1.0)
        dark_fraction = float(np.mean(border_values < max(100.0, interior_median - 25.0)))
        interior_bright = clamp((interior_median - 65.0) / 130.0, 0.0, 1.0)
        return 0.55 * contrast + 0.30 * dark_fraction + 0.15 * interior_bright

    @staticmethod
    def _red_ring_score(canonical: np.ndarray) -> Tuple[float, Tuple[float, float, float, float, float]]:
        if canonical.ndim != 3:
            return 0.0, (0.0, 0.0, 0.0, 0.0, 0.0)
        b, g, r = cv2.split(canonical.astype(np.int16))
        hsv = cv2.cvtColor(canonical, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        red_strength = r - np.maximum(g, b)
        red_hue = (hue <= 10) | (hue >= 170)
        red_mask = (
            (r >= 75)
            & (red_strength >= 28)
            & (r - g >= 32)
            & red_hue
            & (saturation >= 85)
            & (value >= 65)
        ).astype(np.uint8)
        height, width = red_mask.shape[:2]
        cx = (width - 1) * 0.5
        cy = (height - 1) * 0.5
        ring_scores: List[float] = []
        pixels_per_mm = min(width, height) / 210.0
        expanded_red = cv2.dilate(red_mask, np.ones((7, 7), dtype=np.uint8), iterations=1)
        angles = np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False)
        cosines = np.cos(angles)
        sines = np.sin(angles)
        for radius_mm in (20, 40, 60, 80, 100):
            radius = radius_mm * pixels_per_mm
            xs = np.rint(cx + radius * cosines).astype(np.int32)
            ys = np.rint(cy + radius * sines).astype(np.int32)
            valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            if not np.any(valid):
                ring_scores.append(0.0)
                continue
            ring_scores.append(float(np.mean(expanded_red[ys[valid], xs[valid]] > 0)))
        strongest = sorted(ring_scores, reverse=True)
        supported_rings = sum(value >= 0.12 for value in ring_scores)
        pattern_score = clamp((strongest[0] + strongest[1] + 0.5 * strongest[2]) / 1.15, 0.0, 1.0)
        multiplicity = clamp((supported_rings - 1) / 2.0, 0.0, 1.0)
        score = pattern_score * multiplicity
        return score, tuple(ring_scores)  # type: ignore[return-value]


class A4TargetTracker:
    def __init__(
        self,
        config: A4TargetConfig = A4TargetConfig(),
        require_red_rings: bool = True,
    ) -> None:
        self.config = config
        self.detector = A4TargetDetector(config)
        self.state = A4TrackState.SEARCH
        self.current: Optional[A4Detection] = None
        self.previous_gray: Optional[np.ndarray] = None
        self.flow_points: Optional[np.ndarray] = None
        self.miss_count = 0
        self.flow_inliers = 0
        self.flow_coverage = 0.0
        self.pending: Optional[A4Detection] = None
        self.pending_count = 0
        self.search_preview: Optional[A4Detection] = None
        self.challenger: Optional[A4Detection] = None
        self.challenger_count = 0
        self.frame_count = 0
        self.require_red_rings = bool(require_red_rings)

    def reset(self) -> None:
        self.state = A4TrackState.SEARCH
        self.current = None
        self.flow_points = None
        self.miss_count = 0
        self.flow_inliers = 0
        self.flow_coverage = 0.0
        self.pending = None
        self.pending_count = 0
        self.search_preview = None
        self.challenger = None
        self.challenger_count = 0
        self.frame_count = 0

    def set_require_red_rings(self, require_red_rings: bool) -> None:
        require_red_rings = bool(require_red_rings)
        if require_red_rings == self.require_red_rings:
            return
        self.require_red_rings = require_red_rings
        self.reset()

    def needs_global_detection(self, frame_index: int) -> bool:
        if self.current is None or self.state in (A4TrackState.SEARCH, A4TrackState.LOST):
            return (frame_index - 1) % self.config.search_interval == 0
        return frame_index % self.config.global_interval == 0

    def update(
        self,
        frame: np.ndarray,
        candidates: Sequence[DetectedRect],
        detection_cycle: bool = True,
    ) -> A4TrackResult:
        self.frame_count += 1
        if self.current is None and not detection_cycle:
            return A4TrackResult(self.search_preview, A4TrackState.SEARCH, False, False, 0, 0)
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        predicted, tracked_points = self._predict(frame, gray)
        detections = [
            self._apply_validation_mode(detection)
            for detection in self.detector.detect(frame, candidates, self.current)
        ]
        if predicted is not None:
            predicted = self._apply_validation_mode(predicted)
        best = detections[0] if detections else None

        if self.current is None:
            result = self._update_search(best, gray)
            self.previous_gray = gray.copy()
            return result

        associated = self._associated_detection(detections)
        if associated is not None and associated.confidence >= self.config.track_confidence:
            replacement = self._update_challenger(best, associated)
            if replacement is not None:
                associated = replacement
            self.current = associated
            self.state = A4TrackState.TRACKING
            self.miss_count = 0
            self.flow_inliers = 0
            self.flow_coverage = 0.0
            self.flow_points = self._select_flow_points(gray, associated)
            result = A4TrackResult(associated, self.state, True, False, 0, 0)
        elif (
            predicted is not None
            and predicted.confidence >= self.config.track_confidence
            and self.flow_inliers >= self.config.min_flow_points
            and self.flow_coverage >= 0.55
        ):
            self.current = predicted
            self.state = A4TrackState.TRACKING
            self.miss_count = 0
            self.flow_points = tracked_points
            result = A4TrackResult(
                predicted,
                self.state,
                True,
                True,
                0,
                self.flow_inliers,
            )
        elif predicted is not None and predicted.confidence >= self.config.occluded_confidence:
            self.current = predicted
            self.state = A4TrackState.OCCLUDED
            self.miss_count += 1
            self.flow_points = tracked_points
            result = A4TrackResult(
                predicted,
                self.state,
                False,
                True,
                self.miss_count,
                self.flow_inliers,
            )
        else:
            self.miss_count += 1
            if self.miss_count <= self.config.occlusion_hold_frames:
                self.state = A4TrackState.OCCLUDED
                result = A4TrackResult(
                    self.current,
                    self.state,
                    False,
                    True,
                    self.miss_count,
                    self.flow_inliers,
                )
            else:
                self.state = A4TrackState.LOST
                lost = A4TrackResult(None, self.state, False, False, self.miss_count, self.flow_inliers)
                self.reset()
                result = lost

        self.previous_gray = gray.copy()
        return result

    def _update_search(self, best: Optional[A4Detection], gray: np.ndarray) -> A4TrackResult:
        self.search_preview = best
        ring_support = 0 if best is None else sum(value >= 0.12 for value in best.scores.ring_scores)
        if (
            best is None
            or best.confidence < self.config.acquire_confidence
            or best.scores.visible_sides < 3
            or best.scores.edge < 0.55
            or best.scores.black_band < 0.55
            or (self.require_red_rings and best.scores.red_rings < 0.25)
            or (self.require_red_rings and ring_support < 2)
        ):
            self.pending = None
            self.pending_count = 0
            self.state = A4TrackState.SEARCH
            return A4TrackResult(best, self.state, False, False, 0, 0)

        if self.pending is not None and _detections_match(self.pending, best, gray.shape[1], gray.shape[0]):
            self.pending_count += 1
        else:
            self.pending = best
            self.pending_count = 1

        if self.pending_count < self.config.acquire_confirm_frames:
            return A4TrackResult(best, A4TrackState.SEARCH, False, False, 0, 0)

        self.current = best
        self.state = A4TrackState.ACQUIRED
        self.miss_count = 0
        self.flow_points = self._select_flow_points(gray, best)
        self.pending = None
        self.pending_count = 0
        self.search_preview = None
        return A4TrackResult(best, self.state, True, False, 0, 0)

    def _apply_validation_mode(self, detection: A4Detection) -> A4Detection:
        if self.require_red_rings:
            return detection
        structural = (
            0.42 * detection.scores.edge
            + 0.42 * detection.scores.black_band
            + 0.16 * detection.scores.pose
        )
        confidence = 0.88 * structural + 0.12 * detection.scores.temporal
        scored_rect = replace(detection.rect, score=confidence)
        return replace(
            detection,
            rect=scored_rect,
            confidence=confidence,
            structural_confidence=structural,
        )

    def _associated_detection(self, detections: Sequence[A4Detection]) -> Optional[A4Detection]:
        if self.current is None:
            return detections[0] if detections else None
        for detection in detections:
            if _detections_match(self.current, detection, 1, 1, normalized=True):
                return detection
        return None

    def _update_challenger(
        self,
        best: Optional[A4Detection],
        accepted: A4Detection,
    ) -> Optional[A4Detection]:
        if best is None or best is accepted or best.confidence < accepted.confidence + self.config.switch_margin:
            self.challenger = None
            self.challenger_count = 0
            return None
        if self.challenger is not None and _detections_match(self.challenger, best, 1, 1, normalized=True):
            self.challenger_count += 1
        else:
            self.challenger = best
            self.challenger_count = 1
        if self.challenger_count < self.config.switch_confirm_frames:
            return None
        replacement = self.challenger
        self.challenger = None
        self.challenger_count = 0
        return replacement

    def _predict(
        self,
        frame: np.ndarray,
        gray: np.ndarray,
    ) -> Tuple[Optional[A4Detection], Optional[np.ndarray]]:
        self.flow_inliers = 0
        self.flow_coverage = 0.0
        if self.current is None or self.previous_gray is None or self.flow_points is None:
            return None, None
        if len(self.flow_points) < self.config.min_flow_points:
            return None, None

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            self.flow_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_points is None or status is None:
            return None, None
        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            self.previous_gray,
            next_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if back_points is None or back_status is None:
            return None, None

        forward = next_points.reshape(-1, 2)
        backward = back_points.reshape(-1, 2)
        original = self.flow_points.reshape(-1, 2)
        valid = (status.reshape(-1) > 0) & (back_status.reshape(-1) > 0)
        valid &= np.linalg.norm(original - backward, axis=1) <= 1.8
        old_good = original[valid]
        new_good = forward[valid]
        if len(old_good) < self.config.min_flow_points:
            return None, None

        delta, inlier_mask = cv2.findHomography(old_good, new_good, cv2.RANSAC, 3.0)
        if delta is None or inlier_mask is None:
            return None, None
        inliers = inlier_mask.reshape(-1) > 0
        self.flow_inliers = int(inliers.sum())
        if self.flow_inliers < self.config.min_flow_points:
            return None, None
        inlier_points = new_good[inliers]
        target_bbox = cv2.boundingRect(self.current.quad.astype(np.float32))
        span_x = float(inlier_points[:, 0].max() - inlier_points[:, 0].min())
        span_y = float(inlier_points[:, 1].max() - inlier_points[:, 1].min())
        coverage_x = clamp(span_x / max(float(target_bbox[2]), 1.0), 0.0, 1.0)
        coverage_y = clamp(span_y / max(float(target_bbox[3]), 1.0), 0.0, 1.0)
        self.flow_coverage = min(coverage_x, coverage_y)

        predicted_quad = cv2.perspectiveTransform(
            self.current.quad.reshape(1, 4, 2).astype(np.float32),
            delta,
        )[0]
        predicted_rect = DetectedRect(
            center=quad_center(predicted_quad),
            box=predicted_quad,
            area=float(abs(cv2.contourArea(predicted_quad))),
            pass_index=-3,
        )
        if self.frame_count % self.config.local_validate_interval == 0:
            predicted = self.detector.evaluate(frame, predicted_rect, self.current)
        else:
            predicted = _propagate_detection(self.current, predicted_rect, delta)
        tracked = new_good[inliers].reshape(-1, 1, 2).astype(np.float32)
        return predicted, tracked

    def _select_flow_points(self, gray: np.ndarray, detection: A4Detection) -> Optional[np.ndarray]:
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.fillConvexPoly(mask, detection.quad.astype(np.int32), 255)
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.config.max_flow_points,
            qualityLevel=0.01,
            minDistance=6,
            mask=mask,
            blockSize=5,
        )
        anchors = np.array(
            [[[x, y]] for x, y, score in detection.edge_points if score >= 0.35],
            dtype=np.float32,
        )
        if points is None:
            points = anchors if anchors.size else None
        elif anchors.size:
            points = np.concatenate((points, anchors), axis=0)
        if points is None:
            return None
        return points[: self.config.max_flow_points].astype(np.float32)


def _propagate_detection(
    previous: A4Detection,
    predicted_rect: DetectedRect,
    delta: np.ndarray,
) -> A4Detection:
    inverse_delta = np.linalg.inv(delta)
    homography = previous.homography @ inverse_delta
    inverse_homography = delta @ previous.inverse_homography
    if previous.edge_points:
        source = np.array(
            [[[x, y] for x, y, _ in previous.edge_points]],
            dtype=np.float32,
        )
        mapped = cv2.perspectiveTransform(source, delta)[0]
        edge_points = tuple(
            (float(point[0]), float(point[1]), previous.edge_points[index][2])
            for index, point in enumerate(mapped)
        )
    else:
        edge_points = ()
    scores = A4FeatureScores(
        edge=previous.scores.edge,
        black_band=previous.scores.black_band,
        red_rings=previous.scores.red_rings,
        pose=_pose_score(predicted_rect.box),
        temporal=1.0,
        side_scores=previous.scores.side_scores,
        ring_scores=previous.scores.ring_scores,
        visible_sides=previous.scores.visible_sides,
    )
    confidence = clamp(previous.confidence * 0.997, 0.0, 1.0)
    rect = DetectedRect(
        center=predicted_rect.center,
        box=predicted_rect.box.astype(np.float32),
        area=predicted_rect.area,
        pass_index=-3,
        score=confidence,
    )
    return A4Detection(
        rect=rect,
        quad=predicted_rect.box.astype(np.float32),
        center=predicted_rect.center,
        area=predicted_rect.area,
        confidence=confidence,
        structural_confidence=previous.structural_confidence,
        homography=homography,
        inverse_homography=inverse_homography,
        canonical_size=previous.canonical_size,
        canonical=previous.canonical,
        scores=scores,
        edge_points=edge_points,
    )


def _scale_quad(quad: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    center = quad.mean(axis=0)
    scaled = quad.copy().astype(np.float32)
    scaled[:, 0] = center[0] + (scaled[:, 0] - center[0]) * scale_x
    scaled[:, 1] = center[1] + (scaled[:, 1] - center[1]) * scale_y
    return scaled


def _quad_inside_reasonable_bounds(quad: np.ndarray, frame_w: int, frame_h: int) -> bool:
    margin_x = frame_w * 0.08
    margin_y = frame_h * 0.08
    return bool(
        np.all(quad[:, 0] >= -margin_x)
        and np.all(quad[:, 0] <= frame_w - 1 + margin_x)
        and np.all(quad[:, 1] >= -margin_y)
        and np.all(quad[:, 1] <= frame_h - 1 + margin_y)
        and cv2.isContourConvex(quad.astype(np.float32))
    )


def _patch_mean(gray: np.ndarray, x: float, y: float, radius: int) -> float:
    ix = int(round(x))
    iy = int(round(y))
    x1 = max(0, ix - radius)
    y1 = max(0, iy - radius)
    x2 = min(gray.shape[1], ix + radius + 1)
    y2 = min(gray.shape[0], iy + radius + 1)
    patch = gray[y1:y2, x1:x2]
    return float(patch.mean()) if patch.size else 0.0


def _robust_support(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values, reverse=True)
    keep = max(1, int(math.ceil(len(ordered) * 0.75)))
    return float(np.mean(ordered[:keep]))


def _pose_score(quad: np.ndarray) -> float:
    ordered = order_quad_points(quad)
    lengths = [float(np.linalg.norm(ordered[(i + 1) % 4] - ordered[i])) for i in range(4)]
    if min(lengths) <= 1.0:
        return 0.0
    opposite = 0.5 * (
        min(lengths[0], lengths[2]) / max(lengths[0], lengths[2])
        + min(lengths[1], lengths[3]) / max(lengths[1], lengths[3])
    )
    observed_ratio = max((lengths[0] + lengths[2]) * 0.5, (lengths[1] + lengths[3]) * 0.5) / max(
        1.0,
        min((lengths[0] + lengths[2]) * 0.5, (lengths[1] + lengths[3]) * 0.5),
    )
    ratio_score = math.exp(-abs(math.log(max(observed_ratio, 1e-3) / math.sqrt(2.0))) / 0.75)
    return clamp(0.65 * opposite + 0.35 * ratio_score, 0.0, 1.0)


def _quick_candidate_score(rect: DetectedRect) -> float:
    quad = order_quad_points(rect.box)
    aspect = _apparent_aspect(quad)
    aspect_score = math.exp(-abs(math.log(max(aspect, 1e-3) / math.sqrt(2.0))) / 0.55)
    pose = _pose_score(quad)
    return 0.65 * pose + 0.35 * aspect_score


def _apparent_aspect(quad: np.ndarray) -> float:
    ordered = order_quad_points(quad)
    horizontal = 0.5 * (
        float(np.linalg.norm(ordered[1] - ordered[0]))
        + float(np.linalg.norm(ordered[2] - ordered[3]))
    )
    vertical = 0.5 * (
        float(np.linalg.norm(ordered[2] - ordered[1]))
        + float(np.linalg.norm(ordered[3] - ordered[0]))
    )
    return max(horizontal, vertical) / max(1.0, min(horizontal, vertical))


def _temporal_score(
    quad: np.ndarray,
    previous: Optional[A4Detection],
    frame_w: int,
    frame_h: int,
) -> float:
    if previous is None:
        return 0.5
    center = quad_center(quad)
    center_distance = math.hypot(center[0] - previous.center[0], center[1] - previous.center[1])
    center_score = 1.0 - clamp(center_distance / max(math.hypot(frame_w, frame_h) * 0.20, 1.0), 0.0, 1.0)
    area = float(abs(cv2.contourArea(quad)))
    area_score = min(area, previous.area) / max(area, previous.area, 1.0)
    corner_distance = float(np.mean(np.linalg.norm(order_quad_points(quad) - previous.quad, axis=1)))
    corner_score = 1.0 - clamp(corner_distance / max(math.sqrt(previous.area) * 0.35, 1.0), 0.0, 1.0)
    return 0.45 * center_score + 0.25 * area_score + 0.30 * corner_score


def _detections_match(
    first: A4Detection,
    second: A4Detection,
    frame_w: int,
    frame_h: int,
    normalized: bool = False,
) -> bool:
    if normalized:
        scale = max(math.sqrt(first.area), math.sqrt(second.area), 1.0)
        max_distance = scale * 0.65
    else:
        max_distance = math.hypot(frame_w, frame_h) * 0.15
    center_distance = math.hypot(first.center[0] - second.center[0], first.center[1] - second.center[1])
    area_ratio = min(first.area, second.area) / max(first.area, second.area, 1.0)
    return center_distance <= max_distance and area_ratio >= 0.45


def _bbox_iou(first: Tuple[int, int, int, int], second: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return float(intersection) / float(max(union, 1))
