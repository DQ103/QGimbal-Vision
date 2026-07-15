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
    occlusion_hold_frames: int = 20
    switch_margin: float = 0.15
    switch_confirm_frames: int = 5
    global_interval: int = 10
    dynamic_global_interval: int = 5
    dynamic_hold_frames: int = 8
    search_interval: int = 6
    local_validate_interval: int = 3
    min_flow_points: int = 12
    max_flow_points: int = 100
    min_apparent_aspect: float = 1.08
    max_apparent_aspect: float = 2.3
    black_candidate_analysis_scale: float = 0.5


@dataclass(frozen=True)
class A4FeatureScores:
    edge: float
    black_band: float
    red_rings: float
    pose: float
    temporal: float
    side_scores: Tuple[float, float, float, float]
    ring_scores: Tuple[float, float, float, float, float]
    band_depths: Tuple[float, float, float, float]
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
    analysis_scale = (
        max(0.25, min(1.0, float(config.black_candidate_analysis_scale)))
        if min(original_w, original_h) >= 480
        else 1.0
    )
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
        compute_red_rings: bool = True,
    ) -> List[A4Detection]:
        detections: List[A4Detection] = []
        max_candidate_area = max((rect.area for rect in candidates), default=1.0)
        ranked = sorted(
            candidates,
            key=lambda rect: _quick_candidate_score(rect, max_candidate_area),
            reverse=True,
        )
        for rect in ranked[: self.config.max_candidates]:
            detection = self.evaluate(
                frame,
                rect,
                previous,
                compute_red_rings=compute_red_rings,
            )
            if detection is not None:
                detections.append(detection)
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections

    def evaluate(
        self,
        frame: np.ndarray,
        rect: DetectedRect,
        previous: Optional[A4Detection] = None,
        compute_red_rings: bool = True,
    ) -> Optional[A4Detection]:
        frame_h, frame_w = frame.shape[:2]
        frame_area = float(frame_w * frame_h)
        if rect.area < frame_area * self.config.min_area_ratio:
            return None
        if rect.area > frame_area * self.config.max_area_ratio:
            return None

        original = order_quad_points(rect.box)
        apparent_aspect = _apparent_aspect(original)
        if not self.config.min_apparent_aspect <= apparent_aspect <= self.config.max_apparent_aspect:
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
        if rect.pass_index == -3:
            scale_variants = ((1.0, 1.0),)
        elif min(horizontal, vertical) < 90.0:
            scale_variants = (
                (1.0, 1.0),
                (1.16, 1.12),
                (1.32, 1.26),
                (1.48, 1.40),
            )
        else:
            scale_variants = ((1.0, 1.0), (1.16, 1.12))
        best: Optional[A4Detection] = None
        for scale_x, scale_y in scale_variants:
            quad = _scale_quad(original, scale_x, scale_y)
            if not _quad_inside_reasonable_bounds(quad, frame_w, frame_h):
                continue
            for canonical_size in canonical_sizes:
                detection = self._evaluate_quad(
                    frame,
                    rect,
                    quad,
                    canonical_size,
                    previous,
                    compute_red_rings=compute_red_rings,
                )
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
        allow_refine: bool = True,
        compute_red_rings: bool = True,
    ) -> Optional[A4Detection]:
        apparent_aspect = _apparent_aspect(quad)
        if not self.config.min_apparent_aspect <= apparent_aspect <= self.config.max_apparent_aspect:
            return None
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
        edge_score, side_scores, band_depths, edge_points = self._edge_support(canonical, inverse)
        black_score = self._black_band_score(canonical, band_depths, side_scores)
        if compute_red_rings:
            red_score, ring_scores = self._red_ring_score(canonical)
        else:
            red_score = 0.0
            ring_scores = (0.0, 0.0, 0.0, 0.0, 0.0)
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
            band_depths=band_depths,
            visible_sides=visible_sides,
        )
        detection = A4Detection(
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
        if not allow_refine or rect.pass_index == -3:
            return detection

        refined_quad = _refine_quad_from_edge_depths(
            quad,
            canonical_size,
            inverse,
            band_depths,
            side_scores,
            tape_width=min(self.config.tape_width, min(width, height) // 5),
        )
        if refined_quad is None:
            return detection
        return _apply_refined_quad(detection, refined_quad)

    def _edge_support(
        self,
        canonical: np.ndarray,
        inverse_homography: np.ndarray,
    ) -> Tuple[
        float,
        Tuple[float, float, float, float],
        Tuple[float, float, float, float],
        Tuple[Tuple[float, float, float], ...],
    ]:
        gray = canonical if canonical.ndim == 2 else cv2.cvtColor(canonical, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        tape = min(self.config.tape_width, min(width, height) // 5)
        max_depth = min(min(width, height) // 3, max(28, tape * 2 + 8))
        depths = np.arange(max_depth, dtype=np.int32)
        fractions = np.linspace(0.10, 0.90, self.config.edge_samples_per_side)
        xs = np.rint(fractions * (width - 1)).astype(np.int32)
        ys = np.rint(fractions * (height - 1)).astype(np.int32)
        smooth = cv2.blur(gray, (5, 5))
        profiles = (
            smooth[depths[:, None], xs[None, :]].T,
            smooth[ys[:, None], (width - 1 - depths)[None, :]],
            smooth[(height - 1 - depths)[:, None], xs[None, :]].T,
            smooth[ys[:, None], depths[None, :]],
        )
        side_values: List[np.ndarray] = []
        side_depths: List[float] = []
        canonical_points: List[Tuple[float, float, float]] = []
        for side_index, profile in enumerate(profiles):
            profile = profile.astype(np.float32)
            separation = min(10, max_depth // 3)
            contrast = profile[:, separation:] - profile[:, :-separation]
            search_start = 1
            search_end = max(search_start + 1, contrast.shape[1] - 2)
            search = contrast[:, search_start:search_end]
            best_indices = np.argmax(search, axis=1) + search_start
            rows = np.arange(len(best_indices))
            dark = profile[rows, best_indices]
            inner_indices = np.minimum(best_indices + separation, max_depth - 1)
            inner = profile[rows, inner_indices]
            inner_edge_depth = best_indices.astype(np.float32) + separation * 0.5
            band_depth = np.clip(inner_edge_depth * 0.5, 1.0, max_depth - 1.0)
            delta = inner - dark
            absolute_contrast = np.clip((delta - 5.0) / 42.0, 0.0, 1.0)
            relative_contrast = np.clip(
                (delta / np.maximum(inner, 40.0) - 0.06) / 0.30,
                0.0,
                1.0,
            )
            contrast_score = np.maximum(absolute_contrast, relative_contrast)
            darkness_score = np.clip((inner - dark - 3.0) / 45.0, 0.0, 1.0)
            support = 0.80 * contrast_score + 0.20 * darkness_score
            side_values.append(support)
            supported_depths = inner_edge_depth[support >= 0.30]
            side_depths.append(
                float(np.median(supported_depths)) if supported_depths.size else float(tape)
            )
            for index, value in enumerate(support):
                depth = float(band_depth[index])
                if side_index == 0:
                    point = (float(xs[index]), depth)
                elif side_index == 1:
                    point = (float(width - 1) - depth, float(ys[index]))
                elif side_index == 2:
                    point = (float(xs[index]), float(height - 1) - depth)
                else:
                    point = (depth, float(ys[index]))
                canonical_points.append((point[0], point[1], float(value)))

        side_scores = tuple(_robust_support(values.tolist()) for values in side_values)
        sorted_sides = sorted(side_scores, reverse=True)
        edge_score = 0.80 * float(np.mean(sorted_sides[:3])) + 0.20 * float(np.mean(sorted_sides))

        source = np.array([[[x, y] for x, y, _ in canonical_points]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(source, inverse_homography)[0]
        edge_points = tuple(
            (float(point[0]), float(point[1]), float(canonical_points[index][2]))
            for index, point in enumerate(mapped)
        )
        return edge_score, side_scores, tuple(side_depths), edge_points

    def _black_band_score(
        self,
        canonical: np.ndarray,
        band_depths: Tuple[float, float, float, float],
        side_scores: Tuple[float, float, float, float],
    ) -> float:
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
        reliable_depths = [
            depth for depth, score in zip(band_depths, side_scores) if score >= 0.32
        ]
        if len(reliable_depths) >= 2:
            mean_depth = float(np.mean(reliable_depths))
            width_consistency = 1.0 - clamp(
                float(np.std(reliable_depths)) / max(mean_depth, 1.0),
                0.0,
                1.0,
            )
        else:
            width_consistency = 0.0
        photometric_score = (
            0.45 * contrast
            + 0.25 * dark_fraction
            + 0.10 * interior_bright
            + 0.20 * width_consistency
        )
        if len(reliable_depths) >= 3:
            edge_band_evidence = 0.55 * float(np.mean(side_scores)) + 0.10 * width_consistency
            return max(photometric_score, edge_band_evidence)
        return photometric_score

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
        self.keyframe: Optional[A4Detection] = None
        self.last_validated_frame = 0
        self.previous_gray: Optional[np.ndarray] = None
        self.flow_points: Optional[np.ndarray] = None
        self.miss_count = 0
        self.flow_inliers = 0
        self.flow_coverage = 0.0
        self.flow_velocity = np.zeros(2, dtype=np.float32)
        self.dynamic_until_frame = 0
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
        self.keyframe = None
        self.last_validated_frame = 0
        self.previous_gray = None
        self.flow_points = None
        self.miss_count = 0
        self.flow_inliers = 0
        self.flow_coverage = 0.0
        self.flow_velocity = np.zeros(2, dtype=np.float32)
        self.dynamic_until_frame = 0
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
        interval = (
            self.config.dynamic_global_interval
            if self.frame_count <= self.dynamic_until_frame
            else self.config.global_interval
        )
        return frame_index % interval == 0

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

        association_reference = (
            predicted
            if predicted is not None and self.frame_count <= self.dynamic_until_frame
            else self.current
        )
        associated = self._associated_detection(detections, association_reference)
        if associated is not None and associated.confidence >= self.config.track_confidence:
            fusion_base = (
                predicted
                if predicted is not None
                and self.flow_inliers >= self.config.min_flow_points
                and self.flow_coverage >= 0.45
                else self.current
            )
            associated = self._fuse_global_detection(frame, associated, fusion_base)
            self.current = associated
            self.keyframe = associated
            self.last_validated_frame = self.frame_count
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
            and (
                self.frame_count % self.config.local_validate_interval == 0
                or self.frame_count - self.last_validated_frame <= self.config.occlusion_hold_frames
            )
        ):
            self.current = predicted
            self.state = A4TrackState.TRACKING
            self.miss_count = 0
            if self.frame_count % self.config.local_validate_interval == 0:
                self.keyframe = predicted
                self.last_validated_frame = self.frame_count
                if (
                    self.flow_inliers < self.config.min_flow_points * 2
                    or self.flow_coverage < 0.65
                ):
                    self.flow_points = self._select_flow_points(gray, predicted)
                else:
                    self.flow_points = tracked_points
            else:
                self.flow_points = tracked_points
            result = A4TrackResult(
                predicted,
                self.state,
                True,
                True,
                0,
                self.flow_inliers,
            )
        elif (
            predicted is not None
            and predicted.confidence >= self.config.occluded_confidence
            and self.frame_count - self.last_validated_frame <= self.config.occlusion_hold_frames
        ):
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
            self.flow_points = None
            self.flow_velocity *= 0.5
            validation_age = self.frame_count - self.last_validated_frame
            if validation_age <= self.config.occlusion_hold_frames:
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
        spatial_support_ok = best is not None and _acquisition_edge_distribution_ok(best)
        self.search_preview = best if spatial_support_ok else None
        ring_support = 0 if best is None else sum(value >= 0.12 for value in best.scores.ring_scores)
        if (
            best is None
            or not spatial_support_ok
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
            return A4TrackResult(self.search_preview, self.state, False, False, 0, 0)

        if self.pending is not None and _acquisition_match(self.pending, best):
            self.pending_count += 1
        else:
            self.pending = best
            self.pending_count = 1

        if self.pending_count < self.config.acquire_confirm_frames:
            return A4TrackResult(best, A4TrackState.SEARCH, False, False, 0, 0)

        self.current = best
        self.keyframe = best
        self.last_validated_frame = self.frame_count
        self.flow_velocity[:] = 0.0
        self.dynamic_until_frame = 0
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

    def _associated_detection(
        self,
        detections: Sequence[A4Detection],
        reference: Optional[A4Detection] = None,
    ) -> Optional[A4Detection]:
        reference = self.current if reference is None else reference
        if reference is None:
            return detections[0] if detections else None
        for detection in detections:
            if _track_innovation_ok(reference, detection):
                return detection
        return None

    def _fuse_global_detection(
        self,
        frame: np.ndarray,
        detection: A4Detection,
        base: Optional[A4Detection],
    ) -> A4Detection:
        if base is None:
            return detection
        alpha = 0.55 if self.frame_count <= self.dynamic_until_frame else 0.25
        blended_quad = (
            (1.0 - alpha) * order_quad_points(base.quad)
            + alpha * order_quad_points(detection.quad)
        ).astype(np.float32)
        blended_rect = DetectedRect(
            center=quad_center(blended_quad),
            box=blended_quad,
            area=float(abs(cv2.contourArea(blended_quad))),
            pass_index=-3,
        )
        evaluated = self.detector.evaluate(frame, blended_rect, base)
        if evaluated is None:
            return base
        return self._apply_validation_mode(evaluated)

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

        initial_points = self.flow_points + self.flow_velocity.reshape(1, 1, 2)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            self.flow_points,
            initial_points,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )
        if next_points is None or status is None:
            return None, None
        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            self.previous_gray,
            next_points,
            self.flow_points.copy(),
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
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

        old_inliers = old_good[inliers]
        flow_vectors = inlier_points - old_inliers
        measured_velocity = np.median(flow_vectors, axis=0).astype(np.float32)
        flow_residual = float(
            np.median(np.linalg.norm(flow_vectors - measured_velocity, axis=1))
        )
        motion_magnitude = float(np.linalg.norm(measured_velocity))
        target_scale = max(math.sqrt(self.current.area), 1.0)
        coherent_motion = (
            self.flow_coverage >= 0.50
            and flow_residual <= max(1.8, 0.018 * target_scale)
        )
        dynamic_motion = (
            coherent_motion
            and motion_magnitude >= max(4.0, 0.015 * target_scale)
        )

        predicted_quad = cv2.perspectiveTransform(
            self.current.quad.reshape(1, 4, 2).astype(np.float32),
            delta,
        )[0]
        if not _flow_motion_ok(
            self.current,
            predicted_quad,
            motion_magnitude=motion_magnitude,
            coherent_motion=dynamic_motion,
        ):
            self.flow_coverage = 0.0
            return None, None
        dynamic_allowance = (
            motion_magnitude * (self.config.local_validate_interval + 1)
            if dynamic_motion
            else 0.0
        )
        if self.keyframe is not None and not _keyframe_drift_ok(
            self.keyframe,
            predicted_quad,
            dynamic_allowance=dynamic_allowance,
        ):
            self.flow_coverage = 0.0
            return None, None
        if coherent_motion:
            self.flow_velocity = (
                0.55 * self.flow_velocity + 0.45 * measured_velocity
            ).astype(np.float32)
        else:
            self.flow_velocity *= 0.5
        if dynamic_motion:
            self.dynamic_until_frame = self.frame_count + self.config.dynamic_hold_frames
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
        border_mask = np.zeros_like(gray, dtype=np.uint8)
        outer = order_quad_points(detection.quad).astype(np.int32)
        center = outer.astype(np.float32).mean(axis=0)
        inner = np.rint(center + (outer.astype(np.float32) - center) * 0.68).astype(np.int32)
        cv2.fillConvexPoly(border_mask, outer, 255)
        cv2.fillConvexPoly(border_mask, inner, 0)

        parts: List[np.ndarray] = []
        samples_per_side = self.config.edge_samples_per_side
        radius = max(7, int(round(math.sqrt(max(detection.area, 1.0)) * 0.025)))
        side_limit = max(4, self.config.max_flow_points // 5)
        for side_index in range(4):
            side_mask = np.zeros_like(gray, dtype=np.uint8)
            start = side_index * samples_per_side
            end = start + samples_per_side
            for x, y, score in detection.edge_points[start:end]:
                if score >= 0.30:
                    cv2.circle(side_mask, (int(round(x)), int(round(y))), radius, 255, -1)
            cv2.bitwise_and(side_mask, border_mask, dst=side_mask)
            side_points = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=side_limit,
                qualityLevel=0.008,
                minDistance=6,
                mask=side_mask,
                blockSize=5,
            )
            if side_points is not None:
                parts.append(side_points)

        filler = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max(20, self.config.max_flow_points // 2),
            qualityLevel=0.01,
            minDistance=6,
            mask=border_mask,
            blockSize=5,
        )
        if filler is not None:
            parts.append(filler)

        existing_count = sum(len(part) for part in parts)
        if existing_count < self.config.min_flow_points * 2:
            fallback: List[Tuple[float, float]] = []
            for side_index in range(4):
                start = side_index * samples_per_side
                side = [point for point in detection.edge_points[start : start + samples_per_side] if point[2] >= 0.45]
                if not side:
                    continue
                for index in np.linspace(0, len(side) - 1, min(2, len(side))).astype(np.int32):
                    fallback.append((side[index][0], side[index][1]))
            if fallback:
                parts.append(np.asarray(fallback, dtype=np.float32).reshape(-1, 1, 2))

        if not parts:
            return None
        combined = np.concatenate(parts, axis=0).reshape(-1, 2)
        rounded = np.rint(combined / 3.0).astype(np.int32)
        _, unique_indices = np.unique(rounded, axis=0, return_index=True)
        combined = combined[np.sort(unique_indices)]
        return combined[: self.config.max_flow_points].reshape(-1, 1, 2).astype(np.float32)


def _apply_refined_quad(detection: A4Detection, refined_quad: np.ndarray) -> A4Detection:
    width, height = detection.canonical_size
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(refined_quad.astype(np.float32), destination)
    inverse_homography = np.linalg.inv(homography)
    edge_points = detection.edge_points
    if edge_points:
        image_points = np.array(
            [[[x, y] for x, y, _ in edge_points]],
            dtype=np.float32,
        )
        canonical_points = cv2.perspectiveTransform(image_points, detection.homography)
        remapped = cv2.perspectiveTransform(canonical_points, inverse_homography)[0]
        edge_points = tuple(
            (float(point[0]), float(point[1]), detection.edge_points[index][2])
            for index, point in enumerate(remapped)
        )
    center = quad_center(refined_quad)
    area = float(abs(cv2.contourArea(refined_quad.astype(np.float32))))
    rect = replace(
        detection.rect,
        center=center,
        box=refined_quad.astype(np.float32),
        area=area,
    )
    return replace(
        detection,
        rect=rect,
        quad=refined_quad.astype(np.float32),
        center=center,
        area=area,
        homography=homography,
        inverse_homography=inverse_homography,
        edge_points=edge_points,
    )


def _refine_quad_from_edge_depths(
    quad: np.ndarray,
    canonical_size: Tuple[int, int],
    inverse_homography: np.ndarray,
    band_depths: Tuple[float, float, float, float],
    side_scores: Tuple[float, float, float, float],
    tape_width: int,
) -> Optional[np.ndarray]:
    width, height = canonical_size
    boundaries = [0.0, float(width - 1), float(height - 1), 0.0]
    corrections = [0.0, 0.0, 0.0, 0.0]
    reliable_sides = 0
    max_correction = max(3.0, tape_width * 0.70)
    for index, (depth, score) in enumerate(zip(band_depths, side_scores)):
        if score < 0.38:
            continue
        correction = clamp(depth - float(tape_width), -max_correction, max_correction)
        corrections[index] = correction
        reliable_sides += 1

    if reliable_sides < 2 or max(abs(value) for value in corrections) < 0.75:
        return None

    top = boundaries[0] + corrections[0]
    right = boundaries[1] - corrections[1]
    bottom = boundaries[2] - corrections[2]
    left = boundaries[3] + corrections[3]
    canonical_outer = np.array(
        [[[left, top], [right, top], [right, bottom], [left, bottom]]],
        dtype=np.float32,
    )
    refined = cv2.perspectiveTransform(canonical_outer, inverse_homography)[0]
    ordered = order_quad_points(quad)
    corner_rms = float(
        np.sqrt(np.mean(np.sum((ordered - order_quad_points(refined)) ** 2, axis=1)))
    )
    scale = max(math.sqrt(abs(float(cv2.contourArea(ordered)))), 1.0)
    if corner_rms > max(18.0, 0.08 * scale):
        return None
    return refined.astype(np.float32)


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
        band_depths=previous.scores.band_depths,
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


def _quick_candidate_score(rect: DetectedRect, max_candidate_area: float) -> float:
    quad = order_quad_points(rect.box)
    aspect = _apparent_aspect(quad)
    aspect_score = math.exp(-abs(math.log(max(aspect, 1e-3) / math.sqrt(2.0))) / 0.55)
    pose = _pose_score(quad)
    area_score = math.sqrt(clamp(rect.area / max(max_candidate_area, 1.0), 0.0, 1.0))
    return 0.50 * pose + 0.25 * aspect_score + 0.25 * area_score


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


def _acquisition_edge_distribution_ok(detection: A4Detection) -> bool:
    samples = max(1, len(detection.edge_points) // 4)
    coverages = []
    for side_index in range(4):
        start = side_index * samples
        side = detection.edge_points[start : start + samples]
        coverage = sum(score >= 0.30 for _, _, score in side) / max(len(side), 1)
        coverages.append(coverage)
    strong_sides = sum(coverage >= 0.55 for coverage in coverages)
    strongest = sorted(coverages, reverse=True)
    return strong_sides >= 3 and float(np.mean(strongest[:3])) >= 0.68


def _acquisition_match(first: A4Detection, second: A4Detection) -> bool:
    scale = max(math.sqrt(first.area), math.sqrt(second.area), 1.0)
    center_distance = math.hypot(
        first.center[0] - second.center[0],
        first.center[1] - second.center[1],
    )
    area_ratio = min(first.area, second.area) / max(first.area, second.area, 1.0)
    corner_rms = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (order_quad_points(first.quad) - order_quad_points(second.quad)) ** 2,
                    axis=1,
                )
            )
        )
    )
    return (
        center_distance <= max(12.0, 0.08 * scale)
        and area_ratio >= 0.80
        and corner_rms <= max(16.0, 0.10 * scale)
    )


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


def _track_innovation_ok(current: A4Detection, candidate: A4Detection) -> bool:
    scale = max(math.sqrt(current.area), 1.0)
    center_distance = math.hypot(
        current.center[0] - candidate.center[0],
        current.center[1] - candidate.center[1],
    )
    area_ratio = min(current.area, candidate.area) / max(current.area, candidate.area, 1.0)
    corner_rms = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (order_quad_points(current.quad) - order_quad_points(candidate.quad)) ** 2,
                    axis=1,
                )
            )
        )
    )
    return (
        center_distance <= max(18.0, 0.12 * scale)
        and area_ratio >= 0.68
        and corner_rms <= max(22.0, 0.14 * scale)
    )


def _flow_motion_ok(
    current: A4Detection,
    predicted_quad: np.ndarray,
    motion_magnitude: float = 0.0,
    coherent_motion: bool = False,
) -> bool:
    scale = max(math.sqrt(current.area), 1.0)
    predicted_center = quad_center(predicted_quad)
    center_distance = math.hypot(
        current.center[0] - predicted_center[0],
        current.center[1] - predicted_center[1],
    )
    predicted_area = float(abs(cv2.contourArea(predicted_quad.astype(np.float32))))
    area_ratio = min(current.area, predicted_area) / max(current.area, predicted_area, 1.0)
    corner_rms = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (order_quad_points(current.quad) - order_quad_points(predicted_quad)) ** 2,
                    axis=1,
                )
            )
        )
    )
    center_limit = max(10.0, 0.045 * scale)
    corner_limit = max(14.0, 0.07 * scale)
    min_area_ratio = 0.82
    if coherent_motion:
        center_limit = min(
            max(30.0, 0.13 * scale),
            max(center_limit, 6.0 + 1.65 * motion_magnitude),
        )
        corner_limit = min(
            max(36.0, 0.16 * scale),
            max(corner_limit, 9.0 + 1.85 * motion_magnitude),
        )
        min_area_ratio = 0.72
    return (
        center_distance <= center_limit
        and area_ratio >= min_area_ratio
        and corner_rms <= corner_limit
    )


def _keyframe_drift_ok(
    keyframe: A4Detection,
    predicted_quad: np.ndarray,
    dynamic_allowance: float = 0.0,
) -> bool:
    scale = max(math.sqrt(keyframe.area), 1.0)
    predicted_center = quad_center(predicted_quad)
    center_distance = math.hypot(
        keyframe.center[0] - predicted_center[0],
        keyframe.center[1] - predicted_center[1],
    )
    predicted_area = float(abs(cv2.contourArea(predicted_quad.astype(np.float32))))
    area_ratio = min(keyframe.area, predicted_area) / max(keyframe.area, predicted_area, 1.0)
    corner_rms = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (order_quad_points(keyframe.quad) - order_quad_points(predicted_quad)) ** 2,
                    axis=1,
                )
            )
        )
    )
    center_limit = max(24.0, 0.16 * scale, dynamic_allowance * 1.25)
    corner_limit = max(30.0, 0.20 * scale, dynamic_allowance * 1.45)
    min_area_ratio = 0.62 if dynamic_allowance > 0.0 else 0.68
    return (
        center_distance <= center_limit
        and area_ratio >= min_area_ratio
        and corner_rms <= corner_limit
    )


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
