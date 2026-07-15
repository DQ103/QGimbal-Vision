from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .a4_target import (
    A4Detection,
    A4TargetConfig,
    A4TargetDetector,
    A4TrackState,
    order_quad_points,
    quad_center,
)
from .e25_edge_tracker import E25EdgeConfig, E25EdgeTracker, EdgeMeasurement
from .e25_motion import AlphaBetaMotionPredictor, MotionMode
from .e25_target_model import DEFAULT_E25_TARGET_MODEL, E25TargetModel
from .rect_detect import DetectedRect


@dataclass(frozen=True)
class E25Confidence:
    identity: float
    measurement: float
    tracking: float
    control_valid: bool


@dataclass(frozen=True)
class E25TrackResult:
    detection: Optional[A4Detection]
    state: A4TrackState
    current: bool
    predicted: bool
    miss_count: int
    flow_inliers: int
    confidence: E25Confidence
    edge_measurement: Optional[EdgeMeasurement]


@dataclass(frozen=True)
class E25PipelineConfig:
    min_area_ratio: float = 0.015
    max_area_ratio: float = 0.70
    acquire_confidence: float = 0.72
    track_identity_confidence: float = 0.48
    acquire_confirm_frames: int = 3
    occlusion_hold_frames: int = 12
    search_interval: int = 6
    static_global_interval: int = 15
    dynamic_global_interval: int = 5
    structural_validate_interval: int = 5
    min_measurement_quality: float = 0.38


class E25VisionPipeline:
    def __init__(
        self,
        config: E25PipelineConfig = E25PipelineConfig(),
        model: E25TargetModel = DEFAULT_E25_TARGET_MODEL,
        require_red_rings: bool = False,
    ) -> None:
        self.config = config
        self.model = model
        self.detector = A4TargetDetector(
            A4TargetConfig(
                min_area_ratio=config.min_area_ratio,
                max_area_ratio=config.max_area_ratio,
                acquire_confidence=config.acquire_confidence,
                track_confidence=config.track_identity_confidence,
            )
        )
        self.edge_tracker = E25EdgeTracker(E25EdgeConfig(), model)
        self.motion = AlphaBetaMotionPredictor()
        self.require_red_rings = bool(require_red_rings)
        self.state = A4TrackState.SEARCH
        self.current: Optional[A4Detection] = None
        self.identity_confidence = 0.0
        self.measurement_quality = 0.0
        self.tracking_confidence = 0.0
        self.pending: Optional[A4Detection] = None
        self.pending_count = 0
        self.search_preview: Optional[A4Detection] = None
        self.miss_count = 0
        self.frame_count = 0

    def reset(self) -> None:
        self.state = A4TrackState.SEARCH
        self.current = None
        self.identity_confidence = 0.0
        self.measurement_quality = 0.0
        self.tracking_confidence = 0.0
        self.pending = None
        self.pending_count = 0
        self.search_preview = None
        self.miss_count = 0
        self.frame_count = 0
        self.motion.reset()

    def set_require_red_rings(self, require_red_rings: bool) -> None:
        require_red_rings = bool(require_red_rings)
        if require_red_rings != self.require_red_rings:
            self.require_red_rings = require_red_rings
            self.reset()

    def needs_global_detection(self, frame_index: int) -> bool:
        if self.current is None or self.state in (A4TrackState.SEARCH, A4TrackState.LOST):
            return (frame_index - 1) % self.config.search_interval == 0
        interval = (
            self.config.dynamic_global_interval
            if self.motion.mode == MotionMode.DYNAMIC
            else self.config.static_global_interval
        )
        return frame_index % interval == 0

    def update(
        self,
        frame: np.ndarray,
        candidates: Sequence[DetectedRect],
        detection_cycle: bool = True,
        dt: float = 1.0 / 30.0,
    ) -> E25TrackResult:
        self.frame_count += 1
        global_detections = []
        if detection_cycle:
            global_detections = [
                self._rescore_detection(detection)
                for detection in self.detector.detect(frame, candidates, self.current)
            ]

        if self.current is None:
            return self._update_search(global_detections[0] if global_detections else None)

        predicted_quad = self.motion.predict_quad(self.current.quad, dt)
        base = self.current
        associated = self._associated_detection(global_detections, predicted_quad)
        if associated is not None and associated.confidence >= self.config.track_identity_confidence:
            alpha = 0.55 if self.motion.mode == MotionMode.DYNAMIC else 0.30
            blended_quad = (
                (1.0 - alpha) * order_quad_points(predicted_quad)
                + alpha * order_quad_points(associated.quad)
            ).astype(np.float32)
            evaluated = self.detector.evaluate(
                frame,
                DetectedRect(
                    center=quad_center(blended_quad),
                    box=blended_quad,
                    area=float(abs(cv2.contourArea(blended_quad))),
                    pass_index=-3,
                ),
                self.current,
            )
            if evaluated is not None:
                base = self._rescore_detection(evaluated)
                predicted_quad = base.quad
                self.identity_confidence = max(self.identity_confidence * 0.75, base.confidence)

        edge_measurement = self.edge_tracker.measure(
            frame,
            predicted_quad,
            base.canonical_size,
        )
        if (
            edge_measurement.quad is not None
            and edge_measurement.visible_sides >= 3
            and edge_measurement.quality >= self.config.min_measurement_quality
        ):
            measured = _reproject_detection(
                base,
                edge_measurement.quad,
                edge_measurement.edge_points,
                edge_measurement.visible_sides,
            )
            if self.frame_count % self.config.structural_validate_interval == 0:
                validated = self.detector.evaluate(frame, measured.rect, self.current)
                if validated is not None:
                    validated = self._rescore_detection(validated)
                    if validated.confidence >= self.config.track_identity_confidence:
                        measured = _reproject_detection(
                            validated,
                            edge_measurement.quad,
                            edge_measurement.edge_points,
                            edge_measurement.visible_sides,
                        )
                        self.identity_confidence = 0.70 * self.identity_confidence + 0.30 * validated.confidence
                    else:
                        self.identity_confidence *= 0.94
                else:
                    self.identity_confidence *= 0.94
            else:
                self.identity_confidence *= 0.998

            estimate = self.motion.update(
                measured.center,
                dt,
                quality=edge_measurement.quality,
            )
            self.measurement_quality = edge_measurement.quality
            self.tracking_confidence = _tracking_quality(
                estimate.residual,
                measured.area,
                edge_measurement.visible_sides,
            )
            control_valid = (
                self.identity_confidence >= self.config.track_identity_confidence
                and self.measurement_quality >= 0.45
                and self.tracking_confidence >= 0.52
                and edge_measurement.visible_sides >= 3
            )
            measured = _with_combined_confidence(
                measured,
                self.identity_confidence,
                self.measurement_quality,
                self.tracking_confidence,
            )
            self.current = measured
            self.state = A4TrackState.TRACKING
            self.miss_count = 0
            return self._result(measured, True, False, edge_measurement, control_valid)

        self.miss_count += 1
        self.identity_confidence *= 0.985
        self.measurement_quality *= 0.70
        self.tracking_confidence *= 0.82
        if self.miss_count <= self.config.occlusion_hold_frames:
            predicted = _reproject_detection(base, predicted_quad, (), 0)
            predicted = _with_combined_confidence(
                predicted,
                self.identity_confidence,
                self.measurement_quality,
                self.tracking_confidence,
            )
            self.current = predicted
            self.state = A4TrackState.OCCLUDED
            return self._result(predicted, False, True, edge_measurement, False)

        result = E25TrackResult(
            detection=None,
            state=A4TrackState.LOST,
            current=False,
            predicted=False,
            miss_count=self.miss_count,
            flow_inliers=0,
            confidence=E25Confidence(
                self.identity_confidence,
                self.measurement_quality,
                self.tracking_confidence,
                False,
            ),
            edge_measurement=edge_measurement,
        )
        self.reset()
        return result

    def _update_search(self, best: Optional[A4Detection]) -> E25TrackResult:
        self.search_preview = best
        if best is None or not self._acquisition_ok(best):
            self.pending = None
            self.pending_count = 0
            self.state = A4TrackState.SEARCH
            return self._result(best, False, False, None, False)

        if self.pending is not None and _detections_match(self.pending, best):
            self.pending_count += 1
        else:
            self.pending = best
            self.pending_count = 1
        if self.pending_count < self.config.acquire_confirm_frames:
            return self._result(best, False, False, None, False)

        self.current = best
        self.identity_confidence = best.confidence
        self.measurement_quality = best.scores.edge
        self.tracking_confidence = 0.75
        self.motion.reset(best.center)
        self.state = A4TrackState.ACQUIRED
        self.miss_count = 0
        self.pending = None
        self.pending_count = 0
        self.search_preview = None
        return self._result(best, True, False, None, False)

    def _rescore_detection(self, detection: A4Detection) -> A4Detection:
        if self.require_red_rings:
            return detection
        structural = (
            0.42 * detection.scores.edge
            + 0.42 * detection.scores.black_band
            + 0.16 * detection.scores.pose
        )
        confidence = 0.88 * structural + 0.12 * detection.scores.temporal
        return replace(
            detection,
            rect=replace(detection.rect, score=confidence),
            structural_confidence=structural,
            confidence=confidence,
        )

    def _acquisition_ok(self, detection: A4Detection) -> bool:
        if detection.confidence < self.config.acquire_confidence:
            return False
        if detection.scores.visible_sides < 3:
            return False
        if detection.scores.edge < 0.55 or detection.scores.black_band < 0.50:
            return False
        if self.require_red_rings and detection.scores.red_rings < 0.25:
            return False
        samples = max(1, len(detection.edge_points) // 4)
        strong_sides = 0
        for side_index in range(4):
            side = detection.edge_points[side_index * samples : (side_index + 1) * samples]
            coverage = sum(score >= 0.30 for _, _, score in side) / max(len(side), 1)
            strong_sides += coverage >= 0.55
        return strong_sides >= 3

    def _associated_detection(
        self,
        detections: Sequence[A4Detection],
        predicted_quad: np.ndarray,
    ) -> Optional[A4Detection]:
        predicted_center = quad_center(predicted_quad)
        predicted_area = abs(float(cv2.contourArea(predicted_quad.astype(np.float32))))
        scale = max(math.sqrt(predicted_area), 1.0)
        for detection in detections:
            center_distance = math.hypot(
                detection.center[0] - predicted_center[0],
                detection.center[1] - predicted_center[1],
            )
            area_ratio = min(predicted_area, detection.area) / max(predicted_area, detection.area, 1.0)
            corner_rms = float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (order_quad_points(predicted_quad) - order_quad_points(detection.quad)) ** 2,
                            axis=1,
                        )
                    )
                )
            )
            if (
                center_distance <= max(24.0, 0.18 * scale)
                and area_ratio >= 0.58
                and corner_rms <= max(28.0, 0.20 * scale)
            ):
                return detection
        return None

    def _result(
        self,
        detection: Optional[A4Detection],
        current: bool,
        predicted: bool,
        edge_measurement: Optional[EdgeMeasurement],
        control_valid: bool,
    ) -> E25TrackResult:
        return E25TrackResult(
            detection=detection,
            state=self.state,
            current=current,
            predicted=predicted,
            miss_count=self.miss_count,
            flow_inliers=0 if edge_measurement is None else edge_measurement.supported_points,
            confidence=E25Confidence(
                identity=max(0.0, min(1.0, self.identity_confidence)),
                measurement=max(0.0, min(1.0, self.measurement_quality)),
                tracking=max(0.0, min(1.0, self.tracking_confidence)),
                control_valid=bool(control_valid),
            ),
            edge_measurement=edge_measurement,
        )


def _reproject_detection(
    detection: A4Detection,
    quad: np.ndarray,
    edge_points: Sequence[Tuple[float, float, float]],
    visible_sides: int,
) -> A4Detection:
    quad = order_quad_points(quad)
    width, height = detection.canonical_size
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(quad, destination)
    inverse_homography = np.linalg.inv(homography)
    center = quad_center(quad)
    area = abs(float(cv2.contourArea(quad)))
    rect = DetectedRect(
        center=center,
        box=quad,
        area=area,
        pass_index=-4,
        score=detection.confidence,
    )
    scores = replace(detection.scores, visible_sides=int(visible_sides))
    return replace(
        detection,
        rect=rect,
        quad=quad,
        center=center,
        area=area,
        homography=homography,
        inverse_homography=inverse_homography,
        scores=scores,
        edge_points=tuple(edge_points),
    )


def _with_combined_confidence(
    detection: A4Detection,
    identity: float,
    measurement: float,
    tracking: float,
) -> A4Detection:
    confidence = max(0.0, min(1.0, 0.45 * identity + 0.35 * measurement + 0.20 * tracking))
    return replace(
        detection,
        rect=replace(detection.rect, score=confidence),
        confidence=confidence,
    )


def _tracking_quality(residual: float, area: float, visible_sides: int) -> float:
    scale = max(math.sqrt(area), 1.0)
    residual_score = 1.0 - min(1.0, float(residual) / max(8.0, 0.045 * scale))
    side_score = min(1.0, visible_sides / 4.0)
    return max(0.0, min(1.0, 0.72 * residual_score + 0.28 * side_score))


def _detections_match(first: A4Detection, second: A4Detection) -> bool:
    scale = max(math.sqrt(first.area), math.sqrt(second.area), 1.0)
    center_distance = math.hypot(
        first.center[0] - second.center[0],
        first.center[1] - second.center[1],
    )
    area_ratio = min(first.area, second.area) / max(first.area, second.area, 1.0)
    return center_distance <= max(14.0, 0.10 * scale) and area_ratio >= 0.78
