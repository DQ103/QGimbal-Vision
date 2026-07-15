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
from .e25_coarse_motion import E25CoarseMotionTracker
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
    min_apparent_aspect: float = 1.20
    max_apparent_aspect: float = 2.30
    acquire_confidence: float = 0.72
    track_identity_confidence: float = 0.48
    acquire_confirm_frames: int = 3
    recovery_hold_frames: int = 24
    max_prediction_frames: int = 4
    search_interval: int = 6
    static_global_interval: int = 15
    dynamic_global_interval: int = 2
    structural_validate_interval: int = 5
    dynamic_validate_interval: int = 2
    min_measurement_quality: float = 0.38
    coarse_min_confidence: float = 0.42
    dynamic_edge_search_scale: float = 2.0


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
                min_apparent_aspect=config.min_apparent_aspect,
                max_apparent_aspect=config.max_apparent_aspect,
                acquire_confidence=config.acquire_confidence,
                track_confidence=config.track_identity_confidence,
            )
        )
        self.edge_tracker = E25EdgeTracker(E25EdgeConfig(), model)
        self.coarse_motion = E25CoarseMotionTracker()
        self.motion = AlphaBetaMotionPredictor()
        self.require_red_rings = bool(require_red_rings)
        self.state = A4TrackState.SEARCH
        self.current: Optional[A4Detection] = None
        self.reliable: Optional[A4Detection] = None
        self.coarse_quad: Optional[np.ndarray] = None
        self.prediction_age = 0
        self.identity_confidence = 0.0
        self.measurement_quality = 0.0
        self.tracking_confidence = 0.0
        self.pending: Optional[A4Detection] = None
        self.pending_count = 0
        self.pending_miss_count = 0
        self.search_preview: Optional[A4Detection] = None
        self.miss_count = 0
        self.frame_count = 0

    def reset(self) -> None:
        self.state = A4TrackState.SEARCH
        self.current = None
        self.reliable = None
        self.coarse_quad = None
        self.prediction_age = 0
        self.identity_confidence = 0.0
        self.measurement_quality = 0.0
        self.tracking_confidence = 0.0
        self.pending = None
        self.pending_count = 0
        self.pending_miss_count = 0
        self.search_preview = None
        self.miss_count = 0
        self.frame_count = 0
        self.motion.reset()
        self.coarse_motion.reset()

    def set_require_red_rings(self, require_red_rings: bool) -> None:
        require_red_rings = bool(require_red_rings)
        if require_red_rings != self.require_red_rings:
            self.require_red_rings = require_red_rings
            self.reset()

    def needs_global_detection(self, frame_index: int) -> bool:
        if self.current is None or self.state in (A4TrackState.SEARCH, A4TrackState.LOST):
            return (frame_index - 1) % self.config.search_interval == 0
        if self.state == A4TrackState.OCCLUDED:
            return True
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
                for detection in self.detector.detect(
                    frame,
                    candidates,
                    self.reliable or self.current,
                )
            ]

        if self.current is None:
            search_candidate = next(
                (detection for detection in global_detections if self._acquisition_ok(detection)),
                global_detections[0] if global_detections else None,
            )
            return self._update_search(
                frame,
                search_candidate,
                detection_cycle,
            )

        tracking_frame = (
            frame
            if frame.ndim == 2
            else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        )
        base = self.reliable or self.current
        if self.coarse_quad is None:
            self.coarse_quad = base.quad.copy()
        coarse = self.coarse_motion.estimate(tracking_frame, self.coarse_quad)
        coarse_valid = (
            coarse.quad is not None
            and coarse.confidence >= self.config.coarse_min_confidence
        )
        coarse_motion_active = coarse_valid and (
            math.hypot(coarse.shift[0], coarse.shift[1]) >= 2.5
            or abs(coarse.scale - 1.0) >= 0.012
            or abs(coarse.rotation_deg) >= 0.8
        )
        dynamic_tracking = (
            coarse_motion_active
            or self.motion.mode == MotionMode.DYNAMIC
            or self.state == A4TrackState.OCCLUDED
            or self.prediction_age > 0
        )
        if coarse_valid:
            predicted_quad = coarse.quad
            self.coarse_quad = coarse.quad.copy()
        else:
            prediction_horizon = min(
                self.prediction_age + 1,
                self.config.max_prediction_frames,
            )
            predicted_quad = _bounded_motion_prediction(
                base.quad,
                self.motion,
                dt,
                prediction_horizon,
            )

        associated = self._associated_detection(
            global_detections,
            predicted_quad,
            recovering=dynamic_tracking,
        )
        if associated is not None and dynamic_tracking and not self._recovery_ok(associated):
            associated = None
        if associated is not None and associated.confidence >= self.config.track_identity_confidence:
            alpha = 0.88 if dynamic_tracking else 0.30
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
                self.reliable or self.current,
            )
            if evaluated is not None:
                base = self._rescore_detection(evaluated)
                predicted_quad = base.quad
                self.identity_confidence = max(self.identity_confidence * 0.75, base.confidence)

        edge_measurement = self.edge_tracker.measure(
            tracking_frame,
            predicted_quad,
            base.canonical_size,
            search_scale=(
                self.config.dynamic_edge_search_scale
                if dynamic_tracking
                else 1.0
            ),
        )
        measurement_valid = (
            edge_measurement.quad is not None
            and edge_measurement.visible_sides >= 3
            and edge_measurement.quality >= self.config.min_measurement_quality
        )
        if measurement_valid:
            measured = _reproject_detection(
                base,
                edge_measurement.quad,
                edge_measurement.edge_points,
                edge_measurement.visible_sides,
            )
            measurement_valid = _measurement_is_plausible(
                measured.quad,
                predicted_quad,
                (self.reliable or base).quad,
                dynamic_tracking,
            )

        if measurement_valid:
            validation_ok = True
            validate_now = (
                (
                    dynamic_tracking
                    and self.frame_count % self.config.dynamic_validate_interval == 0
                )
                or self.frame_count % self.config.structural_validate_interval == 0
            )
            if validate_now:
                validated = self.detector.evaluate(
                    frame,
                    measured.rect,
                    self.reliable or self.current,
                )
                if validated is not None:
                    validated = self._rescore_detection(validated)
                    validation_ok = (
                        self._recovery_ok(validated)
                        if dynamic_tracking
                        else validated.confidence >= self.config.track_identity_confidence
                    )
                    if validation_ok:
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
                    validation_ok = False
                    self.identity_confidence *= 0.94
            else:
                self.identity_confidence *= 0.998

            if dynamic_tracking and not validation_ok:
                measurement_valid = False

        if measurement_valid:
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
            self.reliable = measured
            self.coarse_quad = measured.quad.copy()
            self.coarse_motion.reset(tracking_frame, measured.quad)
            self.state = A4TrackState.TRACKING
            self.miss_count = 0
            self.prediction_age = 0
            return self._result(measured, True, False, edge_measurement, control_valid)

        if associated is not None and self._recovery_ok(associated):
            estimate = self.motion.update(
                associated.center,
                dt,
                quality=max(0.55, associated.scores.edge),
            )
            self.identity_confidence = max(
                self.identity_confidence * 0.80,
                associated.confidence,
            )
            self.measurement_quality = associated.scores.edge
            self.tracking_confidence = max(
                0.58,
                _tracking_quality(
                    estimate.residual,
                    associated.area,
                    associated.scores.visible_sides,
                ),
            )
            recovered = _with_combined_confidence(
                associated,
                self.identity_confidence,
                self.measurement_quality,
                self.tracking_confidence,
            )
            self.current = recovered
            self.reliable = recovered
            self.coarse_quad = recovered.quad.copy()
            self.coarse_motion.reset(tracking_frame, recovered.quad)
            self.state = A4TrackState.ACQUIRED
            self.miss_count = 0
            self.prediction_age = 0
            return self._result(recovered, True, False, edge_measurement, False)

        self.miss_count += 1
        self.prediction_age += 1
        self.identity_confidence *= 0.985
        self.measurement_quality *= 0.70
        self.tracking_confidence *= 0.82
        if self.miss_count <= self.config.recovery_hold_frames:
            display_quad = (
                predicted_quad
                if self.prediction_age <= self.config.max_prediction_frames
                else (self.reliable or base).quad
            )
            predicted = _reproject_detection(base, display_quad, (), 0)
            predicted = _with_combined_confidence(
                predicted,
                self.identity_confidence,
                self.measurement_quality,
                self.tracking_confidence,
            )
            self.current = predicted
            self.coarse_quad = display_quad.copy()
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

    def _update_search(
        self,
        frame: np.ndarray,
        best: Optional[A4Detection],
        detection_cycle: bool,
    ) -> E25TrackResult:
        if not detection_cycle:
            return self._result(self.search_preview, False, False, None, False)

        self.search_preview = best
        if best is None or not self._acquisition_ok(best):
            self.pending_miss_count += 1
            if self.pending_miss_count >= 3:
                self.pending = None
                self.pending_count = 0
                self.pending_miss_count = 0
            self.state = A4TrackState.SEARCH
            return self._result(best, False, False, None, False)

        if self.pending is not None and _detections_match(self.pending, best):
            self.pending = best
            self.pending_count += 1
            self.pending_miss_count = 0
        elif (
            self.pending is not None
            and best.confidence < self.pending.confidence + 0.02
            and self.pending_miss_count < 2
        ):
            self.pending_miss_count += 1
            return self._result(best, False, False, None, False)
        else:
            self.pending = best
            self.pending_count = 1
            self.pending_miss_count = 0
        if self.pending_count < self.config.acquire_confirm_frames:
            return self._result(best, False, False, None, False)

        self.current = best
        self.reliable = best
        self.coarse_quad = best.quad.copy()
        self.identity_confidence = best.confidence
        self.measurement_quality = best.scores.edge
        self.tracking_confidence = 0.75
        self.motion.reset(best.center)
        self.coarse_motion.reset(frame, best.quad)
        self.state = A4TrackState.ACQUIRED
        self.miss_count = 0
        self.prediction_age = 0
        self.pending = None
        self.pending_count = 0
        self.pending_miss_count = 0
        self.search_preview = None
        return self._result(best, True, False, None, False)

    def _rescore_detection(self, detection: A4Detection) -> A4Detection:
        if self.require_red_rings:
            return detection
        paper = _paper_surface_score(detection.canonical)
        structural = (
            0.34 * detection.scores.edge
            + 0.34 * detection.scores.black_band
            + 0.12 * detection.scores.pose
            + 0.20 * paper
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
        if _paper_surface_score(detection.canonical) < 0.45:
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
        recovering: bool = False,
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
            center_ratio = 0.52 if recovering else 0.18
            corner_ratio = 0.48 if recovering else 0.20
            min_area_ratio = 0.42 if recovering else 0.58
            if (
                center_distance <= max(24.0, center_ratio * scale)
                and area_ratio >= min_area_ratio
                and corner_rms <= max(28.0, corner_ratio * scale)
            ):
                return detection
        return None

    def _recovery_ok(self, detection: A4Detection) -> bool:
        if detection.confidence < self.config.track_identity_confidence:
            return False
        if detection.scores.visible_sides < 3:
            return False
        if detection.scores.edge < 0.48 or detection.scores.black_band < 0.44:
            return False
        if _paper_surface_score(detection.canonical) < 0.42:
            return False
        if self.require_red_rings and detection.scores.red_rings < 0.20:
            return False
        return True

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


def _bounded_motion_prediction(
    reliable_quad: np.ndarray,
    motion: AlphaBetaMotionPredictor,
    dt: float,
    horizon_frames: int,
) -> np.ndarray:
    reliable_quad = order_quad_points(reliable_quad)
    horizon = max(1, int(horizon_frames))
    shift = motion.velocity.astype(np.float32) * max(float(dt), 0.0) * horizon
    scale = max(math.sqrt(abs(float(cv2.contourArea(reliable_quad)))), 1.0)
    max_shift = max(18.0, 0.38 * scale)
    magnitude = float(np.linalg.norm(shift))
    if magnitude > max_shift:
        shift *= max_shift / max(magnitude, 1e-6)
    return reliable_quad + shift.reshape(1, 2)


def _measurement_is_plausible(
    measured_quad: np.ndarray,
    predicted_quad: np.ndarray,
    reliable_quad: np.ndarray,
    dynamic: bool,
) -> bool:
    measured = order_quad_points(measured_quad)
    predicted = order_quad_points(predicted_quad)
    reliable = order_quad_points(reliable_quad)
    if not cv2.isContourConvex(measured.astype(np.float32)):
        return False
    measured_area = abs(float(cv2.contourArea(measured)))
    predicted_area = abs(float(cv2.contourArea(predicted)))
    reliable_area = abs(float(cv2.contourArea(reliable)))
    if min(measured_area, predicted_area, reliable_area) <= 1.0:
        return False
    area_ratio = min(measured_area, predicted_area) / max(measured_area, predicted_area)
    min_area_ratio = 0.44 if dynamic else 0.62
    if area_ratio < min_area_ratio:
        return False

    scale = max(math.sqrt(reliable_area), 1.0)
    predicted_center = quad_center(predicted)
    measured_center = quad_center(measured)
    reliable_center = quad_center(reliable)
    prediction_error = math.hypot(
        measured_center[0] - predicted_center[0],
        measured_center[1] - predicted_center[1],
    )
    reliable_jump = math.hypot(
        measured_center[0] - reliable_center[0],
        measured_center[1] - reliable_center[1],
    )
    corner_rms = float(
        np.sqrt(np.mean(np.sum((measured - predicted) ** 2, axis=1)))
    )
    prediction_ratio = 0.34 if dynamic else 0.14
    reliable_ratio = 0.62 if dynamic else 0.22
    corner_ratio = 0.42 if dynamic else 0.18
    return (
        prediction_error <= max(20.0, prediction_ratio * scale)
        and reliable_jump <= max(28.0, reliable_ratio * scale)
        and corner_rms <= max(24.0, corner_ratio * scale)
    )


def _detections_match(first: A4Detection, second: A4Detection) -> bool:
    scale = max(math.sqrt(first.area), math.sqrt(second.area), 1.0)
    center_distance = math.hypot(
        first.center[0] - second.center[0],
        first.center[1] - second.center[1],
    )
    area_ratio = min(first.area, second.area) / max(first.area, second.area, 1.0)
    return center_distance <= max(14.0, 0.10 * scale) and area_ratio >= 0.78


def _paper_surface_score(canonical: np.ndarray) -> float:
    height, width = canonical.shape[:2]
    margin_x = max(2, int(round(width * 0.14)))
    margin_y = max(2, int(round(height * 0.14)))
    interior = canonical[margin_y : height - margin_y, margin_x : width - margin_x]
    if interior.size == 0:
        return 0.0
    if interior.ndim == 2:
        median_value = float(np.median(interior))
        return max(0.0, min(1.0, (median_value - 70.0) / 120.0))

    hsv = cv2.cvtColor(interior, cv2.COLOR_BGR2HSV)
    median_saturation = float(np.median(hsv[:, :, 1]))
    median_value = float(np.median(hsv[:, :, 2]))
    neutral_score = 1.0 - max(0.0, min(1.0, (median_saturation - 45.0) / 80.0))
    brightness_score = max(0.0, min(1.0, (median_value - 75.0) / 120.0))
    return max(0.0, min(1.0, 0.72 * neutral_score + 0.28 * brightness_score))
