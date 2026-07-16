from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .a4_target import order_quad_points, quad_center
from .e25_target_model import DEFAULT_E25_TARGET_MODEL, E25TargetModel


@dataclass(frozen=True)
class EdgeSample:
    side: int
    fraction: float
    point: Tuple[float, float]
    score: float
    inner_depth: float


@dataclass(frozen=True)
class SideMeasurement:
    side: int
    line: Optional[np.ndarray]
    samples: Tuple[EdgeSample, ...]
    confidence: float
    coverage: float
    visible: bool


@dataclass(frozen=True)
class EdgeMeasurement:
    quad: Optional[np.ndarray]
    sides: Tuple[SideMeasurement, ...]
    visible_sides: int
    quality: float
    supported_points: int
    edge_points: Tuple[Tuple[float, float, float], ...]


@dataclass(frozen=True)
class _SideGeometry:
    side: int
    start: np.ndarray
    vector: np.ndarray
    normal: np.ndarray
    length: float
    tape_depth: float


@dataclass(frozen=True)
class E25EdgeConfig:
    samples_per_side: int = 16
    min_sample_score: float = 0.28
    min_line_points: int = 6
    min_side_coverage: float = 0.48
    min_side_confidence: float = 0.34
    profile_step_px: float = 1.0


class E25EdgeTracker:
    def __init__(
        self,
        config: E25EdgeConfig = E25EdgeConfig(),
        model: E25TargetModel = DEFAULT_E25_TARGET_MODEL,
    ) -> None:
        self.config = config
        self.model = model

    def measure(
        self,
        frame: np.ndarray,
        predicted_quad: np.ndarray,
        canonical_size: Tuple[int, int],
        search_scale: float = 1.0,
    ) -> EdgeMeasurement:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        quad = order_quad_points(predicted_quad)
        center = np.asarray(quad_center(quad), dtype=np.float32)
        side_lengths = np.array(
            [np.linalg.norm(quad[(index + 1) % 4] - quad[index]) for index in range(4)],
            dtype=np.float32,
        )

        geometries: List[_SideGeometry] = []
        for side_index in range(4):
            start = quad[side_index]
            end = quad[(side_index + 1) % 4]
            side_vector = end - start
            side_length = max(float(np.linalg.norm(side_vector)), 1.0)
            tangent = side_vector / side_length
            midpoint = 0.5 * (start + end)
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
            if float(np.dot(normal, center - midpoint)) < 0.0:
                normal *= -1.0

            adjacent_length = 0.5 * (
                side_lengths[(side_index - 1) % 4]
                + side_lengths[(side_index + 1) % 4]
            )
            tape_depth = max(
                3.0,
                float(adjacent_length)
                * self.model.tape_ratio_for_normal(canonical_size, side_index),
            )
            geometries.append(
                _SideGeometry(
                    side=side_index,
                    start=start,
                    vector=side_vector,
                    normal=normal,
                    length=side_length,
                    tape_depth=tape_depth,
                )
            )

        samples_by_side = self._measure_all_side_samples(
            gray,
            geometries,
            search_scale,
        )
        sides: List[SideMeasurement] = []
        all_points: List[Tuple[float, float, float]] = []
        for geometry, samples in zip(geometries, samples_by_side):
            side_measurement = self._fit_side(
                geometry.side,
                samples,
                geometry.length,
            )
            sides.append(side_measurement)
            all_points.extend((sample.point[0], sample.point[1], sample.score) for sample in samples)

        visible_sides = sum(side.visible for side in sides)
        refined_quad = self._quad_from_sides(quad, sides) if visible_sides >= 3 else None
        if refined_quad is not None and not self._quad_is_reasonable(quad, refined_quad):
            refined_quad = None

        confidences = sorted((side.confidence for side in sides), reverse=True)
        quality = 0.0
        if confidences:
            quality = 0.75 * float(np.mean(confidences[:3])) + 0.25 * float(np.mean(confidences))
            quality *= min(1.0, visible_sides / 3.0)
        supported_points = sum(
            sample.score >= self.config.min_sample_score
            for side in sides
            for sample in side.samples
        )
        return EdgeMeasurement(
            quad=refined_quad,
            sides=tuple(sides),
            visible_sides=visible_sides,
            quality=max(0.0, min(1.0, quality)),
            supported_points=int(supported_points),
            edge_points=tuple(all_points),
        )

    def _measure_all_side_samples(
        self,
        gray: np.ndarray,
        geometries: Sequence[_SideGeometry],
        search_scale: float,
    ) -> List[List[EdgeSample]]:
        search_scale = max(1.0, min(2.5, float(search_scale)))
        expansion = search_scale - 1.0
        fractions = np.linspace(
            0.08,
            0.92,
            self.config.samples_per_side,
            dtype=np.float32,
        )
        outside_values = [
            max(6.0, (0.70 + 0.90 * expansion) * geometry.tape_depth)
            for geometry in geometries
        ]
        inside_values = [
            max(14.0, (2.10 + 0.90 * expansion) * geometry.tape_depth)
            for geometry in geometries
        ]
        offsets = np.arange(
            -max(outside_values),
            max(inside_values) + self.config.profile_step_px,
            self.config.profile_step_px,
            dtype=np.float32,
        )

        predicted_by_side = [
            geometry.start.reshape(1, 2)
            + fractions.reshape(-1, 1) * geometry.vector.reshape(1, 2)
            for geometry in geometries
        ]
        predicted_outer = np.vstack(predicted_by_side)
        normals = np.vstack(
            [
                np.repeat(
                    geometry.normal.reshape(1, 2),
                    self.config.samples_per_side,
                    axis=0,
                )
                for geometry in geometries
            ]
        )
        profile_points = (
            predicted_outer[:, None, :]
            + offsets.reshape(1, -1, 1) * normals[:, None, :]
        )
        profiles = cv2.remap(
            gray,
            profile_points[:, :, 0],
            profile_points[:, :, 1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).astype(np.float32)
        profiles = cv2.blur(
            profiles,
            (5, 1),
            borderType=cv2.BORDER_REPLICATE,
        )

        samples_by_side: List[List[EdgeSample]] = []
        count = self.config.samples_per_side
        for index, geometry in enumerate(geometries):
            start = index * count
            end = start + count
            local_columns = (
                (offsets >= -outside_values[index])
                & (offsets <= inside_values[index])
            )
            local_points = profile_points[start:end, local_columns]
            valid_rows = (
                np.all(local_points[:, :, 0] >= 1.0, axis=1)
                & np.all(local_points[:, :, 1] >= 1.0, axis=1)
                & np.all(local_points[:, :, 0] < gray.shape[1] - 1.0, axis=1)
                & np.all(local_points[:, :, 1] < gray.shape[0] - 1.0, axis=1)
            )
            samples_by_side.append(
                self._samples_from_profiles(
                    geometry,
                    fractions,
                    predicted_outer[start:end],
                    profiles[start:end],
                    offsets,
                    valid_rows,
                    expansion,
                )
            )
        return samples_by_side

    def _samples_from_profiles(
        self,
        geometry: _SideGeometry,
        fractions: np.ndarray,
        predicted_outer: np.ndarray,
        profiles: np.ndarray,
        offsets: np.ndarray,
        valid_rows: np.ndarray,
        expansion: float,
    ) -> List[EdgeSample]:
        tape_depth = geometry.tape_depth
        separation = max(2, int(round(0.18 * tape_depth)))
        if profiles.shape[1] <= separation + 4:
            return self._empty_samples(
                geometry.side,
                fractions,
                predicted_outer,
                tape_depth,
            )
        contrast = profiles[:, separation:] - profiles[:, :-separation]
        transition_offsets = 0.5 * (offsets[separation:] + offsets[:-separation])
        valid_columns = (
            (transition_offsets >= (0.35 - 0.85 * expansion) * tape_depth)
            & (transition_offsets <= (1.75 + 0.85 * expansion) * tape_depth)
        )
        if not np.any(valid_columns):
            return self._empty_samples(
                geometry.side,
                fractions,
                predicted_outer,
                tape_depth,
            )
        candidate_indices = np.flatnonzero(valid_columns)
        best_indices = candidate_indices[
            np.argmax(contrast[:, valid_columns], axis=1)
        ]
        rows = np.arange(len(fractions))
        inner_depths = transition_offsets[best_indices]
        dark = profiles[rows, best_indices]
        bright = profiles[
            rows,
            np.minimum(best_indices + separation, profiles.shape[1] - 1),
        ]
        delta = bright - dark
        contrast_scores = np.clip((delta - 5.0) / 48.0, 0.0, 1.0)
        relative_scores = np.clip(
            (delta / np.maximum(bright, 40.0) - 0.05) / 0.32,
            0.0,
            1.0,
        )
        darkness_scores = np.clip((delta - 2.0) / 42.0, 0.0, 1.0)
        depth_scores = np.exp(
            -np.abs(inner_depths - tape_depth) / max(tape_depth, 2.0)
        )
        scores = np.clip(
            0.55 * np.maximum(contrast_scores, relative_scores)
            + 0.20 * darkness_scores
            + 0.25 * depth_scores,
            0.0,
            1.0,
        )
        measured_outer = predicted_outer + (
            (inner_depths - tape_depth).reshape(-1, 1)
            * geometry.normal.reshape(1, 2)
        )

        samples: List[EdgeSample] = []
        for index, fraction in enumerate(fractions):
            outer = predicted_outer[index]
            if not valid_rows[index]:
                samples.append(
                    EdgeSample(
                        side=geometry.side,
                        fraction=float(fraction),
                        point=(float(outer[0]), float(outer[1])),
                        score=0.0,
                        inner_depth=tape_depth,
                    )
                )
                continue
            point = measured_outer[index]
            samples.append(
                EdgeSample(
                    side=geometry.side,
                    fraction=float(fraction),
                    point=(float(point[0]), float(point[1])),
                    score=float(scores[index]),
                    inner_depth=float(inner_depths[index]),
                )
            )
        return samples

    @staticmethod
    def _empty_samples(
        side_index: int,
        fractions: np.ndarray,
        predicted_outer: np.ndarray,
        tape_depth: float,
    ) -> List[EdgeSample]:
        return [
            EdgeSample(
                side=side_index,
                fraction=float(fraction),
                point=(float(outer[0]), float(outer[1])),
                score=0.0,
                inner_depth=tape_depth,
            )
            for fraction, outer in zip(fractions, predicted_outer)
        ]

    def _fit_side(
        self,
        side_index: int,
        samples: Sequence[EdgeSample],
        side_length: float,
    ) -> SideMeasurement:
        supported = [sample for sample in samples if sample.score >= self.config.min_sample_score]
        if len(supported) < self.config.min_line_points:
            return SideMeasurement(side_index, None, tuple(samples), 0.0, 0.0, False)
        points = np.asarray([sample.point for sample in supported], dtype=np.float32)
        line = self._fit_line(points)
        residuals = self._line_residuals(points, line)
        residual_limit = max(1.8, 0.012 * side_length)
        inliers = residuals <= residual_limit
        if int(np.count_nonzero(inliers)) < self.config.min_line_points:
            return SideMeasurement(side_index, None, tuple(samples), 0.0, 0.0, False)
        points = points[inliers]
        line = self._fit_line(points)
        supported_inliers = [sample for sample, keep in zip(supported, inliers) if keep]
        fractions = [sample.fraction for sample in supported_inliers]
        coverage = max(fractions) - min(fractions) if fractions else 0.0
        mean_score = float(np.mean([sample.score for sample in supported_inliers]))
        residual_score = 1.0 - min(
            1.0,
            float(np.median(self._line_residuals(points, line))) / max(residual_limit, 1e-3),
        )
        confidence = 0.55 * mean_score + 0.25 * min(1.0, coverage / 0.70) + 0.20 * residual_score
        visible = (
            coverage >= self.config.min_side_coverage
            and confidence >= self.config.min_side_confidence
        )
        return SideMeasurement(
            side=side_index,
            line=line if visible else None,
            samples=tuple(samples),
            confidence=max(0.0, min(1.0, confidence)),
            coverage=float(coverage),
            visible=visible,
        )

    @staticmethod
    def _fit_line(points: np.ndarray) -> np.ndarray:
        vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_HUBER, 0.0, 0.01, 0.01).reshape(-1)
        normal = np.array([-vy, vx], dtype=np.float64)
        normal /= max(float(np.linalg.norm(normal)), 1e-9)
        c = -float(normal[0] * x0 + normal[1] * y0)
        return np.array([normal[0], normal[1], c], dtype=np.float64)

    @staticmethod
    def _line_residuals(points: np.ndarray, line: np.ndarray) -> np.ndarray:
        return np.abs(points[:, 0] * line[0] + points[:, 1] * line[1] + line[2])

    def _quad_from_sides(
        self,
        predicted_quad: np.ndarray,
        sides: Sequence[SideMeasurement],
    ) -> Optional[np.ndarray]:
        predicted_lines = [
            self._line_from_points(predicted_quad[index], predicted_quad[(index + 1) % 4])
            for index in range(4)
        ]
        lines = [
            side.line if side.visible and side.line is not None else predicted_lines[index]
            for index, side in enumerate(sides)
        ]
        corners = (
            self._intersect(lines[3], lines[0]),
            self._intersect(lines[0], lines[1]),
            self._intersect(lines[1], lines[2]),
            self._intersect(lines[2], lines[3]),
        )
        if any(corner is None for corner in corners):
            return None
        return order_quad_points(np.asarray(corners, dtype=np.float32))

    @staticmethod
    def _line_from_points(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        direction = np.asarray(second, dtype=np.float64) - np.asarray(first, dtype=np.float64)
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        normal /= max(float(np.linalg.norm(normal)), 1e-9)
        return np.array([normal[0], normal[1], -float(np.dot(normal, first))], dtype=np.float64)

    @staticmethod
    def _intersect(first: np.ndarray, second: np.ndarray) -> Optional[np.ndarray]:
        matrix = np.array([[first[0], first[1]], [second[0], second[1]]], dtype=np.float64)
        determinant = float(np.linalg.det(matrix))
        if abs(determinant) < 1e-7:
            return None
        values = np.linalg.solve(matrix, np.array([-first[2], -second[2]], dtype=np.float64))
        return values.astype(np.float32)

    @staticmethod
    def _quad_is_reasonable(predicted: np.ndarray, measured: np.ndarray) -> bool:
        predicted_area = abs(float(cv2.contourArea(predicted.astype(np.float32))))
        measured_area = abs(float(cv2.contourArea(measured.astype(np.float32))))
        if predicted_area <= 1.0 or measured_area <= 1.0 or not cv2.isContourConvex(measured.astype(np.float32)):
            return False
        area_ratio = min(predicted_area, measured_area) / max(predicted_area, measured_area)
        center_distance = math.hypot(
            quad_center(predicted)[0] - quad_center(measured)[0],
            quad_center(predicted)[1] - quad_center(measured)[1],
        )
        scale = max(math.sqrt(predicted_area), 1.0)
        return area_ratio >= 0.58 and center_distance <= max(30.0, 0.16 * scale)
