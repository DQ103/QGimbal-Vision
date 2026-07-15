from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class E25TargetModel:
    width_mm: float = 210.0
    height_mm: float = 297.0
    tape_width_mm: float = 18.0
    ring_radii_mm: Tuple[float, ...] = (20.0, 40.0, 60.0, 80.0, 100.0)

    @property
    def center_mm(self) -> Tuple[float, float]:
        return self.width_mm * 0.5, self.height_mm * 0.5

    def dimensions_for_canonical(self, canonical_size: Tuple[int, int]) -> Tuple[float, float]:
        width, height = canonical_size
        if width <= height:
            return self.width_mm, self.height_mm
        return self.height_mm, self.width_mm

    def canonical_to_mm(
        self,
        point: Tuple[float, float],
        canonical_size: Tuple[int, int],
    ) -> Tuple[float, float]:
        width_mm, height_mm = self.dimensions_for_canonical(canonical_size)
        width, height = canonical_size
        return (
            float(point[0]) * width_mm / max(float(width), 1.0),
            float(point[1]) * height_mm / max(float(height), 1.0),
        )

    def image_to_mm(
        self,
        point: Tuple[float, float],
        homography: np.ndarray,
        canonical_size: Tuple[int, int],
    ) -> Tuple[float, float]:
        source = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        canonical = cv2.perspectiveTransform(source, homography)[0, 0]
        return self.canonical_to_mm((float(canonical[0]), float(canonical[1])), canonical_size)

    def tape_ratio_for_normal(self, canonical_size: Tuple[int, int], side_index: int) -> float:
        width_mm, height_mm = self.dimensions_for_canonical(canonical_size)
        normal_dimension = height_mm if side_index in (0, 2) else width_mm
        return self.tape_width_mm / normal_dimension


DEFAULT_E25_TARGET_MODEL = E25TargetModel()
