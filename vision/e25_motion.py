from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class MotionMode(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"


@dataclass(frozen=True)
class MotionEstimate:
    center: Tuple[float, float]
    velocity: Tuple[float, float]
    speed: float
    residual: float
    mode: MotionMode


class AlphaBetaMotionPredictor:
    def __init__(
        self,
        dynamic_speed_px_s: float = 110.0,
        static_confirm_frames: int = 6,
    ) -> None:
        self.dynamic_speed_px_s = float(dynamic_speed_px_s)
        self.static_confirm_frames = int(static_confirm_frames)
        self.center: Optional[np.ndarray] = None
        self.velocity = np.zeros(2, dtype=np.float32)
        self.mode = MotionMode.STATIC
        self.static_count = self.static_confirm_frames

    def reset(self, center: Optional[Tuple[float, float]] = None) -> None:
        self.center = None if center is None else np.asarray(center, dtype=np.float32)
        self.velocity[:] = 0.0
        self.mode = MotionMode.STATIC
        self.static_count = self.static_confirm_frames

    def predict_center(self, dt: float) -> Optional[np.ndarray]:
        if self.center is None:
            return None
        return self.center + self.velocity * max(float(dt), 0.0)

    def predict_quad(self, quad: np.ndarray, dt: float) -> np.ndarray:
        if self.center is None:
            return np.asarray(quad, dtype=np.float32).copy()
        shift = self.velocity * max(float(dt), 0.0)
        return np.asarray(quad, dtype=np.float32) + shift.reshape(1, 2)

    def update(
        self,
        center: Tuple[float, float],
        dt: float,
        quality: float = 1.0,
    ) -> MotionEstimate:
        measurement = np.asarray(center, dtype=np.float32)
        dt = max(float(dt), 1e-3)
        quality = max(0.0, min(1.0, float(quality)))
        if self.center is None:
            self.reset((float(measurement[0]), float(measurement[1])))
            return self.estimate(0.0)

        predicted = self.center + self.velocity * dt
        innovation = measurement - predicted
        residual = float(np.linalg.norm(innovation))
        measured_velocity = (measurement - self.center) / dt
        measured_speed = float(np.linalg.norm(measured_velocity))

        moving = measured_speed >= self.dynamic_speed_px_s or residual >= 5.0
        if moving:
            self.mode = MotionMode.DYNAMIC
            self.static_count = 0
        else:
            self.static_count += 1
            if self.static_count >= self.static_confirm_frames:
                self.mode = MotionMode.STATIC

        if self.mode == MotionMode.DYNAMIC:
            alpha = 0.72 * quality + 0.18
            beta = 0.38 * quality + 0.08
        else:
            alpha = 0.30 * quality + 0.12
            beta = 0.10 * quality + 0.03

        self.center = predicted + alpha * innovation
        self.velocity = self.velocity + (beta / dt) * innovation
        if self.mode == MotionMode.STATIC:
            self.velocity *= 0.72
        return self.estimate(residual)

    def estimate(self, residual: float = 0.0) -> MotionEstimate:
        center = np.zeros(2, dtype=np.float32) if self.center is None else self.center
        speed = float(np.linalg.norm(self.velocity))
        return MotionEstimate(
            center=(float(center[0]), float(center[1])),
            velocity=(float(self.velocity[0]), float(self.velocity[1])),
            speed=speed,
            residual=float(residual),
            mode=self.mode,
        )
