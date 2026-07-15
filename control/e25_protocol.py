from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class E25ControlPacket:
    sequence: int
    target_valid: bool
    laser_valid: bool
    control_valid: bool
    error_x_mm: float
    error_y_mm: float
    identity_confidence: float
    measurement_quality: float
    tracking_confidence: float

    def encode(self) -> bytes:
        line = (
            f"@E25,{int(self.sequence)},{int(self.target_valid)},"
            f"{int(self.laser_valid)},{int(self.control_valid)},"
            f"{self.error_x_mm:.3f},{self.error_y_mm:.3f},"
            f"{self.identity_confidence:.3f},{self.measurement_quality:.3f},"
            f"{self.tracking_confidence:.3f}\r\n"
        )
        return line.encode("ascii")
