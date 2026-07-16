from __future__ import annotations

import math
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


REQUEST_SIZE = 10
FEEDBACK_SIZE = 42

CMD_NOP = 0x00
CMD_ENABLE = 0x01
CMD_DISABLE = 0x02
CMD_CURRENT_CTRL = 0x03
CMD_SPEED_CTRL = 0x04
CMD_ANGLE_CTRL = 0x05
CMD_LOW_SPEED_CTRL = 0x06
CMD_STEP_ANGLE_CTRL = 0x07
CMD_RESET_IMU = 0xFB
CMD_DISABLE_LASER = 0xFC
CMD_ENABLE_LASER = 0xFD
CMD_DISABLE_STABILITY = 0xFE
CMD_ENABLE_STABILITY = 0xFF


def crc8_qgimbal(data: bytes) -> int:
    """CRC-8, polynomial 0x07, init/xorout 0, no reflection."""
    crc = 0
    for value in data:
        crc ^= value
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def encode_request(command: int, yaw: float = 0.0, pitch: float = 0.0) -> bytes:
    yaw_value = float(yaw)
    pitch_value = float(pitch)
    if not math.isfinite(yaw_value) or not math.isfinite(pitch_value):
        raise ValueError("QGimbal command values must be finite")
    payload = struct.pack("<Bff", int(command) & 0xFF, yaw_value, pitch_value)
    return payload + bytes((crc8_qgimbal(payload),))


@dataclass(frozen=True)
class GimbalTelemetry:
    status: int
    imu_speed: tuple[float, float]
    imu_angle: tuple[float, float]
    motor_current: tuple[float, float]
    motor_speed: tuple[float, float]
    motor_angle: tuple[float, float]
    received_at: float

    @property
    def enabled(self) -> bool:
        return bool(self.status & 0x01)

    @property
    def stability_enabled(self) -> bool:
        return bool(self.status & 0x02)

    @property
    def laser_enabled(self) -> bool:
        return bool(self.status & 0x04)

    @property
    def fault(self) -> bool:
        return bool(self.status & 0x08)


def decode_feedback(frame: bytes, *, received_at: float | None = None) -> GimbalTelemetry:
    if len(frame) != FEEDBACK_SIZE:
        raise ValueError(f"QGimbal feedback must be {FEEDBACK_SIZE} bytes")
    if crc8_qgimbal(frame[:-1]) != frame[-1]:
        raise ValueError("invalid QGimbal feedback CRC")
    unpacked = struct.unpack("<BffffffffffB", frame)
    values = unpacked[1:11]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("QGimbal feedback contains a non-finite value")
    return GimbalTelemetry(
        status=unpacked[0],
        imu_speed=(values[0], values[1]),
        imu_angle=(values[2], values[3]),
        motor_current=(values[4], values[5]),
        motor_speed=(values[6], values[7]),
        motor_angle=(values[8], values[9]),
        received_at=time.monotonic() if received_at is None else received_at,
    )


@dataclass
class GimbalSerialStub:
    """QGimbal V1 serial adapter.

    Requests are 10-byte ``command + yaw/pitch float32 + CRC8`` frames. The
    background reader continuously drains and validates 42-byte telemetry frames.
    Keeping the historical class name avoids breaking existing launch scripts.
    """

    port: Optional[str] = None
    baudrate: int = 115200
    response_timeout_s: float = 0.30

    _ser: object | None = field(default=None, init=False, repr=False)
    _reader: threading.Thread | None = field(default=None, init=False, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _write_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _feedback_condition: threading.Condition = field(default_factory=threading.Condition, init=False, repr=False)
    _telemetry: GimbalTelemetry | None = field(default=None, init=False, repr=False)
    _feedback_sequence: int = field(default=0, init=False)
    _crc_errors: int = field(default=0, init=False)
    _last_error: str | None = field(default=None, init=False)

    @property
    def is_open(self) -> bool:
        return self._ser is not None

    @property
    def latest_telemetry(self) -> GimbalTelemetry | None:
        with self._feedback_condition:
            return self._telemetry

    @property
    def feedback_sequence(self) -> int:
        with self._feedback_condition:
            return self._feedback_sequence

    @property
    def crc_errors(self) -> int:
        return self._crc_errors

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def open(self) -> bool:
        if self.port is None:
            return False
        try:
            import serial  # type: ignore

            self._ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.05,
                write_timeout=0.10,
            )
        except Exception as exc:
            self._ser = None
            self._last_error = str(exc)
            return False
        self._stop_event.clear()
        self._reader = threading.Thread(
            target=self._reader_loop,
            name="qgimbal-serial-reader",
            daemon=True,
        )
        self._reader.start()
        return True

    def close(self) -> None:
        ser = self._ser
        if ser is None:
            return
        self.send_zero()
        self.disable(wait_for_feedback=False)
        self._stop_event.set()
        reader = self._reader
        if reader is not None and reader.is_alive():
            reader.join(timeout=0.25)
        self._reader = None
        self._ser = None
        try:
            ser.close()
        except Exception as exc:  # pragma: no cover - backend-specific failure
            self._last_error = str(exc)

    def build_packet(self, yaw_rpm: float, pitch_rpm: float) -> bytes:
        return encode_request(CMD_SPEED_CTRL, yaw_rpm, pitch_rpm)

    def send_rpm(self, yaw_rpm: float, pitch_rpm: float) -> bool:
        return self._send_command(CMD_SPEED_CTRL, yaw_rpm, pitch_rpm)

    def send_zero(self) -> bool:
        return self.send_rpm(0.0, 0.0)

    def nop(self, *, wait_for_feedback: bool = True) -> bool:
        return self._command_with_optional_response(CMD_NOP, wait_for_feedback)

    def enable(self, *, wait_for_feedback: bool = True) -> bool:
        before = self.feedback_sequence
        if not self._send_command(CMD_ENABLE, 0.0, 0.0):
            return False
        if not wait_for_feedback:
            return True
        if not self.wait_for_feedback(after_sequence=before):
            return False
        telemetry = self.latest_telemetry
        if telemetry is None or not telemetry.enabled or telemetry.fault:
            self._last_error = "lower controller did not enter a healthy enabled state"
            return False
        return True

    def disable(self, *, wait_for_feedback: bool = True) -> bool:
        return self._command_with_optional_response(CMD_DISABLE, wait_for_feedback)

    def set_laser(self, enabled: bool, *, wait_for_feedback: bool = True) -> bool:
        command = CMD_ENABLE_LASER if enabled else CMD_DISABLE_LASER
        return self._command_with_optional_response(command, wait_for_feedback)

    def set_stability(self, enabled: bool, *, wait_for_feedback: bool = True) -> bool:
        command = CMD_ENABLE_STABILITY if enabled else CMD_DISABLE_STABILITY
        return self._command_with_optional_response(command, wait_for_feedback)

    def _command_with_optional_response(self, command: int, wait_for_feedback: bool) -> bool:
        before = self.feedback_sequence
        if not self._send_command(command, 0.0, 0.0):
            return False
        return not wait_for_feedback or self.wait_for_feedback(after_sequence=before)

    def wait_for_feedback(self, *, after_sequence: int, timeout_s: float | None = None) -> bool:
        deadline = time.monotonic() + (
            self.response_timeout_s if timeout_s is None else max(0.0, timeout_s)
        )
        with self._feedback_condition:
            while self._feedback_sequence <= after_sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._feedback_condition.wait(timeout=remaining)
            return True

    def telemetry_is_fresh(self, max_age_s: float = 0.5) -> bool:
        telemetry = self.latest_telemetry
        return telemetry is not None and (time.monotonic() - telemetry.received_at) <= max_age_s

    def _send_command(self, command: int, yaw: float, pitch: float) -> bool:
        ser = self._ser
        if ser is None:
            return False
        try:
            packet = encode_request(command, yaw, pitch)
            with self._write_lock:
                written = ser.write(packet)
            if written != len(packet):
                self._last_error = f"short serial write: {written}/{len(packet)}"
                return False
            return True
        except Exception as exc:
            self._last_error = str(exc)
            return False

    def _reader_loop(self) -> None:
        buffer = bytearray()
        while not self._stop_event.is_set():
            ser = self._ser
            if ser is None:
                return
            try:
                chunk = ser.read(256)
            except Exception as exc:
                self._last_error = str(exc)
                return
            if not chunk:
                continue
            buffer.extend(chunk)
            while len(buffer) >= FEEDBACK_SIZE:
                candidate = bytes(buffer[:FEEDBACK_SIZE])
                try:
                    telemetry = decode_feedback(candidate)
                except ValueError:
                    del buffer[0]
                    self._crc_errors += 1
                    continue
                del buffer[:FEEDBACK_SIZE]
                with self._feedback_condition:
                    self._telemetry = telemetry
                    self._feedback_sequence += 1
                    self._feedback_condition.notify_all()
