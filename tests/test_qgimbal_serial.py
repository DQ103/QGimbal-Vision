import math
import struct

import pytest

from control.serial_stub import (
    CMD_SPEED_CTRL,
    FEEDBACK_SIZE,
    REQUEST_SIZE,
    GimbalSerialStub,
    crc8_qgimbal,
    decode_feedback,
    encode_request,
)


def test_crc8_known_vector() -> None:
    assert crc8_qgimbal(b"123456789") == 0xF4


def test_speed_request_layout() -> None:
    frame = encode_request(CMD_SPEED_CTRL, 1.5, -2.25)
    assert len(frame) == REQUEST_SIZE
    assert frame.hex() == "040000c03f000010c006"
    assert frame[-1] == crc8_qgimbal(frame[:-1])


def test_request_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError):
        encode_request(CMD_SPEED_CTRL, math.nan, 0.0)
    with pytest.raises(ValueError):
        encode_request(CMD_SPEED_CTRL, 0.0, math.inf)


def test_feedback_layout_matches_g3519_order() -> None:
    payload = struct.pack(
        "<Bffffffffff",
        0x35,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
    )
    frame = payload + bytes((crc8_qgimbal(payload),))
    telemetry = decode_feedback(frame, received_at=123.0)
    assert len(frame) == FEEDBACK_SIZE
    assert telemetry.imu_speed == (1.0, 2.0)
    assert telemetry.imu_angle == (3.0, 4.0)
    assert telemetry.motor_current == (5.0, 6.0)
    assert telemetry.motor_speed == (7.0, 8.0)
    assert telemetry.motor_angle == (9.0, 10.0)
    assert telemetry.received_at == 123.0


def test_feedback_rejects_bad_crc() -> None:
    frame = bytearray(FEEDBACK_SIZE)
    frame[-1] = 1
    with pytest.raises(ValueError):
        decode_feedback(bytes(frame))


def test_legacy_class_build_packet_uses_v1_speed_command() -> None:
    adapter = GimbalSerialStub()
    assert adapter.build_packet(1.5, -2.25) == encode_request(
        CMD_SPEED_CTRL, 1.5, -2.25
    )
