#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.serial_stub import GimbalSerialStub


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the vision adapter against the G3519 PTY simulator")
    parser.add_argument("--bridge", required=True, help="Path to run_qgimbal_pty_sim.py")
    parser.add_argument("--simulator", required=True, help="Path to qgimbal_protocol_sim")
    args = parser.parse_args()

    process = subprocess.Popen(
        [
            sys.executable,
            args.bridge,
            "--simulator",
            args.simulator,
            "--",
            "--output",
            "simulated",
            "--bmi-valid",
            "1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    port = process.stdout.readline().strip()
    if not port:
        stderr = process.stderr.read() if process.stderr is not None else ""
        raise RuntimeError(f"PTY bridge did not provide a port: {stderr}")

    adapter = GimbalSerialStub(port=port, baudrate=115200)
    try:
        adapter.open()
        if not adapter.enable(wait_for_feedback=True):
            raise RuntimeError(adapter.last_error or "enable response timed out")
        before = adapter.feedback_sequence
        if not adapter.send_rpm(20.0, -10.0):
            raise RuntimeError(adapter.last_error or "speed write failed")
        if not adapter.wait_for_feedback(after_sequence=before, timeout_s=1.0):
            raise RuntimeError("speed response timed out")
        telemetry = adapter.latest_telemetry
        if telemetry is None or not telemetry.enabled:
            raise RuntimeError("simulator did not report enabled state")
        print(
            f"QGimbal PTY roundtrip ok: status=0x{telemetry.status:02X} "
            f"feedback_seq={adapter.feedback_sequence}"
        )
        return 0
    finally:
        adapter.close()
        time.sleep(0.05)
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
