from control.config import ControlConfig
from control.tracker_control import GimbalTracker


def test_tracker_outputs_zero_when_disabled() -> None:
    cfg = ControlConfig(enabled=False)
    t = GimbalTracker(cfg)
    should_send, out = t.update(640, 480, (320.0, 240.0), dt=0.01)
    assert should_send is False
    assert out.yaw_rpm == 0.0
    assert out.pitch_rpm == 0.0


def test_tracker_deadband() -> None:
    cfg = ControlConfig(deadband_px=10.0)
    t = GimbalTracker(cfg)
    should_send, out = t.update(640, 480, (325.0, 245.0), dt=0.02)
    assert should_send is True
    assert out.err_x_px == 0.0
    assert out.err_y_px == 0.0


def test_tracker_sends_zero_immediately_when_target_is_lost() -> None:
    cfg = ControlConfig(enabled=True, lost_timeout_s=0.4)
    tracker = GimbalTracker(cfg)
    should_send, moving = tracker.update(640, 480, (500.0, 240.0), dt=0.02, now=1.0)
    assert should_send is True
    assert moving.yaw_rpm != 0.0

    should_send, stopped = tracker.update(640, 480, None, dt=0.02, now=1.02)
    assert should_send is True
    assert stopped.yaw_rpm == 0.0
    assert stopped.pitch_rpm == 0.0
