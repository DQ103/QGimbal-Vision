import cv2
import numpy as np

from vision.competition_vision import (
    AimReadyGate,
    HybridLaserConfig,
    HybridLaserTracker,
    TargetTracker,
    VisionStage,
    resolve_stage,
    stage_error,
)
from vision.rect_detect import DetectedRect, RectSelectionConfig


def make_rect(center=(200.0, 120.0), size=(160.0, 100.0)) -> DetectedRect:
    cx, cy = center
    w, h = size
    box = np.array(
        [
            [cx - w / 2.0, cy - h / 2.0],
            [cx + w / 2.0, cy - h / 2.0],
            [cx + w / 2.0, cy + h / 2.0],
            [cx - w / 2.0, cy + h / 2.0],
        ],
        dtype=np.float32,
    )
    return DetectedRect(center=center, box=box, area=w * h)


def current_target(rect=None):
    tracker = TargetTracker(
        RectSelectionConfig(min_area_ratio=0.001, max_area_ratio=0.9),
        miss_confirm_frames=3,
    )
    return tracker.update([rect or make_rect()], 400, 240)


def test_target_tracker_holds_two_misses_but_marks_them_invalid() -> None:
    tracker = TargetTracker(
        RectSelectionConfig(min_area_ratio=0.001, max_area_ratio=0.9),
        miss_confirm_frames=3,
    )
    found = tracker.update([make_rect()], 400, 240)
    miss_one = tracker.update([], 400, 240)
    miss_two = tracker.update([], 400, 240)
    miss_three = tracker.update([], 400, 240)

    assert found is not None and found.current and not found.held
    assert miss_one is not None and miss_one.held and not miss_one.current
    assert miss_two is not None and miss_two.held and not miss_two.current
    assert miss_three is None


def test_aim_gate_requires_consecutive_current_measurements() -> None:
    gate = AimReadyGate(enter_radius_ratio=0.10, exit_radius_ratio=0.15, confirm_frames=3)
    target = current_target()
    assert target is not None

    assert not gate.update(target, 400, 240).ready
    assert not gate.update(target, 400, 240).ready
    status = gate.update(target, 400, 240)
    assert status.ready

    held = target.__class__(target.rect, current=False, held=True, miss_count=1)
    assert not gate.update(held, 400, 240).ready


def test_hybrid_laser_finds_white_core_inside_blue_violet_halo() -> None:
    frame = np.zeros((240, 400, 3), dtype=np.uint8)
    cv2.circle(frame, (210, 125), 10, (255, 0, 0), thickness=-1)
    cv2.circle(frame, (210, 125), 3, (255, 255, 255), thickness=-1)
    target = current_target()
    assert target is not None

    tracker = HybridLaserTracker()
    laser = tracker.update(frame, target, enabled=True)

    assert laser is not None and laser.current
    assert abs(laser.center[0] - 210.0) < 2.0
    assert abs(laser.center[1] - 125.0) < 2.0
    assert laser.violet_pixels > 0


def test_hybrid_laser_rejects_dim_white_reflection_without_violet_support() -> None:
    frame = np.zeros((240, 400, 3), dtype=np.uint8)
    cv2.circle(frame, (210, 125), 4, (190, 190, 190), thickness=-1)
    target = current_target()
    assert target is not None

    tracker = HybridLaserTracker(HybridLaserConfig(fallback_min_luma=210))
    assert tracker.update(frame, target, enabled=True) is None


def test_k230_07_profile_requires_violet_support_even_for_white_core() -> None:
    frame = np.zeros((240, 400, 3), dtype=np.uint8)
    cv2.circle(frame, (210, 125), 4, (255, 255, 255), thickness=-1)
    target = current_target()
    assert target is not None

    tracker = HybridLaserTracker(
        HybridLaserConfig(require_violet=True, strict_violet=True)
    )

    assert tracker.update(frame, target, enabled=True) is None


def test_laser_hold_does_not_advance_align_laser_stage() -> None:
    frame = np.zeros((240, 400, 3), dtype=np.uint8)
    cv2.circle(frame, (210, 125), 10, (255, 0, 0), thickness=-1)
    cv2.circle(frame, (210, 125), 3, (255, 255, 255), thickness=-1)
    target = current_target()
    assert target is not None

    gate = AimReadyGate(enter_radius_ratio=0.10, exit_radius_ratio=0.15, confirm_frames=1)
    aim = gate.update(target, 400, 240)
    tracker = HybridLaserTracker(HybridLaserConfig(hold_frames=2))
    current = tracker.update(frame, target, enabled=aim.ready)
    held = tracker.update(np.zeros_like(frame), target, enabled=aim.ready)

    assert current is not None and current.current
    assert resolve_stage(target, aim, current) == VisionStage.ALIGN_LASER
    assert held is not None and held.held and not held.current
    assert resolve_stage(target, aim, held) == VisionStage.SEARCH_LASER
    assert stage_error(VisionStage.ALIGN_LASER, target, aim, current) == (-10.0, -5.0)
