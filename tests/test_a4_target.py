import cv2
import numpy as np

from vision.a4_target import (
    A4TargetConfig,
    A4TargetDetector,
    A4TargetTracker,
    A4TrackState,
    find_black_band_candidates,
    map_image_point_to_a4,
)
from vision.rect_detect import DetectedRect


def make_canonical_target(with_rings=True):
    target = np.full((594, 420, 3), 238, dtype=np.uint8)
    tape = 36
    cv2.rectangle(target, (0, 0), (419, 593), (12, 12, 12), thickness=tape)
    if with_rings:
        for radius in (40, 80, 120, 160, 200):
            cv2.circle(target, (210, 297), radius, (15, 15, 210), thickness=3)
        cv2.circle(target, (210, 297), 2, (10, 10, 220), thickness=-1)
    return target


def project_target(target, quad, frame_size=(960, 540)):
    frame_w, frame_h = frame_size
    frame = np.full((frame_h, frame_w, 3), 150, dtype=np.uint8)
    source = np.array([[0, 0], [419, 0], [419, 593], [0, 593]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(source, np.asarray(quad, dtype=np.float32))
    warped = cv2.warpPerspective(target, homography, (frame_w, frame_h))
    mask = cv2.warpPerspective(np.full(target.shape[:2], 255, dtype=np.uint8), homography, (frame_w, frame_h))
    frame[mask > 0] = warped[mask > 0]
    return frame


def candidate_from_quad(quad):
    quad = np.asarray(quad, dtype=np.float32)
    center = tuple(quad.mean(axis=0))
    return DetectedRect(center=center, box=quad, area=float(abs(cv2.contourArea(quad))))


def test_a4_detector_validates_black_tape_and_red_rings() -> None:
    quad = np.array([[300, 40], [650, 65], [680, 500], [260, 480]], dtype=np.float32)
    frame = project_target(make_canonical_target(), quad)
    detector = A4TargetDetector(A4TargetConfig(min_area_ratio=0.005))

    detection = detector.evaluate(frame, candidate_from_quad(quad))

    assert detection is not None
    assert detection.confidence >= 0.65
    assert detection.scores.black_band >= 0.55
    assert detection.scores.red_rings >= 0.35
    assert detection.scores.visible_sides >= 3
    assert len(detection.edge_points) == 64


def test_red_rings_separate_target_from_plain_black_frame() -> None:
    quad = np.array([[300, 40], [650, 65], [680, 500], [260, 480]], dtype=np.float32)
    detector = A4TargetDetector(A4TargetConfig(min_area_ratio=0.005))
    target_detection = detector.evaluate(
        project_target(make_canonical_target(with_rings=True), quad),
        candidate_from_quad(quad),
    )
    distractor_detection = detector.evaluate(
        project_target(make_canonical_target(with_rings=False), quad),
        candidate_from_quad(quad),
    )

    assert target_detection is not None
    assert distractor_detection is not None
    assert target_detection.scores.red_rings > distractor_detection.scores.red_rings + 0.25
    assert target_detection.confidence > distractor_detection.confidence + 0.05


def test_tracker_requires_three_frames_then_holds_occlusion() -> None:
    quad = np.array([[300, 40], [650, 65], [680, 500], [260, 480]], dtype=np.float32)
    frame = project_target(make_canonical_target(), quad)
    candidate = candidate_from_quad(quad)
    tracker = A4TargetTracker(
        A4TargetConfig(
            min_area_ratio=0.005,
            acquire_confidence=0.65,
            acquire_confirm_frames=3,
            occlusion_hold_frames=4,
        )
    )

    first = tracker.update(frame, [candidate])
    second = tracker.update(frame, [candidate])
    third = tracker.update(frame, [candidate])
    occluded = frame.copy()
    cv2.rectangle(occluded, (250, 40), (520, 510), (40, 40, 40), thickness=-1)
    held = tracker.update(occluded, [])

    assert first.state == A4TrackState.SEARCH and not first.current
    assert second.state == A4TrackState.SEARCH and not second.current
    assert third.state == A4TrackState.ACQUIRED and third.current
    assert held.state == A4TrackState.OCCLUDED
    assert held.detection is not None
    assert held.predicted


def test_a4_mapping_returns_physical_center() -> None:
    quad = np.array([[300, 40], [650, 65], [680, 500], [260, 480]], dtype=np.float32)
    frame = project_target(make_canonical_target(), quad)
    detector = A4TargetDetector(A4TargetConfig(min_area_ratio=0.005))
    detection = detector.evaluate(frame, candidate_from_quad(quad))
    assert detection is not None

    center_mm = map_image_point_to_a4(detection, detection.center)

    assert abs(center_mm[0] - 105.0) < 1.0
    assert abs(center_mm[1] - 148.5) < 1.0


def test_black_band_candidate_generation_recovers_target_region() -> None:
    quad = np.array([[300, 40], [650, 65], [680, 500], [260, 480]], dtype=np.float32)
    frame = project_target(make_canonical_target(), quad)

    candidates = find_black_band_candidates(
        frame,
        A4TargetConfig(min_area_ratio=0.005, max_area_ratio=0.8),
    )

    assert candidates
    target_center = quad.mean(axis=0)
    assert any(np.linalg.norm(np.asarray(candidate.center) - target_center) < 80.0 for candidate in candidates)


def test_optical_flow_predicts_target_motion_without_global_detection() -> None:
    quad = np.array([[300, 40], [650, 65], [680, 500], [260, 480]], dtype=np.float32)
    config = A4TargetConfig(
        min_area_ratio=0.005,
        acquire_confidence=0.65,
        acquire_confirm_frames=1,
        occlusion_hold_frames=4,
    )
    tracker = A4TargetTracker(config)
    first_frame = project_target(make_canonical_target(), quad)
    acquired = tracker.update(first_frame, [candidate_from_quad(quad)])
    shifted_quad = quad + np.array([8.0, 5.0], dtype=np.float32)
    shifted_frame = project_target(make_canonical_target(), shifted_quad)

    predicted = tracker.update(shifted_frame, [])

    assert acquired.current
    assert predicted.state == A4TrackState.TRACKING
    assert predicted.current
    assert predicted.predicted
    assert predicted.detection is not None
    assert abs(predicted.detection.center[0] - acquired.detection.center[0] - 8.0) < 3.0
    assert abs(predicted.detection.center[1] - acquired.detection.center[1] - 5.0) < 3.0
    assert predicted.flow_inliers >= config.min_flow_points


def test_black_frame_mode_can_lock_without_red_rings() -> None:
    quad = np.array([[300, 40], [650, 65], [680, 500], [260, 480]], dtype=np.float32)
    frame = project_target(make_canonical_target(with_rings=False), quad)
    candidate = candidate_from_quad(quad)
    config = A4TargetConfig(
        min_area_ratio=0.005,
        acquire_confidence=0.65,
        acquire_confirm_frames=1,
    )

    full_target_tracker = A4TargetTracker(config, require_red_rings=True)
    black_frame_tracker = A4TargetTracker(config, require_red_rings=False)

    full_result = full_target_tracker.update(frame, [candidate])
    black_result = black_frame_tracker.update(frame, [candidate])

    assert full_result.state == A4TrackState.SEARCH
    assert not full_result.current
    assert black_result.state == A4TrackState.ACQUIRED
    assert black_result.current


def test_search_candidate_remains_visible_between_detection_cycles() -> None:
    quad = np.array([[300, 40], [650, 65], [680, 500], [260, 480]], dtype=np.float32)
    frame = project_target(make_canonical_target(with_rings=False), quad)
    tracker = A4TargetTracker(
        A4TargetConfig(min_area_ratio=0.005, acquire_confidence=0.99),
        require_red_rings=False,
    )

    detected = tracker.update(frame, [candidate_from_quad(quad)], detection_cycle=True)
    between_cycles = tracker.update(frame, [], detection_cycle=False)

    assert detected.detection is not None
    assert between_cycles.detection is not None
    assert between_cycles.state == A4TrackState.SEARCH
