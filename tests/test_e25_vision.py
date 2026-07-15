import cv2
import numpy as np

from control.e25_protocol import E25ControlPacket
from vision.a4_target import A4TrackState, order_quad_points, quad_center
from vision.competition_vision import TrackedTarget
from vision.e25_coarse_motion import E25CoarseMotionTracker
from vision.e25_edge_tracker import E25EdgeTracker
from vision.e25_laser import E25LaserTracker
from vision.e25_motion import AlphaBetaMotionPredictor, MotionMode
from vision.e25_pipeline import E25PipelineConfig, E25VisionPipeline
from vision.e25_target_model import E25TargetModel
from vision.rect_detect import DetectedRect


def make_target() -> np.ndarray:
    target = np.full((594, 420, 3), 238, dtype=np.uint8)
    target[:36, :] = 12
    target[-36:, :] = 12
    target[:, :36] = 12
    target[:, -36:] = 12
    return target


def project_target(target: np.ndarray, quad: np.ndarray, frame_size=(960, 540)) -> np.ndarray:
    width, height = frame_size
    frame = np.full((height, width, 3), 150, dtype=np.uint8)
    source = np.array([[0, 0], [419, 0], [419, 593], [0, 593]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source, quad.astype(np.float32))
    warped = cv2.warpPerspective(target, transform, (width, height))
    mask = cv2.warpPerspective(np.full(target.shape[:2], 255, dtype=np.uint8), transform, (width, height))
    frame[mask > 0] = warped[mask > 0]
    return frame


def candidate(quad: np.ndarray) -> DetectedRect:
    quad = quad.astype(np.float32)
    return DetectedRect(
        center=tuple(quad.mean(axis=0)),
        box=quad,
        area=abs(float(cv2.contourArea(quad))),
    )


def test_target_model_center_and_orientation() -> None:
    model = E25TargetModel()
    assert model.center_mm == (105.0, 148.5)
    assert model.dimensions_for_canonical((210, 297)) == (210.0, 297.0)
    assert model.dimensions_for_canonical((297, 210)) == (297.0, 210.0)
    assert model.canonical_to_mm((105.0, 148.5), (210, 297)) == (105.0, 148.5)


def test_motion_predictor_switches_dynamic_then_returns_static() -> None:
    predictor = AlphaBetaMotionPredictor(dynamic_speed_px_s=100.0, static_confirm_frames=3)
    predictor.reset((100.0, 100.0))
    moving = predictor.update((110.0, 100.0), 1.0 / 30.0)
    assert moving.mode == MotionMode.DYNAMIC
    predicted = predictor.predict_center(1.0 / 30.0)
    assert predicted is not None and predicted[0] > moving.center[0]
    center = moving.center
    for _ in range(8):
        estimate = predictor.update(center, 1.0 / 30.0)
        center = estimate.center
    assert estimate.mode == MotionMode.STATIC


def test_coarse_motion_estimates_large_translation() -> None:
    quad = np.array([[310, 55], [635, 65], [660, 495], [280, 480]], dtype=np.float32)
    shifted = quad + np.array([75.0, -30.0], dtype=np.float32)
    target = make_target()
    tracker = E25CoarseMotionTracker()
    tracker.reset(project_target(target, quad), quad)

    estimate = tracker.estimate(project_target(target, shifted), quad)

    assert estimate.quad is not None
    assert estimate.inliers >= 6
    error = np.sqrt(
        np.mean(
            np.sum(
                (order_quad_points(estimate.quad) - order_quad_points(shifted)) ** 2,
                axis=1,
            )
        )
    )
    assert error < 8.0


def test_edge_tracker_refines_shifted_prediction() -> None:
    quad = np.array([[310, 55], [635, 65], [660, 495], [280, 480]], dtype=np.float32)
    frame = project_target(make_target(), quad)
    predicted = quad + np.array([6.0, -4.0], dtype=np.float32)

    measurement = E25EdgeTracker().measure(frame, predicted, (210, 297))

    assert measurement.quad is not None
    assert measurement.visible_sides >= 3
    assert measurement.quality >= 0.40
    error = np.sqrt(
        np.mean(np.sum((order_quad_points(measurement.quad) - order_quad_points(quad)) ** 2, axis=1))
    )
    assert error < 10.0


def test_edge_tracker_updates_with_one_partially_occluded_side() -> None:
    quad = np.array([[310, 55], [635, 65], [660, 495], [280, 480]], dtype=np.float32)
    frame = project_target(make_target(), quad)
    cv2.rectangle(frame, (410, 430), (560, 525), (200, 200, 200), thickness=-1)

    measurement = E25EdgeTracker().measure(frame, quad, (210, 297))

    assert measurement.quad is not None
    assert measurement.visible_sides >= 3


def test_pipeline_acquires_and_tracks_translation() -> None:
    quad = np.array([[310, 55], [635, 65], [660, 495], [280, 480]], dtype=np.float32)
    config = E25PipelineConfig(
        min_area_ratio=0.005,
        acquire_confidence=0.60,
        acquire_confirm_frames=1,
        structural_validate_interval=3,
    )
    pipeline = E25VisionPipeline(config, require_red_rings=False)
    first_frame = project_target(make_target(), quad)
    acquired = pipeline.update(first_frame, [candidate(quad)], detection_cycle=True)
    shifted = quad + np.array([14.0, 7.0], dtype=np.float32)
    tracked = pipeline.update(project_target(make_target(), shifted), [], detection_cycle=False)

    assert acquired.state == A4TrackState.ACQUIRED
    assert tracked.state == A4TrackState.TRACKING
    assert tracked.detection is not None
    assert tracked.confidence.measurement >= 0.38
    assert np.linalg.norm(np.asarray(tracked.detection.center) - np.asarray(quad_center(shifted))) < 8.0


def test_pipeline_tracks_single_frame_fast_translation() -> None:
    quad = np.array([[310, 55], [635, 65], [660, 495], [280, 480]], dtype=np.float32)
    shifted = quad + np.array([75.0, -30.0], dtype=np.float32)
    target = make_target()
    pipeline = E25VisionPipeline(
        E25PipelineConfig(
            min_area_ratio=0.005,
            acquire_confidence=0.60,
            acquire_confirm_frames=1,
        ),
        require_red_rings=False,
    )
    pipeline.update(project_target(target, quad), [candidate(quad)], detection_cycle=True)

    tracked = pipeline.update(project_target(target, shifted), [], detection_cycle=False)

    assert tracked.state == A4TrackState.TRACKING
    assert tracked.current and not tracked.predicted
    assert tracked.detection is not None
    assert np.linalg.norm(
        np.asarray(tracked.detection.center) - np.asarray(quad_center(shifted))
    ) < 10.0


def test_pipeline_recovers_next_frame_after_extreme_translation() -> None:
    quad = np.array([[310, 55], [635, 65], [660, 495], [280, 480]], dtype=np.float32)
    shifted = quad + np.array([150.0, -40.0], dtype=np.float32)
    target = make_target()
    pipeline = E25VisionPipeline(
        E25PipelineConfig(
            min_area_ratio=0.005,
            acquire_confidence=0.60,
            acquire_confirm_frames=1,
        ),
        require_red_rings=False,
    )
    pipeline.update(project_target(target, quad), [candidate(quad)], detection_cycle=True)
    shifted_frame = project_target(target, shifted)

    held = pipeline.update(shifted_frame, [], detection_cycle=False)
    recovered = pipeline.update(shifted_frame, [candidate(shifted)], detection_cycle=True)

    assert held.state == A4TrackState.OCCLUDED
    assert held.predicted and not held.current
    assert recovered.state in (A4TrackState.ACQUIRED, A4TrackState.TRACKING)
    assert recovered.current and recovered.detection is not None
    assert np.linalg.norm(
        np.asarray(recovered.detection.center) - np.asarray(quad_center(shifted))
    ) < 10.0


def test_pipeline_prediction_freezes_instead_of_recursively_drifting() -> None:
    quad = np.array([[310, 55], [635, 65], [660, 495], [280, 480]], dtype=np.float32)
    shifted = quad + np.array([18.0, 5.0], dtype=np.float32)
    target = make_target()
    pipeline = E25VisionPipeline(
        E25PipelineConfig(
            min_area_ratio=0.005,
            acquire_confidence=0.60,
            acquire_confirm_frames=1,
            max_prediction_frames=4,
            recovery_hold_frames=20,
        ),
        require_red_rings=False,
    )
    pipeline.update(project_target(target, quad), [candidate(quad)], detection_cycle=True)
    reliable = pipeline.update(project_target(target, shifted), [], detection_cycle=False)
    assert reliable.current and reliable.detection is not None
    reliable_center = np.asarray(reliable.detection.center)
    blank = np.full((540, 960, 3), 150, dtype=np.uint8)

    centers = []
    for _ in range(12):
        held = pipeline.update(blank, [], detection_cycle=False)
        assert held.state == A4TrackState.OCCLUDED
        assert held.detection is not None and not held.current
        centers.append(np.asarray(held.detection.center))

    assert np.linalg.norm(centers[-1] - centers[-2]) < 0.1
    assert np.linalg.norm(centers[-1] - reliable_center) < 150.0


def test_fast_recovery_does_not_switch_to_saturated_distractor() -> None:
    quad = np.array([[310, 55], [635, 65], [660, 495], [280, 480]], dtype=np.float32)
    distractor_quad = quad + np.array([80.0, -15.0], dtype=np.float32)
    target = make_target()
    orange_target = make_target()
    orange_target[36:-36, 36:-36] = (34, 61, 132)
    pipeline = E25VisionPipeline(
        E25PipelineConfig(
            min_area_ratio=0.005,
            acquire_confidence=0.60,
            acquire_confirm_frames=1,
        ),
        require_red_rings=False,
    )
    pipeline.update(project_target(target, quad), [candidate(quad)], detection_cycle=True)

    result = pipeline.update(
        project_target(orange_target, distractor_quad),
        [candidate(distractor_quad)],
        detection_cycle=True,
    )

    assert result.state == A4TrackState.OCCLUDED
    assert not result.current
    assert not result.confidence.control_valid


def test_pipeline_preserves_acquisition_across_sparse_detection_cycles() -> None:
    quad = np.array([[310, 55], [635, 65], [660, 495], [280, 480]], dtype=np.float32)
    frame = project_target(make_target(), quad)
    pipeline = E25VisionPipeline(
        E25PipelineConfig(
            min_area_ratio=0.005,
            acquire_confidence=0.60,
            acquire_confirm_frames=3,
        ),
        require_red_rings=False,
    )

    first = pipeline.update(frame, [candidate(quad)], detection_cycle=True)
    for _ in range(5):
        between = pipeline.update(frame, [], detection_cycle=False)
    second = pipeline.update(frame, [candidate(quad)], detection_cycle=True)
    for _ in range(5):
        pipeline.update(frame, [], detection_cycle=False)
    acquired = pipeline.update(frame, [candidate(quad)], detection_cycle=True)

    assert first.state == A4TrackState.SEARCH
    assert between.state == A4TrackState.SEARCH
    assert second.state == A4TrackState.SEARCH
    assert acquired.state == A4TrackState.ACQUIRED
    assert acquired.current


def test_pipeline_rejects_near_square_black_frame_distractor() -> None:
    square = np.array([[380, 150], [580, 150], [580, 350], [380, 350]], dtype=np.float32)
    frame = project_target(make_target(), square)
    pipeline = E25VisionPipeline(
        E25PipelineConfig(
            min_area_ratio=0.005,
            acquire_confidence=0.60,
            acquire_confirm_frames=1,
            min_apparent_aspect=1.20,
        ),
        require_red_rings=False,
    )

    result = pipeline.update(frame, [candidate(square)], detection_cycle=True)

    assert result.state == A4TrackState.SEARCH
    assert not result.current


def test_pipeline_rejects_saturated_landscape_frame_distractor() -> None:
    quad = np.array([[300, 110], [660, 120], [650, 380], [310, 370]], dtype=np.float32)
    orange_target = make_target()
    orange_target[36:-36, 36:-36] = (34, 61, 132)
    frame = project_target(orange_target, quad)
    pipeline = E25VisionPipeline(
        E25PipelineConfig(
            min_area_ratio=0.005,
            acquire_confidence=0.60,
            acquire_confirm_frames=1,
        ),
        require_red_rings=False,
    )

    result = pipeline.update(frame, [candidate(quad)], detection_cycle=True)

    assert result.state == A4TrackState.SEARCH
    assert not result.current


def test_e25_laser_rejects_warm_glare() -> None:
    frame = np.full((240, 400, 3), (194, 171, 209), dtype=np.uint8)
    cv2.circle(frame, (165, 100), 12, (125, 230, 253), thickness=-1)
    cv2.circle(frame, (165, 100), 3, (255, 255, 255), thickness=-1)
    cv2.circle(frame, (210, 125), 10, (250, 105, 168), thickness=-1)
    cv2.circle(frame, (210, 125), 3, (255, 253, 255), thickness=-1)
    rect = candidate(np.array([[120, 70], [280, 70], [280, 170], [120, 170]], dtype=np.float32))
    target = TrackedTarget(rect, current=True, held=False, miss_count=0)

    laser = E25LaserTracker().update(frame, target, enabled=True)

    assert laser is not None and laser.current
    assert abs(laser.center[0] - 210.0) < 3.0
    assert abs(laser.center[1] - 125.0) < 3.0


def test_control_packet_is_ascii_and_explicitly_gated() -> None:
    packet = E25ControlPacket(
        sequence=7,
        target_valid=True,
        laser_valid=True,
        control_valid=False,
        error_x_mm=1.25,
        error_y_mm=-2.5,
        identity_confidence=0.9,
        measurement_quality=0.8,
        tracking_confidence=0.7,
    ).encode()
    assert packet == b"@E25,7,1,1,0,1.250,-2.500,0.900,0.800,0.700\r\n"
