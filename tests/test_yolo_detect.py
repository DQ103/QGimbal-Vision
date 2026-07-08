import json
import base64
import subprocess
import time

import cv2
import numpy as np

from vision.yolo_detect import AsyncYoloDetector, parse_yolo_response, yolo_boxes_to_rects
from scripts.yolo_json_worker_cli_adapter import normalize_response


def test_parse_yolo_response_filters_confidence_and_label() -> None:
    line = json.dumps(
        {
            "frame_id": 1,
            "detections": [
                {"bbox": [10, 20, 110, 120], "confidence": 0.9, "label": "target"},
                {"bbox": [0, 0, 20, 20], "confidence": 0.1, "label": "target"},
                {"bbox": [30, 30, 70, 70], "confidence": 0.8, "label": "other"},
            ],
        }
    )

    boxes = parse_yolo_response(line, min_confidence=0.25, labels={"target"})

    assert len(boxes) == 1
    assert boxes[0].label == "target"
    assert boxes[0].confidence == 0.9


def test_yolo_boxes_to_rects_converts_to_detected_rect() -> None:
    line = json.dumps({"detections": [{"bbox": [10, 20, 110, 120], "confidence": 0.9, "label": "target"}]})

    rects = yolo_boxes_to_rects(parse_yolo_response(line))

    assert len(rects) == 1
    rect = rects[0]
    assert rect.center == (60.0, 70.0)
    assert rect.area == 10000.0
    assert rect.pass_index == 100
    np.testing.assert_allclose(rect.box[0], [10.0, 20.0])


def test_async_yolo_detector_reads_fixed_worker_result() -> None:
    detector = AsyncYoloDetector.from_shell_command(
        "python3 scripts/yolo_json_worker_fixed.py --bbox 0.25,0.25,0.75,0.75",
        timeout_s=1.0,
    )
    try:
        detector.submit(np.zeros((100, 200, 3), dtype=np.uint8))
        result = None
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            result = detector.latest()
            if result is not None:
                break
            time.sleep(0.01)

        assert result is not None
        assert result.frame_size == (200, 100)
        assert len(result.rects) == 1
        assert result.rects[0].center == (100.0, 50.0)
        assert result.rects[0].pass_index == 100
    finally:
        detector.close()


def test_cli_adapter_normalizes_external_json_output() -> None:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    request = {
        "frame_id": 7,
        "width": 16,
        "height": 12,
        "format": "jpg_b64",
        "image": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }
    command = (
        "python3 scripts/yolo_json_worker_cli_adapter.py "
        "--command \"python3 -c 'import json; "
        "print(json.dumps({{\\\"detections\\\":[{{\\\"bbox\\\":[1,2,3,4],\\\"confidence\\\":0.8,\\\"label\\\":\\\"target\\\"}}]}}))'\""
    )

    proc = subprocess.run(
        command,
        input=json.dumps(request) + "\n",
        text=True,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=2.0,
        check=True,
    )
    response = json.loads(proc.stdout)

    assert response["frame_id"] == 7
    assert response["detections"][0]["bbox"] == [1, 2, 3, 4]


def test_cli_adapter_parses_allwinner_yolov8_output() -> None:
    stderr = """
detection num: 3
 1:  87%, [ 130,  136,  568,  419], bicycle
16:  95%, [ 131,  220,  308,  541], dog
 2:  68%, [ 467,   74,  695,  171], car
"""

    response = normalize_response("", stderr, frame_id=8, parser="allwinner-yolo")

    assert response["frame_id"] == 8
    assert response["detections"][0] == {
        "bbox": [130.0, 136.0, 568.0, 419.0],
        "confidence": 0.87,
        "label": "bicycle",
        "class_id": 1,
    }
    assert response["detections"][1]["label"] == "dog"
