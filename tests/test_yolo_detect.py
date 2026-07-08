import json

import numpy as np

from vision.yolo_detect import parse_yolo_response, yolo_boxes_to_rects


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
