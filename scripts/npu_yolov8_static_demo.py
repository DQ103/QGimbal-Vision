#!/usr/bin/env python3
"""Generate a static web page for the Allwinner YOLOv8 NPU demo."""

from __future__ import annotations

import argparse
import html
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import cv2


DETECTION_RE = re.compile(
    r"\s*(?P<class_id>\d+):\s*(?P<conf>\d+)%.*,"
    r"\s*\[\s*(?P<x1>\d+),\s*(?P<y1>\d+),\s*(?P<x2>\d+),\s*(?P<y2>\d+)\],\s*(?P<label>.+?)\s*$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo",
        default="/home/radxa/awnpu_model_zoo/examples/yolov8/build_native/yolov8_demo_a733",
        help="Path to yolov8_demo_a733.",
    )
    parser.add_argument(
        "--model",
        default="/home/radxa/awnpu_model_zoo/examples/yolov8/model/yolov8n_6_uint8_a733.nb",
        help="Path to A733 .nb model.",
    )
    parser.add_argument(
        "--image",
        default="/home/radxa/awnpu_model_zoo/examples/yolov8/model/dog.jpg",
        help="Input image.",
    )
    parser.add_argument(
        "--runtime-dir",
        default="/home/radxa/awnpu_model_zoo/common/npuruntime/lib_linux_aarch64/A733",
        help="Directory containing libVIPhal.so.",
    )
    parser.add_argument(
        "--out",
        default="/home/radxa/npu_yolov8_demo",
        help="Output directory served by the static HTTP server.",
    )
    return parser.parse_args()


def parse_detections(text: str) -> list[dict[str, object]]:
    detections: list[dict[str, object]] = []
    for line in text.splitlines():
        match = DETECTION_RE.match(line)
        if not match:
            continue
        detections.append(
            {
                "class_id": int(match.group("class_id")),
                "confidence": int(match.group("conf")),
                "box": (
                    int(match.group("x1")),
                    int(match.group("y1")),
                    int(match.group("x2")),
                    int(match.group("y2")),
                ),
                "label": match.group("label"),
            }
        )
    return detections


def find_runtime_line(text: str) -> str:
    for line in text.splitlines():
        if "run time for this network" in line:
            return line.strip()
    return ""


def draw_detections(image_path: Path, detections: list[dict[str, object]], output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"failed to read input image: {image_path}")

    colors = [(40, 220, 40), (40, 160, 255), (255, 180, 40), (220, 80, 220)]
    for index, item in enumerate(detections):
        x1, y1, x2, y2 = item["box"]  # type: ignore[misc]
        color = colors[index % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

        label = f"{item['label']} {item['confidence']}%"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
        )
        label_y = max(0, y1 - text_height - baseline - 8)
        cv2.rectangle(
            image,
            (x1, label_y),
            (x1 + text_width + 12, label_y + text_height + baseline + 8),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1 + 6, label_y + text_height + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def write_page(
    out_dir: Path,
    demo_output: str,
    detections: list[dict[str, object]],
    runtime_line: str,
    wall_time_ms: float,
) -> None:
    rows = "".join(
        f"<tr><td>{html.escape(str(item['label']))}</td>"
        f"<td>{item['confidence']}%</td><td>{html.escape(str(item['box']))}</td></tr>"
        for item in detections
    )
    if not rows:
        rows = "<tr><td colspan=\"3\">No detections parsed</td></tr>"

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>A7A NPU YOLOv8 Demo</title>
<style>
body{{margin:0;background:#0f1115;color:#e8e8e8;font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1120px;margin:0 auto;padding:20px}}
header{{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin-bottom:14px}}
h1{{font-size:24px;margin:0;font-weight:650}}
.stats{{font-size:14px;color:#b8beca;text-align:right}}
.stage{{background:#171a21;border:1px solid #2a2f3a;border-radius:8px;padding:12px}}
img{{display:block;width:100%;height:auto;border-radius:4px}}
table{{width:100%;border-collapse:collapse;margin-top:14px;font-size:14px}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid #2a2f3a}}
th{{color:#b8beca;font-weight:600}}
pre{{white-space:pre-wrap;background:#171a21;border:1px solid #2a2f3a;border-radius:8px;padding:12px;overflow:auto;color:#d6d8df}}
</style>
</head>
<body>
<main>
<header>
<div><h1>A7A NPU YOLOv8 Demo</h1></div>
<div class="stats">{html.escape(runtime_line or "runtime not found")}<br>wall time: {wall_time_ms:.1f} ms</div>
</header>
<section class="stage"><img src="dog_annotated.jpg?t={int(time.time())}" alt="YOLOv8 NPU annotated result"></section>
<table><thead><tr><th>label</th><th>confidence</th><th>box</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Raw Output</h2>
<pre>{html.escape(demo_output)}</pre>
</main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    args = parse_args()
    demo = Path(args.demo)
    model = Path(args.model)
    image = Path(args.image)
    runtime_dir = Path(args.runtime_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{runtime_dir}:{env.get('LD_LIBRARY_PATH', '')}"

    command = [str(demo), "-nb", str(model), "-i", str(image)]
    start = time.monotonic()
    proc = subprocess.run(
        command,
        cwd=str(demo.parent),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    wall_time_ms = (time.monotonic() - start) * 1000.0
    demo_output = proc.stdout

    (out_dir / "demo_output.txt").write_text(demo_output, encoding="utf-8")
    shutil.copy2(image, out_dir / "dog.jpg")

    detections = parse_detections(demo_output)
    draw_detections(image, detections, out_dir / "dog_annotated.jpg")
    write_page(out_dir, demo_output, detections, find_runtime_line(demo_output), wall_time_ms)

    print(f"returncode={proc.returncode}")
    print(f"detections={len(detections)}")
    print(find_runtime_line(demo_output))
    print(out_dir / "index.html")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
