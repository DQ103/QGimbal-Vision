# 摄像头读取并显示画面
# 使用: python main.py --camera 0

"""
简单的摄像头预览脚本（使用 OpenCV）。
参数：
  --camera   摄像头索引（默认 0）
  --fps      是否在画面左上角显示 FPS（可选，0/1，默认 1）

按 'q' 或 ESC 退出。
"""

import argparse
import time
import sys
import math

import cv2
import numpy as np

DEFAULT_CAMERA = 1  # 摄像头索引（默认 0）
DEFAULT_WIDTH = 640  # 期望宽度
DEFAULT_HEIGHT = 480  # 期望高度
DEFAULT_FPS = 120  # 期望帧率


def parse_args():
    p = argparse.ArgumentParser(description="OpenCV 摄像头显示示例")
    p.add_argument('--camera', type=int, default=DEFAULT_CAMERA, help=f'摄像头索引（默认 {DEFAULT_CAMERA}）')
    p.add_argument('--fps', type=int, choices=[0, 1], default=1, help='是否在画面上显示 FPS（0/1）')
    return p.parse_args()


def angle_between(v1, v2):
    # 计算两向量之间的夹角（度数）
    dot = v1.dot(v2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 * n2 == 0:
        return 0.0
    cos = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos))


def detect_rectangles(frame, min_area_ratio=0.005, max_area_ratio=0.5, angle_tol=25.0):
    """
    在输入 BGR 图像中检测矩形（包括旋转矩形）。返回矩形的 box points 和相关信息。
    - min_area_ratio: 与图像面积的最小比率（过小的轮廓会被丢弃）
    - angle_tol: 角度容忍度（判断为矩形时，四个角接近 90 度的容差）
    """
    h, w = frame.shape[:2]
    img_area = h * w
    min_area = img_area * min_area_ratio
    max_area = img_area * max_area_ratio

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 大津法二值化
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    opened = cv2.erode(opened, kernel, iterations=1)

    # 边缘检测
    edges = cv2.Canny(opened, 25, 75)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # 尝试多边形逼近
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2).astype(np.float32)

            # 验证角度接近直角
            angles = []
            for i in range(4):
                p0 = pts[i]
                p1 = pts[(i + 1) % 4]
                p2 = pts[(i + 2) % 4]
                angles.append(angle_between(p0 - p1, p2 - p1))
            # 确保每个角都在容差范围内或矩形足够规则
            if all(abs(a - 90) < angle_tol for a in angles):
                # 计算最短边与最长边比率，确保不是过于扭曲的矩形
                dists = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
                min_dist = min(dists)
                max_dist = max(dists)
                if max_dist / min_dist > 3:
                    continue
                rects.append({'center': tuple(np.mean(pts, axis=0)), 'box': pts, 'area': area})
    # 按面积降序返回（优先较大的矩形）
    rects = sorted(rects, key=lambda r: r['area'], reverse=True)
    return rects


def draw_detected_rect(frame, r):
    """在图像上绘制单个检测到的矩形并标注信息（如果 r 为 None 则不绘制）。"""
    if r is None:
        return
    box = r['box'].astype(np.int32)
    cv2.polylines(frame, [box], isClosed=True, color=(0, 255, 0), thickness=3)
    (cx, cy) = r['center']
    label = f"A:{int(r['area'])}"
    cv2.putText(frame, label, (int(cx) - 80, int(cy) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"无法打开摄像头索引 {args.camera}. 请检查设备或更换索引。")
        sys.exit(2)

    # 设置参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, DEFAULT_FPS)

    win_name = f"Camera {args.camera}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("无法从摄像头读取到帧，正在重试...")
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, -1)

            # 对每帧执行矩形检测
            rects = detect_rectangles(frame, min_area_ratio=0.005, max_area_ratio=0.5, angle_tol=25.0)
            if rects:
                draw_detected_rect(frame, rects[0])

            # 计算 FPS（指数移动平均以平滑显示）
            if args.fps:
                now = time.time()
                dt = now - prev_time
                prev_time = now
                if dt > 0:
                    alpha = 0.98
                    inst_fps = 1.0 / dt
                    fps = alpha * fps + (1 - alpha) * inst_fps if fps > 0 else inst_fps
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow(win_name, frame)

            key = cv2.waitKey(1) & 0xFF
            # 按 'q' 或 ESC 退出
            if key == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        print('\n收到中断，退出...')
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
