# 摄像头读取并显示画面
# 使用: python main.py --camera 0

"""
简单的摄像头预览脚本（使用 OpenCV）。
参数：
  --camera        摄像头索引（默认 0）
  --display       是否显示图形化窗口（0/1，默认 1）
                 - 1：显示图像（现有效果），并叠加矩形与 FPS
                 - 0：不显示窗口，在终端输出 FPS + 检测到的矩形中心点坐标/面积
  --print-interval 终端输出间隔秒数（仅 --display 0 时生效，默认 0.5）

GUI 模式按 'q' 或 ESC 退出；无窗口模式请按 Ctrl+C 退出。
"""

import argparse
import time
import sys

import cv2

from vision.rect_detect import detect_rectangles, draw_detected_rect

DEFAULT_CAMERA = 1  # 摄像头索引（默认 0）
DEFAULT_WIDTH = 640  # 期望宽度
DEFAULT_HEIGHT = 480  # 期望高度
DEFAULT_FPS = 120  # 期望帧率
DEFAULT_DISPLAY = 1
DEFAULT_PRINT_INTERVAL = 0.05


def parse_args():
    p = argparse.ArgumentParser(description="OpenCV 摄像头显示示例")
    p.add_argument('--camera', type=int, default=DEFAULT_CAMERA, help=f'摄像头索引（默认 {DEFAULT_CAMERA}）')
    p.add_argument('--display', type=int, choices=[0, 1], default=DEFAULT_DISPLAY,
                   help=f'是否显示图形化窗口（0/1，默认 {DEFAULT_DISPLAY}）')
    p.add_argument('--print-interval', type=float, default=DEFAULT_PRINT_INTERVAL,
                   help=f'终端输出间隔秒数（仅 --display 0 生效，默认 {DEFAULT_PRINT_INTERVAL}）')
    return p.parse_args()


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

    display = bool(args.display)

    win_name = f"Camera {args.camera}"
    if display:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # 无窗口模式：终端输出节流
    last_print = 0.0
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
            best = rects[0] if rects else None

            # 计算 FPS（指数移动平均以平滑显示）
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                alpha = 0.98
                inst_fps = 1.0 / dt
                fps = alpha * fps + (1 - alpha) * inst_fps if fps > 0 else inst_fps

            if display:
                if best is not None:
                    draw_detected_rect(frame, best)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow(win_name, frame)
                key = cv2.waitKey(1) & 0xFF
                # 按 'q' 或 ESC 退出
                if key == ord('q') or key == 27:
                    break
            else:
                # 无窗口：终端输出 FPS + 检测结果（按间隔打印，避免刷屏）
                if args.print_interval <= 0 or (now - last_print) >= args.print_interval:
                    last_print = now
                    if best is None:
                        print(f"fps={fps:.1f} rect=none")
                    else:
                        cx, cy = best.center
                        area = best.area
                        print(f"fps={fps:.1f} cx={cx:.1f} cy={cy:.1f} area={area:.0f}")

    except KeyboardInterrupt:
        print('\n收到中断，退出...')
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
