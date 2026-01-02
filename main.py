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
import cv2

DEFAULT_CAMERA = 1  # 摄像头索引（默认 0）
DEFAULT_WIDTH = 640  # 期望宽度
DEFAULT_HEIGHT = 480  # 期望高度
DEFAULT_FPS = 120  # 期望帧率


def parse_args():
    p = argparse.ArgumentParser(description="OpenCV 摄像头显示示例")
    p.add_argument('--camera', type=int, default=DEFAULT_CAMERA, help=f'摄像头索引（默认 {DEFAULT_CAMERA}）')
    p.add_argument('--fps', type=int, choices=[0, 1], default=1, help='是否在画面上显示 FPS（0/1）')
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
