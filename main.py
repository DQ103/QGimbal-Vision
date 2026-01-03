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

from control.config import ControlConfig
from control.serial_stub import GimbalSerialStub
from control.tracker_control import GimbalTracker

DEFAULT_CAMERA = 1  # 摄像头索引（默认 0）
DEFAULT_WIDTH = 640  # 期望宽度
DEFAULT_HEIGHT = 480  # 期望高度
DEFAULT_FPS = 120  # 期望帧率
DEFAULT_DISPLAY = 1
DEFAULT_PRINT_INTERVAL = 0.05

# 控制默认参数（可通过命令行覆盖）
DEFAULT_CONTROL_ENABLED = 1
DEFAULT_MAX_RPM = 20.0
DEFAULT_DEADBAND_PX = 0.0
DEFAULT_LOST_TIMEOUT_S = 0.4


def parse_args():
    p = argparse.ArgumentParser(description="OpenCV 摄像头显示示例")
    p.add_argument('--camera', type=int, default=DEFAULT_CAMERA, help=f'摄像头索引（默认 {DEFAULT_CAMERA}）')
    p.add_argument('--display', type=int, choices=[0, 1], default=DEFAULT_DISPLAY,
                   help=f'是否显示图形化窗口（0/1，默认 {DEFAULT_DISPLAY}）')
    p.add_argument('--print-interval', type=float, default=DEFAULT_PRINT_INTERVAL,
                   help=f'终端输出间隔秒数（仅 --display 0 生效，默认 {DEFAULT_PRINT_INTERVAL}）')

    # 控制相关
    p.add_argument('--control', type=int, choices=[0, 1], default=DEFAULT_CONTROL_ENABLED,
                   help=f'是否启用 PID 控制输出（0/1，默认 {DEFAULT_CONTROL_ENABLED}）')
    p.add_argument('--max-rpm', type=float, default=DEFAULT_MAX_RPM,
                   help=f'最大转速输出（RPM，默认 {DEFAULT_MAX_RPM}）')
    p.add_argument('--deadband-px', type=float, default=DEFAULT_DEADBAND_PX,
                   help=f'像素死区（默认 {DEFAULT_DEADBAND_PX}）')
    p.add_argument('--lost-timeout', type=float, default=DEFAULT_LOST_TIMEOUT_S,
                   help=f'丢目标超时后复位控制器的时间（秒，默认 {DEFAULT_LOST_TIMEOUT_S}）')

    # 串口相关（协议在 control/serial_stub.py 内实现）
    p.add_argument('--serial-port', type=str, default=None, help='串口端口号，例如 COM3；不填则不发送')
    p.add_argument('--serial-baud', type=int, default=115200, help='串口波特率（默认 115200）')

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

    # 控制器初始化（串口协议先留 stub，你后续替换 send_rpm 即可）
    ctrl_cfg = ControlConfig(
        enabled=bool(args.control),
        deadband_px=float(args.deadband_px),
        lost_timeout_s=float(args.lost_timeout),
        max_rpm_yaw=float(args.max_rpm),
        max_rpm_pitch=float(args.max_rpm),
    )
    tracker = GimbalTracker(ctrl_cfg)
    serial = GimbalSerialStub(port=args.serial_port, baudrate=int(args.serial_baud))
    serial.open()

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

            # PID 控制：将目标中心追踪到屏幕中心，输出 yaw/pitch rpm
            h, w = frame.shape[:2]
            target_center = best.center if best is not None else None
            ret, ctrl_out = tracker.update(frame_w=w, frame_h=h, target_center=target_center, dt=max(dt, 1e-6), now=now)
            if ret:
                serial.send_rpm(ctrl_out.yaw_rpm, ctrl_out.pitch_rpm)

            if display:
                if best is not None:
                    draw_detected_rect(frame, best)

                # 画面中心点
                cv2.drawMarker(frame, (w // 2, h // 2), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"err(px)=({ctrl_out.err_x_px:.0f},{ctrl_out.err_y_px:.0f}) rpm=({ctrl_out.yaw_rpm:.1f},{ctrl_out.pitch_rpm:.1f})",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

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
                        print(f"fps={fps:.1f} rect=none rpm=({ctrl_out.yaw_rpm:.1f},{ctrl_out.pitch_rpm:.1f})")
                    else:
                        cx, cy = best.center
                        area = best.area
                        print(
                            f"fps={fps:.1f} cx={cx:.1f} cy={cy:.1f} area={area:.0f} "
                            f"err=({ctrl_out.err_x_px:.0f},{ctrl_out.err_y_px:.0f}) rpm=({ctrl_out.yaw_rpm:.1f},{ctrl_out.pitch_rpm:.1f})"
                        )

    except KeyboardInterrupt:
        print('\n收到中断，退出...')
    finally:
        cap.release()
        serial.close()
        if display:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
