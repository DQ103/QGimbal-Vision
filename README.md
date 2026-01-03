# Gimbal-vision

一个简单的 OpenCV 摄像头预览 + 矩形检测示例。

## 结构

- `main.py`：摄像头采集 / FPS / 显示与输出
- `vision/rect_detect.py`：矩形检测与绘制（已从 `main.py` 抽离）

## 安装依赖

```powershell
pip install -r requirements.txt
```

## 运行

GUI 模式（显示窗口，按 `q` 或 `ESC` 退出）：

```powershell
python main.py --camera 0 --display 1
```

无窗口模式（只在终端输出 FPS + 检测到的矩形中心点/面积，按 `Ctrl+C` 退出）：

```powershell
python main.py --camera 0 --display 0 --print-interval 0.5
```

## 说明

`vision.rect_detect.detect_rectangles()` 返回按面积从大到小排序的矩形列表；`main.py` 默认取第 1 个作为 `best`。

## 追踪控制（PID）

项目已加入一个“识别结果(cx,cy) → 双轴 PID → yaw/pitch 速度(rpm)”的控制模块：

- `control/pid.py`：基础 PID（积分限幅/输出限幅）
- `control/tracker_control.py`：将图像误差映射为 `yaw_rpm/pitch_rpm`
- `control/serial_stub.py`：串口发送 stub（目前 no-op，协议部分你后续补上）

### 坐标系约定

- 图像坐标：x 向右为正，y 向下为正
- 误差定义：`err = target_center - image_center`
- 云台正方向未知时，可用 `ControlConfig.invert_yaw/invert_pitch` 反转输出方向

### 运行示例

启用控制输出（默认已启用），并设置最大输出 rpm / 死区：

```powershell
python main.py --camera 0 --display 1 --control 1 --max-rpm 120 --deadband-px 6
```

无窗口模式查看控制输出：

```powershell
python main.py --camera 0 --display 0 --control 1 --print-interval 0.1
```

> 注意：当前 `send_rpm()` 是空实现，不会实际控制云台。你把协议写好后，只需要替换 `control/serial_stub.py` 中的发送逻辑。
