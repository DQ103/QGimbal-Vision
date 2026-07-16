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

Radxa Cubie A7A + Radxa Camera 8M 219 在 Armbian vendor 内核下建议先用
GStreamer 测试视频预览。脚本默认使用 `DISPLAY=:0`、`NV12`、`xvimagesink`，
与官方示例一致。VNC 桌面中运行：

```bash
scripts/vnc_imx219_preview.sh
```

如果窗口打不开，可换 sink：

```bash
scripts/vnc_imx219_preview.sh --sink xvimagesink
scripts/vnc_imx219_preview.sh --sink autovideosink
```

如果要验证最大出帧率，可关闭 AWISP：

```bash
scripts/vnc_imx219_preview.sh --awisp 0
```

Radxa 上建议使用系统 OpenCV，避免 pip 版 OpenCV 缺少 GStreamer 支持：

```bash
sudo apt-get install -y python3-opencv python3-numpy v4l-utils
```

运行本项目采集与识别：

```bash
cd ~/QGimbal-Vision
python3 main.py --display 0 --print-interval 1 --size 1920x1080 --fps 30 --format NV12 --capture-mode raw --awisp 0 --detect-scale 0.5 --flip 0
```

上面的命令是实时控制推荐路径：NV12 直接进 OpenCV，只用 Y 平面做检测，
不经过 `videoconvert` 转 BGR，并关闭 AWISP。实测在 1920x1080 下稳定约 30 FPS。
如果使用官方 `en-awisp=1`，当前板子上 1920x1080 实测约 21-22 FPS，且可能继续降帧。
`--flip 1` 会启用软件 180 度翻转，会降低性能；需要旧 BGR 管线时可用
`--capture-mode bgr`，但帧率会明显下降。

当前矩形检测默认启用多轮检测和目标评分：

- `--detect-multi-pass 1`：先走原 Otsu+morphology 路径，失败后逐步放宽 Canny / 多边形逼近 / 角度容忍参数
- `--rect-center-weight 0.25`：目标越靠近画面中心，评分越高
- `--rect-prev-weight 0.70`：目标越接近上一帧目标，评分越高
- `--rect-max-aspect 5.0`：过滤过细长的假矩形
- `--rect-max-area-ratio 0.5`：过滤过大的背景轮廓

这部分借鉴了 K230 版本的多轮检测和 `area + center + previous target` 评分策略，
替代了早期“直接取最大面积矩形”的逻辑。

如需在桌面上显示 OpenCV 窗口：

```bash
DISPLAY=:0 python3 main.py --display 1 --size 1920x1080 --fps 30 --format NV12 --capture-mode raw --awisp 0 --detect-scale 0.5 --flip 0 --display-mode gray --display-scale 0.5
```

桌面显示建议用灰度缩放模式，主要用于调试画面。`--display-mode color`
会把 NV12 转成 BGR 用于 `imshow()`，更吃 CPU。高分辨率 3280x2464 可采集，
但不建议用于实时控制。

纯 CLI 镜像或不启动桌面时，可以开启 MJPEG HTTP 预览流：

```bash
python3 main.py --display 0 --size 1920x1080 --fps 30 --format NV12 \
  --capture-mode raw --awisp 0 --largemode 0 --detect-scale 0.25 \
  --display-mode color --stream-port 8080 --stream-scale 0.33 \
  --stream-every 8 --stream-quality 65
```

然后在本机浏览器打开：

```text
http://192.168.0.98:8080/
```

网页右侧提供预览调色滑杆：

- `Realtime`：低成本实时预览预设，优先保证 30 FPS
- `Soft` / `Balanced` / `Strong`：针对 A7A + IMX219 + `--awisp 0` 的自动调色预设，适合校准观察，但更吃 CPU
- `Auto WB` / `Auto Levels`：自动白平衡和亮度范围拉伸
- `Black point` / `White point`：自动亮度拉伸使用的黑白点百分位
- `Brightness` / `Contrast` / `Gamma`：调整明暗和灰阶
- `Saturation` / `Vibrance` / `Hue`：调整饱和度、低饱和区域增强和色相
- `Red gain` / `Green gain` / `Blue gain`：调整 RGB 通道增益
- `Clarity` / `Sharpness`：局部对比和锐化

这些调色只作用于 MJPEG 预览流，不改变矩形检测和控制使用的原始 Y 平面。
如果要尝试底层 V4L2 控制项，可以在启动时追加：

```bash
--v4l2-ctrl auto_exposure_bias=6,wide_dynamic_range=1,color_effects=9
```

当前实测这些控制项对 `--awisp 0` 的 raw/NV12 路径改善有限；稳定 30 FPS 的主路径仍建议
关闭 AWISP，再用网页预览调色辅助观察。

### P4 YOLO/NPU 分支

`p4-yolo-npu` 分支加入了外部 YOLO/NPU worker 的接入骨架。默认仍使用传统 CV：

```bash
--detector rect
```

如果已经有板端 NPU/VIPLite worker，可以切换：

```bash
--detector yolo --yolo-command "./build/a7a_yolo_worker --model model.nbg"
```

比赛更推荐 hybrid 模式：YOLO 有结果时使用 YOLO，YOLO 无结果时回落到传统矩形检测。
YOLO worker 在后台线程异步运行，主循环不会等待 NPU/YOLO 返回。

```bash
python3 main.py --display 0 --size 1920x1080 --fps 30 --format NV12 \
  --capture-mode raw --awisp 0 --detect-scale 0.25 \
  --detector hybrid --yolo-scale 0.33 \
  --yolo-every 8 \
  --yolo-command "python3 scripts/yolo_json_worker_stub.py"
```

验证 YOLO 结果接入主流程时，用 fixed worker：

```bash
python3 main.py --display 0 --size 1920x1080 --fps 30 --format NV12 \
  --capture-mode raw --awisp 0 --detect-scale 0.25 \
  --detector hybrid --yolo-scale 0.33 --yolo-every 8 \
  --yolo-command "python3 scripts/yolo_json_worker_fixed.py --bbox 0.42,0.35,0.58,0.65"
```

看到日志或画面里 `pass=100`，说明目标来自 YOLO worker。

Python 侧与外部 worker 使用 JSON Lines 协议：

```json
{"frame_id":1,"width":640,"height":360,"format":"jpg_b64","image":"..."}
```

worker 每行返回：

```json
{"frame_id":1,"detections":[{"bbox":[x1,y1,x2,y2],"confidence":0.9,"label":"target"}]}
```

可用 worker/adapter：

- `scripts/yolo_json_worker_stub.py`：空检测，用于验证 fallback
- `scripts/yolo_json_worker_fixed.py`：固定框，用于验证 YOLO 结果进入 PID
- `scripts/yolo_json_worker_cli_adapter.py`：把一次性外部 NPU demo 包成 JSONL worker
- `npu_worker/a7a_yolo_worker.cpp`：常驻 C++ worker 骨架，后续替换为 VIPLite/NBG 推理

真正 NPU worker 只要遵守同一协议即可替换。最终性能建议使用常驻 C++ worker，并把
JPEG/base64 协议改成 raw NV12/BGR 或共享内存。

也可以用 VLC/ffplay 打开：

```text
http://192.168.0.98:8080/stream.mjpg
```

`--stream-every 8` 表示每 8 帧推送 1 帧预览，不影响每帧检测和控制。若要更流畅的预览，
可以调到 `5`；若调色后性能余量不足，优先使用 `Realtime` 或关闭自动调色项。

无窗口模式（只在终端输出 FPS + 检测到的矩形中心点/面积，按 `Ctrl+C` 退出）：

```powershell
python main.py --camera 0 --display 0 --print-interval 0.5
```

## 说明

`vision.rect_detect.detect_rectangles()` 保留单轮检测兼容接口；
`detect_rectangles_multi_pass()` 会按多轮参数返回第一轮成功的候选矩形。
`main.py` 使用 `RectSelector` 从候选矩形中综合面积、中心距离、历史连续性选出 `best`。

## 追踪控制（PID）

项目已加入一个“识别结果(cx,cy) → 双轴 PID → yaw/pitch 速度(rpm)”的控制模块：

- `control/pid.py`：基础 PID（积分限幅/输出限幅）
- `control/tracker_control.py`：将图像误差映射为 `yaw_rpm/pitch_rpm`
- `control/serial_stub.py`：QGimbal V1 串口适配器（10 字节命令、42 字节反馈、CRC8）

### 坐标系约定

- 图像坐标：x 向右为正，y 向下为正
- 误差定义：`err = target_center - image_center`
- 云台正方向未知时，可用 `ControlConfig.invert_yaw/invert_pitch` 反转输出方向

### 运行示例

指定串口并显式启用控制输出，设置最大输出 rpm / 死区：

```powershell
python main.py --camera 0 --display 1 --control 1 --serial-port /dev/ttyUSB0 --serial-baud 115200 --max-rpm 120 --deadband-px 6
```

Radxa 启动脚本也支持环境变量：

```bash
CONTROL=1 SERIAL_PORT=/dev/ttyUSB0 SERIAL_BAUD=115200 \
INVERT_YAW=1 INVERT_PITCH=0 scripts/run_a7a_imx415_e25_web.sh
```

无窗口模式查看控制输出：

```powershell
python main.py --camera 0 --display 0 --control 1 --print-interval 0.1
```

> 安全默认值为 `--control 0`。启用后程序会先发送 Enable 并等待有效反馈；丢失目标会立即持续发送零速度，退出时发送零速度和 Disable。
