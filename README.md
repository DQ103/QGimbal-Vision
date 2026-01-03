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

