import time
import os

from media.sensor import *
from media.display import *
from media.media import *
from machine import UART
from machine import FPIOA

import cv_lite


PICTURE_WIDTH = 400
PICTURE_HEIGHT = 240

DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 480

UART_TX_PIN = 11
UART_RX_PIN = 12
UART_BAUDRATE = 9600

# Target rectangle detection.
ROI_SCALE_W = 1.0
ROI_SCALE_H = 1.0
MIN_TARGET_AREA = 2000
MAX_TARGET_AREA_RATIO = 0.85
MAX_TARGET_ASPECT_RATIO = 5.0

DETECT_PASSES = (
    (80, 150, 0.04, 0.30, 5),
    (60, 140, 0.05, 0.45, 5),
    (40, 120, 0.06, 0.60, 3),
)

TARGET_DETECT_INTERVAL = 1
TARGET_LOST_LIMIT = 3
TARGET_CENTER_WEIGHT = 0.25
TARGET_PREV_WEIGHT = 0.70
# 405nm blue-violet laser spot detection in LAB color space.
# Laser detection is only enabled after a target rectangle is available, and
# searches only near that rectangle. This avoids locking onto room reflections.
VIOLET_THRESHOLD_PASSES = (
    ((35, 100, 0, 127, -128, -84),),
    ((15, 100, 0, 127, -128, -75),),
)

STRICT_ONLY = True
USE_BRIGHTEST_CORE_REFINE = True
REQUIRE_BRIGHT_CORE = True
BRIGHT_MIN_LUMA = 150
BRIGHT_DELTA_FROM_MAX = 25
USE_BRIGHT_CORE_REFINE = True
BRIGHT_CORE_THRESHOLDS = (
    (82, 100, -128, 127, -128, 127),
)
CORE_SEARCH_PAD = 10
CORE_MIN_PIXELS = 1
CORE_MAX_PIXELS = 180
CORE_MAX_RECT_SIDE = 24
CORE_MAX_ASPECT_RATIO = 3.0

LASER_MIN_PIXELS = 2
LASER_MIN_AREA = 2
LASER_MAX_PIXELS = 220
LASER_IDEAL_PIXELS = 18
LASER_MAX_RECT_SIDE = 28
LASER_MAX_ASPECT_RATIO = 2.5
LASER_TARGET_PAD = 8
LASER_SEARCH_PAD = 70
LASER_LOST_LIMIT = 3
LASER_MERGE_MARGIN = 4
LASER_KEEP_WEIGHT = 260.0
LASER_SIZE_WEIGHT = 160.0
LASER_ROUND_WEIGHT = 120.0
LASER_DENSITY_WEIGHT = 80.0
LASER_CORE_WEIGHT = 320.0

DRAW_ROI = False
DRAW_IMAGE_CENTER = True
PRINT_EVERY_FRAME = True
PRINT_FULL_PACKET = False

# Stage 1: move the target rectangle center close to the calibrated aim center.
# M0 can turn on the laser after AIM_READY becomes 1.
AIM_CENTER_X = 206
AIM_CENTER_Y = 122
AIM_READY_THRESHOLD_X = 4
AIM_READY_THRESHOLD_Y = 4

# Keep these off at first. Enable only after the basic image is stable.
LOCK_CAMERA_EXPOSURE = False
EXPOSURE_US = 12000
GAIN_DB = 12


sensor = None
uart = None


def clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def safe_sensor_call(obj, name, *args, **kwargs):
    fn = getattr(obj, name, None)
    if not callable(fn):
        return
    try:
        fn(*args, **kwargs)
    except BaseException as e:
        print("warn:", name, e)


def draw_text(img, x, y, text, color=(255, 255, 255), scale=1):
    fn = getattr(img, "draw_string_advanced", None)
    if callable(fn):
        fn(x, y, 16 * scale, text, color=color)
        return
    img.draw_string(x, y, text, color=color, scale=scale)


def rect_area(rect):
    return float(rect[2]) * float(rect[3])


def rect_aspect(rect):
    w = float(rect[2])
    h = float(rect[3])
    if w <= 0 or h <= 0:
        return 999.0
    return max(w, h) / min(w, h)


def rect_corners_in_full_image(rect, roi_x, roi_y):
    pts = rect[4:]
    if len(pts) < 8:
        return None
    return [
        (float(pts[i]) + roi_x, float(pts[i + 1]) + roi_y)
        for i in range(0, 8, 2)
    ]


def line_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 0.001:
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return px, py


def rect_center_in_full_image(rect, roi_x, roi_y):
    corners = rect_corners_in_full_image(rect, roi_x, roi_y)
    if corners:
        center = line_intersection(corners[0], corners[2], corners[1], corners[3])
        if center:
            return center

    x = float(rect[0])
    y = float(rect[1])
    w = float(rect[2])
    h = float(rect[3])
    return x + w / 2.0 + roi_x, y + h / 2.0 + roi_y


def rect_bbox_in_full_image(rect, roi_x, roi_y):
    corners = rect_corners_in_full_image(rect, roi_x, roi_y)
    if corners:
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        x = int(clamp(min(xs), 0, PICTURE_WIDTH - 1))
        y = int(clamp(min(ys), 0, PICTURE_HEIGHT - 1))
        x2 = int(clamp(max(xs), 1, PICTURE_WIDTH))
        y2 = int(clamp(max(ys), 1, PICTURE_HEIGHT))
        return (x, y, max(1, x2 - x), max(1, y2 - y))

    x = int(clamp(float(rect[0]) + roi_x, 0, PICTURE_WIDTH - 1))
    y = int(clamp(float(rect[1]) + roi_y, 0, PICTURE_HEIGHT - 1))
    w = int(clamp(float(rect[2]), 1, PICTURE_WIDTH - x))
    h = int(clamp(float(rect[3]), 1, PICTURE_HEIGHT - y))
    return (x, y, w, h)


def expand_roi(roi, pad):
    x, y, w, h = roi
    x1 = clamp(x - pad, 0, PICTURE_WIDTH - 1)
    y1 = clamp(y - pad, 0, PICTURE_HEIGHT - 1)
    x2 = clamp(x + w + pad, 1, PICTURE_WIDTH)
    y2 = clamp(y + h + pad, 1, PICTURE_HEIGHT)
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


def draw_target_rect(img, rect, roi_x, roi_y):
    corners = rect_corners_in_full_image(rect, roi_x, roi_y)
    if corners:
        for i in range(4):
            x1, y1 = int(corners[i][0]), int(corners[i][1])
            x2, y2 = int(corners[(i + 1) % 4][0]), int(corners[(i + 1) % 4][1])
            img.draw_line(x1, y1, x2, y2, color=(0, 255, 0), thickness=4)
        return

    x, y, w, h = rect_bbox_in_full_image(rect, roi_x, roi_y)
    img.draw_rectangle(x, y, w, h, color=(0, 255, 0), thickness=3)


def detect_target_rects(img_np, roi_w, roi_h):
    for i, params in enumerate(DETECT_PASSES):
        canny_th1, canny_th2, epsilon, max_angle_cos, gaussian_ksize = params
        rects = cv_lite.grayscale_find_rectangles_with_corners(
            [roi_h, roi_w],
            img_np,
            canny_th1,
            canny_th2,
            epsilon,
            MIN_TARGET_AREA / (roi_w * roi_h),
            max_angle_cos,
            gaussian_ksize,
        )
        if rects:
            return rects, i
    return [], -1


def pick_best_target(rects, roi_x, roi_y, last_cx, last_cy):
    max_area = PICTURE_WIDTH * PICTURE_HEIGHT * MAX_TARGET_AREA_RATIO
    max_center_dist = ((PICTURE_WIDTH / 2) ** 2 + (PICTURE_HEIGHT / 2) ** 2) ** 0.5

    best = None
    best_score = -1.0

    for rect in rects:
        area = rect_area(rect)
        if area < MIN_TARGET_AREA or area > max_area:
            continue
        if rect_aspect(rect) > MAX_TARGET_ASPECT_RATIO:
            continue

        cx, cy = rect_center_in_full_image(rect, roi_x, roi_y)
        dx_center = cx - PICTURE_WIDTH / 2
        dy_center = cy - PICTURE_HEIGHT / 2
        center_dist = (dx_center * dx_center + dy_center * dy_center) ** 0.5
        center_score = 1.0 - clamp(center_dist / max_center_dist, 0.0, 1.0)
        area_score = clamp(area / max_area, 0.0, 1.0)

        if last_cx is None or last_cy is None:
            prev_score = center_score
        else:
            dx_prev = cx - last_cx
            dy_prev = cy - last_cy
            prev_dist = (dx_prev * dx_prev + dy_prev * dy_prev) ** 0.5
            prev_score = 1.0 - clamp(prev_dist / max_center_dist, 0.0, 1.0)

        score = area_score + TARGET_CENTER_WEIGHT * center_score + TARGET_PREV_WEIGHT * prev_score
        if score > best_score:
            best_score = score
            best = rect

    return best


class TargetTracker:
    def __init__(self, roi_x, roi_y, roi_w, roi_h):
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.roi_w = roi_w
        self.roi_h = roi_h
        self.frame_count = 0
        self.lost_frames = TARGET_LOST_LIMIT
        self.last_rect = None
        self.last_cx = None
        self.last_cy = None
        self.last_area = 0.0
        self.last_bbox = None
        self.last_pass_id = -1

    def update(self, img_np):
        self.frame_count += 1
        should_detect = (
            self.last_rect is None
            or self.lost_frames > 0
            or TARGET_DETECT_INTERVAL <= 1
            or (self.frame_count % TARGET_DETECT_INTERVAL) == 0
        )

        if not should_detect:
            return self.current(), -2

        rects, pass_id = detect_target_rects(img_np, self.roi_w, self.roi_h)
        best = pick_best_target(rects, self.roi_x, self.roi_y, self.last_cx, self.last_cy)

        if best:
            cx, cy = rect_center_in_full_image(best, self.roi_x, self.roi_y)
            self.last_rect = best
            self.last_cx = cx
            self.last_cy = cy
            self.last_area = rect_area(best)
            self.last_bbox = rect_bbox_in_full_image(best, self.roi_x, self.roi_y)
            self.last_pass_id = pass_id
            self.lost_frames = 0
            return self.current(), pass_id

        self.lost_frames += 1
        if self.lost_frames >= TARGET_LOST_LIMIT:
            self.clear()
            return None, -1

        return self.current(), -3

    def current(self):
        if self.last_rect is None:
            return None
        return (self.last_rect, self.last_cx, self.last_cy, self.last_area, self.last_bbox)

    def clear(self):
        self.last_rect = None
        self.last_cx = None
        self.last_cy = None
        self.last_area = 0.0
        self.last_bbox = None
        self.last_pass_id = -1


def blob_attr(blob, name, index, default=0):
    attr = getattr(blob, name, None)
    if callable(attr):
        return attr()
    if attr is not None:
        return attr
    try:
        return blob[index]
    except BaseException:
        return default


def blob_x(blob):
    return int(blob_attr(blob, "x", 0))


def blob_y(blob):
    return int(blob_attr(blob, "y", 1))


def blob_w(blob):
    return int(blob_attr(blob, "w", 2))


def blob_h(blob):
    return int(blob_attr(blob, "h", 3))


def blob_pixels(blob):
    return int(blob_attr(blob, "pixels", 4))


def blob_cx(blob):
    return int(blob_attr(blob, "cx", 5, blob_x(blob) + blob_w(blob) // 2))


def blob_cy(blob):
    return int(blob_attr(blob, "cy", 6, blob_y(blob) + blob_h(blob) // 2))


class CorePoint:
    def __init__(self, x, y, w, h, pixels, cx, cy):
        self._x = int(x)
        self._y = int(y)
        self._w = int(w)
        self._h = int(h)
        self._pixels = int(pixels)
        self._cx = int(cx)
        self._cy = int(cy)

    def x(self):
        return self._x

    def y(self):
        return self._y

    def w(self):
        return self._w

    def h(self):
        return self._h

    def pixels(self):
        return self._pixels

    def cx(self):
        return self._cx

    def cy(self):
        return self._cy


def make_blob_roi(blob, pad):
    x1 = clamp(blob_x(blob) - pad, 0, PICTURE_WIDTH - 1)
    y1 = clamp(blob_y(blob) - pad, 0, PICTURE_HEIGHT - 1)
    x2 = clamp(blob_x(blob) + blob_w(blob) + pad, 1, PICTURE_WIDTH)
    y2 = clamp(blob_y(blob) + blob_h(blob) + pad, 1, PICTURE_HEIGHT)
    return (x1, y1, x2 - x1, y2 - y1)


def blob_is_valid(blob):
    w = blob_w(blob)
    h = blob_h(blob)
    pixels = blob_pixels(blob)

    if pixels < LASER_MIN_PIXELS or pixels > LASER_MAX_PIXELS:
        return False
    if w <= 0 or h <= 0:
        return False
    if w > LASER_MAX_RECT_SIDE or h > LASER_MAX_RECT_SIDE:
        return False
    if max(w, h) / min(w, h) > LASER_MAX_ASPECT_RATIO:
        return False
    return True


def core_is_valid(blob):
    w = blob_w(blob)
    h = blob_h(blob)
    pixels = blob_pixels(blob)

    if pixels < CORE_MIN_PIXELS or pixels > CORE_MAX_PIXELS:
        return False
    if w <= 0 or h <= 0:
        return False
    if w > CORE_MAX_RECT_SIDE or h > CORE_MAX_RECT_SIDE:
        return False
    if max(w, h) / min(w, h) > CORE_MAX_ASPECT_RATIO:
        return False
    return True


def pixel_luma(pixel):
    if isinstance(pixel, tuple) or isinstance(pixel, list):
        if len(pixel) >= 3:
            r = int(pixel[0])
            g = int(pixel[1])
            b = int(pixel[2])
            return (77 * r + 150 * g + 29 * b) // 256
        if len(pixel) == 1:
            return int(pixel[0])
    return int(pixel)


def find_brightest_core_by_pixels(img, violet_blob):
    if not USE_BRIGHTEST_CORE_REFINE:
        return None, None

    roi = make_blob_roi(violet_blob, CORE_SEARCH_PAD)
    rx, ry, rw, rh = roi

    max_luma = -1
    for y in range(ry, ry + rh):
        for x in range(rx, rx + rw):
            try:
                luma = pixel_luma(img.get_pixel(x, y))
            except BaseException:
                return None, roi
            if luma > max_luma:
                max_luma = luma

    if max_luma < BRIGHT_MIN_LUMA:
        return None, roi

    cutoff = max_luma - BRIGHT_DELTA_FROM_MAX
    if cutoff < BRIGHT_MIN_LUMA:
        cutoff = BRIGHT_MIN_LUMA

    sum_w = 0
    sum_x = 0
    sum_y = 0
    count = 0
    min_x = PICTURE_WIDTH
    min_y = PICTURE_HEIGHT
    max_x = 0
    max_y = 0

    for y in range(ry, ry + rh):
        for x in range(rx, rx + rw):
            luma = pixel_luma(img.get_pixel(x, y))
            if luma < cutoff:
                continue
            weight = luma - cutoff + 1
            sum_w += weight
            sum_x += x * weight
            sum_y += y * weight
            count += 1
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

    if count <= 0 or sum_w <= 0:
        return None, roi
    if count > CORE_MAX_PIXELS:
        return None, roi

    cx = int(sum_x / sum_w)
    cy = int(sum_y / sum_w)
    core = CorePoint(min_x, min_y, max(1, max_x - min_x + 1), max(1, max_y - min_y + 1), count, cx, cy)
    if not core_is_valid(core):
        return None, roi

    return core, roi


def find_bright_core_by_blob(img, violet_blob):
    if not USE_BRIGHT_CORE_REFINE:
        return None, None

    roi = make_blob_roi(violet_blob, CORE_SEARCH_PAD)
    cores = img.find_blobs(
        BRIGHT_CORE_THRESHOLDS,
        roi=roi,
        pixels_threshold=CORE_MIN_PIXELS,
        area_threshold=CORE_MIN_PIXELS,
        merge=True,
        margin=1,
    )

    best = None
    best_score = -1.0
    vx = blob_cx(violet_blob)
    vy = blob_cy(violet_blob)

    for core in cores:
        if not core_is_valid(core):
            continue

        dx = blob_cx(core) - vx
        dy = blob_cy(core) - vy
        dist = (dx * dx + dy * dy) ** 0.5
        close_score = 1.0 - clamp(dist / max(1.0, float(CORE_SEARCH_PAD + max(blob_w(violet_blob), blob_h(violet_blob)))), 0.0, 1.0)
        size_score = clamp(float(blob_pixels(core)) / 20.0, 0.0, 1.0)
        score = 200.0 * close_score + 80.0 * size_score

        if score > best_score:
            best_score = score
            best = core

    return best, roi


def find_bright_core(img, violet_blob):
    core, roi = find_brightest_core_by_pixels(img, violet_blob)
    if core is not None:
        return core, roi
    return find_bright_core_by_blob(img, violet_blob)


def spot_cx(blob, core):
    if core is not None:
        return blob_cx(core)
    return blob_cx(blob)


def spot_cy(blob, core):
    if core is not None:
        return blob_cy(core)
    return blob_cy(blob)


def blob_density(blob):
    area = blob_w(blob) * blob_h(blob)
    if area <= 0:
        return 0.0
    return clamp(float(blob_pixels(blob)) / float(area), 0.0, 1.0)


def point_in_roi(x, y, roi):
    rx, ry, rw, rh = roi
    return x >= rx and x <= rx + rw and y >= ry and y <= ry + rh


def score_laser_blob(blob, last_x, last_y, core):
    pixels = blob_pixels(blob)

    if pixels <= LASER_IDEAL_PIXELS:
        size_score = clamp(float(pixels) / float(LASER_IDEAL_PIXELS), 0.0, 1.0)
    else:
        size_score = 1.0 - clamp(float(pixels - LASER_IDEAL_PIXELS) / float(LASER_MAX_PIXELS - LASER_IDEAL_PIXELS), 0.0, 1.0)

    score = LASER_SIZE_WEIGHT * size_score

    if last_x is not None and last_y is not None:
        dx = spot_cx(blob, core) - last_x
        dy = spot_cy(blob, core) - last_y
        dist = (dx * dx + dy * dy) ** 0.5
        keep_score = 1.0 - clamp(dist / LASER_SEARCH_PAD, 0.0, 1.0)
        score += LASER_KEEP_WEIGHT * keep_score

    w = blob_w(blob)
    h = blob_h(blob)
    aspect = max(w, h) / min(w, h)
    round_score = 1.0 - clamp((aspect - 1.0) / (LASER_MAX_ASPECT_RATIO - 1.0), 0.0, 1.0)
    score += LASER_ROUND_WEIGHT * round_score
    score += LASER_DENSITY_WEIGHT * blob_density(blob)
    if core is not None:
        score += LASER_CORE_WEIGHT

    return score


class LaserTracker:
    def __init__(self):
        self.last_x = None
        self.last_y = None
        self.lost_frames = LASER_LOST_LIMIT

    def reset(self):
        self.last_x = None
        self.last_y = None
        self.lost_frames = LASER_LOST_LIMIT

    def find_best_in_roi(self, img, roi):
        best = None
        best_score = -1.0
        best_pass_id = -1
        best_core = None
        best_core_roi = None
        candidate_count = 0

        for pass_id, thresholds in enumerate(VIOLET_THRESHOLD_PASSES):
            if STRICT_ONLY and pass_id > 0:
                break

            blobs = img.find_blobs(
                thresholds,
                roi=roi,
                pixels_threshold=LASER_MIN_PIXELS,
                area_threshold=LASER_MIN_AREA,
                merge=True,
                margin=LASER_MERGE_MARGIN,
            )

            pass_best = None
            pass_best_score = -1.0
            pass_best_core = None
            pass_best_core_roi = None

            for blob in blobs:
                if not blob_is_valid(blob):
                    continue

                core, core_roi = find_bright_core(img, blob)
                if REQUIRE_BRIGHT_CORE and core is None:
                    continue

                candidate_count += 1
                score = score_laser_blob(blob, self.last_x, self.last_y, core)
                if score > pass_best_score:
                    pass_best_score = score
                    pass_best = blob
                    pass_best_core = core
                    pass_best_core_roi = core_roi

            if pass_best is not None:
                best = pass_best
                best_score = pass_best_score
                best_pass_id = pass_id
                best_core = pass_best_core
                best_core_roi = pass_best_core_roi
                break

        return best, best_pass_id, best_score, candidate_count, best_core, best_core_roi

    def update(self, img, target_bbox):
        if target_bbox is None:
            self.reset()
            return None, None, -1, 0.0, 0, None, None

        target_roi = expand_roi(target_bbox, LASER_TARGET_PAD)
        best, pass_id, score, candidate_count, core, core_roi = self.find_best_in_roi(img, target_roi)

        if best:
            self.set_last(best, core)
            return best, target_roi, pass_id, score, candidate_count, core, core_roi

        self.lost_frames += 1
        if self.lost_frames >= LASER_LOST_LIMIT:
            self.last_x = None
            self.last_y = None
        return None, target_roi, -1, 0.0, candidate_count, None, None

    def set_last(self, blob, core):
        self.last_x = spot_cx(blob, core)
        self.last_y = spot_cy(blob, core)
        self.lost_frames = 0


def send_fusion_packet(target, laser, laser_core, target_visible_for_laser):
    if target:
        tv = 1
        tx = int(target[1])
        ty = int(target[2])
    else:
        tv = 0
        tx = 0
        ty = 0

    if laser:
        lv = 1
        lx = spot_cx(laser, laser_core)
        ly = spot_cy(laser, laser_core)
    else:
        lv = 0
        lx = 0
        ly = 0

    if target and laser:
        stage = 2
        ex = tx - lx
        ey = ty - ly
    elif target:
        stage = 1
        ex = tx - AIM_CENTER_X
        ey = ty - AIM_CENTER_Y
    else:
        stage = 0
        ex = 0
        ey = 0

    aim_ready = 1 if (
        target_visible_for_laser
        and abs(tx - AIM_CENTER_X) <= AIM_READY_THRESHOLD_X
        and abs(ty - AIM_CENTER_Y) <= AIM_READY_THRESHOLD_Y
    ) else 0

    packet = "@3,{},{},{},{},{},{},{},{},{},{}\r\n".format(
        stage,
        tv,
        tx,
        ty,
        lv,
        lx,
        ly,
        ex,
        ey,
        aim_ready,
    )
    if PRINT_EVERY_FRAME:
        if PRINT_FULL_PACKET:
            print(packet, end="")
        else:
            print("dbg s:{} e:{},{} r:{}".format(stage, ex, ey, aim_ready))
    if uart:
        uart.write(packet)


def init_hardware():
    global sensor, uart

    os.exitpoint(os.EXITPOINT_ENABLE)

    fpioa = FPIOA()
    fpioa.set_function(UART_TX_PIN, FPIOA.UART2_TXD)
    fpioa.set_function(UART_RX_PIN, FPIOA.UART2_RXD)

    sensor = Sensor(id=2)
    sensor.reset()
    sensor.set_vflip(True)
    sensor.set_hmirror(True)
    sensor.set_framesize(width=PICTURE_WIDTH, height=PICTURE_HEIGHT, chn=CAM_CHN_ID_0)
    sensor.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_0)

    if LOCK_CAMERA_EXPOSURE:
        safe_sensor_call(sensor, "set_auto_exposure", False, exposure_us=EXPOSURE_US)
        safe_sensor_call(sensor, "set_auto_gain", False, gain_db=GAIN_DB)
        safe_sensor_call(sensor, "set_auto_whitebal", False)

    Display.init(Display.ST7701, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, to_ide=True)
    MediaManager.init()
    sensor.run()

    uart = UART(
        UART.UART2,
        baudrate=UART_BAUDRATE,
        bits=UART.EIGHTBITS,
        parity=UART.PARITY_NONE,
        stop=UART.STOPBITS_ONE,
    )


def cleanup():
    global sensor, uart

    if isinstance(sensor, Sensor):
        sensor.stop()

    Display.deinit()
    os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
    time.sleep_ms(100)
    MediaManager.deinit()
    uart = None


def main():
    roi_w = int(PICTURE_WIDTH * ROI_SCALE_W)
    roi_h = int(PICTURE_HEIGHT * ROI_SCALE_H)
    roi_x = (PICTURE_WIDTH - roi_w) // 2
    roi_y = (PICTURE_HEIGHT - roi_h) // 2

    target_tracker = TargetTracker(roi_x, roi_y, roi_w, roi_h)
    laser_tracker = LaserTracker()
    clock = time.clock()

    while True:
        os.exitpoint()
        clock.tick()

        img = sensor.snapshot(chn=CAM_CHN_ID_0)

        if DRAW_ROI:
            img.draw_rectangle(roi_x, roi_y, roi_w, roi_h, color=(255, 255, 0), thickness=2)

        roi_img = img.copy(roi=(roi_x, roi_y, roi_w, roi_h))
        gray_img = roi_img.to_grayscale(copy=False)
        img_np = gray_img.to_numpy_ref()

        target, pass_id = target_tracker.update(img_np)
        target_visible_for_laser = target is not None and pass_id != -3
        target_bbox = target[4] if target_visible_for_laser else None
        laser, laser_roi, laser_pass_id, laser_score, laser_candidate_count, laser_core, laser_core_roi = laser_tracker.update(img, target_bbox)

        if target:
            rect, tx, ty, area, bbox = target
            draw_target_rect(img, rect, roi_x, roi_y)
            img.draw_cross(int(tx), int(ty), size=15, color=(255, 0, 0), thickness=2)
            draw_text(img, 10, 10, "target p:{}".format(pass_id), color=(255, 0, 0), scale=1)

            pre_ex = int(tx - AIM_CENTER_X)
            pre_ey = int(ty - AIM_CENTER_Y)
            aim_ready = (
                target_visible_for_laser
                and
                abs(pre_ex) <= AIM_READY_THRESHOLD_X
                and abs(pre_ey) <= AIM_READY_THRESHOLD_Y
            )
            draw_text(
                img,
                10,
                28,
                "pre ex:{} ey:{} ready:{}".format(pre_ex, pre_ey, 1 if aim_ready else 0),
                color=(255, 0, 0),
                scale=1,
            )

        if laser_roi:
            img.draw_rectangle(laser_roi[0], laser_roi[1], laser_roi[2], laser_roi[3], color=(255, 255, 0), thickness=1)
            if laser_core_roi:
                img.draw_rectangle(laser_core_roi[0], laser_core_roi[1], laser_core_roi[2], laser_core_roi[3], color=(255, 255, 255), thickness=1)

        if laser:
            lx = spot_cx(laser, laser_core)
            ly = spot_cy(laser, laser_core)
            img.draw_rectangle(blob_x(laser), blob_y(laser), blob_w(laser), blob_h(laser), color=(0, 255, 0), thickness=2)
            img.draw_cross(lx, ly, color=(0, 255, 0), size=12, thickness=2)
            if laser_core is not None:
                img.draw_rectangle(blob_x(laser_core), blob_y(laser_core), blob_w(laser_core), blob_h(laser_core), color=(255, 255, 255), thickness=2)
                img.draw_cross(blob_cx(laser_core), blob_cy(laser_core), color=(255, 255, 255), size=8, thickness=1)
            draw_text(img, 10, 46, "laser x:{} y:{} p:{}".format(lx, ly, laser_pass_id), color=(0, 255, 0), scale=1)
        elif target:
            draw_text(img, 10, 46, "laser none c:{}".format(laser_candidate_count), color=(255, 255, 0), scale=1)

        if target and laser:
            ex = int(target[1] - spot_cx(laser, laser_core))
            ey = int(target[2] - spot_cy(laser, laser_core))
            draw_text(img, 10, 64, "laser err x:{} y:{}".format(ex, ey), color=(255, 255, 255), scale=1)

        if DRAW_IMAGE_CENTER:
            img.draw_cross(
                AIM_CENTER_X,
                AIM_CENTER_Y,
                color=(0, 0, 255),
                size=12,
                thickness=1,
            )

        draw_text(img, 10, PICTURE_HEIGHT - 24, "fps:{:.1f}".format(clock.fps()), color=(255, 255, 255), scale=1)
        send_fusion_packet(target, laser, laser_core, target_visible_for_laser)

        Display.show_image(
            img,
            x=(DISPLAY_WIDTH - PICTURE_WIDTH) // 2,
            y=(DISPLAY_HEIGHT - PICTURE_HEIGHT) // 2,
        )


try:
    init_hardware()
    main()
except KeyboardInterrupt:
    print("user stop")
except BaseException as e:
    print("error:", e)
finally:
    cleanup()
