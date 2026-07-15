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

# Target rectangle detection, based on 05.
ROI_SCALE_W = 1.0
ROI_SCALE_H = 1.0

AIM_CENTER_X = 206
AIM_CENTER_Y = 122
# When the target center is inside this radius around the calibrated center,
# K230 starts accepting laser detection and tells M0 it can turn the laser on.
AIM_READY_RADIUS = 35

MIN_TARGET_AREA = 2000
MAX_TARGET_AREA_RATIO = 0.85
MAX_TARGET_ASPECT_RATIO = 5.0
TARGET_CENTER_WEIGHT = 0.25
TARGET_PREV_WEIGHT = 0.70
TARGET_MISS_CONFIRM_FRAMES = 3

DETECT_PASSES = (
    (80, 150, 0.04, 0.30, 5),
    (60, 140, 0.05, 0.45, 5),
    (40, 120, 0.06, 0.60, 3),
)

# Bright laser detection, based on 08.
DYNAMIC_THRESHOLD = True
FIXED_THRESHOLD = 220
MIN_DYNAMIC_THRESHOLD = 165
THRESH_DELTA_FROM_MAX = 8
CENTROID_DELTA_FROM_MAX = 6

# Laser is searched only after the target is current-frame valid and centered.
LASER_DETECT_AFTER_READY_ONLY = True
LASER_TARGET_PAD = 8

LASER_MIN_PIXELS = 1
LASER_MIN_AREA = 1
LASER_MAX_PIXELS = 420
LASER_IDEAL_PIXELS = 45
LASER_MIN_CORE_PIXELS = 8
LASER_MAX_CORE_PIXELS = 260
LASER_MAX_CORE_ASPECT_RATIO = 2.0
LASER_MAX_RECT_SIDE = 34
LASER_MAX_ASPECT_RATIO = 2.5
LASER_MERGE_MARGIN = 2
LASER_MISS_HOLD_FRAMES = 2

LASER_SIZE_WEIGHT = 260.0
LASER_ROUND_WEIGHT = 120.0
LASER_BRIGHT_WEIGHT = 220.0
LASER_DENSITY_WEIGHT = 70.0
LASER_CORE_SIZE_WEIGHT = 220.0
LASER_TEMPORAL_SIZE_WEIGHT = 180.0

POSITION_SMOOTH_ALPHA = 0.75

DRAW_ROI = False
DRAW_IMAGE_CENTER = True
DRAW_LASER_ROI = True
PRINT_EVERY_FRAME = True
PRINT_FULL_PACKET = False
SEND_UART = True

# Keep these off first. GC2093 may report them unsupported on some firmware.
LOCK_CAMERA_EXPOSURE = False
EXPOSURE_US = 6000
GAIN_DB = 6


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


def detect_rects_multi_pass(img_np, roi_w, roi_h):
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


def pick_best_rect(rects, roi_x, roi_y, last_cx, last_cy):
    if not rects:
        return None

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
        self.last_rect = None
        self.last_cx = None
        self.last_cy = None
        self.last_area = 0.0
        self.last_bbox = None
        self.miss_count = TARGET_MISS_CONFIRM_FRAMES

    def update(self, img_np):
        rects, pass_id = detect_rects_multi_pass(img_np, self.roi_w, self.roi_h)
        best = pick_best_rect(rects, self.roi_x, self.roi_y, self.last_cx, self.last_cy)

        if best:
            cx, cy = rect_center_in_full_image(best, self.roi_x, self.roi_y)
            self.last_rect = best
            self.last_cx = cx
            self.last_cy = cy
            self.last_area = rect_area(best)
            self.last_bbox = rect_bbox_in_full_image(best, self.roi_x, self.roi_y)
            self.miss_count = 0
            return self.current(), pass_id, True, False

        self.miss_count = min(self.miss_count + 1, TARGET_MISS_CONFIRM_FRAMES)
        if self.last_rect is not None and self.miss_count < TARGET_MISS_CONFIRM_FRAMES:
            return self.current(), -3, False, True

        self.clear()
        return None, -1, False, False

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


class BrightCore:
    def __init__(self, x, y, w, h, pixels, cx, cy, max_luma):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.pixels = int(pixels)
        self.cx = int(cx)
        self.cy = int(cy)
        self.max_luma = int(max_luma)


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


def max_luma_in_roi(gray, roi):
    try:
        stats = gray.get_statistics(roi=roi)
        stats_max = getattr(stats, "max", None)
        if callable(stats_max):
            return int(stats_max())
        if stats_max is not None:
            return int(stats_max)
    except BaseException:
        pass

    rx, ry, rw, rh = roi
    max_luma = -1
    for y in range(ry, ry + rh):
        for x in range(rx, rx + rw):
            luma = pixel_luma(gray.get_pixel(x, y))
            if luma > max_luma:
                max_luma = luma
    return max_luma


def dynamic_threshold(gray, roi):
    if not DYNAMIC_THRESHOLD:
        return FIXED_THRESHOLD, max_luma_in_roi(gray, roi)

    max_luma = max_luma_in_roi(gray, roi)
    if max_luma < MIN_DYNAMIC_THRESHOLD:
        return 256, max_luma

    threshold = max(MIN_DYNAMIC_THRESHOLD, max_luma - THRESH_DELTA_FROM_MAX)
    threshold = min(threshold, 254)
    return int(threshold), max_luma


def laser_blob_is_valid(blob):
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


def blob_density(blob):
    area = blob_w(blob) * blob_h(blob)
    if area <= 0:
        return 0.0
    return clamp(float(blob_pixels(blob)) / float(area), 0.0, 1.0)


def refine_bright_core(gray, blob, base_threshold):
    rx = blob_x(blob)
    ry = blob_y(blob)
    rw = blob_w(blob)
    rh = blob_h(blob)

    max_luma = max_luma_in_roi(gray, (rx, ry, rw, rh))
    if max_luma < MIN_DYNAMIC_THRESHOLD:
        return None

    cutoff = max(base_threshold, max_luma - CENTROID_DELTA_FROM_MAX)

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
            luma = pixel_luma(gray.get_pixel(x, y))
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
        return None
    if count < LASER_MIN_CORE_PIXELS or count > LASER_MAX_CORE_PIXELS:
        return None

    core_w = max(1, max_x - min_x + 1)
    core_h = max(1, max_y - min_y + 1)
    if max(core_w, core_h) / min(core_w, core_h) > LASER_MAX_CORE_ASPECT_RATIO:
        return None

    return BrightCore(
        min_x,
        min_y,
        core_w,
        core_h,
        count,
        int(sum_x / sum_w),
        int(sum_y / sum_w),
        max_luma,
    )


def ratio_similarity(a, b):
    if a is None or b is None or a <= 0 or b <= 0:
        return 0.0
    return float(min(a, b)) / float(max(a, b))


def score_laser_blob(blob, core, last_pixels, last_area):
    pixels = blob_pixels(blob)
    if pixels <= LASER_IDEAL_PIXELS:
        size_score = clamp(float(pixels) / float(LASER_IDEAL_PIXELS), 0.0, 1.0)
    else:
        size_score = 1.0 - clamp(float(pixels - LASER_IDEAL_PIXELS) / float(LASER_MAX_PIXELS - LASER_IDEAL_PIXELS), 0.0, 1.0)

    score = LASER_SIZE_WEIGHT * size_score

    w = blob_w(blob)
    h = blob_h(blob)
    aspect = max(w, h) / min(w, h)
    round_score = 1.0 - clamp((aspect - 1.0) / (LASER_MAX_ASPECT_RATIO - 1.0), 0.0, 1.0)
    score += LASER_ROUND_WEIGHT * round_score
    score += LASER_DENSITY_WEIGHT * blob_density(blob)

    if core is not None:
        bright_score = clamp(float(core.max_luma - MIN_DYNAMIC_THRESHOLD) / float(255 - MIN_DYNAMIC_THRESHOLD), 0.0, 1.0)
        score += LASER_BRIGHT_WEIGHT * bright_score
        if core.pixels <= LASER_IDEAL_PIXELS:
            core_size_score = clamp(float(core.pixels) / float(LASER_IDEAL_PIXELS), 0.0, 1.0)
        else:
            core_size_score = 1.0 - clamp(float(core.pixels - LASER_IDEAL_PIXELS) / float(LASER_MAX_CORE_PIXELS - LASER_IDEAL_PIXELS), 0.0, 1.0)
        score += LASER_CORE_SIZE_WEIGHT * core_size_score

    spot_pixels = core.pixels if core is not None else pixels
    spot_area = (core.w * core.h) if core is not None else (w * h)
    if last_pixels is not None and last_area is not None:
        temporal_size_score = 0.6 * ratio_similarity(spot_pixels, last_pixels)
        temporal_size_score += 0.4 * ratio_similarity(spot_area, last_area)
        score += LASER_TEMPORAL_SIZE_WEIGHT * temporal_size_score

    return score


class BrightLaserTracker:
    def __init__(self):
        self.last_x = None
        self.last_y = None
        self.last_pixels = None
        self.last_area = None
        self.lost_frames = LASER_MISS_HOLD_FRAMES
        self.output_valid = False
        self.output_x = 0
        self.output_y = 0
        self.output_pixels = 0
        self.held = False

    def reset(self):
        self.last_x = None
        self.last_y = None
        self.last_pixels = None
        self.last_area = None
        self.lost_frames = LASER_MISS_HOLD_FRAMES
        self.output_valid = False
        self.output_x = 0
        self.output_y = 0
        self.output_pixels = 0
        self.held = False

    def find_best_in_roi(self, gray, roi):
        threshold, max_luma = dynamic_threshold(gray, roi)
        if threshold > 255:
            return None, None, threshold, max_luma, 0

        blobs = gray.find_blobs(
            [(threshold, 255)],
            roi=roi,
            pixels_threshold=LASER_MIN_PIXELS,
            area_threshold=LASER_MIN_AREA,
            merge=True,
            margin=LASER_MERGE_MARGIN,
        )

        best_blob = None
        best_core = None
        best_score = -1.0
        candidate_count = 0

        for blob in blobs:
            if not laser_blob_is_valid(blob):
                continue

            core = refine_bright_core(gray, blob, threshold)
            candidate_count += 1
            score = score_laser_blob(blob, core, self.last_pixels, self.last_area)
            if score > best_score:
                best_score = score
                best_blob = blob
                best_core = core

        return best_blob, best_core, threshold, max_luma, candidate_count

    def update(self, gray, roi):
        if roi is None:
            self.reset()
            return None, None, None, 256, 0, 0

        blob, core, threshold, max_luma, candidate_count = self.find_best_in_roi(gray, roi)

        if blob is None:
            self.lost_frames += 1
            self.held = self.output_valid and self.lost_frames <= LASER_MISS_HOLD_FRAMES
            if not self.held:
                self.output_valid = False
            return None, None, roi, threshold, max_luma, candidate_count

        if core is not None:
            raw_x = core.cx
            raw_y = core.cy
            pixels = core.pixels
            area = core.w * core.h
        else:
            raw_x = blob_cx(blob)
            raw_y = blob_cy(blob)
            pixels = blob_pixels(blob)
            area = blob_w(blob) * blob_h(blob)

        if self.last_x is None or self.last_y is None:
            filtered_x = raw_x
            filtered_y = raw_y
        else:
            filtered_x = int(POSITION_SMOOTH_ALPHA * raw_x + (1.0 - POSITION_SMOOTH_ALPHA) * self.last_x)
            filtered_y = int(POSITION_SMOOTH_ALPHA * raw_y + (1.0 - POSITION_SMOOTH_ALPHA) * self.last_y)

        self.last_x = filtered_x
        self.last_y = filtered_y
        self.last_pixels = pixels
        self.last_area = area
        self.lost_frames = 0
        self.output_valid = True
        self.output_x = filtered_x
        self.output_y = filtered_y
        self.output_pixels = pixels
        self.held = False
        return blob, core, roi, threshold, max_luma, candidate_count


def target_ready(target, target_current):
    if not target_current or target is None:
        return False
    tx = int(target[1])
    ty = int(target[2])
    dx = tx - AIM_CENTER_X
    dy = ty - AIM_CENTER_Y
    return (dx * dx + dy * dy) <= (AIM_READY_RADIUS * AIM_READY_RADIUS)


def send_fusion_packet(target, target_current, laser_tracker, ready):
    if target:
        tv = 1
        tx = int(target[1])
        ty = int(target[2])
    else:
        tv = 0
        tx = 0
        ty = 0

    if laser_tracker.output_valid:
        lv = 1
        lx = int(laser_tracker.output_x)
        ly = int(laser_tracker.output_y)
    else:
        lv = 0
        lx = 0
        ly = 0

    if target and laser_tracker.output_valid:
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

    ready_flag = 1 if ready else 0
    packet = "@9,{},{},{},{},{},{},{},{},{},{}\r\n".format(
        stage,
        tv,
        tx,
        ty,
        lv,
        lx,
        ly,
        ex,
        ey,
        ready_flag,
    )

    if PRINT_EVERY_FRAME:
        if PRINT_FULL_PACKET:
            print(packet, end="")
        else:
            print("dbg s:{} e:{},{} r:{}".format(stage, ex, ey, ready_flag))

    if SEND_UART and uart:
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

    Display.init(Display.ST7701, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, to_ide=False)
    MediaManager.init()
    sensor.run()

    if SEND_UART:
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
    laser_tracker = BrightLaserTracker()
    clock = time.clock()

    while True:
        os.exitpoint()
        clock.tick()

        img = sensor.snapshot(chn=CAM_CHN_ID_0)
        gray = img.to_grayscale(copy=True)

        if DRAW_ROI:
            img.draw_rectangle(roi_x, roi_y, roi_w, roi_h, color=(255, 255, 0), thickness=2)

        roi_gray = gray.copy(roi=(roi_x, roi_y, roi_w, roi_h))
        img_np = roi_gray.to_numpy_ref()

        target, pass_id, target_current, target_held = target_tracker.update(img_np)
        ready = target_ready(target, target_current)

        target_bbox = target[4] if target and target_current else None
        if target_bbox and ((not LASER_DETECT_AFTER_READY_ONLY) or ready):
            laser_roi = expand_roi(target_bbox, LASER_TARGET_PAD)
        else:
            laser_roi = None

        laser_blob, laser_core, laser_roi, laser_threshold, laser_max_luma, laser_candidates = laser_tracker.update(gray, laser_roi)

        if target:
            rect, tx, ty, area, bbox = target
            draw_target_rect(img, rect, roi_x, roi_y)
            img.draw_cross(int(tx), int(ty), size=15, color=(255, 0, 0), thickness=2)
            pre_ex = int(tx - AIM_CENTER_X)
            pre_ey = int(ty - AIM_CENTER_Y)
            target_state = "target" if target_current else "hold"
            draw_text(
                img,
                10,
                10,
                "{} p:{} e:{},{} r:{}".format(target_state, pass_id, pre_ex, pre_ey, 1 if ready else 0),
                color=(255, 0, 0) if target_current else (255, 255, 0),
                scale=1,
            )
        else:
            draw_text(img, 10, 10, "no target", color=(255, 0, 0), scale=1)

        if DRAW_IMAGE_CENTER:
            img.draw_cross(AIM_CENTER_X, AIM_CENTER_Y, color=(0, 0, 255), size=12, thickness=1)

        if laser_roi:
            img.draw_rectangle(laser_roi[0], laser_roi[1], laser_roi[2], laser_roi[3], color=(255, 255, 0), thickness=1)

        if laser_blob is not None:
            img.draw_rectangle(blob_x(laser_blob), blob_y(laser_blob), blob_w(laser_blob), blob_h(laser_blob), color=(0, 255, 0), thickness=2)
            if laser_core is not None:
                img.draw_rectangle(laser_core.x, laser_core.y, laser_core.w, laser_core.h, color=(255, 255, 255), thickness=1)
            img.draw_cross(laser_tracker.output_x, laser_tracker.output_y, color=(255, 255, 255), size=12, thickness=2)
            draw_text(
                img,
                10,
                30,
                "laser x:{} y:{} p:{}".format(laser_tracker.output_x, laser_tracker.output_y, laser_tracker.output_pixels),
                color=(0, 255, 0),
                scale=1,
            )
        elif laser_tracker.output_valid and laser_tracker.held:
            img.draw_cross(laser_tracker.output_x, laser_tracker.output_y, color=(255, 255, 0), size=12, thickness=2)
            draw_text(
                img,
                10,
                30,
                "laser hold x:{} y:{}".format(laser_tracker.output_x, laser_tracker.output_y),
                color=(255, 255, 0),
                scale=1,
            )
        elif target and ready:
            draw_text(
                img,
                10,
                30,
                "laser none thr:{} max:{} c:{}".format(int(laser_threshold), int(laser_max_luma), laser_candidates),
                color=(255, 255, 0),
                scale=1,
            )

        if target and laser_tracker.output_valid:
            ex = int(target[1] - laser_tracker.output_x)
            ey = int(target[2] - laser_tracker.output_y)
            draw_text(img, 10, 50, "laser err:{},{}".format(ex, ey), color=(255, 255, 255), scale=1)

        draw_text(img, 10, PICTURE_HEIGHT - 24, "fps:{:.1f}".format(clock.fps()), color=(255, 255, 255), scale=1)
        send_fusion_packet(target, target_current, laser_tracker, ready)

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
