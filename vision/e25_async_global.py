from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
from queue import Empty, Full
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .a4_target import A4Detection, find_black_band_candidates, merge_candidates
from .e25_pipeline import (
    E25GlobalDetectionContext,
    E25PipelineConfig,
    E25VisionPipeline,
)
from .rect_detect import DetectedRect, detect_rectangles, detect_rectangles_multi_pass


@dataclass(frozen=True)
class E25GlobalDetectionResult:
    frame_index: int
    detections: Tuple[A4Detection, ...]
    require_red_rings: bool
    elapsed_ms: float
    error: str = ""


@dataclass(frozen=True)
class _WorkerJob:
    frame_index: int
    context: E25GlobalDetectionContext


class AsyncE25GlobalDetector:
    def __init__(
        self,
        tracker: E25VisionPipeline,
        frame_shape: Tuple[int, int, int],
        detect_scale: float,
        multi_pass: bool,
        max_area_ratio: float,
        min_area_ratio: float,
        worker_threads: int = 2,
    ) -> None:
        if len(frame_shape) != 3 or frame_shape[2] != 3:
            raise ValueError("E25 async global process requires a BGR frame")
        self.tracker = tracker
        self.frame_shape = tuple(int(value) for value in frame_shape)
        self.gray_shape = self.frame_shape[:2]
        self._ctx = mp.get_context("spawn")
        self._lock = self._ctx.Lock()
        self._jobs = self._ctx.Queue(maxsize=1)
        self._results = self._ctx.Queue(maxsize=1)
        self._frame_shm = shared_memory.SharedMemory(
            create=True,
            size=int(np.prod(self.frame_shape)),
        )
        self._gray_shm = shared_memory.SharedMemory(
            create=True,
            size=int(np.prod(self.gray_shape)),
        )
        self._frame_view = np.ndarray(
            self.frame_shape,
            dtype=np.uint8,
            buffer=self._frame_shm.buf,
        )
        self._gray_view = np.ndarray(
            self.gray_shape,
            dtype=np.uint8,
            buffer=self._gray_shm.buf,
        )
        self._process = self._ctx.Process(
            target=_worker_main,
            args=(
                self._frame_shm.name,
                self.frame_shape,
                self._gray_shm.name,
                self.gray_shape,
                self._lock,
                self._jobs,
                self._results,
                tracker.config,
                float(detect_scale),
                bool(multi_pass),
                float(max_area_ratio),
                float(min_area_ratio),
                max(1, int(worker_threads)),
            ),
            daemon=True,
        )
        self._process.start()

    def submit(
        self,
        frame: np.ndarray,
        gray: np.ndarray,
        frame_index: int,
    ) -> None:
        if frame.shape != self.frame_shape or gray.shape != self.gray_shape:
            return
        with self._lock:
            np.copyto(self._frame_view, frame)
            np.copyto(self._gray_view, gray)
        _put_latest(
            self._jobs,
            _WorkerJob(
                frame_index=int(frame_index),
                context=self.tracker.global_detection_context(),
            ),
        )

    def latest(self, after_frame_index: int) -> Optional[E25GlobalDetectionResult]:
        newest = None
        while True:
            try:
                result = self._results.get_nowait()
            except Empty:
                break
            if newest is None or result.frame_index > newest.frame_index:
                newest = result
        if newest is None or newest.frame_index <= after_frame_index:
            return None
        return newest

    def close(self) -> None:
        try:
            _put_latest(self._jobs, None)
            self._process.join(timeout=1.5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=0.5)
        finally:
            self._jobs.close()
            self._results.close()
            self._frame_shm.close()
            self._gray_shm.close()
            self._frame_shm.unlink()
            self._gray_shm.unlink()


def _worker_main(
    frame_shm_name: str,
    frame_shape: Tuple[int, int, int],
    gray_shm_name: str,
    gray_shape: Tuple[int, int],
    lock,
    jobs,
    results,
    pipeline_config: E25PipelineConfig,
    detect_scale: float,
    multi_pass: bool,
    max_area_ratio: float,
    min_area_ratio: float,
    worker_threads: int,
) -> None:
    cv2.setNumThreads(worker_threads)
    frame_shm = shared_memory.SharedMemory(name=frame_shm_name)
    gray_shm = shared_memory.SharedMemory(name=gray_shm_name)
    frame_view = np.ndarray(frame_shape, dtype=np.uint8, buffer=frame_shm.buf)
    gray_view = np.ndarray(gray_shape, dtype=np.uint8, buffer=gray_shm.buf)
    detector = E25VisionPipeline(pipeline_config, require_red_rings=False)
    try:
        while True:
            job = jobs.get()
            if job is None:
                return
            started = time.perf_counter()
            try:
                with lock:
                    frame = frame_view.copy()
                    gray = gray_view.copy()
                base_rects = _detect_with_scale(
                    gray,
                    detect_scale,
                    multi_pass,
                    max_area_ratio,
                    min_area_ratio,
                )
                black_rects = find_black_band_candidates(
                    gray,
                    job.context.candidate_config,
                )
                rects = merge_candidates(base_rects, black_rects, limit=12)
                detections = detector.detect_global(frame, rects, job.context)
                result = E25GlobalDetectionResult(
                    frame_index=job.frame_index,
                    detections=tuple(detections),
                    require_red_rings=job.context.require_red_rings,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )
            except Exception as exc:
                result = E25GlobalDetectionResult(
                    frame_index=job.frame_index,
                    detections=(),
                    require_red_rings=job.context.require_red_rings,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    error=str(exc),
                )
            _put_latest(results, result)
    finally:
        frame_shm.close()
        gray_shm.close()


def _detect_with_scale(
    frame: np.ndarray,
    detect_scale: float,
    multi_pass: bool,
    max_area_ratio: float,
    min_area_ratio: float,
) -> list[DetectedRect]:
    detect_func = detect_rectangles_multi_pass if multi_pass else detect_rectangles
    if detect_scale == 1.0:
        if multi_pass:
            return detect_func(
                frame,
                min_area_ratio=min_area_ratio,
                max_area_ratio=max_area_ratio,
            )
        return detect_func(
            frame,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            angle_tol=25.0,
        )

    small = cv2.resize(
        frame,
        (0, 0),
        fx=detect_scale,
        fy=detect_scale,
        interpolation=cv2.INTER_AREA,
    )
    if multi_pass:
        rects = detect_func(
            small,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
        )
    else:
        rects = detect_func(
            small,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            angle_tol=25.0,
        )
    inverse = 1.0 / detect_scale
    return [
        DetectedRect(
            center=(rect.center[0] * inverse, rect.center[1] * inverse),
            box=rect.box * inverse,
            area=rect.area * inverse * inverse,
            pass_index=rect.pass_index,
            score=rect.score,
        )
        for rect in rects
    ]


def _put_latest(queue, item) -> None:
    try:
        queue.put_nowait(item)
        return
    except Full:
        pass
    try:
        queue.get_nowait()
    except Empty:
        pass
    try:
        queue.put_nowait(item)
    except Full:
        pass
