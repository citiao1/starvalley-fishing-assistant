from __future__ import annotations

import logging
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QImage

from .config import AppPaths, AppSettings
from .feed_detector import FeedZeroDetector, scaled_roi
from .input_control import WindowsInputController
from .onnx_detector import OnnxBiteDetector


class BiteDetector:
    def __init__(
        self,
        model_path: Path,
        confidence: float,
        image_size: int,
        use_cuda: bool,
    ) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.image_size = image_size
        self.use_cuda = use_cuda
        self.model: Any = None
        self.device = "cpu"

    def load(self) -> str:
        from ultralytics import YOLO

        self.model = YOLO(str(self.model_path))
        self._refresh_device()
        return f"模型已加载: {self.model_path.name} / device={self.device}"

    def _refresh_device(self) -> None:
        try:
            import torch

            self.device = "0" if self.use_cuda and torch.cuda.is_available() else "cpu"
        except Exception:
            self.device = "cpu"

    def configure(self, confidence: float, use_cuda: bool) -> bool:
        previous_device = self.device
        self.confidence = confidence
        self.use_cuda = use_cuda
        self._refresh_device()
        return self.device != previous_device

    def detect(self, frame: np.ndarray) -> tuple[list[tuple[int, int, int, int, float]], float]:
        if self.model is None:
            return [], 0.0
        results = self.model.predict(
            source=frame,
            imgsz=self.image_size,
            conf=self.confidence,
            device=self.device,
            half=self.device != "cpu",
            max_det=5,
            verbose=False,
        )
        detections: list[tuple[int, int, int, int, float]] = []
        best_confidence = 0.0
        if not results:
            return detections, best_confidence
        result = results[0]
        if result.boxes is None:
            return detections, best_confidence
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            # Class 0 is the bite icon in the project dataset. Keep the
            # numeric id authoritative so a harmless names-map variation
            # cannot silently disable auto-reel detection.
            if class_id != 0:
                continue
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            detections.append((x1, y1, x2, y2, confidence))
            best_confidence = max(best_confidence, confidence)
        return detections, best_confidence

    @staticmethod
    def resize_for_inference(
        frame: np.ndarray,
        max_width: int = 1280,
    ) -> tuple[np.ndarray, float, float]:
        height, width = frame.shape[:2]
        if width <= max_width:
            return frame, 1.0, 1.0
        target_width = max_width
        target_height = max(1, round(height * target_width / width))
        resized = cv2.resize(
            frame,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        return resized, width / target_width, height / target_height


class ScreenCapture:
    def __init__(self, monitor_index: int) -> None:
        self.monitor_index = monitor_index
        self._mss = None
        self._dxcam = None
        self.monitor: dict[str, int] | None = None
        self.backend_name = "未打开"
        self.backend_error = ""

    def open(self) -> None:
        import mss

        self._mss = mss.MSS()
        monitors = self._mss.monitors
        primary = next(
            (
                index
                for index, monitor in enumerate(monitors)
                if index > 0 and monitor.get("is_primary")
            ),
            1,
        )
        if self.monitor_index <= 0:
            self.monitor_index = primary
        elif len(monitors) <= self.monitor_index:
            self.monitor_index = 1
        self.monitor = monitors[self.monitor_index]
        self.backend_name = "mss"
        self.backend_error = ""

        try:
            import dxcam

            output_index = (
                None
                if self.monitor_index == primary
                else max(0, self.monitor_index - 1)
            )
            width = int(self.monitor["width"])
            height = int(self.monitor["height"])
            self._dxcam = dxcam.create(
                output_idx=output_index,
                region=(0, 0, width, height),
                output_color="BGR",
                processor_backend="cv2",
            )
            self._dxcam.start(
                region=(0, 0, width, height),
                target_fps=60,
                video_mode=True,
            )
            self.backend_name = f"dxcam (output {output_index})"
        except Exception as exc:
            self.backend_error = f"{type(exc).__name__}: {exc}"
            self.backend_name = "mss fallback"
            if self._dxcam is not None:
                try:
                    self._dxcam.release()
                except Exception:
                    pass
                self._dxcam = None

    def grab(self) -> np.ndarray:
        if self._mss is None or self.monitor is None:
            self.open()
        assert self._mss is not None
        assert self.monitor is not None
        if self._dxcam is not None:
            frame = self._dxcam.get_latest_frame()
            if frame is not None:
                return np.asarray(frame, dtype=np.uint8)
        return np.asarray(self._mss.grab(self.monitor), dtype=np.uint8)[:, :, :3]

    @property
    def size(self) -> tuple[int, int]:
        if self.monitor is None:
            return 0, 0
        return int(self.monitor["width"]), int(self.monitor["height"])

    def close(self) -> None:
        if self._dxcam is not None:
            try:
                self._dxcam.stop()
            except Exception:
                pass
            try:
                self._dxcam.release()
            except Exception:
                pass
            self._dxcam = None
        if self._mss is not None:
            self._mss.close()
            self._mss = None


class SessionLogger:
    def __init__(self, log_path: Path) -> None:
        self.logger = logging.getLogger("fishing_assistant")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.addHandler(handler)

    def write(self, level: str, message: str) -> None:
        getattr(self.logger, level, self.logger.info)(message)


class DetectionWorker(QObject):
    frame_ready = Signal(QImage)
    metrics_ready = Signal(object)
    status_ready = Signal(str)
    log_ready = Signal(str, str)
    finished = Signal()

    def __init__(self, paths: AppPaths, settings: AppSettings) -> None:
        super().__init__()
        self.paths = paths
        self.settings = replace(settings)
        self._running = False
        self._stop_event = threading.Event()
        self._capture: ScreenCapture | None = None
        self._bite: BiteDetector | None = None
        self._feed: FeedZeroDetector | None = None
        self._input = WindowsInputController()
        self._logger = SessionLogger(paths.log_file)
        self._bite_streak = 0
        self._zero_streak = 0
        self._feed_waiting_confirmation = False
        self._next_reel_at = 0.0
        self._last_status = "等待启动"
        self._last_action = "暂无动作"
        self._last_log_at: dict[str, float] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._inference_future: Future | None = None
        self._latest_boxes: list[tuple[int, int, int, int, float]] = []
        self._latest_feed_label = "not_found"
        self._latest_bite_confidence = 0.0
        self._latest_zero_score = 0.0
        self._latest_nonzero_score = 0.0
        self._bite_absent_streak = 0
        self._bite_event_handled = False
        self._capture_frames = 0
        self._inference_frames = 0
        self._preview_started_at = 0.0
        self._last_metrics_at = 0.0
        self._last_inference_ms = 0.0
        self._last_capture_ms = 0.0

    @Slot(object)
    def apply_settings(self, settings: AppSettings) -> None:
        self.settings = replace(settings)
        if self._bite is not None:
            device_changed = self._bite.configure(
                settings.confidence,
                settings.use_cuda,
            )
            if device_changed:
                self._log(
                    "info",
                    f"推理设备已切换为 {self._bite.device}",
                    force=True,
                )
        if not settings.master_enabled:
            self._bite_streak = 0
            self._zero_streak = 0
            self._feed_waiting_confirmation = False

    def emergency_stop_now(self) -> None:
        self._input.trigger_emergency_stop()
        self.settings = replace(
            self.settings,
            master_enabled=False,
            auto_reel_enabled=False,
            auto_feed_enabled=False,
        )
        self._bite_streak = 0
        self._zero_streak = 0
        self._feed_waiting_confirmation = False
        self._set_status("紧急停止：输入已锁定")
        self._log("warning", "用户触发紧急停止，所有自动输入已锁定", force=True)

    def test_input(self, action: str) -> tuple[bool, str]:
        if action == "z":
            ok = self._input.press_z(self.settings.input_method)
        elif action == "left_click":
            ok = self._input.left_click(self.settings.input_method)
        else:
            self._input.last_result = f"未知输入测试动作: {action}"
            ok = False
        return ok, self._input.last_result

    def run(self) -> None:
        self._running = True
        self._stop_event.clear()
        self._preview_started_at = 0.0
        self._capture_frames = 0
        self._inference_frames = 0
        self._bite_streak = 0
        self._zero_streak = 0
        self._feed_waiting_confirmation = False
        self._bite_event_handled = False
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="fishing-inference",
        )
        try:
            self._input.clear_emergency_stop()
            self._capture = ScreenCapture(self.settings.monitor_index)
            self._capture.open()
            self._log("info", f"屏幕采集已打开: {self._capture.size}", force=True)
            self._log("info", f"屏幕采集后端: {self._capture.backend_name}", force=True)
            if self._capture.backend_error:
                self._log(
                    "warning",
                    f"DXcam 不可用，已回退 mss：{self._capture.backend_error}",
                    force=True,
                )

            if self.settings.inference_backend == "onnx":
                if not self.paths.onnx_model_path.exists():
                    raise FileNotFoundError(
                        f"找不到 ONNX 权重: {self.paths.onnx_model_path}"
                    )
                torch_dll_dir = (
                    Path(sys.prefix)
                    / "Lib"
                    / "site-packages"
                    / "torch"
                    / "lib"
                )
                dll_search_paths = (
                    (torch_dll_dir,) if torch_dll_dir.exists() else ()
                )
                self._bite = OnnxBiteDetector(
                    self.paths.onnx_model_path,
                    self.settings.confidence,
                    self.settings.yolo_imgsz,
                    self.settings.use_cuda,
                    dll_search_paths=dll_search_paths,
                    execution_provider=self.settings.onnx_provider,
                )
            else:
                if not self.paths.model_path.exists():
                    raise FileNotFoundError(
                        f"找不到 YOLO 权重: {self.paths.model_path}"
                    )
                self._bite = BiteDetector(
                    self.paths.model_path,
                    self.settings.confidence,
                    self.settings.yolo_imgsz,
                    self.settings.use_cuda,
                )
            self._log("info", self._bite.load(), force=True)

            if not self.paths.zero_template_path.exists():
                raise FileNotFoundError(f"找不到红 0 模板: {self.paths.zero_template_path}")
            if not self.paths.nonzero_template_path.exists():
                raise FileNotFoundError(f"找不到红 2 模板: {self.paths.nonzero_template_path}")
            self._feed = FeedZeroDetector.from_images(
                self.paths.zero_template_path,
                self.paths.nonzero_template_path,
            )
            self._log("info", "OpenCV 红色数字检测器已加载", force=True)
            # Start the FPS window after model/template initialization so the
            # startup delay does not permanently depress the displayed rate.
            self._preview_started_at = time.perf_counter()
            self._capture_frames = 0
            self._inference_frames = 0
            self._set_status("预览运行中")

            preview_interval = 1.0 / max(
                5.0,
                min(60.0, self.settings.preview_fps),
            )
            inference_interval = 1.0 / max(1.0, self.settings.inference_fps)
            next_inference_at = 0.0
            while not self._stop_event.is_set():
                started = time.perf_counter()
                capture_started = time.perf_counter()
                frame = self._capture.grab()
                self._last_capture_ms = (time.perf_counter() - capture_started) * 1000
                self._capture_frames += 1

                target_description = self._input.observe_foreground_window()
                if target_description:
                    self._log("info", f"已绑定目标窗口：{target_description}")

                now = time.monotonic()
                if (
                    self._inference_future is not None
                    and self._inference_future.done()
                ):
                    try:
                        result = self._inference_future.result()
                        self._last_inference_ms = result[-1]
                        self._inference_frames += 1
                        self._apply_detection_result(result[:-1], now)
                    except Exception as exc:
                        self._log(
                            "error",
                            f"推理线程错误: {type(exc).__name__}: {exc}",
                            force=True,
                        )
                    finally:
                        self._inference_future = None

                if (
                    self._inference_future is None
                    and now >= next_inference_at
                ):
                    settings_snapshot = replace(self.settings)
                    inference_frame, scale_x, scale_y = (
                        self._bite.resize_for_inference(frame)
                        if self._bite is not None
                        else (frame, 1.0, 1.0)
                    )
                    feed_x1, feed_y1, feed_x2, feed_y2 = (
                        self._feed_roi_bounds(frame)
                    )
                    self._inference_future = self._executor.submit(
                        self._detect_frame,
                        inference_frame,
                        frame[feed_y1:feed_y2, feed_x1:feed_x2].copy(),
                        scale_x,
                        scale_y,
                        settings_snapshot,
                    )
                    next_inference_at = now + inference_interval

                annotated = self._annotate(
                    frame,
                    self._latest_boxes,
                    self._latest_feed_label,
                )
                self.frame_ready.emit(self._to_qimage(annotated))
                self._emit_metrics(frame)
                elapsed = time.perf_counter() - started
                time.sleep(max(0.0, preview_interval - elapsed))
        except Exception as exc:
            self._set_status("运行错误")
            self._log("error", f"{type(exc).__name__}: {exc}", force=True)
        finally:
            if self._capture is not None:
                self._capture.close()
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._executor = None
            self._inference_future = None
            self._running = False
            self._log("info", "检测线程已停止", force=True)
            self.finished.emit()

    def request_stop(self) -> None:
        self._stop_event.set()

    def _detect_frame(
        self,
        inference_frame: np.ndarray,
        feed_roi: np.ndarray,
        box_scale_x: float,
        box_scale_y: float,
        settings: AppSettings,
    ) -> tuple[
        list[tuple[int, int, int, int, float]],
        float,
        str,
        float,
        float,
        float,
    ]:
        started = time.perf_counter()
        bite_boxes: list[tuple[int, int, int, int, float]] = []
        bite_confidence = 0.0
        feed_label, zero_score, nonzero_score = "not_found", 0.0, 0.0

        if settings.bite_detection_enabled and self._bite is not None:
            detected_boxes, bite_confidence = self._bite.detect(inference_frame)
            bite_boxes = [
                (
                    round(x1 * box_scale_x),
                    round(y1 * box_scale_y),
                    round(x2 * box_scale_x),
                    round(y2 * box_scale_y),
                    confidence,
                )
                for x1, y1, x2, y2, confidence in detected_boxes
            ]
        if self._feed is not None:
            feed_label, zero_score, nonzero_score = self._feed.classify_roi(feed_roi)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return (
            bite_boxes,
            bite_confidence,
            feed_label,
            zero_score,
            nonzero_score,
            elapsed_ms,
        )

    def _apply_detection_result(
        self,
        result: tuple[
            list[tuple[int, int, int, int, float]],
            float,
            str,
            float,
            float,
        ],
        now: float,
    ) -> None:
        (
            bite_boxes,
            bite_confidence,
            feed_label,
            zero_score,
            nonzero_score,
        ) = result
        self._latest_boxes = bite_boxes
        self._latest_feed_label = feed_label
        self._latest_bite_confidence = bite_confidence
        self._latest_zero_score = zero_score
        self._latest_nonzero_score = nonzero_score

        if bite_boxes:
            self._bite_streak += 1
            self._bite_absent_streak = 0
        else:
            self._bite_streak = 0
            self._bite_absent_streak += 1
            if self._bite_absent_streak >= 3:
                self._bite_event_handled = False

        if feed_label == "zero":
            self._zero_streak += 1
        else:
            self._zero_streak = 0
            if self._feed_waiting_confirmation:
                self._feed_waiting_confirmation = False
                self._log(
                    "info",
                    f"检测到红 0 已消失，体力恢复已确认，"
                    f"当前={feed_label} zero={zero_score:.3f} "
                    f"nonzero={nonzero_score:.3f}",
                    force=True,
                )

        action_status = "等待"
        if bite_boxes:
            action_status = "检测到上钩"
        elif feed_label == "zero":
            action_status = "检测到红 0"
        if self.settings.master_enabled and not self.settings.preview_only:
            if (
                self.settings.auto_reel_enabled
                and self._bite_streak >= self.settings.bite_confirm_frames
                and bite_confidence >= self.settings.bite_action_confidence
                and not self._bite_event_handled
                and now >= self._next_reel_at
            ):
                if self._input.left_click(self.settings.input_method):
                    self._next_reel_at = now + self.settings.reel_cooldown_seconds
                    self._last_action = f"{time.strftime('%H:%M:%S')} 左键收杆"
                    action_status = "已收杆，进入冷却"
                    self._bite_event_handled = True
                    self._log(
                        "info",
                        f"执行左键收杆，置信度={bite_confidence:.3f}，"
                        f"输入={self._input.last_result}",
                        force=True,
                    )
                    self._bite_streak = 0
                else:
                    self._next_reel_at = now + 0.5
                    self._log(
                        "error",
                        f"左键收杆发送失败：{self._input.last_result}",
                        force=True,
                    )
            if (
                self.settings.auto_feed_enabled
                and self._zero_streak >= self.settings.feed_confirm_frames
            ):
                # 喂食只插入 Z，不读取、释放或恢复用户正在按住的移动键。
                # 这样不会为了抢占 Z 改变 W/A/S/D 等按键状态。
                sent = self._input.press_z(self.settings.input_method)
                self._feed_waiting_confirmation = True
                self._last_action = (
                    f"{time.strftime('%H:%M:%S')} Z 已发送，等待体力恢复"
                )
                action_status = "已发送 Z，等待体力恢复"
                # SendInput 成功只代表系统接收了按键，不代表游戏执行了
                # 喂食。清掉确认计数，继续观察红 0；仍是红 0 时会再次尝试。
                self._zero_streak = 0
                if sent:
                    self._log(
                        "info",
                        f"发送 Z，等待体力恢复确认，zero={zero_score:.3f}, "
                        f"nonzero={nonzero_score:.3f}，"
                        f"输入={self._input.last_result}",
                        force=True,
                    )
                else:
                    self._log(
                        "error",
                        f"Z 喂食发送失败：{self._input.last_result}",
                        force=False,
                    )
        elif self.settings.preview_only:
            action_status = f"预览模式：{action_status}"
        elif not self.settings.master_enabled:
            action_status = "总开关关闭"

        self._set_status(action_status)

    @staticmethod
    def _feed_roi_bounds(frame: np.ndarray) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = scaled_roi(frame.shape[1], frame.shape[0])
        return (
            max(0, x1),
            max(0, y1),
            min(frame.shape[1], x2),
            min(frame.shape[0], y2),
        )

    def _emit_metrics(self, frame: np.ndarray) -> None:
        now = time.perf_counter()
        if now - self._last_metrics_at < 0.2:
            return
        self._last_metrics_at = now
        elapsed = max(0.001, now - self._preview_started_at)
        self.metrics_ready.emit(
            {
                "resolution": f"{frame.shape[1]} × {frame.shape[0]}",
                "capture_backend": self._capture.backend_name if self._capture else "未知",
                "preview_fps": self._capture_frames / elapsed,
                "inference_fps": self._inference_frames / elapsed,
                "capture_ms": self._last_capture_ms,
                "inference_ms": self._last_inference_ms,
                "bite_confidence": self._latest_bite_confidence,
                "bite_streak": self._bite_streak,
                "feed_label": self._latest_feed_label,
                "zero_score": self._latest_zero_score,
                "nonzero_score": self._latest_nonzero_score,
                "status": self._last_status,
                "last_action": self._last_action_text(),
                "preview_only": self.settings.preview_only,
                "target_window": self._input.target_description,
                "target_privilege": self._input.target_privilege,
                "input_result": self._input.last_result,
            }
        )

    def _annotate(
        self,
        frame: np.ndarray,
        boxes: list[tuple[int, int, int, int, float]],
        feed_label: str,
    ) -> np.ndarray:
        max_width = 960
        if frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            annotated = cv2.resize(
                frame,
                (max_width, max(1, round(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            annotated = frame.copy()

        scale_x = annotated.shape[1] / frame.shape[1]
        scale_y = annotated.shape[0] / frame.shape[0]
        for x1, y1, x2, y2, confidence in boxes:
            preview_box = (
                round(x1 * scale_x),
                round(y1 * scale_y),
                round(x2 * scale_x),
                round(y2 * scale_y),
            )
            cv2.rectangle(
                annotated,
                preview_box[:2],
                preview_box[2:],
                (55, 220, 130),
                2,
            )
            cv2.putText(
                annotated,
                f"bite_icon {confidence:.2f}",
                (preview_box[0], max(24, preview_box[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (55, 220, 130),
                1,
                cv2.LINE_AA,
            )
        x1, y1, x2, y2 = scaled_roi(annotated.shape[1], annotated.shape[0])
        roi_color = (70, 190, 255) if feed_label == "zero" else (160, 160, 160)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), roi_color, 2)
        cv2.putText(
            annotated,
            f"stamina: {feed_label}",
            (max(10, x1 - 155), max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            roi_color,
            1,
            cv2.LINE_AA,
        )
        return annotated

    @staticmethod
    def _to_qimage(frame: np.ndarray) -> QImage:
        max_width = 960
        if frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            frame = cv2.resize(
                frame,
                (max_width, max(1, round(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()

    def _set_status(self, status: str) -> None:
        if status != self._last_status:
            self._last_status = status
            self.status_ready.emit(status)

    def _log(self, level: str, message: str, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_log_at.get(message, 0.0) < 2.0:
            return
        self._last_log_at[message] = now
        self._logger.write(level, message)
        self.log_ready.emit(level, message)

    def _last_action_text(self) -> str:
        cooldown = ""
        if self._next_reel_at > time.monotonic():
            cooldown = " / 收杆冷却中"
        return self._last_action + cooldown


class WorkerController(QObject):
    settings_changed = Signal(object)
    stop_requested = Signal()
    emergency_requested = Signal()

    def __init__(self, paths: AppPaths, settings: AppSettings) -> None:
        super().__init__()
        self.worker = DetectionWorker(paths, settings)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

    def start(self) -> None:
        if not self.thread.isRunning():
            self.thread.start()

    def stop(self) -> None:
        self.worker.request_stop()

    def emergency_stop(self) -> None:
        self.worker.emergency_stop_now()

    def test_input(self, action: str) -> tuple[bool, str]:
        return self.worker.test_input(action)

    def update_settings(self, settings: AppSettings) -> None:
        self.worker.apply_settings(replace(settings))
