from __future__ import annotations

import os
import hashlib
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class OnnxBiteDetector:
    """ONNX Runtime detector matching the current YOLO preprocessing contract."""

    def __init__(
        self,
        model_path: Path,
        confidence: float,
        image_size: int,
        use_cuda: bool,
        dll_search_paths: tuple[Path, ...] = (),
        execution_provider: str = "cuda",
    ) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.image_size = image_size
        self.use_cuda = use_cuda
        self.dll_search_paths = dll_search_paths
        self.execution_provider = execution_provider
        self.session: Any = None
        self.input_name = ""
        self.device = "cpu"
        self._dll_handles: list[Any] = []
        self.last_timing: dict[str, float] = {}
        self.input_shape: tuple[Any, ...] = ()
        self.output_shapes: list[tuple[Any, ...]] = []
        self.model_sha256 = ""

    def load(self) -> str:
        if not self.model_path.exists():
            raise FileNotFoundError(f"找不到 ONNX 权重: {self.model_path}")
        self._open_session()
        providers = ", ".join(self.session.get_providers())
        self.model_sha256 = _sha256_file(self.model_path)
        return (
            f"ONNX 模型已加载: {self.model_path.name} / "
            f"device={self.device} / providers={providers} / "
            f"input={self.input_shape} / sha256={self.model_sha256[:16]}"
        )

    def _open_session(self) -> None:
        import onnxruntime as ort

        for path in self.dll_search_paths:
            if path.exists() and hasattr(os, "add_dll_directory"):
                self._dll_handles.append(os.add_dll_directory(str(path)))
            os.environ["PATH"] = f"{path}{os.pathsep}{os.environ.get('PATH', '')}"

        provider_name = {
            "cuda": "CUDAExecutionProvider",
            "directml": "DmlExecutionProvider",
            "cpu": "CPUExecutionProvider",
        }.get(
            self.execution_provider if self.use_cuda or self.execution_provider != "cuda" else "cpu",
            "CPUExecutionProvider",
        )
        requested = (
            [provider_name, "CPUExecutionProvider"]
            if provider_name != "CPUExecutionProvider"
            else ["CPUExecutionProvider"]
        )
        self.session = None
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=requested,
        )
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        self.input_shape = tuple(input_info.shape)
        self.output_shapes = [tuple(item.shape) for item in self.session.get_outputs()]
        active = self.session.get_providers()
        active_provider = next(
            (name for name in requested if name in active),
            "CPUExecutionProvider",
        )
        self.device = {
            "CUDAExecutionProvider": "cuda",
            "DmlExecutionProvider": "directml",
            "CPUExecutionProvider": "cpu",
        }[active_provider]

    def configure(
        self,
        confidence: float,
        use_cuda: bool,
        image_size: int,
        execution_provider: str,
    ) -> bool:
        previous_device = (self.device, self.image_size, self.execution_provider)
        previous_requested_provider = self.execution_provider
        previous_effective_provider = (
            previous_requested_provider
            if self.use_cuda or previous_requested_provider != "cuda"
            else "cpu"
        )
        self.confidence = confidence
        self.image_size = int(image_size)
        self.use_cuda = use_cuda
        self.execution_provider = execution_provider
        current_provider = (
            self.execution_provider
            if self.use_cuda or self.execution_provider != "cuda"
            else "cpu"
        )
        if (
            self.session is None
            or current_provider != previous_effective_provider
        ):
            self._open_session()
        return (self.device, self.image_size, self.execution_provider) != previous_device

    def detect(
        self,
        frame: np.ndarray,
    ) -> tuple[list[tuple[int, int, int, int, float]], float]:
        if self.session is None:
            return [], 0.0

        started = time.perf_counter()
        timing: dict[str, float] = {}
        stage_started = time.perf_counter()
        image, gain, pad_x, pad_y = self._letterbox(frame)
        timing["letterbox_ms"] = (time.perf_counter() - stage_started) * 1000
        stage_started = time.perf_counter()
        tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(
            tensor.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        )
        timing["tensor_ms"] = (time.perf_counter() - stage_started) * 1000
        stage_started = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: tensor})
        timing["session_ms"] = (time.perf_counter() - stage_started) * 1000
        stage_started = time.perf_counter()
        prediction = np.asarray(outputs[0])
        if prediction.ndim == 3:
            prediction = prediction[0]
        if prediction.shape[0] < prediction.shape[1]:
            prediction = prediction.transpose(1, 0)
        if prediction.shape[1] < 5:
            timing["postprocess_ms"] = (time.perf_counter() - stage_started) * 1000
            timing["total_ms"] = (time.perf_counter() - started) * 1000
            self.last_timing = timing
            return [], 0.0

        boxes_xywh = prediction[:, :4]
        class_scores = prediction[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_scores)), class_ids]
        keep = (class_ids == 0) & (scores >= self.confidence)
        if not np.any(keep):
            timing["postprocess_ms"] = (time.perf_counter() - stage_started) * 1000
            timing["total_ms"] = (time.perf_counter() - started) * 1000
            self.last_timing = timing
            return [], 0.0

        boxes_xywh = boxes_xywh[keep]
        scores = scores[keep]
        boxes: list[list[int]] = []
        for cx, cy, width, height in boxes_xywh:
            x1 = (float(cx) - float(width) / 2 - pad_x) / gain
            y1 = (float(cy) - float(height) / 2 - pad_y) / gain
            x2 = (float(cx) + float(width) / 2 - pad_x) / gain
            y2 = (float(cy) + float(height) / 2 - pad_y) / gain
            boxes.append(
                [
                    max(0, round(x1)),
                    max(0, round(y1)),
                    max(0, round(x2 - x1)),
                    max(0, round(y2 - y1)),
                ]
            )

        selected = cv2.dnn.NMSBoxes(
            boxes,
            [float(score) for score in scores],
            self.confidence,
            0.7,
        )
        detections: list[tuple[int, int, int, int, float]] = []
        for raw_index in selected:
            index = int(np.asarray(raw_index).reshape(-1)[0])
            x, y, width, height = boxes[index]
            detections.append(
                (
                    x,
                    y,
                    min(frame.shape[1], x + width),
                    min(frame.shape[0], y + height),
                    float(scores[index]),
                )
            )
        detections.sort(key=lambda item: item[4], reverse=True)
        detections = detections[:5]
        timing["postprocess_ms"] = (time.perf_counter() - stage_started) * 1000
        timing["total_ms"] = (time.perf_counter() - started) * 1000
        self.last_timing = timing
        return detections, max((item[4] for item in detections), default=0.0)

    def _letterbox(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, float, float, float]:
        height, width = frame.shape[:2]
        target_height, target_width = self._target_input_size()
        gain = min(target_width / width, target_height / height)
        resized_width = round(width * gain)
        resized_height = round(height * gain)
        pad_width = target_width - resized_width
        pad_height = target_height - resized_height
        if not self._has_fixed_input_shape():
            stride = 32
            pad_width %= stride
            pad_height %= stride
        left = pad_width / 2
        top = pad_height / 2

        if (width, height) != (resized_width, resized_height):
            frame = cv2.resize(
                frame,
                (resized_width, resized_height),
                interpolation=cv2.INTER_LINEAR,
            )
        border_left = round(left - 0.1)
        border_right = round(left + 0.1)
        border_top = round(top - 0.1)
        border_bottom = round(top + 0.1)
        frame = cv2.copyMakeBorder(
            frame,
            border_top,
            border_bottom,
            border_left,
            border_right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return frame, gain, left, top

    def _has_fixed_input_shape(self) -> bool:
        return (
            len(self.input_shape) == 4
            and isinstance(self.input_shape[2], int)
            and isinstance(self.input_shape[3], int)
        )

    def _target_input_size(self) -> tuple[int, int]:
        if self._has_fixed_input_shape():
            return int(self.input_shape[2]), int(self.input_shape[3])
        return self.image_size, self.image_size

    @staticmethod
    def resize_for_inference(
        frame: np.ndarray,
        max_width: int = 960,
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
