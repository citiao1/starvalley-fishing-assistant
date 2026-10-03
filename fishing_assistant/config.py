from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", project_root()))
    return project_root()


@dataclass
class AppSettings:
    master_enabled: bool = False
    bite_detection_enabled: bool = True
    auto_reel_enabled: bool = False
    auto_feed_enabled: bool = False
    preview_only: bool = True
    preview_enabled: bool = True
    pause_when_inactive: bool = True
    confidence: float = 0.05
    bite_action_confidence: float = 0.15
    bite_confirm_frames: int = 1
    feed_confirm_frames: int = 3
    feed_min_interval_seconds: float = 0.35
    feed_retry_interval_seconds: float = 0.75
    feed_input_failure_limit: int = 3
    reel_cooldown_seconds: float = 1.5
    inference_result_max_age_seconds: float = 0.30
    inference_fps: float = 25.0
    preview_fps: float = 12.0
    capture_fps: float = 30.0
    capture_buffer: int = 2
    capture_source: str = "game_window"
    window_probe_interval_seconds: float = 0.35
    opencv_threads: int = 1
    yolo_imgsz: int = 960
    use_cuda: bool = True
    input_method: str = "sendinput_scan"
    monitor_index: int = 0
    capture_device_index: int = 0
    onnx_device_id: int = 1
    inference_backend: str = "pytorch"
    onnx_provider: str = "cuda"


class AppPaths:
    def __init__(self) -> None:
        self.root = project_root()
        self.resources = resource_root()
        self.data_dir = self.root / "app_data"
        self.log_dir = self.data_dir / "logs"
        self.settings_file = self.data_dir / "settings.json"
        self.log_file = self.log_dir / "session.log"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_path(self) -> Path:
        bundled = self.resources / "models" / "best.pt"
        if bundled.exists():
            return bundled
        return self.root / "runs" / "fishing_yolo11n_v1" / "weights" / "best.pt"

    @property
    def onnx_model_path(self) -> Path:
        bundled = self.resources / "models" / "best.onnx"
        if bundled.exists():
            return bundled
        return self.root / "models" / "best.onnx"

    @property
    def icon_path(self) -> Path:
        return self.resources / "assets" / "app_icon.ico"

    @property
    def zero_template_path(self) -> Path:
        return self.resources / "templates" / "feed_red_0.png"

    @property
    def nonzero_template_path(self) -> Path:
        return self.resources / "templates" / "feed_red_2.png"


def load_settings(paths: AppPaths) -> AppSettings:
    defaults = asdict(AppSettings())
    try:
        if paths.settings_file.exists():
            values = json.loads(paths.settings_file.read_text(encoding="utf-8"))
            if isinstance(values, dict):
                defaults.update(
                    {key: value for key, value in values.items() if key in defaults}
                )
    except (OSError, TypeError, ValueError):
        pass

    backend = os.environ.get("FISHING_ASSISTANT_BACKEND", "").strip().lower()
    provider = os.environ.get("FISHING_ASSISTANT_ONNX_PROVIDER", "").strip().lower()
    if backend in {"pytorch", "onnx"}:
        defaults["inference_backend"] = backend
    if provider in {"cuda", "directml", "cpu"}:
        defaults["onnx_provider"] = provider
    for key in (
        "master_enabled",
        "bite_detection_enabled",
        "auto_reel_enabled",
        "auto_feed_enabled",
        "preview_only",
        "preview_enabled",
        "pause_when_inactive",
        "use_cuda",
    ):
        defaults[key] = _coerce_bool(defaults[key], defaults[key])
    defaults["capture_source"] = _safe_choice(
        defaults["capture_source"],
        {"game_window", "monitor"},
        "game_window",
    )
    defaults["inference_backend"] = _safe_choice(
        defaults["inference_backend"],
        {"pytorch", "onnx"},
        "pytorch",
    )
    defaults["onnx_provider"] = _safe_choice(
        defaults["onnx_provider"],
        {"cuda", "directml", "cpu"},
        "cuda",
    )
    defaults["input_method"] = _safe_choice(
        defaults["input_method"],
        {
            "sendinput_scan",
            "sendinput_hybrid",
            "sendinput_raw_scan",
            "sendinput_vk",
            "postmessage",
        },
        "sendinput_scan",
    )
    defaults["confidence"] = _bounded_float(defaults["confidence"], 0.005, 0.99, 0.05)
    defaults["bite_action_confidence"] = _bounded_float(
        defaults["bite_action_confidence"],
        0.02,
        0.99,
        0.15,
    )
    defaults["bite_confirm_frames"] = _bounded_int(
        defaults["bite_confirm_frames"],
        1,
        10,
        1,
    )
    defaults["feed_confirm_frames"] = _bounded_int(
        defaults["feed_confirm_frames"],
        1,
        10,
        3,
    )
    defaults["feed_min_interval_seconds"] = _bounded_float(
        defaults["feed_min_interval_seconds"],
        0.2,
        1.0,
        0.35,
    )
    defaults["feed_retry_interval_seconds"] = max(
        defaults["feed_min_interval_seconds"],
        _bounded_float(
            defaults["feed_retry_interval_seconds"],
            0.2,
            3.0,
            0.75,
        ),
    )
    defaults["feed_input_failure_limit"] = _bounded_int(
        defaults["feed_input_failure_limit"],
        1,
        10,
        3,
    )
    defaults["reel_cooldown_seconds"] = _bounded_float(
        defaults["reel_cooldown_seconds"],
        0.2,
        10.0,
        1.5,
    )
    defaults["inference_result_max_age_seconds"] = _bounded_float(
        defaults["inference_result_max_age_seconds"],
        0.1,
        1.0,
        0.3,
    )
    defaults["inference_fps"] = _bounded_float(
        defaults["inference_fps"],
        1.0,
        60.0,
        25.0,
    )
    defaults["preview_fps"] = _bounded_float(
        defaults["preview_fps"],
        5.0,
        30.0,
        12.0,
    )
    defaults["capture_fps"] = _bounded_float(defaults["capture_fps"], 5.0, 60.0, 30.0)
    defaults["capture_buffer"] = _bounded_int(defaults["capture_buffer"], 1, 8, 2)
    defaults["window_probe_interval_seconds"] = _bounded_float(
        defaults["window_probe_interval_seconds"],
        0.25,
        2.0,
        0.35,
    )
    defaults["opencv_threads"] = _bounded_int(defaults["opencv_threads"], 1, 8, 1)
    defaults["monitor_index"] = _bounded_int(defaults["monitor_index"], 0, 16, 0)
    defaults["capture_device_index"] = _bounded_int(
        defaults["capture_device_index"],
        0,
        16,
        0,
    )
    defaults["onnx_device_id"] = _bounded_int(defaults["onnx_device_id"], 0, 16, 1)
    defaults["yolo_imgsz"] = _bounded_int(defaults["yolo_imgsz"], 320, 1280, 960)
    return AppSettings(**defaults)


def save_settings(paths: AppPaths, settings: AppSettings) -> None:
    paths.settings_file.write_text(
        json.dumps(asdict(settings), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _bounded_int(value: object, lower: int, upper: int, fallback: int) -> int:
    try:
        return max(lower, min(upper, int(value)))
    except (TypeError, ValueError):
        return fallback


def _coerce_bool(value: object, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return fallback


def _safe_choice(value: object, choices: set[str], fallback: str) -> str:
    normalized = str(value).strip().lower()
    return normalized if normalized in choices else fallback


def _bounded_float(
    value: object,
    lower: float,
    upper: float,
    fallback: float,
) -> float:
    try:
        return max(lower, min(upper, float(value)))
    except (TypeError, ValueError):
        return fallback
