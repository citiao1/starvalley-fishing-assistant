from __future__ import annotations

import json
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
    confidence: float = 0.05
    bite_action_confidence: float = 0.15
    bite_confirm_frames: int = 1
    feed_confirm_frames: int = 3
    reel_cooldown_seconds: float = 1.5
    inference_fps: float = 30.0
    preview_fps: float = 30.0
    yolo_imgsz: int = 1280
    use_cuda: bool = True
    input_method: str = "sendinput_scan"
    monitor_index: int = 0


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
    def icon_path(self) -> Path:
        return self.resources / "assets" / "app_icon.ico"

    @property
    def zero_template_path(self) -> Path:
        return self.resources / "templates" / "feed_red_0.png"

    @property
    def nonzero_template_path(self) -> Path:
        return self.resources / "templates" / "feed_red_2.png"


def load_settings(paths: AppPaths) -> AppSettings:
    if not paths.settings_file.exists():
        return AppSettings()
    try:
        values = json.loads(paths.settings_file.read_text(encoding="utf-8"))
        defaults = asdict(AppSettings())
        defaults.update({key: value for key, value in values.items() if key in defaults})
        return AppSettings(**defaults)
    except (OSError, TypeError, ValueError):
        return AppSettings()


def save_settings(paths: AppPaths, settings: AppSettings) -> None:
    paths.settings_file.write_text(
        json.dumps(asdict(settings), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
