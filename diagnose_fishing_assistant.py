from __future__ import annotations

import importlib.util
import json
import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fishing_assistant.config import AppPaths  # noqa: E402
from fishing_assistant.feed_detector import FeedZeroDetector  # noqa: E402


def check_import(name: str) -> dict:
    try:
        module = __import__(name)
        return {"ok": True, "version": getattr(module, "__version__", "installed")}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    paths = AppPaths()
    report: dict = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "root": str(ROOT),
        "paths": {
            "model": str(paths.model_path),
            "zero_template": str(paths.zero_template_path),
            "nonzero_template": str(paths.nonzero_template_path),
            "log": str(paths.log_file),
        },
        "files": {},
        "imports": {},
        "checks": {},
    }

    for name, path in (
        ("model", paths.model_path),
        ("zero_template", paths.zero_template_path),
        ("nonzero_template", paths.nonzero_template_path),
    ):
        report["files"][name] = {
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
        }

    for name in ("cv2", "numpy", "torch", "ultralytics", "mss", "PySide6", "PyInstaller"):
        report["imports"][name] = check_import(name)

    try:
        detector = FeedZeroDetector.from_images(
            paths.zero_template_path,
            paths.nonzero_template_path,
        )
        report["checks"]["feed_templates"] = {
            "ok": True,
            "zero_shape": list(detector.zero_template.shape),
            "nonzero_shape": list(detector.nonzero_template.shape),
        }
    except Exception as exc:
        report["checks"]["feed_templates"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        from ultralytics import YOLO

        model = YOLO(str(paths.model_path))
        report["checks"]["yolo_model"] = {"ok": True, "names": model.names}
    except Exception as exc:
        report["checks"]["yolo_model"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        import torch
        import torchvision
        from torchvision.ops import nms

        boxes = torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]],
            dtype=torch.float32,
        )
        scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
        kept = nms(boxes, scores, 0.5)
        report["checks"]["torchvision_nms"] = {
            "ok": True,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "has_ops": bool(torchvision.extension._has_ops()),
            "kept": kept.tolist(),
        }
    except Exception as exc:
        report["checks"]["torchvision_nms"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        import mss

        with mss.MSS() as capture:
            report["checks"]["screen_capture"] = {
                "ok": True,
                "monitors": capture.monitors,
            }
    except Exception as exc:
        report["checks"]["screen_capture"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    output = paths.data_dir / "diagnostics.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n诊断报告已写入: {output}")
    return 0 if all(item.get("ok", False) for item in report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
