from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fishing_assistant.config import AppSettings, load_settings


class SettingsMigrationTests(unittest.TestCase):
    def make_paths(self, directory: str):
        return SimpleNamespace(settings_file=Path(directory) / "settings.json")

    def test_missing_new_fields_use_safe_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_paths(directory)
            paths.settings_file.write_text(
                json.dumps(
                    {
                        "preview_only": True,
                        "inference_fps": 25,
                        "onnx_provider": "directml",
                    }
                ),
                encoding="utf-8",
            )

            settings = load_settings(paths)

        self.assertIsInstance(settings, AppSettings)
        self.assertTrue(settings.preview_enabled)
        self.assertTrue(settings.pause_when_inactive)
        self.assertEqual(settings.capture_source, "game_window")
        self.assertEqual(settings.capture_buffer, 2)
        self.assertEqual(settings.opencv_threads, 1)
        self.assertEqual(settings.onnx_device_id, 1)

    def test_invalid_runtime_values_are_clamped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_paths(directory)
            paths.settings_file.write_text(
                json.dumps(
                    {
                        "capture_source": "invalid",
                        "capture_fps": 999,
                        "capture_buffer": -5,
                        "window_probe_interval_seconds": 0,
                        "opencv_threads": "bad",
                        "monitor_index": -1,
                        "onnx_device_id": 99,
                    }
                ),
                encoding="utf-8",
            )

            settings = load_settings(paths)

        self.assertEqual(settings.capture_source, "game_window")
        self.assertEqual(settings.capture_fps, 60.0)
        self.assertEqual(settings.capture_buffer, 1)
        self.assertEqual(settings.window_probe_interval_seconds, 0.25)
        self.assertEqual(settings.opencv_threads, 1)
        self.assertEqual(settings.monitor_index, 0)
        self.assertEqual(settings.onnx_device_id, 16)


if __name__ == "__main__":
    unittest.main()
