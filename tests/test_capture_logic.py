from __future__ import annotations

import unittest
from pathlib import Path

from fishing_assistant.config import AppSettings
from fishing_assistant.engine import ScreenCapture, _needs_frame_processing
from ctypes import wintypes

from fishing_assistant.input_control import TargetWindowInfo, _same_hwnd
from fishing_assistant.onnx_detector import OnnxBiteDetector


class CaptureLogicTests(unittest.TestCase):
    def test_no_detection_or_preview_disables_frame_processing(self) -> None:
        settings = AppSettings(
            bite_detection_enabled=False,
            auto_feed_enabled=False,
            preview_enabled=False,
        )
        self.assertFalse(_needs_frame_processing(settings))

    def test_any_visual_work_keeps_frame_processing_enabled(self) -> None:
        self.assertTrue(
            _needs_frame_processing(
                AppSettings(
                    bite_detection_enabled=True,
                    auto_feed_enabled=False,
                    preview_enabled=False,
                )
            )
        )
        self.assertTrue(
            _needs_frame_processing(
                AppSettings(
                    bite_detection_enabled=False,
                    auto_feed_enabled=True,
                    preview_enabled=False,
                )
            )
        )
        self.assertTrue(
            _needs_frame_processing(
                AppSettings(
                    bite_detection_enabled=False,
                    auto_feed_enabled=False,
                    preview_enabled=True,
                )
            )
        )

    def make_capture(self) -> ScreenCapture:
        capture = ScreenCapture.__new__(ScreenCapture)
        capture._monitors = [
            {"left": 0, "top": 0, "width": 3840, "height": 1080},
            {"left": 0, "top": 0, "width": 1920, "height": 1080},
            {"left": 1920, "top": 0, "width": 1920, "height": 1080},
        ]
        return capture

    def test_window_rect_uses_monitor_with_largest_overlap(self) -> None:
        capture = self.make_capture()
        selected = capture._monitor_for_rect((1500, 100, 2050, 900))
        self.assertIsNotNone(selected)
        self.assertEqual(selected[0], 1)

        selected = capture._monitor_for_rect((1900, 100, 2500, 900))
        self.assertIsNotNone(selected)
        self.assertEqual(selected[0], 2)

    def test_window_rect_falls_back_to_nearest_monitor(self) -> None:
        capture = self.make_capture()
        selected = capture._monitor_for_rect((5000, 100, 5200, 300))
        self.assertIsNotNone(selected)
        self.assertEqual(selected[0], 2)

    def test_target_window_description_is_stable(self) -> None:
        info = TargetWindowInfo(
            hwnd=123,
            pid=456,
            process_name="petitplanet.exe",
            title="星布谷地",
            valid=True,
        )
        self.assertEqual(
            info.description,
            "petitplanet.exe hwnd=123 pid=456 title='星布谷地'",
        )

    def test_hwnd_object_and_integer_compare_by_value(self) -> None:
        self.assertTrue(_same_hwnd(wintypes.HWND(123), 123))
        self.assertFalse(_same_hwnd(wintypes.HWND(123), 456))
        self.assertFalse(_same_hwnd(wintypes.HWND(0), 0))

    def test_window_region_does_not_start_capture_before_explicit_start(self) -> None:
        capture = self.make_capture()
        capture.capture_source = "game_window"
        capture._primary_monitor_index = 1
        capture.monitor_index = 1
        capture.monitor = capture._monitors[1]
        capture._started = False
        capture._dxcam = None
        capture._screen_region = None
        capture._mss_region = None
        capture._region_key = None
        capture._start_dxcam = lambda: (_ for _ in ()).throw(
            AssertionError("DXcam must not start while resolving the window")
        )
        info = TargetWindowInfo(
            hwnd=123,
            pid=456,
            process_name="petitplanet.exe",
            title="星布谷地",
            valid=True,
            client_rect=(100, 100, 1200, 800),
        )

        self.assertTrue(capture.resolve_target_region(info))
        self.assertFalse(capture._started)
        self.assertIsNone(capture._dxcam)
        self.assertEqual(capture.size, (1100, 700))

    def test_invalid_window_does_not_fall_back_to_monitor_region(self) -> None:
        capture = self.make_capture()
        capture.capture_source = "game_window"
        capture._primary_monitor_index = 1
        capture.monitor_index = 1
        capture.monitor = capture._monitors[1]
        capture._started = False
        capture._screen_region = None
        capture._mss_region = None
        capture._region_key = None

        info = TargetWindowInfo(
            hwnd=0,
            pid=0,
            process_name="",
            title="",
            valid=False,
            client_rect=None,
        )

        self.assertFalse(capture.resolve_target_region(info))
        self.assertEqual(capture.size, (0, 0))
        self.assertEqual(capture.backend_name, "等待游戏窗口")

    def test_start_is_idempotent_until_capture_is_paused(self) -> None:
        capture = self.make_capture()
        capture.capture_source = "game_window"
        capture._mss = object()
        capture.monitor = capture._monitors[1]
        capture._screen_region = (100, 100, 1200, 800)
        capture._mss_region = {
            "left": 100,
            "top": 100,
            "width": 1100,
            "height": 700,
        }
        capture._started = False
        calls: list[int] = []

        def fake_start_dxcam() -> None:
            calls.append(1)

        capture._start_dxcam = fake_start_dxcam

        self.assertTrue(capture.start())
        self.assertTrue(capture.start())
        self.assertEqual(len(calls), 1)
        self.assertTrue(capture.is_started)


class OnnxSessionSafetyTests(unittest.TestCase):
    def test_load_falls_back_to_device_zero(self) -> None:
        detector = OnnxBiteDetector(
            Path("models/best.onnx"),
            confidence=0.05,
            image_size=960,
            use_cuda=True,
            execution_provider="directml",
            device_id=1,
        )

        class FakeValue:
            def __init__(self, name: str, shape: tuple[object, ...]) -> None:
                self.name = name
                self.shape = shape

        class FakeSession:
            def get_providers(self):
                return ["DmlExecutionProvider", "CPUExecutionProvider"]

            def get_inputs(self):
                return [FakeValue("images", ("batch", 3, "height", "width"))]

            def get_outputs(self):
                return [FakeValue("output", (1, 5, 10))]

        calls: list[int] = []

        def fake_open() -> None:
            calls.append(detector.device_id)
            if detector.device_id == 1:
                raise RuntimeError("device 1 unavailable")
            detector.session = FakeSession()
            detector.input_name = "images"
            detector.input_shape = ("batch", 3, "height", "width")
            detector.output_shapes = [(1, 5, 10)]
            detector.device = "directml:0"

        detector._open_session = fake_open
        message = detector.load()

        self.assertEqual(calls, [1, 0])
        self.assertEqual(detector.device_id, 0)
        self.assertEqual(detector.device, "directml:0")
        self.assertIn("device_id=1", detector.last_error)
        self.assertIn("directml:0", message)

    def test_failed_reconfigure_keeps_old_session_and_settings(self) -> None:
        detector = OnnxBiteDetector(
            Path("missing.onnx"),
            confidence=0.05,
            image_size=960,
            use_cuda=True,
            execution_provider="directml",
            device_id=0,
        )
        old_session = object()
        detector.session = old_session
        detector.device = "directml:0"
        detector.input_name = "images"
        detector.input_shape = ("batch", 3, "height", "width")

        def fail_open() -> None:
            raise RuntimeError("provider unavailable")

        detector._open_session = fail_open
        changed = detector.configure(
            confidence=0.2,
            use_cuda=True,
            image_size=768,
            execution_provider="cuda",
            device_id=1,
        )

        self.assertFalse(changed)
        self.assertIs(detector.session, old_session)
        self.assertEqual(detector.device, "directml:0")
        self.assertEqual(detector.image_size, 960)
        self.assertEqual(detector.execution_provider, "directml")
        self.assertEqual(detector.device_id, 0)
        self.assertEqual(detector.confidence, 0.05)
        self.assertIn("provider unavailable", detector.last_error)


if __name__ == "__main__":
    unittest.main()
