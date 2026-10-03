from __future__ import annotations

import unittest
from pathlib import Path

from fishing_assistant.engine import ScreenCapture
from fishing_assistant.input_control import TargetWindowInfo
from fishing_assistant.onnx_detector import OnnxBiteDetector


class CaptureLogicTests(unittest.TestCase):
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
