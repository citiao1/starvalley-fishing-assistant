from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import replace

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
        QVBoxLayout,
        QWidget,
    )

from .config import AppPaths, AppSettings, load_settings, save_settings
from .engine import WorkerController


class HotkeyWatcher(QObject):
    emergency = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._running = True
        self._was_down = False

    @Slot()
    def run(self) -> None:
        while self._running:
            is_down = bool(ctypes.windll.user32.GetAsyncKeyState(0x7B) & 0x8000)
            if is_down and not self._was_down:
                self.emergency.emit()
            self._was_down = is_down
            time.sleep(0.08)

    def stop(self) -> None:
        self._running = False


class StatusValue(QLabel):
    def __init__(self, value: str = "--") -> None:
        super().__init__(value)
        self.setObjectName("statusValue")
        self.setWordWrap(True)


class FrameMailbox(QObject):
    """Coalesce preview frames so a slow UI never consumes an old queue."""

    frame_available = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._latest = None
        self._notification_pending = False

    def push(self, image) -> None:
        with self._lock:
            self._latest = image
            if self._notification_pending:
                return
            self._notification_pending = True
        self.frame_available.emit()

    def take(self):
        with self._lock:
            image = self._latest
            self._latest = None
            self._notification_pending = False
            return image


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.paths = AppPaths()
        self.settings = load_settings(self.paths)
        self.controller: WorkerController | None = None
        self.hotkey_thread: QThread | None = None
        self.hotkey_watcher: HotkeyWatcher | None = None
        self._session_started = False
        self._last_action_at = "--"
        self._display_frames = 0
        self._display_fps = 0.0
        self._display_fps_started = time.perf_counter()
        self._initializing = True
        self.settings_dialog: QDialog | None = None
        self.frame_mailbox = FrameMailbox()

        self.setWindowTitle("星布谷地 · 自动钓鱼助手")
        if self.paths.icon_path.exists():
            self.setWindowIcon(QIcon(str(self.paths.icon_path)))
        self.resize(1440, 900)
        self.setMinimumSize(1120, 720)
        self._build_ui()
        self.frame_mailbox.frame_available.connect(self._on_frame)
        self._apply_settings_to_controls()
        self._initializing = False
        self._start_hotkey_watcher()

    def _build_ui(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background: #12161c; color: #e8edf2; font-size: 13px; }
            QMainWindow { background: #0e1116; }
            QFrame#panel, QGroupBox {
                background: #181e26; border: 1px solid #2b3542; border-radius: 8px;
            }
            QGroupBox { margin-top: 12px; padding: 16px 12px 12px 12px; }
            QGroupBox::title {
                subcontrol-origin: margin; left: 12px; padding: 0 6px;
                color: #9eb3c8; font-weight: 700;
            }
            QLabel#title { color: #f5f7fa; font-size: 24px; font-weight: 800; }
            QLabel#subtitle { color: #8b9aaa; font-size: 12px; }
            QLabel#section { color: #9eb3c8; font-weight: 700; }
            QLabel#statusValue { color: #eef4fa; font-size: 14px; font-weight: 700; }
            QLabel#metricKey { color: #8291a1; }
            QCheckBox { spacing: 8px; padding: 6px 2px; }
            QCheckBox::indicator {
                width: 38px; height: 20px; border-radius: 10px;
                background: #303b48; border: 1px solid #4a5968;
            }
            QCheckBox::indicator:checked { background: #35c58a; border: 1px solid #55e0a5; }
            QCheckBox::indicator:unchecked { background: #303b48; }
            QSpinBox, QDoubleSpinBox {
                background: #11161c; border: 1px solid #354252; border-radius: 5px;
                padding: 5px; min-height: 25px;
            }
            QPushButton {
                background: #263241; border: 1px solid #405269; border-radius: 5px;
                padding: 8px 12px; font-weight: 700;
            }
            QPushButton:hover { background: #314155; }
            QPushButton#primary { background: #2b9e73; border-color: #49c493; }
            QPushButton#danger { background: #a94d52; border-color: #d36b70; }
            QPushButton#danger:hover { background: #c35a60; }
            QPlainTextEdit {
                background: #0d1116; border: 1px solid #2b3542; border-radius: 5px;
                color: #b9c8d6; font-family: Consolas, monospace; font-size: 12px;
            }
            QScrollArea { border: 0; }
            """
        )
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(20, 18, 20, 18)
        root_layout.setSpacing(14)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("星布谷地 · 自动钓鱼助手")
        title.setObjectName("title")
        subtitle = QLabel("视觉识别与安全动作控制台")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()
        self.settings_button = QPushButton("⚙ 设置")
        self.settings_button.setToolTip("打开检测参数设置")
        self.settings_button.clicked.connect(self._open_settings_dialog)
        header.addWidget(self.settings_button, alignment=Qt.AlignmentFlag.AlignTop)
        self.run_state = QLabel("未启动")
        self.run_state.setStyleSheet("color:#f0b35b; font-weight:800; padding:8px 12px;")
        header.addWidget(self.run_state, alignment=Qt.AlignmentFlag.AlignTop)
        root_layout.addLayout(header)

        columns = QHBoxLayout()
        columns.setSpacing(14)
        root_layout.addLayout(columns, 1)

        left = QVBoxLayout()
        left.setSpacing(12)
        preview_panel = QFrame()
        preview_panel.setObjectName("panel")
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        preview_title = QLabel("实时预览")
        preview_title.setObjectName("section")
        preview_layout.addWidget(preview_title)
        self.preview = QLabel("启动采集后显示屏幕预览")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(760, 520)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview.setStyleSheet(
            "background:#0b0f14; border:1px solid #27313c; color:#6e7d8d;"
        )
        preview_layout.addWidget(self.preview, 1)
        left.addWidget(preview_panel, 1)

        action_row = QHBoxLayout()
        self.start_button = QPushButton("启动采集与识别")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.start_session)
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.stop_session)
        self.emergency_button = QPushButton("紧急停止（F12）")
        self.emergency_button.setObjectName("danger")
        self.emergency_button.clicked.connect(self.emergency_stop)
        action_row.addWidget(self.start_button)
        action_row.addWidget(self.stop_button)
        action_row.addWidget(self.emergency_button)
        action_row.addStretch()
        left.addLayout(action_row)
        columns.addLayout(left, 1)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_content = QWidget()
        right = QVBoxLayout(right_content)
        right.setContentsMargins(2, 0, 2, 0)
        right.setSpacing(10)
        right_scroll.setWidget(right_content)
        columns.addWidget(right_scroll, 0)
        right_scroll.setMinimumWidth(350)
        right_scroll.setMaximumWidth(410)

        switches = QGroupBox("安全开关")
        switches_layout = QVBoxLayout(switches)
        self.master = QCheckBox("总开关")
        self.bite_detection = QCheckBox("上钩检测")
        self.auto_reel = QCheckBox("自动收杆（左键）")
        self.auto_feed = QCheckBox("自动喂食（Z）")
        self.preview_only = QCheckBox("预览 / 仅识别模式")
        for checkbox in (
            self.master,
            self.bite_detection,
            self.auto_reel,
            self.auto_feed,
            self.preview_only,
        ):
            checkbox.stateChanged.connect(self._settings_changed)
            switches_layout.addWidget(checkbox)
        right.addWidget(switches)

        metrics = QGroupBox("实时状态")
        metrics_grid = QGridLayout(metrics)
        self.resolution = StatusValue()
        self.bite_value = StatusValue()
        self.feed_value = StatusValue()
        self.status_value = StatusValue("等待启动")
        self.action_value = StatusValue()
        self.performance_value = StatusValue()
        self.target_value = StatusValue()
        for row, key, value in (
            (0, "采集分辨率", self.resolution),
            (1, "上钩置信度", self.bite_value),
            (2, "红色数字", self.feed_value),
            (3, "当前状态", self.status_value),
            (4, "最近动作", self.action_value),
            (5, "性能", self.performance_value),
            (6, "输入目标", self.target_value),
        ):
            key_label = QLabel(key)
            key_label.setObjectName("metricKey")
            metrics_grid.addWidget(key_label, row, 0)
            metrics_grid.addWidget(value, row, 1)
        right.addWidget(metrics)

        self.settings_dialog = QDialog(self)
        self.settings_dialog.setWindowTitle("检测参数设置")
        self.settings_dialog.setModal(True)
        self.settings_dialog.setMinimumWidth(420)
        settings_layout = QVBoxLayout(self.settings_dialog)
        settings_layout.setContentsMargins(18, 18, 18, 14)
        settings_hint = QLabel(
            "参数会自动保存，并在下一次安全推理边界生效。ONNX 后端可选 DirectML、CUDA 或 CPU。"
        )
        settings_hint.setObjectName("subtitle")
        settings_hint.setWordWrap(True)
        settings_layout.addWidget(settings_hint)

        params = QGroupBox("检测参数")
        form = QFormLayout(params)
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.005, 0.99)
        self.confidence.setSingleStep(0.005)
        self.confidence.setDecimals(2)
        self.bite_action_confidence = QDoubleSpinBox()
        self.bite_action_confidence.setRange(0.02, 0.99)
        self.bite_action_confidence.setSingleStep(0.01)
        self.bite_action_confidence.setDecimals(2)
        self.bite_confirm = QSpinBox()
        self.bite_confirm.setRange(1, 10)
        self.feed_confirm = QSpinBox()
        self.feed_confirm.setRange(1, 10)
        self.feed_min_interval = QDoubleSpinBox()
        self.feed_min_interval.setRange(0.2, 1.0)
        self.feed_min_interval.setSingleStep(0.05)
        self.feed_min_interval.setDecimals(2)
        self.feed_retry_interval = QDoubleSpinBox()
        self.feed_retry_interval.setRange(0.2, 3.0)
        self.feed_retry_interval.setSingleStep(0.05)
        self.feed_retry_interval.setDecimals(2)
        self.feed_failure_limit = QSpinBox()
        self.feed_failure_limit.setRange(1, 10)
        self.result_max_age = QDoubleSpinBox()
        self.result_max_age.setRange(0.1, 1.0)
        self.result_max_age.setSingleStep(0.05)
        self.result_max_age.setDecimals(2)
        self.inference_fps = QDoubleSpinBox()
        self.inference_fps.setRange(1, 60)
        self.inference_fps.setSingleStep(1)
        self.preview_fps = QDoubleSpinBox()
        self.preview_fps.setRange(5, 30)
        self.preview_fps.setSingleStep(1)
        self.yolo_imgsz = QSpinBox()
        self.yolo_imgsz.setRange(320, 1280)
        self.yolo_imgsz.setSingleStep(32)
        self.reel_cooldown = QDoubleSpinBox()
        self.reel_cooldown.setRange(0.2, 10)
        self.reel_cooldown.setSingleStep(0.1)
        self.input_method = QComboBox()
        self.input_method.addItem("SendInput（兼容模式，推荐）", "sendinput_scan")
        self.input_method.addItem("SendInput（纯扫描码）", "sendinput_raw_scan")
        self.input_method.addItem("SendInput（虚拟键）", "sendinput_vk")
        self.input_method.addItem("窗口消息（后台优先）", "postmessage")
        self.onnx_provider = QComboBox()
        self.onnx_provider.addItem("DirectML", "directml")
        self.onnx_provider.addItem("CUDA", "cuda")
        self.onnx_provider.addItem("CPU", "cpu")
        self.use_cuda = QCheckBox("PyTorch 使用 CUDA")
        form.addRow("YOLO 置信度", self.confidence)
        form.addRow("收杆最低置信度", self.bite_action_confidence)
        form.addRow("上钩确认帧", self.bite_confirm)
        form.addRow("红 0 确认帧", self.feed_confirm)
        form.addRow("喂食最小间隔", self.feed_min_interval)
        form.addRow("喂食失败重试间隔", self.feed_retry_interval)
        form.addRow("喂食失败暂停次数", self.feed_failure_limit)
        form.addRow("动作结果最大年龄", self.result_max_age)
        form.addRow("推理频率 FPS", self.inference_fps)
        form.addRow("预览频率 FPS", self.preview_fps)
        form.addRow("模型输入尺寸", self.yolo_imgsz)
        form.addRow("收杆冷却", self.reel_cooldown)
        form.addRow("输入方式", self.input_method)
        form.addRow("ONNX 推理后端", self.onnx_provider)
        form.addRow("PyTorch 推理设备", self.use_cuda)
        for widget in (
            self.confidence,
            self.bite_action_confidence,
            self.bite_confirm,
            self.feed_confirm,
            self.feed_min_interval,
            self.feed_retry_interval,
            self.feed_failure_limit,
            self.result_max_age,
            self.inference_fps,
            self.preview_fps,
            self.yolo_imgsz,
            self.reel_cooldown,
        ):
            widget.valueChanged.connect(self._settings_changed)
        self.input_method.currentIndexChanged.connect(self._settings_changed)
        self.onnx_provider.currentIndexChanged.connect(self._settings_changed)
        self.use_cuda.stateChanged.connect(self._settings_changed)
        settings_layout.addWidget(params)
        settings_buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close,
            parent=self.settings_dialog,
        )
        settings_buttons.rejected.connect(self.settings_dialog.reject)
        settings_layout.addWidget(settings_buttons)

        input_tests = QGroupBox("输入测试")
        input_tests_layout = QHBoxLayout(input_tests)
        self.test_z_button = QPushButton("延迟测试 Z")
        self.test_click_button = QPushButton("延迟测试左键")
        self.test_z_button.clicked.connect(lambda: self._schedule_input_test("z"))
        self.test_click_button.clicked.connect(lambda: self._schedule_input_test("left_click"))
        input_tests_layout.addWidget(self.test_z_button)
        input_tests_layout.addWidget(self.test_click_button)
        right.addWidget(input_tests)

        diagnostics = QGroupBox("诊断日志")
        diagnostics_layout = QVBoxLayout(diagnostics)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(180)
        diagnostics_layout.addWidget(self.log_view)
        right.addWidget(diagnostics)
        right.addStretch()

        footer = QLabel(
            f"模型: {self.paths.model_path}    |    日志: {self.paths.log_file}    |    全局停止: F12"
        )
        footer.setObjectName("subtitle")
        footer.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root_layout.addWidget(footer)

    @Slot()
    def _open_settings_dialog(self) -> None:
        if self.settings_dialog is None:
            return
        self.settings_dialog.adjustSize()
        self.settings_dialog.exec()

    def _apply_settings_to_controls(self) -> None:
        self.master.setChecked(self.settings.master_enabled)
        self.bite_detection.setChecked(self.settings.bite_detection_enabled)
        self.auto_reel.setChecked(self.settings.auto_reel_enabled)
        self.auto_feed.setChecked(self.settings.auto_feed_enabled)
        self.preview_only.setChecked(self.settings.preview_only)
        self.confidence.setValue(self.settings.confidence)
        self.bite_action_confidence.setValue(self.settings.bite_action_confidence)
        self.bite_confirm.setValue(self.settings.bite_confirm_frames)
        self.feed_confirm.setValue(self.settings.feed_confirm_frames)
        self.feed_min_interval.setValue(self.settings.feed_min_interval_seconds)
        self.feed_retry_interval.setValue(self.settings.feed_retry_interval_seconds)
        self.feed_failure_limit.setValue(self.settings.feed_input_failure_limit)
        self.result_max_age.setValue(self.settings.inference_result_max_age_seconds)
        self.inference_fps.setValue(self.settings.inference_fps)
        self.preview_fps.setValue(self.settings.preview_fps)
        self.yolo_imgsz.setValue(self.settings.yolo_imgsz)
        self.reel_cooldown.setValue(self.settings.reel_cooldown_seconds)
        input_index = self.input_method.findData(self.settings.input_method)
        self.input_method.setCurrentIndex(max(0, input_index))
        provider_index = self.onnx_provider.findData(self.settings.onnx_provider)
        self.onnx_provider.setCurrentIndex(max(0, provider_index))
        self.use_cuda.setChecked(self.settings.use_cuda)

    def _read_settings(self) -> AppSettings:
        return replace(
            self.settings,
            master_enabled=self.master.isChecked(),
            bite_detection_enabled=self.bite_detection.isChecked(),
            auto_reel_enabled=self.auto_reel.isChecked(),
            auto_feed_enabled=self.auto_feed.isChecked(),
            preview_only=self.preview_only.isChecked(),
            confidence=self.confidence.value(),
            bite_action_confidence=self.bite_action_confidence.value(),
            bite_confirm_frames=self.bite_confirm.value(),
            feed_confirm_frames=self.feed_confirm.value(),
            feed_min_interval_seconds=self.feed_min_interval.value(),
            feed_retry_interval_seconds=max(
                self.feed_min_interval.value(),
                self.feed_retry_interval.value(),
            ),
            feed_input_failure_limit=self.feed_failure_limit.value(),
            inference_result_max_age_seconds=self.result_max_age.value(),
            inference_fps=self.inference_fps.value(),
            preview_fps=self.preview_fps.value(),
            yolo_imgsz=self.yolo_imgsz.value(),
            reel_cooldown_seconds=self.reel_cooldown.value(),
            input_method=str(self.input_method.currentData()),
            use_cuda=self.use_cuda.isChecked(),
            onnx_provider=str(self.onnx_provider.currentData()),
        )

    @Slot()
    def _settings_changed(self) -> None:
        if self._initializing:
            return
        self.settings = self._read_settings()
        save_settings(self.paths, self.settings)
        if self.controller is not None:
            self.controller.update_settings(self.settings)

    def start_session(self) -> None:
        if self._session_started:
            return
        self.settings = self._read_settings()
        self._display_frames = 0
        self._display_fps = 0.0
        self._display_fps_started = time.perf_counter()
        self._session_started = True
        self.run_state.setText("启动中")
        self.run_state.setStyleSheet("color:#f0b35b; font-weight:800; padding:8px 12px;")
        self.controller = WorkerController(self.paths, self.settings)
        self.controller.worker.frame_ready.connect(
            self.frame_mailbox.push,
            Qt.ConnectionType.DirectConnection,
        )
        self.controller.worker.metrics_ready.connect(self._on_metrics)
        self.controller.worker.status_ready.connect(self._on_status)
        self.controller.worker.log_ready.connect(self._on_log)
        self.controller.worker.input_test_result.connect(self._on_input_test_result)
        self.controller.thread.finished.connect(self._on_session_finished)
        self.controller.start()
        self._on_log("info", "开始启动检测会话")

    def stop_session(self) -> None:
        if self.controller is not None:
            self.controller.stop()
            self._on_log("info", "正在停止检测会话")

    def _schedule_input_test(self, action: str) -> None:
        if self.controller is None or not self._session_started:
            self._on_log("warning", "请先启动检测会话，再执行输入测试")
            return
        self._on_log("info", f"{action} 输入测试将在 2 秒后发送，请将游戏保持在前台")
        QTimer.singleShot(2000, lambda: self._run_input_test(action))

    def _run_input_test(self, action: str) -> None:
        if self.controller is None or not self._session_started:
            return
        self.controller.test_input(action)
        self._on_log(
            "info",
            f"输入测试 {action} 已提交到检测线程",
        )

    @Slot(bool, str)
    def _on_input_test_result(self, ok: bool, result: str) -> None:
        self._on_log(
            "info" if ok else "error",
            f"输入测试结果: {'成功' if ok else '失败'} / {result}",
        )

    def emergency_stop(self) -> None:
        self.master.setChecked(False)
        self.auto_reel.setChecked(False)
        self.auto_feed.setChecked(False)
        if self.controller is not None:
            self.controller.emergency_stop()
        self._on_log("warning", "紧急停止：已关闭总开关、自动收杆和自动喂食")

    @Slot()
    def _on_frame(self) -> None:
        image = self.frame_mailbox.take()
        if image is None:
            return
        self._display_frames += 1
        now = time.perf_counter()
        elapsed = now - self._display_fps_started
        if elapsed >= 0.5:
            self._display_fps = self._display_frames / elapsed
            self._display_frames = 0
            self._display_fps_started = now
        pixmap = QPixmap.fromImage(image)
        self.preview.setPixmap(
            pixmap.scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )

    @Slot(object)
    def _on_metrics(self, metrics: dict) -> None:
        self.resolution.setText(metrics["resolution"])
        self.bite_value.setText(
            f'{metrics["bite_confidence"]:.3f}  / 连续 {metrics["bite_streak"]} 帧'
        )
        self.feed_value.setText(
            f'{metrics["feed_label"]} / {metrics["feed_state"]} '
            f'(0={metrics["zero_score"]:.3f}, 非0={metrics["nonzero_score"]:.3f})'
        )
        self.status_value.setText(metrics["status"])
        self.action_value.setText(metrics["last_action"])
        self.performance_value.setText(
            f'采集 {metrics["capture_fps"]:.1f} FPS / '
            f'画面 {self._display_fps:.1f} FPS / '
            f'推理 {metrics["inference_fps"]:.1f} FPS / '
            f'推理 {metrics["inference_ms"]:.0f} ms / '
            f'采集 {metrics["capture_ms"]:.1f} ms / '
            f'结果年龄 {metrics["result_age_ms"]:.0f} ms / '
            f'跳过 {metrics["inference_skipped"]} / '
            f'推理 p95 {metrics["inference_stats"]["p95"]:.0f} ms / '
            f'会话 p95 {metrics["session_stats"]["p95"]:.0f} ms / '
            f'{metrics["onnx_device"]} input={metrics["onnx_input"]} / '
            f'{metrics["capture_backend"]}'
        )
        self.target_value.setText(
            f'{metrics["target_window"]} / {metrics["target_privilege"]} / '
            f'前台={metrics["target_foreground"]} / '
            f'{metrics["input_result"]}'
        )

    @Slot(str)
    def _on_status(self, status: str) -> None:
        self.status_value.setText(status)
        self.run_state.setText("运行中")
        self.run_state.setStyleSheet("color:#55e0a5; font-weight:800; padding:8px 12px;")

    @Slot(str, str)
    def _on_log(self, level: str, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{stamp}] {level.upper():7s} {message}")

    def _on_session_finished(self) -> None:
        self._session_started = False
        self.run_state.setText("已停止")
        self.run_state.setStyleSheet("color:#8b9aaa; font-weight:800; padding:8px 12px;")
        self.controller = None

    def _start_hotkey_watcher(self) -> None:
        self.hotkey_thread = QThread(self)
        self.hotkey_watcher = HotkeyWatcher()
        self.hotkey_watcher.moveToThread(self.hotkey_thread)
        self.hotkey_thread.started.connect(self.hotkey_watcher.run)
        self.hotkey_watcher.emergency.connect(self.emergency_stop)
        self.hotkey_thread.start()

    def closeEvent(self, event) -> None:
        self.emergency_stop()
        if self.controller is not None:
            self.controller.stop()
            self.controller.thread.quit()
            self.controller.thread.wait()
        if self.hotkey_watcher is not None:
            self.hotkey_watcher.stop()
        if self.hotkey_thread is not None:
            self.hotkey_thread.quit()
            self.hotkey_thread.wait(1000)
        event.accept()


def run() -> int:
    app = QApplication([])
    app.setApplicationName("星布谷地自动钓鱼助手")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = MainWindow()
    window.show()
    return app.exec()
