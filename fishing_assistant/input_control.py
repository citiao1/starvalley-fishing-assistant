from __future__ import annotations

import ctypes
import os
import threading
import time
from ctypes import wintypes


ULONG_PTR = ctypes.c_size_t


class _Point(ctypes.Structure):
    _fields_ = (("x", wintypes.LONG), ("y", wintypes.LONG))


class _Rect(ctypes.Structure):
    _fields_ = (
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    )


class _KeyboardInput(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class _MouseInput(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class _InputUnion(ctypes.Union):
    _fields_ = (("mi", _MouseInput), ("ki", _KeyboardInput))


class _Input(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = (("type", wintypes.DWORD), ("u", _InputUnion))


class _TokenElevation(ctypes.Structure):
    _fields_ = (("TokenIsElevated", wintypes.DWORD),)


class WindowsInputController:
    """Input backends for games that reject one particular Win32 path."""

    TARGET_PROCESS_NAMES = {"petitplanet.exe"}
    INPUT_MOUSE = 0
    INPUT_KEYBOARD = 1
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_SCANCODE = 0x0008
    MAPVK_VK_TO_VSC = 0
    VK_Z = 0x5A
    KEY_HOLD_SECONDS = 0.075
    PRIORITY_KEYS = (
        0x57,  # W
        0x41,  # A
        0x53,  # S
        0x44,  # D
        0x25,  # Left
        0x26,  # Up
        0x27,  # Right
        0x28,  # Down
        0x20,  # Space
        0x10,  # Shift
        0x11,  # Control
        0x12,  # Alt
        0x51,  # Q
        0x45,  # E
        0x52,  # R
        0x46,  # F
    )

    WM_ACTIVATE = 0x0006
    WM_KEYDOWN = 0x0100
    WM_KEYUP = 0x0101
    WM_CHAR = 0x0102
    WM_LBUTTONDOWN = 0x0201
    WM_LBUTTONUP = 0x0202
    MK_LBUTTON = 0x0001
    WA_ACTIVE = 1
    SW_RESTORE = 9
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    TOKEN_QUERY = 0x0008
    TOKEN_INFORMATION_CLASS_ELEVATION = 20
    _cached_user32 = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._emergency_stop = threading.Event()
        self._target_hwnd = wintypes.HWND(0)
        self._target_description = "未绑定"
        self._target_privilege = "未检测"
        self._target_process_name = ""
        self._next_target_probe_at = 0.0
        self.last_result = "未执行"

    @staticmethod
    def _user32():
        if WindowsInputController._cached_user32 is not None:
            return WindowsInputController._cached_user32
        user32 = ctypes.windll.user32
        user32.EnumWindows.argtypes = (
            ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM),
            wintypes.LPARAM,
        )
        user32.EnumWindows.restype = wintypes.BOOL
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.GetAsyncKeyState.argtypes = (wintypes.INT,)
        user32.GetAsyncKeyState.restype = ctypes.c_short
        user32.GetWindowThreadProcessId.argtypes = (
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        )
        user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        user32.IsWindow.argtypes = (wintypes.HWND,)
        user32.IsWindow.restype = wintypes.BOOL
        user32.IsWindowVisible.argtypes = (wintypes.HWND,)
        user32.IsWindowVisible.restype = wintypes.BOOL
        user32.GetWindowTextW.argtypes = (
            wintypes.HWND,
            wintypes.LPWSTR,
            ctypes.c_int,
        )
        user32.GetWindowTextW.restype = ctypes.c_int
        user32.GetClientRect.argtypes = (wintypes.HWND, ctypes.POINTER(_Rect))
        user32.GetClientRect.restype = wintypes.BOOL
        user32.ClientToScreen.argtypes = (
            wintypes.HWND,
            ctypes.POINTER(_Point),
        )
        user32.ClientToScreen.restype = wintypes.BOOL
        user32.ScreenToClient.argtypes = (
            wintypes.HWND,
            ctypes.POINTER(_Point),
        )
        user32.ScreenToClient.restype = wintypes.BOOL
        user32.GetWindowRect.argtypes = (
            wintypes.HWND,
            ctypes.POINTER(_Rect),
        )
        user32.GetWindowRect.restype = wintypes.BOOL
        user32.GetCursorPos.argtypes = (ctypes.POINTER(_Point),)
        user32.GetCursorPos.restype = wintypes.BOOL
        user32.SetCursorPos.argtypes = (ctypes.c_int, ctypes.c_int)
        user32.SetCursorPos.restype = wintypes.BOOL
        user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
        user32.ShowWindow.restype = wintypes.BOOL
        user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
        user32.SetForegroundWindow.restype = wintypes.BOOL
        user32.BringWindowToTop.argtypes = (wintypes.HWND,)
        user32.BringWindowToTop.restype = wintypes.BOOL
        user32.SetFocus.argtypes = (wintypes.HWND,)
        user32.SetFocus.restype = wintypes.HWND
        user32.AttachThreadInput.argtypes = (
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.BOOL,
        )
        user32.AttachThreadInput.restype = wintypes.BOOL
        user32.PostMessageW.argtypes = (
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        )
        user32.PostMessageW.restype = wintypes.BOOL
        user32.MapVirtualKeyW.argtypes = (wintypes.UINT, wintypes.UINT)
        user32.MapVirtualKeyW.restype = wintypes.UINT
        WindowsInputController._cached_user32 = user32
        return user32

    @classmethod
    def _is_target_window(cls, hwnd: wintypes.HWND) -> bool:
        user32 = cls._user32()
        if not hwnd or not user32.IsWindow(hwnd) or not user32.IsWindowVisible(hwnd):
            return False
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return cls._process_name(pid.value) in cls.TARGET_PROCESS_NAMES

    @classmethod
    def _find_target_window(cls) -> wintypes.HWND:
        """Find the game's top-level window even when the assistant is focused."""
        user32 = cls._user32()
        foreground = user32.GetForegroundWindow()
        if cls._is_target_window(foreground):
            return foreground

        candidates: list[wintypes.HWND] = []
        callback_type = ctypes.WINFUNCTYPE(
            wintypes.BOOL,
            wintypes.HWND,
            wintypes.LPARAM,
        )

        @callback_type
        def callback(hwnd, _lparam):
            if cls._is_target_window(hwnd):
                candidates.append(hwnd)
            return True

        user32.EnumWindows(callback, 0)
        if not candidates:
            return wintypes.HWND(0)

        # Prefer a window with a usable client area and a non-empty title.
        def score(hwnd):
            rect = _Rect()
            has_rect = bool(user32.GetClientRect(hwnd, ctypes.byref(rect)))
            area = max(0, int(rect.right)) * max(0, int(rect.bottom))
            title_score = 1 if cls._window_text(hwnd) else 0
            return title_score, area

        return max(candidates, key=score)

    @staticmethod
    def _process_elevated(pid: int) -> bool | None:
        kernel32 = ctypes.windll.kernel32
        advapi32 = ctypes.windll.advapi32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        kernel32.OpenProcess.restype = wintypes.HANDLE
        advapi32.OpenProcessToken.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.HANDLE),
        )
        advapi32.OpenProcessToken.restype = wintypes.BOOL
        advapi32.GetTokenInformation.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        )
        advapi32.GetTokenInformation.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL

        process = (
            kernel32.GetCurrentProcess()
            if pid == os.getpid()
            else kernel32.OpenProcess(
                WindowsInputController.PROCESS_QUERY_LIMITED_INFORMATION,
                False,
                pid,
            )
        )
        if not process:
            return None
        token = wintypes.HANDLE(0)
        try:
            if not advapi32.OpenProcessToken(
                process,
                WindowsInputController.TOKEN_QUERY,
                ctypes.byref(token),
            ):
                return None
            elevation = _TokenElevation()
            returned = wintypes.DWORD()
            if not advapi32.GetTokenInformation(
                token,
                WindowsInputController.TOKEN_INFORMATION_CLASS_ELEVATION,
                ctypes.byref(elevation),
                ctypes.sizeof(elevation),
                ctypes.byref(returned),
            ):
                return None
            return bool(elevation.TokenIsElevated)
        finally:
            if token:
                kernel32.CloseHandle(token)
            if pid != os.getpid():
                kernel32.CloseHandle(process)

    @classmethod
    def _privilege_description(cls, pid: int) -> str:
        target_elevated = cls._process_elevated(pid)
        self_elevated = cls._process_elevated(os.getpid())
        if target_elevated is None or self_elevated is None:
            return "权限未知"
        if target_elevated and not self_elevated:
            return "目标为管理员，本程序非管理员"
        if target_elevated and self_elevated:
            return "管理员权限匹配"
        return "普通权限"

    @staticmethod
    def _send_inputs(inputs: ctypes.Array) -> int:
        user32 = WindowsInputController._user32()
        send_input = user32.SendInput
        send_input.argtypes = (
            wintypes.UINT,
            ctypes.POINTER(_Input),
            ctypes.c_int,
        )
        send_input.restype = wintypes.UINT
        sent = int(
            send_input(
                len(inputs),
                ctypes.cast(inputs, ctypes.POINTER(_Input)),
                ctypes.sizeof(_Input),
            )
        )
        return sent

    @staticmethod
    def _window_text(hwnd: wintypes.HWND) -> str:
        if not hwnd:
            return ""
        buffer = ctypes.create_unicode_buffer(256)
        WindowsInputController._user32().GetWindowTextW(hwnd, buffer, len(buffer))
        return buffer.value.strip()

    @classmethod
    def describe_window(cls, hwnd: wintypes.HWND) -> str:
        if not hwnd:
            return "hwnd=0"
        pid = wintypes.DWORD()
        cls._user32().GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        title = cls._window_text(hwnd) or "<无标题>"
        process_name = cls._process_name(pid.value) or "<未知进程>"
        return (
            f"{process_name} hwnd={int(hwnd)} "
            f"pid={pid.value} title={title!r}"
        )

    @classmethod
    def _process_name(cls, pid: int) -> str:
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.argtypes = (
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.QueryFullProcessImageNameW.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        )
        kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL

        process = kernel32.OpenProcess(
            cls.PROCESS_QUERY_LIMITED_INFORMATION,
            False,
            pid,
        )
        if not process:
            return ""
        try:
            buffer = ctypes.create_unicode_buffer(1024)
            length = wintypes.DWORD(len(buffer))
            if not kernel32.QueryFullProcessImageNameW(
                process,
                0,
                buffer,
                ctypes.byref(length),
            ):
                return ""
            return os.path.basename(buffer.value).lower()
        finally:
            kernel32.CloseHandle(process)

    @property
    def target_description(self) -> str:
        return self._target_description

    @property
    def target_privilege(self) -> str:
        return self._target_privilege

    @property
    def target_is_foreground(self) -> bool:
        user32 = self._user32()
        return bool(
            self._target_hwnd
            and user32.GetForegroundWindow() == self._target_hwnd
            and self._is_target_window(self._target_hwnd)
        )

    def observe_foreground_window(self) -> str | None:
        """Bind a visible top-level window belonging to PetitPlanet.exe."""
        user32 = self._user32()
        if self._target_hwnd and self._is_target_window(self._target_hwnd):
            return None
        self._target_hwnd = wintypes.HWND(0)
        self._target_description = "未绑定"
        self._target_privilege = "未检测"
        self._target_process_name = ""
        now = time.monotonic()
        if now < self._next_target_probe_at:
            return None
        self._next_target_probe_at = now + 0.25
        hwnd = self._find_target_window()
        if not hwnd:
            self.last_result = "未找到 PetitPlanet.exe 可见窗口"
            return None
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value == os.getpid():
            return None
        process_name = self._process_name(pid.value)
        if process_name not in self.TARGET_PROCESS_NAMES:
            self.last_result = (
                f"前台进程 {process_name or '未知'} 不是 "
                f"PetitPlanet.exe，已拒绝绑定"
            )
            return None
        self._target_hwnd = hwnd
        self._target_description = self.describe_window(hwnd)
        self._target_privilege = self._privilege_description(pid.value)
        self._target_process_name = process_name
        return self._target_description

    def clear_emergency_stop(self) -> None:
        self._emergency_stop.clear()

    def trigger_emergency_stop(self) -> None:
        self._emergency_stop.set()

    @property
    def emergency_stopped(self) -> bool:
        return self._emergency_stop.is_set()

    def _resolve_target(self):
        self.observe_foreground_window()
        user32 = self._user32()
        if (
            self._target_hwnd
            and self._is_target_window(self._target_hwnd)
            and user32.GetForegroundWindow() == self._target_hwnd
        ):
            return self._target_hwnd
        self.last_result = "目标游戏窗口未在前台，已拒绝发送"
        foreground = user32.GetForegroundWindow()
        if foreground:
            self.last_result += f" (当前={self.describe_window(foreground)})"
        else:
            self.last_result += " (没有前台窗口)"
        return wintypes.HWND(0)

    @staticmethod
    def _input_mode(mode: str) -> str:
        if mode in {"sendinput_raw_scan", "sendinput_vk"}:
            return mode
        # Keep the existing setting name as the compatible default. This
        # matches common game automation implementations that provide both
        # VK and scan code without forcing KEYEVENTF_SCANCODE.
        return "sendinput_hybrid"

    @classmethod
    def _held_priority_keys(cls) -> list[int]:
        user32 = cls._user32()
        return [
            virtual_key
            for virtual_key in cls.PRIORITY_KEYS
            if user32.GetAsyncKeyState(virtual_key) & 0x8000
        ]

    def _make_key_input(
        self,
        virtual_key: int,
        key_up: bool,
        mode: str,
    ) -> _Input | None:
        user32 = self._user32()
        scan_code = int(
            user32.MapVirtualKeyW(virtual_key, self.MAPVK_VK_TO_VSC)
        )
        if not scan_code:
            return None
        use_raw_scan = mode == "sendinput_raw_scan"
        flags = self.KEYEVENTF_SCANCODE if use_raw_scan else 0
        if key_up:
            flags |= self.KEYEVENTF_KEYUP
        return _Input(
            type=self.INPUT_KEYBOARD,
            u=_InputUnion(
                ki=_KeyboardInput(
                    wVk=0 if use_raw_scan else virtual_key,
                    wScan=scan_code,
                    dwFlags=flags,
                    time=0,
                    dwExtraInfo=0,
                )
            ),
        )

    def _sendinput_key(self, mode: str, priority: bool = False) -> bool:
        hwnd = self._resolve_target()
        activated = self._activate_target(hwnd)
        if not hwnd:
            return False
        if not activated:
            return False
        user32 = self._user32()
        scan_code = int(user32.MapVirtualKeyW(self.VK_Z, self.MAPVK_VK_TO_VSC))
        if not scan_code:
            self.last_result = "MapVirtualKeyW 未返回扫描码"
            return False
        mode = self._input_mode(mode)
        held_keys = self._held_priority_keys() if priority else []
        released = 0
        restored = 0
        if held_keys:
            release_inputs = [
                self._make_key_input(virtual_key, True, mode)
                for virtual_key in held_keys
            ]
            release_inputs = [item for item in release_inputs if item is not None]
            if release_inputs:
                release_array = (_Input * len(release_inputs))(*release_inputs)
                released = self._send_inputs(release_array)

        key_down = self._make_key_input(self.VK_Z, False, mode)
        key_up = self._make_key_input(self.VK_Z, True, mode)
        if key_down is None or key_up is None:
            self.last_result = "MapVirtualKeyW 未返回按键扫描码"
            return False
        key_down_array = (_Input * 1)(key_down)
        key_up_array = (_Input * 1)(key_up)
        down_sent = self._send_inputs(key_down_array)
        time.sleep(self.KEY_HOLD_SECONDS)
        up_sent = self._send_inputs(key_up_array)
        if held_keys:
            restore_inputs = [
                self._make_key_input(virtual_key, False, mode)
                for virtual_key in held_keys
            ]
            restore_inputs = [item for item in restore_inputs if item is not None]
            if restore_inputs:
                restore_array = (_Input * len(restore_inputs))(*restore_inputs)
                restored = self._send_inputs(restore_array)
        held_text = (
            ",".join(f"0x{key:02X}" for key in held_keys)
            if priority
            else "not_checked"
        )
        self.last_result = (
            f"SendInput mode={mode} down={down_sent}/1 up={up_sent}/1 "
            f"priority={int(priority)} held={held_text} "
            f"released={released}/{len(held_keys)} restored={restored}/{len(held_keys)} "
            f"activated={int(activated)} {self._target_privilege} "
            f"{self.describe_window(hwnd)}"
        )
        time.sleep(0.035)
        return down_sent == 1 and up_sent == 1

    def _activate_target(self, hwnd) -> bool:
        if not hwnd:
            self.last_result = "没有目标窗口"
            return False
        user32 = self._user32()
        if user32.GetForegroundWindow() == hwnd:
            return True
        self.last_result = (
            f"目标未在前台，已拒绝输入 "
            f"{self._target_privilege} {self.describe_window(hwnd)}"
        )
        return False

    def _post_key(self) -> bool:
        hwnd = self._resolve_target()
        if not hwnd:
            self.last_result = "没有目标窗口"
            return False
        user32 = self._user32()
        scan_code = int(user32.MapVirtualKeyW(self.VK_Z, self.MAPVK_VK_TO_VSC))
        down_lparam = 1 | (scan_code << 16)
        up_lparam = down_lparam | (1 << 30) | (1 << 31)
        user32.PostMessageW(hwnd, self.WM_ACTIVATE, self.WA_ACTIVE, 0)
        time.sleep(0.015)
        results = (
            bool(user32.PostMessageW(hwnd, self.WM_KEYDOWN, self.VK_Z, down_lparam)),
            bool(user32.PostMessageW(hwnd, self.WM_KEYUP, self.VK_Z, up_lparam)),
        )
        self.last_result = (
            f"PostMessage key={sum(results)}/2 "
            f"{self._target_privilege} {self.describe_window(hwnd)}"
        )
        time.sleep(0.035)
        return all(results)

    def press_z(self, method: str = "postmessage", priority: bool = False) -> bool:
        with self._lock:
            if self.emergency_stopped:
                self.last_result = "紧急停止已锁定"
                return False
            if method in {
                "sendinput_scan",
                "sendinput_hybrid",
                "sendinput_raw_scan",
                "sendinput_vk",
            }:
                return self._sendinput_key(method, priority=priority)
            return self._post_key()

    def _post_click(self) -> bool:
        hwnd = self._resolve_target()
        if not hwnd:
            self.last_result = "没有目标窗口"
            return False
        user32 = self._user32()
        user32.PostMessageW(hwnd, self.WM_ACTIVATE, self.WA_ACTIVE, 0)
        time.sleep(0.015)
        point = _Point()
        if not user32.GetCursorPos(ctypes.byref(point)):
            point = _Point(0, 0)
        if not user32.ScreenToClient(hwnd, ctypes.byref(point)):
            point = _Point(0, 0)
        client = _Rect()
        if user32.GetClientRect(hwnd, ctypes.byref(client)):
            if 0 <= point.x < client.right and 0 <= point.y < client.bottom:
                x, y = point.x, point.y
            else:
                x = max(0, (client.right - client.left) // 2)
                y = max(0, (client.bottom - client.top) // 2)
        else:
            x, y = 0, 0
        lparam = (y << 16) | (x & 0xFFFF)
        down = bool(
            user32.PostMessageW(hwnd, self.WM_LBUTTONDOWN, self.MK_LBUTTON, lparam)
        )
        time.sleep(0.06)
        up = bool(user32.PostMessageW(hwnd, self.WM_LBUTTONUP, 0, lparam))
        self.last_result = (
            f"PostMessage click={int(down) + int(up)}/2 "
            f"{self._target_privilege} {self.describe_window(hwnd)}"
        )
        return down and up

    def _sendinput_click(self) -> bool:
        hwnd = self._resolve_target()
        activated = self._activate_target(hwnd)
        if not hwnd:
            return False
        if not activated:
            return False
        mouse_down = (_Input * 1)(
            _Input(type=self.INPUT_MOUSE, u=_InputUnion())
        )
        mouse_up = (_Input * 1)(
            _Input(type=self.INPUT_MOUSE, u=_InputUnion())
        )
        mouse_down[0].mi.dwFlags = self.MOUSEEVENTF_LEFTDOWN
        mouse_up[0].mi.dwFlags = self.MOUSEEVENTF_LEFTUP
        down_sent = self._send_inputs(mouse_down)
        time.sleep(0.06)
        up_sent = self._send_inputs(mouse_up)
        self.last_result = (
            f"SendInput mouse down={down_sent}/1 up={up_sent}/1 "
            f"activated={int(activated)} "
            f"{self._target_privilege} {self.describe_window(hwnd)}"
        )
        return down_sent == 1 and up_sent == 1

    def left_click(self, method: str = "postmessage") -> bool:
        with self._lock:
            if self.emergency_stopped:
                self.last_result = "紧急停止已锁定"
                return False
            if method == "postmessage":
                return self._post_click()
            return self._sendinput_click()
