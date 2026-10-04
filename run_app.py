import ctypes
import os
import sys
import traceback
from pathlib import Path


def _prepare_dll_search_path() -> None:
    if not getattr(sys, "frozen", False):
        return
    internal = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    search_roots = (
        internal,
        internal / "PySide6",
        internal / "shiboken6",
        internal / "torch",
        internal / "cv2",
    )
    handles = []
    for root in search_roots:
        if root.exists() and hasattr(os, "add_dll_directory"):
            handles.append(os.add_dll_directory(str(root)))
    globals()["_dll_directory_handles"] = handles

    # Qt6Core depends on ICU DLLs that PyInstaller places beside the
    # PySide6 package. Load those dependencies by absolute path first, then
    # load the binding DLLs with the normal secure search mode.
    preload_paths = (
        *sorted(internal.glob("icudt*.dll")),
        *sorted(internal.glob("icuuc*.dll")),
        internal / "shiboken6" / "shiboken6.abi3.dll",
        internal / "PySide6" / "Qt6Core.dll",
        internal / "PySide6" / "pyside6.abi3.dll",
    )
    preload_handles = []
    preload_errors = []
    for path in preload_paths:
        if not path.exists():
            continue
        try:
            preload_handles.append(ctypes.WinDLL(str(path)))
        except OSError as exc:
            preload_errors.append(
                f"{path.name}: winerror={exc.winerror!r}, {exc}"
            )
    globals()["_preload_dll_handles"] = preload_handles
    if preload_errors:
        data_dir = Path(sys.executable).resolve().parent / "app_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "qt_dll_preload_error.log").write_text(
            "\n".join(preload_errors),
            encoding="utf-8",
        )


_prepare_dll_search_path()
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _write_startup_error(exc: BaseException) -> None:
    root = (
        Path(sys.executable).resolve().parent
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parent
    )
    data_dir = root / "app_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "startup_error.log").write_text(
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        encoding="utf-8",
    )


try:
    from fishing_assistant.ui import run
except BaseException as exc:
    _write_startup_error(exc)
    raise


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except SystemExit:
        raise
    except BaseException as exc:
        _write_startup_error(exc)
        raise
