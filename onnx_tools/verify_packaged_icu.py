"""Reject incompatible ICU DLLs collected from unrelated PATH entries."""

import sys
from pathlib import Path

import pefile


def verify(release_dir: Path) -> None:
    internal = release_dir / "_internal"
    with pefile.PE(str(internal / "PySide6" / "Qt6Core.dll")) as qt:
        imports = {
            entry.dll.decode("ascii"): {
                symbol.name for symbol in entry.imports if symbol.name
            }
            for entry in qt.DIRECTORY_ENTRY_IMPORT
            if entry.dll.lower().startswith(b"icu")
        }
    for name, required in imports.items():
        candidates = list(internal.rglob(name))
        if not candidates:
            # Windows supplies ICU on supported systems; do not redistribute
            # an unrelated implementation just because its DLL name matches.
            import os

            candidates = [Path(os.environ["SystemRoot"]) / "System32" / name]
        for candidate in candidates:
            with pefile.PE(str(candidate)) as dll:
                exported = {
                    symbol.name for symbol in dll.DIRECTORY_ENTRY_EXPORT.symbols
                }
            missing = required - exported
            if missing:
                raise RuntimeError(f"{candidate}: missing exports {sorted(missing)!r}")
            print(f"ICU imports verified: {candidate}")


if __name__ == "__main__":
    verify(Path(sys.argv[1]))
