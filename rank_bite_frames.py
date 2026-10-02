from __future__ import annotations

import csv
import math
import re
import shutil
from collections import deque
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "dataset" / "images" / "bite_candidates"
OUTPUT = ROOT / "dataset_v3" / "bite_ranked"
CSV_PATH = ROOT / "dataset_v3" / "bite_scores.csv"


def yellow_pixel(rgb: tuple[int, int, int]) -> bool:
    r, g, b = rgb
    return r > 145 and g > 85 and b < 150 and r > g * 1.02 and g > b * 1.15


def components(image: Image.Image) -> list[tuple[int, int, int, int, int]]:
    image = image.resize((image.width // 2, image.height // 2), Image.Resampling.BILINEAR)
    width, height = image.size
    pixels = image.load()
    x0, x1 = 35, width - 35
    y0, y1 = 55, height - 70
    seen: set[tuple[int, int]] = set()
    result = []

    for y in range(y0, y1):
        for x in range(x0, x1):
            if (x, y) in seen or not yellow_pixel(pixels[x, y]):
                continue

            stack = [(x, y)]
            seen.add((x, y))
            points = []

            while stack:
                px, py = stack.pop()
                points.append((px, py))
                for nx, ny in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
                    if nx < x0 or nx >= x1 or ny < y0 or ny >= y1:
                        continue
                    if (nx, ny) in seen or not yellow_pixel(pixels[nx, ny]):
                        continue
                    seen.add((nx, ny))
                    stack.append((nx, ny))

            if len(points) < 15:
                continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            left, right = min(xs), max(xs)
            top, bottom = min(ys), max(ys)
            result.append((len(points), left, top, right, bottom))

    return result


def score(path: Path) -> tuple[float, tuple[int, int, int, int, int] | None]:
    with Image.open(path) as image:
        candidates = components(image)

    best_score = 0.0
    best_component = None
    for area, left, top, right, bottom in candidates:
        width = right - left + 1
        height = bottom - top + 1
        if width < 12 or height < 12 or width > 90 or height > 90:
            continue

        aspect = min(width, height) / max(width, height)
        fill = area / (width * height)
        size_bonus = max(0.0, 1.0 - abs(math.log(max(area, 1) / 500.0)) / 4.0)
        component_score = 4.0 * aspect + 3.0 * fill + 2.0 * size_bonus

        if component_score > best_score:
            best_score = component_score
            best_component = (area, left, top, right, bottom)

    return best_score, best_component


def group_key(path: Path) -> str:
    return re.sub(r"\.\d+$", "", path.stem)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[Path]] = {}
    for path in sorted(SOURCE.glob("bite_*.jpg")):
        groups.setdefault(group_key(path), []).append(path)

    rows = []
    for key, paths in groups.items():
        ranked = []
        for path in paths:
            value, component = score(path)
            ranked.append((value, path, component))

        ranked.sort(key=lambda item: item[0], reverse=True)
        for rank, (value, path, component) in enumerate(ranked[:10], start=1):
            target = OUTPUT / f"{key}_rank{rank:02d}{path.suffix}"
            shutil.copy2(path, target)
            rows.append(
                {
                    "event": key,
                    "rank": rank,
                    "score": f"{value:.4f}",
                    "source": str(path),
                    "output": str(target),
                    "component": component or "",
                }
            )

    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["event", "rank", "score", "source", "output", "component"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"events={len(groups)}")
    print(f"selected={len(rows)}")
    print(f"output={OUTPUT}")
    print(f"scores={CSV_PATH}")


if __name__ == "__main__":
    main()
