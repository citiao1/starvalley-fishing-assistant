from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# Coordinates are defined for the 2560x1440 game capture and scaled for
# other capture sizes with the same aspect ratio.
BASE_WIDTH = 2560
BASE_HEIGHT = 1440
DIGIT_ROI = (2078, 1168, 2140, 1230)
CANONICAL_SIZE = (48, 64)


def scaled_roi(width: int, height: int) -> tuple[int, int, int, int]:
    sx = width / BASE_WIDTH
    sy = height / BASE_HEIGHT
    x1, y1, x2, y2 = DIGIT_ROI
    return (
        round(x1 * sx),
        round(y1 * sy),
        round(x2 * sx),
        round(y2 * sy),
    )


def red_mask(image: np.ndarray) -> np.ndarray:
    """Keep the orange-red digit while rejecting the grey UI background."""
    b, g, r = cv2.split(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    warm_red = (
        (r > 100)
        & (r > g.astype(np.int16) * 1.28)
        & (r > b.astype(np.int16) * 1.35)
        & (g < 190)
    )
    orange_red = ((h <= 25) | (h >= 170)) & (s > 55) & (v > 70)
    mask = (warm_red & orange_red).astype(np.uint8) * 255

    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def largest_component(mask: np.ndarray) -> Optional[np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None

    component_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = stats[component_index, cv2.CC_STAT_LEFT]
    y = stats[component_index, cv2.CC_STAT_TOP]
    w = stats[component_index, cv2.CC_STAT_WIDTH]
    h = stats[component_index, cv2.CC_STAT_HEIGHT]
    area = stats[component_index, cv2.CC_STAT_AREA]
    if area < 20 or w < 5 or h < 8:
        return None

    component = (labels[y : y + h, x : x + w] == component_index).astype(
        np.uint8
    ) * 255
    return component


def canonicalize(component: np.ndarray) -> np.ndarray:
    target_w, target_h = CANONICAL_SIZE
    h, w = component.shape[:2]
    scale = min((target_w - 8) / w, (target_h - 8) / h)
    resized = cv2.resize(
        component,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = np.zeros((target_h, target_w), np.uint8)
    x = (target_w - resized.shape[1]) // 2
    y = (target_h - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def make_template(image: np.ndarray) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = scaled_roi(image.shape[1], image.shape[0])
    roi = image[y1:y2, x1:x2]
    component = largest_component(red_mask(roi))
    return canonicalize(component) if component is not None else None


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_bin = a > 0
    b_bin = b > 0
    union = np.logical_or(a_bin, b_bin).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a_bin, b_bin).sum() / union)


class FeedZeroDetector:
    def __init__(self, zero_template: np.ndarray, nonzero_template: np.ndarray):
        self.zero_template = zero_template
        self.nonzero_template = nonzero_template

    @classmethod
    def from_images(cls, zero_path: Path, nonzero_path: Path) -> "FeedZeroDetector":
        zero_image = cv2.imread(str(zero_path))
        nonzero_image = cv2.imread(str(nonzero_path))
        if zero_image is None:
            raise FileNotFoundError(zero_path)
        if nonzero_image is None:
            raise FileNotFoundError(nonzero_path)

        zero_template = make_template(zero_image)
        nonzero_template = make_template(nonzero_image)
        if zero_template is None or nonzero_template is None:
            raise RuntimeError("Could not isolate the red digit in one of the templates")
        return cls(zero_template, nonzero_template)

    def classify(self, image: np.ndarray) -> tuple[str, float, float]:
        x1, y1, x2, y2 = scaled_roi(image.shape[1], image.shape[0])
        roi = image[y1:y2, x1:x2]
        component = largest_component(red_mask(roi))
        if component is None:
            return "not_found", 0.0, 0.0

        current = canonicalize(component)
        zero_score = similarity(current, self.zero_template)
        nonzero_score = similarity(current, self.nonzero_template)
        if zero_score >= 0.55 and zero_score > nonzero_score + 0.04:
            return "zero", zero_score, nonzero_score
        return "nonzero", zero_score, nonzero_score


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect the fixed red stamina digit.")
    parser.add_argument("--zero", required=True, type=Path, help="Screenshot containing red 0")
    parser.add_argument("--nonzero", required=True, type=Path, help="Screenshot containing red 2/other digit")
    parser.add_argument("--image", required=True, type=Path, nargs="+", help="Images to classify")
    args = parser.parse_args()

    detector = FeedZeroDetector.from_images(args.zero, args.nonzero)
    for path in args.image:
        image = cv2.imread(str(path))
        if image is None:
            print(f"{path}: unreadable")
            continue
        label, zero_score, nonzero_score = detector.classify(image)
        print(
            f"{path}: {label} "
            f"(zero={zero_score:.3f}, nonzero={nonzero_score:.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
