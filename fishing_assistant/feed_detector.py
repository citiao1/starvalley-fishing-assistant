from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


BASE_WIDTH = 2560
BASE_HEIGHT = 1440
DIGIT_ROI = (2078, 1168, 2140, 1230)
CANONICAL_SIZE = (48, 64)


def read_image(path: Path) -> np.ndarray | None:
    """Read images from Windows paths that may contain non-ASCII characters."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def scaled_roi(width: int, height: int) -> tuple[int, int, int, int]:
    sx = width / BASE_WIDTH
    sy = height / BASE_HEIGHT
    x1, y1, x2, y2 = DIGIT_ROI
    return round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy)


def red_mask(image: np.ndarray) -> np.ndarray:
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
    return (
        (labels[y : y + h, x : x + w] == component_index).astype(np.uint8) * 255
    )


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
    component = largest_component(red_mask(image[y1:y2, x1:x2]))
    return canonicalize(component) if component is not None else None


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_bin = a > 0
    b_bin = b > 0
    union = np.logical_or(a_bin, b_bin).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a_bin, b_bin).sum() / union)


def has_enclosed_hole(component: np.ndarray) -> bool:
    """Return whether the red component contains a closed interior hole."""
    contours, hierarchy = cv2.findContours(
        component,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if hierarchy is None:
        return False
    return any(int(item[3]) >= 0 for item in hierarchy[0])


class FeedZeroDetector:
    def __init__(
        self,
        zero_template: np.ndarray,
        nonzero_template: np.ndarray,
        min_zero_score: float = 0.68,
        min_margin: float = 0.20,
    ):
        self.zero_template = zero_template
        self.nonzero_template = nonzero_template
        self.min_zero_score = min_zero_score
        self.min_margin = min_margin

    @classmethod
    def from_images(cls, zero_path: Path, nonzero_path: Path) -> "FeedZeroDetector":
        zero_image = read_image(zero_path)
        nonzero_image = read_image(nonzero_path)
        if zero_image is None:
            raise FileNotFoundError(zero_path)
        if nonzero_image is None:
            raise FileNotFoundError(nonzero_path)
        zero_template = make_template(zero_image)
        nonzero_template = make_template(nonzero_image)
        if zero_template is None or nonzero_template is None:
            raise RuntimeError("无法从模板中分离红色数字")
        return cls(zero_template, nonzero_template)

    def classify(self, image: np.ndarray) -> tuple[str, float, float]:
        x1, y1, x2, y2 = scaled_roi(image.shape[1], image.shape[0])
        return self.classify_roi(image[y1:y2, x1:x2])

    def classify_roi(self, roi: np.ndarray) -> tuple[str, float, float]:
        """Classify only the fixed stamina-digit crop."""
        component = largest_component(red_mask(roi))
        if component is None:
            return "not_found", 0.0, 0.0
        current = canonicalize(component)
        zero_score = similarity(current, self.zero_template)
        nonzero_score = similarity(current, self.nonzero_template)
        if not has_enclosed_hole(component):
            return "nonzero", zero_score, nonzero_score
        # A red component alone is not enough: other red UI text can have a
        # superficial overlap with the zero template. Require both a usable
        # absolute score and a clear margin over the non-zero template.
        if (
            zero_score >= self.min_zero_score
            and zero_score - nonzero_score >= self.min_margin
        ):
            return "zero", zero_score, nonzero_score
        return "nonzero", zero_score, nonzero_score
