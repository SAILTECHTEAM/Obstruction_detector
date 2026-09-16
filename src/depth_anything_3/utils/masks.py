"""Shared mask operations used by person and depth-occlusion analysis."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_yolo_seg_class_mask(
    txt_path: Path | None, target_shape: tuple[int, int], class_id: int
) -> np.ndarray:
    """Rasterize one class from a normalized YOLO segmentation TXT file."""
    height, width = target_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    if txt_path is None:
        return mask
    if not txt_path.is_file():
        raise FileNotFoundError(f"YOLO segmentation TXT not found: {txt_path}")
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        values = line.strip().split()
        if len(values) < 7 or int(float(values[0])) != class_id:
            continue
        coordinates = [float(value) for value in values[1:]]
        if len(coordinates) % 2 == 1:
            coordinates = coordinates[:-1]
        if len(coordinates) < 6:
            continue
        polygon = np.asarray(coordinates, dtype=np.float32).reshape(-1, 2)
        polygon[:, 0] = np.clip(polygon[:, 0] * width, 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1] * height, 0, height - 1)
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255)
    return mask


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask, kernel, iterations=1)


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.erode(
        mask,
        kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def clean_mask(mask: np.ndarray, open_size: int, close_size: int) -> np.ndarray:
    """Apply optional opening and closing to a boolean or uint8 mask."""
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    if open_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    if close_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return mask_u8


def mask_region(mask: np.ndarray, area_ratio: float) -> dict[str, int | float]:
    """Return one bounding region enclosing all nonzero pixels in a mask."""
    points = cv2.findNonZero(mask)
    if points is None:
        raise ValueError("Cannot create a region from an empty mask")
    x, y, width, height = cv2.boundingRect(points)
    return {
        "x": int(x), "y": int(y), "width": int(width), "height": int(height),
        "area_pixels": int(np.count_nonzero(mask)), "area_ratio": float(area_ratio), "is_person": 1,
    }


def find_large_regions(
    mask_u8: np.ndarray, min_area_ratio: float
) -> tuple[list[dict[str, int | float]], np.ndarray]:
    """Find connected mask components at or above a fraction of image area."""
    height, width = mask_u8.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    regions: list[dict[str, int | float]] = []
    large_mask = np.zeros_like(mask_u8)
    for label in range(1, count):
        x, y, box_width, box_height, area = stats[label]
        area_ratio = float(area / (height * width))
        if area_ratio < min_area_ratio:
            continue
        large_mask[labels == label] = 255
        regions.append({
            "x": int(x), "y": int(y), "width": int(box_width), "height": int(box_height),
            "area_pixels": int(area), "area_ratio": area_ratio,
        })
    regions.sort(key=lambda region: float(region["area_ratio"]), reverse=True)
    return regions, large_mask


def validate_odd_size(name: str, value: int) -> None:
    if value < 1 or value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer, got {value}")
