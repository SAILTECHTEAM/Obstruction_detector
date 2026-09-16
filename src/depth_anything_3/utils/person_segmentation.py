"""Optional Ultralytics YOLO segmentation helpers.

Ultralytics is imported lazily so the rest of Depth Anything 3 continues to
work when that optional dependency is not installed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class UltralyticsPersonSegmenter:
    """Run YOLO instance segmentation and return a union person mask."""

    def __init__(
        self,
        model: str | Path,
        *,
        person_class_id: int = 0,
        confidence: float = 0.25,
        device: str | None = None,
        image_size: int = 640,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required when --yolo-model is used. "
                "Install it in the active virtual environment first."
            ) from exc

        if not 0 <= confidence <= 1:
            raise ValueError("YOLO confidence must be between 0 and 1")
        if image_size <= 0:
            raise ValueError("YOLO image size must be positive")

        self.model = YOLO(str(model))
        self.person_class_id = person_class_id
        self.confidence = confidence
        self.device = device
        self.image_size = image_size

    def predict_mask(
        self,
        image_bgr: np.ndarray,
        target_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Return all detected people as one uint8 mask."""
        height, width = target_shape or image_bgr.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        for instance_mask in self.predict_instance_masks(image_bgr, target_shape):
            mask[instance_mask > 0] = 255
        return mask

    def predict_instance_masks(
        self,
        image_bgr: np.ndarray,
        target_shape: tuple[int, int] | None = None,
    ) -> list[np.ndarray]:
        """Return one uint8 mask per detected person instance.

        ``image_bgr`` follows OpenCV channel order. YOLO's normalized polygons
        are rasterized directly at ``target_shape``, avoiding an intermediate
        resize from the original video resolution to the depth-map resolution.
        """
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"Expected an HxWx3 BGR image, got {image_bgr.shape}")

        if target_shape is None:
            height, width = image_bgr.shape[:2]
        else:
            height, width = target_shape
        masks: list[np.ndarray] = []

        predict_kwargs: dict[str, object] = {
            "source": image_bgr,
            "classes": [self.person_class_id],
            "conf": self.confidence,
            "imgsz": self.image_size,
            "verbose": False,
            "retina_masks": True,
        }
        if self.device:
            predict_kwargs["device"] = self.device

        results = self.model.predict(**predict_kwargs)
        if not results or results[0].masks is None:
            return masks

        for polygon in results[0].masks.xyn:
            points = np.asarray(polygon, dtype=np.float32)
            if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
                continue
            points[:, 0] = np.clip(points[:, 0] * width, 0, width - 1)
            points[:, 1] = np.clip(points[:, 1] * height, 0, height - 1)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
            masks.append(mask)
        return masks
