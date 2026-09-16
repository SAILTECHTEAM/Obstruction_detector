"""Shared loading, inference, and sampling helpers for depth-analysis scripts."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from depth_anything_3.api import DepthAnything3


def load_bgr_image(path: Path) -> np.ndarray:
    """Load an image in OpenCV BGR order."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def load_depth_map(path: Path, index: int = 0) -> np.ndarray:
    """Load one HxW depth map from ``.npy`` or DA3 ``results.npz`` output."""
    if not path.is_file():
        raise FileNotFoundError(f"Depth result not found: {path}")
    if path.suffix.lower() == ".npy":
        depth = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "depth" not in data:
                raise KeyError(f"{path} does not contain a 'depth' array")
            depth = np.asarray(data["depth"], dtype=np.float32)
    else:
        raise ValueError(f"Only .npy and .npz depth maps are supported: {path}")

    if depth.ndim == 2:
        return depth
    if depth.ndim == 3 and 0 <= index < depth.shape[0]:
        return depth[index]
    if depth.ndim == 3:
        raise IndexError(f"Index {index} is out of range; shape={depth.shape}")
    raise ValueError(f"Expected (H, W) or (N, H, W), got {depth.shape}")


def validate_inference_device(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; pass --device cpu")


def load_depth_model(model_id: str, device: str) -> DepthAnything3:
    """Load DA3 once for one or more inference calls."""
    validate_inference_device(device)
    print(f"Loading model: {model_id}")
    return DepthAnything3.from_pretrained(model_id).to(device).eval()


def infer_depth_from_path(
    model: DepthAnything3, path: Path, process_res: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Infer depth from an image path and return its optional intrinsics."""
    print(f"Inferring depth: {path}")
    prediction = model.inference(
        [str(path)],
        process_res=process_res,
        process_res_method="upper_bound_resize",
        export_dir=None,
    )
    depth = np.asarray(prediction.depth[0], dtype=np.float32)
    intrinsics = None if prediction.intrinsics is None else prediction.intrinsics[0]
    return depth, intrinsics


def infer_depth_from_bgr(
    model: DepthAnything3, image_bgr: np.ndarray, process_res: int
) -> np.ndarray:
    """Infer depth from an OpenCV BGR frame."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    prediction = model.inference(
        [image_rgb],
        process_res=process_res,
        process_res_method="upper_bound_resize",
        export_dir=None,
        include_processed_images=False,
    )
    return np.asarray(prediction.depth[0], dtype=np.float32).copy()


def infer_depth_pair(
    first_path: Path,
    second_path: Path,
    model_id: str,
    device: str,
    process_res: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run two independent depth inferences using one loaded DA3 model."""
    model = load_depth_model(model_id, device)
    first_depth, _ = infer_depth_from_path(model, first_path, process_res)
    second_depth, _ = infer_depth_from_path(model, second_path, process_res)
    return first_depth, second_depth


def valid_depth_mask(*depth_maps: np.ndarray) -> np.ndarray:
    """Return pixels that are finite and positive in every supplied depth map."""
    if not depth_maps:
        raise ValueError("At least one depth map is required")
    valid = np.ones(depth_maps[0].shape, dtype=bool)
    for depth in depth_maps:
        if depth.shape != valid.shape:
            raise ValueError("Depth maps must have identical shapes")
        valid &= np.isfinite(depth) & (depth > 0)
    return valid


def depth_difference(first_depth: np.ndarray, second_depth: np.ndarray) -> np.ndarray:
    """Return ``second - first``, using NaN where either map is invalid."""
    valid = valid_depth_mask(first_depth, second_depth)
    difference = np.full(first_depth.shape, np.nan, dtype=np.float32)
    difference[valid] = second_depth[valid] - first_depth[valid]
    return difference


def finite_percentiles(*depth_maps: np.ndarray) -> tuple[float, float]:
    """Return robust shared display limits for one or more depth maps."""
    values = np.concatenate([depth[np.isfinite(depth)] for depth in depth_maps])
    if values.size == 0:
        raise ValueError("The depth maps contain no finite values")
    low, high = np.percentile(values, (2, 98))
    if low == high:
        high = low + 1e-6
    return float(low), float(high)


def robust_patch_depth(
    depth: np.ndarray, center: tuple[float, float], patch_size: int
) -> float:
    """Return the median positive depth in an image-clipped square patch."""
    if patch_size < 1 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer")
    height, width = depth.shape
    center_x, center_y = (int(round(value)) for value in center)
    radius = patch_size // 2
    values = depth[
        max(0, center_y - radius) : min(height, center_y + radius + 1),
        max(0, center_x - radius) : min(width, center_x + radius + 1),
    ]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        raise ValueError(f"No finite positive depth values in {patch_size}x{patch_size} patch")
    return float(np.median(values))
