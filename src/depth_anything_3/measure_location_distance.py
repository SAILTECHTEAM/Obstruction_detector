"""Measure the distance between two image locations using Depth Anything 3.

The script always reports ``depth_difference``: the difference between the two
robust metric-depth estimates.  To calculate lateral distance and full
``3d_distance``, pass calibrated camera intrinsics with ``--intrinsics`` or
``--intrinsics-npy``.  Supplying ``--distortion-npy`` undistorts the image and
the target centres before depth inference and back-projection.

Example:
    uv run python -m depth_anything_3.measure_location_distance image.jpg \
        --bbox-a 120 80 260 420 --bbox-b 380 100 540 440

    uv run python -m depth_anything_3.measure_location_distance image.jpg \
        --point-a 190 250 --point-b 460 270
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from depth_anything_3.utils.depth_analysis import (
    infer_depth_from_path,
    load_depth_model,
    robust_patch_depth,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input RGB image")
    target_a = parser.add_mutually_exclusive_group(required=True)
    target_a.add_argument(
        "--bbox-a",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="First bounding box in image pixels (top-left then bottom-right)",
    )
    target_a.add_argument(
        "--point-a",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="First image-space point in pixels",
    )
    target_b = parser.add_mutually_exclusive_group(required=True)
    target_b.add_argument(
        "--bbox-b",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Second bounding box in image pixels (top-left then bottom-right)",
    )
    target_b.add_argument(
        "--point-b",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Second image-space point in pixels",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=15,
        help="Odd side length, in depth-map pixels, of the centre patch (default: 15)",
    )
    parser.add_argument(
        "--model-dir",
        default="depth-anything/DA3METRIC-LARGE",
        help="Depth Anything 3 model ID or local model directory",
    )
    parser.add_argument("--device", default="cuda", help="Inference device")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument(
        "--intrinsics",
        nargs=4,
        type=float,
        metavar=("FX", "FY", "CX", "CY"),
        help=(
            "Calibrated input-image camera intrinsics in pixels. Required for "
            "full 3-D distance when the model does not infer intrinsics."
        ),
    )
    parser.add_argument(
        "--intrinsics-npy",
        type=Path,
        help="Path to a calibrated 3x3 camera-matrix .npy file",
    )
    parser.add_argument(
        "--distortion-npy",
        type=Path,
        help="Path to OpenCV distortion-coefficients .npy file; requires --intrinsics-npy or --intrinsics",
    )
    return parser.parse_args()


def validate_bbox(box: list[float], name: str) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    if not x1 < x2 or not y1 < y2:
        raise ValueError(f"{name} must satisfy X1 < X2 and Y1 < Y2, got {box}")
    return x1, y1, x2, y2


def image_point_to_depth_point(
    point: tuple[float, float],
    image_width: int,
    image_height: int,
    depth_width: int,
    depth_height: int,
) -> tuple[float, float]:
    """Map an input-image point to depth-map coordinates."""
    point_x, point_y = point
    if not (0 <= point_x < image_width and 0 <= point_y < image_height):
        raise ValueError(
            f"Point ({point_x:.1f}, {point_y:.1f}) is outside "
            f"the {image_width}x{image_height} image"
        )
    return point_x * depth_width / image_width, point_y * depth_height / image_height


def target_to_depth_point(
    bbox: list[float] | None,
    point: list[float] | None,
    name: str,
    image_width: int,
    image_height: int,
    depth_width: int,
    depth_height: int,
) -> tuple[tuple[float, float], dict[str, object]]:
    """Convert one CLI target to a depth-map point and a JSON description."""
    if bbox is not None:
        x1, y1, x2, y2 = validate_bbox(bbox, name)
        image_point = ((x1 + x2) / 2, (y1 + y2) / 2)
        description: dict[str, object] = {
            "type": "bbox",
            "coordinates": [x1, y1, x2, y2],
        }
    elif point is not None:
        image_point = (float(point[0]), float(point[1]))
        description = {"type": "point", "coordinates": list(image_point)}
    else:  # argparse's mutually exclusive group guarantees this cannot occur.
        raise ValueError(f"No target was supplied for {name}")

    depth_point = image_point_to_depth_point(
        image_point, image_width, image_height, depth_width, depth_height
    )
    description["image_point"] = list(image_point)
    return depth_point, description


def camera_point(
    center: tuple[float, float], depth: float, intrinsics: np.ndarray | None
) -> np.ndarray:
    """Back-project a depth-map pixel into the camera coordinate system."""
    if intrinsics is None:
        raise ValueError(
            "Model did not return intrinsics; cannot calculate 3-D distance"
        )
    intrinsics = np.asarray(intrinsics, dtype=np.float64).copy()
    if intrinsics.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 intrinsics matrix, got {intrinsics.shape}")

    # DA3's inferred intrinsics use the same processed-image coordinates as
    # the returned depth map, so ``center`` has already been mapped correctly.
    x, y = center
    ray = np.linalg.inv(intrinsics) @ np.array([x, y, 1.0])
    return ray * depth


def resolve_intrinsics(
    model_intrinsics: np.ndarray | None,
    supplied_intrinsics: list[float] | None,
    image_width: int,
    image_height: int,
    depth_width: int,
    depth_height: int,
) -> tuple[np.ndarray | None, str | None]:
    """Return intrinsics in depth-map coordinates and describe their source."""
    if supplied_intrinsics is not None:
        fx, fy, cx, cy = supplied_intrinsics
        if fx <= 0 or fy <= 0:
            raise ValueError("--intrinsics FX and FY must be positive")
        scale_x = depth_width / image_width
        scale_y = depth_height / image_height
        return (
            np.array(
                [
                    [fx * scale_x, 0.0, cx * scale_x],
                    [0.0, fy * scale_y, cy * scale_y],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            "user_supplied",
        )
    if model_intrinsics is not None:
        return np.asarray(model_intrinsics, dtype=np.float64), "model_inferred"
    return None, None


def load_intrinsics_npy(path: Path) -> list[float]:
    """Load OpenCV's 3x3 camera matrix as ``[fx, fy, cx, cy]``."""
    matrix = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"--intrinsics-npy must contain a 3x3 matrix, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("--intrinsics-npy contains non-finite values")
    return [float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[0, 2]), float(matrix[1, 2])]


def undistort_point(
    point: tuple[float, float], intrinsics: list[float], distortion: np.ndarray
) -> tuple[float, float]:
    """Map a pixel from the source image to the same-sized undistorted image."""
    fx, fy, cx, cy = intrinsics
    camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    points = np.asarray([[point]], dtype=np.float64)
    corrected = cv2.undistortPoints(points, camera_matrix, distortion, P=camera_matrix)
    return float(corrected[0, 0, 0]), float(corrected[0, 0, 1])


def main() -> None:
    args = parse_args()
    if args.patch_size < 1 or args.patch_size % 2 == 0:
        raise ValueError("--patch-size must be a positive odd integer")
    if args.process_res < 1:
        raise ValueError("--process-res must be positive")
    if args.intrinsics is not None and args.intrinsics_npy is not None:
        raise ValueError("Use either --intrinsics or --intrinsics-npy, not both")
    supplied_intrinsics = args.intrinsics
    if args.intrinsics_npy is not None:
        supplied_intrinsics = load_intrinsics_npy(args.intrinsics_npy.resolve())
    if args.distortion_npy is not None and supplied_intrinsics is None:
        raise ValueError("--distortion-npy requires --intrinsics or --intrinsics-npy")
    image_path = args.image.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # PIL keeps this independent of OpenCV and handles common image formats.
    from PIL import Image

    with Image.open(image_path) as image:
        image_width, image_height = image.size

    model = load_depth_model(args.model_dir, args.device)
    distortion = None
    if args.distortion_npy is None:
        depth_map, model_intrinsics = infer_depth_from_path(model, image_path, args.process_res)
    else:
        distortion = np.asarray(np.load(args.distortion_npy.resolve(), allow_pickle=False), dtype=np.float64)
        if distortion.size < 4 or not np.all(np.isfinite(distortion)):
            raise ValueError("--distortion-npy must contain finite OpenCV distortion coefficients")
        from depth_anything_3.utils.depth_analysis import infer_depth_from_bgr, load_bgr_image

        fx, fy, cx, cy = supplied_intrinsics
        camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        undistorted_image = cv2.undistort(load_bgr_image(image_path), camera_matrix, distortion, None, camera_matrix)
        depth_map = infer_depth_from_bgr(model, undistorted_image, args.process_res)
        model_intrinsics = None

    depth_height, depth_width = depth_map.shape
    center_a, target_a = target_to_depth_point(
        args.bbox_a,
        args.point_a,
        "--bbox-a",
        image_width,
        image_height,
        depth_width,
        depth_height,
    )
    center_b, target_b = target_to_depth_point(
        args.bbox_b,
        args.point_b,
        "--bbox-b",
        image_width,
        image_height,
        depth_width,
        depth_height,
    )
    if distortion is not None:
        # target_to_depth_point validated the source target and recorded its
        # source-image centre. Convert that centre to undistorted image pixels
        # before mapping it to the DA3 depth-map grid.
        corrected_a = undistort_point(tuple(target_a["image_point"]), supplied_intrinsics, distortion)
        corrected_b = undistort_point(tuple(target_b["image_point"]), supplied_intrinsics, distortion)
        center_a = image_point_to_depth_point(corrected_a, image_width, image_height, depth_width, depth_height)
        center_b = image_point_to_depth_point(corrected_b, image_width, image_height, depth_width, depth_height)
        target_a["undistorted_image_point"] = list(corrected_a)
        target_b["undistorted_image_point"] = list(corrected_b)
    depth_a = robust_patch_depth(depth_map, center_a, args.patch_size)
    depth_b = robust_patch_depth(depth_map, center_b, args.patch_size)
    intrinsics, intrinsics_source = resolve_intrinsics(
        model_intrinsics,
        supplied_intrinsics,
        image_width,
        image_height,
        depth_width,
        depth_height,
    )
    point_a = point_b = None
    three_d_distance = None
    if intrinsics is not None:
        point_a = camera_point(center_a, depth_a, intrinsics)
        point_b = camera_point(center_b, depth_b, intrinsics)
        three_d_distance = float(np.linalg.norm(point_a - point_b))

    result = {
        "target_a": target_a,
        "target_b": target_b,
        "depth_map_shape": [depth_height, depth_width],
        "patch_size": args.patch_size,
        "center_a_depth_pixel": [float(center_a[0]), float(center_a[1])],
        "center_b_depth_pixel": [float(center_b[0]), float(center_b[1])],
        "depth_a": depth_a,
        "depth_b": depth_b,
        "depth_difference": abs(depth_a - depth_b),
        "point_a_camera": None if point_a is None else point_a.tolist(),
        "point_b_camera": None if point_b is None else point_b.tolist(),
        "3d_distance": three_d_distance,
        "intrinsics_source": (
            "intrinsics_npy" if args.intrinsics_npy is not None else intrinsics_source
        ),
        "distortion_npy": None if args.distortion_npy is None else str(args.distortion_npy.resolve()),
        "3d_distance_note": (
            None
            if intrinsics is not None
            else "Pass --intrinsics FX FY CX CY to calculate full 3-D distance."
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
