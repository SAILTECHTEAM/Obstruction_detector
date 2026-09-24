"""Estimate camera intrinsics from chessboard calibration photos.

``--chessboard-cols`` and ``--chessboard-rows`` are the number of *inner
corners*, not the number of squares.  For example, a board with 10 by 7
squares has 9 by 6 inner corners.

Example:
    python calibrate_camera.py ./calibration_photos \\
        --chessboard-cols 9 --chessboard-rows 6 \\
        --square-size 0.024 --output ./calibration/intrinsics.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_folder", type=Path, help="Folder containing chessboard photos")
    parser.add_argument("--chessboard-cols", type=int, required=True, help="Inner corners across each row")
    parser.add_argument("--chessboard-rows", type=int, required=True, help="Inner corners down each column")
    parser.add_argument("--square-size", type=float, default=1.0, help="Chessboard square size; any unit (default: 1)")
    parser.add_argument("--output", type=Path, required=True, help="Destination 3x3 intrinsics .npy file")
    parser.add_argument("--recursive", action="store_true", help="Also search subfolders")
    parser.add_argument("--min-images", type=int, default=10, help="Minimum successful photos required (default: 10)")
    parser.add_argument(
        "--debug-dir", type=Path,
        help=("Folder for images with detected inner corners drawn. Defaults to "
              "<output-stem>_corners beside --output."),
    )
    parser.add_argument("--no-debug-images", action="store_true", help="Do not save corner-detection overlays")
    return parser.parse_args()


def image_paths(folder: Path, recursive: bool) -> list[Path]:
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def find_corners(gray: np.ndarray, pattern_size: tuple[int, int]) -> tuple[bool, np.ndarray | None]:
    """Use the more robust SB detector when available, with a classic fallback."""
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray, pattern_size, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )
        if found:
            return found, corners
    found, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners


def main() -> None:
    args = parse_args()
    if args.chessboard_cols < 2 or args.chessboard_rows < 2:
        raise ValueError("Chessboard dimensions must both be at least 2")
    if args.square_size <= 0 or args.min_images < 3:
        raise ValueError("--square-size must be positive and --min-images must be at least 3")
    folder = args.image_folder.resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Calibration image folder does not exist: {folder}")
    paths = image_paths(folder, args.recursive)
    if not paths:
        raise FileNotFoundError(f"No supported images found in {folder}")

    pattern_size = (args.chessboard_cols, args.chessboard_rows)
    object_template = np.zeros((args.chessboard_cols * args.chessboard_rows, 3), np.float32)
    object_template[:, :2] = np.mgrid[0:args.chessboard_cols, 0:args.chessboard_rows].T.reshape(-1, 2)
    object_template *= args.square_size
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    accepted: list[str] = []
    rejected: list[str] = []
    image_size: tuple[int, int] | None = None
    output = args.output.resolve()
    debug_dir = None
    if not args.no_debug_images:
        debug_dir = (args.debug_dir or output.with_name(f"{output.stem}_corners")).resolve()
        debug_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            rejected.append(f"{path.name}: unreadable")
            continue
        size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = size
        if size != image_size:
            rejected.append(f"{path.name}: image size {size}, expected {image_size}")
            continue
        found, corners = find_corners(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), pattern_size)
        if not found or corners is None:
            rejected.append(f"{path.name}: chessboard not found")
            continue
        object_points.append(object_template.copy())
        image_points.append(corners)
        accepted.append(path.name)
        if debug_dir:
            cv2.drawChessboardCorners(image, pattern_size, corners, True)
            # Preserve the image name so it is easy to compare the overlay
            # against its source capture.
            cv2.imwrite(str(debug_dir / path.name), image)

    if len(accepted) < args.min_images:
        raise RuntimeError(
            f"Only {len(accepted)} of {len(paths)} photos contained the requested chessboard; "
            f"need at least {args.min_images}. Check the inner-corner dimensions."
        )
    assert image_size is not None
    rms, intrinsics, distortion, rotations, translations = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, intrinsics.astype(np.float64))
    # Distortion is deliberately saved alongside intrinsics because it is needed
    # when undistorting frames before using the matrix.
    distortion_path = output.with_name(f"{output.stem}_distortion.npy")
    np.save(distortion_path, distortion.astype(np.float64))
    report = {
        "image_size": list(image_size), "rms_reprojection_error_pixels": float(rms),
        "used_images": accepted, "rejected_images": rejected,
        "intrinsics_path": str(output), "distortion_path": str(distortion_path),
        "corner_overlay_directory": None if debug_dir is None else str(debug_dir),
        "intrinsics": intrinsics.tolist(), "distortion": distortion.ravel().tolist(),
        "views": len(rotations), "translations": len(translations),
    }
    report_path = output.with_name(f"{output.stem}_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
