"""Classify people relative to glass and measure two locations in one run.

The distance targets are measured on ``current_image``.  This command reuses
the existing person/glass classification and distance helper functions while
loading the DA3 model only once when depth maps are not supplied.

Example:
    uv run python -m depth_anything_3.analyze_glass_and_distance \\
        assets/images/empty_glass.jpg assets/images/current.jpg \\
        --yolo-model models/yolo11n-seg.pt \\
        --point-a 190 250 --point-b 460 270
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image

from depth_anything_3.detect_person_behind_glass import (
    DEFAULT_OUTPUT_DIR,
    classify_person_against_reference,
    draw_person_segmentation,
    draw_results,
)
from depth_anything_3.measure_location_distance import (
    camera_point,
    resolve_intrinsics,
    target_to_depth_point,
)
from depth_anything_3.utils.depth_analysis import (
    infer_depth_from_path,
    load_bgr_image,
    load_depth_map,
    load_depth_model,
    robust_patch_depth,
)
from depth_anything_3.utils.person_segmentation import UltralyticsPersonSegmenter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_image", type=Path, help="Empty-room image with glass")
    parser.add_argument("current_image", type=Path, help="Current image containing people")
    parser.add_argument("--reference-depth", type=Path, help="Existing reference .npy or results.npz")
    parser.add_argument("--current-depth", type=Path, help="Existing current .npy or results.npz")
    parser.add_argument("--yolo-model", required=True, help="Ultralytics segmentation weights")
    parser.add_argument("--yolo-confidence", type=float, default=0.25)
    parser.add_argument("--yolo-device", help="YOLO device, e.g. cuda:0 or cpu")
    parser.add_argument("--yolo-image-size", type=int, default=640)
    parser.add_argument("--person-class-id", type=int, default=0)
    parser.add_argument("--model-dir", default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--device", default="cuda", help="DA3 inference device")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--depth-threshold", type=float, default=0.20)
    parser.add_argument("--min-front-fraction", type=float, default=0.60)
    parser.add_argument("--erode-pixels", type=int, default=5)
    parser.add_argument("--min-valid-pixels", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "combined")
    target_a = parser.add_mutually_exclusive_group(required=True)
    target_a.add_argument("--bbox-a", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"))
    target_a.add_argument("--point-a", nargs=2, type=float, metavar=("X", "Y"))
    target_b = parser.add_mutually_exclusive_group(required=True)
    target_b.add_argument("--bbox-b", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"))
    target_b.add_argument("--point-b", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--patch-size", type=int, default=15)
    parser.add_argument("--intrinsics", nargs=4, type=float, metavar=("FX", "FY", "CX", "CY"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.reference_depth is None) != (args.current_depth is None):
        raise ValueError("Use --reference-depth and --current-depth together")
    if args.patch_size < 1 or args.patch_size % 2 == 0:
        raise ValueError("--patch-size must be a positive odd integer")
    if args.process_res < 1 or args.depth_threshold <= 0:
        raise ValueError("--process-res and --depth-threshold must be positive")
    if not 0 < args.min_front_fraction <= 1:
        raise ValueError("--min-front-fraction must be in (0, 1]")
    if args.erode_pixels < 0 or args.min_valid_pixels < 1:
        raise ValueError("--erode-pixels must be nonnegative and --min-valid-pixels positive")

    reference_path = args.reference_image.resolve()
    current_path = args.current_image.resolve()
    current_bgr = load_bgr_image(current_path)
    current_intrinsics = None
    if args.reference_depth is None:
        model = load_depth_model(args.model_dir, args.device)
        reference_depth, _ = infer_depth_from_path(model, reference_path, args.process_res)
        current_depth, current_intrinsics = infer_depth_from_path(model, current_path, args.process_res)
    else:
        reference_depth = load_depth_map(args.reference_depth.resolve())
        current_depth = load_depth_map(args.current_depth.resolve())
    if reference_depth.shape != current_depth.shape:
        raise ValueError("Reference and current depth maps must have identical shapes")

    with Image.open(current_path) as image:
        image_width, image_height = image.size
    depth_height, depth_width = current_depth.shape
    center_a, target_a = target_to_depth_point(args.bbox_a, args.point_a, "--bbox-a", image_width, image_height, depth_width, depth_height)
    center_b, target_b = target_to_depth_point(args.bbox_b, args.point_b, "--bbox-b", image_width, image_height, depth_width, depth_height)
    depth_a = robust_patch_depth(current_depth, center_a, args.patch_size)
    depth_b = robust_patch_depth(current_depth, center_b, args.patch_size)
    intrinsics, intrinsics_source = resolve_intrinsics(current_intrinsics, args.intrinsics, image_width, image_height, depth_width, depth_height)
    point_a = point_b = None
    three_d_distance = None
    if intrinsics is not None:
        point_a = camera_point(center_a, depth_a, intrinsics)
        point_b = camera_point(center_b, depth_b, intrinsics)
        three_d_distance = float(((point_a - point_b) ** 2).sum() ** 0.5)

    segmenter = UltralyticsPersonSegmenter(args.yolo_model, person_class_id=args.person_class_id, confidence=args.yolo_confidence, device=args.yolo_device, image_size=args.yolo_image_size)
    person_masks = segmenter.predict_instance_masks(current_bgr, current_depth.shape)
    people = [classify_person_against_reference(reference_depth, current_depth, mask, args.depth_threshold, args.min_front_fraction, args.erode_pixels, args.min_valid_pixels) for mask in person_masks]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "yolo_person_segmentation.png"), draw_person_segmentation(current_bgr, person_masks))
    cv2.imwrite(str(output_dir / "classification_overlay.png"), draw_results(current_bgr, person_masks, people))
    report = {
        "reference_image": str(reference_path), "current_image": str(current_path),
        "people": people, "detected_person_count": len(people),
        "distance": {
            "target_a": target_a, "target_b": target_b,
            "depth_a": depth_a, "depth_b": depth_b,
            "depth_difference": abs(depth_a - depth_b), "3d_distance": three_d_distance,
            "intrinsics_source": intrinsics_source,
            "3d_distance_note": None if intrinsics is not None else "Pass --intrinsics FX FY CX CY to calculate full 3-D distance.",
        },
    }
    (output_dir / "combined_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report and overlays: {output_dir}")


if __name__ == "__main__":
    main()
