"""Classify detected people as in front of or behind a fixed glass barrier.

The reference image must show the same fixed camera view with no people.  Its
depth map is used as the depth of the room and glass at each pixel.

Example:
    uv run python -m depth_anything_3.detect_person_behind_glass \
        assets/images/empty_glass.jpg assets/images/current.jpg \
        --yolo-model yolov8n-seg.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from depth_anything_3.utils.person_segmentation import UltralyticsPersonSegmenter
from depth_anything_3.utils.depth_analysis import (
    depth_difference,
    infer_depth_pair,
    load_bgr_image,
    load_depth_map,
    valid_depth_mask,
)
from depth_anything_3.utils.masks import erode_mask


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "person_glass_position"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_image", type=Path, help="Empty-room image with glass")
    parser.add_argument("current_image", type=Path, help="Current image containing people")
    parser.add_argument("--reference-depth", type=Path, help="Existing reference results.npz")
    parser.add_argument("--current-depth", type=Path, help="Existing current results.npz")
    parser.add_argument("--yolo-model", required=True, help="Ultralytics segmentation weights")
    parser.add_argument("--yolo-confidence", type=float, default=0.25)
    parser.add_argument("--yolo-device", help="YOLO device, e.g. cuda:0 or cpu")
    parser.add_argument("--yolo-image-size", type=int, default=640)
    parser.add_argument("--person-class-id", type=int, default=0)
    parser.add_argument("--model-dir", default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--device", default="cuda", help="DA3 inference device")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument(
        "--depth-threshold",
        type=float,
        default=0.20,
        help="Metres a person must be nearer than reference to be in front (default: 0.20)",
    )
    parser.add_argument(
        "--min-front-fraction",
        type=float,
        default=0.60,
        help="Required fraction of valid inner-mask pixels nearer than threshold (default: 0.60)",
    )
    parser.add_argument(
        "--erode-pixels",
        type=int,
        default=5,
        help="Remove this many depth-map pixels from each person-mask boundary (default: 5)",
    )
    parser.add_argument("--min-valid-pixels", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def classify_person_against_reference(
    reference_depth: np.ndarray,
    current_depth: np.ndarray,
    person_mask: np.ndarray,
    depth_threshold: float,
    min_front_fraction: float,
    erode_pixels: int,
    min_valid_pixels: int,
) -> dict[str, float | int | str]:
    """Classify one person using depth relative to the empty-room reference.

    A person in front of glass must be materially closer than the depth at the
    same reference pixels.  Otherwise the person is behind the known glass
    barrier (or DA3's depth is dominated by that glass surface).
    """
    inner_mask = erode_mask(person_mask, erode_pixels) > 0
    valid = inner_mask & valid_depth_mask(reference_depth, current_depth)
    valid_count = int(np.count_nonzero(valid))
    if valid_count < min_valid_pixels:
        return {
            "classification": "unknown",
            "valid_pixels": valid_count,
            "front_fraction": 0.0,
            "median_depth_difference": float("nan"),
        }

    difference = depth_difference(reference_depth, current_depth)[valid]
    front_fraction = float(np.mean(difference < -depth_threshold))
    classification = (
        "in_front_of_glass"
        if front_fraction >= min_front_fraction
        else "behind_glass"
    )
    return {
        "classification": classification,
        "valid_pixels": valid_count,
        "front_fraction": front_fraction,
        "median_depth_difference": float(np.median(difference)),
    }


def draw_results(image_bgr: np.ndarray, masks: list[np.ndarray], results: list[dict[str, object]]) -> np.ndarray:
    """Draw a concise per-person classification overlay."""
    output = image_bgr.copy()
    for index, (mask, result) in enumerate(zip(masks, results), start=1):
        resized_mask = cv2.resize(
            mask, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        points = cv2.findNonZero(resized_mask)
        if points is None:
            continue
        x, y, width, height = cv2.boundingRect(points)
        is_front = result["classification"] == "in_front_of_glass"
        color = (0, 200, 0) if is_front else (0, 165, 255)
        if result["classification"] == "unknown":
            color = (0, 0, 255)
        label = f"Person {index}: {str(result['classification']).replace('_', ' ')}"
        cv2.rectangle(output, (x, y), (x + width, y + height), color, 2)
        cv2.putText(output, label, (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return output


def draw_person_segmentation(
    image_bgr: np.ndarray, masks: list[np.ndarray]
) -> np.ndarray:
    """Draw each YOLO person instance mask with a distinct colour and label."""
    output = image_bgr.copy()
    colors = ((255, 80, 80), (80, 220, 80), (80, 160, 255), (220, 80, 220))
    for index, mask in enumerate(masks, start=1):
        resized_mask = cv2.resize(
            mask, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        selected = resized_mask > 0
        if not np.any(selected):
            continue
        color = colors[(index - 1) % len(colors)]
        color_layer = np.empty_like(output)
        color_layer[:] = color
        output[selected] = cv2.addWeighted(
            output[selected], 0.5, color_layer[selected], 0.5, 0
        )
        points = cv2.findNonZero(resized_mask)
        x, y, width, height = cv2.boundingRect(points)
        cv2.rectangle(output, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            output,
            f"YOLO person {index}",
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def main() -> None:
    args = parse_args()
    if (args.reference_depth is None) != (args.current_depth is None):
        raise ValueError("Use --reference-depth and --current-depth together")
    if args.depth_threshold <= 0:
        raise ValueError("--depth-threshold must be positive")
    if not 0 < args.min_front_fraction <= 1:
        raise ValueError("--min-front-fraction must be in (0, 1]")
    if args.erode_pixels < 0 or args.min_valid_pixels < 1:
        raise ValueError("--erode-pixels must be nonnegative and --min-valid-pixels positive")

    reference_path = args.reference_image.resolve()
    current_path = args.current_image.resolve()
    current_bgr = load_bgr_image(current_path)
    if args.reference_depth is None:
        reference_depth, current_depth = infer_depth_pair(
            reference_path, current_path, args.model_dir, args.device, args.process_res
        )
    else:
        reference_depth = load_depth_map(args.reference_depth.resolve())
        current_depth = load_depth_map(args.current_depth.resolve())
    if reference_depth.shape != current_depth.shape:
        raise ValueError("Reference and current depth maps must have identical shapes")

    segmenter = UltralyticsPersonSegmenter(
        args.yolo_model,
        person_class_id=args.person_class_id,
        confidence=args.yolo_confidence,
        device=args.yolo_device,
        image_size=args.yolo_image_size,
    )
    person_masks = segmenter.predict_instance_masks(current_bgr, reference_depth.shape)
    results: list[dict[str, object]] = []
    for person_mask in person_masks:
        result: dict[str, object] = classify_person_against_reference(
            reference_depth,
            current_depth,
            person_mask,
            args.depth_threshold,
            args.min_front_fraction,
            args.erode_pixels,
            args.min_valid_pixels,
        )
        results.append(result)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    segmentation_overlay = draw_person_segmentation(current_bgr, person_masks)
    cv2.imwrite(str(output_dir / "yolo_person_segmentation.png"), segmentation_overlay)
    overlay = draw_results(current_bgr, person_masks, results)
    cv2.imwrite(str(output_dir / "classification_overlay.png"), overlay)
    report = {
        "reference_image": str(reference_path),
        "current_image": str(current_path),
        "depth_threshold": args.depth_threshold,
        "min_front_fraction": args.min_front_fraction,
        "detected_person_count": len(person_masks),
        "people": results,
    }
    (output_dir / "classification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved YOLO segmentation: {output_dir / 'yolo_person_segmentation.png'}")
    print(f"Saved overlay: {output_dir / 'classification_overlay.png'}")


if __name__ == "__main__":
    main()
