"""Video occlusion detection with optional depth-normalized region area.

Depth mode compensates for perspective by converting each connected region's
image area to its equivalent area at a configurable reference depth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from depth_anything_3.detect_depth_occlusion import find_large_regions
from depth_anything_3.detect_video_occlusion import (
    analyze_frame as analyze_frame_fixed,
    build_parser as build_base_parser,
    run_video,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/video_occlusion_depth_ratio_runs"


def find_regions_with_depth_ratio(
    mask_u8: np.ndarray,
    current_depth: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], np.ndarray]:
    """Filter connected regions using fixed or depth-normalized area ratio."""
    if args.area_mode == "fixed":
        regions, large_mask = find_large_regions(mask_u8, args.min_area_ratio)
        for region in regions:
            region["area_mode"] = "fixed"
            region["required_area_ratio"] = float(args.min_area_ratio)
        return regions, large_mask

    height, width = mask_u8.shape
    image_area = height * width
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    regions: list[dict[str, object]] = []
    accepted_mask = np.zeros_like(mask_u8)
    for label_id in range(1, count):
        x, y, box_width, box_height, area = stats[label_id]
        area_ratio = float(area / image_area)
        if area_ratio < args.component_min_area_ratio:
            continue

        component_mask = (labels == label_id).astype(np.uint8) * 255
        depth_mask = component_mask
        if args.region_depth_erode > 0:
            size = args.region_depth_erode * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            eroded = cv2.erode(component_mask, kernel, iterations=1)
            if np.any(eroded):
                depth_mask = eroded

        depth_values = current_depth[
            (depth_mask > 0) & np.isfinite(current_depth) & (current_depth > 0)
        ]
        if depth_values.size == 0:
            continue

        region_depth = float(np.median(depth_values))
        dynamic_required_ratio = float(
            args.reference_area_ratio
            * (args.reference_depth / region_depth) ** 2
        )
        dynamic_required_ratio = float(
            np.clip(
                dynamic_required_ratio,
                args.dynamic_min_area_ratio,
                args.dynamic_max_area_ratio,
            )
        )
        depth_adjusted_ratio = float(
            area_ratio * (region_depth / args.reference_depth) ** 2
        )

        if area_ratio < dynamic_required_ratio:
            continue

        accepted_mask[labels == label_id] = 255
        regions.append(
            {
                "x": int(x),
                "y": int(y),
                "width": int(box_width),
                "height": int(box_height),
                "area_pixels": int(area),
                "area_ratio": area_ratio,
                "area_mode": "depth",
                "median_depth": region_depth,
                "required_area_ratio": dynamic_required_ratio,
                "depth_adjusted_area_ratio": depth_adjusted_ratio,
                "reference_depth": float(args.reference_depth),
                "reference_area_ratio": float(args.reference_area_ratio),
            }
        )

    regions.sort(
        key=lambda region: float(region["depth_adjusted_area_ratio"]), reverse=True
    )
    return regions, accepted_mask


def analyze_frame_depth_ratio(
    normal_depth: np.ndarray,
    current_depth: np.ndarray,
    normal_person_mask: np.ndarray,
    current_person_mask: np.ndarray,
    frame_bgr: np.ndarray,
    args: argparse.Namespace,
):
    return analyze_frame_fixed(
        normal_depth,
        current_depth,
        normal_person_mask,
        current_person_mask,
        frame_bgr,
        args,
        region_finder=find_regions_with_depth_ratio,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser()
    parser.description = (
        "Compare video frames with a normal frame using fixed or "
        "depth-normalized occlusion area."
    )
    parser.set_defaults(output_root=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--area-mode",
        choices=("fixed", "depth"),
        default="depth",
        help="fixed uses --min-area-ratio; depth compensates for perspective",
    )
    parser.add_argument(
        "--reference-depth",
        type=float,
        default=3.0,
        help="Reference distance in metric-depth units (normally metres)",
    )
    parser.add_argument(
        "--reference-area-ratio",
        type=float,
        default=0.03,
        help="Required image-area ratio for an object at the reference depth",
    )
    parser.add_argument(
        "--dynamic-min-area-ratio",
        type=float,
        default=0.003,
        help="Lower clamp for the depth-derived area threshold",
    )
    parser.add_argument(
        "--dynamic-max-area-ratio",
        type=float,
        default=0.30,
        help="Upper clamp for the depth-derived area threshold",
    )
    parser.add_argument(
        "--component-min-area-ratio",
        type=float,
        default=0.0005,
        help="Discard tiny components before computing their median depth",
    )
    parser.add_argument(
        "--region-depth-erode",
        type=int,
        default=2,
        help="Erosion radius used before taking region median depth",
    )
    return parser


def validate_depth_ratio_args(args: argparse.Namespace) -> None:
    if args.reference_depth <= 0:
        raise ValueError("--reference-depth must be positive")
    if not 0 < args.reference_area_ratio < 1:
        raise ValueError("--reference-area-ratio must be in (0, 1)")
    if not 0 <= args.component_min_area_ratio < 1:
        raise ValueError("--component-min-area-ratio must be in [0, 1)")
    if not 0 < args.dynamic_min_area_ratio <= args.dynamic_max_area_ratio <= 1:
        raise ValueError(
            "Dynamic area clamps must satisfy 0 < minimum <= maximum <= 1"
        )
    if args.region_depth_erode < 0:
        raise ValueError("--region-depth-erode must be nonnegative")


def main() -> None:
    args = build_parser().parse_args()
    validate_depth_ratio_args(args)
    print(f"Area mode: {args.area_mode}")
    if args.area_mode == "depth":
        print(
            "Depth-normalized area: "
            f"{args.reference_area_ratio * 100:.2f}% at "
            f"depth {args.reference_depth:.2f}"
        )
    run_video(args, frame_analyzer=analyze_frame_depth_ratio)


if __name__ == "__main__":
    main()
