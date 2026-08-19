"""Detect a large new occluder by comparing two metric depth maps.

The normal and occluded images are inferred independently so that a multi-view
model does not force the two different scenes to be geometrically consistent.
Existing ``results.npz`` files can be supplied to skip model inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from depth_anything_3.api import DepthAnything3


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NORMAL_IMAGE = PROJECT_ROOT / "assets/images/1.png"
DEFAULT_OCCLUDED_IMAGE = PROJECT_ROOT / "assets/images/2.png"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/occlusion_detection"


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def load_depth(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Depth result not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "depth" not in data:
            raise KeyError(f"{path} does not contain a 'depth' array")
        depth = np.asarray(data["depth"], dtype=np.float32)

    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError(f"Expected one HxW depth map in {path}, got {depth.shape}")
    return depth


def predict_depths(
    normal_path: Path,
    occluded_path: Path,
    model_id: str,
    device: str,
    process_res: int,
) -> tuple[np.ndarray, np.ndarray]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    print(f"Loading model: {model_id}")
    model = DepthAnything3.from_pretrained(model_id).to(device).eval()

    def infer_one(path: Path) -> np.ndarray:
        print(f"Inferring depth: {path}")
        prediction = model.inference(
            [str(path)],
            process_res=process_res,
            process_res_method="upper_bound_resize",
            export_dir=None,
        )
        depth = np.asarray(prediction.depth[0], dtype=np.float32)
        return depth

    # Run separately: the second image contains scene content absent from the first.
    normal_depth = infer_one(normal_path)
    occluded_depth = infer_one(occluded_path)
    return normal_depth, occluded_depth


def clean_mask(mask: np.ndarray, open_size: int, close_size: int) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8) * 255
    if open_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_size, open_size)
        )
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    if close_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_size, close_size)
        )
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return mask_u8


def find_large_regions(
    mask_u8: np.ndarray, min_area_ratio: float
) -> tuple[list[dict[str, int | float]], np.ndarray]:
    height, width = mask_u8.shape
    image_area = height * width
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    regions: list[dict[str, int | float]] = []
    large_mask = np.zeros_like(mask_u8)
    for label in range(1, count):
        x, y, box_width, box_height, area = stats[label]
        area_ratio = float(area / image_area)
        if area_ratio < min_area_ratio:
            continue
        large_mask[labels == label] = 255
        regions.append(
            {
                "x": int(x),
                "y": int(y),
                "width": int(box_width),
                "height": int(box_height),
                "area_pixels": int(area),
                "area_ratio": area_ratio,
            }
        )
    regions.sort(key=lambda region: float(region["area_ratio"]), reverse=True)
    return regions, large_mask


def draw_overlay(
    occluded_bgr: np.ndarray,
    mask_u8: np.ndarray,
    regions: list[dict[str, int | float]],
) -> np.ndarray:
    output = occluded_bgr.copy()
    image_height, image_width = output.shape[:2]
    depth_height, depth_width = mask_u8.shape

    resized_mask = cv2.resize(
        mask_u8, (image_width, image_height), interpolation=cv2.INTER_NEAREST
    )
    red_layer = np.zeros_like(output)
    red_layer[:, :, 2] = 255
    changed = resized_mask > 0
    # OpenCV returns None for addWeighted() when both indexed arrays are empty.
    # An empty mask is a valid "no large occlusion" result, so skip blending.
    if np.any(changed):
        output[changed] = cv2.addWeighted(
            output[changed], 0.55, red_layer[changed], 0.45, 0
        )

    scale_x = image_width / depth_width
    scale_y = image_height / depth_height
    for index, region in enumerate(regions, start=1):
        x1 = round(int(region["x"]) * scale_x)
        y1 = round(int(region["y"]) * scale_y)
        x2 = round((int(region["x"]) + int(region["width"])) * scale_x)
        y2 = round((int(region["y"]) + int(region["height"])) * scale_y)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 5)
        label = f"Occlusion {index}: {float(region['area_ratio']) * 100:.1f}%"
        text_y = max(35, y1 - 12)
        cv2.putText(
            output,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return output


def save_overview(
    normal_depth: np.ndarray,
    occluded_depth: np.ndarray,
    difference: np.ndarray,
    mask_u8: np.ndarray,
    overlay_bgr: np.ndarray,
    output_path: Path,
) -> None:
    finite_depth = np.concatenate(
        [normal_depth[np.isfinite(normal_depth)], occluded_depth[np.isfinite(occluded_depth)]]
    )
    depth_min, depth_max = np.percentile(finite_depth, (2, 98))
    finite_difference = np.abs(difference[np.isfinite(difference)])
    diff_limit = max(float(np.percentile(finite_difference, 98)), 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    normal_plot = axes[0, 0].imshow(
        normal_depth, cmap="Spectral_r", vmin=depth_min, vmax=depth_max
    )
    axes[0, 0].set_title("Normal depth")
    fig.colorbar(normal_plot, ax=axes[0, 0], label="Depth")

    occluded_plot = axes[0, 1].imshow(
        occluded_depth, cmap="Spectral_r", vmin=depth_min, vmax=depth_max
    )
    axes[0, 1].set_title("Occluded depth")
    fig.colorbar(occluded_plot, ax=axes[0, 1], label="Depth")

    difference_plot = axes[0, 2].imshow(
        difference, cmap="RdBu_r", vmin=-diff_limit, vmax=diff_limit
    )
    axes[0, 2].set_title("Depth difference (occluded - normal)")
    fig.colorbar(difference_plot, ax=axes[0, 2], label="Depth difference")

    axes[1, 0].imshow(mask_u8, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("Large-region mask")
    axes[1, 1].imshow(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Detected large occlusion")
    axes[1, 2].axis("off")

    for axis in axes.flat:
        axis.axis("off")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect a large new occluder from the difference of two depth maps."
    )
    parser.add_argument(
        "normal_image",
        nargs="?",
        type=Path,
        default=DEFAULT_NORMAL_IMAGE,
        help="Normal/reference RGB image",
    )
    parser.add_argument(
        "occluded_image",
        nargs="?",
        type=Path,
        default=DEFAULT_OCCLUDED_IMAGE,
        help="RGB image containing the new occluder",
    )
    parser.add_argument(
        "--normal-depth",
        type=Path,
        help="Existing normal results.npz; must be used with --occluded-depth",
    )
    parser.add_argument(
        "--occluded-depth",
        type=Path,
        help="Existing occluded results.npz; must be used with --normal-depth",
    )
    parser.add_argument(
        "--model-dir", default="depth-anything/DA3METRIC-LARGE", help="DA3 model"
    )
    parser.add_argument("--device", default="cuda", help="Inference device")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument(
        "--depth-threshold",
        type=float,
        default=0.5,
        help="Minimum decrease in depth required for an occlusion candidate",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.03,
        help="Minimum connected-region area divided by the full depth-map area",
    )
    parser.add_argument("--blur-size", type=int, default=5)
    parser.add_argument("--open-size", type=int, default=3)
    parser.add_argument("--close-size", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def validate_odd_size(name: str, value: int) -> None:
    if value < 1 or value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer, got {value}")


def main() -> None:
    args = parse_args()
    normal_image_path = args.normal_image.resolve()
    occluded_image_path = args.occluded_image.resolve()
    normal_bgr = load_image(normal_image_path)
    occluded_bgr = load_image(occluded_image_path)

    if (args.normal_depth is None) != (args.occluded_depth is None):
        raise ValueError("Use --normal-depth and --occluded-depth together")
    if not 0 < args.min_area_ratio < 1:
        raise ValueError("--min-area-ratio must be between 0 and 1")
    if args.depth_threshold <= 0:
        raise ValueError("--depth-threshold must be positive")
    for name in ("blur_size", "open_size", "close_size"):
        validate_odd_size(name, int(getattr(args, name)))

    if args.normal_depth is not None:
        normal_depth = load_depth(args.normal_depth.resolve())
        occluded_depth = load_depth(args.occluded_depth.resolve())
    else:
        normal_depth, occluded_depth = predict_depths(
            normal_image_path,
            occluded_image_path,
            args.model_dir,
            args.device,
            args.process_res,
        )

    if normal_depth.shape != occluded_depth.shape:
        raise ValueError(
            "Depth maps must have identical shapes for direct subtraction, got "
            f"{normal_depth.shape} and {occluded_depth.shape}"
        )

    valid = (
        np.isfinite(normal_depth)
        & np.isfinite(occluded_depth)
        & (normal_depth > 0)
        & (occluded_depth > 0)
    )
    difference = np.full(normal_depth.shape, np.nan, dtype=np.float32)
    difference[valid] = occluded_depth[valid] - normal_depth[valid]

    filtered_difference = np.nan_to_num(difference, nan=0.0)
    if args.blur_size > 1:
        filtered_difference = cv2.GaussianBlur(
            filtered_difference, (args.blur_size, args.blur_size), 0
        )

    # A newly inserted occluder normally makes the corresponding scene pixels closer.
    candidate = valid & (filtered_difference <= -args.depth_threshold)
    mask_u8 = clean_mask(candidate, args.open_size, args.close_size)
    mask_u8[~valid] = 0
    regions, large_mask_u8 = find_large_regions(mask_u8, args.min_area_ratio)
    detected = bool(regions)

    overlay_bgr = draw_overlay(occluded_bgr, large_mask_u8, regions)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "depth_difference.npy", difference)
    cv2.imwrite(str(output_dir / "occlusion_mask.png"), large_mask_u8)
    cv2.imwrite(str(output_dir / "occlusion_overlay.png"), overlay_bgr)
    save_overview(
        normal_depth,
        occluded_depth,
        difference,
        large_mask_u8,
        overlay_bgr,
        output_dir / "overview.png",
    )

    report = {
        "large_occlusion_detected": detected,
        "decision": "OCCLUSION" if detected else "NO_LARGE_OCCLUSION",
        "normal_image": str(normal_image_path),
        "occluded_image": str(occluded_image_path),
        "depth_shape": list(normal_depth.shape),
        "depth_threshold": args.depth_threshold,
        "minimum_area_ratio": args.min_area_ratio,
        "candidate_area_ratio": float(np.count_nonzero(mask_u8) / mask_u8.size),
        "detected_area_ratio": float(
            np.count_nonzero(large_mask_u8) / large_mask_u8.size
        ),
        "largest_region_area_ratio": (
            float(regions[0]["area_ratio"]) if regions else 0.0
        ),
        "regions": regions,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== Depth occlusion detection ===")
    print(f"Decision: {report['decision']}")
    print(f"Candidate area: {report['candidate_area_ratio'] * 100:.2f}%")
    print(f"Large regions: {len(regions)}")
    if regions:
        print(f"Largest region: {report['largest_region_area_ratio'] * 100:.2f}%")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
