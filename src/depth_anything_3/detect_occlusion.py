"""Detect depth occlusions in one image or every frame of a video.

Use ``--area-mode depth`` to make the minimum occlusion area depth-aware.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from depth_anything_3.utils.person_segmentation import UltralyticsPersonSegmenter
from depth_anything_3.utils.depth_analysis import (
    infer_depth_from_bgr as infer_depth,
    load_bgr_image,
    load_depth_model,
    valid_depth_mask,
    depth_difference,
    validate_inference_device,
)
from depth_anything_3.utils.masks import (
    clean_mask,
    dilate_mask,
    erode_mask,
    find_large_regions,
    load_yolo_seg_class_mask,
    mask_region,
    validate_odd_size,
)
from depth_anything_3.utils.logger import LOG_LEVELS, logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/video_occlusion_runs"
DEFAULT_IMAGE_OUTPUT_DIR = PROJECT_ROOT / "outputs/image_occlusion"


def draw_overlay(
    image_bgr: np.ndarray,
    mask_u8: np.ndarray,
    regions: list[dict[str, int | float]],
) -> np.ndarray:
    """Draw detected depth/person regions on an image or video frame."""
    output = image_bgr.copy()
    image_height, image_width = output.shape[:2]
    depth_height, depth_width = mask_u8.shape
    resized_mask = cv2.resize(mask_u8, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    changed = resized_mask > 0
    if np.any(changed):
        red = np.zeros_like(output)
        red[:, :, 2] = 255
        output[changed] = cv2.addWeighted(output[changed], 0.55, red[changed], 0.45, 0)
    for index, region in enumerate(regions, start=1):
        x1 = round(int(region["x"]) * image_width / depth_width)
        y1 = round(int(region["y"]) * image_height / depth_height)
        x2 = round((int(region["x"]) + int(region["width"])) * image_width / depth_width)
        y2 = round((int(region["y"]) + int(region["height"])) * image_height / depth_height)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"Person occlusion" if int(region.get("is_person", 0)) else f"Occlusion {index}"
        cv2.putText(output, label, (x1, max(28, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return output


def make_run_directory(output_root: Path, video_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    safe_stem = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in video_path.stem
    )
    run_dir = output_root / f"{timestamp}_{safe_stem}"
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{timestamp}_{safe_stem}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def find_regions_with_depth_ratio(
    mask_u8: np.ndarray, current_depth: np.ndarray, args: argparse.Namespace
) -> tuple[list[dict[str, object]], np.ndarray]:
    """Filter regions by an area threshold adjusted for estimated depth."""
    height, width = mask_u8.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    accepted = np.zeros_like(mask_u8)
    regions: list[dict[str, object]] = []
    for label_id in range(1, count):
        x, y, box_width, box_height, area = stats[label_id]
        area_ratio = float(area / (height * width))
        if area_ratio < args.component_min_area_ratio:
            continue
        component = (labels == label_id).astype(np.uint8) * 255
        depth_mask = erode_mask(component, args.region_depth_erode) if args.region_depth_erode else component
        if not np.any(depth_mask):
            depth_mask = component
        values = current_depth[(depth_mask > 0) & np.isfinite(current_depth) & (current_depth > 0)]
        if values.size == 0:
            continue
        region_depth = float(np.median(values))
        required_ratio = float(np.clip(
            args.reference_area_ratio * (args.reference_depth / region_depth) ** 2,
            args.dynamic_min_area_ratio,
            args.dynamic_max_area_ratio,
        ))
        if area_ratio < required_ratio:
            continue
        accepted[labels == label_id] = 255
        regions.append({
            "x": int(x), "y": int(y), "width": int(box_width), "height": int(box_height),
            "area_pixels": int(area), "area_ratio": area_ratio, "area_mode": "depth",
            "median_depth": region_depth, "required_area_ratio": required_ratio,
            "depth_adjusted_area_ratio": float(area_ratio * (region_depth / args.reference_depth) ** 2),
        })
    regions.sort(key=lambda region: float(region["depth_adjusted_area_ratio"]), reverse=True)
    return regions, accepted


def analyze_frame(
    normal_depth: np.ndarray,
    current_depth: np.ndarray,
    normal_person_mask: np.ndarray,
    current_person_mask: np.ndarray,
    frame_bgr: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object], np.ndarray, np.ndarray]:
    if current_depth.shape != normal_depth.shape:
        raise ValueError(
            "Normal frame and video frame produced different depth shapes: "
            f"{normal_depth.shape} vs {current_depth.shape}. "
            "Use a normal frame with the same aspect ratio as the video."
        )

    valid = valid_depth_mask(normal_depth, current_depth)
    difference = depth_difference(normal_depth, current_depth)

    filtered_difference = np.nan_to_num(difference, nan=0.0)
    if args.blur_size > 1:
        filtered_difference = cv2.GaussianBlur(
            filtered_difference, (args.blur_size, args.blur_size), 0
        )

    candidate = valid & (filtered_difference <= -args.depth_threshold)
    mask_u8 = clean_mask(candidate, args.open_size, args.close_size)
    mask_u8[~valid] = 0

    # The one-third rule uses the raw current mask. Dilation is only used when
    # removing small/moving people from the depth-change mask.
    person_area_ratio = float(
        np.count_nonzero(current_person_mask) / current_person_mask.size
    )
    person_occlusion = person_area_ratio >= args.person_alert_ratio

    all_person_mask = cv2.bitwise_or(normal_person_mask, current_person_mask)
    ignored_person_mask = dilate_mask(all_person_mask, args.person_dilate)
    mask_u8[ignored_person_mask > 0] = 0

    if args.area_mode == "fixed":
        regions, large_depth_mask = find_large_regions(
            mask_u8, args.min_area_ratio
        )
    else:
        regions, large_depth_mask = find_regions_with_depth_ratio(mask_u8, current_depth, args)
    depth_occlusion = bool(regions)

    display_regions = list(regions)
    final_mask = large_depth_mask.copy()
    if person_occlusion:
        final_mask = cv2.bitwise_or(final_mask, current_person_mask)
        display_regions.append(mask_region(current_person_mask, person_area_ratio))

    if person_occlusion:
        decision = "PERSON_OCCLUSION"
    elif depth_occlusion:
        decision = "DEPTH_OCCLUSION"
    else:
        decision = "NO_LARGE_OCCLUSION"

    overlay = draw_overlay(frame_bgr, final_mask, display_regions)
    color = (0, 0, 255) if decision != "NO_LARGE_OCCLUSION" else (0, 180, 0)
    # Draw outlined text directly on the frame without an opaque background box.
    cv2.putText(
        overlay,
        decision,
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        decision,
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )
    details = (
        f"person={person_area_ratio * 100:.1f}%  "
        f"depth-change={np.count_nonzero(mask_u8) / mask_u8.size * 100:.1f}%"
    )
    cv2.putText(
        overlay,
        details,
        (16, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        details,
        (16, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    record: dict[str, object] = {
        "decision": decision,
        "detected": decision != "NO_LARGE_OCCLUSION",
        "person_area_ratio": person_area_ratio,
        "person_alert_ratio": args.person_alert_ratio,
        "person_occlusion_detected": person_occlusion,
        "depth_occlusion_detected": depth_occlusion,
        "candidate_area_ratio": float(np.count_nonzero(mask_u8) / mask_u8.size),
        "largest_region_area_ratio": (
            float(regions[0]["area_ratio"]) if regions else 0.0
        ),
        "regions": regions,
    }
    return overlay, record, final_mask, ignored_person_mask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare every video frame with a separate normal reference frame."
    )
    parser.add_argument("normal_frame", type=Path, help="Normal/reference image")
    parser.add_argument("input_path", type=Path, help="Image or video to inspect")
    parser.add_argument("--input-type", choices=("auto", "image", "video"), default="auto")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show Depth Anything 3 internal INFO messages",
    )
    parser.add_argument(
        "--yolo-model",
        help="Ultralytics segmentation weights, for example yolo11n-seg.pt",
    )
    parser.add_argument("--model-dir", default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--device", default="cuda", help="DA3 device")
    parser.add_argument(
        "--yolo-device",
        help="Ultralytics device, for example cuda:0 or cpu (default: auto)",
    )
    parser.add_argument("--yolo-confidence", type=float, default=0.25)
    parser.add_argument("--yolo-image-size", type=int, default=640)
    parser.add_argument("--person-class-id", type=int, default=0)
    parser.add_argument("--normal-yolo-txt", type=Path, help="Optional reference YOLO segmentation TXT (image mode)")
    parser.add_argument("--current-yolo-txt", type=Path, help="Optional current YOLO segmentation TXT (image mode)")
    parser.add_argument("--person-alert-ratio", type=float, default=1.0 / 3.0)
    parser.add_argument("--person-dilate", type=int, default=7)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--depth-threshold", type=float, default=0.5)
    parser.add_argument("--min-area-ratio", type=float, default=0.03)
    parser.add_argument("--area-mode", choices=("fixed", "depth"), default="fixed")
    parser.add_argument("--reference-depth", type=float, default=3.0)
    parser.add_argument("--reference-area-ratio", type=float, default=0.03)
    parser.add_argument("--dynamic-min-area-ratio", type=float, default=0.003)
    parser.add_argument("--dynamic-max-area-ratio", type=float, default=0.30)
    parser.add_argument("--component-min-area-ratio", type=float, default=0.0005)
    parser.add_argument("--region-depth-erode", type=int, default=2)
    parser.add_argument("--blur-size", type=int, default=5)
    parser.add_argument("--open-size", type=int, default=3)
    parser.add_argument("--close-size", type=int, default=7)
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Analyze every Nth frame; skipped frames are copied without analysis",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many input frames; 0 processes the entire video",
    )
    parser.add_argument(
        "--save-alert-frames",
        action="store_true",
        help="Save individual annotated PNGs for detected frames",
    )
    parser.add_argument("--codec", default="mp4v", help="FourCC output codec")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_IMAGE_OUTPUT_DIR)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def validate_args(args: argparse.Namespace, *, video_mode: bool) -> None:
    if not args.normal_frame.is_file():
        raise FileNotFoundError(f"Normal frame not found: {args.normal_frame}")
    if not args.input_path.is_file():
        raise FileNotFoundError(f"Input not found: {args.input_path}")
    if video_mode and not args.yolo_model:
        raise ValueError("--yolo-model is required for video input")
    if not video_mode and not args.yolo_model and not (args.normal_yolo_txt and args.current_yolo_txt):
        raise ValueError("Image input requires --yolo-model or both YOLO TXT masks")
    validate_inference_device(args.device)
    if not 0 < args.person_alert_ratio <= 1:
        raise ValueError("--person-alert-ratio must be in (0, 1]")
    if not 0 < args.min_area_ratio < 1:
        raise ValueError("--min-area-ratio must be in (0, 1)")
    if args.person_dilate < 0:
        raise ValueError("--person-dilate must be nonnegative")
    if args.depth_threshold <= 0:
        raise ValueError("--depth-threshold must be positive")
    if args.frame_step < 1:
        raise ValueError("--frame-step must be at least 1")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be nonnegative")
    if len(args.codec) != 4:
        raise ValueError("--codec must contain exactly four characters")
    for name in ("blur_size", "open_size", "close_size"):
        validate_odd_size(name, int(getattr(args, name)))
    if args.area_mode == "depth":
        if args.reference_depth <= 0 or not 0 < args.reference_area_ratio < 1:
            raise ValueError("Depth-aware reference depth must be positive and area ratio in (0, 1)")
        if not 0 < args.dynamic_min_area_ratio <= args.dynamic_max_area_ratio <= 1:
            raise ValueError("Depth-aware area limits must satisfy 0 < min <= max <= 1")
        if args.region_depth_erode < 0:
            raise ValueError("--region-depth-erode must be nonnegative")


def run_video(args: argparse.Namespace) -> Path:
    """Run the shared video loop with a replaceable per-frame analyzer."""
    args.normal_frame = args.normal_frame.resolve()
    args.input_path = args.input_path.resolve()
    validate_args(args, video_mode=True)

    run_dir = make_run_directory(args.output_root.resolve(), args.input_path)
    alert_dir = run_dir / "alert_frames"
    if args.save_alert_frames:
        alert_dir.mkdir()

    print(f"Run directory: {run_dir}")
    timer_start = time.perf_counter()
    depth_model = load_depth_model(args.model_dir, args.device)
    model_load_seconds = time.perf_counter() - timer_start
    print(f"Loading YOLO Seg model: {args.yolo_model}")
    timer_start = time.perf_counter()
    person_segmenter = UltralyticsPersonSegmenter(
        args.yolo_model,
        person_class_id=args.person_class_id,
        confidence=args.yolo_confidence,
        device=args.yolo_device,
        image_size=args.yolo_image_size,
    )
    yolo_load_seconds = time.perf_counter() - timer_start

    timer_start = time.perf_counter()
    normal_bgr = load_bgr_image(args.normal_frame)
    normal_depth = infer_depth(depth_model, normal_bgr, args.process_res)
    reference_depth_seconds = time.perf_counter() - timer_start
    timer_start = time.perf_counter()
    normal_person_mask = person_segmenter.predict_mask(
        normal_bgr, normal_depth.shape
    )
    reference_segmentation_seconds = time.perf_counter() - timer_start

    capture = cv2.VideoCapture(str(args.input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.input_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    output_video = run_dir / "annotated.mp4"
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*args.codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not create output video with codec {args.codec}: {output_video}"
        )

    summary: dict[str, object] = {
        "normal_frame": str(args.normal_frame),
        "video": str(args.input_path),
        "output_video": str(output_video),
        "video_fps": fps,
        "video_frame_count": total_frames,
        "processed_frames": 0,
        "detected_frames": 0,
        "person_occlusion_frames": 0,
        "depth_occlusion_frames": 0,
        "max_person_area_ratio": 0.0,
        "elapsed_seconds": 0.0,
        "area_mode": getattr(args, "area_mode", "fixed"),
        "timing_seconds": {
            "model_load": model_load_seconds,
            "yolo_load": yolo_load_seconds,
            "reference_depth": reference_depth_seconds,
            "reference_segmentation": reference_segmentation_seconds,
        },
    }

    start_time = time.perf_counter()
    frame_index = 0
    total_processing_seconds = 0.0
    stage_totals = {"depth": 0.0, "segmentation": 0.0, "comparison": 0.0}
    results_path = run_dir / "frame_results.jsonl"
    progress_total = min(total_frames, args.max_frames) if args.max_frames else total_frames
    progress = tqdm(
        total=progress_total or None,
        desc="Detecting occlusion",
        unit="frame",
    )
    try:
        with results_path.open("w", encoding="utf-8") as result_file:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                if args.max_frames and frame_index >= args.max_frames:
                    break

                if frame_index % args.frame_step != 0:
                    writer.write(frame_bgr)
                    frame_index += 1
                    progress.update(1)
                    continue

                frame_start = time.perf_counter()
                current_depth = infer_depth(depth_model, frame_bgr, args.process_res)
                depth_seconds = time.perf_counter() - frame_start
                timer_start = time.perf_counter()
                current_person_mask = person_segmenter.predict_mask(
                    frame_bgr, normal_depth.shape
                )
                segmentation_seconds = time.perf_counter() - timer_start
                timer_start = time.perf_counter()
                overlay, record, final_mask, ignored_person_mask = analyze_frame(
                    normal_depth,
                    current_depth,
                    normal_person_mask,
                    current_person_mask,
                    frame_bgr,
                    args,
                )
                comparison_seconds = time.perf_counter() - timer_start
                record["frame_index"] = frame_index
                record["timestamp_seconds"] = frame_index / fps
                record["processing_seconds"] = depth_seconds + segmentation_seconds + comparison_seconds
                record["timing_seconds"] = {
                    "depth": depth_seconds,
                    "segmentation": segmentation_seconds,
                    "comparison": comparison_seconds,
                }
                result_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                summary["processed_frames"] = int(summary["processed_frames"]) + 1
                total_processing_seconds += float(record["processing_seconds"])
                stage_totals["depth"] += depth_seconds
                stage_totals["segmentation"] += segmentation_seconds
                stage_totals["comparison"] += comparison_seconds
                summary["max_person_area_ratio"] = max(
                    float(summary["max_person_area_ratio"]),
                    float(record["person_area_ratio"]),
                )
                if bool(record["detected"]):
                    summary["detected_frames"] = int(summary["detected_frames"]) + 1
                    if args.save_alert_frames:
                        cv2.imwrite(
                            str(alert_dir / f"frame_{frame_index:08d}.png"), overlay
                        )
                        cv2.imwrite(
                            str(alert_dir / f"frame_{frame_index:08d}_mask.png"),
                            final_mask,
                        )
                        cv2.imwrite(
                            str(
                                alert_dir
                                / f"frame_{frame_index:08d}_ignored_person.png"
                            ),
                            ignored_person_mask,
                        )
                if bool(record["person_occlusion_detected"]):
                    summary["person_occlusion_frames"] = int(
                        summary["person_occlusion_frames"]
                    ) + 1
                if bool(record["depth_occlusion_detected"]):
                    summary["depth_occlusion_frames"] = int(
                        summary["depth_occlusion_frames"]
                    ) + 1

                writer.write(overlay)
                frame_index += 1
                progress.update(1)
    finally:
        progress.close()
        capture.release()
        writer.release()

    summary["elapsed_seconds"] = time.perf_counter() - start_time
    summary["input_frames_read"] = frame_index
    summary["average_processing_seconds"] = (
        total_processing_seconds / int(summary["processed_frames"])
        if summary["processed_frames"]
        else 0.0
    )
    summary["timing_seconds"]["average_per_analyzed_frame"] = {
        name: total / int(summary["processed_frames"])
        if summary["processed_frames"]
        else 0.0
        for name, total in stage_totals.items()
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== Video occlusion detection completed ===")
    print(f"Input frames read: {frame_index}")
    print(f"Frames analyzed: {summary['processed_frames']}")
    print(f"Detected frames: {summary['detected_frames']}")
    print(f"Average analyzed-frame time: {summary['average_processing_seconds']:.3f}s")
    timing = summary["timing_seconds"]
    averages = timing["average_per_analyzed_frame"]
    print(
        "Timing: "
        f"model-load={timing['model_load']:.3f}s, "
        f"YOLO-load={timing['yolo_load']:.3f}s, "
        f"reference-depth={timing['reference_depth']:.3f}s, "
        f"reference-segmentation={timing['reference_segmentation']:.3f}s"
    )
    print(
        "Average stages: "
        f"depth={averages['depth']:.3f}s, "
        f"segmentation={averages['segmentation']:.3f}s, "
        f"comparison={averages['comparison']:.3f}s"
    )
    print(f"Annotated video: {output_video}")
    print(f"Frame results: {results_path}")
    print(f"Summary: {run_dir / 'summary.json'}")
    return run_dir


def run_image(args: argparse.Namespace) -> Path:
    """Analyze one image against its normal/reference image."""
    args.normal_frame = args.normal_frame.resolve()
    args.input_path = args.input_path.resolve()
    validate_args(args, video_mode=False)
    timings: dict[str, float] = {}
    timer_start = time.perf_counter()
    model = load_depth_model(args.model_dir, args.device)
    timings["model_load"] = time.perf_counter() - timer_start
    timer_start = time.perf_counter()
    normal_bgr = load_bgr_image(args.normal_frame)
    current_bgr = load_bgr_image(args.input_path)
    timings["image_load"] = time.perf_counter() - timer_start
    timer_start = time.perf_counter()
    normal_depth = infer_depth(model, normal_bgr, args.process_res)
    timings["reference_depth"] = time.perf_counter() - timer_start
    timer_start = time.perf_counter()
    current_depth = infer_depth(model, current_bgr, args.process_res)
    timings["current_depth"] = time.perf_counter() - timer_start
    if args.yolo_model:
        timer_start = time.perf_counter()
        segmenter = UltralyticsPersonSegmenter(
            args.yolo_model, person_class_id=args.person_class_id,
            confidence=args.yolo_confidence, device=args.yolo_device, image_size=args.yolo_image_size,
        )
        timings["yolo_load"] = time.perf_counter() - timer_start
        timer_start = time.perf_counter()
        normal_person_mask = segmenter.predict_mask(normal_bgr, normal_depth.shape)
        current_person_mask = segmenter.predict_mask(current_bgr, current_depth.shape)
        timings["segmentation"] = time.perf_counter() - timer_start
    else:
        timer_start = time.perf_counter()
        normal_person_mask = load_yolo_seg_class_mask(args.normal_yolo_txt.resolve(), normal_depth.shape, args.person_class_id)
        current_person_mask = load_yolo_seg_class_mask(args.current_yolo_txt.resolve(), current_depth.shape, args.person_class_id)
        timings["segmentation"] = time.perf_counter() - timer_start
    timer_start = time.perf_counter()
    overlay, record, final_mask, ignored_mask = analyze_frame(
        normal_depth, current_depth, normal_person_mask, current_person_mask, current_bgr, args
    )
    timings["comparison"] = time.perf_counter() - timer_start
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "occlusion_overlay.png"), overlay)
    cv2.imwrite(str(output_dir / "occlusion_mask.png"), final_mask)
    cv2.imwrite(str(output_dir / "ignored_person_mask.png"), ignored_mask)
    record["timing_seconds"] = timings
    (output_dir / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    print("Timing:", ", ".join(f"{name}={seconds:.3f}s" for name, seconds in timings.items()))
    print(f"Saved results: {output_dir}")
    return output_dir


def main() -> None:
    args = parse_args()
    # DA3 logs preprocessing, forward-pass, and conversion timing for every
    # frame at INFO level. The script has its own concise timing summary.
    logger.level = LOG_LEVELS["INFO"] if args.verbose else LOG_LEVELS["WARN"]
    video_suffixes = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
    input_type = args.input_type
    if input_type == "auto":
        input_type = "video" if args.input_path.suffix.lower() in video_suffixes else "image"
    if input_type == "video":
        run_video(args)
    else:
        run_image(args)


if __name__ == "__main__":
    main()
