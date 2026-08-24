"""Detect depth occlusions in a video using a separate normal reference frame."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from depth_anything_3.api import DepthAnything3
from depth_anything_3.detect_depth_occlusion import (
    clean_mask,
    dilate_mask,
    draw_overlay,
    find_large_regions,
    mask_region,
    validate_odd_size,
)
from depth_anything_3.person_segmentation import UltralyticsPersonSegmenter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/video_occlusion_runs"


def load_bgr_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read normal frame: {path}")
    return image


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


def infer_depth(
    model: DepthAnything3,
    image_bgr: np.ndarray,
    process_res: int,
) -> np.ndarray:
    # DA3 converts NumPy arrays through PIL and therefore expects RGB channel order.
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    prediction = model.inference(
        [image_rgb],
        process_res=process_res,
        process_res_method="upper_bound_resize",
        export_dir=None,
        include_processed_images=False,
    )
    depth = np.asarray(prediction.depth[0], dtype=np.float32).copy()
    del prediction
    return depth


def analyze_frame(
    normal_depth: np.ndarray,
    current_depth: np.ndarray,
    normal_person_mask: np.ndarray,
    current_person_mask: np.ndarray,
    frame_bgr: np.ndarray,
    args: argparse.Namespace,
    region_finder=None,
) -> tuple[np.ndarray, dict[str, object], np.ndarray, np.ndarray]:
    if current_depth.shape != normal_depth.shape:
        raise ValueError(
            "Normal frame and video frame produced different depth shapes: "
            f"{normal_depth.shape} vs {current_depth.shape}. "
            "Use a normal frame with the same aspect ratio as the video."
        )

    valid = (
        np.isfinite(normal_depth)
        & np.isfinite(current_depth)
        & (normal_depth > 0)
        & (current_depth > 0)
    )
    difference = np.full(normal_depth.shape, np.nan, dtype=np.float32)
    difference[valid] = current_depth[valid] - normal_depth[valid]

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

    if region_finder is None:
        regions, large_depth_mask = find_large_regions(
            mask_u8, args.min_area_ratio
        )
    else:
        regions, large_depth_mask = region_finder(mask_u8, current_depth, args)
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
    parser.add_argument("normal_frame", type=Path, help="Normal reference image")
    parser.add_argument("video", type=Path, help="Input video")
    parser.add_argument(
        "--yolo-model",
        required=True,
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
    parser.add_argument("--person-alert-ratio", type=float, default=1.0 / 3.0)
    parser.add_argument("--person-dilate", type=int, default=7)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--depth-threshold", type=float, default=0.5)
    parser.add_argument("--min-area-ratio", type=float, default=0.03)
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
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.normal_frame.is_file():
        raise FileNotFoundError(f"Normal frame not found: {args.normal_frame}")
    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for DA3 but is not available")
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


def run_video(args: argparse.Namespace, frame_analyzer=analyze_frame) -> Path:
    """Run the shared video loop with a replaceable per-frame analyzer."""
    args.normal_frame = args.normal_frame.resolve()
    args.video = args.video.resolve()
    validate_args(args)

    run_dir = make_run_directory(args.output_root.resolve(), args.video)
    alert_dir = run_dir / "alert_frames"
    if args.save_alert_frames:
        alert_dir.mkdir()

    print(f"Run directory: {run_dir}")
    print(f"Loading DA3 model: {args.model_dir}")
    depth_model = DepthAnything3.from_pretrained(args.model_dir).to(args.device).eval()
    print(f"Loading YOLO Seg model: {args.yolo_model}")
    person_segmenter = UltralyticsPersonSegmenter(
        args.yolo_model,
        person_class_id=args.person_class_id,
        confidence=args.yolo_confidence,
        device=args.yolo_device,
        image_size=args.yolo_image_size,
    )

    normal_bgr = load_bgr_image(args.normal_frame)
    print("Inferring normal-frame depth and person mask...")
    normal_depth = infer_depth(depth_model, normal_bgr, args.process_res)
    normal_person_mask = person_segmenter.predict_mask(
        normal_bgr, normal_depth.shape
    )

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

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
        "video": str(args.video),
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
    }

    start_time = time.perf_counter()
    frame_index = 0
    results_path = run_dir / "frame_results.jsonl"
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
                    continue

                frame_start = time.perf_counter()
                current_depth = infer_depth(depth_model, frame_bgr, args.process_res)
                current_person_mask = person_segmenter.predict_mask(
                    frame_bgr, normal_depth.shape
                )
                overlay, record, final_mask, ignored_person_mask = frame_analyzer(
                    normal_depth,
                    current_depth,
                    normal_person_mask,
                    current_person_mask,
                    frame_bgr,
                    args,
                )
                record["frame_index"] = frame_index
                record["timestamp_seconds"] = frame_index / fps
                record["processing_seconds"] = time.perf_counter() - frame_start
                result_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                summary["processed_frames"] = int(summary["processed_frames"]) + 1
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
                if int(summary["processed_frames"]) % 10 == 0:
                    print(
                        f"Processed {summary['processed_frames']} frame(s); "
                        f"input frame {frame_index}/{total_frames or '?'}"
                    )
                frame_index += 1
    finally:
        capture.release()
        writer.release()

    summary["elapsed_seconds"] = time.perf_counter() - start_time
    summary["input_frames_read"] = frame_index
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== Video occlusion detection completed ===")
    print(f"Input frames read: {frame_index}")
    print(f"Frames analyzed: {summary['processed_frames']}")
    print(f"Detected frames: {summary['detected_frames']}")
    print(f"Annotated video: {output_video}")
    print(f"Frame results: {results_path}")
    print(f"Summary: {run_dir / 'summary.json'}")
    return run_dir


def main() -> None:
    run_video(parse_args())


if __name__ == "__main__":
    main()
