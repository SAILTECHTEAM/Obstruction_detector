"""Shared PyTorch Profiler runner for Depth Anything 3 models."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from depth_anything_3.api import DepthAnything3


def build_parser(model_id: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Profile {model_id} on one image with PyTorch Profiler."
    )
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="DA3 processing resolution; use the same value for both models",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of unmeasured warm-up runs",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="Number of runs used for latency statistics",
    )
    parser.add_argument(
        "--profile-runs",
        type=int,
        default=3,
        help="Number of extra runs recorded by PyTorch Profiler",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/profiler"),
        help="Directory for JSON summary, text report, and Chrome trace",
    )
    return parser


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_once(model: DepthAnything3, image: Path, process_res: int) -> None:
    model.inference(
        [str(image)],
        process_res=process_res,
        process_res_method="upper_bound_resize",
        export_dir=None,
    )


def profile_model(model_id: str, label: str) -> None:
    args = build_parser(model_id).parse_args()
    image = args.image.resolve()
    if not image.is_file():
        raise FileNotFoundError(f"Input image not found: {image}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this profiler is intended for GPU comparison")
    if args.warmup < 0 or args.repeat < 1 or args.profile_runs < 1:
        raise ValueError("warmup must be >= 0; repeat and profile-runs must be >= 1")

    device = torch.device("cuda")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_id}")
    print(f"Image: {image}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Process resolution: {args.process_res}")

    synchronize()
    load_start = time.perf_counter()
    model = DepthAnything3.from_pretrained(model_id).to(device).eval()
    synchronize()
    load_seconds = time.perf_counter() - load_start
    model_memory_mb = torch.cuda.memory_allocated(device) / 1024**2
    print(f"Model load time: {load_seconds:.3f} s")
    print(f"GPU memory after model load: {model_memory_mb:.1f} MiB")

    print(f"Warming up ({args.warmup} run(s))...")
    for _ in range(args.warmup):
        run_once(model, image, args.process_res)
    synchronize()

    torch.cuda.reset_peak_memory_stats(device)
    latencies_ms: list[float] = []
    print(f"Measuring latency ({args.repeat} run(s))...")
    for _ in range(args.repeat):
        synchronize()
        start = time.perf_counter()
        run_once(model, image, args.process_res)
        synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000)

    peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    extra_peak_memory_mb = max(0.0, peak_memory_mb - model_memory_mb)

    print(f"Recording operator trace ({args.profile_runs} run(s))...")
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        for _ in range(args.profile_runs):
            with record_function(f"{label}_inference"):
                run_once(model, image, args.process_res)
                synchronize()
            profiler.step()

    trace_path = output_dir / f"{label}_trace.json"
    report_path = output_dir / f"{label}_operators.txt"
    summary_path = output_dir / f"{label}_summary.json"

    profiler.export_chrome_trace(str(trace_path))
    operator_report = profiler.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=50
    )
    report_path.write_text(operator_report, encoding="utf-8")

    summary = {
        "model": model_id,
        "image": str(image),
        "gpu": torch.cuda.get_device_name(device),
        "process_resolution": args.process_res,
        "warmup_runs": args.warmup,
        "timed_runs": args.repeat,
        "model_load_seconds": load_seconds,
        "latencies_ms": latencies_ms,
        "latency_mean_ms": statistics.fmean(latencies_ms),
        "latency_median_ms": statistics.median(latencies_ms),
        "latency_p95_ms": float(np.percentile(latencies_ms, 95)),
        "gpu_memory_after_load_mib": model_memory_mb,
        "gpu_peak_memory_mib": peak_memory_mb,
        "gpu_extra_peak_during_inference_mib": extra_peak_memory_mb,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nTiming result (Profiler overhead excluded)")
    print(f"Mean:   {summary['latency_mean_ms']:.3f} ms")
    print(f"Median: {summary['latency_median_ms']:.3f} ms")
    print(f"P95:    {summary['latency_p95_ms']:.3f} ms")
    print(f"GPU peak memory: {peak_memory_mb:.1f} MiB")
    print(f"Extra inference peak: {extra_peak_memory_mb:.1f} MiB")
    print(f"Summary: {summary_path}")
    print(f"Operators: {report_path}")
    print(f"Chrome trace: {trace_path}")

