"""Infer and save a Depth Anything 3 depth map for one image.

The existing DA3 export functions save the raw depth map in a compressed NPZ
file and a JPG preview beside the processed input image.

Example:
    uv run python -m depth_anything_3.export_depth_map assets/images/image.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

from depth_anything_3.utils.depth_analysis import load_depth_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input image")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for the depth-map files "
            "(default: outputs/depth_maps/<image filename>)"
        ),
    )
    parser.add_argument(
        "--model-dir",
        default="depth-anything/DA3METRIC-LARGE",
        help="Depth Anything 3 model ID or local model directory",
    )
    parser.add_argument("--device", default="cuda", help="Inference device")
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="Maximum processing resolution (default: 504)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.process_res < 1:
        raise ValueError("--process-res must be positive")
    image_path = args.image.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Could not read image: {image_path}")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else PROJECT_ROOT / "outputs" / "depth_maps" / image_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_depth_model(args.model_dir, args.device)
    print(f"Inferring depth: {image_path}")
    # Reuse DA3's exporter: mini_npz writes raw depth and depth_vis writes a
    # colour preview alongside the processed source image.
    model.inference(
        [str(image_path)],
        process_res=args.process_res,
        process_res_method="upper_bound_resize",
        export_dir=str(output_dir),
        export_format="mini_npz-depth_vis",
    )

    print("Saved raw depth map:", output_dir / "exports" / "mini_npz" / "results.npz")
    print("Saved depth preview:", output_dir / "depth_vis" / "0000.jpg")


if __name__ == "__main__":
    main()
