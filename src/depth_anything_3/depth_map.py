"""Compare two Depth Anything 3 depth result files.

The default paths match the two metric-depth results in this workspace. Other
files can be supplied on the command line; see ``python depth_map.py --help``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEPTH1 = (
    PROJECT_ROOT / "outputs/metric-result1/exports/mini_npz/results.npz"
)
DEFAULT_DEPTH2 = (
    PROJECT_ROOT / "outputs/metric-result2/exports/mini_npz/results.npz"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/depth_comparison.png"


def load_depth(path: Path, index: int = 0) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Depth result not found: {path}")

    if path.suffix.lower() == ".npy":
        depth = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "depth" not in data:
                raise KeyError(f"{path} does not contain a 'depth' array")
            depth = np.asarray(data["depth"], dtype=np.float32)
    else:
        raise ValueError(f"Only .npy and .npz are supported: {path}")

    if depth.ndim == 2:
        return depth

    if depth.ndim == 3:
        if not 0 <= index < depth.shape[0]:
            raise IndexError(
                f"Index {index} is out of range; shape={depth.shape}"
            )
        return depth[index]

    raise ValueError(
        f"Expected (H, W) or (N, H, W), got {depth.shape}"
    )

def finite_percentiles(*depths: np.ndarray) -> tuple[float, float]:
    """Return robust shared display limits for multiple depth maps."""
    values = np.concatenate([depth[np.isfinite(depth)] for depth in depths])
    if values.size == 0:
        raise ValueError("The depth maps contain no finite values")
    low, high = np.percentile(values, (2, 98))
    if low == high:
        high = low + 1e-6
    return float(low), float(high)


def compare_depths(
    depth1: np.ndarray,
    depth2: np.ndarray,
    output_path: Path,
    show: bool = True,
) -> None:
    """Plot the two depths and the signed difference (depth2 - depth1)."""
    if depth1.shape != depth2.shape:
        raise ValueError(
            "The two depth maps must have the same size for pixel-wise comparison, "
            f"but got {depth1.shape} and {depth2.shape}. "
            "Use images with the same resolution and align them before inference."
        )

    valid = np.isfinite(depth1) & np.isfinite(depth2)
    difference = np.full(depth1.shape, np.nan, dtype=np.float32)
    difference[valid] = depth2[valid] - depth1[valid]

    depth_min, depth_max = finite_percentiles(depth1, depth2)
    finite_diff = np.abs(difference[np.isfinite(difference)])
    if finite_diff.size == 0:
        raise ValueError("The two depth maps have no mutually valid pixels")
    diff_limit = float(np.percentile(finite_diff, 98))
    if diff_limit == 0:
        diff_limit = 1e-6

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    depth_image1 = axes[0].imshow(
        depth1, cmap="Spectral_r", vmin=depth_min, vmax=depth_max
    )
    axes[0].set_title("Depth 1")
    fig.colorbar(depth_image1, ax=axes[0], label="Predicted depth")

    depth_image2 = axes[1].imshow(
        depth2, cmap="Spectral_r", vmin=depth_min, vmax=depth_max
    )
    axes[1].set_title("Depth 2")
    fig.colorbar(depth_image2, ax=axes[1], label="Predicted depth")

    diff_image = axes[2].imshow(
        difference,
        cmap="RdBu_r",
        vmin=-diff_limit,
        vmax=diff_limit,
    )
    axes[2].set_title("Depth difference (Depth 2 - Depth 1)")
    fig.colorbar(diff_image, ax=axes[2], label="Depth difference")

    for axis in axes:
        axis.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Saved comparison image to: {output_path}")
    print(
        "Difference statistics: "
        f"median={np.nanmedian(difference):.4f}, "
        f"mean={np.nanmean(difference):.4f}, "
        f"min={np.nanmin(difference):.4f}, "
        f"max={np.nanmax(difference):.4f}"
    )

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw two DA3 depth maps and their signed difference."
    )
    parser.add_argument(
        "--depth1", type=Path, default=DEFAULT_DEPTH1, help="First results.npz"
    )
    parser.add_argument(
        "--depth2", type=Path, default=DEFAULT_DEPTH2, help="Second results.npz"
    )
    parser.add_argument(
        "--index1", type=int, default=0, help="Depth index in the first npz"
    )
    parser.add_argument(
        "--index2", type=int, default=0, help="Depth index in the second npz"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path"
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Save the figure without opening a window"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    depth1 = load_depth(args.depth1, args.index1)
    depth2 = load_depth(args.depth2, args.index2)
    compare_depths(depth1, depth2, args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
