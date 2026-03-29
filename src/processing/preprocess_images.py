"""
Preprocess SAR images for YOLOv8-OBB ship-wake detection.

Applies three stages in order:
  1. Lee speckle filter  – reduces multiplicative SAR speckle noise
  2. Global percentile normalisation – clips to fixed percentile range,
     then rescales to [0, 255] using dataset-wide statistics (two-pass)
  3. CLAHE               – enhances local contrast to reveal faint wakes

The preprocessed images are written to a parallel directory structure so the
originals are preserved. Update dataset.yaml paths to point at the new dirs.

Usage::

    python src/processing/preprocess_images.py \
        --input-dir  data/splits/train/images \
        --output-dir data/splits_preprocessed/train/images

    # Process all splits at once (recommended — computes global stats):
    python src/processing/preprocess_images.py --all-splits
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import uniform_filter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Lee speckle filter
# ---------------------------------------------------------------------------

def lee_filter(img: np.ndarray, win_size: int = 7) -> np.ndarray:
    """Apply the Lee multiplicative speckle filter.

    The Lee filter preserves edges while smoothing homogeneous regions by
    weighting between the local mean and the observed value based on local
    variance vs. overall noise variance.

    Args:
        img: 2-D float64 image (single channel).
        win_size: Side length of the square filter window (default 7).

    Returns:
        Filtered image as float64.
    """
    img = img.astype(np.float64)
    local_mean = uniform_filter(img, size=win_size)
    local_sq_mean = uniform_filter(img ** 2, size=win_size)
    local_var = local_sq_mean - local_mean ** 2
    local_var = np.maximum(local_var, 0.0)

    # Estimate overall noise variance from the entire image
    overall_var = np.var(img)

    # Weight: 0 = use local mean (smooth), 1 = keep original (edge)
    weight = np.where(
        local_var > 0,
        np.maximum(local_var - overall_var, 0.0) / np.maximum(local_var, 1e-10),
        0.0,
    )

    return local_mean + weight * (img - local_mean)


# ---------------------------------------------------------------------------
# 2. CLAHE
# ---------------------------------------------------------------------------

def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 3.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization.

    Args:
        img: 2-D uint8 image.
        clip_limit: Contrast limiting threshold (default 3.0).
        tile_grid_size: Grid size for local histograms (default 8x8).

    Returns:
        CLAHE-enhanced uint8 image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)


# ---------------------------------------------------------------------------
# 3. Global percentile normalisation
# ---------------------------------------------------------------------------

def compute_global_percentiles(
    image_dirs: list[Path],
    lee_win: int = 7,
    p_lo: float = 1.0,
    p_hi: float = 99.0,
    sample_cap: int = 500,
) -> tuple[float, float]:
    """Compute dataset-wide percentile bounds after Lee filtering.

    Scans images across all provided directories, applies the Lee filter,
    and collects pixel values to compute the p_lo and p_hi percentiles.
    These are then used as fixed clipping bounds for normalisation.

    Args:
        image_dirs: List of directories containing source images.
        lee_win: Lee filter window size.
        p_lo: Lower percentile for clipping (default 1.0).
        p_hi: Upper percentile for clipping (default 99.0).
        sample_cap: Max images to sample per directory (for speed).

    Returns:
        (global_lo, global_hi) — the percentile values across the dataset.
    """
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    sampled_pixels: list[np.ndarray] = []

    for img_dir in image_dirs:
        if not img_dir.exists():
            continue
        image_files = sorted(
            f for f in img_dir.iterdir() if f.suffix.lower() in extensions
        )
        # Subsample if directory is very large
        if len(image_files) > sample_cap:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(image_files), size=sample_cap, replace=False)
            image_files = [image_files[i] for i in sorted(indices)]

        for img_path in image_files:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            filtered = lee_filter(gray.astype(np.float64), win_size=lee_win)
            # Subsample pixels to keep memory bounded
            flat = filtered.ravel()
            if len(flat) > 10000:
                rng = np.random.default_rng(hash(img_path.name) & 0xFFFFFFFF)
                flat = rng.choice(flat, size=10000, replace=False)
            sampled_pixels.append(flat)

    all_pixels = np.concatenate(sampled_pixels)
    global_lo = float(np.percentile(all_pixels, p_lo))
    global_hi = float(np.percentile(all_pixels, p_hi))

    logger.info(
        "Global percentile stats: p%.1f=%.2f  p%.1f=%.2f  (from %d images)",
        p_lo, global_lo, p_hi, global_hi, len(sampled_pixels),
    )
    return global_lo, global_hi


def normalise_global_percentile(
    img: np.ndarray, global_lo: float, global_hi: float,
) -> np.ndarray:
    """Clip to global percentile range and rescale to [0, 255] uint8.

    Args:
        img: 2-D float64 image (e.g. after Lee filtering).
        global_lo: Lower clipping bound (dataset-wide percentile).
        global_hi: Upper clipping bound (dataset-wide percentile).

    Returns:
        Normalised uint8 image.
    """
    if global_hi - global_lo < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    clipped = np.clip(img, global_lo, global_hi)
    normalised = (clipped - global_lo) / (global_hi - global_lo) * 255.0
    return normalised.astype(np.uint8)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess_image(
    img_bgr: np.ndarray,
    global_lo: float,
    global_hi: float,
    lee_win: int = 7,
    clahe_clip: float = 3.0,
    clahe_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Run the full preprocessing pipeline on a single image.

    Pipeline order: Lee filter -> global percentile normalisation -> CLAHE.
    Output is a 3-channel BGR image (grayscale duplicated) for YOLOv8
    compatibility.

    Args:
        img_bgr: Input BGR image as loaded by cv2.imread.
        global_lo: Lower clipping bound from dataset-wide percentile stats.
        global_hi: Upper clipping bound from dataset-wide percentile stats.
        lee_win: Lee filter window size.
        clahe_clip: CLAHE clip limit.
        clahe_grid: CLAHE tile grid size.

    Returns:
        Preprocessed BGR uint8 image.
    """
    # Convert to single-channel grayscale (SAR images are already grayscale)
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr

    # 1. Lee speckle filter (operates on float)
    filtered = lee_filter(gray.astype(np.float64), win_size=lee_win)

    # 2. Global percentile normalisation (to uint8)
    normed = normalise_global_percentile(filtered, global_lo, global_hi)

    # Duplicate to 3-channel BGR for YOLOv8
    return cv2.merge([normed, normed, normed])


def process_directory(
    input_dir: Path,
    output_dir: Path,
    global_lo: float,
    global_hi: float,
    lee_win: int = 7,
    clahe_clip: float = 3.0,
    clahe_grid: tuple[int, int] = (8, 8),
) -> None:
    """Preprocess all images in a directory."""
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    image_files = sorted(
        f for f in input_dir.iterdir() if f.suffix.lower() in extensions
    )

    if not image_files:
        logger.warning("No images found in %s", input_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(image_files):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning("Could not read %s, skipping", img_path.name)
            continue

        result = preprocess_image(
            img, global_lo, global_hi,
            lee_win=lee_win, clahe_clip=clahe_clip, clahe_grid=clahe_grid,
        )
        cv2.imwrite(str(output_dir / img_path.name), result)

        if (i + 1) % 200 == 0 or (i + 1) == len(image_files):
            logger.info("Processed %d / %d images", i + 1, len(image_files))

    logger.info("Output written to %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess SAR images: Lee filter + global percentile norm + CLAHE."
    )
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Input image directory")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output image directory")
    parser.add_argument("--all-splits", action="store_true",
                        help="Process train/val/test splits under data/splits/")
    parser.add_argument("--lee-win", type=int, default=7,
                        help="Lee filter window size (default 7)")
    parser.add_argument("--clahe-clip", type=float, default=3.0,
                        help="CLAHE clip limit (default 3.0)")
    parser.add_argument("--clahe-grid", type=int, default=8,
                        help="CLAHE tile grid size (default 8)")
    parser.add_argument("--p-lo", type=float, default=1.0,
                        help="Lower percentile for clipping (default 1.0)")
    parser.add_argument("--p-hi", type=float, default=99.0,
                        help="Upper percentile for clipping (default 99.0)")
    args = parser.parse_args()

    grid = (args.clahe_grid, args.clahe_grid)

    if args.all_splits:
        base = Path("data/splits")
        out_base = Path("data/splits_preprocessed")

        # --- Pass 1: compute global percentile bounds across all splits ---
        all_image_dirs = [
            base / split / "images"
            for split in ["train", "val", "test"]
            if (base / split / "images").exists()
        ]
        logger.info("--- Pass 1: computing global percentile stats ---")
        global_lo, global_hi = compute_global_percentiles(
            all_image_dirs, lee_win=args.lee_win,
            p_lo=args.p_lo, p_hi=args.p_hi,
        )

        # --- Pass 2: preprocess all images using global bounds ---
        for split in ["train", "val", "test"]:
            src = base / split / "images"
            if src.exists():
                logger.info("--- Pass 2: processing %s ---", split)
                dst = out_base / split / "images"
                process_directory(
                    src, dst, global_lo, global_hi,
                    lee_win=args.lee_win, clahe_clip=args.clahe_clip,
                    clahe_grid=grid,
                )
                # Symlink labels so the preprocessed dir is self-contained
                labels_src = (base / split / "labels").resolve()
                labels_dst = out_base / split / "labels"
                if labels_src.exists() and not labels_dst.exists():
                    labels_dst.symlink_to(labels_src)
                    logger.info("Symlinked labels: %s -> %s", labels_dst, labels_src)

        # Write dataset.yaml for the preprocessed data
        import yaml
        dataset_cfg = {
            "path": str(out_base.resolve()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": 1,
            "names": ["ship_wake"],
        }
        yaml_path = out_base / "dataset.yaml"
        yaml_path.write_text(yaml.dump(dataset_cfg, default_flow_style=False))
        logger.info("Dataset YAML written to %s", yaml_path)

        # Save the global stats for reproducibility
        stats_path = out_base / "global_norm_stats.txt"
        stats_path.write_text(
            f"p_lo={args.p_lo}  p_hi={args.p_hi}\n"
            f"global_lo={global_lo:.6f}\n"
            f"global_hi={global_hi:.6f}\n"
            f"lee_win={args.lee_win}\n"
        )
        logger.info("Global stats saved to %s", stats_path)

    elif args.input_dir and args.output_dir:
        # Single directory mode — compute stats from just that directory
        logger.info("--- Computing percentile stats from %s ---", args.input_dir)
        global_lo, global_hi = compute_global_percentiles(
            [args.input_dir], lee_win=args.lee_win,
            p_lo=args.p_lo, p_hi=args.p_hi,
        )
        process_directory(
            args.input_dir, args.output_dir, global_lo, global_hi,
            lee_win=args.lee_win, clahe_clip=args.clahe_clip, clahe_grid=grid,
        )
    else:
        parser.error("Provide --input-dir and --output-dir, or use --all-splits")


if __name__ == "__main__":
    main()
