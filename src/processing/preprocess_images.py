"""
Preprocess SAR images for YOLOv8-OBB ship-wake detection.

Applies three stages in order:
  1. Lee speckle filter  – reduces multiplicative SAR speckle noise
  2. CLAHE               – enhances local contrast to reveal faint wakes
  4. Per-image normalisation – rescales to full [0, 255] uint8 range

The preprocessed images are written to a parallel directory structure so the
originals are preserved. Update dataset.yaml paths to point at the new dirs.

Usage::

    python src/processing/preprocess_images.py \
        --input-dir  data/splits/train/images \
        --output-dir data/splits_preprocessed/train/images

    # Process all splits at once:
    python src/processing/preprocess_images.py --all-splits
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import uniform_filter, uniform_filter1d

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
# 4. Per-image normalisation
# ---------------------------------------------------------------------------

def normalise_minmax(img: np.ndarray) -> np.ndarray:
    """Rescale image intensities to the full [0, 255] uint8 range.

    Args:
        img: 2-D image (any numeric type).

    Returns:
        Min-max normalised uint8 image.
    """
    img = img.astype(np.float64)
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    normalised = (img - lo) / (hi - lo) * 255.0
    return normalised.astype(np.uint8)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess_image(
    img_bgr: np.ndarray,
    lee_win: int = 7,
    clahe_clip: float = 3.0,
    clahe_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Run the full preprocessing pipeline on a single image.

    Pipeline order: Lee filter -> CLAHE -> min-max normalisation.
    Output is a 3-channel BGR image (grayscale duplicated) for YOLOv8
    compatibility.

    Args:
        img_bgr: Input BGR image as loaded by cv2.imread.
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

    # 4. Per-image normalisation (to uint8 for CLAHE input)
    normed = normalise_minmax(filtered)

    # 2. CLAHE (requires uint8)
    enhanced = apply_clahe(normed, clip_limit=clahe_clip, tile_grid_size=clahe_grid)

    # Duplicate to 3-channel BGR for YOLOv8
    return cv2.merge([enhanced, enhanced, enhanced])


def process_directory(
    input_dir: Path,
    output_dir: Path,
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
            img, lee_win=lee_win, clahe_clip=clahe_clip, clahe_grid=clahe_grid
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
        description="Preprocess SAR images: Lee filter + CLAHE + normalisation."
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
    args = parser.parse_args()

    grid = (args.clahe_grid, args.clahe_grid)
    kwargs = dict(lee_win=args.lee_win, clahe_clip=args.clahe_clip, clahe_grid=grid)

    if args.all_splits:
        base = Path("data/splits")
        out_base = Path("data/splits_preprocessed")
        for split in ["train", "val", "test"]:
            src = base / split / "images"
            if src.exists():
                logger.info("--- Processing %s ---", split)
                # Copy labels alongside preprocessed images
                dst = out_base / split / "images"
                process_directory(src, dst, **kwargs)
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
    elif args.input_dir and args.output_dir:
        process_directory(args.input_dir, args.output_dir, **kwargs)
    else:
        parser.error("Provide --input-dir and --output-dir, or use --all-splits")


if __name__ == "__main__":
    main()
