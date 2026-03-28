"""
Convert OpenSARWake raw OBB labels to YOLOv8-OBB format.

Raw format (per line):
    x1 y1 x2 y2 x3 y3 x4 y4 class_name extra_field

YOLOv8-OBB format (per line):
    class_idx x1_norm y1_norm x2_norm y2_norm x3_norm y3_norm x4_norm y4_norm

Coordinates are normalised to [0, 1] by dividing by image dimensions.
Vertices that fall outside the image are clipped to [0, 1].

Additional preprocessing options:
  --clip          Clip OBB vertices to image bounds (default: on)
  --skip-oob      Skip annotations where >50% of area is outside image
  --min-area      Skip annotations smaller than this fraction of image area
  --verify-images Cross-check that every label has a matching image file

Usage::

    python src/processing/convert_labels.py \
        --labels-dir  data/splits/train/labels \
        --images-dir  data/splits/train/images \
        --img-size 1024 \
        --skip-oob --min-area 0.0005
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CLASS_MAP = {"ship_wake": 0}


def polygon_area(pts: np.ndarray) -> float:
    """Shoelace formula for the area of a polygon given Nx2 vertices."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def clip_polygon_to_unit(pts: np.ndarray) -> np.ndarray:
    """Clip polygon vertices to [0, 1]."""
    return np.clip(pts, 0.0, 1.0)


def convert_line(
    line: str,
    img_w: int,
    img_h: int,
    *,
    clip: bool = True,
    skip_oob_frac: float | None = 0.5,
    min_area_frac: float | None = None,
) -> str | None:
    """Convert one raw annotation line to YOLOv8-OBB format.

    Returns the converted line, or None if the annotation should be skipped.
    """
    parts = line.strip().split()
    if len(parts) < 9:
        return None

    coords_px = [float(v) for v in parts[:8]]
    class_name = parts[8]

    if class_name not in CLASS_MAP:
        logger.warning("Unknown class %r, skipping", class_name)
        return None

    class_idx = CLASS_MAP[class_name]

    # Normalise pixel coords to [0, 1]
    pts = np.array(coords_px).reshape(4, 2)
    pts[:, 0] /= img_w
    pts[:, 1] /= img_h

    # Calculate area before clipping for OOB check
    area_before = polygon_area(pts)

    if clip:
        pts = clip_polygon_to_unit(pts)

    area_after = polygon_area(pts)

    # Skip if too much of the annotation is outside the image
    if skip_oob_frac is not None and area_before > 0:
        visible = area_after / area_before
        if visible < (1.0 - skip_oob_frac):
            return None

    # Skip tiny annotations
    if min_area_frac is not None and area_after < min_area_frac:
        return None

    coords_str = " ".join(f"{v:.6f}" for v in pts.flatten())
    return f"{class_idx} {coords_str}"


def convert_label_file(
    label_path: Path,
    output_path: Path,
    img_w: int,
    img_h: int,
    **kwargs,
) -> dict:
    """Convert a single label file. Returns stats dict."""
    stats = {"total": 0, "kept": 0, "skipped": 0}

    lines = label_path.read_text().strip().splitlines()
    converted = []

    for line in lines:
        if not line.strip():
            continue
        stats["total"] += 1
        result = convert_line(line, img_w, img_h, **kwargs)
        if result is not None:
            converted.append(result)
            stats["kept"] += 1
        else:
            stats["skipped"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(converted) + "\n" if converted else "")
    return stats


def convert_split(
    labels_dir: Path,
    images_dir: Path | None,
    output_dir: Path,
    img_w: int,
    img_h: int,
    verify_images: bool = False,
    **kwargs,
) -> None:
    """Convert all label files in a directory."""
    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        logger.warning("No .txt files found in %s", labels_dir)
        return

    totals = {"total": 0, "kept": 0, "skipped": 0, "files": 0, "missing_img": 0}

    for lf in label_files:
        if verify_images and images_dir is not None:
            stem = lf.stem
            img_exists = any(
                (images_dir / f"{stem}{ext}").exists()
                for ext in [".png", ".jpg", ".jpeg", ".tif"]
            )
            if not img_exists:
                totals["missing_img"] += 1
                logger.warning("No image for label %s", lf.name)
                continue

        stats = convert_label_file(lf, output_dir / lf.name, img_w, img_h, **kwargs)
        totals["files"] += 1
        totals["total"] += stats["total"]
        totals["kept"] += stats["kept"]
        totals["skipped"] += stats["skipped"]

    logger.info(
        "Converted %d files: %d annotations kept, %d skipped, %d missing images",
        totals["files"],
        totals["kept"],
        totals["skipped"],
        totals["missing_img"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert OpenSARWake labels to YOLOv8-OBB format."
    )
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: overwrite in place)")
    parser.add_argument("--img-size", type=int, default=1024,
                        help="Image size (assumes square, default 1024)")
    parser.add_argument("--no-clip", action="store_true",
                        help="Do not clip vertices to image bounds")
    parser.add_argument("--skip-oob", action="store_true",
                        help="Skip annotations with >50%% area outside image")
    parser.add_argument("--min-area", type=float, default=None,
                        help="Skip annotations smaller than this fraction of image area")
    parser.add_argument("--verify-images", action="store_true",
                        help="Check that each label has a matching image")
    parser.add_argument("--backup", action="store_true",
                        help="Back up original labels before overwriting")
    args = parser.parse_args()

    output_dir = args.output_dir or args.labels_dir

    if args.backup and output_dir == args.labels_dir:
        backup_dir = args.labels_dir.parent / (args.labels_dir.name + "_backup")
        if not backup_dir.exists():
            shutil.copytree(args.labels_dir, backup_dir)
            logger.info("Backed up original labels to %s", backup_dir)

    convert_split(
        labels_dir=args.labels_dir,
        images_dir=args.images_dir,
        output_dir=output_dir,
        img_w=args.img_size,
        img_h=args.img_size,
        clip=not args.no_clip,
        skip_oob_frac=0.5 if args.skip_oob else None,
        min_area_frac=args.min_area,
        verify_images=args.verify_images,
    )


if __name__ == "__main__":
    main()
