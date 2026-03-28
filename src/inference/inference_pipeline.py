"""
Step 3 – Inference Pipeline with Image Tiling
Runs YOLOv8-OBB inference on large SAR satellite scenes by dividing them into
overlapping tiles and merging the predictions with Non-Maximum Suppression (NMS).

Usage::

    python src/inference/inference_pipeline.py \
        --weights  runs/train/weights/best.pt \
        --image    data/scene.tif \
        --output   results/detections.json \
        --tile-size 1024 \
        --overlap   128
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image tiling
# ---------------------------------------------------------------------------

def tile_image(
    image: np.ndarray,
    tile_size: int = 1024,
    overlap: int = 128,
) -> list[dict[str, Any]]:
    """Slice a large image into overlapping tiles.

    Args:
        image: Input image as a NumPy array with shape ``(H, W)`` or
            ``(H, W, C)``.
        tile_size: Side length of each square tile in pixels (default 1024).
        overlap: Overlap between adjacent tiles in pixels (default 128).

    Returns:
        A list of dicts, each with keys:

        * ``tile`` – the cropped image array;
        * ``row_start``, ``col_start`` – top-left corner coordinates in the
          original image.
    """
    h, w = image.shape[:2]
    stride = tile_size - overlap
    tiles: list[dict[str, Any]] = []

    for row_start in range(0, h, stride):
        for col_start in range(0, w, stride):
            row_end = min(row_start + tile_size, h)
            col_end = min(col_start + tile_size, w)
            tile = image[row_start:row_end, col_start:col_end]
            tiles.append(
                {
                    "tile": tile,
                    "row_start": row_start,
                    "col_start": col_start,
                }
            )

    logger.debug("Tiled image (%dx%d) into %d tiles", h, w, len(tiles))
    return tiles


# ---------------------------------------------------------------------------
# Oriented bounding box NMS
# ---------------------------------------------------------------------------

def _obb_iou(box_a: list[float], box_b: list[float]) -> float:
    """Approximate IoU for two oriented bounding boxes (OBBs) via Shapely.

    Each box is described as ``[cx, cy, w, h, angle_deg]``.

    Args:
        box_a: First OBB as ``[cx, cy, w, h, angle_deg]``.
        box_b: Second OBB as ``[cx, cy, w, h, angle_deg]``.

    Returns:
        Intersection-over-Union value in ``[0, 1]``.
    """
    try:
        from shapely.affinity import rotate
        from shapely.geometry import box as shapely_box
    except ImportError:
        # Graceful degradation: treat as non-overlapping when Shapely is absent.
        return 0.0

    def _to_poly(b: list[float]):
        cx, cy, bw, bh, angle = b
        rect = shapely_box(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
        return rotate(rect, angle, origin=(cx, cy))

    poly_a = _to_poly(box_a)
    poly_b = _to_poly(box_b)
    if poly_a.area + poly_b.area == 0:
        return 0.0
    intersection = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    return intersection / union if union > 0 else 0.0


def nms_obb(
    detections: list[dict[str, Any]],
    iou_threshold: float = 0.45,
) -> list[dict[str, Any]]:
    """Apply NMS to a list of oriented bounding box detections.

    Args:
        detections: List of detection dicts, each with at least:

            * ``"bbox"`` – ``[cx, cy, w, h, angle_deg]``
            * ``"confidence"`` – float score in ``[0, 1]``

        iou_threshold: IoU threshold above which the lower-confidence box is
            suppressed (default 0.45).

    Returns:
        Filtered list of detections after NMS.
    """
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept: list[dict[str, Any]] = []

    while sorted_dets:
        best = sorted_dets.pop(0)
        kept.append(best)
        sorted_dets = [
            d
            for d in sorted_dets
            if _obb_iou(best["bbox"], d["bbox"]) < iou_threshold
        ]

    return kept


# ---------------------------------------------------------------------------
# Inference on a single tile
# ---------------------------------------------------------------------------

def run_inference_on_tile(
    model,
    tile: np.ndarray,
    confidence_threshold: float = 0.25,
) -> list[dict[str, Any]]:
    """Run YOLOv8-OBB inference on a single tile.

    Args:
        model: Loaded ``ultralytics.YOLO`` model instance.
        tile: Image tile as a NumPy array.
        confidence_threshold: Minimum confidence to retain a detection
            (default 0.25).

    Returns:
        List of detection dicts with keys ``"bbox"``, ``"confidence"``, and
        ``"class"``.
    """
    results = model(tile, verbose=False)
    detections: list[dict[str, Any]] = []

    for result in results:
        if result.obb is None:
            continue
        for obb_box in result.obb:
            conf = float(obb_box.conf[0])
            if conf < confidence_threshold:
                continue
            cx, cy, bw, bh, angle = obb_box.xywhr[0].tolist()
            cls = int(obb_box.cls[0])
            detections.append(
                {
                    "bbox": [cx, cy, bw, bh, angle],
                    "confidence": conf,
                    "class": cls,
                }
            )

    return detections


# ---------------------------------------------------------------------------
# Full-scene inference
# ---------------------------------------------------------------------------

def infer_scene(
    weights_path: str | Path,
    image_path: str | Path,
    tile_size: int = 1024,
    overlap: int = 128,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> list[dict[str, Any]]:
    """Run full-scene SAR inference using a tiled YOLOv8-OBB model.

    Large SAR scenes are split into overlapping tiles, detections are gathered
    per-tile with their absolute pixel coordinates, and finally deduplicated
    with NMS.

    Args:
        weights_path: Path to the trained YOLOv8-OBB weights file (``best.pt``).
        image_path: Path to the SAR scene image (GeoTIFF, PNG, JPEG, …).
        tile_size: Tile side length in pixels (default 1024).
        overlap: Tile overlap in pixels (default 128).
        confidence_threshold: Detection confidence threshold (default 0.25).
        iou_threshold: NMS IoU threshold (default 0.45).

    Returns:
        List of detection dicts (after NMS) with global pixel coordinates:

        * ``"bbox"`` – ``[cx, cy, w, h, angle_deg]`` in the full-scene
          coordinate system;
        * ``"confidence"`` – detection confidence;
        * ``"class"`` – class index.
    """
    try:
        import cv2  # type: ignore
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "ultralytics and opencv-python are required: pip install ultralytics opencv-python"
        ) from exc

    logger.info("Loading weights from %s", weights_path)
    model = YOLO(str(weights_path))

    logger.info("Reading scene image from %s", image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Convert greyscale SAR images to 3-channel for YOLO.
    if image.ndim == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    tiles = tile_image(image, tile_size=tile_size, overlap=overlap)
    all_detections: list[dict[str, Any]] = []

    for tile_info in tiles:
        tile_dets = run_inference_on_tile(
            model, tile_info["tile"], confidence_threshold=confidence_threshold
        )
        for det in tile_dets:
            # Translate bbox centre from tile coords to global image coords.
            cx, cy, bw, bh, angle = det["bbox"]
            det["bbox"] = [
                cx + tile_info["col_start"],
                cy + tile_info["row_start"],
                bw,
                bh,
                angle,
            ]
        all_detections.extend(tile_dets)

    logger.info(
        "Pre-NMS detections: %d across %d tiles", len(all_detections), len(tiles)
    )
    final_detections = nms_obb(all_detections, iou_threshold=iou_threshold)
    logger.info("Post-NMS detections: %d", len(final_detections))
    return final_detections


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run tiled YOLOv8-OBB inference on a SAR scene image."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to trained YOLOv8-OBB weights (best.pt).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the SAR scene image.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/detections.json"),
        help="Output JSON file for detections (default: results/detections.json).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Tile size in pixels (default: 1024).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=128,
        help="Tile overlap in pixels (default: 128).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45).",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    detections = infer_scene(
        weights_path=args.weights,
        image_path=args.image,
        tile_size=args.tile_size,
        overlap=args.overlap,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(detections, fh, indent=2)

    logger.info("Detections written to %s", args.output)


if __name__ == "__main__":
    main()
