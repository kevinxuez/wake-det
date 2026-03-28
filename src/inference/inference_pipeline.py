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


def _aabb_iou(box_a: list[float], box_b: list[float]) -> float:
    """Fast axis-aligned bounding box IoU for coarse-pass NMS.

    Computes the AABB enclosing the rotated OBB, then calculates standard IoU.

    Args:
        box_a: OBB as ``[cx, cy, w, h, angle_deg]``.
        box_b: OBB as ``[cx, cy, w, h, angle_deg]``.

    Returns:
        Axis-aligned IoU in ``[0, 1]``.
    """
    import math

    def _obb_to_aabb(b: list[float]) -> tuple[float, float, float, float]:
        cx, cy, bw, bh, angle = b
        rad = math.radians(angle)
        cos_a, sin_a = abs(math.cos(rad)), abs(math.sin(rad))
        # Half-extents of the axis-aligned enclosure
        half_w = (bw * cos_a + bh * sin_a) / 2
        half_h = (bw * sin_a + bh * cos_a) / 2
        return cx - half_w, cy - half_h, cx + half_w, cy + half_h

    x1a, y1a, x2a, y2a = _obb_to_aabb(box_a)
    x1b, y1b, x2b, y2b = _obb_to_aabb(box_b)

    inter_x1 = max(x1a, x1b)
    inter_y1 = max(y1a, y1b)
    inter_x2 = min(x2a, x2b)
    inter_y2 = min(y2a, y2b)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    area_a = (x2a - x1a) * (y2a - y1a)
    area_b = (x2b - x1b) * (y2b - y1b)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _greedy_nms(
    detections: list[dict[str, Any]],
    iou_fn,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    """Generic greedy NMS using a pluggable IoU function."""
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept: list[dict[str, Any]] = []
    while sorted_dets:
        best = sorted_dets.pop(0)
        kept.append(best)
        sorted_dets = [
            d for d in sorted_dets
            if iou_fn(best["bbox"], d["bbox"]) < iou_threshold
        ]
    return kept


def nms_obb(
    detections: list[dict[str, Any]],
    iou_threshold: float = 0.45,
) -> list[dict[str, Any]]:
    """Single-pass polygon NMS (legacy interface).

    Args:
        detections: Detection dicts with ``"bbox"`` and ``"confidence"``.
        iou_threshold: IoU threshold (default 0.45).

    Returns:
        Filtered detections.
    """
    return _greedy_nms(detections, _obb_iou, iou_threshold)


def nms_obb_two_stage(
    detections: list[dict[str, Any]],
    horizontal_iou: float = 0.8,
    polygon_iou: float = 0.1,
    confidence_threshold: float = 0.05,
) -> list[dict[str, Any]]:
    """Two-stage NMS from the OpenSARWake paper (Xu & Wang, 2024).

    Stage 1: Axis-aligned bounding box NMS with a loose IoU threshold (0.8)
    to cheaply reduce redundant overlapping detections.

    Stage 2: Polygon-based OBB NMS with a tight IoU threshold (0.1) for
    precise deduplication of oriented wakes.

    Args:
        detections: Detection dicts with ``"bbox"`` ``[cx, cy, w, h, angle]``
            and ``"confidence"``.
        horizontal_iou: Stage-1 axis-aligned IoU threshold (default 0.8).
        polygon_iou: Stage-2 polygon IoU threshold (default 0.1).
        confidence_threshold: Minimum confidence to enter NMS (default 0.05,
            matching the paper; lower than typical 0.25 to preserve faint
            turbulent wakes).

    Returns:
        Filtered detections after both NMS stages.
    """
    # Pre-filter by confidence
    dets = [d for d in detections if d["confidence"] >= confidence_threshold]
    if not dets:
        return []

    # Stage 1: fast horizontal NMS (coarse)
    after_stage1 = _greedy_nms(dets, _aabb_iou, horizontal_iou)
    logger.debug("Two-stage NMS: %d → %d after horizontal pass", len(dets), len(after_stage1))

    # Stage 2: precise polygon NMS (tight)
    after_stage2 = _greedy_nms(after_stage1, _obb_iou, polygon_iou)
    logger.debug("Two-stage NMS: %d → %d after polygon pass", len(after_stage1), len(after_stage2))

    return after_stage2


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
    confidence_threshold: float = 0.05,
    iou_threshold: float = 0.45,
    two_stage_nms: bool = True,
) -> list[dict[str, Any]]:
    """Run full-scene SAR inference using a tiled YOLOv8-OBB model.

    Large SAR scenes are split into overlapping tiles, detections are gathered
    per-tile with their absolute pixel coordinates, and finally deduplicated
    with NMS.

    By default uses the two-stage NMS strategy from the OpenSARWake paper:
    coarse axis-aligned pass (IoU 0.8) followed by tight polygon pass
    (IoU 0.1).  The lower confidence threshold (0.05 vs 0.25) preserves
    faint turbulent wakes that would otherwise be discarded.

    Args:
        weights_path: Path to the trained YOLOv8-OBB weights file (``best.pt``).
        image_path: Path to the SAR scene image (GeoTIFF, PNG, JPEG, …).
        tile_size: Tile side length in pixels (default 1024).
        overlap: Tile overlap in pixels (default 128).
        confidence_threshold: Detection confidence threshold (default 0.05).
        iou_threshold: NMS IoU threshold for single-pass mode (default 0.45).
        two_stage_nms: If True (default), use two-stage NMS from the
            OpenSARWake paper instead of single-pass polygon NMS.

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
    if two_stage_nms:
        final_detections = nms_obb_two_stage(
            all_detections,
            confidence_threshold=confidence_threshold,
        )
        logger.info("Post-NMS detections (two-stage): %d", len(final_detections))
    else:
        filtered = [d for d in all_detections if d["confidence"] >= confidence_threshold]
        final_detections = nms_obb(filtered, iou_threshold=iou_threshold)
        logger.info("Post-NMS detections (single-pass): %d", len(final_detections))
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
        default=0.05,
        help="Detection confidence threshold (default: 0.05, per OpenSARWake paper).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold for single-pass mode (default: 0.45).",
    )
    parser.add_argument(
        "--single-pass-nms",
        action="store_true",
        help="Use legacy single-pass NMS instead of two-stage (default: two-stage).",
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
        two_stage_nms=not args.single_pass_nms,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(detections, fh, indent=2)

    logger.info("Detections written to %s", args.output)


if __name__ == "__main__":
    main()
