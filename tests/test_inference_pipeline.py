"""Tests for the inference tiling pipeline."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.inference.inference_pipeline import nms_obb, tile_image


# ---------------------------------------------------------------------------
# tile_image
# ---------------------------------------------------------------------------

class TestTileImage:
    def test_single_tile_for_small_image(self):
        """An image smaller than one tile should produce exactly one tile."""
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        tiles = tile_image(img, tile_size=1024, overlap=0)
        assert len(tiles) == 1
        assert tiles[0]["row_start"] == 0
        assert tiles[0]["col_start"] == 0

    def test_tile_count_no_overlap(self):
        """Exact multiple of tile_size with no overlap → predictable count."""
        img = np.zeros((2048, 2048, 3), dtype=np.uint8)
        tiles = tile_image(img, tile_size=1024, overlap=0)
        # 2 rows × 2 cols = 4 tiles
        assert len(tiles) == 4

    def test_tile_count_with_overlap(self):
        """Overlapping tiles produce more tiles than non-overlapping."""
        img = np.zeros((2048, 2048, 3), dtype=np.uint8)
        no_overlap_tiles = tile_image(img, tile_size=1024, overlap=0)
        with_overlap_tiles = tile_image(img, tile_size=1024, overlap=256)
        assert len(with_overlap_tiles) > len(no_overlap_tiles)

    def test_tile_origin_coordinates(self):
        """First tile should start at (0, 0)."""
        img = np.zeros((3000, 3000, 3), dtype=np.uint8)
        tiles = tile_image(img, tile_size=1024, overlap=128)
        assert tiles[0]["row_start"] == 0
        assert tiles[0]["col_start"] == 0

    def test_tile_content_is_crop(self):
        """Tile content must match the corresponding crop of the original."""
        img = np.random.randint(0, 256, (2000, 2000, 3), dtype=np.uint8)
        tiles = tile_image(img, tile_size=512, overlap=0)
        t = tiles[0]
        expected = img[t["row_start"] : t["row_start"] + 512, t["col_start"] : t["col_start"] + 512]
        np.testing.assert_array_equal(t["tile"], expected)

    def test_greyscale_image(self):
        """Greyscale (2-D) images should be tiled without error."""
        img = np.zeros((1024, 1024), dtype=np.uint8)
        tiles = tile_image(img, tile_size=512, overlap=0)
        assert len(tiles) == 4

    def test_empty_image_returns_empty_list(self):
        """An image with zero height or width should return an empty list without crash."""
        img = np.zeros((0, 1024, 3), dtype=np.uint8)
        tiles = tile_image(img, tile_size=1024, overlap=0)
        assert tiles == []


# ---------------------------------------------------------------------------
# nms_obb
# ---------------------------------------------------------------------------

class TestNmsObb:
    def _make_det(self, cx, cy, w, h, angle, conf):
        return {"bbox": [cx, cy, w, h, angle], "confidence": conf, "class": 0}

    def test_empty_input(self):
        assert nms_obb([]) == []

    def test_single_detection_kept(self):
        det = self._make_det(100, 100, 50, 20, 0, 0.9)
        result = nms_obb([det])
        assert len(result) == 1

    def test_non_overlapping_detections_all_kept(self):
        """Well-separated boxes should all survive NMS."""
        dets = [
            self._make_det(100, 100, 50, 20, 0, 0.9),
            self._make_det(500, 500, 50, 20, 0, 0.8),
            self._make_det(900, 900, 50, 20, 0, 0.7),
        ]
        result = nms_obb(dets, iou_threshold=0.45)
        assert len(result) == 3

    def test_identical_boxes_suppressed(self):
        """Two identical boxes: only the one with higher confidence is kept."""
        det_high = self._make_det(100, 100, 50, 20, 0, 0.95)
        det_low = self._make_det(100, 100, 50, 20, 0, 0.5)
        result = nms_obb([det_high, det_low], iou_threshold=0.45)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.95

    def test_output_sorted_by_confidence(self):
        """Highest-confidence detection should be first in the output."""
        dets = [
            self._make_det(100, 100, 50, 20, 0, 0.6),
            self._make_det(200, 200, 50, 20, 0, 0.9),
            self._make_det(300, 300, 50, 20, 0, 0.75),
        ]
        result = nms_obb(dets, iou_threshold=0.45)
        confs = [d["confidence"] for d in result]
        assert confs == sorted(confs, reverse=True)
