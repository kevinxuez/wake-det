"""Tests for the AIS telemetry correlation module."""

from __future__ import annotations

import pytest

from src.integration.ais_correlation import (
    correlate_detections,
    haversine_distance,
    interpolate_ais_position,
)


# ---------------------------------------------------------------------------
# haversine_distance
# ---------------------------------------------------------------------------

class TestHaversineDistance:
    def test_zero_distance(self):
        assert haversine_distance(10.0, 20.0, 10.0, 20.0) == 0.0

    def test_one_degree_latitude(self):
        """~1 degree of latitude ≈ 111 km."""
        dist = haversine_distance(0.0, 0.0, 0.0, 1.0)
        assert 110_000 < dist < 112_000

    def test_symmetry(self):
        d1 = haversine_distance(0, 0, 5, 5)
        d2 = haversine_distance(5, 5, 0, 0)
        import math
        assert math.isclose(d1, d2)


# ---------------------------------------------------------------------------
# interpolate_ais_position
# ---------------------------------------------------------------------------

class TestInterpolateAisPosition:
    def _make_ping(self, ts: str, lon: float, lat: float) -> dict:
        return {"timestamp": ts, "lon": lon, "lat": lat}

    def test_exact_ping_match(self):
        """When the target time exactly matches a ping, return that position."""
        pings = [self._make_ping("2024-06-15T10:30:00", 120.0, 35.0)]
        pos = interpolate_ais_position(pings, "2024-06-15T10:30:00")
        assert pos is not None
        assert pos["lon"] == 120.0
        assert pos["lat"] == 35.0

    def test_midpoint_interpolation(self):
        """Target time halfway between two pings should return the midpoint."""
        pings = [
            self._make_ping("2024-06-15T10:00:00", 120.0, 35.0),
            self._make_ping("2024-06-15T11:00:00", 121.0, 36.0),
        ]
        pos = interpolate_ais_position(pings, "2024-06-15T10:30:00")
        assert pos is not None
        assert abs(pos["lon"] - 120.5) < 1e-6
        assert abs(pos["lat"] - 35.5) < 1e-6

    def test_empty_pings_returns_none(self):
        assert interpolate_ais_position([], "2024-06-15T10:00:00") is None

    def test_extrapolation_large_gap_returns_none(self):
        """Target time > 10 min from any ping should return None."""
        pings = [self._make_ping("2024-06-15T10:00:00", 120.0, 35.0)]
        pos = interpolate_ais_position(pings, "2024-06-15T12:00:00")
        assert pos is None

    def test_out_of_order_pings_sorted(self):
        """Pings in non-chronological order should still interpolate correctly."""
        pings = [
            self._make_ping("2024-06-15T11:00:00", 121.0, 36.0),
            self._make_ping("2024-06-15T10:00:00", 120.0, 35.0),
        ]
        pos = interpolate_ais_position(pings, "2024-06-15T10:30:00")
        assert pos is not None
        assert abs(pos["lon"] - 120.5) < 1e-6


# ---------------------------------------------------------------------------
# correlate_detections
# ---------------------------------------------------------------------------

class TestCorrelateDetections:
    def _make_detection(self, lon: float, lat: float) -> dict:
        return {
            "bbox": [0, 0, 50, 20, 0],
            "confidence": 0.9,
            "class": 0,
            "lon": lon,
            "lat": lat,
        }

    def _make_track(self, lon: float, lat: float) -> list[dict]:
        return [{"timestamp": "2024-06-15T10:32:00", "lon": lon, "lat": lat}]

    def test_dark_vessel_no_ais(self):
        """Detection with no AIS tracks should be flagged dark."""
        dets = [self._make_detection(120.0, 35.0)]
        results = correlate_detections(dets, {}, "2024-06-15T10:32:00", threshold_m=150)
        assert results[0]["dark_vessel"] is True

    def test_cooperative_vessel_nearby(self):
        """Detection with a matching AIS ping within threshold is cooperative."""
        dets = [self._make_detection(120.0, 35.0)]
        tracks = {"MMSI_001": self._make_track(120.0, 35.0)}
        results = correlate_detections(dets, tracks, "2024-06-15T10:32:00", threshold_m=150)
        assert results[0]["dark_vessel"] is False
        assert results[0]["closest_mmsi"] == "MMSI_001"
        assert results[0]["closest_distance_m"] == 0.0

    def test_dark_vessel_far_ais(self):
        """Detection far from every AIS ping (>threshold) should be dark."""
        dets = [self._make_detection(120.0, 35.0)]
        # AIS vessel is ~200 km away
        tracks = {"MMSI_002": self._make_track(122.0, 37.0)}
        results = correlate_detections(dets, tracks, "2024-06-15T10:32:00", threshold_m=150)
        assert results[0]["dark_vessel"] is True

    def test_multiple_detections_mixed(self):
        """Mixed scenario: one cooperative, one dark."""
        dets = [
            self._make_detection(120.0, 35.0),  # should match MMSI_A
            self._make_detection(130.0, 40.0),  # no nearby AIS
        ]
        tracks = {"MMSI_A": self._make_track(120.0, 35.0)}
        results = correlate_detections(dets, tracks, "2024-06-15T10:32:00", threshold_m=150)
        assert results[0]["dark_vessel"] is False
        assert results[1]["dark_vessel"] is True

    def test_missing_lon_lat_handled(self):
        """Detections without lon/lat should get dark_vessel=None without crash."""
        det = {"bbox": [0, 0, 50, 20, 0], "confidence": 0.8, "class": 0}
        results = correlate_detections([det], {}, "2024-06-15T10:32:00")
        assert results[0]["dark_vessel"] is None

    def test_returns_same_count(self):
        """Output list length must match input list length."""
        dets = [self._make_detection(120 + i, 35 + i) for i in range(5)]
        results = correlate_detections(dets, {}, "2024-06-15T10:32:00")
        assert len(results) == len(dets)
