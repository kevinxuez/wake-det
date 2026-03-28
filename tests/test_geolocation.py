"""Tests for the azimuth-shift geolocation correction module."""

from __future__ import annotations

import math

import pytest

from src.processing.geolocation import (
    compute_azimuth_shift,
    correct_ship_position,
    estimate_radial_velocity,
    metres_to_degrees,
    pixel_to_latlon,
)
from src.integration.ais_correlation import haversine_distance


# ---------------------------------------------------------------------------
# compute_azimuth_shift
# ---------------------------------------------------------------------------

class TestComputeAzimuthShift:
    def test_zero_radial_velocity(self):
        """A stationary target should produce zero azimuth shift."""
        shift = compute_azimuth_shift(
            slant_range=700_000,
            satellite_velocity=7_500,
            radial_velocity=0.0,
        )
        assert shift == 0.0

    def test_positive_radial_velocity(self):
        """Positive radial velocity should produce a positive shift."""
        shift = compute_azimuth_shift(
            slant_range=700_000,
            satellite_velocity=7_500,
            radial_velocity=5.0,
        )
        expected = (700_000 / 7_500) * 5.0
        assert math.isclose(shift, expected, rel_tol=1e-9)

    def test_negative_radial_velocity(self):
        """Negative radial velocity (toward satellite) should give negative shift."""
        shift = compute_azimuth_shift(
            slant_range=700_000,
            satellite_velocity=7_500,
            radial_velocity=-3.0,
        )
        assert shift < 0

    def test_zero_satellite_velocity_raises(self):
        with pytest.raises(ValueError, match="satellite_velocity"):
            compute_azimuth_shift(700_000, 0, 5.0)

    def test_typical_sentinel1_values(self):
        """Typical Sentinel-1 parameters should produce a plausible shift."""
        shift = compute_azimuth_shift(
            slant_range=700_000,   # ~700 km
            satellite_velocity=7_500,  # ~7.5 km/s
            radial_velocity=5.0,   # 5 m/s ship
        )
        # Expected: (700000 / 7500) * 5 ≈ 466.7 m
        assert 400 < shift < 550


# ---------------------------------------------------------------------------
# estimate_radial_velocity
# ---------------------------------------------------------------------------

class TestEstimateRadialVelocity:
    def test_roundtrip_with_azimuth_shift(self):
        """estimate_radial_velocity should invert compute_azimuth_shift."""
        v_r = 4.2
        slant = 650_000
        v_sat = 7_400
        shift = compute_azimuth_shift(slant, v_sat, v_r)
        recovered = estimate_radial_velocity(shift, slant, v_sat)
        assert math.isclose(recovered, v_r, rel_tol=1e-9)

    def test_zero_slant_range_raises(self):
        with pytest.raises(ValueError, match="slant_range"):
            estimate_radial_velocity(100.0, 0, 7_500)


# ---------------------------------------------------------------------------
# metres_to_degrees
# ---------------------------------------------------------------------------

class TestMetresToDegrees:
    def test_one_degree_latitude(self):
        """~111 km should correspond to ~1 degree of latitude."""
        _, delta_lat = metres_to_degrees(111_320, reference_latitude=0)
        assert math.isclose(delta_lat, 1.0, rel_tol=0.01)

    def test_longitude_compressed_at_high_latitude(self):
        """Longitude degree width is smaller at higher latitudes."""
        dlon_eq, _ = metres_to_degrees(1000, reference_latitude=0)
        dlon_60, _ = metres_to_degrees(1000, reference_latitude=60)
        assert dlon_60 > dlon_eq  # fewer metres per degree at 60°


# ---------------------------------------------------------------------------
# pixel_to_latlon
# ---------------------------------------------------------------------------

class TestPixelToLatlon:
    def _typical_transform(self):
        """A typical Sentinel-1 GRD affine transform (10 m pixels, NW corner)."""
        return (120.0, 8.983e-5, 0.0, 35.0, 0.0, -8.983e-5)

    def test_origin_pixel(self):
        """Pixel (0, 0) should return the transform origin."""
        t = self._typical_transform()
        lon, lat = pixel_to_latlon(0, 0, t)
        assert math.isclose(lon, 120.0) and math.isclose(lat, 35.0)

    def test_positive_pixel_offset(self):
        """Moving one pixel east should increase longitude slightly."""
        t = self._typical_transform()
        lon0, _ = pixel_to_latlon(0, 0, t)
        lon1, _ = pixel_to_latlon(1, 0, t)
        assert lon1 > lon0


# ---------------------------------------------------------------------------
# correct_ship_position
# ---------------------------------------------------------------------------

class TestCorrectShipPosition:
    def test_zero_shift_returns_apex(self):
        """A zero azimuth shift should return the wake-apex position."""
        lon, lat = correct_ship_position(
            ship_lon=120.5,
            ship_lat=35.2,
            wake_apex_lon=120.502,
            wake_apex_lat=35.198,
            azimuth_shift_m=0.0,
            heading_deg=90.0,
        )
        assert math.isclose(lon, 120.502)
        assert math.isclose(lat, 35.198)

    def test_northbound_ship_shifts_latitude(self):
        """A northbound ship (heading=0°) shift should primarily affect latitude."""
        lon, lat = correct_ship_position(
            ship_lon=120.0,
            ship_lat=35.0,
            wake_apex_lon=120.0,
            wake_apex_lat=35.0,
            azimuth_shift_m=1000.0,
            heading_deg=0.0,
        )
        # Latitude should change, longitude should not.
        assert math.isclose(lon, 120.0, abs_tol=1e-10)
        assert not math.isclose(lat, 35.0)

    def test_output_is_close_to_input_for_small_shift(self):
        """Small shifts should produce output close to the input position."""
        lon, lat = correct_ship_position(
            ship_lon=120.0,
            ship_lat=35.0,
            wake_apex_lon=120.001,
            wake_apex_lat=35.001,
            azimuth_shift_m=10.0,
            heading_deg=45.0,
        )
        dist = haversine_distance(120.001, 35.001, lon, lat)
        assert dist < 50  # less than 50 m change from apex


# ---------------------------------------------------------------------------
# haversine_distance
# ---------------------------------------------------------------------------

class TestHaversineDistance:
    def test_same_point_is_zero(self):
        assert haversine_distance(10.0, 20.0, 10.0, 20.0) == 0.0

    def test_known_distance(self):
        """Distance between two points ~1 degree of latitude apart ≈ 111 km."""
        dist = haversine_distance(0.0, 0.0, 0.0, 1.0)
        assert 110_000 < dist < 112_000

    def test_symmetry(self):
        d1 = haversine_distance(0, 0, 10, 20)
        d2 = haversine_distance(10, 20, 0, 0)
        assert math.isclose(d1, d2)
