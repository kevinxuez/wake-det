"""
Step 5 – Azimuth-Shift Geolocation Correction
Corrects the azimuth displacement of a moving ship's radar return relative to
the wake apex, as caused by the Doppler effect in SAR imaging geometry.

The azimuth shift formula is:

    ζ = (R₀ / V) · v_r(x_m, y_n)

where:
    ζ   – azimuth displacement in metres
    R₀  – slant range from satellite to target (metres)
    V   – effective satellite velocity (m/s)
    v_r – radial (line-of-sight) velocity of the ship (m/s)

Usage::

    from src.processing.geolocation import (
        compute_azimuth_shift,
        correct_ship_position,
        pixel_to_latlon,
    )

    shift_m = compute_azimuth_shift(
        slant_range=700_000,
        satellite_velocity=7_500,
        radial_velocity=3.5,
    )
    lon, lat = correct_ship_position(
        ship_lon=120.5,
        ship_lat=35.2,
        wake_apex_lon=120.502,
        wake_apex_lat=35.198,
        azimuth_shift_m=shift_m,
        heading_deg=270.0,
    )
"""

from __future__ import annotations

import logging
import math
from typing import Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Earth radius used for approximate metre-to-degree conversions.
_EARTH_RADIUS_M = 6_371_000.0


# ---------------------------------------------------------------------------
# Core physics
# ---------------------------------------------------------------------------

def compute_azimuth_shift(
    slant_range: float,
    satellite_velocity: float,
    radial_velocity: float,
) -> float:
    """Compute the along-track (azimuth) displacement due to target motion.

    Implements the standard SAR azimuth-shift formula:

    .. math::

        \\zeta = \\frac{R_0}{V} \\cdot v_r

    Args:
        slant_range: Slant range from the satellite to the target in metres
            (R₀).
        satellite_velocity: Effective SAR platform velocity in m/s (V).
            Typical Sentinel-1 value ≈ 7 500 m/s.
        radial_velocity: Radial (line-of-sight) velocity of the target in m/s
            (v_r).  Positive values correspond to motion away from the
            satellite.

    Returns:
        Azimuth displacement in metres (ζ).  Positive values indicate
        displacement in the direction of satellite travel.

    Raises:
        ValueError: If *satellite_velocity* is zero (division by zero).
    """
    if satellite_velocity == 0:
        raise ValueError("satellite_velocity must be non-zero.")
    shift = (slant_range / satellite_velocity) * radial_velocity
    logger.debug(
        "Azimuth shift: R0=%.0f m, V=%.1f m/s, v_r=%.2f m/s → ζ=%.2f m",
        slant_range,
        satellite_velocity,
        radial_velocity,
        shift,
    )
    return shift


def estimate_radial_velocity(
    azimuth_displacement_m: float,
    slant_range: float,
    satellite_velocity: float,
) -> float:
    """Back-calculate the ship's radial velocity from a measured azimuth shift.

    Rearranges the azimuth-shift formula to solve for v_r:

    .. math::

        v_r = \\zeta \\cdot \\frac{V}{R_0}

    Args:
        azimuth_displacement_m: Measured azimuth displacement in metres (ζ).
        slant_range: Slant range in metres (R₀).
        satellite_velocity: Effective satellite velocity in m/s (V).

    Returns:
        Estimated radial velocity of the ship in m/s.

    Raises:
        ValueError: If *slant_range* is zero.
    """
    if slant_range == 0:
        raise ValueError("slant_range must be non-zero.")
    return azimuth_displacement_m * (satellite_velocity / slant_range)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def metres_to_degrees(
    metres: float, reference_latitude: float
) -> Tuple[float, float]:
    """Convert a distance in metres to approximate (Δlongitude, Δlatitude).

    Args:
        metres: Distance to convert (metres).
        reference_latitude: Latitude at which the conversion is performed
            (degrees).  Used to correct for longitude compression.

    Returns:
        ``(delta_lon_deg, delta_lat_deg)`` – signed degree offsets.
    """
    delta_lat = metres / _EARTH_RADIUS_M * (180.0 / math.pi)
    delta_lon = metres / (
        _EARTH_RADIUS_M * math.cos(math.radians(reference_latitude))
    ) * (180.0 / math.pi)
    return delta_lon, delta_lat


def correct_ship_position(
    ship_lon: float,
    ship_lat: float,
    wake_apex_lon: float,
    wake_apex_lat: float,
    azimuth_shift_m: float,
    heading_deg: float,
) -> Tuple[float, float]:
    """Compute the true geographic position of a moving ship.

    Due to the SAR azimuth shift the detected hull position is displaced from
    the real position.  This function anchors the ship to the wake apex and
    applies the azimuth-shift correction to recover true coordinates.

    Strategy:
    1. Compute the unit vector along the ship's heading.
    2. Project *azimuth_shift_m* along that vector to obtain the correction.
    3. Subtract the shift from the (already approximately correct) wake-apex
       position to yield the true ship centre.

    Args:
        ship_lon: Detected (uncorrected) ship longitude in degrees.
        ship_lat: Detected (uncorrected) ship latitude in degrees.
        wake_apex_lon: Longitude of the wake apex (degrees).
        wake_apex_lat: Latitude of the wake apex (degrees).
        azimuth_shift_m: Azimuth displacement to apply (metres), as returned
            by :func:`compute_azimuth_shift`.
        heading_deg: Ship heading in degrees (0 = North, clockwise).

    Returns:
        ``(corrected_lon, corrected_lat)`` – true geographic position.
    """
    # Heading unit vector in (east, north) components.
    heading_rad = math.radians(heading_deg)
    east_component = math.sin(heading_rad)
    north_component = math.cos(heading_rad)

    # Convert the azimuth shift into degree offsets.
    delta_lon_per_m, delta_lat_per_m = metres_to_degrees(1.0, wake_apex_lat)

    corrected_lon = wake_apex_lon - azimuth_shift_m * east_component * delta_lon_per_m
    corrected_lat = wake_apex_lat - azimuth_shift_m * north_component * delta_lat_per_m

    logger.debug(
        "Position corrected: (%.6f, %.6f) → (%.6f, %.6f)",
        ship_lon,
        ship_lat,
        corrected_lon,
        corrected_lat,
    )
    return corrected_lon, corrected_lat


def pixel_to_latlon(
    pixel_x: float,
    pixel_y: float,
    transform: tuple[float, ...],
) -> Tuple[float, float]:
    """Convert pixel coordinates to geographic coordinates using an affine
    transform (as stored in GeoTIFF metadata).

    Args:
        pixel_x: Column (x) coordinate in pixels.
        pixel_y: Row (y) coordinate in pixels.
        transform: Six-element affine transform coefficients
            ``(x_origin, pixel_width, rotation_x, y_origin, rotation_y, pixel_height)``
            in the convention used by GDAL / rasterio.

    Returns:
        ``(longitude, latitude)`` in decimal degrees.
    """
    x_origin, pixel_width, rotation_x, y_origin, rotation_y, pixel_height = transform
    lon = x_origin + pixel_x * pixel_width + pixel_y * rotation_x
    lat = y_origin + pixel_x * rotation_y + pixel_y * pixel_height
    return lon, lat


def apply_geolocation_correction(
    detections: list[dict],
    transform: tuple[float, ...],
    slant_range: float,
    satellite_velocity: float,
) -> list[dict]:
    """Apply azimuth-shift geolocation correction to a list of detections.

    For each detection the function:
    1. Converts pixel coordinates to lon/lat using the image *transform*.
    2. Estimates the radial velocity from the azimuth offset between the
       detected hull centroid and the wake apex.
    3. Applies :func:`correct_ship_position` to get true coordinates.

    The corrected ``lon`` and ``lat`` fields are written back into each
    detection dict.  The raw pixel coordinates are preserved unchanged.

    Args:
        detections: List of detection dicts, each containing at minimum:

            * ``"bbox"`` – ``[cx, cy, w, h, angle_deg]`` in pixel coords;
            * ``"wake_apex_pixel"`` – ``[x, y]`` pixel location of wake apex
              (optional; if absent, the centroid is used as the apex).

        transform: Affine transform for pixel-to-geographic conversion.
        slant_range: SAR slant range in metres.
        satellite_velocity: Effective satellite velocity in m/s.

    Returns:
        The same list with ``"lon"`` and ``"lat"`` fields added to each dict.
    """
    corrected = []
    for det in detections:
        cx, cy = det["bbox"][0], det["bbox"][1]
        angle_deg = det["bbox"][4] if len(det["bbox"]) >= 5 else 0.0

        ship_lon, ship_lat = pixel_to_latlon(cx, cy, transform)

        apex_pixel = det.get("wake_apex_pixel")
        if apex_pixel is not None:
            apex_lon, apex_lat = pixel_to_latlon(
                apex_pixel[0], apex_pixel[1], transform
            )
        else:
            # Fall back to using the hull centroid as the apex.
            apex_lon, apex_lat = ship_lon, ship_lat

        # Estimate the azimuth displacement from the hull-to-apex offset.
        pixel_height = abs(transform[5])
        azimuth_offset_px = cy - (apex_pixel[1] if apex_pixel else cy)
        azimuth_offset_m = azimuth_offset_px * pixel_height

        radial_vel = estimate_radial_velocity(
            azimuth_offset_m, slant_range, satellite_velocity
        )
        shift = compute_azimuth_shift(slant_range, satellite_velocity, radial_vel)

        corrected_lon, corrected_lat = correct_ship_position(
            ship_lon=ship_lon,
            ship_lat=ship_lat,
            wake_apex_lon=apex_lon,
            wake_apex_lat=apex_lat,
            azimuth_shift_m=shift,
            heading_deg=angle_deg,
        )

        det_out = dict(det)
        det_out["lon"] = corrected_lon
        det_out["lat"] = corrected_lat
        det_out["radial_velocity_ms"] = radial_vel
        corrected.append(det_out)

    return corrected
