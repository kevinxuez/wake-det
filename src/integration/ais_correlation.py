"""
Step 6 – AIS Telemetry Correlation
Correlates SAR ship-wake detections with live Automatic Identification System
(AIS) telemetry to identify dark, non-cooperative vessels.

Algorithm:
1. Load AIS track data (list of timestamped position pings).
2. Interpolate each vessel's position to the exact SAR image acquisition time.
3. For each SAR detection, find the closest AIS-interpolated position.
4. If no AIS position falls within a threshold (default 150 m), flag the
   detection as a *dark vessel*.

Usage::

    from src.integration.ais_correlation import (
        interpolate_ais_position,
        correlate_detections,
    )

    dark = correlate_detections(
        detections=detections,
        ais_tracks=ais_tracks,
        acquisition_time="2024-06-15T10:32:00Z",
        threshold_m=150.0,
    )
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_EARTH_RADIUS_M = 6_371_000.0


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_distance(
    lon1: float, lat1: float, lon2: float, lat2: float
) -> float:
    """Calculate the great-circle distance between two geographic points.

    Args:
        lon1: Longitude of the first point (degrees).
        lat1: Latitude of the first point (degrees).
        lon2: Longitude of the second point (degrees).
        lat2: Latitude of the second point (degrees).

    Returns:
        Distance in metres.
    """
    lon1_r, lat1_r, lon2_r, lat2_r = (
        math.radians(lon1),
        math.radians(lat1),
        math.radians(lon2),
        math.radians(lat2),
    )
    dlon = lon2_r - lon1_r
    dlat = lat2_r - lat1_r
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# AIS position interpolation
# ---------------------------------------------------------------------------

def _parse_iso(ts: str) -> float:
    """Parse an ISO-8601 timestamp string to a POSIX timestamp (seconds)."""
    ts = ts.rstrip("Z")
    if "+" in ts:
        ts = ts.split("+")[0]
    dt = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
    return dt.timestamp()


def interpolate_ais_position(
    pings: list[dict[str, Any]],
    target_time: str,
) -> dict[str, float] | None:
    """Linearly interpolate an AIS vessel position to a target timestamp.

    Each ping in *pings* must have:
    * ``"timestamp"`` – ISO-8601 string;
    * ``"lon"`` – longitude in decimal degrees;
    * ``"lat"`` – latitude in decimal degrees.

    Pings are sorted by timestamp internally.

    Args:
        pings: Chronological list of AIS position pings for one vessel.
        target_time: ISO-8601 timestamp of the SAR image acquisition.

    Returns:
        ``{"lon": float, "lat": float}`` interpolated position, or ``None``
        if interpolation is not possible (fewer than 2 pings, or target time
        is outside the ping window by more than 10 minutes).
    """
    if not pings:
        return None

    sorted_pings = sorted(pings, key=lambda p: _parse_iso(p["timestamp"]))
    t_target = _parse_iso(target_time)

    # Find bracketing pings.
    before = [p for p in sorted_pings if _parse_iso(p["timestamp"]) <= t_target]
    after = [p for p in sorted_pings if _parse_iso(p["timestamp"]) >= t_target]

    if not before or not after:
        # Target time is outside the track window.
        nearest = min(sorted_pings, key=lambda p: abs(_parse_iso(p["timestamp"]) - t_target))
        gap = abs(_parse_iso(nearest["timestamp"]) - t_target)
        if gap > 600:  # 10 minutes
            logger.debug("AIS ping gap too large (%.0f s); skipping vessel.", gap)
            return None
        return {"lon": float(nearest["lon"]), "lat": float(nearest["lat"])}

    p0 = before[-1]
    p1 = after[0]

    t0 = _parse_iso(p0["timestamp"])
    t1 = _parse_iso(p1["timestamp"])

    if t0 == t1:
        return {"lon": float(p0["lon"]), "lat": float(p0["lat"])}

    frac = (t_target - t0) / (t1 - t0)
    lon_interp = float(p0["lon"]) + frac * (float(p1["lon"]) - float(p0["lon"]))
    lat_interp = float(p0["lat"]) + frac * (float(p1["lat"]) - float(p0["lat"]))
    return {"lon": lon_interp, "lat": lat_interp}


# ---------------------------------------------------------------------------
# Detection correlation
# ---------------------------------------------------------------------------

def correlate_detections(
    detections: list[dict[str, Any]],
    ais_tracks: dict[str, list[dict[str, Any]]],
    acquisition_time: str,
    threshold_m: float = 150.0,
) -> list[dict[str, Any]]:
    """Correlate SAR wake detections with AIS tracks to find dark vessels.

    For each detection the function:
    1. Interpolates every AIS track to the SAR acquisition time.
    2. Finds the closest interpolated AIS position.
    3. If the closest AIS position is beyond *threshold_m*, the detection is
       flagged as a **dark vessel** (``"dark_vessel": True``).

    Args:
        detections: List of detection dicts, each with ``"lon"`` and ``"lat"``
            (geographic coordinates after geolocation correction).
        ais_tracks: Dict mapping vessel MMSI / ID strings to lists of AIS
            position pings.  Each ping must have ``"timestamp"``, ``"lon"``,
            and ``"lat"`` fields.
        acquisition_time: ISO-8601 timestamp of the SAR acquisition.
        threshold_m: Distance threshold in metres; detections with no AIS ping
            within this distance are considered dark (default 150 m).

    Returns:
        Annotated detection list with additional fields per detection:

        * ``"dark_vessel"`` – ``True`` if no matching AIS ping was found;
        * ``"closest_mmsi"`` – MMSI of the closest AIS vessel (or ``None``);
        * ``"closest_distance_m"`` – distance to the closest AIS position (or
          ``None``).
    """
    # Pre-compute interpolated AIS positions once for efficiency.
    interpolated: dict[str, dict[str, float] | None] = {}
    for mmsi, pings in ais_tracks.items():
        interpolated[mmsi] = interpolate_ais_position(pings, acquisition_time)

    results: list[dict[str, Any]] = []
    dark_count = 0

    for det in detections:
        det_lon = det.get("lon")
        det_lat = det.get("lat")

        if det_lon is None or det_lat is None:
            logger.warning(
                "Detection missing lon/lat; skipping AIS correlation: %s", det
            )
            det_out = dict(det)
            det_out["dark_vessel"] = None
            det_out["closest_mmsi"] = None
            det_out["closest_distance_m"] = None
            results.append(det_out)
            continue

        min_dist = float("inf")
        closest_mmsi: str | None = None

        for mmsi, pos in interpolated.items():
            if pos is None:
                continue
            dist = haversine_distance(det_lon, det_lat, pos["lon"], pos["lat"])
            if dist < min_dist:
                min_dist = dist
                closest_mmsi = mmsi

        is_dark = min_dist > threshold_m
        if is_dark:
            dark_count += 1

        det_out = dict(det)
        det_out["dark_vessel"] = is_dark
        det_out["closest_mmsi"] = closest_mmsi
        det_out["closest_distance_m"] = min_dist if min_dist != float("inf") else None
        results.append(det_out)

    logger.info(
        "AIS correlation complete: %d detection(s), %d dark vessel(s) identified.",
        len(results),
        dark_count,
    )
    return results
