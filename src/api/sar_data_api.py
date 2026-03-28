"""
Step 4 – SAR Data API Clients
Provides authenticated clients for Copernicus Data Space Ecosystem (Sentinel-1
GRD), ICEYE, and Capella Space SAR imagery APIs.

Usage::

    from src.api.sar_data_api import CopernicusClient, ICEYEClient, CapellaClient

    # Copernicus (free, no credentials required for public products)
    client = CopernicusClient()
    products = client.search(
        aoi_wkt="POLYGON((...))",
        start_date="2024-01-01",
        end_date="2024-01-07",
        product_type="GRD",
    )

    # ICEYE (commercial, requires API token)
    iceye = ICEYEClient(api_key="your_iceye_api_key")
    order_id = iceye.order_image(scene_id="ICEYE-…")

    # Capella Space (commercial, requires API credentials)
    capella = CapellaClient(username="user", password="pass")
    scenes = capella.search(aoi_wkt="POLYGON((...))", start_date="2024-01-01")
"""

from __future__ import annotations

import logging
import os
import urllib.parse
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30  # seconds


# ---------------------------------------------------------------------------
# Copernicus Data Space Ecosystem (Sentinel-1 GRD)
# ---------------------------------------------------------------------------

class CopernicusClient:
    """Client for the Copernicus Data Space Ecosystem OData / OpenSearch API.

    The Copernicus Data Space provides free access to Sentinel-1 Level-1
    Ground Range Detected (GRD) products via a public REST API.

    Args:
        base_url: Base URL of the OData endpoint (default: official EU
            Copernicus Data Space).
        timeout: HTTP request timeout in seconds (default 30).
    """

    BASE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()

    def search(
        self,
        aoi_wkt: str,
        start_date: str,
        end_date: str,
        product_type: str = "GRD",
        platform: str = "SENTINEL-1",
        max_results: int = 20,
    ) -> list[dict[str, Any]]:
        """Search for Sentinel-1 products intersecting an area of interest.

        Args:
            aoi_wkt: Area of interest as WKT string (e.g. ``POLYGON((...))``).
            start_date: Start of sensing window, ``YYYY-MM-DD``.
            end_date: End of sensing window, ``YYYY-MM-DD``.
            product_type: Product type filter (default ``"GRD"``).
            platform: Satellite platform name (default ``"SENTINEL-1"``).
            max_results: Maximum number of results to return (default 20).

        Returns:
            List of product metadata dicts returned by the API.
        """
        filter_parts = [
            f"Collection/Name eq '{platform}'",
            f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{product_type}')",
            f"ContentDate/Start gt {start_date}T00:00:00.000Z",
            f"ContentDate/Start lt {end_date}T23:59:59.999Z",
            f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}')",
        ]

        params = {
            "$filter": " and ".join(filter_parts),
            "$top": max_results,
            "$orderby": "ContentDate/Start desc",
        }

        url = f"{self._base_url}/Products"
        logger.info("Searching Copernicus catalogue: %s", url)
        resp = self._session.get(url, params=params, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        products: list[dict[str, Any]] = data.get("value", [])
        logger.info("Found %d Sentinel-1 products.", len(products))
        return products

    def download(self, product_id: str, output_dir: Path) -> Path:
        """Download a Sentinel-1 product archive.

        Args:
            product_id: UUID of the product to download.
            output_dir: Local directory where the archive will be saved.

        Returns:
            Path to the downloaded file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        url = f"{self._base_url}/Products({product_id})/$value"
        logger.info("Downloading product %s …", product_id)

        with self._session.get(url, stream=True, timeout=self._timeout) as resp:
            resp.raise_for_status()
            filename = _extract_filename(resp, f"{product_id}.zip")
            dest = output_dir / filename
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    fh.write(chunk)

        logger.info("Product saved to %s", dest)
        return dest


# ---------------------------------------------------------------------------
# ICEYE API client (commercial)
# ---------------------------------------------------------------------------

class ICEYEClient:
    """Minimal client for the ICEYE machine-to-machine SAR imagery API.

    Args:
        api_key: ICEYE API key.  If not provided, read from the
            ``ICEYE_API_KEY`` environment variable.
        base_url: ICEYE API base URL.
        timeout: HTTP request timeout in seconds (default 30).

    Raises:
        ValueError: If no API key can be resolved.
    """

    BASE_URL = "https://api.iceye.com/v1"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        resolved_key = api_key or os.environ.get("ICEYE_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "ICEYE API key is required.  Pass api_key= or set ICEYE_API_KEY."
            )
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {resolved_key}"})

    def search(
        self,
        aoi_wkt: str,
        start_date: str,
        end_date: str,
        product_type: str = "SLC",
    ) -> list[dict[str, Any]]:
        """Search the ICEYE archive for scenes intersecting an AOI.

        Args:
            aoi_wkt: Area of interest as a WKT string.
            start_date: Start date ``YYYY-MM-DD``.
            end_date: End date ``YYYY-MM-DD``.
            product_type: Product type (default ``"SLC"``).

        Returns:
            List of scene metadata dicts.
        """
        params = {
            "aoi": aoi_wkt,
            "start_date": start_date,
            "end_date": end_date,
            "product_type": product_type,
        }
        url = f"{self._base_url}/archive/search"
        logger.info("Searching ICEYE archive…")
        resp = self._session.get(url, params=params, timeout=self._timeout)
        resp.raise_for_status()
        results: list[dict[str, Any]] = resp.json().get("scenes", [])
        logger.info("Found %d ICEYE scenes.", len(results))
        return results

    def order_image(self, scene_id: str) -> str:
        """Place a tasking / download order for a specific scene.

        Args:
            scene_id: ICEYE scene identifier.

        Returns:
            Order ID string.
        """
        url = f"{self._base_url}/orders"
        payload = {"scene_id": scene_id}
        logger.info("Placing ICEYE order for scene %s …", scene_id)
        resp = self._session.post(url, json=payload, timeout=self._timeout)
        resp.raise_for_status()
        order_id: str = resp.json()["order_id"]
        logger.info("Order placed: %s", order_id)
        return order_id


# ---------------------------------------------------------------------------
# Capella Space API client (commercial)
# ---------------------------------------------------------------------------

class CapellaClient:
    """Minimal client for the Capella Space SAR imagery API.

    Args:
        username: Capella Space account username.  If not provided, reads from
            ``CAPELLA_USERNAME``.
        password: Capella Space account password.  If not provided, reads from
            ``CAPELLA_PASSWORD``.
        base_url: Capella API base URL.
        timeout: HTTP request timeout in seconds (default 30).

    Raises:
        ValueError: If credentials cannot be resolved.
    """

    BASE_URL = "https://api.capellaspace.com"

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        base_url: str = BASE_URL,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        resolved_user = username or os.environ.get("CAPELLA_USERNAME", "")
        resolved_pass = password or os.environ.get("CAPELLA_PASSWORD", "")
        if not resolved_user or not resolved_pass:
            raise ValueError(
                "Capella Space credentials are required.  Pass username= / "
                "password= or set CAPELLA_USERNAME / CAPELLA_PASSWORD."
            )
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        self._token: str | None = None
        self._authenticate(resolved_user, resolved_pass)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _authenticate(self, username: str, password: str) -> None:
        """Obtain a bearer token from the Capella Auth endpoint."""
        url = f"{self._base_url}/token"
        resp = self._session.post(
            url,
            json={"username": username, "password": password},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        self._token = resp.json()["accessToken"]
        self._session.headers.update({"Authorization": f"Bearer {self._token}"})
        logger.info("Capella Space authentication successful.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        aoi_wkt: str,
        start_date: str,
        end_date: str,
        collections: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search the Capella STAC catalogue for SAR scenes.

        Args:
            aoi_wkt: Area of interest as WKT string.
            start_date: Start date ``YYYY-MM-DD``.
            end_date: End date ``YYYY-MM-DD``.
            collections: STAC collection IDs to search (defaults to all SAR).

        Returns:
            List of STAC feature dicts.
        """
        url = f"{self._base_url}/catalog/search"
        payload: dict[str, Any] = {
            "intersects": {"type": "Polygon", "coordinates": _wkt_to_coords(aoi_wkt)},
            "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        }
        if collections:
            payload["collections"] = collections

        logger.info("Searching Capella STAC catalogue…")
        resp = self._session.post(url, json=payload, timeout=self._timeout)
        resp.raise_for_status()
        features: list[dict[str, Any]] = resp.json().get("features", [])
        logger.info("Found %d Capella scenes.", len(features))
        return features

    def download(self, asset_url: str, output_dir: Path) -> Path:
        """Download a Capella asset (signed S3 URL).

        Args:
            asset_url: Pre-signed download URL from the Capella STAC response.
            output_dir: Local directory to save the file.

        Returns:
            Path to the downloaded file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading Capella asset…")
        resp = self._session.get(asset_url, stream=True, timeout=self._timeout)
        resp.raise_for_status()
        filename = _extract_filename(resp, "capella_asset.tif")
        dest = output_dir / filename
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)
        logger.info("Asset saved to %s", dest)
        return dest


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _extract_filename(response: requests.Response, fallback: str) -> str:
    """Extract filename from Content-Disposition header, or use fallback."""
    cd = response.headers.get("Content-Disposition", "")
    if "filename=" in cd:
        parts = cd.split("filename=")
        return parts[-1].strip().strip('"')
    return fallback


def _wkt_to_coords(wkt: str) -> list:
    """Very lightweight WKT POLYGON → GeoJSON coordinate array converter.

    Only handles simple (no-hole) WKT polygons of the form
    ``POLYGON((lon lat, ...))`` as typically produced by bbox helpers.

    Args:
        wkt: WKT polygon string.

    Returns:
        Outer ring as a list of ``[lon, lat]`` pairs, suitable for GeoJSON.
    """
    # Strip POLYGON(( … )) wrapper.
    inner = wkt.strip().upper().lstrip("POLYGON").strip().lstrip("(").rstrip(")")
    coords = []
    for pair in inner.split(","):
        parts = pair.strip().split()
        if len(parts) >= 2:
            coords.append([float(parts[0]), float(parts[1])])
    return coords
