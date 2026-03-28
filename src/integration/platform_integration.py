"""
Step 6 – Platform Integration
Pushes corrected SAR wake detections to an external intelligence platform via
REST API / Webhooks, and ingests image chips via a Media Set upload endpoint.

Usage::

    from src.integration.platform_integration import PlatformClient

    client = PlatformClient(
        base_url="https://platform.example.com/api",
        api_key="your_platform_api_key",
    )

    # Push detections as objects
    client.push_detections(detections)

    # Upload image chip for analyst review
    client.upload_image_chip(
        image_path=Path("chips/detection_001.png"),
        metadata={"detection_id": "001"},
    )

    # Stream via webhook
    client.send_webhook(detections, webhook_url="https://hooks.example.com/sar")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30  # seconds


class PlatformClient:
    """REST client for pushing SAR wake detections to an intelligence platform.

    Supports:
    * Detection object ingestion via a JSON REST endpoint.
    * Image chip upload via a multipart Media Set endpoint.
    * Outbound webhook delivery.

    Args:
        base_url: Base URL of the platform REST API.
        api_key: API key for authentication.  Falls back to
            ``PLATFORM_API_KEY`` environment variable.
        timeout: HTTP request timeout in seconds (default 30).

    Raises:
        ValueError: If no API key can be resolved.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        resolved_key = api_key or os.environ.get("PLATFORM_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "Platform API key is required.  Pass api_key= or set PLATFORM_API_KEY."
            )
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            }
        )

    # ------------------------------------------------------------------
    # Detection ingestion
    # ------------------------------------------------------------------

    def push_detections(
        self,
        detections: list[dict[str, Any]],
        dataset_id: str = "sar_wake_detections",
    ) -> dict[str, Any]:
        """Push a list of corrected wake detections to the platform.

        Each detection dict should contain at minimum ``"lon"``, ``"lat"``,
        ``"confidence"``, and ``"class"`` fields as produced by the geolocation
        correction pipeline.

        Args:
            detections: List of detection metadata dicts.
            dataset_id: Platform dataset / layer identifier to write to
                (default ``"sar_wake_detections"``).

        Returns:
            Platform API response as a dict.
        """
        payload = {
            "dataset_id": dataset_id,
            "detections": detections,
        }
        url = f"{self._base_url}/detections"
        logger.info(
            "Pushing %d detection(s) to platform dataset '%s' …",
            len(detections),
            dataset_id,
        )
        resp = self._session.post(url, json=payload, timeout=self._timeout)
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        logger.info("Platform ingestion response: %s", result)
        return result

    # ------------------------------------------------------------------
    # Image chip ingestion (Media Sets)
    # ------------------------------------------------------------------

    def upload_image_chip(
        self,
        image_path: Path,
        metadata: dict[str, Any] | None = None,
        media_set_id: str = "sar_chips",
    ) -> dict[str, Any]:
        """Upload a SAR image chip to the platform Media Set.

        Args:
            image_path: Path to the image chip file on disk.
            metadata: Optional key/value metadata to attach to the media item.
            media_set_id: Platform Media Set identifier (default
                ``"sar_chips"``).

        Returns:
            Platform API response as a dict.
        """
        url = f"{self._base_url}/media-sets/{media_set_id}/items"
        # Remove the Content-Type header for multipart uploads; requests sets
        # the correct boundary automatically.
        headers = {
            k: v
            for k, v in self._session.headers.items()
            if k.lower() != "content-type"
        }

        with open(image_path, "rb") as fh:
            files = {"file": (image_path.name, fh, "image/png")}
            data = {"metadata": str(metadata or {})}
            logger.info("Uploading image chip %s to media set '%s' …", image_path.name, media_set_id)
            resp = self._session.post(
                url,
                headers=headers,
                files=files,
                data=data,
                timeout=self._timeout,
            )
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        logger.info("Image chip uploaded: %s", result)
        return result

    # ------------------------------------------------------------------
    # Webhook delivery
    # ------------------------------------------------------------------

    def send_webhook(
        self,
        detections: list[dict[str, Any]],
        webhook_url: str,
        event_type: str = "sar_wake_detection",
    ) -> int:
        """Deliver detections to an outbound webhook endpoint.

        Args:
            detections: List of detection metadata dicts.
            webhook_url: Destination webhook URL.
            event_type: Event type label included in the payload (default
                ``"sar_wake_detection"``).

        Returns:
            HTTP status code returned by the webhook endpoint.
        """
        payload = {
            "event": event_type,
            "count": len(detections),
            "detections": detections,
        }
        # Webhook calls use a fresh session (no auth header).
        logger.info(
            "Sending webhook with %d detection(s) → %s", len(detections), webhook_url
        )
        resp = requests.post(webhook_url, json=payload, timeout=self._timeout)
        resp.raise_for_status()
        logger.info("Webhook delivered, status %d", resp.status_code)
        return resp.status_code
