"""Registry + catalog for ad-platform connectors."""

from __future__ import annotations

from typing import Any

from ..base import IntegrationError, dependency_installed
from .base import AdPlatformConnector
from .google_ads import GoogleAdsConnector
from .meta_ads import MetaAdsConnector
from .tiktok_ads import TikTokAdsConnector

# probe_module = the direct-API SDK; None means "no first-party SDK".
_REGISTRY: dict[str, dict[str, Any]] = {
    "google_ads": {"cls": GoogleAdsConnector, "probe_module": "google.ads.googleads"},
    "meta_ads": {"cls": MetaAdsConnector, "probe_module": "facebook_business"},
    "tiktok_ads": {"cls": TikTokAdsConnector, "probe_module": None},
}


def ad_platform_ids() -> list[str]:
    return list(_REGISTRY)


def build_ad_platform(
    platform: str, config: dict[str, Any] | None = None
) -> AdPlatformConnector:
    spec = _REGISTRY.get(platform)
    if spec is None:
        raise IntegrationError(
            f"Unknown ad platform {platform!r}; known: {', '.join(_REGISTRY)}"
        )
    return spec["cls"](config)


def list_ad_platforms() -> list[dict[str, Any]]:
    """Catalog metadata for every ad platform, ranked easiest-first."""
    rank = {"easy": 0, "moderate": 1, "hard": 2}
    out: list[dict[str, Any]] = []
    for platform, spec in _REGISTRY.items():
        info = spec["cls"].INFO
        probe = spec["probe_module"]
        d = info.as_dict()
        d["sdk_installed"] = dependency_installed(probe) if probe else False
        out.append(d)
    out.sort(key=lambda d: (rank.get(d["ease"], 9), d["label"]))
    return out
