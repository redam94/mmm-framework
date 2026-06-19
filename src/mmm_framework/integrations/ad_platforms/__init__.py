"""Ad-platform spend connectors (stubs) + the MFF reshape helper.

from mmm_framework.integrations.ad_platforms import list_ad_platforms, spend_to_mff
"""

from __future__ import annotations

from .base import (
    AdPlatformConnector,
    AdPlatformNotImplemented,
    PlatformInfo,
    spend_to_mff,
)
from .google_ads import GoogleAdsConnector
from .meta_ads import MetaAdsConnector
from .registry import (
    ad_platform_ids,
    build_ad_platform,
    list_ad_platforms,
)
from .tiktok_ads import TikTokAdsConnector

__all__ = [
    "AdPlatformConnector",
    "AdPlatformNotImplemented",
    "PlatformInfo",
    "spend_to_mff",
    "GoogleAdsConnector",
    "MetaAdsConnector",
    "TikTokAdsConnector",
    "build_ad_platform",
    "list_ad_platforms",
    "ad_platform_ids",
]
