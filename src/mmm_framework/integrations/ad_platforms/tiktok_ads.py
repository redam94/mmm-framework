"""TikTok Ads spend connector (stub).

Direct API: the TikTok Business / Marketing API exposes a reporting endpoint
(``/report/integrated/get/``). There is no first-party Python SDK on par with
Google/Meta — integrations are typically hand-rolled over REST with an OAuth
app + long-lived access token, which makes this a moderate-effort connector.

Recommended path: managed connectors (Fivetran, Supermetrics) land TikTok
reporting in BigQuery; read with ``BigQueryDataSource``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import AdPlatformConnector, PlatformInfo

INFO = PlatformInfo(
    platform="tiktok_ads",
    label="TikTok Ads",
    status="stub",
    ease="moderate",
    official_sdk=None,
    auth="OAuth app + long-lived access token (Business API)",
    recommended_path=(
        "land reporting in BigQuery via a managed connector (Fivetran/"
        "Supermetrics) and read with BigQueryDataSource"
    ),
    metrics=["spend", "impressions", "clicks", "conversions"],
    notes="No first-party Python SDK; REST over the Business API.",
)


class TikTokAdsConnector(AdPlatformConnector):
    INFO = INFO

    def fetch_spend(
        self, *, start: str, end: str, granularity: str = "weekly", **kwargs: Any
    ) -> pd.DataFrame:
        raise self._not_implemented()
