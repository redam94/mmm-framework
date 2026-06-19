"""Google Ads spend connector (stub).

Direct API: the official ``google-ads`` Python client runs GAQL queries against
the Google Ads API. Auth needs an OAuth2 client + refresh token AND a
developer token (the developer token requires approval, which is the main
friction). On GCP the OAuth flow can reuse the same project identity as Vertex.

Recommended path: BigQuery Data Transfer Service has a **native Google Ads
connector** that lands campaign/spend tables in BigQuery on a schedule — then
read them with ``BigQueryDataSource``. Zero API client to maintain.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import AdPlatformConnector, PlatformInfo

INFO = PlatformInfo(
    platform="google_ads",
    label="Google Ads",
    status="stub",
    ease="moderate",
    official_sdk="google-ads",
    auth="OAuth2 client + refresh token + an approved developer token",
    recommended_path=(
        "land spend in BigQuery via the BigQuery Data Transfer Service "
        "Google Ads connector, then read it with BigQueryDataSource"
    ),
    metrics=["cost", "impressions", "clicks", "conversions"],
    notes="Best GCP fit: BQ Data Transfer is first-party and needs no API client.",
)


class GoogleAdsConnector(AdPlatformConnector):
    INFO = INFO

    def fetch_spend(
        self, *, start: str, end: str, granularity: str = "weekly", **kwargs: Any
    ) -> pd.DataFrame:
        raise self._not_implemented()
