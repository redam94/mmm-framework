"""Meta (Facebook/Instagram) Ads spend connector (stub).

Direct API: the official ``facebook_business`` SDK hits the Marketing API's
Insights endpoint. Auth uses a long-lived System User access token tied to a
Business Manager — no per-user OAuth dance, which makes Meta one of the more
straightforward direct integrations once the token is provisioned.

Recommended path: managed connectors (Fivetran, Supermetrics, Stitch) land Meta
Insights in BigQuery; read with ``BigQueryDataSource``. Or call the Insights
endpoint directly and map with ``spend_to_mff``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import AdPlatformConnector, PlatformInfo

INFO = PlatformInfo(
    platform="meta_ads",
    label="Meta Ads",
    status="stub",
    ease="easy",
    official_sdk="facebook_business",
    auth="long-lived System User access token (Business Manager)",
    recommended_path=(
        "land Insights in BigQuery via Fivetran/Supermetrics and read with "
        "BigQueryDataSource, or call the Marketing API Insights endpoint directly"
    ),
    metrics=["spend", "impressions", "clicks", "actions"],
    notes="Most mature direct SDK; System User token avoids per-user OAuth.",
)


class MetaAdsConnector(AdPlatformConnector):
    INFO = INFO

    def fetch_spend(
        self, *, start: str, end: str, granularity: str = "weekly", **kwargs: Any
    ) -> pd.DataFrame:
        raise self._not_implemented()
