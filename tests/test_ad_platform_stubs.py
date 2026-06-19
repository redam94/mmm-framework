"""Ad-platform connector stubs + the spend_to_mff reshape helper."""

from __future__ import annotations

import pandas as pd
import pytest

from mmm_framework.integrations import IntegrationError
from mmm_framework.integrations.ad_platforms import (
    AdPlatformNotImplemented,
    GoogleAdsConnector,
    build_ad_platform,
    list_ad_platforms,
    spend_to_mff,
)
from mmm_framework.integrations.ad_platforms.base import MFF_COLUMNS


def test_catalog_ranked_easiest_first():
    cat = list_ad_platforms()
    ids = [c["platform"] for c in cat]
    assert set(ids) == {"google_ads", "meta_ads", "tiktok_ads"}
    # Meta is the only "easy" one -> sorts first.
    assert ids[0] == "meta_ads"
    for c in cat:
        assert c["status"] == "stub"
        assert c["ease"] in {"easy", "moderate", "hard"}
        assert isinstance(c["sdk_installed"], bool)
        assert c["recommended_path"] and "BigQuery" in c["recommended_path"]
        assert isinstance(c["metrics"], list) and c["metrics"]


def test_build_and_unknown():
    conn = build_ad_platform("google_ads")
    assert isinstance(conn, GoogleAdsConnector)
    assert conn.platform == "google_ads"
    with pytest.raises(IntegrationError):
        build_ad_platform("snapchat")


def test_stub_fetch_raises_with_guidance():
    conn = build_ad_platform("tiktok_ads")
    with pytest.raises(AdPlatformNotImplemented) as ei:
        conn.fetch_spend(start="2023-01-01", end="2023-03-31")
    assert "BigQuery" in str(ei.value)
    status = conn.test_connection()
    assert status.ok is False and status.source == "tiktok_ads"


def test_spend_to_mff_shape_and_mapping():
    df = pd.DataFrame(
        {
            "date": ["2023-01-02", "2023-01-09"],
            "region": ["East", "West"],
            "cost": [5000, 4500],
            "impressions": [100000, 90000],
        }
    )
    mff = spend_to_mff(
        df,
        date_col="date",
        value_cols={"cost": "GoogleAds_Search", "impressions": "GoogleAds_Search_impr"},
        geo_col="region",
    )
    assert list(mff.columns) == MFF_COLUMNS
    assert len(mff) == 4  # 2 rows x 2 metrics
    assert set(mff["VariableName"]) == {"GoogleAds_Search", "GoogleAds_Search_impr"}
    spend_east = mff[
        (mff["VariableName"] == "GoogleAds_Search") & (mff["Geography"] == "East")
    ]
    assert spend_east["VariableValue"].iloc[0] == 5000
    # Unused dimensions are blank, not dropped.
    assert (mff["Product"] == "").all()


def test_spend_to_mff_validates_columns():
    df = pd.DataFrame({"date": ["2023-01-02"], "cost": [1]})
    with pytest.raises(IntegrationError):
        spend_to_mff(df, date_col="missing", value_cols={"cost": "X"})
    with pytest.raises(IntegrationError):
        spend_to_mff(df, date_col="date", value_cols={"nope": "X"})
