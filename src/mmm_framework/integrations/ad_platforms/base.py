"""Common contract for ad-platform spend connectors.

An :class:`AdPlatformConnector` pulls daily/weekly spend (and optionally
impressions/clicks/conversions) for a marketing channel and returns it in MFF
long format so it slots straight into the loader. The concrete connectors in
this package are **stubs**: the contract, auth notes, and the recommended
ingestion path are specified, but the live API calls are left unimplemented.

Why stubs? For MMM the lowest-effort, most robust path is almost never a
hand-rolled API client — it is to land platform exports in BigQuery via a
managed transfer (BigQuery Data Transfer Service, Fivetran, Supermetrics) and
read them through :class:`~mmm_framework.integrations.BigQueryDataSource`. See
``technical-docs/ad-platform-integrations.md``. The stubs make the direct-API
path discoverable and uniform without pretending to ship maintained API clients.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..base import ConnectionStatus, IntegrationError

# MFF long-format columns produced by spend_to_mff (mirrors MFFColumnConfig defaults).
MFF_COLUMNS = [
    "Period",
    "Geography",
    "Product",
    "Campaign",
    "Outlet",
    "Creative",
    "VariableName",
    "VariableValue",
]


class AdPlatformNotImplemented(IntegrationError):
    """Raised by stub connectors whose live API path is not yet implemented."""


@dataclass(frozen=True)
class PlatformInfo:
    """Catalog metadata describing an ad platform's integration story."""

    platform: str  # stable id, e.g. "google_ads"
    label: str  # display name, e.g. "Google Ads"
    status: str  # "stub" | "available"
    ease: str  # "easy" | "moderate" | "hard"  (direct-API integration effort)
    official_sdk: str | None  # pip package for the direct API, if any
    auth: str  # how the direct API authenticates
    recommended_path: str  # the lowest-effort ingestion route (usually BigQuery)
    metrics: list[str] = field(default_factory=list)
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "label": self.label,
            "status": self.status,
            "ease": self.ease,
            "official_sdk": self.official_sdk,
            "auth": self.auth,
            "recommended_path": self.recommended_path,
            "metrics": list(self.metrics),
            "notes": self.notes,
        }


class AdPlatformConnector(ABC):
    """Pull marketing spend from an ad platform into MFF long format."""

    INFO: PlatformInfo  # subclasses set this

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    @property
    def platform(self) -> str:
        return self.INFO.platform

    @abstractmethod
    def fetch_spend(
        self,
        *,
        start: str,
        end: str,
        granularity: str = "weekly",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Return spend (+ optional metrics) for [start, end] in MFF long format."""

    def test_connection(self) -> ConnectionStatus:
        """Default probe for an unimplemented connector."""
        return ConnectionStatus(
            ok=False,
            detail=f"{self.INFO.label} connector is a stub — {self.INFO.recommended_path}",
            source=self.INFO.platform,
            target=self.INFO.label,
        )

    def _not_implemented(self) -> AdPlatformNotImplemented:
        return AdPlatformNotImplemented(
            f"The {self.INFO.label} direct-API connector is a stub. "
            f"Recommended path: {self.INFO.recommended_path}. "
            f"To implement the direct API, use {self.INFO.official_sdk or 'the official SDK'} "
            f"({self.INFO.auth})."
        )


def spend_to_mff(
    df: pd.DataFrame,
    *,
    date_col: str,
    value_cols: dict[str, str],
    geo_col: str | None = None,
    product_col: str | None = None,
    campaign_col: str | None = None,
) -> pd.DataFrame:
    """Reshape a tidy platform export into MFF long format.

    ``value_cols`` maps source metric columns to the MFF ``VariableName`` they
    should become, e.g. ``{"cost": "GoogleAds_Search", "impressions":
    "GoogleAds_Search_impr"}``. Dimension columns are optional; absent ones are
    emitted blank. This is the bridge a (future) live connector — or a user with
    a raw CSV/BigQuery export — uses to reach :func:`mmm_framework.load_mff`.
    """
    if date_col not in df.columns:
        raise IntegrationError(f"date_col {date_col!r} not in DataFrame columns")
    missing = [c for c in value_cols if c not in df.columns]
    if missing:
        raise IntegrationError(f"value_cols not found in DataFrame: {missing}")

    frames: list[pd.DataFrame] = []
    for src_col, var_name in value_cols.items():
        block = pd.DataFrame(
            {
                "Period": df[date_col].values,
                "Geography": df[geo_col].values if geo_col else "",
                "Product": df[product_col].values if product_col else "",
                "Campaign": df[campaign_col].values if campaign_col else "",
                "Outlet": "",
                "Creative": "",
                "VariableName": var_name,
                "VariableValue": pd.to_numeric(df[src_col], errors="coerce").values,
            }
        )
        frames.append(block)
    out = pd.concat(frames, ignore_index=True)
    return out[MFF_COLUMNS]
