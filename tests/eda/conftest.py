"""Shared fixtures/helpers for the pre-fit data-quality (eda) tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def to_mff_long(
    wide: pd.DataFrame,
    *,
    geography: str | None = None,
) -> pd.DataFrame:
    """Convert a Period-indexed wide frame into MFF long rows."""
    records = []
    for period, row in wide.iterrows():
        for var, val in row.items():
            rec = {
                "Period": pd.Timestamp(period).strftime("%Y-%m-%d"),
                "VariableName": var,
                "VariableValue": val,
            }
            if geography is not None:
                rec["Geography"] = geography
            records.append(rec)
    return pd.DataFrame(records)


def simple_wide(n: int = 104, seed: int = 0) -> pd.DataFrame:
    """A small clean wide panel: Sales KPI + 2 media + 1 control."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2022-01-03", periods=n, freq="W-MON")
    t = np.arange(n)
    tv = np.clip(100 + 40 * np.sin(2 * np.pi * t / 9.0) + rng.normal(0, 8, n), 1, None)
    search = np.clip(
        60 + 25 * np.sin(2 * np.pi * t / 7.0 + 1) + rng.normal(0, 5, n), 1, None
    )
    price = 10 + 0.4 * np.cos(2 * np.pi * t / 52.0) + rng.normal(0, 0.05, n)
    sales = (
        300
        + 30 * np.sin(2 * np.pi * t / 52.0)
        + 0.8 * tv
        + 0.9 * search
        - 12 * (price - 10)
        + rng.normal(0, 10, n)
    )
    return pd.DataFrame(
        {"Sales": sales, "TV": tv, "Search": search, "Price": price}, index=weeks
    )


@pytest.fixture
def spec() -> dict:
    return {
        "kpi": "Sales",
        "kpi_level": "national",
        "media_channels": [{"name": "TV"}, {"name": "Search"}],
        "control_variables": [{"name": "Price"}],
    }
