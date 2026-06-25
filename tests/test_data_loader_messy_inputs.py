"""Messy-input regression tests for the MFF loader + the pre-fit data-quality
gate (Phase 1 / D1, QW-1).

Real client exports carry duplicate rows, currency-formatted numbers, and the
odd corrupt date. Previously these silently corrupted the KPI (duplicate rows
summed) or crashed with an opaque TypeError / raw pandas ValueError. These tests
pin the new fail-loud behavior and the data-quality gate on the fit path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.config import (
    DimensionType,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
)
from mmm_framework.data_loader import (
    MFFValidationError,
    coerce_numeric_column,
    load_mff,
    validate_mff_structure,
)

DIM_COLS = ["Geography", "Product", "Campaign", "Outlet", "Creative"]


def _mff_config(duplicate_policy: str = "error") -> MFFConfig:
    return MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD])
        ],
        duplicate_policy=duplicate_policy,
    )


def _long_df(rows: list[dict]) -> pd.DataFrame:
    """Build a complete MFF-long frame; missing dim columns default to '-'."""
    out = []
    for r in rows:
        base = {c: "-" for c in DIM_COLS}
        base.update(r)
        out.append(base)
    return pd.DataFrame(out)


def _clean_rows(n_weeks: int = 8):
    periods = pd.date_range("2021-01-04", periods=n_weeks, freq="W-MON")
    rows = []
    for i, p in enumerate(periods):
        iso = p.strftime("%Y-%m-%d")
        rows.append({"Period": iso, "VariableName": "Sales", "VariableValue": 100 + i})
        rows.append({"Period": iso, "VariableName": "TV", "VariableValue": 10 + i})
    return periods, rows


# ---------------------------------------------------------------------------
# coerce_numeric_column
# ---------------------------------------------------------------------------
def test_coerce_currency_and_thousands():
    s = pd.Series(["$1,234.5", "1,000", "  50 ", "(123)"])
    out = coerce_numeric_column(s, column="VariableValue")
    np.testing.assert_allclose(out.to_numpy(), [1234.5, 1000.0, 50.0, -123.0])


def test_coerce_null_tokens_become_nan():
    s = pd.Series(["", "NA", "n/a", "null", None, "5"])
    out = coerce_numeric_column(s, column="VariableValue")
    assert out.iloc[-1] == 5.0
    assert out.iloc[:-1].isna().all()


def test_coerce_unparseable_raises_clear_error():
    s = pd.Series(["100", "definitely not a number", "200"])
    with pytest.raises(MFFValidationError, match="non-numeric"):
        coerce_numeric_column(s, column="VariableValue")


def test_coerce_numeric_passthrough():
    s = pd.Series([1, 2, 3], dtype="int64")
    out = coerce_numeric_column(s, column="VariableValue")
    assert out.dtype == float and list(out) == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Currency strings through the full loader
# ---------------------------------------------------------------------------
def test_load_mff_accepts_currency_strings():
    periods, rows = _clean_rows()
    for r in rows:
        r["VariableValue"] = f"${r['VariableValue']:,}"  # "$100", "$1,000", ...
    panel = load_mff(_long_df(rows), _mff_config())
    assert np.isfinite(np.asarray(panel.y, dtype=float)).all()


def test_load_mff_unparseable_value_raises():
    periods, rows = _clean_rows()
    rows[2]["VariableValue"] = "oops"
    with pytest.raises(MFFValidationError, match="non-numeric"):
        load_mff(_long_df(rows), _mff_config())


# ---------------------------------------------------------------------------
# Duplicate rows
# ---------------------------------------------------------------------------
def test_duplicate_rows_raise_by_default():
    periods, rows = _clean_rows()
    # Exact-duplicate Sales cell (same full key) -- silent-corruption case.
    dup = dict(rows[0])
    rows.append(dup)
    with pytest.raises(MFFValidationError, match="duplicate"):
        load_mff(_long_df(rows), _mff_config())


def test_duplicate_policy_sum_aggregates():
    periods, rows = _clean_rows()
    rows.append(dict(rows[0]))  # duplicate the first Sales cell (value 100)
    panel = load_mff(_long_df(rows), _mff_config(duplicate_policy="sum"))
    # First period's Sales should be 100 + 100 = 200 under the explicit policy.
    assert float(np.asarray(panel.y, dtype=float)[0]) == pytest.approx(200.0)


def test_finer_granularity_is_aggregated_not_flagged():
    """Two rows for the same period that differ on a dimension the model does NOT
    split on are legitimate finer granularity -- they must be summed, not flagged
    as duplicates."""
    periods, rows = _clean_rows()
    # A second Sales row for period 0, differing only on Outlet (not a config dim).
    extra = dict(rows[0])
    extra["Outlet"] = "OnlineVideo"
    rows[0]["Outlet"] = "LinearTV"
    rows.append(extra)
    panel = load_mff(_long_df(rows), _mff_config())  # default "error" -> must NOT raise
    assert float(np.asarray(panel.y, dtype=float)[0]) == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Date validation
# ---------------------------------------------------------------------------
def test_bad_date_in_later_row_raises_validation_error():
    periods, rows = _clean_rows()
    rows[4]["Period"] = "not-a-date"  # a non-first row
    df = _long_df(rows)
    with pytest.raises(MFFValidationError, match="parse"):
        validate_mff_structure(df, _mff_config())


def test_bad_date_through_load_is_validation_error_not_raw():
    periods, rows = _clean_rows()
    rows[4]["Period"] = "2021-13-99"  # invalid date, non-first row
    with pytest.raises(MFFValidationError):
        load_mff(_long_df(rows), _mff_config())
