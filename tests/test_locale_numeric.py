"""Locale-aware numeric parsing (Phase 4 / T-LOC).

Non-US client data uses EU number formatting (1.234,56). The loader must ingest
it; the default '.' path stays byte-identical to the US behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.config import DimensionType, KPIConfig, MediaChannelConfig, MFFConfig
from mmm_framework.data_loader import coerce_numeric_column, load_mff

DIMS = ["Geography", "Product", "Campaign", "Outlet", "Creative"]


def _coerce(values, decimal):
    return coerce_numeric_column(
        pd.Series(values), column="VariableValue", decimal=decimal
    ).tolist()


def test_default_dot_unchanged():
    # US formatting + the D1 cases stay identical under the default.
    assert _coerce(["1,234.5", "1,000", "$2,500.50"], ".") == [1234.5, 1000.0, 2500.50]


def test_eu_comma_decimal():
    assert _coerce(["1.234,56", "1,5", "€1.000,00"], ",") == [1234.56, 1.5, 1000.0]


def test_auto_detects_per_value():
    assert _coerce(["1.234,56"], "auto") == [1234.56]  # both -> last is decimal
    assert _coerce(["1,234.56"], "auto") == [1234.56]
    assert _coerce(["1,5"], "auto") == [1.5]  # lone comma, 1 trailing -> decimal
    assert _coerce(["1,234"], "auto") == [1234.0]  # lone comma, 3 trailing -> thousands
    assert _coerce(["1234.5"], "auto") == [1234.5]


def test_eu_accounting_negative():
    assert _coerce(["(1.234,56)"], ",") == [-1234.56]


def test_load_mff_with_eu_decimal():
    periods = pd.date_range("2021-01-04", periods=6, freq="W-MON")
    rows = []
    for i, p in enumerate(periods):
        base = {c: "-" for c in DIMS}
        iso = p.strftime("%Y-%m-%d")
        rows.append({**base, "Period": iso, "VariableName": "Sales",
                     "VariableValue": f"{1000 + i}.{i:02d}".replace(".", ",")})
        rows.append({**base, "Period": iso, "VariableName": "TV",
                     "VariableValue": f"{10 + i},5"})
    df = pd.DataFrame(rows)
    cfg = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD])],
        decimal_separator=",",
    )
    panel = load_mff(df, cfg)
    assert np.isfinite(np.asarray(panel.y, dtype=float)).all()
    assert float(np.asarray(panel.X_media["TV"])[0]) == pytest.approx(10.5)
