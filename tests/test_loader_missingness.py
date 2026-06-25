"""The main loader surfaces missingness instead of filling it silently (D3).

The production MFFLoader fills missing media/control cells (and tolerates KPI
gaps) — previously without any signal. These tests pin that it now warns and
records what was filled.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from mmm_framework.config import (
    ControlVariableConfig,
    DimensionType,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
)
from mmm_framework.data_loader import load_mff

DIMS = ["Geography", "Product", "Campaign", "Outlet", "Creative"]


def _rows(periods, *, kpi=True, tv_skip=(), price=False):
    rows = []
    for i, p in enumerate(periods):
        base = {c: "-" for c in DIMS}
        iso = p.strftime("%Y-%m-%d")
        if kpi:
            rows.append({**base, "Period": iso, "VariableName": "Sales",
                         "VariableValue": 100 + i})
        if i not in tv_skip:
            rows.append({**base, "Period": iso, "VariableName": "TV",
                         "VariableValue": 10 + i})
        if price:
            rows.append({**base, "Period": iso, "VariableName": "Price",
                         "VariableValue": 5.0 + i})
    return pd.DataFrame(rows)


def _cfg(controls=False):
    return MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD])],
        controls=(
            [ControlVariableConfig(name="Price", dimensions=[DimensionType.PERIOD])]
            if controls
            else []
        ),
        frequency="W",
    )


def test_media_gap_fill_warns_and_records():
    periods = pd.date_range("2021-01-03", periods=8, freq="W")
    # TV missing for weeks 2 and 4 (rows absent) -> filled with 0, must warn.
    df = _rows(periods, tv_skip=(2, 4))
    with pytest.warns(UserWarning, match="missing and filled"):
        panel = load_mff(df, _cfg())
    assert panel.media_stats["TV"]["filled_count"] == 2
    assert panel.media_stats["TV"]["filled_pct"] == pytest.approx(2 / 8)


def test_clean_media_no_fill_warning():
    periods = pd.date_range("2021-01-03", periods=8, freq="W")
    df = _rows(periods)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # no warning expected
        panel = load_mff(df, _cfg())
    assert panel.media_stats["TV"]["filled_count"] == 0


def test_noncontiguous_kpi_warns():
    # 8 weeks of dates but drop week 4 entirely from BOTH series -> KPI gap.
    full = pd.date_range("2021-01-03", periods=8, freq="W")
    keep = full.delete(4)
    df = _rows(keep)
    with pytest.warns(UserWarning, match="non-contiguous"):
        load_mff(df, _cfg())
