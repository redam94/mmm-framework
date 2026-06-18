"""Smoke tests: every chart builder returns a serializable, capped go.Figure."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from mmm_framework.eda import (
    collinearity_analysis,
    decompose_series,
    detect_outliers,
    load_eda_panel,
    missingness_matrix,
    spend_share,
    stationarity_tests,
)
from mmm_framework.eda.charts import (
    fig_before_after,
    fig_correlation_heatmap,
    fig_decomposition,
    fig_distributions,
    fig_kpi_vs_media,
    fig_missingness,
    fig_outlier_series,
    fig_outlier_severity,
    fig_spend_share,
    fig_sparkline_grid,
    fig_stationarity,
    fig_vif,
)

from .conftest import simple_wide, to_mff_long

PLOT_MAX_BYTES = 5 * 1024 * 1024  # mirrors workspace store_plot default cap


@pytest.fixture(scope="module")
def panel(tmp_path_factory):
    """A 156-week, 6-variable, 3-geo panel — the heavy display case."""
    tmp = tmp_path_factory.mktemp("charts")
    frames = []
    for seed, geo in enumerate(["East", "West", "Central"]):
        wide = simple_wide(n=156, seed=seed)
        wide["Social"] = np.random.default_rng(seed + 10).uniform(5, 80, 156)
        wide["Display"] = np.random.default_rng(seed + 20).uniform(5, 60, 156)
        frames.append(to_mff_long(wide, geography=geo))
    path = tmp / "panel.csv"
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    spec = {
        "kpi": "Sales",
        "media_channels": [{"name": c} for c in ("TV", "Search", "Social", "Display")],
        "control_variables": [{"name": "Price"}],
    }
    return load_eda_panel(str(path), spec)


def assert_valid_figure(fig):
    assert isinstance(fig, go.Figure)
    payload = json.loads(fig.to_json())
    assert isinstance(payload.get("data"), list) and payload["data"]
    assert len(fig.to_json().encode()) < PLOT_MAX_BYTES
    return payload


def test_sparkline_grid(panel):
    assert_valid_figure(fig_sparkline_grid(panel))


def test_distributions(panel):
    assert_valid_figure(fig_distributions(panel))


def test_missingness(panel):
    assert_valid_figure(fig_missingness(missingness_matrix(panel)))


def test_correlation_heatmap(panel):
    result = collinearity_analysis(panel)
    assert_valid_figure(fig_correlation_heatmap(result["correlation"]))


def test_vif(panel):
    result = collinearity_analysis(panel)
    assert result["vif"]
    assert_valid_figure(fig_vif(result["vif"]))


def test_spend_share(panel):
    share = spend_share(panel)
    assert share["hhi"] is not None
    assert_valid_figure(
        fig_spend_share(share["share_over_time"], share["shares"], share["hhi"])
    )


def test_decomposition(panel):
    series = panel.df_wide["Sales"].groupby(level="Period").sum()
    result = decompose_series(series, 52, variable="Sales")
    assert result.method == "stl"
    assert_valid_figure(fig_decomposition(result))


def test_decomposition_fallback_short_series(panel):
    series = panel.df_wide["Sales"].groupby(level="Period").sum().iloc[:60]
    result = decompose_series(series, 52, variable="Sales")
    assert result.method == "rolling_median"
    assert result.seasonal is None
    assert_valid_figure(fig_decomposition(result))


def test_kpi_vs_media(panel):
    assert_valid_figure(fig_kpi_vs_media(panel))


def test_stationarity(panel):
    series = panel.df_wide["Sales"].groupby(level="Period").sum()
    results = {"Sales": stationarity_tests(series)}
    assert results["Sales"]["verdict"] != "insufficient_data"
    assert_valid_figure(fig_stationarity(results))


def test_outlier_series_and_severity(panel):
    report = detect_outliers(panel)
    series = panel.df_wide["Sales"].groupby(level="Period").sum()
    fig = fig_outlier_series(
        series, [f for f in report.flags if f.variable == "Sales"], variable="Sales"
    )
    assert_valid_figure(fig)
    if report.flags:
        assert_valid_figure(fig_outlier_severity(report.flags))


def test_before_after(panel):
    series = panel.df_wide["TV"].groupby(level="Period").sum()
    treated = series.clip(upper=float(series.quantile(0.99)))
    assert_valid_figure(fig_before_after(series, treated, "TV"))
