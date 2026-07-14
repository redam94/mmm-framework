"""Regression tests for duplicate channel names duplicating rows in the Oracle
ROI and component-decomposition tables.

Root cause (2026-07-13): ``MFFConfig`` had no uniqueness guard on media names,
so a spec/DAG/tool call listing the same channel twice produced duplicate
``media_names`` → duplicate ``panel.coords.channels`` → duplicate
``BayesianMMM.channel_names``. Because ``X_media`` is built as a dict keyed by
name, the duplicate collapsed to ONE data column while the name axis kept BOTH
labels — so ``_get_channel_names`` (which both reporting tables iterate) returned
a longer, duplicated list and every channel rendered twice.

Fixes: (1) ``MFFConfig.dedupe_channel_names`` drops repeat-named media/control
entries at config time; (2) ``reporting.helpers.utils._get_channel_names``
de-dupes defensively for models fitted/saved before the guard.
"""

import numpy as np
import pandas as pd

from mmm_framework.config.enums import DimensionType
from mmm_framework.config.mff import MFFConfig
from mmm_framework.config.variables import (
    ControlVariableConfig,
    KPIConfig,
    MediaChannelConfig,
)
from mmm_framework.data_loader import MFFLoader
from mmm_framework.reporting.helpers.utils import _get_channel_names


def test_mffconfig_dedupes_duplicate_media_and_control_names():
    cfg = MFFConfig(
        kpi=KPIConfig(name="Sales"),
        media_channels=[
            MediaChannelConfig(name="TV"),
            MediaChannelConfig(name="Search"),
            MediaChannelConfig(name="TV"),  # duplicate — dropped
        ],
        controls=[
            ControlVariableConfig(name="Price"),
            ControlVariableConfig(name="Price"),  # duplicate — dropped
        ],
    )
    assert cfg.media_names == ["TV", "Search"]  # first occurrence kept, order preserved
    assert cfg.control_names == ["Price"]
    assert len(cfg.media_channels) == 2


def test_get_channel_names_dedupes_a_prefix_fitted_model():
    class _Model:  # duck-typed model whose axis already carries duplicates
        channel_names = ["TV", "Search", "TV", "Social", "Search"]

    assert _get_channel_names(_Model()) == ["TV", "Search", "Social"]


def test_get_channel_names_falls_back_to_panel_and_handles_empty():
    class _Panel:
        channel_names = ["A", "A", "B"]

    class _ModelNoAxis:  # no channel_names on the model itself → panel fallback
        panel = _Panel()

    assert _get_channel_names(_ModelNoAxis()) == ["A", "B"]
    assert _get_channel_names(object()) == []


def test_report_builder_dedupes_legacy_tripled_rows():
    """A report built from a legacy session whose roi_metrics / decomposition
    were persisted concatenated 2-3× must render each channel/component once."""
    from mmm_framework.agents.report_builder import _dedupe_rows

    roi = [{"channel": "TV", "roi_mean": 0.18}, {"channel": "Search", "roi_mean": 3.44}]
    decomp = [{"component": "Baseline"}, {"component": "TV"}, {"component": "Controls"}]
    assert _dedupe_rows(roi * 3, "channel") == roi
    assert _dedupe_rows(decomp * 3, "component") == decomp
    # Rows lacking the identity key are never collapsed; empty is safe.
    assert _dedupe_rows([{"x": 1}, {"x": 1}], "channel") == [{"x": 1}, {"x": 1}]
    assert _dedupe_rows([], "component") == []


def _national_mff_frame(n=20):
    periods = pd.date_range("2023-01-02", periods=n, freq="W-MON").strftime("%Y-%m-%d")
    rows = []
    for var, base in [
        ("Sales", 1000.0),
        ("TV", 100.0),
        ("Digital", 80.0),
        ("Price", 10.0),
    ]:
        for p in periods:
            rows.append(
                {
                    "VariableName": var,
                    "VariableValue": base + abs(np.random.normal(0, 5)),
                    "Period": p,
                    "Geography": None,
                    "Product": None,
                    "Campaign": None,
                    "Outlet": None,
                    "Creative": None,
                }
            )
    return pd.DataFrame(rows)


def test_loader_panel_axis_matches_columns_with_a_duplicate_channel():
    """End-to-end: a config that repeats a media channel must not yield a panel
    whose channel axis is longer than the actual media matrix (the exact desync
    that duplicated the table rows)."""
    cfg = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
            MediaChannelConfig(name="Digital", dimensions=[DimensionType.PERIOD]),
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),  # dup
        ],
        controls=[
            ControlVariableConfig(name="Price", dimensions=[DimensionType.PERIOD])
        ],
    )
    loader = MFFLoader(cfg)
    loader.load(_national_mff_frame())
    panel = loader.build_panel()

    channels = list(panel.coords.channels)
    assert channels == ["TV", "Digital"]  # de-duplicated, order preserved
    # The name axis length must equal the real media-matrix width (no desync).
    assert len(channels) == panel.X_media.shape[1]
