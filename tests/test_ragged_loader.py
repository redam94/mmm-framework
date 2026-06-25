"""RaggedMFFLoader end-to-end tests (Phase 2 / D2).

The ragged loader was dead code: build_panel() crashed on first use
(passed an unknown explicit_nan_mask= kwarg to PanelDataset and read a
nonexistent MFFConfig.preserve_explicit_nan). It is exported in __all__ and the
agent prompt. These tests pin the repaired behavior: it builds a valid panel,
preserves EXPLICIT NaNs, and fills genuinely-missing rows.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from mmm_framework.config import (
    DimensionType,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
)
from mmm_framework.data_loader import PanelDataset, load_ragged_mff

DIMS = ["Geography", "Product", "Campaign", "Outlet", "Creative"]


def _ragged_df():
    # 10 Sunday-anchored weeks (matches default frequency="W"). Sales is
    # explicitly NaN at week 5; TV's row for week 3 is entirely absent (a gap).
    periods = pd.date_range("2021-01-03", periods=10, freq="W")
    rows = []
    for i, p in enumerate(periods):
        base = {c: "-" for c in DIMS}
        rows.append(
            {
                **base,
                "Period": p.strftime("%Y-%m-%d"),
                "VariableName": "Sales",
                "VariableValue": (np.nan if i == 5 else 100 + i),
            }
        )
        if i != 3:
            rows.append(
                {
                    **base,
                    "Period": p.strftime("%Y-%m-%d"),
                    "VariableName": "TV",
                    "VariableValue": 10 + i,
                }
            )
    return pd.DataFrame(rows)


def _cfg():
    return MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD])
        ],
        frequency="W",
    )


def test_ragged_build_panel_does_not_crash():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = load_ragged_mff(_ragged_df(), _cfg())
    assert isinstance(panel, PanelDataset)
    assert panel.n_obs == 10  # complete weekly index


def test_explicit_nan_preserved_and_tracked():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = load_ragged_mff(_ragged_df(), _cfg())
    # The week-5 Sales value was explicitly NaN in the source -> tracked + kept NaN.
    assert panel.explicit_nan_mask is not None
    assert int(panel.explicit_nan_mask["kpi"].sum()) == 1
    assert bool(np.isnan(np.asarray(panel.y, dtype=float)[5]))


def test_missing_row_is_filled_not_flagged():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = load_ragged_mff(_ragged_df(), _cfg())
    # TV's week-3 row was absent (a gap) -> filled with fill_missing_media (0.0),
    # and NOT counted as an explicit NaN.
    assert float(panel.X_media["TV"].iloc[3]) == 0.0
    assert int(panel.explicit_nan_mask["TV"].sum()) == 0


def test_standard_loader_panel_has_no_nan_mask():
    # The fully-filled standard loader leaves explicit_nan_mask as None.
    from mmm_framework.data_loader import load_mff

    df = _ragged_df()
    df = df[df["VariableValue"].notna()]  # drop the explicit NaN so load_mff is happy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = load_mff(df, _cfg())
    assert panel.explicit_nan_mask is None
