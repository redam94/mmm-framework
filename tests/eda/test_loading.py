"""Tests for the EDA panel loader."""

from __future__ import annotations

import pandas as pd
import pytest

from mmm_framework.eda import load_eda_panel

from .conftest import simple_wide, to_mff_long


def _write_long(tmp_path, wide, geography=None, name="data.csv"):
    path = tmp_path / name
    to_mff_long(wide, geography=geography).to_csv(path, index=False)
    return str(path)


class TestLongFormat:
    def test_roles_from_spec(self, tmp_path, spec):
        path = _write_long(tmp_path, simple_wide())
        panel = load_eda_panel(path, spec)
        assert panel.roles_source == "spec"
        assert panel.kpi == "Sales"
        assert panel.media == ["TV", "Search"]
        assert panel.controls == ["Price"]
        assert panel.unassigned == []
        assert panel.dims == []
        assert panel.freq is not None and panel.freq.startswith("W")
        assert panel.df_wide.shape == (104, 4)

    def test_heuristic_roles_without_spec(self, tmp_path):
        path = _write_long(tmp_path, simple_wide())
        panel = load_eda_panel(path, None)
        assert panel.roles_source == "heuristic"
        assert panel.kpi == "Sales"
        assert "TV" in panel.media and "Search" in panel.media
        assert "Price" in panel.unassigned

    def test_spec_with_bare_string_vars(self, tmp_path):
        path = _write_long(tmp_path, simple_wide())
        panel = load_eda_panel(
            path,
            {
                "kpi": "Sales",
                "media_channels": ["TV", "Search"],
                "control_variables": ["Price"],
            },
        )
        assert panel.media == ["TV", "Search"]
        assert panel.controls == ["Price"]

    def test_unassigned_variables_are_kept(self, tmp_path, spec):
        wide = simple_wide()
        wide["Mystery"] = 1.5
        path = _write_long(tmp_path, wide)
        panel = load_eda_panel(path, spec)
        assert "Mystery" in panel.variables
        assert "Mystery" in panel.unassigned

    def test_duplicate_rows_counted_not_aggregated(self, tmp_path, spec):
        wide = simple_wide(n=20)
        long = to_mff_long(wide)
        dup = long.iloc[[0]].copy()
        dup["VariableValue"] = 9999.0
        long = pd.concat([long, dup], ignore_index=True)
        path = tmp_path / "dup.csv"
        long.to_csv(path, index=False)
        panel = load_eda_panel(str(path), spec)
        assert panel.duplicate_rows == 1
        # First occurrence kept — the 9999 duplicate must not be summed in.
        first_var = long.iloc[0]["VariableName"]
        first_period = pd.Timestamp(long.iloc[0]["Period"])
        assert panel.df_wide.loc[first_period, first_var] != pytest.approx(
            wide.iloc[0][first_var] + 9999.0
        )

    def test_geo_panel_multiindex_and_slices(self, tmp_path, spec):
        east = to_mff_long(simple_wide(seed=1), geography="East")
        west = to_mff_long(simple_wide(seed=2), geography="West")
        path = tmp_path / "geo.csv"
        pd.concat([east, west], ignore_index=True).to_csv(path, index=False)
        panel = load_eda_panel(str(path), spec)
        assert panel.dims == ["Geography"]
        assert panel.is_panel
        slices = list(panel.slices("TV"))
        assert len(slices) == 2
        dim_values = sorted(s[0]["Geography"] for s in slices)
        assert dim_values == ["East", "West"]
        for _, series in slices:
            assert len(series) == 104
            assert isinstance(series.index, pd.DatetimeIndex)


class TestWideFormat:
    def test_wide_csv_fallback(self, tmp_path, spec):
        wide = simple_wide()
        path = tmp_path / "wide.csv"
        wide.rename_axis("Date").reset_index().to_csv(path, index=False)
        panel = load_eda_panel(str(path), spec)
        assert panel.df_long is None
        assert panel.kpi == "Sales"
        assert panel.df_wide.shape == (104, 4)
        assert panel.freq is not None and panel.freq.startswith("W")

    def test_wide_csv_without_date_column_raises(self, tmp_path):
        path = tmp_path / "nodate.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="date column"):
            load_eda_panel(str(path), None)
