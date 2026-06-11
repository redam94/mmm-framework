"""Each validator check fires on a frame seeded with exactly that defect."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmm_framework.eda import load_eda_panel
from mmm_framework.eda.validators import validate_dataset

from .conftest import simple_wide, to_mff_long


def make_panel(tmp_path, wide=None, spec=None, geography=None, long=None, name="v.csv"):
    if long is None:
        long = to_mff_long(
            wide if wide is not None else simple_wide(), geography=geography
        )
    path = tmp_path / name
    long.to_csv(path, index=False)
    return load_eda_panel(str(path), spec)


def checks_fired(report, severity=None):
    issues = report.issues if severity is None else report.by_severity(severity)
    return {i.check for i in issues}


class TestCleanData:
    def test_clean_frame_passes(self, tmp_path, spec):
        report = validate_dataset(make_panel(tmp_path, simple_wide(), spec), spec=spec)
        assert report.passed, report.summary()
        assert checks_fired(report, "error") == set()


class TestIndividualChecks:
    def test_date_gaps(self, tmp_path, spec):
        wide = simple_wide()
        wide = wide.drop(wide.index[[10, 11, 40]])
        report = validate_dataset(make_panel(tmp_path, wide, spec), spec=spec)
        assert "date_gaps" in checks_fired(report, "error")
        gap = next(i for i in report.issues if i.check == "date_gaps")
        assert len(gap.affected) == 3

    def test_duplicate_rows(self, tmp_path, spec):
        long = to_mff_long(simple_wide(n=20))
        long = pd.concat([long, long.iloc[[3]]], ignore_index=True)
        report = validate_dataset(make_panel(tmp_path, long=long, spec=spec), spec=spec)
        assert "duplicate_rows" in checks_fired(report, "error")

    def test_constant_series(self, tmp_path, spec):
        wide = simple_wide()
        wide["Price"] = 10.0
        report = validate_dataset(make_panel(tmp_path, wide, spec), spec=spec)
        issues = [i for i in report.issues if i.check == "constant_series"]
        assert [i.variable for i in issues] == ["Price"]

    def test_missingness_warning_and_error(self, tmp_path, spec):
        wide = simple_wide()
        wide.loc[wide.index[:8], "Search"] = np.nan  # ~7.7% -> warning
        wide.loc[wide.index[: int(0.4 * len(wide))], "TV"] = np.nan  # 40% -> error
        long = to_mff_long(wide).dropna(subset=["VariableValue"])
        report = validate_dataset(make_panel(tmp_path, long=long, spec=spec), spec=spec)
        by_var = {
            i.variable: i.severity for i in report.issues if i.check == "missingness"
        }
        assert by_var.get("TV") == "error"
        assert by_var.get("Search") == "warning"

    def test_negative_spend(self, tmp_path, spec):
        wide = simple_wide()
        wide.loc[wide.index[5], "TV"] = -120.0
        report = validate_dataset(make_panel(tmp_path, wide, spec), spec=spec)
        neg = [i for i in report.issues if i.check == "negative_spend"]
        assert len(neg) == 1 and neg[0].severity == "error"
        assert neg[0].variable == "TV"

    def test_negative_control_is_not_flagged(self, tmp_path, spec):
        wide = simple_wide()
        wide["Price"] = wide["Price"] - 20  # negative control values are fine
        report = validate_dataset(make_panel(tmp_path, wide, spec), spec=spec)
        assert all(
            i.variable != "Price" for i in report.issues if i.check == "negative_spend"
        )

    def test_zero_inflation(self, tmp_path, spec):
        wide = simple_wide()
        wide.loc[wide.index[: int(0.7 * len(wide))], "Search"] = 0.0
        report = validate_dataset(make_panel(tmp_path, wide, spec), spec=spec)
        zi = [i for i in report.issues if i.check == "zero_inflation"]
        assert [i.variable for i in zi] == ["Search"]
        assert zi[0].severity == "warning"

    def test_scale_pathology(self, tmp_path, spec):
        wide = simple_wide()
        wide["TV"] = wide["TV"] * 1e6  # dollars vs $000s mismatch
        report = validate_dataset(make_panel(tmp_path, wide, spec), spec=spec)
        assert "scale_pathology" in checks_fired(report, "warning")

    def test_short_history(self, tmp_path):
        spec6 = {
            "kpi": "Sales",
            "media_channels": [{"name": "TV"}, {"name": "Search"}]
            + [{"name": f"M{i}"} for i in range(4)],
            "control_variables": [{"name": "Price"}],
        }
        wide = simple_wide(n=30)
        for i in range(4):
            wide[f"M{i}"] = np.random.default_rng(i).uniform(1, 100, 30)
        report = validate_dataset(make_panel(tmp_path, wide, spec6), spec=spec6)
        assert "short_history" in checks_fired(report, "warning")

    def test_long_history_no_warning(self, tmp_path, spec):
        report = validate_dataset(
            make_panel(tmp_path, simple_wide(156), spec), spec=spec
        )
        assert "short_history" not in checks_fired(report)

    def test_panel_consistency_variable_absent_in_one_geo(self, tmp_path, spec):
        east = to_mff_long(simple_wide(seed=1), geography="East")
        west_wide = simple_wide(seed=2).drop(columns=["Search"])
        west = to_mff_long(west_wide, geography="West")
        long = pd.concat([east, west], ignore_index=True)
        report = validate_dataset(make_panel(tmp_path, long=long, spec=spec), spec=spec)
        pc = [i for i in report.issues if i.check == "panel_consistency"]
        assert any(i.variable == "Search" and "West" in i.affected for i in pc)

    def test_panel_consistency_misaligned_ranges(self, tmp_path, spec):
        east = to_mff_long(simple_wide(seed=1), geography="East")
        west = to_mff_long(simple_wide(seed=2, n=80), geography="West")
        long = pd.concat([east, west], ignore_index=True)
        report = validate_dataset(make_panel(tmp_path, long=long, spec=spec), spec=spec)
        pc = [i for i in report.issues if i.check == "panel_consistency"]
        assert any("period ranges" in i.message for i in pc)
