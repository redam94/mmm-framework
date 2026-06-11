"""Tests for mmm_framework.synth.mff — scenario -> MFF flattening."""

import json

import numpy as np
import pytest

from mmm_framework.synth import MFF_COLUMNS, generate_mff
from mmm_framework.synth import dgp, dgp_geo
from mmm_framework.synth.mff import scenario_to_mff


def test_national_mff_shape_and_columns():
    df, truth = generate_mff("clean", seed=0)
    assert list(df.columns) == MFF_COLUMNS
    sc = dgp.make_clean(seed=0)
    n_vars = 1 + len(sc.spend.columns) + len(sc.controls.columns)
    assert len(df) == len(sc.weeks) * n_vars
    assert df["Geography"].isna().all()
    assert set(truth["channels"]) <= set(df["VariableName"].unique())
    assert "Sales" in set(df["VariableName"])


def test_mff_values_match_scenario():
    sc = dgp.make_clean(seed=0)
    df = scenario_to_mff(sc)
    sales = df[df.VariableName == "Sales"].sort_values("Period")
    np.testing.assert_allclose(sales.VariableValue.to_numpy(), sc.y.to_numpy())
    tv = df[df.VariableName == "TV"].sort_values("Period")
    np.testing.assert_allclose(tv.VariableValue.to_numpy(), sc.spend["TV"].to_numpy())


def test_truth_summary_is_json_safe():
    for name in ("realistic", "mixed_data_errors", "dense_controls"):
        _, truth = generate_mff(name)
        json.dumps(truth)  # must not raise
        assert truth["scenario"] == name
        assert set(truth["true_roas"]) == set(truth["channels"])


def test_n_weeks_threads_through():
    df, _ = generate_mff("realistic", seed=1, n_weeks=120)
    assert df["Period"].nunique() == 120


def test_n_weeks_minimum_enforced():
    with pytest.raises(ValueError, match="at least 52"):
        generate_mff("clean", n_weeks=10)


def test_unknown_scenario_raises():
    with pytest.raises(KeyError, match="Unknown scenario"):
        generate_mff("not_a_world")


def test_geographies_upgrade_to_panel():
    geos = ["Northeast", "Midwest", "Pacific"]
    df, truth = generate_mff("realistic", seed=3, geographies=geos, n_weeks=104)
    assert truth["scenario"] == "geo_heterogeneous"
    assert sorted(df["Geography"].dropna().unique()) == sorted(geos)
    assert set(truth["true_contribution_by_geo"]) == set(geos)
    # every geo has every variable at every period
    counts = df.groupby("Geography", dropna=False)["VariableValue"].count()
    assert counts.nunique() == 1


def test_geo_product_includes_product_dimension():
    df, truth = generate_mff("geo_product")
    assert df["Product"].dropna().nunique() == 2
    assert truth["products"] == ["Core", "Premium"]
    json.dumps(truth)


def test_custom_geos_seeded_reproducibly():
    a = dgp_geo.make_geo_clean(seed=5, geos=["A", "B"], n_weeks=80)
    b = dgp_geo.make_geo_clean(seed=5, geos=["A", "B"], n_weeks=80)
    np.testing.assert_allclose(a.y.to_numpy(), b.y.to_numpy())
    assert a.geos == ["A", "B"]
