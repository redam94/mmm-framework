"""Sanity checks for the geo / geo x product synthetic worlds."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on path

from tests.synth import dgp_geo  # noqa: E402


@pytest.mark.parametrize("name", list(dgp_geo.SCENARIOS))
def test_world_shapes_and_truth_consistency(name):
    sc = dgp_geo.build(name)
    n_cells = len(sc.geos) * (len(sc.products) if sc.products else 1)
    n_obs = len(sc.weeks) * n_cells

    assert len(sc.y) == n_obs
    assert sc.spend.shape == (n_obs, 4)
    assert list(sc.spend.index.names)[:2] == ["Period", "Geography"]
    assert (sc.y > 0).all()
    assert (sc.spend.to_numpy() > 0).all()

    # Per-cell truth sums to the national truth.
    np.testing.assert_allclose(
        sc.true_contribution_by_geo.sum(axis=0).to_numpy(),
        sc.true_contribution.to_numpy(),
        rtol=1e-10,
    )
    assert len(sc.cells) == n_cells
    # Media truly matters in every cell (no degenerate world).
    assert (sc.true_contribution_by_geo.to_numpy() > 0).all()


@pytest.mark.parametrize("name", list(dgp_geo.SCENARIOS))
def test_panel_builds_and_model_constructs(name):
    from mmm_framework.config import InferenceMethod, ModelConfig
    from mmm_framework.model import BayesianMMM, TrendConfig, TrendType

    sc = dgp_geo.build(name)
    panel = sc.panel()
    n_cells = len(sc.geos) * (len(sc.products) if sc.products else 1)
    assert panel.coords.has_geo
    assert panel.coords.n_geos == len(sc.geos)

    cfg = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_PYMC,
        n_draws=10,
        n_tune=10,
        n_chains=1,
        use_parametric_adstock=True,
    )
    mmm = BayesianMMM(panel, cfg, TrendConfig(type=TrendType.LINEAR))
    assert mmm.n_cells == n_cells
    assert mmm.n_obs == len(sc.y)
    # Balanced panel: every cell observed every period.
    counts = np.bincount(mmm.cell_idx, minlength=n_cells)
    assert (counts == len(sc.weeks)).all()
    assert mmm.model is not None  # graph builds


def test_geo_slice_matches_panel():
    sc = dgp_geo.build("geo_clean")
    g = sc.geos[0]
    sl = sc.geo_scenario(g)
    mask = sc.spend.index.get_level_values("Geography") == g
    np.testing.assert_allclose(
        sl.spend.to_numpy(), sc.spend[mask].to_numpy(), rtol=1e-12
    )
    np.testing.assert_allclose(
        sl.true_contribution.to_numpy(),
        sc.true_contribution_by_geo.loc[g].to_numpy(),
        rtol=1e-12,
    )
    # The slice builds a national-style panel.
    p = sl.panel()
    assert not p.coords.has_geo
    assert p.n_obs == len(sc.weeks)


def test_national_aggregate_sums_panel():
    sc = dgp_geo.build("geo_clean")
    nat = sc.national_scenario()
    assert len(nat.y) == len(sc.weeks)
    np.testing.assert_allclose(
        nat.spend.sum().to_numpy(),
        sc.spend.sum().to_numpy(),
        rtol=1e-12,
    )
    np.testing.assert_allclose(float(nat.y.sum()), float(sc.y.sum()), rtol=1e-12)
    np.testing.assert_allclose(
        nat.true_contribution.to_numpy(), sc.true_contribution.to_numpy(), rtol=1e-12
    )
    assert not nat.representable


def test_heterogeneous_world_has_real_geo_spread():
    sc = dgp_geo.build("geo_heterogeneous")
    roas = sc.true_roas_by_geo
    # Effectiveness multipliers create at least 2x ROI spread within a channel.
    spread = roas.max(axis=0) / roas.min(axis=0)
    assert (spread > 2.0).any(), spread
    assert not sc.representable
    # Performance chasing: TV budget share is largest where TV works best.
    tv_spend = sc.spend["TV"].groupby(level="Geography").sum()
    assert tv_spend.idxmax() == "North" and tv_spend.idxmin() == "West"
