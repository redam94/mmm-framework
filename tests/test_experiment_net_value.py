"""Phase 3 net experiment economics: E[reallocation gain] − E[test loss] as one
decision figure with a distribution (planning/experiment_value.py)."""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.planning.experiment_value import (
    ExperimentNetValue,
    compute_experiment_net_value,
    _decay_weights,
)
from test_planning_opportunity_cost import CHANNELS, GEOS, FakeMMM, _design


def _loss_draws(mean=-1000.0, sd=200.0, n=400, seed=0):
    return np.random.default_rng(seed).normal(mean, sd, n)


# ── gain side ────────────────────────────────────────────────────────────────


def test_gain_is_margin_times_decayed_evoi():
    nv = compute_experiment_net_value(
        channel="TV",
        evoi_kpi_units=1000.0,
        margin_per_kpi=0.5,
        response_horizon_weeks=26,
        half_life_weeks=None,  # no decay
    )
    assert nv.unit == "$"
    assert nv.decay_factor == pytest.approx(1.0)
    assert nv.reallocation_gain == pytest.approx(500.0)


def test_gain_capped_at_evpi():
    nv = compute_experiment_net_value(
        channel="TV",
        evoi_kpi_units=5000.0,
        evpi_kpi_units=1000.0,
        margin_per_kpi=1.0,
        half_life_weeks=None,
    )
    assert nv.reallocation_gain == pytest.approx(1000.0)
    assert nv.assumptions["evpi_capped"] is True


def test_decay_reduces_gain_monotonically():
    kw = dict(channel="TV", evoi_kpi_units=1000.0, margin_per_kpi=1.0)
    no_decay = compute_experiment_net_value(**kw, half_life_weeks=None)
    slow = compute_experiment_net_value(**kw, half_life_weeks=52.0)
    fast = compute_experiment_net_value(**kw, half_life_weeks=4.0)
    assert no_decay.reallocation_gain > slow.reallocation_gain > fast.reallocation_gain
    assert 0.0 < fast.decay_factor < slow.decay_factor < 1.0


def test_discounting_reduces_gain():
    kw = dict(
        channel="TV", evoi_kpi_units=1000.0, margin_per_kpi=1.0, half_life_weeks=None
    )
    flat = compute_experiment_net_value(**kw)
    disc = compute_experiment_net_value(**kw, discount_rate_annual=0.20)
    assert disc.reallocation_gain < flat.reallocation_gain


def test_decay_weights_shape():
    w = _decay_weights(10, 5.0, 0.0)
    assert len(w) == 10
    assert w[0] == pytest.approx(1.0)
    assert w[5] == pytest.approx(0.5)  # one half-life


# ── loss + net + distribution ────────────────────────────────────────────────


def test_net_positive_cheap_high_learning_test():
    # small loss (mean −200 KPI), big learning
    nv = compute_experiment_net_value(
        channel="TV",
        evoi_kpi_units=5000.0,
        kpi_delta_draws=_loss_draws(mean=-200.0, sd=50.0),
        spend_delta=0.0,
        margin_per_kpi=1.0,
        half_life_weeks=None,
    )
    assert nv.net_value > 0
    assert nv.prob_net_positive > 0.95
    assert nv.net_value_p5 < nv.net_value < nv.net_value_p95
    assert nv.breakeven_horizon_weeks is not None


def test_net_negative_expensive_low_learning_test():
    nv = compute_experiment_net_value(
        channel="TV",
        evoi_kpi_units=50.0,
        kpi_delta_draws=_loss_draws(mean=-5000.0, sd=200.0),
        margin_per_kpi=1.0,
        half_life_weeks=None,
    )
    assert nv.net_value < 0
    assert nv.prob_net_positive < 0.05


def test_money_saving_holdout_breakeven_zero():
    # holdout: forgone KPI margin (−1000 × $0.1 = −$100) < saved spend ($5000)
    nv = compute_experiment_net_value(
        channel="TV",
        evoi_kpi_units=1000.0,
        kpi_delta_draws=_loss_draws(mean=-1000.0, sd=100.0),
        spend_delta=-5000.0,  # signed: holdout SAVES spend
        margin_per_kpi=0.1,
        half_life_weeks=None,
    )
    assert nv.net_profit_during_test > 0  # the test itself makes money
    assert nv.test_loss == pytest.approx(0.0, abs=1.0)
    assert nv.breakeven_horizon_weeks == 0.0
    assert nv.net_value > nv.reallocation_gain  # gain PLUS the test's own profit


def test_never_breakeven_warns():
    nv = compute_experiment_net_value(
        channel="TV",
        evoi_kpi_units=1.0,  # negligible learning
        kpi_delta_draws=_loss_draws(mean=-50_000.0, sd=100.0),
        margin_per_kpi=1.0,
        half_life_weeks=4.0,  # decays fast
        response_horizon_weeks=26,
    )
    assert nv.breakeven_horizon_weeks is None
    assert any("never repays" in w for w in nv.warnings)


# ── degradation ──────────────────────────────────────────────────────────────


def test_no_evoi_is_insufficient():
    nv = compute_experiment_net_value(
        channel="TV",
        evoi_kpi_units=None,
        kpi_delta_draws=_loss_draws(),
        margin_per_kpi=1.0,
    )
    assert nv.basis == "insufficient"
    assert nv.reallocation_gain is None
    assert nv.net_value is None
    assert nv.test_loss is not None  # loss side still reported


def test_no_margin_kpi_units_basis():
    nv = compute_experiment_net_value(
        channel="TV",
        evoi_kpi_units=1000.0,
        kpi_delta_draws=_loss_draws(mean=-200.0),
        spend_delta=5000.0,  # must be EXCLUDED (units don't mix)
        margin_per_kpi=None,
        half_life_weeks=None,
    )
    assert nv.unit == "KPI units"
    assert nv.reallocation_gain == pytest.approx(1000.0)
    # spend excluded: loss ≈ 200 KPI, not 5200
    assert nv.test_loss < 500.0
    assert any("spend change is excluded" in w for w in nv.warnings)


def test_model_anchored_basis_flag():
    a = compute_experiment_net_value(
        channel="TV", evoi_kpi_units=100.0, margin_per_kpi=1.0, model_anchored=True
    )
    b = compute_experiment_net_value(
        channel="TV", evoi_kpi_units=100.0, margin_per_kpi=1.0, model_anchored=False
    )
    assert a.basis == "model_anchored"
    assert b.basis == "evoi_bounded"


def test_to_dict_json_safe():
    nv = compute_experiment_net_value(
        channel="TV", evoi_kpi_units=None, margin_per_kpi=None
    )
    d = nv.to_dict()
    assert d["basis"] == "insufficient"
    assert all(not (isinstance(v, float) and not np.isfinite(v)) for v in d.values())


# ── integration with compute_opportunity_cost (return_draws) ─────────────────


def test_oc_return_draws_and_net_value_pipeline():
    from mmm_framework.planning.opportunity_cost import compute_opportunity_cost

    mmm = FakeMMM(GEOS, 40, CHANNELS, seed=1)
    design = _design(GEOS, design="holdout", duration=8)
    oc = compute_opportunity_cost(
        mmm, design, margin_per_kpi=0.5, max_draws=30, return_draws=True
    )
    assert oc.draws is not None and "kpi_delta" in oc.draws
    assert "draws" not in oc.to_dict()  # never serialized

    nv = compute_experiment_net_value(
        channel="TV",
        evoi_kpi_units=500.0,
        opportunity_cost_result=oc,
        half_life_weeks=26.0,
    )
    assert isinstance(nv, ExperimentNetValue)
    assert nv.unit == "$"  # margin pulled off the OC result
    assert nv.margin_per_kpi == pytest.approx(0.5)
    assert nv.prob_net_positive is not None  # draws flowed through
    assert nv.net_value is not None
    # consistency: net = gain + E[signed profit during test]
    assert nv.net_value == pytest.approx(
        nv.reallocation_gain + nv.net_profit_during_test, rel=1e-9
    )


def test_economics_op_emits_net_value_block(tmp_path):
    """The experiment_economics op attaches a JSON-safe net_value block when a
    model is present (even when the EVOI loopback degrades → 'insufficient')."""
    import json

    import pandas as pd

    from mmm_framework.agents import model_ops as mo
    from mmm_framework.synth import generate_mff

    df, key = generate_mff("geo_heterogeneous", seed=3, n_weeks=80)
    path = tmp_path / "geo.csv"
    df.to_csv(path, index=False)
    geos = sorted(pd.unique(df["Geography"].dropna()))
    n_periods = int(df["Period"].nunique())
    mmm = FakeMMM(geos, n_periods, key["channels"], seed=2)

    res = mo.experiment_economics(
        mmm,
        None,
        design_params={
            "dataset_path": str(path),
            "kpi": "Sales",
            "channel": key["channels"][0],
            "design_key": "geo_lift",
            "duration": 8,
            "design": "holdout",
            "seed": 7,
        },
        run_simulation=False,
        margin=0.5,
    )
    assert res["error"] is None
    eco = res["dashboard"]["experiment_economics"]
    assert eco.get("net_value") is not None, eco.get("net_value_error")
    nv = eco["net_value"]
    # the loss side must be present (draws flowed through the OC result)
    assert nv["test_loss"] is not None
    assert nv["unit"] == "$"
    json.dumps(res["dashboard"], default=str)  # JSON boundary safe


def test_oc_without_draws_summary_fallback():
    from mmm_framework.planning.opportunity_cost import compute_opportunity_cost

    mmm = FakeMMM(GEOS, 40, CHANNELS, seed=1)
    design = _design(GEOS, design="holdout", duration=8)
    oc = compute_opportunity_cost(mmm, design, margin_per_kpi=0.5, max_draws=30)
    assert oc.draws is None
    nv = compute_experiment_net_value(
        channel="TV", evoi_kpi_units=500.0, opportunity_cost_result=oc
    )
    assert nv.prob_net_positive is None
    assert nv.net_value is not None
    assert any("no draws" in w for w in nv.warnings)
