"""Phase 1 experiment-method framework: registry + geo estimators (synthetic
control, TBR, GBR, DiD-MMT). Estimators are validated by recovering a KNOWN
planted lift on a synthetic geo panel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.planning import methods, simulation
from mmm_framework.planning.methods.gbr import gbr_estimator, gbr_iroas
from mmm_framework.planning.methods.did_mmt import did_mmt_estimator
from mmm_framework.planning.methods.synthetic_control import (
    synthetic_control_analysis,
    synthetic_control_estimator,
    synthetic_control_weights,
)
from mmm_framework.planning.methods.tbr import tbr_counterfactual, tbr_estimator
from mmm_framework.planning.simulation import Assignment, SimPanel, Window

# ── synthetic planted-lift geo panel ─────────────────────────────────────────


def make_geo_panel(
    *, seed=0, n_geo=20, n_wk=40, n_treat=2, pre=30, lift_per_week=0.0, level=100.0
):
    """A geo × week KPI panel with a shared common factor + independent geo
    levels, and a known lift added to the first ``n_treat`` geos over the test
    window. Returns (panel, assignment, window, true_total)."""
    rng = np.random.default_rng(seed)
    kpi = (
        rng.normal(level, 8, size=(n_wk, n_geo)) + rng.normal(0, 10, size=n_wk)[:, None]
    )
    geos = [f"g{i}" for i in range(n_geo)]
    kw = pd.DataFrame(kpi, columns=geos)
    treat = tuple(geos[:n_treat])
    ctrl = tuple(geos[n_treat:])
    pairs = tuple((geos[i], geos[i + n_treat]) for i in range(n_treat))
    kw.loc[kw.index[pre:n_wk], list(treat)] += lift_per_week
    panel = SimPanel(
        kpi_wide=kw,
        spend_wide=None,
        kpi_national=kw.sum(axis=1),
        spend_national=kw.sum(axis=1) * 0.0,
        residuals=None,
        periods=list(range(n_wk)),
        geos=geos,
    )
    asg = Assignment(
        kind="geo", treatment_geos=treat, control_geos=ctrl, pairs=pairs, seed=seed
    )
    win = Window(
        pre_slice=slice(0, pre),
        test_slice=slice(pre, n_wk),
        t_pre=pre,
        t_test=n_wk - pre,
    )
    true_total = lift_per_week * (n_wk - pre) * n_treat
    return panel, asg, win, true_total


# ── registry ─────────────────────────────────────────────────────────────────


def test_registry_lists_all_geo_methods():
    keys = {m.key for m in methods.list_methods()}
    assert {"synthetic_control", "regadj_geo", "tbr", "gbr", "did_mmt"} <= keys
    for k in ("synthetic_control", "tbr", "gbr", "did_mmt"):
        assert methods.has_method(k)
        assert methods.get_method(k).requirement.family == "geo"


def test_methods_for_data_gates_on_geos():
    # too few geos → geo methods unsupported
    rows = {r["key"]: r for r in methods.methods_for_data(n_geos=2, n_weeks=52)}
    assert rows["synthetic_control"]["supported"] is False
    assert "geos" in rows["synthetic_control"]["reason"]
    # enough geos + weeks → supported
    rows = {r["key"]: r for r in methods.methods_for_data(n_geos=20, n_weeks=52)}
    assert rows["synthetic_control"]["supported"] is True
    assert rows["gbr"]["supported"] is True


def test_new_estimators_wired_into_leaderboard():
    for k in ("synthetic_control", "tbr", "gbr"):
        assert k in simulation._GEO_ESTIMATORS


# ── recovery of a planted lift ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "estimator", [tbr_estimator, gbr_estimator, synthetic_control_estimator]
)
def test_estimator_recovers_planted_total(estimator):
    panel, asg, win, true_total = make_geo_panel(seed=1, lift_per_week=50.0)
    est = estimator(panel, asg, win).estimate
    # within 15% of the true 1000 (n_treat=2 x 10 weeks x 50)
    assert true_total == pytest.approx(1000.0)
    assert est == pytest.approx(true_total, rel=0.15)


def test_did_mmt_recovers_per_pair_average():
    # DiD-MMT (per-pair) recovers total / n_pairs — its known estimand scale.
    panel, asg, win, true_total = make_geo_panel(seed=1, lift_per_week=50.0, n_treat=2)
    est = did_mmt_estimator(panel, asg, win).estimate
    assert est == pytest.approx(true_total / len(asg.pairs), rel=0.2)


def test_estimators_null_is_near_zero():
    panel, asg, win, _ = make_geo_panel(seed=3, lift_per_week=0.0)
    for estimator in (tbr_estimator, gbr_estimator, synthetic_control_estimator):
        est = estimator(panel, asg, win).estimate
        # null effect is small relative to the treated level (~200 for 2 geos)
        assert abs(est) < 200.0


# ── synthetic control specifics ──────────────────────────────────────────────


def test_scm_weights_are_convex():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=(30, 6))
    y = x @ np.array([0.3, 0.2, 0.1, 0.2, 0.1, 0.1]) + rng.normal(0, 0.05, 30)
    w = synthetic_control_weights(x, y)
    assert w.min() >= -1e-9
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_scm_placebo_pvalue_small_on_real_effect():
    panel, asg, win, _ = make_geo_panel(seed=1, lift_per_week=50.0)
    res = synthetic_control_analysis(panel, asg, win)
    assert res["placebo_p_value"] <= 0.1
    assert res["standardized_effect"] > 5.0


def test_scm_placebo_pvalue_uniform_on_null():
    ps = [
        synthetic_control_analysis(*make_geo_panel(seed=s, lift_per_week=0.0)[:3])[
            "placebo_p_value"
        ]
        for s in range(12)
    ]
    # under the null the standardized placebo p-value should not systematically
    # flag significance — its mean sits well away from 0.
    assert np.mean(ps) > 0.2


def test_scm_handles_level_mismatch():
    # treated aggregate has a much higher level than any single donor; demeaned
    # SCM must not read the level gap as an effect.
    panel, asg, win, _ = make_geo_panel(
        seed=5, n_treat=3, lift_per_week=0.0, level=100.0
    )
    est = synthetic_control_estimator(panel, asg, win).estimate
    assert abs(est) < 300.0  # << the ~300 treated level


# ── TBR counterfactual ───────────────────────────────────────────────────────


def test_tbr_counterfactual_tracks_pre_period():
    rng = np.random.default_rng(0)
    x = rng.normal(100, 10, size=50)
    y = 2.0 + 0.8 * x + rng.normal(0, 1, size=50)
    cf = tbr_counterfactual(x[:40], y[:40], x[40:])
    pred = cf["pred_mean"]
    assert len(pred) == 10
    # predicted counterfactual close to the (untreated) truth on held-out weeks
    assert np.mean(np.abs(pred - y[40:])) < 3.0
    assert cf["cum_sd"] > 0.0


def test_gbr_iroas_divides_by_spend():
    assert gbr_iroas(1000.0, 500.0) == pytest.approx(2.0)
    assert gbr_iroas(1000.0, 0.0) is None


@pytest.mark.slow
def test_tbr_bsts_headline_recovers_effect():
    """The full BSTS counterfactual (headline read-out) recovers a planted
    cumulative effect with a credible band that covers it."""
    from mmm_framework.planning.methods.tbr import tbr_causal_impact

    rng = np.random.default_rng(0)
    n_pre, t_test = 40, 10
    x = rng.normal(100, 10, size=n_pre + t_test)
    y = 5 + 0.9 * x + rng.normal(0, 2, size=n_pre + t_test)
    y_obs_test = y[n_pre:] + 30.0  # +30/week planted effect
    res = tbr_causal_impact(
        x[:n_pre], y[:n_pre], x[n_pre:], y_test=y_obs_test, method="advi", draws=400
    )
    assert res["cumulative_effect"] == pytest.approx(300.0, rel=0.2)
    assert res["cumulative_lower"] <= 300.0 <= res["cumulative_upper"]
    assert res["prob_positive"] > 0.9
