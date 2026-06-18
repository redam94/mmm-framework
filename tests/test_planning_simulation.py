"""A/A and A/B simulation tests (planning/simulation.py).

The headline test pins the SIM-1 correction: on an autocorrelated null series
the analytic decision rule's false-positive rate inflates well past alpha, and
the design-calibrated critical value restores nominal size. These build a
SimPanel directly from synthetic frames — no model fit needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.planning import simulation as S


def _panel_from_wide(kpi_wide: pd.DataFrame) -> S.SimPanel:
    spend = pd.DataFrame(
        np.abs(np.random.default_rng(0).normal(100, 10, kpi_wide.shape)),
        index=kpi_wide.index,
        columns=kpi_wide.columns,
    )
    return S.SimPanel(
        kpi_wide=kpi_wide,
        spend_wide=spend,
        kpi_national=kpi_wide.sum(axis=1),
        spend_national=spend.sum(axis=1),
        residuals=None,
        periods=list(kpi_wide.index),
        geos=list(kpi_wide.columns),
    )


def _white_noise_panel(n=240, g=8, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        rng.normal(1000, 50, (n, g)) + np.arange(g) * 30.0,
        columns=[f"G{i}" for i in range(g)],
    )
    return _panel_from_wide(df)


def _ar1_panel(n=240, g=8, phi=0.85, seed=2):
    rng = np.random.default_rng(seed)
    cols = {}
    for i in range(g):
        e = rng.normal(0, 50, n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + e[t]
        cols[f"G{i}"] = 1000 + i * 30.0 + x
    return _panel_from_wide(pd.DataFrame(cols))


def test_norm_ppf_and_wilson():
    assert S._norm_ppf(0.975) == pytest.approx(1.95996, abs=1e-3)
    assert S._norm_ppf(0.8) == pytest.approx(0.84162, abs=1e-3)
    lo, hi = S._wilson_ci(5, 100)
    assert 0 < lo < 0.05 < hi < 0.2


def test_aa_white_noise_is_calibrated():
    panel = _white_noise_panel()
    assignment = S.build_geo_assignment(panel, seed=7)
    aa = S.run_aa_simulation(
        panel, S.pooled_did_estimator, assignment, duration=8, max_windows=200, seed=7
    )
    assert aa.n_windows >= 30
    # the calibrated critical value gives ~alpha by construction
    assert aa.fpr_at_crit == pytest.approx(0.05, abs=0.04)
    # on iid noise the analytic rule is also roughly valid (not badly inflated)
    assert aa.fpr_at_nominal < 0.20


def test_aa_autocorrelation_inflates_nominal_fpr():
    """The headline SIM-1 result: AR(1) noise inflates the analytic-SE FPR well
    past the calibrated rate, and the harness exposes it."""
    panel = _ar1_panel(phi=0.85)
    assignment = S.build_geo_assignment(panel, seed=7)
    aa = S.run_aa_simulation(
        panel, S.pooled_did_estimator, assignment, duration=8, max_windows=200, seed=7
    )
    # calibrated rate stays near alpha...
    assert aa.fpr_at_crit == pytest.approx(0.05, abs=0.05)
    # ...but the nominal (analytic-SE) rate is materially inflated and flagged
    assert aa.fpr_at_nominal > aa.fpr_at_crit
    assert aa.fpr_at_nominal > 0.10
    assert aa.null_method == "analytic_se"


def test_probit_mde_recovers_known_threshold():
    # power(effect) = Phi(slope*(effect - mde0)); recover mde0 at power 0.8
    mde0, slope = 1.0, 3.0
    effects = np.linspace(0.2, 2.5, 9)
    powers = 0.5 * (
        1 + np.vectorize(__import__("math").erf)(slope * (effects - mde0) / np.sqrt(2))
    )
    mde, method = S._probit_mde(effects, powers, target=0.8)
    assert method == "probit_fit"
    # 0.8 power is at effect = mde0 + z_.80/slope = 1.0 + 0.8416/3 ≈ 1.28
    assert mde == pytest.approx(1.0 + 0.8416 / slope, abs=0.1)


def test_probit_mde_isotonic_handles_nonmonotone():
    effects = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    powers = np.array([0.1, 0.55, 0.5, 0.85, 0.95])  # one dip
    mde, method = S._probit_mde(np.abs(effects), powers, target=0.8)
    assert mde is not None and mde > 0


def test_aa_insufficient_windows_flagged():
    panel = _white_noise_panel(n=24, g=6)
    assignment = S.build_geo_assignment(panel, seed=3)
    aa = S.run_aa_simulation(
        panel, S.pooled_did_estimator, assignment, duration=8, max_windows=50, seed=3
    )
    assert aa.status == "insufficient_windows"


def test_national_assignment_and_onoff_runs():
    panel = _white_noise_panel(n=120, g=1)
    # single-geo national series
    nat = pd.Series(panel.kpi_wide.iloc[:, 0].to_numpy(), name="Sales")
    p = S.SimPanel(None, None, nat, nat * 0 + 100, None, list(range(120)), [])
    assignment = S.build_national_assignment(duration=8, seed=5)
    aa = S.run_aa_simulation(
        p, S.national_onoff_estimator, assignment, duration=8, max_windows=80, seed=5
    )
    assert isinstance(aa, S.AAResult)


@pytest.fixture(scope="module")
def geo_csv(tmp_path_factory):
    from mmm_framework.synth import generate_mff

    df, key = generate_mff("geo_heterogeneous", seed=3, n_weeks=130)
    path = tmp_path_factory.mktemp("sim") / "geo.csv"
    df.to_csv(path, index=False)
    return str(path), key


def test_leaderboard_structure_and_ranking(geo_csv):
    path, key = geo_csv
    lb = S.methodology_leaderboard(
        path,
        "Sales",
        key["channels"][0],
        duration=8,
        spend_delta_window=5000.0,
        target_mde_roas=2.0,
        max_aa_windows=120,
        max_ab_windows=50,
        seed=7,
    )
    assert lb["kind"] == "geo"
    assert len(lb["methodologies"]) == 3
    keys = {m["key"] for m in lb["methodologies"]}
    assert keys == {"pooled_did", "per_pair_did", "regadj_geo"}
    # valid methods sort ahead of invalid ones
    valids = [m["valid"] for m in lb["methodologies"]]
    assert valids == sorted(valids, reverse=True)
    for m in lb["methodologies"]:
        assert m["fpr"] is None or 0.0 <= m["fpr"] <= 1.0
        assert 0.0 <= m["power_at_expected_effect"] <= 1.0


def test_national_onoff_recovers_injected_total_not_2x():
    """Review finding 3: the on/off estimate must equal the injected total, not
    t_test/t_hi (~2x) of it."""
    n = 60
    base = pd.Series(np.full(n, 1000.0), name="Sales")
    panel = S.SimPanel(None, None, base, base * 0 + 100, None, list(range(n)), [])
    assignment = S.build_national_assignment(duration=8, seed=5)
    injector = S.LiftInjector(kind="model_anchored", expected_total=400.0)
    window = S.Window(slice(0, 40), slice(40, 48), 40, 8)
    _t, kpi = injector.inject_national(panel, assignment, window, scale=1.0)
    inj = S.SimPanel(None, None, kpi, base * 0 + 100, None, panel.periods, panel.geos)
    r = S.national_onoff_estimator(inj, assignment, window)
    assert r.estimate == pytest.approx(400.0, rel=0.05)  # not ~800


def test_pooled_did_estimand_and_scale():
    """Review finding 2: pooled DiD recovers total/n_pairs, and _estimand_scale
    reports exactly that so bias/coverage are scored on the right scale."""
    n, g = 60, 8
    df = pd.DataFrame(np.full((n, g), 1000.0), columns=[f"G{i}" for i in range(g)])
    panel = _panel_from_wide(df)
    assignment = S.build_geo_assignment(panel, seed=7)
    injector = S.LiftInjector(kind="model_anchored", expected_total=800.0)
    window = S.Window(slice(0, 40), slice(40, 48), 40, 8)
    _t, kpi_wide = injector.inject_geo(panel, assignment, window, 1.0)
    inj = S.SimPanel(
        kpi_wide,
        panel.spend_wide,
        panel.kpi_national,
        panel.spend_national,
        None,
        panel.periods,
        panel.geos,
    )
    r = S.pooled_did_estimator(inj, assignment, window)
    n_pairs = len(assignment.pairs)
    assert r.estimate == pytest.approx(800.0 / n_pairs, rel=0.05)
    assert S._estimand_scale("pooled_did", assignment) == pytest.approx(1.0 / n_pairs)
    assert S._estimand_scale("regadj_geo", assignment) == 1.0


def test_probit_mde_at_floor_for_saturated_curve():
    """Review finding 5: a fully-powered curve returns a finite MDE (at_floor),
    not None (which would mislabel the strongest estimator as not powered)."""
    effects = np.array([100.0, 200.0, 300.0, 400.0])
    powers = np.array([1.0, 1.0, 1.0, 1.0])
    mde, method = S._probit_mde(effects, powers, target=0.8)
    assert method == "at_floor"
    assert mde == pytest.approx(100.0)


def test_fixed_injector_used_without_model(geo_csv):
    path, key = geo_csv
    lb = S.methodology_leaderboard(
        path,
        "Sales",
        key["channels"][0],
        duration=8,
        spend_delta_window=5000.0,
        expected_effect_total=None,
        max_aa_windows=80,
        max_ab_windows=40,
        seed=1,
    )
    assert lb["injection_basis"] in ("spend_share", "uniform")
    assert lb["expected_effect"] > 0
