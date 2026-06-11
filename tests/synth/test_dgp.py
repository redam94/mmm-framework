"""Fast (no-MCMC) checks that each scenario actually violates its assumption.

These guard the *generators* -- that the data really breaks what it claims to,
that ground truth is internally consistent, and that representability flags are
right -- so the slow stress matrix (``run_stress_matrix.py``) is interpretable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on path

from tests.synth import dgp  # noqa: E402


@pytest.mark.parametrize("name", dgp.PRIORITY)
def test_scenario_builds_and_is_consistent(name):
    sc = dgp.build(name)
    n = len(sc.y)
    assert n == len(sc.spend) == len(sc.controls)
    assert list(sc.spend.columns) == dgp.CHANNELS or sc.name == "aurora_kitchen_sink"
    assert not sc.y.isna().any()
    assert not sc.spend.isna().any().any()
    assert (sc.spend.to_numpy() >= 0).all()
    # ground truth present for every channel
    assert set(sc.true_contribution.index) == set(sc.channels)
    assert set(sc.true_roas.index) == set(sc.channels)
    assert sc.true_contribution.notna().all()
    # the panel actually constructs
    panel = sc.panel()
    assert panel.y.shape[0] == n


def test_clean_is_the_positive_control():
    sc = dgp.build("clean")
    assert sc.violates == ""  # nothing broken
    assert sc.representable
    # exogenous: spend is near-uncorrelated across channels
    corr = np.corrcoef(sc.spend.to_numpy().T)
    off = corr[np.triu_indices(len(sc.channels), 1)]
    assert np.abs(off).mean() < 0.4
    # all true contributions positive and material
    assert (sc.true_contribution > 0).all()


def test_multicollinearity_is_actually_collinear():
    sc = dgp.build("multicollinearity")
    assert sc.notes["mean_pairwise_corr"] > 0.85


def test_confounding_links_spend_to_latent_demand():
    sc = dgp.build("unobserved_confounding")
    demand = sc.notes["latent_demand"]
    corr = {c: np.corrcoef(sc.spend[c].to_numpy(), demand)[0, 1] for c in sc.channels}
    non_chasers = [c for c in sc.channels if c not in sc.notes["chasers"]]
    worst_non_chaser = max(corr[c] for c in non_chasers)
    # the demand-chasing channels correlate with the (hidden) confounder, and
    # do so markedly more than the non-chasing channels
    for c in sc.notes["chasers"]:
        assert corr[c] > 0.3
        assert corr[c] > worst_non_chaser + 0.1
    # demand-blind: the proxy is NOT among the controls
    assert "CategoryDemand" not in sc.controls.columns


def test_confounding_controlled_adds_the_proxy_without_changing_truth():
    blind = dgp.build("unobserved_confounding")
    fixed = dgp.build("confounding_controlled")
    assert "CategoryDemand" in fixed.controls.columns
    # same generative world -> identical causal ground truth
    np.testing.assert_allclose(
        blind.true_contribution.values, fixed.true_contribution.values
    )


def test_negative_effect_is_unrepresentable():
    sc = dgp.build("negative_effect")
    assert not sc.representable
    assert sc.true_contribution["Display"] < 0  # positive-only prior can't reach this


def test_synergy_double_counts_across_channels():
    sc = dgp.build("synergy")
    # with an interaction, the primed channels' counterfactual contributions
    # exceed their clean-world values (the synergy is credited to both)
    assert sc.true_contribution["TV"] > dgp.build("clean").true_contribution["TV"]
    assert (
        sc.true_contribution["Search"] > dgp.build("clean").true_contribution["Search"]
    )


def test_spend_outliers_inflate_the_normalization_max():
    sc = dgp.build("spend_outliers")
    true_sum = sc.notes["true_spend_sum"]
    for c in sc.channels:
        # the observed series carries a spike far above the rest of the channel
        col = sc.spend[c]
        assert col.max() > 10 * np.median(col)
        # but true ROAS was computed on the un-spiked spend
        assert sc.true_roas[c] == pytest.approx(
            sc.true_contribution[c] / true_sum[c], rel=1e-6
        )


def test_mixed_data_errors_structure_and_truth():
    """The realistic defect mix: each error is really in the OBSERVED spend,
    truth is computed on the uncorrupted spend, and the within-scenario
    false-positive control (Display) is genuinely untouched."""
    sc = dgp.build("mixed_data_errors")
    errors = sc.notes["errors"]
    n = len(sc.y)

    # x10 decimal shift on TV: the observed week becomes the series max and
    # sits well above every legitimate week (it was injected on a non-dark
    # week, so x10 lands at ~4-10x the true max — NOT a cartoonish 15x)
    w = errors["TV"]["week"]
    true_sum = sc.notes["true_spend_sum"]
    rest_max = sc.spend["TV"].drop(index=w).max()
    assert sc.spend["TV"].iloc[w] == sc.spend["TV"].max()
    assert sc.spend["TV"].iloc[w] > 3 * rest_max
    # x2 double-count on Social: present but NOT an extreme of the series
    w = errors["Social"]["week"]
    assert sc.spend["Social"].iloc[w] < 3 * sc.spend["Social"].median() * 2

    # missed load: observed Search is zero that week, but the TRUE spend
    # (which generated sales) was non-dark
    w = errors["Search"]["week"]
    assert sc.spend["Search"].iloc[w] == 0.0
    assert sc.notes["true_search_spend"].iloc[w] > 0
    # Search is always-on apart from the missed load (no flighting troughs)
    others = sc.spend["Search"].drop(sc.spend.index[w])
    assert float(others.min()) > 0.3 * float(others.median())

    # error positions may fall near the edges (that's the point) but in-range
    for e in errors.values():
        assert 0 <= e["week"] < n

    # promo shocks are real demand: KPI lifted at those weeks vs neighbors
    for w in sc.notes["promo_weeks"]:
        lo, hi = max(0, w - 4), min(n, w + 5)
        neighbors = [t for t in range(lo, hi) if t != w]
        assert sc.y.iloc[w] > sc.y.iloc[neighbors].mean() + 60

    # truth/ROAS computed on uncorrupted spend
    for c in sc.channels:
        assert sc.true_roas[c] == pytest.approx(
            sc.true_contribution[c] / true_sum[c], rel=1e-6
        )


def test_adstock_misspec_has_long_carryover():
    sc = dgp.build("adstock_misspec")
    assert sc.notes["true_l_max"] > 8  # beyond the model's default window


def test_time_varying_beta_has_a_break():
    sc = dgp.build("time_varying_beta")
    assert 0 < sc.notes["break_week"] < len(sc.y)


def test_trend_break_shock_and_media_ramp_coincide():
    sc = dgp.build("trend_break")
    brk = sc.notes["break_week"]
    assert 0 < brk < len(sc.y)
    # the KPI level drops at the break (compare adjacent windows)
    pre = sc.y.iloc[brk - 10 : brk].mean()
    post = sc.y.iloc[brk : brk + 10].mean()
    assert post < pre - 50
    # TV/Display spend ramps right at the break -> confounded with the recovery
    for c in ("TV", "Display"):
        s = sc.spend[c].to_numpy()
        assert s[brk:].mean() > 1.3 * s[:brk].mean()
    # non-ramped channels stay level
    s = sc.spend["Search"].to_numpy()
    assert abs(s[brk:].mean() / s[:brk].mean() - 1.0) < 0.3


def test_seasonality_misspec_spikes_align_with_social_q4():
    sc = dgp.build("seasonality_misspec")
    hol = sc.notes["holiday_indicator"]
    q4 = sc.notes["q4_indicator"]
    assert hol.sum() == 12  # 4 weeks/year * 3 years
    assert (hol * q4 == hol).all()  # every holiday week is inside Social's Q4 push
    # Social spends materially more in Q4 than elsewhere
    s = sc.spend["Social"].to_numpy()
    assert s[q4 == 1].mean() > 2 * s[q4 == 0].mean()
    # holiday weeks carry visible extra KPI vs neighboring non-holiday weeks
    y = sc.y.to_numpy()
    assert y[hol == 1].mean() > y[(q4 == 1) & (hol == 0)].mean() + 50


def test_dense_controls_structure_and_decoys():
    sc = dgp.build("dense_controls")
    assert len(sc.controls.columns) == 25  # 1 confounder + 2 precision + 18 + 4
    # decoys really track their channel's spend but carry no causal effect
    for c in dgp.CHANNELS:
        rho = np.corrcoef(sc.controls[f"decoy_{c.lower()}"], sc.spend[c].to_numpy())[
            0, 1
        ]
        assert rho > 0.8
    # the confounder is marked so selection machinery can exempt it
    assert sc.control_roles["demand_proxy"] == "confounder"
    # TV/Search chase the latent demand the proxy measures
    d = sc.notes["latent_demand"]
    corr = {c: float(np.corrcoef(sc.spend[c], d)[0, 1]) for c in sc.channels}
    assert corr["TV"] > corr["Social"] and corr["Search"] > corr["Display"]


def test_realistic_many_factor_structure():
    sc = dgp.build("realistic")
    assert len(sc.channels) == 7
    assert (
        len(sc.controls.columns) == 13
    )  # 2 confounders + 4 precision + 6 noise + 1 mediator
    roles = sc.notes["roles"]
    assert sum(v == "confounder" for v in roles.values()) == 2
    assert sum(v == "precision" for v in roles.values()) == 4
    assert sum(v == "irrelevant" for v in roles.values()) == 6
    assert sum(v == "mediator" for v in roles.values()) == 1
    assert sc.notes["mediator"] in sc.controls.columns
    # the two weak channels are near-collinear (bought together) -> their split
    # is unidentifiable observationally, even though each carries a real effect.
    weak = sc.notes["weak_channels"]
    assert len(weak) == 2
    rho = float(np.corrcoef(sc.spend[weak[0]], sc.spend[weak[1]])[0, 1])
    assert rho > 0.9
    # the demand-chasing channels correlate with the hidden confounder more than the rest
    d = sc.notes["latent_demand"]
    corr = {c: float(np.corrcoef(sc.spend[c], d)[0, 1]) for c in sc.channels}
    assert corr["Search"] > corr["Social"] and corr["TV"] > corr["Social"]
    assert sc.representable and (sc.true_contribution > 0).all()


def test_truth_matches_manual_counterfactual_for_clean():
    # Independent re-derivation of the counterfactual estimand for one channel.
    sc = dgp.build("clean")
    from tests.synth.dgp import _ALPHA, _AMP, _LAM, _geom_adstock, _logistic_sat

    spend = sc.spend.to_numpy(float)
    maxes = {c: float(sc.spend[c].max()) for c in sc.channels}
    # reconstruct baseline deterministically from the same seed path is fiddly;
    # instead verify the additive identity: zeroing a channel removes exactly its
    # own contribution, so truth[c] == sum(amp_c * sat_c).
    for i, c in enumerate(sc.channels):
        xn = spend[:, i] / maxes[c]
        contrib = _AMP[c] * _logistic_sat(_geom_adstock(xn, _ALPHA[c]), _LAM[c])
        assert sc.true_contribution[c] == pytest.approx(contrib.sum(), rel=1e-6)
