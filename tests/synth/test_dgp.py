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


def test_adstock_misspec_has_long_carryover():
    sc = dgp.build("adstock_misspec")
    assert sc.notes["true_l_max"] > 8  # beyond the model's default window


def test_time_varying_beta_has_a_break():
    sc = dgp.build("time_varying_beta")
    assert 0 < sc.notes["break_week"] < len(sc.y)


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
