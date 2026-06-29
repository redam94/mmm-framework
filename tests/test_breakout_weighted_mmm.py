"""Breakout-Weighted MMM — the regularized, uncertainty-quantified replacement
for a PSO breakout-weight optimizer.

The in-house PSO picks per-breakout weights that minimize in-sample MSE with no
regularization and no uncertainty; it overfits to noise and reports a confident
point even when the mix is unidentifiable. ``BreakoutWeightedMMM`` makes the
weights partial-pooled random effects that shrink toward equal-weighting, with
the between-breakout spread ``τ`` estimated — so it RECOVERS real breakout signal
when it exists (``breakout_heterogeneous``), COLLAPSES to equal-weighting when it
doesn't (``breakout_homogeneous``), and reports WIDE posteriors when the mix is
unidentifiable (``breakout_collinear``). Validated with real (slow) NUTS fits.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from mmm_framework.config import ModelConfig
from mmm_framework.model import TrendConfig
from mmm_framework.model.trend_config import TrendType

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../examples/garden_models")
    ),
)
from breakout_weighted_mmm import (  # noqa: E402
    BreakoutWeightedMMM,
    BreakoutWeightedParams,
    breakout_aggregated_panel,
    breakout_dataset,
)

# NUTS settings: 4 chains + enough tuning so the non-centered weight hierarchy
# mixes (MAP is unstable for variance-component τ models — see the model
# docstring). Assert ordering / correlation / interval coverage, not magnitudes.
_FIT = dict(draws=500, tune=800, chains=4, target_accept=0.9, random_seed=11)


def _model(scenario: str, *, groups: dict | None = None, sigma: float = 0.3):
    dataset, sc, gr = breakout_dataset(scenario)
    return (
        BreakoutWeightedMMM(
            dataset,
            ModelConfig(use_parametric_adstock=True),
            TrendConfig(type=TrendType.LINEAR),
            model_params={
                "breakout_groups": groups if groups is not None else gr,
                "breakout_weight_sigma": sigma,
            },
        ),
        sc,
    )


def _weights(mmm, channel: str = "TV") -> np.ndarray:
    """Posterior-mean breakout weights, ordered as the model's sub-streams."""
    subs = mmm._breakout_names[channel]
    return np.array(
        [
            float(mmm._trace.posterior[f"breakout_weight_{channel}_{s}"].mean())
            for s in subs
        ]
    )


# ---------------------------------------------------------------------------
# fast: graph shape + the exact nested null (no sampling)
# ---------------------------------------------------------------------------


def test_breakout_channel_exposes_weight_block():
    """A breakout channel gets the non-centered weight hierarchy + a length-K
    weight vector; plain channels keep a scalar ``beta_<ch>``."""
    mmm, _ = _model("breakout_heterogeneous")
    assert mmm.channel_names == ["Search", "Social", "Display", "TV"]
    m = mmm.model  # builds lazily
    names = set(m.named_vars)
    for v in (
        "breakout_logtau_TV",
        "breakout_z_TV",
        "breakout_weights_TV",
        "breakout_share_TV",
        "breakout_weight_TV_TV_Premium",
        "breakout_weight_TV_TV_Standard",
        "breakout_weight_TV_TV_Remnant",
    ):
        assert v in names, v
    assert tuple(m["breakout_weights_TV"].shape.eval()) == (3,)
    # plain channels stay scalar
    for ch in ("Search", "Social", "Display"):
        assert f"breakout_logtau_{ch}" not in names
        assert tuple(m[f"beta_{ch}"].shape.eval()) == ()
    # the channel axis is the VIRTUAL (grouped) one
    assert list(m.coords["channel"]) == ["Search", "Social", "Display", "TV"]


def test_empty_groups_is_a_plain_mmm():
    """With no breakout groups the graph carries no ``breakout_*`` variables — the
    media likelihood is the plain base build over the raw columns."""
    mmm, _ = _model("breakout_heterogeneous", groups={})
    m = mmm.model
    assert [n for n in m.named_vars if "breakout" in n] == []
    # every raw column is its own plain channel
    assert mmm.channel_names == [
        "TV_Premium",
        "TV_Standard",
        "TV_Remnant",
        "Search",
        "Social",
        "Display",
    ]


def test_weights_collapse_to_one_at_z_zero_and_preserve_totals():
    """The exact nested null: at ``z = 0`` every weight is 1 (so the weighted
    aggregate equals the unweighted aggregate), and the sum-preserving constraint
    ``Σ_k w_k S_k = Σ_k S_k`` holds at any ``z``."""
    import pytensor

    mmm, _ = _model("breakout_heterogeneous")
    m = mmm.model
    fn = pytensor.function(
        [m["breakout_z_TV"], m["breakout_logtau_TV"]],
        m["breakout_weights_TV"],
        on_unused_input="ignore",
    )
    np.testing.assert_allclose(fn(np.zeros(3), 0.5), np.ones(3), atol=1e-9)

    S = mmm._breakout_totals["TV"]
    w = fn(np.array([1.0, -1.0, 0.5]), 0.4)
    assert not np.allclose(w, 1.0)  # a non-trivial mix
    np.testing.assert_allclose(float(w @ S), float(S.sum()), rtol=1e-9)


def test_invalid_breakout_groups_raise():
    """Unknown sub-streams, overlapping groups, and parent-name collisions fail
    fast at construction; a 1-member group is demoted with a warning."""
    dataset, sc, _ = breakout_dataset("breakout_heterogeneous")

    def build(groups):
        return BreakoutWeightedMMM(
            dataset,
            ModelConfig(use_parametric_adstock=True),
            TrendConfig(type=TrendType.LINEAR),
            model_params={"breakout_groups": groups},
        )

    with pytest.raises(ValueError, match="not a media column"):
        build({"TV": ["TV_Premium", "Nope"]})
    with pytest.raises(ValueError, match="more than one breakout group"):
        build(
            {"TV": ["TV_Premium", "TV_Standard"], "TV2": ["TV_Standard", "TV_Remnant"]}
        )
    with pytest.raises(ValueError, match="collides with an existing media column"):
        build({"Search": ["TV_Premium", "TV_Standard"]})
    with pytest.warns(UserWarning, match="< 2 members"):
        mmm = build({"TV": ["TV_Premium"]})
    assert "TV" not in mmm._breakout_names  # demoted to plain


def test_config_schema_defaults():
    p = BreakoutWeightedParams()
    assert p.breakout_groups == {}
    assert p.breakout_weight_sigma == 0.3


# ---------------------------------------------------------------------------
# slow: recovery / collapse / honesty against the planted truth (real NUTS)
#
# One shared fixture fits all three sibling worlds once, so the assertions can be
# both absolute (per scenario) and COMPARATIVE (the spread τ and the posterior
# widths move the right way across scenarios) without extra fits. Absolute τ
# cutoffs are deliberately avoided — the HalfNormal(σ_w) prior already permits
# some spread, so the honest signals are the weight intervals (do they exclude /
# cover equal-weighting?) and the cross-scenario comparisons.
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBreakoutRecovery:
    @pytest.fixture(scope="class")
    def fits(self):
        out = {}
        for scen in (
            "breakout_heterogeneous",
            "breakout_homogeneous",
            "breakout_collinear",
        ):
            mmm, sc = _model(scen)
            mmm.fit(progressbar=False, **_FIT)
            out[scen] = (mmm, sc, mmm.breakout_weights_summary().set_index("breakout"))
        return out

    def test_recovers_heterogeneous_weights(self, fits):
        """Breakouts that genuinely differ AND flight independently: the partial-
        pooled weights track the planted (share-mean-1) truth, and the strongest
        breakout is detected ABOVE equal-weighting (its interval excludes 1)."""
        from scipy.stats import pearsonr

        mmm, sc, df = fits["breakout_heterogeneous"]
        subs = mmm._breakout_names["TV"]
        truth = np.array([sc.notes["true_weights"][s] for s in subs])
        fitted = _weights(mmm)

        r = pearsonr(fitted, truth)[0]
        assert (
            r > 0.5
        ), f"weights should track planted truth; corr={r:.3f} {dict(zip(subs, fitted))}"
        assert subs[int(np.argmax(fitted))] == "TV_Premium", dict(zip(subs, fitted))
        assert not bool(df.loc["TV_Premium", "covers_equal"]), df
        assert df.loc["TV_Premium", "hdi_low"] > 1.0, df

    def test_collapses_on_homogeneous(self, fits):
        """Equally-effective breakouts (true weights all 1): the honest model
        collapses to equal-weighting — every weight's interval covers 1 and the
        point weights sit near 1 — where an unregularized optimizer invents
        spurious unequal weights that still lower in-sample MSE."""
        mmm, _, df = fits["breakout_homogeneous"]
        assert bool(df["covers_equal"].all()), df
        fitted = _weights(mmm)
        assert np.max(np.abs(fitted - 1.0)) < 0.5, dict(
            zip(mmm._breakout_names["TV"], fitted)
        )

    def test_spread_tracks_real_heterogeneity(self, fits):
        """τ is the falsification statistic: the recovered between-breakout spread
        is larger when the breakouts genuinely differ AND are identifiable than
        when they are truly equal."""
        het_tau = float(fits["breakout_heterogeneous"][2]["logtau_mean"].iloc[0])
        homo_tau = float(fits["breakout_homogeneous"][2]["logtau_mean"].iloc[0])
        assert (
            het_tau > homo_tau
        ), f"heterogeneous τ={het_tau:.3f} should exceed homogeneous τ={homo_tau:.3f}"

    def test_is_honest_when_unidentifiable(self, fits):
        """Near-collinear sub-streams: the mix is unidentifiable, so the honest
        model reports WIDER weight posteriors than the identifiable fit and does
        NOT confidently separate every breakout from equal-weighting — the
        opposite of a point optimizer's false confidence."""
        het_df = fits["breakout_heterogeneous"][2]
        col_df = fits["breakout_collinear"][2]
        het_width = float((het_df["hdi_high"] - het_df["hdi_low"]).mean())
        col_width = float((col_df["hdi_high"] - col_df["hdi_low"]).mean())
        assert col_width > 1.3 * het_width, (
            f"collinear weight posteriors should be much wider: "
            f"collinear={col_width:.3f} vs heterogeneous={het_width:.3f}"
        )
        assert bool(col_df["covers_equal"].any()), col_df


@pytest.mark.slow
def test_serialization_round_trip():
    """A fitted breakout model serializes and reloads (against the original sub-
    stream panel), and the reloaded weights match — even though the model's
    channel axis (virtual) differs from the panel's columns (sub-streams)."""
    import tempfile

    from mmm_framework.serialization import MMMSerializer

    mmm, _ = _model("breakout_heterogeneous")
    mmm.fit(progressbar=False, draws=200, tune=400, chains=2, random_seed=7)
    before = mmm.breakout_weights_summary().set_index("breakout")["weight_mean"]

    with tempfile.TemporaryDirectory() as d:
        MMMSerializer.save(mmm, d)
        loaded = MMMSerializer.load(d, mmm.panel)

    assert type(loaded).__name__ == "BreakoutWeightedMMM"
    assert loaded.channel_names == mmm.channel_names
    assert loaded._breakout_names == mmm._breakout_names
    after = loaded.breakout_weights_summary().set_index("breakout")["weight_mean"]
    for s in before.index:
        assert abs(float(before[s]) - float(after[s])) < 1e-6, s
