"""Simulation-Based Calibration tests.

Fast tests use a closed-form Normal–Normal conjugate model (no PyMC) where the
posterior is analytic and SBC is provably uniform — so they validate the rank
math, the simultaneous bands, and (critically) the *direction* of the shape
classifier (∪ = overconfident, ∩ = overdispersed; Talts 2018). A slow test runs
the real MMM refit loop with a tiny n_sims as a wiring smoke.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.diagnostics.sbc import (
    compute_param_stat,
    ecdf_diff_band,
    miscalibration_score,
    rank_hist_band,
    run_sbc,
    uniformity_chisq,
)


# ── conjugate Normal–Normal posterior (analytic) ─────────────────────────────
def _conjugate_draw_and_fit(*, tau=2.0, sigma=1.0, n=20, L=200, distortion=None):
    """Return a draw_and_fit(rng) for SBC. ``distortion`` warps the 'posterior':
    None=calibrated, ('scale', f)=multiply sd, ('shift', k)=shift by k·sd."""

    def f(rng):
        theta = float(rng.normal(0.0, tau))
        y = rng.normal(theta, sigma, n)
        s2 = 1.0 / (1.0 / tau**2 + n / sigma**2)
        mu = s2 * (y.sum() / sigma**2)
        sd = np.sqrt(s2)
        if distortion is None:
            draws = rng.normal(mu, sd, L)
        elif distortion[0] == "scale":
            draws = rng.normal(mu, sd * distortion[1], L)
        elif distortion[0] == "shift":
            draws = rng.normal(mu + distortion[1] * sd, sd, L)
        else:
            raise ValueError(distortion)
        return {"theta": theta}, {"theta": draws}

    return f


class TestRankMathAndUniformity:
    def test_calibrated_is_uniform(self):
        res = run_sbc(_conjugate_draw_and_fit(L=200), n_sims=400, L=200, seed=1)
        p = res.params[0]
        assert p.chi2_pvalue > 0.05
        assert p.shape == "uniform"
        assert p.calibrated
        assert p.miscalibration < 0.15

    def test_false_positive_rate_near_alpha(self):
        # the chi2 test should reject a truly-uniform run ~alpha of the time
        rejects = 0
        trials = 30
        for s in range(trials):
            res = run_sbc(_conjugate_draw_and_fit(L=150), n_sims=200, L=150, seed=s)
            if res.params[0].chi2_pvalue <= 0.05:
                rejects += 1
        assert rejects <= 6  # ~alpha=0.05 with slack for 30 trials

    def test_uniformity_chisq_exact_bins(self):
        # exactly-uniform ranks over {0..L} → tiny chi2, large p
        L = 99
        ranks = np.tile(np.arange(L + 1), 4)  # perfectly flat
        chi2, p, counts = uniformity_chisq(ranks, L, n_bins=20)
        assert p > 0.99
        assert counts.sum() == ranks.size


class TestShapeClassifierDirection:
    """Pins the Talts 2018 direction — the agent's fix advice depends on it."""

    def test_overconfident_is_smile(self):
        # posterior too NARROW (÷3 sd) → ranks pile at the edges → ∪
        res = run_sbc(
            _conjugate_draw_and_fit(distortion=("scale", 1 / 3.0)),
            n_sims=300,
            L=200,
            seed=2,
        )
        p = res.params[0]
        assert p.shape == "smile(∪)"
        assert not p.calibrated
        assert p.dispersion_z > 2

    def test_overdispersed_is_frown(self):
        # posterior too WIDE (×3 sd) → ranks bunch in the centre → ∩
        res = run_sbc(
            _conjugate_draw_and_fit(distortion=("scale", 3.0)),
            n_sims=300,
            L=200,
            seed=3,
        )
        p = res.params[0]
        assert p.shape == "frown(∩)"
        assert not p.calibrated
        assert p.dispersion_z < -2

    def test_biased_high_is_left_skew(self):
        # draws shifted UP → posterior overestimates → few draws ≤ θ* → low ranks
        res = run_sbc(
            _conjugate_draw_and_fit(distortion=("shift", 1.2)),
            n_sims=300,
            L=200,
            seed=4,
        )
        p = res.params[0]
        assert p.shape == "left-skew"
        assert p.mean_norm_rank < 0.5

    def test_biased_low_is_right_skew(self):
        res = run_sbc(
            _conjugate_draw_and_fit(distortion=("shift", -1.2)),
            n_sims=300,
            L=200,
            seed=5,
        )
        p = res.params[0]
        assert p.shape == "right-skew"
        assert p.mean_norm_rank > 0.5


class TestMiscalibrationScore:
    def test_monotone_bounds(self):
        L, B = 99, 20
        # perfectly uniform → ~0
        flat = np.tile(np.arange(L + 1), 5)
        _, _, counts = uniformity_chisq(flat, L, B)
        assert miscalibration_score(counts, L, B) < 0.02
        # all mass in one bin → ≈ 1 - 1/B
        spike = np.zeros(500, dtype=int)
        _, _, counts2 = uniformity_chisq(spike, L, B)
        assert miscalibration_score(counts2, L, B) > 0.9 * (1 - 1 / B)


class TestBands:
    def test_rank_hist_band_coverage(self):
        # the simultaneous band should contain ALL bins for ~95% of uniform runs
        N, L, B = 200, 150, 20
        lower, upper = rank_hist_band(N, L, B, prob=0.95)
        rng = np.random.default_rng(0)
        inside = 0
        trials = 300
        for _ in range(trials):
            ranks = rng.integers(0, L + 1, size=N)
            _, _, counts = uniformity_chisq(ranks, L, B)
            if np.all((counts >= lower) & (counts <= upper)):
                inside += 1
        cov = inside / trials
        assert 0.90 <= cov <= 0.995  # ~nominal simultaneous coverage

    def test_ecdf_diff_band_shape(self):
        z, lo, hi = ecdf_diff_band(200, prob=0.95, n_points=60)
        assert z.shape == lo.shape == hi.shape
        assert np.all(hi >= lo)
        # band pinches toward the endpoints (binomial variance → 0 at z=0,1)
        assert (hi - lo)[0] < (hi - lo)[len(z) // 2]


class TestResultPlumbing:
    def test_compute_param_stat_dashboard_json_safe(self):
        import json

        ranks = np.random.default_rng(0).integers(0, 101, size=120)
        stat = compute_param_stat("beta", ranks, L=100, n_bins=20)
        d = stat.to_dashboard()
        json.dumps(d)  # must not raise (numpy scalars cast)
        assert set(d) >= {
            "name",
            "shape",
            "chi2_pvalue",
            "miscalibration",
            "calibrated",
        }

    def test_result_summary_and_worst(self):
        res = run_sbc(_conjugate_draw_and_fit(), n_sims=80, L=120, seed=7)
        assert "SBC" in res.summary()
        assert res.worst() is not None
        import json

        json.dumps(res.to_dashboard())


class TestInterpretationFallback:
    def test_deterministic_interpretation_grounded(self):
        from mmm_framework.agents.sbc_insights import interpret_sbc

        # an overconfident param → the fix text must mention "narrow"/overconfident
        res = run_sbc(
            _conjugate_draw_and_fit(distortion=("scale", 1 / 3.0)),
            n_sims=300,
            L=200,
            seed=9,
        )
        txt = interpret_sbc(res.to_dashboard(), llm=None)  # offline fallback
        low = txt.lower()
        assert "too narrow" in low or "overconfident" in low  # correct cause
        assert "edges" in low  # geometric read of the ∪ shape
        assert "widen" in low  # actionable fix
        assert res.params[0].name in txt  # grounded in the actual param


@pytest.mark.slow
def test_run_mmm_sbc_smoke():
    """The real MMM refit loop runs end-to-end at tiny n_sims (wiring/shape only —
    N=4 is far too small to judge calibration)."""
    import numpy as np
    import pandas as pd

    from mmm_framework.config import (
        DimensionType,
        InferenceMethod,
        KPIConfig,
        MFFConfig,
        MediaChannelConfig,
        ModelConfig,
    )
    from mmm_framework.data_loader import PanelCoordinates, PanelDataset
    from mmm_framework.diagnostics.sbc import run_mmm_sbc
    from mmm_framework.model import BayesianMMM, TrendConfig, TrendType

    periods = pd.date_range("2022-01-03", periods=30, freq="W-MON")
    n = len(periods)
    rng = np.random.default_rng(0)
    tv = np.abs(rng.normal(100, 25, n))
    se = np.abs(rng.normal(50, 15, n))
    y = pd.Series(800 + 1.3 * tv + 2.0 * se + rng.normal(0, 20, n), name="Sales")
    coords = PanelCoordinates(
        periods=periods,
        geographies=None,
        products=None,
        channels=["TV", "Search"],
        controls=None,
    )
    cfg = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
            MediaChannelConfig(name="Search", dimensions=[DimensionType.PERIOD]),
        ],
        controls=[],
    )
    panel = PanelDataset(
        y=y,
        X_media=pd.DataFrame({"TV": tv, "Search": se}),
        X_controls=None,
        coords=coords,
        index=periods,
        config=cfg,
    )
    model = BayesianMMM(
        panel,
        ModelConfig(inference_method=InferenceMethod.BAYESIAN_PYMC),
        TrendConfig(type=TrendType.LINEAR),
    )

    res = run_mmm_sbc(
        model, n_sims=4, L=40, sampler="numpyro", tune=80, chains=2, seed=1
    )
    names = {p.name for p in res.params}
    assert {"beta_TV", "beta_Search", "sigma"} <= names
    for p in res.params:
        assert len(p.int_ranks) <= res.n_sims_effective
        assert np.all((p.int_ranks >= 0) & (p.int_ranks <= 40))
    import json

    json.dumps(res.to_dashboard())  # msgpack/JSON-safe across the kernel boundary

    # op wiring: returns a well-formed model-op result with valid figures
    from mmm_framework.agents.model_ops import simulation_based_calibration as op

    out = op(model, n_sims=4, L=40, sampler="numpyro")
    assert out["error"] is None
    assert out["plots"] and "data" in out["plots"][0]["figure"]
    assert out["tables"] and out["assumption"]["category"] == "prior"
