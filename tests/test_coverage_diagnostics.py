"""Interval-coverage diagnostics tests.

Fast tests drive the pure statistics with analytic Normal posteriors where the
truth is known exactly: a calibrated pipeline must show nominal coverage, an
overconfident one (posterior sd shrunk) must under-cover with the
"intervals too narrow" flag, and a biased one must raise the bias flag —
pinning both the numbers and the *direction* of the diagnosis. A slow test runs
the real simulate→refit MMM loop as a wiring smoke.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from mmm_framework.diagnostics.coverage import (
    DEFAULT_LEVELS,
    build_recovery_result,
    coverage_from_draws,
    coverage_from_ranks,
    failure_mode_guide,
    jeffreys_interval,
    recovery_diagnosis,
)


# ── analytic "simulate → refit" pipelines (no PyMC) ──────────────────────────
def _normal_pipeline(
    *,
    n_sims: int = 400,
    L: int = 400,
    sd_factor: float = 1.0,
    shift: float = 0.0,
    truth: float = 0.0,
    s: float = 1.0,
    seed: int = 0,
):
    """Posterior draws for repeated refits at a fixed truth.

    Calibrated case: posterior mean ~ N(truth, s), posterior sd = s, so
    z = (mean − truth)/s ~ N(0,1) and central intervals have exact nominal
    coverage. ``sd_factor`` warps the reported width; ``shift`` biases the
    location (in units of s).
    """
    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(n_sims):
        m = rng.normal(truth + shift * s, s)
        sims.append(rng.normal(m, s * sd_factor, L))
    return sims


class TestPureCoverageStats:
    def test_jeffreys_interval_bounds(self):
        lo, hi = jeffreys_interval(0, 20)
        assert lo == 0.0 and 0 < hi < 0.3
        lo, hi = jeffreys_interval(20, 20)
        assert hi == 1.0 and 0.7 < lo < 1.0
        lo, hi = jeffreys_interval(9, 10)
        assert lo < 0.9 < hi
        assert jeffreys_interval(0, 0) == (0.0, 1.0)

    def test_calibrated_normal_has_nominal_coverage(self):
        sims = _normal_pipeline(n_sims=600, L=1000)
        stats = coverage_from_draws(0.0, sims, DEFAULT_LEVELS)
        for st in stats:
            assert st.n == 600
            # calibrated pipeline: within MC noise of nominal at every level
            assert abs(st.coverage - st.level) < 0.05
        cov90 = next(st for st in stats if abs(st.level - 0.9) < 1e-9)
        assert 0.85 < cov90.coverage < 0.95
        assert cov90.verdict == "ok"

    def test_overconfident_underscovers_and_flags_width(self):
        sims = _normal_pipeline(sd_factor=0.4)
        stats = coverage_from_draws(0.0, sims)
        cov90 = next(st for st in stats if abs(st.level - 0.9) < 1e-9)
        assert cov90.verdict == "under"
        assert cov90.coverage < 0.75
        means = np.array([np.mean(d) for d in sims])
        sds = np.array([np.std(d, ddof=1) for d in sims])
        diag = recovery_diagnosis(0.0, means, sds)
        assert "overconfident (intervals too narrow)" in diag["flags"]
        assert abs(diag["bias_z"]) < 4.0  # no spurious bias claim

    def test_biased_flags_direction(self):
        sims = _normal_pipeline(shift=3.0, n_sims=200)
        means = np.array([np.mean(d) for d in sims])
        sds = np.array([np.std(d, ddof=1) for d in sims])
        diag = recovery_diagnosis(0.0, means, sds)
        assert "biased high" in diag["flags"]
        low = recovery_diagnosis(0.0, -means, sds)  # mirrored → biased low
        assert "biased low" in low["flags"]
        cov90 = next(
            st for st in coverage_from_draws(0.0, sims) if abs(st.level - 0.9) < 1e-9
        )
        assert cov90.verdict == "under"

    def test_conservative_overcovers(self):
        sims = _normal_pipeline(sd_factor=2.5, n_sims=300)
        cov50 = next(
            st for st in coverage_from_draws(0.0, sims) if abs(st.level - 0.5) < 1e-9
        )
        assert cov50.verdict == "over"
        means = np.array([np.mean(d) for d in sims])
        sds = np.array([np.std(d, ddof=1) for d in sims])
        diag = recovery_diagnosis(0.0, means, sds)
        assert "conservative (intervals too wide)" in diag["flags"]

    def test_coverage_from_ranks_uniform_vs_edges(self):
        L = 100
        rng = np.random.default_rng(3)
        uniform = rng.integers(0, L + 1, size=800)
        for st in coverage_from_ranks(uniform, L):
            assert st.verdict != "under"
        # all ranks at the extremes = truth always in the tails → under-coverage
        edges = np.concatenate([np.zeros(400), np.full(400, L)]).astype(int)
        cov90 = next(
            st for st in coverage_from_ranks(edges, L) if abs(st.level - 0.9) < 1e-9
        )
        assert cov90.coverage == 0.0
        assert cov90.verdict == "under"


class TestRecoveryResultAssembly:
    def _result(self, **kw):
        sims_ok = _normal_pipeline(n_sims=60, L=200, seed=1)
        sims_bad = _normal_pipeline(n_sims=60, L=200, sd_factor=0.3, seed=2)
        defaults = dict(
            kinds={"beta_TV": "parameter", "contribution_TV": "contribution"},
            truth_source="posterior_mean",
            sampler="numpyro",
            L=200,
            seed=0,
            n_sims_requested=60,
        )
        defaults.update(kw)
        return build_recovery_result(
            {"beta_TV": 0.0, "contribution_TV": 0.0},
            {"beta_TV": sims_ok, "contribution_TV": sims_bad},
            **defaults,
        )

    def test_dashboard_json_safe_and_worst(self):
        res = self._result()
        payload = res.to_dashboard()
        json.dumps(payload)  # msgpack/JSON-safe across the kernel boundary
        assert res.worst().name == "contribution_TV"
        assert not res.all_nominal
        assert res.n_sims_effective == 60
        target = next(t for t in res.targets if t.name == "contribution_TV")
        assert target.kind == "contribution"
        assert target.coverage_at(0.9).verdict == "under"
        assert "overconfident (intervals too narrow)" in target.flags

    def test_summary_headline_and_caveats(self):
        res = self._result()
        txt = res.summary()
        assert "UNDER-COVERAGE DETECTED" in txt
        assert "contribution_TV" in txt
        small = build_recovery_result(
            {"x": 0.0},
            {"x": _normal_pipeline(n_sims=8, L=100)},
            n_sims_requested=8,
        )
        assert any("Monte-Carlo" in c for c in small.caveats)
        advi = build_recovery_result(
            {"x": 0.0},
            {"x": _normal_pipeline(n_sims=40, L=100)},
            sampler="advi",
            n_sims_requested=40,
        )
        assert any("ADVI" in c for c in advi.caveats)

    def test_failure_mode_guide_mentions_key_causes(self):
        guide = failure_mode_guide().lower()
        for phrase in ("approximate", "misspecification", "confounding", "prior"):
            assert phrase in guide


class TestSBCCoverageIntegration:
    def test_sbc_param_stat_carries_coverage(self):
        from mmm_framework.diagnostics.sbc import build_sbc_result, compute_param_stat

        L = 100
        rng = np.random.default_rng(7)
        stat = compute_param_stat("beta_TV", rng.integers(0, L + 1, 500), L)
        assert stat.coverage, "SBC stats should include coverage levels"
        cov90 = stat.coverage_at(0.9)
        assert cov90 is not None and 0.8 < cov90.coverage < 1.0
        res = build_sbc_result(
            {"beta_TV": rng.integers(0, L + 1, 200)}, L=L, n_sims_requested=200
        )
        json.dumps(res.to_dashboard())
        assert "coverage" in res.to_dashboard()["params"][0]
        assert "90% interval covers" in res.summary()


class TestCoverageCharts:
    def test_charts_render_from_dashboard_dicts(self):
        res = TestRecoveryResultAssembly()._result()
        targets = [t.to_dashboard() for t in res.targets]
        from mmm_framework.validation.charts.diagnostics import (
            create_coverage_calibration_curve,
            create_recovery_coverage_chart,
        )

        fig = create_recovery_coverage_chart(targets, level=0.9)
        assert fig.data  # bar + error bars present
        curve = create_coverage_calibration_curve(targets[0])
        assert len(curve.data) == 3  # diagonal + band + empirical


def _tiny_model():
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
    return BayesianMMM(
        panel,
        ModelConfig(inference_method=InferenceMethod.BAYESIAN_PYMC),
        TrendConfig(type=TrendType.LINEAR),
    )


@pytest.mark.slow
def test_run_recovery_coverage_smoke():
    """The real simulate→refit loop runs end-to-end at tiny n_sims (wiring/shape
    only — N=3 is far too small to judge coverage)."""
    from mmm_framework.diagnostics.coverage import run_recovery_coverage

    model = _tiny_model()
    res = run_recovery_coverage(
        model, truth="prior", n_sims=3, L=40, tune=80, chains=2, sampler="numpyro"
    )
    names = {t.name for t in res.targets}
    assert {"beta_TV", "beta_Search", "sigma"} <= names
    assert {"contribution_TV", "contribution_Search"} <= names
    kinds = {t.name: t.kind for t in res.targets}
    assert kinds["contribution_TV"] == "contribution"
    for t in res.targets:
        assert t.n_sims <= 3
        for st in t.levels:
            assert 0.0 <= st.coverage <= 1.0
    json.dumps(res.to_dashboard())

    # op wiring: well-formed model-op result with valid figures + assumption
    from mmm_framework.agents.model_ops import recovery_coverage_check as op

    out = op(model, truth="prior", n_sims=3, L=40, sampler="numpyro")
    assert out["error"] is None
    assert out["plots"] and "data" in out["plots"][0]["figure"]
    assert out["tables"]
    assert out["assumption"]["category"] == "other"
    assert "coverage" in out["dashboard"]
