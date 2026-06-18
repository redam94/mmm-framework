"""Tests pinning the rescaled default priors (Weibull adstock scale, trend).

Fast tests pin the configuration values; ``@pytest.mark.slow`` tests re-run the
measured synth-harness failures that motivated the rescaling.

These defaults were re-derived empirically from the synth stress harness
(``tests/synth/dgp.py``):

* ``AdstockConfig.weibull`` default scale prior scales with ``l_max``:
  ``Gamma(2, 2 / m)`` with mean ``m = max(2, (l_max - 9) / 2)``. The old fixed
  ``Gamma(2, 1)`` (mean 2 weeks) produced a divergence storm (71 divergences,
  r-hat 1.07) on a 26-week window with a slow true kernel; a *proportional*
  rule (mean ``l_max / 3``) was measured to trap chains in an aliased
  delayed-kernel mode (r-hat ~1.8) on mid-size windows with periodic
  flighting, hence the offset/floored form (legacy prior up to ``l_max=13``).
* ``TrendConfig.growth_prior_sigma`` (0.1 -> 0.5) and
  ``TrendConfig.changepoint_prior_scale`` (0.05 -> 0.5): trends enter on
  standardized ``y`` over ``t in [0, 1]``, so the old widths pinned the trend
  near zero and pushed real trends/breaks into media and intercept.

Any future change to these defaults should be intentional: update the asserted
values here together with fresh harness measurements.
"""

from __future__ import annotations

import pytest

from mmm_framework.config import AdstockConfig, AdstockType, PriorConfig
from mmm_framework.model import TrendConfig, TrendType


class TestWeibullScalePriorScaling:
    """The default Weibull scale prior must adapt to the lag window."""

    @pytest.mark.parametrize("l_max", [4, 8, 12, 26, 52])
    def test_default_scale_prior_mean_follows_documented_rule(self, l_max: int) -> None:
        cfg = AdstockConfig.weibull(l_max=l_max)

        assert cfg.type == AdstockType.WEIBULL
        assert cfg.scale_prior is not None
        params = cfg.scale_prior.params
        # Documented rule: Gamma(2.0, 2.0 / m), m = max(2, (l_max - 9) / 2).
        expected_mean = max(2.0, (l_max - 9.0) / 2.0)
        assert params["alpha"] == 2.0
        assert params["beta"] == pytest.approx(2.0 / expected_mean)
        # Gamma mean alpha/beta equals the rule's mean.
        assert params["alpha"] / params["beta"] == pytest.approx(expected_mean)

    def test_default_scale_prior_scales_between_windows(self) -> None:
        def mean(cfg: AdstockConfig) -> float:
            p = cfg.scale_prior.params
            return p["alpha"] / p["beta"]

        # Short/mid windows keep the legacy mean-2 prior...
        assert mean(AdstockConfig.weibull(l_max=8)) == pytest.approx(2.0)
        assert mean(AdstockConfig.weibull(l_max=12)) == pytest.approx(2.0)
        # ...long windows get a proportionally slower prior.
        assert mean(AdstockConfig.weibull(l_max=26)) == pytest.approx(8.5)
        assert mean(AdstockConfig.weibull(l_max=52)) == pytest.approx(21.5)

    def test_default_shape_prior_is_window_independent(self) -> None:
        # Shape is dimensionless; it must not vary with l_max.
        p8 = AdstockConfig.weibull(l_max=8).shape_prior.params
        p26 = AdstockConfig.weibull(l_max=26).shape_prior.params
        assert p8 == p26 == {"alpha": 2.0, "beta": 1.0}

    def test_explicit_scale_prior_passthrough_unchanged(self) -> None:
        explicit = PriorConfig.gamma(alpha=4.0, beta=0.5)
        cfg = AdstockConfig.weibull(l_max=26, scale_prior=explicit)
        assert cfg.scale_prior is explicit
        assert cfg.scale_prior.params == {"alpha": 4.0, "beta": 0.5}

    def test_explicit_shape_prior_passthrough_unchanged(self) -> None:
        explicit = PriorConfig.gamma(alpha=3.0, beta=1.5)
        cfg = AdstockConfig.weibull(l_max=26, shape_prior=explicit)
        assert cfg.shape_prior is explicit


class TestTrendConfigDefaults:
    """Pin the loosened trend prior defaults (intentional-drift guard)."""

    def test_growth_prior_sigma_default(self) -> None:
        assert TrendConfig().growth_prior_sigma == 0.5

    def test_changepoint_prior_scale_default(self) -> None:
        assert TrendConfig().changepoint_prior_scale == 0.5

    def test_other_trend_defaults_unchanged(self) -> None:
        cfg = TrendConfig()
        assert cfg.type == TrendType.LINEAR
        assert cfg.n_changepoints == 10
        assert cfg.changepoint_range == 0.8
        assert cfg.growth_prior_mu == 0.0

    def test_roundtrip_preserves_new_defaults(self) -> None:
        restored = TrendConfig.from_dict(TrendConfig().to_dict())
        assert restored.growth_prior_sigma == 0.5
        assert restored.changepoint_prior_scale == 0.5


# =============================================================================
# Fit-based regression tests on the synth stress harness (slow)
# =============================================================================

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root


def _fit(panel, trend_cfg, *, draws=400, tune=400, target_accept=0.95):
    from mmm_framework.config import InferenceMethod, ModelConfig
    from mmm_framework.model import BayesianMMM

    cfg = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_NUMPYRO,
        n_draws=draws,
        n_tune=tune,
        n_chains=2,
        target_accept=target_accept,
        use_parametric_adstock=True,
        optim_seed=0,
    )
    mmm = BayesianMMM(panel, cfg, trend_cfg)
    res = mmm.fit(random_seed=0)
    return mmm, res


def _med_abs_rel_err(mmm, sc) -> float:
    import numpy as np

    contrib = mmm.compute_counterfactual_contributions(
        compute_uncertainty=False, random_seed=0
    )
    est = contrib.total_contributions
    errs = [
        abs(float(est[c]) - float(sc.true_contribution[c]))
        / abs(float(sc.true_contribution[c]))
        for c in sc.channels
    ]
    return float(np.median(errs))


@pytest.mark.slow
class TestRescaledDefaultsFitBased:
    """Re-run the measured failures that motivated the new defaults."""

    def test_weibull_l26_default_prior_samples_cleanly(self) -> None:
        """Old fixed Gamma(2,1) scale prior: 71 divergences / r-hat 1.07 here."""
        from tests.synth import dgp

        sc = dgp.build("adstock_misspec")
        panel = sc.panel()
        for mc in panel.config.media_channels:
            mc.adstock = AdstockConfig.weibull(l_max=26)

        mmm, res = _fit(panel, _linear_trend(), draws=500, tune=500)

        # The Weibull kernel must actually be in the graph for this test to
        # mean anything (otherwise clean sampling is trivial).
        posterior_vars = set(res.trace.posterior.data_vars)
        assert any(v.startswith("adstock_scale_") for v in posterior_vars)
        assert any(v.startswith("adstock_shape_") for v in posterior_vars)

        assert int(res.diagnostics["divergences"]) <= 5
        assert float(res.diagnostics["rhat_max"]) < 1.05
        # Recovery on par with a hand-informed prior under this same protocol
        # (measured: informed 51.5-52.8%, new default 54.8-55.1%, with ~+/-10pt
        # seed-to-seed noise on this weakly identified 26-lag kernel).
        assert _med_abs_rel_err(mmm, sc) < 0.65

    def test_linear_trend_default_recovers_trend_amplitude(self) -> None:
        """Old growth_prior_sigma=0.1 recovered only ~26 of ~60 KPI units ptp."""
        import numpy as np

        from tests.synth import dgp

        sc = dgp.build("clean")
        mmm, res = _fit(sc.panel(), _linear_trend())

        decomp = mmm.compute_component_decomposition()
        fitted_ptp = float(np.ptp(np.asarray(decomp.trend, dtype=float)))
        # True linear trend spans ~59.6 KPI units over the series.
        assert fitted_ptp > 40.0
        # Clean-world guard: media recovery stays tight.
        assert _med_abs_rel_err(mmm, sc) < 0.15

    def test_piecewise_trend_default_tracks_structural_break(self) -> None:
        """Old changepoint_prior_scale=0.05: corr 0.33 / med err ~26% here."""
        import numpy as np

        from mmm_framework.model import TrendConfig, TrendType
        from tests.synth import dgp

        sc = dgp.build("trend_break")
        mmm, res = _fit(sc.panel(), TrendConfig(type=TrendType.PIECEWISE))

        n = len(sc.y)
        t = np.arange(n)
        brk = int(n * 0.5)
        true_trend = 60.0 * (t / n) + np.where(t >= brk, -140.0 + 1.45 * (t - brk), 0.0)
        decomp = mmm.compute_component_decomposition()
        fitted = np.asarray(decomp.trend, dtype=float)

        assert float(np.corrcoef(fitted, true_trend)[0, 1]) > 0.6
        assert _med_abs_rel_err(mmm, sc) < 0.25


def _linear_trend():
    from mmm_framework.model import TrendConfig, TrendType

    return TrendConfig(type=TrendType.LINEAR)
