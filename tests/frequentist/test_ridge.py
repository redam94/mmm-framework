"""The penalized linear solve, and the ridge-vs-MAP question it settles.

Epic #180 opens by asking what ``frequentist_ridge`` buys over
``fit(method="map")``, since ridge regression *is* MAP estimation under Gaussian
priors. The answer — and this file is where it stops being a docstring assertion
and becomes an executable fact — is that the equivalence is real but does **not**
describe this framework's default configuration:

* ``TestRidgeIsMapUnderGaussianPriors`` fits a model whose media coefficients have
  *explicitly configured* Normal priors and shows ridge reproduces MAP exactly,
  with ``lambda_j = sigma^2 / tau_j^2``;
* ``TestRidgeIsNotMapByDefault`` shows the shipped default — ``Gamma(mu=1.5,
  sigma=1)`` on the coefficient, or ``LogNormal(0, 1)`` on ROI — is not Gaussian,
  so the two estimators genuinely differ.

Without the second test, a future reader could reasonably "simplify" the
frequentist path into an alias for ``method="map"`` and silently change what the
media block estimates.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mmm_framework.config import MediaChannelConfig, MFFConfig, ModelConfig, PriorConfig
from mmm_framework.config.enums import SaturationType
from mmm_framework.frequentist.design import build_design_matrix
from mmm_framework.frequentist.ridge import fit_ridge
from mmm_framework.model import BayesianMMM
from mmm_framework.model.base import _PRECISION_CONTROL_PRIOR_SIGMA
from mmm_framework.model.trend_config import TrendConfig, TrendType

from test_design_equivalence import CHANNELS, _configure, _panel

ALPHA = {c: {"alpha": 0.55} for c in CHANNELS}
LAM = {c: {"sat_lam": 2.7} for c in CHANNELS}


def _design(model_config=None, trend=TrendType.LINEAR, panel=None):
    panel = panel if panel is not None else _panel()
    _configure(panel, "geometric", SaturationType.LOGISTIC)
    return build_design_matrix(
        panel,
        alpha=ALPHA,
        lam=LAM,
        model_config=model_config or ModelConfig(),
        trend_config=TrendConfig(type=trend),
    )


class TestClosedForm:
    def test_zero_penalty_is_ordinary_least_squares(self):
        d = _design()
        got = fit_ridge(d, penalty=0.0).theta
        expected, *_ = np.linalg.lstsq(d.X, d.y, rcond=None)
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-10)

    def test_recovers_planted_coefficients_in_the_ols_limit(self):
        """A design with a known theta and no noise must be solved exactly."""
        d = _design()
        rng = np.random.default_rng(0)
        planted = rng.standard_normal(d.n_params)
        got = fit_ridge(d, y=d.X @ planted, penalty=0.0).theta
        np.testing.assert_allclose(got, planted, rtol=1e-8, atol=1e-9)

    def test_augmented_system_matches_the_normal_equations(self):
        """The augmented solve must equal (X'X + lambda P)^-1 X'y."""
        d = _design()
        lam = 3.7
        P = np.diag(d.penalize.astype(float))
        expected = np.linalg.solve(d.X.T @ d.X + lam * P, d.X.T @ d.y)
        np.testing.assert_allclose(
            fit_ridge(d, penalty=lam).theta, expected, rtol=1e-8, atol=1e-10
        )

    def test_penalty_shrinks_only_the_penalized_block(self):
        d = _design()
        weak = fit_ridge(d, penalty=1e-8)
        strong = fit_ridge(d, penalty=1e4)
        pen = d.penalize.astype(bool)
        assert np.abs(strong.theta[pen]).sum() < np.abs(weak.theta[pen]).sum()

    def test_negative_penalty_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            fit_ridge(_design(), penalty=-1.0)


class TestEffectiveDof:
    def test_equals_column_count_with_no_penalty(self):
        d = _design()
        fit = fit_ridge(d, penalty=0.0)
        assert fit.effective_dof == pytest.approx(d.n_params, rel=1e-6)

    def test_falls_toward_the_unpenalized_count_as_the_penalty_grows(self):
        d = _design()
        n_free = int((~d.penalize.astype(bool)).sum())
        seq = [fit_ridge(d, penalty=p).effective_dof for p in (0.0, 1.0, 1e3, 1e9)]
        assert all(a >= b - 1e-6 for a, b in zip(seq, seq[1:], strict=False))
        assert seq[-1] == pytest.approx(n_free, abs=1e-3)


class TestScaleInvariance:
    def test_changing_a_channels_units_leaves_its_contribution_unchanged(self):
        """Media is normalized by its own training max, so a unit change cancels.

        This is the property that makes a single penalty meaningful across
        channels: if it failed, the fitted contribution would depend on whether
        spend was recorded in dollars or thousands of dollars.
        """
        base = _panel()
        scaled = _panel()
        scaled.X_media = scaled.X_media.copy()
        scaled.X_media["TV"] = scaled.X_media["TV"] * 1000.0

        d0, d1 = _design(panel=base), _design(panel=scaled)
        f0, f1 = fit_ridge(d0, penalty=1.0), fit_ridge(d1, penalty=1.0)

        i = d0.columns.index("media_TV")
        np.testing.assert_allclose(
            d0.X[:, i] * f0.theta[i], d1.X[:, i] * f1.theta[i], rtol=1e-8, atol=1e-10
        )


class TestNonNegativity:
    def test_pins_a_negative_coefficient_at_exactly_zero_and_flags_it(self):
        """A coefficient at its constraint has no meaningful interval.

        The bootstrap will misreport it if it is treated as interior, so the fit
        surfaces the active constraint rather than leaving it to be inferred from
        a suspiciously small number.
        """
        d = _design()
        i = d.columns.index("media_TV")
        # A world where TV's true effect is negative.
        planted = np.zeros(d.n_params)
        planted[d.columns.index("intercept")] = 1.0
        planted[i] = -5.0
        fit = fit_ridge(d, y=d.X @ planted, penalty=0.0, nonneg=True)

        assert fit.theta[i] == 0.0
        assert fit.at_boundary[i]
        assert not fit.at_boundary[d.columns.index("intercept")]

    def test_unpenalized_structural_columns_keep_their_sign(self):
        """A negative trend or seasonal coefficient is meaningful; only the
        penalized (media/control/geo) block is constrained."""
        d = _design()
        planted = np.zeros(d.n_params)
        planted[d.columns.index("trend_slope")] = -3.0
        planted[d.columns.index("intercept")] = 2.0
        fit = fit_ridge(d, y=d.X @ planted, penalty=0.0, nonneg=True)
        assert fit.theta[d.columns.index("trend_slope")] < 0


class TestDiagonalPenalty:
    def test_per_column_weights_shrink_unequally(self):
        """`beta_controls` has role-dependent prior widths, so P is diagonal.

        A confounder gets a wide prior (weak penalty) because shrinking it
        re-opens the back-door; a uniform scalar penalty would lose that.
        """
        d = _design()
        i, j = d.columns.index("media_TV"), d.columns.index("media_Digital")
        w = d.penalize.astype(float)
        w[i] = 100.0  # hammer TV
        w[j] = 0.01  # barely touch Digital
        fit = fit_ridge(d, penalty=1.0, penalize=w)
        flat = fit_ridge(d, penalty=1.0)
        assert abs(fit.theta[i]) < abs(flat.theta[i])
        assert abs(fit.theta[j]) > abs(flat.theta[j])

    def test_boolean_mask_is_the_uniform_case(self):
        d = _design()
        a = fit_ridge(d, penalty=2.0, penalize=d.penalize.astype(bool))
        b = fit_ridge(d, penalty=2.0, penalize=d.penalize.astype(float))
        np.testing.assert_allclose(a.theta, b.theta, rtol=0, atol=0)

    def test_negative_weights_are_rejected(self):
        d = _design()
        w = d.penalize.astype(float)
        w[0] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            fit_ridge(d, penalty=1.0, penalize=w)


def _map_fit(panel, model_config, trend_config):
    mmm = BayesianMMM(panel, model_config, trend_config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mmm.fit(method="map", random_seed=0, progressbar=False)
    post = mmm._trace.posterior
    return mmm, {k: np.asarray(post[k].values).ravel() for k in post.data_vars}


@pytest.mark.slow
class TestRidgeIsMapUnderGaussianPriors:
    """The equivalence, made executable.

    MAP maximizes ``-||y - X.theta||^2 / (2 sigma^2) - sum_j theta_j^2/(2 tau_j^2)``
    over the Gaussian blocks, whose argmax given ``sigma`` is exactly ridge with
    ``lambda_j = sigma^2 / tau_j^2``. Transforms are held at the MAP's own
    estimates so the comparison is conditional-on-transforms, as the frequentist
    path always is.
    """

    def test_matches_map_to_optimizer_tolerance(self):
        panel = _panel()
        _configure(panel, "geometric", SaturationType.LOGISTIC)
        tau_media = 0.4
        base = panel.config
        panel.config = MFFConfig(
            kpi=base.kpi,
            media_channels=[
                MediaChannelConfig(
                    name=c,
                    dimensions=base.media_channels[0].dimensions,
                    adstock=base.media_channels[0].adstock,
                    saturation=base.media_channels[0].saturation,
                    # An EXPLICIT Normal prior -- `_explicit_prior` honors only
                    # fields the caller actually set, so factory defaults would
                    # not reach the graph.
                    coefficient_prior=PriorConfig(
                        distribution="Normal", params={"mu": 0.0, "sigma": tau_media}
                    ),
                )
                for c in CHANNELS
            ],
            controls=base.controls,
        )
        cfg = ModelConfig()
        trend = TrendConfig(type=TrendType.NONE)  # avoids the non-zero-mean slope prior
        mmm, point = _map_fit(panel, cfg, trend)

        design = build_design_matrix(
            panel,
            alpha={
                c: {"alpha": float(point[f"adstock_alpha_{c}"][0])} for c in CHANNELS
            },
            lam={c: {"sat_lam": float(point[f"sat_lam_{c}"][0])} for c in CHANNELS},
            model_config=cfg,
            trend_config=trend,
        )

        sigma = float(point["sigma"][0])
        tau = np.empty(design.n_params)
        for i, name in enumerate(design.columns):
            if name == "intercept":
                tau[i] = cfg.intercept_prior_sigma
            elif name.startswith("season_"):
                tau[i] = mmm.seasonality_config.prior_sigma_for("yearly")
            elif name.startswith("media_"):
                tau[i] = tau_media
            elif name.startswith("control_"):
                tau[i] = _PRECISION_CONTROL_PRIOR_SIGMA
            else:  # pragma: no cover - the fixture has no other columns
                raise AssertionError(f"unmapped column {name}")

        fit = fit_ridge(design, penalty=sigma**2, penalize=1.0 / tau**2)

        expected = np.array(
            [
                float(point["intercept"][0]),
                *(
                    float(np.atleast_1d(point["season_yearly"])[j])
                    for j in range(sum(c.startswith("season_") for c in design.columns))
                ),
                *(float(point[f"beta_{c}"][0]) for c in CHANNELS),
                float(np.atleast_1d(point["beta_controls"])[0]),
            ]
        )
        np.testing.assert_allclose(fit.theta, expected, rtol=2e-3, atol=2e-3)


class TestRidgeIsNotMapByDefault:
    """The converse, which is what justifies the epic.

    The shipped media prior is ``Gamma(mu=1.5, sigma=1)`` (coefficient mode) or
    ``LogNormal(0, 1)`` on ROI (the agent default). Neither is Gaussian, so no
    choice of ridge penalty reproduces MAP on the media block. If this assertion
    ever flips, the frequentist path really has become a synonym for
    ``method="map"`` and the epic's justification is gone.

    Read off the built graph rather than from a fit: the prior family is a
    structural property, and asserting it directly is both faster and immune to
    an optimizer's mood.
    """

    @pytest.mark.parametrize("mode", ["coefficient", "roi"])
    def test_default_media_prior_is_not_gaussian(self, mode):
        panel = _panel()
        _configure(panel, "geometric", SaturationType.LOGISTIC)
        mmm = BayesianMMM(
            panel,
            ModelConfig(media_prior_mode=mode),
            TrendConfig(type=TrendType.LINEAR),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graph = mmm.model

        families = {
            rv.name: rv.owner.op.__class__.__name__.replace("RV", "")
            for rv in graph.free_RVs
            if rv.name.startswith(("beta_TV", "beta_Digital", "roi_TV", "roi_Digital"))
        }
        assert families, "no media parameter found on the graph"
        assert set(families.values()) <= {"Gamma", "LogNormal"}, families
        assert "Normal" not in set(families.values()), families

    def test_the_gaussian_blocks_really_are_gaussian(self):
        """The other half of the picture: those blocks ARE ridge-equivalent."""
        panel = _panel()
        _configure(panel, "geometric", SaturationType.LOGISTIC)
        mmm = BayesianMMM(panel, ModelConfig(), TrendConfig(type=TrendType.LINEAR))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graph = mmm.model
        families = {
            rv.name: rv.owner.op.__class__.__name__.replace("RV", "")
            for rv in graph.free_RVs
        }
        for name in ("intercept", "trend_slope", "season_yearly", "beta_controls"):
            assert families[name] == "Normal", (name, families[name])
