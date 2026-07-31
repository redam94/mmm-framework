"""The report's ROI extractor reads contributions the canonical way (#276).

`BayesianMMMExtractor._compute_channel_roi` re-implemented
`reporting.helpers.roi._get_contribution_samples`' three-branch precedence and
got two branches arithmetically wrong:

* a scalar-per-draw contribution was multiplied by `n_obs` — the canonical form
  has no such factor;
* the `beta_<channel>` fallback used `beta * y_std * n_obs * 0.5` under a
  comment calling itself a rough estimate, against a canonical
  `beta * media_sum * y_std`. The ratio is `media_sum / (0.5 * n_obs)` — twice
  the mean per-period spend.

Both fire exactly for models exposing only a coefficient: the bespoke garden
models, whose ROI table is the number they are read for, with no marker
distinguishing the branch from the primary one.

Measured on a 60-week clean panel (`y_std = 56.66`, `beta = 0.4` for every
channel), the old formula gave **every channel the identical contribution
679.9** regardless of its spend, against canonical totals of 79,575 / 52,300 /
41,060 / 33,343 — ratios of 117.05x, 76.93x, 60.39x, 49.04x.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.config import InferenceMethod, ModelConfig
from mmm_framework.model import BayesianMMM, TrendConfig, TrendType
from mmm_framework.reporting.extractors.bayesian import BayesianMMMExtractor
from mmm_framework.reporting.helpers.measurement import resolve_channel_divisor
from mmm_framework.reporting.helpers.roi import (
    _get_contribution_samples,
    compute_roi_with_uncertainty,
)
from mmm_framework.utils.arviz_compat import posterior_from_dict

N_DRAWS = 8
BETA = 0.4


def _model(n_weeks: int = 60):
    from mmm_framework.synth import dgp

    panel = dgp.build("clean", seed=3, n_weeks=n_weeks).panel()
    cfg = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_PYMC, n_chains=1, n_draws=N_DRAWS
    )
    return BayesianMMM(panel, cfg, TrendConfig(type=TrendType.LINEAR))


def _beta_only_model():
    """The fallback population: a coefficient and nothing else."""
    m = _model()
    post = {"intercept": np.zeros((1, N_DRAWS)), "sigma": np.full((1, N_DRAWS), 0.1)}
    for ch in m.channel_names:
        post[f"beta_{ch}"] = np.full((1, N_DRAWS), BETA)
    m._trace = posterior_from_dict(post)
    return m


def _scalar_contribution_model():
    """A registered per-draw SCALAR contribution — no time axis."""
    m = _model()
    post = {"intercept": np.zeros((1, N_DRAWS)), "sigma": np.full((1, N_DRAWS), 0.1)}
    for i, ch in enumerate(m.channel_names):
        post[f"contribution_{ch}"] = np.full((1, N_DRAWS), 10.0 + i)
    m._trace = posterior_from_dict(post)
    return m


def _panel_contribution_model():
    """The PRIMARY branch: a registered (obs, channel) `channel_contributions`."""
    m = _model()
    rng = np.random.default_rng(0)
    cc = rng.normal(0.05, 0.01, (1, N_DRAWS, m.n_obs, len(m.channel_names)))
    post = {
        "intercept": np.zeros((1, N_DRAWS)),
        "sigma": np.full((1, N_DRAWS), 0.1),
        "channel_contributions": cc,
    }
    m._trace = posterior_from_dict(post)
    return m


def _extractor_contribution(m, roi_results, ch):
    """Invert the section's own ROI back to its contribution numerator."""
    return roi_results[ch]["mean"] * resolve_channel_divisor(m, ch).total


def _executable_source(obj) -> str:
    """Source with comments stripped, so a comment ABOUT the old arithmetic
    (this PR adds several) cannot be mistaken for the arithmetic itself."""
    import ast
    import inspect
    import textwrap

    return ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(obj))))


class TestTheDefectiveFactorsAreGone:
    def test_no_n_obs_or_half_factors_survive(self):
        from mmm_framework.reporting.extractors import bayesian

        for fn in (
            bayesian.BayesianMMMExtractor._compute_channel_roi,
            bayesian.BayesianMMMExtractor._get_component_totals,
        ):
            src = _executable_source(fn)
            assert "n_obs * 0.5" not in src, fn.__name__
            assert "y_std * n_obs" not in src, fn.__name__

    def test_no_manufactured_interval_constants_remain(self):
        from mmm_framework.reporting.extractors import bayesian

        src = _executable_source(
            bayesian.BayesianMMMExtractor._compute_marketing_attribution
        )
        assert "0.85" not in src and "1.15" not in src


class TestBetaOnlyFallback:
    """Acceptance criterion 1: one contribution total, two surfaces."""

    def test_report_roi_matches_the_canonical_reader(self):
        m = _beta_only_model()
        roi = BayesianMMMExtractor(m)._compute_channel_roi()
        assert roi, "no ROI produced on the beta-only fallback"

        for ch in m.channel_names:
            canonical = float(
                np.mean(
                    _get_contribution_samples(
                        m, m._trace.posterior, ch, m.y_mean, m.y_std
                    )
                )
            )
            assert _extractor_contribution(m, roi, ch) == pytest.approx(
                canonical, rel=1e-9
            )

    def test_report_roi_matches_roi_metrics(self):
        """`roi_metrics` (compute_roi_with_uncertainty) is the other surface."""
        m = _beta_only_model()
        roi = BayesianMMMExtractor(m)._compute_channel_roi()
        df = compute_roi_with_uncertainty(m, hdi_prob=0.94)
        by_channel = dict(zip(df["channel"], df["contribution_mean"]))

        for ch in m.channel_names:
            assert _extractor_contribution(m, roi, ch) == pytest.approx(
                float(by_channel[ch]), rel=1e-9
            )

    def test_the_old_formula_was_spend_blind(self):
        """Non-vacuity, and the sharpest statement of the defect.

        `beta * y_std * n_obs * 0.5` contains no spend term, so every channel
        got the SAME contribution however much was spent on it.
        """
        m = _beta_only_model()
        old = BETA * m.y_std * m.n_obs * 0.5
        totals = {
            ch: float(
                np.mean(
                    _get_contribution_samples(
                        m, m._trace.posterior, ch, m.y_mean, m.y_std
                    )
                )
            )
            for ch in m.channel_names
        }
        assert len({round(v, 6) for v in totals.values()}) == len(totals), (
            "channels must differ once spend is in the numerator"
        )
        ratios = sorted(v / old for v in totals.values())
        assert ratios[0] > 40 and ratios[-1] > 100

        # The ratio is exactly media_sum / (0.5 * n_obs).
        for c, ch in enumerate(m.channel_names):
            media_sum = float(np.asarray(m.panel.X_media)[:, c].sum())
            assert totals[ch] / old == pytest.approx(
                media_sum / (0.5 * m.n_obs), rel=1e-9
            )


class TestScalarPerDrawBranch:
    """Acceptance criterion 3: a branch no test exercised."""

    def test_no_spurious_n_obs_factor(self):
        m = _scalar_contribution_model()
        roi = BayesianMMMExtractor(m)._compute_channel_roi()
        assert roi

        for i, ch in enumerate(m.channel_names):
            want = (10.0 + i) * m.y_std  # NOT x n_obs
            assert _extractor_contribution(m, roi, ch) == pytest.approx(want, rel=1e-9)

    def test_the_old_factor_would_have_been_n_obs_times_larger(self):
        m = _scalar_contribution_model()
        roi = BayesianMMMExtractor(m)._compute_channel_roi()
        ch = m.channel_names[0]
        got = _extractor_contribution(m, roi, ch)
        assert got * m.n_obs == pytest.approx(10.0 * m.y_std * m.n_obs, rel=1e-9)


class TestPrimaryBranchUnchanged:
    """Acceptance criterion 4: the common path must not move."""

    def test_channel_contributions_path_is_exact(self):
        m = _panel_contribution_model()
        roi = BayesianMMMExtractor(m)._compute_channel_roi()
        assert roi

        cc = np.asarray(m._trace.posterior["channel_contributions"].values, dtype=float)
        cc = cc.reshape(-1, *cc.shape[-2:])  # (S, obs, channel)
        for c, ch in enumerate(m.channel_names):
            want = float(np.mean(cc[:, :, c].sum(axis=-1) * m.y_std))
            assert _extractor_contribution(m, roi, ch) == pytest.approx(want, rel=1e-9)


class TestComponentTotalsAgree:
    def test_totals_match_the_roi_numerator(self):
        """The same channel cannot have two totals in one document."""
        m = _beta_only_model()
        ex = BayesianMMMExtractor(m)
        roi = ex._compute_channel_roi()
        totals = ex._get_component_totals()
        assert totals

        for ch in m.channel_names:
            assert totals[ch] == pytest.approx(
                _extractor_contribution(m, roi, ch), rel=1e-9
            )


class TestNoManufacturedInterval:
    def test_absent_hdi_yields_no_interval_rather_than_plus_minus_15pct(self):
        from types import SimpleNamespace

        m = _beta_only_model()
        ex = BayesianMMMExtractor(m)

        contribs = SimpleNamespace(
            total_contributions=np.array([100.0, 200.0]),
            contribution_hdi_low=None,
            contribution_hdi_high=None,
        )
        m.compute_counterfactual_contributions = lambda **kw: contribs

        got = ex._compute_marketing_attribution()
        assert got["mean"] == pytest.approx(300.0)
        assert got["lower"] is None and got["upper"] is None
        # The manufactured values must not reappear.
        assert got["lower"] != pytest.approx(255.0)
        assert got["upper"] != pytest.approx(345.0)

    def test_a_real_hdi_is_still_used(self):
        from types import SimpleNamespace

        m = _beta_only_model()
        ex = BayesianMMMExtractor(m)
        contribs = SimpleNamespace(
            total_contributions=np.array([100.0, 200.0]),
            contribution_hdi_low=np.array([90.0, 180.0]),
            contribution_hdi_high=np.array([110.0, 220.0]),
        )
        m.compute_counterfactual_contributions = lambda **kw: contribs

        got = ex._compute_marketing_attribution()
        assert got["lower"] == pytest.approx(270.0)
        assert got["upper"] == pytest.approx(330.0)

    def test_the_render_sites_survive_a_missing_interval(self):
        from mmm_framework.reporting.sections import _format_interval

        assert _format_interval(None, None, lambda v: f"{v}", 80) == ""
