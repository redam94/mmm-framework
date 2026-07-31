"""A windowed estimand answers the window, or refuses (#278).

Two gaps in `estimands/evaluate.py`, both of which returned a plausible number
rather than an error.

1. `contribution_roi` silently ignored `Estimand.window`. Measured on a 60-week
   MAP fit: a `TimeWindow(0, 9)` request returned **0.4335166608306849** — byte
   identical to the full-series value, `status="ok"` — where the true windowed
   ROI is 0.3942. Self-consistent and wrong.

2. The engine had no multiplicative guard, while `model/base.py` has two. The
   in-graph contribution Deterministic is an additive-scale quantity, so under a
   multiplicative specification `contribution × y_std` is a log-scale number
   published as an original-scale one — the identical defect
   `sample_channel_contributions` (#220/#251) and `compute_marginal_contributions`
   already refuse.

   The asymmetry was the actual bug, and it is resolved **per quantity, not per
   engine**: the contrast-based estimands go through `predict_under`, which
   returns the original scale, so differencing them is exactly the remedy those
   two guards prescribe in their own error text. They stay unguarded.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.config import InferenceMethod, ModelConfig, ModelSpecification
from mmm_framework.estimands.evaluate import EstimandEvaluator
from mmm_framework.estimands.registry import get as get_estimand
from mmm_framework.estimands.spec import TimeWindow
from mmm_framework.model import BayesianMMM, TrendConfig, TrendType
from mmm_framework.reporting.helpers.roi import (
    ContributionScaleUnsupported,
    ContributionWindowUnsupported,
    _get_contribution_samples,
)
from mmm_framework.utils.arviz_compat import posterior_from_dict

WINDOW = TimeWindow(start=0, end=9)


def _fitted(spec=ModelSpecification.ADDITIVE, n_weeks: int = 60):
    from mmm_framework.synth import dgp

    panel = dgp.build("clean", seed=3, n_weeks=n_weeks).panel()
    cfg = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_PYMC,
        n_chains=1,
        n_draws=30,
        n_tune=30,
        specification=spec,
    )
    m = BayesianMMM(panel, cfg, TrendConfig(type=TrendType.LINEAR))
    m.fit(method="map", random_seed=0, progressbar=False)
    return m


def _one(model, estimand):
    return list(EstimandEvaluator(model).evaluate([estimand]).values())[0]


@pytest.fixture(scope="module")
def additive_model():
    return _fitted()


class TestWindowIsHonoured:
    def test_windowed_roi_is_not_the_full_series_value(self, additive_model):
        m = additive_model
        base = get_estimand("contribution_roi")
        ch = m.channel_names[0]

        full = _one(m, base.model_copy(update={"target": ch}))
        win = _one(m, base.model_copy(update={"target": ch, "window": WINDOW}))

        assert full.status == "ok" and win.status == "ok"
        assert win.mean != full.mean, (
            "the windowed request returned the full-series number"
        )

    def test_windowed_roi_equals_the_hand_computed_window(self, additive_model):
        """Graded against the arithmetic, not against another estimand."""
        m = additive_model
        ch = m.channel_names[0]
        c_idx = list(m.channel_names).index(ch)

        cc = np.asarray(
            m._trace.posterior["channel_contributions"].values, dtype=float
        )
        cc = cc.reshape(-1, *cc.shape[-2:])
        mask = m._get_time_mask(WINDOW.as_tuple())
        contrib = float(np.mean(cc[:, mask, c_idx].sum(axis=-1) * m.y_std))
        spend = float(np.asarray(m.panel.X_media)[mask, c_idx].sum())

        win = _one(
            m,
            get_estimand("contribution_roi").model_copy(
                update={"target": ch, "window": WINDOW}
            ),
        )
        assert win.mean == pytest.approx(contrib / spend, rel=1e-9)

    def test_numerator_and_denominator_share_the_window(self, additive_model):
        """A windowed numerator over a full-series denominator is neither."""
        m = additive_model
        ch = m.channel_names[0]
        c_idx = list(m.channel_names).index(ch)
        mask = m._get_time_mask(WINDOW.as_tuple())

        full_spend = float(np.asarray(m.panel.X_media)[:, c_idx].sum())
        win_spend = float(np.asarray(m.panel.X_media)[mask, c_idx].sum())
        assert win_spend < full_spend  # non-vacuity

        cc = np.asarray(
            m._trace.posterior["channel_contributions"].values, dtype=float
        ).reshape(-1, m.n_obs, len(m.channel_names))
        win_contrib = float(np.mean(cc[:, mask, c_idx].sum(axis=-1) * m.y_std))

        win = _one(
            m,
            get_estimand("contribution_roi").model_copy(
                update={"target": ch, "window": WINDOW}
            ),
        )
        # The mismatched ratio would be win_contrib / full_spend.
        assert win.mean != pytest.approx(win_contrib / full_spend, rel=1e-6)
        assert win.mean == pytest.approx(win_contrib / win_spend, rel=1e-9)

    def test_the_unwindowed_path_is_untouched(self, additive_model):
        """`mask.all()` is exactly "no window", so bit-stability is preserved."""
        m = additive_model
        ch = m.channel_names[0]
        posterior = m._trace.posterior

        without = _get_contribution_samples(m, posterior, ch, m.y_mean, m.y_std)
        all_true = _get_contribution_samples(
            m, posterior, ch, m.y_mean, m.y_std, mask=np.ones(m.n_obs, dtype=bool)
        )
        np.testing.assert_array_equal(without, all_true)


class TestUnwindowableShapesRefuse:
    """Silence is the one option that is not acceptable."""

    def test_scalar_per_draw_contribution_raises(self):
        m = _fitted()
        ch = m.channel_names[0]
        post = posterior_from_dict(
            {
                "intercept": np.zeros((1, 4)),
                f"contribution_{ch}": np.full((1, 4), 5.0),  # no obs axis
            }
        )
        with pytest.raises(ContributionWindowUnsupported, match="per-draw scalar"):
            _get_contribution_samples(
                m, post.posterior, ch, m.y_mean, m.y_std,
                mask=m._get_time_mask(WINDOW.as_tuple()),
            )

    def test_the_same_shape_is_fine_unwindowed(self):
        m = _fitted()
        ch = m.channel_names[0]
        post = posterior_from_dict(
            {
                "intercept": np.zeros((1, 4)),
                f"contribution_{ch}": np.full((1, 4), 5.0),
            }
        )
        got = _get_contribution_samples(m, post.posterior, ch, m.y_mean, m.y_std)
        assert float(np.mean(got)) == pytest.approx(5.0 * m.y_std)

    def test_the_refusal_is_not_swallowed_into_the_beta_fallback(self):
        """The `channel_contributions` branch sits inside a broad `except`.

        Letting it catch the refusal would fall through to the `beta_<channel>`
        fallback and answer with a DIFFERENT number — the silent-wrong-answer
        behaviour the refusal exists to prevent.
        """
        m = _fitted()
        ch = m.channel_names[0]
        post = posterior_from_dict(
            {
                "intercept": np.zeros((1, 4)),
                # per-draw scalar per channel: no observation axis
                "channel_contributions": np.full((1, 4, len(m.channel_names)), 2.0),
                # a usable fallback the refusal must NOT silently defer to
                **{f"beta_{c}": np.full((1, 4), 0.4) for c in m.channel_names},
            }
        )
        with pytest.raises(ContributionWindowUnsupported):
            _get_contribution_samples(
                m, post.posterior, ch, m.y_mean, m.y_std,
                mask=m._get_time_mask(WINDOW.as_tuple()),
            )

    def test_the_engine_reports_it_rather_than_answering(self):
        m = _fitted()
        ch = m.channel_names[0]
        m._trace = posterior_from_dict(
            {
                "intercept": np.zeros((1, 4)),
                f"contribution_{ch}": np.full((1, 4), 5.0),
            }
        )
        res = _one(
            m,
            get_estimand("contribution_roi").model_copy(
                update={"target": ch, "window": WINDOW}
            ),
        )
        assert res.status != "ok"
        assert res.mean is None


class TestMultiplicativeGuardAgrees:
    """`model/base.py` refused in two places and the engine in none."""

    def test_model_refuses_the_additive_scale_methods(self):
        m = _fitted(spec=ModelSpecification.MULTIPLICATIVE)
        assert m._multiplicative
        with pytest.raises(NotImplementedError, match="multiplicative"):
            m.sample_channel_contributions()
        with pytest.raises(NotImplementedError, match="multiplicative"):
            m.compute_marginal_contributions()

    def test_engine_now_refuses_the_same_quantity(self):
        m = _fitted(spec=ModelSpecification.MULTIPLICATIVE)
        res = _one(
            m,
            get_estimand("contribution_roi").model_copy(
                update={"target": m.channel_names[0]}
            ),
        )
        assert res.status != "ok", (
            "the in-graph contribution path published a log-scale number as an "
            "original-scale one"
        )
        assert res.mean is None

    def test_contrast_estimands_stay_available(self):
        """Deliberately NOT guarded: `predict_under` returns the original scale.

        Refusing these would be the over-broad reading — they are the very
        remedy the model's own error text prescribes.
        """
        m = _fitted(spec=ModelSpecification.MULTIPLICATIVE)
        res = _one(
            m,
            get_estimand("counterfactual_roi").model_copy(
                update={"target": m.channel_names[0]}
            ),
        )
        assert res.status == "ok"
        assert res.mean is not None and np.isfinite(res.mean)

    def test_additive_models_are_unaffected(self, additive_model):
        res = _one(
            additive_model,
            get_estimand("contribution_roi").model_copy(
                update={"target": additive_model.channel_names[0]}
            ),
        )
        assert res.status == "ok" and res.mean is not None


class TestTheRefusalCoversBothCallers:
    """A guard in one caller left the higher-traffic surface publishing it.

    `_get_contribution_samples` has two consumers: the estimand engine AND
    `compute_roi_with_uncertainty`, which the classic report's ROI table
    renders. Guarding only the engine produced a SELF-CONTRADICTING report — the
    Estimand Results section omitted `contribution_roi` as unsupported while the
    ROI table in the same file printed 0.00 "Underperforming", a 550x
    understatement of the correct original-scale 0.61.
    """

    def test_compute_roi_with_uncertainty_refuses(self):
        from mmm_framework.reporting.helpers.roi import compute_roi_with_uncertainty

        m = _fitted(spec=ModelSpecification.MULTIPLICATIVE)
        with pytest.raises(ContributionScaleUnsupported, match="LOG scale"):
            compute_roi_with_uncertainty(m, hdi_prob=0.94)

    def test_the_report_section_degrades_rather_than_publishing(self):
        from mmm_framework.reporting.extractors.bayesian import BayesianMMMExtractor

        m = _fitted(spec=ModelSpecification.MULTIPLICATIVE)
        assert BayesianMMMExtractor(m)._compute_channel_roi() is None

    def test_additive_models_are_untouched(self, additive_model):
        from mmm_framework.reporting.helpers.roi import compute_roi_with_uncertainty

        df = compute_roi_with_uncertainty(additive_model, hdi_prob=0.94)
        assert len(df) == len(additive_model.channel_names)


class TestLinkScaleModelsAreCaughtToo:
    """`_multiplicative` is only ONE of the two ways the KPI bridge fails.

    A count/bounded likelihood sets `y_std = 1.0`, so `to_kpi_units` is the
    identity over logits. Measured on the binomial awareness garden model, an
    unguarded `contribution_roi` published 0.0071 for a channel whose
    original-scale ROI-equivalent is ~1.5 — over 200x too small, `status="ok"`,
    `units="ROI"`.
    """

    def test_the_two_reasons_are_both_named(self):
        from mmm_framework.config import LikelihoodConfig
        from mmm_framework.model.component_scale import kpi_scale_bridge_reason

        m = _fitted()
        assert kpi_scale_bridge_reason(m) is None

        mult = _fitted(spec=ModelSpecification.MULTIPLICATIVE)
        assert "LOG scale" in (kpi_scale_bridge_reason(mult) or "")

        m.model_config.likelihood = LikelihoodConfig(family="binomial")
        assert "LINK scale" in (kpi_scale_bridge_reason(m) or "")

    @pytest.mark.parametrize("family", ["normal", "student_t"])
    def test_gaussian_families_are_allowed(self, family):
        from mmm_framework.config import LikelihoodConfig
        from mmm_framework.model.component_scale import kpi_scale_bridge_reason

        m = _fitted()
        m.model_config.likelihood = LikelihoodConfig(family=family)
        assert kpi_scale_bridge_reason(m) is None

    def test_an_unrecognized_model_is_not_refused(self):
        """Over-broad is the other failure mode: a Mock must not be refused.

        `getattr(mock, "_multiplicative", False)` returns a truthy Mock, so a
        loose predicate refuses every duck-typed model and test double — and
        `_get_contribution_samples` is deliberately tolerant of those.
        """
        from unittest.mock import Mock

        from mmm_framework.model.component_scale import kpi_scale_bridge_reason

        assert kpi_scale_bridge_reason(Mock()) is None
        assert kpi_scale_bridge_reason(object()) is None


class TestRefusalsNeverEscapeTheBatch:
    def test_a_denominator_refusal_returns_unsupported(self):
        from mmm_framework.estimands.spec import (
            Contrast,
            Contribution,
            Estimand,
            Outcome,
            ZeroInput,
        )

        m = _fitted(spec=ModelSpecification.MULTIPLICATIVE)
        ch = m.channel_names[0]
        share = Estimand(
            name="contribution_share",
            kind="share",
            numerator=Contrast(
                quantity=Outcome(),
                baseline=ZeroInput(target=ch),
                op="difference",
                reduce="sum",
            ),
            denominator=Contribution(target=ch, source="in_graph_deterministic"),
        )
        res = m.evaluate_estimands([get_estimand("contribution"), share])
        # The batch survives: the other estimand still has its result.
        assert any(v.status == "ok" for v in res.values())
        assert res["contribution_share"].status == "unsupported"
        assert res["contribution_share"].reason


class TestAnEmptyWindowIsARefusal:
    def test_out_of_range_window_says_so(self, additive_model):
        """It used to vanish from the results dict — neither answer nor refusal."""
        est = get_estimand("contribution_roi").model_copy(
            update={
                "target": additive_model.channel_names[0],
                "window": TimeWindow(start=500, end=600),
            }
        )
        res = EstimandEvaluator(additive_model).evaluate([est])
        assert res, "the estimand vanished from the results dict"
        got = list(res.values())[0]
        assert got.status == "unsupported"
        assert "selects no observations" in got.reason

    def test_the_causal_assumptions_no_longer_claim_the_full_period(self):
        text = get_estimand("contribution_roi").causal_assumptions
        assert "over the full period" not in text
        assert "window" in text
