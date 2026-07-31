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
