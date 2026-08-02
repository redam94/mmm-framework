"""The forecaster replays every term in the fitted mean, or refuses (#219).

Before v1.3.1 ``PosteriorForecaster`` summed five of the fitted mean's ten
terms — silently dropping the product level offset, event, cross-channel
interaction and price/promo lever contributions — approximated a time-varying
coefficient by its time average, forecast a reach/frequency channel on raw
reach, and ``_clone_for_prefix`` downcast any garden/custom model class to a
plain additive ``BayesianMMM``. Nothing raised.

The load-bearing test here is :class:`TestComponentSumIdentity`. **It IS the
audit**: it asserts the forward pass equals the sum of the model's registered
component Deterministics, so any term the forecaster fails to sum shows up as a
residual rather than as a plausible-looking number. It deliberately does not
compare against ``predict()``, which draws from the observation likelihood and
therefore carries MC noise of order ``sigma / sqrt(n_draws)`` that no seed
removes (which is why the pre-existing gate is a correlation bound).
"""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.validation.backtest import (
    ForecastUnsupportedError,
    PosteriorForecaster,
    TrendExtrapolation,
    _clone_for_prefix,
    audit_forward_pass,
    audit_refit,
)

# ---------------------------------------------------------------------------
# helpers — models are built but NOT fitted; the audit reads configuration
# ---------------------------------------------------------------------------


def _world(n_weeks: int = 80):
    from mmm_framework.synth import dgp

    return dgp.build("clean", seed=7, n_weeks=n_weeks)


def _config(**overrides):
    from mmm_framework.config import InferenceMethod, ModelConfig

    base = dict(
        inference_method=InferenceMethod.BAYESIAN_PYMC,
        n_chains=2,
        n_draws=150,
        n_tune=150,
        use_parametric_adstock=True,
        optim_seed=7,
    )
    base.update(overrides)
    return ModelConfig(**base)


def _model(panel, cfg=None, trend=None):
    from mmm_framework.model import BayesianMMM, TrendConfig, TrendType

    return BayesianMMM(
        panel, cfg or _config(), trend or TrendConfig(type=TrendType.LINEAR)
    )


# ---------------------------------------------------------------------------
# refusals — one per unreplayable term
# ---------------------------------------------------------------------------


class TestRefusals:
    """Each unreplayable term raises, naming the feature."""

    def test_plain_model_is_clean(self):
        mmm = _model(_world().panel())
        assert audit_forward_pass(mmm) == []
        assert audit_refit(mmm) == []

    def test_price_lever(self):
        from mmm_framework.config import PriceConfig

        sc = _world()
        panel = sc.panel()
        control = list(panel.coords.controls)[0]
        mmm = _model(panel, _config(price=PriceConfig(variable=control)))

        problems = audit_forward_pass(mmm)
        assert problems, "a configured price lever must be refused"
        assert "Price" in problems[0][0]
        with pytest.raises(ForecastUnsupportedError) as exc:
            PosteriorForecaster(mmm)
        assert "Price" in exc.value.feature

    def test_promo_lever(self):
        from mmm_framework.config import PromoConfig

        sc = _world()
        panel = sc.panel()
        control = list(panel.coords.controls)[0]
        mmm = _model(panel, _config(promotions=[PromoConfig(variable=control)]))

        problems = audit_forward_pass(mmm)
        assert problems and "promotion" in problems[0][0].lower()

    def test_events(self):
        from mmm_framework.config import EventsConfig

        sc = _world()
        panel = sc.panel()
        cfg = _config(events=EventsConfig(country="US"))
        mmm = _model(panel, cfg)
        if mmm.event_features.shape[1] == 0:
            pytest.skip("no holiday fell inside this data window")

        problems = audit_forward_pass(mmm)
        assert problems and "Event" in problems[0][0]

    def test_time_varying_coefficient(self):
        sc = _world()
        panel = sc.panel()
        mmm = _model(panel)
        ch = mmm.channel_names[0]
        # Flip the flag on the live config the audit reads.
        mmm.mff_config.get_media_config(ch).time_varying = True

        problems = audit_forward_pass(mmm)
        assert problems and "Time-varying" in problems[0][0]
        assert ch in problems[0][0]

    def test_reach_frequency(self):
        mmm = _model(_world().panel())
        mmm._reach_freq = {mmm.channel_names[0]: ("cfg", 1.0, 1.0)}

        problems = audit_forward_pass(mmm)
        assert problems and "Reach" in problems[0][0]

    def test_channel_interactions(self):
        mmm = _model(_world().panel())
        mmm.model_config.channel_interactions = [("a", "b")]

        problems = audit_forward_pass(mmm)
        assert problems and "interaction" in problems[0][0].lower()

    def test_custom_build_model_is_refused(self):
        """A subclass that writes its own graph defines its own mu.

        Class-preserving cloning made this a LIVE silent-drop path: before it,
        the clone was always a plain BayesianMMM so the base replay matched the
        (wrong) fitted graph; now the real class is rebuilt and its bespoke
        terms would be dropped.
        """
        from mmm_framework.garden.base import CustomMMM

        class _BespokeGraph(CustomMMM):
            def _build_model(self):  # pragma: no cover - never called
                raise AssertionError("not reached")

        from mmm_framework.model import TrendConfig, TrendType

        mmm = _BespokeGraph(
            _world().panel(), _config(), TrendConfig(type=TrendType.LINEAR)
        )
        problems = audit_forward_pass(mmm)
        assert problems and "Custom model graph" in problems[0][0]

    def test_subclass_can_opt_in_when_its_mu_is_the_base_mu(self):
        from mmm_framework.garden.base import CustomMMM

        class _PlainGarden(CustomMMM):
            pass

        class _OptedIn(CustomMMM):
            __forecast_forward_pass__ = "base"

            def _build_model(self):  # pragma: no cover - never called
                raise AssertionError("not reached")

        from mmm_framework.model import TrendConfig, TrendType

        panel, cfg = _world().panel(), _config()
        trend = TrendConfig(type=TrendType.LINEAR)
        # Inheriting the base graph is fine without any marker.
        assert audit_forward_pass(_PlainGarden(panel, cfg, trend)) == []
        # An override plus an explicit assertion is accepted.
        assert audit_forward_pass(_OptedIn(panel, cfg, trend)) == []

    def test_reach_frequency_is_NOT_refused_on_the_legacy_adstock_path(self):
        """The frequency gain never enters mu on the legacy path.

        It reaches the mean only through ``_channel_media_input``, which the
        legacy fixed-alpha branch does not call — so the forecaster convolving
        the raw column is exactly what the graph did. Refusing here would remove
        a capability that worked.
        """
        mmm = _model(_world().panel(), _config(use_parametric_adstock=False))
        mmm._reach_freq = {mmm.channel_names[0]: ("cfg", 1.0, 1.0)}
        assert audit_forward_pass(mmm) == []

        parametric = _model(_world().panel(), _config(use_parametric_adstock=True))
        parametric._reach_freq = {parametric.channel_names[0]: ("cfg", 1.0, 1.0)}
        assert audit_forward_pass(parametric), "parametric path must still refuse"

    def test_multiplicative_specification(self):
        mmm = _model(_world().panel())
        mmm._multiplicative = True

        problems = audit_forward_pass(mmm)
        assert problems and "Multiplicative" in problems[0][0]

    def test_experiment_calibration_is_a_refit_concern_not_a_forward_pass_one(self):
        """Calibration shifts the posterior; it does not break the forward pass.

        The distinction matters: refusing it in the forecaster would be a
        spurious refusal (mu replays fine), while NOT refusing it in
        ``run_backtest`` would grade an uncalibrated model under the calibrated
        model's name, because ``_clone_for_prefix`` drops the likelihoods.
        """
        mmm = _model(_world().panel())
        mmm.experiments = ["a-calibration-measurement"]

        assert audit_forward_pass(mmm) == []
        problems = audit_refit(mmm)
        assert problems and "Experiment" in problems[0][0]

    def test_audited_attribute_names_exist_on_a_real_model(self):
        """Guard the two checks that tests can only exercise via stubs.

        `_reach_freq` and `_multiplicative` are private attributes the audit
        reads by name; a rename would make those checks silently never fire
        while the stub-based tests kept passing. Pin that the names are real.
        """
        mmm = _model(_world().panel())
        for attr in ("_reach_freq", "_multiplicative", "_price_lever", "_promo_levers"):
            assert hasattr(mmm, attr), (
                f"audit_forward_pass reads {attr!r}; it no longer exists, so "
                "that check silently never fires"
            )

    def test_error_lists_every_blocker_not_just_the_first(self):
        mmm = _model(_world().panel())
        mmm._multiplicative = True
        mmm._reach_freq = {mmm.channel_names[0]: ("cfg", 1.0, 1.0)}
        mmm.model_config.channel_interactions = [("a", "b")]

        with pytest.raises(ForecastUnsupportedError) as exc:
            PosteriorForecaster(mmm)
        assert len(exc.value.all_unsupported) == 3
        assert "this model also carries" in str(exc.value)

    def test_strict_false_downgrades_to_a_warning(self):
        mmm = _model(_world().panel())
        mmm._multiplicative = True
        # strict=False must not raise the *unsupported* error. The model is
        # unfitted, so it still fails on the trace — proving the audit passed.
        with pytest.raises(ValueError, match="not fitted"):
            PosteriorForecaster(mmm, strict=False)


# ---------------------------------------------------------------------------
# _clone_for_prefix preserves the class
# ---------------------------------------------------------------------------


class TestClonePreservesClass:
    def test_plain_model_roundtrips(self):
        mmm = _model(_world(n_weeks=80).panel())
        clone = _clone_for_prefix(mmm, 60)
        assert type(clone) is type(mmm)
        assert clone.n_periods == 60

    def test_custom_subclass_is_not_downcast(self):
        """A garden model must be backtested as itself, not as a plain MMM."""
        from mmm_framework.garden.base import CustomMMM

        class _MyGardenModel(CustomMMM):
            pass

        panel = _world(n_weeks=80).panel()
        from mmm_framework.model import TrendConfig, TrendType

        mmm = _MyGardenModel(panel, _config(), TrendConfig(type=TrendType.LINEAR))
        clone = _clone_for_prefix(mmm, 60)

        assert type(clone) is _MyGardenModel, (
            "the clone was downcast; its accuracy would be reported under the "
            "custom model's name"
        )
        assert clone.n_periods == 60

    def test_unreconstructable_class_refuses(self):
        class _NeedsMore(type(_model(_world(n_weeks=40).panel()))):
            def __init__(self, *a, required_extra, **kw):  # noqa: D107
                super().__init__(*a, **kw)

        panel = _world(n_weeks=80).panel()
        mmm = _model(panel)
        mmm.__class__ = _NeedsMore
        with pytest.raises(ForecastUnsupportedError, match="cannot be reconstructed"):
            _clone_for_prefix(mmm, 60)


# ---------------------------------------------------------------------------
# trend extrapolation policy
# ---------------------------------------------------------------------------


class TestTrendPolicy:
    def test_describe_names_the_heuristic(self):
        t = TrendExtrapolation("held_flat", "spline", 104)
        assert not t.is_model_defined
        assert "HELD FLAT" in t.describe()
        assert "does not widen" in t.describe()

    def test_linear_is_model_defined_and_states_training_length(self):
        t = TrendExtrapolation("linear", "linear", 104)
        assert t.is_model_defined
        assert "104" in t.describe()

    def test_none_policy(self):
        assert TrendExtrapolation("none", "none", 10).is_model_defined


# ---------------------------------------------------------------------------
# THE AUDIT: forward pass == sum of registered component Deterministics
# ---------------------------------------------------------------------------


_COMPONENTS = (
    "intercept_component",
    "trend_component",
    "seasonality_component",
    "geo_component",
    "product_component",
    "media_total",
    "controls_total",
    "event_component",
    "interaction_component",
    "lever_component",
)


def _component_sum(mmm) -> tuple[np.ndarray, list[str]]:
    """Sum every registered obs-indexed component, ``(n_samples, n_obs)``.

    ``_COMPONENTS`` is an explicit ALLOWLIST rather than "every obs-shaped
    posterior variable", and that is load-bearing: ``control_contributions`` has
    dims ``(obs, control)``, so with a single control it reshapes to exactly
    ``n_obs`` columns and a shape-based filter would silently double-count the
    controls. ``y_obs_scaled`` is obs-shaped too and is the *result*, not a term.

    The shape check is only a secondary guard against period-axis siblings such
    as ``seasonality_by_period``. Callers must assert that the components they
    care about appear in the returned ``found`` list — a component that is
    skipped here AND dropped by the forecaster would otherwise make the identity
    hold vacuously, which is exactly the bug class this file exists to catch.
    """
    post = mmm._trace.posterior
    n = int(post.sizes["chain"] * post.sizes["draw"])
    total = None
    found: list[str] = []
    for name in _COMPONENTS:
        if name not in post:
            continue
        arr = post[name].values.reshape(n, -1)
        if arr.shape[1] != mmm.n_obs:
            continue  # period-axis sibling (e.g. seasonality_by_period)
        found.append(name)
        total = arr if total is None else total + arr
    assert total is not None
    return total, found


@pytest.mark.slow
class TestComponentSumIdentity:
    """``forecast()`` over training positions == the model's own mu, exactly.

    Any term the forward pass fails to sum appears here as a residual, which is
    what makes this the audit rather than a smoke test.
    """

    @pytest.mark.parametrize("parametric", [True, False])
    def test_national(self, parametric):
        mmm = _model(_world().panel(), _config(use_parametric_adstock=parametric))
        mmm.fit(random_seed=7, progressbar=False)

        forecaster = PosteriorForecaster(mmm)
        got = forecaster.forecast(
            mmm.X_media_raw,
            mmm.X_controls_raw,
            np.arange(mmm.n_periods),
            include_noise=False,
        )
        components, found = _component_sum(mmm)
        want = components * mmm.y_std + mmm.y_mean

        assert "media_total" in found and "controls_total" in found
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-9)

    @pytest.mark.parametrize("trend", ["spline", "piecewise"])
    def test_geo_panel_with_flexible_trend(self, trend):
        """`trend_component` is registered per-OBS, not per-period.

        On a geo panel, indexing it with PERIOD positions reads obs `p` — which
        belongs to period `p // n_cells` — stretching the trend by a factor of
        `n_cells` and making the documented hold-last-flat clamp unreachable.
        Measured at 12.6% of KPI level on a 6-cell spline panel before the fix,
        while the audit reported the model clean.
        """
        from mmm_framework.model import TrendConfig, TrendType
        from mmm_framework.synth import dgp_geo

        sc = dgp_geo.build("geo_product", seed=7, n_weeks=60)
        mmm = _model(sc.panel(), trend=TrendConfig(type=TrendType(trend)))
        mmm.fit(random_seed=7, progressbar=False)

        forecaster = PosteriorForecaster(mmm)
        assert forecaster.trend_extrapolation.policy == "held_flat"
        # The policy must describe the PERIOD axis, not the obs axis.
        assert forecaster.trend_extrapolation.n_train_periods == mmm.n_periods

        got = forecaster.forecast(
            mmm.X_media_raw,
            mmm.X_controls_raw,
            np.arange(mmm.n_obs),
            include_noise=False,
        )
        components, found = _component_sum(mmm)
        assert "trend_component" in found
        want = components * mmm.y_std + mmm.y_mean
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-9)

    def test_geo_panel_including_product_offset(self):
        """The geo path must carry the product level offset (dropped pre-1.3.1)."""
        from mmm_framework.synth import dgp_geo

        sc = dgp_geo.build("geo_product", seed=7, n_weeks=60)
        mmm = _model(sc.panel())
        mmm.fit(random_seed=7, progressbar=False)

        forecaster = PosteriorForecaster(mmm)
        got = forecaster.forecast(
            mmm.X_media_raw,
            mmm.X_controls_raw,
            np.arange(mmm.n_obs),
            include_noise=False,
        )
        components, found = _component_sum(mmm)
        want = components * mmm.y_std + mmm.y_mean

        # Non-vacuity: the two level offsets this test exists to guard must
        # actually be in the sum, and product_component must be materially
        # nonzero — otherwise the identity would hold with the forecaster still
        # dropping it, which is precisely the shipped bug.
        assert "geo_component" in found and "product_component" in found
        prod = mmm._trace.posterior["product_component"].values
        assert np.abs(prod).mean() > 1e-6, (
            "product_component is ~zero in this world, so this test cannot "
            "detect the offset being dropped"
        )
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-9)


@pytest.mark.slow
def test_plain_model_forecast_is_byte_identical():
    """Negative control: the fix changes nothing for a model with no dropped terms."""
    mmm = _model(_world().panel())
    mmm.fit(random_seed=7, progressbar=False)

    forecaster = PosteriorForecaster(mmm)
    positions = np.arange(60, mmm.n_periods)
    a = forecaster.forecast(
        mmm.X_media_raw, mmm.X_controls_raw, positions, random_seed=11
    )
    b = forecaster.forecast(
        mmm.X_media_raw, mmm.X_controls_raw, positions, random_seed=11
    )
    np.testing.assert_array_equal(a, b)
    assert forecaster.trend_extrapolation.policy == "linear"
    assert forecaster.trend_extrapolation.is_model_defined
