"""Price & promotion first-class levers — #138.

Promote a control column to a price lever (log-price elasticity, ≤ 0) or a promo
lever (lift with own carryover). Contract:

* off by default — price/promo are ordinary linear controls (no lever RVs, R0.1);
* on ⇒ the named column is REMOVED from the linear control block (no double
  count), the price lever emits a sign-guarded ``price_elasticity``, the promo
  lever a non-negative ``beta_promo_<var>`` with its own adstock;
* the levers are a separate decomposition line (waterfall closes);
* recovers a planted negative elasticity + positive promo lift.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from mmm_framework import BayesianMMM, ModelConfig, ModelConfigBuilder, TrendConfig
from mmm_framework.config import (
    AdstockConfig,
    ControlVariableConfig,
    DimensionType,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
    PriceConfig,
    PromoConfig,
)
from mmm_framework.data_loader import PanelCoordinates, PanelDataset
from mmm_framework.model import TrendType


def _geom(x, a):
    o = np.zeros_like(x, dtype=float)
    acc = 0.0
    for t in range(len(x)):
        acc = x[t] + a * acc
        o[t] = acc
    return o


def _panel(*, with_effect: bool = True, n: int = 130, seed: int = 3) -> PanelDataset:
    rng = np.random.default_rng(seed)
    per = pd.date_range("2022-01-03", periods=n, freq="W-MON")
    chans = ["TV", "Search"]
    X = pd.DataFrame({c: np.abs(rng.normal(100, 30, n)) for c in chans})
    price = 10.0 + rng.normal(0, 1.5, n)
    ref = float(np.median(price))
    promo = (rng.random(n) < 0.25) * rng.uniform(0.2, 0.6, n)
    y = 1000.0 + 2 * X["TV"] + 1.5 * X["Search"] + rng.normal(0, 25, n)
    if with_effect:
        y = (
            y
            - 320.0 * np.log(price / ref)  # negative price elasticity
            + 140.0 * _geom(promo / promo.max(), 0.4)  # promo lift + carryover
        )
    ctrls = pd.DataFrame({"Price": price, "Promo": promo})
    coords = PanelCoordinates(
        periods=per,
        geographies=None,
        products=None,
        channels=chans,
        controls=["Price", "Promo"],
    )
    cfg = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(
                name=c, dimensions=[DimensionType.PERIOD], adstock=AdstockConfig.none()
            )
            for c in chans
        ],
        controls=[
            ControlVariableConfig(name="Price", dimensions=[DimensionType.PERIOD]),
            ControlVariableConfig(name="Promo", dimensions=[DimensionType.PERIOD]),
        ],
    )
    return PanelDataset(
        y=pd.Series(y, name="Sales"),
        X_media=X,
        X_controls=ctrls,
        coords=coords,
        index=per,
        config=cfg,
    )


def _levers():
    return dict(
        price=PriceConfig(variable="Price", reference="median"),
        promotions=[PromoConfig(variable="Promo", adstock_lmax=4)],
    )


def test_off_price_is_a_linear_control():
    m = BayesianMMM(_panel(), ModelConfig(), TrendConfig(type=TrendType.LINEAR))
    named = set(m.model.named_vars)
    assert "price_elasticity" not in named and "lever_component" not in named
    assert m.n_controls == 2  # Price + Promo remain linear controls


def test_on_promotes_levers_and_excludes_from_controls():
    m = BayesianMMM(
        _panel(), ModelConfig(**_levers()), TrendConfig(type=TrendType.LINEAR)
    )
    named = set(m.model.named_vars)
    free = {v.name for v in m.model.free_RVs}
    assert "price_elasticity" in named  # sign-guarded elasticity Deterministic
    assert "price_elasticity_mag" in free
    assert "beta_promo_Promo" in free
    assert "promo_alpha_Promo" in free  # own carryover
    assert "lever_component" in named
    # levers removed from the linear control block (no double-count)
    assert m.control_names == []
    assert m.n_controls == 0


def test_unknown_lever_warns():
    with pytest.warns(UserWarning, match="not a control column"):
        BayesianMMM(
            _panel(),
            ModelConfig(price=PriceConfig(variable="Nope")),
            TrendConfig(type=TrendType.LINEAR),
        )


@pytest.mark.slow
class TestLeverFit:
    def _fit(self):
        cfg = (
            ModelConfigBuilder()
            .map_fit()
            .with_price(_levers()["price"])
            .with_promotions(*_levers()["promotions"])
            .build()
        )
        m = BayesianMMM(_panel(), cfg, TrendConfig(type=TrendType.LINEAR))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(random_seed=0)
        return m

    def test_recovers_negative_elasticity_and_positive_promo(self):
        m = self._fit()
        pe = float(m._trace.posterior["price_elasticity"].mean())
        bp = float(m._trace.posterior["beta_promo_Promo"].mean())
        assert pe < 0  # planted price elasticity is negative
        assert bp > 0  # planted promo lift is positive

    def test_decomposition_has_levers_and_closes(self):
        m = self._fit()
        dec = m.compute_component_decomposition()
        assert dec.total_levers is not None
        total = (
            dec.intercept
            + dec.trend
            + dec.seasonality
            + dec.media_total
            + dec.controls_total
            + dec.levers
        )
        post = m._trace.posterior

        def cm(v):
            return (
                post[v].mean(dim=["chain", "draw"]).values
                if v in post
                else np.zeros(m.n_obs)
            )

        expected = m.y_mean + m.y_std * (
            cm("intercept_component")
            + cm("trend_component")
            + cm("seasonality_component")
            + cm("media_total")
            + cm("controls_total")
            + cm("lever_component")
        )
        np.testing.assert_allclose(total, expected, rtol=1e-8)
        assert "Price & Promotion" in dec.summary()["Component"].tolist()

        from mmm_framework.reporting.extractors.bayesian import BayesianMMMExtractor

        assert "Price & Promotion" in BayesianMMMExtractor(m)._get_component_totals()


# ===========================================================================
# Swappable levers, with both normalization constants frozen at fit time (#222)
#
# Levers shipped as `pt.as_tensor_variable` constants, so there was no container
# to intervene on. Adding one is not enough: the price reference was resolved
# INSIDE the graph builder and the promo scale computed inline from the same
# series, so a uniform swap renormalizes itself away EXACTLY —
#   log(0.5p / mean(0.5p)) == log(p / mean(p))
#   0.5·promo / max|0.5·promo| == promo / max|promo|
# The counterfactual would return "no effect" with full confidence, which is
# indistinguishable from a well-identified null. Both constants are therefore
# frozen in `_prepare_levers` and read as Python floats in-graph.
# ===========================================================================


def _lever_model():
    m = BayesianMMM(
        _panel(), ModelConfig(**_levers()), TrendConfig(type=TrendType.LINEAR)
    )
    _ = m.model  # the graph builds lazily
    return m


def _draw(model, names, seed=7):
    """Paired draw: the same seed gives identical RV values, so any difference
    between two calls is attributable to the data swap alone."""
    with model:
        out = pm.draw([model[n] for n in names], draws=1, random_seed=seed)
    return [np.asarray(o) for o in out]


class TestSwappableLevers:
    def test_container_and_coord_exist_with_frozen_constants(self):
        m = _lever_model()
        assert m.lever_names == ["Price", "Promo"]
        assert "X_levers_raw" in m.model.named_vars
        assert list(m.model.coords["lever"]) == ["Price", "Promo"]
        # Frozen, not recomputed: the reference is the declared statistic of the
        # TRAINING price (the fixture declares "median"), and the promo scale the
        # max of the TRAINING promo.
        price_raw = m.X_levers_raw[:, 0]
        promo_raw = m.X_levers_raw[:, 1]
        assert m._price_ref == pytest.approx(float(np.median(price_raw)))
        assert m._promo_scale["Promo"] == pytest.approx(
            float(np.max(np.abs(promo_raw))) + 1e-9
        )

    def test_in_graph_log_price_matches_numpy(self):
        m = _lever_model()
        comp, el = _draw(m.model, ["price_component", "price_elasticity"])
        expected = float(el) * np.log(
            np.maximum(m.X_levers_raw[:, 0], 1e-9) / m._price_ref
        )
        assert np.allclose(comp, expected, atol=1e-12)

    def test_uniform_price_swap_moves_the_component_by_log_of_the_factor(self):
        """THE regression. Under a recomputed reference this delta is exactly 0
        for every factor — the swap cancels in the ratio."""
        m = _lever_model()
        X0 = m.X_levers_raw.copy()
        base, el = _draw(m.model, ["price_component", "price_elasticity"])
        with m.model:
            pm.set_data({"X_levers_raw": X0 * np.array([0.95, 1.0])})
        swapped, _ = _draw(m.model, ["price_component", "price_elasticity"])

        shift = (swapped - base) / float(el)
        assert np.allclose(shift, np.log(0.95), atol=1e-10)
        # and state what the broken version produced, so a regression is caught
        # rather than merely untested:
        recomputed_ref = float(np.median(X0[:, 0] * 0.95))
        renormalized = np.log(X0[:, 0] * 0.95 / recomputed_ref) - np.log(
            X0[:, 0] / float(np.median(X0[:, 0]))
        )
        assert np.allclose(renormalized, 0.0, atol=1e-12), (
            "a recomputed reference makes this swap an exact no-op — that is "
            "the bug this test exists for"
        )

    def test_uniform_promo_swap_scales_the_component(self):
        m = _lever_model()
        X0 = m.X_levers_raw.copy()
        (base,) = _draw(m.model, ["promo_component"])
        with m.model:
            pm.set_data({"X_levers_raw": X0 * np.array([1.0, 0.5])})
        (swapped,) = _draw(m.model, ["promo_component"])
        assert np.abs(swapped - base).max() > 1e-6  # NOT a no-op
        assert np.allclose(swapped, 0.5 * base, atol=1e-12)

    def test_zero_promo_zeroes_its_component_and_leaves_media_untouched(self):
        m = _lever_model()
        X0 = m.X_levers_raw.copy()
        (media_before,) = _draw(m.model, ["channel_contributions"])
        with m.model:
            pm.set_data({"X_levers_raw": X0 * np.array([1.0, 0.0])})
        promo, media_after = _draw(m.model, ["promo_component", "channel_contributions"])
        assert np.allclose(promo, 0.0, atol=1e-12)
        assert np.array_equal(media_before, media_after)

    def test_the_swap_context_restores_the_training_values(self):
        m = _lever_model()
        X0 = m.X_levers_raw.copy()
        with m._swapped_media_data(None, None, X_levers=X0 * 0.5):
            assert not np.allclose(m.model["X_levers_raw"].get_value(), X0)
        assert np.allclose(m.model["X_levers_raw"].get_value(), X0)

    def test_swap_refuses_a_wrong_shape_and_a_lever_free_model(self):
        m = _lever_model()
        with pytest.raises(ValueError, match=r"must be \(\d+, 2\)"):
            with m._swapped_media_data(None, None, X_levers=np.zeros((m.n_obs, 1))):
                pass
        plain = BayesianMMM(_panel(), ModelConfig(), TrendConfig(type=TrendType.LINEAR))
        _ = plain.model
        with pytest.raises(ValueError, match="no price or promotion lever"):
            with plain._swapped_media_data(None, None, X_levers=np.zeros((plain.n_obs, 1))):
                pass


def test_a_lever_on_a_panel_with_no_controls_warns_rather_than_vanishing():
    """`_prepare_levers` short-circuited on `n_controls == 0` BEFORE the
    per-variable lookup, so a configured lever on a control-free panel was a
    silent no-op that never reached the "not a control column" warning."""
    panel = _panel()
    panel.X_controls = None
    panel.coords.controls = []
    with pytest.warns(UserWarning, match="not a control column"):
        m = BayesianMMM(
            panel,
            ModelConfig(price=PriceConfig(variable="Price")),
            TrendConfig(type=TrendType.LINEAR),
        )
    assert m.n_levers == 0


# ---------------------------------------------------------------------------
# no-lever byte identity
#
# The issue asks for a stored reference and notes none exists. These are it:
# a model with no lever configured must be unchanged by any of the above.
# ---------------------------------------------------------------------------

#: Captured from `main` before the lever container was added.
_NO_LEVER_NAMED_VAR_COUNT = 26
_NO_LEVER_INITIAL_POINT = {
    "beta_Search_log__": 0.4054651081081644,
    "beta_TV_log__": 0.4054651081081644,
    "beta_controls": 0.0,
    "intercept": 0.0,
    "sat_lam_Search_log__": 0.40663425997828095,
    "sat_lam_TV_log__": 0.40663425997828095,
    "season_yearly": 0.0,
    "sigma_log__": -0.6931471805599453,
    "trend_slope": 0.0,
}


def test_no_lever_graph_is_byte_identical():
    m = BayesianMMM(_panel(), ModelConfig(), TrendConfig(type=TrendType.LINEAR))
    mod = m.model
    assert len(mod.named_vars) == _NO_LEVER_NAMED_VAR_COUNT
    assert "lever" not in mod.coords
    assert "X_levers_raw" not in mod.named_vars
    assert m.n_levers == 0 and m.X_levers_raw is None

    point = {k: float(np.asarray(v).sum()) for k, v in mod.initial_point().items()}
    assert point == pytest.approx(_NO_LEVER_INITIAL_POINT)
