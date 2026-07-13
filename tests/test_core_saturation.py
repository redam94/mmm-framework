"""Per-channel saturation in the core ``BayesianMMM``.

The core model historically hard-coded logistic saturation
(``1 - exp(-sat_lam * x)``) for every channel, silently ignoring
``MediaChannelConfig.saturation``. It now honors the configured
:class:`SaturationType` per channel, with the default kept LOGISTIC so default
models are bit-identical to the historical graph (same ``sat_lam_<ch>``
Exponential(0.5) RVs, same formula).

Covers:
* default config -> ``sat_lam_*`` only (names unchanged, graph contract kept),
* per-channel hill -> ``sat_half_*``/``sat_slope_*`` and no ``sat_lam_*``,
* michaelis_menten / tanh -> ``sat_half_*``; none -> no saturation RV,
* the shared pytensor helper matches the closed forms numerically,
* a tiny seeded fit per type runs without NaN,
* counterfactual / marginal / what-if calls work on a hill model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.config import (
    DimensionType,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
    ModelConfig,
    SaturationConfig,
    SaturationType,
)
from mmm_framework.data_loader import PanelCoordinates, PanelDataset
from mmm_framework.model import BayesianMMM, TrendConfig, TrendType
from mmm_framework.model.base import _apply_saturation_pt

CHANNELS = ["TV", "Digital"]


def _panel(saturations: dict[str, SaturationConfig | None]) -> PanelDataset:
    """Tiny 50-week two-channel panel; per-channel saturation configs."""
    rng = np.random.default_rng(0)
    periods = pd.date_range("2022-01-03", periods=50, freq="W-MON")
    n = len(periods)
    coords = PanelCoordinates(
        periods=periods,
        geographies=None,
        products=None,
        channels=CHANNELS,
        controls=[],
    )
    media_channels = []
    for c in CHANNELS:
        kwargs = {"name": c, "dimensions": [DimensionType.PERIOD]}
        if saturations.get(c) is not None:
            kwargs["saturation"] = saturations[c]
        media_channels.append(MediaChannelConfig(**kwargs))
    config = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=media_channels,
        controls=[],
    )
    tv = np.abs(rng.normal(100, 30, n))
    dig = np.abs(rng.normal(80, 20, n))
    # Zero-spend weeks: the hill path must survive x == 0 (gradient guard).
    tv[::7] = 0.0
    y = 1000 + 2.0 * tv + 1.5 * dig + rng.normal(0, 40, n)
    return PanelDataset(
        y=pd.Series(y, name="Sales"),
        X_media=pd.DataFrame({"TV": tv, "Digital": dig}),
        X_controls=None,
        coords=coords,
        index=periods,
        config=config,
    )


def _mmm(
    saturations: dict[str, SaturationConfig | None],
    parametric_adstock: bool = False,
) -> BayesianMMM:
    cfg = ModelConfig(use_parametric_adstock=parametric_adstock)
    return BayesianMMM(_panel(saturations), cfg, TrendConfig(type=TrendType.NONE))


def _free_rv_names(m: BayesianMMM) -> set[str]:
    return {v.name for v in m.model.free_RVs}


# ---------------------------------------------------------------------------
# (a) default = logistic, names unchanged (bit-identity contract)
# ---------------------------------------------------------------------------


class TestDefaultIsLogistic:
    def test_default_media_config_is_logistic(self):
        assert MediaChannelConfig(name="TV").saturation.type == (
            SaturationType.LOGISTIC
        )
        # No lam prior configured -> the model uses the historical
        # Exponential(lam=0.5) builtin.
        assert MediaChannelConfig(name="TV").saturation.lam_prior is None

    @pytest.mark.parametrize("parametric", [False, True])
    def test_default_builds_sat_lam_only(self, parametric):
        m = _mmm({}, parametric_adstock=parametric)
        free = _free_rv_names(m)
        for c in CHANNELS:
            assert f"sat_lam_{c}" in free
            assert f"sat_half_{c}" not in free
            assert f"sat_slope_{c}" not in free

    def test_default_sat_lam_is_exponential_half(self):
        """The default RV keeps the historical Exponential(lam=0.5) prior."""
        m = _mmm({})
        rv = m.model["sat_lam_TV"]
        assert rv.owner.op.name == "exponential"
        # Exponential rate parameter == 0.5 (PyMC stores the scale 1/lam).
        scale = rv.owner.inputs[-1].eval()
        assert float(scale) == pytest.approx(2.0)

    def test_default_named_vars_match_historical_set(self):
        """Graph contract: a default model has exactly the historical nodes."""
        m = _mmm({})
        expected_free = {
            "intercept",
            "sigma",
            "season_yearly",
            *{f"adstock_{c}" for c in CHANNELS},
            *{f"sat_lam_{c}" for c in CHANNELS},
            *{f"beta_{c}" for c in CHANNELS},
        }
        assert _free_rv_names(m) == expected_free


# ---------------------------------------------------------------------------
# (b) per-channel dispatch creates the right RVs
# ---------------------------------------------------------------------------


class TestPerChannelDispatch:
    @pytest.mark.parametrize("parametric", [False, True])
    def test_hill_channel_gets_half_and_slope(self, parametric):
        m = _mmm({"TV": SaturationConfig.hill()}, parametric_adstock=parametric)
        free = _free_rv_names(m)
        assert "sat_half_TV" in free
        assert "sat_slope_TV" in free
        assert "sat_lam_TV" not in free
        # Other channel keeps the logistic default.
        assert "sat_lam_Digital" in free
        assert "sat_half_Digital" not in free

    def test_michaelis_menten_channel(self):
        m = _mmm({"TV": SaturationConfig.michaelis_menten()})
        free = _free_rv_names(m)
        assert "sat_half_TV" in free
        assert "sat_slope_TV" not in free
        assert "sat_lam_TV" not in free

    def test_tanh_channel(self):
        m = _mmm({"TV": SaturationConfig.tanh()})
        free = _free_rv_names(m)
        assert "sat_half_TV" in free
        assert "sat_slope_TV" not in free
        assert "sat_lam_TV" not in free

    def test_root_channel_gets_exponent_only(self):
        m = _mmm({"TV": SaturationConfig.root()})
        free = _free_rv_names(m)
        assert "sat_exponent_TV" in free
        assert "sat_half_TV" not in free
        assert "sat_lam_TV" not in free
        # Untouched channel keeps the default logistic RV.
        assert "sat_lam_Digital" in free
        assert "sat_exponent_Digital" not in free

    def test_none_channel_has_no_saturation_rv(self):
        m = _mmm({"TV": SaturationConfig.none()})
        free = _free_rv_names(m)
        assert "sat_lam_TV" not in free
        assert "sat_half_TV" not in free
        assert "sat_slope_TV" not in free
        # Untouched channel still saturates.
        assert "sat_lam_Digital" in free


# ---------------------------------------------------------------------------
# (d) the shared helper matches the closed forms
# ---------------------------------------------------------------------------


class TestApplySaturationHelper:
    X = np.array([0.0, 0.05, 0.2, 0.5, 0.8, 1.0, 1.3])

    def _eval(self, kind, params):
        import pytensor.tensor as pt

        x = pt.vector("x")
        out = _apply_saturation_pt(x, kind, params)
        return out.eval({x: self.X})

    def test_logistic_matches_closed_form(self):
        lam = 0.7
        got = self._eval(SaturationType.LOGISTIC, {"sat_lam": lam})
        want = 1 - np.exp(np.clip(-lam * self.X, -20, 0))
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_hill_matches_closed_form(self):
        k, s = 0.45, 3.0
        got = self._eval(SaturationType.HILL, {"sat_half": k, "sat_slope": s})
        xc = np.clip(self.X, 1e-9, None)
        want = xc**s / (xc**s + k**s)
        np.testing.assert_allclose(got, want, rtol=1e-10)
        # Exact-ish zero at zero spend and monotone increasing.
        assert got[0] < 1e-20
        assert np.all(np.diff(got) > 0)

    def test_michaelis_menten_matches_closed_form(self):
        k = 0.3
        got = self._eval(SaturationType.MICHAELIS_MENTEN, {"sat_half": k})
        np.testing.assert_allclose(got, self.X / (self.X + k), rtol=1e-12)

    def test_tanh_matches_closed_form(self):
        k = 0.6
        got = self._eval(SaturationType.TANH, {"sat_half": k})
        np.testing.assert_allclose(got, np.tanh(self.X / k), rtol=1e-12)

    def test_root_matches_closed_form(self):
        k = 0.5
        got = self._eval(SaturationType.ROOT, {"sat_exponent": k})
        want = np.clip(self.X, 1e-9, None) ** k
        np.testing.assert_allclose(got, want, rtol=1e-10)
        # Near-zero at zero spend and monotone increasing. (Concavity is
        # asserted on a uniform grid in tests/test_transforms.py.)
        assert got[0] < 1e-4
        assert np.all(np.diff(got) > 0)

    def test_none_is_identity(self):
        got = self._eval(SaturationType.NONE, {})
        np.testing.assert_allclose(got, self.X, rtol=0)


class TestReportingRootHelper:
    """The report/marginal-ROI numpy helpers understand the root form."""

    def test_apply_and_derivative_match_closed_form(self):
        from mmm_framework.reporting.helpers.saturation import (
            _apply_saturation,
            _apply_saturation_derivative,
        )

        k = np.array([0.5])  # per-posterior-sample exponent
        params = {"type": "root", "exponent": k}
        x = 0.4
        np.testing.assert_allclose(_apply_saturation(x, params), x**0.5, rtol=1e-10)
        # f'(x) = k·x^(k-1)
        np.testing.assert_allclose(
            _apply_saturation_derivative(x, params),
            0.5 * x ** (0.5 - 1.0),
            rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# (c) tiny seeded fits run clean for every type
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestTinyFits:
    @pytest.mark.parametrize(
        "sat",
        [
            None,  # default logistic
            SaturationConfig.hill(),
            SaturationConfig.michaelis_menten(),
            SaturationConfig.tanh(),
            SaturationConfig.root(),
            SaturationConfig.none(),
        ],
        ids=["logistic", "hill", "michaelis_menten", "tanh", "root", "none"],
    )
    def test_fit_no_nan(self, sat):
        m = _mmm({c: sat for c in CHANNELS}, parametric_adstock=True)
        results = m.fit(draws=50, tune=50, chains=1, cores=1, random_seed=0)
        post = results.trace.posterior
        for name in post.data_vars:
            assert np.isfinite(post[name].values).all(), f"NaN in {name}"

    def test_counterfactual_and_marginal_on_hill_model(self):
        m = _mmm(
            {c: SaturationConfig.hill() for c in CHANNELS}, parametric_adstock=True
        )
        m.fit(draws=50, tune=50, chains=1, cores=1, random_seed=0)

        contrib = m.compute_counterfactual_contributions(random_seed=1)
        assert set(contrib.total_contributions.index) == set(CHANNELS)
        assert np.isfinite(contrib.total_contributions.to_numpy()).all()

        marg = m.compute_marginal_contributions(spend_increase_pct=10.0, random_seed=1)
        assert len(marg) == len(CHANNELS)
        assert np.isfinite(marg["Marginal ROAS"].to_numpy()).all()

        scen = m.what_if_scenario({"TV": 1.2}, random_seed=1)
        assert np.isfinite(scen["outcome_change"])
