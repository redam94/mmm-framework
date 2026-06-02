"""Tests for prior-vs-posterior learning diagnostics.

The core :func:`parameter_learning` is tested on synthetic prior/posterior samples
with *known* answers (no model fit), which is where the metric semantics live. A
slow integration test exercises the model methods end-to-end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.diagnostics import parameter_learning


def _normal(rng, mean, sd, n=8000):
    return rng.normal(mean, sd, n)


class TestParameterLearningCore:
    def test_strong_learning_contracts_and_shifts(self):
        rng = np.random.default_rng(0)
        prior = {"theta": _normal(rng, 0.0, 1.0)}
        posterior = {"theta": _normal(rng, 2.0, 0.2)}
        df = parameter_learning(prior, posterior)
        row = df.set_index("parameter").loc["theta"]
        # Var ratio 0.04 -> contraction ~ 0.96.
        assert row["contraction"] == pytest.approx(0.96, abs=0.03)
        # Mean moved ~2 prior sds.
        assert row["shift_z"] == pytest.approx(2.0, abs=0.1)
        # Narrow + shifted -> negligible overlap, strong verdict.
        assert row["overlap"] < 0.2
        assert row["verdict"] == "strong"

    def test_prior_dominated_when_posterior_equals_prior(self):
        # The identity the user asked for: posterior == prior => nothing learned.
        rng = np.random.default_rng(1)
        s = _normal(rng, -0.3, 0.5)
        df = parameter_learning({"psi": s.copy()}, {"psi": s.copy()})
        row = df.set_index("parameter").loc["psi"]
        assert row["contraction"] == pytest.approx(0.0, abs=1e-9)
        assert row["overlap"] == pytest.approx(1.0, abs=1e-9)
        assert row["verdict"] == "prior-dominated"

    def test_sign_constrained_prior_not_learned(self):
        # Mimics psi = -HalfNormal: a one-sided prior. If the posterior just restates
        # it, "entirely below zero" is vacuous and the verdict must be prior-dominated.
        rng = np.random.default_rng(2)
        prior = {"psi": -np.abs(_normal(rng, 0.0, 0.3))}
        posterior = {"psi": -np.abs(_normal(rng, 0.0, 0.3))}  # same law, no learning
        df = parameter_learning(prior, posterior)
        row = df.set_index("parameter").loc["psi"]
        assert row["post_mean"] < 0  # trivially true under the prior
        assert row["contraction"] == pytest.approx(0.0, abs=0.05)
        assert row["overlap"] > 0.85
        assert row["verdict"] == "prior-dominated"

    def test_sign_constrained_prior_when_learned(self):
        # Same one-sided prior, but the data pins psi tightly to a specific value.
        rng = np.random.default_rng(3)
        prior = {"psi": -np.abs(_normal(rng, 0.0, 0.3))}
        posterior = {"psi": _normal(rng, -0.25, 0.02)}  # narrow, located
        df = parameter_learning(prior, posterior)
        row = df.set_index("parameter").loc["psi"]
        assert row["contraction"] > 0.9
        assert row["verdict"] == "strong"

    def test_widening_gives_negative_contraction_not_clipped(self):
        rng = np.random.default_rng(4)
        prior = {"theta": _normal(rng, 0.0, 1.0)}
        posterior = {"theta": _normal(rng, 0.0, 2.0)}  # posterior WIDER than prior
        df = parameter_learning(prior, posterior)
        c = df.set_index("parameter").loc["theta", "contraction"]
        assert c < 0  # 1 - 4 = -3-ish; a real prior-data-conflict warning, not clipped
        assert c == pytest.approx(-3.0, abs=0.3)

    def test_multidim_flattening_and_alignment(self):
        rng = np.random.default_rng(5)
        # shape (n_samples, 2, 2); element [1,0] strongly learned, others prior-like.
        prior = rng.normal(0, 1, (4000, 2, 2))
        post = prior.copy()
        post[:, 1, 0] = rng.normal(3.0, 0.1, 4000)  # only this element moves/narrows
        df = parameter_learning({"M": prior}, {"M": post}).set_index("parameter")
        assert set(df.index) == {"M[0,0]", "M[0,1]", "M[1,0]", "M[1,1]"}
        assert df.loc["M[1,0]", "verdict"] == "strong"
        assert df.loc["M[0,0]", "verdict"] == "prior-dominated"

    def test_var_names_filters_by_base_or_element(self):
        rng = np.random.default_rng(6)
        prior = {"beta": rng.normal(0, 1, (2000, 3)), "sigma": _normal(rng, 1, 0.5)}
        post = {"beta": rng.normal(1, 0.3, (2000, 3)), "sigma": _normal(rng, 1, 0.4)}
        # Base name keeps every element of beta and drops sigma.
        df = parameter_learning(prior, post, var_names=["beta"])
        assert set(df["parameter"]) == {"beta[0]", "beta[1]", "beta[2]"}
        # Exact element name also works.
        df2 = parameter_learning(prior, post, var_names=["beta[1]", "sigma"])
        assert set(df2["parameter"]) == {"beta[1]", "sigma"}

    def test_sorted_by_contraction_and_columns(self):
        rng = np.random.default_rng(7)
        prior = {"a": _normal(rng, 0, 1), "b": _normal(rng, 0, 1)}
        post = {"a": _normal(rng, 0, 1), "b": _normal(rng, 5, 0.1)}  # b learned, a not
        df = parameter_learning(prior, post)
        assert list(df.columns) == [
            "parameter", "prior_mean", "prior_sd", "post_mean", "post_sd",
            "contraction", "overlap", "shift_z", "verdict",
        ]
        # Ascending contraction -> the un-learned 'a' sorts first.
        assert df.iloc[0]["parameter"] == "a"
        assert (df["overlap"].between(0, 1) | df["overlap"].isna()).all()

    def test_degenerate_prior_is_undetermined(self):
        df = parameter_learning({"c": np.full(500, 2.0)}, {"c": np.full(500, 2.0)})
        assert df.set_index("parameter").loc["c", "verdict"] == "undetermined"

    def test_only_shared_parameters_compared(self):
        rng = np.random.default_rng(8)
        df = parameter_learning(
            {"a": _normal(rng, 0, 1), "x": _normal(rng, 0, 1)},
            {"a": _normal(rng, 0, 1), "y": _normal(rng, 0, 1)},
        )
        assert set(df["parameter"]) == {"a"}


@pytest.mark.slow
class TestParameterLearningIntegration:
    """End-to-end through the model method (requires a small fit)."""

    def _panel(self):
        from mmm_framework.config import (
            ControlVariableConfig, DimensionType, KPIConfig, MediaChannelConfig,
            MFFConfig,
        )
        from mmm_framework.data_loader import PanelCoordinates, PanelDataset

        rng = np.random.default_rng(0)
        periods = pd.date_range("2022-01-03", periods=80, freq="W-MON")
        n = len(periods)
        tv = np.abs(rng.normal(100, 30, n))
        digital = np.abs(rng.normal(80, 20, n))
        price = 10 + rng.normal(0, 0.5, n)
        # TV genuinely drives sales; Digital is near-noise -> different learning.
        y = 1000 + 4.0 * tv + 25 * (price - price.mean()) + rng.normal(0, 60, n)
        coords = PanelCoordinates(
            periods=periods, geographies=None, products=None,
            channels=["TV", "Digital"], controls=["Price"],
        )
        config = MFFConfig(
            kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
            media_channels=[
                MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
                MediaChannelConfig(name="Digital", dimensions=[DimensionType.PERIOD]),
            ],
            controls=[ControlVariableConfig(name="Price", dimensions=[DimensionType.PERIOD])],
        )
        return PanelDataset(
            y=pd.Series(y, name="Sales"),
            X_media=pd.DataFrame({"TV": tv, "Digital": digital}),
            X_controls=pd.DataFrame({"Price": price}),
            coords=coords, index=periods, config=config,
        )

    def test_bayesian_mmm_parameter_learning(self):
        from mmm_framework import BayesianMMM, ModelConfigBuilder, TrendConfig, TrendType

        mmm = BayesianMMM(
            self._panel(),
            ModelConfigBuilder().bayesian_pymc().build(),
            TrendConfig(type=TrendType.LINEAR),
        )
        mmm.fit(draws=200, tune=200, chains=2, cores=1, random_seed=0)
        df = mmm.compute_parameter_learning(prior_samples=800)
        assert not df.empty
        assert {"contraction", "overlap", "shift_z", "verdict"} <= set(df.columns)
        # Every free RV is represented (beta_TV etc.).
        assert any(p.startswith("beta_TV") for p in df["parameter"])
        # Contraction is a finite ratio (may be negative); overlap in [0, 1].
        assert df["contraction"].notna().any()
        assert (df["overlap"].dropna().between(0, 1)).all()

    def test_requires_fit(self):
        from mmm_framework import BayesianMMM, ModelConfigBuilder, TrendConfig, TrendType

        mmm = BayesianMMM(
            self._panel(),
            ModelConfigBuilder().bayesian_pymc().build(),
            TrendConfig(type=TrendType.LINEAR),
        )
        with pytest.raises(ValueError, match="not fitted"):
            mmm.compute_parameter_learning()
