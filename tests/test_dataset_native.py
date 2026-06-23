"""Tests for Phase A3: native (non-MFF) Dataset loading + the TRIALS likelihood
hook. A wide, role-tagged table loads into a Dataset directly — so a genuinely
non-MMM family (CFA / LCA indicators, a survey) brings its own data shape without
riding an MMM panel.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

from mmm_framework import Dataset
from mmm_framework.config import DatasetRole, DatasetSchema, ModelConfig
from mmm_framework.config.dataset import RoleBinding
from mmm_framework.dataset_loader import load_dataset

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../examples/garden_models")
    ),
)
from bayesian_cfa import BayesianCFA, CFAConfig  # noqa: E402


def _indicator_frame(n: int = 160, seed: int = 0) -> pd.DataFrame:
    """A wide 2-factor × 3-indicator table with a planted structure."""
    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    cols = {}
    for i in range(3):
        cols[f"x{i + 1}"] = 0.8 * f1 + 0.3 * rng.normal(size=n)
    for i in range(3):
        cols[f"x{i + 4}"] = 0.8 * f2 + 0.3 * rng.normal(size=n)
    return pd.DataFrame(cols)


def _indicator_schema(df: pd.DataFrame) -> DatasetSchema:
    return DatasetSchema(
        bindings=[RoleBinding(name=c, role=DatasetRole.INDICATOR) for c in df.columns]
    )


# --------------------------------------------------------------------------- #
# from_wide / load_dataset
# --------------------------------------------------------------------------- #
class TestNativeLoad:
    def test_from_wide_tags_roles_and_coords(self):
        df = _indicator_frame()
        ds = Dataset.from_wide(df, _indicator_schema(df))
        assert isinstance(ds, Dataset)
        assert ds.columns_for(DatasetRole.INDICATOR) == list(df.columns)
        assert list(ds.observed().columns) == list(df.columns)
        # Cross-sectional: synthetic time axis of length n, no geo/product.
        assert ds.coords.n_periods == len(df)
        assert not ds.coords.has_geo and not ds.coords.has_product
        assert ds.n_channels == 0  # no predictors

    def test_load_dataset_from_dataframe_and_csv(self, tmp_path):
        df = _indicator_frame(n=40)
        schema = _indicator_schema(df)
        ds_mem = load_dataset(df, schema)
        path = tmp_path / "indicators.csv"
        df.to_csv(path, index=False)
        ds_csv = load_dataset(str(path), schema)
        assert np.allclose(
            ds_mem.observed().to_numpy(np.float64),
            ds_csv.observed().to_numpy(np.float64),
        )

    def test_from_wide_rejects_unknown_columns(self):
        df = _indicator_frame(n=20)
        bad = DatasetSchema(
            bindings=[RoleBinding(name="nope", role=DatasetRole.INDICATOR)]
        )
        with pytest.raises(ValueError, match="not present"):
            Dataset.from_wide(df, bad)

    def test_trials_helper(self):
        n = 30
        df = pd.DataFrame({"aware": np.arange(n), "n": np.full(n, 500.0)})
        schema = DatasetSchema(
            bindings=[
                RoleBinding(name="aware", role=DatasetRole.TARGET),
                RoleBinding(name="n", role=DatasetRole.TRIALS),
            ]
        )
        ds = load_dataset(df, schema)
        trials = ds.trials()
        assert trials is not None and trials.shape == (n,)
        assert np.allclose(trials, 500.0)
        # No TRIALS column → None.
        assert _indicator_schema(_indicator_frame(n=5))
        assert (
            load_dataset(
                _indicator_frame(n=5), _indicator_schema(_indicator_frame(n=5))
            ).trials()
            is None
        )


# --------------------------------------------------------------------------- #
# A non-MMM model fits on natively-loaded data (no MMM panel)
# --------------------------------------------------------------------------- #
class TestNativeFit:
    def test_cfa_fits_on_native_indicators(self):
        df = _indicator_frame()
        ds = load_dataset(df, _indicator_schema(df))
        model = BayesianCFA(
            ds,
            ModelConfig(),
            model_params=CFAConfig(n_factors=2, factor_assignment=[0, 0, 0, 1, 1, 1]),
        )
        assert model.indicator_names == list(df.columns)
        assert model.channel_names == []
        results = model.fit(method="map")
        assert results.approximate is True
        # Recovers a positive loading on the first indicator (planted ≈ 0.8+).
        assert float(results.trace.posterior["loading_x1"].mean()) > 0.4


# --------------------------------------------------------------------------- #
# build_model native path
# --------------------------------------------------------------------------- #
class TestBuildModelNative:
    def test_build_model_loads_native_dataset(self, tmp_path):
        from mmm_framework.agents.fitting import build_model

        df = _indicator_frame(n=80)
        path = tmp_path / "ind.csv"
        df.to_csv(path, index=False)
        spec = {
            "dataset": _indicator_schema(df).model_dump(mode="json"),
            "model_params": {"n_factors": 2, "factor_assignment": [0, 0, 0, 1, 1, 1]},
            "inference": {"chains": 2, "draws": 100, "tune": 100},
        }
        mmm = build_model(spec, str(path), model_cls=BayesianCFA)
        # Loaded natively: indicator columns tagged, no channels, no kpi needed.
        assert mmm.indicator_names == list(df.columns)
        assert mmm.channel_names == []
        assert mmm.dataset.columns_for(DatasetRole.INDICATOR) == list(df.columns)


# --------------------------------------------------------------------------- #
# TRIALS role → binomial likelihood (per-observation denominator)
# --------------------------------------------------------------------------- #
class TestTrialsLikelihood:
    def test_awareness_reads_per_obs_trials(self):
        from awareness_structural_mmm import AwarenessStructuralMMM
        from mmm_framework.config import LikelihoodConfig
        from mmm_framework.model import TrendConfig
        from mmm_framework.model.trend_config import TrendType

        n = 40
        rng = np.random.default_rng(7)
        tv = np.abs(rng.normal(100, 25, n))
        rate = 1.0 / (1.0 + np.exp(-(0.6 + 1.4 * (tv / tv.max()))))
        # Per-period sample sizes ~1800 (vary by period); aware counts therefore
        # routinely EXCEED the scalar fallback (number_of_trials default 1000), so a
        # binomial with the scalar n would have observed > n → -inf logp. A
        # successful MAP fit proves the per-observation TRIALS column was used.
        trials = rng.integers(1600, 2000, n).astype(float)
        aware = rng.binomial(trials.astype(int), rate).astype(float)
        assert aware.max() > 1000  # exceeds the scalar fallback
        df = pd.DataFrame({"Awareness": aware, "TV": tv, "n": trials})
        schema = DatasetSchema(
            bindings=[
                RoleBinding(name="Awareness", role=DatasetRole.TARGET),
                RoleBinding(name="TV", role=DatasetRole.PREDICTOR),
                RoleBinding(name="n", role=DatasetRole.TRIALS),
            ]
        )
        ds = load_dataset(df, schema)
        assert ds.trials() is not None
        assert not np.allclose(ds.trials(), ds.trials()[0])  # varies per obs

        mmm = AwarenessStructuralMMM(
            ds,
            ModelConfig(likelihood=LikelihoodConfig.binomial(n_trials=1000)),
            TrendConfig(type=TrendType.NONE),
        )
        rv = next(v for v in mmm.model.observed_RVs if v.name == "y_obs")
        assert "Binomial" in type(rv.owner.op).__name__
        # The scalar n=1000 would make some observed counts invalid; a converged
        # MAP fit confirms the per-observation trials column drove the denominator.
        results = mmm.fit(method="map", random_seed=0)
        assert results.approximate is True
        assert "awareness_retention" in results.trace.posterior
