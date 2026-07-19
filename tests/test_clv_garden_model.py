"""End-to-end proof for the LTV non-MMM family — BayesianCLV (BG/NBD +
Gamma-Gamma) on synthetic transactions with KNOWN population parameters:
builds, recovers the planted truth, predicts the holdout, surfaces CLV
estimands, round-trips serialization, and renders the latent-structure table."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from mmm_framework.config import ModelConfig
from mmm_framework.ltv import transactions_to_rfm
from mmm_framework.model import TrendConfig
from mmm_framework.model.trend_config import TrendType
from mmm_framework.synth.dgp_clv import make_clv_world

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../examples/garden_models")
    ),
)
from bayesian_clv import BayesianCLV, CLVConfig, rfm_panel  # noqa: E402


@pytest.fixture(scope="module")
def world():
    return make_clv_world(seed=7, n_customers=1500)


@pytest.fixture(scope="module")
def rfm(world):
    return transactions_to_rfm(
        world.transactions, value_col="value", observation_end=world.observation_end
    )


def _build(rfm, world, **params):
    mp = {"horizon_periods": world.truth["holdout_weeks"], **params}
    return BayesianCLV(
        rfm_panel(rfm),
        ModelConfig(),
        TrendConfig(type=TrendType.NONE),
        model_params=mp,
    )


@pytest.fixture(scope="module")
def fitted(rfm, world):
    mmm = _build(rfm, world)
    mmm.fit(method="map", random_seed=7)
    return mmm


# ── build-time contract (fast) ───────────────────────────────────────────────


def test_clv_is_non_mmm_kind_with_latent_structure():
    from mmm_framework.garden.contract import has_latent_structure, is_mmm_model

    assert BayesianCLV.__garden_model_kind__ == "clv"
    assert not is_mmm_model(BayesianCLV)
    assert has_latent_structure(BayesianCLV)


def test_config_schema_defaults_and_rejects_unknown():
    cfg = CLVConfig()
    assert cfg.horizon_periods == 52
    with pytest.raises(Exception):
        CLVConfig(unknown_knob=1)


def test_graph_builds_with_finite_logp(rfm, world):
    mmm = _build(rfm, world)
    model = mmm.model
    assert np.isfinite(
        float(
            model.point_logps().to_pandas().sum()
            if hasattr(model.point_logps(), "to_pandas")
            else sum(model.point_logps().values())
        )
    )


def test_rfm_validation_rejects_bad_rows(rfm, world):
    bad = rfm.copy()
    bad.loc[bad.index[0], "T"] = 0.0
    with pytest.raises(ValueError, match="T>0"):
        _build(bad, world)  # _prepare_data validates at construction


def test_monetary_off_falls_back_to_purchases(rfm, world):
    mmm = _build(rfm, world, monetary_model=False)
    model = mmm.model
    names = [v.name for v in model.free_RVs]
    assert "p_gg" not in names and "q_raw" not in names


# ── recovery + prediction (one MAP fit shared by the module) ─────────────────


@pytest.mark.slow
def test_map_recovers_planted_population_parameters(fitted, world):
    post = fitted._trace.posterior
    truth = world.truth
    for name, key, rel in (
        ("r", "r", 0.35),
        ("alpha", "alpha", 0.35),
        ("a", "a", 0.5),
        ("b", "b", 0.5),
        ("p_gg", "p_gg", 0.35),
        ("q_gg", "q_gg", 0.5),
        ("gamma_gg", "gamma_gg", 0.5),
    ):
        est = float(post[name].mean().values)
        assert est == pytest.approx(truth[key], rel=rel), name


@pytest.mark.slow
def test_predicted_holdout_purchases_track_actual(fitted, world, rfm):
    """The operational validation: predicted expected purchases over the holdout
    horizon vs the customers' ACTUAL holdout transactions."""
    post = fitted._trace.posterior
    pred = post["expected_purchases"].mean(("chain", "draw")).values
    actual = (
        world.holdout_transactions.groupby("customer_id")
        .size()
        .reindex(rfm.index, fill_value=0)
        .to_numpy()
    )
    # aggregate total within 30%, per-customer rank correlation clearly positive
    assert pred.sum() == pytest.approx(actual.sum(), rel=0.30)
    from scipy.stats import spearmanr

    rho = spearmanr(pred, actual).statistic
    assert rho > 0.3


@pytest.mark.slow
def test_estimands_and_summary(fitted):
    res = fitted.evaluate_estimands()
    keys = set(res)
    assert {"mean_clv", "total_clv", "mean_expected_purchases", "mean_p_alive"} <= keys
    for k in ("mean_clv", "total_clv", "mean_p_alive"):
        assert res[k].status == "ok"
        assert np.isfinite(res[k].mean)
    assert 0.0 <= res["mean_p_alive"].mean <= 1.0

    table = fitted.customer_value_summary()
    assert len(table) >= 6
    assert (table["mean"].iloc[:4] > 0).all()


@pytest.mark.slow
def test_serialization_roundtrip(fitted, rfm, world, tmp_path):
    from mmm_framework.serialization import MMMSerializer

    path = str(tmp_path / "clv_model")
    MMMSerializer.save(fitted, path)
    loaded = MMMSerializer.load(path, rfm_panel(rfm))
    assert type(loaded).__name__ == "BayesianCLV"
    post = loaded._trace.posterior
    np.testing.assert_allclose(
        float(post["mean_clv"].mean().values),
        float(fitted._trace.posterior["mean_clv"].mean().values),
        rtol=1e-6,
    )
    # summaries still work on the reloaded model
    assert len(loaded.customer_value_summary()) >= 6
