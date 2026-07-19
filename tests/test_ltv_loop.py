"""Phase 5: LTV into the measurement loop — per-segment CLV (acquisition
channels), the cohort CLV KPI series, CLV−CAC economics, and the clv_value op
feeding value_per_conversion to acquisition experiments."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

from mmm_framework.config import ModelConfig
from mmm_framework.ltv import clv_to_cac, new_customer_clv_series, transactions_to_rfm
from mmm_framework.model import TrendConfig
from mmm_framework.model.trend_config import TrendType
from mmm_framework.synth.dgp_clv import make_clv_world

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../examples/garden_models")
    ),
)
from bayesian_clv import BayesianCLV, rfm_panel, segment_model_params  # noqa: E402

CHANNELS = {
    "Search": {"share": 0.4, "lam_mult": 1.3, "value_mult": 2.0},
    "Social": {"share": 0.6, "lam_mult": 0.8, "value_mult": 0.6},
}


@pytest.fixture(scope="module")
def world():
    return make_clv_world(seed=11, n_customers=1200, channels=CHANNELS)


@pytest.fixture(scope="module")
def rfm(world):
    return transactions_to_rfm(
        world.transactions,
        value_col="value",
        observation_end=world.observation_end,
        segment_col="acquisition_channel",
    )


@pytest.fixture(scope="module")
def fitted(rfm, world):
    mp = {"horizon_periods": 26, **segment_model_params(rfm)}
    mmm = BayesianCLV(
        rfm_panel(rfm),
        ModelConfig(),
        TrendConfig(type=TrendType.NONE),
        model_params=mp,
    )
    mmm.fit(method="map", random_seed=11)
    return mmm


# ── DGP + preprocessing carry-through (fast) ─────────────────────────────────


def test_dgp_plants_channel_heterogeneity(world):
    tx = world.transactions
    assert set(tx["acquisition_channel"].unique()) == {"Search", "Social"}
    # Search customers spend genuinely more per transaction (2x vs 0.6x mult)
    means = tx.groupby("acquisition_channel")["value"].mean()
    assert means["Search"] > 2.0 * means["Social"]
    assert world.truth["channels"] == CHANNELS


def test_rfm_carries_segment(rfm):
    assert "segment" in rfm.columns
    assert set(rfm["segment"].unique()) == {"Search", "Social"}


def test_dgp_without_channels_single_segment():
    w = make_clv_world(seed=3, n_customers=50, calibration_weeks=20)
    assert (w.transactions["acquisition_channel"] == "all").all()


def test_segment_model_params_helper(rfm):
    mp = segment_model_params(rfm)
    assert mp["segment_column"] == "segment_code"
    assert mp["segment_labels"] == ["Search", "Social"]
    assert segment_model_params(rfm.drop(columns=["segment"])) == {}


# ── segment CLV recovery (one MAP fit shared by the module) ──────────────────


@pytest.mark.slow
def test_segment_clv_recovers_planted_ordering(fitted):
    seg = fitted.segment_clv_means()
    assert set(seg) == {"Search", "Social"}
    # planted: Search has 2x value and 1.3x rate vs Social's 0.6x/0.8x —
    # the recovered segment CLV must be decisively higher (>2x)
    assert seg["Search"] > 2.0 * seg["Social"]


@pytest.mark.slow
def test_segment_estimands_surface(fitted):
    res = fitted.evaluate_estimands()
    assert res["segment_clv_Search"].status == "ok"
    assert res["segment_clv_Social"].status == "ok"
    assert res["segment_clv_Search"].mean > res["segment_clv_Social"].mean


@pytest.mark.slow
def test_summary_includes_segment_rows(fitted):
    table = fitted.customer_value_summary()
    quantities = set(table["quantity"])
    assert "Segment CLV — Search" in quantities
    assert "Segment CLV — Social" in quantities


# ── clv_to_cac economics (fast) ──────────────────────────────────────────────


def test_clv_to_cac_ranks_on_net_value():
    # Search: high CLV, high CAC; Social: low CLV, low CAC — Search still wins
    table = clv_to_cac(
        {"Search": 38.0, "Social": 15.0}, {"Search": 25.0, "Social": 12.0}
    )
    assert list(table.index) == ["Search", "Social"]
    assert table.loc["Search", "clv_minus_cac"] == pytest.approx(13.0)
    assert table.loc["Search", "clv_to_cac"] == pytest.approx(38.0 / 25.0)


def test_clv_to_cac_missing_cac_flagged_not_dropped():
    table = clv_to_cac({"A": 10.0, "B": 20.0}, {"A": 5.0})
    assert "B" in table.index
    assert np.isnan(table.loc["B", "cac"])


# ── cohort CLV KPI series (fast) ─────────────────────────────────────────────


def test_new_customer_clv_series_shapes_and_totals(world):
    tx = world.transactions
    acquired = tx.groupby("customer_id")["date"].min()
    clv = pd.Series(10.0, index=acquired.index)  # flat $10/customer
    series = new_customer_clv_series(tx, clv)
    assert series["new_customers"].sum() == len(acquired)
    assert series["cohort_clv"].sum() == pytest.approx(10.0 * len(acquired))
    assert (series["mean_clv"].dropna() <= 10.0 + 1e-9).all()
    assert series["n_unvalued"].sum() == 0


def test_new_customer_clv_series_unvalued_customers_flagged(world):
    tx = world.transactions
    acquired = tx.groupby("customer_id")["date"].min()
    clv = pd.Series(10.0, index=acquired.index[:-5])  # 5 customers unvalued
    series = new_customer_clv_series(tx, clv)
    assert series["n_unvalued"].sum() == 5
    assert series["new_customers"].sum() == len(acquired)  # still counted


# ── clv_value op (fast via stub; slow via real fit) ──────────────────────────


def test_clv_value_op_errors_without_clv_model():
    from mmm_framework.agents.model_ops import clv_value

    class NotCLV:
        _trace = None

    res = clv_value(NotCLV())
    assert res["error"] is not None
    assert "bayesian_clv" in res["error"]


@pytest.mark.slow
def test_clv_value_op_serves_value_per_conversion(fitted):
    import json

    from mmm_framework.agents.model_ops import clv_value

    res = clv_value(fitted)
    assert res["error"] is None
    payload = res["dashboard"]["clv_value"]
    assert payload["value_per_conversion"] == pytest.approx(payload["mean_clv"])
    assert set(payload["segment_clv"]) == {"Search", "Social"}
    json.dumps(res["dashboard"], default=str)

    # segment selection + CAC table
    res2 = clv_value(fitted, segment="Search", cac={"Search": 25.0, "Social": 12.0})
    p2 = res2["dashboard"]["clv_value"]
    assert p2["value_per_conversion"] == pytest.approx(p2["segment_clv"]["Search"])
    assert len(p2["clv_to_cac"]) == 2
    # unknown segment → clean error
    res3 = clv_value(fitted, segment="TV")
    assert res3["error"] is not None
