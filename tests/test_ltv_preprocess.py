"""RFM preprocessing (ltv/preprocess.py): hand-computed fixtures + edge cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.ltv import rfm_arrays, transactions_to_rfm


def _log():
    """3 customers: A = 3 purchases (2 repeats), B = one-time, C = 2 purchases
    incl. a same-day duplicate that must merge."""
    rows = [
        ("A", "2024-01-01", 10.0),
        ("A", "2024-01-15", 20.0),
        ("A", "2024-02-12", 30.0),
        ("B", "2024-01-08", 50.0),
        ("C", "2024-01-01", 5.0),
        ("C", "2024-01-01", 7.0),  # same-day duplicate — merges into one event
        ("C", "2024-01-29", 12.0),
    ]
    return pd.DataFrame(rows, columns=["customer_id", "date", "value"])


def test_rfm_hand_computed_weekly():
    rfm = transactions_to_rfm(
        _log(), value_col="value", observation_end="2024-03-04", freq="W"
    )
    a, b, c = rfm.loc["A"], rfm.loc["B"], rfm.loc["C"]
    # A: first 01-01, last 02-12 → recency 6w; T = 9w; 2 repeats
    assert a.frequency == 2
    assert a.recency == pytest.approx(6.0)
    assert a["T"] == pytest.approx(9.0)
    assert a.monetary == pytest.approx((20.0 + 30.0) / 2)  # repeats only
    # B: one-time buyer
    assert b.frequency == 0
    assert b.recency == 0.0
    assert np.isnan(b.monetary)
    # C: same-day duplicate merged (values summed → first event 12), 1 repeat
    assert c.frequency == 1
    assert c.recency == pytest.approx(4.0)
    assert c.monetary == pytest.approx(12.0)


def test_rfm_daily_unit_and_default_end():
    rfm = transactions_to_rfm(_log(), freq="D")  # end defaults to last txn 02-12
    assert rfm.loc["A", "T"] == pytest.approx(42.0)  # 01-01 → 02-12
    assert rfm.loc["A", "recency"] == pytest.approx(42.0)
    assert "monetary" not in rfm.columns  # frequency-only mode


def test_rfm_invariants_and_arrays():
    rfm = transactions_to_rfm(_log(), value_col="value", observation_end="2024-03-04")
    assert (rfm["recency"] <= rfm["T"]).all()
    assert ((rfm["frequency"] == 0) == (rfm["recency"] == 0)).all()
    x, t_x, T, m = rfm_arrays(rfm)
    assert x.shape == t_x.shape == T.shape == m.shape == (3,)
    assert x.dtype == float


def test_rfm_drops_post_window_transactions():
    rfm = transactions_to_rfm(
        _log(), value_col="value", observation_end="2024-01-20", freq="W"
    )
    assert rfm.loc["A", "frequency"] == 1  # 02-12 purchase excluded
    assert rfm.loc["C", "frequency"] == 0  # 01-29 excluded


def test_rfm_validation():
    with pytest.raises(ValueError, match="freq"):
        transactions_to_rfm(_log(), freq="Q")
    with pytest.raises(ValueError, match="no transactions"):
        transactions_to_rfm(_log(), observation_end="2020-01-01")
