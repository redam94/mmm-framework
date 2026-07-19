"""Transaction-log → RFM summaries for the buy-till-you-die LTV models.

The BG/NBD + Gamma-Gamma likelihoods condition on exactly four per-customer
sufficient statistics computed over a calibration window:

* ``frequency`` — number of REPEAT purchases (transactions − 1; the first
  purchase is the acquisition event and carries no information about the
  purchase rate under the model),
* ``recency`` — time of the LAST purchase measured from the FIRST purchase
  (0 for one-time buyers),
* ``T`` — the customer's age: observation end − first purchase,
* ``monetary`` — mean value of the REPEAT transactions (the Gamma-Gamma input;
  NaN for one-time buyers, who carry no monetary signal).

pandas only — no model imports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_FREQ_TO_DAYS = {"D": 1.0, "W": 7.0, "M": 30.4375}


def transactions_to_rfm(
    df: pd.DataFrame,
    *,
    customer_col: str = "customer_id",
    date_col: str = "date",
    value_col: str | None = None,
    observation_end: pd.Timestamp | str | None = None,
    freq: str = "W",
    segment_col: str | None = None,
) -> pd.DataFrame:
    """Collapse a transaction log to one row per customer (RFM summaries).

    Multiple transactions by the same customer on the same date are merged into
    one purchase (values summed) — sub-period repeats are indistinguishable at
    the model's time resolution and would otherwise inflate ``frequency``.

    Args:
        df: long transaction log (one row per transaction).
        value_col: transaction value column; omit for frequency-only (no
            ``monetary`` output column).
        observation_end: end of the calibration window (transactions after it
            are dropped). Defaults to the last transaction date.
        freq: time unit for ``recency``/``T`` — 'D', 'W' (default) or 'M'.
        segment_col: per-customer attribute (e.g. acquisition channel) carried
            through as a ``segment`` column — the FIRST transaction's value per
            customer (the acquisition event's attribution).

    Returns:
        DataFrame indexed by customer with columns ``frequency``, ``recency``,
        ``T``, ``n_txn`` (raw purchase count) and, when ``value_col`` is given,
        ``monetary`` (mean REPEAT-transaction value; NaN for one-time buyers),
        plus ``segment`` when ``segment_col`` is given.
    """
    if freq not in _FREQ_TO_DAYS:
        raise ValueError(f"freq must be one of {sorted(_FREQ_TO_DAYS)}, got {freq!r}")
    unit_days = _FREQ_TO_DAYS[freq]

    keep = [customer_col, date_col] + ([value_col] if value_col else [])
    if segment_col:
        keep.append(segment_col)
    work = df[keep].copy()
    work[date_col] = pd.to_datetime(work[date_col])
    if observation_end is None:
        observation_end = work[date_col].max()
    observation_end = pd.to_datetime(observation_end)
    work = work[work[date_col] <= observation_end]
    if work.empty:
        raise ValueError("no transactions on or before observation_end")

    # merge same-day repeats: one purchase event per (customer, date)
    grouped = (
        work.groupby([customer_col, date_col], as_index=False)[value_col].sum()
        if value_col
        else work.drop_duplicates([customer_col, date_col])
    )

    g = grouped.groupby(customer_col)[date_col]
    first = g.min()
    last = g.max()
    n_txn = g.count()

    # fractional days (total_seconds), NOT .dt.days — day-truncation would
    # zero the recency of close-together repeats and break the x>0 ⇒ t_x>0
    # invariant the likelihood relies on.
    day = 86400.0 * unit_days
    out = pd.DataFrame(
        {
            "frequency": (n_txn - 1).astype(int),
            "recency": (last - first).dt.total_seconds() / day,
            "T": (observation_end - first).dt.total_seconds() / day,
            "n_txn": n_txn.astype(int),
        }
    )

    if value_col:
        # monetary = mean value of REPEAT purchases only (drop the first event)
        grouped = grouped.sort_values([customer_col, date_col])
        grouped["_rank"] = grouped.groupby(customer_col).cumcount()
        repeats = grouped[grouped["_rank"] > 0]
        out["monetary"] = repeats.groupby(customer_col)[value_col].mean()
        # one-time buyers: NaN (no repeat-value signal), preserved explicitly
        out["monetary"] = out["monetary"].astype(float)

    if segment_col:
        # acquisition attribution: the FIRST transaction's segment value
        first_rows = work.sort_values(date_col).groupby(customer_col).first()
        out["segment"] = first_rows[segment_col].astype(str)

    out.index.name = customer_col
    # invariants the likelihood relies on
    assert (out["recency"] <= out["T"] + 1e-9).all()
    assert ((out["frequency"] == 0) == (out["recency"] == 0)).all()
    return out


def rfm_arrays(
    rfm: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """``(frequency, recency, T, monetary-or-None)`` float arrays for the model."""
    x = rfm["frequency"].to_numpy(dtype=float)
    t_x = rfm["recency"].to_numpy(dtype=float)
    T = rfm["T"].to_numpy(dtype=float)
    m = rfm["monetary"].to_numpy(dtype=float) if "monetary" in rfm.columns else None
    return x, t_x, T, m


__all__ = ["transactions_to_rfm", "rfm_arrays"]
