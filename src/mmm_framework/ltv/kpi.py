"""LTV → measurement-loop bridges (Phase 5).

Three ways lifetime value enters the loop:

* :func:`new_customer_clv_series` — a **cohort CLV KPI**: weekly acquired-
  customer lifetime value (new customers per period × their posterior CLV).
  Feed it to an MMM / experiment as a monetary KPI (``margin_per_kpi = 1``) so
  media is valued on the LIFETIME value of the customers it acquires, not the
  first purchase.
* :func:`clv_to_cac` — per-segment ``CLV / CAC`` and ``CLV − CAC`` given
  per-segment acquisition costs: the blended acquisition-economics view (a
  channel acquiring low-frequency one-time buyers is correctly discounted vs
  one acquiring high-CLV customers even at equal CPA).
* the ghost-ads / net-economics hookup — a fitted :class:`BayesianCLV`'s
  ``mean_clv`` (or a segment's CLV) IS the ``value_per_conversion`` /
  ``margin_per_kpi`` for an acquisition experiment; the ``clv_value`` model op
  serves it to the calculators.

pandas/numpy only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def new_customer_clv_series(
    transactions: pd.DataFrame,
    clv_per_customer: pd.Series,
    *,
    customer_col: str = "customer_id",
    date_col: str = "date",
    freq: str = "W-MON",
) -> pd.DataFrame:
    """Weekly (or ``freq``-period) cohort CLV: for each period, the number of
    NEWLY ACQUIRED customers and the sum of their (posterior-mean) CLV.

    ``clv_per_customer`` is indexed by customer id (e.g. the posterior mean of
    the fitted model's per-customer ``clv`` deterministic). Customers in the
    log but missing from the CLV index contribute count but zero value (and are
    counted in ``n_unvalued`` for honesty).

    Returns a DataFrame indexed by period start with columns
    ``new_customers``, ``cohort_clv``, ``mean_clv``, ``n_unvalued``.
    """
    work = transactions[[customer_col, date_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col])
    acquired = work.groupby(customer_col)[date_col].min()

    clv = pd.Series(clv_per_customer).astype(float)
    aligned = clv.reindex(acquired.index)
    period = acquired.dt.to_period(freq.split("-")[0]).dt.start_time

    frame = pd.DataFrame(
        {
            "period": period,
            "clv": aligned.fillna(0.0),
            "valued": aligned.notna(),
        }
    )
    out = frame.groupby("period").agg(
        new_customers=("clv", "size"),
        cohort_clv=("clv", "sum"),
        n_unvalued=("valued", lambda s: int((~s).sum())),
    )
    out["mean_clv"] = np.where(
        out["new_customers"] > 0, out["cohort_clv"] / out["new_customers"], 0.0
    )
    out.index.name = "period"
    return out


def clv_to_cac(
    segment_clv: dict[str, float],
    cac: dict[str, float],
) -> pd.DataFrame:
    """Blended acquisition economics per segment/channel.

    Args:
        segment_clv: posterior mean CLV per acquired customer, by segment
            (e.g. from the fitted model's ``segment_clv_<name>`` deterministics).
        cac: customer-acquisition cost per segment (spend / new customers —
            supplied by the caller; the CLV model has no spend data).

    Returns:
        DataFrame with ``clv``, ``cac``, ``clv_minus_cac`` and ``clv_to_cac``
        per segment, sorted by ``clv_minus_cac`` descending. Segments missing a
        CAC get NaN economics (still listed, flagged).
    """
    rows = []
    for seg, clv in segment_clv.items():
        c = cac.get(seg)
        rows.append(
            {
                "segment": seg,
                "clv": float(clv),
                "cac": float(c) if c is not None else float("nan"),
                "clv_minus_cac": float(clv - c) if c is not None else float("nan"),
                "clv_to_cac": (
                    float(clv / c) if c is not None and c > 0 else float("nan")
                ),
            }
        )
    out = pd.DataFrame(rows).set_index("segment")
    return out.sort_values("clv_minus_cac", ascending=False)


__all__ = ["new_customer_clv_series", "clv_to_cac"]
