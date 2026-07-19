"""Synthetic customer-transaction worlds with KNOWN BG/NBD + Gamma-Gamma truth.

Simulates the exact generative story the model assumes (so recovery is a clean
test of the implementation, not of model misspecification):

* purchase rate    ``lam_i ~ Gamma(r, rate=alpha)``,
* dropout          ``p_i ~ Beta(a, b)`` — a death coin flipped after every
  REPEAT purchase (NOT after the acquisition purchase: the likelihood's
  survival term is ``E[(1-p)^x]`` over the ``x`` repeats, so an extra
  acquisition-time flip would over-produce zero-repeat customers and bias
  ``r``/``alpha`` low — verified empirically),
* transaction value ``z ~ Gamma(p_gg, rate=nu_i)`` with ``nu_i ~ Gamma(q_gg,
  rate=gamma_gg)`` (per-customer spend scale, independent of frequency).

Acquisitions are staggered across the first half of the calibration window so
customer ages ``T`` vary. Transactions after ``observation_end`` (up to
``holdout_weeks`` later) land in ``holdout_transactions`` — the classic
calibration/holdout split for validating predicted purchases.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CLVScenario:
    """Transaction log + answer key."""

    transactions: pd.DataFrame  # customer_id, date, value (calibration window)
    holdout_transactions: pd.DataFrame  # same schema, after observation_end
    observation_end: pd.Timestamp
    truth: dict = field(default_factory=dict)


def make_clv_world(
    seed: int = 7,
    *,
    n_customers: int = 2000,
    calibration_weeks: int = 52,
    holdout_weeks: int = 26,
    r: float = 0.8,
    alpha: float = 6.0,
    a: float = 1.3,
    b: float = 3.0,
    p_gg: float = 6.0,
    q_gg: float = 4.0,
    gamma_gg: float = 15.0,
    channels: dict[str, dict] | None = None,
    start: str = "2024-01-01",
) -> CLVScenario:
    """Simulate a transaction log with known population parameters.

    Defaults keep ``a`` comfortably away from the ``a = 1`` numerical pole in
    the conditional-expectation formula and ``q_gg > 1`` so the Gamma-Gamma
    population mean exists. Expected behavior at the defaults: mean purchase
    rate ``r/alpha ≈ 0.13``/week, mean dropout ``a/(a+b) ≈ 0.30`` per purchase,
    mean transaction value ``p·gamma/(q-1) = 30``.

    ``channels`` plants ACQUISITION-CHANNEL heterogeneity (the Phase-5 segment
    truth): ``{name: {"share": w, "lam_mult": m1, "value_mult": m2}}`` — each
    customer is acquired by one channel; its purchase rate is scaled by
    ``lam_mult`` and its transaction values by ``value_mult``, so segment CLV
    genuinely differs by a KNOWN ordering. ``None`` → one ``"all"`` channel,
    byte-identical population behavior. NB per-channel multipliers make the
    POOLED population deviate from a single BG/NBD — segment recovery tests
    should assert the ORDERING/ratio of segment CLV, not exact pooled params.
    """
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    obs_end = start_ts + pd.Timedelta(weeks=calibration_weeks)
    total_weeks = calibration_weeks + holdout_weeks

    lam = rng.gamma(shape=r, scale=1.0 / alpha, size=n_customers)
    p_die = rng.beta(a, b, size=n_customers)
    nu = rng.gamma(shape=q_gg, scale=1.0 / gamma_gg, size=n_customers)

    if channels:
        names = list(channels)
        shares = np.array([channels[c].get("share", 1.0) for c in names], dtype=float)
        shares = shares / shares.sum()
        ch_idx = rng.choice(len(names), size=n_customers, p=shares)
        lam = lam * np.array([channels[names[i]].get("lam_mult", 1.0) for i in ch_idx])
        # value_mult scales E[z] = p/nu, so divide nu by the multiplier
        nu = nu / np.array([channels[names[i]].get("value_mult", 1.0) for i in ch_idx])
        customer_channel = np.array([names[i] for i in ch_idx])
    else:
        customer_channel = np.full(n_customers, "all")

    # staggered acquisition: uniform over the first half of the calibration
    # window, so ages T range over [calibration_weeks/2, calibration_weeks].
    acq_week = rng.uniform(0.0, calibration_weeks / 2.0, size=n_customers)

    rows: list[tuple[int, float, float]] = []  # (customer, week, value)
    for i in range(n_customers):
        t = acq_week[i]
        rows.append((i, t, rng.gamma(p_gg, 1.0 / nu[i])))
        # wait -> repeat purchase -> death coin (no flip after acquisition)
        while True:
            t = t + rng.exponential(1.0 / lam[i]) if lam[i] > 0 else np.inf
            if t > total_weeks:
                break
            rows.append((i, t, rng.gamma(p_gg, 1.0 / nu[i])))
            if rng.random() < p_die[i]:
                break

    df = pd.DataFrame(rows, columns=["customer_id", "week", "value"])
    df["date"] = start_ts + pd.to_timedelta(df["week"] * 7.0, unit="D")
    df["acquisition_channel"] = customer_channel[df["customer_id"].to_numpy()]
    df = df.drop(columns=["week"]).sort_values(["customer_id", "date"])
    calib = df[df["date"] <= obs_end].reset_index(drop=True)
    holdout = df[df["date"] > obs_end].reset_index(drop=True)

    return CLVScenario(
        transactions=calib,
        holdout_transactions=holdout,
        observation_end=obs_end,
        truth={
            "r": r,
            "alpha": alpha,
            "a": a,
            "b": b,
            "p_gg": p_gg,
            "q_gg": q_gg,
            "gamma_gg": gamma_gg,
            "n_customers": n_customers,
            "calibration_weeks": calibration_weeks,
            "holdout_weeks": holdout_weeks,
            "mean_purchase_rate": r / alpha,
            "mean_dropout": a / (a + b),
            "mean_txn_value": p_gg * gamma_gg / (q_gg - 1.0),
            "channels": channels,
            "seed": seed,
        },
    )


__all__ = ["CLVScenario", "make_clv_world"]
