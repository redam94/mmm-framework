"""Short-term vs long-term contribution split (issue #106).

A weekly MMM with adstock captures **activation** (the response the same week
spend lands) and **carryover** (the adstock tail — persistence over the next few
weeks). It does **not** capture true **long-term brand equity**: the multi-
quarter / multi-year lift that brand-building creates through mental
availability, pricing power, and base-demand growth, which decays far more slowly
than any adstock window. Left unaddressed this is the single most consequential
way an MMM misleads a CMO — it systematically **under-credits brand** and
over-rotates budget to performance channels.

This module makes the honest split explicit:

* :func:`carryover_split` — decomposes each channel's measured effect into the
  fraction that lands **immediately** (week 0) vs the fraction that **carries
  over** within the adstock window. This is genuinely estimable from the fitted
  adstock; it is a *within-window* split, NOT brand equity.
* :func:`long_term_scenario` — applies an **external, clearly-labelled
  assumption** (a long-term multiplier, e.g. from published brand meta-analyses)
  to show how the picture would shift IF long-term effects add to the measured
  short-term. This is a scenario, never a model output.
* :func:`build_long_term_facts` — assembles the report payload: the per-channel
  split, whether a structural brand funnel is present, and the multiplier
  scenario when configured.

The load-bearing honesty is in the report copy: when only the within-window
carryover is available, the section says plainly that long-term brand is **not
measured here** and lists the data required to measure it.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "carryover_split",
    "long_term_scenario",
    "build_long_term_facts",
    "DEFAULT_LONG_TERM_MULTIPLIER",
]

#: A conservative default long-term multiplier for the *scenario* view when the
#: caller asks for one without a value. Published brand meta-analyses (e.g. Les
#: Binet & Peter Field; Nielsen/Analytic Partners long-term studies) commonly
#: find total effects roughly 1.5–2× the short-term measured effect for
#: brand-heavy channels — 2.0 is a round, widely-cited midpoint. It is an
#: ASSUMPTION the user should replace with their own evidence.
DEFAULT_LONG_TERM_MULTIPLIER = 2.0


def _weights(arr: Any) -> np.ndarray | None:
    try:
        w = np.asarray(arr, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    w = w[np.isfinite(w)]
    return w if w.size else None


def carryover_split(adstock_weights: Any) -> dict[str, float] | None:
    """Immediate vs carryover fractions from a channel's adstock decay weights.

    ``adstock_weights`` is the per-lag weight vector (need not be normalized).
    Returns ``{"immediate_pct", "carryover_pct", "effective_weeks"}`` where the
    percentages sum to 1 and ``effective_weeks`` is the number of lags carrying
    ≥1% of the weight (a plain "how many weeks does it keep working" read).
    ``None`` when the weights are unusable.
    """
    w = _weights(adstock_weights)
    if w is None or w.sum() <= 0:
        return None
    w = w / w.sum()
    immediate = float(w[0])
    carryover = float(1.0 - immediate)
    effective_weeks = int(np.sum(w >= 0.01))
    return {
        "immediate_pct": immediate,
        "carryover_pct": max(0.0, carryover),
        "effective_weeks": effective_weeks,
    }


def long_term_scenario(
    short_term: float, multiplier: float = DEFAULT_LONG_TERM_MULTIPLIER
) -> dict[str, float]:
    """Apply a long-term multiplier to a measured short-term contribution/ROI.

    Returns ``{"multiplier", "long_term", "uplift"}``. This is an **assumption-
    driven scenario** (``long_term = short_term * multiplier``), not a model
    estimate — the report labels it as such.
    """
    m = float(multiplier)
    lt = float(short_term) * m
    return {"multiplier": m, "long_term": lt, "uplift": lt - float(short_term)}


def build_long_term_facts(
    channels: Sequence[str],
    adstock_curves: Mapping[str, Any] | None,
    *,
    contribution: Mapping[str, float] | None = None,
    half_lives: Mapping[str, float] | None = None,
    multiplier: float | None = None,
    has_structural_funnel: bool = False,
) -> dict[str, Any] | None:
    """Assemble the ``bundle.long_term`` payload.

    Parameters
    ----------
    channels:
        Channel names (drives ordering).
    adstock_curves:
        ``{channel: lag_weights}`` (``bundle.adstock_curves``). Drives the
        immediate-vs-carryover split. When absent, the payload still returns with
        an empty per-channel list so the section renders its caveat.
    contribution:
        Optional ``{channel: total_contribution}`` — when present, the split is
        also expressed in KPI units.
    half_lives:
        Optional ``{channel: half_life_weeks}`` for display.
    multiplier:
        Optional long-term multiplier → adds an assumption-driven scenario per
        channel (and blended). ``None`` = caveat-only (no scenario).
    has_structural_funnel:
        Whether the model exposes a survey/brand funnel (a longer-horizon brand
        path); drives the section copy.
    """
    adstock_curves = adstock_curves or {}
    rows: list[dict[str, Any]] = []
    for ch in channels:
        split = carryover_split(adstock_curves.get(ch))
        if split is None:
            continue
        row: dict[str, Any] = {"channel": str(ch), **split}
        if half_lives and ch in half_lives and np.isfinite(half_lives[ch]):
            row["half_life"] = float(half_lives[ch])
        contrib = None
        if contribution and ch in contribution:
            try:
                contrib = float(contribution[ch])
            except (TypeError, ValueError):
                contrib = None
        if contrib is not None and np.isfinite(contrib):
            row["contribution"] = contrib
            row["immediate_contribution"] = contrib * split["immediate_pct"]
            row["carryover_contribution"] = contrib * split["carryover_pct"]
            if multiplier is not None:
                sc = long_term_scenario(contrib, multiplier)
                row["scenario_long_term"] = sc["long_term"]
                row["scenario_uplift"] = sc["uplift"]
        rows.append(row)

    if not rows and not has_structural_funnel:
        # Nothing estimable AND no brand funnel — still return so the section can
        # render the "only short-term is measured" caveat.
        return {
            "channels": [],
            "has_structural_funnel": False,
            "multiplier": multiplier,
            "measured": "none",
        }

    facts: dict[str, Any] = {
        "channels": rows,
        "has_structural_funnel": bool(has_structural_funnel),
        "multiplier": multiplier,
        "measured": "carryover" if rows else "funnel",
    }
    # Blended (contribution-weighted) split, when contributions are available.
    contribs = [r.get("contribution") for r in rows if r.get("contribution")]
    if contribs:
        total = float(sum(contribs))
        imm = float(sum(r.get("immediate_contribution", 0.0) for r in rows))
        car = float(sum(r.get("carryover_contribution", 0.0) for r in rows))
        if total > 0:
            facts["blended"] = {
                "immediate_pct": imm / total,
                "carryover_pct": car / total,
                "total_contribution": total,
            }
            if multiplier is not None:
                sc = long_term_scenario(total, multiplier)
                facts["blended"]["scenario_long_term"] = sc["long_term"]
                facts["blended"]["scenario_uplift"] = sc["uplift"]
    return facts
