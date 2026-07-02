"""Experiment-registry readouts -> summary observations (the model-free bridge).

A team with historical lift tests and **no MMM and no panel** can still fit the
continuous-learning surface: this module converts the experiment lifecycle
registry's rows (dicts from ``sessions.list_experiments``) into the summary
observations :func:`mmm_framework.continuous_learning.model.fit` consumes
(``lift ~ Normal(scale * (R(test) - R(base)), se)``). It is the model-free
counterpart of the MMM's ``apply_experiment_calibration`` (which folds the same
readouts into the PyMC graph); the two share the structural-stationarity caveat
— the response curve is assumed stable between the test window and now.

Mapping rules (pinned in ``technical-docs/continuous-learning-wiring.md`` §2.3):

* Only ``status in {"completed", "calibrated"}`` rows with a ``value`` + ``se``
  and a channel in the program's channel list convert (exact-name match first,
  case-insensitive as a fallback); everything else lands in ``skipped`` with a
  reason.
* The signed spend delta per period per treated unit comes from, in order:
  ``readout.spend_per_period`` (already per treated unit) ->
  ``design.weekly_spend_delta`` (the treated-cell TOTAL across all treated
  geos, stored ``abs()`` — divided by ``n_units`` and the sign restored from
  the design's holdout marker) -> skipped.
* Lifts are SIGNED: a holdout (negative spend delta) that measured a working
  channel produces a NEGATIVE lift (cutting spend lost KPI), matching the
  model likelihood ``lift ~ N(scale * (R(test) - R(base)), se)`` where
  ``R(test) < R(base)`` for a holdout. ``roas`` readouts multiply up to a
  total lift (``value * total_spend_delta``, signed); ``contribution``
  readouts pass through with the sign flipped for holdouts; ``mroas``
  readouts are slopes, not lifts — skipped.
* Arms-aware matching: a readout with a ``subchannel`` targets the arm
  ``f"{channel}{ARM_SEP}{subchannel}"``; a channel-level readout on a *split*
  parent is skipped (a total-lift constraint across an arm group is future
  work).

Pure numpy + stdlib — no pandas, no sqlite; callers own the registry access.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np

from .arms import ARM_SEP

_CONVERTIBLE_STATUSES = ("completed", "calibrated")


def _is_holdout(design: dict[str, Any]) -> bool:
    """Whether the design cut spend (negative delta) rather than raising it.

    The pinned rule checks ``design.design_type == "holdout"`` /
    ``design.design == "holdout"``; real design snapshots from
    ``planning/design.py`` encode it as ``design_type = f"{method} — holdout"``
    and ``intensity_pct = -100``, so those markers are honored too.
    """
    dd = str(design.get("design") or "").strip().lower()
    dt = str(design.get("design_type") or "").strip().lower()
    dk = str(design.get("design_key") or "").strip().lower()
    if dd == "holdout" or dt == "holdout":
        return True
    if "holdout" in dt or "holdout" in dk:
        return True
    intensity = design.get("intensity_pct")
    if intensity is not None:
        try:
            return float(intensity) < 0
        except (TypeError, ValueError):
            return False
    return False


def _resolve_n_periods(
    exp: dict[str, Any], design: dict[str, Any], period_days: float
) -> int | None:
    """Test length in periods: the date span at ``period_days`` cadence, else
    ``design.duration``, else ``None`` (caller skips)."""
    start, end = exp.get("start_date"), exp.get("end_date")
    if start and end:
        try:
            d0 = date.fromisoformat(str(start)[:10])
            d1 = date.fromisoformat(str(end)[:10])
        except ValueError:
            d0 = d1 = None
        if d0 is not None and d1 is not None and d1 >= d0:
            return max(1, int(round((d1 - d0).days / float(period_days))))
    duration = design.get("duration")
    if duration:
        try:
            return max(1, int(duration))
        except (TypeError, ValueError):
            return None
    return None


def experiments_to_summaries(
    experiments: list[dict[str, Any]],
    *,
    channels: list[str],
    spend_ref: np.ndarray,
    center_scaled: np.ndarray,
    period_days: float = 7.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert registry experiment dicts into summary observations.

    Args:
        experiments: rows from ``sessions.list_experiments`` (plain dicts with
            ``status``/``channel``/``estimand``/``value``/``se`` plus the
            ``design`` and ``readout`` JSON snapshots).
        channels: the program's channel (or flattened arm) names, length ``K``.
        spend_ref: dollars per scaled unit per channel, shape ``(K,)``.
        center_scaled: the program's baseline allocation in scaled units,
            shape ``(K,)`` — the counterfactual every readout is measured
            against.
        period_days: registry-date cadence (7 = weekly).

    Returns:
        ``(summaries, skipped)`` — ``summaries`` are ready for
        ``fit(data={"summaries": ...})`` (each also carries ``experiment_id`` /
        ``channel`` provenance); ``skipped`` items are ``{"id", "reason"}``.
    """
    k = len(channels)
    ref = np.asarray(spend_ref, dtype=float)
    center = np.asarray(center_scaled, dtype=float)
    if ref.shape != (k,) or np.any(ref <= 0) or not np.all(np.isfinite(ref)):
        raise ValueError(f"spend_ref must be positive with shape ({k},)")
    if center.shape != (k,):
        raise ValueError(f"center_scaled must have shape ({k},), got {center.shape}")

    by_exact = {c: i for i, c in enumerate(channels)}
    by_lower: dict[str, list[int]] = {}
    for i, c in enumerate(channels):
        by_lower.setdefault(c.lower(), []).append(i)
    split_parents = {c.split(ARM_SEP, 1)[0].lower() for c in channels if ARM_SEP in c}

    def match_channel(name: str) -> tuple[int | None, str | None]:
        """Resolve a readout name to a channel index: ``(index, skip_reason)``.

        Exact match wins; case-insensitive matching is only a fallback. When
        two program channels collide case-insensitively and the name matches
        neither exactly, the readout is skipped with a clear reason rather
        than silently attributed to the last colliding channel.
        """
        idx = by_exact.get(name)
        if idx is not None:
            return idx, None
        candidates = by_lower.get(name.lower())
        if not candidates:
            return None, None
        if len(candidates) > 1:
            collide = [channels[i] for i in candidates]
            return None, (
                f"channel {name!r} matches multiple program channels "
                f"case-insensitively ({collide}) and none exactly"
            )
        return candidates[0], None

    summaries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    def skip(exp: dict[str, Any], reason: str) -> None:
        skipped.append({"id": exp.get("id"), "reason": reason})

    for exp in experiments:
        status = str(exp.get("status") or "").lower()
        if status not in _CONVERTIBLE_STATUSES:
            skip(exp, f"status {status!r} is not completed/calibrated")
            continue
        readout = exp.get("readout") or {}
        design = exp.get("design") or {}

        value = readout.get("value", exp.get("value"))
        se = readout.get("se", exp.get("se"))
        if value is None or se is None:
            skip(exp, "missing value/se readout")
            continue
        value, se = float(value), float(se)
        if not (np.isfinite(value) and np.isfinite(se)) or se <= 0:
            skip(exp, f"se must be positive and finite, got {se}")
            continue

        channel = str(exp.get("channel") or "")
        subchannel = exp.get("subchannel") or readout.get("subchannel")
        if subchannel:
            arm = f"{channel}{ARM_SEP}{subchannel}"
            c, ambiguous = match_channel(arm)
            if c is None:
                skip(exp, ambiguous or f"arm {arm!r} not in program channels")
                continue
        else:
            c, ambiguous = match_channel(channel)
            if c is None:
                if ambiguous:
                    skip(exp, ambiguous)
                elif channel.lower() in split_parents:
                    skip(exp, "channel-level readout on a split parent")
                else:
                    skip(exp, f"channel {channel!r} not in program channels")
                continue

        estimand = str(readout.get("estimand") or exp.get("estimand") or "roas").lower()
        if estimand == "mroas":
            skip(exp, "mroas readouts are slopes, not lifts")
            continue
        if estimand not in ("roas", "contribution"):
            skip(exp, f"unsupported estimand {estimand!r}")
            continue

        n_units = int(
            readout.get("n_treated_units")
            or len(design.get("treatment_geos") or [])
            or 1
        )

        spend_pp = readout.get("spend_per_period")
        if spend_pp is not None:
            # already signed by the caller AND per treated unit
            spend_delta = float(spend_pp)
        else:
            wsd = design.get("weekly_spend_delta")
            if wsd is None:
                skip(
                    exp,
                    "no spend level: readout.spend_per_period or "
                    "design.weekly_spend_delta required",
                )
                continue
            # design.weekly_spend_delta is the treated-cell TOTAL across all
            # treated geos (planning/design.py sums over treatment_geos) and
            # is stored abs(); divide down to per treated unit and restore
            # the sign from the design's holdout marker.
            spend_delta = (
                abs(float(wsd))
                / max(n_units, 1)
                * (-1.0 if _is_holdout(design) else 1.0)
            )

        n_periods = _resolve_n_periods(exp, design, period_days)
        if n_periods is None:
            skip(exp, "no test window: start/end dates or design.duration required")
            continue

        total_spend_delta = spend_delta * n_units * n_periods
        if estimand == "roas":
            if abs(total_spend_delta) <= 0:
                skip(exp, "roas readout with a zero total spend delta")
                continue
            # SIGNED: a holdout's total spend delta is negative, so a
            # positive measured iROAS becomes a NEGATIVE lift (cutting
            # spend lost KPI) — matching R(test) < R(base) in the model.
            lift = value * total_spend_delta
            se_lift = se * abs(total_spend_delta)
        else:  # contribution: already a total KPI lift; flip for holdouts
            lift = value * (-1.0 if spend_delta < 0 else 1.0)
            se_lift = se

        spend_test = center.copy()
        spend_test[c] = max(0.0, center[c] + spend_delta / ref[c])
        summaries.append(
            {
                "spend_test": spend_test,
                "spend_base": center.copy(),
                "lift": float(lift),
                "se": float(se_lift),
                "scale": float(n_units * n_periods),
                "experiment_id": exp.get("id"),
                "channel": channels[c],
            }
        )
    return summaries, skipped
