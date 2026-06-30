"""Evidence-coded channel classification for the Augur "Media Performance
Readout".

Every channel is sorted into one of four *evidence tiers* — the language the
Augur report uses to color confidence, not just the point estimate:

* **scale**  — the entire credible interval clears break-even (safe to lean in)
* **test**   — high central return but a wide interval straddling break-even
               (worth money, once a test confirms it)
* **hold**   — near break-even, no clear case to move it
* **reduce** — the entire interval sits below break-even (redirect the spend)

The classifier is pure (numpy-only) and duck-types the bundle: it reads
``channel_roi`` (mean/lower/upper per channel), ``channel_spend``,
``channel_contribution``, ``current_spend`` and (optionally) ``estimands`` for a
per-channel marginal ROAS. It is the single source of truth shared by
``reporting.insights`` (narrative) and ``reporting.augur_sections`` (layout).
"""

from __future__ import annotations

from typing import Any

import numpy as np

TIERS: tuple[str, ...] = ("scale", "test", "hold", "reduce")

# Per-tier display metadata. ``css`` is the suffix used by the Augur CSS classes
# (t-scale / t-test / t-hold / t-reduce) and ``color`` is the chart hue.
TIER_META: dict[str, dict[str, str]] = {
    "scale": {
        "action": "Scale",
        "increase": "Increase",
        "read": "Confidently profitable",
        "css": "t-scale",
        "color": "#5a7a3a",  # sage
    },
    "test": {
        "action": "Test",
        "increase": "Hold &amp; learn",
        "read": "High upside, unproven",
        "css": "t-test",
        "color": "#b8860b",  # gold
    },
    "hold": {
        "action": "Hold",
        "increase": "Maintain",
        "read": "Near break-even",
        "css": "t-hold",
        "color": "#4a6d8a",  # steel
    },
    "reduce": {
        "action": "Reduce",
        "increase": "Decrease",
        "read": "Below break-even",
        "css": "t-reduce",
        "color": "#a04535",  # rust
    },
}

# Sort priority: scale first (lean in), then test, hold, reduce.
_TIER_RANK = {t: i for i, t in enumerate(TIERS)}


def _roi_triple(roi: Any) -> tuple[float, float, float] | None:
    """Coerce a channel ROI record to ``(mean, lower, upper)`` finite floats."""
    if not isinstance(roi, dict):
        return None
    try:
        mean = float(roi["mean"])
        lower = float(roi.get("lower", mean))
        upper = float(roi.get("upper", mean))
    except (KeyError, TypeError, ValueError):
        return None
    if not all(np.isfinite(v) for v in (mean, lower, upper)):
        return None
    if lower > upper:  # defensive: keep [lower, upper] ordered
        lower, upper = upper, lower
    return mean, lower, upper


def classify_tier(
    mean: float,
    lower: float,
    upper: float,
    break_even: float = 1.0,
) -> str:
    """Assign an evidence tier from a ROI point estimate + credible interval.

    The whole interval clearing break-even is *scale*; the whole interval below
    is *reduce*; a high central estimate with a straddling interval is *test*;
    everything else (near break-even, low central) is *hold*.
    """
    if lower >= break_even:
        return "scale"
    if upper <= break_even:
        return "reduce"
    if mean >= break_even:
        return "test"
    return "hold"


def channel_marginal_roas(bundle: Any, channel: str) -> float | None:
    """Best-effort per-channel marginal ROAS from the bundle's estimands.

    Looks for a ``marginal_roas`` estimand keyed per channel
    (``marginal_roas:<channel>``) or a bare ``marginal_roas`` whose label names
    the channel. Returns ``None`` when unavailable — the classifier never
    requires it.
    """
    estimands = getattr(bundle, "estimands", None)
    if not isinstance(estimands, dict):
        return None
    for key in (f"marginal_roas:{channel}", f"marginal_roas:{channel.lower()}"):
        rec = estimands.get(key)
        if isinstance(rec, dict) and rec.get("status", "ok") == "ok":
            try:
                v = float(rec["mean"])
                if np.isfinite(v):
                    return v
            except (KeyError, TypeError, ValueError):
                pass
    # Bare estimand whose label references the channel.
    rec = estimands.get("marginal_roas")
    if isinstance(rec, dict):
        label = str(rec.get("label", "")).lower()
        if channel.lower() in label:
            try:
                v = float(rec["mean"])
                if np.isfinite(v):
                    return v
            except (KeyError, TypeError, ValueError):
                pass
    return None


def channel_rows(bundle: Any, break_even: float = 1.0) -> list[dict[str, Any]]:
    """Per-channel evidence rows, ordered by recommended priority.

    Each row carries everything the scorecard, ROI forest, reallocation cards
    and deep dives need: ``name``, ROI ``mean/lower/upper``, ``spend``,
    ``contribution`` (mean), ``mroas``, ``tier`` and its display metadata,
    ``spend_share``. Channels without a usable ROI record are skipped.
    """
    roi_map = getattr(bundle, "channel_roi", None) or {}
    spend_map = getattr(bundle, "channel_spend", None) or {}
    contrib_map = getattr(bundle, "channel_contribution", None) or {}
    current_spend = getattr(bundle, "current_spend", None) or {}

    def _spend_of(name: str) -> float | None:
        raw = spend_map.get(name, current_spend.get(name))
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    # Denominator must use the SAME channel_spend→current_spend fallback as the
    # per-channel numerator, and span exactly the channels that get a row — else
    # spend shares would not sum to 1 when a channel is only in current_spend.
    total_spend = 0.0
    for name in roi_map:
        if _roi_triple(roi_map[name]) is None:
            continue
        s = _spend_of(name)
        if s is not None and s > 0:
            total_spend += s

    rows: list[dict[str, Any]] = []
    for name, roi in roi_map.items():
        triple = _roi_triple(roi)
        if triple is None:
            continue
        mean, lower, upper = triple

        spend = _spend_of(name)

        contrib = contrib_map.get(name)
        contrib_mean = None
        contrib_ci: tuple[float, float] | None = None
        if isinstance(contrib, dict):
            try:
                contrib_mean = float(contrib["mean"])
                contrib_ci = (
                    float(contrib.get("lower", contrib_mean)),
                    float(contrib.get("upper", contrib_mean)),
                )
            except (KeyError, TypeError, ValueError):
                contrib_mean, contrib_ci = None, None
        elif contrib is not None:
            try:
                contrib_mean = float(contrib)
            except (TypeError, ValueError):
                contrib_mean = None

        mroas = channel_marginal_roas(bundle, name)
        tier = classify_tier(mean, lower, upper, break_even)
        meta = TIER_META[tier]

        spend_share = (
            spend / total_spend if (spend is not None and total_spend > 0) else None
        )

        rows.append(
            {
                "name": name,
                "roi": mean,
                "roi_lower": lower,
                "roi_upper": upper,
                "spend": spend,
                "contribution": contrib_mean,
                "contribution_ci": contrib_ci,
                "mroas": mroas,
                "tier": tier,
                "action": meta["action"],
                "read": meta["read"],
                "css": meta["css"],
                "color": meta["color"],
                "spend_share": spend_share,
            }
        )

    rows.sort(key=lambda r: (_TIER_RANK[r["tier"]], -r["roi"]))
    return rows


def reallocation_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group classified rows by tier (preserving priority order within each)."""
    groups: dict[str, list[dict[str, Any]]] = {t: [] for t in TIERS}
    for r in rows:
        groups[r["tier"]].append(r)
    return groups


def test_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Channels whose return is most worth resolving with an experiment.

    Ordered by the *width* of the credible interval (relative to the mean) — the
    widest, most decision-blocking uncertainty first. ``test`` and ``reduce``
    (spend-down) channels are the natural candidates.
    """

    def _rel_width(r: dict[str, Any]) -> float:
        spread = r["roi_upper"] - r["roi_lower"]
        return spread / max(abs(r["roi"]), 0.01)

    cands = [r for r in rows if r["tier"] in ("test", "reduce")]
    cands.sort(key=_rel_width, reverse=True)
    return cands


def illustrative_flighting_totals(
    rows: list[dict[str, Any]],
    n_weeks: int = 52,
    seed: int = 20260629,
) -> dict[str, list[float]] | None:
    """Deterministic *illustrative* weekly current-vs-recommended portfolio spend.

    The MMM bundle carries period-totals, not a weekly media calendar, so this
    derives a plausible weekly shape from each channel's annual spend and tier:
    ``reduce`` channels are trimmed and their freed budget is redeployed into
    ``scale`` channels, with the recommended line smoothed (continuity earns
    more than spikes). Clearly an illustration — the section tags it as such.

    Returns ``{"weeks", "current_total", "recommended_total"}`` or ``None`` when
    no channel has a usable spend level.
    """
    spendable = [r for r in rows if r.get("spend") and r["spend"] > 0]
    if not spendable:
        return None

    rng = np.random.default_rng(seed)
    weeks = list(range(1, n_weeks + 1))
    w = np.arange(n_weeks)

    def _weekly_weights(tier: str) -> np.ndarray:
        """A normalized weekly shape per tier (bursty for reduce, ramped for scale)."""
        noise = 0.85 + 0.3 * rng.random(n_weeks)
        if tier == "reduce":
            base = np.where((w % 12) < 3, 1.5, 0.15)  # heavy bursts + dark
        elif tier == "scale":
            base = 0.45 + 1.05 * (w / n_weeks)  # ramps up over the year
        elif tier == "test":
            base = np.where((w % 4) == 0, 1.0, 0.2)  # pulsed
        else:  # hold
            base = 1.0 + 0.18 * np.sin(w / 5.0)  # steady, gentle wave
        weights = np.clip(base * noise, 0.0, None)
        s = weights.sum()
        return weights / s if s > 0 else np.ones(n_weeks) / n_weeks

    def _smooth(arr: np.ndarray, win: int = 3) -> np.ndarray:
        kernel = np.ones(2 * win + 1)
        padded = np.pad(arr, win, mode="edge")
        return np.convolve(padded, kernel, mode="valid") / kernel.sum()

    # Freed budget from trimming reduce channels is redeployed into "recipient"
    # channels: scale if any, else test/hold. If there is nowhere to redeploy
    # (e.g. an all-reduce portfolio) we DON'T trim — so the recommended plan
    # always sums to the same annual budget as the current plan.
    recipient_tiers = (
        ("scale",) if any(r["tier"] == "scale" for r in spendable) else ("test", "hold")
    )
    recipient_spend_total = sum(
        r["spend"] for r in spendable if r["tier"] in recipient_tiers
    )
    trim_reduce = recipient_spend_total > 0

    current = np.zeros(n_weeks)
    recommended = np.zeros(n_weeks)
    freed = 0.0

    # Pass 1: current = each channel's spend on its tier's shape; bank freed $.
    per_channel_rec: list[tuple[dict[str, Any], np.ndarray, float]] = []
    for r in spendable:
        shape = _weekly_weights(r["tier"])
        current += shape * r["spend"]
        if r["tier"] == "reduce" and trim_reduce:
            keep = 0.70  # trim a reduce channel ~30%
            freed += r["spend"] * (1 - keep)
            rec_spend = r["spend"] * keep
            rec_shape = _smooth(shape)  # smooth the bursts
        else:
            rec_spend = r["spend"]
            rec_shape = shape
        per_channel_rec.append((r, rec_shape, rec_spend))

    # Pass 2: redeploy freed budget into recipient channels (∝ spend).
    for r, rec_shape, rec_spend in per_channel_rec:
        bonus = 0.0
        if r["tier"] in recipient_tiers and recipient_spend_total > 0:
            bonus = freed * (r["spend"] / recipient_spend_total)
        total = rec_spend + bonus
        s = rec_shape.sum()
        norm = rec_shape / s if s > 0 else np.ones(n_weeks) / n_weeks
        recommended += norm * total

    return {
        "weeks": weeks,
        "current_total": [float(v) for v in current],
        "recommended_total": [float(v) for v in recommended],
    }


__all__ = [
    "TIERS",
    "TIER_META",
    "classify_tier",
    "channel_marginal_roas",
    "channel_rows",
    "reallocation_groups",
    "test_candidates",
    "illustrative_flighting_totals",
]
