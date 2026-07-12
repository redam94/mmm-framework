"""Triangulation panel — MMM × experiment × platform (issue #104).

The platform's one true causal anchor is experiment calibration, but nothing puts
the three estimates of a channel's effect next to each other: the **MMM** (a
model-identified incremental effect), the **experiment** readout (a directly
measured incremental effect), and the **platform-reported** figure (usually
last-touch / correlational). Convergent evidence across independent methods is
the single most persuasive thing you can show a skeptical CFO; divergence, shown
honestly, is where the real conversation happens — and the most common
divergence (platform >> incremental) is exactly the one a budget owner must
understand before trusting a dashboard number.

This module is the pure reconciliation engine:

* :class:`TriangulationSource` — one measurement of a channel's return, tagged
  with whether it is **incremental** (MMM / experiment) or a correlational
  last-touch figure (platform), plus its attribution window / method note.
* :func:`reconcile_channel` — classifies agreement across a channel's sources,
  picks a reconciled recommendation (the experiment anchors it when present —
  the causal gold standard — else the MMM), and writes plain-language notes
  explaining *why* the sources differ.
* :class:`TriangulationResult` — the per-channel panel; ``to_dict()`` feeds the
  report section, the interactive report, and the agent tool.

The builder that pulls the three sources from a fitted model + the experiment
registry + a platform-figures dict lives in
:func:`triangulation_from_model` so the reconciliation logic stays testable
without a fit.
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "TriangulationSource",
    "ChannelTriangulation",
    "TriangulationResult",
    "reconcile_channel",
    "build_triangulation",
    "triangulation_from_model",
]

#: How the three sources display (label + a stable css class for the chip/marker).
SOURCE_META: dict[str, dict[str, str]] = {
    "experiment": {"label": "Experiment", "css": "tri-experiment"},
    "mmm": {"label": "MMM", "css": "tri-mmm"},
    "platform": {"label": "Platform", "css": "tri-platform"},
}

# Two incremental estimates "agree" when their intervals overlap, or (lacking
# intervals) their points are within this relative tolerance.
_AGREE_TOL = 0.30
# A platform figure this many times the incremental estimate is flagged as
# last-touch inflation.
_INFLATION_RATIO = 1.4


@dataclass
class TriangulationSource:
    """One method's estimate of a channel's return.

    ``incremental`` distinguishes a causal / incremental estimate (MMM,
    experiment) from a correlational last-touch platform figure — the axis that
    makes platform-vs-MMM divergence *expected*, not alarming.
    """

    source: str  # "mmm" | "experiment" | "platform"
    value: float
    lower: float | None = None
    upper: float | None = None
    metric: str = "roi"  # "roi" | "roas" | "mroas" | "lift" | "contribution"
    incremental: bool = True
    attribution_window: str | None = None
    period: str | None = None
    note: str = ""

    @property
    def label(self) -> str:
        return SOURCE_META.get(self.source, {}).get("label", self.source.title())

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "label": self.label,
            "value": _f(self.value),
            "lower": _f(self.lower),
            "upper": _f(self.upper),
            "metric": self.metric,
            "incremental": bool(self.incremental),
            "attribution_window": self.attribution_window,
            "period": self.period,
            "note": self.note,
        }


@dataclass
class ChannelTriangulation:
    """The reconciled picture for one channel across its sources."""

    channel: str
    sources: list[TriangulationSource]
    agreement: str  # "convergent" | "divergent" | "platform-inflated" | "single-source"
    reconciled: dict[str, Any]  # {"value","lower","upper","basis"}
    notes: list[str] = field(default_factory=list)

    @property
    def convergent(self) -> bool:
        return self.agreement == "convergent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "sources": [s.to_dict() for s in self.sources],
            "agreement": self.agreement,
            "reconciled": self.reconciled,
            "notes": list(self.notes),
        }


@dataclass
class TriangulationResult:
    """The full triangulation panel across channels."""

    channels: list[ChannelTriangulation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "channels": [c.to_dict() for c in self.channels],
            "summary": self._summary(),
        }

    def _summary(self) -> dict[str, Any]:
        agg = {
            "convergent": 0,
            "divergent": 0,
            "platform-inflated": 0,
            "single-source": 0,
        }
        for c in self.channels:
            agg[c.agreement] = agg.get(c.agreement, 0) + 1
        return {"n_channels": len(self.channels), "by_agreement": agg}


# =============================================================================
# Reconciliation
# =============================================================================
def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _intervals_overlap(a: TriangulationSource, b: TriangulationSource) -> bool | None:
    """Do two sources' credible/confidence intervals overlap? ``None`` if either
    lacks an interval."""
    if a.lower is None or a.upper is None or b.lower is None or b.upper is None:
        return None
    return not (a.upper < b.lower or b.upper < a.lower)


def _points_close(a: float, b: float, tol: float = _AGREE_TOL) -> bool:
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom <= tol


def _agree(a: TriangulationSource, b: TriangulationSource) -> bool:
    ov = _intervals_overlap(a, b)
    if ov is not None:
        return ov
    return _points_close(a.value, b.value)


def _fmt(v: float | None) -> str:
    return f"{v:.2f}×" if v is not None and np.isfinite(v) else "—"


def reconcile_channel(
    channel: str,
    sources: Sequence[TriangulationSource],
    *,
    tol: float = _AGREE_TOL,
    inflation_ratio: float = _INFLATION_RATIO,
) -> ChannelTriangulation:
    """Classify agreement, pick a reconciled recommendation, and explain the gaps.

    Rules:

    * The **experiment** anchors the reconciled number when present (a direct
      causal measurement); otherwise the **MMM**. The platform figure never
      anchors a recommendation — it is context.
    * Agreement is judged among the **incremental** sources (MMM + experiment):
      "convergent" if they agree, "divergent" if they don't, "single-source" if
      only one exists. A platform figure materially above the incremental
      estimate flags "platform-inflated" (unless the incremental sources
      themselves diverge, which dominates).
    """
    srcs = list(sources)
    by_source = {s.source: s for s in srcs}
    incremental = [s for s in srcs if s.incremental]
    mmm = by_source.get("mmm")
    exp = by_source.get("experiment")
    plat = by_source.get("platform")

    # --- reconciled recommendation (experiment > MMM) ---
    anchor = exp or mmm or (srcs[0] if srcs else None)
    reconciled: dict[str, Any] = {
        "value": _f(anchor.value) if anchor else None,
        "lower": _f(anchor.lower) if anchor else None,
        "upper": _f(anchor.upper) if anchor else None,
        "basis": anchor.source if anchor else None,
    }

    notes: list[str] = []

    # --- agreement among incremental sources ---
    incr_agree = True
    if len(incremental) >= 2:
        for i in range(len(incremental)):
            for j in range(i + 1, len(incremental)):
                if not _agree(incremental[i], incremental[j]):
                    incr_agree = False

    if len(incremental) >= 2 and incr_agree:
        agreement = "convergent"
    elif len(incremental) >= 2 and not incr_agree:
        agreement = "divergent"
    else:
        agreement = "single-source"

    # --- platform inflation (only meaningful vs an incremental anchor) ---
    incr_anchor = exp or mmm
    if (
        plat is not None
        and not plat.incremental
        and incr_anchor is not None
        and _f(plat.value) is not None
        and _f(incr_anchor.value) not in (None, 0.0)
        and plat.value >= inflation_ratio * incr_anchor.value
    ):
        # Only relabel to platform-inflated when the incremental sources agree
        # (a genuine divergence between MMM and experiment is the bigger story).
        if agreement != "divergent":
            agreement = "platform-inflated"
        window = f" ({plat.attribution_window})" if plat.attribution_window else ""
        notes.append(
            f"The platform reports {_fmt(plat.value)}{window}, but that is a "
            f"last-touch / correlational figure — it credits conversions the "
            f"channel did not necessarily cause. The incremental estimate "
            f"({incr_anchor.label} {_fmt(incr_anchor.value)}) is the return that "
            f"would actually disappear if the channel went dark. Budget on the "
            f"incremental number, not the platform's."
        )

    # --- MMM vs experiment narrative ---
    if exp is not None and mmm is not None:
        if _agree(exp, mmm):
            notes.append(
                f"The experiment ({_fmt(exp.value)}) and the MMM "
                f"({_fmt(mmm.value)}) agree within uncertainty — convergent "
                f"evidence from two independent methods, the most trustworthy "
                f"read available."
            )
        else:
            notes.append(
                f"The experiment ({_fmt(exp.value)}) and the MMM "
                f"({_fmt(mmm.value)}) disagree. Common causes: the experiment "
                f"measured a different window or intensity, or the MMM is picking "
                f"up confounding. Prefer the experiment (a direct causal "
                f"measurement) and re-fit with it calibrated in."
            )
    elif exp is None and mmm is not None:
        notes.append(
            f"Only the MMM measures {channel} — no experiment has confirmed it, "
            f"so this effect is model-identified, not experiment-validated. "
            f"An incrementality test would move it to the top evidence tier."
        )
    elif exp is not None and mmm is None:
        notes.append(
            f"{channel}'s return here is a direct experiment readout "
            f"({_fmt(exp.value)}); fold it into the MMM to propagate it into the "
            f"full budget picture."
        )

    return ChannelTriangulation(
        channel=channel,
        sources=srcs,
        agreement=agreement,
        reconciled=reconciled,
        notes=notes,
    )


def build_triangulation(
    sources_by_channel: dict[str, Sequence[TriangulationSource]],
    *,
    tol: float = _AGREE_TOL,
    inflation_ratio: float = _INFLATION_RATIO,
) -> TriangulationResult:
    """Reconcile every channel's sources into a :class:`TriangulationResult`.

    Channels with no sources are skipped; the order follows insertion order.
    """
    channels = [
        reconcile_channel(ch, srcs, tol=tol, inflation_ratio=inflation_ratio)
        for ch, srcs in sources_by_channel.items()
        if srcs
    ]
    return TriangulationResult(channels=channels)


# =============================================================================
# Builder (model + experiments + platform → sources)
# =============================================================================
# ExperimentEstimand → whether the readout is a return-per-dollar comparable to
# the MMM's contribution ROI (roas/mroas) vs a raw contribution.
_ROAS_LIKE = {"roas", "mroas"}


def _mmm_roi_sources(model: Any, hdi_prob: float) -> dict[str, TriangulationSource]:
    """Per-channel MMM contribution-ROI as triangulation sources."""
    out: dict[str, TriangulationSource] = {}
    try:
        from ..agents.estimand_rows import evaluate_estimand_rows

        rows = evaluate_estimand_rows(model)
    except Exception:  # noqa: BLE001 — fall back to no MMM sources
        return out
    for row in rows or []:
        if row.get("estimand") != "contribution_roi" or row.get("status") != "ok":
            continue
        ch = row.get("channel")
        mean = _f(row.get("mean"))
        if ch is None or mean is None:
            continue
        out[str(ch)] = TriangulationSource(
            source="mmm",
            value=mean,
            lower=_f(row.get("hdi_low")),
            upper=_f(row.get("hdi_high")),
            metric="roi",
            incremental=True,
            note="Model-identified incremental return.",
        )
    return out


def _experiment_sources(
    experiments: Iterable[Any],
) -> dict[str, TriangulationSource]:
    """Per-channel experiment readouts as triangulation sources.

    Accepts either ``ExperimentMeasurement`` objects (``model.experiments``) or
    plain dicts from the registry (``{channel, value, se, estimand, ...}``).
    Only ROAS/mROAS estimands are placed on the return-per-dollar axis; a raw
    contribution readout is kept but marked with its metric.
    """
    out: dict[str, TriangulationSource] = {}
    for exp in experiments or []:
        ch = _attr(exp, "channel")
        value = _f(_attr(exp, "value"))
        if ch is None or value is None:
            continue
        estimand = str(_attr(exp, "estimand") or "roas").lower()
        # ExperimentEstimand enum renders as "ExperimentEstimand.ROAS"; take the tail.
        estimand = estimand.rsplit(".", 1)[-1]
        # Only ROAS/mROAS readouts are on the return-per-$ axis the MMM's
        # contribution ROI lives on. A raw contribution readout (KPI units) is a
        # different scale — it belongs to a separate panel, so skip it here.
        if estimand not in _ROAS_LIKE:
            continue
        se = _f(_attr(exp, "se"))
        lower = value - 1.96 * se if se is not None else None
        upper = value + 1.96 * se if se is not None else None
        out[str(ch)] = TriangulationSource(
            source="experiment",
            value=value,
            lower=lower,
            upper=upper,
            metric="mroas" if estimand == "mroas" else "roas",
            incremental=True,
            attribution_window=_attr(exp, "attribution_window"),
            note="Directly measured incremental lift.",
        )
    return out


def _platform_sources(
    platform: dict[str, Any] | None,
) -> dict[str, TriangulationSource]:
    """Per-channel platform-reported figures as (non-incremental) sources.

    ``platform`` maps ``channel -> {value, metric?, attribution_window?,
    incremental?}``. Platform figures default to *non-incremental* (last-touch),
    which is what drives the inflation caveat; pass ``incremental=True`` for a
    platform that reports a genuine geo-lift.
    """
    out: dict[str, TriangulationSource] = {}
    for ch, fig in (platform or {}).items():
        if isinstance(fig, (int, float)):
            fig = {"value": fig}
        value = _f((fig or {}).get("value"))
        if value is None:
            continue
        out[str(ch)] = TriangulationSource(
            source="platform",
            value=value,
            lower=_f(fig.get("lower")),
            upper=_f(fig.get("upper")),
            metric=str(fig.get("metric", "roas")),
            incremental=bool(fig.get("incremental", False)),
            attribution_window=fig.get("attribution_window", "last-touch"),
            note=fig.get("note", "Platform-reported attribution."),
        )
    return out


def triangulation_from_model(
    model: Any,
    *,
    experiments: Iterable[Any] | None = None,
    platform: dict[str, Any] | None = None,
    hdi_prob: float = 0.94,
) -> TriangulationResult:
    """Assemble a triangulation panel from a fitted model + experiments + platform.

    * MMM sources come from the model's ``contribution_roi`` estimand per channel.
    * Experiment sources come from ``experiments`` (registry readouts) or, when
      that is ``None``, the model's in-graph calibrations (``model.experiments``).
    * Platform sources come from the ``platform`` dict (optional).

    Channels are the union across all three sources so a channel measured only by
    an experiment or only by the platform still appears.
    """
    mmm = _mmm_roi_sources(model, hdi_prob)
    exps = (
        experiments if experiments is not None else getattr(model, "experiments", None)
    )
    experiment = _experiment_sources(exps or [])
    plat = _platform_sources(platform)

    channels: list[str] = []
    for group in (mmm, experiment, plat):
        for ch in group:
            if ch not in channels:
                channels.append(ch)

    sources_by_channel: dict[str, list[TriangulationSource]] = {}
    for ch in channels:
        # Display order: experiment (the anchor) → MMM → platform.
        srcs = [g[ch] for g in (experiment, mmm, plat) if ch in g]
        sources_by_channel[ch] = srcs
    return build_triangulation(sources_by_channel, tol=_AGREE_TOL)


# =============================================================================
# Small utilities
# =============================================================================
def _attr(obj: Any, name: str) -> Any:
    """Read an attribute from an object OR a key from a dict."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def esc(s: Any) -> str:
    """HTML-escape for the report renderers (channel names are user-controlled)."""
    return _html.escape(str(s))
