"""CMO / media-planner narrative insights for the Augur "Media Performance
Readout".

The numbers come from a Bayesian MMM; this module turns them into the *prose*
an executive reads — the headline, the standfirst, each channel's "what to do",
and the recommended tests / next steps. Two layers, by design:

1. **Deterministic facts** (``report_facts``) are derived purely from the
   :class:`MMMDataBundle` and the shared tier classifier — never invented.
2. A **templated fallback** (``_fallback_insights``) renders a complete,
   grounded narrative from those facts with no LLM, so a report is *never*
   empty and is reproducible offline / in tests.
3. When an LLM is supplied, ``build_report_insights`` **enriches** the headline,
   standfirst and per-channel prose (grounded strictly in the same facts),
   best-effort — an LLM failure silently falls back to the templated text.

This mirrors the deck's grounding discipline (``agents.deck_insights``) but
operates on the report bundle and keeps the reporting package free of any
dependency on the agent layer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .helpers.reallocation import channel_rows, test_candidates

# Slots the Augur sections read. Every key is always present after
# build_report_insights (templated fallback guarantees completeness).
INSIGHT_SLOTS = (
    "headline",
    "standfirst",
    "kpi_gloss",
    "fit_gloss",
    "tests_intro",
    "next_steps",
)


# ─────────────────────────────────────────────────────────────────────────────
# Formatting (self-contained; the sections re-format for display, this is prose)
# ─────────────────────────────────────────────────────────────────────────────
def _money(v: float | None, currency: str = "$") -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    av = abs(v)
    sign = "-" if v < 0 else ""
    if av >= 1e9:
        return f"{sign}{currency}{av / 1e9:.1f}B"
    if av >= 1e6:
        return f"{sign}{currency}{av / 1e6:.1f}M"
    if av >= 1e3:
        return f"{sign}{currency}{av / 1e3:.1f}K"
    return f"{sign}{currency}{av:.0f}"


def _x(v: float | None) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.2f}"


def _pct(v: float | None) -> str:
    """``v`` is a proportion in [0, 1]."""
    return "n/a" if v is None or not np.isfinite(v) else f"{v * 100:.0f}%"


def _names(rows: list[dict], joiner: str = " and ") -> str:
    labels = [str(r["name"]).replace("_", " ") for r in rows]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return joiner.join(labels)
    return ", ".join(labels[:-1]) + joiner + labels[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic facts
# ─────────────────────────────────────────────────────────────────────────────
def report_facts(bundle: Any, break_even: float = 1.0) -> dict[str, Any]:
    """Pull every grounded fact the narrative needs from the bundle."""
    rows = channel_rows(bundle, break_even=break_even)
    groups: dict[str, list[dict]] = {"scale": [], "test": [], "hold": [], "reduce": []}
    for r in rows:
        groups[r["tier"]].append(r)

    def _triple(d: Any) -> dict[str, float] | None:
        if not isinstance(d, dict):
            return None
        try:
            return {
                "mean": float(d["mean"]),
                "lower": float(d.get("lower", d["mean"])),
                "upper": float(d.get("upper", d["mean"])),
            }
        except (KeyError, TypeError, ValueError):
            return None

    # Goodness-of-fit summary for the "does the model hold up?" gloss.
    r2 = None
    fit = getattr(bundle, "fit_statistics", None)
    if isinstance(fit, dict):
        try:
            r2 = float(fit.get("r2")) if fit.get("r2") is not None else None
        except (TypeError, ValueError):
            r2 = None
    coverage_pct = None
    ci_level = None
    pp = getattr(bundle, "posterior_predictive", None)
    if isinstance(pp, dict):
        try:
            if pp.get("r2") is not None and r2 is None:
                r2 = float(pp["r2"])
        except (TypeError, ValueError):
            pass
        ci_level = pp.get("ci_level")
        cov = pp.get("coverage")
        # coverage is a list of {nominal, empirical}; report the point nearest
        # the headline CI level (default 0.8).
        if isinstance(cov, list) and cov:
            target = float(ci_level) if ci_level else 0.8
            try:
                nearest = min(
                    cov, key=lambda c: abs(float(c.get("nominal", 0)) - target)
                )
                coverage_pct = float(nearest.get("empirical"))
            except (TypeError, ValueError):
                coverage_pct = None

    return {
        "currency": getattr(getattr(bundle, "_config", None), "currency_symbol", "$"),
        "break_even": break_even,
        "rows": rows,
        "groups": groups,
        "scale": groups["scale"],
        "test": groups["test"],
        "hold": groups["hold"],
        "reduce": groups["reduce"],
        "top": rows[0] if rows else None,
        "worst": rows[-1] if rows else None,
        "test_candidates": test_candidates(rows),
        "blended_roi": _triple(getattr(bundle, "blended_roi", None)),
        "marketing_share": _triple(getattr(bundle, "marketing_contribution_pct", None)),
        "marketing_revenue": _triple(
            getattr(bundle, "marketing_attributed_revenue", None)
        ),
        "total_revenue": getattr(bundle, "total_revenue", None),
        "r2": r2,
        "coverage_pct": coverage_pct,
        "ci_level": float(ci_level) if ci_level else 0.8,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Templated fallback (always complete, always grounded)
# ─────────────────────────────────────────────────────────────────────────────
def _fallback_headline(f: dict[str, Any]) -> str:
    top, worst = f["top"], f["worst"]
    if f["scale"] and f["reduce"]:
        return (
            f"Scale {f['scale'][0]['name'].replace('_', ' ')}, trim "
            f"{f['reduce'][0]['name'].replace('_', ' ')} — same budget, better return"
        )
    if f["scale"]:
        return f"{f['scale'][0]['name'].replace('_', ' ')} is the channel to lean into"
    if f["reduce"]:
        return "Marketing earns its keep — in the mix, not the total"
    if top and worst and top is not worst:
        return "Where marketing pays back — and where it doesn't yet"
    return "Reading the marketing mix, with the uncertainty shown"


def _fallback_standfirst(f: dict[str, Any]) -> str:
    cur = f["currency"]
    bits: list[str] = []
    share = f["marketing_share"]
    roi = f["blended_roi"]
    if share and roi:
        bits.append(
            f"Marketing drives about {_pct(share['mean'])} of revenue at a blended "
            f"return of {cur}{_x(roi['mean'])} per {cur}1 of spend"
        )
    elif roi:
        bits.append(
            f"The blended return sits at {cur}{_x(roi['mean'])} per {cur}1 of spend"
        )
    move: list[str] = []
    if f["reduce"]:
        move.append(f"out of {_names(f['reduce'])} (below break-even)")
    if f["scale"]:
        move.append(f"into {_names(f['scale'])} (the range clears break-even)")
    if move:
        sentence = "The gain is not in spending more — it is in moving budget " + (
            " and ".join(move)
        )
        if f["test"]:
            sentence += f", while testing {_names(f['test'])} before funding at risk"
        bits.append(sentence)
    if not bits:
        bits.append(
            "Each channel is read against break-even with its full credible "
            "interval, so the action follows the confidence, not just the point estimate"
        )
    return ". ".join(bits) + "."


def _fallback_channel(r: dict[str, Any], f: dict[str, Any]) -> str:
    cur = f["currency"]
    name = str(r["name"]).replace("_", " ")
    roi, lo, hi = _x(r["roi"]), _x(r["roi_lower"]), _x(r["roi_upper"])
    mroas = r.get("mroas")
    top_name = (
        f["scale"][0]["name"].replace("_", " ")
        if f["scale"]
        else (f["top"]["name"].replace("_", " ") if f["top"] else "the proven winner")
    )
    if r["tier"] == "scale":
        tail = (
            f" and the next dollar (marginal {_x(mroas)}) still pays back"
            if mroas is not None
            else ""
        )
        return (
            f"{name} returns {cur}{roi} per {cur}1 and its entire 80% range "
            f"({lo}–{hi}) clears break-even{tail}. Increase weight and hold it "
            f"continuously rather than in bursts; re-check the saturation point "
            f"after each step up."
        )
    if r["tier"] == "test":
        return (
            f"{name}'s central {cur}{roi} is attractive but the 80% range "
            f"({lo}–{hi}) is too wide to fund on faith. Hold spend roughly steady "
            f"and run a matched-market holdout to settle whether the true return "
            f"sits near the top or the bottom of that range."
        )
    if r["tier"] == "hold":
        return (
            f"{name} sits near break-even ({cur}{roi} per {cur}1, 80% {lo}–{hi}). "
            f"Maintain weight and continuity — the case is neither to scale nor to "
            f"cut until a test or more data moves it off the line."
        )
    # reduce
    return (
        f"{name} returns {cur}{roi} per {cur}1 across its whole range ({lo}–{hi}) — "
        f"below break-even. Trim spend in steps and redirect the freed budget into "
        f"{top_name}; pair the spend-down with a test so the cut is calibrated."
    )


def _fallback_tests_intro(f: dict[str, Any]) -> str:
    cands = f["test_candidates"]
    if not cands:
        return (
            "Experiments convert channels from model-only to evidence-backed, "
            "shrinking the ranges that force a hold."
        )
    return (
        "Each test converts a channel from model-only to evidence-backed, "
        f"shrinking the ranges that currently force a hold. The priorities are "
        f"{_names(cands[:3])}."
    )


def _fallback_next_steps(f: dict[str, Any]) -> str:
    parts: list[str] = []
    if f["scale"] or f["reduce"]:
        grow = _names(f["scale"]) or "the proven winners"
        trim = _names(f["reduce"]) or "the weakest returns"
        parts.append(
            f"Planning: action the reallocation — grow {grow} continuously, trim "
            f"{trim}, and hold the rest steady into the next flight."
        )
    if f["test_candidates"]:
        parts.append(
            "Analytics: stand up the recommended tests and pre-register the "
            "read-outs."
        )
    parts.append(
        "Together: re-run the model once tests land to calibrate the estimates "
        "and re-cut the budget with tighter ranges."
    )
    return " ".join(parts)


def _fallback_fit_gloss(f: dict[str, Any]) -> str:
    r2 = f["r2"]
    cov = f["coverage_pct"]
    lvl = int(round(f["ci_level"] * 100))
    if r2 is not None and cov is not None:
        return (
            f"The model reproduces observed revenue closely (R² = {r2:.2f}), and "
            f"about {cov * 100:.0f}% of points fall inside the {lvl}% predictive "
            f"band — calibrated enough to act on."
        )
    if r2 is not None:
        return (
            f"The model reproduces observed revenue closely (R² = {r2:.2f}); the "
            f"posterior-predictive checks below show where it holds and where it strains."
        )
    return (
        "The checks below show how well the model reproduces observed revenue — "
        "the basis for trusting the numbers above."
    )


def _fallback_kpi_gloss(f: dict[str, Any]) -> str:
    roi = f["blended_roi"]
    if roi and roi["mean"] < f["break_even"]:
        return (
            "The blended return sits just below break-even — the opportunity is in "
            "the mix, not the total."
        )
    if roi:
        return "Marketing is paying back on average; the mix is where the upside is."
    return ""


def _fallback_insights(f: dict[str, Any]) -> dict[str, str]:
    """A complete, grounded narrative with no LLM."""
    out: dict[str, str] = {
        "headline": _fallback_headline(f),
        "standfirst": _fallback_standfirst(f),
        "kpi_gloss": _fallback_kpi_gloss(f),
        "fit_gloss": _fallback_fit_gloss(f),
        "tests_intro": _fallback_tests_intro(f),
        "next_steps": _fallback_next_steps(f),
    }
    for r in f["rows"]:
        out[f"channel:{r['name']}"] = _fallback_channel(r, f)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# LLM enrichment (best-effort, grounded)
# ─────────────────────────────────────────────────────────────────────────────
_HEADLINE_SYS = (
    "You are a senior marketing-effectiveness analyst writing the opening of a "
    "client MMM readout for a CMO and media planners. Synthesize across the whole "
    "portfolio; be specific, decision-oriented, and plain-spoken. Ground every "
    "claim in the facts provided — never invent numbers. No preamble, no markdown, "
    "no quotes."
)
_CHANNEL_SYS = (
    "You are a senior marketing-effectiveness analyst writing one channel's "
    "recommendation for a client MMM readout. Be concise, quantitative, and "
    "decision-oriented (what return / next-dollar marginal ROI / evidence tier "
    "imply, and the specific spend action). Ground every claim in the facts — "
    "never invent numbers. 1–2 sentences (~40 words). No preamble, no markdown, "
    "no bullet markers."
)


def _facts_blob(f: dict[str, Any]) -> str:
    cur = f["currency"]
    lines: list[str] = []
    roi, share, rev = f["blended_roi"], f["marketing_share"], f["marketing_revenue"]
    if rev:
        lines.append(
            f"Marketing-attributed revenue: {_money(rev['mean'], cur)} "
            f"(80% {_money(rev['lower'], cur)}–{_money(rev['upper'], cur)})."
        )
    if share:
        lines.append(
            f"Marketing share of revenue: {_pct(share['mean'])} "
            f"(80% {_pct(share['lower'])}–{_pct(share['upper'])})."
        )
    if roi:
        lines.append(
            f"Blended return per {cur}1: {cur}{_x(roi['mean'])} "
            f"(80% {cur}{_x(roi['lower'])}–{cur}{_x(roi['upper'])}); "
            f"break-even is {cur}1.00."
        )
    if f["r2"] is not None:
        lines.append(f"Model fit R²: {f['r2']:.2f}.")
    lines.append("Channels (return per $1, 80% CI, evidence tier, action):")
    for r in f["rows"]:
        mroas = f", next-dollar {_x(r['mroas'])}" if r.get("mroas") is not None else ""
        lines.append(
            f"  - {r['name']}: {_x(r['roi'])} (80% {_x(r['roi_lower'])}–"
            f"{_x(r['roi_upper'])}){mroas} — {r['read']} → {r['action']}."
        )
    return "\n".join(lines)


def _clean(reply: Any) -> str:
    """Plain text from a LangChain reply (content may be a list of blocks)."""
    s = reply.content if hasattr(reply, "content") else reply
    if isinstance(s, list):
        parts = []
        for blk in s:
            if isinstance(blk, dict):
                parts.append(str(blk.get("text") or blk.get("content") or ""))
            elif isinstance(blk, str):
                parts.append(blk)
        s = " ".join(p for p in parts if p)
    elif not isinstance(s, str):
        s = str(s)
    return " ".join(s.split()).strip().strip('"')


def _split_headline(text: str) -> tuple[str, str]:
    low = text.lower()
    head, standfirst = text, ""
    if "standfirst:" in low:
        i = low.index("standfirst:")
        head, standfirst = text[:i], text[i + len("standfirst:") :]
    if "headline:" in head.lower():
        j = head.lower().index("headline:")
        head = head[j + len("headline:") :]
    return head.strip().strip('"').strip(), standfirst.strip().strip('"').strip()


def _cap_headline(headline: str, max_words: int = 16) -> str:
    first = headline.split(". ")[0].strip().rstrip(".")
    words = first.split()
    if len(words) > max_words:
        first = " ".join(words[:max_words]).rstrip(",;:")
    return first


def _enrich_with_llm(
    f: dict[str, Any],
    insights: dict[str, str],
    llm: Any,
    *,
    max_channels: int = 10,
) -> dict[str, str]:
    """Overlay LLM-written headline/standfirst/per-channel prose on the fallback.

    Best-effort: if the first call fails (LLM unreachable / mis-configured) the
    templated text is kept unchanged.
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception:
        return insights

    blob = _facts_blob(f)

    # 1) synthesis -> headline + standfirst
    try:
        prompt = (
            f"Portfolio facts:\n{blob}\n\n"
            "Return EXACTLY two labelled parts:\n"
            "HEADLINE: a punchy title of AT MOST 12 words — the single most "
            "important finding for a CMO (no numbers needed).\n"
            "STANDFIRST: 1–2 sentences (≤45 words) naming the top reallocation "
            "move and the key supporting numbers."
        )
        r = llm.invoke(
            [SystemMessage(content=_HEADLINE_SYS), HumanMessage(content=prompt)]
        )
        headline, standfirst = _split_headline(_clean(r))
        if headline:
            insights["headline"] = _cap_headline(headline)
        if standfirst:
            insights["standfirst"] = standfirst
    except Exception:
        return insights  # LLM unreachable — keep the full templated fallback

    # 2) per-channel "what to do"
    for i, row in enumerate(f["rows"]):
        if i >= max_channels:
            break
        try:
            prompt = (
                f"Portfolio context:\n{blob}\n\n"
                f"Write the recommendation for {row['name']} "
                f"(return {_x(row['roi'])} per $1, 80% {_x(row['roi_lower'])}–"
                f"{_x(row['roi_upper'])}, "
                f"next-dollar {_x(row.get('mroas'))}, tier: {row['read']} → "
                f"{row['action']})."
            )
            r = llm.invoke(
                [SystemMessage(content=_CHANNEL_SYS), HumanMessage(content=prompt)]
            )
            txt = _clean(r)
            if txt:
                insights[f"channel:{row['name']}"] = txt
        except Exception:
            continue

    return insights


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────
def build_report_insights(
    bundle: Any,
    *,
    llm: Any | None = None,
    audience: str = "cmo",
    break_even: float = 1.0,
) -> dict[str, str]:
    """CMO / media-planner narrative for the Augur report.

    Always returns a complete dict (templated fallback for every slot). When
    ``llm`` is provided, the headline, standfirst and per-channel prose are
    enriched, best-effort — any failure degrades to the templated text.

    Parameters
    ----------
    bundle : MMMDataBundle
        Extracted model outputs (channel_roi, blended_roi, estimands, …).
    llm : optional
        A LangChain chat model (``.invoke``). ``None`` → templated only.
    audience : str
        Reserved for future audience-specific tuning ("cmo" / "planner").
    break_even : float
        ROI reference (1.0 = a dollar back per dollar spent).
    """
    facts = report_facts(bundle, break_even=break_even)
    insights = _fallback_insights(facts)
    if llm is not None:
        try:
            insights = _enrich_with_llm(facts, insights, llm)
        except Exception:
            pass
    return insights


__all__ = ["build_report_insights", "report_facts", "INSIGHT_SLOTS"]
