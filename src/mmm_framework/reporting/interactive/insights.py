"""Narrative layer for the interactive MMM Results Report.

Two-layer prose discipline, identical to
:func:`mmm_framework.reporting.prefit.build_prefit_insights`:

1. **Templated insights** (``llm=None``) — complete, grounded in the computed
   facts, reproducible offline.
2. **AI-enriched insights** — when a LangChain chat model is supplied the
   standfirst and section glosses are rewritten from the same facts,
   best-effort; any failure silently keeps the templated text.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["INTERACTIVE_INSIGHT_SLOTS", "build_interactive_insights"]

INTERACTIVE_INSIGHT_SLOTS = (
    "standfirst",
    "exec_gloss",
    "decomp_gloss",
    "fit_gloss",
    "ppc_stats_gloss",
    "loo_pit_gloss",
    "roi_gloss",
    "yoy_gloss",
    "estimands_gloss",
    "curves_gloss",
    "carryover_gloss",
    "pathways_gloss",
    "latent_gloss",
    "prior_posterior_gloss",
    "realloc_gloss",
    "sensitivity_gloss",
    "ppc_gloss",
    "assumptions_gloss",
    "next_steps",
)


def _fmt_ci(row: dict[str, Any] | None, digits: int = 2) -> str:
    if not row or row.get("mean") is None:
        return "—"
    return (
        f"{row['mean']:.{digits}f} "
        f"({row.get('lower', float('nan')):.{digits}f}–"
        f"{row.get('upper', float('nan')):.{digits}f})"
    )


def _sensitivity_spread(facts: dict[str, Any]) -> list[tuple[str, float, float]]:
    """Per-channel (min, max) of the sensitivity means across specs."""
    sens = facts.get("sensitivity") or {}
    out: list[tuple[str, float, float]] = []
    for ch, s in (sens.get("series") or {}).items():
        vals = [v for v in s.get("mean", []) if v is not None]
        if len(vals) >= 2:
            out.append((ch, min(vals), max(vals)))
    return out


def _fallback_insights(f: dict[str, Any]) -> dict[str, str]:
    meta = f.get("meta", {})
    head = f.get("headline", {})
    kpi = meta.get("kpi") or "the KPI"
    n_ch = len(meta.get("channels", []))
    window = ""
    if meta.get("date_start") and meta.get("date_end"):
        window = f" over {meta['date_start']} – {meta['date_end']}"
    interval = int(round(float(meta.get("interval", 0.9)) * 100))

    out: dict[str, str] = {}
    approx = (
        " The fit is an APPROXIMATE posterior (see the banner) — intervals "
        "here are not calibrated."
        if meta.get("approximate")
        else ""
    )
    out["standfirst"] = (
        f"This is the fitted readout of a Bayesian marketing mix model of {kpi} "
        f"across {n_ch} media channel{'s' if n_ch != 1 else ''}{window}. Every "
        f"number carries a {interval}% credible interval, and the date-window "
        f"selectors re-aggregate the same posterior draws — they never refit "
        f"the model.{approx}"
    )

    share = head.get("media_share")
    blended = head.get("blended_roi")
    bits = []
    if head.get("media_total"):
        bits.append(
            f"media is credited with {_fmt_ci(head['media_total'], 0)} of "
            f"{head.get('total_kpi', 0):,.0f} total {kpi}"
        )
    if share:
        bits.append(
            f"a {share['mean']:.0%} share ({share['lower']:.0%}–{share['upper']:.0%})"
        )
    if blended:
        bits.append(f"blended media ROI {_fmt_ci(blended)}")
    out["exec_gloss"] = (
        "Over the selected window "
        + (
            "; ".join(bits)
            if bits
            else "the summary cards recompute from the posterior"
        )
        + ". Narrow the window to any campaign or fiscal period — the cards "
        "and their intervals recompute from the same posterior draws."
    )

    fit = head.get("fit") or {}
    r2 = fit.get("r2")
    cov = fit.get("coverage90")
    out["fit_gloss"] = (
        "The band around the prediction is the posterior-predictive interval "
        "— where the model says data like ours should fall. "
        + (
            f"In-sample R² is {r2:.2f}"
            + (
                f" and the 90% band covers {cov:.0%} of observed periods "
                "(close to 90% is well calibrated)."
                if cov is not None
                else "."
            )
            if r2 is not None
            else "Fit statistics are shown per series."
        )
    )

    share_txt = f"media's {share['mean']:.0%} share" if share else "the media share"
    out["decomp_gloss"] = (
        "The stacked view separates what marketing added each week from what "
        f"the business would have done anyway — {share_txt} sits on top of "
        "the baseline (trend, seasonality, controls). Below it, spend share "
        "vs effect share shows which channels punch above or below their "
        "budget weight."
    )

    med = f.get("mediation")
    if med:
        n_ind = sum(1 for lk in med.get("links", []) if lk.get("kind") == "indirect")
        meds = ", ".join(med.get("mediators", [])) or "the mediator"
        out["pathways_gloss"] = (
            f"This model is structural: {n_ind} channel effect"
            f"{'s' if n_ind != 1 else ''} reach the outcome indirectly "
            f"through {meds}, alongside direct response. The Sankey shows "
            "both routes at once — a channel judged only on its direct arrow "
            "would be under-credited for everything it moves upstream."
        )
    else:
        out["pathways_gloss"] = (
            "Structural models decompose each channel's effect into a direct "
            "route and indirect routes through mediators; this model has no "
            "mediation structure."
        )
    lat = f.get("latent")
    if lat:
        n_load = len(lat.get("loadings") or [])
        n_traj = len(lat.get("trajectories") or [])
        neg = sum(1 for r in lat.get("loadings") or [] if r.get("mean", 0) < 0)
        out[
            "latent_gloss"
        ] = "The model infers unobserved drivers jointly with the media " "effects, so their uncertainty propagates into every ROI above. " + (
            f"{n_load} indicators load on the factor"
            + (f" ({neg} negatively)" if neg else "")
            + ". "
            if n_load
            else ""
        ) + (
            f"{n_traj} latent state{'s' if n_traj != 1 else ''} "
            "tracked over time below."
            if n_traj
            else ""
        )
    else:
        out["latent_gloss"] = (
            "Models with latent factors report their loadings and inferred "
            "states here; this model has none."
        )

    stats = (f.get("ppc_stats") or {}).get("stats") or []
    extreme = [s["label"] for s in stats if s.get("extreme")]
    if stats and not extreme:
        out["ppc_stats_gloss"] = (
            "A model can track the weekly series and still be the wrong "
            "model. These checks ask whether datasets simulated from the "
            "fitted posterior share the KPI's character — its extremes, "
            "volatility and week-to-week persistence. All "
            f"{len(stats)} test statistics here sit comfortably inside their "
            "replicated distributions, so the model is reproducing the data's "
            "properties, not just interpolating it."
        )
    elif stats:
        out["ppc_stats_gloss"] = (
            "A model can track the weekly series and still be the wrong "
            "model. These checks ask whether datasets simulated from the "
            "fitted posterior share the KPI's character. "
            f"{len(extreme)} of {len(stats)} statistics — "
            f"{', '.join(extreme[:4])} — fall in the extreme tails, meaning "
            "the model systematically misses that property of the data; "
            "conclusions that lean on it deserve extra scrutiny."
        )
    else:
        out["ppc_stats_gloss"] = (
            "Posterior-predictive test statistics compare properties of "
            "replicated datasets (extremes, volatility, persistence) to the "
            "observed KPI; Bayesian p-values near 0 or 1 flag properties the "
            "model cannot reproduce."
        )

    pitf = (f.get("ppc_stats") or {}).get("loo_pit")
    if pitf:
        kind = (
            "leave-one-out (PSIS-LOO) predictive"
            if pitf.get("weighting") == "psis-loo"
            else "posterior-predictive"
        )
        if pitf.get("calibrated"):
            out["loo_pit_gloss"] = (
                "A sharper calibration lens: the probability-integral "
                "transform asks, for every single observation, where it fell "
                f"within its {kind} distribution. Under honest uncertainty "
                "those positions scatter uniformly on (0, 1) — and here they "
                f"do (KS p = {pitf.get('ks_p', 0):.2f} across "
                f"{pitf.get('n', 0)} observations), so the intervals are "
                "trustworthy observation by observation, not just on average."
            )
        else:
            out["loo_pit_gloss"] = (
                "A sharper calibration lens: the probability-integral "
                "transform asks, for every single observation, where it fell "
                f"within its {kind} distribution. Under honest uncertainty "
                "those positions scatter uniformly on (0, 1) — here they do "
                f"not (KS p = {pitf.get('ks_p', 0):.3f} across "
                f"{pitf.get('n', 0)} observations). The histogram's shape "
                "says how: a ∪ shape means overconfident (too-narrow) "
                "intervals, a ∩ shape too-wide ones, and a slope means "
                "systematic bias."
            )
    else:
        out["loo_pit_gloss"] = (
            "The LOO-PIT check places every observation within its "
            "leave-one-out predictive distribution; uniform positions on "
            "(0, 1) mean calibrated predictive uncertainty. It was not "
            "computed for this fit."
        )

    rows = sorted(
        head.get("channels", []), key=lambda r: r.get("roi_mean") or 0, reverse=True
    )
    if rows:
        best, worst = rows[0], rows[-1]
        out["roi_gloss"] = (
            f"{best['channel']} leads at {_fmt_ci({'mean': best['roi_mean'], 'lower': best['roi_lower'], 'upper': best['roi_upper']})} "
            f"{best['label']}; {worst['channel']} trails at "
            f"{_fmt_ci({'mean': worst['roi_mean'], 'lower': worst['roi_lower'], 'upper': worst['roi_upper']})}. "
            "Whiskers are posterior credible intervals: where they cross the "
            "reference line, the sign of the return is genuinely uncertain. "
            "Use the year-over-year view to see whether performance moved."
        )
    else:
        out["roi_gloss"] = (
            "Per-channel returns with credible intervals over the selected window."
        )

    yoy = (f.get("yoy") or {}).get("latest")
    if yoy:
        drivers = sorted(
            yoy.get("drivers", []), key=lambda d: abs(d["mean"]), reverse=True
        )
        top = drivers[0] if drivers else None
        base = yoy.get("baseline") or {}
        delta = yoy.get("delta", 0.0)
        out["yoy_gloss"] = (
            f"From {yoy['year_a']} to {yoy['year_b']} the KPI moved by "
            f"{delta:+,.0f}. "
            + (
                f"The largest media driver was {top['name']} "
                f"({top['mean']:+,.0f}; {top['lower']:+,.0f} to "
                f"{top['upper']:+,.0f}), "
                if top
                else ""
            )
            + f"while baseline and non-media factors account for "
            f"{base.get('mean', 0):+,.0f}. The waterfall decomposes any pair "
            "of years the same way, with credible intervals on every media "
            "driver."
        )
    else:
        out["yoy_gloss"] = (
            "The waterfall bridges one year's total KPI to the next, "
            "splitting the change into per-channel media drivers (with "
            "credible intervals) and the residual baseline movement."
        )

    out["estimands_gloss"] = (
        "The same posterior supports several causal questions. Average ROI "
        'answers "what did past spend return"; marginal ROAS answers "what '
        'would the next dollar return" — the two disagree whenever a channel '
        "is saturated, and budget decisions should follow the marginal number."
    )
    out["curves_gloss"] = (
        "Each curve re-evaluates the fitted response at scaled versions of the "
        "channel's actual spend history, so carryover and flighting are "
        "respected. The marker pins today's average weekly spend; the band is "
        "posterior uncertainty about the curve itself."
    )
    hl = [
        (ch, c["half_life"]["mean"])
        for ch, c in (f.get("carryover") or {}).items()
        if c.get("half_life")
    ]
    if hl:
        hl.sort(key=lambda t: t[1], reverse=True)
        out["carryover_gloss"] = (
            f"Carryover is longest for {hl[0][0]} (half-life "
            f"≈ {hl[0][1]:.1f} weeks) and shortest for {hl[-1][0]} "
            f"(≈ {hl[-1][1]:.1f}). Channels with long memory under-credit in "
            "short attribution windows — judge them on windows longer than "
            "their half-life."
        )
    else:
        out["carryover_gloss"] = (
            "Posterior carryover kernels per channel — how much of each "
            "week's effect spills into later weeks."
        )
    out["prior_posterior_gloss"] = (
        "Both densities live on the decision scale (return per unit spend), "
        "not raw coefficients. Where the posterior is much narrower than the "
        "prior, the data spoke; where it hugs the prior, the estimate is "
        "still prior-driven and deserves experimental validation."
    )
    out["realloc_gloss"] = (
        "The reallocator moves budget along each channel's fitted response "
        "curve, holding everything else fixed. Results are approximate — "
        "posterior draws are interpolated between precomputed spend levels — "
        "and assume the historical flighting pattern scales proportionally."
    )
    spread = _sensitivity_spread(f)
    fragile = [
        ch
        for ch, lo, hi in spread
        if lo < 1.0 < hi or (hi - lo) > 0.5 * max(abs(hi), 1e-9)
    ]
    out["sensitivity_gloss"] = (
        "Each column re-estimates channel returns under an alternative "
        "analysis choice — sub-windows, influential-week exclusions, and a "
        "different counterfactual estimator. "
        + (
            f"Findings for {', '.join(fragile[:3])} move materially across "
            "specs — treat those with caution and prioritize them for testing."
            if fragile
            else "Rankings that survive every column are robust findings."
        )
    )
    out["ppc_gloss"] = (
        "Recorded from the pre-fit design: what the model's priors implied "
        "before seeing the data's verdict. If the observed series sits "
        "comfortably inside the prior fan, the posterior above was reached "
        "from open-minded starting assumptions rather than baked in."
    )
    out["assumptions_gloss"] = (
        "Every attribution above is a causal claim resting on assumptions the "
        "data cannot fully check: declared confounders controlled, no "
        "unobserved demand-chasing, and the fitted functional family. Stating "
        "them converts an implicit leap of faith into a reviewable claim."
    )
    out["next_steps"] = (
        "Validate the largest and least certain channel effects with designed "
        "experiments (geo holdouts or flighting tests), then recalibrate the "
        "model on the readouts — the experiment tools in this workspace "
        "design and power those tests from this same posterior."
    )
    return out


_SYS = (
    "You are a careful marketing-science writer. Ground every sentence in the "
    "facts given; never invent numbers. Plain prose, no markdown, no headers."
)

_LLM_SLOT_LABELS = {
    "STANDFIRST": "standfirst",
    "EXEC": "exec_gloss",
    "FIT": "fit_gloss",
    "PPC_STATS": "ppc_stats_gloss",
    "LOO_PIT": "loo_pit_gloss",
    "DECOMPOSITION": "decomp_gloss",
    "ROI": "roi_gloss",
    "YOY": "yoy_gloss",
    "ESTIMANDS": "estimands_gloss",
    "CURVES": "curves_gloss",
    "CARRYOVER": "carryover_gloss",
    "PATHWAYS": "pathways_gloss",
    "LATENT": "latent_gloss",
    "PRIOR_POSTERIOR": "prior_posterior_gloss",
    "REALLOCATION": "realloc_gloss",
    "SENSITIVITY": "sensitivity_gloss",
    "PRIOR_PREDICTIVE": "ppc_gloss",
    "ASSUMPTIONS": "assumptions_gloss",
    "NEXT_STEPS": "next_steps",
}


def _facts_blob(f: dict[str, Any]) -> str:
    meta = f.get("meta", {})
    head = f.get("headline", {})
    lines = [
        f"KPI: {meta.get('kpi')}. Channels: {', '.join(meta.get('channels', []))}.",
        f"Window: {meta.get('date_start')} – {meta.get('date_end')} "
        f"({meta.get('n_periods')} periods). Fit method: {meta.get('fit_method')}"
        f"{' (APPROXIMATE — uncertainty not calibrated)' if meta.get('approximate') else ''}.",
        f"Total KPI: {head.get('total_kpi', 0):,.0f}. "
        f"Media-attributed: {_fmt_ci(head.get('media_total'), 0)}.",
    ]
    if head.get("media_share"):
        s = head["media_share"]
        lines.append(
            f"Media share of KPI: {s['mean']:.0%} ({s['lower']:.0%}–{s['upper']:.0%})."
        )
    if head.get("blended_roi"):
        lines.append(f"Blended media ROI: {_fmt_ci(head['blended_roi'])}.")
    for r in head.get("channels", []):
        lines.append(
            f"Channel {r['channel']}: {r['label']} {r['roi_mean']:.2f} "
            f"({r['roi_lower']:.2f}–{r['roi_upper']:.2f}), reference "
            f"{r.get('reference', 1.0):g}, window spend {r['spend']:,.0f}."
        )
    fit = head.get("fit") or {}
    if fit.get("r2") is not None:
        lines.append(
            f"National fit: R2 {fit['r2']:.2f}"
            + (
                f", 90% predictive coverage {fit['coverage90']:.0%}"
                if fit.get("coverage90") is not None
                else ""
            )
            + "."
        )
    for ch, c in (f.get("carryover") or {}).items():
        h = c.get("half_life") or {}
        if h.get("mean") is not None:
            lines.append(
                f"Carryover {ch}: half-life {h['mean']:.1f} wks "
                f"({h.get('lower', 0):.1f}–{h.get('upper', 0):.1f})."
            )
    for ch, lo, hi in _sensitivity_spread(f):
        lines.append(f"Sensitivity {ch}: ROI ranges {lo:.2f}–{hi:.2f} across specs.")
    med = f.get("mediation")
    if med:
        for lk in med.get("links", [])[:12]:
            lines.append(
                f"Pathway {lk['source']} -> {lk['target']} ({lk['kind']}): "
                f"{lk['mean']:.3g} ({lk['lower']:.3g} to {lk['upper']:.3g}), "
                f"units: {med.get('units')}."
            )
    lat = f.get("latent")
    if lat:
        for r in (lat.get("loadings") or [])[:10]:
            lines.append(f"Loading {r['indicator']} on {r['factor']}: {r['mean']:.2f}.")
        for tr in lat.get("trajectories") or []:
            lines.append(f"Latent state tracked: {tr['name']}.")
    yoy = (f.get("yoy") or {}).get("latest")
    if yoy:
        lines.append(
            f"YoY {yoy['year_a']}->{yoy['year_b']}: total change "
            f"{yoy.get('delta', 0):+,.0f}; baseline "
            f"{(yoy.get('baseline') or {}).get('mean', 0):+,.0f}; drivers "
            + ", ".join(
                f"{d['name']} {d['mean']:+,.0f}" for d in yoy.get("drivers", [])
            )
            + "."
        )
    for s in (f.get("ppc_stats") or {}).get("stats") or []:
        lines.append(
            f"PPC stat {s['label']}: observed {s['observed']:.3g}, replicate "
            f"mean {s['rep_mean']:.3g}, Bayes p {s['bayes_p']:.2f}"
            + (" (EXTREME)" if s.get("extreme") else "")
            + "."
        )
    pit = (f.get("ppc_stats") or {}).get("loo_pit")
    if pit:
        khat = pit.get("khat") or {}
        lines.append(
            f"LOO-PIT ({pit.get('weighting')}): KS p {pit.get('ks_p', 0):.3f} "
            f"over {pit.get('n', 0)} observations — "
            + ("calibrated" if pit.get("calibrated") else "MISCALIBRATED")
            + (
                f"; Pareto k-hat > 0.7 for {khat['n_high']} obs"
                if khat.get("n_high")
                else ""
            )
            + "."
        )
    ppc = f.get("ppc_prior") or {}
    if ppc.get("coverage_90") is not None:
        lines.append(
            f"Prior predictive: 90% band covered {ppc['coverage_90'] * 100:.0f}% "
            "of observed periods."
        )
    diag = f.get("diagnostics") or {}
    if diag.get("rhat_max") is not None:
        lines.append(f"Max R-hat: {diag['rhat_max']}.")
    return "\n".join(lines)


def _clean_reply(reply: Any) -> str:
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
    return s.strip()


def _enrich_with_llm(
    f: dict[str, Any], insights: dict[str, str], llm: Any
) -> dict[str, str]:
    """Overlay LLM-written glosses on the templated fallback (best-effort)."""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception:  # noqa: BLE001
        return insights

    prompt = (
        f"Fitted MMM results facts:\n{_facts_blob(f)}\n\n"
        "Write the narrative for the interactive results report. Return "
        "EXACTLY these labelled parts, each 1–3 sentences of plain prose:\n"
        "STANDFIRST: the one-paragraph story of what this model found.\n"
        "EXEC: interpret the headline attribution / share / blended ROI.\n"
        "FIT: how well the model tracks the data, and any caveat.\n"
        "PPC_STATS: interpret the posterior-predictive test-statistic "
        "p-values — which KPI properties (extremes, volatility, "
        "autocorrelation) the model reproduces or misses.\n"
        "LOO_PIT: interpret the LOO-PIT calibration verdict — whether the "
        "per-observation predictive intervals are honest, too narrow, or "
        "too wide (only if LOO-PIT facts exist).\n"
        "DECOMPOSITION: the media-vs-baseline story and spend-vs-effect "
        "imbalances.\n"
        "ROI: which channels lead/trail and where uncertainty matters.\n"
        "YOY: what drove the year-over-year KPI change (media vs baseline).\n"
        "PATHWAYS: direct vs mediated effect routes (only if mediation "
        "facts exist).\n"
        "LATENT: what the latent factors/loadings say (only if latent facts "
        "exist).\n"
        "ESTIMANDS: average vs marginal returns — where they disagree here.\n"
        "CURVES: what the response curves imply about headroom/saturation.\n"
        "CARRYOVER: which channels have long memory and what that changes.\n"
        "PRIOR_POSTERIOR: where the data moved beliefs vs stayed prior-driven.\n"
        "REALLOCATION: how to read the (approximate) reallocation results.\n"
        "SENSITIVITY: which findings are robust vs fragile across specs.\n"
        "PRIOR_PREDICTIVE: what the pre-fit checks say about the design.\n"
        "ASSUMPTIONS: the causal caveats a decision-maker must hold.\n"
        "NEXT_STEPS: the concrete experiments/validation to run next."
    )
    try:
        r = llm.invoke([SystemMessage(content=_SYS), HumanMessage(content=prompt)])
        text = _clean_reply(r)
    except Exception:  # noqa: BLE001
        return insights

    labels = "|".join(_LLM_SLOT_LABELS)
    for m in re.finditer(
        rf"(?im)^({labels})\s*:\s*(.+?)(?=^\s*(?:{labels})\s*:|\Z)",
        text,
        flags=re.S | re.M,
    ):
        slot = _LLM_SLOT_LABELS[m.group(1).upper()]
        val = " ".join(m.group(2).split()).strip()
        if val:
            insights[slot] = val
    return insights


def build_interactive_insights(
    facts: dict[str, Any], *, llm: Any | None = None
) -> dict[str, str]:
    """Narrative for the interactive MMM Results Report.

    Always returns a complete dict (templated fallback for every slot in
    :data:`INTERACTIVE_INSIGHT_SLOTS`); an ``llm`` enriches best-effort.
    """
    insights = _fallback_insights(facts)
    if llm is not None:
        try:
            insights = _enrich_with_llm(facts, insights, llm)
        except Exception:  # noqa: BLE001
            pass
    return insights
