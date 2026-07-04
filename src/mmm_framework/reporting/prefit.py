"""Model Design Readout — the *pre-fit* sibling of the Media Performance Readout.

A pre-registration document generated **before** the model is fitted, so there
is a durable record of what was assumed, which priors were chosen, what those
priors imply (prior predictive + implied response curves), whether the
inference machinery is calibrated (SBC), and how the specification evolved
(change record) — before the final model ever saw the data's verdict.

Same editorial shell as the Augur readout (masthead, numbered sticky contents
nav, cream/ink palette, evidence chips), same two-layer prose discipline as
:mod:`mmm_framework.reporting.insights`:

1. **Templated insights** (:func:`build_prefit_insights` with ``llm=None``) —
   complete, grounded, reproducible offline.
2. **AI-generated insights** — when an LLM is supplied the standfirst and the
   section glosses are enriched, best-effort; any failure silently keeps the
   templated text.

Usage::

    from mmm_framework.reporting import PrefitReadoutGenerator

    gen = PrefitReadoutGenerator(model, sbc=sbc_result, revisions=[...])
    html = gen.generate_report()          # templated insights
    html = PrefitReadoutGenerator(model, llm=chat_model).generate_report()  # AI
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from loguru import logger

from .augur_theme import AUGUR_FONTS_LINK, MASTHEAD_LOGO_SVG, augur_css
from .charts.prior import (
    create_prior_adstock_band,
    create_prior_component_chart,
    create_prior_density_chart,
    create_prior_predictive_fan,
    create_prior_saturation_band,
    create_prior_stat_distribution,
    create_sbc_ecdf_diff,
    create_sbc_rank_histogram,
)
from .config import ChannelColors, ColorPalette, ColorScheme, ReportConfig
from .helpers.prefit import (
    PRIOR_GROUP_ORDER,
    enumerate_model_priors,
    model_assumptions,
    prior_component_facts,
    prior_estimand_facts,
    prior_predictive_facts,
    prior_response_curves,
    sample_prior,
)

__all__ = [
    "PrefitReadoutGenerator",
    "build_prefit_insights",
    "prefit_facts",
    "PREFIT_INSIGHT_SLOTS",
]


def _esc(s: Any) -> str:
    return _html.escape(str(s))


def _fmt(v: float | None, digits: int = 3) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    av = abs(v)
    if av != 0 and (av >= 10000 or av < 0.001):
        return f"{v:.2e}"
    return f"{v:,.{digits}g}" if av >= 1 else f"{v:.{digits}f}"


# ─────────────────────────────────────────────────────────────────────────────
# Facts (deterministic; everything the prose + sections read)
# ─────────────────────────────────────────────────────────────────────────────
def prefit_facts(
    model: Any,
    *,
    sbc: Any = None,
    revisions: list[dict[str, Any]] | None = None,
    n_prior_samples: int = 500,
    random_seed: int = 42,
    max_density_params: int = 12,
) -> dict[str, Any]:
    """Compute every grounded fact the Model Design Readout needs.

    Samples the prior once and derives the priors table, assumptions, prior
    predictive facts, implied response curves and density samples from it.
    ``sbc`` may be an :class:`~mmm_framework.diagnostics.sbc.SBCResult` or an
    already-serialized dashboard dict.
    """
    idata = sample_prior(model, n_prior_samples, random_seed)
    prior_ds = getattr(idata, "prior", None)

    priors = [r.to_dict() for r in enumerate_model_priors(model, prior_ds)]
    assumptions = [r.to_dict() for r in model_assumptions(model)]
    try:
        ppc = prior_predictive_facts(model, idata)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"prefit: prior predictive facts unavailable: {e}")
        ppc = None
    curves = prior_response_curves(model, prior_ds)
    try:
        components = prior_component_facts(model, idata)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"prefit: prior component facts unavailable: {e}")
        components = {}
    try:
        estimands = prior_estimand_facts(model, idata)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"prefit: prior estimand facts unavailable: {e}")
        estimands = {}

    # Density samples for the small-multiples grid: media effects first, then
    # carryover / saturation / noise — capped so the grid stays readable.
    densities: list[dict[str, Any]] = []
    if prior_ds is not None:
        chosen = [r["name"] for r in priors if r["group"] == "Media effects"]
        for group in (
            "Carryover (adstock)",
            "Saturation",
            "Observation noise",
            "Baseline",
        ):
            chosen += [r["name"] for r in priors if r["group"] == group]
        seen: set[str] = set()
        for name in chosen:
            if name in seen or name not in getattr(prior_ds, "data_vars", {}):
                continue
            seen.add(name)
            vals = np.asarray(prior_ds[name].values, dtype=float).reshape(-1)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                densities.append({"name": name, "samples": vals})
            if len(densities) >= max_density_params:
                break

    sbc_dash = None
    if sbc is not None:
        sbc_dash = sbc.to_dashboard() if hasattr(sbc, "to_dashboard") else dict(sbc)

    meta: dict[str, Any] = {
        "kpi": None,
        "channels": [str(c) for c in getattr(model, "channel_names", [])],
        "controls": [str(c) for c in getattr(model, "control_names", [])],
        "n_obs": int(getattr(model, "n_obs", 0) or 0),
        "n_geos": len(getattr(model, "geo_names", []) or []),
        "n_free_params": len(priors),
        "date_start": None,
        "date_end": None,
    }
    try:
        meta["kpi"] = str(model.mff_config.kpi.name)
    except Exception:  # noqa: BLE001
        pass
    try:
        periods = model.panel.coords.periods
        meta["date_start"] = str(periods[0])[:10]
        meta["date_end"] = str(periods[-1])[:10]
        meta["n_periods"] = len(periods)
    except Exception:  # noqa: BLE001
        meta["n_periods"] = None
    lik = getattr(getattr(model, "model_config", None), "likelihood", None)
    meta["likelihood"] = (
        getattr(getattr(lik, "family", None), "value", None) or "normal"
    )
    meta["link"] = getattr(getattr(lik, "link", None), "value", None) or "identity"

    return {
        "meta": meta,
        "assumptions": assumptions,
        "priors": priors,
        "ppc": ppc,
        "curves": curves,
        "components": components,
        "estimands": estimands,
        "densities": densities,
        "sbc": sbc_dash,
        "revisions": list(revisions or []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Insights: templated fallback + optional LLM enrichment
# ─────────────────────────────────────────────────────────────────────────────
PREFIT_INSIGHT_SLOTS = (
    "standfirst",
    "assumptions_gloss",
    "priors_gloss",
    "response_gloss",
    "components_gloss",
    "ppc_gloss",
    "estimands_gloss",
    "sbc_gloss",
    "revisions_gloss",
    "next_steps",
)


def _fallback_prefit_insights(f: dict[str, Any]) -> dict[str, str]:
    meta = f.get("meta", {})
    kpi = meta.get("kpi") or "the KPI"
    n_ch = len(meta.get("channels", []))
    ppc = f.get("ppc")
    sbc = f.get("sbc")
    revisions = f.get("revisions", [])
    n_priors = len(f.get("priors", []))
    n_calibrated = sum(1 for r in f.get("priors", []) if r.get("calibrated"))

    out: dict[str, str] = {}

    window = ""
    if meta.get("date_start") and meta.get("date_end"):
        window = f" over {meta['date_start']} – {meta['date_end']}"
    out["standfirst"] = (
        f"This document is the pre-registered design of a Bayesian marketing mix "
        f"model of {kpi} across {n_ch} media channel{'s' if n_ch != 1 else ''}"
        f"{window}. It records every structural assumption, all {n_priors} prior "
        f"distributions, what those priors imply before seeing the data's verdict, "
        f"and the checks the specification passed — so the final fitted model can "
        f"be judged against choices made in advance, not after the fact."
    )

    out["assumptions_gloss"] = (
        "Each row below is a modeling choice, not a finding. They are stated "
        "before fitting precisely so that, if a result later looks surprising, "
        "the first question — was this baked in by an assumption? — can be "
        "answered from the record."
    )

    cal = (
        f" {n_calibrated} media-effect prior"
        f"{'s are' if n_calibrated != 1 else ' is'} experiment-calibrated "
        "(anchored to randomized lift evidence)."
        if n_calibrated
        else ""
    )
    out["priors_gloss"] = (
        f"The model estimates {n_priors} free parameters. Priors are deliberately "
        f"weakly informative: they rule out the physically absurd (negative media "
        f"effects, carryover that grows over time) while leaving the magnitude "
        f"of every effect for the data to decide.{cal}"
    )

    out["response_gloss"] = (
        "Before any data, the priors already commit each channel to a concave "
        "response and a decaying carryover. The bands below show how loosely: a "
        "wide band is a prior holding the question open, a narrow band is a "
        "commitment that should be defensible on subject-matter grounds."
    )

    components = f.get("components") or {}
    if components:
        names = [str(v.get("label", k)).lower() for k, v in components.items()]
        out["components_gloss"] = (
            f"The structural pieces of the model — {', '.join(names)} — each "
            "have a prior life of their own, shown here in original KPI units "
            "per period. Wiggly individual traces are single prior draws; the "
            "band is the envelope. Trend and seasonality should look loose but "
            "plausible in magnitude; if a component's prior band dwarfs the "
            "KPI itself, the prior — not the data — will drive that part of "
            "the decomposition."
        )
    else:
        out["components_gloss"] = (
            "This specification does not expose per-component prior draws "
            "(trend / seasonality / controls), so the component-level prior "
            "check is unavailable."
        )

    estimands = f.get("estimands") or {}
    if estimands.get("channels"):
        rows = estimands["channels"]
        widest = max(rows, key=lambda r: r["upper"] - r["lower"])
        blended = estimands.get("blended")
        bits = [
            f"Before any data, the priors already imply a distribution for each "
            f"channel's return. Across {len(rows)} channels the prior "
            f"{rows[0]['label']} intervals below should be WIDE — a pre-data "
            f"interval that already excludes plausible outcomes has decided the "
            f"answer in advance"
        ]
        if blended:
            bits.append(
                f"the blended prior return centers at {blended['mean']:.2f} "
                f"(90% {blended['lower']:.2f}–{blended['upper']:.2f})"
            )
        share = estimands.get("marketing_share")
        if share:
            bits.append(
                f"and the priors put marketing's share of the KPI at "
                f"{share['mean'] * 100:.0f}% "
                f"(90% {share['lower'] * 100:.0f}%–{share['upper'] * 100:.0f}%) — "
                f"sanity-check that against what is commercially believable"
            )
        out["estimands_gloss"] = (
            "; ".join(bits) + f". The widest prior is {widest['channel']}'s "
            f"({widest['lower']:.2f}–{widest['upper']:.2f})."
        )
    else:
        out["estimands_gloss"] = (
            "Prior estimand distributions are unavailable for this "
            "specification (no channel-contribution deterministic in the "
            "graph), so the pre-data ROI check is skipped."
        )

    if ppc:
        cov = ppc.get("coverage_90")
        neg = ppc.get("frac_negative", 0.0)
        bits = [
            (
                f"Across {ppc.get('n_draws', 0)} datasets simulated purely from the "
                f"priors, the 90% prior-predictive band covers "
                f"{cov * 100:.0f}% of the observed series"
                if cov is not None and np.isfinite(cov)
                else "Prior-predictive simulation summarizes what the priors deem possible"
            )
        ]
        zbar = ppc.get("scale_z_abs_mean")
        if zbar is not None and np.isfinite(zbar):
            if zbar <= 1.0:
                bits.append(
                    f"On the original scale the observed series sits well inside "
                    f"the prior's typical spread ({zbar:.1f} prior-sd from the "
                    f"prior median on average)"
                )
            else:
                bits.append(
                    f"On the original scale the observed series runs "
                    f"{zbar:.1f} prior-sd from the prior median on average — the "
                    f"prior's location or scale is off; revisit the intercept/"
                    f"noise priors before fitting"
                )
        if neg > 0.05:
            bits.append(
                f"{neg * 100:.0f}% of simulated outcomes are negative — wide, but "
                "tolerable if the likelihood scale is otherwise sane; tighten the "
                "intercept/noise priors if this grows"
            )
        else:
            bits.append(f"only {neg * 100:.1f}% of simulated outcomes are negative")
        out["ppc_gloss"] = (
            ". ".join(b[0].upper() + b[1:] for b in bits)
            + ". A healthy prior wraps the observed data "
            "loosely — neither excluding it nor allowing absurd magnitudes."
        )
    else:
        out["ppc_gloss"] = (
            "Prior-predictive simulation was not available for this "
            "specification; run it before the final fit."
        )

    if sbc:
        n_params = len(sbc.get("params", []))
        bad = [p for p in sbc.get("params", []) if not p.get("calibrated", True)]
        if sbc.get("all_calibrated"):
            out["sbc_gloss"] = (
                f"Simulation-based calibration refit the model on "
                f"{sbc.get('n_sims_effective', '?')} datasets drawn from its own "
                f"priors; all {n_params} checked parameters recover their known "
                "values with calibrated uncertainty. The inference machinery is "
                "trustworthy for this specification."
            )
        else:
            worst = ", ".join(p.get("name", "?") for p in bad[:4])
            out["sbc_gloss"] = (
                f"Simulation-based calibration flags "
                f"{len(bad)} of {n_params} checked parameters ({worst}) as "
                "miscalibrated — the sampler cannot fully recover known values "
                "for them. Interpret those posteriors with caution, or revise "
                "the specification before the final fit."
            )
    else:
        out["sbc_gloss"] = (
            "Simulation-based calibration has not been run for this "
            "specification yet. It is the strongest pre-fit check available — "
            "it verifies the machinery can recover known answers — and is "
            "recommended before the final fit."
        )

    if revisions:
        out["revisions_gloss"] = (
            f"The specification below is revision {len(revisions)} of this "
            "model. Every change is listed with its rationale so the final "
            "model's provenance is auditable."
        )
    else:
        out["revisions_gloss"] = (
            "This is the initial specification — no revisions have been "
            "recorded against it yet. Subsequent changes should be logged here "
            "with their rationale before re-fitting."
        )

    out["next_steps"] = (
        "With the design recorded, the next step is the fit itself, followed by "
        "convergence diagnostics, posterior-predictive checks and the Media "
        "Performance Readout. Any change to priors or structure after seeing "
        "results should come back through this document first."
    )
    return out


_PREFIT_SYS = (
    "You are a senior Bayesian statistician writing the narrative for a "
    "pre-registration document of a marketing mix model (a 'model design "
    "readout' produced BEFORE fitting). Audience: an analytics lead and a "
    "skeptical reviewer. Be precise, plain-spoken and honest about what is an "
    "assumption vs. what is evidence. Ground every claim strictly in the facts "
    "provided — never invent numbers. No markdown, no preamble."
)


def _prefit_facts_blob(f: dict[str, Any]) -> str:
    meta = f.get("meta", {})
    lines = [
        f"KPI: {meta.get('kpi')}. Channels: {', '.join(meta.get('channels', []))}.",
        f"Observations: {meta.get('n_obs')} rows, {meta.get('n_periods')} periods, "
        f"{meta.get('n_geos')} geographies. Likelihood: {meta.get('likelihood')} "
        f"({meta.get('link')} link).",
        f"Free parameters: {meta.get('n_free_params')}.",
    ]
    for a in f.get("assumptions", []):
        lines.append(f"Assumption — {a['topic']}: {a['setting']}.")
    ppc = f.get("ppc")
    if ppc:
        cov = ppc.get("coverage_90")
        lines.append(
            f"Prior predictive: 90% band covers {cov * 100:.0f}% of observed periods; "
            f"{ppc.get('frac_negative', 0) * 100:.1f}% of simulated outcomes negative "
            f"({ppc.get('n_draws')} draws)."
            if cov is not None and np.isfinite(cov)
            else "Prior predictive summary unavailable."
        )
        zbar = ppc.get("scale_z_abs_mean")
        if zbar is not None and np.isfinite(zbar):
            lines.append(
                f"Original-scale fit of the prior: observed series averages "
                f"{zbar:.1f} prior-sd from the prior median."
            )
    comps = f.get("components") or {}
    for key, comp in comps.items():
        lines.append(
            f"Prior component — {comp.get('label', key)}: typical magnitude "
            f"{comp.get('abs_scale', 0):,.0f} KPI units/period."
        )
    est = f.get("estimands") or {}
    for r in est.get("channels", []):
        lines.append(
            f"Prior estimand — {r['channel']} {r['label']}: mean {r['mean']:.2f} "
            f"(90% {r['lower']:.2f}–{r['upper']:.2f}), "
            f"P(>{r['reference']:g}) = {r['p_above_reference']:.0%}."
        )
    if est.get("blended"):
        b = est["blended"]
        lines.append(
            f"Prior blended return: {b['mean']:.2f} (90% {b['lower']:.2f}–{b['upper']:.2f})."
        )
    if est.get("marketing_share"):
        s = est["marketing_share"]
        lines.append(
            f"Prior marketing share of KPI: {s['mean']:.0%} "
            f"(90% {s['lower']:.0%}–{s['upper']:.0%})."
        )
    sbc = f.get("sbc")
    if sbc:
        bad = [p for p in sbc.get("params", []) if not p.get("calibrated", True)]
        lines.append(
            f"SBC: {'all calibrated' if sbc.get('all_calibrated') else 'MISCALIBRATED'} "
            f"({sbc.get('n_sims_effective')} sims, {len(sbc.get('params', []))} params"
            + (
                f"; flagged: {', '.join(p.get('name', '?') + ' ' + p.get('shape', '') for p in bad[:5])}"
                if bad
                else ""
            )
            + ")."
        )
    else:
        lines.append("SBC: not run.")
    revs = f.get("revisions", [])
    if revs:
        lines.append(f"Revisions recorded: {len(revs)}.")
        for r in revs[-5:]:
            lines.append(
                f"  - {r.get('date', '?')}: {r.get('change', '?')}"
                + (f" (rationale: {r['rationale']})" if r.get("rationale") else "")
            )
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


_LLM_SLOT_LABELS = {
    "STANDFIRST": "standfirst",
    "ASSUMPTIONS": "assumptions_gloss",
    "PRIORS": "priors_gloss",
    "COMPONENTS": "components_gloss",
    "PRIOR_PREDICTIVE": "ppc_gloss",
    "PRIOR_ESTIMANDS": "estimands_gloss",
    "SBC": "sbc_gloss",
    "NEXT_STEPS": "next_steps",
}


def _enrich_prefit_with_llm(
    f: dict[str, Any], insights: dict[str, str], llm: Any
) -> dict[str, str]:
    """Overlay LLM-written section glosses on the templated fallback (best-effort)."""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception:  # noqa: BLE001
        return insights

    blob = _prefit_facts_blob(f)
    prompt = (
        f"Model design facts:\n{blob}\n\n"
        "Write the narrative for the pre-fit design readout. Return EXACTLY "
        "these labelled parts, each 1–3 sentences of plain prose:\n"
        "STANDFIRST: what this model is and why the design is recorded pre-fit.\n"
        "ASSUMPTIONS: how a reviewer should read the structural assumptions.\n"
        "PRIORS: what the priors commit to vs. leave open (mention calibrated "
        "priors only if the facts show any).\n"
        "COMPONENTS: interpret the prior trend/seasonality/control/media "
        "magnitudes over time — are they plausibly scaled vs. the KPI?\n"
        "PRIOR_PREDICTIVE: interpret the coverage / original-scale z / "
        "negativity numbers.\n"
        "PRIOR_ESTIMANDS: what the priors already say about channel returns "
        "and marketing's share — are they suitably open-minded?\n"
        "SBC: interpret the calibration verdict (or urge running it if absent).\n"
        "NEXT_STEPS: what happens between this document and the final fit."
    )
    try:
        r = llm.invoke(
            [SystemMessage(content=_PREFIT_SYS), HumanMessage(content=prompt)]
        )
        text = _clean_reply(r)
    except Exception:  # noqa: BLE001
        return insights

    # Parse labelled parts; unknown labels are ignored, missing ones keep the
    # templated fallback.
    import re

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


def build_prefit_insights(
    facts: dict[str, Any], *, llm: Any | None = None
) -> dict[str, str]:
    """Narrative for the Model Design Readout.

    Always returns a complete dict (templated fallback for every slot in
    :data:`PREFIT_INSIGHT_SLOTS`). When ``llm`` is provided the standfirst and
    section glosses are enriched, best-effort — any failure degrades to the
    templated text.
    """
    insights = _fallback_prefit_insights(facts)
    if llm is not None:
        try:
            insights = _enrich_prefit_with_llm(facts, insights, llm)
        except Exception:  # noqa: BLE001
            pass
    return insights


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class _NavEntry:
    section_id: str
    title: str


class PrefitReadoutGenerator:
    """Generate the pre-fit **Model Design Readout** HTML document.

    Parameters
    ----------
    model:
        A configured (typically *unfitted*) ``BayesianMMM``. May be ``None``
        when ``facts`` is supplied directly (tests / cached facts).
    config:
        A :class:`~mmm_framework.reporting.config.ReportConfig`; only the
        masthead fields (title/client/subtitle/analysis_period/confidential),
        ``color_scheme`` and the Plotly embedding knobs are read. Defaults to
        the Augur palette with the title "Model Design Readout".
    sbc:
        Optional ``SBCResult`` or its ``to_dashboard()`` dict. When absent and
        ``run_sbc`` is true, SBC is run here (see below).
    revisions:
        Optional change record: a list of ``{date, author, change, rationale}``
        dicts, oldest first.
    llm:
        Optional LangChain chat model. When given, section prose is
        AI-enriched (grounded in the computed facts); ``None`` keeps the
        templated narrative.
    facts:
        Precomputed :func:`prefit_facts` output — skips model access entirely.
    run_sbc:
        When true (the default) and no ``sbc`` result was supplied, a
        simulation-based-calibration smoke run (``sbc_kwargs``, default
        ``n_sims=20, L=50``) is executed against the model before rendering —
        the readout should never silently omit the strongest pre-fit check.
        EXPENSIVE (one refit per simulation); pass ``run_sbc=False`` for a
        fast render or supply a stored result. Only applies when a ``model``
        is given; failures degrade to the "not yet run" section.
    """

    #: Modest smoke-run defaults — enough to catch gross miscalibration while
    #: keeping the default render tractable. Serialized WITH raw ranks so the
    #: ECDF-difference panel renders.
    DEFAULT_SBC_KWARGS = {"n_sims": 20, "L": 50}

    def __init__(
        self,
        model: Any = None,
        *,
        config: ReportConfig | None = None,
        sbc: Any = None,
        revisions: list[dict[str, Any]] | None = None,
        llm: Any | None = None,
        facts: dict[str, Any] | None = None,
        n_prior_samples: int = 500,
        random_seed: int = 42,
        run_sbc: bool = True,
        sbc_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if model is None and facts is None:
            raise ValueError(
                "PrefitReadoutGenerator needs a model or precomputed facts."
            )
        self.model = model
        self.config = config or ReportConfig(
            title="Model Design Readout",
            color_scheme=ColorScheme.from_palette(ColorPalette.AUGUR),
            confidential=True,
        )
        self.llm = llm
        self.channel_colors = ChannelColors()
        if facts is not None:
            self.facts = facts
            if sbc is not None and not facts.get("sbc"):
                self.facts["sbc"] = (
                    sbc.to_dashboard() if hasattr(sbc, "to_dashboard") else dict(sbc)
                )
            if revisions:
                self.facts["revisions"] = list(revisions)
        else:
            self.facts = prefit_facts(
                model,
                sbc=sbc,
                revisions=revisions,
                n_prior_samples=n_prior_samples,
                random_seed=random_seed,
            )
        if run_sbc and model is not None and not self.facts.get("sbc"):
            self.facts["sbc"] = self._run_sbc(model, sbc_kwargs)
        self.insights = build_prefit_insights(self.facts, llm=llm)

    @staticmethod
    def _run_sbc(model: Any, sbc_kwargs: dict[str, Any] | None) -> dict | None:
        """Default SBC smoke run — best-effort, serialized with raw ranks."""
        kwargs = {**PrefitReadoutGenerator.DEFAULT_SBC_KWARGS, **(sbc_kwargs or {})}
        try:
            from ..diagnostics.sbc import run_mmm_sbc

            result = run_mmm_sbc(model, **kwargs)
            dash = result.to_dashboard()
            dash["params"] = [
                p.to_dashboard(max_ranks=len(p.int_ranks)) for p in result.params
            ]
            return dash
        except Exception as e:  # noqa: BLE001
            logger.warning(f"prefit: default SBC run failed: {e}")
            return None

    # ── public API ──────────────────────────────────────────────────────────
    def generate_report(self) -> str:
        """Render the full HTML document."""
        sections: list[tuple[_NavEntry, str]] = []
        for builder in (
            self._section_purpose,
            self._section_spec,
            self._section_assumptions,
            self._section_priors,
            self._section_response,
            self._section_components,
            self._section_ppc,
            self._section_estimands,
            self._section_sbc,
            self._section_revisions,
            self._section_signoff,
        ):
            entry = builder()
            if entry is not None:
                sections.append(entry)
        return self._assemble(sections)

    def save_report(self, path: str) -> str:
        html_doc = self.generate_report()
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html_doc)
        return path

    # ── section helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _wrap(sec_id: str, eyebrow: str, title: str, body: str) -> str:
        return (
            f'<section class="section" id="{sec_id}">'
            f'<div class="section-eyebrow">{_esc(eyebrow)}</div>'
            f"<h2>{_esc(title)}</h2>{body}</section>"
        )

    @staticmethod
    def _chip(kind: str, label: str) -> str:
        return f'<span class="tier-chip t-{kind}">{_esc(label)}</span>'

    def _insight(self, key: str) -> str:
        return _esc(self.insights.get(key, ""))

    # ── sections ─────────────────────────────────────────────────────────────
    def _section_purpose(self) -> tuple[_NavEntry, str]:
        meta = self.facts["meta"]
        ppc = self.facts.get("ppc")
        sbc = self.facts.get("sbc")

        kpis = []
        n_ch = len(meta.get("channels", []))
        kpis.append(
            f'<div class="kpi"><div class="label">Media channels</div>'
            f'<div class="value">{n_ch}</div>'
            f'<div class="ci">{_esc(", ".join(meta.get("channels", [])[:6]))}'
            f'{"…" if n_ch > 6 else ""}</div></div>'
        )
        window = (
            f"{meta.get('date_start', '?')} – {meta.get('date_end', '?')}"
            if meta.get("date_start")
            else "n/a"
        )
        kpis.append(
            f'<div class="kpi"><div class="label">Observation window</div>'
            f'<div class="value">{meta.get("n_periods") or "—"}</div>'
            f'<div class="ci">periods · {_esc(window)}</div></div>'
        )
        kpis.append(
            f'<div class="kpi"><div class="label">Free parameters</div>'
            f'<div class="value">{meta.get("n_free_params", 0)}</div>'
            f'<div class="ci">priors enumerated below</div></div>'
        )

        chips = []
        if ppc:
            cov = ppc.get("coverage_90")
            if cov is not None and np.isfinite(cov):
                chips.append(
                    self._chip("scale", f"Prior covers data ({cov * 100:.0f}%)")
                    if cov >= 0.8
                    else self._chip("test", f"Prior coverage {cov * 100:.0f}%")
                )
            if ppc.get("frac_negative", 0) > 0.05:
                chips.append(
                    self._chip(
                        "reduce",
                        f"{ppc['frac_negative'] * 100:.0f}% negative prior draws",
                    )
                )
        if sbc:
            chips.append(
                self._chip("scale", "Inference engine calibrated (SBC)")
                if sbc.get("all_calibrated")
                else self._chip("reduce", "SBC flags miscalibration")
            )
        else:
            chips.append(self._chip("test", "SBC not yet run"))
        chip_row = (
            '<div style="display:flex;gap:.5rem;flex-wrap:wrap;margin-top:1rem">'
            + "".join(chips)
            + "</div>"
            if chips
            else ""
        )

        body = (
            f'<p class="lede">{self._insight("standfirst")}</p>'
            f'<div class="kpi-grid">{"".join(kpis)}</div>'
            f"{chip_row}"
            '<div class="rec"><h4>Why pre-register a model design</h4><ul>'
            '<li><span class="marker"></span><span><b>It separates choices from '
            "findings</b> — assumptions and priors are on record before the data "
            "could tempt anyone to bend them.</span></li>"
            '<li><span class="marker"></span><span><b>It makes the final model '
            "auditable</b> — every change between this document and the fitted "
            "model must be justified in the change record.</span></li>"
            '<li><span class="marker"></span><span><b>It catches broken designs '
            "early</b> — prior predictive and calibration checks fail cheaply "
            "here, not expensively after the fit.</span></li>"
            "</ul></div>"
        )
        return _NavEntry("purpose", "Why this document"), self._wrap(
            "purpose",
            "Pre-registration",
            "The model design, on record before the fit",
            body,
        )

    def _section_spec(self) -> tuple[_NavEntry, str]:
        meta = self.facts["meta"]
        rows = [
            ("KPI", meta.get("kpi") or "—"),
            ("Media channels", ", ".join(meta.get("channels", [])) or "—"),
            ("Controls", ", ".join(meta.get("controls", [])) or "none"),
            ("Observations", f"{meta.get('n_obs', 0):,} rows"),
            (
                "Panel",
                f"{meta.get('n_periods') or '—'} periods × "
                f"{max(meta.get('n_geos', 0), 1)} geograph"
                f"{'ies' if meta.get('n_geos', 0) > 1 else 'y'}",
            ),
            ("Likelihood", f"{meta.get('likelihood')} ({meta.get('link')} link)"),
        ]
        table = (
            '<table class="data-table"><thead><tr><th>Component</th>'
            "<th>Specification</th></tr></thead><tbody>"
            + "".join(
                f'<tr><td class="chname">{_esc(k)}</td><td>{_esc(v)}</td></tr>'
                for k, v in rows
            )
            + "</tbody></table>"
        )
        body = (
            "<p>The scope of the model: what is being explained, by what, over "
            "what window. Everything downstream — priors, checks, the eventual "
            "readout — refers to this specification.</p>" + table
        )
        return _NavEntry("spec", "The model at a glance"), self._wrap(
            "spec", "Specification", "What is being modeled", body
        )

    def _section_assumptions(self) -> tuple[_NavEntry, str]:
        rows_html = []
        for a in self.facts.get("assumptions", []):
            chs = ""
            if a.get("channels"):
                shown = ", ".join(a["channels"][:5])
                more = f" +{len(a['channels']) - 5}" if len(a["channels"]) > 5 else ""
                chs = f'<div class="note" style="margin-top:.25rem">{_esc(shown + more)}</div>'
            rows_html.append(
                f'<tr><td class="chname">{_esc(a["topic"])}</td>'
                f'<td class="mono">{_esc(a["setting"])}</td>'
                f"<td>{_esc(a['detail'])}{chs}</td></tr>"
            )
        table = (
            '<table class="data-table"><thead><tr><th>Assumption</th>'
            "<th>Setting</th><th>What it means</th></tr></thead>"
            f'<tbody>{"".join(rows_html)}</tbody></table>'
        )
        body = f'<p class="lede">{self._insight("assumptions_gloss")}</p>{table}'
        return _NavEntry("assumptions", "Structural assumptions"), self._wrap(
            "assumptions", "Structural assumptions", "What we are assuming", body
        )

    def _section_priors(self) -> tuple[_NavEntry, str]:
        priors = self.facts.get("priors", [])
        by_group: dict[str, list[dict]] = {}
        for r in priors:
            by_group.setdefault(r["group"], []).append(r)

        rows_html: list[str] = []
        for group in PRIOR_GROUP_ORDER:
            grp = by_group.get(group)
            if not grp:
                continue
            rows_html.append(
                f'<tr><td colspan="6" style="background:var(--cream-100);'
                "font-weight:600;color:var(--ink-600);font-size:.78rem;"
                'text-transform:uppercase;letter-spacing:.06em">'
                f"{_esc(group)}</td></tr>"
            )
            for r in grp:
                rng = (
                    f"{_fmt(r['lower'])} – {_fmt(r['upper'])}"
                    if r.get("lower") is not None
                    else "—"
                )
                note = self._chip("scale", "calibrated") if r.get("calibrated") else ""
                dims = (
                    f' <span class="note">({_esc(r["dims"])})</span>'
                    if r.get("dims")
                    else ""
                )
                rows_html.append(
                    f'<tr><td class="mono">{_esc(r["name"])}{dims}</td>'
                    f"<td>{_esc(r['family'])}</td>"
                    f'<td class="mono">{_fmt(r.get("mean"))}</td>'
                    f'<td class="mono">{_fmt(r.get("sd"))}</td>'
                    f'<td class="mono">{rng}</td><td>{note}</td></tr>'
                )
        table = (
            '<table class="data-table"><thead><tr><th>Parameter</th><th>Family</th>'
            "<th>Prior mean</th><th>SD</th><th>90% range</th><th></th></tr></thead>"
            f'<tbody>{"".join(rows_html)}</tbody></table>'
        )

        density_grid = ""
        densities = self.facts.get("densities", [])
        if densities:
            cells = []
            for i, d in enumerate(densities):
                chart = create_prior_density_chart(
                    d["name"],
                    d["samples"],
                    self.config,
                    div_id=f"priorDensity_{i}",
                    color=self.config.color_scheme.primary_dark,
                )
                cells.append(f'<div class="sat-cell">{chart}</div>')
            density_grid = (
                "<h3>Key priors, as distributions</h3>"
                "<p>Prior mass is what the model believes before evidence; the "
                "table above summarizes it, these curves show it.</p>"
                f'<div class="sat-grid">{"".join(cells)}</div>'
            )

        body = (
            f'<p class="lede">{self._insight("priors_gloss")}</p>'
            f"{table}"
            '<p class="note" style="margin-top:.85rem">Prior mean / SD / range are '
            "empirical, from the same prior draws used in the predictive checks "
            "below. Vector parameters (per-geo effects) are summarized across "
            "elements.</p>"
            f"{density_grid}"
        )
        return _NavEntry("priors", "The priors, in full"), self._wrap(
            "priors", "Priors", "Every prior the model will fit under", body
        )

    def _section_response(self) -> tuple[_NavEntry, str] | None:
        curves = self.facts.get("curves", {})
        if not curves:
            return None
        sat_cells, ad_cells = [], []
        for i, (ch, entry) in enumerate(curves.items()):
            color = self.channel_colors.get(ch)
            if "saturation" in entry:
                sat_cells.append(
                    '<div class="sat-cell">'
                    + create_prior_saturation_band(
                        ch,
                        entry["saturation"],
                        self.config,
                        div_id=f"priorSat_{i}",
                        color=color,
                    )
                    + "</div>"
                )
            if "adstock" in entry:
                ad_cells.append(
                    '<div class="sat-cell">'
                    + create_prior_adstock_band(
                        ch,
                        entry["adstock"],
                        self.config,
                        div_id=f"priorAdstock_{i}",
                        color=color,
                    )
                    + "</div>"
                )
        if not sat_cells and not ad_cells:
            return None
        parts = [f'<p class="lede">{self._insight("response_gloss")}</p>']
        if sat_cells:
            parts.append(
                "<h3>Saturation, before data</h3>"
                '<div class="sat-grid">' + "".join(sat_cells) + "</div>"
                '<p class="chart-caption">Median prior response (line) and 90% '
                "prior band per channel, over spend as a share of the observed "
                "maximum. The fitted curve must land inside what these priors "
                "allow.</p>"
            )
        if ad_cells:
            parts.append(
                "<h3>Carryover, before data</h3>"
                '<div class="sat-grid">' + "".join(ad_cells) + "</div>"
                '<p class="chart-caption">Prior-implied share of a period’s '
                "effect remaining at each later lag (median and 90% band).</p>"
            )
        return _NavEntry("response", "Implied response shapes"), self._wrap(
            "response",
            "Implied response",
            "The response shapes the priors imply",
            "".join(parts),
        )

    _COMPONENT_COLORS = {
        "trend": "#4a6d8a",  # steel
        "seasonality": "#b8860b",  # gold
        "controls": "#7d6a96",  # violet
        "media": "#5a7a3a",  # sage
    }

    def _section_components(self) -> tuple[_NavEntry, str] | None:
        components = self.facts.get("components") or {}
        if not components:
            return None
        cells = []
        for i, (key, comp) in enumerate(components.items()):
            chart = create_prior_component_chart(
                str(comp.get("label", key)),
                comp,
                self.config,
                div_id=f"priorComponent_{i}",
                color=self._COMPONENT_COLORS.get(key),
            )
            cells.append(f'<div class="chart-card">{chart}</div>')
        body = (
            f'<p class="lede">{self._insight("components_gloss")}</p>'
            f'<div class="chart-grid-2">{"".join(cells)}</div>'
            '<p class="chart-caption">Each panel is that component\'s prior '
            "contribution to the KPI per period, in original units: the 90% "
            "prior band, its median, and a few individual prior draws (thin "
            "traces). The dashed line is zero — a component whose band strays "
            "far from plausible magnitudes is a prior problem to fix before "
            "fitting, not a finding.</p>"
        )
        return _NavEntry("components", "Priors over time"), self._wrap(
            "components",
            "Structural priors in time",
            "What the priors say about trend, seasonality and drivers",
            body,
        )

    def _section_ppc(self) -> tuple[_NavEntry, str]:
        ppc = self.facts.get("ppc")
        if not ppc:
            body = f"<p>{self._insight('ppc_gloss')}</p>"
            return _NavEntry("prior-predictive", "Prior predictive checks"), self._wrap(
                "prior-predictive",
                "Prior predictive checks",
                "Could this model have produced our data?",
                body,
            )

        fan = create_prior_predictive_fan(
            list(ppc["dates"]),
            ppc["observed"],
            ppc["bands"],
            self.config,
            div_id="priorPredictiveFan",
            kpi_label=str(ppc.get("kpi_label", "KPI")),
            sample_traces=ppc.get("traces"),
        )
        mean_hist = create_prior_stat_distribution(
            ppc["rep_means"],
            ppc["obs_mean"],
            self.config,
            div_id="priorRepMeans",
            stat_label="replicate mean",
        )
        sd_hist = create_prior_stat_distribution(
            ppc["rep_sds"],
            ppc["obs_sd"],
            self.config,
            div_id="priorRepSds",
            stat_label="replicate std. dev.",
        )

        cov = ppc.get("coverage_90")
        neg = ppc.get("frac_negative", 0.0)
        zbar = ppc.get("scale_z_abs_mean")
        z_cell = (
            f'<div class="kpi"><div class="label">Original-scale distance</div>'
            f'<div class="value">{zbar:.1f}σ</div>'
            f'<div class="ci">observed vs prior median, in prior-sd units</div></div>'
            if zbar is not None and np.isfinite(zbar)
            else f'<div class="kpi"><div class="label">Negative simulated outcomes</div>'
            f'<div class="value">{neg * 100:.1f}%</div>'
            f'<div class="ci">of all prior-predictive draws</div></div>'
        )
        kpis = (
            '<div class="kpi-grid">'
            f'<div class="kpi"><div class="label">90% band coverage of observed</div>'
            f'<div class="value">{cov * 100:.0f}%</div>'
            f'<div class="ci">of {len(ppc["dates"])} periods inside the prior band</div></div>'
            f"{z_cell}"
            f'<div class="kpi"><div class="label">Simulated datasets</div>'
            f'<div class="value">{ppc.get("n_draws", 0)}</div>'
            f'<div class="ci">{neg * 100:.1f}% of draws negative</div></div>'
            "</div>"
            if cov is not None and np.isfinite(cov)
            else ""
        )

        body = (
            f'<p class="lede">{self._insight("ppc_gloss")}</p>'
            f"{kpis}"
            f'<div class="chart-card">{fan}</div>'
            '<p class="chart-caption">Observed KPI (dark line, original scale) '
            "against the prior-predictive fan: the 50% and 90% bands of datasets "
            "simulated before any fitting, with individual prior draws as thin "
            "traces. The prior should wrap the data loosely — and single draws "
            "should look like plausible (if noisy) histories of the business.</p>"
            "<h3>Replicate statistics vs. the observed data</h3>"
            '<div class="chart-grid-2">'
            f'<div class="chart-card">{mean_hist}</div>'
            f'<div class="chart-card">{sd_hist}</div>'
            "</div>"
            '<p class="chart-caption">Where the observed mean and variability '
            "fall within what the priors deem possible. An observed value in "
            "the far tail means the priors fight the data before the fit even "
            "starts.</p>"
        )
        return _NavEntry("prior-predictive", "Prior predictive checks"), self._wrap(
            "prior-predictive",
            "Prior predictive checks",
            "Could this model have produced our data?",
            body,
        )

    def _section_estimands(self) -> tuple[_NavEntry, str] | None:
        est = self.facts.get("estimands") or {}
        rows = est.get("channels") or []
        if not rows:
            return None

        # Headline KPI cards: blended prior return + prior marketing share.
        cards = []
        blended = est.get("blended")
        if blended:
            cards.append(
                f'<div class="kpi"><div class="label">Blended prior return / $1</div>'
                f'<div class="value">{blended["mean"]:.2f}</div>'
                f'<div class="ci">90% {blended["lower"]:.2f} – {blended["upper"]:.2f}'
                f'{" · monetary channels only" if blended.get("partial") else ""}</div></div>'
            )
        share = est.get("marketing_share")
        if share:
            cards.append(
                f'<div class="kpi"><div class="label">Prior marketing share of KPI</div>'
                f'<div class="value">{share["mean"] * 100:.0f}%</div>'
                f'<div class="ci">90% {share["lower"] * 100:.0f}% – '
                f'{share["upper"] * 100:.0f}%</div></div>'
            )
        cards.append(
            f'<div class="kpi"><div class="label">Channels checked</div>'
            f'<div class="value">{len(rows)}</div>'
            f'<div class="ci">prior contribution ÷ observed spend/volume</div></div>'
        )
        kpis = f'<div class="kpi-grid">{"".join(cards)}</div>'

        table_rows = []
        for r in rows:
            table_rows.append(
                f'<tr><td class="chname">{_esc(r["channel"])}</td>'
                f"<td>{_esc(r['label'])}</td>"
                f'<td class="mono">{r["mean"]:.2f}</td>'
                f'<td class="mono">{r["lower"]:.2f} – {r["upper"]:.2f}</td>'
                f'<td class="mono">{r["p_above_reference"] * 100:.0f}%</td></tr>'
            )
        table = (
            '<table class="data-table"><thead><tr><th>Channel</th><th>Metric</th>'
            "<th>Prior mean</th><th>90% range</th><th>P(above break-even)</th>"
            f'</tr></thead><tbody>{"".join(table_rows)}</tbody></table>'
        )

        cells = []
        for i, r in enumerate(rows):
            draws = r.get("draws")
            if draws is None:
                continue
            chart = create_prior_density_chart(
                f"{r['channel']}",
                draws,
                self.config,
                div_id=f"priorEstimand_{i}",
                color=self.channel_colors.get(str(r["channel"])),
                reference=r["reference"],
                reference_label=(
                    "break-even" if r.get("is_monetary", True) else "no effect"
                ),
            )
            cells.append(f'<div class="sat-cell">{chart}</div>')
        grid = (
            "<h3>Prior return per channel, as distributions</h3>"
            f'<div class="sat-grid">{"".join(cells)}</div>'
            '<p class="chart-caption">The prior distribution of each channel\'s '
            "return (total prior contribution divided by its observed spend or "
            "volume — the same semantics as the fitted contribution-ROI "
            "estimand). Before data, these should be wide and straddle the "
            "dashed break-even line; a prior that already excludes plausible "
            "returns has pre-decided the answer.</p>"
            if cells
            else ""
        )

        body = (
            f'<p class="lede">{self._insight("estimands_gloss")}</p>'
            f"{kpis}{table}{grid}"
            '<p class="note" style="margin-top:.85rem">All values are in '
            "original units (KPI per unit spend / volume), computed from the "
            "same prior draws as the predictive checks above. These are the "
            "numbers the fitted readout will report — shown here first under "
            "the priors alone, so nobody can later mistake a prior artifact "
            "for evidence.</p>"
        )
        return _NavEntry("prior-estimands", "Priors on the estimands"), self._wrap(
            "prior-estimands",
            "Prior estimands",
            "What the priors already say about returns",
            body,
        )

    def _section_sbc(self) -> tuple[_NavEntry, str]:
        sbc = self.facts.get("sbc")
        if not sbc:
            body = (
                f"<p class='lede'>{self._insight('sbc_gloss')}</p>"
                "<p class='note'>Run it via the validation workspace "
                "(<span class='mono'>simulation_based_calibration</span>) or the "
                "agent's <span class='mono'>run_calibration_check</span> tool, "
                "then regenerate this readout to include the verdict.</p>"
            )
            return _NavEntry("sbc", "Calibration (SBC)"), self._wrap(
                "sbc",
                "Simulation-based calibration",
                "Can the machinery recover known answers?",
                body,
            )

        params = list(sbc.get("params", []))
        params.sort(key=lambda p: p.get("miscalibration", 0.0), reverse=True)

        rows_html = []
        for p in params:
            ok = p.get("calibrated", False)
            chip = (
                self._chip("scale", "calibrated")
                if ok
                else self._chip("reduce", p.get("shape", "miscalibrated"))
            )
            pv = p.get("chi2_pvalue")
            rows_html.append(
                f'<tr><td class="mono">{_esc(p.get("name", "?"))}</td>'
                f"<td>{chip}</td>"
                f'<td class="mono">{_fmt(pv) if pv is not None else "—"}</td>'
                f'<td class="mono">{_fmt(p.get("miscalibration"))}</td>'
                f'<td class="mono">{_fmt(p.get("bias_z"), 2)}</td>'
                f'<td class="mono">{_fmt(p.get("dispersion_z"), 2)}</td></tr>'
            )
        table = (
            '<table class="data-table"><thead><tr><th>Parameter</th><th>Verdict</th>'
            "<th>χ² p-value</th><th>Miscalibration</th><th>Bias z</th>"
            "<th>Dispersion z</th></tr></thead>"
            f'<tbody>{"".join(rows_html)}</tbody></table>'
        )

        # Chart grid: worst offenders first (all when few, else the flagged +
        # worst up to 6).
        to_plot = [p for p in params if not p.get("calibrated", True)][:6]
        if len(to_plot) < min(6, len(params)):
            for p in params:
                if p not in to_plot:
                    to_plot.append(p)
                if len(to_plot) >= min(6, len(params)):
                    break
        cells = []
        for i, p in enumerate(to_plot):
            hist = create_sbc_rank_histogram(p, self.config, div_id=f"sbcHist_{i}")
            if hist:
                cells.append(f'<div class="sat-cell">{hist}</div>')
            ecdf = create_sbc_ecdf_diff(p, self.config, div_id=f"sbcEcdf_{i}")
            if ecdf:
                cells.append(f'<div class="sat-cell">{ecdf}</div>')
        grid = (
            f'<div class="sat-grid">{"".join(cells)}</div>'
            '<p class="chart-caption">Rank histograms of the true parameter '
            "within its posterior draws across simulated refits. Calibrated "
            "inference is flat inside the grey band; a ∪ shape means the "
            "posterior is overconfident, a ∩ shape too wide, skew means bias "
            "(Talts et al. 2018).</p>"
            if cells
            else ""
        )

        meta_line = (
            f"{sbc.get('n_sims_effective', '?')} simulations · "
            f"L={sbc.get('L', '?')} posterior draws per refit · "
            f"sampler {_esc(str(sbc.get('sampler', '?')))}"
        )
        caveats = "".join(
            f'<li><span class="marker"></span><span>{_esc(c)}</span></li>'
            for c in sbc.get("caveats", [])
        )
        caveat_html = (
            f'<div class="rec" style="margin-top:1rem"><h4>Caveats</h4><ul>{caveats}</ul></div>'
            if caveats
            else ""
        )

        body = (
            f'<p class="lede">{self._insight("sbc_gloss")}</p>'
            f'<p class="note">{meta_line}</p>'
            f"{table}{grid}{caveat_html}"
        )
        return _NavEntry("sbc", "Calibration (SBC)"), self._wrap(
            "sbc",
            "Simulation-based calibration",
            "Can the machinery recover known answers?",
            body,
        )

    def _section_revisions(self) -> tuple[_NavEntry, str]:
        revisions = self.facts.get("revisions", [])
        if revisions:
            rows_html = []
            for i, r in enumerate(revisions, start=1):
                rows_html.append(
                    f'<tr><td class="mono">{i:02d}</td>'
                    f'<td class="mono">{_esc(r.get("date", "—"))}</td>'
                    f"<td>{_esc(r.get('author', '—'))}</td>"
                    f'<td class="chname">{_esc(r.get("change", "—"))}</td>'
                    f"<td>{_esc(r.get('rationale', '—'))}</td></tr>"
                )
            table = (
                '<table class="data-table"><thead><tr><th>#</th><th>Date</th>'
                "<th>Author</th><th>Change</th><th>Rationale</th></tr></thead>"
                f'<tbody>{"".join(rows_html)}</tbody></table>'
            )
        else:
            table = ""
        body = f'<p class="lede">{self._insight("revisions_gloss")}</p>{table}'
        return _NavEntry("revisions", "Change record"), self._wrap(
            "revisions", "Change record", "How this specification evolved", body
        )

    def _section_signoff(self) -> tuple[_NavEntry, str]:
        loop = (
            '<div class="loop">'
            '<span class="step current">Design</span><span class="arrow">→</span>'
            '<span class="step">Fit</span><span class="arrow">→</span>'
            '<span class="step">Diagnose</span><span class="arrow">→</span>'
            '<span class="step">Prioritize</span><span class="arrow">→</span>'
            '<span class="step">Experiment</span><span class="arrow">→</span>'
            '<span class="step">Calibrate</span>'
            "</div>"
        )
        steps = (
            '<ul class="next-steps">'
            '<li><span class="who">Analytics</span><span>Fit the model exactly as '
            "specified here; any deviation goes through the change record first."
            "</span></li>"
            '<li><span class="who">Analytics</span><span>Check convergence '
            "(R-hat, ESS, divergences) and posterior-predictive fit before "
            "reading any effect.</span></li>"
            '<li><span class="who">Together</span><span>Judge the fitted model in '
            "the Media Performance Readout against the commitments made in this "
            "document.</span></li>"
            "</ul>"
        )
        body = f"<p class='lede'>{self._insight('next_steps')}</p>{loop}{steps}"
        return _NavEntry("signoff", "Next steps"), self._wrap(
            "signoff", "Next steps", "Between this document and the fit", body
        )

    # ── document assembly (mirrors MMMReportGenerator._assemble_html_augur) ──
    def _assemble(self, sections: list[tuple[_NavEntry, str]]) -> str:
        cfg = self.config
        generated_date = cfg.generated_date or datetime.now().strftime("%B %Y")

        meta_bits = []
        if cfg.client:
            meta_bits.append(f"Prepared for {_esc(cfg.client)}")
        if cfg.subtitle:
            meta_bits.append(_esc(cfg.subtitle))
        if cfg.analysis_period:
            meta_bits.append(_esc(cfg.analysis_period))
        meta_bits.append(_esc(generated_date))
        meta_line = '<span class="sep">·</span>'.join(meta_bits)
        conf = '<div class="conf">Confidential</div>' if cfg.confidential else ""
        header = (
            '<header class="report-header">'
            f'<div class="masthead-logo">{MASTHEAD_LOGO_SVG}</div>'
            '<div class="masthead-text">'
            '<div class="masthead-eyebrow">Marketing mix modeling · '
            "Model design readout · Pre-fit</div>"
            f"<h1>{_esc(cfg.title)}</h1>"
            f'<div class="meta">{meta_line}</div>{conf}</div></header>'
        )

        nav_items = "".join(
            f'<a class="nav-item" href="#{entry.section_id}">'
            f'<span class="nav-num">{i:02d}</span>{_esc(entry.title)}</a>'
            for i, (entry, _) in enumerate(sections, start=1)
        )
        nav_html = (
            '<nav class="report-nav" id="reportNav" aria-label="Report contents">'
            '<div class="nav-head">Contents</div>'
            f"{nav_items}</nav>"
        )

        footer = (
            '<footer class="report-footer">'
            "<p>This is a pre-fit design document: every number in it derives "
            "from the model's <em>priors</em> and simulation-based checks, not "
            "from a fitted posterior. Prior means, bands and implied response "
            "curves describe what the specification assumes before evidence. "
            f"Generated {_esc(generated_date)} with the MMM Framework.</p></footer>"
        )

        sections_html = "".join(body for _, body in sections)
        content = (
            f'<div class="report-body">{nav_html}'
            f'<div class="report-container">{header}{sections_html}{footer}</div></div>'
        )

        plotly_script = (
            f'<script src="https://cdn.plot.ly/plotly-{cfg.plotly_cdn_version}.min.js"></script>'
            if cfg.include_plotly_js
            else ""
        )
        scrollspy = """
<script>
(function() {
  var items = Array.prototype.slice.call(document.querySelectorAll('.report-nav .nav-item'));
  if (!items.length) return;
  var sections = items.map(function(a){ return document.getElementById(a.getAttribute('href').slice(1)); }).filter(Boolean);
  function setActive(id){ items.forEach(function(a){ a.classList.toggle('active', a.getAttribute('href').slice(1)===id); }); }
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function(entries){
      entries.forEach(function(e){ if (e.isIntersecting) setActive(e.target.id); });
    }, { rootMargin:'-18% 0px -72% 0px', threshold:0 });
    sections.forEach(function(s){ io.observe(s); });
  }
  if (sections[0]) setActive(sections[0].id);
})();
</script>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{_esc(cfg.title)}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    {AUGUR_FONTS_LINK}
    {plotly_script}
    <style>
{augur_css(cfg.color_scheme)}
    </style>
</head>
<body>
    {content}
    {scrollspy}
</body>
</html>"""
