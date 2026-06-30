"""Agentic interpretation of Simulation-Based Calibration results.

Turns the deterministic SBC statistics (per-parameter shape, χ² p-value,
miscalibration score, mean rank) into a plain-English read of *what the
calibration plots mean* and *what to do about it* — fixes or caveats targeted at
a modeller. Two layers, mirroring the deck/insights pattern:

* :func:`_deterministic_interpretation` — an always-valid floor built straight
  from the numbers (no LLM), using the Talts 2018 shape→cause→fix table.
* :func:`interpret_sbc` — best-effort LLM polish/prioritization grounded ONLY in
  those numbers; on any failure it returns the deterministic text.

The LLM call (when used) MUST run host-side, never inside the model-op / kernel
(the kernel env is scrubbed of API keys) — the job worker passes a built ``llm``.
"""

from __future__ import annotations

from typing import Any

# Talts 2018 direction (rank = #{posterior draws ≤ θ*}). The agent's advice
# hinges on this; do NOT invert it.
_SHAPE_READ: dict[str, tuple[str, str]] = {
    "uniform": (
        "calibrated — ranks look uniform",
        "Intervals for this parameter can be trusted at face value.",
    ),
    "smile(∪)": (
        "posterior TOO NARROW (overconfident) — ranks pile at the edges, so the "
        "true value lands in the posterior tails too often",
        "Reported credible intervals UNDER-cover. Widen the priors (especially "
        "the observation-noise σ prior), check for an over-informative likelihood "
        "or a too-tight parameter prior, and do NOT trust this parameter's "
        "intervals until SBC passes.",
    ),
    "frown(∩)": (
        "posterior TOO WIDE (overdispersed) — ranks bunch in the centre",
        "Intervals OVER-cover (conservative, not dangerous). Tighten the prior "
        "for this parameter if sharper inference is wanted.",
    ),
    "left-skew": (
        "posterior BIASED HIGH (systematically overestimates the truth) — ranks "
        "concentrate at low values",
        "Check the prior mean / centring and the standardization for this "
        "parameter; a biased point estimate will mis-rank channels.",
    ),
    "right-skew": (
        "posterior BIASED LOW (systematically underestimates) — ranks concentrate "
        "at high values",
        "Check the prior mean / centring and standardization for this parameter.",
    ),
}


def _pname(p: dict[str, Any]) -> str:
    return str(p.get("name", "?"))


def _deterministic_interpretation(sbc: dict[str, Any]) -> str:
    """A grounded, always-valid interpretation built from the SBC numbers."""
    params = sbc.get("params", []) or []
    n_eff = sbc.get("n_sims_effective", 0)
    sampler = sbc.get("sampler", "?")
    L = sbc.get("L", "?")
    all_cal = sbc.get("all_calibrated", False)

    lines: list[str] = ["### Simulation-Based Calibration — interpretation", ""]
    if all_cal:
        lines.append(
            f"**Verdict: calibrated.** Across {n_eff} simulations "
            f"(L={L} {sampler} draws/fit), every parameter's SBC ranks are "
            "consistent with uniform — the inference engine recovers the priors "
            "it was given, so the posterior intervals it reports have nominal "
            "coverage on data generated from this model."
        )
    else:
        bad = [p for p in params if not p.get("calibrated", True)]
        worst = max(params, key=lambda p: p.get("miscalibration", 0.0), default=None)
        lines.append(
            f"**Verdict: miscalibration detected** in {len(bad)} of "
            f"{len(params)} parameters across {n_eff} simulations "
            f"(L={L} {sampler} draws/fit). The credible intervals for the flagged "
            "parameters do not have nominal coverage — treat them with caution "
            "until the model is fixed and SBC re-run."
        )
        if worst is not None:
            lines.append(
                f"Worst offender: **{_pname(worst)}** "
                f"(miscalibration {worst.get('miscalibration', 0):.2f})."
            )

    # Per-parameter detail, worst first.
    ordered = sorted(params, key=lambda p: p.get("miscalibration", 0.0), reverse=True)
    lines.append("")
    lines.append("**By parameter:**")
    for p in ordered:
        read, fix = _SHAPE_READ.get(
            p.get("shape", "uniform"), ("unclassified", "Inspect the plots.")
        )
        if p.get("calibrated", False):
            lines.append(
                f"- `{_pname(p)}` — ✅ {read} "
                f"(χ² p={p.get('chi2_pvalue', float('nan')):.2f})."
            )
        else:
            lines.append(
                f"- `{_pname(p)}` — ⚠️ {read} "
                f"(χ² p={p.get('chi2_pvalue', float('nan')):.2f}, "
                f"mean-rank {p.get('mean_norm_rank', 0.5):.2f}, "
                f"miscal {p.get('miscalibration', 0):.2f}). **Fix:** {fix}"
            )

    caveats = sbc.get("caveats") or []
    if caveats:
        lines.append("")
        lines.append("**Caveats:**")
        for c in caveats:
            lines.append(f"- {c}")
    return "\n".join(lines)


_SYS = (
    "You are a Bayesian-diagnostics expert interpreting Simulation-Based "
    "Calibration (SBC) results for a marketing-mix modeller. You are given "
    "SUMMARY STATISTICS, not an image. Explain, concisely and actionably, whether "
    "the inference engine is calibrated, name the geometric failure per "
    "parameter, and recommend concrete fixes or caveats.\n"
    "Direction (Talts 2018; rank = number of posterior draws ≤ the true value): "
    "U-shape/∪ with mass at the edges => posterior TOO NARROW / overconfident "
    "(intervals under-cover); ∩-shape with mass in the centre => posterior TOO "
    "WIDE / overdispersed (intervals over-cover); mean rank < 0.5 => estimate "
    "biased HIGH; mean rank > 0.5 => biased LOW. Never invent numbers; ground "
    "every statement in the provided statistics. Prefer prioritized, plain "
    "guidance over restating every number. Use short markdown."
)


def _facts(sbc: dict[str, Any]) -> str:
    lines = [
        f"all_calibrated={sbc.get('all_calibrated')}",
        f"n_sims_effective={sbc.get('n_sims_effective')}, L={sbc.get('L')}, "
        f"sampler={sbc.get('sampler')}, n_failed_fits={sbc.get('n_failed_fits')}",
        "parameters (worst miscalibration first):",
    ]
    for p in sorted(
        sbc.get("params", []), key=lambda q: q.get("miscalibration", 0.0), reverse=True
    ):
        lines.append(
            f"  - {p.get('name')}: shape={p.get('shape')}, "
            f"chi2_p={p.get('chi2_pvalue'):.3f}, "
            f"mean_rank={p.get('mean_norm_rank'):.3f}, "
            f"miscalibration={p.get('miscalibration'):.3f}, "
            f"calibrated={p.get('calibrated')}"
        )
    for c in sbc.get("caveats", []) or []:
        lines.append(f"caveat: {c}")
    return "\n".join(lines)


def _clean(reply: Any) -> str:
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


def interpret_sbc(sbc: dict[str, Any], llm: Any | None = None) -> str:
    """Grounded interpretation of an SBC ``to_dashboard()`` payload.

    Returns the deterministic floor when ``llm`` is ``None`` or the call fails;
    otherwise an LLM-polished read grounded strictly in the SBC statistics.
    """
    fallback = _deterministic_interpretation(sbc)
    if not sbc.get("params"):
        return fallback
    if llm is None:
        return fallback
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = (
            "SBC statistics:\n"
            + _facts(sbc)
            + "\n\nWrite the interpretation: overall verdict, the failure shape "
            "and a concrete fix/caveat for each MISCALIBRATED parameter, and a "
            "one-line bottom line. If everything is calibrated, say so plainly."
        )
        txt = _clean(
            llm.invoke([SystemMessage(content=_SYS), HumanMessage(content=prompt)])
        )
        return txt or fallback
    except Exception:
        return fallback


__all__ = ["interpret_sbc"]
