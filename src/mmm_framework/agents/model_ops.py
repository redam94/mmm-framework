"""Relocatable model-reading operations (Phase 2, PR-A).

Each op takes the fitted model (and optionally ``results``) and returns a
JSON-serializable projection ``{content, dashboard, error}`` — the model-touching
compute AND the scalar/markdown extraction — with **no** langchain / graph-state
coupling. That keeps the same function runnable in-process today (the tool calls
it directly) and, in Phase 2 PR-B, *inside* the per-session kernel where the model
lives (dispatched via ``Kernel.run_model_op`` over a display_data MIME channel).

Contract: an op never raises for an expected compute failure — it returns
``error`` (a user-facing message string) so the failure crosses the future kernel
boundary as data. ``content`` is markdown; ``dashboard`` is a dict merged into
``dashboard_data``. Returns are built from ``float()``-cast scalars (the raw
computes return DataFrames / ndarrays / dataclasses that are NOT JSON-safe — the
projection happens here, before any boundary).
"""

from __future__ import annotations

from typing import Any

# Returned (as the `error`) when an op runs but no fitted model is available —
# shared so the in-process and subprocess kernels produce the identical message.
NO_MODEL_MSG = "No fitted model found in state. Please fit the model first."


def _ok(content: str, dashboard: dict) -> dict:
    return {"content": content, "dashboard": dashboard, "error": None}


def _err(message: str) -> dict:
    return {"content": None, "dashboard": {}, "error": message}


def roi_metrics(mmm: Any, results: Any = None, *, hdi_prob: float = 0.94) -> dict:
    try:
        from mmm_framework.reporting.helpers import compute_roi_with_uncertainty

        roi_df = compute_roi_with_uncertainty(mmm, hdi_prob=hdi_prob)
        content = (
            "### ROI Analysis\n\n"
            "| Channel | Mean ROI | 94% HDI | Prob Profitable |\n|---|---|---|---|\n"
        )
        for _, row in roi_df.iterrows():
            ci = f"[{row['roi_hdi_low']:.2f}, {row['roi_hdi_high']:.2f}]"
            content += (
                f"| {row['channel']} | {row['roi_mean']:.2f} | {ci} | "
                f"{row['prob_profitable']:.1%} |\n"
            )
        return _ok(content, {"roi_metrics": roi_df.to_dict(orient="records")})
    except Exception as e:  # noqa: BLE001
        return _err(f"Error computing ROI: {str(e)}")


def component_decomposition(mmm: Any, results: Any = None) -> dict:
    try:
        from mmm_framework.reporting.helpers import compute_component_decomposition

        decomp_list = compute_component_decomposition(mmm, include_time_series=False)
        content = "### Component Decomposition\n\n"
        content += "| Component | Contribution | Percentage |\n|---|---|---|\n"
        decomp_json = []
        for d in decomp_list:
            content += f"| {d.component} | {d.total_contribution:,.0f} | {d.pct_of_total:.1%} |\n"
            decomp_json.append(
                {
                    "component": d.component,
                    "total_contribution": float(d.total_contribution),
                    "pct_of_total": float(d.pct_of_total),
                }
            )
        return _ok(content, {"decomposition": decomp_json})
    except Exception as e:  # noqa: BLE001
        return _err(f"Error computing decomposition: {str(e)}")


def model_diagnostics(mmm: Any, results: Any = None) -> dict:
    try:
        from mmm_framework.reporting.helpers import _get_diagnostics

        diag = _get_diagnostics(mmm)
        if not diag:
            return _err(
                "Diagnostics could not be extracted. Make sure ArviZ is installed "
                "and the model sampled correctly."
            )
        content = "### Model Diagnostics\n\n"
        content += f"**Converged:** {'✅ Yes' if diag.get('converged') else '⚠️ No'}\n"
        content += f"**Divergences:** {diag.get('divergences', 0)} (Should be 0)\n"
        content += f"**Max R-hat:** {diag.get('rhat_max', 'N/A')} (Should be < 1.01)\n"
        content += (
            f"**Min Bulk ESS:** {diag.get('ess_bulk_min', 'N/A')} (Should be > 400)\n"
        )
        content += (
            f"**Min Tail ESS:** {diag.get('ess_tail_min', 'N/A')} (Should be > 400)\n"
        )
        return _ok(content, {"diagnostics": diag})
    except Exception as e:  # noqa: BLE001
        return _err(f"Error computing diagnostics: {str(e)}")


def adstock_weights(mmm: Any, results: Any = None) -> dict:
    try:
        from mmm_framework.reporting.helpers import compute_adstock_weights

        adstock = compute_adstock_weights(mmm)
        content = "### Adstock (Carryover) Effects\n\n"
        content += (
            "| Channel | Half-life (Periods) | Total Carryover % | Alpha (Decay Rate) |\n"
            "|---|---|---|---|\n"
        )
        adstock_json = {}
        for ch, result in adstock.items():
            content += (
                f"| {ch} | {result.half_life:.1f} | {result.total_carryover:.1%} | "
                f"{result.alpha_mean:.3f} |\n"
            )
            adstock_json[ch] = {
                "half_life": float(result.half_life),
                "total_carryover": float(result.total_carryover),
                "alpha_mean": float(result.alpha_mean),
            }
        return _ok(content, {"adstock": adstock_json})
    except Exception as e:  # noqa: BLE001
        return _err(f"Error computing adstock weights: {str(e)}")


def saturation_curves(mmm: Any, results: Any = None) -> dict:
    try:
        import numpy as np
        from mmm_framework.reporting.helpers import (
            compute_saturation_curves_with_uncertainty,
        )

        # The helper subsamples posterior draws via np.random.choice; pin the RNG
        # (save/restore global state) so repeated calls are reproducible — matters
        # more once this runs in a re-spawnable kernel.
        _st = np.random.get_state()
        np.random.seed(12345)
        try:
            curves = compute_saturation_curves_with_uncertainty(mmm)
        finally:
            np.random.set_state(_st)

        content = "### Saturation (Diminishing Returns) Analysis\n\n"
        content += (
            "| Channel | Current Saturation Level | Marginal Response (Next $1) |\n"
            "|---|---|---|\n"
        )
        saturation_json = {}
        for ch, curve in curves.items():
            sat_pct = curve.saturation_level
            status = (
                "🔴 High"
                if sat_pct > 0.8
                else "🟡 Medium" if sat_pct > 0.5 else "🟢 Low"
            )
            content += (
                f"| {ch} | {sat_pct:.1%} ({status}) | "
                f"{curve.marginal_response_at_current:.3f} |\n"
            )
            saturation_json[ch] = {
                "saturation_level": float(sat_pct),
                "marginal_response_at_current": float(
                    curve.marginal_response_at_current
                ),
                "status": status.split(" ")[1],
            }
        return _ok(content, {"saturation": saturation_json})
    except Exception as e:  # noqa: BLE001
        return _err(f"Error computing saturation: {str(e)}")


def budget_scenario(
    mmm: Any, results: Any = None, *, spend_changes: dict = None
) -> dict:
    """What-if budget scenario (live posterior-predictive). `spend_changes` maps
    channel -> fractional change. Returns a markdown summary (no dashboard)."""
    try:
        import json as _json

        changes = spend_changes or {}
        result = mmm.what_if_scenario(changes)
        text = _json.dumps(result, indent=2, default=str)
        content = (
            f"### Budget scenario\nApplied: {changes}\n```json\n{text[:4000]}\n```"
        )
        return _ok(content, {})
    except Exception as e:  # noqa: BLE001
        return _err(f"Scenario failed: {e}")


def marginal_analysis(
    mmm: Any, results: Any = None, *, spend_increase_pct: float = 10.0, channels=None
) -> dict:
    """Marginal contributions / mROAS for a spend bump (live posterior-predictive
    per channel). Returns a markdown table (no dashboard)."""
    try:
        df = mmm.compute_marginal_contributions(
            spend_increase_pct=spend_increase_pct, channels=channels
        )
        content = (
            f"### Marginal analysis (+{spend_increase_pct}% spend)\n```\n"
            f"{df.to_string()[:4000]}\n```"
        )
        return _ok(content, {})
    except Exception as e:  # noqa: BLE001
        return _err(f"Marginal analysis failed: {e}")


def prior_predictive_check(
    mmm: Any, results: Any = None, *, n_samples: int = 500
) -> dict:
    """Sample the prior predictive (live PyMC) and summarize the implied KPI
    scale. Returns markdown + dashboard + an `assumption` payload the causal tool
    wrapper records host-side (record_assumption must NOT run in the kernel)."""
    try:
        idata = mmm.sample_prior_predictive(samples=int(n_samples))
    except Exception as e:  # noqa: BLE001
        return _err(f"Prior predictive sampling failed: {e}")
    try:
        import numpy as np

        pp = idata.prior_predictive
        var = list(pp.data_vars)[0]
        arr = pp[var].values.reshape(-1)
        summary = {
            "samples": int(arr.size),
            "min": float(np.nanmin(arr)),
            "p05": float(np.nanpercentile(arr, 5)),
            "median": float(np.nanmedian(arr)),
            "p95": float(np.nanpercentile(arr, 95)),
            "max": float(np.nanmax(arr)),
            "frac_negative": float(np.mean(arr < 0)),
        }
    except Exception as e:  # noqa: BLE001
        summary = {"error": str(e)}

    flag_neg = summary.get("frac_negative", 0) > 0.05
    lines = ["### Prior Predictive Check", ""]
    if "error" in summary:
        lines.append(f"Could not summarize: {summary['error']}")
    else:
        lines.append(f"- Samples: {summary['samples']:,}")
        lines.append(
            f"- Implied KPI range (5–95%): [{summary['p05']:,.0f}, {summary['p95']:,.0f}]"
        )
        lines.append(f"- Median: {summary['median']:,.0f}")
        lines.append(f"- Fraction negative: {summary['frac_negative']:.1%}")
        if flag_neg:
            lines.append(
                "\n⚠️ >5% of prior-predictive draws are negative. Tighten priors before fitting."
            )
    res = _ok("\n".join(lines), {"prior_predictive_summary": summary})
    res["assumption"] = {
        "key": "prior_predictive_check",
        "value": summary,
        "rationale": "Prior predictive sanity check. "
        + (
            "⚠️ More than 5% of samples imply negative outcomes — consider tighter priors."
            if flag_neg
            else "Implied outcome range looks plausible."
        ),
        "category": "prior",
        "change_note": f"n_samples={int(n_samples)}",
    }
    return res


def leave_one_out(
    mmm: Any, results: Any = None, *, component_to_drop: str = ""
) -> dict:
    """Reweight the existing posterior decomposition with one component zeroed
    (NOT a refit). Returns markdown + dashboard + an `assumption` payload."""
    try:
        from mmm_framework.reporting.helpers import compute_component_decomposition

        decomp = compute_component_decomposition(mmm, include_time_series=False)
    except Exception as e:  # noqa: BLE001
        return _err(f"Could not compute decomposition: {e}")

    target = (component_to_drop or "").strip().lower()
    components = [(d.component, float(d.total_contribution)) for d in decomp]
    match_idx = next(
        (i for i, (c, _) in enumerate(components) if c.lower() == target), -1
    )
    if match_idx < 0:
        return _err(
            f"Component `{component_to_drop}` not found. Known: "
            + ", ".join(f"`{c}`" for c, _ in components)
        )

    total = sum(v for _, v in components)
    dropped_name, dropped_val = components[match_idx]
    remaining = [(c, v) for c, v in components if c != dropped_name]
    remaining_total = sum(v for _, v in remaining)
    new_pct = [
        (c, v / remaining_total if remaining_total else 0.0) for c, v in remaining
    ]
    pct_loss = dropped_val / total if total else 0.0
    sensitivity = {
        "dropped": dropped_name,
        "fraction_dropped": pct_loss,
        "remaining_decomposition": [
            {"component": c, "pct_of_remaining": p} for c, p in new_pct
        ],
    }
    lines = [
        "⚠️ **This is NOT a sensitivity refit.** It only reweights the *existing* "
        "posterior decomposition assuming the dropped channel contributed zero. "
        "Use this for quick what-if framing only; for honest sensitivity to "
        "the fit, re-run `fit_mmm_model` with the channel removed.",
        "",
        f"### Leave-one-out: drop `{dropped_name}`",
        "",
        f"- `{dropped_name}` accounts for **{pct_loss:.1%}** of fitted KPI.",
        "- Remaining components renormalize to:",
    ]
    for c, p in new_pct:
        lines.append(f"  - `{c}`: {p:.1%}")
    res = _ok("\n".join(lines), {"sensitivity_loo": sensitivity})
    res["assumption"] = {
        "key": f"sensitivity::loo::{dropped_name}",
        "value": sensitivity,
        "rationale": (
            f"Leave-one-out: dropping `{dropped_name}` removes {pct_loss:.1%} "
            "of total fitted KPI (post-hoc reweighting; not a refit)."
        ),
        "category": "other",
        "change_note": "leave-one-out decomposition",
    }
    return res


# Registry: the name -> op map the kernel dispatch (PR-B) resolves against.
OPS = {
    "roi_metrics": roi_metrics,
    "component_decomposition": component_decomposition,
    "model_diagnostics": model_diagnostics,
    "adstock_weights": adstock_weights,
    "saturation_curves": saturation_curves,
    "budget_scenario": budget_scenario,
    "marginal_analysis": marginal_analysis,
    "prior_predictive_check": prior_predictive_check,
    "leave_one_out": leave_one_out,
}
