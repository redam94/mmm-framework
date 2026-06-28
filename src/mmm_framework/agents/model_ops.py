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

Ops with tabular output also return a ``tables`` key — a list of structured
table payloads (``agents.tables`` builders, pure data) that the host-side tool
wrapper stores content-addressed and surfaces as dashboard refs. The builders
are import-light by design so this works inside the subprocess/container
kernel too.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mmm_framework.agents.tables import df_to_table_json, records_to_table_json

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
        res = _ok(content, {"roi_metrics": roi_df.to_dict(orient="records")})
        res["tables"] = [
            df_to_table_json(
                roi_df,
                title="ROI by Channel",
                source="get_roi_metrics",
                group="results",
                columns=[
                    {"key": "channel", "label": "Channel", "type": "string"},
                    {"key": "roi_mean", "label": "Mean ROI", "type": "number"},
                    {"key": "roi_hdi_low", "label": "HDI Low", "type": "number"},
                    {"key": "roi_hdi_high", "label": "HDI High", "type": "number"},
                    {
                        "key": "prob_profitable",
                        "label": "Prob Profitable",
                        "type": "percent",
                    },
                ],
            )
        ]
        return res
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
        res = _ok(content, {"decomposition": decomp_json})
        res["tables"] = [
            records_to_table_json(
                decomp_json,
                title="Component Decomposition",
                source="get_component_decomposition",
                group="results",
                columns=[
                    {"key": "component", "label": "Component", "type": "string"},
                    {
                        "key": "total_contribution",
                        "label": "Contribution",
                        "type": "number",
                    },
                    {"key": "pct_of_total", "label": "% of Total", "type": "percent"},
                ],
            )
        ]
        return res
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
        res = _ok(content, {"diagnostics": diag})
        res["tables"] = [
            records_to_table_json(
                [
                    {
                        "converged": bool(diag.get("converged")),
                        "divergences": diag.get("divergences", 0),
                        "rhat_max": diag.get("rhat_max"),
                        "ess_bulk_min": diag.get("ess_bulk_min"),
                        "ess_tail_min": diag.get("ess_tail_min"),
                    }
                ],
                title="MCMC Diagnostics",
                source="get_model_diagnostics",
                group="results",
            )
        ]
        return res
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
        res = _ok(content, {"adstock": adstock_json})
        res["tables"] = [
            records_to_table_json(
                [{"channel": ch, **vals} for ch, vals in adstock_json.items()],
                title="Adstock (Carryover) Effects",
                source="get_adstock_weights",
                group="results",
                columns=[
                    {"key": "channel", "label": "Channel", "type": "string"},
                    {
                        "key": "half_life",
                        "label": "Half-life (Periods)",
                        "type": "number",
                    },
                    {
                        "key": "total_carryover",
                        "label": "Total Carryover",
                        "type": "percent",
                    },
                    {"key": "alpha_mean", "label": "Alpha (Decay)", "type": "number"},
                ],
            )
        ]
        return res
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
        res = _ok(content, {"saturation": saturation_json})
        res["tables"] = [
            records_to_table_json(
                [{"channel": ch, **vals} for ch, vals in saturation_json.items()],
                title="Saturation by Channel",
                source="get_saturation_curves",
                group="results",
                columns=[
                    {"key": "channel", "label": "Channel", "type": "string"},
                    {
                        "key": "saturation_level",
                        "label": "Saturation Level",
                        "type": "percent",
                    },
                    {
                        "key": "marginal_response_at_current",
                        "label": "Marginal Response",
                        "type": "number",
                    },
                    {"key": "status", "label": "Status", "type": "string"},
                ],
            )
        ]
        return res
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
        if isinstance(result, dict) and "outcome_change_hdi" in result:
            hdi = result["outcome_change_hdi"]
            pct = int(round(result.get("hdi_prob", 0.94) * 100))
            p = result.get("prob_positive")
            content += (
                f"\n\n**Outcome change:** {result.get('outcome_change', 0):,.0f} "
                f"({pct}% credible interval [{hdi[0]:,.0f}, {hdi[1]:,.0f}]; "
                f"P(scenario beats baseline) = {p:.0%}, {result.get('n_draws')} "
                "paired posterior draws). A point estimate alone is not "
                "decision-grade."
            )
        res = _ok(content, {})
        if isinstance(result, dict) and result:
            rows = [
                {"metric": str(k), "value": v}
                for k, v in result.items()
                if isinstance(v, (int, float, str, bool)) or v is None
            ]
            if rows:
                res["tables"] = [
                    records_to_table_json(
                        rows,
                        title="Budget Scenario",
                        source="run_budget_scenario",
                        group="results",
                    )
                ]
        return res
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
        preview = df.head(10)
        content = (
            f"### Marginal analysis (+{spend_increase_pct}% spend)\n\n"
            f"{len(df)} row(s); first {len(preview)} shown — the full formatted "
            f"table is in the dashboard.\n```\n{preview.to_string()[:2000]}\n```"
        )
        res = _ok(content, {})
        res["tables"] = [
            df_to_table_json(
                df,
                title=f"Marginal Analysis (+{spend_increase_pct:g}% spend)",
                source="run_marginal_analysis",
                group="results",
            )
        ]
        return res
    except Exception as e:  # noqa: BLE001
        return _err(f"Marginal analysis failed: {e}")


def compute_estimands(
    mmm: Any, results: Any = None, *, estimands=None, random_seed=None
) -> dict:
    """Realize the model's declarative estimands (the counterfactual causal lens).

    Uses the model's ``declared_estimands`` if any, else the capability defaults
    (``contribution_roi`` / ``marginal_roas`` / ``contribution``). Returns one row
    per realized estimand (wildcard-channel estimands expand per channel) with
    mean + HDI + units; unsupported estimands are reported, not dropped. This is
    the registry-driven surface that subsumes the framework's scattered estimand
    logic — ``get_roi_metrics`` is left as-is.
    """
    try:
        from .estimand_rows import evaluate_estimand_rows

        rows = evaluate_estimand_rows(mmm, estimands=estimands, random_seed=random_seed)
        if not rows:
            return _err(
                "No estimands to compute (model declares none and the "
                "capability defaults are empty)."
            )

        content = (
            "### Estimands\n\n"
            "| Estimand | Channel | Mean | 94% HDI | Units | Status |\n"
            "|---|---|---|---|---|---|\n"
        )
        for row in rows:
            if row["status"] != "ok" or row["mean"] is None:
                ci = "—"
                mean = "—"
            else:
                mean = f"{row['mean']:.3f}"
                lo, hi = row["hdi_low"], row["hdi_high"]
                ci = (
                    f"[{lo:.3f}, {hi:.3f}]"
                    if lo is not None and hi is not None
                    else "—"
                )
            content += (
                f"| {row['estimand']} | {row['channel']} | {mean} | {ci} | "
                f"{row['units']} | {row['status']} |\n"
            )

        res = _ok(content, {"estimands": rows})
        res["tables"] = [
            records_to_table_json(
                rows,
                title="Estimands",
                source="compute_estimands",
                group="results",
                columns=[
                    {"key": "estimand", "label": "Estimand", "type": "string"},
                    {"key": "channel", "label": "Channel", "type": "string"},
                    {"key": "mean", "label": "Mean", "type": "number"},
                    {"key": "hdi_low", "label": "HDI Low", "type": "number"},
                    {"key": "hdi_high", "label": "HDI High", "type": "number"},
                    {"key": "units", "label": "Units", "type": "string"},
                    {"key": "prob_positive", "label": "P(>0)", "type": "percent"},
                ],
            )
        ]
        return res
    except Exception as e:  # noqa: BLE001
        return _err(f"Error computing estimands: {e}")


def prior_predictive_check(
    mmm: Any,
    results: Any = None,
    *,
    n_samples: int = 500,
    spec: dict | None = None,
    dataset_path: str | None = None,
) -> dict:
    """Sample the prior predictive (live PyMC) and summarize the implied KPI
    scale. Returns markdown + dashboard + an `assumption` payload the causal tool
    wrapper records host-side (record_assumption must NOT run in the kernel).

    Runs PRE-FIT by design: when ``spec`` + ``dataset_path`` are available
    (passed by the tool wrapper from session state), an unfitted model graph is
    built from the ACTIVE spec — no fit needed, and the check always reflects
    the priors that the NEXT fit would use (a fitted model's graph carries the
    priors from fit time, which go stale the moment the user tweaks them). The
    fitted model is only a fallback when no spec/dataset is in state."""
    if spec and dataset_path:
        try:
            from mmm_framework.agents.fitting import build_model

            mmm = build_model(spec, dataset_path)
            mode = "pre-fit (built from the active spec; reflects current priors)"
        except Exception as e:  # noqa: BLE001
            return _err(f"Could not build the model for a pre-fit check: {e}")
    elif mmm is not None:
        mode = "post-fit fallback (fitted model's graph — priors as of fit time)"
    else:
        return _err(
            "No model in the session and no spec/dataset to build one from. "
            "Configure a model and load a dataset first."
        )
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
    lines = ["### Prior Predictive Check", "", f"- Mode: {mode}"]
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
        "change_note": f"n_samples={int(n_samples)}, mode={mode}",
    }
    return res


# Dispatchers skip the fitted-model gate for this op: it builds an unfitted
# model from spec+dataset when none exists (the pre-fit path).
prior_predictive_check.allow_unfitted = True


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


def save_model(mmm: Any, results: Any = None, *, name: str = "model") -> dict:
    """Serialize the model to ``<cwd>/mmm_models/<name>``. The kernel runs in the
    session work_dir, so for the subprocess this lands in the per-session
    workspace (where the model lives); in-process it lands under the API cwd,
    unchanged from before."""
    try:
        import os as _os
        from mmm_framework.serialization import MMMSerializer

        save_dir = _os.path.join("mmm_models", str(name))
        _os.makedirs(save_dir, exist_ok=True)
        MMMSerializer.save(mmm, save_dir)
        return _ok(f"Model saved as **{name}** at `{save_dir}/`.", {})
    except Exception as e:  # noqa: BLE001
        return _err(f"Save failed: {e}")


def optimize_budget(
    mmm: Any,
    results: Any = None,
    *,
    total_budget: float | None = None,
    budget_change_pct: float | None = None,
    min_multiplier: float = 0.0,
    max_multiplier: float = 2.0,
    bounds: dict | None = None,
    max_draws: int = 200,
) -> dict:
    """Optimal budget allocation from the fitted model's response curves, with
    per-draw re-optimization for allocation stability. Returns markdown +
    a dashboard payload + a structured table.

    ``bounds`` sets PER-CHANNEL spend limits as ``{channel: [low, high]}``
    multipliers of current spend (e.g. a partner-committed cap or a frozen line),
    overriding the global ``min_multiplier``/``max_multiplier`` for those channels.
    An unknown channel or a malformed range is rejected (a silently-ignored
    constraint would hand a planner an unexecutable plan)."""
    norm_bounds: dict[str, tuple[float, float]] | None = None
    if bounds:
        names = list(getattr(mmm, "channel_names", []) or [])
        unknown = [c for c in bounds if c not in names]
        if unknown:
            return _err(
                f"Unknown channel(s) in bounds: {unknown}. Valid channels: {names}."
            )
        norm_bounds = {}
        for c, lohi in bounds.items():
            try:
                lo, hi = float(lohi[0]), float(lohi[1])
            except Exception:  # noqa: BLE001
                return _err(
                    f"bounds[{c!r}] must be [low, high] multipliers, got {lohi!r}."
                )
            if lo < 0 or hi < lo:
                return _err(
                    f"bounds[{c!r}] must satisfy 0 <= low <= high, got [{lo}, {hi}]."
                )
            norm_bounds[c] = (lo, hi)

    try:
        from mmm_framework.planning import optimize_budget as _optimize

        res = _optimize(
            mmm,
            total_budget=total_budget,
            budget_change_pct=budget_change_pct,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
            bounds=norm_bounds,
            max_draws=max_draws,
            random_seed=42,
        )
    except Exception as e:  # noqa: BLE001
        return _err(f"Budget optimization failed: {e}")

    t = res.table
    lines = ["### Budget Optimization", ""]
    lines.append(
        f"- Budget allocated: {res.total_budget:,.0f} "
        f"(current total spend: {t['current_spend'].sum():,.0f})"
    )
    lines.append(
        f"- Expected KPI uplift vs current allocation: {res.expected_uplift:,.0f} "
        f"(90% interval [{res.uplift_hdi[0]:,.0f}, {res.uplift_hdi[1]:,.0f}]; "
        f"P(uplift > 0) = {res.prob_positive_uplift:.0%}; {res.n_draws} draws)"
    )
    biggest = t.reindex(t["change_pct"].abs().sort_values(ascending=False).index).head(
        3
    )
    for _, r in biggest.iterrows():
        lines.append(
            f"- `{r['channel']}`: {r['current_share_pct']:.0f}% → "
            f"{r['optimal_share_pct']:.0f}% of budget ({r['change_pct']:+.0f}% spend); "
            f"optimal share 90% range [{r['optimal_share_p5']:.0f}%, {r['optimal_share_p95']:.0f}%]"
        )
    if norm_bounds:
        bound_str = ", ".join(
            f"`{c}` [{lo:g}x–{hi:g}x]" for c, (lo, hi) in norm_bounds.items()
        )
        lines.append(f"- Per-channel constraints applied: {bound_str}")
    for n in res.notes:
        lines.append(f"- ⚠️ {n}")
    lines.append(
        "\nCaveats: optimal within the sampled spend range only (no evidence "
        "beyond observed spend); wide per-channel share ranges mean the data "
        "does not pin down the optimum — see `recommend_experiments`."
    )

    summary = {
        "total_budget": res.total_budget,
        "expected_uplift": res.expected_uplift,
        "uplift_hdi": list(res.uplift_hdi),
        "prob_positive_uplift": res.prob_positive_uplift,
        "allocation": t.to_dict(orient="records"),
    }
    out = _ok("\n".join(lines), {"budget_optimization": summary})
    out["tables"] = [
        df_to_table_json(
            t.round(2),
            title=f"Optimal Budget Allocation ({res.total_budget:,.0f})",
            source="optimize_budget",
            group="results",
        )
    ]
    return out


def _normalize_channel_bounds(
    mmm: Any, bounds: dict | None
) -> tuple[dict | None, str | None]:
    """Validate per-channel ``{channel: [low, high]}`` multiplier bounds against
    the model's channels. Returns (normalized, error_message)."""
    if not bounds:
        return None, None
    names = list(getattr(mmm, "channel_names", []) or [])
    unknown = [c for c in bounds if c not in names]
    if unknown:
        return (
            None,
            f"Unknown channel(s) in bounds: {unknown}. Valid channels: {names}.",
        )
    norm: dict[str, tuple[float, float]] = {}
    for c, lohi in bounds.items():
        try:
            lo, hi = float(lohi[0]), float(lohi[1])
        except Exception:  # noqa: BLE001
            return None, f"bounds[{c!r}] must be [low, high] multipliers, got {lohi!r}."
        if lo < 0 or hi < lo:
            return (
                None,
                f"bounds[{c!r}] must satisfy 0 <= low <= high, got [{lo}, {hi}].",
            )
        norm[c] = (lo, hi)
    return norm, None


def plan_budget(
    mmm: Any,
    results: Any = None,
    *,
    total_budget: float | None = None,
    budget_change_pct: float | None = None,
    min_multiplier: float = 0.0,
    max_multiplier: float = 2.0,
    bounds: dict | None = None,
    by_geo: bool = False,
    flighting: dict | None = None,
    max_draws: int = 200,
) -> dict:
    """Decision-grade budget plan for the Planner studio: optimal allocation
    (national or per-geo) plus an optional forward flighting calendar.

    Wraps :func:`planning.optimize_budget` (or ``optimize_budget_by_geo`` when
    ``by_geo`` and the model is a geo panel) and, if ``flighting`` is given,
    spreads the recommended per-channel budgets across future periods. Returns a
    single ``budget_plan`` dashboard payload the FE renders without a chat
    round-trip."""
    import pandas as pd

    norm_bounds, err = _normalize_channel_bounds(mmm, bounds)
    if err:
        return _err(err)

    use_geo = (
        bool(by_geo)
        and bool(getattr(mmm, "has_geo", False))
        and int(getattr(mmm, "n_geos", 1)) > 1
    )

    try:
        from mmm_framework import planning as _pl

        if use_geo:
            res = _pl.optimize_budget_by_geo(
                mmm,
                total_budget=total_budget,
                budget_change_pct=budget_change_pct,
                min_multiplier=min_multiplier,
                max_multiplier=max_multiplier,
                bounds=norm_bounds,
                max_draws=max_draws,
                random_seed=42,
            )
        else:
            res = _pl.optimize_budget(
                mmm,
                total_budget=total_budget,
                budget_change_pct=budget_change_pct,
                min_multiplier=min_multiplier,
                max_multiplier=max_multiplier,
                bounds=norm_bounds,
                max_draws=max_draws,
                random_seed=42,
            )
    except Exception as e:  # noqa: BLE001
        return _err(f"Budget planning failed: {e}")

    t = res.table
    current_total = float(t["current_spend"].sum())

    # National roll-up (the headline + the budgets the flighting calendar spreads).
    if use_geo:
        roll = t.groupby("channel", as_index=False).agg(
            current_spend=("current_spend", "sum"),
            optimal_spend=("optimal_spend", "sum"),
        )
        opt_tot = max(float(roll["optimal_spend"].sum()), 1e-9)
        cur_tot = max(float(roll["current_spend"].sum()), 1e-9)
        roll["current_share_pct"] = 100 * roll["current_spend"] / cur_tot
        roll["optimal_share_pct"] = 100 * roll["optimal_spend"] / opt_tot
        roll["change_pct"] = (
            100
            * (roll["optimal_spend"] - roll["current_spend"])
            / roll["current_spend"].replace(0, pd.NA)
        ).fillna(0.0)
        national = roll
    else:
        national = t

    channel_budgets = {
        str(r["channel"]): float(r["optimal_spend"]) for _, r in national.iterrows()
    }

    plan: dict[str, Any] = {
        "by_geo": use_geo,
        "total_budget": float(res.total_budget),
        "current_total": current_total,
        "expected_uplift": float(res.expected_uplift),
        "uplift_hdi": [float(res.uplift_hdi[0]), float(res.uplift_hdi[1])],
        "prob_positive_uplift": float(res.prob_positive_uplift),
        "n_draws": int(res.n_draws),
        "allocation": national.round(2).to_dict(orient="records"),
        "notes": list(res.notes),
    }
    if use_geo:
        plan["geo_allocation"] = t.round(2).to_dict(orient="records")
        plan["geos"] = list(getattr(mmm, "geo_names", []))

    if flighting:
        try:
            fl = _pl.build_flighting_schedule(
                channel_budgets,
                int(flighting.get("n_periods", 13)),
                pattern=str(flighting.get("pattern", "even")),
                front_load=float(flighting.get("front_load", 0.65)),
                pulse_on=int(flighting.get("pulse_on", 1)),
                pulse_off=int(flighting.get("pulse_off", 1)),
                seasonal=flighting.get("seasonal"),
                period_labels=flighting.get("period_labels"),
            )
            plan["flighting"] = fl
        except Exception as e:  # noqa: BLE001
            plan["notes"].append(f"Flighting skipped: {e}")

    lines = ["### Budget Plan", ""]
    lines.append(
        f"- Budget allocated: {res.total_budget:,.0f} (current {current_total:,.0f})"
    )
    lines.append(
        f"- Expected uplift vs current: {res.expected_uplift:,.0f} "
        f"(90% [{res.uplift_hdi[0]:,.0f}, {res.uplift_hdi[1]:,.0f}]; "
        f"P(>0)={res.prob_positive_uplift:.0%})"
    )
    if use_geo:
        lines.append(
            f"- Allocated per geography across {len(plan.get('geos', []))} geos"
        )
    if plan.get("flighting"):
        fl = plan["flighting"]
        lines.append(f"- {fl['n_periods']}-period {fl['pattern']} flighting calendar")
    for n in plan["notes"]:
        lines.append(f"- ⚠️ {n}")

    out = _ok("\n".join(lines), {"budget_plan": plan})
    tables = [
        df_to_table_json(
            national.round(2),
            title="Recommended Allocation",
            source="plan_budget",
            group="results",
        )
    ]
    if use_geo:
        tables.append(
            df_to_table_json(
                t.round(2),
                title="Per-Geo Allocation",
                source="plan_budget",
                group="results",
            )
        )
    if plan.get("flighting"):
        tables.append(
            records_to_table_json(
                plan["flighting"]["schedule"],
                title=f"Flighting Calendar ({plan['flighting']['pattern']})",
                source="plan_budget",
                group="results",
            )
        )
    out["tables"] = tables
    return out


def plan_scenario(
    mmm: Any,
    results: Any = None,
    *,
    spend_changes: dict | None = None,
    time_period: list | tuple | None = None,
    max_draws: int = 200,
) -> dict:
    """Structured what-if scenario for the Planner (uncertainty included).

    ``spend_changes`` maps channel -> fractional change (e.g. ``{"TV": 0.2}`` is
    +20%). Returns a clean ``budget_scenario`` dashboard payload (baseline vs
    scenario outcome, credible interval, P(beats baseline), per-channel deltas)."""
    changes = spend_changes or {}
    if not changes:
        return _err("Provide spend_changes: {channel: fractional_change}.")
    names = list(getattr(mmm, "channel_names", []) or [])
    unknown = [c for c in changes if c not in names]
    if unknown:
        return _err(f"Unknown channel(s) in spend_changes: {unknown}. Valid: {names}.")

    try:
        tp = tuple(time_period) if time_period else None
        result = mmm.what_if_scenario(
            changes, time_period=tp, max_draws=max_draws, random_seed=42
        )
    except Exception as e:  # noqa: BLE001
        return _err(f"Scenario failed: {e}")

    sc: dict[str, Any] = {
        "spend_changes_applied": {k: float(v) for k, v in changes.items()},
        "time_period": list(tp) if tp else None,
        "baseline_outcome": float(result.get("baseline_outcome", 0.0)),
        "scenario_outcome": float(result.get("scenario_outcome", 0.0)),
        "outcome_change": float(result.get("outcome_change", 0.0)),
        "outcome_change_pct": float(result.get("outcome_change_pct", 0.0)),
        "channel_details": result.get("spend_changes", {}),
    }
    if "outcome_change_hdi" in result:
        hdi = result["outcome_change_hdi"]
        sc["outcome_change_hdi"] = [float(hdi[0]), float(hdi[1])]
        sc["prob_positive"] = float(result.get("prob_positive", 0.0))
        sc["n_draws"] = int(result.get("n_draws", 0))
        sc["hdi_prob"] = float(result.get("hdi_prob", 0.94))

    lines = [
        "### Budget Scenario",
        "",
        f"- Outcome change: {sc['outcome_change']:,.0f} ({sc['outcome_change_pct']:+.1f}%)",
    ]
    if "outcome_change_hdi" in sc:
        lines.append(
            f"- {int(sc['hdi_prob'] * 100)}% interval "
            f"[{sc['outcome_change_hdi'][0]:,.0f}, {sc['outcome_change_hdi'][1]:,.0f}]; "
            f"P(beats baseline) = {sc['prob_positive']:.0%} ({sc['n_draws']} draws)"
        )

    out = _ok("\n".join(lines), {"budget_scenario": sc})
    rows = []
    for ch, d in (sc["channel_details"] or {}).items():
        row = {"channel": ch}
        if isinstance(d, dict):
            for k, v in d.items():
                row[k] = float(v) if isinstance(v, (int, float)) else v
        rows.append(row)
    if rows:
        out["tables"] = [
            records_to_table_json(
                rows,
                title="Scenario Spend Changes",
                source="plan_scenario",
                group="results",
            )
        ]
    return out


def experiment_design(
    mmm: Any,
    results: Any = None,
    *,
    top_k: int = 3,
    max_draws: int = 200,
) -> dict:
    """Rank channels by experiment value (spend at stake × ROAS uncertainty ×
    allocation instability) and propose concrete lift-test designs with
    calibration snippets for the next fit."""
    try:
        from mmm_framework.planning import recommend_experiments as _recommend

        try:
            table, designs = _recommend(
                mmm,
                top_k=int(top_k),
                max_draws=max_draws,
                random_seed=42,
                method="eig_evoi",
            )
            method = "eig_evoi"
        except Exception:  # noqa: BLE001 — EIG/EVOI is the default; never block
            table, designs = _recommend(
                mmm,
                top_k=int(top_k),
                max_draws=max_draws,
                random_seed=42,
                method="heuristic",
            )
            method = "heuristic"
    except Exception as e:  # noqa: BLE001
        return _err(f"Experiment design recommendation failed: {e}")

    lines = ["### Experiment Design Recommendations", ""]
    if method == "eig_evoi":
        lines.append(
            "Priority = √(EIG × EVOI), normalized — what the experiment teaches "
            "(expected information gain) times what that learning is worth to "
            "the budget decision (expected value of information)."
        )
    else:
        lines.append(
            "Priority = spend share × ROAS uncertainty (CV) × (1 + allocation "
            "instability) — test where uncertainty is large AND moves real money."
        )
    for i, d in enumerate(designs, 1):
        lines.append(f"\n**{i}. `{d['channel']}`** (priority {d['priority']:.3f})")
        lines.append(f"   - Why: {d['why']}")
        lines.append(f"   - Design: {d['design_type']}")
        lines.append(
            f"   - Duration: ≥ {d['min_duration_periods']} periods "
            f"({d['duration_rationale']})"
        )
        lines.append(
            f"   - Target precision: SE ≤ {d['target_se']:.2f} on measured ROAS "
            f"({d['target_se_rationale']})"
        )
    lines.append(
        "\nWhen a result lands, calibrate it into the next fit (see the "
        "`calibration_snippet` per design in the dashboard payload) — do not "
        "reconcile informally."
    )

    out = _ok(
        "\n".join(lines),
        {
            "experiment_design": {
                "ranking": table.to_dict(orient="records"),
                "designs": designs,
            }
        },
    )
    out["tables"] = [
        df_to_table_json(
            table.round(3),
            title="Experiment Priority Ranking",
            source="experiment_design",
            group="results",
        )
    ]
    return out


def experiment_priorities(
    mmm: Any,
    results: Any = None,
    *,
    evidence: dict | None = None,
    as_of: str | None = None,
    n_outcomes: int = 48,
    max_draws: int = 200,
) -> dict:
    """The EIG/EVOI priority grid: per-channel expected information gain,
    expected value of information for the budget decision, the 2×2 quadrant
    classification, and information-decay re-test triggers (when the registry
    supplies per-channel evidence dates via ``evidence``)."""
    try:
        from mmm_framework.planning import compute_experiment_priorities as _grid

        grid, portfolio = _grid(
            mmm,
            evidence=evidence,
            as_of=as_of,
            n_outcomes=int(n_outcomes),
            max_draws=int(max_draws),
            random_seed=42,
        )
    except Exception as e:  # noqa: BLE001
        return _err(f"Experiment priority computation failed: {e}")

    rows = [g.to_dict() for g in grid]
    quad_lists: dict[str, list[str]] = {}
    for g in grid:
        quad_lists.setdefault(g.quadrant, []).append(g.channel)

    lines = ["### EIG/EVOI Experiment Priorities", ""]
    lines.append(
        f"- Portfolio EVPI (value of perfect information): "
        f"{portfolio['evpi']:,.0f} KPI units over the response window"
    )
    for quad, label in (
        ("test_now", "🎯 Test now (high EIG × high EVOI)"),
        ("learn_cheaply", "📚 Learn cheaply (informative; decision robust)"),
        ("monitor", "👁 Monitor (high stakes; already precise)"),
        ("deprioritize", "💤 Deprioritize"),
    ):
        if quad in quad_lists:
            lines.append(f"- {label}: {', '.join(f'`{c}`' for c in quad_lists[quad])}")
    retests = [g.channel for g in grid if g.retest_due]
    if retests:
        lines.append(
            f"- ⏰ Re-test due (evidence decayed past threshold): "
            f"{', '.join(f'`{c}`' for c in retests)}"
        )
    top = grid[0] if grid else None
    if top:
        lines.append(
            f"\nTop priority: `{top.channel}` — EIG {top.eig:.2f} nats, "
            f"EVOI {top.evoi:,.0f} ({top.evpi_share:.0%} of EVPI)."
        )

    out = _ok(
        "\n".join(lines),
        {
            "experiment_priorities": {
                "channels": rows,
                "portfolio": portfolio,
                "matrix": quad_lists,
            }
        },
    )
    out["tables"] = [
        records_to_table_json(
            [
                {
                    k: r[k]
                    for k in (
                        "channel",
                        "quadrant",
                        "priority",
                        "eig",
                        "evoi",
                        "evpi_share",
                        "roi_mean",
                        "roi_sd",
                        "spend_share",
                        "retest_due",
                    )
                }
                for r in rows
            ],
            title="EIG/EVOI Priority Grid",
            source="compute_experiment_priorities",
            group="results",
        )
    ]
    return out


def _design_from_params(design_params: dict) -> dict:
    """Run the PURE designer from a params dict carried across the kernel
    boundary (the kernel has no host state, so dataset_path/kpi/channel must
    travel in design_params)."""
    from mmm_framework.planning.design import design_experiment, design_options

    dataset_path = design_params.get("dataset_path")
    kpi = design_params.get("kpi")
    channel = design_params.get("channel")
    if not (dataset_path and kpi and channel):
        raise ValueError(
            "design_params must carry dataset_path, kpi and channel "
            "(the kernel cannot read host state)."
        )
    key = (
        design_params.get("design_key")
        or design_options(dataset_path, kpi, channel)["recommended"]
    )
    if key == "national_flighting":
        kw = dict(
            duration=int(design_params.get("duration", 12)),
            amplitude_pct=float(design_params.get("amplitude_pct", 50.0)),
            block_weeks=int(design_params.get("block_weeks", 2)),
            seed=int(design_params.get("seed", 42)),
        )
        _levels = design_params.get("levels")
        if _levels:
            kw["levels"] = tuple(float(m) for m in _levels)
    else:
        kw = dict(
            duration=int(design_params.get("duration", 8)),
            design=design_params.get("design", "scaling"),
            intensity_pct=float(design_params.get("intensity_pct", 50.0)),
            seed=int(design_params.get("seed", 42)),
        )
        if design_params.get("n_pairs") is not None:
            kw["n_pairs"] = int(design_params["n_pairs"])
    return design_experiment(dataset_path, kpi, channel, design_key=key, **kw)


def experiment_economics(
    mmm: Any,
    results: Any = None,
    *,
    design_params: dict,
    run_simulation: bool = True,
    include_loopback: bool | None = None,
    margin: float | None = None,
    price: float | None = None,
    kpi_kind: str = "revenue",
    loss_threshold: float | None = None,
    max_draws: int = 200,
    random_seed: int = 42,
) -> dict:
    """Model-anchored experiment economics: the pure design, the model's
    expected-effect anchor + powered-to-detect verdict, the short-term
    opportunity cost of deviating from BAU (with posterior uncertainty), and an
    A/A·A/B methodology comparison. Runs design-only when no model is fitted."""
    try:
        design = _design_from_params(design_params)
    except ValueError as e:
        return _err(f"Could not design the experiment: {e}")
    except Exception as e:  # noqa: BLE001
        return _err(f"Experiment design failed: {e}")

    channel = design["channel"]
    kpi = design.get("kpi", "")
    duration = int(design.get("duration", 8) or 8)
    dataset_path = design_params.get("dataset_path")
    if include_loopback is None:
        include_loopback = bool(run_simulation)

    payload: dict[str, Any] = {
        "channel": channel,
        "kpi": kpi,
        "design_key": design.get("design_key"),
        "design_type": design.get("design_type"),
        "duration": duration,
        "randomized": design.get("randomized", True),
        "se_roas": design.get("se_roas"),
        "mde_roas": design.get("mde_roas"),
        "se_source": design.get("se_source"),
        "model_anchored": mmm is not None,
        "anchor": None,
        "opportunity_cost": None,
        "simulation": None,
    }
    lines = [
        f"### Experiment economics — `{channel}` ({design.get('design_type')})",
        "",
    ]

    if mmm is None:
        payload["note"] = (
            "No fitted model — analytic design plus the A/A·A/B methodology check on "
            "historical data (which needs no model). Fit a model to add the "
            "expected-effect anchor and opportunity cost."
        )
        lines.append(
            f"- {duration}-week test → SE(ROAS) ≈ {design.get('se_roas', 0):.2f}, "
            f"MDE ≈ {design.get('mde_roas', 0):.2f} (analytic; fit a model for the "
            "expected-effect anchor and opportunity cost)."
        )

    # ── Model-anchored expected effect, powered-to-detect verdict, OC (need a fit) ──
    evoi_channel: float | None = None
    if mmm is not None:
        from mmm_framework.planning import (
            compute_experiment_priorities,
            compute_opportunity_cost,
            model_anchored_effect,
            powered_to_detect,
            realized_sigma_exp_for_anchor,
        )

        # The anchor and the opportunity cost both need the SAME paired BAU /
        # experiment posterior passes (the dominant cost). Compute them once and
        # share, so the op runs 2 passes instead of 4. Falls back to per-call
        # sampling if the shared build fails.
        _bau = _exp = None
        try:
            from mmm_framework.planning.opportunity_cost import (
                _resolve_treated_rows,
                build_experiment_media,
            )

            _tmask, _tcodes, _wc, _deff, _w = _resolve_treated_rows(
                mmm, design, duration=duration
            )
            _x_exp, _chi, _nr = build_experiment_media(
                mmm, design, treated_mask=_tmask, window_codes=_wc
            )
            _x_bau = np.asarray(getattr(mmm, "X_media_raw"), dtype=float)
            _bau = mmm.sample_channel_contributions(
                X_media=_x_bau, max_draws=max_draws, random_seed=random_seed
            )
            _exp = mmm.sample_channel_contributions(
                X_media=_x_exp, max_draws=max_draws, random_seed=random_seed
            )
        except Exception:  # noqa: BLE001 — sharing is an optimization only
            _bau = _exp = None

        try:
            anchor = model_anchored_effect(
                mmm,
                design,
                max_draws=max_draws,
                random_seed=random_seed,
                contrib_bau=_bau,
                contrib_exp=_exp,
            )
            verdict = powered_to_detect(
                anchor,
                design.get("power_curve"),
                duration,
                float(design.get("se_roas") or 0.0),
            )
            sigma_exp, incr_draws = realized_sigma_exp_for_anchor(
                np.asarray(anchor["incremental_roas_draws"], dtype=float),
                float(design.get("se_roas") or 0.0),
            )
            design["model_anchor"] = {
                "expected_effect": anchor,
                "verdict": verdict,
                "realized_sigma_exp": sigma_exp,
                "roas_units": "incremental ROAS (KPI per $) over the treated cells",
            }
            payload["anchor"] = {
                "roas_at_current_median": anchor["roas_at_current_median"],
                "incremental_roas_median": anchor["incremental_roas_median"],
                "incremental_roas_hdi": anchor["incremental_roas_hdi"],
                "expected_incremental_kpi_median": anchor[
                    "expected_incremental_kpi_median"
                ],
                "verdict": verdict["verdict"],
                "assurance": verdict["assurance"],
                "prob_detectable": verdict["prob_detectable"],
                "recommended_duration": verdict["recommended_duration"],
                "extrapolation_warning": anchor["extrapolation_warning"],
            }
            lines += [
                f"- Model expects incremental ROAS ≈ "
                f"{anchor['incremental_roas_median']:.2f} "
                f"(90% HDI [{anchor['incremental_roas_hdi'][0]:.2f}, "
                f"{anchor['incremental_roas_hdi'][1]:.2f}]); "
                f"MDE ≈ {design.get('mde_roas', 0):.2f}.",
                f"- **{verdict['verdict'].upper()}** — assurance "
                f"{(verdict['assurance'] or 0):.0%} of detecting the expected effect"
                + (
                    f"; reach power at ≈ {verdict['recommended_duration']} weeks."
                    if verdict.get("recommended_duration")
                    else "."
                ),
            ]

            # Loopback: feed the realized precision + incremental estimand into
            # the EIG/EVOI grid (draw-paired with the curves at the same max_draws).
            if include_loopback:
                try:
                    grid, _portfolio = compute_experiment_priorities(
                        mmm,
                        sigma_exp_overrides={channel: sigma_exp},
                        roi_draws_overrides={channel: incr_draws},
                        max_draws=max_draws,
                        random_seed=random_seed,
                    )
                    row = next((g for g in grid if g.channel == channel), None)
                    if row is not None:
                        evoi_channel = float(row.evoi)
                        payload["anchor"]["eig"] = float(row.eig)
                        payload["anchor"]["evoi"] = evoi_channel
                        payload["anchor"]["quadrant"] = row.quadrant
                except Exception as e:  # noqa: BLE001 — loopback is a refinement
                    payload["anchor"]["loopback_error"] = str(e)
        except Exception as e:  # noqa: BLE001
            design["model_anchor_error"] = str(e)
            payload["anchor_error"] = str(e)

        # ── Opportunity cost / short-term risk ──
        try:
            oc = compute_opportunity_cost(
                mmm,
                design,
                margin_per_kpi=margin,
                price=price,
                kpi_kind=kpi_kind,
                loss_threshold=loss_threshold,
                evoi_kpi_units=evoi_channel,
                response_horizon_weeks=int(getattr(mmm, "n_periods", 0) or 0) or None,
                max_draws=max_draws,
                random_seed=random_seed,
                contrib_bau=_bau,
                contrib_exp=_exp,
            )
            payload["opportunity_cost"] = oc.to_dict()
            lines.append(
                f"- Short-term risk: forgo ≈ {oc.forgone_kpi_median:,.0f} KPI "
                f"({(oc.pct_of_window_kpi or 0):.1%} of the treated window), "
                f"spend Δ {oc.spend_delta:+,.0f}"
                + (
                    f"; net ${oc.net_profit_impact_median:+,.0f} "
                    f"(downside ${oc.opportunity_cost_dollar_median:,.0f})."
                    if oc.net_profit_impact_median is not None
                    else " (supply a margin for net-$ impact)."
                )
            )
            if oc.learning_to_cost_ratio is not None:
                lines.append(
                    f"- Learning-vs-cost: {oc.learning_to_cost_ratio:.2f} "
                    "(EVOI per week ÷ KPI cost per week)."
                )
        except Exception as e:  # noqa: BLE001
            payload["opportunity_cost_error"] = str(e)

    # ── A/A & A/B methodology comparison (runs pre-fit too — needs no model) ──
    if run_simulation:
        try:
            from mmm_framework.planning import methodology_leaderboard

            expected_total = None
            if payload.get("anchor"):
                expected_total = abs(
                    float(payload["anchor"]["expected_incremental_kpi_median"])
                )
            spend_window = None
            if payload.get("opportunity_cost"):
                spend_window = float(payload["opportunity_cost"]["abs_spend_change"])
            elif design.get("weekly_spend_delta"):
                # Pre-fit fallback: realized $ change ≈ weekly delta × test weeks.
                spend_window = float(design["weekly_spend_delta"]) * duration
            sim = methodology_leaderboard(
                dataset_path,
                kpi,
                channel,
                mmm=mmm,
                design=design,
                duration=duration,
                target_mde_roas=float(design.get("mde_roas") or 0.0) or None,
                spend_delta_window=spend_window or 0.0,
                expected_effect_total=expected_total,
                seed=int(design_params.get("seed", 42)),
            )
            payload["simulation"] = sim
            chosen = sim.get("chosen_key")
            invalid = [m["key"] for m in sim["methodologies"] if not m["valid"]]
            lines.append(
                "- Methodology check: "
                + (
                    f"recommend `{chosen}`"
                    if chosen
                    else "no methodology cleared the validity bar on this data"
                )
                + (
                    "; flagged (inflated false-positive rate): "
                    f"{', '.join('`%s`' % k for k in invalid)}."
                    if invalid
                    else "."
                )
            )
        except Exception as e:  # noqa: BLE001
            payload["simulation_error"] = str(e)

    payload["design"] = design
    out = _ok(
        "\n".join(lines),
        {"experiment_economics": payload, "experiment_design_plan": design},
    )

    # Tables: methodology comparison + opportunity-cost metrics.
    tables = []
    sim = payload.get("simulation")
    if sim and sim.get("methodologies"):
        tables.append(
            records_to_table_json(
                [
                    {
                        "methodology": m["label"],
                        "valid": m["valid"],
                        "fpr": m["fpr"],
                        "fpr_tolerance": m["fpr_tolerance"],
                        "n_eff_windows": m["n_eff_windows"],
                        "empirical_mde_roas": m["empirical_mde_roas"],
                        "power_at_expected": m["power_at_expected_effect"],
                        "recommended": m["key"] == sim.get("chosen_key"),
                    }
                    for m in sim["methodologies"]
                ],
                title="Experiment methodology comparison (A/A · A/B)",
                source="experiment_economics",
                group="results",
            )
        )
    oc = payload.get("opportunity_cost")
    if oc:
        oc_rows = [
            ("Forgone KPI (median)", oc.get("forgone_kpi_median")),
            ("Forgone KPI (p95 worst case)", oc.get("forgone_kpi_p95")),
            ("% of treated-window KPI", oc.get("pct_of_window_kpi")),
            ("Spend Δ (signed)", oc.get("spend_delta")),
            ("Spend at risk", oc.get("spend_at_risk")),
            ("Net $ impact (median)", oc.get("net_profit_impact_median")),
            ("Opportunity cost $ (p95)", oc.get("opportunity_cost_dollar_median")),
            ("Learning-vs-cost ratio", oc.get("learning_to_cost_ratio")),
        ]
        tables.append(
            records_to_table_json(
                [{"metric": k, "value": v} for k, v in oc_rows if v is not None],
                title="Short-term risk / opportunity cost",
                source="experiment_economics",
                group="results",
            )
        )
    if tables:
        out["tables"] = tables
    return out


experiment_economics.allow_unfitted = True  # design-only path works pre-fit


def experiment_optimizer(
    mmm: Any,
    results: Any = None,
    *,
    dataset_path: str,
    kpi: str,
    channel: str,
    margin: float | None = None,
    price: float | None = None,
    kpi_kind: str = "revenue",
    duration_min: int = 4,
    duration_max: int = 12,
    intensity_min: float = 50.0,
    intensity_max: float = 100.0,
    durations: list[int] | None = None,
    scaling_intensities: list[float] | None = None,
    include_holdout: bool = True,
    footprints: list[str] | None = None,
    max_draws: int = 80,
    random_seed: int = 42,
) -> dict:
    """Suggest a runnable experiment setup for ``channel`` and the Pareto front
    of designs (MDE × power × short-term cost × duration), over the duration and
    spend-variation ranges — using the fitted model for the opportunity cost,
    statistical power, and the adstock-derived cool-down."""
    if mmm is None:
        return _err(NO_MODEL_MSG)
    if not (dataset_path and kpi and channel):
        return _err(
            "optimizer params must carry dataset_path, kpi and channel "
            "(the kernel cannot read host state)."
        )
    try:
        from mmm_framework.planning import suggest_experiment

        out = suggest_experiment(
            mmm,
            dataset_path,
            kpi,
            channel,
            margin=margin,
            price=price,
            kpi_kind=kpi_kind,
            duration_min=int(duration_min),
            duration_max=int(duration_max),
            intensity_min=float(intensity_min),
            intensity_max=float(intensity_max),
            durations=tuple(durations) if durations else None,
            scaling_intensities=(
                tuple(scaling_intensities) if scaling_intensities else None
            ),
            include_holdout=bool(include_holdout),
            footprints=tuple(footprints) if footprints else ("full", "half"),
            max_draws=int(max_draws),
            random_seed=int(random_seed),
        )
    except Exception as e:  # noqa: BLE001
        return _err(f"Experiment optimization failed: {e}")

    rec = out.get("recommended")
    cool = out.get("cooldown") or {}
    lines = [f"### Suggested experiment for `{channel}`", ""]
    if rec:
        # to_dict() maps non-finite floats to None; coerce before formatting.
        _intensity = rec.get("intensity_pct") or 0.0
        _mde = rec.get("mde_roas") or 0.0
        _tradeoff = rec.get("tradeoff") or 0.0
        groups = (
            f"test {rec.get('treatment_geos')} vs control {rec.get('control_geos')}"
            if rec.get("treatment_geos")
            else f"national flighting (±{_intensity:.0f}%, "
            f"{rec.get('block_weeks')}-week blocks)"
        )
        lines += [
            f"- **{rec.get('design_key')} / {rec.get('mode')}** — "
            f"{_intensity:+.0f}% spend, {rec.get('footprint')} footprint, "
            f"**{rec.get('duration')}-week** test.",
            f"- MDE ≈ {_mde:.2f} ROAS; short-term tradeoff ≈ "
            f"{_tradeoff:,.0f} ({rec.get('tradeoff_basis')}).",
            (
                f"- **Power {rec['power']:.0%}** to detect the expected effect "
                f"(target {float(out.get('power_target') or 0.8):.0%}) — "
                f"{'meets' if rec.get('powered') else 'BELOW'} the bar."
                if rec.get("power") is not None
                else "- Power: not available (no model effect estimate)."
            ),
        ]
        _pb = rec.get("power_breakdown")
        if _pb:

            def _pp(x):
                return f"{x:.0%}" if isinstance(x, (int, float)) else "n/a"

            lines.append(
                f"- Flighting power by estimand ({_pb.get('n_levels')} spend "
                f"levels): ROAS {_pp(_pb.get('roas'))}, contribution "
                f"{_pp(_pb.get('contribution'))}, **mROAS {_pp(_pb.get('mroas'))}** "
                + (
                    "(curve identified — multi-level)."
                    if _pb.get("mroas_identified")
                    else "(on/off: mROAS is a secant, not the curve — add spend "
                    "levels to identify it)."
                )
            )
        lines += [
            f"- Setup: {groups}.",
            f"- **Cool-down: {cool.get('cooldown_weeks')} weeks** "
            f"({cool.get('basis')}) before the treated cells are back to BAU.",
        ]
        if rec.get("duration_requested") and rec["duration_requested"] != rec.get(
            "duration"
        ):
            lines.append(
                f"- ⚠️ Realized window {rec.get('duration')}w (requested "
                f"{rec['duration_requested']}w) — ragged panel / adstock memory."
            )
    else:
        lines.append("No feasible design found for this channel/data.")
    for _note in out.get("notes") or []:
        lines.append(f"- _{_note}_")
    lines.append(
        f"\nPareto front: {len(out.get('pareto_indices') or [])} non-dominated of "
        f"{out.get('n_candidates')} designs (lowest MDE × highest power "
        f"[target {float(out.get('power_target') or 0.8):.0%}] × smallest "
        "short-term cost × shortest duration). See the table for the full front."
    )

    op_out = _ok("\n".join(lines), {"experiment_optimization": out})
    pareto = out.get("pareto") or []
    if pareto:
        op_out["tables"] = [
            records_to_table_json(
                [
                    {
                        "design": f"{c['design_key']}/{c['mode']}",
                        "footprint": c["footprint"],
                        "intensity_pct": c["intensity_pct"],
                        "duration": c["duration"],
                        "mde_roas": c["mde_roas"],
                        "power": c["power"],
                        "power_roas": (c.get("power_breakdown") or {}).get("roas"),
                        "power_mroas": (c.get("power_breakdown") or {}).get("mroas"),
                        "n_levels": (c.get("power_breakdown") or {}).get("n_levels"),
                        "tradeoff": c["tradeoff"],
                        "tradeoff_basis": c["tradeoff_basis"],
                        "forgone_kpi": c["forgone_kpi_median"],
                        "powered": c["powered"],
                        "recommended": c["is_recommended"],
                    }
                    for c in pareto
                ],
                title="Experiment Pareto front (MDE × power × cost × duration)",
                source="experiment_optimizer",
                group="results",
            )
        ]
    return op_out


# ── Structural-parameter identification (multi-level flighting) ────────────────


def _structural_anchor(mmm: Any, channel: str) -> tuple[dict | None, str]:
    """Extract the v1-gated structural anchor for ``channel`` from a fitted model,
    or ``(None, reason)`` when the gate fails (Tier 2 refuses; Tier 1 still runs).

    Gate (locked design): national single cell, PARAMETRIC geometric adstock +
    logistic saturation, with ``beta_<ch>``/``adstock_alpha_<ch>``/``sat_lam_<ch>``
    present. The legacy path has only a Beta mix weight, not a decay ``alpha`` —
    so it refuses rather than read a mix weight as carryover.
    """
    try:
        from mmm_framework.config.enums import AdstockType, SaturationType
        from mmm_framework.reporting.helpers.utils import (
            _flatten_samples,
            _get_posterior,
        )
    except Exception as e:  # noqa: BLE001
        return None, f"import failed: {e}"

    names = list(getattr(mmm, "channel_names", []) or [])
    if channel not in names:
        return None, f"channel {channel!r} not in the fitted model"
    c = names.index(channel)
    if not bool(getattr(mmm, "use_parametric_adstock", False)):
        return None, (
            "the fit uses legacy non-parametric adstock (a Beta mix weight, not a "
            "decay alpha) — structural identification needs a parametric geometric fit"
        )
    n_cells = int(getattr(mmm, "n_cells", 1) or 1)
    if n_cells > 1:
        return None, "structural identification v1 is national (single-cell) only"
    try:
        a_cfg = mmm._get_adstock_config(channel)
        s_cfg = mmm._get_saturation_config(channel)
    except Exception as e:  # noqa: BLE001
        return None, f"could not read the channel transform config: {e}"
    if a_cfg.type != AdstockType.GEOMETRIC:
        return None, f"v1 supports geometric adstock only (got {a_cfg.type})"
    if s_cfg.type != SaturationType.LOGISTIC:
        return None, f"v1 supports logistic saturation only (got {s_cfg.type})"

    post = _get_posterior(mmm)
    if post is None:
        return None, "no posterior on the fitted model"
    keys = {
        "beta": f"beta_{channel}",
        "alpha": f"adstock_alpha_{channel}",
        "lam": f"sat_lam_{channel}",
    }
    draws: dict[str, np.ndarray] = {}
    for p, k in keys.items():
        if k not in post:
            return None, f"posterior is missing {k!r}"
        draws[p] = np.asarray(_flatten_samples(post[k].values), dtype=float)

    try:
        x_media = np.asarray(getattr(mmm, "X_media_raw"), dtype=float)
        op_spend = float(x_media[:, c].mean())
        raw_max = float(mmm._media_raw_max[channel])
        y_std = float(mmm.y_std)
        n_periods = int(getattr(mmm, "n_periods", x_media.shape[0]) or x_media.shape[0])
        sigma_lo = float(np.median(_flatten_samples(post["sigma"].values))) * y_std
    except Exception as e:  # noqa: BLE001
        return None, f"could not read model scaling/sigma: {e}"

    # pessimistic noise = response-regression residual SD (demand-detrended).
    try:
        from mmm_framework.planning.design import _regression_residual_sd

        sigma_hi = float(
            _regression_residual_sd(np.asarray(mmm.y_raw, dtype=float), x_media[:, c])
        )
    except Exception:  # noqa: BLE001
        sigma_hi = sigma_lo
    if not (math.isfinite(sigma_hi) and sigma_hi > 0):
        sigma_hi = sigma_lo

    return (
        {
            "channel_index": c,
            "draws": draws,
            "op_spend": op_spend,
            "raw_max": raw_max,
            "y_std": y_std,
            "n_periods": n_periods,
            "l_max": int(a_cfg.l_max),
            "normalize": bool(a_cfg.normalize),
            "sigma_lo": float(sigma_lo),
            "sigma_hi": float(sigma_hi),
            "x_media": x_media,
        },
        "ok",
    )


def _structural_self_check(
    mmm: Any, anchor: dict, *, n_draws: int, random_seed: int
) -> tuple[bool, str]:
    """Fail-closed check that the numpy forward op byte-mirrors the fitted graph:
    compare the median channel contribution from ``sample_channel_contributions``
    to the numpy forward op evaluated over the same historical series across a
    posterior sample. Mismatch → Tier 2 is disabled (Tier 1 still runs)."""
    try:
        from mmm_framework.planning.identification import _forward_contribution

        c = anchor["channel_index"]
        x_media = anchor["x_media"]
        model_contrib = mmm.sample_channel_contributions(
            X_media=x_media, max_draws=int(n_draws), random_seed=int(random_seed)
        )
        model_med = np.median(np.asarray(model_contrib)[:, :, c], axis=0)

        # numpy forward op median over a posterior subsample of the same size.
        rng = np.random.default_rng(int(random_seed))
        d = anchor["draws"]
        n = min(d["beta"].size, d["alpha"].size, d["lam"].size)
        idx = rng.choice(n, size=min(int(n_draws), n), replace=False)
        hist_mults = x_media[:, c] / max(anchor["op_spend"], 1e-12)
        series = []
        for i in idx:
            ci, _ = _forward_contribution(
                hist_mults,
                anchor["op_spend"],
                anchor["raw_max"],
                anchor["y_std"],
                float(d["beta"][i]),
                float(d["alpha"][i]),
                float(d["lam"][i]),
                l_max=anchor["l_max"],
                normalize=anchor["normalize"],
            )
            series.append(ci)
        mine_med = np.median(np.asarray(series), axis=0)

        scale = float(np.median(np.abs(model_med))) or 1.0
        rrmse = float(np.sqrt(np.mean((mine_med - model_med) ** 2)) / scale)
        if model_med.std() > 0 and mine_med.std() > 0:
            corr = float(np.corrcoef(mine_med, model_med)[0, 1])
        else:
            corr = 0.0
        ok = bool(np.isfinite(rrmse) and rrmse < 0.15 and corr > 0.9)
        return ok, f"rRMSE={rrmse:.3f}, corr={corr:.3f}"
    except Exception as e:  # noqa: BLE001
        return False, f"self-check raised: {e}"


def identify_structural_parameters(
    mmm: Any,
    results: Any = None,
    *,
    dataset_path: str,
    kpi: str,
    channel: str,
    levels: list[float] | None = None,
    block_weeks: int | None = None,
    duration: int = 12,
    max_draws: int = 200,
    self_check_draws: int = 60,
    random_seed: int = 42,
) -> dict:
    """Design a multi-level flighting schedule and report how well its refit
    would identify the channel's STRUCTURAL parameters — saturation curve
    (``psi``), adstock carryover (``alpha``), and coefficient (``beta``).

    An OPTIMISTIC UPPER BOUND (local Laplace design over the manufactured
    exogenous variance) on what the next refit achieves — never a guarantee. The
    recommended estimator is the full structural refit with the experiment weeks
    appended. Requires a parametric geometric + logistic national fit; otherwise
    reports the reduced-form curve/marginal power only."""
    if mmm is None:
        return _err(NO_MODEL_MSG)
    if not (dataset_path and kpi and channel):
        return _err(
            "identification params must carry dataset_path, kpi and channel "
            "(the kernel cannot read host state)."
        )

    from mmm_framework.planning import identification as _ident
    from mmm_framework.planning.design import flighting_design
    from mmm_framework.planning.experiment_optimizer import cooldown_weeks

    # Cool-down (= adstock washout) sets the minimum block so carryover doesn't
    # smear the contrast — the design must clear it to identify alpha.
    cool = cooldown_weeks(mmm, channel)
    blk = int(block_weeks or cool.get("cooldown_weeks") or 2)

    anchor, reason = _structural_anchor(mmm, channel)

    # In-support clamp on the requested levels (curvature credit needs the top
    # level within historical per-period spend support).
    lv = list(levels) if levels else [0.5, 1.0, 1.5]
    extrap = False
    if anchor is not None:
        top_in_support = 0.98 * anchor["raw_max"] / max(anchor["op_spend"], 1e-12)
        clamped = [min(float(m), top_in_support) for m in lv]
        extrap = any(float(m) > top_in_support + 1e-9 for m in lv)
        lv = clamped

    try:
        design = flighting_design(
            dataset_path,
            kpi,
            channel,
            levels=tuple(lv),
            block_weeks=blk,
            duration=int(duration),
            seed=int(random_seed),
        )
    except Exception as e:  # noqa: BLE001
        return _err(f"Could not build the flighting schedule: {e}")

    schedule = design.get("schedule") or []
    mults = np.array([s["multiplier"] for s in schedule], dtype=float)
    n_levels = int(design.get("n_levels", 0))

    payload: dict[str, Any] = {
        "channel": channel,
        "kpi": kpi,
        "design_key": "national_flighting",
        "block_weeks": blk,
        "duration": int(design.get("duration", duration) or duration),
        "n_levels": n_levels,
        "schedule": schedule,
        "cooldown_weeks": cool.get("cooldown_weeks"),
        "block_ge_cooldown": blk >= int(cool.get("cooldown_weeks") or 0),
        "reduced_form": design.get("estimand_ses"),
        "extrapolation_warning": bool(extrap),
        "structural": None,
        "structural_gated": anchor is not None,
        "structural_gate_reason": reason,
    }
    lines = [
        f"### Structural identification — `{channel}` (multi-level flighting)",
        "",
        f"- {payload['duration']}-week budget-neutral schedule, {n_levels} spend "
        f"levels, {blk}-week blocks "
        f"({'>=' if payload['block_ge_cooldown'] else '<'} {cool.get('cooldown_weeks')}w "
        "adstock washout"
        + (
            ""
            if payload["block_ge_cooldown"]
            else " — block too short, carryover smears the contrast"
        )
        + ").",
    ]
    if extrap:
        lines.append(
            "- ⚠️ Requested a spend level beyond the channel's observed per-period "
            "range — clamped to in-support; curve identification cannot rest on "
            "extrapolation."
        )

    if anchor is None:
        payload["note"] = (
            "Structural (beta/alpha/psi) identification is unavailable for this "
            f"fit ({reason}). The reduced-form curve / marginal-ROAS power is "
            "still computed (multi-level flighting traces the saturation curve)."
        )
        lines += ["", f"- Structural block skipped: _{reason}_."]
        if n_levels >= 3:
            lines.append(
                "- The schedule's ≥3 spend levels still let the next fit trace the "
                "saturation curve (reduced-form marginal-ROAS identification)."
            )
        return _ok("\n".join(lines), {"structural_identification": payload})

    ok, detail = _structural_self_check(
        mmm, anchor, n_draws=int(self_check_draws), random_seed=int(random_seed)
    )
    payload["self_check"] = {"passed": ok, "detail": detail}
    if not ok:
        payload["note"] = (
            "The numpy forward op did not byte-mirror the fitted graph "
            f"({detail}); structural identification is withheld (fail-closed). "
            "Reduced-form curve power still applies."
        )
        lines += ["", f"- ⚠️ Structural self-check failed ({detail}); withheld."]
        return _ok("\n".join(lines), {"structural_identification": payload})

    in_support = not extrap and (
        max(mults) * anchor["op_spend"] <= anchor["raw_max"] + 1e-9
    )
    struct = _ident.structural_identification(
        mults,
        anchor["op_spend"],
        anchor["raw_max"],
        anchor["y_std"],
        anchor["draws"],
        sigma_lo=anchor["sigma_lo"],
        sigma_hi=anchor["sigma_hi"],
        l_max=anchor["l_max"],
        normalize=anchor["normalize"],
        in_support=in_support,
    )
    if struct is None:
        payload["note"] = "structural design degenerate (near-singular / no contrast)."
        lines += ["", "- Structural design degenerate — no identifiable contrast."]
        return _ok("\n".join(lines), {"structural_identification": payload})

    payload["structural"] = struct
    p = struct["params"]

    def _pct(x):
        return f"{x:.0%}" if isinstance(x, (int, float)) else "n/a"

    def _row(name, key):
        d = p[key]
        if not d["claimed"]:
            hint = (
                "add ≥3 in-support spend levels"
                if key == "lam"
                else (
                    "sharpen / lengthen the spend pulses"
                    if key == "alpha"
                    else "widen the spend contrast"
                )
            )
            return f"- **{name}**: not identified by this design — {hint}."
        # Lead with contraction (the honest experiment-driven identification
        # axis); power is the resolve-from-0 number (UI-consistent, prior-heavy).
        return (
            f"- **{name}**: contraction {_pct(d['contraction'])} of the prior "
            f"(MDE {d['mde']:.3g}"
            + (
                f" = {_pct(d['mde_relative'])} of the estimate"
                if d.get("mde_relative")
                else ""
            )
            + f"; power {_pct(d['power'])} to resolve from 0)."
        )

    lines += [
        "",
        "**Structural identification (optimistic upper bound on the refit):**",
        _row("Coefficient β", "beta"),
        _row("Adstock α (carryover)", "alpha"),
        _row("Saturation curve ψ", "lam"),
    ]
    if not struct.get("identifies_anything"):
        payload["note"] = (
            "this schedule contributes no structural information (flat / collinear "
            "contrast) — the reduced-form curve power still applies."
        )
        lines.append(
            "- This schedule does not move the structural parameters off their "
            "priors — add spend levels and sharpen the pulses."
        )
        return _ok("\n".join(lines), {"structural_identification": payload})
    binding = struct.get("binding_power")
    binding_c = struct.get("binding_contraction")
    lines.append(
        f"- **Binding identification: contraction {_pct(binding_c)}** "
        f"(power {_pct(binding)} to resolve from 0, target {_pct(struct['power_target'])}) "
        "— the worst-identified claimed parameter; recommended estimator: a full "
        "structural refit with the experiment weeks appended."
    )
    if struct.get("n_clamped"):
        lines.append(
            f"- {struct['n_clamped']} test week(s) are fully saturated (no marginal "
            "info there) — keep levels in the responsive range."
        )

    op_out = _ok("\n".join(lines), {"structural_identification": payload})
    rows = []
    for name, key in (("beta", "beta"), ("alpha", "alpha"), ("lam", "lam")):
        d = p[key]
        rows.append(
            {
                "parameter": {
                    "beta": "coefficient",
                    "alpha": "adstock",
                    "lam": "saturation",
                }[key],
                "claimed": d["claimed"],
                "identified": d["identified"],
                "power": d["power"],
                "contraction": d["contraction"],
                "mde": d["mde"],
                "prior_sd": d["prior_sd"],
                "post_sd": d["post_sd"],
            }
        )
    op_out["tables"] = [
        records_to_table_json(
            rows,
            title="Structural identification by parameter (β, α, ψ)",
            source="identify_structural_parameters",
            group="results",
        )
    ]
    return op_out


# Registry: the name -> op map the kernel dispatch (PR-B) resolves against.
def garden_compat(
    mmm: Any = None,
    results: Any = None,
    *,
    source_path: str | None = None,
    class_name: str | None = None,
    scenarios: Any = ("clean",),
    fit_method: str = "map",
    n_weeks: int = 104,
    check_carryover: bool = True,
) -> dict:
    """Run the Model Garden compatibility suite on a candidate model class.

    Resolves the class from its stored source (a KERNEL-SIDE import — untrusted
    expert code runs only inside the session kernel, never the host) and grades
    it against synthetic worlds with known causal truth. Returns the standard
    payload plus a ``compat_report`` key the host tool stores on the registry
    row to gate ``draft -> tested``. ``allow_unfitted`` — it builds + fits its
    own candidate, so no session model is required.
    """
    if not source_path:
        return _err("garden_compat requires a `source_path` to the model source.")
    try:
        from mmm_framework.garden.compat import run_compatibility_check
        from mmm_framework.garden.loader import load_garden_class_from_path

        cls = load_garden_class_from_path(source_path, class_name)
    except Exception as e:  # noqa: BLE001
        return _err(f"Could not load garden model: {e}")
    try:
        report = run_compatibility_check(
            cls,
            scenarios=tuple(scenarios) if scenarios else ("clean",),
            fit_method=fit_method,
            n_weeks=int(n_weeks),
            check_carryover=bool(check_carryover),
        )
    except Exception as e:  # noqa: BLE001
        return _err(f"Compatibility check crashed: {e}")

    res = _ok(report["summary"], {"garden_compat": report})
    res["compat_report"] = report
    try:
        res["tables"] = [
            records_to_table_json(
                [
                    {
                        "tier": t["name"],
                        "result": (
                            "skip"
                            if t["skipped"]
                            else ("pass" if t["passed"] else "FAIL")
                        ),
                        "blocking": "yes" if t["blocking"] else "",
                        "detail": t["detail"],
                    }
                    for t in report.get("tiers", [])
                ],
                title=f"Compatibility tiers — {report.get('class_name', 'model')}",
                source="garden_compat",
                group="garden",
            )
        ]
    except Exception:  # noqa: BLE001
        pass
    return res


garden_compat.allow_unfitted = True  # builds + fits its own candidate model


def garden_tune_suggestions(mmm: Any, results: Any = None) -> dict:
    """Inspect a fitted model's convergence + parameter-learning diagnostics and
    propose concrete changes to improve fit quality and fitting time — the
    helper-agent signal source for ``suggest_model_improvements``."""
    if mmm is None:
        return _err(NO_MODEL_MSG)
    try:
        from mmm_framework.diagnostics import compute_fit_diagnostics

        diag = compute_fit_diagnostics(mmm, results)
    except Exception as e:  # noqa: BLE001
        return _err(f"Could not compute diagnostics: {e}")

    conv = diag.get("convergence") or {}
    learn = diag.get("learning") or {}
    # A non-fitted model yields an all-null convergence block (ok=true) and a
    # learning_error — which would otherwise read as "fit looks healthy". Treat
    # "no usable convergence metric AND no learning block" as un-diagnosable and
    # return the failure as data. (A MAP fit has null convergence metrics but a
    # real learning block, so it is not caught here.)
    conv_has_metrics = any(
        conv.get(k) is not None for k in ("divergences", "rhat_max", "ess_bulk_min")
    )
    if not learn and not conv_has_metrics:
        detail = diag.get("learning_error") or diag.get("convergence_error") or ""
        return _err(
            "Could not compute fit diagnostics — pass a fitted model"
            + (f" ({detail})" if detail else ".")
        )
    suggestions: list[dict] = []  # {area, priority, issue, action}

    if bool(getattr(results, "approximate", False)):
        suggestions.append(
            {
                "area": "inference",
                "priority": "high",
                "issue": "This is an APPROXIMATE fit (MAP/ADVI/Pathfinder) — uncertainty is not calibrated.",
                "action": "Re-fit with NUTS (method='nuts') before trusting intervals or making decisions.",
            }
        )

    divergences = conv.get("divergences")
    if divergences:
        suggestions.append(
            {
                "area": "accuracy",
                "priority": "high",
                "issue": f"{divergences} divergent transition(s) — the sampler hit difficult geometry.",
                "action": "Raise target_accept to 0.95–0.99, increase tune, and tighten / reparameterize the worst priors.",
            }
        )
    rhat = conv.get("rhat_max")
    rhat_thr = conv.get("rhat_threshold", 1.01)
    if rhat is not None and rhat > rhat_thr:
        suggestions.append(
            {
                "area": "accuracy",
                "priority": "high",
                "issue": f"Max R-hat {rhat:.3f} > {rhat_thr} — chains have not converged.",
                "action": "Increase tune and draws, run ≥4 chains, and address prior-dominated parameters.",
            }
        )
    ess = conv.get("ess_bulk_min")
    ess_thr = conv.get("ess_threshold", 400)
    if ess is not None and ess < ess_thr:
        suggestions.append(
            {
                "area": "accuracy",
                "priority": "medium",
                "issue": f"Min bulk ESS {ess:.0f} < {ess_thr} — effective sample size is low.",
                "action": "Increase draws; if ESS stays low, reparameterize the slow parameters.",
            }
        )

    counts = learn.get("verdict_counts") or {}
    prior_dom = sum(v for k, v in counts.items() if "prior" in str(k).lower())
    if prior_dom:
        worst = ", ".join(
            str(p.get("parameter")) for p in (learn.get("parameters") or [])[:5]
        )
        suggestions.append(
            {
                "area": "accuracy",
                "priority": "medium",
                "issue": f"{prior_dom} parameter(s) look prior-dominated (the data barely moved them): {worst}",
                "action": "Supply more informative priors, add calibrating experiments, or collect more spend variation in those channels.",
            }
        )

    suggestions.append(
        {
            "area": "speed",
            "priority": "low",
            "issue": "Fitting-time levers.",
            "action": "Use a fast NUTS backend (nuts_sampler='numpyro' or 'nutpie'); for quick model checks use method='map'/'advi' (seconds) before a full NUTS run.",
        }
    )

    has_issue = any(s["priority"] in ("high", "medium") for s in suggestions)
    headline = (
        "⚠️ The fit has issues worth addressing:"
        if has_issue
        else "✅ No convergence or learning problems detected — the fit looks healthy."
    )
    lines = ["### Model improvement suggestions", "", headline, ""]
    for s in suggestions:
        lines.append(
            f"- **[{s['priority']}] {s['area']}** — {s['issue']}\n  → {s['action']}"
        )
    res = _ok(
        "\n".join(lines),
        {"garden_tune_suggestions": {"diagnostics": diag, "suggestions": suggestions}},
    )
    try:
        res["tables"] = [
            records_to_table_json(
                suggestions,
                title="Improvement suggestions",
                source="garden_tune_suggestions",
                group="garden",
            )
        ]
    except Exception:  # noqa: BLE001
        pass
    return res


# ── Validation / verification ops (Phase 1) ──────────────────────────────────
# Wrap the framework's already-built validation machinery (validation/*) as
# first-class model-ops so the chat agent can run posterior-predictive checks,
# residual diagnostics, channel collinearity/VIF, causal refutation, out-of-time
# cross-validation, and a one-call battery — each returning a markdown verdict, a
# structured table, and themed plots (routed through _modelop_command's plot
# publishing). Each sub-check is best-effort: a failure becomes an error string,
# never an exception that aborts the turn.


def _fig_json(fig: Any) -> dict:
    """Plotly ``go.Figure`` -> JSON-safe dict (crosses the kernel boundary)."""
    import json as _json

    return _json.loads(fig.to_json())


def posterior_predictive_checks(
    mmm: Any, results: Any = None, *, random_seed: int = 42
) -> dict:
    """Posterior predictive checks: do datasets replicated from the posterior look
    like the observed KPI? Bayesian p-values near 0.5 = good; near 0/1 = misfit."""
    try:
        from mmm_framework.validation.charts import diagnostics as vch
        from mmm_framework.validation.posterior_predictive import PPCValidator

        ppc = PPCValidator(mmm).run(random_seed=random_seed)
        verdict = (
            "✅ adequate"
            if ppc.overall_pass
            else "⚠️ flagged: " + ", ".join(ppc.problematic_checks)
        )
        content = (
            "### Posterior predictive checks\n\n"
            f"**Overall:** {verdict}\n\n"
            "Each check compares a feature of the observed KPI (mean, variance, "
            "autocorrelation, extremes) to the same feature in datasets replicated "
            "from the posterior. A Bayesian p-value near 0.5 means the model "
            "reproduces that feature; near 0 or 1 flags a misfit."
        )
        plots = []
        for title, mk in [
            (
                "PPC density overlay",
                lambda: vch.create_ppc_density_plot(ppc.y_obs, ppc.y_rep),
            ),
            ("PPC test statistics", lambda: vch.create_ppc_statistics_plot(ppc.checks)),
        ]:
            try:
                plots.append({"title": title, "figure": _fig_json(mk())})
            except Exception:  # noqa: BLE001
                pass
        return {
            "content": content,
            "dashboard": {"validation_ppc": ppc.to_dict()},
            "tables": [
                df_to_table_json(
                    ppc.summary(),
                    title="Posterior predictive checks",
                    source="posterior_predictive_checks",
                    group="validation",
                )
            ],
            "plots": plots,
            "error": None,
        }
    except Exception as e:  # noqa: BLE001
        return _err(f"Posterior predictive checks failed: {e}")


def residual_diagnostics(mmm: Any, results: Any = None) -> dict:
    """Residual diagnostics: autocorrelation (Durbin-Watson / Ljung-Box),
    heteroscedasticity (Breusch-Pagan) and normality (Shapiro / Jarque-Bera) of
    the model residuals, plus ACF/PACF and Q-Q plots."""
    try:
        from mmm_framework.validation.charts import diagnostics as vch
        from mmm_framework.validation.residual_diagnostics import ResidualDiagnostics

        rd = ResidualDiagnostics(mmm).run_all()
        verdict = (
            "✅ residuals look adequate"
            if rd.overall_adequate
            else "⚠️ " + "; ".join(rd.recommendations or ["one or more tests failed"])
        )
        content = (
            "### Residual diagnostics\n\n"
            f"**Overall:** {verdict}\n\n"
            "Structure left in the residuals (autocorrelation, changing variance, "
            "non-normality) means the model is missing something — adstock shape, a "
            "control, or the wrong likelihood."
        )
        plots = []
        for title, mk in [
            ("Residual panel", lambda: vch.create_residual_panel(rd)),
            (
                "Residuals vs fitted",
                lambda: vch.create_residual_vs_fitted(rd.residuals, rd.fitted_values),
            ),
            (
                "Residual autocorrelation",
                lambda: vch.create_acf_chart(
                    rd.acf_values, rd.pacf_values, n_obs=len(rd.residuals)
                ),
            ),
            ("Q-Q plot", lambda: vch.create_qq_plot(rd.residuals)),
        ]:
            try:
                plots.append({"title": title, "figure": _fig_json(mk())})
            except Exception:  # noqa: BLE001
                pass
        return {
            "content": content,
            "dashboard": {"validation_residuals": rd.to_dict()},
            "tables": [
                df_to_table_json(
                    rd.summary(),
                    title="Residual diagnostics",
                    source="residual_diagnostics",
                    group="validation",
                )
            ],
            "plots": plots,
            "error": None,
        }
    except Exception as e:  # noqa: BLE001
        return _err(f"Residual diagnostics failed: {e}")


def channel_diagnostics(mmm: Any, results: Any = None) -> dict:
    """Per-channel identifiability: VIF / collinearity clusters, per-channel
    R-hat/ESS. High VIF (>5–10) means a channel's effect can't be cleanly
    separated from another's — its ROI is unstable."""
    try:
        from mmm_framework.validation.channel_diagnostics import ChannelDiagnostics
        from mmm_framework.validation.charts import diagnostics as vch

        cd = ChannelDiagnostics(mmm).run_all()
        flags = []
        if cd.multicollinearity_warning:
            flags.append("multicollinearity")
        if cd.convergence_warning:
            flags.append("per-channel convergence")
        if getattr(cd, "weak_identification_warning", False):
            flags.append("weak identification")
        verdict = (
            "✅ no channel-level red flags" if not flags else "⚠️ " + ", ".join(flags)
        )
        content = (
            "### Channel diagnostics\n\n"
            f"**Overall:** {verdict}\n\n"
            "Per-channel VIF (collinearity), R-hat/ESS, and collinear-cluster "
            "detection."
        )
        for issue in cd.identifiability_issues or []:
            content += f"\n- {issue}"
        plots = []
        try:
            plots.append(
                {
                    "title": "Variance inflation factors",
                    "figure": _fig_json(vch.create_vif_chart(cd)),
                }
            )
        except Exception:  # noqa: BLE001
            pass
        return {
            "content": content,
            "dashboard": {"validation_channels": cd.to_dict()},
            "tables": [
                df_to_table_json(
                    cd.summary(),
                    title="Channel diagnostics",
                    source="channel_diagnostics",
                    group="validation",
                )
            ],
            "plots": plots,
            "error": None,
        }
    except Exception as e:  # noqa: BLE001
        return _err(f"Channel diagnostics failed: {e}")


def refutation_suite(mmm: Any, results: Any = None, *, q: float = 1.0) -> dict:
    """Sensitivity to unobserved confounding: the Robustness Value per channel —
    the share of residual variance an omitted confounder would need to explain
    (in both spend and KPI) to nullify the effect. Low RV = fragile."""
    try:
        from mmm_framework.validation.sensitivity_unobserved import (
            UnobservedConfoundingAnalysis,
        )

        sens = UnobservedConfoundingAnalysis(mmm).run(q=q)
        fragile = sens.fragile_channels
        verdict = (
            "✅ no channel is fragile to plausible unobserved confounding"
            if not fragile
            else "⚠️ fragile to plausible confounding: " + ", ".join(fragile)
        )
        thr = sens.channels[0].fragile_threshold if sens.channels else 0.10
        content = (
            "### Unobserved-confounding robustness\n\n"
            f"**Overall:** {verdict}\n\n"
            f"The Robustness Value is how much residual variance an unobserved "
            f"confounder would need to explain — in both the channel's spend and "
            f"the KPI — to drive its effect to zero. Below {thr:.0%} is fragile.\n\n"
            f"*{sens.caveat}*"
        )
        return {
            "content": content,
            "dashboard": {"validation_refutation": sens.to_dict()},
            "tables": [
                df_to_table_json(
                    sens.summary(),
                    title="Unobserved-confounding robustness",
                    source="refutation_suite",
                    group="validation",
                )
            ],
            "error": None,
        }
    except Exception as e:  # noqa: BLE001
        return _err(f"Refutation suite failed: {e}")


def cross_validation(
    mmm: Any,
    results: Any = None,
    *,
    horizon: int = 13,
    max_origins: int = 2,
    draws: int = 300,
    tune: int = 300,
) -> dict:
    """Out-of-time cross-validation (rolling-origin backtest). REFITS the model on
    expanding windows and grades genuine out-of-sample forecasts vs naive
    baselines — slow (one refit per origin)."""
    try:
        from mmm_framework.validation.backtest import BacktestConfig, run_backtest

        cfg = BacktestConfig(
            horizon=horizon, max_origins=max_origins, draws=draws, tune=tune
        )
        res = run_backtest(mmm, cfg, progressbar=False)
        summ = res.summary()
        content = (
            "### Out-of-time cross-validation\n\n"
            f"Rolling-origin backtest ({res.n_origins} origin(s), horizon "
            f"{horizon}): the model refits on expanding windows and forecasts the "
            "held-out horizon. Beating the seasonal-naive baseline on MAPE is "
            "genuine predictive skill; interval coverage near nominal means honest "
            "uncertainty."
        )
        return {
            "content": content,
            "dashboard": {"validation_cv": {"n_origins": int(res.n_origins)}},
            "tables": [
                df_to_table_json(
                    summ,
                    title="Backtest accuracy (model vs baselines)",
                    source="cross_validation",
                    group="validation",
                )
            ],
            "error": None,
        }
    except Exception as e:  # noqa: BLE001
        return _err(f"Cross-validation failed: {e}")


def validate_model(mmm: Any, results: Any = None, *, random_seed: int = 42) -> dict:
    """One-call validation battery: convergence + posterior-predictive + residual +
    channel-identifiability + confounding-robustness, with a consolidated verdict
    table. Each sub-check degrades gracefully (Error row) rather than aborting."""
    rows: list[dict] = []
    dash: dict = {}
    plots: list[dict] = []

    def _row(check, verdict, detail):
        rows.append({"check": check, "verdict": verdict, "detail": detail})

    try:
        from mmm_framework.diagnostics import compute_fit_diagnostics

        conv = (compute_fit_diagnostics(mmm, results) or {}).get("convergence") or {}
        _row(
            "Convergence",
            "Pass" if conv.get("ok") else "Warn",
            f"R-hat max {conv.get('rhat_max')}, min bulk-ESS "
            f"{conv.get('ess_bulk_min')}, {conv.get('divergences')} divergences",
        )
        dash["validation_convergence"] = conv
    except Exception as e:  # noqa: BLE001
        _row("Convergence", "Error", str(e))

    try:
        from mmm_framework.validation.charts import diagnostics as vch
        from mmm_framework.validation.posterior_predictive import PPCValidator

        ppc = PPCValidator(mmm).run(random_seed=random_seed)
        _row(
            "Posterior predictive",
            "Pass" if ppc.overall_pass else "Warn",
            (
                "reproduces the data"
                if ppc.overall_pass
                else "flagged: " + ", ".join(ppc.problematic_checks)
            ),
        )
        dash["validation_ppc"] = ppc.to_dict()
        try:
            plots.append(
                {
                    "title": "PPC density overlay",
                    "figure": _fig_json(
                        vch.create_ppc_density_plot(ppc.y_obs, ppc.y_rep)
                    ),
                }
            )
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001
        _row("Posterior predictive", "Error", str(e))

    try:
        from mmm_framework.validation.residual_diagnostics import ResidualDiagnostics

        rd = ResidualDiagnostics(mmm).run_all()
        _row(
            "Residuals",
            "Pass" if rd.overall_adequate else "Warn",
            (
                "adequate"
                if rd.overall_adequate
                else "; ".join(rd.recommendations or ["tests failed"])
            ),
        )
        dash["validation_residuals"] = rd.to_dict()
    except Exception as e:  # noqa: BLE001
        _row("Residuals", "Error", str(e))

    try:
        from mmm_framework.validation.channel_diagnostics import ChannelDiagnostics

        cd = ChannelDiagnostics(mmm).run_all()
        cflags = []
        if cd.multicollinearity_warning:
            cflags.append("collinearity")
        if cd.convergence_warning:
            cflags.append("per-channel convergence")
        _row(
            "Channel identifiability",
            "Pass" if not cflags else "Warn",
            "clean" if not cflags else ", ".join(cflags),
        )
        dash["validation_channels"] = cd.to_dict()
    except Exception as e:  # noqa: BLE001
        _row("Channel identifiability", "Error", str(e))

    try:
        from mmm_framework.validation.sensitivity_unobserved import (
            UnobservedConfoundingAnalysis,
        )

        sens = UnobservedConfoundingAnalysis(mmm).run()
        fragile = sens.fragile_channels
        _row(
            "Confounding robustness",
            "Pass" if not fragile else "Warn",
            "robust" if not fragile else "fragile: " + ", ".join(fragile),
        )
        dash["validation_refutation"] = sens.to_dict()
    except Exception as e:  # noqa: BLE001
        _row("Confounding robustness", "Error", str(e))

    n_warn = sum(1 for r in rows if r["verdict"] != "Pass")
    headline = (
        "✅ Model passes all validation checks"
        if n_warn == 0
        else f"⚠️ {n_warn} of {len(rows)} checks need attention"
    )
    content = (
        "### Model validation battery\n\n"
        f"**{headline}**\n\n"
        "Ran convergence, posterior-predictive, residual, channel-identifiability "
        "and confounding-robustness checks. The table shows each verdict; run the "
        "individual tools (e.g. residual diagnostics, refutation suite) for the "
        "detail tables and plots."
    )
    return {
        "content": content,
        "dashboard": dash,
        "tables": [
            records_to_table_json(
                rows,
                title="Validation battery",
                source="validate_model",
                group="validation",
                columns=[
                    {"key": "check", "label": "Check", "type": "string"},
                    {"key": "verdict", "label": "Verdict", "type": "string"},
                    {"key": "detail", "label": "Detail", "type": "string"},
                ],
            )
        ],
        "plots": plots,
        "error": None,
    }


def slide_deck_notes(
    mmm: Any,
    results: Any = None,
    *,
    client: str | None = None,
    kpi_name: str = "Revenue",
    currency: str = "$",
    break_even: float = 1.0,
    margin: float | None = None,
    hdi_prob: float = 0.8,
) -> dict:
    """Deterministic deck OUTLINE: build the deck and return each slide's light
    facts (key / kind / title / notes / scalar metrics / is_summary) for the AI
    insight + synthesis pass. No charts, no AI."""
    try:
        from mmm_framework.agents.deck_insights import light_metrics, slide_key
        from mmm_framework.reporting.deck import build_deck

        deck = build_deck(
            mmm,
            results,
            client=client,
            kpi_name=kpi_name,
            currency=currency,
            break_even=break_even,
            margin=margin,
            hdi_prob=hdi_prob,
        )
        notes = [
            {
                "key": slide_key(s),
                "kind": s.kind,
                "title": s.title,
                "notes": s.notes,
                "is_summary": s.is_summary,
                "metrics": light_metrics(s),
            }
            for s in deck.slides
        ]
        return _ok(f"Outlined {len(notes)} slides.", {"slide_deck_notes": notes})
    except Exception as e:  # noqa: BLE001
        return _err(f"Error building deck outline: {e}")


def render_slide_deck(
    mmm: Any,
    results: Any = None,
    *,
    insights: dict | None = None,
    client: str | None = None,
    kpi_name: str = "Revenue",
    currency: str = "$",
    break_even: float = 1.0,
    margin: float | None = None,
    hdi_prob: float = 0.8,
    template_path: str | None = None,
    filename: str = "agent_slide_deck.pptx",
) -> dict:
    """Render the .pptx by filling the template from the model + (optional) AI
    ``insights``, writing it into the session workspace. Deterministic given
    ``insights``."""
    try:
        from mmm_framework.agents import workspace as _ws
        from mmm_framework.agents.runtime import get_current_thread
        from mmm_framework.reporting.deck.builder import build_pptx

        out = str(_ws.report_path(filename, get_current_thread()))
        build_pptx(
            mmm,
            template_path=template_path,
            out_path=out,
            insights=insights or {},
            client=client,
            kpi_name=kpi_name,
            currency=currency,
            break_even=break_even,
            margin=margin,
            hdi_prob=hdi_prob,
        )
        return _ok(
            "Rendered the slide deck.",
            {
                "slide_deck": {
                    "path": out,
                    "filename": filename,
                    "thread": get_current_thread(),
                }
            },
        )
    except Exception as e:  # noqa: BLE001
        return _err(f"Error rendering slide deck: {e}")


OPS = {
    "roi_metrics": roi_metrics,
    "posterior_predictive_checks": posterior_predictive_checks,
    "residual_diagnostics": residual_diagnostics,
    "channel_diagnostics": channel_diagnostics,
    "refutation_suite": refutation_suite,
    "cross_validation": cross_validation,
    "validate_model": validate_model,
    "compute_estimands": compute_estimands,
    "garden_compat": garden_compat,
    "garden_tune_suggestions": garden_tune_suggestions,
    "component_decomposition": component_decomposition,
    "model_diagnostics": model_diagnostics,
    "adstock_weights": adstock_weights,
    "saturation_curves": saturation_curves,
    "budget_scenario": budget_scenario,
    "plan_budget": plan_budget,
    "plan_scenario": plan_scenario,
    "marginal_analysis": marginal_analysis,
    "prior_predictive_check": prior_predictive_check,
    "leave_one_out": leave_one_out,
    "save_model": save_model,
    "optimize_budget": optimize_budget,
    "experiment_design": experiment_design,
    "experiment_priorities": experiment_priorities,
    "experiment_economics": experiment_economics,
    "experiment_optimizer": experiment_optimizer,
    "identify_structural_parameters": identify_structural_parameters,
    "slide_deck_notes": slide_deck_notes,
    "render_slide_deck": render_slide_deck,
}
