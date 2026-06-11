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

from typing import Any

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
    max_draws: int = 200,
) -> dict:
    """Optimal budget allocation from the fitted model's response curves, with
    per-draw re-optimization for allocation stability. Returns markdown +
    a dashboard payload + a structured table."""
    try:
        from mmm_framework.planning import optimize_budget as _optimize

        res = _optimize(
            mmm,
            total_budget=total_budget,
            budget_change_pct=budget_change_pct,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
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

        table, designs = _recommend(
            mmm, top_k=int(top_k), max_draws=max_draws, random_seed=42
        )
    except Exception as e:  # noqa: BLE001
        return _err(f"Experiment design recommendation failed: {e}")

    lines = ["### Experiment Design Recommendations", ""]
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
    "save_model": save_model,
    "optimize_budget": optimize_budget,
    "experiment_design": experiment_design,
}
