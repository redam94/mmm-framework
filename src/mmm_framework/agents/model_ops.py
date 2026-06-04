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


# Registry: the name -> op map the kernel dispatch (PR-B) resolves against.
OPS = {
    "roi_metrics": roi_metrics,
    "component_decomposition": component_decomposition,
    "model_diagnostics": model_diagnostics,
    "adstock_weights": adstock_weights,
    "saturation_curves": saturation_curves,
    "budget_scenario": budget_scenario,
    "marginal_analysis": marginal_analysis,
}
