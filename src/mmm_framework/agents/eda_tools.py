"""Agent tools for pre-fit data quality: validation, EDA, outlier
detection + remediation.

Thin wrappers around :mod:`mmm_framework.eda` following the same contract as
the tools in ``agents/tools.py``: ``@tool`` + injected state/config, returning
a ``Command`` whose update carries a ToolMessage, ``dashboard_data`` (with
content-addressed plot refs), and — for ``apply_outlier_treatment`` — a new
``dataset_path`` / ``model_spec``.

Helper imports from ``agents.tools`` happen inside functions (the same
direction as ``causal_tools``) because ``tools.py`` imports ``EDA_TOOLS`` from
here at module load.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any, Optional

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

InjectedConfig = Annotated[RunnableConfig, InjectedToolArg]

_OUTLIER_REPORT_RELPATH = Path("eda") / "outlier_report.json"

EDA_ANALYSES = (
    "profile",
    "distributions",
    "correlation",
    "collinearity",
    "spend_share",
    "seasonality",
    "kpi_vs_media",
    "stationarity",
)


# ── shared helpers ────────────────────────────────────────────────────────────


def _load_panel(state: dict):
    """Load the EDAPanel for the session dataset. Returns (panel, error_msg)."""
    ds_path = state.get("dataset_path")
    if not ds_path or not Path(ds_path).exists():
        return None, (
            "No dataset loaded. Upload a dataset or call `generate_synthetic_data` "
            "first."
        )
    from mmm_framework.agents.tools import _normalized_spec
    from mmm_framework.eda import load_eda_panel

    try:
        panel = load_eda_panel(ds_path, _normalized_spec(state.get("model_spec")))
    except Exception as exc:
        return None, f"Could not load dataset for analysis: {exc}"
    return panel, None


def _publish_figures(
    figs: list[tuple[str, Any]],
    dashboard_data: dict,
    thread_id: str | None,
) -> tuple[list[dict], int]:
    """Normalize -> serialize -> content-address figures; append {'id','title'}
    refs to ``dashboard_data['plots']``. Mirrors the ``execute_python`` flow
    (tools.py): oversize/invalid figures are dropped and counted, never inlined.
    """
    from mmm_framework.agents import workspace as _ws
    from mmm_framework.agents.branding import (
        apply_brand_colors,
        is_active as _brand_active,
        resolve_branding,
    )
    from mmm_framework.agents.tools import _normalize_figure

    existing = dashboard_data.get("plots", [])
    refs: list[dict] = []
    dropped = 0
    branding = resolve_branding(thread_id)
    for title, fig in figs:
        try:
            fig_json = json.loads(_normalize_figure(fig).to_json())
            if _brand_active(branding):
                fig_json = apply_brand_colors(fig_json, branding)
            pid = _ws.store_plot(fig_json, thread_id)
        except ValueError as exc:
            dropped += 1
            logging.getLogger("mmm_audit").warning(
                "plot_rejected thread=%s reason=%s", thread_id, exc
            )
            continue
        except Exception:
            dropped += 1
            continue
        refs.append({"id": pid, "title": title})
    dashboard_data["plots"] = existing + refs
    return refs, dropped


def _plots_note(refs: list[dict], dropped: int) -> str:
    note = ""
    if refs:
        note += (
            f"\n\n*Generated {len(refs)} interactive chart(s). "
            "View them in the Plots tab.*"
        )
    if dropped:
        note += f"\n\n*{dropped} chart(s) omitted (too large or invalid).*"
    return note


def _update_eda_envelope(
    dashboard_data: dict,
    *,
    issues: list[dict] | None = None,
    actions: list[dict] | None = None,
    applied_ids: list[str] | None = None,
    damaged: list[str] | None = None,
) -> dict:
    """Maintain the compact ``dashboard_data['eda']`` envelope the frontend EDA
    tab renders (issues with severity, outlier actions with status). The
    verbose ``data_quality.*`` blobs are kept unchanged for back-compat."""
    import time

    env = dict(dashboard_data.get("eda") or {})
    if issues is not None:
        env["issues"] = issues
    if actions is not None:
        env["outlier_actions"] = actions
    if applied_ids:
        applied = set(applied_ids)
        env["outlier_actions"] = [
            ({**a, "status": "applied"} if a.get("action_id") in applied else a)
            for a in (env.get("outlier_actions") or [])
        ]
    if damaged is not None:
        env["normalization_damaged"] = damaged
    env["updated_at"] = time.time()
    dashboard_data["eda"] = env
    return env


def _aggregate_kpi_series(panel, variable: str):
    """A variable as one Period-indexed series (panel slices summed)."""
    col = panel.df_wide[variable].astype(float)
    if panel.dims:
        col = col.groupby(level=panel.date_col).sum(min_count=1)
    return col


# ── tools ─────────────────────────────────────────────────────────────────────


@tool
def validate_data(
    state: Annotated[dict, InjectedState],
    strict: bool = False,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    Run pre-fit data-quality validation on the loaded dataset BEFORE configuring
    or fitting a model: missingness, date gaps, duplicate rows, constant series,
    zero-inflated media, negative spend, scale (unit) mismatches, short history
    vs. the configured spec's parameter count, and cross-geo/product consistency.

    Errors are problems that will break or silently bias a fit; warnings need a
    judgment call — surface them to the user. Set strict=True to treat warnings
    as blocking when deciding whether to proceed to fit_mmm_model.
    """
    from mmm_framework.agents.tools import _activate_thread

    tid = _activate_thread(config)
    panel, err = _load_panel(state)
    if err:
        return Command(
            update={"messages": [ToolMessage(content=err, tool_call_id=tool_call_id)]}
        )

    from mmm_framework.agents.tools import _normalized_spec
    from mmm_framework.eda import missingness_matrix, validate_dataset
    from mmm_framework.eda.charts import fig_missingness

    report = validate_dataset(panel, spec=_normalized_spec(state.get("model_spec")))

    lines = [f"### Data validation — {'PASSED' if report.passed else 'FAILED'}"]
    lines.append(
        f"{report.n_variables} variables x {report.n_periods} periods "
        f"(roles from {panel.roles_source})."
    )
    order = {"error": 0, "warning": 1, "info": 2}
    issues_sorted = sorted(report.issues, key=lambda x: order.get(x.severity, 3))
    if report.issues:
        # Errors verbatim (the LLM must act on them); warnings/info summarized —
        # the full formatted table is in the dashboard.
        errors = [i for i in issues_sorted if i.severity == "error"]
        if errors:
            lines.append("\n| Severity | Check | Variable | Finding |")
            lines.append("|---|---|---|---|")
            for i in errors:
                lines.append(
                    f"| {i.severity.upper()} | {i.check} | {i.variable or '—'} | "
                    f"{i.message} |"
                )
        others = [i for i in issues_sorted if i.severity != "error"]
        if others:
            by_check: dict[str, int] = {}
            for i in others:
                by_check[i.check] = by_check.get(i.check, 0) + 1
            lines.append(
                f"\n{len(others)} warning/info finding(s): "
                + ", ".join(f"{c} ×{n}" for c, n in by_check.items())
                + ". Full detail in the EDA tab."
            )
    else:
        lines.append("\nNo issues found.")
    n_errors = len(report.by_severity("error"))
    n_warnings = len(report.by_severity("warning"))
    if strict and (n_errors or n_warnings):
        lines.append(
            f"\n**Strict mode:** {n_errors} error(s) + {n_warnings} warning(s) — "
            "resolve (or get explicit user sign-off) before fitting."
        )
    elif n_errors:
        lines.append(
            f"\n**{n_errors} error(s) must be resolved before fitting.** "
            "Discuss the warnings with the user."
        )

    dashboard_data = dict(state.get("dashboard_data") or {})
    dq = dict(dashboard_data.get("data_quality") or {})
    dq["validation"] = report.to_dict()
    dashboard_data["data_quality"] = dq
    issue_records = [
        {
            "severity": i.severity,
            "check": i.check,
            "variable": i.variable or "",
            "message": i.message,
        }
        for i in issues_sorted
    ]
    _update_eda_envelope(dashboard_data, issues=issue_records)

    figs = [("Data availability", fig_missingness(missingness_matrix(panel)))]
    refs, dropped = _publish_figures(figs, dashboard_data, tid)
    note = _plots_note(refs, dropped)
    if issue_records:
        from mmm_framework.agents.tables import (
            publish_tables,
            records_to_table_json,
            tables_note,
        )

        trefs, tdropped = publish_tables(
            [
                records_to_table_json(
                    issue_records,
                    title="Data Validation Findings",
                    source="validate_data",
                    group="eda",
                )
            ],
            dashboard_data,
            tid,
        )
        note += tables_note(trefs, tdropped)

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="\n".join(lines) + note,
                    tool_call_id=tool_call_id,
                )
            ],
            "dashboard_data": dashboard_data,
        }
    )


@tool
def run_eda(
    state: Annotated[dict, InjectedState],
    analyses: Optional[list[str]] = None,
    variables: Optional[list[str]] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    MMM-aware exploratory data analysis with interactive charts.

    analyses: any subset of ["profile", "distributions", "correlation",
    "collinearity", "spend_share", "seasonality", "kpi_vs_media",
    "stationarity"]; default runs all. variables: optionally restrict to
    specific variables (default: roles from the model spec, or all).

    Run after the dataset is loaded — ideally before configure_model — to
    understand scales, correlations/VIF (channels that cannot be separated),
    spend concentration, seasonality/trend strength, and stationarity. Charts
    land in the Plots tab; the message returns the key numbers.
    """
    from mmm_framework.agents.tools import _activate_thread

    tid = _activate_thread(config)
    panel, err = _load_panel(state)
    if err:
        return Command(
            update={"messages": [ToolMessage(content=err, tool_call_id=tool_call_id)]}
        )

    requested = [a for a in (analyses or list(EDA_ANALYSES)) if a in EDA_ANALYSES]
    unknown = [a for a in (analyses or []) if a not in EDA_ANALYSES]

    from mmm_framework.eda import (
        EDAConfig,
        collinearity_analysis,
        decompose_series,
        profile_panel,
        seasonal_period_for_freq,
        spend_share,
        stationarity_tests,
    )
    from mmm_framework.eda.charts import (
        fig_correlation_heatmap,
        fig_decomposition,
        fig_distributions,
        fig_kpi_vs_media,
        fig_spend_share,
        fig_sparkline_grid,
        fig_stationarity,
        fig_vif,
    )

    cfg = EDAConfig()
    sections: dict[str, Any] = {}
    figs: list[tuple[str, Any]] = []
    tables: list[dict] = []
    lines: list[str] = ["### Exploratory data analysis"]
    if unknown:
        lines.append(f"(Ignored unknown analyses: {', '.join(unknown)}.)")

    period = seasonal_period_for_freq(panel.freq)
    focus = [v for v in (variables or []) if v in panel.variables] or None

    if "profile" in requested:
        prof = profile_panel(panel)
        sections["profile"] = prof.to_dict(orient="records")
        lines.append(
            f"\n**Profile** — {len(panel.variables)} variables "
            f"({len(panel.media)} media, {len(panel.controls)} controls, "
            f"KPI: {panel.kpi or 'unknown'}; roles from {panel.roles_source}). "
            f"Frequency: {panel.freq or 'unknown'}."
        )
        prof_records = prof.to_dict(orient="records")
        lines.append("| Variable | Role | Missing % | Zero % | Mean | Max |")
        lines.append("|---|---|---|---|---|---|")
        for rec in prof_records[:15]:
            lines.append(
                f"| {rec['variable']} | {rec['role']} | "
                f"{rec.get('missing_pct', 0):.1f} | {rec.get('zero_pct', 0):.1f} | "
                f"{rec.get('mean', float('nan')):.4g} | "
                f"{rec.get('max', float('nan')):.4g} |"
            )
        if len(prof_records) > 15:
            lines.append(
                f"\n({len(prof_records) - 15} more variable(s) — full profile "
                "table in the dashboard.)"
            )
        from mmm_framework.agents.tables import df_to_table_json

        tables.append(
            df_to_table_json(
                prof, title="Variable Profile", source="run_eda", group="eda"
            )
        )
        figs.append(("Variables over time", fig_sparkline_grid(panel, focus)))

    if "distributions" in requested:
        figs.append(("Value distributions", fig_distributions(panel, focus)))

    coll = None
    if "correlation" in requested or "collinearity" in requested:
        coll = collinearity_analysis(panel, cfg, focus)
        sections["collinearity"] = {k: v for k, v in coll.items() if k != "correlation"}

    if "correlation" in requested and coll and len(coll["correlation"]):
        top = coll["top_pairs"][:5]
        if top:
            pairs_txt = ", ".join(f"{p['a']}~{p['b']} r={p['r']:.2f}" for p in top)
            lines.append(f"\n**Top correlations**: {pairs_txt}")
        if coll["top_pairs"]:
            from mmm_framework.agents.tables import records_to_table_json

            tables.append(
                records_to_table_json(
                    [
                        {"variable_a": p["a"], "variable_b": p["b"], "r": p["r"]}
                        for p in coll["top_pairs"][:50]
                    ],
                    title="Top Correlations",
                    source="run_eda",
                    group="eda",
                )
            )
        figs.append(
            ("Correlation matrix", fig_correlation_heatmap(coll["correlation"]))
        )

    if "collinearity" in requested and coll and coll["vif"]:
        if coll["high_vif"]:
            lines.append(
                f"\n**Collinearity**: VIF > {cfg.vif_threshold:g} for "
                f"{', '.join(coll['high_vif'])} — their individual effects will be "
                "weakly identified."
            )
        for cl in coll["clusters"]:
            lines.append(
                f"- Cluster {cl['channels']} (|r| up to {cl['max_correlation']:.2f})"
            )
        if coll["condition_number"]:
            lines.append(
                f"- Design condition number: {coll['condition_number']:.1f}"
                + (" (> 30: ill-conditioned)" if coll["condition_number"] > 30 else "")
            )
        from mmm_framework.agents.tables import records_to_table_json

        tables.append(
            records_to_table_json(
                [
                    {
                        "channel": ch,
                        "vif": v,
                        "high": ch in (coll["high_vif"] or []),
                    }
                    for ch, v in coll["vif"].items()
                ],
                title="Variance Inflation Factors",
                source="run_eda",
                group="eda",
            )
        )
        figs.append(("VIF", fig_vif(coll["vif"], cfg.vif_threshold)))

    if "spend_share" in requested:
        share = spend_share(panel)
        sections["spend_share"] = {
            "totals": share["totals"],
            "shares": share["shares"],
            "hhi": share["hhi"],
        }
        if share["hhi"] is not None:
            shares_txt = ", ".join(f"{c}: {s:.0%}" for c, s in share["shares"].items())
            lines.append(f"\n**Spend share** — {shares_txt} (HHI {share['hhi']:.2f}).")
            figs.append(
                (
                    "Spend share over time",
                    fig_spend_share(
                        share["share_over_time"], share["shares"], share["hhi"]
                    ),
                )
            )

    if "seasonality" in requested:
        season_vars = focus or [v for v in [panel.kpi] if v]
        season_info = {}
        for var in season_vars:
            series = _aggregate_kpi_series(panel, var)
            result = decompose_series(series, period, variable=var)
            season_info[var] = result.to_dict()
            figs.append((f"Decomposition — {var}", fig_decomposition(result)))
            strength_txt = f"trend strength {result.trend_strength:.2f}"
            if result.seasonal_strength is not None:
                strength_txt += f", seasonal strength {result.seasonal_strength:.2f}"
            lines.append(
                f"\n**Seasonality ({var})** — {result.method}: {strength_txt}."
            )
        sections["seasonality"] = season_info

    if "kpi_vs_media" in requested and panel.kpi and panel.media:
        figs.append((f"{panel.kpi} vs media", fig_kpi_vs_media(panel)))

    if "stationarity" in requested:
        stat_vars = focus or [v for v in [panel.kpi, *panel.media] if v]
        stat = {
            var: stationarity_tests(_aggregate_kpi_series(panel, var))
            for var in stat_vars
        }
        sections["stationarity"] = stat
        verdicts = ", ".join(f"{v}: {r['verdict']}" for v, r in stat.items())
        lines.append(f"\n**Stationarity** — {verdicts}.")
        figs.append(("Stationarity", fig_stationarity(stat)))

    dashboard_data = dict(state.get("dashboard_data") or {})
    dq = dict(dashboard_data.get("data_quality") or {})
    from mmm_framework.eda.results import _json_safe

    dq["eda"] = _json_safe(sections)
    dashboard_data["data_quality"] = dq
    refs, dropped = _publish_figures(figs, dashboard_data, tid)
    note = _plots_note(refs, dropped)
    if tables:
        from mmm_framework.agents.tables import publish_tables, tables_note

        trefs, tdropped = publish_tables(tables, dashboard_data, tid)
        note += tables_note(trefs, tdropped)

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="\n".join(lines) + note,
                    tool_call_id=tool_call_id,
                )
            ],
            "dashboard_data": dashboard_data,
        }
    )


@tool
def detect_outliers(
    state: Annotated[dict, InjectedState],
    variables: Optional[list[str]] = None,
    sensitivity: str = "default",
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    Time-series-aware outlier detection on the KPI and media series, with
    concrete remediation recommendations. Seasonal peaks are NOT flagged;
    the detectors target: isolated media spend spikes (data-entry errors that
    corrupt the channel's max-normalization and flatten its saturation curve),
    missed-load zeros in always-on channels, KPI demand shocks (promos —
    modeled with event dummies, not deleted), heavy-tailed noise patterns, and
    sustained level shifts (trend breaks). Note small errors (e.g. a 2x
    double-count) are statistically indistinguishable from heavy flight weeks
    and will NOT be flagged.

    sensitivity: "low" | "default" | "high". Each recommendation has an
    action_id — review them WITH THE USER, then apply the confirmed ones via
    apply_outlier_treatment(action_ids=[...]). Never apply unconfirmed
    treatments.
    """
    from mmm_framework.agents import workspace as _ws
    from mmm_framework.agents.tools import _activate_thread

    tid = _activate_thread(config)
    panel, err = _load_panel(state)
    if err:
        return Command(
            update={"messages": [ToolMessage(content=err, tool_call_id=tool_call_id)]}
        )

    from mmm_framework.eda import OutlierConfig
    from mmm_framework.eda import detect_outliers as _detect
    from mmm_framework.eda import recommend_treatments
    from mmm_framework.eda.charts import fig_outlier_series, fig_outlier_severity

    cfg = OutlierConfig.for_sensitivity(sensitivity)
    report = _detect(panel, cfg, variables)
    report.actions = recommend_treatments(panel, report.flags, cfg)

    # Persist the full report so apply_outlier_treatment can resolve action_ids
    # server-side (with a staleness stamp for the source dataset).
    ds_path = state.get("dataset_path")
    try:
        report.dataset_path = ds_path
        report.dataset_mtime = Path(ds_path).stat().st_mtime if ds_path else None
        out = _ws.thread_dir(tid) / _OUTLIER_REPORT_RELPATH
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    except Exception:
        logging.getLogger("mmm_audit").warning(
            "outlier_report_persist_failed thread=%s", tid, exc_info=True
        )

    lines = ["### Outlier detection"]
    if not report.flags:
        lines.append(
            f"No outliers found at sensitivity `{sensitivity}` across "
            f"{', '.join(panel.media + ([panel.kpi] if panel.kpi else []))}."
        )
    else:
        lines.append("\n| Flag | Kind | Value | Expected | Score | Methods |")
        lines.append("|---|---|---|---|---|---|")
        for f in sorted(report.flags, key=lambda x: -x.score):
            lines.append(
                f"| {f.flag_id} | {f.kind} | {f.value:.4g} | {f.expected:.4g} | "
                f"{f.score:.2f} | {','.join(f.methods)} |"
            )
        damaged = [
            v for v, s in report.per_variable.items() if s.get("normalization_damaged")
        ]
        if damaged:
            lines.append(
                f"\n⚠️ **Normalization damage**: in {', '.join(damaged)} a single "
                "point sets the channel's saturation scale — fitting as-is will "
                "flatten those response curves."
            )
        if report.actions:
            lines.append(
                "\n**Recommended treatments** (apply with "
                "`apply_outlier_treatment` after user confirmation):"
            )
            lines.append("\n| action_id | Strategy | Rationale |")
            lines.append("|---|---|---|")
            for a in report.actions:
                lines.append(f"| `{a.action_id}` | {a.strategy} | {a.rationale} |")

    figs: list[tuple[str, Any]] = []
    flagged_vars = sorted({f.variable for f in report.flags})
    for var in flagged_vars:
        series = _aggregate_kpi_series(panel, var)
        var_flags = [f for f in report.flags if f.variable == var]
        figs.append(
            (f"Outliers — {var}", fig_outlier_series(series, var_flags, variable=var))
        )
    if report.flags:
        figs.append(("Outlier severity", fig_outlier_severity(report.flags)))

    dashboard_data = dict(state.get("dashboard_data") or {})
    dq = dict(dashboard_data.get("data_quality") or {})
    dq["outliers"] = report.to_dict()
    dashboard_data["data_quality"] = dq
    damaged = [
        v for v, s in report.per_variable.items() if s.get("normalization_damaged")
    ]
    action_records = [
        {
            "action_id": a.action_id,
            "strategy": a.strategy,
            "variable": (a.flag_ids[0].split("@")[0] if a.flag_ids else ""),
            "rationale": a.rationale,
            "status": "proposed",
        }
        for a in report.actions
    ]
    _update_eda_envelope(dashboard_data, actions=action_records, damaged=damaged)
    refs, dropped = _publish_figures(figs, dashboard_data, tid)
    note = _plots_note(refs, dropped)
    if report.flags:
        from mmm_framework.agents.tables import (
            publish_tables,
            records_to_table_json,
            tables_note,
        )

        tables = [
            records_to_table_json(
                [
                    {
                        "flag_id": f.flag_id,
                        "variable": f.variable,
                        "kind": f.kind,
                        "value": float(f.value),
                        "expected": float(f.expected),
                        "score": float(f.score),
                        "methods": ",".join(f.methods),
                    }
                    for f in sorted(report.flags, key=lambda x: -x.score)
                ],
                title="Outlier Flags",
                source="detect_outliers",
                group="eda",
            )
        ]
        if action_records:
            tables.append(
                records_to_table_json(
                    action_records,
                    title="Recommended Treatments",
                    source="detect_outliers",
                    group="eda",
                )
            )
        trefs, tdropped = publish_tables(tables, dashboard_data, tid)
        note += tables_note(trefs, tdropped)

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="\n".join(lines) + note,
                    tool_call_id=tool_call_id,
                )
            ],
            "dashboard_data": dashboard_data,
        }
    )


def _apply_outlier_treatment_core(
    state: dict,
    tid: str,
    action_ids: list[str],
    reason: str | None,
) -> tuple[str | None, str | None, dict]:
    """Shared implementation behind the ``apply_outlier_treatment`` tool and the
    ``POST /outliers/{thread_id}/apply`` endpoint (UI confirm buttons).

    Returns ``(error, summary_markdown, state_update)`` where ``state_update``
    contains NO messages (the tool wrapper adds the ToolMessage; the endpoint
    must apply a state-only update — appending an orphan ToolMessage from
    outside a tool call corrupts Anthropic threads)."""
    from mmm_framework.agents import workspace as _ws

    report_file = _ws.thread_dir(tid) / _OUTLIER_REPORT_RELPATH
    if not report_file.exists():
        return (
            "No outlier report found for this session — run `detect_outliers` first.",
            None,
            {},
        )

    from mmm_framework.eda.results import OutlierReport

    try:
        report = OutlierReport.from_dict(json.loads(report_file.read_text()))
    except Exception as exc:
        return (
            f"Could not read the outlier report ({exc}) — rerun `detect_outliers`.",
            None,
            {},
        )

    ds_path = state.get("dataset_path")
    if not ds_path or not Path(ds_path).exists():
        return ("No dataset loaded — nothing to treat.", None, {})
    stale = report.dataset_path != ds_path
    try:
        stale = stale or (
            report.dataset_mtime is not None
            and Path(ds_path).stat().st_mtime != report.dataset_mtime
        )
    except OSError:
        stale = True
    if stale:
        return (
            "The outlier report is stale (the dataset changed since it was "
            "produced). Rerun `detect_outliers` and use fresh action_ids.",
            None,
            {},
        )

    by_id = {a.action_id: a for a in report.actions}
    selected = [by_id[a] for a in action_ids if a in by_id]
    missing = [a for a in action_ids if a not in by_id]
    if missing:
        return (
            f"Unknown action_id(s): {', '.join(missing)}. Valid ids: "
            f"{', '.join(by_id) or '(none)'}.",
            None,
            {},
        )
    if not selected:
        return ("No actions selected — pass at least one action_id.", None, {})

    panel, err = _load_panel(state)
    if err:
        return (err, None, {})

    data_actions = [
        a
        for a in selected
        if a.strategy in ("winsorize", "impute", "dummy", "exclude_periods")
    ]
    new_dataset_path = None
    applied_lines: list[str] = []
    figs: list[tuple[str, Any]] = []

    if data_actions:
        if panel.df_long is None:
            return (
                "Data treatments need the MFF (long) dataset format; this dataset "
                "is a wide CSV. Apply changes via `execute_python` instead.",
                None,
                {},
            )
        from mmm_framework.eda import apply_treatments
        from mmm_framework.eda.charts import fig_before_after

        treated = apply_treatments(
            panel.df_long,
            data_actions,
            report.flags,
            date_col=(
                panel.date_col if panel.date_col in panel.df_long.columns else "Period"
            ),
            kpi=panel.kpi,
        )
        out_dir = _ws.thread_dir(tid)
        before_snapshot = _ws.snapshot_dir(out_dir)
        new_dataset_path = str(out_dir / f"treated_{Path(ds_path).stem}.csv")
        treated.to_csv(new_dataset_path, index=False)
        try:
            _ws.register_generated_files(tid, before_snapshot, kind="dataset")
        except Exception:
            pass

        from mmm_framework.eda import load_eda_panel

        treated_panel = load_eda_panel(new_dataset_path, None)
        for a in data_actions:
            applied_lines.append(f"- `{a.action_id}` ({a.strategy})")
            if a.strategy in ("winsorize", "impute"):
                flag_var = a.flag_ids[0].split("@")[0]
                if flag_var in panel.variables and flag_var in treated_panel.variables:
                    before_s = _aggregate_kpi_series(panel, flag_var)
                    after_s = _aggregate_kpi_series(treated_panel, flag_var)
                    figs.append(
                        (
                            f"Treatment — {flag_var}",
                            fig_before_after(before_s, after_s, flag_var),
                        )
                    )

    # Spec changes: event-dummy controls + setting updates (e.g. trend.type).
    import copy as _copy

    from mmm_framework.agents.tools import _normalized_spec

    candidate = _copy.deepcopy(_normalized_spec(state.get("model_spec")))
    spec_changed = False
    for a in selected:
        sc = a.spec_change or {}
        if "add_control" in sc:
            controls = candidate.setdefault("control_variables", [])
            if not any(c.get("name") == sc["add_control"] for c in controls):
                controls.append({"name": sc["add_control"]})
                spec_changed = True
                applied_lines.append(
                    f"- added control `{sc['add_control']}` to the model spec"
                )
        elif "setting_path" in sc:
            node = candidate
            parts = sc["setting_path"].split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = sc["value"]
            spec_changed = True
            applied_lines.append(f"- set `{sc['setting_path']}` = `{sc['value']}`")
        if a.strategy == "note" and not sc:
            applied_lines.append(f"- noted: {a.rationale}")

    summary = ["### Outlier treatment applied", *applied_lines]
    if new_dataset_path:
        summary.append(
            f"\nTreated dataset written to `{Path(new_dataset_path).name}` "
            "(the original file is untouched); the session now uses the treated "
            "copy."
        )
    summary.append(
        '\nRecord this with `record_assumption` (category: "data") so the '
        "audit trail captures why the data was modified."
    )

    update: dict = {}
    if spec_changed:
        # Same lock semantics as tools._commit_spec, but message-free: conflicting
        # changes to user-locked fields are reverted into pending_spec_changes.
        from mmm_framework.agents.spec_locks import merge_pending, reconcile_with_locks

        current = state.get("model_spec") or {}
        locked = list(state.get("locked_fields") or [])
        merged, new_pending = reconcile_with_locks(
            candidate, current, locked, reason=reason or "outlier remediation"
        )
        pending = merge_pending(state.get("pending_spec_changes"), new_pending)
        if new_pending:
            blocked = ", ".join(f"`{p['path']}`" for p in new_pending)
            summary.append(
                f"\n⚠️ {len(new_pending)} change(s) touch fields the user locked "
                f"manually ({blocked}). These were NOT applied — they need "
                "explicit user confirmation."
            )
        dashboard_data = dict(state.get("dashboard_data") or {})
        dashboard_data["model_spec"] = merged
        dashboard_data["locked_fields"] = locked
        dashboard_data["pending_spec_changes"] = pending
        update["model_spec"] = merged
        update["pending_spec_changes"] = pending
    else:
        dashboard_data = dict(state.get("dashboard_data") or {})

    _update_eda_envelope(dashboard_data, applied_ids=[a.action_id for a in selected])
    refs, dropped = _publish_figures(figs, dashboard_data, tid)
    if refs or dropped:
        summary.append(_plots_note(refs, dropped))
    update["dashboard_data"] = dashboard_data
    if new_dataset_path:
        update["dataset_path"] = new_dataset_path

    return None, "\n".join(summary), update


@tool
def apply_outlier_treatment(
    state: Annotated[dict, InjectedState],
    action_ids: list[str],
    reason: str = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    Apply remediation actions from the latest detect_outliers report by their
    action_ids. ONLY call this after the user has confirmed which actions to
    apply.

    Winsorize/exclude actions write a treated COPY of the dataset
    (treated_<name>.csv) into the session workspace and switch dataset_path to
    it — the original file is never modified. Dummy actions also add the event
    dummy as a control variable in the model spec; trend actions update the
    trend settings. Afterwards, record the treatment with record_assumption
    (category: "data").
    """
    from mmm_framework.agents.tools import _activate_thread

    tid = _activate_thread(config)
    error, summary, update = _apply_outlier_treatment_core(
        state, tid, action_ids, reason
    )
    if error:
        return Command(
            update={"messages": [ToolMessage(content=error, tool_call_id=tool_call_id)]}
        )
    update["messages"] = [ToolMessage(content=summary, tool_call_id=tool_call_id)]
    return Command(update=update)


EDA_TOOLS = [validate_data, run_eda, detect_outliers, apply_outlier_treatment]

__all__ = [
    "validate_data",
    "run_eda",
    "detect_outliers",
    "apply_outlier_treatment",
    "EDA_TOOLS",
]
