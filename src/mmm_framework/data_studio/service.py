"""Data Studio service layer — staging manifest IO, EDA-on-frame, commit.

Heavy agent imports (``agents.tools`` / ``agents.spec_locks``) are deferred to
inside the functions that need them, so ``import
mmm_framework.data_studio.service`` stays cheap and the module avoids the
LangGraph import cycle. Figures are serialised INLINE (``fig.to_json()``) and
returned in the HTTP response — they are NEVER pushed to
``dashboard_data["plots"]`` (studio previews are ephemeral and must not pollute
the session Plots tab or bloat the checkpointed agent state).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from mmm_framework.eda import (
    EDAConfig,
    OutlierConfig,
    collinearity_analysis,
    decompose_series,
    load_eda_panel_from_df,
    missingness_matrix,
    profile_panel,
    seasonal_period_for_freq,
    spend_share,
    stationarity_tests,
    validate_dataset,
)
from mmm_framework.eda.results import _json_safe

from .transforms import (
    PipelineResult,
    apply_pipeline,
    is_long_frame,
    resolve_date_col,
    user_columns,
)

logger = logging.getLogger("mmm_audit")

_MFF_DIMS = ("Geography", "Product", "Campaign", "Outlet", "Creative")
_PREVIEW_ROWS = 50

#: Analyses the studio EDA endpoint understands (mapped per UI tab).
STUDIO_ANALYSES = (
    "overview",
    "distributions",
    "correlation",
    "missingness",
    "seasonality",
    "spend_share",
    "stationarity",
    "kpi_vs_media",
    "outliers",
)


# ── staging paths + manifest IO ───────────────────────────────────────────────


def _staging_root(tid: str) -> Path:
    from mmm_framework.agents import workspace as _ws

    return _ws.thread_dir(tid) / "data_studio"


def staging_root(tid: str) -> Path:
    return _staging_root(tid)


def raw_dir(tid: str) -> Path:
    d = _staging_root(tid) / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path(tid: str) -> Path:
    return _staging_root(tid) / "manifest.json"


def read_manifest(tid: str) -> dict | None:
    p = _manifest_path(tid)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_manifest(tid: str, manifest: dict) -> None:
    p = _manifest_path(tid)
    p.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = time.time()
    p.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def discard_staging(tid: str) -> None:
    import shutil

    root = _staging_root(tid)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)


# ── frame IO ──────────────────────────────────────────────────────────────────


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(p)
    if suffix in (".tsv",):
        return pd.read_csv(p, sep="\t")
    return pd.read_csv(p)


def load_raw_frame(tid: str, manifest: dict | None = None) -> pd.DataFrame:
    manifest = manifest or read_manifest(tid)
    if not manifest or not manifest.get("raw", {}).get("path"):
        raise FileNotFoundError("No staged dataset for this session.")
    return read_table(manifest["raw"]["path"])


def current_result(tid: str, manifest: dict | None = None) -> PipelineResult:
    """Raw frame + the persisted pipeline, freshly replayed (no cached frame)."""
    manifest = manifest or read_manifest(tid)
    raw = load_raw_frame(tid, manifest)
    return apply_pipeline(
        raw, (manifest or {}).get("steps") or [], (manifest or {}).get("roles") or {}
    )


# ── roles ─────────────────────────────────────────────────────────────────────


def coerce_date_col(df: pd.DataFrame, date_col: str | None) -> pd.DataFrame:
    """Parse ``date_col`` to datetime ROBUSTLY (never raising).

    Real uploads carry ``01/11/2021`` / ``11-01-2021`` / mixed formats; the EDA
    loader's strict ``pd.to_datetime`` raises on those, which would crash the
    panel build (and silently null role inference). Try a few coercing parses
    (ISO, day-first, mixed) and keep the one that parses the most values; leave
    the column untouched if none parse (it isn't really a date).
    """
    if not date_col or date_col not in df.columns:
        return df
    s = df[date_col]
    if pd.api.types.is_datetime64_any_dtype(s):
        return df
    candidates = []
    for kw in (
        {},
        {"dayfirst": True},
        {"format": "mixed"},
        {"format": "mixed", "dayfirst": True},
    ):
        try:
            candidates.append(pd.to_datetime(s, errors="coerce", **kw))
        except Exception:
            continue
    if not candidates:
        return df
    best = max(candidates, key=lambda x: int(x.notna().sum()))
    if int(best.notna().sum()) == 0:
        return df  # not a date column after all — leave it as-is
    out = df.copy()
    out[date_col] = best
    return out


def infer_roles(df: pd.DataFrame) -> dict[str, str]:
    """Heuristic default role map (column -> role) for a freshly staged upload."""
    roles: dict[str, str] = {}
    date_col = resolve_date_col(df)
    try:
        # Coerce the date column first so a non-ISO format doesn't blow up the
        # panel build — otherwise role inference silently yields nothing.
        panel = load_eda_panel_from_df(coerce_date_col(df, date_col))
        if panel.kpi:
            roles[panel.kpi] = "kpi"
        for m in panel.media:
            roles[m] = "media"
        for c in panel.controls:
            roles[c] = "control"
    except Exception:
        pass
    if date_col:
        roles[date_col] = "date"
    if not is_long_frame(df):
        for d in _MFF_DIMS:
            if d in df.columns:
                roles[d] = "group"
    return roles


def _spec_from_roles(roles: dict[str, str]) -> dict | None:
    kpi = next((c for c, r in roles.items() if r == "kpi"), None)
    media = [c for c, r in roles.items() if r == "media"]
    controls = [c for c, r in roles.items() if r == "control"]
    spec: dict[str, Any] = {}
    if kpi:
        spec["kpi"] = kpi
    if media:
        spec["media_channels"] = [{"name": c} for c in media]
    if controls:
        spec["control_variables"] = [{"name": c} for c in controls]
    return spec or None


# ── preview / pipeline ────────────────────────────────────────────────────────


def _records(df: pd.DataFrame, n: int = _PREVIEW_ROWS) -> list[dict]:
    """JSON-safe head rows (pandas to_json handles dates + NaN→null)."""
    return json.loads(df.head(n).to_json(orient="records", date_format="iso"))


def preview_payload(tid: str, manifest: dict, result: PipelineResult) -> dict:
    raw = load_raw_frame(tid, manifest)
    df = result.df
    return {
        "columns": user_columns(df),
        "all_columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "roles": result.roles,
        "date_col": result.date_col,
        "is_long": result.is_long,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "preview_rows": _records(df),
        "diff": {
            "rows_before": int(len(raw)),
            "rows_after": int(len(df)),
            "cols_before": int(raw.shape[1]),
            "cols_after": int(df.shape[1]),
        },
        "warnings": result.warnings,
    }


def set_pipeline(tid: str, steps: list[dict], roles: dict[str, str] | None) -> dict:
    """Persist a (replace-whole) pipeline + roles to the manifest, return preview.

    Raises :class:`~mmm_framework.data_studio.transforms.TransformError` on a
    structurally invalid step (the endpoint maps it to HTTP 400).
    """
    manifest = read_manifest(tid)
    if not manifest:
        raise FileNotFoundError("No staged dataset for this session.")
    raw = load_raw_frame(tid, manifest)
    base_roles = roles if roles is not None else manifest.get("roles") or {}
    result = apply_pipeline(raw, steps or [], base_roles)  # validates ops (may raise)
    manifest["steps"] = steps or []
    manifest["roles"] = result.roles
    write_manifest(tid, manifest)
    return preview_payload(tid, manifest, result)


# ── EDA on the staged frame (inline figures) ──────────────────────────────────


def _fig_json(fig) -> dict:
    blob = json.loads(fig.to_json())
    return {"data": blob.get("data", []), "layout": blob.get("layout", {})}


def _table_from_df(title: str, df: pd.DataFrame) -> dict:
    safe = json.loads(df.to_json(orient="split", date_format="iso"))
    return {
        "title": title,
        "columns": [str(c) for c in safe["columns"]],
        "rows": safe["data"],
    }


def _table_from_records(title: str, records: list[dict]) -> dict:
    cols = list(records[0].keys()) if records else []
    rows = [[_json_safe(r.get(c)) for c in cols] for r in records]
    return {"title": title, "columns": cols, "rows": rows}


def _aggregate_series(panel, var: str) -> pd.Series:
    col = panel.df_wide[var].astype(float)
    if panel.dims:
        col = col.groupby(level=panel.date_col).sum(min_count=1)
    return col


def outlier_suggestions(panel, sensitivity: str = "default", variables=None):
    """Run detection + remediation; map each action to a studio transform step."""
    from mmm_framework.eda import detect_outliers as _detect
    from mmm_framework.eda import recommend_treatments

    cfg = OutlierConfig.for_sensitivity(sensitivity)
    report = _detect(panel, cfg, variables)
    actions = recommend_treatments(panel, report.flags, cfg)
    flags_by_id = {f.flag_id: f for f in report.flags}

    suggestions: list[dict] = []
    for a in actions:
        flags = [flags_by_id[fid] for fid in a.flag_ids if fid in flags_by_id]
        var = flags[0].variable if flags else ""
        periods = sorted({f.period for f in flags})
        step: dict | None = None
        if a.strategy == "winsorize" and flags:
            step = {
                "op": "winsorize",
                "column": var,
                "periods": periods,
                "cap_value": float(a.params.get("cap_value")),
            }
        elif a.strategy == "impute" and flags:
            step = {
                "op": "impute",
                "column": var,
                "periods": periods,
                "value": float(a.params.get("value")),
            }
        elif a.strategy == "dummy":
            step = {
                "op": "event_dummy",
                "name": a.params.get("dummy_name"),
                "periods": list(a.params.get("periods", [])),
            }
        suggestions.append(
            {
                "action_id": a.action_id,
                "strategy": a.strategy,
                "variable": var,
                "rationale": a.rationale,
                "step": step,
                "spec_change": a.spec_change,
            }
        )
    damaged = [
        v for v, s in report.per_variable.items() if s.get("normalization_damaged")
    ]
    return report, suggestions, damaged


def _validation_issues(panel, spec) -> list[dict]:
    try:
        report = validate_dataset(panel, spec=spec)
        order = {"error": 0, "warning": 1, "info": 2}
        issues = sorted(report.issues, key=lambda x: order.get(x.severity, 3))
        return [
            {
                "severity": i.severity,
                "check": i.check,
                "variable": i.variable or "",
                "message": i.message,
            }
            for i in issues
        ]
    except Exception:
        return []


def run_eda_on_frame(
    df: pd.DataFrame,
    roles: dict[str, str],
    analyses: list[str] | None = None,
    variables: list[str] | None = None,
    sensitivity: str = "default",
) -> dict:
    """Run requested analyses on the staged frame; return inline figures/tables.

    Mirrors the engine calls of ``agents.eda_tools.run_eda`` / ``detect_outliers``
    but returns a plain JSON payload (no Command, no plot-store side effects).
    """
    from mmm_framework.eda.charts import (
        fig_correlation_heatmap,
        fig_decomposition,
        fig_distributions,
        fig_kpi_vs_media,
        fig_missingness,
        fig_outlier_series,
        fig_outlier_severity,
        fig_spend_share,
        fig_sparkline_grid,
        fig_stationarity,
        fig_vif,
    )

    spec = _spec_from_roles(roles)
    # Robustly parse the date column first; a non-ISO format would otherwise make
    # the panel build raise and 500 the whole tab.
    df = coerce_date_col(df, resolve_date_col(df, roles))
    try:
        panel = load_eda_panel_from_df(df, spec)
    except Exception as exc:
        date_col = resolve_date_col(df, roles)
        hint = (
            f" The date column `{date_col}` may not be parseable — add a "
            "**Parse date** step (with day-first if your dates are DD/MM/YYYY)."
            if date_col
            else " Assign a **date** role to the period column."
        )
        return {
            "analyses": {},
            "issues": [],
            "outlier_suggestions": [],
            "normalization_damaged": [],
            "warnings": [f"Could not build the dataset for analysis: {exc}.{hint}"],
            "meta": {},
        }
    cfg = EDAConfig()
    requested = [a for a in (analyses or list(STUDIO_ANALYSES)) if a in STUDIO_ANALYSES]
    focus = [v for v in (variables or []) if v in panel.variables] or None

    out: dict[str, Any] = {
        "analyses": {},
        "issues": _validation_issues(panel, spec),
        "outlier_suggestions": [],
        "normalization_damaged": [],
        "warnings": [],
        "meta": {
            "kpi": panel.kpi,
            "media": panel.media,
            "controls": panel.controls,
            "freq": panel.freq,
            "roles_source": panel.roles_source,
            "n_variables": len(panel.variables),
        },
    }

    def _section(name: str, figures=None, tables=None, stats=None) -> None:
        out["analyses"][name] = {
            "figures": figures or [],
            "tables": tables or [],
            "stats": _json_safe(stats or {}),
        }

    def _safe(name: str, fn) -> None:
        try:
            fn()
        except Exception as exc:  # degenerate series / too-few-points etc.
            out["warnings"].append(f"{name}: {exc}")

    if "overview" in requested:

        def _go():
            prof = profile_panel(panel)
            _section(
                "overview",
                figures=[
                    {
                        "key": "sparklines",
                        "title": "Variables over time",
                        **_fig_json(fig_sparkline_grid(panel, focus)),
                    }
                ],
                tables=[_table_from_df("Variable profile", prof)],
                stats={
                    "n_variables": len(panel.variables),
                    "freq": panel.freq,
                    "kpi": panel.kpi,
                    "n_media": len(panel.media),
                    "n_controls": len(panel.controls),
                    "roles_source": panel.roles_source,
                },
            )

        _safe("overview", _go)

    if "distributions" in requested:
        _safe(
            "distributions",
            lambda: _section(
                "distributions",
                figures=[
                    {
                        "key": "distributions",
                        "title": "Value distributions",
                        **_fig_json(fig_distributions(panel, focus)),
                    }
                ],
            ),
        )

    if "correlation" in requested:

        def _go():
            coll = collinearity_analysis(panel, cfg, focus)
            figures, tables = [], []
            if len(coll["correlation"]):
                figures.append(
                    {
                        "key": "correlation",
                        "title": "Correlation matrix",
                        **_fig_json(fig_correlation_heatmap(coll["correlation"])),
                    }
                )
            if coll["vif"]:
                figures.append(
                    {
                        "key": "vif",
                        "title": "Variance inflation factors",
                        **_fig_json(fig_vif(coll["vif"], cfg.vif_threshold)),
                    }
                )
                tables.append(
                    _table_from_records(
                        "Variance inflation factors",
                        [
                            {
                                "channel": ch,
                                "vif": v,
                                "high": ch in (coll["high_vif"] or []),
                            }
                            for ch, v in coll["vif"].items()
                        ],
                    )
                )
            if coll["top_pairs"]:
                tables.append(
                    _table_from_records(
                        "Top correlations",
                        [
                            {"variable_a": p["a"], "variable_b": p["b"], "r": p["r"]}
                            for p in coll["top_pairs"][:50]
                        ],
                    )
                )
            _section(
                "correlation",
                figures=figures,
                tables=tables,
                stats={
                    "condition_number": coll["condition_number"],
                    "high_vif": coll["high_vif"],
                    "clusters": coll["clusters"],
                },
            )

        _safe("correlation", _go)

    if "missingness" in requested:
        _safe(
            "missingness",
            lambda: _section(
                "missingness",
                figures=[
                    {
                        "key": "missingness",
                        "title": "Data availability",
                        **_fig_json(fig_missingness(missingness_matrix(panel))),
                    }
                ],
            ),
        )

    if "spend_share" in requested:

        def _go():
            share = spend_share(panel)
            figures = []
            if share["hhi"] is not None:
                figures.append(
                    {
                        "key": "spend_share",
                        "title": "Spend share over time",
                        **_fig_json(
                            fig_spend_share(
                                share["share_over_time"], share["shares"], share["hhi"]
                            )
                        ),
                    }
                )
            _section(
                "spend_share",
                figures=figures,
                stats={
                    "totals": share["totals"],
                    "shares": share["shares"],
                    "hhi": share["hhi"],
                },
            )

        _safe("spend_share", _go)

    if "seasonality" in requested:

        def _go():
            period = seasonal_period_for_freq(panel.freq)
            figures, info = [], {}
            for var in focus or [v for v in [panel.kpi] if v]:
                res = decompose_series(
                    _aggregate_series(panel, var), period, variable=var
                )
                figures.append(
                    {
                        "key": f"decomp_{var}",
                        "title": f"Decomposition — {var}",
                        **_fig_json(fig_decomposition(res)),
                    }
                )
                info[var] = {
                    "method": res.method,
                    "trend_strength": res.trend_strength,
                    "seasonal_strength": res.seasonal_strength,
                }
            _section("seasonality", figures=figures, stats=info)

        _safe("seasonality", _go)

    if "stationarity" in requested:

        def _go():
            stat_vars = focus or [v for v in [panel.kpi, *panel.media] if v]
            stat = {
                v: stationarity_tests(_aggregate_series(panel, v)) for v in stat_vars
            }
            _section(
                "stationarity",
                figures=[
                    {
                        "key": "stationarity",
                        "title": "Stationarity",
                        **_fig_json(fig_stationarity(stat)),
                    }
                ],
                stats={v: r.get("verdict") for v, r in stat.items()},
            )

        _safe("stationarity", _go)

    if "kpi_vs_media" in requested and panel.kpi and panel.media:
        _safe(
            "kpi_vs_media",
            lambda: _section(
                "kpi_vs_media",
                figures=[
                    {
                        "key": "kpi_vs_media",
                        "title": f"{panel.kpi} vs media",
                        **_fig_json(fig_kpi_vs_media(panel)),
                    }
                ],
            ),
        )

    if "outliers" in requested:

        def _go():
            report, suggestions, damaged = outlier_suggestions(
                panel, sensitivity, focus
            )
            figures, tables = [], []
            for var in sorted({f.variable for f in report.flags}):
                var_flags = [f for f in report.flags if f.variable == var]
                figures.append(
                    {
                        "key": f"outliers_{var}",
                        "title": f"Outliers — {var}",
                        **_fig_json(
                            fig_outlier_series(
                                _aggregate_series(panel, var), var_flags, variable=var
                            )
                        ),
                    }
                )
            if report.flags:
                figures.append(
                    {
                        "key": "outlier_severity",
                        "title": "Outlier severity",
                        **_fig_json(fig_outlier_severity(report.flags)),
                    }
                )
                tables.append(
                    _table_from_records(
                        "Outlier flags",
                        [
                            {
                                "flag_id": f.flag_id,
                                "variable": f.variable,
                                "kind": f.kind,
                                "value": float(f.value),
                                "expected": float(f.expected),
                                "score": float(f.score),
                            }
                            for f in sorted(report.flags, key=lambda x: -x.score)
                        ],
                    )
                )
            out["outlier_suggestions"] = suggestions
            out["normalization_damaged"] = damaged
            _section(
                "outliers",
                figures=figures,
                tables=tables,
                stats={"n_flags": len(report.flags)},
            )

        _safe("outliers", _go)

    return out


# ── commit ────────────────────────────────────────────────────────────────────


def _granularity(freq: str | None) -> str:
    f = (freq or "").upper()
    if f.startswith("D") or f.startswith("B"):
        return "daily"
    if f.startswith("M"):
        return "monthly"
    return "weekly"


def build_commit_artifact(
    tid: str, result: PipelineResult
) -> tuple[str, pd.DataFrame, dict]:
    """Materialise the cleaned frame as MFF-long CSV + return (path, df_long, spec_patch).

    Wide frames are melted on the date-role column over the role-bearing columns;
    MFF-long frames pass through (only the date column is normalised to ``Period``).
    """
    from mmm_framework.agents import workspace as _ws

    df = result.df.copy()
    roles = result.roles
    # Normalise the date column to real datetime (handles DD/MM/YYYY etc.) so the
    # committed CSV is unambiguous and load_mff parses it.
    df = coerce_date_col(df, result.date_col)
    kpi = next((c for c, r in roles.items() if r == "kpi"), None)
    media = [c for c, r in roles.items() if r == "media"]
    controls = [c for c, r in roles.items() if r == "control"]
    group_cols = [c for c, r in roles.items() if r == "group"]

    if not kpi:
        raise ValueError("Assign a KPI role to one column before committing.")
    if not media:
        raise ValueError("Assign at least one media role before committing.")

    if is_long_frame(df):
        date_col = result.date_col or "Period"
        if date_col != "Period" and date_col in df.columns:
            df = df.rename(columns={date_col: "Period"})
        df_long = df
    else:
        date_col = result.date_col
        if not date_col or date_col not in df.columns:
            raise ValueError(
                "Assign a date role to the period column before committing."
            )
        value_cols = [c for c in ([kpi, *media, *controls]) if c in df.columns]
        group_present = [c for c in group_cols if c in df.columns]
        id_vars = [date_col, *group_present]
        df_long = df.melt(
            id_vars=id_vars,
            value_vars=value_cols,
            var_name="VariableName",
            value_name="VariableValue",
        )
        rename = {date_col: "Period"}
        for i, gc in enumerate(group_present):
            if gc not in _MFF_DIMS and i < len(_MFF_DIMS):
                rename[gc] = _MFF_DIMS[i]
        df_long = df_long.rename(columns=rename)

    # The MFF loader requires the canonical dimension columns to exist (national
    # data leaves them None). Add any that are missing so build_model's MFF
    # branch accepts the committed file.
    for dim in _MFF_DIMS:
        if dim not in df_long.columns:
            df_long[dim] = None
    ordered = ["Period", *_MFF_DIMS, "VariableName", "VariableValue"]
    df_long = df_long[[c for c in ordered if c in df_long.columns]]

    dataset_path = str(_ws.thread_dir(tid) / "data_studio_dataset.csv")
    before = _ws.snapshot_dir(_ws.thread_dir(tid))
    df_long.to_csv(dataset_path, index=False)
    try:
        _ws.register_generated_files(tid, before, kind="dataset")
    except Exception:
        pass

    # Only treat as a geo panel when the user actually assigned a group role —
    # the canonical MFF dimension columns are always present (None for national).
    has_geo = bool(group_cols)
    panel = load_eda_panel_from_df(df_long)
    spec_patch: dict[str, Any] = {
        "kpi": kpi,
        "media_channels": [{"name": c} for c in media],
        "control_variables": [{"name": c} for c in controls],
        "time_granularity": _granularity(panel.freq),
    }
    if has_geo:
        spec_patch["kpi_level"] = "geo"
    return dataset_path, df_long, spec_patch


def commit_core(
    state: dict, tid: str, reason: str | None
) -> tuple[str | None, str | None, dict]:
    """Promote the staged, cleaned frame to the session's working dataset.

    Returns ``(error, summary_markdown, state_update)``. ``state_update`` carries
    NO messages — the endpoint applies it via ``aupdate_state`` (same orphan-
    ToolMessage invariant as :func:`agents.eda_tools._apply_outlier_treatment_core`).
    """
    manifest = read_manifest(tid)
    if not manifest:
        return ("No staged dataset for this session — upload one first.", None, {})
    try:
        result = current_result(tid, manifest)
    except Exception as exc:
        return (f"Could not rebuild the staged dataset: {exc}", None, {})

    try:
        dataset_path, df_long, spec_patch = build_commit_artifact(tid, result)
    except ValueError as exc:
        return (str(exc), None, {})

    # Lazy agent imports (kept out of module load to avoid the LangGraph cycle).
    from mmm_framework.agents.eda_tools import _update_eda_envelope
    from mmm_framework.agents.spec_locks import merge_pending, reconcile_with_locks
    from mmm_framework.agents.spec_normalize import _normalized_spec
    from mmm_framework.agents.tools import _build_dataset_dashboard

    _, dataset_info = _build_dataset_dashboard(df_long, dataset_path)

    candidate = _normalized_spec(state.get("model_spec"))
    candidate = {**candidate, **spec_patch}
    current = state.get("model_spec") or {}
    locked = list(state.get("locked_fields") or [])
    merged, new_pending = reconcile_with_locks(
        candidate, current, locked, reason=reason or "data studio commit"
    )
    pending = merge_pending(state.get("pending_spec_changes"), new_pending)

    dashboard_data = dict(state.get("dashboard_data") or {})
    dashboard_data["dataset"] = dataset_info
    dashboard_data["model_spec"] = merged
    dashboard_data["locked_fields"] = locked
    dashboard_data["pending_spec_changes"] = pending
    dashboard_data["data_studio"] = {
        "active": False,
        "committed": True,
        "filename": manifest.get("raw", {}).get("name"),
        "staging_id": manifest.get("staging_id"),
        "n_rows": int(len(df_long)),
        "n_cols": int(df_long.shape[1]),
        "updated_at": time.time(),
    }
    # Surface a post-commit data-quality read in the EDA tab.
    try:
        panel = load_eda_panel_from_df(df_long, _spec_from_roles(result.roles))
        issues = _validation_issues(panel, _spec_from_roles(result.roles))
        if issues:
            _update_eda_envelope(dashboard_data, issues=issues)
    except Exception:
        pass

    manifest["committed"] = True
    write_manifest(tid, manifest)

    summary = [
        "### Dataset committed",
        f"`{Path(dataset_path).name}` — {len(df_long):,} rows, "
        f"KPI **{spec_patch['kpi']}**, {len(spec_patch['media_channels'])} media channel(s), "
        f"{len(spec_patch['control_variables'])} control(s).",
    ]
    if new_pending:
        blocked = ", ".join(f"`{p['path']}`" for p in new_pending)
        summary.append(
            f"\n⚠️ {len(new_pending)} role change(s) touch user-locked fields "
            f"({blocked}) — not applied; confirm them in the Model tab."
        )

    update = {
        "dataset_path": dataset_path,
        "model_spec": merged,
        "pending_spec_changes": pending,
        "dashboard_data": dashboard_data,
    }
    return None, "\n".join(summary), update


def light_summary(
    tid: str, manifest: dict, result: PipelineResult | None = None
) -> dict:
    """The compact ``dashboard_data['data_studio']`` pointer (no frames/figures)."""
    n_rows = int(len(result.df)) if result else 0
    n_cols = int(result.df.shape[1]) if result else 0
    return {
        "active": True,
        "committed": bool(manifest.get("committed")),
        "staging_id": manifest.get("staging_id"),
        "filename": manifest.get("raw", {}).get("name"),
        "kind": manifest.get("raw", {}).get("kind"),
        "n_steps": len(manifest.get("steps") or []),
        "n_rows": n_rows,
        "n_cols": n_cols,
        "updated_at": manifest.get("updated_at") or time.time(),
    }


def init_manifest(
    tid: str, raw_path: str, name: str, kind: str, size_bytes: int
) -> dict:
    """Create (replacing any prior) the staging manifest for a new upload."""
    df = read_table(raw_path)
    roles = infer_roles(df)
    manifest = {
        "staging_id": uuid.uuid4().hex[:12],
        "raw": {
            "path": raw_path,
            "name": name,
            "kind": kind,
            "size_bytes": int(size_bytes),
        },
        "steps": [],
        "roles": roles,
        "committed": False,
        "created_at": time.time(),
    }
    write_manifest(tid, manifest)
    return manifest


__all__ = [
    "STUDIO_ANALYSES",
    "read_manifest",
    "write_manifest",
    "discard_staging",
    "init_manifest",
    "load_raw_frame",
    "current_result",
    "infer_roles",
    "preview_payload",
    "set_pipeline",
    "run_eda_on_frame",
    "build_commit_artifact",
    "commit_core",
    "light_summary",
]
