"""Project estimand aggregation for the Performance page.

Turns the estimand rows persisted on each ``model_run`` artifact (see
``agents.fitting`` / ``agents.estimand_rows``) into **comparability clusters**:
one cluster per ``(estimand, KPI)`` so that the same estimand measured on the
same KPI by different models sits side-by-side, while a model on a *different*
KPI lands in its own cluster and is never silently compared. Two ROI estimands
of the same statistical *kind* but different methodology (``contribution_roi``
vs ``counterfactual_roi``) stay distinct clusters — they are different numbers.

The labeling / reference-value / evidence logic mirrors the report's
``EstimandsSection`` exactly (``reporting/sections.py``) so the dashboard and the
generated report tell the same story:

* reference value: ``1.0`` for ratio kinds (ROI / ROAS / x / multiple), else ``0.0``
* evidence vs the credible interval: lower > ref -> "strong"; upper < ref ->
  "below"; otherwise "uncertain"; missing/unsupported -> "na".

``group_estimands`` is pure (no DB) and unit-tested; ``build_project_estimands``
reads the sessions store.
"""

from __future__ import annotations

from typing import Any

# Display labels mirror reporting/sections.py::EstimandsSection._KIND_LABELS.
_ESTIMAND_LABELS: dict[str, str] = {
    "contribution_roi": "Contribution ROI",
    "counterfactual_roi": "Counterfactual ROI",
    "marginal_roas": "Marginal ROAS",
    "contribution": "Incremental contribution",
    "awareness_lift": "Awareness lift",
    "cost_per_conversion": "Cost per conversion",
}

# Canonical sort order for estimands within a KPI section; unknowns sort after,
# alphabetically.
_ESTIMAND_ORDER = [
    "contribution_roi",
    "counterfactual_roi",
    "marginal_roas",
    "contribution",
    "awareness_lift",
    "cost_per_conversion",
]


def estimand_label(name: str) -> str:
    """Human label for an estimand name (title-cased fallback for unknowns)."""
    return _ESTIMAND_LABELS.get(name, name.replace("_", " ").title())


def is_ratio_kind(kind: str | None, units: str | None) -> bool:
    """A ratio estimand (reference 1.0) — mirrors EstimandsSection._is_ratio_kind."""
    k = (kind or "").lower()
    u = (units or "").lower()
    return "roi" in k or "roas" in k or u in {"ratio", "x", "multiple"}


def classify_evidence(
    *,
    status: str | None,
    mean: float | None,
    lower: float | None,
    upper: float | None,
    reference: float,
) -> str:
    """Evidence label vs the no-effect reference: strong / below / uncertain / na."""
    if status not in (None, "ok") or mean is None or lower is None or upper is None:
        return "na"
    if lower > reference:
        return "strong"
    if upper < reference:
        return "below"
    return "uncertain"


def _estimand_sort_key(name: str) -> tuple[int, str]:
    try:
        return (_ESTIMAND_ORDER.index(name), "")
    except ValueError:
        return (len(_ESTIMAND_ORDER), name)


def group_estimands(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Cluster per-run estimand rows into ``(estimand, KPI)`` comparability groups.

    Parameters
    ----------
    runs
        One dict per fitted run with keys ``run_id``, ``label``, ``model_kind``,
        ``model_key``, ``kpi``, ``created_at`` and ``estimands`` (a list of rows
        as produced by :func:`mmm_framework.agents.estimand_rows.evaluate_estimand_rows`).

    Returns
    -------
    dict with ``runs`` (incl. ``is_latest_for_model`` for the default selection),
    ``kpis``, and ``groups`` (each a comparability cluster). Pure; no I/O.
    """
    # Latest run per structural model identity (for the FE default selection).
    # Ties on created_at break deterministically on run_id so the flagged "latest"
    # run is stable regardless of input order.
    latest_for_key: dict[str, tuple[float, str]] = {}
    for r in runs:
        mk = r.get("model_key") or r.get("run_id")
        rid = r.get("run_id") or ""
        ca = float(r.get("created_at") or 0)
        if mk not in latest_for_key or (ca, rid) > latest_for_key[mk]:
            latest_for_key[mk] = (ca, rid)

    run_summaries: list[dict[str, Any]] = []
    # group key -> accumulator
    groups: dict[str, dict[str, Any]] = {}

    for r in runs:
        run_id = r.get("run_id")
        kpi = r.get("kpi") or ""
        model_key = r.get("model_key") or run_id
        est_rows = r.get("estimands") or []
        run_summaries.append(
            {
                "run_id": run_id,
                "label": r.get("label") or run_id,
                "model_kind": r.get("model_kind") or "mmm",
                "model_key": model_key,
                "kpi": kpi,
                "created_at": r.get("created_at"),
                "n_estimands": len(est_rows),
                "is_latest_for_model": latest_for_key.get(model_key, (0.0, ""))[1]
                == run_id,
            }
        )

        for row in est_rows:
            name = row.get("estimand") or ""
            if not name:
                continue
            gkey = f"{name}|||{kpi}"
            grp = groups.get(gkey)
            if grp is None:
                kind = row.get("kind") or ""
                units = row.get("units") or ""
                ratio = is_ratio_kind(kind, units)
                grp = {
                    "key": gkey,
                    "estimand": name,
                    "label": estimand_label(name),
                    "kpi": kpi,
                    "kind": kind,
                    "units": units,
                    "is_ratio": ratio,
                    "reference": 1.0 if ratio else 0.0,
                    "channels": [],
                    "_models": {},  # run_id -> model entry (collapsed below)
                }
                groups[gkey] = grp
            # Fill kind/units if the first row was blank.
            if not grp["kind"] and row.get("kind"):
                grp["kind"] = row.get("kind")
            if not grp["units"] and row.get("units"):
                grp["units"] = row.get("units")

            channel = row.get("channel") or "—"
            if channel not in grp["channels"]:
                grp["channels"].append(channel)

            model = grp["_models"].get(run_id)
            if model is None:
                model = {
                    "run_id": run_id,
                    "label": r.get("label") or run_id,
                    "model_kind": r.get("model_kind") or "mmm",
                    "model_key": model_key,
                    "created_at": r.get("created_at"),
                    "rows": {},  # channel -> cell
                }
                grp["_models"][run_id] = model

            ref = 1.0 if grp["is_ratio"] else 0.0
            mean = row.get("mean")
            lower = row.get("hdi_low")
            upper = row.get("hdi_high")
            model["rows"][channel] = {
                "channel": channel,
                "mean": mean,
                "lower": lower,
                "upper": upper,
                "units": row.get("units") or grp["units"],
                "status": row.get("status") or "ok",
                "evidence": classify_evidence(
                    status=row.get("status"),
                    mean=mean,
                    lower=lower,
                    upper=upper,
                    reference=ref,
                ),
                "prob_positive": row.get("prob_positive"),
                "prob_profitable": row.get("prob_profitable"),
            }

    # Materialize groups: order channels, collapse model dicts -> lists, count
    # models that actually carry a usable number.
    out_groups: list[dict[str, Any]] = []
    for grp in groups.values():
        channels = grp.pop("channels")
        models_map = grp.pop("_models")
        models = []
        n_with_data = 0
        for m in models_map.values():
            row_map = m.pop("rows")
            m["rows"] = [row_map[ch] for ch in channels if ch in row_map]
            if any(
                c.get("status") == "ok" and c.get("mean") is not None for c in m["rows"]
            ):
                n_with_data += 1
            models.append(m)
        models.sort(key=lambda m: -(float(m.get("created_at") or 0)))
        grp["channels"] = channels
        grp["models"] = models
        grp["n_models"] = len(models)
        grp["n_models_with_data"] = n_with_data
        out_groups.append(grp)

    out_groups.sort(key=lambda g: (g["kpi"], _estimand_sort_key(g["estimand"])))

    kpis = sorted({g["kpi"] for g in out_groups})
    run_summaries.sort(key=lambda r: -(float(r.get("created_at") or 0)))
    return {"runs": run_summaries, "kpis": kpis, "groups": out_groups}


def _model_key(model_kind: str, kpi: str, channels: list[str]) -> str:
    """Structural model identity for grouping latest-run-per-model on the FE."""
    return "|".join([model_kind or "mmm", kpi or "", ",".join(sorted(channels or []))])


def build_project_estimands(project_id: str | None) -> dict[str, Any]:
    """Read the project's ``model_run`` artifacts and group their persisted
    estimand rows. Runs fitted before estimand persistence (or with no estimands)
    are skipped; run the backfill (``python -m mmm_framework.api.backfill
    --what estimands``) to populate them."""
    from . import sessions as sessions_store

    runs: list[dict[str, Any]] = []
    for s in sessions_store.list_sessions(project_id=project_id):
        for art in sessions_store.list_artifacts(s["thread_id"]):
            if art.get("kind") != "model_run":
                continue
            p = art.get("payload") or {}
            est = p.get("estimands")
            if not est:
                continue
            kpi = p.get("kpi") or ""
            channels = p.get("channels") or []
            model_kind = p.get("model_kind") or "mmm"
            runs.append(
                {
                    "run_id": p.get("run_id") or art["id"],
                    "label": p.get("run_name") or p.get("run_id") or art["id"],
                    "model_kind": model_kind,
                    "model_key": _model_key(model_kind, kpi, channels),
                    "kpi": kpi,
                    "created_at": art.get("created_at"),
                    "estimands": est,
                }
            )
    # Deterministic input order (newest first, run_id tiebreak) so the grouped
    # output — incl. the channel union, which follows first-seen order — is stable
    # across calls regardless of how the store enumerates sessions/artifacts.
    runs.sort(key=lambda r: (-(float(r.get("created_at") or 0)), r.get("run_id") or ""))
    return group_estimands(runs)
