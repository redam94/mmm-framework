"""Server-side triangulation join — MMM × experiment × platform (issue #119).

The reconciliation engine (:mod:`mmm_framework.reporting.triangulation`) already
puts the three independent estimates of a channel's return next to each other and
classifies convergent / divergent / platform-inflated. This module wires it to
**persisted** data so the Oracle / an endpoint can pull the panel WITHOUT
reloading or re-fitting a model:

* MMM sources come from the per-channel ``contribution_roi`` estimand rows already
  persisted on each ``model_run`` artifact (via :func:`build_project_estimands` —
  no model load);
* experiment sources come from the experiment registry
  (:func:`sessions.list_experiments`, calibrated + completed readouts);
* platform figures are optional (an inline dict until the platform-ingestion
  follow-up, #120, adds a persistence slot).

``project_triangulation_sources`` is the pure join (unit-testable without a DB);
``build_project_triangulation`` reads the sessions store.
"""

from __future__ import annotations

from typing import Any

from ..reporting.triangulation import triangulation_from_records

#: Experiment statuses whose readouts are trustworthy enough to triangulate: a
#: calibrated experiment has been folded into a fit, a completed one has a final
#: readout. Draft/planned/running rows have no defensible value yet.
_TRIANGULATABLE_STATUSES = ("calibrated", "completed")


def _mmm_rows_for_kpi(
    estimands: dict[str, Any],
    *,
    kpi: str | None = None,
    run_id: str | None = None,
) -> tuple[list[dict[str, Any]], str | None, str | None]:
    """Pick the ``contribution_roi`` cells for one model from grouped estimands.

    Returns ``(rows, chosen_kpi, chosen_run_id)``. The chosen group is the
    ``contribution_roi`` cluster for ``kpi`` (or, when unset, the one whose latest
    model is newest); within it the chosen model is ``run_id`` (or the latest).
    ``rows`` are ``{channel, mean, lower, upper}`` cells with a usable mean.
    """
    groups = [
        g
        for g in (estimands.get("groups") or [])
        if g.get("estimand") == "contribution_roi"
        and (kpi is None or g.get("kpi") == kpi)
        and g.get("models")
    ]
    if not groups:
        return [], kpi, run_id
    # Newest model wins when no KPI is pinned (models are pre-sorted newest-first).
    groups.sort(key=lambda g: -(float((g["models"][0] or {}).get("created_at") or 0)))
    grp = groups[0]
    models = grp["models"]
    model = None
    if run_id is not None:
        model = next((m for m in models if m.get("run_id") == run_id), None)
    if model is None:
        model = models[0]
    rows = [
        {
            "channel": c.get("channel"),
            "mean": c.get("mean"),
            "lower": c.get("lower"),
            "upper": c.get("upper"),
        }
        for c in model.get("rows") or []
        if c.get("status") in (None, "ok") and c.get("mean") is not None
    ]
    return rows, grp.get("kpi"), model.get("run_id")


def project_triangulation_sources(
    estimands: dict[str, Any],
    experiments: list[dict[str, Any]],
    *,
    kpi: str | None = None,
    run_id: str | None = None,
    platform: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reconcile persisted MMM estimands + registry experiments (+ platform).

    Pure: takes the :func:`build_project_estimands` output + a list of experiment
    registry dicts and returns a serializable payload. No DB / no model.
    """
    mmm_rows, chosen_kpi, chosen_run = _mmm_rows_for_kpi(
        estimands, kpi=kpi, run_id=run_id
    )
    # Only calibrated/completed experiments with a numeric readout can triangulate.
    # Keep ONE readout per channel: since ``experiments`` arrives highest-priority
    # first (calibrated before completed, newest first — see
    # build_project_triangulation), first-seen wins, so a calibrated readout is
    # preferred over a merely-completed one and a stale readout never shadows a
    # fresh calibration.
    exps: list[dict[str, Any]] = []
    seen_channels: set[str] = set()
    for e in experiments:
        ch = e.get("channel")
        if (
            ch is None
            or ch in seen_channels
            or e.get("status") not in _TRIANGULATABLE_STATUSES
            or e.get("value") is None
        ):
            continue
        seen_channels.add(ch)
        exps.append(e)
    result = triangulation_from_records(mmm_rows, experiments=exps, platform=platform)
    payload = result.to_dict()
    payload["kpi"] = chosen_kpi
    payload["run_id"] = chosen_run
    payload["sources_available"] = {
        "mmm": len(mmm_rows),
        "experiment": len(exps),
        "platform": len(platform or {}),
    }
    return payload


def build_project_triangulation(
    project_id: str | None,
    *,
    kpi: str | None = None,
    run_id: str | None = None,
    platform: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Triangulation panel for a project, joined from persisted data only.

    Reads the project's persisted MMM ``contribution_roi`` estimands + the
    experiment registry and reconciles them channel-by-channel. No model is
    loaded, so this is cheap enough to serve from a request.
    """
    from . import sessions as sessions_store
    from .estimands import build_project_estimands

    estimands = build_project_estimands(project_id)
    experiments: list[dict[str, Any]] = []
    for status in _TRIANGULATABLE_STATUSES:
        experiments.extend(
            sessions_store.list_experiments(project_id=project_id, status=status)
        )
    return project_triangulation_sources(
        estimands, experiments, kpi=kpi, run_id=run_id, platform=platform
    )
