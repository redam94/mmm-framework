"""Recommendation scorecard — predicted vs realized (issue #109).

Nothing builds a CMO's (and CFO's) trust in a model faster than watching its past
calls come true — or seeing it honestly own the misses. The raw material is
already persisted: each fitted run's per-channel ROI (the *prediction* at
recommendation time) and the experiment registry's calibrated readouts (the
*realized* incremental return). This joins them, per channel, into an
accountability view: predicted (with its credible interval) vs realized, the
error, and whether the realized value landed inside the predicted interval — so
the model's honesty (interval calibration) is auditable over time.

All from persisted data — no model load. ``project_scorecard_rows`` is the pure
join; ``build_project_scorecard`` reads the sessions store.
"""

from __future__ import annotations

from typing import Any

#: Experiment statuses whose readouts are the realized ground truth.
_REALIZED_STATUSES = ("calibrated", "completed")


def _roi_by_run(estimands: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``{run_id: {channel: cell}}`` for the ``contribution_roi`` estimand, where
    ``cell`` carries ``mean``/``lower``/``upper`` — the model's predicted ROI."""
    out: dict[str, dict[str, Any]] = {}
    created: dict[str, float] = {}
    for g in estimands.get("groups") or []:
        if g.get("estimand") != "contribution_roi":
            continue
        for m in g.get("models") or []:
            rid = m.get("run_id")
            if rid is None:
                continue
            created[rid] = float(m.get("created_at") or 0)
            bucket = out.setdefault(rid, {})
            for cell in m.get("rows") or []:
                ch = cell.get("channel")
                if ch is not None and cell.get("mean") is not None:
                    bucket[ch] = cell
    # stash created_at for latest-run fallback ordering
    out["__created__"] = created  # type: ignore[assignment]
    return out


def project_scorecard_rows(
    estimands: dict[str, Any],
    experiments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Join persisted MMM ``contribution_roi`` predictions to realized experiment
    readouts. Pure — takes :func:`build_project_estimands` output + registry
    experiment dicts. Returns ``{rows, calibration, n_recommendations}``.

    Each row pairs a channel's realized experiment ROAS with the ROI the model
    predicted for it — preferring the experiment's ``recommending_run_id`` (the
    run that recommended the test), else the latest run that estimated the
    channel. ``in_interval`` records whether the realized value landed inside the
    predicted credible interval (the calibration signal).
    """
    roi = _roi_by_run(estimands)
    created: dict[str, float] = roi.pop("__created__", {})  # type: ignore[assignment]

    def _latest_run_with(ch: str) -> str | None:
        cand = [rid for rid, chans in roi.items() if ch in chans]
        return max(cand, key=lambda r: created.get(r, 0.0)) if cand else None

    rows: list[dict[str, Any]] = []
    n_cal = 0
    hits = 0
    for e in experiments:
        if e.get("status") not in _REALIZED_STATUSES or e.get("value") is None:
            continue
        ch = e.get("channel")
        if ch is None:
            continue
        realized = float(e["value"])
        rid = e.get("recommending_run_id")
        pred = (roi.get(rid) or {}).get(ch) if rid else None
        if pred is None:
            rid = _latest_run_with(ch)
            pred = (roi.get(rid) or {}).get(ch) if rid else None

        row: dict[str, Any] = {
            "channel": ch,
            "experiment_id": e.get("id"),
            "estimand": e.get("estimand"),
            "run_id": rid,
            "realized": realized,
            "realized_se": e.get("se"),
            "end_date": e.get("end_date"),
        }
        if pred is None:
            row.update(
                predicted=None,
                predicted_lower=None,
                predicted_upper=None,
                error=None,
                error_pct=None,
                in_interval=None,
            )
        else:
            p_mean = float(pred["mean"])
            lo = pred.get("lower")
            hi = pred.get("upper")
            error = realized - p_mean
            in_ci = (
                bool(lo is not None and hi is not None and lo <= realized <= hi)
                if lo is not None and hi is not None
                else None
            )
            if in_ci is not None:
                n_cal += 1
                hits += int(in_ci)
            row.update(
                predicted=p_mean,
                predicted_lower=lo,
                predicted_upper=hi,
                error=error,
                error_pct=(error / p_mean if abs(p_mean) > 1e-9 else None),
                in_interval=in_ci,
            )
        rows.append(row)

    # newest realized first
    rows.sort(key=lambda r: (r.get("end_date") or ""), reverse=True)
    calibration = {
        "n_with_interval": n_cal,
        "hits": hits,
        "coverage": (hits / n_cal if n_cal else None),
    }
    return {
        "rows": rows,
        "calibration": calibration,
        "n_recommendations": len(rows),
    }


def build_project_scorecard(project_id: str | None) -> dict[str, Any]:
    """The recommendation scorecard for a project, from persisted data only."""
    from . import sessions as sessions_store
    from .estimands import build_project_estimands

    estimands = build_project_estimands(project_id)
    experiments: list[dict[str, Any]] = []
    if project_id:
        for status in _REALIZED_STATUSES:
            experiments.extend(
                sessions_store.list_experiments(project_id=project_id, status=status)
            )
    return project_scorecard_rows(estimands, experiments)
