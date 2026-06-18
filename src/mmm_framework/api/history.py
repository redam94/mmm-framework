"""Host-side run-metrics persistence and history/coverage/priority assembly.

The kernel computes a model-only metrics snapshot at fit time
(``planning.history.compute_run_metrics``); this module is the host half:

- ``persist_run_metrics``: enrich the snapshot with registry calibration
  status (experiment-backed vs model-only, evidence age) and write the
  ``run_metrics`` row. Registry/DB access stays host-side — kernels never
  touch the sessions store.
- ``build_history_series``: pivot the stored snapshots into per-channel and
  portfolio trajectory series for the Performance page (no model loads).
- ``build_calibration_coverage``: channels × evidence tier, with information
  decay applied at read time.
- ``build_priorities_payload``: the latest EIG/EVOI grid with decay/re-test
  status recomputed against the registry as of today (closed-form over the
  stored roi_sd — no model load).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from mmm_framework.api import sessions as sessions_store
from mmm_framework.planning.eig import (
    channel_half_life,
    reexperiment_due,
)


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s)).date()
    except Exception:
        return None


def _evidence_age_days(ev: dict | None, as_of: date) -> float | None:
    if not ev:
        return None
    d = _parse_date(ev.get("end_date"))
    if d is None:
        ts = ev.get("updated_at")
        if ts is None:
            return None
        d = datetime.fromtimestamp(float(ts), tz=timezone.utc).date()
    return float(max((as_of - d).days, 0))


def enrich_channel_metrics(
    metrics: dict[str, Any], evidence: dict[str, dict], as_of: date | None = None
) -> dict[str, Any]:
    """Stamp per-channel calibration status from the registry evidence map.
    Mutates and returns ``metrics``."""
    as_of = as_of or date.today()
    for name, ch in (metrics.get("channels") or {}).items():
        ev = evidence.get(name)
        ch["calibration_status"] = "experiment_backed" if ev else "model_only"
        ch["evidence_age_days"] = _evidence_age_days(ev, as_of)
        ch["evidence_experiment_id"] = (ev or {}).get("experiment_id")
        ch["evidence_calibrated_run_id"] = (ev or {}).get("calibrated_run_id")
    return metrics


def persist_run_metrics(
    model_run: dict[str, Any],
    thread_id: str | None,
    *,
    artifact_id: str | None = None,
    created_at: float | None = None,
) -> dict[str, Any] | None:
    """Enrich + persist the metrics snapshot riding a ``model_run`` record.
    Returns the enriched metrics, or None when the run carries no metrics.
    Never raises (callers treat this as best-effort)."""
    try:
        metrics = model_run.get("metrics")
        run_id = model_run.get("run_id")
        if not metrics or not run_id:
            return None
        project_id = None
        if thread_id:
            sess = sessions_store.get_session(thread_id)
            project_id = (sess or {}).get("project_id")
        evidence = (
            sessions_store.latest_calibrated_evidence(project_id) if project_id else {}
        )
        enrich_channel_metrics(metrics, evidence)
        sessions_store.record_run_metrics(
            run_id,
            metrics,
            thread_id=thread_id,
            project_id=project_id,
            artifact_id=artifact_id,
            created_at=created_at,
        )
        return metrics
    except Exception:
        import logging

        logging.getLogger(__name__).exception(
            "run-metrics persistence failed (fit result unaffected)"
        )
        return None


# ── Read-time assembly ────────────────────────────────────────────────────────


def build_history_series(project_id: str) -> dict[str, Any]:
    """Trajectory series for the Performance page, from run_metrics rows only
    (oldest first). Channels appearing in any run are included; runs missing a
    channel simply skip that point (series carry run_id per point)."""
    rows = sessions_store.list_run_metrics(project_id)
    runs: list[dict[str, Any]] = []
    channels: list[str] = []
    roi: dict[str, list] = {}
    spend_share: dict[str, list] = {}
    share_gap: dict[str, list] = {}
    calibration: dict[str, list] = {}
    portfolio: list[dict[str, Any]] = []

    for row in rows:
        m = row["metrics"]
        run_id = row["run_id"]
        created_at = row["created_at"]
        runs.append(
            {
                "run_id": run_id,
                "created_at": created_at,
                "timestamp_iso": datetime.fromtimestamp(
                    created_at, tz=timezone.utc
                ).isoformat(),
                "n_draws": m.get("n_draws"),
                "schema_version": row.get("schema_version"),
            }
        )
        for name, ch in (m.get("channels") or {}).items():
            if name not in roi:
                channels.append(name)
                roi[name], spend_share[name] = [], []
                share_gap[name], calibration[name] = [], []
            roi[name].append(
                {
                    "run_id": run_id,
                    "mean": ch.get("roi_mean"),
                    "sd": ch.get("roi_sd"),
                    "hdi_low": ch.get("roi_hdi_low"),
                    "hdi_high": ch.get("roi_hdi_high"),
                    "ci_width": ch.get("ci_width"),
                }
            )
            spend_share[name].append({"run_id": run_id, "value": ch.get("spend_share")})
            share_gap[name].append({"run_id": run_id, "value": ch.get("share_gap")})
            calibration[name].append(
                {
                    "run_id": run_id,
                    "status": ch.get("calibration_status", "model_only"),
                    "evidence_age_days": ch.get("evidence_age_days"),
                }
            )
        p = m.get("portfolio") or {}
        portfolio.append(
            {
                "run_id": run_id,
                "created_at": created_at,
                "marginal_roi": p.get("portfolio_marginal_roi"),
                "expected_uplift": p.get("expected_uplift"),
                "mean_ci_width": p.get("mean_ci_width"),
                "evpi": p.get("evpi"),
                "v_current": p.get("v_current"),
                "prob_positive_uplift": p.get("prob_positive_uplift"),
                "total_spend": p.get("total_spend"),
            }
        )

    return {
        "runs": runs,
        "channels": channels,
        "series": {
            "roi": roi,
            "spend_share": spend_share,
            "share_gap": share_gap,
            "calibration": calibration,
        },
        "portfolio": portfolio,
    }


def _latest_metrics_row(project_id: str) -> dict[str, Any] | None:
    rows = sessions_store.list_run_metrics(project_id)
    return rows[-1] if rows else None


def latest_model_run_payload(project_id: str) -> dict[str, Any] | None:
    """The newest model_run artifact payload in the project — the design
    endpoints resolve the dataset path + KPI from here (the design engine is
    data-only; no model load)."""
    latest: tuple[float, dict] | None = None
    try:
        for s in sessions_store.list_sessions(project_id=project_id):
            for a in sessions_store.list_artifacts(s["thread_id"]):
                if a.get("kind") != "model_run":
                    continue
                ts = a.get("created_at") or 0
                if latest is None or ts > latest[0]:
                    latest = (ts, a.get("payload") or {})
    except Exception:
        return None
    return latest[1] if latest else None


def _latest_model_run_created_at(project_id: str) -> float | None:
    """created_at of the newest model_run artifact in the project (to flag a
    stale metrics row when a newer fit produced no metrics)."""
    latest: float | None = None
    try:
        for s in sessions_store.list_sessions(project_id=project_id):
            for a in sessions_store.list_artifacts(s["thread_id"]):
                if a.get("kind") == "model_run":
                    ts = a.get("created_at")
                    if ts is not None and (latest is None or ts > latest):
                        latest = ts
    except Exception:
        return None
    return latest


def build_calibration_coverage(
    project_id: str, *, as_of: str | None = None
) -> dict[str, Any]:
    """Channels × evidence tier with decay applied at read time.

    Tier: ``running`` (a test is in market or its readout awaits calibration —
    the live edge of the program), ``calibrated`` (experiment-backed, evidence
    fresh), ``stale`` (experiment-backed but decayed past the re-test
    threshold AND older than the freshness floor), or ``model_only``.
    Coverage counts calibrated+stale+running-with-evidence as
    experiment-backed.
    """
    as_of_date = _parse_date(as_of) or date.today()
    latest = _latest_metrics_row(project_id)
    metrics = (latest or {}).get("metrics") or {}
    metric_channels: dict[str, dict] = metrics.get("channels") or {}
    evidence = sessions_store.latest_calibrated_evidence(project_id)
    experiments = sessions_store.list_experiments(project_id=project_id)

    per_channel: dict[str, dict[str, Any]] = {}
    names = list(metric_channels) or sorted({e["channel"] for e in experiments})
    for name in names:
        ch = metric_channels.get(name, {})
        ev = evidence.get(name)
        exps = [e for e in experiments if e["channel"] == name]
        age_days = _evidence_age_days(ev, as_of_date)
        tier = "model_only"
        retest_due = False
        eig_decayed = None
        half_life = channel_half_life(name)
        if ev is not None:
            tier = "calibrated"
            roi_sd = ch.get("roi_sd")
            sigma_exp = ch.get("sigma_exp")
            if age_days is not None and roi_sd and sigma_exp:
                retest_due, eig_decayed = reexperiment_due(
                    float(roi_sd), age_days / 7.0, half_life, float(sigma_exp)
                )
                if retest_due:
                    tier = "stale"
        # the live edge wins the tier: a test in market (or a readout waiting
        # for calibration) is the program ACTING on this channel right now
        in_flight = next(
            (e for e in exps if e["status"] in ("running", "completed")),
            None,
        )
        if in_flight is not None:
            tier = "running"
        per_channel[name] = {
            "channel": name,
            "tier": tier,
            "in_flight_status": (in_flight or {}).get("status"),
            "in_flight_started": (in_flight or {}).get("start_date"),
            "spend_share": ch.get("spend_share"),
            "n_experiments": len(exps),
            "n_calibrated": sum(1 for e in exps if e["status"] == "calibrated"),
            "last_experiment_end": (ev or {}).get("end_date"),
            "evidence_age_days": age_days,
            "half_life_weeks": half_life,
            "eig_decayed": eig_decayed,
            "retest_due": retest_due,
            "experiment_ids": [e["id"] for e in exps],
        }

    rows = list(per_channel.values())
    backed = [
        r
        for r in rows
        if r["tier"] in ("calibrated", "stale")
        or (r["tier"] == "running" and r["last_experiment_end"])
    ]
    shares = [r["spend_share"] for r in rows if r["spend_share"] is not None]
    backed_share = sum(r["spend_share"] for r in backed if r["spend_share"] is not None)
    return {
        "channels": rows,
        "coverage_pct": 100.0 * len(backed) / len(rows) if rows else 0.0,
        "spend_weighted_coverage_pct": (
            100.0 * backed_share / sum(shares) if shares and sum(shares) > 0 else 0.0
        ),
        "as_of": as_of_date.isoformat(),
        "run_id": (latest or {}).get("run_id"),
    }


def build_priorities_payload(
    project_id: str, *, as_of: str | None = None
) -> dict[str, Any] | None:
    """The latest stored EIG/EVOI grid with information decay and registry
    state applied at read time. Returns None when the project has no metrics.

    Decay is closed-form over the stored roi_sd + evidence dates, so this
    never loads a model: stored ``eig`` reflects fit time; ``eig_decayed`` /
    ``retest_due`` reflect TODAY.
    """
    latest = _latest_metrics_row(project_id)
    if latest is None:
        return None
    metrics = latest["metrics"]
    as_of_date = _parse_date(as_of) or date.today()
    evidence = sessions_store.latest_calibrated_evidence(project_id)

    channels: list[dict[str, Any]] = []
    matrix: dict[str, list[str]] = {}
    for name, ch in (metrics.get("channels") or {}).items():
        row = {
            "channel": name,
            "spend": ch.get("spend"),
            "spend_share": ch.get("spend_share"),
            "roi_mean": ch.get("roi_mean"),
            "roi_sd": ch.get("roi_sd"),
            "roi_hdi_low": ch.get("roi_hdi_low"),
            "roi_hdi_high": ch.get("roi_hdi_high"),
            "sigma_exp": ch.get("sigma_exp"),
            "marginal_roi": ch.get("marginal_roi"),
            "eig": ch.get("eig"),
            "evoi": ch.get("evoi"),
            "priority": ch.get("priority"),
            "quadrant": ch.get("quadrant", "deprioritize"),
            "calibration_status": ch.get("calibration_status", "model_only"),
        }
        ev = evidence.get(name)
        age_days = _evidence_age_days(ev, as_of_date)
        row["weeks_since_evidence"] = None if age_days is None else age_days / 7.0
        row["eig_decayed"] = None
        row["retest_due"] = False
        roi_sd, sigma_exp = ch.get("roi_sd"), ch.get("sigma_exp")
        if age_days is not None and roi_sd and sigma_exp:
            hl = channel_half_life(name)
            row["retest_due"], row["eig_decayed"] = reexperiment_due(
                float(roi_sd), age_days / 7.0, hl, float(sigma_exp)
            )
        channels.append(row)
        matrix.setdefault(row["quadrant"], []).append(name)

    channels.sort(key=lambda r: (r.get("priority") or 0.0), reverse=True)
    newest_run_ts = _latest_model_run_created_at(project_id)
    return {
        "run_id": latest["run_id"],
        "computed_at": latest["created_at"],
        "as_of": as_of_date.isoformat(),
        "channels": channels,
        "portfolio": metrics.get("portfolio") or {},
        "response_curves": metrics.get("response_curves"),
        "matrix": matrix,
        "stale": bool(
            newest_run_ts is not None and newest_run_ts > latest["created_at"] + 1.0
        ),
    }
