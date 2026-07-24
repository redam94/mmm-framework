"""Server-side in-flight pacing — actual vs plan (issue #123).

Auto-sources the PLANNED series from the project's latest saved budget plan and
compares it against the stored ACTUAL delivery (the delivery registry), computing
per-channel divergence, off-pace flags, and an alert summary — all model-free (no
fit load), so it can be served from a request. The expected-outcome delta (which
needs the fitted response curves) is left to the ``check_pacing`` agent tool /
the report ``pacing=`` path.

``project_pacing`` is the pure join (unit-testable without a DB / a model);
``build_project_pacing`` reads the sessions store.
"""

from __future__ import annotations

from typing import Any

from ..planning.pacing import DEFAULT_PACING_THRESHOLD, compute_pacing


def _records_from_json(data: Any) -> list[dict[str, Any]]:
    """Normalize parsed JSON delivery into ``{channel, period, spend}`` records.

    Accepts a list of records, a ``{channel: {period: spend}}`` map, or a
    ``{channel: total}`` map."""
    out: list[dict[str, Any]] = []
    if isinstance(data, list):
        for d in data:
            if isinstance(d, dict) and d.get("channel") is not None:
                out.append(
                    {
                        "channel": d.get("channel"),
                        "period": str(d.get("period", "") or ""),
                        "spend": d.get("spend"),
                    }
                )
    elif isinstance(data, dict):
        for ch, v in data.items():
            if isinstance(v, dict):
                for period, spend in v.items():
                    out.append({"channel": ch, "period": str(period), "spend": spend})
            else:
                out.append({"channel": ch, "period": "", "spend": v})
    return out


def _records_from_frame(df: Any) -> list[dict[str, Any]]:
    """Delivery records from a pandas frame — LONG (``channel``/``spend`` columns,
    optional ``period``/``date``/``week``) or WIDE (a date/period column + one
    column per channel)."""
    cols = {str(c).lower(): c for c in df.columns}
    period_col = cols.get("period") or cols.get("date") or cols.get("week")
    out: list[dict[str, Any]] = []
    if "channel" in cols and "spend" in cols:
        ch_col, sp_col = cols["channel"], cols["spend"]
        for _, r in df.iterrows():
            out.append(
                {
                    "channel": str(r[ch_col]),
                    "period": str(r[period_col]) if period_col else "",
                    "spend": r[sp_col],
                }
            )
        return out
    # WIDE: every non-period column is a channel; each row is a period.
    channel_cols = [c for c in df.columns if c != period_col]
    for i, r in df.iterrows():
        period = str(r[period_col]) if period_col is not None else str(i)
        for ch in channel_cols:
            out.append({"channel": str(ch), "period": period, "spend": r[ch]})
    return out


def parse_delivery_records(raw: bytes, filename: str = "") -> list[dict[str, Any]]:
    """Parse an uploaded delivery file (CSV/TSV or JSON) into ``{channel, period,
    spend}`` records (issue #123). Format is chosen by extension, falling back to
    sniffing the first non-space char (``[``/``{`` → JSON). Lenient — bad
    spend/channel rows are dropped downstream by ``upsert_delivery``."""
    name = (filename or "").lower()
    text = raw.decode("utf-8", errors="replace")
    stripped = text.lstrip()
    is_json = name.endswith(".json") or (
        not name.endswith((".csv", ".tsv", ".txt")) and stripped[:1] in "[{"
    )
    if is_json:
        import json as _json

        return _records_from_json(_json.loads(text))
    import io as _io

    import pandas as pd

    sep = "\t" if name.endswith(".tsv") else ","
    df = pd.read_csv(_io.StringIO(text), sep=sep)
    return _records_from_frame(df)


def _planned_from_plan(
    plan_payload: Any,
) -> tuple[Any | None, str | None]:
    """Extract a planned series from a saved budget-plan payload.

    Prefers the flighting **schedule** (per-period rows → aligns period-by-period
    with delivery); falls back to the per-channel **allocation** totals. Returns
    ``(planned, basis)`` where ``planned`` is a ``compute_pacing`` input and
    ``basis`` is ``"flighting"`` / ``"allocation"`` / ``None``.
    """
    if not isinstance(plan_payload, dict):
        return None, None
    fl = plan_payload.get("flighting") or {}
    sched = fl.get("schedule")
    if isinstance(sched, list) and sched:
        return {"schedule": sched}, "flighting"
    alloc = plan_payload.get("allocation")
    if isinstance(alloc, list) and alloc:
        planned: dict[str, float] = {}
        for a in alloc:
            if not isinstance(a, dict):
                continue
            ch = a.get("channel")
            if ch is None:
                continue
            val = a.get("optimal_spend")
            if val is None:
                val = a.get("current_spend")
            if val is not None:
                try:
                    planned[str(ch)] = float(val)
                except (TypeError, ValueError):
                    continue
        if planned:
            return planned, "allocation"
    return None, None


def _actual_from_delivery(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pivot delivery rows ``[{channel, period, spend}]`` into per-period rows
    ``[{period, <channel>: spend}]`` ordered by period, for ``compute_pacing``."""
    by_period: dict[str, dict[str, float]] = {}
    for r in rows or []:
        ch = r.get("channel")
        if ch is None:
            continue
        try:
            spend = float(r.get("spend"))
        except (TypeError, ValueError):
            continue
        period = str(r.get("period", "") or "")
        bucket = by_period.setdefault(period, {})
        bucket[str(ch)] = bucket.get(str(ch), 0.0) + spend
    out: list[dict[str, Any]] = []
    for period in sorted(by_period):
        row: dict[str, Any] = {"period": period}
        row.update(by_period[period])
        out.append(row)
    return out


def _alert_summary(payload: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Off-pace alert digest from a pacing payload: the flagged channels + the
    single worst divergence, so a planner (or the T5 re-eval surface) can lead
    with 'N channels off-pace, worst is X'."""
    flagged = list(payload.get("flagged") or [])
    worst: dict[str, Any] | None = None
    for c in payload.get("channels") or []:
        if c.get("channel") not in flagged:
            continue
        d = abs(float(c.get("divergence_pct") or 0.0))
        if worst is None or d > worst["abs_divergence"]:
            worst = {
                "channel": c.get("channel"),
                "divergence_pct": c.get("divergence_pct"),
                "abs_divergence": d,
                "status": c.get("status"),
            }
    return {
        "off_pace": bool(flagged),
        "n_flagged": len(flagged),
        "flagged": flagged,
        "threshold": threshold,
        "worst": worst,
        "portfolio_divergence_pct": payload.get("divergence_pct"),
    }


def project_pacing(
    plan_payload: Any,
    delivery_rows: list[dict[str, Any]],
    *,
    threshold: float = DEFAULT_PACING_THRESHOLD,
) -> dict[str, Any]:
    """Pure: compute the pacing payload + alert digest from a plan + delivery.

    Returns the ``PacingResult.to_dict()`` shape augmented with ``available`` /
    ``reason`` / ``plan_basis`` / ``alert``. ``available`` is ``False`` (with a
    machine-readable ``reason``) when there is no saved plan or no delivery yet,
    so the FE can show the right empty state instead of a misleading zero.
    """
    planned, plan_basis = _planned_from_plan(plan_payload)
    actual = _actual_from_delivery(delivery_rows)
    if planned is None:
        return {
            "available": False,
            "reason": "no_plan",
            "channels": [],
            "flagged": [],
            "threshold": threshold,
        }
    if not actual:
        return {
            "available": False,
            "reason": "no_delivery",
            "plan_basis": plan_basis,
            "channels": [],
            "flagged": [],
            "threshold": threshold,
        }
    res = compute_pacing(planned, actual, threshold=threshold)
    payload = res.to_dict()
    payload["available"] = True
    payload["plan_basis"] = plan_basis
    payload["alert"] = _alert_summary(payload, threshold)
    return payload


def build_project_pacing(
    project_id: str | None,
    *,
    threshold: float = DEFAULT_PACING_THRESHOLD,
) -> dict[str, Any]:
    """Pacing for a project, from persisted data only (no model load).

    Auto-sources the planned series from the latest saved budget plan and joins
    the stored actual delivery. Cheap enough to serve from a request.
    """
    from . import sessions as sessions_store

    plan = (
        sessions_store.latest_budget_plan_for_project(project_id)
        if project_id
        else None
    )
    plan_payload = (plan or {}).get("plan_payload")
    delivery = sessions_store.list_delivery(project_id) if project_id else []
    payload = project_pacing(plan_payload, delivery, threshold=threshold)
    payload["plan_id"] = (plan or {}).get("plan_id")
    payload["plan_name"] = (plan or {}).get("name")
    return payload


#: Synthetic per-project thread holding the persisted pacing-alert artifact.
def _pacing_alert_thread(project_id: str) -> str:
    return f"__pacingalerts__{project_id}"


def latest_pacing_alert(project_id: str) -> dict[str, Any] | None:
    """The most recently persisted off-pace alert for a project (issue #123), or
    ``None``. Written by the background sweep so a planner is alerted without
    opening the pacing panel."""
    from . import sessions as sessions_store

    for art in sessions_store.list_artifacts(_pacing_alert_thread(project_id)):
        if art.get("kind") == "pacing_alert":
            return art.get("payload")
    return None


def sweep_pacing_alerts(now: float | None = None) -> dict[str, int]:
    """Recompute pacing for every project and persist an off-pace alert artifact
    (issue #123) — the proactive cadence, so a planner is flagged between fits
    without opening the panel. Off-pace projects get an upserted ``pacing_alert``
    artifact; projects that recover have theirs cleared. Returns a digest
    ``{scanned, off_pace, persisted, cleared}``."""
    from . import sessions as sessions_store

    scanned = off_pace = persisted = cleared = 0
    for p in sessions_store.list_projects():
        pid = p.get("project_id")
        if not pid:
            continue
        scanned += 1
        try:
            pac = build_project_pacing(pid)
        except Exception:  # noqa: BLE001 — one bad project must not sink the sweep
            continue
        alert = (pac.get("alert") or {}) if pac.get("available") else {}
        tid = _pacing_alert_thread(pid)
        existing = [
            a
            for a in sessions_store.list_artifacts(tid)
            if a.get("kind") == "pacing_alert"
        ]
        if alert.get("off_pace"):
            off_pace += 1
            payload = {
                "project_id": pid,
                "alert": alert,
                "plan_name": pac.get("plan_name"),
                "computed_at": now,
            }
            try:
                if existing:
                    sessions_store.update_artifact_payload(existing[0]["id"], payload)
                else:
                    sessions_store.add_artifact(tid, "pacing_alert", payload)
                persisted += 1
            except Exception:  # noqa: BLE001
                pass
        elif existing:
            # Recovered — clear the stale alert.
            try:
                sessions_store.update_artifact_payload(
                    existing[0]["id"],
                    {"project_id": pid, "alert": None, "computed_at": now},
                )
                cleared += 1
            except Exception:  # noqa: BLE001
                pass
    return {
        "scanned": scanned,
        "off_pace": off_pace,
        "persisted": persisted,
        "cleared": cleared,
    }
