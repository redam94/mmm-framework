"""Deployment observability — audit-pipeline integrity, off-host ship backlog,
and fit activity. Operator/health info (no tenant data), surfaced at
``GET /observability``.
"""

from __future__ import annotations

import time
from typing import Any

from mmm_framework.api import sessions as store


def system_health(*, fit_window_hours: float = 24.0) -> dict[str, Any]:
    """One snapshot of reliability signals for an operator/health check."""
    audit: dict[str, Any] = {"available": False}
    try:
        from mmm_framework.agents import audit_shipper, audit_sink

        path = audit_sink.current_log_path()
        if path:
            chain_ok, chain_err = audit_sink.verify(path)
            audit = {
                "available": True,
                "chain_ok": chain_ok,  # tamper-evident chain intact?
                "chain_error": chain_err,
                "events": audit_sink.event_counts(),
                "ship": audit_shipper.ship_status(path),  # off-host backlog
            }
        else:
            audit = {"available": False, "reason": "audit sink not installed"}
    except Exception as exc:  # pragma: no cover - never break the health check
        audit = {"available": False, "reason": str(exc)}

    try:
        fits = store.run_metrics_activity(time.time() - fit_window_hours * 3600)
    except Exception:  # pragma: no cover
        fits = {"total": 0, "recent": 0, "last_at": None}

    return {
        "ok": True,
        "audit": audit,
        "fits": {**fits, "window_hours": fit_window_hours},
    }
