"""Structured auth audit events.

Emits to the ``mmm_audit`` logger (the same sink the kernel/security events use,
see ``agents/audit_sink.py``), tagged with the acting principal's ``user_id`` and
``org_id`` so "who did what" is answerable. Best-effort — auditing must never
break the request, so emission is wrapped.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

_audit = logging.getLogger("mmm_audit")


def audit_event(
    event: str,
    *,
    user_id: str | None = None,
    org_id: str | None = None,
    **fields: Any,
) -> None:
    """Record one auth/security audit event.

    ``event`` is a stable verb like ``auth.login`` / ``auth.logout`` /
    ``auth.token_revoked`` / ``auth.invite_created`` / ``auth.password_reset``.

    Fields are attached via ``extra`` so the hash-chain sink stores them as a
    structured, **queryable** ``fields`` object (not buried in the message) — this
    is what makes the per-org audit export possible.
    """
    payload = {"user_id": user_id, "org_id": org_id, **fields}
    try:
        _audit.info(event, extra={"audit_event": event, "audit_fields": payload})
    except Exception:  # pragma: no cover - auditing is best-effort
        pass


def _audit_log_path() -> str | None:
    """Resolve the active audit-log path (installed handler, else env)."""
    try:
        from mmm_framework.agents import audit_sink

        p = audit_sink.current_log_path()
        if p:
            return p
    except Exception:
        pass
    return os.environ.get("MMM_AUDIT_LOG")


def read_audit_events(
    org_id: str | None = None,
    *,
    since: float | None = None,
    limit: int = 1000,
    path: str | None = None,
) -> list[dict[str, Any]]:
    """Read audit records (newest ``limit``) for ``org_id`` from the hash-chained
    JSONL. Filters on the structured ``fields.org_id``. Returns [] if no log."""
    path = path or _audit_log_path()
    if not path or not os.path.isfile(path):
        return []
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            flds = rec.get("fields") or {}
            if org_id is not None and flds.get("org_id") != org_id:
                continue
            if since is not None and rec.get("ts", 0) < since:
                continue
            out.append(rec)
    return out[-limit:]


def audit_from_principal(
    event: str, principal: Any | None = None, **fields: Any
) -> None:
    """Convenience: pull user_id/org_id off an :class:`AuthContext`-like object."""
    audit_event(
        event,
        user_id=getattr(principal, "user_id", None),
        org_id=getattr(principal, "org_id", None),
        **fields,
    )
