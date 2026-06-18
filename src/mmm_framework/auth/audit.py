"""Structured auth audit events.

Emits to the ``mmm_audit`` logger (the same sink the kernel/security events use,
see ``agents/audit_sink.py``), tagged with the acting principal's ``user_id`` and
``org_id`` so "who did what" is answerable. Best-effort — auditing must never
break the request, so emission is wrapped.
"""

from __future__ import annotations

import json
import logging
import time
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
    """
    record = {
        "ts": time.time(),
        "event": event,
        "user_id": user_id,
        "org_id": org_id,
        **fields,
    }
    try:
        _audit.info("audit %s", json.dumps(record, default=str, sort_keys=True))
    except Exception:  # pragma: no cover - auditing is best-effort
        pass


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
