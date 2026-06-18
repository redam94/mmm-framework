"""Off-host audit shipper.

The hash-chained audit log (``audit_sink.py``) is tamper-EVIDENT, but only as
durable as the local file — an attacker with host access can still delete it.
This forwards new records to a remote sink so an off-host copy exists. Off by
default; enabled by setting ``MMM_AUDIT_SHIP_URL``. A small cursor file tracks
the last shipped ``seq`` so records ship exactly once (at-least-once on retry: a
sink failure leaves the cursor un-advanced, so the batch re-ships next flush).

Run ``flush_audit_to_remote()`` on a schedule (cron / a background tick); it is a
no-op when no ship URL is configured. ``ship_status()`` reports the backlog for
the observability endpoint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable


def _cursor_path(audit_path: str) -> str:
    return audit_path + ".shipped"


def _read_cursor(cursor_path: str) -> int:
    # -1 = nothing shipped yet (audit seq is 0-based, so the cursor must start
    # below the first record's seq or record 0 would never ship).
    try:
        txt = Path(cursor_path).read_text().strip()
        return int(txt) if txt else -1
    except (OSError, ValueError):
        return -1


def _write_cursor(cursor_path: str, seq: int) -> None:
    Path(cursor_path).write_text(str(int(seq)))


def _read_records(audit_path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        with open(audit_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    continue
    except OSError:
        return []
    return out


def ship_pending(
    audit_path: str,
    sink: Callable[[list[dict[str, Any]]], None],
    *,
    cursor_path: str | None = None,
    batch: int = 200,
) -> dict[str, Any]:
    """Ship every record with ``seq`` past the cursor through ``sink`` (in
    batches), advancing the cursor per successful batch. Returns a summary."""
    cursor_path = cursor_path or _cursor_path(audit_path)
    after = _read_cursor(cursor_path)
    records = [r for r in _read_records(audit_path) if int(r.get("seq", -1)) > after]
    if not records:
        return {"shipped": 0, "cursor": after}
    last = after
    for i in range(0, len(records), batch):
        chunk = records[i : i + batch]
        sink(chunk)  # raises on failure -> cursor not advanced past this batch
        last = int(chunk[-1].get("seq", last))
        _write_cursor(cursor_path, last)
    return {"shipped": len(records), "cursor": last}


def http_sink(
    url: str, token: str | None = None, timeout: float = 10.0
) -> Callable[[list[dict[str, Any]]], None]:
    """A sink that POSTs ``{"records": [...]}`` to ``url`` (stdlib, no deps)."""
    import urllib.request

    def _sink(records: list[dict[str, Any]]) -> None:
        body = json.dumps({"records": records}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(url, data=body, method="POST", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            if resp.status >= 300:
                raise RuntimeError(f"audit ship failed: HTTP {resp.status}")

    return _sink


def ship_url() -> str | None:
    return os.environ.get("MMM_AUDIT_SHIP_URL")


def _resolve_path(audit_path: str | None) -> str | None:
    if audit_path:
        return audit_path
    try:
        from mmm_framework.agents import audit_sink

        return audit_sink.current_log_path()
    except Exception:
        return None


def flush_audit_to_remote(audit_path: str | None = None) -> dict[str, Any]:
    """Ship any pending audit records to ``MMM_AUDIT_SHIP_URL``. No-op if unset."""
    url = ship_url()
    if not url:
        return {"configured": False, "shipped": 0}
    path = _resolve_path(audit_path)
    if not path:
        return {"configured": True, "shipped": 0, "cursor": 0}
    sink = http_sink(url, os.environ.get("MMM_AUDIT_SHIP_TOKEN"))
    return {"configured": True, **ship_pending(path, sink)}


def ship_status(audit_path: str | None = None) -> dict[str, Any]:
    """Backlog snapshot: configured?, cursor, pending count, total records."""
    path = _resolve_path(audit_path)
    if not path:
        return {"configured": bool(ship_url()), "cursor": 0, "pending": 0, "total": 0}
    cursor = _read_cursor(_cursor_path(path))
    records = _read_records(path)
    pending = sum(1 for r in records if int(r.get("seq", -1)) > cursor)
    return {
        "configured": bool(ship_url()),
        "cursor": cursor,
        "pending": pending,
        "total": len(records),
    }
