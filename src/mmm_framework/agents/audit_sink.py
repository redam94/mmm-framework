"""Tamper-evident off-host audit sink (Phase 4d).

The `mmm_audit` logger (kernel spawn/evict/death/timeout/denied-egress, plot
rejections, spawn refusals — Phase 3) is the seam. This adds a `logging.Handler`
that writes each event as a JSON line carrying a **hash chain**: every record's
``hash`` is ``sha256(prev_hash + canonical(record))``, so deleting, editing, or
reordering any record breaks the chain — detectable by ``verify()``. The JSONL is
the off-host hand-off (a shipper tails it to durable storage); the chain makes
on-host tampering before shipping evident.

Also keeps in-memory per-event counters for the ``/metrics`` endpoint (Phase 4d
metrics), so the audit log is the single source for both.

Install at app startup:  ``install_audit_sink()``  (path from ``MMM_AUDIT_LOG``).
Verify a log:            ``ok, err = verify(path)``.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
import threading

_GENESIS = "0" * 64
_RESERVED = ("prev_hash", "hash", "seq", "ts", "level", "event")

# Process-wide event counters (event -> count), read by /metrics.
_COUNTS: "collections.Counter[str]" = collections.Counter()
_COUNTS_LOCK = threading.Lock()


def event_counts() -> dict[str, int]:
    with _COUNTS_LOCK:
        return dict(_COUNTS)


def _canonical(rec: dict) -> str:
    # Stable, separator-fixed encoding so the hash is reproducible across processes.
    return json.dumps(rec, sort_keys=True, separators=(",", ":"), default=str)


def _chain_hash(prev_hash: str, body: dict) -> str:
    return hashlib.sha256((prev_hash + _canonical(body)).encode("utf-8")).hexdigest()


class HashChainAuditHandler(logging.Handler):
    """Append each ``mmm_audit`` record to ``path`` as a hash-chained JSON line."""

    def __init__(self, path: str):
        super().__init__()
        self._path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._prev, self._seq = self._resume()

    def _resume(self) -> "tuple[str, int]":
        """Continue the chain across restarts: pick up the last record's hash/seq
        so an append after a restart still links (the chain spans the file)."""
        prev, seq = _GENESIS, 0
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    prev, seq = rec["hash"], rec["seq"] + 1
        except FileNotFoundError:
            pass
        except Exception:
            # An unreadable/corrupt tail shouldn't crash logging; start a fresh
            # chain segment from genesis (verify() will flag the discontinuity).
            prev, seq = _GENESIS, 0
        return prev, seq

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Structured fields if _audit attached them (extra=), else parse the
            # "event k=v ..." message for the event name.
            event = getattr(record, "audit_event", None)
            fields = getattr(record, "audit_fields", None)
            msg = record.getMessage()
            if event is None:
                event = msg.split(" ", 1)[0] if msg else record.name
            with self._lock:
                body = {
                    "seq": self._seq,
                    "ts": round(record.created, 3),
                    "level": record.levelname,
                    "event": event,
                    "fields": fields if isinstance(fields, dict) else {},
                    "msg": msg,
                    "prev_hash": self._prev,
                }
                body["hash"] = _chain_hash(self._prev, body)
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(body, default=str) + "\n")
                self._prev = body["hash"]
                self._seq += 1
                with _COUNTS_LOCK:
                    _COUNTS[str(event)] += 1
        except Exception:
            self.handleError(record)


def current_log_path() -> str | None:
    """Path of the installed hash-chain audit handler, if any (else None)."""
    for h in logging.getLogger("mmm_audit").handlers:
        if isinstance(h, HashChainAuditHandler):
            return h._path
    return None


def install_audit_sink(path: str | None = None) -> HashChainAuditHandler | None:
    """Attach the hash-chain handler to the ``mmm_audit`` logger. ``path`` defaults
    to ``$MMM_AUDIT_LOG`` or ``<workspace>/audit/audit.jsonl``. Idempotent."""
    if path is None:
        path = os.environ.get("MMM_AUDIT_LOG")
    if path is None:
        try:
            from mmm_framework.agents import workspace as _ws

            path = str(_ws.workspace_root() / "audit" / "audit.jsonl")
        except Exception:
            path = "audit.jsonl"
    logger = logging.getLogger("mmm_audit")
    for h in logger.handlers:
        if isinstance(h, HashChainAuditHandler):
            return h  # already installed
    handler = HashChainAuditHandler(path)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return handler


def verify(path: str) -> "tuple[bool, str | None]":
    """Re-walk the hash chain. Returns ``(True, None)`` if intact, else
    ``(False, reason)`` naming the first record that fails (edited / deleted /
    reordered)."""
    prev, expect_seq = _GENESIS, 0
    try:
        f = open(path, "r", encoding="utf-8")
    except FileNotFoundError:
        return False, f"audit log not found: {path}"
    with f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                return False, f"line {lineno}: not JSON ({e})"
            if rec.get("prev_hash") != prev:
                return False, f"seq {rec.get('seq')}: prev_hash break (reorder/delete)"
            if rec.get("seq") != expect_seq:
                return False, f"line {lineno}: seq gap (expected {expect_seq})"
            body = {k: v for k, v in rec.items() if k != "hash"}
            if _chain_hash(prev, body) != rec.get("hash"):
                return False, f"seq {rec.get('seq')}: hash mismatch (record edited)"
            prev, expect_seq = rec["hash"], expect_seq + 1
    return True, None
