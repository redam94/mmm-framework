"""Realized-KPI actuals: parsing and panel reconciliation (issue #227, part 1).

There was no realized-KPI store anywhere — ``delivery`` holds spend only, so a
"variance to plan" could only ever restate a forecast under actual spend. The
store (``sessions.record_actuals`` / ``latest_actuals_for_project``) is
**as-of-dated and append-preserving**: re-stating a period is a new row under a
new ``as_of``, never an overwrite, so a restatement is visible.

Two disciplines this module enforces:

* **Actuals are an independent record, not a derivative of the panel.** When
  the fitted panel and the uploaded actuals disagree for the same period,
  :func:`reconcile_against_panel` REPORTS the signed disagreement; it never
  silently picks one source.
* **Unmatched periods shift nothing.** An actuals period outside the panel's
  vocabulary is reported as unmatched rather than dropped or fuzzily joined.

Lean-core: numpy + stdlib only — importable from the kernel, the server and
tests without fastapi/langchain.
"""

from __future__ import annotations

import csv
import io
from typing import Any

import numpy as np

__all__ = ["parse_actuals_records", "reconcile_against_panel"]


def parse_actuals_records(raw: bytes, filename: str = "") -> list[dict[str, Any]]:
    """Parse an uploaded actuals file into ``{period, kpi_value}`` records.

    CSV/TSV with a period column (``period`` / ``date`` / ``week``) and a value
    column (``kpi_value`` / ``kpi`` / ``value`` / ``actual``), or JSON (a list
    of record dicts, or a ``{period: value}`` mapping). Mirrors the delivery
    parser's format sniffing. Lenient — malformed rows are dropped by
    ``record_actuals`` downstream.
    """
    name = (filename or "").lower()
    text = raw.decode("utf-8", errors="replace")
    stripped = text.lstrip()
    is_json = name.endswith(".json") or (
        not name.endswith((".csv", ".tsv", ".txt")) and stripped[:1] in "[{"
    )
    if is_json:
        import json as _json

        data = _json.loads(text)
        if isinstance(data, dict):
            return [{"period": str(k), "kpi_value": v} for k, v in data.items()]
        return [dict(r) for r in data if isinstance(r, dict)]

    delim = "\t" if (name.endswith(".tsv") or "\t" in text.splitlines()[0]) else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delim)
    period_keys = ("period", "date", "week")
    value_keys = ("kpi_value", "kpi", "value", "actual", "actuals")
    out: list[dict[str, Any]] = []
    for row in reader:
        low = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
        period = next((low[k] for k in period_keys if k in low and low[k]), None)
        value = next((low[k] for k in value_keys if k in low and low[k]), None)
        if period is None or value is None:
            continue
        rec: dict[str, Any] = {"period": str(period).strip(), "kpi_value": value}
        if low.get("kpi_name"):
            rec["kpi_name"] = str(low["kpi_name"])
        if low.get("source"):
            rec["source"] = str(low["source"])
        out.append(rec)
    return out


def reconcile_against_panel(
    model: Any, actuals: list[dict[str, Any]], *, atol: float = 1e-9
) -> dict[str, Any]:
    """Signed per-period gap between uploaded actuals and the panel's own KPI.

    The panel's per-period KPI is the national aggregation of ``y_raw`` over
    the model's period labels (geo×period rows nansum per period). Returns
    ``{periods: [{period, actual, panel, gap}], unmatched: [...],
    max_abs_gap, agrees}`` — zero disagreement reads ``agrees=True``; anything
    else is reported with its sign. **Never silently prefers one source**: the
    caller (a report, the bridge) decides what a disagreement means, with both
    numbers in hand.
    """
    y = np.asarray(getattr(model, "y_raw"), dtype=float)
    # Period labels: the model's period index vocabulary. `time_idx` maps each
    # observation (possibly geo×period) onto a period position.
    labels = [str(p) for p in _period_labels(model)]
    time_idx = np.asarray(getattr(model, "time_idx", np.arange(y.shape[0])), dtype=int)
    panel_by_period: dict[str, float] = {}
    for pos, label in enumerate(labels):
        mask = time_idx == pos
        if mask.any():
            panel_by_period[label] = float(np.nansum(y[mask]))

    rows: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for rec in actuals:
        period = str(rec.get("period") or "")
        try:
            actual = float(rec.get("kpi_value"))
        except (TypeError, ValueError):
            continue
        if period not in panel_by_period:
            unmatched.append(period)
            continue
        panel_val = panel_by_period[period]
        rows.append(
            {
                "period": period,
                "actual": actual,
                "panel": panel_val,
                "gap": actual - panel_val,
            }
        )
    max_gap = max((abs(r["gap"]) for r in rows), default=0.0)
    return {
        "periods": rows,
        "unmatched": unmatched,
        "n_matched": len(rows),
        "max_abs_gap": float(max_gap),
        "agrees": bool(rows) and max_gap <= atol and not unmatched,
    }


def _period_labels(model: Any) -> list[Any]:
    """The model's period vocabulary, best-effort across panel shapes."""
    panel = getattr(model, "panel", None)
    coords = getattr(panel, "coords", None)
    periods = getattr(coords, "periods", None)
    if periods is not None:
        try:
            return [p.date().isoformat() for p in periods]
        except Exception:  # noqa: BLE001 — non-datetime labels stay as-is
            return [str(p) for p in periods]
    index = getattr(model, "index", None) or getattr(panel, "index", None)
    if index is not None:
        try:
            return [p.date().isoformat() for p in index]
        except Exception:  # noqa: BLE001
            return [str(p) for p in index]
    n = int(getattr(model, "n_periods", getattr(model, "n_obs", 0)) or 0)
    return list(range(n))
