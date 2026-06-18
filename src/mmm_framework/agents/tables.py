"""Structured table payloads for the agent dashboard.

Tabular tool output used to reach the frontend as printed-DataFrame text that
the UI re-parsed with regexes. Instead, tools now build a small JSON table
payload (``{title, columns, rows, ...}``), store it content-addressed next to
the plots (``workspace.store_table``), and put only a lightweight
``{id, title, source, group}`` ref into ``dashboard_data["tables"]`` — the SSE
stream re-sends the dashboard on every message, so rows must never ride in it.
The frontend fetches the payload once from ``GET /tables/{id}`` (immutable).

The builders (`df_to_table_json`, `records_to_table_json`) are pure and
import-light because model ops call them **inside** the subprocess/container
kernel; only ``publish_tables`` (host-side) touches the workspace store.
"""

from __future__ import annotations

import logging
from typing import Any

from mmm_framework.agents.kernels import _json_safe

TABLE_ROW_CAP = 200

# Renderer hints understood by the frontend DataTable.
_COLUMN_TYPES = ("string", "number", "percent", "currency", "date")


def _label_for(key: str) -> str:
    return str(key).replace("_", " ").strip().title() or str(key)


def _infer_type(values: list[Any]) -> str:
    """Infer a column renderer hint from sample values (post-_json_safe)."""
    saw_number = False
    for v in values:
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return "string"
        saw_number = True
    return "number" if saw_number else "string"


def _normalize_columns(
    columns: list[dict | str] | None,
    keys: list[str],
    rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build the final ``[{key, label, type}]`` column list.

    ``columns`` may name a subset/order (strings or dicts with optional
    label/type); anything missing is inferred from the rows.
    """
    sample = rows[:50]
    out: list[dict[str, str]] = []
    specs: list[dict] = []
    if columns:
        for c in columns:
            specs.append({"key": c} if isinstance(c, str) else dict(c))
    else:
        specs = [{"key": k} for k in keys]
    for spec in specs:
        key = str(spec.get("key", ""))
        if not key:
            continue
        ctype = spec.get("type")
        if ctype not in _COLUMN_TYPES:
            ctype = _infer_type([r.get(key) for r in sample])
        out.append(
            {
                "key": key,
                "label": str(spec.get("label") or _label_for(key)),
                "type": ctype,
            }
        )
    return out


def records_to_table_json(
    records: list[dict[str, Any]],
    *,
    title: str,
    source: str,
    group: str = "results",
    columns: list[dict | str] | None = None,
    max_rows: int = TABLE_ROW_CAP,
) -> dict:
    """Build a table payload from a list of row dicts (JSON-safe, row-capped)."""
    total = len(records)
    rows = [_json_safe(dict(r)) for r in records[:max_rows]]
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    return {
        "title": str(title),
        "columns": _normalize_columns(columns, keys, rows),
        "rows": rows,
        "total_rows": total,
        "truncated": total > max_rows,
        "source": str(source),
        "group": str(group),
    }


def df_to_table_json(
    df,
    *,
    title: str,
    source: str,
    group: str = "results",
    columns: list[dict | str] | None = None,
    max_rows: int = TABLE_ROW_CAP,
) -> dict:
    """Build a table payload from a pandas DataFrame.

    A meaningful (non-Range) index is reset into ordinary columns so it shows
    up in the rendered table.
    """
    import pandas as pd

    frame = df
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    if not isinstance(frame.index, pd.RangeIndex) or frame.index.name:
        frame = frame.reset_index()
    frame = frame.head(max_rows)
    # Stringify non-scalar cells (Periods, Timestamps, tuples) up front;
    # numbers pass through and get NaN->None via _json_safe.
    records = []
    for rec in frame.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for k, v in rec.items():
            if isinstance(v, (bool, int, float, str)) or v is None:
                clean[str(k)] = v
            elif hasattr(v, "item") and not isinstance(v, (bytes,)):
                clean[str(k)] = v
            else:
                clean[str(k)] = str(v)
        records.append(clean)
    total = int(len(df)) if hasattr(df, "__len__") else len(records)
    table = records_to_table_json(
        records,
        title=title,
        source=source,
        group=group,
        columns=columns,
        max_rows=max_rows,
    )
    table["total_rows"] = total
    table["truncated"] = total > max_rows
    return table


def publish_tables(
    tables: list[dict],
    dashboard_data: dict,
    thread_id: str | None,
) -> tuple[list[dict], int]:
    """Store table payloads content-addressed; append ``{id,title,source,group}``
    refs to ``dashboard_data['tables']``. Host-side only (mirrors
    ``eda_tools._publish_figures``): oversize/invalid payloads are dropped and
    counted, never inlined."""
    from mmm_framework.agents import workspace as _ws

    existing = dashboard_data.get("tables") or []
    refs: list[dict] = []
    dropped = 0
    for table in tables or []:
        try:
            tid = _ws.store_table(table, thread_id)
        except ValueError as exc:
            dropped += 1
            logging.getLogger("mmm_audit").warning(
                "table_rejected thread=%s reason=%s", thread_id, exc
            )
            continue
        except Exception:
            dropped += 1
            continue
        refs.append(
            {
                "id": tid,
                "title": table.get("title") or "Table",
                "source": table.get("source") or "",
                "group": table.get("group") or "results",
            }
        )
    dashboard_data["tables"] = list(existing) + refs
    return refs, dropped


def tables_note(refs: list[dict], dropped: int) -> str:
    """One-line ToolMessage suffix mirroring ``eda_tools._plots_note``."""
    note = ""
    if refs:
        note += f"\n\n*{len(refs)} formatted table(s) rendered in the dashboard.*"
    if dropped:
        note += f"\n\n*{dropped} table(s) omitted (too large or invalid).*"
    return note
