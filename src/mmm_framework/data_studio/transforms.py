"""Pure, replayable, shape-aware data-cleaning transforms.

A *pipeline* is an ordered list of step dicts ``{"op": ..., **params}``.
:func:`apply_pipeline` replays them over a copy of the raw frame, so the result
is a deterministic function of ``(raw, steps)`` — there is no cached intermediate
to go stale, and re-applying a pipeline is idempotent.

Every op works on BOTH layouts:

* **wide** — one row per period, one column per variable (a typical user CSV);
* **MFF-long** — ``Period, VariableName, VariableValue`` (+ dimension columns).

The layout is detected per frame by :func:`is_long_frame`. For a wide frame the
"columns" are physical column names; for a long frame they are the distinct
``VariableName`` values. Value edits (winsorize/impute/fill/cast) therefore
target physical columns on wide frames and ``VariableValue`` slices on long ones.

Structural validation errors (unknown op, missing required param) raise
:class:`TransformError` (the API maps these to HTTP 400). Runtime/data errors on
an otherwise-valid step (a column the user just renamed away, a cast that can't
coerce) are recorded as ``warnings`` and the step is skipped, so a mid-edit
pipeline still previews instead of aborting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: MFF columns that act as dimensions (besides the date column).
_DIMENSION_COLS = ("Geography", "Product", "Campaign", "Outlet", "Creative")
_DATE_KEYWORDS = ("date", "week", "period", "time")
_STRUCTURAL_LONG = ("VariableName", "VariableValue")


class TransformError(ValueError):
    """A structurally invalid transform step (unknown op / missing param)."""


@dataclass
class PipelineResult:
    df: pd.DataFrame
    roles: dict[str, str]  # user-column -> role (kpi|media|control|date|group|ignore)
    date_col: str | None
    warnings: list[str] = field(default_factory=list)
    is_long: bool = False


# ── layout helpers ────────────────────────────────────────────────────────────


def is_long_frame(df: pd.DataFrame) -> bool:
    """True when ``df`` is MFF-long (has ``VariableName`` + ``VariableValue``)."""
    return all(c in df.columns for c in _STRUCTURAL_LONG)


def user_columns(df: pd.DataFrame) -> list[str]:
    """The columns a user assigns roles to / edits — variable names on a long
    frame, physical columns on a wide frame."""
    if is_long_frame(df):
        return sorted(df["VariableName"].dropna().astype(str).unique().tolist())
    return [str(c) for c in df.columns]


def resolve_date_col(
    df: pd.DataFrame, roles: dict[str, str] | None = None
) -> str | None:
    """The date column: an explicit ``date`` role wins, else the first column
    whose name matches a date keyword, else ``Period`` if present."""
    roles = roles or {}
    for col, role in roles.items():
        if role == "date" and col in df.columns:
            return col
    if is_long_frame(df):
        if "Period" in df.columns:
            return "Period"
    for c in df.columns:
        if any(k in str(c).lower() for k in _DATE_KEYWORDS):
            return str(c)
    return "Period" if "Period" in df.columns else None


def _req(step: dict, key: str):
    if key not in step or step[key] in (None, ""):
        raise TransformError(f"`{step.get('op')}` requires `{key}`")
    return step[key]


def _periods_index(values) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        pd.to_datetime(pd.Series(list(values or [])), errors="coerce")
    )


# ── per-op handlers ───────────────────────────────────────────────────────────
# Each handler: (df, step, roles) -> df. Mutating `roles` in place is allowed.


def _op_rename(df: pd.DataFrame, step: dict, roles: dict) -> pd.DataFrame:
    src, dst = str(_req(step, "from")), str(_req(step, "to"))
    if is_long_frame(df):
        df = df.copy()
        df.loc[df["VariableName"].astype(str) == src, "VariableName"] = dst
    else:
        if src not in df.columns:
            raise RuntimeError(f"no column {src!r}")
        df = df.rename(columns={src: dst})
    if src in roles:
        roles[dst] = roles.pop(src)
    return df


def _op_drop_columns(df: pd.DataFrame, step: dict, roles: dict) -> pd.DataFrame:
    cols = [str(c) for c in (_req(step, "columns") or [])]
    if is_long_frame(df):
        df = df[~df["VariableName"].astype(str).isin(cols)].copy()
    else:
        present = [c for c in cols if c in df.columns]
        df = df.drop(columns=present)
    for c in cols:
        roles.pop(c, None)
    return df


_CAST_FN = {
    "number": lambda s: pd.to_numeric(s, errors="coerce"),
    "float": lambda s: pd.to_numeric(s, errors="coerce").astype("float64"),
    "integer": lambda s: pd.to_numeric(s, errors="coerce").astype("Int64"),
    "int": lambda s: pd.to_numeric(s, errors="coerce").astype("Int64"),
    "string": lambda s: s.astype("string"),
    "category": lambda s: s.astype("category"),
    "boolean": lambda s: s.astype("boolean"),
    "bool": lambda s: s.astype("boolean"),
    "datetime": lambda s: pd.to_datetime(s, errors="coerce"),
}


def _op_cast(df: pd.DataFrame, step: dict, roles: dict) -> pd.DataFrame:
    col, dtype = str(_req(step, "column")), str(_req(step, "dtype")).lower()
    fn = _CAST_FN.get(dtype)
    if fn is None:
        raise TransformError(f"unknown dtype {dtype!r}")
    df = df.copy()
    if is_long_frame(df):
        mask = df["VariableName"].astype(str) == col
        if not mask.any():
            raise RuntimeError(f"no variable {col!r}")
        df.loc[mask, "VariableValue"] = fn(df.loc[mask, "VariableValue"])
    else:
        if col not in df.columns:
            raise RuntimeError(f"no column {col!r}")
        df[col] = fn(df[col])
    return df


def _op_parse_date(df: pd.DataFrame, step: dict, roles: dict) -> pd.DataFrame:
    col = str(step.get("column") or resolve_date_col(df, roles) or "")
    if not col or col not in df.columns:
        raise RuntimeError("no date column to parse")
    fmt = step.get("format") or None
    dayfirst = bool(step.get("dayfirst", False))
    df = df.copy()
    if fmt:
        df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")
    else:
        # No explicit format: infer per-element (handles mixed/non-ISO), honouring
        # day-first for DD/MM/YYYY inputs.
        df[col] = pd.to_datetime(
            df[col], errors="coerce", dayfirst=dayfirst, format="mixed"
        )
    roles[col] = "date"
    return df


def _fill_series(s: pd.Series, strategy: str, value) -> pd.Series:
    if strategy == "zero":
        return s.fillna(0)
    if strategy == "constant":
        return s.fillna(value)
    if strategy == "ffill":
        return s.ffill()
    if strategy == "bfill":
        return s.bfill()
    if strategy in ("mean", "median"):
        num = pd.to_numeric(s, errors="coerce")
        fillv = num.mean() if strategy == "mean" else num.median()
        return num.fillna(fillv)
    if strategy == "interpolate":
        return pd.to_numeric(s, errors="coerce").interpolate(limit_direction="both")
    raise TransformError(f"unknown fill strategy {strategy!r}")


def _op_fill_missing(df: pd.DataFrame, step: dict, roles: dict) -> pd.DataFrame:
    strategy = str(step.get("strategy") or step.get("method") or "zero").lower()
    value = step.get("value")
    df = df.copy()
    if is_long_frame(df):
        cols = [str(c) for c in (step.get("columns") or [])]
        if cols:
            mask = df["VariableName"].astype(str).isin(cols)
        else:
            mask = pd.Series(True, index=df.index)
        # Fill within each variable's own time series.
        for var, idx in df[mask].groupby("VariableName").groups.items():
            df.loc[idx, "VariableValue"] = _fill_series(
                df.loc[idx, "VariableValue"], strategy, value
            ).to_numpy()
    else:
        cols = [c for c in (step.get("columns") or df.columns) if c in df.columns]
        for c in cols:
            df[c] = _fill_series(df[c], strategy, value)
    return df


def _op_drop_duplicates(df: pd.DataFrame, step: dict, roles: dict) -> pd.DataFrame:
    keep = step.get("keep", "first")
    subset = step.get("subset")
    if is_long_frame(df) and not subset:
        date_col = resolve_date_col(df, roles) or "Period"
        dims = [c for c in _DIMENSION_COLS if c in df.columns]
        subset = ["VariableName", date_col, *dims]
    subset = [c for c in (subset or []) if c in df.columns] or None
    return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)


def _compare_mask(s: pd.Series, operator: str, value, value2=None) -> pd.Series:
    op = operator
    if op in ("==", "eq"):
        return s.astype(str) == str(value)
    if op in ("!=", "ne"):
        return s.astype(str) != str(value)
    num = pd.to_numeric(s, errors="coerce")
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if op in ("<", "lt"):
        return num < v
    if op in ("<=", "le"):
        return num <= v
    if op in (">", "gt"):
        return num > v
    if op in (">=", "ge"):
        return num >= v
    if op == "between":
        v2 = pd.to_numeric(pd.Series([value2]), errors="coerce").iloc[0]
        return (num >= v) & (num <= v2)
    if op == "in":
        vals = {str(x) for x in (value or [])}
        return s.astype(str).isin(vals)
    if op == "notin":
        vals = {str(x) for x in (value or [])}
        return ~s.astype(str).isin(vals)
    if op in ("isnull", "isna"):
        return s.isna()
    if op in ("notnull", "notna"):
        return s.notna()
    raise TransformError(f"unknown filter operator {operator!r}")


def _op_filter_rows(df: pd.DataFrame, step: dict, roles: dict) -> pd.DataFrame:
    col = str(_req(step, "column"))
    operator = str(_req(step, "operator"))
    if col not in df.columns:
        raise RuntimeError(f"no column {col!r}")
    mask = _compare_mask(df[col], operator, step.get("value"), step.get("value2"))
    return df[mask.fillna(False)].reset_index(drop=True)


def _op_date_range(df: pd.DataFrame, step: dict, roles: dict) -> pd.DataFrame:
    date_col = str(step.get("column") or resolve_date_col(df, roles) or "")
    if not date_col or date_col not in df.columns:
        raise RuntimeError("no date column for date_range")
    dates = pd.to_datetime(df[date_col], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if step.get("start"):
        mask &= dates >= pd.Timestamp(step["start"])
    if step.get("end"):
        mask &= dates <= pd.Timestamp(step["end"])
    return df[mask.fillna(False)].reset_index(drop=True)


def _value_cell_mask(df: pd.DataFrame, column: str, periods, roles: dict) -> pd.Series:
    """For a LONG frame: rows of `column` (optionally restricted to `periods`)."""
    date_col = resolve_date_col(df, roles) or "Period"
    mask = df["VariableName"].astype(str) == column
    if periods:
        pidx = _periods_index(periods)
        mask &= pd.to_datetime(df[date_col], errors="coerce").isin(pidx)
    return mask


def _op_winsorize(df: pd.DataFrame, step: dict, roles: dict) -> pd.DataFrame:
    col = str(_req(step, "column"))
    cap = float(_req(step, "cap_value"))
    periods = step.get("periods")
    df = df.copy()
    if is_long_frame(df):
        mask = _value_cell_mask(df, col, periods, roles)
        if not mask.any():
            raise RuntimeError(f"no rows for {col!r}")
        vals = pd.to_numeric(df.loc[mask, "VariableValue"], errors="coerce")
        df.loc[mask, "VariableValue"] = np.minimum(vals.to_numpy(), cap)
    else:
        if col not in df.columns:
            raise RuntimeError(f"no column {col!r}")
        series = pd.to_numeric(df[col], errors="coerce")
        if periods:
            date_col = resolve_date_col(df, roles)
            pmask = pd.to_datetime(df[date_col], errors="coerce").isin(
                _periods_index(periods)
            )
            df.loc[pmask, col] = np.minimum(series[pmask].to_numpy(), cap)
        else:
            df[col] = np.minimum(series.to_numpy(), cap)
    return df


def _op_impute(df: pd.DataFrame, step: dict, roles: dict) -> pd.DataFrame:
    col = str(_req(step, "column"))
    value = float(_req(step, "value"))
    periods = step.get("periods")
    df = df.copy()
    if is_long_frame(df):
        mask = _value_cell_mask(df, col, periods, roles)
        if not mask.any():
            raise RuntimeError(f"no rows for {col!r}")
        df.loc[mask, "VariableValue"] = value
    else:
        if col not in df.columns:
            raise RuntimeError(f"no column {col!r}")
        if periods:
            date_col = resolve_date_col(df, roles)
            pmask = pd.to_datetime(df[date_col], errors="coerce").isin(
                _periods_index(periods)
            )
            df.loc[pmask, col] = value
        else:
            df[col] = value
    return df


def _op_event_dummy(df: pd.DataFrame, step: dict, roles: dict) -> pd.DataFrame:
    name = str(_req(step, "name"))
    periods = _req(step, "periods")
    pidx = _periods_index(periods)
    df = df.copy()
    if is_long_frame(df):
        date_col = resolve_date_col(df, roles) or "Period"
        # Build the dummy on the grain of the KPI (or, failing that, the first
        # variable) — one row per existing (period, dims) row of that variable.
        kpi = next((c for c, r in roles.items() if r == "kpi"), None)
        grain = kpi or (df["VariableName"].astype(str).iloc[0] if len(df) else None)
        if grain is None:
            raise RuntimeError("empty frame; cannot add dummy")
        template = df[df["VariableName"].astype(str) == grain].copy()
        template["VariableName"] = name
        template["VariableValue"] = (
            pd.to_datetime(template[date_col], errors="coerce").isin(pidx).astype(float)
        )
        df = pd.concat([df, template], ignore_index=True)
    else:
        date_col = resolve_date_col(df, roles)
        if not date_col or date_col not in df.columns:
            raise RuntimeError("no date column for event_dummy")
        df[name] = (
            pd.to_datetime(df[date_col], errors="coerce").isin(pidx).astype(float)
        )
    roles[name] = "control"
    return df


_OPS = {
    "rename": _op_rename,
    "drop_columns": _op_drop_columns,
    "cast": _op_cast,
    "parse_date": _op_parse_date,
    "fill_missing": _op_fill_missing,
    "drop_duplicates": _op_drop_duplicates,
    "filter_rows": _op_filter_rows,
    "date_range": _op_date_range,
    "winsorize": _op_winsorize,
    "impute": _op_impute,
    "event_dummy": _op_event_dummy,
}

#: Op names the API/UI may send (for validation + the op picker).
OP_NAMES = tuple(_OPS)


# ── pipeline ──────────────────────────────────────────────────────────────────


def _prune_roles(roles: dict[str, str], df: pd.DataFrame) -> dict[str, str]:
    cols = set(user_columns(df))
    # The date column is structural on long frames (Period) — keep its role.
    keep_extra = {
        c for c, r in roles.items() if r in ("date", "group") and c in df.columns
    }
    return {c: r for c, r in roles.items() if c in cols or c in keep_extra}


def apply_pipeline(
    df: pd.DataFrame,
    steps: list[dict] | None,
    roles: dict[str, str] | None = None,
) -> PipelineResult:
    """Replay ``steps`` over a copy of ``df``.

    ``roles`` is the starting column→role map (heuristic defaults from the
    service); it is carried forward (rename/drop reconcile it) and returned
    pruned to the surviving columns. Raises :class:`TransformError` on a
    structurally invalid step; records data-level failures as ``warnings``.
    """
    work = df.copy()
    role_map: dict[str, str] = dict(roles or {})
    warnings: list[str] = []
    for i, step in enumerate(steps or []):
        if not isinstance(step, dict) or "op" not in step:
            raise TransformError(f"step {i + 1} is missing an `op`")
        handler = _OPS.get(str(step["op"]))
        if handler is None:
            raise TransformError(f"unknown transform op {step['op']!r}")
        try:
            work = handler(work, step, role_map)
        except TransformError:
            raise
        except Exception as exc:  # data-level issue — skip, keep previewing
            warnings.append(f"step {i + 1} ({step['op']}) skipped: {exc}")
    return PipelineResult(
        df=work,
        roles=_prune_roles(role_map, work),
        date_col=resolve_date_col(work, role_map),
        warnings=warnings,
        is_long=is_long_frame(work),
    )


__all__ = [
    "TransformError",
    "PipelineResult",
    "apply_pipeline",
    "is_long_frame",
    "user_columns",
    "resolve_date_col",
    "OP_NAMES",
]
