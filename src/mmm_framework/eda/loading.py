"""
Load a dataset into an :class:`EDAPanel` — the shared input for all pre-fit
data-quality analyses.

Unlike :func:`mmm_framework.load_mff` (which drops variables not in the
config and applies fit-time alignment), the EDA loader keeps EVERY variable
so analyses can see both what the model will use and what it would ignore.
Variable roles come from the agent's ``model_spec`` when available, with a
keyword heuristic fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

#: MFF columns that act as dimensions (besides Period).
MFF_DIMENSION_COLS = ["Geography", "Product", "Campaign", "Outlet", "Creative"]

_MEDIA_KEYWORDS = (
    "spend",
    "impression",
    "grp",
    "click",
    "tv",
    "search",
    "social",
    "display",
    "radio",
    "print",
    "video",
    "media",
    "ooh",
    "digital",
)
_KPI_KEYWORDS = ("sales", "revenue", "conversion", "kpi", "units", "orders", "signups")


@dataclass
class EDAPanel:
    """A wide panel plus role metadata, ready for EDA/validation/outliers."""

    df_wide: pd.DataFrame  # index: Period or (Period, *dims); columns: variables
    df_long: pd.DataFrame | None  # original MFF rows (None for wide CSVs)
    kpi: str | None
    media: list[str]
    controls: list[str]
    unassigned: list[str]
    dims: list[str]  # active dimension columns, e.g. ["Geography"]
    date_col: str
    freq: str | None  # "W" / "D" / "MS" / None when undetectable
    roles_source: str  # "spec" | "heuristic"
    duplicate_rows: int = 0  # duplicate (variable, period, dims) rows found
    source_path: str | None = None

    @property
    def variables(self) -> list[str]:
        return list(self.df_wide.columns)

    @property
    def is_panel(self) -> bool:
        return bool(self.dims)

    def series(
        self, variable: str, dim_values: dict[str, str] | None = None
    ) -> pd.Series:
        """One variable as a Period-indexed series (a single panel slice)."""
        if not self.dims:
            return self.df_wide[variable]
        sub = self.df_wide[variable]
        for d, v in (dim_values or {}).items():
            sub = sub.xs(v, level=d)
        return sub

    def slices(self, variable: str):
        """Yield ``(dim_values, series)`` per panel slice (one slice if national)."""
        if not self.dims:
            yield {}, self.df_wide[variable].dropna()
            return
        col = self.df_wide[variable]
        non_period = [lvl for lvl in col.index.names if lvl != self.date_col]
        level = non_period[0] if len(non_period) == 1 else non_period
        for key, grp in col.groupby(level=level):
            key = key if isinstance(key, tuple) else (key,)
            dim_values = dict(zip(non_period, [str(k) for k in key]))
            yield dim_values, grp.droplevel(non_period).dropna()


def _infer_freq(periods: pd.DatetimeIndex) -> str | None:
    if len(periods) < 3:
        return None
    freq = pd.infer_freq(periods.sort_values().unique())
    if freq:
        return freq
    # Fall back to median spacing.
    deltas = pd.Series(periods.sort_values().unique()).diff().dropna()
    if deltas.empty:
        return None
    days = deltas.median().days
    if days == 1:
        return "D"
    if 6 <= days <= 8:
        # Anchor weekly cadence to the observed weekday so date_range
        # comparisons line up (e.g. "W-MON", not Sunday-anchored "W").
        anchor = periods.min().day_name()[:3].upper()
        return f"W-{anchor}"
    if 28 <= days <= 31:
        return "MS"
    return None


def seasonal_period_for_freq(freq: str | None) -> int | None:
    """Observations per year for STL, by frequency."""
    if freq is None:
        return None
    f = freq.upper()
    if f.startswith("W"):
        return 52
    if f.startswith("D") or f.startswith("B"):
        return 7  # weekly cycle dominates daily marketing data
    if f.startswith("M"):
        return 12
    if f.startswith("Q"):
        return 4
    return None


def _roles_from_spec(
    spec: dict, available: list[str]
) -> tuple[str | None, list[str], list[str]]:
    kpi = spec.get("kpi")
    if kpi not in available:
        kpi = None

    def _names(items) -> list[str]:
        out = []
        for it in items or []:
            name = it if isinstance(it, str) else (it or {}).get("name")
            if name and name in available:
                out.append(name)
        return out

    media = _names(spec.get("media_channels"))
    controls = _names(spec.get("control_variables"))
    return kpi, media, controls


def _roles_from_heuristics(
    available: list[str],
) -> tuple[str | None, list[str], list[str]]:
    media = [v for v in available if any(k in v.lower() for k in _MEDIA_KEYWORDS)]
    kpi_candidates = [
        v
        for v in available
        if v not in media and any(k in v.lower() for k in _KPI_KEYWORDS)
    ]
    kpi = kpi_candidates[0] if kpi_candidates else None
    return kpi, media, []


def load_eda_panel(dataset_path: str, spec: dict | None = None) -> EDAPanel:
    """Load a dataset (MFF long CSV, or wide CSV fallback) for EDA.

    Duplicate (VariableName, Period, dims) rows are counted and resolved by
    keeping the FIRST occurrence — the count is surfaced on the panel so the
    validator reports it instead of silently aggregating.
    """
    path = Path(dataset_path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if "VariableName" in df.columns and "VariableValue" in df.columns:
        return _load_long(df, spec, str(path))
    return _load_wide(df, spec, str(path))


def load_eda_panel_from_df(
    df: pd.DataFrame, spec: dict | None = None, source_path: str | None = None
) -> EDAPanel:
    """Build an :class:`EDAPanel` from an in-memory DataFrame (the Data Studio's
    staged/transformed frame) — the path-free sibling of :func:`load_eda_panel`.

    Same MFF-long vs wide dispatch; no disk IO so the studio can re-run EDA on
    every pipeline edit without writing a temp file.
    """
    if "VariableName" in df.columns and "VariableValue" in df.columns:
        return _load_long(df, spec, source_path)
    return _load_wide(df, spec, source_path)


def _load_long(
    df: pd.DataFrame, spec: dict | None, source_path: str | None
) -> EDAPanel:
    date_col = "Period"
    if date_col not in df.columns:
        # Tolerate alternative date column names in otherwise-MFF files.
        candidates = [
            c
            for c in df.columns
            if any(k in c.lower() for k in ("date", "week", "period", "time"))
        ]
        if not candidates:
            raise ValueError("MFF file has no Period/date column")
        date_col = candidates[0]

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work["VariableValue"] = pd.to_numeric(work["VariableValue"], errors="coerce")

    # Active dimensions: present AND more than one distinct value.
    dims = [
        c
        for c in MFF_DIMENSION_COLS
        if c in work.columns and work[c].dropna().nunique() > 1
    ]

    key_cols = ["VariableName", date_col] + dims
    dup_mask = work.duplicated(subset=key_cols, keep="first")
    n_dups = int(dup_mask.sum())
    deduped = work[~dup_mask]

    index_cols = [date_col] + dims
    wide = (
        deduped.set_index(index_cols + ["VariableName"])["VariableValue"]
        .unstack("VariableName")
        .sort_index()
    )
    wide.columns.name = None

    periods = wide.index.get_level_values(date_col) if dims else wide.index
    freq = _infer_freq(pd.DatetimeIndex(periods))

    available = list(wide.columns)
    kpi, media, controls = (None, [], [])
    roles_source = "heuristic"
    if spec and (spec.get("kpi") or spec.get("media_channels")):
        kpi, media, controls = _roles_from_spec(spec, available)
        roles_source = "spec"
    if not media and not kpi:
        kpi, media, controls = _roles_from_heuristics(available)
        roles_source = "heuristic"
    assigned = {kpi, *media, *controls} - {None}
    unassigned = [v for v in available if v not in assigned]

    return EDAPanel(
        df_wide=wide,
        df_long=df,
        kpi=kpi,
        media=media,
        controls=controls,
        unassigned=unassigned,
        dims=dims,
        date_col=date_col,
        freq=freq,
        roles_source=roles_source,
        duplicate_rows=n_dups,
        source_path=source_path,
    )


def _load_wide(
    df: pd.DataFrame, spec: dict | None, source_path: str | None
) -> EDAPanel:
    date_cols = [
        c
        for c in df.columns
        if any(k in c.lower() for k in ("date", "week", "period", "time"))
    ]
    if not date_cols:
        raise ValueError(
            "Could not find a date column (looked for date/week/period/time) "
            "in non-MFF (wide) file"
        )
    date_col = date_cols[0]
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])

    dup_mask = work.duplicated(subset=[date_col], keep="first")
    n_dups = int(dup_mask.sum())
    work = work[~dup_mask]

    numeric = work.select_dtypes(include="number")
    wide = numeric.set_axis(work[date_col], axis=0).sort_index()
    wide.index.name = "Period"

    freq = _infer_freq(pd.DatetimeIndex(wide.index))

    available = list(wide.columns)
    kpi, media, controls = (None, [], [])
    roles_source = "heuristic"
    if spec and (spec.get("kpi") or spec.get("media_channels")):
        kpi, media, controls = _roles_from_spec(spec, available)
        roles_source = "spec"
    if not media and not kpi:
        kpi, media, controls = _roles_from_heuristics(available)
        roles_source = "heuristic"
    assigned = {kpi, *media, *controls} - {None}
    unassigned = [v for v in available if v not in assigned]

    return EDAPanel(
        df_wide=wide,
        df_long=None,
        kpi=kpi,
        media=media,
        controls=controls,
        unassigned=unassigned,
        dims=[],
        date_col="Period",
        freq=freq,
        roles_source=roles_source,
        duplicate_rows=n_dups,
        source_path=source_path,
    )


__all__ = [
    "EDAPanel",
    "load_eda_panel",
    "load_eda_panel_from_df",
    "seasonal_period_for_freq",
    "MFF_DIMENSION_COLS",
]
