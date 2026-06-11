"""
Dataset profiling: per-variable summary statistics, missingness, and
spend share / concentration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from .loading import EDAPanel


def profile_panel(panel: EDAPanel) -> pd.DataFrame:
    """Per-variable summary statistics (one row per variable)."""
    rows = []
    for var in panel.variables:
        col = panel.df_wide[var].astype(float)
        values = col.dropna().to_numpy()
        role = (
            "kpi"
            if var == panel.kpi
            else (
                "media"
                if var in panel.media
                else "control" if var in panel.controls else "unassigned"
            )
        )
        row = {
            "variable": var,
            "role": role,
            "n": int(col.size),
            "missing_pct": float(col.isna().mean() * 100.0),
            "zero_pct": float((col == 0).mean() * 100.0),
        }
        if values.size:
            q1, med, q3 = np.percentile(values, [25, 50, 75])
            row.update(
                mean=float(values.mean()),
                std=float(values.std()),
                min=float(values.min()),
                q1=float(q1),
                median=float(med),
                q3=float(q3),
                max=float(values.max()),
                skew=float(sps.skew(values)) if values.size > 2 else np.nan,
                kurtosis=float(sps.kurtosis(values)) if values.size > 3 else np.nan,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def missingness_matrix(panel: EDAPanel) -> pd.DataFrame:
    """Availability per (period, variable): 1 observed, 0 missing.

    For panel data, a cell counts as observed when ANY slice has a value;
    the fraction observed is returned instead of a binary flag.
    """
    wide = panel.df_wide
    if not panel.dims:
        return wide.notna().astype(float)
    return wide.notna().groupby(level=panel.date_col).mean()


def spend_share(panel: EDAPanel) -> dict[str, object]:
    """Total + over-time spend shares and the HHI concentration index."""
    media = [m for m in panel.media if m in panel.df_wide.columns]
    if not media:
        return {"totals": {}, "shares": {}, "hhi": None, "share_over_time": None}

    spend = panel.df_wide[media].clip(lower=0)
    if panel.dims:
        spend = spend.groupby(level=panel.date_col).sum()
    totals = spend.sum()
    grand = float(totals.sum())
    shares = (totals / grand) if grand > 0 else totals * 0.0
    # Herfindahl–Hirschman index: 1/n (even split) .. 1.0 (single channel).
    hhi = float((shares**2).sum()) if grand > 0 else None

    row_total = spend.sum(axis=1)
    share_over_time = spend.div(row_total.where(row_total > 0), axis=0)

    return {
        "totals": {c: float(totals[c]) for c in media},
        "shares": {c: float(shares[c]) for c in media},
        "hhi": hhi,
        "share_over_time": share_over_time,
    }


__all__ = ["profile_panel", "missingness_matrix", "spend_share"]
