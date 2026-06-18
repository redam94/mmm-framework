"""
Pre-fit collinearity analysis on the raw design: correlations, VIF, and
weakly-identified variable clusters.

Reuses :class:`mmm_framework.validation.channel_diagnostics.VIFCalculator`
and :func:`~mmm_framework.validation.channel_diagnostics.detect_collinear_clusters`
— the same machinery the post-fit diagnostics use — so pre-fit warnings and
post-fit findings agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmm_framework.validation.channel_diagnostics import (
    VIFCalculator,
    detect_collinear_clusters,
)

from .config import EDAConfig
from .loading import EDAPanel


def collinearity_analysis(
    panel: EDAPanel,
    config: EDAConfig | None = None,
    variables: list[str] | None = None,
) -> dict[str, object]:
    """Correlation matrix + VIF + collinear clusters for media (+ controls).

    Returns a dict with keys: ``variables``, ``correlation`` (DataFrame),
    ``vif`` (dict), ``high_vif`` (list), ``clusters`` (list of dicts),
    ``condition_number``, ``top_pairs`` (list of dicts).
    """
    cfg = config or EDAConfig()
    if variables is None:
        variables = [
            v for v in (*panel.media, *panel.controls) if v in panel.df_wide.columns
        ]
    variables = [v for v in variables if panel.df_wide[v].notna().any()]
    if len(variables) < 2:
        return {
            "variables": variables,
            "correlation": pd.DataFrame(),
            "vif": {},
            "high_vif": [],
            "clusters": [],
            "condition_number": None,
            "top_pairs": [],
        }

    frame = panel.df_wide[variables].astype(float)
    if panel.dims:
        frame = frame.groupby(level=panel.date_col).sum(min_count=1)
    frame = frame.dropna()

    corr = frame.corr()
    X = frame.to_numpy()
    vif = VIFCalculator().compute(X, variables)
    high_vif = [v for v, s in vif.items() if np.isfinite(s) and s > cfg.vif_threshold]

    clusters = detect_collinear_clusters(corr, variables, cfg.correlation_threshold)

    try:
        cond = float(np.linalg.cond(corr.to_numpy()))
        condition_number = cond if np.isfinite(cond) else None
    except Exception:
        condition_number = None

    pairs = []
    for i, a in enumerate(variables):
        for j in range(i + 1, len(variables)):
            b = variables[j]
            pairs.append({"a": a, "b": b, "r": float(corr.iloc[i, j])})
    top_pairs = sorted(pairs, key=lambda p: -abs(p["r"]))[: cfg.top_correlations]

    return {
        "variables": variables,
        "correlation": corr,
        "vif": vif,
        "high_vif": high_vif,
        "clusters": [
            {
                "channels": c.channels,
                "max_correlation": c.max_correlation,
                "explanation": c.explanation,
            }
            for c in clusters
        ],
        "condition_number": condition_number,
        "top_pairs": top_pairs,
    }


__all__ = ["collinearity_analysis"]
