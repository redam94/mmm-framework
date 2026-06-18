"""Cross-brand portfolio benchmarking & governance.

For an agency or holding-co running many brands (projects), aggregate the latest
``run_metrics`` per project into: a per-brand summary, a cross-channel ROI
benchmark (so a brand can be ranked against the portfolio), and governance
signals (model freshness, calibration coverage). The land-and-expand surface.

``build_portfolio_benchmark`` is a pure function over project + run dicts;
``now_ts`` is passed in so it stays deterministic and testable.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _median(xs: list[float]) -> float | None:
    return float(np.median(xs)) if xs else None


def _quantile(xs: list[float], q: float) -> float | None:
    return float(np.quantile(xs, q)) if xs else None


def _percentile_rank(value: float, xs: list[float]) -> int | None:
    """Percentile of ``value`` within ``xs`` (fraction at or below)."""
    if not xs:
        return None
    return round(100 * sum(1 for x in xs if x <= value) / len(xs))


def build_portfolio_benchmark(
    projects: list[dict[str, Any]],
    runs_by_project: dict[str, list[dict[str, Any]]],
    *,
    now_ts: float,
    calibrated_by_project: dict[str, int] | None = None,
    stale_after_days: float = 90.0,
) -> dict[str, Any]:
    """Benchmark a portfolio of projects from their latest run_metrics.

    ``runs_by_project`` maps project_id -> runs (OLDEST-first, as
    ``list_run_metrics`` returns); the latest is used.
    """
    calibrated_by_project = calibrated_by_project or {}
    channel_roi: dict[str, list[float]] = {}
    channel_mroi: dict[str, list[float]] = {}
    per_project: list[dict[str, Any]] = []

    for p in projects:
        pid = p["project_id"]
        runs = runs_by_project.get(pid) or []
        latest = runs[-1] if runs else None
        summary: dict[str, Any] = {
            "project_id": pid,
            "name": p.get("name"),
            "n_runs": len(runs),
            "last_fit_at": None,
            "age_days": None,
            "n_channels": 0,
            "portfolio_marginal_roi": None,
            "top_channel": None,
            "stale": None,
            "n_calibrated": int(calibrated_by_project.get(pid, 0)),
        }
        if latest:
            m = latest.get("metrics") or {}
            chans = m.get("channels") or {}
            ts = latest.get("created_at")
            summary["last_fit_at"] = ts
            summary["n_channels"] = len(chans)
            summary["portfolio_marginal_roi"] = (m.get("portfolio") or {}).get(
                "portfolio_marginal_roi"
            )
            if ts:
                age = (now_ts - ts) / 86400.0
                summary["age_days"] = round(age, 1)
                summary["stale"] = age > stale_after_days
            roi_items = [
                (n, c.get("roi_mean"))
                for n, c in chans.items()
                if c.get("roi_mean") is not None
            ]
            if roi_items:
                top = max(roi_items, key=lambda kv: kv[1])
                summary["top_channel"] = {"channel": top[0], "roi_mean": float(top[1])}
            for name, c in chans.items():
                if c.get("roi_mean") is not None:
                    channel_roi.setdefault(name, []).append(float(c["roi_mean"]))
                if c.get("marginal_roi") is not None:
                    channel_mroi.setdefault(name, []).append(float(c["marginal_roi"]))
        per_project.append(summary)

    # Cross-channel benchmark across brands (latest run each).
    channels = []
    for name in sorted(channel_roi):
        rois = sorted(channel_roi[name])
        channels.append(
            {
                "channel": name,
                "n_brands": len(rois),
                "roi_median": _median(rois),
                "roi_p25": _quantile(rois, 0.25),
                "roi_p75": _quantile(rois, 0.75),
                "roi_min": rois[0],
                "roi_max": rois[-1],
                "marginal_roi_median": _median(sorted(channel_mroi.get(name, []))),
            }
        )

    # Rank each brand's channels against the portfolio distribution.
    for s in per_project:
        runs = runs_by_project.get(s["project_id"]) or []
        latest = runs[-1] if runs else None
        vs: dict[str, Any] = {}
        if latest:
            for name, c in (
                (latest.get("metrics") or {}).get("channels") or {}
            ).items():
                roi = c.get("roi_mean")
                if roi is not None and channel_roi.get(name):
                    vs[name] = {
                        "roi_mean": float(roi),
                        "percentile": _percentile_rank(float(roi), channel_roi[name]),
                    }
        s["vs_portfolio"] = vs

    with_fit = [s for s in per_project if s["last_fit_at"]]
    ages = [s["age_days"] for s in with_fit if s["age_days"] is not None]
    governance = {
        "n_projects": len(projects),
        "n_with_fit": len(with_fit),
        "n_stale": sum(1 for s in with_fit if s.get("stale")),
        "n_fresh": sum(1 for s in with_fit if s.get("stale") is False),
        "n_calibrated_projects": sum(1 for s in per_project if s["n_calibrated"] > 0),
        "median_model_age_days": _median(sorted(ages)),
        "stale_after_days": stale_after_days,
    }

    return {"projects": per_project, "channels": channels, "governance": governance}
