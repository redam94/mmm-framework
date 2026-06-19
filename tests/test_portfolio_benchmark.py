"""P2 — cross-brand portfolio benchmarking & governance."""

from __future__ import annotations

from mmm_framework.api.portfolio_benchmark import build_portfolio_benchmark

DAY = 86400.0


def _run(ts, channels, portfolio_mroi=None):
    return {
        "created_at": ts,
        "metrics": {
            "channels": {
                n: {"roi_mean": r, "marginal_roi": mr}
                for n, (r, mr) in channels.items()
            },
            "portfolio": {"portfolio_marginal_roi": portfolio_mroi},
        },
    }


def test_cross_brand_benchmark_and_governance():
    projects = [
        {"project_id": "A", "name": "Acme"},
        {"project_id": "B", "name": "Beta"},
    ]
    now = 1_000_000.0
    runs = {
        "A": [_run(now - DAY * 10, {"TV": (2.0, 1.0), "Search": (3.0, 2.0)}, 1.5)],
        "B": [_run(now - DAY * 120, {"TV": (1.0, 0.5), "Social": (0.8, 0.4)}, 0.9)],
    }
    out = build_portfolio_benchmark(
        projects,
        runs,
        now_ts=now,
        calibrated_by_project={"A": 2, "B": 0},
        stale_after_days=90,
    )

    a = next(s for s in out["projects"] if s["project_id"] == "A")
    b = next(s for s in out["projects"] if s["project_id"] == "B")
    assert a["n_channels"] == 2 and a["n_calibrated"] == 2 and a["stale"] is False
    assert a["top_channel"] == {"channel": "Search", "roi_mean": 3.0}
    assert b["stale"] is True  # 120 days > 90

    tv = next(c for c in out["channels"] if c["channel"] == "TV")
    assert tv["n_brands"] == 2
    assert (tv["roi_min"], tv["roi_median"], tv["roi_max"]) == (1.0, 1.5, 2.0)
    search = next(c for c in out["channels"] if c["channel"] == "Search")
    assert search["n_brands"] == 1  # only Acme runs Search

    # rank each brand's TV ROI against the portfolio
    assert a["vs_portfolio"]["TV"]["percentile"] == 100  # 2.0 is top of [1,2]
    assert b["vs_portfolio"]["TV"]["percentile"] == 50  # 1.0 is bottom of [1,2]

    g = out["governance"]
    assert g["n_projects"] == 2 and g["n_with_fit"] == 2
    assert g["n_stale"] == 1 and g["n_fresh"] == 1
    assert g["n_calibrated_projects"] == 1


def test_empty_portfolio():
    out = build_portfolio_benchmark([], {}, now_ts=0.0)
    assert out["projects"] == []
    assert out["channels"] == []
    assert out["governance"]["n_projects"] == 0
    assert out["governance"]["median_model_age_days"] is None


def test_project_without_runs():
    out = build_portfolio_benchmark(
        [{"project_id": "A", "name": "Acme"}], {"A": []}, now_ts=0.0
    )
    s = out["projects"][0]
    assert s["n_runs"] == 0 and s["last_fit_at"] is None and s["stale"] is None
    assert s["vs_portfolio"] == {}
    assert out["governance"]["n_with_fit"] == 0
