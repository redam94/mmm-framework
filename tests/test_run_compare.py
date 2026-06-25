"""Run-comparison delta backend (Phase 2 / G4).

`compare_runs` answers "why did this channel's ROI change since the last
refresh?" from persisted run_metrics — the testable backend for the side-by-side
comparison view.
"""

from __future__ import annotations

import pytest

from mmm_framework.api import runs as runs_mod


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    return S


def _metrics(tv_roi, search_roi, total_spend):
    def ch(roi):
        return {
            "roi_mean": roi,
            "marginal_roi": roi * 0.7,
            "spend": 100.0,
            "spend_share": 0.5,
            "roi_hdi_low": roi - 0.3,
            "roi_hdi_high": roi + 0.3,
        }

    return {
        "schema_version": 1,
        "channels": {"TV": ch(tv_roi), "Search": ch(search_roi)},
        "portfolio": {
            "total_spend": total_spend,
            "portfolio_marginal_roi": 1.5,
            "mean_ci_width": 0.4,
        },
    }


def test_compare_runs_per_channel_delta(store):
    store.record_run_metrics("runA", _metrics(2.1, 1.0, 200.0), project_id="p1")
    store.record_run_metrics("runB", _metrics(1.6, 1.2, 220.0), project_id="p1")

    cmp = runs_mod.compare_runs("runA", "runB")
    tv = next(c for c in cmp["channels"] if c["channel"] == "TV")
    assert tv["roi_mean"]["a"] == 2.1
    assert tv["roi_mean"]["b"] == 1.6
    assert tv["roi_mean"]["delta"] == pytest.approx(-0.5)
    search = next(c for c in cmp["channels"] if c["channel"] == "Search")
    assert search["roi_mean"]["delta"] == pytest.approx(0.2)
    assert cmp["portfolio"]["total_spend"]["delta"] == pytest.approx(20.0)
    assert cmp["run_a"]["project_id"] == "p1"


def test_compare_runs_added_and_removed_channels(store):
    store.record_run_metrics("runA", _metrics(2.0, 1.0, 200.0), project_id="p1")
    m = _metrics(2.0, 1.0, 200.0)
    m["channels"].pop("Search")  # Search removed in B
    m["channels"]["Radio"] = {  # Radio added in B
        "roi_mean": 1.1,
        "marginal_roi": 0.8,
        "spend": 50.0,
        "spend_share": 0.25,
        "roi_hdi_low": 0.9,
        "roi_hdi_high": 1.3,
    }
    store.record_run_metrics("runB", m, project_id="p1")

    cmp = runs_mod.compare_runs("runA", "runB")
    radio = next(c for c in cmp["channels"] if c["channel"] == "Radio")
    assert radio["in_a"] is False and radio["in_b"] is True
    assert radio["roi_mean"]["a"] is None
    assert radio["roi_mean"]["delta"] is None  # no delta when one side is missing
    search = next(c for c in cmp["channels"] if c["channel"] == "Search")
    assert search["in_a"] is True and search["in_b"] is False


def test_compare_runs_missing_metrics_raises(store):
    store.record_run_metrics("runA", _metrics(2.0, 1.0, 200.0), project_id="p1")
    with pytest.raises(ValueError, match="No run metrics"):
        runs_mod.compare_runs("runA", "does-not-exist")
