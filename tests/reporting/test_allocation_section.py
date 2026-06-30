"""AllocationSection (B5): data-gated budget-allocation report section."""

from __future__ import annotations

from mmm_framework.reporting.config import ReportConfig, SectionConfig
from mmm_framework.reporting.data_extractors import MMMDataBundle
from mmm_framework.reporting.sections import AllocationSection

_PLAN = {
    "total_budget": 1200.0,
    "expected_uplift": 80.0,
    "uplift_hdi": [20.0, 150.0],
    "prob_positive_uplift": 0.92,
    "allocation": [
        {
            "channel": "TV",
            "current_spend": 600.0,
            "optimal_spend": 700.0,
            "change_pct": 16.7,
        },
        {
            "channel": "Search",
            "current_spend": 600.0,
            "optimal_spend": 500.0,
            "change_pct": -16.7,
        },
    ],
    "geo_allocation": [
        {"geo": "North", "channel": "TV", "optimal_spend": 400.0, "change_pct": 10.0},
        {
            "geo": "South",
            "channel": "Search",
            "optimal_spend": 300.0,
            "change_pct": -5.0,
        },
    ],
    "geos": ["North", "South"],
    "flighting": {
        "pattern": "even",
        "channels": ["TV", "Search"],
        "schedule": [
            {"period": "P1", "TV": 350.0, "Search": 250.0, "total": 600.0},
            {"period": "P2", "TV": 350.0, "Search": 250.0, "total": 600.0},
        ],
    },
}


def _section(plan, enabled=True):
    bundle = MMMDataBundle()
    bundle.allocation_results = plan
    return AllocationSection(
        data=bundle,
        config=ReportConfig(),
        section_config=SectionConfig(enabled=enabled),
    )


def test_renders_allocation_geo_and_flighting():
    html = _section(_PLAN).render()
    assert "Budget Allocation Plan" in html
    assert "Recommended allocation" in html
    assert "TV" in html and "Search" in html
    assert "Allocation by geography" in html and "North" in html
    assert "Flighting calendar (even)" in html and "P1" in html
    assert "+17%" in html or "+16%" in html  # change_pct rendered


def test_empty_without_plan():
    bundle = MMMDataBundle()  # no allocation_results
    section = AllocationSection(
        data=bundle, config=ReportConfig(), section_config=SectionConfig(enabled=True)
    )
    assert section.render() == ""


def test_disabled_returns_empty():
    assert _section(_PLAN, enabled=False).render() == ""


def test_national_only_omits_geo_and_flighting():
    plan = {
        k: v
        for k, v in _PLAN.items()
        if k not in ("geo_allocation", "flighting", "geos")
    }
    html = _section(plan).render()
    assert "Recommended allocation" in html
    assert "Allocation by geography" not in html
    assert "Flighting calendar" not in html


def test_escapes_channel_names():
    plan = {
        "total_budget": 100.0,
        "expected_uplift": 1.0,
        "uplift_hdi": [0.0, 2.0],
        "prob_positive_uplift": 0.5,
        "allocation": [
            {
                "channel": "<script>x</script>",
                "current_spend": 50.0,
                "optimal_spend": 60.0,
                "change_pct": 20.0,
            },
        ],
    }
    html = _section(plan).render()
    assert "<script>x</script>" not in html
    assert "&lt;script&gt;" in html


def test_config_default_off():
    assert ReportConfig().allocation.enabled is False
