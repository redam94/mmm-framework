"""Opportunity-cost / short-term-risk tests (planning/opportunity_cost.py).

These pin the sign conventions and the geo/window row mapping that the
adversarial design review flagged as the highest-risk corrections — using a
duck-typed FakeMMM so they run fast (no MCMC fit). The model only needs a
handful of attributes + a deterministic ``sample_channel_contributions`` that
responds to the perturbed media, which is enough to exercise every path.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.planning.opportunity_cost import (
    _resolve_margin,
    _resolve_treated_rows,
    build_experiment_media,
    compute_opportunity_cost,
)


class FakeMMM:
    """Additive, saturating (sqrt) response, deterministic per draw — so two
    ``sample_channel_contributions`` passes at the same max_draws are paired
    exactly (mirrors the real Deterministic node, F1)."""

    def __init__(
        self, geo_names, n_periods, channel_names, *, seed=0, ragged_drop=None
    ):
        rng = np.random.default_rng(seed)
        self.geo_names = list(geo_names)
        self.n_geos = len(geo_names)
        self.has_geo = self.n_geos > 1
        self.n_periods = n_periods
        self.channel_names = list(channel_names)
        rows = [(p, g) for p in range(n_periods) for g in range(self.n_geos)]
        if ragged_drop:
            drop = set(ragged_drop)
            rows = [r for r in rows if r not in drop]
        self.time_idx = np.array([p for p, _ in rows], dtype=np.int64)
        self.geo_idx = np.array([g for _, g in rows], dtype=np.int64)
        self.n_obs = len(rows)
        c = len(channel_names)
        self._betas = rng.uniform(0.8, 2.0, c)
        self.X_media_raw = rng.uniform(50.0, 200.0, (self.n_obs, c))
        base = (self._betas * np.sqrt(self.X_media_raw)).sum(1)
        self.y_raw = 1000.0 + base + rng.normal(0, 3, self.n_obs)
        self.y_std = float(self.y_raw.std())

    def sample_channel_contributions(
        self, X_media=None, max_draws=200, random_seed=None
    ):
        x = self.X_media_raw if X_media is None else np.asarray(X_media, float)
        d = max(2, min(int(max_draws), 40))
        f = 1.0 + 0.05 * np.linspace(-1.0, 1.0, d)
        sqrtx = np.sqrt(np.clip(x, 0.0, None))
        return self._betas[None, None, :] * sqrtx[None, :, :] * f[:, None, None]


def _design(geo_names, *, design="holdout", channel="TV", duration=8, key="geo_lift"):
    treat = list(geo_names[0::2])
    ctrl = list(geo_names[1::2])
    intensity = -100.0 if design == "holdout" else 50.0
    return {
        "design_key": key,
        "channel": channel,
        "kpi": "Sales",
        "design": design,
        "design_type": f"randomized geo lift — {design}",
        "intensity_pct": intensity,
        "duration": duration,
        "treatment_geos": treat,
        "control_geos": ctrl,
        "weekly_spend_delta": 9999.0,  # abs magnitude — must NOT be used by OC
        "se_roas": 0.1,
        "mde_roas": 0.28,
        "power_curve": [
            {"duration": d, "mde_roas": 0.28 * (8.0 / d) ** 0.5} for d in (4, 8, 12, 16)
        ],
    }


GEOS = [f"G{i}" for i in range(8)]
CHANNELS = ["TV", "Search", "Social"]


def test_geo_row_mapping_selects_treated_cells():
    mmm = FakeMMM(GEOS, 40, CHANNELS, seed=1)
    design = _design(GEOS, design="scaling", duration=8)
    mask, codes, window_codes, dur_eff, warns = _resolve_treated_rows(
        mmm, design, duration=8
    )
    treated_codes = [mmm.geo_names.index(g) for g in design["treatment_geos"]]
    assert sorted(codes) == sorted(treated_codes)
    assert dur_eff == 8
    # every selected row is a treated geo in the last 8 periods, nothing else
    assert np.all(np.isin(mmm.geo_idx[mask], treated_codes))
    assert set(mmm.time_idx[mask].tolist()) == set(range(32, 40))
    assert mask.sum() == len(treated_codes) * 8


def test_geo_name_mismatch_raises_not_zero_risk():
    mmm = FakeMMM(GEOS, 40, CHANNELS, seed=1)
    design = _design(GEOS)
    design["treatment_geos"] = ["NOT_A_GEO", "ALSO_MISSING"]
    with pytest.raises(ValueError, match="not found"):
        _resolve_treated_rows(mmm, design, duration=8)


def test_geo_name_case_insensitive_fallback():
    mmm = FakeMMM(["North", "South", "East", "West"], 40, CHANNELS, seed=1)
    design = _design(["North", "South", "East", "West"])
    design["treatment_geos"] = ["north", "EAST"]  # different casing
    mask, codes, *_ = _resolve_treated_rows(mmm, design, duration=8)
    assert sorted(codes) == [0, 2]  # North, East


def test_holdout_sign_conventions():
    """Holdout: spend_delta < 0 (saved), kpi_delta <= 0 (lost), and when margin
    is below ROAS the net profit impact is POSITIVE (you save more than you
    forgo) → opportunity_cost_dollar ≈ 0. This is the OC-3/OC-4 correction."""
    mmm = FakeMMM(GEOS, 60, CHANNELS, seed=2)
    design = _design(GEOS, design="holdout")
    oc = compute_opportunity_cost(
        mmm, design, margin_per_kpi=0.1, kpi_kind="revenue", max_draws=40
    )
    assert oc.spend_delta < 0  # going dark saves spend
    assert oc.kpi_delta_median <= 0  # lost KPI
    assert oc.forgone_kpi_median >= 0
    assert oc.prob_kpi_loss > 0.5
    # net = margin*kpi_delta - spend_delta ; with small margin the saved spend wins
    assert oc.net_profit_impact_median > 0
    assert oc.opportunity_cost_dollar_median == pytest.approx(0.0, abs=1e-6)


def test_scaling_sign_conventions():
    mmm = FakeMMM(GEOS, 60, CHANNELS, seed=2)
    design = _design(GEOS, design="scaling")
    oc = compute_opportunity_cost(mmm, design, max_draws=40)
    assert oc.spend_delta > 0  # extra dollars committed
    assert oc.kpi_delta_median >= 0  # gained KPI
    assert oc.spend_at_risk == pytest.approx(oc.spend_delta)
    assert oc.net_profit_impact_median is None  # no margin supplied
    assert oc.margin_source == "none"


def test_spend_delta_is_internal_not_design_weekly():
    """spend_delta must come from the perturbed vs BAU matrices, never the
    design's abs() weekly_spend_delta (OC-4)."""
    mmm = FakeMMM(GEOS, 50, CHANNELS, seed=3)
    design = _design(GEOS, design="holdout")
    design["weekly_spend_delta"] = 9999.0  # positive magnitude trap
    oc = compute_opportunity_cost(mmm, design, max_draws=30)
    assert oc.spend_delta < 0
    assert abs(oc.spend_delta) != pytest.approx(9999.0 * 8)  # not the design value


def test_ragged_window_excludes_absent_cells_and_shrinks():
    """A treated geo missing its most recent weeks must never have those absent
    cells selected; when the common coverage falls below the requested window,
    duration_effective shrinks and a warning is recorded (OC-1)."""
    # short panel: treated geo 0 loses its last 4 of 10 periods, so the common
    # coverage across treated geos (0,2,4,6) is only 6 weeks → window can't be 8.
    drop = [(p, 0) for p in (6, 7, 8, 9)]
    mmm = FakeMMM(GEOS, 10, CHANNELS, seed=4, ragged_drop=drop)
    design = _design(GEOS, design="scaling", duration=8)
    mask, codes, window_codes, dur_eff, warns = _resolve_treated_rows(
        mmm, design, duration=8
    )
    assert dur_eff == 6  # geo 0 only reports periods 0..5
    assert any("ragged" in w.lower() or "shrank" in w.lower() for w in warns)
    # the absent (period, geo 0) cells are never selected
    g0 = mmm.geo_names.index(GEOS[0])
    selected = set(zip(mmm.time_idx[mask].tolist(), mmm.geo_idx[mask].tolist()))
    assert not any((p, g0) in selected for p in (6, 7, 8, 9))


def test_paired_delta_is_exact_no_ppc_noise():
    """Two contribution passes at the same max_draws are draw-paired, so the
    untouched channels' contributions are byte-identical (OC-2/F1)."""
    mmm = FakeMMM(GEOS, 40, CHANNELS, seed=5)
    x = mmm.X_media_raw
    a = mmm.sample_channel_contributions(X_media=x, max_draws=20)
    b = mmm.sample_channel_contributions(X_media=x.copy(), max_draws=20)
    assert np.array_equal(a, b)


def test_build_experiment_media_holdout_zeroes_treated():
    mmm = FakeMMM(GEOS, 40, CHANNELS, seed=6)
    design = _design(GEOS, design="holdout")
    mask, _codes, window_codes, _d, _w = _resolve_treated_rows(mmm, design, duration=8)
    x_exp, ch_idx, n_rows = build_experiment_media(
        mmm, design, treated_mask=mask, window_codes=window_codes
    )
    assert np.all(x_exp[mask, ch_idx] == 0.0)  # gone dark
    # other channels and non-treated rows untouched
    untouched = ~mask
    assert np.array_equal(x_exp[untouched], mmm.X_media_raw[untouched])


def test_margin_resolution_rules():
    assert _resolve_margin(None, None, 0.6, "revenue", None) == (0.6, "param")
    assert _resolve_margin(None, None, 0.6, "units", None) == (None, "none")
    assert _resolve_margin(None, None, 0.6, "units", 10.0) == (6.0, "param")
    assert _resolve_margin(None, None, 0.6, "other", None) == (None, "none")
    m, src = _resolve_margin(
        {"economics": {"gross_margin": 0.4}}, None, None, "revenue", None
    )
    assert (m, src) == (0.4, "preferences")
    # non-positive margin ignored
    assert _resolve_margin(None, None, -1.0, "revenue", None) == (None, "none")


def test_learning_to_cost_basis_labels():
    mmm = FakeMMM(GEOS, 60, CHANNELS, seed=7)
    design = _design(GEOS, design="scaling")
    # no EVOI → 'unavailable'
    oc = compute_opportunity_cost(mmm, design, max_draws=30)
    assert oc.learning_to_cost_basis == "unavailable"
    assert oc.learning_to_cost_ratio is None
    # EVOI floored at 0 → channel already precise
    oc2 = compute_opportunity_cost(mmm, design, evoi_kpi_units=0.0, max_draws=30)
    assert oc2.learning_to_cost_basis == "channel_already_precise"
    # positive EVOI → kpi_per_week ratio
    oc3 = compute_opportunity_cost(mmm, design, evoi_kpi_units=500.0, max_draws=30)
    assert oc3.learning_to_cost_basis == "kpi_per_week"
    assert oc3.learning_to_cost_ratio is not None


def test_national_flighting_uses_union_window_on_disjoint_coverage():
    """Review finding 4: a national pulse has no control arm, so it must use the
    UNION of geo coverage — disjoint geo coverage must NOT raise 'no common
    reporting periods' (that rule is only for geo-lift treatment vs control)."""
    # geo 0 reports periods 0..29, geo 1 reports 30..59 (fully disjoint)
    drop = [(p, 0) for p in range(30, 60)] + [(p, 1) for p in range(0, 30)]
    mmm = FakeMMM(["A", "B"], 60, CHANNELS, seed=9, ragged_drop=drop)
    design = {
        "design_key": "national_flighting",
        "channel": "TV",
        "kpi": "Sales",
        "duration": 8,
        "intensity_pct": 0.0,
        "schedule": [
            {"week_offset": w, "multiplier": 1.5 if w % 2 else 0.5} for w in range(8)
        ],
    }
    mask, codes, window_codes, dur_eff, warns = _resolve_treated_rows(
        mmm, design, duration=8
    )
    assert mask.any()  # union, not an empty intersection
    assert dur_eff == 8
    assert set(window_codes.tolist()) == set(range(52, 60))  # last 8 of the union


def test_to_dict_is_json_safe():
    import json

    mmm = FakeMMM(GEOS, 50, CHANNELS, seed=8)
    design = _design(GEOS, design="holdout")
    oc = compute_opportunity_cost(mmm, design, margin_per_kpi=0.3, max_draws=30)
    json.dumps(oc.to_dict())  # must not raise
