"""Adversarial, multi-seed grading of the outlier detectors.

``test_outliers.py`` grades the canonical single-defect scenarios; this file
removes the remaining cherry-picking:

* **multi-seed sweeps** — recall/false-positive rates must hold across seeds,
  not on one lucky draw;
* **realistic error magnitudes & positions** — decimal shifts (x10) next to
  flight peaks, missed-load zeros, errors near series edges, errors colliding
  with seasonal peaks — not just a 15x spike in the middle of the series;
* **benign look-alikes must survive** — a genuinely heavy (real) promo flight
  week must NOT be flagged;
* **documented detection limits** — a 2x double-count is statistically
  indistinguishable from a heavy flight week; the suite asserts it is NOT
  flagged so the limit is encoded, not hidden;
* **hard worlds** — the 7-channel `realistic` scenario (confounders, a
  near-collinear channel pair, mediator) as a false-positive control, and a
  geo panel where only one slice is corrupted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synth.dgp import (  # noqa: E402
    build as build_scenario,
    make_clean,
    make_mixed_data_errors,
    make_seasonality_misspec,
    make_spend_outliers,
)

from mmm_framework.eda import (
    EDAPanel,
    detect_outliers,
    recommend_treatments,
)  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]


def panel_from_frames(wide: pd.DataFrame, kpi, media, controls) -> EDAPanel:
    return EDAPanel(
        df_wide=wide,
        df_long=None,
        kpi=kpi,
        media=list(media),
        controls=list(controls),
        unassigned=[],
        dims=[],
        date_col="Period",
        freq="W-MON",
        roles_source="spec",
        source_path=None,
    )


def panel_from_scenario(sc) -> EDAPanel:
    cols = {"Sales": sc.y.to_numpy()}
    cols.update({c: sc.spend[c].to_numpy() for c in sc.spend.columns})
    cols.update({c: sc.controls[c].to_numpy() for c in sc.controls.columns})
    wide = pd.DataFrame(cols, index=sc.weeks)
    return panel_from_frames(wide, "Sales", sc.spend.columns, sc.controls.columns)


def spike_flags(report, variable=None):
    return [
        f
        for f in report.flags
        if f.kind == "isolated_spike" and (variable is None or f.variable == variable)
    ]


# ---------------------------------------------------------------------------
# multi-seed sweeps (no single lucky draw)
# ---------------------------------------------------------------------------


class TestMultiSeed:
    def test_spend_outlier_recall_holds_across_seeds(self):
        """Spike positions are random per seed — recall must not depend on
        where the dgp happened to put them."""
        for seed in SEEDS:
            sc = make_spend_outliers(seed=seed)
            report = detect_outliers(panel_from_scenario(sc))
            for channel, week in sc.notes["spike_weeks"].items():
                expected_period = str(sc.weeks[week].date())
                assert any(
                    f.period == expected_period for f in spike_flags(report, channel)
                ), f"seed {seed}: missed {channel} spike at {expected_period}"

    def test_clean_world_fp_rate_aggregated_across_seeds(self):
        total_flags, total_points = 0, 0
        for seed in SEEDS:
            sc = make_clean(seed=seed)
            report = detect_outliers(panel_from_scenario(sc))
            point_flags = [f for f in report.flags if f.kind != "level_shift"]
            total_flags += len(point_flags)
            total_points += len(sc.weeks) * (1 + len(sc.spend.columns))
            # media must be pristine in a clean world (spike OR drop)
            media_flags = [f for f in point_flags if f.variable != "Sales"]
            assert len(media_flags) <= 1, (
                f"seed {seed}: {len(media_flags)} media false positives: "
                f"{[(f.variable, f.period, f.kind) for f in media_flags]}"
            )
        assert total_flags / total_points < 0.015

    def test_mixed_errors_graded_across_seeds(self):
        """The realistic defect mix: x10 caught, missed-load caught, x2
        documented as NOT caught, untouched channel stays clean."""
        tv_hits = se_hits = 0
        for seed in SEEDS:
            sc = make_mixed_data_errors(seed=20 + seed)
            report = detect_outliers(panel_from_scenario(sc))
            errors = sc.notes["errors"]

            tv_period = str(sc.weeks[errors["TV"]["week"]].date())
            tv_hits += any(f.period == tv_period for f in spike_flags(report, "TV"))

            se_period = str(sc.weeks[errors["Search"]["week"]].date())
            se_hits += any(
                f.kind == "isolated_drop" and f.period == se_period
                for f in report.flags
                if f.variable == "Search"
            )

            # KNOWN LIMIT: the x2 double-count looks like a heavy flight week.
            so_period = str(sc.weeks[errors["Social"]["week"]].date())
            assert not any(
                f.period == so_period for f in spike_flags(report, "Social")
            ), (
                f"seed {seed}: a 2x double-count was flagged — if the detector "
                "now separates 2x errors from real flights, update this test "
                "AND the detect_outliers docstring (it documents the limit)."
            )

            # within-scenario false-positive control
            assert not [
                f for f in report.flags if f.variable == "Display"
            ], f"seed {seed}: flags on the untouched control channel"
        assert tv_hits >= 4, f"x10 decimal shift caught only {tv_hits}/5 seeds"
        assert se_hits >= 4, f"missed-load zero caught only {se_hits}/5 seeds"


# ---------------------------------------------------------------------------
# realistic remediation semantics on the mixed world
# ---------------------------------------------------------------------------


class TestMixedErrorRemediation:
    def test_drop_gets_impute_not_winsorize_and_promos_get_dummies(self):
        sc = make_mixed_data_errors()
        panel = panel_from_scenario(sc)
        report = detect_outliers(panel)
        actions = recommend_treatments(panel, report.flags, report.config)

        drop_flag_ids = {f.flag_id for f in report.flags if f.kind == "isolated_drop"}
        assert drop_flag_ids
        imputes = [a for a in actions if a.strategy == "impute"]
        assert {fid for a in imputes for fid in a.flag_ids} >= drop_flag_ids
        # the impute target is the local baseline, not zero and not the max
        search = panel.df_wide["Search"]
        for a in imputes:
            assert (
                0.3 * float(search.median()) < a.params["value"] < float(search.max())
            )

        # real KPI promo shocks -> dummy controls (modeled, never "corrected")
        promo_periods = {str(sc.weeks[w].date()) for w in sc.notes["promo_weeks"]}
        kpi_flags = {f.period for f in report.flags if f.variable == "Sales"}
        found_promos = promo_periods & kpi_flags
        assert found_promos, "promo shocks not surfaced at all"
        dummy_targets = {
            fid for a in actions if a.strategy == "dummy" for fid in a.flag_ids
        }
        for p in found_promos:
            assert f"Sales@{p}" in dummy_targets
        # and no winsorize recommendation ever touches the KPI
        for a in actions:
            if a.strategy == "winsorize":
                assert not any(fid.startswith("Sales@") for fid in a.flag_ids)


# ---------------------------------------------------------------------------
# positions and collisions the easy scenario never exercises
# ---------------------------------------------------------------------------


class TestHardPositions:
    def test_spike_near_series_edges_is_caught(self):
        for week in (2, 152):
            sc = make_clean(seed=3)
            panel = panel_from_scenario(sc)
            wide = panel.df_wide.copy()
            spiked_value = float(wide["TV"].max() * 8.0)
            wide.iloc[week, wide.columns.get_loc("TV")] = spiked_value
            panel = panel_from_frames(
                wide, "Sales", sc.spend.columns, sc.controls.columns
            )
            report = detect_outliers(panel)
            period = str(wide.index[week].date())
            assert any(
                f.period == period for f in spike_flags(report, "TV")
            ), f"missed an 8x spike at week {week} (near the series edge)"

    def test_spike_on_seasonal_peak_is_caught(self):
        """A decimal shift landing ON the seasonal peak of the seasonally
        concentrated channel (Social spends 3x in Q4, holiday weeks on top)
        must still be caught — the hardest collision: error on the very week
        where big numbers are legitimate."""
        sc = make_seasonality_misspec()
        panel = panel_from_scenario(sc)
        wide = panel.df_wide.copy()
        hol_weeks = [t for t in range(len(wide)) if (t % 52) in (47, 48, 50, 51)]
        social = wide["Social"].to_numpy()
        week = max(hol_weeks, key=lambda t: social[t])  # THE seasonal peak
        wide.iloc[week, wide.columns.get_loc("Social")] = float(social[week] * 10.0)
        panel = panel_from_frames(wide, "Sales", sc.spend.columns, sc.controls.columns)
        report = detect_outliers(panel)
        period = str(wide.index[week].date())
        assert any(f.period == period for f in spike_flags(report, "Social"))


# ---------------------------------------------------------------------------
# benign look-alikes must survive
# ---------------------------------------------------------------------------


class TestBenignLookAlikes:
    def test_real_heavy_flight_week_not_flagged(self):
        """One genuinely heavy week — 4x the median, correctly recorded, with
        the corresponding KPI lift — is a planned flight, not an error."""
        for seed in SEEDS:
            sc = make_clean(seed=seed)
            panel = panel_from_scenario(sc)
            wide = panel.df_wide.copy()
            week = 70 + seed
            burst = float(wide["TV"].median() * 4.0)
            wide.iloc[week, wide.columns.get_loc("TV")] = burst
            wide.iloc[week, wide.columns.get_loc("Sales")] += 40.0  # its real lift
            panel = panel_from_frames(
                wide, "Sales", sc.spend.columns, sc.controls.columns
            )
            report = detect_outliers(panel)
            period = str(wide.index[week].date())
            assert not any(
                f.period == period for f in spike_flags(report, "TV")
            ), f"seed {seed}: a real 4x flight week was flagged as an error"

    def test_dark_weeks_in_flighted_channels_not_flagged_as_drops(self):
        """Flighting troughs (near-zero weeks) are normal — across seeds, the
        drop detector must stay quiet on the pulsed clean world."""
        for seed in SEEDS:
            sc = make_clean(seed=seed)
            report = detect_outliers(panel_from_scenario(sc))
            drops = [f for f in report.flags if f.kind == "isolated_drop"]
            assert not drops, f"seed {seed}: flighting trough misread as missed load"


# ---------------------------------------------------------------------------
# hard worlds
# ---------------------------------------------------------------------------


class TestHardWorlds:
    def test_realistic_seven_channel_world_fp_control(self):
        """7 channels, confounders, a near-collinear pair, a mediator — and no
        injected data errors. The detector must not invent any."""
        sc = build_scenario("realistic")
        report = detect_outliers(panel_from_scenario(sc))
        media_error_flags = [
            f for f in report.flags if f.kind in ("isolated_spike", "isolated_drop")
        ]
        assert len(media_error_flags) <= 1, (
            f"false positives in the realistic world: "
            f"{[(f.variable, f.period, f.kind) for f in media_error_flags]}"
        )

    def test_realistic_world_with_injected_decimal_shift(self):
        """x10 on a HEAVY (p75) week of bursty 7-channel Video: 5-6x the
        channel's p99 — sets the normalization scale, must be caught."""
        sc = build_scenario("realistic")
        panel = panel_from_scenario(sc)
        wide = panel.df_wide.copy()
        target = "Video"
        col = wide[target].to_numpy()
        week = int(np.argsort(col)[int(len(col) * 0.75)])
        wide.iloc[week, wide.columns.get_loc(target)] = float(col[week] * 10.0)
        panel = panel_from_frames(wide, "Sales", sc.spend.columns, sc.controls.columns)
        report = detect_outliers(panel)
        period = str(wide.index[week].date())
        assert any(f.period == period for f in spike_flags(report, target))

    def test_decimal_shift_on_dark_week_of_bursty_channel_is_a_known_limit(self):
        """KNOWN LIMIT, encoded deliberately: in a hyper-bursty channel
        (Video p99/median ~ 9), a x10 shift on a MEDIAN week lands INSIDE the
        normal flight range — undetectable in principle. The same numbers show
        it is also harmless: it does not move the channel's normalization
        scale, so the failure mode the detector exists for never materializes."""
        sc = build_scenario("realistic")
        panel = panel_from_scenario(sc)
        wide = panel.df_wide.copy()
        target = "Video"
        col = wide[target].to_numpy()
        week = int(np.argsort(col)[len(col) // 2])
        shifted = float(col[week] * 10.0)
        # the premise: the corrupted value sits within the legit flight range
        assert shifted < 1.5 * float(np.percentile(col, 99))
        # ... so normalization damage is negligible
        assert shifted < 1.5 * float(col.max())
        wide.iloc[week, wide.columns.get_loc(target)] = shifted
        panel = panel_from_frames(wide, "Sales", sc.spend.columns, sc.controls.columns)
        report = detect_outliers(panel)
        period = str(wide.index[week].date())
        assert not any(f.period == period for f in spike_flags(report, target)), (
            "a within-flight-range value was flagged — if the detector can now "
            "separate these, great, but re-check the benign look-alike tests "
            "before relaxing this."
        )

    def test_geo_panel_spike_flagged_in_the_right_slice_only(self):
        """A spike in ONE geography must carry that geo in its dims and leave
        the other geos clean — per-slice detection, not pooled."""
        from .conftest import simple_wide, to_mff_long
        from mmm_framework.eda import load_eda_panel
        import tempfile

        frames = []
        for seed, geo in enumerate(["East", "West", "Central"]):
            wide = simple_wide(n=156, seed=seed)
            if geo == "West":
                wide.loc[wide.index[88], "TV"] = float(wide["TV"].max() * 10.0)
            frames.append(to_mff_long(wide, geography=geo))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "geo.csv"
            pd.concat(frames, ignore_index=True).to_csv(path, index=False)
            panel = load_eda_panel(
                str(path),
                {
                    "kpi": "Sales",
                    "media_channels": [{"name": "TV"}, {"name": "Search"}],
                    "control_variables": [{"name": "Price"}],
                },
            )
        report = detect_outliers(panel)
        tv_spikes = spike_flags(report, "TV")
        assert tv_spikes, "missed the geo-local spike"
        assert all(
            f.dims.get("Geography") == "West" for f in tv_spikes
        ), f"spike attributed to the wrong slice: {[f.dims for f in tv_spikes]}"
