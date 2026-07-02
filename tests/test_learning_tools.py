"""Continuous-learning agent tools (wiring §3.4) + the service layer they wrap.

Fast, no-MCMC: the service's config validation / design / ingest / CSV paths
are pure; the tool bodies run against a seeded sessions store + workspace with
``service.fit_and_plan`` monkeypatched to a canned SNAPSHOT (the real fit is
covered by tests/test_learning_endpoints.py's tiny-NUTS end-to-end)."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    return S


@pytest.fixture()
def session(env):
    pid = env.create_project("P")["project_id"]
    tid = env.create_session("s", project_id=pid)["thread_id"]
    return env, pid, tid, {"configurable": {"thread_id": tid}}


def _text(cmd) -> str:
    return cmd.update["messages"][0].content


CANNED_SNAPSHOT = {
    "schema_version": 1,
    "fitted_at": 0.0,
    "evidence": {
        "n_rows": 24,
        "n_summaries": 0,
        "n_waves": 1,
        "shape_identified": {"Chatter": True, "Pulse": False},
    },
    "diagnostics": {"max_rhat": 1.02, "min_ess": 90, "n_draws": 40, "flags": []},
    "recommendation": {"Chatter": 170000.0, "Pulse": 110000.0},
    "recommendation_scaled": {"Chatter": 1.21, "Pulse": 0.79},
    "allocation_sd": {"Chatter": 9000.0, "Pulse": 8000.0},
    "funding": [
        {
            "channel": "Chatter",
            "mroas_mean": 1.8,
            "prob_above_line": 0.94,
            "funded": True,
            "verdict": "FUND",
        },
        {
            "channel": "Pulse",
            "mroas_mean": 0.7,
            "prob_above_line": 0.22,
            "funded": False,
            "verdict": "CUT",
        },
    ],
    "regret": {
        "e_regret_kpi": 0.5,
        "e_regret_dollars": 6.5,
        "enbs": -24993.5,
        "stop": True,
        "margin": 1.0,
        "population": 13.0,
        "wave_cost": 25000.0,
    },
    "gamma": [
        {
            "pair": ["Chatter", "Pulse"],
            "mean": -0.42,
            "p5": -0.7,
            "p95": -0.1,
            "sign": "neg",
            "prior_dominated": False,
        }
    ],
    "response_curves": {
        "Chatter": {
            "spend_dollars": [0.0, 140000.0, 280000.0],
            "mean": [0.0, 1.0, 1.4],
            "lo": [0.0, 0.8, 1.1],
            "hi": [0.0, 1.2, 1.7],
            "current": 140000.0,
        },
        "Pulse": {
            "spend_dollars": [0.0, 140000.0, 280000.0],
            "mean": [0.0, 0.4, 0.55],
            "lo": [0.0, 0.2, 0.3],
            "hi": [0.0, 0.6, 0.8],
            "current": 140000.0,
        },
    },
    "warnings": [],
}


# ── service: config validation + dollars↔scaled ──────────────────────────────


class TestNewProgramState:
    def test_defaults_center_ref_and_budget_scaling(self):
        from mmm_framework.continuous_learning import service as svc

        state = svc.new_program_state(
            {"channels": ["A", "B"], "budget": 200000, "value_per_unit": 5.0}
        )
        # center defaults to budget/K; spend_ref = center -> scaled center 1.0
        assert np.allclose(state.center, [1.0, 1.0])
        assert np.allclose(state.spend_ref, [100000.0, 100000.0])
        assert state.B == pytest.approx(2.0)  # budget in scaled units
        assert state.mode == "fixed" and state.activation == "hill"

    @pytest.mark.parametrize(
        "patch, match",
        [
            ({"channels": []}, "non-empty"),
            ({"channels": ["A", "A"]}, "duplicates"),
            ({"budget": 0}, "budget"),
            ({"value_per_unit": -1}, "value_per_unit"),
            ({"activation": "quadratic"}, "activation"),
            ({"mode": "sideways"}, "mode"),
            ({"pair_signs": {"01": "neg"}}, "pair_signs keys"),
            ({"pair_signs": {"0,1": "sorta"}}, "pair sign"),
            ({"pair_signs": {"0,7": "neg"}}, "invalid for"),
            ({"center": {"A": -5.0}}, "center"),
        ],
    )
    def test_validation_errors(self, patch, match):
        from mmm_framework.continuous_learning import service as svc

        cfg = {"channels": ["A", "B"], "budget": 100.0, "value_per_unit": 1.0}
        cfg.update(patch)
        with pytest.raises(ValueError, match=match):
            svc.new_program_state(cfg)

    def test_uniform_global_spend_ref_for_heterogeneous_centers(self):
        from mmm_framework.continuous_learning import service as svc

        cfg = {
            "channels": ["A", "B"],
            "budget": 140000,
            "value_per_unit": 5.0,
            "center": {"A": 100000, "B": 40000},
        }
        state = svc.new_program_state(cfg)
        # ONE global reference (mean center, floored at $1) for every channel:
        # the scaled budget simplex is then EXACT in dollars.
        assert np.allclose(state.spend_ref, [70000.0, 70000.0])
        assert np.allclose(state.center, [100000 / 70000, 40000 / 70000])
        assert float(state.B) * 70000.0 == pytest.approx(140000.0)
        # explicit per-channel overrides still win (heterogeneous allowed)
        state2 = svc.new_program_state({**cfg, "spend_ref": {"A": 1000.0}})
        assert np.allclose(state2.spend_ref, [1000.0, 70000.0])

    def test_free_mode_requires_unit_spend_ref(self):
        from mmm_framework.continuous_learning import service as svc

        cfg = {
            "channels": ["A", "B"],
            "budget": 200000,
            "value_per_unit": 5.0,
            "mode": "free",
        }
        # free mode prices spend in scaled units -> only sound at ref == $1
        with pytest.raises(ValueError, match="free"):
            svc.new_program_state(cfg)
        ok = svc.new_program_state({**cfg, "spend_ref": {"A": 1.0, "B": 1.0}})
        assert ok.mode == "free"

    def test_arms_expand_with_default_signs_and_group_budgets(self):
        from mmm_framework.continuous_learning import ARM_SEP
        from mmm_framework.continuous_learning import service as svc

        cfg = {
            "channels": ["Search", "Display"],
            "arms": {"Search": ["Brand", "NonBrand"]},
            "budget": 300000,
            "value_per_unit": 5.0,
            "center": {"Search": 200000, "Display": 100000},
        }
        state = svc.new_program_state(cfg)
        assert state.channels == [
            f"Search{ARM_SEP}Brand",
            f"Search{ARM_SEP}NonBrand",
            "Display",
        ]
        # parent-level center splits equally over its arms
        assert np.allclose(state.spend_ref, [100000.0, 100000.0, 100000.0])
        # within-parent siblings substitute; cross-parent weak
        assert state.pair_signs[(0, 1)] == "neg"
        assert state.pair_signs[(0, 2)] == "weak"

        groups = svc.group_budgets_for(state, cfg)
        assert len(groups) == 1
        idx, bg = groups[0]
        assert idx == [0, 1]
        # the split parent's scaled budget = its arms' scaled center (factor 1
        # here since budget == sum(center))
        assert bg == pytest.approx(float(state.center[0] + state.center[1]))
        # no arms -> no groups
        assert svc.group_budgets_for(state, {"channels": []}) is None


# ── service: design / CSV / ingest ───────────────────────────────────────────


class TestDesignAndIngest:
    def _state(self):
        from mmm_framework.continuous_learning import service as svc

        return svc.new_program_state(
            {"channels": ["Chatter", "Pulse"], "budget": 280000, "value_per_unit": 5.0}
        )

    def test_design_wave_cells_labels_warning(self):
        from mmm_framework.continuous_learning import service as svc

        state = self._state()
        d = svc.design_wave(state, delta=0.6, n_geo=6, n_holdout=1)
        assert d["n_cells"] == 1 + 4 + 2 + 2  # K=2, 1 probed pair
        assert d["cell_labels"][0] == "center"
        assert d["cell_labels"][-1] == "Pulse shutoff"
        assert len(d["cells_dollars"]) == d["n_cells"]
        assert d["warnings"] and "cells" in d["warnings"][0]  # 9 cells > 6 geos
        assert d["assignment"]["cell_idx"][0] == -1  # holdout marker
        # probe_pairs=[] drops the off-axis cells
        assert svc.design_wave(state, probe_pairs=[])["n_cells"] == 7

    def test_design_wave_stratified_on_flag(self):
        from mmm_framework.continuous_learning import service as svc

        state = self._state()
        # no ingested data yet -> round-robin
        d = svc.design_wave(state, n_geo=8)
        assert d["assignment"]["stratified_on"] is None
        # ingest a wave -> stratified on the accumulated per-geo KPI
        rows = [
            {"geo": f"g{i}", "Chatter": 140000.0, "Pulse": 70000.0, "y": 5.0 + i}
            for i in range(8)
        ]
        svc.ingest_wave_rows(state, rows)
        d2 = svc.design_wave(state)  # n_geo defaults to the pinned geo set
        assert d2["assignment"]["stratified_on"] == "accumulated_kpi"
        assert len(d2["assignment"]["cell_idx"]) == 8
        # explicit opt-out keeps round-robin
        d3 = svc.design_wave(state, stratify=False)
        assert d3["assignment"]["stratified_on"] is None

    def test_design_wave_optimize_without_posterior_warns(self):
        from mmm_framework.continuous_learning import service as svc

        state = self._state()
        d = svc.design_wave(state, optimize=True, n_geo=10)
        assert "kg" not in d
        assert any("knowledge-gradient" in w for w in d["warnings"])
        assert d["delta"] == 0.6  # the fixed-delta path is preserved

    def test_design_wave_optimize_with_posterior_picks_candidate(self):
        from mmm_framework.continuous_learning import Posterior
        from mmm_framework.continuous_learning import service as svc
        from mmm_framework.continuous_learning.model import pair_name

        state = self._state()
        rng = np.random.default_rng(0)
        n = 120
        s = {
            "beta": np.abs(rng.normal(1.5, 0.2, (n, 2))),
            "kappa": np.abs(rng.normal(0.8, 0.1, (n, 2))),
            "alpha": np.clip(rng.normal(2.0, 0.2, (n, 2)), 0.5, 5),
            "sigma": np.abs(rng.normal(0.5, 0.05, n)),
            pair_name(state.channels, (0, 1)): rng.normal(0.0, 0.3, n),
        }
        state.posterior = Posterior(
            samples=s, channels=state.channels, pairs=[(0, 1)], pair_signs={}
        )
        d = svc.design_wave(
            state,
            optimize=True,
            candidate_deltas=[0.5, 0.8],
            n_geo=12,
            kg_n_outcomes=8,
        )
        assert d["kg"]["used"] is True
        assert d["delta"] in (0.5, 0.8) and d["delta"] == d["kg"]["chosen_delta"]
        assert len(d["kg"]["scores"]) == 2
        # cell labels reflect the CHOSEN delta
        pct = int(round(d["delta"] * 100))
        assert f"Chatter +{pct}%" in d["cell_labels"]

    def test_rows_from_csv_and_week_column(self):
        from mmm_framework.continuous_learning import service as svc

        rows = svc.rows_from_csv(
            "geo,week,Chatter,Pulse,y\n"
            "g0,2026-01-05,140000,140000,5.2\n"
            "g1,2026-01-05,224000,0,4.9\n"
        )
        assert rows[0]["geo"] == "g0" and rows[0]["Chatter"] == 140000.0
        assert rows[1]["Pulse"] == 0.0 and rows[1]["y"] == 4.9
        with pytest.raises(ValueError, match="no data rows"):
            svc.rows_from_csv("geo,y\n")

    def test_ingest_scales_dollars_and_pins_geos(self):
        from mmm_framework.continuous_learning import service as svc

        state = self._state()
        rows = [
            {"geo": "g0", "chatter": 140000.0, "PULSE": 70000.0, "y": 5.0},
            {"geo": "g1", "chatter": 280000.0, "PULSE": 0.0, "y": 6.0},
        ]
        out = svc.ingest_wave_rows(state, rows)  # case-insensitive columns
        assert out == {"n_rows": 2, "n_geo": 2, "warnings": []}
        assert np.allclose(state.data["spend"][0], [1.0, 0.5])  # scaled by ref
        assert state.geo_ids == ["g0", "g1"]
        # a later wave with a NEW geo is rejected (stable-geo-set rule)
        with pytest.raises(ValueError, match="stable geo set"):
            svc.ingest_wave_rows(
                state,
                [{"geo": "g9", "chatter": 1.0, "PULSE": 1.0, "y": 1.0}],
            )

    def test_ingest_missing_columns(self):
        from mmm_framework.continuous_learning import service as svc

        state = self._state()
        with pytest.raises(ValueError, match="spend columns"):
            svc.ingest_wave_rows(state, [{"geo": "g0", "Chatter": 1.0, "y": 1.0}])
        with pytest.raises(ValueError, match="geo column"):
            svc.ingest_wave_rows(state, [{"Chatter": 1.0, "Pulse": 1.0, "y": 1.0}])
        with pytest.raises(ValueError, match="non-numeric"):
            svc.ingest_wave_rows(
                state,
                [{"geo": "g0", "Chatter": "lots", "Pulse": 1.0, "y": 1.0}],
            )


# ── service: state-file IO (atomic write, corrupt/missing -> clean error) ─────


def test_state_file_atomic_write_and_corruption_error(env):
    from pathlib import Path

    from mmm_framework.continuous_learning import service as svc

    state = svc.new_program_state(
        {"channels": ["A"], "budget": 100.0, "value_per_unit": 1.0}
    )
    path = svc.save_program_state("p", "prog", state)
    d = Path(path).parent
    # atomic replace: no tmp litter left behind
    assert [p.name for p in d.iterdir()] == ["state.npz"]
    back = svc.load_program_state("p", "prog")
    assert back.channels == ["A"]

    # a partial/corrupt file raises the clean ProgramStateError, never BadZipFile
    Path(path).write_bytes(b"definitely not a zip")
    with pytest.raises(svc.ProgramStateError, match="corrupt"):
        svc.load_program_state("p", "prog")
    with pytest.raises(svc.ProgramStateError, match="missing"):
        svc.load_program_state("p", "never-created")


# ── service: ENBS population (geo-periods) ────────────────────────────────────


def test_resolve_population_geo_periods(env):
    from mmm_framework.continuous_learning import service as svc

    state = svc.new_program_state(
        {"channels": ["A", "B"], "budget": 200000, "value_per_unit": 5.0}
    )
    # explicit override = final geo-periods, untouched
    assert svc.resolve_population(state, {"horizon_periods": 13}, 99.0) == (99.0, None)
    # summaries-only (no geo set): horizon × 1 with a per-geo-understated warning
    pop, warn = svc.resolve_population(state, {"horizon_periods": 13})
    assert pop == 13.0 and "per-geo-understated" in warn
    # once the geo set is pinned, population = n_geos × horizon
    rows = [{"geo": f"g{i}", "A": 100000.0, "B": 100000.0, "y": 5.0} for i in range(4)]
    svc.ingest_wave_rows(state, rows)
    pop, warn = svc.resolve_population(state, {"horizon_periods": 13})
    assert pop == 52.0 and warn is None
    # legacy config key `population` is read as horizon periods (back-compat)
    pop, _ = svc.resolve_population(state, {"population": 2})
    assert pop == 8.0


# ── service: fit_and_plan snapshot units (fake fit/plan, no MCMC) ─────────────


class TestFitAndPlanSnapshot:
    def _state(self, svc):
        return svc.new_program_state(
            {"channels": ["A", "B"], "budget": 200000, "value_per_unit": 5.0}
        )

    def _fake_fit_plan(self, svc, state, monkeypatch, mroas_draws, n_rows=0):
        k = len(state.channels)

        class Post:
            pairs = []
            diagnostics = {
                "max_rhat": 1.0,
                "min_ess": 100.0,
                "evidence": {"n_rows": n_rows, "n_summaries": 0},
            }
            n_draws = int(mroas_draws.shape[0])

            def gamma_summary(self):
                return {}

        class Plan:
            recommendation = np.ones(k)
            alloc_sd = np.zeros(k)
            e_regret = 0.5

        Plan.mroas_draws = mroas_draws
        post, plan = Post(), Plan()
        state.fit = lambda **kw: post
        state.plan = lambda **kw: plan
        monkeypatch.setattr(svc, "_response_curves", lambda *a, **kw: {})

    def test_margin_adjusts_funding_verdicts(self, monkeypatch):
        from mmm_framework.continuous_learning import service as svc

        state = self._state(svc)
        ref = float(state.spend_ref[0])  # $100k global
        draws = np.column_stack(
            [np.full(50, 1.5 * ref), np.full(50, 3.0 * ref)]  # per-$: 1.5 / 3.0
        )
        self._fake_fit_plan(svc, state, monkeypatch, draws)
        snap = svc.fit_and_plan(state, margin=0.5, population=1.0, wave_cost=0.0)
        rows = {r["channel"]: r for r in snap["funding"]}
        # mroas_mean stays per REVENUE dollar; the verdict runs on margin-adjusted
        assert rows["A"]["mroas_mean"] == pytest.approx(1.5)
        assert rows["A"]["mroas_margin_adjusted"] == pytest.approx(0.75)
        assert rows["A"]["verdict"] == "CUT" and not rows["A"]["funded"]
        assert rows["B"]["mroas_margin_adjusted"] == pytest.approx(1.5)
        assert rows["B"]["verdict"] == "FUND"
        assert any("margin" in w and "A" in w for w in snap["warnings"])

        # margin = 1 -> no adjustment, no flip warning
        snap1 = svc.fit_and_plan(state, margin=1.0, population=1.0, wave_cost=0.0)
        rows1 = {r["channel"]: r for r in snap1["funding"]}
        assert rows1["A"]["verdict"] == "FUND"
        assert rows1["A"]["mroas_margin_adjusted"] == pytest.approx(1.5)
        assert not any("changes the funding verdict" in w for w in snap1["warnings"])

    def test_n_waves_extra_warnings_and_center_distance(self, monkeypatch):
        from mmm_framework.continuous_learning import service as svc

        state = self._state(svc)
        # panel spend 10× the configured center -> the per-geo vs total tell
        rows = [
            {"geo": f"g{i}", "A": 1000000.0, "B": 1000000.0, "y": 5.0} for i in range(4)
        ]
        svc.ingest_wave_rows(state, rows)
        draws = np.full((50, 2), 1.5 * float(state.spend_ref[0]))
        self._fake_fit_plan(svc, state, monkeypatch, draws, n_rows=4)
        snap = svc.fit_and_plan(
            state,
            margin=1.0,
            population=1.0,
            wave_cost=0.0,
            n_waves=3,
            extra_warnings=["custom warning"],
        )
        # n_waves counts ingested evidence batches passed by the caller
        assert snap["evidence"]["n_waves"] == 3
        assert "custom warning" in snap["warnings"]
        assert any("far from the program center" in w for w in snap["warnings"])


# ── service: past-experiment import is idempotent ─────────────────────────────


def test_import_experiment_summaries_dedup(session):
    env, pid, tid, _cfg = session
    from mmm_framework.continuous_learning import service as svc

    state = svc.new_program_state(
        {"channels": ["Chatter", "Pulse"], "budget": 280000, "value_per_unit": 5.0}
    )
    exp = env.upsert_experiment(
        project_id=pid,
        channel="Chatter",
        status="completed",
        start_date="2026-01-05",
        end_date="2026-03-02",
        readout={
            "value": 12.0,
            "se": 2.0,
            "estimand": "contribution",
            "spend_per_period": 56000.0,
            "n_treated_units": 4,
        },
    )
    # a within-call repeat is skipped
    report = svc.import_experiment_summaries(state, [exp, dict(exp)])
    assert report["imported"] == 1
    assert report["imported_ids"] == [exp["id"]]
    assert [s["reason"] for s in report["skipped"]] == ["already imported"]
    assert len(state.summaries) == 1
    # re-importing the same experiment later is a no-op (idempotent)
    report2 = svc.import_experiment_summaries(state, [exp])
    assert report2["imported"] == 0 and report2["imported_ids"] == []
    assert report2["skipped"] == [{"id": exp["id"], "reason": "already imported"}]
    assert len(state.summaries) == 1


# ── program resolution ────────────────────────────────────────────────────────


def test_resolve_program(session):
    env, pid, tid, _cfg = session
    from mmm_framework.agents import learning_tools as LT

    # none yet -> actionable error
    prog, err = LT._resolve_program(pid, None)
    assert prog is None and "start_learning_program" in err

    p1 = env.create_learning_program(project_id=pid, channels=["A"], config={})
    prog, err = LT._resolve_program(pid, None)
    assert err is None and prog["id"] == p1["id"]  # single active -> picked

    p2 = env.create_learning_program(project_id=pid, channels=["B"], config={})
    prog, err = LT._resolve_program(pid, None)
    assert prog is None and "Multiple active" in err and p2["id"] in err

    # explicit id wins; unknown/cross-project ids error
    prog, err = LT._resolve_program(pid, p1["id"])
    assert err is None and prog["id"] == p1["id"]
    prog, err = LT._resolve_program(pid, "nope")
    assert prog is None and "Unknown learning program" in err
    other = env.create_project("Q")["project_id"]
    foreign = env.create_learning_program(project_id=other, channels=["Z"], config={})
    prog, err = LT._resolve_program(pid, foreign["id"])
    assert prog is None and "Unknown learning program" in err

    # stopped programs are not auto-resolved but are listed in the error
    env.update_learning_program(p1["id"], status="stopped")
    env.update_learning_program(p2["id"], status="stopped")
    prog, err = LT._resolve_program(pid, None)
    assert prog is None and "No ACTIVE learning program" in err


# ── tool bodies (fit mocked) ──────────────────────────────────────────────────


def test_start_design_record_status_stop_flow(session, monkeypatch):
    env, pid, tid, cfg = session
    from mmm_framework.agents import learning_tools as LT
    from mmm_framework.continuous_learning import service as svc

    # start
    cmd = LT.start_learning_program.func(
        name="Nomi Q3",
        channels=["Chatter", "Pulse"],
        budget_per_period=280000.0,
        value_per_unit=5.0,
        state={},
        wave_cost=25000.0,
        horizon_periods=13,
        config=cfg,
        tool_call_id="t1",
    )
    assert "Nomi Q3" in _text(cmd)
    payload = cmd.update["dashboard_data"]["learning_program"]
    prog_id = payload["program_id"]
    prog = env.get_learning_program(prog_id)
    assert prog["project_id"] == pid and prog["channels"] == ["Chatter", "Pulse"]
    assert prog["config"]["wave_cost"] == 25000.0
    # horizon rides the config; population (geo-periods) is resolved at fit time
    assert prog["config"]["horizon_periods"] == 13
    assert "population" not in prog["config"]
    from pathlib import Path

    assert Path(prog["state_path"]).exists()

    # bad config -> friendly message, no row
    bad = LT.start_learning_program.func(
        name="bad",
        channels=[],
        budget_per_period=1.0,
        value_per_unit=1.0,
        state={},
        config=cfg,
        tool_call_id="t1b",
    )
    assert "Could not start" in _text(bad)

    # design
    cmd = LT.design_learning_wave.func(
        state={}, n_geo=12, n_holdout=2, config=cfg, tool_call_id="t2"
    )
    txt = _text(cmd)
    assert "9 cells" in txt and "record_learning_wave" in txt
    assert cmd.update["dashboard_data"]["tables"]  # design table published
    waves = env.list_learning_waves(prog_id)
    assert len(waves) == 1 and waves[0]["status"] == "designed"

    # bad probe_pairs string -> friendly message
    bad = LT.design_learning_wave.func(
        state={}, probe_pairs="banana", config=cfg, tool_call_id="t2b"
    )
    assert "probe_pairs" in _text(bad)

    # record (fit mocked to the canned snapshot)
    monkeypatch.setattr(svc, "fit_and_plan", lambda state, **kw: dict(CANNED_SNAPSHOT))
    rows = [
        {"geo": f"g{i}", "Chatter": 140000.0, "Pulse": 140000.0, "y": 5.0}
        for i in range(4)
    ]
    cmd = LT.record_learning_wave.func(
        state={}, rows=rows, config=cfg, tool_call_id="t3"
    )
    txt = _text(cmd)
    assert "FUND" in txt and "CUT" in txt and "stop testing" in txt
    dd = cmd.update["dashboard_data"]
    assert dd["learning_program"]["snapshot"]["schema_version"] == 1
    assert len(dd["tables"]) == 2  # allocation + funding
    assert len(dd["plots"]) == 2  # funding bars + response curves
    prog = env.get_learning_program(prog_id)
    assert prog["summary"]["recommendation"]["Chatter"] == 170000.0
    ingested = [
        w for w in env.list_learning_waves(prog_id) if w["status"] == "ingested"
    ]
    assert len(ingested) == 1 and ingested[0]["source"] == "wave"

    # status (reads the stored snapshot, no refit)
    cmd = LT.get_learning_program_status.func(state={}, config=cfg, tool_call_id="t4")
    txt = _text(cmd)
    assert "Chatter" in txt and "ENBS" in txt.replace("Enbs", "ENBS")
    assert "Chatter×Pulse" in txt  # data-informed synergy line

    # stopping: recommend-only first, then confirm
    cmd = LT.check_learning_stopping.func(config=cfg, tool_call_id="t5")
    assert "confirm_stop" in _text(cmd)
    assert env.get_learning_program(prog_id)["status"] == "active"
    cmd = LT.check_learning_stopping.func(
        confirm_stop=True, config=cfg, tool_call_id="t6"
    )
    assert "stopped" in _text(cmd)
    assert env.get_learning_program(prog_id)["status"] == "stopped"
    # overridden economics can flip the verdict (tiny wave cost -> keep testing)
    cmd = LT.check_learning_stopping.func(
        program_id=prog_id, wave_cost=0.5, config=cfg, tool_call_id="t7"
    )
    assert "keep testing" in _text(cmd)


def test_record_wave_period_col_reaches_the_service(session, monkeypatch):
    """[9]/[13] A national-time-effect program is usable through the agent
    tool: record_learning_wave(period_col=...) threads to ingest_wave_rows,
    and omitting it auto-detects a week column instead of dead-ending."""
    env, pid, tid, cfg = session
    from mmm_framework.agents import learning_tools as LT
    from mmm_framework.continuous_learning import service as svc

    config = {
        "channels": ["Chatter", "Pulse"],
        "budget": 280000,
        "value_per_unit": 5.0,
        "time_effect": "national",
    }
    state = svc.new_program_state(config)
    prog = env.create_learning_program(
        project_id=pid, name="National", channels=state.channels, config=config
    )
    path = svc.save_program_state(pid, prog["id"], state)
    env.update_learning_program(prog["id"], state_path=path)
    monkeypatch.setattr(svc, "fit_and_plan", lambda state, **kw: dict(CANNED_SNAPSHOT))

    rows = [
        {"geo": f"g{i}", "week": wk, "Chatter": 140000.0, "Pulse": 140000.0, "y": 5.0}
        for wk in ("2026-01-05", "2026-01-12")
        for i in range(3)
    ]
    cmd = LT.record_learning_wave.func(
        state={},
        rows=rows,
        period_col="week",
        program_id=prog["id"],
        config=cfg,
        tool_call_id="t1",
    )
    assert "Wave recorded: 6 rows" in _text(cmd)
    saved = svc.load_program_state(pid, prog["id"])
    assert "period_idx" in saved.data
    np.testing.assert_array_equal(np.unique(saved.data["period_idx"]), [0, 1])

    # omitted period_col -> auto-detected 'week' column, surfaced as a warning
    cmd = LT.record_learning_wave.func(
        state={},
        rows=[
            {
                "geo": f"g{i}",
                "week": "2026-01-19",
                "Chatter": 140000.0,
                "Pulse": 140000.0,
                "y": 5.0,
            }
            for i in range(3)
        ],
        program_id=prog["id"],
        config=cfg,
        tool_call_id="t2",
    )
    assert "auto-detected" in _text(cmd)
    saved = svc.load_program_state(pid, prog["id"])
    np.testing.assert_array_equal(np.unique(saved.data["period_idx"]), [0, 1, 2])


def test_record_wave_requires_rows_or_csv(session):
    env, pid, tid, cfg = session
    from mmm_framework.agents import learning_tools as LT

    env.create_learning_program(project_id=pid, channels=["A"], config={})
    cmd = LT.record_learning_wave.func(state={}, config=cfg, tool_call_id="t")
    assert "Provide the wave's observations" in _text(cmd)
    # with rows but no state file on disk -> the tool reports it clearly
    cmd = LT.record_learning_wave.func(
        state={},
        rows=[{"geo": "g0", "A": 1.0, "y": 1.0}],
        config=cfg,
        tool_call_id="t2",
    )
    assert "state file is missing" in _text(cmd)


def test_import_past_experiments_tool(session, monkeypatch):
    env, pid, tid, cfg = session
    from mmm_framework.agents import learning_tools as LT
    from mmm_framework.continuous_learning import service as svc

    LT.start_learning_program.func(
        name="prog",
        channels=["Chatter", "Pulse"],
        budget_per_period=280000.0,
        value_per_unit=5.0,
        state={},
        config=cfg,
        tool_call_id="t1",
    )

    # nothing to import yet
    cmd = LT.import_past_experiments.func(state={}, config=cfg, tool_call_id="t2")
    assert "No completed/calibrated experiments" in _text(cmd)

    good = env.upsert_experiment(
        project_id=pid,
        channel="Chatter",
        status="completed",
        start_date="2026-01-05",
        end_date="2026-03-02",
        readout={
            "value": 12.0,
            "se": 2.0,
            "estimand": "contribution",
            "spend_per_period": 56000.0,
            "n_treated_units": 4,
        },
    )
    env.upsert_experiment(project_id=pid, channel="Mystery", status="completed")

    monkeypatch.setattr(svc, "fit_and_plan", lambda state, **kw: dict(CANNED_SNAPSHOT))
    cmd = LT.import_past_experiments.func(state={}, config=cfg, tool_call_id="t3")
    txt = _text(cmd)
    assert "Imported **1**" in txt and "Skipped" in txt
    waves = env.list_learning_waves(
        cmd.update["dashboard_data"]["learning_program"]["program_id"]
    )
    assert waves[-1]["source"] == "experiment_import"
    # provenance = ONLY the successfully imported ids (skips ride observations)
    assert waves[-1]["experiment_ids"] == [good["id"]]
    # the summaries persisted into the saved state (real ingest, mocked fit)
    prog_id = waves[-1]["program_id"]
    state = svc.load_program_state(pid, prog_id)
    assert len(state.summaries) == 1
    assert state.summaries[0]["channel"] == "Chatter"

    # a repeat call is idempotent end-to-end: the readout is NOT re-ingested
    cmd = LT.import_past_experiments.func(state={}, config=cfg, tool_call_id="t4")
    txt = _text(cmd)
    assert "None of the experiments could be converted" in txt
    assert "already imported" in txt
    assert len(svc.load_program_state(pid, prog_id).summaries) == 1
    assert len(env.list_learning_waves(prog_id)) == len(waves)  # no new wave row

    # explicit ids from ANOTHER project are never imported (scoping)
    other = env.create_project("Q")["project_id"]
    foreign = env.upsert_experiment(
        project_id=other,
        channel="Chatter",
        status="completed",
        start_date="2026-01-05",
        end_date="2026-03-02",
        readout={
            "value": 9.0,
            "se": 2.0,
            "estimand": "contribution",
            "spend_per_period": 56000.0,
            "n_treated_units": 4,
        },
    )
    cmd = LT.import_past_experiments.func(
        state={}, experiment_ids=[foreign["id"], "ghost"], config=cfg, tool_call_id="t5"
    )
    txt = _text(cmd)
    assert "different project" in txt and "not found in this project" in txt
    assert len(svc.load_program_state(pid, prog_id).summaries) == 1


def test_tools_are_registered_as_spine(monkeypatch):
    """All six tools ride TOOLS/EXPERT_TOOLS and survive every mode (spine)."""
    from mmm_framework.agents import tools as T

    names = {
        "start_learning_program",
        "import_past_experiments",
        "design_learning_wave",
        "record_learning_wave",
        "get_learning_program_status",
        "check_learning_stopping",
    }
    all_names = {t.name for t in T.EXPERT_TOOLS}
    assert names <= all_names
    assert not (names & T._MMM_ONLY_TOOL_NAMES)
    assert not (names & T.HEAVY_TOOL_NAMES)
    for mode in ("mmm", "causal_inference", "general_bayes", "descriptive"):
        bound = {t.name for t in T.get_tools_for_mode(mode, role="expert")}
        assert names <= bound, mode
