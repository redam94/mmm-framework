"""Phase-A wiring tests for the continuous-learning core library.

Covers the 2026-07-02 review fixes (F1-F9), the summary-observation likelihood
(past experiments, no panel), the registry->summaries converter, the
``.npz``/JSON persistence, sub-channel arms + grouped budget constraints, and
the one-Thompson-pass ``plan_from_posterior``. Fast tests fabricate posteriors
(no MCMC); the slow tests run tiny NUTS chains (<=100 draws, <=24 geos).

Contract: ``technical-docs/continuous-learning-wiring.md`` §2.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

import mmm_framework.continuous_learning as cl
from mmm_framework.continuous_learning import model, planner, surface

# ── helpers ───────────────────────────────────────────────────────────────────


def _fake_posterior(world, n_geo=40, n=300, seed=0, extra_sigma=True):
    """A Gaussian-ish posterior around a known world — no MCMC needed."""
    rng = np.random.default_rng(seed)
    k = world.n_channels
    s = {
        "beta": np.abs(world.beta + 0.15 * rng.standard_normal((n, k))),
        "kappa": np.abs(world.kappa + 0.08 * rng.standard_normal((n, k))),
        "alpha": np.clip(world.alpha + 0.15 * rng.standard_normal((n, k)), 0.5, 5),
        "A": rng.normal(4, 0.3, n),
        "sigma_a": np.abs(rng.normal(1, 0.1, n)),
        "a_geo": rng.normal(4, 1, (n, n_geo)),
    }
    if extra_sigma:
        s["sigma"] = np.abs(rng.normal(0.5, 0.05, n))
    for idx, (i, j) in enumerate(world.pairs):
        s[model.pair_name(world.channels, (i, j))] = world.gamma_pairs[
            idx
        ] + 0.15 * rng.standard_normal(n)
    return cl.Posterior(
        samples=s, channels=world.channels, pairs=world.pairs, pair_signs={}
    )


def _tiny_panel(k=2, n_geo=6, t=3, seed=0):
    """A minimal syntactically-valid panel (never actually fit)."""
    rng = np.random.default_rng(seed)
    n = n_geo * t
    return {
        "spend": np.abs(rng.normal(0.8, 0.2, (n, k))),
        "geo_idx": np.tile(np.arange(n_geo), t),
        "n_geo": n_geo,
        "y": rng.normal(4.0, 1.0, n),
    }


def _summary(center, c, delta, lift, se=0.1, scale=10.0):
    test = np.asarray(center, dtype=float).copy()
    test[c] = max(0.0, test[c] + delta)
    return {
        "spend_test": test,
        "spend_base": np.asarray(center, dtype=float).copy(),
        "lift": float(lift),
        "se": float(se),
        "scale": float(scale),
    }


# ── F1: data-contract validation in fit() ─────────────────────────────────────


class TestFitValidation:
    def test_out_of_range_geo_idx_rejected(self):
        data = _tiny_panel()
        data["geo_idx"] = data["geo_idx"] + 1  # 1-based: max == n_geo
        with pytest.raises(ValueError, match="geo_idx must lie in"):
            cl.fit(data, channels=["A", "B"])

    def test_negative_geo_idx_rejected(self):
        data = _tiny_panel()
        data["geo_idx"] = data["geo_idx"].copy()
        data["geo_idx"][0] = -1
        with pytest.raises(ValueError, match="geo_idx must lie in"):
            cl.fit(data, channels=["A", "B"])

    def test_float_geo_idx_rejected(self):
        data = _tiny_panel()
        data["geo_idx"] = data["geo_idx"].astype(float)
        with pytest.raises(ValueError, match="integer-typed"):
            cl.fit(data, channels=["A", "B"])

    def test_non_finite_spend_and_y_rejected(self):
        data = _tiny_panel()
        data["spend"] = data["spend"].copy()
        data["spend"][0, 0] = np.nan
        with pytest.raises(ValueError, match="spend contains non-finite"):
            cl.fit(data, channels=["A", "B"])
        data = _tiny_panel()
        data["y"] = data["y"].copy()
        data["y"][3] = np.inf
        with pytest.raises(ValueError, match="y contains non-finite"):
            cl.fit(data, channels=["A", "B"])

    def test_row_count_mismatch_rejected(self):
        data = _tiny_panel()
        data["y"] = data["y"][:-1]
        with pytest.raises(ValueError, match="row counts disagree"):
            cl.fit(data, channels=["A", "B"])

    def test_no_evidence_at_all_rejected(self):
        with pytest.raises(ValueError, match="fit needs evidence"):
            cl.fit({"n_geo": 0}, channels=["A", "B"])
        with pytest.raises(ValueError, match="fit needs evidence"):
            cl.fit(
                {
                    "spend": np.zeros((0, 2)),
                    "geo_idx": np.zeros(0, dtype=int),
                    "y": np.zeros(0),
                    "n_geo": 0,
                },
                channels=["A", "B"],
            )

    def test_bad_summaries_rejected_before_mcmc(self):
        center = np.array([0.8, 0.8])
        good = _summary(center, 0, 0.4, 1.0)
        bad_shape = dict(good, spend_test=np.array([0.8, 0.8, 0.8]))
        with pytest.raises(ValueError, match="shape"):
            cl.fit({"summaries": [bad_shape], "n_geo": 0}, channels=["A", "B"])
        bad_se = dict(good, se=0.0)
        with pytest.raises(ValueError, match="se must be positive"):
            cl.fit({"summaries": [bad_se], "n_geo": 0}, channels=["A", "B"])
        missing = {k: v for k, v in good.items() if k != "lift"}
        with pytest.raises(ValueError, match="missing keys"):
            cl.fit({"summaries": [missing], "n_geo": 0}, channels=["A", "B"])


# ── F2: once-only JAX platform guard ──────────────────────────────────────────


def test_platform_guard_is_once_only(monkeypatch):
    # the sentinel flips on first call and short-circuits every later call —
    # even if the env asks for a different platform afterwards.
    monkeypatch.setattr(model, "_PLATFORM_SET", False)
    monkeypatch.setenv("MMM_CL_JAX_PLATFORM", "keep")
    model._ensure_cpu()  # "keep" -> config untouched, sentinel set
    assert model._PLATFORM_SET is True
    monkeypatch.setenv("MMM_CL_JAX_PLATFORM", "cpu")
    model._ensure_cpu()  # no-op: sentinel already set
    assert model._PLATFORM_SET is True


# ── F3/F4: one-Thompson-pass planning ─────────────────────────────────────────


class TestPlanFromPosterior:
    def test_readouts_share_one_sample(self):
        world = cl.make_world(seed=0)
        post = _fake_posterior(world)
        plan = cl.plan_from_posterior(post, 3.2, 5.0, q=16, seed=0)
        # the funded set and the regret consensus describe THE SAME vector
        np.testing.assert_array_equal(plan.consensus, plan.recommendation)
        assert plan.recommendation.sum() == pytest.approx(3.2, abs=1e-6)
        # the funding line was evaluated exactly at the recommendation, on the
        # same posterior draws the Thompson pass used
        for t, d in enumerate(plan.draws[:3]):
            _, gr = planner._surface_fns(post.draw_params(int(d)))
            np.testing.assert_allclose(
                plan.mroas_draws[t], 5.0 * gr(plan.recommendation), rtol=1e-6
            )
        assert plan.e_regret >= 0.0
        assert plan.allocs.shape == (16, 4)
        assert plan.alloc_sd.shape == (4,)

    def test_to_dict_is_json_safe(self):
        world = cl.make_world(seed=1)
        post = _fake_posterior(world)
        plan = cl.plan_from_posterior(post, 3.2, 5.0, q=8, seed=0)
        payload = json.loads(json.dumps(plan.to_dict()))
        assert payload["channels"] == world.channels
        assert len(payload["recommendation"]) == 4
        assert payload["n_draws"] == 8
        assert isinstance(payload["funded"][0], bool)

    def test_learning_state_plan_delegates(self):
        world = cl.make_world(seed=0)
        post = _fake_posterior(world)
        state = cl.LearningState(
            channels=world.channels, center=np.full(4, 0.8), B=3.2, value=5.0
        )
        state.posterior = post
        plan = state.plan(q=8, seed=0)
        direct = cl.plan_from_posterior(post, 3.2, 5.0, q=8, seed=0, mode="fixed")
        np.testing.assert_array_equal(plan.recommendation, direct.recommendation)

    def test_expected_regret_warm_start_never_below_consensus_profit(self):
        world = cl.make_world(seed=0)
        post = _fake_posterior(world)
        e_regret, consensus, alloc_sd, profit_sd = cl.expected_regret(
            post, 3.2, 5.0, q=12, seed=2
        )
        assert e_regret >= 0.0
        assert consensus.sum() == pytest.approx(3.2, abs=1e-6)
        assert alloc_sd.shape == (4,) and np.isfinite(profit_sd)


# ── F5: recenter floor ────────────────────────────────────────────────────────


class TestRecenterFloor:
    def test_recenter_clamps_and_warns(self):
        state = cl.LearningState(
            channels=["A", "B", "C", "D"], center=np.full(4, 0.8), B=3.2, value=5.0
        )
        floor = 0.05 * 3.2 / 4  # min_frac * B / K
        with pytest.warns(UserWarning, match="CCD floor"):
            state.recenter(np.array([1.6, 1.6, 0.0, 0.001]))
        assert np.all(state.center >= floor - 1e-12)
        np.testing.assert_allclose(state.center[:2], [1.6, 1.6])

    def test_min_frac_zero_restores_old_behavior(self):
        state = cl.LearningState(
            channels=["A", "B"], center=np.full(2, 0.8), B=2.0, value=5.0
        )
        state.recenter(np.array([2.0, 0.0]), min_frac=0.0)
        np.testing.assert_array_equal(state.center, [2.0, 0.0])


# ── F6: geo-identity guard ────────────────────────────────────────────────────


class TestGeoIdentityGuard:
    def _wave(self, geo_ids=None, seed=0):
        data = _tiny_panel(seed=seed)
        if geo_ids is not None:
            data["geo_ids"] = geo_ids
        return data

    def test_mismatched_geo_ids_raise_with_set_difference(self):
        ids0 = [f"g{i}" for i in range(6)]
        ids1 = ids0[:-1] + ["g_new"]
        state = cl.LearningState(
            channels=["A", "B"], center=np.full(2, 0.8), B=2.0, value=5.0
        )
        state.ingest(self._wave(ids0))
        assert state.geo_ids == ids0
        with pytest.raises(ValueError, match="dropped=\\['g5'\\].*added=\\['g_new'\\]"):
            state.ingest(self._wave(ids1, seed=1))

    def test_reordered_geo_ids_raise(self):
        ids0 = [f"g{i}" for i in range(6)]
        state = cl.LearningState(
            channels=["A", "B"], center=np.full(2, 0.8), B=2.0, value=5.0
        )
        state.ingest(self._wave(ids0))
        with pytest.raises(ValueError, match="reordered"):
            state.ingest(self._wave(list(reversed(ids0)), seed=1))

    def test_count_only_check_retained_without_ids(self):
        state = cl.LearningState(
            channels=["A", "B"], center=np.full(2, 0.8), B=2.0, value=5.0
        )
        state.ingest(self._wave())
        state.ingest(self._wave(seed=1))  # same n_geo, no ids -> fine
        bad = _tiny_panel(n_geo=5, seed=2)
        with pytest.raises(ValueError, match="stable geo set"):
            state.ingest(bad)


# ── F7: knowledge-gradient fantasy-noise default ──────────────────────────────


def test_kg_noise_defaults_to_posterior_sigma():
    world = cl.make_world(seed=0)
    post = _fake_posterior(world)
    assert planner._default_noise(post) == pytest.approx(
        float(np.mean(post.samples["sigma"]))
    )
    no_sigma = _fake_posterior(world, extra_sigma=False)
    assert planner._default_noise(no_sigma) == pytest.approx(0.6)
    # and the public signature's default is None (resolved inside)
    assert inspect.signature(cl.knowledge_gradient).parameters["noise"].default is None


# ── F8: scaling helpers ───────────────────────────────────────────────────────


class TestScaling:
    def test_round_trip_vector_and_panel(self):
        ref = np.array([200_000.0, 140_000.0, 90_000.0])
        dollars = np.array([100_000.0, 140_000.0, 45_000.0])
        scaled = cl.to_scaled(dollars, ref)
        np.testing.assert_allclose(scaled, [0.5, 1.0, 0.5])
        np.testing.assert_allclose(cl.to_dollars(scaled, ref), dollars)
        panel = np.tile(dollars, (5, 1))
        np.testing.assert_allclose(cl.to_dollars(cl.to_scaled(panel, ref), ref), panel)

    def test_validation(self):
        with pytest.raises(ValueError, match="strictly positive"):
            cl.to_scaled(np.ones(2), np.array([1.0, 0.0]))
        with pytest.raises(ValueError, match="1-D"):
            cl.to_scaled(np.ones(2), np.ones((2, 2)))
        with pytest.raises(ValueError, match="last-axis"):
            cl.to_scaled(np.ones(3), np.ones(2))


# ── §2.3: experiments -> summaries converter ──────────────────────────────────


class TestExperimentsToSummaries:
    CHANNELS = ["TV", "Search"]
    REF = np.array([100_000.0, 50_000.0])
    CENTER = np.array([1.0, 1.0])

    def _convert(self, experiments, channels=None):
        return cl.experiments_to_summaries(
            experiments,
            channels=channels or self.CHANNELS,
            spend_ref=self.REF,
            center_scaled=self.CENTER,
        )

    def _base_exp(self, **over):
        exp = {
            "id": "e1",
            "status": "completed",
            "channel": "Search",
            "estimand": "contribution",
            "value": 500.0,
            "se": 100.0,
            "start_date": "2026-01-01",
            "end_date": "2026-01-29",  # 28 days -> 4 weekly periods
            "design": {},
            "readout": {"spend_per_period": 10_000.0, "n_treated_units": 5},
        }
        exp.update(over)
        return exp

    def test_contribution_passthrough_and_spend_mapping(self):
        summaries, skipped = self._convert([self._base_exp()])
        assert skipped == []
        (s,) = summaries
        assert s["lift"] == pytest.approx(500.0)
        assert s["se"] == pytest.approx(100.0)
        assert s["scale"] == pytest.approx(5 * 4)  # n_units * n_periods
        # Search shifted by +10k/period at 50k per scaled unit -> +0.2
        np.testing.assert_allclose(s["spend_test"], [1.0, 1.2])
        np.testing.assert_allclose(s["spend_base"], [1.0, 1.0])
        assert s["experiment_id"] == "e1" and s["channel"] == "Search"

    def test_roas_multiplies_up_to_total_lift(self):
        exp = self._base_exp(estimand="roas", value=2.0, se=0.5)
        exp["readout"] = dict(exp["readout"], estimand="roas")
        summaries, skipped = self._convert([exp])
        assert skipped == []
        (s,) = summaries
        total_delta = 10_000.0 * 5 * 4
        assert s["lift"] == pytest.approx(2.0 * total_delta)
        assert s["se"] == pytest.approx(0.5 * total_delta)

    def test_holdout_sign_restored_from_design(self):
        exp = self._base_exp()
        exp["readout"] = {"n_treated_units": 5}  # no spend_per_period
        exp["design"] = {
            "design_type": "Randomized matched-pair geo lift — holdout",
            "weekly_spend_delta": 10_000.0,  # treated-cell TOTAL, stored abs()
        }
        summaries, skipped = self._convert([exp])
        assert skipped == []
        (s,) = summaries
        # holdout: the test cell SPENT LESS than the baseline. The design's
        # weekly_spend_delta is the TOTAL across the 5 treated geos, so each
        # geo-period moves by -10k/5 = -2k at 50k per scaled unit -> -0.04
        assert s["spend_test"][1] == pytest.approx(1.0 - 0.04)
        # and a positive measured contribution on a holdout means cutting
        # spend LOST KPI -> the imported lift is NEGATIVE (R(test) < R(base))
        assert s["lift"] == pytest.approx(-500.0)
        assert s["se"] == pytest.approx(100.0)  # se stays positive

    def test_roas_holdout_lift_negative_scaleup_positive(self):
        # signed readout.spend_per_period path (per treated unit)
        holdout = self._base_exp(estimand="roas", value=2.0, se=0.5)
        holdout["readout"] = {
            "estimand": "roas",
            "spend_per_period": -10_000.0,  # holdouts record it NEGATIVE
            "n_treated_units": 5,
        }
        scaleup = self._base_exp(id="e2", estimand="roas", value=2.0, se=0.5)
        scaleup["readout"] = {
            "estimand": "roas",
            "spend_per_period": 10_000.0,
            "n_treated_units": 5,
        }
        summaries, skipped = self._convert([holdout, scaleup])
        assert skipped == []
        down, up = summaries
        total = 10_000.0 * 5 * 4  # per-unit delta x units x periods
        assert down["lift"] == pytest.approx(-2.0 * total)  # NEGATIVE lift
        assert down["se"] == pytest.approx(0.5 * total)
        assert down["spend_test"][1] == pytest.approx(1.0 - 0.2)
        assert up["lift"] == pytest.approx(2.0 * total)  # positive lift
        assert up["se"] == pytest.approx(0.5 * total)
        assert up["spend_test"][1] == pytest.approx(1.0 + 0.2)

    def test_design_delta_is_cell_total_divided_across_units(self):
        # design.weekly_spend_delta sums over ALL treated geos
        # (planning/design.py), so the per-geo shift is delta/(n_units*ref)
        exp = self._base_exp(estimand="roas", value=2.0, se=0.5)
        exp["readout"] = {"estimand": "roas", "n_treated_units": 5}
        exp["design"] = {
            "design_type": "matched-market scale-up",
            "weekly_spend_delta": 10_000.0,
        }
        summaries, skipped = self._convert([exp])
        assert skipped == []
        (s,) = summaries
        assert s["spend_test"][1] == pytest.approx(1.0 + 10_000.0 / (5 * 50_000.0))
        # the TOTAL spend delta stays the cell total x periods
        total = 10_000.0 * 4
        assert s["lift"] == pytest.approx(2.0 * total)
        assert s["se"] == pytest.approx(0.5 * total)

    def test_mroas_skipped_with_pinned_reason(self):
        exp = self._base_exp(estimand="mroas")
        exp["readout"] = dict(exp["readout"], estimand="mroas")
        summaries, skipped = self._convert([exp])
        assert summaries == []
        assert skipped == [
            {"id": "e1", "reason": "mroas readouts are slopes, not lifts"}
        ]

    def test_skip_reasons(self):
        exps = [
            self._base_exp(id="a", status="planned"),
            self._base_exp(id="b", value=None, se=None, readout={}),
            self._base_exp(id="c", channel="Radio"),
            self._base_exp(id="d", readout={}, design={}),  # no spend anywhere
            self._base_exp(
                id="e", start_date=None, end_date=None, design={}
            ),  # no window
        ]
        summaries, skipped = self._convert(exps)
        assert summaries == []
        reasons = {s["id"]: s["reason"] for s in skipped}
        assert "not completed/calibrated" in reasons["a"]
        assert "missing value/se" in reasons["b"]
        assert "not in program channels" in reasons["c"]
        assert "no spend level" in reasons["d"]
        assert "no test window" in reasons["e"]

    def test_n_periods_falls_back_to_design_duration(self):
        exp = self._base_exp(start_date=None, end_date=None)
        exp["design"] = {"duration": 8}
        summaries, _ = self._convert([exp])
        assert summaries[0]["scale"] == pytest.approx(5 * 8)

    def test_n_units_falls_back_to_treatment_geos(self):
        exp = self._base_exp()
        exp["readout"] = {"spend_per_period": 10_000.0}  # no n_treated_units
        exp["design"] = {"treatment_geos": ["g1", "g2", "g3"]}
        summaries, _ = self._convert([exp])
        assert summaries[0]["scale"] == pytest.approx(3 * 4)

    def test_case_insensitive_channel_match(self):
        exp = self._base_exp(channel="search")
        summaries, skipped = self._convert([exp])
        assert skipped == [] and summaries[0]["channel"] == "Search"

    def test_exact_match_wins_and_case_collision_skips(self):
        # channels differing only by case: an exact match attributes correctly;
        # an inexact name that collides case-insensitively skips loudly instead
        # of silently landing on the LAST colliding channel.
        channels = ["TV", "tv"]
        ref = np.array([100_000.0, 50_000.0])
        exact = self._base_exp(channel="TV")
        ambiguous = self._base_exp(id="e2", channel="Tv")
        summaries, skipped = cl.experiments_to_summaries(
            [exact, ambiguous],
            channels=channels,
            spend_ref=ref,
            center_scaled=np.ones(2),
        )
        assert len(summaries) == 1
        assert summaries[0]["channel"] == "TV"  # index 0, not the last match
        assert summaries[0]["spend_test"][0] == pytest.approx(1.0 + 0.1)
        assert summaries[0]["spend_test"][1] == pytest.approx(1.0)
        (sk,) = skipped
        assert sk["id"] == "e2" and "case-insensitively" in sk["reason"]

    def test_subchannel_matches_arm_and_split_parent_blocks_channel_level(self):
        spec = cl.expand_arms(["TV", "Search"], {"Search": ["Brand", "NonBrand"]})
        channels = spec.channels  # ["TV", "Search │ Brand", "Search │ NonBrand"]
        ref = np.array([100_000.0, 25_000.0, 25_000.0])
        center = np.ones(3)
        arm_exp = self._base_exp(subchannel="Brand")
        parent_exp = self._base_exp(id="e2")  # channel-level on split parent
        summaries, skipped = cl.experiments_to_summaries(
            [arm_exp, parent_exp],
            channels=channels,
            spend_ref=ref,
            center_scaled=center,
        )
        assert len(summaries) == 1
        assert summaries[0]["channel"] == f"Search{cl.ARM_SEP}Brand"
        assert skipped == [
            {"id": "e2", "reason": "channel-level readout on a split parent"}
        ]


# ── §2.4: persistence ─────────────────────────────────────────────────────────


class TestSerialize:
    def test_posterior_payload_round_trip_is_exact(self):
        world = cl.make_world(seed=0)
        post = _fake_posterior(world, n=50)
        post.pair_signs = dict(cl.PAIR_SIGNS_EXAMPLE)
        post.spend_ref = np.array([1.0, 2.0, 3.0, 4.0])
        post.diagnostics = {"max_rhat": 1.01, "min_ess": 250.0}
        payload = json.loads(json.dumps(cl.posterior_to_payload(post)))
        back = cl.posterior_from_payload(payload)
        assert back.channels == post.channels
        assert back.pairs == post.pairs
        assert back.pair_signs == post.pair_signs
        assert back.activation == post.activation
        np.testing.assert_array_equal(back.spend_ref, post.spend_ref)
        for key in post.samples:
            np.testing.assert_array_equal(back.samples[key], post.samples[key])

    def test_state_npz_round_trip_is_exact(self, tmp_path):
        world = cl.make_world(seed=0)
        post = _fake_posterior(world, n=40)
        center = np.full(4, 0.8)
        state = cl.LearningState(
            channels=world.channels,
            center=center,
            B=3.2,
            value=5.0,
            pair_signs=dict(cl.PAIR_SIGNS_EXAMPLE),
            cap=2.0,
            spend_ref=np.array([1.0, 2.0, 3.0, 4.0]),
        )
        data = cl.simulate_panel(
            world, center, n_geo=10, t_pre=2, t_test=2, delta=0.6, seed=1
        )
        data["geo_ids"] = [f"g{i}" for i in range(10)]
        state.ingest(data)
        state.ingest_summaries(
            [_summary(center, 0, 0.3, 12.0), _summary(center, 1, -0.3, -8.0)]
        )
        state.posterior = post
        state.history.append(
            cl.WaveRecord(
                wave=0,
                n_rows=40,
                e_regret=0.5,
                enbs=0.2,
                stop=False,
                recommendation=[0.8] * 4,
                funded=[True] * 4,
                mroas_mean=[1.5] * 4,
                prob_above_line=[0.9] * 4,
                profit_gap=0.1,
                profit_gap_rel=0.02,
                max_rhat=1.01,
                n_summaries=2,
            )
        )
        path = tmp_path / "state.npz"
        out = cl.state_to_npz(state, path)
        assert out == str(path)
        back = cl.state_from_npz(path)

        assert back.channels == state.channels
        assert back.pairs == state.pairs
        assert back.pair_signs == state.pair_signs
        assert (back.B, back.value, back.mode, back.cap) == (3.2, 5.0, "fixed", 2.0)
        assert back.geo_ids == state.geo_ids
        np.testing.assert_array_equal(back.center, state.center)
        np.testing.assert_array_equal(back.spend_ref, state.spend_ref)
        np.testing.assert_array_equal(back.data["spend"], state.data["spend"])
        np.testing.assert_array_equal(back.data["geo_idx"], state.data["geo_idx"])
        np.testing.assert_array_equal(back.data["y"], state.data["y"])
        assert back.data["n_geo"] == state.data["n_geo"]
        assert len(back.summaries) == 2
        for s0, s1 in zip(state.summaries, back.summaries):
            np.testing.assert_array_equal(s0["spend_test"], s1["spend_test"])
            np.testing.assert_array_equal(s0["spend_base"], s1["spend_base"])
            assert (s0["lift"], s0["se"], s0["scale"]) == (
                s1["lift"],
                s1["se"],
                s1["scale"],
            )
        for key in post.samples:
            np.testing.assert_array_equal(
                back.posterior.samples[key], post.samples[key]
            )
        assert back.history[0] == state.history[0]

    def test_state_without_posterior_or_data(self, tmp_path):
        state = cl.LearningState(
            channels=["A", "B"], center=np.full(2, 0.8), B=2.0, value=5.0
        )
        state.ingest_summaries([_summary(np.full(2, 0.8), 0, 0.3, 5.0)])
        path = tmp_path / "s.npz"
        cl.state_to_npz(state, path)
        back = cl.state_from_npz(path)
        assert back.posterior is None and back.data is None
        assert len(back.summaries) == 1


# ── §2.5: arms + grouped budgets ──────────────────────────────────────────────


class TestArms:
    def test_expand_arms_structure(self):
        spec = cl.expand_arms(
            ["TV", "Search", "Social"], {"Search": ["Brand", "NonBrand"]}
        )
        assert spec.channels == [
            "TV",
            f"Search{cl.ARM_SEP}Brand",
            f"Search{cl.ARM_SEP}NonBrand",
            "Social",
        ]
        assert spec.parents == ["TV", "Search", "Search", "Social"]
        assert spec.groups == {"TV": [0], "Search": [1, 2], "Social": [3]}
        assert spec.split_parents() == ["Search"]

    def test_expand_arms_validation(self):
        with pytest.raises(ValueError, match="unknown channels"):
            cl.expand_arms(["TV"], {"Radio": ["A"]})
        with pytest.raises(ValueError, match="duplicate arm names"):
            cl.expand_arms(["TV"], {"TV": ["A", "A"]})

    def test_pair_helpers_and_default_signs(self):
        spec = cl.expand_arms(["TV", "Search"], {"Search": ["Brand", "NonBrand"]})
        assert cl.within_parent_pairs(spec) == [(1, 2)]
        assert cl.cross_parent_pairs(spec) == [(0, 1), (0, 2)]
        assert cl.cross_parent_pairs(spec, [("TV", "Search")]) == [(0, 1), (0, 2)]
        signs = cl.default_arm_pair_signs(spec)
        assert signs[(1, 2)] == "neg"  # siblings substitute
        assert signs[(0, 1)] == "weak" and signs[(0, 2)] == "weak"
        over = cl.default_arm_pair_signs(spec, base={(0, 1): "pos"})
        assert over[(0, 1)] == "pos" and over[(1, 2)] == "neg"

    def test_grouped_budget_allocation(self):
        # symmetric concave channels: the group constraint binds, the rest of
        # the budget flows to the ungrouped channel.
        params = {
            "beta": np.ones(3),
            "kappa": np.ones(3),
            "alpha": np.ones(3),
            "gamma": np.zeros((3, 3)),
        }
        alloc, _ = cl.allocate_under_sample(
            params,
            B=3.0,
            value=5.0,
            mode="fixed",
            group_budgets=[([0, 1], 1.0)],
            n_starts=4,
        )
        assert alloc[:2].sum() == pytest.approx(1.0, abs=1e-6)
        assert alloc.sum() == pytest.approx(3.0, abs=1e-6)
        # groups covering ALL channels with sum == B (global constraint dropped)
        alloc2, _ = cl.allocate_under_sample(
            params,
            B=3.0,
            value=5.0,
            mode="fixed",
            group_budgets=[([0, 1], 1.0), ([2], 2.0)],
            n_starts=4,
        )
        assert alloc2[:2].sum() == pytest.approx(1.0, abs=1e-6)
        assert alloc2[2] == pytest.approx(2.0, abs=1e-6)

    def test_grouped_budget_validation(self):
        params = {
            "beta": np.ones(3),
            "kappa": np.ones(3),
            "alpha": np.ones(3),
            "gamma": np.zeros((3, 3)),
        }
        with pytest.raises(ValueError, match="overlap"):
            cl.allocate_under_sample(
                params, B=3.0, value=5.0, group_budgets=[([0, 1], 1.0), ([1], 0.5)]
            )
        with pytest.raises(ValueError, match="out of range"):
            cl.allocate_under_sample(
                params, B=3.0, value=5.0, group_budgets=[([3], 1.0)]
            )
        with pytest.raises(ValueError, match="sum to"):
            cl.allocate_under_sample(
                params, B=3.0, value=5.0, group_budgets=[([0, 1], 4.0)]
            )
        with pytest.raises(ValueError, match="ungrouped"):
            # groups cover all channels but leave budget nobody can absorb
            cl.allocate_under_sample(
                params,
                B=3.0,
                value=5.0,
                group_budgets=[([0, 1, 2], 2.0)],
            )

    def test_plan_from_posterior_respects_group_budgets(self):
        world = cl.make_world(seed=0)
        post = _fake_posterior(world)
        plan = cl.plan_from_posterior(
            post, 3.2, 5.0, q=8, seed=0, group_budgets=[([0, 1], 1.5)]
        )
        assert plan.recommendation[:2].sum() == pytest.approx(1.5, abs=1e-4)
        assert plan.recommendation.sum() == pytest.approx(3.2, abs=1e-6)


# ── §2.2 loop-side summary plumbing (fast) ────────────────────────────────────


class TestLearningStateSummaries:
    def test_ingest_summaries_validates_eagerly(self):
        state = cl.LearningState(
            channels=["A", "B"], center=np.full(2, 0.8), B=2.0, value=5.0
        )
        with pytest.raises(ValueError, match="se must be positive"):
            state.ingest_summaries([_summary(np.full(2, 0.8), 0, 0.3, 5.0, se=-1.0)])
        state.ingest_summaries([_summary(np.full(2, 0.8), 0, 0.3, 5.0)])
        assert len(state.summaries) == 1

    def test_fit_requires_some_evidence(self):
        state = cl.LearningState(
            channels=["A", "B"], center=np.full(2, 0.8), B=2.0, value=5.0
        )
        with pytest.raises(RuntimeError, match="wave or some summaries"):
            state.fit()

    def test_wave_record_carries_n_summaries(self):
        rec = cl.WaveRecord(
            wave=0,
            n_rows=10,
            e_regret=0.1,
            enbs=0.0,
            stop=True,
            recommendation=[1.0],
            funded=[True],
            mroas_mean=[1.2],
            prob_above_line=[0.8],
            profit_gap=0.0,
            profit_gap_rel=0.0,
            max_rhat=None,
        )
        assert rec.n_summaries == 0  # back-compat default


# ── [29]: data-driven prior scaling (natural-unit y) ──────────────────────────


class TestPriorScaling:
    def test_unit_reproduces_o1_priors(self):
        assert model._resolve_prior_scaling("unit", np.array([5.0, 9.0]), []) == (
            0.0,
            1.0,
        )

    def test_auto_panel_uses_y_mean_and_std_decade(self):
        y = np.array([40_000.0, 60_000.0, 50_000.0, 30_000.0])
        y_loc, y_scale = model._resolve_prior_scaling("auto", y, [])
        assert y_loc == pytest.approx(float(np.mean(y)))
        # std(y) ~ 11.2k -> the nearest decade is 1e4
        assert y_scale == pytest.approx(1e4)

    def test_auto_is_a_no_op_for_o1_data(self):
        # the priors were calibrated to O(1) synthetic worlds — auto must keep
        # them EXACTLY there (decade quantization), not jitter a well-tuned
        # geometry by a small std multiple
        rng = np.random.default_rng(0)
        y = rng.normal(4.0, 1.2, 500)
        y_loc, y_scale = model._resolve_prior_scaling("auto", y, [])
        assert y_scale == 1.0
        assert y_loc == pytest.approx(4.0, abs=0.2)

    def test_auto_summaries_only_uses_lift_magnitudes(self):
        summaries = [
            {"lift": 400_000.0, "scale": 20.0},  # 20k per geo-period
            {"lift": -100_000.0, "scale": 10.0},  # 10k per geo-period
        ]
        y_loc, y_scale = model._resolve_prior_scaling("auto", None, summaries)
        # max(mean, std) of |lift|/scale ~ 15k -> the nearest decade is 1e4
        assert y_loc == 0.0
        assert y_scale == pytest.approx(1e4)

    def test_constant_y_is_floored(self):
        _, y_scale = model._resolve_prior_scaling("auto", np.full(8, 3.0), [])
        assert y_scale == pytest.approx(1e-6)

    def test_invalid_mode_rejected_before_mcmc(self):
        with pytest.raises(ValueError, match="prior_scaling"):
            model._resolve_prior_scaling("bogus", np.array([1.0]), [])
        with pytest.raises(ValueError, match="prior_scaling"):
            cl.fit(_tiny_panel(), channels=["A", "B"], prior_scaling="bogus")

    def test_prior_sites_scale_with_y_scale(self):
        """The intercept/noise/effect prior sites all live on the y scale."""
        import jax
        from numpyro.infer import Predictive

        data = _tiny_panel()

        def draw(y_scale, y_loc=0.0):
            pred = Predictive(model.model, num_samples=400)
            return pred(
                jax.random.PRNGKey(0),
                data["spend"],
                data["geo_idx"],
                data["n_geo"],
                None,
                channels=["A", "B"],
                pairs=[(0, 1)],
                pair_signs={},
                y_loc=y_loc,
                y_scale=y_scale,
            )

        unit = draw(1.0)  # defaults: the original O(1) priors
        big = draw(1e4, y_loc=5e4)  # a revenue-scale KPI
        assert float(np.mean(np.asarray(big["A"]))) == pytest.approx(5e4, rel=0.2)
        ratio_a = float(np.std(np.asarray(big["A"])) / np.std(np.asarray(unit["A"])))
        assert ratio_a == pytest.approx(1e4, rel=0.25)
        for site in ("sigma", "sigma_a", "beta"):
            r = float(np.mean(np.asarray(big[site])) / np.mean(np.asarray(unit[site])))
            assert r == pytest.approx(1e4, rel=0.25), site
        # the interaction prior widens with y too (effects live on the y scale)
        r = float(
            np.std(np.asarray(big["gamma_A_B"])) / np.std(np.asarray(unit["gamma_A_B"]))
        )
        assert r == pytest.approx(1e4, rel=0.25)


# ── [25]: knowledge-gradient guards on summaries-only evidence ────────────────


class TestKnowledgeGradientGuards:
    def _summaries_only_posterior(self, world, n=50, seed=0):
        """A posterior with surface sites only — no A/sigma_a/a_geo/sigma."""
        rng = np.random.default_rng(seed)
        k = world.n_channels
        s = {
            "beta": np.abs(world.beta + 0.1 * rng.standard_normal((n, k))),
            "kappa": np.abs(world.kappa + 0.05 * rng.standard_normal((n, k))),
            "alpha": np.clip(world.alpha + 0.1 * rng.standard_normal((n, k)), 0.5, 5),
        }
        for idx, (i, j) in enumerate(world.pairs):
            s[model.pair_name(world.channels, (i, j))] = world.gamma_pairs[
                idx
            ] + 0.1 * rng.standard_normal(n)
        return cl.Posterior(samples=s, channels=world.channels, pairs=world.pairs)

    def test_kg_raises_clearly_on_summaries_only_posterior(self):
        world = cl.make_world(seed=0)
        post = self._summaries_only_posterior(world)
        candidate = cl.central_composite(np.full(4, 0.8), 0.6, [])
        with pytest.raises(ValueError, match="panel-fitted posterior"):
            cl.knowledge_gradient(
                post,
                candidate,
                lambda *args: post,
                B=3.2,
                value=5.0,
                n_fantasy=1,
                q=8,
                noise=0.6,
            )

    def test_refit_fn_raises_clearly_without_a_panel(self):
        with pytest.raises(ValueError, match="panel base dataset"):
            cl.refit_fn_from_data(
                {"summaries": [_summary(np.full(2, 0.8), 0, 0.3, 5.0)], "n_geo": 0},
                channels=["A", "B"],
            )


# ── [30]: budget-relative feasibility tolerances ──────────────────────────────


def test_group_budget_tolerances_are_relative_to_budget_scale():
    """Dollar-scale budgets: an exactly-intended partition carries ~1 ulp of fp
    error (>> 1e-9), which the old absolute tolerance spuriously rejected."""
    params = {
        "beta": np.ones(2),
        "kappa": np.ones(2),
        "alpha": np.ones(2),
        "gamma": np.zeros((2, 2)),
    }
    B = 5.6e9
    b1 = B / 3
    b2 = B - b1 + 1e-6  # ~1 ulp at this scale: sum > B + 1e-9, << 1e-9 * B
    assert b1 + b2 > B + 1e-9  # the fp error the absolute tolerance tripped on
    alloc, _ = cl.allocate_under_sample(
        params,
        B=B,
        value=5.0,
        mode="fixed",
        group_budgets=[([0], b1), ([1], b2)],
        n_starts=2,
    )
    assert alloc[0] == pytest.approx(b1, rel=1e-6)
    assert alloc[1] == pytest.approx(b2, rel=1e-6)


# ── [32]: pair-sign key orientation ───────────────────────────────────────────


class TestPairSignNormalization:
    def test_reversed_key_is_normalized(self):
        assert model.normalize_pair_signs({(1, 0): "neg"}) == {(0, 1): "neg"}

    def test_duplicates_merge_and_conflicts_raise(self):
        assert model.normalize_pair_signs({(1, 0): "neg", (0, 1): "neg"}) == {
            (0, 1): "neg"
        }
        with pytest.raises(ValueError, match="conflicting"):
            model.normalize_pair_signs({(1, 0): "neg", (0, 1): "pos"})
        with pytest.raises(ValueError, match="itself"):
            model.normalize_pair_signs({(1, 1): "neg"})

    def test_reversed_key_reaches_the_model_prior(self):
        # a "(1, 0)" entry must constrain gamma_A_B — previously it validated
        # but was silently ignored (the pair fell back to the "weak" prior).
        import jax
        from numpyro.infer import Predictive

        data = _tiny_panel()
        pred = Predictive(model.model, num_samples=100)
        out = pred(
            jax.random.PRNGKey(0),
            data["spend"],
            data["geo_idx"],
            data["n_geo"],
            None,
            channels=["A", "B"],
            pairs=[(0, 1)],
            pair_signs={(1, 0): "neg"},
        )
        assert np.all(np.asarray(out["gamma_A_B"]) <= 0.0)


# ── [33]: R̂ site list follows the fitted activation family ────────────────────


class TestDiagnosticsShapeSites:
    class _FakeMCMC:
        def __init__(self, grouped):
            self._grouped = grouped

        def get_samples(self, group_by_chain=False):
            return self._grouped

    def test_rhat_covers_the_fitted_familys_shape_sites(self):
        rng = np.random.default_rng(0)
        mixed = rng.standard_normal((2, 200, 2))  # healthy chains
        split = np.stack(  # chains far apart -> R-hat >> 1.1
            [rng.standard_normal((200, 2)), rng.standard_normal((200, 2)) + 8.0]
        )
        grouped = {
            "beta": mixed,
            "sigma": mixed[..., 0],
            "kappa1": split,
            "kappa2": mixed,
            "alpha1": mixed,
            "alpha2": mixed,
            "w": mixed,
        }
        fake = self._FakeMCMC(grouped)
        hill = model._diagnostics(fake, "hill")  # kappa/alpha absent -> skipped
        mixture = model._diagnostics(fake, "hill_mixture")
        assert hill["max_rhat"] is not None and hill["max_rhat"] < 1.1
        # the unmixed hill_mixture shape site is now inside the gate
        assert mixture["max_rhat"] > 1.5


# ── NegBinomial likelihood + national time effect (opt-in extensions) ─────────


def _nb_posterior(world, n_geo=40, n=200, seed=0):
    """A fake NegBinomial-fit posterior: ``phi`` instead of ``sigma``."""
    rng = np.random.default_rng(seed)
    post = _fake_posterior(world, n_geo=n_geo, n=n, seed=seed, extra_sigma=False)
    post.samples["phi"] = np.abs(rng.normal(10.0, 1.0, n))
    post.likelihood = "negbinomial"
    return post


class TestNegBinomialValidation:
    def test_negative_counts_rejected_with_cuped_hint(self):
        data = _tiny_panel()
        data["y"] = np.round(np.abs(data["y"]) * 10)
        data["y"][0] = -3.0
        with pytest.raises(ValueError, match="cuped_adjust"):
            cl.fit(data, channels=["A", "B"], likelihood="negbinomial")

    def test_non_integer_counts_rejected(self):
        data = _tiny_panel()  # continuous Gaussian y
        with pytest.raises(ValueError, match="integer counts"):
            cl.fit(data, channels=["A", "B"], likelihood="negbinomial")

    def test_valid_counts_pass_and_round_to_int(self):
        data = _tiny_panel()
        data["y"] = np.round(np.abs(data["y"]) * 100) + 5e-7  # int within 1e-6
        _, _, y, _, period_idx, n_period = model._validate_panel(
            data, 2, likelihood="negbinomial"
        )
        np.testing.assert_allclose(y, np.round(y))
        assert period_idx is None and n_period == 0

    def test_unknown_likelihood_rejected(self):
        with pytest.raises(ValueError, match="likelihood"):
            cl.fit(_tiny_panel(), channels=["A", "B"], likelihood="poisson")


class TestPeriodIdxValidation:
    def _panel_with_periods(self, dtype=int):
        data = _tiny_panel()  # 6 geos x 3 weeks, week-major rows
        data["period_idx"] = np.repeat(np.arange(3), 6).astype(dtype)
        return data

    def test_valid_period_idx_derives_n_period(self):
        data = self._panel_with_periods()
        *_, period_idx, n_period = model._validate_panel(data, 2)
        assert n_period == 3
        np.testing.assert_array_equal(period_idx, data["period_idx"])

    def test_float_period_idx_rejected(self):
        data = self._panel_with_periods(dtype=float)
        with pytest.raises(ValueError, match="integer-typed"):
            model._validate_panel(data, 2)

    def test_negative_period_idx_rejected(self):
        data = self._panel_with_periods()
        data["period_idx"][0] = -1
        with pytest.raises(ValueError, match="non-negative"):
            model._validate_panel(data, 2)

    def test_wrong_length_period_idx_rejected(self):
        data = self._panel_with_periods()
        data["period_idx"] = data["period_idx"][:-1]
        with pytest.raises(ValueError, match="period_idx must have shape"):
            model._validate_panel(data, 2)

    def test_national_requires_period_idx(self):
        with pytest.raises(ValueError, match="period_idx"):
            cl.fit(_tiny_panel(), channels=["A", "B"], time_effect="national")

    def test_unknown_time_effect_rejected(self):
        with pytest.raises(ValueError, match="time_effect"):
            cl.fit(_tiny_panel(), channels=["A", "B"], time_effect="weekly")

    def test_summaries_only_national_fit_succeeds(self):
        """[10] Summaries carry no periods and tau cancels in the lift
        difference, so a summaries-only fit with time_effect='national' must
        run (only a NON-empty panel lacking period_idx errors)."""
        center = np.full(2, 0.8)
        data = {
            "n_geo": 0,
            "summaries": [
                _summary(center, 0, 0.3, 5.0),
                _summary(center, 1, 0.3, 1.0),
            ],
        }
        post = cl.fit(
            data,
            channels=["A", "B"],
            time_effect="national",
            num_warmup=20,
            num_samples=20,
            num_chains=1,
            seed=0,
        )
        assert post.time_effect == "national"
        # no panel rows -> no tau/noise sites sampled
        assert "tau" not in post.samples and "sigma" not in post.samples
        # ... and the LearningState wrapper takes the same path
        state = cl.LearningState(
            channels=["A", "B"],
            center=center,
            B=2.0,
            value=5.0,
            time_effect="national",
        )
        state.ingest_summaries([_summary(center, 0, 0.3, 5.0)])
        post2 = state.fit(num_warmup=10, num_samples=10, num_chains=1, seed=0)
        assert post2.time_effect == "national"


class TestNegBinomialGuards:
    def test_default_noise_raises_instead_of_falling_back(self):
        # a NB posterior has no 'sigma' site: the old 0.6 fallback would fire
        # and fantasize Gaussian outcomes around count data — silent wrongness.
        world = cl.make_world(seed=0)
        post = _nb_posterior(world)
        with pytest.raises(NotImplementedError, match="likelihood"):
            planner._default_noise(post)

    def test_low_count_mean_fit_warns_about_link_derivative(self):
        """[3] The planner's marginal readouts differentiate the LATENT
        surface; at low counts sigmoid(mu) << 1 and they overstate the
        count-scale marginal response — fit() must warn loudly."""
        import warnings as _warnings

        data = _tiny_panel()
        data["y"] = np.round(np.abs(data["y"]))  # tiny counts, mean(y) < 20
        assert float(np.mean(data["y"])) < 20.0
        with pytest.warns(UserWarning, match="sigmoid"):
            cl.fit(
                data,
                channels=["A", "B"],
                likelihood="negbinomial",
                num_warmup=10,
                num_samples=10,
                num_chains=1,
                seed=0,
            )
        # high-count data stays silent (no crying wolf on healthy programs)
        data_hi = _tiny_panel()
        data_hi["y"] = np.round(np.abs(data_hi["y"]) * 100.0)
        assert float(np.mean(data_hi["y"])) >= 20.0
        with _warnings.catch_warnings(record=True) as rec:
            _warnings.simplefilter("always")
            cl.fit(
                data_hi,
                channels=["A", "B"],
                likelihood="negbinomial",
                num_warmup=10,
                num_samples=10,
                num_chains=1,
                seed=0,
            )
        assert not [w for w in rec if "sigmoid" in str(w.message)]

    def test_theta_moments_raises(self):
        from mmm_framework.continuous_learning import acquisition

        world = cl.make_world(seed=0)
        post = _nb_posterior(world)
        with pytest.raises(NotImplementedError, match="likelihood"):
            acquisition.theta_moments(post)

    def test_surface_readouts_still_work_for_nb(self):
        # thompson/recommend/mroas/regret read only beta/gamma/shape — a NB
        # posterior must flow through them untouched.
        world = cl.make_world(seed=0)
        post = _nb_posterior(world)
        rec = cl.recommend_allocation(post, B=3.2, value=5.0, q=8, mode="fixed")
        assert rec.sum() == pytest.approx(3.2, abs=0.05)
        mroas_mean, prob_above, _ = cl.marginal_roas(post, rec, value=5.0, q=8)
        assert np.all(np.isfinite(mroas_mean))
        assert np.asarray(prob_above).shape == (4,)
        plan = cl.plan_from_posterior(post, 3.2, 5.0, q=8, seed=0)
        assert np.isfinite(plan.e_regret) and plan.e_regret >= 0.0

    def test_refit_fn_rejects_time_effect(self):
        with pytest.raises(NotImplementedError, match="period identity"):
            cl.refit_fn_from_data(
                _tiny_panel(), channels=["A", "B"], time_effect="national"
            )


class TestPeriodIngestOffsets:
    def _wave(self, t=3, n_geo=6, seed=0, with_period=True):
        data = _tiny_panel(n_geo=n_geo, t=t, seed=seed)
        if with_period:
            data["period_idx"] = np.repeat(np.arange(t), n_geo)
        return data

    def _state(self, time_effect="national"):
        return cl.LearningState(
            channels=["A", "B"],
            center=np.full(2, 0.8),
            B=2.0,
            value=5.0,
            time_effect=time_effect,
        )

    def test_offsets_accumulate_across_waves(self):
        state = self._state()
        state.ingest(self._wave(t=3))
        state.ingest(self._wave(t=2, seed=1))
        pidx = state.data["period_idx"]
        assert pidx.shape == state.data["y"].shape
        # wave 1 periods 0..2, wave 2 shifted by max+1 -> 3..4, no aliasing
        np.testing.assert_array_equal(np.unique(pidx), np.arange(5))
        np.testing.assert_array_equal(pidx[18:], np.repeat([3, 4], 6))

    def test_wave_without_period_raises_on_a_national_program(self):
        state = self._state()
        with pytest.raises(ValueError, match="period_idx"):
            state.ingest(self._wave(with_period=False))

    def test_mixed_presence_raises(self):
        # earlier waves ingested without periods (time_effect flipped later):
        # the accumulated panel has nothing to offset against -> loud error.
        state = self._state(time_effect="none")
        state.ingest(self._wave(with_period=False))
        state.time_effect = "national"
        with pytest.raises(ValueError, match="earlier waves"):
            state.ingest(self._wave(seed=1))

    def test_none_program_ignores_period_idx(self):
        state = self._state(time_effect="none")
        state.ingest(self._wave())  # wave carries period_idx; program opts out
        assert "period_idx" not in state.data


class TestDgpPeriodsAndNoiseFamily:
    def test_period_idx_layout_matches_the_week_major_row_order(self):
        world = cl.make_world(seed=0)
        center = np.full(4, 0.8)
        data = cl.simulate_panel(
            world, center, n_geo=5, t_pre=2, t_test=3, delta=0.6, seed=0
        )
        np.testing.assert_array_equal(data["period_idx"], np.repeat(np.arange(5), 5))
        assert data["period_idx"].shape == data["y"].shape
        wave = cl.simulate_wave(
            world, data["design"], data["a_geo"], t_test=3, center=center, seed=1
        )
        np.testing.assert_array_equal(wave["period_idx"], np.repeat(np.arange(3), 5))

    def test_defaults_byte_identical_and_tau_scale_opt_in(self):
        world = cl.make_world(seed=0)
        center = np.full(4, 0.8)
        base = cl.simulate_panel(
            world, center, n_geo=5, t_pre=2, t_test=3, delta=0.6, seed=0
        )
        again = cl.simulate_panel(
            world,
            center,
            n_geo=5,
            t_pre=2,
            t_test=3,
            delta=0.6,
            seed=0,
            tau_scale=0.0,
            noise_family="normal",
        )
        np.testing.assert_array_equal(base["y"], again["y"])  # same rng stream
        np.testing.assert_array_equal(base["tau_true"], np.zeros(5))
        shocked = cl.simulate_panel(
            world, center, n_geo=5, t_pre=2, t_test=3, delta=0.6, seed=0, tau_scale=1.0
        )
        assert shocked["tau_true"].shape == (5,)
        assert not np.allclose(shocked["y"], base["y"])

    def test_negbinomial_noise_family_draws_counts(self):
        world = cl.make_world(seed=0)
        world.phi_true = 25.0
        center = np.full(4, 0.8)
        data = cl.simulate_panel(
            world,
            center,
            n_geo=6,
            t_pre=2,
            t_test=3,
            delta=0.6,
            seed=0,
            noise_family="negbinomial",
        )
        assert np.all(data["y"] >= 0)
        np.testing.assert_allclose(data["y"], np.round(data["y"]))
        with pytest.raises(ValueError, match="noise_family"):
            cl.simulate_panel(
                world,
                center,
                n_geo=4,
                t_pre=1,
                t_test=1,
                delta=0.6,
                seed=0,
                noise_family="lognormal",
            )


class TestServiceLikelihoodTimeEffect:
    CFG = {"channels": ["TV", "Search"], "budget": 1000.0, "value_per_unit": 5.0}

    def test_new_program_state_carries_the_knobs(self):
        from mmm_framework.continuous_learning import service

        state = service.new_program_state(
            dict(self.CFG, likelihood="negbinomial", time_effect="national")
        )
        assert state.likelihood == "negbinomial"
        assert state.time_effect == "national"
        default = service.new_program_state(dict(self.CFG))
        assert default.likelihood == "normal" and default.time_effect == "none"

    def test_new_program_state_validates_the_knobs(self):
        from mmm_framework.continuous_learning import service

        with pytest.raises(ValueError, match="likelihood"):
            service.new_program_state(dict(self.CFG, likelihood="poisson"))
        with pytest.raises(ValueError, match="time_effect"):
            service.new_program_state(dict(self.CFG, time_effect="weekly"))

    def test_rows_from_csv_period_col_stays_string(self):
        from mmm_framework.continuous_learning import service

        csv_text = "geo,week,TV,Search,y\ng0,1,100,50,4.0\ng1,1,100,50,4.5\n"
        rows = service.rows_from_csv(csv_text, period_col="week")
        assert rows[0]["week"] == "1"  # not coerced to 1.0

    def test_ingest_wave_rows_maps_sorted_period_labels(self):
        from mmm_framework.continuous_learning import service

        state = cl.LearningState(
            channels=["TV", "Search"],
            center=np.full(2, 0.8),
            B=2.0,
            value=5.0,
            time_effect="national",
            spend_ref=np.array([100.0, 50.0]),
        )
        rows = []
        for wk in ("2026-01-12", "2026-01-05"):  # reverse label order on purpose
            for g in ("g0", "g1"):
                rows.append(
                    {"geo": g, "week": wk, "TV": 100.0, "Search": 50.0, "y": 4.0}
                )
        out = service.ingest_wave_rows(state, rows, period_col="week")
        assert out["n_rows"] == 4
        np.testing.assert_array_equal(state.data["period_idx"], [1, 1, 0, 0])

    def test_ingest_wave_rows_requires_the_period_column_on_national(self):
        from mmm_framework.continuous_learning import service

        state = cl.LearningState(
            channels=["TV"],
            center=np.full(1, 0.8),
            B=1.0,
            value=5.0,
            time_effect="national",
        )
        rows = [{"geo": "g0", "TV": 100.0, "y": 4.0}]
        with pytest.raises(ValueError, match="pass period_col"):
            service.ingest_wave_rows(state, rows)
        with pytest.raises(ValueError, match="period column"):
            service.ingest_wave_rows(state, rows, period_col="week")

    def test_ingest_wave_rows_autodetects_the_period_column(self):
        """[9]/[13] A national program whose caller never passes period_col
        still ingests when the rows carry a recognizable week/date/period
        column — with a warning note (explicit period_col stays silent)."""
        from mmm_framework.continuous_learning import service

        def _state():
            return cl.LearningState(
                channels=["TV", "Search"],
                center=np.full(2, 0.8),
                B=2.0,
                value=5.0,
                time_effect="national",
                spend_ref=np.array([100.0, 50.0]),
            )

        rows = []
        for wk in ("2026-01-05", "2026-01-12"):
            for g in ("g0", "g1"):
                rows.append(
                    {"geo": g, "week": wk, "TV": 100.0, "Search": 50.0, "y": 4.0}
                )
        state = _state()
        out = service.ingest_wave_rows(state, rows)  # period_col omitted
        assert out["n_rows"] == 4
        assert any("auto-detected" in w for w in out["warnings"])
        np.testing.assert_array_equal(state.data["period_idx"], [0, 0, 1, 1])
        # explicit period_col: same mapping, no auto-detect note
        state2 = _state()
        out2 = service.ingest_wave_rows(state2, rows, period_col="week")
        assert out2["warnings"] == []
        np.testing.assert_array_equal(
            state2.data["period_idx"], state.data["period_idx"]
        )


class TestSerializeLikelihoodTimeEffect:
    def test_posterior_payload_round_trips_new_fields(self):
        world = cl.make_world(seed=0)
        post = _nb_posterior(world, n=20)
        post.time_effect = "national"
        payload = json.loads(json.dumps(cl.posterior_to_payload(post)))
        back = cl.posterior_from_payload(payload)
        assert back.likelihood == "negbinomial"
        assert back.time_effect == "national"
        np.testing.assert_array_equal(back.samples["phi"], post.samples["phi"])

    def test_old_posterior_payload_defaults(self):
        world = cl.make_world(seed=0)
        payload = cl.posterior_to_payload(_fake_posterior(world, n=10))
        payload.pop("likelihood")
        payload.pop("time_effect")
        back = cl.posterior_from_payload(payload)
        assert back.likelihood == "normal" and back.time_effect == "none"

    def test_state_npz_round_trips_new_fields_and_period_idx(self, tmp_path):
        state = cl.LearningState(
            channels=["A", "B"],
            center=np.full(2, 0.8),
            B=2.0,
            value=5.0,
            likelihood="negbinomial",
            time_effect="national",
        )
        wave = _tiny_panel()
        wave["y"] = np.round(np.abs(wave["y"]) * 10)
        wave["period_idx"] = np.repeat(np.arange(3), 6)
        state.ingest(wave)
        path = tmp_path / "state.npz"
        cl.state_to_npz(state, path)
        back = cl.state_from_npz(path)
        assert back.likelihood == "negbinomial" and back.time_effect == "national"
        np.testing.assert_array_equal(back.data["period_idx"], wave["period_idx"])

    def test_old_state_npz_without_new_keys_loads_with_defaults(self, tmp_path):
        state = cl.LearningState(
            channels=["A", "B"], center=np.full(2, 0.8), B=2.0, value=5.0
        )
        state.ingest(_tiny_panel())
        p_new = tmp_path / "new.npz"
        cl.state_to_npz(state, p_new)
        # simulate a pre-feature file: strip the new meta keys
        with np.load(p_new, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"].item()))
            arrays = {k: z[k] for k in z.files if k != "meta"}
        meta.pop("likelihood", None)
        meta.pop("time_effect", None)
        p_old = tmp_path / "old.npz"
        np.savez_compressed(p_old, meta=json.dumps(meta), **arrays)
        back = cl.state_from_npz(p_old)
        assert back.likelihood == "normal" and back.time_effect == "none"
        assert "period_idx" not in back.data

    def test_plain_state_still_writes_schema_version_1(self, tmp_path):
        """[5] Default-config programs must keep stamping version 1 so OLD
        readers (which cap at 1) can still load them."""
        state = cl.LearningState(
            channels=["A", "B"], center=np.full(2, 0.8), B=2.0, value=5.0
        )
        state.ingest(_tiny_panel())
        world = cl.make_world(seed=0)
        state.posterior = _fake_posterior(world, n=10)
        p = tmp_path / "plain.npz"
        cl.state_to_npz(state, p)
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"].item()))
        assert meta["schema_version"] == 1
        cl.state_from_npz(p)  # and it still round-trips
        # ... same for the JSON posterior payload
        assert (
            cl.posterior_to_payload(_fake_posterior(world, n=10))["schema_version"] == 1
        )

    def test_new_semantics_state_writes_schema_version_2(self, tmp_path):
        """[5] Non-default likelihood/time_effect (or a persisted period_idx)
        stamps version 2, so an OLD reader refuses the file loudly instead of
        silently refitting a count/time-effect program as Gaussian/no-tau."""
        state = cl.LearningState(
            channels=["A", "B"],
            center=np.full(2, 0.8),
            B=2.0,
            value=5.0,
            likelihood="negbinomial",
            time_effect="national",
        )
        wave = _tiny_panel()
        wave["y"] = np.round(np.abs(wave["y"]) * 10)
        wave["period_idx"] = np.repeat(np.arange(3), 6)
        state.ingest(wave)
        p = tmp_path / "nb_tau.npz"
        cl.state_to_npz(state, p)
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"].item()))
        assert meta["schema_version"] == 2
        back = cl.state_from_npz(p)  # the NEW reader accepts <= 2
        assert back.likelihood == "negbinomial"
        # a lone NB posterior payload is version 2 as well
        world = cl.make_world(seed=0)
        assert cl.posterior_to_payload(_nb_posterior(world))["schema_version"] == 2

    def test_reader_rejects_a_newer_schema_version(self, tmp_path):
        state = cl.LearningState(
            channels=["A", "B"], center=np.full(2, 0.8), B=2.0, value=5.0
        )
        state.ingest(_tiny_panel())
        p = tmp_path / "v3.npz"
        cl.state_to_npz(state, p)
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"].item()))
            arrays = {k: z[k] for k in z.files if k != "meta"}
        meta["schema_version"] = 3
        np.savez_compressed(p, meta=json.dumps(meta), **arrays)
        with pytest.raises(ValueError, match="newer"):
            cl.state_from_npz(p)


# ── slow: tiny-NUTS integration gates ─────────────────────────────────────────


def _summaries_from_truth(beta, kappa, alpha, center, scale=20.0):
    """Noise-free summary observations off a known 2-channel Hill surface."""
    gamma0 = np.zeros((len(beta), len(beta)))

    def resp(s):
        return float(surface.incremental(np.asarray(s), beta, kappa, alpha, gamma0))

    out = []
    for c in range(len(beta)):
        for mult in (1.6, 1.3, 0.4, 0.0):
            test = center.copy()
            test[c] = center[c] * mult
            lift = scale * (resp(test) - resp(center))
            out.append(
                {
                    "spend_test": test,
                    "spend_base": center.copy(),
                    "lift": lift,
                    "se": max(0.02 * abs(lift), 0.02),
                    "scale": scale,
                }
            )
    return out


@pytest.mark.slow
def test_summaries_only_fit_recovers_beta_ordering():
    """A team with historical readouts and NO panel fits the surface and gets
    the channel ordering right (the funded set is trustworthy even when the
    curve shape stays prior-dominated)."""
    beta = np.array([2.0, 0.5])
    kappa = np.array([0.8, 0.8])
    alpha = np.array([2.0, 2.0])
    center = np.array([0.8, 0.8])
    summaries = _summaries_from_truth(beta, kappa, alpha, center)
    post = cl.fit(
        {"summaries": summaries, "n_geo": 0},
        channels=["Strong", "Weak"],
        pairs=[],  # no interactions: 8 summaries identify main effects only
        num_warmup=100,
        num_samples=100,
        num_chains=1,
        seed=0,
    )
    assert post.diagnostics["evidence"] == {"n_rows": 0, "n_summaries": 8}
    assert "a_geo" not in post.samples  # no panel -> no geo-intercept plate
    beta_hat = post.samples["beta"].mean(0)
    assert beta_hat[0] > beta_hat[1]  # planted ordering recovered
    # the activation-agnostic planner runs on a summaries-only posterior
    rec = cl.recommend_allocation(post, B=1.6, value=5.0, q=20, mode="fixed")
    assert rec.sum() == pytest.approx(1.6, abs=0.05)
    assert rec[0] > rec[1]


@pytest.fixture(scope="module")
def arms_state():
    """One tiny joint (panel + summaries) fit on an arms world, reused below."""
    spec = cl.expand_arms(["Search", "Display"], {"Search": ["Brand", "NonBrand"]})
    world = cl.TrueWorld(
        beta=np.array([1.8, 1.2, 0.9]),
        kappa=np.array([0.7, 0.8, 0.9]),
        alpha=np.array([2.0, 1.8, 2.2]),
        gamma_pairs=np.array([-0.4, 0.0, 0.2]),
        channels=spec.channels,
        a_level=4.0,
        sigma_a=1.0,
    )
    center = np.full(3, 0.8)
    data = cl.simulate_panel(
        world, center, n_geo=24, t_pre=2, t_test=4, delta=0.6, noise=0.5, seed=1
    )
    data["geo_ids"] = [f"geo_{i}" for i in range(24)]
    state = cl.LearningState(
        channels=spec.channels,
        center=center,
        B=2.4,
        value=5.0,
        pair_signs=cl.default_arm_pair_signs(spec),
    )
    state.ingest(data)
    state.ingest_summaries(
        [
            _summary(center, 0, 0.4, 6.0, se=1.0, scale=8.0),
            _summary(center, 2, -0.4, -3.0, se=1.0, scale=8.0),
        ]
    )
    state.fit(num_warmup=100, num_samples=100, num_chains=1, seed=0)
    return spec, state


@pytest.mark.slow
def test_panel_plus_summaries_joint_fit_runs(arms_state):
    _, state = arms_state
    post = state.posterior
    assert post.diagnostics["evidence"] == {"n_rows": 24 * 6, "n_summaries": 2}
    assert post.samples["beta"].shape == (100, 3)
    assert "a_geo" in post.samples  # the panel block is on in a joint fit
    plan = state.plan(q=24, seed=0)
    assert plan.recommendation.sum() == pytest.approx(2.4, abs=0.05)
    assert np.isfinite(plan.e_regret) and plan.e_regret >= 0


@pytest.mark.slow
def test_serialize_round_trip_reproduces_the_plan(arms_state, tmp_path):
    _, state = arms_state
    path = tmp_path / "state.npz"
    cl.state_to_npz(state, path)
    back = cl.state_from_npz(path)
    plan0 = state.plan(q=24, seed=7)
    plan1 = back.plan(q=24, seed=7)
    np.testing.assert_array_equal(plan0.recommendation, plan1.recommendation)
    np.testing.assert_array_equal(plan0.mroas_mean, plan1.mroas_mean)
    assert plan0.e_regret == plan1.e_regret


@pytest.mark.slow
def test_grouped_budget_arms_allocation(arms_state):
    """The two Search arms' recommended spends sum to the parent budget."""
    spec, state = arms_state
    search_idx = spec.groups["Search"]
    parent_budget = 1.4
    plan = state.plan(q=24, seed=0, group_budgets=[(search_idx, parent_budget)])
    assert plan.recommendation[search_idx].sum() == pytest.approx(
        parent_budget, abs=1e-4
    )
    assert plan.recommendation.sum() == pytest.approx(2.4, abs=1e-6)


@pytest.mark.slow
def test_natural_scale_kpi_fits_with_auto_prior_scaling():
    """A revenue-scale KPI (y ~ 1e4-1e5, per the never-normalize-y contract)
    fits sanely under prior_scaling='auto': the posterior noise lands near the
    true noise on the y scale and the channel ordering is recovered. Under the
    old hard-coded O(1) priors sigma was capped near ~3 by HalfNormal tails and
    beta could not carry natural-unit incremental KPI."""
    world = cl.make_world(seed=0)
    center = np.array([0.8, 0.8, 0.8, 0.8])
    data = cl.simulate_panel(
        world, center, n_geo=40, t_pre=4, t_test=6, delta=0.6, noise=0.5, seed=1
    )
    data["y"] = data["y"] * 1e4  # natural units: a revenue-scale KPI
    post = cl.fit(
        data,
        channels=world.channels,
        pair_signs=cl.PAIR_SIGNS_EXAMPLE,
        num_warmup=200,
        num_samples=200,
        num_chains=1,
        seed=0,
    )
    ps = post.diagnostics["prior_scaling"]
    assert ps["mode"] == "auto" and ps["y_scale"] > 1e3
    sigma_hat = float(np.mean(post.samples["sigma"]))
    assert 0.25e4 < sigma_hat < 1.5e4  # near true noise x 1e4, not O(1)-capped
    beta_hat = post.samples["beta"].mean(0)
    assert int(np.argmax(beta_hat)) == int(np.argmax(world.beta))
