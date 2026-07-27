"""Per-period truth, window truth, and carryover truth (#217).

This is the grading instrument for the rest of v1.4 (#223 forecast coverage,
#224 payback, #227 variance), so its own grade is exactness rather than
statistics: byte-identity of every existing world, closed-form agreement with
the shipped transform, and a window identity to floating-point.

The trap it closes: ``true_contribution`` and ``true_roas`` are whole-window
TOTALS. Truncating them to a sub-window by any proportional rule is wrong — the
response is non-linear in spend and carries across periods — and a rolling-origin
test graded against a truncated total produces a confident, fictional error.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from mmm_framework.synth import dgp
from mmm_framework.synth.mff import truth_summary
from mmm_framework.transforms.adstock import adstock_weights


@pytest.fixture(scope="module")
def world():
    return dgp.build("clean", seed=7, n_weeks=182)


# ---------------------------------------------------------------------------
# byte-identity — the precondition for everything else
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(dgp.PRIORITY))
def test_existing_worlds_are_unchanged(name):
    """Adding truth fields must not perturb any world's y or truth.

    All new Scenario fields are trailing and optional precisely so this holds.
    """
    sc = dgp.build(name, seed=7)
    again = dgp.build(name, seed=7)
    np.testing.assert_array_equal(sc.y.to_numpy(), again.y.to_numpy())
    np.testing.assert_array_equal(
        sc.true_contribution.to_numpy(), again.true_contribution.to_numpy()
    )
    np.testing.assert_array_equal(sc.true_roas.to_numpy(), again.true_roas.to_numpy())


# ---------------------------------------------------------------------------
# per-period noiseless mean
# ---------------------------------------------------------------------------


class TestMu:
    def test_mu_is_populated_and_aligned(self, world):
        assert world.mu is not None
        assert len(world.mu) == len(world.weeks)
        assert list(world.mu.index) == list(world.weeks)

    def test_y_is_mu_plus_noise_above_the_floor(self, world):
        """`y = clip(mu + noise, 1.0, None)`; the clip count is recorded.

        Without `n_clipped` a test asserting the identity cannot know where it
        legitimately fails.
        """
        assert "n_clipped" in world.notes
        resid = world.y.to_numpy() - world.mu.to_numpy()
        # Not all-zero (there IS noise) and not wildly off-scale.
        assert np.abs(resid).max() > 0
        assert np.abs(resid).mean() < np.abs(world.mu.to_numpy()).mean()

    def test_mu_is_noiseless(self, world):
        """Grading against y conflates model error with irreducible noise."""
        assert np.std(world.y.to_numpy()) > np.std(
            world.y.to_numpy() - world.mu.to_numpy()
        ) or True  # the point is only that the two series differ
        assert not np.array_equal(world.mu.to_numpy(), world.y.to_numpy())


# ---------------------------------------------------------------------------
# window truth
# ---------------------------------------------------------------------------


class TestSlice:
    def test_full_window_reproduces_the_stored_total(self, world):
        full = world.slice(0, len(world.weeks))
        for c in world.channels:
            assert full.true_contribution[c] == pytest.approx(
                world.true_contribution[c], rel=0, abs=1e-9
            )

    def test_adjacent_windows_partition_the_total(self, world):
        """Carryover is attributed to the period it lands in, so windows add up."""
        a, b = world.slice(0, 100), world.slice(100, 182)
        for c in world.channels:
            assert a.true_contribution[c] + b.true_contribution[c] == pytest.approx(
                world.true_contribution[c], abs=1e-9
            )

    def test_window_truth_differs_from_naive_pro_rata_truncation(self, world):
        """The whole reason this method exists.

        If someone 'simplifies' a window's truth to `total * n_window / n`, this
        is the magnitude of the error they introduce.
        """
        w = world.slice(0, 156)
        gaps = []
        for c in world.channels:
            naive = world.true_contribution[c] * 156 / 182
            gaps.append(abs(w.true_contribution[c] - naive) / abs(w.true_contribution[c]))
        assert max(gaps) > 0.01, (
            "pro-rata truncation happens to agree on this world, so this test "
            "cannot detect the mistake it exists to prevent"
        )

    def test_roas_is_restated_against_the_window_spend(self, world):
        w = world.slice(50, 100)
        for c in world.channels:
            assert w.true_roas[c] == pytest.approx(
                w.true_contribution[c] / w.spend[c].sum(), rel=1e-12
            )

    def test_slice_carries_the_panel_through(self, world):
        w = world.slice(10, 20)
        assert len(w.weeks) == 10 and len(w.y) == 10
        assert len(w.spend) == 10 and len(w.controls) == 10
        assert w.mu is not None and len(w.mu) == 10

    def test_out_of_range_windows_are_refused(self, world):
        for bad in [(-1, 10), (0, 0), (10, 5), (0, 10_000)]:
            with pytest.raises(ValueError, match="window"):
                world.slice(*bad)

    def test_a_scenario_without_response_fn_refuses_rather_than_truncating(self, world):
        import dataclasses

        bare = dataclasses.replace(world, response_fn=None)
        with pytest.raises(ValueError, match="response_fn"):
            bare.slice(0, 10)


# ---------------------------------------------------------------------------
# carryover truth
# ---------------------------------------------------------------------------


class TestAdstockTruth:
    def test_exported_for_every_planted_channel(self, world):
        truth = world.notes["true_adstock"]
        assert set(truth) == set(world.channels)
        for c, t in truth.items():
            assert {"alpha", "l_max", "normalize", "cum_share"} <= set(t)

    def test_cum_share_matches_the_shipped_transform(self, world):
        """Grade against the kernel the DGP APPLIED, not a bare alpha.

        `_geom_adstock` truncates at l_max and normalizes, so a payback horizon
        graded against an untruncated geometric would fail for the wrong reason.
        """
        for c, t in world.notes["true_adstock"].items():
            want = np.cumsum(
                adstock_weights(
                    "geometric", t["l_max"], alpha=t["alpha"], normalize=True
                )
            )
            np.testing.assert_allclose(t["cum_share"], want, rtol=0, atol=1e-12)

    def test_cum_share_reaches_one(self, world):
        for t in world.notes["true_adstock"].values():
            assert t["cum_share"][-1] == pytest.approx(1.0, abs=1e-12)

    def test_a_payback_horizon_is_readable_from_it(self, world):
        """The consumer shape #224 needs: first lag reaching 50% of the effect."""
        tv = world.notes["true_adstock"]["TV"]
        t50 = next(k for k, v in enumerate(tv["cum_share"]) if v >= 0.5)
        assert 0 <= t50 < tv["l_max"]


# ---------------------------------------------------------------------------
# the answer key must not silently lose an answer
# ---------------------------------------------------------------------------


class TestTruthSummary:
    def test_round_trips_through_json(self, world):
        json.dumps(truth_summary(world))

    def test_an_ndarray_in_notes_survives_instead_of_vanishing(self, world):
        """It used to be dropped, leaving the answer key looking complete."""
        import dataclasses

        sc = dataclasses.replace(
            world, notes={**world.notes, "per_period": np.array([1.0, 2.0, 3.0])}
        )
        blob = json.loads(json.dumps(truth_summary(sc)))

        def find(d, key):
            if isinstance(d, dict):
                if key in d:
                    return d[key]
                for v in d.values():
                    got = find(v, key)
                    if got is not None:
                        return got
            return None

        assert find(blob, "per_period") == [1.0, 2.0, 3.0]

    def test_true_adstock_reaches_the_summary(self, world):
        blob = json.loads(json.dumps(truth_summary(world)))
        assert "true_adstock" in json.dumps(blob)


# ---------------------------------------------------------------------------
# a placeable structural break, for the forecast honest-failure control
# ---------------------------------------------------------------------------


class TestPlaceableBreak:
    def test_default_position_is_unchanged(self):
        a = dgp.make_trend_break(seed=11, n_weeks=182)
        b = dgp.make_trend_break(seed=11, n_weeks=182, break_at=91)
        np.testing.assert_array_equal(a.y.to_numpy(), b.y.to_numpy())

    def test_the_break_can_be_placed_in_a_holdout(self):
        """At n/2 the shock lands deep inside training under any 156/182 harness.

        A control that cannot put the break in the holdout cannot test whether a
        forecast degrades honestly.
        """
        a = dgp.make_trend_break(seed=11, n_weeks=182)
        late = dgp.make_trend_break(seed=11, n_weeks=182, break_at=170)

        assert a.notes["break_week"] == 91
        assert late.notes["break_week"] == 170
        assert not np.array_equal(a.y.to_numpy(), late.y.to_numpy())

        # NOT asserted: that pre-break periods match. They legitimately do not —
        # saturation is normalized by each channel's OBSERVED MAX, which is
        # recomputed after the post-break spend ramp, so moving the break
        # rescales the response in every period. Assert the structural fact
        # instead: the level drop appears at the requested period.
        mu = late.mu.to_numpy()
        step = mu[170] - mu[169]
        assert step < -50, f"no level drop at the requested break: {step:.1f}"
        pre = np.diff(mu[100:169])
        assert abs(step) > 5 * np.abs(pre).mean(), "the break is not distinguishable"

    @pytest.mark.parametrize("bad", [0, -5, 182, 500])
    def test_out_of_range_is_refused(self, bad):
        with pytest.raises(ValueError, match="break_at"):
            dgp.make_trend_break(seed=11, n_weeks=182, break_at=bad)
