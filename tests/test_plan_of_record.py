"""Append-only, hash-chained committed plans and the commitment gate — #225.

Two failure modes this exists for, and the tests are organised around them:

1. **In-place edits.** `budget_plans` upserts every column, so editing an old
   plan silently retargets what pacing compares delivery against — every
   historical variance becomes retroactively wrong while the audit trail still
   looks clean. A committed version is therefore immutable and hash-chained over
   its PAYLOAD, so altering a committed number breaks verification.

2. **Committing something indefensible.** #223 makes a forecast computable with
   caveats; this decides what may be *committed*, and the milestone's rule
   forbids "disclose and commit anyway". A held-flat trend past the horizon cap,
   autocorrelated residuals, spend beyond observed support, and a single-draw
   posterior each REFUSE. Overrides are recorded in the payload — an override
   nobody can see is indistinguishable from a gate that never fired — except
   provenance, which is not overridable, because waiving it produces exactly the
   unreproducible commitment the module exists to prevent.
"""

from __future__ import annotations

import pytest

from mmm_framework.platform import sessions as S
from mmm_framework.platform.plan_of_record import (
    DEFAULT_FLEXIBLE_TREND_HORIZON_CAP,
    assess_committability,
    build_commit_payload,
    provenance_gaps,
)

PROVENANCE = {
    "run_id": "run_1",
    "spec_hash": "spec_abc",
    "data_fingerprint": "data_def",
    "model_path": "/models/run_1",
}


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    return S


def _forecast(**over):
    fields = {
        "interval_widens_with_horizon": True,
        "trend_extrapolation": {"policy": "linear", "trend_type": "linear"},
        "residual_autocorrelation": {"autocorrelated": False, "ljung_box_p": 0.42},
        "extrapolated_channels": [],
        "interval_available": True,
    }
    fields.update(over.pop("fields", {}))
    fc = {
        "periods": ["2025-01-06"] * over.pop("n_periods", 8),
        "mean": [100.0],
        "draws_b64": "AAAA",
        "n_draws": 200,
        "caveat_fields": fields,
    }
    fc.update(over)
    return fc


# ---------------------------------------------------------------------------
# append-only store
# ---------------------------------------------------------------------------


class TestVersioning:
    def test_versions_increment_per_family(self, store):
        a = store.commit_plan_version(plan_family="f1", org_id="o", payload={"n": 1})
        b = store.commit_plan_version(plan_family="f1", org_id="o", payload={"n": 2})
        other = store.commit_plan_version(plan_family="f2", org_id="o", payload={"n": 1})
        assert (a["version"], b["version"]) == (1, 2)
        assert other["version"] == 1, "a different family counts from 1"

    def test_an_earlier_version_is_byte_identical_after_a_later_commit(self, store):
        v1 = store.commit_plan_version(
            plan_family="f", org_id="o", payload={"forecast": {"total": 100}}
        )
        store.commit_plan_version(
            plan_family="f", org_id="o", payload={"forecast": {"total": 999}}
        )
        assert store.get_plan_version(v1["id"])["payload"] == {
            "forecast": {"total": 100}
        }

    def test_committing_supersedes_the_previous_version(self, store):
        v1 = store.commit_plan_version(plan_family="f", org_id="o", payload={"n": 1})
        v2 = store.commit_plan_version(plan_family="f", org_id="o", payload={"n": 2})
        assert store.get_plan_version(v1["id"])["status"] == "superseded"
        assert store.get_plan_version(v2["id"])["status"] == "committed"

    def test_the_plan_of_record_is_the_committed_version_not_the_newest_draft(
        self, store
    ):
        store.commit_plan_version(
            plan_family="f", org_id="o", project_id="p", payload={"n": 1}
        )
        v2 = store.commit_plan_version(
            plan_family="f", org_id="o", project_id="p", payload={"n": 2}
        )
        assert store.latest_committed_plan("p")["id"] == v2["id"]
        assert store.latest_committed_plan("other") is None

    def test_mutating_a_committed_version_raises(self, store):
        v = store.commit_plan_version(plan_family="f", org_id="o", payload={"n": 1})
        with pytest.raises(S.ImmutablePlanVersionError, match="committed and immutable"):
            store.update_plan_version(v["id"], name="edited")


class TestChainVerification:
    def test_an_untouched_chain_verifies(self, store):
        for i in range(3):
            store.commit_plan_version(plan_family="f", org_id="o", payload={"n": i})
        assert store.verify_plan_chain("f") == {"intact": True, "n": 3}

    def test_tampering_with_a_payload_breaks_it_and_names_the_revision(self, store):
        v1 = store.commit_plan_version(plan_family="f", org_id="o", payload={"n": 1})
        store.commit_plan_version(plan_family="f", org_id="o", payload={"n": 2})
        # the chain covers the PAYLOAD, not a digest of the metadata — so
        # editing a committed number is what has to break it
        with store._conn() as c:
            c.execute(
                "UPDATE plan_versions SET payload_json = ? WHERE id = ?",
                ('{"n": 999}', v1["id"]),
            )
        out = store.verify_plan_chain("f")
        assert out["intact"] is False
        assert out["broken_at"] == v1["id"] and out["broken_version"] == 1

    def test_an_empty_family_verifies_vacuously(self, store):
        assert store.verify_plan_chain("nothing-here") == {"intact": True, "n": 0}


# ---------------------------------------------------------------------------
# the commitment gate
# ---------------------------------------------------------------------------


class TestCommitmentGate:
    def test_a_clean_forecast_commits(self):
        r = assess_committability(_forecast(), provenance=PROVENANCE)
        assert r.committable and not r.refusals

    def test_no_forecast_is_refused(self):
        r = assess_committability(None, provenance=PROVENANCE)
        assert not r.committable
        assert r.blocking_gates() == ["forecast"]

    def test_a_held_flat_trend_past_the_cap_is_refused(self):
        """The band does not widen with horizon under that policy, so a long
        window commits to an interval that is decorative past the cap."""
        long = _forecast(
            n_periods=DEFAULT_FLEXIBLE_TREND_HORIZON_CAP + 1,
            fields={
                "interval_widens_with_horizon": False,
                "trend_extrapolation": {"policy": "held_flat", "trend_type": "spline"},
            },
        )
        r = assess_committability(long, provenance=PROVENANCE)
        assert "trend_horizon" in r.blocking_gates()
        # and the SAME trend inside the cap is fine — the gate is the horizon,
        # not the trend family
        short = _forecast(
            n_periods=DEFAULT_FLEXIBLE_TREND_HORIZON_CAP,
            fields={
                "interval_widens_with_horizon": False,
                "trend_extrapolation": {"policy": "held_flat", "trend_type": "spline"},
            },
        )
        assert assess_committability(short, provenance=PROVENANCE).committable

    def test_autocorrelated_residuals_are_refused(self):
        r = assess_committability(
            _forecast(
                fields={
                    "residual_autocorrelation": {
                        "autocorrelated": True,
                        "ljung_box_p": 0.001,
                    }
                }
            ),
            provenance=PROVENANCE,
        )
        assert "residual_autocorrelation" in r.blocking_gates()
        assert "too narrow" in r.refusals[0].reason

    def test_spend_beyond_observed_support_is_refused(self):
        r = assess_committability(
            _forecast(
                fields={
                    "extrapolated_channels": [{"channel": "TV", "multiple": 1.6}]
                }
            ),
            provenance=PROVENANCE,
        )
        assert "spend_support" in r.blocking_gates()
        assert "curve fiction" in r.refusals[0].reason

    def test_a_forecast_with_no_interval_is_refused(self):
        """A MAP posterior has one draw; committing its point estimate as a plan
        of record states a precision the fit does not have."""
        r = assess_committability(
            _forecast(fields={"interval_available": False}), provenance=PROVENANCE
        )
        assert "interval_available" in r.blocking_gates()

    def test_every_refusal_names_a_remedy_not_just_a_problem(self):
        bad = _forecast(
            n_periods=52,
            fields={
                "interval_widens_with_horizon": False,
                "trend_extrapolation": {"policy": "held_flat", "trend_type": "spline"},
                "residual_autocorrelation": {
                    "autocorrelated": True,
                    "ljung_box_p": 0.002,
                },
                "extrapolated_channels": [{"channel": "TV", "multiple": 2.0}],
            },
        )
        r = assess_committability(bad, provenance=PROVENANCE)
        assert len(r.refusals) == 3
        for refusal in r.refusals:
            assert refusal.remedy, f"{refusal.gate} states no remedy"


class TestOverrides:
    def test_an_override_unblocks_and_is_recorded(self):
        fc = _forecast(
            fields={"extrapolated_channels": [{"channel": "TV", "multiple": 1.6}]}
        )
        note = "CMO accepted the test-and-learn risk on 2025-01-06"
        r = assess_committability(
            fc, provenance=PROVENANCE, overrides={"spend_support": note}
        )
        assert r.committable
        assert r.overrides == {"spend_support": note}
        # and it reaches the frozen payload — an override nobody can see is
        # indistinguishable from a gate that never fired
        payload = build_commit_payload(forecast=fc, committability=r)
        assert payload["committability"]["overrides"]["spend_support"] == note

    def test_an_override_is_gate_specific(self):
        fc = _forecast(
            fields={
                "extrapolated_channels": [{"channel": "TV", "multiple": 1.6}],
                "residual_autocorrelation": {
                    "autocorrelated": True,
                    "ljung_box_p": 0.001,
                },
            }
        )
        r = assess_committability(
            fc, provenance=PROVENANCE, overrides={"spend_support": "accepted"}
        )
        assert not r.committable
        assert r.blocking_gates() == ["residual_autocorrelation"]

    def test_an_empty_or_unknown_override_does_not_waive_anything(self):
        fc = _forecast(
            fields={"extrapolated_channels": [{"channel": "TV", "multiple": 1.6}]}
        )
        for ov in ({"spend_support": ""}, {"not_a_gate": "sure"}):
            assert not assess_committability(
                fc, provenance=PROVENANCE, overrides=ov
            ).committable


class TestProvenance:
    def test_gaps_are_named_individually(self):
        assert provenance_gaps({}) == [
            "run_id",
            "spec_hash",
            "data_fingerprint",
            "model_path",
        ]
        assert provenance_gaps({**PROVENANCE, "model_path": None}) == ["model_path"]

    def test_a_missing_model_path_blocks(self):
        """Auto-save failure is a caught, non-fatal branch at fit time, so a run
        can exist with no model on disk — and a commitment that cannot reload
        its model cannot be reproduced."""
        r = assess_committability(
            _forecast(), provenance={**PROVENANCE, "model_path": None}
        )
        assert not r.committable
        assert r.missing_provenance == ["model_path"]

    def test_provenance_cannot_be_overridden(self):
        """Waiving it produces exactly the unreproducible commitment this
        module exists to prevent."""
        r = assess_committability(
            _forecast(),
            provenance={"run_id": "r"},
            overrides={"provenance": "trust me"},
        )
        assert not r.committable
        assert "provenance" in r.blocking_gates()

    def test_an_unresolved_valuation_blocks_a_dollar_plan(self):
        r = assess_committability(
            _forecast(), provenance=PROVENANCE, valuation={"value_per_kpi": None}
        )
        assert not r.committable
        assert "valuation" in r.blocking_gates()


class TestCommitPayload:
    def test_the_forecast_is_stored_whole_including_its_draws(self):
        """A window-total interval cannot be recovered from per-period bounds,
        and variance work grades against draws."""
        fc = _forecast()
        payload = build_commit_payload(forecast=fc, provenance=PROVENANCE)
        assert payload["forecast"]["draws_b64"] == "AAAA"
        assert payload["forecast"]["n_draws"] == 200
        assert payload["provenance"] == PROVENANCE
        assert payload["schema_version"] == 1

    def test_a_committed_payload_round_trips_through_the_store(self, store):
        fc = _forecast()
        r = assess_committability(fc, provenance=PROVENANCE)
        payload = build_commit_payload(
            forecast=fc, provenance=PROVENANCE, committability=r
        )
        v = store.commit_plan_version(
            plan_family="f",
            org_id="o",
            project_id="p",
            payload=payload,
            run_id=PROVENANCE["run_id"],
            spec_hash=PROVENANCE["spec_hash"],
            data_fingerprint=PROVENANCE["data_fingerprint"],
        )
        back = store.get_plan_version(v["id"])
        assert back["payload"] == payload
        assert back["run_id"] == "run_1" and back["spec_hash"] == "spec_abc"
        assert store.verify_plan_chain("f")["intact"]

    def test_a_52_week_4_channel_commitment_stays_small(self, store):
        """The sessions.db checkpoint-bloat precedent: per-period draws are
        thinned and base64-encoded, so a committed plan is kilobytes."""
        import base64
        import json

        import numpy as np

        draws = np.random.default_rng(0).normal(100, 5, (200, 52)).astype("<f4")
        fc = _forecast(
            n_periods=52,
            draws_b64=base64.b64encode(draws.tobytes()).decode(),
            by_channel={f"ch{i}": [1.0] * 52 for i in range(4)},
        )
        payload = build_commit_payload(forecast=fc, provenance=PROVENANCE)
        size = len(json.dumps(payload))
        assert size < 100_000, f"committed payload is {size} bytes"
