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
        assert store.verify_plan_chain("f", "o") == {"intact": True, "n": 3}

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
        out = store.verify_plan_chain("f", "o")
        assert out["intact"] is False
        assert out["broken_at"] == v1["id"] and out["broken_version"] == 1

    def test_an_empty_family_verifies_vacuously(self, store):
        assert store.verify_plan_chain("nothing-here", "o") == {"intact": True, "n": 0}


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
        assert store.verify_plan_chain("f", "o")["intact"]

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


# ---------------------------------------------------------------------------
# Reproduce from provenance — #225's graded criterion
#
# The claim a commitment makes is not "here is a number" but "here is a number
# anyone can regenerate". A stored forecast nobody can reproduce is a screenshot
# wearing a commitment's clothes, and the variance measured against it cannot be
# defended.
# ---------------------------------------------------------------------------


class TestReproductionRefusals:
    """Refusals need no model — they are about the inputs, not the fit.

    Note the distinction these pin: "I refuse to check" and "I checked and it
    differs" are different statements about a commitment, and conflating them
    would blame the model for a moved file.
    """

    def _version(self, prov, payload_extra=None):
        return {
            "payload": {
                "forecast": {"mean": [1.0], "lower": [0.5], "upper": [1.5]},
                "provenance": prov,
                **(payload_extra or {}),
            }
        }

    def test_missing_provenance_refuses_rather_than_reporting_a_mismatch(self):
        from mmm_framework.platform.plan_of_record import reproduce_committed_plan

        r = reproduce_committed_plan(self._version({"run_id": "r"}))
        assert r.refused and not r.reproduced
        assert "spec_hash" in r.reason

    def test_a_missing_model_directory_refuses(self, tmp_path):
        from mmm_framework.platform.plan_of_record import reproduce_committed_plan

        r = reproduce_committed_plan(
            self._version({**PROVENANCE, "model_path": str(tmp_path / "gone")})
        )
        assert r.refused
        assert "is gone" in r.reason

    def test_a_changed_dataset_refuses_and_says_which_hash(self, tmp_path):
        """The committed number was correct for the data it was made on;
        recomputing against different data would not verify it."""
        from mmm_framework.platform.plan_of_record import reproduce_committed_plan

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        csv = tmp_path / "data.csv"
        csv.write_text("a,b\n1,2\n")

        r = reproduce_committed_plan(
            self._version(
                {
                    **PROVENANCE,
                    "model_path": str(model_dir),
                    "dataset_path": str(csv),
                    "data_fingerprint": "not-the-current-hash",
                }
            )
        )
        assert r.refused and not r.reproduced
        assert "has changed" in r.reason
        assert "not-the-current-hash" in r.reason

    def test_an_unreadable_dataset_refuses(self, tmp_path):
        from mmm_framework.platform.plan_of_record import reproduce_committed_plan

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        r = reproduce_committed_plan(
            self._version(
                {
                    **PROVENANCE,
                    "model_path": str(model_dir),
                    "dataset_path": str(tmp_path / "vanished.csv"),
                }
            )
        )
        assert r.refused and "no longer readable" in r.reason

    def test_no_dataset_path_refuses(self, tmp_path):
        """The panel is rebuilt from the dataset, so its path is part of what
        makes a commitment reproducible."""
        from mmm_framework.platform.plan_of_record import reproduce_committed_plan

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        r = reproduce_committed_plan(
            self._version({**PROVENANCE, "model_path": str(model_dir)})
        )
        assert r.refused and "no dataset path" in r.reason

    def test_a_run_without_a_saved_spec_refuses(self, tmp_path):
        """Without the run's own spec the panel cannot be rebuilt AS FITTED —
        and a current session spec may have been edited since."""
        from mmm_framework.platform.plan_of_record import reproduce_committed_plan

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        csv = tmp_path / "data.csv"
        csv.write_text("a,b\n1,2\n")
        from mmm_framework.platform.runs import data_fingerprint

        fp = data_fingerprint(str(csv))
        r = reproduce_committed_plan(
            self._version(
                {
                    **PROVENANCE,
                    "model_path": str(model_dir),
                    "dataset_path": str(csv),
                    "data_fingerprint": fp["md5"],
                }
            )
        )
        assert r.refused and "no model spec" in r.reason


def _fp(path):
    from mmm_framework.platform.runs import data_fingerprint

    return data_fingerprint(path)["md5"]


@pytest.mark.slow
class TestReproductionEndToEnd:
    """The graded criterion: commit, reload from provenance, recompute."""

    def _fit_and_save(self, tmp_path):
        """Fit from the SAME long-format MFF the reproduction path will reload,
        so the round trip exercises the real loader rather than a shortcut."""
        import contextlib
        import io
        import warnings

        from mmm_framework.agents.fitting import build_model
        from mmm_framework.serialization import MMMSerializer
        from mmm_framework.synth.mff import generate_mff

        frame, _answer = generate_mff("clean", seed=0, n_weeks=120)
        csv = tmp_path / "world.csv"
        frame.to_csv(csv, index=False)

        spec = {
            "kpi": "Sales",
            "media_channels": [
                {"name": c} for c in ("TV", "Search", "Social", "Display")
            ],
            "control_variables": [{"name": "Price"}],
            "trend": {"type": "linear"},
            "inference": {"method": "map"},
        }
        model = build_model(spec, str(csv))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stderr(io.StringIO()):
                model.fit(method="map", random_seed=0)
        save_dir = str(tmp_path / "run_1")
        MMMSerializer.save(model, save_dir)
        import json

        (tmp_path / "run_1" / "run_metadata.json").write_text(
            json.dumps({"spec": spec, "kpi": spec["kpi"]})
        )
        return model, frame, spec, save_dir, str(csv)

    def test_a_committed_forecast_regenerates_to_tolerance(self, store, tmp_path):
        import json
        import warnings

        from mmm_framework.planning.forecast import forecast_under_plan
        from mmm_framework.platform.plan_of_record import reproduce_committed_plan

        model, frame, spec, save_dir, csv_path = self._fit_and_save(tmp_path)

        # Plan the LAST 8 weeks of observed spend forward — inside observed
        # support, so the commitment gate would not refuse it either.
        plan = {
            c: [float(x) for x in model.X_media_raw[-8:, i]]
            for i, c in enumerate(model.channel_names)
        }
        controls = {
            c: [float(model.X_controls_raw[-8:, i].mean())] * 8
            for i, c in enumerate(model.control_names)
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = forecast_under_plan(
                model, plan, future_controls=controls, random_seed=42
            )


        payload = {
            "forecast": {
                "mean": [float(x) for x in fc.mean],
                "lower": [float(x) for x in fc.lower],
                "upper": [float(x) for x in fc.upper],
                "interval": fc.interval,
                "n_draws": fc.n_draws,
            },
            "plan_media": plan,
            "plan_controls": controls,
            "random_seed": 42,
            "provenance": {
                "run_id": "run_1",
                "spec_hash": "abc",
                "data_fingerprint": _fp(csv_path),
                "dataset_path": csv_path,
                "model_path": save_dir,
            },
        }
        v = store.commit_plan_version(
            plan_family="fam", org_id="o", project_id="p", payload=payload
        )

        r = reproduce_committed_plan(store.get_plan_version(v["id"]))
        assert not r.refused, r.reason
        assert r.reproduced, f"max diff {r.max_abs_diff} > {r.tolerance}: {r.diffs}"
        assert r.max_abs_diff <= 1e-9

    def test_a_tampered_snapshot_reproduces_FALSE_rather_than_refusing(
        self, store, tmp_path
    ):
        """The other half of the distinction: inputs intact, numbers moved."""
        import json
        import warnings

        from mmm_framework.planning.forecast import forecast_under_plan
        from mmm_framework.platform.plan_of_record import reproduce_committed_plan

        model, frame, spec, save_dir, csv_path = self._fit_and_save(tmp_path)
        # Plan the LAST 8 weeks of observed spend forward — inside observed
        # support, so the commitment gate would not refuse it either.
        plan = {
            c: [float(x) for x in model.X_media_raw[-8:, i]]
            for i, c in enumerate(model.channel_names)
        }
        controls = {
            c: [float(model.X_controls_raw[-8:, i].mean())] * 8
            for i, c in enumerate(model.control_names)
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = forecast_under_plan(
                model, plan, future_controls=controls, random_seed=42
            )

        inflated = [float(x) * 1.10 for x in fc.mean]  # someone "improved" it
        payload = {
            "forecast": {
                "mean": inflated,
                "lower": [float(x) for x in fc.lower],
                "upper": [float(x) for x in fc.upper],
                "interval": fc.interval,
                "n_draws": fc.n_draws,
            },
            "plan_media": plan,
            "plan_controls": controls,
            "random_seed": 42,
            "provenance": {
                "run_id": "run_1",
                "spec_hash": "abc",
                "data_fingerprint": _fp(csv_path),
                "dataset_path": csv_path,
                "model_path": save_dir,
            },
        }
        v = store.commit_plan_version(
            plan_family="fam2", org_id="o", project_id="p", payload=payload
        )
        r = reproduce_committed_plan(store.get_plan_version(v["id"]))
        assert not r.refused, "the inputs are intact — this is a mismatch, not a refusal"
        assert not r.reproduced
        assert r.max_abs_diff > 1e-9


# ---------------------------------------------------------------------------
# Tenant isolation (#225 follow-up)
#
# `plan_family` is a caller-supplied string. Keying versions on it ALONE — as
# the first cut did — means two tenants who both call theirs "FY25" share one
# lineage. Measured before fixing: org B's commit superseded org A's plan of
# record, org B's FIRST plan was numbered v2, a family-scoped list returned org
# A's payload, and org A's hash chain depended on org B's rows.
# ---------------------------------------------------------------------------


class TestTenantIsolation:
    def test_two_orgs_can_use_the_same_family_name(self, store):
        a = store.commit_plan_version(
            plan_family="FY25", org_id="org_a", project_id="pa", payload={"who": "A"}
        )
        b = store.commit_plan_version(
            plan_family="FY25", org_id="org_b", project_id="pb", payload={"who": "B"}
        )
        assert a["version"] == 1 and b["version"] == 1, "version sequences are per org"

    def test_one_orgs_commit_does_not_supersede_anothers(self, store):
        a = store.commit_plan_version(
            plan_family="FY25", org_id="org_a", project_id="pa", payload={"who": "A"}
        )
        store.commit_plan_version(
            plan_family="FY25", org_id="org_b", project_id="pb", payload={"who": "B"}
        )
        assert store.get_plan_version(a["id"])["status"] == "committed"

    def test_a_family_scoped_list_does_not_leak_another_tenant(self, store):
        store.commit_plan_version(
            plan_family="FY25", org_id="org_a", project_id="pa", payload={"who": "A"}
        )
        store.commit_plan_version(
            plan_family="FY25", org_id="org_b", project_id="pb", payload={"who": "B"}
        )
        rows = store.list_plan_versions(plan_family="FY25", org_id="org_b")
        assert [r["payload"]["who"] for r in rows] == ["B"]

    def test_a_family_only_query_is_refused_rather_than_answered(self, store):
        """Answering it would return whichever tenants happened to pick that
        name — so it is not a query the store will serve."""
        with pytest.raises(ValueError, match="requires org_id"):
            store.list_plan_versions(plan_family="FY25")

    def test_each_orgs_chain_verifies_independently(self, store):
        store.commit_plan_version(plan_family="FY25", org_id="org_a", payload={"n": 1})
        store.commit_plan_version(plan_family="FY25", org_id="org_b", payload={"n": 1})
        assert store.verify_plan_chain("FY25", "org_a") == {"intact": True, "n": 1}
        assert store.verify_plan_chain("FY25", "org_b") == {"intact": True, "n": 1}

    def test_tampering_in_one_org_does_not_break_anothers_chain(self, store):
        a = store.commit_plan_version(
            plan_family="FY25", org_id="org_a", payload={"n": 1}
        )
        store.commit_plan_version(plan_family="FY25", org_id="org_b", payload={"n": 1})
        with store._conn() as c:
            c.execute(
                "UPDATE plan_versions SET payload_json = ? WHERE id = ?",
                ('{"n": 999}', a["id"]),
            )
        assert store.verify_plan_chain("FY25", "org_a")["intact"] is False
        assert store.verify_plan_chain("FY25", "org_b")["intact"] is True

    def test_the_chain_hash_covers_the_org(self, store):
        """Otherwise a row could be moved between tenants and still verify."""
        from mmm_framework.platform.sessions import _plan_version_hash

        a = _plan_version_hash("", "org_a", "FY25", 1, "{}", 1.0)
        b = _plan_version_hash("", "org_b", "FY25", 1, "{}", 1.0)
        assert a != b
