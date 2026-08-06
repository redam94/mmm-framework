"""Variance to plan (#227): delivery-driven vs unexplained, and it closes.

What these pin, in the issue's own terms:

* **Only two buckets are identifiable without a refit** — per-channel delivery
  variance (a paired counterfactual on the COMMITTED posterior) and a LABELLED
  unexplained remainder. The refit "effectiveness" split is refused with the
  reason stated, and the word "effectiveness" appears on no surface.
* **The bridge closes exactly** — rows sum to actual − committed to 1e-9, by
  construction, with supplied adjustment lines subtracting from the remainder
  (never restating a channel).
* **The committed interval leads** — the realized total is scored against the
  committed window-total DRAWS before anything is called a variance.
* **Exact-truth recovery** — in a clean synthetic world where the plan is the
  true future spend rescaled (TV×1.3, Search×0.7) and delivery is the truth,
  the delivery rows carry the right signs and the paired-draw interval covers
  the world's ``response_fn`` delivery truth.
* **The #225 criterion actually holds now** — a committed snapshot built the
  way the forecast op builds it (plan_media/plan_controls/random_seed inside
  the forecast payload) reproduces from provenance to 1e-9, and a mutated
  dataset refuses. Until the forecast op emitted those keys this criterion
  read as met while every real commitment refused reproduction.

Fits: ONE module-scoped NUTS-lite fit (2x500) on the 156-week training slice
of a 182-week clean world — sliced, never rebuilt at a different length (#217).
"""

from __future__ import annotations

import contextlib
import io
import json
import warnings

import numpy as np
import pytest

from mmm_framework.finance.lines import MODELLED, SUPPLIED, BridgeLine
from mmm_framework.planning.variance import (
    REFIT_REFUSAL,
    supplied_line,
    variance_to_plan,
)
from mmm_framework.synth import dgp, mff

N_WEEKS = 182
N_TRAIN = 156
SEED = 7
FC_SEED = 42
#: The committed plan is the TRUE future spend rescaled — so "delivery variance"
#: has a known causal truth (the world's own response under each spend path).
SCALE = {"TV": 1.3, "Search": 0.7}


def _spec(sc):
    return {
        "kpi": "Sales",
        "kpi_level": "national",
        "media_channels": [
            {
                "name": c,
                "adstock": {"type": "geometric"},
                "saturation": {"type": "hill"},
            }
            for c in sc.channels
        ],
        "control_variables": [{"name": c} for c in sc.controls.columns],
    }


@pytest.fixture(scope="module")
def world():
    return dgp.make_clean(seed=SEED, n_weeks=N_WEEKS)


@pytest.fixture(scope="module")
def fitted(world, tmp_path_factory):
    from mmm_framework.agents.fitting import build_model

    tmp = tmp_path_factory.mktemp("vb")
    train = world.slice(0, N_TRAIN)
    csv = tmp / "world.csv"
    mff.scenario_to_mff(train).to_csv(csv, index=False)
    spec = _spec(train)
    model = build_model(spec, str(csv))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stderr(io.StringIO()):
            model.fit(draws=500, tune=500, chains=2, random_seed=42)
    return {"model": model, "csv": csv, "tmp": tmp, "spec": spec}


def _future(world):
    """(plan_media, actual_media, controls) over the forecast window.

    ``actual_media`` is the world's TRUE spend (what generated ``y``); the
    committed plan is that truth rescaled — so the delivery bucket has a
    causal ground truth to be graded against.
    """
    truth = {
        c: [float(v) for v in world.spend[c].to_numpy()[N_TRAIN:]]
        for c in world.channels
    }
    plan = {c: [v * SCALE.get(c, 1.0) for v in series] for c, series in truth.items()}
    controls = {
        c: [float(v) for v in world.controls[c].to_numpy()[N_TRAIN:]]
        for c in world.controls.columns
    }
    return plan, truth, controls


def _committed(model, plan, controls, *, max_draws=200, seed=FC_SEED):
    """A committed payload shaped the way the forecast op emits it (#227):
    the reproduction inputs live INSIDE the forecast snapshot."""
    from mmm_framework.planning.forecast import forecast_under_plan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = forecast_under_plan(
            model,
            plan,
            future_controls=controls,
            interval=0.9,
            max_draws=max_draws,
            random_seed=seed,
        )
    snap = {
        "periods": list(fc.periods),
        "mean": [float(x) for x in fc.mean],
        "lower": [None if np.isnan(x) else float(x) for x in fc.lower],
        "upper": [None if np.isnan(x) else float(x) for x in fc.upper],
        "interval": float(fc.interval),
        "draws_b64": fc.draws_b64,
        "n_draws": int(fc.n_draws),
        "plan_media": plan,
        "plan_controls": controls,
        "random_seed": seed,
    }
    return {"forecast": snap, "provenance": {"random_seed": seed}}, fc


def _actuals(world, payload):
    periods = payload["forecast"]["periods"]
    y = world.y.to_numpy(float)[N_TRAIN:]
    return [
        {"period": p, "kpi_value": float(v)} for p, v in zip(periods, y, strict=True)
    ]


@pytest.fixture(scope="module")
def bridge_inputs(world, fitted):
    plan, truth, controls = _future(world)
    payload, fc = _committed(fitted["model"], plan, controls)
    return {
        "plan": plan,
        "truth": truth,
        "controls": controls,
        "payload": payload,
        "fc": fc,
        "actuals": _actuals(world, payload),
    }


# ---------------------------------------------------------------------------
# Supplied lines — no fit needed
# ---------------------------------------------------------------------------


class TestSuppliedLines:
    def test_requires_a_source_note(self):
        with pytest.raises(ValueError, match="source"):
            supplied_line("Gross-to-net", -100.0, source_note="")

    def test_carries_supplied_provenance_and_note(self):
        ln = supplied_line("Gross-to-net", -100.0, source_note="ERP export 2026-07")
        assert ln.provenance is SUPPLIED
        assert "ERP export" in ln.source_note
        assert "supplied" in ln.describe()

    def test_refuses_interval_fields(self):
        """A supplied number has no posterior; inventing an interval for it
        would dress a manual adjustment as a measurement."""
        with pytest.raises(ValueError, match="[Ii]nterval|lower"):
            BridgeLine(
                name="x",
                value=1.0,
                provenance=SUPPLIED,
                source_note="s",
                lower=0.5,
                upper=1.5,
            )

    def test_refuses_per_channel_restatement(self):
        with pytest.raises(ValueError, match="TOTAL"):
            supplied_line("Net factor", -5.0, source_note="s", channel="TV")

    def test_refuses_non_dollar_kpi(self):
        with pytest.raises(ValueError, match="not dollar-denominated"):
            supplied_line("Net factor", -5.0, source_note="s", kpi_kind_is_dollar=False)


# ---------------------------------------------------------------------------
# The bridge itself — closure, buckets, verdict, refusals
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBridgeCloses:
    def test_rows_sum_to_the_gap_exactly(self, fitted, bridge_inputs, world):
        b = variance_to_plan(
            fitted["model"],
            bridge_inputs["payload"],
            bridge_inputs["truth"],
            bridge_inputs["actuals"],
        )
        assert b.closes
        total = sum(ln.value for ln in b.rows)
        assert total == pytest.approx(b.gap, abs=max(abs(b.gap) * 1e-9, 1e-9))
        assert b.gap == pytest.approx(b.actual_kpi - b.committed_kpi)

    def test_every_channel_gets_a_delivery_row(self, fitted, bridge_inputs):
        b = variance_to_plan(
            fitted["model"],
            bridge_inputs["payload"],
            bridge_inputs["truth"],
            bridge_inputs["actuals"],
        )
        names = [ln.name for ln in b.rows]
        for ch in fitted["model"].channel_names:
            assert f"Delivery — {ch}" in names
        assert "Unexplained" in names

    def test_delivery_rows_are_modelled_and_remainder_is_residual(
        self, fitted, bridge_inputs
    ):
        b = variance_to_plan(
            fitted["model"],
            bridge_inputs["payload"],
            bridge_inputs["truth"],
            bridge_inputs["actuals"],
        )
        for ln in b.rows:
            if ln.name.startswith("Delivery — "):
                assert ln.provenance is MODELLED
        (unexp,) = [ln for ln in b.rows if ln.name == "Unexplained"]
        assert unexp.provenance.value == "residual"
        # Labelled for its contents, not attributed.
        assert "labelled" in unexp.note.lower()

    def test_the_period_set_is_stated(self, fitted, bridge_inputs):
        b = variance_to_plan(
            fitted["model"],
            bridge_inputs["payload"],
            bridge_inputs["truth"],
            bridge_inputs["actuals"],
        )
        assert len(b.period_set) == N_WEEKS - N_TRAIN
        assert b.period_set == bridge_inputs["payload"]["forecast"]["periods"]

    def test_supplied_lines_subtract_from_unexplained_and_it_still_closes(
        self, fitted, bridge_inputs
    ):
        base = variance_to_plan(
            fitted["model"],
            bridge_inputs["payload"],
            bridge_inputs["truth"],
            bridge_inputs["actuals"],
        )
        adj = supplied_line("Gross-to-net", -250.0, source_note="ERP 2026-07")
        b = variance_to_plan(
            fitted["model"],
            bridge_inputs["payload"],
            bridge_inputs["truth"],
            bridge_inputs["actuals"],
            supplied=[adj],
        )
        assert b.closes
        assert b.unexplained == pytest.approx(base.unexplained + 250.0)
        # The delivery rows are untouched: a supplied line maps the TOTAL.
        for ln_b, ln_base in zip(b.rows, base.rows, strict=False):
            if ln_b.name.startswith("Delivery — "):
                assert ln_b.value == ln_base.value

    def test_non_supplied_provenance_in_supplied_list_raises(
        self, fitted, bridge_inputs
    ):
        fake = BridgeLine(name="x", value=1.0, provenance=MODELLED)
        with pytest.raises(ValueError, match="SUPPLIED"):
            variance_to_plan(
                fitted["model"],
                bridge_inputs["payload"],
                bridge_inputs["truth"],
                bridge_inputs["actuals"],
                supplied=[fake],
            )


@pytest.mark.slow
class TestIntervalVerdictLeads:
    def test_actuals_at_forecast_mean_are_within(self, fitted, bridge_inputs):
        payload = bridge_inputs["payload"]
        mean_actuals = [
            {"period": p, "kpi_value": float(v)}
            for p, v in zip(
                payload["forecast"]["periods"],
                payload["forecast"]["mean"],
                strict=True,
            )
        ]
        b = variance_to_plan(
            fitted["model"], payload, bridge_inputs["truth"], mean_actuals
        )
        assert b.within_committed_interval is True
        assert b.committed_lower is not None
        assert b.interval_mass == pytest.approx(0.9)
        assert any("WITHIN the committed interval" in c for c in b.caveats)

    def test_a_wild_miss_is_outside(self, fitted, bridge_inputs):
        payload = bridge_inputs["payload"]
        wild = [
            {"period": p, "kpi_value": float(v) * 3.0 + 1e4}
            for p, v in zip(
                payload["forecast"]["periods"],
                payload["forecast"]["mean"],
                strict=True,
            )
        ]
        b = variance_to_plan(fitted["model"], payload, bridge_inputs["truth"], wild)
        assert b.within_committed_interval is False
        assert any("OUTSIDE the committed interval" in c for c in b.caveats)

    def test_verdict_comes_from_draws_not_per_period_bounds(
        self, fitted, bridge_inputs
    ):
        """The committed band is the per-draw WINDOW-TOTAL percentiles. Summing
        per-period bounds would give a much wider band (perfect-correlation
        worst case) — pin that the computed band is strictly narrower."""
        payload = bridge_inputs["payload"]
        b = variance_to_plan(
            fitted["model"],
            payload,
            bridge_inputs["truth"],
            bridge_inputs["actuals"],
        )
        lower_sum = sum(v for v in payload["forecast"]["lower"] if v is not None)
        upper_sum = sum(v for v in payload["forecast"]["upper"] if v is not None)
        assert b.committed_lower > lower_sum
        assert b.committed_upper < upper_sum


@pytest.mark.slow
class TestRefusals:
    def test_partial_actuals_coverage_refuses(self, fitted, bridge_inputs):
        with pytest.raises(ValueError, match="not fully covered"):
            variance_to_plan(
                fitted["model"],
                bridge_inputs["payload"],
                bridge_inputs["truth"],
                bridge_inputs["actuals"][:-1],
            )

    def test_refit_split_is_refused_with_the_reason_stated(self, fitted, bridge_inputs):
        b = variance_to_plan(
            fitted["model"],
            bridge_inputs["payload"],
            bridge_inputs["truth"],
            bridge_inputs["actuals"],
            refit_run_id="run_newer",
        )
        assert REFIT_REFUSAL in b.refusals
        # The two identifiable buckets are still delivered.
        assert any(ln.name.startswith("Delivery — ") for ln in b.rows)
        assert b.closes

    def test_the_word_effectiveness_appears_nowhere(self, fitted, bridge_inputs):
        b = variance_to_plan(
            fitted["model"],
            bridge_inputs["payload"],
            bridge_inputs["truth"],
            bridge_inputs["actuals"],
            refit_run_id="run_newer",
        )
        blob = json.dumps(b.to_dict()).lower()
        assert "effectiveness" not in blob

    def test_a_model_that_does_not_reproduce_the_snapshot_refuses(
        self, fitted, bridge_inputs
    ):
        """A 'delivery bucket' computed on a different posterior would be the
        refused refit comparison wearing the committed label."""
        payload = json.loads(json.dumps(bridge_inputs["payload"]))
        payload["forecast"]["mean"] = [v * 1.1 for v in payload["forecast"]["mean"]]
        with pytest.raises(ValueError, match="does not reproduce"):
            variance_to_plan(
                fitted["model"],
                payload,
                bridge_inputs["truth"],
                bridge_inputs["actuals"],
            )

    def test_subtolerance_drift_is_carried_as_its_own_row(self, fitted, bridge_inputs):
        """Below the refusal threshold, drift becomes an explicit row so the
        closure identity is algebraic, not 'usually within tolerance'."""
        payload = json.loads(json.dumps(bridge_inputs["payload"]))
        payload["forecast"]["mean"][0] += 1e-7
        b = variance_to_plan(
            fitted["model"],
            payload,
            bridge_inputs["truth"],
            bridge_inputs["actuals"],
        )
        assert any(ln.name == "Reproduction drift" for ln in b.rows)
        assert b.closes

    def test_at_boundary_diagnostics_suppress_reconciliation_framing(
        self, fitted, bridge_inputs
    ):
        b = variance_to_plan(
            fitted["model"],
            bridge_inputs["payload"],
            bridge_inputs["truth"],
            bridge_inputs["actuals"],
            fit_diagnostics={"at_boundary": ["TV"]},
        )
        assert any("independent reconciliation" in c for c in b.caveats)

    def test_mixed_divisor_portfolio_suppresses_the_dollar_headline(
        self, fitted, bridge_inputs
    ):
        b = variance_to_plan(
            fitted["model"],
            bridge_inputs["payload"],
            bridge_inputs["truth"],
            bridge_inputs["actuals"],
            value_per_kpi=2.0,
            channel_meta={"TV": True, "Search": False},
        )
        assert b.dollar_headline_suppressed
        assert b.to_dict()["rows_dollars"] is None

    def test_dollar_rows_present_when_valuation_resolves_cleanly(
        self, fitted, bridge_inputs
    ):
        b = variance_to_plan(
            fitted["model"],
            bridge_inputs["payload"],
            bridge_inputs["truth"],
            bridge_inputs["actuals"],
            value_per_kpi=2.0,
            value_source="preferences",
        )
        d = b.to_dict()["rows_dollars"]
        assert d is not None
        (unexp,) = [ln for ln in b.rows if ln.name == "Unexplained"]
        assert d["Unexplained"] == pytest.approx(unexp.value * 2.0)


# ---------------------------------------------------------------------------
# Exact-truth recovery — the world's response_fn is the referee
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestExactTruthRecovery:
    def _true_delivery(self, world, plan, channel=None):
        """response_fn(S_actual) − response_fn(S_plan) summed over the window,
        on the FULL spend history so training-period carryover cancels."""
        actual = world.spend.to_numpy(float)
        planned = actual.copy()
        chans = world.channels if channel is None else [channel]
        for c in chans:
            i = world.channels.index(c)
            planned[N_TRAIN:, i] = np.asarray(plan[c], dtype=float)
        mu_a = world.response_fn(actual)
        mu_p = world.response_fn(planned)
        return float((mu_a - mu_p)[N_TRAIN:].sum())

    def test_perturbed_channels_carry_the_right_sign(
        self, world, fitted, bridge_inputs
    ):
        """TV was planned at 1.3× truth (actual under-delivers → negative row);
        Search at 0.7× (actual over-delivers → positive row)."""
        b = variance_to_plan(
            fitted["model"],
            bridge_inputs["payload"],
            bridge_inputs["truth"],
            bridge_inputs["actuals"],
        )
        rows = {ln.name: ln.value for ln in b.rows}
        assert rows["Delivery — TV"] < 0
        assert rows["Delivery — Search"] > 0
        # Sanity: the truth agrees on the signs.
        assert self._true_delivery(world, bridge_inputs["plan"], "TV") < 0
        assert self._true_delivery(world, bridge_inputs["plan"], "Search") > 0

    def test_delivery_interval_covers_the_response_fn_truth(
        self, world, fitted, bridge_inputs
    ):
        b = variance_to_plan(
            fitted["model"],
            bridge_inputs["payload"],
            bridge_inputs["truth"],
            bridge_inputs["actuals"],
        )
        truth = self._true_delivery(world, bridge_inputs["plan"])
        assert b.delivery_lower is not None
        assert b.delivery_lower <= truth <= b.delivery_upper, (
            f"true delivery {truth:.1f} outside "
            f"[{b.delivery_lower:.1f}, {b.delivery_upper:.1f}]"
        )

    def test_unexplained_is_small_when_the_world_is_clean(
        self, world, fitted, bridge_inputs
    ):
        """Actuals came from the world's own mu + noise; a clean-world fit
        should leave the unexplained line noise-sized, not signal-sized."""
        b = variance_to_plan(
            fitted["model"],
            bridge_inputs["payload"],
            bridge_inputs["truth"],
            bridge_inputs["actuals"],
        )
        n_fc = N_WEEKS - N_TRAIN
        noise_sd_window = 22.0 * np.sqrt(n_fc)  # world noise: N(0, 22) weekly
        assert abs(b.unexplained) < 6 * noise_sd_window


# ---------------------------------------------------------------------------
# The model op — markdown leads with the verdict, never says "effectiveness"
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestModelOp:
    def _run(self, fitted, bridge_inputs, **kw):
        from mmm_framework.agents import model_ops as M

        return M.OPS["variance_to_plan"](
            fitted["model"],
            None,
            committed_version={"payload": bridge_inputs["payload"]},
            actual_media=bridge_inputs["truth"],
            actuals=bridge_inputs["actuals"],
            **kw,
        )

    def test_op_returns_a_closing_bridge(self, fitted, bridge_inputs):
        res = self._run(fitted, bridge_inputs)
        assert res["error"] is None, res["error"]
        facts = res["dashboard"]["variance"]
        assert facts["closes"] is True
        assert "Variance to plan" in res["content"]

    def test_op_markdown_never_says_effectiveness(self, fitted, bridge_inputs):
        res = self._run(fitted, bridge_inputs, refit_run_id="run_newer")
        assert "effectiveness" not in res["content"].lower()
        assert "effectiveness" not in json.dumps(res["dashboard"]).lower()

    def test_op_refit_refusal_travels_with_the_run_diff(self, fitted, bridge_inputs):
        res = self._run(
            fitted,
            bridge_inputs,
            refit_run_id="run_newer",
            run_diff={"channels": ["TV roi 1.2 -> 1.4"]},
        )
        facts = res["dashboard"]["variance"]
        assert any("refused" in r.lower() for r in facts["refusals"])
        assert facts["run_diff"]["channels"] == ["TV roi 1.2 -> 1.4"]

    def test_op_supplied_line_without_valuation_refuses(self, fitted, bridge_inputs):
        res = self._run(
            fitted,
            bridge_inputs,
            supplied=[{"name": "GTN", "value": -10.0, "source_note": "ERP"}],
            valuation={"value_per_kpi": None, "source": "none"},
        )
        assert res["error"] is not None
        assert "not dollar-denominated" in res["error"]


# ---------------------------------------------------------------------------
# Reproduction roundtrip — the #225 criterion, end to end, now actually held
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestReproductionRoundtrip:
    def _saved_run(self, fitted, tmp_path):
        import shutil

        from mmm_framework.platform.runs import data_fingerprint
        from mmm_framework.serialization import MMMSerializer

        save_dir = tmp_path / "run_vb"
        MMMSerializer.save(fitted["model"], str(save_dir))
        (save_dir / "run_metadata.json").write_text(
            json.dumps({"run_name": "run_vb", "spec": fitted["spec"]})
        )
        csv = tmp_path / "world.csv"
        shutil.copy(fitted["csv"], csv)
        prov = {
            "run_id": "run_vb",
            "spec_hash": "spec-hash",
            "data_fingerprint": data_fingerprint(str(csv)),
            "model_path": str(save_dir),
            "dataset_path": str(csv),
        }
        return csv, prov

    def test_commit_reproduces_from_provenance_to_1e9(
        self, fitted, bridge_inputs, tmp_path
    ):
        """The forecast snapshot carries plan_media/plan_controls/random_seed
        (the #227 fix); reproduction reloads the model from disk, rebuilds the
        panel from the SAVED spec, and re-forecasts to 1e-9."""
        from mmm_framework.platform.plan_of_record import reproduce_committed_plan

        csv, prov = self._saved_run(fitted, tmp_path)
        version = {
            "payload": {
                "forecast": bridge_inputs["payload"]["forecast"],
                "provenance": prov,
            }
        }
        r = reproduce_committed_plan(version)
        assert not r.refused, r.reason
        assert r.reproduced, f"{r.reason} (max diff {r.max_abs_diff})"
        assert r.max_abs_diff <= 1e-9

    def test_a_mutated_dataset_refuses_rather_than_reverifying(
        self, fitted, bridge_inputs, tmp_path
    ):
        from mmm_framework.platform.plan_of_record import reproduce_committed_plan

        csv, prov = self._saved_run(fitted, tmp_path)
        text = csv.read_text()
        csv.write_text(text.replace(text.splitlines()[1], "tampered,0,0,0", 1))
        version = {
            "payload": {
                "forecast": bridge_inputs["payload"]["forecast"],
                "provenance": prov,
            }
        }
        r = reproduce_committed_plan(version)
        assert r.refused and not r.reproduced
        assert "has changed" in r.reason


# ---------------------------------------------------------------------------
# REST — refusals surface at POST time with the reason (409), not as job errors
# ---------------------------------------------------------------------------


class TestVarianceEndpoints:
    @pytest.fixture()
    def client(self, tmp_path, monkeypatch):
        from mmm_framework.platform import sessions as S

        monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
        monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
        S.init_db()
        import mmm_framework_server.main as main
        from fastapi.testclient import TestClient

        with TestClient(main.app) as c:
            yield c

    @pytest.fixture()
    def project(self, client):
        return client.post("/projects", json={"name": "P"}).json()["project_id"]

    def _commit(self, project, payload):
        from mmm_framework.platform import sessions as S

        return S.commit_plan_version(
            plan_family=f"proj-{project}",
            org_id="org_default",
            payload=payload,
            project_id=project,
        )

    def test_unknown_project_is_404(self, client):
        r = client.post("/projects/nope/variance", json={})
        assert r.status_code == 404

    def test_unknown_job_is_404(self, client, project):
        r = client.get(f"/projects/{project}/variance/nope")
        assert r.status_code == 404

    def test_no_committed_plan_refuses_409_with_the_reason(self, client, project):
        r = client.post(f"/projects/{project}/variance", json={})
        assert r.status_code == 409
        assert "No committed plan of record" in r.json()["detail"]

    def test_commitment_without_plan_media_refuses(self, client, project):
        self._commit(project, {"forecast": {"periods": ["w1"], "mean": [1.0]}})
        r = client.post(f"/projects/{project}/variance", json={})
        assert r.status_code == 409
        assert "no per-period spend plan" in r.json()["detail"]

    def test_missing_actuals_refuses_and_names_the_fix(self, client, project):
        self._commit(
            project,
            {
                "forecast": {
                    "periods": ["w1", "w2"],
                    "mean": [1.0, 1.0],
                    "plan_media": {"TV": [1.0, 2.0]},
                }
            },
        )
        r = client.post(f"/projects/{project}/variance", json={})
        assert r.status_code == 409
        assert "No realized KPI" in r.json()["detail"]
        assert "actuals" in r.json()["detail"]

    def test_delivery_gaps_refuse_rather_than_assuming_plan_as_delivered(
        self, client, project
    ):
        self._commit(
            project,
            {
                "forecast": {
                    "periods": ["w1", "w2"],
                    "mean": [1.0, 1.0],
                    "plan_media": {"TV": [1.0, 2.0]},
                }
            },
        )
        client.post(
            f"/projects/{project}/actuals",
            files={
                "file": (
                    "a.csv",
                    b"period,kpi_value\nw1,10\nw2,11\n",
                    "text/csv",
                )
            },
        )
        r = client.post(f"/projects/{project}/variance", json={})
        assert r.status_code == 409
        detail = r.json()["detail"]
        assert "does not cover" in detail
        assert "fabricate" in detail

    def test_supplied_line_without_source_note_refuses(self, client, project):
        self._commit(
            project,
            {
                "forecast": {
                    "periods": ["w1"],
                    "mean": [1.0],
                    "plan_media": {"TV": [1.0]},
                }
            },
        )
        r = client.post(
            f"/projects/{project}/variance",
            json={"supplied": [{"name": "GTN", "value": -5.0, "source_note": ""}]},
        )
        # Pydantic accepts the shape; the wrapper refuses the blank note.
        assert r.status_code == 409
        assert "source_note" in r.json()["detail"]
