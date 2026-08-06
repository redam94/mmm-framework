"""Capability-reachability gate (issue #228).

A shipped capability can be unreachable while CI stays green: the code, the
config field, the spec and a passing test file all exist, and no user-facing
surface can invoke it. Price/promo levers lived that way for a release (config
+ builder + spec + tests, while ``unconsumed_spec_path(['price'], …)`` returned
"the model builder never reads top-level `price`"), and 20 of the 41 model ops
have no session-export mapping while the comment saying "must be mirrored here"
is enforced by nothing.

This gate makes the decision explicit for every new capability:

* every ``ModelConfig`` field is either **wired** to an agent spec path that
  ``unconsumed_spec_path`` accepts, or listed in ``FIELD_NOT_REACHABLE`` with
  a substantive reason;
* every ``model_ops.OPS`` key is either mapped by ``session_export._OP_TOOLS``
  or listed in ``OP_NOT_EXPORTABLE`` with a substantive reason;
* the tool registration sets stay internally consistent.

The allowlists are the trap the issue names: every reason must be non-empty
and contain no non-reason ("TODO", "later", "tbd", "n/a"), and — the
self-cleaning half — an allowlisted entry that BECOMES reachable fails the
gate until its entry is deleted, so the list cannot rot into a rubber stamp.
"""

from __future__ import annotations

import re

import pytest

# ── The spec the probes run against ──────────────────────────────────────────

SPEC = {
    "kpi": "Sales",
    "kpi_level": "national",
    "time_granularity": "weekly",
    "media_channels": [{"name": "tv"}],
    "control_variables": [{"name": "price"}],
}

#: ModelConfig field → (agent spec path, sample value) proving the field is
#: reachable from a model spec. ``unconsumed_spec_path`` must accept the path.
FIELD_SPEC_PATHS: dict[str, tuple[str, object]] = {
    "specification": ("specification", "additive"),
    "likelihood": ("likelihood", "normal"),
    "media_prior_mode": ("media_prior_mode", "roi"),
    # inference.method takes fit methods + frequentist estimators; the value
    # frequentist_ridge lands on ModelConfig.inference_method (the Bayesian
    # BACKEND values are builder-only: .bayesian_numpyro() etc.).
    "inference_method": ("inference.method", "frequentist_ridge"),
    "fit_method": ("inference.method", "nuts"),
    "n_chains": ("inference.chains", 4),
    "n_draws": ("inference.draws", 1000),
    "n_tune": ("inference.tune", 1000),
    "target_accept": ("inference.target_accept", 0.9),
    "seasonality": ("seasonality", {"yearly": 4}),
    "price": ("price", {"variable": "price"}),
    "promotions": ("promotions", [{"variable": "price"}]),
    "intercept_prior_mu": ("priors.intercept.mu", 0.0),
    "intercept_prior_sigma": ("priors.intercept.sigma", 1.0),
    "media_roi_prior_mu": ("priors.media_default.roi_mu", 1.0),
    "media_roi_prior_sigma": ("priors.media_default.roi_sigma", 0.5),
}

#: Fields with NO agent spec path, each with the reason it is allowed to stay
#: that way. Deleting a row here without wiring the field fails the gate;
#: wiring the field without deleting the row also fails (staleness check).
FIELD_NOT_REACHABLE: dict[str, str] = {
    "events": (
        "ModelConfigBuilder.with_events exists but only the Python API reaches "
        "it; the agent spec has no `events` path. Known audit gap — wiring it "
        "means a spec key in agents/fitting.build_model plus an "
        "unconsumed_spec_path registry entry, at which point this row must be "
        "deleted."
    ),
    "channel_interactions": (
        "Python-API-only: consumed when BayesianMMM is constructed directly; "
        "no agent spec path writes it. Same wiring recipe as `events` when it "
        "is promoted."
    ),
    "reach_frequency": (
        "Python-API-only lever; the agent spec has no `reach_frequency` path. "
        "Same wiring recipe as `events` when it is promoted."
    ),
    "control_selection": (
        "Python-API-only: spike-and-slab control selection is configured on "
        "ModelConfig directly; no agent spec path writes it."
    ),
    "hierarchical": (
        "Geo pooling is driven by kpi_level + the dataset's geography "
        "dimension in the agent path; the HierarchicalConfig field is the "
        "Python API's explicit override and has no spec key."
    ),
    "use_grouped_media_priors": (
        "Python-API experiment toggle; deliberately not exposed to the agent "
        "spec so the oracle cannot silently regroup media priors."
    ),
    "use_parametric_adstock": (
        "Python-API experiment toggle; the agent path always uses the "
        "convolutional adstock, so a spec key would fork behaviour between "
        "surfaces."
    ),
    "ridge_alpha": (
        "Frequentist estimator knob read from ModelConfig by "
        "frequentist/ridge.py; the agent spec selects the estimator via "
        "inference.method and takes its defaults — per-knob spec paths were "
        "deliberately not exposed (technical-docs/frequentist-estimation.md)."
    ),
    "bootstrap_samples": (
        "Frequentist bootstrap size; same deliberate non-exposure as " "ridge_alpha."
    ),
    "optim_maxiter": (
        "Frequentist optimizer iteration cap; same deliberate non-exposure "
        "as ridge_alpha."
    ),
    "optim_seed": (
        "Frequentist optimizer seed; the agent path seeds fits via "
        "inference.random_seed on the fit call, not per-optimizer."
    ),
}

#: OPS keys with no session-export mapping, each with the reason. The export
#: renders these ops' stored dashboard artifacts instead of replaying them; a
#: row here says WHY no deterministic replay arg-mapper can (yet) exist.
OP_NOT_EXPORTABLE: dict[str, str] = {
    "plan_budget": (
        "Server planner-job op only — no agent tool invokes it, so a chat "
        "session export never contains a call to replay."
    ),
    "plan_scenario": ("Server planner-job op only, like plan_budget."),
    "check_pacing": (
        "The tool reads the project's delivery ledger + saved budget plan "
        "from the sessions store at call time; a replay outside the server "
        "has no store to read, so the export keeps the stored pacing table."
    ),
    "forecast_plan": (
        "Args include a forward calendar derived from the fitted panel at "
        "call time plus optional flighting expansion; the export keeps the "
        "stored forecast artifact (which since #227 carries the normalized "
        "plan + seed needed to reproduce it via reproduce_committed_plan)."
    ),
    "variance_to_plan": (
        "The tool passes store-fetched payloads (committed version, delivery, "
        "actuals, valuation) across the kernel boundary; an export cannot "
        "reconstruct the stores, so the bridge is kept as its stored artifact."
    ),
    "cfo_summary": (
        "Depends on the project's valuation preference resolved server-side "
        "(kpi_to_dollars over preferences/branding); the exported notebook "
        "has no preference store."
    ),
    "clv_value": (
        "Reads the project's CLV garden model + preferences from the store; "
        "no deterministic standalone replay."
    ),
    "confounding_sensitivity": (
        "Invoked from causal_tools with benchmark covariates chosen "
        "interactively; the export keeps the stored sensitivity surface."
    ),
    "endogeneity": (
        "Tool-invoked with the session's lever context; export keeps the "
        "stored screen table. A replay mapper is one tuple when someone "
        "needs it — the op takes only max_lag."
    ),
    "experiment_design": (
        "Design ops read the experiment registry + design engine state from "
        "the sessions store; exports keep the stored design cards."
    ),
    "experiment_priorities": ("Same store dependency as experiment_design."),
    "experiment_economics": (
        "Same store dependency as experiment_design, plus the valuation " "preference."
    ),
    "experiment_optimizer": (
        "Same store dependency as experiment_design; the Pareto sweep also "
        "depends on ranges chosen in the chat turn."
    ),
    "identify_structural_parameters": (
        "Multi-level flighting identification takes design ranges from the "
        "chat turn; export keeps the stored identification report."
    ),
    "garden_compat": (
        "Runs against a Model Garden ref resolved from the store; the "
        "exported notebook pins the garden model by source instead."
    ),
    "garden_tune_suggestions": ("Same garden-ref store dependency as garden_compat."),
    "spec_curve": (
        "The sweep grid is built from the session spec at call time and runs "
        "many fits; the export keeps the stored curve + BMA table rather "
        "than re-running a multi-fit sweep on the reader's machine."
    ),
    "triangulation": (
        "Joins MMM estimates with the experiment registry and platform "
        "figures from the store; export keeps the stored panel."
    ),
    "slide_deck_notes": (
        "Deck ops render branded artifacts with the project's branding "
        "preference; the export links the produced deck instead."
    ),
    "render_slide_deck": (
        "Same branding-preference dependency as slide_deck_notes; the deck "
        "file itself is the artifact."
    ),
}

_BANNED_REASON = re.compile(r"\b(todo|later|tbd|n/?a|fix ?me)\b", re.I)


# ── The pure checker (unit-testable with planted omissions) ─────────────────


def reachability_violations(
    field_names: set[str],
    wired_fields: set[str],
    field_allowlist: dict[str, str],
    op_names: set[str],
    mapped_ops: set[str],
    op_allowlist: dict[str, str],
) -> list[str]:
    """Every violation, named. Empty list == the gate passes."""
    out: list[str] = []
    for f in sorted(field_names - wired_fields - set(field_allowlist)):
        out.append(
            f"ModelConfig.{f} is reachable from no agent spec path and is "
            "not allowlisted — wire it or add a FIELD_NOT_REACHABLE reason"
        )
    for f in sorted(set(field_allowlist) & wired_fields):
        out.append(
            f"ModelConfig.{f} is allowlisted as not-reachable but IS wired — "
            "delete its FIELD_NOT_REACHABLE row"
        )
    for f in sorted(set(field_allowlist) - field_names):
        out.append(f"FIELD_NOT_REACHABLE names unknown field {f!r}")
    for o in sorted(op_names - mapped_ops - set(op_allowlist)):
        out.append(
            f"model op {o!r} has no session-export mapping and is not "
            "allowlisted — add an _OP_TOOLS entry or an OP_NOT_EXPORTABLE "
            "reason"
        )
    for o in sorted(set(op_allowlist) & mapped_ops):
        out.append(
            f"model op {o!r} is allowlisted as not-exportable but IS mapped "
            "in _OP_TOOLS — delete its OP_NOT_EXPORTABLE row"
        )
    for o in sorted(set(op_allowlist) - op_names):
        out.append(f"OP_NOT_EXPORTABLE names unknown op {o!r}")
    for name, reason in {**field_allowlist, **op_allowlist}.items():
        if not str(reason).strip():
            out.append(f"allowlist entry {name!r} has an empty reason")
        elif _BANNED_REASON.search(str(reason)):
            out.append(
                f"allowlist entry {name!r} carries a non-reason "
                f"({_BANNED_REASON.search(str(reason)).group(0)!r})"
            )
    return out


# ── Live inputs ──────────────────────────────────────────────────────────────


def _model_config_fields() -> set[str]:
    from mmm_framework.config import ModelConfig

    return set(ModelConfig.model_fields)


def _wired_fields() -> set[str]:
    from mmm_framework.agents.fitting import unconsumed_spec_path

    wired = set()
    for field, (path, value) in FIELD_SPEC_PATHS.items():
        if unconsumed_spec_path(path.split("."), value, SPEC) is None:
            wired.add(field)
    return wired


def _ops() -> set[str]:
    from mmm_framework.agents import model_ops

    return set(model_ops.OPS)


def _mapped_ops() -> set[str]:
    from mmm_framework.agents.session_export import _OP_TOOLS

    return {op_key for (op_key, _mapper, _extra) in _OP_TOOLS.values()}


# ── The gate ─────────────────────────────────────────────────────────────────


class TestReachabilityGate:
    def test_no_unaccounted_capability(self):
        violations = reachability_violations(
            _model_config_fields(),
            _wired_fields(),
            FIELD_NOT_REACHABLE,
            _ops(),
            _mapped_ops(),
            OP_NOT_EXPORTABLE,
        )
        assert violations == [], "\n".join(violations)

    def test_every_declared_spec_path_actually_consumes(self):
        """FIELD_SPEC_PATHS itself must not rot: a mapping whose path the
        validator rejects is a wiring claim without the wiring."""
        from mmm_framework.agents.fitting import unconsumed_spec_path

        broken = {
            field: unconsumed_spec_path(path.split("."), value, SPEC)
            for field, (path, value) in FIELD_SPEC_PATHS.items()
            if unconsumed_spec_path(path.split("."), value, SPEC) is not None
        }
        assert broken == {}, f"declared-but-unconsumed spec paths: {broken}"


class TestGateDetectsPlantedOmissions:
    """The acceptance criterion: a dummy field / op makes the gate fail
    naming both; allowlisting with a reason passes; blanking the reason
    fails again."""

    def _base(self):
        return dict(
            field_names=_model_config_fields(),
            wired_fields=_wired_fields(),
            field_allowlist=dict(FIELD_NOT_REACHABLE),
            op_names=_ops(),
            mapped_ops=_mapped_ops(),
            op_allowlist=dict(OP_NOT_EXPORTABLE),
        )

    def test_dummy_field_and_op_are_named(self):
        kw = self._base()
        kw["field_names"] = kw["field_names"] | {"dummy_lever"}
        kw["op_names"] = kw["op_names"] | {"dummy_op"}
        violations = reachability_violations(**kw)
        assert any("dummy_lever" in v for v in violations)
        assert any("dummy_op" in v for v in violations)

    def test_allowlisting_with_a_reason_passes(self):
        kw = self._base()
        kw["field_names"] = kw["field_names"] | {"dummy_lever"}
        kw["field_allowlist"][
            "dummy_lever"
        ] = "test fixture: reachable only from the harness"
        assert reachability_violations(**kw) == []

    def test_blanking_the_reason_fails_again(self):
        kw = self._base()
        kw["field_names"] = kw["field_names"] | {"dummy_lever"}
        kw["field_allowlist"]["dummy_lever"] = "   "
        assert any("empty reason" in v for v in reachability_violations(**kw))

    def test_a_non_reason_is_rejected(self):
        kw = self._base()
        kw["field_names"] = kw["field_names"] | {"dummy_lever"}
        kw["field_allowlist"]["dummy_lever"] = "TODO wire this later"
        assert any("non-reason" in v for v in reachability_violations(**kw))

    def test_stale_allowlist_entry_is_rejected(self):
        """Wiring a capability must force its allowlist row out."""
        kw = self._base()
        kw["field_allowlist"]["price"] = "no longer true — price is wired"
        assert any(
            "IS wired" in v and "price" in v for v in reachability_violations(**kw)
        )


class TestToolRegistrationConsistency:
    """The multi-list registration problem: TOOLS, EXPERT_TOOLS, HEAVY,
    MMM-only, causal, milestone and the export map must agree, and nothing
    enforced it before #228."""

    @pytest.fixture(scope="class")
    def names(self):
        # TOOLS, not EXPERT_TOOLS: the milestone set includes
        # delegate_to_expert, which only the orchestrator-facing list carries.
        from mmm_framework.agents.tools import TOOLS

        return {t.name for t in TOOLS}

    def test_heavy_tools_exist(self, names):
        from mmm_framework.agents.tools import HEAVY_TOOL_NAMES

        assert HEAVY_TOOL_NAMES <= names

    def test_mmm_only_tools_exist(self, names):
        from mmm_framework.agents.tools import _MMM_ONLY_TOOL_NAMES

        assert _MMM_ONLY_TOOL_NAMES <= names

    def test_causal_tools_exist(self, names):
        from mmm_framework.agents.tools import _CAUSAL_TOOL_NAMES

        assert _CAUSAL_TOOL_NAMES <= names

    def test_milestone_tools_exist_and_match_step_labels(self, names):
        from mmm_framework.agents.workflow_guard import (
            MILESTONE_TOOLS,
            STEP_LABELS,
        )

        assert MILESTONE_TOOLS <= names
        assert set(STEP_LABELS) == set(MILESTONE_TOOLS)

    def test_export_map_names_real_tools_and_real_ops(self, names):
        from mmm_framework.agents.session_export import _OP_TOOLS

        assert set(_OP_TOOLS) <= names
        assert _mapped_ops() <= _ops()
