"""Tests for Part B1: the oracle modeling modes — mode helpers, the prompt
decomposition (MMM byte-identical), mode-aware tool gating, and session storage.
"""

from __future__ import annotations

import pytest

from mmm_framework.agents import modes
from mmm_framework.agents.prompts import (
    DELEGATION_PREAMBLE,
    EXPERT_PREAMBLE,
    MMM_SYSTEM_PROMPT,
    assemble_system_prompt,
)
from mmm_framework.agents.tools import (
    EXPERT_TOOLS,
    ORCHESTRATOR_TOOLS,
    get_tools_for_mode,
)


# --------------------------------------------------------------------------- #
# modes.py
# --------------------------------------------------------------------------- #
class TestModes:
    def test_valid_and_normalize(self):
        assert modes.is_valid_mode("mmm")
        assert modes.is_valid_mode("descriptive")
        assert not modes.is_valid_mode("nope")
        assert not modes.is_valid_mode(None)
        assert modes.normalize_mode("general_bayes") == "general_bayes"
        assert modes.normalize_mode("garbage") == "mmm"
        assert modes.normalize_mode(None) == "mmm"

    def test_suggested_mode_for_kind(self):
        assert modes.suggested_mode_for_kind("mmm") == "mmm"
        assert modes.suggested_mode_for_kind("") == "mmm"
        assert modes.suggested_mode_for_kind("cfa") == "descriptive"
        assert modes.suggested_mode_for_kind("latent_class") == "descriptive"
        assert modes.suggested_mode_for_kind("something_else") == "general_bayes"

    def test_reconcile_consistent_for_mmm(self):
        r = modes.reconcile_mode_with_model("mmm", {"model_kind": "mmm"})
        assert r["consistent"] is True
        assert r["note"] is None

    def test_reconcile_flags_cfa_in_mmm_mode(self):
        r = modes.reconcile_mode_with_model("mmm", {"model_kind": "cfa"})
        assert r["consistent"] is False
        assert r["suggested_mode"] == "descriptive"
        assert "descriptive" in (r["note"] or "").lower()

    def test_reconcile_consistent_when_descriptive_matches_cfa(self):
        r = modes.reconcile_mode_with_model("descriptive", {"model_kind": "cfa"})
        assert r["consistent"] is True


# --------------------------------------------------------------------------- #
# prompts.py — MMM byte-identity + non-MMM coherence
# --------------------------------------------------------------------------- #
class TestPrompts:
    def test_mmm_assembly_matches_constants(self):
        # `mmm` mode reproduces the verbatim MMM prompt; the role preambles compose
        # exactly as the previous inline graph.py assembly did.
        assert assemble_system_prompt(mode="mmm", role=None) == MMM_SYSTEM_PROMPT
        assert (
            assemble_system_prompt(mode="mmm", role="orchestrator")
            == DELEGATION_PREAMBLE + MMM_SYSTEM_PROMPT
        )
        assert (
            assemble_system_prompt(mode="mmm", role="expert")
            == EXPERT_PREAMBLE + MMM_SYSTEM_PROMPT
        )

    def test_override_short_circuits(self):
        assert assemble_system_prompt(mode="descriptive", override="HELLO") == "HELLO"

    def test_unknown_mode_falls_back_to_mmm(self):
        assert assemble_system_prompt(mode="bogus", role=None) == MMM_SYSTEM_PROMPT

    @pytest.mark.parametrize(
        "mode", ["causal_inference", "general_bayes", "descriptive"]
    )
    def test_non_mmm_modes_differ_but_keep_discipline(self, mode):
        p = assemble_system_prompt(mode=mode, role=None)
        assert p != MMM_SYSTEM_PROMPT
        # Causal-measurement discipline preserved in EVERY mode.
        assert "define_research_question" in p
        assert "record_assumption" in p
        assert "prior_predictive_check" in p
        # Bayesian-modeling framing, not MMM-only.
        assert "Bayesian" in p

    def test_descriptive_has_no_roi_or_dag_push(self):
        p = assemble_system_prompt(mode="descriptive", role=None)
        assert "Descriptive / Measurement" in p
        assert "estimands" in p.lower()

    def test_role_preambles_apply_to_non_mmm(self):
        p = assemble_system_prompt(mode="general_bayes", role="orchestrator")
        assert p.startswith("## Two-tier execution")
        e = assemble_system_prompt(mode="general_bayes", role="expert")
        assert e.startswith("## You are the EXPERT")


# --------------------------------------------------------------------------- #
# tool gating — golden + partition
# --------------------------------------------------------------------------- #
class TestToolGating:
    def _names(self, ts):
        return [t.name for t in ts]

    def test_mmm_reproduces_existing_toolsets(self):
        assert self._names(get_tools_for_mode("mmm", "orchestrator")) == self._names(
            ORCHESTRATOR_TOOLS
        )
        assert self._names(get_tools_for_mode("mmm", "expert")) == self._names(
            EXPERT_TOOLS
        )

    @pytest.mark.parametrize(
        "mode", ["causal_inference", "general_bayes", "descriptive"]
    )
    def test_non_mmm_drops_mmm_only_tools(self, mode):
        ex = self._names(get_tools_for_mode(mode, "expert"))
        for n in ("get_roi_metrics", "run_budget_optimizer", "design_experiment_plan"):
            assert n not in ex
        # Spine is always present.
        for n in (
            "fit_mmm_model",
            "get_estimands",
            "execute_python",
            "list_garden_models",
        ):
            assert n in ex

    def test_descriptive_drops_causal_tools(self):
        ex = self._names(get_tools_for_mode("descriptive", "expert"))
        assert "propose_dag" not in ex
        assert "propose_dag" in self._names(
            get_tools_for_mode("causal_inference", "expert")
        )

    def test_orchestrator_keeps_delegate_drops_heavy(self):
        orch = self._names(get_tools_for_mode("descriptive", "orchestrator"))
        assert "delegate_to_expert" in orch
        assert "fit_mmm_model" not in orch  # HEAVY tool removed for orchestrator
        # Expert never gets delegate.
        assert "delegate_to_expert" not in self._names(
            get_tools_for_mode("mmm", "expert")
        )


# --------------------------------------------------------------------------- #
# session storage of modeling_mode
# --------------------------------------------------------------------------- #
class TestSessionMode:
    @pytest.fixture
    def store(self, tmp_path, monkeypatch):
        from mmm_framework.api import sessions as S

        monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
        S.init_db()
        return S

    def test_create_with_mode(self, store):
        s = store.create_session("S1", modeling_mode="descriptive")
        assert s["modeling_mode"] == "descriptive"
        got = store.get_session(s["thread_id"])
        assert got["modeling_mode"] == "descriptive"

    def test_create_default_is_mmm(self, store):
        s = store.create_session("S2")
        assert s["modeling_mode"] == "mmm"

    def test_update_mode(self, store):
        s = store.create_session("S3")
        assert store.update_session(s["thread_id"], modeling_mode="causal_inference")
        assert store.get_session(s["thread_id"])["modeling_mode"] == "causal_inference"

    def test_legacy_row_reads_as_mmm(self, store):
        # A row inserted without modeling_mode (NULL) must read as 'mmm'.
        s = store.create_session("S4")
        with store._conn() as c:
            c.execute(
                "UPDATE sessions SET modeling_mode = NULL WHERE thread_id = ?",
                (s["thread_id"],),
            )
        assert store.get_session(s["thread_id"])["modeling_mode"] == "mmm"
        rows = store.list_sessions()
        assert all(r["modeling_mode"] == "mmm" for r in rows)
