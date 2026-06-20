"""Tests that the Model Garden agent surface is wired correctly: the ops are
registered in the OPS dict with the right flags + payload shape, the tools are
registered in the right tiers, and the static (no-exec) source class finder
handles the common cases. All fast — no fitting."""

from __future__ import annotations

from mmm_framework.agents import model_ops, tools


class TestOpsRegistration:
    def test_garden_ops_registered(self):
        assert "garden_compat" in model_ops.OPS
        assert "garden_tune_suggestions" in model_ops.OPS

    def test_garden_compat_allows_unfitted(self):
        assert getattr(model_ops.OPS["garden_compat"], "allow_unfitted", False) is True

    def test_garden_compat_requires_source_path(self):
        res = model_ops.garden_compat(source_path=None)
        assert res["error"] and "source_path" in res["error"]

    def test_tune_suggestions_needs_model(self):
        res = model_ops.garden_tune_suggestions(None)
        assert res["error"] == model_ops.NO_MODEL_MSG


class TestToolRegistration:
    GARDEN_TOOLS = (
        "register_garden_model",
        "list_garden_models",
        "load_garden_model",
        "test_garden_model",
        "publish_garden_model",
        "suggest_model_improvements",
    )

    def test_all_in_tools_and_expert(self):
        names = {t.name for t in tools.TOOLS}
        expert = {t.name for t in tools.EXPERT_TOOLS}
        for g in self.GARDEN_TOOLS:
            assert g in names, g
            assert g in expert, g

    def test_test_garden_model_is_heavy_and_delegated(self):
        assert "test_garden_model" in tools.HEAVY_TOOL_NAMES
        orch = {t.name for t in tools.ORCHESTRATOR_TOOLS}
        # heavy tools are removed from the fast orchestrator tier
        assert "test_garden_model" not in orch
        # light garden tools remain available to the orchestrator
        assert "list_garden_models" in orch


class TestStaticClassFinder:
    def test_single_class(self):
        name, err = tools._garden_static_class_name(
            "from mmm_framework.garden import CustomMMM\nclass A(CustomMMM): pass"
        )
        assert name == "A" and err is None

    def test_explicit_marker(self):
        name, err = tools._garden_static_class_name(
            "class A: pass\nclass B: pass\nGARDEN_MODEL = B"
        )
        assert name == "B" and err is None

    def test_ambiguous(self):
        name, err = tools._garden_static_class_name("class A: pass\nclass B: pass")
        assert name is None and "multiple" in err

    def test_no_class(self):
        name, err = tools._garden_static_class_name("x = 1")
        assert name is None and "no class" in err

    def test_syntax_error(self):
        name, err = tools._garden_static_class_name("class A(:")
        assert name is None and "syntax" in err.lower()
