"""P2 — the generate_model_defense_report agent tool (glue over the tested core).

The full refutation run needs a fitted model (slow); here we cover the wiring:
the tool is registered and its no-model guard returns a clear message.
"""

from __future__ import annotations


def test_tool_is_registered():
    from mmm_framework.agents import tools

    names = [getattr(t, "name", getattr(t, "__name__", "")) for t in tools.TOOLS]
    assert "generate_model_defense_report" in names


def test_no_fitted_model_returns_clear_message():
    from mmm_framework.agents.tools import _MODEL_CACHE, generate_model_defense_report

    _MODEL_CACHE.pop("fitted_model", None)
    cmd = generate_model_defense_report.func(
        state={"dashboard_data": {}},
        report_title=None,
        tool_call_id="t1",
        config=None,
    )
    msg = cmd.update["messages"][0].content
    assert "No fitted model" in msg
