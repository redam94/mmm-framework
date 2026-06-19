"""P2 — model-defense (causal-rigor) report builder + renderer."""

from __future__ import annotations

from html.parser import HTMLParser

from mmm_framework.reporting.model_defense import (
    build_model_defense,
    model_defense_report,
    render_model_defense_html,
)


def _t(name, kind, passed, channel=None):
    return {
        "name": name,
        "kind": kind,
        "passed": passed,
        "channel": channel,
        "description": "",
        "original_effect": 1.0,
        "refuted_effect": 0.0 if kind == "vanish" else 1.0,
    }


def _ref(tests, underpowered=False):
    n_passed = sum(1 for t in tests if t["passed"])
    return {
        "tests": tests,
        "n_passed": n_passed,
        "n_failed": len(tests) - n_passed,
        "all_passed": all(t["passed"] for t in tests),
        "underpowered": underpowered,
    }


def test_robust_when_all_pass_and_converged():
    p = build_model_defense(
        _ref([_t("placebo", "vanish", True), _t("subset", "stable", True)]),
        convergence={"divergences": 0, "rhat_max": 1.005},
    )
    assert p["level"] == "strong"
    assert p["n_passed"] == 2 and p["n_failed"] == 0
    assert p["convergence"]["ok"] is True


def test_one_failure_is_qualified():
    p = build_model_defense(_ref([_t("a", "vanish", True), _t("b", "stable", False)]))
    assert p["level"] == "qualified"
    assert "one test" in p["verdict"]


def test_two_failures_is_caution():
    p = build_model_defense(_ref([_t("a", "vanish", False), _t("b", "stable", False)]))
    assert p["level"] == "caution"


def test_underpowered_pass_downgrades_to_qualified():
    p = build_model_defense(_ref([_t("a", "vanish", True)], underpowered=True))
    assert p["level"] == "qualified"


def test_nonconvergence_dominates_even_when_tests_pass():
    p = build_model_defense(
        _ref([_t("a", "vanish", True), _t("b", "stable", True)]),
        convergence={"divergences": 12, "rhat_max": 1.05},
    )
    assert p["level"] == "caution"
    assert p["convergence"]["ok"] is False


def test_calibration_caveat_present():
    p0 = build_model_defense(_ref([_t("a", "vanish", True)]))
    assert any("not anchored to a randomized experiment" in c for c in p0["caveats"])
    p1 = build_model_defense(
        _ref([_t("a", "vanish", True)]), n_calibrated_experiments=2
    )
    assert any("anchored to 2 real experiments" in c for c in p1["caveats"])


def test_empty_suite_is_unknown():
    assert build_model_defense({"tests": []})["level"] == "unknown"


def _unclosed(html: str) -> int:
    VOID = {
        "br",
        "img",
        "meta",
        "link",
        "input",
        "hr",
        "source",
        "path",
        "circle",
        "rect",
        "line",
        "polyline",
        "area",
        "col",
        "use",
    }

    class P(HTMLParser):
        def __init__(s):
            super().__init__()
            s.st = []

        def handle_starttag(s, t, a):
            if t not in VOID:
                s.st.append(t)

        def handle_endtag(s, t):
            if s.st and s.st[-1] == t:
                s.st.pop()
            elif t in s.st:
                while s.st and s.st.pop() != t:
                    pass

    p = P()
    p.feed(html)
    return len(p.st)


def test_render_html_wellformed_and_contains_verdict():
    p = build_model_defense(
        _ref([_t("placebo", "vanish", True, "TV"), _t("subset", "stable", False)]),
        convergence={"divergences": 0, "rhat_max": 1.008},
    )
    html = render_model_defense_html(p, title="Trust report")
    assert "Trust report" in html
    assert "Mostly robust" in html  # 1-failure verdict
    assert "1/2 causal refutation tests passed" in html
    assert _unclosed(html) == 0


def test_convenience_one_call():
    html = model_defense_report(_ref([_t("a", "vanish", True)]), title="X")
    assert "<html" in html and "Robust" in html
    assert _unclosed(html) == 0
