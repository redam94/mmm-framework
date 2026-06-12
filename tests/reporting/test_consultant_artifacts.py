"""Tests for the consultant artifacts generator (fast, no model required)."""

from __future__ import annotations

import re

import pytest

from mmm_framework.reporting import (
    ARTIFACTS,
    ArtifactSpec,
    render_artifact,
    write_all,
)
from mmm_framework.reporting.consultant_artifacts import _build_index

EXPECTED_NAMES = {
    "diagnostic_checklist",
    "preregistration_memo",
    "data_onboarding_checklist",
    "exec_summary_template",
    "engagement_timeline",
}


def _assert_tag_balance(html: str) -> None:
    """Simple open/close tag count balance for the non-void tags we emit."""
    for tag in (
        "div",
        "span",
        "section",
        "table",
        "thead",
        "tbody",
        "tr",
        "td",
        "th",
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "ol",
        "li",
        "a",
        "footer",
        "header",
        "html",
        "body",
        "head",
    ):
        opens = len(re.findall(rf"<{tag}[\s>]", html))
        closes = html.count(f"</{tag}>")
        assert opens == closes, f"<{tag}> imbalance: {opens} open vs {closes} close"


class TestRegistry:
    def test_registry_names(self):
        assert set(ARTIFACTS) == EXPECTED_NAMES

    def test_specs_have_title_description_builder(self):
        for name, spec in ARTIFACTS.items():
            assert isinstance(spec, ArtifactSpec)
            assert spec.title
            assert spec.description
            assert callable(spec.build)
            assert spec.filename(name).endswith(".html")
            assert "_" not in spec.filename(name)

    def test_unknown_artifact_raises(self):
        with pytest.raises(KeyError, match="Unknown artifact"):
            render_artifact("nonexistent")


class TestRendering:
    @pytest.mark.parametrize("name", sorted(EXPECTED_NAMES))
    def test_renders_standalone_document(self, name):
        html = render_artifact(name)
        spec = ARTIFACTS[name]
        assert html.startswith("<!DOCTYPE html>")
        # Masthead
        assert "MMM Framework — Consultant Artifact" in html
        assert f"<h1>{spec.title}</h1>" in html
        # Version/date line
        assert re.search(r"v\d+\.\d+\.\d+ · \d{4}-\d{2}-\d{2}", html)
        # Colophon
        assert "Built with the MMM Framework" in html
        assert "Apache-2.0" in html
        assert "github.com/redam94/mmm-framework" in html
        assert "python -m mmm_framework.reporting.consultant_artifacts" in html

    @pytest.mark.parametrize("name", sorted(EXPECTED_NAMES))
    def test_tag_balance(self, name):
        _assert_tag_balance(render_artifact(name))

    def test_index_tag_balance_and_links(self):
        html = _build_index()
        _assert_tag_balance(html)
        for name, spec in ARTIFACTS.items():
            assert spec.filename(name) in html
            assert spec.title in html

    @pytest.mark.parametrize("name", sorted(EXPECTED_NAMES))
    def test_print_styles_present(self, name):
        html = render_artifact(name)
        assert "@page" in html
        assert "@media print" in html
        assert "break-inside: avoid" in html


class TestContentMarkers:
    def test_diagnostic_checklist_content(self):
        html = render_artifact("diagnostic_checklist")
        # Decision-table rows (faithful to stress-05 gauntlet)
        assert "Divergences / r-hat" in html
        assert "the combined effect is identified, the split isn't" in html
        assert "Never read a green PPC as causal validation" in html
        assert "never let media absorb a break" in html
        assert "The proxy control is mandatory; calibrate the channel" in html
        # EDA pre-flight
        assert "Outlier screen" in html
        assert "Demand-chasing screen" in html
        # Fix ladder
        assert "Estimand first" in html
        assert "Experiments calibrate what structure can" in html
        # Doctrine
        assert "never the" in html and "attribution" in html
        # Printable checkbox span, not emoji
        assert 'class="check-box"' in html
        assert "☐" not in html

    def test_preregistration_memo_content(self):
        html = render_artifact("preregistration_memo")
        assert "power-ceiling check" in html.lower()
        assert "Matched-market difference-in-differences" in html
        assert "Randomized matched-pair geo lift" in html
        assert "Intention-to-treat" in html
        assert "amendments" in html
        assert "Stopping rule" in html
        assert "fill-field" in html
        assert "signature-line" in html

    def test_data_onboarding_content(self):
        html = render_artifact("data_onboarding_checklist")
        assert "VariableName" in html and "VariableValue" in html
        assert "104+" in html
        assert "50% zero-spend" in html
        assert "do(spend = 0)" in html
        assert "forward-fill" in html
        assert "PII" in html

    def test_exec_summary_content(self):
        html = render_artifact("exec_summary_template")
        assert "Evidence tier" in html
        assert "Validated" in html
        assert "No naked point estimates" in html
        assert "What would change our mind" in html
        assert "80% CI" in html
        assert "fill-field" in html

    def test_engagement_timeline_content(self):
        html = render_artifact("engagement_timeline")
        assert "W1–2" in html and "W12" in html
        assert "EIG/EVOI" in html
        assert "Pre-Registration Memo" in html
        assert "The loop continues" in html
        assert "re-evaluate" in html


class TestWriteAll:
    def test_write_all_writes_five_plus_index(self, tmp_path):
        paths = write_all(tmp_path / "artifacts")
        assert len(paths) == 6
        names = {p.name for p in paths}
        assert "index.html" in names
        assert "diagnostic-checklist.html" in names
        assert "preregistration-memo.html" in names
        assert "data-onboarding-checklist.html" in names
        assert "exec-summary-template.html" in names
        assert "engagement-timeline.html" in names
        for p in paths:
            assert p.exists()
            content = p.read_text(encoding="utf-8")
            assert content.startswith("<!DOCTYPE html>")
            assert len(content) > 2000
