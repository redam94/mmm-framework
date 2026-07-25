"""Idempotency contract for the docs SEO builder (``docs/tools/build_seo.py``).

The tool injects a JSON-LD block carrying ``dateModified`` into every docs page.
It used to derive that date from ``git log -1 -- <path>`` and rewrite every page
on every run, which self-perpetuates: the run rewrites a page, the rewrite gets
committed, the commit moves the file's git date, and the next run rewrites it
again. Editing one page therefore produced date-only diffs on ~70 others, which
every contributor had to notice and hand-revert (#174).

The contract these tests pin:

* a second consecutive run on an unchanged tree writes **nothing**;
* editing one page re-dates **that page only**;
* the date is a recorded fact (``docs/shared/seo-manifest.json``), so it does
  not drift with unrelated commits.

The module chdir's to ``docs/`` at import time, so the fixture restores the
process cwd and each test runs the builder inside its own ``tmp_path``.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

BUILD_SEO = Path(__file__).resolve().parents[1] / "docs" / "tools" / "build_seo.py"

PAGE = """<!DOCTYPE html>
<html><head>
<title>Example Page — MMM Framework</title>
<meta name="description" content="A page used by the SEO builder tests.">
</head>
<body><h1>Example</h1><p>{body}</p></body></html>
"""


@pytest.fixture(scope="module")
def seo():
    """Import build_seo.py, restoring the cwd it changes on import."""
    cwd = os.getcwd()
    try:
        spec = importlib.util.spec_from_file_location("build_seo", BUILD_SEO)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)
    return module


@pytest.fixture
def site(tmp_path, monkeypatch, seo):
    """A throwaway docs directory the builder can run against."""
    (tmp_path / "example.html").write_text(PAGE.format(body="first"), encoding="utf-8")
    (tmp_path / "shared").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(seo, "MANIFEST_PATH", "shared/seo-manifest.json")
    return tmp_path


def _run(seo, manifest):
    seo.augment_pages(manifest)
    return manifest


class TestIdempotency:
    def test_second_consecutive_run_writes_nothing(self, seo, site):
        """The issue's acceptance criterion."""
        manifest = _run(seo, {})
        after_first = (site / "example.html").read_text()
        mtime = (site / "example.html").stat().st_mtime_ns

        _run(seo, manifest)
        assert (site / "example.html").read_text() == after_first
        assert (site / "example.html").stat().st_mtime_ns == mtime, (
            "the page was rewritten byte-identically — a no-op run must not "
            "touch the file at all"
        )

    def test_unchanged_page_keeps_its_date_when_today_moves(self, seo, site):
        """A page nobody edited must not be re-dated just because time passed
        (the mechanism by which unrelated commits used to churn ~70 pages)."""
        manifest = _run(seo, {})
        first_date = manifest["example.html"]["modified"]

        monkey_tomorrow = "2099-01-01"
        original, seo.TODAY = seo.TODAY, monkey_tomorrow
        try:
            _run(seo, manifest)
        finally:
            seo.TODAY = original

        assert manifest["example.html"]["modified"] == first_date
        assert monkey_tomorrow not in (site / "example.html").read_text()

    def test_editing_one_page_re_dates_only_that_page(self, seo, site):
        (site / "other.html").write_text(PAGE.format(body="other"), encoding="utf-8")
        manifest = _run(seo, {})
        untouched_before = (site / "other.html").read_text()

        # A real content edit, then a run stamped with a later date.
        (site / "example.html").write_text(
            (site / "example.html").read_text().replace("first", "edited"),
            encoding="utf-8",
        )
        original, seo.TODAY = seo.TODAY, "2099-01-01"
        try:
            _run(seo, manifest)
        finally:
            seo.TODAY = original

        assert manifest["example.html"]["modified"] == "2099-01-01"
        assert '"dateModified": "2099-01-01"' in (site / "example.html").read_text()
        assert (site / "other.html").read_text() == untouched_before


class TestMigration:
    def test_first_run_adopts_the_date_already_on_the_page(self, seo, site):
        """Introducing the manifest must not re-date the whole site: a page that
        already carries a stamp keeps it even though the manifest is empty."""
        manifest = _run(seo, {})
        stamped = manifest["example.html"]["modified"]

        # Simulate a fresh checkout with no manifest but augmented pages.
        original, seo.TODAY = seo.TODAY, "2099-01-01"
        try:
            rebuilt: dict = {}
            _run(seo, rebuilt)
        finally:
            seo.TODAY = original

        assert rebuilt["example.html"]["modified"] == stamped


class TestHelpers:
    def test_content_hash_ignores_the_injected_block(self, seo, site):
        """The hash is taken over the stripped page, so re-injecting the block
        (with a different date) must not look like a content change."""
        raw = (site / "example.html").read_text()
        _run(seo, {})
        augmented = (site / "example.html").read_text()
        assert seo.SENTINEL in augmented
        assert seo.content_hash(seo.strip_block(augmented)) == seo.content_hash(raw)

    def test_stamped_date_reads_the_injected_value(self, seo, site):
        _run(seo, {})
        page = (site / "example.html").read_text()
        assert seo.stamped_date(page) is not None
        assert seo.stamped_date("<html>no block</html>") is None

    def test_write_if_changed_skips_identical_content(self, seo, tmp_path):
        target = tmp_path / "out.txt"
        assert seo.write_if_changed(str(target), "hello") is True
        assert seo.write_if_changed(str(target), "hello") is False
        assert seo.write_if_changed(str(target), "bye") is True
        assert target.read_text() == "bye"

    def test_today_is_derived_not_hardcoded(self, seo):
        """A hand-maintained date constant goes stale and mis-stamps new pages."""
        import datetime
        import re

        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", seo.TODAY)
        assert seo.TODAY >= "2026-07-25"
        source = BUILD_SEO.read_text()
        assert "datetime.date.today()" in source
        # Guard against a literal creeping back in as the default.
        assert not re.search(r'^TODAY = "\d{4}-\d{2}-\d{2}"', source, re.M)
        assert datetime.date.fromisoformat(seo.TODAY)


class TestManifestIsTracked:
    def test_manifest_exists_and_covers_the_real_site(self):
        """A missing/ignored manifest would silently restore the churn: every
        checkout would re-date every page on its first run."""
        import json

        docs = BUILD_SEO.parents[1]
        manifest = docs / "shared" / "seo-manifest.json"
        assert manifest.exists(), f"{manifest} is missing — run tools/build_seo.py"
        pages = json.loads(manifest.read_text())["pages"]
        html_pages = {
            p.name
            for p in docs.glob("*.html")
            if p.name not in {"TEMPLATE.html", "TEMPLATE-SIDEBAR.html", "404.html"}
        }
        missing = sorted(html_pages - set(pages))
        assert not missing, (
            "pages absent from the SEO manifest (they will be re-dated on the "
            "next build): " + ", ".join(missing)
        )
