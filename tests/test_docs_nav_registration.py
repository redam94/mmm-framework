"""Docs-site navigation registration gate.

``docs/shared/components.js`` holds two hand-maintained registration surfaces
that the site renders from:

* ``SERIES`` — ordered page lists that drive the Previous/Next cards at the
  foot of every page in a series.
* ``PAGE_TIERS`` — the audience chip (overview / analyst / technical).

Both fail **silently**: a page missing from ``SERIES`` renders with no footer
navigation at all, and one missing from ``PAGE_TIERS`` renders with no chip.
Nothing looks broken, so nothing gets noticed — 7 of the then-27 research posts
had drifted out of ``SERIES`` before anyone spotted it (#175). Every other
registration surface either is generated (``search-index.json``, ``sitemap.xml``)
or fails visibly (a post missing from ``blog.html`` vanishes from the index).

These tests are the missing signal. They parse the two JavaScript object
literals with regexes — both are flat lists of quoted strings, so no JS engine
is needed — and a parser guard fails loudly if a refactor ever makes that
parse return nothing, rather than letting the checks pass vacuously.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
COMPONENTS_JS = DOCS_DIR / "shared" / "components.js"
BLOG_INDEX = DOCS_DIR / "blog.html"

#: The series that indexes the research blog; its order must match blog.html.
BLOG_SERIES = "Measurement research"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _js_object_literal(source: str, name: str) -> str:
    """Return the ``{...}`` body of ``const <name> = {...}`` by brace matching."""
    start = source.index(f"const {name} = {{")
    open_idx = source.index("{", start)
    depth = 0
    for i in range(open_idx, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[open_idx : i + 1]
    raise AssertionError(f"unterminated object literal for {name} in {COMPONENTS_JS}")


_SERIES_KEY_RE = re.compile(r"^\s+'([^']+)':\s*\[", re.M)
_ENTRY_RE = re.compile(r"\[\s*'([^']+)'\s*,\s*'((?:[^'\\]|\\.)*)'\s*\]")
_TIER_RE = re.compile(r"'([^']+\.html)':\s*TIER_(\w+)")


def _series() -> dict[str, list[tuple[str, str]]]:
    """``{series name: [(href, title), ...]}`` in declaration order."""
    block = _js_object_literal(COMPONENTS_JS.read_text(encoding="utf-8"), "SERIES")
    keys = [(m.start(), m.group(1)) for m in _SERIES_KEY_RE.finditer(block)]
    out: dict[str, list[tuple[str, str]]] = {}
    for i, (pos, name) in enumerate(keys):
        end = keys[i + 1][0] if i + 1 < len(keys) else len(block)
        out[name] = _ENTRY_RE.findall(block[pos:end])
    return out


def _page_tiers() -> dict[str, str]:
    block = _js_object_literal(COMPONENTS_JS.read_text(encoding="utf-8"), "PAGE_TIERS")
    return dict(_TIER_RE.findall(block))


def _blog_pages() -> list[str]:
    return sorted(p.name for p in DOCS_DIR.glob("blog-*.html") if p.name != "blog.html")


def _blog_index_order() -> list[str]:
    """Post hrefs in the order their cards appear on ``blog.html``."""
    html = BLOG_INDEX.read_text(encoding="utf-8")
    seen: list[str] = []
    for href in re.findall(r'<a class="post-card"[^>]*href="(blog-[^"]+\.html)"', html):
        if href not in seen:
            seen.append(href)
    return seen


# ---------------------------------------------------------------------------
# Parser guards — these checks must never pass vacuously
# ---------------------------------------------------------------------------


def test_components_js_parses() -> None:
    series = _series()
    assert len(series) >= 5, f"only parsed {len(series)} series — parser drifted?"
    assert BLOG_SERIES in series, f"{BLOG_SERIES!r} not found in SERIES"
    assert all(pages for pages in series.values()), "a series parsed as empty"
    assert len(_page_tiers()) >= 20, "PAGE_TIERS parsed suspiciously small"
    assert len(_blog_pages()) >= 20, "blog-*.html glob found suspiciously few posts"
    assert len(_blog_index_order()) >= 20, "no post cards parsed from blog.html"


# ---------------------------------------------------------------------------
# SERIES
# ---------------------------------------------------------------------------


def test_every_series_entry_points_at_an_existing_page() -> None:
    missing = [
        f"{name}: {href}"
        for name, pages in _series().items()
        for href, _ in pages
        if not (DOCS_DIR / href).exists()
    ]
    assert not missing, "SERIES references pages that do not exist:\n  " + "\n  ".join(
        missing
    )


def test_no_page_appears_in_two_series() -> None:
    """``initSeriesNav`` returns after the first match, so a duplicate means the
    later series' navigation silently never renders for that page."""
    seen: dict[str, str] = {}
    dupes = []
    for name, pages in _series().items():
        for href, _ in pages:
            if href in seen:
                dupes.append(f"{href} in both {seen[href]!r} and {name!r}")
            seen[href] = name
    assert not dupes, "\n  ".join(dupes)


def test_every_blog_post_is_registered_in_the_series() -> None:
    """The #175 failure: a post outside SERIES renders no prev/next cards."""
    registered = [href for href, _ in _series()[BLOG_SERIES]]
    missing = sorted(set(_blog_pages()) - set(registered))
    assert not missing, (
        f"blog posts missing from SERIES[{BLOG_SERIES!r}] in docs/shared/components.js "
        "— they will render with no Previous/Next cards:\n  " + "\n  ".join(missing)
    )


def test_series_has_no_duplicate_entries() -> None:
    for name, pages in _series().items():
        hrefs = [href for href, _ in pages]
        dupes = {h for h in hrefs if hrefs.count(h) > 1}
        assert not dupes, f"{name!r} lists {sorted(dupes)} more than once"


def test_series_order_matches_the_blog_index() -> None:
    """Prev/Next must walk the posts in the order the index presents them."""
    series_order = [href for href, _ in _series()[BLOG_SERIES]]
    index_order = _blog_index_order()
    assert series_order == index_order, (
        "SERIES order disagrees with the blog.html card order.\n"
        f"  SERIES: {series_order}\n"
        f"  blog.html: {index_order}"
    )


def test_series_titles_are_not_empty() -> None:
    empty = [
        f"{name}: {href}"
        for name, pages in _series().items()
        for href, title in pages
        if not title.strip()
    ]
    assert not empty, "series entries with a blank title:\n  " + "\n  ".join(empty)


# ---------------------------------------------------------------------------
# PAGE_TIERS
# ---------------------------------------------------------------------------


def test_every_tier_entry_points_at_an_existing_page() -> None:
    missing = sorted(p for p in _page_tiers() if not (DOCS_DIR / p).exists())
    assert (
        not missing
    ), "PAGE_TIERS references pages that do not exist:\n  " + "\n  ".join(missing)


def test_tier_values_are_known() -> None:
    unknown = {p: t for p, t in _page_tiers().items() if t not in _KNOWN_TIERS}
    assert not unknown, f"unknown tier constants: {unknown}"


_KNOWN_TIERS = {"OVERVIEW", "ANALYST", "TECHNICAL"}


def test_every_blog_post_has_an_audience_tier() -> None:
    """Same silent failure as SERIES: no entry means no chip and no reading time."""
    tiers = _page_tiers()
    missing = sorted(p for p in _blog_pages() if p not in tiers)
    assert not missing, (
        "blog posts missing from PAGE_TIERS in docs/shared/components.js "
        "— they will render with no audience chip:\n  " + "\n  ".join(missing)
    )


@pytest.mark.parametrize("page", _blog_index_order())
def test_every_indexed_post_exists(page: str) -> None:
    """A card on blog.html pointing at a deleted/renamed post is a 404."""
    assert (DOCS_DIR / page).exists(), f"blog.html links to missing page {page}"


# ---------------------------------------------------------------------------
# NAV_GROUPS registration (#228) — a page in the nav must exist on disk and
# carry an audience tier. Before this gate, a new page added to NAV_GROUPS
# without a PAGE_TIERS entry rendered with no chip and nothing failed.
# ---------------------------------------------------------------------------

_NAV_HREF_RE = re.compile(r"href:\s*'([^']+\.html)'")


def _nav_pages() -> list[str]:
    """Every page referenced in NAV_GROUPS (brace-matched array literal)."""
    source = COMPONENTS_JS.read_text(encoding="utf-8")
    start = source.index("const NAV_GROUPS = [")
    open_idx = source.index("[", start)
    depth = 0
    for i in range(open_idx, len(source)):
        if source[i] == "[":
            depth += 1
        elif source[i] == "]":
            depth -= 1
            if depth == 0:
                body = source[open_idx : i + 1]
                return _NAV_HREF_RE.findall(body)
    raise AssertionError(f"unterminated NAV_GROUPS array in {COMPONENTS_JS}")


#: Pages allowed in the nav without a tier chip, each with the reason.
_NO_TIER_OK = {
    "index.html": "the landing page renders its own hero, not a doc chip",
    "artifacts/index.html": (
        "the baked-artifacts gallery lives in a subdirectory; PAGE_TIERS "
        "keys are flat docs/*.html filenames and the chip renderer resolves "
        "by basename"
    ),
}


def test_nav_groups_parses() -> None:
    """Parser guard: an empty parse must fail loudly, not pass vacuously."""
    pages = _nav_pages()
    assert len(pages) >= 30, f"NAV_GROUPS parse looks broken: {len(pages)} pages"


def test_every_nav_page_exists_on_disk() -> None:
    missing = sorted(p for p in set(_nav_pages()) if not (DOCS_DIR / p).exists())
    assert (
        not missing
    ), "NAV_GROUPS links to pages that do not exist:\n  " + "\n  ".join(missing)


def test_every_nav_page_has_an_audience_tier() -> None:
    tiers = _page_tiers()
    missing = sorted(
        p for p in set(_nav_pages()) if p not in tiers and p not in _NO_TIER_OK
    )
    assert not missing, (
        "pages in NAV_GROUPS missing from PAGE_TIERS in "
        "docs/shared/components.js — they render with no audience chip:\n  "
        + "\n  ".join(missing)
    )
