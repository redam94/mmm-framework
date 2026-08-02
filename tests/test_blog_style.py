"""Gate the blog prose style spec (``technical-docs/writing-guide.md``).

Two things are tested here:

1. every page in ``docs/blog*.html`` satisfies the machine-checkable invariants, and
2. the checker itself actually catches each violation class, and does not fire on the constructs
   that legitimately look like violations.

(2) matters more than it sounds. During the 2026-08-02 style pass the checker produced false
failures three separate times (apostrophes pairing across JSON-LD strings, a case change after a
colon became a period, trailing punctuation attaching to a number), and each one cost a round of
investigation. A checker nobody trusts gets switched off.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
CHECKER = DOCS / "tools" / "check_blog_style.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_blog_style", CHECKER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


cbs = _load()

BLOG_PAGES = sorted(DOCS.glob("blog*.html"))


def test_blog_pages_exist() -> None:
    assert (
        len(BLOG_PAGES) >= 40
    ), f"expected the full essay series, found {len(BLOG_PAGES)}"
    assert DOCS / "blog.html" in BLOG_PAGES


@pytest.mark.parametrize("page", BLOG_PAGES, ids=lambda p: p.name)
def test_page_satisfies_style_invariants(page: Path) -> None:
    errors, _ = cbs.check_invariants(page)
    assert not errors, f"{page.name}:\n  " + "\n  ".join(errors)


def test_writing_guide_is_present_and_referenced() -> None:
    """The spec has to be findable, or the next writer will not follow it."""
    guide = REPO / "technical-docs" / "writing-guide.md"
    assert (
        guide.exists()
    ), "technical-docs/writing-guide.md is the spec these tests enforce"
    body = guide.read_text(encoding="utf-8")
    for rule in ("No em dashes", "negation pivot", "Semicolons"):
        assert rule in body, f"writing-guide.md no longer states the {rule!r} rule"


# --------------------------------------------------------------------------------------
# The checker catches what it claims to catch.
# --------------------------------------------------------------------------------------

MINIMAL = """<!doctype html>
<html><head>
<meta name="description" content="A search blurb.">
<meta property="og:description" content="A social blurb.">
<meta name="twitter:description" content="A social blurb.">
</head><body>
<aside class="sidebar"><ul><li><a href="#one">One</a></li></ul></aside>
<main class="main-content">
<h1 id="one">One</h1>
<p>A paragraph of ordinary prose that states a claim and then stops.</p>
<pre><code># illustrative
model.fit(draws=1000)</code></pre>
<p>Inline math \\( x \\gt 1 \\) and a display block.</p>
$$ y = \\alpha \\; x $$
<p>Costs 5 dollars and change, per the 2026 report (Miao&ndash;Geng, 2018).</p>
</main></body></html>
"""


def _page(tmp_path: Path, body: str = MINIMAL, name: str = "blog-fixture.html") -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_clean_fixture_passes(tmp_path: Path) -> None:
    errors, _ = cbs.check_invariants(_page(tmp_path))
    assert not errors, errors


@pytest.mark.parametrize(
    "swap_in, expect",
    [
        ("<p>A claim &mdash; and its aside.</p>", "em dash"),
        ("<p>A claim — and its aside.</p>", "em dash"),
        ("<p>A claim &#8212; and its aside.</p>", "em dash"),
        ('<script>var t = "Figure 1 — the ladder";</script>', "em dash"),
        ("<p>Let $\\alpha_k$ be the coefficient.</p>", "single-$"),
        ('<p>See <a href="#nowhere">below</a>.</p>', "dangling in-page anchor"),
        # Every spelling below was confirmed to bypass an earlier version of the checker.
        # Browsers accept hex, zero-padding, and a missing closing semicolon.
        ("<p>A claim &#x2014; and its aside.</p>", "em dash"),
        ("<p>A claim &#X2014; and its aside.</p>", "em dash"),
        ("<p>A claim &#08212; and its aside.</p>", "em dash"),
        ("<p>A claim &#8212 and its aside.</p>", "em dash"),
        ("<p>A claim &mdash and its aside.</p>", "em dash"),
        # Codepoints that render indistinguishably from an em dash.
        ("<p>A claim ― and its aside.</p>", "em dash"),  # U+2015 horizontal bar
        ("<p>A claim ⸺ and its aside.</p>", "em dash"),  # U+2E3A two-em dash
        ("<p>A claim ﹘ and its aside.</p>", "em dash"),  # U+FE58 small em dash
        # Forms that only become an em dash at runtime.
        ('<style>.x::after { content: "\\2014"; }</style>', "em dash"),
        ("<script>var t = '\\u2014';</script>", "em dash"),
        # Attribute values render too.
        ('<p><img src="a.png" alt="before &#x2014; after"></p>', "em dash"),
        ('<p><span title="a &#x2014; b">x</span></p>', "em dash"),
        # Entity-encoded semicolons must still count against the budget.
        ("<p>" + "one thing&#59; another. " * 30 + "</p>", "semicolon"),
        ("<p>" + "one thing&semi; another. " * 30 + "</p>", "semicolon"),
    ],
)
def test_violations_are_caught(tmp_path: Path, swap_in: str, expect: str) -> None:
    doc = MINIMAL.replace("</main>", f"{swap_in}\n</main>")
    errors, _ = cbs.check_invariants(_page(tmp_path, doc))
    assert any(expect in e for e in errors), f"expected {expect!r}, got {errors}"


def test_fake_main_close_does_not_truncate_the_scan(tmp_path: Path) -> None:
    """A literal ``</main>`` in a comment or JS string must not hide the rest of the body.

    A non-greedy match stopped there, so everything after it escaped the semicolon budget, the
    word counts, and the reading-time computation.
    """
    for decoy in ("<!-- </main> -->", "<script>var s = '</main>';</script>"):
        doc = MINIMAL.replace(
            "</main>", f"{decoy}\n<p>" + "one thing; another. " * 30 + "</p>\n</main>"
        )
        errors, _ = cbs.check_invariants(_page(tmp_path, doc))
        assert any(
            "semicolon" in e for e in errors
        ), f"decoy {decoy!r} hid the body: {errors}"


def test_em_dash_in_head_is_caught(tmp_path: Path) -> None:
    doc = MINIMAL.replace("A search blurb.", "A search blurb &mdash; with an aside.")
    errors, _ = cbs.check_invariants(_page(tmp_path, doc))
    assert any("em dash" in e for e in errors)


def test_og_twitter_description_mismatch_is_caught(tmp_path: Path) -> None:
    doc = MINIMAL.replace(
        '<meta name="twitter:description" content="A social blurb.">',
        '<meta name="twitter:description" content="A different blurb.">',
    )
    errors, _ = cbs.check_invariants(_page(tmp_path, doc))
    assert any("twitter:description" in e for e in errors)


def test_semicolon_flood_is_caught(tmp_path: Path) -> None:
    doc = MINIMAL.replace(
        "</main>", "<p>" + "one thing; another thing. " * 30 + "</p>\n</main>"
    )
    errors, _ = cbs.check_invariants(_page(tmp_path, doc))
    assert any("semicolon" in e for e in errors)


# --------------------------------------------------------------------------------------
# The checker does NOT fire on constructs that only look like violations. Each of these was
# an actual false positive at some point.
# --------------------------------------------------------------------------------------


def test_en_dashes_are_left_alone(tmp_path: Path) -> None:
    """En dashes do real work in number ranges and hyphenated author pairs."""
    doc = MINIMAL.replace(
        "</main>",
        "<p>Rambachan&ndash;Roth (2023&ndash;26), pp. 19&ndash;46.</p>\n</main>",
    )
    errors, _ = cbs.check_invariants(_page(tmp_path, doc))
    assert not errors, errors


def test_latex_spacing_commands_are_not_semicolons(tmp_path: Path) -> None:
    """``\\;`` is math spacing. Counting it as punctuation puts every post 10x over budget."""
    doc = MINIMAL.replace(
        "</main>", "<p>Then</p>\n" + "$$ a \\; b \\; c $$\n" * 12 + "</main>"
    )
    errors, _ = cbs.check_invariants(_page(tmp_path, doc))
    assert not errors, errors


def test_dollar_amounts_do_not_read_as_math(tmp_path: Path) -> None:
    """Two prices in one sentence look exactly like a $...$ span. This corpus is about money."""
    doc = MINIMAL.replace(
        "</main>",
        "<p>It cost $40 and returned $95 in revenue, against a $1,200 budget.</p>\n</main>",
    )
    errors, _ = cbs.check_invariants(_page(tmp_path, doc))
    assert not errors, errors


def test_abbreviated_sidebar_labels_are_allowed(tmp_path: Path) -> None:
    """The TOC deliberately shortens headings: 28 headings across 9 pages do this on purpose."""
    doc = MINIMAL.replace('<a href="#one">One</a>', '<a href="#one">One, Briefly</a>')
    errors, _ = cbs.check_invariants(_page(tmp_path, doc))
    assert not errors, errors


def test_search_and_social_descriptions_may_differ(tmp_path: Path) -> None:
    """`description` is search copy and `og:description` is social copy; only og == twitter."""
    errors, _ = cbs.check_invariants(_page(tmp_path))
    assert not any("description" in e for e in errors), errors


# --------------------------------------------------------------------------------------
# Migration mode.
# --------------------------------------------------------------------------------------


def test_prose_helpers_exclude_code_and_math() -> None:
    text = cbs.prose_text(MINIMAL)
    assert "model.fit" not in text, "code blocks must not count as prose"
    assert "alpha" not in text, "math must not count as prose"
    assert "ordinary prose" in text


def test_number_tokens_ignore_trailing_punctuation() -> None:
    """ "0.37 &mdash; x" becoming "0.37, x" must not read as a changed figure."""
    a = cbs._numbers("<main><p>retention 0.37 &mdash; paid search</p></main>")
    b = cbs._numbers("<main><p>retention 0.37, paid search</p></main>")
    assert a == b


def test_js_skeleton_survives_apostrophes_in_json_ld() -> None:
    """Blanking single-quoted strings first pairs `platform's` across the whole document."""
    doc = (
        '<script type="application/ld+json">{"description": "the platform\'s pacing system"}</script>'
        "<script>var keep = compute(1, 2, 3);</script>"
    )
    assert "compute(1, 2, 3)" in cbs._js_skeleton(doc)


def test_chip_words_uses_textcontent_semantics() -> None:
    """Tags contribute nothing; they are not separators.

    ``1,198<strong>%</strong>`` is one token in the browser. Substituting a space per tag makes it
    two, which moved the displayed reading time on 5 of 42 pages.
    """
    assert cbs.chip_words("<main><p>1,198<strong>%</strong> lift</p></main>") == 2
    assert cbs.chip_words("<main><p>one two three</p></main>") == 3


def test_prose_checks_see_a_page_with_no_main_element() -> None:
    """blog.html is <section> blocks with no <main>; a <main>-only extractor returned ""."""
    doc = "<html><head></head><body><section><p>one two three four</p></section></body></html>"
    assert cbs.prose_text(doc).split() == ["one", "two", "three", "four"]
    index_prose = cbs.prose_text((DOCS / "blog.html").read_text(encoding="utf-8"))
    assert (
        len(index_prose.split()) > 500
    ), "the index page's prose must actually be scanned"


def test_structure_check_is_not_dead_code(tmp_path: Path) -> None:
    """HTMLParser.error() has not been called by the stdlib since 3.5, so overriding it caught
    nothing. Structural problems must be detected by tracking the tags these checks depend on.
    """
    doc = MINIMAL.replace("</main>", "")  # unclosed <main>
    errors, _ = cbs.check_invariants(_page(tmp_path, doc))
    assert any("main" in e for e in errors), errors
    two = MINIMAL.replace('<main class="main-content">', "<main><main>")
    errors, _ = cbs.check_invariants(_page(tmp_path, two))
    assert any("main" in e for e in errors), errors


def test_migration_mode_does_not_mask_a_sign_flip() -> None:
    """The em-dash normaliser must not neutralise '-', or a frozen code block can change value."""
    frozen = "<pre><code>roi = -0.5</code></pre>"
    flipped = "<pre><code>roi = 0.5</code></pre>"
    assert cbs._pre_blocks(frozen) != cbs._pre_blocks(flipped)
    # The normaliser is what decides whether a difference is waved through as punctuation-only.
    norm = lambda s: re.sub(r"[\s,.;:]+", "", re.sub(cbs.EM_DASH, " ", s)).casefold()
    assert norm(frozen) != norm(flipped), "sign flip must survive normalisation"


def test_reading_times_on_index_match_runtime_computation() -> None:
    """Nothing else catches this; it had drifted on 28 of 42 cards."""
    index = DOCS / "blog.html"
    errors = cbs._check_reading_times(index.read_text(encoding="utf-8"), DOCS)
    assert not errors, "\n  ".join(errors)


def test_index_card_count_matches_essay_count() -> None:
    body = (DOCS / "blog.html").read_text(encoding="utf-8")
    cards = set(re.findall(r'<a class="post-card" href="(blog-[^"]+)"', body))
    essays = {p.name for p in BLOG_PAGES if p.name != "blog.html"}
    assert cards == essays, f"index missing {essays - cards}, stale {cards - essays}"
