#!/usr/bin/env python3
"""Enforce the blog prose style spec on docs/blog*.html.

The spec itself is ``technical-docs/writing-guide.md``. This script checks the parts of it a
machine can check, plus the repo-specific traps that a hand edit keeps falling into.

Run from anywhere::

    python3 docs/tools/check_blog_style.py                 # all blog pages
    python3 docs/tools/check_blog_style.py docs/blog-x.html
    python3 docs/tools/check_blog_style.py --advisory      # also print prose smells
    python3 docs/tools/check_blog_style.py --against HEAD  # migration mode, see below

**Default (invariant) mode** checks properties every blog page must always satisfy. It needs no
git history, so it works on a new post before it is committed. ``tests/test_blog_style.py`` gates
it in CI.

**Migration mode** (``--against <ref>``) additionally diffs each file against a git ref and fails
if anything that a *style* edit must never touch has moved: code blocks, math, headings, ids,
hrefs, citations, ``<time>``, JS logic, or word count beyond a tolerance. Use it while rewriting an
existing post. It is deliberately not part of the test suite, because legitimate content edits
change all of those things.

Rules that are advisory only (``--advisory``) are the ones whose regexes have real false positives:
"less X than Y" is a banned rhetorical pivot in "less a stance than a habit" and perfectly good
prose in "less lift than the geo test showed". A human decides those.

Known limits, so nobody mistakes a pass for a proof:

- The single-``$`` check only fires when the span contains a backslash, ``^`` or ``_``. A
  marker-free span like ``$P(x) = 1$`` is not caught. Tightening it further starts flagging real
  prose, because this corpus discusses budgets and a sentence with two dollar amounts in it looks
  identical to a math span.
- Nothing here judges meaning, rhythm, or whether a sentence reads well, which is most of the
  guide. A green run means no mechanical violation, not that the prose is good.
"""

from __future__ import annotations

import argparse
import html.parser
import re
import subprocess
import sys
from collections import Counter
from html import unescape
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"

# Words per minute used by shared/components.js::initPageMetaChips. The index cards in blog.html
# hard-code a reading time that must agree with what that function computes at runtime.
WORDS_PER_MINUTE = 220

# Prose semicolon budget from the guide is "roughly one per thousand words". The gate sits at 2.0
# so that a single unavoidable semicolon in a short post is not a build failure; the corpus at the
# time this was written peaked at 0.95.
SEMICOLON_PER_1K_LIMIT = 2.0

# Every spelling that renders as an em dash, or as a mark indistinguishable from one.
# Browsers accept numeric character references in decimal or hex, zero-padded, and (in most
# contexts) without the closing semicolon, so a checker that greps for `—` and `&mdash;` alone lets
# `&#x2014;` and `&#8212` straight through. `—` and the CSS `\2014` cover strings that only
# become an em dash at runtime.
EM_DASH = (
    r"[—―⸺⸻﹘︱]"  # em dash, horizontal bar, 2-em, 3-em, small
    r"|&mdash;?|&horbar;?"  # named, with or without the semicolon
    r"|&#0*8212;?|&#[xX]0*2014;?"  # decimal / hex numeric references
    r"|\\u0*2014|\\0*2014\b"  # JS and CSS escape sequences
)

ADVISORY = {
    "negation pivot": r"(isn't|is not|It's not|It is not) [^.<]{1,70}?(, it's|\. It's|, it is)",
    "not just X but": r"[Nn]ot just .{1,60}? but ",
    "less X than Y": r"\b[Ll]ess .{1,40}? than ",
    "hedge stack": r"worth noting|\bThat said\b|\bImportantly\b|While it's true",
    "announcement": r"Here's the thing|[Tt]he reality is\b|[Tt]he truth is\b|Let's be clear",
    "summary closer": r"\bUltimately\b|[Aa]t the end of the day",
    "throat clearing": r"In today's world|More than ever",
    "rhetorical transition": r"Why does this matter|What does this mean|So what does",
    "participial tail": r", (allowing|making|ensuring|highlighting) [a-z]",
    "banned vocab": (
        r"\b(delve|tapestry|realm|underscores?|testament to|pivotal|crucial|seamless|"
        r"leverages?|leveraging|foster|myriad|plethora|holistic|multifaceted|nuanced)\b"
    ),
}


class _TextContent(html.parser.HTMLParser):
    """Collect text the way the DOM's ``textContent`` does.

    Tags contribute *nothing*, they do not act as separators: ``1,198<strong>%</strong>`` is one
    token in the browser, and substituting a space for each tag turns it into two. That difference
    moved the displayed reading time on 5 of 42 pages.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    @property
    def text(self) -> str:
        return "".join(self.parts)


class _Structure(html.parser.HTMLParser):
    """Track whether the elements these checks depend on are present and balanced.

    ``HTMLParser.error()`` has not been called by the stdlib since Python 3.5, so overriding it to
    raise is dead code and every malformed page passes. This tracks the tags that actually matter
    here instead of pretending to validate HTML.
    """

    WATCH = ("main", "head", "body", "aside")

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.opened: Counter = Counter()
        self.closed: Counter = Counter()

    def handle_starttag(self, tag, attrs):
        if tag in self.WATCH:
            self.opened[tag] += 1

    def handle_endtag(self, tag):
        if tag in self.WATCH:
            self.closed[tag] += 1

    def problems(self) -> list[str]:
        out = []
        for tag in ("head", "body"):
            if self.opened[tag] != 1:
                out.append(f"expected exactly one <{tag}>, found {self.opened[tag]}")
            elif self.closed[tag] != 1:
                out.append(
                    f"<{tag}> is not closed exactly once (found {self.closed[tag]} </{tag}>)"
                )
        # <main> is optional: the index page is built from <section> blocks. When present it must
        # be singular and closed, because the prose extractor keys off it.
        if self.opened["main"] > 1:
            out.append(f"expected at most one <main>, found {self.opened['main']}")
        elif self.opened["main"] == 1 and self.closed["main"] != 1:
            out.append(
                f"<main> is not closed exactly once (found {self.closed['main']} </main>)"
            )
        if self.opened["aside"] != self.closed["aside"]:
            out.append("unbalanced <aside>")
        return out


def _main_of(doc: str) -> str:
    """The page's content element: <main> where there is one, otherwise <body>.

    The index page (``blog.html``) is built from ``<section>`` blocks with no ``<main>``, so a
    <main>-only extractor silently returned "" for it and every prose check on the index passed
    vacuously.

    Greedy, and with HTML comments removed first. A non-greedy match stops at the first literal
    ``</main>`` in the source, which can sit inside a comment or a JS string well before the real
    one; everything after it then becomes invisible to the semicolon budget and the word counts.
    """
    doc = re.sub(r"(?s)<!--.*?-->", " ", doc)
    for pattern in (r"(?s)<main\b.*</main>", r"(?s)<body\b.*</body>"):
        m = re.search(pattern, doc)
        if m:
            return m.group(0)
    return ""


def _strip_scripts(s: str) -> str:
    s = re.sub(r"(?s)<script.*?</script>", " ", s)
    return re.sub(r"(?s)<style.*?</style>", " ", s)


def prose_text(doc: str) -> str:
    """Human-readable body prose: no markup, no code, no math, no scripts."""
    m = _strip_scripts(_main_of(doc))
    m = re.sub(r"(?s)<pre.*?</pre>", " ", m)
    # LaTeX spacing commands (\; \,) are math, not punctuation, and would swamp any semicolon count.
    for pat in (r"(?s)\$\$.*?\$\$", r"(?s)\\\(.*?\\\)", r"(?s)\\\[.*?\\\]"):
        m = re.sub(pat, " ", m)
    t = re.sub(r"(?s)<[^>]+>", " ", m)
    # Decode entities rather than deleting them: blanking `&#59;` would hide a semicolon from the
    # budget, and blanking `&amp;` would silently shorten the word count. Tags are already gone, so
    # decoding cannot reintroduce markup.
    t = unescape(t)
    return re.sub(r"\s+", " ", t).strip()


def chip_words(doc: str) -> int:
    """Word count exactly as ``components.js::initPageMetaChips`` computes it.

    That function does ``content.textContent.trim().split(/\\s+/).length`` on ``.main-content``,
    so this must be textContent semantics, not a regex approximation. No page in the series puts a
    <script> inside <main>; if one ever did, textContent would include its source, and so does this.
    """
    p = _TextContent()
    p.feed(_main_of(doc))
    p.close()
    return len(p.text.split())


def _pre_blocks(doc: str) -> list[str]:
    return re.findall(r"(?s)<pre.*?</pre>", doc)


def _math(doc: str) -> list[str]:
    m = _strip_scripts(_main_of(doc))
    out = re.findall(r"(?s)\$\$.*?\$\$", m)
    out += re.findall(r"(?s)\\\(.*?\\\)", m)
    out += re.findall(r"(?s)\\\[.*?\\\]", m)
    return [re.sub(r"\s+", " ", x) for x in out]


def _headings(doc: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", f"{i}|{t}").strip()
        for i, t in re.findall(
            r'<h[1-4][^>]*id="([^"]+)"[^>]*>(.*?)</h[1-4]>', _main_of(doc), re.S
        )
    ]


def _references(doc: str) -> list[str]:
    m = re.search(r'(?s)<h2 id="references".*?</ul>', _main_of(doc))
    if not m:
        return []
    return [
        re.sub(r"\s+", " ", x).strip()
        for x in re.findall(r"(?s)<li>(.*?)</li>", m.group(0))
    ]


def _hrefs(doc: str) -> Counter:
    return Counter(re.findall(r'href="([^"]+)"', doc))


def _times(doc: str) -> list[tuple[str, str]]:
    return re.findall(r'<time datetime="([^"]+)">([^<]*)</time>', doc)


def _numbers(doc: str) -> Counter:
    # Trailing punctuation is stripped: replacing an em dash after a figure with a comma or period
    # attaches it to the token ("0.37" -> "0.37,") without the figure having moved. An internal
    # comma is kept, because 1,000 is a different number from 1.
    toks = (
        re.sub(r"[.,]+$", "", t) for t in re.findall(r"\d[\d,.]*%?", prose_text(doc))
    )
    return Counter(t for t in toks if t)


def _js_skeleton(doc: str) -> str:
    """<script> code with string literals blanked, so prose edits inside strings are invisible.

    Double quotes are blanked first: JSON-LD prose is double-quoted and routinely contains
    apostrophes ("the platform's pacing system"), which would otherwise pair up across the whole
    document and swallow real code.
    """
    js = "\n".join(re.findall(r"(?s)<script[^>]*>(.*?)</script>", doc))
    js = re.sub(r'"(?:[^"\\\n]|\\.)*"', '""', js)
    js = re.sub(r"`(?:[^`\\]|\\.)*`", "``", js)
    js = re.sub(r"'(?:[^'\\\n]|\\.)*'", "''", js)
    return re.sub(r"\s+", " ", js)


def check_invariants(path: Path, advisory: bool = False) -> tuple[list[str], list[str]]:
    """Properties every blog page must satisfy, checkable without git history."""
    doc = path.read_text(encoding="utf-8")
    err: list[str] = []
    warn: list[str] = []

    hits = re.findall(EM_DASH, doc)
    if hits:
        forms = Counter("literal" if h == "—" else "entity" for h in hits)
        err.append(
            f"{len(hits)} em dash(es): {dict(forms)}. The entity form is easy to miss."
        )

    # Single-$ is deliberately NOT a math delimiter on this site, so that "$40" renders as money.
    # A $...$ pair whose contents look like LaTeX is therefore an author expecting math and
    # getting literal text. Currency is the common case here and must not trip this, so the span
    # only counts when it carries a backslash command, a superscript, or a subscript.
    body = _strip_scripts(doc)
    if re.search(r"(?<!\$)\$(?!\$)[^$]{0,80}[\\^_][^$]{0,80}\$(?!\$)", body, re.S):
        err.append(
            "single-$ math delimiter; this site enables only $$, \\( \\) and \\[ \\], so a "
            "single-$ span renders as literal text"
        )

    structure = _Structure()
    structure.feed(doc)
    structure.close()
    err.extend(structure.problems())

    ids = set(re.findall(r'\sid="([^"]+)"', doc))
    for anchor in sorted(set(re.findall(r'href="#([^"]+)"', doc))):
        if anchor not in ids:
            err.append(f"dangling in-page anchor #{anchor}")

    og = re.search(r'<meta property="og:description" content="([^"]*)"', doc)
    tw = re.search(r'<meta name="twitter:description" content="([^"]*)"', doc)
    if og and tw and og.group(1) != tw.group(1):
        # build_seo.py derives twitter:description from og:description, so a mismatch means the
        # page will silently change on the next SEO build.
        err.append("og:description and twitter:description disagree")

    text = prose_text(doc)
    words = len(text.split())
    semis = text.count(";")
    if words:
        per_1k = 1000 * semis / words
        if per_1k > SEMICOLON_PER_1K_LIMIT:
            err.append(
                f"{semis} prose semicolons in {words} words ({per_1k:.1f}/1k); "
                f"the guide allows roughly 1/1k, gate is {SEMICOLON_PER_1K_LIMIT}"
            )

    if path.name == "blog.html":
        err.extend(_check_reading_times(doc, path.parent))

    if advisory:
        for label, pat in ADVISORY.items():
            found = re.findall(pat, _strip_scripts(doc))
            if found:
                sample = ", ".join(
                    repr(h if isinstance(h, str) else h[0])[:44] for h in found[:3]
                )
                warn.append(f"{label} x{len(found)}: {sample}")

    return err, warn


def _check_reading_times(doc: str, docs_dir: Path) -> list[str]:
    """blog.html hard-codes a reading time per card; components.js recomputes it at runtime.

    Nothing else catches the drift, and it drifted on 28 of 42 cards before anyone looked.
    """
    err = []
    pat = re.compile(
        r'<a class="post-card" href="(blog-[^"]+)".*?&approx; (\d+) min read', re.S
    )
    for target, stated in pat.findall(doc):
        page = docs_dir / target
        if not page.exists():
            err.append(f"index card points at missing page {target}")
            continue
        want = max(
            1, round(chip_words(page.read_text(encoding="utf-8")) / WORDS_PER_MINUTE)
        )
        if want != int(stated):
            err.append(
                f"index card for {target} says {stated} min read, computes {want}"
            )
    return err


def check_against(
    path: Path, ref: str, tolerance: float = 0.07
) -> tuple[list[str], list[str]]:
    """Migration mode: nothing a *style* edit must not touch may have moved since `ref`.

    Returns (errors, warnings).
    """
    rel = path.relative_to(REPO).as_posix()
    got = subprocess.run(
        ["git", "show", f"{ref}:{rel}"], cwd=REPO, capture_output=True, text=True
    )
    if got.returncode != 0:
        return [], [f"no {ref} version of {rel} to diff against"]
    old, new = got.stdout, path.read_text(encoding="utf-8")
    err: list[str] = []
    warn: list[str] = []

    before = len(prose_text(old).split())
    after = len(prose_text(new).split())
    if before:
        delta = (after - before) / before
        if abs(delta) > tolerance:
            err.append(
                f"body words moved {delta:+.1%} ({before} -> {after}); a style pass that shrinks "
                f"a post is deleting substance, limit is +/-{tolerance:.0%}"
            )

    def norm(value) -> str:
        """Collapse a value so that *only* an em-dash replacement compares equal.

        An em dash removed from a code comment, a reference separator, or a heading is a legitimate
        edit inside an otherwise frozen zone. Only the marks that can stand in for an em dash are
        neutralised, and case is folded because turning a colon into a period capitalises the next
        word.

        Deliberately NOT neutralised: ``-``, ``(``, ``)``. Stripping those let a sign flip
        (``roi = -0.5`` becoming ``roi = 0.5``) compare equal inside a frozen code block, which is
        exactly the change migration mode exists to catch.
        """
        s = (
            repr(sorted(value.items()))
            if isinstance(value, Counter)
            else (value if isinstance(value, str) else repr(value))
        )
        return re.sub(r"[\s,.;:]+", "", re.sub(EM_DASH, " ", s)).casefold()

    for name, fn, fatal in (
        ("code blocks", _pre_blocks, True),
        ("math", _math, True),
        ("headings", _headings, True),
        ("references", _references, True),
        ("hrefs", _hrefs, True),
        ("<time>", _times, True),
        ("JS logic", _js_skeleton, True),
        ("numbers in prose", _numbers, False),
    ):
        a, b = fn(old), fn(new)
        if a == b:
            continue
        if fatal and name != "math" and norm(a) == norm(b):
            # Report it rather than skipping silently. The normalisation is a heuristic, and a
            # heuristic that hides its own decisions is how a real change gets waved through.
            warn.append(f"{name}: em-dash-only change, rest of the content identical")
            continue
        detail = ""
        if isinstance(a, Counter):
            detail = f"removed={dict(list((a - b).items())[:5])} added={dict(list((b - a).items())[:5])}"
        elif isinstance(a, list) and len(a) != len(b):
            detail = f"{len(a)} -> {len(b)} items"
        else:
            seq_a = a if isinstance(a, list) else [a]
            seq_b = b if isinstance(b, list) else [b]
            for x, y in zip(seq_a, seq_b):
                if x != y:
                    detail = f"{str(x)[:80]!r} -> {str(y)[:80]!r}"
                    break
        (err if fatal else warn).append(f"{name} changed: {detail}")
    return err, warn


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "files", nargs="*", help="blog pages (default: all docs/blog*.html)"
    )
    ap.add_argument("--advisory", action="store_true", help="also report prose smells")
    ap.add_argument(
        "--against", metavar="REF", help="also diff against a git ref (migration mode)"
    )
    args = ap.parse_args(argv)

    files = [Path(f).resolve() for f in args.files] or sorted(DOCS.glob("blog*.html"))
    failed = 0
    for f in files:
        err, warn = check_invariants(f, advisory=args.advisory)
        if args.against:
            diff_err, diff_warn = check_against(f, args.against)
            err += diff_err
            warn += diff_warn
        if err:
            failed += 1
        status = "FAIL" if err else ("WARN" if warn else "PASS")
        print(f"{status}  {f.name}")
        for e in err:
            print(f"        ERROR  {e}")
        for w in warn:
            print(f"        warn   {w}")
    print(f"\n{len(files)} pages, {failed} with errors")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
