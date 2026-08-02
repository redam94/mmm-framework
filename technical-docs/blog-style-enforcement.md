# Blog prose style: the spec, the tooling, and the traps

The spec is **[`writing-guide.md`](writing-guide.md)**. Read it before writing or editing anything
in `docs/blog*.html`. This file is the repo-specific half: what the guide means for these
particular HTML pages, what enforces it, and the things that have already gone wrong.

Applied to the whole series on 2026-08-02 (commits `fdfc4d9`, `ebce1d0`): 42 essays plus the index,
2,670 em dashes removed, prose semicolons 669 to 44 (4.3 to 0.3 per thousand words), body prose
+0.26%.

(Those figures cover all 43 pages measured after the follow-up repair pass. `fdfc4d9`'s commit
message quotes 2,634 / 685 to 60 / +0.11%, which is the same work measured over the 42 essays only,
before the index page and the repairs landed. Both were right when written; this table is the one
to reuse.)

That last number is the one to watch on any future pass. The guide's structural rules (cut summary
closers, cut participial tails, break the default triple) all point toward deletion, so a style
pass that *shrinks* a post is usually deleting substance rather than restructuring it. Hold each
file to +/-7%.

## Tooling

```bash
python3 docs/tools/check_blog_style.py                    # invariants, all blog pages
python3 docs/tools/check_blog_style.py docs/blog-x.html   # one page
python3 docs/tools/check_blog_style.py --advisory         # + prose smells needing human judgment
python3 docs/tools/check_blog_style.py --against HEAD     # + "did a style edit move content?"
```

`tests/test_blog_style.py` runs the invariant mode over every page in CI, and separately tests that
the checker catches each violation class and stays quiet on the constructs that merely look like
violations. That second half exists because the checker produced false failures three times during
the original pass, and a checker nobody trusts gets switched off.

**Invariant mode** (gated): zero em dashes in any spelling, no single-`$` LaTeX span,
`og:description` matching `twitter:description`, no dangling in-page anchors, prose semicolons
under 2.0 per thousand words, the structural elements the other checks depend on being present and
balanced, and index reading times that agree with the runtime computation.

It does *not* validate HTML. An earlier version claimed to, by overriding
`html.parser.HTMLParser.error()` — a hook the stdlib has not called since Python 3.5, so every
malformed page passed. If you want real validation, add a parser that reports it; do not trust that
name again.

**Migration mode** (`--against <ref>`, not gated) is for rewriting an existing post. It fails if
anything a *style* edit must never touch has moved: code blocks, math, headings, ids, hrefs,
citations, `<time>`, JS logic, or word count beyond 7%. It is not in the test suite because
ordinary content edits legitimately change all of those.

## What "no em dashes" means in these files

Three zones, all of which count:

- **`<main>` prose**, including `<li>` text, `<p class="lead">`, `<figcaption>`, and callout boxes.
- **`<head>` descriptions.** `description` is search copy and `og:description` is social copy. They
  are allowed to differ and do on 23 of 43 pages, so do not "fix" that. What *is* required is
  `og:description` matching `twitter:description`, because `build_seo.py` derives the latter from
  the former and a mismatch silently reverts on the next build.
- **Prose strings inside `<script>`**: figure titles, axis labels, annotations, hover text. Plotly
  renders HTML entities literally, so a plot title needs plain ASCII punctuation, never `&mdash;`
  or `&amp;`.

JS `//` comments and comments inside `<pre><code>` blocks count too. Executable content does not
change; a comment's punctuation may.

Replace each dash by rewriting the sentence around it. A bracketing pair usually wants parentheses
or commas; a single dash before a payoff usually wants a period, with the second half promoted to
its own sentence, which is a free rhythm win. Forbidden substitutes: `&mdash;`, `&#8212;`, ` - `,
and the en dash. **Existing en dashes stay** — they do real work in number ranges (`2023–26`) and
hyphenated author pairs (`Rambachan–Roth`).

## Traps

**Em dashes hide as entities.** `&mdash;` was as common in these files as the literal character
(1,251 against 1,419 before the pass), and posts written later use the entity almost exclusively.
Grepping `<main>` for `—` alone reported 12 of 42 posts clean while they carried 37 to 90
`&mdash;` each. Browsers also accept hex, zero-padded, and unterminated references (`&#x2014;`,
`&#08212;`, `&#8212`), plus codepoints that render identically (U+2015 HORIZONTAL BAR is the
dangerous one). The checker's `EM_DASH` pattern covers all of them; a hand grep should too.

**`build_seo.py` can silently undo the work.** Line 110 held an em dash inside the sitewide JSON-LD
description that is injected into *every* page. Any SEO rebuild would have put it back on all 43.
Fixed. Lines 217 and 479 also contain the character and must keep it: that is parsing logic
(splitting `Brand — Page` titles, stripping trailing punctuation).

**Removing a negation pivot loses information.** This is the big one. "X is not A, it is B" tells
the reader that A is the plausible wrong belief, so deleting the first half can invert the claim.
Reviewers found 80 instances in the original pass. Real examples:

| what happened | why it matters |
|---|---|
| "the coefficient is not the incremental effect, it is that effect tangled up with…" became "the coefficient is the incremental effect tangled up with…" | the lead now asserts roughly the opposite of the post's thesis |
| a dropped **"only"** | turned a necessary condition into a sufficient one, contradicting the same post two paragraphs earlier |
| "an ELBO that oscillates **without improving**" lost its qualifier | ADVI traces oscillate constantly while improving; healthy behaviour got reclassified as pathology |

**Repair by writing a third phrasing, never by reverting.** Reverting restores the banned
construction. "The coefficient *mixes* the incremental effect with everything that made spend rise
and fall" carries the meaning and obeys the guide.

**Splitting a sentence can strip attribution.** "Berman showed X and noted Y" split in two leaves Y
in the post's own voice. Watch this wherever a cited result is followed by its qualifier.

**Some quoted text is quoted from `src/`.** Several posts quote docstrings verbatim, and a reader
can clone the repo and grep them. Two posts had text edited *inside* the quote marks; the fix is to
restore the quotation byte-faithful and move connective words outside it. Note that a grep for a
quoted phrase fails on line-wrapped source, so normalise whitespace before checking.

**LaTeX `\;` is not a semicolon.** Strip `$$…$$`, `\(…\)` and `\[…\]` before counting, or every
post looks ten times over budget.

**Sidebar labels are deliberately abbreviated.** The `<aside class="sidebar">` TOC shortens heading
text for 28 headings across 9 pages, so exact matching is not a valid check. But the two are
separate strings: if you rewrite a heading, decide whether the TOC label needs the same edit. Never
change the `id` — other posts deep-link to it, and migration mode only sees an `id` change through
the `id|text` pairs it compares, so an id renamed together with its heading text is not separately
flagged.

**Index reading times rot silently.** `docs/blog.html` hard-codes `&approx; N min read` per card
while `components.js::initPageMetaChips` computes `round(words/220)` from the post's `<main>`
textContent at runtime. 28 of 42 disagreed before anyone looked, by up to 6 minutes. Now gated.

**Commit prose before running `build_seo.py`.** `git_date()` reads the last *commit* date, so
running it over uncommitted edits stamps the previous commit's date and drifts roughly 70 unrelated
pages. Committing first confined the 2026-08-02 run's `dateModified` bumps to exactly the 43 pages
whose prose changed. It still rewrote 129 HTML files, because fixing the em dash in the sitewide
JSON-LD description changes that one string on every page; those diffs are one line each.

## Method that worked for a bulk pass

A written brief plus the deterministic checker, one editor agent per file self-verifying to green,
then independent reviewers reading the *diffs* for what a script cannot see. The two halves caught
disjoint sets of problems: the checker never flagged a meaning change, and the reviewers never
flagged a broken anchor. Running only one of them would have shipped the other's failures.

Reviewers were told to be skeptical, which is right, and produced some over-calls: of 152 findings
they raised, 145 were judged real and fixed and 7 were rejected with reasons. Those counts, and the
80-instance figure for negation-pivot damage above, are the record of that one run and cannot be
re-derived from the repo; treat them as an order of magnitude, not a measurement.

Budget for the review, not just the rewrite. The rewriting agents reported themselves clean and
were, mechanically. Every meaning-level defect in the list above came from the separate pass that
read the diffs.
