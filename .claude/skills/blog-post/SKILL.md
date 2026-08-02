---
name: blog-post
description: Write, rewrite, or review a post on the MMM Framework research blog (docs/blog-*.html, the "Modern Measurement Research" series). Applies the house prose spec in technical-docs/writing-guide.md (no em dashes, ~1 semicolon per 1000 words, no negation pivots, varied rhythm) and the repo machinery in technical-docs/blog-style-enforcement.md. Use this WHENEVER the request touches a docs/blog*.html page or the research blog at all — drafting a new essay, editing or restyling an existing one, adding a figure, fixing prose, updating the blog index, or checking a post before commit — and also when someone asks for "a post about X" for this project without naming the file, since the blog is where those go. Encodes the page skeleton, the five registration points, the build order, and the gates.
---

# Writing for the research blog

The blog is 42 essays plus an index at `docs/blog.html`. They are technical research
essays for marketing analysts, grounded in primary literature and in this codebase.
Every number and citation in them has been checked, and a reader can clone the repo
and grep the docstrings they quote.

Two documents govern the work. Read them; do not work from this file's summary of them.

- **`technical-docs/writing-guide.md`** is the prose spec. Read it in full before
  writing or editing a single sentence.
- **`technical-docs/blog-style-enforcement.md`** is what the spec means for these
  HTML pages, plus the traps that have already bitten. Read it before your first edit.

## Start by asking which job this is

The three jobs have different risks, and the middle one is where damage happens.

**Drafting a new post.** The prose spec applies from the first sentence. The bulk of
the work beyond writing is registration: a new post has to be added to five places or
it renders wrong, silently. See `references/registration.md`.

**Rewriting or restyling an existing post.** Everything factual is frozen. Use
migration mode (below) to prove it. The failure mode is not a typo, it is a claim
that quietly changed meaning; read *Repairing without reverting* before you start.

**Reviewing a post.** Run the checker, then read for what the checker cannot see.
The checker has never once flagged a meaning change.

## Write

Start from `assets/post-skeleton.html`, or copy an existing post. **Do not start from
`docs/TEMPLATE-SIDEBAR.html`** — it registers a fourth KaTeX delimiter pair, single
`$`, and this corpus is full of dollar amounts that would be eaten as math.

`references/page-anatomy.md` has the full structure: the `<head>` block and its order,
the sidebar TOC, the byline, the real callout classes (`.note`, `.definition`,
`.warning`, `.takeaway-box`), tables, code blocks, the references list, and what goes
after `</main>`. Read it when you need the markup.

`references/figures.md` covers the `blogviz` toolkit if the post has figures.

Five things are easy to get wrong and worth holding in mind while drafting:

- **The `<time datetime="YYYY-MM-DD">` byline is the publication date.** `build_seo.py`
  scrapes the first one in the file for `datePublished`. Get it wrong and a wrong date
  ships to search engines.
- **Math uses `$$…$$`, `\[…\]`, `\(…\)`. Never a single `$`.** Inside math write `\lt`
  and `\le`, never a bare `<`, which breaks the HTML. Same in code blocks: `&lt;`.
- **Every Python-ish block starts with `# illustrative`.** `tests/test_docs_snippets.py`
  otherwise resolves the imports and method calls against the real API and fails the
  page. The marker is the opt-out for code that is meant to be read, not run.
- **`og:description` must equal `twitter:description`.** `build_seo.py` derives the
  latter from the former, so a mismatch silently reverts on the next build.
  `description` is search copy and may differ from both; that is deliberate.
- **Nav and footer are injected by `components.js`.** Writing a literal `<nav>` or
  `<footer>` gives the page two of them.

## Check

```bash
python3 docs/tools/check_blog_style.py docs/blog-<slug>.html              # gated invariants
python3 docs/tools/check_blog_style.py docs/blog-<slug>.html --advisory   # smells to judge
```

Invariant mode needs no git history, so it works on a post that does not exist yet in
git. It enforces: zero em dashes in any spelling, no single-`$` math, og/twitter
descriptions agreeing, no dangling anchors, semicolons under 2.0 per thousand words,
and (on `blog.html`) index reading times matching what the site computes at runtime.

`--advisory` prints things a regex cannot adjudicate: negation pivots, "less X than Y",
hedge stacks, banned vocabulary. Some are false positives by design. "Less lift than
the geo test showed" is a quantitative comparison and fine; "less a stance than a
habit" is the banned rhetorical substitution. You decide, and say which you left.

**A green run is not a good post.** The checker cannot see rhythm, meaning, or whether
a paragraph earns its place, which is most of the guide. Read the prose.

For the rhythm half, this helps:

```bash
python3 .claude/skills/blog-post/scripts/rhythm_audit.py docs/blog-<slug>.html -v
```

It reports sentence-length spread, paragraphs where three consecutive sentences fall
within five words of each other, and repeated sentence openers. It gates nothing and
there is no target score — a clump can be doing real work. Calibrate against a published
post: `blog-optimizers-curse.html` reads 3 clumped paragraphs of 40, a median sentence of
18 words with a range of 1 to 60, and 14 sentences under 8 words against 26 over 35. A
page where clumping is everywhere and the range is narrow is prose assembled to a shape
rather than to an argument.

Reading the output is not a substitute for reading the prose aloud.

### Rewriting an existing post

```bash
python3 docs/tools/check_blog_style.py docs/blog-<slug>.html --against HEAD
```

Migration mode fails if a *style* edit moved anything a style edit must never move:
code blocks, math, headings, ids, hrefs, citations, `<time>`, JS logic, or word count
beyond ±7%. Run it before you commit a rewrite.

The ±7% band matters more than it looks. Every structural rule in the guide points
toward deletion, so a style pass that *shrinks* a post is usually cutting substance
rather than restructuring it. The 2026-08-02 pass over all 43 pages moved total body
prose by +0.26%, which is what a real restructuring looks like.

## Repairing without reverting

This is the single most important thing in this skill, because it is where a
well-intentioned edit does real damage.

The guide bans the negation pivot: "X is not A, it is B." That ban stands. But the
construction carries information — it tells the reader that A is the plausible wrong
belief — and deleting the first half can invert the claim. Eighty instances of this
turned up in one pass over the series. Real examples:

| edit | what it cost |
|---|---|
| "the coefficient is not the incremental effect, it is that effect tangled up with…" became "the coefficient is the incremental effect tangled up with…" | the lead now asserts roughly the opposite of the post's thesis |
| a dropped **"only"** | a necessary condition became a sufficient one, contradicting the same post two paragraphs earlier |
| "an ELBO that oscillates **without improving**" lost its qualifier | ADVI traces oscillate constantly while improving; healthy behaviour got reclassified as pathology |
| a sentence split after "Berman showed X and noted Y" | Y moved into the post's own voice, and the next paragraph still says "read that qualifier carefully" |

When you find one of these, **write a third version**. Reverting restores the banned
construction; leaving it loses the meaning. There is almost always a phrasing that
does both jobs:

```
Original (banned):  The coefficient is not the incremental effect of advertising, it
                    is that effect tangled up with everything that made spend move.
Broken rewrite:     The coefficient is the incremental effect of advertising tangled
                    up with everything that made spend move.
Correct repair:     The coefficient mixes the incremental effect of advertising with
                    everything that made spend move.
```

The same applies to a dropped "only" or "merely" (restore the exclusivity with a word,
not a dash), a lost attribution (put the author's name in the sentence that now carries
the claim), and a dropped technical qualifier (put it back; precision beats brevity).

**Quotations are frozen.** Several posts quote docstrings from `src/` verbatim and a
reader can grep them. Never edit inside quote marks and never split a quotation to
insert a connective. Move your own words outside the quotes. When checking a quote
against source, normalise whitespace first, because the source is line-wrapped.

## Register, build, commit

A new post needs five registrations, three of them tested and two silent. Reading times
on the index have their own gate. The order of build and commit matters. All of it,
with the exact edits and what breaks when skipped, is in **`references/registration.md`**
— read it before committing a new post.

The short version, and the one ordering constraint that bites:

```bash
# 1. after the post + blog.html card + components.js registrations are written:
uv run pytest tests/test_blog_style.py tests/test_docs_snippets.py \
              tests/test_docs_nav_registration.py tests/test_docs_seo_build.py -q

# 2. COMMIT the prose first
# 3. only then regenerate, because build_seo.py stamps dates from the last commit
cd docs && python3 tools/build_search_index.py && python3 tools/build_seo.py

# 4. commit the regenerated artifacts, including shared/seo-manifest.json
```

Running the builders before committing stamps every touched page with the *previous*
commit's date and drifts roughly 70 unrelated pages.

## Reviewing someone else's post

Read the diff, not just the result. Look for the meaning-level failures above, for
prose that reads worse than what it replaced, and for faked rhythm — long sentences
chopped at the comma into uniform medium ones, or a short punchy sentence dropped into
every paragraph like a tic. The guide asks for variation that sounds like a person
thinking, which is not the same as variation a script could produce.

Be skeptical, and be willing to be wrong. In the one large review of this series, 145
of 152 findings were real and 7 were over-calls; saying "I flagged this and then
checked it and it is fine" is a useful outcome.
