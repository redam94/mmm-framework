# Registering a new post, and the build order

A new post has to be added to **five** places besides the post file. Three are tested.
Two fail silently. One of the silent ones is currently 15 posts behind.

The reason to care: in 2026-07 seven of the then-27 posts had drifted out of the series
list and rendered with **no Previous/Next cards at all** — no error, no empty container,
nothing. Hand-appending is how that happens.

## Contents

- [The five registrations](#the-five-registrations)
- [Reading times](#reading-times)
- [Build and commit order](#build-and-commit-order)
- [Gates](#gates)
- [Full checklist](#full-checklist)

## The five registrations

### 1. `docs/blog.html` — the card

Inside the right `<section class="arc-section">` (Part I–IV), in the position the post
belongs:

```html
<a class="post-card" href="blog-prior-was-doing-more.html">
    <div class="post-card-meta">
        <time datetime="2026-07-26">Jul 26, 2026</time>
        <span class="post-tag">Causal inference</span>
        <span>&approx; 13 min read</span>
    </div>
    <h3>The Prior Was Doing More Than You Thought</h3>
    <p>A second estimation path made a hidden dependency visible...</p>
</a>
```

The blurb is its own copy, not a quotation of the lead, and the prose spec applies to it.

### 2. `docs/shared/components.js` — `SERIES['Measurement research']`

Drives the Previous/Next cards. One line, single quotes, apostrophes backslash-escaped,
**at the same position the card occupies in `blog.html`**:

```js
        ['blog-my-new-post.html', 'My New Post Title'],
```

Silent failure: `initSeriesNav` does `pages.findIndex(...)`, gets `-1`, `continue`s, and
the page renders with no prev/next cards.

### 3. `docs/shared/components.js` — `PAGE_TIERS`

Drives **both** the audience chip and the `≈ N min read` chip:

```js
        'blog-my-new-post.html': TIER_ANALYST,
```

Valid constants: `TIER_OVERVIEW`, `TIER_ANALYST`, `TIER_TECHNICAL`. Most posts are
analyst; the heavier methodological ones are technical.

Silent failure, double damage: `initPageMetaChips` returns early on a missing tier, so
the page gets neither chip. The reading time is computed, not registered, but it never
renders. The page just looks subtly unfinished next to every other post.

### 4. `docs/tools/build_seo.py` — the llms.txt SERIES list

```python
    ("Modern measurement research (blog)", ["blog.html",
        "blog-activity-bias.html",
        ...
        "blog-my-new-post.html"]),
```

**Nothing tests this one.** `llms.txt` simply omits the post, which defeats the point of
an LLM-discovery index. As of 2026-08-02 the list carries 27 of 42 posts; 15 are missing.
If you are adding a post, add it here, and consider fixing the backlog while you are in
the file.

### 5. Regenerated artifacts

`shared/seo-manifest.json`, `shared/search-index.json`, `shared/glossary.json`,
`sitemap.xml`, `llms.txt`, plus the `<head>` block `build_seo.py` injects into the post.
Commit all of them. The manifest is gated and fails loudly; the rest fail quietly.

There is **no** per-post registration for the glossary or the site nav. Do not go looking.

## Reading times

`blog.html` hard-codes `&approx; N min read` per card. `components.js::initPageMetaChips`
recomputes `Math.max(1, Math.round(words / 220))` at runtime from the post's `<main>`
`textContent`. When those disagree the index advertises one number and the page shows
another, and nothing but the style checker notices. 28 of 42 had drifted before anyone
looked, by up to 6 minutes.

Do not estimate it. Ask:

```bash
python3 docs/tools/check_blog_style.py docs/blog.html
```

It reports the computed value for any card that disagrees. Note `textContent`
concatenates across tags, so `1,198<strong>%</strong>` is one word, not two.

## Build and commit order

One ordering constraint bites, and it is not optional.

```bash
# 1. write the post, the card, and the components.js registrations, then:
uv run pytest tests/test_blog_style.py tests/test_docs_snippets.py \
              tests/test_docs_nav_registration.py tests/test_docs_seo_build.py -q

# 2. COMMIT the prose and registrations
git add docs/ && git commit -m "docs(blog): ..."

# 3. only now regenerate
cd docs && python3 tools/build_search_index.py && python3 tools/build_seo.py

# 4. commit the regenerated artifacts
```

`build_seo.py::git_date()` reads the last **commit** date. Run it over uncommitted edits
and it stamps pages with the *previous* commit's date, which historically drifted about
70 unrelated pages and had to be undone by hand with `git checkout --`. Committing first
confines the `dateModified` churn to the pages that actually changed.

Two more facts about that builder:

- `datePublished` comes from the visible `<time datetime>` byline, not from git.
- `build_seo.py` writes the JSON-LD and Twitter-card block **into the post file**. A post
  committed without running it ships with no structured data.

There are no Makefile targets for any of this. The commands are the interface.

## Gates

| test | what it catches |
|---|---|
| `test_docs_nav_registration.py` | post missing from `SERIES` or `PAGE_TIERS`; `SERIES` order not matching the `blog.html` card order; duplicates; a page in two series |
| `test_docs_seo_build.py` | pages absent from the SEO manifest |
| `test_blog_style.py` | the prose invariants, on every `docs/blog*.html` by glob, plus index reading times |
| `test_docs_snippets.py` | Python blocks referencing APIs that do not exist |

Nothing gates the `build_seo.py` llms.txt list.

## Full checklist

1. Write `docs/blog-<slug>.html`: byline `<time>`, `og:description` equal to
   `twitter:description`, sidebar TOC, `<h2 id="references">` last.
2. Add the card to the right Part in `docs/blog.html`.
3. Add `['href', 'Title']` to `SERIES['Measurement research']` at the same position.
4. Add `'href': TIER_*` to `PAGE_TIERS`.
5. Add the filename to `build_seo.py`'s llms.txt SERIES tuple.
6. `python3 docs/tools/check_blog_style.py docs/blog-<slug>.html` and `--advisory`.
7. Fix the card's reading time to what `check_blog_style.py docs/blog.html` computes.
8. Run the four test files.
9. Commit prose + registrations.
10. `cd docs && python3 tools/build_search_index.py && python3 tools/build_seo.py`.
11. Commit the regenerated artifacts.

Step 7 has to land before step 10, or the manifest hash for `blog.html` is stale and the
next contributor's build re-dates it for no reason.
