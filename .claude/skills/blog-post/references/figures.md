# Figures

Interactive figures use Plotly plus `docs/shared/blogviz.js`, which exports one global,
`window.BV`. No build step, no bundler, no npm. Plotly is pinned at 2.27.0 corpus-wide.

Do not roll a per-page helper. `blogviz.js` exists precisely because the older math and
causal guide pages each wrote their own and all of them broke in dark mode.

Causal DAGs are a different mechanism entirely; see the last section.

## Contents

- [The BV surface](#the-bv-surface)
- [Markup](#markup)
- [The script](#the-script)
- [Theme-awareness](#theme-awareness)
- [Controls](#controls)
- [Readouts](#readouts)
- [Strings inside Plotly](#strings-inside-plotly)
- [Determinism](#determinism)
- [Verifying a figure](#verifying-a-figure)
- [DAGs](#dags)

## The BV surface

That is all of it. There is no BV chart constructor; you call `Plotly.react` yourself.

Drawing — `BV.register(fn)` (runs `fn` now and on every theme flip), `BV.redraw()`,
`BV.palette()` (token colors: primary, primaryDark, primaryLight, accent, accentDark,
accentLight, warning, danger, success, text, muted, border, grid, surface, bgAlt),
`BV.cycle()` (7-color categorical order), `BV.band(color, alpha)` (translucent rgba,
default 0.18), `BV.layout(overrides)`, `BV.config`.

Controls — `BV.sliders(specs, onChange)`.

Math — `BV.linspace`, `BV.clamp`, `BV.mulberry32(seed)`, `BV.randn(rng)`, `BV.erf`,
`BV.normCdf(x, mu, sd)`, `BV.mean`, `BV.quantile(sorted, q)`.

Two signatures to get right: `BV.quantile` requires an **already sorted** array, and
`BV.randn` takes the mulberry32 **closure**, not a seed.

`BV.layout(overrides)` merges onto a transparent-background base (Source Sans 3 in the
muted token, margin `{l:62,r:24,t:30,b:54}`, closest hovermode, themed hoverlabel) and
runs `xaxis`/`yaxis`/`xaxis2`/`yaxis2` through an axis themer that supplies gridcolor,
zerolinecolor, linecolor, tickcolor, tickfont and `automargin`. Pass titles and dtick;
the colors come for free.

**A third axis is not themed.** `yaxis3`, polar and ternary subplots get Plotly's default
near-black grid, invisible on the dark surface. `shapes` and `annotations` also pass
through untouched, so colour them from `BV.palette()` by hand.

## Markup

The house default is the interactive box. Order matters.

```html
<div class="interactive-box">
    <h4>Figure title</h4>
    <p style="margin-top:0;font-size:0.92rem;color:var(--color-text-muted);">
        What the reader is looking at and what the controls do.</p>

    <div class="control-row">
        <label for="f1K">Alternatives K</label>
        <input type="range" id="f1K" min="2" max="20" step="1" value="10">
        <span class="value" id="f1KVal">10</span>
    </div>

    <div class="chart-container chart-themed" id="f1Chart" style="height:330px;"></div>

    <div class="viz-readout">
        <div class="stat"><span class="lbl">Expected disappointment</span>
            <span class="val" id="f1Dis">&hellip;</span></div>
    </div>

    <p class="chart-caption" style="margin-bottom:0;">What the figure shows.</p>
</div>
```

The inline `style="height:…"` is **mandatory** — `.chart-container` defaults to 350px and
Plotly sizes to `clientHeight`. `margin-bottom:0` on the caption is correct only inside
the box.

A static figure is just the themed container plus `<p class="chart-caption">` in the prose
flow. Only two exist in the corpus. Static still means `BV.register(fn)`, never a bare
`Plotly.newPlot`, or it will not re-theme.

## The script

**All** figures on a page live in one `<script>`, placed after
`<script src="shared/components.js"></script>` and before the KaTeX init. Wrap it:

```html
<script>
(function () {
    'use strict';

    // ---- Figure 1: the optimizer's curse ----------------------------------
    function curve(K) { /* pure data */ }

    BV.register(function () {
        var p = BV.palette();
        var K = parseInt(document.getElementById('f1K').value, 10);   // read INSIDE
        Plotly.react('f1Chart', traces, BV.layout({ xaxis: {title: 'K'} }), BV.config);
        document.getElementById('f1Dis').textContent = fmt(value);
    });
    BV.sliders([{ id: 'f1K', value: 'f1KVal', fmt: function (v) { return v; } }], BV.redraw);
})();
</script>
```

Read control state **inside** the register callback. The same callback services both the
slider and the theme observer, so it has to re-read every time.

Load order: the figure script needs `BV` defined, so it must come after `blogviz.js`
(which is in `<head>`, un-deferred along with Plotly for exactly this reason).

## Theme-awareness

`BV.palette()` resolves CSS custom properties off `document.documentElement` at call
time, and a MutationObserver on `<html>`'s `data-theme` calls `redraw()`. So: call
`BV.palette()` inside the callback, use `Plotly.react`, and re-theming is free.

The load-bearing extra step is markup. **Every container needs both classes,
`chart-container chart-themed`.** Plotly injects its own `.js-plotly-plot { background:#fff }`,
which the older PT-based pages rely on because their charts hardcode dark text.
`styles.css:1699` opts a container out of that white card:

```css
[data-theme="dark"] .chart-themed.js-plotly-plot { background: var(--color-surface); }
```

It only touches containers that explicitly carry the class. Omit `chart-themed` and your
theme-aware chart renders as a jarring white rectangle in dark mode, with no error and no
test failure.

## Controls

`BV.sliders(specs, onChange)` takes `[{ id, value, fmt }]` where `id` is the range
input's id and `value` is the id of the `<span class="value">`. It wires an `input`
listener that updates the label then calls `onChange`, which is essentially always
`BV.redraw`. It also runs once at wire-up so the label matches the initial `value`.

`BV.sliders` **silently no-ops on a typo'd id** — the slider moves and nothing redraws.

`BV.redraw()` re-runs every figure on the page, so an expensive Monte Carlo makes an
unrelated slider laggy. The corpus memoizes the expensive curve in an IIFE-scoped var.

Buttons are plain `<button type="button" class="viz-btn">` with a hand-written listener
that mutates an IIFE-scoped variable, updates the button's own label, and calls
`BV.redraw()`. The button's initial `textContent`, the state variable's default, and
whatever the caption calls "the defaults" must all agree; nothing enforces that.

## Readouts

The draw fn writes stat values by `textContent`. A "Verdict" stat also sets `className`
to `'val'`, `'val ok'` or `'val warn'` so the qualitative judgement re-derives from the
current settings instead of being frozen in prose. Setting `className` wipes the base
class, so always write the full `'val ok'`, never `'ok'`.

Stat values are seeded with `&hellip;` in the HTML. If the draw fn throws before reaching
the assignments you get a live chart above a row of ellipses and nothing fails loudly,
because `BV.redraw` swallows per-figure exceptions.

## Strings inside Plotly

The surrounding prose uses named entities (`&sigma;`, `&times;`, `&approx;`, `&nbsp;`).
Strings that go **into** Plotly must not: trace `name`, axis `title`, `hovertemplate`,
`annotations[].text`, `text` arrays and category labels render entities literally, as do
anything written with `textContent`.

Use literal Unicode there: β θ α κ π λ σ × ÷ ± ≈ → −.

Plotly's own markup subset (`<br>`, `<b>`, `<sup>`, `<extra></extra>` in hovertemplates)
is fine and used heavily. Four live violations exist in the corpus, including
`blog-modelled-one-p.html:496` and `blog-prior-was-doing-more.html:378` — treat them as
cautionary examples rather than precedent.

## Determinism

Every random draw comes from `BV.mulberry32(seed)` with a literal seed, so every reader
sees identical numbers. This is the only reason a caption can quote a figure's output to
three decimals. Use separate seeds for a Monte Carlo aggregate and for a single
illustrative display so changing one does not disturb the other.

`Math.random()` anywhere in a figure makes the caption unverifiable. Resample buttons are
the deliberate exception, and they advance a seed rather than calling `Math.random`.

## Verifying a figure

**No tool checks that a caption's numbers match what the figure computes.** A green
`make tests` proves nothing here: a figure can be blank, throwing, white-carded in dark
mode, or captioned with numbers it never produces, and every gate still passes. A past
sweep of this kind found 5 caption numbers that disagreed with the figure, including one
major claim the figure's own tooltips contradicted.

Run the page:

```bash
cd docs && python3 -m http.server 8899
```

Drive it with the mambaforge Python Playwright at
`/opt/homebrew/Caskroom/mambaforge/base/bin/python` (Node Playwright is not installed).

- `wait_for_selector('#f1Chart .plot-container', state="attached")` — `state="attached"`
  matters, the default `visible` times out because Plotly reports the container hidden.
- Read each readout with `inner_text('#f1Dis')` and compare against the caption.
- For a caption's non-default claims ("drop to K = 3 and it falls to roughly 0.84σ"), set
  the slider and dispatch the event the toolkit listens for:
  `el.value = v; el.dispatchEvent(new Event('input', {bubbles:true}))`.
- Flip `document.documentElement.setAttribute('data-theme','dark')` and confirm the
  redraw fires and the background follows the surface token.
- Collect `console` and `pageerror` events and assert they are empty, since `BV.redraw`
  swallows per-figure exceptions.

Also check the **sign** of anything labelled as a loss or a gap. One live readout labelled
"Expected disappointment" prints a negative number because the code computes
`mu - estimate`; no tool would catch that.

Adding a figure changes the body word count, which can drift `blog.html`'s hard-coded
reading time for that post. Re-run `python3 docs/tools/check_blog_style.py docs/blog.html`.

## DAGs

Causal diagrams are hand-drawn inline SVG, not Plotly, and use no BV. They need a
per-page `drawDag(id, nodes, edges)` helper plus a `.dagfig` CSS block whose fills and
strokes are `var(--color-*)` tokens, which is how they stay theme-correct with no redraw
registration.

Copy the helper and the CSS from `blog-table-2-fallacy.html` rather than inventing one.
The node role classes (exposure / outcome / covar / unmeasured) are shared visual
vocabulary across the causal posts.
