"""Author workshop notebook 05 — From Draws to Decisions (run from ``nbs/``).

    uv run python build_workshop_05_from_draws_to_decisions.py
    PYTHONPATH=.. uv run jupyter nbconvert --to notebook --execute --inplace \
        workshop_05_from_draws_to_decisions.ipynb --ExecutePreprocessor.timeout=2400

Notebook 05 of the 6-part *workshop* series: the Bayesian causal workflow for
marketing analysts with no Bayesian background. This is the capstone on
**derived quantities**: turning the posterior from workshop_03/04 into the
numbers a CMO actually asks for — ROAS distributions, decision probabilities,
marginal (next-dollar) ROAS, response curves with bands, what-if scenarios, and
a budget reallocation with an honest uncertainty interval. One doctrine
throughout: **compute per draw, summarize last**.

One fit (the same known-truth "clean" world as workshop_03; ~10 s at this
config), then everything else is posterior arithmetic and paired
posterior-predictive calls. Plotly for the main charts; two ipywidgets live
explorations, each paired with a static panel so the baked notebook teaches
without a kernel.

Authored as md/code cells via nbformat (pattern: ``build_stress_00_rosy_picture.py``).
Every computational cell ends with asserts encoding its claim — directional and
seed-tolerant for MCMC quantities. Markdown never hardcodes fit numbers.
"""

from __future__ import annotations

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


CELLS = [
    # ====================================================================
    # 0. Hook
    # ====================================================================
    md(r"""
# Workshop 05 — From Draws to Decisions

Your CMO doesn't want a posterior. She wants three sentences:

> *Which channel do we cut? Where does the next million go? And how sure are we?*

Workshops 00–04 built the machinery: probability as belief (00), priors (01),
MCMC sampling (02), a first `BayesianMMM` fit on a world where we *know* the
right answer (03), and how to read the raw posterior — draws, HDIs, the joint
structure between parameters (04). This notebook is the bridge from that
machinery to her three sentences. And the bridge has exactly **one rule**:

> ### Compute per draw, summarize last.

A **derived quantity** *(any business number computed from model parameters —
ROAS, an uplift, a ranking)* is computed by pushing **every posterior draw**
through the formula, producing a whole *distribution* of answers, and only
*then* summarizing — a median, an interval, a probability. The tempting
shortcut — push one summary (the posterior mean) through the formula — gives
a different, and usually wrong, number. Section 1 shows you why with actual
numbers; the rest of the notebook is that one skill applied six ways:

1. **The doctrine, demonstrated** — why `f(average) ≠ average of f`.
2. **ROAS as a distribution** — and the framework one-liner that matches it.
3. **Decision probabilities** — P(ROAS > breakeven), P(A beats B). 🎛️
4. **Marginal ROAS** — the *next* dollar is not the *average* dollar.
5. **Response curves with bands** — who has room to grow, who is saturated.
6. **What-if scenarios** — uplift as a distribution. 🎛️
7. **A reallocation, with honesty** — when the interval says "coin flip".

We reuse the exact fit from workshop_03: the synthetic **"clean" world** from
`tests/synth/dgp.py`, where every channel's true causal contribution — and
therefore its true ROAS — is recorded. We get to grade every decision number
against the answer key.
"""),
    code(r"""
import sys, pathlib, warnings, logging, time
import numpy as np, pandas as pd
import arviz as az
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")
# pymc emits sampler warnings at ERROR level -> CRITICAL to keep outputs clean
for _n in ("pymc", "numpyro", "jax", "arviz", "pytensor"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)
try:  # the framework logs via loguru, which bypasses stdlib logging
    from loguru import logger as _loguru
    _loguru.disable("mmm_framework")
except ImportError:
    pass
sys.path.insert(0, str(pathlib.Path.cwd().parent))  # repo root (run from nbs/)

import contextlib, os
@contextlib.contextmanager
def quiet():
    "Hide the samplers' progress bars / chatter; our own prints stay visible."
    with open(os.devnull, "w") as _dn, contextlib.redirect_stdout(_dn), \
            contextlib.redirect_stderr(_dn):
        yield

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = "notebook_connected"  # interactive charts survive baking

INK, ACCENT, SKY, BERRY, LEAF, AMBER, MUTED = (
    "#2b2118", "#b5651d", "#3b6ea5", "#a63a50", "#3f7d5e", "#d98a2b", "#8a8079")

def style(fig, title, xtitle=None, ytitle=None, height=380):
    # One consistent look for every plotly figure in the workshop.
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": INK}},
        template="plotly_white", height=height,
        font={"color": INK}, legend={"orientation": "h", "y": -0.18},
        margin={"l": 60, "r": 20, "t": 50, "b": 60},
    )
    if xtitle: fig.update_xaxes(title_text=xtitle)
    if ytitle: fig.update_yaxes(title_text=ytitle)
    return fig

def kde_xy(x, n=240, pad=0.08):
    # Smooth density curve for a 1-D sample (for uplift distributions).
    span = x.max() - x.min()
    g = np.linspace(x.min() - pad * span, x.max() + pad * span, n)
    return g, gaussian_kde(x)(g)

from tests.synth import dgp  # the known-truth synthetic worlds (workshop_03)
sc = dgp.build("clean")
PAL = dict(zip(sc.channels, (ACCENT, SKY, BERRY, LEAF)))

truth_tbl = pd.DataFrame({
    "total spend": sc.spend[sc.channels].sum().round(0),
    "TRUE contribution": sc.true_contribution.round(0),
    "TRUE ROAS": sc.true_roas.round(3),
})
display(truth_tbl)

assert list(PAL) == sc.channels, "palette must be keyed by the world's channels"
assert (sc.true_roas > 0).all() and (sc.true_contribution > 0).all()
print("✓ the answer key is loaded:", ", ".join(sc.channels))
"""),
    # ====================================================================
    # 1. The fit (one, reused everywhere)
    # ====================================================================
    md(r"""
## 0 — One fit, reused everywhere

Same model, same settings, same seed as workshop_03 — a fast-but-honest
configuration (NumPyro NUTS, 500 draws × 2 chains, parametric adstock). Every
number in this notebook is derived from this single posterior; nothing below
refits anything. (If the diagnostics jargon here reads like static, workshop_02
is the decoder ring.)

A note on units before we start: this world's KPI is in **sales units**, so
"ROAS" here means *incremental KPI units per dollar of spend* — a number like
`0.8` says a dollar buys 0.8 units. If your KPI were revenue dollars it would
be literal return-on-ad-spend. The logic is identical; only the unit label
changes.
"""),
    code(r"""
from mmm_framework.config import InferenceMethod, ModelConfig
from mmm_framework.model import BayesianMMM, TrendConfig, TrendType
from mmm_framework.analysis import MMMAnalyzer

# Importing pymc installs its own log handler; re-quiet and stop propagation.
for _n in ("pymc", "pymc.sampling", "pymc.stats.convergence",
           "numpyro", "jax", "arviz", "pytensor"):
    _lg = logging.getLogger(_n); _lg.setLevel(logging.CRITICAL); _lg.propagate = False

cfg = ModelConfig(
    inference_method=InferenceMethod.BAYESIAN_NUMPYRO,
    n_draws=500, n_tune=500, n_chains=2,
    use_parametric_adstock=True, optim_seed=0,
)
mmm = BayesianMMM(sc.panel(), cfg, TrendConfig(type=TrendType.LINEAR))
t0 = time.perf_counter()
with quiet():
    fit = mmm.fit(random_seed=0)
print(f"fit in {time.perf_counter() - t0:.0f}s   "
      f"r-hat max={fit.diagnostics['rhat_max']:.3f}  "
      f"divergences={fit.diagnostics['divergences']}  "
      f"min bulk ESS={fit.diagnostics['ess_bulk_min']:.0f}")

# CLAIM: the sampler converged (workshop_02's gates) and the model sees the
# same channels, in the same order, as the world's answer key.
assert fit.diagnostics["rhat_max"] < 1.05 and fit.diagnostics["divergences"] == 0
assert mmm.channel_names == sc.channels
print("✓ one clean fit — everything below is arithmetic on its draws")
"""),
    # ====================================================================
    # 2. The doctrine, demonstrated
    # ====================================================================
    md(r"""
## 1 — The doctrine: `f(average) ≠ average of f`

Here is the **plug-in shortcut** *(collapsing the posterior to one "best"
parameter value, then computing your business number from it)* and why it
quietly misreports.

Workshop_03's model says TV's weekly effect follows a **saturation curve**
*(diminishing returns: each extra dollar buys a bit less than the last)*:

$$\text{weekly effect} \;=\; \beta_{TV}\,\bigl(1 - e^{-\lambda_{TV}\, x}\bigr)$$

where $x$ is that week's (normalized) spend and $\beta_{TV}, \lambda_{TV}$ are
parameters the posterior knows only as **clouds of draws** — 1,000 plausible
values each, *correlated with each other* (workshop_04 §joint structure: a
flatter curve with a bigger β can explain the same data as a steeper curve
with a smaller β).

Two ways to compute "TV's effect at a mid-curve spend level":

- **Plug-in**: average β and λ first, push the *averages* through the curve.
- **Per-draw**: push *each of the 1,000 (β, λ) pairs* through the curve, get
  1,000 plausible effects, *then* average.

If the curve were a straight line these would agree. It is not, and they don't:
"""),
    code(r"""
post = fit.trace.posterior
b_tv = post["beta_TV"].values.flatten()      # 1,000 plausible betas
lam_tv = post["sat_lam_TV"].values.flatten() # 1,000 plausible curve shapes
x0 = 0.5  # a mid-curve spend level (0 = dark, 1 = TV's biggest week on record)

plug_in = b_tv.mean() * (1 - np.exp(-lam_tv.mean() * x0))   # f(average params)
per_draw = b_tv * (1 - np.exp(-lam_tv * x0))                # f(each draw)...
honest = per_draw.mean()                                    # ...summarize LAST

print(f"plug-in   f(mean beta, mean lambda) = {plug_in:.4f}")
print(f"per-draw  mean of f(beta_i, lambda_i) = {honest:.4f}")
print(f"the plug-in shortcut overstates TV's effect by {plug_in / honest - 1:+.1%}")

# CLAIM: the shortcut is biased here — the curve is concave and beta/lambda
# are correlated, so the curve-of-averages sits above the average-of-curves.
# (Mathematicians file the curvature half under "Jensen's inequality".)
assert plug_in > honest, "expected the plug-in shortcut to overstate"
assert plug_in / honest - 1 > 0.02, "overstatement should be material (>2%)"
print("✓ same posterior, same formula, two answers — only the per-draw one is right")
"""),
    code(r"""
# The picture: 100 plausible response curves, their honest average, and the
# plug-in curve that doesn't correspond to ANY plausible world.
mx_tv = float(mmm._media_raw_max["TV"])           # TV's biggest observed week ($)
s_grid = np.linspace(0, 1.2 * mx_tv, 80)          # weekly spend grid, in dollars
xg = s_grid / mx_tv
curves = mmm.y_std * b_tv[:, None] * (1 - np.exp(-lam_tv[:, None] * xg[None, :]))
mean_curve = curves.mean(axis=0)                                  # per draw, then average
plug_curve = mmm.y_std * b_tv.mean() * (1 - np.exp(-lam_tv.mean() * xg))  # average, then curve

rng = np.random.default_rng(0)
fig = go.Figure()
for k in rng.choice(len(curves), 100, replace=False):
    fig.add_trace(go.Scatter(x=s_grid, y=curves[k], mode="lines", showlegend=False,
                             line={"color": PAL["TV"], "width": 0.7}, opacity=0.16,
                             hoverinfo="skip"))
fig.add_trace(go.Scatter(x=s_grid, y=mean_curve, name="per-draw (honest): average of curves",
                         line={"color": LEAF, "width": 3.5}))
fig.add_trace(go.Scatter(x=s_grid, y=plug_curve, name="plug-in: curve of averaged params",
                         line={"color": BERRY, "width": 3, "dash": "dash"}))
style(fig, "TV response: the plug-in curve floats above every honest summary",
      "weekly TV spend ($)", "weekly contribution (KPI units)", height=430)
fig.show()

gap = plug_curve - mean_curve
print(f"plug-in sits above the honest curve at every positive spend level "
      f"(largest gap: {gap.max():.1f} KPI units/week)")

# CLAIM: the gap is one-sided across the whole spend range, not a fluke at x0.
assert (gap[xg > 0.1] > 0).all()
print("✓ the dashed line is a curve no posterior draw believes in")
"""),
    md(r"""
**Why this is THE transferable skill.** Every number your CMO asks for — ROAS,
an uplift, "which channel is best" — is a *nonlinear* function of parameters
(ratios, curves, rankings, max's). For all of them the plug-in shortcut is
biased, and — worse — it produces a *single* number, hiding how sure you are.
The per-draw recipe fixes both at once: the spread of the 1,000 answers **is**
your uncertainty, for free.

So, the rule, once more, on a card:

> **Compute per draw, summarize last.**
> 1. Write the business formula.
> 2. Evaluate it on *each* posterior draw → a distribution of answers.
> 3. Summarize at the very end: median for "the number", an HDI for "give or
>    take", a probability for "how sure".

The rest of the notebook is this card, applied.
"""),
    # ====================================================================
    # 3. ROAS as a distribution
    # ====================================================================
    md(r"""
## 2 — ROAS is a distribution (and we can grade it)

The fitted model carries a per-draw quantity called `channel_contributions`:
for every posterior draw, every week, every channel, the KPI units that
channel's spend generated. That array is the raw material for nearly every
derived quantity in this notebook. The recipe for ROAS:

1. **Per draw**: sum a channel's weekly contributions over the whole period →
   that draw's total contribution. Divide by the channel's actual total spend
   → *that draw's ROAS*.
2. **Summarize last**: 1,000 draws → a ROAS *distribution* per channel →
   median + 90% **HDI** *(highest-density interval — the narrowest range
   holding 90% of the draws; workshop_04)*.

And because this world is synthetic, the table gets a column real analysts
never see: the **true ROAS**, and whether our interval caught it.
"""),
    code(r"""
# Per-draw channel contributions: dims (chain, draw, week, channel), in the
# model's standardized scale -> x y_std puts them back in KPI units.
cc = post["channel_contributions"].values
CONTRIB = cc.reshape(-1, cc.shape[2], cc.shape[3]) * mmm.y_std  # (1000, weeks, 4)
TOTALS = CONTRIB.sum(axis=1)                                    # (1000 draws, 4 channels)
SPEND = np.array([mmm.X_media_raw[:, i].sum() for i in range(len(sc.channels))])
ROAS = TOTALS / SPEND                                           # 1,000 ROAS values per channel

rows = []
for i, c in enumerate(sc.channels):
    lo, hi = az.hdi(ROAS[:, i], hdi_prob=0.90)
    t = float(sc.true_roas[c])
    rows.append({"channel": c, "ROAS median": np.median(ROAS[:, i]),
                 "90% HDI low": lo, "90% HDI high": hi,
                 "TRUE ROAS": t, "truth inside?": lo <= t <= hi})
roas_tbl = pd.DataFrame(rows).set_index("channel")
display(roas_tbl.round(3))

caught = int(roas_tbl["truth inside?"].sum())
print(f"intervals caught the truth for {caught}/4 channels")

# CLAIM: per-draw ROAS is a full distribution per channel, and its 90% HDIs
# are honest — they cover the known true ROAS for (at least most) channels.
assert ROAS.shape == (1000, 4)
assert caught >= 3, "clean-world ROAS coverage collapsed"
assert (roas_tbl["ROAS median"] > 0).all()
print("✓ ROAS computed per draw — with intervals that pass the answer-key test")
"""),
    code(r"""
# The money chart: four ROAS posteriors side by side, truth marked in ink.
fig = go.Figure()
for i, c in enumerate(sc.channels):
    fig.add_trace(go.Violin(
        y=ROAS[:, i], x0=c, name=c, line_color=PAL[c], fillcolor=PAL[c],
        opacity=0.55, meanline_visible=True, points=False, width=0.9,
        hoverinfo="y+name",
    ))
fig.add_trace(go.Scatter(
    x=sc.channels, y=[sc.true_roas[c] for c in sc.channels],
    mode="markers", name="TRUE ROAS (the answer key)",
    marker={"symbol": "diamond", "size": 13, "color": INK,
            "line": {"width": 1.5, "color": "white"}},
))
style(fig, "ROAS posterior per channel — a distribution, not a number",
      None, "ROAS (KPI units per $)", height=440)
fig.update_layout(violinmode="overlay", showlegend=True)
fig.show()

# CLAIM: the violins genuinely overlap — channel comparisons will need
# probabilities (next section), not a glance at point estimates.
spread = roas_tbl["90% HDI high"] - roas_tbl["90% HDI low"]
assert (spread > 0.1).all(), "intervals are implausibly tight"
print("✓ every channel's ROAS carries real width — remember it when ranking")
"""),
    md(r"""
### 2.1 — The framework's one-liner is the same computation, packaged

You will not hand-roll this every Monday. `MMMAnalyzer(mmm).compute_channel_roi()`
does exactly what we just did: it computes per-draw contributions (by zeroing
each channel's spend and differencing paired posterior predictions — the
**counterfactual** *(the "what if this channel had been dark?" comparison)*),
sums per draw, divides by spend, summarizes last. Two reading notes:

- its `ROI` column is the per-draw **mean** (not median), and
- its `Contribution HDI Low/High` columns are *absolute contributions* (94%
  HDI by default) — divide them by `Total Spend` to get the ROAS band.

If the helper and our hand-rolled numbers agree, we've learned something
valuable: **the framework's helpers ARE the per-draw doctrine, packaged.**
"""),
    code(r"""
analyzer = MMMAnalyzer(mmm)   # the analysis API wraps a *fitted* model
with quiet():
    roi_df = analyzer.compute_channel_roi(random_seed=0)
display(roi_df.round(3))

R = roi_df.set_index("Channel")
for i, c in enumerate(sc.channels):
    # point estimate: helper's ROI == our per-draw mean (same draws, same math)
    assert np.isclose(R.loc[c, "ROI"], ROAS[:, i].mean(), rtol=1e-6), c
    # band: helper's contribution HDI / spend ~= our 94% HDI on per-draw ROAS
    lo94, hi94 = az.hdi(TOTALS[:, i], hdi_prob=0.94)
    assert np.isclose(R.loc[c, "Contribution HDI Low"], lo94, rtol=0.05), c
    assert np.isclose(R.loc[c, "Contribution HDI High"], hi94, rtol=0.05), c

hand = pd.Series(ROAS.mean(axis=0), index=sc.channels, name="hand-rolled per-draw mean")
print(pd.concat([R["ROI"].rename("compute_channel_roi()"), hand], axis=1).round(6))
print("✓ the one-liner reproduces the hand-rolled numbers — use it with confidence")
"""),
    # ====================================================================
    # 4. Decision probabilities
    # ====================================================================
    md(r"""
## 3 — Decision probabilities: numbers your CFO can argue with

Once ROAS is 1,000 draws instead of one number, questions that classically
require a statistics consult become *counting*:

- **"Does Search clear our breakeven multiple?"** → count the draws above the
  threshold: `(ROAS_search > t).mean()`. That is a **decision probability**
  *(the posterior probability that a business statement is true)*.
- **"Is Social actually better than Display?"** → count the draws where it is:
  `(ROAS_social > ROAS_display).mean()`. Crucially this is computed
  **within each draw**, so the comparison automatically respects everything
  the two estimates share (same data, same baseline, workshop_04's joint
  structure).

No p-values, no test statistics, no "significance" rituals — a direct
probability of the thing you actually asked. *"There's an 81% chance Social
out-earns Display per dollar"* is a sentence a CFO can argue with, budget
against, or demand more data about.

One honesty note before the table: in this synthetic world the answer key says
**every channel's true ROAS sits below 1** (scroll back to the §0 table). A
good model should *agree* — watch the `P(ROAS > 1)` row.
"""),
    code(r"""
p_break_even = {c: float((ROAS[:, i] > 1.0).mean()) for i, c in enumerate(sc.channels)}
print("P(ROAS > 1.0):", {c: round(p, 3) for c, p in p_break_even.items()})

# Pairwise "beats" matrix: cell (row, col) = P(row's ROAS > col's ROAS),
# computed draw-by-draw so shared uncertainty cancels fairly.
BEATS = pd.DataFrame(
    {ca: {cb: float((ROAS[:, ia] > ROAS[:, ib]).mean()) for ib, cb in enumerate(sc.channels)}
     for ia, ca in enumerate(sc.channels)}
).T.loc[sc.channels, sc.channels]
BEATS.index.name = "P(row beats column)"
display(BEATS.round(3))

# CLAIM 1: the model agrees with the answer key that no channel clears 1.0 --
# every break-even probability is low (truth: all true ROAS < 1 in this world).
assert (sc.true_roas < 1).all()
assert max(p_break_even.values()) < 0.30
# CLAIM 2: pairwise calls vary from near-certain to genuine toss-ups.
assert BEATS.loc["Social", "TV"] > 0.90, "Social vs TV should be near-certain"
assert 0.55 < BEATS.loc["Social", "Display"] < 0.98, "Social vs Display is a real call"
assert abs(BEATS.loc["Search", "Display"] - 0.5) < 0.35, "expected at least one toss-up"
print("✓ decision probabilities: some calls are near-certain, others are honest coin flips")
"""),
    code(r"""
# Static panel: the full breakeven curve per channel + the beats-matrix heatmap.
t_grid = np.linspace(0.3, 1.3, 101)
p_curves = {c: (ROAS[:, i][None, :] > t_grid[:, None]).mean(axis=1)
            for i, c in enumerate(sc.channels)}

fig = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42],
                    subplot_titles=("P(ROAS > t) as the bar t rises",
                                    "P(row beats column)"))
for c in sc.channels:
    fig.add_trace(go.Scatter(x=t_grid, y=p_curves[c], name=c,
                             line={"color": PAL[c], "width": 2.5}), row=1, col=1)
fig.add_vline(x=1.0, line_dash="dot", line_color=MUTED, row=1, col=1,
              annotation_text="breakeven 1.0")
fig.add_trace(go.Heatmap(
    z=BEATS.values, x=sc.channels, y=sc.channels, zmin=0, zmax=1,
    colorscale=[[0, "white"], [1, SKY]], showscale=False,
    text=np.vectorize(lambda v: f"{v:.0%}")(BEATS.values), texttemplate="%{text}",
), row=1, col=2)
fig.update_yaxes(autorange="reversed", row=1, col=2)
fig.update_xaxes(title_text="ROAS threshold t", row=1, col=1)
fig.update_yaxes(title_text="probability", row=1, col=1)
style(fig, "Decision probabilities, read straight off the draws", height=420)
fig.show()

# CLAIM: each P(ROAS > t) curve is a survival curve -- it can only fall as the
# threshold rises -- and the matrix diagonal is exactly 0 (a draw never
# strictly beats itself).
for c in sc.channels:
    assert (np.diff(p_curves[c]) <= 1e-12).all()
assert all(BEATS.loc[c, c] == 0.0 for c in sc.channels)  # a draw never beats itself
print("✓ static panel drawn — same draws, three thresholds of confidence")
"""),
    md(r"""
### 🎛️ Live exploration (run me!) — your finance team's breakeven bar

Different finance teams put the bar in different places (fully-loaded margin,
contribution margin, "must beat last year's blended 0.7…"). Drag the threshold
and watch which channels survive. The static panel above shows the whole curve
in case you're reading without a live kernel.
"""),
    code(r"""
# 🎛️ Live exploration (run me!) — slider over the breakeven threshold.
from ipywidgets import interact, FloatSlider, Dropdown, IntSlider

def p_above(t):
    return {c: float((ROAS[:, i] > t).mean()) for i, c in enumerate(sc.channels)}

def show_breakeven(threshold=0.8):
    probs = p_above(threshold)
    fig = go.Figure(go.Bar(
        x=sc.channels, y=[probs[c] for c in sc.channels],
        marker_color=[PAL[c] for c in sc.channels],
        text=[f"{probs[c]:.0%}" for c in sc.channels], textposition="outside",
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color=MUTED,
                  annotation_text="coin flip")
    style(fig, f"P(ROAS > {threshold:.2f}) per channel", None, "probability", height=360)
    fig.update_yaxes(range=[0, 1.12])
    fig.show()

interact(show_breakeven,
         threshold=FloatSlider(min=0.4, max=1.2, step=0.05, value=0.8,
                               description="breakeven"));

# CLAIM: the readout reacts sensibly across the slider's range.
assert p_above(0.45)["Social"] > 0.95          # low bar: Social near-certainly clears
assert p_above(1.20)["TV"] < 0.05              # high bar: TV near-certainly doesn't
assert all(p_above(0.40)[c] >= p_above(1.20)[c] for c in sc.channels)
print("✓ slide the bar; the probabilities move, the draws never change")
"""),
    # ====================================================================
    # 5. Marginal ROAS
    # ====================================================================
    md(r"""
## 4 — Marginal ROAS: the next dollar is not the average dollar

The §2 ROAS — total contribution ÷ total spend — is the **average ROAS**: what
each dollar earned *on average, historically*. But your CMO's second question
was about the **next** dollar. Under saturation those differ, always in the
same direction: on a curve that flattens, the next dollar lands on a *flatter*
part than the historical average dollar did. **Marginal ROAS** *(the return on
a small additional dollar of spend, at today's operating point)* is therefore
**below** average ROAS for every channel — and the *gap* differs by channel,
which is exactly what makes rankings flip.

Picture it on the response curve (drawn live in §4.1): average ROAS is the
slope of the **secant** *(the straight line from zero to today's operating
point)*; marginal ROAS is the slope of the **tangent** *(the line that just
kisses the curve at the operating point)*. Flat curve at the point → tangent
slope collapses even while the secant still looks healthy.

The framework's `compute_marginal_contributions(compute_uncertainty=True)`
simulates a +10% spend nudge per channel through **paired** posterior draws
(same seed for baseline and nudged prediction, so the shared noise cancels
draw-by-draw — the per-draw doctrine again) and reports marginal ROAS with
HDIs. We'll also hand-roll the per-draw version to (a) prove it matches and
(b) get something the table doesn't print: *the probability that the ranking
itself flips*.
"""),
    code(r"""
with quiet():
    marg = mmm.compute_marginal_contributions(
        spend_increase_pct=10, compute_uncertainty=True, hdi_prob=0.90, random_seed=0)
MARG = marg.set_index("Channel")
display(marg.round(3))

avg_roas = pd.Series(ROAS.mean(axis=0), index=sc.channels)
cmp_tbl = pd.DataFrame({
    "average ROAS": avg_roas,
    "marginal ROAS": MARG["Marginal ROAS"],
    "marginal / average": MARG["Marginal ROAS"] / avg_roas,
    "rank (average)": avg_roas.rank(ascending=False).astype(int),
    "rank (marginal)": MARG["Marginal ROAS"].rank(ascending=False).astype(int),
})
display(cmp_tbl.round(3))

# Hand-rolled per-draw marginal ROAS: identical recipe, kept as draws.
with quiet():
    p_base0 = mmm.predict(random_seed=0)
base0_tot = p_base0.y_pred_samples.sum(axis=1)
MDRAWS = {}
for i, c in enumerate(sc.channels):
    X_up = mmm.X_media_raw.copy(); X_up[:, i] *= 1.10
    with quiet():
        p_up = mmm.predict(X_media=X_up, random_seed=0)   # PAIRED seed: noise cancels
    MDRAWS[c] = (p_up.y_pred_samples.sum(axis=1) - base0_tot) / (0.10 * SPEND[i])
MDRAWS = pd.DataFrame(MDRAWS)

for c in sc.channels:   # the helper IS the per-draw computation, packaged (again)
    assert np.isclose(MDRAWS[c].mean(), MARG.loc[c, "Marginal ROAS"], rtol=1e-6), c

# CLAIM 1: saturation taxes every next dollar -- marginal < average, all channels.
assert (cmp_tbl["marginal ROAS"] < cmp_tbl["average ROAS"] - 0.05).all()
# CLAIM 2: the best channel is best on BOTH yardsticks in this world...
assert avg_roas.idxmax() == MARG["Marginal ROAS"].idxmax() == "Social"
# CLAIM 3: ...but the BOTTOM of the ranking flips: Search out-earns TV per
# historical dollar, yet its curve is the most saturated, so TV's NEXT dollar
# beats Search's. (Point ranking is seed-pinned; the probability is the truth.)
assert cmp_tbl["marginal / average"].idxmin() == "Search", "Search takes the biggest haircut"
assert avg_roas["Search"] > avg_roas["TV"]
assert MARG.loc["TV", "Marginal ROAS"] > MARG.loc["Search", "Marginal ROAS"]
p_flip = float((MDRAWS["TV"] > MDRAWS["Search"]).mean())
print(f"P(TV's next dollar beats Search's next dollar) = {p_flip:.2f}")
assert 0.50 < p_flip < 0.95, "the flip should be probable but not certain"
print("✓ the ranking flipped at the bottom — and we know HOW confident to be about it")
"""),
    code(r"""
# Average vs marginal, side by side, with 90% HDIs on both.
fig = go.Figure()
avg_lo = np.array([az.hdi(ROAS[:, i], hdi_prob=0.90)[0] for i in range(4)])
avg_hi = np.array([az.hdi(ROAS[:, i], hdi_prob=0.90)[1] for i in range(4)])
fig.add_trace(go.Bar(
    x=sc.channels, y=avg_roas.values, name="average ROAS (historical dollar)",
    marker_color=[PAL[c] for c in sc.channels], opacity=0.85,
    error_y={"type": "data", "symmetric": False,
             "array": avg_hi - avg_roas.values, "arrayminus": avg_roas.values - avg_lo,
             "color": INK, "thickness": 1.4},
))
fig.add_trace(go.Bar(
    x=sc.channels, y=MARG["Marginal ROAS"].values, name="marginal ROAS (next dollar)",
    marker_color=[PAL[c] for c in sc.channels], opacity=0.40,
    marker_pattern_shape="/",
    error_y={"type": "data", "symmetric": False,
             "array": (MARG["Marginal ROAS HDI High"] - MARG["Marginal ROAS"]).values,
             "arrayminus": (MARG["Marginal ROAS"] - MARG["Marginal ROAS HDI Low"]).values,
             "color": INK, "thickness": 1.4},
))
style(fig, "Every next dollar earns less than the historical average dollar",
      None, "ROAS (KPI units per $)", height=430)
fig.update_layout(barmode="group")
fig.show()

# CLAIM: the chart encodes the same flip the table showed (solid vs hatched).
assert len(fig.data) == 2 and len(fig.data[0].y) == 4
print("✓ solid bars rank one way at the bottom; hatched bars rank the other way")
"""),
    md(r"""
### 4.1 — Secant vs tangent, on the actual curve

Why is Search hit hardest? Because it *operates deepest into its curve*. Below:
Search's posterior-median response curve, the secant (slope ≈ average return
per weekly dollar) and the tangent at the operating point (slope ≈ the next
dollar's return). The two slopes answer different questions; under saturation
they must disagree, and the flatter the curve at your operating point, the
bigger the disagreement.

(Units note: this view is a *steady-state weekly* simplification — constant
spend, carryover settled — so its slopes won't numerically equal the §4 table,
which nudges the real, varying 156-week calendar. Same concepts, cleaner
picture; trust the table for the numbers.)
"""),
    code(r"""
ch = "Search"
i_ch = sc.channels.index(ch)
b_s = post[f"beta_{ch}"].values.flatten()
lam_s = post[f"sat_lam_{ch}"].values.flatten()
mx_s = float(mmm._media_raw_max[ch])
s_grid = np.linspace(0, 1.15 * mx_s, 90)
curve_med = mmm.y_std * np.median(
    b_s[:, None] * (1 - np.exp(-lam_s[:, None] * (s_grid / mx_s)[None, :])), axis=0)

s_op = mmm.X_media_raw[:, i_ch].mean()              # today's typical week ($)
f_op = float(np.interp(s_op, s_grid, curve_med))
secant_slope = f_op / s_op
tangent_slope = float(np.median(mmm.y_std * b_s * lam_s * np.exp(-lam_s * s_op / mx_s) / mx_s))
tan_x = np.array([max(s_op - 0.45 * mx_s, 0), min(s_op + 0.45 * mx_s, s_grid[-1])])
tan_y = f_op + tangent_slope * (tan_x - s_op)

fig = go.Figure()
fig.add_trace(go.Scatter(x=s_grid, y=curve_med, name=f"{ch} response curve (posterior median)",
                         line={"color": PAL[ch], "width": 3}))
fig.add_trace(go.Scatter(x=[0, s_op], y=[0, f_op], name="secant — average ROAS",
                         line={"color": LEAF, "width": 2.5, "dash": "dot"}))
fig.add_trace(go.Scatter(x=tan_x, y=tan_y, name="tangent — marginal ROAS",
                         line={"color": BERRY, "width": 2.5, "dash": "dash"}))
fig.add_trace(go.Scatter(x=[s_op], y=[f_op], mode="markers", name="operating point (avg week)",
                         marker={"color": INK, "size": 12, "symbol": "diamond"}))
style(fig, f"{ch}: the secant is what you earned; the tangent is what you'll earn next",
      f"weekly {ch} spend ($)", "weekly contribution (KPI units)", height=430)
fig.show()

print(f"secant slope  (≈ avg return / weekly $)      : {secant_slope:.3f}")
print(f"tangent slope (≈ next dollar's return / $)   : {tangent_slope:.3f}")
# CLAIM: at a positive operating point on a concave curve, tangent < secant --
# the geometric reason marginal ROAS must sit below average ROAS.
assert 0 < tangent_slope < secant_slope
print("✓ the picture and the §4 table tell one story: the next dollar earns less")
"""),
    md(r"""
**Honesty checkpoint.** In this world the *top* of the ranking didn't move —
Social wins on both yardsticks — and the flip happened at the *bottom*
(TV vs Search), with the per-draw probability telling us it's roughly a 6-in-10
call rather than a certainty. When would the flip hit the top? Whenever your
historically-best channel is also your most-saturated one — very common for
brands that scaled their proven winner for years. Then average ROAS says
"double down" while marginal ROAS says "you already ate that curve" — and only
the marginal number is about the *future*. That, plus an honest probability on
the call, is what this section is for.
"""),
    # ====================================================================
    # 6. Response curves with bands
    # ====================================================================
    md(r"""
## 5 — Response curves with bands: who has room to grow

Tangent-vs-secant was one channel at one point. The full deliverable is each
channel's **response curve** *(expected weekly contribution as a function of
weekly spend)* drawn **per draw** — so the curve comes with a credibility band,
not just a line. Read each panel with two questions:

- **Where is the operating point?** (diamond = today's average week)
- **Is the curve still climbing there, or flat?** Steep → headroom: the next
  dollar still buys real lift. Flat → saturated: more budget mostly buys
  confirmation that the curve is flat.

The printed `saturation at op` is the share of each channel's estimated
ceiling already consumed at its typical week (0 = untouched, 1 = fully
saturated). Same steady-state caveat as §4.1 applies.
"""),
    code(r"""
fig = make_subplots(rows=2, cols=2, subplot_titles=sc.channels,
                    horizontal_spacing=0.09, vertical_spacing=0.14)
satfrac = {}
for i, c in enumerate(sc.channels):
    r, col = i // 2 + 1, i % 2 + 1
    b_c = post[f"beta_{c}"].values.flatten()
    lam_c = post[f"sat_lam_{c}"].values.flatten()
    mx = float(mmm._media_raw_max[c])
    sg = np.linspace(0, 1.25 * mx, 70)
    fan = mmm.y_std * b_c[:, None] * (1 - np.exp(-lam_c[:, None] * (sg / mx)[None, :]))
    lo5, lo25, med, hi75, hi95 = np.percentile(fan, [5, 25, 50, 75, 95], axis=0)
    s_op = mmm.X_media_raw[:, i].mean()
    satfrac[c] = float(np.median(1 - np.exp(-lam_c * s_op / mx)))

    fig.add_trace(go.Scatter(x=np.r_[sg, sg[::-1]], y=np.r_[hi95, lo5[::-1]],
                             fill="toself", fillcolor=PAL[c], opacity=0.18,
                             line={"width": 0}, hoverinfo="skip",
                             name="90% band", showlegend=(i == 0)), row=r, col=col)
    fig.add_trace(go.Scatter(x=np.r_[sg, sg[::-1]], y=np.r_[hi75, lo25[::-1]],
                             fill="toself", fillcolor=PAL[c], opacity=0.30,
                             line={"width": 0}, hoverinfo="skip",
                             name="50% band", showlegend=(i == 0)), row=r, col=col)
    fig.add_trace(go.Scatter(x=sg, y=med, line={"color": PAL[c], "width": 2.5},
                             name="median curve", showlegend=(i == 0)), row=r, col=col)
    fig.add_trace(go.Scatter(x=[s_op], y=[np.interp(s_op, sg, med)], mode="markers",
                             marker={"color": INK, "size": 11, "symbol": "diamond"},
                             name="operating point", showlegend=(i == 0)), row=r, col=col)

    # CLAIM per channel: the curve rises (median) and flattens (mean is concave;
    # the mean of concave curves is concave by construction -- the median can
    # wiggle microscopically where draws cross).
    assert (np.diff(med) > -1e-9).all(), f"{c}: curve should be non-decreasing"
    assert (np.diff(fan.mean(axis=0), 2) < 1e-9).all(), f"{c}: curve should be concave"

style(fig, "Response curves from posterior draws — diamond marks today's average week",
      height=620)
fig.update_xaxes(title_text="weekly spend ($)", row=2)
fig.update_yaxes(title_text="weekly contribution", col=1)
fig.show()

sat_tbl = pd.Series(satfrac, name="saturation at op").sort_values(ascending=False)
print(sat_tbl.round(2).to_string())
# CLAIM: Search operates deepest into its curve; Social shallowest -- which is
# exactly the §4 marginal-ROAS story told geometrically.
assert sat_tbl.idxmax() == "Search" and sat_tbl.idxmin() == "Social"
assert satfrac["Search"] > satfrac["Social"] + 0.10
assert all(0 < v < 1 for v in satfrac.values())
print("✓ steep-curve channels have headroom; Search has eaten most of its curve")
"""),
    # ====================================================================
    # 7. What-if scenarios
    # ====================================================================
    md(r"""
## 6 — What-if scenarios: uplift is a distribution too

"What if we put 20% more into Social next year?" The framework one-liner is
`mmm.what_if_scenario({"Social": 1.2})` — a dict of **spend multipliers** per
channel. It returns a plain dict: `baseline_outcome`, `scenario_outcome`,
`outcome_change(_pct)`, a `spend_changes` breakdown, and the two posterior-mean
weekly prediction paths.

Useful — but it summarizes to a *point*. You know the doctrine by now: the
uplift deserves a distribution. The hand-rolled version is two **paired**
`mmm.predict()` calls (same `random_seed`, so each draw's simulated noise
cancels in the difference) and a per-draw subtraction. Same per-draw recipe
the marginal table used; we just keep the draws this time.
"""),
    code(r"""
with quiet():
    wi = mmm.what_if_scenario({"Social": 1.2}, random_seed=123)
print("what_if_scenario returns:", sorted(wi.keys()))
print(f"point answer: {wi['outcome_change']:+,.0f} KPI units "
      f"({wi['outcome_change_pct']:+.2f}%) for "
      f"${wi['spend_changes']['Social']['change']:,.0f} more Social spend")

# Hand-rolled uplift DISTRIBUTIONS: +20% on each channel, one at a time.
UPLIFT_SEED = 123
with quiet():
    P0 = mmm.predict(random_seed=UPLIFT_SEED)
BASE_TOT = P0.y_pred_samples.sum(axis=1)            # per-draw baseline totals
UP20, DOLLARS20 = {}, {}
for i, c in enumerate(sc.channels):
    X_s = mmm.X_media_raw.copy(); X_s[:, i] *= 1.20
    with quiet():
        p_s = mmm.predict(X_media=X_s, random_seed=UPLIFT_SEED)  # PAIRED
    UP20[c] = p_s.y_pred_samples.sum(axis=1) - BASE_TOT
    DOLLARS20[c] = 0.20 * SPEND[i]

up_tbl = pd.DataFrame({
    "extra spend ($)": pd.Series(DOLLARS20),
    "mean uplift": {c: UP20[c].mean() for c in sc.channels},
    "90% HDI low": {c: az.hdi(UP20[c], hdi_prob=0.90)[0] for c in sc.channels},
    "90% HDI high": {c: az.hdi(UP20[c], hdi_prob=0.90)[1] for c in sc.channels},
    "P(uplift > 0)": {c: float((UP20[c] > 0).mean()) for c in sc.channels},
}).loc[sc.channels]
display(up_tbl.round(2))

# CLAIM 1: the one-liner's point answer IS our per-draw mean (same seed, same
# paired draws) -- the helper and the doctrine are the same computation.
assert np.isclose(wi["outcome_change"], UP20["Social"].mean(), rtol=1e-6)
# CLAIM 2: more spend on a positive-effect channel lifts the KPI: every
# uplift distribution sits clearly above zero -- and Social (the marginal-ROAS
# winner with high P from §4) is near-certain.
assert (up_tbl["P(uplift > 0)"] > 0.90).all()
assert up_tbl.loc["Social", "P(uplift > 0)"] > 0.95
print("✓ a what-if answer with a width — not just a point")
"""),
    code(r"""
# Static panel: the four +20% uplift distributions, absolute and per-dollar.
fig = make_subplots(rows=1, cols=2, subplot_titles=(
    "uplift of +20% spend (KPI units)", "same uplift per extra dollar"))
for c in sc.channels:
    gx, gy = kde_xy(UP20[c])
    fig.add_trace(go.Scatter(x=gx, y=gy, name=c, line={"color": PAL[c], "width": 2.5},
                             fill="tozeroy", opacity=0.55), row=1, col=1)
    gx2, gy2 = kde_xy(UP20[c] / DOLLARS20[c])
    fig.add_trace(go.Scatter(x=gx2, y=gy2, name=c, line={"color": PAL[c], "width": 2.5},
                             showlegend=False), row=1, col=2)
fig.add_vline(x=0, line_color=INK, line_width=1.5, row=1, col=1)
fig.add_vline(x=0, line_color=INK, line_width=1.5, row=1, col=2)
style(fig, "Same +20%, two readings: total lift vs efficiency of the extra dollars",
      height=400)
fig.update_yaxes(showticklabels=False)
fig.show()

abs_mean = {c: float(UP20[c].mean()) for c in sc.channels}
per_dollar = {c: float(UP20[c].mean() / DOLLARS20[c]) for c in sc.channels}
print("mean uplift  :", {c: round(v) for c, v in abs_mean.items()})
print("per extra $  :", {c: round(v, 3) for c, v in per_dollar.items()})

# CLAIM: "+20%" means very different DOLLARS per channel. TV's +20% is the
# biggest check, so it buys the biggest absolute lift -- while per dollar it
# is one of the weakest. Per-dollar uplift is just marginal ROAS again
# (Social > Search, matching §4's ranking).
assert abs_mean["TV"] > abs_mean["Search"]
assert per_dollar["Social"] > per_dollar["Search"] + 0.1
print("✓ absolute lift rewards big budgets; per-dollar lift reveals efficiency")
"""),
    md(r"""
### 🎛️ Live exploration (run me!) — move a budget, watch the uplift

Pick a channel and a budget change; each move re-runs the paired what-if
against the same baseline draws and redraws the uplift distribution (about a
second per move — it's two posterior-predictive passes). Try Social +30%
versus TV +30%, then try *cutting* Search 20% and notice the distribution
lands further from zero than the +20% case — on a concave curve, the dollars
you remove were earning more than the dollars you'd add. The static panel
above is the +20% snapshot of the same machinery.
"""),
    code(r"""
# 🎛️ Live exploration (run me!) — per-channel budget slider -> uplift density.
def show_move(channel="Social", pct_change=20):
    if pct_change == 0:
        print("a 0% change is a $0 scenario — drag the slider")
        return
    i = sc.channels.index(channel)
    X_s = mmm.X_media_raw.copy(); X_s[:, i] *= 1 + pct_change / 100
    with quiet():
        p_s = mmm.predict(X_media=X_s, random_seed=UPLIFT_SEED)  # paired vs BASE_TOT
    up = p_s.y_pred_samples.sum(axis=1) - BASE_TOT
    lo, hi = az.hdi(up, hdi_prob=0.90)
    gx, gy = kde_xy(up)
    fig = go.Figure(go.Scatter(x=gx, y=gy, fill="tozeroy",
                               line={"color": PAL[channel], "width": 2.5}, name=channel))
    fig.add_vline(x=0, line_color=INK, line_width=1.5)
    fig.add_vline(x=float(up.mean()), line_dash="dot", line_color=MUTED,
                  annotation_text=f"mean {up.mean():+,.0f}")
    style(fig, f"{channel} {pct_change:+d}% (${pct_change/100*SPEND[i]:+,.0f}): "
               f"P(uplift > 0) = {(up > 0).mean():.0%},  90% HDI [{lo:,.0f}, {hi:,.0f}]",
          "total KPI uplift vs current plan", None, height=360)
    fig.update_yaxes(showticklabels=False)
    fig.show()

interact(show_move,
         channel=Dropdown(options=sc.channels, value="Social", description="channel"),
         pct_change=IntSlider(min=-50, max=50, step=10, value=20,
                              description="spend %"));

# CLAIM: the machinery is symmetric and honest -- CUTTING a positive-effect
# channel hurts (mean uplift < 0, with high probability).
i_so = sc.channels.index("Social")
X_cut = mmm.X_media_raw.copy(); X_cut[:, i_so] *= 0.80
with quiet():
    p_cut = mmm.predict(X_media=X_cut, random_seed=UPLIFT_SEED)
down = p_cut.y_pred_samples.sum(axis=1) - BASE_TOT
assert down.mean() < 0 and (down < 0).mean() > 0.90
assert abs(down.mean()) > abs(UP20["Social"].mean()) * 0.8  # concavity: cuts bite hard
print(f"✓ Social -20%: mean {down.mean():+,.0f}, P(loss) = {(down < 0).mean():.0%} — cuts bite")
"""),
    # ====================================================================
    # 8. Reallocation with honesty
    # ====================================================================
    md(r"""
## 7 — A simple reallocation, with honesty

The capstone scenario: **move budget, keep the total constant**. Section 4
nominated the donor and the recipient: Search's next dollar earns the least
(most saturated), Social's earns the most. So: cut Search 20% and give those
dollars to Social. Total spend unchanged — pure reallocation.

And as a control on ourselves, the *plausible-looking mistake*: rank channels
by **average** ROAS instead. On that table Search out-earns TV, so a naive
analyst happily moves TV money *into* Search — into the most saturated curve
on the books. Both scenarios get the full per-draw treatment; both get graded
the same way.
"""),
    code(r"""
def realloc(frm, to, cut):
    # Shift `cut` of channel `frm`'s budget onto `to`; total spend unchanged.
    fi, ti = sc.channels.index(frm), sc.channels.index(to)
    X_r = mmm.X_media_raw.copy()
    dollars = cut * X_r[:, fi].sum()
    X_r[:, fi] *= (1 - cut)
    X_r[:, ti] *= 1 + dollars / X_r[:, ti].sum()
    assert np.isclose(X_r.sum(), mmm.X_media_raw.sum()), "reallocation must conserve budget"
    with quiet():
        p_r = mmm.predict(X_media=X_r, random_seed=UPLIFT_SEED)  # paired vs BASE_TOT
    return p_r.y_pred_samples.sum(axis=1) - BASE_TOT, dollars

GOOD_UP, good_dollars = realloc("Search", "Social", 0.20)   # marginal-ROAS-driven
BAD_UP, bad_dollars = realloc("TV", "Search", 0.20)         # average-ROAS-driven

verdict = pd.DataFrame({
    "moved ($)": [good_dollars, bad_dollars],
    "mean gain": [GOOD_UP.mean(), BAD_UP.mean()],
    "90% HDI low": [az.hdi(GOOD_UP, hdi_prob=0.90)[0], az.hdi(BAD_UP, hdi_prob=0.90)[0]],
    "90% HDI high": [az.hdi(GOOD_UP, hdi_prob=0.90)[1], az.hdi(BAD_UP, hdi_prob=0.90)[1]],
    "P(gain > 0)": [(GOOD_UP > 0).mean(), (BAD_UP > 0).mean()],
}, index=["Search → Social (marginal-driven)", "TV → Search (average-driven)"])
display(verdict.round(2))

# CLAIM 1: the marginal-driven move is a probable win at constant total spend.
assert GOOD_UP.mean() > 0 and (GOOD_UP > 0).mean() > 0.85
# CLAIM 2: the average-ROAS-driven move is NOT -- its gain interval straddles
# zero and the odds tilt toward a loss. Average ROAS pointed the money at the
# most saturated curve; marginal ROAS would never have.
bad_lo, bad_hi = az.hdi(BAD_UP, hdi_prob=0.90)
assert bad_lo < 0 < bad_hi, "expected the naive move's interval to straddle zero"
assert (BAD_UP > 0).mean() < 0.5
print("✓ same dollars, two rankings: one probable win, one coin flip tilted to a loss")
"""),
    code(r"""
# The closing chart: two reallocation gain distributions against the zero line.
fig = go.Figure()
for up, name, color in ((GOOD_UP, "Search → Social (marginal-driven)", LEAF),
                        (BAD_UP, "TV → Search (average-driven)", BERRY)):
    gx, gy = kde_xy(up)
    fig.add_trace(go.Scatter(x=gx, y=gy, name=name, fill="tozeroy", opacity=0.55,
                             line={"color": color, "width": 2.5}))
    fig.add_annotation(x=float(up.mean()), y=float(gy.max()) * 1.04,
                       text=f"P(gain>0) = {(up > 0).mean():.0%}", showarrow=False,
                       font={"color": color, "size": 12})
fig.add_vline(x=0, line_color=INK, line_width=2)
style(fig, "Budget reallocation at constant total spend: the interval IS the deliverable",
      "total KPI gain vs current plan", None, height=420)
fig.update_yaxes(showticklabels=False)
fig.show()

# CLAIM: the two densities sit on opposite sides of zero by their bulk.
assert np.median(GOOD_UP) > 0 > np.median(BAD_UP)
print("✓ this chart is the CMO answer: what to move, what it's worth, how sure we are")
"""),
    md(r"""
**The final teaching beat — read the chart like a grown-up.**

- **The interval is the deliverable.** "Move Search money to Social" is worth
  recommending *because* its gain distribution clears zero with high
  probability — and even then, look how wide it is. Report the width. A
  reallocation whose gain interval straddles zero is a **coin flip**, and
  recommending a coin flip as a sure thing is how MMM teams lose credibility.
- **Bigger moves buy more risk, not more certainty.** Re-run §7's helper with
  a 30% cut and watch the interval widen toward zero faster than the mean
  grows — the recipient's curve flattens as you pile dollars onto it.
- **All of this assumed the model is causally right.** Every distribution in
  this notebook is honest *about the model's own uncertainty* — sampling
  noise, parameter trade-offs. None of it defends against the model being
  *wrong about causality*: this clean world has no hidden demand driver, but
  real worlds do, and the stress series shows that a confounded model produces
  beautifully tight intervals around the wrong answer
  ([stress_00](stress_00_the_rosy_picture.ipynb) for the doctrine,
  [stress_05](stress_05_the_gauntlet.ipynb) for everything-at-once). The fix
  is upstream of any chart here: causal structure, data hygiene, and
  randomized experiments folded into the model as calibration
  ([math_05](math_05_calibration.ipynb)).
"""),
    # ====================================================================
    # 9. Series close
    # ====================================================================
    md(r"""
## 8 — The whole arc, and where to go next

You came in six notebooks ago with no Bayesian background. Here is the arc you
just walked:

| # | notebook | the one-line takeaway |
|---|---|---|
| 00 | Thinking in Distributions | uncertainty is a distribution, and Bayes' rule is how evidence updates it |
| 01 | Priors | what you believed before the data is a model input — own it, test it |
| 02 | Sampling | MCMC turns "impossible integral" into "draws you can count" — if the diagnostics pass |
| 03 | Your First MMM | adstock + saturation + a Bayesian regression, fit on a world with an answer key |
| 04 | Reading the Posterior | draws, HDIs, and the joint structure between parameters — the posterior is the product |
| 05 | From Draws to Decisions *(this one)* | **compute per draw, summarize last** — ROAS, probabilities, marginal dollars, reallocations |

**Where to go next in this repo:**

- **The Aurora story series** ([00_overview](00_overview.ipynb) →
  05_unified_workflow) — the same framework on a richer fictional brand,
  told as a full analyst engagement.
- **The math series** ([math_00](math_00_overview.ipynb)…math_06) — every
  formula this workshop hand-waved, derived properly: adstock, saturation,
  the likelihood, calibration with experiments, extensions.
- **The stress series** ([stress_00](stress_00_the_rosy_picture.ipynb)…stress_06)
  — what happens when the model's assumptions are *wrong* and the diagnostics
  stay green; required reading before you trust any real-data MMM.
- **[mmm_walkthrough.ipynb](mmm_walkthrough.ipynb)** — the practitioner's
  end-to-end workflow on realistic data, v1 → v3.
"""),
    md(r"""
## Glossary

| term | plain English |
|---|---|
| **derived quantity** | any business number computed *from* model parameters (ROAS, an uplift, a ranking) rather than estimated directly |
| **per-draw computation** | evaluating the business formula once per posterior draw, yielding a distribution of answers; summarize (median/HDI/probability) only at the end |
| **plug-in shortcut** | the biased alternative: collapse the posterior to one "best" value first, then compute — `f(average) ≠ average of f` for nonlinear `f` |
| **average ROAS** | total contribution ÷ total spend: what the *historical* dollar earned (the secant's slope) |
| **marginal ROAS** | the return on the *next* dollar at today's operating point (the tangent's slope); always below average ROAS under saturation |
| **response curve** | expected weekly contribution as a function of weekly spend; drawn per draw it comes with a credibility band |
| **operating point** | where on its response curve a channel currently spends; steep = headroom, flat = saturated |
| **what-if / counterfactual** | re-predicting the KPI under altered spend (paired draws so noise cancels); the difference per draw is the uplift distribution |
| **decision probability** | the posterior probability of a business statement — P(ROAS > breakeven), P(A beats B), P(reallocation gains) — computed by counting draws |
| **HDI** | highest-density interval: the narrowest range holding a given share (e.g. 90%) of the draws — the honest "give or take" |

*Built on `mmm_framework` — model: `BayesianMMM`; helpers used here:
`MMMAnalyzer.compute_channel_roi`, `compute_marginal_contributions`,
`what_if_scenario`, `predict` (paired seeds), and the
`channel_contributions` posterior. The known-truth world is
`tests/synth/dgp.build("clean")`.*
"""),
]


def main():
    nb = new_notebook(cells=CELLS)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "title": "Workshop 05 — From Draws to Decisions",
    })
    path = "workshop_05_from_draws_to_decisions.ipynb"
    with open(path, "w") as fh:
        nbformat.write(nb, fh)
    print(f"wrote {path} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
