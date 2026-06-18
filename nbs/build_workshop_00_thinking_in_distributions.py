"""Author workshop notebook 00 — Thinking in Distributions (run from ``nbs/``).

    uv run python build_workshop_00_thinking_in_distributions.py
    PYTHONPATH=.. uv run jupyter nbconvert --to notebook --execute --inplace \
        workshop_00_thinking_in_distributions.ipynb --ExecutePreprocessor.timeout=2400

Notebook 00 of the 6-part *workshop* series: the Bayesian causal workflow for
marketing analysts with no Bayesian background. This opener teaches probability
as belief, Bayes' rule on a grid, priors (and how data overwhelms them),
credible intervals / HDIs, and Monte Carlo for derived business quantities —
pure numpy/scipy, no PyMC, no framework imports. Plotly for the main teaching
charts; ipywidgets sliders for live exploration, each paired with a static
panel so the baked notebook teaches without a kernel.

Authored as md/code cells via nbformat (pattern: ``build_stress_00_rosy_picture.py``).
Every computational cell ends with asserts encoding the claim it just made.
All randomness seeded with ``np.random.default_rng(0)``.
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
# Workshop 00 — Thinking in Distributions

Your team just ran an A/B test on two ad creatives. The new creative (B) got a
**5.4% click-through rate**; the old one (A) got **5.1%**. Your media buyer
asks the only question that matters:

> *"Is B actually better — and how sure are we?"*

"5.4 is bigger than 5.1" is not an answer. Both numbers came from a finite
sample; run the test again and you'd get different numbers. The honest answer
is a **probability**: *"there's an X% chance B is better, and if it is, the
upside is roughly Y dollars a month, give or take Z."* This notebook teaches
you the machinery that produces sentences like that.

Why does this matter for **Marketing Mix Modeling (MMM)**? Because every MMM
decision — shift budget to Search, cut Display, scale TV — is a bet placed
under uncertainty. The whole point of the *Bayesian* approach used in
`mmm_framework` is that the model's output is not a single ROAS number but a
**distribution** of plausible ROAS values, and decisions get made from the
whole distribution. This is workshop 0 of 6:

| # | notebook | what you'll learn |
|---|---|---|
| **00** | **Thinking in Distributions** *(this one)* | Bayes' rule, priors, posteriors, credible intervals, Monte Carlo |
| 01 | Priors | choosing priors, prior predictive checks, the MMM prior families |
| 02 | Sampling | why MCMC exists, a Metropolis demo, NUTS, diagnostics |
| 03 | Your First MMM | MMM terminology + first `BayesianMMM` fit on a known-truth world |
| 04 | Reading the Posterior | draws, HDIs, forest plots, learning, posterior predictive checks |
| 05 | From Draws to Decisions | derived quantities per draw, ROAS distributions, decisions |

**No Bayesian background assumed.** Everything here is plain `numpy` + `scipy`
— no MCMC, no PyMC, no framework code. Total runtime: under a minute.
"""),
    code(r"""
import numpy as np
from scipy import stats

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.renderers.default = "notebook_connected"  # interactive charts survive baking

rng = np.random.default_rng(0)  # one seed for every random draw in this notebook

# The series palette — same colors in every figure.
INK, SKY, BERRY, LEAF, AMBER, MUTED = (
    "#2b2118", "#3b6ea5", "#a63a50", "#3f7d5e", "#d98a2b", "#8a8079")

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

THETA = np.linspace(0, 1, 2001)  # a fine grid over "all possible CTR values"

print("numpy", np.__version__, "| scipy:", stats.beta.__class__.__module__.split('.')[0])
print("palette:", INK, SKY, BERRY, LEAF, AMBER)

# CLAIM: setup is deterministic and the grid covers [0, 1] finely.
assert THETA[0] == 0.0 and THETA[-1] == 1.0 and len(THETA) == 2001
assert rng.bit_generator.seed_seq.entropy == 0
print("setup ok — seeded, gridded, styled")
"""),
    # ====================================================================
    # 1. Probability as belief
    # ====================================================================
    md(r"""
## 1 — Probability as belief: distributions over things you don't know

The click-through rate of creative B is a fixed fact about the world — you just
don't know it. Classical statistics treats it as an unknown constant and only
lets you make probability statements about *data*. The Bayesian move is
simpler and more natural: **use probability to describe your uncertainty about
the quantity itself**.

So instead of one number for B's CTR, you hold a **probability distribution**
— *a curve over all possible values, where height means plausibility* — over
it. A **random variable** here is just *a quantity you're uncertain about*
(a CTR, a channel's ROAS, next quarter's sales lift), and the distribution is
your current state of knowledge about it.

The shape of the curve **is** the message:

- A **wide** distribution says *"could be anywhere in this range"* — you know
  little. Acting on its peak alone is gambling.
- A **narrow** distribution says *"pinned down to here"* — you know a lot.

Below: three different states of knowledge about a CTR, all centered on the
same best guess of 5%. Hover over the curves — same peak, wildly different
business implications.
"""),
    code(r"""
# Three states of knowledge about a CTR, all with mean 5%.
# Beta(a, b) is the standard distribution for rates: it lives on [0, 1].
beliefs = {
    "barely a guess  (≈20 impressions of evidence)":  (stats.beta(1, 19),    SKY),
    "some history    (≈200 impressions of evidence)": (stats.beta(10, 190),  AMBER),
    "well measured   (≈2000 impressions of evidence)":(stats.beta(100, 1900),BERRY),
}

fig = go.Figure()
for name, (dist, color) in beliefs.items():
    fig.add_trace(go.Scatter(
        x=THETA * 100, y=dist.pdf(THETA), name=name, mode="lines",
        line={"color": color, "width": 2.5},
        hovertemplate="CTR %{x:.2f}%<br>plausibility %{y:.1f}<extra>" + name + "</extra>"))
fig.add_vline(x=5, line_dash="dot", line_color=MUTED,
              annotation_text="same best guess: 5%")
style(fig, "Three beliefs about one CTR — same center, very different certainty",
      "click-through rate (%)", "plausibility (probability density)")
fig.update_xaxes(range=[0, 15])
fig.show()

means = [d.mean() for d, _ in beliefs.values()]
sds = [d.std() for d, _ in beliefs.values()]
print("means:", np.round(means, 4), "  (identical by construction)")
print("standard deviations:", np.round(sds, 4), "  (shrinking = more certain)")

# CLAIM: all three beliefs share the same mean, and certainty (1/sd) grows
# with the amount of evidence behind the belief.
assert np.allclose(means, 0.05)
assert sds[0] > sds[1] > sds[2]
print("ok — same center, monotonically narrowing uncertainty")
"""),
    md(r"""
Read those three curves as business situations:

- **Wide (blue)** — a brand-new creative, no history. A 5% CTR is your best
  guess, but 2% or 10% are completely live possibilities. You would not promise
  the CMO anything based on this.
- **Medium (amber)** — a few days of data. The curve has pulled in; 10% is now
  implausible, but 4% vs 6% — a 50% difference in clicks! — is still open.
- **Narrow (berry)** — a mature campaign. The CTR is effectively known to
  within a few tenths of a point. Forecasts built on this are safe.

Every Bayesian analysis is a machine for moving **left curve → right curve**:
start with what you believe, feed in data, end with a sharper belief. The rule
that does the moving is next.
"""),
    # ====================================================================
    # 2. Bayes' rule, visually
    # ====================================================================
    md(r"""
## 2 — Bayes' rule, visually

Three ingredients, three names you'll use for the rest of this series:

- **Prior** — *your belief about the quantity before seeing the new data*,
  expressed as a distribution. Ours: a CTR around 5%, loosely held (like the
  blue-ish curve above) — "new creatives in this account usually land near 5%".
- **Likelihood** — *for each candidate value, how probable is the data you
  actually observed?* If the true CTR were 1%, seeing 11 clicks in 200
  impressions would be a shock; if it were 5.5%, it'd be typical. The
  likelihood scores every candidate this way.
- **Posterior** — *your updated belief after the data*: prior reweighted by
  likelihood.

**Bayes' rule** is just the recipe for combining them:

$$\underbrace{P(\theta \mid \text{data})}_{\text{posterior}} \;\propto\; \underbrace{P(\text{data} \mid \theta)}_{\text{likelihood}} \times \underbrace{P(\theta)}_{\text{prior}}$$

where $\theta$ ("theta") is the unknown CTR and $\propto$ means "proportional
to — multiply the curves pointwise, then rescale so the result is a proper
distribution". That's the whole rule. Let's watch it run on the creative-B
data: **11 clicks out of 200 impressions** (a 5.5% observed rate).
"""),
    code(r"""
# Bayes' rule on a grid: multiply two curves, rescale. That's it.
PRIOR_A, PRIOR_B = 2.0, 38.0          # Beta(2, 38): mean 5%, loosely held
k_obs, n_obs = 11, 200                # observed: 11 clicks in 200 impressions

prior_pdf   = stats.beta(PRIOR_A, PRIOR_B).pdf(THETA)
likelihood  = stats.binom.pmf(k_obs, n_obs, THETA)          # score every candidate CTR
posterior   = prior_pdf * likelihood                        # Bayes' rule (unnormalized)
posterior  /= np.trapezoid(posterior, THETA)                # rescale to a proper density

like_scaled = likelihood / np.trapezoid(likelihood, THETA)  # same scale, for plotting

fig = go.Figure()
for y, name, color, dash in [
    (prior_pdf,   "prior — belief before data",        SKY,   "dot"),
    (like_scaled, "likelihood — what the data says",   AMBER, "dash"),
    (posterior,   "posterior — belief after data",     BERRY, "solid"),
]:
    fig.add_trace(go.Scatter(x=THETA * 100, y=y, name=name, mode="lines",
                             line={"color": color, "width": 2.5, "dash": dash}))
mle = k_obs / n_obs
fig.add_vline(x=mle * 100, line_dash="dot", line_color=MUTED,
              annotation_text=f"observed rate {mle:.1%}")
style(fig, f"Bayes' rule: prior × likelihood ∝ posterior   (data: {k_obs}/{n_obs} clicks)",
      "click-through rate (%)", "plausibility")
fig.update_xaxes(range=[0, 15])
fig.show()

prior_mean = PRIOR_A / (PRIOR_A + PRIOR_B)
post_mean  = np.trapezoid(THETA * posterior, THETA)
post_sd    = np.sqrt(np.trapezoid((THETA - post_mean) ** 2 * posterior, THETA))
prior_sd   = stats.beta(PRIOR_A, PRIOR_B).std()
print(f"prior mean {prior_mean:.3%} | observed rate {mle:.3%} | posterior mean {post_mean:.3%}")
print(f"prior sd {prior_sd:.4f}  ->  posterior sd {post_sd:.4f}")

# CLAIM 1: the posterior integrates to 1 — it is a proper distribution.
assert abs(np.trapezoid(posterior, THETA) - 1.0) < 1e-9
# CLAIM 2: the posterior mean is a compromise BETWEEN prior mean and observed rate.
assert min(prior_mean, mle) < post_mean < max(prior_mean, mle)
# CLAIM 3: the data made the belief sharper, not vaguer.
assert post_sd < prior_sd
print("ok — posterior is a proper, sharper compromise between prior and data")
"""),
    md(r"""
Read the figure like a negotiation:

- The **prior** (blue) votes for "around 5%, but I'm flexible".
- The **likelihood** (amber) votes for "the 200 impressions point at 5.5%".
- The **posterior** (berry) is the settlement — it sits *between* the prior's
  center and the observed rate (the printed numbers above show exactly where),
  and it's *narrower* than the prior, because two sources of information agree
  more than one alone.

With only 200 impressions, the prior still has real influence. What happens as
the data piles up? Drag the slider.
"""),
    md(r"""
### 🎛️ Live exploration (run me!) — watch the posterior sharpen

Drag **impressions** below (the observed CTR stays fixed at 5.5%). Watch two
things: the posterior *narrows*, and it *migrates* from the prior's center
toward the data's 5.5%. The static panel in the next cell shows four
representative slider stops in case you're reading this without a live kernel.
"""),
    code(r"""
# 🎛️ Live exploration (run me!) — slider over sample size.
from ipywidgets import interact, SelectionSlider

TRUE_CTR = 0.055

def posterior_on_grid(k, n, a=PRIOR_A, b=PRIOR_B):
    # prior x likelihood on the grid, rescaled to integrate to 1
    post = stats.beta(a, b).pdf(THETA) * stats.binom.pmf(k, n, THETA)
    return post / np.trapezoid(post, THETA)

def show_sharpening(impressions=200):
    k = round(TRUE_CTR * impressions)
    post = posterior_on_grid(k, impressions)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=THETA * 100, y=stats.beta(PRIOR_A, PRIOR_B).pdf(THETA),
                             name="prior", line={"color": SKY, "dash": "dot", "width": 2}))
    fig.add_trace(go.Scatter(x=THETA * 100, y=post, name="posterior",
                             line={"color": BERRY, "width": 2.5}))
    fig.add_vline(x=TRUE_CTR * 100, line_dash="dot", line_color=MUTED,
                  annotation_text="observed rate 5.5%")
    style(fig, f"{k} clicks / {impressions} impressions", "CTR (%)", "plausibility", height=340)
    fig.update_xaxes(range=[0, 15])
    fig.show()

interact(show_sharpening,
         impressions=SelectionSlider(options=[10, 30, 100, 200, 300, 1000, 3000, 10000],
                                     value=200, description="impressions"));

def grid_sd(k, n):
    post = posterior_on_grid(k, n)
    m = np.trapezoid(THETA * post, THETA)
    return np.sqrt(np.trapezoid((THETA - m) ** 2 * post, THETA))

# CLAIM: more impressions => sharper posterior (sd shrinks ~10x from n=10 to n=10000).
sd_small, sd_big = grid_sd(round(TRUE_CTR * 10), 10), grid_sd(round(TRUE_CTR * 10000), 10000)
print(f"posterior sd at n=10: {sd_small:.4f}   at n=10,000: {sd_big:.4f}")
assert sd_big < sd_small / 5
print("ok — the posterior sharpens as evidence accumulates")
"""),
    code(r"""
# Static companion — four representative slider stops (n = 10 / 100 / 1000 / 10000).
NS = [10, 100, 1000, 10000]
fig = make_subplots(rows=2, cols=2, subplot_titles=[
    f"n = {n:,} impressions  ({round(TRUE_CTR * n)} clicks)" for n in NS])
stats_by_n = {}
for i, n in enumerate(NS):
    r, c = i // 2 + 1, i % 2 + 1
    post = posterior_on_grid(round(TRUE_CTR * n), n)
    m = np.trapezoid(THETA * post, THETA)
    stats_by_n[n] = {"mean": m, "sd": grid_sd(round(TRUE_CTR * n), n)}
    fig.add_trace(go.Scatter(x=THETA * 100, y=stats.beta(PRIOR_A, PRIOR_B).pdf(THETA),
                             name="prior", legendgroup="p", showlegend=(i == 0),
                             line={"color": SKY, "dash": "dot", "width": 1.8}), row=r, col=c)
    fig.add_trace(go.Scatter(x=THETA * 100, y=post, name="posterior",
                             legendgroup="q", showlegend=(i == 0),
                             line={"color": BERRY, "width": 2.2}), row=r, col=c)
    fig.add_vline(x=TRUE_CTR * 100, line_dash="dot", line_color=MUTED, row=r, col=c)
fig.update_xaxes(range=[0, 15])
style(fig, "The same update at four sample sizes — the posterior migrates to the data and sharpens",
      height=560)
fig.update_xaxes(title_text="CTR (%)", row=2, col=1)
fig.update_xaxes(title_text="CTR (%)", row=2, col=2)
fig.show()

sds = [stats_by_n[n]["sd"] for n in NS]
dist_to_data = [abs(stats_by_n[n]["mean"] - TRUE_CTR) for n in NS]
print("posterior sd by n:   ", " | ".join(f"{n:>6,}: {s:.4f}" for n, s in zip(NS, sds)))
print("|mean − 5.5%| by n:  ", " | ".join(f"{n:>6,}: {d:.4f}" for n, d in zip(NS, dist_to_data)))

# CLAIM 1: uncertainty shrinks monotonically with sample size.
assert sds[0] > sds[1] > sds[2] > sds[3]
# CLAIM 2: the posterior mean ends up closer to the observed rate than it started.
assert dist_to_data[-1] < dist_to_data[0]
print("ok — monotone sharpening; the posterior converges on the data")
"""),
    # ====================================================================
    # 3. The grid trick
    # ====================================================================
    md(r"""
## 3 — The grid trick: there is no magic

Notice what we actually *did* to compute every posterior so far:

1. lay down a fine grid of candidate CTR values (2,001 points from 0 to 1),
2. at each point, multiply `prior × likelihood`,
3. rescale so the curve integrates to 1.

Three lines of numpy. No theorems, no special functions, no sampler. This is
worth dwelling on, because Bayesian inference has a reputation for being
mysterious — and for *one unknown quantity*, it's brute-forceable on a laptop
in microseconds.

It happens that for this particular model there's also a pencil-and-paper
shortcut: a Beta prior updated with click/no-click data gives a Beta posterior
with the counts just added on — $\text{Beta}(a, b) \rightarrow
\text{Beta}(a + \text{clicks},\; b + \text{non-clicks})$. Statisticians call
this **conjugacy** — *a happy algebraic coincidence where the posterior stays
in the prior's family*. It's a convenience, **not a requirement**: the grid
gives the same answer with zero algebra. Let's prove they match.
"""),
    code(r"""
# Brute-force grid posterior vs. the pencil-and-paper (conjugate) shortcut.
grid_post = posterior_on_grid(k_obs, n_obs)                       # the 3-line recipe
conj_post = stats.beta(PRIOR_A + k_obs,                           # the shortcut:
                       PRIOR_B + (n_obs - k_obs)).pdf(THETA)      # just add the counts

fig = go.Figure()
fig.add_trace(go.Scatter(x=THETA * 100, y=conj_post, mode="lines",
                         name="closed form: Beta(2+11, 38+189)",
                         line={"color": LEAF, "width": 5}, opacity=0.45))
fig.add_trace(go.Scatter(x=THETA[::40] * 100, y=grid_post[::40], mode="markers",
                         name="brute-force grid (every 40th point)",
                         marker={"color": INK, "size": 6, "symbol": "x"}))
style(fig, "Same posterior, two routes — the grid IS the math",
      "CTR (%)", "plausibility")
fig.update_xaxes(range=[0, 15])
fig.show()

max_gap = np.max(np.abs(grid_post - conj_post))
print(f"largest pointwise difference between the two curves: {max_gap:.2e}")
print(f"(for scale, the curves peak around {conj_post.max():.1f})")

# CLAIM: the brute-force grid posterior matches the conjugate closed form
# essentially exactly — Bayes' rule is multiplication and rescaling, full stop.
assert max_gap < 1e-3
assert max_gap / conj_post.max() < 1e-4
print("ok — grid ≈ closed form to numerical precision; no magic anywhere")
"""),
    md(r"""
So why doesn't everyone just use grids for everything? **The curse of
dimensionality.** One unknown needs 2,001 grid points. An MMM with 30 unknowns
(a coefficient, a carryover rate, and a saturation parameter per channel, plus
trend and seasonality) would need $2001^{30}$ points — more than the number of
atoms in the observable universe. That wall is exactly why **MCMC** exists,
and it's the entire subject of **workshop_02**. For today, one unknown and a
grid are all we need.
"""),
    # ====================================================================
    # 4. Priors matter (and stop mattering)
    # ====================================================================
    md(r"""
## 4 — Priors matter (and then stop mattering)

The prior is the part of Bayes that makes newcomers nervous: *"isn't that just
injecting opinion?"* Let's stress-test it. Three analysts look at the same
creative before launch:

- **Skeptic** (berry): "new creatives in this category flop — CTR around 2%."
- **Optimist** (leaf): "the brief was great — CTR around 10%."
- **Agnostic** (sky): a **flat prior** — *every CTR from 0% to 100% equally
  plausible* — "I refuse to assume anything."

Then all three watch the same data come in: the familiar 5.5% click rate.
"""),
    code(r"""
# Three priors, one dataset, three posteriors.
PRIORS = {  # name: (a, b, color) — Beta(a, b); a+b ≈ strength in pseudo-impressions
    "skeptic — Beta(2, 98), mean 2%":   (2.0, 98.0, BERRY),
    "agnostic — Beta(1, 1), flat":      (1.0, 1.0,  SKY),
    "optimist — Beta(10, 90), mean 10%":(10.0, 90.0, LEAF),
}
k4, n4 = 11, 200   # same evidence for everyone: 11 clicks / 200 impressions

fig = make_subplots(rows=1, cols=2, subplot_titles=(
    "BEFORE: three very different priors", "AFTER 200 impressions: three posteriors"))
summary = {}
for name, (a, b, color) in PRIORS.items():
    post = posterior_on_grid(k4, n4, a, b)
    pm = np.trapezoid(THETA * post, THETA)
    summary[name] = {"prior_mean": a / (a + b), "post_mean": pm}
    fig.add_trace(go.Scatter(x=THETA * 100, y=stats.beta(a, b).pdf(THETA), name=name,
                             legendgroup=name, line={"color": color, "width": 2.3}),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=THETA * 100, y=post, name=name, legendgroup=name,
                             showlegend=False, line={"color": color, "width": 2.3}),
                  row=1, col=2)
for c in (1, 2):
    fig.add_vline(x=5.5, line_dash="dot", line_color=MUTED, row=1, col=c)
fig.update_xaxes(range=[0, 16])
style(fig, "Same data, three analysts — the data drags every belief toward 5.5%", height=400)
fig.update_xaxes(title_text="CTR (%)", row=1, col=1)
fig.update_xaxes(title_text="CTR (%)", row=1, col=2)
fig.show()

mle4 = k4 / n4
for name, s in summary.items():
    print(f"{name:38s} prior mean {s['prior_mean']:>5.1%}  ->  posterior mean {s['post_mean']:.2%}")
print(f"{'observed rate':38s} {mle4:.2%}")

# CLAIM 1: every posterior mean lands strictly BETWEEN its prior mean and the data.
for s in summary.values():
    lo, hi = sorted([s["prior_mean"], mle4])
    assert lo < s["post_mean"] < hi
# CLAIM 2: the analysts started far apart and ended close together —
# disagreement shrinks by an order of magnitude after one batch of data.
prior_spread = max(s["prior_mean"] for s in summary.values()) - min(s["prior_mean"] for s in summary.values())
post_spread  = max(s["post_mean"]  for s in summary.values()) - min(s["post_mean"]  for s in summary.values())
print(f"spread of opinions: before {prior_spread:.1%}  ->  after {post_spread:.1%}")
assert post_spread < prior_spread / 10
print("ok — every belief moved toward the data; disagreement collapsed")
"""),
    md(r"""
### 🎛️ Live exploration (run me!) — prior strength vs. evidence

A prior has a *location* (where it's centered) **and** a *strength* (how many
"pseudo-impressions" of conviction back it). The slider below controls both
the skeptic's/optimist's strength and how much data arrives. Two things to
try:

1. Crank **prior_strength** up with little data — the posteriors stay apart.
   This is what "the priors are driving the results" looks like.
2. Now crank **impressions** — watch the three posteriors fuse into one curve
   regardless of strength. **Data overwhelms priors at a rate you can see.**
"""),
    code(r"""
# 🎛️ Live exploration (run me!) — sliders over prior strength and sample size.
from ipywidgets import IntSlider

def three_posteriors(prior_strength=100, impressions=200):
    k = round(TRUE_CTR * impressions)
    cfg = {  # rebuild the three priors at the chosen strength (flat stays flat)
        "skeptic":  (0.02 * prior_strength, 0.98 * prior_strength, BERRY),
        "agnostic": (1.0, 1.0, SKY),
        "optimist": (0.10 * prior_strength, 0.90 * prior_strength, LEAF),
    }
    fig = go.Figure()
    means = {}
    for name, (a, b, color) in cfg.items():
        post = posterior_on_grid(k, impressions, a, b)
        means[name] = np.trapezoid(THETA * post, THETA)
        fig.add_trace(go.Scatter(x=THETA * 100, y=post, name=name,
                                 line={"color": color, "width": 2.4}))
    fig.add_vline(x=TRUE_CTR * 100, line_dash="dot", line_color=MUTED)
    spread = max(means.values()) - min(means.values())
    style(fig, f"strength {prior_strength} pseudo-impressions | {k}/{impressions} observed | "
               f"posterior means disagree by {spread:.2%}",
          "CTR (%)", "plausibility", height=340)
    fig.update_xaxes(range=[0, 16])
    fig.show()
    return means

interact(three_posteriors,
         prior_strength=IntSlider(min=20, max=2000, step=20, value=100,
                                  description="prior strength"),
         impressions=SelectionSlider(options=[20, 200, 2000, 20000], value=200,
                                     description="impressions"));

def spread_at(strength, n):
    k = round(TRUE_CTR * n)
    means = [np.trapezoid(THETA * posterior_on_grid(k, n, a, b), THETA)
             for a, b in [(0.02 * strength, 0.98 * strength), (1, 1),
                          (0.10 * strength, 0.90 * strength)]]
    return max(means) - min(means)

# CLAIM 1: stronger priors + thin data => more disagreement between analysts.
assert spread_at(1000, 200) > spread_at(100, 200)
# CLAIM 2: at any fixed strength, more data => less disagreement.
assert spread_at(1000, 20000) < spread_at(1000, 200) / 5
print(f"disagreement at strength=1000: n=200 -> {spread_at(1000, 200):.2%}, "
      f"n=20,000 -> {spread_at(1000, 20000):.2%}")
print("ok — prior strength delays consensus; data forces it")
"""),
    code(r"""
# Static companion — four representative slider stops (strength fixed at 100).
NS4 = [20, 200, 2000, 20000]
fig = make_subplots(rows=2, cols=2, subplot_titles=[
    f"n = {n:,} impressions" for n in NS4])
spreads = []
for i, n in enumerate(NS4):
    r, c = i // 2 + 1, i % 2 + 1
    k = round(TRUE_CTR * n)
    means = {}
    for name, (a, b, color) in PRIORS.items():
        post = posterior_on_grid(k, n, a, b)
        means[name] = np.trapezoid(THETA * post, THETA)
        fig.add_trace(go.Scatter(x=THETA * 100, y=post, name=name.split(" —")[0],
                                 legendgroup=name, showlegend=(i == 0),
                                 line={"color": color, "width": 2.0}), row=r, col=c)
    fig.add_vline(x=TRUE_CTR * 100, line_dash="dot", line_color=MUTED, row=r, col=c)
    spreads.append(max(means.values()) - min(means.values()))
fig.update_xaxes(range=[0, 16])
style(fig, "Three analysts converge: posterior disagreement melts as data grows", height=560)
fig.update_xaxes(title_text="CTR (%)", row=2, col=1)
fig.update_xaxes(title_text="CTR (%)", row=2, col=2)
fig.show()

print("posterior-mean disagreement by n:",
      " | ".join(f"{n:>6,}: {s:.2%}" for n, s in zip(NS4, spreads)))

# CLAIM: the disagreement between the three analysts shrinks monotonically
# with sample size — priors matter exactly until the data outweighs them.
assert spreads[0] > spreads[1] > spreads[2] > spreads[3]
print("ok — monotone convergence: priors matter, then stop mattering")
"""),
    md(r"""
**The punchline.** A prior is not contamination — it's a *dial for encoding
what you already know*, with the guarantee that data turns the dial back at a
rate you can see and measure. Two consequences for MMM:

- **Priors are how domain knowledge gets in.** "ROAS is almost certainly not
  negative", "carryover decays within weeks, not years", "a lift test said
  this channel's effect is about 1.2× spend" — all of these become priors in
  workshop_01 and workshop_03.
- **Weekly MMM data is the thin-data regime.** Three years of weekly data is
  156 observations — closer to the `n=200` panel above than the `n=20,000`
  one. Priors *will* matter in your MMM, which is exactly why workshop_01
  teaches you to choose and check them deliberately, rather than pretend
  they're not there.
"""),
    # ====================================================================
    # 5. From distribution to decision
    # ====================================================================
    md(r"""
## 5 — From distribution to decision

Back to the opening question. Full data from the A/B test:

| creative | clicks | impressions | observed CTR |
|---|---|---|---|
| A (old) | 204 | 4,000 | 5.1% |
| B (new) | 216 | 4,000 | 5.4% |

A **point estimate** — *a single-number summary like the mean or the peak* —
says "B: 5.4% beats A: 5.1%" and throws away everything we just learned to
care about: the shape. The Bayesian answer keeps the shape and answers the
buyer's actual question directly: **what is the probability that B's true CTR
exceeds A's?**

The recipe is disarmingly simple. Build a posterior for each creative, then
**sample** from both — draw thousands of plausible (CTR_A, CTR_B) pairs — and
count how often B wins.
"""),
    code(r"""
# P(B beats A): build two posteriors, draw from both, count wins.
clicks_A, imps_A = 204, 4000
clicks_B, imps_B = 216, 4000

# Flat prior + data -> Beta posteriors (the conjugate shortcut from section 3).
post_A = stats.beta(1 + clicks_A, 1 + imps_A - clicks_A)
post_B = stats.beta(1 + clicks_B, 1 + imps_B - clicks_B)

N_DRAWS = 200_000
draws_A = post_A.rvs(N_DRAWS, random_state=rng)   # plausible values of A's true CTR
draws_B = post_B.rvs(N_DRAWS, random_state=rng)   # plausible values of B's true CTR
p_b_wins = float(np.mean(draws_B > draws_A))

x_pct = np.linspace(0.035, 0.075, 800)
fig = make_subplots(rows=1, cols=2, subplot_titles=(
    "the two posteriors — note the overlap", "the difference, CTR_B − CTR_A"))
fig.add_trace(go.Scatter(x=x_pct * 100, y=post_A.pdf(x_pct), name="A (old creative)",
                         line={"color": SKY, "width": 2.5}), row=1, col=1)
fig.add_trace(go.Scatter(x=x_pct * 100, y=post_B.pdf(x_pct), name="B (new creative)",
                         line={"color": BERRY, "width": 2.5}), row=1, col=1)
diff_pp = (draws_B - draws_A) * 100
fig.add_trace(go.Histogram(x=diff_pp, nbinsx=80, marker_color=AMBER,
                           name="B − A (sampled)", showlegend=False), row=1, col=2)
fig.add_vline(x=0, line_color=INK, line_width=2, row=1, col=2)
style(fig, f"Is B better? P(CTR_B > CTR_A) = {p_b_wins:.0%} — read it straight off the draws",
      height=400)
fig.update_xaxes(title_text="CTR (%)", row=1, col=1)
fig.update_xaxes(title_text="difference (percentage points)", row=1, col=2)
fig.update_yaxes(title_text="plausibility", row=1, col=1)
fig.show()

# Cross-check the sampled answer against an (almost) exact grid computation:
# P(B > A) = ∫ pdf_B(t) · cdf_A(t) dt.
p_exact = np.trapezoid(post_B.pdf(THETA) * post_A.cdf(THETA), THETA)
print(f"P(B beats A): sampled {p_b_wins:.4f} vs grid-exact {p_exact:.4f}")

# CLAIM 1: sampling reproduces the exact answer to ~3 decimals.
assert abs(p_b_wins - p_exact) < 0.01
# CLAIM 2: the verdict is genuinely uncertain — B is favored, but far from sure.
assert 0.55 < p_b_wins < 0.90
print("ok — 'B is probably better, but it's not a lock' — now quantified")
"""),
    md(r"""
So B is *probably* better — the printed number above is your answer to "how
sure are we", and it's a far more honest deliverable than either "B won" or
"the difference was not statistically significant".

To report the *size* of an effect with its uncertainty, Bayesians use a
**credible interval** — *a range that contains the true value with a stated
probability, given your model and data*. The version we'll use throughout the
series is the **HDI (highest-density interval)** — *the narrowest such range:
every value inside it is more plausible than every value outside it*. A "90%
HDI of 5.0%–6.1%" means exactly what it sounds like: **"90% chance the true
CTR is between 5.0% and 6.1%."** (You'll meet HDIs again on every MMM
parameter in workshop_04.)
"""),
    code(r"""
# The 90% HDI of creative B's CTR, computed from the same draws.
def hdi_from_draws(draws, prob=0.90):
    # narrowest window containing `prob` of the sorted draws
    s = np.sort(draws)
    k = int(np.ceil(prob * len(s)))
    widths = s[k - 1:] - s[: len(s) - k + 1]
    i = int(np.argmin(widths))
    return s[i], s[i + k - 1]

hdi_lo, hdi_hi = hdi_from_draws(draws_B, 0.90)

mask = (x_pct >= hdi_lo) & (x_pct <= hdi_hi)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_pct * 100, y=post_B.pdf(x_pct), name="posterior of B's CTR",
                         line={"color": BERRY, "width": 2.5}))
fig.add_trace(go.Scatter(x=x_pct[mask] * 100, y=post_B.pdf(x_pct[mask]),
                         fill="tozeroy", mode="none", fillcolor="rgba(166,58,80,0.25)",
                         name=f"90% HDI: {hdi_lo:.2%} – {hdi_hi:.2%}"))
fig.add_vline(x=post_B.mean() * 100, line_dash="dot", line_color=INK,
              annotation_text="posterior mean")
style(fig, "Creative B: the 90% HDI — every value inside is more plausible than any outside",
      "CTR (%)", "plausibility")
fig.show()

mass_in_hdi = post_B.cdf(hdi_hi) - post_B.cdf(hdi_lo)
et_lo, et_hi = post_B.ppf(0.05), post_B.ppf(0.95)   # the "equal-tailed" alternative
print(f"90% HDI: [{hdi_lo:.2%}, {hdi_hi:.2%}]  width {(hdi_hi - hdi_lo):.2%}  "
      f"true mass inside: {mass_in_hdi:.1%}")
print(f"equal-tailed 90%: [{et_lo:.2%}, {et_hi:.2%}]  width {(et_hi - et_lo):.2%}")

# CLAIM 1: the sample-based HDI really does contain ~90% of the posterior mass.
assert 0.88 < mass_in_hdi < 0.92
# CLAIM 2: the HDI is the NARROWEST 90% interval (no wider than equal-tailed).
assert (hdi_hi - hdi_lo) <= (et_hi - et_lo) + 1e-4
print("ok — the HDI holds 90% of the belief in the tightest possible range")
"""),
    md(r"""
**A gentle word about p-values.** If you've run classical A/B tests, you'd
test "B = A" and get a p-value — the probability of data this extreme *if the
creatives were truly identical*. That's a useful but indirect quantity, and
it's famously easy to misread as "the probability B is better", which it is
not. The Bayesian quantities you just computed are the direct versions: P(B
beats A) is literally the probability B is better, and the HDI is literally a
range you believe contains the truth with stated probability. Neither
framework is "wrong" — but when the deliverable is a *decision under
uncertainty*, it helps that the numbers answer the decision-maker's question
in her own words.
"""),
    # ====================================================================
    # 6. Monte Carlo as a superpower
    # ====================================================================
    md(r"""
## 6 — Monte Carlo as a superpower: from CTRs to dollars

Here's the trick that powers the entire back half of this series. The buyer
doesn't actually care about CTR — she cares about **revenue**. Suppose the
campaign serves **2,000,000 impressions a month** and each click is worth
**$0.50** in expected revenue. The monthly revenue uplift from switching to B
is:

$$\text{uplift} = (\text{CTR}_B - \text{CTR}_A) \times 2{,}000{,}000 \times \$0.50$$

This is a **derived quantity** — *a number computed from unknowns rather than
estimated directly*. And here is the superpower, called **Monte Carlo** —
*answering questions about distributions by drawing many random samples and
just counting*: **push every posterior draw through the formula, and you get
the full distribution of the derived quantity, for free.** No new model, no
calculus — arithmetic on the draws you already have.
"""),
    code(r"""
# Push every (CTR_A, CTR_B) draw through the revenue formula.
IMPRESSIONS_PER_MONTH = 2_000_000
REVENUE_PER_CLICK = 0.50

uplift = (draws_B - draws_A) * IMPRESSIONS_PER_MONTH * REVENUE_PER_CLICK  # $ / month

q05, q50, q95 = np.percentile(uplift, [5, 50, 95])
p_positive = float(np.mean(uplift > 0))

counts, edges = np.histogram(uplift, bins=90)
centers = (edges[:-1] + edges[1:]) / 2
fig = go.Figure()
fig.add_trace(go.Bar(x=centers[centers <= 0], y=counts[centers <= 0],
                     marker_color=BERRY, name="switch loses money"))
fig.add_trace(go.Bar(x=centers[centers > 0], y=counts[centers > 0],
                     marker_color=LEAF, name="switch makes money"))
fig.add_vline(x=0, line_color=INK, line_width=2)
fig.add_vline(x=q50, line_dash="dot", line_color=INK,
              annotation_text=f"median ${q50:,.0f}")
style(fig, f"Monthly revenue uplift from switching to B — P(uplift > 0) = {p_positive:.0%}",
      "monthly uplift ($)", "number of draws")
fig.update_layout(bargap=0, barmode="overlay")
fig.show()

print(f"median uplift ${q50:,.0f}/month | 90% interval ${q05:,.0f} to ${q95:,.0f} "
      f"| P(uplift > 0) = {p_positive:.1%}")

# CLAIM 1: a positive-dollar uplift is the SAME event as B beating A — pushing
# draws through a monotone formula preserves every probability statement.
assert p_positive == p_b_wins
# CLAIM 2: the 90% interval straddles zero — the switch is favored but NOT a
# sure thing, and the distribution (not the median) is what says so.
assert q05 < 0 < q95 and q50 > 0
print("ok — same draws, now denominated in dollars; uncertainty came along for the ride")
"""),
    md(r"""
Look at what just happened: the question changed from statistics ("is B's rate
higher?") to business ("how many dollars, with what risk?") and the *method*
didn't change at all — same draws, one line of arithmetic. A point estimate
would have said "switch, you'll make the median amount". The distribution
shows the red tail: a real chance the switch loses money. Whether that risk is
acceptable depends on what the switch *costs* — which is a slider, not a fact.

This per-draw arithmetic is **exactly** how `mmm_framework` turns posterior
parameter draws into ROAS distributions, budget-shift scenarios, and
"probability this channel clears its hurdle rate" — the entire subject of
**workshop_05**.

### 🎛️ Live exploration (run me!) — does the upside clear the switching cost?

Switching creatives isn't free (production, trafficking, QA). Drag the
**cost** slider and watch the decision quantity — P(uplift beats the cost this
month) — move. The static curve in the next cell shows the whole picture at
once.
"""),
    code(r"""
# 🎛️ Live exploration (run me!) — slider over one-time switching cost.
def switch_decision(cost=2000):
    p_beats = float(np.mean(uplift > cost))
    net = uplift - cost
    counts, edges = np.histogram(net, bins=90)
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure()
    fig.add_trace(go.Bar(x=centers[centers <= 0], y=counts[centers <= 0],
                         marker_color=BERRY, name="net loss"))
    fig.add_trace(go.Bar(x=centers[centers > 0], y=counts[centers > 0],
                         marker_color=LEAF, name="net gain"))
    fig.add_vline(x=0, line_color=INK, line_width=2)
    style(fig, f"cost ${cost:,} -> P(first-month net gain) = {p_beats:.0%}, "
               f"expected net ${net.mean():,.0f}",
          "first-month net ($ uplift − cost)", "number of draws", height=340)
    fig.update_layout(bargap=0, barmode="overlay")
    fig.show()

interact(switch_decision,
         cost=IntSlider(min=0, max=12000, step=500, value=2000, description="cost ($)"));

# CLAIM: raising the bar can only lower the probability of clearing it,
# and at zero cost the decision reduces to "is B better at all?".
p_at = lambda c: float(np.mean(uplift > c))
assert p_at(0) == p_b_wins
assert p_at(0) >= p_at(2000) >= p_at(6000) >= p_at(12000)
print(f"P(beats cost): $0 -> {p_at(0):.0%} | $2k -> {p_at(2000):.0%} | "
      f"$6k -> {p_at(6000):.0%} | $12k -> {p_at(12000):.0%}")
print("ok — one distribution answers every version of the decision question")
"""),
    code(r"""
# Static companion — the full decision curve, with four representative slider stops.
costs = np.arange(0, 12001, 250)
p_curve = np.array([np.mean(uplift > c) for c in costs])
marks = [0, 2000, 6000, 12000]

fig = go.Figure()
fig.add_trace(go.Scatter(x=costs, y=p_curve * 100, mode="lines",
                         name="P(uplift beats cost)", line={"color": SKY, "width": 3}))
fig.add_trace(go.Scatter(x=marks, y=[np.mean(uplift > c) * 100 for c in marks],
                         mode="markers+text", name="slider stops",
                         text=[f"${c:,}" for c in marks], textposition="top right",
                         marker={"color": AMBER, "size": 11, "line": {"color": INK, "width": 1}}))
fig.add_hline(y=50, line_dash="dot", line_color=MUTED,
              annotation_text="coin flip (50%)")
style(fig, "The decision curve: probability the switch pays for itself in month one",
      "one-time switching cost ($)", "P(net gain) (%)")
fig.update_yaxes(range=[0, 100])
fig.show()

even_money = float(costs[np.searchsorted(-p_curve, -0.5)])
print(f"the switch is better-than-even-money up to a cost of about ${even_money:,.0f}")

# CLAIM: the decision curve is monotone nonincreasing in cost, starts at
# P(B beats A), and crosses 50% somewhere inside the plotted range.
assert np.all(np.diff(p_curve) <= 1e-12)
assert p_curve[0] == p_b_wins
assert p_curve[0] > 0.5 > p_curve[-1]
print("ok — every budget question is a vertical slice of one posterior")
"""),
    # ====================================================================
    # 7. Glossary + what's next
    # ====================================================================
    md(r"""
## 7 — What you can now say (glossary + what's next)

You started with "5.4% vs 5.1% — is it better?" and ended with a probability
that B wins, a credible range for its CTR, a dollar-denominated uplift
distribution, and a cost threshold below which switching is better than even
money — all from two Beta curves and counting. The terms you picked up:

| term | plain-English meaning |
|---|---|
| **probability distribution** | a curve over all possible values of an unknown; height = plausibility |
| **prior** | your belief about an unknown *before* the new data, written as a distribution |
| **likelihood** | for each candidate value, how probable the observed data would be |
| **posterior** | your updated belief: prior × likelihood, rescaled — the output of Bayes' rule |
| **conjugacy** | a happy algebraic coincidence where the posterior stays in the prior's family (a shortcut, never a requirement) |
| **point estimate** | a single-number summary (mean, peak) — convenient, and it throws away the shape |
| **credible interval** | a range containing the true value with stated probability, given model + data |
| **HDI** (highest-density interval) | the *narrowest* credible interval: everything inside is more plausible than anything outside |
| **Monte Carlo** | answering questions about distributions by drawing many samples and counting |
| **derived quantity** | a number computed *from* unknowns (revenue uplift, ROAS) — push the draws through the formula and it inherits a full distribution |

And the three ideas to carry forward:

1. **Uncertainty is the product.** The shape of the posterior — not its peak —
   is what a decision needs.
2. **Priors are knowledge, with a receipt.** They encode what you know, and
   you watched data overwhelm them at a visible, measurable rate.
3. **Draws are a universal currency.** Any business quantity you can compute
   from the unknowns, you can compute *per draw* — and get its uncertainty for
   free.

**Next — [workshop_01_priors.ipynb](workshop_01_priors.ipynb):** today the
priors were handed to you. Next time you learn to *choose* them: what a
sensible prior for an adstock rate or a ROAS looks like, how to sanity-check a
prior by simulating data from it before fitting (**prior predictive checks**),
and the prior families `mmm_framework` uses for every part of an MMM.
"""),
]


def main():
    nb = new_notebook(cells=CELLS)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "title": "Workshop 00 — Thinking in Distributions",
    })
    path = "workshop_00_thinking_in_distributions.ipynb"
    with open(path, "w") as fh:
        nbformat.write(nb, fh)
    print(f"wrote {path} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
