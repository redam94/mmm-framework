"""Author workshop notebook 01 — Priors: Saying What You Know (run from ``nbs/``).

    uv run python builders/build_workshop_01_priors.py
    PYTHONPATH=.. uv run jupyter nbconvert --to notebook --execute --inplace \
        workshop/workshop_01_priors.ipynb --ExecutePreprocessor.timeout=2400

Notebook 01 of the 6-part *workshop* series: the Bayesian causal workflow for
marketing analysts with no Bayesian background. This one teaches priors —
choosing them, the MMM distribution zoo (HalfNormal / Beta / Gamma / Normal),
how parameter priors imply priors on *behavior* (adstock curves, response
curves), the prior predictive check, how ``mmm_framework`` configures priors
(``PriorConfig`` / builders / ``roi_prior``), and prior sensitivity.

Authored as md/code cells via nbformat (pattern: ``build_stress_00_rosy_picture.py``).
Main teaching charts are plotly (renderer ``notebook_connected``); three
``ipywidgets.interact`` live-exploration cells, each paired with a static
multi-setting figure. Every computational cell ends with asserts encoding the
claim it plotted; randomness is seeded; the framework prior-predictive cell
asserts robust invariants only (its ``samples=`` API ignores ``random_seed``).
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
# Workshop 01 — Priors: Saying What You Know

Before you've seen a single week of data, you already know things. TV's effect
on sales isn't negative. Carryover from a radio ad doesn't last two years. The
return on a search campaign isn't 500-to-1. None of that came from this year's
spreadsheet — it came from how marketing works.

A **prior** (your model's starting beliefs about a parameter, written as a
probability distribution — the idea you met in
[workshop_00](workshop_00_thinking_in_distributions.ipynb)) is how you write
that knowledge down so the model can use it. And here is the part that
surprises people: *refusing to choose a prior is itself a choice* — usually the
choice that ROAS of −80 and ROAS of +500 are exactly as believable as ROAS
of 2. That's not neutrality. That's a strong, silly opinion wearing a
"no opinion" costume.

This notebook is about making the opinion explicit, reasonable, and *checkable*:

1. **The prior vocabulary** — flat vs weakly informative vs informative, in
   plain English.
2. **The distribution zoo for MMM** — which shape fits which job (channel
   effects, carryover, saturation, controls), with live sliders.
3. **Priors on parameters are priors on behavior** — the big idea: a prior on
   a decay rate is really a prior on *carryover curves*.
4. **The prior predictive check** — the cheapest, highest-value check in the
   whole workflow: simulate fake sales from your priors *before* fitting.
5. **How `mmm_framework` says it** — the real `PriorConfig` / builder API.
6. **Sensitivity in one picture** — when priors matter and when they wash out.

> Series: [workshop_00](workshop_00_thinking_in_distributions.ipynb) ·
> **workshop_01 (this one)** ·
> [workshop_02](workshop_02_sampling.ipynb) (how the machine actually computes
> posteriors) · [workshop_03](workshop_03_first_mmm.ipynb) (your first real
> MMM fit) · workshop_04 (reading the posterior) · workshop_05 (decisions).
"""),
    code(r"""
import sys, pathlib, warnings, logging
import numpy as np, pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
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
    "Hide library chatter (sampler banners etc.); our own prints stay visible."
    with open(os.devnull, "w") as _dn, contextlib.redirect_stdout(_dn), \
            contextlib.redirect_stderr(_dn):
        yield

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = "notebook_connected"

INK, SKY, BERRY, LEAF, AMBER = "#2b2118", "#3b6ea5", "#a63a50", "#3f7d5e", "#d98a2b"
MUTED = "#8a8079"

def style(fig, title, xt="", yt="", h=420, w=840):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK)),
        xaxis_title=xt, yaxis_title=yt, width=w, height=h,
        template="plotly_white", font=dict(color=INK, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        margin=dict(t=70, b=50, l=60, r=20),
    )
    return fig

rng = np.random.default_rng(42)
print("plotly renderer:", pio.renderers.default)
assert pio.renderers.default == "notebook_connected"
assert all(c.startswith("#") and len(c) == 7 for c in (INK, SKY, BERRY, LEAF, AMBER))
"""),
    # ====================================================================
    # 1. The prior vocabulary
    # ====================================================================
    md(r"""
## 1 — The prior vocabulary: three ways to "know" something

Take one concrete parameter: **the effect of TV** — incremental sales dollars
generated per $1 of TV spend (you may know this as ROAS). Three analysts write
down three priors:

- A **flat (uninformative) prior** — a distribution that spreads belief evenly
  over a huge range, trying to say "I know nothing." Here: *"any value between
  −10 and +10 is equally likely."* Sounds humble. But look at what it actually
  asserts: TV **destroying** $8 of sales per dollar is exactly as plausible as
  TV earning $2 — and half of all belief sits on TV being harmful.
- A **weakly informative prior** — a distribution that rules out the absurd
  while staying loose about everything plausible. Here: *"the effect is not
  negative, probably modest, and values beyond ~6 would be extraordinary."*
  This is the workhorse of practical Bayesian modeling.
- An **informative prior** — a distribution that commits to real knowledge,
  e.g. from last year's model or industry benchmarks. Here: *"around 0.8,
  give or take 0.3, and never negative."*

Run the cell to see all three on one axis.
"""),
    code(r"""
# Three priors for "incremental sales per $1 of TV spend".
flat = stats.uniform(loc=-10, scale=20)                  # flat on [-10, 10]
weak = stats.halfnorm(scale=2.0)                         # HalfNormal(sigma=2)
info = stats.truncnorm(a=(0 - 0.8) / 0.3, b=np.inf, loc=0.8, scale=0.3)

x = np.linspace(-10, 10, 2001)
fig = go.Figure()
for dist, name, color in [
    (flat, "flat: Uniform(-10, 10) — 'no opinion'", MUTED),
    (weak, "weakly informative: HalfNormal(σ=2)", SKY),
    (info, "informative: TruncNormal(0.8, 0.3), ≥ 0", BERRY),
]:
    fig.add_trace(go.Scatter(x=x, y=dist.pdf(x), name=name,
                             line=dict(color=color, width=2.5), fill="tozeroy",
                             opacity=0.85))
fig.add_vline(x=0, line=dict(color=INK, width=1, dash="dot"))
fig.add_annotation(x=-5, y=float(flat.pdf(0)) * 3, text="half the flat prior's<br>belief: TV HURTS sales",
                   showarrow=True, arrowhead=2, ax=0, ay=-60, font=dict(color=MUTED))
style(fig, "Three priors for the same parameter: TV effect ($ sales per $1 spend)",
      "effect of TV ($ per $)", "prior density").show()

p_flat_negative = flat.cdf(0)
p_weak_extreme = weak.sf(6)            # P(effect > 6) under the weak prior
p_info_sane = info.cdf(1.4) - info.cdf(0.2)
print(f"flat prior:  P(TV effect < 0)      = {p_flat_negative:.0%}")
print(f"weak prior:  P(TV effect > 6)      = {p_weak_extreme:.2%}")
print(f"info prior:  P(0.2 < effect < 1.4) = {p_info_sane:.0%}")

# CLAIM: the 'no opinion' prior is a strong opinion — half its mass says TV
# hurts sales; the weak prior keeps extreme effects rare; the informative
# prior concentrates belief in the benchmark range.
assert abs(p_flat_negative - 0.5) < 1e-12
assert p_weak_extreme < 0.01
assert p_info_sane > 0.90
"""),
    md(r"""
Read the three curves as three *sentences*:

| prior | what it literally says | when to use it |
|---|---|---|
| flat | "−8 is as plausible as +2; harm is a coin flip" | almost never — it isn't neutral, it's reckless |
| weakly informative | "not negative, probably small, extremes need extraordinary evidence" | your default for MMM effect parameters |
| informative | "around 0.8 ± 0.3 — I have benchmarks/lift tests and I'm using them" | when you genuinely have the evidence (and can defend it in a meeting) |

One more vocabulary item before the zoo: the numbers that *define* a prior —
the σ=2 in HalfNormal(σ=2), the (0.8, 0.3) in the informative one — are called
**hyperparameters** (the knobs you set on a prior distribution, as opposed to
the parameters the model learns from data). Choosing a prior = choosing a
shape from the zoo + setting its hyperparameters.
"""),
    # ====================================================================
    # 2. The zoo — HalfNormal / TruncatedNormal
    # ====================================================================
    md(r"""
## 2 — The distribution zoo for MMM

An MMM has four recurring kinds of parameter, and each has a distribution
family whose *shape matches the job*. This section is a field guide.

### 2.1 HalfNormal & TruncatedNormal → channel effects

A media channel's effect on sales **can't be negative** (advertising may be
useless, but it isn't anti-persuasive — and if you truly believe a channel
hurts sales, you want a different model, not a negative coefficient) and is
**probably small**. So we want a shape whose **support** (the set of values a
distribution allows at all) starts at zero, puts plenty of belief near zero,
and fades smoothly into a tail:

- **HalfNormal(σ)** — a bell curve folded at zero. One knob: σ, "how big could
  this effect plausibly be?" Doubling σ doubles every quantile — it stretches
  the whole belief, it doesn't move it.
- **TruncatedNormal(μ, σ, lower=0)** — a bell centered at your benchmark μ,
  chopped at zero. Use it when you have a *location* in mind, not just a scale.
"""),
    code(r"""
xe = np.linspace(0, 7, 1400)
fig = go.Figure()
for sigma, color in [(0.5, LEAF), (1.0, SKY), (2.0, AMBER)]:
    fig.add_trace(go.Scatter(x=xe, y=stats.halfnorm(scale=sigma).pdf(xe),
                             name=f"HalfNormal(σ={sigma})",
                             line=dict(color=color, width=2.5)))
tn = stats.truncnorm(a=(0 - 1.0) / 0.5, b=np.inf, loc=1.0, scale=0.5)
fig.add_trace(go.Scatter(x=xe, y=tn.pdf(xe), name="TruncNormal(μ=1, σ=0.5, ≥0)",
                         line=dict(color=BERRY, width=2.5, dash="dash")))
style(fig, "Channel-effect priors: zero-or-positive, probably small",
      "channel effect (sales per unit of saturated media pressure)",
      "prior density").show()

q90 = {s: stats.halfnorm(scale=s).ppf(0.90) for s in (0.5, 1.0, 2.0)}
print("90th percentile of HalfNormal(σ):",
      {s: round(q, 2) for s, q in q90.items()})

# CLAIM 1: support starts at zero — no belief in negative effects.
assert stats.halfnorm(scale=2.0).cdf(0) == 0.0 and tn.cdf(0) < 1e-12
# CLAIM 2: σ is a pure stretch — quantiles scale linearly with it.
assert np.isclose(q90[2.0], 4 * q90[0.5]) and np.isclose(q90[1.0], 2 * q90[0.5])
# CLAIM 3: the truncated normal concentrates around its benchmark μ=1.
assert tn.cdf(2.0) - tn.cdf(0.3) > 0.85
"""),
    # ====================================================================
    # 2.2 Beta — adstock retention + half-life
    # ====================================================================
    md(r"""
### 2.2 Beta → adstock retention (a fraction in [0, 1])

**Adstock** is MMM's word for *carryover* — an ad you run this week still
nudges sales next week, a bit less the week after, and so on. The simplest
("geometric") version has one parameter: the **retention rate** α — the
fraction of this week's advertising pressure that survives into next week.
After one week the impulse is worth $\alpha$, after two weeks $\alpha^2$,
after $k$ weeks $\alpha^k$.

A retention rate is a **fraction** — it lives strictly between 0 and 1 — so
its natural prior family is the **Beta distribution** (the flexible
distribution *on* [0, 1] you met with conversion rates in workshop_00). Its
two hyperparameters (α, β — confusingly also called a, b; not the same α as
the retention rate!) shape where the belief sits: Beta(2, 2) is a gentle hill
centered at ½; Beta(8, 2) piles belief up near 0.8; Beta(1, 3) leans toward
fast decay.

Here is the move that makes priors *talkable in a marketing meeting*: convert
retention into **half-life** — the number of weeks until the carryover effect
has decayed to half its original strength. Setting $\alpha^h = \tfrac12$ and
solving gives

$$h(\alpha) = \frac{\ln(0.5)}{\ln(\alpha)}.$$

A retention of 0.5 means a half-life of exactly 1 week (analytically); higher
retention stretches it fast. So a Beta prior on retention **is** a prior on
half-lives — and you can ask the brand team "does a 3-week TV half-life sound
right?" instead of "is Beta(8, 2) okay?".
"""),
    code(r"""
SEED_BETA = 3
r = np.random.default_rng(SEED_BETA)
beta_priors = [("Beta(1, 3) — fast decay", 1, 3, LEAF),
               ("Beta(2, 2) — open-minded", 2, 2, SKY),
               ("Beta(8, 2) — long memory", 8, 2, BERRY)]

fig = make_subplots(rows=1, cols=2, subplot_titles=(
    "prior on retention rate α", "implied prior on carryover HALF-LIFE (weeks)"))
xa = np.linspace(0.001, 0.999, 999)
halflife_med = {}
for name, a, b, color in beta_priors:
    fig.add_trace(go.Scatter(x=xa, y=stats.beta(a, b).pdf(xa), name=name,
                             line=dict(color=color, width=2.5), legendgroup=name),
                  row=1, col=1)
    draws = r.beta(a, b, 20000)
    hl = np.log(0.5) / np.log(draws)
    halflife_med[(a, b)] = float(np.median(hl))
    fig.add_trace(go.Histogram(x=hl[hl < 12], nbinsx=60, name=name,
                               marker_color=color, opacity=0.55,
                               legendgroup=name, showlegend=False),
                  row=1, col=2)
fig.update_xaxes(title_text="retention rate α", row=1, col=1)
fig.update_xaxes(title_text="half-life (weeks)", row=1, col=2)
fig.update_layout(barmode="overlay")
style(fig, "A Beta prior on retention IS a prior on half-life", h=420, w=940).show()

print("median implied half-life (weeks):",
      {k: round(v, 2) for k, v in halflife_med.items()})

# CLAIM 1: retention α = 0.5 ⇒ half-life exactly 1 week (h = ln.5/ln.5).
assert np.isclose(np.log(0.5) / np.log(0.5), 1.0)
# CLAIM 2: Beta draws live strictly inside (0, 1) — a fraction's support.
assert (draws > 0).all() and (draws < 1).all()
# CLAIM 3: the long-memory prior implies materially longer half-lives.
assert halflife_med[(8, 2)] > 2 * halflife_med[(2, 2)] > halflife_med[(1, 3)]
# CLAIM 4 (bonus identity): total adstock weight over an infinite horizon is
# the geometric series sum 1/(1-α) — carryover multiplies media pressure.
alpha0 = 0.7
assert np.isclose((alpha0 ** np.arange(2000)).sum(), 1 / (1 - alpha0))
"""),
    code(r"""
# 🎛️ Live exploration (run me!) — drag a and b; watch retention AND half-life.
# (Static partner: the 3-prior panel above shows Beta(1,3) / Beta(2,2) / Beta(8,2).)
from ipywidgets import interact, FloatSlider

def beta_halflife_stats(a, b, n=20000, seed=SEED_BETA):
    "Pure compute: retention pdf grid + implied half-life draws + median."
    rr = np.random.default_rng(seed)
    draws = rr.beta(a, b, n)
    hl = np.log(0.5) / np.log(draws)
    return draws, hl, float(np.median(hl))

def show_beta_prior(a=2.0, b=2.0):
    draws, hl, med = beta_halflife_stats(a, b)
    f = make_subplots(rows=1, cols=2, subplot_titles=(
        f"Beta({a:g}, {b:g}) on retention α",
        f"implied half-life — median {med:.1f} weeks"))
    f.add_trace(go.Scatter(x=xa, y=stats.beta(a, b).pdf(xa),
                           line=dict(color=SKY, width=2.5), showlegend=False),
                row=1, col=1)
    f.add_trace(go.Histogram(x=hl[hl < 15], nbinsx=60, marker_color=AMBER,
                             showlegend=False), row=1, col=2)
    f.update_xaxes(title_text="retention rate α", row=1, col=1)
    f.update_xaxes(title_text="half-life (weeks)", row=1, col=2)
    style(f, "Your prior on carryover, in marketing units", h=380, w=940).show()

interact(show_beta_prior,
         a=FloatSlider(value=2.0, min=0.5, max=12.0, step=0.5),
         b=FloatSlider(value=2.0, min=0.5, max=12.0, step=0.5));

# CLAIM: the compute behind the widget is sane across representative settings —
# half-life medians rise as the prior shifts toward high retention.
meds = [beta_halflife_stats(a, b)[2] for a, b in [(1, 3), (2, 2), (8, 2), (12, 2)]]
assert meds == sorted(meds) and meds[0] < 1.0 < meds[-1]
"""),
    # ====================================================================
    # 2.3 Gamma / saturation rate
    # ====================================================================
    md(r"""
### 2.3 Gamma & friends → the saturation rate λ

**Saturation** is the other workhorse transform: doubling spend doesn't double
sales — audiences run out, auctions get pricier. `mmm_framework`'s core model
uses the logistic curve

$$f(x) = 1 - e^{-\lambda x},$$

where $x$ is media pressure scaled so the biggest observed week is 1, and the
**saturation rate** λ controls how fast returns diminish. λ must be positive
and could be anywhere from "barely saturating" to "saturates immediately" — a
job for a positive, right-skewed family like **Gamma(α, β)** (a flexible
distribution over positive numbers; mean α/β) or the one-knob **Exponential**
(the framework's built-in default for this parameter, with mean 2).

Like retention → half-life, λ has a marketing translation: the
**half-saturation point** — the spend level at which you've already banked
half of the channel's maximum possible effect. Setting $f(x_{1/2}) = \tfrac12$
gives

$$x_{1/2} = \frac{\ln 2}{\lambda}.$$

Multiply by the channel's max weekly spend and your prior on λ becomes a
statement like *"we hit half of TV's ceiling somewhere around $X thousand a
week"* — something a media planner can actually argue with. Watch the printout
below: a chunk of the default prior's belief puts the half-saturation point
**beyond the biggest week you've ever bought** — i.e. "we never even got
halfway to saturating this channel." Decide whether you believe that.
"""),
    code(r"""
SEED_LAM = 5
MAX_WEEKLY_SPEND_K = 50  # this channel's biggest observed week: $50k
lam_priors = [("Exponential(mean 2) — framework default", stats.expon(scale=2.0), SKY),
              ("Gamma(2, 1) — mean 2, thinner near 0", stats.gamma(a=2.0, scale=1.0), LEAF),
              ("Gamma(6, 2) — mean 3, confident", stats.gamma(a=6.0, scale=0.5), BERRY)]

fig = make_subplots(rows=1, cols=2, subplot_titles=(
    "prior on saturation rate λ",
    f"implied HALF-SATURATION spend ($k, max week = {MAX_WEEKLY_SPEND_K})"))
xl = np.linspace(0.001, 8, 1200)
beyond_max = {}
for name, dist, color in lam_priors:
    fig.add_trace(go.Scatter(x=xl, y=dist.pdf(xl), name=name,
                             line=dict(color=color, width=2.5), legendgroup=name),
                  row=1, col=1)
    lam_draws = dist.rvs(20000, random_state=np.random.default_rng(SEED_LAM))
    xhalf = np.log(2) / lam_draws                  # in 'share of max week' units
    beyond_max[name] = float((xhalf > 1).mean())   # half-sat beyond max spend
    spend_k = xhalf * MAX_WEEKLY_SPEND_K
    fig.add_trace(go.Histogram(x=spend_k[spend_k < 3 * MAX_WEEKLY_SPEND_K],
                               nbinsx=60, marker_color=color, opacity=0.55,
                               legendgroup=name, showlegend=False), row=1, col=2)
fig.add_vline(x=MAX_WEEKLY_SPEND_K, line=dict(color=INK, dash="dot"), row=1, col=2)
fig.update_xaxes(title_text="λ", row=1, col=1)
fig.update_xaxes(title_text="half-saturation spend ($k/week)", row=1, col=2)
fig.update_layout(barmode="overlay")
style(fig, "A prior on λ IS a prior on the half-saturation spend ln2/λ",
      h=420, w=940).show()

print("P(half-saturation point beyond the biggest week ever bought):")
for name, p in beyond_max.items():
    print(f"  {name}: {p:.0%}")

# CLAIM 1: the half-saturation identity — f(ln2/λ) = 0.5 EXACTLY, every draw.
f_at_xhalf = 1 - np.exp(-lam_draws * (np.log(2) / lam_draws))
assert np.allclose(f_at_xhalf, 0.5)
# CLAIM 2: λ draws are strictly positive (a rate's support).
assert (lam_draws > 0).all()
# CLAIM 3: the default Exponential prior puts a non-trivial share of belief on
# 'this channel never reached half-saturation in the data' (x_half > max week).
assert 0.15 < beyond_max["Exponential(mean 2) — framework default"] < 0.45
# CLAIM 4: the confident Gamma(6, 2) prior mostly keeps x_half inside the data.
assert beyond_max["Gamma(6, 2) — mean 3, confident"] < 0.10
"""),
    # ====================================================================
    # 2.4 Normal + zoo summary
    # ====================================================================
    md(r"""
### 2.4 Normal → control & seasonal coefficients

Controls are different: a price increase, a competitor launch, a heat wave can
push sales *either way*. For sign-free coefficients the plain **Normal(μ, σ)**
is right: centered at zero ("no effect until the data says otherwise"),
symmetric (a positive effect is as believable as a negative one of the same
size), with σ setting how big an effect you'd entertain. The framework's
default for controls is Normal(0, 1) on standardized data — the same idea
powers seasonal (Fourier) coefficients.

### The zoo, summarized

| parameter | lives in | family | marketing translation of the prior |
|---|---|---|---|
| channel effect β | [0, ∞) | HalfNormal / TruncatedNormal / Gamma | "not harmful, probably modest" |
| adstock retention α | (0, 1) | Beta | a **half-life** in weeks |
| saturation rate λ | (0, ∞) | Gamma / Exponential | a **half-saturation spend** in $ |
| control / seasonal coef | (−∞, ∞) | Normal | "either direction, probably small" |
"""),
    code(r"""
xn = np.linspace(-4, 4, 1201)
fig = go.Figure()
for sigma, color in [(0.5, LEAF), (1.0, SKY)]:
    fig.add_trace(go.Scatter(x=xn, y=stats.norm(0, sigma).pdf(xn),
                             name=f"Normal(0, {sigma}) — sign-free control prior",
                             line=dict(color=color, width=2.5)))
fig.add_vline(x=0, line=dict(color=INK, width=1, dash="dot"))
style(fig, "Control-coefficient priors: centered at 'no effect', open to either sign",
      "control coefficient (standardized)", "prior density").show()

# CLAIM: the Normal control prior is exactly even-handed about sign.
assert np.isclose(stats.norm(0, 1).cdf(0), 0.5)
assert np.isclose(stats.norm(0, 0.5).ppf(0.9), -stats.norm(0, 0.5).ppf(0.1))
"""),
    # ====================================================================
    # 3. Priors on parameters imply priors on behavior
    # ====================================================================
    md(r"""
## 3 — Priors on parameters are priors on BEHAVIOR

Here is the single most useful mental shift in this notebook. You never
actually care about α or λ. You care about *curves*: how carryover decays week
by week, how sales respond as spend ramps. Every draw from the prior on α is a
whole **decay curve**; every draw from the prior on λ is a whole **response
curve**. So a prior on a parameter is really a prior over *families of
behavior* — and the honest way to inspect a prior is to draw a hundred
parameter values and plot the hundred curves they imply (a "spaghetti plot").

If the spaghetti contains curves your gut rejects — carryover lasting half a
year for a flash-sale channel, response curves still climbing steeply at twice
your maximum budget — your prior said that, whether you meant to or not.
"""),
    code(r"""
SEED_FAN = 9
rf = np.random.default_rng(SEED_FAN)
N_CURVES = 100
weeks_out = np.arange(0, 13)

alpha_draws = rf.beta(2, 2, N_CURVES)              # prior on retention
decay_curves = alpha_draws[:, None] ** weeks_out[None, :]   # impulse worth α^k

fig = go.Figure()
for i in range(N_CURVES):
    fig.add_trace(go.Scatter(x=weeks_out, y=decay_curves[i], mode="lines",
                             line=dict(color=SKY, width=1), opacity=0.18,
                             showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=weeks_out, y=np.median(decay_curves, axis=0),
                         name="median curve", line=dict(color=INK, width=3)))
fig.add_hline(y=0.5, line=dict(color=BERRY, dash="dot"),
              annotation_text="half strength", annotation_font_color=BERRY)
style(fig, f"{N_CURVES} carryover curves drawn from the Beta(2, 2) retention prior",
      "weeks after the ad ran", "remaining effect (share of week-0 impact)").show()

# CLAIM 1: every prior-drawn curve starts at full strength and only decays.
assert np.allclose(decay_curves[:, 0], 1.0)
assert (np.diff(decay_curves, axis=1) <= 0).all()
# CLAIM 2: each curve crosses half strength exactly at its own half-life
# h = ln(.5)/ln(α) — the identity from §2.2, now visible as geometry.
hl = np.log(0.5) / np.log(alpha_draws)
assert np.allclose(alpha_draws ** hl, 0.5)
"""),
    code(r"""
lam_fan = rf.exponential(2.0, N_CURVES)            # the framework's default λ prior
xgrid = np.linspace(0, 1.2, 241)                   # spend as share of max week
sat_curves = 1 - np.exp(-lam_fan[:, None] * xgrid[None, :])

fig = go.Figure()
for i in range(N_CURVES):
    fig.add_trace(go.Scatter(x=xgrid, y=sat_curves[i], mode="lines",
                             line=dict(color=LEAF, width=1), opacity=0.18,
                             showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=xgrid, y=np.median(sat_curves, axis=0),
                         name="median curve", line=dict(color=INK, width=3)))
fig.add_vline(x=1.0, line=dict(color=BERRY, dash="dot"),
              annotation_text="biggest week ever bought", annotation_font_color=BERRY)
style(fig, f"{N_CURVES} response curves drawn from the Exponential(mean 2) λ prior",
      "media pressure (share of max observed week)",
      "effect (share of channel ceiling)").show()

share_linearish = float((sat_curves[:, -1] < 0.5).mean())
print(f"share of prior curves still below half their ceiling at 1.2× max spend: "
      f"{share_linearish:.0%}")

# CLAIM 1: every prior response curve starts at zero, rises, never exceeds 1.
assert np.allclose(sat_curves[:, 0], 0.0)
assert (np.diff(sat_curves, axis=1) >= 0).all() and (sat_curves <= 1).all()
# CLAIM 2: the half-saturation identity holds on every drawn curve.
assert np.allclose(1 - np.exp(-lam_fan * (np.log(2) / lam_fan)), 0.5)
# CLAIM 3 (the gut check): a real minority of prior curves say the channel
# would STILL be under half its ceiling even at 1.2× the max observed week.
assert 0.05 < share_linearish < 0.55
"""),
    code(r"""
# Static partner for the widget below: the same two fans under four
# representative hyperparameter settings.
settings = [("Beta(1,3) + Exp(mean 1)", (1, 3), 1.0),
            ("Beta(2,2) + Exp(mean 2)", (2, 2), 2.0),
            ("Beta(8,2) + Exp(mean 2)", (8, 2), 2.0),
            ("Beta(8,2) + Exp(mean 4)", (8, 2), 4.0)]
fig = make_subplots(rows=2, cols=4, vertical_spacing=0.14, horizontal_spacing=0.05,
                    subplot_titles=[s[0] for s in settings] + [""] * 4,
                    row_titles=["carryover", "response"])
for j, (label, (a, b), lam_mean) in enumerate(settings, start=1):
    rs = np.random.default_rng(SEED_FAN)
    al = rs.beta(a, b, 60)
    lm = rs.exponential(lam_mean, 60)
    for i in range(60):
        fig.add_trace(go.Scatter(x=weeks_out, y=al[i] ** weeks_out, mode="lines",
                                 line=dict(color=SKY, width=1), opacity=0.15,
                                 showlegend=False, hoverinfo="skip"), row=1, col=j)
        fig.add_trace(go.Scatter(x=xgrid, y=1 - np.exp(-lm[i] * xgrid), mode="lines",
                                 line=dict(color=LEAF, width=1), opacity=0.15,
                                 showlegend=False, hoverinfo="skip"), row=2, col=j)
fig.update_yaxes(range=[0, 1.02])
style(fig, "What different hyperparameters SAY: four prior worlds, two behaviors each",
      h=520, w=1000).show()

# CLAIM: the four settings produce visibly different prior worlds — median
# week-4 carryover rises from the fast-decay to the long-memory prior.
m14 = np.median(np.random.default_rng(SEED_FAN).beta(1, 3, 60) ** 4)
m84 = np.median(np.random.default_rng(SEED_FAN).beta(8, 2, 60) ** 4)
assert m84 > 5 * m14
"""),
    code(r"""
# 🎛️ Live exploration (run me!) — reshape both priors and watch the behavior
# fans respond. (Static partner: the 4-setting grid above.)
from ipywidgets import interact, FloatSlider

def behavior_fans(a, b, lam_mean, n=80, seed=SEED_FAN):
    "Pure compute: prior-drawn decay and response curve fans."
    rr = np.random.default_rng(seed)
    al = rr.beta(a, b, n)
    lm = rr.exponential(lam_mean, n)
    return al[:, None] ** weeks_out[None, :], 1 - np.exp(-lm[:, None] * xgrid[None, :])

def show_fans(retention_a=2.0, retention_b=2.0, lam_mean=2.0):
    dec, sat = behavior_fans(retention_a, retention_b, lam_mean)
    f = make_subplots(rows=1, cols=2, subplot_titles=(
        f"carryover fan — Beta({retention_a:g}, {retention_b:g})",
        f"response fan — Exponential(mean {lam_mean:g})"))
    for i in range(dec.shape[0]):
        f.add_trace(go.Scatter(x=weeks_out, y=dec[i], mode="lines",
                               line=dict(color=SKY, width=1), opacity=0.15,
                               showlegend=False, hoverinfo="skip"), row=1, col=1)
        f.add_trace(go.Scatter(x=xgrid, y=sat[i], mode="lines",
                               line=dict(color=LEAF, width=1), opacity=0.15,
                               showlegend=False, hoverinfo="skip"), row=1, col=2)
    f.update_yaxes(range=[0, 1.02])
    style(f, "Your hyperparameters, as the behavior they imply", h=400, w=940).show()

interact(show_fans,
         retention_a=FloatSlider(value=2.0, min=0.5, max=12.0, step=0.5),
         retention_b=FloatSlider(value=2.0, min=0.5, max=12.0, step=0.5),
         lam_mean=FloatSlider(value=2.0, min=0.25, max=6.0, step=0.25));

# CLAIM: the widget's compute is sane across representative settings — fans
# stay inside [0, 1] and respect monotonicity for every slider combo tested.
for a, b, lmn in [(0.5, 0.5, 0.25), (2, 2, 2), (8, 2, 4), (12, 12, 6)]:
    dec, sat = behavior_fans(a, b, lmn)
    assert (np.diff(dec, axis=1) <= 0).all() and (np.diff(sat, axis=1) >= 0).all()
    assert dec.min() >= 0 and dec.max() <= 1 and sat.min() >= 0 and sat.max() <= 1
"""),
    # ====================================================================
    # 4. The prior predictive check
    # ====================================================================
    md(r"""
## 4 — The prior predictive check: simulate sales before you fit

Curves are behavior one transform at a time. The full-strength version of the
same idea runs the **whole model forward from its priors**: draw one value of
*every* parameter from its prior, push spend through the model, add noise —
out comes one completely fake year of weekly sales. Repeat a few hundred times
and you have the **prior predictive distribution** (the spread of datasets
your model considers plausible *before seeing any real data*). Comparing that
spread against common sense is the **prior predictive check**.

Doctrine, worth memorizing: **this is the cheapest, highest-value check in the
entire workflow.** It needs no fitting, no MCMC, no waiting — and it catches
the embarrassing failures (negative sales! 10× revenue weeks!) at the moment
they're easiest to fix: before the data ever gets a vote.

Our toy world: two channels (TV-ish and Search-ish), 104 weeks, sales in
$thousands. The model is exactly the MMM skeleton you'll meet for real in
[workshop_03](workshop_03_first_mmm.ipynb):
intercept + β·saturate(adstock(spend)) per channel + noise. Suppose history
says weekly sales live roughly between $800k and $1.3M. First attempt: an
analyst who "doesn't want to assume anything" sets huge, vague priors.
"""),
    code(r"""
# ---- the toy world: spend for two channels, 104 weeks (seeded) -------------
SEED_WORLD, N_WEEKS = 7, 104
rw = np.random.default_rng(SEED_WORLD)
spend_tv = rw.gamma(3.0, 15.0, N_WEEKS)        # lumpy, bursty ($k)
spend_se = rw.gamma(5.0, 6.0, N_WEEKS)         # steadier ($k)
x_tv, x_se = spend_tv / spend_tv.max(), spend_se / spend_se.max()
SALES_LO, SALES_HI = 800, 1300                 # last year's actual range ($k)

def geometric_adstock(x, alpha, l_max=8):
    w = alpha ** np.arange(l_max)
    return np.convolve(x, w / w.sum())[: len(x)]

def saturate(x, lam):
    return 1 - np.exp(-lam * x)

def prior_predictive(seed, n_draws, intercept_mu, intercept_sd, beta_sd, noise_sd):
    "Run the 2-channel MMM forward from its priors -> fake sales (draws, weeks)."
    rr = np.random.default_rng(seed)
    ys = np.empty((n_draws, N_WEEKS))
    for d in range(n_draws):
        b1, b2 = abs(rr.normal(0, beta_sd)), abs(rr.normal(0, beta_sd))  # HalfNormal
        a1, a2 = rr.beta(2, 2), rr.beta(2, 2)
        l1, l2 = rr.exponential(2.0), rr.exponential(2.0)
        ic = rr.normal(intercept_mu, intercept_sd)
        sd = abs(rr.normal(0, noise_sd))
        mu = (ic + b1 * saturate(geometric_adstock(x_tv, a1), l1)
                 + b2 * saturate(geometric_adstock(x_se, a2), l2))
        ys[d] = mu + rr.normal(0, sd, N_WEEKS)
    return ys

def fan_plot(ys, title, n_paths=40):
    lo5, hi95 = np.percentile(ys, [5, 95], axis=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[np.arange(N_WEEKS), np.arange(N_WEEKS)[::-1]],
                             y=np.r_[hi95, lo5[::-1]], fill="toself",
                             fillcolor="rgba(59,110,165,0.18)",
                             line=dict(width=0), name="90% of prior draws"))
    for i in range(n_paths):
        fig.add_trace(go.Scatter(x=np.arange(N_WEEKS), y=ys[i], mode="lines",
                                 line=dict(color=SKY, width=1), opacity=0.25,
                                 showlegend=False, hoverinfo="skip"))
    fig.add_hrect(y0=SALES_LO, y1=SALES_HI, fillcolor=LEAF, opacity=0.15, line_width=0)
    fig.add_hline(y=0, line=dict(color=BERRY, width=1.5, dash="dash"),
                  annotation_text="zero sales", annotation_font_color=BERRY)
    fig.add_annotation(x=2, y=(SALES_LO + SALES_HI) / 2, xanchor="left",
                       text="where real weekly sales<br>actually live",
                       showarrow=False, font=dict(color=LEAF, size=11))
    return style(fig, title, "week", "weekly sales ($k)", h=440)

y_vague = prior_predictive(11, 300, intercept_mu=0, intercept_sd=2000,
                           beta_sd=3000, noise_sd=500)
fan_plot(y_vague, "Prior predictive under 'assume nothing' priors — fake sales the model finds plausible").show()

share_neg_weeks = float((y_vague < 0).any(axis=1).mean())
share_in_band = float(((y_vague > 400) & (y_vague < 2200)).mean())
print(f"prior draws containing at least one NEGATIVE sales week: {share_neg_weeks:.0%}")
print(f"share of simulated weekly values within shouting distance "
      f"($400k–$2.2M) of reality: {share_in_band:.0%}")

# CLAIM: the 'no assumptions' priors confidently predict nonsense — a large
# share of fake years contain negative sales weeks, and most simulated values
# land nowhere near the range sales have ever occupied.
assert share_neg_weeks > 0.30
assert share_in_band < 0.40
"""),
    code(r"""
# Round 2: say what you actually know. Sales hover around $1M/week (intercept
# Normal(1000, 150)); no channel plausibly adds more than a few hundred $k at
# full saturation (beta HalfNormal(150)); week-to-week noise is tens of $k.
y_sane = prior_predictive(11, 300, intercept_mu=1000, intercept_sd=150,
                          beta_sd=150, noise_sd=60)
fan_plot(y_sane, "Prior predictive after the check — fake sales now look like sales").show()

share_neg_weeks2 = float((y_sane < 0).any(axis=1).mean())
share_in_band2 = float(((y_sane > 400) & (y_sane < 2200)).mean())
print(f"prior draws with a negative sales week: {share_neg_weeks2:.0%}")
print(f"simulated weekly values within $400k–$2.2M: {share_in_band2:.0%}")

# CLAIM: weakly informative priors produce prior-predictive sales that respect
# basic reality — essentially no negative weeks, nearly all mass in a sane
# range — while remaining far looser than the data will be.
assert share_neg_weeks2 < 0.02
assert share_in_band2 > 0.95
# ...and the band is still generous, not dogmatic: its 90% width comfortably
# exceeds the actual historical range, leaving the data room to speak.
width = np.percentile(y_sane, 95) - np.percentile(y_sane, 5)
assert width > (SALES_HI - SALES_LO)
"""),
    md(r"""
Compare the two fans. Nothing about the second set of priors is "biased" — it
still entertains flat TV, powerful TV, fast and slow carryover, every
saturation speed. It has simply stopped insisting that *negative million-dollar
sales weeks* deserve belief. That's the whole craft: **rule out the absurd,
stay loose about the plausible.**

Two habits to take away:

- Run the prior predictive **every time you change a prior or add a channel.**
  It costs seconds.
- Judge it against **outside knowledge** (what sales *can* be), never against
  the dataset you're about to fit — tuning priors to the data you'll fit is
  using the data twice.

In [workshop_04](workshop_04_reading_the_posterior.ipynb) you'll meet the
posterior predictive check — same trick, run *after* fitting.
"""),
    # ====================================================================
    # 5. How the framework says it
    # ====================================================================
    md(r"""
## 5 — How `mmm_framework` says it

Everything above was numpy so you could see the gears. In the framework, a
prior is a small config object — `PriorConfig` — holding a distribution name
and its hyperparameters, and you attach priors to the things they govern:
the adstock config carries the retention prior, the saturation config carries
the λ prior, the channel config carries the effect-size prior. You can build
them directly or through fluent builders; both produce identical configs.
"""),
    code(r"""
from mmm_framework.config import PriorConfig, PriorType
from mmm_framework.builders import PriorConfigBuilder

# Direct construction (classmethod shortcuts)...
effect_prior = PriorConfig.half_normal(sigma=2.0)
retention_prior = PriorConfig.beta(alpha=8, beta=2)        # long-memory carryover
rate_prior = PriorConfig.gamma(alpha=6, beta=2)            # confident saturation

# ...or the fluent builder — same object either way.
effect_prior_b = PriorConfigBuilder().half_normal(sigma=2.0).build()
retention_prior_b = PriorConfigBuilder().beta(alpha=8, beta=2).build()

for p in (effect_prior, retention_prior, rate_prior):
    print(p.distribution.value, p.params)

# CLAIM: builder and classmethod construct the SAME config; the config is a
# plain (distribution, hyperparameters) record matching the zoo of §2.
assert effect_prior == effect_prior_b and retention_prior == retention_prior_b
assert effect_prior.distribution == PriorType.HALF_NORMAL
assert effect_prior.params == {"sigma": 2.0}
assert retention_prior.params == {"alpha": 8.0, "beta": 2.0}
"""),
    code(r"""
from mmm_framework.config import AdstockConfig, SaturationConfig, MediaChannelConfig, DimensionType
from mmm_framework.builders import MediaChannelConfigBuilder

# A channel's priors live on its transform configs + the channel itself.
tv_channel = MediaChannelConfig(
    name="TV",
    dimensions=[DimensionType.PERIOD],
    adstock=AdstockConfig.geometric(           # carryover: retention prior
        l_max=8, alpha_prior=PriorConfig.beta(alpha=3, beta=3)),
    saturation=SaturationConfig.logistic(),    # λ keeps the built-in Exponential default
    # Effect-size prior override for the core model. NOTE the field name:
    # despite being called roi_prior it is a prior on the (saturation-scaled)
    # coefficient beta_TV — when None, the core model's default is
    # Gamma(mu=1.5, sigma=1.0), which gently favors paying channels.
    roi_prior=PriorConfig.half_normal(sigma=1.0),
)

# The fluent route builds the same kind of object.
search_channel = (MediaChannelConfigBuilder("Search")
                  .with_geometric_adstock(l_max=4)
                  .with_logistic_saturation()
                  .with_positive_prior(sigma=2.0)   # sets coefficient_prior (HalfNormal)
                  .build())

print("TV retention prior:   ", tv_channel.adstock.alpha_prior.params)
print("TV effect prior:      ", tv_channel.roi_prior.distribution.value,
      tv_channel.roi_prior.params)
print("Search adstock l_max: ", search_channel.adstock.l_max)

# CLAIM: the configs carry exactly the priors we wrote — retention Beta(3,3)
# with an 8-week window on TV, a HalfNormal(1) effect prior on TV, and a
# 4-week carryover window on Search with a HalfNormal(2) coefficient prior.
assert tv_channel.adstock.alpha_prior.params == {"alpha": 3.0, "beta": 3.0}
assert tv_channel.adstock.l_max == 8
assert tv_channel.roi_prior.params == {"sigma": 1.0}
assert search_channel.adstock.l_max == 4
assert search_channel.coefficient_prior.distribution == PriorType.HALF_NORMAL
assert search_channel.coefficient_prior.params == {"sigma": 2.0}
"""),
    md(r"""
### The framework's own prior predictive

Now the real thing: hand those channel configs to `BayesianMMM` (the model
class you'll meet properly in [workshop_03](workshop_03_first_mmm.ipynb)) and
ask it to simulate sales from its priors — no fitting involved. Two details
worth knowing:

- The framework **standardizes** the KPI internally (works in "standard
  deviations from the mean" units), which is itself a quiet prior-hygiene
  trick: priors live on a unitless scale, so defaults transfer across brands.
  The simulated KPI back in original units is exposed as the `y_obs_scaled`
  variable — and because it's a *derived* quantity, it lands in the `prior`
  group of the returned object (the `prior_predictive` group holds the
  standardized-scale `y_obs`).
- `sample_prior_predictive(samples=...)` does not honor a random seed, so the
  checks below assert on **robust invariants** (shapes, supports, coverage)
  rather than exact numbers.
"""),
    code(r"""
from mmm_framework.config import (ControlVariableConfig, InferenceMethod, KPIConfig,
                                  MFFConfig, ModelConfig)
from mmm_framework.data_loader import PanelCoordinates, PanelDataset
from mmm_framework.model import BayesianMMM, TrendConfig, TrendType

# Re-quiet after pymc import (it installs its own log handlers).
for _n in ("pymc", "pymc.sampling", "numpyro", "jax", "arviz", "pytensor"):
    _lg = logging.getLogger(_n); _lg.setLevel(logging.CRITICAL); _lg.propagate = False

# Package the toy world from §4 as a framework panel.
weeks_idx = pd.date_range("2024-01-01", periods=N_WEEKS, freq="W-MON")
y_hist = pd.Series(  # plausible observed sales, only used for scaling/overlay
    1000 + 150 * saturate(geometric_adstock(x_tv, 0.6), 2.0)
         + 100 * saturate(geometric_adstock(x_se, 0.3), 3.0)
         + np.random.default_rng(1).normal(0, 40, N_WEEKS), name="Sales")
spend_df = pd.DataFrame({"TV": spend_tv, "Search": spend_se}, index=weeks_idx)
controls_df = pd.DataFrame(
    {"price_index": np.random.default_rng(2).normal(100, 3, N_WEEKS)}, index=weeks_idx)

mff = MFFConfig(
    kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
    media_channels=[tv_channel, search_channel],
    controls=[ControlVariableConfig(name="price_index", dimensions=[DimensionType.PERIOD])],
)
panel = PanelDataset(
    y=y_hist, X_media=spend_df, X_controls=controls_df,
    coords=PanelCoordinates(periods=weeks_idx, geographies=None, products=None,
                            channels=["TV", "Search"], controls=["price_index"]),
    index=weeks_idx, config=mff)

cfg = ModelConfig(inference_method=InferenceMethod.BAYESIAN_NUMPYRO,
                  use_parametric_adstock=True)  # honor the configured adstock priors
mmm = BayesianMMM(panel, cfg, TrendConfig(type=TrendType.LINEAR))
with quiet():
    idata = mmm.sample_prior_predictive(samples=300)

pp = idata.prior["y_obs_scaled"].values[0]        # (draws, weeks), original $k scale
lo5, hi95 = np.percentile(pp, [5, 95], axis=0)
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.r_[np.arange(N_WEEKS), np.arange(N_WEEKS)[::-1]],
                         y=np.r_[hi95, lo5[::-1]], fill="toself",
                         fillcolor="rgba(217,138,43,0.20)", line=dict(width=0),
                         name="90% of framework prior draws"))
for i in range(30):
    fig.add_trace(go.Scatter(x=np.arange(N_WEEKS), y=pp[i], mode="lines",
                             line=dict(color=AMBER, width=1), opacity=0.3,
                             showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=np.arange(N_WEEKS), y=y_hist, mode="markers",
                         marker=dict(color=INK, size=4), name="observed sales"))
style(fig, "mmm_framework's prior predictive: simulated sales from YOUR PriorConfigs",
      "week", "weekly sales ($k)", h=440).show()

# CLAIMS (robust invariants only — the samples= API ignores random_seed):
# 1. y_obs_scaled is a prior-group Deterministic with shape (chain, draw, week).
assert "y_obs_scaled" in idata.prior and "prior_predictive" in [g.strip("/") for g in idata.groups]
assert pp.shape == (300, N_WEEKS) and np.isfinite(pp).all()
# 2. the configured priors were honored: TV effect draws are HalfNormal(1)
#    (never negative), retention draws live in (0,1) with a Beta(3,3)-ish mean.
beta_tv = idata.prior["beta_TV"].values.ravel()
alpha_tv = idata.prior["adstock_alpha_TV"].values.ravel()
assert (beta_tv >= 0).all() and (alpha_tv > 0).all() and (alpha_tv < 1).all()
assert 0.3 < alpha_tv.mean() < 0.7
# 3. the prior fan brackets the observed sales level (sane original-scale check).
assert np.percentile(pp, 2) < float(y_hist.mean()) < np.percentile(pp, 98)
print("framework prior-predictive fan brackets observed sales ✓")
"""),
    md(r"""
That's the full configuration loop you'll reuse for real in workshop_03:
**zoo shape → `PriorConfig` → attach to `AdstockConfig` / `SaturationConfig` /
channel → simulate sales → eyeball → adjust.** The fitting part — making the
data argue with these priors — is exactly what
[workshop_02](workshop_02_sampling.ipynb) (the machinery) and workshop_03
(the practice) are about. We deliberately stop here; priors first, sampling
second.
"""),
    # ====================================================================
    # 6. Sensitivity in one picture
    # ====================================================================
    md(r"""
## 6 — Sensitivity in one picture: when priors matter

The last worry every newcomer has: *"doesn't the prior just decide the
answer?"* The honest reply is **it depends on how much the data knows** — and
you can see the whole story in one picture, no MCMC required.

Tiny model: weekly incremental sales $y_t = \beta x_t + \varepsilon_t$, where
$x_t$ is a known media-pressure index and the noise level is known. For this
special case the posterior has an exact pencil-and-paper formula (the Normal
prior is **conjugate** here — maths jargon for "prior and posterior come out
in the same family, so no sampler is needed"), so we can compare three priors
of very different strength instantly:

- near-flat: Normal(0, 50) — says almost nothing;
- weakly informative: Normal(1, 1) — "around 1-ish, loosely";
- strongly informative *and miscalibrated*: Normal(0.5, 0.15) — confident,
  and (unknown to its author) centered well below the truth.

The world's true β is 1.2. We give the model **6 weeks** of data, then **3
years**, and watch the three posteriors.
"""),
    code(r"""
SEED_SENS, TRUE_BETA, NOISE_SD = 21, 1.2, 1.0
rs = np.random.default_rng(SEED_SENS)
N_BIG, N_SMALL = 156, 6
x_sens = rs.uniform(0.2, 1.0, N_BIG)
y_sens = TRUE_BETA * x_sens + rs.normal(0, NOISE_SD, N_BIG)

PRIORS = {"near-flat: Normal(0, 50)": (0.0, 50.0, MUTED),
          "weakly informative: Normal(1, 1)": (1.0, 1.0, SKY),
          "confident & WRONG: Normal(0.5, 0.15)": (0.5, 0.15, BERRY)}

def exact_posterior(mu0, sd0, x, y, sigma=NOISE_SD):
    "Conjugate Normal posterior for beta in y = beta*x + noise (sigma known)."
    prec = 1 / sd0**2 + (x**2).sum() / sigma**2
    mean = (mu0 / sd0**2 + (x * y).sum() / sigma**2) / prec
    return mean, prec**-0.5

grid = np.linspace(-0.6, 2.6, 1201)
fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=(
    f"after {N_SMALL} weeks of data", f"after {N_BIG} weeks (3 years)"))
post = {}
for col, n in [(1, N_SMALL), (2, N_BIG)]:
    for name, (mu0, sd0, color) in PRIORS.items():
        m, s = exact_posterior(mu0, sd0, x_sens[:n], y_sens[:n])
        post[(name, n)] = (m, s)
        fig.add_trace(go.Scatter(x=grid, y=stats.norm(m, s).pdf(grid), name=name,
                                 line=dict(color=color, width=2.5),
                                 legendgroup=name, showlegend=(col == 1)),
                      row=1, col=col)
    fig.add_vline(x=TRUE_BETA, line=dict(color=LEAF, width=2, dash="dot"),
                  annotation_text="truth" if col == 1 else None,
                  annotation_font_color=LEAF, row=1, col=col)
fig.update_xaxes(title_text="TV effect β", range=[-0.4, 2.4])
style(fig, "Three priors, two data sizes: priors matter when data is scarce",
      h=430, w=960).show()

means_small = [post[(k, N_SMALL)][0] for k in PRIORS]
means_big = [post[(k, N_BIG)][0] for k in PRIORS]
spread_small, spread_big = np.ptp(means_small), np.ptp(means_big)
flat_vs_weak_big = abs(post[("near-flat: Normal(0, 50)", N_BIG)][0]
                       - post[("weakly informative: Normal(1, 1)", N_BIG)][0])
print(f"spread of the three posterior means:  {N_SMALL} weeks -> {spread_small:.2f}, "
      f"{N_BIG} weeks -> {spread_big:.2f}")
print(f"near-flat vs weakly-informative gap at {N_BIG} weeks: {flat_vs_weak_big:.3f}")

# CLAIM 1: with scarce data, the choice of prior visibly moves the answer.
assert spread_small > 2 * spread_big
# CLAIM 2: with 3 years of data, flat and weakly-informative priors agree —
# reasonable priors WASH OUT as evidence accumulates.
assert flat_vs_weak_big < 0.05
# CLAIM 3: the overconfident wrong prior is still dragging the answer even at
# 3 years — its posterior mean sits between its prior mean and the truth.
m_conf = post[("confident & WRONG: Normal(0.5, 0.15)", N_BIG)][0]
assert 0.5 < m_conf < TRUE_BETA
# CLAIM 4: more data tightens every posterior.
for k in PRIORS:
    assert post[(k, N_BIG)][1] < post[(k, N_SMALL)][1]
"""),
    code(r"""
# 🎛️ Live exploration (run me!) — how confident a prior, how much data?
# (Static partner: the 3-prior × 2-sample-size panel above.)
from ipywidgets import interact, FloatSlider, IntSlider

def show_sensitivity(prior_mean=0.5, prior_sd=0.5, n_weeks=12):
    m, s = exact_posterior(prior_mean, prior_sd, x_sens[:n_weeks], y_sens[:n_weeks])
    f = go.Figure()
    f.add_trace(go.Scatter(x=grid, y=stats.norm(prior_mean, prior_sd).pdf(grid),
                           name="prior", line=dict(color=MUTED, width=2, dash="dash")))
    f.add_trace(go.Scatter(x=grid, y=stats.norm(m, s).pdf(grid),
                           name="posterior", line=dict(color=SKY, width=3)))
    f.add_vline(x=TRUE_BETA, line=dict(color=LEAF, width=2, dash="dot"),
                annotation_text="truth", annotation_font_color=LEAF)
    style(f, f"prior Normal({prior_mean:g}, {prior_sd:g}) + {n_weeks} weeks "
             f"-> posterior mean {m:.2f}", "TV effect β", "density",
          h=400, w=840).show()

interact(show_sensitivity,
         prior_mean=FloatSlider(value=0.5, min=-1.0, max=2.5, step=0.1),
         prior_sd=FloatSlider(value=0.5, min=0.05, max=5.0, step=0.05),
         n_weeks=IntSlider(value=12, min=2, max=N_BIG, step=2));

# CLAIM: the conjugate identity behind the widget — posterior precision is
# EXACTLY prior precision + data precision, for representative slider settings.
for mu0, sd0, n in [(0.5, 0.05, 6), (0.0, 5.0, 12), (1.0, 0.5, 156)]:
    m, s = exact_posterior(mu0, sd0, x_sens[:n], y_sens[:n])
    assert np.isclose(1 / s**2, 1 / sd0**2 + (x_sens[:n] ** 2).sum() / NOISE_SD**2)
    assert s < sd0  # data can only sharpen a conjugate Normal belief
"""),
    md(r"""
Three lessons, straight off the picture:

1. **Small data ⇒ the prior is load-bearing.** With six weeks, the three
   analysts walk out of the meeting with three different βs. Choose priors
   like they matter — because here they do.
2. **Reasonable priors wash out.** With three years, near-flat and weakly
   informative land on the same answer. The weakly informative one got there
   with less drama (narrower at every stage, no absurd detours).
3. **Overconfident wrong priors don't wash out politely.** The Normal(0.5,
   0.15) author is *still* biased after three years. Informative priors are
   powerful exactly because they're strong — earn them with evidence (lift
   tests, last year's model), and prior-predictive-check them like everything
   else.

And the MMM-specific kicker, foreshadowing the rest of the series: a real MMM
is *permanently* in the left panel. Channels move together, carryover and
saturation trade off against each other, and three years of weekly data is
only 156 rows against dozens of parameters — the data is never as loud as
you'd hope (**weak identification**, the situation where the data alone can't
pin parameters down). In MMM, priors aren't training wheels you outgrow;
they're part of the vehicle.

## 7 — Glossary & what's next

| term | plain English |
|---|---|
| **prior** | your starting beliefs about a parameter, written as a distribution |
| **flat / uninformative prior** | spreads belief evenly over a huge range; sounds neutral, usually isn't |
| **weakly informative prior** | rules out the absurd, loose about the plausible — your default |
| **informative prior** | commits to real external knowledge (benchmarks, lift tests) |
| **hyperparameter** | a knob that defines a prior (the σ in HalfNormal(σ)) |
| **support** | the set of values a distribution allows at all |
| **adstock / retention rate** | carryover; the fraction of ad pressure surviving each week |
| **half-life** | weeks until carryover halves: ln(0.5)/ln(α) |
| **saturation rate λ** | how fast diminishing returns bite in 1−e^(−λx) |
| **half-saturation point** | spend where you've banked half the ceiling: ln2/λ |
| **prior predictive distribution** | fake datasets simulated from priors alone |
| **prior predictive check** | comparing those fakes against common sense, before fitting |
| **conjugate** | prior/posterior in the same family — exact pencil-and-paper update |
| **weak identification** | when the data alone can't pin parameters down — MMM's natural habitat |

**Where you are in the series.** You can now *say what you know* and verify
the model heard you. But everything here dodged a question: for real models
there's no pencil-and-paper posterior — so how does the machine actually
compute one? That's
**[workshop_02 — Sampling: How the Machine Learns](workshop_02_sampling.ipynb)**:
why we draw samples instead of solving equations, a Metropolis sampler you
can watch stumble, and the diagnostics that tell you whether to trust the
draws. Then workshop_03 puts priors + sampler together on your first real
`BayesianMMM` fit.
"""),
]


def main():
    nb = new_notebook(cells=CELLS)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "title": "Workshop 01 — Priors: Saying What You Know",
    })
    path = "workshop/workshop_01_priors.ipynb"
    with open(path, "w") as fh:
        nbformat.write(nb, fh)
    print(f"wrote {path} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
