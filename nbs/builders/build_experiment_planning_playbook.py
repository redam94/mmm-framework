"""Author demos/experiment_planning_playbook.ipynb (run from ``nbs/``).

    uv run --with nbformat python builders/build_experiment_planning_playbook.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        demos/experiment_planning_playbook.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

The experiment-planning playbook, end to end, on one synthetic geo world with a
known answer key: the method registry (SCM / TBR / GBR / DiD-MMT / ghost ads /
switchback / flighting), planning each design with honest power math, the
A/A·A/B methodology leaderboard, EIG/EVOI experiment selection (including
non-geo designs through the sigma_exp bridge), test economics (opportunity cost
+ net test value), the **net-value Pareto optimizer** with its two-anchor
Gaussian EVOI surrogate (2026-07-19), a surrogate-vs-MC validation, and the
close of the loop: calibrating the MMM with the readout. Companion to
``docs/experiment-playbook.html``.
"""

from __future__ import annotations

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


SETUP = r"""
import warnings; warnings.filterwarnings("ignore")
import os
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.templates.default = "plotly_white"
pio.renderers.default = "notebook_connected"
pd.set_option("display.width", 160)

import logging
from loguru import logger
logger.disable("mmm_framework")
for _n in ("pymc", "pymc.sampling", "numpyro", "jax", "arviz"):
    logging.getLogger(_n).setLevel(logging.ERROR)

# palette (matches the docs site's editorial scheme)
INK, MUTED = "#1f2430", "#8a8f98"
GOOD, BAD, GOLD = "#3d7a5c", "#b4552d", "#c9962e"
PALETTE = {"TV": "#4464ad", "Search": "#c9962e", "Social": "#3d7a5c", "Display": "#b4552d"}

def style(fig, height=380, title=None, **kw):
    fig.update_layout(height=height, title=title, margin=dict(t=60, l=60, r=30, b=50),
                      font=dict(size=12), **kw)
    return fig

KPI = "Sales"
print("Experiment-planning playbook — setup ready.")
"""


CELLS = [
    md(r"""
# The Experiment-Planning Playbook
### Every design the framework supports, how to power it, price it, pick it — and fold it back into the MMM

An MMM tells you where the model is *uncertain*; an experiment is how you buy the
missing information. This notebook demonstrates the full planning surface on one
synthetic world with a **known answer key**:

1. **The method registry** — SCM, TBR/CausalImpact, GBR, DiD-MMT, ghost ads,
   switchbacks, flighting — one catalogue, filtered by what your data supports.
2. **Planning each design** with honest power math (matched pairs, ITT dilution,
   carryover-aware blocks, AR(1) design effects) — including power on the
   client's scale: **maximum detectable cost per conversion**, and why the
   skewed CPA distribution can't be produced by flipping the lift numbers.
3. **The methodology leaderboard** — audition every estimator on your own history
   (A/A false-positive rate before A/B power).
4. **Which test to run** — the EIG/EVOI priority grid, including **non-geo**
   designs through the `sigma_exp` bridge.
5. **What it's worth** — opportunity cost, net test value, and the **net-value
   Pareto optimizer** (new: a two-anchor Gaussian EVOI surrogate prices every
   candidate design in dollars).
6. **Closing the loop** — the readout becomes a calibration likelihood in the
   next fit.

Companion doc: [`docs/experiment-playbook.html`](../../docs/experiment-playbook.html).
"""),
    code(SETUP),
    md(r"""
## The world — a geo panel with a known answer key

Eight markets, a year and a half of weekly data, four channels. Because the
world is synthetic we know the true ROAS of every channel — the luxury no real
dataset gives you, and the reason a playbook should be rehearsed on synthetic
ground truth first. (A *deliberately* modest history: with only 78 weeks the
posterior stays wide enough that experiments have real value to price — a
longer, cleaner panel would leave less for a test to teach.) We also keep a
**national** spine (no geo splits) to demonstrate the designs you fall back on
when there is no panel.
"""),
    code(r"""
import tempfile
from mmm_framework.synth import dgp_geo, generate_mff
from mmm_framework.synth.mff import geo_scenario_to_mff

WORK = os.path.join(tempfile.gettempdir(), "mmm_experiment_playbook")
os.makedirs(WORK, exist_ok=True)

GEOS = ["North", "South", "East", "West", "Metro", "Coast", "Plains", "Hills"]
sc = dgp_geo.build("geo_heterogeneous", seed=3, geos=GEOS, n_weeks=78)
geo_df = geo_scenario_to_mff(sc)
GEO_CSV = os.path.join(WORK, "geo_world.csv")
geo_df.to_csv(GEO_CSV, index=False)

nat_df, nat_key = generate_mff("realistic", seed=5, n_weeks=130)
NAT_CSV = os.path.join(WORK, "national_world.csv")
nat_df.to_csv(NAT_CSV, index=False)

CHANNELS = ["TV", "Search", "Social", "Display"]
print(f"geo world:      {len(GEOS)} geos x 78 weeks -> {GEO_CSV}")
print(f"national world: 1 geo x 130 weeks           -> {NAT_CSV}")
"""),
    md(r"""
## 1 · The method registry — one catalogue of experiment methodologies

Every supported methodology is a named `MethodSpec` in `planning/methods/`:
its data requirements, its estimator, its power math, its references. The
design tools, the REST API and the agent all enumerate this same registry —
"which methods can my data support?" has one answer everywhere.
"""),
    code(r"""
from mmm_framework.planning.methods import list_methods, methods_for_data

cat = pd.DataFrame([{
    "key": m.key, "name": m.name, "family": m.requirement.family,
    "min_geos": m.requirement.min_geos or "-",
    "min_pre_weeks": m.requirement.min_pre_weeks,
    "needs_panel": m.requirement.needs_panel,
} for m in list_methods()])
print("the registry:")
print(cat.to_string(index=False))

# ...filtered by what THIS dataset can actually support:
support = pd.DataFrame(methods_for_data(n_geos=len(GEOS), n_weeks=78))
print("\nsupported on the 8-geo panel:")
print(support[["key", "supported", "reason"]].fillna("").to_string(index=False))
"""),
    md(r"""
## 2 · Planning a geo experiment — matched pairs + a power curve

`design_options` says what the data supports; `design_experiment` builds a
runnable plan: markets matched into pairs on pre-period behaviour, treatment
randomized within pair, and a **power curve** — the minimum detectable effect
(MDE) at every candidate duration. The MDE is the design's blind spot: an
effect smaller than it will come back "inconclusive" no matter what is true.
"""),
    code(r"""
from mmm_framework.planning import design_experiment, design_options

opts = design_options(GEO_CSV, KPI, "TV")
print(f"design families supported: {opts['designs']}   recommended: {opts['recommended']}")

plan = design_experiment(GEO_CSV, KPI, "TV", design="holdout", n_pairs=4, duration=8)
asg = pd.DataFrame(plan["assignment"])[["treatment", "control", "correlation", "residual_correlation"]]
print("\nmatched pairs (treated market goes dark; its twin keeps spending):")
print(asg.round(3).to_string(index=False))
print(f"\nMDE(ROAS) at {plan['duration']}w: {plan['mde_roas']:.2f}   (SE source: {plan['se_source']})")

pc = pd.DataFrame(plan["power_curve"])
fig = go.Figure(go.Scatter(x=pc["duration"], y=pc["mde_roas"], mode="lines+markers",
                           line=dict(color=PALETTE["TV"], width=3), showlegend=False))
fig.add_vline(x=plan["duration"], line=dict(color=MUTED, width=1.5, dash="dash"),
              annotation_text=f"chosen {plan['duration']}w")
style(fig, height=340, title="Geo holdout power curve — longer test, smaller detectable effect")
fig.update_layout(xaxis_title="test duration (weeks)", yaxis_title="MDE (ROAS)")
fig.show()
"""),
    md(r"""
## 3 · Choosing the *analysis* — the same design, five estimators

The footprint and schedule are one decision; **how you read the result out** is
another. Passing `method=` picks the named methodology: the plan carries the
method's metadata and its analysis-plan estimator line is rewritten, so the
pre-registered plan and the eventual readout use the same estimator by
construction. (Where the estimators genuinely differ — validity and power on
*your* history — is the leaderboard's job, next.)
"""),
    code(r"""
rows = []
for m in ["did_mmt", "synthetic_control", "regadj_geo", "tbr", "gbr"]:
    d = design_experiment(GEO_CSV, KPI, "TV", method=m, design="holdout", n_pairs=4, duration=8)
    rows.append({
        "method": m, "name": d.get("method_name", m),
        "design_key": d["design_key"], "mde_roas": round(d["mde_roas"], 2),
        "references": ", ".join(d.get("method_references") or []) or "-",
    })
print(pd.DataFrame(rows).to_string(index=False))
print("\nSame footprint, same MDE — the method chooses the ESTIMATOR the readout")
print("and the economics simulation will use, not the field design.")
"""),
    md(r"""
## 4 · Ghost ads — user-level power with no panel at all

Ghost ads randomize *individuals*: treated users see the real ad, ghost/PSA
users a logged placebo. The randomization lives in the ad server, so this
method needs **no geo panel and no fitted model** — it is a standalone power
calculator. Three honesty devices: **ITT vs TOT** (partial exposure dilutes the
measurable effect), a **rare-event flag** (the normal approximation needs ~30
events per arm), and a **break-even lift** when you price the conversion.
"""),
    code(r"""
from mmm_framework.planning.methods import (
    GhostAdsDesign, ghost_ads_power, ghost_ads_simulate, ghost_ads_users_for_mde,
)

design = GhostAdsDesign(
    users_reached=400_000, baseline_rate=0.021, treated_fraction=0.5,
    exposure_rate=0.65, value_per_conversion=38.0, cost_per_user=0.05,
)
p = ghost_ads_power(design)
print(f"MDE (absolute lift): {p['mde_abs']:.4f}   relative: {p['mde_rel']:.1%}")
print(f"ITT MDE {p['itt_mde']:.4f}  vs TOT MDE {p['tot_mde']:.4f}  (exposure {p['exposure_rate']:.0%})")
print(f"incremental value at the MDE: ${p['incremental_value_at_mde']:,.0f}  "
      f"media cost: ${p['media_cost']:,.0f}")
print(f"break-even lift: {p['breakeven_lift_abs']:.4f}   rare-event regime: {p['rare_event_regime']}")

need = ghost_ads_users_for_mde(design, target_lift_abs=0.002)
print(f"\nusers required for a +0.2pp MDE: {need:,}")

sim = ghost_ads_simulate(design, p["mde_abs"])
print(f"Monte-Carlo check — empirical FPR: {sim['empirical_fpr']:.1%} (should be ~5%), "
      f"empirical power at the MDE: {sim['empirical_power']:.0%} (should be ~80%)")

reach = np.array([50_000, 100_000, 200_000, 400_000, 800_000, 1_600_000])
mdes = [ghost_ads_power(GhostAdsDesign(users_reached=int(n), baseline_rate=0.021,
                                       exposure_rate=0.65))["mde_rel"] for n in reach]
fig = go.Figure(go.Scatter(x=reach, y=mdes, mode="lines+markers",
                           line=dict(color=PALETTE["Search"], width=3), showlegend=False))
style(fig, height=340, title="Ghost ads: reach buys precision (relative MDE vs users reached)")
fig.update_layout(xaxis_title="users reached", yaxis_title="MDE (relative lift)",
                  xaxis_type="log", yaxis_tickformat=".0%")
fig.show()
"""),
    md(r"""
### Cost per conversion — the number clients ask for, and why you can't just flip the lift

Clients rarely ask "what lift can this test detect?". They ask **"what did a
conversion cost me?"** — spend divided by the extra conversions the media
caused. It is tempting to take the lift analysis and simply divide: flip the
estimate, flip the error bars, done. **That flip is wrong**, and the reason is
worth one minute of intuition:

The *lift* estimate behaves nicely — a symmetric bell curve around the truth.
But cost per conversion **divides a fixed cost by that bell curve**, and
division treats the two sides of the bell very differently. A readout that
comes in a little *high* on conversions moves the cost down a few dollars; a
readout that comes in a little *low* moves the cost up a *lot* — and a readout
near **zero** conversions sends the cost toward infinity. Same symmetric
wobble in, wildly lopsided wobble out.

Three practical consequences:

1. **The average simulated readout overshoots the truth** — the long expensive
   tail drags the mean up, so "expected CPA" quietly flatters to the pessimist
   and the median is the better single number.
2. **The symmetric ± error bar lies.** Flipping the lift interval's width
   (the "delta-method" shortcut) produces an interval that under-covers — and
   in a weak readout it can even dip *below zero* (a negative cost per
   conversion!). The honest interval is **asymmetric**, and when the lift
   interval touches zero it is **unbounded above**: the data genuinely cannot
   rule out an arbitrarily bad CPA, and the report should say so.
3. **The planning number that survives the flip is the *maximum detectable
   cost per conversion*** — invert the detectable-lift *bound*, never the
   point estimate: a design that can detect lift ≥ MDE can certify
   "CPA is at most cost ÷ MDE". Anything worse is indistinguishable from
   "the media did nothing".

`planning.cpa` packages all three (`simulate_cpa_distribution`,
`cpa_interval`, `max_detectable_cpa`, `cpa_power`), and the ghost-ads
calculator now reports its **max detectable CPA** directly. Watch each one on
the 400k-user design.
"""),
    code(r"""
from mmm_framework.planning import simulate_cpa_distribution

COST_PER_USER = design.cost_per_user
true_lift = 1.25 * p["mde_abs"]          # a realistic, mildly-comfortable truth
sim = simulate_cpa_distribution(true_lift=true_lift, se_lift=p["se_null"],
                                cost=COST_PER_USER, n_sims=20_000, seed=7)
tc = sim["true_cpa"]

fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12, subplot_titles=(
    "What the test measures: the LIFT (well-behaved)",
    "What the client sees: the COST PER CONVERSION (skewed)"))
fig.add_trace(go.Histogram(x=sim["lift_draws"] * 1000, nbinsx=60,
                           marker_color=PALETTE["Search"], showlegend=False), row=1, col=1)
fig.add_vline(x=true_lift * 1000, line=dict(color=INK, width=2, dash="dash"),
              annotation_text="truth", row=1, col=1)
show = sim["cpa_draws"][(sim["cpa_draws"] > 0) & (sim["cpa_draws"] < 3 * tc)]
beyond = 100 * (1 - len(show) / len(sim["cpa_draws"]))
fig.add_trace(go.Histogram(x=show, nbinsx=60, marker_color=BAD, showlegend=False),
              row=1, col=2)
fig.add_vline(x=tc, line=dict(color=INK, width=2, dash="dash"),
              annotation_text=f"truth ${tc:.0f}", row=1, col=2)
fig.add_vline(x=sim["median"], line=dict(color=GOOD, width=2),
              annotation_text="median", annotation_position="bottom right", row=1, col=2)
fig.add_vline(x=sim["mean"], line=dict(color=GOLD, width=2),
              annotation_text="mean (dragged up)", row=1, col=2)
fig.add_annotation(x=3 * tc, y=0, xref="x2", yref="paper", ax=-40, ay=-30,
                   text=f"{beyond:.0f}% of readouts land beyond this axis →",
                   font=dict(size=11, color=BAD))
fig.update_xaxes(title_text="measured lift (per 1,000 users)", row=1, col=1)
fig.update_xaxes(title_text="implied cost per conversion ($)", row=1, col=2)
style(fig, height=380,
      title="The SAME 20,000 simulated readouts, before and after the division")
fig.show()

print(f"true CPA: ${tc:.0f}    median readout: ${sim['median']:.0f}    "
      f"mean readout: ${sim['mean']:.0f}  (+{100*(sim['mean_over_true']-1):.0f}% overshoot)")
print(f"skewness of the CPA readout distribution: {sim['skewness']:.0f}  "
      f"(a symmetric bell curve would be ~0)")
print(f"P(readout says MORE THAN DOUBLE the true cost): {sim['p_over_2x_true']:.1%}")
print(f"P(no measurable conversions at all — CPA 'infinite'): {sim['frac_no_positive_lift']:.1%}")
"""),
    md(r"""
The bell curve survives the test; it does **not** survive the division. And
because the tail only points one way, every summary a client instinctively
reaches for — the mean, the ± error bar — is biased in the *flattering-to-panic*
direction. Here is what that does to the error bars on a single unlucky (but
perfectly plausible) readout, and to how often each interval style actually
contains the truth:
"""),
    code(r"""
from mmm_framework.planning import cpa_interval

# one comfortable readout and one unlucky-but-plausible one
lucky = cpa_interval(lift=true_lift, se_lift=p["se_null"], cost=COST_PER_USER)
unlucky = cpa_interval(lift=0.6 * p["mde_abs"], se_lift=p["se_null"], cost=COST_PER_USER)

fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.14,
    subplot_titles=("Two interval styles on one unlucky readout",
                    "How often does each interval contain the truth?"))
y_naive, y_honest = 1.0, 0.0
# naive symmetric: dips below zero
fig.add_trace(go.Scatter(x=[unlucky["naive_lo"], unlucky["naive_hi"]], y=[y_naive] * 2,
    mode="lines", line=dict(color=BAD, width=10), opacity=0.5, name="naive ± flip"), row=1, col=1)
fig.add_trace(go.Scatter(x=[unlucky["cpa"]], y=[y_naive], mode="markers",
    marker=dict(color=BAD, size=13), showlegend=False), row=1, col=1)
# honest: lower bound + unbounded above
xmax = max(unlucky["naive_hi"], unlucky["cpa"]) * 1.6
fig.add_trace(go.Scatter(x=[unlucky["lo"], xmax], y=[y_honest] * 2,
    mode="lines", line=dict(color=GOOD, width=10), opacity=0.5,
    name="honest (inverted bound)"), row=1, col=1)
fig.add_trace(go.Scatter(x=[unlucky["cpa"]], y=[y_honest], mode="markers",
    marker=dict(color=GOOD, size=13), showlegend=False), row=1, col=1)
fig.add_annotation(x=xmax, y=y_honest, xref="x1", yref="y1", ax=-8, ay=-26,
    text="unbounded → the data cannot rule out ANY cost", font=dict(size=11, color=GOOD))
fig.add_vline(x=0, line=dict(color=INK, width=1.5, dash="dot"), row=1, col=1)
fig.add_annotation(x=unlucky["naive_lo"], y=y_naive, xref="x1", yref="y1", ax=10, ay=-28,
    text="a NEGATIVE cost per conversion?", font=dict(size=11, color=BAD))
fig.update_yaxes(tickmode="array", tickvals=[y_naive, y_honest],
                 ticktext=["naive ± flip", "honest"], range=[-0.7, 1.7], row=1, col=1)
fig.update_xaxes(title_text="cost per conversion ($)", row=1, col=1)
# coverage
fig.add_trace(go.Bar(x=["naive ± flip", "honest (inverted bound)"],
    y=[sim["coverage_naive"], sim["coverage_inverted"]], marker_color=[BAD, GOOD],
    text=[f"{sim['coverage_naive']:.1%}", f"{sim['coverage_inverted']:.1%}"],
    textposition="outside", showlegend=False), row=1, col=2)
fig.add_hline(y=sim["nominal"], line=dict(color=INK, width=1.5, dash="dot"),
              annotation_text="promised 95%", row=1, col=2)
fig.update_yaxes(tickformat=".0%", range=[0.85, 1.0], row=1, col=2)
style(fig, height=400, title="The symmetric flip lies twice: impossible values AND broken promises",
      legend=dict(orientation="h", y=-0.25))
fig.show()

print(f"unlucky readout (measured lift = 60% of the MDE):")
print(f"  naive ± flip:  ${unlucky['naive_lo']:.0f} .. ${unlucky['naive_hi']:.0f}   "
      "<- a negative lower bound is a nonsense number")
print(f"  honest:        ${unlucky['lo']:.0f} .. unbounded   ({unlucky['status']})")
print(f"comfortable readout: honest interval ${lucky['lo']:.0f} .. ${lucky['hi']:.0f} "
      f"around ${lucky['cpa']:.0f} — note the longer expensive arm ({lucky['status']})")
"""),
    md(r"""
So what *should* go on the planning slide? Two numbers, both built from the
detectable-lift **bound** rather than any flipped estimate:

- the design's **maximum detectable cost per conversion** — the worst CPA the
  test can still tell apart from "the media did nothing", and
- the **power to certify a CPA target** — the chance the test comes back with
  its (asymmetric) upper bound below the number the client cares about. Because
  the expensive arm of the interval is long, *certifying* a CPA is much harder
  than merely estimating one — the curve below is the honest version of that
  conversation.
"""),
    code(r"""
from mmm_framework.planning import cpa_power
from mmm_framework.planning.methods import ghost_ads_users_for_cpa

print(f"this design's MAX detectable CPA: ${p['max_detectable_cpa']:.0f}  "
      f"(= cost/user ${COST_PER_USER:.02f} ÷ MDE lift {p['mde_abs']:.4f}, {p['cpa_basis']} basis)")
for target in (20.0, 35.0):
    n = ghost_ads_users_for_cpa(design, target)
    print(f"users so that a true CPA of ${target:.0f} is detectable: {n:,}")
print(f"\npower to come away with a BOUNDED CPA at all: "
      f"{cpa_power(COST_PER_USER, p['se_null'], true_lift):.0%}")
for target in (40.0, 60.0, 100.0):
    pw = cpa_power(COST_PER_USER, p["se_null"], true_lift, target_cpa=target)
    print(f"power to CERTIFY 'CPA ≤ ${target:.0f}' (truth ${tc:.0f}): {pw:.0%}")

reach = np.array([50_000, 100_000, 200_000, 400_000, 800_000, 1_600_000, 3_200_000])
cap = [ghost_ads_power(GhostAdsDesign(users_reached=int(n), baseline_rate=0.021,
                                      exposure_rate=0.65, cost_per_user=0.05)
                       )["max_detectable_cpa"] for n in reach]
targets = np.linspace(tc * 1.05, tc * 5, 60)
certify = [cpa_power(COST_PER_USER, p["se_null"], true_lift, target_cpa=float(t))
           for t in targets]

fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.13, subplot_titles=(
    "Reach buys a worse-CPA detection ceiling",
    f"Power to certify 'CPA ≤ target' (truth ${tc:.0f})"))
fig.add_trace(go.Scatter(x=reach, y=cap, mode="lines+markers",
    line=dict(color=PALETTE["Search"], width=3), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=[design.users_reached], y=[p["max_detectable_cpa"]],
    mode="markers", marker=dict(size=15, symbol="star", color=GOLD,
    line=dict(width=1.5, color=INK)), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=targets, y=certify, mode="lines",
    line=dict(color=GOOD, width=3), showlegend=False), row=1, col=2)
fig.add_hline(y=0.8, line=dict(color=INK, width=1.5, dash="dot"),
              annotation_text="80% bar", row=1, col=2)
fig.add_vline(x=tc, line=dict(color=INK, width=1.5, dash="dash"),
              annotation_text="true CPA", row=1, col=2)
fig.update_xaxes(title_text="users reached", type="log", row=1, col=1)
fig.update_yaxes(title_text="max detectable CPA ($)", row=1, col=1)
fig.update_xaxes(title_text="CPA target to certify ($)", row=1, col=2)
fig.update_yaxes(title_text="power", tickformat=".0%", row=1, col=2)
style(fig, height=380,
      title="Planning on the CPA scale — built from the detectable-lift bound, never the flipped estimate")
fig.show()
print("\nNote how far right of the TRUE CPA the certification curve crosses 80%:")
print("the long expensive arm of the interval is what the client is really paying to shorten.")
"""),
    md(r"""
## 5 · Switchbacks — time-randomized on/off blocks

No geo split, no user-level randomization? Randomize **time**. Two things make
naive switchback power dishonest, and the design handles both:

- **Carryover** — a block shorter than the adstock memory smears treatment into
  control blocks; the block length respects the washout and the analysis plan
  drops a burn-in at each boundary.
- **Autocorrelation** — serially-correlated weeks mean fewer effective
  observations; the design effect uses the AR(1) coefficient of the *detrended
  levels* (first-differencing a smooth series flips the correlation negative
  and silently zeroes the correction).
"""),
    code(r"""
from mmm_framework.planning.methods import switchback_design

sb = switchback_design(NAT_CSV, KPI, "TV", duration=12, amplitude_pct=50.0)
pw = sb["switchback_power"]
ar1 = sb["ar1"]
print(f"block length: {sb['block_weeks']}w  (carryover {sb['carryover_weeks']}w, "
      f"{sb['carryover_source']})   burn-in {sb['burn_in_weeks']}w/block   "
      f"{sb['n_blocks']} blocks, {sb['n_switches']} switches")
print(f"AR(1) rho (detrended levels): {ar1['rho']:.2f}  ->  design effect {ar1['deff']:.2f}")
print(f"SE iid: {pw['se_iid']:.1f}   SE honest: {pw['se_honest']:.1f}   "
      f"MDE honest: {pw['mde_honest']:.1f} {pw['mde_units']}")

sched = pd.DataFrame(sb["schedule"])
fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.14,
                    subplot_titles=("Randomized block schedule", "The honesty tax"))
fig.add_trace(go.Bar(x=sched["week_offset"], y=sched["multiplier"],
                     marker_color=PALETTE["TV"], showlegend=False), row=1, col=1)
fig.add_hline(y=1.0, line=dict(color=INK, width=1, dash="dot"), row=1, col=1)
fig.add_trace(go.Bar(x=["iid<br>(naive)", "AR(1)-honest"], y=[pw["mde_iid"], pw["mde_honest"]],
                     marker_color=[MUTED, PALETTE["TV"]],
                     text=[f"{pw['mde_iid']:.0f}", f"{pw['mde_honest']:.0f}"],
                     textposition="outside", showlegend=False), row=1, col=2)
fig.update_xaxes(title_text="week", row=1, col=1)
fig.update_yaxes(title_text="spend x BAU", row=1, col=1)
fig.update_yaxes(title_text="MDE (KPI/week)", row=1, col=2)
style(fig, height=360, title="Switchback: carryover-aware blocks + the AR(1) design effect")
fig.show()
assert pw["se_honest"] >= pw["se_iid"]   # autocorrelation can only cost you
"""),
    md(r"""
## 6 · The methodology leaderboard — audition estimators on your own history

Analytic power assumes away the mess in real panels. The leaderboard runs every
estimator the data supports through **A/A** (no effect — does its 5% test
really reject 5%?) and **A/B** (inject a known lift — what does it actually
detect?) *on your own history*, then ranks by **validity → power → cost**. An
estimator whose naive test fires on noise gets a block-calibrated critical
value before it is allowed to compete.
"""),
    code(r"""
from mmm_framework.planning import methodology_leaderboard

board = methodology_leaderboard(GEO_CSV, KPI, "TV", duration=8,
                                max_aa_windows=40, max_ab_windows=24, seed=42)
lb = pd.DataFrame(board["methodologies"])[
    ["key", "valid", "fpr", "fpr_at_crit", "empirical_mde_roas", "power_at_expected_effect"]
]
print(lb.round(3).to_string(index=False))
print(f"\nrecommended method on this history: {board['chosen_key']}")

fig = go.Figure()
fig.add_trace(go.Bar(name="naive 5% rule", x=lb["key"], y=lb["fpr"], marker_color=BAD))
fig.add_trace(go.Bar(name="block-calibrated", x=lb["key"], y=lb["fpr_at_crit"], marker_color=GOOD))
fig.add_hline(y=0.05, line=dict(color=INK, width=1.5, dash="dot"),
              annotation_text="honest 5% size")
style(fig, height=380, title="A/A false-positive rate by estimator — validity before power",
      barmode="group", legend=dict(orientation="h", y=-0.25))
fig.update_layout(yaxis_tickformat=".0%", yaxis_title="A/A false-positive rate")
fig.show()
"""),
    md(r"""
## 7 · Fit the MMM — the model that anchors everything downstream

Everything so far ran **pre-fit** (pure pandas + numpy). The rest of the
playbook — expected effects, EIG/EVOI, economics, the net-value optimizer —
is anchored on a fitted model. A short NUTS run is enough for a demo.
"""),
    code(r"""
from mmm_framework.agents.fitting import build_and_fit

spec = {
    "kpi": KPI, "kpi_level": "geo",
    "media_channels": [{"name": n} for n in CHANNELS],
    "control_variables": [],
    "inference": {"draws": 150, "tune": 150, "chains": 2, "random_seed": 0},
    "seasonality": {"yearly": 2},
    "trend": {"type": "linear"},
}
model, results, _ = build_and_fit(spec, GEO_CSV)
print(f"fitted: {len(model.channel_names)} channels, {model.n_geos} geos, "
      f"{model.n_periods} periods  (approximate={results.approximate})")
"""),
    md(r"""
## 8 · Which channel deserves a test? — the EIG/EVOI priority grid

Two axes separate "wide posterior" from "worth testing":

- **EIG** — what a test of achievable precision would *teach*:
  $\mathrm{EIG} = \tfrac12 \ln(1 + \sigma_k^2/\sigma_{\text{exp}}^2)$ nats.
- **EVOI** — what that learning is *worth to the budget decision*
  (preposterior Monte-Carlo: simulate outcomes, reweight, re-optimize), capped
  at EVPI.

The quadrants name the verdicts: `test_now` (uncertain **and** budget-critical),
`learn_cheaply`, `monitor`, `deprioritize`. A channel can have huge EIG and
near-zero EVOI — you'd learn a lot, but the budget wouldn't move.
"""),
    code(r"""
from mmm_framework.planning import compute_experiment_priorities

grid, portfolio = compute_experiment_priorities(model, max_draws=120, random_seed=42)
gdf = pd.DataFrame([g.to_dict() for g in grid])[
    ["channel", "spend_share", "roi_mean", "roi_sd", "sigma_exp", "eig", "evoi", "quadrant", "priority"]
]
print(gdf.round(3).to_string(index=False))
print(f"\nportfolio EVPI (value of PERFECT information): {portfolio['evpi']:,.1f} KPI-units")

fig = go.Figure()
for _, r in gdf.iterrows():
    fig.add_trace(go.Scatter(
        x=[r["eig"]], y=[r["evoi"]], mode="markers+text", text=[r["channel"]],
        textposition="top center", name=r["channel"],
        marker=dict(size=14 + 60 * r["spend_share"], color=PALETTE[r["channel"]],
                    line=dict(color="white", width=1.5)), showlegend=False))
fig.add_vline(x=portfolio["eig_threshold"], line=dict(color=MUTED, width=1, dash="dot"))
fig.add_hline(y=portfolio["evoi_threshold"], line=dict(color=MUTED, width=1, dash="dot"))
style(fig, height=420, title="EIG x EVOI — top-right (test_now) earns the experiment; bubble = spend share")
fig.update_layout(xaxis_title="EIG (nats) — what the test teaches",
                  yaxis_title="EVOI (KPI units) — what the learning is worth")
fig.show()
top = gdf.iloc[0]["channel"]
print(f"top priority: {top} ({gdf.iloc[0]['quadrant']})")
"""),
    md(r"""
## 9 · Non-geo designs compete on the same scale — the `sigma_exp` bridge

Nothing in EIG/EVOI is geographic. Both consume exactly one fact about a
candidate design: the standard error it would achieve **on the ROI scale**.
Every planner above emits one:

| design | its $\sigma_{\text{exp}}$ |
|---|---|
| geo holdout / scaling | power-curve MDE at the chosen duration ÷ 2.8 |
| ghost ads | lift MDE × value per conversion ÷ test spend ÷ 2.8 |
| switchback | `switchback_power`'s honest SE, bridged by weekly spend |
| flighting | the design-matrix `se_roas` / `se_mroas` |

So a user-level RCT on Search competes against a geo holdout on TV *on one
scale* — the experiment-selection problem stops caring what kind of experiment
it is.
"""),
    code(r"""
# Bridge the ghost-ads design to the ROI scale for Search:
# MDE in conversions x $/conversion / media cost -> an ROAS-scale MDE -> /2.8 -> sigma_exp.
ghost_roas_mde = p["incremental_value_at_mde"] / p["media_cost"]
ghost_sigma = ghost_roas_mde / 2.8
print(f"ghost-ads bridge for Search: ROAS-scale MDE {ghost_roas_mde:.2f} -> sigma_exp {ghost_sigma:.3f}")

grid2, _ = compute_experiment_priorities(
    model, max_draws=120, random_seed=42,
    sigma_exp_overrides={"Search": ghost_sigma},
)
g2 = pd.DataFrame([g.to_dict() for g in grid2])[["channel", "sigma_exp", "eig", "evoi", "quadrant"]]
before = gdf.set_index("channel").loc["Search"]
after = g2.set_index("channel").loc["Search"]
print(g2.round(3).to_string(index=False))
print(f"\nSearch under its own achievable design: sigma_exp {before['sigma_exp']:.3f} -> "
      f"{after['sigma_exp']:.3f},  EIG {before['eig']:.2f} -> {after['eig']:.2f} nats")
"""),
    md(r"""
## 10 · What does the test cost, and what is it worth? — net test value

EVOI prices the upside; a real test also has a downside — the profit given up
while it runs. `compute_experiment_net_value` nets the two into one figure:

$$\text{net value} = \underbrace{\min(\mathrm{EVOI}, \mathrm{EVPI}) \cdot m \cdot \bar d}_{\text{decayed reallocation gain}} + \underbrace{\mathbb{E}[m\,\Delta\mathrm{KPI} - \Delta\mathrm{spend}]}_{\text{signed profit during the test}}$$

with a **decay haircut** $\bar d$ (information has a half-life; EVOI values a
posterior that stays sharp forever) and an **EVPI cap**. The signed convention
keeps a money-saving holdout coherent: saved spend can outweigh forgone margin,
making the "loss" negative and the break-even horizon zero.
"""),
    code(r"""
from mmm_framework.planning import compute_experiment_net_value, compute_opportunity_cost
from mmm_framework.planning.eig import channel_half_life

MARGIN = 0.5
row = gdf.set_index("channel").loc[top]
test_design = design_experiment(GEO_CSV, KPI, top, design="holdout", n_pairs=4, duration=8)

oc = compute_opportunity_cost(model, test_design, margin_per_kpi=MARGIN,
                              evoi_kpi_units=float(row["evoi"]),
                              max_draws=120, return_draws=True)
nv = compute_experiment_net_value(
    channel=top,
    evoi_kpi_units=float(row["evoi"]),
    evpi_kpi_units=float(portfolio["evpi"]),
    margin_per_kpi=MARGIN,
    response_horizon_weeks=26,
    half_life_weeks=channel_half_life(top),
    model_anchored=True,
    opportunity_cost_result=oc,
)
print(f"reallocation gain (decayed, capped): ${nv.reallocation_gain:,.0f}   "
      f"(decay factor {nv.decay_factor:.2f}, half-life {nv.half_life_weeks:.0f}w)")
print(f"signed profit during the test:       ${nv.net_profit_during_test:,.0f}")
print(f"NET VALUE of running this test:      ${nv.net_value:,.0f}")
print(f"P(net > 0): {nv.prob_net_positive:.0%}    "
      f"break-even horizon: {nv.breakeven_horizon_weeks} weeks    basis: {nv.basis}")

parts = [("gain from the learning", nv.reallocation_gain),
         ("profit impact during test", nv.net_profit_during_test),
         ("net value", nv.net_value)]
fig = go.Figure(go.Bar(x=[a for a, _ in parts], y=[b for _, b in parts],
                       marker_color=[GOOD if v >= 0 else BAD for _, v in parts],
                       text=[f"${v:,.0f}" for _, v in parts], textposition="outside",
                       showlegend=False))
fig.add_hline(y=0, line=dict(color=INK, width=1))
style(fig, height=360, title=f"Is the {top} test worth running? gain vs loss, in dollars")
fig.show()
"""),
    md(r"""
## 11 · **New** — the Pareto optimizer prices every design in net dollars

Within one channel there is still a design space: holdout vs scaling,
footprint, intensity, duration. `suggest_experiment` sweeps a bounded grid on
four lower-is-better objectives — MDE, power shortfall, cost, duration — and
returns the non-dominated **Pareto front** plus the powered "knee".

**What's new (2026-07-19):** with a margin, the cost axis is upgraded to the
**net value of testing** per candidate. The expensive part — a preposterior
Monte-Carlo EVOI per design — is priced by a **calibrated Gaussian surrogate**
(Raiffa–Schlaifer preposterior form):

$$\mathrm{EVOI}(\sigma) \approx k \cdot s(\sigma)\,\Psi\!\big(\delta/s(\sigma)\big),
\qquad s(\sigma) = \tau\sqrt{\tau^2/(\tau^2+\sigma^2)}$$

with $(k, \delta)$ fitted to **two** anchored MC EVOIs placed at the extremes
of the grid's design precisions — every candidate interpolates inside the
calibrated bracket. The front then reads the way a CFO wants it read: designs
that are precise, powered, fast, *and expected to create money net of their own
cost*.
"""),
    code(r"""
from mmm_framework.planning import suggest_experiment

out = suggest_experiment(
    model, GEO_CSV, KPI, top,
    margin=MARGIN,
    duration_min=6, duration_max=16,
    intensity_min=-100, intensity_max=100,
    max_draws=60, random_seed=42,
)
assert out["net_value_axis"] is True, "margin known -> the net-value axis engages"

anch = out["evoi_anchor"]
print(f"EVOI surrogate: {len(anch['anchors'])} MC anchors at sigma = "
      f"{[round(a[0], 3) for a in anch['anchors']]}  ->  k={anch['k']:.1f}, "
      f"delta={anch['delta']:.4f}, tau={anch['tau']:.4f}, EVPI={anch['evpi']:.1f}")
print(f"cost axis: {out['tradeoff_label']}\n")

cands = pd.DataFrame(out["candidates"])
front = cands[cands["on_pareto"]]
cols = ["mode", "footprint", "intensity_pct", "duration", "mde_roas", "power",
        "evoi_kpi", "reallocation_gain", "net_value", "powered", "is_recommended"]
print("the Pareto front (non-dominated designs), priced in net dollars:")
print(front[cols].round(2).to_string(index=False))

rec = out["recommended"]
dom = cands[~cands["on_pareto"] & ~cands["is_recommended"]]
fr = cands[cands["on_pareto"] & ~cands["is_recommended"]]
fig = go.Figure()
fig.add_trace(go.Scatter(x=dom["mde_roas"], y=dom["net_value"], mode="markers",
    name="dominated", marker=dict(size=9, color=MUTED, opacity=0.45)))
fig.add_trace(go.Scatter(x=fr["mde_roas"], y=fr["net_value"], mode="markers",
    name="Pareto front", marker=dict(size=14, color=fr["duration"], colorscale="YlGnBu",
    reversescale=True, colorbar=dict(title="weeks", thickness=10, len=0.6),
    line=dict(width=2, color=INK))))
fig.add_trace(go.Scatter(x=[rec["mde_roas"]], y=[rec["net_value"]], mode="markers",
    name="recommended", marker=dict(size=20, symbol="star", color=GOLD,
    line=dict(width=1.5, color=INK))))
fig.add_hline(y=0, line=dict(color=INK, width=1, dash="dot"),
              annotation_text="test pays for itself above this line")
style(fig, height=430,
      title=f"{top}: the design space priced in dollars — upper-left is best (precise AND net-positive)",
      legend=dict(orientation="h", y=-0.22))
fig.update_layout(xaxis_title="MDE (ROAS) — lower is more precise",
                  yaxis_title="net value of testing ($)")
fig.show()
print(f"\nrecommended: {rec['mode']} / {rec['footprint']} / {rec['intensity_pct']:+.0f}% / "
      f"{rec['duration']}w — net value ${rec['net_value']:,.0f}, "
      f"cool-down {out['cooldown']['cooldown_weeks']}w before a clean re-read")
"""),
    md(r"""
## 12 · Trust but verify — the surrogate against the exact preposterior MC

The net-value axis leans on the surrogate, so check it against ground truth:
run the *exact* preposterior Monte-Carlo EVOI at several design precisions and
overlay the fitted curve. Anchored at the extremes, every candidate is an
**interpolation** — the regime where the surrogate stays within a few tens of
percent of the MC, which is what the Pareto axis needs: it has to *rank*
designs correctly, not reproduce each EVOI to the decimal. (Extrapolating far
beyond the weak anchor under-estimates — which is exactly why the optimizer
anchors at the extremes of its own grid.)
"""),
    code(r"""
from mmm_framework.planning import (
    compute_evoi_for_channel, compute_evpi, compute_response_curves,
    fit_evoi_surrogate, optimize_budget,
)

curves = compute_response_curves(model, max_draws=60, random_seed=42)
ci = curves.channel_names.index(top)
g1 = int(np.argmin(np.abs(curves.multipliers - 1.0)))
roi_draws = curves.contributions[:, ci, g1] / max(float(curves.base_spend[ci]), 1e-12)
tau = float(roi_draws.std())

opt = optimize_budget(curves=curves, random_seed=42)
port = compute_evpi(curves, total_budget=opt.total_budget,
                    per_draw_alloc=opt.per_draw_alloc, optimal_alloc=opt.optimal_alloc)

rng = np.random.default_rng(42)
od = (rng.integers(0, len(roi_draws), size=48), rng.standard_normal(48))
sig_lo, sig_hi = tau / 2, 3 * tau
anchors = [(s, compute_evoi_for_channel(curves, ci, roi_draws, float(s),
                                        optimal_alloc=opt.optimal_alloc,
                                        total_budget=opt.total_budget,
                                        outcome_draws=od))
           for s in (sig_lo, sig_hi)]
sur = fit_evoi_surrogate(tau, anchors)

sig_grid = np.linspace(sig_lo, sig_hi, 9)
mc = [compute_evoi_for_channel(curves, ci, roi_draws, float(s),
                               optimal_alloc=opt.optimal_alloc,
                               total_budget=opt.total_budget, outcome_draws=od)
      for s in sig_grid]
sg = np.linspace(sig_lo * 0.8, sig_hi * 1.1, 80)
# NB: the raw (uncapped) surrogate vs the raw MC — the fidelity question.
# In production the optimizer additionally caps at EVPI (a conservative floor
# that can clip everything to zero when the portfolio decision is already firm).
fig = go.Figure()
fig.add_trace(go.Scatter(x=sg, y=[sur(float(s)) for s in sg],
    mode="lines", name="Gaussian surrogate (2 anchors)",
    line=dict(color=PALETTE[top], width=3)))
fig.add_trace(go.Scatter(x=sig_grid, y=mc, mode="markers", name="exact preposterior MC",
    marker=dict(size=10, color=INK, symbol="circle-open", line=dict(width=2))))
fig.add_trace(go.Scatter(x=[a[0] for a in anchors], y=[a[1] for a in anchors],
    mode="markers", name="MC anchors", marker=dict(size=14, color=GOLD,
    line=dict(width=1.5, color=INK))))
style(fig, height=400,
      title=f"{top}: surrogate vs exact EVOI — two MC evaluations price the whole design grid",
      legend=dict(orientation="h", y=-0.22))
fig.update_layout(xaxis_title="design precision sigma_exp (ROI scale)",
                  yaxis_title="EVOI (KPI units)")
fig.show()

interior = [(s, m, sur(float(s))) for s, m in zip(sig_grid, mc) if m > 0]
ratios = [su / m for _, m, su in interior]
print(f"surrogate/MC ratio across the bracket: "
      f"min {min(ratios):.2f}, median {np.median(ratios):.2f}, max {max(ratios):.2f}")
print(f"portfolio EVPI (the production cap on any gain): {port.evpi:,.1f} KPI-units")
"""),
    md(r"""
## 13 · Close the loop — the readout calibrates the next fit

The test ran (here: simulated). Its result is direct evidence about the tested
channel that the next fit must not ignore — folded in as a **likelihood term on
the model's own estimand**: the measured incremental ROAS, with its standard
error, constrains the in-graph ROAS of that channel over that window.

Two outcomes are possible, and both are informative. If the readout **agrees**
with the observational fit, the posterior sharpens where the test measured. If
it **disagrees** — as the simulated readout below deliberately does — the
posterior is *pulled toward the measured value*, and the residual tension is a
**model-misspecification signal** that surfaces in diagnostics instead of being
silently overwritten. That visibility is the point of the likelihood route over
hand-tightening a prior: a hand-set prior would have hidden the conflict.
"""),
    code(r"""
from mmm_framework.calibration import ExperimentEstimand, ExperimentMeasurement

# ROI posterior BEFORE calibration (per-draw ROI at current spend)
spend_top = float(model.X_media_raw[:, ci].sum())
contrib_pre = model.sample_channel_contributions(max_draws=200, random_seed=7)
roi_pre = contrib_pre[:, :, ci].sum(axis=1) / spend_top

# the (simulated) readout: the holdout measured incremental ROAS = 1.2 +/- 0.25
# — deliberately HIGHER than the observational fit believes, so we can watch
# the likelihood pull the posterior and expose the tension.
exp = ExperimentMeasurement(
    channel=top,
    test_period=(60, 67),                 # integer period indices into the panel
    value=1.2, se=0.25,
    estimand=ExperimentEstimand.ROAS,
)
model.add_experiment_calibration([exp])
results_cal = model.fit(draws=150, tune=150, chains=2, random_seed=0)

contrib_post = model.sample_channel_contributions(max_draws=200, random_seed=7)
roi_post = contrib_post[:, :, ci].sum(axis=1) / spend_top

fig = go.Figure()
for name, draws, color in [("before calibration", roi_pre, MUTED),
                           ("after calibration", roi_post, PALETTE[top])]:
    fig.add_trace(go.Histogram(x=draws, name=name, opacity=0.6, nbinsx=40,
                               marker_color=color, histnorm="probability density"))
fig.add_vline(x=1.2, line=dict(color=INK, width=2, dash="dash"),
              annotation_text="experiment readout (1.2)")
style(fig, height=380, barmode="overlay",
      title=f"{top} ROI posterior: the readout's likelihood pulls it toward the measured value",
      legend=dict(orientation="h", y=-0.22))
fig.update_layout(xaxis_title="ROI (window total)", yaxis_title="density")
fig.show()
shift_se = abs(roi_post.mean() - roi_pre.mean()) / max(roi_pre.std(), 1e-9)
print(f"ROI posterior mean: {roi_pre.mean():.2f} -> {roi_post.mean():.2f} "
      f"(readout said 1.2 +/- 0.25) — a {shift_se:.1f}-sd shift toward the experiment")
print(f"ROI posterior sd:   {roi_pre.std():.3f} -> {roi_post.std():.3f}")
print("The posterior lands BETWEEN the observational fit and the readout, and the")
print("unresolved gap is a misspecification flag to chase (missing confounder,")
print("wrong saturation), not a number to overwrite.")
"""),
    md(r"""
## Wrap-up — the playbook in one breath

1. **Enumerate** what your data supports (`list_methods` / `methods_for_data` /
   `design_options`) — geo panels unlock SCM/TBR/GBR/DiD; no panel still leaves
   ghost ads, switchbacks and flighting.
2. **Plan** with honest power math — matched pairs and power curves for geo,
   ITT dilution and rare-event flags for ghost ads, carryover-aware blocks and
   AR(1) design effects for switchbacks.
3. **Translate power to the client's metric** — the **maximum detectable cost
   per conversion** (`planning.cpa`): invert the detectable-lift *bound*, never
   the point estimate — the CPA readout is right-skewed, its mean overshoots,
   the symmetric ± flip under-covers (and can go negative), and a weak readout's
   honest interval is *unbounded above*.
4. **Audition the estimator** on your own history (`methodology_leaderboard`) —
   validity (A/A) before power (A/B).
5. **Pick the channel** with EIG × EVOI — and bring *non-geo* designs onto the
   same scale through the `sigma_exp` bridge.
6. **Price the test** — opportunity cost, decayed/EVPI-capped gain, one net
   dollar figure with `P(net > 0)` and a break-even horizon.
7. **Optimize the design** — the Pareto front now ranks candidates by the **net
   value of testing**, priced by the two-anchor Gaussian EVOI surrogate.
8. **Calibrate** — the readout becomes a likelihood term; the posterior moves
   toward what the test measured.

**Further reading:** [`docs/experiment-playbook.html`](../../docs/experiment-playbook.html)
(the reference version of this playbook), the `lifecycle_00–06` notebook series
(the same loop run as an ongoing measurement program), and
`technical-docs/experiment-net-economics.md` (the net-value spec, including the
surrogate's derivation and its verified accuracy envelope).
"""),
]


def main() -> None:
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    out = "demos/experiment_planning_playbook.ipynb"
    with open(out, "w") as f:
        nbformat.write(nb, f)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    print(f"wrote {out} with {len(nb.cells)} cells ({n_code} code)")


if __name__ == "__main__":
    main()
