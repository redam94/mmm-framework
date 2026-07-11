"""Author causal_04_latent_confounders.ipynb — Notebook 4 of 11.

    uv run --with nbformat python builders/build_causal_04_latent_confounders.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        causal/causal_04_latent_confounders.ipynb --ExecutePreprocessor.timeout=3600 \
        --ExecutePreprocessor.kernel_name=python3

Rung 4 — structural learning II: the economic-health world. A latent
macro factor confounds spend and sales and is measured only through four noisy
indicators. Three rungs of adjustment, graded: ignore it / use the indicators
as controls / model the measurement jointly (LatentFactorMMM). The honest
punchline: B and C tie on media point error here — C's win is recovering the
MEASUREMENT MODEL itself (loadings incl. a planted negative sign, factor corr
0.98), and the shared residual floor is what experiments exist to break.
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
import sys
for _p in ("../builders", "builders", os.path.join(os.getcwd(), "nbs", "builders")):
    if os.path.isfile(os.path.join(_p, "causal_common.py")) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.templates.default = "plotly_white"
pio.renderers.default = "notebook_connected"
pd.set_option("display.width", 140)

import logging
from loguru import logger
logger.disable("mmm_framework")
for _n in ("pymc", "pymc.sampling", "numpyro", "jax", "arviz"):
    logging.getLogger(_n).setLevel(logging.ERROR)

import causal_common as C
print(f"{C.BRAND} — {C.TAGLINE}")
"""


CELLS = [
    md(r"""
# Structural Learning II — Latent Confounders
### The common cause you can't see, but can measure

*Notebook 4 of 11 — Causal Inference in Practice.*

Patio furniture is a discretionary purchase. When the economy hums — hiring
up, confidence up, houses turning over — Veranda Home sells more *and* spends
more, because performance budgets get topped up in good quarters. **Economic
health** is a textbook confounder: it drives both sides of every spend-sales
correlation.

Nobody has a column called `economic_health` in their warehouse. What they
have is four noisy **indicators** that each reflect it partially: GDP growth,
consumer confidence, unemployment (inversely!), retail sales. This notebook
climbs three rungs on one world, grading each against the sealed key:

- **Rung A — ignore it.** The naive MMM (Price is the only control).
- **Rung B — indicators as controls.** Dump all four into the control set.
  This is what most practitioners do, and it genuinely helps — but the
  indicators are *noisy stand-ins* for the confounder, so attenuated,
  unannounced residual bias survives.
- **Rung C — model the measurement.** A joint **latent-factor MMM**
  (`LatentFactorMMM`): one latent economic-health factor, estimated *inside
  the same PyMC graph* from its four indicators, entering the sales equation
  as the de-confounder. The factor's uncertainty propagates into every media
  coefficient — no two-stage plug-in.

We'll be strict about what rung C does and doesn't buy. Spoiler: on this
world its media *point estimates* land about where rung B's do — and it
recovers something rung B cannot even express.
"""),
    code(SETUP),
    md(r"""
## 1 — The world on the desk

Sales, four channels, Price — and a macro dashboard of four indicators. The
indicators visibly co-move (they share one driver), but none of them *is* the
driver: each carries its own noise, and unemployment runs upside-down.
"""),
    code(r"""
sc = C.scenario_for("econ_naive")
C.check_truth(sc)
inds = C.world("economic_health").notes["indicators"]

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.45, 0.55],
                    vertical_spacing=0.07,
                    subplot_titles=["sales", "the macro dashboard (4 noisy indicators)"])
fig.add_trace(go.Scatter(x=sc.weeks, y=sc.y, name="Sales",
                         line=dict(color=C.INK, width=2)), row=1, col=1)
for name, arr in inds.items():
    fig.add_trace(go.Scatter(x=sc.weeks, y=arr, name=name, opacity=0.8,
                             line=dict(width=1.4)), row=2, col=1)
C.style(fig, height=560, title="What Veranda can see — sales and macro indicators")

icorr = pd.DataFrame(inds).corr().round(2)
print("indicator co-movement:\n", icorr.to_string())
"""),
    code(r"""
C.dag(
    nodes={"economic health": (0.5, 1.05),
           "gdp": (0.06, 0.62), "confidence": (0.34, 0.62),
           "unemployment": (0.66, 0.62), "retail": (0.94, 0.62),
           "spend": (0.15, 0.0), "sales": (0.85, 0.0)},
    edges=[("economic health", "gdp"), ("economic health", "confidence"),
           ("economic health", "unemployment"), ("economic health", "retail"),
           ("spend", "sales")],
    bad_edges=[("economic health", "spend"), ("economic health", "sales")],
    title="A latent confounder with four measurements (red: the back-door)",
    node_colors={"economic health": "#e8d5a3"},
    height=440,
)
"""),
    md(r"""
## 2 — Rung A: ignore it

The naive fit. One methodological note: on this world we fit **without a
linear trend** for all three rungs, matching the framework's own recovery
test — the world's only slow structure *is* the economic factor, so the
comparison isolates de-confounding cleanly. (We'll show below what a trend
term accidentally does.)
"""),
    code(r"""
fitA = C.fit_world("econ_naive", trend="none")
gA = fitA["grade"]
print(gA[["true", "est", "rel_err", "covered"]].round(2).to_string())
C.truth_forest(gA, title="Rung A — the naive read: Search credited ~9x its true effect",
               height=380)
"""),
    code(r"""
# The chasers (Search, Social spend track the economy) absorb the boom-time
# correlation. Search's over-credit is spectacular; Display — a REAL channel —
# is crushed to near-zero to compensate.
assert gA.loc["Search", "rel_err"] > 3.0
assert gA.loc["Display", "rel_err"] < -0.5
assert not gA.loc["Search", "covered"]

# Aside: the same naive model WITH a linear trend (the series default) is
# less wrong — the trend soaks up the economy's growth component. An
# accidental, partial de-confounder you shouldn't count on.
gA_trend = C.fit_world("econ_naive", verbose=False)["grade"]
print(f"Search over-credit: {gA.loc['Search','rel_err']:+.0%} without trend, "
      f"{gA_trend.loc['Search','rel_err']:+.0%} with a linear trend.")
assert gA.loc["Search", "rel_err"] > gA_trend.loc["Search", "rel_err"]
"""),
    md(r"""
## 3 — Rung B: indicators as controls

The practitioner's move: throw all four indicators into the control set
(marked `CONFOUNDER`, per rung 1's lesson). This *works*, mostly — the four
series jointly span much of the economy's variance, and the bias collapses by
roughly a factor of four.

But look closely at what's left. The indicators each carry idiosyncratic
noise, so they can never absorb *all* of the factor's variance —
**errors-in-variables attenuation**. A residual back-door stays open, the
Search estimate stays materially inflated, and nothing in the fit output
announces it.
"""),
    code(r"""
fitB = C.fit_world("econ_ind", trend="none")
gB = fitB["grade"]
print(gB[["true", "est", "rel_err", "covered"]].round(2).to_string())

for c in ("Search", "Social"):
    assert abs(gB.loc[c, "rel_err"]) < abs(gA.loc[c, "rel_err"]), c
assert gB.loc["Search", "rel_err"] > 0.8   # ...but far from clean

# Marked vs unmarked barely matters here (the indicators' signal survives
# default shrinkage) — consistent with nb01's honest aside.
gB_un = C.fit_world("econ_ind_unmarked", trend="none", verbose=False)["grade"]
print(f"\nSearch rel err — A: {gA.loc['Search','rel_err']:+.0%}, "
      f"B marked: {gB.loc['Search','rel_err']:+.0%}, "
      f"B unmarked: {gB_un.loc['Search','rel_err']:+.0%}")
"""),
    md(r"""
## 4 — Rung C: model the measurement

`LatentFactorMMM` (the framework's worked garden-model example) makes the
structure explicit: a latent AR(1) factor with a **measurement block** —
`indicator_k = loading_k x factor + noise` — estimated jointly with the MMM.
Identification is delicate and instructive:

- the realized factor is **standardized in-graph** (otherwise the AR variance
  trades off against the loadings and the sampler collapses the loadings to
  zero);
- the **first loading is constrained positive** (`gdp_growth`, the anchor) to
  pin the factor's orientation — every other loading is free to go negative,
  which is exactly what unemployment should do.

The fit is 4-chain NUTS (MAP is documented-unstable for this ~150-parameter
latent model; don't). Cached by the series.
"""),
    code(r"""
lat = C.fit_latent()
meta = lat["meta"]
assert meta["rhat_max"] < 1.1

loadings = lat["loadings"].set_index("indicator")
truth_l = meta["true_loadings"]
show = loadings[["loading", "hdi_low", "hdi_high"]].copy()
show["true"] = pd.Series(truth_l)
print(show.round(2).to_string())

fig = go.Figure()
order = list(show.index)
for i, name in enumerate(order):
    r = show.loc[name]
    fig.add_trace(go.Scatter(x=[r["hdi_low"], r["hdi_high"]], y=[i, i], mode="lines",
                             line=dict(color=C.PALETTE["demand"], width=5), opacity=0.5,
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[r["loading"]], y=[i], mode="markers",
                             marker=dict(color=C.PALETTE["demand"], size=11),
                             name="posterior loading", showlegend=(i == 0)))
    fig.add_trace(go.Scatter(x=[r["true"]], y=[i], mode="markers",
                             marker=dict(color=C.TRUTH, symbol="line-ns-open", size=18,
                                         line=dict(width=3)),
                             name="planted truth", showlegend=(i == 0)))
fig.add_vline(x=0, line_color=C.MUTED, line_dash="dot")
fig.update_yaxes(tickvals=list(range(len(order))), ticktext=order, autorange="reversed")
C.style(fig, title="The measurement model, recovered — including the negative sign",
        height=400)
"""),
    code(r"""
# Sign recovery on all four loadings — including unemployment's NEGATIVE one
# (a model that forced positive loadings would have broken this world).
assert loadings.loc["gdp_growth", "loading"] > 0
assert loadings.loc["consumer_confidence", "loading"] > 0
assert loadings.loc["retail_sales", "loading"] > 0
assert loadings.loc["unemployment", "loading"] < 0
assert meta["corr_factor"] > 0.9
print(f"corr(recovered factor, true economic health) = {meta['corr_factor']:.3f}")
"""),
    code(r"""
F = lat["factor"]
w = lat["sc"].weeks
tr = F["true"]
s = np.sign(np.corrcoef(F["mean"], tr)[0, 1])
tr_al = (tr - tr.mean()) / tr.std() * F["mean"].std() * s + F["mean"].mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=w, y=F["hi"], line=dict(width=0), showlegend=False,
                         hoverinfo="skip"))
fig.add_trace(go.Scatter(x=w, y=F["lo"], fill="tonexty", line=dict(width=0),
                         fillcolor="rgba(120,130,150,0.25)", name="90% band"))
fig.add_trace(go.Scatter(x=w, y=F["mean"], name="recovered factor",
                         line=dict(color=C.PALETTE["demand"], width=2)))
fig.add_trace(go.Scatter(x=w, y=tr_al, name="true economic health (aligned)",
                         line=dict(color=C.TRUTH, width=1.4, dash="dot")))
C.style(fig, title="The confounder, made visible — a nameable series you can monitor",
        height=420)
"""),
    md(r"""
## 5 — The ladder, graded honestly

Same estimand for all three rungs (counterfactual contribution vs the sealed
key), chaser channels averaged:
"""),
    code(r"""
gC = lat["grade"]
chasers = ["Search", "Social"]

def chaser_err(g):
    return float(np.mean([abs(g.loc[c, "rel_err"]) for c in chasers]))

ladder = pd.Series({
    "A — ignore it": chaser_err(gA),
    "B — indicators as controls": chaser_err(gB),
    "C — joint latent factor": chaser_err(gC),
}, name="mean |rel err|, chasers")
print(ladder.map("{:.0%}".format).to_string())

fig = go.Figure(go.Bar(x=ladder.index, y=ladder.values,
                       marker_color=[C.BAD, "#c9a45a", C.GOOD], opacity=0.9,
                       text=[f"{v:.0%}" for v in ladder.values],
                       textposition="outside"))
fig.update_yaxes(title="mean |relative error|, Search & Social", tickformat=".0%",
                 range=[0, ladder.max() * 1.25])
C.style(fig, title="Three rungs on the same world", height=420)
"""),
    code(r"""
# Both structural rungs collapse the naive bias ~4x. And the honest part:
# on THIS world, B and C essentially TIE on media point error — four
# indicators used as plain controls span about as much of the confounder's
# variance as the one-factor model extracts from them.
assert chaser_err(gB) < 0.5 * chaser_err(gA)
assert chaser_err(gC) < 0.5 * chaser_err(gA)
assert abs(chaser_err(gC) - chaser_err(gB)) < 0.6
print("B ≈ C on points. So why ever pay for rung C?")
"""),
    md(r"""
## 6 — What rung C actually buys

If B and C tie on point error, rung C is not an accuracy purchase — it's a
**knowledge** purchase:

1. **The measurement model is itself a graded causal claim.** C recovered the
   loadings (with the planted negative sign) and the factor's path (corr
   0.98). B's output is four regression coefficients on noisy proxies —
   nothing you could hand to an economist, reuse next year, or falsify.
2. **A nameable series to monitor.** C produces "economic health, weekly,
   with uncertainty" — a *thing* the CFO can look at, extend with new
   indicators, or compare against external indices. B produces nothing
   reusable.
3. **Honest propagation.** C's media intervals carry the factor's estimation
   uncertainty inside the same posterior. B silently treats noisy proxies as
   if they were the confounder itself.
4. **A platform.** The measurement block composes with everything the rest of
   this series does — mediators (nb03), calibration (nb06+), decay (nb09).

And one thing **neither** rung buys: both leave the *same* residual Search
bias, because it lives in the part of the confounder the indicators simply
don't measure. No observational structure can reach it. That floor — bias you
can name but not remove — is exactly the price of an experiment, which is
where the series goes next.
"""),
    code(r"""
C.write_artifact("causal_04_latent_recovery.json", dict(
    ladder={k: float(v) for k, v in ladder.items()},
    search_rel_err=dict(A=float(gA.loc["Search", "rel_err"]),
                        B=float(gB.loc["Search", "rel_err"]),
                        C=float(gC.loc["Search", "rel_err"])),
    loadings={n: dict(post=float(loadings.loc[n, "loading"]),
                      true=float(truth_l[n])) for n in loadings.index},
    factor_corr=float(meta["corr_factor"]),
    rhat_max=float(meta["rhat_max"]),
))
print("artifact written: causal_04_latent_recovery.json")
"""),
    md(r"""
---
**Next — [05 · Measuring one experiment](causal_05_measuring_one_experiment.ipynb):**
down from structure, out into the field. Before an experiment can repair a
model, someone has to *read it honestly* — designs, estimators, false-positive
rates under autocorrelation, and a leaderboard that names the method your geo
data can actually support.
"""),
]


def main():
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    out = "causal/causal_04_latent_confounders.ipynb"
    with open(out, "w") as f:
        nbformat.write(nb, f)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
