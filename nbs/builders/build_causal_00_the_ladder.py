"""Author causal_00_the_ladder.ipynb — Notebook 0 of 11 (Causal Inference in Practice).

    uv run --with nbformat python builders/build_causal_00_the_ladder.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        causal/causal_00_the_ladder.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

The series opener: why a dashboard cannot tell you what caused what, why a
*better model* doesn't fix a *causal* problem, and the ladder of evidence the
next ten notebooks climb. Uses the cached `uc` fit (fast) plus zero-fit numpy
throughout; the one MMM it shows is reloaded from the shared cache.
"""

from __future__ import annotations

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


# The canonical setup cell, shared by the whole series.
SETUP = r"""
import warnings; warnings.filterwarnings("ignore")
import os
os.environ.setdefault("TQDM_DISABLE", "1")   # quiet sampling progress bars
import sys
# Make the shared module importable whether the kernel runs from nbs/causal/,
# nbs/, or the repo root.
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
# The Ladder of Evidence
### Why your dashboard can't tell you what caused what — and what can

*Notebook 0 of 11 — Causal Inference in Practice.*

**Veranda Home** sells patio furniture, planters, and garden tools across the
country. Like every brand, it has a dashboard: weekly spend on four channels,
weekly sales, three years of history. And like every brand, it keeps asking the
dashboard a question the dashboard cannot answer:

> *"What did each channel actually **cause**?"*

The dashboard shows what happened. Causation is a claim about what would have
happened **otherwise** — a counterfactual. No amount of staring at observed
history settles a counterfactual by itself; you need assumptions, structure, or
interventions. This series is about acquiring each of those honestly, in order
of strength:

1. **Adjustment** — closing back-doors with control variables (nb 01–02),
2. **Structure** — modeling the causal graph you believe in: mediators and
   latent confounders (nb 03–04),
3. **Experiments** — measuring one intervention well (nb 05), folding it into
   the model (nb 06–07),
4. **Programs** — designing the *next* experiment (nb 08) and budgeting a whole
   year of measurement (nb 09), until the loop closes (nb 10).

**What makes this series different:** every world we model is synthetic, drawn
from `mmm_framework.synth`, and ships with a **sealed answer key** — the true
causal contribution of every channel, computed from the data-generating process
itself. The analyst never sees the key while modelling. We unseal it only to
*grade* each claim. Nothing here is a vibe; every headline number is asserted
against the truth, and the grading machinery itself is verified every time it
runs (`C.check_truth`).
"""),
    code(SETUP),
    md(r"""
## 1 — The dashboard

Here are Veranda Home's three years. Spend and sales move together; every
channel's spend is positively correlated with the KPI. A dashboard reader—or a
last-click report, or a correlation matrix—would call this marketing that
works.

*(What we, the narrators, know and the analyst doesn't: this history was drawn
from the `unobserved_confounding` world. We'll pretend otherwise until the
reveal.)*
"""),
    code(r"""
sc = C.scenario_for("uc")

# Verify the series' grading machinery against this world's sealed answer key:
# the reconstructed true response must reproduce the key to float precision.
C.check_truth(sc)

fig = make_subplots(specs=[[{"secondary_y": True}]])
for c in sc.channels:
    fig.add_trace(go.Scatter(x=sc.weeks, y=sc.spend[c], name=c, stackgroup="spend",
                             mode="lines", line=dict(width=0.4, color=C.PALETTE[c])),
                  secondary_y=False)
fig.add_trace(go.Scatter(x=sc.weeks, y=sc.y, name="Sales",
                         line=dict(color=C.INK, width=2.2)), secondary_y=True)
fig.update_yaxes(title_text="weekly spend ($000s, stacked)", secondary_y=False)
fig.update_yaxes(title_text="sales (KPI units)", secondary_y=True)
C.style(fig, title="Veranda Home — the dashboard: spend and sales, three years",
        height=460)
"""),
    code(r"""
# The dashboard-level "evidence": every channel correlates with sales.
corr = pd.Series({c: float(np.corrcoef(sc.spend[c], sc.y)[0, 1]) for c in sc.channels},
                 name="corr(spend, sales)")
print(corr.round(2).to_string())
assert (corr > 0).all()   # everything "works", says the dashboard
"""),
    md(r"""
## 2 — The naive read

The simplest quantitative version of the dashboard's story: regress sales on
spend (ordinary least squares, all four channels at once) and call each slope
"the ROAS". This is what a spreadsheet, a BI tool trendline, or a junior
analyst's first model does.
"""),
    code(r"""
X = np.column_stack([np.ones(len(sc.weeks))] + [sc.spend[c] for c in sc.channels])
beta = np.linalg.lstsq(X, sc.y.to_numpy(float), rcond=None)[0]
naive = pd.Series(beta[1:], index=sc.channels, name="naive OLS 'ROAS'")
print(naive.round(2).to_string())
"""),
    md(r"""
## 3 — Unsealing the answer key

Time to grade. The world's answer key holds each channel's **true causal
ROAS** — total incremental sales caused by the channel, per dollar, computed by
zeroing that channel's spend inside the data-generating process and summing the
difference (carryover included). This is the exact counterfactual question the
CMO is asking.
"""),
    code(r"""
ratio = (naive / sc.true_roas).rename("naive ÷ true")
tbl = pd.concat([naive.round(2), sc.true_roas.round(2), ratio.round(2)], axis=1)
print(tbl.to_string())

fig = go.Figure()
fig.add_trace(go.Bar(x=sc.channels, y=[naive[c] for c in sc.channels],
                     name="naive OLS read", marker_color=[C.PALETTE[c] for c in sc.channels],
                     opacity=0.85))
fig.add_trace(go.Bar(x=sc.channels, y=[sc.true_roas[c] for c in sc.channels],
                     name="true causal ROAS (sealed key)", marker_color=C.TRUTH,
                     opacity=0.9))
fig.update_layout(barmode="group")
fig.update_yaxes(title="ROAS ($ per $)")
C.style(fig, title="The naive read vs the truth", height=420)
"""),
    code(r"""
# The two demand-chasing channels read HIGH — Social's naive read is nearly
# double its true value. TV, which actually carries the largest true return
# per dollar of the four, reads LOWEST.
assert ratio["Social"] > 1.5          # Social wildly over-read
assert ratio["Search"] > 1.0          # Search over-read
assert naive.idxmin() == "TV" or ratio["TV"] < 0.6   # TV badly under-read
print(f"Social's naive read is {ratio['Social']:.1f}x its true ROAS; "
      f"TV's is {ratio['TV']:.1f}x.")
"""),
    md(r"""
## 4 — Why: the hidden third cause

The world we drew this history from has a **latent demand** signal — seasonal
appetite for patio furniture, housing turnover, weather, consumer mood — that
does two things at once:

- it lifts **sales** directly (people buy more in good weeks), and
- it lifts **spend**, because Veranda's budgeting *chases* demand: Search and
  Social budgets are turned up when the category heats up.

That double arrow is a **back-door path**: spend ← demand → sales. Correlation
flows along it even when advertising does nothing. The channels that chase
demand hardest (Search, Social) inherit the most borrowed correlation — which
is exactly the pattern the naive read produced.
"""),
    code(r"""
C.dag(
    nodes={"latent demand": (0.5, 1.0), "media spend": (0.0, 0.0), "sales": (1.0, 0.0)},
    edges=[("media spend", "sales")],
    bad_edges=[("latent demand", "media spend"), ("latent demand", "sales")],
    title="The open back-door: demand drives BOTH spend and sales",
    node_colors={"latent demand": "#e8d5a3", "media spend": "#cfe0f0",
                 "sales": "#d5e8d4"},
)
"""),
    md(r"""
## 5 — "Just build a better model" doesn't fix it

Maybe the naive read fails because OLS ignores adstock and saturation? Let's
use the real thing: a Bayesian MMM with geometric carryover, saturating
response curves, a trend, and full uncertainty — the same model family this
whole framework is built on. (The fit is cached by `causal_common`; the series
fits each world once.)
"""),
    code(r"""
fit = C.fit_world("uc")
g = fit["grade"]
print(g[["true", "est", "rel_err", "covered"]].round(2).to_string())
C.truth_forest(g, title="A REAL Bayesian MMM on the same history — still fooled",
               height=380)
"""),
    code(r"""
# The MMM is a far better model of the DATA — and still over-credits the
# demand-chasers, because the bias is in the WORLD, not in the estimator.
# Search's and Social's 90% intervals don't even contain the truth.
assert g.loc["Search", "rel_err"] > 0.15
assert g.loc["Social", "rel_err"] > 0.10
assert not g.loc["Search", "covered"]
print("The model is confident AND wrong about the chasers — "
      "confounding is not a model-quality problem.")
"""),
    md(r"""
## 6 — Same dashboard, different world

Here's the uncomfortable part. Draw a **different** hidden world — `clean`,
where spend is set by a media calendar that ignores demand entirely (no
back-door at all) — and put its dashboard next to Veranda's. The correlation
tables are the same *kind* of table. Nothing on either dashboard announces
which world generated it.

In the clean world the naive OLS read is biased too — but in the **opposite
direction** (it under-reads everything, because straight lines can't represent
carryover and saturation). Same method, same-looking data, opposite failure.
The lesson: **the size and direction of your error is a property of the world,
and the world is exactly what observational data won't show you.**
"""),
    code(r"""
sc_clean = C.scenario_for("clean")
C.check_truth(sc_clean)

Xc = np.column_stack([np.ones(len(sc_clean.weeks))] +
                     [sc_clean.spend[c] for c in sc_clean.channels])
naive_clean = pd.Series(np.linalg.lstsq(Xc, sc_clean.y.to_numpy(float), rcond=None)[0][1:],
                        index=sc_clean.channels)
ratio_clean = naive_clean / sc_clean.true_roas

cmp = pd.DataFrame({
    "confounded world (naive ÷ true)": ratio.round(2),
    "clean world (naive ÷ true)": ratio_clean.round(2),
})
print(cmp.to_string())

# In the clean world the naive read UNDER-shoots every channel; in the
# confounded world the chasers OVER-shoot. Same method, opposite errors.
assert (ratio_clean < 1.0).all()
assert ratio["Social"] > 1.0 and ratio["Search"] > 1.0
"""),
    md(r"""
## 7 — The ladder

So: dashboards can't answer causal questions, and better curve-fitting doesn't
rescue them. What does? **Stronger evidence, rung by rung.** Each rung of this
series buys a specific, nameable thing — and has a specific, nameable failure
mode that the next rung exists to fix.

| Rung | Evidence | What it buys | Notebook |
|---|---|---|---|
| 0 | A dashboard | Nothing causal — the cautionary tale | **00** (here) |
| 1 | Adjustment (controls) | Closes back-doors you can *observe* | **01** |
| 2 | A causal MMM + estimands | Explicit claims, refutation checks, honest "I don't know" | **02** |
| 3 | Structural mediation | The funnel: brand channels that work *through* awareness | **03** |
| 4 | Latent-factor structure | Confounders you can't see but can *measure* | **04** |
| 5 | One experiment, read honestly | Ground truth for one channel, one window | **05** |
| 6 | Calibration | The experiment repairs the model — surgically | **06** |
| 7 | A portfolio of experiments | Evidence composes; conflicts surface | **07** |
| 8 | Designed experiments | The *next* test, chosen by information value | **08** |
| 9 | A measurement program | A year of learning, budgeted by EVPI | **09** |
| 10 | The closed loop | Fit → design → measure → calibrate → converge | **10** |

Every notebook keeps three habits:

1. **Sealed keys.** The truth exists, is machine-checked (`C.check_truth`), and
   is only unsealed to grade.
2. **Directional asserts.** Every headline claim is executed as an `assert` —
   if the world or the framework drifts, the notebook fails to bake rather
   than silently telling a stale story.
3. **One brand, many worlds.** Veranda Home's dashboard looks the same in every
   notebook. What changes is the *hidden history* — which is the whole point.
"""),
    code(r"""
C.write_artifact("causal_00_naive_vs_truth.json", dict(
    world="unobserved_confounding",
    naive_roas={c: float(naive[c]) for c in sc.channels},
    true_roas={c: float(sc.true_roas[c]) for c in sc.channels},
    inflation={c: float(ratio[c]) for c in sc.channels},
    clean_inflation={c: float(ratio_clean[c]) for c in sc_clean.channels},
    mmm_rel_err={c: float(g.loc[c, "rel_err"]) for c in g.index},
    mmm_rhat_max=float(fit["diag"]["rhat_max"]),
))
print("artifact written: causal_00_naive_vs_truth.json")
"""),
    md(r"""
---
**Next — [01 · Confounding and adjustment](causal_01_confounding_and_adjustment.ipynb):**
three possible hidden histories of the same brand, what a control variable
*actually* buys you, why marking a confounder's causal role matters, and the
one disease no control can cure.
"""),
]


def main():
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    out = "causal/causal_00_the_ladder.ipynb"
    with open(out, "w") as f:
        nbformat.write(nb, f)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
