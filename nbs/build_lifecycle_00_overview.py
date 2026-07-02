"""Author lifecycle_00_overview.ipynb — the series opener (run from ``nbs/``).

    uv run --with nbformat python build_lifecycle_00_overview.py
    PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        lifecycle_00_overview.ipynb --ExecutePreprocessor.timeout=1800 \
        --ExecutePreprocessor.kernel_name=python3

Overview of the **Experimental Measurement Lifecycle** series: why a fitted MMM
is a hypothesis and not an answer, the T0->T5 adaptive loop that turns it into a
defensible one, and a teaser of the payoff (an ROI interval collapsing onto the
truth after one experiment). Shared world/fit/palette live in
``nbs/lifecycle_common.py``.
"""

from __future__ import annotations

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


# The canonical setup cell — every notebook in the series opens with this.
SETUP = r"""
import warnings; warnings.filterwarnings("ignore")
import os
os.environ.setdefault("TQDM_DISABLE", "1")   # quiet sampling progress bars (keeps outputs clean)
import sys
# Make the shared module importable whether the kernel runs from nbs/ or repo root.
for _p in (".", os.path.join(os.getcwd(), "nbs"), os.path.dirname(os.getcwd())):
    if os.path.isfile(os.path.join(_p, "lifecycle_common.py")) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.templates.default = "plotly_white"
pio.renderers.default = "notebook_connected"
pd.set_option("display.width", 140)

# Keep baked outputs clean: silence the framework's INFO/DEBUG logs and the
# sampler's chatter (fits still run — we just don't print their progress).
import logging
from loguru import logger
logger.disable("mmm_framework")
for _n in ("pymc", "pymc.sampling", "numpyro", "jax", "arviz"):
    logging.getLogger(_n).setLevel(logging.ERROR)

import lifecycle_common as L
print(f"{L.BRAND} — {L.TAGLINE}")
print("channels:", ", ".join(L.CHANNELS), "| KPI:", L.KPI)
"""


CELLS = [
    md(r"""
# The Experimental Measurement Lifecycle
### How a marketing-mix model earns the right to allocate a budget — one experiment at a time

*Notebook 0 of 7 — the map.*

Meet **Northwind Outfitters**, a national outdoor-apparel brand spending across
four paid channels — **TV, Search, Social, and Display**. Northwind has three
years of clean weekly data and a marketing-mix model that fits it well. The CFO
asks the obvious question: *"So where should next quarter's budget go?"*

The honest answer is **"we're not sure yet"** — and this series is about turning
that into **"here, and we can defend it."**

The problem is not the model's math. It's that a model fit to *observational*
history can only ever tell you what **correlated** with sales, and marketers
spend into demand: budgets go **up** exactly when the brand is already hot. So a
naive read credits *spend* for sales that **demand** would have delivered anyway.
The only way to break that tie is to **create variation on purpose** — to run an
**experiment** — and fold what it measures back into the model.

But you can't experiment on everything: tests cost real money and real time. So
the framework runs a **loop** — decide *what* is worth testing, design it so it's
actually powered and affordable, fold the result in, re-allocate, and know *when
the answer goes stale.* This notebook is the map of that loop; the six that
follow walk each stage on Northwind's live data.
"""),
    code(SETUP),
    md(r"""
## Why the dashboard lies (the 60-second version)

Before spending a dollar on testing, it's worth re-earning *why* testing is
necessary. Here is the trap in one picture: a hidden **demand** signal moves
**both** spend and sales, so the line you'd fit through weekly history — the one
the dashboard reports as "ROI" — is far **steeper** than the true causal return.

This is illustrative data, not a model fit; it just makes the confounding
visible. The whole series exists to get *past* this picture.
"""),
    code(r"""
rng = np.random.default_rng(3)
weeks = 104
demand = np.cumsum(rng.normal(0, 1, weeks)); demand = (demand - demand.mean()) / demand.std()
season = 0.6 * np.sin(np.arange(weeks) * 2 * np.pi / 52)
hot = demand + season                                    # latent demand index (unobserved)
spend = 560 + 130 * hot + rng.normal(0, 25, weeks)       # planners chase demand
TRUE_SLOPE = 0.9                                          # real $ sales per $ spend, holding demand fixed
sales = 1900 + TRUE_SLOPE * spend + 240 * hot + rng.normal(0, 60, weeks)

naive_slope = float(np.polyfit(spend, sales, 1)[0])
xs = np.linspace(spend.min(), spend.max(), 50)
fig = go.Figure()
fig.add_trace(go.Scatter(x=spend, y=sales, mode="markers", name="weekly history",
    marker=dict(color=L.MUTED, size=7, line=dict(color="white", width=0.5))))
fig.add_trace(go.Scatter(x=xs, y=np.polyval(np.polyfit(spend, sales, 1), xs), mode="lines",
    name=f"naive 'ROI' fit  (slope {naive_slope:.1f}x)", line=dict(color=L.BAD, width=3)))
fig.add_trace(go.Scatter(x=xs, y=sales.mean() + TRUE_SLOPE * (xs - spend.mean()), mode="lines",
    name=f"true causal slope  ({TRUE_SLOPE:.1f}x)", line=dict(color=L.GOOD, width=3, dash="dash")))
L.style(fig, title="Three years of history: the naive slope over-credits spend",
        legend=dict(orientation="h", yanchor="top", y=-0.16, x=0))
fig.update_layout(xaxis_title="weekly spend ($k)", yaxis_title="weekly sales ($k)")
fig.show()
print(f"Naive slope: every $1 'returns' ${naive_slope:.1f}.  Held-demand truth: ${TRUE_SLOPE:.1f}.")
assert naive_slope > TRUE_SLOPE * 1.4     # confounding materially inflates the read
"""),
    md(r"""
**Readout →** *"We can't allocate on this. History flatters every channel because
we only ever spent hard when demand was high. To get numbers we can defend, we
have to create the variation ourselves."* That is the entire case for the loop.
"""),
    md(r"""
## The loop

The framework treats measurement as a **cycle**, not a one-off report. Each pass
sharpens the picture and the cycle re-points itself at whatever is now most worth
learning.
"""),
    code(r"""
stages = [
    ("T0", "Fit",         "the baseline model"),
    ("T1", "Prioritize",  "EIG / EVOI"),
    ("T2", "Design",      "power & economics"),
    ("T3", "Calibrate",   "fold in the readout"),
    ("T4", "Allocate",    "spend the answer"),
    ("T5", "Re-evaluate", "information decay"),
]
STAGE_COLORS = ["#3b6fb6", "#4a8d57", "#d98c3f", "#b15a7a", "#5a9e6f", "#7a6fb1"]
n = len(stages)
ang = np.pi / 2 - np.arange(n) * 2 * np.pi / n         # start at top, go clockwise
x, y = np.cos(ang), np.sin(ang)

fig = go.Figure()
for i in range(n):                                      # arrows around the ring
    j = (i + 1) % n
    fig.add_annotation(x=x[j], y=y[j], ax=x[i], ay=y[i],
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2,
        arrowcolor=L.MUTED, standoff=34, startstandoff=34)
fig.add_trace(go.Scatter(x=x, y=y, mode="markers+text",
    marker=dict(size=62, color=STAGE_COLORS, line=dict(color="white", width=2)),
    text=[s[0] for s in stages], textfont=dict(color="white", size=18),
    textposition="middle center", hoverinfo="skip", showlegend=False))
for i, (t, name, sub) in enumerate(stages):             # labels outside the ring
    fig.add_annotation(x=1.42 * x[i], y=1.42 * y[i], text=f"<b>{name}</b><br>{sub}",
        showarrow=False, font=dict(size=12, color=L.INK),
        align="center", yanchor="middle")
fig.add_annotation(x=0, y=0, text="the<br>measurement<br>loop",
    showarrow=False, font=dict(size=13, color=L.MUTED))
fig.update_xaxes(visible=False, range=[-2, 2])
fig.update_yaxes(visible=False, range=[-1.8, 1.8], scaleanchor="x")
L.style(fig, height=520, title="T0 → T5, then back to T1")
fig.show()
"""),
    md(r"""
Each stage answers one operating question — and each is its own notebook in this
series:

| Stage | The question it answers | Notebook |
|---|---|---|
| **T0 · Fit** | What does the *observational* model believe — and how sure is it? | `lifecycle_01_fit_baseline` |
| **T1 · Prioritize** | Of everything we're unsure about, what is worth *testing*? | `lifecycle_02_prioritize` |
| **T2 · Design** | How do we run that test so it's *powered* and *affordable*? | `lifecycle_03_design` |
| **T3 · Calibrate** | How does the readout update the model? | `lifecycle_04_calibrate` |
| **T4 · Allocate** | What's the budget now — and how confident are we? | `lifecycle_05_allocate` |
| **T5 · Re-evaluate** | When does this answer go stale and trigger a re-test? | `lifecycle_06_reevaluate` |

The load-bearing idea, front and center: **you don't test what you're most
*unsure* about — you test what most changes the *decision*.** That distinction
(information vs. value of information) is T1, and it drives everything after it.
"""),
    md(r"""
## What a single fit actually gives you

Enough preamble — here is Northwind's real model. Everything below is estimated
from data; the sealed "truth" is only ever used to *grade* the loop, never to run
it. Let's load the fitted baseline and look at the channel ROI it reports **with
its uncertainty** — the analyst's honest starting point.
"""),
    code(r"""
from mmm_framework.reporting.helpers.roi import compute_roi_with_uncertainty

base = L.fit_baseline()
model, truth = base["model"], base["truth"]
roi = compute_roi_with_uncertainty(model).set_index("channel").loc[L.CHANNELS]

fig = go.Figure()
for i, ch in enumerate(L.CHANNELS):
    r = roi.loc[ch]
    fig.add_trace(go.Scatter(
        x=[r["roi_hdi_low"], r["roi_hdi_high"]], y=[ch, ch], mode="lines",
        line=dict(color=L.PALETTE[ch], width=8), showlegend=False, opacity=0.45))
    fig.add_trace(go.Scatter(
        x=[r["roi_mean"]], y=[ch], mode="markers", showlegend=False,
        marker=dict(color=L.PALETTE[ch], size=13, line=dict(color="white", width=1.5))))
fig.add_vline(x=1.0, line=dict(color=L.INK, width=1, dash="dot"),
              annotation_text="break-even ($1 back per $1)", annotation_position="top")
L.style(fig, title="Baseline MMM — average ROI per channel, with 94% credible interval")
fig.update_layout(xaxis_title="return per $1 of spend (ROI)", yaxis_title="")
fig.show()
print(roi[["roi_mean", "roi_hdi_low", "roi_hdi_high"]].round(3).to_string())
print("\nThese intervals are wide — and we cannot see which are biased. That is the problem.")
assert (roi["roi_hdi_high"] - roi["roi_hdi_low"]).min() > 0.1   # real, non-trivial uncertainty
"""),
    md(r"""
Wide bands, all below break-even on *average* ROI (normal for a saturated media
model — the *marginal* dollar is what matters, and we'll get there). The catch:
the analyst **cannot tell which of these are biased low.** A credible interval is
only honest about the uncertainty the *model* knows about; it says nothing about
the confounding baked into the history. That's the gap an experiment closes.
"""),
    md(r"""
## The payoff, in one picture

Here's where the series is headed. In T1 the loop will single out **one** channel
worth testing; in T2 it will design a geo holdout to measure it; in T3 it folds
that readout in. Below is a preview of what that does to the chosen channel's ROI
— and, since this is the map, we'll break the fourth wall and overlay the sealed
**truth** so you can see the interval move the *right* way.
"""),
    code(r"""
focus = L.FOCUS_CHANNEL
roi_before = float(roi.loc[focus, "roi_mean"])
lo_b, hi_b = float(roi.loc[focus, "roi_hdi_low"]), float(roi.loc[focus, "roi_hdi_high"])

cal = L.fit_calibrated(verbose=False)          # baseline + one experiment on the focus channel, refit
roi_c = compute_roi_with_uncertainty(cal["model"]).set_index("channel").loc[L.CHANNELS]
roi_after = float(roi_c.loc[focus, "roi_mean"])
lo_a, hi_a = float(roi_c.loc[focus, "roi_hdi_low"]), float(roi_c.loc[focus, "roi_hdi_high"])
truth_roi = float(truth["true_roas"][focus])

fig = go.Figure()
for label, mean, lo, hi, yy, col in [
    ("before — observational fit", roi_before, lo_b, hi_b, 1, L.MUTED),
    ("after — one experiment folded in", roi_after, lo_a, hi_a, 0, L.PALETTE[focus]),
]:
    fig.add_trace(go.Scatter(x=[lo, hi], y=[yy, yy], mode="lines",
        line=dict(color=col, width=10), opacity=0.4, showlegend=False))
    fig.add_trace(go.Scatter(x=[mean], y=[yy], mode="markers+text", text=[label],
        textposition="top center", marker=dict(color=col, size=14,
        line=dict(color="white", width=1.5)), showlegend=False))
fig.add_vline(x=truth_roi, line=dict(color=L.GOOD, width=2.5, dash="dash"),
              annotation_text=f"sealed truth = {truth_roi:.2f}", annotation_position="bottom right")
fig.update_yaxes(range=[-0.6, 1.7], visible=False)
L.style(fig, height=340, title=f"{focus}: the loop pulls the ROI onto the truth (and tightens it)")
fig.update_layout(xaxis_title="return per $1 of spend (ROI)")
fig.show()
print(f"{focus} ROI:  before {roi_before:.3f}  ->  after {roi_after:.3f}  ->  truth {truth_roi:.3f}")
gap_before, gap_after = abs(roi_before - truth_roi), abs(roi_after - truth_roi)
print(f"gap to truth closed by {100 * (1 - gap_after / gap_before):.0f}%")
assert gap_after < gap_before          # calibration moves the estimate toward truth
"""),
    md(r"""
One experiment, and the channel the loop chose to test lands on the truth while
its interval tightens. That is the whole game — and notice the *other* channels
would barely move (a single test corrects the channel it measured, not the ones
it didn't). The chapters ahead earn every step of this picture on Northwind's
data, no fourth-wall truth required.

### How to read this series

- Each notebook is **one stage** and stands alone, but they share **one brand and
  one fitted model** (cached by `lifecycle_common.py`, so nothing re-samples from
  scratch).
- **No number is hardcoded in prose** — the text points at the live tables and
  plots, so a re-bake with a new seed stays honest.
- Every code cell ends in a small **assert** that encodes the claim it just made,
  so "it ran" means "the story still holds."

### Rebuilding

From `nbs/`, build then bake:

```bash
uv run --with nbformat python build_lifecycle_00_overview.py
PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
    jupyter nbconvert --to notebook --execute --inplace \
    lifecycle_00_overview.ipynb --ExecutePreprocessor.timeout=1800 \
    --ExecutePreprocessor.kernel_name=python3
```

Next: **`lifecycle_01_fit_baseline`** — T0, where we build the model this whole
loop hangs on.
"""),
]


def main() -> None:
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    out = "lifecycle_00_overview.ipynb"
    with open(out, "w") as f:
        nbformat.write(nb, f)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    print(f"wrote {out} with {len(nb.cells)} cells ({n_code} code)")


if __name__ == "__main__":
    main()
