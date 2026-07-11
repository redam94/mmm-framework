"""Author lifecycle_02_prioritize.ipynb — T1 of the series (run from ``nbs/``).

    uv run --with nbformat python builders/build_lifecycle_02_prioritize.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        lifecycle/lifecycle_02_prioritize.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

T1 · Prioritize: the EIG / EVOI / EVPI priority grid and its 2x2 quadrant. The
load-bearing lesson of the whole series — you don't test what you're most
*unsure* about, you test what most changes the *decision* — lives here, computed
on Northwind's live baseline posterior. Shared world/fit/palette live in
``nbs/builders/lifecycle_common.py``.
"""

from __future__ import annotations

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


# The canonical setup cell — verbatim across the series.
SETUP = r"""
import warnings; warnings.filterwarnings("ignore")
import os
os.environ.setdefault("TQDM_DISABLE", "1")
import sys
for _p in ("../builders", ".", os.path.join(os.getcwd(), "nbs", "builders"), os.path.dirname(os.getcwd())):
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

import logging
from loguru import logger
logger.disable("mmm_framework")
for _n in ("pymc", "pymc.sampling", "numpyro", "jax", "arviz"):
    logging.getLogger(_n).setLevel(logging.ERROR)

import lifecycle_common as L
print(f"{L.BRAND} — {L.TAGLINE}")
"""


CELLS = [
    md(r"""
# T1 · Prioritize — what is worth testing?
### Notebook 2 of 7 — the Experimental Measurement Lifecycle

**Recap (from `lifecycle_01_fit_baseline`).** We fit Northwind's baseline MMM and
got exactly what an *observational* fit gives you: four channel ROIs with **wide,
possibly-biased credible intervals**. The model is honest about what it doesn't
know — but it cannot tell us which of those numbers are biased, and we cannot
afford to experiment on everything (tests cost real money and real weeks). So T1
asks the sharpest question in the whole loop:

> **Of everything we're unsure about, what is actually worth an experiment?**

The load-bearing lesson of this entire series lives right here: **you don't test
what you're most *unsure* about — you test what most changes the *decision*.**
Three quantities pull those ideas apart:

- **EIG — Expected Information Gain** *(nats).* How much a well-run experiment
  would **teach** us about a channel: pure uncertainty reduction. A vague channel
  has high EIG whether or not the answer would ever change what we do.
- **EVOI — Expected Value of Information** *(KPI units).* How much that learning
  is **worth to the budget** — the sales the *better allocation* buys once the
  test resolves. A channel can be very uncertain yet have *low* EVOI if the money
  barely moves when you learn the truth.
- **EVPI — Expected Value of *Perfect* Information** *(KPI units).* The **ceiling**
  — what it would be worth to erase *all* channel uncertainty at once. Each
  channel's EVOI is a slice of that pie, so `EVOI / EVPI` tells you how much of
  the total prize a single test can capture.

Cross **EIG** against **EVOI** and you get a **2×2 quadrant** that sorts the whole
portfolio into one action per channel. Everything below is computed from the T0
posterior alone — the sealed truth is never consulted.
"""),
    code(SETUP),
    md(r"""
## The priority grid

`compute_experiment_priorities` walks the fitted posterior once: for each channel
it reads the **average-ROI** draws (contribution-at-current-spend ÷ spend),
turns their spread into **EIG**, simulates how resolving that channel would
re-point the optimizer to get **EVOI**, and normalizes both into a composite
**priority** (a geometric mean, so a channel can't win by being loud on one axis
and silent on the other). We fix `max_draws=300` and `random_seed=42` so the grid
is stable across re-bakes.
"""),
    code(r"""
from mmm_framework.planning.priority import compute_experiment_priorities, QUADRANTS

base = L.fit_baseline()
model = base["model"]

# One pass over the posterior -> per-channel EIG/EVOI + a portfolio summary.
grid, port = compute_experiment_priorities(model, max_draws=300, random_seed=42)
df = pd.DataFrame([g.to_dict() for g in grid])          # already sorted by priority

cols = ["channel", "roi_mean", "roi_sd", "eig", "evoi", "evpi_share", "priority", "quadrant"]
print(df[cols].round(4).to_string(index=False))
print("\nportfolio:")
for k in ["v_current", "evpi", "total_budget", "eig_threshold", "evoi_threshold", "design_type"]:
    print(f"  {k:>15}: {port[k]}")
print(f"\nquadrant map (EIG, EVOI) -> action: {QUADRANTS}")

# The loop's #1 pick is the channel this series carries end-to-end.
assert grid[0].channel == L.FOCUS_CHANNEL
"""),
    md(r"""
## The 2×2: information vs. value of information

The table sorts by **priority**, but the *shape* of the decision is clearest as a
picture. Plot every channel by its **EIG** (x — how much we'd learn) against its
**EVOI** (y — what that learning is worth), split each axis at its **median**, and
the four quadrants name themselves:

- **`test_now`** *(high EIG, high EVOI)* — uncertain **and** decision-critical.
  This is where the test budget goes.
- **`monitor`** *(low EIG, high EVOI)* — high stakes, but the fit is already
  precise; watch it for drift instead of paying to test it.
- **`learn_cheaply`** *(high EIG, low EVOI)* — you'd learn a lot, but the budget
  barely moves; only worth it if a test is nearly free.
- **`deprioritize`** *(low, low)* — leave it alone.

Marker **size** is each channel's share of current spend.
"""),
    code(r"""
xt, yt = port["eig_threshold"], port["evoi_threshold"]

fig = go.Figure()
fig.add_vline(x=xt, line=dict(color=L.MUTED, width=1.5, dash="dash"),
              annotation_text="median EIG", annotation_position="top")
fig.add_hline(y=yt, line=dict(color=L.MUTED, width=1.5, dash="dash"),
              annotation_text="median EVOI", annotation_position="right")
for _, r in df.iterrows():
    ch = r["channel"]
    fig.add_trace(go.Scatter(
        x=[r["eig"]], y=[r["evoi"]], mode="markers+text",
        text=[f"<b>{ch}</b>"], textposition="top center",
        marker=dict(color=L.PALETTE[ch], size=20 + 150 * r["spend_share"],
                    line=dict(color="white", width=1.5), opacity=0.9),
        showlegend=False, hovertext=ch))

# Name the four corners straight from QUADRANTS (paper coords -> always in frame).
corner = {("high", "high"): (0.985, 0.96, "right", "top"),
          ("low", "high"):  (0.015, 0.96, "left", "top"),
          ("high", "low"):  (0.985, 0.04, "right", "bottom"),
          ("low", "low"):   (0.015, 0.04, "left", "bottom")}
for key, action in QUADRANTS.items():
    px, py, xa, ya = corner[key]
    fig.add_annotation(x=px, y=py, xref="paper", yref="paper", text=f"<i>{action}</i>",
                       showarrow=False, xanchor=xa, yanchor=ya,
                       font=dict(size=12, color=L.MUTED))

xr = (df["eig"].max() - df["eig"].min()) * 0.35 + 0.02
yr = (df["evoi"].max() - df["evoi"].min()) * 0.20 + 0.3
fig.update_xaxes(range=[df["eig"].min() - xr, df["eig"].max() + xr])
fig.update_yaxes(range=[df["evoi"].min() - yr, df["evoi"].max() + yr])
L.style(fig, height=520, title="The experiment-priority quadrant — where each channel lands")
fig.update_layout(xaxis_title="EIG — expected information gain (nats, how much we'd learn)",
                  yaxis_title="EVOI — value of that learning (KPI units)")
fig.show()

focus_quad = df.set_index("channel").loc[L.FOCUS_CHANNEL, "quadrant"]
print(f"{L.FOCUS_CHANNEL} lands in: {focus_quad}")
assert focus_quad == "test_now"     # uncertain AND budget-critical
"""),
    md(r"""
## Why not just test whatever we're least sure about?

Here is the trap the naive instinct falls into. Look at the two **high-stakes**
channels — the loop's pick (`test_now`) and the one it tells us only to
`monitor`. Both move a lot of budget, so an uncertainty-first analyst might test
either. The difference is what an experiment would *teach*: the `monitor` channel
already has a **tighter posterior** (smaller `roi_sd`), so there's little left to
learn — its **EIG is low** even though its stakes are high. The loop's pick is
uncertain **and** high-stakes, so resolving it both teaches us a lot *and* moves
the money. That double-qualification is exactly what separates the two.
"""),
    code(r"""
di = df.set_index("channel")
focus = L.FOCUS_CHANNEL
monitor_ch = df[df["quadrant"] == "monitor"].sort_values("evoi", ascending=False)["channel"].iloc[0]

cmp = di.loc[[focus, monitor_ch], ["roi_sd", "eig", "evoi", "quadrant"]]
print(cmp.round(4).to_string())
print(f"\n{focus}:  uncertain (roi_sd HIGH) AND high EVOI  ->  {di.loc[focus, 'quadrant']}")
print(f"{monitor_ch}:  high EVOI but already precise (roi_sd LOW)  ->  {di.loc[monitor_ch, 'quadrant']}")

chans = [monitor_ch, focus]
bar_cols = [L.PALETTE[c] for c in chans]
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.18,
                    subplot_titles=("EIG — how much we'd learn (nats)",
                                    "EVOI — what that learning is worth (KPI units)"))
fig.add_trace(go.Bar(x=chans, y=di.loc[chans, "eig"], marker_color=bar_cols,
                     text=di.loc[chans, "eig"].round(3), textposition="outside",
                     showlegend=False), row=1, col=1)
fig.add_trace(go.Bar(x=chans, y=di.loc[chans, "evoi"], marker_color=bar_cols,
                     text=di.loc[chans, "evoi"].round(2), textposition="outside",
                     showlegend=False), row=1, col=2)
L.style(fig, height=420, title=f"{focus} vs {monitor_ch}: same stakes, very different information")
fig.show()

# The channel we're told to TEST has more to teach us than the one we only MONITOR.
assert di.loc[focus, "eig"] >= di.loc[monitor_ch, "eig"]
"""),
    md(r"""
## The ceiling: how much of *perfect* information can one test buy?

Every EVOI is measured against **EVPI** — the value of resolving *all* channel
uncertainty at once, the most any amount of testing could ever be worth. Each
channel's `EVOI / EVPI` (**`evpi_share`**) is its slice of that ceiling. A single
national-pulse test is a modest instrument, so no one slice is large in absolute
terms — but the ranking is unambiguous about where the very first dollar of test
budget belongs.
"""),
    code(r"""
order = df.sort_values("evpi_share", ascending=True)
fig = go.Figure()
fig.add_trace(go.Bar(
    x=order["evpi_share"], y=order["channel"], orientation="h",
    marker_color=[L.PALETTE[c] for c in order["channel"]],
    text=[f"{v * 100:.2f}%" for v in order["evpi_share"]], textposition="outside",
    showlegend=False))
L.style(fig, height=380, title="Share of the value of *perfect* information one test could capture")
fig.update_layout(xaxis_title="EVOI / EVPI  (share of the ceiling)",
                  xaxis_tickformat=".1%", yaxis_title="")
fig.show()
print(f"EVPI (value of resolving ALL uncertainty): {port['evpi']:.2f} {L.KPI}-units")
print(f"Top slice: {L.FOCUS_CHANNEL} captures the largest share of it with one test.")

# The focus channel isn't just highest-priority — it's the single biggest slice of EVPI.
assert di.loc[L.FOCUS_CHANNEL, "evpi_share"] == df["evpi_share"].max()
"""),
    md(r"""
## Readout

**Display it is.** It is the one channel that is both **uncertain** and
**budget-critical** — the only `test_now` in the portfolio — so it earns the first
experiment. TV matters just as much to the budget, but we already know its return
well enough; we'll **monitor** it for drift (that's T5). Search and Social can
wait. Crucially, this whole ranking was decided **without ever peeking at the
sealed truth** — EIG and EVOI came entirely from the posterior we fit in T0.

Now we need a test that can actually measure Display — one that is **powered** and
**affordable**, because a good idea we can't measure is worth nothing.

Next: **`lifecycle_03_design`** — T2, where we design that experiment (power &
economics). Previously: **`lifecycle_01_fit_baseline`** built the fit this ranking
stands on.
"""),
]


def main() -> None:
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    out = "lifecycle/lifecycle_02_prioritize.ipynb"
    with open(out, "w") as f:
        nbformat.write(nb, f)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    print("wrote", n_code, "code cells")


if __name__ == "__main__":
    main()
