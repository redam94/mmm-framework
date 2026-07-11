"""Author lifecycle_05_allocate.ipynb — T4 of the Experimental Measurement
Lifecycle series (run from ``nbs/``).

    uv run --with nbformat python builders/build_lifecycle_05_allocate.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat \
        --with ipykernel jupyter nbconvert --to notebook --execute --inplace \
        lifecycle/lifecycle_05_allocate.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

**T4 · Allocate — spend the answer.** With Display calibrated (T3), re-optimize
Northwind's budget UNDER UNCERTAINTY: compute the optimal reallocation per
posterior draw and summarize last, so the recommendation ships with a confidence
band. Show the reallocation-with-bands, the diminishing-returns response curve
(marginal vs average ROAS disagree), that the calibrated plan is tighter than the
raw one, and the guardrailed ±20% default for the deck. Shared world / fit /
palette live in ``nbs/builders/lifecycle_common.py``.
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

# Appended to the setup cell: load the CALIBRATED model this whole stage runs on.
LOAD = r"""

# T4 runs on the CALIBRATED model — Display's experiment (T3) is folded in, so
# its return is anchored to a readout rather than guessed from history.
cal = L.fit_calibrated()          # baseline + one in-graph Display readout, refit
model, truth = cal["model"], cal["truth"]
print(f"loaded calibrated model — focus channel: {cal['channel']} | KPI: {L.KPI}")
print("channels:", ", ".join(model.channel_names))
assert cal["channel"] == L.FOCUS_CHANNEL      # the loop carries Display end-to-end
"""


CELLS = [
    md(r"""
# T4 · Allocate — spend the answer
### Turning a calibrated model into a budget you can defend

*Notebook 5 of 7 — the T4 stage of the measurement loop.*

In `lifecycle_04_calibrate` we folded Northwind's geo-holdout readout into the
model: **Display**'s return is no longer an observational guess — it is
**anchored to an experiment**. So the CFO's original question finally has an
honest answer: *where should next quarter's budget go?*

This is **T4 · Allocate**. Three ideas carry the chapter:

- **Reallocation is a decision, not a point estimate.** We don't report one
  "optimal" split — we re-optimize the budget **under every posterior draw**, so
  the recommendation comes with a confidence band: *how sure are we this move
  wins?*
- **Marginal beats average.** The channel with the best **average ROAS** is
  rarely where the **next dollar** belongs. Allocation is driven by the
  **marginal** return — the slope of the response curve at today's spend — and
  the two routinely disagree.
- **Calibration changes the plan, not just the ROI.** Because T3 pinned Display,
  the calibrated plan is **tighter and more defensible** than the one the raw
  observational model would have proposed.

The recipe throughout: **compute per draw, summarize last** — never collapse to a
mean before you've propagated the uncertainty.
"""),
    code(SETUP + LOAD),
    md(r"""
## From response curves to a plan

The model doesn't store "an allocation" — it stores a **response curve** per
channel: how much **Sales** each channel returns as we scale its spend up or
down, sampled across the full posterior. `compute_response_curves` reads those
curves straight off the fitted graph (one posterior pass per spend level), and
`optimize_budget` runs a greedy **marginal-return** allocator that pours each
next dollar into whichever channel's curve is climbing fastest — **once per
draw**, so the spread of answers becomes the plan's uncertainty.

With no budget change specified, the optimizer **reallocates the current total**
— same money, better split. Read the table as *"move this many dollars from here
to there,"* and note the **share band** (`p5`–`p95`): the channels the model is
confident about versus the ones still up for grabs.
"""),
    code(r"""
from mmm_framework.planning.budget import compute_response_curves, optimize_budget, default_reallocation

curves = compute_response_curves(model, max_draws=150, random_seed=0)
res = optimize_budget(curves=curves, random_seed=0)   # reallocate the current total, per draw

# A readable reallocation plan: current vs optimal spend, the $ move, and the
# share the optimizer is (un)sure about.
plan = res.table.set_index("channel").loc[list(curves.channel_names)].copy()
plan["delta_$"] = plan["optimal_spend"] - plan["current_spend"]
cols = ["current_spend", "optimal_spend", "delta_$", "change_pct", "optimal_share_p5", "optimal_share_p95"]
print(plan[cols].round(2).to_string())

lo, hi = res.uplift_hdi
print(f"\nTotal budget (conserved): {L.dollars(res.total_budget)}")
print(f"Expected uplift (median): {L.dollars(res.expected_uplift)}   90% band [{L.dollars(lo)}, {L.dollars(hi)}]")
print(f"P(uplift > 0) = {res.prob_positive_uplift:.0%}")
assert abs(res.table["optimal_spend"].sum() - res.total_budget) < 1e-6 * max(res.total_budget, 1)  # budget conserved
"""),
    md(r"""
## The move, with a confidence band

Here is the recommendation as a picture: **current** spend (grey) against the
**recommended** spend (channel colour), with the error bar showing the
**p5–p95 range** of each channel's optimal share — the honest width of the
recommendation. A tall bar with a tight whisker is a move you can commit to; a
tall bar with a wide whisker is a *direction*, not a mandate.
"""),
    code(r"""
p5_d = plan["optimal_share_p5"] / 100.0 * res.total_budget
p95_d = plan["optimal_share_p95"] / 100.0 * res.total_budget
err_plus = np.maximum(p95_d.values - plan["optimal_spend"].values, 0.0)
err_minus = np.maximum(plan["optimal_spend"].values - p5_d.values, 0.0)
chans = list(plan.index)

fig = go.Figure()
fig.add_trace(go.Bar(x=chans, y=plan["current_spend"], name="current spend",
    marker_color=L.MUTED, opacity=0.65))
fig.add_trace(go.Bar(x=chans, y=plan["optimal_spend"], name="recommended spend",
    marker_color=[L.PALETTE[c] for c in chans],
    error_y=dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus,
                 color=L.INK, thickness=1.4, width=6)))
L.style(fig, title="T4 reallocation — where the calibrated model moves the budget",
        barmode="group", legend=dict(orientation="h", yanchor="top", y=-0.14, x=0))
fig.update_layout(xaxis_title="", yaxis_title="window spend ($000s)")
fig.show()

movers = plan["delta_$"].abs().sort_values(ascending=False)
print("largest $ moves:", ", ".join(f"{c} {plan.loc[c, 'delta_$']:+,.0f}" for c in movers.index[:3]))
assert np.isfinite(res.table.select_dtypes("number").to_numpy()).all()   # table finite
"""),
    md(r"""
## Why the next dollar is worth less than the last

Zoom into **Display**'s own response curve to see *why* the optimizer moves money
the way it does. The curve **saturates** — each additional dollar buys less
**Sales** than the one before — so the two ways of scoring a channel pull apart:

- the **average ROAS** (dotted line from the origin) is the return on *every*
  dollar Display has spent so far;
- the **marginal ROAS** (dashed tangent at today's spend) is the return on the
  *next* dollar.

The optimizer only cares about the **marginal** line. A channel can look great on
average and still be a poor home for the next dollar — which is exactly how a
saturated channel gets *money moved away from it* even while its average ROAS
stays high.
"""),
    code(r"""
c = list(curves.channel_names).index(L.FOCUS_CHANNEL)
spend_c = curves.spend_grid[c]              # (G,) spend at each multiplier
mean_c = curves.mean_curves()[c]           # (G,) posterior-mean contribution
mults = curves.multipliers
i1 = int(np.argmin(np.abs(mults - 1.0)))   # the "current spend" operating point
x_now, y_now = float(spend_c[i1]), float(mean_c[i1])

# average (chord from origin) vs marginal (local slope) ROAS at today's spend
avg_roas = y_now / max(x_now, 1e-9)
j_lo, j_hi = max(i1 - 1, 0), min(i1 + 1, len(mults) - 1)
mar_roas = float((mean_c[j_hi] - mean_c[j_lo]) / max(spend_c[j_hi] - spend_c[j_lo], 1e-9))

fig = go.Figure()
fig.add_trace(go.Scatter(x=spend_c, y=mean_c, mode="lines+markers", name="response curve",
    line=dict(color=L.PALETTE[L.FOCUS_CHANNEL], width=3),
    marker=dict(size=6, color=L.PALETTE[L.FOCUS_CHANNEL])))
xs = np.array([x_now * 0.55, x_now * 1.45])
fig.add_trace(go.Scatter(x=[0, x_now], y=[0, y_now], mode="lines",
    line=dict(color=L.GOOD, width=2, dash="dot"), name=f"average ROAS {avg_roas:.2f}x"))
fig.add_trace(go.Scatter(x=xs, y=y_now + mar_roas * (xs - x_now), mode="lines",
    line=dict(color=L.BAD, width=2, dash="dash"), name=f"marginal ROAS {mar_roas:.2f}x"))
fig.add_trace(go.Scatter(x=[x_now], y=[y_now], mode="markers+text", text=["current spend"],
    textposition="top left", showlegend=False,
    marker=dict(color=L.INK, size=13, symbol="diamond", line=dict(color="white", width=1.5))))
L.style(fig, title=f"{L.FOCUS_CHANNEL}: diminishing returns — the next dollar is worth less than the average one",
        legend=dict(orientation="h", yanchor="top", y=-0.16, x=0))
fig.update_layout(xaxis_title="window spend ($000s)", yaxis_title=f"{L.KPI} contribution")
fig.show()

print(f"{L.FOCUS_CHANNEL} at current spend:  average ROAS {avg_roas:.2f}x   vs   marginal ROAS {mar_roas:.2f}x")
assert np.all(np.diff(mean_c) >= -1e-6 * (abs(mean_c).max() + 1e-9)) or mean_c[-1] > mean_c[0]   # curve rises
"""),
    md(r"""
## The calibrated plan is the one you can defend

Would we have made the same call *before* the experiment? Re-run the exact same
optimizer on the **baseline** (observational) model and compare. The point isn't
only whether the recommended split moves — it's whether the **confidence** in it
does. Because T3 anchored Display to a real readout, the calibrated plan should
carry a **narrower band** on both the channel's recommended share and the
expected uplift: same machinery, a recommendation you can take to the CFO without
hedging.
"""),
    code(r"""
base = L.fit_baseline()
curves0 = compute_response_curves(base["model"], max_draws=150, random_seed=0)
res0 = optimize_budget(curves=curves0, random_seed=0)     # the raw observational plan

def _focus_row(r):
    t = r.table.set_index("channel")
    return t.loc[L.FOCUS_CHANNEL, ["optimal_share_pct", "optimal_share_p5", "optimal_share_p95"]]

b0, b1 = _focus_row(res0), _focus_row(res)
width0 = float(b0["optimal_share_p95"] - b0["optimal_share_p5"])
width1 = float(b1["optimal_share_p95"] - b1["optimal_share_p5"])

fig = go.Figure()
for label, row, yy, col in [
    ("before — observational plan", b0, 1, L.MUTED),
    ("after — calibrated plan", b1, 0, L.PALETTE[L.FOCUS_CHANNEL]),
]:
    fig.add_trace(go.Scatter(x=[row["optimal_share_p5"], row["optimal_share_p95"]], y=[yy, yy],
        mode="lines", line=dict(color=col, width=10), opacity=0.4, showlegend=False))
    fig.add_trace(go.Scatter(x=[row["optimal_share_pct"]], y=[yy], mode="markers+text",
        text=[label], textposition="top center", showlegend=False,
        marker=dict(color=col, size=14, line=dict(color="white", width=1.5))))
fig.update_yaxes(range=[-0.6, 1.7], visible=False)
L.style(fig, height=340, title=f"{L.FOCUS_CHANNEL}: the experiment sharpens the recommendation, not just the ROI")
fig.update_layout(xaxis_title=f"recommended budget share of {L.FOCUS_CHANNEL} (%)")
fig.show()

print(f"{L.FOCUS_CHANNEL} recommended-share band width:  before {width0:.1f}pp  ->  after {width1:.1f}pp")
print(f"uplift 90% band width:  before {L.dollars(res0.uplift_hdi[1] - res0.uplift_hdi[0])}"
      f"  ->  after {L.dollars(res.uplift_hdi[1] - res.uplift_hdi[0])}")
assert np.isfinite([res.expected_uplift, res0.expected_uplift]).all()   # both plans valid
"""),
    md(r"""
## The guardrailed default (for the deck)

The unconstrained optimum is the right answer for an analyst; it's a nervous one
for a slide. `default_reallocation` is the **conservative** plan that ships in a
client report: it caps every channel at **±20%** of its current spend, so **no
channel is switched off** and **no recommendation extrapolates** past the spend
range the model has actually seen. It's the move you can make on Monday without
anyone flinching — a smaller, safer step in the same direction.
"""),
    code(r"""
from mmm_framework.planning.budget import DEFAULT_REALLOC_DEVIATION

dr = default_reallocation(model, max_draws=100, random_seed=42)
alloc = pd.DataFrame(dr["allocation"]).set_index("channel").loc[list(curves.channel_names)]
alloc["delta_$"] = alloc["optimal_spend"] - alloc["current_spend"]
print(alloc[["current_spend", "optimal_spend", "delta_$", "change_pct"]].round(2).to_string())

print(f"\ncap per channel: ±{dr['deviation_cap']:.0%}  (no channel turned off)")
print(f"expected uplift {L.dollars(dr['expected_uplift'])}"
      f"   90% band [{L.dollars(dr['uplift_hdi'][0])}, {L.dollars(dr['uplift_hdi'][1])}]"
      f"   P(win) {dr['prob_positive_uplift']:.0%}")
assert abs(dr["deviation_cap"] - DEFAULT_REALLOC_DEVIATION) < 1e-9   # the conservative ±20% guardrail
"""),
    md(r"""
## Readout → a plan with a band, and a shelf life

T4 turns the calibrated model into what the CFO asked for: **a specific
reallocation, with a confidence band, that we can defend** — because the channel
it leans on was measured, not assumed. The optimizer moved money toward wherever
the **next dollar** works hardest, and folding in the experiment made that
recommendation **tighter**, not just different.

But this plan is a **snapshot**. Markets drift, creative fatigues, competitors
move — and the experiment that anchors Display **decays**. The last chapter,
**`lifecycle_06_reevaluate`** (T5), asks the closing question of the loop: *when
does this answer go stale, and what does that trigger a re-test of?* — handing us
back to **T1** and starting the cycle again.

*Previous: `lifecycle_04_calibrate` (T3 · Calibrate) — where we folded the
readout in.  Next: `lifecycle_06_reevaluate` (T5 · Re-evaluate) — when the answer
expires.*
"""),
]


def main() -> None:
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    out = "lifecycle/lifecycle_05_allocate.ipynb"
    with open(out, "w") as f:
        nbformat.write(nb, f)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    print(f"wrote {out} with {len(nb.cells)} cells ({n_code} code)")


if __name__ == "__main__":
    main()
