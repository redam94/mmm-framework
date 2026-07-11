"""Author causal_01_confounding_and_adjustment.ipynb — Notebook 1 of 11.

    uv run --with nbformat python builders/build_causal_01_confounding_and_adjustment.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        causal/causal_01_confounding_and_adjustment.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

Rung 1 of the ladder: back-door adjustment. Three hidden histories of the same
brand (demand hidden / demand measured / budgets paced to sales), what a
control variable actually buys, why the door only closes as far as the proxy
is good, and the one disease no control can cure. All fits are the shared
cached numpyro posteriors (~15s each cold, instant warm).
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
# Confounding and Adjustment
### What a control variable actually buys you

*Notebook 1 of 11 — Causal Inference in Practice.*

In [notebook 00](causal_00_the_ladder.ipynb) Veranda Home's dashboard—and a
genuinely good Bayesian MMM—over-credited the two channels whose budgets chase
demand. The disease was an **open back-door**: a common cause (latent demand)
driving both spend and sales.

The classic prescription is **adjustment**: find a variable that *measures* the
common cause, condition on it, and the spurious correlation is blocked. This
notebook tests that prescription the only way that settles anything — against
sealed answer keys — across **three possible hidden histories** of the same
brand:

- **History A — demand hidden.** The confounder exists, and nothing on the data
  warehouse measures it. (`unobserved_confounding`)
- **History B — demand measured.** Same world, but a syndicated *category
  demand index* lands in the warehouse — a noisy proxy for the confounder.
  (`confounding_controlled`)
- **History C — budgets paced to sales.** No hidden third cause at all;
  instead, finance sets next week's budgets as a share of recent revenue —
  spend and sales cause *each other*. (`reverse_causality`)

Three rules to take away, each demonstrated below:

1. Conditioning on a real confounder **closes the door** — most of the bias
   disappears.
2. The door only closes **as far as the proxy is good** — a noisy measurement
   of demand leaves residual bias, and the model won't warn you.
3. Adjustment is powerless against **simultaneity** — when spend *responds to*
   sales, there is no third variable to condition on; the back-door runs
   through the KPI itself.
"""),
    code(SETUP),
    md(r"""
## 1 — Three DAGs, one dashboard

Draw the causal graph of each history. The dashboards these worlds generate are
statistically similar — positive spend-sales correlations everywhere. The
*graphs* are what differ, and the graph is what decides whether adjustment can
work. (Red arrows are the paths that transmit non-causal correlation.)
"""),
    code(r"""
C.dag(
    nodes={"latent demand": (0.5, 1.0), "spend": (0.0, 0.0), "sales": (1.0, 0.0)},
    edges=[("spend", "sales")],
    bad_edges=[("latent demand", "spend"), ("latent demand", "sales")],
    title="History A — demand hidden: the back-door is OPEN",
    node_colors={"latent demand": "#e8d5a3", "spend": "#cfe0f0", "sales": "#d5e8d4"},
)
"""),
    code(r"""
C.dag(
    nodes={"demand": (0.5, 1.0), "demand index": (0.92, 1.0),
           "spend": (0.0, 0.0), "sales": (1.0, 0.0)},
    edges=[("spend", "sales"), ("demand", "demand index")],
    bad_edges=[("demand", "spend"), ("demand", "sales")],
    title="History B — demand measured: condition on the index, close the door",
    node_colors={"demand": "#e8d5a3", "demand index": "#f2e6c8",
                 "spend": "#cfe0f0", "sales": "#d5e8d4"},
)
"""),
    code(r"""
C.dag(
    nodes={"spend (this week)": (0.0, 0.0), "sales (this week)": (1.0, 0.0),
           "sales (last week)": (0.5, 1.0)},
    edges=[("spend (this week)", "sales (this week)")],
    bad_edges=[("sales (last week)", "spend (this week)")],
    title="History C — budget pacing: the 'confounder' is the KPI itself",
    node_colors={"sales (last week)": "#e8d5a3",
                 "spend (this week)": "#cfe0f0", "sales (this week)": "#d5e8d4"},
)
"""),
    md(r"""
## 2 — Fit all three histories

Same model family for each (Bayesian MMM: geometric adstock, saturating
response, linear trend, numpyro NUTS), fit once and cached by the series. In
History B the demand index enters **marked as a `CONFOUNDER`** — the framework
routes it to a wide, un-shrunk coefficient prior, because shrinking a
confounder's coefficient re-opens the very door you're trying to close.
"""),
    code(r"""
fits = {k: C.fit_world(k) for k in ("uc", "cc", "rc")}
for k, label in [("uc", "A — demand hidden"), ("cc", "B — demand measured"),
                 ("rc", "C — budget pacing")]:
    print(f"\n=== History {label} ===")
    print(fits[k]["grade"][["true", "est", "rel_err", "covered"]].round(2).to_string())
"""),
    md(r"""
## 3 — Grading: what the control bought

Histories A and B are the **same world** — same latent demand, same chasing
budgets, same true ROAS. The only difference is one column in the analyst's
dataset. That makes the comparison surgical: any improvement is attributable
to adjustment and nothing else.
"""),
    code(r"""
chasers = ["Search", "Social"]
g_uc, g_cc = fits["uc"]["grade"], fits["cc"]["grade"]

rows = []
for c in g_uc.index:
    rows.append({"channel": c, "history": "A — demand hidden",
                 "rel_err": g_uc.loc[c, "rel_err"]})
    rows.append({"channel": c, "history": "B — demand measured",
                 "rel_err": g_cc.loc[c, "rel_err"]})
cmp = pd.DataFrame(rows)

fig = go.Figure()
for h, color in [("A — demand hidden", C.BAD), ("B — demand measured", C.GOOD)]:
    sub = cmp[cmp["history"] == h]
    fig.add_trace(go.Bar(x=sub["channel"], y=sub["rel_err"], name=h,
                         marker_color=color, opacity=0.85))
fig.add_hline(y=0, line_color=C.INK, line_width=1)
fig.update_layout(barmode="group")
fig.update_yaxes(title="relative error vs sealed truth", tickformat="+.0%")
C.style(fig, title="One extra column: attribution error, before and after adjustment",
        height=420)
"""),
    code(r"""
# Rule 1: conditioning on the measured confounder shrinks the chasers' bias.
for c in chasers:
    assert abs(g_cc.loc[c, "rel_err"]) < abs(g_uc.loc[c, "rel_err"]), c
# Social's interval now covers the truth (it didn't in History A).
assert g_cc.loc["Social", "covered"] and not g_uc.loc["Social", "covered"]

# Rule 2: the door is only MOSTLY closed. The index is a noisy proxy for true
# demand, so residual confounding survives — Search is still over-read, and
# nothing inside the model announces it.
assert g_cc.loc["Search", "rel_err"] > 0.10
print(f"Search bias: {g_uc.loc['Search','rel_err']:+.0%} (hidden) -> "
      f"{g_cc.loc['Search','rel_err']:+.0%} (measured). "
      "Better — not gone. A proxy closes the door only as far as it is a good proxy.")
"""),
    md(r"""
### Aside — does the `CONFOUNDER` marking matter?

The framework asks you to *declare* a control's causal role, because unmarked
controls get the default precision-control prior (narrower, shrunk toward
zero), and shrinking a confounder can leave the door half-open. Here's the same
History B fit with the index left **unmarked**:
"""),
    code(r"""
g_un = C.fit_world("cc_unmarked")["grade"]
aside = pd.DataFrame({
    "A — no control": g_uc["rel_err"],
    "B — marked CONFOUNDER": g_cc["rel_err"],
    "B — unmarked (shrunk prior)": g_un["rel_err"],
}).loc[["TV", "Search", "Social", "Display"]]
print(aside.map("{:+.0%}".format).to_string())

# In THIS world the index's signal is strong enough to survive the default
# shrinkage — both variants close the door about equally. The honest reading:
# marking the role is cheap insurance, not magic. It matters exactly when the
# confounder's coefficient is small enough for shrinkage to bite; you don't
# know in advance whether that's your world.
for c in chasers:
    assert abs(g_un.loc[c, "rel_err"]) < abs(g_uc.loc[c, "rel_err"]), c
"""),
    md(r"""
## 4 — History C: the disease adjustment can't touch

In History C there is no hidden demand at all. Budgets are **paced**: finance
sets next week's spend partly from recent revenue. Spend and sales are jointly
determined — the arrow into spend starts at the KPI itself.

No control variable can fix this. Conditioning works by blocking a path
through a *third* variable; here the offending path runs through last week's
**outcome**. (Conditioning on lagged sales does not rescue you either — lagged
sales carries the adstocked effect of last week's media, so it's a bad control
on the causal path.) The honest fixes live further up the ladder: model the
feedback structurally, or **experiment** — randomization severs the arrow into
spend by construction.

How bad is it here? Veranda paces gently (a fraction of budget tracks
revenue), and the media calendar keeps its identifying pulses, so the damage is
moderate — visible, directional, and invisible from inside the model:
"""),
    code(r"""
g_rc = fits["rc"]["grade"]
C.truth_forest(g_rc, title="History C — budget pacing: the paced channels drift", height=380)
"""),
    code(r"""
# The hardest-paced channel (Search, pacing 1.0) is the most over-read; the
# bias pattern follows the pacing intensities, not the media plan.
assert g_rc["rel_err"].idxmax() == "Search"
assert g_rc.loc["Search", "rel_err"] > 0.05
print("Pacing intensities:", fits["rc"]["sc"].notes["pacing"])
print(f"Search over-read by {g_rc.loc['Search','rel_err']:+.0%} — "
      "and every diagnostic (R-hat, fit quality, intervals) looks healthy.")
"""),
    md(r"""
## 5 — The adjustment scorecard

Averaging |relative error| over the demand-chasing channels (the ones
confounding actually attacks):
"""),
    code(r"""
def chaser_score(g):
    return float(np.mean([abs(g.loc[c, "rel_err"]) for c in chasers]))

score = pd.Series({
    "A — demand hidden": chaser_score(g_uc),
    "B — index, marked": chaser_score(g_cc),
    "B — index, unmarked": chaser_score(g_un),
}, name="mean |rel err| (chasers)")

fig = go.Figure(go.Bar(
    x=score.index, y=score.values,
    marker_color=[C.BAD, C.GOOD, "#7fae8a"], opacity=0.9,
    text=[f"{v:.0%}" for v in score.values], textposition="outside"))
fig.update_yaxes(title="mean |relative error|, Search & Social", tickformat=".0%",
                 range=[0, max(score.values) * 1.3])
C.style(fig, title="What one honest control variable buys", height=400)
"""),
    code(r"""
assert score["A — demand hidden"] > score["B — index, marked"]

C.write_artifact("causal_01_adjustment_scorecard.json", dict(
    chasers=chasers,
    chaser_score={k: float(v) for k, v in score.items()},
    search_rel_err=dict(hidden=float(g_uc.loc["Search", "rel_err"]),
                        marked=float(g_cc.loc["Search", "rel_err"]),
                        unmarked=float(g_un.loc["Search", "rel_err"])),
    rc_rel_err={c: float(g_rc.loc[c, "rel_err"]) for c in g_rc.index},
    rhat_max={k: float(fits[k]["diag"]["rhat_max"]) for k in fits},
))
print("artifact written: causal_01_adjustment_scorecard.json")
"""),
    md(r"""
## What this rung bought — and what it can't

**Bought:** with a measured confounder, most of the chasers' bias disappears.
Adjustment is real, cheap, and the first thing to reach for. Declare causal
roles when you do it.

**Can't:**

- an **unmeasured** confounder (History A) leaves you exactly where the
  dashboard left you — and nothing inside the model tells you;
- a **noisy proxy** closes the door partway and, again, doesn't tell you how
  far;
- **simultaneity** (History C) has no door to close.

Both failures share a signature: *the model is confident and wrong.* The rest
of the series is about earning confidence honestly — first by modeling more of
the structure we believe in ([03 — structural mediation](causal_03_structural_mediation.ipynb),
[04 — latent confounders](causal_04_latent_confounders.ipynb)), then by buying
ground truth with experiments (05–07).

---
**Next — [02 · The MMM as a causal model](causal_02_mmm_as_causal_model.ipynb):**
the realistic seven-channel world; what an MMM's numbers actually *claim*
(estimands), the bad-control trap, refutation tests, and the model's most
underrated skill — saying "I don't know."
"""),
]


def main():
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    out = "causal/causal_01_confounding_and_adjustment.ipynb"
    with open(out, "w") as f:
        nbformat.write(nb, f)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
