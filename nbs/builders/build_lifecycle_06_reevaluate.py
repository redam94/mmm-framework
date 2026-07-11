"""Author lifecycle_06_reevaluate.ipynb — T5 (the finale) of the Experimental
Measurement Lifecycle series (run from ``nbs/``).

    uv run --with nbformat python builders/build_lifecycle_06_reevaluate.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat \
        --with ipykernel jupyter nbconvert --to notebook --execute --inplace \
        lifecycle/lifecycle_06_reevaluate.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

**T5 · Re-evaluate — when the answer goes stale.** Evidence ages: as weeks pass
since Display's experiment, the model's certainty about it DECAYS at a
channel-specific half-life until a fresh test is worth running again — which
re-points the whole loop back to T1. This chapter closes the cycle: the decay
sweep + two-part re-test gate, re-prioritizing with experiment dates fed back in,
and a "measurement program" one-pager the analyst brings to the media team.
Shared world / fit / palette live in ``nbs/builders/lifecycle_common.py``.
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

# Appended to the setup cell: load the baseline model this stage reasons over.
LOAD = r"""

# T5 reasons about the model's *certainty over time*. We load the cached baseline
# (T0) — the object whose posterior the decay clock erodes — and name the channel
# the loop has carried since T1.
base = L.fit_baseline()
model = base["model"]
ch = L.FOCUS_CHANNEL
print(f"loaded baseline — the loop's carried channel is {ch} | KPI: {L.KPI}")
print("channels:", ", ".join(model.channel_names))
assert ch in model.channel_names       # the channel we've followed T1 -> T4 is real
"""


CELLS = [
    md(r"""
# T5 · Re-evaluate — when does the answer go stale?
### Notebook 6 of 7 — the finale of the Experimental Measurement Lifecycle

**Recap (from `lifecycle_05_allocate`).** T4 handed the CFO a real answer: a
reallocation with a confidence band, leaning on **Display** because T3 *measured*
it rather than assumed it. But that plan is a **snapshot**. Creative fatigues,
auctions shift, competitors move — and the single most perishable thing in the
whole plan is the **experiment** that anchors Display. A readout is a photograph,
not a subscription.

So T5 asks the closing question of the loop:

> **When does a measured answer go stale — and what does that trigger a re-test of?**

The mechanism is **information decay**. A fresh experiment pins a channel's return
tightly; as weeks pass, the *effective* uncertainty grows back at a
channel-specific **half-life** (fast auction surfaces forget in ~6 months; stable
broadcast holds a year). We track the **value of running a fresh test** as that
certainty erodes, and when it climbs back over an operational bar, the channel is
**due** — which re-points the loop straight back to **T1 · Prioritize**. That is
what makes this a *loop* and not a report: it never ends, it **re-prioritizes**.
"""),
    code(SETUP + LOAD),
    md(r"""
## Evidence has a shelf life

Picture a single channel the day after its experiment lands. Its return is pinned
to a tight **posterior standard deviation** — call it `sigma_post`. The framework
models certainty as *decaying* from there: the effective uncertainty grows as
`sigma_eff²(t) = sigma_post² · exp(λ·t)` with `λ = ln2 / half-life`, so the
channel "forgets" its experiment on a schedule set by its **class** (Display is a
fast digital surface → a ~26-week half-life).

We score staleness by the **EIG of a *fresh* test** — how many **nats** a new
experiment of realistic precision would now buy. Crucially, the achievable design
precision `sigma_exp` is **grounded in the design** (footprint, geo vs national),
**not** derived from `sigma_post` — otherwise every channel would look equally
worth testing and the whole ranking would collapse. As certainty decays, a fresh
test buys *more*, so this EIG **rises** with time. The re-test rule is a **two-part
gate**:

1. **decayed EIG ≥ threshold** (`DEFAULT_RETEST_THRESHOLD_NATS`) — a fresh test
   would teach enough to be worth the money; **and**
2. **age ≥ freshness floor** (`MIN_RETEST_AGE_WEEKS`) — a channel tested last
   month isn't "stale", it's "recently measured and still uncertain"; re-running
   an identical test immediately just buys the same photograph twice.
"""),
    code(r"""
from mmm_framework.planning.eig import (
    reexperiment_due, decayed_sigma, channel_half_life,
    DEFAULT_RETEST_THRESHOLD_NATS, MIN_RETEST_AGE_WEEKS,
)

hl = channel_half_life(ch)              # weeks — Display is a fast digital surface
sigma_post, sigma_exp = 0.10, 0.30      # post-experiment ROI sd; DESIGN-achievable sd (independent of sigma_post)
weeks = np.arange(0, 120)
eig = np.array([reexperiment_due(sigma_post, w, hl, sigma_exp)[1] for w in weeks])   # decayed EIG (nats)
due = np.array([reexperiment_due(sigma_post, w, hl, sigma_exp)[0] for w in weeks])   # both gates pass?
first_due = int(weeks[due][0]) if due.any() else None

fig = go.Figure()
# The freshness floor: too soon to re-test, no matter how wide the posterior gets.
fig.add_vrect(x0=0, x1=MIN_RETEST_AGE_WEEKS, fillcolor=L.MUTED, opacity=0.10, line_width=0,
              annotation_text="freshness floor", annotation_position="top left")
fig.add_trace(go.Scatter(x=weeks, y=eig, mode="lines", name="value of a fresh test (EIG)",
    line=dict(color=L.PALETTE[ch], width=3)))
fig.add_hline(y=DEFAULT_RETEST_THRESHOLD_NATS, line=dict(color=L.BAD, width=2, dash="dash"),
              annotation_text=f"re-test bar = {DEFAULT_RETEST_THRESHOLD_NATS} nats",
              annotation_position="bottom right")
if first_due is not None:
    fig.add_vline(x=first_due, line=dict(color=L.GOOD, width=2, dash="dot"),
                  annotation_text=f"due at ~{first_due} wks", annotation_position="top right")
    fig.add_trace(go.Scatter(x=[first_due], y=[eig[first_due]], mode="markers", showlegend=False,
        marker=dict(color=L.GOOD, size=13, line=dict(color="white", width=1.5))))
L.style(fig, title=f"{ch}: evidence ages — a fresh test is worth more the longer we wait",
        legend=dict(orientation="h", yanchor="top", y=-0.16, x=0))
fig.update_layout(xaxis_title="weeks since the experiment", yaxis_title="EIG of a fresh test (nats)")
fig.show()

print(f"{ch} half-life: {hl:.0f} wks  |  freshness floor: {MIN_RETEST_AGE_WEEKS:.0f} wks")
print(f"post-test ROI sd {sigma_post} inflates to {decayed_sigma(sigma_post, first_due, hl):.3f} by week {first_due}")
print(f"BOTH gates clear at ~{first_due} weeks -> that is when a re-test earns its keep.")
assert first_due is not None and first_due >= MIN_RETEST_AGE_WEEKS   # the floor binds; a re-test is never premature
"""),
    md(r"""
**Readout →** the value of a fresh test starts *below* the bar (we just measured
Display — nothing to re-learn yet) and rises as certainty erodes, clearing the bar
only after both gates pass. That crossing is not a curiosity — it's a **trigger**.
When a channel goes due, the loop doesn't sit on it; it hands the channel back to
**T1** and re-runs the priority grid *with the experiment history folded in*.
"""),
    md(r"""
## Re-prioritize — feed the experiment dates back into T1

Here is the loop actually closing. We call the **same** `compute_experiment_priorities`
grid from `lifecycle_02_prioritize` — but now we pass an **evidence log**: when each
channel was last measured. The grid decays each channel's certainty against
**today**, re-derives the decayed EIG, and flags who is **due for a re-test**.

To make the point vivid we give the log two very different entries: **Display**,
just measured (its T3 experiment), and **TV**, measured long ago. A channel with
**no** evidence has never been tested, so its decay clock hasn't started
(`weeks_since_evidence` / `eig_decayed` come back `NaN`, `retest_due` `False`) —
it's a candidate for a *first* test, not a *re*-test.
"""),
    code(r"""
from mmm_framework.planning.priority import compute_experiment_priorities

# The experiment registry, as ISO end-dates. Display was just measured (T3);
# TV's readout is old. Search/Social have never been tested.
evidence = {L.FOCUS_CHANNEL: {"end_date": "2026-06-01"},   # fresh — the T3 experiment
            "TV": {"end_date": "2024-06-01"}}              # stale — measured two years ago

grid2, port2 = compute_experiment_priorities(
    model, evidence=evidence, as_of="2026-07-01", max_draws=300, random_seed=42)
df2 = pd.DataFrame([g.to_dict() for g in grid2])
print("grid columns:", list(df2.columns))     # print live so the readout adapts if a field is renamed

di2 = df2.set_index("channel")
view = ["roi_mean", "roi_sd", "weeks_since_evidence", "eig_decayed", "retest_due", "quadrant"]
print("\n" + di2.loc[L.CHANNELS, view].round(4).to_string())

# Decayed EIG for the two channels that HAVE an evidence clock, against the re-test bar.
ev_df = df2[df2["eig_decayed"].notna()].copy().sort_values("eig_decayed")
fig = go.Figure()
fig.add_trace(go.Bar(
    x=ev_df["channel"], y=ev_df["eig_decayed"],
    marker_color=[(L.BAD if d else L.GOOD) for d in ev_df["retest_due"]],
    text=[f"{w:.0f} wks old<br>{'DUE' if d else 'fresh'}"
          for w, d in zip(ev_df["weeks_since_evidence"], ev_df["retest_due"])],
    textposition="outside", showlegend=False))
fig.add_hline(y=port2["retest_threshold_nats"], line=dict(color=L.INK, width=2, dash="dash"),
              annotation_text="re-test bar", annotation_position="top left")
L.style(fig, height=420, title="Decayed value of a fresh test — old evidence clears the bar, fresh evidence does not")
fig.update_layout(xaxis_title="", yaxis_title="decayed EIG (nats)")
fig.show()

due_now = df2[df2["retest_due"]]["channel"].tolist()
print(f"\nflagged DUE for a re-test: {due_now or 'none'}")
print(f"{L.FOCUS_CHANNEL} (just measured): retest_due = {bool(di2.loc[L.FOCUS_CHANNEL, 'retest_due'])} "
      f"(fresh — inside the {MIN_RETEST_AGE_WEEKS:.0f}-wk freshness floor)")
assert bool(di2.loc[L.FOCUS_CHANNEL, "retest_due"]) is False   # the channel we just tested is NOT due
assert "TV" in due_now                                         # the loop re-points to the stale channel
"""),
    md(r"""
**Readout →** the loop **re-points**. The channel we just measured (Display) is
*fresh* and drops off the test list; the channel with old evidence (TV) has decayed
past the bar and is now the one worth a fresh experiment. Notice what T5 did *not*
do: it didn't declare victory and stop. It quietly re-opened T1 on a **different**
channel. That is the cycle turning over — the same machinery that first sent us to
Display now sends us somewhere new.
"""),
    md(r"""
## The measurement program — one page for the media team

Everything the loop knows, on a single page. This is the artifact the analyst
actually brings to the room: per channel, the **current ROI** (Display's is now
the *calibrated* number — anchored to its experiment, not guessed from history),
its **status** in the priority quadrant, **when it was last tested**, and the **one
next action** the loop recommends — *test it, monitor it, hold it,* or *re-test in
~N weeks* (computed from exactly the same decay gate as the chart above). It reads
top-to-bottom as a standing operating plan, not a one-off analysis.
"""),
    code(r"""
from mmm_framework.reporting.helpers.roi import compute_roi_with_uncertainty
from mmm_framework.planning.eig import channel_half_life, reexperiment_due

# Display's ROI reflects its folded-in experiment (T3) -> use the CALIBRATED model.
cal = L.fit_calibrated(verbose=False)
roi_cal = compute_roi_with_uncertainty(cal["model"]).set_index("channel")

def _next_action(name: str) -> str:
    r = di2.loc[name]
    if name not in evidence:                       # never tested -> action follows the quadrant
        return {"test_now": "run the first test", "monitor": "monitor for drift",
                "learn_cheaply": "learn cheaply", "deprioritize": "hold"}[r["quadrant"]]
    if bool(r["retest_due"]):                       # tested, but decayed past the bar
        return "re-test now (stale)"
    hl_n = channel_half_life(name)                  # tested & fresh -> when will it go stale?
    w0 = float(r["weeks_since_evidence"])
    nxt = next((k for k in range(int(np.ceil(w0)), 400)
                if reexperiment_due(float(r["roi_sd"]), k, hl_n, float(r["sigma_exp"]))[0]), None)
    return "fresh — hold" if nxt is None else f"re-test in ~{int(round(nxt - w0))} wks"

prog = pd.DataFrame([{
    "Channel": c,
    "Current ROI": f"{float(roi_cal.loc[c, 'roi_mean']):.2f}x",
    "Status": di2.loc[c, "quadrant"],
    "Last tested": evidence[c]["end_date"] if c in evidence else "never",
    "Wks since": ("—" if pd.isna(di2.loc[c, "weeks_since_evidence"])
                  else f"{float(di2.loc[c, 'weeks_since_evidence']):.0f}"),
    "Next action": _next_action(c),
} for c in L.CHANNELS])
print(prog.to_string(index=False))

zebra = ["#f5f8fa" if i % 2 else "white" for i in range(len(prog))]
tint = {c: L.PALETTE[c] for c in L.CHANNELS}
fig = go.Figure(go.Table(
    columnwidth=[70, 78, 96, 88, 62, 130],
    header=dict(values=[f"<b>{c}</b>" for c in prog.columns], fill_color=L.INK,
                font=dict(color="white", size=13), align="left", height=32),
    cells=dict(values=[prog[c] for c in prog.columns],
               fill_color=[[tint[c] for c in prog["Channel"]]] + [zebra] * (prog.shape[1] - 1),
               font=dict(color=[["white"] * len(prog)] + [[L.INK] * len(prog)] * (prog.shape[1] - 1), size=12),
               align="left", height=30)))
L.style(fig, height=260, title=f"{L.BRAND} — the measurement program (T5 standing plan)")
fig.show()
print(f"\n({L.FOCUS_CHANNEL}'s ROI is experiment-anchored; the rest are the observational fit.)")
assert len(prog) == len(L.CHANNELS) and set(prog["Channel"]) == set(L.CHANNELS)   # one row per channel
"""),
    md(r"""
## The loop closes

That's the whole cycle, run end-to-end on one brand. Each stage answered one
operating question and handed the next its input — and the last stage handed the
*first* its trigger:

| Stage | What we did on Northwind | Notebook |
|---|---|---|
| **T0 · Fit** | Fit the baseline MMM — honest but **possibly biased** ROIs. | `lifecycle_01_fit_baseline` |
| **T1 · Prioritize** | EIG × EVOI singled out **Display**: uncertain *and* decision-critical. | `lifecycle_02_prioritize` |
| **T2 · Design** | Sized a **powered, affordable** geo holdout to measure it. | `lifecycle_03_design` |
| **T3 · Calibrate** | Folded the readout in — Display's ROI is now **measured**. | `lifecycle_04_calibrate` |
| **T4 · Allocate** | Re-optimized under uncertainty — a plan **with a band**. | `lifecycle_05_allocate` |
| **T5 · Re-evaluate** | Watched the evidence **decay**; re-pointed the loop to **TV**. | `lifecycle_06_reevaluate` |

**T5 doesn't end the loop — it re-enters it at T1.** A measured answer has a shelf
life, so the moment Display's evidence decays (or TV's already has), the priority
grid re-opens on whatever is now most worth learning, and the cycle turns again.
There is no final report; there is a **standing program**.
"""),
    md(r"""
### The honest caveats

Three things worth saying plainly, because the loop only works if you believe them:

- **A fit is a hypothesis, not an answer.** The T0 model's credible intervals are
  honest about the uncertainty the *model* can see — never about the **confounding**
  baked into observational history. That gap is why we test at all (nb00's opening
  picture).
- **Experiments are the ground truth the loop is organized around.** Everything —
  the priority grid, the calibration, the allocation band — exists to spend a
  scarce testing budget where a real-world measurement will move the decision most.
- **The loop never "finishes."** Markets drift and evidence decays, so certainty
  is a *depreciating asset*. The win isn't a permanent number; it's a **process**
  that always knows what to measure next.

### Series recap

Six stages, one brand, one fitted model carried the whole way — and no number
hardcoded in prose, every claim pinned by a live `assert`. You now have the full
map *and* every step walked on real data.

*Previous: `lifecycle_05_allocate` (T4 · Allocate) — the plan with a band. To start
the tour over from the top: `lifecycle_00_overview` — the map of the whole loop.*

**— fin.**
"""),
]


def main() -> None:
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    out = "lifecycle/lifecycle_06_reevaluate.ipynb"
    with open(out, "w") as f:
        nbformat.write(nb, f)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    print(f"wrote {out} with {len(nb.cells)} cells ({n_code} code)")


if __name__ == "__main__":
    main()
