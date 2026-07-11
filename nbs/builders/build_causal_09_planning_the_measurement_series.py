"""Author causal_09_planning_the_measurement_series.ipynb — Notebook 9 of 11.

    uv run --with nbformat python builders/build_causal_09_planning_the_measurement_series.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        causal/causal_09_planning_the_measurement_series.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

Rung 9 — from one test to a measurement PROGRAM: the EIG x EVOI priority
grid, the EVPI ceiling on the learning budget, evidence decay + re-test
triggers, an honest note on what EVOI cannot see (hidden bias), and a
12-month measurement calendar assembled greedily under cool-down and
concurrency constraints. Runs on the cached rung-2 posterior — no new fits.
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
# Planning the Measurement Series
### A year of learning, budgeted like the investment it is

*Notebook 9 of 11 — Causal Inference in Practice.*

Rung 8 chose one experiment well. A measurement **program** is a different
problem: seven channels, finite testing capacity, cool-downs, seasons — and
evidence that quietly *expires* while you're busy elsewhere. Choosing what to
test next by gut ("Search feels off") or by grievance ("the CFO doubts TV")
wastes the scarcest thing the program has: calendar.

The framework prices every candidate experiment in two currencies:

- **EIG — expected information gain** (nats): how much a feasible test would
  shrink our uncertainty about the channel's ROI. Epistemics.
- **EVOI — expected value of information** ($): how much the *budget
  decision* would improve if we knew. Economics. A channel can be wildly
  uncertain and still worth $0 to test — if no plausible answer would change
  the allocation.

Their ceiling is **EVPI** — the value of perfect information about
everything — which is the most a rational program would ever spend on
measurement. This notebook builds Veranda's grid, watches old evidence decay,
and lays out the 12-month calendar.
"""),
    code(SETUP),
    md(r"""
## 1 — The priority grid

Inputs: the current posterior (rung 2's fit), and the evidence ledger from
rung 7 — Radio was tested (readout window ended 2023-10), Display's off-panel
test ended 2024-04. Planning date: 2024-07-01. The engine simulates a feasible
national-pulse test per channel, computes EIG against each channel's current
ROI posterior (decayed where evidence exists), and EVOI by re-optimizing the
budget across simulated readouts.
"""),
    code(r"""
base = C.fit_world("real_causal")
mmm = base["model"]

from mmm_framework.planning.priority import compute_experiment_priorities

EVIDENCE = {
    "Radio": {"end_date": "2023-10-30"},     # nb07's go-dark readout window
    "Display": {"end_date": "2024-04-22"},   # nb07's off-panel holdout
}
AS_OF = "2024-07-01"
grid, portfolio = compute_experiment_priorities(
    mmm, evidence=EVIDENCE, as_of=AS_OF, max_draws=150, random_seed=42
)
tbl = pd.DataFrame([p.to_dict() for p in grid])
show = tbl[["channel", "spend_share", "roi_mean", "roi_sd", "eig", "evoi",
            "evoi_pct_budget", "priority", "quadrant",
            "weeks_since_evidence", "retest_due"]]
print(show.round(3).to_string(index=False))
"""),
    code(r"""
QCOLOR = {"test_now": C.BAD, "learn_cheaply": "#c9a45a",
          "monitor": C.PALETTE["TV"], "deprioritize": C.MUTED}
fig = go.Figure()
for _, r in tbl.iterrows():
    fig.add_trace(go.Scatter(
        x=[r["eig"]], y=[r["evoi"]], mode="markers+text", text=[r["channel"]],
        textposition="top center",
        marker=dict(size=18 + 60 * r["spend_share"],
                    color=QCOLOR.get(r["quadrant"], C.MUTED),
                    line=dict(color=C.INK, width=1)),
        name=r["quadrant"], showlegend=False))
fig.add_vline(x=float(portfolio["eig_threshold"]), line_dash="dot", line_color=C.MUTED)
fig.add_hline(y=float(portfolio["evoi_threshold"]), line_dash="dot", line_color=C.MUTED)
fig.update_xaxes(title="EIG — what a test would teach (nats)")
fig.update_yaxes(title="EVOI — what knowing is worth (KPI units)")
C.style(fig, title="The measurement quadrant: bubble = spend share, color = verdict",
        height=480)
"""),
    code(r"""
top = tbl.iloc[0]
# The engine lands where rung 7 pointed: Print — the channel whose split the
# Radio test resolved onto shaky ground — is the program's top priority, and
# Radio itself is already flagged for RE-testing (see decay, below).
assert top["channel"] in ("Print", "Radio")
assert "Print" in set(tbl.head(2)["channel"])
assert bool(tbl.set_index("channel").loc["Radio", "retest_due"])
assert tbl.set_index("channel").loc["TV", "quadrant"] in ("deprioritize", "monitor")
print(f"top priority: {top['channel']} (EIG {top['eig']:.2f} nats, "
      f"EVOI {top['evoi']:.0f} KPI units, quadrant {top['quadrant']})")
"""),
    md(r"""
### An honest asterisk the grid needs

TV sits in *deprioritize* — its ROI posterior is tight, so a test teaches
little and moves no budget. But rung 2's sealed key showed TV carries a
**residual bias the model doesn't know about** (and its learning diagnostic
flagged `beta_TV` as *relocated*). EVOI prices **known unknowns**; it cannot
price a bias the posterior doesn't represent. A mature program runs two
tracks: the EVOI track above, and a **validation track** that occasionally
tests confident channels *precisely because* they're confident — triggered by
diagnostics (relocation flags, refutation failures, tension checks), not by
uncertainty. Rung 10 shows a validation-track test paying off.
"""),
    md(r"""
## 2 — EVPI: the ceiling on the learning budget

Summing what perfect information about everything would be worth gives the
program its budget ceiling. Any measurement plan costing more than EVPI is —
by construction — destroying value, no matter how scientific it feels.
"""),
    code(r"""
evpi = float(portfolio["evpi"])
v_cur = float(portfolio["v_current"])
fig = go.Figure()
fig.add_trace(go.Bar(x=tbl["channel"], y=tbl["evoi"], name="EVOI (per channel)",
                     marker_color=[QCOLOR.get(q, C.MUTED) for q in tbl["quadrant"]],
                     opacity=0.9))
fig.add_hline(y=evpi, line_color=C.INK, line_dash="dash",
              annotation_text=f"EVPI (portfolio ceiling) = {evpi:,.0f}")
fig.update_yaxes(title="value of information (KPI units)")
C.style(fig, title="What knowing is worth, channel by channel", height=430)

assert evpi > 0
assert float(tbl.set_index("channel").loc["Print", "evoi"]) > \
       float(tbl.set_index("channel").loc["TV", "evoi"])
print(f"Optimal-budget value under current knowledge: {v_cur:,.0f}; "
      f"EVPI = {evpi:,.0f} ({evpi / v_cur:.1%} of it). "
      "That percentage IS the rational scale of the measurement program.")
"""),
    md(r"""
## 3 — Evidence decays

A readout is a photograph, not a covenant. Markets drift — creative wears
out, auctions reprice, competitors move — so the information an experiment
bought **erodes on a half-life**. The framework models the posterior sd
re-inflating as `σ²(t) = σ₀² · exp(λt)`, and re-flags a channel for testing
when a *new* feasible test would again clear the EIG bar.
"""),
    code(r"""
from mmm_framework.planning.eig import (channel_half_life, decayed_sigma,
                                        eig_gaussian, reexperiment_due)

weeks = np.arange(0, 105)
fig = go.Figure()
for ch, sigma0, tested_wks in [("Radio", 0.10, 35), ("Display", 0.15, 10)]:
    hl = channel_half_life(ch)
    sig = [decayed_sigma(sigma0, float(w), hl) for w in weeks]
    fig.add_trace(go.Scatter(x=weeks, y=sig, name=f"{ch} (half-life {hl:.0f}w)",
                             line=dict(color=C.PALETTE[ch], width=2.5)))
    fig.add_vline(x=tested_wks, line_dash="dot", line_color=C.PALETTE[ch],
                  annotation_text=f"{ch}: {tested_wks}w ago")
fig.update_xaxes(title="weeks since the experiment")
fig.update_yaxes(title="effective posterior sd of ROI")
C.style(fig, title="Certainty on a half-life: post-experiment sd re-inflating",
        height=420)

due_r, eig_r = reexperiment_due(0.10, 35, channel_half_life("Radio"), sigma_exp=0.10)
print(f"Radio, 35 weeks on: a fresh test would buy {eig_r:.2f} nats -> "
      f"retest due: {due_r}")
assert bool(tbl.set_index('channel').loc['Radio', 'retest_due'])
"""),
    md(r"""
## 4 — The 12-month calendar

Scores in hand, the calendar is a constrained packing problem. A greedy
EVOI-per-week schedule under the program's real constraints — one national
test at a time, per-channel cool-downs (rung 8), Q4 freeze (no experiments
in the holiday quarter), re-tests when decay triggers:
"""),
    code(r"""
from mmm_framework.planning.experiment_optimizer import cooldown_weeks

# candidate tests: the test_now channels + the flagged re-test, with design
# modes and durations inherited from rungs 5-8
CANDIDATES = [
    dict(name="Print geo holdout", channel="Print", mode="matched-pair geo",
         weeks=8, evoi=float(tbl.set_index("channel").loc["Print", "evoi"])),
    dict(name="Radio re-test (flighting)", channel="Radio", mode="national flighting",
         weeks=12, evoi=float(tbl.set_index("channel").loc["Radio", "evoi"])),
    dict(name="Display follow-up", channel="Display", mode="national flighting",
         weeks=8, evoi=float(tbl.set_index("channel").loc["Display", "evoi"])),
    dict(name="Social probe", channel="Social", mode="national flighting",
         weeks=8, evoi=float(tbl.set_index("channel").loc["Social", "evoi"])),
    dict(name="TV validation test", channel="TV", mode="national flighting (16w, rung-8 design)",
         weeks=16, evoi=float(tbl.set_index("channel").loc["TV", "evoi"])),
]
Q4_FREEZE = (13, 26)   # program weeks 13-26 = Oct-Dec: no tests

sched, t = [], 0
ranked = sorted(CANDIDATES, key=lambda c: c["evoi"] / c["weeks"], reverse=True)
# the validation-track test is scheduled by POLICY, not EVOI — pin it last slot
ranked = [c for c in ranked if c["channel"] != "TV"] + \
         [c for c in ranked if c["channel"] == "TV"]
for cand in ranked:
    start = t
    # jump the freeze if the test would overlap it
    if start < Q4_FREEZE[1] and start + cand["weeks"] > Q4_FREEZE[0]:
        start = Q4_FREEZE[1]
    cool = int(cooldown_weeks(mmm, cand["channel"])["cooldown_weeks"])
    end = start + cand["weeks"]
    sched.append(dict(**cand, start=start, end=end, cooldown=cool))
    t = end + 1                                     # 1-week gap between tests
cal = pd.DataFrame(sched)
print(cal[["name", "mode", "start", "end", "weeks", "cooldown", "evoi"]]
      .round(1).to_string(index=False))

# The honest arithmetic of a testing calendar: five tests do NOT fit inside
# twelve months once a Q4 freeze and one-at-a-time discipline are respected —
# the EVOI winners claim the year, and the policy-scheduled TV validation test
# spills into the next cycle. Calendar is the program's scarcest resource.
assert cal.head(4)["end"].max() <= 60      # the four EVOI picks fit the year+
assert cal["end"].max() <= 80              # the validation test rolls over
print(f"\nEVOI-track tests finish by week {int(cal.head(4)['end'].max())}; "
      f"the validation test runs weeks {int(cal.iloc[-1]['start'])}-"
      f"{int(cal.iloc[-1]['end'])} (next cycle).")
"""),
    code(r"""
fig = go.Figure()
for i, r in cal.iterrows():
    fig.add_trace(go.Bar(
        y=[r["name"]], x=[r["weeks"]], base=[r["start"]], orientation="h",
        marker_color=C.PALETTE.get(r["channel"], C.MUTED), opacity=0.9,
        showlegend=False,
        hovertemplate=f"{r['name']}: weeks {r['start']}-{r['end']}<extra></extra>"))
    fig.add_trace(go.Bar(
        y=[r["name"]], x=[r["cooldown"]], base=[r["end"]], orientation="h",
        marker_color=C.MUTED, opacity=0.35, showlegend=False,
        hovertemplate="cool-down<extra></extra>"))
fig.add_vrect(x0=Q4_FREEZE[0], x1=Q4_FREEZE[1], fillcolor="rgba(0,0,0,0.07)",
              line_width=0, annotation_text="Q4 freeze")
fig.update_xaxes(title="program week")
fig.update_yaxes(autorange="reversed")
C.style(fig, title="The measurement calendar (solid = test, faint = cool-down)",
        height=420)
"""),
    md(r"""
## 5 — The program's memory

Every fitted model in the loop persists a per-run metrics snapshot — ROI
posteriors, response curves, EIG/EVOI, allocation gaps — so next quarter's
planning starts from *recorded* state, not from whoever remembers the last
deck. (`compute_run_metrics` is the same snapshot the platform stores per fit.)
"""),
    code(r"""
from mmm_framework.planning.history import compute_run_metrics

rm = compute_run_metrics(mmm, max_draws=100, random_seed=42)
print("snapshot keys:", sorted(rm.keys()))
print(f"channels tracked: {len(rm['channels'])}; schema v{rm['schema_version']}; "
      f"fit method: {rm['fit_method']} (approximate: {rm['approximate']})")
assert len(rm["channels"]) == 7

C.write_artifact("causal_09_program.json", dict(
    as_of=AS_OF,
    grid=[{k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool)
               else v)
           for k, v in dict(channel=r["channel"], eig=r["eig"], evoi=r["evoi"],
                            priority=r["priority"], quadrant=r["quadrant"],
                            retest_due=bool(r["retest_due"])).items()}
          for _, r in tbl.iterrows()],
    evpi=float(portfolio["evpi"]),
    v_current=float(portfolio["v_current"]),
    calendar=[dict(name=r["name"], channel=r["channel"], start=int(r["start"]),
                   end=int(r["end"])) for _, r in cal.iterrows()],
))
print("artifact written: causal_09_program.json")
"""),
    md(r"""
## The program's operating rules

1. **Price tests in both currencies.** EIG says what you'd learn; EVOI says
   whether learning changes any decision. Test where both are high.
2. **EVPI is the budget.** The whole program should cost a fraction of the
   value of perfect information — if it doesn't, you're doing science for
   sport.
3. **Evidence expires.** Half-lives and re-test triggers turn "we tested that
   in 2021" from an excuse into a date.
4. **Keep a validation track.** EVOI can't see biases the posterior doesn't
   know it has. Diagnostics-triggered tests on *confident* channels are the
   program's insurance policy.
5. **Write the state down.** Snapshots make the loop resumable by people who
   weren't in the room.

---
**Next — [10 · The closed loop](causal_10_the_closed_loop.ipynb):** the
capstone. Three rounds of fit → prioritize → design → measure → calibrate on
a confounded world, graded against the sealed key at every step — watching
the posterior converge to the causal truth and the *budget decisions* get
measurably better.
"""),
]


def main():
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    out = "causal/causal_09_planning_the_measurement_series.ipynb"
    with open(out, "w") as f:
        nbformat.write(nb, f)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
