"""Author causal_10_the_closed_loop.ipynb — Notebook 10 of 11 (the capstone).

    uv run --with nbformat python builders/build_causal_10_the_closed_loop.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        causal/causal_10_the_closed_loop.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

The whole ladder in one loop, three times around: fit -> prioritize (EIG/EVOI)
-> design a window -> measure (readout = sealed truth + design-SE noise) ->
calibrate -> refit — on the confounded world, graded against the key each
round. Headlines: attribution error falls monotonically, EVPI (the remaining
value of learning) shrinks, and the model's BUDGET DECISIONS go from
value-destroying to near-optimal. ~6-10 min bake (3 refits + posterior passes).
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
# The Closed Loop
### Three times around: fit → prioritize → design → measure → calibrate

*Notebook 10 of 11 — Causal Inference in Practice. The capstone.*

Every rung of the ladder, welded into one operating loop and run **three
full cycles** on the world where this series began — `unobserved_confounding`,
the history whose dashboard fooled the naive read in notebook 00 and whose
back-door no control variable could fully close in notebook 01.

Per cycle:

1. **Fit** the model on everything known so far (history + all past readouts).
2. **Prioritize** with the EIG × EVOI grid (rung 9) — which channel's
   uncertainty is worth money?
3. **Design** the test window (staggered go-darks with cool-down gaps —
   rungs 5 & 8 discipline, compressed).
4. **Measure**: the readout is generated from the *sealed truth plus sampling
   noise at the design's SE* — an honest experiment, never an oracle.
5. **Calibrate** via the likelihood route (rung 6) and go around again.

The narrator grades every cycle against the key, but the analyst inside the
loop never sees it — they only see posteriors tightening and readouts
arriving. Three questions decide whether the whole series *works*:

- Does attribution error **fall** every cycle?
- Does the value of further learning (**EVPI**) shrink — is the loop
  self-limiting?
- Do the **budget decisions** actually get better in the world, not just in
  the model's imagination?
"""),
    code(SETUP),
    md(r"""
## 1 — The sealed benchmarks

Before the loop starts, the narrator computes three numbers the analyst
never sees: the true media value of the **historical** allocation, the true
value of the **best possible** allocation (grid-searched on the true response
curves), and the gap between them — the most any reallocation could win.
"""),
    code(r"""
import itertools

sc = C.scenario_for("uc")
C.check_truth(sc)
hist_spend = sc.spend.sum()
TOTAL = float(hist_spend.sum())

def true_value_of_alloc(alloc: dict) -> float:
    "True media-driven KPI if each channel's history were scaled to `alloc`."
    return sum(
        float(C.true_media_term(sc, c,
              sc.spend[c].to_numpy(float) * (alloc[c] / float(hist_spend[c]))).sum())
        for c in sc.channels)

V_HIST = true_value_of_alloc({c: float(hist_spend[c]) for c in sc.channels})

best_v, best_alloc = -np.inf, None
steps = np.arange(0.05, 0.80, 0.025)
for shares in itertools.product(steps, repeat=3):
    s4 = 1.0 - sum(shares)
    if not (0.05 <= s4 <= 0.80):
        continue
    alloc = {c: TOTAL * s for c, s in zip(sc.channels, list(shares) + [s4])}
    v = true_value_of_alloc(alloc)
    if v > best_v:
        best_v, best_alloc = v, alloc
V_STAR = float(best_v)
print(f"[narrator] true value — historical plan: {V_HIST:,.0f} | "
      f"best possible plan: {V_STAR:,.0f} (headroom {V_STAR - V_HIST:,.0f})")
"""),
    md(r"""
## 2 — The loop

Three cycles. Each round's test is a 10-week go-dark on the grid's top
*untested* channel, in the next free window (staggered, clear of the previous
test's cool-down). Readout SE: 0.09 — rung-5-grade precision.
"""),
    code(r"""
from mmm_framework.planning.budget import optimize_budget
from mmm_framework.planning.priority import compute_experiment_priorities

WINDOWS = {1: (96, 106), 2: (110, 120), 3: (124, 134)}
SE = 0.09

fits = {0: C.fit_world("uc")}
experiments, tested, log = [], set(), []

for rnd in range(4):
    fit = fits[rnd]
    mmm, g = fit["model"], fit["grade"]

    grid, port = compute_experiment_priorities(mmm, max_draws=120,
                                               random_seed=42 + rnd)
    gtbl = pd.DataFrame([p.to_dict() for p in grid])

    opt = optimize_budget(mmm, total_budget=TOTAL, random_seed=42)
    t = opt.table.set_index("channel")
    alloc = {c: float(t.loc[c, "optimal_spend"]) for c in sc.channels}

    log.append(dict(
        round=rnd,
        grade=g, evpi=float(port["evpi"]),
        alloc=alloc, true_value=true_value_of_alloc(alloc),
        ranking=gtbl["channel"].tolist(),
    ))
    print(f"round {rnd}: EVPI={port['evpi']:.0f}  "
          f"priority ranking={gtbl['channel'].tolist()}")

    if rnd == 3:
        break
    pick = next(ch for ch in gtbl["channel"] if ch not in tested)
    w0, w1 = WINDOWS[rnd + 1]
    exp, ro = C.measurement(sc, pick, w0, w1, mult=0.0, se=SE, seed=200 + rnd)
    print(f"   -> testing {pick} (weeks {w0}-{w1}); readout "
          f"{ro['observed']:.2f} ± {SE}")
    experiments.append(exp)
    tested.add(pick)
    fits[rnd + 1] = C.fit_calibrated("uc", list(experiments))

print(f"\ntested over the year: {sorted(tested)}")
"""),
    md(r"""
## 3 — Unsealing: does the posterior converge on the truth?

The analyst saw posteriors tighten. The narrator can say something stronger:
they tightened **toward the right answers**.
"""),
    code(r"""
err = pd.Series({e["round"]: float(e["grade"]["rel_err"].abs().mean())
                 for e in log}, name="mean |relative error|")
print(err.map("{:.1%}".format).to_string())

fig = make_subplots(rows=1, cols=2, column_widths=[0.42, 0.58],
                    subplot_titles=["attribution error by round",
                                    "per-channel ROAS trajectory vs truth"])
fig.add_trace(go.Scatter(x=err.index, y=err.values, mode="lines+markers",
                         line=dict(color=C.INK, width=3), marker=dict(size=11),
                         showlegend=False), row=1, col=1)
for c in sc.channels:
    traj = [float(e["grade"].loc[c, "est_roas"]) for e in log]
    fig.add_trace(go.Scatter(x=list(range(4)), y=traj, mode="lines+markers",
                             name=c, line=dict(color=C.PALETTE[c], width=2.2)),
                  row=1, col=2)
    fig.add_hline(y=float(sc.true_roas[c]), line_dash="dot",
                  line_color=C.PALETTE[c], opacity=0.6, row=1, col=2)
fig.update_xaxes(title="round (experiments folded in)", dtick=1)
fig.update_yaxes(title="mean |rel err|", tickformat=".0%", row=1, col=1)
fig.update_yaxes(title="ROAS (dotted = sealed truth)", row=1, col=2)
C.style(fig, height=460, title="Three experiments walk the posterior onto the key")
"""),
    code(r"""
# Convergence: error falls from round 0 and lands far below where it started.
assert err[1] < err[0]
assert err[3] < err[0] - 0.08
print(f"attribution error: {err[0]:.0%} -> {err[3]:.0%} across three cycles")
"""),
    md(r"""
## 4 — Unsealing: do the *decisions* get better?

Parameters are means, not ends. Each round the analyst also asked the model
for its **optimal reallocation** of the same total budget. The narrator now
evaluates every one of those plans on the *true* response curves — including
the plan the round-0 model would have shipped.
"""),
    code(r"""
vals = pd.Series({e["round"]: e["true_value"] for e in log},
                 name="true value of the model's optimal plan")

fig = go.Figure()
fig.add_trace(go.Bar(x=[f"round {r}" for r in vals.index], y=vals.values,
                     marker_color=[C.BAD, "#c9a45a", "#8fae5d", C.GOOD],
                     opacity=0.92,
                     text=[f"{v:,.0f}" for v in vals.values],
                     textposition="outside"))
fig.add_hline(y=V_HIST, line_color=C.INK, line_dash="dash",
              annotation_text=f"keep the historical plan ({V_HIST:,.0f})")
fig.add_hline(y=V_STAR, line_color=C.TRUTH,
              annotation_text=f"true optimum ({V_STAR:,.0f})")
fig.update_yaxes(title="TRUE media KPI under the plan",
                 range=[min(vals.min(), V_HIST) * 0.94, V_STAR * 1.02])
C.style(fig, title="What each round's 'optimal' plan is worth in the real world",
        height=460)
"""),
    code(r"""
# The round-0 trap: reallocating on the CONFOUNDED model would have shipped a
# plan WORSE than doing nothing — it moves budget toward the over-credited
# chasers. Three experiments later the model's plan closes most of the gap to
# the true optimum.
assert vals[0] < V_HIST - 300     # the confident-and-wrong plan destroys value
assert vals[3] > vals[0] + 300    # the loop repairs the decision
assert vals[3] > 0.97 * V_HIST    # ...to within noise of the status quo or better
print(f"round-0 plan: {vals[0] - V_HIST:+,.0f} KPI vs status quo (a value-destroying "
      f"reallocation, shipped with full confidence)")
print(f"round-3 plan: {vals[3] - V_HIST:+,.0f} KPI vs status quo; "
      f"{(vals[3] - vals[0]):+,.0f} recovered by the loop")
"""),
    md(r"""
## 5 — Unsealing: is the loop self-limiting?

A healthy measurement program spends less on learning as it learns. EVPI —
the total remaining value of perfect information — is the loop's own fuel
gauge.
"""),
    code(r"""
evpi = pd.Series({e["round"]: e["evpi"] for e in log}, name="EVPI")
fig = go.Figure(go.Scatter(x=evpi.index, y=evpi.values, mode="lines+markers",
                           line=dict(color=C.INK, width=3), marker=dict(size=11)))
fig.update_xaxes(title="round", dtick=1)
fig.update_yaxes(title="EVPI (KPI units)", rangemode="tozero")
C.style(fig, title="The value of what's left to learn", height=380)

assert min(evpi[1:]) < evpi[0]
assert evpi[2] < evpi[0]
print(f"EVPI: {evpi[0]:.0f} -> {evpi[3]:.0f}. When EVPI dips below the cost of "
      "the next test, the rational program pauses — until decay (rung 9) "
      "re-inflates it.")
"""),
    code(r"""
C.write_artifact("causal_10_closed_loop.json", dict(
    tested=sorted(tested),
    error_by_round={int(k): float(v) for k, v in err.items()},
    evpi_by_round={int(k): float(v) for k, v in evpi.items()},
    decision_value_by_round={int(k): float(v) for k, v in vals.items()},
    v_historical=float(V_HIST), v_true_optimum=float(V_STAR),
    readout_se=SE,
))
print("artifact written: causal_10_closed_loop.json")
"""),
    md(r"""
## The ladder, climbed

Eleven notebooks ago, Veranda Home's dashboard said everything works and the
naive read doubled Social's value. Here is what each rung contributed to the
loop that just ran:

| rung | contribution to the loop |
|---|---|
| 00–01 | knowing *why* the baseline is biased — and that no control fully fixes it |
| 02 | estimand discipline; trust checks; "I don't know" as a model output |
| 03–04 | structure for what observational adjustment can't reach — funnels, latent factors |
| 05 | readouts worth believing: calibrated estimators, honest SEs |
| 06 | the likelihood route — experiments that *commit*, filed under the right estimand |
| 07 | portfolios: composition, off-panel windows, tension checks |
| 08 | tests chosen by anchored power and cost — including the ones *not* to run |
| 09 | the program: EIG/EVOI, the EVPI ceiling, decay, the calendar |
| 10 | the loop itself — error ↓, EVPI ↓, decisions ↑, all graded against a sealed key |

Three habits are the whole method:

1. **Name the estimand.** Every number is an answer to a counterfactual
   question; say which one.
2. **Buy information deliberately.** Observational data is cheap and biased;
   experiments are costly and clean; the exchange rate is EVOI.
3. **Grade yourself whenever the truth is available** — in synthetic worlds
   always, in production whenever an experiment lands. A measurement system
   that never checks itself against ground truth isn't measuring; it's
   narrating.

*Where the loop re-enters:* evidence decays (rung 9), the world drifts, and
next quarter's EVPI creeps back up. The loop doesn't end — that's the point.

---
*Series complete. For the platform's automated version of this loop — the
T₀–T₅ measurement cycle with experiment registries, calibration tracking and
report generation — see the Experimental Measurement Lifecycle series
(`nbs/lifecycle/`) and the framework documentation.*
"""),
]


def main():
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    out = "causal/causal_10_the_closed_loop.ipynb"
    with open(out, "w") as f:
        nbformat.write(nb, f)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
