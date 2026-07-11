"""Author causal_08_designing_the_next_experiment.ipynb — Notebook 8 of 11.

    uv run --with nbformat python builders/build_causal_08_designing_the_next_experiment.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        causal/causal_08_designing_the_next_experiment.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

Rung 8 — the fitted model joins the design loop: model-anchored predictions of
what a candidate test would read, the short-term cost of learning
(budget-neutral flighting ≈ free), the Pareto choice across
MDE x power x cost x duration, the design engine saying NO to an underpowered
national Print test, adstock cooldowns, and what it takes to identify the
saturation CURVE rather than just the effect. No new fits — everything runs
against the cached rung-2 posterior.
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
# Designing the Next Experiment
### The model stops being the patient and becomes the surgeon's assistant

*Notebook 8 of 11 — Causal Inference in Practice.*

Rung 5 designed a test using only the raw geo panel. That was deliberate — an
experiment's *validity* must never depend on the model it will later repair.
But its *efficiency* should absolutely use the model. A fitted posterior
knows things a bare panel doesn't:

- **what the test will probably read** (the model-anchored effect — so you
  don't run a test whose likely readout is smaller than its MDE);
- **what the test costs in the short term** (posterior counterfactual of the
  test schedule vs business-as-usual);
- **how long carryover contaminates the next test** on the same channel
  (adstock washout → cool-down);
- **whether a schedule identifies the *curve*** (saturation, carryover
  parameters) or merely the average effect.

Veranda's post-rung-7 to-do list has two names on it: **TV** (the stubborn
residual under-read) and **Print** (the twin that inherited the pair-sum
bias). We design for both — and the engine will give them *opposite*
verdicts. All of this runs against the cached rung-2 posterior; designing an
experiment costs posterior passes, not refits.
"""),
    code(SETUP),
    md(r"""
## 1 — Candidate one: a TV flighting test

National data → the recommended design family is **budget-neutral
flighting**: alternate spend between a high and a low level in randomized
blocks. Same annual budget, honest identification from the *contrast*.
"""),
    code(r"""
base = C.fit_world("real_causal")
mmm = base["model"]
path = C.mff_csv("real_causal")

from mmm_framework.planning.design import design_experiment

d_tv = design_experiment(path, "Sales", "TV")
print(f"design: {d_tv['design_key']} | {d_tv['duration']} weeks | "
      f"levels {d_tv['levels']} | budget-neutral: {d_tv['budget_neutral']}")
print(f"design MDE: {d_tv['mde_roas']:.2f} ROAS")
assert d_tv["design_key"] == "national_flighting" and d_tv["budget_neutral"]
"""),
    md(r"""
## 2 — What would we actually see? The model-anchored readout

The design's MDE says what's *detectable*. The posterior says what's
*expected*: push the candidate schedule through the fitted response curves
(spend perturbation on the test window only) and read the distribution of
incremental ROAS the test would measure. If the expected readout sits below
the MDE, the test as specified is a coin flip — better to know now.
"""),
    code(r"""
from mmm_framework.planning.design_anchor import model_anchored_effect

anch = model_anchored_effect(mmm, d_tv, max_draws=60, random_seed=42)
draws = np.asarray(anch["incremental_roas_draws"], float)
med, (lo, hi) = anch["incremental_roas_median"], anch["incremental_roas_hdi"]
print(f"model-anchored incremental ROAS: median {med:.2f}, 90% HDI [{lo:.2f}, {hi:.2f}]")

fig = go.Figure(go.Histogram(x=draws, nbinsx=30, marker_color=C.PALETTE["TV"],
                             opacity=0.85))
fig.add_vline(x=d_tv["mde_roas"], line_color=C.BAD, line_dash="dash",
              annotation_text=f"default design MDE ({d_tv['mde_roas']:.2f})")
fig.add_vline(x=med, line_color=C.INK, annotation_text="expected readout")
fig.update_xaxes(title="incremental ROAS the test would measure")
C.style(fig, title="What the default 12-week TV flighting would read", height=400)

assert np.isfinite(draws).all()
assert 0 < med < 1.5
assert d_tv["mde_roas"] > med   # the DEFAULT design is underpowered — fix below
print("Expected readout sits BELOW the default design's MDE — as specified, "
      "this test would probably return 'inconclusive'. The optimizer's job "
      "is to fix exactly this.")
"""),
    md(r"""
## 3 — What does learning cost? (Almost nothing, if you design it that way)

The posterior counterfactual of running the schedule vs business-as-usual:
KPI forgone, dollars at risk, probability of a net loss over the window.
Budget-neutral flighting is the free lunch of measurement — the same dollars,
rearranged.
"""),
    code(r"""
from mmm_framework.planning.opportunity_cost import compute_opportunity_cost

oc = compute_opportunity_cost(mmm, d_tv, max_draws=60, random_seed=42,
                              kpi_kind="revenue")
annual_tv = float(base["sc"].spend["TV"].sum())
print(f"net spend delta over the window : {oc.spend_delta:+.1f} "
      f"({oc.spend_delta / annual_tv:+.1%} of annual TV — randomized blocks "
      "against a varying baseline round to *approximately* neutral)")
print(f"dollars at risk                 : {oc.spend_at_risk:,.0f}")
print(f"expected KPI delta              : {oc.expected_kpi_delta:+.1f} "
      f"(median {oc.kpi_delta_median:+.1f})")
print(f"P(KPI loss over window)         : {oc.prob_kpi_loss:.0%}")
assert oc.spend_at_risk == 0.0
assert abs(oc.spend_delta) < 0.03 * annual_tv
"""),
    md(r"""
## 4 — The Pareto choice

Candidate designs trade four objectives — **MDE ↓, power shortfall ↓,
short-term cost ↓, duration ↓** — and no single design wins all four.
`suggest_experiment` sweeps the design space (intensity × duration), keeps
the non-dominated frontier, and recommends the knee among *powered* designs.
"""),
    code(r"""
from mmm_framework.planning.experiment_optimizer import suggest_experiment

sug = suggest_experiment(mmm, path, "Sales", "TV",
                         duration_min=6, duration_max=16,
                         intensity_min=30.0, intensity_max=100.0,
                         max_draws=40, random_seed=42)
cand = pd.DataFrame(sug["candidates"])
cols = [c for c in ("design_key", "intensity_pct", "duration", "mde_roas",
                    "power_shortfall", "tradeoff") if c in cand.columns]
print(cand[cols].round(3).to_string(index=False))
rec = sug["recommended"]
print(f"\nrecommended: ±{rec['intensity_pct']:.0f}% intensity, {rec['duration']} weeks "
      f"-> MDE {rec['mde_roas']:.2f} ROAS, power shortfall {rec['power_shortfall']:.2f}")
assert len(sug["pareto"]) >= 1
assert rec["power_shortfall"] < 0.10
assert 6 <= rec["duration"] <= 16
"""),
    code(r"""
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=cand["duration"], y=cand["mde_roas"], mode="markers+text",
    text=[f"±{i:.0f}%" for i in cand["intensity_pct"]], textposition="top center",
    marker=dict(size=[26 if i == sug["recommended_index"] else 14 for i in range(len(cand))],
                color=[C.GOOD if ps < 0.1 else C.BAD for ps in cand["power_shortfall"]],
                line=dict(color=C.INK, width=1)),
    showlegend=False))
fig.add_hline(y=float(anch["incremental_roas_median"]), line_dash="dot", line_color=C.INK,
              annotation_text="expected readout")
fig.update_xaxes(title="duration (weeks)")
fig.update_yaxes(title="MDE (ROAS)")
C.style(fig, title="The design frontier — green designs are powered; the big dot is the pick",
        height=430)
"""),
    md(r"""
## 5 — The engine's most valuable answer: "no"

Now Print — the channel nb07 flagged as the top re-test priority. Print's
national spend is a rounding error of TV's, so a national budget-neutral
flighting contrast moves almost no dollars, and its MDE is an order of
magnitude above any plausible ROAS:
"""),
    code(r"""
sug_p = suggest_experiment(mmm, path, "Sales", "Print",
                           duration_min=6, duration_max=16,
                           intensity_min=50.0, intensity_max=100.0,
                           max_draws=40, random_seed=42)
cand_p = pd.DataFrame(sug_p["candidates"])
print(cand_p[[c for c in ("intensity_pct", "duration", "mde_roas", "power_shortfall")
              if c in cand_p.columns]].round(2).to_string(index=False))
rec_p = sug_p["recommended"]
print(f"\nbest available: MDE {rec_p['mde_roas']:.1f} ROAS, "
      f"power shortfall {rec_p['power_shortfall']:.2f}")
"""),
    code(r"""
# Every national candidate is hopelessly underpowered for Print. The honest
# recommendation is NOT 'run the least-bad one' — it's CHANGE THE MODE:
# a matched-pair geo holdout (rung 5's machinery) concentrates the contrast
# in a subset of markets, or measure the Radio+Print PAIR jointly.
assert rec_p["power_shortfall"] > 0.3
print("The design engine just saved a quarter's Print budget from buying an "
      "'inconclusive'. A test you shouldn't run is a result too.")
"""),
    md(r"""
## 6 — Cool-downs: carryover contaminates the calendar

Two tests on the same channel back-to-back are not independent: the first
test's adstock is still paying out during the second's window. The fitted
carryover implies a **washout period** per channel — the minimum gap before
the next test (or the minimum block length inside a flighting schedule).
"""),
    code(r"""
from mmm_framework.planning.experiment_optimizer import cooldown_weeks

cools = {ch: cooldown_weeks(mmm, ch) for ch in ("TV", "Search", "Print")}
cool_tbl = pd.DataFrame([
    dict(channel=ch, adstock_alpha=round(c["alpha"], 2),
         half_life_weeks=round(c["half_life"], 2),
         cooldown_weeks=c["cooldown_weeks"])
    for ch, c in cools.items()
])
print(cool_tbl.to_string(index=False))
assert 2 <= cools["TV"]["cooldown_weeks"] <= 8
"""),
    md(r"""
## 7 — Identifying the curve, not just the effect

Everything above measures *an average effect*. Budget **allocation** needs
the *curve* — saturation shape and carryover — and rung 2 showed those
parameters barely move under observational data. A multi-level flighting
schedule (≥3 spend levels) is the design that can, in principle, trace the
curve. `identify_structural_parameters` computes, per structural parameter,
the power and the expected prior→posterior contraction the schedule would
buy — Fisher-information math against the fitted posterior, with a fail-closed
self-check that the forward op mirrors the graph.
"""),
    code(r"""
from mmm_framework.agents.model_ops import identify_structural_parameters

rows = []
ident = {}
for dur in (12, 24):
    r = identify_structural_parameters(mmm, dataset_path=path, kpi="Sales",
                                       channel="TV", duration=dur, random_seed=42)
    assert r["error"] is None
    pay = r["dashboard"]["structural_identification"]
    assert pay["self_check"]["passed"]
    st = pay["structural"]
    ident[dur] = st
    for p, v in st["params"].items():
        rows.append(dict(duration=dur, param=p, power=v["power"],
                         contraction=v["contraction"]))
ident_tbl = pd.DataFrame(rows).pivot_table(index="param", columns="duration",
                                           values=["power", "contraction"])
print(ident_tbl.round(4).to_string())
print(f"\nidentifies_anything: 12w={ident[12]['identifies_anything']}, "
      f"24w={ident[24]['identifies_anything']}")
"""),
    code(r"""
# The honest arithmetic of curve identification: even a well-built 12-week
# 3-level schedule DETECTS the effect with high power, yet barely contracts
# the structural priors — and only the 24-week commitment crosses the
# engine's identification bar. Curvature knowledge is expensive; buy it
# deliberately or don't claim it.
assert not ident[12]["identifies_anything"]
assert ident[24]["identifies_anything"]
for p in ("beta", "lam"):
    assert ident[24]["params"][p]["contraction"] > ident[12]["params"][p]["contraction"]
print("Measure effects cheaply and often; measure curves rarely and on purpose.")
"""),
    code(r"""
C.write_artifact("causal_08_design.json", dict(
    tv_default=dict(mde_roas=float(d_tv["mde_roas"]),
                    expected_readout=float(anch["incremental_roas_median"])),
    tv_recommended=dict(intensity_pct=float(rec["intensity_pct"]),
                        duration=int(rec["duration"]),
                        mde_roas=float(rec["mde_roas"]),
                        power_shortfall=float(rec["power_shortfall"])),
    print_verdict=dict(best_mde_roas=float(rec_p["mde_roas"]),
                       power_shortfall=float(rec_p["power_shortfall"])),
    opportunity=dict(spend_at_risk=float(oc.spend_at_risk),
                     prob_kpi_loss=float(oc.prob_kpi_loss)),
    cooldown_weeks={ch: int(c["cooldown_weeks"]) for ch, c in cools.items()},
    identification=dict(w12=bool(ident[12]["identifies_anything"]),
                        w24=bool(ident[24]["identifies_anything"])),
))
print("artifact written: causal_08_design.json")
"""),
    md(r"""
## The design checklist this rung leaves behind

1. **Anchor before you run.** If the model's expected readout < the design's
   MDE, redesign or don't run.
2. **Prefer budget-neutral contrasts** — flighting turns measurement into a
   rearrangement of dollars you were spending anyway.
3. **Let the engine say no.** An underpowered mode (national Print) is a
   changed design, not a smaller ambition.
4. **Respect cool-downs.** Carryover makes the testing calendar a scheduling
   problem, not a wishlist.
5. **Decide which you're buying: the effect or the curve.** They cost
   different orders of magnitude.

---
**Next — [09 · Planning the measurement series](causal_09_planning_the_measurement_series.ipynb):**
from one test to a *program*: which channels deserve experiments at all, in
what order, on what calendar — EIG, EVOI, the EVPI ceiling, and evidence that
decays while you're not looking.
"""),
]


def main():
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    out = "causal/causal_08_designing_the_next_experiment.ipynb"
    with open(out, "w") as f:
        nbformat.write(nb, f)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
