"""Author causal_05_measuring_one_experiment.ipynb — Notebook 5 of 11.

    uv run --with nbformat python builders/build_causal_05_measuring_one_experiment.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        causal/causal_05_measuring_one_experiment.ipynb --ExecutePreprocessor.timeout=1800 \
        --ExecutePreprocessor.kernel_name=python3

Rung 5 — experiment measurement methodology, before any model touches it:
matched-pair geo design on Veranda's 10 test markets, the A/A discipline
(false-positive rates under autocorrelation, design-calibrated critical
values), injected-truth A/B power, and the methodology leaderboard. Pure
pandas — no MCMC anywhere in this notebook.
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
# Measuring One Experiment
### Reading a lift test honestly is a skill, not a formality

*Notebook 5 of 11 — Causal Inference in Practice.*

Rungs 1–4 pushed observational modeling as far as it goes and hit the same
wall every time: a residual the data cannot settle. The cure is an
**experiment** — randomization severs every back-door by construction. But an
experiment only *buys* truth if it's read honestly, and the field is littered
with lift tests that weren't:

- treatment markets picked by convenience (the big ones), not by matching;
- significance declared with a t-test whose assumptions the data violates;
- estimators chosen *after* seeing the data, because one of them "found" lift.

Veranda Home has ten test markets and wants to measure **Search** — the
channel every observational rung kept over-crediting. Before a single dollar
moves, this notebook does the three pieces of homework a pre-registration
demands, all on **pre-period data only**:

1. **Design** — who gets treated, who's the control, and what's detectable
   (`planning.design`: matched randomized pairs, placebo-calibrated power);
2. **Estimator calibration** — run every candidate estimator on *null* data
   (**A/A tests**) and measure its actual false-positive rate;
3. **Power against injected truth** — plant known lifts in real history
   (**A/B simulation**) and verify the chosen estimator finds them.

Everything here is pure pandas on the geo panel — no MCMC, no MMM. The model
re-enters in [notebook 06](causal_06_calibrating_the_model.ipynb), *after* the
readout can be trusted.
"""),
    code(SETUP),
    md(r"""
## 1 — Ten markets, one question

The brand at DMA grain (`geo_heterogeneous` world): ten markets with their own
baselines, budget shares, and — because this world is honest about geography —
their own true regional effectiveness. The answer key stays sealed until the
end, as always.
"""),
    code(r"""
path, truth = C.geo_csv()
raw = pd.read_csv(path)
kpi = raw[raw["VariableName"] == "Sales"]
wide = kpi.pivot_table(index="Period", columns="Geography", values="VariableValue")
wide.index = pd.to_datetime(wide.index)

fig = go.Figure()
for gname in wide.columns:
    fig.add_trace(go.Scatter(x=wide.index, y=wide[gname], name=gname,
                             line=dict(width=1.3), opacity=0.8))
fig.update_yaxes(title="weekly sales (KPI units)")
C.style(fig, title="Veranda Home — ten test markets, weekly sales", height=460)
print(f"{wide.shape[1]} markets x {wide.shape[0]} weeks")
"""),
    md(r"""
## 2 — Design: matched pairs, then a coin flip

`geo_lift_design` builds the treatment geometry a referee would accept:

- **matched pairs** — markets paired by pre-period KPI similarity (residualized
  against spend), so each treated market carries its own counterfactual twin;
- **randomization inside each pair** — a coin flip decides which twin gets
  treated, severing any lingering association between assignment and demand;
- **placebo-calibrated power** — the design's minimum detectable effect (MDE)
  comes from re-running the analysis on hundreds of *pre-period placebo
  windows*, not from a textbook variance formula.
"""),
    code(r"""
from mmm_framework.planning.design import design_options, geo_lift_design

opts = design_options(path, "Sales", "Search")
print(f"designs the data supports: {opts['designs']} (recommended: {opts['recommended']})")

design = geo_lift_design(path, "Sales", "Search", design="holdout",
                         duration=8, seed=42)
pairs = pd.DataFrame({"treated": design["treatment_geos"],
                      "control": design["control_geos"]})
print(pairs.to_string(index=False))
print(f"\n8-week holdout | design MDE: {design['mde_roas']:.2f} ROAS "
      f"| readout SE: {design['se_roas']:.3f} "
      f"| placebo windows: {design['placebo']['n_windows']}")
assert design["randomized"] and design["n_pairs"] >= 4
"""),
    md(r"""
## 3 — The A/A discipline: test your test on nothing

Before trusting any estimator with a real experiment, run it on **null data**:
hundreds of rolling windows in which *no experiment happened*, asking how
often it cries "significant!" anyway. Weekly KPI series are autocorrelated —
adjacent weeks share trend, season, and demand shocks — so the *effective*
sample size is far below the nominal one, and naive standard errors are too
small.

Three candidate estimators, same frozen assignment:
"""),
    code(r"""
from mmm_framework.planning.simulation import (
    build_sim_panel, build_geo_assignment, run_aa_simulation,
    pooled_did_estimator, per_pair_did_estimator, regadj_geo_estimator,
)

panel = build_sim_panel(path, "Sales", "Search")
assignment = build_geo_assignment(panel, seed=42)

ESTIMATORS = {
    "pooled DiD": pooled_did_estimator,
    "per-pair DiD": per_pair_did_estimator,
    "regression-adjusted": regadj_geo_estimator,
}
aa = {}
for nm, est in ESTIMATORS.items():
    aa[nm] = run_aa_simulation(panel, est, assignment, duration=8, seed=42, name=nm)

aa_tbl = pd.DataFrame([
    dict(estimator=nm,
         windows=r.n_windows,
         fpr_naive=r.fpr_at_nominal,
         fpr_calibrated=r.fpr_at_crit,
         inflated=r.fpr_inflated)
    for nm, r in aa.items()
])
print(aa_tbl.round(3).to_string(index=False))
"""),
    code(r"""
fig = go.Figure()
fig.add_trace(go.Bar(x=aa_tbl["estimator"], y=aa_tbl["fpr_naive"],
                     name="naive t-test", marker_color=C.BAD, opacity=0.88))
fig.add_trace(go.Bar(x=aa_tbl["estimator"], y=aa_tbl["fpr_calibrated"],
                     name="design-calibrated critical value", marker_color=C.GOOD,
                     opacity=0.88))
fig.add_hline(y=0.05, line_dash="dot", line_color=C.INK,
              annotation_text="nominal α = 5%")
fig.update_layout(barmode="group")
fig.update_yaxes(title="false-positive rate on NULL data", tickformat=".0%")
C.style(fig, title="A/A tests: how often each estimator finds an effect that isn't there",
        height=430)
"""),
    code(r"""
# The regression-adjusted estimator — the most 'sophisticated' of the three —
# is the one that lies: its naive false-positive rate is several times the
# nominal 5%. Meanwhile the design-calibrated critical value (the empirical
# 95th percentile of |placebo estimates|) restores honest size for EVERY
# estimator, by construction.
assert aa["regression-adjusted"].fpr_at_nominal > 0.15
assert aa["regression-adjusted"].fpr_inflated
for nm in ESTIMATORS:
    assert 0.0 <= aa[nm].fpr_at_crit <= 0.12
print("Sophistication is not validity. Calibrate on placebo history, "
      "pre-register the critical value, THEN run the test.")
"""),
    md(r"""
## 4 — Power against injected truth

The A/A test protects against false positives; the **A/B simulation** measures
power against false *negatives*. Take real history, inject a **known** lift
into the treated markets across a grid of effect sizes, and count how often
each (calibrated) estimator detects it. Because the injected lift is known,
this is the estimator graded against a sealed key — the same epistemics as the
rest of the series, at simulation speed.
"""),
    code(r"""
from mmm_framework.planning.simulation import fixed_lift_injector, run_ab_simulation

injector = fixed_lift_injector(panel, assignment, duration=8, pct_of_baseline=0.10)
ab = {}
for nm in ("pooled DiD", "per-pair DiD"):
    ab[nm] = run_ab_simulation(panel, ESTIMATORS[nm], assignment, injector,
                               duration=8, aa_result=aa[nm], seed=42, name=nm)

rows = []
for nm, r in ab.items():
    for e in r.per_effect:
        rows.append(dict(estimator=nm, scale=e.get("scale"), power=e.get("power")))
power_tbl = pd.DataFrame(rows).dropna()
print(power_tbl.pivot_table(index="scale", columns="estimator",
                            values="power").round(2).to_string())

fig = go.Figure()
for nm, color in [("pooled DiD", C.PALETTE["TV"]), ("per-pair DiD", C.PALETTE["Search"])]:
    sub = power_tbl[power_tbl["estimator"] == nm].sort_values("scale")
    fig.add_trace(go.Scatter(x=sub["scale"], y=sub["power"],
                             mode="lines+markers", name=nm,
                             line=dict(color=color, width=2.5)))
fig.add_hline(y=0.8, line_dash="dot", line_color=C.INK, annotation_text="80% power")
fig.update_xaxes(title="injected lift (multiple of the expected effect)")
fig.update_yaxes(title="empirical power (calibrated test)", tickformat=".0%")
C.style(fig, title="Injected-truth power curves", height=430)
"""),
    code(r"""
# At the expected effect size (a 10%-of-baseline lift), both calibrated DiD
# estimators are decisively powered.
for nm, r in ab.items():
    print(f"{nm}: power at expected effect = {r.power_at_expected:.0%} ({r.status})")
assert ab["per-pair DiD"].power_at_expected > 0.8
assert ab["pooled DiD"].power_at_expected > 0.8
"""),
    md(r"""
## 5 — The leaderboard: one honest recommendation

`methodology_leaderboard` runs the full A/A + A/B battery for every estimator
the data supports, on one frozen assignment, and ranks by **validity first**
(honest false-positive rate), then power, then cost. The output is the
sentence a pre-registration needs: *this design, this estimator, this critical
value.*
"""),
    code(r"""
from mmm_framework.planning.simulation import methodology_leaderboard

lb = methodology_leaderboard(path, "Sales", "Search", duration=8, seed=42)
lb_tbl = pd.DataFrame(lb["methodologies"])
show_cols = [c for c in ("key", "valid", "fpr", "power", "mde_roas", "n_windows")
             if c in lb_tbl.columns]
print(lb_tbl[show_cols].round(3).to_string(index=False))
print(f"\nrecommended methodology: {lb['chosen_key']}")
for cv in lb.get("caveats", []):
    print(" caveat:", cv)
"""),
    code(r"""
# A DiD variant must win: valid size and strong power on this panel. The
# regression-adjusted estimator must NOT win (its A/A size is broken here).
assert lb["chosen_key"] in ("pooled_did", "per_pair_did")
invalid = {m["key"] for m in lb["methodologies"] if not m.get("valid", True)}
assert "regadj_geo" in invalid
"""),
    md(r"""
## 6 — What a pre-registration now says

The homework above compresses to five lines, written *before* the test:

> **Estimand:** incremental Search ROAS, 8-week window, treated DMAs.
> **Design:** 5 matched randomized pairs (holdout).
> **Estimator:** pooled DiD.
> **Decision rule:** effect declared iff |estimate| > the placebo-calibrated
> 95th-percentile critical value.
> **Power:** >80% at the expected effect; MDE as designed.

Every choice was made on pre-period data — nothing was picked because it
"found lift". That discipline is what makes the number an experiment produces
worth folding into a model — which is exactly what happens next.
"""),
    code(r"""
C.write_artifact("causal_05_methodology.json", dict(
    n_geos=int(len(wide.columns)),
    design=dict(n_pairs=int(design["n_pairs"]),
                mde_roas=float(design["mde_roas"]),
                se_roas=float(design["se_roas"]),
                duration=int(design["duration"])),
    aa_fpr={nm: dict(naive=float(r.fpr_at_nominal), calibrated=float(r.fpr_at_crit),
                     inflated=bool(r.fpr_inflated)) for nm, r in aa.items()},
    power_at_expected={nm: float(r.power_at_expected) for nm, r in ab.items()},
    leaderboard_choice=str(lb["chosen_key"]),
))
print("artifact written: causal_05_methodology.json")
"""),
    md(r"""
---
**Next — [06 · Calibrating the model](causal_06_calibrating_the_model.ipynb):**
the experiment meets the MMM. Two routes for folding a lift readout into the
posterior — informative priors vs an in-graph likelihood — and the estimand
bookkeeping that decides whether calibration heals the model or quietly
poisons it.
"""),
]


def main():
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    out = "causal/causal_05_measuring_one_experiment.ipynb"
    with open(out, "w") as f:
        nbformat.write(nb, f)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
