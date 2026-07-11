"""Author causal_06_calibrating_the_model.ipynb — Notebook 6 of 11.

    uv run --with nbformat python builders/build_causal_06_calibrating_the_model.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        causal/causal_06_calibrating_the_model.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

Rung 6 — folding one honest experiment into the model, on the confounded
world where calibration has real causal work to do. The two routes (informative
prior vs in-graph likelihood) with the measured punchline that only the
likelihood route repairs confounding here; the estimand-discipline demo
(average ROAS recorded as marginal silently VOIDS the test); and the surgical
nature of calibration. Refits are fast (cached baseline + 2x500 numpyro).
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
# Calibrating the Model
### One honest experiment meets a confounded posterior

*Notebook 6 of 11 — Causal Inference in Practice.*

The Search lift test designed in [notebook 05](causal_05_measuring_one_experiment.ipynb)
has run: an 8-week go-dark, read with the pre-registered calibrated estimator.
The readout lands on the desk of the analyst who owns Veranda's national MMM —
the model from rung 1 that over-credits Search by ~40% and *cannot know it*.

This notebook is about the merge. Two mechanically different routes exist for
folding an experiment into a Bayesian MMM, and they are **not** equivalent:

- **Route A — informative priors** (`ExperimentCalibrator`): translate the
  lift into a prior on the channel's coefficient, refit. The prior *suggests*.
- **Route B — in-graph likelihood** (`add_experiment_calibration`): the
  readout enters the model as one more **observation**, with its own
  likelihood term on the model-implied window ROAS. The likelihood *commits* —
  the posterior must reconcile the experiment with the history, jointly
  re-negotiating the coefficient, the saturation curve, and the carryover.

We grade both against the sealed key, then run the quietest failure in
applied calibration: recording the readout under the **wrong estimand** — the
average-vs-marginal confusion from rung 2 — and watching a six-figure
experiment evaporate without a single warning.

*(Simulation honesty note: the readout value below is generated from the
world's sealed truth plus sampling noise at the design's SE — a well-run
experiment, not an oracle.)*
"""),
    code(SETUP),
    md(r"""
## 1 — The patient: a confidently confounded model

Rung 1's baseline on the `unobserved_confounding` history. Search: +43%
over-credited, truth outside the 90% interval, all diagnostics green.
"""),
    code(r"""
sc = C.scenario_for("uc")
C.check_truth(sc)
base = C.fit_world("uc")
g0 = base["grade"]
print(g0[["true", "est", "rel_err", "covered"]].round(2).to_string())
assert g0.loc["Search", "rel_err"] > 0.15 and not g0.loc["Search", "covered"]
"""),
    md(r"""
## 2 — The readout, and what an experiment actually measures

The go-dark window: weeks 100–111, Search off in the treated markets. Below,
the narrator's view — the world's true Search response with and without the
window's spend. Note the **carryover tail**: the lost effect extends past the
window's end, because this week's spend was still working next month. A
competent readout (and the model's in-graph estimand) accounts for it.
"""),
    code(r"""
W0, W1, SE = 100, 112, 0.08
exp, r = C.measurement(sc, "Search", W0, W1, mult=0.0, se=SE, seed=5)
x = sc.spend["Search"].to_numpy(float)
x_dark = x.copy(); x_dark[W0:W1] = 0.0
resp_full = C.true_media_term(sc, "Search", x)
resp_dark = C.true_media_term(sc, "Search", x_dark)

fig = go.Figure()
fig.add_trace(go.Scatter(x=sc.weeks, y=resp_full, name="Search response (as run)",
                         line=dict(color=C.PALETTE["Search"], width=2)))
fig.add_trace(go.Scatter(x=sc.weeks, y=resp_dark, name="go-dark counterfactual",
                         line=dict(color=C.INK, width=1.6, dash="dot"),
                         fill="tonexty", fillcolor="rgba(217,140,63,0.25)"))
fig.add_vrect(x0=sc.weeks[W0], x1=sc.weeks[W1 - 1], fillcolor="rgba(0,0,0,0.06)",
              line_width=0, annotation_text="dark window")
fig.update_yaxes(title="true Search contribution (KPI units)")
C.style(fig, title="What the experiment measured (narrator view) — note the carryover tail",
        height=420)

print(f"readout: window incremental ROAS = {r['observed']:.2f} ± {SE} "
      f"(true window ROAS: {r['true_roas']:.2f})")
"""),
    md(r"""
## 3 — Route A: the experiment as a prior

`ExperimentCalibrator` reduces the lift to the coefficient scale using the
fitted model's own response shape (the *design factor*), derives a tight Gamma
prior, and refits. Watch the derivation do its job perfectly — and then watch
what the data does to it.
"""),
    code(r"""
from mmm_framework.calibration import ExperimentCalibrator
from mmm_framework.validation.results import LiftTestResult

lt = LiftTestResult(
    channel="Search", test_period=r["period"],
    measured_lift=r["observed"] * r["delta_spend"],   # KPI units
    lift_se=SE * r["delta_spend"],
)
cal = ExperimentCalibrator(base["model"])
report = cal.derive_priors([lt])
cc = report.channel_calibrations[0]
print(f"derived prior: beta target {cc.beta_target:.2f} ± {cc.beta_sigma:.2f} "
      f"(the confounded fit had beta ≈ {cc.beta_fit_mean:.2f})")
assert report.calibrated_channels == ["Search"]

outA = cal.calibrate([lt], refit=True, draws=C.DRAWS, tune=C.TUNE,
                     chains=C.CHAINS, random_seed=7)
gA = C.grade(sc, outA.model)
print(gA[["true", "est", "rel_err", "covered"]].round(2).to_string())
"""),
    code(r"""
# The prior said "cut Search roughly in half" — and the posterior barely
# moved. A prior worth one experiment's information faces 156 weeks of
# confounded likelihood, and the likelihood wins. A prior is a SUGGESTION.
assert gA.loc["Search", "rel_err"] > 0.25
print(f"Route A Search error: {g0.loc['Search','rel_err']:+.0%} -> "
      f"{gA.loc['Search','rel_err']:+.0%}. The confounded data steamrolled the prior.")
"""),
    md(r"""
*(Route A is not broken — it's built for a different job: a portable library
of lift-test priors that composes across model versions and vintages by
inverse-variance. As a **correction device against confounded in-sample
data**, though, what you need is a likelihood.)*

## 4 — Route B: the experiment as data

`add_experiment_calibration` attaches the readout to the graph as an
observation of the model-implied window ROAS. Now agreement with the
experiment is part of what "fitting the data" *means* — the sampler must find
parameters that explain the history **and** the test.
"""),
    code(r"""
outB = C.fit_calibrated("uc", [exp])
gB = outB["grade"]
print(gB[["true", "est", "rel_err", "covered"]].round(2).to_string())

three = pd.Series({
    "baseline (no experiment)": g0.loc["Search", "est"],
    "Route A — prior": gA.loc["Search", "est"],
    "Route B — likelihood": gB.loc["Search", "est"],
})
fig = go.Figure(go.Bar(x=three.index, y=three.values,
                       marker_color=[C.MUTED, "#c9a45a", C.GOOD], opacity=0.9))
fig.add_hline(y=float(g0.loc["Search", "true"]), line_color=C.TRUTH, line_width=2,
              annotation_text="sealed truth")
fig.update_yaxes(title="Search total contribution (KPI units)")
C.style(fig, title="One experiment, two routes — only the likelihood commits",
        height=420)
"""),
    code(r"""
# Route B lands Search on the truth (within noise) and the interval covers.
assert abs(gB.loc["Search", "rel_err"]) < 0.15
assert bool(gB.loc["Search", "covered"])

# And calibration is SURGICAL: Social — the other chaser, untested — keeps
# its confounding bias almost untouched. An experiment repairs what it
# measured, not what you wish it had measured.
assert gB.loc["Social", "rel_err"] > 0.2
print(f"Search: {g0.loc['Search','rel_err']:+.0%} -> {gB.loc['Search','rel_err']:+.0%}   "
      f"Social (untested): {g0.loc['Social','rel_err']:+.0%} -> "
      f"{gB.loc['Social','rel_err']:+.0%}")
"""),
    md(r"""
## 5 — The quietest failure: the wrong estimand

Rung 2 drew the line between **average** ROAS (what a go-dark measures) and
**marginal** ROAS (what the next dollar buys). Here's what happens when the
analyst files this readout under the wrong one — same number, same SE,
declared as *marginal* at +10% spend:
"""),
    code(r"""
from mmm_framework.calibration import ExperimentEstimand, ExperimentMeasurement

wrong = ExperimentMeasurement(
    channel="Search", test_period=r["period"],
    value=float(r["observed"]), se=SE,
    estimand=ExperimentEstimand.MROAS, spend_lift_pct=10.0,
    distribution="normal", name="wrong_estimand",
)
outW = C.fit_calibrated("uc", [wrong])
gW = outW["grade"]

cmp = pd.Series({
    "baseline": g0.loc["Search", "rel_err"],
    "correct estimand (avg ROAS)": gB.loc["Search", "rel_err"],
    "wrong estimand (as marginal)": gW.loc["Search", "rel_err"],
}, name="Search relative error")
print(cmp.map("{:+.0%}".format).to_string())

fig = go.Figure(go.Bar(x=cmp.index, y=cmp.values,
                       marker_color=[C.MUTED, C.GOOD, C.BAD], opacity=0.9))
fig.add_hline(y=0, line_color=C.TRUTH)
fig.update_yaxes(title="Search relative error vs truth", tickformat="+.0%")
C.style(fig, title="Same number, wrong slot: the experiment evaporates", height=400)
"""),
    code(r"""
# The wrong-estimand calibration leaves Search essentially as biased as the
# baseline. Why so QUIET? Saturation puts the marginal below the average —
# the confounded model's marginal ROAS near current spend already sits close
# to the misfiled number, so the likelihood term is satisfied WITHOUT moving
# anything. No error, no warning, no correction: the test is simply spent.
assert gW.loc["Search", "rel_err"] > 0.25
assert abs(gB.loc["Search", "rel_err"]) < 0.15
print("Estimand bookkeeping isn't pedantry — it's the difference between "
      "an experiment that repairs the model and one that vanishes into it.")
"""),
    md(r"""
## 6 — Calibration discipline, in five lines

1. **Match the estimand.** A go-dark window measures *average window ROAS*;
   a small flighting delta measures *marginal*. File it where it happened.
2. **Prefer the likelihood route** when the goal is correcting *this* model
   against *this* (possibly confounded) history; use prior-route calibration
   for portable evidence libraries.
3. **Carry the SE honestly.** The likelihood weighs the experiment by its
   precision; understating the SE overrules 156 weeks of data with 8, and
   overstating it buys nothing.
4. **Expect surgery, not absolution.** Untested channels keep their biases —
   which is why nb07 builds a *portfolio*.
5. **Re-check afterwards.** A calibrated fit is a new model; the refutation
   battery and learning diagnostics from rung 2 still apply.
"""),
    code(r"""
C.write_artifact("causal_06_calibration.json", dict(
    readout=dict(observed=float(r["observed"]), true=float(r["true_roas"]),
                 se=SE, window=[int(W0), int(W1)]),
    search_rel_err=dict(baseline=float(g0.loc["Search", "rel_err"]),
                        route_a_prior=float(gA.loc["Search", "rel_err"]),
                        route_b_likelihood=float(gB.loc["Search", "rel_err"]),
                        wrong_estimand=float(gW.loc["Search", "rel_err"])),
    social_untested_rel_err=dict(baseline=float(g0.loc["Social", "rel_err"]),
                                 after_b=float(gB.loc["Social", "rel_err"])),
    derived_prior=dict(beta_target=float(cc.beta_target),
                       beta_sigma=float(cc.beta_sigma),
                       beta_fit_mean=float(cc.beta_fit_mean)),
))
print("artifact written: causal_06_calibration.json")
"""),
    md(r"""
---
**Next — [07 · Many experiments](causal_07_many_experiments.ipynb):** evidence
as a portfolio. Multiple readouts of different vintages, an **off-panel**
experiment from a window the model never saw, a conflicting legacy study —
and the collinear Radio/Print ridge from rung 2, snapped shut by a single
well-placed test.
"""),
]


def main():
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    out = "causal/causal_06_calibrating_the_model.ipynb"
    with open(out, "w") as f:
        nbformat.write(nb, f)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
