"""Author causal_07_many_experiments.ipynb — Notebook 7 of 11.

    uv run --with nbformat python builders/build_causal_07_many_experiments.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        causal/causal_07_many_experiments.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

Rung 7 — evidence as a portfolio, on the realistic seven-channel world:
a Radio go-dark that snaps the Radio/Print ridge (and honestly transfers the
pair-sum's observational bias onto the untested twin), an OFF-PANEL Display
test from a window the model never saw (eval_spend / cold_start), and a
conflicting legacy study caught by a pre-merge tension check — then merged
anyway to show the damage. Three fast refits on the cached baseline.
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
# Many Experiments
### Evidence composes — and a portfolio needs bookkeeping

*Notebook 7 of 11 — Causal Inference in Practice.*

One experiment repaired one channel. A real measurement program accumulates
**a portfolio**: readouts of different channels, windows, designs and
vintages — plus the occasional inherited "study" of dubious provenance. This
notebook folds three very different pieces of evidence into the realistic
seven-channel model from rung 2 and watches what each one does:

1. **A Radio go-dark** (on-panel) — aimed straight at the Radio/Print
   collinear ridge, the model's loudest "I don't know".
2. **A Display holdout from *outside* the modeling window** — the model was
   fit through 2023, the test ran in spring 2024. Off-panel calibration
   evaluates the channel's *global response curve* at the test's spend level,
   so window overlap is unnecessary (the price: a structural-stationarity
   assumption, which we state).
3. **A legacy Search study** from a previous agency — tight SE, impressive
   deck, wrong number. Caught by a **tension check before merging**… and then
   merged anyway, to measure exactly what a bad certificate costs.

Three lessons, each graded against the sealed key: experiments compose;
a test on one twin of a collinear pair resolves the *split* but not the
*sum*; and the portfolio's most dangerous member is the confident wrong one.
"""),
    code(SETUP),
    md(r"""
## 1 — The baseline and its honest ignorance

Rung 2's causal fit, reloaded from cache. The ridge: Radio and Print flighted
in lockstep (spend correlation ≈ 0.996), so the data pins their combined
effect while each channel's own interval stays enormous.
"""),
    code(r"""
sc = C.scenario_for("real_causal")
C.check_truth(sc)
base = C.fit_world("real_causal")
g0 = base["grade"]
w_0 = (g0["hi"] - g0["lo"]).rename("width")
print(g0[["true", "est", "rel_err", "covered"]].round(2).to_string())
print("\n90% interval widths:", {c: round(float(w_0[c])) for c in ("Radio", "Print")})

pair_est = g0.loc["Radio", "est"] + g0.loc["Print", "est"]
pair_true = g0.loc["Radio", "true"] + g0.loc["Print", "true"]
print(f"Radio+Print combined: est {pair_est:.0f} vs true {pair_true:.0f} "
      f"({pair_est/pair_true - 1:+.0%}) — note the SUM itself is under-read.")
"""),
    md(r"""
## 2 — Experiment 1: the Radio go-dark

Ten dark weeks for Radio, read with rung 5's discipline. Fold it in via the
likelihood route and re-grade.
"""),
    code(r"""
expR, rR = C.measurement(sc, "Radio", 90, 100, mult=0.0, se=0.10, seed=11)
print(f"Radio readout: window ROAS {rR['observed']:.2f} ± 0.10 "
      f"(true {rR['true_roas']:.2f}), window {rR['period']}")

s1 = C.fit_calibrated("real_causal", [expR])
g1 = s1["grade"]
w_1 = g1["hi"] - g1["lo"]
print(g1.loc[["Radio", "Print"], ["true", "est", "rel_err", "covered"]].round(2).to_string())
"""),
    code(r"""
# The tested channel is repaired: Radio lands near truth and its interval
# collapses by ~3/4. And the twin moves too — Print's interval nearly halves,
# because resolving Radio resolves the SPLIT of the pinned pair-sum.
assert abs(g1.loc["Radio", "rel_err"]) < 0.20
assert w_1["Radio"] < 0.5 * w_0["Radio"]
assert w_1["Print"] < 0.7 * w_0["Print"]

# The honest part most case studies omit: Print's POINT estimate got WORSE.
# The data pinned the pair-sum ~30% low; the experiment fixed Radio's share
# of it, so the whole sum deficit now lands on Print — with more confidence.
assert abs(g1.loc["Print", "rel_err"]) > abs(g0.loc["Print", "rel_err"])
print(f"Print: {g0.loc['Print','rel_err']:+.0%} -> {g1.loc['Print','rel_err']:+.0%} "
      "(inherits the pair-sum's observational bias).\n"
      "One test resolves a collinear pair's SPLIT, not its SUM — "
      "the untested twin just became the program's top re-test priority.")
"""),
    code(r"""
fig = go.Figure()
for stage, g, op in [("before", g0, 0.45), ("after Radio test", g1, 0.95)]:
    fig.add_trace(go.Bar(x=["Radio", "Print"],
                         y=[g.loc["Radio", "est"], g.loc["Print", "est"]],
                         name=stage, opacity=op,
                         marker_color=[C.PALETTE["Radio"], C.PALETTE["Print"]],
                         error_y=dict(type="data", symmetric=False,
                                      array=[g.loc[c, "hi"] - g.loc[c, "est"] for c in ("Radio", "Print")],
                                      arrayminus=[g.loc[c, "est"] - g.loc[c, "lo"] for c in ("Radio", "Print")])))
fig.add_trace(go.Scatter(x=["Radio", "Print"],
                         y=[g0.loc["Radio", "true"], g0.loc["Print", "true"]],
                         mode="markers", name="sealed truth",
                         marker=dict(color=C.TRUTH, symbol="line-ew-open", size=26,
                                     line=dict(width=3))))
fig.update_layout(barmode="group")
fig.update_yaxes(title="total contribution (KPI units)")
C.style(fig, title="The ridge snaps — onto the tested channel", height=430)
"""),
    md(r"""
## 3 — Experiment 2: off-panel Display

The Display holdout ran **after** the modeling window closed — spring 2024,
while the panel ends in late 2023. Classic grounds for "sorry, the test
doesn't overlap the data, we can't use it." With off-panel calibration you
can: declare the test's **spend level** (`eval_spend` per period), duration
(`eval_periods`), and adstock regime (`cold_start`: the channel ramps from
zero in the test markets), and the likelihood attaches to the channel's
*global response curve* evaluated at that spend — no training rows involved.

The assumption you buy it with, stated out loud: **structural stationarity**
— the response curve (saturation, carryover, effectiveness) is the same in
spring 2024 as in the fitted window. Time-varying worlds break this; decay
handling arrives in nb09.
"""),
    code(r"""
from mmm_framework.calibration import ExperimentEstimand, ExperimentMeasurement

S_LVL, W = 45.0, 8   # $45k/week for 8 weeks, one national test cell
lift_true = float(C.true_media_term(sc, "Display", np.full(W, S_LVL)).sum())
roas_true = lift_true / (W * S_LVL)
obs = roas_true + float(np.random.default_rng(1).normal(0, 0.08))
print(f"off-panel truth at ${S_LVL:.0f}k/wk (cold start): ROAS {roas_true:.2f}; "
      f"readout {obs:.2f} ± 0.08")

expD = ExperimentMeasurement(
    channel="Display", test_period=("2024-03-04", "2024-04-22"),
    value=float(obs), se=0.08,
    estimand=ExperimentEstimand.ROAS, distribution="normal",
    eval_spend=S_LVL, eval_periods=W, eval_units=1, adstock_state="cold_start",
    name="display_offpanel_2024",
)
s2 = C.fit_calibrated("real_causal", [expR, expD])
g2 = s2["grade"]
w_2 = g2["hi"] - g2["lo"]
print(g2.loc[["Display"], ["true", "est", "rel_err", "covered"]].round(2).to_string())
"""),
    code(r"""
# The out-of-window test still disciplines the curve: Display's interval
# halves and its point estimate doesn't degrade.
assert w_2["Display"] < 0.6 * w_0["Display"]
assert abs(g2.loc["Display", "rel_err"]) <= abs(g0.loc["Display", "rel_err"]) + 0.05
print(f"Display width: {w_0['Display']:.0f} -> {w_2['Display']:.0f}  "
      f"(rel err {g0.loc['Display','rel_err']:+.0%} -> {g2.loc['Display','rel_err']:+.0%})")
"""),
    md(r"""
## 4 — Experiment 3: the confident wrong certificate

An inherited deck claims Search ROAS **0.95 ± 0.10** ("agency lift study,
2021"). Before merging *anything*, run the tension check: compare the claimed
value against the current model's posterior for the same estimand, combining
both uncertainties. This is the portfolio's smoke detector.
"""),
    code(r"""
LEGACY_VAL, LEGACY_SE = 0.95, 0.10
m_roas = float(g2.loc["Search", "est_roas"])
sd_roas = float((g2.loc["Search", "roas_hi"] - g2.loc["Search", "roas_lo"]) / 3.29)
z = (LEGACY_VAL - m_roas) / np.sqrt(LEGACY_SE**2 + sd_roas**2)
print(f"model-implied Search avg ROAS: {m_roas:.2f} ± {sd_roas:.2f}")
print(f"legacy claim: {LEGACY_VAL:.2f} ± {LEGACY_SE:.2f}   ->   tension z = {z:.1f}")
assert z > 2.0
print("\n>2σ tension BEFORE merging. Policy: investigate provenance — was the "
      "estimand matched? the estimator calibrated? the market representative? "
      "Park it in the ledger as 'disputed'. We merge it anyway below, "
      "as narrators, to price the mistake.")
"""),
    code(r"""
expS = ExperimentMeasurement(
    channel="Search", test_period=("2021-06-07", "2021-08-30"),
    value=LEGACY_VAL, se=LEGACY_SE,
    estimand=ExperimentEstimand.ROAS, distribution="normal",
    name="legacy_search_2021",
)
s3 = C.fit_calibrated("real_causal", [expR, expD, expS])
g3 = s3["grade"]
w_3 = g3["hi"] - g3["lo"]

print(g3.loc[["Search"], ["true", "est", "rel_err", "covered"]].round(2).to_string())
# The bad certificate drags a HEALTHY estimate away from truth — and
# tightens the interval while doing it. Confidently wrong, by injection.
assert g3.loc["Search", "rel_err"] > 0.15
assert w_3["Search"] < w_2["Search"]
print(f"Search: {g2.loc['Search','rel_err']:+.0%} -> {g3.loc['Search','rel_err']:+.0%}, "
      f"width {w_2['Search']:.0f} -> {w_3['Search']:.0f}. "
      "A likelihood commits — including to lies.")
"""),
    md(r"""
## 5 — The portfolio ledger

Every readout, its vintage, estimand, SE, and what it did to the posterior —
the bookkeeping nb09's planning engine will consume. Below, the interval-width
trajectory as evidence accumulated (legacy study excluded from the final
model; it stays in the ledger as *disputed*).
"""),
    code(r"""
stages = ["no experiments", "+ Radio go-dark", "+ Display off-panel", "+ legacy (disputed)"]
widths = {"no experiments": w_0, "+ Radio go-dark": w_1,
          "+ Display off-panel": w_2, "+ legacy (disputed)": w_3}
rows = []
for st in stages:
    for ch in ("Radio", "Print", "Display", "Search"):
        rows.append(dict(stage=st, channel=ch, width=float(widths[st][ch])))
ledger_w = pd.DataFrame(rows)
C.contraction_chart(ledger_w, title="What each piece of evidence bought (90% interval width)",
                    height=440)
"""),
    code(r"""
ledger = pd.DataFrame([
    dict(name="radio_dark_2023", channel="Radio", estimand="avg ROAS (window)",
         vintage="2023-10", value=round(rR["observed"], 2), se=0.10,
         status="merged", origin="pre-registered geo holdout (nb05 discipline)"),
    dict(name="display_offpanel_2024", channel="Display", estimand="avg ROAS (off-panel, cold start)",
         vintage="2024-04", value=round(float(obs), 2), se=0.08,
         status="merged", origin="post-window national holdout"),
    dict(name="legacy_search_2021", channel="Search", estimand="avg ROAS (claimed)",
         vintage="2021-08", value=LEGACY_VAL, se=LEGACY_SE,
         status="disputed (tension z=%.1f)" % z, origin="inherited agency deck"),
])
print(ledger.to_string(index=False))

C.write_artifact("causal_07_portfolio.json", dict(
    ridge=dict(radio_err=dict(before=float(g0.loc["Radio", "rel_err"]),
                              after=float(g1.loc["Radio", "rel_err"])),
               print_err=dict(before=float(g0.loc["Print", "rel_err"]),
                              after=float(g1.loc["Print", "rel_err"])),
               width_shrink=dict(radio=float(1 - w_1["Radio"] / w_0["Radio"]),
                                 print=float(1 - w_1["Print"] / w_0["Print"]))),
    offpanel=dict(true_roas=float(roas_true), observed=float(obs),
                  width_shrink=float(1 - w_2["Display"] / w_0["Display"])),
    conflict=dict(tension_z=float(z),
                  search_err_clean=float(g2.loc["Search", "rel_err"]),
                  search_err_poisoned=float(g3.loc["Search", "rel_err"])),
))
print("\nartifact written: causal_07_portfolio.json")
"""),
    md(r"""
## Portfolio discipline

1. **Evidence composes** through the likelihood — each honest readout
   repaired its target and tightened its neighborhood.
2. **Collinear pairs need two answers.** A test on one twin resolves the
   split; the pair-*sum* stays whatever the observational data said it was.
   Budget the second test (or design one that moves both — nb08).
3. **Tension-check before merging.** A >2σ gap between a claim and the
   current posterior is a provenance investigation, not a data point.
4. **Vintage matters.** The 2021 study wouldn't be trustworthy even if
   honest — evidence decays as the world drifts. Pricing that decay is
   nb09's job.

---
**Next — [08 · Designing the next experiment](causal_08_designing_the_next_experiment.ipynb):**
stop reading tests and start *choosing* them: model-anchored power, the
short-term cost of learning, a Pareto frontier of designs, and schedules that
identify saturation itself.
"""),
]


def main():
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    out = "causal/causal_07_many_experiments.ipynb"
    with open(out, "w") as f:
        nbformat.write(nb, f)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
