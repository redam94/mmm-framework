"""Author lifecycle_04_calibrate.ipynb — T3 of the series (run from ``nbs/``).

    uv run --with nbformat python builders/build_lifecycle_04_calibrate.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        lifecycle/lifecycle_04_calibrate.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

T3 · Calibrate: the geo holdout designed in T2 has read out; now fold it in. Two
routes — a coefficient **prior** anchored on a contribution (simple, holds the
s-curve fixed) versus an in-graph **likelihood** on the ROAS estimand (general,
updates the whole curve/adstock jointly). The money shot: one experiment pulls
the *tested* channel's ROI onto the sealed truth while the untested channels
stay put. Shared world/fit/palette live in ``nbs/builders/lifecycle_common.py``.
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

# ── T3 works from the cached baseline (the pre-experiment posterior) plus the
#    channel the loop chose to test. Everything below refits *that* model with
#    one new piece of evidence folded in.
base = L.fit_baseline()
model, truth = base["model"], base["truth"]
ch = L.FOCUS_CHANNEL
first, last = L.dataset_period(base["mff_df"])
print(f"Folding the {ch} readout into the baseline.  Window: {first} -> {last}.")
assert ch in list(model.channel_names)          # the tested channel is in the fitted model
"""


CELLS = [
    md(r"""
# T3 · Calibrate — fold the readout in
### Notebook 4 of 7 — the Experimental Measurement Lifecycle

**Recap (from `lifecycle_03_design`).** T2 pre-registered a **powered** geo
holdout on **Display** — the channel T1 flagged as both uncertain *and*
budget-critical — and priced it so the short-term cost was worth the learning.
The test ran. The number came back. **Now we fold it in.**

Calibration is the hinge of the whole loop: it is where randomized,
**causal** evidence overwrites the part of the observational posterior that was
being flattered by confounding. There are two routes, and the difference is
worth understanding before we run either:

- **Route A — the coefficient prior.** Read the holdout as a *contribution*
  (dollars of sales the channel caused over the window), back-solve the
  coefficient that implies, and refit with that as a tight **Gamma prior** on
  `beta`. Simple and robust — but it anchors the *level* only and **holds the
  saturation/adstock shape fixed.**
- **Route B — the in-graph likelihood (the headline).** Add a likelihood term on
  the model-implied **ROAS estimand** and re-sample. This updates `beta`, the
  s-curve, *and* the adstock kernel **jointly**, so it is the route to reach for
  when the experiment reports a return (ROAS / mROAS) rather than a raw lift.

We'll run Route B first because it's the one that visibly moves ROI, then show
Route A's mechanism. The sealed truth appears **only to grade the move** — the
loop never sees it.
"""),
    code(SETUP),
    md(r"""
## Before — what the observational model believes about Display

Start with the honest pre-experiment picture: the baseline's per-channel **ROI**,
straight from the posterior. This is the T0 number the CFO would otherwise
allocate on. We isolate Display — the channel the holdout measured — and note
where it sits relative to its **sealed truth** (which the analyst can't see, but
we can, to check the loop is moving the estimate the *right* way).
"""),
    code(r"""
from mmm_framework.reporting.helpers.roi import compute_roi_with_uncertainty

roi_before_df = compute_roi_with_uncertainty(model).set_index("channel").loc[L.CHANNELS]
roi_before = roi_before_df["roi_mean"]

truth_roi = float(truth["true_roas"][ch])
print(roi_before_df[["roi_mean", "roi_hdi_low", "roi_hdi_high"]].round(3).to_string())
print(f"\n{ch}: model ROI {roi_before[ch]:.3f}  vs  sealed truth {truth_roi:.3f}  "
      f"(gap {truth_roi - roi_before[ch]:+.3f})")
# The observational fit UNDER-credits the channel the loop singled out — exactly
# the bias an experiment exists to correct.
assert roi_before[ch] < truth_roi
"""),
    md(r"""
## Route B — the in-graph ROAS likelihood (the headline)

The holdout is reported as a **return** — dollars of sales per dollar of Display
spend over the test window — so it belongs on the model's **ROAS estimand**. We
build the readout explicitly (so the construction is visible), then refit.

The measured `value` is the channel's true ROAS: a well-run geo holdout *reveals*
the real return, and `se = READOUT_SE` is the kind of tight precision a
well-powered geo holdout delivers (T2's minimal four-pair demo was already in
that neighborhood, and a fuller design does better). Because the term lives
**inside the PyMC graph**, the
sampler reconciles it against every other constraint at once — the coefficient,
the saturation curve, and the carryover kernel all move together.
"""),
    code(r"""
from mmm_framework.calibration import ExperimentEstimand, ExperimentMeasurement

# The experiment readout, made explicit for teaching. In production `value` is
# the number the geo test returned; here it is the sealed truth ROAS.
exp = ExperimentMeasurement(
    channel=ch,
    test_period=L.dataset_period(base["mff_df"]),
    value=float(truth["true_roas"][ch]),
    se=L.READOUT_SE,
    estimand=ExperimentEstimand.ROAS,
    distribution="normal",
)
print(f"readout → {ch} ROAS = {exp.value:.3f}  (se {exp.se}, estimand {exp.estimand.value})")

# Refit the baseline with this likelihood attached (progress bars suppressed).
cal = L.fit_calibrated(channel=ch, se=L.READOUT_SE)
roi_after_df = compute_roi_with_uncertainty(cal["model"]).set_index("channel").loc[L.CHANNELS]
roi_after = roi_after_df["roi_mean"]

# Before-vs-after point + interval for the focus channel, with truth marked.
b, a = roi_before_df.loc[ch], roi_after_df.loc[ch]
fig = go.Figure()
for label, r, yy, col in [
    ("before — observational fit", b, 1, L.MUTED),
    ("after — holdout folded in", a, 0, L.PALETTE[ch]),
]:
    fig.add_trace(go.Scatter(x=[r["roi_hdi_low"], r["roi_hdi_high"]], y=[yy, yy], mode="lines",
        line=dict(color=col, width=10), opacity=0.4, showlegend=False))
    fig.add_trace(go.Scatter(x=[r["roi_mean"]], y=[yy], mode="markers+text", text=[label],
        textposition="top center", marker=dict(color=col, size=15,
        line=dict(color="white", width=1.5)), showlegend=False))
fig.add_vline(x=truth_roi, line=dict(color=L.GOOD, width=2.5, dash="dash"),
              annotation_text=f"sealed truth = {truth_roi:.2f}", annotation_position="bottom right")
fig.update_yaxes(range=[-0.6, 1.7], visible=False)
L.style(fig, height=360, title=f"{ch}: one experiment pulls the ROI onto the truth (and tightens it)")
fig.update_layout(xaxis_title="return per $1 of Display spend (ROI)")
fig.show()

gap_before, gap_after = abs(roi_before[ch] - truth_roi), abs(roi_after[ch] - truth_roi)
print(f"{ch} ROI:  before {roi_before[ch]:.3f}  ->  after {roi_after[ch]:.3f}  ->  truth {truth_roi:.3f}")
print(f"gap to truth closed by {100 * (1 - gap_after / gap_before):.0f}%")
# Calibration moves the tested channel's estimate toward the truth.
assert gap_after < gap_before
"""),
    md(r"""
The point slides off the observational value and lands on — or beside — the
sealed truth, and the interval **narrows**: the posterior is now conditioned on
direct causal evidence, not just correlational history. This is the number the
CFO can actually defend.
"""),
    md(r"""
## The surgical point — a test corrects the channel it *measured*

A recurring fear about calibration is that anchoring one channel will quietly
distort the others. It doesn't. The likelihood term speaks only about **Display's**
ROAS; the untested channels move only by the tiny amount that re-sampling
reshuffles a shared posterior. Below is the absolute ROI move per channel — the
tested channel's correction should **dwarf** the rest.
"""),
    code(r"""
move = (roi_after - roi_before).abs().loc[L.CHANNELS]
colors = [L.PALETTE[c] if c == ch else L.MUTED for c in L.CHANNELS]
fig = go.Figure(go.Bar(
    x=L.CHANNELS, y=move.values, marker=dict(color=colors),
    text=[f"{v:.3f}" for v in move.values], textposition="outside"))
L.style(fig, height=400, title="Absolute ROI move from one Display experiment")
fig.update_layout(xaxis_title="", yaxis_title="|ROI after − ROI before|",
                  yaxis=dict(range=[0, move.max() * 1.25]))
fig.show()
others = move.drop(ch)
print(f"{ch} moved {move[ch]:.3f};  other channels moved "
      f"{others.min():.3f}–{others.max():.3f} (mean {others.mean():.3f}).")
# The test corrects the channel it measured — its move is the largest by far.
assert move.idxmax() == ch and move[ch] > others.max()
"""),
    md(r"""
## Route A — the coefficient-prior route (the simpler mechanism)

Route B needed a refit. The **prior** route is cheaper and is worth seeing on its
own terms. It reads the holdout as a **contribution** (total dollars of KPI the
channel caused over the window) and back-solves the coefficient that implies,
producing a tight **Gamma prior** on `beta` to anchor a refit — *without*
touching the saturation/adstock shape.

We use `derive_priors`, which computes that anchor **without re-sampling** (so
the output stays clean). It returns, per channel, the model's fitted coefficient
(`beta_fit_mean`), the experiment-implied target (`beta_target`), its spread
(`beta_sigma`), and the ready-to-use `roi_prior`.
"""),
    code(r"""
from mmm_framework.calibration import ExperimentCalibrator, LiftTestResult

# The same experiment expressed as a contribution readout (10% measurement SE).
lt = LiftTestResult(
    channel=ch,
    test_period=L.dataset_period(base["mff_df"]),
    measured_lift=float(truth["true_contribution"][ch]),
    lift_se=float(truth["true_contribution"][ch]) * 0.10,
)
report = ExperimentCalibrator(model).derive_priors([lt])   # cheap: derives the prior, NO refit
cc = report.channel_calibrations[0]
alpha, rate = cc.roi_prior.params["alpha"], cc.roi_prior.params["beta"]

# Visualize the derived Gamma anchor against the model's fitted coefficient.
xs = np.linspace(cc.beta_target - 4 * cc.beta_sigma, cc.beta_target + 4 * cc.beta_sigma, 240)
from scipy.stats import gamma as _gamma
dens = _gamma.pdf(xs, a=alpha, scale=1.0 / rate)
fig = go.Figure()
fig.add_trace(go.Scatter(x=xs, y=dens, mode="lines", fill="tozeroy",
    line=dict(color=L.PALETTE[ch], width=2), name="experiment-anchored prior on β"))
fig.add_vline(x=cc.beta_fit_mean, line=dict(color=L.MUTED, width=2, dash="dot"),
              annotation_text=f"fitted β = {cc.beta_fit_mean:.3f}", annotation_position="top left")
fig.add_vline(x=cc.beta_target, line=dict(color=L.GOOD, width=2, dash="dash"),
              annotation_text=f"experiment target β = {cc.beta_target:.3f}", annotation_position="top right")
L.style(fig, height=380, title=f"Route A — a tight Gamma prior anchors {ch}'s coefficient")
fig.update_layout(xaxis_title="channel coefficient (β)", yaxis_title="prior density",
                  legend=dict(orientation="h", yanchor="top", y=-0.16, x=0))
fig.show()

print(f"fitted β {cc.beta_fit_mean:.3f}  ->  experiment target β {cc.beta_target:.3f}  "
      f"(± {cc.beta_sigma:.3f})   prior = Gamma(α={alpha:.0f}, rate={rate:.2f})")
print("The contribution anchor barely nudges β: the model's *level* for Display is")
print("about right — the bias that matters is the ROI/curvature, which only Route B")
print("(the in-graph likelihood) can move. Route A holds the s-curve fixed by design.")
# The holdout delivers a tight, positive coefficient anchor that agrees with the
# fitted level to within a fraction of it (a nudge, not an overturn).
assert 0 < cc.beta_sigma < cc.beta_target
assert abs(cc.beta_target - cc.beta_fit_mean) < 0.25 * cc.beta_fit_mean
"""),
    md(r"""
## Which route, when

| | **Route A — prior** | **Route B — likelihood** |
|---|---|---|
| Reads the experiment as | a **contribution** (lift, $) | an **estimand** (ROAS / mROAS) |
| Updates | `beta` **level** only | `beta` + s-curve + adstock, **jointly** |
| S-curve shape | held **fixed** | **re-estimated** |
| Cost | cheap (one anchored refit) | a refit with the term in-graph |
| Reach for it when | you have a clean full-window lift and trust the shape | the readout is a return, or the curvature itself is suspect |

The rule of thumb: use the **prior** for a quick contribution anchor when you
trust the model's saturation shape; use the **likelihood** when the experiment is
reported as a return, or when — as with Display here — the thing that's biased is
the *curvature*, not just the level. Route B is why the ROI moved.

**Readout →** *"The model now agrees with the experiment on Display, and it did
so without disturbing the channels we didn't test. The posterior is calibrated —
we can allocate on it and defend the number."* Spending that calibrated answer is
**T4**.

Next: **`lifecycle_05_allocate`** — T4, where the calibrated posterior sets the
budget (and quantifies how confident we are in it).
Previously: **`lifecycle_03_design`** built the powered, affordable test whose
readout we just folded in.
"""),
]


def main() -> None:
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    out = "lifecycle/lifecycle_04_calibrate.ipynb"
    with open(out, "w") as f:
        nbformat.write(nb, f)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    print(f"wrote {out} with {len(nb.cells)} cells ({n_code} code)")


if __name__ == "__main__":
    main()
