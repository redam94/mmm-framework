"""Author demos/critique_to_decision.ipynb (run from ``nbs/``).

    uv run --with nbformat python builders/build_critique_to_decision.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        demos/critique_to_decision.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

Four subsystems on ONE world with a sealed answer key, in the order a real
engagement meets them:

1. **Model critique** — ``ModelValidator``: convergence, PPC, residuals, channel
   diagnostics, LOO. The world is ``make_unobserved_confounding``, so the fit
   grades **excellent** on every one of them and is badly wrong. That is the
   notebook's thesis: goodness-of-fit tests measure fit, and the thing that is
   broken here is identification.
2. **Specification search** — a pre-registered ``SpecSet`` through
   ``run_spec_curve``. The spread across specs is *smaller* than the distance to
   truth, which is the honest lesson: a spec curve prices fragility, not bias.
3. **Media optimization toward different estimands** — average ROI vs marginal
   ROAS vs the sealed truth give three different channel rankings, and the plan
   the default objective recommends is one the model scores as a coin flip
   against standing still. The act also catches a live defect in the constrained
   allocator (issue #290) by direct search, in-notebook.
4. **Experimental measurement** — EIG/EVOI priorities and quadrants, a design
   with a target SE, then ``add_experiment_calibration`` + refit. The two tested
   channels land almost exactly on the truth; the two untested ones end as wrong
   as they started, which is the point.

Every number is computed in-notebook; nothing is transcribed. The answer key is
held back until each act has committed to its reading.
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
import os, sys, time
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path

pio.templates.default = "plotly_white"
pio.renderers.default = "notebook_connected"
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

import logging
from loguru import logger
logger.disable("mmm_framework")
for _n in ("pymc", "pymc.sampling", "numpyro", "jax", "arviz", "pytensor"):
    logging.getLogger(_n).setLevel(logging.ERROR)

# palette (matches the docs site's editorial scheme)
INK, MUTED, TRUTH = "#1f2430", "#8a8f98", "#111418"
GOOD, BAD, GOLD = "#3d7a5c", "#b4552d", "#c9962e"
PALETTE = {"TV": "#4464ad", "Search": "#c9962e", "Social": "#3d7a5c", "Display": "#b4552d"}

def style(fig, height=380, title=None, **kw):
    fig.update_layout(height=height, title=title, margin=dict(t=64, l=64, r=30, b=52),
                      font=dict(size=12), **kw)
    return fig

ART = Path.cwd().parent / "artifacts" / "critique_to_decision"
ART.mkdir(parents=True, exist_ok=True)

SEED = 7
KPI = "Sales"
print("Setup ready. Artifacts →", ART)
"""


WORLD = r"""
from mmm_framework.synth import dgp, mff

# A 3-year national world. The generator plants a known causal truth; we do not
# look at it until each act has committed to a reading.
scenario = dgp.make_unobserved_confounding(seed=SEED, n_weeks=156)
CHANNELS = list(scenario.channels)

DATA = ART / "world.csv"
mff.scenario_to_mff(scenario).to_csv(DATA, index=False)

# The answer key. Sealed in a function so it is not sitting in a variable that
# the analysis cells could accidentally read.
def answer_key():
    t = mff.truth_summary(scenario)
    return pd.Series(t["true_roas"]), pd.Series(t["true_contribution"])

panel = pd.read_csv(DATA)
print(f"{len(panel.Period.unique())} weeks · channels: {', '.join(CHANNELS)}")
print(f"variables: {', '.join(sorted(panel.VariableName.unique()))}")
print(f"\nwhat the generator says it violates: {mff.truth_summary(scenario)['violates']}")
"""


FIT = r"""
from mmm_framework.agents.fitting import build_model

SPEC = {
    "kpi": KPI,
    "kpi_level": "national",
    "media_channels": [
        {"name": c, "adstock": {"type": "geometric"}, "saturation": {"type": "hill"}}
        for c in CHANNELS
    ],
    "control_variables": [{"name": "Price"}],
    "inference": {"method": "nuts", "draws": 1000, "tune": 1000, "chains": 4,
                  "random_seed": 42},
}

t0 = time.time()
mmm = build_model(SPEC, str(DATA))
mmm.fit(draws=1000, tune=1000, chains=4, random_seed=42)
print(f"NUTS fit in {time.time()-t0:.0f}s")
"""


CRITIQUE = r"""
from mmm_framework.validation import ModelValidator, ValidationConfigBuilder

cfg = ValidationConfigBuilder().standard().without_plots().build()
t0 = time.time()
report = ModelValidator(mmm).validate(cfg)
print(f"validated in {time.time()-t0:.0f}s\n")
report.summary()
"""


CRITIQUE_DETAIL = r"""
conv = report.convergence
loo = report.model_comparison.models[0].loo if report.model_comparison else None

rows = [
    ("Convergence", "PASS" if conv.converged else "FAIL",
     f"max R-hat {conv.rhat_max:.3f} · min bulk-ESS {conv.ess_bulk_min:.0f} · "
     f"{conv.divergences} divergences"),
    ("Posterior predictive", "PASS" if report.ppc.overall_pass else "FAIL",
     f"{len(report.ppc.problematic_checks)} problematic check(s)"),
    ("Residuals", "PASS" if report.residuals.overall_adequate else "FAIL",
     f"{len(report.residuals.test_results)} tests run"),
    ("LOO-CV", "n/a" if loo is None else "computed",
     "unavailable" if loo is None
     else f"elpd {loo.elpd_loo:.1f} (se {loo.se_elpd_loo:.1f}) · "
          f"p_loo {loo.p_loo:.1f} · {loo.n_bad_k} bad Pareto-k"),
]
print(pd.DataFrame(rows, columns=["check", "verdict", "detail"]).to_string(index=False))
print(f"\noverall quality      : {report.overall_quality}")
print(f"critical issues      : {len(report.critical_issues)}")
print(f"recommendations      : {len(report.recommendations) or 'none'}")
"""


CRITIQUE_REVEAL = r"""
from mmm_framework.analysis import MMMAnalyzer

roi_tbl = MMMAnalyzer(mmm).compute_channel_roi().set_index("Channel")
est = roi_tbl["ROI"]
true_roi, true_contrib = answer_key()

audit = pd.DataFrame({"estimated ROI": est, "true ROI": true_roi})
audit["error"] = audit["estimated ROI"] - audit["true ROI"]
audit["error %"] = 100 * audit["error"] / audit["true ROI"]
audit["model rank"] = audit["estimated ROI"].rank(ascending=False).astype(int)
audit["true rank"] = audit["true ROI"].rank(ascending=False).astype(int)
audit = audit.loc[CHANNELS]
print(audit.round(3).to_string())

flips = int((audit["model rank"] != audit["true rank"]).sum())
print(f"\nmean |error|  : {audit['error'].abs().mean():.3f} ROI points")
print(f"rank changes  : {flips} of {len(audit)} channels")
print(f"model's best  : {audit['estimated ROI'].idxmax()}   "
      f"| actually best: {audit['true ROI'].idxmax()}")
"""


CRITIQUE_CHART = r"""
ch = list(audit.index)
fig = go.Figure()
for i, c in enumerate(ch):
    lo = roi_tbl.loc[c, "Contribution HDI Low"] / roi_tbl.loc[c, "Total Spend"]
    hi = roi_tbl.loc[c, "Contribution HDI High"] / roi_tbl.loc[c, "Total Spend"]
    fig.add_trace(go.Scatter(x=[lo, hi], y=[i, i], mode="lines", opacity=0.45,
                             line=dict(color=PALETTE[c], width=6),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[audit.loc[c, "estimated ROI"]], y=[i], mode="markers",
                             marker=dict(color=PALETTE[c], size=13),
                             name="posterior (90% interval)", showlegend=(i == 0),
                             hovertemplate=f"{c}: %{{x:.3f}}<extra>estimated</extra>"))
    fig.add_trace(go.Scatter(x=[audit.loc[c, "true ROI"]], y=[i], mode="markers",
                             marker=dict(color=TRUTH, symbol="line-ns-open", size=20,
                                         line=dict(width=3)),
                             name="planted truth", showlegend=(i == 0),
                             hovertemplate=f"{c}: %{{x:.3f}}<extra>truth</extra>"))
fig.update_yaxes(tickmode="array", tickvals=list(range(len(ch))), ticktext=ch)
fig.update_xaxes(title="ROI (KPI per unit spend)")
style(fig, 380, "A model that grades 'excellent' — against what actually happened")
fig.show()
"""


SPEC_CURVE = r"""
from mmm_framework.validation import run_spec_curve, SpecSet, SpecVariant

# Pre-registered BEFORE seeing any of it. Two transform axes plus the one that
# actually matters here — the conditioning set.
spec_set = SpecSet(
    variants=[
        SpecVariant(name="geometric x hill", adstock="geometric",
                    saturation="hill", primary=True),
        SpecVariant(name="geometric x logistic", adstock="geometric",
                    saturation="logistic"),
        SpecVariant(name="weibull x hill", adstock="weibull", saturation="hill"),
        SpecVariant(name="no Price control", adstock="geometric",
                    saturation="hill", controls=[]),
    ],
    rationale="Two transform axes and the conditioning set, fixed in advance.",
)

sweep_spec = dict(SPEC)
sweep_spec["inference"] = {"method": "nuts", "draws": 500, "tune": 500,
                           "chains": 2, "random_seed": 42}
t0 = time.time()
curve = run_spec_curve(sweep_spec, str(DATA), variants=spec_set,
                       compute_loo=True, max_draws=400, random_seed=42)
print(f"{len(curve.fits)} specs fit in {time.time()-t0:.0f}s\n")

# SpecFit.roi is {channel: {mean, lower, upper}} — the per-draw posteriors live
# on .roi_draws and are what the BMA mixture resamples.
per_spec = pd.DataFrame(
    {f.name: {c: d["mean"] for c, d in (f.roi or {}).items()} for f in curve.fits
     if not f.error}
).loc[CHANNELS]
print("ROI by specification:")
print(per_spec.round(3).to_string())

print("\nweights:", {k: round(v, 3) for k, v in curve.weights.items()})
print("per-spec LOO elpd:",
      {f.name: (None if not f.loo else round(f.loo["elpd_loo"], 1)) for f in curve.fits})
failed = [f.name for f in curve.fits if f.error]
print("failed specs:", failed or "none")
"""


SPEC_ROBUST = r"""
rob = pd.DataFrame(curve.robustness).T.loc[CHANNELS]
bma = pd.DataFrame(curve.bma).T.loc[CHANNELS]

verdict = pd.DataFrame({
    "spec min": rob["min"],
    "spec max": rob["max"],
    "spread %": rob["spread_pct"],
    "BMA mean": bma["mean"],
    "true ROI": true_roi.loc[CHANNELS],
})
verdict["spec range"] = verdict["spec max"] - verdict["spec min"]
verdict["distance to truth"] = (verdict["BMA mean"] - verdict["true ROI"]).abs()
verdict["truth inside spec range"] = (
    (verdict["true ROI"] >= verdict["spec min"]) & (verdict["true ROI"] <= verdict["spec max"])
)
print(verdict.round(3).to_string())

inside = int(verdict["truth inside spec range"].sum())
print(f"\ntruth falls inside the spec range for {inside} of {len(verdict)} channels")
print(f"median spec range      : {verdict['spec range'].median():.3f} ROI points")
print(f"median distance to truth: {verdict['distance to truth'].median():.3f} ROI points")
"""


SPEC_CHART = r"""
fig = go.Figure()
for i, c in enumerate(CHANNELS):
    fig.add_trace(go.Scatter(x=[verdict.loc[c, "spec min"], verdict.loc[c, "spec max"]],
                             y=[i, i], mode="lines+markers", opacity=0.6,
                             line=dict(color=PALETTE[c], width=6),
                             marker=dict(size=8, color=PALETTE[c]),
                             name="range across specs", showlegend=(i == 0),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[verdict.loc[c, "BMA mean"]], y=[i], mode="markers",
                             marker=dict(color=PALETTE[c], size=13, symbol="diamond"),
                             name="model-averaged", showlegend=(i == 0),
                             hovertemplate=f"{c}: %{{x:.3f}}<extra>BMA</extra>"))
    fig.add_trace(go.Scatter(x=[verdict.loc[c, "true ROI"]], y=[i], mode="markers",
                             marker=dict(color=TRUTH, symbol="line-ns-open", size=20,
                                         line=dict(width=3)),
                             name="planted truth", showlegend=(i == 0),
                             hovertemplate=f"{c}: %{{x:.3f}}<extra>truth</extra>"))
fig.update_yaxes(tickmode="array", tickvals=list(range(len(CHANNELS))), ticktext=CHANNELS)
fig.update_xaxes(title="ROI")
style(fig, 380, "Every spec agrees with every other spec, and they are all wrong together")
fig.show()
"""


OPT_RANKINGS = r"""
from mmm_framework.planning import (compute_response_curves, optimize_budget,
                                    objective_label)

curves = compute_response_curves(mmm, max_draws=200, random_seed=42)
total_budget = float(np.asarray(mmm.X_media_raw).sum())
plan_mean = optimize_budget(curves=curves, total_budget=total_budget,
                            objective="mean", random_seed=42)

rank = pd.DataFrame({
    "average ROI": est.loc[CHANNELS],
    "marginal ROAS": pd.Series(plan_mean.marginal_roas).loc[CHANNELS],
    "true ROI": true_roi.loc[CHANNELS],
})
for col in rank.columns:
    rank[f"rank · {col}"] = rank[col].rank(ascending=False).astype(int)
print(rank.round(3).to_string())

print(f"\ntop channel by average ROI  : {rank['average ROI'].idxmax()}")
print(f"top channel by marginal ROAS: {rank['marginal ROAS'].idxmax()}")
print(f"top channel in truth        : {rank['true ROI'].idxmax()}")
"""


OPT_OBJECTIVES = r"""
current = pd.Series(np.asarray(mmm.X_media_raw).sum(axis=0), index=CHANNELS)

plans, summary_rows = {}, []
for obj in ["mean", "p10", "cvar5"]:
    p = optimize_budget(curves=curves, total_budget=total_budget,
                        objective=obj, random_seed=42)
    plans[obj] = p
    summary_rows.append({
        "objective": obj,
        "what it maximizes": objective_label(obj),
        "median uplift": p.expected_uplift,
        "uplift 5-95%": f"[{p.uplift_hdi[0]:,.0f}, {p.uplift_hdi[1]:,.0f}]",
        "P(better than today)": p.prob_positive_uplift,
        "expected regret": p.expected_regret,
    })
print(pd.DataFrame(summary_rows).round(3).to_string(index=False))

alloc = pd.DataFrame({"today": current},
                     index=CHANNELS)
for obj, p in plans.items():
    alloc[obj] = pd.Series(dict(zip(curves.channel_names, p.optimal_alloc))).loc[CHANNELS]
print("\nallocation (spend units):")
print(alloc.round(0).to_string())
print("\nshare of budget (%):")
print((100 * alloc / alloc.sum()).round(1).to_string())
"""


OPT_SOLVER_CHECK = r"""
from mmm_framework.planning.budget import objective_curves

# The downside plans came back identical to today. Before reading that as "the
# current plan is already downside-optimal", check it: score the objective at
# today's allocation and at random feasible allocations. If the optimizer's
# answer can be beaten by sampling, it was not an optimum.
spend_grid = curves.multipliers[None, :] * current.values[:, None]

def objective_value(alloc, obj):
    oc = objective_curves(curves, obj)
    return float(sum(np.interp(alloc[i], spend_grid[i], oc[i])
                     for i in range(len(CHANNELS))))

rng = np.random.default_rng(0)
check = []
for obj in ["p10", "cvar5"]:
    v_today = objective_value(current.values, obj)
    v_opt = objective_value(np.asarray(plans[obj].optimal_alloc), obj)
    best, n_better = v_today, 0
    for _ in range(20_000):
        s = rng.dirichlet(np.ones(len(CHANNELS)) * 1.2) * total_budget
        if (s > 2 * current.values).any():
            continue
        v = objective_value(s, obj)
        best = max(best, v)
        n_better += v > v_today + 1e-6
    check.append({"objective": obj, "today": v_today, "optimizer": v_opt,
                  "best sampled": best, "gap": best - v_opt,
                  "samples beating it": n_better})
print(pd.DataFrame(check).round(1).to_string(index=False))
"""


OPT_CHART = r"""
fig = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42],
                    subplot_titles=("Where each objective puts the money",
                                    "Is the plan better than today?"))
for j, col in enumerate(alloc.columns):
    fig.add_trace(go.Bar(x=CHANNELS, y=100 * alloc[col] / alloc[col].sum(),
                         name=col, marker_color=[PALETTE[c] for c in CHANNELS],
                         opacity=[1.0, 0.55, 0.75, 0.9][j], showlegend=False,
                         text=[col] * len(CHANNELS), textposition="none",
                         hovertemplate=f"{col} · %{{x}}: %{{y:.1f}}%<extra></extra>"),
                  row=1, col=1)
fig.add_trace(go.Bar(x=list(plans), y=[plans[o].prob_positive_uplift for o in plans],
                     marker_color=[BAD if plans[o].prob_positive_uplift < 0.5 else GOOD
                                   for o in plans],
                     showlegend=False,
                     hovertemplate="%{x}: P=%{y:.3f}<extra></extra>"), row=1, col=2)
fig.add_hline(y=0.5, line=dict(color=MUTED, dash="dot"), row=1, col=2)
fig.update_yaxes(title="share of budget (%)", row=1, col=1)
fig.update_yaxes(title="P(uplift > 0)", range=[0, 1], row=1, col=2)
fig.update_xaxes(tickangle=-20, row=1, col=1)
style(fig, 420, "The objective, not the model, decides the plan", barmode="group")
fig.show()
"""


EXPERIMENTS = r"""
from mmm_framework.planning import recommend_experiments

ranked, briefs = recommend_experiments(mmm, curves=curves, optimization=plan_mean,
                                       top_k=len(CHANNELS), random_seed=42)
cols = ["channel", "spend_share_pct", "roas_median", "roas_cv",
        "eig", "evoi", "evpi_share", "quadrant", "priority"]
print(ranked[cols].round(3).to_string(index=False))

top = briefs[0]
print(f"\n— brief for {top['channel']} —")
print(f"design        : {top['design_type']}")
print(f"duration      : {top['min_duration_periods']} periods ({top['duration_rationale']})")
print(f"target SE     : {top['target_se']:.4f}")
print(f"rationale     : {top['target_se_rationale']}")
print(f"\nwhy: {top['why']}")
"""


EXPERIMENTS_CHART = r"""
fig = go.Figure()
for _, r in ranked.iterrows():
    c = r["channel"]
    fig.add_trace(go.Scatter(
        x=[r["eig"]], y=[r["evoi"]], mode="markers+text",
        marker=dict(size=14 + 46 * r["spend_share_pct"] / 100, color=PALETTE[c],
                    opacity=0.85, line=dict(color=INK, width=1)),
        text=[f"  {c} · {r['quadrant']}"], textposition="middle right",
        textfont=dict(size=11), showlegend=False,
        hovertemplate=(f"{c}<br>EIG %{{x:.3f}} nats<br>EVOI %{{y:,.0f}} KPI units"
                       f"<br>{r['spend_share_pct']:.0f}% of spend<extra></extra>")))
fig.update_xaxes(title="EIG — how much the experiment would teach (nats)")
fig.update_yaxes(title="EVOI — what that knowledge is worth (KPI units)")
style(fig, 420, "What to test next: learning vs. what the learning is worth")
fig.update_layout(xaxis_range=[0, float(ranked["eig"].max()) * 1.45])
fig.show()
"""


CALIBRATE = r"""
from mmm_framework.calibration.likelihood import ExperimentMeasurement, ExperimentEstimand

# Run the top two recommended tests. A real readout carries measurement error;
# here it is centred on the truth with the SE the brief asked for, which is the
# best case an experiment can deliver — the point is what calibration can and
# cannot repair, not how lucky the readout was.
tested = list(ranked["channel"].head(2))
weeks = sorted(pd.to_datetime(panel.Period.unique()))
window = (str(weeks[-24].date()), str(weeks[-1].date()))

measurements = [
    ExperimentMeasurement(channel=c, test_period=window, value=float(true_roi[c]),
                          se=0.06, estimand=ExperimentEstimand.ROAS)
    for c in tested
]
print(f"calibrating with readouts on: {', '.join(tested)}  ·  window {window[0]} → {window[1]}")

mmm_cal = build_model(SPEC, str(DATA))
mmm_cal.add_experiment_calibration(measurements)
t0 = time.time()
mmm_cal.fit(draws=1000, tune=1000, chains=4, random_seed=42)
print(f"refit with the experiment likelihood in {time.time()-t0:.0f}s")
"""


CALIBRATE_RESULT = r"""
est_cal = MMMAnalyzer(mmm_cal).compute_channel_roi().set_index("Channel")["ROI"]

final = pd.DataFrame({
    "true ROI": true_roi.loc[CHANNELS],
    "before": est.loc[CHANNELS],
    "after": est_cal.loc[CHANNELS],
})
final["|err| before"] = (final["before"] - final["true ROI"]).abs()
final["|err| after"] = (final["after"] - final["true ROI"]).abs()
final["tested?"] = ["yes" if c in tested else "—" for c in CHANNELS]
print(final.round(3).to_string())

print(f"\nROI MAE before : {final['|err| before'].mean():.4f}")
print(f"ROI MAE after  : {final['|err| after'].mean():.4f}")
print(f"reduction      : {100 * (1 - final['|err| after'].mean() / final['|err| before'].mean()):.0f}%")

t_mask = final["tested?"] == "yes"
print(f"\ntested channels   — MAE {final.loc[t_mask, '|err| before'].mean():.3f}"
      f" → {final.loc[t_mask, '|err| after'].mean():.3f}")
print(f"untested channels — MAE {final.loc[~t_mask, '|err| before'].mean():.3f}"
      f" → {final.loc[~t_mask, '|err| after'].mean():.3f}")
"""


CALIBRATE_CHART = r"""
fig = go.Figure()
for i, c in enumerate(CHANNELS):
    fig.add_trace(go.Scatter(
        x=[final.loc[c, "before"], final.loc[c, "after"]], y=[i, i],
        mode="lines+markers", line=dict(color=PALETTE[c], width=3),
        marker=dict(size=[10, 15], color=PALETTE[c],
                    symbol=["circle-open", "circle"]),
        showlegend=False,
        hovertemplate=f"{c}: %{{x:.3f}}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[final.loc[c, "true ROI"]], y=[i], mode="markers",
        marker=dict(color=TRUTH, symbol="line-ns-open", size=20, line=dict(width=3)),
        name="planted truth", showlegend=(i == 0), hoverinfo="skip"))
fig.update_yaxes(tickmode="array", tickvals=list(range(len(CHANNELS))),
                 ticktext=[f"{c} {'(tested)' if c in tested else ''}" for c in CHANNELS])
fig.update_xaxes(title="ROI")
style(fig, 380, "Open circle = before calibration, filled = after, tick = truth")
fig.show()
"""


CLOSE = r"""
ledger = pd.DataFrame([
    {"act": "1 · critique", "question": "does the model fit?",
     "verdict": report.overall_quality,
     "moved us toward truth?": "no — every check passed"},
    {"act": "2 · spec search", "question": "is the answer fragile?",
     "verdict": f"median spread {verdict['spread %'].median():.0f}%",
     "moved us toward truth?": f"no — truth inside range for {inside}/{len(verdict)}"},
    {"act": "3 · optimization", "question": "what should we do?",
     "verdict": f"P(better)={plans['mean'].prob_positive_uplift:.2f} on the mean objective",
     "moved us toward truth?": "no — but it priced the disagreement"},
    {"act": "4 · experiment", "question": "what is actually true?",
     "verdict": f"MAE {final['|err| before'].mean():.3f} → {final['|err| after'].mean():.3f}",
     "moved us toward truth?": "yes — for what it measured"},
])
print(ledger.to_string(index=False))
"""


CELLS = [
    md(r"""
# From critique to decision

**Four things a model has to survive** — a critique, a specification search, an
optimizer pointed at a business target, and an experiment — run end to end on
one synthetic world whose causal truth is known and held back.

The world is `make_unobserved_confounding`: latent demand drives both sales and
the spend on two of the four channels. That back door is open and nothing in the
dataset closes it. It is the most common way a real MMM is wrong, and it is
chosen here because of how the notebook ends up reading:

> the fit grades **excellent** on every diagnostic in the validator, and its
> best channel is the truth's third-best.

Each act commits to a reading before the answer key is consulted. The point is
not that the model is bad — it is that three of these four tools cannot tell
you it is bad, and knowing which one can is the whole skill.
"""),
    code(SETUP),
    md(r"""
## The world

Three years of weekly national data: four media channels, a price control, and a
KPI. The generator plants a known contribution and ROI per channel.
"""),
    code(WORLD),
    md(r"""
## The model

A conventional specification, and a deliberately conventional one: geometric
adstock, Hill saturation, price as the single control. This is what a competent
analyst writes on day one.
"""),
    code(FIT),
    md(r"""
---

# Act 1 · Model critique

`ModelValidator` runs the standard battery: sampler convergence, a posterior
predictive check, residual diagnostics, per-channel collinearity and
convergence, and LOO-CV for out-of-sample fit.
"""),
    code(CRITIQUE),
    code(CRITIQUE_DETAIL),
    md(r"""
Everything passes. R-hat is clean, the posterior predictive covers the data, the
residuals are adequate, there is no collinearity flag, and LOO computes without
a single bad Pareto-k.

On the evidence available to it, this is a good model. Now the answer key.
"""),
    code(CRITIQUE_REVEAL),
    code(CRITIQUE_CHART),
    md(r"""
### What just happened

Two channels chase latent demand. The model has no way to see that demand, so it
credits them for sales the demand would have produced anyway, and it takes that
credit *out of* the channels that were not chasing anything.

Read the chart again: the planted truth sits **outside the 90% posterior
interval** for the channels that moved most. That is not a calibration problem
to be fixed with more draws. The posterior is a correct summary of the wrong
model.

**The lesson.** Every check in that battery asks a version of "does this model
reproduce the data?" — and a confounded model reproduces the data *beautifully*,
because the confounder is in the data. Goodness of fit is not identification, so
no amount of it will ever surface this. A critique is necessary and it is not
sufficient, and treating a green validation report as a licence to act is the
single most expensive mistake available here.
"""),
    md(r"""
---

# Act 2 · Specification search

The usual response to "the model might be wrong" is to fit more models. A spec
curve does it honestly: the variants are **registered in advance**, so the
spread is evidence rather than a menu to choose from after seeing the answers.
"""),
    code(SPEC_CURVE),
    code(SPEC_ROBUST),
    code(SPEC_CHART),
    md(r"""
### What just happened

The specs agree with each other. Swapping the adstock kernel, swapping the
saturation curve, even dropping the price control moves each channel's ROI by a
fraction of the distance to the truth. Every spec is wrong, and they are wrong
*together*, in the same direction, by about the same amount.

**The lesson.** A spec curve prices **fragility** — how much the answer depends
on choices you know you made. It cannot price **bias** — how much the answer
depends on a variable you never had. Those are different quantities and the
spread only measures the first. A tight spec curve is regularly presented as
evidence of a robust result; here it is four models sharing one blind spot, and
the tightness is the confounder's signature rather than a reassurance.

Note which variant moves things least: dropping the price control. The
conditioning set is where causal answers actually live, and it is the axis most
spec curves never touch.
"""),
    md(r"""
---

# Act 3 · Media optimization toward different estimands

Now the model has to produce a decision. "Optimal" is not a property of the
model — it is a property of the **target you point it at**, and the targets
disagree.

First, three ways of asking "which channel is best?"
"""),
    code(OPT_RANKINGS),
    md(r"""
**Average ROI** is total contribution over total spend — a backward-looking
verdict on money already committed. **Marginal ROAS** is the return on the *next*
unit, which is the only one of the two that answers a budget question, and after
saturation the two routinely disagree. Here they nominate different winners, and
the truth nominates a third ordering.

Now hold the estimand fixed and vary the **risk** attitude instead.
"""),
    code(OPT_OBJECTIVES),
    code(OPT_CHART),
    md(r"""
### What just happened

The `mean` objective wants a real reallocation: it defunds Display completely and
moves the money into Search. And then look at what it says about its own
recommendation — `P(better than today)` is about a coin flip, the 5–95% uplift
interval spans zero comfortably in both directions, and expected regret is
material. The plan is a bet the model itself cannot distinguish from standing
still.

That is the honest output of an optimizer working on a posterior this wide. It is
also the number that never survives the trip to a slide, where "reallocate to
Search, +uplift" is what gets written down.

Note which channel it defunds. Display is the truth's **second-best** channel,
and it is defunded to zero on the strength of an ROI the model underestimates by
a third. Act 1's bias is not an abstraction; this is the sentence where it
becomes a budget.
"""),
    md(r"""
### A note on the tooling

The two risk objectives came back with *no change at all* — the same allocation
as today, uplift zero. That reads as "the current plan is already
downside-optimal", which would be a tidy finding.

It should not be believed without a check, so here is the check: score the
objective at today's plan, at the optimizer's plan, and at random feasible
plans.
"""),
    code(OPT_SOLVER_CHECK),
    md(r"""
Random sampling beats the optimizer's answer, by a wide margin and often. So the
downside "optimum" was not an optimum — the constrained solver stalled on its
warm start, and its warm start is today's allocation. Filed as
[#290](https://github.com/redam94/mmm-framework/issues/290); it affects every
risk objective, `mode="free"`, and the group/bound constraints, and it fails
silently in the direction that looks reassuring.

It belongs in this notebook rather than in a footnote. Three acts in, the message
has been that the tools which describe a model cannot tell you it is biased. The
same skepticism is owed to the tool that turns the model into a decision, and
"no change recommended" is exactly the kind of output that never gets questioned
because it asks nothing of anyone.

**The lesson.** The estimand and the objective move the recommendation at least
as much as the modelling choices in Act 2 did, so every published allocation has
to name which target it optimized — two plans built on the same posterior with
different objectives are not comparable, and the framework refuses to delta them
for that reason. And an optimizer's output is a claim like any other. Ask it what
it is claiming against.
"""),
    md(r"""
---

# Act 4 · Experimental measurement

Three acts have described the uncertainty with increasing precision and reduced
it by exactly nothing. An experiment is the only instrument here that adds
information from outside the model.

The question is which one to run. `recommend_experiments` scores each channel by
**EIG** (how much a test would teach, in nats) and **EVOI** (what that knowledge
is worth, in KPI units — the decision it would change).
"""),
    code(EXPERIMENTS),
    code(EXPERIMENTS_CHART),
    md(r"""
The two are genuinely different questions. A channel can be very uncertain and
not worth testing because it is too small to change the plan; another can be
well estimated and worth testing because it carries most of the budget. The
quadrants encode that: `test_now` is high on both, `monitor` is worth watching
but not paying for, `deprioritize` would teach little and change less.

Now run the top two and fold the readouts into the likelihood.
"""),
    code(CALIBRATE),
    code(CALIBRATE_RESULT),
    code(CALIBRATE_CHART),
    md(r"""
### What just happened

The tested channels move almost exactly onto the truth — their error nearly
vanishes. This is the first thing in the notebook that moved an estimate toward
the right answer, and it worked for the reason nothing else could: it brought in
a fact the dataset did not contain.

Now the part that matters just as much. **The untested channels barely moved at
all** — their error is essentially where it started. The overall improvement is
correspondingly modest, and reading only the headline MAE would flatter what
happened: the average hides that one group was fixed and the other was not
touched.

Calibration is not a general-purpose de-biaser. It repairs what it measures, plus
whatever the model's structure propagates from there, and here the structure
propagates almost nothing — the channels are additive and the confounding is
channel-specific, so learning TV's true ROI says nothing about Social's.

**The lesson.** Experiments are the only instrument here with the power to fix an
identification problem, and their reach is exactly as wide as their design. Which
is the argument for choosing deliberately rather than testing whatever is easiest
to test: EIG/EVOI is how you decide which slice of the error you can afford to
buy down this quarter, knowing the rest of it stays.
"""),
    md(r"""
---

## The ledger
"""),
    code(CLOSE),
    md(r"""
## What to take from this

**A green validation report is a floor, not a verdict.** Convergence, PPC,
residuals, and LOO all test whether the model reproduces the data. A confounded
model reproduces the data. Run the battery — a model that fails it is certainly
broken — and then keep going.

**A tight spec curve can be a warning.** Agreement across specs means your
choices did not matter. If the thing that matters is a variable nobody has, every
spec inherits it, and the tightness measures the shared blind spot rather than
the answer's quality. Vary the conditioning set, not just the transforms.

**Name the target before reading the plan.** Average ROI and marginal ROAS answer
different questions and nominated different winners here; expected value and
downside value answer different questions again. The choice of target moved the
recommendation further than any modelling decision in Act 2 did, and the plan the
default objective produced was one the model itself scored as a coin flip.

**Audit the optimizer too.** The downside objectives reported "no change", which
is the most trustworthy-looking output a planning tool can emit, and a few
thousand random samples showed it was a stalled solve. Skepticism that stops at
the model and exempts the machinery around it has not gone far enough.

**Only one of these four instruments adds information.** The first three
characterize what the model believes with steadily more sophistication. The
experiment is the one that changes what it believes, and it changes it only where
you pointed it — the channels left untested here ended exactly as wrong as they
started. Which is the argument for pointing it deliberately: EIG for what you
would learn, EVOI for whether the learning would change anything.

### Where to go next

- `nbs/stress/` — the pressure-test series: more ways a model is silently wrong.
- `nbs/causal/` — the causal-inference series, including the back-door problem
  this notebook builds on (`causal_01`, `causal_04`).
- `nbs/lifecycle/` — the T0→T5 measurement loop as an operating rhythm.
- `nbs/demos/experiment_planning_playbook.ipynb` — the full design surface:
  geo methods, power, switchback, net experiment economics.
- `technical-docs/confounding-sensitivity.md` — pricing the assumption this
  notebook violates, without needing an experiment to do it.
"""),
]


def main() -> None:
    nb = new_notebook(cells=CELLS)
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    out = "demos/critique_to_decision.ipynb"
    with open(out, "w") as fh:
        nbformat.write(nb, fh)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
