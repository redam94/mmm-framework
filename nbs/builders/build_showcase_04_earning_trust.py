"""Author showcase/showcase_04_earning_trust.ipynb (run from ``nbs/``).

    uv run --with nbformat python builders/build_showcase_04_earning_trust.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        showcase/showcase_04_earning_trust.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

Feature-showcase notebook 04 — "Before you believe a number". The trust stack
on a world built to deceive (make_unobserved_confounding): the declarative
estimand registry (three different numbers all called ROI), parameter-learning
and identification diagnostics, the bias-sensitivity surface with the
"confounder as strong as Price" benchmark, rolling-origin backtesting, the
spec-curve robustness sweep, the causal refutation suite, and experiment
calibration pulling an over-credited channel back toward its sealed truth.
Every number is computed in-notebook.
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
import os, time
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

import logging
from loguru import logger
logger.disable("mmm_framework")
for _n in ("pymc", "pymc.sampling", "numpyro", "jax", "arviz", "pytensor"):
    logging.getLogger(_n).setLevel(logging.ERROR)

INK, MUTED, TRUTH = "#1f2430", "#8a8f98", "#111418"
GOOD, BAD, GOLD = "#3d7a5c", "#b4552d", "#c9962e"
PALETTE = {"TV": "#4464ad", "Search": "#c9962e", "Social": "#3d7a5c", "Display": "#b4552d"}

def style(fig, height=380, title=None, **kw):
    fig.update_layout(height=height, title=title, margin=dict(t=64, l=64, r=30, b=52),
                      font=dict(size=12), **kw)
    return fig

ART = Path.cwd().parent / "artifacts" / "showcase"
ART.mkdir(parents=True, exist_ok=True)
print("Setup ready.")
"""

WORLD = r"""
from mmm_framework.synth import dgp, mff

# A world with a hidden common cause: latent demand pushes BOTH the budgets of
# the "chasing" channels (Search, Social) AND baseline sales. The model never
# sees demand — only Price. Its sealed truth (true_roas per channel) lets us
# grade every number the model produces.
scenario = dgp.make_unobserved_confounding(seed=1, n_weeks=120)
CHANNELS = list(scenario.spend.columns)
CHASERS = scenario.notes["chasers"]
demand = scenario.notes["latent_demand"]

DATA = ART / "confounded.csv"
mff_df = mff.scenario_to_mff(scenario)
mff_df.to_csv(DATA, index=False)
PERIODS = sorted(mff_df["Period"].astype(str).unique())

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    subplot_titles=("The hidden driver and the budgets that chase it",
                                    "The KPI the model must explain"))
z = lambda v: (v - np.mean(v)) / np.std(v)
fig.add_trace(go.Scatter(x=scenario.weeks, y=z(demand), name="latent demand (UNOBSERVED)",
                         line=dict(color=TRUTH, width=3, dash="dot")), row=1, col=1)
for ch in ["Search", "TV"]:
    fig.add_trace(go.Scatter(x=scenario.weeks, y=z(scenario.spend[ch].to_numpy()),
                             name=f"{ch} spend (z)", line=dict(color=PALETTE[ch], width=1.6)),
                  row=1, col=1)
fig.add_trace(go.Scatter(x=scenario.weeks, y=scenario.y, name="Sales",
                         line=dict(color=INK, width=1.8)), row=2, col=1)
fig.update_yaxes(title="z-score", row=1, col=1)
fig.update_yaxes(title="Sales", row=2, col=1)
style(fig, 520, "One latent demand curve drives both Search's budget and the KPI")
fig.show()

r_search = np.corrcoef(z(demand), z(scenario.spend["Search"]))[0, 1]
r_tv = np.corrcoef(z(demand), z(scenario.spend["TV"]))[0, 1]
print(f"corr(latent demand, Search spend) = {r_search:.2f}  (chaser)")
print(f"corr(latent demand, TV spend)     = {r_tv:.2f}  (barely chases)")
print("\nsealed truth (causal ROAS):")
print(scenario.true_roas.round(3).to_string())
"""

FIT = r"""
from mmm_framework.agents.fitting import build_model

SPEC = {
    "kpi": "Sales",
    "kpi_level": "national",
    "media_channels": [
        {"name": c, "adstock": {"type": "geometric"}, "saturation": {"type": "hill"}}
        for c in CHANNELS
    ],
    "control_variables": [{"name": c} for c in scenario.controls.columns],
}
t0 = time.time()
mmm = build_model(SPEC, str(DATA))
mmm.fit(draws=500, tune=500, chains=2, random_seed=42)
print(f"NUTS-lite fit (2 chains x 500) in {time.time()-t0:.0f}s")
"""

ESTIMANDS = r"""
from mmm_framework.estimands import registry, EstimandEvaluator

names = ["contribution_roi", "counterfactual_roi", "marginal_roas"]
t0 = time.time()
evaluator = EstimandEvaluator(mmm, random_seed=42)
est = evaluator.evaluate([registry.get(n) for n in names])
print(f"evaluated {len(est)} estimand realizations in {time.time()-t0:.1f}s\n")

rows = []
for n in names:
    for ch in CHANNELS:
        r = est[f"{n}:{ch}"]
        rows.append({"estimand": n, "channel": ch, "mean": r.mean,
                     "low": r.hdi_low, "high": r.hdi_high,
                     "interval": r.interval_definition, "units": r.units})
edf = pd.DataFrame(rows)
print(edf.round(3).to_string(index=False))

# A refusal is a first-class result, never an exception: ask a plain MMM for a
# latent-variable estimand and the registry answers with a stated reason.
# The evaluator keys results by their expanded target, so take the result
# whatever its key — the point here is the refusal, not the key shape.
ref = next(iter(evaluator.evaluate([registry.get("awareness_lift")]).values()))
print(f"\nawareness_lift -> status={ref.status!r}\n  reason: {ref.reason}")
"""

ESTIMAND_CHART = r"""
symbols = {"contribution_roi": "circle", "counterfactual_roi": "diamond",
           "marginal_roas": "square"}
offsets = {"contribution_roi": -0.22, "counterfactual_roi": 0.0, "marginal_roas": 0.22}
fig = go.Figure()
for n in names:
    sub = edf[edf["estimand"] == n]
    ys = [CHANNELS.index(c) + offsets[n] for c in sub["channel"]]
    fig.add_trace(go.Scatter(
        x=sub["mean"], y=ys, mode="markers", name=n,
        error_x=dict(type="data", symmetric=False,
                     array=sub["high"] - sub["mean"], arrayminus=sub["mean"] - sub["low"],
                     thickness=1.2, width=0),
        marker=dict(symbol=symbols[n], size=11,
                    color=[PALETTE[c] for c in sub["channel"]],
                    line=dict(color="white", width=1))))
for i, ch in enumerate(CHANNELS):
    fig.add_trace(go.Scatter(x=[scenario.true_roas[ch]], y=[i], mode="markers",
                             marker=dict(color=TRUTH, symbol="line-ns-open", size=24,
                                         line=dict(width=3)),
                             name="sealed causal truth", showlegend=(i == 0)))
fig.update_yaxes(tickmode="array", tickvals=list(range(len(CHANNELS))), ticktext=CHANNELS)
fig.update_xaxes(title="incremental Sales per $1 (94% interval)")
style(fig, 460, "Three different numbers are all called ROI — and the chasers are over-credited")
fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.18, x=0))
fig.show()

for ch in CHANNELS:
    m = float(edf[(edf.estimand == "contribution_roi") & (edf.channel == ch)]["mean"].iloc[0])
    t = float(scenario.true_roas[ch])
    tag = "CHASER" if ch in CHASERS else "      "
    print(f"{ch:8s} {tag}  contribution_roi {m:6.3f}   truth {t:6.3f}   over-credit {m - t:+.3f}")
"""

LEARNING = r"""
t0 = time.time()
learn = mmm.compute_parameter_learning(random_seed=7)
print(f"parameter_learning in {time.time()-t0:.0f}s")

vcolor = {"prior-dominated": BAD, "weak": GOLD, "moderate": MUTED,
          "strong": GOOD, "relocated": INK}
top = learn.head(14).iloc[::-1]
fig = go.Figure(go.Bar(
    x=top["contraction"], y=top["parameter"], orientation="h",
    marker=dict(color=[vcolor.get(v, MUTED) for v in top["verdict"]]),
    text=top["verdict"], textposition="outside", cliponaxis=False))
fig.add_vline(x=0, line=dict(color=INK, width=1))
fig.add_vline(x=0.5, line=dict(color=MUTED, dash="dot"),
              annotation_text="strong learning", annotation_position="top")
fig.update_xaxes(title="prior -> posterior contraction (1 = data determined it)")
style(fig, 480, "Which parameters did the data actually teach us about?")
fig.show()

n_dom = int((learn["verdict"] == "prior-dominated").sum())
print(f"{n_dom} of {len(learn)} parameters are prior-dominated "
      "(their posterior mostly restates the prior).")
print(learn[["parameter", "contraction", "overlap", "shift_z", "verdict"]]
      .head(6).round(2).to_string(index=False))
"""

IDENT = r"""
# A fresh MAP fit with the identification check on. The curvature report names
# what a point fit could not resolve: ridges (parameter trade-offs), uninformed
# coordinates, prior-determined coordinates — and counts how many parameters
# the DATA effectively determines, versus how many the model has.
t0 = time.time()
mmm_map = build_model(SPEC, str(DATA))
res_map = mmm_map.fit(method="map", identification_check=True, random_seed=3)
ident = res_map.diagnostics.get("identification", {})
print(f"MAP + identification check in {time.time()-t0:.0f}s\n")

print(f"verdict              : {ident.get('verdict')}")
print(f"parameters in model  : {ident.get('n_parameters')}")
eff = ident.get("effective_parameters")
print(f"effective parameters : {eff:.1f}" if eff is not None else "effective parameters : n/a")
print(f"condition index      : {ident.get('condition_index'):.1f}"
      if ident.get("condition_index") is not None else "condition index      : n/a")
for fd in (ident.get("flat_directions") or [])[:3]:
    combo = " + ".join(f"{v:+.2f}*{n}" for n, v in fd.get("loadings", [])[:3])
    print(f"ridge ({fd.get('kind')}): {combo}")
for p in (ident.get("prior_determined") or [])[:4]:
    print(f"prior-determined: {p.get('parameter')} "
          f"(data supplies {p.get('informed_fraction', 0):.0%} of its curvature)")
"""

SENS = r"""
from mmm_framework.validation.confounding_sensitivity import run_confounding_sensitivity

t0 = time.time()
report = run_confounding_sensitivity(mmm, random_seed=42)
print(f"confounding sensitivity (closed-form, no refit) in {time.time()-t0:.1f}s\n")
print(report.summary().to_string(index=False))
print(f"\nfragile channels     : {report.fragile_channels or 'none'}")
print(f"unassessable channels: {report.unassessable_channels or 'none'}")
print(f"\ncaveat: {report.caveat}")
"""

SURFACE = r"""
focus = "Search"
row = next(c for c in report.channels if c.channel == focus)
surf = row.sensitivity.surface
tip_mu = row.sensitivity.tipping_mu
bench_ok = [b for b in row.benchmarks if b.status == "ok" and np.isfinite(b.fractional_bias)]
strongest = max(bench_ok, key=lambda b: b.fractional_bias) if bench_ok else None

fig = go.Figure(go.Heatmap(
    x=list(surf.mu_grid), y=list(surf.sigma_grid),
    z=[list(r) for r in surf.prob], zmin=0, zmax=1,
    colorscale=[[0, BAD], [0.5, "#f5f0e8"], [1, GOOD]],
    colorbar=dict(title="P(above<br>break-even)")))
fig.add_contour(x=list(surf.mu_grid), y=list(surf.sigma_grid),
                z=[list(r) for r in surf.prob],
                contours=dict(start=surf.threshold, end=surf.threshold, size=1,
                              coloring="lines", showlabels=True),
                line=dict(color=INK, width=2), showscale=False)
if tip_mu.crossed and tip_mu.value is not None:
    fig.add_vline(x=tip_mu.value, line=dict(color=INK, dash="dash"),
                  annotation_text=f"tipping point {tip_mu.value:.0%}",
                  annotation_position="top")
if strongest is not None:
    fig.add_vline(x=strongest.fractional_bias, line=dict(color=PALETTE["TV"], width=2),
                  annotation_text=f"confounder as strong as {strongest.covariate} "
                                  f"({strongest.kd:g}x) -> {strongest.fractional_bias:.0%}",
                  annotation_position="bottom right")
fig.update_xaxes(title="assumed bias location mu (fraction of the estimate)", tickformat=".0%")
fig.update_yaxes(title="assumed bias spread sigma", tickformat=".0%")
style(fig, 460, f"{focus}: the analyst positions that do and do not support the conclusion")
fig.show()

print(row.describe())
"""

TIPPING = r"""
fig = go.Figure()
ys, tips, benches = [], [], []
for i, c in enumerate(report.channels):
    tm = c.sensitivity.tipping_mu
    tip = tm.value if (tm.crossed and tm.value is not None) else np.nan
    ok = [b.fractional_bias for b in c.benchmarks
          if b.status == "ok" and np.isfinite(b.fractional_bias)]
    ys.append(c.channel); tips.append(tip); benches.append(max(ok) if ok else np.nan)
fig.add_trace(go.Bar(y=ys, x=tips, orientation="h", name="tipping point (bias that flips the call)",
                     marker_color=[PALETTE[c] for c in ys], opacity=0.85))
fig.add_trace(go.Scatter(y=ys, x=benches, mode="markers",
                         name="strongest observed-covariate benchmark",
                         marker=dict(color=TRUTH, symbol="line-ns-open", size=22,
                                     line=dict(width=3))))
fig.update_xaxes(title="bias as a fraction of the channel's own estimate", tickformat=".0%")
style(fig, 400, "How much hidden bias would change the call — and how much is plausible")
fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))
fig.show()

for c in report.channels:
    v = c.benchmark_exceeds_tipping_point
    verdict = {True: "a plausible confounder OVERTURNS it", False: "survives Price-strength confounding",
               None: "not comparable (missing side)"}[v]
    print(f"{c.channel:8s} verdict={c.sensitivity.verdict:12s} {verdict}")
"""

BACKTEST = r"""
from mmm_framework.validation import BacktestConfig, run_backtest

bt_config = BacktestConfig(
    min_train_size=78, horizon=13, step=13, max_origins=2,
    draws=150, tune=150, chains=2, coverage_levels=(0.8,), random_seed=42,
)
t0 = time.time()
bt = run_backtest(mmm, bt_config, progressbar=False)
print(f"backtest: {bt.n_origins} refits + {len(bt.records)} graded forecasts "
      f"in {time.time()-t0:.0f}s\n")
print(bt.summary().round(3).to_string(index=False))
print("\nper-origin fit health:")
print(bt.fits.round(2).to_string(index=False))
"""

BACKTEST_CHART = r"""
fig = go.Figure()
fig.add_trace(go.Scatter(x=scenario.weeks, y=scenario.y, name="actual Sales",
                         line=dict(color=MUTED, width=1.6)))
fold_colors = [PALETTE["TV"], PALETTE["Social"], PALETTE["Display"]]
for k, (origin, grp) in enumerate(bt.records.groupby("origin")):
    col = fold_colors[k % len(fold_colors)]
    fig.add_trace(go.Scatter(
        x=pd.concat([grp["date"], grp["date"][::-1]]),
        y=pd.concat([grp["hi_80"], grp["lo_80"][::-1]]),
        fill="toself", fillcolor=col, opacity=0.18, line=dict(width=0),
        name="80% interval", showlegend=(k == 0), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=grp["date"], y=grp["y_pred"], mode="lines+markers",
                             name=f"fold {k+1} forecast (origin wk {origin})",
                             line=dict(color=col, width=2.2), marker=dict(size=5)))
    fig.add_vline(x=grp["date"].iloc[0], line=dict(color=col, dash="dot", width=1))
fig.update_yaxes(title="Sales")
style(fig, 460, "Out-of-time folds: train on the past, forecast a quarter, grade against actuals")
fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, x=0))
fig.show()

cov = float(bt.records["cov_80"].mean())
print(f"MMM out-of-time MAPE {bt.mape:.1%}; 80% intervals covered {cov:.0%} of actuals.")
print("Doctrine: this CONFOUNDED model forecasts fine — the backtest validates the")
print("predictive model, not the causal one. Section 2 showed the same fit "
      "over-crediting the demand chasers.")
"""

SPECCURVE = r"""
from mmm_framework.validation import run_spec_curve, default_spec_variants

variants = default_spec_variants(SPEC)
print("pre-registered spec set:", [v.name for v in variants])

# Each variant is a full refit. To keep this demo fast we inject a variational
# fit (seconds per spec); a production sweep uses the default NUTS path and its
# calibrated uncertainty. The sweep machinery is identical either way.
def advi_fit(spec, dataset_path):
    m = build_model(spec, dataset_path)
    m.fit(method="advi", n=6000, draws=300, random_seed=42)
    return m

t0 = time.time()
sc = run_spec_curve(SPEC, str(DATA), variants=variants, compute_loo=False,
                    fit_fn=advi_fit, max_draws=300, random_seed=42)
print(f"spec curve: {len(sc.specs)} specs in {time.time()-t0:.0f}s")

focus = "Search"
ok_fits = [f for f in sc.fits if not f.error and focus in f.roi]
ok_fits.sort(key=lambda f: f.roi[focus]["mean"])
xs = [f.roi[focus]["mean"] for f in ok_fits]
names_x = [f.name + (" (primary)" if f.primary else "") for f in ok_fits]
fig = go.Figure()
bma = sc.bma.get(focus, {})
if bma:
    fig.add_hrect(y0=bma["lower"], y1=bma["upper"], fillcolor=GOLD, opacity=0.15,
                  line_width=0, annotation_text="model-averaged (equal weights)",
                  annotation_position="top left")
    fig.add_hline(y=bma["mean"], line=dict(color=GOLD, width=2))
fig.add_hline(y=float(scenario.true_roas[focus]), line=dict(color=TRUTH, dash="dash"),
              annotation_text="sealed causal truth", annotation_position="bottom right")
fig.add_trace(go.Scatter(
    x=names_x, y=xs, mode="markers",
    error_y=dict(type="data", symmetric=False,
                 array=[f.roi[focus]["upper"] - f.roi[focus]["mean"] for f in ok_fits],
                 arrayminus=[f.roi[focus]["mean"] - f.roi[focus]["lower"] for f in ok_fits]),
    marker=dict(color=[PALETTE[focus] if f.primary else MUTED for f in ok_fits],
                size=12, line=dict(color="white", width=1)),
    showlegend=False))
fig.update_yaxes(title=f"{focus} ROI")
style(fig, 430, f"{focus}'s over-credit survives every defensible spec — robustly wrong")
fig.show()

print(f"\nrobustness summary for {focus}: {sc.robustness.get(focus)}")
"""

REFUTATION = r"""
from mmm_framework.validation import ModelValidator, ValidationConfig
from mmm_framework.validation.config import CausalRefutationConfig

cfg = ValidationConfig()
cfg.run_ppc = cfg.run_residuals = cfg.run_channel_diagnostics = False
cfg.run_causal_refutation = True
cfg.verbose = False
cfg.causal_refutation = CausalRefutationConfig(
    draws=120, tune=120, chains=2, run_random_common_cause=False, random_seed=1234,
)

t0 = time.time()
summary = ModelValidator(mmm).validate(cfg)
cr = summary.causal_refutation
print(f"refutation suite ({len(cr.tests)} refits) in {time.time()-t0:.0f}s\n")
print(cr.summary().to_string(index=False))
print(f"\npassed {cr.n_passed}/{len(cr.tests)}; underpowered flag: {cr.underpowered}")
for t in cr.tests:
    print(f"\n{t.name} [{t.kind}] {'PASS' if t.passed else 'FAIL'} — {t.details or t.description}")
"""

CALIB_PRIOR = r"""
from mmm_framework.calibration import ExperimentCalibrator, LiftTestResult

# A synthetic geo holdout on Search. In production this number comes from a real
# experiment; here the sealed truth plays that role (a well-run holdout reveals
# the true incremental contribution), with a 10% measurement SE.
ch = "Search"
lt = LiftTestResult(
    channel=ch,
    test_period=(PERIODS[0], PERIODS[-1]),
    measured_lift=float(scenario.true_contribution[ch]),
    lift_se=float(scenario.true_contribution[ch]) * 0.10,
)
calibrator = ExperimentCalibrator(mmm)
prior_report = calibrator.derive_priors([lt])   # cheap: derives the prior, NO refit
cc = prior_report.channel_calibrations[0]
print(prior_report.summary().to_string(index=False))

from scipy.stats import gamma as _gamma
a_, rate = cc.roi_prior.params["alpha"], cc.roi_prior.params["beta"]
xs = np.linspace(max(1e-6, cc.beta_target - 4 * cc.beta_sigma),
                 cc.beta_target + 4 * cc.beta_sigma, 240)
fig = go.Figure(go.Scatter(x=xs, y=_gamma.pdf(xs, a=a_, scale=1.0 / rate), mode="lines",
                           fill="tozeroy", line=dict(color=PALETTE[ch], width=2),
                           name="experiment-anchored prior on beta"))
fig.add_vline(x=cc.beta_fit_mean, line=dict(color=BAD, width=2, dash="dot"),
              annotation_text=f"fitted beta = {cc.beta_fit_mean:.3f} (over-credited)",
              annotation_position="top right")
fig.add_vline(x=cc.beta_target, line=dict(color=GOOD, width=2, dash="dash"),
              annotation_text=f"experiment target = {cc.beta_target:.3f}",
              annotation_position="top left")
fig.update_xaxes(title="Search coefficient (beta)")
fig.update_yaxes(title="prior density")
style(fig, 380, "One experiment becomes a tight prior anchored below the confounded fit")
fig.show()
"""

CALIB_REFIT = r"""
from mmm_framework.validation.spec_curve import channel_roi_draws

t0 = time.time()
outcome = calibrator.calibrate([lt], draws=200, tune=200, chains=2, random_seed=11)
print(f"calibrated refit in {time.time()-t0:.0f}s")

roi_before = channel_roi_draws(mmm, CHANNELS, random_seed=42)
roi_after = channel_roi_draws(outcome.model, CHANNELS, random_seed=42)
truth = float(scenario.true_roas[ch])

fig = go.Figure()
for label, draws, yy, col in [
    ("before — confounded observational fit", roi_before[ch], 1, BAD),
    ("after — experiment folded in", roi_after[ch], 0, PALETTE[ch]),
]:
    lo, hi = np.percentile(draws, [3, 97])
    fig.add_trace(go.Scatter(x=[lo, hi], y=[yy, yy], mode="lines",
                             line=dict(color=col, width=10), opacity=0.4, showlegend=False))
    fig.add_trace(go.Scatter(x=[float(np.mean(draws))], y=[yy], mode="markers+text",
                             text=[label], textposition="top center",
                             marker=dict(color=col, size=15, line=dict(color="white", width=1.5)),
                             showlegend=False))
fig.add_vline(x=truth, line=dict(color=GOOD, width=2.5, dash="dash"),
              annotation_text=f"sealed truth = {truth:.2f}", annotation_position="bottom right")
fig.update_yaxes(range=[-0.6, 1.7], visible=False)
fig.update_xaxes(title=f"{ch} ROI (incremental Sales per $1)")
style(fig, 360, f"{ch}: one experiment pulls the confounded ROI back toward the truth")
fig.show()

b, a = float(np.mean(roi_before[ch])), float(np.mean(roi_after[ch]))
print(f"{ch} ROI: before {b:.3f} -> after {a:.3f} -> truth {truth:.3f}")
print(f"gap to truth closed by {100 * (1 - abs(a - truth) / abs(b - truth)):.0f}%")
"""

CELLS = [
    md(r"""
# Before you believe a number

An MMM will always produce an ROI. The question this notebook answers is what
the framework gives you **before you believe it** — the trust stack:

1. **Estimands** (`estimands/`) — three different numbers are all called "ROI";
   the declarative registry names them and computes each honestly.
2. **Diagnostics** (`diagnostics/`) — did the data actually inform the
   parameters (prior→posterior learning), can it (identification), and how much
   hidden confounding would flip the call (bias sensitivity + the
   "confounder as strong as Price" benchmark)?
3. **Validation** (`validation/`) — out-of-time backtesting, the spec-curve
   robustness sweep, and the causal refutation suite.
4. **Calibration** (`calibration/`) — the one tool that genuinely fixes
   confounding: fold a randomized experiment into the model.

To make the stakes real, everything runs on a world **built to deceive**:
latent demand drives both the budgets of two "chasing" channels and baseline
sales, and the model never sees it. The world carries a sealed causal answer
key, so every diagnostic below has something real to find.
"""),
    code(SETUP),
    md(r"""
## 1 · A world built to deceive, and one honest fit

`synth.dgp.make_unobserved_confounding` is the classic MMM trap: budgets rise
when demand is expected to rise. Search and Social chase latent demand hard;
TV and Display barely do. The model gets spend, Sales, and Price — never
demand. We fit the framework's standard Bayesian MMM (geometric adstock, Hill
saturation) with a NUTS-lite budget.
"""),
    code(WORLD),
    code(FIT),
    md(r"""
## 2 · Three different numbers are all called "ROI"

Ask three tools for "the ROI" and you can get three answers, all defensible,
none interchangeable. The estimand registry makes each one a **named,
serializable object** with its own definition, assumptions, and interval:

- `contribution_roi` — the decomposition the dashboard shows: the model's
  in-graph channel contribution divided by spend.
- `counterfactual_roi` — set the channel's spend to zero, re-predict, and
  difference: "what would we have lost". Saturation and carryover make this a
  genuinely different number from the decomposition.
- `marginal_roas` — the return on the **next** 10% of spend, which is what a
  budget decision actually needs. On a saturated channel it sits far below both.

One fit, all three, side by side, against the sealed truth.
"""),
    code(ESTIMANDS),
    code(ESTIMAND_CHART),
    md(r"""
Two things to take away. First, the three estimands genuinely differ per
channel (the marginal number is systematically lower — saturation at work), so
"the ROI" without a name is not a number. Second, the demand chasers come back
**over-credited**: the model attributes demand-driven sales to the spend that
chased them. Nothing in the fit itself warned us — that is what the rest of
this notebook is for.
"""),
    md(r"""
## 3 · Did the data actually speak? Prior→posterior learning

A posterior can look confident while merely restating its prior. The learning
diagnostic compares each parameter's prior to its posterior and reports the
**contraction** (how much the variance shrank), the overlap, and the location
shift, with a verdict per parameter. Prior-dominated parameters are the ones
whose "estimates" the data never touched.
"""),
    code(LEARNING),
    md(r"""
## 4 · Could the data have spoken? Identification

Learning asks whether the data moved the posterior; identification asks
whether it *could*. A guarded MAP fit with `identification_check=True` reads
the curvature at the optimum and names three distinct failures that are
routinely confused: **ridges** (parameter combinations that trade off, e.g. a
saturation elbow against its coefficient), **uninformed** coordinates (no
curvature at all), and **prior-determined** coordinates (plenty of curvature,
all of it the prior's). `effective_parameters` counts how many parameters the
data determines, against how many the model has.
"""),
    code(IDENT),
    md(r"""
## 5 · Pricing the assumption no dataset can check

Every causal read rests on "no unobserved confounder" — exactly what this world
violates. You cannot test that assumption, but you can **price** it: decompose
the observed effect as `truth + bias`, put a prior on the bias, and ask which
commitments still support the conclusion. Everything is closed-form on the
draws the model already produced (no refit, no seed).

Two outputs matter. The **tipping point**: how large a bias would flip the
channel below break-even. And the **benchmark** that keeps the tipping point
from being a slider of guesses: Cinelli–Hazlett bounds priced against the
covariates we *did* measure — "a confounder as strong as Price would move this
channel by X% of its estimate".
"""),
    code(SENS),
    code(SURFACE),
    code(TIPPING),
    md(r"""
Read the surface: green positions (analyst commitments about the bias) leave
the channel's case intact, red ones do not, and the black contour is the
decision threshold. The benchmark line is what makes it actionable — when a
confounder merely *as strong as Price* already implies a bias past the tipping
point, the honest verdict is that the conclusion is fragile. On this world it
should be: we planted a confounder far stronger than Price.

### SBC and interval coverage — the slow half of the diagnostics story

Two more diagnostics live in `diagnostics/` that this notebook deliberately
does not run, because each needs tens of refits: **simulation-based
calibration** (`run_mmm_sbc` — draw truths from the prior, refit, check the
ranks are uniform) and **recovery coverage** (`run_recovery_coverage` — does a
90% interval actually cover the truth 90% of the time; the top cause of "my
90% interval covers 50%" is an approximate fit). The API is two calls:

```python
from mmm_framework.diagnostics import run_mmm_sbc, run_recovery_coverage
sbc = run_mmm_sbc(mmm, n_sims=100, draws=300, tune=300)
cov = run_recovery_coverage(mmm, n_replicates=50)
```

The dedicated companions bake them properly: `technical-docs/coverage-diagnostics.md`
and the stress/causal notebook series linked at the end.
"""),
    md(r"""
## 6 · Backtesting: grade the forecasts, and know what that does not prove

The rolling-origin backtest refits the exact specification on an expanding
training window, forecasts a quarter past each cutoff, and grades against
held-out actuals and naive baselines. No information leaks across the cutoff —
carryover is convolved from training-window spend only.

It also sets up this notebook's most important negative lesson, so watch for
it below.
"""),
    code(BACKTEST),
    code(BACKTEST_CHART),
    md(r"""
The confounded model forecasts **well** — it beats both naive baselines and
its intervals cover. Forecast accuracy validates the *predictive* model, not
the causal one: latent demand helps the model predict (spend proxies demand)
while poisoning the attribution. A good backtest must never be read as
evidence that the ROI is right.
"""),
    md(r"""
## 7 · The spec curve: is the number an artifact of one modeling choice?

Any single specification embeds choices (adstock form, saturation form) that a
reviewer can reasonably contest. The spec-curve sweep pre-registers a grid of
defensible variants, fits each, and model-averages the estimand with **equal
weights** — deliberately not predictive stacking, because two specs can predict
the KPI equally well while splitting it very differently between media and
baseline.

It is also an honesty check on robustness itself, and this world makes the
point sharply.
"""),
    code(SPECCURVE),
    md(r"""
Every defensible spec over-credits Search by a similar amount, so the estimate
is **robustly wrong**: the spec curve tests sensitivity to *modeling choices*,
and confounding is not a modeling choice — it is in the data. Robustness
across specs is necessary for trust, never sufficient. That is why the
sensitivity analysis of section 5 and the experiment of section 8 exist.
"""),
    md(r"""
## 8 · The refutation suite: tests the model can fail

Four falsification tests refit the model on perturbed data, each with a
pre-stated pass criterion. Because the media priors have positive support, a
placebo coefficient never collapses to zero — so the vanishing tests are
graded on **fit**, not on the coefficient: scrambled media must add no
explanatory power, and a scrambled KPI must be unfittable. Stability tests
check the coefficients barely move under a random subset. A "pass" from an
underpowered refit is flagged rather than oversold.
"""),
    code(REFUTATION),
    md(r"""
Note what a clean sweep here means: the model is not *spuriously* fitting
noise, and its estimates are stable. Like the backtest, refutation cannot see
a confounder that lives in the data itself — the suite is one gate in the
stack, not a certificate of causality.

## 9 · Calibration: the one tool that actually fixes it

Diagnostics priced the problem; an experiment solves it. A geo holdout on
Search measures its true incremental contribution directly. `calibration/`
turns that readout into model structure two ways: a tight, experiment-anchored
**prior** on the channel coefficient (cheap, shown first), and an in-graph
likelihood on the ROAS estimand (the headline route, demonstrated in the
lifecycle series). Here the sealed truth plays the role of the experiment
readout, with a realistic 10% measurement SE.
"""),
    code(CALIB_PRIOR),
    code(CALIB_REFIT),
    md(r"""
The over-credited channel slides back toward its sealed truth, and only direct
causal evidence achieved that — no amount of fit quality, spec robustness, or
refutation passing could. That ordering is the design philosophy of the whole
framework: diagnose what you can, price what you cannot, and buy the missing
information with an experiment.

## Where to go deeper

- **Estimands** — `technical-docs/estimands.md`; windowed estimands and
  interval provenance in `technical-docs/engineering-notes.md` (*Declarative
  estimands*).
- **Bias sensitivity + benchmarking** — `technical-docs/confounding-sensitivity.md`;
  the no-unobserved-confounding pricing shipped in the diagnostics module.
- **Coverage / SBC** — `technical-docs/coverage-diagnostics.md`; the
  `nbs/stress/` series pressure-tests five silent failure modes, and
  `nbs/causal/` (00–10) builds the causal ladder from scratch.
- **Backtesting** — the dedicated `nbs/demos/` backtest notebook and
  `mmm_framework.validation.backtest` docstrings (forecast doctrine included).
- **Identification** — `technical-docs/sampling-failure-playbook.md` and the
  *ZeroDivisionError … MAP / Laplace* note in the engineering notes.
- **Calibration end-to-end** — `nbs/lifecycle/` (T0→T5 measurement loop;
  `lifecycle_04_calibrate` runs the in-graph ROAS-likelihood route) and
  `nbs/causal/causal_06_calibrating_the_model`.
- **Experiment design economics** — `technical-docs/experiment-economics.md`
  (what the next test is worth before you run it).
"""),
]


def main() -> None:
    nb = new_notebook(cells=CELLS)
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    out = "showcase/showcase_04_earning_trust.ipynb"
    with open(out, "w") as fh:
        nbformat.write(nb, fh)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
