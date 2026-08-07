"""Author showcase/showcase_02_model_anatomy.ipynb (run from ``nbs/``).

    uv run --with nbformat python builders/build_showcase_02_model_anatomy.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        showcase/showcase_02_model_anatomy.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

Feature-showcase notebook 02 — "One model, every knob". A tour of the model
anatomy: the ModelConfigBuilder fluent API, adstock and saturation families,
trend and seasonality bases, ROI-mode vs coefficient-mode media priors,
likelihood families and the multiplicative specification, the opt-in structure
levers (price/promo, events, reach/frequency, interactions, control selection)
shown reaching the graph via named RVs, the three NUTS backends, fast
approximate fits with the calibration caveat measured, and the frequentist
ridge path compared to Bayesian on the same sealed-truth world.
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
from mmm_framework.synth import dgp

sc = dgp.make_clean(seed=7, n_weeks=104)
CH = sc.channels                       # ["TV", "Search", "Social", "Display"]
panel = sc.panel()                     # the PanelDataset every model takes

fig = make_subplots(specs=[[{"secondary_y": True}]])
for c in CH:
    fig.add_trace(go.Scatter(x=sc.weeks, y=sc.spend[c], name=c, stackgroup="spend",
                             mode="none", fillcolor=PALETTE[c], opacity=0.8))
fig.add_trace(go.Scatter(x=sc.weeks, y=sc.y, name="Sales (KPI)",
                         line=dict(color=INK, width=2)), secondary_y=True)
fig.update_yaxes(title="weekly spend", secondary_y=False)
fig.update_yaxes(title="KPI", secondary_y=True)
style(fig, 400, "One small world, four channels, a sealed causal answer key")
fig.show()

print("channels:", CH, "| weeks:", len(sc.weeks), "| controls:", list(sc.controls.columns))
print("true per-channel contribution (counterfactual zero-out, KPI units):")
print(sc.true_contribution.round(0).to_string())
"""

BUILDER = r"""
from mmm_framework import BayesianMMM, ModelConfig, ModelConfigBuilder, TrendConfig
from mmm_framework.model import TrendType

config = (
    ModelConfigBuilder()
    .additive()                    # or .multiplicative()  (semi-log)
    .bayesian_numpyro()            # or .bayesian_pymc() / .bayesian_nutpie() / .frequentist_ridge()
    .with_chains(4).with_draws(1000).with_tune(1000)
    .with_target_accept(0.9)
    .with_media_prior_mode("roi")  # default media prior on the DECISION scale
    .build()
)
print(type(config).__name__, "-", "a validated Pydantic object, not a dict")
print({k: v for k, v in config.model_dump().items()
       if k in ("specification", "inference_method", "n_chains", "n_draws",
                "media_prior_mode", "media_roi_prior_mu", "media_roi_prior_sigma")})

TREND = TrendConfig(type=TrendType.LINEAR)
print("\ntrend types available:", [t.value for t in TrendType])
"""

ADSTOCK_KERNELS = r"""
from mmm_framework.transforms import adstock_weights

L = 12
fig = make_subplots(rows=1, cols=3, shared_yaxes=True, subplot_titles=(
    "geometric - decay from week 0",
    "delayed - the peak can move",
    "weibull - shape controls the whole profile"))
for a, col in zip([0.3, 0.6, 0.85], ["#9db3e0", "#4464ad", "#22345c"]):
    fig.add_trace(go.Bar(x=np.arange(L), y=adstock_weights("geometric", L, alpha=a),
                         name=f"alpha={a}", marker_color=col), row=1, col=1)
for th, col in zip([0.0, 2.0, 4.0], ["#e0cf9d", "#c9962e", "#7a5a15"]):
    fig.add_trace(go.Bar(x=np.arange(L), y=adstock_weights("delayed", L, alpha=0.75, theta=th),
                         name=f"theta={th}", marker_color=col), row=1, col=2)
for k, col in zip([0.8, 2.0, 3.5], ["#9dc7b2", "#3d7a5c", "#1e4534"]):
    fig.add_trace(go.Bar(x=np.arange(L), y=adstock_weights("weibull", L, shape=k, scale=4.0),
                         name=f"shape={k}", marker_color=col), row=1, col=3)
for c in (1, 2, 3):
    fig.update_xaxes(title="lag (weeks)", row=1, col=c)
fig.update_yaxes(title="weight", row=1, col=1)
style(fig, 380, "Three adstock families - same question, three answers about WHEN effect lands",
      barmode="group", legend=dict(orientation="h", y=-0.25))
fig.show()
"""

ADSTOCK_PULSE = r"""
from mmm_framework.transforms import apply_adstock

x = sc.spend["TV"].to_numpy()[:40]
fig = go.Figure()
fig.add_trace(go.Bar(x=np.arange(40), y=x, name="raw TV spend",
                     marker_color=PALETTE["TV"], opacity=0.35))
for kind, kw, col in [("geometric", dict(alpha=0.7), PALETTE["TV"]),
                      ("weibull", dict(shape=2.0, scale=4.0), GOOD)]:
    w = adstock_weights(kind, 12, **kw)
    fig.add_trace(go.Scatter(x=np.arange(40), y=apply_adstock(x, w),
                             name=f"adstocked ({kind})", line=dict(color=col, width=2.5)))
fig.update_xaxes(title="week"); fig.update_yaxes(title="spend / effective spend")
style(fig, 380, "Adstock reshapes the spend curve the response actually sees")
fig.show()
"""

SATURATION = r"""
# Hill / Michaelis-Menten / tanh live in the graph only; the single place their
# formulas exist is model.base._apply_saturation_pt (the likelihood AND the
# counterfactual estimands both route through it, so they cannot drift apart).
# We evaluate that exact function here rather than re-typing the math.
import pytensor.tensor as pt
from mmm_framework.model.base import _apply_saturation_pt
from mmm_framework.config import SaturationType
from mmm_framework.transforms import logistic_saturation, root_saturation

xg = np.linspace(0, 1.2, 200)
xt = pt.constant(xg)
curves = {
    "logistic": logistic_saturation(xg, lam=3.0),
    "hill": _apply_saturation_pt(xt, SaturationType.HILL,
                                 {"sat_half": 0.4, "sat_slope": 2.5}).eval(),
    "michaelis_menten": _apply_saturation_pt(xt, SaturationType.MICHAELIS_MENTEN,
                                             {"sat_half": 0.3}).eval(),
    "tanh": _apply_saturation_pt(xt, SaturationType.TANH, {"sat_half": 0.5}).eval(),
    "root": root_saturation(xg, exponent=0.5),
}
cols = {"logistic": PALETTE["TV"], "hill": PALETTE["Search"],
        "michaelis_menten": PALETTE["Social"], "tanh": PALETTE["Display"], "root": INK}
fig = go.Figure()
for name, y in curves.items():
    fig.add_trace(go.Scatter(x=xg, y=y, name=name, line=dict(color=cols[name], width=2.5)))
fig.add_trace(go.Scatter(x=xg, y=xg, name="none (identity)",
                         line=dict(color=MUTED, dash="dot")))
fig.update_xaxes(title="adstocked, normalized spend")
fig.update_yaxes(title="saturated response")
style(fig, 420, "Five saturation families - only Hill (slope>1) gives a genuine S-curve")
fig.show()
"""

TREND_FIG = r"""
from mmm_framework.transforms import create_bspline_basis, create_piecewise_trend_matrix

n = len(sc.weeks)
t = np.linspace(0, 1, n)
rng = np.random.default_rng(3)

fig = make_subplots(rows=1, cols=3, subplot_titles=(
    "linear - one slope", "piecewise - slope changes at changepoints",
    "spline - smooth local flexibility"))
fig.add_trace(go.Scatter(x=t, y=0.8 * t, line=dict(color=PALETTE["TV"], width=2.5),
                         showlegend=False), row=1, col=1)
s, A = create_piecewise_trend_matrix(t, n_changepoints=8)
for _ in range(3):
    delta = rng.normal(0, 0.6, len(s))
    # Prophet-style: g(t) = (k + A@delta) * t + A @ (-s * delta)
    y = (0.5 + A @ delta) * t + A @ (-s * delta)
    fig.add_trace(go.Scatter(x=t, y=y, line=dict(width=2), opacity=0.75,
                             showlegend=False), row=1, col=2)
for x0 in s:
    fig.add_vline(x=float(x0), line=dict(color=MUTED, width=0.5, dash="dot"), row=1, col=2)
B = create_bspline_basis(t, n_knots=6)
for _ in range(3):
    fig.add_trace(go.Scatter(x=t, y=B @ rng.normal(0, 0.5, B.shape[1]),
                             line=dict(width=2), opacity=0.75, showlegend=False), row=1, col=3)
for c in (1, 2, 3):
    fig.update_xaxes(title="time (scaled)", row=1, col=c)
style(fig, 360, "Trend types trade flexibility for identification risk (GP is the far end)")
fig.show()
print("piecewise design:", A.shape, "| bspline basis:", B.shape,
      "| basis rows sum to 1:", bool(np.allclose(B.sum(axis=1), 1.0)))
"""

SEASONALITY = r"""
from mmm_framework.transforms import create_fourier_features

wk = np.arange(104)
rng = np.random.default_rng(11)
fig = go.Figure()
for order, col in [(1, "#9db3e0"), (2, PALETTE["TV"]), (4, "#22345c")]:
    F = create_fourier_features(wk, period=52.0, order=order)
    coefs = rng.normal(0, 1.0, F.shape[1]) / np.repeat(np.arange(1, order + 1), 2)
    fig.add_trace(go.Scatter(x=wk, y=F @ coefs, name=f"order {order} ({F.shape[1]} cols)",
                             line=dict(color=col, width=2.5)))
fig.update_xaxes(title="week"); fig.update_yaxes(title="seasonal component")
style(fig, 380, "Fourier seasonality - more harmonics buy sharper shape, and more overfit risk")
fig.show()
"""

ROI_PRIORS = r"""
import pymc as pm

m_roi = BayesianMMM(panel, ModelConfig(media_prior_mode="roi"), TREND)
m_coef = BayesianMMM(panel, ModelConfig(), TREND)   # historical coefficient default

free_roi = {v.name for v in m_roi.model.free_RVs}
free_coef = {v.name for v in m_coef.model.free_RVs}
print("ROI mode free RVs (media):   ", sorted(n for n in free_roi if n.startswith("roi_")))
print("coef mode free RVs (media):  ", sorted(n for n in free_coef if n.startswith("beta_")))
print("in ROI mode, beta_TV is derived in-graph:",
      "beta_TV" in {d.name for d in m_roi.model.deterministics})

roi_draws = {c: pm.draw(m_roi.model[f"roi_{c}"], draws=4000, random_seed=5) for c in CH}
fig = make_subplots(rows=1, cols=2, subplot_titles=(
    "prior ROI per channel (ROI mode)", "what the same channels' beta priors imply"))
for c in CH:
    fig.add_trace(go.Histogram(x=roi_draws[c], nbinsx=80, name=c, opacity=0.55,
                               marker_color=PALETTE[c]), row=1, col=1)
beta_draws = pm.draw(m_coef.model["beta_TV"], draws=4000, random_seed=5)
fig.add_trace(go.Histogram(x=beta_draws, nbinsx=80, name="beta_TV (coef mode, Gamma)",
                           marker_color=MUTED, opacity=0.7), row=1, col=2)
fig.add_vline(x=1.0, line=dict(color=BAD, dash="dash"), row=1, col=1)
fig.add_annotation(x=1.0, y=1.02, yref="y domain", text="break-even",
                   showarrow=False, font=dict(color=BAD), row=1, col=1)
fig.update_xaxes(title="prior ROI ($ KPI per $ spend)", range=[0, 6], row=1, col=1)
fig.update_xaxes(title="standardized coefficient", row=1, col=2)
style(fig, 400, "An ROI prior lives on the decision scale; a coefficient prior does not",
      barmode="overlay", legend=dict(orientation="h", y=-0.25))
fig.show()
print(f"prior P(ROI > 1) for TV: {(roi_draws['TV'] > 1).mean():.0%} "
      "(LogNormal(0,1) default: median exactly break-even)")
"""

LIKELIHOOD = r"""
from mmm_framework.config import LikelihoodConfig, ModelSpecification
from scipy import stats

# Student-t likelihood: same mean structure, heavier observation tails - one
# promo-week outlier stops dragging every coefficient.
cfg_t = ModelConfig(likelihood=LikelihoodConfig.student_t(nu=4))
m_t = BayesianMMM(panel, cfg_t, TREND)
print("student_t builds; likelihood:", m_t.model_config.likelihood.family.value,
      "| link:", m_t.model_config.likelihood.link.value,
      "| nu:", m_t.model_config.likelihood.params["nu"])
print("families available:", [f.value for f in type(cfg_t.likelihood.family)])

# Multiplicative (semi-log) specification: KPI modeled on the log scale, each
# channel's lift runs 1x -> exp(beta) at full saturation. Same saturation curve.
m_mult = BayesianMMM(panel, ModelConfig(specification=ModelSpecification.MULTIPLICATIVE), TREND)
_ = m_mult.model
print("multiplicative builds; _multiplicative flag:", m_mult._multiplicative,
      "(requires a strictly positive KPI and a Gaussian-scale likelihood)")

xg = np.linspace(-6, 6, 400)
fig = go.Figure()
fig.add_trace(go.Scatter(x=xg, y=stats.norm.pdf(xg), name="normal",
                         line=dict(color=PALETTE["TV"], width=2.5)))
fig.add_trace(go.Scatter(x=xg, y=stats.t.pdf(xg, df=4), name="student_t (nu=4)",
                         line=dict(color=BAD, width=2.5)))
fig.update_yaxes(type="log", title="density (log scale)")
fig.update_xaxes(title="residual (sd units)")
style(fig, 360, "Student-t buys outlier robustness - a 4-sigma week is ~30x more plausible")
fig.show()
print(f"P(|resid| > 4 sd): normal {2*stats.norm.sf(4):.2e} vs t(4) {2*stats.t.sf(4, df=4):.2e}")
"""

ANATOMY_PANEL = r"""
# Same world, more columns: we extend the clean world's panel with lever and
# structure columns (promo flags, an ad frequency series, two noise controls)
# so every opt-in feature below can be built on ONE dataset. Structure only -
# nothing here is fitted.
from mmm_framework.config import (
    CausalControlRole, ControlVariableConfig, DimensionType,
    KPIConfig, MediaChannelConfig, MFFConfig,
)
from mmm_framework.data_loader import PanelCoordinates, PanelDataset

rng = np.random.default_rng(21)
n = len(sc.weeks)
ctrl = pd.DataFrame({
    "Price": sc.controls["Price"].to_numpy(),
    "Promo": (rng.random(n) < 0.22) * rng.uniform(0.2, 0.6, n),
    "Frequency": 2.0 + np.abs(rng.normal(1.0, 0.5, n)),
    "Weather": rng.normal(0, 1, n),
    "Competitor": rng.normal(0, 1, n),
})
roles = {"Price": CausalControlRole.CONFOUNDER}
mff_cfg = MFFConfig(
    kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
    media_channels=[MediaChannelConfig(name=c, dimensions=[DimensionType.PERIOD]) for c in CH],
    controls=[ControlVariableConfig(name=c, dimensions=[DimensionType.PERIOD],
                                    causal_role=roles.get(c)) for c in ctrl.columns],
)
anatomy = PanelDataset(
    y=sc.y.rename("Sales"), X_media=sc.spend.copy(), X_controls=ctrl,
    coords=PanelCoordinates(periods=sc.weeks, geographies=None, products=None,
                            channels=CH, controls=list(ctrl.columns)),
    index=sc.weeks, config=mff_cfg,
)
print("anatomy panel:", anatomy.X_media.shape, "media,", anatomy.X_controls.shape, "controls")
"""

REACHABILITY = r"""
from mmm_framework.config import (
    ChannelInteraction, ControlSelectionConfig, EventsConfig, EventSpec,
    FrequencyResponse, PriceConfig, PromoConfig, ReachFrequencyConfig,
)

base = BayesianMMM(anatomy, ModelConfig(), TREND)
base_named = set(base.model.named_vars)
print(f"baseline graph: {len(base_named)} named vars; every feature below is OFF "
      "by default (graph byte-identical without it)\n")

features = {
    "price + promo levers": ModelConfig(
        price=PriceConfig(variable="Price", reference="median"),
        promotions=[PromoConfig(variable="Promo", adstock_lmax=4)]),
    "events": ModelConfig(
        events=EventsConfig(custom_events=[EventSpec(
            name="Launch", dates=[str(sc.weeks[40].date())], post_weeks=3, decay=0.5)])),
    "reach x frequency": ModelConfig(
        reach_frequency=[ReachFrequencyConfig(
            channel="TV", frequency_column="Frequency",
            response=FrequencyResponse.HILL)]),
    "channel interactions": ModelConfig(
        channel_interactions=[ChannelInteraction(
            channel_a="TV", channel_b="Search", expected_sign="positive")]),
    "control selection": ModelConfig(
        control_selection=ControlSelectionConfig(method="horseshoe", expected_nonzero=2)),
}
rows = []
for label, cfg in features.items():
    m = BayesianMMM(anatomy, cfg, TREND)
    new = sorted(set(m.model.named_vars) - base_named)
    rows.append({"feature": label, "new named vars in the graph": ", ".join(new)})
with pd.option_context("display.max_colwidth", 120):
    print(pd.DataFrame(rows).to_string(index=False))

m_lever = BayesianMMM(anatomy, features["price + promo levers"], TREND)
print("\nlever contract: Price/Promo leave the linear control block ->",
      "remaining controls:", m_lever.control_names)
m_sel = BayesianMMM(anatomy, features["control selection"], TREND)
print("selection contract: the CONFOUNDER (Price) is exempt from shrinkage ->",
      sorted(v.name for v in m_sel.model.free_RVs if "confounder" in v.name))
"""

BACKENDS = r"""
for one_liner in ("bayesian_pymc", "bayesian_numpyro", "bayesian_nutpie"):
    cfg = getattr(ModelConfigBuilder(), one_liner)().build()
    print(f".{one_liner}():  inference_method = {cfg.inference_method.value}")
print("\nAll three sample the IDENTICAL graph - only the NUTS implementation "
      "differs, so they target the same posterior. Which is fastest is "
      "model-dependent (the ranking inverts between a national model and a geo "
      "panel); nbs/demos/nuts_backends.ipynb benchmarks them properly.")
"""

FITS = r"""
def timed_fit(config=None, **fit_kw):
    m = BayesianMMM(panel, config or ModelConfig(), TREND)
    t0 = time.time()
    res = m.fit(random_seed=42, **fit_kw)
    return m, res, time.time() - t0

m_nuts, r_nuts, s_nuts = timed_fit(ModelConfig(n_chains=2, n_draws=400, n_tune=400))
m_map, r_map, s_map = timed_fit(method="map")
m_advi, r_advi, s_advi = timed_fit(method="advi")

print(f"NUTS-lite (numpyro, 2x400): {s_nuts:5.1f}s  approximate={r_nuts.approximate}")
print(f"MAP point estimate        : {s_map:5.1f}s  approximate={r_map.approximate}")
print(f"ADVI (mean-field VI)      : {s_advi:5.1f}s  approximate={r_advi.approximate}")
"""

APPROX_FIG = r"""
def beta_stats(m):
    post = m._trace.posterior
    return ({c: float(post[f"beta_{c}"].mean()) for c in CH},
            {c: float(post[f"beta_{c}"].std()) for c in CH})

mu_n, sd_n = beta_stats(m_nuts)
mu_a, sd_a = beta_stats(m_advi)
mu_m, _ = beta_stats(m_map)

fig = go.Figure()
fig.add_trace(go.Bar(x=CH, y=[sd_n[c] for c in CH], name="NUTS (calibrated)",
                     marker_color=PALETTE["TV"]))
fig.add_trace(go.Bar(x=CH, y=[sd_a[c] for c in CH], name="ADVI (approximate)",
                     marker_color=BAD))
for i, c in enumerate(CH):
    fig.add_annotation(x=i, y=max(sd_n[c], sd_a[c]) * 1.06,
                       text=f"x{sd_a[c] / sd_n[c]:.2f}", showarrow=False,
                       font=dict(color=INK, size=11))
fig.update_yaxes(title="posterior sd of beta")
style(fig, 380, "Approximate fits misstate the uncertainty - mean-field ADVI here "
      "shrinks or distorts every interval", barmode="group")
fig.show()

print("point estimates broadly agree (that is what approximate fits are FOR - "
      "fast model checking):")
print(pd.DataFrame({"NUTS mean": mu_n, "ADVI mean": mu_a, "MAP point": mu_m}).round(3))
print("\nCaveat, verbatim from the API: approximate methods' uncertainty is NOT "
      "calibrated and R-hat/ESS do not apply - never use them for final inference.")
"""

RIDGE = r"""
from mmm_framework.config import InferenceMethod

m_ridge = BayesianMMM(
    panel,
    ModelConfig(inference_method=InferenceMethod.FREQUENTIST_RIDGE,
                bootstrap_samples=60, optim_maxiter=48),
    TREND,
)
t0 = time.time()
r_ridge = m_ridge.fit(random_seed=42)
s_ridge = time.time() - t0
print(f"ridge (transform search + closed-form solve + moving-block bootstrap): {s_ridge:.1f}s")
print("MMMResults.converged for a frequentist fit:", r_ridge.converged,
      "(None - MCMC convergence does not apply; intervals are CIs, not credible)")

def contribs(m):
    cc = np.asarray(m._trace.posterior["channel_contributions"].values)
    flat = cc.reshape(-1, m.n_obs, len(CH)).sum(axis=1) * m.y_std
    return flat.mean(axis=0), np.percentile(flat, 5, axis=0), np.percentile(flat, 95, axis=0)

mean_n, lo_n, hi_n = contribs(m_nuts)
mean_r, lo_r, hi_r = contribs(m_ridge)
truth = np.asarray([sc.true_contribution[c] for c in CH])

fig = go.Figure()
fig.add_trace(go.Bar(x=CH, y=truth, name="planted truth", marker_color=TRUTH, opacity=0.85))
fig.add_trace(go.Bar(x=CH, y=mean_n, name="Bayesian NUTS (90% credible)",
                     marker_color=PALETTE["TV"],
                     error_y=dict(array=hi_n - mean_n, arrayminus=mean_n - lo_n)))
fig.add_trace(go.Bar(x=CH, y=mean_r, name="frequentist ridge (90% bootstrap CI)",
                     marker_color=GOLD,
                     error_y=dict(array=hi_r - mean_r, arrayminus=mean_r - lo_r)))
fig.update_yaxes(title="total incremental KPI attributed")
style(fig, 400, "On a clean world both paths recover the truth - the paths diverge "
      "under confounding, not here", barmode="group")
fig.show()

err = pd.DataFrame({
    "NUTS rel err": np.abs(mean_n - truth) / truth,
    "ridge rel err": np.abs(mean_r - truth) / truth}, index=CH)
print((err * 100).round(1).astype(str) + "%")
"""

CELLS = [
    md(r"""
# One model, every knob

`BayesianMMM` is a single PyMC model with a large, explicit configuration
surface: what shape the carryover takes, how returns diminish, what the
baseline does over time, where the default priors live, what the observation
noise looks like, and which opt-in structure (price levers, events,
reach x frequency, synergies, control selection) enters the graph. This
notebook walks every knob on **one small synthetic world** with a sealed
causal answer key, building many models but fitting only a handful.

Two design rules run through everything here:

- **Off by default.** Every opt-in feature leaves the graph byte-identical
  when unused, and announces itself with *named random variables* when on -
  you can always ask the graph what it contains.
- **Config is validated, not stringly.** Everything below is a Pydantic
  object built through a fluent builder; a typo fails at build time, not
  after an hour of sampling.
"""),
    code(SETUP),
    md(r"""
## 1 - The world

`synth.dgp.make_clean` draws data from the model's *exact* generative family
(geometric adstock, concave saturation, additive, Gaussian) and seals the
causal truth - each channel's true contribution is the counterfactual zero-out
on the noiseless mean, the same estimand the model reports. That makes it the
right world for an anatomy tour: any recovery gap is the fit's fault, never
the world's.
"""),
    code(WORLD),
    md(r"""
## 2 - The fluent builder

`ModelConfigBuilder` composes a validated `ModelConfig`. Each method is a
sentence fragment: pick a specification, pick an inference paradigm, set the
sampler budget, choose where the default media prior lives. `.build()` runs
the Pydantic validation, so illegal combinations (a frequentist paradigm with
a Bayesian fit method, a target-accept outside (0,1)) fail right here.
"""),
    code(BUILDER),
    md(r"""
## 3 - Adstock: when does the effect land?

Media effects persist past the week the money is spent. The adstock kernel is
a lag-indexed weight vector, and the family choice is a *claim about shape*:
**geometric** says effect peaks immediately and decays; **delayed** lets the
peak move later (a considered purchase); **weibull** controls the whole
profile with a shape parameter. `transforms.adstock_weights` builds exactly
the kernels the graph uses, so we can chart them directly.
"""),
    code(ADSTOCK_KERNELS),
    code(ADSTOCK_PULSE),
    md(r"""
## 4 - Saturation: diminishing returns, five ways

A dollar into a saturated channel buys less than the first dollar did. The
framework ships five saturation families (`SaturationType`), and they are
genuinely different shapes: logistic and Michaelis-Menten are strictly
concave, root is concave with an unbounded marginal at zero, tanh saturates
hard, and only **Hill with slope > 1** produces a true S-curve (a convex
launch region before the concave tail). The formulas live in one place in the
graph - `model.base._apply_saturation_pt` - which both the likelihood and
every counterfactual estimand route through, so we evaluate that exact
function below rather than re-typing the math.
"""),
    code(SATURATION),
    md(r"""
## 5 - Trend and seasonality: what the baseline is allowed to do

Whatever the baseline cannot express, the media terms will absorb - so trend
flexibility is an attribution decision, not an aesthetic one. `TrendType`
offers none / linear / piecewise / spline / gaussian_process in increasing
flexibility (the GP is a Hilbert-space approximation configured via
`TrendConfigBuilder`). Seasonality is Fourier: sine/cosine pairs at harmonics
of the period, configured per component (`SeasonalityConfigBuilder`
`.with_yearly(order) / .with_monthly / .with_weekly`).
"""),
    code(TREND_FIG),
    code(SEASONALITY),
    md(r"""
## 6 - Where the default media prior lives: ROI mode vs coefficient mode

The historical default puts a Gamma prior on each channel's *standardized
coefficient* - a scale no planner has intuition about. `media_prior_mode="roi"`
instead samples each channel's **prior ROI directly**
(`roi_<ch> ~ LogNormal(0, 1)`, median exactly break-even) and derives the
coefficient in-graph so that on the observed media, contribution divided by
spend *equals* that ROI. The prior now lives on the decision scale: "before
seeing data, I believe a dollar of TV returns about a dollar, could be a
quarter, could be four" is a sentence a stakeholder can veto. Channels with an
experiment-calibrated `roi_prior` or explicit `coefficient_prior` are
unaffected - the mode only sets the *default*.
"""),
    code(ROI_PRIORS),
    md(r"""
## 7 - The observation model: likelihood families and the multiplicative form

The likelihood is a claim about the noise. `normal` is the default;
`student_t` keeps the same mean structure but stops a single outlier week
from dragging every coefficient (the tails below are the whole argument).
Count and share KPIs get `poisson` / `negative_binomial` / `binomial` /
`beta` families with canonical links, validated at config time.

Orthogonally, `.multiplicative()` switches the *specification*: the KPI is
modeled on the log scale and each channel contributes a multiplicative lift
running from 1x at zero spend to `exp(beta)` at full saturation - percentage
thinking instead of additive units, with the same saturation curves.
"""),
    code(LIKELIHOOD),
    md(r"""
## 8 - Opt-in structure, and the reachability story

Every structural lever follows the same contract: **off by default** (the
graph is byte-identical without it) and **on means named RVs appear** - the
graph itself testifies that your config reached it. This is not cosmetic: a
prior or lever that silently fails to reach the graph is the worst failure
mode a config system can have, so the test suite pins these names, and you
can run the same check in a notebook. We extend the world's panel with lever
columns (promo flags, an ad-frequency series, two noise controls) and build -
never fit - each variant:

- **price / promo levers** - a control column promoted to a first-class lever:
  a sign-guarded log-price elasticity, a promo lift with its own carryover;
- **events** - windowed, decaying regressors for sharp dates (a launch,
  Black Friday) the smooth Fourier terms cannot represent;
- **reach x frequency** - a channel's effect becomes `reach * g(frequency)`
  with a frequency-wearout curve;
- **channel interactions** - an explicit `beta_ij * sat_i * sat_j` synergy /
  cannibalization term;
- **control selection** - horseshoe / spike-slab / LASSO shrinkage over
  *selectable* controls, with declared confounders exempt (a confounder is in
  the model to close a back-door, so shrinking it away would be self-defeating).
"""),
    code(ANATOMY_PANEL),
    code(REACHABILITY),
    md(r"""
## 9 - Inference: three NUTS backends, fast approximate checks

The Bayesian paradigm has three interchangeable NUTS engines - a one-line
choice with zero effect on the posterior being targeted. Below them sit the
*approximate* fit methods (`map`, `laplace`, `advi`, `fullrank_advi`,
`pathfinder`), which answer "is this model sane?" in seconds. The trade is
explicit and measured next: point estimates land close, **intervals do not**.
"""),
    code(BACKENDS),
    code(FITS),
    code(APPROX_FIG),
    md(r"""
## 10 - The frequentist path

`.frequentist_ridge()` is a **different paradigm**, not a faster Bayesian:
adstock/saturation are chosen by rolling-origin out-of-sample search,
coefficients solve in closed form, and intervals come from a moving-block
residual bootstrap. (`frequentist_cvxpy` adds *hard* constraints - the one
thing a prior cannot express.) On this clean world both paths recover the
truth; the v1.3 evaluation (epic #180) found the divergence lives on
*confounded* worlds, where ridge over-credited media by +41.6% against MAP's
+5.9% on the same data. The verdict shipped with the feature: ridge's value
is **constraints and speed, not accuracy** - use the Bayesian path for
inference you plan to act on. See `technical-docs/frequentist-estimation.md`.
"""),
    code(RIDGE),
    md(r"""
## 11 - The quiet machinery: `utils/`

Two modules do unglamorous, load-bearing work under everything above.
`utils/standardization` owns the scaling story: the KPI and controls are
standardized before they meet the graph, media is max-normalized per channel,
and every reported contribution is mapped back to original units through one
set of `ScalingParameters` - which is why the decompositions in section 10
are in KPI units, and why "never center a media-dependent signal in-graph" is
a repo-level blocker rule. `utils/arviz_compat` shields the codebase from
ArviZ/PyMC container drift (ArviZ 1.x traces are xarray DataTrees): all
posterior access routes through it, so a version bump breaks one shim, not
forty call sites.

## Where to go deeper

**Specs (technical-docs/):**
[`frequentist-estimation.md`](../../technical-docs/frequentist-estimation.md) -
the full ridge-vs-Bayes evaluation behind the v1.3 verdict -
[`custom-model-config.md`](../../technical-docs/custom-model-config.md) -
per-model config + pluggable likelihood -
[`extension-model-priors.md`](../../technical-docs/extension-model-priors.md) -
priors on the extension families -
[`sampling-failure-playbook.md`](../../technical-docs/sampling-failure-playbook.md) -
the escalation ladder when a fit diverges -
[`coverage-diagnostics.md`](../../technical-docs/coverage-diagnostics.md) -
what "approximate intervals are not calibrated" costs, measured -
plus *ROI-based default media priors* and *Agent-set priors don't change the
model* in [`engineering-notes.md`](../../technical-docs/engineering-notes.md).

**Sibling notebook series:**
`math_01`/`math_02`/`math_03` derive the adstock, saturation and
seasonality/trend mathematics with property checks - `workshop_00..05` is the
beginner Bayesian-MMM course - `causal 00-10` covers why these knobs are
causal claims - `stress_00..06` pressure-tests the model on worlds that
violate its assumptions - `demos/nuts_backends.ipynb` benchmarks the three
NUTS engines properly - `lifecycle_00..06` runs the fitted model through the
T0-T5 measurement loop.
"""),
]


def main() -> None:
    nb = new_notebook(cells=CELLS)
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    out = "showcase/showcase_02_model_anatomy.ipynb"
    with open(out, "w") as fh:
        nbformat.write(nb, fh)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
