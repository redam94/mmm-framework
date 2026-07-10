"""Author structural_nested_mmm.ipynb — the StructuralNestedMMM example notebook.

    uv run --with nbformat python build_structural_nested.py
    TQDM_DISABLE=1 uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        structural_nested_mmm.ipynb --ExecutePreprocessor.timeout=3600 \
        --ExecutePreprocessor.kernel_name=python3

A single-brand walkthrough of the multi-mediator structural MMM
(`mmm_extensions/models/structural.py`, spec technical-docs/structural-nested-mmm.md):
build the `make_brand_funnel` ground-truth world (binomial awareness tracker ->
Likert consideration -> sales, with a latent demand confounder), configure the
funnel with the factory helpers, fit with NUTS (the settings the slow recovery
test pinned down), and grade the recovered structure against the sealed answer
key — latent states, structural parameters, mediation decomposition, ROAS, and
the exact-counterfactual machinery.
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
os.environ.setdefault("TQDM_DISABLE", "1")   # quiet sampling progress bars

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
# pymc's advisory chatter lives on CHILD loggers that set their own level at
# import — silence them directly, not just the parent.
for _n in ("pymc", "pymc.sampling", "pymc.sampling.mcmc", "pymc.stats",
           "pymc.stats.convergence", "numpyro", "jax", "jax._src.xla_bridge",
           "arviz"):
    logging.getLogger(_n).setLevel(logging.ERROR)

# --- palette (validated categorical slots, assigned in fixed order) ---------
CH = {"TV": "#2a78d6", "Display": "#1baf7a", "Social": "#eda100", "Search": "#008300"}
MED = {"awareness": "#4a3aa7", "consideration": "#e34948", "demand": "#e87ba4"}
LIKERT = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # ordinal blue ramp
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e1e0d9", "#898781"

def style(fig, title, height=420, legend=None, **layout):
    # House chart chrome. legend=None -> no legend expected; 'below' /
    # 'below-xtitle' place a horizontal legend under the plot area (the extra
    # bottom margin clears the date ticks, and the x-axis title too).
    b = {"below": 85, "below-xtitle": 110}.get(legend, 50)
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK)),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif',
                  size=12, color=INK2),
        height=height, margin=dict(l=60, r=30, t=60, b=b),
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb", **layout)
    if legend:
        offset = 30 if legend == "below" else 58   # px below the plot bottom
        fig.update_layout(legend=dict(
            orientation="h", x=0, xanchor="left", yanchor="top",
            y=-offset / (height - 60 - b)))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor="#c3c2b7")
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor="#c3c2b7")
    return fig

def band(hex_color, alpha=0.18):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

import pymc, mmm_framework
print(f"pymc {pymc.__version__}  |  mmm_framework {mmm_framework.__version__}")
"""


CELLS = [
    md(r"""
# The brand funnel — `StructuralNestedMMM`

**A structural, multi-mediator MMM: surveys, latent states, and a mediator DAG.**

A classic MMM regresses sales on transformed media. A *nested* (mediation) MMM adds one
middle layer — media moves a mediator, the mediator moves sales. But real brand funnels
are messier than one Gaussian mediator:

- **Awareness** is measured by a *binary tracker* ("have you seen this brand?") asked of
  `n_t` respondents per week — and `n_t` varies, so weekly sampling noise is
  `p(1-p)/n_t`, not a constant. Population awareness also *persists*: most people aware
  last week are aware this week. That is an **AR(1) latent state** with a **binomial
  measurement**, not a Gaussian regression.
- **Consideration** is a *5-point Likert item* (ordered category counts), driven by media
  **and by awareness** (a mediator → mediator edge) **and by price** (a control inside a
  mediator equation).
- A **latent demand trend** moves consideration *and* sales — a common cause that, left
  out, confounds the mediated path.

`StructuralNestedMMM` expresses all of this: a **structural equation system over a DAG of
latent mediator states**, each with its own dynamics (static / AR(1) / random-walk) and
measurement family (Gaussian / binomial / ordered / fully latent), plus shared latent
factors. This notebook fits the framework's ground-truth **brand-funnel world** and grades
the recovery against the sealed answer key:

```text
    TV ──────────────▶ awareness            AR(1) logit state (ρ = 0.85),
                          │                 weekly binomial tracker, varying n_t
    Display ──┐           ▼
    Social  ──┼──────▶ consideration        static latent index, 5-point Likert
    price   ──┤           │                 (Multinomial category counts)
    demand  ──┘           ▼
    demand, price ────▶  sales   ◀───────── Search (pure direct response)
```

TV, Display and Social are **fully mediated** in truth (zero direct effect); Search is
pure direct response. The demand trend is never observed. Spec:
`technical-docs/structural-nested-mmm.md`.
"""),
    code(SETUP),
    md(r"""
## 1 · A world with a sealed answer key

`make_brand_funnel` generates the funnel above with **known structural parameters** —
156 weeks of media spend, sales, the two mediator surveys, and every latent series. The
scenario's `notes` carry the answer key (true persistence, true edge coefficients, the
realized demand trend); `true_roas` is the **counterfactual** ground truth, computed with
the same structure-preserving intervention the model itself uses. We look at the data
first and keep the answer key sealed until the model has committed to its estimates.

Two things the survey blocks make concrete:

- **Awareness** arrives as `(counts, trials)` pairs on ~75% of weeks — `NaN` weeks simply
  carry no measurement likelihood (no imputation, no interpolation).
- **Consideration** arrives as a `(weeks × 5)` matrix of Likert category counts on ~85%
  of weeks.
"""),
    code(r"""
from mmm_framework.synth.dgp import make_brand_funnel

sc = make_brand_funnel(seed=21, n_weeks=156)
weeks, channels = sc.weeks, sc.channels

aw_counts = sc.notes["awareness_counts"]        # (156,) float, NaN = no survey
aw_trials = sc.notes["awareness_trials"]        # (156,) respondents per week
cons_counts = sc.notes["consideration_counts"]  # (156, 5) Likert category counts

aw_obs = np.isfinite(aw_counts)
cons_obs = np.isfinite(cons_counts).all(axis=1)
print(f"channels          : {', '.join(channels)}")
print(f"window            : {weeks[0].date()} -> {weeks[-1].date()}  ({len(weeks)} weeks)")
print(f"awareness tracker : {aw_obs.sum()}/{len(weeks)} weeks observed, "
      f"n_t in [{np.nanmin(aw_trials):.0f}, {np.nanmax(aw_trials):.0f}]")
print(f"consideration     : {cons_obs.sum()}/{len(weeks)} weeks observed (5-point Likert)")
print(f"controls          : {list(sc.controls.columns)}")
""".strip("\n")),
    code(r"""
# Media spend and the KPI — two panels, one time axis (never a dual axis).
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    subplot_titles=("weekly media spend (KPI $000s scale)", "weekly sales"))
for c in channels:
    fig.add_trace(go.Scatter(x=weeks, y=sc.spend[c], name=c, mode="lines",
                             line=dict(color=CH[c], width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=weeks, y=sc.y, name="sales", mode="lines",
                         line=dict(color=INK, width=2), showlegend=False), row=2, col=1)
style(fig, "The brand-funnel world — pulsed flighting per channel, three years weekly",
      height=520, legend="below")
fig.show()
""".strip("\n")),
    md(r"""
### The mediator surveys

The awareness tracker's error bars below are the *binomial* 95% intervals
`± 1.96·√(p̂(1-p̂)/n_t)` — precision varies week to week with the survey volume, which is
exactly the information the binomial measurement model will use (a 600-respondent week
counts for more than a 150-respondent week). The Likert panel shows the weekly *share* of
respondents in each consideration category — an ordered scale, so it gets one hue,
light → dark.
"""),
    code(r"""
p_hat = aw_counts / aw_trials
se = np.sqrt(p_hat * (1 - p_hat) / aw_trials)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                    subplot_titles=("awareness tracker — weekly p̂ with binomial 95% CI",
                                    "consideration Likert — weekly category shares"))
fig.add_trace(go.Scatter(
    x=weeks[aw_obs], y=p_hat[aw_obs], mode="markers", name="tracker p̂",
    marker=dict(color=MED["awareness"], size=5),
    error_y=dict(type="data", array=1.96 * se[aw_obs], color=band(MED["awareness"], 0.55),
                 thickness=1.2, width=0), showlegend=False), row=1, col=1)

shares = cons_counts / np.nansum(cons_counts, axis=1, keepdims=True)
for k in range(5):
    fig.add_trace(go.Bar(x=weeks[cons_obs], y=shares[cons_obs, k],
                         name=f"category {k + 1}", marker_color=LIKERT[k],
                         marker_line_width=0), row=2, col=1)
fig.update_layout(barmode="stack", bargap=0)
style(fig, "What the model gets to see — two surveys, neither Gaussian, neither complete",
      height=560, legend="below")
fig.update_yaxes(title_text="share aware", row=1, col=1)
fig.update_yaxes(title_text="share of responses", row=2, col=1)
fig.show()
""".strip("\n")),
    md(r"""
## 2 · Configuring the funnel

Three factory helpers cover the common cases; each returns a frozen spec you could also
build by hand (`MediatorSpec` / `LatentFactorSpec`):

- **`binary_survey_mediator`** — AR(1) latent state on the logit scale + binomial
  measurement with per-week trials. `persistence="high"` sets a Beta(9, 1.5) prior on ρ.
  **Adstock is OFF** for this equation: the AR(1) state *is* the carryover, and stacking
  geometric adstock on top of a geometric state decay would create an α↔ρ ridge (two
  nearly interchangeable memories — the model warns if you force both on).
- **`likert_mediator`** — static latent index + cumulative-logit Multinomial over the
  category counts. Location lives in the *cutpoints* (an ordered mediator has no free
  intercept — the two would be exactly confounded). Awareness enters via `parents`,
  price via `controls`, the demand trend via `latent_factors`.
- **`latent_demand_factor`** — a smooth AR(1) factor, unit-standardized in-graph
  (media-independent, so this is counterfactual-safe), shared by consideration **and**
  the outcome. Its **sign anchor** resolves to the first *measured* mediator consumer
  (consideration): a HalfNormal loading only pins a factor's sign if the data holds that
  loading materially nonzero — anchoring at a small outcome loading lets the reflected
  mode `(−F, −w)` escape through the anchor's cost-free zero, which showed up as
  split-chain R-hat ≈ 1.75 during this model's design review.

Two more choices mirror the mediation-recovery lessons — one explicit, one a default.
`affects_outcome=False` for awareness must be **set explicitly** (the DAG above has no
awareness → sales edge — its influence flows *through* consideration — but the Python API
defaults every mediator's outcome path ON). The **tight direct-effect priors**
(`Normal(0, 0.3)`) for mediated channels come for free from the factories, because an
over-wide direct path steals the mediated signal.
"""),
    code(r"""
from mmm_framework.mmm_extensions import (
    StructuralNestedConfig,
    StructuralNestedMMM,
    binary_survey_mediator,
    latent_demand_factor,
    likert_mediator,
)

config = StructuralNestedConfig(
    mediators=(
        binary_survey_mediator(
            "awareness", ["TV"],
            persistence="high",        # Beta(9, 1.5) prior on rho
            affects_outcome=False,     # feeds consideration, not sales directly
        ),
        likert_mediator(
            "consideration", ["Display", "Social"],
            parents=["awareness"],     # mediator -> mediator edge
            controls=["Price"],        # a control INSIDE a mediator equation
            latent_factors=["demand"],
            n_categories=5,
        ),
    ),
    latent_factors=(latent_demand_factor("demand"),),
)

print("topological order :", config.topological_order())
for m in config.mediators:
    print(f"  {m.name:<14} dynamics={m.dynamics.value:<7} "
          f"likelihood={m.measurement.likelihood.value:<9} "
          f"adstock={'on' if m.adstock_enabled else 'off (state carries memory)'}")
""".strip("\n")),
    md(r"""
The constructor takes the survey blocks directly — `mediator_data` for the observations
(`NaN` = no survey that week), `mediator_trials` for the binomial weekly sample sizes.
Search appears in no mediator spec, so it automatically gets a plain full-strength direct
`beta_Search`; TV/Display/Social get *tight* `delta_direct_*` paths alongside their
mediated routes.
"""),
    code(r"""
model = StructuralNestedMMM(
    sc.spend.to_numpy(float),
    sc.y.to_numpy(float),
    channels,
    config,
    mediator_data={"awareness": aw_counts, "consideration": cons_counts},
    mediator_trials={"awareness": aw_trials},
    X_controls=sc.controls.to_numpy(float),
    control_names=["Price"],
    index=weeks,
)

graph = model.model   # the PyMC graph builds lazily on first access
rvs = sorted(v.name for v in graph.free_RVs)
print(f"{len(rvs)} free RVs. The structural ones:")
for name in rvs:
    if not name.startswith(("alpha_", "lambda_TV", "lambda_Display",
                            "lambda_Social", "lambda_Search")):
        print("  ", name)
""".strip("\n")),
    md(r"""
Reading that list against the DAG: each funnel edge is one named parameter
(`beta_TV_to_awareness`, `lambda_awareness_to_consideration`,
`phi_Price_to_consideration`, `w_demand_to_consideration`, …), each latent state has its
own dynamics RVs (`awareness_persistence`, per-week state noise), the Likert measurement
carries its generatively-ordered cutpoints, and the outcome equation has
`gamma_consideration`, the tight `delta_direct_*` terms, and `beta_Search`. Note what is
*absent*: no `gamma_awareness` (it doesn't feed sales directly) and no `level_consideration`
(the cutpoints own location).

### Prior-predictive sanity

One check worth showing because it guards a real trap: the Likert cutpoints are built
**generatively ordered** (anchor + cumulative positive gaps), *not* with a `transform=ordered`
constraint — transforms only reshape logp space, so forward prior draws would come out
unordered and the Multinomial would silently launder negative cell probabilities. Ordered
cutpoints in the *prior predictive* prove the construction is generative.
"""),
    code(r"""
prior = model.sample_prior_predictive(samples=300, random_seed=0)

cp = prior.prior["consideration_cutpoints"].values.reshape(-1, 4)
assert np.all(np.diff(cp, axis=-1) >= 0), "cutpoints must be ordered in FORWARD draws"
print(f"cutpoints ordered in all {cp.shape[0]} prior draws ✓")

p_prior = prior.prior["awareness_probability"].values.reshape(-1, len(weeks))
lo, mid, hi = np.quantile(p_prior, [0.05, 0.5, 0.95], axis=0)
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.r_[weeks, weeks[::-1]], y=np.r_[hi, lo[::-1]],
                         fill="toself", fillcolor=band(MED["awareness"]),
                         line=dict(width=0), name="prior 90% band", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=weeks, y=mid, mode="lines", name="prior median",
                         line=dict(color=MED["awareness"], width=2, dash="dot")))
fig.add_trace(go.Scatter(x=weeks[aw_obs], y=p_hat[aw_obs], mode="markers",
                         name="tracker p̂", marker=dict(color=INK, size=5)))
style(fig, "Prior predictive — awareness probability covers the tracker without hugging it",
      legend="below")
fig.update_yaxes(title_text="P(aware)", range=[0, 1])
fig.show()
""".strip("\n")),
    md(r"""
## 3 · Fit

NUTS, four chains — the same settings the framework's slow recovery test pinned down.
Latent-state graphs carry a per-week innovation vector for every dynamic state, so
approximate methods (MAP/ADVI/Pathfinder) are smoke-checks only here; the model warns if
you try to lean on them. One geometry detail happens automatically: the densely-observed
awareness tracker (≥50% of weeks) gets the **centered** AR-noise parameterization — a
strong measurement pins the state path, and the non-centered form would funnel against
its scale — while sparse mediators keep the non-centered form.
""".strip("\n")),
    code(r"""
# pymc re-levels its child loggers at import, and (since pymc 6) logs the
# low-ESS advisory at ERROR level — silence needs CRITICAL, right before the fit.
for _n in ("pymc", "pymc.sampling", "pymc.sampling.mcmc",
           "pymc.stats", "pymc.stats.convergence"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

res = model.fit(
    draws=500, tune=1000, chains=4,
    target_accept=0.95, random_seed=11,
    nuts_sampler="numpyro", progressbar=False,
)

d = res.diagnostics
print(f"fit method : {d.get('fit_method')}   approximate: {d.get('approximate')}")
print(f"R-hat max  : {d.get('rhat_max'):.3f}")
if d.get("ess_bulk_min") is not None:
    print(f"ESS (bulk) : {d.get('ess_bulk_min'):.0f} minimum")
if d.get("divergences") is not None:
    print(f"divergences: {d.get('divergences')}")
assert d.get("rhat_max") is not None and d["rhat_max"] < 1.1, d
""".strip("\n")),
    md(r"""
## 4 · Did it find the funnel? Unsealing the answer key

The model never saw the true awareness probability, the true consideration index, or the
demand trend. Compare its posterior latent series against all three. The demand panel is
the strongest claim: an **unobserved** common cause, reconstructed purely from the
co-movement it induces between the consideration survey and sales.
"""),
    code(r"""
post = res.trace.posterior

def series_band(name):
    v = post[name].values.reshape(-1, len(weeks))
    lo, mid, hi = np.quantile(v, [0.05, 0.5, 0.95], axis=0)
    return lo, mid, hi, v.mean(axis=0)

def zs(x):
    return (x - x.mean()) / x.std()

lo_p, mid_p, hi_p, mean_p = series_band("awareness_probability")
mean_z = post["consideration_latent"].mean(("chain", "draw")).values
mean_f = post["factor_demand"].mean(("chain", "draw")).values

corr_p = np.corrcoef(mean_p, sc.notes["p_awareness"])[0, 1]
corr_z = np.corrcoef(mean_z, sc.notes["z_consideration"])[0, 1]
corr_f = np.corrcoef(mean_f, sc.notes["latent_demand"])[0, 1]

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                    subplot_titles=(f"awareness — posterior vs truth (corr {corr_p:.2f})",
                                    f"consideration index, z-scored (corr {corr_z:.2f})",
                                    f"latent demand — never observed (corr {corr_f:.2f})"))

fig.add_trace(go.Scatter(x=np.r_[weeks, weeks[::-1]], y=np.r_[hi_p, lo_p[::-1]],
                         fill="toself", fillcolor=band(MED["awareness"]),
                         line=dict(width=0), name="posterior 90% band",
                         hoverinfo="skip"), row=1, col=1)
fig.add_trace(go.Scatter(x=weeks, y=mean_p, mode="lines", name="posterior mean",
                         line=dict(color=MED["awareness"], width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=weeks, y=sc.notes["p_awareness"], mode="lines", name="truth",
                         line=dict(color=INK, width=2, dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=weeks[aw_obs], y=p_hat[aw_obs], mode="markers",
                         name="tracker p̂", marker=dict(color=MUTED, size=4)), row=1, col=1)

fig.add_trace(go.Scatter(x=weeks, y=zs(mean_z), mode="lines", showlegend=False,
                         line=dict(color=MED["consideration"], width=2)), row=2, col=1)
fig.add_trace(go.Scatter(x=weeks, y=zs(sc.notes["z_consideration"]), mode="lines",
                         showlegend=False, line=dict(color=INK, width=2, dash="dash")),
              row=2, col=1)

fig.add_trace(go.Scatter(x=weeks, y=zs(mean_f), mode="lines", showlegend=False,
                         line=dict(color=MED["demand"], width=2)), row=3, col=1)
fig.add_trace(go.Scatter(x=weeks, y=sc.notes["latent_demand"], mode="lines",
                         showlegend=False, line=dict(color=INK, width=2, dash="dash")),
              row=3, col=1)

style(fig, "Latent-state recovery — colored: posterior, dashed ink: sealed truth",
      height=720, legend="below")
fig.update_yaxes(title_text="P(aware)", row=1, col=1)
fig.show()

assert corr_p > 0.7 and corr_z > 0.7 and abs(corr_f) > 0.6
""".strip("\n")),
    md(r"""
### Structural parameters

Every funnel edge is directly comparable to the answer key because each measured mediator
**pins its own scale** — the binomial link fixes awareness on the absolute probability
scale, and the Likert cutpoint geometry anchors the consideration index against
unit-logistic response noise. That scale-pinning is the load-bearing identification idea
in this model family: media → mediator paths are identified by the *mediator data*, not
scavenged from the outcome residual.
"""),
    code(r"""
true_params = sc.notes["true_params"]
PARAMS = [
    ("awareness_persistence",            "ρ  awareness persistence",   true_params["rho_awareness"]),
    ("beta_TV_to_awareness",             "β  TV → awareness",          true_params["b_tv_to_awareness"]),
    ("beta_Display_to_consideration",    "β  Display → consideration", true_params["b_display_to_consideration"]),
    ("beta_Social_to_consideration",     "β  Social → consideration",  true_params["b_social_to_consideration"]),
    ("lambda_awareness_to_consideration","λ  awareness → consideration", true_params["lambda_awareness_to_consideration"]),
    ("phi_Price_to_consideration",       "φ  price → consideration",   true_params["phi_price_to_consideration"]),
    ("w_demand_to_consideration",        "w  demand → consideration",  true_params["w_demand_to_consideration"]),
]

rows, fig = [], go.Figure()
for var, label, truth in PARAMS:
    draws = post[var].values.ravel()
    lo, mean, hi = np.quantile(draws, 0.05), draws.mean(), np.quantile(draws, 0.95)
    rows.append({"parameter": label, "posterior_mean": mean,
                 "q5": lo, "q95": hi, "truth": truth})
    fig.add_trace(go.Scatter(x=[lo, hi], y=[label, label], mode="lines",
                             line=dict(color=band(MED["awareness"], 0.9), width=8),
                             showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=[r["posterior_mean"] for r in rows],
                         y=[r["parameter"] for r in rows], mode="markers",
                         name="posterior mean (90% CI)",
                         marker=dict(color=MED["awareness"], size=11,
                                     line=dict(color="white", width=1.5))))
fig.add_trace(go.Scatter(x=[r["truth"] for r in rows], y=[r["parameter"] for r in rows],
                         mode="markers", name="sealed truth",
                         marker=dict(color=INK, symbol="diamond", size=11,
                                     line=dict(color="white", width=1.5))))
fig.add_vline(x=0, line=dict(color=MUTED, width=1, dash="dot"))
style(fig, "Structural edges — posterior vs the answer key",
      height=430, legend="below")
fig.update_yaxes(autorange="reversed")
fig.show()

tbl = pd.DataFrame(rows).set_index("parameter")
print(tbl.round(3).to_string())

# gamma is on the standardized-y scale in-graph; times y_std it is KPI units
# per unit of consideration index — directly comparable to the DGP's 2500.
gamma_kpi = float(post["gamma_consideration"].values.mean()) * model.y_std
print(f"\nγ  consideration → sales : {gamma_kpi:,.0f} KPI units per index point "
      f"(truth {true_params['gamma_consideration']:,.0f})")

assert float(post["awareness_persistence"].mean()) > 0.6
assert float(post["beta_TV_to_awareness"].mean()) > 0.4
assert float(post["lambda_awareness_to_consideration"].mean()) > 0.8
assert float(post["phi_Price_to_consideration"].mean()) < -0.3   # unconstrained prior; data-driven sign
assert float(post["w_demand_to_consideration"].mean()) > 0.3
""".strip("\n")),
    md(r"""
The price edge is worth pausing on: its prior is a symmetric `Normal(0, 1)` — nothing
pushed it negative except the Likert data inside the mediator equation. And ρ lands near
the true 0.85, which is what licenses the "awareness carries over" story quantitatively:
a sustained driver is amplified by the steady-state gain `1/(1-ρ)` through the state.

## 5 · Mediation decomposition — exact counterfactuals, not coefficient products

With sigmoid links and AR dynamics in the path, a channel's total effect is **not** a
product of coefficients. `get_mediation_effects()` computes it the only exact way: swap
the channel's spend to zero with `pm.set_data`, recompute `mu` under every posterior draw
with the latent innovations **held fixed** (same demand shocks, same survey noise —
different media), and difference. The design review that hardened this model found the
classic failure this avoids: any in-graph *centering* of a media-dependent signal gets
recomputed under the counterfactual swap and makes the summed mediated contrast
identically zero. Downstream equations therefore consume **uncentered natural-scale**
signals, and intercepts absorb the levels.
"""),
    code(r"""
me = model.get_mediation_effects().set_index("channel").loc[channels]

fig = go.Figure()
fig.add_trace(go.Bar(y=channels, x=[me.loc[c, "total_indirect"] for c in channels],
                     name="mediated (via the funnel)", orientation="h",
                     marker=dict(color=MED["awareness"], line_width=0)))
fig.add_trace(go.Bar(y=channels, x=[me.loc[c, "direct_effect"] for c in channels],
                     name="direct", orientation="h",
                     marker=dict(color="#c3c2b7", line_width=0)))
for c in channels:
    pm_share = me.loc[c, "proportion_mediated"]
    # anchor at the END of the positive stack (a negative direct extends left,
    # so total_effect can sit INSIDE the mediated bar)
    bar_end = max(me.loc[c, "direct_effect"], 0.0) + max(me.loc[c, "total_indirect"], 0.0)
    fig.add_annotation(y=c, x=bar_end,
                       text=f" {max(pm_share, 0):.0%} mediated", showarrow=False,
                       xanchor="left", font=dict(size=11, color=INK2))
fig.update_layout(barmode="relative", bargap=0.35)
style(fig, "Total incremental sales per channel — mediated vs direct (truth: brand channels 100% mediated)",
      height=380, legend="below-xtitle")
fig.update_xaxes(title_text="incremental sales over the window (KPI units)")
fig.show()

cols = ["direct_effect", "total_indirect", "total_effect", "proportion_mediated"]
print(me[cols].round(1).to_string())
print("\ntruth — mediated share:", sc.notes["mediated_share"])

for c in ("TV", "Display", "Social"):
    assert me.loc[c, "total_effect"] > 0
    assert me.loc[c, "proportion_mediated"] > 0.25   # truth 1.0; centering bug would give ~0
""".strip("\n")),
    md(r"""
The brand channels come out **dominantly mediated** (truth: fully mediated) and Search
stays direct — the funnel structure, not just the fit, is what got recovered. A
proportion *above* 1 simply means the tight direct path landed slightly **negative** for
that channel (mediated = total − direct, so a below-zero direct pushes the ratio past
100%) — the decomposition is a difference of posterior means, not a forced simplex. The
`proportion_mediated > 0` guard is exactly the design-review regression test: the
in-graph-centering bug would have produced beautiful convergence *and* an identically-zero
mediated share.

`get_pathway_effects()` complements the exact totals with a *linearized* per-path view
(coefficient products × mean sigmoid slopes × AR steady-state gains) — the "how much of
TV flows via awareness → consideration" table. Linearized means approximate; use it for
routing shares, not headline totals.
"""),
    code(r"""
pw = model.get_pathway_effects()
print(pw.round(2).to_string(index=False))
pairs = set(zip(pw["channel"], pw["mediator"]))
# TV's influence terminates at consideration (via awareness); awareness itself has no
# outcome edge, so no TV-via-awareness row exists.
assert ("TV", "consideration") in pairs and ("TV", "awareness") not in pairs
""".strip("\n")),
    md(r"""
## 6 · ROAS against the sealed truth
"""),
    code(r"""
roas = model.get_channel_roas().set_index("channel").loc[channels]

fig = go.Figure()
for c in channels:
    r = roas.loc[c]
    fig.add_trace(go.Scatter(x=[r["roas"] - 1.64 * r["roas_sd"], r["roas"] + 1.64 * r["roas_sd"]],
                             y=[c, c], mode="lines", showlegend=False, hoverinfo="skip",
                             line=dict(color=band(CH[c], 0.45), width=8)))
fig.add_trace(go.Scatter(x=[roas.loc[c, "roas"] for c in channels], y=channels,
                         mode="markers", name="model (90% interval)",
                         marker=dict(color=[CH[c] for c in channels], size=13,
                                     line=dict(color="white", width=1.5))))
fig.add_trace(go.Scatter(x=[sc.true_roas[c] for c in channels], y=channels,
                         mode="markers", name="sealed truth",
                         marker=dict(color=INK, symbol="diamond", size=13,
                                     line=dict(color="white", width=1.5))))
style(fig, "Counterfactual ROAS per channel — model vs truth",
      height=360, legend="below-xtitle")
fig.update_xaxes(title_text="incremental sales per unit of spend")
fig.update_yaxes(autorange="reversed")
fig.show()

print(roas[["contribution", "spend", "roas", "roas_sd"]].round(2).to_string())
print("\ntrue ROAS:")
print(sc.true_roas.round(2).to_string())

assert (roas["roas"] > 0).all()
assert 0.5 < roas.loc["Search", "roas"] / sc.true_roas["Search"] < 2.0
""".strip("\n")),
    md(r"""
Read this honestly: **structure is recovered sharply, magnitudes come with funnel-sized
uncertainty**. Search — one hop, no latent states in the way — lands within a factor the
direct-response literature would envy. The brand channels' point estimates carry wider
bands because their effects thread latent states and survey measurements — TV through an
AR(1) state, a sigmoid link *and* the Likert measurement; Display and Social through the
static consideration index and its Likert measurement — and the posterior *says so* via
`roas_sd` rather than pretending otherwise. With
few survey waves and high ρ, some slow media effect is genuinely absorbed by the state —
which is why the recovery contract asserts signs, shares, persistence and latent
correlations, not point magnitudes.

## 7 · Under the hood: the intervention is structure-preserving

Everything in sections 5–6 rests on one primitive worth seeing raw: swap the media data,
recompute the deterministic graph under the *same* posterior draws, restore the data.
Because `mu` (and every latent signal) is deterministic given the draws, this is an exact
recompute — no new sampling noise. Watch what turning TV off does to the awareness
*state*: the lift decays at the recovered ρ, and the demand wiggles stay identical in
both worlds because the innovations are held fixed.
"""),
    code(r"""
import pymc

X_cf = model.X_media.copy()
X_cf[:, channels.index("TV")] = 0.0
with model.model:
    try:
        pymc.set_data({"X_media": X_cf})
        ppc = pymc.sample_posterior_predictive(
            res.trace, var_names=["awareness_probability", "mu"], progressbar=False)
    finally:
        pymc.set_data({"X_media": model.X_media})

p_cf = ppc.posterior_predictive["awareness_probability"].mean(("chain", "draw")).values
mu_cf = ppc.posterior_predictive["mu"].values
mu_base = res.trace.posterior["mu"].values
tv_effect_draws = (mu_base - mu_cf).sum(axis=-1)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                    subplot_titles=("awareness probability — observed TV vs TV switched off",
                                    "weekly sales lift attributable to TV (posterior mean)"))
fig.add_trace(go.Scatter(x=weeks, y=mean_p, mode="lines", name="with TV",
                         line=dict(color=CH["TV"], width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=weeks, y=p_cf, mode="lines", name="TV = 0",
                         line=dict(color=MUTED, width=2, dash="dash")), row=1, col=1)
lift = (mu_base - mu_cf).mean(axis=(0, 1))
fig.add_trace(go.Scatter(x=weeks, y=lift, mode="lines", showlegend=False,
                         line=dict(color=CH["TV"], width=2),
                         fill="tozeroy", fillcolor=band(CH["TV"])), row=2, col=1)
style(fig, "One counterfactual, drawn out — same demand shocks, different media",
      height=560, legend="below")
fig.update_yaxes(title_text="P(aware)", row=1, col=1)
fig.update_yaxes(title_text="Δ sales", row=2, col=1)
fig.show()

print(f"TV total effect, recomputed here : {tv_effect_draws.mean():,.0f}")
print(f"TV total effect, section 5 table : {me.loc['TV', 'total_effect']:,.0f}")
assert np.isclose(tv_effect_draws.mean(), me.loc["TV", "total_effect"], rtol=1e-3)
""".strip("\n")),
    md(r"""
## 8 · Where this fits in the platform

- **Spec & design rationale** — `technical-docs/structural-nested-mmm.md`; the recovery
  contract lives in `tests/mmm_extensions/test_structural_nested.py` (this notebook's fit
  settings and thresholds mirror its slow recovery suite).
- **Agent / DAG fit path** — a mediator DAG with structural features (mediator→mediator
  or control→mediator edges, survey likelihoods, latent factors) resolves to
  `dag_model_type="structural_nested_mmm"` automatically; `synth.brand_funnel_mff()`
  emits this world as an MFF long table *with* the survey columns
  (`awareness_count`/`awareness_trials`, `consideration_cat_1..5`) for that path.
- **Persistence** — `MMMSerializer.save(model, ...)` round-trips the whole family
  (extended flavor, panel-free reload); counterfactual methods work on the reloaded model.
- **Reporting** — the extended report extractor understands this model's vocabulary
  (`effect_<m>_on_y`, `indirect_<ch>_via_<med>`, `delta_direct_<ch>`), so the classic and
  client (Augur) report templates render from a fitted `StructuralNestedMMM` directly.

**What to remember when you build your own funnel:**

1. **Measure your mediators.** Every measured mediator pins its own scale (binomial:
   absolute; Likert: cutpoint-anchored; Gaussian: standardized) — that's what identifies
   media→mediator paths from the survey rather than the outcome residual. A fully latent
   middle stage is allowed but collapses to a rescaled composite edge.
2. **Let dynamics own the carryover.** AR(1)/random-walk mediators get saturation-only
   media inputs by default; forcing adstock on top creates an α↔ρ ridge.
3. **Anchor latent factors where the data is strong** — a factor needs ≥2 observation
   channels including a measured mediator, and its sign anchor should be a loading the
   data holds clearly nonzero.
4. **Keep direct paths tight** (the factory default) and set `affects_outcome` to mirror
   your DAG — in the Python API every mediator's outcome path defaults ON; only the
   agent's DAG fit path derives it from the drawn mediator → KPI edges.
5. **Trust counterfactual totals over coefficient products** — `get_mediation_effects()`
   / `get_channel_roas()` are exact under the fitted structure; `get_pathway_effects()`
   is the linearized routing view.
"""),
]


def main() -> None:
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    out = "structural_nested_mmm.ipynb"
    with open(out, "w") as f:
        nbformat.write(nb, f)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    print(f"wrote {out} with {len(nb.cells)} cells ({n_code} code)")


if __name__ == "__main__":
    main()
