"""Author showcase/showcase_03_beyond_one_kpi.ipynb (run from ``nbs/``).

    uv run --with nbformat python builders/build_showcase_03_beyond_one_kpi.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        showcase/showcase_03_beyond_one_kpi.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

Feature-showcase notebook 03 — "When one KPI is not the world". Tours the
model families beyond single-outcome BayesianMMM: NestedMMM mediation with a
direct/indirect decomposition, MultivariateMMM cross-effects (halo found,
absent cannibalization near zero), CombinedMMM + StructuralNestedMMM as
graph-build/config, DAG-driven model construction with backdoor
identification, the Model Garden contract + compat suite on a non-MMM
Bayesian CFA, LTV/CLV preprocessing, and the IV / front-door estimators.
Every number is computed in-notebook; fits are MAP (seconds, uncertainty
not calibrated — stated where it matters).
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

NESTED_DGP = r"""
# A world where the sales number alone cannot tell the story:
#   TV and Search work THROUGH awareness (brand route, planted).
#   Social sells directly (performance route, planted).
rng = np.random.default_rng(11)
n = 104
idx = pd.date_range("2023-01-02", periods=n, freq="W-MON")

tv = np.clip(rng.normal(120, 45, n), 0, None) * (rng.random(n) > 0.25)   # flighted
search = np.clip(rng.normal(80, 18, n), 5, None)                          # always-on
social = np.clip(rng.normal(60, 30, n), 0, None)                          # bursty
X_media = np.column_stack([tv, search, social])
CH = ["TV", "Search", "Social"]

sat = lambda x: x / (x + np.median(x[x > 0]))   # simple concavity for the DGP
awareness = 30 + 22 * sat(tv) + 14 * sat(search) + rng.normal(0, 1.5, n)
sales = 2000 + 38 * awareness + 900 * sat(social) + rng.normal(0, 55, n)

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                    subplot_titles=("Weekly media spend", "Brand awareness (tracker)",
                                    "Sales"))
for c, x in zip(CH, [tv, search, social]):
    fig.add_trace(go.Scatter(x=idx, y=x, name=c, line=dict(color=PALETTE[c], width=1.6)),
                  row=1, col=1)
fig.add_trace(go.Scatter(x=idx, y=awareness, name="awareness",
                         line=dict(color=INK, width=1.8), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=idx, y=sales, name="sales",
                         line=dict(color=GOOD, width=1.8), showlegend=False), row=3, col=1)
style(fig, 520, "One KPI, two routes to it: brand media moves awareness, awareness moves sales")
fig.show()
"""

NESTED_FIT = r"""
from mmm_framework.mmm_extensions.builders import (
    MediatorConfigBuilder,
    NestedModelConfigBuilder,
)
from mmm_framework.mmm_extensions.models import NestedMMM

mediator = (
    MediatorConfigBuilder("awareness")
    .fully_observed(observation_noise=0.05)   # a weekly tracker, not a sparse survey
    .with_positive_media_effect(sigma=1.0)
    .build()
)
config = (
    NestedModelConfigBuilder()
    .add_mediator(mediator)
    .map_channels_to_mediator("awareness", ["TV", "Search"])   # the claimed brand route
    .build()
)

nested = NestedMMM(
    X_media=X_media,
    y=sales,
    channel_names=CH,
    config=config,
    mediator_data={"awareness": awareness},
    index=idx,
)
t0 = time.time()
nested.fit(method="map", random_seed=3)
print(f"NestedMMM MAP fit in {time.time()-t0:.0f}s "
      "(approximate — a point estimate for a fast tour; NUTS for real intervals)")
"""

NESTED_WATERFALL = r"""
med = nested.get_mediation_effects()
print(med.round(3).to_string(index=False))

fig = go.Figure()
fig.add_trace(go.Bar(x=med["channel"], y=med["direct_effect"], name="direct route",
                     marker_color=MUTED))
fig.add_trace(go.Bar(x=med["channel"], y=med["indirect_via_awareness"],
                     name="through awareness", marker_color=GOLD))
fig.add_trace(go.Scatter(x=med["channel"], y=med["total_effect"], mode="markers",
                         name="total effect", marker=dict(color=INK, size=13,
                                                          symbol="diamond")))
fig.update_layout(barmode="relative")
fig.update_yaxes(title="effect on sales (original units, per unit saturated spend)")
style(fig, 400, "Mediation decomposition separates the brand route from the direct route")
fig.show()

for _, r in med.iterrows():
    print(f"{r['channel']:>7}: {r['proportion_mediated']:.0%} of its effect is mediated")
"""

MV_DGP_FIT = r"""
# Two products, one media plan. Planted truth:
#   product A is driven by TV + Search;
#   product B is driven by Social AND receives a HALO of +0.35 per unit of A's sales;
#   there is NO cannibalization of A by B (planted 0) — but we DECLARE one,
#   to see whether the model invents an effect that is not there.
rng2 = np.random.default_rng(21)
sales_a = 1800 + 900 * sat(tv) + 600 * sat(search) + rng2.normal(0, 90, n)
sales_b = 1200 + 500 * sat(social) + 0.35 * sales_a + rng2.normal(0, 60, n)
PSI_TRUE = np.array([[0.0, 0.35], [0.0, 0.0]])   # rows: source, cols: target

from mmm_framework.mmm_extensions.builders import (
    MultivariateModelConfigBuilder,
    OutcomeConfigBuilder,
    cannibalization_effect,
    halo_effect,
)
from mmm_framework.mmm_extensions.models import MultivariateMMM

mv_config = (
    MultivariateModelConfigBuilder()
    .add_outcome(OutcomeConfigBuilder("sales_a").with_positive_media_effects(0.5).build())
    .add_outcome(OutcomeConfigBuilder("sales_b").with_positive_media_effects(0.5).build())
    .add_cross_effect(halo_effect(source="sales_a", target="sales_b"))
    .add_cross_effect(cannibalization_effect(source="sales_b", target="sales_a"))
    .build()
)

mv = MultivariateMMM(
    X_media=X_media,
    outcome_data={"sales_a": sales_a, "sales_b": sales_b},
    channel_names=CH,
    config=mv_config,
    index=idx,
)
t0 = time.time()
mv.fit(method="map", random_seed=5)
print(f"MultivariateMMM MAP fit in {time.time()-t0:.0f}s")
print()
print(mv.get_cross_effects_summary().round(3).to_string(index=False))
"""

MV_MATRIX = r"""
psi_est = (
    mv.trace.posterior["psi_matrix"].mean(dim=("chain", "draw")).values
)
names = mv.outcome_names

fig = make_subplots(rows=1, cols=2, subplot_titles=("estimated psi (MAP)", "planted psi"),
                    horizontal_spacing=0.15)
for col, mat in ((1, psi_est), (2, PSI_TRUE)):
    fig.add_trace(
        go.Heatmap(z=mat, x=[f"→ {t}" for t in names], y=[f"{s} →" for s in names],
                   zmin=-0.4, zmax=0.4, colorscale="RdBu", reversescale=True,
                   showscale=(col == 1),
                   text=[[f"{v:+.3f}" for v in row] for row in mat],
                   texttemplate="%{text}", textfont=dict(size=14)),
        row=1, col=col,
    )
fig.update_yaxes(autorange="reversed")
style(fig, 360, "The planted halo is found; the declared-but-absent cannibalization stays at zero")
fig.show()

halo_err = abs(psi_est[0, 1] - 0.35)
print(f"halo  sales_a → sales_b: estimated {psi_est[0, 1]:+.3f} vs planted +0.350 "
      f"(abs error {halo_err:.3f})")
print(f"cannibalization sales_b → sales_a: estimated {psi_est[1, 0]:+.3f} vs planted 0 "
      "— the sign-constrained prior lets the data push it to the boundary, not past it")
"""

MV_CORR = r"""
corr = mv.get_correlation_matrix()
fig = go.Figure(
    go.Heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index),
               zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
               text=[[f"{v:.2f}" for v in row] for row in corr.values],
               texttemplate="%{text}", textfont=dict(size=14))
)
fig.update_yaxes(autorange="reversed")
style(fig, 340, "What the structural terms do not explain lands in the residual correlation")
fig.show()
print("An LKJ prior on the outcome residuals means shared shocks (weather, a site outage)")
print("are absorbed as correlation instead of being forced through the cross-effect.")
"""

COMBINED_BUILD = r"""
from mmm_framework.mmm_extensions.config import CombinedModelConfig
from mmm_framework.mmm_extensions.models import CombinedMMM

combined_config = CombinedModelConfig(
    nested=config,                     # the awareness mediation from section 1
    multivariate=mv_config,            # the two outcomes + cross-effects from section 2
    mediator_to_outcome_map={"awareness": ("sales_a",)},
)
combined = CombinedMMM(
    X_media=X_media,
    outcome_data={"sales_a": sales_a, "sales_b": sales_b},
    channel_names=CH,
    config=combined_config,
    mediator_data={"awareness": awareness},
    index=idx,
)
t0 = time.time()
graph = combined.model          # builds the full PyMC graph, no sampling
by_kind: dict[str, int] = {}
for rv in graph.free_RVs:
    key = rv.name.split("_")[0]
    by_kind[key] = by_kind.get(key, 0) + 1
print(f"CombinedMMM graph built in {time.time()-t0:.0f}s — {len(graph.free_RVs)} free "
      f"random variables:")
for k, v in sorted(by_kind.items()):
    print(f"  {k:<10} x{v}")
"""

STRUCTURAL_CONFIG = r"""
from mmm_framework.mmm_extensions import (
    MediatorDynamics,
    MediatorLikelihood,
    MediatorMeasurement,
    MediatorSpec,
    StructuralNestedConfig,
)

survey_config = StructuralNestedConfig(
    mediators=(
        MediatorSpec(
            name="awareness",
            channels=("TV",),
            dynamics=MediatorDynamics.AR1,        # awareness persists week to week
            measurement=MediatorMeasurement(
                likelihood=MediatorLikelihood.BINOMIAL,   # "aware: yes/no" counts
                design_effect=1.5,                        # clustered survey sample
            ),
        ),
        MediatorSpec(
            name="consideration",
            channels=("Search",),
            parents=("awareness",),               # a funnel: awareness feeds consideration
            measurement=MediatorMeasurement(
                likelihood=MediatorLikelihood.ORDERED,    # 5-point Likert
                n_categories=5,
            ),
        ),
    ),
)
print(survey_config.mediators[0])
"""

DAG_DRAW = r"""
from mmm_framework.dag_model_builder import DAGSpec, DAGNode, DAGEdge, NodeType

dag = DAGSpec(
    nodes=[
        DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA),
        DAGNode(id="search", variable_name="Search", node_type=NodeType.MEDIA),
        DAGNode(id="awareness", variable_name="Awareness", node_type=NodeType.MEDIATOR),
        DAGNode(id="demand", variable_name="CategoryDemand", node_type=NodeType.CONTROL),
        DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
    ],
    edges=[
        DAGEdge(source="tv", target="awareness"),
        DAGEdge(source="tv", target="sales"),
        DAGEdge(source="awareness", target="sales"),
        DAGEdge(source="search", target="sales"),
        DAGEdge(source="demand", target="search"),   # budgets chase demand ...
        DAGEdge(source="demand", target="sales"),    # ... and demand drives sales
    ],
)

POS = {"tv": (0, 1.6), "search": (0, 0.0), "awareness": (1, 1.6),
       "demand": (1, -0.9), "sales": (2, 0.5)}
KIND_COLOR = {NodeType.MEDIA: PALETTE["TV"], NodeType.MEDIATOR: GOLD,
              NodeType.CONTROL: BAD, NodeType.KPI: GOOD}
BACKDOOR = {("demand", "search"), ("demand", "sales")}

fig = go.Figure()
for e in dag.edges:
    (x0, y0), (x1, y1) = POS[e.source], POS[e.target]
    on_backdoor = (e.source, e.target) in BACKDOOR
    fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y",
                       axref="x", ayref="y", showarrow=True, arrowhead=3,
                       arrowwidth=3 if on_backdoor else 1.6,
                       arrowcolor=BAD if on_backdoor else MUTED,
                       standoff=34, startstandoff=34)
for node in dag.nodes:
    x, y = POS[node.id]
    fig.add_trace(go.Scatter(
        x=[x], y=[y], mode="markers+text", text=[node.variable_name],
        textposition="bottom center", textfont=dict(size=13, color=INK),
        marker=dict(size=34, color=KIND_COLOR[node.node_type], opacity=0.9),
        name=node.node_type.value, showlegend=False))
fig.update_xaxes(visible=False, range=[-0.5, 2.5])
fig.update_yaxes(visible=False, range=[-1.7, 2.4])
style(fig, 420, "The DAG is the model spec — and the red fork is a backdoor into Search")
fig.show()
"""

DAG_IDENTIFY = r"""
from mmm_framework.dag_model_builder import resolve_model_type
from mmm_framework.dag_model_builder.identification import identification_report
from mmm_framework.dag_model_builder.narrative import dag_human_reading

print(f"resolve_model_type(dag) -> {resolve_model_type(dag).value}  "
      "(the mediator makes it a NestedMMM)\n")

rep = identification_report(dag, treatment_id="search", outcome_id="sales")
print(f"Search -> Sales identifiable: {rep.identifiable}")
print(f"proposed adjustment set    : {rep.adjustment_set}")
for p in rep.backdoor_paths:
    state = "BLOCKED by " + ", ".join(p.blocked_by) if p.is_blocked else "OPEN"
    print(f"  backdoor path {p.render()}  [{state}]")
for note in rep.notes:
    print(f"  note: {note}")

from IPython.display import Markdown, display
display(Markdown(dag_human_reading(dag)))
"""

DAG_BUILD = r"""
# The same spec that draws the picture BUILDS the model. On a synthetic world
# whose columns match a simple DAG, the builder loads the MFF, resolves the
# model type, and returns a ready-to-fit model object.
from mmm_framework.dag_model_builder import DAGModelBuilder, create_simple_dag
from mmm_framework.synth import dgp, mff

world = dgp.make_clean(seed=7, n_weeks=80)
WORLD_CSV = ART / "showcase03_clean.csv"
mff.scenario_to_mff(world).to_csv(WORLD_CSV, index=False)

simple_dag = create_simple_dag(
    kpi_name="Sales",
    media_names=list(world.channels),
    control_names=list(world.controls.columns),
)
builder = DAGModelBuilder().with_dag(simple_dag).with_mff_data(str(WORLD_CSV))
print(f"validate()      -> valid={builder.validate().valid}")
print(f"get_model_type()-> {builder.get_model_type().value}")
model = builder.build()
print(f"build()         -> {type(model).__name__} "
      f"({len(model.panel.coords.channels)} channels, unfitted and ready)")
"""

GARDEN_CFA_FIT = r"""
import sys
sys.path.insert(0, str(Path.cwd().parents[1] / "examples" / "garden_models"))
from bayesian_cfa import BayesianCFA, synthetic_cfa_panel

from mmm_framework.config import ModelConfig
from mmm_framework.garden import validate_class
from mmm_framework.garden.contract import is_mmm_model, model_kind
from mmm_framework.model import TrendConfig
from mmm_framework.model.trend_config import TrendType

print(f"model_kind(BayesianCFA) = {model_kind(BayesianCFA)!r}; "
      f"is_mmm_model = {is_mmm_model(BayesianCFA)}")
print(f"validate_class(BayesianCFA) -> {validate_class(BayesianCFA) or 'no violations'}")

panel, TRUE_LOAD = synthetic_cfa_panel(n=400, seed=7)
cfa = BayesianCFA(
    panel, ModelConfig(), TrendConfig(type=TrendType.NONE),
    model_params={"n_factors": 2, "factor_assignment": [0, 0, 0, 1, 1, 1]},
)
t0 = time.time()
cfa.fit(method="map", random_seed=7)
print(f"\nBayesianCFA MAP fit in {time.time()-t0:.0f}s")

est = cfa.evaluate_estimands()   # same evaluate_estimands() path an MMM's ROI uses
for name, r in est.items():
    print(f"  estimand {name:10s} mean={r.mean:.3f}  status={r.status}")
"""

GARDEN_CFA_CHART = r"""
loadings = cfa.factor_loadings_summary()
print(loadings.round(3).to_string(index=False))

fig = go.Figure()
for factor, color in (("F1", PALETTE["TV"]), ("F2", PALETTE["Social"])):
    sub = loadings[loadings["factor"] == factor]
    fig.add_trace(go.Bar(
        x=sub["indicator"], y=sub["loading"], name=factor, marker_color=color,
        error_y=dict(type="data", symmetric=False,
                     array=sub["hdi_high"] - sub["loading"],
                     arrayminus=sub["loading"] - sub["hdi_low"])))
fig.add_hline(y=TRUE_LOAD, line=dict(color=TRUTH, dash="dot"),
              annotation_text=f"planted loading {TRUE_LOAD}")
fig.update_yaxes(title="standardized loading")
style(fig, 380, "A non-MMM garden model recovers its planted 2-factor structure")
fig.show()
"""

GARDEN_COMPAT = r"""
from mmm_framework.garden import run_compatibility_check

t0 = time.time()
report = run_compatibility_check(
    BayesianCFA,
    scenarios=("clean",),
    fit_method="map",
    n_weeks=60,
    check_carryover=False,
)
print(f"compat suite in {time.time()-t0:.0f}s — blocking_passed={report['blocking_passed']}")

tiers = report["tiers"]
status = ["skipped" if t["skipped"] else ("passed" if t["passed"] else "failed")
          for t in tiers]
colors = {"passed": GOOD, "skipped": MUTED, "failed": BAD}
fig = go.Figure(go.Bar(
    y=[t["name"] for t in tiers][::-1], x=[1] * len(tiers),
    orientation="h", marker_color=[colors[s] for s in status][::-1],
    text=status[::-1], textposition="inside", insidetextanchor="middle",
    textfont=dict(size=13, color="white")))
fig.update_xaxes(visible=False)
fig.update_yaxes(title=None)
style(fig, 380, "Compat tiers: a declared non-MMM passes the blocking tiers; MMM-only tiers skip, not fail",
      showlegend=False)
fig.show()
"""

LTV_RFM = r"""
from mmm_framework.ltv import transactions_to_rfm
from mmm_framework.synth.dgp_clv import make_clv_world

clv_world = make_clv_world(seed=7, n_customers=1500)
rfm = transactions_to_rfm(
    clv_world.transactions, value_col="value",
    observation_end=clv_world.observation_end,
)
print(rfm.head().round(2).to_string())
print(f"\n{len(rfm)} customers; "
      f"{(rfm['frequency'] == 0).mean():.0%} never purchased again in the window")

fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                    subplot_titles=("recency vs frequency (age-adjusted)",
                                    "repeat purchases per customer"))
fig.add_trace(go.Scatter(
    x=rfm["T"] - rfm["recency"], y=rfm["frequency"], mode="markers",
    marker=dict(color=PALETTE["TV"], size=5, opacity=0.35), showlegend=False),
    row=1, col=1)
fig.add_trace(go.Histogram(x=rfm["frequency"], marker_color=GOLD, showlegend=False),
              row=1, col=2)
fig.update_xaxes(title="weeks since last purchase", row=1, col=1)
fig.update_yaxes(title="repeat purchases", row=1, col=1)
fig.update_xaxes(title="repeat purchases", row=1, col=2)
fig.update_yaxes(title="customers", row=1, col=2)
style(fig, 380, "RFM compresses a transaction log into exactly what BG/NBD needs")
fig.show()
"""

ESTIMATORS = r"""
from mmm_framework.estimators import frontdoor_estimate, two_stage_least_squares

# Planted world: latent demand u drives BOTH spend (budget chasing) and sales.
# No measured control closes that backdoor. True causal effect of spend = 2.0
# (spend lifts site visits by 0.5, each visit is worth 4).
rng3 = np.random.default_rng(31)
m_ = 300
u = rng3.normal(0, 1, m_)                             # unobserved demand
z = rng3.normal(0, 1, m_)                             # auction-price shock (instrument)
spend = 50 + 8 * u + 5 * z + rng3.normal(0, 2, m_)    # spend chases demand
visits = 0.5 * spend + rng3.normal(0, 1.5, m_)        # mediator: u does NOT touch it
y = 200 + 4.0 * visits + 15 * u + rng3.normal(0, 4, m_)
TRUE_EFFECT = 0.5 * 4.0

# Naive OLS (the back-door estimate with the confounder unmeasured)
X = np.column_stack([spend, np.ones(m_)])
beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
resid = y - X @ beta
se = float(np.sqrt((resid @ resid) / (m_ - 2) * np.linalg.pinv(X.T @ X)[0, 0]))
naive = (float(beta[0]), float(beta[0]) - 1.96 * se, float(beta[0]) + 1.96 * se)

iv = two_stage_least_squares(y, spend, z)
fd = frontdoor_estimate(y, spend, visits)
print(f"naive OLS  : {naive[0]:.2f}  [{naive[1]:.2f}, {naive[2]:.2f}]  <- biased")
print(f"2SLS (IV)  : {iv.effect:.2f}  [{iv.ci_low:.2f}, {iv.ci_high:.2f}]  "
      f"first-stage F={iv.first_stage_f:.0f} (weak if <10)")
print(f"front-door : {fd.effect:.2f}  [{fd.ci_low:.2f}, {fd.ci_high:.2f}]")

rows = [("naive OLS", *naive, BAD), ("2SLS (IV)", iv.effect, iv.ci_low, iv.ci_high, GOOD),
        ("front-door", fd.effect, fd.ci_low, fd.ci_high, GOOD)]
fig = go.Figure()
for i, (name, eff, lo, hi, color) in enumerate(rows):
    fig.add_trace(go.Scatter(x=[lo, hi], y=[i, i], mode="lines",
                             line=dict(color=color, width=6), opacity=0.4,
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[eff], y=[i], mode="markers",
                             marker=dict(color=color, size=13), showlegend=False))
fig.add_vline(x=TRUE_EFFECT, line=dict(color=TRUTH, dash="dot"),
              annotation_text=f"planted truth {TRUE_EFFECT}")
fig.update_yaxes(tickmode="array", tickvals=[0, 1, 2],
                 ticktext=[r[0] for r in rows])
fig.update_xaxes(title="estimated effect of one spend unit on sales")
style(fig, 340, "When the backdoor cannot be closed, IV and front-door still get home")
fig.show()
"""

CELLS = [
    md(r"""
# When one KPI is not the world

A single-outcome MMM answers one question: how did media move *this* number.
Real businesses ask more — did TV work through the brand or at the shelf, did
product B's growth come out of product A, is the survey mediator trustworthy,
what is a customer *worth*, and what happens when the model's core assumption
(no unobserved confounding) simply fails. This notebook tours the framework's
answers:

1. **`NestedMMM`** — mediation: direct vs through-awareness decomposition
2. **`MultivariateMMM`** — two outcomes, halo + cannibalization cross-effects
3. **`CombinedMMM` + `StructuralNestedMMM`** — both at once, and survey-grade mediators
4. **`dag_model_builder`** — the causal graph *is* the model spec, with a backdoor reading
5. **Model Garden** — a bespoke non-MMM family (Bayesian CFA) riding the same pipeline
6. **`ltv/`** — CLV preprocessing for "worth", not just "sales"
7. **`estimators/`** — IV and front-door when the backdoor stays open

Every number here is computed in this notebook. Fits use `method="map"` so
the tour bakes in minutes: MAP is a **point estimate** — honest for showing
structure, not for decision-grade intervals. Refit with NUTS before trusting
uncertainty.
"""),
    code(SETUP),
    md(r"""
## 1 · NestedMMM — did TV sell, or did it build the brand that sells?

A flat MMM regresses sales on all channels and forces every effect through one
door. `NestedMMM` models the claimed causal route explicitly: media →
mediator (awareness) → sales, with optional direct paths. The payoff is a
decomposition a flat model cannot give you — how much of a channel's effect
travels through the brand. In the world below the routing is **planted**: TV
and Search work only through awareness, Social only sells directly.
"""),
    code(NESTED_DGP),
    code(NESTED_FIT),
    code(NESTED_WATERFALL),
    md(r"""
The decomposition reads the planted structure back: the brand channels come
out mostly mediated, Social mostly direct. The `proportion_mediated` column
is the strategic number — it says which budget lines are building an asset
(awareness) and which are harvesting demand. Mediator observation can be much
weaker than this weekly tracker: `partially_observed()` handles monthly
surveys, `fully_latent()` drops observation entirely and identifies the state
from structure alone (the payback notebook shows what that costs).
"""),
    md(r"""
## 2 · MultivariateMMM — two products, one media plan

When one brand's media lifts (halo) or steals (cannibalization) another's
sales, per-product single-KPI models silently double-count. `MultivariateMMM`
fits all outcomes jointly: correlated residuals via an LKJ prior, plus
explicit signed cross-effects `psi[source → target]` in original units. The
test below is deliberately adversarial — we **declare** a cannibalization that
does not exist in the data, alongside a halo that does.
"""),
    code(MV_DGP_FIT),
    code(MV_MATRIX),
    code(MV_CORR),
    md(r"""
Both halves matter. Finding the halo is the easy claim; *not* inventing the
declared cannibalization is what makes the estimate trustworthy — the
sign-constrained prior can only shrink an absent effect to its zero boundary.
One honest caveat travels with any psi on observed outcomes: it is confounded
with residual correlation, so read it as a cross-outcome association whose
sign you asserted, not a free-standing causal discovery.
"""),
    md(r"""
## 3 · CombinedMMM and StructuralNestedMMM — both worlds at once

`CombinedMMM` is the composition: mediated pathways **and** multiple
correlated outcomes in one graph, with `mediator_to_outcome_map` saying which
outcome each mediator feeds. Building the graph is cheap, so we do it for
real; fitting it well is NUTS-grade work (the geometry is exactly the
label-switching, ridge-prone kind the extension models document), so this
tour stops at the build.
"""),
    code(COMBINED_BUILD),
    md(r"""
### The survey-mediator story

`NestedMMM` treats a mediator as a Gaussian series. Real brand trackers are
nothing like that: they arrive as **"we asked 500 people, 210 said yes"** —
a binomial count whose precision depends on that week's sample size — or as
5-point Likert scales, and awareness *persists* (this week's awareness is
mostly last week's). `StructuralNestedMMM` exists for exactly this: per-
mediator dynamics (static / AR(1) / random walk), measurement families
(Gaussian / binomial / ordered / latent), mediator-on-mediator funnels, and
shared latent factors. The config below is real and validated at
construction; a full fit-and-recover walkthrough lives in
`nbs/demos/structural_nested.ipynb`.
"""),
    code(STRUCTURAL_CONFIG),
    md(r"""
Two defaults in that config carry hard-won lessons: adstock is auto-disabled
on AR(1) mediators (the state already carries the media effect forward —
adstock on top creates an alpha/rho ridge), and the direct-effect prior is
deliberately tight because an over-wide direct path steals the mediated
signal (`technical-docs/nested-recovery-search.md` documents the search that
found this).
"""),
    md(r"""
## 4 · dag_model_builder — the graph is the spec

Everything above required knowing which model class to reach for. The DAG
builder inverts that: you state the causal claims as a graph, and the
framework resolves the model family, translates node configs, and — before
any fitting — prices the causal question itself: which effects are identified,
and what must be adjusted for. Here the graph contains the classic MMM trap:
category demand drives *both* Search spend (budgets chase demand) and sales.
"""),
    code(DAG_DRAW),
    code(DAG_IDENTIFY),
    md(r"""
The identification report is the pre-fit contract: Search's effect is only
identified *because* `CategoryDemand` is measured and adjustable — delete
that node and the same report says NOT identified, before a single draw is
sampled. The plain-English reading (`dag_human_reading`) is what the agent
UI shows a non-statistician. And the spec is executable:
"""),
    code(DAG_BUILD),
    md(r"""
## 5 · Model Garden — when the framework's families are not enough

The Garden is the escape hatch with a contract: author a bespoke model as a
`CustomMMM` subclass, declare its kind, and it rides the same build → fit →
estimand → serialize → report pipeline. The strongest proof is a model that
is not an MMM at all — a **Bayesian confirmatory factor analysis** that
declares `__garden_model_kind__ = "cfa"`, has zero media channels, and still
answers `evaluate_estimands()` like any MMM answers ROI.
"""),
    code(GARDEN_CFA_FIT),
    code(GARDEN_CFA_CHART),
    md(r"""
Before a garden model is registered for the oracle to use, it must survive
the **compatibility suite** — tiered checks from static contract reading
through build, fit, trace conventions, scaling invariance and accuracy on
synthetic worlds. The tiers are capability-gated: a declared non-MMM *skips*
the MMM-only tiers rather than failing them, so "passed" keeps meaning
something.
"""),
    code(GARDEN_COMPAT),
    md(r"""
A Bayesian latent class analysis (`examples/garden_models/bayesian_lca.py`)
ships the same way, as do the brand-building and CLV models below — the
Garden is how one pipeline serves a growing zoo without forking.
"""),
    md(r"""
## 6 · ltv/ — measuring worth, not just sales

An MMM values a conversion at the sale amount. If customers repeat-purchase,
that undercounts the channels that acquire *durable* customers. The `ltv`
package brings the classic BG/NBD + Gamma-Gamma machinery in: a pandas-only
preprocessor collapses a raw transaction log to RFM per customer, and the
**`BayesianCLV`** garden model fits it (planted-truth recovery is gated in
`tests/test_clv_garden_model.py`). Below, the preprocessing on a synthetic
world with known population parameters.
"""),
    code(LTV_RFM),
    md(r"""
Fitting is one more garden fit — `BayesianCLV(rfm_panel(rfm), ModelConfig(),
TrendConfig(type=TrendType.NONE), model_params={"horizon_periods": 26})` then
`.fit(method="map")` — and the result plugs into planning via
`ltv.clv_to_cac` and `new_customer_clv_series` (CLV-valued KPIs for the
MMM itself). We leave the fit to `nbs/demos/atelier_ltv_problems.ipynb`,
which runs five LTV problem settings end-to-end; here the point is that
"customer worth" is a first-class KPI, not a spreadsheet afterthought.
"""),
    md(r"""
## 7 · estimators/ — when the backdoor stays open

Every model above assumes measured confounders. `estimators/` covers the
case the DAG layer flags as *not* backdoor-identifiable: a linear **2SLS**
estimator for when you have an instrument, and a linear **front-door**
estimator for when a clean mediator carries the whole effect. Both are pure
NumPy and report their own weakness diagnostics (a first-stage F below 10
means the instrument is too weak to trust).
"""),
    code(ESTIMATORS),
    md(r"""
The naive estimate is confidently wrong — its interval excludes the truth —
which is the exact failure mode the DAG layer's "NOT identified" verdict is
warning about. The estimators are deliberately linear and humble; their job
is triangulation against the Bayesian model, not replacement of it.

## Where to go deeper

- **Extension models** — `technical-docs/extension-model-priors.md` (priors,
  trend/seasonality wiring), `technical-docs/structural-nested-mmm.md` and
  `technical-docs/nested-recovery-search.md` (why the structural defaults are
  what they are), `nbs/demos/structural_nested.ipynb` (full survey-mediator
  fit), `nbs/stress/stress_04_extension_traps.ipynb` (where these models break).
- **DAG builder + causal reading** — `nbs/causal/` notebooks 00–10 (the
  ladder, confounding, mediation, sealed answer keys),
  `technical-docs/confounding-sensitivity.md` (pricing the assumption).
- **Model Garden** — `technical-docs/custom-model-config.md`,
  `technical-docs/non-mmm-families.md`, `examples/garden_models/` (CFA, LCA,
  CLV, structural brand models), `docs/` Atelier pages for the in-app IDE.
- **LTV / CLV** — `nbs/demos/atelier_ltv_problems.ipynb` (five LTV problems,
  five Atelier models), `tests/test_ltv_loop.py` (CLV-valued planning).
- **Estimands across all families** — `technical-docs/estimands.md`: the
  declarative registry that let the CFA answer `evaluate_estimands()` above.
- **Sibling series** — `workshop_*` (Bayesian basics), `math_*` (the
  transforms' mathematics), `lifecycle_*` (the T₀–T₅ measurement loop).
"""),
]


def main() -> None:
    nb = new_notebook(cells=CELLS)
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    out = "showcase/showcase_03_beyond_one_kpi.ipynb"
    with open(out, "w") as fh:
        nbformat.write(nb, fh)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
