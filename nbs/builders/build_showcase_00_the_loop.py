"""Author showcase/showcase_00_the_loop.ipynb (run from ``nbs/``).

    uv run --with nbformat python builders/build_showcase_00_the_loop.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        showcase/showcase_00_the_loop.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

Series opener for the Feature Showcase: what mmm-framework is (production
Bayesian MMM built directly on PyMC, not a PyMC-Marketing subclass), the
fit -> validate -> plan -> commit -> measure loop, and one complete miniature
pass: a synthetic world with a sealed causal answer key, MFFLoader, one
NUTS-lite BayesianMMM fit, ROI-with-uncertainty vs planted truth, a weekly
decomposition, an HTML report, a serialization round-trip, and a treemap of
the whole package annotated by which showcase notebook covers each part.
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
for _n in ("pymc", "pymc.sampling", "pymc.stats.convergence",
           "numpyro", "jax", "arviz", "pytensor"):
    _lg = logging.getLogger(_n); _lg.setLevel(logging.ERROR); _lg.propagate = False

INK, MUTED, TRUTH = "#1f2430", "#8a8f98", "#111418"
GOOD, BAD, GOLD = "#3d7a5c", "#b4552d", "#c9962e"
PALETTE = {"TV": "#4464ad", "Search": "#c9962e", "Social": "#3d7a5c", "Display": "#b4552d"}

def style(fig, height=380, title=None, **kw):
    fig.update_layout(height=height, title=title, margin=dict(t=64, l=64, r=30, b=52),
                      font=dict(size=12), **kw)
    return fig

ART = Path.cwd().parent / "artifacts" / "showcase"
ART.mkdir(parents=True, exist_ok=True)
print("Setup ready. Artifacts ->", ART)
"""

WORLD = r"""
from mmm_framework.synth import dgp

t0 = time.time()
world = dgp.make_clean(seed=3, n_weeks=104)   # 2 years, weekly, national
CHANNELS = list(world.spend.columns)
weeks = world.weeks

# The answer key: per-week TRUE incremental sales for each channel, computed
# from the world's own structural response function by the same counterfactual
# the model will later use (zero the channel, difference the noiseless mean).
fn = world.response_fn
S = world.spend.to_numpy(float)
mu_full = fn(S)
true_weekly = {}
for j, c in enumerate(CHANNELS):
    S0 = S.copy(); S0[:, j] = 0.0
    true_weekly[c] = mu_full - fn(S0)
true_weekly = pd.DataFrame(true_weekly, index=weeks)
true_base = pd.Series(fn(np.zeros_like(S)), index=weeks)

print(f"world built in {time.time()-t0:.1f}s — {len(weeks)} weeks, "
      f"{len(CHANNELS)} channels, 1 control ({list(world.controls.columns)[0]})")
print("\nplanted totals (the sealed answer key):")
key = pd.DataFrame({"true incremental sales": world.true_contribution,
                    "true ROAS": world.true_roas}).round(2)
print(key.to_string())
"""

WORLD_FIG = r"""
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    row_heights=[0.45, 0.55],
                    subplot_titles=("Weekly media spend by channel",
                                    "The KPI the model will see (truth + noise)"))
for c in CHANNELS:
    fig.add_trace(go.Scatter(x=weeks, y=world.spend[c], name=c, mode="lines",
                             line=dict(color=PALETTE[c], width=1.6)), row=1, col=1)
fig.add_trace(go.Scatter(x=weeks, y=world.y, name="observed Sales", mode="lines",
                         line=dict(color=INK, width=1.8)), row=2, col=1)
fig.add_trace(go.Scatter(x=weeks, y=world.mu, name="noiseless mean (hidden)",
                         mode="lines", line=dict(color=MUTED, width=1.2, dash="dot")),
              row=2, col=1)
fig.update_yaxes(title="spend", row=1, col=1)
fig.update_yaxes(title="Sales", row=2, col=1)
style(fig, 520, "Four channels pulse independently while the KPI carries trend, "
                "season and noise")
fig.show()
"""

KEY_FIG = r"""
fig = go.Figure()
fig.add_trace(go.Scatter(x=weeks, y=true_base, name="true baseline",
                         mode="lines", stackgroup="one",
                         line=dict(width=0.5, color=MUTED),
                         fillcolor="rgba(138,143,152,0.35)"))
for c in CHANNELS:
    fig.add_trace(go.Scatter(x=weeks, y=true_weekly[c], name=f"{c} (true)",
                             mode="lines", stackgroup="one",
                             line=dict(width=0.5, color=PALETTE[c])))
fig.add_trace(go.Scatter(x=weeks, y=world.y, name="observed Sales", mode="lines",
                         line=dict(color=INK, width=1.4)))
fig.update_yaxes(title="Sales per week")
style(fig, 420, "The answer key: every week's sales split into its true causes")
fig.show()

share = world.true_contribution.sum() / world.y.sum()
print(f"media truly drives {share:.0%} of total sales in this world; "
      f"the rest is baseline (trend, season, price)")
"""

LOAD = r"""
from mmm_framework.config import (ControlVariableConfig, DimensionType, KPIConfig,
                                  MediaChannelConfig, MFFConfig)
from mmm_framework.data_loader import MFFLoader
from mmm_framework.synth import mff

DATA = ART / "the_loop_world.csv"
mff.scenario_to_mff(world).to_csv(DATA, index=False)
print(pd.read_csv(DATA).head(3).to_string(index=False), "\n")

mff_config = MFFConfig(
    kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
    media_channels=[MediaChannelConfig(name=c, dimensions=[DimensionType.PERIOD])
                    for c in CHANNELS],
    controls=[ControlVariableConfig(name="Price", dimensions=[DimensionType.PERIOD])],
)
panel = MFFLoader(mff_config).load(str(DATA)).build_panel()
print(f"PanelDataset: {panel.n_obs} observations, KPI '{panel.config.kpi.name}', "
      f"channels {list(panel.X_media.columns)}")
"""

FIT = r"""
from mmm_framework import BayesianMMM, ModelConfigBuilder, TrendConfig, TrendType

model_config = (
    ModelConfigBuilder()
    .bayesian_numpyro()     # NUTS via NumPyro/JAX — the default fast backend
    .with_chains(2)         # showcase-speed; production runs use 4 x 2000
    .with_draws(400)
    .with_tune(400)
    .build()
)
mmm = BayesianMMM(panel, model_config, TrendConfig(type=TrendType.LINEAR))

t0 = time.time()
results = mmm.fit(random_seed=42)
print(f"NUTS fit in {time.time()-t0:.0f}s")

gate = {
    "r-hat max": round(results.diagnostics["rhat_max"], 3),
    "divergences": int(results.diagnostics["divergences"]),
    "min bulk ESS": int(results.diagnostics["ess_bulk_min"]),
}
print("diagnostics gate:", gate)
assert results.diagnostics["rhat_max"] < 1.05 and gate["divergences"] == 0
print("gate is green — results may be read")
"""

ROI = r"""
from mmm_framework.analysis import MMMAnalyzer

analyzer = MMMAnalyzer(mmm)
t0 = time.time()
roi = analyzer.compute_channel_roi(random_seed=42).set_index("Channel")
print(f"compute_channel_roi in {time.time()-t0:.0f}s")
cols = ["Total Spend", "Total Contribution", "ROI",
        "Contribution HDI Low", "Contribution HDI High"]
print(roi[cols].round(2).to_string())
"""

ROI_FIG = r"""
fig = go.Figure()
for i, c in enumerate(CHANNELS):
    r = roi.loc[c]
    lo = r["Contribution HDI Low"] / r["Total Spend"]   # ROI interval = contribution
    hi = r["Contribution HDI High"] / r["Total Spend"]  # HDI over the same spend
    fig.add_trace(go.Scatter(x=[lo, hi], y=[i, i], mode="lines",
                             line=dict(color=PALETTE[c], width=6), opacity=0.45,
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[r["ROI"]], y=[i], mode="markers",
                             marker=dict(color=PALETTE[c], size=13),
                             name="posterior mean (94% HDI)", showlegend=(i == 0)))
    fig.add_trace(go.Scatter(x=[world.true_roas[c]], y=[i], mode="markers",
                             marker=dict(color=TRUTH, symbol="line-ns-open", size=20,
                                         line=dict(width=3)),
                             name="planted truth", showlegend=(i == 0)))
fig.update_yaxes(tickmode="array", tickvals=list(range(len(CHANNELS))),
                 ticktext=CHANNELS)
fig.update_xaxes(title="ROAS — incremental sales per unit of spend")
style(fig, 360, "ROI comes with an interval, and the interval covers the "
                "planted truth")
fig.show()

covered = sum(
    roi.loc[c, "Contribution HDI Low"] / roi.loc[c, "Total Spend"]
    <= world.true_roas[c]
    <= roi.loc[c, "Contribution HDI High"] / roi.loc[c, "Total Spend"]
    for c in CHANNELS
)
print(f"94% interval covers planted ROAS for {covered} of {len(CHANNELS)} channels")
"""

DECOMP = r"""
contrib = analyzer.compute_counterfactual_contributions(random_seed=42)
fitted_weekly = contrib.channel_contributions.set_axis(weeks, axis=0)
fitted_base = pd.Series(contrib.baseline_prediction, index=weeks) \
    - fitted_weekly.sum(axis=1)

fig = go.Figure()
fig.add_trace(go.Scatter(x=weeks, y=fitted_base, name="baseline (fitted)",
                         mode="lines", stackgroup="one",
                         line=dict(width=0.5, color=MUTED),
                         fillcolor="rgba(138,143,152,0.35)"))
for c in CHANNELS:
    fig.add_trace(go.Scatter(x=weeks, y=fitted_weekly[c], name=c, mode="lines",
                             stackgroup="one",
                             line=dict(width=0.5, color=PALETTE[c])))
fig.add_trace(go.Scatter(x=weeks, y=world.y, name="observed Sales",
                         mode="markers", marker=dict(color=INK, size=3)))
fig.update_yaxes(title="Sales per week")
style(fig, 420, "The fitted decomposition rebuilds every observed week from "
                "base plus media")
fig.show()
"""

DECOMP_TRUTH_FIG = r"""
fig = make_subplots(rows=2, cols=2, shared_xaxes=True,
                    subplot_titles=CHANNELS, vertical_spacing=0.10)
for i, c in enumerate(CHANNELS):
    r, col = i // 2 + 1, i % 2 + 1
    fig.add_trace(go.Scatter(x=weeks, y=true_weekly[c], name="planted truth",
                             mode="lines", line=dict(color=TRUTH, width=1.2,
                                                     dash="dot"),
                             showlegend=(i == 0)), row=r, col=col)
    fig.add_trace(go.Scatter(x=weeks, y=fitted_weekly[c], name="fitted",
                             mode="lines", line=dict(color=PALETTE[c], width=1.6),
                             showlegend=(i == 0)), row=r, col=col)
style(fig, 520, "Week by week, the fitted contributions track the planted truth")
fig.show()

err = {c: float(np.mean(np.abs(fitted_weekly[c] - true_weekly[c])))
       for c in CHANNELS}
print("mean absolute weekly error (Sales units):",
      {k: round(v, 1) for k, v in err.items()})
"""

REPORT = r"""
from mmm_framework.reporting import MMMReportGenerator, ReportConfig

t0 = time.time()
generator = MMMReportGenerator(model=mmm, results=results, config=ReportConfig())
report_path = generator.to_html(ART / "showcase_00_report.html")
print(f"report written in {time.time()-t0:.0f}s ->", report_path)
print(f"size: {report_path.stat().st_size / 1e6:.1f} MB — a single portable HTML "
      f"file; open it in any browser")
"""

SERIALIZE = r"""
from mmm_framework.serialization import MMMSerializer

MMMSerializer.save(mmm, ART / "showcase_00_model")
loaded = MMMSerializer.load(ART / "showcase_00_model", panel)  # core save needs
                                                               # the panel back
same = np.allclose(
    loaded.predict(return_original_scale=True, random_seed=1).y_pred_mean,
    mmm.predict(return_original_scale=True, random_seed=1).y_pred_mean,
)
print("round-trip predictions identical:", same)
assert same
"""

TREEMAP = r"""
import mmm_framework
PKG = Path(mmm_framework.__file__).parent

COVERAGE = {  # subpackage -> (showcase notebook, theme)
    **{p: ("01 data", "data in") for p in
       ["data_loader.py", "dataset.py", "dataset_loader.py", "data_preparation.py",
        "datasets", "data_studio", "eda", "excel_config", "integrations", "synth",
        "storage"]},
    **{p: ("02 models", "the core model") for p in
       ["model", "config", "builders", "transforms", "dag_model_builder", "utils",
        "analysis.py", "serialization.py"]},
    **{p: ("03 families", "beyond one MMM") for p in
       ["mmm_extensions", "garden", "ltv", "estimators", "frequentist"]},
    **{p: ("04 trust", "earning belief") for p in
       ["diagnostics", "validation", "calibration", "estimands", "reporting"]},
    **{p: ("05 planning", "spending the belief") for p in
       ["planning", "finance", "continuous_learning"]},
    **{p: ("06 operate", "the app layer") for p in
       ["agents", "platform", "auth", "api", "security", "jobs.py", "lineage.py"]},
}
NB_COLOR = {"01 data": "#4464ad", "02 models": "#3d7a5c", "03 families": "#c9962e",
            "04 trust": "#b4552d", "05 planning": "#6b5b95", "06 operate": "#8a8f98"}

def loc(p: Path) -> int:
    files = [p] if p.is_file() else list(p.rglob("*.py"))
    return sum(len(f.read_text().splitlines()) for f in files
               if "__pycache__" not in str(f))

labels, parents, values, colors = [], [], [], []
for nb, color in NB_COLOR.items():
    labels.append(nb); parents.append(""); values.append(0)
    colors.append(color)
for name, (nb, _) in sorted(COVERAGE.items()):
    p = PKG / name
    if not p.exists():
        continue
    labels.append(name.removesuffix(".py")); parents.append(nb)
    values.append(loc(p)); colors.append(NB_COLOR[nb])

fig = go.Figure(go.Treemap(labels=labels, parents=parents, values=values,
                           marker=dict(colors=colors), branchvalues="remainder",
                           textinfo="label+value",
                           hovertemplate="%{label}: %{value} lines of code"
                                         "<extra>%{parent}</extra>"))
style(fig, 560, "The whole package, sized by lines of code and grouped by the "
                "showcase notebook that covers it")
fig.show()

total = sum(values)
print(f"{total:,} lines of Python across "
      f"{sum(1 for pa in parents if pa)} top-level modules — this notebook "
      f"touched perhaps a tenth of it")
"""

CELLS = [
    md(r"""
# The measurement loop in one sitting

**mmm-framework** is a production Bayesian Marketing Mix Modeling framework
built **directly on PyMC**. It is not a wrapper around PyMC-Marketing and not a
subclass of it (that library appears only as an optional reporting interop
target). The design bet is different: methodological rigor as a feature —
genuine uncertainty on every number, pre-specified analyses, and refusals
instead of silently wrong outputs.

The framework is organized around one loop, and so is this notebook series:

> **fit** a model → **validate** that it earned belief → **plan** spend and
> experiments against it → **commit** a plan of record → **measure** what
> actually happened, and feed that back into the next fit.

This opener runs one complete pass in miniature, on a synthetic world whose
true causal effects we planted ourselves — so every claim the model makes can
be graded against a sealed answer key:

1. build a **world with a causal answer key** and chart what is really going on
2. load it the way real data arrives: an **MFF file** through `MFFLoader`
3. one **NUTS fit** of `BayesianMMM`, gated on convergence diagnostics
4. **ROI with uncertainty**, graded against the planted truth
5. the **weekly decomposition**, graded the same way
6. a portable **HTML report**, and a **save/load round-trip**
7. a map of the **whole package**, annotated by which notebook covers what

Fits here are deliberately small (2 chains, 400 draws, 104 weeks) so the whole
notebook bakes in minutes. Production guidance is 4 chains and 2000 draws.
"""),
    code(SETUP),
    md(r"""
## 1 · A world where we know the answer

Real marketing data never comes with ground truth, which is exactly why an MMM
framework needs synthetic worlds that do. `synth.dgp.make_clean` builds a
2-year national world from the model's own structural family — geometric
adstock, concave saturation, trend, seasonality, a price control — and keeps
the **response function** that generated it. Truth here is defined the same
way the model reports effects: zero out a channel, difference the noiseless
mean. Truth and estimate are the same estimand on the same scale, so the gap
between them is pure model error.

(`make_clean` is the positive control. Its siblings — `make_unobserved_confounding`,
`make_adstock_misspec`, and a dozen more — each break exactly one assumption;
notebook **04 trust** and the `stress_` series live on them.)
"""),
    code(WORLD),
    code(WORLD_FIG),
    md(r"""
The observed KPI is the grey dotted structural mean plus noise. The model will
only ever see the solid line — but *we* can see the full causal split:
"""),
    code(KEY_FIG),
    md(r"""
## 2 · Data in: the Master Flat File

Real engagements deliver data as a long "Master Flat File" (MFF): one row per
(variable, period), every KPI, channel and control stacked in one CSV.
`MFFLoader` takes that file plus an `MFFConfig` declaring each column's role
and returns the `PanelDataset` every model in the framework consumes. We
round-trip our world through a real CSV so the path is the one you would use
on client data. (Notebook **01 data** covers the wide-format reshaper, geo
panels, Data Studio and the EDA gates.)
"""),
    code(LOAD),
    md(r"""
## 3 · Fit: one Bayesian model, one diagnostics gate

`BayesianMMM` takes the panel, a `ModelConfig` (built fluently), and a trend
config. The fit is full NUTS — a joint posterior over every effect
coefficient, adstock decay, saturation curve, trend, seasonality and noise
parameter at once. That joint posterior is where honest uncertainty comes
from, and it is why every downstream number can carry an interval.

House rule: **no result is read before the diagnostics gate** — r-hat near 1,
zero divergences, healthy effective sample size. The assert below is the gate;
if sampling had failed, this notebook would refuse to continue rather than
show you numbers from a broken chain.
"""),
    code(FIT),
    md(r"""
## 4 · Validate: grade the model against the answer key

`MMMAnalyzer.compute_channel_roi` reports each channel's return as
incremental-sales-per-unit-spend, with the contribution's 94% HDI alongside.
Dividing that interval by the (fixed, observed) spend gives the ROI interval.
The planted true ROAS is the black tick — on a clean world the intervals
should cover it, and that coverage claim is itself checked by a dedicated
diagnostic (`diagnostics/coverage.py`, notebook **04 trust**).
"""),
    code(ROI),
    code(ROI_FIG),
    md(r"""
The same counterfactual machinery yields a week-by-week decomposition —
prediction with all channels minus prediction with one channel zeroed. This is
the chart a CMO reads: where did each week's sales come from?
"""),
    code(DECOMP),
    md(r"""
And because this world has an answer key, we can grade the decomposition
per-week per-channel — a check no real dataset permits:
"""),
    code(DECOMP_TRUTH_FIG),
    md(r"""
## 5 · Report and persist

`MMMReportGenerator` renders the fitted model into a single self-contained
HTML file — decomposition, ROI, response curves, diagnostics, each section
evidence-coded. `MMMSerializer` persists the model; a core save stores the
trace and config, and `load()` takes the panel back to rebuild the graph.
"""),
    code(REPORT),
    code(SERIALIZE),
    md(r"""
The loop's remaining stations — **plan** (budget optimization, experiment
design priced in expected value of information), **commit** (a plan of record
with committed intervals), **measure** (variance-to-plan, experiment
calibration feeding the next fit) — get notebooks **05** and **06** to
themselves. What this pass showed is the spine they all attach to: a fitted
posterior that survived its gate.
"""),
    md(r"""
## 6 · The map: what else is in the box

The treemap below is computed live from the installed package — every
top-level module, sized by lines of code, grouped by the showcase notebook
that covers it:

- **01 data** — loaders, wide-format reshaping, geo panels, Data Studio, EDA
  gates, connectors, the synthetic worlds
- **02 models** — `BayesianMMM` internals, transforms, configs and builders,
  the DAG-driven model builder
- **03 families** — nested/multivariate/combined extensions, the Model Garden,
  LTV, frequentist ridge
- **04 trust** — diagnostics (SBC, coverage, identification), validation
  (backtests, spec curves, refutation), calibration to experiments, reporting
- **05 planning** — EIG/EVOI experiment design, budget optimization, the
  finance basis behind every dollar claim, continuous learning
- **06 operate** — the app layer: the LangGraph agent (`agents/`), the
  platform services (`platform/`), auth and security, plus the separate
  FastAPI server package (`server/`) and React frontend that are not inside
  this package at all. Notebook 06 covers these in prose; they are services,
  not notebook APIs.
"""),
    code(TREEMAP),
    md(r"""
## Where to go deeper

**Sibling notebook series** (all under `nbs/`):

- `workshop_00..05` — beginner Bayesian workshop; build up to your first MMM
- `math_00..06` — the mathematics companion (adstock, saturation, the full
  Bayesian model)
- `causal/00..10` — MMM as a causal model: confounding, mediation,
  experiments, the closed loop
- `lifecycle_00..06` — the T0→T5 measurement loop on a fitted MMM, in full
- `stress_00..06` — the violation worlds; how the model fails and what
  catches it
- `demos/` — focused deep dives (payback horizon, critique-to-decision,
  backtest validation, continuous learning)

**Specs and guides** (`technical-docs/` and `docs/`):

- `technical-docs/engineering-notes.md` — the subsystem index; invariants and
  traps per module
- `technical-docs/estimands.md`, `technical-docs/coverage-diagnostics.md`,
  `technical-docs/experiment-economics.md` — the trust and planning contracts
- `technical-docs/sampling-failure-playbook.md` — when a fit will not converge
- `docs/getting-started.html` and the API reference — install, quickstart,
  full API

**Next in this series**: `showcase_01_data` — everything between a client CSV
and a fit-ready panel.
"""),
]


def main() -> None:
    nb = new_notebook(cells=CELLS)
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    out = "showcase/showcase_00_the_loop.ipynb"
    with open(out, "w") as fh:
        nbformat.write(nb, fh)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
