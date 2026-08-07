"""Author showcase/showcase_01_data_foundations.ipynb (run from ``nbs/``).

    uv run --with nbformat python builders/build_showcase_01_data_foundations.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        showcase/showcase_01_data_foundations.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

"Data in, honestly" — the data layer of the framework, end to end: the MFF
format and the fluent MFFLoader, the bundled example datasets with their sealed
answer keys, the synthetic worlds as a first-class feature (response_fn,
Scenario.slice, geo panels), pre-fit EDA data-quality checks graded against
planted defects, the Data Studio's replayable clean pipeline, the Excel config
round trip, scaling parameters, and the native non-MFF dataset path. Every
number is computed in-notebook; no model is fit here.
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

MFF_BUILD = r"""
from mmm_framework.synth import dgp, mff

world = dgp.make_clean(seed=7, n_weeks=104)
frame = mff.scenario_to_mff(world)
CLEAN_CSV = ART / "showcase01_clean.csv"
frame.to_csv(CLEAN_CSV, index=False)

print(f"{len(frame)} rows = {frame['VariableName'].nunique()} variables x "
      f"{frame['Period'].nunique()} weeks\n")
print(frame.head(6).to_string(index=False))
"""

MFF_CHART = r"""
ROLES = {"Sales": ("KPI", INK)}
ROLES.update({c: ("media", PALETTE[c]) for c in world.channels})
ROLES.update({c: ("control", MUTED) for c in world.controls.columns})

wide = frame.pivot_table(index="VariableName", columns="Period",
                         values="VariableValue", sort=False)
order = ["Sales", *world.channels, *world.controls.columns]
wide = wide.loc[order]
z = wide.sub(wide.min(axis=1), axis=0).div(
    wide.max(axis=1) - wide.min(axis=1), axis=0)

fig = go.Figure(go.Heatmap(
    z=z.to_numpy(), x=list(wide.columns), y=order,
    colorscale=[[0, "#f4f5f7"], [1, INK]], showscale=False,
    hovertemplate="%{y} · %{x}<br>row-normalized %{z:.2f}<extra></extra>"))
fig.update_yaxes(autorange="reversed",
                 ticktext=[f"{v}  ({ROLES[v][0]})" for v in order],
                 tickvals=order)
fig.update_xaxes(title="Period (weekly)")
style(fig, 360, "An MFF is one long table: a block of (period, value) rows per variable")
fig.show()
"""

LOADER = r"""
from mmm_framework import MFFLoader
from mmm_framework.config import (
    ControlVariableConfig,
    DimensionType,
    KPIConfig,
    MFFConfig,
    create_national_media_config,
)

mff_config = MFFConfig(
    kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
    media_channels=[create_national_media_config(c) for c in world.channels],
    controls=[ControlVariableConfig(name=c, dimensions=[DimensionType.PERIOD])
              for c in world.controls.columns],
)

panel = MFFLoader(mff_config).load(str(CLEAN_CSV)).build_panel()
print(f"PanelDataset: {panel.n_obs} obs, {panel.n_channels} channels, "
      f"controls={list(panel.coords.controls)}")
print(f"index: {type(panel.index).__name__} "
      f"({panel.index.min().date()} .. {panel.index.max().date()})")
"""

EXAMPLE = r"""
from mmm_framework import load_example
from mmm_framework.datasets import list_examples, load_example_answer_key

for name, desc in list_examples().items():
    print(f"{name:>9}: {desc}")

example = load_example("national")
key = load_example_answer_key("national")
print(f"\nloaded 'national': {example.n_obs} obs, {example.n_channels} channels")

spend_tot = example.X_media.sum()
chans = list(spend_tot.index)
true_roas = [key["true_roas"][c] for c in chans]

fig = make_subplots(rows=1, cols=2, subplot_titles=(
    "Total spend by channel", "TRUE causal ROAS (sealed answer key)"))
colors = [PALETTE.get(c, MUTED) for c in chans]
fig.add_trace(go.Bar(x=chans, y=spend_tot.to_numpy(), marker_color=colors,
                     showlegend=False), row=1, col=1)
fig.add_trace(go.Bar(x=chans, y=true_roas, marker_color=colors,
                     showlegend=False), row=1, col=2)
fig.add_hline(y=1.0, line=dict(color=BAD, dash="dot"), row=1, col=2)
fig.update_yaxes(title="spend", row=1, col=1)
fig.update_yaxes(title="true ROAS", row=1, col=2)
style(fig, 380, "The bundled example ships graded homework: spend in, sealed truth beside it")
fig.show()
"""

RESPONSE_FN = r"""
spend_arr = world.spend.to_numpy(float)
scales = np.linspace(0.0, 2.0, 21)

fig = go.Figure()
for i, c in enumerate(world.channels):
    zeroed = spend_arr.copy(); zeroed[:, i] = 0.0
    base = world.response_fn(zeroed).sum()
    incr = []
    for s in scales:
        scaled = spend_arr.copy(); scaled[:, i] = spend_arr[:, i] * s
        incr.append(world.response_fn(scaled).sum() - base)
    fig.add_trace(go.Scatter(x=scales, y=incr, mode="lines", name=c,
                             line=dict(color=PALETTE[c], width=2.5)))
    fig.add_trace(go.Scatter(x=[1.0], y=[incr[10]], mode="markers",
                             marker=dict(color=PALETTE[c], size=10),
                             showlegend=False, hoverinfo="skip"))
fig.add_vline(x=1.0, line=dict(color=MUTED, dash="dot"))
fig.update_xaxes(title="spend multiplier (1.0 = as observed)")
fig.update_yaxes(title="true incremental Sales (total)")
style(fig, 400, "response_fn is the sealed answer key: the world's true curve, queryable at any spend")
fig.show()

print("true_contribution (from the same function, at multiplier 1.0):")
print(world.true_contribution.round(1).to_string())
"""

SLICE = r"""
windows = [(0, 26), (26, 52), (52, 78), (78, 104)]
labels = [f"wk {a}-{b}" for a, b in windows]
recomputed = {c: [] for c in world.channels}
for a, b in windows:
    win = world.slice(a, b)     # truth RECOMPUTED on the window
    for c in world.channels:
        recomputed[c].append(win.true_contribution[c])

fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=(
    "naive: total x window share", "Scenario.slice: recomputed"))
for c in ("TV", "Social"):
    naive = [world.true_contribution[c] * (b - a) / 104 for a, b in windows]
    fig.add_trace(go.Bar(x=labels, y=naive, name=c, marker_color=PALETTE[c],
                         opacity=0.55, showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=recomputed[c], name=c,
                         marker_color=PALETTE[c]), row=1, col=2)
fig.update_yaxes(title="true incremental Sales", row=1, col=1)
style(fig, 380, "A window's truth is recomputed through the response, never truncated pro-rata",
      barmode="group")
fig.show()

worst = max(world.channels, key=lambda c: max(
    abs(recomputed[c][i] - world.true_contribution[c] * (b - a) / 104)
    / recomputed[c][i]
    for i, (a, b) in enumerate(windows)))
errs = [abs(recomputed[worst][i] - world.true_contribution[worst]*(b-a)/104)
        / recomputed[worst][i] for i, (a, b) in enumerate(windows)]
print(f"pro-rata truncation is off by up to {max(errs):.1%} for {worst} — "
      "carryover and saturation make the response non-proportional in time")
"""

GEO = r"""
from mmm_framework.synth import dgp_geo

geo_world = dgp_geo.make_geo_clean(seed=20, n_weeks=104)
print(f"{len(geo_world.geos)} geos x {len(geo_world.weeks)} weeks; "
      f"index levels: {list(geo_world.spend.index.names)}")

fig = make_subplots(rows=2, cols=2, subplot_titles=geo_world.geos,
                    shared_xaxes=True, vertical_spacing=0.12)
for i, g in enumerate(geo_world.geos):
    r, c = divmod(i, 2)
    y_g = geo_world.y.xs(g, level="Geography")
    tv_g = geo_world.spend["TV"].xs(g, level="Geography")
    fig.add_trace(go.Bar(x=geo_world.weeks, y=tv_g, marker_color=PALETTE["TV"],
                         opacity=0.45, name="TV spend", showlegend=(i == 0)),
                  row=r + 1, col=c + 1)
    fig.add_trace(go.Scatter(x=geo_world.weeks, y=y_g, mode="lines",
                             line=dict(color=INK, width=1.6), name="Sales",
                             showlegend=(i == 0)), row=r + 1, col=c + 1)
style(fig, 520, "Geo worlds plant per-market truth: shared response, different scale per DMA")
fig.show()

print("\ntrue contribution by geo (answer key rows per market):")
print(geo_world.true_contribution_by_geo.round(0).to_string())
"""

EDA_PROFILE = r"""
from mmm_framework.eda import (
    detect_outliers,
    load_eda_panel_from_df,
    profile_panel,
    recommend_treatments,
    validate_dataset,
)

messy = dgp.make_mixed_data_errors(seed=21, n_weeks=104)
messy_long = mff.scenario_to_mff(messy)
spec = {"kpi": "Sales",
        "media_channels": [{"name": c} for c in messy.channels],
        "control_variables": [{"name": c} for c in messy.controls.columns]}
eda_panel = load_eda_panel_from_df(messy_long, spec)

prof = profile_panel(eda_panel)
print(prof[["variable", "role", "n", "missing_pct", "zero_pct"]].to_string(index=False))

# Structural validation fires on structurally broken data. This messy world is
# structurally fine (its defects are VALUES, not structure), so break a copy:
broken = messy_long.copy()
i = broken[broken["VariableName"] == "TV"].index[5]
broken.loc[i, "VariableValue"] = -500.0
report = validate_dataset(load_eda_panel_from_df(broken, spec))
print("\n" + report.summary())
"""

EDA_OUTLIERS = r"""
out = detect_outliers(eda_panel)
flags = out.flags_frame()
print(flags[["variable", "period", "kind", "score", "methods"]].to_string(index=False))

planted = messy.notes["errors"]
period_of = lambda wk: str(messy.weeks[wk].date())

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                    subplot_titles=("TV — planted x10 decimal shift",
                                    "Search — planted missed-load zero",
                                    "Social — planted x2 double-count"))
for row, ch in enumerate(("TV", "Search", "Social"), start=1):
    s = eda_panel.series(ch)
    fig.add_trace(go.Scatter(x=s.index, y=s, mode="lines", name=ch,
                             line=dict(color=PALETTE[ch], width=1.6),
                             showlegend=False), row=row, col=1)
    wk = planted[ch]["week"]
    fig.add_trace(go.Scatter(x=[messy.weeks[wk]], y=[s.iloc[wk]], mode="markers",
                             marker=dict(color=TRUTH, symbol="circle-open", size=16,
                                         line=dict(width=2.5)),
                             name="planted defect", showlegend=(row == 1)),
                  row=row, col=1)
    hits = flags[flags.variable == ch]
    fig.add_trace(go.Scatter(x=pd.to_datetime(hits.period), y=hits.value,
                             mode="markers",
                             marker=dict(color=BAD, symbol="x", size=11),
                             name="detector flag", showlegend=(row == 1)),
                  row=row, col=1)
style(fig, 560, "Two of three planted defects are flagged; the x2 double-count is a stated detection limit")
fig.show()
"""

EDA_ACTIONS = r"""
actions = recommend_treatments(eda_panel, out.flags)
for a in actions[:5]:
    print(f"{a.action_id:<28} [{a.strategy}] {a.rationale[:95]}")
print(f"... {len(actions)} recommended action(s) total")

kind_counts = flags.groupby(["variable", "kind"]).size().reset_index(name="n")
kind_color = {"isolated_spike": BAD, "isolated_drop": GOLD, "level_shift": MUTED,
              "heavy_tail_member": "#7a86b8", "kpi_shock": INK}
fig = go.Figure()
for kind, sub in kind_counts.groupby("kind"):
    fig.add_trace(go.Bar(x=sub.variable, y=sub.n, name=kind,
                         marker_color=kind_color.get(kind, MUTED)))
fig.update_yaxes(title="flags", dtick=2)
style(fig, 340, "Flags come typed: a spend spike is not a KPI shock, and the fixes differ",
      barmode="stack")
fig.show()
"""

STUDIO = r"""
from mmm_framework.data_studio.transforms import apply_pipeline

tv_wk = planted["TV"]["week"]
se_wk = planted["Search"]["week"]
tv_cap = float(messy.spend["TV"].drop(messy.spend.index[tv_wk]).max())
promo_periods = [str(messy.weeks[w].date()) for w in messy.notes["promo_weeks"]]

steps = [
    {"op": "winsorize", "column": "TV", "cap_value": tv_cap,
     "periods": [period_of(tv_wk)]},
    {"op": "impute", "column": "Search",
     "value": float(messy.notes["true_search_spend"].iloc[se_wk]),
     "periods": [period_of(se_wk)]},
    {"op": "event_dummy", "name": "promo_event", "periods": promo_periods},
]
result = apply_pipeline(messy_long, steps, roles={"Sales": "kpi"})
print(f"cleaned: {len(result.df)} rows (was {len(messy_long)}), "
      f"warnings={result.warnings or 'none'}")
print(f"new columns/roles: {result.roles.get('promo_event')!r} role added for promo_event")

def series_of(df, name):
    sub = df[df["VariableName"] == name].sort_values("Period")
    return pd.to_datetime(sub["Period"]), sub["VariableValue"].to_numpy(float)

fig = make_subplots(rows=1, cols=2, subplot_titles=(
    "TV — x10 spike winsorized to the honest max",
    "Search — missed load imputed from the true series"))
for col, ch in enumerate(("TV", "Search"), start=1):
    x0, raw = series_of(messy_long, ch)
    x1, fixed = series_of(result.df, ch)
    fig.add_trace(go.Scatter(x=x0, y=raw, mode="lines", name="raw",
                             line=dict(color=BAD, width=1.3),
                             showlegend=(col == 1)), row=1, col=col)
    fig.add_trace(go.Scatter(x=x1, y=fixed, mode="lines", name="cleaned",
                             line=dict(color=GOOD, width=1.8),
                             showlegend=(col == 1)), row=1, col=col)
style(fig, 380, "The pipeline is a replayable list of steps — raw data is never mutated")
fig.show()
"""

EXCEL = r"""
from mmm_framework.excel_config import TemplateGenerator, TemplateParser, discover_mff

discovery = discover_mff(str(CLEAN_CSV))
for v in discovery.variables:
    print(f"{v.name:>8}: {v.role.value:<8} dims={v.dimensions}")

xlsx = TemplateGenerator.from_mff(str(CLEAN_CSV),
                                  output_path=ART / "showcase01_config.xlsx")
mff_cfg, model_cfg, trend_cfg = TemplateParser.parse(xlsx)
print(f"\nround trip: kpi={mff_cfg.kpi.name!r}, "
      f"media={[m.name for m in mff_cfg.media_channels]}, "
      f"controls={[c.name for c in mff_cfg.controls]}")
print(f"model: {model_cfg.inference_method}, trend: {trend_cfg.type}")
"""

SCALING = r"""
from mmm_framework.data_preparation import DataPreparator

prepared = DataPreparator(panel, adstock_alphas=[0.0, 0.5, 0.8]).prepare()
sp = prepared.scaling_params
print(f"y: standardized with mean={sp.y_mean:.1f}, std={sp.y_std:.1f}")
print("media: each channel normalized by its max adstocked value:")
for ch, mx in sp.media_max.items():
    print(f"  {ch:>8}: /{mx:.1f}")
print("\nround trip:", sp.to_dict().keys(), "->",
      type(sp.from_dict(sp.to_dict())).__name__)
"""

NATIVE = r"""
from mmm_framework.config import DatasetRole, DatasetSchema
from mmm_framework.config.dataset import RoleBinding
from mmm_framework.dataset_loader import load_dataset

rng = np.random.default_rng(3)
factor = rng.normal(size=120)
survey = pd.DataFrame(
    {f"q{i + 1}": 0.8 * factor + 0.4 * rng.normal(size=120) for i in range(4)})

schema = DatasetSchema(bindings=[
    RoleBinding(name=c, role=DatasetRole.INDICATOR) for c in survey.columns])
ds = load_dataset(survey, schema)
print(ds.summary())
print(f"indicator columns: {ds.columns_for(DatasetRole.INDICATOR)}")
print(f"cross-sectional coords: n_periods={ds.coords.n_periods}, "
      f"geo={ds.coords.has_geo}")
"""

CELLS = [
    md(r"""
# Data in, honestly

Every claim an MMM makes inherits the sins of its input data. This notebook
tours the framework's data layer end to end — how data gets *in*, how quality
problems get *found*, and how the framework grades itself with worlds where the
truth is known:

1. the **MFF format** (one long table) and the fluent **`MFFLoader`**
2. **`load_example`** — bundled datasets with sealed answer keys
3. the **synthetic worlds** as a feature: `response_fn`, `Scenario.slice`,
   geo panels
4. **pre-fit EDA**: validation, profiling, outlier detection graded against
   planted defects
5. the **Data Studio** replayable clean pipeline
6. the **Excel config** round trip
7. cloud **integrations and storage** (prose — network-dependent)
8. **scaling parameters** and the **native non-MFF dataset** path

No model is fit here. This is the part of the workflow where most real MMM
projects quietly go wrong, so it gets its own front-door tour.
"""),
    code(SETUP),
    md(r"""
## 1 · The MFF — one long table for everything

The Master Flat File exists because wide spreadsheets break the moment
variables live on different grains: national TV, DMA-level Search, a weekly
KPI. MFF stores one row per (variable, period, dimensions) cell, so each
variable carries its **own** dimensionality and the loader aligns them —
no manual reshaping, no accidental broadcast.

We build one from a synthetic world rather than typing fake numbers, because
that world will hand us a sealed answer key later.
"""),
    code(MFF_BUILD),
    code(MFF_CHART),
    md(r"""
## 2 · `MFFLoader` — the fluent path from CSV to panel

The loader is configuration-first: an `MFFConfig` declares the KPI, the media
channels, and the controls with their dimensions, and
`MFFLoader(config).load(csv).build_panel()` does validation, date parsing,
numeric coercion (a `"$1,234"` cell fails loudly, and early), dimension
alignment, and produces the `PanelDataset` every model in the framework takes.
"""),
    code(LOADER),
    md(r"""
## 3 · `load_example` — a real dataset in one line

Two curated datasets ship inside the package so a first fit needs zero data
wrangling. Both come from the synthetic worlds, which means both ship a
**sealed answer key**: each channel's true causal contribution and ROAS. An
example dataset you can grade yourself against is worth ten you cannot.
"""),
    code(EXAMPLE),
    md(r"""
## 4 · The synthetic worlds are a feature, not a test fixture

`mmm_framework.synth` builds *worlds* — spend, KPI, controls, and the full
causal ground truth, exported together. Beyond the clean control world there
are ~20 violation worlds (unobserved confounding, reverse causality, data
errors, trend breaks ...) used across the stress and causal notebook series.

Two design choices matter most:

**`response_fn` is the answer key itself.** Every world carries its structural
response, `spend (n, C) -> mean (n,)`. The truth is not a stored number; it is
a function you can query at any counterfactual spend — the same estimand the
model will be asked for.
"""),
    code(RESPONSE_FN),
    md(r"""
**`Scenario.slice` recomputes truth instead of truncating it.** Total
contributions are whole-window numbers; carving out a sub-window by
proportional scaling is simply wrong under carryover and saturation. `slice`
re-evaluates the response on the full spend history and sums the window — so a
backtest window is graded against its *actual* truth.
"""),
    code(SLICE),
    md(r"""
**Geo worlds plant per-market truth.** `synth.dgp_geo` builds panel worlds —
several DMAs sharing a response family but differing in scale (and, in the
heterogeneous world, in per-geo effectiveness). The answer key has one row per
market, which is what lets the geo-hierarchy notebooks grade partial pooling.
"""),
    code(GEO),
    md(r"""
## 5 · Pre-fit EDA — find the defects before the model launders them

`mmm_framework.eda` is the pre-fit sibling of the post-fit validation suite:
it profiles, validates, and outlier-scans the raw dataset *before*
`BayesianMMM` sees it. The demo below is honest in both directions: we use
`make_mixed_data_errors`, a world with **planted, documented recording
defects** — a x10 decimal shift on TV, a missed-load zero on Search, a x2
double-count on Social — so detector hits and misses can be graded, not
eyeballed.
"""),
    code(EDA_PROFILE),
    code(EDA_OUTLIERS),
    md(r"""
The miss is deliberate content: a x2 double-count on a heavy flight week is
statistically indistinguishable from a real heavy flight week. The world's
docstring calls it *"a KNOWN detection limit, kept here so the grading is
honest about it"* — a detector that claimed to catch it would be lying.

Note also what the detector does **not** recommend deleting: the KPI shocks
are real promo demand, and their recommended treatment is an event dummy
(model it), never a correction (erase it).
"""),
    code(EDA_ACTIONS),
    md(r"""
## 6 · Data Studio transforms — cleaning as a replayable pipeline

The Data Studio's clean step is a list of step dicts replayed over a copy of
the raw frame: `result = f(raw, steps)`, deterministic and idempotent, with no
cached intermediate to go stale. The same ops work on wide CSVs and MFF-long
frames. Structurally invalid steps raise; data-level failures (a column just
renamed away) are recorded as warnings so a mid-edit pipeline still previews.

Here we apply exactly what the EDA flags recommended.
"""),
    code(STUDIO),
    md(r"""
## 7 · Excel config — the analyst on-ramp

Not every model owner writes Python. `excel_config` discovers the variables in
an MFF, generates a pre-filled workbook (roles, dimensions, media settings,
priors on an Advanced sheet), and parses the edited workbook back into the
same `MFFConfig` / `ModelConfig` / `TrendConfig` objects the code path uses —
one configuration surface, two front doors.
"""),
    code(EXCEL),
    md(r"""
## 8 · Integrations, storage, lineage — the cloud edges (not run here)

These are network-dependent, so this section shows the real call shapes
without executing them.

**Connectors** pull tabular data into pandas; the DataFrame then feeds the
MFF loader unchanged. Google Cloud SDKs are optional
(`pip install 'mmm-framework[gcp]'`) and imported lazily:

```python
from mmm_framework.integrations import build_data_source

src = build_data_source("bigquery", {"project": "acme", "dataset": "mmm"})
df = src.read_dataframe(query="SELECT * FROM mmm.weekly_spend")
panel = MFFLoader(mff_config).load(df).build_panel()
```

**Object storage** abstracts where artifacts live — local filesystem in a
notebook, S3-compatible stores in the hosted posture:

```python
from mmm_framework.storage import get_object_store

store = get_object_store()          # LocalObjectStore or S3ObjectStore via env
```

**Lineage** is a property of the pieces you have already seen rather than a
separate system: the Data Studio pipeline *is* the provenance of a cleaned
dataset (raw + steps reproduce it exactly), the EDA outlier report records the
`dataset_path` and `dataset_mtime` it was computed against so a stale report
is detectable, and a fitted model's serialized bundle carries its config and
scaling parameters (next section) so no downstream number silently re-derives
its basis.
"""),
    md(r"""
## 9 · Scaling — the parameters travel with the model

Models fit on standardized data: the KPI is z-scored, each channel is
normalized by its max adstocked value, controls are standardized. Every one of
those choices must be **inverted** to report a dollar number, so
`ScalingParameters` is a first-class, serializable object — it rides inside
the model bundle, and predictions on new data reuse the *training* scale
rather than quietly re-deriving a new one.
"""),
    code(SCALING),
    md(r"""
## 10 · The native, non-MFF path

Some datasets in this framework are not marketing panels at all — a CFA
indicator matrix, a survey, an LCA table have no KPI/media/control roles to
declare. `load_dataset` takes a wide, role-tagged table plus a
`DatasetSchema` and produces a `Dataset` directly; bespoke model families
extend the schema with their own roles. This is the front door the non-MMM
families (CFA / LCA) and Model Garden customs use.
"""),
    code(NATIVE),
    md(r"""
## Where to go deeper

- `technical-docs/engineering-notes.md` — *Data Studio*, *Non-MMM model
  families*, and *Impression-/click-measured media* (what happens when the
  modeled variable is impressions, not dollars).
- `technical-docs/non-mmm-families.md` — the native dataset path in anger
  (CFA / LCA).
- `nbs/stress_00..06` — the violation worlds from section 4 used to
  pressure-test the model, one assumption at a time.
- `nbs/causal/` (00–10) — why the answer key is a *counterfactual* and what
  that buys you.
- `nbs/lifecycle_00..06` — the full T0→T5 measurement loop these datasets
  feed.
- `nbs/workshop_03_first_mmm.ipynb` — the gentlest possible first fit on
  `load_example` data.
- `docs/` site — Getting Started and the Data Studio guide pages mirror this
  notebook for non-notebook readers.
"""),
]


def main() -> None:
    nb = new_notebook(cells=CELLS)
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    out = "showcase/showcase_01_data_foundations.ipynb"
    with open(out, "w") as fh:
        nbformat.write(nb, fh)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
