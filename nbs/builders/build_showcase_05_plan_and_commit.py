"""Author showcase/showcase_05_plan_and_commit.ipynb (run from ``nbs/``).

    uv run --with nbformat python builders/build_showcase_05_plan_and_commit.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        showcase/showcase_05_plan_and_commit.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

Feature-showcase notebook 05: "A number the CFO can commit to". The full v1.4
plan-and-commit loop on one small NUTS-lite fit — finance valuation with
provenance and refusals, bridge-line vocabulary, decomposition closure, the
budget optimizer and frontier, promo-depth decision arms, payback horizons,
calendar + flighting, forecast-under-plan with caveats first, the plan of
record (commit, hash chain, reproduce to 1e-9, tamper refusals), mid-flight
pacing, and the variance-to-plan bridge. Every number is computed in-notebook.
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
PALETTE = {"TV": "#4464ad", "Search": "#c9962e", "Social": "#3d7a5c",
           "Display": "#b4552d", "Promo depth (Promo)": "#7a4bb3"}

def style(fig, height=380, title=None, **kw):
    fig.update_layout(height=height, title=title, margin=dict(t=64, l=64, r=30, b=52),
                      font=dict(size=12), **kw)
    return fig

ART = Path.cwd().parent / "artifacts" / "showcase"
ART.mkdir(parents=True, exist_ok=True)
print("Setup ready.")
"""

WORLD_FIT = r"""
from mmm_framework.synth import dgp, mff
from mmm_framework.agents.fitting import build_model

N_WEEKS, N_TRAIN, N_PLAN = 117, 104, 13
world = dgp.make_clean(seed=7, n_weeks=N_WEEKS)
train = world.slice(0, N_TRAIN)
CHANNELS = list(world.channels)

DATA = ART / "commit_world.csv"
mff.scenario_to_mff(train).to_csv(DATA, index=False)

SPEC = {
    "kpi": "Sales",
    "kpi_level": "national",
    "media_channels": [
        {"name": c, "adstock": {"type": "geometric"}, "saturation": {"type": "hill"}}
        for c in CHANNELS
    ],
    "control_variables": [{"name": c} for c in train.controls.columns],
    # Linear trend on purpose: it extrapolates in closed form, so the commit
    # gate's held-flat-trend refusal has nothing to object to (section 7).
    "trend": {"type": "linear"},
}
t0 = time.time()
mmm = build_model(SPEC, str(DATA))
mmm.fit(draws=400, tune=400, chains=2, random_seed=42)
print(f"NUTS-lite fit (2 chains x 400 draws) in {time.time()-t0:.0f}s")
"""

FINANCE = r"""
from mmm_framework.finance import KpiValuation, UnresolvedValueError, kpi_to_dollars
from pydantic import ValidationError

# Nothing set -> UNRESOLVED, a first-class state. Never a silent 1.0.
rv0 = kpi_to_dollars()
print("nothing set        :", rv0.describe())
try:
    rv0.require("Fund-to-breakeven allocation")
except UnresolvedValueError as e:
    print("require() refuses  :", str(e)[:120], "...")

# The precedence chain, each answer carrying its source.
print("\npreferences        :",
      kpi_to_dollars(preferences={"economics": {"kind": "revenue",
                                                "gross_margin": 0.42}}).describe())
print("spec block         :",
      kpi_to_dollars(spec={"valuation": {"kind": "units", "gross_margin": 0.5,
                                         "price": 5.0}}).describe())
RV = kpi_to_dollars(
    override={"kind": "units", "gross_margin": 0.5, "price": 5.0},
    preferences={"economics": {"kind": "revenue", "gross_margin": 0.42}},
)
print("override wins      :", RV.describe())
VALUE = RV.require("every dollar number in this notebook")

# The 40x refusal: a user typing 40 to mean 40% used to multiply every profit
# number by 40 — margins are fractions, and the type now enforces it.
try:
    KpiValuation(gross_margin=40)
except ValidationError as e:
    print("\ngross_margin=40    : REFUSED —", str(e).splitlines()[2].strip())
"""

LINES = r"""
from mmm_framework.finance import BridgeLine, provenance_of

# The four provenances a bridge line can carry, each with its own rules.
ok = BridgeLine("Media (modelled)", 4200.0, provenance="modelled",
                lower=3800.0, upper=4600.0, interval_mass=0.90, basis="components")
print(ok.describe())
print(BridgeLine("Observed KPI", 20000.0, provenance="observed").describe())
print(BridgeLine("Gross-to-net true-up", -180.0, provenance="supplied",
                 source_note="ERP export 2027-05-02, finance-approved").describe())

# The refusals are in the type, not in reviewer discipline.
for label, kwargs in [
    ("interval without stated mass",
     dict(provenance="modelled", lower=1.0, upper=2.0)),
    ("SUPPLIED without a source note", dict(provenance="supplied")),
    ("SUPPLIED wearing an interval",
     dict(provenance="supplied", source_note="ERP", lower=1.0, upper=2.0,
          interval_mass=0.9)),
]:
    try:
        BridgeLine("x", 1.5, **kwargs)
    except ValueError as e:
        print(f"\nREFUSED ({label}):\n  {str(e)[:120]}...")

# Anything unstated reads as the pessimistic case: a leftover that absorbs
# the model residual inside a bar labelled something else.
print("\nunstated provenance reads as:", provenance_of("mystery").value)
"""

CLOSURE = r"""
from mmm_framework.finance import decomposition_closure

facts = decomposition_closure(mmm, random_seed=0)
print(f"closes: {facts.closes}   basis: {facts.basis}   "
      f"specification: {facts.specification}")
print(facts.residual_reading(), "\n")
for ln in facts.lines:
    print(" ", ln.describe())

vals = [ln.value for ln in facts.lines]
labels = [f"{ln.name}<br><i>{ln.provenance.value}</i>" for ln in facts.lines]
fig = go.Figure(go.Waterfall(
    orientation="v",
    measure=["relative"] * len(vals) + ["total"],
    x=labels + ["Observed KPI<br><i>observed</i>"],
    y=vals + [0.0],
    connector={"line": {"color": MUTED, "width": 1}},
    increasing={"marker": {"color": GOOD}},
    decreasing={"marker": {"color": BAD}},
    totals={"marker": {"color": INK}},
))
fig.update_yaxes(title="KPI units (training window)")
style(fig, 400, "The decomposition closes because the residual is a named line")
fig.show()
"""

OPT = r"""
from mmm_framework.planning import compute_response_curves, optimize_budget
from mmm_framework.finance import UnresolvedValueError

t0 = time.time()
curves = compute_response_curves(mmm, max_draws=150, random_seed=42)
res = optimize_budget(curves=curves, min_multiplier=0.7, max_multiplier=1.3,
                      random_seed=42)
print(f"curves + fixed reallocation in {time.time()-t0:.0f}s   "
      f"(objective: {res.objective_label}, mode: {res.mode})\n")

ALLOC_COLS = ["channel", "current_spend", "optimal_spend", "change_pct",
              "optimal_share_pct"]
print(res.table[ALLOC_COLS].round(1).to_string(index=False))
print(f"\nexpected uplift : {res.expected_uplift:,.0f} KPI units "
      f"[{res.uplift_hdi[0]:,.0f}, {res.uplift_hdi[1]:,.0f}]   "
      f"P(uplift>0) = {res.prob_positive_uplift:.0%}")
print(f"expected regret of committing to this single plan: "
      f"{res.expected_regret:,.0f} KPI units")

# The refusal: mode='free' trades KPI against dollars, so it cannot run on an
# assumed $1/KPI. Fixed reallocation needs no valuation and never refuses.
try:
    optimize_budget(curves=curves, mode="free")
except UnresolvedValueError as e:
    print(f"\nmode='free' without a valuation -> {type(e).__name__}:")
    print("  " + str(e)[:150] + "...")

res_free = optimize_budget(curves=curves, mode="free",
                           value_per_kpi=VALUE, value_source=RV.source)
print(f"\nmode='free' with the resolved valuation: the total becomes an OUTPUT "
      f"— fund-to-breakeven total {res_free.total_budget:,.0f} "
      f"(current {float(curves.base_spend.sum()):,.0f}; "
      f"value {VALUE:.2f}/KPI from source '{res_free.value_source}')")
"""

OPT_CHART = r"""
fig = go.Figure()
fig.add_trace(go.Bar(x=res.table["channel"], y=res.table["current_spend"],
                     name="current", marker_color=MUTED, opacity=0.7))
fig.add_trace(go.Bar(x=res.table["channel"], y=res.table["optimal_spend"],
                     name="recommended (±30% box)",
                     marker_color=[PALETTE[c] for c in res.table["channel"]]))
for _, r_ in res.table.iterrows():
    fig.add_annotation(x=r_["channel"], y=max(r_["current_spend"], r_["optimal_spend"]),
                       text=f"{r_['change_pct']:+.0f}%", showarrow=False, yshift=14,
                       font=dict(size=12, color=INK))
fig.update_yaxes(title="training-window spend")
style(fig, 380, "A reallocation the model can defend — bounded to ±30% of observed spend",
      barmode="group")
fig.show()
"""

FRONTIER = r"""
from mmm_framework.planning import budget_frontier

t0 = time.time()
fr = budget_frontier(curves=curves, n_points=9, random_seed=42)
print(f"frontier ({len(fr.points)} budgets) in {time.time()-t0:.0f}s")

bud = [p.total_budget for p in fr.points]
ret = [p.expected_return for p in fr.points]
p5 = [p.return_p5 for p in fr.points]
p95 = [p.return_p95 for p in fr.points]
mroi_dollar = [p.marginal_roi * VALUE for p in fr.points]

fig = make_subplots(rows=1, cols=2, subplot_titles=(
    "optimized media KPI vs budget (90% band)",
    "the next dollar's return, in dollars"))
fig.add_trace(go.Scatter(x=bud + bud[::-1], y=p95 + p5[::-1], fill="toself",
                         fillcolor="rgba(68,100,173,0.18)",
                         line=dict(width=0), showlegend=False,
                         hoverinfo="skip"), row=1, col=1)
fig.add_trace(go.Scatter(x=bud, y=ret, mode="lines+markers", name="frontier",
                         line=dict(color=PALETTE["TV"], width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=[fr.current_total], y=[fr.current_return],
                         mode="markers", name="current plan",
                         marker=dict(color=BAD, size=13, symbol="diamond")),
              row=1, col=1)
fig.add_trace(go.Scatter(x=bud, y=mroi_dollar, mode="lines+markers",
                         name="marginal $ per $", line=dict(color=GOLD, width=2)),
              row=1, col=2)
fig.add_hline(y=1.0, line=dict(color=BAD, dash="dot"), row=1, col=2,
              annotation_text="breakeven ($1 back per $1)")
fig.update_xaxes(title="total budget", row=1, col=1)
fig.update_xaxes(title="total budget", row=1, col=2)
fig.update_yaxes(title="media KPI (window)", row=1, col=1)
fig.update_yaxes(title="marginal $ / $", row=1, col=2)
style(fig, 400, "More budget buys less at the margin — the frontier prices where to stop")
fig.show()
print("(the marginal-$ panel exists because a valuation resolved; "
      "in KPI units the frontier needs none)")
"""

PROMO_FIT = r"""
promo = dgp.build("promo_and_media")
pt = promo.notes
PDATA = ART / "promo_world.csv"
mff.scenario_to_mff(promo).to_csv(PDATA, index=False)

PSPEC = {
    "kpi": "Sales",
    "kpi_level": "national",
    "media_channels": [
        {"name": c, "adstock": {"type": "geometric"},
         "saturation": {"type": "logistic"}}
        for c in promo.channels
    ],
    "control_variables": [{"name": c} for c in promo.controls.columns],
    # The lever declarations: Price and Promo leave the linear control block
    # and get their own transforms, priors and cost bases.
    "price": {"variable": "Price", "reference": "median"},
    "promotions": [{"variable": "Promo", "adstock_lmax": 8}],
}
t0 = time.time()
pmmm = build_model(PSPEC, str(PDATA))
pmmm.fit(draws=400, tune=400, chains=2, random_seed=42)
print(f"promo-world NUTS-lite fit in {time.time()-t0:.0f}s | levers: {pmmm.lever_names}")
"""

PROMO_ROI = r"""
from mmm_framework.planning import promo_roi

# Refusal 1: no valuation. Margin-back-per-margin-given-away is a dollar ratio.
try:
    promo_roi(pmmm, "Promo", unit_cost=pt["promo_unit_cost"])
except UnresolvedValueError as e:
    print("without a valuation:", type(e).__name__, "—", str(e)[:90], "...\n")

# Refusals 2+3: the COST basis is refusable too. A 0/1 event flag has no
# depth (no discount cost exists); a column outside [0,1] has unknown units.
class _Stub:
    lever_names = ["Promo"]
    X_levers_raw = None

for label, series in [("0/1 event flag", np.array([0.0, 1.0, 0.0, 1.0, 1.0])),
                      ("unknown units", np.array([0.0, 12.0, 0.0, 40.0]))]:
    stub = _Stub(); stub.X_levers_raw = series[:, None]
    try:
        promo_roi(stub, "Promo", unit_cost=1.0, value_per_kpi=1.0)
    except ValueError as e:
        print(f"REFUSED ({label}): {str(e)[:105]}...")

# With the world's frozen economics resolved through the same finance chain:
RVP = kpi_to_dollars(override={"kind": "units",
                               "gross_margin": pt["gross_margin"],
                               "price": pt["price_reference"]})
r = promo_roi(pmmm, "Promo", unit_cost=pt["promo_unit_cost"],
              value_per_kpi=RVP.require("Promo ROI"), value_source="world economics")
print(f"\npromo ROI: {r.roi_mean:.2f} margin-$ back per margin-$ given away "
      f"[{r.roi_lower:.2f}, {r.roi_upper:.2f}] at {r.interval_mass:.0%} mass")
print(f"lift {r.lift_kpi_mean:,.0f} KPI units | cost basis: avg depth "
      f"{r.avg_depth:.3f} x unit cost {r.unit_cost:,.0f} = {r.realized_cost:,.0f}")
print("caveat:", r.caveats[-1][:140], "...")
"""

PROMO_CHART = r"""
from mmm_framework.planning import build_arm_curves

t0 = time.time()
ac = build_arm_curves(pmmm, promo_var="Promo", unit_cost=pt["promo_unit_cost"],
                      max_draws=100, random_seed=42)
print(f"arm curves in {time.time()-t0:.0f}s — arms:",
      [f"{a.name} ({a.kind})" for a in ac.arms])

pvalue = RVP.require("arm value curves")
mean_c = ac.mean_curves()          # (n_arms, G) KPI units
grid = ac.spend_grid               # (n_arms, G) realized cost
fig = go.Figure()
for i, arm in enumerate(ac.arms):
    is_promo = arm.kind == "promo"
    fig.add_trace(go.Scatter(
        x=grid[i], y=mean_c[i] * pvalue, mode="lines",
        name=f"{arm.name} ({arm.kind})",
        line=dict(color=PALETTE.get(arm.name, "#7a4bb3"),
                  width=4 if is_promo else 2,
                  dash=None if is_promo else "dot"),
    ))
fig.update_xaxes(title="realized cost — media $ spent, or promo margin given away")
fig.update_yaxes(title="expected value returned ($)")
style(fig, 400, "Every decision arm on ONE cost axis — promo depth competes for the same dollar")
fig.show()
print("(media is near-saturated in this world while promo is not — the joint "
      "optimizer funds promo from the same budget; see demos/promo_depth_optimization)")
"""

PAYBACK = r"""
from mmm_framework.planning import channel_payback

t0 = time.time()
payback = channel_payback(mmm)
print(f"channel_payback in {time.time()-t0:.1f}s\n")
rows = []
for ch, p in payback.channels.items():
    t50, t90 = p.horizons["t50"], p.horizons["t90"]
    rows.append({"channel": ch, "status": p.status,
                 "t50 (wk)": f"{t50['mean']:.2f}",
                 "t90 (wk)": f"{t90['mean']:.2f}",
                 "tail beyond l_max": f"{p.truncated_tail_mass:.1%}",
                 "carryover learning": p.learning_verdict})
print(pd.DataFrame(rows).to_string(index=False))
print("\nrun-level caveats:", payback.caveats or "none")

fig = go.Figure()
for i, ch in enumerate(CHANNELS):
    p = payback.channels[ch]
    t50, t90 = p.horizons["t50"], p.horizons["t90"]
    fig.add_trace(go.Scatter(x=[t50["mean"], t90["mean"]], y=[i, i], mode="lines",
                             line=dict(color=PALETTE[ch], width=2),
                             showlegend=False, hoverinfo="skip"))
    for h, lab, off in ((t50, "t50", -0.13), (t90, "t90", 0.13)):
        fig.add_trace(go.Scatter(x=[h["lower"], h["upper"]], y=[i + off, i + off],
                                 mode="lines", line=dict(color=PALETTE[ch], width=6),
                                 opacity=0.4, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[h["mean"]], y=[i + off], mode="markers",
                                 marker=dict(color=PALETTE[ch], size=11,
                                             symbol="circle" if lab == "t50" else "diamond"),
                                 name=f"{lab} (90% interval)", showlegend=(i == 0)))
    fig.add_annotation(x=t90["upper"], y=i, xshift=32, showarrow=False,
                       text=f"tail {p.truncated_tail_mass:.0%}",
                       font=dict(size=11, color=MUTED))
fig.update_yaxes(tickmode="array", tickvals=list(range(len(CHANNELS))),
                 ticktext=CHANNELS)
fig.update_xaxes(title="weeks until 50% / 90% of the effect has landed")
style(fig, 380, "Half the effect lands late — with intervals, and the truncated tail disclosed")
fig.show()
"""

CAL_FLIGHT = r"""
from mmm_framework.planning import build_flighting_schedule
from mmm_framework.planning.calendar import PlanningCalendar

cal = PlanningCalendar.from_model(mmm, N_PLAN, fy_start_month=2)
print(f"calendar: {cal.cadence}, {cal.n_periods} periods, "
      f"{cal.periods()[0]} .. {cal.periods()[-1]}")
print("fiscal-year groups:", {fy: len(ps) for fy, ps in cal.fiscal_groups().items()})

# The window budget: the recommended allocation, prorated to the 13-week plan.
window_budgets = {r_["channel"]: float(r_["optimal_spend"]) * N_PLAN / N_TRAIN
                  for _, r_ in res.table.iterrows()}
sched = build_flighting_schedule(
    window_budgets, N_PLAN, pattern="even", calendar=cal,
    per_channel_pattern={"TV": "front_loaded", "Search": "even",
                         "Social": "pulsed", "Display": "back_loaded"})
print("\nfirst two scheduled periods:")
print(pd.DataFrame(sched["schedule"][:2]).round(0).to_string(index=False))

# The refusal: a calendar that cannot label every period refuses rather than
# inventing P14..P17 (whose lexicographic sort once shuffled a pacing series).
try:
    build_flighting_schedule(window_budgets, N_PLAN + 4, calendar=cal)
except ValueError as e:
    print("\nREFUSED (calendar coverage):", str(e)[:110], "...")

fig = go.Figure()
for ch in CHANNELS:
    fig.add_trace(go.Scatter(x=sched["periods"], y=sched["by_channel"][ch],
                             mode="lines+markers", name=ch,
                             line=dict(color=PALETTE[ch], width=2)))
fig.update_yaxes(title="weekly spend")
fig.update_xaxes(tickangle=45)
style(fig, 380, "Same budgets, four shapes — flighting is a decision, not a default")
fig.show()
"""

FORECAST = r"""
from mmm_framework.planning import forecast_under_plan

# The plan we will commit: the recommended reallocation, shaped like the last
# 13 observed weeks and clipped inside observed support (the commit gate
# refuses curve fiction, and we intend to commit this one).
plan_scale = {r_["channel"]: float(r_["optimal_spend"] / r_["current_spend"])
              for _, r_ in res.table.iterrows()}
plan_media = {}
for c in CHANNELS:
    base = train.spend[c].to_numpy()[-N_PLAN:]
    cap = 0.95 * float(train.spend[c].max()) / max(float(base.max()), 1e-9)
    plan_media[c] = [float(v) for v in base * min(plan_scale[c], cap)]
# Controls are a planning ASSUMPTION — required, because there is no
# defensible default for the future value of a control.
plan_controls = {c: [float(v) for v in world.controls[c].to_numpy()[N_TRAIN:]]
                 for c in world.controls.columns}

fc = forecast_under_plan(mmm, plan_media, future_controls=plan_controls,
                         calendar=cal, interval=0.9, max_draws=200,
                         random_seed=42)

# Caveats FIRST — computed from the fit, never template sentences.
stmts = fc.caveats.statements()
print("caveats (computed, before any number):")
for s in (stmts or ["  (none fired: linear trend, iid residuals, plan inside support)"]):
    print("  -", s[:150])
h = fc.headline()
print(f"\nheadline: {h['total']:,.0f} KPI units over {h['n_periods']} weeks, "
      f"{h['interval']:.0%} {h['interval_noun']} interval on the WINDOW TOTAL "
      f"[{h['total_lower']:,.0f}, {h['total_upper']:,.0f}]")

# An aggressive plan does not error — it computes, and its caveats say why the
# number is fiction. Committing it is what gets refused (next section).
wild = {c: [v * (2.5 if c == "TV" else 1.0) for v in plan_media[c]]
        for c in CHANNELS}
fc_wild = forecast_under_plan(mmm, wild, future_controls=plan_controls,
                              interval=0.9, max_draws=200, random_seed=42)
print("\nthe 2.5x-TV plan's own caveat:",
      [s for s in fc_wild.caveats.statements() if "observed spend" in s][0][:150], "...")
"""

FORECAST_CHART = r"""
H = 26
hist_x = list(range(N_TRAIN - H, N_TRAIN))
fut_x = list(range(N_TRAIN, N_TRAIN + N_PLAN))
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist_x, y=train.y.to_numpy()[-H:], mode="lines",
                         name="observed KPI", line=dict(color=INK, width=1.6)))
fig.add_trace(go.Scatter(x=fut_x + fut_x[::-1],
                         y=list(fc.upper) + list(fc.lower)[::-1],
                         fill="toself", fillcolor="rgba(68,100,173,0.20)",
                         line=dict(width=0), name="90% predictive interval"))
fig.add_trace(go.Scatter(x=fut_x, y=fc.mean, mode="lines+markers",
                         name="forecast under the plan",
                         line=dict(color=PALETTE["TV"], width=2)))
fig.add_vline(x=N_TRAIN - 0.5, line=dict(color=MUTED, dash="dot"))
fig.add_annotation(x=N_TRAIN - 0.5, y=1.02, yref="paper", showarrow=False,
                   text="plan window starts", font=dict(color=MUTED, size=11))
fig.update_xaxes(title="week")
fig.update_yaxes(title="KPI")
style(fig, 400, "The band is PREDICTIVE (noise included) — the interval a plan is judged against")
fig.show()
"""

COMMIT_GATE = r"""
import dataclasses, hashlib, json
from mmm_framework.platform.plan_of_record import (assess_committability,
                                                   build_commit_payload)
from mmm_framework.platform.runs import data_fingerprint
from mmm_framework.serialization import MMMSerializer

RUN_DIR = ART / "run_1"
MMMSerializer.save(mmm, str(RUN_DIR))
(RUN_DIR / "run_metadata.json").write_text(json.dumps({"spec": SPEC, "kpi": "Sales"}))
fp = data_fingerprint(str(DATA))
prov = {"run_id": "run_1",
        "spec_hash": hashlib.md5(json.dumps(SPEC, sort_keys=True).encode()).hexdigest(),
        "data_fingerprint": fp["md5"], "dataset_path": str(DATA),
        "model_path": str(RUN_DIR), "random_seed": 42}

def fc_payload(f):
    return {"periods": list(f.periods),
            "mean": [float(x) for x in f.mean],
            "lower": [None if np.isnan(x) else float(x) for x in f.lower],
            "upper": [None if np.isnan(x) else float(x) for x in f.upper],
            "interval": float(f.interval), "draws_b64": f.draws_b64,
            "n_draws": int(f.n_draws),
            "caveat_fields": dataclasses.asdict(f.caveats)}

# The wild plan is REFUSED at commit time — a variance against curve fiction
# measures nothing. The gate is overridable; provenance gaps never are.
wild_c = assess_committability(fc_payload(fc_wild), provenance=prov,
                               valuation=RV.to_dict())
print("2.5x-TV plan committable:", wild_c.committable)
for r_ in wild_c.refusals:
    print(f"  [{r_.gate}] overridable={r_.overridable}: {r_.reason[:120]}...")
    print(f"      remedy: {r_.remedy[:100]}")

comm = assess_committability(fc_payload(fc), provenance=prov,
                             valuation=RV.to_dict())
if not comm.committable:
    ov = {r_.gate: "Acknowledged in this walkthrough" for r_ in comm.refusals
          if r_.overridable}
    comm = assess_committability(fc_payload(fc), provenance=prov,
                                 valuation=RV.to_dict(), overrides=ov)
    print("\nwaived gates (RECORDED in the payload):", list(comm.overrides))
print("\nsane plan committable:", comm.committable,
      "| missing provenance:", comm.missing_provenance or "none")
"""

COMMIT_CHAIN = r"""
import mmm_framework.platform.sessions as S

S.DB_PATH = ART / "sessions.db"
if S.DB_PATH.exists():
    S.DB_PATH.unlink()
S.init_db()

snapshot = fc_payload(fc)
snapshot.update({"plan_media": plan_media, "plan_controls": plan_controls,
                 "random_seed": 42})
payload = build_commit_payload(
    forecast=snapshot,
    allocation=res.table[ALLOC_COLS].to_dict("records"),
    flighting=sched, calendar=cal.to_dict(), provenance=prov,
    valuation=RV.to_dict(),
    objective={"objective": res.objective, "mode": res.mode},
    committability=comm)

v1 = S.commit_plan_version(plan_family="fy27-q1", org_id="acme",
                           project_id="demo", payload=payload,
                           name="FY27-Q1 plan", run_id="run_1",
                           spec_hash=prov["spec_hash"],
                           data_fingerprint=fp["md5"], committed_by="notebook")
v2 = S.commit_plan_version(plan_family="fy27-q1", org_id="acme",
                           project_id="demo", payload=payload,
                           name="FY27-Q1 plan (resubmitted after CFO review)",
                           run_id="run_1", spec_hash=prov["spec_hash"],
                           data_fingerprint=fp["md5"], committed_by="notebook")

print(f"v1: version={v1['version']}  prev_hash={v1['prev_hash']!r}")
print(f"    hash     ={v1['hash'][:24]}...")
print(f"v2: version={v2['version']}  prev_hash={v2['prev_hash'][:24]}...  "
      f"(links to v1: {v2['prev_hash'] == v1['hash']})")
print(f"    status: v1 is now '{S.get_plan_version(v1['id'])['status']}', "
      f"v2 is '{v2['status']}' — payloads are never touched")
print("\nchain verification:", S.verify_plan_chain("fy27-q1", "acme"))
"""

REPRODUCE = r"""
from mmm_framework.platform.plan_of_record import reproduce_committed_plan

committed = S.get_plan_version(v2["id"])
t0 = time.time()
rep = reproduce_committed_plan(committed)
print(f"reproduced: {rep.reproduced}   max |diff| = {rep.max_abs_diff:.2e} "
      f"(tolerance {rep.tolerance})   in {time.time()-t0:.0f}s")
print("per-array diffs:", {k: f"{v:.2e}" for k, v in rep.diffs.items()})

# Now change one byte of the dataset. Reproduction REFUSES — "the data moved"
# and "the commitment does not reproduce" are different statements, and
# recomputing against different data would not verify the committed number.
orig = DATA.read_bytes()
DATA.write_bytes(orig.replace(b".", b",", 1))
rep2 = reproduce_committed_plan(committed)
print(f"\nafter the CSV changed -> refused={rep2.refused}:")
print("  " + rep2.reason[:170] + "...")
DATA.write_bytes(orig)

# And tamper with the STORE itself: edit a committed payload in sqlite by
# hand. The chain hash covers the payload JSON, so verification breaks at
# exactly the edited version.
import sqlite3
con = sqlite3.connect(S.DB_PATH)
raw = json.loads(con.execute("SELECT payload_json FROM plan_versions WHERE id=?",
                             (v1["id"],)).fetchone()[0])
raw["forecast"]["mean"][0] = float(raw["forecast"]["mean"][0]) * 1.1
con.execute("UPDATE plan_versions SET payload_json=? WHERE id=?",
            (json.dumps(raw, sort_keys=True, default=str), v1["id"]))
con.commit(); con.close()
print("\nafter editing v1's payload in the DB:", S.verify_plan_chain("fy27-q1", "acme"))
"""

PACING = r"""
from mmm_framework.planning import compute_pacing, expected_outcome_delta

# Six of thirteen weeks in, delivery has drifted: TV under-delivered, Search
# over-paced. Rows carry period labels, so pacing joins BY LABEL — a mid-
# flight upload of weeks 5-8 is never compared against the plan's weeks 1-4.
ELAPSED = 6
DRIFT = {"TV": 0.82, "Search": 1.24, "Social": 1.0, "Display": 0.95}
rng = np.random.default_rng(3)
delivered = {c: np.asarray(plan_media[c]) * DRIFT[c]
             * rng.normal(1.0, 0.03, N_PLAN) for c in CHANNELS}
periods = list(fc.periods)
plan_rows = [{"period": p, **{c: float(plan_media[c][i]) for c in CHANNELS}}
             for i, p in enumerate(periods)]
act_rows = [{"period": p, **{c: float(delivered[c][i]) for c in CHANNELS}}
            for i, p in enumerate(periods[:ELAPSED])]

pace = compute_pacing(plan_rows, act_rows)
print(f"join: {pace.join}   elapsed fraction of plan: {pace.elapsed_fraction:.2f}   "
      f"portfolio divergence: {pace.divergence_pct:+.1%}")
for chn in pace.channels:
    print(f"  {chn.channel:8s} planned {chn.planned:>10,.0f}   "
          f"actual {chn.actual:>10,.0f}   {chn.divergence_pct:+7.1%}   {chn.status}")

# What the divergence COSTS, read off the fitted curves ON THEIR OWN AXIS.
# The curves cover the model's full 104-week window, so the elapsed 6-week
# totals are projected by their share of THAT axis; the legacy read fed
# part-window totals into a full-window curve and landed on the steep left.
p_tot = {c.channel: c.planned for c in pace.channels}
a_tot = {c.channel: c.actual for c in pace.channels}
legacy = expected_outcome_delta(curves, p_tot, a_tot)
fixed = expected_outcome_delta(curves, p_tot, a_tot,
                               elapsed_fraction=ELAPSED / N_TRAIN)
print(f"\nlegacy read  : {legacy['mean']:+,.0f} KPI  ({legacy['window_basis'][:44]}...)")
print(f"elapsed-aware: {fixed['mean']:+,.0f} KPI "
      f"[{fixed['lower']:+,.0f}, {fixed['upper']:+,.0f}]")
print(f"window_basis : {fixed['window_basis']}")

fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38],
                    subplot_titles=("cumulative spend, plan vs delivered",
                                    "divergence at week 6"))
for ch in CHANNELS:
    fig.add_trace(go.Scatter(x=periods, y=np.cumsum(plan_media[ch]), mode="lines",
                             line=dict(color=PALETTE[ch], width=1.4, dash="dot"),
                             name=f"{ch} plan", legendgroup=ch), row=1, col=1)
    fig.add_trace(go.Scatter(x=periods[:ELAPSED],
                             y=np.cumsum(delivered[ch][:ELAPSED]),
                             mode="lines+markers",
                             line=dict(color=PALETTE[ch], width=2.6),
                             name=f"{ch} actual", legendgroup=ch,
                             showlegend=False), row=1, col=1)
divs = [c.divergence_pct for c in pace.channels]
fig.add_trace(go.Bar(x=[c.channel for c in pace.channels], y=divs,
                     marker_color=[BAD if abs(d) > pace.threshold else MUTED
                                   for d in divs], showlegend=False), row=1, col=2)
fig.add_hline(y=pace.threshold, line=dict(color=MUTED, dash="dot"), row=1, col=2)
fig.add_hline(y=-pace.threshold, line=dict(color=MUTED, dash="dot"), row=1, col=2)
fig.update_yaxes(tickformat="+.0%", title="vs plan", row=1, col=2)
fig.update_xaxes(tickangle=45)
style(fig, 420, "Six weeks in: Search over-paces, TV under-delivers — priced on the curve's own axis")
fig.show()
"""

VARIANCE = r"""
from mmm_framework.planning import supplied_line, variance_to_plan

# The season ends: delivery stayed drifted all 13 weeks; the realized KPI is
# the world's own y over the window. The bridge runs against the COMMITTED
# version fetched back from the store — never against a newer fit.
actual_media = {c: [float(v) for v in delivered[c]] for c in CHANNELS}
actuals = [{"period": p, "kpi_value": float(v)}
           for p, v in zip(periods, world.y.to_numpy()[N_TRAIN:], strict=True)]
gtn = supplied_line("Gross-to-net true-up", -120.0,
                    source_note="ERP export 2027-05-02, finance-approved")

bridge = variance_to_plan(mmm, committed["payload"], actual_media, actuals,
                          supplied=[gtn], value_per_kpi=VALUE,
                          value_source=RV.source)

# The verdict LEADS: a miss inside the committed band is the uncertainty that
# was signed, not a story owed.
print("within the committed interval:", bridge.within_committed_interval)
print(f"committed band [{bridge.committed_lower:,.0f}, {bridge.committed_upper:,.0f}] "
      f"at {bridge.interval_mass:.0%} mass")
print(f"delivery total {bridge.delivery_total:+,.0f} "
      f"[{bridge.delivery_lower:+,.0f}, {bridge.delivery_upper:+,.0f}]   "
      f"unexplained {bridge.unexplained:+,.0f}\n")
print(pd.DataFrame([{"line": ln.name, "KPI units": round(ln.value, 1),
                     "provenance": ln.provenance.value}
                    for ln in bridge.rows]).to_string(index=False))
print(f"\nrows sum to {sum(ln.value for ln in bridge.rows):+,.1f} "
      f"= gap {bridge.gap:+,.1f}   (closes: {bridge.closes})")

fig = go.Figure(go.Waterfall(
    orientation="v",
    measure=["absolute"] + ["relative"] * len(bridge.rows) + ["total"],
    x=["Committed"] + [ln.name for ln in bridge.rows] + ["Actual"],
    y=[bridge.committed_kpi] + [ln.value for ln in bridge.rows] + [0.0],
    connector={"line": {"color": MUTED, "width": 1}},
    increasing={"marker": {"color": GOOD}},
    decreasing={"marker": {"color": BAD}},
    totals={"marker": {"color": INK}},
))
fig.update_yaxes(title="KPI units")
fig.update_xaxes(tickangle=30)
style(fig, 440, "Committed → delivery → supplied → unexplained → actual: every step named, sums exact")
fig.show()
"""

VARIANCE_REFUSALS = r"""
# The refit split is REFUSED, with the two identifiable buckets still
# delivered. "Effectiveness variance" from a refit mixes more data, a
# different window and Monte Carlo noise under a causal-sounding label.
b3 = variance_to_plan(mmm, committed["payload"], actual_media, actuals,
                      refit_run_id="run_refit_2027_05")
print("REFUSAL:", b3.refusals[0][:150], "...")
print("delivery lines still present:",
      sum(1 for ln in b3.rows if ln.name.startswith("Delivery — ")))
assert "effectiveness" not in json.dumps(b3.to_dict()).lower()
print('the word "effectiveness" appears nowhere in the payload\n')

# A half-realized window refuses: it would compare a full-window commitment
# against a part-window actual.
try:
    variance_to_plan(mmm, committed["payload"], actual_media, actuals[:-1])
except ValueError as e:
    print("REFUSED (coverage):", str(e)[:120], "...")

# And a model that cannot re-produce the committed snapshot refuses: its
# "delivery bucket" would be a different posterior wearing the committed label.
tampered = json.loads(json.dumps(committed["payload"]))
tampered["forecast"]["mean"] = [float(v) * 1.1
                                for v in tampered["forecast"]["mean"]]
try:
    variance_to_plan(mmm, tampered, actual_media, actuals)
except ValueError as e:
    print("REFUSED (reproduction):", str(e)[:120], "...")
"""

CELLS = [
    md(r"""
# A number the CFO can commit to

An MMM that stops at "here is a posterior" leaves the hard part to a
spreadsheet: turning KPI units into dollars, picking a budget, laying it on a
calendar, promising a number, and answering for the miss. This notebook walks
that whole loop on one small fitted model, and the theme throughout is that
**every dollar states its basis, and everything that cannot be known honestly
is refused by name** rather than defaulted.

1. **Finance** — `kpi_to_dollars` valuation with provenance, the
   SUPPLIED/MODELLED bridge-line vocabulary, decomposition closure
2. **Optimize** — fixed reallocation, the budget frontier, and the
   `mode='free'` refusal without a valuation
3. **Decision arms** — promo depth competing with media on one cost axis
4. **Payback** — when the money comes back, intervals and tail mass attached
5. **Calendar + flighting** — when the money goes out
6. **Forecast under the plan** — caveats first, predictive interval second
7. **Commit** — the plan of record: gates, hash chain, reproduce to 1e-9
8. **Pacing** — mid-flight divergence, priced on the curve's own axis
9. **Variance to plan** — the bridge from committed to actual that closes
   exactly, and refuses the refit "effectiveness" bucket
"""),
    code(SETUP),
    md(r"""
## 1 · One small world, one NUTS-lite fit

117 weeks of the clean synthetic world (the model's exact generative family).
The model trains on the first 104; the last 13 — one fiscal quarter — are the
window the plan will commit. One fit powers every section below.
"""),
    code(WORLD_FIT),
    md(r"""
## 2 · What is one KPI unit worth? Resolved once, with provenance

Every planning surface that recommends a *dollar* divides by a
value-per-KPI-unit. Before v1.4 a missing valuation silently fell back to
`1.0` — an assertion that one KPI unit is worth one dollar, wrong by ~1000x on
a KPI denominated in thousands, and it shipped a wrong number in production.
`finance.kpi_to_dollars` resolves the value through one precedence chain
(caller override → model spec → project preferences → branding), returns it
with its `source` attached, and treats **unresolved as a state, not a number**.
"""),
    code(FINANCE),
    md(r"""
### The bridge-line vocabulary

Two bars on a P&L bridge can look identical and mean different things: one
read off the fit, one typed in by finance. `BridgeLine` makes the difference
structural — MODELLED lines carry intervals with a stated mass, OBSERVED
totals carry none, a RESIDUAL is a named gap, and a SUPPLIED number requires a
source note and may not wear an interval (an assertion dressed as an
estimate). Anything unstated defaults to ABSORBING, the pessimistic reading.
"""),
    code(LINES),
    md(r"""
### Decomposition closure — the ledger check before any dollar leaves

Before pricing decisions off a decomposition, reconcile it: the components
plus a **disclosed** unexplained line must sum to the observed KPI. The
alternative — folding the residual into a bar labelled "base demand" — is how
baselines quietly absorb model error. Note what the reading insists on: a
small residual validates the accounting, not the baseline.
"""),
    code(CLOSURE),
    md(r"""
## 3 · The budget optimizer — and the one mode that needs money

`optimize_budget` reallocates in KPI units, so **fixed-budget mode needs no
valuation at all** — a positive constant does not move an argmax. Bounds keep
the recommendation inside the spend range the model has evidence for, and the
decision-uncertainty numbers (P(uplift>0), expected regret) travel with the
plan. `mode='free'` is different: funding every channel to breakeven trades
KPI against dollars, so without a resolved valuation it refuses — this exact
default-`1.0` path once told a client to fund channels until a KPI unit cost
a dollar.
"""),
    code(OPT),
    code(OPT_CHART),
    md(r"""
### The frontier — what the *next* dollar buys

A single optimal plan answers "how should I spend B". The frontier answers the
CFO's actual question — "should B be bigger?" — by re-optimizing at each
budget and reading the local slope. With the valuation resolved, that slope
becomes dollars-back-per-dollar, and the crossing at $1 is the stopping
argument.
"""),
    code(FRONTIER),
    md(r"""
## 4 · Decision arms — promo depth competes for the same money

A promotion's cost is not a spend line; it is **margin given away**
(depth × price × units). The decision-arm surface re-parameterizes every arm
by its *realized cost*, so promo depth and media spend sit on one budget
axis and the whole allocator (constraints, risk objectives, per-draw
uncertainty) runs unchanged. This takes a second small fit, because the
levers must be declared — `Price` and `Promo` get their own transforms and
priors instead of hiding in the linear control block.
"""),
    code(PROMO_FIT),
    code(PROMO_ROI),
    code(PROMO_CHART),
    md(r"""
## 5 · Payback — when the money comes back

"When does a dollar pay back?" reads the model's **least identified**
parameter, so nothing renders without a per-draw interval, a learning verdict
(did the data actually move the carryover prior?), and the truncated tail mass
— the share of effect the adstock window discards and silently re-spreads
inside itself, which makes every horizon read structurally optimistic. The
disclosure is on the chart, not in a footnote.
"""),
    code(PAYBACK),
    md(r"""
## 6 · The calendar and the flighting shape

Budgets say how much; flighting says when. `PlanningCalendar.from_model`
derives the forward calendar from the panel's own date index (a plan cannot
silently sit on a different cadence than the fit), labels periods with ISO
dates — `P1..Pn` labels once shuffled a pacing series because `P10` sorts
before `P2` — and groups them into fiscal years. The schedule then spreads
each channel's budget across the window, one pattern per channel.
"""),
    code(CAL_FLIGHT),
    md(r"""
## 7 · Forecast under the plan — caveats first, by construction

A forecast is a counterfactual under a plan the model never observed, so
`forecast_under_plan` computes its failure modes from the fit — trend
extrapolation policy, residual autocorrelation, spend beyond observed
support, interval availability — and `headline()` refuses to render the
number without them. The interval is **predictive** (observation noise
included): the interval a plan is later judged against. The window total's
band comes from summing *draws*, never per-period bounds — the periods are
correlated and their errors partly cancel.
"""),
    code(FORECAST),
    code(FORECAST_CHART),
    md(r"""
## 8 · Commit — a plan of record, not a screenshot

A committed plan is the number a variance is later measured against, and two
things ruin that: a commitment nobody can regenerate, and a commitment to
curve fiction. So committing is **gated**: spend beyond observed support,
autocorrelated residuals, a held-flat trend past 13 periods, or a missing
interval each refuse — waivable only with an acknowledgement that is
*recorded in the payload*. Missing provenance (run id, spec hash, data
fingerprint, model path) is never waivable: waiving it would recreate exactly
the unreproducible commitment the store exists to prevent.

Each committed version is immutable and **hash-chained** — the chain covers
the payload JSON itself, so editing a committed number after the fact breaks
verification at that exact version.
"""),
    code(COMMIT_GATE),
    code(COMMIT_CHAIN),
    md(r"""
### Reproduce, or refuse

The claim a commitment makes is not "here is a number" but "here is a number
anyone can regenerate". `reproduce_committed_plan` reloads the model from the
recorded run, rebuilds the panel from the **saved** spec and the fingerprinted
dataset, re-runs the forecast with the recorded plan and seed, and demands the
snapshot back to 1e-9. And it distinguishes two failures a lesser check would
conflate: "the inputs moved, I refuse to check" versus "I checked and the
numbers differ".
"""),
    code(REPRODUCE),
    md(r"""
## 9 · Mid-flight: pacing, priced on the curve's own axis

Six weeks in, delivery has drifted. Pacing joins plan and actuals **by period
label**, flags channels beyond the threshold, and prices the divergence off
the fitted response curves. The subtlety is the axis: the curves cover the
model's full 104-week fitted window, so elapsed totals are projected to that
axis (a stated proportional-flighting assumption) and the delta scaled back —
the legacy read fed part-window totals into a full-window curve and landed on
the steep left of saturation. The payload names its `window_basis` so the
reader knows which read happened.
"""),
    code(PACING),
    md(r"""
## 10 · Season over: the variance bridge that closes — and refuses

The honest bridge from committed to actual has exactly **two identifiable
buckets** without a refit. *Delivery variance*: the committed posterior prices
the spend divergence as a paired counterfactual, per channel. *Unexplained*:
the labelled remainder — baseline movement, competitor action, data error,
noise — disclosed, never attributed. Human adjustments enter as SUPPLIED
lines against the total. The rows sum to actual − committed to 1e-9 by
construction, and the interval verdict leads: a miss inside the committed
band is the uncertainty that was signed.

The tempting third bucket — refit on the realized season and call the
difference "effectiveness" — is refused with the reason stated, because that
subtraction mixes more data, a different window, and sampling noise under a
causal-sounding label.
"""),
    code(VARIANCE),
    code(VARIANCE_REFUSALS),
    md(r"""
## Where to go deeper

- **Specs** — `technical-docs/engineering-notes.md` (*Variance to plan*,
  *Model-anchored experiment economics*, *ROI-based default media priors*),
  `technical-docs/experiment-economics.md`,
  `technical-docs/impression-level-roi.md` (why `compute_response_curves`
  refuses a non-monetary channel by name).
- **Full-depth demos this notebook compresses** —
  `nbs/demos/payback_horizon.ipynb` (truncation illusion, prior-domination
  gate, misspecification negative control),
  `nbs/demos/promo_depth_optimization.ipynb` (the joint solve graded against
  a planted optimum, the price what-if that refuses to recommend),
  `nbs/demos/variance_to_plan.ipynb` (the delivery bucket graded against a
  causal answer key).
- **The measurement loop around the plan** — `nbs/lifecycle_00..06` (T₀–T₅:
  fit → prioritize → design → calibrate → allocate → re-evaluate).
- **Foundations** — `nbs/workshop_05_from_draws_to_decisions.ipynb` (why
  decisions read draws, not means), `nbs/math_02_saturation.ipynb` (the curve
  every allocation reads).
- **Platform surfaces** — the Planner page (commit action, pacing ledger,
  Performance → Variance) and the agent tools (`forecast_plan`,
  `get_variance_to_plan`) run these same ops; `docs/api-contracts.html` pins
  the payload shapes.
"""),
]


def main() -> None:
    nb = new_notebook(cells=CELLS)
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    out = "showcase/showcase_05_plan_and_commit.ipynb"
    with open(out, "w") as fh:
        nbformat.write(nb, fh)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
