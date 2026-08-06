"""Author demos/promo_depth_optimization.ipynb (run from ``nbs/``).

    uv run --with nbformat python builders/build_promo_depth_optimization.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        demos/promo_depth_optimization.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

Usage walkthrough for the decision-arm surface (issue #226): the
`promo_and_media` world with its frozen economics, promo ROI with its
refusals, the cost-space reduction, the joint media+promo-depth optimization
graded against the planted optimum, the price what-if that refuses to
recommend, and the endogenous negative control. Every number is computed
in-notebook.
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

ART = Path.cwd().parent / "artifacts" / "promo_depth"
ART.mkdir(parents=True, exist_ok=True)
print("Setup ready.")
"""

WORLD = r"""
from mmm_framework.synth import dgp, mff

scenario = dgp.build("promo_and_media")
t = scenario.notes

print(scenario.description, "\n")
econ = {k: t[k] for k in ("gross_margin", "promo_unit_cost", "price_reference",
                          "true_promo_lift", "true_promo_alpha")}
print("frozen economics + planted effects:")
for k, v in econ.items():
    print(f"  {k:20s} {v:,.4g}")

opt, cur = t["true_optimal_split"], t["current_split"]
print(f"\nplanted OPTIMAL split : media {opt['media_cost']:,.0f} | promo {opt['promo_cost']:,.0f}"
      f"  (promo share {opt['promo_share']:.0%}, avg depth {opt['avg_promo_depth']:.3f})")
print(f"observed CURRENT split: media {cur['media_cost']:,.0f} | promo {cur['promo_cost']:,.0f}"
      f"  (promo share {cur['promo_cost']/(cur['promo_cost']+cur['media_cost']):.0%})")
print(f"planted profit gap    : {opt['profit'] - cur['profit']:,.0f} (conditional on the stated economics)")
"""

FIT = r"""
from mmm_framework.agents.fitting import build_model

DATA = ART / "promo_world.csv"
mff.scenario_to_mff(scenario).to_csv(DATA, index=False)

SPEC = {
    "kpi": "Sales",
    "kpi_level": "national",
    "media_channels": [
        {"name": c, "adstock": {"type": "geometric"},
         "saturation": {"type": "logistic"}}
        for c in scenario.channels
    ],
    "control_variables": [{"name": c} for c in scenario.controls.columns],
    # The lever declarations (#138/#222): Price and Promo leave the linear
    # control block and get their own transforms + priors.
    "price": {"variable": "Price", "reference": "median"},
    "promotions": [{"variable": "Promo", "adstock_lmax": 8}],
}
t0 = time.time()
mmm = build_model(SPEC, str(DATA))
mmm.fit(draws=800, tune=800, chains=4, random_seed=42)
print(f"NUTS fit in {time.time()-t0:.0f}s | levers: {mmm.lever_names}")
"""

ROI = r"""
from mmm_framework.planning import promo_roi
from mmm_framework.finance import UnresolvedValueError

value_per_kpi = t["gross_margin"] * t["price_reference"]

# Refusal 1: no valuation -> never a silent $1/KPI.
try:
    promo_roi(mmm, "Promo", unit_cost=t["promo_unit_cost"])
except UnresolvedValueError as e:
    print("without a valuation:", type(e).__name__)

r = promo_roi(mmm, "Promo", unit_cost=t["promo_unit_cost"],
              value_per_kpi=value_per_kpi, value_source="world economics")
print(f"\npromo ROI: {r.roi_mean:.2f} margin-$ back per margin-$ given away "
      f"[{r.roi_lower:.2f}, {r.roi_upper:.2f}] ({int(r.interval_mass*100)}%)")
print(f"lift: {r.lift_kpi_mean:,.0f} KPI units (planted {t['true_promo_lift']:,.0f} "
      f"-> {r.lift_kpi_mean/t['true_promo_lift']:.0%} recovered)")
print(f"cost basis: avg depth {r.avg_depth:.3f} x unit cost {r.unit_cost:,.0f} = {r.realized_cost:,.0f}")
print("\ncaveat:", r.caveats[-1])
"""

ROI_REFUSALS = r"""
# Refusal 2 + 3: the cost basis is refusable. A 0/1 event flag has no depth
# (no discount cost exists); a column outside [0,1] has unknown units.
class _Stub:
    lever_names = ["Promo"]
    X_levers_raw = None

for label, series in [
    ("event flag", np.array([0.0, 1.0, 0.0, 1.0, 1.0])),
    ("unknown units", np.array([0.0, 12.0, 0.0, 40.0])),
]:
    stub = _Stub(); stub.X_levers_raw = series[:, None]
    try:
        promo_roi(stub, "Promo", unit_cost=1.0, value_per_kpi=1.0)
    except ValueError as e:
        print(f"{label}: {str(e)[:110]}...")
"""

ARMS = r"""
from mmm_framework.planning import build_arm_curves, optimize_arms

t0 = time.time()
curves = build_arm_curves(mmm, promo_var="Promo", unit_cost=t["promo_unit_cost"],
                          max_draws=150, random_seed=42)
print(f"arm curves in {time.time()-t0:.0f}s")
for a in curves.arms:
    print(f"  {a.name:22s} kind={a.kind:5s} units={a.level_units}")

budget = t["true_optimal_split"]["budget"]
res = optimize_arms(curves=curves, total_budget=budget,
                    value_per_kpi=value_per_kpi, value_source="world economics",
                    min_multiplier=0.3, max_multiplier=1.5, random_seed=42)
cols = ["channel", "arm_kind", "optimal_spend", "optimal_level", "level_units",
        "change_pct", "within_observed_range"]
print("\n" + res.table[cols].round(3).to_string(index=False))
"""

GRADE = r"""
promo_row = res.table[res.table.arm_kind == "promo"].iloc[0]
rec_share = float(promo_row["optimal_spend"]) / budget
cur_share = t["current_split"]["promo_cost"] / budget
true_share = t["true_optimal_split"]["promo_share"]

fig = go.Figure(go.Bar(
    x=["current", "recommended", "planted optimum"],
    y=[cur_share, rec_share, true_share],
    marker_color=[MUTED, PALETTE["Promo depth (Promo)"], TRUTH],
))
fig.update_yaxes(title="promo share of total outlay", tickformat=".0%")
style(fig, 340, "The recommendation moves toward the planted optimum")
fig.show()

# TRUE planted profit of the recommendation (noiseless structural mean, same
# frozen economics the answer key used — conditional on them, as the world's
# own notes insist).
fn = t["lever_response_fn"]; n = len(scenario.y)
spend = scenario.spend.to_numpy(float)
alloc = {r_["channel"]: r_["optimal_spend"] for _, r_ in res.table.iterrows()
         if r_["arm_kind"] == "media"}
scale = np.array([alloc[c] / spend[:, i].sum() for i, c in enumerate(scenario.channels)])
d_rec = float(promo_row["optimal_level"])
mu_rec = fn(spend * scale[None, :], np.full(n, d_rec), t["price"]).sum()
profit_rec = value_per_kpi * mu_rec - (sum(alloc.values()) + d_rec * t["promo_unit_cost"])

print(f"recommended promo share : {rec_share:.1%}  (current {cur_share:.1%}, planted optimum {true_share:.1%})")
print(f"TRUE profit — current   : {t['current_split']['profit']:,.0f}")
print(f"TRUE profit — recommended: {profit_rec:,.0f}")
print(f"TRUE profit — planted opt: {t['true_optimal_split']['profit']:,.0f}")
print(f"decision regret vs optimum: {t['true_optimal_split']['profit'] - profit_rec:,.0f}")
"""

PRICE = r"""
from mmm_framework.planning import price_whatif

w = price_whatif(mmm, 0.95, max_draws=150)
print(f"scenario: {w['price_var']} x {w['factor']} (a 5% cut)")
print(f"KPI delta: {w['kpi_delta_mean']:,.0f} "
      f"[{w['kpi_delta_lower']:,.0f}, {w['kpi_delta_upper']:,.0f}]")
print(f"recommendation: {w['recommendation']}")
print(f"\nwhy it refuses:\n  {w['refusal_reason']}")
"""

ENDO = r"""
from mmm_framework.diagnostics.endogeneity import endogeneity_diagnostic

sc_en = dgp.build("promo_endogenous")
DATA_EN = ART / "promo_endo.csv"
mff.scenario_to_mff(sc_en).to_csv(DATA_EN, index=False)
spec_en = dict(SPEC)
spec_en["control_variables"] = [{"name": c} for c in sc_en.controls.columns]
m_en = build_model(spec_en, str(DATA_EN))
m_en.fit(draws=800, tune=800, chains=4, random_seed=42)

d = endogeneity_diagnostic(m_en)
rows = [{"column": r_["channel"], "kind": r_["kind"],
         "demand→setting": round(r_["demand_leads_spend"], 3),
         "setting→demand": round(r_["spend_leads_demand"], 3),
         "flagged": r_["endogenous"]} for r_ in d["channels"]]
print(pd.DataFrame(rows).to_string(index=False))
print("\nflagged levers:", d["flagged_levers"])

r_en = promo_roi(m_en, "Promo", unit_cost=sc_en.notes["promo_unit_cost"],
                 value_per_kpi=1.0)
r_ex = promo_roi(mmm, "Promo", unit_cost=t["promo_unit_cost"], value_per_kpi=1.0)
print(f"\nlift recovery — exogenous timing : {r_ex.lift_kpi_mean/t['true_promo_lift']:.0%}")
print(f"lift recovery — clearance timing : {r_en.lift_kpi_mean/sc_en.notes['true_promo_lift']:.0%}")
"""

CELLS = [
    md(r"""
# Promo depth vs media — decision arms with per-arm cost bases

The optimizer's decision vector was dollars of media spend. A promotion's cost
is not a spend line — it is **margin given away** (depth × price × units) — so
"should we fund a deeper promo or more media?" had no honest answer. This
notebook walks the #226 surface:

1. the `promo_and_media` world, whose economics and optimal split are **frozen
   before any model runs**
2. `promo_roi` — margin dollars back per margin dollar given away, with its
   refusals (event flags, unknown units, missing valuation)
3. the **cost-space reduction**: every arm re-parameterized by realized cost,
   so the existing allocator (constraints, risk objectives, per-draw
   uncertainty) runs untouched
4. the joint solve, graded against the planted optimum on **true** profit
5. `price_whatif` — a price scenario that evaluates and **refuses to recommend**
6. the endogenous negative control: clearance timing attenuates the lift, and
   the extended endogeneity screen says which lever and why

The headline the epic wanted — "trade a price cut against media" — is
deliberately *not* shipped as a recommendation: the repo's own measurement
recovers 39% of a planted price elasticity, confidently. Promo depth is the
shipping headline.
"""),
    code(SETUP),
    md(r"""
## 1 · A world with frozen economics

The answer key is planted from DGP parameters **first** and frozen in the
world's notes — the optimizer never sees the construction, and the profit
claim below is labelled *conditional on the stated economics* because the
optimizer is handed the same margin and promo cost the answer key used.
Media is near-saturated at current spend; promo is far from saturation. The
joint optimum genuinely wants promo money.
"""),
    code(WORLD),
    md(r"""
## 2 · Fit with the levers declared

`Price` and `Promo` are declared as **levers**, not linear controls: price
becomes a sign-guarded log-price elasticity, promo a lift with its own
carryover. (The discrimination test in the suite pins that a model WITHOUT the
promo lever misses the planted split by >15pp — a world that cannot fail is
decoration.)
"""),
    code(FIT),
    md(r"""
## 3 · Promo ROI, and what it refuses

Margin dollars returned per margin dollar given away — the number a trade
planner actually argues about. It needs a valuation and a cost basis, and both
are refusable.
"""),
    code(ROI),
    code(ROI_REFUSALS),
    md(r"""
## 4 · The joint solve — promo competes for the same money

`build_arm_curves` puts every arm on one **realized-cost** grid: media arms at
their spend, the promo arm at `depth × unit_cost`. The budget constraint is
already `Σ cost = B`, so the whole existing machinery — risk objectives,
constraints, the multi-start solver, per-draw decision uncertainty, the
out-of-support flag — runs unchanged, and the media-only path stays
bit-identical (pinned by test). New: a **concavity gate** — the greedy
allocator's exactness precondition is finally checked, and a failing arm
forces the multi-start constrained solver with a note.
"""),
    code(ARMS),
    md(r"""
The promo row's `optimal_spend` is **margin given away**, not a media buy —
which is why the table carries `arm_kind` and reports the recommendation in
the arm's own units (`optimal_level`, an average weekly depth).
"""),
    code(GRADE),
    md(r"""
The recommendation moves decisively toward the planted optimum. It does not
reach it — the fitted promo lift recovers ~85% of truth and the fitted media
saturation is imperfect — and the honest grade is the **decision regret**
against the planted optimum, printed above, not a victory lap.
"""),
    md(r"""
## 5 · Price: evaluate, label, refuse

The price lever's elasticity is measured at ~39% of planted truth by the
repo's own published simulation of this exact mechanism. So the price surface
answers stated hypotheticals and **refuses to emit a recommendation** — a
Granger-style screen not flagging is weak evidence of exogeneity, and the
default posture is refusal regardless of the flag.
"""),
    code(PRICE),
    md(r"""
## 6 · The negative control — clearance timing

In `promo_endogenous`, last week's soft demand triggers this week's deeper
promo and price cut. The lever now correlates with the error term: the naive
lift is **attenuated by construction**, and the extended endogeneity screen —
which now walks lever columns with a `kind` — shows the demand→setting lead.
"""),
    code(ENDO),
    md(r"""
## Where this surfaces

- **Planner tables** — allocation rows carry `arm_kind` / `level_units` /
  `optimal_level`; the frontend shows Kind and Level columns whenever a
  non-media arm is present.
- **Refusals upstream** — `compute_response_curves` now refuses a channel
  whose divisor is not monetary (impressions summed into a dollar budget), and
  `goal_seek` refuses a mixed-arm portfolio by name (its monotone-frontier
  proof covers concave spend curves only).
- **The valuation chain** — the agent tool's `value_per_kpi` default changed
  from a silent `1.0` to `None`: fund-to-breakeven now refuses without a
  declared valuation instead of asserting one KPI unit = one dollar.

### Reading list

- `nbs/demos/payback_horizon.ipynb` — the same epistemics discipline for
  response timing.
- `docs/blog-modelled-one-p.html` — why the price arm refuses: the 39%
  measurement.
- `tests/test_decision_arms.py` — the analytic equal-marginal-profit gate, the
  10-seed milestone, and the attenuation floor.
"""),
]


def main() -> None:
    nb = new_notebook(cells=CELLS)
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    out = "demos/promo_depth_optimization.ipynb"
    with open(out, "w") as fh:
        nbformat.write(nb, fh)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
