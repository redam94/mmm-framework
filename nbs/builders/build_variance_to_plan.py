"""Author demos/variance_to_plan.ipynb (run from ``nbs/``).

    uv run --with nbformat python builders/build_variance_to_plan.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        demos/variance_to_plan.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

Usage walkthrough for the variance-to-plan surface (issue #227): commit a plan
whose forecast snapshot carries its own reproduction inputs, let the season
play out in a synthetic world with a causal answer key, then build the
two-bucket bridge — delivery variance on the COMMITTED posterior plus a
labelled unexplained remainder — and grade the delivery rows against the
world's own ``response_fn``. The refusals are demonstrated live: the refit
"effectiveness" split, per-channel supplied restatement, partial actuals
coverage, and a model that does not reproduce the committed snapshot. Every
number is computed in-notebook.
"""

from __future__ import annotations

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


INTRO = r"""
# Variance to plan — a bridge that closes, and refuses what it cannot know

A plan was committed. A season happened. The CFO asks: **why did we miss?**

The honest answer has exactly **two identifiable buckets** without a refit:

1. **Delivery variance** — spend diverged from plan, and the *committed*
   model can price that divergence as a paired counterfactual:
   $g_{\text{plan}}(S_{\text{actual}}) - g_{\text{plan}}(S_{\text{plan}})$,
   per channel, holding the posterior fixed.
2. **Unexplained** — realized KPI minus the committed model's forecast under
   the spend actually delivered. It mixes baseline movement, competitor
   action, data error, model error and noise, and it is **labelled** for what
   it contains rather than attributed.

The tempting third bucket — refit the model on the realized season and call
$g_{\text{new}}(S_{\text{actual}}) - g_{\text{plan}}(S_{\text{actual}})$
"effectiveness variance" — is **refused**, with the reason stated. That
subtraction mixes more data, a different training window, any spec changes and
Monte Carlo noise; giving it a causal-sounding label manufactures a claim no
one measured. What *can* be said is what changed between the runs, and the
surface attaches exactly that instead.

Two more disciplines run through everything below:

* **The bridge closes exactly.** The rows sum to actual − committed to 1e-9,
  by construction, with any human-supplied adjustment lines (gross-to-net,
  returns) subtracting from the remainder under a `SUPPLIED` provenance that
  demands a source note.
* **The committed interval leads.** Before anything is called a variance, the
  realized total is scored against the committed *window-total draws*. A miss
  inside the committed band is within the committed uncertainty — the bridge
  then explains composition, not a surprise.

This notebook demonstrates the whole loop in a synthetic world with a causal
answer key, so the delivery bucket can be graded against the truth.
"""

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
PALETTE = {"TV": "#4464ad", "Search": "#c9962e", "Social": "#3d7a5c", "Display": "#b4552d"}

def style(fig, height=380, title=None, **kw):
    fig.update_layout(height=height, title=title, margin=dict(t=64, l=64, r=30, b=52),
                      font=dict(size=12), **kw)
    return fig

ART = Path.cwd().parent / "artifacts" / "variance_to_plan"
ART.mkdir(parents=True, exist_ok=True)
print("Setup ready.")
"""

WORLD_MD = r"""
## 1 · A world with an answer key, and a committed model

182 weeks of the clean synthetic world (`make_clean` — the model's exact
generative family, so nothing below is confounded by misspecification). The
model sees the **first 156 weeks**; the last 26 are the fiscal window the plan
covers. The world keeps its structural `response_fn`, which is what lets us
grade the delivery bucket against a causal truth later.
"""

WORLD = r"""
from mmm_framework.synth import dgp, mff
from mmm_framework.agents.fitting import build_model

N_WEEKS, N_TRAIN = 182, 156
world = dgp.make_clean(seed=7, n_weeks=N_WEEKS)
train = world.slice(0, N_TRAIN)
CHANNELS = list(world.channels)

DATA = ART / "clean.csv"
mff.scenario_to_mff(train).to_csv(DATA, index=False)

SPEC = {
    "kpi": "Sales",
    "kpi_level": "national",
    "media_channels": [
        {"name": c, "adstock": {"type": "geometric"}, "saturation": {"type": "hill"}}
        for c in CHANNELS
    ],
    "control_variables": [{"name": c} for c in train.controls.columns],
}
t0 = time.time()
mmm = build_model(SPEC, str(DATA))
mmm.fit(draws=800, tune=800, chains=4, random_seed=42)
print(f"NUTS fit in {time.time()-t0:.0f}s")
"""

COMMIT_MD = r"""
## 2 · Commit a plan whose snapshot can be reproduced

The plan is the world's *true* future spend, rescaled: **TV at 1.3×** and
**Search at 0.7×** of what will actually be delivered. (Framing it this way
plants a known delivery divergence: TV will under-deliver against plan,
Search will over-deliver.)

The committed payload is shaped the way the `forecast_plan` op emits it since
#227: the forecast snapshot carries the **normalized per-period plan, the
controls assumption, the seed, and the per-period draws**. The plan and seed
make the commitment *reproducible* — anyone can regenerate the number. The
draws make it *gradeable* — a window-total interval cannot be recovered from
per-period bounds, so the payload stores the draws whole.
"""

COMMIT = r"""
from mmm_framework.planning.forecast import forecast_under_plan

truth_future = {c: [float(v) for v in world.spend[c].to_numpy()[N_TRAIN:]] for c in CHANNELS}
SCALE = {"TV": 1.3, "Search": 0.7}
plan_media = {c: [v * SCALE.get(c, 1.0) for v in s] for c, s in truth_future.items()}
plan_controls = {c: [float(v) for v in world.controls[c].to_numpy()[N_TRAIN:]]
                 for c in world.controls.columns}

fc = forecast_under_plan(mmm, plan_media, future_controls=plan_controls,
                         interval=0.9, max_draws=200, random_seed=42)

committed_payload = {
    "forecast": {
        "periods": list(fc.periods),
        "mean": [float(x) for x in fc.mean],
        "lower": [None if np.isnan(x) else float(x) for x in fc.lower],
        "upper": [None if np.isnan(x) else float(x) for x in fc.upper],
        "interval": float(fc.interval),
        "draws_b64": fc.draws_b64,
        "n_draws": int(fc.n_draws),
        "plan_media": plan_media,
        "plan_controls": plan_controls,
        "random_seed": 42,
    },
    "provenance": {"random_seed": 42},
}

committed_total = float(np.sum(fc.mean))
draws = fc.draws()                      # (n_draws, n_periods)
window_totals = draws.sum(axis=1)       # the per-draw WINDOW totals
lo, hi = np.percentile(window_totals, [5, 95])
print(f"Committed: {committed_total:,.0f} KPI units over {len(fc.periods)} periods")
print(f"90% committed band on the WINDOW TOTAL: [{lo:,.0f}, {hi:,.0f}]")
print(f"(sum of per-period bounds would claim [{np.nansum(fc.lower):,.0f}, "
      f"{np.nansum(fc.upper):,.0f}] — the perfect-correlation worst case)")
"""

SEASON_MD = r"""
## 3 · The season plays out

Delivery lands at the world's true spend (the divergence we planted), and the
realized KPI is the world's own `y` over the window — the truth plus the
world's observation noise, exactly what an actuals upload would contain.
"""

SEASON = r"""
actual_media = truth_future                       # what was actually spent
periods = committed_payload["forecast"]["periods"]
actuals = [{"period": p, "kpi_value": float(v)}
           for p, v in zip(periods, world.y.to_numpy()[N_TRAIN:], strict=True)]
actual_total = sum(a["kpi_value"] for a in actuals)
print(f"Realized KPI: {actual_total:,.0f}  vs committed {committed_total:,.0f}  "
      f"(gap {actual_total - committed_total:+,.0f})")
"""

BRIDGE_MD = r"""
## 4 · The bridge

`variance_to_plan` re-runs the committed forecast twice with the recorded
seed — once under the plan, once under actual spend — on the **committed
posterior**. Per-channel rows come from the forecast's own decomposition, the
paired per-draw totals give the delivery interval, and the remainder to the
realized KPI is the labelled unexplained line. The verdict against the
committed interval **leads**.
"""

BRIDGE = r"""
from mmm_framework.planning import variance_to_plan

bridge = variance_to_plan(mmm, committed_payload, actual_media, actuals)

print("Within the committed interval:", bridge.within_committed_interval)
print(f"Committed band: [{bridge.committed_lower:,.0f}, {bridge.committed_upper:,.0f}] "
      f"at {bridge.interval_mass:.0%} mass")
print(f"Delivery total: {bridge.delivery_total:+,.0f} "
      f"[{bridge.delivery_lower:+,.0f}, {bridge.delivery_upper:+,.0f}]")
print(f"Unexplained  : {bridge.unexplained:+,.0f}\n")

rows = pd.DataFrame([{
    "line": ln.name, "KPI units": round(ln.value, 1),
    "provenance": ln.provenance.value,
} for ln in bridge.rows])
print(rows.to_string(index=False))
print(f"\nrows sum to {sum(ln.value for ln in bridge.rows):+,.1f} "
      f"= gap {bridge.gap:+,.1f}  (closes: {bridge.closes})")
"""

WATERFALL = r"""
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
style(fig, 420, "The bridge from committed to actual — every step named, sums exact")
fig.update_yaxes(title="KPI units")
fig.show()
"""

TRUTH_MD = r"""
## 5 · Grading the delivery bucket against the world's own physics

The world kept its `response_fn`, so the *true* value of each channel's spend
divergence is computable exactly: the structural mean under actual spend minus
under planned spend, summed over the window (evaluated on the full history so
training-period carryover cancels). The planted divergences read back with the
right signs, and the paired-draw interval prices the estimate honestly.
"""

TRUTH = r"""
def true_delivery(channel=None):
    actual = world.spend.to_numpy(float)
    planned = actual.copy()
    for c in ([channel] if channel else CHANNELS):
        i = CHANNELS.index(c)
        planned[N_TRAIN:, i] = np.asarray(plan_media[c], dtype=float)
    return float((world.response_fn(actual) - world.response_fn(planned))[N_TRAIN:].sum())

est = {ln.name.replace("Delivery — ", ""): ln.value
       for ln in bridge.rows if ln.name.startswith("Delivery — ")}
cmp_rows = [{"channel": c, "true delivery variance": round(true_delivery(c), 1),
             "bridge row": round(est.get(c, 0.0), 1)} for c in CHANNELS]
print(pd.DataFrame(cmp_rows).to_string(index=False))

tt = true_delivery()
print(f"\nTotal truth {tt:+,.1f} vs bridge {bridge.delivery_total:+,.1f} "
      f"[{bridge.delivery_lower:+,.1f}, {bridge.delivery_upper:+,.1f}] — "
      f"covered: {bridge.delivery_lower <= tt <= bridge.delivery_upper}")

fig = go.Figure()
for i, c in enumerate(CHANNELS):
    fig.add_trace(go.Bar(x=[c], y=[est.get(c, 0.0)], marker_color=PALETTE[c],
                         name="bridge row", showlegend=(i == 0)))
    fig.add_trace(go.Scatter(x=[c], y=[true_delivery(c)], mode="markers",
                             marker=dict(color=TRUTH, symbol="line-ew-open", size=26,
                                         line=dict(width=3)),
                             name="response_fn truth", showlegend=(i == 0)))
style(fig, 360, "Per-channel delivery variance vs the causal answer key")
fig.update_yaxes(title="KPI units")
fig.show()
"""

SUPPLIED_MD = r"""
## 6 · Supplied lines — auditable, total-only, dollar-gated

Finance often holds an adjustment the model never saw: gross-to-net, returns,
trade spend true-ups. Those enter as `SUPPLIED` lines — a point value with a
**required source note**, no invented interval — and they subtract from the
unexplained remainder, never from a channel. The refusals are part of the
surface: a supplied line cannot restate a channel (that would be a net-scaled
ROI the model never estimated wearing the model's label), and it cannot apply
to a non-dollar KPI.
"""

SUPPLIED = r"""
from mmm_framework.planning import supplied_line

gtn = supplied_line("Gross-to-net true-up", -180.0,
                    source_note="ERP export 2026-07-31, finance-approved")
b2 = variance_to_plan(mmm, committed_payload, actual_media, actuals,
                      supplied=[gtn], value_per_kpi=2.0, value_source="preferences")

for ln in b2.rows:
    if ln.provenance.value in ("supplied", "residual"):
        print(f"{ln.name:28s} {ln.value:+10,.1f}   [{ln.provenance.value}]")
print(f"still closes: {b2.closes}\n")

for bad_call in (
    dict(name="Net factor", value=-5.0, source_note="s", channel="TV"),
    dict(name="Net factor", value=-5.0, source_note="s", kpi_kind_is_dollar=False),
    dict(name="Net factor", value=-5.0, source_note=""),
):
    try:
        supplied_line(bad_call.pop("name"), bad_call.pop("value"), **bad_call)
    except ValueError as e:
        print("REFUSED:", str(e)[:110], "…")
"""

REFUSAL_MD = r"""
## 7 · The refusals are the product

**The refit split.** Passing a `refit_run_id` does not produce an
"effectiveness" bucket — it produces a stated refusal, while the two
identifiable buckets are still delivered. (In the platform surface, the run
diff — what actually changed between the committed run and the refit — is
attached in the refusal's place.) The word "effectiveness" appears nowhere in
the payload, by test.

**Partial coverage.** A bridge over a half-realized window would compare a
full-window commitment against a part-window actual, so it refuses.

**A model that does not reproduce the snapshot.** If the model handed to the
bridge cannot re-produce the committed forecast bit-for-bit, its "delivery
bucket" would be a different posterior's opinion wearing the committed label —
the refused refit comparison in disguise.
"""

REFUSAL = r"""
import json

b3 = variance_to_plan(mmm, committed_payload, actual_media, actuals,
                      refit_run_id="run_refit_2026_08")
print("REFUSAL:", b3.refusals[0][:140], "…")
print("delivery lines still present:",
      sum(1 for ln in b3.rows if ln.name.startswith("Delivery — ")))
assert "effectiveness" not in json.dumps(b3.to_dict()).lower()
print('the word "effectiveness" appears nowhere in the payload\n')

try:
    variance_to_plan(mmm, committed_payload, actual_media, actuals[:-1])
except ValueError as e:
    print("REFUSED (coverage):", str(e)[:120], "…")

tampered = json.loads(json.dumps(committed_payload))
tampered["forecast"]["mean"] = [v * 1.1 for v in tampered["forecast"]["mean"]]
try:
    variance_to_plan(mmm, tampered, actual_media, actuals)
except ValueError as e:
    print("REFUSED (reproduction):", str(e)[:120], "…")
"""

VERDICT_MD = r"""
## 8 · The interval verdict leads

The same gap reads completely differently depending on where it sits relative
to the committed band. Score the realized total against the committed
*window-total draws* first; only then explain composition. Here: actuals at
the forecast mean (inside), and a wild miss (outside).
"""

VERDICT = r"""
mean_actuals = [{"period": p, "kpi_value": float(v)}
                for p, v in zip(periods, committed_payload["forecast"]["mean"], strict=True)]
wild_actuals = [{"period": p, "kpi_value": float(v) * 3.0}
                for p, v in zip(periods, committed_payload["forecast"]["mean"], strict=True)]

for label, acts in (("actuals ≈ forecast mean", mean_actuals),
                    ("actuals at 3× (wild miss)", wild_actuals)):
    b = variance_to_plan(mmm, committed_payload, actual_media, acts)
    print(f"{label:28s} → within_committed_interval = {b.within_committed_interval}")
    print("   ", [c for c in b.caveats if "committed interval" in c][0][:110], "…")
"""

PACING_MD = r"""
## 9 · A related fix: mid-flight pacing now reads the curve on its own axis

`expected_outcome_delta` prices in-flight divergence off the fitted response
curves. Those curves are computed over the model's **full fitted window** —
but mid-flight, the pacing totals cover only the elapsed fraction. The old
read fed elapsed totals straight into a full-window curve, landing on the
steep left of the saturation curve (and `np.interp` clamps silently past the
grid). The fix projects totals to the curve's axis (`x / f`, a stated
proportional-flighting assumption) and scales the delta back by `f` — and the
payload now names its window basis so the reader knows which read happened.
"""

PACING = r"""
from mmm_framework.planning.pacing import expected_outcome_delta

class ToyCurves:
    # A saturating response surface with a known shape (not a fit).
    channel_names = ["TV"]
    spend_grid = np.array([np.linspace(0.0, 1_000_000.0, 60)])
    contributions = np.array([[8_000.0 * (1 - np.exp(-np.linspace(0.0, 1_000_000.0, 60) / 300_000.0))]])

planned = {"TV": 250_000.0}   # elapsed-window totals, 25% through the year
actual = {"TV": 310_000.0}

legacy = expected_outcome_delta(ToyCurves(), planned, actual)
fixed = expected_outcome_delta(ToyCurves(), planned, actual, elapsed_fraction=0.25)
print(f"legacy read : {legacy['mean']:+,.0f}  ({legacy['window_basis']})")
print(f"elapsed-aware: {fixed['mean']:+,.0f}  ({fixed['window_basis'][:64]}…)")
"""

CLOSE_MD = r"""
## 10 · Where this lives in the platform

* **Engine** — `mmm_framework.planning.variance.variance_to_plan` (this
  notebook), pure compute on a committed payload.
* **Stores** — realized-KPI actuals are as-of-dated and append-preserving
  (`POST /projects/{id}/actuals`); delivery is the pacing ledger; the
  committed plan of record is an immutable, hash-chained version.
* **REST** — `POST /projects/{id}/variance` starts a non-blocking job that
  loads the **committed** run (never the latest); refusals surface at POST
  time as a 409 with the stated reason. Poll
  `GET /projects/{id}/variance/{job_id}`.
* **Agent** — the `get_variance_to_plan` tool assembles the same inputs and
  runs the same op; a refit gets the refusal plus the run diff.
* **Report** — `MMMReportGenerator(..., variance=bridge.to_dict())` renders
  the classic and Augur sections, verdict first.
* **Frontend** — Performance → Variance builds and polls the job, showing
  the same bridge with provenance chips.

The discipline the whole surface enforces: **a number nobody can regenerate is
not a commitment, a bucket nobody measured is not a variance, and a miss
inside the committed band is not a story owed — it is the uncertainty that was
signed.**
"""


def build() -> nbformat.NotebookNode:
    nb = new_notebook(
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        }
    )
    nb.cells = [
        md(INTRO),
        code(SETUP),
        md(WORLD_MD),
        code(WORLD),
        md(COMMIT_MD),
        code(COMMIT),
        md(SEASON_MD),
        code(SEASON),
        md(BRIDGE_MD),
        code(BRIDGE),
        code(WATERFALL),
        md(TRUTH_MD),
        code(TRUTH),
        md(SUPPLIED_MD),
        code(SUPPLIED),
        md(REFUSAL_MD),
        code(REFUSAL),
        md(VERDICT_MD),
        code(VERDICT),
        md(PACING_MD),
        code(PACING),
        md(CLOSE_MD),
    ]
    return nb


if __name__ == "__main__":
    out = "demos/variance_to_plan.ipynb"
    nbformat.write(build(), out)
    print(f"wrote {out} ({len(build().cells)} cells)")
