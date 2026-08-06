"""Author demos/payback_horizon.ipynb (run from ``nbs/``).

    uv run --with nbformat python builders/build_payback_horizon.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        demos/payback_horizon.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

Usage walkthrough for the payback-horizon surface (issue #224): channel_payback
on a world with a sealed carryover answer key, the truncation illusion measured
analytically, the prior-domination gate demonstrated with two defensible
priors, the misspecification negative control (gates fire, direction SHORT),
refusals by name, and the finance-sense break-even with a valuation. Every
number is computed in-notebook.
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

ART = Path.cwd().parent / "artifacts" / "payback_horizon"
ART.mkdir(parents=True, exist_ok=True)
print("Setup ready.")
"""

FIT = r"""
from mmm_framework.synth import dgp, mff
from mmm_framework.agents.fitting import build_model

scenario = dgp.make_clean(seed=7, n_weeks=156)
CHANNELS = list(scenario.channels)
DATA = ART / "clean.csv"
mff.scenario_to_mff(scenario).to_csv(DATA, index=False)

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
mmm.fit(draws=800, tune=800, chains=4, random_seed=42)
print(f"NUTS fit in {time.time()-t0:.0f}s")
"""

BASIC = r"""
from mmm_framework.planning import channel_payback

t0 = time.time()
payback = channel_payback(mmm)
print(f"channel_payback in {time.time()-t0:.1f}s\n")

rows = []
for ch, p in payback.channels.items():
    t50, t90 = p.horizons["t50"], p.horizons["t90"]
    rows.append({
        "channel": ch,
        "status": p.status,
        "t50 (wk)": f"{t50['mean']:.2f}",
        "t50 90% interval": f"[{t50['lower']:.2f}, {t50['upper']:.2f}]",
        "t90 (wk)": f"{t90['mean']:.2f}",
        "tail beyond l_max": f"{p.truncated_tail_mass:.1%}",
        "carryover learning": p.learning_verdict,
    })
print(pd.DataFrame(rows).to_string(index=False))
print("\nautocorrelation gate:", payback.autocorrelation)
print("run-level caveats  :", payback.caveats or "none")
"""

TRUTH = r"""
def truth_crossing(ch, share):
    cum = np.asarray(scenario.notes["true_adstock"][ch]["cum_share"], dtype=float)
    j = int(np.searchsorted(cum, share))
    prev = cum[j - 1] if j > 0 else 0.0
    return j + (share - prev) / (cum[j] - prev)

fig = go.Figure()
for i, ch in enumerate(CHANNELS):
    p = payback.channels[ch]
    h = p.horizons["t50"]
    fig.add_trace(go.Scatter(x=[h["lower"], h["upper"]], y=[i, i], mode="lines",
                             line=dict(color=PALETTE[ch], width=6), opacity=0.45,
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[h["mean"]], y=[i], mode="markers",
                             marker=dict(color=PALETTE[ch], size=13),
                             name="posterior (90% interval)", showlegend=(i == 0)))
    fig.add_trace(go.Scatter(x=[truth_crossing(ch, 0.5)], y=[i], mode="markers",
                             marker=dict(color=TRUTH, symbol="line-ns-open", size=20,
                                         line=dict(width=3)),
                             name="planted truth", showlegend=(i == 0)))
fig.update_yaxes(tickmode="array", tickvals=list(range(len(CHANNELS))), ticktext=CHANNELS)
fig.update_xaxes(title="t50 — lag at which half the effect has landed (weeks)")
style(fig, 360, "Per-draw t50 against the sealed answer key")
fig.show()

covered = sum(
    payback.channels[ch].horizons["t50"]["lower"]
    <= truth_crossing(ch, 0.5)
    <= payback.channels[ch].horizons["t50"]["upper"]
    for ch in CHANNELS
)
print(f"interval covers planted truth for {covered} of {len(CHANNELS)} channels")
"""

TRUNCATION = r"""
from mmm_framework.transforms.carryover import carryover_crossing_lags

alpha = 0.8
lmaxes = [8, 12, 16, 26, 52, 200]
t90s, tails = [], []
for L in lmaxes:
    w = (alpha ** np.arange(L)) / (alpha ** np.arange(L)).sum()
    t90s.append(float(carryover_crossing_lags(w[None, :], 0.9)[0]))
    tails.append(float(alpha ** L))

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("t90 vs the truncation window",
                                    "Untruncated mass beyond l_max"))
fig.add_trace(go.Scatter(x=[str(l) for l in lmaxes], y=t90s, mode="lines+markers",
                         marker=dict(color=BAD, size=10), showlegend=False), row=1, col=1)
fig.add_hline(y=t90s[-1], line=dict(color=MUTED, dash="dot"), row=1, col=1)
fig.add_trace(go.Bar(x=[str(l) for l in lmaxes], y=tails, marker_color=GOLD,
                     showlegend=False), row=1, col=2)
fig.update_xaxes(title="l_max", row=1, col=1); fig.update_xaxes(title="l_max", row=1, col=2)
fig.update_yaxes(title="t90 (weeks)", row=1, col=1)
fig.update_yaxes(title="tail mass", tickformat=".0%", row=1, col=2)
style(fig, 380, f"The truncation illusion — geometric α={alpha}")
fig.show()

print(f"α={alpha}: t90 at l_max=8 reads {t90s[0]:.1f} wk against an untruncated {t90s[-1]:.1f} wk")
print(f"mass discarded and re-spread inside the 8-lag window: {tails[0]:.1%}")
"""

PRIORS = r"""
# Two DEFENSIBLE default priors ship in this codebase for the same parameter:
#   AdstockConfig.geometric() -> Beta(1,3)   (prior-implied half-life ~0.5 wk)
#   bare AdstockConfig()      -> Beta(2,2)   (prior-implied half-life ~1.0 wk)
# On a short, weakly-informative world the DATA cannot arbitrate — and the
# honest output is a flagged verdict, not two equally confident horizons.
short = dgp.make_clean(seed=11, n_weeks=40)
SHORT_CSV = ART / "short.csv"
mff.scenario_to_mff(short).to_csv(SHORT_CSV, index=False)

def short_fit(alpha_prior=None):
    spec = {
        "kpi": "Sales", "kpi_level": "national",
        "media_channels": [
            {"name": c, "adstock": {"type": "geometric"}, "saturation": {"type": "hill"}}
            for c in short.channels
        ],
        "control_variables": [{"name": c} for c in short.controls.columns],
    }
    if alpha_prior:
        spec["priors"] = {"media": {c: {"adstock_alpha": alpha_prior}
                                    for c in short.channels}}
    m = build_model(spec, str(SHORT_CSV))
    m.fit(draws=500, tune=500, chains=2, random_seed=42)
    return m

m_13 = short_fit({"distribution": "beta", "params": {"alpha": 1, "beta": 3}})
m_22 = short_fit(None)  # graph default Beta(2,2)
r13, r22 = channel_payback(m_13), channel_payback(m_22)

rows = []
for ch in short.channels:
    a, b = r13.channels[ch], r22.channels[ch]
    rows.append({
        "channel": ch,
        "t50 under Beta(1,3)": f"{a.horizons['t50']['mean']:.2f}",
        "t50 under Beta(2,2)": f"{b.horizons['t50']['mean']:.2f}",
        "verdict (1,3)": a.learning_verdict,
        "verdict (2,2)": b.learning_verdict,
        "flagged?": "YES" if (a.prior_dominated or b.prior_dominated
                              or a.status == "downgraded" or b.status == "downgraded")
                    else "",
    })
print(pd.DataFrame(rows).to_string(index=False))
"""

MISSPEC = r"""
mis = dgp.make_adstock_misspec(seed=7, n_weeks=156)
MIS_CSV = ART / "misspec.csv"
mff.scenario_to_mff(mis).to_csv(MIS_CSV, index=False)
spec_mis = {
    "kpi": "Sales", "kpi_level": "national",
    "media_channels": [
        {"name": c, "adstock": {"type": "geometric"}, "saturation": {"type": "hill"}}
        for c in mis.channels
    ],
    "control_variables": [{"name": c} for c in mis.controls.columns],
}
m_mis = build_model(spec_mis, str(MIS_CSV))
m_mis.fit(draws=800, tune=800, chains=4, random_seed=42)
pb_mis = channel_payback(m_mis)

def mis_truth(ch, share):
    cum = np.asarray(mis.notes["true_adstock"][ch]["cum_share"], dtype=float)
    j = int(np.searchsorted(cum, share)); prev = cum[j-1] if j > 0 else 0.0
    return j + (share - prev) / (cum[j] - prev)

rows = []
for ch, p in pb_mis.channels.items():
    rows.append({
        "channel": ch, "status": p.status,
        "fitted t90": f"{p.horizons['t90']['mean']:.1f}",
        "planted t90 (Weibull, 26 lags)": f"{mis_truth(ch, 0.9):.1f}",
        "direction": "SHORT" if p.horizons["t90"]["mean"] < mis_truth(ch, 0.9) else "long",
    })
print(pd.DataFrame(rows).to_string(index=False))
print("\nautocorrelation gate:", {k: v for k, v in pb_mis.autocorrelation.items()})
print("\nrun-level caveat:\n ", pb_mis.caveats[0] if pb_mis.caveats else "none")
"""

REFUSAL = r"""
from mmm_framework.mmm_extensions.config import MediatorConfig, MediatorType, NestedModelConfig
from mmm_framework.mmm_extensions.models.nested import NestedMMM

n = 60
rng = np.random.default_rng(0)
media = np.abs(rng.normal(100, 20, (n, 2)))
idx = pd.date_range("2022-01-03", periods=n, freq="W-MON")
r = np.random.default_rng(1)
aware = 40 + 0.3 * media[:, 0] + r.normal(0, 4, n)
sales = 1000 + 4 * aware + 2 * media[:, 1] + r.normal(0, 40, n)
nested = NestedMMM(
    media, sales, ["TV", "Digital"],
    NestedModelConfig(mediators=(MediatorConfig(name="Awareness",
                                                mediator_type=MediatorType.FULLY_LATENT),)),
    index=idx,
)
nested.fit(method="map", random_seed=0)

pb_nested = channel_payback(nested)
for ch, p in pb_nested.channels.items():
    print(f"{ch}: {p.status}")
print("\nreason:\n ", list(pb_nested.channels.values())[0].reason)
"""

BREAKEVEN = r"""
from mmm_framework.planning import payback_breakeven, discount_weights
from mmm_framework.finance import UnresolvedValueError

# Refuses without a valuation — never a silent value_per_kpi=1.0.
try:
    payback_breakeven(mmm)
except UnresolvedValueError as e:
    print("without a valuation:", type(e).__name__, "—", e)

be = payback_breakeven(mmm, value_per_kpi=2.5, value_source="notebook-declared",
                       discount_rate_annual=0.10)
rows = []
for ch, b in be.items():
    rows.append({
        "channel": ch, "status": b.status,
        "break-even (wk)": (f"{b.breakeven_mean:.1f}" if b.breakeven_mean is not None else "—"),
        "90% interval": (f"[{b.breakeven_lower:.1f}, {b.breakeven_upper:.1f}]"
                          if b.breakeven_lower is not None else "—"),
        "P(never repays)": f"{b.prob_never:.0%}",
    })
print(pd.DataFrame(rows).to_string(index=False))

w = discount_weights(26, rate_annual=0.10)
print(f"\ndiscount_weights(26, 10%/yr): week-25 weight {w[-1]:.4f} "
      f"(mean haircut {1 - w.mean():.1%} — why the default rate is 0.0)")
"""

CELLS = [
    md(r"""
# The payback horizon — when a channel's effect actually lands

`planning.payback` answers the most CFO-legible question an MMM gets asked —
*"when does a dollar pay back?"* — and it does so knowing the question rests on
the model's **least identified** parameter. This notebook walks the whole
surface:

1. `channel_payback` on a world with a **sealed carryover answer key**
2. the **truncation illusion**, measured analytically
3. the **prior-domination gate** — two defensible priors, one flagged verdict
4. the **negative control** — a misspecified carryover window fires the gates
5. **refusals by name** — model families with no single kernel to read
6. the separately-named **finance-sense break-even**, with a valuation

The design rule throughout (issue #224): nothing renders without (a) a
per-draw interval, (b) a learning verdict, (c) the truncated tail mass, and
(d) a stated basis.
"""),
    code(SETUP),
    md(r"""
## 1 · A model, and the one named quantity

`t50`/`t90` are the interpolated lags at which the fitted carryover kernel
crosses 50%/90% of its total effect, computed **per posterior draw** — the
interval is the ETI of the transform, never the transform of the mean (the
crossing is convex in the decay rate, so those genuinely differ).
"""),
    code(FIT),
    code(BASIC),
    md(r"""
Everything the number needs travels with it: the kernel family and window, the
tail mass the window discards, whether the carryover parameters actually
learned from the data, and an autocorrelation gate that is quiet here because
this world's carryover is inside the model's family.
"""),
    md(r"""
### Against the sealed truth

`make_clean` plants its kernels and exports `notes["true_adstock"]` —
`cum_share[k]` is the fraction of effect landed by lag *k*, exactly what a
payback horizon reads.
"""),
    code(TRUTH),
    md(r"""
## 2 · The truncation illusion

The default `l_max=8` with `normalize=True` renormalizes the truncated kernel
to sum to one — the untruncated tail is silently redistributed **inside** the
window. That makes every horizon read off it structurally optimistic, and the
bias is largest exactly for the long-carryover brand channels the comparison
is usually about. `channel_payback` prints the tail mass with every result;
here is the illusion measured directly.
"""),
    code(TRUNCATION),
    md(r"""
## 3 · The prior may be the answer

Two default priors for the same decay parameter ship in this codebase —
`AdstockConfig.geometric()` uses Beta(1,3), a bare `AdstockConfig()` falls to
the graph's Beta(2,2) — with prior-implied half-lives differing **2x**. On a
short world the data cannot arbitrate, and the honest output is a *flagged
verdict*, not two equally confident horizons.
"""),
    code(PRIORS),
    md(r"""
## 4 · Negative control — a misspecified carryover window

`make_adstock_misspec` plants a 26-lag Weibull peaking ~6–8 weeks out; the
model fits geometric truncated at 8. Two things must happen: the horizons come
back **downgraded**, and the stated direction of bias is **short** — a
truncated kernel cannot express mass beyond its window.

Note which check catches it: the residual Ljung-Box does *not* fire here
(p≈0.16) while the posterior-predictive lag-1 autocorrelation check is extreme
(p≈0.01) — which is why the gate runs both.
"""),
    code(MISSPEC),
    md(r"""
## 5 · Refusals by name

Three families have no single carryover kernel to read a horizon from, and
each fails differently: extension models (one hardcoded geometric at fixed
`l_max`), structural models with AR(1) mediators (persistence lives in the
state's ρ, on a stated ridge with α), and dual-stock brand models (the
registered `adstock_alpha` is only the *fast* stock). A payback that cannot be
computed honestly comes back as a refusal with the mechanism named — not a
blank, and never a wrong number.
"""),
    code(REFUSAL),
    md(r"""
## 6 · The finance-sense break-even

A separately named quantity: the lag at which cumulative **discounted dollar
return** on a dollar of spend reaches 1, per draw. It needs a value per KPI
unit and refuses without one; draws that never repay inside the window are
reported as `P(never repays)` rather than dropped — on this world (planted
ROAS below 1) that probability is the headline, and hiding it would be the
exact trap this module exists to avoid.
"""),
    code(BREAKEVEN),
    md(r"""
## Where this surfaces

- **Reports** — the classic and Augur reports gain a payback section, the
  interactive report a card; all data-gated on the bundle's `payback` payload,
  which the extractor fills best-effort.
- **Agent** — the `get_payback_horizon` tool → `payback_horizon` model op;
  session export replays it as `run_op(model_ops.payback_horizon, ...)`.
- **REST** — `POST /projects/{id}/planner/payback` (job) +
  `GET .../payback/{job_id}` (poll).
- **Provenance** — `metadata.json` and the run-metrics history record basis /
  family / `l_max` / tail mass, so a reloaded model cannot silently re-derive
  a horizon on a different basis.

### Reading list

- `technical-docs/engineering-notes.md` — *Impression-/click-measured media*
  (why break-even refuses on efficiency channels) and the #218 half-life
  disambiguation this builds on.
- `nbs/demos/critique_to_decision.ipynb` — the same epistemics discipline
  applied across critique, spec search, optimization and experiments.
"""),
]


def main() -> None:
    nb = new_notebook(cells=CELLS)
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    out = "demos/payback_horizon.ipynb"
    with open(out, "w") as fh:
        nbformat.write(nb, fh)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
