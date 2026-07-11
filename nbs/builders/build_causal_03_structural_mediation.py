"""Author causal_03_structural_mediation.ipynb — Notebook 3 of 11.

    uv run --with nbformat python builders/build_causal_03_structural_mediation.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        causal/causal_03_structural_mediation.ipynb --ExecutePreprocessor.timeout=3600 \
        --ExecutePreprocessor.kernel_name=python3

Rung 3 — structural learning I: posit the brand funnel as a causal DAG
(TV -> awareness -> consideration -> sales, with a latent demand confounder)
and fit it jointly against binomial + Likert survey data with
StructuralNestedMMM. The expensive notebook of the series: the funnel fit is
full 4-chain NUTS (~12 min cold), cached as derived tables by causal_common
so re-bakes take seconds.
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
os.environ.setdefault("TQDM_DISABLE", "1")
import sys
for _p in ("../builders", "builders", os.path.join(os.getcwd(), "nbs", "builders")):
    if os.path.isfile(os.path.join(_p, "causal_common.py")) and _p not in sys.path:
        sys.path.insert(0, _p)

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
for _n in ("pymc", "pymc.sampling", "numpyro", "jax", "arviz"):
    logging.getLogger(_n).setLevel(logging.ERROR)

import causal_common as C
print(f"{C.BRAND} — {C.TAGLINE}")
"""


CELLS = [
    md(r"""
# Structural Learning I — The Brand Funnel
### TV → awareness → consideration → sales, learned from surveys

*Notebook 3 of 11 — Causal Inference in Practice.*

Veranda Home's CMO believes something no flat MMM can even *express*: that TV
doesn't sell patio sets this week — it builds **awareness**, awareness feeds
**consideration**, and consideration converts to sales over months. The media
team believes Display and Social work one step lower in the funnel. Search
just harvests.

Rung 2 ended with a warning: conditioning on a funnel variable *amputates* the
effect flowing through it. The fix is not to ignore the funnel — it's to
**model it**. This notebook posits the funnel as a structural causal model and
fits the whole system jointly:

- an **awareness** state with AR(1) persistence (brand stock decays slowly),
  observed through a weekly **survey**: `aware_count` out of `n` respondents —
  a *binomial* outcome with week-varying sample size;
- a **consideration** state driven by awareness, Display, Social, price, and a
  **latent demand** factor, observed through a 5-point **Likert** survey
  (ordered category counts);
- **sales**, driven by consideration and Search.

Everything — media effects, funnel edges, survey measurement, latent demand,
sales — is one PyMC graph (`StructuralNestedMMM`), so survey information
flows *backwards* to discipline the media coefficients, and uncertainty flows
*forward* from the surveys into every causal claim. And because this world is
synthetic (`make_brand_funnel`), every structural parameter we estimate has a
sealed true value to be graded against.
"""),
    code(SETUP),
    md(r"""
## 1 — The world: spend, sales, and two surveys

The funnel world ships the standard dashboard *plus* the two survey tracks a
real brand team would run. Note the awareness tracker's realism: it's missing
~25% of weeks, and the weekly sample size varies — some weeks 200 respondents,
some 800. A binomial measurement model turns that varying `n` into varying
precision instead of pretending every week is equally trustworthy.
"""),
    code(r"""
sc = C.funnel_world()
counts, trials = sc.notes["awareness_counts"], sc.notes["awareness_trials"]
cons = sc.notes["consideration_counts"]          # (n_weeks, 5) category counts
obs = ~np.isnan(counts)

print(f"{len(sc.weeks)} weeks | channels: {sc.channels}")
print(f"awareness tracker: {int(obs.sum())}/{len(counts)} weeks observed, "
      f"n per week {int(np.nanmin(trials))}–{int(np.nanmax(trials))}")
print(f"consideration Likert: {int((~np.isnan(cons).any(axis=1)).sum())} weeks, "
      f"5 categories")

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.36, 0.32, 0.32],
                    vertical_spacing=0.06,
                    subplot_titles=["media spend ($000s, stacked)",
                                    "awareness tracker (share aware, dot size = sample n)",
                                    "consideration Likert (share of top-2 box)"])
for c in sc.channels:
    fig.add_trace(go.Scatter(x=sc.weeks, y=sc.spend[c], name=c, stackgroup="s",
                             mode="lines", line=dict(width=0.4, color=C.PALETTE[c])),
                  row=1, col=1)
share = counts / trials
fig.add_trace(go.Scatter(x=sc.weeks[obs], y=share[obs], mode="markers",
                         name="aware share", marker=dict(color=C.PALETTE["awareness"],
                         size=3 + 8 * (trials[obs] / np.nanmax(trials)), opacity=0.7)),
              row=2, col=1)
top2 = cons[:, 3:].sum(axis=1) / cons.sum(axis=1)
fig.add_trace(go.Scatter(x=sc.weeks, y=top2, mode="markers", name="consider (top-2)",
                         marker=dict(color=C.PALETTE["consideration"], size=5, opacity=0.7)),
              row=3, col=1)
C.style(fig, height=680, title="The dashboard a brand team actually has", showlegend=True)
"""),
    md(r"""
## 2 — The wrong model first: a flat MMM

Hand this world to the standard four-channel MMM (surveys ignored — a flat
model has nowhere to put them). Two honest observations will matter later:

1. The flat model's **totals** are not crazy. Mediation by itself doesn't
   bias a total-effect estimate — the effect still flows from spend to sales.
   (The latent demand confounder does the damage here, mildly.)
2. The flat model has **no opinion** about anything the CMO actually asked:
   how long brand stock lasts, which funnel stage each channel moves, what a
   survey point is worth. Those aren't parameters it has.
"""),
    code(r"""
flat = C.fit_world("funnel_flat")
g_flat = flat["grade"]
print(g_flat[["true", "est", "rel_err", "covered"]].round(2).to_string())
C.truth_forest(g_flat, title="Flat MMM on the funnel world — totals OK, mechanism absent",
               height=380)
"""),
    md(r"""
## 3 — Positing the structure

Now say what we believe, as a graph. `StructuralNestedMMM` takes the funnel as
configuration: each mediator declares its drivers (channels, parent mediators,
controls, latent factors), its state dynamics, and its **measurement** —
binomial for the awareness tracker, cumulative-logit ordered for the Likert.
The latent demand factor is declared with no measurement of its own: it is
identified purely by the co-movement it induces between consideration and
sales.
"""),
    code(r"""
model, _ = C.build_funnel_model()
cfg = model.config
print("mediator DAG order:", cfg.topological_order())
for m in cfg.mediators:
    print(f"  {m.name}: channels={list(m.channels)} parents={list(m.parents)} "
          f"controls={list(m.controls)} factors={list(m.latent_factors)} "
          f"dynamics={m.dynamics.value} likelihood={m.measurement.likelihood.value}")

C.dag(
    nodes={"TV": (0.0, 1.0), "awareness": (0.3, 1.0), "consideration": (0.62, 0.55),
           "sales": (1.0, 0.55), "Display": (0.0, 0.55), "Social": (0.0, 0.1),
           "Search": (0.62, 0.0), "Price": (0.3, 0.1), "demand": (0.95, 1.05)},
    edges=[("TV", "awareness"), ("awareness", "consideration"),
           ("consideration", "sales"), ("Display", "consideration"),
           ("Social", "consideration"), ("Search", "sales"),
           ("Price", "consideration")],
    bad_edges=[("demand", "consideration"), ("demand", "sales")],
    title="The posited structural model (red: the latent demand confounder)",
    height=430,
)
"""),
    md(r"""
## 4 — The joint fit

This is the one genuinely expensive fit of the series: full NUTS, 4 chains ×
500 draws after 1000 tuning steps — the settings the framework's own recovery
test pins (approximate methods like MAP are *not* trustworthy for latent-state
models; the series never uses them here). `causal_common` caches the derived
tables, so only the first bake pays.
"""),
    code(r"""
f = C.fit_funnel()
meta = f["meta"]
print({k: round(v, 3) if isinstance(v, float) else v
       for k, v in meta.items() if k != "fit"})
assert meta["rhat_max"] < 1.1          # converged (the recovery-test bar)
"""),
    md(r"""
## 5 — Grading the structure itself

Here is what a flat MMM could never produce, graded parameter by parameter
against the sealed key: the brand-stock persistence, the TV→awareness edge,
the awareness→consideration edge, the price effect on consideration, and the
demand→consideration loading.
"""),
    code(r"""
tp = f["sc"].notes["true_params"]
recov = pd.DataFrame([
    ("awareness persistence ρ", "awareness_persistence", tp["rho_awareness"]),
    ("TV → awareness", "beta_TV_to_awareness", tp["b_tv_to_awareness"]),
    ("awareness → consideration", "lambda_awareness_to_consideration",
     tp["lambda_awareness_to_consideration"]),
    ("price → consideration", "phi_Price_to_consideration",
     tp["phi_price_to_consideration"]),
    ("demand → consideration", "w_demand_to_consideration",
     tp["w_demand_to_consideration"]),
], columns=["parameter", "var", "true"])
recov["post_mean"] = recov["var"].map(lambda v: f["params"][v]["mean"])
recov["lo"] = recov["var"].map(lambda v: f["params"][v]["lo"])
recov["hi"] = recov["var"].map(lambda v: f["params"][v]["hi"])
recov["covered"] = (recov["lo"] <= recov["true"]) & (recov["true"] <= recov["hi"])
print(recov[["parameter", "true", "post_mean", "lo", "hi", "covered"]]
      .round(3).to_string(index=False))

fig = go.Figure()
for i, r in recov.iterrows():
    fig.add_trace(go.Scatter(x=[r["lo"], r["hi"]], y=[i, i], mode="lines",
                             line=dict(color=C.PALETTE["consideration"], width=5),
                             opacity=0.5, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[r["post_mean"]], y=[i], mode="markers",
                             marker=dict(color=C.PALETTE["consideration"], size=11),
                             name="posterior (90%)", showlegend=(i == 0)))
    fig.add_trace(go.Scatter(x=[r["true"]], y=[i], mode="markers",
                             marker=dict(color=C.TRUTH, symbol="line-ns-open", size=18,
                                         line=dict(width=3)),
                             name="sealed truth", showlegend=(i == 0)))
fig.update_yaxes(tickvals=list(range(len(recov))), ticktext=recov["parameter"],
                 autorange="reversed")
C.style(fig, title="Structural parameters vs the sealed key", height=420)
"""),
    code(r"""
# All five structural parameters covered by their 90% intervals — the model
# didn't just fit the data, it recovered the MECHANISM that generated it.
assert bool(recov["covered"].all())

rho = f["params"]["awareness_persistence"]["mean"]
half_life = np.log(0.5) / np.log(rho)
assert 3.0 < half_life < 7.0     # truth: ρ=0.85 → ~4.3 weeks
print(f"Brand-stock half-life: {half_life:.1f} weeks (ρ = {rho:.3f}, true 0.85).")
print("That number is a media-planning decision — TV flighting cadence — that "
      "no flat MMM could have produced.")
"""),
    md(r"""
## 6 — The latent states, recovered

The model also had to infer three *time series* it never observed directly:
the true awareness probability (through the noisy, gappy binomial tracker),
the latent consideration level (through the Likert counts), and — hardest —
the **latent demand factor**, which has no survey at all.
"""),
    code(r"""
L = f["latents"]
w = f["sc"].weeks
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                    subplot_titles=[
                        f"awareness probability (corr with truth: {meta['corr_awareness']:.2f})",
                        f"consideration latent (corr: {meta['corr_consideration']:.2f})",
                        f"latent demand — NO survey exists (|corr|: {meta['corr_demand']:.2f})"])
for row, (m, lo, hi, tr, col) in enumerate([
        ("p_aw_mean", "p_aw_lo", "p_aw_hi", "p_aw_true", C.PALETTE["awareness"]),
        ("z_cons_mean", "z_cons_lo", "z_cons_hi", "z_cons_true", C.PALETTE["consideration"]),
        ("demand_mean", "demand_lo", "demand_hi", "demand_true", C.PALETTE["demand"])],
        start=1):
    fig.add_trace(go.Scatter(x=w, y=L[hi], line=dict(width=0), showlegend=False,
                             hoverinfo="skip"), row=row, col=1)
    fig.add_trace(go.Scatter(x=w, y=L[lo], fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(120,130,150,0.25)", showlegend=False,
                             hoverinfo="skip"), row=row, col=1)
    fig.add_trace(go.Scatter(x=w, y=L[m], line=dict(color=col, width=2),
                             name="posterior", showlegend=(row == 1)), row=row, col=1)
    tr_v = L[tr]
    if row == 3:  # demand is identified up to sign/scale — align for display
        s = np.sign(np.corrcoef(L[m], tr_v)[0, 1])
        tr_v = (tr_v - tr_v.mean()) / tr_v.std() * L[m].std() * s + L[m].mean()
    fig.add_trace(go.Scatter(x=w, y=tr_v, line=dict(color=C.TRUTH, width=1.4,
                             dash="dot"), name="truth", showlegend=(row == 1)),
                  row=row, col=1)
C.style(fig, height=760, title="Three unobserved series, recovered through their footprints")
"""),
    code(r"""
assert meta["corr_awareness"] > 0.7      # recovery-test bars
assert meta["corr_consideration"] > 0.7
assert meta["corr_demand"] > 0.6
"""),
    md(r"""
## 7 — The mediation decomposition

The headline for the CMO: **how much of each channel's effect flows through
the funnel?** These are exact posterior counterfactuals — the model re-runs
its own world with each channel off, holding the inferred shocks fixed — not
back-of-envelope path products. The sealed key says TV, Display and Social are
*fully* mediated (share = 1.0) and Search not at all (0.0).
"""),
    code(r"""
me = f["mediation"].set_index("channel")
true_share = f["sc"].notes["mediated_share"]
show = me[["direct_effect", "total_indirect", "total_effect", "proportion_mediated"]].copy()
show["true_share"] = pd.Series(true_share)
print(show.round(2).to_string())

chs = list(me.index)
fig = go.Figure()
fig.add_trace(go.Bar(x=chs, y=[me.loc[c, "direct_effect"] for c in chs],
                     name="direct", marker_color=C.MUTED, opacity=0.85))
fig.add_trace(go.Bar(x=chs, y=[me.loc[c, "total_indirect"] for c in chs],
                     name="through the funnel", marker_color=C.PALETTE["awareness"],
                     opacity=0.9))
fig.update_layout(barmode="relative")
fig.update_yaxes(title="contribution to sales (KPI units)")
C.style(fig, title="Where each channel's effect flows", height=420)
"""),
    code(r"""
# The design-review guard the framework itself tests: mediated totals must be
# genuinely nonzero for the brand channels (an in-graph centering bug would
# make them identically zero), and Search must show ~no mediated path.
for ch in ("TV", "Display"):
    assert me.loc[ch, "total_effect"] > 0
    assert me.loc[ch, "proportion_mediated"] > 0.25       # truth: 1.0
assert abs(me.loc["Search", "proportion_mediated"]) < 0.05  # truth: 0.0
print("The funnel story survives grading: brand channels flow through the "
      "funnel; Search doesn't. (TV's share posterior even brackets 1.0 — its "
      "direct effect is indistinguishable from zero.)")
"""),
    md(r"""
## 8 — What the structure bought (and what it didn't)

Put the two models side by side honestly:

| question | flat MMM | structural model | sealed truth |
|---|---|---|---|
| total effects | respectable | comparable, wider (honest) | ✓ graded above |
| how long does brand stock last? | *(no such parameter)* | ρ ⇒ ~4-week half-life | 0.85 ⇒ 4.3 wks |
| which stage does each channel move? | *(no such parameter)* | TV→awareness; Display/Social→consideration | ✓ |
| mediated share | *(no such parameter)* | ≈1.0 brand, 0 Search | 1.0 / 0.0 |
| what is a survey worth? | ignored | week-precision weighted, backwards-flowing | — |

The structural model is **not** a magic accuracy upgrade on totals — its
total-effect intervals are *wider* than the flat model's, because it is
honest about survey noise and latent-state uncertainty. What it buys is the
**mechanism**: the parameters the CMO's actual decisions (flighting cadence,
funnel-stage investment, survey program budget) depend on. If you only need
totals, rung 2 was enough. If your decisions mention the funnel, only
structure answers the question you're asking.
"""),
    code(r"""
C.write_artifact("causal_03_mediation_recovery.json", dict(
    rhat_max=float(meta["rhat_max"]),
    fit_seconds=float(meta["seconds"]),
    structural_recovery={r["var"]: dict(true=float(r["true"]),
                                        post=float(r["post_mean"]),
                                        covered=bool(r["covered"]))
                         for _, r in recov.iterrows()},
    half_life_weeks=float(half_life),
    latent_corr=dict(awareness=float(meta["corr_awareness"]),
                     consideration=float(meta["corr_consideration"]),
                     demand=float(meta["corr_demand"])),
    proportion_mediated={c: float(me.loc[c, "proportion_mediated"]) for c in me.index},
    true_mediated_share={k: float(v) for k, v in true_share.items()},
    flat_grade={c: float(g_flat.loc[c, "rel_err"]) for c in g_flat.index},
))
print("artifact written: causal_03_mediation_recovery.json")
"""),
    md(r"""
---
**Next — [04 · Latent confounders](causal_04_latent_confounders.ipynb):** the
other structural move. When the confounder itself is invisible but leaves
fingerprints on things you *can* measure (GDP growth, consumer confidence,
unemployment...), you can model the measurement — and close a back-door no
control variable quite reaches.
"""),
]


def main():
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    out = "causal/causal_03_structural_mediation.ipynb"
    with open(out, "w") as f:
        nbformat.write(nb, f)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
