"""Author causal_02_mmm_as_causal_model.ipynb — Notebook 2 of 11.

    uv run --with nbformat python builders/build_causal_02_mmm_as_causal_model.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        causal/causal_02_mmm_as_causal_model.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

Rung 2: treat the fitted MMM as a causal model with explicit estimands, on the
`realistic` seven-channel world. Estimand taxonomy (counterfactual contribution
/ average ROAS / marginal ROAS — each graded against the key, including the
marginal), the bad-control trap live, the refutation battery + parameter-
learning trust checks (and what they can't catch), and the Radio/Print
collinear ridge — the model's honest "I don't know" that nb07 resolves. Uses
the cached real_causal/real_bad fits; the refutation suite refits in-notebook
(~2-4 min).
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
# The MMM as a Causal Model
### Estimands, bad controls, refutation — and the dignity of "I don't know"

*Notebook 2 of 11 — Causal Inference in Practice.*

Rung 1 taught adjustment on a four-channel toy. Veranda Home's real modeling
problem looks nothing like a toy: **seven channels** (two of them weak and
bought on the same calendar), **thirteen candidate controls** (two genuine
confounders, four precision variables, six irrelevant series someone exported
from the warehouse, and one variable that will quietly sabotage the model if
you include it), demand-chasing budgets, and realistically low media
signal-to-noise.

This is the framework's `realistic` synthetic world, and this notebook treats
the fitted MMM the way a careful practitioner should — as a **causal model
making explicit claims**:

1. Every reported number is an **estimand** — a named counterfactual question.
   We grade three of them (total contribution, average ROAS, *and* marginal
   ROAS) against the sealed key.
2. Control selection is **causal homework**, not a kitchen sink. We spring the
   **bad-control trap** on purpose and measure the damage.
3. Trust is **earned by attack** — refutation tests and prior→posterior
   learning diagnostics — and we're explicit about what those checks *cannot*
   catch.
4. Some questions the data **cannot answer**; the model's job is to say so
   with wide intervals, not to invent a number. That honesty sets up the
   experiments of nb05–07.
"""),
    code(SETUP),
    md(r"""
## 1 — The world on the analyst's desk

The data dictionary below is the analyst's causal homework, done: each control
tagged by *why* it belongs (or doesn't) in the model. Two things deserve
special note — `brand_awareness` is a **mediator** (TV and Video build it, it
drives sales: conditioning on it would block the very effect we're estimating),
and Radio/Print are bought on **one flighting calendar** (Print's spend is a
scaled copy of Radio's).
"""),
    code(r"""
sc = C.scenario_for("real_causal")
C.check_truth(sc)

roles = C.world("realistic").notes["roles"]
dd = pd.DataFrame(
    [(k, v) for k, v in roles.items()], columns=["variable", "causal role"]
).sort_values("causal role").reset_index(drop=True)
print(dd.to_string(index=False))

spend_share = (sc.spend.sum() / sc.spend.sum().sum()).rename("spend share")
print()
print(spend_share.map("{:.0%}".format).to_string())
print(f"\ncorr(Radio spend, Print spend) = "
      f"{np.corrcoef(sc.spend['Radio'], sc.spend['Print'])[0,1]:.3f}")
"""),
    md(r"""
## 2 — The causal fit

The model gets the defensible control set: both confounders (marked
`CONFOUNDER`, so they take wide un-shrunk priors), the four precision
controls, the six irrelevant series (a real warehouse export always has them —
the model should shrink them to zero on its own), and **not** the mediator.
"""),
    code(r"""
fit = C.fit_world("real_causal")
mmm, g = fit["model"], fit["grade"]
print(g[["true", "est", "rel_err", "covered"]].round(2).to_string())
C.truth_forest(g, title="Counterfactual contribution vs the sealed key — the causal fit",
               height=460)
"""),
    code(r"""
# Mostly honest: five of seven channels within ~±20% and covered. The residual
# wrinkle is TV — the hardest demand-chaser AND a mediated channel — still
# under-read even with the right controls (the demand index is a noisy proxy;
# rung 1's lesson compounds at scale). Structure (nb03) and experiments
# (nb05+) exist precisely for what adjustment leaves behind.
assert (g["rel_err"].abs() < 0.45).all()          # nobody is wildly invented
assert int(g["covered"].sum()) >= 5               # most intervals honest
print(f"TV residual bias: {g.loc['TV','rel_err']:+.0%} "
      f"(covered: {bool(g.loc['TV','covered'])})")
"""),
    md(r"""
## 3 — Three estimands, three different questions

"What's the ROAS?" is not one question. The three below have different
answers, different uses, and different failure modes — a report that doesn't
say which one it's quoting is not yet a claim:

| estimand | counterfactual question | decision it serves |
|---|---|---|
| **Total contribution** | "What if this channel had never run?" | annual planning, PR |
| **Average ROAS** | contribution ÷ dollars spent | budget *defense* |
| **Marginal ROAS** | "What does the *next* 10% buy?" | budget *allocation* |

Because our worlds ship an executable truth, we can grade **all three** —
including the marginal, which almost never gets graded in real life.
"""),
    code(r"""
marg = mmm.compute_marginal_contributions(
    spend_increase_pct=10.0, compute_uncertainty=False, random_seed=0
).set_index("Channel")

# True marginal ROAS from the DGP: nudge each channel's whole spend path +10%
# and difference the true response (carryover included).
true_marg = {}
for c in sc.channels:
    x = sc.spend[c].to_numpy(float)
    lift = (C.true_media_term(sc, c, 1.10 * x) - C.true_media_term(sc, c, x)).sum()
    true_marg[c] = float(lift / (0.10 * x.sum()))

est = pd.DataFrame({
    "avg ROAS (est)": g["est_roas"],
    "avg ROAS (true)": g["true_roas"],
    "marginal ROAS (est)": marg["Marginal ROAS"],
    "marginal ROAS (true)": pd.Series(true_marg),
}).loc[sc.channels]
print(est.round(2).to_string())
"""),
    code(r"""
# Saturation makes the marginal LESS than the average for every channel — in
# truth and in the estimate. Ranking by average ROAS and allocating by it is
# a category error this table makes visible.
assert (est["marginal ROAS (true)"] < est["avg ROAS (true)"]).all()
assert (est["marginal ROAS (est)"] < est["avg ROAS (est)"]).all()

fig = go.Figure()
for col, color, sym in [("avg ROAS (est)", C.MUTED, "circle"),
                        ("marginal ROAS (est)", C.INK, "diamond")]:
    fig.add_trace(go.Scatter(x=est.index, y=est[col], mode="markers", name=col,
                             marker=dict(size=13, color=color, symbol=sym)))
for col, dashc in [("avg ROAS (true)", C.GOOD), ("marginal ROAS (true)", C.BAD)]:
    fig.add_trace(go.Scatter(x=est.index, y=est[col], mode="markers", name=col,
                             marker=dict(size=17, color=dashc, symbol="line-ew-open",
                                         line=dict(width=3))))
fig.update_yaxes(title="ROAS ($ per $)")
C.style(fig, title="Average vs marginal — two different questions, graded separately",
        height=440)
"""),
    md(r"""
## 4 — The bad-control trap, sprung on purpose

`brand_awareness` correlates beautifully with sales. A naive modeler adds it
("more signal!"). But it is a **mediator**: TV and Video work *through* it. In
a total-effect model, conditioning on a mediator amputates exactly the effect
you're paid to measure.

The framework refuses a control explicitly marked `MEDIATOR` at build time —
so we do what the naive modeler does and add it *unmarked*:
"""),
    code(r"""
fit_bad = C.fit_world("real_bad")
g_bad = fit_bad["grade"]

med = ["TV", "Video"]
tbl = pd.DataFrame({
    "true": g.loc[med, "true"],
    "causal fit": g.loc[med, "est"],
    "with mediator": g_bad.loc[med, "est"],
})
tbl["amputated"] = 1 - tbl["with mediator"] / tbl["causal fit"]
print(tbl.round(2).to_string())

fig = go.Figure()
fig.add_trace(go.Bar(x=med, y=tbl["causal fit"], name="causal control set",
                     marker_color=C.GOOD, opacity=0.85))
fig.add_trace(go.Bar(x=med, y=tbl["with mediator"], name="+ brand_awareness (mediator)",
                     marker_color=C.BAD, opacity=0.85))
fig.add_trace(go.Scatter(x=med, y=tbl["true"], mode="markers", name="sealed truth",
                         marker=dict(color=C.TRUTH, symbol="line-ew-open", size=24,
                                     line=dict(width=3))))
fig.update_layout(barmode="group")
fig.update_yaxes(title="total contribution (KPI units)")
C.style(fig, title="Conditioning on the funnel blocks the funnel", height=420)
"""),
    code(r"""
# Both mediated channels lose a material slice of their measured effect the
# moment the mediator enters the control set.
assert tbl.loc["TV", "with mediator"] < 0.90 * tbl.loc["TV", "causal fit"]
assert tbl.loc["Video", "with mediator"] < 0.90 * tbl.loc["Video", "causal fit"]
print("A variable can be predictive, well-measured, statistically significant — "
      "and still make the model causally worse. Roles first, R² later.")
"""),
    md(r"""
## 5 — Earning trust by attack: the refutation battery

A causal claim you haven't tried to break isn't a claim, it's a hope. The
framework's refutation suite attacks the fitted model four ways — each test
refits the model on perturbed data:

- **placebo treatment** (permuted spend): the measured effect should *vanish*;
- **negative-control outcome**: an outcome the media can't touch should show
  *no* effect;
- **random common cause**: adding a synthetic confounder-lookalike should
  *not move* the estimates;
- **data subset**: the story should survive on 80% of the weeks.
"""),
    code(r"""
from mmm_framework.validation import ModelValidator, ValidationConfigBuilder

# The refutation suite refits the model several times; silence the samplers'
# advisory chatter (CRITICAL: pymc logs its convergence advisories at ERROR).
for _n in ("pymc", "pymc.sampling", "pymc.sampling.mcmc", "pymc.stats",
           "pymc.stats.convergence", "numpyro", "arviz"):
    _lg = logging.getLogger(_n); _lg.setLevel(logging.CRITICAL); _lg.propagate = False

vcfg = (
    ValidationConfigBuilder()
    .quick()
    .with_causal_refutation(draws=300, tune=300, chains=2)
    .build()
)
summary = ModelValidator(mmm).validate(vcfg)
ref = summary.causal_refutation
reft = ref.summary()
print(reft.to_string(index=False))
print(f"\npassed {ref.n_passed}/{len(ref.tests)}"
      + ("  (underpowered)" if ref.underpowered else ""))
"""),
    code(r"""
# The battery must RUN and mostly pass on a defensible fit — but note what it
# grades: internal consistency under perturbation. It re-fits the same
# structure on perturbed data; it cannot see that TV's point estimate carries
# residual confounding bias, because the perturbations preserve the very
# back-door that causes it. Refutation is a lie detector for broken models,
# not a truth detector for biased worlds.
assert len(ref.tests) >= 3
assert ref.n_passed >= len(ref.tests) - 1
print("Refutation ≠ verification: a model can pass every internal check "
      "and still be wrong about the world. (That's what experiments are for.)")
"""),
    md(r"""
## 6 — Did the data actually teach the model anything?

The second trust check: compare each parameter's prior to its posterior. A
parameter whose posterior is just its prior echoed back
(**prior-dominated**) is a number the *analyst* chose, not one the data
produced — and it deserves an asterisk in any deck.
"""),
    code(r"""
lrn = mmm.compute_parameter_learning(prior_samples=2000)
media_lrn = lrn[lrn["parameter"].str.match(r"beta_(TV|Search|Social|Display|Video|Radio|Print)$")]
print(media_lrn[["parameter", "prior_mean", "post_mean", "contraction", "verdict"]]
      .round(3).to_string(index=False))
"""),
    code(r"""
# The diagnostic's headline here: beta_TV carries the RELOCATION signature —
# the lowest contraction of any media beta, paired with the largest
# prior→posterior shift: the data pulled the coefficient far to one side
# without narrowing it. That is the model waving a small flag over exactly
# the channel we know (from the key) carries residual bias.
tv_row = media_lrn.set_index("parameter").loc["beta_TV"]
assert tv_row["contraction"] == media_lrn.set_index("parameter")["contraction"].min()
print(f"beta_TV: contraction={tv_row['contraction']:.2f}, verdict={tv_row['verdict']!r}")
"""),
    md(r"""
## 7 — The dignity of "I don't know": the Radio/Print ridge

Radio and Print carry real, material effects in truth — but their spend moves
in lockstep (corr ≈ 0.996). Observational data can identify their **combined**
effect; the **split** between them is simply not in the data. The honest
output is exactly what the model produces: enormous intervals on each.
"""),
    code(r"""
rel_width = ((g["hi"] - g["lo"]) / g["est"].abs()).rename("interval ÷ estimate")
print(rel_width.round(2).sort_values(ascending=False).to_string())

fig = go.Figure(go.Bar(
    x=rel_width.index, y=rel_width.values,
    marker_color=[C.BAD if c in ("Radio", "Print") else C.MUTED for c in rel_width.index],
    opacity=0.9))
fig.update_yaxes(title="90% interval width ÷ |estimate|")
C.style(fig, title="The model KNOWS it doesn't know Radio vs Print", height=400)
"""),
    code(r"""
# The two collinear channels carry — by far — the widest relative intervals.
top2 = set(rel_width.sort_values(ascending=False).index[:2])
assert top2 == {"Radio", "Print"}
assert float(np.corrcoef(sc.spend["Radio"], sc.spend["Print"])[0, 1]) > 0.95

C.write_artifact("causal_02_estimands.json", dict(
    grade={c: dict(true=float(g.loc[c, "true"]), est=float(g.loc[c, "est"]),
                   rel_err=float(g.loc[c, "rel_err"]), covered=bool(g.loc[c, "covered"]))
           for c in g.index},
    marginal={c: dict(est=float(est.loc[c, "marginal ROAS (est)"]),
                      true=float(est.loc[c, "marginal ROAS (true)"])) for c in est.index},
    bad_control={c: dict(causal=float(tbl.loc[c, "causal fit"]),
                         with_mediator=float(tbl.loc[c, "with mediator"]),
                         true=float(tbl.loc[c, "true"])) for c in tbl.index},
    refutation=dict(n_tests=int(len(ref.tests)), n_passed=int(ref.n_passed)),
    ridge=dict(radio_print_spend_corr=float(np.corrcoef(sc.spend["Radio"],
                                                        sc.spend["Print"])[0, 1]),
               rel_width={c: float(rel_width[c]) for c in rel_width.index}),
))
print("artifact written: causal_02_estimands.json")
"""),
    md(r"""
## What this rung bought

A fitted MMM, treated properly, is a **structured set of causal claims**:
named estimands, a defended control set, checks that attack the model from
inside, and wide intervals where the data is silent. On the realistic world
that discipline gets five of seven channels right and — crucially — *labels*
the places it can't settle: TV's residual bias (invisible internally, flagged
only obliquely by the relocation diagnostic) and the Radio/Print split (loudly
visible as variance).

Both open questions have the same answer, and it isn't a better prior:

- for channels that work **through a funnel**, model the funnel →
  **[03 · Structural mediation](causal_03_structural_mediation.ipynb)**;
- for what no structure can settle, **buy information** with an experiment →
  notebooks 05–07, where a single Radio lift test will snap the ridge shut.
"""),
]


def main():
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    out = "causal/causal_02_mmm_as_causal_model.ipynb"
    with open(out, "w") as f:
        nbformat.write(nb, f)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
