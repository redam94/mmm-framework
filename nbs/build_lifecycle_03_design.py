"""Author lifecycle_03_design.ipynb — T2 of the series (run from ``nbs/``).

    uv run --with nbformat python build_lifecycle_03_design.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        lifecycle_03_design.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

T2 · Design: pre-register a test for Display that is (a) POWERED to detect a real
effect and (b) worth its short-term cost. National flighting vs the geo-holdout
gold standard, model-anchored power, opportunity-cost economics, the
autocorrelation false-positive lesson, and the Pareto-front optimizer — all on
Northwind's live baseline. Shared world/fit/palette live in
``nbs/lifecycle_common.py``.
"""

from __future__ import annotations

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


# The canonical setup cell — verbatim across the series.
SETUP = r"""
import warnings; warnings.filterwarnings("ignore")
import os
os.environ.setdefault("TQDM_DISABLE", "1")
import sys
for _p in (".", os.path.join(os.getcwd(), "nbs"), os.path.dirname(os.getcwd())):
    if os.path.isfile(os.path.join(_p, "lifecycle_common.py")) and _p not in sys.path:
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

import lifecycle_common as L
print(f"{L.BRAND} — {L.TAGLINE}")

# ── The pieces T2 works from: the fitted baseline + a CSV spine. The design and
#    economics helpers read an MFF *csv* (not the in-memory panel), so we
#    materialize one alongside the cached fit.
csv = L.national_csv()
base = L.fit_baseline()
model, truth = base["model"], base["truth"]
ch = L.FOCUS_CHANNEL
first, last = L.dataset_period(base["mff_df"])
print(f"Designing a test for {ch} on {L.KPI}.  National spine: {first} -> {last}.")
assert ch in list(model.channel_names)          # the channel the loop chose is fittable
"""


CELLS = [
    md(r"""
# T2 · Design — a test that is *powered* and *affordable*
### Notebook 3 of 7 — the Experimental Measurement Lifecycle

**Recap (from `lifecycle_02_prioritize`).** T1 crossed **information** (EIG)
against **value of information** (EVOI) and returned one unambiguous verdict:
**Display** is the single channel that is both *uncertain* and *budget-critical* —
the only `test_now` in the portfolio. So Display earns the first experiment. But a
good idea we can't *measure* is worth nothing, and every test spends real money
and real weeks. T2 turns "test Display" into a concrete, defensible design.

Two questions gate any experiment, and we answer both **from the model, before
spending a dollar:**

1. **Is it powered?** Will the design actually *detect* an effect of the size the
   model expects — or is its **minimum detectable effect (MDE)** so large the test
   comes back "inconclusive" no matter what's true?
2. **Is it worth it?** What does deviating from business-as-usual **cost** in the
   short term, and does the learning justify the spend?

And one discipline ties it together: **pre-registration.** We lock the
**estimand**, the **method**, the **MDE**, and the **decision rule** *before* the
test runs. Pre-specifying the analysis is what kills **researcher degrees of
freedom** — the freedom to keep re-cutting the data until a flattering number
appears. A pre-registered null is an honest null.
"""),
    code(SETUP),
    md(r"""
## Option A — a national flighting pulse (what the spine data supports)

Northwind's baseline is fit on **national** weekly data — one row per week, no
geographies. With national data the only experiment you can run is a
**flighting** test: take today's budget and shove it **up and down on a
randomized schedule** so the model can finally tell the channel apart from the
season. It's **budget-neutral** — no new money — which is its whole appeal and its
whole weakness. The manufactured variance is small, so the design is a **weak
instrument**: its **MDE** (the smallest ROAS it can reliably detect) is large.
""") ,
    code(r"""
from mmm_framework.planning.design import design_experiment

# National data -> design_experiment auto-recommends the flighting family.
dz = design_experiment(csv, L.KPI, ch)          # design_key == 'national_flighting'
print(f"design:   {dz['design_key']}  ({dz['design_type']})")
print(f"duration: {dz['duration']} weeks   budget-neutral: {dz['budget_neutral']}")
print(f"SE(ROAS): {dz['se_roas']:.2f}    MDE(ROAS): {dz['mde_roas']:.2f}")
print(f"exogenous share of window spend variance: {dz['identification']['exogenous_share']:.0%}")

sched = pd.DataFrame(dz["schedule"])
fig = go.Figure(go.Bar(x=sched["week_offset"], y=sched["multiplier"],
                       marker_color=L.PALETTE[ch], showlegend=False))
fig.add_hline(y=1.0, line=dict(color=L.INK, width=1, dash="dot"),
              annotation_text="business-as-usual spend", annotation_position="top left")
L.style(fig, height=340, title=f"National flighting schedule for {ch} — budget-neutral on/off pulses")
fig.update_layout(xaxis_title="week of test window", yaxis_title="spend  ×  BAU")
fig.show()
print(f"\nThe MDE is well above a {dz['mde_roas']:.1f}x return — larger than any plausible truth.")
print("A budget-neutral national pulse simply cannot power a Display test.")
assert dz["mde_roas"] > 1        # national flighting is a weak instrument
"""),
    md(r"""
## Option B — a geo holdout (the gold standard)

The clean way to measure Display is to **create real variation across markets**:
take the brand's DMA-level panel, **match markets into pairs**, randomly send one
of each pair **dark** on Display, and read the **difference-in-differences**
against its still-spending twin. Because the treated markets genuinely stop
spending, the contrast is large and the **MDE collapses** — this is a strong
instrument, and it needs no model at all to compute (pure pandas).

Two design details do the heavy lifting:

- **Matching on the model's residuals, not raw KPI.** Raw KPI correlation is a
  **trap**: two markets look like a perfect pair because they share the same
  national trend and seasonality — but a DiD's *noise* lives in the
  **idiosyncratic** movement that's left *after* removing that shared structure.
  So we match on **residual co-movement**, and report both so you can see the gap.
- **A placebo band.** Slide the estimator over history where *nothing happened*;
  the "lifts" it finds by chance are the **falsification bar** — a real readout
  has to clear it.
"""),
    code(r"""
from mmm_framework.planning import design as D

gcsv, _ = L.geo_csv()
frame = D.load_design_frame(gcsv, L.KPI, ch)
gl = D.geo_lift_design(gcsv, L.KPI, ch, design="holdout", n_pairs=4, duration=8, randomize=True)

asg = pd.DataFrame(gl["assignment"])[["treatment", "control", "correlation", "residual_correlation"]]
print("randomized geo holdout — treated markets go dark on Display, controls keep spending:")
print(asg.round(3).to_string(index=False))
print(f"\nSE(ROAS): {gl['se_roas']:.2f}    MDE(ROAS): {gl['mde_roas']:.2f}    (SE source: {gl['se_source']})")
print(f"placebo falsification band: p95 |lift-by-chance| = {gl['placebo']['p95_abs']:.0f} "
      f"{L.KPI}-units over {gl['placebo']['n_windows']} historical windows")

# Matching diagnostic: raw correlation is a TRAP; residual co-movement drives DiD precision.
mp = pd.DataFrame(D.matched_pairs(frame["kpi_wide"], n_pairs=4, spend_wide=frame["spend_wide"]))
mp = mp[["geo_a", "geo_b", "correlation", "residual_correlation"]]

pc = pd.DataFrame(gl["power_curve"])
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.16,
    subplot_titles=("Power curve — longer test, smaller detectable effect",
                    "Minimum detectable effect: national vs geo"))
fig.add_trace(go.Scatter(x=pc["duration"], y=pc["mde_roas"], mode="lines+markers",
    line=dict(color=L.PALETTE[ch], width=3), marker=dict(size=8), showlegend=False), row=1, col=1)
fig.add_vline(x=gl["duration"], line=dict(color=L.MUTED, width=1.5, dash="dash"),
    annotation_text=f"chosen {gl['duration']}w", row=1, col=1)
fig.add_trace(go.Bar(x=["national<br>flighting", "geo<br>holdout"],
    y=[dz["mde_roas"], gl["mde_roas"]], marker_color=[L.MUTED, L.PALETTE[ch]],
    text=[f"{dz['mde_roas']:.2f}", f"{gl['mde_roas']:.2f}"], textposition="outside",
    showlegend=False), row=1, col=2)
fig.update_xaxes(title_text="test duration (weeks)", row=1, col=1)
fig.update_yaxes(title_text="MDE (ROAS)", row=1, col=1)
fig.update_yaxes(title_text="MDE (ROAS)", row=1, col=2)
L.style(fig, height=430, title=f"{ch}: the geo holdout is a far sharper instrument")
fig.show()
print("\nmatched pairs — raw correlation flatters on shared seasonality; the RESIDUAL")
print("co-movement is what the DiD's noise actually depends on:")
print(mp.round(3).to_string(index=False))
assert gl["mde_roas"] < dz["mde_roas"]      # the geo holdout is far more powered
"""),
    md(r"""
## Is it powered? — the model's own effect, versus the design's blind spot

An MDE is only meaningful *relative to the effect you expect*. So we ask the
**fitted model** for its best guess at Display's **incremental ROAS** — a full
posterior, `model_anchored_effect` — and check whether the national design could
actually **see** an effect that size. `powered_to_detect` returns a blunt verdict:
`powered`, `underpowered`, `overpowered`, or `inconclusive`. Watching the national
pulse fail this test honestly is the point — it's *why* the pre-registered plan
will be the geo holdout, not this.
"""),
    code(r"""
from mmm_framework.planning import design_anchor as DA

eff = DA.model_anchored_effect(model, dz, max_draws=100, random_seed=42)
ver = DA.powered_to_detect(eff, dz.get("power_curve"), int(dz["duration"]), float(dz["se_roas"]))

lo, hi = eff["incremental_roas_hdi"]
med = eff["incremental_roas_median"]
mde = ver["mde_roas"]
pdet = ver["prob_detectable"]; asur = ver["assurance"]
_p = lambda v: f"{v:.0%}" if v is not None else "n/a"
print(f"model-anchored incremental ROAS:  median {med:.2f}   (90% HDI {lo:.2f} .. {hi:.2f})")
print(f"average ROAS at current spend (context): {eff['roas_at_current_median']:.2f}")
print(f"design MDE(ROAS): {mde:.2f}   ->   VERDICT: {ver['verdict'].upper()}")
print(f"P[the effect is detectable]: {_p(pdet)}    signed assurance: {_p(asur)}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=[lo, hi], y=[0, 0], mode="lines",
    line=dict(color=L.PALETTE[ch], width=12), opacity=0.4, showlegend=False))
fig.add_trace(go.Scatter(x=[med], y=[0], mode="markers", name="model-implied effect",
    marker=dict(color=L.PALETTE[ch], size=16, line=dict(color="white", width=1.5))))
fig.add_vline(x=mde, line=dict(color=L.BAD, width=2.5, dash="dash"),
    annotation_text=f"national design can only detect > {mde:.1f}x", annotation_position="top left")
fig.add_vline(x=0, line=dict(color=L.INK, width=1, dash="dot"))
fig.update_yaxes(visible=False, range=[-1, 1])
L.style(fig, height=300, title=f"{ch}: the expected effect sits deep inside the national design's blind spot",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, x=0))
fig.update_layout(xaxis_title="incremental ROAS")
fig.show()
assert ver["verdict"] in {"powered", "underpowered", "overpowered", "inconclusive"}
"""),
    md(r"""
## Is it worth it? — the short-term economics

Running any test means **deviating from BAU**, and that deviation moves the KPI
during the test window. `compute_opportunity_cost` prices that from the model:
the counterfactual KPI delta versus BAU (with posterior uncertainty), the
**signed** spend change, the net-$ impact at a stated margin, and a
**learning-to-cost ratio** (the test's EVOI value against what it costs).

The **sign of `spend_delta` is load-bearing**. A go-dark **holdout** spends
*less* — money saved — so a holdout can be **net-positive**: the spend you don't
burn can outweigh the margin you forgo. A pulse that runs spend-heavy in the
window costs money. The framework computes this sign *internally* from the
perturbed matrix — never from the design's `abs()` magnitude — so it can't invert.
"""),
    code(r"""
from mmm_framework.planning import opportunity_cost as OC

oc = OC.compute_opportunity_cost(model, dz, margin_per_kpi=0.5, kpi_kind="revenue",
                                 evoi_kpi_units=100.0, max_draws=100)
print(f"expected KPI change in the window:     {oc.expected_kpi_delta:+.1f} {L.KPI}-units")
print(f"spend change (signed):                 {L.dollars(oc.spend_delta)}")
print(f"net profit impact (median):            {L.dollars(oc.net_profit_impact_median)}")
print(f"P[net loss]:                           {oc.prob_net_loss:.0%}")
print(f"opportunity cost ($, the CFO's number):{L.dollars(oc.opportunity_cost_dollar_median)}")
print(f"learning-to-cost ratio (EVOI vs cost): {oc.learning_to_cost_ratio:.2f}")

comps = [("margin on KPI change", 0.5 * oc.kpi_delta_median),
         ("spend change (signed)", -oc.spend_delta),
         ("net impact", oc.net_profit_impact_median)]
fig = go.Figure(go.Bar(x=[c[0] for c in comps], y=[c[1] for c in comps],
    marker_color=[L.GOOD if v >= 0 else L.BAD for _, v in comps],
    text=[L.dollars(v) for _, v in comps], textposition="outside", showlegend=False))
fig.add_hline(y=0, line=dict(color=L.INK, width=1))
L.style(fig, height=360, title=f"{ch} test — short-term profit-and-loss of buying the measurement")
fig.update_layout(yaxis_title=f"$ ({L.KPI} native units)")
fig.show()
assert oc.opportunity_cost_dollar_median is not None and oc.opportunity_cost_dollar_median >= 0
"""),
    md(r"""
## The trap that pre-registration exists to catch — autocorrelation

Here's the failure a naive analyst walks into. Weekly sales are **autocorrelated**
— a good week begets a good week — so the textbook "is `Δ/SE > 1.96`?"
significance rule **fires far too often** on noise alone. We prove it with an
**A/A test**: run the estimator over historical windows where *nothing happened*.
An honest 5% rule should flag ~5% of them. It flags many times that. The fix is a
**block-calibrated critical value** — a bar tuned to *this* data's autocorrelation
— which restores the true 5% size. An **A/B test** (inject a known lift) then reads
off the design's **empirical power** and MDE. This is exactly the number you must
**pre-commit to** so you can't move the goalposts after the readout.
"""),
    code(r"""
from mmm_framework.planning import simulation as SIM

sp = SIM.build_sim_panel(csv, L.KPI, ch)
asn = SIM.build_national_assignment(duration=12, seed=42)

# A/A: no real effect -> an honest 5% rule flags ~5% of windows.
aa = SIM.run_aa_simulation(sp, SIM.national_onoff_estimator, asn,
                           duration=12, max_windows=60, name="national_onoff")
# A/B: inject a known 10%-of-baseline lift -> empirical power + MDE.
inj = SIM.fixed_lift_injector(sp, asn, duration=12, pct_of_baseline=0.1)
ab = SIM.run_ab_simulation(sp, SIM.national_onoff_estimator, asn, inj,
                           duration=12, aa_result=aa, max_windows=40)

mde_txt = f"{ab.empirical_mde:.0f}" if ab.empirical_mde is not None else "n/a"
print(f"A/A false-positive rate at the naive 5% rule:            {aa.fpr_at_nominal:.0%}   (should be 5%!)")
print(f"A/A false-positive rate at the block-calibrated cutoff:  {aa.fpr_at_crit:.0%}")
print(f"inflated? {aa.fpr_inflated}   calibrated critical |Δ| = {aa.crit_value:.0f} {L.KPI}-units "
      f"over {aa.n_windows} windows")
print(f"A/B empirical power at the expected lift: {ab.power_at_expected:.0%}   empirical MDE: {mde_txt} {L.KPI}-units")

fig = go.Figure(go.Bar(
    x=["naive analytic<br>5% rule", "block-calibrated<br>critical value"],
    y=[aa.fpr_at_nominal, aa.fpr_at_crit], marker_color=[L.BAD, L.GOOD],
    text=[f"{aa.fpr_at_nominal:.0%}", f"{aa.fpr_at_crit:.0%}"], textposition="outside",
    showlegend=False))
fig.add_hline(y=0.05, line=dict(color=L.INK, width=1.5, dash="dot"),
              annotation_text="true 5% size", annotation_position="top left")
L.style(fig, height=360, title="Autocorrelation makes the naive significance test fire ~10× too often")
fig.update_layout(yaxis_title="false-positive rate (A/A)", yaxis_tickformat=".0%")
fig.show()
assert aa.fpr_at_nominal > aa.fpr_at_crit     # the naive rule is anti-conservative
"""),
    md(r"""
## The optimizer — sweep the affordable design space, keep the Pareto front

Rather than eyeball one design, `suggest_experiment` sweeps a grid of footprints ×
intensities × durations and scores each on **four objectives — all lower-is-better**:
**MDE** (precision), **power shortfall** below the target, **short-term cost**
(the economics above), and **duration** (time in market). It returns the
**non-dominated Pareto front** and recommends the **knee** among the *powered*
designs. Run on the national spine, every candidate it can build is
**under-powered** — a final, quantified confirmation that the national data cannot
carry this test, and that the pre-registered plan must be the **geo holdout**.
"""),
    code(r"""
from mmm_framework.planning.experiment_optimizer import suggest_experiment, cooldown_weeks

cd = cooldown_weeks(model, ch)                  # adstock washout before a clean re-read
sug = suggest_experiment(model, csv, L.KPI, ch, max_draws=40)

cols = ["index", "mode", "footprint", "duration", "mde_roas", "power",
        "power_shortfall", "tradeoff", "powered", "on_pareto"]
show = pd.DataFrame(sug["candidates"])[cols].copy()
print(f"optimizer swept {sug['n_candidates']} affordable national designs "
      f"(target power {sug['power_target']:.0%}); 4 objectives, all lower-better:")
print(show.round(3).to_string(index=False))

rec = sug["recommended"]
rpw = f"{rec['power']:.0%}" if rec.get("power") is not None else "n/a"
print(f"\nrecommended (knee of the Pareto front): {rec['mode']} / {rec['duration']}w, "
      f"MDE {rec['mde_roas']:.2f}, power {rpw}, powered={rec['powered']}")
print(f"cool-down before a clean re-read: {cd['cooldown_weeks']} weeks "
      f"(adstock alpha {cd['alpha']:.2f}, basis '{cd['basis']}')")
print("Every national candidate is UNDER-powered -> pre-register the GEO holdout instead.")

pw = show["power"].fillna(0.0)
fig = go.Figure(go.Bar(x=show["index"].astype(str), y=pw,
    marker_color=[L.GOOD if p else L.BAD for p in show["powered"]],
    text=[f"{v:.0%}" for v in pw], textposition="outside", showlegend=False))
fig.add_hline(y=sug["power_target"], line=dict(color=L.INK, width=1.5, dash="dot"),
              annotation_text=f"target {sug['power_target']:.0%}", annotation_position="top left")
L.style(fig, height=340, title="Statistical power of every national design the optimizer tried")
fig.update_layout(xaxis_title="candidate index", yaxis_title="power to detect the model's effect",
                  yaxis_tickformat=".0%")
fig.show()
assert sug["recommended"] is not None
"""),
    md(r"""
## The pre-registration — locked before a single week runs

Everything above collapses into one short, **pre-committed** protocol. We write it
down *now*, so that when the readout lands we already know exactly what it means —
no post-hoc re-cutting, no moving goalposts.

- **Estimand.** Display's **incremental ROAS** — the causal return the geo holdout
  measures, folded into the next fit as the `roas` estimand (that's T3).
- **Method.** A **randomized geo holdout**: the matched treatment/control DMAs
  assigned above, treated markets going **dark** on Display for the chosen window,
  read as a difference-in-differences against their twins.
- **Minimum detectable effect.** The geo design's MDE plotted above — locked as the
  smallest effect we commit, in advance, to be able to detect. (The national
  flighting alternative was ruled out precisely because its MDE was too coarse.)
- **Decision rule.** If the measured incremental ROAS **clears the placebo band**
  *and* its interval **excludes break-even**, we re-allocate toward Display; if it
  lands inside the placebo band, we treat it as **noise** and hold. Either way the
  number updates the model.
- **Cool-down.** Wait the adstock washout computed above before reading the treated
  markets as "back to BAU" or scheduling the next test.

**Readout →** *"Display is worth testing, and now we have a test that can actually
measure it and a budget we can defend spending on it. Run it — then fold the
number back into the model."* That folding-in is **T3**.

Next: **`lifecycle_04_calibrate`** — T3, where the readout updates the posterior.
Previously: **`lifecycle_02_prioritize`** picked Display to test in the first place.
"""),
]


def main() -> None:
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    out = "lifecycle_03_design.ipynb"
    with open(out, "w") as f:
        nbformat.write(nb, f)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    print(f"wrote {out} with {len(nb.cells)} cells ({n_code} code)")


if __name__ == "__main__":
    main()
