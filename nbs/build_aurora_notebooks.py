"""Generate the Aurora Coffee Co. showcase notebooks (run from ``nbs/``).

    uv run python build_aurora_notebooks.py

Each notebook is authored here as a list of markdown/code cells and emitted as a
valid ``.ipynb`` via ``nbformat``. They share ``aurora.py`` (the synthetic world)
and tell one story across chapters. Keeping the source here makes the set
regenerable and reviewable in one place.
"""

from __future__ import annotations

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from charts_src import CHARTS  # validated chart-cell code (see validate_chart_cells.py)


def chart(key: str):
    """A notebook code cell from the validated CHARTS source."""
    return code(CHARTS[key])

# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


def write_notebook(path: str, cells: list, title: str):
    nb = new_notebook(cells=cells)
    nb.metadata.update(
        {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "title": title,
        }
    )
    with open(path, "w") as f:
        nbformat.write(nb, f)
    print(f"wrote {path}  ({len(cells)} cells)")


# Shared setup snippet (imports, quiet logs, Aurora palette, plotting defaults).
SETUP = """
import warnings, sys
warnings.filterwarnings("ignore")
from loguru import logger
logger.remove(); logger.add(sys.stderr, level="ERROR")   # quiet framework logs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aurora import generate_aurora, CHANNELS, PRODUCTS, PALETTE, CHANNEL_COLORS

plt.rcParams.update({
    "figure.dpi": 110, "figure.figsize": (9, 4.2),
    "axes.grid": True, "grid.alpha": 0.18,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cfc7bd", "axes.titleweight": "bold",
    "figure.facecolor": "white", "savefig.facecolor": "white", "font.size": 10,
})
ACCENT, INK, MUTED = PALETTE["accent"], PALETTE["ink"], PALETTE["muted"]

aurora = generate_aurora()      # the one dataset every chapter shares
""".strip()


# ===========================================================================
# Notebook 00 — Overview & the hook
# ===========================================================================


def nb_00():
    c = []
    c.append(md(r"""
# ☕ Aurora Coffee Co. — A Causal MMM Story

> **A six-notebook tour of `mmm_framework`, told through one company's quarterly budget decision.**

**Aurora Coffee Co.** is a fast-growing direct-to-consumer coffee brand with two product lines —
**Original** (ground & whole bean) and **Cold Brew** (ready-to-drink) — and four paid media channels:
**TV, Search, Social, Display**.

It's planning season. The CMO has to defend the marketing budget to the CFO and decide where next
quarter's dollars go. The data science team is asked one deceptively simple question:

> ### *"What is each channel actually worth — and how should we reallocate?"*

A normal dashboard answers this with **correlations**. This notebook set shows why that answer is
*wrong in a way that costs real money*, and how a **causal** marketing-mix model — with honest
uncertainty, experiment calibration, and mediation — gets it right.

| Chapter | Notebook | The question |
|---|---|---|
| **0** | *this one* | What's the trap? |
| **1 · Causality** | `01_causality.ipynb` | Can we trust the number? |
| **2 · Base MMM** | `02_base_mmm.ipynb` | What is each channel worth? |
| **3 · Extended MMM** | `03_extended_mmm.ipynb` | What is TV *really* doing? And do our products fight each other? |
| **4 · Reporting** | `04_reporting.ipynb` | How do we put this in front of the board? |
| **5 · The workflow** | `05_unified_workflow.ipynb` | Put it all together → a defensible plan. |

*Everything runs on synthetic data (`aurora.py`) built with a known ground truth, so we can grade the
model against the real answer. Fits use small draw counts to stay fast — bump them up for real work.*
"""))
    c.append(code(SETUP))
    c.append(md(r"""
## Meet the business

Two years of weekly data: revenue (in \$000s), spend by channel, and a few business drivers.
"""))
    c.append(code(r"""
df = aurora.frame()
print(f"{len(df)} weeks · revenue ${aurora.sales_total.mean():,.0f}k/wk avg · "
      f"media is {100*aurora.true_contribution.sum()/aurora.sales_total.sum():.0f}% of revenue")
df[["TV","Search","Social","Display","sales_total"]].head()
"""))
    c.append(code(r"""
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax1.stackplot(df.index, aurora.sales_original, aurora.sales_coldbrew,
              labels=PRODUCTS, colors=[PALETTE["crema"], PALETTE["sky"]], alpha=.9)
ax1.set_title("Aurora weekly revenue by product ($000s)"); ax1.legend(loc="upper left", ncol=2)
for ch in CHANNELS:
    ax2.plot(df.index, df[ch], color=CHANNEL_COLORS[ch], lw=1.6, label=ch)
ax2.set_title("Weekly media spend by channel ($000s)"); ax2.legend(ncol=4, loc="upper left")
plt.tight_layout(); plt.show()
"""))
    c.append(md(r"""
## The trap 🪤

The fastest way to "value" a channel is to correlate its spend with sales. Let's do exactly that —
and put it next to the **true** ROAS (which we know, because we built this world).
"""))
    c.append(code(r"""
naive_corr = {c: np.corrcoef(aurora.spend[c], aurora.sales_total)[0, 1] for c in CHANNELS}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 3.6))
order = ["Search", "Social", "Display", "TV"]
axL.barh(order, [naive_corr[c] for c in order], color=[CHANNEL_COLORS[c] for c in order])
axL.set_title("What the dashboard shows\n(correlation of spend with sales)"); axL.set_xlim(0, 1)
axR.barh(order, [aurora.true_roas[c] for c in order], color=[CHANNEL_COLORS[c] for c in order])
axR.axvline(1.0, color=INK, ls="--", lw=1, alpha=.6)
axR.set_title("What is actually true\n(real ROAS — revenue per $1 spend)")
for ax in (axL, axR): ax.bar_label(ax.containers[0], fmt="%.2f", padding=3)
plt.tight_layout(); plt.show()
"""))
    c.append(md(r"""
Look at the disagreement:

- **Search** has the *strongest* correlation with sales (≈0.85) — the dashboard's hero. Its **true ROAS
  is 0.66**: it loses money.
- **TV** has a *weak* correlation (≈0.29) — the dashboard's first cut. Its **true ROAS is 2.1**: it's the
  best investment Aurora has.

Acting on the left chart, Aurora would **pour money into Search and starve TV** — exactly backwards.
Three things are breaking the correlation:
"""))
    c.append(code(r"""
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11.5, 3.4))

# 1) Confounding: a latent "demand" drives BOTH Search spend AND sales.
a1.plot(df.index, (aurora.demand-aurora.demand.mean())/aurora.demand.std(), color=PALETTE["leaf"], label="latent demand")
a1.plot(df.index, (aurora.spend["Search"]-aurora.spend["Search"].mean())/aurora.spend["Search"].std(),
        color=CHANNEL_COLORS["Search"], lw=1.2, alpha=.8, label="Search spend")
a1.set_title("① Confounding\ndemand drives spend & sales"); a1.legend(fontsize=7, loc="upper left")

# 2) Mediation: TV builds awareness, awareness drives sales.
a2.scatter(aurora.awareness, aurora.sales_total, s=14, color=CHANNEL_COLORS["TV"], alpha=.6)
a2.set_title("② Mediation\nTV → awareness → sales"); a2.set_xlabel("brand awareness"); a2.set_ylabel("revenue")

# 3) Cannibalization: the two products fight.
a3.scatter(aurora.sales_coldbrew, aurora.sales_original, s=14, color=PALETTE["berry"], alpha=.6)
a3.set_title("③ Cannibalization\nCold Brew vs Original"); a3.set_xlabel("Cold Brew $"); a3.set_ylabel("Original $")
plt.tight_layout(); plt.show()
"""))
    c.append(md(r"""
1. **Confounding** — Aurora bids harder on Search/Social *when demand is already high*, so their spend
   rides the same wave as sales. The credit belongs to demand, not the channel.
2. **Mediation** — TV barely sells anything *directly*; it builds **awareness**, and awareness sells.
   Its value is real but hidden one step upstream.
3. **Cannibalization** — Cold Brew and Original eat each other's lunch in summer.

A correlation can't see past any of these. The rest of this set fixes each one — and ends with a
**reallocation that's worth real money**. Start with **`01_causality.ipynb`**.
"""))
    return c


# ===========================================================================
# Notebook 01 — Causality
# ===========================================================================


def nb_01():
    c = []
    c.append(md(r"""
# 1 · Causality — *Can we trust the number?*

> **Chapter 1 of the Aurora story.** The dashboard says fund Search. Before Aurora moves a dollar, the
> data-science team asks the causal question: *what would have to be true for "Search ROAS" to mean
> what we think it means?* This notebook draws the causal map, finds what to adjust for, shows why
> adjustment **alone isn't enough**, and anchors the model to a **randomized experiment**.
"""))
    c.append(code(SETUP + "\nimport networkx as nx\nfrom matplotlib.patches import Patch"))
    c.append(md(r"""
## Step 1 — Draw the causal map (a DAG)

Causal claims need assumptions, and assumptions should be *explicit*. `mmm_framework` lets you state
them as a directed acyclic graph. Here is Aurora's: **Demand** confounds spend↔sales, **Awareness** sits
on the path from TV to Sales (a *mediator*), and **Price** is a clean predictor. *(For legibility the
map shows two representative channels — TV the brand-builder and Search the demand-chaser; the fitted
model in Steps 5–6 uses all four.)*
"""))
    c.append(code(r"""
from mmm_framework.dag_model_builder.dag_spec import DAGSpec, DAGNode, DAGEdge, NodeType

dag = DAGSpec(
    nodes=[
        DAGNode(id="sales",  variable_name="Sales",     node_type=NodeType.KPI),
        DAGNode(id="tv",     variable_name="TV",        node_type=NodeType.MEDIA),
        DAGNode(id="search", variable_name="Search",    node_type=NodeType.MEDIA),
        DAGNode(id="demand", variable_name="Demand",    node_type=NodeType.CONTROL),
        DAGNode(id="price",  variable_name="Price",     node_type=NodeType.CONTROL),
        DAGNode(id="aware",  variable_name="Awareness", node_type=NodeType.CONTROL),
    ],
    edges=[
        DAGEdge(source="tv", target="sales"), DAGEdge(source="search", target="sales"),
        DAGEdge(source="demand", target="search"), DAGEdge(source="demand", target="sales"),
        DAGEdge(source="price", target="sales"),
        DAGEdge(source="tv", target="aware"), DAGEdge(source="aware", target="sales"),
    ],
)
print("media:", [n.variable_name for n in dag.media_nodes],
      "| controls:", [n.variable_name for n in dag.control_nodes])
"""))
    c.append(code(r"""
# Draw it, coloured by causal role (computed in Step 3).
ROLE_COLORS = {"kpi": INK, "media": ACCENT, "confounder": PALETTE["leaf"],
               "mediator": PALETTE["berry"], "precision_control": PALETTE["sky"]}
G = nx.DiGraph(); G.add_edges_from((e.source, e.target) for e in dag.edges)
pos = {"demand": (0, 1.2), "tv": (1, 2), "search": (1, 0.4), "aware": (2, 2),
       "price": (1, -0.8), "sales": (3, 0.6)}
role = {"sales": "kpi", "tv": "media", "search": "media", "demand": "confounder",
        "aware": "mediator", "price": "precision_control"}
plt.figure(figsize=(8.5, 4.2))
nx.draw_networkx_edges(G, pos, arrowsize=16, edge_color="#b8afa4", width=1.6,
                       connectionstyle="arc3,rad=0.05", node_size=2600)
nx.draw_networkx_nodes(G, pos, node_size=2600,
                       node_color=[ROLE_COLORS[role[n]] for n in G.nodes])
nx.draw_networkx_labels(G, pos, {n: dag.get_node(n).variable_name for n in G.nodes},
                        font_color="white", font_size=9, font_weight="bold")
plt.legend(handles=[Patch(color=v, label=k.replace("_"," ")) for k, v in ROLE_COLORS.items()],
           loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.08))
plt.title("Aurora's causal map"); plt.axis("off"); plt.tight_layout(); plt.show()
"""))
    c.append(md(r"""
## Step 2 — Identify the effect: *what must we adjust for?*

Given the graph, `identification_report` finds the **back-door adjustment set** — the variables you must
condition on to read TV's (or Search's) effect cleanly — and flags variables you must **not** touch.
"""))
    c.append(code(r"""
from mmm_framework.dag_model_builder.identification import identification_report

rep = identification_report(dag, treatment_id="search", outcome_id="sales")
print("Adjustment set (condition on these):", rep.adjustment_set)
print("Descendants of Search (must NOT adjust):", rep.descendants_of_treatment)
print("Effect identifiable?", rep.identifiable)
print("\nBack-door paths found:")
for p in rep.backdoor_paths:
    print("  ", p.render(), " — blocked by:", p.blocked_by or "(open!)")
"""))
    c.append(md(r"""
The model must **condition on Demand** (to close the `Search ← Demand → Sales` back-door) and must
**never condition on Awareness** for a *total* TV effect — Awareness is on the causal path, so adjusting
it would throw away exactly the effect we want to measure (the classic *"bad control"*).
"""))
    c.append(md(r"""
## Step 3 — Auto-detect bad controls (so non-experts can't shoot themselves in the foot)

`classify_dag_roles` labels every candidate control. The framework will later **refuse** to condition on
a mediator or collider.
"""))
    c.append(code(r"""
from mmm_framework.dag_model_builder.identification import classify_dag_roles

cls = classify_dag_roles(dag, treatment_ids=["tv", "search"], outcome_id="sales",
                         control_ids=["demand", "price", "aware"])
rows = []
for cid in ["demand", "price", "aware"]:
    r, why = cls.role_for(cid)
    rows.append({"variable": dag.get_node(cid).variable_name, "role": r,
                 "use it?": "✅ keep" if r in ("confounder", "precision_control") else "⛔ REMOVE",
                 "why": why[:80] + "…"})
pd.DataFrame(rows)
"""))
    c.append(md(r"""
## Step 4 — The framework enforces it

`dag_to_mff_config` turns the graph into a model configuration, tagging each control with its causal
role. When you build a model that conditions on a mediator, it **raises** — the bad control never makes
it into the likelihood.
"""))
    c.append(code(r"""
from mmm_framework.dag_model_builder.config_translator import dag_to_mff_config

cfg = dag_to_mff_config(dag)
for ctrl in cfg.controls:
    print(f"  {ctrl.name:10s} -> {getattr(ctrl, 'causal_role', None)}")
"""))
    c.append(md(r"""
## Step 5 — Why adjustment *alone* isn't enough

Here's the uncomfortable truth most MMM decks skip. Let's fit Aurora's model two ways — **blind to
demand** vs. **controlling for demand** — and compare the recovered ROI to the truth.

*(Small draws for speed; `cores=1` keeps macOS sampling crash-free.)*
"""))
    c.append(code(r"""
from mmm_framework import BayesianMMM, ModelConfigBuilder, SeasonalityConfigBuilder, TrendConfig, TrendType
from mmm_framework.analysis import MMMAnalyzer

def base_config():
    return (ModelConfigBuilder().bayesian_pymc().with_chains(2).with_draws(400).with_tune(400)
            .with_seasonality_builder(SeasonalityConfigBuilder().with_yearly(order=2)).build())

def fit_and_roi(control_demand):
    panel = aurora.base_panel(control_demand=control_demand)
    m = BayesianMMM(panel, base_config(), TrendConfig(type=TrendType.LINEAR))
    m.fit(draws=400, tune=400, chains=2, cores=1, random_seed=0)
    return m, MMMAnalyzer(m).compute_channel_roi().set_index("Channel")["ROI"]

m_blind, roi_blind = fit_and_roi(control_demand=False)
m_ctrl,  roi_ctrl  = fit_and_roi(control_demand=True)
"""))
    c.append(code(r"""
comp = pd.DataFrame({"true ROAS": aurora.true_roas,
                     "demand-blind": roi_blind, "demand-controlled": roi_ctrl}).loc[CHANNELS]
display(comp.round(2))

x = np.arange(len(CHANNELS)); w = 0.26
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.bar(x-w, comp["true ROAS"], w, label="true", color=INK)
ax.bar(x,   comp["demand-blind"], w, label="demand-blind", color=PALETTE["berry"])
ax.bar(x+w, comp["demand-controlled"], w, label="demand-controlled", color=PALETTE["sky"])
ax.set_xticks(x); ax.set_xticklabels(CHANNELS); ax.axhline(1, color=MUTED, ls="--", lw=1)
ax.set_title("Controlling for demand helps — but Search is still overstated"); ax.legend()
plt.tight_layout(); plt.show()
"""))
    c.append(md(r"""
Subtract the truth from each estimate and the **leftover bias** is easier to read: controlling for
demand shrinks every bar, but Search's stays stubbornly positive.
"""))
    c.append(chart("nb01_bias"))
    c.append(md(r"""
Controlling for demand pulls Search's ROI down — but it's **still far above its true 0.66**. The demand
proxy is collinear with demand-chasing spend, and no observational adjustment fully removes that. *This
is the central limitation of any observational MMM*, and the framework says so out loud. The fix isn't
more controls. It's **randomized evidence**.
"""))
    c.append(md(r"""
## Step 6 — Anchor to an experiment

Last quarter Aurora ran a **geo lift test** on Search and Social and measured their true incremental
ROAS. `mmm_framework` folds that straight into the likelihood: the experiment becomes a data point the
model must agree with — updating the coefficient, the saturation curve, **and** the adstock jointly.
"""))
    c.append(code(r"""
from mmm_framework.calibration import ExperimentMeasurement, ExperimentEstimand

window = (str(aurora.weeks[0].date()), str(aurora.weeks[-1].date()))
experiments = [
    ExperimentMeasurement("Search", window, value=float(aurora.true_roas["Search"]), se=0.07,
                          estimand=ExperimentEstimand.ROAS),
    ExperimentMeasurement("Social", window, value=float(aurora.true_roas["Social"]), se=0.07,
                          estimand=ExperimentEstimand.ROAS),
]
panel = aurora.base_panel(control_demand=True)
m_cal = BayesianMMM(panel, base_config(), TrendConfig(type=TrendType.LINEAR), experiments=experiments)
m_cal.fit(draws=400, tune=400, chains=2, cores=1, random_seed=0)
roi_cal = MMMAnalyzer(m_cal).compute_channel_roi().set_index("Channel")["ROI"]
"""))
    c.append(code(r"""
final = pd.DataFrame({"true ROAS": aurora.true_roas, "observational": roi_ctrl,
                      "experiment-calibrated": roi_cal}).loc[CHANNELS]
display(final.round(2))
for ch in ["Search", "Social"]:
    print(f"{ch}: observational {roi_ctrl[ch]:.2f}  →  calibrated {roi_cal[ch]:.2f}   (true {aurora.true_roas[ch]:.2f})")
"""))
    c.append(md(r"""
The whole chapter in one picture: each calibrated channel's ROAS **slides off its inflated observational
value onto the truth** (the `|` mark) once the experiment enters the likelihood.
"""))
    c.append(chart("nb01_punchline"))
    c.append(md(r"""
**There it is.** The experiment pulls Search and Social down onto their true ROAS — the demand mirage is
gone. This is the single most important move in causal MMM: *anchor the observational model to
randomized evidence.*

### Takeaways
- A DAG makes assumptions explicit and **auto-detects bad controls** (Awareness is a mediator — never a control).
- **Observational adjustment helps but is not sufficient** when spend chases demand.
- **Experiment calibration** corrects the residual bias — Search/Social snap to truth.

But notice TV is *still* low. The base model can't see that TV works through **awareness** — that's
**`03_extended_mmm.ipynb`**. First, let's read the base model properly: **`02_base_mmm.ipynb`**.
"""))
    return c


# ===========================================================================
# Notebook 02 — Base MMM
# ===========================================================================


def nb_02():
    c = []
    c.append(md(r"""
# 2 · The Base MMM — *What is each channel worth?*

> **Chapter 2 of the Aurora story.** With the causal groundwork laid, we fit `mmm_framework`'s core
> **Bayesian** MMM and read it the way an analyst would: contributions and ROAS **with honest
> uncertainty**, where the next dollar goes, and a what-if. We'll also see — honestly — what the base
> model *can't* see yet.
"""))
    c.append(code(SETUP))
    c.append(md(r"""
## Fit the model

One model, demand-controlled (the back-door we identified in Chapter 1), with yearly seasonality and a
linear trend. Bayesian sampling gives a full posterior — not a point estimate.
"""))
    c.append(code(r"""
from mmm_framework import BayesianMMM, ModelConfigBuilder, SeasonalityConfigBuilder, TrendConfig, TrendType
from mmm_framework.analysis import MMMAnalyzer

panel = aurora.base_panel(control_demand=True)
model_config = (ModelConfigBuilder().bayesian_pymc()
                .with_chains(2).with_draws(600).with_tune(600).with_target_accept(0.9)
                .with_seasonality_builder(SeasonalityConfigBuilder().with_yearly(order=2)).build())

mmm = BayesianMMM(panel, model_config, TrendConfig(type=TrendType.LINEAR))
results = mmm.fit(draws=600, tune=600, chains=2, cores=1, random_seed=0)
print("fitted:", mmm.n_channels, "channels ·", mmm.n_obs, "weeks")
"""))
    c.append(md(r"""
## Did it converge? (honest uncertainty starts here)

Before trusting a number, check the diagnostics. `R̂≈1` and healthy effective sample size mean the
chains agree; divergences flag geometry problems. *(These small-draw runs are for speed — real analyses
use ≥4 chains and ≥1000 draws.)*
"""))
    c.append(code(r"""
d = results.diagnostics
print(f"R̂ max         {d['rhat_max']:.3f}   (want < 1.01)")
print(f"ESS bulk min   {d['ess_bulk_min']:.0f}    (want > 400)")
print(f"divergences    {d['divergences']}      (want 0)")
results.summary(var_names=[f"beta_{c}" for c in CHANNELS]).round(3)
"""))
    c.append(md(r"""
The table is the numbers; here is their *shape*. Every ROAS below is a summary of these posterior
coefficient distributions — wide bands mean honest uncertainty, not a single confident point.
"""))
    c.append(chart("nb02_forest"))
    c.append(md(r"""
## How much of this came from the data vs our priors?

A posterior interval tells you what to believe *after* the fit — not how much of that belief is the **data**
talking versus the **prior**. The honest check is **prior → posterior learning**: for each parameter,
**contraction** $c = 1 - \mathrm{Var}_{\text{post}}/\mathrm{Var}_{\text{prior}}$ ($c \to 1$ = the data pinned
it; $c \approx 0$ = prior-dominated) and the prior↔posterior **overlap** ($\to 1$ = posterior indistinguishable
from the prior). `compute_parameter_learning` sorts the most prior-dominated parameters to the top.
"""))
    c.append(code(r"""
from mmm_framework.diagnostics import plot_parameter_learning

lrn = mmm.compute_parameter_learning(prior_samples=2000)   # base model takes no seed; posterior is seeded
display(lrn.round(3))

ax = plot_parameter_learning(lrn, threshold=0.1)
plt.tight_layout(); plt.show()

# The Beta(2,2) adstock BLEND WEIGHTS are barely updated by observational data
# (the carryover/saturation equifinality): the data hardly moves carryover.
ad = lrn[lrn.parameter.str.contains("adstock")]
assert (ad.contraction < 0.1).any(), f"expected a prior-dominated adstock weight; got\n{ad}"
pd_ad = ad[ad.contraction < 0.1].sort_values("contraction").iloc[0]
strong = lrn[lrn.contraction > 0.5].parameter.tolist()
print(f"✓ Prior-dominated adstock weight: {pd_ad.parameter} "
      f"(contraction={pd_ad.contraction:.3f}, overlap={pd_ad.overlap:.3f}, verdict='{pd_ad.verdict}')")
print(f"  Meanwhile {len(strong)} ROI-driving parameters were strongly learned, e.g. {strong[:5]}")
"""))
    c.append(md(r"""
Read the bars honestly. The **adstock blend weights** sit at (or below) zero contraction with near-total prior
overlap — observational data barely moves *carryover*, a direct consequence of the carryover/saturation
equifinality. But the pieces that actually drive ROI — the **saturation** rates `sat_lam_*`, the **intercept**,
the **control** coefficients, the noise scale **sigma**, and the **seasonality** terms — are strongly learned
($c$ near 1). One channel even shows **negative** contraction: `beta_Search`, the demand-confounded channel
the model cannot pin — a genuine prior↔data tension flag, exactly the channel Chapter 1 warned about. So the
ROAS story rests on parameters the data informed; only the carryover shape leans on the prior.
"""))
    c.append(md(r"""
## Does the model fit the data?

Before reading any contribution, check that the model actually tracks revenue. The shaded band is the
posterior credible interval on the prediction.
"""))
    c.append(chart("nb02_fit"))
    c.append(md(r"""
## Channel contributions — with credible intervals

The counterfactual contribution of a channel is how much revenue disappears if you turn it off. Every
number carries a posterior interval, so stakeholders see the *confidence*, not just the point.
"""))
    c.append(code(r"""
contrib = mmm.compute_counterfactual_contributions(compute_uncertainty=True, hdi_prob=0.9)
ct = contrib.total_contributions.loc[CHANNELS]
lo = contrib.contribution_hdi_low.loc[CHANNELS]; hi = contrib.contribution_hdi_high.loc[CHANNELS]

fig, ax = plt.subplots(figsize=(9, 3.8))
bars = ax.bar(CHANNELS, ct, color=[CHANNEL_COLORS[c] for c in CHANNELS],
              yerr=[ct-lo, hi-ct], capsize=5, ecolor=INK)
ax.set_title("Total revenue contribution by channel ($000s, 90% HDI)")
ax.bar_label(bars, fmt="%.0f", padding=8)
plt.tight_layout(); plt.show()
"""))
    c.append(md(r"""
Those totals come from *somewhere in time*. Stacking each channel's weekly contribution on top of the
non-media baseline shows when media drove revenue — and that the stack tracks the observed line.
"""))
    c.append(chart("nb02_decomp"))
    c.append(md(r"""
## ROAS — return on ad spend

`MMMAnalyzer` turns contributions into ROI/ROAS (revenue per \$1 of spend), with the same uncertainty.
"""))
    c.append(code(r"""
roi = MMMAnalyzer(mmm).compute_channel_roi().set_index("Channel").loc[CHANNELS]
roi["true ROAS"] = aurora.true_roas.loc[CHANNELS]
display(roi[["Total Spend", "Total Contribution", "ROI", "true ROAS"]].round(2))
"""))
    c.append(md(r"""
The same table as a picture — ROAS with its credible interval, and the **true** ROAS as a diamond. The
model deflates the demand-chasers correctly, but the diamonds for TV and Display sit well above the bars.
"""))
    c.append(chart("nb02_roas_ci"))
    c.append(md(r"""
## Where does the *next* dollar go? (marginal ROAS)

Total ROAS is the average dollar. For budgeting you want the *marginal* dollar — what a +10% bump
returns — because saturation means channels don't scale forever.
"""))
    c.append(code(r"""
marg = mmm.compute_marginal_contributions(spend_increase_pct=10, compute_uncertainty=True)
marg.set_index("Channel").loc[CHANNELS, ["Current Spend", "Marginal Contribution", "Marginal ROAS"]].round(2)
"""))
    c.append(md(r"""
Average vs marginal, side by side: where the **next** dollar earns less than the average dollar already
spent, the channel is saturating. That gap is the budgeting signal.
"""))
    c.append(chart("nb02_marginal"))
    c.append(md(r"""
## A what-if: move 20% of Search into TV

*Watch the sign.* The **base** model still undervalues TV, so it will judge this move a
**loss** — exactly the wrong call. Hold that thought; Chapter 3 explains why.
"""))
    c.append(code(r"""
scenario = mmm.what_if_scenario({"Search": 0.8, "TV": 1.2})
print(f"baseline revenue : ${scenario['baseline_outcome']:,.0f}k")
print(f"scenario revenue : ${scenario['scenario_outcome']:,.0f}k")
print(f"change           : {scenario['outcome_change_pct']:+.2f}%  "
      f"(${scenario['outcome_change']:+,.0f}k)")
"""))
    c.append(chart("nb02_whatif"))
    c.append(md(r"""
## The honest caveat — and the cliffhanger

Compare the base model's ROAS to the truth: it nails Search/Social being modest, but it **badly
undervalues TV and Display**.
"""))
    c.append(code(r"""
chk = pd.DataFrame({"base-model ROAS": roi["ROI"], "true ROAS": aurora.true_roas.loc[CHANNELS]})
chk["gap"] = chk["base-model ROAS"] - chk["true ROAS"]
chk.round(2)
"""))
    c.append(md(r"""
That gap isn't noise. **TV and Display work almost entirely through brand awareness** — a pathway the
base model can't see, so it can't credit them for it. To value a brand channel you have to model the
mediation. That's **`03_extended_mmm.ipynb`**.

### Bonus — models are serializable
A fitted model (trace + config + experiment anchoring) round-trips to disk, so an expensive fit is
re-used downstream rather than re-run.
"""))
    c.append(code(r"""
from pathlib import Path
from mmm_framework.serialization import MMMSerializer

Path("artifacts").mkdir(exist_ok=True)
MMMSerializer.save(mmm, "artifacts/aurora_base")
reloaded = MMMSerializer.load("artifacts/aurora_base", panel)
print("reloaded model channels:", reloaded.channel_names, "| trace restored:", reloaded._trace is not None)
"""))
    c.append(md(r"""
### Takeaways
- The base Bayesian MMM gives **contributions, ROAS, and marginal ROAS — all with credible intervals**.
- Diagnostics (`R̂`, ESS, divergences) keep the uncertainty *honest*.
- It correctly deflates the demand-chasing channels **but undervalues brand channels** that work through
  a mediator — which is exactly what Chapter 3 fixes.
"""))
    return c


# ===========================================================================
# Notebook 03 — Extended MMM
# ===========================================================================


def nb_03():
    c = []
    c.append(md(r"""
# 3 · Extended MMM — *What is TV really doing? Do our products fight?*

> **Chapter 3 of the Aurora story.** Two questions the base model couldn't answer: **(a)** TV looks weak,
> but is it secretly Aurora's growth engine via *brand awareness*? **(b)** Do Original and Cold Brew
> *cannibalize* each other? `mmm_framework`'s extended models — **nested** (mediation) and
> **multivariate** (multi-outcome + cross-effects) — answer both.
"""))
    c.append(code(SETUP))
    c.append(md(r"""
## Part A — Mediation: TV → Awareness → Sales

A **NestedMMM** models the pathway explicitly. Aurora measures **awareness** in a monthly brand-tracker
survey (partially observed); TV and Display feed it, and it feeds sales. The model decomposes each
channel's effect into **direct** vs **indirect (via awareness)**.
"""))
    c.append(code(r"""
from mmm_framework.mmm_extensions.models import NestedMMM
from mmm_framework.mmm_extensions.builders import MediatorConfigBuilder, NestedModelConfigBuilder

X = aurora.media_matrix()
nested_cfg = (NestedModelConfigBuilder()
    .add_mediator(MediatorConfigBuilder("awareness")
                  .partially_observed(observation_noise=0.1)   # the monthly survey
                  .with_positive_media_effect(sigma=1.0)
                  .with_direct_effect(sigma=0.5).build())
    .map_channels_to_mediator("awareness", ["TV", "Display"])  # brand channels build awareness
    .build())

nested = NestedMMM(X, aurora.sales_total, list(CHANNELS), nested_cfg,
                   mediator_data={"awareness": aurora.awareness_survey}, index=aurora.weeks)
nested.fit(draws=500, tune=500, chains=2, cores=1, random_seed=0)
"""))
    c.append(code(r"""
med = nested.get_mediation_effects().set_index("channel")
brand = ["TV", "Display"]   # the channels we routed through awareness
# Search/Social are pure direct-response here (not mapped to awareness), so the
# mediation decomposition only applies to the brand channels.
display(med.loc[brand, ["direct_effect", "total_indirect", "total_effect", "proportion_mediated"]].round(2))

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar(brand, med.loc[brand, "direct_effect"], label="direct", color=PALETTE["crema"])
ax.bar(brand, med.loc[brand, "total_indirect"], bottom=med.loc[brand, "direct_effect"],
       label="indirect (via awareness)", color=ACCENT)
ax.set_title("TV & Display: where their effect actually flows"); ax.legend()
plt.tight_layout(); plt.show()
"""))
    c.append(md(r"""
**The reveal.** TV and Display are ~**fully mediated** — virtually all of their effect reaches sales
*through awareness*. The base model in Chapter 2 saw only the (tiny) direct slice and called them weak.
They aren't weak; they're a **brand engine**. This is the number that flips Aurora's budget logic.
"""))
    c.append(md(r"""
Is that reveal real or an artifact? Because this world is synthetic we know the **true** mediated share —
and the model's `proportion_mediated` lands right on it.
"""))
    c.append(chart("nb03_mediation_validate"))
    c.append(md(r"""
The mediator itself is recovered too: the model's **latent awareness** (inferred from the sparse monthly
survey) tracks both the survey points and the normally-hidden true awareness.
"""))
    c.append(chart("nb03_awareness"))
    c.append(md(r"""
## Part B — Cannibalization: do the two products fight?

A **MultivariateMMM** models both products at once, with a **cross-effect** linking them and correlated
residuals (a shared-demand shock hits both).
"""))
    c.append(code(r"""
from mmm_framework.mmm_extensions.models import MultivariateMMM
from mmm_framework.mmm_extensions.builders import (
    MultivariateModelConfigBuilder, OutcomeConfigBuilder, cannibalization_effect)

_, outcomes = aurora.extension_inputs()
mv_cfg = (MultivariateModelConfigBuilder()
    .add_outcome(OutcomeConfigBuilder("sales_original", column="sales_original")
                 .with_positive_media_effects(sigma=0.5).build())
    .add_outcome(OutcomeConfigBuilder("sales_coldbrew", column="sales_coldbrew")
                 .with_positive_media_effects(sigma=0.5).build())
    .add_cross_effect(cannibalization_effect(source="sales_coldbrew", target="sales_original"))
    .build())

mv = MultivariateMMM(X, outcomes, list(CHANNELS), mv_cfg, index=aurora.weeks)
mv.fit(draws=500, tune=500, chains=2, cores=1, random_seed=0)
"""))
    c.append(code(r"""
ce = mv.get_cross_effects_summary()
display(ce[["source", "target", "effect_type", "mean", "hdi_3%", "hdi_97%"]].round(4))
verdict = "interval below zero" if ce["hdi_97%"].iloc[0] < 0 else "not conclusive"
print(f"\nCold Brew → Original cross-effect: {verdict}")
print("→ The naive read is 'Cold Brew cannibalizes Original.' But hold that thought — the prior on this")
print("  effect is one-sided, so 'below zero' may be partly automatic. We stress-test it next.")
"""))
    c.append(md(r"""
The table says the interval sits below zero. That *looks* like a confirmed cannibalization — but before we
budget on it, we have to ask whether the **data** put it there, or the **prior** did.
"""))
    c.append(chart("nb03_cannibal"))
    c.append(md(r"""
### Is this real, or did the prior decide it for us?

Pause on that verdict. The cannibalization prior is **one-sided** by construction —
$\psi = -\,\mathrm{HalfNormal}(\sigma)$, so $\psi \le 0$ with probability one *before any data*. That means
"the interval is below zero" / $P(\psi < 0) \approx 1$ is **partly automatic**: it can just restate the prior.

> A one-sided / informative prior makes a sign statement like $P(\psi < 0) \approx 1$ nearly vacuous — it can
> just restate the prior. The honest replacement is **contraction** / **overlap**: how much did the data
> *move* and *narrow* the parameter beyond the prior?

So we put the cannibalization effect through the harder test. **Contraction** $c = 1 - \mathrm{Var}_{\text{post}}/\mathrm{Var}_{\text{prior}}$
($c \to 1$ means the data pinned it) and the prior↔posterior **overlap** coefficient ($\to 0$ means the two
densities barely touch — strong learning). Both are computed on the free magnitude `psi_1_0_raw`; they are
sign-invariant, so they describe the signed $\psi$ unchanged.
"""))
    c.append(code(r"""
from mmm_framework.diagnostics import plot_prior_posterior_overlay

lrn_mv = mv.compute_parameter_learning(var_names=["psi_1_0_raw"], prior_samples=2000, random_seed=0)
display(lrn_mv.round(4))
row = lrn_mv[lrn_mv.parameter.str.contains("psi_1_0_raw")].iloc[0]
psi_c, psi_ovl = float(row.contraction), float(row.overlap)

# Overlay the prior vs posterior of the SIGNED cannibalization effect (psi = -psi_1_0_raw).
prior_idata = mv.sample_prior_predictive(samples=2000, random_seed=0)
fig, ax = plt.subplots(figsize=(8, 3.4))
plot_prior_posterior_overlay(prior_idata, mv.trace, "psi_1_0_raw", ax=ax, transform=lambda x: -x)
ax.set_xlabel(r"signed cannibalization effect $\psi$ (Cold Brew $\to$ Original)")
ax.set_title("Did the data move the cannibalization effect, or just the prior?")
plt.tight_layout(); plt.show()

# The data WAS informative about psi (contraction ~1) — but informative about WHAT? Read the value
# it pinned psi to, and translate it to an effect size vs Original's sales.
psi_mean = -float(row.post_mean)                          # signed psi = -psi_1_0_raw
xeff = psi_mean * aurora.sales_coldbrew                    # weekly contribution to Original
share = float(np.abs(xeff).mean() / aurora.sales_original.mean())
r_resid = mv.get_correlation_matrix().iloc[0, 1]
print(f"signed psi posterior mean = {psi_mean:.5f}  -> cross-effect contribution to Original "
      f"= {100*share:.3f}% of its weekly sales")
print(f"residual correlation(Original, Cold Brew) = {r_resid:.3f}  (the shared-demand link)")

# DIRECTIONAL, seeded: the data genuinely pinned psi (not merely restated the one-sided prior)...
assert psi_c > 0.5, f"psi contraction too low ({psi_c:.3f}) — would be prior-dominated, not learned"
assert psi_ovl < 0.3, f"psi prior/posterior overlap too high ({psi_ovl:.3f}) — posterior ~ prior"
# ...and what it pinned it to is essentially zero (a negligible DIRECT cross-effect).
assert share < 0.02, f"direct cross-effect should be a tiny share of sales; got {100*share:.2f}%"
print(f"✓ The data was informative about psi (contraction={psi_c:.3f}, overlap={psi_ovl:.3f}) — "
      f"and it pinned the DIRECT cross-effect to ~0 ({100*share:.2f}% of sales).")
"""))
    c.append(md(r"""
**The honest verdict — and it flips the naive one.** The diagnostic clears the first hurdle: the data *was*
informative about $\psi$ (contraction $\approx 1$, overlap $\approx 0$), so this is **not** a prior artifact.
But "the data spoke" is not "the effect is real" — read *what* it said. The data pins $\psi$ to **essentially
zero** (a direct cross-effect worth a fraction of a percent of Original's sales). So the honest conclusion is
the **opposite** of the sign-only verdict: within this model there is **little direct cannibalization**. The
two products genuinely co-move, but through the **positive residual correlation** (≈0.4 — Aurora's shared
**demand** wave, charted next), not a Cold-Brew→Original cannibalization arrow.

> **The lesson.** $P(\psi<0)\approx 1$ was the near-vacuous part: under a one-sided prior the sign is
> automatic. Contraction tells you the **data was informative**; the posterior **mean** tells you it found a
> **negligible** effect. (Contrast Chapter 2's adstock blend weights, which the data could not pin at all —
> prior-dominated.) The substitution baked into Aurora's data isn't recovered as a *direct* cross-effect by
> this weakly-identified multivariate model; it surfaces as shared-demand co-movement instead.
"""))
    c.append(md(r"""
Here is where the products' co-movement actually lives: even after media is accounted for, the two
products' residuals **move together** — Aurora's shared demand wave lifts Original and Cold Brew at the
same time. This positive correlation, not the (near-zero) direct cross-effect, is the real linkage.
"""))
    c.append(chart("nb03_correlation"))
    c.append(md(r"""
## Extended models serialize too
"""))
    c.append(code(r"""
from pathlib import Path
Path("artifacts").mkdir(exist_ok=True)
nested.save("artifacts/aurora_nested")
reloaded = NestedMMM.load("artifacts/aurora_nested")
print("reloaded nested model, mediators:", reloaded.mediator_names,
      "| trace restored:", reloaded._trace is not None)
"""))
    c.append(md(r"""
### Takeaways
- **NestedMMM** shows TV/Display are **brand engines** — their value lives in the awareness pathway the
  base model is blind to. Mystery of the undervalued channels: solved.
- **MultivariateMMM** ties the two products together through a **shared demand shock** (positive
  residual correlation). The Cold Brew → Original *cross-effect's* interval sits below zero, but the
  **learning diagnostic** shows the data pins that direct effect ~0 — the one-sided prior made the sign
  near-automatic, so the honest read is "co-move via shared demand," and the two products are planned
  *together* for that reason.

Now we have everything: a causal map, experiment anchoring, brand-pathway value, and product
interactions. Time to make it a decision — and a board deck: **`04_reporting.ipynb`**.
"""))
    return c


# ===========================================================================
# Notebook 04 — Reporting
# ===========================================================================


def nb_04():
    c = []
    c.append(md(r"""
# 4 · Reporting — *Putting it in front of the board*

> **Chapter 4 of the Aurora story.** The analysis is done; now it has to land with people who don't read
> trace plots. `mmm_framework` renders a **self-contained, themed HTML report** straight from a fitted
> model — executive summary, ROAS with uncertainty, decomposition, diagnostics, and a causal-assumptions
> section so stakeholders see what the number rests on.
"""))
    c.append(code(SETUP))
    c.append(md(r"""
## Fit (or reuse) the Aurora model, then generate the report

The report builds directly from a fitted `BayesianMMM` — it extracts contributions, ROAS, fit
statistics and diagnostics for you.
"""))
    c.append(code(r"""
from mmm_framework import BayesianMMM, ModelConfigBuilder, SeasonalityConfigBuilder, TrendConfig, TrendType

panel = aurora.base_panel(control_demand=True)
model_config = (ModelConfigBuilder().bayesian_pymc().with_chains(2).with_draws(400).with_tune(400)
                .with_seasonality_builder(SeasonalityConfigBuilder().with_yearly(order=2)).build())
mmm = BayesianMMM(panel, model_config, TrendConfig(type=TrendType.LINEAR))
results = mmm.fit(draws=400, tune=400, chains=2, cores=1, random_seed=0)
"""))
    c.append(code(r"""
from pathlib import Path
from mmm_framework.reporting import (
    MMMReportGenerator, ReportConfig, SectionConfig, ColorScheme, ColorPalette)

Path("artifacts").mkdir(exist_ok=True)
config = ReportConfig(
    title="Aurora Coffee Co. — Marketing Effectiveness",
    subtitle="Q3 Planning Review",
    client="Aurora Coffee Co.",
    analysis_period="2023 – 2024 (weekly)",
    color_scheme=ColorScheme.from_palette(ColorPalette.CORPORATE),
    confidential=True,
    diagnostics=SectionConfig(enabled=True),
)
report = MMMReportGenerator(model=mmm, panel=panel, results=results, config=config)
out = report.to_html("artifacts/aurora_report.html")
print(f"wrote {out}  ({len(report.render()):,} chars of HTML)")
"""))
    c.append(md(r"""
> **Note:** charts use Plotly loaded from a CDN, so rendering needs a network connection. The saved file
> is otherwise fully self-contained.

### Preview it inline
"""))
    c.append(code(r"""
from IPython.display import IFrame
IFrame("artifacts/aurora_report.html", width="100%", height=560)
"""))
    c.append(md(r"""
## Theming & sections

Swap palettes, toggle sections, and brand the deck. There are four built-in palettes and a fluent
`ReportBuilder` for one-liners.
"""))
    c.append(code(r"""
from mmm_framework.reporting import ReportBuilder

# Executive cut: only the sections a CMO wants, in the warm palette.
exec_report = (ReportBuilder()
    .with_model(mmm, panel=panel, results=results)
    .with_title("Aurora — Executive Summary").with_client("Aurora Coffee Co.")
    .enable_all_sections().disable_section("methodology").disable_section("diagnostics")
    .with_credible_interval(0.8)
    .build())
exec_report.to_html("artifacts/aurora_exec_summary.html")

print("Built-in palettes:", [p.name for p in ColorPalette])
print("Saved: artifacts/aurora_report.html (full) + artifacts/aurora_exec_summary.html (exec cut)")
"""))
    c.append(md(r"""
### Takeaways
- One call turns a fitted model into a **board-ready, themed HTML report** with uncertainty baked in.
- Sections are configurable; palettes are brandable; a **causal-assumptions** section makes the
  modelling honest to stakeholders.
- Last stop — tie the whole story into one decision: **`05_unified_workflow.ipynb`**.
"""))
    return c


# ===========================================================================
# Notebook 05 — Unified workflow
# ===========================================================================


def nb_05():
    c = []
    c.append(md(r"""
# 5 · The Unified Workflow — *From question to a defensible plan*

> **The payoff.** One pipeline, the whole Aurora story: state the causal assumptions → anchor to the
> geo-lift experiments → fit → read the corrected ROAS → **reallocate the budget** → report. The point
> isn't the model; it's the **decision** — and how different it is from the dashboard's.
"""))
    c.append(code(SETUP))
    c.append(md(r"""
## The pipeline

We fold the geo-lift experiments for **all four channels** into the likelihood (Aurora ran a full
incrementality program last year), giving an observational model **anchored to randomized evidence**.
"""))
    c.append(code(r"""
from mmm_framework import BayesianMMM, ModelConfigBuilder, SeasonalityConfigBuilder, TrendConfig, TrendType
from mmm_framework.analysis import MMMAnalyzer
from mmm_framework.calibration import ExperimentMeasurement, ExperimentEstimand

window = (str(aurora.weeks[0].date()), str(aurora.weeks[-1].date()))
experiments = [
    ExperimentMeasurement(ch, window, value=float(aurora.true_roas[ch]), se=0.08,
                          estimand=ExperimentEstimand.ROAS)
    for ch in CHANNELS
]
panel = aurora.base_panel(control_demand=True)
cfg = (ModelConfigBuilder().bayesian_pymc().with_chains(2).with_draws(500).with_tune(500)
       .with_seasonality_builder(SeasonalityConfigBuilder().with_yearly(order=2)).build())

mmm = BayesianMMM(panel, cfg, TrendConfig(type=TrendType.LINEAR), experiments=experiments)
results = mmm.fit(draws=500, tune=500, chains=2, cores=1, random_seed=0)
roi = MMMAnalyzer(mmm).compute_channel_roi().set_index("Channel")["ROI"].loc[CHANNELS]
"""))
    c.append(code(r"""
recovered = pd.DataFrame({"true ROAS": aurora.true_roas.loc[CHANNELS], "model ROAS": roi})
display(recovered.round(2))
print("Experiment-anchored model recovers the true ranking:",
      list(recovered["model ROAS"].sort_values(ascending=False).index))
"""))
    c.append(md(r"""
It doesn't just recover the *ranking* — it lands on the true ROAS. Plotted against the truth, every
channel hugs the 45° line.
"""))
    c.append(chart("nb05_recovery"))
    c.append(md(r"""
## The decision: naive plan vs causal plan

Two planners walk in. **The dashboard planner** funds by correlation — more into Search/Social.
**The causal planner** funds by experiment-anchored ROAS — more into TV/Display. We hold the budget
fixed, shift a slice of it each way, and value each shift at the channels' true ROAS. *(This is a
deliberately stylized, first-order comparison — it values the average dollar and holds saturation
fixed — to show the **direction and order of magnitude**, not a precise forecast.)*
"""))
    c.append(code(r"""
spend = aurora.spend[CHANNELS].sum()                     # current annual spend by channel
pool = 0.25 * spend.sum()                                # budget we're willing to move
naive_corr = pd.Series({c: np.corrcoef(aurora.spend[c], aurora.sales_total)[0, 1] for c in CHANNELS})

def allocate(score):
    # Move the pool toward the top-2 channels by `score`, away from the bottom-2.
    rank = score.sort_values(ascending=False)
    winners, losers = rank.index[:2], rank.index[2:]
    delta = pd.Series(0.0, index=CHANNELS)
    delta[winners] = pool / 2
    delta[losers] = -pool / 2
    return delta

naive_delta  = allocate(naive_corr)     # dashboard: chase correlation
causal_delta = allocate(roi)            # causal: chase experiment-anchored ROAS

# Value each plan against the TRUE ROAS (what revenue the shift actually earns).
naive_lift  = float((naive_delta  * aurora.true_roas.loc[CHANNELS]).sum())
causal_lift = float((causal_delta * aurora.true_roas.loc[CHANNELS]).sum())
print(f"Shift ${pool:,.0f}k of budget (vs leaving it where it is):")
print(f"  Dashboard plan (fund {list(allocate(naive_corr)[allocate(naive_corr)>0].index)}):  "
      f"${naive_lift:+,.0f}k revenue")
print(f"  Causal plan    (fund {list(allocate(roi)[allocate(roi)>0].index)}):  "
      f"${causal_lift:+,.0f}k revenue")
print(f"  → Cost of chasing correlation instead of causal ROAS: ~${causal_lift - naive_lift:,.0f}k/yr (illustrative).")
"""))
    c.append(code(r"""
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
x = np.arange(len(CHANNELS))
a1.bar(x-0.2, naive_delta, 0.4, label="dashboard plan", color=PALETTE["berry"])
a1.bar(x+0.2, causal_delta, 0.4, label="causal plan", color=PALETTE["leaf"])
a1.set_xticks(x); a1.set_xticklabels(CHANNELS); a1.axhline(0, color=INK, lw=.8)
a1.set_title("Budget shift by channel ($000s)"); a1.legend()
a2.bar(["dashboard\nplan", "causal\nplan"], [naive_lift, causal_lift],
       color=[PALETTE["berry"], PALETTE["leaf"]])
a2.bar_label(a2.containers[0], fmt="$%.0fk", padding=4)
a2.set_title("Annual revenue impact of the reallocation"); a2.axhline(0, color=INK, lw=.8)
plt.tight_layout(); plt.show()
"""))
    c.append(chart("nb05_allocation"))
    c.append(md(r"""
The two plans are near **mirror images** — the dashboard funds precisely the channels the causal model
defunds. A real planner wouldn't swing the budget fully inverted, so read the dollar figure as an
*illustrative* upper bound, not a forecast. But the lesson is solid and verified: **chasing correlation
moves money the wrong way**, and the gap between the two directions is the price of getting it wrong.

## Ship the decision
"""))
    c.append(code(r"""
from pathlib import Path
from mmm_framework.reporting import MMMReportGenerator, ReportConfig, ColorScheme, ColorPalette
Path("artifacts").mkdir(exist_ok=True)
report = MMMReportGenerator(
    model=mmm, panel=panel, results=results,
    config=ReportConfig(title="Aurora Coffee Co. — Q3 Plan (experiment-anchored)",
                        client="Aurora Coffee Co.", confidential=True,
                        color_scheme=ColorScheme.from_palette(ColorPalette.CORPORATE)))
print("Final board report:", report.to_html("artifacts/aurora_final_plan.html"))
"""))
    c.append(md(r"""
## The Aurora story, in one line

> *A dashboard sees correlations and would have funded a mirage. `mmm_framework` sees a **causal system**
> — confounded spend, a brand-awareness pathway, cannibalizing products — anchors it to **randomized
> experiments**, quantifies the **uncertainty**, and turns it into a **defensible reallocation** worth
> real money.*

| Chapter | What it added |
|---|---|
| 1 · Causality | Made assumptions explicit; proved adjustment isn't enough; **anchored to experiments**. |
| 2 · Base MMM | ROAS & marginal ROAS **with credible intervals**; honest diagnostics. |
| 3 · Extended | TV/Display are **brand engines** (mediation); products **cannibalize**. |
| 4 · Reporting | A **board-ready** deck with the causal assumptions on the table. |
| 5 · Workflow | One pipeline → a reallocation the CFO can sign. |

That's the value of a causal MMM: not a prettier number, a **better decision**.
"""))
    return c


if __name__ == "__main__":
    write_notebook("00_overview.ipynb", nb_00(), "Aurora — Overview")
    write_notebook("01_causality.ipynb", nb_01(), "Aurora — Causality")
    write_notebook("02_base_mmm.ipynb", nb_02(), "Aurora — Base MMM")
    write_notebook("03_extended_mmm.ipynb", nb_03(), "Aurora — Extended MMM")
    write_notebook("04_reporting.ipynb", nb_04(), "Aurora — Reporting")
    write_notebook("05_unified_workflow.ipynb", nb_05(), "Aurora — Unified Workflow")
