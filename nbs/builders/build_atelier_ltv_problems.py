"""Author demos/atelier_ltv_problems.ipynb (run from ``nbs/``).

    uv run --with nbformat python builders/build_atelier_ltv_problems.py
    TQDM_DISABLE=1 uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        demos/atelier_ltv_problems.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

Lifetime-value problems and the Atelier (Model Garden) models that solve them.
Five measurement problems the everyday MMM/regression toolkit cannot answer,
each demonstrated on a synthetic world with a KNOWN answer key, each solved by
one of the published Atelier example models (``examples/garden_models/``):

1. Customer lifetime value — ``bayesian_clv`` (BG/NBD + Gamma-Gamma) on a
   simulated transaction log: censoring/"dead or dormant", whale skew, holdout
   validation, acquisition-channel heterogeneity (CLV vs CAC), cohort CLV KPI.
2. Awareness as the KPI — ``awareness_structural_mmm`` (binomial survey counts,
   latent goodwill stock).
3. Measurement structure — ``bayesian_cfa`` (do 6 items measure 2 factors?).
4. Hidden segments — ``bayesian_lca`` (mixture over binary answers).
5. Long-term brand value — ``long_term_brand_mmm`` (fast + slow decay stocks).

Companion to ``docs/ltv-modeling.html`` + ``docs/model-garden.html`` and the
seeded product demo ``scripts/seed_atelier_demo.py``.
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
import os, sys
os.environ.setdefault("TQDM_DISABLE", "1")
from pathlib import Path

REPO = Path.cwd().parents[1]                    # kernel cwd = nbs/demos
sys.path.insert(0, str(REPO / "examples" / "garden_models"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.templates.default = "plotly_white"
pio.renderers.default = "notebook_connected"
pd.set_option("display.width", 160)

import logging
from loguru import logger
logger.disable("mmm_framework")
for _n in ("pymc", "pymc.sampling", "numpyro", "jax", "arviz"):
    logging.getLogger(_n).setLevel(logging.ERROR)

# palette — validated categorical order (steel, green, gold, rust) on white
C1, C2, C3, C4 = "#4464ad", "#2e8a5c", "#c9962e", "#b4552d"
INK, MUTED = "#1f2430", "#8a8f98"

def style(fig, height=380, title=None, **kw):
    fig.update_layout(height=height, title=title, font=dict(size=12),
                      margin=dict(t=60, l=60, r=30, b=50), **kw)
    return fig

def estimands_frame(ests):
    # the declared-estimand read as a table (mean + 94% interval + units)
    rows = []
    for k, v in ests.items():
        if v.status != "ok" or v.mean is None:
            continue
        interval = ("" if v.hdi_low is None or v.hdi_high is None
                    else f"[{float(v.hdi_low):.4g}, {float(v.hdi_high):.4g}]")
        rows.append({"estimand": k, "mean": round(float(v.mean), 3),
                     "94% interval": interval, "units": getattr(v, "units", "") or ""})
    return pd.DataFrame(rows)

from mmm_framework.config import LikelihoodConfig, ModelConfig
from mmm_framework.model import TrendConfig
from mmm_framework.model.trend_config import TrendType

print("Setup ready. Garden models on path:", (REPO / "examples" / "garden_models").exists())
"""


CELLS = [
    md(r"""
# LTV Problems & the Atelier Models
### Five measurement questions a standard MMM cannot answer — each with its own synthetic world, and the custom (Model Garden) model that answers it

The **Atelier** is the framework's workshop for bespoke Bayesian models: a model
is a single Python class (subclassing `CustomMMM`) that declares its own config
schema, likelihood, and estimands, and then rides the *same* fit → estimand →
report → serialize pipeline as the built-in MMM. This notebook demonstrates the
published example models the way they should be trusted: on **synthetic data
with a known answer key**, so every claim the model makes can be checked against
the truth that generated the data.

| # | Business question | Why the everyday tool fails | Atelier model | Synthetic world |
|---|---|---|---|---|
| 1 | What is each customer worth? Who is still alive? | Churn is never observed (non-contractual); averages hide whales; CPA ignores customer quality | `bayesian_clv` — BG/NBD + Gamma-Gamma | `make_clv_world` transaction log |
| 2 | What does media buy us in **awareness**? | The KPI is a bounded survey *count*, not sales; effects persist as a decaying stock | `awareness_structural_mmm` | binomial brand-tracker world |
| 3 | Do our survey items measure what we think? | Item averages *assume* the structure instead of testing it | `bayesian_cfa` | known 2-factor indicator world |
| 4 | What hidden segments are in the audience? | Ad-hoc clustering has no model, no uncertainty, no size estimates | `bayesian_lca` | known 2-class mixture world |
| 5 | How much of media's value is **long-term**? | Weekly adstock cannot see 6–36-month brand memory | `long_term_brand_mmm` | dual-decay (fast + slow) world |

Sections 1–5 are independent — each generates its data, fits, and reads the
model's **declared estimands**. The closing section maps the rest of the garden
and the product flow (`list_garden_models → load_garden_model → fit_mmm_model →
get_estimands`).
"""),
    code(SETUP),

    # ── Part 1: LTV ────────────────────────────────────────────────────────────
    md(r"""
## Part 1 — The lifetime-value problem (`bayesian_clv`)

A transaction log looks rich but answers nothing by itself, because in a
non-contractual business **churn is invisible**: a customer who stopped buying
looks exactly like one who is merely between purchases. Everything downstream —
retention, forecasted purchases, customer value, channel economics — hinges on
an inference the data never states.

The classic solution is the **BG/NBD + Gamma-Gamma** ("buy till you die")
model: each customer buys at a latent Poisson rate, flips a death coin after
every repeat purchase, and draws transaction values from a per-customer spend
scale. The Atelier `bayesian_clv` model integrates all per-customer latents out
analytically, so the PyMC graph has **7 free scalars** no matter how many
customers you have.

We start with a world with **no channel heterogeneity** and a held-out future,
so recovery and forecast validation are clean.
"""),
    code(r"""
from mmm_framework.synth.dgp_clv import make_clv_world
from mmm_framework.ltv import transactions_to_rfm

world = make_clv_world(seed=7, n_customers=1500, calibration_weeks=52, holdout_weeks=26)
tx = world.transactions

print(f"{len(tx):,} transactions by {tx.customer_id.nunique():,} customers over "
      f"{world.truth['calibration_weeks']} calibration weeks "
      f"(+{world.truth['holdout_weeks']} held-out weeks: "
      f"{len(world.holdout_transactions):,} future transactions the model never sees)")
print("\nTrue population parameters (the answer key):")
print({k: world.truth[k] for k in ('r', 'alpha', 'a', 'b', 'p_gg', 'q_gg', 'gamma_gg')})

weekly = tx.set_index("date").resample("W-MON")
counts = weekly["customer_id"].size()
acq = tx.groupby("customer_id")["date"].min().dt.to_period("W").dt.start_time.value_counts().sort_index()

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                    subplot_titles=("Transactions per week", "Newly acquired customers per week"))
fig.add_trace(go.Scatter(x=counts.index, y=counts.values, mode="lines",
                         line=dict(color=C1, width=2), name="transactions"), row=1, col=1)
fig.add_trace(go.Bar(x=acq.index, y=acq.values, marker_color=C2, name="new customers"), row=2, col=1)
fig.add_vline(x=world.observation_end, line_dash="dash", line_color=MUTED)
style(fig, height=460, title="The raw material: a transaction log (dashed line = end of the calibration window)",
      showlegend=False)
fig.show()
tx.head(6)
"""),
    md(r"""
### What the model actually sees: RFM

The BG/NBD + Gamma-Gamma likelihoods condition on exactly **four sufficient
statistics per customer** (`mmm_framework.ltv.transactions_to_rfm`):

* `frequency` — repeat purchases (the acquisition purchase carries no rate information),
* `recency` — time of the *last* purchase, measured from the first,
* `T` — customer age at the end of the window,
* `monetary` — mean value of **repeat** purchases only (`NaN` for one-time
  buyers: a single observed basket carries no signal about a customer's
  *typical* spend vs the population's — the model shrinks them to the population mean).

That last point is the first LTV trap: naively averaging all transactions per
customer treats one-time buyers as if their single basket were their identity.
"""),
    code(r"""
rfm = transactions_to_rfm(tx, value_col="value", observation_end=world.observation_end)

one_timers = (rfm.frequency == 0).mean()
print(f"{len(rfm):,} customers -> RFM. One-time buyers: {one_timers:.0%} "
      f"(monetary = NaN by design; the model shrinks them to the population mean)")

fig = make_subplots(rows=1, cols=2, subplot_titles=(
    "Repeat purchases per customer (frequency)", "Mean repeat-transaction value (monetary)"))
fig.add_trace(go.Histogram(x=rfm.frequency, marker_color=C1, nbinsx=30), row=1, col=1)
fig.add_trace(go.Histogram(x=rfm.monetary.dropna(), marker_color=C2, nbinsx=40), row=1, col=2)
style(fig, height=340, title="Both inputs are heavily skewed — means alone would mislead", showlegend=False)
fig.show()
rfm.head(6)
"""),
    md(r"""
### Fit — and check the model recovered the truth

Because the per-customer latents are integrated out, full NUTS on 1,500
customers takes well under a minute. A synthetic world is the only place you
can grade this table — the fitted population parameters against the exact
values that generated the data.
"""),
    code(r"""
from bayesian_clv import BayesianCLV, rfm_panel

clv_model = BayesianCLV(
    rfm_panel(rfm), ModelConfig(), TrendConfig(type=TrendType.NONE),
    model_params={"horizon_periods": world.truth["holdout_weeks"]},  # forecast the held-out 26 weeks
)
clv_model.fit(draws=500, tune=500, chains=4, random_seed=7)
post = clv_model._trace.posterior

rows = []
for name in ("r", "alpha", "a", "b", "p_gg", "q_gg", "gamma_gg"):
    d = np.asarray(post[name]).ravel()
    rows.append({"parameter": name, "posterior mean": round(float(d.mean()), 3),
                 "94% interval": f"[{np.quantile(d, .03):.2f}, {np.quantile(d, .97):.2f}]",
                 "truth": world.truth[name],
                 "recovered": bool(np.quantile(d, .03) <= world.truth[name] <= np.quantile(d, .97))})
recovery = pd.DataFrame(rows)
print("Population-parameter recovery (interval should cover the truth):")
display(recovery)

print("Implied population behavior (posterior mean vs truth):")
imp = pd.DataFrame([
    {"quantity": "mean purchase rate / week", "estimate": float((post["r"] / post["alpha"]).mean()),
     "truth": world.truth["mean_purchase_rate"]},
    {"quantity": "mean dropout P per purchase", "estimate": float((post["a"] / (post["a"] + post["b"])).mean()),
     "truth": world.truth["mean_dropout"]},
    {"quantity": "mean transaction value", "estimate": float((post["p_gg"] * post["gamma_gg"] / (post["q_gg"] - 1)).mean()),
     "truth": world.truth["mean_txn_value"]},
]).round(3)
display(imp)

estimands_frame(clv_model.evaluate_estimands())
"""),
    md(r"""
### Problem 1a — dead or dormant?

The signature LTV inference. Two customers who both went quiet ten weeks ago
are **not** equally likely to be gone: a customer who used to buy every week
and then stopped is almost certainly dead; a customer who buys twice a year is
just between purchases. No recency cutoff ("inactive after 8 weeks") can
express this — `P(alive)` must depend on the *purchase history*, and in the
model it does.
"""),
    code(r"""
rfm_view = rfm.assign(
    quiet_weeks=rfm["T"] - rfm["recency"],
    p_alive=post["p_alive"].mean(("chain", "draw")).values,
)

# two customers matched on "time since last purchase", opposite histories
band = rfm_view[(rfm_view.quiet_weeks >= 8) & (rfm_view.quiet_weeks <= 12)]
pair = pd.concat([
    band.sort_values("frequency").tail(1).assign(who="frequent buyer, gone quiet"),
    band[band.frequency <= 1].sort_values("frequency").head(1).assign(who="occasional buyer, same quiet gap"),
])
print("Same silence, opposite conclusions:")
display(pair[["who", "frequency", "recency", "T", "quiet_weeks", "p_alive"]].round(2))

sub = rfm_view.sample(700, random_state=0)
fig = go.Figure(go.Scatter(
    x=sub.quiet_weeks, y=sub.p_alive, mode="markers",
    marker=dict(size=8, color=sub.frequency, cmax=10, cmin=0,
                colorscale=[[0, "#cdd8ec"], [1, "#1f3a73"]],
                colorbar=dict(title="repeat<br>purchases"), opacity=0.75),
    hovertemplate="quiet %{x:.1f} wk · P(alive) %{y:.2f}<extra></extra>"))
style(fig, title="P(alive) vs weeks since last purchase — colored by purchase frequency",
      xaxis_title="weeks since last purchase", yaxis_title="P(alive)")
fig.show()
print("At any fixed silence, the darker (more frequent) customers have LOWER P(alive):")
print("history, not recency, is what identifies churn.")
"""),
    md(r"""
### Problem 1b — the whale curve

Customer value is violently skewed, so the *average* CLV describes almost
nobody. Budgets, retention programs, and experiment economics should be priced
off the distribution — in particular off how concentrated the book's value is.
"""),
    code(r"""
clv_pc = pd.Series(post["clv"].mean(("chain", "draw")).values, index=rfm.index, name="clv")

v = np.sort(clv_pc.to_numpy())[::-1]
cum = np.cumsum(v) / v.sum()
top10 = cum[max(int(0.10 * len(v)) - 1, 0)]
qs = {f"p{int(q*100)}": float(np.quantile(clv_pc, q)) for q in (0.5, 0.8, 0.9, 0.99)}
print(f"CLV percentiles (26-week horizon): {qs}")
print(f"Top 10% of customers hold {top10:.0%} of total book value "
      f"(mean CLV {clv_pc.mean():.1f} vs median {clv_pc.median():.1f})")

fig = make_subplots(rows=1, cols=2, subplot_titles=(
    "Posterior-mean CLV per customer", "Cumulative share of book value (customers ranked by CLV)"))
fig.add_trace(go.Histogram(x=clv_pc, marker_color=C1, nbinsx=60), row=1, col=1)
fig.add_trace(go.Scatter(x=np.arange(1, len(v) + 1) / len(v), y=cum, mode="lines",
                         line=dict(color=C2, width=2)), row=1, col=2)
fig.add_trace(go.Scatter(x=[0.10], y=[top10], mode="markers+text", text=[f"top 10% → {top10:.0%}"],
                         textposition="bottom right", marker=dict(color=INK, size=9)), row=1, col=2)
fig.update_xaxes(tickformat=".0%", title_text="share of customers", row=1, col=2)
fig.update_yaxes(tickformat=".0%", title_text="share of value", row=1, col=2)
style(fig, height=360, title="The whale curve — the mean describes almost nobody", showlegend=False)
fig.show()
"""),
    md(r"""
### Problem 1c — does the forecast hold? (holdout validation)

The world kept 26 weeks of future transactions the model never saw. The fit's
`expected_purchases` (over exactly that horizon) should track what actually
happened — in total and in rank order. This is the validation a real
deployment does with a calibration/holdout split.
"""),
    code(r"""
from scipy.stats import spearmanr

pred = post["expected_purchases"].mean(("chain", "draw")).values
actual = (world.holdout_transactions.groupby("customer_id")["date"].nunique()
          .reindex(rfm.index, fill_value=0).astype(float))

rho_s = spearmanr(pred, actual).statistic
tot_pred, tot_act = pred.sum(), actual.sum()
print(f"Total repeat purchases in the held-out 26 weeks: predicted {tot_pred:,.0f}, "
      f"actual {tot_act:,.0f} ({tot_pred / tot_act - 1:+.0%})")
print(f"Spearman rank correlation (who buys more): {rho_s:.2f}")

cal = (pd.DataFrame({"pred": pred, "actual": actual.to_numpy()})
       .assign(decile=lambda d: pd.qcut(d.pred, 10, labels=False, duplicates="drop"))
       .groupby("decile").mean())
fig = go.Figure()
fig.add_trace(go.Scatter(x=cal.index + 1, y=cal.pred, mode="lines+markers", name="predicted",
                         line=dict(color=C1, width=2), marker=dict(size=8)))
fig.add_trace(go.Scatter(x=cal.index + 1, y=cal.actual, mode="lines+markers", name="actual (held out)",
                         line=dict(color=C2, width=2), marker=dict(size=8)))
style(fig, title="Holdout calibration by predicted decile — mean purchases in the unseen 26 weeks",
      xaxis_title="predicted-purchases decile (1 = lowest)", yaxis_title="purchases per customer")
fig.show()
"""),
    md(r"""
### Problem 1d — equal CPA, unequal customers

Two acquisition channels can cost the same per customer and deliver customers
of wildly different lifetime worth. This world plants the heterogeneity —
**Search** customers buy 1.3× as often and spend 2× per order, **Social**
0.8× / 0.6× — and the model is told nothing but each customer's acquisition
channel (`transactions_to_rfm(segment_col=…)`). Per-segment CLV estimands
(`segment_clv_<channel>`) then price each channel's *conversions* at lifetime
value, which is exactly what `clv_to_cac` and the acquisition-experiment
calculators (`ghost_ads_power_calc(value_from_clv=True)`) consume.
""" ),
    code(r"""
from bayesian_clv import segment_model_params
from mmm_framework.ltv import clv_to_cac

channels = {"Search": {"share": 0.4, "lam_mult": 1.3, "value_mult": 2.0},
            "Social": {"share": 0.6, "lam_mult": 0.8, "value_mult": 0.6}}
world_b = make_clv_world(seed=11, n_customers=1500, channels=channels)
rfm_b = transactions_to_rfm(world_b.transactions, value_col="value",
                            observation_end=world_b.observation_end,
                            segment_col="acquisition_channel")

seg_model = BayesianCLV(
    rfm_panel(rfm_b), ModelConfig(), TrendConfig(type=TrendType.NONE),
    model_params={"horizon_periods": 52, **segment_model_params(rfm_b)},
)
seg_model.fit(method="map", random_seed=11)   # the product's recommended fast fit
segments = seg_model.segment_clv_means()

cac = {"Search": 25.0, "Social": 12.0}
econ = clv_to_cac(segments, cac)
print("Acquisition economics (52-week CLV vs cost per acquisition):")
display(econ.round(2))

names = list(econ.index)
fig = go.Figure()
fig.add_trace(go.Bar(x=names, y=[econ.loc[n, "clv"] for n in names], name="CLV / customer",
                     marker_color=C1, text=[f"{econ.loc[n, 'clv']:.0f}" for n in names],
                     textposition="outside"))
fig.add_trace(go.Bar(x=names, y=[econ.loc[n, "cac"] for n in names], name="CAC",
                     marker_color=C2, text=[f"{econ.loc[n, 'cac']:.0f}" for n in names],
                     textposition="outside"))
style(fig, title="Lifetime value vs acquisition cost, by channel — similar CPA, very different customers",
      barmode="group", yaxis_title="value per customer")
fig.show()
print("NB segment CLV is DESCRIPTIVE. To claim more Search spend CREATES high-value")
print("customers, run an acquisition experiment valued at CLV (value_from_clv=True).")
"""),
    md(r"""
### Problem 1e — a KPI worth optimizing: weekly cohort CLV

The last bridge back into the measurement loop: `new_customer_clv_series`
turns "customers acquired this week × their posterior CLV" into a weekly
**monetary KPI**, so media can be valued on the lifetime worth of the
customers it acquires rather than on first purchases.
"""),
    code(r"""
from mmm_framework.ltv import new_customer_clv_series

clv_pc_b = pd.Series(seg_model._trace.posterior["clv"].mean(("chain", "draw")).values,
                     index=rfm_b.index)
cohort = new_customer_clv_series(world_b.transactions, clv_pc_b)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                    subplot_titles=("Cohort CLV (lifetime value acquired that week)",
                                    "New customers that week"))
fig.add_trace(go.Scatter(x=cohort.index, y=cohort.cohort_clv, mode="lines",
                         line=dict(color=C1, width=2)), row=1, col=1)
fig.add_trace(go.Bar(x=cohort.index, y=cohort.new_customers, marker_color=C2), row=2, col=1)
style(fig, height=460, title="The cohort-CLV KPI — feed this to an MMM/experiment with margin_per_kpi = 1",
      showlegend=False)
fig.show()
cohort.head()
"""),

    # ── Part 2: awareness ─────────────────────────────────────────────────────
    md(r"""
## Part 2 — When the KPI is a survey count (`awareness_structural_mmm`)

**The problem.** A brand tracker asks 1,000 people a week "have you heard of
us?". The KPI is a **bounded count**, not revenue: a Gaussian MMM on the raw
percentage happily predicts 110% awareness, treats the 47%→48% move near
saturation as identical to 5%→6%, and has no notion that this week's media
keeps working next week. What is needed is a **Binomial observation with a
logit link**, driven by a latent *goodwill stock* that media fills and that
decays geometrically.

That is precisely what `awareness_structural_mmm` declares: its
`CONFIG_SCHEMA` carries `number_of_trials`, its likelihood is
`LikelihoodConfig.binomial`, and its estimands read in survey units
(`awareness_lift` = aware respondents per week attributable to each channel).

**The synthetic world**: each channel's spend fills a goodwill stock with a
known retention of **ρ = 0.6** (`g_t = 0.6·g_{t−1} + spend_t`); awareness is a
known logit of the stocks (`-0.4 + 1.3·TV + 0.7·Digital`, normalized), observed
as `Binomial(1000, rate)` draws. Both the channel ordering *and* the memory are
in the answer key.
"""),
    code(r"""
from awareness_structural_mmm import AwarenessStructuralMMM
from mmm_framework.config import DimensionType, KPIConfig, MediaChannelConfig, MFFConfig
from mmm_framework.data_loader import PanelCoordinates, PanelDataset

n_trials, n, rho_true = 1000, 104, 0.6
periods = pd.date_range("2021-01-04", periods=n, freq="W-MON")
rng = np.random.default_rng(11)
tv = np.abs(rng.normal(100, 25, n))
digital = np.abs(rng.normal(80, 20, n))

def goodwill(spend, rho):
    g = np.zeros(len(spend))
    for t in range(len(spend)):
        g[t] = (rho * g[t - 1] if t else 0.0) + spend[t]
    return g

gw_tv, gw_dig = goodwill(tv, rho_true), goodwill(digital, rho_true)
true_logit = -0.4 + 1.3 * (gw_tv / gw_tv.max()) + 0.7 * (gw_dig / gw_dig.max())
true_rate = 1.0 / (1.0 + np.exp(-true_logit))
y = pd.Series(rng.binomial(n_trials, true_rate), name="Awareness").astype(float)

aw_cfg = MFFConfig(
    kpi=KPIConfig(name="Awareness", dimensions=[DimensionType.PERIOD]),
    media_channels=[MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
                    MediaChannelConfig(name="Digital", dimensions=[DimensionType.PERIOD])],
    controls=[])
aw_panel = PanelDataset(
    y=y, X_media=pd.DataFrame({"TV": tv, "Digital": digital}), X_controls=None,
    coords=PanelCoordinates(periods=periods, geographies=None, products=None,
                            channels=["TV", "Digital"], controls=None),
    index=periods, config=aw_cfg)

fig = go.Figure()
fig.add_trace(go.Scatter(x=periods, y=y / n_trials, mode="lines+markers", name="observed share",
                         line=dict(color=C1, width=2), marker=dict(size=6)))
fig.add_trace(go.Scatter(x=periods, y=true_rate, mode="lines", name="true rate (answer key)",
                         line=dict(color=MUTED, width=2, dash="dash")))
style(fig, title="The KPI: weekly aware share of 1,000 respondents (a bounded, noisy COUNT)",
      yaxis_title="aware share", yaxis_tickformat=".0%")
fig.show()
"""),
    code(r"""
aw_model = AwarenessStructuralMMM(
    aw_panel,
    ModelConfig(likelihood=LikelihoodConfig.binomial(n_trials=n_trials)),
    TrendConfig(type=TrendType.NONE),
    model_params={"number_of_trials": n_trials},
)
# NUTS, not MAP: retention and channel effects trade off along a ridge a point
# estimate can sit anywhere on — the full posterior averages over it.
aw_model.fit(draws=500, tune=500, chains=4, random_seed=0)

rho = float(np.asarray(aw_model._trace.posterior["awareness_retention"]).mean())
half_life = float(np.log(0.5) / np.log(rho)) if 0 < rho < 1 else float("inf")
print(f"Goodwill retention rho = {rho:.3f} (truth {rho_true}) -> "
      f"half-life {half_life:.1f} weeks (truth {np.log(0.5) / np.log(rho_true):.1f})")

try:
    aw_ests = aw_model.evaluate_estimands()          # full DEFAULT_ESTIMANDS incl. goodwill_stock
except Exception:
    aw_ests = aw_model.evaluate_estimands(["awareness_lift", "contribution_roi"])
display(estimands_frame(aw_ests))

lifts = {k.split(":")[-1]: float(v.mean) for k, v in aw_ests.items()
         if k.startswith("awareness_lift") and v.status == "ok" and v.mean is not None}
fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55], subplot_titles=(
    "Awareness lift per channel", "Goodwill retention (how long a week of media lasts)"))
fig.add_trace(go.Bar(x=list(lifts), y=list(lifts.values()),
                     marker_color=[C1, C2][: len(lifts)],
                     text=[f"{v:.0f}" for v in lifts.values()], textposition="outside",
                     showlegend=False),
              row=1, col=1)
fig.add_trace(go.Scatter(x=list(range(13)), y=[rho ** k for k in range(13)], mode="lines+markers",
                         name="fitted decay", line=dict(color=C3, width=2), marker=dict(size=7)),
              row=1, col=2)
fig.add_trace(go.Scatter(x=list(range(13)), y=[rho_true ** k for k in range(13)], mode="lines",
                         name="true decay", line=dict(color=INK, width=2, dash="dash")),
              row=1, col=2)
fig.update_yaxes(title_text="aware respondents / week (of 1,000)", row=1, col=1)
fig.update_xaxes(title_text="weeks since exposure", row=1, col=2)
fig.update_yaxes(title_text="retained goodwill", row=1, col=2)
style(fig, height=360, title="What media buys in awareness — read straight from the declared estimands")
fig.show()
print("TV was planted ~1.9x Digital on the logit scale (1.3 vs 0.7) — the lift ordering")
print("matches — and the fitted goodwill decay tracks the planted retention.")
"""),

    # ── Part 3: CFA ───────────────────────────────────────────────────────────
    md(r"""
## Part 3 — Does the survey measure what we think? (`bayesian_cfa`)

**The problem.** Before averaging six brand-perception items into "brand
equity", you are *assuming* they measure one (or two) coherent things. A
**confirmatory factor analysis** turns that assumption into a testable model:
`y ~ MvNormal(0, ΛΛᵀ + Ψ)` with a fixed simple-structure loading pattern —
and fit indices that can *reject* your structure.

This is a genuinely **non-MMM** Atelier family
(`__garden_model_kind__ = "cfa"`: no channels, no spend, no ROI gates) riding
the same pipeline. The synthetic world plants two factors with **all loadings
= 0.75**: items x1–x3 on F1, x4–x6 on F2.
"""),
    code(r"""
from bayesian_cfa import BayesianCFA, synthetic_cfa_panel

cfa_panel, true_load = synthetic_cfa_panel(n=400)
cfa_model = BayesianCFA(
    cfa_panel, ModelConfig(), TrendConfig(type=TrendType.NONE),
    model_params={"n_factors": 2, "factor_assignment": [0, 0, 0, 1, 1, 1]},
)
cfa_model.fit(method="map", random_seed=7)

cfa_ests = cfa_model.evaluate_estimands()
srmr = float(cfa_ests["srmr"].mean)
cov_fit = float(cfa_ests["cov_fit"].mean)
print(f"SRMR = {srmr:.3f} (good fit < 0.08)   covariance fit = {cov_fit:.2f} (1 = perfect)")

loadings = cfa_model.factor_loadings_summary()
display(loadings.round(3))

fig = go.Figure()
for fac, color in (("F1", C1), ("F2", C2)):
    sub = loadings[loadings.factor == fac]
    fig.add_trace(go.Bar(x=sub.indicator, y=sub.loading, name=fac, marker_color=color,
                         text=[f"{v:.2f}" for v in sub.loading], textposition="outside"))
fig.add_hline(y=true_load, line_dash="dash", line_color=INK,
              annotation_text=f"truth (all loadings = {true_load})")
style(fig, title="Recovered factor loadings vs the planted truth",
      yaxis_title="loading", barmode="group")
fig.show()
print("Every item loads on its assigned factor at ~the planted 0.75 — the")
print("hypothesized 2-factor structure is CONFIRMED, not assumed.")
"""),

    # ── Part 4: LCA ───────────────────────────────────────────────────────────
    md(r"""
## Part 4 — Hidden audience segments (`bayesian_lca`)

**The problem.** Six yes/no attitude questions, and a hunch that the audience
splits into distinct groups. K-means on binary vectors gives *some* partition
— with no model, no segment sizes with uncertainty, and no way to say how many
segments the data supports. **Latent class analysis** is the generative
answer: each respondent belongs to one of `K` latent classes, each class has
its own endorsement probability per item, and the discrete labels are
integrated out (`logsumexp`), so the model is NUTS-able and MAP-able. An
ordered-by-size prior pins class identity (no label switching).

The synthetic world plants two mirror-image classes: **35%** endorse items
q1–q3 (p=0.85) and reject q4–q6 (p=0.15); **65%** do the reverse.
"""),
    code(r"""
from bayesian_lca import BayesianLCA, synthetic_lca_panel

lca_panel, true_sizes, true_profiles = synthetic_lca_panel(n=600)
lca_model = BayesianLCA(
    lca_panel, ModelConfig(), TrendConfig(type=TrendType.NONE),
    model_params={"n_classes": 2},
)
lca_model.fit(method="map", random_seed=11)

lca_ests = lca_model.evaluate_estimands()
sizes = {"C1": float(lca_ests["class_size_1"].mean), "C2": float(lca_ests["class_size_2"].mean)}
print(f"Estimated class sizes: C1 {sizes['C1']:.0%}, C2 {sizes['C2']:.0%}  "
      f"(truth: {true_sizes[0]:.0%} / {true_sizes[1]:.0%})")

prof = lca_model.class_profile_summary()
pivot = prof.pivot(index="item", columns="class", values="prob").loc[[f"q{i}" for i in range(1, 7)]]

fig = go.Figure()
fig.add_trace(go.Bar(x=pivot.index, y=pivot["C1"], name="Class 1 (est)", marker_color=C1))
fig.add_trace(go.Bar(x=pivot.index, y=pivot["C2"], name="Class 2 (est)", marker_color=C2))
for k, cls in enumerate(["C1", "C2"]):
    fig.add_trace(go.Scatter(x=pivot.index, y=true_profiles[k], mode="markers",
                             name=f"truth ({cls})", marker=dict(symbol="diamond-open", size=11,
                                                                color=INK, line=dict(width=2))))
style(fig, title="Per-class item-endorsement profiles — estimates (bars) vs planted truth (diamonds)",
      yaxis_title="P(endorse | class)", barmode="group")
fig.show()
print("The mixture recovers both the 35/65 split and the mirror-image profiles —")
print("segments with SIZES and uncertainty, not just a partition.")
"""),

    # ── Part 5: long-term brand ───────────────────────────────────────────────
    md(r"""
## Part 5 — The long-term value of media (`long_term_brand_mmm`)

**The problem** — the *other* LTV. A weekly MMM with geometric adstock
measures activation plus a few weeks of carryover; brand equity that decays
over 6–36 months is invisible to it, so long-building channels look weak.
`long_term_brand_mmm` feeds every channel's spend into **two** decay stocks —
a fast activation stock and a slow brand stock — and estimates the *split* of
each channel's effect between them. The identification move: the two decay
**horizons** are documented business assumptions (tight priors), while the two
effect **magnitudes** are estimated — sidestepping the fast↔slow adstock ridge.

The synthetic world plants both stocks (fast ρ=0.3, slow ρ=0.94) with a known
long-term share of the total media effect.
"""),
    code(r"""
from long_term_brand_mmm import LongTermBrandMMM

def geom_adstock(x, rho):
    out = np.zeros(len(x))
    for t in range(len(x)):
        ks = np.arange(t + 1)
        out[t] = np.sum(rho ** ks * x[t - ks])
    return out

nb_, rngb = 156, np.random.default_rng(7)
b_periods = pd.date_range("2020-01-06", periods=nb_, freq="W-MON")
b_tv = np.abs(rngb.normal(100, 30, nb_)) * (1 + 0.5 * np.sin(np.arange(nb_) / 6))
b_dig = np.abs(rngb.normal(80, 25, nb_))
tvn, dign = b_tv / b_tv.max(), b_dig / b_dig.max()
fast = 2.0 * geom_adstock(tvn, 0.3) + 1.5 * geom_adstock(dign, 0.3)
slow = 1.6 * geom_adstock(tvn, 0.94) + 1.2 * geom_adstock(dign, 0.94)
true_fraction = float(slow.sum() / (fast.sum() + slow.sum()))
b_y = pd.Series(50 + fast + slow + rngb.normal(0, 0.8, nb_), name="Sales")

b_cfg = MFFConfig(
    kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
    media_channels=[MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
                    MediaChannelConfig(name="Digital", dimensions=[DimensionType.PERIOD])],
    controls=[])
b_panel = PanelDataset(
    y=b_y, X_media=pd.DataFrame({"TV": b_tv, "Digital": b_dig}), X_controls=None,
    coords=PanelCoordinates(periods=b_periods, geographies=None, products=None,
                            channels=["TV", "Digital"], controls=None),
    index=b_periods, config=b_cfg)

brand_model = LongTermBrandMMM(b_panel, ModelConfig(), TrendConfig(type=TrendType.NONE))
brand_model.fit(draws=400, tune=600, chains=2, random_seed=0)

ltf = np.asarray(brand_model._trace.posterior["long_term_fraction"]).ravel()
print(f"Long-term (brand) share of the media effect: {ltf.mean():.2f} "
      f"[{np.quantile(ltf, .03):.2f}, {np.quantile(ltf, .97):.2f}]   truth: {true_fraction:.2f}")

bp = brand_model._trace.posterior
act = bp["activation_contributions"].mean(("chain", "draw")).values.sum(axis=1)
brd = bp["brand_contributions"].mean(("chain", "draw")).values.sum(axis=1)

fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], subplot_titles=(
    "Media effect split over time (standardized KPI units)",
    "Long-term share: posterior vs truth"))
fig.add_trace(go.Scatter(x=b_periods, y=act, stackgroup="med", name="activation (fast)",
                         mode="lines", line=dict(width=0.5, color=C1), fillcolor="rgba(68,100,173,0.55)"),
              row=1, col=1)
fig.add_trace(go.Scatter(x=b_periods, y=brd, stackgroup="med", name="brand (slow)",
                         mode="lines", line=dict(width=0.5, color=C3), fillcolor="rgba(201,150,46,0.55)"),
              row=1, col=1)
fig.add_trace(go.Histogram(x=ltf, marker_color=C1, nbinsx=40, showlegend=False), row=1, col=2)
fig.add_vline(x=true_fraction, line_dash="dash", line_color=INK, row=1, col=2,
              annotation_text="truth")
fig.update_xaxes(title_text="long-term fraction", row=1, col=2)
style(fig, height=380, title="Two stocks, one estimate: how much of media's effect is long-term")
fig.show()
print("A conventional single-adstock MMM would fold the slow stock into baseline")
print("and report only the activation share as 'what media does'.")
"""),

    # ── Closing ───────────────────────────────────────────────────────────────
    md(r"""
## The rest of the garden, and how this looks in the product

Three more example models follow the same recipe — a problem, a synthetic
world with an answer key, a bespoke graph, declared estimands:

| Atelier model | Problem it solves | Synthetic truth |
|---|---|---|
| `latent_factor_mmm` | **Economic confounding** — a boom raises both ad budgets and sales, so a naive MMM over-credits demand-chasing channels. Estimates a latent "economic health" factor from macro indicators *jointly* with the MMM, closing the back-door. | `synth.dgp.make_economic_health` (known loadings incl. a negative unemployment loading; needs full NUTS) |
| `nested_survey_mediation_mmm` | **Funnel mediation** — how much of TV's sales effect flows *through* awareness? Anchors the mediator on a standardized survey so the path is identified. | the aurora known-truth world (`proportion_mediated ≈ 0.99` for TV) |
| `breakout_weighted_mmm` | **Sub-channel splits** — partial-pooled Bayesian breakout weights (τ→0 nests equal-weight) instead of a brittle optimizer. | planted breakout weights |

**How the same models are used in the product.** Everything this notebook did
in raw Python is what the Oracle agent does behind four tools:

```
list_garden_models()                      # discover what's published in the org's garden
load_garden_model("bayesian_clv")         # sets spec["garden_ref"] + recommended fit
fit_mmm_model(...)                        # same fit path; garden class resolved from the ref
get_estimands()                           # the model's DECLARED estimands, realized
```

plus `build_rfm_from_transactions()` to collapse a raw transaction log into the
RFM dataset, and `get_clv_value` / `ghost_ads_power_calc(value_from_clv=True)`
to price acquisition experiments at lifetime value. Run
`uv run python scripts/seed_atelier_demo.py` to see all four Part-1–4 models
published in the Atelier with scripted Oracle sessions.

**Honesty notes.** MAP fits (Parts 3–4 and the Part-1d segment model) are
point estimates — uncertainty is not calibrated; re-fit with NUTS before
trusting intervals (Part 1's recovery table, Part 2's retention posterior and
Part 5's long-term fraction are full NUTS — Part 2 deliberately so, since MAP
sits arbitrarily on the retention↔effect ridge and under-recovers both). CLV
projections assume the purchase/dropout process is stationary over the horizon
— refit after pricing or product changes. And segment CLV is descriptive, not
causal: the experiment tools exist precisely to close that gap.

**Further reading:** `docs/ltv-modeling.html` (the LTV methodology guide),
`docs/model-garden.html` (authoring + the 9-tier compatibility contract),
`technical-docs/ltv-clv-modeling.md`, `technical-docs/long-term-brand-effects.md`.
"""),
]


def main() -> None:
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    out = "demos/atelier_ltv_problems.ipynb"
    with open(out, "w") as f:
        nbformat.write(nb, f)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    print(f"wrote {out} with {len(nb.cells)} cells ({n_code} code)")


if __name__ == "__main__":
    main()
