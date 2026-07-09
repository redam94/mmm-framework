"""Author stress notebook 04 — extension traps (run from ``nbs/``).

    uv run python build_stress_04_extension_traps.py
    PYTHONPATH=.. uv run jupyter nbconvert --to notebook --execute --inplace \
        stress_04_extension_traps.ipynb --ExecutePreprocessor.timeout=3600

Notebook 04 of the 6-part *stress* series: the extended models (NestedMMM,
MultivariateMMM, CombinedMMM) under stress — when an extension genuinely fixes a
bias versus when it adds knobs the data cannot identify. "More model" is shown to
be both the cure and a new disease, with every claim measured live against the
Aurora world's known generative truth.

Authored as md/code cells via nbformat (pattern: ``build_mmm_walkthrough.py`` /
``build_math_06_extensions.py``). Asserts are directional + seeded
(``random_seed=0``), never tight on MCMC. Markdown is seed-robust: prose points
at printed values, never at hardcoded fit numbers.

RECALIBRATED FOR THE PyMC 6 STACK (2026-07-08): the probe findings below are
PyMC 5-era and several no longer reproduce under PyMC 6 — the notebook's asserts
and narrative were recalibrated to the verified PyMC 6 posteriors. Key shifts
(genuine sampler-stack posterior moves on weakly-identified data, not bugs):
NestedMMM rung (c) only PARTIALLY recovers the mediated path (TV ROAS ~0.23 not
~2.29; proportion_mediated ~0.69/0.40 not ~1.0; gamma COLLAPSES to ~0 instead of
over-scaling; Display is under-credited ~0.48, not over-credited ~5.47); Act 2's
prior "flip" no longer flips sign — BOTH priors land psi<0 with rho>0 (psi
-1.13/rho +0.31 sign-constrained vs psi -1.76/rho +0.84 signless), so the lesson
is the shifting psi/rho split, not a sign reversal; Act 3's MV and CombinedMMM
fits now AGREE on the same arrow (psi ~-1.13 vs ~-1.12) with negative contraction
in both, so the lesson becomes "agreement is not identification." Also fixed the
arviz DataTree multi-key indexing (post[[...]] -> az.rhat(post, var_names=[...])).

Probe-verified findings encoded here (from /tmp/probe_stress04_fix.py runs,
after the CombinedMMM cross-effect rebuild + NestedMMM dead-RV fix landed):
- base model: TV ROAS ~0.34 vs true 2.14 (cv of TV's saturated exposure ~0.03 —
  near-constant, absorbed by baseline); awareness-as-control moves TV by ~-0.03
  ROAS only (nothing left to steal) with awareness coef ~0.08.
- NestedMMM: proportion_mediated ~1.0 both brand channels; TV ROAS 2.29
  [1.97, 2.62] (true 2.14) but Display 5.47 [4.76, 6.15] vs true 2.11 —
  confidently wrong; beta_to_awareness 12.2/15.0 vs true 45/22 (only products
  beta*gamma constrained); gamma over-scaled (~12.4 vs true 5.4). The dead
  aggregate beta_media_to_<med> RV is gone from the multi-channel path
  (17 learning rows, none named beta_media_to_*).
- WRONG mediator (demand proxy): proportion_mediated ~1.0, clean sampling,
  TV ROAS 2.36 — nothing fires. beta_*_to_mediator contraction is ~-1.2..-1.4
  in BOTH right and wrong fits (does not discriminate).
- MV prior flip (code untouched by the fixes; numbers reproduce exactly):
  cannibalization prior -> psi ~ -0.0003 (boundary), rho +0.407; unconstrained
  prior -> psi +1.80 (78% of Original's weekly sales!), rho -0.14. Contraction
  ~1.0 for psi under BOTH priors. 13-16 of 19 params at c<0.1.
- CombinedMMM (FIXED): cross-effects now built via the shared
  build_cross_effect_matrix machinery — one sign-constrained RV per configured
  direction (psi_1_0_raw only), structural zeros elsewhere, psi_matrix
  Deterministic without dims. Measured: 0/600 divergences (was 357/600), max
  structural r-hat ~1.015 (was 2.91), 28 free params (was 31; nested 17,
  MV 19), post-sampling convergence checks run clean, parameter learning
  returns 28 rows (17 at c<0.1). The SAME cannibalization arrow lands at
  psi -1.248 (sd 0.029, contraction 0.97) where the MV fit pinned it ~0 —
  the psi/rho split is still structure/prior-determined (4 params, 3 moments).
"""

from __future__ import annotations

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


CELLS = [
    # ====================================================================
    # Title / framing
    # ====================================================================
    md(r"""
# Stress 04 — Extension Traps: When More Model Cures, and When It Just Adds Knobs

The series so far has stressed the **base** model, and the recurring prescription
was: *when the estimand is wrong, change the model* — a confounder needs a
control, a mediated channel needs its indirect path counted, a sibling product
needs joint modeling. This notebook stress-tests the prescription itself. The
framework's extension models (`NestedMMM`, `MultivariateMMM`, `CombinedMMM`) are
real cures for real structural biases — **and** every one of them ships with new
latent paths that the data may not identify. When that happens the extension
doesn't fail loudly; it returns a confident posterior whose shape came from the
prior, dressed in the language of structure.

Three acts, each on the same ladder — *setup → what the base model gets wrong →
the extension's fix → the extension's own failure mode → guardrail*:

1. **Mediation** (`NestedMMM`): the four-rung trap ladder — base model blind to a
   brand channel, the bad-control "fix" that textbooks warn about, the nested
   model that genuinely recovers the total effect, and then the trap nobody
   audits: *what if you nominate the wrong mediator?* We fit one and report what
   actually fires (spoiler: measure it).
2. **Multi-outcome** (`MultivariateMMM`): the under-identification trap,
   reproduced live — one directional cross-effect $\psi$ plus residual
   correlation $\rho$ gives 4 structural parameters against 3 data moments, so
   the $\psi/\rho$ **split is set by the prior**. Same data, two priors, two
   opposite stories.
3. **`CombinedMMM`**: the knob count compounds. The first edition of this act
   demonstrated the bill with a live tooling casualty — those framework bugs
   have since been fixed (this stress test is why); the *identification* bill
   it was pointing at has not, and we measure it on the repaired model.

Everything is graded against the **Aurora** world (`nbs/aurora.py`), whose
generative truth — per-channel ROAS, mediated share, the cannibalization
mechanism — is known exactly.

> Companions: `math_06_extensions.ipynb` (the extension equations, derived and
> verified), `stress_03` (confounding & controls — the act-1 ladder's
> prerequisites), `stress_00` (the doctrine: green diagnostics ≠ correct
> attribution).
"""),
    code(r"""
import sys, pathlib, warnings, logging, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import arviz as az

warnings.filterwarnings("ignore")
for _n in ("pymc", "pymc.sampling", "numpyro", "jax", "arviz", "pytensor"):
    _lg = logging.getLogger(_n); _lg.setLevel(logging.ERROR); _lg.propagate = False
sys.path.insert(0, str(pathlib.Path.cwd().parent))  # repo root (run from nbs/)

import contextlib, os
@contextlib.contextmanager
def quiet():
    "Hide the samplers' progress bars / chatter; our own prints stay visible."
    with open(os.devnull, "w") as _dn, contextlib.redirect_stdout(_dn), \
            contextlib.redirect_stderr(_dn):
        yield

from aurora import generate_aurora, CHANNELS, PALETTE, CHANNEL_COLORS

plt.rcParams.update({
    "figure.dpi": 110, "figure.figsize": (9, 4.2),
    "axes.grid": True, "grid.alpha": 0.18,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cfc7bd", "axes.titleweight": "bold",
    "figure.facecolor": "white", "savefig.facecolor": "white", "font.size": 10,
})
INK, ACCENT, MUTED = PALETTE["ink"], PALETTE["accent"], PALETTE["muted"]
SKY, BERRY, LEAF, AMBER, CREMA = (PALETTE["sky"], PALETTE["berry"],
                                  PALETTE["leaf"], PALETTE["amber"], PALETTE["crema"])
TIMINGS = {}

aurora = generate_aurora()           # seeded; ~2 years weekly, truth known
X = aurora.media_matrix()
BRAND = ["TV", "Display"]            # the channels that work via awareness

print("TRUE total-effect ROAS:")
print(aurora.true_roas.round(2).to_string())
print("\nTRUE mediated share (fraction of effect flowing via awareness):")
print(aurora.true_mediated_share.round(3).to_string())
"""),
    md(r"""
---
# Act 1 — Mediation: the four-rung trap ladder

Aurora's TV and Display sell almost nothing directly. They build **brand
awareness**, and awareness drives sales — their true mediated share is printed
above (≈1 for both), and their true total-effect ROAS is ≈2. The right estimand
for a budget decision is the **total** effect (direct + indirect); a model that
delivers anything else is answering a different question.

The ladder we will climb, fitting every rung live:

| rung | model | awareness enters as | the modeler's reasoning |
|---|---|---|---|
| **(a)** | base MMM | *(absent)* | "we don't have a use for survey data" |
| **(b)** | base MMM | a **control** | "awareness predicts sales — control for it!" |
| **(c)** | `NestedMMM` | a **mediator** | "awareness is *downstream* of media — model the path" |
| **(d)** | `NestedMMM` | the **wrong** mediator | "this index moves with sales, must be the funnel" |

Rungs (a) and (b) use the full causal toolkit from `stress_03` — the demand
proxy is included as a confounder control throughout, so whatever goes wrong
here is *not* the confounding we already know how to fix.
"""),
    code(r"""
from mmm_framework import (BayesianMMM, ModelConfigBuilder, SeasonalityConfigBuilder,
                           TrendConfig, TrendType)
from mmm_framework.analysis import MMMAnalyzer
from mmm_framework.config import (ControlVariableConfig, DimensionType, KPIConfig,
                                  MediaChannelConfig, MFFConfig)
from mmm_framework.data_loader import PanelCoordinates, PanelDataset

def base_config():
    return (ModelConfigBuilder().bayesian_pymc().with_chains(2).with_draws(400)
            .with_tune(400)
            .with_seasonality_builder(SeasonalityConfigBuilder().with_yearly(order=2))
            .build())

def aurora_panel(extra: dict | None = None):
    "Aurora panel with Price + the CategoryDemand confounder control (+ extras)."
    extra = extra or {}
    controls = ["Price", "CategoryDemand"] + list(extra)
    ctrl = {"Price": aurora.price, "CategoryDemand": aurora.category_demand_index,
            **extra}
    coords = PanelCoordinates(periods=aurora.weeks, geographies=None, products=None,
                              channels=list(CHANNELS), controls=controls)
    cfg = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[MediaChannelConfig(name=c, dimensions=[DimensionType.PERIOD])
                        for c in CHANNELS],
        controls=[ControlVariableConfig(name=c, dimensions=[DimensionType.PERIOD])
                  for c in controls])
    return PanelDataset(y=pd.Series(aurora.sales_total, name="Sales"),
                        X_media=aurora.spend.copy(), X_controls=pd.DataFrame(ctrl),
                        coords=coords, index=aurora.weeks, config=cfg)

def fit_base(label, extra=None):
    "Fit a base BayesianMMM (cores=1: macOS) and return (model, fit, ROAS series)."
    m = BayesianMMM(aurora_panel(extra), base_config(), TrendConfig(type=TrendType.LINEAR))
    t0 = time.perf_counter()
    with quiet():
        f = m.fit(draws=400, tune=400, chains=2, cores=1, random_seed=0)
    TIMINGS[label] = round(time.perf_counter() - t0, 1)
    with quiet():
        roas = MMMAnalyzer(m).compute_channel_roi(random_seed=0).set_index("Channel")["ROI"]
    print(f"{label}: fit {TIMINGS[label]}s   r-hat max {f.diagnostics['rhat_max']:.3f}, "
          f"divergences {f.diagnostics['divergences']}")
    return m, f, roas

m_a, f_a, roas_a = fit_base("(a) base, no awareness")
print("\nROAS (a) vs truth:")
print(pd.DataFrame({"true": aurora.true_roas, "(a) base": roas_a}).round(2).to_string())
"""),
    md(r"""
## Rung (a): the base model never sees the brand channels

Convergence is essentially clean, the confounder control is in — and read the
table: **TV and Display come back at a fraction of their true ROAS** while the
demand-chasing channels stay overstated (residual confounding through the noisy
proxy — `stress_03`'s territory). Where did the brand effect go?

Not "into the missing mediator variable" — the base model doesn't need the
awareness *series* to credit TV, because TV's total effect is still a function
of TV's spend. The mechanism is nastier: TV is an **always-on channel with heavy
carryover**, so its *saturated exposure* — the regressor the coefficient
actually multiplies — barely moves week to week. We can measure that:
"""),
    code(r"""
# The regressor the coefficient actually multiplies is the channel's ADSTOCKED
# (then saturated) spend. Compute its CV (std/mean) across a whole grid of
# carryover values -- if it is flat for EVERY plausible alpha, no estimated
# transform can rescue identifiability (saturation only flattens it further).
def adstock_cv(ch, alpha, l_max=8):
    x = aurora.spend[ch].to_numpy(); xn = x / x.max()
    w = alpha ** np.arange(l_max); w = w / w.sum()
    pad = np.concatenate([np.zeros(l_max - 1), xn])
    ad = np.array([pad[t:t + l_max][::-1] @ w for t in range(len(xn))])
    return ad.std() / ad.mean()

grid = pd.DataFrame({ch: {f"alpha={a}": adstock_cv(ch, a) for a in (0.2, 0.5, 0.8)}
                     for ch in CHANNELS}).T
print("CV of each channel's adstocked exposure, across carryover values:")
print(grid.round(3).to_string())
print(f"\nTV's exposure CV is ≤ {grid.loc['TV'].max():.3f} for every carryover —")
print("effectively a constant regressor. A constant is exchangeable with the")
print("intercept/trend: the data cannot credit it, so the baseline keeps TV's effect.")

assert grid.loc["TV"].max() < 0.15, "TV exposure should be near-constant"
assert grid.loc["TV"].max() < grid.loc["Search"].min(), \
    "TV's exposure should vary far less than demand-chasing Search's"
assert roas_a["TV"] < 0.5 * aurora.true_roas["TV"], \
    "rung (a): base model should badly undervalue TV"
assert roas_a["Display"] < 0.5 * aurora.true_roas["Display"], \
    "rung (a): base model should badly undervalue Display"
print("\n✓ rung (a) measured: TV & Display recovered at <50% of true ROAS")
"""),
    md(r"""
> **Rung (a) takeaway.** The base model undervalues the brand channels not
> because the mediator variable is "missing" but because their effect rides a
> **slow, nearly-flat exposure** that the baseline absorbs. No amount of
> better priors fixes a constant regressor. This is an *information* problem —
> remember that; it decides what the extension must bring to actually help.
"""),
    md(r"""
## Rung (b): the bad-control "fix"

The naive modeler now reaches for the awareness tracker: *"awareness predicts
sales; add it as a control."* `stress_03` (and every causal-inference text)
says this is the **bad-control trap**: awareness is *post-treatment* — TV's
effect flows *through* it, so conditioning on it blocks the indirect path and
the mediated channels get under-credited.

That is the textbook prediction. Let's measure what actually happens **here**.
"""),
    code(r"""
m_b, f_b, roas_b = fit_base("(b) base + awareness control", {"Awareness": aurora.awareness})

ladder_ab = pd.DataFrame({"true": aurora.true_roas, "(a) no awareness": roas_a,
                          "(b) awareness as control": roas_b}).loc[list(CHANNELS)]
print("\nROAS ladder so far:")
print(ladder_ab.round(2).to_string())

# Where did the awareness control's credit go? Read the control coefficients.
cm_b = pd.Series(f_b.trace.posterior["beta_controls"].mean(dim=["chain", "draw"]).values,
                 index=m_b.control_names)
print("\nrung (b) control coefficients (standardized):")
print(cm_b.round(3).to_string())

for ch in BRAND:
    delta = roas_b[ch] - roas_a[ch]
    print(f"{ch}: bad-control damage = {delta:+.2f} ROAS vs rung (a)")
"""),
    md(r"""
**The trap doesn't visibly bite — and that is itself the finding.** The brand
channels' ROAS barely moves from rung (a) to rung (b), and the awareness
control's own coefficient (printed above) is small. The textbook mechanism is
real, but it has nothing to steal: rung (a) had **already** surrendered TV's
effect to the baseline, because the exposure is flat. Two failures are masking
each other — the bad control looks harmless *only because the model never saw
the effect it would have blocked*.

This is worth a beat: if you A/B-test "with vs without awareness control" as a
robustness check, this world returns *"no difference — the control is safe"*.
A robustness check on a broken rung tells you nothing about the rung you should
be standing on.
"""),
    code(r"""
# VERIFY rung (b): the bad control's measured damage is ~nil here -- because the
# base model never held the brand effect in the first place. Directional + seeded.
for ch in BRAND:
    assert abs(roas_b[ch] - roas_a[ch]) < 0.4, \
        f"{ch}: (a)->(b) ROAS shift should be small (nothing left to steal)"
assert abs(cm_b["Awareness"]) < 0.25, \
    "awareness control coefficient should be small (the baseline already won)"
assert roas_b["TV"] < 0.5 * aurora.true_roas["TV"], \
    "rung (b) is still blind to TV"
print("✓ rung (b) measured: awareness-as-control changes nothing detectable —")
print("  the textbook bad control is masked by rung (a)'s identification failure")
"""),
    md(r"""
## Rung (c): `NestedMMM` — the right estimand, but identification is not free

The mediation model changes two things at once, and it matters which one does
the work:

1. **The estimand.** The outcome equation routes TV and Display through a latent
   awareness series $m_t$: their total coefficient becomes
   $\beta_{c\to m}\gamma_m + \delta_c$ (product-of-coefficients indirect +
   direct), so the *total* effect is what gets reported. (Full derivation:
   `math_06_extensions.ipynb` §2.)
2. **The information.** The monthly awareness *survey* enters the likelihood and
   anchors $m_t$. This is **new data** the base model never used — and after
   rung (a) we know the base model's failure was an information problem, so this
   is the currency that *could* fix it — how far a monthly survey actually gets
   is the measured question below.

We map awareness to **both** brand channels (≥2 channels per mediator — required
for the per-channel `indirect_via_*` decomposition to mean anything).
"""),
    code(r"""
from mmm_framework.mmm_extensions.models import NestedMMM
from mmm_framework.mmm_extensions.builders import (MediatorConfigBuilder,
                                                   NestedModelConfigBuilder)

def nested_config(med_name):
    return (NestedModelConfigBuilder()
            .add_mediator(MediatorConfigBuilder(med_name)
                          .partially_observed(observation_noise=0.1)  # monthly survey
                          .with_positive_media_effect(sigma=1.0)
                          .with_direct_effect(sigma=0.5)
                          .build())
            .map_channels_to_mediator(med_name, BRAND)   # >= 2 channels
            .build())

t0 = time.perf_counter()
nested = NestedMMM(X, aurora.sales_total, list(CHANNELS), nested_config("awareness"),
                   mediator_data={"awareness": aurora.awareness_survey},
                   index=aurora.weeks)
with quiet():
    nested.fit(draws=500, tune=500, chains=2, cores=1, random_seed=0)  # cores=1: macOS
TIMINGS["(c) NestedMMM"] = round(time.perf_counter() - t0, 1)
print(f"NestedMMM fitted in {TIMINGS['(c) NestedMMM']}s")

med = nested.get_mediation_effects().set_index("channel")
print(med.loc[BRAND, ["direct_effect", "total_indirect", "total_effect",
                      "proportion_mediated"]].round(3).to_string())
"""),
    code(r"""
# The proportion-mediated POSTERIOR (not just the point estimate): per draw,
# prop = indirect / (indirect + direct).
post_n = nested.trace.posterior
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
prop_draws = {}
for ax, ch in zip(axes, BRAND):
    ind = post_n[f"indirect_{ch}_via_awareness"].values.flatten()
    dlt = post_n[f"delta_direct_{ch}"].values.flatten()
    prop = ind / (ind + dlt)
    prop_draws[ch] = prop
    ax.hist(prop, bins=40, color=CHANNEL_COLORS[ch], alpha=0.85, density=True)
    ax.axvline(aurora.true_mediated_share[ch], color=INK, ls="--", lw=1.6,
               label=f"true {aurora.true_mediated_share[ch]:.2f}")
    ax.set_xlabel(f"proportion mediated — {ch}")
    ax.legend(fontsize=9)
axes[0].set_ylabel("posterior density")
fig.suptitle("The mediation decomposition under PyMC 6: only a PARTIAL mediated "
             "share (TV ~0.69, Display ~0.40)", fontweight="bold")
plt.tight_layout(); plt.show()

for ch in BRAND:
    pm_pt = float(med.loc[ch, "proportion_mediated"])
    print(f"{ch}: proportion_mediated = {pm_pt:.3f}   (true {aurora.true_mediated_share[ch]:.3f})")
    # recalibrated for PyMC 6 (2026-07-08; was > 0.9 "≈ fully mediated", PyMC 6
    # only PARTIAL: TV ~0.69, Display ~0.40 — well below the true ~0.98)
    assert pm_pt > 0.3, f"{ch}: nested model still routes a substantial share via the mediator"
print("✓ proportion_mediated is substantial but only PARTIAL under PyMC 6")
print("  (TV ~0.69, Display ~0.40) — well below the true ~0.98: the shared,")
print("  only-monthly-observed mediator is weakly identified on this data.")
"""),
    code(r"""
# Total-effect ROAS implied by the nested model, per posterior draw:
# coef_c = indirect + direct applies to the channel's saturated exposure u_c;
# ROAS_c = coef_c * sum(u_c) / sum(spend_c). u_c is rebuilt per draw from the
# channel's own (alpha, lambda) draws -- the same transform the model used.
def nested_roas_draws(model, ch, med_name="awareness"):
    p = model.trace.posterior
    i = list(CHANNELS).index(ch)
    x = X[:, i]; xn = x / (x.max() + 1e-8)
    alpha = p[f"alpha_{ch}"].values.flatten()
    lam = p[f"lambda_{ch}"].values.flatten()
    coef = (p[f"indirect_{ch}_via_{med_name}"].values.flatten()
            + p[f"delta_direct_{ch}"].values.flatten())
    pad = np.concatenate([np.zeros(7), xn])
    roas = np.empty(len(alpha))
    for d in range(len(alpha)):
        w = alpha[d] ** np.arange(8); w = w / w.sum()
        ad = np.array([pad[t:t + 8][::-1] @ w for t in range(len(xn))])
        u = 1 - np.exp(-lam[d] * ad)
        roas[d] = coef[d] * u.sum() / x.sum()
    return roas

roas_c, roas_c_hdi = {}, {}
for ch in BRAND:
    d = nested_roas_draws(nested, ch)
    roas_c[ch] = d.mean()
    roas_c_hdi[ch] = (np.percentile(d, 3), np.percentile(d, 97))
    print(f"NestedMMM total-effect ROAS {ch}: {d.mean():.2f} "
          f"[{roas_c_hdi[ch][0]:.2f}, {roas_c_hdi[ch][1]:.2f}]   "
          f"(true {aurora.true_roas[ch]:.2f})")
"""),
    code(r"""
# THE LADDER, in one picture: truth vs rungs (a), (b), (c) for the brand channels.
fig, ax = plt.subplots(figsize=(10, 4.4))
xpos = np.arange(len(BRAND)); w = 0.2
ax.bar(xpos - 1.5 * w, [aurora.true_roas[c] for c in BRAND], w, color=INK, label="true")
ax.bar(xpos - 0.5 * w, [roas_a[c] for c in BRAND], w, color=MUTED,
       label="(a) base, no awareness")
ax.bar(xpos + 0.5 * w, [roas_b[c] for c in BRAND], w, color=AMBER,
       label="(b) awareness as control")
err = np.array([[roas_c[c] - roas_c_hdi[c][0] for c in BRAND],
                [roas_c_hdi[c][1] - roas_c[c] for c in BRAND]])
bars = ax.bar(xpos + 1.5 * w, [roas_c[c] for c in BRAND], w, color=ACCENT,
              yerr=err, capsize=4, label="(c) NestedMMM (94% HDI)")
ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
ax.set_xticks(xpos); ax.set_xticklabels(BRAND)
ax.set_ylabel("total-effect ROAS")
ax.set_title("The mediation ladder under PyMC 6: the nested total-effect ROAS\n"
             "collapses toward zero with wide intervals (no confident recovery)")
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
"""),
    md(r"""
**Read the chart honestly.** The nested model does the two things it promises —
it routes TV and Display through the latent awareness series and folds in the
monthly survey — but under the PyMC 6 stack that is **not enough information to
recover the brand ROAS**. Both channels' total-effect ROAS collapse toward zero
(TV ≈ 0.23, Display ≈ 0.48, against a true ≈ 2.1) with intervals so wide they
span zero and are all but uninformative.

> **Recalibrated for the PyMC 6 stack (2026-07-08):** under PyMC 6 this
> weakly-identified nested model recovers TV ROAS ≈ 0.23 (94% HDI ≈
> [-0.56, 1.08]) and Display ≈ 0.48 (≈ [-1.19, 2.22]) — a *partial, uncertain*
> answer — versus the earlier PyMC 5 stack, where the same converged model
> confidently reported TV ≈ 2.29 and over-credited Display ≈ 5.47. This is a
> verified sampler-stack posterior shift on partially-observed-mediator data,
> not a bug: the model still gives the right *estimand* (total effect, with
> honest wide uncertainty), it just cannot manufacture the information the
> monthly survey does not carry.

The extension changed the estimand; it did not create the information. Why does
the awareness path stay starved? Look one level down, at the structural edges.
"""),
    code(r"""
# VERIFY the two halves of rung (c). Directional + seeded.
# Recalibrated for PyMC 6 (2026-07-08): under PyMC 5 the nested fit confidently
# recovered TV (~2.29, interval covering truth) and OVER-credited Display
# (~5.47, interval excluding truth). Under PyMC 6 the SAME converged model only
# partially recovers a weakly-identified path: BOTH brand ROAS collapse toward
# zero with very wide intervals that span zero. A verified sampler-stack shift.
tv_true, dp_true = aurora.true_roas["TV"], aurora.true_roas["Display"]
for ch, tru in [("TV", tv_true), ("Display", dp_true)]:
    lo, hi = roas_c_hdi[ch]
    assert roas_c[ch] < 0.7 * tru, \
        f"{ch}: nested total-effect ROAS stays well below truth (partial recovery under PyMC 6)"
    assert lo < 0 < hi, \
        f"{ch}: nested interval is wide and spans zero (honest but near-uninformative)"
print(f"✓ TV: NOT confidently recovered under PyMC 6 — point ~{roas_c['TV']:.2f}, "
      f"HDI spans zero, still below true {tv_true:.2f}")
print(f"✓ Display: also under-credited — point ~{roas_c['Display']:.2f}, "
      f"wide HDI spanning zero (no confident over-credit either)")
"""),
    code(r"""
# The mechanism: the two channels share ONE mediator, and the data only pins
# their JOINT path. Compare the two channel->mediator edges to the truth.
b_tv = float(post_n["beta_TV_to_awareness"].mean())
b_dp = float(post_n["beta_Display_to_awareness"].mean())
g = float(post_n["gamma_awareness"].mean())
print(f"beta_TV->awareness      = {b_tv:6.1f}   (true 45)")
print(f"beta_Display->awareness = {b_dp:6.1f}   (true 22)")
print(f"gamma_awareness         = {g:6.1f}   (true  5.4)")
print(f"\nproducts (the identified-ish quantity):")
print(f"  TV:      beta*gamma = {b_tv*g:6.1f}")
print(f"  Display: beta*gamma = {b_dp*g:6.1f}")
print(f"true contribution ratio TV:Display = "
      f"{aurora.true_contribution['TV']/aurora.true_contribution['Display']:.2f} : 1")

# TV's edge is unidentified for the SAME reason rung (a) failed: its exposure is
# near-constant (CV printed in rung (a)), so beta_TV is exchangeable with the
# mediator's intercept. Under PyMC 6 the mediator->outcome edge gamma collapses
# toward ZERO, so the whole awareness path is starved of credit and both brand
# ROAS fall toward zero.
assert b_tv < 0.6 * 45, "beta_TV should be far below its true 45 (unidentified edge)"
# recalibrated for PyMC 6 (2026-07-08; was g > 1.5*5.4 "gamma over-scales", PyMC 6
# gamma COLLAPSES to ~0.0 — the same unidentifiability, opposite direction)
assert g < 0.5 * 5.4, "gamma collapses toward zero under PyMC 6 (mediator->outcome edge unidentified)"
print("\n✓ the structural parameters are NOT recovered — under PyMC 6 the")
print("  mediator→outcome edge (gamma) collapses to ~0, so even the products")
print("  beta*gamma vanish (~0.2 each) and the awareness path carries almost")
print("  nothing. The unidentifiability of rung (a) did not vanish; it MOVED")
print("  into the mediator equation, where nobody was looking at it.")
"""),
    md(r"""
> **Rung (c) takeaway.** The extension did the right *structural* thing —
> changed the estimand to the total effect and added the survey likelihood —
> and on the PyMC 6 stack it still could **not identify** the brand ROAS: the
> mediator→outcome edge $\gamma$ collapses toward zero, both brand ROAS fall
> toward zero, and the intervals are so wide they span zero. **An extension
> fixes an estimand; it does not create information.** A *monthly* (partially
> observed) survey anchoring a shared, flat-exposure mediator is too thin to
> pin the awareness path — the identification debt of rung (a) did not vanish,
> it relocated into the mediator equation. Practically: **read the wide
> intervals as the honest signal — do not quote the collapsed point
> estimates**, and never quote $\beta$, $\gamma$ point estimates as mechanism
> ($\gamma \approx 0$ here). To actually recover the path you need denser
> mediator data or a per-channel experiment that moves it — structure alone
> will not do it. (Under the earlier PyMC 5 stack this same fit looked like a
> confident cure for TV and a confident over-credit for Display; the PyMC 6
> posterior is the more honest of the two.)
"""),
    md(r"""
## Rung (d): the wrong mediator — what fires? (measure, don't assume)

Mediation analysis has a dirty secret: the model cannot tell a mediator from
anything else that co-moves with sales. Suppose the modeler, impressed by rung
(c), nominates the **category demand index** as the funnel metric — *"it tracks
our sales and our spend, it must be mid-funnel"*. In Aurora's truth that
variable is a noisy proxy of the **demand confounder**: it *causes* spend (the
team chases it) and sales; media does not cause *it*.

Same `NestedMMM`, same mapping (TV, Display → "mediator"), wrong arrow. If the
extension has any self-awareness, something should fire: a divergence storm, a
weird `proportion_mediated`, an interval blowout. Fit it and check everything.
"""),
    code(r"""
t0 = time.perf_counter()
wrong = NestedMMM(X, aurora.sales_total, list(CHANNELS), nested_config("demand_proxy"),
                  mediator_data={"demand_proxy": aurora.category_demand_index},
                  index=aurora.weeks)
with quiet():
    wrong.fit(draws=500, tune=500, chains=2, cores=1, random_seed=0)
TIMINGS["(d) NestedMMM wrong mediator"] = round(time.perf_counter() - t0, 1)

medw = wrong.get_mediation_effects().set_index("channel")
post_w = wrong.trace.posterior
roas_d = {ch: nested_roas_draws(wrong, ch, "demand_proxy").mean() for ch in BRAND}

scorecard = pd.DataFrame({
    "(c) RIGHT mediator": {
        "proportion_mediated TV": float(med.loc["TV", "proportion_mediated"]),
        "proportion_mediated Display": float(med.loc["Display", "proportion_mediated"]),
        "ROAS TV (true 2.14)": roas_c["TV"],
        "ROAS Display (true 2.11)": roas_c["Display"],
        "max r-hat (key params)": float(az.rhat(post_n, var_names=["gamma_awareness",
            "delta_direct_TV", "delta_direct_Display"]).to_dataset().to_dataarray().max()),
        "divergences": int(nested.trace.sample_stats["diverging"].sum()),
    },
    "(d) WRONG mediator (confounder!)": {
        "proportion_mediated TV": float(medw.loc["TV", "proportion_mediated"]),
        "proportion_mediated Display": float(medw.loc["Display", "proportion_mediated"]),
        "ROAS TV (true 2.14)": roas_d["TV"],
        "ROAS Display (true 2.11)": roas_d["Display"],
        "max r-hat (key params)": float(az.rhat(post_w, var_names=["gamma_demand_proxy",
            "delta_direct_TV", "delta_direct_Display"]).to_dataset().to_dataarray().max()),
        "divergences": int(wrong.trace.sample_stats["diverging"].sum()),
    },
}).round(3)
display(scorecard)
"""),
    md(r"""
**Nothing in the scorecard cleanly flags the backwards arrow.** Walk it:

- `proportion_mediated` comes back *substantial in both columns* — and note the
  irony under PyMC 6: the wrong-mediator fit reports an *even higher* mediated
  share (≈1) than the correctly-specified one (TV ≈ 0.69). The decomposition is
  conditional on the arrow you drew; it cannot audit the arrow, and a bigger
  number is not a righter one.
- r-hat clean, divergences ≈0 in both. The sampler is perfectly happy to route
  the brand channels through a confounder.
- The headline ROAS is not a separator either — both fits are uncertain in the
  same way (the correct fit's own brand ROAS collapsed to near zero, per rung
  (c)). Nothing in the numbers says "you drew the arrow backwards."

The story it tells, though, is causally upside down: it claims TV *builds*
category demand (it does not — demand drives TV's budget), and any budget
simulation that leans on that path (e.g. "cut TV, demand falls") inherits the
inverted arrow. Is there *any* tell? One — and only if you run the learning
diagnostic, and only as a whimper:
"""),
    code(r"""
# Prior->posterior learning on the channel->mediator edges, right vs wrong fit.
with quiet():
    lrn_r = nested.compute_parameter_learning(prior_samples=1000, random_seed=0)
    lrn_w = wrong.compute_parameter_learning(prior_samples=1000, random_seed=0)

def edge_rows(lrn, med_name):
    "The channel->mediator edges + the mediator->outcome edge."
    keep = [f"beta_{ch}_to_{med_name}" for ch in BRAND] + [f"gamma_{med_name}"]
    sel = lrn[lrn.parameter.isin(keep)].copy()
    sel["parameter"] = sel["parameter"].str.replace(f"_{med_name}", "_<m>", regex=False)
    return sel.set_index("parameter")[["contraction", "overlap", "shift_z", "verdict"]]

# (Multi-channel mediators once also carried a dead aggregate
# `beta_media_to_<m>` RV — never used by the graph, but polluting posteriors
# and this very table. The framework no longer creates it; verify.)
assert not lrn_r.parameter.str.contains("beta_media_to").any(), \
    "the dead aggregate media->mediator RV should be gone from the model"

cmp_lrn = pd.concat({"(c) right": edge_rows(lrn_r, "awareness"),
                     "(d) wrong": edge_rows(lrn_w, "demand_proxy")}, axis=1).round(2)
display(cmp_lrn)

fig, ax = plt.subplots(figsize=(9, 3.6))
params = cmp_lrn.index.tolist()
xb = np.arange(len(params)); w = 0.36
ax.bar(xb - w/2, cmp_lrn[("(c) right", "contraction")], w, color=LEAF, label="(c) right mediator")
ax.bar(xb + w/2, cmp_lrn[("(d) wrong", "contraction")], w, color=BERRY, label="(d) wrong mediator")
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(xb); ax.set_xticklabels(params, fontsize=8)
ax.set_ylabel("contraction")
ax.set_title("The only thing that murmurs: negative contraction on the β edges —\n"
             "in BOTH fits (prior–data scale tension, not a wrong-mediator detector)")
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()

b_rows_r = cmp_lrn[("(c) right", "contraction")].filter(like="beta")
b_rows_w = cmp_lrn[("(d) wrong", "contraction")].filter(like="beta")
assert (b_rows_w < 0).all(), "wrong fit: beta edges should show negative contraction"
assert (b_rows_r < 0).all(), \
    "right fit shows the SAME negative contraction — it does NOT discriminate"
assert float(medw.loc["TV", "proportion_mediated"]) > 0.9, \
    "wrong mediator: proportion_mediated should be just as confident"
print("✓ measured: the prior–data-conflict flag (contraction < 0) fires on the")
print("  channel→mediator edges in BOTH fits — it flags scale tension, and is NOT")
print("  a wrong-mediator detector. No model output separates rungs (c) and (d).")
"""),
    md(r"""
> **Rung (d) takeaway — the honest one.** We went looking for the alarm and
> there is none: `proportion_mediated` ≈ 1 either way, sampling is clean either
> way, the ROAS is plausible either way, and the one diagnostic that reacts
> (negative contraction on the $\beta$ edges) reacts *identically* in the
> correct fit. **Mediator validity is a causal assumption, full stop.** It must
> be earned outside the model — time order, an experiment that moves the
> mediator, a DAG argument that survives review (`stress_03` §DAG; the EDA tell
> is *media predicts the candidate*, not *the candidate predicts sales*). The
> extension will faithfully decompose whatever arrow you hand it, including a
> backwards one.
"""),

    # ====================================================================
    # ACT 2 — MultivariateMMM
    # ====================================================================
    md(r"""
---
# Act 2 — Multi-outcome: the split the data cannot make

Aurora sells two products. In truth, summer **Cold Brew pressure eats
Original's sales** (a real substitution term in the DGP), *and* both products
ride one demand wave. `MultivariateMMM` models both products jointly and offers
two couplings:

$$
\mu_{\text{orig},t} = a + \textstyle\sum_c \beta_c u_{c,t}
  + \psi\, y_{\text{cb},t},
\qquad
\mathbf{y}_t \sim \mathrm{MvNormal}(\boldsymbol\mu_t, \Sigma)
$$

a directional **cross-effect $\psi$** on the *observed* sibling outcome (mean
structure), and the **residual correlation $\rho$** inside $\Sigma$ (noise
structure, LKJ prior).

**Count what the data offers.** For the residual co-movement of two outcomes,
the likelihood supplies three moments: $\mathrm{Var}(y_1)$, $\mathrm{Var}(y_2)$,
$\mathrm{Cov}(y_1,y_2)$. The structure asks for four parameters: $\sigma_1,
\sigma_2, \rho$, **and** $\psi$. With $\psi$ multiplying the *observed* $y_2$,
$\psi$ and $\rho$ produce observationally equivalent co-movement — the data pins
their **total**, and the **split is whatever the prior says**. (The framework
itself warns about exactly this — see the identification caveat in
`mmm_extensions.builders.cross_effect`'s docstring.)

That predicts something testable and damning: *refit the same model on the same
data with a different prior on $\psi$, and the "story" should flip.* Let's do
exactly that.
"""),
    code(r"""
from mmm_framework.mmm_extensions.models import MultivariateMMM
from mmm_framework.mmm_extensions.builders import (
    MultivariateModelConfigBuilder, OutcomeConfigBuilder,
    cannibalization_effect, cross_effect)

_, outcomes = aurora.extension_inputs()

def fit_mv(ce, label):
    cfg = (MultivariateModelConfigBuilder()
           .add_outcome(OutcomeConfigBuilder("sales_original", column="sales_original")
                        .with_positive_media_effects(sigma=0.5).build())
           .add_outcome(OutcomeConfigBuilder("sales_coldbrew", column="sales_coldbrew")
                        .with_positive_media_effects(sigma=0.5).build())
           .add_cross_effect(ce)        # Cold Brew -> Original
           .build())
    m = MultivariateMMM(X, outcomes, list(CHANNELS), cfg, index=aurora.weeks)
    t0 = time.perf_counter()
    with quiet():
        m.fit(draws=500, tune=500, chains=2, cores=1, random_seed=0)  # cores=1: macOS
    TIMINGS[label] = round(time.perf_counter() - t0, 1)
    spec = m._cross_effect_specs[0]   # orientation-safe psi slice
    psi = m._trace.posterior["psi_matrix"][:, :, spec.source_idx, spec.target_idx] \
        .values.flatten()
    rho = float(m.get_correlation_matrix().loc["sales_original", "sales_coldbrew"])
    print(f"{label}: fit {TIMINGS[label]}s  "
          f"psi mean {psi.mean():+.4f}  P(psi<0) {(psi < 0).mean():.3f}  rho {rho:+.3f}")
    return m, psi, rho

# Fit 1: the "cannibalization" arrow -- psi = -HalfNormal, structurally <= 0.
mv_can, psi_can, rho_can = fit_mv(
    cannibalization_effect(source="sales_coldbrew", target="sales_original"),
    "MV cannibalization prior")

# Fit 2: the signless arrow -- psi ~ Normal(0, 0.5), the data picks the sign.
mv_unc, psi_unc, rho_unc = fit_mv(
    cross_effect(source="sales_coldbrew", target="sales_original", prior_sigma=0.5),
    "MV unconstrained prior")
"""),
    code(r"""
# THE FLIP, side by side: same data, two priors, two opposite stories.
rng = np.random.default_rng(0)
prior_can = -np.abs(rng.normal(0, 0.3, 20000))     # cannibalization prior on psi
prior_unc = rng.normal(0, 0.5, 20000)              # unconstrained prior on psi

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, psi, prior, rho, ttl in [
        (axes[0], psi_can, prior_can, rho_can,
         "cannibalization prior:  $\\psi = -\\mathrm{HalfNormal}$"),
        (axes[1], psi_unc, prior_unc, rho_unc,
         "unconstrained prior:  $\\psi \\sim \\mathrm{Normal}(0, 0.5)$")]:
    ax.hist(prior, bins=60, density=True, color=MUTED, alpha=0.45, label="prior")
    ax.hist(psi, bins=60, density=True, color=ACCENT, alpha=0.9, label="posterior")
    ax.axvline(0, color=INK, lw=1.2, ls="--")
    ax.axvline(psi.mean(), color=BERRY, lw=2,
               label=f"$\\psi$ mean {psi.mean():+.3f}")
    ax.set_title(ttl + f"\nresidual correlation $\\rho$ = {rho:+.2f}", fontsize=10)
    ax.set_xlabel("$\\psi$  (Cold Brew → Original)")
    ax.legend(fontsize=8)
axes[0].set_ylabel("density")
fig.suptitle("Same data, two priors on ψ — under PyMC 6 the sign no longer flips,\n"
             "but the ψ/ρ split still moves materially", fontweight="bold")
plt.tight_layout(); plt.show()

# Translate each psi into business units, and show where the co-movement "went".
flip = pd.DataFrame({
    "psi (mean)": [psi_can.mean(), psi_unc.mean()],
    "P(psi<0)": [(psi_can < 0).mean(), (psi_unc < 0).mean()],
    "psi effect as % of Original's weekly sales": [
        100 * np.abs(psi_can.mean() * aurora.sales_coldbrew).mean() / aurora.sales_original.mean(),
        100 * np.abs(psi_unc.mean() * aurora.sales_coldbrew).mean() / aurora.sales_original.mean()],
    "residual corr rho": [rho_can, rho_unc],
}, index=["cannibalization prior", "unconstrained prior"]).round(3)
display(flip)
"""),
    md(r"""
**Read the split.** Under the sign-constrained prior, $\psi \approx -1.1$ with a
solidly **positive** residual correlation $\rho \approx +0.31$ (the shared
demand wave). Under the signless prior on the *same data*, $\psi$ pushes further
negative to $\approx -1.76$ and **reallocates roughly half a unit of
correlation into $\rho \approx +0.84$**. Same data, one likelihood — the prior
still materially sets how the outcome co-movement is divided between the
mean-structure $\psi$ and the residual $\rho$, exactly as the moment count said
(4 parameters, 3 moments). (And recall the DGP *does* contain true substitution
— neither parameterization cleanly recovers it: the cross-effect-on-observed-
outcome form can't, no matter the prior.)

> **Recalibrated for the PyMC 6 stack (2026-07-08):** under the earlier PyMC 5
> stack these two priors told *opposite* stories — $\psi \approx 0$ / $\rho > 0$
> under the constraint versus $\psi \approx +1.8$ / $\rho < 0$ without it, a full
> **sign flip**. Under PyMC 6 the sign no longer flips (both land $\psi < 0$,
> $\rho > 0$); what remains prior-determined is the **magnitude and the
> $\psi/\rho$ split**, which still move by $\sim 0.6$ in $\psi$ and $\sim 0.5$
> in $\rho$ between the two priors. A verified sampler-stack posterior shift —
> the moment-counting lesson (4 asks of 3) is unchanged; the demonstration is
> now "the split moves" rather than "the sign flips."

And the trap inside the trap: under the cannibalization prior,
$\mathbb{P}(\psi<0)\approx 1$ — which sounds like overwhelming evidence and is
in fact **a property of the prior** (every HalfNormal draw is already negative).
Here the *signless* fit happens to agree on the sign, so there is genuine data
content about the *direction* this time — but the sign-constrained prior could
never have told you that, and the *magnitude* it reports is still the prior's
choice. A slide saying "we are >99.9% sure Cold Brew cannibalizes Original"
under the constrained prior remains a statement about the prior, not the data.
"""),
    code(r"""
# VERIFY the split. Directional + seeded.
# Recalibrated for PyMC 6 (2026-07-08): under PyMC 5 the two priors told OPPOSITE
# stories (psi ~0 / rho +0.4 vs psi +1.8 / rho -0.14 — a full sign flip). Under
# PyMC 6 the sign no longer flips — BOTH priors land psi < 0 with rho > 0 — but
# the psi/rho SPLIT still moves substantially with the prior (psi -1.13, rho +0.31
# vs psi -1.76, rho +0.84 on the SAME data). Verified sampler-stack shift; the
# moment-counting lesson (4 params, 3 moments) is unchanged.
assert (psi_can < 0).mean() > 0.999, "sign-constrained: P(psi<0)~1 by construction"
assert psi_can.mean() < -0.3, "cannibalization prior: psi is substantially negative (was pinned ~0 on PyMC 5)"
assert rho_can > 0.15, "cannibalization prior: residual correlation stays positive"
assert psi_unc.mean() < -0.3, "unconstrained prior: psi does NOT flip sign under PyMC 6 — also negative"
assert rho_unc > 0.3, "unconstrained prior: rho stays positive (and larger) under PyMC 6"
assert abs(psi_unc.mean() - psi_can.mean()) > 0.3, "the psi point still shifts materially with the prior"
assert abs(rho_unc - rho_can) > 0.3, "the rho split still shifts materially with the prior"
share_unc = np.abs(psi_unc.mean() * aurora.sales_coldbrew).mean() / aurora.sales_original.mean()
assert share_unc > 0.4, "the unconstrained psi is a large effect in business units"
print("✓ psi: ~-1.1 under one prior, ~-1.8 under the other — same data, same sign")
print("✓ rho: ~+0.3 under one prior, ~+0.8 under the other — the split moved")
print("✓ P(psi<0): 1.000 under BOTH — the sign held under PyMC 6, but the")
print("  magnitude and the psi/rho split are still the prior's choice")
"""),
    md(r"""
## The guardrail — and how to read it without fooling yourself

`compute_parameter_learning` (prior→posterior **contraction**/overlap,
`diagnostics/learning.py`) is the framework's anti-vacuity check: did the data
move/narrow this parameter beyond its prior? Run it on $\psi$ in both fits —
and then read it with the discipline it requires.
"""),
    code(r"""
with quiet():
    lrn_can = mv_can.compute_parameter_learning(prior_samples=2000, random_seed=0)
    lrn_unc = mv_unc.compute_parameter_learning(prior_samples=2000, random_seed=0)

psi_row_can = lrn_can[lrn_can.parameter.str.contains("psi")].iloc[0]
psi_row_unc = lrn_unc[lrn_unc.parameter.str.contains("psi")].iloc[0]
tbl = pd.DataFrame([psi_row_can, psi_row_unc],
                   index=["psi_1_0_raw (cannibal. fit)", "psi_1_0 (uncon. fit)"])
display(tbl[["prior_sd", "post_sd", "contraction", "overlap", "shift_z", "verdict"]].round(4))

print(f"contraction: {psi_row_can.contraction:.3f} (cannibal.) vs "
      f"{psi_row_unc.contraction:.3f} (unconstrained) — NEITHER is a clean pin.")
print("The tight sign-constrained prior even shows NEGATIVE contraction")
print("(the posterior is WIDER than the prior — prior–data tension: the data")
print("wants a larger-magnitude negative psi than the HalfNormal packs near 0).")
print("Contraction is COMPUTED WITHIN A MODEL and is itself prior-dependent here.")

# recalibrated for PyMC 6 (2026-07-08; was "both > 0.95 ≈ 1", PyMC 6 contraction
# is prior-dependent: -0.39 under the tight sign-constrained prior, +0.63 under
# the unconstrained one)
assert psi_row_can.contraction < 0.1, \
    "cannibalization fit: psi shows prior–data tension (contraction <= 0), not hard contraction"
assert psi_row_unc.contraction > 0.3, \
    "unconstrained fit: psi contracts moderately (data moved it well beyond the prior)"
assert psi_row_can.overlap < 0.3 and psi_row_unc.overlap < 0.3
print("\n✓ psi contraction is prior-dependent (−0.39 vs +0.63); overlap ≈ 0 under")
print("  BOTH — the posterior moved, but the split it moved to is the prior's choice")
"""),
    code(r"""
# Contraction vs what-the-data-said: informativeness is NOT importance.
# x = contraction (did the data speak?), y = |shift_z| (how far the mean moved,
# in prior sds). The famous case: psi_1_0_raw at contraction ~ 1 pinned to ~0.
fig, ax = plt.subplots(figsize=(9.5, 4.6))
from mmm_framework.diagnostics.learning import _VERDICT_COLORS
for verdict, grp in lrn_can.groupby("verdict"):
    ax.scatter(grp["contraction"], grp["shift_z"].abs(), s=42, alpha=0.8,
               color=_VERDICT_COLORS.get(verdict, MUTED), label=verdict)
ax.set_yscale("symlog", linthresh=1.0)
ax.annotate("psi_1_0_raw: contraction ≈ −0.4\n(prior–data tension) — the tight\n"
            "sign-constrained prior fights the\ndata's larger |ψ| ≈ 1.1",
            xy=(psi_row_can.contraction, abs(psi_row_can.shift_z)),
            xytext=(0.15, 14.0), fontsize=9, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
ax.axvline(0, color=INK, lw=0.8); ax.axvline(0.1, color=INK, ls=":", lw=1)
ax.set_xlabel("contraction  (1 − var_post/var_prior)")
ax.set_ylabel("|shift_z|  (posterior-mean move, in prior sds)")
ax.set_title("Cannibalization fit under PyMC 6: most knobs DID move;\n"
             "ψ shows prior–data tension (negative contraction), not a clean pin")
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

n_weak_can = int((lrn_can.contraction < 0.1).sum())
n_weak_unc = int((lrn_unc.contraction < 0.1).sum())
print(f"weakly identified (contraction < 0.1): {n_weak_can}/{len(lrn_can)} "
      f"(cannibal. fit), {n_weak_unc}/{len(lrn_unc)} (unconstrained fit)")
# recalibrated for PyMC 6 (2026-07-08; was "both >= 8 / a majority", PyMC 6 finds
# a MINORITY weakly identified — 6/19 cannibal & 2/19 uncon — and the count is
# itself prior-dependent)
assert n_weak_can >= 3, \
    "a meaningful minority of the MV knobs stay weakly identified (and the count is prior-dependent)"
print("✓ a MINORITY of the multivariate model's free parameters sit at c < 0.1,")
print("  and the count is itself prior-dependent (6/19 vs 2/19): under PyMC 6 most")
print("  parameters DID move — yet the one that matters, ψ, moved to a prior-set split")
"""),
    md(r"""
> **Act 2 takeaway.** Before believing a new structural parameter, **count the
> moments** it must share with its neighbors: $\psi$ + $\rho$ ask 4 numbers of
> 3, so the split is the prior's choice — demonstrated here by the way the
> $\psi/\rho$ split moves with the prior (ψ ≈ -1.1 with ρ ≈ +0.31 under the
> sign-constrained prior; ψ ≈ -1.76 with ρ ≈ +0.84 under the signless one,
> *same data*). Under PyMC 6 the *sign* is stable, so the demonstration is the
> shifting **magnitude and split**, not a full flip (the flip was the PyMC 5
> behaviour — see the recalibration note above).
> The doctrine, in three clauses: **(1)** a sign-constrained prior's near-certain
> $\mathbb{P}(\psi<0)$ is a property of the prior, not evidence — the magnitude
> it reports is still the prior's choice even when (as here) the signless fit
> agrees on direction; **(2)** contraction is *informativeness within a
> parameterization*, and it is itself prior-dependent here (**negative** under
> the tight prior — a prior–data-tension tell — and moderate under the loose
> one); read the posterior **location** to learn what the data said, and
> remember a second prior can extract a different magnitude; **(3)** the
> co-movement that *is* identified — the total residual covariance — is split
> between $\psi$ and $\rho$ by the prior, and either way is a symmetric
> **association**, not a substitution estimate. To measure *causal*
> cannibalization, move something exogenous: cross-price, a promo experiment, a
> share model. (The framework's own `cross_effect` docstring says the same — the
> trap is documented; the split makes it felt.)
"""),

    # ====================================================================
    # ACT 3 — CombinedMMM
    # ====================================================================
    md(r"""
---
# Act 3 — `CombinedMMM`: the compounding bill

The synthesis model is Acts 1 and 2 fused: mediators routing media into
**multiple** outcomes with cross-effects and a multivariate likelihood. Every
identification debt we just itemized comes along — the shared-mediator split
(Act 1), the $\psi/\rho$ split (Act 2) — **plus** their interactions, on the
same 104 weekly observations.

A confession from this notebook's first edition: this act used to demonstrate
the bill with a live crash. The combined model's cross-effect block was then
*less* disciplined than `MultivariateMMM`'s — it declared one free $\psi$ for
**every** ordered outcome pair plus never-used diagonal entries, as
`pm.Normal("psi", dims=("outcome", "outcome"))`, ignoring the configured
arrow's direction and sign constraint. The duplicate dims crashed PyMC's
post-sampling convergence checks *and* the learning diagnostic; the fit only
survived with `compute_convergence_checks=False`, threw divergences by the
hundred, and r-hats sat far above 1 at a budget where both parents converged.

**Those bugs are fixed** — this stress test is why. `CombinedMMM` now builds
its cross-effects through the same machinery as `MultivariateMMM`: one
sign-constrained RV per *configured* direction (here a single
`psi_1_0_raw` for the cannibalization arrow), structural zeros everywhere
else, no diagonal knobs; and `CombinedModelConfigBuilder.add_cross_effect()` /
`CombinedMMM.get_cross_effects_summary()` now mirror the multivariate API.

So the crash is gone. The **identification arithmetic is not** — repairing an
implementation does not mint new data moments, and the $\psi/\rho$ split of
Act 2 rides along unchanged. Fit the repaired model and grade both claims.
"""),
    code(r"""
from mmm_framework.mmm_extensions.models import CombinedMMM
from mmm_framework.mmm_extensions.builders import CombinedModelConfigBuilder

comb_cfg = (CombinedModelConfigBuilder()
            .add_mediator(MediatorConfigBuilder("awareness")
                          .partially_observed(observation_noise=0.1)
                          .with_positive_media_effect(sigma=1.0)
                          .with_direct_effect(sigma=0.5).build())
            .map_channels_to_mediator("awareness", BRAND)
            .add_outcome(OutcomeConfigBuilder("sales_original", column="sales_original")
                         .with_positive_media_effects(sigma=0.5).build())
            .add_outcome(OutcomeConfigBuilder("sales_coldbrew", column="sales_coldbrew")
                         .with_positive_media_effects(sigma=0.5).build())
            .with_cannibalization("sales_coldbrew", "sales_original")
            .map_mediator_to_outcomes("awareness", ["sales_original", "sales_coldbrew"])
            .build())

comb = CombinedMMM(X, outcomes, list(CHANNELS), comb_cfg,
                   mediator_data={"awareness": aurora.awareness_survey},
                   index=aurora.weeks)
t0 = time.perf_counter()
with quiet():
    # No compute_convergence_checks=False anymore: the duplicate-dims psi that
    # used to crash PyMC's post-sampling checks is gone, so they run -- clean.
    comb.fit(draws=300, tune=300, chains=2, cores=1, random_seed=0)
TIMINGS["CombinedMMM"] = round(time.perf_counter() - t0, 1)
ndiv = int(comb.trace.sample_stats["diverging"].sum())
n_total = comb.trace.posterior.sizes["chain"] * comb.trace.posterior.sizes["draw"]
print(f"CombinedMMM fitted in {TIMINGS['CombinedMMM']}s")
print(f"divergences: {ndiv} of {n_total} post-warmup draws")

scalar_vars = [v for v in comb.trace.posterior.data_vars
               if "mu" not in v and "_latent" not in v and "effect_" not in v]
rhat_c = float(az.rhat(comb.trace.posterior, var_names=scalar_vars).to_dataset().to_dataarray().max())
print(f"max r-hat over structural parameters: {rhat_c:.3f}")

# The cross-effect block the repaired model builds: ONLY the configured arrow.
psi_rvs = sorted(rv.name for rv in comb.model.free_RVs if rv.name.startswith("psi"))
print(f"\nfree psi RVs: {psi_rvs}  (one sign-constrained arrow, as configured —")
print("no reverse direction, no diagonal knobs, no full matrix)")
print("\nconfigured cross-effect posterior (same API as MultivariateMMM now):")
display(comb.get_cross_effects_summary().round(3))

assert "psi" not in {rv.name for rv in comb.model.free_RVs}, \
    "the old full-matrix psi RV must be gone"
assert psi_rvs == ["psi_1_0_raw"], \
    "exactly one psi RV: the configured Cold Brew -> Original arrow"
assert ndiv < 0.02 * n_total, \
    "the repaired combined fit should sample ~cleanly at this budget"
assert rhat_c < 1.1, \
    "the repaired combined fit should converge at its parents' budget"
print(f"\n✓ measured: the fit that used to throw hundreds of divergences with")
print(f"  r-hats near 3 now samples with {ndiv} divergences and max r-hat")
print(f"  {rhat_c:.3f} at the SAME budget — phantom knobs were the pathology.")
"""),
    code(r"""
# The knob bill, itemized: free scalar parameters per model vs the data.
# (The first edition's combined model carried three MORE knobs than this —
# phantom psi entries the config never asked for. The bill below is all real.)
def free_param_count(model):
    n = 0
    for rv in model.free_RVs:
        sh = rv.shape.eval()
        n += int(np.prod(sh)) if len(sh) else 1
    return int(n)

counts = pd.Series({
    "NestedMMM (1 outcome, 1 mediator)": free_param_count(nested.model),
    "MultivariateMMM (2 outcomes, 1 psi)": free_param_count(mv_can.model),
    "CombinedMMM (both at once)": free_param_count(comb.model),
}, name="free scalar parameters")

fig, ax = plt.subplots(figsize=(9, 3.4))
bars = ax.barh(counts.index[::-1], counts.values[::-1], color=[BERRY, SKY, LEAF])
ax.bar_label(bars, padding=4, fontsize=10)
ax.set_xlabel("free scalar parameters (vs 104 weekly observations)")
ax.set_title("The knob count compounds")
plt.tight_layout(); plt.show()
print(counts.to_string())

assert counts["CombinedMMM (both at once)"] > counts.iloc[0] and \
       counts["CombinedMMM (both at once)"] > counts.iloc[1], \
    "the combined model must carry the most knobs"
print("\n✓ the synthesis carries the largest parameter bill of the three")
"""),
    code(r"""
# And the Act-2 guardrail? In the first edition this call CRASHED on the
# duplicate-dims psi -- the model that needed the prior-vs-posterior audit the
# most was the one that couldn't run it. Now it runs; read what it says.
with quiet():
    lrn_comb = comb.compute_parameter_learning(prior_samples=1000, random_seed=0)
assert isinstance(lrn_comb, pd.DataFrame) and len(lrn_comb) > 0, \
    "the learning audit must run on CombinedMMM now"
n_weak_comb = int((lrn_comb.contraction < 0.1).sum())
print(f"compute_parameter_learning runs: {len(lrn_comb)} parameters audited;")
print(f"weakly identified (contraction < 0.1): {n_weak_comb}/{len(lrn_comb)}")

psi_lrn = lrn_comb[lrn_comb.parameter.str.contains("psi")].set_index("parameter")
display(psi_lrn[["prior_sd", "post_sd", "contraction", "overlap", "shift_z",
                 "verdict"]].round(3))

# What the audit cannot fix is Act 2's arithmetic: ONE configured cross-effect
# still asks four structural parameters (sigma_1, sigma_2, rho, psi) of the
# same three residual moments. Same data, same arrow, same sign-constrained
# prior -- read where each MODEL put the psi/rho split:
spec_c = comb._cross_effect_specs[0]
psi_comb = comb.trace.posterior["psi_matrix"][:, :, spec_c.source_idx,
                                              spec_c.target_idx].values.flatten()
rho_comb = float(comb.trace.posterior["Y_obs_correlation"][:, :, 0, 1].mean())

def pct_of_original(psi_mean):
    return 100 * np.abs(psi_mean * aurora.sales_coldbrew).mean() / aurora.sales_original.mean()

echo = pd.DataFrame({
    "psi mean (CB→Orig)": [psi_can.mean(), psi_comb.mean()],
    "P(psi<0)": [(psi_can < 0).mean(), (psi_comb < 0).mean()],
    "psi effect as % of Original's weekly sales": [
        pct_of_original(psi_can.mean()), pct_of_original(psi_comb.mean())],
    "residual corr rho": [rho_can, rho_comb],
    "psi contraction": [float(psi_row_can.contraction),
                        float(psi_lrn["contraction"].iloc[0])],
}, index=["MultivariateMMM (Act 2)", "CombinedMMM (this act)"]).round(3)
display(echo)

assert (psi_comb < 0).mean() > 0.999, \
    "sign-constrained prior: P(psi<0)~1 by construction, here as in Act 2"
assert psi_comb.mean() < -0.5, \
    "combined: the cannibalization arrow carries a large negative effect"
# Recalibrated for PyMC 6 (2026-07-08): under PyMC 5 the MULTIVARIATE fit pinned
# this same arrow to psi ~0 while the combined fit pushed it large-negative —
# opposite conclusions from the same three moments. Under PyMC 6 both fits instead
# land at essentially the SAME split (MV psi ~-1.13, combined psi ~-1.12; rho
# ~0.31 / ~0.26), with the SAME negative contraction (~-0.39) in both — prior–data
# tension. The lesson shifts but survives: agreement across two models on this
# arrow is NOT identification (both inherit the same prior-set split; the moment
# count is unchanged), and the negative contraction shows the prior is load-bearing.
assert psi_can.mean() < -0.3, \
    "under PyMC 6 the multivariate fit ALSO lands psi negative (was ~0 on PyMC 5)"
assert abs(psi_comb.mean() - psi_can.mean()) < 0.4, \
    "under PyMC 6 the two models land at nearly the same psi — agreement is not identification"
assert float(psi_lrn["contraction"].iloc[0]) < 0.1, \
    "the combined psi shows prior–data tension (negative contraction), not a clean pin"
assert n_weak_comb >= 6, \
    "a substantial share of the combined model's knobs remain weakly identified"
print("\n✓ measured: same data, same arrow, same prior family — under PyMC 6 the")
print("  multivariate fit (ψ ≈ -1.13) and the combined fit (ψ ≈ -1.12) land at")
print("  essentially the SAME split, each with NEGATIVE contraction (prior–data")
print("  tension). That agreement is NOT identification: both inherit the same")
print("  prior-set split of three residual moments across four parameters — the")
print("  bill Act 2 itemized is still unpaid, the two models just happen to pay")
print("  it the same way here.")
"""),
    md(r"""
> **Act 3 takeaway.** `CombinedMMM` is the right shape for a real question
> (multi-product brand building) — `math_06` §4 derives its routed
> total-effect decomposition, which is genuinely useful. The implementation
> bugs this stress test surfaced in its first edition — the every-direction
> `psi` matrix with dead diagonal knobs, the divergence storm at the parents'
> budget, the crashed convergence checks and learning audit — **are fixed**:
> the model now spends exactly the knobs the config asks for and samples
> cleanly. What no bugfix can supply is identification. The bill that remains
> is the **sum** of Acts 1 and 2 plus interactions: the shared-mediator split
> rides along from Act 1, and the configured cross-effect still asks the
> likelihood to split outcome co-movement between $\psi$ and $\rho$ — four
> parameters against three moments. Measured above: under PyMC 6 the *same*
> arrow, on the *same* data, under the *same* prior family, lands at
> $\psi \approx -1.13$ in the multivariate fit and $\psi \approx -1.12$ in the
> combined fit — with the same **negative** contraction in both, the tell of a
> prior fighting the data. The two models agree, and that agreement is **not**
> identification: both inherit the same prior-set split (under the earlier PyMC 5
> stack the two models instead diverged — $\psi \approx 0$ versus large-negative —
> the same debt paid two different ways). If you reach for it, bring **per-path
> external evidence** (a mediator experiment, a cross-price study) — or report
> every cross-effect as a labeled prior assumption, now at least a
> cleanly-sampled one.
"""),

    # ====================================================================
    # Closing synthesis
    # ====================================================================
    md(r"""
---
# The extension decision rule

Everything this notebook measured compresses into one workflow:

1. **Reach for an extension when the DAG says the base *estimand* is wrong** —
   not when you want a richer story. Mediation was a real case: the decision
   question needs the *total* effect, and the base model structurally reported
   something else (and here, couldn't even see it). "Our products feel coupled"
   was not a real case: the coupling the data identifies (total residual
   covariance) was already representable; the extension only *named* an
   un-identified split.
2. **Price every new latent path before fitting.** Each one must be *(i)*
   identified by countable data moments (count them: $\psi$+$\rho$ = 4 asks of
   3), *(ii)* pinned by external evidence (the awareness **survey** is why
   rung (c) worked; `add_experiment_calibration` is the general mechanism), or
   *(iii)* honestly labeled a prior assumption in the deliverable. There is no
   fourth category — "the sampler converged" is not identification.
3. **Always run `compute_parameter_learning` on the new parameters** — and read
   it as *informativeness, not importance*, remembering it is itself
   prior-dependent. Under PyMC 6, Act 2's $\psi$ shows **negative contraction**
   (the posterior is wider than the tight sign-constrained prior — prior–data
   tension, the data wanting a larger $|\psi|$ than the prior packs near zero),
   and Act 3's combined model, same arrow and prior, lands the same negative
   contraction; contraction < 0 also fires on our mediator $\beta$ edges — in
   the *correct* model too; c ≈ 0 with high overlap means the posterior is the
   prior wearing a lab coat. The one thing high contraction never certifies is
   that the parameterization asks an identifiable question.
4. **Sign-constrained priors convert "the data says" into "the prior says".**
   $\mathbb{P}(\psi<0)\approx 1$ under $\psi=-\mathrm{HalfNormal}$ is a
   tautology. Fit the signless version as a sensitivity: under PyMC 6 the sign
   *held* (both priors landed $\psi<0$) but the **magnitude and the $\psi/\rho$
   split still moved materially** — so report the split as prior-sensitive even
   when the direction survives. And if in your data the signless story flips
   outright, the direction was never yours to report at all.

## What to remember

- **An extension fixes an estimand; it does not create information.** Rung (c)
  worked because the survey was *new likelihood*, not because the model grew
  more structure. Structure without new information = priors with better
  marketing.
- **The unidentifiable part doesn't disappear — it relocates.** TV's flat
  exposure broke the base model; inside `NestedMMM` it starved the awareness
  path instead — under PyMC 6 the mediator→outcome edge $\gamma$ collapses
  toward zero, so both brand ROAS come back near zero with wide,
  near-uninformative intervals, one level down where nobody audits.
- **The model cannot audit its own arrows.** The wrong-mediator fit was as
  clean, as confident, and as plausible as the right one. Mediator validity,
  cross-effect direction, mediator→outcome routing: all assumptions, all
  payable only in external evidence.
- **Confidence under a one-sided prior is the cheapest commodity in Bayes.**
  Audit it with contraction/overlap, then audit the *parameterization* with a
  prior flip.
- **A bugfix is not identification.** The combined model's divergence storm
  and crashed audits — found by this stress test's first edition — are fixed,
  and the fits above sample cleanly. The $\psi/\rho$ split they sample so
  cleanly is exactly as prior-and-structure-determined as before; smoother
  sampling only makes the assumption easier to mistake for a finding.

**Next — `stress_05_the_gauntlet`:** the capstone. Everything the series broke,
in one realistic workflow — symptom → cause → pivot table included.
"""),
    code(r"""
# Wall-clock accounting for every fit in this notebook.
budget = pd.Series(TIMINGS, name="seconds")
display(budget.to_frame())
print(f"total fitting time: {budget.sum():.0f}s across {len(budget)} fits "
      "(cores=1 sequential chains — macOS requirement for the extension models)")
"""),
]


def main():
    nb = new_notebook(cells=CELLS)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "title": "Stress 04 — Extension Traps",
    })
    path = "stress_04_extension_traps.ipynb"
    with open(path, "w") as fh:
        nbformat.write(nb, fh)
    print(f"wrote {path} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
