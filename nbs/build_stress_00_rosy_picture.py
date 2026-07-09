"""Author the series-opener stress notebook (run from ``nbs/``).

    uv run python build_stress_00_rosy_picture.py
    PYTHONPATH=.. uv run jupyter nbconvert --to notebook --execute --inplace \
        stress_00_the_rosy_picture.ipynb --ExecutePreprocessor.timeout=2400

Notebook 00 of the 6-part *stress* series: pressure-testing the MMM framework on
hard, realistic synthetic data. Series thesis: **green diagnostics do not equal
correct attribution** — this opener states the doctrine, demonstrates it live
(clean positive control vs unobserved confounding, identical convergence), loads
the recorded stress matrix, and tours what each diagnostic can and cannot catch.

Authored as md/code cells via nbformat (pattern: ``build_mmm_walkthrough.py``).
Every computational cell ends with asserts encoding its claim — directional and
seed-tolerant for MCMC (pattern: ``build_math_01_adstock.py``). Markdown stays
seed-robust: prose points at the live tables, never at hardcoded fit numbers
(recorded-artifact numbers from ``tests/synth/results/`` are fair game).
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
    # 1. Hook + doctrine
    # ====================================================================
    md(r"""
# Stress 00 — The Rosy Picture

Your MMM converged. R-hat is 1.00 across the board, zero divergences, the
effective sample size is comfortable, and the posterior-mean fit traces the KPI
beautifully. **Should you believe the ROAS table?**

No — not on that evidence. Every check in that list validates the
**computation**: the sampler explored *the posterior of the model you wrote
down*. None of it validates the **causal claim**: that the model you wrote down
attributes sales to the channels that actually caused them. A Markov chain
converges just as cleanly on a confounded regression as on a correct one; the
in-sample fit of a wrong model can be *better* than that of the right one,
because a confounder is a great predictor. Computational diagnostics are
necessary, and nowhere near sufficient.

Call it the **rosy-picture problem**: a workflow whose every gate is green while
the headline numbers — per-channel contribution and ROAS — are silently wrong.
This series pressure-tests `mmm_framework` on synthetic worlds built to break
specific assumptions, with the causal truth known exactly, so the gap between
"all checks pass" and "the answer is right" is *measurable*. This opener:

1. **The estimand doctrine** — what "true contribution" means here and why the
   error is measurable at all.
2. **Positive control, live** — the model *can* work: fit the clean world and
   recover truth.
3. **The punchline, live** — fit a confounded world: *identical* green
   convergence, badly wrong attribution — then **see it**: posterior-predictive
   bands, week-by-week contribution bands against the true paths, and the
   posterior-correlation separability lens (§3.1).
4. **The recorded stress matrix** — a 16-world sweep, 8 silent failures.
5. **The diagnostic toolkit** — what each check actually tests, demonstrated on
   the fits above.
6. **The series roadmap.**

> Companions: `nbs/mmm_walkthrough.ipynb` (the v1→v3 modeling workflow on the
> realistic world), `tests/synth/` (the DGP + harness this notebook reuses), and
> `technical-docs/mmm-robustness-report.md`.
"""),
    code(r"""
import sys, pathlib, warnings, logging, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
# pymc emits sampler warnings at ERROR level -> CRITICAL to keep outputs clean
for _n in ("pymc", "numpyro", "jax", "arviz", "pytensor"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)
try:  # the validation suite logs via loguru, which bypasses stdlib logging
    from loguru import logger as _loguru
    _loguru.disable("mmm_framework")
except ImportError:
    pass
sys.path.insert(0, str(pathlib.Path.cwd().parent))  # repo root (run from nbs/)

import contextlib, os
@contextlib.contextmanager
def quiet():
    "Hide the samplers' progress bars / chatter; our own prints stay visible."
    with open(os.devnull, "w") as _dn, contextlib.redirect_stdout(_dn), \
            contextlib.redirect_stderr(_dn):
        yield

from tests.synth import dgp  # scenario factories with known causal ground truth

plt.rcParams.update({
    "figure.figsize": (9, 4), "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110,
})
INK, ACCENT, SKY, BERRY, LEAF, AMBER, MUTED = (
    "#2b2118", "#b5651d", "#3b6ea5", "#a63a50", "#3f7d5e", "#d98a2b", "#8a8079")
PAL = {"TV": ACCENT, "Search": SKY, "Social": BERRY, "Display": LEAF}
GOOD, BAD = LEAF, BERRY
print("synthetic worlds available:", len(dgp.SCENARIOS))
"""),
    md(r"""
## 1 — The estimand doctrine: truth and estimate must be the same question

A stress test is only as honest as its definition of "truth". Ours comes from
`tests/synth/dgp.py`, and it is chosen to close a loophole that invalidates most
MMM "recovery" demos:

- **Ground truth** for a channel is the **counterfactual zero-out evaluated on
  the noiseless structural mean**: take the world's true response function,
  compute the mean KPI path with all spend as observed, recompute it with that
  one channel's spend set to zero everywhere, and sum the difference. Carryover
  handles itself — zero spend adstocks to zero.
- **The model's estimate** of a channel, via
  `compute_counterfactual_contributions`, is computed *the same way*: zero the
  channel's column, difference the posterior predictions.

Truth and estimate are therefore the **same estimand on the same KPI scale** —
the comparison is apples-to-apples by construction, and the gap between them
*is* the attribution error, not an artifact of mismatched definitions. Two
consequences worth internalizing:

1. For confounded or endogenous worlds, the truth is the **causal** media
   effect: zeroing spend does not change the demand-driven baseline. So when the
   model's estimate departs from it, that gap is exactly the **causal bias** the
   diagnostics ought to catch (and, we will see, mostly don't).
2. Because the estimand is the model's own, *nothing here is unfair to the
   model*. We grade it on the question it claims to answer.

A scenario is **representable** when the truth lies inside the model's
hypothesis space (e.g. positive effects under the positive-only prior). Failures
on representable worlds are the model's problem; failures on unrepresentable
ones are expected by construction and labeled as such.
"""),
    # ====================================================================
    # 2. Positive control
    # ====================================================================
    md(r"""
## 2 — Positive control: the model CAN work

`dgp.build("clean")` draws data from the model's *exact* generative family —
geometric adstock, `1−exp(−λx)` saturation, additive channels, linear trend,
Fourier-representable seasonality, Gaussian noise, and **exogenous** spend.
If the model can't recover this world, nothing downstream means anything.

Fits use a fast-but-honest config (numpyro NUTS, 500 draws × 2 chains,
parametric adstock so the model estimates the same kernel family the world was
generated with). The helper grades every channel against the known truth.
"""),
    code(r"""
from mmm_framework.config import InferenceMethod, ModelConfig
from mmm_framework.model import BayesianMMM, TrendConfig, TrendType

# Importing pymc installs its own log handler; re-quiet and stop propagation.
# (CRITICAL, not ERROR: pymc logs sampler/ESS warnings at ERROR level.)
for _n in ("pymc", "pymc.sampling", "pymc.stats.convergence",
           "numpyro", "jax", "arviz", "pytensor"):
    _lg = logging.getLogger(_n); _lg.setLevel(logging.CRITICAL); _lg.propagate = False

DRAWS, CHAINS = 500, 2
FITS = {}  # label -> (scenario, model, fit, grade table)

def fit_world(name, label=None):
    "Build a scenario, fit the MMM on it, grade every channel vs causal truth."
    label = label or name
    sc = dgp.build(name)
    cfg = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_NUMPYRO,
        n_draws=DRAWS, n_tune=DRAWS, n_chains=CHAINS,
        use_parametric_adstock=True, optim_seed=0,
    )
    mmm = BayesianMMM(sc.panel(), cfg, TrendConfig(type=TrendType.LINEAR))
    t0 = time.perf_counter()
    with quiet():
        fit = mmm.fit(random_seed=0)
        contrib = mmm.compute_counterfactual_contributions(
            compute_uncertainty=True, hdi_prob=0.9, random_seed=0)  # seed: HDI inflates otherwise
    rows = []
    for c in sc.channels:
        t = float(sc.true_contribution[c]); e = float(contrib.total_contributions[c])
        lo = float(contrib.contribution_hdi_low[c]); hi = float(contrib.contribution_hdi_high[c])
        rows.append({"channel": c, "true": round(t), "est": round(e),
                     "lo": round(lo), "hi": round(hi),
                     "rel_err": (e - t) / abs(t), "in 90% HDI": lo <= t <= hi})
    g = pd.DataFrame(rows).set_index("channel")
    FITS[label] = (sc, mmm, fit, g)
    print(f"fit '{label}' in {time.perf_counter() - t0:.0f}s   "
          f"r-hat max={fit.diagnostics['rhat_max']:.3f}  "
          f"divergences={fit.diagnostics['divergences']}  "
          f"min bulk ESS={fit.diagnostics['ess_bulk_min']:.0f}")
    show = g.copy()
    show["rel_err"] = show["rel_err"].map("{:+.0%}".format)
    return sc, mmm, fit, g, show

def truth_vs_est(ax, g, title):
    "Grouped truth-vs-estimate bars; HDI whiskers on the estimate."
    x = np.arange(len(g)); w = 0.38
    ax.bar(x - w / 2, g["true"], w, color=INK, alpha=0.8, label="causal truth")
    yerr = np.vstack([(g["est"] - g["lo"]).clip(lower=0),
                      (g["hi"] - g["est"]).clip(lower=0)])
    ax.bar(x + w / 2, g["est"], w, color=[PAL[c] for c in g.index],
           yerr=yerr, capsize=4, error_kw={"lw": 1.4, "ecolor": INK},
           label="estimate (90% HDI)")
    ax.set_xticks(x); ax.set_xticklabels(g.index)
    ax.set_title(title, fontsize=10); ax.set_ylabel("total contribution (KPI units)")
    ax.legend(fontsize=8)

sc_clean, m_clean, f_clean, g_clean, show = fit_world("clean")
show
"""),
    code(r"""
fig, ax = plt.subplots(figsize=(7.5, 4.2))
truth_vs_est(ax, g_clean, "Clean world (model's own assumptions): truth vs estimate")
plt.tight_layout(); plt.show()

med_err = g_clean["rel_err"].abs().median()
print(f"median |relative error| = {med_err:.0%}   "
      f"coverage = {g_clean['in 90% HDI'].mean():.0%}")

# CLAIM: on data from its own assumptions the model recovers the causal truth —
# decent point estimates and honest intervals — with green convergence.
assert med_err < 0.20, f"clean-world recovery degraded: median |err| {med_err:.0%}"
assert g_clean["in 90% HDI"].mean() >= 0.75, "clean-world coverage collapsed"
assert f_clean.diagnostics["rhat_max"] < 1.05 and f_clean.diagnostics["divergences"] == 0
print("✓ positive control recovered: assumptions hold ⇒ attribution is right")
"""),
    md(r"""
> **Takeaway.** When its assumptions hold, the model earns the rosy picture:
> point estimates land near truth, the 90% intervals cover it, convergence is
> clean. Every failure in the rest of this series is therefore attributable to
> the *broken assumption*, not to the sampler or the implementation. That is
> what a positive control is for.
"""),
    # ====================================================================
    # 3. The punchline: unobserved confounding
    # ====================================================================
    md(r"""
## 3 — Same green diagnostics, wrong answer: unobserved confounding

Now the classic MMM killer. In `dgp.build("unobserved_confounding")` a **latent
demand** signal (seasonal + growth + AR(1), hidden from the model) drives *both*
the baseline **and** the media budgets: Search and Social chase demand the way
real bidding/pacing systems do. Everything else — response curves, noise,
flighting — is the clean world. The only change is that spend is no longer
exogenous: there is an **open back-door** from demand to sales that the model
cannot block, because the confounder is not in the data.

The model will fit this world *happily*. Demand-driven spend is an excellent
predictor of demand-driven sales — the likelihood loves it.
"""),
    code(r"""
sc_conf, m_conf, f_conf, g_conf, show = fit_world("unobserved_confounding")
show
"""),
    code(r"""
# Side by side: identical-looking convergence, very different attribution.
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=False)
truth_vs_est(axes[0], g_clean, "CLEAN — exogenous spend")
truth_vs_est(axes[1], g_conf, "CONFOUNDED — spend chases hidden demand")
for ax, (_, _, f, g) in zip(axes, (FITS["clean"], FITS["unobserved_confounding"])):
    ax.text(0.02, 0.97,
            f"r-hat {f.diagnostics['rhat_max']:.2f} · {f.diagnostics['divergences']} div · "
            f"med |err| {g['rel_err'].abs().median():.0%}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox={"boxstyle": "round", "fc": "white", "ec": MUTED})
fig.suptitle("The rosy-picture problem: the sampler is equally happy on both worlds",
             fontsize=11)
plt.tight_layout(); plt.show()
"""),
    code(r"""
# The two tables an analyst would look at, side by side.
diag = pd.DataFrame({
    lbl: {
        "r-hat max": round(f.diagnostics["rhat_max"], 3),
        "divergences": f.diagnostics["divergences"],
        "min bulk ESS": round(f.diagnostics["ess_bulk_min"]),
    } for lbl, (_, _, f, _) in FITS.items()
})
err = pd.DataFrame({
    lbl: {
        "median |rel err|": f"{g['rel_err'].abs().median():.0%}",
        "worst |rel err|": f"{g['rel_err'].abs().max():.0%}",
        "90% HDI coverage": f"{g['in 90% HDI'].mean():.0%}",
    } for lbl, (_, _, _, g) in FITS.items()
})
print("— sampler diagnostics (what the analyst checks) —"); display(diag)
print("— attribution error vs causal truth (what the analyst can't see) —"); display(err)

chasers = sc_conf.notes["chasers"]
print(f"demand-chasing channels in the DGP: {chasers}")
print(g_conf.loc[chasers, ["true", "est"]].assign(
    rel_err=g_conf.loc[chasers, "rel_err"].map("{:+.0%}".format)))

# CLAIM 1: convergence is green on the CONFOUNDED fit too (the silent part).
assert f_conf.diagnostics["rhat_max"] < 1.05 and f_conf.diagnostics["divergences"] == 0
# CLAIM 2: the demand-chasing channels are materially OVER-credited — they
# inherit the hidden demand's effect through the open back-door (directional,
# seeded). Magnitude varies with defaults (the 2026-06-10 trend-prior fix lets
# the trend absorb part of the demand growth, halving the bias from ~+110% to
# ~+44% — better, still far outside any acceptable error, still mis-ranked).
assert g_conf.loc["Search", "rel_err"] > 0.25, "Search overestimate vanished?"
assert g_conf.loc["Social", "rel_err"] > 0.20, "Social overestimate vanished?"
assert not g_conf.loc["Search", "in 90% HDI"], "Search truth inside HDI?"
# CLAIM 3: attribution is far worse than the clean control's.
assert g_conf["rel_err"].abs().median() > 2 * g_clean["rel_err"].abs().median()
print("\n✓ identical green convergence; the demand-chasers over-credited; intervals confidently wrong")
"""),
    md(r"""
Read the two printed tables together. The **left** one — the one every analyst
checks — cannot tell the worlds apart: both fits converge cleanly at comparable
r-hat, zero divergences, healthy ESS. The **right** one — visible only because
this data is synthetic — shows the confounded fit mis-crediting the
demand-chasing channels by tens of percent, with the truth *outside* the 90%
interval: not just wrong, **confidently** wrong. (In the recorded stress matrix,
run at higher fidelity, the same scenario logs Search at **+153%** with the same
green convergence — see the next section.)

The mechanism is mundane: Search spend rises exactly when hidden demand lifts
the baseline, so the likelihood's cheapest explanation hands demand's effect to
Search. The sampler then converges *correctly* on that wrong posterior. R-hat
measures whether the chains agree — and they agree on the biased answer.

> **Takeaway.** Convergence diagnostics are a property of the **sampler**, not
> of the **causal structure**. They are completely blind to confounding — the
> single most damaging failure mode in observational MMM.
"""),
    # ====================================================================
    # 3.1 Seeing the uncertainty (fit bands, contribution paths, correlation)
    # ====================================================================
    md(r"""
### 3.1 — Seeing the uncertainty you're being asked to trust

Scalar diagnostics compress the posterior to a verdict; **fit plots keep the
uncertainty visible**. Three views, each answering a different question, all
cheap to make from the posterior you already have:

| view | question it answers | built from |
|---|---|---|
| 🟦 **posterior-predictive band** over the KPI | *does the model reproduce the data?* | `mmm.predict()` samples |
| 🟧 **contribution-over-time bands** per channel | *what is each channel doing, week by week — and how sure are we?* | the `channel_contributions` posterior (`(chain, draw, week, channel)`), KPI-scaled by `y_std` |
| 🟥 **posterior correlation of channel totals** | *can the model tell the channels apart?* | the same draws, summed over weeks |

Because this data is synthetic we can overlay the **true weekly contribution
path** on the bands — the view a real analyst never gets, and exactly where the
two worlds finally separate.
"""),
    code(r"""
# View 1 — the posterior-predictive band: BOTH fits reproduce the KPI superbly.
def fit_band(ax, label):
    sc, mmm, fit, _ = FITS[label]
    with quiet():
        pred = mmm.predict(random_seed=0)
    lo, hi = np.percentile(pred.y_pred_samples, [5, 95], axis=0)
    ax.fill_between(sc.weeks, lo, hi, color=SKY, alpha=0.30, label="90% predictive band")
    ax.plot(sc.weeks, pred.y_pred_mean, color=SKY, lw=1.2, label="posterior mean")
    ax.plot(sc.weeks, sc.y, ".", color=INK, ms=2.5, alpha=0.7, label="observed")
    r2 = 1 - np.var(sc.y - pred.y_pred_mean) / np.var(sc.y)
    cover = float(((sc.y >= lo) & (sc.y <= hi)).mean())
    ax.set_title(f"{label}   R²={r2:.2f} · band covers {cover:.0%} of weeks", fontsize=10)
    ax.legend(fontsize=7.5)
    return r2

fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8), sharey=False)
r2_clean = fit_band(axes[0], "clean")
r2_conf = fit_band(axes[1], "unobserved_confounding")
fig.suptitle("View 1 — the fit plot CANNOT separate the worlds: both bands hug the data", fontsize=11)
plt.tight_layout(); plt.show()

# CLAIM: predictive quality is excellent on BOTH fits — fit plots test fit.
assert r2_clean > 0.75 and r2_conf > 0.75
print(f"✓ both fits reproduce the KPI (R² {r2_clean:.2f} vs {r2_conf:.2f}) — "
      "a beautiful fit plot is not evidence of correct attribution")
"""),
    code(r"""
# Views 2-3 share one ingredient: per-draw contribution paths in KPI units.
def contrib_draws(label):
    "channel_contributions posterior -> (n_draws, n_weeks, n_channels), KPI scale."
    _, mmm, fit, _ = FITS[label]
    post = fit.trace.posterior["channel_contributions"].values
    return post.reshape(-1, post.shape[2], post.shape[3]) * mmm.y_std

def true_path(sc, c):
    "The DGP's true weekly contribution of channel c (same estimand, same scale)."
    xn = sc.spend[c].to_numpy(float) / float(sc.spend[c].max())
    return dgp._AMP[c] * dgp._logistic_sat(dgp._geom_adstock(xn, dgp._ALPHA[c]), dgp._LAM[c])

# The weekly truth path integrates to the scenario's recorded total truth —
# the estimand identity that makes the overlay honest:
for lbl in ("clean", "unobserved_confounding"):
    sc = FITS[lbl][0]
    for c in sc.channels:
        assert abs(true_path(sc, c).sum() - sc.true_contribution[c]) < 1e-6

# View 2 — contribution-over-time bands vs the truth path.
SHOW = ["TV", "Search"]   # the best-recovered and the worst-confounded channel
fig, axes = plt.subplots(2, 2, figsize=(12.5, 6.4), sharex="col")
band_stats = {}
for i, lbl in enumerate(["clean", "unobserved_confounding"]):
    sc = FITS[lbl][0]
    draws = contrib_draws(lbl)
    for j, c in enumerate(SHOW):
        ax = axes[i, j]
        ci = sc.channels.index(c)
        lo, hi = np.percentile(draws[:, :, ci], [5, 95], axis=0)
        mean = draws[:, :, ci].mean(axis=0)
        tp = true_path(sc, c)
        inside = float(((tp >= lo) & (tp <= hi)).mean())
        below = float((tp < lo).mean())
        band_stats[(lbl, c)] = {"inside": inside, "truth below band": below}
        ax.fill_between(sc.weeks, lo, hi, color=PAL[c], alpha=0.30,
                        label="90% band (posterior)")
        ax.plot(sc.weeks, mean, color=PAL[c], lw=1.2, label="posterior mean")
        ax.plot(sc.weeks, tp, color=INK, lw=1.4, ls="--", label="TRUE path")
        ax.set_title(f"{lbl} — {c}: truth inside band {inside:.0%} of weeks",
                     fontsize=9.5)
        if i == 0 and j == 0:
            ax.legend(fontsize=7.5)
for ax in axes[:, 0]:
    ax.set_ylabel("weekly contribution (KPI)")
fig.suptitle("View 2 — the week-by-week claim: the confounded fit's Search band "
             "drifts high — over-crediting that drops its coverage of the truth", fontsize=11)
plt.tight_layout(); plt.show()

stats = pd.DataFrame(band_stats).T.map("{:.0%}".format)
display(stats)

# CLAIM 1: on the clean fit the bands are honest — the truth path lives inside.
assert band_stats[("clean", "TV")]["inside"] >= 0.7
assert band_stats[("clean", "Search")]["inside"] >= 0.7
# CLAIM 2 (recalibrated for PyMC 6): the confounded fit over-credits Search here
# too, but the WEEKLY bands are wider than the earlier PyMC 5 stack's, so the miss
# is partial rather than total. The over-crediting still shows: the truth path
# slips BELOW Search's band in a meaningful share of weeks (never on the clean
# fit), and its coverage of the truth DROPS relative to the clean fit — the weekly
# echo of the +42% over-credit in the totals. The headline stays the total.
assert band_stats[("unobserved_confounding", "Search")]["truth below band"] >= 0.20, \
    "confounded Search should over-credit (truth below its band) in a chunk of weeks"  # recalibrated for PyMC 6 (2026-07-08; was >=0.6, PyMC 6 ~0.31)
assert (band_stats[("unobserved_confounding", "Search")]["inside"]
        < band_stats[("clean", "Search")]["inside"]), \
    "confounded Search should cover the truth path LESS than the clean fit"  # recalibrated for PyMC 6 (2026-07-08; was <=0.35 absolute, PyMC 6 conf ~0.69 < clean ~0.85 — now a relative-drop check)
# ...while a non-chasing channel stays honest in the SAME wrong fit:
assert band_stats[("unobserved_confounding", "TV")]["inside"] >= 0.7
print("✓ the clean bands cover the truth; the confounded Search band drifts high — "
      "truth slips below it in a chunk of weeks and its coverage drops vs clean")
"""),
    code(r"""
# View 3 — can the model tell the channels APART? Posterior correlation of
# per-draw channel totals: the joint posterior's own report on separability.
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
corr_stats = {}
for ax, lbl in zip(axes, ["clean", "unobserved_confounding"]):
    sc = FITS[lbl][0]
    totals = contrib_draws(lbl).sum(axis=1)          # (n_draws, n_channels)
    corr = pd.DataFrame(np.corrcoef(totals.T), index=sc.channels, columns=sc.channels)
    corr_stats[lbl] = corr
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(4)); ax.set_xticklabels(sc.channels, rotation=45, fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels(sc.channels, fontsize=8)
    for (a, b), v in np.ndenumerate(corr.values):
        ax.text(b, a, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(v) > 0.6 else INK)
    ax.set_title(lbl, fontsize=10)
fig.colorbar(im, ax=axes, fraction=0.03, label="posterior corr of channel totals")
fig.suptitle("View 3 — channel separability: near-zero in BOTH worlds", fontsize=11)
plt.show()

off = lambda c: c.values[np.triu_indices(4, 1)]
maxcorr_clean = float(np.abs(off(corr_stats["clean"])).max())
maxcorr_conf = float(np.abs(off(corr_stats["unobserved_confounding"])).max())
print(f"max |pairwise corr|: clean {maxcorr_clean:.2f}, confounded {maxcorr_conf:.2f}")

# CLAIM: both posteriors separate the channels cleanly — including the wrong
# one. Correlation diagnoses IDENTIFICATION BETWEEN CHANNELS, not causality.
assert maxcorr_clean < 0.35 and maxcorr_conf < 0.35
print("✓ both fits separate the channels — the confounded posterior is not "
      "confused about the split, it is confidently wrong about the level")
"""),
    md(r"""
**Reading the three views together.**

- **View 1** (predictive band) is a *fit* check with a fit check's blind spot:
  both worlds pass beautifully. If your defense of an MMM is "look how well it
  tracks the data," you have defended the confounded model too.
- **View 2** (contribution bands vs truth) is where the failure *would* be
  visible — the confounded Search band **drifts high**: the truth path slips
  below it in a meaningful share of weeks (never on the clean fit) and its
  coverage of the truth drops relative to clean, the weekly echo of the
  over-credit visible in the totals. On the PyMC 6 stack the weekly bands are
  wide enough that the miss is partial rather than the near-total exclusion the
  earlier stack showed — so the *total* remains the sharper indictment and the
  weekly view is the softer, honest corroboration. The catch is unchanged: the
  dashed truth line is unobservable in real life. What the view gives a real
  analyst is the model's full week-by-week claim with honest widths — the
  artifact to put in front of domain experts and to test against lift windows,
  instead of a single ROAS number stripped of its uncertainty.

> **Recalibrated for the PyMC 6 stack (2026-07-08):** under PyMC 6 this confounded
> fit widens Search's weekly contribution band enough to cover the truth path in
> ~70% of weeks (the truth slips *below* it in ~30%, versus 0% on the clean fit),
> where the earlier PyMC 5 stack excluded the truth almost every week. The totals
> still over-credit Search with the truth outside the 90% HDI — a verified
> sampler-stack posterior shift, not a bug: the confounding bias is intact, only
> its week-by-week signature softened.
- **View 3** (posterior correlation of channel totals) reports how much the
  channel estimates *trade off against each other* inside the joint posterior.
  Here both worlds show ~zero — the flighting calendars are distinct, so the
  likelihood can tell the channels apart, **even in the world where the levels
  are wrong**. The lens that stays quiet here gets loud exactly where it should:
  on the shared-calendar collinear world of
  [stress_03](stress_03_confounding_and_selection.ipynb), these correlations go
  strongly negative — the posterior's way of saying *"the split is a seesaw."*

> **Takeaway.** Plot the uncertainty, not just the verdicts: predictive bands
> validate *fit*, contribution bands expose the model's *actual week-by-week
> claim*, and the totals' correlation matrix reports *which numbers are jointly
> pinned down vs traded off*. None of them detects confounding — that lesson
> stands — but they turn "trust the table" into reviewable, testable claims.
"""),
    # ====================================================================
    # 4. The recorded stress matrix
    # ====================================================================
    md(r"""
## 4 — The full stress matrix: eight silent failures

The two fits above are one row each of a systematic sweep. The harness in
`tests/synth/harness.py` fits the *same* model on 16 worlds — fourteen each
breaking one specific assumption, plus the clean control and a kitchen-sink
capstone — and
records recovery error, interval coverage, and every diagnostic. The recorded
results live in `tests/synth/results/stress_matrix.csv` (rendered prose:
`stress_matrix.md`; method: `technical-docs/mmm-robustness-report.md`).

The harness's **silent failure** definition, exactly:

- the world is **representable** (truth inside the model's hypothesis space), and
- recovery is **wrong** — median per-channel |error| > 25%, *or* some channel
  off by > 50%, *or* coverage of the truth < 75% — and
- the checks an analyst acts on are **green**: r-hat < 1.05, zero divergences,
  and no channel flagged by the unobserved-confounding robustness value.

PPC and the causal refutation suite are *reported* but **excluded from the
gate**: both are fit-level checks whose verdicts have proven config-sensitive —
under the pre-fix trend prior both false-alarmed on the clean control, and under
current defaults they pass on clean *and* on several wrong-attribution worlds
(§5). A check whose verdict tracks the sampler configuration rather than the
causal claim cannot serve as an all-clear either way.
"""),
    code(r"""
mx = pd.read_csv("../tests/synth/results/stress_matrix.csv")

# Recompute the harness verdict from its recorded ingredients (gate above).
green = (mx["rhat_max"] < 1.05) & (mx["divergences"] == 0) & (mx["fragile_channels"] == "-")
bad = ((mx["median_abs_rel_error"] > 0.25) | (mx["max_abs_rel_error"] > 0.50)
       | (mx["coverage_rate"] < 0.75))
recomputed = mx["representable"] & green & bad
assert (recomputed == mx["silent_failure"]).all(), "gate drifted from recorded verdicts"

view = pd.DataFrame({
    "assumption broken": mx["violates"].fillna("— (control)").replace("", "— (control)"),
    "med |err|": mx["median_abs_rel_error"].map("{:.0%}".format),
    "max |err|": mx["max_abs_rel_error"].map("{:.0%}".format),
    "coverage": mx["coverage_rate"].map("{:.0%}".format),
    "r-hat": mx["rhat_max"].round(2),
    "div": mx["divergences"],
    "ppc": np.where(mx["ppc_pass"], "pass", "FAIL"),
    "refut": np.where(mx["refutation_all_passed"], "pass", "FAIL"),
    "verdict": np.where(mx["silent_failure"], "🔴 SILENT FAILURE",
               np.where(~mx["representable"], "⚪ expected (unrepresentable)", "🟢 ok")),
}, index=mx["name"])
display(view)

silent = sorted(mx.loc[mx["silent_failure"], "name"])
print("silent failures:", silent)

# CLAIM: eight representable worlds break the model while every gated check is green.
assert silent == ["confounding_controlled", "dense_controls", "multicollinearity",
                  "saturation_misspec", "seasonality_misspec", "spend_outliers",
                  "trend_break", "unobserved_confounding"], silent
clean_row = mx.set_index("name").loc["clean"]
assert clean_row["median_abs_rel_error"] < 0.25 and clean_row["coverage_rate"] >= 0.75
# PPC/refutation pass on clean under current defaults (under the old starved
# trend prior both false-alarmed on it — see section 5); either way they are
# excluded from the gate because their verdicts track config, not causality.
print("✓ verdicts reproduce from the recorded gates; clean control recovered; "
      f"clean PPC pass={bool(clean_row['ppc_pass'])}, "
      f"refutation all-pass={bool(clean_row['refutation_all_passed'])}")
"""),
    md(r"""
Eight **silent failures** — representable worlds, green gates, wrong answers:

1. **`unobserved_confounding`** — the fit above. Demand-chasing channels
   inherit the hidden demand's credit (Search is grossly over-credited in every
   recording). → *stress_03* works this failure and its pivots (proxies,
   calibration).
2. **`confounding_controlled`** — the trap *inside the fix*: the demand proxy
   **is** in the model, but it is a *noisy* measurement of the confounder, so
   the back-door is only half-closed and Search stays badly over-credited.
   Adding "the" control is not the same as closing the door. → *stress_03*.
3. **`multicollinearity`** — all channels flighted off one shared calendar;
   the data identifies the *combined* effect but the per-channel split is
   arbitrary (worst-channel splits of +60-80% *with* full coverage — the
   intervals are honest, the point estimates are noise). → *stress_03*.
4. **`saturation_misspec`** — the world saturates with an S-shaped threshold
   (Hill ≈ 3); the model's strictly concave `1−exp` curve over-credits low-spend
   weeks, inflating totals by ≈+40%. Functional form is a *causal* assumption in
   disguise. → *stress_01* (and the fix: the core model now honors per-channel
   `SaturationConfig.hill()`).
5. **`spend_outliers`** — one ~15× data-entry spike per channel inflates the
   max-normalizer; every real week collapses into the curve's near-linear
   sliver. Contributions attenuate ~40-50% with **near-zero coverage**. The fix
   is data hygiene, not modeling. → *stress_01*.
6. **`trend_break`** — a mid-series level shock plus a reactive media ramp;
   the linear trend can't bend, so the misfit lands on the ramped channels
   (both blamed, not credited — the shock dominates the recovery).
   → *stress_02* (piecewise trends — whose default changepoint prior was fixed
   as a result — and event controls).
7. **`seasonality_misspec`** — growing seasonal amplitude + holiday spikes
   under an order-2 Fourier; totals stay plausible while the *split* scrambles
   (the seasonally-flighted Social takes a ~2× over-credit at harness fidelity;
   at faster settings the leak lands elsewhere — where it goes is itself
   unstable). → *stress_02* (holiday dummies beat more Fourier).
8. **`dense_controls`** — 25 candidate controls on 156 weeks; nothing is
   misspecified, yet the sheer control load costs enough precision to break a
   channel (worst ≈−50%) with everything green. → *stress_03* (selection
   priors, confounders exempt).

The non-silent rows matter too: `adstock_misspec` is *almost* silent (−62% total
yet flagged only by the RV's fragility note), and the unrepresentable rows
(`negative_effect`, `aurora_kitchen_sink`) show what it looks like when the truth
is outside the hypothesis space entirely — *stress_04* and *stress_05* live
there. The kitchen-sink row deserves a beat: attribution is catastrophically
wrong (median error ≈90%) while the sampler barely complains (r-hat ~1.02, a
handful of divergences at most, depending on the recording). The hardest world
in the suite hardly trips the sampler.
"""),
    # ====================================================================
    # 5. Toolkit tour
    # ====================================================================
    md(r"""
## 5 — The diagnostic toolkit: what each check can and cannot catch

If the routine checks can't catch silent failures, what is each one *for*? Run
the framework's validation suite on **both** fits — the one we know is right and
the one we know is wrong — and see which checks discriminate. (Cheap config:
PPC, unobserved-confounding robustness values, and the 4-test causal refutation
suite at low fidelity — the same setup as the recorded harness.)
"""),
    code(r"""
from mmm_framework.validation import ModelValidator, ValidationConfigBuilder

def validate(mmm, label):
    b = (ValidationConfigBuilder().silent().without_residuals()
         .without_channel_diagnostics().with_unobserved_confounding()
         .with_causal_refutation(draws=120, tune=120, chains=2))
    cfg = b.build(); cfg.run_model_comparison = False; cfg.run_ppc = True
    t0 = time.perf_counter()
    with quiet():
        s = ModelValidator(mmm).validate(cfg)
    print(f"validated '{label}' in {time.perf_counter() - t0:.0f}s")
    return s

s_clean = validate(m_clean, "clean")
s_conf = validate(m_conf, "unobserved_confounding")

def report(s, f):
    refut_failed = [t.name for t in s.causal_refutation.tests if not t.passed]
    return {
        "r-hat / divergences": f"{f.diagnostics['rhat_max']:.2f} / {f.diagnostics['divergences']}",
        "PPC": "pass" if s.ppc.overall_pass else "FAIL: " + ", ".join(s.ppc.problematic_checks),
        "refutation suite": "pass" if not refut_failed else "FAIL: " + ", ".join(refut_failed),
        "confounding RV (per channel)": " ".join(
            f"{c.channel}={c.robustness_value:.2f}" for c in s.unobserved_confounding.channels),
        "RV-fragile channels": ", ".join(s.unobserved_confounding.fragile_channels) or "none",
    }

tour = pd.DataFrame({"CLEAN (attribution is right)": report(s_clean, f_clean),
                     "CONFOUNDED (attribution is wrong)": report(s_conf, f_conf)})
display(tour)

# CLAIM 1 (recalibrated for PyMC 6): PPC's verdict is FIT- and STACK-sensitive,
# never a causal all-clear. On the PyMC 6 stack PPC's autocorrelation check FIRES
# on the confounded fit — but for a FIT reason: the unmodeled AR(1) demand leaves
# autocorrelated residuals. That is PPC catching a symptom incidental to the
# confounder happening to be autocorrelated (a white-noise confounder would sail
# through it), and the verdict FLIPPED from the earlier PyMC 5 stack, which cleared
# this same fit. PPC still passes on the clean control.
assert s_clean.ppc.overall_pass, "PPC should still clear the clean control"
assert not s_conf.ppc.overall_pass, \
    "under PyMC 6 PPC's autocorrelation check fires on the confounded fit"  # recalibrated for PyMC 6 (2026-07-08; PyMC 5 cleared this fit, PyMC 6 FAILs it on Autocorrelation)
# CLAIM 2: at this fidelity the refutation suite clears both fits — it cannot
# separate the right answer from the wrong one here. (At the recorded harness
# fidelity it DOES fail placebo on the confounded world — but it also fails
# placebo on several worlds whose attribution is fine, e.g. reverse_causality;
# poor specificity is the disqualifier, not blindness.)
clean_failed = [t.name for t in s_clean.causal_refutation.tests if not t.passed]
conf_failed = [t.name for t in s_conf.causal_refutation.tests if not t.passed]
assert len(conf_failed) == 0 or len(clean_failed) > 0, (clean_failed, conf_failed)
# CLAIM 3: the RV check flags NOTHING on the truly confounded fit.
assert len(s_conf.unobserved_confounding.fragile_channels) == 0
print("✓ under PyMC 6 the refutation suite and confounding RV still wave the "
      "confounded fit through; PPC does fire — but on autocorrelation, a FIT "
      "symptom of the unmodeled AR(1) demand and a stack-sensitive verdict, not "
      "a confounding detector")
"""),
    md(r"""
The table is the doctrine in miniature — and the *one* check that separates the
two columns separates them for the wrong reason:

- **R-hat / ESS / divergences** test *only* "did the chains explore the
  posterior of the model as written". Green on both worlds. Necessary; never
  evidence of correctness.
- **Posterior predictive checks** test "does replicated data resemble observed
  data" — a *fit* property. On the PyMC 6 stack PPC's autocorrelation check
  *does* fire on the confounded column — but read *why* before you celebrate:
  the hidden demand is an **AR(1)** process, so leaving it unmodeled dumps
  autocorrelated structure into the residuals, and PPC catches *that fit
  symptom*, not the confounding. A confounder that was white noise would ride in
  through media, replicate the data beautifully, and pass PPC clean —
  attribution just as wrong, PPC just as green. And the verdict is
  stack-sensitive: this same fit *cleared* PPC on the earlier PyMC 5 stack, and
  under the framework's old, too-tight trend prior PPC *false-alarmed on the
  clean control* (residual autocorrelation from a starved trend, fixed
  2026-06-10 as a result of this stress work). Three stacks, three PPC verdicts
  on the same causal question: PPC's answer tracks the fit and the sampler
  configuration, not the causal validity.
- **The unobserved-confounding robustness value** (RV) asks "how strong would a
  hidden confounder need to be to nullify this channel?" — the right *question*,
  but its answer is computed *from the same confounded posterior*, with no
  external information about whether such a confounder exists. RVs here are
  similar across both worlds, and the truly confounded channels are not flagged.
  Sensitivity analysis bounds the damage; it cannot detect it.
- **The causal refutation suite** (placebo treatment, negative control, random
  common cause, data subset) refits the model under perturbations. Read its row
  in the live table: at this fidelity it clears **both** columns. At the
  recorded harness fidelity it does fail `placebo_treatment` on the confounded
  world — but it fails the same test on worlds whose attribution is *fine*
  (`reverse_causality`, heavy-tailed noise), and across recordings it has also
  false-alarmed on the clean control. A detector with that specificity is a
  *red-flag generator* for development, never a green light — and never an
  all-clear.

What's missing from every row: **information the model doesn't have** — a held
back-door (a real confounder measured well), or randomized evidence (a lift
test). That is why the recovery arc later in the series runs through *design*
(DAG, controls, data hygiene) and *calibration* (experiments), not through more
diagnostics.
"""),
    md(r"""
### Prior→posterior parameter learning — the honest non-catcher

One more tool, demonstrated live because it is the most misread:
`compute_parameter_learning` compares each parameter's prior to its posterior —
**contraction** (1 − posterior var / prior var), **overlap**, and **shift** —
and issues a verdict per parameter. Doctrine, before the table:

- **Contraction measures informativeness, not importance.** A channel can carry
  the largest contribution while its β has the *least* contraction — the data
  determines the channel's *contribution* tightly while β individually trades
  off against saturation (equifinality; see `math_01`/`math_02`).
- A **prior-dominated** verdict means the posterior re-states the prior — the
  data didn't speak. That is a *reporting* obligation (the number is your
  assumption), not a defect.
- **Negative contraction** (posterior *wider* than prior) signals prior–data
  conflict worth investigating.
"""),
    code(r"""
learn = m_clean.compute_parameter_learning(prior_samples=1000, random_seed=0)
cols = ["parameter", "contraction", "overlap", "shift_z", "verdict"]
display(learn[cols].round(3))

fig, ax = plt.subplots(figsize=(8, 5.5))
ll = learn.sort_values("contraction")
vc = {"prior-dominated": BERRY, "weak": AMBER, "moderate": SKY, "strong": LEAF}
ax.barh(ll["parameter"], ll["contraction"], color=[vc.get(v, MUTED) for v in ll["verdict"]])
ax.axvline(0, color=INK, lw=0.8)
ax.set_xlabel("prior→posterior contraction (1 − var_post/var_prior)")
ax.set_title("Parameter learning on the CLEAN fit — colors = verdict", fontsize=10)
handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in vc.values()]
ax.legend(handles, vc.keys(), fontsize=8, loc="lower right")
plt.tight_layout(); plt.show()

betas = learn[learn["parameter"].str.fullmatch(r"beta_(TV|Search|Social|Display)")]
betas = betas.set_index(betas["parameter"].str.replace("beta_", "", regex=False))
peek = betas[["contraction", "verdict"]].join(g_clean[["true", "rel_err"]])
peek["rel_err"] = peek["rel_err"].map("{:+.0%}".format)
display(peek.sort_values("true", ascending=False))

# CLAIM: learning verdicts vary across parameters of a WELL-RECOVERED fit —
# contraction is about what the data pins down, not about what matters.
assert len(betas) == 4 and betas["verdict"].nunique() >= 2
# The channel with the LARGEST true contribution is not the most-contracted beta:
top_truth = g_clean["true"].idxmax()
assert betas.loc[top_truth, "contraction"] < betas["contraction"].max()
print(f"✓ '{top_truth}' carries the largest true contribution but not the most-learned beta")
print("  → contraction = informativeness, NOT importance")
"""),
    md(r"""
Read the join table: this is the **clean** fit — recovery is good and every
truth is inside its interval — yet the per-channel βs span verdicts, and the
β-contraction ranking does **not** reproduce the contribution ranking. A
channel's *contribution* can be tightly identified while its β individually
shares the credit with the saturation and adstock parameters (the equifinality
trade-off from `math_01`/`math_02`). If you used "which β learned most" as a
measure of channel importance you would mis-rank these channels — on a fit
whose attribution is *correct*. Use contraction for what it is: an audit of
**which reported numbers are data vs which are priors restated**.

### The toolkit, summarized

| check | actually tests | catches | cannot catch |
|---|---|---|---|
| R-hat / ESS / divergences | sampler explored the posterior | broken sampling, non-identification *sometimes* (divergences) | any causal error — converges happily on biased posteriors |
| In-sample fit / R² | predictive adequacy | gross misspecification | confounding (a confounder *improves* fit) |
| PPC | replicated ≈ observed data | wrong noise model, missing dynamics | mis-attribution at same fit quality; false-positives on good models |
| Confounding RV | how big a hidden confounder must be to kill the effect | nothing by itself — it's a bound, computed from the possibly-biased posterior | whether such a confounder exists |
| Refutation suite | effect behaves sanely under placebo/perturbation | implementation bugs, fragile effects | calm, stable bias; false-positives at low fidelity |
| Parameter learning | posterior ≠ prior, per parameter | prior-dominated reporting, prior–data conflict | bias in well-identified parameters (confounded β *learns strongly* — toward the wrong value) |

Every row tests *computation, fit, or sensitivity*. None tests **attribution**.
Attribution is testable only against information from outside the likelihood:
structure you assert (a DAG), or randomization you buy (experiments).
"""),
    # ====================================================================
    # 6. Roadmap + what to remember
    # ====================================================================
    md(r"""
## 6 — The series roadmap

Each notebook fits worlds where specific assumptions fail, shows which symptoms
(if any) surface, and works the **pivot** — the modeling or design change that
actually fixes attribution, graded against the same counterfactual truth.

| notebook | worlds | failure modes | pivots |
|---|---|---|---|
| **stress_00 — the rosy picture** *(this one)* | clean, unobserved_confounding, recorded matrix | doctrine: green ≠ right | the toolkit's honest scope |
| **stress_01 — carryover & shape** | adstock_misspec, saturation_misspec, spend_outliers | wrong kernel/curve family, normalization poisoning | richer kernels, curve checks, data hygiene |
| **stress_02 — time structure** | trend_break, seasonality_misspec, time_varying_beta | breaks & seasonality leaking into media | piecewise trends, holiday controls, TV-beta |
| **stress_03 — confounding & selection** | unobserved_confounding, confounding_controlled, multicollinearity, dense_controls | open/half-closed back-doors, collinear flighting, selection traps | DAG-driven roles, confounder priors, calibration |
| **stress_04 — extension traps** | mediation/multi-KPI worlds | when NestedMMM / MultivariateMMM / Combined help vs add unidentifiable knobs | structure only where identified |
| **stress_05 — the gauntlet** | aurora_kitchen_sink | everything at once | full workflow + symptom→cause→pivot table |

## What to remember

1. **Green diagnostics validate the computation, not the causal claim.** Both
   fits in this notebook converged identically; one over-credited its
   demand-chasing channels by tens of percent, the truth outside the interval.
2. **Insist on a comparable estimand.** Truth here is the counterfactual
   zero-out on the structural mean — the same question the model answers — so
   error is measurable, not rhetorical.
3. **Run a positive control.** The clean world is what earns every later
   accusation: same model, same settings, correct answer.
4. **Know each check's scope.** Sampler checks test sampling; PPC tests fit; RV
   bounds sensitivity; learning audits informativeness. None tests attribution.
5. **Silent failures are the norm, not the exception** — 5 of the 10
   representable broken-assumption worlds pass every gated check while badly
   mis-attributing.
6. **The fixes are upstream of the fit**: causal structure, data hygiene, and
   experiments. The rest of the series is about exactly that.

**Next:** `stress_01_carryover_and_shape.ipynb` — the assumptions you can
actually *see* in the data (carryover windows, response curvature, spend
spikes), why the model misses them silently, and the pivots that recover.
"""),
]


def main():
    nb = new_notebook(cells=CELLS)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "title": "Stress 00 — The Rosy Picture",
    })
    path = "stress_00_the_rosy_picture.ipynb"
    with open(path, "w") as fh:
        nbformat.write(nb, fh)
    print(f"wrote {path} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
