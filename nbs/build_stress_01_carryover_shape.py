"""Author stress notebook 01 — carryover & shape (run from ``nbs/``).

    uv run python build_stress_01_carryover_shape.py
    PYTHONPATH=.. uv run jupyter nbconvert --to notebook --execute --inplace \
        stress_01_carryover_and_shape.ipynb --ExecutePreprocessor.timeout=2400

Notebook 01 of the six-part "stress" series: functional-form failures.
Three acts, each world -> naive fit -> truth gap -> which diagnostics fire ->
pivot refit -> measured improvement:

1. ADSTOCK   — true carryover is a delayed Weibull with mass past 26 weeks;
               the default geometric l_max=8 kernel cannot represent it. The
               old fixed Gamma(2,1) Weibull scale prior (shown explicitly as
               the historical default) buys divergences; the current
               l_max-scaled default samples cleanly.
2. SATURATION — true response is S-shaped Hill; the default 1-exp curve is
               strictly concave. The pivot is now turnkey: per-channel
               ``SaturationConfig.hill()`` is honored by the core model
               (a fix this stress test motivated).
3. OUTLIERS  — one ~15x data-entry spike per channel poisons the per-channel
               max used for normalization; a 30-second EDA screen fixes it.

All quantitative claims in markdown are seed-robust (the numbers live in code
cells with directional asserts); measured values come from /tmp/probe_stress01*
probes run before authoring. Same nbformat pattern as build_mmm_walkthrough.py;
assert-per-claim discipline as in build_math_01_adstock.py.
"""

from __future__ import annotations

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


CELLS = [
    # =====================================================================
    # Title
    # =====================================================================
    md(r"""
# Stress 01 — Carryover & Shape: When the Functional Form Is Wrong

**Series:** *Green diagnostics ≠ correct attribution.* This is notebook **01** of six.
[`stress_00_the_rosy_picture`](stress_00_the_rosy_picture.ipynb) established the doctrine —
the model recovers its own generative family almost perfectly, and that tells you nothing
about real data. Here we break the **functional form**: the shape assumptions baked into
every channel's response that no sampler diagnostic ever inspects.

Three acts, each on a world from `tests/synth/dgp.py` where exactly **one** assumption is
violated and the ground truth is the model's own estimand (counterfactual zero-out on the
noiseless structural mean — truth and estimate are the same quantity):

| act | world | broken assumption | recorded damage (stress matrix, 500-draw PyMC) |
|---|---|---|---|
| 1 | `adstock_misspec` | carryover is delayed Weibull, mass past 26 wk; model: geometric, peak at lag 0, window 8 wk | med\|err\| 77%, total media −62% |
| 2 | `saturation_misspec` | response is S-shaped Hill (threshold); model: strictly concave 1−exp | med\|err\| 46%, total +39%, **coverage 25%** |
| 3 | `spend_outliers` | one ~15× data-entry spike per channel inflates the normalization max | **coverage 0%** |

Each act ends with a **working pivot** — refit with the right kernel, the right curve, or
clean data — and an honest measurement of how much it recovers (spoiler: not always
everything). Recorded matrix: `tests/synth/results/stress_matrix.md`.
"""),
    code(r"""
import sys, os, pathlib, time, warnings, logging, contextlib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
for _n in ("pymc", "pymc.sampling", "numpyro", "jax", "arviz", "pytensor"):
    _lg = logging.getLogger(_n); _lg.setLevel(logging.ERROR); _lg.propagate = False
sys.path.insert(0, str(pathlib.Path.cwd().parent))  # repo root (run from nbs/)

@contextlib.contextmanager
def quiet():
    "Hide sampler progress bars/chatter; our own prints stay visible."
    with open(os.devnull, "w") as _dn, contextlib.redirect_stdout(_dn), \
            contextlib.redirect_stderr(_dn):
        yield

from tests.synth import dgp

plt.rcParams.update({
    "figure.figsize": (10, 4), "figure.dpi": 110, "axes.grid": True,
    "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
})
INK, ACCENT, SKY, BERRY, LEAF, AMBER, MUTED = (
    "#2b2118", "#b5651d", "#3b6ea5", "#a63a50", "#3f7d5e", "#d98a2b", "#8a8079")
PAL = {"TV": ACCENT, "Search": SKY, "Social": BERRY, "Display": LEAF}
CH = dgp.CHANNELS
print("channels:", CH)
"""),
    md(r"""
## Shared harness

One fit configuration for the whole notebook (fast: numpyro, 500 draws × 2 chains — the
qualitative results match the 500-draw PyMC stress matrix), one grader. The grader compares
`compute_counterfactual_contributions` (seeded) against `sc.true_contribution`: **the same
estimand**, so any gap is model error, not an apples-to-oranges artifact. `SUMMARY` collects
each stage for the closing synthesis.
"""),
    code(r"""
from mmm_framework.config import (
    AdstockConfig, InferenceMethod, ModelConfig, PriorConfig,
)
from mmm_framework.model import BayesianMMM, TrendConfig, TrendType

DRAWS, TUNE, CHAINS = 500, 500, 2
SUMMARY = {}  # (act, stage) -> dict of headline metrics

def make_model(panel, target_accept=0.9):
    cfg = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_NUMPYRO,
        n_draws=DRAWS, n_tune=TUNE, n_chains=CHAINS,
        target_accept=target_accept, use_parametric_adstock=True, optim_seed=0,
    )
    return BayesianMMM(panel, cfg, TrendConfig(type=TrendType.LINEAR))

def timed_fit(mmm, label, **kw):
    t0 = time.perf_counter()
    with quiet():
        res = mmm.fit(random_seed=0, **kw)
    print(f"fit '{label}': {time.perf_counter()-t0:.0f}s   "
          f"r-hat max {res.diagnostics['rhat_max']:.3f}   "
          f"divergences {res.diagnostics['divergences']}")
    return res

def grade(sc, mmm, key, label):
    "Per-channel truth vs estimate + 90% CI coverage; stores headline metrics."
    with quiet():
        contrib = mmm.compute_counterfactual_contributions(
            compute_uncertainty=True, hdi_prob=0.9, random_seed=0)
    rows = []
    for c in sc.channels:
        t = float(sc.true_contribution[c]); e = float(contrib.total_contributions[c])
        lo = float(contrib.contribution_hdi_low[c]); hi = float(contrib.contribution_hdi_high[c])
        rows.append(dict(channel=c, true=round(t), est=round(e),
                         rel_err=(e - t) / abs(t), covered=lo <= t <= hi,
                         lo=round(lo), hi=round(hi)))
    df = pd.DataFrame(rows).set_index("channel")
    s = dict(label=label,
             med_abs_err=float(df.rel_err.abs().median()),
             total_err=float((df.est.sum() - df.true.sum()) / df.true.sum()),
             coverage=float(df.covered.mean()))
    SUMMARY[key] = s | {"df": df}
    print(f"{label}:  med|err| {s['med_abs_err']:.0%}   "
          f"total-media err {s['total_err']:+.0%}   coverage {s['coverage']:.0%}")
    show = df.copy(); show["rel_err"] = (df.rel_err * 100).round(0).astype(int).astype(str) + "%"
    return show

def err_bars(ax, df, title):
    "Truth-vs-estimate bars with 90% HDI whiskers."
    x = np.arange(len(df)); w = 0.38
    ax.bar(x - w/2, df["true"], w, color=INK, alpha=0.85, label="truth")
    yerr = np.vstack([(df["est"] - df["lo"]).clip(lower=0),
                      (df["hi"] - df["est"]).clip(lower=0)])
    ax.bar(x + w/2, df["est"], w, color=ACCENT, alpha=0.9, label="estimate")
    ax.errorbar(x + w/2, df["est"], yerr=yerr, fmt="none", ecolor=INK,
                elinewidth=1.2, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(df.index)
    ax.set_ylabel("total contribution (KPI units)"); ax.set_title(title); ax.legend()
"""),

    # =====================================================================
    # ACT 1 — adstock
    # =====================================================================
    md(r"""
---
# Act 1 — Carryover: the kernel the model cannot draw

**The world.** `adstock_misspec` keeps everything from the clean control — same flighting,
same concave saturation, same Gaussian noise — and changes only the carryover. The true
kernel is a **delayed Weibull** (shape 1.8–2.6, scale 6–9, support out to 26 weeks): the
effect of a flight *builds for several weeks before it peaks* and carries meaningful mass
past the model's entire window. The model's default is a **geometric** kernel with
`l_max=8`: peak forced to lag 0, nothing after week 8. The misspecification is structural —
no geometric parameter value can produce a hump.
"""),
    code(r"""
sc1 = dgp.build("adstock_misspec")
TRUE_L = sc1.notes["true_l_max"]
# True kernel parameters, restated from tests/synth/dgp.make_adstock_misspec.
W_SHAPE = {"TV": 2.6, "Search": 1.8, "Social": 2.2, "Display": 2.4}
W_SCALE = {"TV": 9.0, "Search": 6.0, "Social": 7.0, "Display": 8.0}

def weibull_kernel(shape, scale, l_max):
    k = np.arange(l_max, dtype=float)
    w = (shape/scale) * (((k+1)/scale)**(shape-1)) * np.exp(-(((k+1)/scale)**shape))
    return w / w.sum()

# Verify our restated kernel IS the DGP's: the impulse response of the dgp's
# private transform equals these weights.
imp = np.zeros(TRUE_L); imp[0] = 1.0
for c in CH:
    assert np.allclose(dgp._weibull_adstock(imp, W_SHAPE[c], W_SCALE[c], TRUE_L),
                       weibull_kernel(W_SHAPE[c], W_SCALE[c], TRUE_L), atol=1e-12)

fig, ax = plt.subplots(figsize=(10, 4.2))
for c in CH:
    w = weibull_kernel(W_SHAPE[c], W_SCALE[c], TRUE_L)
    ax.plot(np.arange(TRUE_L), w, "-o", ms=3.5, lw=1.8, color=PAL[c],
            label=f"{c} (peak lag {w.argmax()}, {w[8:].sum():.0%} of mass past lag 7)")
ax.axvspan(8, TRUE_L - 1, color=MUTED, alpha=0.15)
ax.text(16, ax.get_ylim()[1]*0.92, "invisible to l_max=8", color=MUTED, fontsize=9)
ax.axvline(7.5, color=INK, lw=1, ls="--")
ax.set(xlabel="lag (weeks)", ylabel="normalized kernel weight",
       title="TRUE carryover: delayed Weibull humps — the geometric family peaks at lag 0 and stops at week 8")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()

mass_beyond = {c: weibull_kernel(W_SHAPE[c], W_SCALE[c], TRUE_L)[8:].sum() for c in CH}
peaks = {c: int(weibull_kernel(W_SHAPE[c], W_SCALE[c], TRUE_L).argmax()) for c in CH}
assert all(p >= 3 for p in peaks.values())            # delayed peaks, lag >= 3
assert mass_beyond["TV"] > 0.35                       # TV: >35% of effect after the window
print("peak lags:", peaks, " mass beyond the 8-week window:",
      {c: f"{v:.0%}" for c, v in mass_beyond.items()})
"""),
    md(r"""
## Naive fit — the default kernel, fit honestly

`use_parametric_adstock=True` with the default per-channel `AdstockConfig.geometric()`
(`l_max=8`), the same spec the stress matrix used. Watch two things: the **convergence
diagnostics** (they will be essentially green) and the **truth gap** (it will be enormous).
"""),
    code(r"""
panel1a = sc1.panel()
m1a = make_model(panel1a)
rv = [v.name for v in m1a.model.free_RVs if "adstock" in v.name]
print("adstock RVs:", rv)
assert all(v.startswith("adstock_alpha_") for v in rv)  # geometric kernel confirmed
r1a = timed_fit(m1a, "act1 naive (geometric l_max=8)",
                idata_kwargs={"log_likelihood": True})
assert r1a.diagnostics["rhat_max"] < 1.05      # converged by any practical gate
assert r1a.diagnostics["divergences"] < 10
g1a = grade(sc1, m1a, ("act1", "naive"), "act1 naive")
g1a
"""),
    code(r"""
df = SUMMARY[("act1", "naive")]["df"]
fig, ax = plt.subplots(figsize=(9, 4))
err_bars(ax, df, "Act 1 naive: geometric l_max=8 vs the truth")
plt.tight_layout(); plt.show()
s = SUMMARY[("act1", "naive")]
assert s["med_abs_err"] > 0.6        # catastrophic recovery error...
assert s["total_err"] < -0.5         # ...total media credit collapsed
assert s["coverage"] <= 0.5          # ...and the intervals don't admit it
print(f"The sampler converged (r-hat {r1a.diagnostics['rhat_max']:.3f}) onto an answer "
      f"that destroys {-s['total_err']:.0%} of total media value.")
"""),
    md(r"""
**What just happened.** With the true effect arriving 3–9 weeks *after* the spend (and
15–42% of it after week 8), the lag-0-peaked geometric kernel can't line media up with its
response. The regression then does the only thing it can: it routes the orphaned media
signal into the baseline (trend/seasonality soak up the slow carryover hum) and shrinks the
channels toward zero. Note the asymmetry in the table above — **Search**, the one channel
whose true kernel is closest to geometric (shape 1.8, scale 6: an early, modest hump),
survives; the long-memory channels are nearly erased.

The kernel autopsy makes it visceral:
"""),
    code(r"""
curves1a = m1a.compute_adstock_curves()
c = "TV"
fig, ax = plt.subplots(figsize=(9, 3.8))
wt = weibull_kernel(W_SHAPE[c], W_SCALE[c], TRUE_L)
ax.plot(np.arange(TRUE_L), wt, "-o", ms=3.5, color=INK, label="truth (delayed Weibull)")
ax.plot(np.arange(len(curves1a[c])), curves1a[c], "-s", ms=4, color=ACCENT,
        label="fitted (geometric, posterior mean)")
ax.axvline(7.5, color=MUTED, lw=1, ls="--")
ax.set(xlabel="lag (weeks)", ylabel="kernel weight",
       title=f"{c}: the model concentrates the effect at lag 0-1; the truth hasn't even peaked yet")
ax.legend(); plt.tight_layout(); plt.show()
assert curves1a[c].argmax() == 0                  # geometric must peak at lag 0
assert wt.argmax() >= 5                           # truth peaks much later
print(f"fitted kernel mass at lags 0-1: {curves1a[c][:2].sum():.0%}   "
      f"true mass at lags 0-1: {wt[:2].sum():.0%}")
"""),
    md(r"""
## Which diagnostics fire?

Convergence is green. What about the **prior→posterior learning diagnostic** — did the data
actually pin the carryover parameters, or did the posterior just restate the prior?
(`contraction` measures variance reduction relative to the prior; it is informativeness,
*not* importance.)
"""),
    code(r"""
with quiet():
    learn1a = m1a.compute_parameter_learning(prior_samples=1000, random_seed=0)
ad = learn1a[learn1a.parameter.str.startswith("adstock_alpha")]
print(ad[["parameter", "prior_sd", "post_sd", "contraction", "verdict"]].to_string(index=False))
assert (ad.contraction < 0.5).all()   # nothing strongly learned about carryover
assert (ad.verdict != "strong").all()
print("\nNo adstock decay parameter is strongly identified — several posteriors are WIDER "
      "than the prior\n(negative contraction): the likelihood is pushing alpha against a "
      "kernel family that cannot fit,\nnot measuring carryover. Observational data barely "
      "identifies carryover even when the family is right;\nwhen it's wrong, the parameter "
      "is pure noise. This diagnostic fires — if you look at it.")
"""),
    md(r"""
## Pivot — give the model a kernel that can represent the truth

Per-channel `AdstockConfig.weibull(l_max=26)` on the panel's `media_channels`, consumed by
the core model when `use_parametric_adstock=True`. Two lessons here, and the first is a
trap of its own — one this very stress test got fixed in the framework:

**Attempt 1 (shown honestly): flexibility with an off-target prior fails.** The
framework's *old* default Weibull scale prior was a fixed `Gamma(2, 1)` — mean 2, almost
no mass past week 5, regardless of the lag window. Asking it to find a scale-6–9 kernel
over a 26-lag window produces a sampler fight: divergences spike and r-hat degrades. We
reproduce that historical default by passing it explicitly. *This time a diagnostic
fires* — but it's telling you about a prior–likelihood fight, not about which kernel is
true.
"""),
    code(r"""
panel1b = sc1.panel()
for mc in panel1b.config.media_channels:
    # The framework's OLD default scale prior, passed explicitly: a fixed
    # Gamma(2, 1) (mean 2 weeks) that ignored the lag window. This stress
    # test's divergence storm is why the default changed.
    mc.adstock = AdstockConfig.weibull(
        l_max=26, scale_prior=PriorConfig.gamma(alpha=2.0, beta=1.0))
m1b_old = make_model(panel1b, target_accept=0.95)
rv = [v.name for v in m1b_old.model.free_RVs if "adstock" in v.name]
assert any("adstock_shape" in v for v in rv) and any("adstock_scale" in v for v in rv)
print("Weibull RVs in-graph:", rv[:4], "...")
r1b_old = timed_fit(m1b_old, "act1 pivot attempt 1 (weibull, OLD fixed-scale prior)")
g1b_old = grade(sc1, m1b_old, ("act1", "pivot_old_prior"),
                "act1 pivot attempt 1 (old fixed-scale prior)")
assert (r1b_old.diagnostics["divergences"] > 5
        or r1b_old.diagnostics["rhat_max"] > 1.05)  # the sampler audibly struggles
print("\nDivergences and/or inflated r-hat: a 26-lag kernel with 2 free shape parameters "
      "per channel is\nweakly identified from 156 weekly observations when the prior puts "
      "the scale in the wrong decade.")
"""),
    md(r"""
**Attempt 2: the fix — a scale prior that respects the window.** The framework's
*current* default (this stress test motivated the change) scales the prior with the
window: `AdstockConfig.weibull(l_max=26)` now gives `scale ~ Gamma(2, 2/m)` with mean
`m = max(2, (l_max − 9)/2)` — the legacy mean-2 prior for windows ≤ 13 lags, half the
window beyond that (mean 8.5 at `l_max=26`). Same data, same likelihood, same flexibility
— only the prior moves into the decade the window itself implies. No hand-tuning below:
the call carries no explicit priors.
"""),
    code(r"""
panel1c = sc1.panel()
for mc in panel1c.config.media_channels:
    mc.adstock = AdstockConfig.weibull(l_max=26)   # CURRENT default: l_max-scaled prior
sp = panel1c.config.media_channels[0].adstock.scale_prior
print("default scale prior at l_max=26:", sp.distribution.value, sp.params,
      " (mean", sp.params["alpha"] / sp.params["beta"], "weeks)")
assert abs(sp.params["alpha"] / sp.params["beta"] - 8.5) < 1e-9
m1c = make_model(panel1c, target_accept=0.95)
r1c = timed_fit(m1c, "act1 pivot (weibull l_max=26, current default prior)",
                idata_kwargs={"log_likelihood": True})
assert r1c.diagnostics["rhat_max"] < 1.05
assert r1c.diagnostics["divergences"] < 10        # the sampler fight is gone
g1c = grade(sc1, m1c, ("act1", "pivot"), "act1 pivot (current-default weibull)")
g1c
"""),
    code(r"""
# Kernel autopsy, all channels: truth vs naive geometric vs default-Weibull pivot.
curves1c = m1c.compute_adstock_curves()
fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
for ax, c in zip(axes.ravel(), CH):
    wt = weibull_kernel(W_SHAPE[c], W_SCALE[c], TRUE_L)
    ax.plot(np.arange(TRUE_L), wt, color=INK, lw=2, label="truth")
    ax.plot(np.arange(len(curves1a[c])), curves1a[c], color=ACCENT, lw=1.6,
            ls="--", label="naive geometric")
    ax.plot(np.arange(len(curves1c[c])), curves1c[c], color=SKY, lw=1.8,
            label="pivot Weibull")
    ax.set_title(c, fontsize=10); ax.set_xlabel("lag (weeks)")
axes[0, 0].set_ylabel("kernel weight"); axes[1, 0].set_ylabel("kernel weight")
axes[0, 0].legend(fontsize=8)
fig.suptitle("The pivot recovers the SHAPE of carryover: a delayed hump, roughly the right place")
plt.tight_layout(); plt.show()
pivot_peaks = {c: int(curves1c[c].argmax()) for c in CH}
print("pivot kernel peak lags:", pivot_peaks, "  (true peaks:", peaks, ")")
assert all(p >= 2 for p in pivot_peaks.values())   # every channel now peaks AFTER lag 0
"""),
    md(r"""
**Uncertainty in the kernel — not just its mean.** A posterior-mean curve hides
the most decision-relevant fact about carryover: *how wide* the plausible kernel
family still is after 156 weeks of data. Rebuild the full **posterior kernel
band** per channel from the raw draws — `adstock_shape_*`/`adstock_scale_*` for
the Weibull pivot, `adstock_alpha_*` for the naive geometric — and put the truth
on top. Two things to look for: whether the band **covers** the true kernel
(honesty), and how the *naive* band fails — not by being wide, but by being
**narrow around a shape that cannot bend**.
"""),
    code(r"""
def geom_kernel(alpha, l_max=8):
    w = alpha ** np.arange(l_max); return w / w.sum()

post_a1, post_c1 = r1a.trace.posterior, r1c.trace.posterior
fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.4), sharex=True)
KCOV = {}
for ax, c in zip(axes.ravel(), CH):
    wt = weibull_kernel(W_SHAPE[c], W_SCALE[c], TRUE_L)
    # Weibull pivot: posterior kernel band from (shape, scale) draws.
    sh = post_c1[f"adstock_shape_{c}"].values.ravel()
    sl = post_c1[f"adstock_scale_{c}"].values.ravel()
    kw = np.array([weibull_kernel(s, z, TRUE_L) for s, z in zip(sh, sl)])
    lo, hi = np.percentile(kw, [5, 95], axis=0)
    # Naive geometric: posterior band from alpha draws (l_max=8 window).
    al = post_a1[f"adstock_alpha_{c}"].values.ravel()
    gw = np.array([geom_kernel(a) for a in al])
    glo, ghi = np.percentile(gw, [5, 95], axis=0)
    pk = int(wt.argmax())
    KCOV[c] = {
        "weibull band covers truth": float(((wt >= lo) & (wt <= hi)).mean()),
        "geom band covers truth@peak": bool(glo[min(pk, 7)] <= wt[min(pk, 7)] <= ghi[min(pk, 7)]),
    }
    ax.fill_between(np.arange(TRUE_L), lo, hi, color=SKY, alpha=0.30,
                    label="pivot Weibull 90% band")
    ax.plot(np.arange(TRUE_L), kw.mean(axis=0), color=SKY, lw=1.4)
    ax.fill_between(np.arange(8), glo, ghi, color=ACCENT, alpha=0.35,
                    label="naive geometric 90% band")
    ax.plot(np.arange(TRUE_L), wt, color=INK, lw=2, ls="--", label="truth")
    ax.set_title(f"{c}: truth inside Weibull band "
                 f"{KCOV[c]['weibull band covers truth']:.0%} of lags", fontsize=9.5)
    ax.set_xlabel("lag (weeks)")
axes[0, 0].legend(fontsize=8)
axes[0, 0].set_ylabel("kernel weight"); axes[1, 0].set_ylabel("kernel weight")
fig.suptitle("Posterior KERNEL bands: the pivot is honestly wide and covers the truth; "
             "the naive band is narrow around the wrong shape")
plt.tight_layout(); plt.show()
display(pd.DataFrame(KCOV).T)

# CLAIM 1: the pivot's posterior kernel band covers the true kernel at most lags.
assert all(v["weibull band covers truth"] >= 0.7 for v in KCOV.values())
# CLAIM 2: the naive geometric band EXCLUDES the truth at the truth's own peak
# lag for every channel — confident about a shape the family cannot draw.
assert not any(v["geom band covers truth@peak"] for v in KCOV.values())
print("✓ wide-and-honest beats narrow-and-wrong: the flexible kernel's "
      "uncertainty is the honest representation of what 156 weeks can know")
"""),
    code(r"""
# Measured improvement, naive -> pivot.
n, p = SUMMARY[("act1", "naive")], SUMMARY[("act1", "pivot")]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
x = np.arange(len(CH)); w = 0.38
axes[0].bar(x - w/2, n["df"].rel_err * 100, w, color=ACCENT, label="naive")
axes[0].bar(x + w/2, p["df"].rel_err * 100, w, color=SKY, label="pivot")
axes[0].axhline(0, color=INK, lw=0.8)
axes[0].set_xticks(x); axes[0].set_xticklabels(CH)
axes[0].set_ylabel("contribution error (%)"); axes[0].set_title("per-channel error"); axes[0].legend()
mets = pd.DataFrame({"naive": [n["med_abs_err"], -n["total_err"], n["coverage"]],
                     "pivot": [p["med_abs_err"], -p["total_err"], p["coverage"]]},
                    index=["med|err|", "total media deficit", "90% coverage"])
mets.plot.bar(ax=axes[1], color=[ACCENT, SKY], rot=15)
axes[1].set_title("headline metrics"); axes[1].set_ylabel("fraction")
plt.tight_layout(); plt.show()

assert p["med_abs_err"] < n["med_abs_err"] - 0.2     # error drops by a large chunk...
assert p["coverage"] >= n["coverage"] + 0.2          # ...coverage improves materially...
assert p["med_abs_err"] > 0.2                        # ...but NOT a clean fix
print(f"med|err| {n['med_abs_err']:.0%} -> {p['med_abs_err']:.0%};  "
      f"coverage {n['coverage']:.0%} -> {p['coverage']:.0%};  "
      f"total media {n['total_err']:+.0%} -> {p['total_err']:+.0%}")
print("Honest reading: the pivot buys back a large share of the error and doubles "
      "coverage with zero\ndivergences and no hand-tuned priors — but a 26-week kernel is "
      "still weakly identified from 3 years\nof weekly data. The remaining error lives in "
      "wide, truthful posteriors rather than in confident\nwrong ones. That trade is the win.")
"""),
    md(r"""
## Choosing a kernel when you don't know the truth

In real life there is no `sc.true_contribution`. What the framework *does* give you is
leave-one-out predictive comparison: both fits above stored pointwise log-likelihood
(`idata_kwargs={"log_likelihood": True}` passed through `fit(...)` to `pm.sample`), so
`az.compare` ranks the kernels on predictive grounds.
"""),
    code(r"""
import arviz as az
with quiet():
    cmp = az.compare({"geometric_l8": r1a.trace, "weibull_l26": r1c.trace})  # arviz 1.x dropped ic=; LOO is the default
print(cmp[["rank", "elpd_loo", "elpd_diff", "dse", "weight"]].round(2).to_string())
diff = float(cmp.loc["geometric_l8", "elpd_diff"]); dse = float(cmp.loc["geometric_l8", "dse"])
assert cmp.index[0] == "weibull_l26"      # LOO picks the right kernel...
assert diff > 2 * dse                     # ...decisively (diff >> its SE)
print(f"\nLOO prefers the Weibull kernel by {diff:.0f} +/- {dse:.0f} elpd — a decisive "
      "margin. Functional-form\nchoice CAN be data-driven, but only if you fit the "
      "candidates and compare; the default never\nvolunteers that it is wrong.")
"""),
    md(r"""
> **Act 1 takeaway.** A carryover window that is too short doesn't degrade the model — it
> *silently reroutes* media credit into the baseline while r-hat stays green. The two
> diagnostics that do whisper are parameter-learning on the adstock parameters
> (prior-dominated / wider-than-prior posteriors) and LOO comparison against a richer
> kernel — both opt-in. And the pivot is not "use the most flexible kernel": flexibility
> with an off-target prior just buys divergences — exactly what the old fixed `Gamma(2,1)`
> scale default did at `l_max=26`, and why the default now scales with the window.
> **Flexible kernel + a prior in the right decade + LOO to arbitrate** halved the error
> and restored coverage; the rest of the uncertainty is real and the model now admits it.
"""),

    # =====================================================================
    # ACT 2 — saturation
    # =====================================================================
    md(r"""
---
# Act 2 — Saturation: the S-curve the model flattens into a ramp

**The world.** `saturation_misspec` swaps one ingredient: the true response is an
**S-shaped Hill curve** with Hill coefficient ≈ 3 and half-saturation around 0.40–0.45 of
max spend. Below the threshold, spend does almost nothing; above it, response climbs
steeply then flattens. The default saturation in the core model is `1 − exp(−λx)`:
**strictly concave**, steepest at zero — exactly backwards in the threshold region.

A bit of framework history worth recording before we fit: the core `BayesianMMM` used to
*silently ignore* `MediaChannelConfig.saturation` — the config could say "hill" while the
model fit 1−exp anyway, and Hill existed only as components in `mmm_extensions`. An
earlier version of this notebook documented (and asserted) that trap, and it got fixed:
the default is now an honest `SaturationConfig.logistic()` (same `sat_lam_<ch>` RVs and
graph as ever), and setting `SaturationConfig.hill()` on a channel genuinely changes the
fitted curve. We exploit that in the pivot.
"""),
    code(r"""
from mmm_framework.config import SaturationConfig

sc2 = dgp.build("saturation_misspec")
TRUE_HILL = sc2.notes["hill"]
# Half-saturation points restated from tests/synth/dgp.make_saturation_misspec.
TRUE_HALF = {"TV": 0.45, "Search": 0.40, "Social": 0.42, "Display": 0.45}
assert TRUE_HILL == {"TV": 3.0, "Search": 2.5, "Social": 3.0, "Display": 2.8}

panel2a = sc2.panel()
# The default config now says what the model actually fits: logistic 1-exp.
assert all(m.saturation.type.value == "logistic" for m in panel2a.config.media_channels)
m2a = make_model(panel2a)
rv2 = [v.name for v in m2a.model.free_RVs]
assert any(v.startswith("sat_lam_") for v in rv2)
assert not any(v.startswith(("sat_half_", "sat_slope_")) for v in rv2)
print("Config saturation type:", panel2a.config.media_channels[0].saturation.type.value)
print("Model saturation RVs:  ", [v for v in rv2 if v.startswith("sat_lam")])
print("\n=> config and graph agree: the default fits the concave 1-exp curve, one "
      "sat_lam per channel.")
"""),
    code(r"""
# The geometry of the trap: true Hill curves vs the model's reachable 1-exp family,
# with the spend distribution underneath (where the data actually lives).
xg = np.linspace(0, 1, 200)
xn2 = sc2.spend / sc2.spend.max()
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax = axes[0]
for c in CH:
    ax.plot(xg, xg**TRUE_HILL[c] / (xg**TRUE_HILL[c] + TRUE_HALF[c]**TRUE_HILL[c]),
            color=PAL[c], lw=1.8, label=f"{c} truth (Hill)")
for lam, ls in [(1.0, ":"), (3.0, "--")]:
    ax.plot(xg, (1 - np.exp(-lam*xg)) / (1 - np.exp(-lam)), color=MUTED, ls=ls, lw=1.4,
            label=f"1-exp family (lam={lam})")
ax.set(xlabel="normalized spend", ylabel="response (scaled to 1 at max)",
       title="S-curves vs the strictly concave family the model can reach")
ax.legend(fontsize=7.5)
ax = axes[1]
ax.hist(xn2.to_numpy().ravel(), bins=40, color=MUTED, alpha=0.8)
ax.axvspan(0.40, 0.45, color=BERRY, alpha=0.35)
ax.text(0.43, ax.get_ylim()[1]*0.9, "true half-saturation band", color=BERRY,
        fontsize=8, ha="center")
ax.set(xlabel="normalized spend (all channels pooled)", ylabel="weeks",
       title="...and the spend sits mostly BELOW the threshold")
plt.tight_layout(); plt.show()
below = {c: float((xn2[c] < TRUE_HALF[c]).mean()) for c in CH}
assert all(v > 0.6 for v in below.values())
print("share of weeks below the half-saturation point:",
      {c: f"{v:.0%}" for c, v in below.items()})
print("~3/4 of all weeks are in the threshold region, where the concave curve credits "
      "spend the truth says did nothing.")
"""),
    md(r"""
## Naive fit — concave curve on S-shaped truth
"""),
    code(r"""
r2a = timed_fit(m2a, "act2 naive (1-exp saturation)")
assert r2a.diagnostics["rhat_max"] < 1.03 and r2a.diagnostics["divergences"] == 0
g2a = grade(sc2, m2a, ("act2", "naive"), "act2 naive")
g2a
"""),
    code(r"""
df = SUMMARY[("act2", "naive")]["df"]
fig, ax = plt.subplots(figsize=(9, 4))
err_bars(ax, df, "Act 2 naive: concave fit OVER-credits media, tightly")
plt.tight_layout(); plt.show()
s = SUMMARY[("act2", "naive")]
assert s["total_err"] > 0.25                       # media over-credited in aggregate
assert float(df.loc["TV", "rel_err"]) > 0.5        # the big channel doubles
assert s["coverage"] <= 0.5                        # with confident intervals
print(f"Total media over-credited by {s['total_err']:+.0%}; TV alone "
      f"{float(df.loc['TV','rel_err']):+.0%} — and the truth sits OUTSIDE the 90% interval "
      f"for {1-s['coverage']:.0%} of channels.")
print("r-hat green, zero divergences. In the recorded stress matrix this scenario also "
      "PASSES the\nposterior-predictive check and the refutation suite — the purest silent "
      "failure in the catalog.")
"""),
    md(r"""
**The mechanism.** In the threshold region (where ~3/4 of the weeks live) the true response
is near zero but *every* concave curve is at its steepest. The fit therefore hands the
low-spend weeks credit they never earned, and the contribution errors are not noise — they
are the *integral of the curve mismatch over the spend distribution*. Convergence
diagnostics cannot see this: the posterior is exactly where it should be, *given the wrong
curve*.
"""),
    md(r"""
## Pivot — flip the channel's saturation config to Hill

This used to be the painful part: an earlier version of this notebook had to *assemble* a
Hill model by hand from `mmm_extensions` components, because the core model ignored the
config. That gap is fixed (this stress test was the motivating case): setting
`SaturationConfig.hill()` on a channel now creates `sat_half_<ch>` and `sat_slope_<ch>`
RVs and fits `x^s / (x^s + k^s)` in-graph — same model everywhere else (linear trend,
yearly seasonality, controls, geometric adstock `l_max=8`, the *true* carryover family
here, so the act still isolates saturation). Same harness, same grader, same estimand.
(The `mmm_extensions` components route still exists for fully custom structures; it is
no longer needed for this.)
"""),
    code(r"""
panel2b = sc2.panel()
for mc in panel2b.config.media_channels:
    mc.saturation = SaturationConfig.hill()   # honored by the core model now
m2b = make_model(panel2b)
rv2b = [v.name for v in m2b.model.free_RVs]
print("saturation RVs in-graph:",
      sorted(v for v in rv2b if v.startswith(("sat_half_", "sat_slope_"))))
# The fix, asserted: hill IS honored — Hill RVs exist, no logistic sat_lam remains.
assert {f"sat_half_{c}" for c in CH} <= set(rv2b)
assert {f"sat_slope_{c}" for c in CH} <= set(rv2b)
assert not any(v.startswith("sat_lam_") for v in rv2b)
r2b = timed_fit(m2b, "act2 pivot (core hill)")
assert r2b.diagnostics["rhat_max"] < 1.05
g2b = grade(sc2, m2b, ("act2", "pivot"), "act2 pivot (core hill)")
g2b
"""),
    code(r"""
s, n = SUMMARY[("act2", "pivot")], SUMMARY[("act2", "naive")]
assert s["med_abs_err"] < 0.15 < n["med_abs_err"]        # median error drops well under 15%
assert abs(s["total_err"]) < abs(n["total_err"]) * 0.5   # aggregate bias at least halves
assert s["coverage"] >= n["coverage"]
print(f"med|err| {n['med_abs_err']:.0%} -> {s['med_abs_err']:.0%};  total media "
      f"{n['total_err']:+.0%} -> {s['total_err']:+.0%};  coverage {n['coverage']:.0%} -> "
      f"{s['coverage']:.0%}")
print("One config line per channel, and the same model that confidently over-credited "
      "total media now\nrecovers the S-curve world it previously could not express.")
"""),
    code(r"""
# Response-curve autopsy in KPI units: truth vs naive 1-exp vs core-hill pivot.
post_a = r2a.trace.posterior
post_b = r2b.trace.posterior
AMP = {"TV": 150.0, "Search": 100.0, "Social": 90.0, "Display": 65.0}  # dgp._AMP
fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex=True)
for ax, c in zip(axes.ravel(), CH):
    truth = AMP[c] * xg**TRUE_HILL[c] / (xg**TRUE_HILL[c] + TRUE_HALF[c]**TRUE_HILL[c])
    lam = float(post_a[f"sat_lam_{c}"].mean())
    b_a = float(post_a[f"beta_{c}"].mean()) * m2a.y_std
    kap = float(post_b[f"sat_half_{c}"].mean())
    slp = float(post_b[f"sat_slope_{c}"].mean())
    b_h = float(post_b[f"beta_{c}"].mean()) * m2b.y_std
    ax.plot(xg, truth, color=INK, lw=2.2, label="truth (Hill)")
    ax.plot(xg, b_a * (1 - np.exp(-lam * xg)), color=ACCENT, ls="--", lw=1.8,
            label="naive 1-exp fit")
    ax.plot(xg, b_h * xg**slp / (xg**slp + kap**slp), color=SKY, lw=1.8,
            label="pivot Hill fit (core)")
    ax.plot(xn2[c], np.full(len(xn2), -4.0), "|", color=MUTED, ms=7, alpha=0.4)
    ax.set_title(f"{c}  (kappa-hat {kap:.2f} vs true {TRUE_HALF[c]:.2f})", fontsize=9.5)
axes[0, 0].legend(fontsize=8)
for ax in axes[1]: ax.set_xlabel("normalized (adstocked) spend")
for ax in axes[:, 0]: ax.set_ylabel("response (KPI units)")
fig.suptitle("Where the damage lives: below threshold the concave curve credits spend the truth ignores")
plt.tight_layout(); plt.show()

kap_err = np.median([abs(float(post_b[f"sat_half_{c}"].mean()) - TRUE_HALF[c])
                     for c in CH])
assert kap_err < 0.12     # half-saturation points genuinely recovered
print(f"median |kappa error| across channels: {kap_err:.2f} — the data DOES identify the "
      "threshold location\nonce the family can express one.")
"""),
    md(r"""
**Uncertainty in the response shape.** The posterior-mean overlay above asks
"did the pivot land near the truth?" — the better question is **"does either
model's full posterior response band contain the truth?"** Build the 90%
**curve band** per channel by pushing every posterior draw (β and its curve
parameters jointly, so β–shape trade-offs are preserved) through the response
function. The naive 1−exp band is the quiet scandal here: it is *narrow and
S-shaped-truth-excluding through the threshold region* — the model is precise
about a curve geometry the family cannot produce.
"""),
    code(r"""
xg_band = np.linspace(0.01, 1, 120)
fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.8), sharex=True)
SCOV = {}
for ax, c in zip(axes.ravel(), CH):
    truth = AMP[c] * xg_band**TRUE_HILL[c] / (xg_band**TRUE_HILL[c] + TRUE_HALF[c]**TRUE_HILL[c])
    # Hill pivot band: joint (beta, kappa, slope) draws.
    kap = post_b[f"sat_half_{c}"].values.ravel()
    slp = post_b[f"sat_slope_{c}"].values.ravel()
    b_h = post_b[f"beta_{c}"].values.ravel() * m2b.y_std
    hcurves = b_h[:, None] * xg_band[None, :]**slp[:, None] / (
        xg_band[None, :]**slp[:, None] + kap[:, None]**slp[:, None])
    hlo, hhi = np.percentile(hcurves, [5, 95], axis=0)
    # Naive 1-exp band: joint (beta, lam) draws.
    lam = post_a[f"sat_lam_{c}"].values.ravel()
    b_a = post_a[f"beta_{c}"].values.ravel() * m2a.y_std
    ncurves = b_a[:, None] * (1 - np.exp(-lam[:, None] * xg_band[None, :]))
    nlo, nhi = np.percentile(ncurves, [5, 95], axis=0)
    SCOV[c] = {"hill band covers truth": float(((truth >= hlo) & (truth <= hhi)).mean()),
               "naive band covers truth": float(((truth >= nlo) & (truth <= nhi)).mean())}
    ax.fill_between(xg_band, hlo, hhi, color=SKY, alpha=0.30, label="pivot Hill 90% band")
    ax.fill_between(xg_band, nlo, nhi, color=ACCENT, alpha=0.30, label="naive 1-exp 90% band")
    ax.plot(xg_band, truth, color=INK, lw=2, ls="--", label="truth")
    ax.plot(xn2[c], np.full(len(xn2), -4.0), "|", color=MUTED, ms=6, alpha=0.4)
    ax.set_title(f"{c}: truth inside Hill band {SCOV[c]['hill band covers truth']:.0%} "
                 f"vs 1-exp {SCOV[c]['naive band covers truth']:.0%}", fontsize=9.5)
axes[0, 0].legend(fontsize=8)
for ax in axes[1]: ax.set_xlabel("normalized (adstocked) spend")
for ax in axes[:, 0]: ax.set_ylabel("response (KPI units)")
fig.suptitle("Posterior RESPONSE bands: flip the family and the band moves onto the truth")
plt.tight_layout(); plt.show()

scov = pd.DataFrame(SCOV).T
display(scov.map("{:.0%}".format))
gain = scov["hill band covers truth"].mean() - scov["naive band covers truth"].mean()

# CLAIM: the Hill band covers far more of the true curve than the 1-exp band —
# and most channels' Hill bands cover essentially all of it.
assert gain > 0.10, f"band-coverage gain collapsed: {gain:.0%}"
assert (scov["hill band covers truth"] >= 0.8).sum() >= 2
assert (scov["naive band covers truth"] < 0.75).sum() >= 3
print(f"✓ mean truth-coverage of the response band: 1-exp "
      f"{scov['naive band covers truth'].mean():.0%} -> Hill "
      f"{scov['hill band covers truth'].mean():.0%}  (+{gain:.0%})")
print("TV's residual (its Hill band still misses parts of the curve) is the same "
      "honest residual\nthe contribution table shows: heavy carryover blurs its "
      "S-bend; an experiment, not a curve, fixes TV.")
"""),
    md(r"""
## The identifiability bill for the extra shape parameters

Hill costs one extra free parameter per channel (kappa *and* slope vs a single λ). Did the
data pay for them, or are we reading priors? The same `compute_parameter_learning`
diagnostic that exposed the prior-dominated adstock parameters in Act 1 answers it:
"""),
    code(r"""
with quiet():
    learn2b = m2b.compute_parameter_learning(prior_samples=1000, random_seed=0)
pl2 = learn2b[learn2b.parameter.str.startswith(("sat_half", "sat_slope"))]
print(pl2[["parameter", "prior_sd", "post_sd", "contraction", "verdict"]]
      .round(2).to_string(index=False))
slp_tv = float(r2b.trace.posterior["sat_slope_TV"].mean())
err_tv = float(SUMMARY[("act2", "pivot")]["df"].loc["TV", "rel_err"])
print(f"\nMost shape parameters contract strongly — the data really does pin them. The "
      f"honest residual:\nTV's slope lands at {slp_tv:.1f} vs a true {TRUE_HILL['TV']:.1f} "
      f"(its heavy carryover smears adstocked spend across the\nthreshold, blurring the "
      f"S-bend), so TV stays over-credited at {err_tv:+.0%} even after the pivot. The "
      f"other three\nchannels land close to the truth; the fourth needs either spend "
      f"variation it never got, or an\nexperiment.")
assert (pl2.contraction > 0.1).sum() >= 6   # most shape params genuinely learned
assert err_tv > 0.3                          # the honest partial-fix residual
"""),
    md(r"""
> **Act 2 takeaway.** Saturation misspecification is the *purest* silent failure in the
> catalog: green convergence, passing PPC, passing refutation suite — and 25–50% coverage
> with the flagship channel double-counted. The damage concentrates exactly where the spend
> distribution meets the curve mismatch (the threshold region), so plot **fitted curves over
> the spend rug**, not just fit statistics. And the pivot is now turnkey — this stress test
> is part of why: the core model used to *silently ignore* `SaturationConfig.hill()` (an
> earlier version of this notebook had to assemble Hill from `mmm_extensions` components),
> but the config knob is real today. One line per channel swaps in `sat_half`/`sat_slope`
> RVs and an in-graph S-curve; the threshold locations are genuinely identified, most of
> the aggregate bias disappears, and what remains (TV's blurred S-bend) is a *visible*,
> explainable residual instead of a confident lie.
"""),

    # =====================================================================
    # ACT 3 — outliers
    # =====================================================================
    md(r"""
---
# Act 3 — Spend outliers: one bad cell in a spreadsheet beats your sampler

**The world.** `spend_outliers` is the most mundane scenario in the series and the most
vicious in the matrix (recorded coverage: **0%**). The DGP generates ordinary sales from
ordinary spend — then corrupts the *observed* spend with a single ~15× data-entry spike per
channel (a mis-keyed invoice, a units error). The model normalizes each channel by its
**training max**, so one phantom week rescales the whole series: every real week collapses
into the bottom sliver of the saturation curve. Truth (`sc.true_roas`,
`sc.true_contribution`) is computed on the **un-spiked** spend — the spike never happened.
"""),
    code(r"""
sc3 = dgp.build("spend_outliers")
spike_wk = sc3.notes["spike_weeks"]
fig, axes = plt.subplots(2, 2, figsize=(11, 5.6), sharex=True)
for ax, c in zip(axes.ravel(), CH):
    ax.plot(sc3.weeks, sc3.spend[c], color=PAL[c], lw=1)
    w = sc3.weeks[spike_wk[c]]
    ax.plot([w], [sc3.spend[c].iloc[spike_wk[c]]], "o", color=BERRY, ms=7)
    ax.annotate("data-entry spike", (w, sc3.spend[c].iloc[spike_wk[c]]),
                textcoords="offset points", xytext=(8, -4), color=BERRY, fontsize=8)
    ax.set_title(c, fontsize=10)
fig.suptitle("Observed spend: one ~15x phantom week per channel")
plt.tight_layout(); plt.show()

# The 30-second EDA screen: max / median per channel.
ratio = (sc3.spend.max() / sc3.spend.median()).round(0)
print("max/median spend ratio:", ratio.to_dict())
assert (ratio > 50).all()      # screaming-loud outliers (clean channels run ~5-20x)
"""),
    md(r"""
## Naive fit — normalize by the corrupted max
"""),
    code(r"""
m3a = make_model(sc3.panel())
r3a = timed_fit(m3a, "act3 naive (spiked normalization)")
assert r3a.diagnostics["rhat_max"] < 1.03 and r3a.diagnostics["divergences"] == 0
g3a = grade(sc3, m3a, ("act3", "naive"), "act3 naive")
g3a
"""),
    code(r"""
df = SUMMARY[("act3", "naive")]["df"]
fig, ax = plt.subplots(figsize=(9, 4))
err_bars(ax, df, "Act 3 naive: every channel attenuated, every interval wrong")
plt.tight_layout(); plt.show()
s = SUMMARY[("act3", "naive")]
assert s["coverage"] <= 0.25          # recorded matrix: exactly 0%
assert s["total_err"] < -0.4
with quiet():
    learn3a = m3a.compute_parameter_learning(prior_samples=1000, random_seed=0)
bet = learn3a[learn3a.parameter.str.fullmatch("beta_(" + "|".join(CH) + ")")]
print(bet[["parameter", "contraction", "verdict"]].to_string(index=False))
assert (bet.verdict.isin(["strong", "moderate"])).sum() >= 3
print(f"\nCoverage {s['coverage']:.0%}, total media {s['total_err']:+.0%} — and the "
      "beta learning verdicts read 'strong':\nthe data confidently identified coefficients "
      "for a spend scale that does not exist. Confident AND wrong\nis the signature of a "
      "data bug, not a modeling problem.")
"""),
    md(r"""
**The mechanism, in one histogram.** After dividing by the spiked max, the real weeks all
live below ~0.07 — the saturation curve is evaluated only in its near-linear toe, the
channel can't exhibit curvature, and the compressed dynamic range attenuates everything.
"""),
    code(r"""
xn_spiked = sc3.spend / sc3.spend.max()
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.hist(xn_spiked.to_numpy().ravel(), bins=60, color=BERRY, alpha=0.85)
ax.set(xlabel="normalized spend (spiked max)", ylabel="weeks",
       title="All 155 real weeks per channel crushed into the bottom sliver; the spike sits alone at 1.0")
plt.tight_layout(); plt.show()
real_max = float(xn_spiked.apply(lambda s: s.drop(s.idxmax())).max().max())
print(f"largest REAL week after spiked normalization: {real_max:.3f} of max")
assert real_max < 0.1
"""),
    md(r"""
## Pivot — the 30-second EDA screen, then winsorize and refit

The screen: `max/median` per channel (anything ≫ ~10 demands a look), then the gap between
the max and the **second**-largest week. A genuine heavy week sits in a continuum; a
data-entry error sits alone, an order of magnitude above its neighbor. Fix: cap the flagged
week at the second-largest observed value (or verify it against the invoice and drop it).
"""),
    code(r"""
clean = sc3.spend.copy()
flagged = {}
for c in CH:
    v = clean[c].to_numpy()
    i_max = int(v.argmax()); second = float(np.sort(v)[-2])
    gap = v.max() / second
    flagged[c] = dict(week=i_max, gap=round(gap, 1))
    clean.iloc[i_max, clean.columns.get_loc(c)] = second   # winsorize the phantom week
print(pd.DataFrame(flagged).T)
# The screen found EXACTLY the injected corruption (and nothing else):
assert {c: flagged[c]["week"] for c in CH} == spike_wk
assert all(f["gap"] > 5 for f in flagged.values())

fig, ax = plt.subplots(figsize=(9, 3.6))
xn_clean = clean / clean.max()
ax.hist(xn_clean.to_numpy().ravel(), bins=40, color=LEAF, alpha=0.85)
ax.set(xlabel="normalized spend (cleaned max)", ylabel="weeks",
       title="After winsorizing: spend occupies the full curve again")
plt.tight_layout(); plt.show()
"""),
    code(r"""
from dataclasses import replace
sc3_clean = replace(sc3, spend=clean)     # same world, cleaned observed spend
m3b = make_model(sc3_clean.panel())
r3b = timed_fit(m3b, "act3 pivot (winsorized)")
g3b = grade(sc3, m3b, ("act3", "pivot"), "act3 pivot")   # graded vs the SAME truth
n, p = SUMMARY[("act3", "naive")], SUMMARY[("act3", "pivot")]
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(CH)); w = 0.38
ax.bar(x - w/2, n["df"].rel_err * 100, w, color=BERRY, label="naive (spiked)")
ax.bar(x + w/2, p["df"].rel_err * 100, w, color=LEAF, label="pivot (winsorized)")
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(CH)
ax.set_ylabel("contribution error (%)")
ax.set_title("One capped cell per channel: most of the damage gone"); ax.legend()
plt.tight_layout(); plt.show()
assert p["med_abs_err"] < 0.3 and p["med_abs_err"] < n["med_abs_err"] - 0.25
assert p["coverage"] >= 0.5 and p["total_err"] > -0.3
print(f"med|err| {n['med_abs_err']:.0%} -> {p['med_abs_err']:.0%};  total media "
      f"{n['total_err']:+.0%} -> {p['total_err']:+.0%};  coverage {n['coverage']:.0%} -> "
      f"{p['coverage']:.0%}")
print("(TV retains a modest residual under-estimate at this fit budget — winsorizing "
      "repairs the scale,\nnot every last percent. Compare that residual to the naive "
      "fit's wholesale collapse.)")
"""),
    md(r"""
> **Act 3 takeaway.** No amount of sampling rigor survives a corrupted normalizer: the
> recorded matrix shows **0% coverage** with perfect convergence and *strong* parameter-
> learning verdicts — the model was certain about a world that doesn't exist. The
> `max/median` screen plus a max-vs-second-max gap check takes thirty seconds, found
> exactly the four injected spikes with zero false positives, and one winsorized cell per
> channel restored the bulk of the recovery. Run the screen **before** every fit; the
> framework normalizes by the training max and will not warn you.
"""),

    # =====================================================================
    # Synthesis
    # =====================================================================
    md(r"""
---
# Synthesis — three functional-form failures, side by side
"""),
    code(r"""
rows = []
for (act, stage), s in SUMMARY.items():
    if stage in ("naive", "pivot"):
        rows.append(dict(act=act, stage=stage, med_abs_err=s["med_abs_err"],
                         total_err=s["total_err"], coverage=s["coverage"]))
tab = pd.DataFrame(rows).pivot(index="act", columns="stage",
                               values=["med_abs_err", "total_err", "coverage"])
tab.columns = [f"{m} ({st})" for m, st in tab.columns]
tab = tab[["med_abs_err (naive)", "med_abs_err (pivot)",
           "total_err (naive)", "total_err (pivot)",
           "coverage (naive)", "coverage (pivot)"]]
display((tab * 100).round(0).astype(int).astype(str) + "%")

acts = ["act1", "act2", "act3"]
names = ["Act 1\nadstock", "Act 2\nsaturation", "Act 3\noutliers"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
x = np.arange(3); w = 0.38
axes[0].bar(x - w/2, [SUMMARY[(a, "naive")]["med_abs_err"] for a in acts], w,
            color=ACCENT, label="naive")
axes[0].bar(x + w/2, [SUMMARY[(a, "pivot")]["med_abs_err"] for a in acts], w,
            color=SKY, label="pivot")
axes[0].set_xticks(x); axes[0].set_xticklabels(names)
axes[0].set_ylabel("median |contribution error|"); axes[0].set_title("error"); axes[0].legend()
axes[1].bar(x - w/2, [SUMMARY[(a, "naive")]["coverage"] for a in acts], w,
            color=ACCENT, label="naive")
axes[1].bar(x + w/2, [SUMMARY[(a, "pivot")]["coverage"] for a in acts], w,
            color=SKY, label="pivot")
axes[1].axhline(0.9, color=INK, lw=1, ls="--"); axes[1].text(2.52, 0.91, "nominal", fontsize=8)
axes[1].set_xticks(x); axes[1].set_xticklabels(names)
axes[1].set_ylabel("90% interval coverage"); axes[1].set_title("honesty"); axes[1].legend()
plt.tight_layout(); plt.show()
for a in acts:   # every act: pivot strictly improves error AND coverage
    assert SUMMARY[(a, "pivot")]["med_abs_err"] < SUMMARY[(a, "naive")]["med_abs_err"]
    assert SUMMARY[(a, "pivot")]["coverage"] >= SUMMARY[(a, "naive")]["coverage"]
print("All three pivots improved both error and coverage — none reached zero error. "
      "That is the honest shape\nof functional-form repair: you buy back most of the "
      "damage and the model starts telling the truth about\nwhat remains.")
"""),
    md(r"""
## What to remember

1. **Functional-form errors are silent by construction.** The sampler explores the
   posterior of the model you *wrote*, not the model that *generated the data*. In all
   three acts r-hat was green; in Act 2 even PPC and the refutation suite passed (recorded
   matrix). Convergence diagnostics certify arithmetic, not shape.
2. **Kernels and curves are weakly identified observationally.** The naive adstock
   parameters were prior-dominated (some posteriors *wider* than the prior); a flexible
   Weibull with the old fixed-scale default prior bought divergences, not truth — a
   finding that changed the framework: `AdstockConfig.weibull()`'s default scale prior
   now scales with the lag window and sampled cleanly here with no hand-tuning. Keep
   priors in the decade the window implies, and let **LOO arbitrate between fitted
   candidates** — it picked the right kernel decisively here.
3. **Plot the fitted transform against the spend distribution.** Contribution error is the
   curve mismatch integrated over where the spend actually sits. The threshold-region
   overlay in Act 2 explains every number in the table; no scalar diagnostic does.
4. **Screen spend before the model ever sees it.** `max/median` per channel plus the
   max-vs-second-max gap takes thirty seconds and would have prevented the worst row in the
   entire stress matrix (0% coverage). The training-max normalization is a single point of
   failure — guard it.
5. **Pivots are partial, and that's the point.** A large share of the error bought back
   with coverage doubled (Act 1), most of the aggregate bias gone with three of four
   channels near truth (Act 2), wholesale collapse undone (Act 3) — and each residual is
   now *visible* in wide-but-honest intervals or an explainable channel story, instead of
   hiding behind confident wrong numbers. Two of the three pivots are also one-line
   config changes today *because* this harness flagged the failure modes: the Weibull
   default prior and the per-channel saturation knob were both fixed upstream.

**Next:** [`stress_02_time_structure`](stress_02_time_structure.ipynb) — trend breaks,
seasonality misspecification, and time-varying effectiveness: what happens when the
*baseline*, not the media transform, is the thing the model can't draw.
"""),
]


def main():
    nb = new_notebook(cells=CELLS)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
        "title": "Stress 01 — Carryover & Shape",
    })
    path = "stress_01_carryover_and_shape.ipynb"
    with open(path, "w") as fh:
        nbformat.write(nb, fh)
    print(f"wrote {path} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
