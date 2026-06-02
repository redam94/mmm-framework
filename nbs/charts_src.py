"""Chart-cell source for the Aurora showcase notebooks.

`CHARTS[key]` is the exact code that `build_aurora_notebooks.py` bakes into a
notebook code cell (the build script imports this dict; the notebooks themselves
are self-contained once built). Kept in one place so the chart code is
reviewable and regression-checkable with `validate_chart_cells.py`, which runs
every block against tiny 50-draw fits before the expensive full bake.

Each block reuses objects the notebook already computed (model, results,
contrib, roi, marg, scenario, nested, med, mv, ce, ...) and the Aurora palette
(CHANNELS, PRODUCTS, PALETTE, CHANNEL_COLORS, ACCENT, INK, MUTED) — no new MCMC
fitting happens in a chart cell."""

CHARTS = {}

# ===========================================================================
# nb_01 — Causality
# ===========================================================================

CHARTS["nb01_punchline"] = r"""
# The calibration result, drawn: each channel's observational ROAS slides onto its truth.
calib = ["Search", "Social"]                       # the channels the experiment anchored
fig, ax = plt.subplots(figsize=(9, 3.4))
for i, ch in enumerate(calib):
    obs, cal, tru = roi_ctrl[ch], roi_cal[ch], aurora.true_roas[ch]
    ax.plot([obs, cal], [i, i], color=CHANNEL_COLORS[ch], lw=2.5, alpha=.45, zorder=1)
    ax.annotate("", xy=(cal, i), xytext=(obs, i),
                arrowprops=dict(arrowstyle="-|>", color=CHANNEL_COLORS[ch], lw=2))
    ax.scatter(obs, i, s=150, color="white", edgecolor=CHANNEL_COLORS[ch], lw=2, zorder=3)
    ax.scatter(cal, i, s=170, color=CHANNEL_COLORS[ch], zorder=3)
    ax.scatter(tru, i, marker="|", s=520, color=INK, lw=2.5, zorder=4)
    ax.text(obs, i + 0.20, f"observational {obs:.2f}", ha="center", fontsize=8, color=MUTED)
    ax.text(cal, i - 0.26, f"calibrated {cal:.2f}", ha="center", fontsize=8,
            weight="bold", color=CHANNEL_COLORS[ch])
ax.axvline(1.0, color=MUTED, ls="--", lw=1, alpha=.6)
ax.set_yticks(range(len(calib))); ax.set_yticklabels(calib)
ax.set_ylim(-0.65, len(calib) - 0.35)
ax.set_xlim(0, max(float(roi_ctrl[calib].max()), 1.2) * 1.18)
ax.set_xlabel("ROAS (revenue per $1 spend)")
ax.set_title("Experiment calibration pulls observational ROAS onto the truth\n( | marks the true ROAS )")
plt.tight_layout(); plt.show()
"""

CHARTS["nb01_bias"] = r"""
# Bias = recovered - true. Controlling for demand shrinks it, but doesn't kill it.
chs = list(CHANNELS)
bias_blind = (comp["demand-blind"] - comp["true ROAS"]).loc[chs]
bias_ctrl  = (comp["demand-controlled"] - comp["true ROAS"]).loc[chs]
x = np.arange(len(chs)); w = 0.36
fig, ax = plt.subplots(figsize=(9, 3.5))
b1 = ax.bar(x - w/2, bias_blind, w, label="demand-blind",       color=PALETTE["berry"])
b2 = ax.bar(x + w/2, bias_ctrl,  w, label="demand-controlled",  color=PALETTE["sky"])
ax.axhline(0, color=INK, lw=1)
ax.set_xticks(x); ax.set_xticklabels(chs)
ax.set_ylabel("ROAS bias (recovered − true)")
ax.set_title("Controlling for demand shrinks the bias — but Search stays overstated")
ax.bar_label(b1, fmt="%+.2f", padding=3, fontsize=8)
ax.bar_label(b2, fmt="%+.2f", padding=3, fontsize=8)
ax.legend()
plt.tight_layout(); plt.show()
"""

# ===========================================================================
# nb_02 — Base MMM
# ===========================================================================

CHARTS["nb02_forest"] = r"""
# Posterior of each channel's coefficient — the uncertainty behind every ROAS number.
import arviz as az
axes = az.plot_forest(results.trace, var_names=[f"beta_{c}" for c in CHANNELS],
                      combined=True, hdi_prob=0.9, colors=ACCENT, figsize=(9, 3.0))
ax = axes[0]
ax.axvline(0, color=MUTED, ls="--", lw=1)
ax.set_title("Posterior of channel coefficients β (90% HDI)")
plt.tight_layout(); plt.show()
"""

CHARTS["nb02_fit"] = r"""
# Actual vs fitted revenue, with the posterior credible band.
pred = mmm.predict(return_original_scale=True, hdi_prob=0.9)
weeks = aurora.weeks
y = aurora.sales_total
r2 = 1 - np.sum((y - pred.y_pred_mean) ** 2) / np.sum((y - y.mean()) ** 2)
fig, ax = plt.subplots(figsize=(11, 3.9))
ax.fill_between(weeks, pred.y_pred_hdi_low, pred.y_pred_hdi_high, color=ACCENT, alpha=.22, label="90% HDI")
ax.plot(weeks, pred.y_pred_mean, color=ACCENT, lw=2, label="model fit")
ax.plot(weeks, y, color=INK, lw=1.4, label="observed")
ax.set_title(f"Model fit: observed vs predicted weekly revenue ($000s)  —  R² = {r2:.2f}")
ax.set_xlabel("week"); ax.set_ylabel("revenue ($000s)")
ax.legend(ncol=3, loc="upper left")
plt.tight_layout(); plt.show()
"""

CHARTS["nb02_decomp"] = r"""
# Revenue decomposition over time. The baseline (everything non-media) is drawn as a LINE
# and each channel's counterfactual contribution is stacked on top — so the top of the
# stack IS the model's fitted revenue, which the observed series (dotted) tracks.
weeks = aurora.weeks
pred = mmm.predict(return_original_scale=True, hdi_prob=0.9)
cc = contrib.channel_contributions                          # DataFrame (n_obs x n_channels), >= 0
media_sum = np.asarray(cc[list(CHANNELS)].sum(axis=1), float)
baseline = np.asarray(pred.y_pred_mean, float) - media_sum  # non-media baseline (reconciles by construction)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(weeks, baseline, color=INK, lw=1.5, label="baseline (non-media)")
cum = baseline.copy()
for ch in CHANNELS:
    top = cum + np.asarray(cc[ch], float)
    ax.fill_between(weeks, cum, top, color=CHANNEL_COLORS[ch], alpha=.85, label=ch)
    cum = top
ax.plot(weeks, aurora.sales_total, color=INK, lw=1, ls=":", alpha=.75, label="observed")
ax.set_ylim(float(baseline.min()) * 0.90, float(max(cum.max(), aurora.sales_total.max())) * 1.03)
ax.set_title("Revenue decomposition over time — media contribution stacked on the baseline")
ax.set_xlabel("week"); ax.set_ylabel("revenue ($000s)")
ax.legend(ncol=3, loc="upper left", fontsize=8)
plt.tight_layout(); plt.show()
"""

CHARTS["nb02_roas_ci"] = r"""
# ROAS with a 90% credible interval, against the oracle true ROAS.
chs = list(CHANNELS)
rfull = MMMAnalyzer(mmm).compute_channel_roi().set_index("Channel").loc[chs]
spend_ch = rfull["Total Spend"]
roas = rfull["ROI"]
roas_lo = rfull["Contribution HDI Low"] / spend_ch     # absolute HDI bound / spend
roas_hi = rfull["Contribution HDI High"] / spend_ch
tru = aurora.true_roas.loc[chs]
x = np.arange(len(chs))
fig, ax = plt.subplots(figsize=(9, 3.9))
bars = ax.bar(x, roas, color=[CHANNEL_COLORS[c] for c in chs], alpha=.9,
              yerr=[roas - roas_lo, roas_hi - roas], capsize=5, ecolor=INK)
ax.scatter(x, tru, marker="D", s=85, color=INK, zorder=4, label="true ROAS")
ax.axhline(1.0, color=MUTED, ls="--", lw=1)
ax.set_xticks(x); ax.set_xticklabels(chs)
ax.set_ylabel("ROAS (revenue per $1 spend)")
ax.set_title("Estimated ROAS (94% HDI) vs true — TV & Display are undervalued")
ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
ax.legend()
plt.tight_layout(); plt.show()
"""

CHARTS["nb02_marginal"] = r"""
# Average ROAS (the dollar already spent) vs marginal ROAS (the next dollar).
chs = list(CHANNELS)
mm = marg.set_index("Channel").loc[chs]
avg = MMMAnalyzer(mmm).compute_channel_roi().set_index("Channel").loc[chs, "ROI"]
mroas = mm["Marginal ROAS"]
mlo, mhi = mm["Marginal ROAS HDI Low"], mm["Marginal ROAS HDI High"]
x = np.arange(len(chs)); w = 0.38
fig, ax = plt.subplots(figsize=(9, 3.9))
ax.bar(x - w/2, avg, w, label="average ROAS",
       color=[CHANNEL_COLORS[c] for c in chs], alpha=.5)
b2 = ax.bar(x + w/2, mroas, w, label="marginal ROAS (+10% spend)",
            color=[CHANNEL_COLORS[c] for c in chs],
            yerr=[mroas - mlo, mhi - mroas], capsize=4, ecolor=INK)
ax.axhline(1.0, color=MUTED, ls="--", lw=1)
ax.set_xticks(x); ax.set_xticklabels(chs)
ax.set_ylabel("ROAS (revenue per $1)")
ax.set_title("Average vs marginal ROAS — where the next dollar goes (94% HDI)")
ax.legend()
plt.tight_layout(); plt.show()
"""

CHARTS["nb02_whatif"] = r"""
# The what-if, drawn: total impact and the week-by-week revenue path.
weeks = aurora.weeks
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.7), gridspec_kw={"width_ratios": [1, 2]})
ba = a1.bar(["baseline", "scenario"],
            [scenario["baseline_outcome"], scenario["scenario_outcome"]],
            color=[MUTED, ACCENT])
a1.bar_label(ba, fmt="$%.0fk", padding=3, fontsize=8)
a1.set_title(f"Total revenue\n{scenario['outcome_change_pct']:+.2f}%  "
             f"(${scenario['outcome_change']:+,.0f}k)")
a1.set_ylabel("revenue ($000s)")
a2.plot(weeks, scenario["baseline_prediction"], color=MUTED, lw=1.6, label="baseline")
a2.plot(weeks, scenario["scenario_prediction"], color=ACCENT, lw=1.6, label="Search −20%, TV +20%")
a2.fill_between(weeks, scenario["baseline_prediction"], scenario["scenario_prediction"],
                color=ACCENT, alpha=.18)
a2.set_title("Weekly revenue under the shift"); a2.set_xlabel("week")
a2.legend(loc="upper left", fontsize=8)
plt.tight_layout(); plt.show()
"""

# ===========================================================================
# nb_03 — Extended MMM
# ===========================================================================

CHARTS["nb03_cannibal"] = r"""
# The cannibalization posterior, drawn — proof the mass sits below zero.
import arviz as az
psi = mv._trace.posterior["psi_matrix"]
target = float(ce["mean"].iloc[0])
cands = {(1, 0): psi[:, :, 1, 0].values.ravel(), (0, 1): psi[:, :, 0, 1].values.ravel()}
samples = min(cands.values(), key=lambda s: abs(float(s.mean()) - target))
lo, hi = az.hdi(samples, hdi_prob=0.94)
fig, ax = plt.subplots(figsize=(9, 3.7))
ax.hist(samples, bins=45, density=True, color=PALETTE["berry"], alpha=.30, edgecolor="white", lw=.4)
sel = samples[(samples >= lo) & (samples <= hi)]
ax.hist(sel, bins=45, density=True, color=PALETTE["berry"], alpha=.8, label="94% HDI")
ax.axvline(0, color=INK, ls="--", lw=2, label="no effect")
ax.axvline(float(samples.mean()), color=PALETTE["berry"], lw=2)
ax.set_title("Cold Brew → Original: the cannibalization posterior sits below zero")
ax.set_xlabel("cross-effect coefficient"); ax.set_ylabel("posterior density")
ax.legend(loc="upper left")
plt.tight_layout(); plt.show()
"""

CHARTS["nb03_mediation_validate"] = r"""
# Did the model recover the mediation? Model proportion_mediated vs the known truth.
chs = list(CHANNELS)
prop_model = med.loc[chs, "proportion_mediated"].fillna(0.0)
prop_true = aurora.true_mediated_share.loc[chs]
x = np.arange(len(chs)); w = 0.38
fig, ax = plt.subplots(figsize=(9, 3.6))
b1 = ax.bar(x - w/2, prop_true, w, label="true mediated share", color=INK, alpha=.85)
b2 = ax.bar(x + w/2, prop_model, w, label="model proportion_mediated",
            color=[CHANNEL_COLORS[c] for c in chs])
ax.set_xticks(x); ax.set_xticklabels(chs); ax.set_ylim(0, 1.32)
ax.set_ylabel("share of effect via awareness")
ax.set_title("Mediation recovered: model vs truth (TV & Display ≈ fully mediated)")
ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=8)
ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=8)
ax.legend(loc="upper center", ncol=2, fontsize=8)
plt.tight_layout(); plt.show()
"""

CHARTS["nb03_awareness"] = r"""
# The latent mediator, recovered: the model's awareness path vs the survey and the truth.
# (z-scored so the model's internal scale is comparable to the 0-100 survey.)
lat = nested._trace.posterior["awareness_latent"].values    # (chain, draw, obs)
flat = lat.reshape(-1, lat.shape[-1])
mean_raw = flat.mean(0)
mu, sd = float(mean_raw.mean()), float(mean_raw.std())
lat_mean = (mean_raw - mu) / sd
lo = (np.percentile(flat, 5, axis=0) - mu) / sd
hi = (np.percentile(flat, 95, axis=0) - mu) / sd

def _z(a):
    a = np.asarray(a, float)
    return (a - np.nanmean(a)) / np.nanstd(a)

weeks = aurora.weeks
surv = aurora.awareness_survey
obs = ~np.isnan(surv)
fig, ax = plt.subplots(figsize=(11, 3.9))
ax.fill_between(weeks, lo, hi, color=PALETTE["sky"], alpha=.22, label="model latent 90%")
ax.plot(weeks, lat_mean, color=PALETTE["sky"], lw=2, label="model latent awareness")
ax.plot(weeks, _z(aurora.awareness), color=INK, ls="--", lw=1.5, label="true awareness")
ax.scatter(weeks[obs], _z(surv)[obs], color=ACCENT, s=45, zorder=5,
           edgecolor="white", lw=.6, label="monthly survey")
# Focus on the data range so the wide week-1 edge uncertainty doesn't compress the story.
data_z = np.concatenate([_z(aurora.awareness), _z(surv)[obs]])
ax.set_ylim(float(np.nanmin(data_z)) - 0.8, float(np.nanmax(data_z)) + 0.8)
ax.set_title("Mediator recovered: latent awareness tracks the survey and the truth (z-scored)")
ax.set_xlabel("week"); ax.set_ylabel("awareness (z-score)")
ax.legend(ncol=2, loc="upper left", fontsize=8)
plt.tight_layout(); plt.show()
"""

CHARTS["nb03_correlation"] = r"""
# The model's OTHER finding: after media, the two products' residuals still move together —
# the shared-demand shock. Shown as the outcome residual-correlation matrix.
import arviz as az
cm = mv.get_correlation_matrix()                 # DataFrame, indexed by outcome_names
labels = list(PRODUCTS)                          # ['Original', 'Cold Brew'] = outcome order
M = cm.values
off = mv._trace.posterior["Y_obs_correlation"].values[:, :, 0, 1].ravel()
lo, hi = az.hdi(off, hdi_prob=0.94)
fig, ax = plt.subplots(figsize=(5.6, 4.6))
im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                color="white" if abs(M[i, j]) > 0.55 else INK, fontsize=15, weight="bold")
cbar = fig.colorbar(im, ax=ax, shrink=0.82); cbar.set_label("residual correlation")
ax.set_title(f"Shared-demand shock: products' residual correlation = {M[0, 1]:+.2f}\n"
             f"(94% HDI [{lo:.2f}, {hi:.2f}] — a demand wave lifts both, net of media)")
plt.tight_layout(); plt.show()
"""

# ===========================================================================
# nb_05 — Unified workflow
# ===========================================================================

CHARTS["nb05_recovery"] = r"""
# The experiment-anchored model recovers the truth — points hug the 45° line.
chs = list(CHANNELS)
rec = recovered.loc[chs]
lim = float(max(rec["true ROAS"].max(), rec["model ROAS"].max())) * 1.15
fig, ax = plt.subplots(figsize=(5.4, 5))
ax.plot([0, lim], [0, lim], color=MUTED, ls="--", lw=1, label="perfect recovery")
# Per-channel label offsets so TV & Display (which nearly coincide) don't collide.
_lbl_off = {"TV": (8, 9), "Display": (8, -15), "Search": (8, 4), "Social": (8, 4)}
for ch in chs:
    ax.scatter(rec.loc[ch, "true ROAS"], rec.loc[ch, "model ROAS"], s=160,
               color=CHANNEL_COLORS[ch], edgecolor="white", lw=1, zorder=3)
    ax.annotate(ch, (rec.loc[ch, "true ROAS"], rec.loc[ch, "model ROAS"]),
                xytext=_lbl_off.get(ch, (8, 4)), textcoords="offset points", fontsize=9,
                weight="bold", color=CHANNEL_COLORS[ch])
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("true ROAS"); ax.set_ylabel("experiment-anchored model ROAS")
ax.set_title("The model recovers the truth\n(points hug the 45° line)")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout(); plt.show()
"""

CHARTS["nb05_allocation"] = r"""
# Current spend vs the causal plan's recommended spend, by channel.
chs = list(CHANNELS)
current = spend.loc[chs]
recommended = (spend + causal_delta).loc[chs]
x = np.arange(len(chs)); w = 0.38
fig, ax = plt.subplots(figsize=(9, 3.7))
b1 = ax.bar(x - w/2, current, w, label="current spend", color=MUTED)
b2 = ax.bar(x + w/2, recommended, w, label="causal-plan spend",
            color=[CHANNEL_COLORS[c] for c in chs])
ax.set_xticks(x); ax.set_xticklabels(chs)
ax.set_ylabel("annual spend ($000s)")
ax.set_title("Reallocation: fund the brand engines (TV/Display), trim the mirages")
ax.bar_label(b1, fmt="%.0f", padding=2, fontsize=8)
ax.bar_label(b2, fmt="%.0f", padding=2, fontsize=8)
ax.legend()
plt.tight_layout(); plt.show()
"""
