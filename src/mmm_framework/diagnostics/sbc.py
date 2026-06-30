"""Simulation-Based Calibration (SBC) for the inference engine (Talts et al. 2018).

SBC machine-checks whether the posteriors this framework reports are *calibrated*
— i.e. whether their credible intervals have nominal coverage — rather than
merely asserting it. It belongs with the **prior-predictive checks**: it runs on
data generated from the model's own prior, so it validates the inference engine
itself, independent of any real dataset.

Algorithm (for each of ``n_sims`` iterations):

1. draw a "true" parameter vector ``θ*`` from the prior and simulate a dataset
   ``y_sim`` from the likelihood at ``θ*`` (one ``sample_prior_predictive`` call
   gives both, jointly);
2. refit the posterior on ``y_sim`` — on the SAME model graph (so ``θ*`` and the
   posterior live on the identical, fixed standardized scale);
3. record the rank ``r = #{posterior draws ≤ θ*}`` of the true value within its
   ``L`` (thinned) posterior draws.

Under a correctly-calibrated inference procedure those ranks are
**Uniform{0..L}**. The shape of the deviation diagnoses the failure (rank counted
as draws ≤ θ*, the Talts convention):

* **∪ / U-shape** (mass at the edges) → posterior **too narrow / overconfident**;
  the true value lands in the posterior tails too often → reported intervals
  **under-cover**. *Fix:* widen priors / loosen the noise prior; do not trust the
  intervals.
* **∩ / frown** (mass in the centre) → posterior **too wide / overdispersed**;
  intervals over-cover (conservative). *Fix:* tighten priors.
* **left-skew** (mass at low ranks, mean rank < L/2) → posterior sits **above**
  the truth → estimates **biased high**. *Fix:* check prior centring /
  standardization.
* **right-skew** (mass at high ranks) → posterior **biased low**.

(NB: this is the standard Talts 2018 direction — ∪ = underdispersed/overconfident.
It is the opposite of the labels in some informal write-ups; the conjugate unit
test in ``tests/test_sbc.py`` pins the direction.)

SBC is **expensive** — one model refit per simulation — so it is an offline
verification tool gated behind a background job, run once per model architecture,
never per fit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# Pure statistics (no PyMC) — directly unit-tested on a conjugate model
# ─────────────────────────────────────────────────────────────────────────────


def _bin_probs(L: int, n_bins: int) -> np.ndarray:
    """Exact per-bin probabilities of a discrete Uniform{0..L} under equal-width
    bins over ``[0, L]``. Handles ``(L+1)`` not divisible by ``n_bins`` so the
    χ² test keeps the right size for any ``L``."""
    edges = np.linspace(-0.5, L + 0.5, n_bins + 1)
    ints = np.arange(L + 1)
    counts, _ = np.histogram(ints, bins=edges)
    return counts / (L + 1)


def uniformity_chisq(
    int_ranks: np.ndarray, L: int, n_bins: int = 20
) -> tuple[float, float, np.ndarray]:
    """χ² goodness-of-fit of the rank histogram against the discrete uniform.

    Returns ``(chi2, p_value, bin_counts)``. ``p_value`` large ⇒ ranks look
    uniform ⇒ calibrated on this parameter.
    """
    r = np.asarray(int_ranks, dtype=float)
    n = r.size
    edges = np.linspace(-0.5, L + 0.5, n_bins + 1)
    observed, _ = np.histogram(r, bins=edges)
    probs = _bin_probs(L, n_bins)
    expected = n * probs
    # guard tiny expected counts (rare with sane n_bins)
    expected = np.where(expected <= 0, 1e-9, expected)
    chi2 = float(np.sum((observed - expected) ** 2 / expected))
    dof = n_bins - 1
    p = float(stats.chi2.sf(chi2, dof))
    return chi2, p, observed


def normalized_ranks(int_ranks: np.ndarray, L: int) -> np.ndarray:
    """``u = (r + 0.5) / (L + 1)`` ∈ (0, 1); Uniform(0, 1) under calibration."""
    return (np.asarray(int_ranks, dtype=float) + 0.5) / (L + 1.0)


def classify_shape(
    int_ranks: np.ndarray, L: int, *, chi2_p: float | None = None, alpha: float = 0.05
) -> dict[str, Any]:
    """Classify the rank-histogram geometry into a calibration verdict.

    Uses N-aware z-scores so thresholds scale with the number of simulations:
    a bias z-score on the mean normalized rank (uniform mean = 0.5) and a
    dispersion z-score on its variance (uniform var = 1/12). Direction follows
    Talts 2018 (rank = #draws ≤ θ*).

    Returns ``{shape, mean_norm_rank, var_norm_rank, skewness, excess_kurtosis,
    bias_z, dispersion_z}`` where ``shape`` ∈ {``uniform``, ``smile(∪)``,
    ``frown(∩)``, ``left-skew``, ``right-skew``}.
    """
    u = normalized_ranks(int_ranks, L)
    n = u.size
    mean_u = float(np.mean(u))
    var_u = float(np.var(u, ddof=1)) if n > 1 else float("nan")
    skew = float(stats.skew(u)) if n > 2 else 0.0
    kurt = float(stats.kurtosis(u)) if n > 3 else 0.0  # excess

    # Sampling SEs under H0 (u ~ Uniform(0,1)): mean var = 1/12.
    se_mean = np.sqrt((1.0 / 12.0) / n)
    bias_z = (mean_u - 0.5) / se_mean if se_mean > 0 else 0.0
    # Var of the sample variance of a Uniform(0,1): (μ4 − σ⁴(n−3)/(n−1))/n,
    # μ4 = 1/80, σ⁴ = 1/144 → ≈ (1/80 − 1/144)/n for moderate n.
    var_var = max((1.0 / 80.0 - 1.0 / 144.0) / n, 1e-12)
    disp_z = (var_u - 1.0 / 12.0) / np.sqrt(var_var) if np.isfinite(var_u) else 0.0

    # If the histogram is statistically indistinguishable from uniform, say so.
    if chi2_p is not None and chi2_p > alpha:
        shape = "uniform"
    elif abs(bias_z) >= 2.0 and abs(bias_z) >= abs(disp_z):
        # mean rank below 0.5 ⇒ few draws ≤ θ* ⇒ posterior sits ABOVE truth ⇒ high
        shape = "left-skew" if mean_u < 0.5 else "right-skew"
    elif disp_z >= 2.0:
        shape = "smile(∪)"  # over-wide variance of ranks ⇒ edges ⇒ too narrow
    elif disp_z <= -2.0:
        shape = "frown(∩)"  # ranks bunched in the centre ⇒ too wide
    else:
        shape = "uniform"

    return {
        "shape": shape,
        "mean_norm_rank": mean_u,
        "var_norm_rank": var_u,
        "skewness": skew,
        "excess_kurtosis": kurt,
        "bias_z": float(bias_z),
        "dispersion_z": float(disp_z),
    }


def miscalibration_score(bin_counts: np.ndarray, L: int, n_bins: int) -> float:
    """Total-variation distance of the rank histogram from uniform, in ``[0, 1)``.

    ``0.5 · Σ_b |O_b/N − p_b|`` with ``p_b`` the exact discrete-uniform bin
    probability. 0 = perfectly uniform; → ``1 − min_b p_b`` when all mass is in
    one bin. Monotone and comparable across parameters, so the agent can rank the
    worst offenders by a single number instead of "reading the picture".
    """
    counts = np.asarray(bin_counts, dtype=float)
    n = counts.sum()
    if n <= 0:
        return 0.0
    probs = _bin_probs(L, n_bins)
    return float(0.5 * np.sum(np.abs(counts / n - probs)))


def _simultaneous_gamma(
    n_sims: int, probs: np.ndarray, prob: float, n_mc: int = 3000, seed: int = 12345
) -> float:
    """Per-bin two-sided level ``γ`` giving simultaneous coverage ``prob`` for the
    bin counts of a ``Multinomial(n_sims, probs)`` (multiplicity adjustment).

    Estimated by Monte-Carlo + binary search (deterministic via fixed seed,
    cached by the public callers). Returns the smallest ``γ`` whose simultaneous
    coverage ≥ ``prob`` (the tightest valid band).
    """
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_sims, probs, size=n_mc)  # (n_mc, B)
    lo, hi = prob, 1.0 - 1e-9
    best = hi
    for _ in range(34):
        g = 0.5 * (lo + hi)
        lows = stats.binom.ppf((1.0 - g) / 2.0, n_sims, probs)
        highs = stats.binom.ppf((1.0 + g) / 2.0, n_sims, probs)
        inside = np.all((counts >= lows) & (counts <= highs), axis=1)
        if inside.mean() >= prob:
            best = g
            hi = g
        else:
            lo = g
    return best


def rank_hist_band(
    n_sims: int, L: int, n_bins: int = 20, prob: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Simultaneous confidence band for the rank-histogram **bin counts**.

    Returns ``(lower, upper)`` arrays of length ``n_bins`` (per-bin counts the
    histogram should fall inside ``prob`` of the time, jointly across bins, under
    calibration). For equal-width bins the band is near-constant. Built from the
    Multinomial null with the multiplicity-adjusted level from
    :func:`_simultaneous_gamma`.
    """
    probs = _bin_probs(L, n_bins)
    g = _simultaneous_gamma(n_sims, probs, prob)
    lower = stats.binom.ppf((1.0 - g) / 2.0, n_sims, probs)
    upper = stats.binom.ppf((1.0 + g) / 2.0, n_sims, probs)
    return lower, upper


def ecdf_diff_band(
    n_sims: int,
    prob: float = 0.95,
    n_points: int = 100,
    n_mc: int = 2000,
    seed: int = 12345,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Säilynoja-style simultaneous band for the **ECDF-difference** plot.

    Under calibration the normalized ranks are Uniform(0,1), so at grid point
    ``z`` the ECDF satisfies ``N·ECDF(z) ~ Binomial(N, z)``. Returns
    ``(z, lower, upper)`` for ``ECDF(z) − z`` such that the whole curve stays in
    the band ``prob`` of the time jointly across the grid (multiplicity-adjusted
    per-point level found by Monte-Carlo + binary search).
    """
    z = np.linspace(1.0 / (n_points + 1), n_points / (n_points + 1.0), n_points)
    rng = np.random.default_rng(seed)
    samples = np.sort(rng.random((n_mc, n_sims)), axis=1)
    ecdf = np.empty((n_mc, n_points))
    for m in range(n_mc):
        ecdf[m] = np.searchsorted(samples[m], z, side="right") / n_sims

    lo, hi = prob, 1.0 - 1e-9
    best = hi
    for _ in range(30):
        g = 0.5 * (lo + hi)
        low = stats.binom.ppf((1.0 - g) / 2.0, n_sims, z) / n_sims
        high = stats.binom.ppf((1.0 + g) / 2.0, n_sims, z) / n_sims
        inside = np.all((ecdf >= low) & (ecdf <= high), axis=1)
        if inside.mean() >= prob:
            best = g
            hi = g
        else:
            lo = g
    low = stats.binom.ppf((1.0 - best) / 2.0, n_sims, z) / n_sims
    high = stats.binom.ppf((1.0 + best) / 2.0, n_sims, z) / n_sims
    return z, low - z, high - z


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SBCParamStat:
    """Per-parameter SBC verdict + the integer ranks behind it."""

    name: str
    int_ranks: np.ndarray
    L: int
    n_sims: int
    n_bins: int
    bin_counts: np.ndarray
    chi2_stat: float
    chi2_pvalue: float
    shape: str
    mean_norm_rank: float
    var_norm_rank: float
    skewness: float
    excess_kurtosis: float
    bias_z: float
    dispersion_z: float
    miscalibration: float
    calibrated: bool

    def to_dashboard(self, *, max_ranks: int = 0) -> dict[str, Any]:
        """JSON/msgpack-safe summary (numpy scalars cast to float/int)."""
        out: dict[str, Any] = {
            "name": self.name,
            "L": int(self.L),
            "n_sims": int(self.n_sims),
            "n_bins": int(self.n_bins),
            "bin_counts": [int(c) for c in self.bin_counts],
            "chi2_stat": float(self.chi2_stat),
            "chi2_pvalue": float(self.chi2_pvalue),
            "shape": self.shape,
            "mean_norm_rank": float(self.mean_norm_rank),
            "var_norm_rank": float(self.var_norm_rank),
            "skewness": float(self.skewness),
            "excess_kurtosis": float(self.excess_kurtosis),
            "bias_z": float(self.bias_z),
            "dispersion_z": float(self.dispersion_z),
            "miscalibration": float(self.miscalibration),
            "calibrated": bool(self.calibrated),
        }
        if max_ranks:
            out["int_ranks"] = [int(r) for r in self.int_ranks[:max_ranks]]
        return out


@dataclass
class SBCResult:
    """Full SBC run: per-parameter stats + run metadata."""

    params: list[SBCParamStat]
    n_sims_requested: int
    n_sims_effective: int
    L: int
    n_bins: int
    sampler: str
    seed: int
    alpha: float
    elapsed_s: float = 0.0
    n_failed_fits: int = 0
    caveats: list[str] = field(default_factory=list)

    @property
    def all_calibrated(self) -> bool:
        return bool(self.params) and all(p.calibrated for p in self.params)

    def worst(self) -> SBCParamStat | None:
        return max(self.params, key=lambda p: p.miscalibration) if self.params else None

    def summary(self) -> str:
        verdict = "CALIBRATED" if self.all_calibrated else "MISCALIBRATION DETECTED"
        lines = [
            f"SBC ({verdict}) — {self.n_sims_effective}/{self.n_sims_requested} "
            f"simulations, L={self.L} {self.sampler} draws/fit:"
        ]
        for p in sorted(self.params, key=lambda q: q.miscalibration, reverse=True):
            flag = "ok" if p.calibrated else f"⚠ {p.shape}"
            lines.append(
                f"  - {p.name}: {flag} (χ² p={p.chi2_pvalue:.3f}, "
                f"miscal={p.miscalibration:.3f})"
            )
        return "\n".join(lines)

    def to_dashboard(self) -> dict[str, Any]:
        return {
            "all_calibrated": self.all_calibrated,
            "n_sims_requested": int(self.n_sims_requested),
            "n_sims_effective": int(self.n_sims_effective),
            "L": int(self.L),
            "n_bins": int(self.n_bins),
            "sampler": self.sampler,
            "seed": int(self.seed),
            "alpha": float(self.alpha),
            "elapsed_s": float(self.elapsed_s),
            "n_failed_fits": int(self.n_failed_fits),
            "caveats": list(self.caveats),
            "params": [p.to_dashboard() for p in self.params],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Assembling stats from ranks (the pure core the conjugate test exercises)
# ─────────────────────────────────────────────────────────────────────────────
def compute_param_stat(
    name: str,
    int_ranks: np.ndarray,
    L: int,
    *,
    n_bins: int = 20,
    alpha: float = 0.05,
    tau: float = 0.15,
) -> SBCParamStat:
    """Build the full per-parameter verdict from its integer ranks."""
    r = np.asarray(int_ranks, dtype=int)
    n = int(r.size)
    nb = min(n_bins, L + 1)
    chi2, p, counts = uniformity_chisq(r, L, nb)
    shp = classify_shape(r, L, chi2_p=p, alpha=alpha)
    miscal = miscalibration_score(counts, L, nb)
    # Calibration verdict follows the N-aware shape classifier (which already
    # folds in the χ² p-value and bias/dispersion z-scores). The miscalibration
    # score is a descriptive magnitude only — at tiny N its TV distance is
    # mechanically inflated, so gating on it would flag a false failure when the
    # run is merely under-powered (the n_sims caveat covers that epistemics).
    calibrated = bool(shp["shape"] == "uniform")
    _ = tau  # retained for API stability; not used as a gate
    return SBCParamStat(
        name=name,
        int_ranks=r,
        L=int(L),
        n_sims=n,
        n_bins=int(nb),
        bin_counts=counts.astype(int),
        chi2_stat=float(chi2),
        chi2_pvalue=float(p),
        shape=shp["shape"],
        mean_norm_rank=shp["mean_norm_rank"],
        var_norm_rank=shp["var_norm_rank"],
        skewness=shp["skewness"],
        excess_kurtosis=shp["excess_kurtosis"],
        bias_z=shp["bias_z"],
        dispersion_z=shp["dispersion_z"],
        miscalibration=miscal,
        calibrated=calibrated,
    )


def build_sbc_result(
    int_ranks_by_param: dict[str, np.ndarray],
    *,
    L: int,
    n_sims_requested: int,
    sampler: str = "generic",
    n_bins: int = 20,
    alpha: float = 0.05,
    tau: float = 0.15,
    seed: int = 0,
    elapsed_s: float = 0.0,
    n_failed_fits: int = 0,
) -> SBCResult:
    """Assemble an :class:`SBCResult` from per-parameter integer ranks."""
    params = [
        compute_param_stat(name, ranks, L, n_bins=n_bins, alpha=alpha, tau=tau)
        for name, ranks in int_ranks_by_param.items()
    ]
    n_eff = max((p.n_sims for p in params), default=0)
    caveats: list[str] = []
    if n_eff < 50:
        caveats.append(
            f"Only {n_eff} effective simulations — the bands are wide and only "
            "gross miscalibration is detectable. Increase n_sims for a sharper read."
        )
    if n_failed_fits:
        caveats.append(
            f"{n_failed_fits} fit(s) failed (NaN/divergence) and were dropped; "
            "treat the verdict cautiously if this is a large fraction."
        )
    if sampler in ("advi", "fullrank_advi"):
        caveats.append(
            "Sampler is variational (ADVI): SBC here tests ADVI's calibration, "
            "which can look mildly miscalibrated even when NUTS would not. "
            "Re-run with the numpyro/NUTS sampler to judge the production posterior."
        )
    return SBCResult(
        params=params,
        n_sims_requested=int(n_sims_requested),
        n_sims_effective=int(n_eff),
        L=int(L),
        n_bins=int(n_bins),
        sampler=sampler,
        seed=int(seed),
        alpha=float(alpha),
        elapsed_s=float(elapsed_s),
        n_failed_fits=int(n_failed_fits),
        caveats=caveats,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Generic SBC loop (used by the conjugate test; no PyMC)
# ─────────────────────────────────────────────────────────────────────────────
def run_sbc(
    draw_and_fit: Callable[
        [np.random.Generator], tuple[dict[str, float], dict[str, np.ndarray]]
    ],
    *,
    n_sims: int,
    L: int,
    n_bins: int = 20,
    alpha: float = 0.05,
    seed: int = 0,
    sampler: str = "generic",
) -> SBCResult:
    """Model-agnostic SBC loop.

    ``draw_and_fit(rng)`` returns ``(theta_star, posterior_draws)`` where
    ``theta_star`` maps param→scalar truth and ``posterior_draws`` maps the same
    param→a 1-D array of posterior draws (length ≥ ``L``; thinned to ``L``).
    """
    rng = np.random.default_rng(seed)
    ranks: dict[str, list[int]] = {}
    t0 = time.perf_counter()
    failed = 0
    for _ in range(n_sims):
        try:
            theta, post = draw_and_fit(rng)
        except Exception:
            failed += 1
            continue
        for name, truth in theta.items():
            draws = np.asarray(post[name], dtype=float).ravel()
            draws = _thin_to(draws, L)
            ranks.setdefault(name, []).append(int(np.sum(draws <= float(truth))))
    elapsed = time.perf_counter() - t0
    return build_sbc_result(
        {k: np.asarray(v) for k, v in ranks.items()},
        L=L,
        n_sims_requested=n_sims,
        sampler=sampler,
        n_bins=n_bins,
        alpha=alpha,
        seed=seed,
        elapsed_s=elapsed,
        n_failed_fits=failed,
    )


def _thin_to(draws: np.ndarray, L: int) -> np.ndarray:
    """Evenly thin ``draws`` to exactly ``L`` samples (kills NUTS autocorrelation
    and pins the rank range to {0..L} across fits)."""
    draws = np.asarray(draws, dtype=float).ravel()
    if draws.size <= L:
        return draws
    idx = np.linspace(0, draws.size - 1, L).round().astype(int)
    return draws[idx]


# ─────────────────────────────────────────────────────────────────────────────
# MMM SBC (refit on simulated y via pm.observe on the SAME graph)
# ─────────────────────────────────────────────────────────────────────────────
def _scalar_param_names(model: Any) -> list[str]:
    """Free-RV names whose draws are scalar per simulation (dims = chain, draw).

    These are the rankable SBC targets (channel β, adstock α, saturation, σ,
    intercept, trend slope). Vector RVs (per-geo β, Fourier seasonality) are
    skipped by default — they would need per-component expansion.
    """
    names = [rv.name for rv in model.model.free_RVs]
    out = []
    for n in names:
        try:
            shape = model.model[n].shape.eval()
            if int(np.prod(shape)) == 1:
                out.append(n)
        except Exception:
            # Fall back to the named dims of the RV value var.
            out.append(n)
    return out


def _fit_swapped(
    swapped_model: Any,
    *,
    sampler: str,
    L: int,
    tune: int,
    chains: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Refit the observation-swapped model and return pooled posterior draws per
    free RV. ``sampler``: ``numpyro``/``nuts`` (NUTS) or ``advi``/``fullrank_advi``."""
    import pymc as pm

    draws_per_chain = max(int(np.ceil(L / max(chains, 1))), 25)
    with swapped_model:
        if sampler in ("advi", "fullrank_advi"):
            method = "fullrank_advi" if sampler == "fullrank_advi" else "advi"
            approx = pm.fit(n=20000, method=method, progressbar=False, random_seed=seed)
            idata = approx.sample(L)
        else:
            nuts_sampler = "numpyro" if sampler in ("numpyro", "nuts") else sampler
            idata = pm.sample(
                draws=draws_per_chain,
                tune=tune,
                chains=chains,
                nuts_sampler=nuts_sampler,
                progressbar=False,
                random_seed=seed,
                compute_convergence_checks=False,
            )
    post = idata.posterior
    return {n: np.asarray(post[n].values).ravel() for n in post.data_vars}


def run_mmm_sbc(
    model: Any,
    *,
    n_sims: int = 64,
    L: int = 100,
    n_bins: int = 20,
    sampler: str = "numpyro",
    params: list[str] | None = None,
    tune: int = 200,
    chains: int = 2,
    seed: int = 0,
    alpha: float = 0.05,
    progress: Callable[[int, int], None] | None = None,
) -> SBCResult:
    """Run SBC for a (built, unfitted) :class:`BayesianMMM`.

    Draws ``n_sims`` paired ``(θ*, y_sim)`` from the model's prior, refits the
    posterior on each ``y_sim`` by swapping the observed data on the SAME model
    graph (``pm.observe`` — keeps ``θ*`` and the posterior on one fixed scale),
    and computes per-parameter ranks.

    EXPENSIVE: one refit per simulation. Use a background job; defaults are tuned
    for a tractable national MMM (``numpyro`` NUTS, ``n_sims=64``, ``L=100``).
    """
    import pymc as pm

    t0 = time.perf_counter()
    # One joint prior draw of (params, y_sim) for all sims.
    prior = model.sample_prior_predictive(samples=n_sims, random_seed=seed)
    targets = params or _scalar_param_names(model)
    # keep only targets actually present in the prior group
    targets = [t for t in targets if t in prior.prior.data_vars]
    if not targets:
        raise ValueError("No scalar prior parameters found to calibrate.")

    obs_name = list(model.model.observed_RVs)[0].name  # typically 'y_obs'
    y_sim_all = prior.prior_predictive[obs_name].values  # (1, n_sims, n_obs)
    theta_all = {t: prior.prior[t].values.reshape(-1) for t in targets}

    ranks: dict[str, list[int]] = {t: [] for t in targets}
    failed = 0
    for i in range(n_sims):
        y_sim = np.asarray(y_sim_all[0, i, :], dtype=float)
        try:
            swapped = pm.observe(model.model, {obs_name: y_sim})
            draws = _fit_swapped(
                swapped,
                sampler=sampler,
                L=L,
                tune=tune,
                chains=chains,
                seed=int(seed) + 1009 * (i + 1),
            )
        except Exception:
            failed += 1
            if progress:
                progress(i + 1, n_sims)
            continue
        for t in targets:
            d = _thin_to(draws[t], L)
            ranks[t].append(int(np.sum(d <= float(theta_all[t][i]))))
        if progress:
            progress(i + 1, n_sims)

    elapsed = time.perf_counter() - t0
    return build_sbc_result(
        {t: np.asarray(ranks[t]) for t in targets if ranks[t]},
        L=L,
        n_sims_requested=n_sims,
        sampler=sampler,
        n_bins=n_bins,
        alpha=alpha,
        seed=seed,
        elapsed_s=elapsed,
        n_failed_fits=failed,
    )


__all__ = [
    "uniformity_chisq",
    "normalized_ranks",
    "classify_shape",
    "miscalibration_score",
    "rank_hist_band",
    "ecdf_diff_band",
    "SBCParamStat",
    "SBCResult",
    "compute_param_stat",
    "build_sbc_result",
    "run_sbc",
    "run_mmm_sbc",
]
