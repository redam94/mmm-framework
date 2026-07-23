"""Interval-coverage diagnostics — does the 90% interval contain the truth 90%
of the time?

Coverage is the property users implicitly assume when they read a credible
interval. This module makes it checkable. Three distinct notions map to three
tools:

1. **Predictive coverage** (observation scale): do posterior-predictive
   intervals contain the observed data at the nominal rate? → posterior
   predictive checks and the report's calibration curve (elsewhere).
2. **Engine calibration** (SBC, :mod:`.sbc`): over datasets simulated from the
   model's own *prior*, are posterior ranks uniform? Uniform ranks ⇔ every
   central interval has nominal coverage *on average over the prior*.
   :func:`coverage_from_ranks` turns SBC ranks into explicit per-level coverage
   numbers ("your 90% interval empirically covers 78% [68–86%]").
3. **Recovery coverage at a fixed truth** (:func:`run_recovery_coverage`, this
   module's headline): pick ONE parameter vector θ* — the fitted posterior
   mean, a prior draw, or user-supplied values — simulate ``n_sims`` datasets
   from the model at θ*, refit each, and count how often the X% interval
   contains θ*, for raw parameters AND per-channel contribution estimands.
   Reports per-target empirical coverage with Monte-Carlo error bars plus a
   bias/width diagnosis: is the failure a *location* problem (posterior sits
   away from the truth) or an *interval-too-narrow* problem (overconfidence)?

What recovery coverage CANNOT see: it simulates data FROM the model, so it can
never detect real-world misspecification (wrong adstock/saturation family,
unobserved confounding, time-varying effects). Read it jointly with an
external-truth check:

* under-covers **here** → the problem is mechanical — an approximate fit
  (MAP/ADVI/Pathfinder uncertainty is not calibrated), a broken sampler, or
  priors so tight the data cannot move the posterior to the truth;
* covers here but failed against an external answer key (e.g. the
  :mod:`mmm_framework.synth` scenarios' ``synthetic_truth.json``) → the gap IS
  structural: misspecification, confounding, or an estimand mismatch
  (contribution ROI and counterfactual ROI are *different numbers*).

The full failure-mode table lives in ``technical-docs/coverage-diagnostics.md``
and :func:`failure_mode_guide`.

Interval convention: central equal-tailed percentile intervals, matching the
framework's ``compute_hdi_bounds`` (which is percentile-based despite the
name), so the coverage measured here is the coverage of the intervals actually
reported.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
from scipy import stats

DEFAULT_LEVELS: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)

# ─────────────────────────────────────────────────────────────────────────────
# Pure statistics (no PyMC) — unit-tested on analytic posteriors
# ─────────────────────────────────────────────────────────────────────────────


def jeffreys_interval(k: int, n: int, prob: float = 0.95) -> tuple[float, float]:
    """Jeffreys (Beta(1/2,1/2)) binomial interval for an empirical proportion.

    The Monte-Carlo error bar on an estimated coverage: with ``n`` simulations
    and ``k`` hits, the coverage estimate is ``k/n`` with this uncertainty.
    """
    if n <= 0:
        return 0.0, 1.0
    a, b = 0.5 + k, 0.5 + (n - k)
    lo = float(stats.beta.ppf((1.0 - prob) / 2.0, a, b)) if k > 0 else 0.0
    hi = float(stats.beta.ppf(1.0 - (1.0 - prob) / 2.0, a, b)) if k < n else 1.0
    return lo, hi


@dataclass
class CoverageLevelStat:
    """Empirical coverage of the central ``level`` interval, with MC error."""

    level: float
    n: int
    hits: int
    coverage: float
    ci_low: float
    ci_high: float
    verdict: str  # "ok" | "under" | "over" (binomial CI excludes the nominal)

    def to_dashboard(self) -> dict[str, Any]:
        return {
            "level": float(self.level),
            "n": int(self.n),
            "hits": int(self.hits),
            "coverage": float(self.coverage),
            "ci_low": float(self.ci_low),
            "ci_high": float(self.ci_high),
            "verdict": self.verdict,
        }


def _level_stat(level: float, hit_flags: np.ndarray) -> CoverageLevelStat:
    flags = np.asarray(hit_flags, dtype=bool)
    n = int(flags.size)
    k = int(flags.sum())
    cov = k / n if n else float("nan")
    lo, hi = jeffreys_interval(k, n)
    if n and hi < level:
        verdict = "under"
    elif n and lo > level:
        verdict = "over"
    else:
        verdict = "ok"
    return CoverageLevelStat(
        level=float(level),
        n=n,
        hits=k,
        coverage=float(cov),
        ci_low=lo,
        ci_high=hi,
        verdict=verdict,
    )


def coverage_from_ranks(
    int_ranks: np.ndarray, L: int, levels: tuple[float, ...] = DEFAULT_LEVELS
) -> list[CoverageLevelStat]:
    """Empirical central-interval coverage read directly off SBC ranks.

    The truth lies inside the central ``level`` posterior interval exactly when
    its normalized rank ``u = (r + 0.5)/(L + 1)`` falls in
    ``[(1−level)/2, 1−(1−level)/2]`` — so an SBC run already contains every
    coverage number; this just states them in user language.
    """
    u = (np.asarray(int_ranks, dtype=float) + 0.5) / (float(L) + 1.0)
    out = []
    for level in levels:
        alpha = 1.0 - float(level)
        out.append(_level_stat(level, (u >= alpha / 2.0) & (u <= 1.0 - alpha / 2.0)))
    return out


def coverage_from_draws(
    truth: float,
    draws_by_sim: list[np.ndarray],
    levels: tuple[float, ...] = DEFAULT_LEVELS,
) -> list[CoverageLevelStat]:
    """Coverage of central equal-tailed percentile intervals across refits.

    ``draws_by_sim[i]`` is the posterior sample for one simulated dataset; the
    truth is fixed. Matches the framework's reported intervals
    (``compute_hdi_bounds`` is percentile-based).
    """
    t = float(truth)
    hits_by_level: dict[float, list[bool]] = {float(lv): [] for lv in levels}
    for draws in draws_by_sim:
        d = np.asarray(draws, dtype=float).ravel()
        d = d[np.isfinite(d)]
        if d.size < 4:
            continue
        for lv in levels:
            alpha = 1.0 - float(lv)
            lo, hi = np.percentile(d, [100.0 * alpha / 2.0, 100.0 * (1 - alpha / 2.0)])
            hits_by_level[float(lv)].append(bool(lo <= t <= hi))
    return [_level_stat(lv, np.asarray(hits_by_level[float(lv)])) for lv in levels]


def recovery_diagnosis(
    truth: float, post_means: np.ndarray, post_sds: np.ndarray
) -> dict[str, Any]:
    """Decompose a coverage failure into bias vs. interval width.

    With ``z_i = (mean_i − θ*)/sd_i`` across refits: under calibrated recovery
    the ``z_i`` are ≈ standard normal, so

    * ``bias_z = mean(z)·√n`` ≈ N(0,1) — large |bias_z| ⇒ the posterior
      *location* is systematically off (positive = estimates sit above truth);
    * ``z_spread = sd(z)`` ≈ 1 — significantly > 1 ⇒ intervals **too narrow**
      (overconfident), < 1 ⇒ too wide (conservative). Significance via
      ``(n−1)·s² ~ χ²(n−1)``.
    """
    m = np.asarray(post_means, dtype=float)
    s = np.asarray(post_sds, dtype=float)
    ok = np.isfinite(m) & np.isfinite(s) & (s > 0)
    z = (m[ok] - float(truth)) / s[ok]
    n = int(z.size)
    if n < 3:
        return {
            "n": n,
            "bias_z": float("nan"),
            "z_spread": float("nan"),
            "spread_p": float("nan"),
            "flags": [],
            "verdict": "insufficient sims",
        }
    bias_z = float(np.mean(z) * np.sqrt(n))
    spread = float(np.std(z, ddof=1))
    chi = (n - 1) * spread**2
    # two-sided p for s² vs 1 under χ²(n−1)
    p_hi = float(stats.chi2.sf(chi, n - 1))  # small ⇒ spread significantly > 1
    p_lo = float(stats.chi2.cdf(chi, n - 1))  # small ⇒ spread significantly < 1
    spread_p = float(min(1.0, 2.0 * min(p_hi, p_lo)))
    flags: list[str] = []
    if abs(bias_z) >= 2.5:
        flags.append("biased high" if bias_z > 0 else "biased low")
    if p_hi < 0.01 and spread > 1.0:
        flags.append("overconfident (intervals too narrow)")
    if p_lo < 0.01 and spread < 1.0:
        flags.append("conservative (intervals too wide)")
    verdict = flags[0] if flags else "ok"
    return {
        "n": n,
        "bias_z": bias_z,
        "z_spread": spread,
        "spread_p": spread_p,
        "flags": flags,
        "verdict": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RecoveryTargetStat:
    """Per-target recovery-coverage verdict (a parameter or an estimand)."""

    name: str
    kind: str  # "parameter" | "contribution"
    truth: float
    n_sims: int
    levels: list[CoverageLevelStat]
    bias_z: float
    z_spread: float
    spread_p: float
    rmse: float
    mean_post_sd: float
    verdict: str
    flags: list[str] = field(default_factory=list)

    def coverage_at(self, level: float) -> CoverageLevelStat | None:
        for st in self.levels:
            if abs(st.level - level) < 1e-9:
                return st
        return None

    def to_dashboard(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "truth": float(self.truth),
            "n_sims": int(self.n_sims),
            "levels": [st.to_dashboard() for st in self.levels],
            "bias_z": float(self.bias_z),
            "z_spread": float(self.z_spread),
            "spread_p": float(self.spread_p),
            "rmse": float(self.rmse),
            "mean_post_sd": float(self.mean_post_sd),
            "verdict": self.verdict,
            "flags": list(self.flags),
        }


@dataclass
class RecoveryCoverageResult:
    """Full recovery-coverage run: per-target stats + run metadata."""

    targets: list[RecoveryTargetStat]
    n_sims_requested: int
    n_sims_effective: int
    n_failed_fits: int
    truth_source: str
    sampler: str
    L: int
    seed: int
    levels: tuple[float, ...]
    elapsed_s: float = 0.0
    caveats: list[str] = field(default_factory=list)

    @property
    def headline_level(self) -> float:
        return 0.9 if 0.9 in self.levels else max(self.levels)

    @property
    def all_nominal(self) -> bool:
        """True when no target's headline-level interval demonstrably under-covers."""
        lv = self.headline_level
        return bool(self.targets) and all(
            (st := t.coverage_at(lv)) is None or st.verdict != "under"
            for t in self.targets
        )

    def worst(self) -> RecoveryTargetStat | None:
        lv = self.headline_level
        scored = [(t.coverage_at(lv), t) for t in self.targets]
        scored = [(c, t) for c, t in scored if c is not None and c.n > 0]
        return min(scored, key=lambda ct: ct[0].coverage)[1] if scored else None

    def summary(self) -> str:
        lv = self.headline_level
        verdict = "COVERAGE OK" if self.all_nominal else "UNDER-COVERAGE DETECTED"
        lines = [
            f"Recovery coverage ({verdict}) — {self.n_sims_effective}/"
            f"{self.n_sims_requested} refits at truth={self.truth_source}, "
            f"{self.sampler}:"
        ]
        ordered = sorted(
            self.targets,
            key=lambda t: (c.coverage if (c := t.coverage_at(lv)) else 1.0),
        )
        for t in ordered:
            c = t.coverage_at(lv)
            if c is None:
                continue
            flag = "ok" if c.verdict != "under" else f"⚠ {t.verdict}"
            lines.append(
                f"  - {t.name}: {c.coverage:.0%} of {lv:.0%} intervals cover "
                f"[{c.ci_low:.0%}–{c.ci_high:.0%}] ({flag})"
            )
        lines.extend(f"  ! {c}" for c in self.caveats)
        return "\n".join(lines)

    def to_dashboard(self) -> dict[str, Any]:
        return {
            "all_nominal": self.all_nominal,
            "headline_level": float(self.headline_level),
            "n_sims_requested": int(self.n_sims_requested),
            "n_sims_effective": int(self.n_sims_effective),
            "n_failed_fits": int(self.n_failed_fits),
            "truth_source": self.truth_source,
            "sampler": self.sampler,
            "L": int(self.L),
            "seed": int(self.seed),
            "levels": [float(v) for v in self.levels],
            "elapsed_s": float(self.elapsed_s),
            "caveats": list(self.caveats),
            "targets": [t.to_dashboard() for t in self.targets],
        }


def build_recovery_result(
    truths: Mapping[str, float],
    draws_by_target: Mapping[str, list[np.ndarray]],
    *,
    kinds: Mapping[str, str] | None = None,
    levels: tuple[float, ...] = DEFAULT_LEVELS,
    truth_source: str = "fixed",
    sampler: str = "generic",
    L: int = 0,
    seed: int = 0,
    n_sims_requested: int = 0,
    n_failed_fits: int = 0,
    elapsed_s: float = 0.0,
    extra_caveats: tuple[str, ...] = (),
) -> RecoveryCoverageResult:
    """Assemble a :class:`RecoveryCoverageResult` from per-target refit draws.

    Pure (no PyMC) — the fast unit tests drive this directly with analytic
    posteriors.
    """
    targets: list[RecoveryTargetStat] = []
    n_eff = 0
    for name, truth in truths.items():
        sims = [
            np.asarray(d, dtype=float).ravel() for d in draws_by_target.get(name, [])
        ]
        sims = [d[np.isfinite(d)] for d in sims]
        sims = [d for d in sims if d.size >= 4]
        if not sims:
            continue
        n_eff = max(n_eff, len(sims))
        level_stats = coverage_from_draws(truth, sims, levels)
        means = np.array([float(np.mean(d)) for d in sims])
        sds = np.array([float(np.std(d, ddof=1)) for d in sims])
        diag = recovery_diagnosis(truth, means, sds)
        targets.append(
            RecoveryTargetStat(
                name=name,
                kind=(kinds or {}).get(name, "parameter"),
                truth=float(truth),
                n_sims=len(sims),
                levels=level_stats,
                bias_z=float(diag["bias_z"]),
                z_spread=float(diag["z_spread"]),
                spread_p=float(diag["spread_p"]),
                rmse=float(np.sqrt(np.mean((means - float(truth)) ** 2))),
                mean_post_sd=float(np.mean(sds)),
                verdict=str(diag["verdict"]),
                flags=list(diag["flags"]),
            )
        )
    caveats = list(extra_caveats)
    if 0 < n_eff < 30:
        caveats.append(
            f"Only {n_eff} effective simulations — the Monte-Carlo error bars on "
            "coverage are wide (e.g. true 90% coverage can produce 70–100% "
            "empirically). Increase n_sims for a sharper read; trust the "
            "binomial intervals, not the point estimates."
        )
    if n_failed_fits:
        caveats.append(
            f"{n_failed_fits} refit(s) failed and were dropped; treat the verdict "
            "cautiously if this is a large fraction."
        )
    if sampler in ("advi", "fullrank_advi"):
        caveats.append(
            "Refits used variational inference (ADVI), whose uncertainty is "
            "systematically too narrow — under-coverage here may reflect ADVI, "
            "not your model. Re-run with the numpyro/NUTS sampler to judge the "
            "production posterior."
        )
    return RecoveryCoverageResult(
        targets=targets,
        n_sims_requested=int(n_sims_requested),
        n_sims_effective=int(n_eff),
        n_failed_fits=int(n_failed_fits),
        truth_source=truth_source,
        sampler=sampler,
        L=int(L),
        seed=int(seed),
        levels=tuple(float(v) for v in levels),
        elapsed_s=float(elapsed_s),
        caveats=caveats,
    )


def failure_mode_guide() -> str:
    """Markdown table of the ways nominal coverage fails and what to do.

    Kept next to the tool so every surface (chat, Validation tab, docs) tells
    one story. The long-form version is ``technical-docs/coverage-diagnostics.md``.
    """
    return (
        "**When coverage fails — the usual suspects**\n\n"
        "| Cause | Signature | Detected by | Fix |\n"
        "|---|---|---|---|\n"
        "| Approximate fit (MAP/ADVI/Pathfinder) | intervals far too narrow; "
        "`results.approximate=True` | this check (z_spread ≫ 1) | re-fit with "
        "NUTS before trusting intervals |\n"
        "| Sampler not converged | high R-hat, low ESS, divergences | "
        "convergence diagnostics | more tune/draws, reparameterize |\n"
        "| Priors too tight (prior–data conflict) | posterior barely moves; "
        "biased toward the prior | this check at a fixed truth (bias flags); "
        "parameter-learning diagnostic | widen priors, justify with evidence |\n"
        "| Inference engine miscalibrated | non-uniform SBC ranks | SBC "
        "(`run_calibration_check`) | fix sampler settings / model geometry |\n"
        "| Model misspecification (wrong adstock/saturation, missing "
        "seasonality, time-varying effects) | posterior concentrates on a "
        "pseudo-truth ≠ truth | NOT visible when simulating from the model — "
        "needs external truth (synthetic scenarios) or PPC/residual checks | "
        "richer structure, spec-curve sensitivity |\n"
        "| Unobserved confounding / demand-chasing spend | tight intervals "
        "around a biased ROI | refutation suite, endogeneity check, "
        "experiments | calibrate with lift tests, add confounder controls |\n"
        "| Estimand mismatch | comparing an interval to a *different* "
        "quantity's truth (contribution ROI vs counterfactual ROI are "
        "different numbers) | estimand registry / definitions | compare "
        "like-for-like |\n"
        "| Too few simulations | coverage estimate itself is noisy | binomial "
        "CI on the coverage | more sims; read the CI, not the point |\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# The recovery loop (PyMC) — simulate at a fixed θ*, refit, measure coverage
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_truth(
    model: Any, truth: str | Mapping[str, Any], seed: int
) -> tuple[dict[str, np.ndarray], str]:
    """Resolve θ* for every free RV (natural/support space).

    ``"posterior_mean"`` needs a fitted model; ``"prior"`` takes one seeded
    prior draw; a mapping overrides individual parameters on top of whichever
    base is available.
    """
    free_names = [rv.name for rv in model.model.free_RVs]

    def _posterior_base() -> dict[str, np.ndarray] | None:
        trace = getattr(model, "_trace", None)
        if trace is None or not hasattr(trace, "posterior"):
            return None
        post = trace.posterior
        base = {}
        for n in free_names:
            if n not in post:
                return None
            base[n] = np.asarray(post[n].mean(dim=("chain", "draw")).values)
        return base

    def _prior_base() -> dict[str, np.ndarray]:
        pri = model.sample_prior_predictive(samples=1, random_seed=seed)
        return {n: np.asarray(pri.prior[n].values)[0, 0] for n in free_names}

    if isinstance(truth, str) and truth == "posterior_mean":
        base = _posterior_base()
        if base is None:
            raise ValueError(
                "truth='posterior_mean' needs a fitted model (no posterior trace "
                "found). Fit first, or use truth='prior'."
            )
        return base, "posterior_mean"
    if isinstance(truth, str) and truth == "prior":
        return _prior_base(), "prior"
    if isinstance(truth, Mapping):
        base = _posterior_base()
        source = "posterior_mean+overrides"
        if base is None:
            base = _prior_base()
            source = "prior+overrides"
        unknown = [k for k in truth if k not in base]
        if unknown:
            raise ValueError(
                f"Unknown parameter(s) in truth overrides: {unknown}. "
                f"Free parameters are: {sorted(base)}"
            )
        for k, v in truth.items():
            base[k] = np.broadcast_to(
                np.asarray(v, dtype=float), np.shape(base[k])
            ).copy()
        return base, source
    raise ValueError(
        f"truth must be 'posterior_mean', 'prior', or a dict of overrides; got {truth!r}"
    )


def _contribution_truths(sim_prior: Any, channel_names: list[str]) -> dict[str, float]:
    """Per-channel total-contribution truth at θ* from the do-graph's constant
    ``channel_contributions`` deterministic (standardized scale; coverage is
    scale-invariant so no unit conversion is needed)."""
    if "channel_contributions" not in sim_prior:
        return {}
    arr = np.asarray(sim_prior["channel_contributions"].values)[0, 0]
    if arr.ndim < 1 or arr.shape[-1] != len(channel_names):
        return {}
    totals = arr.reshape(-1, arr.shape[-1]).sum(axis=0)
    return {
        f"contribution_{ch}": float(totals[i]) for i, ch in enumerate(channel_names)
    }


def run_recovery_coverage(
    model: Any,
    *,
    truth: str | Mapping[str, Any] = "posterior_mean",
    n_sims: int = 24,
    levels: tuple[float, ...] = DEFAULT_LEVELS,
    sampler: str = "numpyro",
    L: int = 200,
    tune: int = 200,
    chains: int = 2,
    seed: int = 0,
    params: list[str] | None = None,
    include_contributions: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> RecoveryCoverageResult:
    """Fixed-truth recovery coverage for a :class:`BayesianMMM`.

    Fixes every free parameter at θ* (``pm.do``), simulates ``n_sims`` datasets
    from the likelihood at θ*, refits the ORIGINAL model on each
    (``pm.observe`` on the same graph — θ* and every posterior share one fixed
    standardized scale), and measures how often each central interval contains
    the truth — for scalar free parameters and per-channel total contributions.

    EXPENSIVE: one refit per simulation, like SBC. Use a background job for
    thorough runs; ``n_sims < 30`` only flags gross failures (the binomial CI
    is reported either way).

    Answers: "IF the world matched my fitted model exactly, would my reported
    intervals cover?" Under-coverage here is a mechanical problem (approximate
    inference, sampler, priors); it deliberately cannot see real-world
    misspecification — see the module docstring.
    """
    import pymc as pm

    from ..utils.arviz_compat import sample_prior_predictive
    from .sbc import _sample_swapped, _scalar_param_names, _thin_to

    t0 = time.perf_counter()
    graph = model.model
    obs_name = list(graph.observed_RVs)[0].name

    theta, truth_source = _resolve_truth(model, truth, seed)

    do_graph = pm.do(graph, {k: v for k, v in theta.items()})
    with do_graph:
        sim = sample_prior_predictive(samples=int(n_sims), random_seed=seed)
    y_sims = np.asarray(sim.prior_predictive[obs_name].values)  # (1, n_sims, ...)

    scalar_params = [p for p in (params or _scalar_param_names(model)) if p in theta]
    truths: dict[str, float] = {}
    kinds: dict[str, str] = {}
    for p in scalar_params:
        val = np.asarray(theta[p], dtype=float).ravel()
        if val.size != 1:
            continue  # vector RV slipped through the scalar filter — skip
        truths[p] = float(val[0])
        kinds[p] = "parameter"

    channel_names = list(getattr(model, "channel_names", []) or [])
    contrib_truths: dict[str, float] = {}
    if include_contributions and channel_names:
        contrib_truths = _contribution_truths(sim.prior, channel_names)
        for k in contrib_truths:
            kinds[k] = "contribution"
    truths.update(contrib_truths)
    if not truths:
        raise ValueError("No scalar parameters or contributions found to check.")

    draws_by_target: dict[str, list[np.ndarray]] = {k: [] for k in truths}
    n_ch = len(channel_names)
    failed = 0
    for i in range(int(n_sims)):
        y_sim = np.asarray(y_sims[0, i], dtype=float)
        try:
            swapped = pm.observe(graph, {obs_name: y_sim})
            idata = _sample_swapped(
                swapped,
                sampler=sampler,
                L=int(L),
                tune=int(tune),
                chains=int(chains),
                seed=int(seed) + 7919 * (i + 1),
            )
        except Exception as e:  # noqa: BLE001
            failed += 1
            if failed == 1:
                warnings.warn(f"recovery refit failed (sim {i}): {e}", stacklevel=2)
            if progress:
                progress(i + 1, int(n_sims))
            continue
        post = idata.posterior
        for p in scalar_params:
            if p in truths and p in post:
                draws_by_target[p].append(
                    _thin_to(np.asarray(post[p].values, dtype=float).ravel(), int(L))
                )
        if contrib_truths and "channel_contributions" in post:
            cc = np.asarray(post["channel_contributions"].values, dtype=float)
            flat = cc.reshape(-1, *cc.shape[2:])  # (chain·draw, ..., channel)
            per_draw = flat.reshape(flat.shape[0], -1, n_ch).sum(axis=1)
            for j, ch in enumerate(channel_names):
                key = f"contribution_{ch}"
                if key in truths:
                    draws_by_target[key].append(_thin_to(per_draw[:, j], int(L)))
        if progress:
            progress(i + 1, int(n_sims))

    extra: list[str] = []
    if truth_source.startswith("posterior_mean"):
        extra.append(
            "Truth = the fitted posterior mean: this checks the machinery at a "
            "well-supported point. It cannot detect real-world misspecification "
            "(data are simulated FROM the model) — for that, fit on synthetic "
            "worlds with an external answer key or run the refutation suite."
        )
    if getattr(model, "_trace", None) is None and truth_source == "prior":
        extra.append(
            "Truth drawn from the prior: if the prior is wide, some θ* land in "
            "regions the data cannot inform, which legitimately widens intervals."
        )

    return build_recovery_result(
        truths,
        draws_by_target,
        kinds=kinds,
        levels=levels,
        truth_source=truth_source,
        sampler=sampler,
        L=int(L),
        seed=int(seed),
        n_sims_requested=int(n_sims),
        n_failed_fits=failed,
        elapsed_s=time.perf_counter() - t0,
        extra_caveats=tuple(extra),
    )


__all__ = [
    "DEFAULT_LEVELS",
    "CoverageLevelStat",
    "RecoveryTargetStat",
    "RecoveryCoverageResult",
    "jeffreys_interval",
    "coverage_from_ranks",
    "coverage_from_draws",
    "recovery_diagnosis",
    "build_recovery_result",
    "failure_mode_guide",
    "run_recovery_coverage",
]
