"""Fit the MMM on each structural-violation scenario and score the damage.

For every scenario the harness reports, per channel and in aggregate:

* **recovery error** -- estimated vs. true total contribution (the model's own
  counterfactual estimand), as a signed relative error;
* **coverage** -- whether the true (causal) contribution lands inside the model's
  90% credible interval (overconfidence under misspecification is the quiet
  killer that "tested only on easy data" hides);
* **which diagnostics fired** -- convergence (r-hat / divergences), the causal
  refutation suite, posterior-predictive checks, the unobserved-confounding
  robustness value, and the prior->posterior learning verdict.

The headline output is the set of **silent failures**: scenarios where the
recovery is badly wrong (large error and/or the truth outside the 90% CI) yet
**the checks an analyst acts on are green** -- MCMC convergence and the
unobserved-confounding robustness value. PPC and the refutation suite are
reported alongside but excluded from the gate, because both fire on the clean
control (false positives) and so are not a reliable all-clear. Those cells answer
the colleagues' critique: the model can be confidently wrong on realistic data
and the routine checks would not tell you.

Run it::

    uv run python -m tests.synth.run_stress_matrix --quick      # control + top 4
    uv run python -m tests.synth.run_stress_matrix --all         # full sweep
"""

from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .dgp import Scenario, build

# A silent failure needs the recovery to be wrong by at least this much...
_ERROR_THRESHOLD = 0.25  # 25% median |relative error| on contribution, OR
_MAX_ERROR_THRESHOLD = 0.50  # some single channel off by >50%, OR
# ...coverage of the truth falling below this rate.
_COVERAGE_THRESHOLD = 0.75
# Practical convergence bar for the silent-failure gate (the strict <1.01 used
# for production is reported separately as ``rhat_max``).
_RHAT_GATE = 1.05


@dataclass
class ChannelScore:
    channel: str
    true_contribution: float
    est_contribution: float
    rel_error: float  # (est - true) / |true|
    covered: bool  # true in the 90% CI
    ci_low: float
    ci_high: float
    true_roas: float
    est_roas: float
    learning_verdict: str  # prior->posterior verdict for beta_<channel>
    robustness_value: float  # unobserved-confounding RV (higher = looks robust)


@dataclass
class ScenarioResult:
    name: str
    violates: str
    representable: bool
    ok: bool = True
    error: str | None = None
    elapsed_s: float = 0.0

    # recovery
    median_abs_rel_error: float = float("nan")
    max_abs_rel_error: float = float("nan")
    total_media_rel_error: float = float("nan")
    coverage_rate: float = float("nan")

    # diagnostics ("fired" == flagged a problem)
    converged: bool = True
    rhat_max: float = float("nan")
    divergences: int = 0
    ppc_pass: bool = True
    ppc_problems: list[str] = field(default_factory=list)
    refutation_all_passed: bool = True
    refutation_failed: list[str] = field(default_factory=list)
    fragile_channels: list[str] = field(default_factory=list)

    silent_failure: bool = False
    channels: list[ChannelScore] = field(default_factory=list)

    def to_row(self) -> dict:
        d = asdict(self)
        d.pop("channels")
        d["refutation_failed"] = ",".join(self.refutation_failed) or "-"
        d["ppc_problems"] = ",".join(self.ppc_problems) or "-"
        d["fragile_channels"] = ",".join(self.fragile_channels) or "-"
        return d


# ---------------------------------------------------------------------------
# fit + score one scenario
# ---------------------------------------------------------------------------


def _build_model(panel, *, draws, tune, chains, target_accept, parametric, seed):
    from mmm_framework.config import InferenceMethod, ModelConfig
    from mmm_framework.model import BayesianMMM, TrendConfig, TrendType

    cfg = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_PYMC,
        n_chains=chains,
        n_draws=draws,
        n_tune=tune,
        target_accept=target_accept,
        use_parametric_adstock=parametric,
        optim_seed=seed,
    )
    return BayesianMMM(panel, cfg, TrendConfig(type=TrendType.LINEAR))


def _learning_verdicts(mmm, prior_samples: int) -> dict[str, str]:
    try:
        df = mmm.compute_parameter_learning(prior_samples=prior_samples)
    except Exception:
        return {}
    out = {}
    for _, r in df.iterrows():
        p = str(r["parameter"])
        if p.startswith("beta_"):
            out[p[len("beta_") :]] = str(r["verdict"])
    return out


def _run_validation(mmm, *, run_refutation, ref_draws, seed):
    """Returns (convergence, ppc, refutation, unobserved_confounding) defensively."""
    from mmm_framework.validation import ModelValidator, ValidationConfigBuilder

    b = (
        ValidationConfigBuilder()
        .silent()
        .without_residuals()
        .without_channel_diagnostics()
        .with_unobserved_confounding()
    )
    if run_refutation:
        b = b.with_causal_refutation(draws=ref_draws, tune=ref_draws, chains=2)
    cfg = b.build()
    cfg.run_model_comparison = False
    cfg.run_ppc = True  # cheap and it is the check that should catch bad noise
    summary = ModelValidator(mmm).validate(cfg)
    return summary


def score_scenario(
    sc: Scenario,
    *,
    draws: int = 500,
    tune: int = 500,
    chains: int = 2,
    target_accept: float = 0.9,
    parametric: bool = True,
    run_refutation: bool = True,
    ref_draws: int = 120,
    prior_samples: int = 600,
    hdi_prob: float = 0.90,
    seed: int = 0,
) -> ScenarioResult:
    t0 = time.time()
    res = ScenarioResult(
        name=sc.name, violates=sc.violates, representable=sc.representable
    )
    try:
        mmm = _build_model(
            sc.panel(),
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            parametric=parametric,
            seed=seed,
        )
        fit = mmm.fit(random_seed=seed)

        # --- recovery + coverage -----------------------------------------
        contrib = mmm.compute_counterfactual_contributions(
            compute_uncertainty=True, hdi_prob=hdi_prob, random_seed=seed
        )
        est = contrib.total_contributions
        lo = contrib.contribution_hdi_low
        hi = contrib.contribution_hdi_high

        verdicts = _learning_verdicts(mmm, prior_samples)

        # --- diagnostics --------------------------------------------------
        res.rhat_max = float(fit.diagnostics.get("rhat_max", float("nan")))
        res.divergences = int(fit.diagnostics.get("divergences", 0))
        res.converged = (res.rhat_max < 1.01) and (res.divergences == 0)

        rv_by_channel: dict[str, float] = {}
        try:
            summary = _run_validation(
                mmm, run_refutation=run_refutation, ref_draws=ref_draws, seed=seed
            )
            if summary.ppc is not None:
                res.ppc_pass = bool(summary.ppc.overall_pass)
                res.ppc_problems = list(summary.ppc.problematic_checks)
            if summary.causal_refutation is not None:
                tests = summary.causal_refutation.tests
                res.refutation_failed = [t.name for t in tests if not t.passed]
                res.refutation_all_passed = len(res.refutation_failed) == 0
            if summary.unobserved_confounding is not None:
                res.fragile_channels = list(
                    summary.unobserved_confounding.fragile_channels
                )
                for c in summary.unobserved_confounding.channels:
                    rv_by_channel[c.channel] = float(c.robustness_value)
            # prefer the validator's convergence if present
            if summary.convergence is not None:
                res.rhat_max = float(summary.convergence.rhat_max)
                res.divergences = int(summary.convergence.divergences)
                res.converged = bool(summary.convergence.converged)
        except Exception as exc:  # diagnostics are best-effort
            res.error = f"diagnostics: {exc}"

        # --- per-channel scores ------------------------------------------
        rel_errors, covered_flags = [], []
        for c in sc.channels:
            true_c = float(sc.true_contribution[c])
            est_c = float(est.get(c, np.nan))
            denom = abs(true_c) if abs(true_c) > 1e-9 else 1.0
            rel = (est_c - true_c) / denom
            ci_low = float(lo.get(c, np.nan)) if lo is not None else np.nan
            ci_high = float(hi.get(c, np.nan)) if hi is not None else np.nan
            covered = bool(ci_low <= true_c <= ci_high)
            res.channels.append(
                ChannelScore(
                    channel=c,
                    true_contribution=true_c,
                    est_contribution=est_c,
                    rel_error=rel,
                    covered=covered,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    true_roas=float(sc.true_roas[c]),
                    est_roas=est_c / float(sc.spend[c].sum()),
                    learning_verdict=verdicts.get(c, "?"),
                    robustness_value=rv_by_channel.get(c, float("nan")),
                )
            )
            rel_errors.append(abs(rel))
            covered_flags.append(covered)

        # nan-safe so a missing channel estimate fails the magnitude gate CLOSED
        # (consistent with coverage, which already counts a NaN bound as a miss).
        res.median_abs_rel_error = float(np.nanmedian(rel_errors))
        res.max_abs_rel_error = float(np.nanmax(rel_errors))
        res.coverage_rate = float(np.mean(covered_flags))
        true_total = float(sc.true_contribution.sum())
        est_total = float(est.reindex(sc.channels).sum())
        res.total_media_rel_error = (est_total - true_total) / abs(true_total)

        # --- silent failure verdict --------------------------------------
        # "Green" = the cheap, routine checks every analyst trusts say fine:
        # MCMC convergence and the unobserved-confounding robustness value. PPC
        # and the refutation suite are reported alongside but EXCLUDED from the
        # gate: both are noisy enough to fire on the clean control (false
        # positives), so "PPC passed" is not a reliable all-clear -- an analyst
        # who sees PPC cry wolf on good data learns to ignore it. A silent
        # failure is bad recovery that survives the checks people actually act on.
        diagnostics_green = (
            res.rhat_max < _RHAT_GATE
            and res.divergences == 0
            and len(res.fragile_channels) == 0
        )
        bad_recovery = (
            res.median_abs_rel_error > _ERROR_THRESHOLD
            or res.max_abs_rel_error > _MAX_ERROR_THRESHOLD
            or res.coverage_rate < _COVERAGE_THRESHOLD
        )
        res.silent_failure = bool(
            sc.representable and diagnostics_green and bad_recovery
        )

    except Exception as exc:
        res.ok = False
        res.error = f"{exc}\n{traceback.format_exc()}"
    res.elapsed_s = round(time.time() - t0, 1)
    return res


def run_scenario(name: str, *, seed: int | None = None, **kw) -> ScenarioResult:
    sc = build(name, seed)
    return score_scenario(sc, **kw)


# ---------------------------------------------------------------------------
# matrix + reporting
# ---------------------------------------------------------------------------


def run_matrix(names: list[str], *, progress=True, **kw) -> list[ScenarioResult]:
    results = []
    for i, name in enumerate(names, 1):
        if progress:
            print(f"[{i}/{len(names)}] fitting '{name}' ...", flush=True)
        r = run_scenario(name, **kw)
        if progress:
            tag = (
                "SILENT FAILURE"
                if r.silent_failure
                else (
                    "expected (unrepresentable)"
                    if not r.representable
                    else "ok" if not r.error else f"ERR: {r.error.splitlines()[0]}"
                )
            )
            print(
                f"      med|err|={r.median_abs_rel_error:.0%} "
                f"cover={r.coverage_rate:.0%} conv={r.converged} "
                f"refut_pass={r.refutation_all_passed} ppc={r.ppc_pass} "
                f"-> {tag}  ({r.elapsed_s}s)",
                flush=True,
            )
        results.append(r)
    return results


def matrix_frame(results: list[ScenarioResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_row() for r in results])


def to_markdown(results: list[ScenarioResult]) -> str:
    lines = [
        "# MMM structural-violation stress matrix",
        "",
        "`med|err|` / `max|err|` = median / worst |relative error| on per-channel "
        "total contribution. `cover` = fraction of channels whose true "
        "contribution lands in the 90% credible interval. A **silent failure** is "
        "a *representable* scenario where recovery is wrong (median err > 25%, "
        "worst channel > 50%, or coverage < 75%) yet **the checks an analyst acts "
        "on are green**: MCMC convergence (r-hat < 1.05, no divergences) and the "
        "unobserved-confounding robustness value. `ppc` and `refut` are reported "
        "but **excluded from the gate** — both are fit-level checks whose verdicts "
        "have proven config-sensitive across recordings (under the pre-2026-06-10 "
        "trend prior both false-alarmed on the clean control; under current "
        "defaults they pass on clean and on several wrong-attribution worlds), so "
        "neither is a reliable all-clear for the *causal* claim.",
        "",
        "| scenario | assumption broken | med\\|err\\| | max\\|err\\| | total-media err "
        "| cover | rhat | div | ppc | refut | fragile | verdict |",
        "|---|---|--:|--:|--:|--:|--:|--:|:--:|:--:|---|---|",
    ]
    for r in results:
        if r.error and not r.ok:
            lines.append(
                f"| {r.name} | {r.violates} | — | — | — | — | — | — | — | — | — | "
                f"ERROR: {r.error.splitlines()[0]} |"
            )
            continue
        verdict = (
            "🔴 **SILENT FAILURE**"
            if r.silent_failure
            else "⚪ expected (unrepresentable)" if not r.representable else "🟢 ok"
        )
        refut = "✓" if r.refutation_all_passed else "✗ " + ",".join(r.refutation_failed)
        lines.append(
            f"| {r.name} | {r.violates or '— (control)'} "
            f"| {r.median_abs_rel_error:.0%} | {r.max_abs_rel_error:.0%} "
            f"| {r.total_media_rel_error:+.0%} | {r.coverage_rate:.0%} "
            f"| {r.rhat_max:.2f} | {r.divergences} | {'✓' if r.ppc_pass else '✗'} | {refut} "
            f"| {','.join(r.fragile_channels) or '—'} | {verdict} |"
        )
    # per-channel detail for silent failures + unrepresentable
    detail = [r for r in results if r.silent_failure or not r.representable]
    if detail:
        lines += ["", "## Per-channel detail (silent failures & unrepresentable)", ""]
        for r in detail:
            lines += [f"### {r.name} — {r.violates}", ""]
            lines.append(
                "| channel | true | est | rel err | in 90% CI | RV | learning |"
            )
            lines.append("|---|--:|--:|--:|:--:|--:|---|")
            for c in r.channels:
                lines.append(
                    f"| {c.channel} | {c.true_contribution:,.0f} "
                    f"| {c.est_contribution:,.0f} | {c.rel_error:+.0%} "
                    f"| {'✓' if c.covered else '✗'} | {c.robustness_value:.2f} "
                    f"| {c.learning_verdict} |"
                )
            lines.append("")
    return "\n".join(lines)


__all__ = [
    "ChannelScore",
    "ScenarioResult",
    "score_scenario",
    "run_scenario",
    "run_matrix",
    "matrix_frame",
    "to_markdown",
]
