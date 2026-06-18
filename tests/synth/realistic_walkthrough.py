"""An iterative modeling walkthrough on the realistic many-factor world.

Follows the framework's documented 9-step scientific workflow
(``docs/scientific-workflow-demo.html``) and treats modeling as iterative:

* **v1 (naive)** — dump every candidate factor in as a plain control, default
  config. The confounders get the regularizing precision prior (shrunk) and the
  ``brand_awareness`` mediator is silently included (post-treatment): media ROI
  is biased and the weak channels are prior-dominated.
* **v2 (causal structure)** — apply the DAG: mark the confounders so they get the
  wide, un-shrunk prior; drop the mediator (the model refuses a control marked
  MEDIATOR); keep the precision controls (the σ=0.5 prior soft-selects the
  irrelevant ones toward 0). Confounding + post-treatment bias fall away.
* **v3 (calibration)** — the prior→posterior learning diagnostic flags the
  channels observational data cannot identify; anchor exactly those with a
  geo-lift experiment (``ExperimentCalibrator``) and refit.

Each step is scored against the world's *known* total causal effect. Results are
written to ``results/walkthrough.json``.

    uv run python -m tests.synth.realistic_walkthrough --quick   # cheap arc check
    uv run python -m tests.synth.realistic_walkthrough           # publication run
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import dgp

RESULTS = Path("tests/synth/results")


# ---------------------------------------------------------------------------
# model construction + scoring against the known truth
# ---------------------------------------------------------------------------


def _build(scenario, *, draws, tune, chains, target_accept, numpyro, seed):
    from mmm_framework.config import InferenceMethod, ModelConfig
    from mmm_framework.model import BayesianMMM, TrendConfig, TrendType

    cfg = ModelConfig(
        inference_method=(
            InferenceMethod.BAYESIAN_NUMPYRO
            if numpyro
            else InferenceMethod.BAYESIAN_PYMC
        ),
        n_chains=chains,
        n_draws=draws,
        n_tune=tune,
        target_accept=target_accept,
        use_parametric_adstock=True,  # geometric kernel — in the data's family
        optim_seed=seed,
    )
    return BayesianMMM(scenario.panel(), cfg, TrendConfig(type=TrendType.LINEAR))


def _variant(sc, *, drop=(), roles=None):
    """A copy of the scenario with a control subset and (optional) causal roles."""
    sub = sc.controls.drop(columns=list(drop)) if drop else sc.controls
    return replace(sc, controls=sub, control_roles=roles)


def _learning_verdicts(mmm, prior_samples):
    try:
        df = mmm.compute_parameter_learning(prior_samples=prior_samples)
    except Exception:
        return {}
    out = {}
    for _, r in df.iterrows():
        p = str(r["parameter"])
        if p.startswith("beta_") and not p.startswith("beta_controls"):
            out[p[len("beta_") :]] = str(r["verdict"])
    return out


def _control_means(mmm, fit):
    """Posterior-mean standardized coefficient per control (to show shrinkage)."""
    post = fit.trace.posterior
    if "beta_controls" not in post:
        return {}
    arr = post["beta_controls"].mean(dim=["chain", "draw"]).values
    return {name: float(arr[i]) for i, name in enumerate(mmm.control_names)}


def score(sc, mmm, fit, *, prior_samples, hdi=0.90, seed=0):
    contrib = mmm.compute_counterfactual_contributions(
        compute_uncertainty=True, hdi_prob=hdi, random_seed=seed
    )
    est, lo, hi = (
        contrib.total_contributions,
        contrib.contribution_hdi_low,
        contrib.contribution_hdi_high,
    )
    verdicts = _learning_verdicts(mmm, prior_samples)
    post = fit.trace.posterior

    def _beta(c):
        v = f"beta_{c}"
        if v in post:
            a = post[v].values
            return float(a.mean()), float(a.std())
        return float("nan"), float("nan")

    weak = set(sc.notes["weak_channels"])
    channels = []
    for c in sc.channels:
        true_c = float(sc.true_contribution[c])
        est_c = float(est.get(c, np.nan))
        b_mean, b_sd = _beta(c)
        lo_c = float(lo.get(c, np.nan)) if lo is not None else np.nan
        hi_c = float(hi.get(c, np.nan)) if hi is not None else np.nan
        channels.append(
            {
                "channel": c,
                "true": true_c,
                "est": est_c,
                "ci_low": lo_c,
                "ci_high": hi_c,
                "covered": bool(lo_c <= true_c <= hi_c),
                "rel_error": (
                    (est_c - true_c) / abs(true_c) if abs(true_c) > 1 else float("nan")
                ),
                # relative width of the 90% contribution CI: >~1.5 means the
                # effect is essentially unidentified from observational data.
                "rel_ci_width": (
                    (hi_c - lo_c) / abs(est_c) if abs(est_c) > 1 else float("nan")
                ),
                "verdict": verdicts.get(c, "?"),
                "beta_mean": b_mean,
                "beta_sd": b_sd,
                "weak": c in weak,
            }
        )
    strong = [c for c in channels if not c["weak"]]
    rec = {
        "channels": channels,
        "median_abs_rel_error_strong": float(
            np.median([abs(c["rel_error"]) for c in strong])
        ),
        "coverage_strong": float(np.mean([c["covered"] for c in strong])),
        "rhat_max": float(fit.diagnostics.get("rhat_max", float("nan"))),
        "divergences": int(fit.diagnostics.get("divergences", 0)),
        "control_means": _control_means(mmm, fit),
    }
    return rec


def _summary_line(tag, rec):
    weak_v = {c["channel"]: c["verdict"] for c in rec["channels"] if c["weak"]}
    return (
        f"[{tag}] strong med|err|={rec['median_abs_rel_error_strong']:.0%} "
        f"cover={rec['coverage_strong']:.0%} rhat={rec['rhat_max']:.3f} "
        f"div={rec['divergences']} weak_verdicts={weak_v}"
    )


# ---------------------------------------------------------------------------
# the walkthrough
# ---------------------------------------------------------------------------


def run(*, draws, tune, chains, prior_samples, refutation, numpyro, seed=0):
    from mmm_framework.config import CausalControlRole

    sc = dgp.make_realistic(seed=seed)
    out = {
        "config": {"draws": draws, "chains": chains, "numpyro": numpyro},
        "steps": {},
    }
    out["truth"] = {c: float(sc.true_contribution[c]) for c in sc.channels}
    out["roles"] = sc.notes["roles"]
    t0 = time.time()

    # --- Step 4: prior predictive check (on the v1 model, before fitting) ---
    print("Step 4 — prior predictive check ...", flush=True)
    m_pp = _build(
        sc,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=0.9,
        numpyro=numpyro,
        seed=seed,
    )
    prior = m_pp.get_prior(samples=400, random_seed=seed)
    pp = prior.prior_predictive["y_obs"].values.reshape(-1) * m_pp.y_std + m_pp.y_mean
    obs = sc.y.to_numpy()
    out["steps"]["prior_predictive"] = {
        "prior_kpi_p5": float(np.percentile(pp, 5)),
        "prior_kpi_p95": float(np.percentile(pp, 95)),
        "observed_kpi_min": float(obs.min()),
        "observed_kpi_max": float(obs.max()),
        "plausible": bool(np.percentile(pp, 5) < obs.mean() < np.percentile(pp, 95)),
    }
    print(
        f"  prior 90% KPI [{np.percentile(pp,5):.0f},{np.percentile(pp,95):.0f}] "
        f"vs observed [{obs.min():.0f},{obs.max():.0f}]",
        flush=True,
    )

    # --- v1: naive — no DAG yet. Include the "obvious" controls and the
    # tempting brand-awareness mediator, but OMIT the demand confounders: their
    # back-door is invisible until you draw the causal story (step 2). ---------
    print("\nv1 (naive, no DAG) — fit ...", flush=True)
    v1 = _variant(sc, drop=tuple(sc.notes["confounders"]))
    m1 = _build(
        v1,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=0.9,
        numpyro=numpyro,
        seed=seed,
    )
    f1 = m1.fit(random_seed=seed)
    r1 = score(v1, m1, f1, prior_samples=prior_samples, seed=seed)
    out["steps"]["v1_naive"] = r1
    print("  " + _summary_line("v1", r1), flush=True)

    # --- v2: causal structure — confounders wide, drop mediator -------------
    print("\nv2 (causal structure) — fit ...", flush=True)
    roles = {c: CausalControlRole.CONFOUNDER for c in sc.notes["confounders"]}
    v2 = _variant(sc, drop=(sc.notes["mediator"],), roles=roles)
    ta = 0.95  # wide-prior confounders × collinearity can need a higher target
    m2 = _build(
        v2,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=ta,
        numpyro=numpyro,
        seed=seed,
    )
    f2 = m2.fit(random_seed=seed)
    if f2.diagnostics.get("divergences", 0) > 0 and ta < 0.99:
        print(
            f"  {f2.diagnostics['divergences']} divergences — step 6 iteration: "
            "raising target_accept to 0.99 and refitting",
            flush=True,
        )
        m2 = _build(
            v2,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.99,
            numpyro=numpyro,
            seed=seed,
        )
        f2 = m2.fit(random_seed=seed)
    r2 = score(v2, m2, f2, prior_samples=prior_samples, seed=seed)
    out["steps"]["v2_causal"] = r2
    print("  " + _summary_line("v2", r2), flush=True)

    # --- v3: calibrate the channels the learning diagnostic can't identify --
    # Two reasons to want an experiment: (a) the model SAYS it can't identify the
    # channel (wide contribution interval / prior-dominated) — Radio/Print; and
    # (b) standard practice — validate your biggest-spend channels, because a
    # confident-but-wrong estimate (residual confounding) is invisible to every
    # observational diagnostic and only a lift test can expose it.
    unident = [
        c["channel"]
        for c in r2["channels"]
        if c["verdict"] == "prior-dominated" or (c.get("rel_ci_width", 0) or 0) > 1.5
    ]
    by_spend = sorted(sc.channels, key=lambda c: float(sc.spend[c].sum()), reverse=True)
    validate = [c for c in by_spend[:2] if c not in unident]
    flagged = unident + validate
    out["steps"]["v3_unidentified"] = unident
    out["steps"]["v3_validate"] = validate
    print(
        f"\nv3 (calibration) — unidentified (wide CI): {unident}; "
        f"validate biggest channels: {validate}. Anchoring all with geo-lift "
        "experiments and refitting ...",
        flush=True,
    )
    out["steps"]["v3_flagged"] = flagged
    if flagged:
        from mmm_framework.calibration import ExperimentCalibrator
        from mmm_framework.validation.results import LiftTestResult

        period = (str(sc.weeks[0].date()), str(sc.weeks[-1].date()))
        lifts = [
            LiftTestResult(
                channel=c,
                test_period=period,
                measured_lift=float(
                    sc.true_contribution[c]
                ),  # a well-powered lift test
                lift_se=float(0.12 * sc.true_contribution[c]),
            )
            for c in flagged
        ]
        cal = ExperimentCalibrator(m2, f2)
        outcome = cal.calibrate(
            lifts, refit=True, draws=draws, tune=tune, chains=chains, random_seed=seed
        )
        if outcome.model is not None:
            r3 = score(
                v2,
                outcome.model,
                outcome.results,
                prior_samples=prior_samples,
                seed=seed,
            )
            out["steps"]["v3_calibrated"] = r3
            print("  " + _summary_line("v3", r3), flush=True)
            m_final = outcome.model
        else:
            print("  calibration produced no usable priors; keeping v2", flush=True)
            m_final = m2
    else:
        m_final = m2

    # --- Step 8: sensitivity / refutation on the final model ----------------
    if refutation:
        print("\nStep 8 — refutation suite on the final model ...", flush=True)
        try:
            from mmm_framework.validation import ModelValidator, ValidationConfigBuilder

            cfg = (
                ValidationConfigBuilder()
                .silent()
                .without_residuals()
                .without_channel_diagnostics()
                .with_unobserved_confounding()
                .with_causal_refutation(draws=150, tune=150, chains=2)
                .build()
            )
            cfg.run_ppc = True
            summ = ModelValidator(m_final).validate(cfg)
            out["steps"]["refutation"] = {
                "tests": (
                    {t.name: bool(t.passed) for t in summ.causal_refutation.tests}
                    if summ.causal_refutation
                    else {}
                ),
                "ppc_pass": bool(summ.ppc.overall_pass) if summ.ppc else None,
                "fragile": (
                    list(summ.unobserved_confounding.fragile_channels)
                    if summ.unobserved_confounding
                    else []
                ),
            }
            print("  refutation:", out["steps"]["refutation"], flush=True)
        except Exception as exc:
            out["steps"]["refutation"] = {"error": str(exc)}
            print("  refutation error:", exc, flush=True)

    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--quick", action="store_true", help="cheap arc check: 200 draws, no refutation"
    )
    p.add_argument("--draws", type=int, default=700)
    p.add_argument("--tune", type=int, default=700)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--pymc", action="store_true", help="use pyMC instead of numpyro")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.quick:
        draws = tune = 200
        chains = 2
        refutation = False
        prior_samples = 400
    else:
        draws, tune, chains = args.draws, args.tune, args.chains
        refutation = True
        prior_samples = 600

    out = run(
        draws=draws,
        tune=tune,
        chains=chains,
        prior_samples=prior_samples,
        refutation=refutation,
        numpyro=not args.pymc,
        seed=args.seed,
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = "_quick" if args.quick else ""
    path = RESULTS / f"walkthrough{suffix}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {path}  ({out['elapsed_s']}s)")

    # arc summary
    steps = out["steps"]
    print("\n=== ARC ===")
    for key in ("v1_naive", "v2_causal", "v3_calibrated"):
        if key in steps:
            r = steps[key]
            print(
                f"  {key:16s} strong med|err|={r['median_abs_rel_error_strong']:.0%} "
                f"cover={r['coverage_strong']:.0%} div={r['divergences']}"
            )


if __name__ == "__main__":
    main()
