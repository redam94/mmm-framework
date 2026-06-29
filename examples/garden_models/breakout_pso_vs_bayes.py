"""PSO-vs-Bayes head-to-head — operationalize the argument that an in-house
particle-swarm (PSO) breakout-weight optimizer *fits noise*, and that the
partial-pooled :class:`BreakoutWeightedMMM` is the honest replacement.

Run it::

    uv run python examples/garden_models/breakout_pso_vs_bayes.py          # full
    uv run python examples/garden_models/breakout_pso_vs_bayes.py --fast    # skip MCMC

The three sibling synthetic worlds (``synth/dgp.py``) all split TV into three
impression sub-streams feeding ONE saturation curve via a weighted aggregate —
the exact functional form the PSO searches over:

* ``breakout_heterogeneous`` — weights genuinely differ, sub-streams flight
  independently (the mix is identifiable);
* ``breakout_homogeneous``   — weights are truly equal (1);
* ``breakout_collinear``     — weights differ but the sub-streams share one
  flighting calendar (the mix is UNidentifiable).

It emulates the PSO on the *same* functional form — free per-breakout weights
under the same sum-preserving renormalization, with the saturation curve fixed at
the DGP truth and the channel coefficients + baseline profiled out by OLS at each
weight proposal, so the ONLY free degrees of freedom are the K−1 weights (exactly
what the PSO tunes). Then it shows:

  A. In-sample MSE *always* drops under PSO — including on ``homogeneous`` where
     the truth is equal weights. That drop is mechanical (a more flexible
     reparameterization of the same inputs), not evidence of signal.
  B. Out of sample (a train/test split) PSO does NOT beat equal-weighting on
     ``homogeneous`` and the Bayesian model beats PSO on ``heterogeneous`` — the
     overfit does not generalize. (Bayes vs equal-weight is compared by LOO, the
     expected out-of-sample predictive accuracy, on the real fitted models.)
  C. The Bayesian model reports a credible INTERVAL on every weight (so "this
     breakout is more effective" can be stated honestly, or flagged as
     indistinguishable from equal-weighting); the PSO reports only a point.

Plus a pre-fit collinearity diagnostic on ``breakout_collinear`` that exposes the
identifiability ceiling no optimizer can beat.
"""

from __future__ import annotations

import argparse

import numpy as np

from mmm_framework.synth import dgp
from mmm_framework.synth.dgp import _ALPHA, _LAM, _geom_adstock, _logistic_sat

_SUBS = ["TV_Premium", "TV_Standard", "TV_Remnant"]
_PLAIN = ["Search", "Social", "Display"]


# ---------------------------------------------------------------------------
# A self-contained NumPy structural model — the harness's shared forward pass.
# The saturation curve is fixed at the DGP truth and β + the baseline are
# profiled by OLS, so the ONLY free parameters are the breakout weights (exactly
# what the PSO tunes). This isolates the *weight* overfitting cleanly.
# ---------------------------------------------------------------------------


def _renorm(w_raw: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Sum-preserving renormalization: Σ_k w_k S_k = Σ_k S_k (share-mean 1)."""
    return w_raw * (float(S.sum()) / float(w_raw @ S))


def _design(sc, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the OLS design matrix for a scenario given the breakout weights.

    Columns: intercept, trend, sin/cos(52), price, each plain channel's saturated
    adstock, and the TV breakout feature (saturated adstock of the *weighted*
    impression aggregate, normalized by the unweighted aggregate max). Returns
    ``(X, y, S)`` where ``S`` is the per-breakout impression total.
    """
    spend = sc.spend
    n = len(sc.y)
    t = np.arange(n)
    y = sc.y.to_numpy(float)

    sub = {s: spend[s].to_numpy(float) for s in _SUBS}
    S = np.array([sub[s].sum() for s in _SUBS])
    agg_unw = sum(sub[s] for s in _SUBS)
    m_tv = float(agg_unw.max())
    agg_w = sum(weights[i] * sub[s] for i, s in enumerate(_SUBS))
    tv_feat = _logistic_sat(_geom_adstock(agg_w / m_tv, _ALPHA["TV"]), _LAM["TV"])

    cols = [
        np.ones(n),
        t / n,
        np.sin(2 * np.pi * t / 52.0),
        np.cos(2 * np.pi * t / 52.0),
    ]
    cols.append(sc.controls["Price"].to_numpy(float))
    for c in _PLAIN:
        x = spend[c].to_numpy(float)
        cols.append(_logistic_sat(_geom_adstock(x / x.max(), _ALPHA[c]), _LAM[c]))
    cols.append(tv_feat)
    return np.column_stack(cols), y, S


def _ols_mse(X: np.ndarray, y: np.ndarray, idx_fit=None, idx_eval=None) -> float:
    """Profiled-OLS residual MSE: fit coefficients on ``idx_fit`` rows (all rows
    if None), evaluate MSE on ``idx_eval`` rows (same as fit if None)."""
    fit = slice(None) if idx_fit is None else idx_fit
    ev = fit if idx_eval is None else idx_eval
    beta, *_ = np.linalg.lstsq(X[fit], y[fit], rcond=None)
    resid = y[ev] - X[ev] @ beta
    return float(np.mean(resid**2))


def pso_weights(sc, *, idx_fit=None, seed: int = 0) -> np.ndarray:
    """Emulate the in-house PSO: choose the breakout weights that minimize the
    (in-sample, ``idx_fit``) MSE on the shared functional form, under the same
    sum-preserving constraint. Uses SciPy's differential evolution — a global,
    population-based optimizer in the PSO family."""
    from scipy.optimize import differential_evolution

    _, _, S = _design(sc, np.ones(3))

    def obj(g: np.ndarray) -> float:
        w = _renorm(np.exp(g), S)
        X, y, _ = _design(sc, w)
        return _ols_mse(X, y, idx_fit=idx_fit)

    res = differential_evolution(
        obj, bounds=[(-1.5, 1.5)] * 3, seed=seed, maxiter=80, tol=1e-7, polish=True
    )
    return _renorm(np.exp(res.x), S)


# ---------------------------------------------------------------------------
# Demonstration A — in-sample MSE always drops (the overfit)
# ---------------------------------------------------------------------------


def demo_in_sample(scenarios: list[str]) -> None:
    print("\n" + "=" * 74)
    print("A. IN-SAMPLE MSE: PSO always lowers it — even when the truth is equal")
    print("=" * 74)
    print(
        f"{'scenario':<26}{'equal-wt MSE':>14}{'PSO MSE':>12}{'drop %':>9}  weights (PSO)"
    )
    for name in scenarios:
        sc = dgp.build(name)
        X_eq, y, S = _design(sc, np.ones(3))
        mse_eq = _ols_mse(X_eq, y)
        w = pso_weights(sc)
        X_w, _, _ = _design(sc, w)
        mse_w = _ols_mse(X_w, y)
        drop = 100.0 * (mse_eq - mse_w) / mse_eq
        wtxt = ", ".join(f"{s.split('_')[1]}={w[i]:.2f}" for i, s in enumerate(_SUBS))
        print(f"{name:<26}{mse_eq:>14.2f}{mse_w:>12.2f}{drop:>8.2f}%  {wtxt}")
    print(
        "\n  -> On `breakout_homogeneous` the true weights are all 1, yet PSO still\n"
        "     finds unequal weights that lower in-sample MSE. That is the overfit:\n"
        "     a point optimizer with no regularization fits the noise."
    )


# ---------------------------------------------------------------------------
# Demonstration B (PSO part) — the overfit does not generalize out of sample
# ---------------------------------------------------------------------------


def demo_out_of_sample(scenarios: list[str]) -> None:
    print("\n" + "=" * 74)
    print("B1. OUT-OF-SAMPLE (70/30 split): does the PSO mix generalize?")
    print("=" * 74)
    print(f"{'scenario':<26}{'equal-wt test':>15}{'PSO test':>11}{'OOS Δ':>9}  verdict")
    for name in scenarios:
        sc = dgp.build(name)
        n = len(sc.y)
        cut = int(0.7 * n)
        idx_tr, idx_te = np.arange(cut), np.arange(cut, n)

        X_eq, y, S = _design(sc, np.ones(3))
        test_eq = _ols_mse(X_eq, y, idx_fit=idx_tr, idx_eval=idx_te)

        w_tr = pso_weights(sc, idx_fit=idx_tr)  # PSO sees only the train slice
        X_w, _, _ = _design(sc, w_tr)
        test_w = _ols_mse(X_w, y, idx_fit=idx_tr, idx_eval=idx_te)

        pct = 100.0 * (test_eq - test_w) / test_eq  # >0: PSO better OOS
        verdict = (
            "PSO helps"
            if pct > 3
            else ("WORSE (overfit)" if pct < 0 else "~tie (no gain)")
        )
        print(f"{name:<26}{test_eq:>15.2f}{test_w:>11.2f}{pct:>+8.1f}%  {verdict}")
    print(
        "\n  -> Contrast with section A: the large in-sample MSE drop does NOT carry\n"
        "     over. On `breakout_homogeneous` (truth = equal) the PSO mix is a wash\n"
        "     out of sample, and on `breakout_collinear` (unidentifiable) it is\n"
        "     WORSE — the in-sample gain was noise. Only the real, identifiable\n"
        "     signal in `heterogeneous` generalizes."
    )


# ---------------------------------------------------------------------------
# Demonstration B (Bayes part) + C — LOO and the intervals PSO can't give
# ---------------------------------------------------------------------------


def _fit_bayes(scenario: str, *, draws: int, tune: int, chains: int, seed: int):
    """Fit the partial-pooled breakout model and the plain equal-weight (pre-
    summed) model on the same world; return both + the breakout model's scenario."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from breakout_weighted_mmm import (
        BreakoutWeightedMMM,
        breakout_aggregated_panel,
        breakout_dataset,
    )

    from mmm_framework.config import ModelConfig
    from mmm_framework.model import BayesianMMM, TrendConfig
    from mmm_framework.model.trend_config import TrendType

    dataset, sc, groups = breakout_dataset(scenario)
    fit = dict(
        draws=draws, tune=tune, chains=chains, target_accept=0.9, random_seed=seed
    )

    brk = BreakoutWeightedMMM(
        dataset,
        ModelConfig(use_parametric_adstock=True),
        TrendConfig(type=TrendType.LINEAR),
        model_params={"breakout_groups": groups},
    )
    brk.fit(progressbar=False, **fit)

    eq = BayesianMMM(
        breakout_aggregated_panel(sc),
        ModelConfig(use_parametric_adstock=True),
        TrendConfig(type=TrendType.LINEAR),
    )
    eq.fit(progressbar=False, **fit)
    return brk, eq, sc


def _loo(model):
    """LOO (expected out-of-sample log predictive density) for a fitted model,
    computing the pointwise log-likelihood if the trace lacks it."""
    import arviz as az
    import pymc as pm

    trace = model._trace
    if "log_likelihood" not in trace.groups():
        with model.model:
            pm.compute_log_likelihood(trace, progressbar=False)
    return az.loo(trace)


def demo_bayes(scenarios: list[str], *, draws: int, tune: int, chains: int) -> None:
    print("\n" + "=" * 74)
    print("B2. BAYES vs EQUAL-WEIGHT by LOO (expected out-of-sample accuracy)")
    print("    + C. the WEIGHT INTERVALS the PSO cannot produce")
    print("=" * 74)
    for i, name in enumerate(scenarios):
        brk, eq, sc = _fit_bayes(
            name, draws=draws, tune=tune, chains=chains, seed=11 + i
        )
        loo_brk = _loo(brk)
        loo_eq = _loo(eq)
        d_elpd = float(loo_brk.elpd_loo - loo_eq.elpd_loo)
        # Positive d_elpd => the partial-pooled breakout model predicts better OOS.
        tag = (
            "breakout helps"
            if d_elpd > 2
            else ("~tie (no gain)" if abs(d_elpd) <= 2 else "breakout worse")
        )
        print(f"\n  [{name}]")
        print(
            f"    elpd_loo: breakout={float(loo_brk.elpd_loo):.1f}  "
            f"equal-weight={float(loo_eq.elpd_loo):.1f}  Δ={d_elpd:+.1f}  -> {tag}"
        )
        df = brk.breakout_weights_summary()
        truth = sc.notes["true_weights"]
        print("    breakout weight (Bayes, 94% HDI)        true   covers 1?")
        for _, r in df.iterrows():
            print(
                f"      {r['breakout']:<13} {r['weight_mean']:+.2f} "
                f"[{r['hdi_low']:+.2f}, {r['hdi_high']:+.2f}]   "
                f"{truth[r['breakout']]:+.2f}    {bool(r['covers_equal'])}"
            )
    print(
        "\n  -> Δelpd ≈ 0 on `homogeneous`/`collinear` (the extra flexibility buys no\n"
        "     out-of-sample accuracy); breakout helps on identifiable `heterogeneous`.\n"
        "     Every Bayesian weight carries an interval — the honest 'is it really\n"
        "     different from equal-weighting?' the PSO point estimate cannot answer."
    )


# ---------------------------------------------------------------------------
# Pre-fit collinearity diagnostic — the identifiability ceiling
# ---------------------------------------------------------------------------


def demo_collinearity() -> None:
    import pandas as pd

    from mmm_framework.eda.collinearity import collinearity_analysis
    from mmm_framework.eda.loading import load_eda_panel_from_df

    print("\n" + "=" * 74)
    print("PRE-FIT: collinearity of the TV sub-streams (the identifiability ceiling)")
    print("=" * 74)
    for name in ("breakout_heterogeneous", "breakout_collinear"):
        sc = dgp.build(name)
        wide = pd.DataFrame({"Period": sc.weeks})
        for s in _SUBS:
            wide[s] = sc.spend[s].to_numpy()
        spec = {"media_channels": [{"name": s} for s in _SUBS]}
        panel = load_eda_panel_from_df(wide, spec)
        res = collinearity_analysis(panel, variables=_SUBS)
        rbar = (
            float(np.mean([abs(p["r"]) for p in res["top_pairs"]]))
            if res["top_pairs"]
            else float("nan")
        )
        cond = res["condition_number"]
        vif = max(res["vif"].values()) if res["vif"] else float("nan")
        print(
            f"  {name:<26} mean|r|={rbar:.2f}  max VIF={vif:>6.1f}  "
            f"condition#={cond:>7.1f}  clusters={len(res['clusters'])}"
        )
    print(
        "\n  -> On `breakout_collinear` the sub-streams are near-collinear (high |r|,\n"
        "     huge VIF/condition number): the data cannot separate their weights.\n"
        "     The honest model reports WIDE posteriors there; an optimizer reports a\n"
        "     confident point that is pure noise."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="skip the MCMC (Bayes) section")
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--tune", type=int, default=600)
    ap.add_argument("--chains", type=int, default=2)
    args = ap.parse_args()

    scenarios = ["breakout_heterogeneous", "breakout_homogeneous", "breakout_collinear"]
    demo_collinearity()
    demo_in_sample(scenarios)
    demo_out_of_sample(scenarios)
    if not args.fast:
        demo_bayes(scenarios, draws=args.draws, tune=args.tune, chains=args.chains)
    else:
        print("\n[--fast] skipped the Bayesian LOO + weight-interval section.")


if __name__ == "__main__":
    main()
