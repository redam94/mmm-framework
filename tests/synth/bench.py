"""Benchmark fit-time and convergence of the MMM across configurations.

Measures *pure* ``fit()`` wall-clock plus convergence (r-hat, min ESS,
divergences) while varying one knob at a time, and times the post-fit operations
the stress matrix relies on (counterfactual contributions, a refutation refit,
parameter-learning). Writes ``results/bench.json`` + prints a markdown summary.

    uv run python -m tests.synth.bench
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .dgp import (
    _baseline,
    _geom_adstock,
    _logistic_sat,
    build,
)


def _synth_panel(n_weeks: int, n_channels: int, seed: int):
    """A clean-style additive panel with an arbitrary number of channels."""
    from mmm_framework.config import (
        ControlVariableConfig,
        DimensionType,
        KPIConfig,
        MediaChannelConfig,
        MFFConfig,
    )
    from mmm_framework.data_loader import PanelCoordinates, PanelDataset

    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2021-01-04", periods=n_weeks, freq="W-MON")
    chans = [f"C{i}" for i in range(n_channels)]
    price = 12.0 + 0.5 * np.cos(2 * np.pi * np.arange(n_weeks) / 52.0)
    baseline = _baseline(rng, n_weeks, price)
    # reuse the pulsed-level machinery via a per-channel periodic burst
    spend = {}
    mu = baseline.copy()
    for i, c in enumerate(chans):
        period = 7.0 + (i % 7)
        phase = rng.random() * 2 * np.pi
        t = np.arange(n_weeks)
        burst = np.clip(np.sin(2 * np.pi * t / period + phase), 0, None)
        level = (0.08 + 1.6 * burst) * rng.lognormal(0, 0.25, n_weeks)
        sp = np.clip(80.0 * level, 0.5, None)
        spend[c] = sp
        xn = sp / sp.max()
        amp = 80.0 + 40.0 * ((i % 4) / 3.0)
        mu = mu + amp * _logistic_sat(_geom_adstock(xn, 0.5), 1.6)
    y = pd.Series(np.clip(mu + rng.normal(0, 20, n_weeks), 1.0, None), name="Sales")
    coords = PanelCoordinates(
        periods=weeks,
        geographies=None,
        products=None,
        channels=chans,
        controls=["Price"],
    )
    config = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name=c, dimensions=[DimensionType.PERIOD]) for c in chans
        ],
        controls=[
            ControlVariableConfig(name="Price", dimensions=[DimensionType.PERIOD])
        ],
    )
    return PanelDataset(
        y=y,
        X_media=pd.DataFrame(spend),
        X_controls=pd.DataFrame({"Price": price}),
        coords=coords,
        index=weeks,
        config=config,
    )


def _model(panel, *, draws, tune, chains, parametric, numpyro, seed, target_accept=0.9):
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
        use_parametric_adstock=parametric,
        optim_seed=seed,
    )
    return BayesianMMM(panel, cfg, TrendConfig(type=TrendType.LINEAR))


def _conv(trace) -> dict:
    """r-hat max, min bulk/tail ESS, divergences, and the worst-converged vars."""
    import arviz as az

    rhat = az.rhat(trace)
    ess_b = az.ess(trace, method="bulk")
    ess_t = az.ess(trace, method="tail")
    rhat_by = {v: float(rhat[v].max()) for v in rhat.data_vars}
    essb_by = {v: float(ess_b[v].min()) for v in ess_b.data_vars}
    try:
        div = int(trace.sample_stats.diverging.sum())
    except Exception:
        div = -1
    worst_rhat = sorted(rhat_by.items(), key=lambda kv: -kv[1])[:5]
    worst_ess = sorted(essb_by.items(), key=lambda kv: kv[1])[:5]
    return {
        "rhat_max": float(max(rhat_by.values())),
        "ess_bulk_min": float(min(essb_by.values())),
        "ess_tail_min": float(min(float(ess_t[v].min()) for v in ess_t.data_vars)),
        "divergences": div,
        "worst_rhat": worst_rhat,
        "worst_ess_bulk": worst_ess,
    }


def time_fit(label, panel, *, draws, tune, chains, parametric, numpyro, seed=0, **kw):
    print(f"  fitting [{label}] ...", flush=True)
    t0 = time.time()
    mmm = _model(
        panel,
        draws=draws,
        tune=tune,
        chains=chains,
        parametric=parametric,
        numpyro=numpyro,
        seed=seed,
        **kw,
    )
    build_s = time.time() - t0
    t1 = time.time()
    try:
        fit = mmm.fit(random_seed=seed)
        sample_s = time.time() - t1
        c = _conv(fit.trace)
        err = None
    except Exception as exc:
        sample_s = time.time() - t1
        c = {
            "rhat_max": float("nan"),
            "ess_bulk_min": float("nan"),
            "ess_tail_min": float("nan"),
            "divergences": -1,
            "worst_rhat": [],
            "worst_ess_bulk": [],
        }
        err = f"{type(exc).__name__}: {exc}"
    n_obs = panel.y.shape[0]
    n_ch = len(panel.X_media.columns)
    rec = {
        "label": label,
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "parametric": parametric,
        "numpyro": numpyro,
        "n_obs": n_obs,
        "n_channels": n_ch,
        "build_s": round(build_s, 2),
        "sample_s": round(sample_s, 1),
        "draws_per_s": round(draws * chains / sample_s, 1) if sample_s else None,
        "error": err,
        **{k: v for k, v in c.items() if k not in ("worst_rhat", "worst_ess_bulk")},
        "worst_rhat": c["worst_rhat"],
        "worst_ess_bulk": c["worst_ess_bulk"],
    }
    print(
        f"    {label}: {sample_s:.0f}s  rhat={rec['rhat_max']:.3f} "
        f"ess_bulk_min={rec['ess_bulk_min']:.0f} div={rec['divergences']}"
        + (f"  ERR {err}" if err else ""),
        flush=True,
    )
    return mmm, fit if err is None else None, rec


def main():
    out = Path("tests/synth/results")
    out.mkdir(parents=True, exist_ok=True)
    records = []
    clean = build("clean")  # 156 weeks, 4 channels

    print("== fit-time / convergence grid ==", flush=True)
    # Baseline + one-knob-at-a-time sweeps.
    base_mmm, base_fit, base = time_fit(
        "base(600d/4c/param/pymc/156w/4ch)",
        clean.panel(),
        draws=600,
        tune=600,
        chains=4,
        parametric=True,
        numpyro=False,
    )
    records.append(base)
    for draws in (300, 1200):
        records.append(
            time_fit(
                f"draws={draws}",
                clean.panel(),
                draws=draws,
                tune=draws,
                chains=4,
                parametric=True,
                numpyro=False,
            )[2]
        )
    records.append(
        time_fit(
            "chains=2",
            clean.panel(),
            draws=600,
            tune=600,
            chains=2,
            parametric=True,
            numpyro=False,
        )[2]
    )
    records.append(
        time_fit(
            "legacy_adstock",
            clean.panel(),
            draws=600,
            tune=600,
            chains=4,
            parametric=False,
            numpyro=False,
        )[2]
    )
    records.append(
        time_fit(
            "numpyro",
            clean.panel(),
            draws=600,
            tune=600,
            chains=4,
            parametric=True,
            numpyro=True,
        )[2]
    )
    records.append(
        time_fit(
            "target_accept=0.99",
            clean.panel(),
            draws=600,
            tune=600,
            chains=4,
            parametric=True,
            numpyro=False,
            target_accept=0.99,
        )[2]
    )
    for w in (104, 260):
        records.append(
            time_fit(
                f"weeks={w}",
                _synth_panel(w, 4, 0),
                draws=600,
                tune=600,
                chains=4,
                parametric=True,
                numpyro=False,
            )[2]
        )
    records.append(
        time_fit(
            "channels=8",
            _synth_panel(156, 8, 0),
            draws=600,
            tune=600,
            chains=4,
            parametric=True,
            numpyro=False,
        )[2]
    )

    # Post-fit operation costs on the baseline fit.
    print("== post-fit operation costs (baseline fit) ==", flush=True)
    ops = {}
    t = time.time()
    base_mmm.compute_counterfactual_contributions(
        compute_uncertainty=True, hdi_prob=0.90, random_seed=0
    )
    ops["counterfactual_contributions_s"] = round(time.time() - t, 1)
    t = time.time()
    base_mmm.compute_parameter_learning(prior_samples=600)
    ops["parameter_learning_s"] = round(time.time() - t, 1)
    t = time.time()
    try:
        from mmm_framework.validation import ModelValidator, ValidationConfigBuilder

        cfg = (
            ValidationConfigBuilder()
            .silent()
            .without_residuals()
            .without_channel_diagnostics()
            .with_causal_refutation(draws=150, tune=150, chains=2)
            .build()
        )
        cfg.run_ppc = False
        ModelValidator(base_mmm).validate(cfg)
        ops["refutation_suite_s"] = round(time.time() - t, 1)
    except Exception as exc:
        ops["refutation_suite_s"] = f"err: {exc}"
    print("  ops:", ops, flush=True)

    payload = {"grid": records, "ops": ops}
    (out / "bench.json").write_text(json.dumps(payload, indent=2, default=str))
    print("\nWrote tests/synth/results/bench.json", flush=True)

    # quick markdown
    cols = [
        "label",
        "n_obs",
        "n_channels",
        "chains",
        "draws",
        "sample_s",
        "draws_per_s",
        "rhat_max",
        "ess_bulk_min",
        "divergences",
    ]
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for r in records:
        print("| " + " | ".join(str(r.get(c)) for c in cols) + " |")
    print("\nworst-converged params (baseline):", base.get("worst_ess_bulk"))


if __name__ == "__main__":
    main()
