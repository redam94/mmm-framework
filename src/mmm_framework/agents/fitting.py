"""Relocatable model-building + fitting (Phase 2, PR-C.2).

``build_and_fit(spec, dataset_path)`` builds, fits, serializes a ``BayesianMMM``
and returns ``(mmm, results, info)`` with **no** langchain / graph-state coupling
— so the SAME function runs in-process for ``InProcessKernel`` and inside the
per-session kernel for ``SubprocessKernel`` (where it removes the Phase-1
boundary: the fitted model becomes a kernel global). The caller deposits
``mmm``/``results`` where the model lives (``MODEL_CACHE`` in-process, kernel
globals in the subprocess); ``build_and_fit`` does NOT touch the cache or build a
``Command``. It raises on failure — the tool/kernel turns that into an error
result. ``info`` is JSON-serializable (summary, model_run record, report_path,
and the ``dashboard`` payload).

``_build_prior`` / ``_mff_config_from_spec`` live here (not ``tools.py``) so the
kernel can import them without a cycle; ``tools.py`` imports them from here.
"""

from __future__ import annotations

from mmm_framework import (
    MFFConfigBuilder,
    KPIConfigBuilder,
    MediaChannelConfigBuilder,
    ControlVariableConfigBuilder,
    ModelConfigBuilder,
    TrendConfigBuilder,
    PriorConfigBuilder,
    BayesianMMM,
    load_mff,
)
from mmm_framework.builders.model import SeasonalityConfigBuilder
from mmm_framework.builders.prior import AdstockConfigBuilder, SaturationConfigBuilder

# Where auto-saved models land (relative to the running process cwd). PR-C.3
# workspace-resolves this so a subprocess-kernel-written model is findable by the
# API process.
_MODELS_DIR = "mmm_models"


def _build_prior(p: dict):
    """Convert a {distribution, params} dict into a PriorConfig."""
    dist = p.get("distribution", "half_normal")
    params = p.get("params", {})
    b = PriorConfigBuilder()
    if dist == "normal":
        b.normal(mu=float(params.get("mu", 0.0)), sigma=float(params.get("sigma", 1.0)))
    elif dist == "log_normal":
        b.log_normal(
            mu=float(params.get("mu", 0.0)), sigma=float(params.get("sigma", 1.0))
        )
    elif dist == "gamma":
        b.gamma(
            alpha=float(params.get("alpha", 2.0)), beta=float(params.get("beta", 1.0))
        )
    elif dist == "beta":
        b.beta(
            alpha=float(params.get("alpha", 2.0)), beta=float(params.get("beta", 2.0))
        )
    elif dist == "truncated_normal":
        b.truncated_normal(
            mu=float(params.get("mu", 0.0)),
            sigma=float(params.get("sigma", 1.0)),
            lower=float(params.get("lower", 0.0)),
        )
    elif dist == "half_student_t":
        b.half_student_t(
            nu=float(params.get("nu", 3.0)), sigma=float(params.get("sigma", 1.0))
        )
    else:  # half_normal (default)
        b.half_normal(sigma=float(params.get("sigma", 1.0)))
    return b.build()


# ── Consumed spec.priors paths ────────────────────────────────────────────────
# Registry of every `priors.*` path that build_model / _mff_config_from_spec
# actually reads. update_model_setting validates writes against this so a prior
# the builder would silently drop is rejected up front (the PPC-no-op bug:
# `priors.intercept` used to be accepted into the spec but never consumed).
# Adding a new prior read below MUST come with a registry entry here.

_MEDIA_PRIOR_PARAMS = {
    "coefficient",
    "adstock_alpha",
    "adstock_theta",
    "saturation_kappa",
    "saturation_slope",
}
_CONTROL_PRIOR_PARAMS = {"coefficient", "allow_negative"}
_SCALAR_PRIOR_KEYS = {
    "intercept": {"mu", "sigma"},
    "seasonality": {
        "prior_sigma",
        "yearly_prior_sigma",
        "monthly_prior_sigma",
        "weekly_prior_sigma",
    },
    "trend": {
        "growth_prior_mu",
        "growth_prior_sigma",
        "changepoint_prior_scale",
        "spline_prior_sigma",
        "gp_lengthscale_prior_mu",
        "gp_lengthscale_prior_sigma",
        "gp_amplitude_prior_sigma",
    },
}


def unconsumed_prior_path(parts: list[str], value, spec: dict) -> str | None:
    """Return an error message when writing ``value`` at the ``priors.*`` path
    ``parts`` would never be read by ``build_model`` (a silently-dropped prior),
    else None. Dict values are expanded so e.g. setting the whole
    ``priors.intercept`` dict validates its keys too. ``spec`` supplies the
    known channel/control names for typo detection."""

    def _leaves(p: list[str], v) -> list[list[str]]:
        if isinstance(v, dict) and v:
            out: list[list[str]] = []
            for k, sub in v.items():
                out.extend(_leaves(p + [str(k)], sub))
            return out
        return [p]

    groups = "media, controls, seasonality, trend, intercept"
    channels = {m.get("name") for m in spec.get("media_channels", [])}
    controls = {c.get("name") for c in spec.get("control_variables", [])}

    for leaf in _leaves(list(parts), value):
        path = ".".join(leaf)
        if len(leaf) < 3:
            return (
                f"`{path}` would have no effect — the model builder never reads "
                f"it. Set a specific prior under one of: {groups} "
                "(e.g. `priors.intercept.sigma`)."
            )
        group = leaf[1]
        if group in _SCALAR_PRIOR_KEYS:
            keys = _SCALAR_PRIOR_KEYS[group]
            if len(leaf) != 3 or leaf[2] not in keys:
                return (
                    f"`{path}` would have no effect — the model builder never "
                    f"reads it. Valid `priors.{group}.*` keys: "
                    f"{', '.join(sorted(keys))}."
                )
        elif group == "media":
            if channels and leaf[2] not in channels:
                return (
                    f"`{path}` names unknown media channel '{leaf[2]}'. "
                    f"Channels in the spec: {', '.join(sorted(channels))}."
                )
            if len(leaf) < 4 or leaf[3] not in _MEDIA_PRIOR_PARAMS:
                return (
                    f"`{path}` would have no effect — the model builder never "
                    f"reads it. Valid per-channel prior params: "
                    f"{', '.join(sorted(_MEDIA_PRIOR_PARAMS))} "
                    "(each a {distribution, params} dict)."
                )
        elif group == "controls":
            if controls and leaf[2] not in controls:
                return (
                    f"`{path}` names unknown control variable '{leaf[2]}'. "
                    f"Controls in the spec: {', '.join(sorted(controls))}."
                )
            if len(leaf) < 4 or leaf[3] not in _CONTROL_PRIOR_PARAMS:
                return (
                    f"`{path}` would have no effect — the model builder never "
                    f"reads it. Valid per-control prior params: "
                    f"{', '.join(sorted(_CONTROL_PRIOR_PARAMS))}."
                )
        else:
            return (
                f"`{path}` would have no effect — the model builder never reads "
                f"`priors.{group}`. Valid prior groups: {groups}."
            )
    return None


def _mff_config_from_spec(spec: dict):
    """Build an ``MFFConfig`` from a (normalized) model_spec dict — shared by fit
    and by ``load_fitted_model`` (the panel a saved model reloads against)."""
    mff_builder = MFFConfigBuilder()

    kpi_builder = KPIConfigBuilder(spec["kpi"])
    if spec.get("kpi_level") == "geo":
        kpi_builder.by_geo()
    else:
        kpi_builder.national()
    mff_builder.with_kpi_builder(kpi_builder)

    media_priors = spec.get("priors", {}).get("media", {})
    control_priors_cfg = spec.get("priors", {}).get("controls", {})

    for media in spec.get("media_channels", []):
        ch_name = media["name"]
        ch_priors = media_priors.get(ch_name, {})
        ch_builder = MediaChannelConfigBuilder(ch_name).national()

        adstock_cfg = media.get("adstock", {})
        adstock_type = adstock_cfg.get("type", "geometric").lower()
        l_max = int(adstock_cfg.get("l_max", 8))
        ab = AdstockConfigBuilder()
        if adstock_type == "weibull":
            ab.weibull().with_max_lag(l_max)
        elif adstock_type == "delayed":
            ab.delayed().with_max_lag(l_max)
        else:
            ab.geometric().with_max_lag(l_max)
        if "adstock_alpha" in ch_priors:
            ab.with_alpha_prior(_build_prior(ch_priors["adstock_alpha"]))
        if "adstock_theta" in ch_priors:
            ab.with_theta_prior(_build_prior(ch_priors["adstock_theta"]))
        ch_builder.with_adstock_builder(ab)

        sat_type = media.get("saturation", {}).get("type", "hill").lower()
        sb = SaturationConfigBuilder()
        if sat_type == "logistic":
            sb.logistic()
        elif sat_type in ("michaelis_menten", "michaelis-menten"):
            sb.michaelis_menten()
        elif sat_type == "tanh":
            sb.tanh()
        else:
            sb.hill()
        if "saturation_kappa" in ch_priors:
            sb.with_kappa_prior(_build_prior(ch_priors["saturation_kappa"]))
        if "saturation_slope" in ch_priors:
            sb.with_slope_prior(_build_prior(ch_priors["saturation_slope"]))
        ch_builder.with_saturation_builder(sb)

        if "coefficient" in ch_priors:
            ch_builder.with_coefficient_prior(_build_prior(ch_priors["coefficient"]))

        mff_builder.add_media_builder(ch_builder)

    for control in spec.get("control_variables", []):
        cv_name = control["name"]
        cv_builder = ControlVariableConfigBuilder(cv_name).national()
        cv_priors = control_priors_cfg.get(cv_name, {})
        if cv_priors.get("allow_negative", True) is False:
            cv_builder.positive_only()
        if "coefficient" in cv_priors:
            cv_builder.with_coefficient_prior(_build_prior(cv_priors["coefficient"]))
        mff_builder.add_control_builder(cv_builder)

    granularity = spec.get("time_granularity", "weekly").lower()
    if granularity == "daily":
        mff_builder.daily()
    elif granularity == "monthly":
        mff_builder.monthly()
    else:
        mff_builder.weekly()
    mff_builder.with_date_format("%Y-%m-%d")
    return mff_builder.build()


def _canonical_trend_type(trend_spec: dict) -> str:
    """Normalize the spec's trend type to the canonical TrendType name,
    tolerating common LLM aliases (``piecewise_linear``, ``gp``)."""
    trend_type = str(trend_spec.get("type", "linear")).strip().lower().replace("-", "_")
    return {"piecewise_linear": "piecewise", "gp": "gaussian_process"}.get(
        trend_type, trend_type
    )


def build_model(spec: dict, dataset_path: str) -> BayesianMMM:
    """Build an UNFITTED ``BayesianMMM`` from a normalized spec + dataset —
    the panel/config/trend stages of ``build_and_fit`` without any sampling.
    The PyMC graph builds lazily on first use, so this is cheap enough for
    pre-fit prior predictive checks. Raises on failure."""
    # 1. MFFConfig + panel
    mff_config = _mff_config_from_spec(spec)
    try:
        panel = load_mff(dataset_path, mff_config)
    except Exception as e:
        msg = str(e)
        latent = [
            c.get("name")
            for c in spec.get("control_variables", [])
            if str(c.get("name", "")).strip().lower()
            in ("trend", "seasonality", "season")
        ]
        if "Missing expected variables" in msg and latent:
            raise ValueError(
                f"{msg} — Hint: {', '.join(repr(n) for n in latent)} in "
                "control_variables look like latent baseline components, not "
                "dataset variables. Remove them from control_variables and use "
                "the built-in `trend` / `seasonality` settings instead "
                "(e.g. update_model_setting('seasonality.yearly', 4))."
            ) from e
        raise

    # 2. Inference + model config
    inf = spec.get("inference", {})
    chains = int(inf.get("chains", 4))
    draws = int(inf.get("draws", 1000))
    tune = int(inf.get("tune", 1000))
    target_accept = float(inf.get("target_accept", 0.85))

    model_config_builder = (
        ModelConfigBuilder()
        .bayesian_numpyro()
        .with_chains(chains)
        .with_draws(draws)
        .with_tune(tune)
        .with_target_accept(target_accept)
    )
    intercept_prior = spec.get("priors", {}).get("intercept", {})
    if "mu" in intercept_prior or "sigma" in intercept_prior:
        model_config_builder.with_intercept_prior(
            mu=float(intercept_prior.get("mu", 0.0)),
            sigma=float(intercept_prior.get("sigma", 0.5)),
        )
    season = spec.get("seasonality", {})
    yearly = int(season.get("yearly", 0))
    monthly = int(season.get("monthly", 0))
    weekly = int(season.get("weekly", 0))
    if yearly > 0 or monthly > 0 or weekly > 0:
        seas_prior = spec.get("priors", {}).get("seasonality", {})

        def _seas_sigma(component: str) -> float | None:
            v = seas_prior.get(f"{component}_prior_sigma")
            return None if v is None else float(v)

        sb = SeasonalityConfigBuilder()
        sb.no_seasonality()  # builder defaults yearly=2; only the spec decides
        if "prior_sigma" in seas_prior:
            sb.with_prior_sigma(float(seas_prior["prior_sigma"]))
        if yearly > 0:
            sb.with_yearly(order=yearly, prior_sigma=_seas_sigma("yearly"))
        if monthly > 0:
            sb.with_monthly(order=monthly, prior_sigma=_seas_sigma("monthly"))
        if weekly > 0:
            sb.with_weekly(order=weekly, prior_sigma=_seas_sigma("weekly"))
        model_config_builder.with_seasonality_builder(sb)
    model_config = model_config_builder.build()

    # 3. Trend config
    trend_spec = spec.get("trend", {})
    trend_type = _canonical_trend_type(trend_spec)
    trend_prior_cfg = spec.get("priors", {}).get("trend", {})

    def _wire_growth_prior(tb_):
        # The base slope (linear `growth`, piecewise `trend_k`) reads these.
        if (
            "growth_prior_mu" in trend_prior_cfg
            or "growth_prior_sigma" in trend_prior_cfg
        ):
            tb_.with_growth_prior(
                mu=float(trend_prior_cfg.get("growth_prior_mu", 0.0)),
                sigma=float(trend_prior_cfg.get("growth_prior_sigma", 0.5)),
            )

    tb = TrendConfigBuilder()
    if trend_type == "piecewise":
        tb.piecewise()
        if "n_changepoints" in trend_spec:
            tb.with_n_changepoints(int(trend_spec["n_changepoints"]))
        if "changepoint_range" in trend_spec:
            tb.with_changepoint_range(float(trend_spec["changepoint_range"]))
        if "changepoint_prior_scale" in trend_prior_cfg:
            tb.with_changepoint_prior_scale(
                float(trend_prior_cfg["changepoint_prior_scale"])
            )
        _wire_growth_prior(tb)
    elif trend_type == "spline":
        tb.spline()
        if "n_knots" in trend_spec:
            tb.with_n_knots(int(trend_spec["n_knots"]))
        if "spline_degree" in trend_spec:
            tb.with_spline_degree(int(trend_spec["spline_degree"]))
        if "spline_prior_sigma" in trend_prior_cfg:
            tb.with_spline_prior_sigma(float(trend_prior_cfg["spline_prior_sigma"]))
    elif trend_type == "gaussian_process":
        tb.gaussian_process()
        if "gp_lengthscale_prior_mu" in trend_prior_cfg:
            tb.with_gp_lengthscale(
                mu=float(trend_prior_cfg["gp_lengthscale_prior_mu"]),
                sigma=float(trend_prior_cfg.get("gp_lengthscale_prior_sigma", 0.2)),
            )
        if "gp_amplitude_prior_sigma" in trend_prior_cfg:
            tb.with_gp_amplitude(
                sigma=float(trend_prior_cfg["gp_amplitude_prior_sigma"])
            )
    elif trend_type == "none":
        pass
    elif trend_type == "linear":
        tb.linear()
        _wire_growth_prior(tb)
    else:
        # Refuse rather than silently fitting a linear trend the user didn't ask for
        raise ValueError(
            f"Unknown trend type '{trend_spec.get('type')}'. "
            "Valid types: none, linear, piecewise, spline, gaussian_process."
        )
    trend_config = tb.build()

    mmm = BayesianMMM(panel, model_config, trend_config)

    # 4. Experiment calibration (closes the measurement loop): the spec carries
    # completed lift readouts as ExperimentMeasurement dicts; they become
    # in-graph likelihood terms on the channel's contribution/ROAS/mROAS.
    exps_spec = spec.get("experiments") or []
    if exps_spec:
        from mmm_framework.calibration.likelihood import ExperimentMeasurement

        try:
            measurements = [ExperimentMeasurement.from_dict(e) for e in exps_spec]
        except Exception as e:  # noqa: BLE001
            raise ValueError(
                f"Invalid experiment calibration entry in spec.experiments: {e}. "
                "Each entry needs channel, test_period [start, end], value, se, "
                "and estimand (contribution | roas | mroas)."
            ) from e
        mmm.add_experiment_calibration(measurements)

    return mmm


def build_and_fit(spec: dict, dataset_path: str):
    """Build + fit + serialize. Returns ``(mmm, results, info)``. Raises on
    failure. ``spec`` must be normalized (bare-string vars already coerced)."""
    import json as _json
    import os as _os
    from datetime import datetime, timezone

    mmm = build_model(spec, dataset_path)

    inf = spec.get("inference", {})
    chains = int(inf.get("chains", 4))
    draws = int(inf.get("draws", 1000))
    tune = int(inf.get("tune", 1000))
    target_accept = float(inf.get("target_accept", 0.85))
    random_seed = int(inf.get("random_seed", 42))
    season = spec.get("seasonality", {})
    yearly = int(season.get("yearly", 0))
    monthly = int(season.get("monthly", 0))
    weekly = int(season.get("weekly", 0))
    trend_type = _canonical_trend_type(spec.get("trend", {}))

    # 4. Fit
    results = mmm.fit(random_seed=random_seed)

    # 5. Summary
    summary = (
        f"Model fitted successfully! "
        f"Observations: {mmm.n_obs}, Channels: {mmm.n_channels}. "
        f"Trend: {trend_type}, Seasonality: yearly={yearly}/monthly={monthly}/weekly={weekly}, "
        f"Inference: {chains} chains × {draws} draws."
    )

    # 6. Report (best-effort)
    report_path = "agent_mmm_report.html"
    try:
        from mmm_framework.reporting.generator import ReportBuilder

        report = ReportBuilder().with_model(mmm, results).enable_all_sections().build()
        report.to_html(report_path)
        summary += f" Full HTML report generated at {report_path}."
    except Exception as e:  # noqa: BLE001
        summary += f" Note: Report generation failed: {str(e)}"
        report_path = None

    # 7. Auto-save to disk
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{run_id}"
    model_path = _os.path.join(_MODELS_DIR, run_name)
    model_saved = False
    try:
        _os.makedirs(model_path, exist_ok=True)
        from mmm_framework.serialization import MMMSerializer

        MMMSerializer.save(mmm, model_path)
        model_saved = True
        summary += f" Auto-saved as **{run_name}**."
    except Exception as save_err:  # noqa: BLE001
        summary += f" (Auto-save failed: {save_err})"

    # 8. Run record
    model_run = {
        "run_id": run_id,
        "run_name": run_name,
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "dataset_path": dataset_path,
        "kpi": spec.get("kpi", ""),
        "channels": [m["name"] for m in spec.get("media_channels", [])],
        "controls": [c["name"] for c in spec.get("control_variables", [])],
        "trend": trend_type,
        "seasonality": {"yearly": yearly, "monthly": monthly, "weekly": weekly},
        "inference": {
            "chains": chains,
            "draws": draws,
            "tune": tune,
            "target_accept": target_accept,
        },
        "model_path": model_path if model_saved else None,
        "report_path": report_path,
        "summary": summary,
        "n_obs": int(mmm.n_obs),
        "n_channels": int(mmm.n_channels),
        # the full normalized spec — so a cold kernel can rebuild the panel and
        # reload this model from disk (PR-C.3 cold-reload).
        "spec": spec,
    }
    if spec.get("experiments"):
        model_run["calibration"] = {
            "experiments": spec.get("experiments"),
            "experiment_ids": spec.get("experiment_ids", []),
        }
        summary += (
            f" Calibrated with {len(spec['experiments'])} experiment likelihood(s)."
        )
        model_run["summary"] = summary
    # 8b. History metrics snapshot (best-effort — a metrics failure must never
    # fail a fit). Kernel-side: model-only compute; the host enriches with
    # registry calibration status and persists to the run_metrics table.
    # Knob: inference.metrics_draws (0 disables).
    metrics_draws = int(inf.get("metrics_draws", 200))
    if metrics_draws > 0:
        try:
            from mmm_framework.planning.history import compute_run_metrics

            model_run["metrics"] = compute_run_metrics(
                mmm, max_draws=metrics_draws, random_seed=random_seed
            )
        except Exception as metrics_err:  # noqa: BLE001
            model_run["metrics_error"] = str(metrics_err)

    # 8c. Model-health snapshot (best-effort, same contract as 8b): sampler
    # convergence + prior→posterior learning verdicts, so the UI can show
    # whether to believe the fit — not just what it estimated.
    try:
        from mmm_framework.diagnostics import compute_fit_diagnostics

        model_run["diagnostics"] = compute_fit_diagnostics(mmm, results)
    except Exception as diag_err:  # noqa: BLE001
        model_run["diagnostics_error"] = str(diag_err)

    if model_saved:
        try:
            with open(_os.path.join(model_path, "run_metadata.json"), "w") as f:
                _json.dump(model_run, f, indent=2, default=str)
        except Exception:
            pass

    # 9. Dashboard payload (incl. ROI + decomposition enrichment)
    dashboard = {
        "model_status": "completed",
        "summary": summary,
        "model_run": model_run,
    }
    if report_path:
        dashboard["report_path"] = report_path
    try:
        from mmm_framework.reporting.helpers import compute_roi_with_uncertainty

        roi_df = compute_roi_with_uncertainty(mmm, hdi_prob=0.94)
        dashboard["roi_metrics"] = roi_df.to_dict(orient="records")
    except Exception:
        pass
    try:
        from mmm_framework.reporting.helpers import compute_component_decomposition

        decomp_list = compute_component_decomposition(mmm, include_time_series=False)
        dashboard["decomposition"] = [
            {
                "component": d.component,
                "total_contribution": float(d.total_contribution),
                "pct_of_total": float(d.pct_of_total),
            }
            for d in decomp_list
        ]
    except Exception:
        pass

    info = {
        "summary": summary,
        "report_path": report_path,
        "model_run": model_run,
        "dashboard": dashboard,
    }
    return mmm, results, info
