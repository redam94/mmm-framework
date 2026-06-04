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


def build_and_fit(spec: dict, dataset_path: str):
    """Build + fit + serialize. Returns ``(mmm, results, info)``. Raises on
    failure. ``spec`` must be normalized (bare-string vars already coerced)."""
    import json as _json
    import os as _os
    from datetime import datetime, timezone

    # 1. MFFConfig + panel
    mff_config = _mff_config_from_spec(spec)
    panel = load_mff(dataset_path, mff_config)

    # 2. Inference + model config
    inf = spec.get("inference", {})
    chains = int(inf.get("chains", 4))
    draws = int(inf.get("draws", 1000))
    tune = int(inf.get("tune", 1000))
    target_accept = float(inf.get("target_accept", 0.85))
    random_seed = int(inf.get("random_seed", 42))

    model_config_builder = (
        ModelConfigBuilder()
        .bayesian_numpyro()
        .with_chains(chains)
        .with_draws(draws)
        .with_tune(tune)
        .with_target_accept(target_accept)
    )
    season = spec.get("seasonality", {})
    yearly = int(season.get("yearly", 0))
    monthly = int(season.get("monthly", 0))
    weekly = int(season.get("weekly", 0))
    if yearly > 0 or monthly > 0 or weekly > 0:
        sb = SeasonalityConfigBuilder()
        if yearly > 0:
            sb.with_yearly(order=yearly)
        if monthly > 0:
            sb.with_monthly(order=monthly)
        if weekly > 0:
            sb.with_weekly(order=weekly)
        model_config_builder.with_seasonality_builder(sb)
    model_config = model_config_builder.build()

    # 3. Trend config
    trend_spec = spec.get("trend", {})
    trend_type = trend_spec.get("type", "linear").lower()
    trend_prior_cfg = spec.get("priors", {}).get("trend", {})
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
    else:  # linear
        tb.linear()
        if (
            "growth_prior_mu" in trend_prior_cfg
            or "growth_prior_sigma" in trend_prior_cfg
        ):
            tb.with_growth_prior(
                mu=float(trend_prior_cfg.get("growth_prior_mu", 0.0)),
                sigma=float(trend_prior_cfg.get("growth_prior_sigma", 0.1)),
            )
    trend_config = tb.build()

    # 4. Fit
    mmm = BayesianMMM(panel, model_config, trend_config)
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
    }
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
