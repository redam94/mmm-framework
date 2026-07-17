"""Spec-curve / Bayesian model-averaging over a pre-registered spec set (#103).

There are dozens of defensible MMM specifications — adstock form, saturation
form, control set, pooling scheme — and each can move a channel's ROI. Landing
on the single spec that gives the "nice" answer is the garden of forking paths.
The honest alternative is to declare a **set** of defensible specs *before*
seeing results, fit them all, and report how each channel's ROI moves across the
whole set. Robustness across specs is itself a headline; fragility across specs
is a warning the single-spec number hides.

This module provides:

* :class:`SpecVariant` / :class:`SpecSet` — a serializable, **pre-registerable**
  declaration of the spec set (store it in the design-readout / assumption log
  before the fit).
* :func:`apply_variant` — deep-merge a variant onto a base spec (the same spec
  dict :func:`mmm_framework.agents.fitting.build_and_fit` consumes).
* :func:`default_spec_variants` — the standard defensible grid (adstock ×
  saturation forms, optionally pooling / prior mode) when the caller does not
  hand-pick one.
* :func:`run_spec_curve` — fit/collect every spec's per-channel ROI posterior,
  compute LOO-stacking weights, and produce a model-averaged (BMA) estimate with
  propagated uncertainty.
* :class:`SpecCurveResult` — the per-spec ROI table, stacking weights, the BMA
  estimate, and a per-channel robustness summary; ``to_dict()`` feeds the report
  robustness section.

The ROI is computed with the framework's canonical semantics (per-channel
contribution draws over the full window ÷ the measurement-aware divisor), so a
spec-curve number is directly comparable to the ``contribution_roi`` estimand.
Fitting and ROI extraction are injectable (``fit_fn`` / ``roi_fn``) so the engine
is unit-testable without MCMC.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

__all__ = [
    "SpecVariant",
    "SpecSet",
    "SpecFit",
    "SpecCurveResult",
    "apply_variant",
    "default_spec_variants",
    "run_spec_curve",
    "channel_roi_draws",
]


# =============================================================================
# Pre-registerable spec-set declaration
# =============================================================================
class SpecVariant(BaseModel):
    """One defensible specification, expressed as overrides on a base spec.

    The structured axes cover the four the review calls out (adstock form,
    saturation form, control set, pooling); ``overrides`` is an escape hatch for
    any other spec key (deep-merged last).
    """

    name: str
    #: Adstock family applied to every media channel ("geometric"/"weibull"/"delayed").
    adstock: str | None = None
    #: Saturation family applied to every media channel ("hill"/"logistic"/"michaelis_menten"/"tanh").
    saturation: str | None = None
    #: Full replacement control set (channel-name list). ``None`` keeps the base's.
    controls: list[str] | None = None
    #: Pooling scheme ("national" / "geo").
    kpi_level: str | None = None
    #: Default media-prior parameterization ("roi" / "coefficient").
    media_prior_mode: str | None = None
    #: Trend family ("none"/"linear"/"piecewise"/"spline"/"gaussian_process").
    trend: str | None = None
    #: Seasonality override, e.g. ``{"yearly": 4}`` (or ``{}`` to disable).
    seasonality: dict[str, Any] | None = None
    #: Arbitrary deep-merge overrides (applied last).
    overrides: dict[str, Any] = Field(default_factory=dict)
    #: Whether this is the pre-registered PRIMARY specification.
    primary: bool = False
    description: str = ""


class SpecSet(BaseModel):
    """A pre-registered set of defensible specs, declared before results.

    Serialize this (``model_dump``) into the design-readout / assumption log
    *before* fitting so the robustness report is auditable — the spec set was
    fixed in advance, not chosen after seeing which answer looked nicest.
    """

    variants: list[SpecVariant] = Field(default_factory=list)
    rationale: str = ""
    #: ISO timestamp of pre-registration (set by the caller; kept as data, not
    #: computed here so the module stays deterministic / clock-free).
    registered_at: str | None = None

    @property
    def primary_variant(self) -> SpecVariant | None:
        for v in self.variants:
            if v.primary:
                return v
        return self.variants[0] if self.variants else None

    def names(self) -> list[str]:
        return [v.name for v in self.variants]


# =============================================================================
# Variant application
# =============================================================================
def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge ``overrides`` into a deep copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_variant(base_spec: dict, variant: SpecVariant) -> dict:
    """Return a new spec = ``base_spec`` with ``variant`` applied.

    The structured axes are applied first (adstock/saturation forms are set on
    *every* media channel; controls/pooling/prior-mode/trend/seasonality set the
    corresponding top-level keys), then ``variant.overrides`` is deep-merged.
    """
    spec = copy.deepcopy(base_spec)

    if variant.adstock is not None:
        for m in spec.get("media_channels", []):
            m.setdefault("adstock", {})["type"] = variant.adstock
    if variant.saturation is not None:
        for m in spec.get("media_channels", []):
            m.setdefault("saturation", {})["type"] = variant.saturation
    if variant.controls is not None:
        spec["control_variables"] = [{"name": c} for c in variant.controls]
    if variant.kpi_level is not None:
        spec["kpi_level"] = variant.kpi_level
    if variant.media_prior_mode is not None:
        spec["media_prior_mode"] = variant.media_prior_mode
    if variant.trend is not None:
        trend = spec.get("trend")
        spec["trend"] = (
            {"type": variant.trend}
            if not isinstance(trend, dict)
            else {
                **trend,
                "type": variant.trend,
            }
        )
    if variant.seasonality is not None:
        spec["seasonality"] = copy.deepcopy(variant.seasonality)
    if variant.overrides:
        spec = _deep_merge(spec, variant.overrides)
    return spec


def _base_forms(base_spec: dict) -> tuple[str, str]:
    """The base spec's dominant adstock / saturation form (from the first channel)."""
    media = base_spec.get("media_channels") or []
    adstock = "geometric"
    saturation = "hill"
    if media:
        adstock = (media[0].get("adstock") or {}).get("type", "geometric").lower()
        saturation = (media[0].get("saturation") or {}).get("type", "hill").lower()
    return adstock, saturation


def default_spec_variants(
    base_spec: dict,
    *,
    adstock_forms: Sequence[str] = ("geometric", "weibull"),
    saturation_forms: Sequence[str] = ("hill", "logistic"),
    include_prior_mode: bool = False,
) -> list[SpecVariant]:
    """The standard defensible spec grid: adstock × saturation forms.

    The base spec's own (adstock, saturation) combination is marked ``primary``.
    ``include_prior_mode`` adds a coefficient-scale-prior sibling of the primary
    (the other big modelling fork). Callers may of course hand-write their own
    :class:`SpecVariant` list instead.
    """
    base_ad, base_sat = _base_forms(base_spec)
    variants: list[SpecVariant] = []
    for ad in adstock_forms:
        for sat in saturation_forms:
            is_primary = ad == base_ad and sat == base_sat
            variants.append(
                SpecVariant(
                    name=f"{ad}×{sat}",
                    adstock=ad,
                    saturation=sat,
                    primary=is_primary,
                    description=f"{ad} adstock, {sat} saturation",
                )
            )
    # Guarantee exactly one primary (the base combo may be outside the grid).
    if not any(v.primary for v in variants) and variants:
        variants[0].primary = True
    if include_prior_mode:
        variants.append(
            SpecVariant(
                name=f"{base_ad}×{base_sat} · coefficient-prior",
                adstock=base_ad,
                saturation=base_sat,
                media_prior_mode="coefficient",
                description="Primary forms with coefficient-scale (not ROI) media priors",
            )
        )
    return variants


# =============================================================================
# ROI extraction (canonical framework semantics)
# =============================================================================
def channel_roi_draws(
    model: Any,
    channels: Sequence[str],
    *,
    max_draws: int = 400,
    random_seed: int = 42,
) -> dict[str, np.ndarray]:
    """Per-channel ROI posterior draws over the full window.

    Uses ``model.sample_channel_contributions`` (KPI-unit per-draw contributions)
    summed over observations, divided by the measurement-aware divisor
    (``resolve_channel_divisor`` — spend $ or efficiency volume) — the same
    numerator/denominator as the ``contribution_roi`` estimand. Returns
    ``{channel: (n_draws,) ndarray}``; a channel with a non-positive divisor is
    omitted.
    """
    from ..reporting.helpers.measurement import resolve_channel_divisor

    contrib = model.sample_channel_contributions(
        max_draws=max_draws, random_seed=random_seed
    )  # (D, n_obs, C)
    contrib = np.asarray(contrib, dtype=float)
    out: dict[str, np.ndarray] = {}
    for i, ch in enumerate(channels):
        if i >= contrib.shape[2]:
            continue
        div = resolve_channel_divisor(model, ch)
        total = float(getattr(div, "total", 0.0) or 0.0)
        if not np.isfinite(total) or total <= 0:
            continue
        out[str(ch)] = contrib[:, :, i].sum(axis=1) / total
    return out


def _eti(draws: np.ndarray, hdi_prob: float) -> tuple[float, float]:
    lo_q = (1.0 - hdi_prob) / 2.0 * 100.0
    d = draws[np.isfinite(draws)]
    if d.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(d, lo_q)), float(np.percentile(d, 100.0 - lo_q))


def _summ(draws: np.ndarray, hdi_prob: float) -> dict[str, float]:
    d = draws[np.isfinite(draws)]
    lo, hi = _eti(draws, hdi_prob)
    return {
        "mean": float(d.mean()) if d.size else float("nan"),
        "lower": lo,
        "upper": hi,
    }


# =============================================================================
# Runner
# =============================================================================
@dataclass
class SpecFit:
    """One spec's realized ROI + model-fit summary."""

    name: str
    primary: bool
    roi: dict[str, dict[str, float]] = field(default_factory=dict)
    roi_draws: dict[str, np.ndarray] = field(default_factory=dict)
    loo: dict[str, float] | None = None
    weight: float = 0.0
    error: str | None = None


@dataclass
class SpecCurveResult:
    """Spec-curve outcome: per-spec ROI, stacking weights, BMA, robustness."""

    channels: list[str]
    specs: list[str]
    primary: str | None
    hdi_prob: float
    fits: list[SpecFit]
    weights: dict[str, float]
    bma: dict[str, dict[str, float]]
    robustness: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly payload for the report bundle / agent tool (no draws)."""
        return {
            "channels": self.channels,
            "specs": self.specs,
            "primary": self.primary,
            "hdi_prob": self.hdi_prob,
            "weights": self.weights,
            "bma": self.bma,
            "robustness": self.robustness,
            "per_spec": {
                f.name: {
                    "primary": f.primary,
                    "roi": f.roi,
                    "loo": f.loo,
                    "weight": f.weight,
                    "error": f.error,
                }
                for f in self.fits
            },
        }


def _default_fit(spec: dict, dataset_path: str) -> Any:
    """Build + fit a model from a spec (no serialization side effects).

    Uses the plain ``build_model`` + ``fit`` path so a spec-curve sweep does not
    register N models / run-history rows. Inference args come from ``spec``.
    """
    from ..agents.fitting import build_model

    model = build_model(spec, dataset_path)
    inf = spec.get("inference", {}) or {}
    method = str(inf.get("method", "nuts")).lower()
    kwargs: dict[str, Any] = {"random_seed": int(inf.get("random_seed", 42))}
    if method == "nuts":
        kwargs.update(
            draws=int(inf.get("draws", 1000)),
            tune=int(inf.get("tune", 1000)),
            chains=int(inf.get("chains", 4)),
            target_accept=float(inf.get("target_accept", 0.85)),
        )
    elif method == "smc":
        # Exact SMC: draws = particles per run, chains = independent runs.
        kwargs.update(
            method="smc",
            draws=int(inf.get("draws", 1000)),
            chains=int(inf.get("chains", 4)),
        )
    else:
        kwargs["method"] = method
    model.fit(**kwargs)
    return model


def _loo_summary(model: Any) -> dict[str, float] | None:
    """LOO summary for a fitted model, computing pointwise log-likelihood first.

    Returns ``{"elpd_loo", "se", "p_loo", "n_bad_k"}`` and stores the LOO-ready
    idata on ``model._trace`` (so :func:`_stacking_weights` can reuse it), or
    ``None`` if LOO cannot be computed.
    """
    import arviz as az

    trace = getattr(model, "_trace", None)
    graph = getattr(model, "model", None)
    if trace is None:
        return None
    try:
        if getattr(trace, "log_likelihood", None) is None and graph is not None:
            import pymc as pm

            with graph:
                pm.compute_log_likelihood(trace)
        loo = az.loo(trace, pointwise=True)
        n_bad = (
            int((np.asarray(loo.pareto_k.values) > 0.7).sum())
            if hasattr(loo, "pareto_k")
            else 0
        )
        return {
            "elpd_loo": float(loo.elpd_loo),
            "se": float(loo.se),
            "p_loo": float(loo.p_loo),
            "n_bad_k": n_bad,
        }
    except Exception as e:  # noqa: BLE001 — LOO is best-effort
        logger.warning(f"spec-curve LOO failed: {e}")
        return None


def _stacking_weights(models: dict[str, Any], names: list[str]) -> dict[str, float]:
    """LOO-stacking weights over the fitted models (Yao et al. 2018).

    Falls back to equal weights when ``az.compare`` cannot run (missing
    log-likelihood, single model, or an arviz error).
    """
    import arviz as az

    usable = {n: m for n, m in models.items() if getattr(m, "_trace", None) is not None}
    if len(usable) < 2:
        return {n: (1.0 if n in usable else 0.0) for n in names}
    compare_dict = {}
    for n, m in usable.items():
        trace = m._trace
        try:
            if getattr(trace, "log_likelihood", None) is None and getattr(
                m, "model", None
            ):
                import pymc as pm

                with m.model:
                    pm.compute_log_likelihood(trace)
            compare_dict[n] = trace
        except Exception:  # noqa: BLE001
            continue
    if len(compare_dict) < 2:
        eq = 1.0 / max(len(usable), 1)
        return {n: (eq if n in usable else 0.0) for n in names}
    try:
        cmp = az.compare(compare_dict, method="stacking")
        weights = {str(idx): float(row["weight"]) for idx, row in cmp.iterrows()}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"spec-curve stacking failed ({e}); using equal weights")
        eq = 1.0 / len(compare_dict)
        weights = {n: eq for n in compare_dict}
    return {n: float(weights.get(n, 0.0)) for n in names}


def _bma_mixture(
    fits: list[SpecFit],
    weights: dict[str, float],
    channels: list[str],
    hdi_prob: float,
    n_draws: int,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    """Model-averaged ROI per channel: a stacking-weighted mixture of the specs'
    ROI posteriors (propagates within- AND between-spec uncertainty)."""
    bma: dict[str, dict[str, float]] = {}
    for ch in channels:
        parts: list[np.ndarray] = []
        for f in fits:
            w = weights.get(f.name, 0.0)
            d = f.roi_draws.get(ch)
            if w <= 0 or d is None or len(d) == 0:
                continue
            k = int(round(w * n_draws))
            if k <= 0:
                continue
            parts.append(rng.choice(d, size=k, replace=True))
        if parts:
            mix = np.concatenate(parts)
            bma[ch] = _summ(mix, hdi_prob)
    return bma


def _robustness(
    fits: list[SpecFit],
    weights: dict[str, float],
    channels: list[str],
    primary: str | None,
) -> dict[str, dict[str, Any]]:
    """Per-channel robustness across specs: the spread of point ROIs, whether the
    sign is stable, and how far the primary sits from the model-average."""
    out: dict[str, dict[str, Any]] = {}
    for ch in channels:
        means = [
            (f.name, f.roi[ch]["mean"])
            for f in fits
            if ch in f.roi and np.isfinite(f.roi[ch].get("mean", np.nan))
        ]
        if not means:
            continue
        vals = np.array([m for _, m in means], dtype=float)
        primary_mean = next((m for n, m in means if n == primary), None)
        # Weighted spread: sign-stability weighted by stacking mass.
        pos_mass = sum(
            weights.get(n, 0.0) for n, m in means if m > 1.0
        )  # ROI reference ~1.0
        total_mass = sum(weights.get(n, 0.0) for n, _ in means) or 1.0
        out[ch] = {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "range": float(vals.max() - vals.min()),
            "spread_pct": (
                float((vals.max() - vals.min()) / abs(vals.mean()) * 100.0)
                if abs(vals.mean()) > 1e-9
                else None
            ),
            "sign_stable": bool(np.all(vals > 1.0) or np.all(vals < 1.0)),
            "profitable_weight": float(pos_mass / total_mass),
            "primary": primary_mean,
            "n_specs": len(means),
        }
    return out


def run_spec_curve(
    base_spec: dict,
    dataset_path: str,
    *,
    variants: Sequence[SpecVariant] | SpecSet | None = None,
    hdi_prob: float = 0.94,
    max_draws: int = 400,
    random_seed: int = 42,
    compute_loo: bool = True,
    fit_fn: Callable[[dict, str], Any] | None = None,
    roi_fn: Callable[..., dict[str, np.ndarray]] | None = None,
) -> SpecCurveResult:
    """Fit a spec set, collect per-channel ROI, and model-average via LOO-stacking.

    Parameters
    ----------
    base_spec, dataset_path:
        The base spec and the dataset file each variant is fit against.
    variants:
        A :class:`SpecSet`, a list of :class:`SpecVariant`, or ``None`` (uses
        :func:`default_spec_variants`).
    hdi_prob, max_draws, random_seed:
        Credible-interval mass, ROI-draw thinning, and seed.
    compute_loo:
        Compute LOO + stacking weights (else equal weights — a plain spec-curve).
    fit_fn, roi_fn:
        Injection points for testing. ``fit_fn(spec, dataset_path) -> model``
        (default: build + fit, no serialization); ``roi_fn(model, channels,
        max_draws=, random_seed=) -> {channel: draws}`` (default:
        :func:`channel_roi_draws`).
    """
    if isinstance(variants, SpecSet):
        variant_list = list(variants.variants)
    elif variants is None:
        variant_list = default_spec_variants(base_spec)
    else:
        variant_list = list(variants)
    if not variant_list:
        raise ValueError("run_spec_curve needs at least one spec variant.")

    fit_fn = fit_fn or _default_fit
    roi_fn = roi_fn or channel_roi_draws
    rng = np.random.default_rng(random_seed)

    channel_order: list[str] = []
    fits: list[SpecFit] = []
    models: dict[str, Any] = {}
    primary_name = next(
        (v.name for v in variant_list if v.primary), variant_list[0].name
    )

    for variant in variant_list:
        fit = SpecFit(name=variant.name, primary=variant.primary)
        try:
            spec = apply_variant(base_spec, variant)
            model = fit_fn(spec, dataset_path)
            channels = [str(c) for c in getattr(model, "channel_names", [])]
            for ch in channels:
                if ch not in channel_order:
                    channel_order.append(ch)
            draws = roi_fn(
                model, channels, max_draws=max_draws, random_seed=random_seed
            )
            fit.roi_draws = draws
            fit.roi = {ch: _summ(d, hdi_prob) for ch, d in draws.items()}
            if compute_loo:
                fit.loo = _loo_summary(model)
            models[variant.name] = model
        except Exception as e:  # noqa: BLE001 — one bad spec must not sink the sweep
            logger.warning(f"spec-curve variant {variant.name!r} failed: {e}")
            fit.error = str(e)
        fits.append(fit)

    weights = _stacking_weights(models, [f.name for f in fits]) if compute_loo else {}
    if not weights:
        ok = [f.name for f in fits if not f.error]
        eq = 1.0 / len(ok) if ok else 0.0
        weights = {f.name: (eq if not f.error else 0.0) for f in fits}
    for f in fits:
        f.weight = weights.get(f.name, 0.0)

    bma = _bma_mixture(fits, weights, channel_order, hdi_prob, max_draws, rng)
    robustness = _robustness(fits, weights, channel_order, primary_name)

    return SpecCurveResult(
        channels=channel_order,
        specs=[f.name for f in fits],
        primary=primary_name,
        hdi_prob=hdi_prob,
        fits=fits,
        weights=weights,
        bma=bma,
        robustness=robustness,
    )
