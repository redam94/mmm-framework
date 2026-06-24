"""Post-hoc estimand realization from a fitted posterior.

:class:`EstimandEvaluator` walks an :class:`~mmm_framework.estimands.spec.Estimand`
and realizes it as a mean + HDI (+ tail-probability summaries) from the fitted
model, via :meth:`BayesianMMM.predict_under` (per-draw posterior-predictive) and
the in-graph ``channel_contributions`` Deterministic.

Bit-stability is the contract: each built-in reproduces its legacy number to the
bit under a fixed seed. This is achieved by (a) reusing the *exact* legacy
sample-extraction helpers for the decomposition path and (b) per-estimand
realization knobs (point arithmetic, HDI function, seed pairing, spend source) —
see :class:`~mmm_framework.estimands.spec.Realization` and
``technical-docs/estimands.md``.

This engine is **numpy-only and post-hoc**; the in-graph (pytensor) realization
lives in :mod:`mmm_framework.estimands.graph` and the two never call into each
other (they share only ``spec.py`` + the tiny window/reduce helpers here).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .capabilities import (
    IN_GRAPH_RESPONSE_CURVE,
    missing_capabilities,
    model_capabilities,
)
from .spec import (
    ALL_CHANNELS,
    Constant,
    Contrast,
    Contribution,
    Estimand,
    EstimandResult,
    LatentVar,
    MarginalSpend,
    Observed,
    ObservedInput,
    Outcome,
    ZeroInput,
)

# A node evaluates to (point, samples): ``samples`` is a per-draw 1-D array, or
# ``None`` for a deterministic scalar (e.g. observed spend).
_NodeValue = tuple[float, "np.ndarray | None"]


def _reduce(arr: np.ndarray, how: str, axis: int) -> np.ndarray:
    return arr.mean(axis=axis) if how == "mean" else arr.sum(axis=axis)


def _apply_op(a, b, op: str):
    if op == "identity":
        return a
    if op == "difference":
        return a - b
    if op == "ratio":
        return a / b
    raise ValueError(f"Unknown contrast op: {op!r}")


class EstimandEvaluator:
    """Realize estimands against a fitted model.

    Parameters
    ----------
    model:
        A fitted ``BayesianMMM`` (or anything implementing
        :class:`~mmm_framework.estimands.spec.SupportsEstimands`).
    random_seed:
        Seed threaded to every ``predict_under`` call. ``None`` reproduces the
        legacy default (unpaired counterfactual draws; a single synthesized seed
        for paired/marginal contrasts).
    """

    def __init__(self, model: Any, *, random_seed: int | None = None):
        self.model = model
        self.random_seed = random_seed
        self.capabilities = model_capabilities(model)
        self._pred_cache: dict[tuple[str, int | None], Any] = {}
        self._batch_seed: int | None = None  # synthesized once for paired contrasts

    # -- public ---------------------------------------------------------------

    def evaluate(self, estimands: list[Estimand]) -> dict[str, EstimandResult]:
        """Realize ``estimands``; wildcard-target estimands expand per channel."""
        results: dict[str, EstimandResult] = {}
        for est in estimands:
            expanded = list(self._expand(est))
            # Cross-channel contribution % needs the full channel set, so it is a
            # post-reduction step (cf. compute_counterfactual_contributions).
            contrib_points: dict[str, float] = {}
            local: dict[str, EstimandResult] = {}
            for concrete, key, channel in expanded:
                res = self._evaluate_one(concrete, key)
                if res is None:
                    continue
                local[key] = res
                if res.status == "ok" and self._is_contribution_numerator(concrete):
                    cp = res.extra.get("contribution_mean")
                    if cp is not None and channel is not None:
                        contrib_points[key] = float(cp)
            total = sum(contrib_points.values())
            for key, res in local.items():
                if key in contrib_points and total != 0:
                    res.extra["contribution_pct"] = contrib_points[key] / total * 100.0
                results[key] = res
        return results

    # -- expansion ------------------------------------------------------------

    @staticmethod
    def _has_wildcard(est: Estimand) -> bool:
        return ALL_CHANNELS in _collect_targets(est.model_dump())

    def _expand(self, est: Estimand):
        """Yield ``(estimand, result_key, channel)`` tuples."""
        if not self._has_wildcard(est):
            yield est, est.name, None
            return
        for channel in getattr(self.model, "channel_names", []) or []:
            concrete = Estimand.model_validate(
                _substitute_target(est.model_dump(), channel)
            )
            yield concrete, f"{est.name}:{channel}", channel

    @staticmethod
    def _is_contribution_numerator(est: Estimand) -> bool:
        num = est.numerator
        return (
            isinstance(num, Contrast)
            and isinstance(num.quantity, Outcome)
            and num.op == "difference"
        )

    # -- single estimand ------------------------------------------------------

    def _evaluate_one(self, est: Estimand, key: str) -> EstimandResult | None:
        missing = missing_capabilities(est.required_capabilities, self.capabilities)
        if missing or IN_GRAPH_RESPONSE_CURVE in est.required_capabilities:
            reason = (
                "off-panel (in-graph response curve) estimands are graph-only"
                if IN_GRAPH_RESPONSE_CURVE in est.required_capabilities
                else f"model is missing capabilities: {', '.join(missing)}"
            )
            return EstimandResult(
                name=key,
                kind=est.kind,
                status="unsupported",
                reason=reason,
                units=est.units,
                hdi_prob=est.hdi_prob,
            )

        mask = self._window_mask(est)
        marginal_factor = self._marginal_factor(est)

        try:
            num_point, num_samples = self._eval_node(
                est.numerator, mask, marginal_factor
            )
        except _SkipChannel as exc:
            return (
                None
                if exc.skip
                else EstimandResult(
                    name=key,
                    kind=est.kind,
                    status="unsupported",
                    reason=str(exc),
                    units=est.units,
                    hdi_prob=est.hdi_prob,
                )
            )

        extra: dict[str, Any] = {"numerator_mean": num_point}
        num_hdi = (
            self._hdi(num_samples, est) if num_samples is not None else (None, None)
        )
        if num_samples is not None:
            extra["numerator_hdi_low"], extra["numerator_hdi_high"] = num_hdi
            # Friendly aliases for the contribution-numerator case.
            extra["contribution_mean"] = num_point
            extra["contribution_hdi_low"], extra["contribution_hdi_high"] = num_hdi

        if est.denominator is None:
            result_point = num_point
            result_samples = num_samples
        else:
            den_point, den_samples = self._eval_node(
                est.denominator, mask, marginal_factor
            )
            extra["spend"] = den_point if den_samples is None else None
            result_point, result_samples, skip = self._combine_ratio(
                est, num_point, num_samples, den_point, den_samples
            )
            if skip:
                return None

        hdi_low, hdi_high = self._result_hdi(est, result_point, result_samples)
        result = EstimandResult(
            name=key,
            kind=est.kind,
            status="ok",
            mean=_finite_or_none(result_point),
            hdi_low=hdi_low,
            hdi_high=hdi_high,
            hdi_prob=est.hdi_prob,
            units=est.units,
            extra=extra,
        )
        self._apply_summaries(est, result, result_samples)
        return result

    # -- ratio combination ----------------------------------------------------

    def _combine_ratio(
        self,
        est: Estimand,
        num_point: float,
        num_samples: "np.ndarray | None",
        den_point: float,
        den_samples: "np.ndarray | None",
    ) -> tuple[float, "np.ndarray | None", bool]:
        """Return ``(point, samples, skip)`` for ``numerator / denominator``."""
        zero_den = (
            den_point is None
            or den_point == 0
            or (den_samples is None and not np.isfinite(den_point))
        )
        if den_samples is None and (den_point is None or den_point <= 0):
            zero_den = True

        if zero_den:
            mode = est.op_ratio_zero_denominator
            if mode == "skip":
                return 0.0, None, True
            if mode == "nan":
                return float("nan"), None, False
            # "zero": marginal/counterfactual fall back to a 0 point estimate.
            return 0.0, None, False

        denom = den_samples if den_samples is not None else den_point
        with np.errstate(divide="ignore", invalid="ignore"):
            result_samples = num_samples / denom if num_samples is not None else None
            if result_samples is None and den_samples is not None:
                # deterministic numerator / per-draw denominator (cost-per-X)
                result_samples = num_point / den_samples

        if est.realization.point_rule == "mean_of_samples":
            result_point = float(np.mean(result_samples))
        else:  # diff_of_means: point from the reduced per-call means
            result_point = num_point / den_point
        return result_point, result_samples, False

    # -- node evaluation ------------------------------------------------------

    def _eval_node(
        self, node, mask: np.ndarray, marginal_factor: float | None
    ) -> _NodeValue:
        if isinstance(node, Contrast):
            return self._eval_contrast(node, mask)
        return self._eval_quantity(node, mask, marginal_factor)

    def _eval_contrast(self, contrast: Contrast, mask: np.ndarray) -> _NodeValue:
        quantity = contrast.quantity
        baseline = contrast.baseline or Observed()
        if isinstance(quantity, Outcome):
            seed_i, seed_b = self._seeds_for(contrast)
            pred_i = self._predict_cached(contrast.intervention, seed_i)
            pred_b = self._predict_cached(baseline, seed_b)
            mi = _reduce(pred_i.y_pred_mean[mask], contrast.reduce, axis=0)
            mb = _reduce(pred_b.y_pred_mean[mask], contrast.reduce, axis=0)
            si = _reduce(pred_i.y_pred_samples[:, mask], contrast.reduce, axis=1)
            sb = _reduce(pred_b.y_pred_samples[:, mask], contrast.reduce, axis=1)
            return float(_apply_op(mi, mb, contrast.op)), _apply_op(si, sb, contrast.op)
        if isinstance(quantity, LatentVar):
            return self._eval_latent_contrast(contrast, quantity, baseline, mask)
        raise _SkipChannel(
            f"contrast over quantity {quantity.type!r} is not supported post-hoc",
            skip=False,
        )

    def _eval_latent_contrast(self, contrast, quantity, baseline, mask) -> _NodeValue:
        """A latent-variable contrast: the named deterministic ``quantity.name``
        re-evaluated under the intervention vs the baseline (via the model's
        :meth:`~mmm_framework.model.base.BayesianMMM.sample_latent_under`), reduced
        over the window and combined by the contrast op. The latent is in its
        native (model) scale; the contrast cancels any constant scaling."""
        model = self.model
        if not callable(getattr(model, "sample_latent_under", None)):
            raise _SkipChannel(
                f"model cannot realize latent contrasts for {quantity.name!r} "
                "(no sample_latent_under)",
                skip=False,
            )
        seed_i, seed_b = self._seeds_for(contrast)
        try:
            di = np.asarray(
                model.sample_latent_under(
                    quantity.name, contrast.intervention, random_seed=seed_i
                )
            )
            db = np.asarray(
                model.sample_latent_under(quantity.name, baseline, random_seed=seed_b)
            )
        except Exception as exc:  # noqa: BLE001 — degrade, never crash the engine
            raise _SkipChannel(
                f"latent contrast for {quantity.name!r} failed: {exc}", skip=False
            ) from exc

        def _collapse(d: np.ndarray) -> np.ndarray:
            if d.ndim == 1:  # scalar-per-draw latent
                return d
            if d.ndim == 2 and d.shape[1] == len(mask):  # obs-indexed
                return _reduce(d[:, mask], contrast.reduce, axis=1)
            raise _SkipChannel(
                f"latent {quantity.name!r} has trailing shape {d.shape[1:]}; latent "
                "contrasts support a scalar or obs-indexed (n_obs,) deterministic — "
                "target a named scalar element for higher dims",
                skip=False,
            )

        si, sb = _collapse(di), _collapse(db)
        samples = _apply_op(si, sb, contrast.op)
        return float(np.mean(samples)), samples

    def _eval_quantity(
        self, quantity, mask: np.ndarray, marginal_factor: float | None
    ) -> _NodeValue:
        if isinstance(quantity, Constant):
            return float(quantity.value), None
        if isinstance(quantity, ObservedInput):
            return self._observed_spend(quantity, mask), None
        if isinstance(quantity, MarginalSpend):
            return self._marginal_spend(quantity, mask, marginal_factor), None
        if isinstance(quantity, Contribution):
            return self._contribution_quantity(quantity, mask)
        if isinstance(quantity, Outcome):
            pred = self._predict_cached(Observed(), self._single_seed())
            return (
                float(pred.y_pred_mean[mask].sum()),
                pred.y_pred_samples[:, mask].sum(axis=1),
            )
        if isinstance(quantity, LatentVar):
            return self._latent_quantity(quantity, mask)
        raise _SkipChannel(
            f"quantity {getattr(quantity, 'type', quantity)!r} unsupported", skip=False
        )

    def _latent_quantity(self, q: LatentVar, mask: np.ndarray) -> _NodeValue:
        """Realize a **bare** latent posterior variable as (mean, per-draw samples).

        Reads ``posterior[name]`` and collapses chain×draw to the sample axis. A
        per-draw **scalar** (e.g. a fit index ``cfi``/``srmr``, or a named scalar
        loading) → mean + HDI directly; an **obs-indexed** latent → mean over the
        window. A vector/matrix latent (e.g. a full loadings matrix) is *not* a
        single estimand — it is surfaced as a table — so it returns ``unsupported``.
        A latent used in a *contrast* (intervention vs baseline) is realized by
        :meth:`_eval_latent_contrast` instead."""
        from mmm_framework.reporting.helpers.utils import _get_posterior

        posterior = _get_posterior(self.model)
        if posterior is None or q.name not in getattr(posterior, "data_vars", {}):
            raise _SkipChannel(
                f"latent variable {q.name!r} is not in the posterior", skip=False
            )
        arr = np.asarray(posterior[q.name].values)
        if arr.ndim < 2:  # arviz posterior vars are (chain, draw, *rest)
            raise _SkipChannel(
                f"latent variable {q.name!r} posterior is not (chain, draw, …)",
                skip=False,
            )
        rest = arr.shape[2:]
        values = arr.reshape(arr.shape[0] * arr.shape[1], *rest)  # (n_samples, *rest)
        if len(rest) == 0:
            samples = values
        elif len(rest) == 1 and rest[0] == 1:
            samples = values[:, 0]
        elif len(rest) == 1 and rest[0] == len(mask):
            samples = values[:, mask].mean(axis=1)  # obs-indexed -> window mean
        else:
            raise _SkipChannel(
                f"latent variable {q.name!r} is array-valued (shape {rest}) — "
                "matrix/vector latents (e.g. a loadings matrix) are surfaced as a "
                "table, not a single estimand; target a named scalar element",
                skip=False,
            )
        samples = np.asarray(samples, dtype=float).ravel()
        return float(np.mean(samples)), samples

    # -- quantity helpers -----------------------------------------------------

    def _observed_spend(self, q: ObservedInput, mask: np.ndarray) -> float:
        if q.source == "panel":
            from mmm_framework.reporting.helpers.roi import _extract_spend_from_model

            return float(_extract_spend_from_model(self.model).get(q.target, 0.0))
        idx = self.model.channel_names.index(q.target)
        return float(self.model.X_media_raw[mask, idx].sum())

    def _marginal_spend(
        self, q: MarginalSpend, mask: np.ndarray, factor: float | None
    ) -> float:
        if factor is None:
            raise _SkipChannel(
                "marginal_spend requires a ScaleInput numerator intervention",
                skip=False,
            )
        idx = self.model.channel_names.index(q.target)
        current = float(self.model.X_media_raw[mask, idx].sum())
        return current * (factor - 1.0)

    def _contribution_quantity(self, q: Contribution, mask: np.ndarray) -> _NodeValue:
        if q.source == "in_graph_deterministic":
            from mmm_framework.reporting.helpers.roi import _get_contribution_samples
            from mmm_framework.reporting.helpers.utils import (
                _get_posterior,
                _get_scaling_params,
            )

            posterior = _get_posterior(self.model)
            y_mean, y_std = _get_scaling_params(self.model)
            samples = _get_contribution_samples(
                self.model, posterior, q.target, y_mean, y_std
            )
            if samples is None or len(samples) == 0:
                raise _SkipChannel(
                    f"no contribution samples for {q.target!r}", skip=True
                )
            return float(np.mean(samples)), np.asarray(samples)
        # counterfactual: synthesize the equivalent zero-out contrast
        contrast = Contrast(
            quantity=Outcome(),
            intervention=Observed(),
            baseline=ZeroInput(target=q.target),
            op="difference",
            reduce="sum",
        )
        return self._eval_contrast(contrast, mask)

    # -- prediction caching & seeds ------------------------------------------

    def _predict_cached(self, intervention, seed: int | None):
        key = (intervention.model_dump_json(), seed)
        if key not in self._pred_cache:
            self._pred_cache[key] = self.model.predict_under(
                intervention, random_seed=seed
            )
        return self._pred_cache[key]

    def _seeds_for(self, contrast: Contrast) -> tuple[int | None, int | None]:
        if self.random_seed is not None:
            return self.random_seed, self.random_seed
        if contrast.paired_seed:
            if self._batch_seed is None:
                self._batch_seed = int(np.random.default_rng().integers(0, 2**31 - 1))
            return self._batch_seed, self._batch_seed
        return None, None

    def _single_seed(self) -> int | None:
        return self.random_seed

    # -- windows, factors, HDI, summaries ------------------------------------

    def _window_mask(self, est: Estimand) -> np.ndarray:
        tp = est.window.as_tuple() if est.window else None
        return self.model._get_time_mask(tp)

    @staticmethod
    def _marginal_factor(est: Estimand) -> float | None:
        num = est.numerator
        if isinstance(num, Contrast):
            iv = num.intervention
            if getattr(iv, "type", None) == "scale_input":
                return float(iv.factor)
        return None

    def _hdi(self, samples: np.ndarray, est: Estimand) -> tuple[float, float]:
        method = est.realization.hdi_method
        if method == "az_hdi":
            from mmm_framework.reporting.helpers.utils import _compute_hdi

            return _compute_hdi(samples, est.hdi_prob)
        if method == "finite_percentile":
            from mmm_framework.model.base import _hdi_finite

            return _hdi_finite(samples, est.hdi_prob)
        from mmm_framework.utils import compute_hdi_bounds

        low, high = compute_hdi_bounds(samples, hdi_prob=est.hdi_prob, axis=0)
        return float(low), float(high)

    def _result_hdi(
        self, est: Estimand, point: float, samples: "np.ndarray | None"
    ) -> tuple[float | None, float | None]:
        # A zero-spend marginal ROAS reports a degenerate (0, 0) interval, not
        # (nan, nan) — matching compute_marginal_contributions.
        if samples is None:
            if est.kind == "marginal_roas" and point == 0.0:
                return 0.0, 0.0
            return None, None
        low, high = self._hdi(samples, est)
        return _finite_or_none(low), _finite_or_none(high)

    def _apply_summaries(self, est, result: EstimandResult, samples) -> None:
        if samples is None:
            return
        finite = samples[np.isfinite(samples)]
        if finite.size == 0:
            return
        for s in est.summaries:
            if s.side == "gt":
                p = float(np.mean(finite > s.threshold))
            else:
                p = float(np.mean(finite < s.threshold))
            result.extra[s.name] = p


# =============================================================================
# helpers
# =============================================================================


class _SkipChannel(Exception):
    """Control-flow signal: ``skip=True`` drops the result (matches a legacy
    ``continue``); ``skip=False`` surfaces it as an ``unsupported`` result."""

    def __init__(self, message: str, *, skip: bool):
        super().__init__(message)
        self.skip = skip


def _finite_or_none(x: float | None) -> float | None:
    if x is None:
        return None
    return float(x) if np.isfinite(x) else None


def _collect_targets(obj: Any) -> set[str]:
    """Every ``target`` value reachable in a serialized estimand dict."""
    found: set[str] = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "target" and isinstance(v, str):
                    found.add(v)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return found


def _substitute_target(obj: Any, channel: str) -> Any:
    """Replace every wildcard ``target`` with ``channel`` (deep, in a copy)."""
    if isinstance(obj, dict):
        return {
            k: (
                channel
                if (k == "target" and v == ALL_CHANNELS)
                else _substitute_target(v, channel)
            )
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_substitute_target(v, channel) for v in obj]
    return obj


__all__ = ["EstimandEvaluator"]
