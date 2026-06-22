"""Report data extractor for non-MMM **confirmatory factor analysis** models.

A CFA has no channels / spend / single KPI, so it populates the CFA-specific
fields of :class:`MMMDataBundle` (factor loadings + fit indices) plus the
model-agnostic convergence diagnostics — leaving the channel/ROI fields empty so
those sections gate off. Reads the model's ``factor_loadings_summary()`` table
and its declared fit-index estimands (``srmr``/``cov_fit``/…)."""

from __future__ import annotations

from typing import Any

from .base import DataExtractor
from .bundle import MMMDataBundle


class FactorAnalysisExtractor(DataExtractor):
    """Extract loadings + fit indices from a CFA (or other latent-structure)
    garden model into an :class:`MMMDataBundle`."""

    def __init__(self, model: Any, ci_prob: float = 0.94, **_: Any):
        self.model = model
        self._ci_prob = ci_prob

    def extract(self) -> MMMDataBundle:
        bundle = MMMDataBundle()
        bundle.model_kind = self._model_kind()
        bundle.model_specification = {"model_kind": bundle.model_kind}
        bundle.factor_loadings = self._loadings()
        bundle.cfa_fit_indices = self._fit_indices()
        bundle.diagnostics = self._diagnostics()
        return bundle

    def _model_kind(self) -> str:
        from ...garden.contract import model_kind

        return model_kind(self.model)

    def _loadings(self) -> list[dict[str, Any]] | None:
        fn = getattr(self.model, "factor_loadings_summary", None)
        if not callable(fn):
            return None
        try:
            df = fn(hdi_prob=self.ci_prob)
        except TypeError:
            df = fn()
        except Exception:  # noqa: BLE001
            return None
        return [
            {
                "indicator": str(r.get("indicator", "")),
                "factor": str(r.get("factor", "")),
                "loading": float(r.get("loading", 0.0)),
                "hdi_low": float(r.get("hdi_low", r.get("loading", 0.0))),
                "hdi_high": float(r.get("hdi_high", r.get("loading", 0.0))),
            }
            for r in df.to_dict("records")
        ]

    def _fit_indices(self) -> dict[str, dict[str, float]] | None:
        fn = getattr(self.model, "evaluate_estimands", None)
        if not callable(fn):
            return None
        try:
            results = fn()  # the model's DEFAULT_ESTIMANDS (fit indices)
        except Exception:  # noqa: BLE001
            return None
        out: dict[str, dict[str, float]] = {}
        for name, r in (results or {}).items():
            status = getattr(r, "status", "ok")
            mean = getattr(r, "mean", None)
            if status != "ok" or mean is None:
                continue
            out[name] = {
                "mean": float(mean),
                "lower": float(getattr(r, "hdi_low", mean) or mean),
                "upper": float(getattr(r, "hdi_high", mean) or mean),
            }
        return out or None

    def _diagnostics(self) -> dict[str, Any] | None:
        try:
            from ...diagnostics import compute_fit_diagnostics

            diag = compute_fit_diagnostics(self.model)
            return diag.get("convergence") or None
        except Exception:  # noqa: BLE001
            return None


__all__ = ["FactorAnalysisExtractor"]
