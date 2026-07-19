"""Report data extractor for **non-MMM latent-structure** garden models — a
confirmatory factor analysis (CFA), a latent class analysis (LCA), or any model
declaring a non-``"mmm"`` ``__garden_model_kind__``.

These have no channels / spend / single KPI, so the extractor populates the
generic latent-structure fields of :class:`MMMDataBundle` (a summary table + the
declared estimands + section titles) plus the model-agnostic convergence
diagnostics — leaving the channel/ROI fields empty so those sections gate off.
The summary table is read from the model's family-appropriate method
(``factor_loadings_summary`` for a CFA, ``class_profile_summary`` for an LCA)."""

from __future__ import annotations

from typing import Any

from .base import DataExtractor
from .bundle import MMMDataBundle

#: Per-family display + data wiring: (section title, summary-table title,
#: estimands-block title, the model method that returns the summary DataFrame).
_FAMILY = {
    "cfa": (
        "Confirmatory Factor Analysis",
        "Factor loadings",
        "Fit indices",
        "factor_loadings_summary",
    ),
    "latent_class": (
        "Latent Class Analysis",
        "Class item-endorsement profiles",
        "Class sizes",
        "class_profile_summary",
    ),
    "clv": (
        "Customer Lifetime Value",
        "Customer value profile",
        "Value estimands",
        "customer_value_summary",
    ),
}
_DEFAULT_FAMILY = ("Latent Structure", "Latent summary", "Estimands", None)


class FactorAnalysisExtractor(DataExtractor):
    """Extract a latent-structure summary + estimands from a non-MMM garden model
    into an :class:`MMMDataBundle` (family-agnostic: CFA, LCA, …)."""

    def __init__(self, model: Any, ci_prob: float = 0.94, **_: Any):
        self.model = model
        self._ci_prob = ci_prob

    def extract(self) -> MMMDataBundle:
        kind = self._model_kind()
        section_title, table_title, estimands_title, method = _FAMILY.get(
            kind, _DEFAULT_FAMILY
        )
        bundle = MMMDataBundle()
        bundle.model_kind = kind
        bundle.model_specification = {"model_kind": kind}
        bundle.latent_section_title = section_title
        bundle.latent_table_title = table_title
        bundle.latent_estimands_title = estimands_title
        bundle.factor_loadings = self._table(method)
        bundle.cfa_fit_indices = self._estimands()
        bundle.diagnostics = self._diagnostics()
        return bundle

    def _model_kind(self) -> str:
        from ...garden.contract import model_kind

        return model_kind(self.model)

    def _table(self, method: str | None) -> list[dict[str, Any]] | None:
        """The latent summary table as a list of column dicts. Tries the
        family-specific method first, then any of the known summary methods, so a
        model that names its method differently still renders."""
        candidates = [method] if method else []
        candidates += [
            "factor_loadings_summary",
            "class_profile_summary",
            "customer_value_summary",
        ]
        for name in candidates:
            if not name:
                continue
            fn = getattr(self.model, name, None)
            if not callable(fn):
                continue
            try:
                df = fn(hdi_prob=self._ci_prob)
            except TypeError:
                df = fn()
            except Exception:  # noqa: BLE001
                continue
            return [
                {k: (float(v) if _is_num(v) else str(v)) for k, v in row.items()}
                for row in df.to_dict("records")
            ]
        return None

    def _estimands(self) -> dict[str, dict[str, float]] | None:
        fn = getattr(self.model, "evaluate_estimands", None)
        if not callable(fn):
            return None
        try:
            results = fn()  # the model's declared / default estimands
        except Exception:  # noqa: BLE001
            return None
        out: dict[str, dict[str, float]] = {}
        for name, r in (results or {}).items():
            mean = getattr(r, "mean", None)
            if getattr(r, "status", "ok") != "ok" or mean is None:
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


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


__all__ = ["FactorAnalysisExtractor"]
