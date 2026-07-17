"""Centralized MCMC convergence verdict + warning.

Single source of truth for *"did the sampler work?"* so the core
(:class:`mmm_framework.model.BayesianMMM`) and the extended models
(:mod:`mmm_framework.mmm_extensions`) agree on the same thresholds, and so a
non-converged fit can never be returned without a signal.

The thresholds mirror the opt-in :class:`mmm_framework.validation.ModelValidator`
and the fit-health snapshot (R-hat per Vehtari et al. 2021; the ~100-per-chain
bulk-ESS rule of thumb at 4 chains):

* ``rhat_max <= 1.01``
* ``ess_bulk_min >= 400``
* ``divergences == 0``

For *approximate* fits (MAP / Laplace / ADVI / Pathfinder) R-hat and ESS are
undefined, so the verdict is ``None`` (not assessable) — callers must not treat
``None`` as "converged". SMC fits are EXACT (``approximate`` False): R-hat/ESS
are computed across the independent SMC runs and the verdict applies —
disagreeing runs (high R-hat) are SMC's multimodality signal; divergences do
not exist for SMC and stay ``None`` (never flagged).
"""

from __future__ import annotations

import math
import warnings
from typing import Any

# Conventional sampler-health thresholds. Kept in step with
# diagnostics.snapshot.RHAT_OK / ESS_BULK_OK and validation.validator.
RHAT_OK = 1.01
ESS_BULK_OK = 400.0


class ConvergenceWarning(UserWarning):
    """Warning category emitted when an MCMC fit fails standard convergence checks.

    Filter or escalate it explicitly, e.g.::

        import warnings
        from mmm_framework.diagnostics.convergence import ConvergenceWarning
        warnings.simplefilter("error", ConvergenceWarning)  # treat as failure
    """


def _finite(x: Any) -> float | None:
    """Finite float or ``None`` (keeps verdicts JSON-safe; no NaN/inf)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def compute_convergence(trace: Any) -> dict[str, Any]:
    """Compute ``divergences`` / ``rhat_max`` / ``ess_bulk_min`` from an ArviZ trace.

    Best-effort: any piece that cannot be computed comes back as ``None`` rather
    than raising, so a diagnostics computation never fails a fit.
    """
    import arviz as az

    from ..utils import arviz_compat

    out: dict[str, Any] = {
        "divergences": None,
        "rhat_max": None,
        "ess_bulk_min": None,
    }
    try:
        out["divergences"] = int(trace.sample_stats.diverging.sum().values)
    except Exception:
        pass
    try:
        out["rhat_max"] = _finite(arviz_compat.dataset_extremum(az.rhat(trace), "max"))
    except Exception:
        pass
    try:
        out["ess_bulk_min"] = _finite(
            arviz_compat.dataset_extremum(az.ess(trace, method="bulk"), "min")
        )
    except Exception:
        pass
    return out


def convergence_flags(diagnostics: dict[str, Any]) -> list[str]:
    """Which checks FAILED: subset of ``{"divergences", "rhat", "ess"}``."""
    flags: list[str] = []
    div = diagnostics.get("divergences")
    rhat = _finite(diagnostics.get("rhat_max"))
    ess = _finite(diagnostics.get("ess_bulk_min"))
    if div:
        flags.append("divergences")
    if rhat is not None and rhat > RHAT_OK:
        flags.append("rhat")
    if ess is not None and ess < ESS_BULK_OK:
        flags.append("ess")
    return flags


def is_converged(diagnostics: dict[str, Any]) -> bool | None:
    """Convergence verdict from a diagnostics dict.

    Returns ``True``/``False`` for NUTS fits, and ``None`` when convergence is
    not assessable — an approximate fit (``diagnostics["approximate"]`` truthy)
    or one with no usable R-hat/ESS/divergence signal. ``None`` is NOT
    "converged"; callers should surface it as "N/A".
    """
    if diagnostics.get("approximate"):
        return None
    rhat = _finite(diagnostics.get("rhat_max"))
    ess = _finite(diagnostics.get("ess_bulk_min"))
    div = diagnostics.get("divergences")
    if rhat is None and ess is None and div is None:
        return None
    return not convergence_flags(diagnostics)


def convergence_warning_message(
    diagnostics: dict[str, Any], label: str = "model"
) -> str | None:
    """Human-readable non-convergence message, or ``None`` if it converged / N/A."""
    flags = convergence_flags(diagnostics)
    if not flags:
        return None
    parts: list[str] = []
    if "divergences" in flags:
        parts.append(
            f"{int(diagnostics.get('divergences') or 0)} divergent transition(s)"
        )
    if "rhat" in flags:
        parts.append(
            f"max R-hat={_finite(diagnostics.get('rhat_max')):.3f} (> {RHAT_OK})"
        )
    if "ess" in flags:
        parts.append(
            f"min bulk-ESS={_finite(diagnostics.get('ess_bulk_min')):.0f} "
            f"(< {ESS_BULK_OK:.0f})"
        )
    return (
        f"{label} fit has NOT converged: "
        + "; ".join(parts)
        + ". The posterior is unreliable — do not trust the intervals/ROI. "
        "Increase tune/draws/chains or reparameterize and re-fit before acting."
    )


def warn_if_not_converged(
    diagnostics: dict[str, Any], label: str = "model", *, stacklevel: int = 3
) -> bool:
    """Emit a :class:`ConvergenceWarning` if the fit failed convergence checks.

    Returns ``True`` if a warning was emitted.
    """
    msg = convergence_warning_message(diagnostics, label)
    if msg is None:
        return False
    warnings.warn(msg, ConvergenceWarning, stacklevel=stacklevel)
    return True


def annotate(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Return ``diagnostics`` with ``flags`` and ``converged`` filled in.

    Mutates and returns the same dict so it round-trips through serialization.
    """
    diagnostics["flags"] = convergence_flags(diagnostics)
    diagnostics["converged"] = is_converged(diagnostics)
    return diagnostics
