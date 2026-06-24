"""Modeling modes for the oracle agent.

The agent now spans general Bayesian modeling, not only MMM. A session carries a
``modeling_mode`` that selects the system-prompt framing and the bound tool set —
**without** losing MMM's causal-measurement discipline (pre-registration, DAG,
assumptions, identification stay available in every mode).

``mmm`` is the default: an unset/legacy mode reproduces the historical behavior
byte-for-byte. The four modes form a rough nesting of *which* discipline is
foregrounded:

- ``mmm``              — full MMM: ROI, budget allocation, the lift-test
                         measurement loop, adstock/saturation.
- ``causal_inference`` — DAG / backdoor identification / estimands / experiment
                         design, but **no** ROI/budget/adstock machinery.
- ``general_bayes``    — the full Bayesian workflow (priors → fit → diagnostics →
                         posterior checks → sensitivity); causal steps optional.
- ``descriptive``      — measurement / latent-structure models (CFA, LCA): fit
                         indices, loadings, class profiles; no DAG, no ROI.
"""

from __future__ import annotations

from typing import Literal

ModelingMode = Literal["mmm", "causal_inference", "general_bayes", "descriptive"]

#: All valid modes (order = UI display order).
VALID_MODES: tuple[str, ...] = (
    "mmm",
    "causal_inference",
    "general_bayes",
    "descriptive",
)

#: The default mode — unset/legacy sessions read as this, preserving behavior.
DEFAULT_MODE: str = "mmm"

#: Human-readable labels for the UI mode switcher.
MODE_LABELS: dict[str, str] = {
    "mmm": "Marketing Mix Modeling",
    "causal_inference": "Causal Inference",
    "general_bayes": "General Bayesian Modeling",
    "descriptive": "Descriptive / Measurement",
}


def is_valid_mode(mode: object) -> bool:
    """True iff ``mode`` is one of the recognized modes."""
    return isinstance(mode, str) and mode in VALID_MODES


def normalize_mode(mode: object) -> str:
    """Coerce any value to a valid mode, falling back to :data:`DEFAULT_MODE`."""
    return mode if is_valid_mode(mode) else DEFAULT_MODE


def suggested_mode_for_kind(kind: object) -> str:
    """The session mode that best fits a model family ``kind``
    (``__garden_model_kind__``). Used only to **suggest** a switch — never to
    silently change a session's mode.
    """
    k = (kind or "").strip().lower() if isinstance(kind, str) else ""
    if k in ("", "mmm"):
        return "mmm"
    if k in ("cfa", "lca", "latent_class", "efa", "factor", "measurement"):
        return "descriptive"
    return "general_bayes"


def reconcile_mode_with_model(mode: object, garden_ref: object) -> dict:
    """Check whether the session ``mode`` agrees with the loaded model's family.

    Returns ``{consistent, model_kind, suggested_mode, note}``. A non-MMM model
    loaded into ``mmm`` mode is the main contradiction (its ROI/budget/experiment
    tools don't apply); ``note`` is a one-line, user-facing switch suggestion (or
    ``None`` when consistent). Never mutates anything.
    """
    mode = normalize_mode(mode)
    kind = "mmm"
    if isinstance(garden_ref, dict):
        kind = (
            garden_ref.get("model_kind")
            or (garden_ref.get("manifest") or {}).get("model_kind")
            or "mmm"
        )
    suggested = suggested_mode_for_kind(kind)

    if kind == "mmm":
        # An MMM model is at home in mmm / causal / general (all surface or allow
        # causal tooling); only a descriptive framing would mismatch it.
        consistent = mode in ("mmm", "causal_inference", "general_bayes")
    else:
        consistent = mode == suggested

    note = None
    if not consistent:
        note = (
            f"The loaded model is a '{kind}' family but this session is in "
            f"'{MODE_LABELS.get(mode, mode)}' mode. Consider switching to "
            f"'{MODE_LABELS.get(suggested, suggested)}' mode — the ROI / budget / "
            f"experiment tools don't apply to this family."
        )
    return {
        "consistent": consistent,
        "model_kind": kind,
        "suggested_mode": suggested,
        "note": note,
    }


__all__ = [
    "ModelingMode",
    "VALID_MODES",
    "DEFAULT_MODE",
    "MODE_LABELS",
    "is_valid_mode",
    "normalize_mode",
    "suggested_mode_for_kind",
    "reconcile_mode_with_model",
]
