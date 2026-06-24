"""Model Garden contract — the interface a custom MMM must satisfy to be
"oracle-compatible" (runnable by the agent + the reporting / analysis stack).

A "garden model" is a bespoke ``BayesianMMM`` subclass (the recommended path —
see :class:`mmm_framework.garden.base.CustomMMM`) or any class that duck-types
the same surface. This module is the **single source of truth** for that surface;
the compatibility suite (:mod:`mmm_framework.garden.compat`) is just the
executable encoding of the checks below.

It is intentionally dependency-light — **no PyMC / model imports at module load**
— so the static structural check :func:`validate_class` can run anywhere (host or
kernel) without paying the heavy model-stack import. The runtime checks
(:func:`validate_instance`, :func:`validate_fitted`) inspect a constructed /
fitted instance and are meant to run kernel-side, where the model lives.
"""

from __future__ import annotations

import inspect
from typing import Any, Protocol, runtime_checkable

#: Bumped when the required surface changes. Stored on every registered model so
#: a consumer can detect a model authored against an older/newer contract.
GARDEN_CONTRACT_VERSION = "1.0"

# Instance attributes every oracle-compatible model exposes (present after
# __init__, and — for ``_trace`` — after fit). Used to unstandardize metrics,
# normalize spend, and locate the panel/posterior in every downstream op.
#
# Split into a model-agnostic base (every Bayesian garden model) and an
# MMM-specific group (channels + spend maxima) that is required only of MMM-kind
# models — a non-MMM family (e.g. a CFA) is exempt via ``is_mmm_model``.
REQUIRED_ATTRS_BASE: tuple[str, ...] = (
    "y_mean",
    "y_std",
    "panel",
    "model_config",
    "has_geo",
    "has_product",
)
REQUIRED_ATTRS_MMM: tuple[str, ...] = (
    "channel_names",
    "_media_raw_max",
)
#: Union, kept for backward compatibility (importers + ``describe_contract``).
REQUIRED_ATTRS: tuple[str, ...] = REQUIRED_ATTRS_BASE + REQUIRED_ATTRS_MMM

#: Methods every model must define (fit is the one true requirement; everything
#: else is inherited from BayesianMMM or duck-typed).
REQUIRED_METHODS: tuple[str, ...] = ("fit",)

#: Methods that each unlock specific ops; absent ones must degrade gracefully.
RECOMMENDED_METHODS: tuple[str, ...] = (
    "predict",
    "sample_channel_contributions",
    "compute_component_decomposition",
)

#: Posterior parameter-name conventions the extraction helpers key off
#: (``compute_adstock_weights`` / ``compute_saturation_curves`` /
#: ``compute_roi_with_uncertainty`` look these prefixes up by channel). These
#: mirror BayesianMMM's graph: a channel ``c`` gets ``beta_c`` (coefficient),
#: ``adstock_alpha_c`` (carryover), and ``sat_half_c`` / ``sat_slope_c``
#: (saturation half-saturation point + slope).
PARAM_PREFIXES: tuple[str, ...] = (
    "beta_",
    "adstock_alpha_",
    "sat_half_",
    "sat_slope_",
)


@runtime_checkable
class SupportsFit(Protocol):
    """``fit(...)`` returning an ``MMMResults`` and setting ``self._trace``."""

    def fit(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class HasScaling(Protocol):
    """Standardization params used to unstandardize every downstream metric."""

    y_mean: float
    y_std: float


@runtime_checkable
class HasMediaMeta(Protocol):
    """Channel labels + raw-spend maxima (predict divides by the latter)."""

    channel_names: list
    _media_raw_max: dict


def is_bayesian_mmm_subclass(cls: Any) -> bool:
    """True if ``cls`` derives from ``BayesianMMM`` / an extended-model base —
    checked by MRO **name** so this stays PyMC-free (no model import)."""
    mro = getattr(cls, "__mro__", ())
    return any(
        getattr(b, "__name__", "") in ("BayesianMMM", "BaseExtendedMMM") for b in mro
    )


#: Default model kind when a class/instance does not declare one — the historical
#: MMM behaviour (channels, spend, ``beta_<channel>`` parameters, channel read-ops).
DEFAULT_MODEL_KIND = "mmm"


def model_kind(obj: Any) -> str:
    """The Garden ``model_kind`` of a class **or** instance.

    A model declares its kind with the class attribute ``__garden_model_kind__``
    (e.g. ``"cfa"``, ``"latent_class"``). Absent that, anything in the
    ``BayesianMMM`` MRO is ``"mmm"``; otherwise ``"unknown"`` so a structurally
    non-MMM class is not silently treated as an MMM."""
    declared = getattr(obj, "__garden_model_kind__", None)
    if isinstance(declared, str) and declared:
        return declared
    cls = obj if isinstance(obj, type) else type(obj)
    return DEFAULT_MODEL_KIND if is_bayesian_mmm_subclass(cls) else "unknown"


def is_mmm_model(obj: Any) -> bool:
    """Whether the MMM-specific gates apply to ``obj`` (class or instance).

    True unless the model **explicitly** declares a non-``"mmm"``
    ``__garden_model_kind__``. A duck-typed / unknown model with no declaration is
    treated as MMM — the historical default, so its channel attributes,
    ``beta_<channel>`` posterior convention, and channel read-ops are still
    checked. Only a declared non-MMM family (e.g. a CFA) opts out."""
    declared = getattr(obj, "__garden_model_kind__", None)
    if isinstance(declared, str) and declared and declared != DEFAULT_MODEL_KIND:
        return False
    return True


def has_latent_structure(obj: Any) -> bool:
    """Whether ``obj`` exposes a latent-structure summary the report can render.

    Duck-typed (a class or instance): True if it defines a callable
    ``factor_loadings_summary`` or ``class_profile_summary``. This is orthogonal
    to :func:`is_mmm_model` — a hybrid family (an MMM that ALSO estimates a latent
    factor, e.g. ``LatentFactorMMM``) is both ``is_mmm_model`` *and*
    ``has_latent_structure``, so its report shows the channel/ROI sections **and**
    a factor-loadings section. A pure CFA/LCA is non-MMM and has latent structure;
    a plain MMM has neither method, so the latent section stays off."""
    return callable(getattr(obj, "factor_loadings_summary", None)) or callable(
        getattr(obj, "class_profile_summary", None)
    )


def _fit_signature_ok(cls: Any) -> bool:
    """``fit`` should accept the standard knobs (method / random_seed) — either
    explicitly or via ``**kwargs``. Lenient: a bad signature is a warning-tier
    problem, not a hard reject."""
    fit = getattr(cls, "fit", None)
    if not callable(fit):
        return False
    try:
        sig = inspect.signature(fit)
    except (TypeError, ValueError):
        return True  # builtins / C-level — assume ok
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return "method" in params or "random_seed" in params


def validate_class(cls: Any) -> list[str]:
    """Static structural check (NO instantiation, no PyMC import).

    Returns a list of human-readable problems — empty means the class is
    structurally compatible. This is the gate run before a candidate is written
    to the registry and the first tier of :func:`compat.run_compatibility_check`.
    """
    problems: list[str] = []
    if not isinstance(cls, type):
        return [f"expected a class, got {type(cls).__name__}"]

    for meth in REQUIRED_METHODS:
        if not callable(getattr(cls, meth, None)):
            problems.append(f"missing required method: {meth}()")

    if not _fit_signature_ok(cls):
        problems.append(
            "fit() should accept `method` / `random_seed` (or **kwargs) so the "
            "agent can request an approximate vs NUTS fit"
        )

    # A garden MMM either subclasses BayesianMMM (and inherits the full,
    # serializable contract) OR must define the channel read surface itself. A
    # model that declares a non-MMM ``__garden_model_kind__`` (e.g. a CFA) is
    # exempt — it has no channels, so it only needs ``fit()`` + a posterior and
    # its own family-specific estimands/report.
    declared_non_mmm = model_kind(cls) not in (DEFAULT_MODEL_KIND, "unknown")
    if not is_bayesian_mmm_subclass(cls) and not declared_non_mmm:
        for meth in ("predict", "sample_channel_contributions"):
            if not callable(getattr(cls, meth, None)):
                problems.append(
                    f"non-BayesianMMM class is missing {meth}(); subclass "
                    "`mmm_framework.BayesianMMM` (recommended), implement the "
                    "full read surface, or declare a non-MMM "
                    "`__garden_model_kind__`"
                )
    return problems


def validate_instance(mmm: Any) -> list[str]:
    """Runtime check on a CONSTRUCTED model (post-__init__, pre/post fit).

    Confirms the required attributes are present and sane. ``channel_names``
    must be a non-empty ordered list and equal the panel's channels (the
    serializer raises on a mismatch).
    """
    problems: list[str] = []
    mmm_kind = is_mmm_model(mmm)
    required = REQUIRED_ATTRS if mmm_kind else REQUIRED_ATTRS_BASE
    for attr in required:
        if not hasattr(mmm, attr):
            problems.append(f"missing required attribute: {attr}")

    # Channel + spend-maxima sanity is MMM-only (a CFA has no channels).
    if mmm_kind:
        names = getattr(mmm, "channel_names", None)
        if names is not None and (not isinstance(names, (list, tuple)) or not names):
            problems.append("channel_names must be a non-empty ordered list")

        raw_max = getattr(mmm, "_media_raw_max", None)
        if isinstance(names, (list, tuple)) and isinstance(raw_max, dict):
            missing = [c for c in names if c not in raw_max]
            if missing:
                problems.append(
                    f"_media_raw_max is missing channels {missing} (predict() will "
                    "produce NaNs for them)"
                )

    for attr in ("y_mean", "y_std"):
        val = getattr(mmm, attr, None)
        if val is not None:
            try:
                float(val)
            except (TypeError, ValueError):
                problems.append(f"{attr} must be a float, got {type(val).__name__}")
    return problems


def validate_fitted(mmm: Any) -> list[str]:
    """Runtime check on a FITTED model: ``_trace`` populated, posterior coords
    align with the model, and the channel parameter-name conventions hold (so
    the extraction helpers return non-empty results)."""
    problems: list[str] = []
    trace = getattr(mmm, "_trace", None)
    if trace is None:
        return ["fit() did not set self._trace (downstream ops have no posterior)"]

    posterior = getattr(trace, "posterior", None)
    if posterior is None:
        return ["trace has no `posterior` group"]

    post_vars = set(getattr(posterior, "data_vars", {}))
    # The ``beta_<channel>`` convention is MMM-only — the ROI / decomposition
    # helpers extract contributions by it. A non-MMM family (CFA) carries its own
    # parameters (loadings, fit indices) and is exempt.
    if is_mmm_model(mmm):
        names = list(getattr(mmm, "channel_names", []) or [])
        has_beta = any(v.startswith("beta_") for v in post_vars) or any(
            f"beta_{c}" in post_vars for c in names
        )
        if names and not has_beta:
            problems.append(
                "posterior has no `beta_<channel>` parameters — ROI / decomposition "
                f"helpers will return empty (saw vars: {sorted(post_vars)[:8]}…)"
            )

    # Posterior must be a well-formed arviz group (chain/draw sample axes) so
    # az.summary / rhat / ess and the contribution extractors can index it.
    try:
        dims = set(posterior.dims)
        missing_axes = [d for d in ("chain", "draw") if d not in dims]
        if missing_axes:
            problems.append(
                f"posterior is missing sample dims {missing_axes} "
                f"(saw {sorted(dims)[:8]}…) — not a standard MCMC/approx trace"
            )
    except Exception:  # noqa: BLE001 — best-effort coord introspection
        pass
    return problems


def find_garden_class(module: Any) -> type:
    """Locate THE garden model class inside an imported module.

    Resolution order: an explicit module-level ``GARDEN_MODEL`` attribute, else
    the single class defined in the module that subclasses BayesianMMM, else the
    single class defined in the module that passes :func:`validate_class`.
    Raises ``ValueError`` when ambiguous or absent.
    """
    explicit = getattr(module, "GARDEN_MODEL", None)
    if isinstance(explicit, type):
        return explicit

    defined = [
        obj
        for _name, obj in vars(module).items()
        if isinstance(obj, type) and getattr(obj, "__module__", None) == module.__name__
    ]
    subclasses = [c for c in defined if is_bayesian_mmm_subclass(c)]
    if len(subclasses) == 1:
        return subclasses[0]
    if len(subclasses) > 1:
        raise ValueError(
            "module defines multiple BayesianMMM subclasses "
            f"({[c.__name__ for c in subclasses]}); set `GARDEN_MODEL = YourClass` "
            "to disambiguate"
        )

    compatible = [c for c in defined if not validate_class(c)]
    if len(compatible) == 1:
        return compatible[0]
    raise ValueError(
        "could not find a unique garden model class in the module; define one "
        "class that subclasses `mmm_framework.BayesianMMM`, or set "
        "`GARDEN_MODEL = YourClass`"
    )


def describe_contract() -> str:
    """Markdown summary of the contract — surfaced to the authoring agent / docs."""
    return (
        f"### Model Garden contract (v{GARDEN_CONTRACT_VERSION})\n\n"
        "A custom model must be **oracle-compatible**. The simplest way is to "
        "subclass `mmm_framework.garden.CustomMMM` (or `mmm_framework.BayesianMMM`) "
        "and override the build/prior hooks — you then inherit the full contract "
        "for free.\n\n"
        "**Required attributes:** "
        + ", ".join(f"`{a}`" for a in REQUIRED_ATTRS)
        + ".\n\n**Required method:** `fit(method=None, random_seed=None, ...) -> "
        "MMMResults` — must set `self._trace` before returning.\n\n"
        "**Recommended methods** (each unlocks ops; absence must not raise): "
        + ", ".join(f"`{m}`" for m in RECOMMENDED_METHODS)
        + ".\n\n**Posterior naming:** channel params follow `beta_<channel>`, "
        "`adstock_alpha_<channel>`, `sat_half_<channel>`, `sat_slope_<channel>`; "
        "per-observation deterministics are registered in the **original** KPI scale; "
        "`MMMResults.approximate=True` for MAP/ADVI/Pathfinder fits.\n\n"
        "**Non-MMM families:** set the class attr `__garden_model_kind__` (e.g. "
        "'cfa', 'latent_class') to opt out of the MMM-specific gates — the channel "
        "attributes (`channel_names`, `_media_raw_max`), the `beta_<channel>` "
        "posterior convention, and the channel read-ops/compat tiers are then not "
        "required. Such a model needs only `fit()` + a posterior trace and its own "
        "family-specific estimands (declared via `DEFAULT_ESTIMANDS`)."
    )


__all__ = [
    "GARDEN_CONTRACT_VERSION",
    "DEFAULT_MODEL_KIND",
    "REQUIRED_ATTRS",
    "REQUIRED_ATTRS_BASE",
    "REQUIRED_ATTRS_MMM",
    "REQUIRED_METHODS",
    "RECOMMENDED_METHODS",
    "PARAM_PREFIXES",
    "SupportsFit",
    "HasScaling",
    "HasMediaMeta",
    "is_bayesian_mmm_subclass",
    "model_kind",
    "is_mmm_model",
    "has_latent_structure",
    "validate_class",
    "validate_instance",
    "validate_fitted",
    "find_garden_class",
    "describe_contract",
]
