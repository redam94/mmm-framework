"""Enumerations for dimensions, variable roles, transforms, priors, and inference."""

from __future__ import annotations

from enum import Enum


class DimensionType(str, Enum):
    """Standard MFF dimensions."""

    PERIOD = "Period"
    GEOGRAPHY = "Geography"
    PRODUCT = "Product"
    CAMPAIGN = "Campaign"
    OUTLET = "Outlet"
    CREATIVE = "Creative"


class VariableRole(str, Enum):
    """Role of a variable in the model."""

    KPI = "kpi"
    MEDIA = "media"
    CONTROL = "control"
    AUXILIARY = "auxiliary"  # For allocation weights, etc.


class CausalControlRole(str, Enum):
    """Causal role of a *control* variable, used to prevent "bad control" bias.

    The functional :class:`VariableRole` (``CONTROL``) says a variable is a
    regressor; it says nothing about *why* it is in the model. Conditioning on
    the wrong kind of variable induces bias rather than removing it (see the
    confounder-vs-precision-control proof in the project README). This enum lets
    the framework distinguish the four causal kinds so it can route each one
    correctly:

    - ``CONFOUNDER``: a common cause of media and the KPI. Must be included and
      must **not** be shrunk toward zero (shrinking a confounder biases the media
      coefficient). Routed to a wide, un-shrunk coefficient prior.
    - ``PRECISION_CONTROL``: a cause of the KPI only (not of media). Safe to
      include for efficiency and safe to shrink/select. This is the default,
      backward-compatible behavior for any unmarked control.
    - ``MEDIATOR``: lies on the causal path media -> ... -> KPI (post-treatment).
      Conditioning on it blocks part of the effect being estimated, so it must
      **not** be used as a control for a total-effect estimate.
    - ``COLLIDER``: a common effect of two causes on a backdoor path.
      Conditioning on it *opens* a spurious path, so it must **not** be used as a
      control.

    ``MEDIATOR`` and ``COLLIDER`` are refused at model-construction time. A
    control left unmarked (``None``) is treated as ``PRECISION_CONTROL`` for
    backward compatibility.
    """

    CONFOUNDER = "confounder"
    PRECISION_CONTROL = "precision_control"
    MEDIATOR = "mediator"
    COLLIDER = "collider"


class AdstockType(str, Enum):
    """Supported adstock transformations."""

    GEOMETRIC = "geometric"
    WEIBULL = "weibull"
    DELAYED = "delayed"
    NONE = "none"


class SaturationType(str, Enum):
    """Supported saturation transformations."""

    HILL = "hill"
    LOGISTIC = "logistic"
    MICHAELIS_MENTEN = "michaelis_menten"
    TANH = "tanh"
    NONE = "none"


class PriorType(str, Enum):
    """Supported prior distributions."""

    HALF_NORMAL = "HalfNormal"
    NORMAL = "Normal"
    LOG_NORMAL = "LogNormal"
    GAMMA = "Gamma"
    BETA = "Beta"
    TRUNCATED_NORMAL = "TruncatedNormal"
    HALF_STUDENT_T = "HalfStudentT"


class AllocationMethod(str, Enum):
    """Methods for allocating national data to sub-dimensions."""

    EQUAL = "equal"  # Equal split across all levels
    POPULATION = "population"  # By population weight
    SALES = "sales"  # By historical sales weight
    CUSTOM = "custom"  # User-provided weights


class InferenceMethod(str, Enum):
    """Available inference methods."""

    BAYESIAN_PYMC = "bayesian_pymc"
    BAYESIAN_NUMPYRO = "bayesian_numpyro"
    FREQUENTIST_RIDGE = "frequentist_ridge"
    FREQUENTIST_CVXPY = "frequentist_cvxpy"


class FitMethod(str, Enum):
    """How the posterior is obtained at fit time.

    ``NUTS`` is full MCMC (the default, the only one suitable for final
    inference). The remaining methods are *approximate* and exist to fit a
    model in seconds so you can spot problems — bad priors, broken geometry,
    pathological saturation/adstock — before paying for a full sample. Treat
    their uncertainty as unreliable.
    """

    NUTS = "nuts"
    MAP = "map"
    ADVI = "advi"
    FULLRANK_ADVI = "fullrank_advi"
    PATHFINDER = "pathfinder"

    @property
    def is_approximate(self) -> bool:
        return self is not FitMethod.NUTS


class ModelSpecification(str, Enum):
    """Model functional form."""

    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


class LikelihoodFamily(str, Enum):
    """Observation (likelihood) family for the KPI.

    ``NORMAL`` is the historical default and the only family the **built-in
    additive** model fits directly on standardized ``y`` (all of its component
    priors are calibrated in KPI standard deviations). ``STUDENT_T`` is a safe,
    heavier-tailed drop-in on the same standardized, identity-link scale.

    The remaining families (``LOGNORMAL`` and the count/bounded
    ``BINOMIAL``/``BETA_BINOMIAL``/``POISSON``/``NEGATIVE_BINOMIAL``/``BETA``)
    change the observation scale and require a non-identity link, so the
    additive model does **not** fit them directly — they are read by models that
    define their own observation block (override ``_build_model`` / subclass
    ``CustomMMM``), e.g. a binomial awareness model. ``is_gaussian`` /
    ``standardizes_y`` route this in :class:`LikelihoodConfig` and the model.
    """

    NORMAL = "normal"
    STUDENT_T = "student_t"
    LOGNORMAL = "lognormal"
    BINOMIAL = "binomial"
    BETA_BINOMIAL = "beta_binomial"
    POISSON = "poisson"
    NEGATIVE_BINOMIAL = "negative_binomial"
    BETA = "beta"

    @property
    def is_gaussian(self) -> bool:
        """Identity-link, continuous families the additive model fits directly
        on standardized ``y`` (the built-in dispatch supports these)."""
        return self in (LikelihoodFamily.NORMAL, LikelihoodFamily.STUDENT_T)

    @property
    def standardizes_y(self) -> bool:
        """Whether ``y`` is z-scored before entering the graph. Count/bounded
        families work in their natural (link) scale and are **not** standardized;
        ``LOGNORMAL`` is fit on standardized ``log(y)`` upstream, so the in-graph
        observation is still standardized."""
        return self in (
            LikelihoodFamily.NORMAL,
            LikelihoodFamily.STUDENT_T,
            LikelihoodFamily.LOGNORMAL,
        )


class LinkFunction(str, Enum):
    """Link mapping the linear predictor ``mu`` to the observation's natural
    parameter. ``IDENTITY`` for Gaussian families; ``LOGIT`` for bounded
    proportions (binomial/beta); ``LOG`` for counts (Poisson/neg-binomial)."""

    IDENTITY = "identity"
    LOGIT = "logit"
    LOG = "log"
