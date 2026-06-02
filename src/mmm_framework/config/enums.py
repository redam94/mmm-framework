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


class ModelSpecification(str, Enum):
    """Model functional form."""

    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
