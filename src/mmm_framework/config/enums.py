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


class MeasurementUnit(str, Enum):
    """How a media channel's modeled variable is measured.

    This drives how the channel's efficiency is reported, **not** how its
    response curve is fit (the curve is always fit on the modeled variable):

    - ``SPEND`` (the default, backward-compatible): the modeled variable *is*
      dollars spent, so ROI / marginal ROAS divide the channel's incremental
      KPI by the summed spend exactly as before.
    - ``IMPRESSIONS`` / ``CLICKS``: the modeled variable is a *volume*, not
      dollars. ROI cannot be formed by summing the variable. If a cost is known
      (a separate ``spend_column``, or a ``cpm`` / ``cpc`` constant) the volume
      is converted to a spend series and normal ROI / mROAS are reported;
      otherwise the framework reports **efficiency** instead — incremental KPI
      per 1,000 impressions (or per click) and the marginal efficiency of an
      extra 1,000 impressions (or click), whose break-even reference is 0, not
      1.0.
    - ``OTHER``: a volume with no natural per-1,000 unit; treated like
      ``IMPRESSIONS`` for efficiency but labeled per unit.

    See :class:`mmm_framework.config.variables.MediaChannelConfig` for the
    companion ``spend_column`` / ``cpm`` / ``cpc`` fields and
    :mod:`mmm_framework.reporting.helpers.measurement` for the resolver that
    turns this into a divisor + metric labels.
    """

    SPEND = "spend"
    IMPRESSIONS = "impressions"
    CLICKS = "clicks"
    OTHER = "other"

    @property
    def is_spend(self) -> bool:
        return self is MeasurementUnit.SPEND


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
    """Supported saturation transformations.

    Note on completeness: two forms practitioners sometimes name separately are
    already available here under different names — the **ADBUDG** S-curve
    ``x**s / (k**s + x**s)`` is exactly :attr:`HILL`, and the **exponential-CDF**
    form ``1 - exp(-lam * x)`` is exactly :attr:`LOGISTIC`. :attr:`ROOT` (power)
    is the genuinely distinct concave form the others don't cover.
    """

    HILL = "hill"
    LOGISTIC = "logistic"
    MICHAELIS_MENTEN = "michaelis_menten"
    TANH = "tanh"
    ROOT = "root"
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

    Two methods are *exact* (asymptotically unbiased posterior samplers):

    * ``NUTS`` — full gradient MCMC, the default and the workhorse for final
      inference.
    * ``SMC`` — tempered Sequential Monte Carlo (``pm.sample_smc``). Slower
      than NUTS on well-behaved posteriors, but it handles **multimodal**
      posteriors NUTS gets mode-locked in (reflected factor modes, label
      switching, adstock↔AR ridges) and yields a **log marginal likelihood**
      for model comparison. Use it to confirm suspected multimodality or to
      compute model evidence — not as a speedup.

    The remaining methods are *approximate* and exist to fit a model in
    seconds so you can spot problems — bad priors, broken geometry,
    pathological saturation/adstock — before paying for a full sample. Treat
    their uncertainty as unreliable. ``MAP`` is a point estimate; ``LAPLACE``
    adds a Gaussian curvature approximation around the MAP point (cheap
    uncertainty, better-behaved than bare MAP on high-dimensional models);
    ``ADVI``/``FULLRANK_ADVI`` are variational; ``PATHFINDER`` is quasi-Newton
    variational (via the declared ``pymc-extras`` dependency, like LAPLACE).
    """

    NUTS = "nuts"
    SMC = "smc"
    MAP = "map"
    LAPLACE = "laplace"
    ADVI = "advi"
    FULLRANK_ADVI = "fullrank_advi"
    PATHFINDER = "pathfinder"

    @property
    def is_approximate(self) -> bool:
        """True for the fast-check methods whose uncertainty is not calibrated.

        NUTS **and SMC** are exact samplers — SMC must not trip the
        "approximate fit — uncertainty is not calibrated" banners/report
        gating even though it is not NUTS.
        """
        return self not in (FitMethod.NUTS, FitMethod.SMC)


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
