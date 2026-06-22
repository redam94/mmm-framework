"""Declarative estimand specification — the counterfactual causal lens.

An :class:`Estimand` is a **named, serializable** counterfactual contrast::

    reduce_over( op( quantity | intervention,  quantity | baseline ) ) / normalizer

It is the single source of truth shared by the two realization engines:

* :mod:`mmm_framework.estimands.evaluate` — *post-hoc* realization (numpy; reads
  the posterior-predictive via :meth:`BayesianMMM.predict_under`, paired/unpaired
  seeds, boolean time masks).
* :mod:`mmm_framework.estimands.graph` — *in-graph* realization (pytensor; builds
  a likelihood-time scalar; deterministic, integer-indexed windows).

This module imports **neither numpy nor pytensor** so it can be loaded anywhere
(host or kernel) for serialization / validation without paying the model-stack
import cost — mirroring :mod:`mmm_framework.garden.contract`.

The schema is intentionally model-agnostic: a future non-MMM family (CFA / LCA /
EFA) declares its own estimands over its own quantities (a latent factor, a class
membership) and gates them with :attr:`Estimand.required_capabilities`. The MMM
built-ins live in :mod:`mmm_framework.estimands.registry`.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Protocol, Union, runtime_checkable

from pydantic import BaseModel, Field

#: Bumped when the serialized estimand surface changes. Stored on every
#: ``Estimand`` so a consumer (a reloaded model, a saved spec) can detect drift.
ESTIMAND_SCHEMA_VERSION = "1.0"

#: Sentinel ``target`` meaning "expand this estimand once per channel". The
#: evaluator substitutes each channel name into every wildcard target and emits
#: one result per channel (keyed ``"{name}:{channel}"``). A non-MMM model with
#: no channels yields nothing — natural capability filtering.
ALL_CHANNELS = "*"


# =============================================================================
# Interventions — how an input is set in the counterfactual world
# =============================================================================


class Observed(BaseModel):
    """The factual world: inputs as observed (no intervention)."""

    type: Literal["observed"] = "observed"


class ZeroInput(BaseModel):
    """Set ``target``'s input to zero (the channel-off counterfactual)."""

    type: Literal["zero_input"] = "zero_input"
    target: str


class ScaleInput(BaseModel):
    """Multiply ``target``'s input by ``factor`` (e.g. ``1.1`` for +10%)."""

    type: Literal["scale_input"] = "scale_input"
    target: str
    factor: float


class SetInput(BaseModel):
    """Set ``target``'s input to a fixed ``value``.

    ``sustained`` / ``carryover_state`` carry the off-panel adstock convention
    for the in-graph engine (steady-state vs cold-start ramp); the post-hoc
    engine sets the column to ``value`` over the active window.
    """

    type: Literal["set_input"] = "set_input"
    target: str
    value: float
    sustained: bool = False
    carryover_state: Literal["steady_state", "cold_start"] | None = None


class CustomIntervention(BaseModel):
    """An intervention realized by a registered callable, keyed by ``ref``."""

    type: Literal["custom"] = "custom"
    ref: str
    params: dict[str, Any] = Field(default_factory=dict)


Intervention = Annotated[
    Union[Observed, ZeroInput, ScaleInput, SetInput, CustomIntervention],
    Field(discriminator="type"),
]


# =============================================================================
# Quantities — what is measured in a world
# =============================================================================


class Outcome(BaseModel):
    """The model's predicted outcome (KPI), in original scale."""

    type: Literal["outcome"] = "outcome"


class Contribution(BaseModel):
    """A channel's contribution.

    ``source="counterfactual"`` realizes it as ``outcome|observed −
    outcome|zero(target)`` (a :class:`Contrast`); ``source="in_graph_deterministic"``
    reads the in-graph ``channel_contributions`` Deterministic from the posterior
    (the dashboard decomposition). The two are **different numbers** — see the
    registry's ``counterfactual_roi`` vs ``contribution_roi``.
    """

    type: Literal["contribution"] = "contribution"
    target: str
    source: Literal["counterfactual", "in_graph_deterministic"] = "counterfactual"


class ObservedInput(BaseModel):
    """An observed input level (spend), used as a denominator.

    ``source="raw"`` sums ``X_media_raw[window, target]``; ``source="panel"``
    uses the reporting spend extractor (``panel.X_media``). The dashboard ROI
    uses ``panel``; the counterfactual / marginal paths use ``raw``.
    """

    type: Literal["observed_input"] = "observed_input"
    target: str
    source: Literal["raw", "panel"] = "raw"


class MarginalSpend(BaseModel):
    """The incremental spend implied by a :class:`ScaleInput`.

    ``intervention_ref="numerator"`` reads the numerator contrast's intervention
    factor (never a second literal percentage — a desync guard): the marginal
    denominator is ``current_spend * (factor - 1)``.
    """

    type: Literal["marginal_spend"] = "marginal_spend"
    target: str
    intervention_ref: str = "numerator"


class LatentVar(BaseModel):
    """A named latent posterior variable (e.g. an awareness state)."""

    type: Literal["latent_var"] = "latent_var"
    name: str


class Constant(BaseModel):
    """A fixed scalar."""

    type: Literal["constant"] = "constant"
    value: float


Quantity = Annotated[
    Union[Outcome, Contribution, ObservedInput, MarginalSpend, LatentVar, Constant],
    Field(discriminator="type"),
]


# =============================================================================
# Contrast — a reduced difference/ratio of a quantity across two worlds
# =============================================================================


class Contrast(BaseModel):
    """``reduce_over(op(quantity|intervention, quantity|baseline))``.

    With ``baseline=None`` the baseline is the factual world (:class:`Observed`).
    ``paired_seed`` is a post-hoc realization hint: when set and no explicit seed
    is supplied, the two worlds share one synthesized seed so observation noise
    cancels in the per-draw difference (the marginal path); the in-graph engine
    ignores it.
    """

    quantity: Quantity
    intervention: Intervention = Field(default_factory=Observed)
    baseline: Intervention | None = None
    op: Literal["difference", "ratio", "identity"] = "difference"
    reduce: Literal["sum", "mean"] = "sum"
    over: list[str] = Field(default_factory=lambda: ["time"])
    paired_seed: bool = False


# =============================================================================
# Result summaries + realization profile
# =============================================================================


class Summary(BaseModel):
    """A posterior tail probability over the estimand's per-draw samples.

    ``side="gt"``/``"lt"`` with ``threshold`` carries ``prob_positive``
    (``gt 0``) and ``prob_profitable`` (``gt 1``).
    """

    kind: Literal["tail_prob"] = "tail_prob"
    name: str
    threshold: float = 0.0
    side: Literal["gt", "lt"] = "gt"


class Realization(BaseModel):
    """Per-estimand realization knobs that pin bit-stable arithmetic.

    These encode *where a naive engine diverges* between the four legacy notions
    (see ``technical-docs/estimands.md``), kept separate from the causal
    definition above:

    * ``point_rule`` — ``"diff_of_means"`` (point = reduced per-call means then
      ``op``; the counterfactual / marginal paths) vs ``"mean_of_samples"`` (point
      = mean of the per-draw ratio; the dashboard decomposition ROI).
    * ``hdi_method`` — ``"percentile"`` (``compute_hdi_bounds``), ``"az_hdi"``
      (``az.hdi`` w/ percentile fallback; the dashboard) or ``"finite_percentile"``
      (``_hdi_finite``; the marginal path, filters non-finite draws).
    """

    point_rule: Literal["diff_of_means", "mean_of_samples"] = "diff_of_means"
    hdi_method: Literal["percentile", "az_hdi", "finite_percentile"] = "percentile"


class TimeWindow(BaseModel):
    """An inclusive ``[start, end]`` window in period-index units.

    Lives at the :class:`Estimand` level so the numerator and denominator share
    one time mask (``BayesianMMM._get_time_mask((start, end))``).
    """

    start: int
    end: int

    def as_tuple(self) -> tuple[int, int]:
        return (self.start, self.end)


# =============================================================================
# Estimand
# =============================================================================


class Estimand(BaseModel):
    """A named, serializable counterfactual estimand.

    The headline value is ``numerator`` (a :class:`Contrast` or bare
    :class:`Quantity`), optionally divided by ``denominator``. ``mean`` /
    ``hdi_low`` / ``hdi_high`` in the result are in :attr:`units`; companion
    quantities (the raw contribution + its HDI, spend, ``contribution_pct``) land
    in ``result.extra``.
    """

    name: str
    kind: str
    numerator: Contrast | Quantity
    denominator: Contrast | Quantity | None = None
    op_ratio_zero_denominator: Literal["zero", "skip", "nan"] = "zero"
    window: TimeWindow | None = None
    hdi_prob: float = 0.94
    summaries: list[Summary] = Field(default_factory=list)
    realization: Realization = Field(default_factory=Realization)
    required_capabilities: list[str] = Field(default_factory=list)
    units: str = ""
    causal_assumptions: str = ""
    schema_version: str = ESTIMAND_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict (mirrors ``ExperimentMeasurement.to_dict``)."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Estimand":
        """Parse a serialized estimand (mirrors ``ExperimentMeasurement.from_dict``)."""
        return cls.model_validate(data)


class EstimandResult(BaseModel):
    """Realized value of an :class:`Estimand` from a fitted posterior.

    ``status="unsupported"`` (with ``reason``) is returned — never raised — when
    ``required_capabilities`` are not met, mirroring the agent ops' ``_err``
    shape. ``mean``/``hdi_*`` are ``None`` for an unsupported result.
    """

    name: str
    kind: str = ""
    status: Literal["ok", "unsupported"] = "ok"
    mean: float | None = None
    hdi_low: float | None = None
    hdi_high: float | None = None
    hdi_prob: float = 0.94
    units: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# =============================================================================
# Model interface (the future non-MMM seam)
# =============================================================================


@runtime_checkable
class SupportsEstimands(Protocol):
    """The surface an :class:`Estimand` is realized against.

    ``BayesianMMM`` implements it today; a future non-MMM family implements the
    same four members (declaring its own capabilities + quantities) to gain the
    whole estimand stack. ``predict_under`` returns a ``PredictionResults``-like
    object exposing ``y_pred_mean`` and ``y_pred_samples``.
    """

    channel_names: list[str]
    declared_estimands: list[Estimand]

    def predict_under(
        self,
        intervention: Intervention,
        time_period: tuple[int, int] | None = ...,
        random_seed: int | None = ...,
    ) -> Any: ...

    def model_capabilities(self) -> set[str]: ...

    def evaluate_estimands(
        self, estimands: list[Estimand] | None = ...
    ) -> dict[str, EstimandResult]: ...


__all__ = [
    "ESTIMAND_SCHEMA_VERSION",
    "ALL_CHANNELS",
    "Observed",
    "ZeroInput",
    "ScaleInput",
    "SetInput",
    "CustomIntervention",
    "Intervention",
    "Outcome",
    "Contribution",
    "ObservedInput",
    "MarginalSpend",
    "LatentVar",
    "Constant",
    "Quantity",
    "Contrast",
    "Summary",
    "Realization",
    "TimeWindow",
    "Estimand",
    "EstimandResult",
    "SupportsEstimands",
]
