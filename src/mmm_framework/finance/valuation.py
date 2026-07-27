"""What one KPI unit is worth in dollars, and where that number came from.

The design decision this module exists to enforce: **"unresolved" is a state,
not a number.** A missing valuation used to fall back to ``1.0``, which is a
silent assertion that one unit of the KPI is worth one dollar — true only by
coincidence, and false by ~1000x on a KPI denominated in thousands. Every
consumer that turns this into money must therefore refuse rather than guess, so
:class:`ResolvedValue` carries ``source`` and :meth:`ResolvedValue.require`
raises when nothing resolved.

The in-repo precedent is ``planning/experiment_value.py``, which already sets
``dollar = margin_per_kpi is not None`` and labels its output non-dollar when
nothing resolves. This generalizes that rather than inventing it.

**Margin is exogenous and never estimated.** Nothing in this framework infers a
gross margin from data; it is always supplied by a human or a saved preference.
``source`` is required in the return type so every rendering surface can say so.

Precedence, highest first:

1. an explicit ``override`` passed by the caller;
2. the model spec's ``valuation.*`` block;
3. the project's saved ``economics`` preference;
4. the project's branding ``economics`` block;
5. **unresolved**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "KpiKind",
    "KpiValuation",
    "ResolvedValue",
    "UnresolvedValueError",
    "kpi_to_dollars",
]


class UnresolvedValueError(ValueError):
    """A dollar-denominated decision was requested with no KPI valuation.

    Raised instead of defaulting to ``1.0``. Carries ``what`` so the caller can
    name the specific decision that needs the number.
    """

    def __init__(self, what: str):
        self.what = what
        super().__init__(
            f"{what} needs a KPI valuation and none is set. One unit of this "
            "KPI is not assumed to be worth one dollar. Supply it explicitly "
            "(kind='revenue' + gross_margin, or kind='units' + gross_margin + "
            "price), set the project's `economics` preference, or use a "
            "fixed-budget allocation, which needs no valuation."
        )


class KpiKind(str, Enum):
    """What the KPI measures, which determines how it converts to dollars."""

    REVENUE = "revenue"
    """KPI is already money. Value per unit = gross margin."""

    UNITS = "units"
    """KPI is a unit/conversion count. Value per unit = gross margin x price."""

    OTHER = "other"
    """KPI is neither (sessions, awareness points). Not convertible to dollars."""


class KpiValuation(BaseModel):
    """A declared statement of what one KPI unit is worth. Never inferred.

    ``gross_margin`` is a FRACTION in ``(0, 1]``. The bound is deliberate: the
    pre-1.4 resolver guarded only ``m <= 0``, so ``gross_margin=40`` (a user
    meaning 40%) was accepted and multiplied every profit number by 40. Since
    ``save_preference`` validated nothing but branding, an agent could persist
    it.
    """

    model_config = ConfigDict(extra="forbid")

    kind: KpiKind = KpiKind.REVENUE
    gross_margin: float | None = Field(default=None, gt=0.0, le=1.0)
    price: float | None = Field(default=None, gt=0.0)
    currency: str = "USD"
    scale: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Dollars per one unit of the KPI column as modelled. Use it when the "
            "KPI is denominated in thousands or millions (scale=1000 / 1e6) — "
            "the exact case that made the 1.0 default ~1000x wrong."
        ),
    )

    @model_validator(mode="after")
    def _check_units_need_price(self) -> "KpiValuation":
        if self.kind is KpiKind.UNITS and self.gross_margin is not None:
            if self.price is None:
                raise ValueError(
                    "kind='units' converts to dollars as gross_margin x price, "
                    "so `price` is required. For a KPI already denominated in "
                    "money use kind='revenue'."
                )
        return self

    def value_per_kpi(self) -> float | None:
        """Dollars per one KPI unit, or ``None`` when not convertible."""
        if self.kind is KpiKind.OTHER or self.gross_margin is None:
            return None
        if self.kind is KpiKind.UNITS:
            if self.price is None:  # pragma: no cover - validator guards this
                return None
            return self.gross_margin * self.price * self.scale
        return self.gross_margin * self.scale


@dataclass(frozen=True)
class ResolvedValue:
    """The resolved value per KPI unit, with its provenance.

    ``is_dollar`` is the load-bearing field: ``False`` means no valuation
    resolved and **no dollar-denominated number may be produced**. Consumers
    call :meth:`require` rather than reading ``value_per_kpi`` directly.
    """

    value_per_kpi: float | None
    source: str
    kind: KpiKind = KpiKind.REVENUE
    currency: str = "USD"
    warnings: list[str] = field(default_factory=list)

    @property
    def is_dollar(self) -> bool:
        return self.value_per_kpi is not None

    def require(self, what: str) -> float:
        """The value, or :class:`UnresolvedValueError` naming what needed it."""
        if self.value_per_kpi is None:
            raise UnresolvedValueError(what)
        return self.value_per_kpi

    def describe(self) -> str:
        if not self.is_dollar:
            reason = (
                "the KPI is not denominated in money or units"
                if self.kind is KpiKind.OTHER
                else "no valuation is set"
            )
            return f"Not dollar-denominated ({reason})."
        return (
            f"{self.value_per_kpi:,.4g} {self.currency} per KPI unit "
            f"(kind={self.kind.value}, source={self.source})."
        )

    def to_dict(self) -> dict[str, Any]:
        """Payload shape carried by plan rows, run metrics and REST responses."""
        return {
            "value_per_kpi": self.value_per_kpi,
            "source": self.source,
            "kind": self.kind.value,
            "currency": self.currency,
            "is_dollar": self.is_dollar,
            "warnings": list(self.warnings),
        }


def _economics(blob: Any) -> dict:
    """The ``economics`` sub-dict of a preferences/branding blob, if present."""
    if not isinstance(blob, dict):
        return {}
    econ = blob.get("economics")
    return econ if isinstance(econ, dict) else {}


def _coerce(blob: Any, *, source: str, warnings: list[str]) -> KpiValuation | None:
    """A stored blob into a validated valuation, or ``None`` with a warning.

    A stored preference is untrusted input — it may predate the bounds, or have
    been written by an agent. Invalid entries are SKIPPED with a warning rather
    than raising, so one bad saved preference cannot break every plan; the
    precedence chain simply falls through to the next candidate.
    """
    if isinstance(blob, KpiValuation):
        return blob
    if not isinstance(blob, dict) or not blob:
        return None
    payload = {k: v for k, v in blob.items() if k in KpiValuation.model_fields}
    if not payload:
        return None
    try:
        return KpiValuation.model_validate(payload)
    except Exception as exc:
        hint = ""
        gm = payload.get("gross_margin")
        if isinstance(gm, (int, float)) and gm > 1:
            hint = f" (gross_margin is a fraction — {float(gm) / 100:g}, not {gm:g})"
        warnings.append(f"ignored invalid {source} valuation: {exc}{hint}")
        return None


def kpi_to_dollars(
    *,
    override: KpiValuation | dict | None = None,
    spec: dict | None = None,
    preferences: dict | None = None,
    branding: dict | None = None,
) -> ResolvedValue:
    """Resolve what one KPI unit is worth, once, with provenance.

    Parameters
    ----------
    override
        An explicit valuation from the caller. Highest precedence.
    spec
        A model spec; its ``valuation`` block is read.
    preferences, branding
        Project-level blobs; their ``economics`` sub-dicts are read.

    Returns
    -------
    ResolvedValue
        ``is_dollar=False`` when nothing resolved, or when the KPI kind is
        ``other``. Never silently 1.0.

    Examples
    --------
    >>> kpi_to_dollars(
    ...     override={"kind": "units", "gross_margin": 0.6, "price": 10.0}
    ... ).value_per_kpi
    6.0
    >>> kpi_to_dollars().is_dollar
    False
    """
    warnings: list[str] = []
    candidates: list[tuple[Any, str]] = [
        (override, "param"),
        ((spec or {}).get("valuation"), "spec"),
        (_economics(preferences), "preferences"),
        (_economics(branding), "branding"),
    ]

    for blob, source in candidates:
        valuation = _coerce(blob, source=source, warnings=warnings)
        if valuation is None:
            continue
        value = valuation.value_per_kpi()
        if value is None:
            # A declared 'other' KPI is a resolved ANSWER — it is genuinely not
            # convertible — so stop here rather than falling through to a
            # lower-precedence blob that would contradict the explicit choice.
            if valuation.kind is KpiKind.OTHER:
                return ResolvedValue(
                    value_per_kpi=None,
                    source=source,
                    kind=valuation.kind,
                    currency=valuation.currency,
                    warnings=warnings,
                )
            continue
        return ResolvedValue(
            value_per_kpi=value,
            source=source,
            kind=valuation.kind,
            currency=valuation.currency,
            warnings=warnings,
        )

    return ResolvedValue(value_per_kpi=None, source="none", warnings=warnings)
