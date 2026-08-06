"""Where a number on a bridge came from, and whether it hides anything.

A waterfall, a P&L rollup and a CFO one-pager are all the same object: a list of
lines that are supposed to add up to a stated total. The lines look alike on the
page and are not alike at all. Some are read off the fit's own components. Some
are data totals with no model in them. And one kind is neither: it is *whatever
is left over*, computed as ``observed - modelled media`` and then labelled "base
demand", which makes the model's residual invisible by construction. The bridge
closes because it was defined to close.

That last kind was the shipped default across several surfaces (issue #220).
Measured on a real report, a baseline bar built that way carried **15.7% of its
own value** as absorbed residual. So the vocabulary here has four entries, not
two, and the fourth exists to be *named* rather than avoided: a fallback that
absorbs the residual is sometimes the only number available, and the honest move
is to render it saying so.

**Absence reads as** :data:`ABSORBING`. Every bridge line written before this
module existed was the leftover kind, and a line that does not state its
provenance is one of those until someone says otherwise. Defaulting the other
way would silently promote old numbers to "modelled".

Lean: dataclasses and the standard library. No numpy, no reporting, no model
import — the reporting stack, the deck, the interactive report and the agent all
build lines, and none of them should have to agree on more than this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "ABSORBING",
    "MODELLED",
    "OBSERVED",
    "RESIDUAL",
    "SUPPLIED",
    "BridgeLine",
    "LineProvenance",
    "absorbs_residual",
    "bridge_gap",
    "provenance_of",
]


class LineProvenance(str, Enum):
    """How a bridge line's value was arrived at.

    A ``str`` enum so it round-trips through JSON, the plan payloads and the
    ``InferenceData`` attrs without an enum import on the far side.
    """

    MODELLED = "modelled"
    """Read off the fit's own components. Sums with its siblings to the FITTED
    total, which is not the observed total."""

    OBSERVED = "observed"
    """A total taken straight from the data. No model, no residual, no
    interval."""

    RESIDUAL = "residual"
    """The named gap between fitted and observed. Disclosing this is what lets
    the modelled lines stay modelled."""

    ABSORBING = "absorbing"
    """Computed as a leftover (``observed - modelled media``), so it carries the
    residual inside a bar labelled something else. Renderable, but only with the
    caveat attached."""

    SUPPLIED = "supplied"
    """A human-supplied adjustment (gross-to-net, returns, trade spend): a
    point value with a REQUIRED source note and no interval fields — supplied
    numbers have no sampling distribution to summarise, and rendering one with
    a band would dress an assertion as an estimate (issue #227)."""


#: Module-level aliases, mirroring ``diagnostics.provenance``'s BAYESIAN /
#: FREQUENTIST, so call sites can compare against a plain string.
MODELLED = LineProvenance.MODELLED
OBSERVED = LineProvenance.OBSERVED
RESIDUAL = LineProvenance.RESIDUAL
ABSORBING = LineProvenance.ABSORBING
SUPPLIED = LineProvenance.SUPPLIED


def provenance_of(value: Any) -> LineProvenance:
    """The provenance a value declares, defaulting to :data:`ABSORBING`.

    Accepts a :class:`LineProvenance`, a :class:`BridgeLine`, a plain string, a
    payload dict carrying a ``provenance`` key, or ``None``. Anything
    unrecognised reads as :data:`ABSORBING` — see the module docstring for why
    the unstated case is the pessimistic one.
    """
    if isinstance(value, LineProvenance):
        return value
    if isinstance(value, BridgeLine):
        return value.provenance
    if isinstance(value, dict):
        value = value.get("provenance")
    if isinstance(value, str):
        try:
            return LineProvenance(value.lower())
        except ValueError:
            return ABSORBING
    return ABSORBING


def absorbs_residual(value: Any) -> bool:
    """``True`` when this line hides the model residual inside its own value."""
    return provenance_of(value) is ABSORBING


@dataclass(frozen=True)
class BridgeLine:
    """One line of a bridge: a name, a number, and where the number came from.

    ``lower``/``upper`` are optional because two of the four provenances have no
    interval to carry — an observed total has no sampling distribution here, and
    a residual is a single arithmetic difference. When they are present,
    ``interval_mass`` must be too: an interval whose mass is unstated is the
    defect #277 fixed elsewhere in this codebase, and there is no reason to
    reintroduce it on a bridge.
    """

    name: str
    value: float
    provenance: LineProvenance = ABSORBING
    lower: float | None = None
    upper: float | None = None
    interval_mass: float | None = None
    basis: str = ""
    """Short machine tag for *how* a modelled line was derived — e.g.
    ``"components"``, ``"mu"``, ``"predictive_mean"``. Free-form; renderers show
    it only in provenance detail."""
    note: str = ""
    """Caveat to render with the line. Required reading for an ABSORBING line."""
    source_note: str = ""
    """Where a SUPPLIED number came from — required, non-blank, for SUPPLIED
    lines. "Finance sent it over" is not a source; name the document, system
    or person."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", provenance_of(self.provenance))
        object.__setattr__(self, "value", float(self.value))
        if self.provenance is SUPPLIED:
            if not str(self.source_note).strip():
                raise ValueError(
                    f"BridgeLine {self.name!r}: a SUPPLIED line requires a "
                    "non-blank source_note — an unattributed adjustment is "
                    "exactly the unauditable plug this vocabulary exists to "
                    "prevent."
                )
            if self.lower is not None or self.upper is not None:
                raise ValueError(
                    f"BridgeLine {self.name!r}: a SUPPLIED line carries no "
                    "interval — a supplied number has no sampling "
                    "distribution, and a band would dress an assertion as an "
                    "estimate."
                )
        if (self.lower is None) != (self.upper is None):
            raise ValueError(
                f"BridgeLine {self.name!r}: an interval needs both bounds; got "
                f"lower={self.lower!r}, upper={self.upper!r}."
            )
        if self.lower is not None and self.interval_mass is None:
            raise ValueError(
                f"BridgeLine {self.name!r}: an interval must state its mass. "
                "Pass interval_mass (e.g. 0.90) alongside lower/upper."
            )

    @property
    def has_interval(self) -> bool:
        return self.lower is not None and self.upper is not None

    @property
    def absorbs_residual(self) -> bool:
        """``True`` when this line's value contains the model residual."""
        return self.provenance is ABSORBING

    def describe(self) -> str:
        """One-line human reading, provenance included.

        The provenance is in the sentence rather than in a tooltip because the
        whole point of the line is that two bars can look identical and mean
        different things.
        """
        head = f"{self.name}: {self.value:,.4g}"
        if self.has_interval:
            pct = int(round(float(self.interval_mass or 0.0) * 100))
            head += f" ({pct}% interval {self.lower:,.4g} to {self.upper:,.4g})"
        if self.provenance is MODELLED:
            tail = "modelled" + (f" from {self.basis}" if self.basis else "")
        elif self.provenance is OBSERVED:
            tail = "observed"
        elif self.provenance is RESIDUAL:
            tail = "residual, observed minus fitted"
        elif self.provenance is SUPPLIED:
            tail = f"supplied ({self.source_note})"
        else:
            tail = "leftover, so it also carries the model residual"
        out = f"{head} [{tail}]"
        return f"{out} {self.note}".rstrip() if self.note else out

    def to_dict(self) -> dict[str, Any]:
        """Payload shape carried by report facts, plan rows and REST responses."""
        return {
            "name": self.name,
            "value": self.value,
            "provenance": self.provenance.value,
            "lower": self.lower,
            "upper": self.upper,
            "interval_mass": self.interval_mass,
            "basis": self.basis,
            "note": self.note,
            "source_note": self.source_note,
            "absorbs_residual": self.absorbs_residual,
        }


@dataclass(frozen=True)
class BridgeGap:
    """What a set of lines fails to account for, against a stated target."""

    target: float
    lines_total: float
    gap: float
    gap_pct: float | None
    closes: bool
    absorbing_lines: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if self.closes:
            msg = f"Bridge closes to {self.target:,.4g}."
            if self.absorbing_lines:
                # The case worth writing carefully. A bridge containing a
                # leftover line closes *by construction* — the leftover was
                # defined as whatever made it close — so "it closes" is the
                # least informative true sentence available here.
                msg += (
                    " It closes by construction: "
                    + ", ".join(self.absorbing_lines)
                    + " is a leftover line and absorbs whatever the other lines"
                    " do not account for, so this understates what is"
                    " unexplained rather than showing there is nothing left."
                )
            return msg
        pct = f"{self.gap_pct:+.4%}" if self.gap_pct is not None else "n/a"
        msg = (
            f"Bridge does not close: lines total {self.lines_total:,.4g} against "
            f"a target of {self.target:,.4g}, a gap of {self.gap:,.4g} ({pct})."
        )
        if self.absorbing_lines:
            msg += (
                " Note that "
                + ", ".join(self.absorbing_lines)
                + " already absorb the residual, so the stated gap understates"
                " what is unaccounted for."
            )
        return msg


def bridge_gap(
    lines: "list[BridgeLine]",
    target: float,
    *,
    tol_ratio: float = 1e-6,
) -> BridgeGap:
    """Whether ``lines`` add up to ``target``, and by how much they miss.

    ``tol_ratio`` is relative to ``abs(target)``, with an absolute floor so a
    target of zero does not demand exact bit equality.

    A closing bridge is **not** evidence the lines are right. A bridge built
    from an ABSORBING line always closes, because the absorbing line was defined
    as the amount needed to close it; that is why the returned object names
    those lines rather than reporting a clean zero and stopping.
    """
    total = float(sum(line.value for line in lines))
    target = float(target)
    gap = target - total
    tol = max(abs(target) * float(tol_ratio), 1e-9)
    return BridgeGap(
        target=target,
        lines_total=total,
        gap=gap,
        gap_pct=(gap / target if abs(target) > 1e-12 else None),
        closes=abs(gap) <= tol,
        absorbing_lines=[ln.name for ln in lines if ln.absorbs_residual],
    )
