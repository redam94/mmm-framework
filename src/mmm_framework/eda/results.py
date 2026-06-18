"""
Result containers for pre-fit data quality (outliers, validation, EDA).

All containers are JSON-safe via ``to_dict()`` (numpy/pandas types are
round-tripped through :class:`~mmm_framework.reporting.charts.base.NumpyEncoder`)
and human-readable via ``summary()``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from mmm_framework.reporting.charts.base import NumpyEncoder

from .config import DataValidationConfig, OutlierConfig


def _json_safe(obj: Any) -> Any:
    """Round-trip through NumpyEncoder so numpy/pandas scalars become plain JSON."""
    return json.loads(json.dumps(obj, cls=NumpyEncoder, default=str))


# ---------------------------------------------------------------------------
# outliers
# ---------------------------------------------------------------------------


@dataclass
class OutlierFlag:
    """One flagged observation in one series."""

    variable: str
    period: str  # ISO date string
    value: float
    expected: float  # robust baseline (STL fit / rolling median)
    methods: list[str]  # detectors that fired
    score: float  # consensus severity in [0, 1]
    kind: str  # isolated_spike | isolated_drop | kpi_shock | level_shift | heavy_tail_member | low_outlier | point_outlier
    dims: dict[str, str] = field(default_factory=dict)  # e.g. {"Geography": "East"}

    @property
    def flag_id(self) -> str:
        suffix = "".join(f"|{k}={v}" for k, v in sorted(self.dims.items()))
        return f"{self.variable}@{self.period}{suffix}"


@dataclass
class RemediationAction:
    """A concrete, applyable treatment for one or more flags."""

    action_id: str  # e.g. "winsorize:TV@2023-07-04"
    flag_ids: list[str]
    strategy: str  # winsorize | dummy | exclude_periods | note
    params: dict[str, Any]
    rationale: str
    spec_change: dict[str, Any] | None = (
        None  # {"add_control": ...} | {"setting_path": ..., "value": ...}
    )


@dataclass
class OutlierReport:
    """Full outlier-detection output: flags + recommended treatments."""

    flags: list[OutlierFlag]
    actions: list[RemediationAction]
    config: OutlierConfig
    # Per-variable diagnostics, e.g. {"TV": {"n_flags": 1, "max_over_p99": 9.4,
    # "excess_kurtosis": 0.7, "normalization_damaged": True}}
    per_variable: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Provenance for staleness checks when the report is persisted.
    dataset_path: str | None = None
    dataset_mtime: float | None = None

    def flags_frame(self) -> pd.DataFrame:
        rows = [
            {
                "flag_id": f.flag_id,
                "variable": f.variable,
                "period": f.period,
                **f.dims,
                "value": f.value,
                "expected": f.expected,
                "kind": f.kind,
                "score": f.score,
                "methods": ",".join(f.methods),
            }
            for f in self.flags
        ]
        return pd.DataFrame(rows)

    def summary(self) -> str:
        if not self.flags:
            return "No outliers detected."
        lines = [f"{len(self.flags)} flagged observation(s):"]
        for f in sorted(self.flags, key=lambda x: -x.score):
            lines.append(
                f"  - {f.flag_id}: value={f.value:.4g} vs expected={f.expected:.4g} "
                f"[{f.kind}, score={f.score:.2f}, via {','.join(f.methods)}]"
            )
        if self.actions:
            lines.append(f"{len(self.actions)} recommended action(s):")
            for a in self.actions:
                lines.append(f"  - {a.action_id} ({a.strategy}): {a.rationale}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        # flag_id is a property; persist it explicitly for action resolution.
        for flag, raw in zip(self.flags, d["flags"]):
            raw["flag_id"] = flag.flag_id
        return _json_safe(d)

    @classmethod
    def from_dict(cls, d: dict) -> "OutlierReport":
        flags = [
            OutlierFlag(**{k: v for k, v in f.items() if k != "flag_id"})
            for f in d.get("flags", [])
        ]
        actions = [RemediationAction(**a) for a in d.get("actions", [])]
        cfg = OutlierConfig(
            **{
                k: tuple(v) if k == "methods" else v
                for k, v in (d.get("config") or {}).items()
            }
        )
        return cls(
            flags=flags,
            actions=actions,
            config=cfg,
            per_variable=d.get("per_variable", {}),
            dataset_path=d.get("dataset_path"),
            dataset_mtime=d.get("dataset_mtime"),
        )


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


@dataclass
class ValidationIssue:
    """One finding from a single validation check."""

    check: str  # e.g. "date_gaps", "negative_spend"
    severity: str  # "error" | "warning" | "info"
    message: str
    variable: str | None = None
    affected: list[Any] = field(default_factory=list)  # periods / rows / values


@dataclass
class DataValidationReport:
    """All validation findings for a dataset."""

    issues: list[ValidationIssue]
    n_periods: int
    n_variables: int
    config: DataValidationConfig

    @property
    def passed(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def by_severity(self, severity: str) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == severity]

    def summary(self) -> str:
        errors = self.by_severity("error")
        warnings = self.by_severity("warning")
        infos = self.by_severity("info")
        head = (
            f"Validation {'PASSED' if self.passed else 'FAILED'}: "
            f"{len(errors)} error(s), {len(warnings)} warning(s), "
            f"{len(infos)} note(s) across {self.n_variables} variables / "
            f"{self.n_periods} periods."
        )
        lines = [head]
        for i in self.issues:
            var = f" [{i.variable}]" if i.variable else ""
            lines.append(f"  - {i.severity.upper()} {i.check}{var}: {i.message}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return _json_safe(asdict(self))


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------


@dataclass
class DecompositionResult:
    """STL (or fallback) decomposition of one series."""

    variable: str
    method: str  # "stl" | "rolling_median"
    period: int | None
    trend: pd.Series
    seasonal: pd.Series | None
    resid: pd.Series
    trend_strength: float  # 1 - Var(resid)/Var(trend + resid)
    seasonal_strength: float | None  # 1 - Var(resid)/Var(seasonal + resid)

    def to_dict(self) -> dict:
        return _json_safe(
            {
                "variable": self.variable,
                "method": self.method,
                "period": self.period,
                "trend_strength": self.trend_strength,
                "seasonal_strength": self.seasonal_strength,
            }
        )


@dataclass
class EDAReport:
    """Aggregated EDA output keyed by analysis name.

    ``sections`` holds JSON-safe dicts (one per analysis run); ``figures``
    holds the plotly figures keyed by a descriptive name — callers decide how
    to render/store them.
    """

    sections: dict[str, Any] = field(default_factory=dict)
    figures: dict[str, Any] = field(default_factory=dict)  # name -> go.Figure

    def to_dict(self) -> dict:
        return _json_safe(self.sections)


__all__ = [
    "OutlierFlag",
    "RemediationAction",
    "OutlierReport",
    "ValidationIssue",
    "DataValidationReport",
    "DecompositionResult",
    "EDAReport",
]
