"""Measurement-aware ROI / efficiency divisor resolution.

A media channel's modeled variable is not always dollars. When it is
**impressions** or **clicks**, you cannot form ROI by summing the variable.
This module is the single place that turns a channel's measurement descriptor
(:class:`~mmm_framework.config.enums.MeasurementUnit` plus the companion
``spend_column`` / ``cpm`` / ``cpc`` fields on
:class:`~mmm_framework.config.variables.MediaChannelConfig`) into:

1. the correct **divisor** for ROI / marginal-ROAS-style metrics, and
2. the **labels / units / break-even reference** that describe the resulting
   number,

so every ROI site (the dashboard ROI, counterfactual ROI, marginal ROAS, the
declarative estimands) and the report / UI stay consistent.

The key simplification: the resolver returns a single ``divisor_total`` — the
*spend-equivalent or volume-equivalent of the (masked) window*. That one number
serves both metrics:

* **average** ROI / efficiency  =  ``contribution / divisor_total``
* **marginal** ROAS / efficiency denominator  =  ``divisor_total * (factor - 1)``
  for a ``ScaleInput(factor)`` perturbation.

Resolution precedence for a channel (highest first):

(d) ``measurement_unit == SPEND`` (the default) → divisor = summed modeled
    variable. **Byte-identical to the historical behavior.**
(a) ``spend_column`` set → divisor = summed external spend series → ROI.
(b) ``cpm`` set → divisor = ``(volume / 1000) * cpm`` → ROI (impressions).
    ``cpc`` set → divisor = ``volume * cpc`` → ROI (clicks).
(c) a volume unit with **no** cost → divisor = ``volume / 1000`` (impressions)
    or ``volume`` (clicks / other) → **efficiency** per 1,000 impressions /
    per click / per unit, whose break-even reference is **0**, not 1.0.

This module imports only numpy + the config enum; it pulls model attributes
defensively (``getattr``) so it also works against the lightweight fake models
used in tests and the extended-MMM family.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from mmm_framework.config.enums import MeasurementUnit

__all__ = [
    "MetricMeta",
    "ChannelDivisor",
    "resolve_channel_divisor",
    "metric_meta_for_channel",
    "resolve_spend_dict",
    "spend_metric_meta",
]


@dataclass(frozen=True)
class MetricMeta:
    """How a channel's ROI-style number should be labeled and interpreted.

    Carried alongside every resolved number so reports, agent ops and the UI
    label it correctly without re-deriving the measurement logic.
    """

    unit: MeasurementUnit
    #: True when the divisor is dollars (an ROI / mROAS), False when it is a
    #: volume (an efficiency per 1,000 impressions / per click / per unit).
    is_monetary: bool
    #: How the cost was obtained: ``"spend_column"`` / ``"cpm"`` / ``"cpc"`` /
    #: ``None`` (the variable already is spend, or no cost is known).
    cost_basis: str | None
    #: Display label for the average metric (e.g. ``"ROI"`` or
    #: ``"Efficiency per 1K impressions"``).
    roi_label: str
    #: Display label for the marginal metric (e.g. ``"Marginal ROAS"`` or
    #: ``"Marginal efficiency per +1K impressions"``).
    marginal_label: str
    #: Short units suffix for a value (e.g. ``"ROI"`` or ``"KPI / 1K impr"``).
    value_units: str
    #: Units of the divisor itself, for a spend/volume column header
    #: (``"$"`` / ``"1K impressions"`` / ``"clicks"`` / ``"units"``).
    divisor_units: str
    #: The no-effect / break-even reference the value is judged against:
    #: ``1.0`` for ROI (return per dollar), ``0.0`` for efficiency.
    reference: float

    @property
    def supports_profitability(self) -> bool:
        """Whether a ``prob_profitable`` / break-even-at-1.0 reading is
        meaningful. False for efficiency metrics (there is no cost to break
        even against — report ``prob_positive`` only)."""
        return self.is_monetary

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit.value,
            "is_monetary": self.is_monetary,
            "cost_basis": self.cost_basis,
            "roi_label": self.roi_label,
            "marginal_label": self.marginal_label,
            "value_units": self.value_units,
            "divisor_units": self.divisor_units,
            "reference": self.reference,
            "supports_profitability": self.supports_profitability,
        }


@dataclass(frozen=True)
class ChannelDivisor:
    """The resolved divisor for one channel over a (masked) window."""

    total: float
    #: False when the channel's modeled series could not be located (the caller
    #: should skip it, mirroring the legacy ``spend <= 0`` skip).
    found: bool
    meta: MetricMeta


# --- metric metadata ---------------------------------------------------------


def _meta_monetary(unit: MeasurementUnit, cost_basis: str | None) -> MetricMeta:
    return MetricMeta(
        unit=unit,
        is_monetary=True,
        cost_basis=cost_basis,
        roi_label="ROI",
        marginal_label="Marginal ROAS",
        value_units="ROI",
        divisor_units="$",
        reference=1.0,
    )


def _meta_efficiency(unit: MeasurementUnit) -> MetricMeta:
    if unit is MeasurementUnit.IMPRESSIONS:
        return MetricMeta(
            unit=unit,
            is_monetary=False,
            cost_basis=None,
            roi_label="Efficiency per 1K impressions",
            marginal_label="Marginal efficiency per +1K impressions",
            value_units="KPI / 1K impr",
            divisor_units="1K impressions",
            reference=0.0,
        )
    if unit is MeasurementUnit.CLICKS:
        return MetricMeta(
            unit=unit,
            is_monetary=False,
            cost_basis=None,
            roi_label="Efficiency per click",
            marginal_label="Marginal efficiency per +1 click",
            value_units="KPI / click",
            divisor_units="clicks",
            reference=0.0,
        )
    return MetricMeta(
        unit=unit,
        is_monetary=False,
        cost_basis=None,
        roi_label="Efficiency per unit",
        marginal_label="Marginal efficiency per +1 unit",
        value_units="KPI / unit",
        divisor_units="units",
        reference=0.0,
    )


def _intended_cost_basis(cfg: Any) -> str | None:
    if cfg is None:
        return None
    if getattr(cfg, "spend_column", None) is not None:
        return "spend_column"
    if getattr(cfg, "cpm", None) is not None:
        return "cpm"
    if getattr(cfg, "cpc", None) is not None:
        return "cpc"
    return None


def spend_metric_meta() -> MetricMeta:
    """A normal monetary (ROI, break-even 1.0) :class:`MetricMeta` — used when a
    caller supplies an explicit external spend series (the legacy path)."""
    return _meta_monetary(MeasurementUnit.SPEND, None)


def metric_meta_for_channel(model: Any, channel: str) -> MetricMeta:
    """The :class:`MetricMeta` for ``channel`` from config alone (no data needed).

    Useful for labeling — including pre-fit / pre-data surfaces. Assumes a
    declared ``spend_column`` will be available; the actual divisor path may
    degrade to efficiency if it is not (see :func:`resolve_channel_divisor`).
    """
    cfg = _get_media_cfg(model, channel)
    unit = _unit_of(cfg)
    if unit is MeasurementUnit.SPEND:
        return _meta_monetary(unit, None)
    basis = _intended_cost_basis(cfg)
    if basis is not None:
        return _meta_monetary(unit, basis)
    return _meta_efficiency(unit)


# --- model introspection -----------------------------------------------------


def _get_mff_config(model: Any) -> Any:
    cfg = getattr(model, "mff_config", None)
    if cfg is not None:
        return cfg
    panel = getattr(model, "panel", None)
    return getattr(panel, "config", None)


def _get_media_cfg(model: Any, channel: str) -> Any:
    mff = _get_mff_config(model)
    if mff is None:
        return None
    getter = getattr(mff, "get_media_config", None)
    if callable(getter):
        return getter(channel)
    return None


def _unit_of(cfg: Any) -> MeasurementUnit:
    """The channel's :class:`MeasurementUnit`, defaulting to ``SPEND``.

    Anything that is not a real ``MeasurementUnit`` value (e.g. a duck-typed /
    mock config in a test) falls back to ``SPEND`` so the resolver degrades to
    the legacy "sum the modeled variable" behavior rather than erroring."""
    if cfg is None:
        return MeasurementUnit.SPEND
    unit = getattr(cfg, "measurement_unit", MeasurementUnit.SPEND)
    if isinstance(unit, MeasurementUnit):
        return unit
    try:
        return MeasurementUnit(unit)
    except (ValueError, TypeError):
        return MeasurementUnit.SPEND


def _channel_index(model: Any, channel: str) -> int | None:
    names = getattr(model, "channel_names", None)
    if names is None:
        names = getattr(getattr(model, "panel", None), "X_media", None)
        names = (
            list(names.columns)
            if names is not None and hasattr(names, "columns")
            else None
        )
    if names is None:
        return None
    try:
        return list(names).index(channel)
    except ValueError:
        return None


def _column_series(data: Any, col_idx: int, col_name: str) -> np.ndarray | None:
    """Per-observation 1-D series for a channel from a DataFrame/array, mirroring
    the column-finding precedence of the legacy ``_extract_spend_from_model``."""
    if data is None:
        return None
    try:
        if hasattr(data, "columns") and col_name in getattr(data, "columns", []):
            return np.asarray(data[col_name].values, dtype=np.float64)
        if hasattr(data, "iloc"):
            if col_idx < data.shape[1]:
                return np.asarray(data.iloc[:, col_idx].values, dtype=np.float64)
        elif hasattr(data, "values"):
            arr = data.values
            if col_idx < arr.shape[1]:
                return np.asarray(arr[:, col_idx], dtype=np.float64)
        else:
            if col_idx < data.shape[1]:
                return np.asarray(data[:, col_idx], dtype=np.float64)
    except Exception:  # noqa: BLE001 - extraction is best-effort
        return None
    return None


def _channel_volume_series(model: Any, channel: str) -> np.ndarray | None:
    """The channel's modeled variable as a per-obs 1-D array.

    Precedence (matches the legacy spend extractor so the SPEND default stays
    byte-identical): ``panel.X_media`` → ``X_media_raw`` → ``X_media``.
    """
    idx = _channel_index(model, channel)
    if idx is None:
        return None
    panel = getattr(model, "panel", None)
    sources = [
        getattr(panel, "X_media", None) if panel is not None else None,
        getattr(model, "X_media_raw", None),
        getattr(model, "X_media", None),
    ]
    for src in sources:
        series = _column_series(src, idx, channel)
        if series is not None:
            return series
    return None


def _channel_spend_series(model: Any, channel: str) -> np.ndarray | None:
    """The external per-obs spend series for ``channel`` (option a), if loaded.

    ``model.spend_raw`` is a ``{channel: np.ndarray}`` mapping populated by the
    loader when a channel declares a ``spend_column``. Absent ⇒ ``None``.
    """
    spend_raw = getattr(model, "spend_raw", None)
    if spend_raw is None:
        return None
    series = None
    if isinstance(spend_raw, dict):
        series = spend_raw.get(channel)
    else:  # array-like aligned to channel order
        idx = _channel_index(model, channel)
        if idx is not None:
            series = _column_series(spend_raw, idx, channel)
    if series is None:
        return None
    return np.asarray(series, dtype=np.float64)


def _masked_sum(series: np.ndarray, mask: np.ndarray | None) -> float:
    if mask is None:
        return float(np.asarray(series, dtype=np.float64).sum())
    series = np.asarray(series, dtype=np.float64)
    m = np.asarray(mask)
    # tolerate a length mismatch (e.g. a geo panel mask vs a national series)
    if m.dtype == bool and m.shape[0] == series.shape[0]:
        return float(series[m].sum())
    return float(series.sum())


def _efficiency_divisor(unit: MeasurementUnit, volume: float) -> float:
    if unit is MeasurementUnit.IMPRESSIONS:
        return volume / 1000.0
    return volume  # clicks / other: per click / per unit


# --- the resolver ------------------------------------------------------------


def resolve_channel_divisor(
    model: Any, channel: str, mask: np.ndarray | None = None
) -> ChannelDivisor:
    """Resolve the ROI/efficiency divisor + labels for one channel.

    Parameters
    ----------
    model:
        A fitted MMM exposing ``channel_names`` + the modeled media (and,
        optionally, ``mff_config`` for the measurement descriptor and
        ``spend_raw`` for an external spend column).
    channel:
        The channel name.
    mask:
        Optional boolean window mask over observations. ``None`` ⇒ full series.
    """
    cfg = _get_media_cfg(model, channel)
    unit = _unit_of(cfg)

    vol_series = _channel_volume_series(model, channel)
    if vol_series is None:
        return ChannelDivisor(0.0, False, metric_meta_for_channel(model, channel))
    volume = _masked_sum(vol_series, mask)

    # (d) default / explicit spend — byte-identical to the legacy sum.
    if cfg is None or unit is MeasurementUnit.SPEND:
        return ChannelDivisor(volume, True, _meta_monetary(unit, None))

    # (a) external spend column.
    if getattr(cfg, "spend_column", None) is not None:
        spend_series = _channel_spend_series(model, channel)
        if spend_series is not None:
            spend = _masked_sum(spend_series, mask)
            return ChannelDivisor(spend, True, _meta_monetary(unit, "spend_column"))
        warnings.warn(
            f"Channel '{channel}' declares spend_column="
            f"{getattr(cfg, 'spend_column', None)!r} but no spend series is loaded "
            "on the model (model.spend_raw); reporting efficiency per volume "
            "instead of ROI.",
            stacklevel=2,
        )
        return ChannelDivisor(
            _efficiency_divisor(unit, volume), True, _meta_efficiency(unit)
        )

    # (b) cost constants → derive a spend series → ROI.
    cpm = getattr(cfg, "cpm", None)
    if cpm is not None:
        return ChannelDivisor(volume / 1000.0 * cpm, True, _meta_monetary(unit, "cpm"))
    cpc = getattr(cfg, "cpc", None)
    if cpc is not None:
        return ChannelDivisor(volume * cpc, True, _meta_monetary(unit, "cpc"))

    # (c) no cost known → efficiency per 1,000 impressions / per click / unit.
    return ChannelDivisor(
        _efficiency_divisor(unit, volume), True, _meta_efficiency(unit)
    )


def resolve_spend_dict(model: Any) -> dict[str, float]:
    """``{channel: divisor_total}`` over the full series for every locatable
    channel — the measurement-aware replacement for the legacy spend extractor."""
    out: dict[str, float] = {}
    for ch in getattr(model, "channel_names", []) or []:
        d = resolve_channel_divisor(model, ch)
        if d.found:
            out[ch] = d.total
    return out
