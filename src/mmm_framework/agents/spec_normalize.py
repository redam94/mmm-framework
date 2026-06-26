"""Spec-normalization helpers, extracted from the agents.tools god-module (H4).

Pure helpers that tolerate the looser specs weaker models emit (bare-string
channels/controls, trend-type aliases, latent components listed as controls) and
classify control names against the dataset. No module-level state, no agent
dependencies — re-exported from ``agents.tools`` for backward compatibility.
"""

from __future__ import annotations

import os
from typing import Any

# Trend-type aliases weaker models emit for the canonical names.
_TREND_TYPE_ALIASES = {
    "piecewise_linear": "piecewise",
    "gp": "gaussian_process",
}

# LLMs / DAG proxies sometimes list the latent baseline components ("Trend",
# "Seasonality") as if they were dataset variables. They are modeled via the
# built-in trend / seasonality components, never as regressor columns — a spec
# that names them as controls fails at load with "Missing expected variables".
_LATENT_CONTROL_COMPONENTS = {
    "trend": "trend",
    "seasonality": "seasonality",
    "season": "seasonality",
}


def _normalize_trend_type(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    t = value.strip().lower().replace("-", "_")
    return _TREND_TYPE_ALIASES.get(t, t)


def _normalize_spec_vars(spec: dict) -> dict:
    """Tolerate ``media_channels`` / ``control_variables`` given as bare name
    strings (``["TV", "Digital"]``) OR as dicts (``[{"name": "TV", ...}]``),
    and trend-type aliases like ``piecewise_linear`` for ``piecewise``.

    Weaker models often emit the simpler string form, which previously crashed
    fit with "string indices must be integers". Normalises every entry to a dict
    with at least a ``name`` key, in place, and returns the spec.
    """
    trend = spec.get("trend")
    if isinstance(trend, dict) and "type" in trend:
        trend["type"] = _normalize_trend_type(trend["type"])
    for key in ("media_channels", "control_variables"):
        items = spec.get(key)
        if isinstance(items, list):
            normalized = []
            for it in items:
                if isinstance(it, str):
                    normalized.append({"name": it})
                elif isinstance(it, dict) and "name" in it:
                    normalized.append(it)
                # silently drop malformed entries (None, numbers, dict w/o name)
            spec[key] = normalized
    return spec


def _dataset_variable_names(dataset_path: str | None) -> set[str] | None:
    """Unique ``VariableName`` values in an MFF dataset, or None when the file
    is absent/unreadable/not long-format (callers then skip validation)."""
    if not dataset_path or not os.path.exists(dataset_path):
        return None
    try:
        import pandas as pd

        if "VariableName" not in pd.read_csv(dataset_path, nrows=0).columns:
            return None
        s = pd.read_csv(dataset_path, usecols=["VariableName"])["VariableName"]
        return set(s.dropna().astype(str).unique())
    except Exception:
        return None


def _partition_latent_controls(
    names: list[str], ds_vars: set[str] | None
) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """Split proposed control names into ``(real, latent, missing)``.

    A name present in the dataset is always a real control (even one literally
    named "Trend"). Otherwise names matching the built-in components are
    diverted to ``latent`` as ``(name, component)`` pairs; the rest go to
    ``missing``. With ``ds_vars`` None (no dataset to check against), non-latent
    names are assumed real.
    """
    real: list[str] = []
    latent: list[tuple[str, str]] = []
    missing: list[str] = []
    for name in names:
        if ds_vars is not None and name in ds_vars:
            real.append(name)
            continue
        comp = _LATENT_CONTROL_COMPONENTS.get(str(name).strip().lower())
        if comp:
            latent.append((name, comp))
        elif ds_vars is None:
            real.append(name)
        else:
            missing.append(name)
    return real, latent, missing


def _normalized_spec(spec: dict | None) -> dict:
    """A deepcopy of ``spec`` with bare-string ``media_channels`` /
    ``control_variables`` normalized to ``{"name": ...}`` dicts.

    For READ/display tools (``get_current_config``, ``save_config``,
    ``load_config``) that subscript ``c['name']`` / ``ch.get(...)`` and would
    otherwise crash with "string indices must be integers" when a weaker model
    emitted the bare-string form. ``fit_mmm_model`` already normalizes its own
    copy; this gives the read paths the same tolerance without mutating state.
    """
    import copy as _copy

    return _normalize_spec_vars(_copy.deepcopy(spec or {}))
