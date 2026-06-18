"""
Pre-fit dataset validation.

Complements the load-time schema checks in
:mod:`mmm_framework.data_loader` with quality checks that would silently
bias (warning) or break (error) a fit: missingness patterns, date gaps,
duplicates, degenerate series, negative spend, scale pathologies, short
history for the configured spec, and panel consistency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DataValidationConfig
from .loading import EDAPanel
from .results import DataValidationReport, ValidationIssue


class DataValidator:
    """Run every ``check_*`` method over a panel and collect issues."""

    def __init__(
        self,
        panel: EDAPanel,
        config: DataValidationConfig | None = None,
        spec: dict | None = None,
    ):
        self.panel = panel
        self.config = config or DataValidationConfig()
        self.spec = spec or {}

    def run(self) -> DataValidationReport:
        issues: list[ValidationIssue] = []
        for name in sorted(dir(self)):
            if name.startswith("check_"):
                issues.extend(getattr(self, name)())
        periods = self._periods()
        return DataValidationReport(
            issues=issues,
            n_periods=len(periods),
            n_variables=len(self.panel.variables),
            config=self.config,
        )

    # -- helpers -------------------------------------------------------------

    def _periods(self) -> pd.DatetimeIndex:
        idx = self.panel.df_wide.index
        values = idx.get_level_values(self.panel.date_col) if self.panel.dims else idx
        return pd.DatetimeIndex(values.unique()).sort_values()

    # -- checks (alphabetical = execution order) ------------------------------

    def check_date_gaps(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        periods = self._periods()
        if len(periods) < 3 or self.panel.freq is None:
            return issues
        expected = pd.date_range(periods.min(), periods.max(), freq=self.panel.freq)
        missing = expected.difference(periods)
        if len(missing):
            issues.append(
                ValidationIssue(
                    check="date_gaps",
                    severity="error",
                    message=(
                        f"{len(missing)} period(s) missing from an otherwise "
                        f"{self.panel.freq} cadence (e.g. "
                        f"{', '.join(str(d.date()) for d in missing[:5])}"
                        f"{'…' if len(missing) > 5 else ''}). Adstock transforms "
                        "assume contiguous periods — gaps silently shift carryover."
                    ),
                    affected=[str(d.date()) for d in missing],
                )
            )
        return issues

    def check_duplicate_rows(self) -> list[ValidationIssue]:
        if self.panel.duplicate_rows <= 0:
            return []
        return [
            ValidationIssue(
                check="duplicate_rows",
                severity="error",
                message=(
                    f"{self.panel.duplicate_rows} duplicate (variable, period"
                    f"{', ' + ', '.join(self.panel.dims) if self.panel.dims else ''}) "
                    "row(s) found. Only the first occurrence was used here, but "
                    "the model loader may aggregate them — resolve the duplicates "
                    "at the source."
                ),
                affected=[self.panel.duplicate_rows],
            )
        ]

    def check_constant_series(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for var in self.panel.variables:
            for dim_values, series in self.panel.slices(var):
                values = series.to_numpy(dtype=float)
                if len(values) == 0:
                    continue
                mean = float(np.nanmean(values))
                std = float(np.nanstd(values))
                cv = std / abs(mean) if mean != 0 else std
                if cv < self.config.near_constant_cv:
                    where = f" in {dim_values}" if dim_values else ""
                    issues.append(
                        ValidationIssue(
                            check="constant_series",
                            severity="warning",
                            message=(
                                f"`{var}`{where} is (near-)constant — it carries no "
                                "signal and is unidentifiable from the intercept."
                            ),
                            variable=var,
                        )
                    )
                    break  # one issue per variable is enough
        return issues

    def check_missingness(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        cfg = self.config
        for var in self.panel.variables:
            col = self.panel.df_wide[var]
            pct = float(col.isna().mean() * 100.0)
            if pct >= cfg.missing_error_pct:
                severity = "error"
            elif pct >= cfg.missing_warn_pct:
                severity = "warning"
            else:
                continue
            issues.append(
                ValidationIssue(
                    check="missingness",
                    severity=severity,
                    message=f"`{var}` is missing in {pct:.1f}% of cells.",
                    variable=var,
                )
            )
        return issues

    def check_negative_spend(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for var in self.panel.media:
            col = self.panel.df_wide[var]
            neg = col[col < 0]
            if len(neg):
                issues.append(
                    ValidationIssue(
                        check="negative_spend",
                        severity="error",
                        message=(
                            f"`{var}` has {len(neg)} negative value(s) "
                            f"(min {float(neg.min()):.4g}). Media spend must be "
                            "non-negative — likely a credit/refund or sign error."
                        ),
                        variable=var,
                        affected=[str(i) for i in neg.index[:10].tolist()],
                    )
                )
        return issues

    def check_panel_consistency(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if not self.panel.dims:
            return issues
        dim = self.panel.dims[0]
        wide = self.panel.df_wide
        slice_values = wide.index.get_level_values(dim).unique()
        for var in self.panel.variables:
            present_in = [
                v for v in slice_values if wide[var].xs(v, level=dim).notna().any()
            ]
            absent_in = [v for v in slice_values if v not in present_in]
            if present_in and absent_in:
                issues.append(
                    ValidationIssue(
                        check="panel_consistency",
                        severity="warning",
                        message=(
                            f"`{var}` is present in {len(present_in)} {dim} "
                            f"value(s) but entirely absent in "
                            f"{', '.join(map(str, absent_in[:5]))}"
                            f"{'…' if len(absent_in) > 5 else ''}."
                        ),
                        variable=var,
                        affected=[str(v) for v in absent_in],
                    )
                )
        # Misaligned period ranges across slices (checked on the KPI if known).
        probe = self.panel.kpi or self.panel.variables[0]
        ranges = {}
        for v in slice_values:
            sub = wide[probe].xs(v, level=dim).dropna()
            if len(sub):
                idx = sub.index.get_level_values(self.panel.date_col)
                ranges[v] = (idx.min(), idx.max())
        if len(set(ranges.values())) > 1:
            shown = ", ".join(
                f"{k}: {a.date()}→{b.date()}" for k, (a, b) in list(ranges.items())[:4]
            )
            issues.append(
                ValidationIssue(
                    check="panel_consistency",
                    severity="warning",
                    message=(
                        f"`{probe}` covers different period ranges across {dim} "
                        f"values ({shown})."
                    ),
                    variable=probe,
                )
            )
        return issues

    def check_scale_pathology(self) -> list[ValidationIssue]:
        scales = {}
        for var in self.panel.variables:
            col = self.panel.df_wide[var].abs()
            med = float(col.median())
            if np.isfinite(med) and med > 0:
                scales[var] = med
        if len(scales) < 2:
            return []
        biggest = max(scales, key=scales.get)
        smallest = min(scales, key=scales.get)
        ratio = scales[biggest] / scales[smallest]
        if ratio > self.config.scale_ratio_threshold:
            return [
                ValidationIssue(
                    check="scale_pathology",
                    severity="warning",
                    message=(
                        f"Variable scales span {ratio:.1e}x (`{biggest}` median "
                        f"~{scales[biggest]:.4g} vs `{smallest}` median "
                        f"~{scales[smallest]:.4g}) — check for unit mismatches "
                        "(e.g. dollars vs $000s)."
                    ),
                )
            ]
        return []

    def check_short_history(self) -> list[ValidationIssue]:
        spec = self.spec
        n_periods = len(self._periods())
        if not spec or not spec.get("media_channels") or n_periods == 0:
            return []
        n_media = len(spec.get("media_channels") or [])
        n_controls = len(spec.get("control_variables") or [])
        # Rough effective parameter count: intercept + sigma + per-channel
        # (beta, adstock, 2x saturation) + controls + seasonality + trend.
        approx_params = 2 + 4 * n_media + n_controls + 4 + 2
        ratio = n_periods / approx_params
        if ratio < self.config.min_obs_per_param:
            return [
                ValidationIssue(
                    check="short_history",
                    severity="warning",
                    message=(
                        f"{n_periods} periods vs ~{approx_params} effective "
                        f"parameters ({ratio:.1f} obs/param < "
                        f"{self.config.min_obs_per_param:g}). Posteriors will "
                        "lean heavily on priors — consider fewer channels, "
                        "pooled priors, or calibration experiments."
                    ),
                )
            ]
        return []

    def check_zero_inflation(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for var in self.panel.media:
            col = self.panel.df_wide[var].dropna()
            if len(col) == 0:
                continue
            zero_pct = float((col == 0).mean() * 100.0)
            if zero_pct > self.config.zero_inflation_warn_pct:
                issues.append(
                    ValidationIssue(
                        check="zero_inflation",
                        severity="warning",
                        message=(
                            f"`{var}` is zero in {zero_pct:.0f}% of periods — "
                            "fine for a strongly flighted channel, but verify "
                            "zeros mean 'dark', not 'unrecorded'."
                        ),
                        variable=var,
                    )
                )
        return issues


def validate_dataset(
    panel: EDAPanel,
    config: DataValidationConfig | None = None,
    spec: dict | None = None,
) -> DataValidationReport:
    """Functional wrapper around :class:`DataValidator`."""
    return DataValidator(panel, config, spec).run()


__all__ = ["DataValidator", "validate_dataset"]
