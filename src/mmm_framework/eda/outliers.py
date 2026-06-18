"""
Time-series-aware outlier detection for MMM datasets.

The detectors distinguish three patterns that generic z-score screens
conflate:

* **Isolated spikes** in media spend (data-entry errors). These are the most
  damaging failure mode for an MMM: the model normalizes each channel by its
  max, so a single 15x spike compresses every real week into a near-linear
  sliver of the saturation curve (see ``tests/synth/dgp.py::make_spend_outliers``).
* **KPI shocks** (real demand events such as promos) — these should usually be
  *modeled* (event dummy), not deleted.
* **Seasonal peaks** — real signal, NOT outliers. The STL-residual detector
  only flags points that the seasonal + trend structure cannot explain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from .config import METHOD_WEIGHTS, OutlierConfig
from .decomposition import decompose_series
from .loading import EDAPanel, seasonal_period_for_freq
from .results import OutlierFlag, OutlierReport


# ---------------------------------------------------------------------------
# per-series detectors (each returns a boolean mask aligned to the input)
# ---------------------------------------------------------------------------


def robust_zscores(values: np.ndarray) -> np.ndarray:
    """MAD-based z-scores; falls back to mean absolute deviation when MAD=0."""
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    if mad > 0:
        return 0.6745 * (values - med) / mad
    mean_ad = np.nanmean(np.abs(values - med))
    if mean_ad > 0:
        return (values - med) / (1.2533 * mean_ad)
    return np.zeros_like(values, dtype=float)


def robust_zscore_outliers(series: pd.Series, threshold: float = 3.5) -> np.ndarray:
    return np.abs(robust_zscores(series.to_numpy(dtype=float))) > threshold


def iqr_outliers(series: pd.Series, multiplier: float = 3.0) -> np.ndarray:
    values = series.to_numpy(dtype=float)
    q1, q3 = np.nanpercentile(values, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return np.zeros(len(values), dtype=bool)
    return (values < q1 - multiplier * iqr) | (values > q3 + multiplier * iqr)


def stl_residual_outliers(
    series: pd.Series,
    config: OutlierConfig,
    period: int | None,
) -> tuple[np.ndarray, pd.Series]:
    """Flag points the seasonal+trend structure cannot explain.

    Returns ``(mask, expected)`` where ``expected`` is the fitted
    (trend + seasonal) baseline — the value the point "should" have had.
    """
    decomp = decompose_series(
        series, period, rolling_window=config.rolling_window, variable=str(series.name)
    )
    resid = decomp.resid.to_numpy(dtype=float)
    mask = np.abs(robust_zscores(resid)) > config.stl_residual_threshold
    expected = series.astype(float) - decomp.resid
    return mask, expected


def level_shift_points(
    adjusted: np.ndarray,
    noise_sigma: float,
    config: OutlierConfig,
) -> list[int]:
    """Detect sustained level shifts (changepoints), one index per shift.

    ``adjusted`` should already have seasonal (and, for a KPI, media-driven)
    structure removed. At each point the trailing/leading window medians are
    compared in units of the standard error of that difference
    (``sigma * sqrt(pi / w)`` for two w-point medians); detections must
    persist for ``level_shift_min_run`` consecutive periods.
    """
    values = np.asarray(adjusted, dtype=float)
    n = len(values)
    w = config.level_shift_window
    if n < 2 * w + config.level_shift_min_run:
        return []
    if not np.isfinite(noise_sigma) or noise_sigma <= 0:
        return []
    sigma_diff = noise_sigma * np.sqrt(np.pi / w)

    diffs = np.zeros(n)
    for t in range(w, n - w):
        left = np.nanmedian(values[t - w : t])
        right = np.nanmedian(values[t : t + w])
        diffs[t] = right - left

    over = np.abs(diffs) > config.level_shift_threshold * sigma_diff
    points: list[int] = []
    run_start = None
    for t in range(n):
        if over[t]:
            if run_start is None:
                run_start = t
        elif run_start is not None:
            if t - run_start >= config.level_shift_min_run:
                run = slice(run_start, t)
                points.append(run_start + int(np.argmax(np.abs(diffs[run]))))
            run_start = None
    if run_start is not None and n - run_start >= config.level_shift_min_run:
        run = slice(run_start, n)
        points.append(run_start + int(np.argmax(np.abs(diffs[run]))))
    return points


def spend_spike_outliers(series: pd.Series, config: OutlierConfig) -> np.ndarray:
    """Isolated spikes in a media series (data-entry error pattern).

    Two complementary branches (either fires):

    * **local isolation** — the point exceeds ``spike_ratio`` x its
      neighborhood max. Catches spikes that tower over their local context.
    * **global extremity** — the point exceeds ``normalization_damage_ratio``
      x the series' p99 (computed EXCLUDING the point) while still being at
      least 2x its neighborhood max. Catches realistic decimal-shift errors
      that land next to flight peaks: a x10 shift on a median week is only
      ~4x its peak neighbors (under the local ratio) yet alone sets the
      channel's normalization scale.

    A genuinely heavy flight week passes neither: it is comparable to other
    peaks (not 3x the p99) and is not isolated from its neighbors.

    The first/last 2 periods are never flagged: their one-sided neighborhoods
    make "isolation" meaningless there (a burst starting at week 0 looks
    isolated against its all-trough right-side neighbors).
    """
    values = series.to_numpy(dtype=float)
    n = len(values)
    k = config.spike_neighborhood
    mask = np.zeros(n, dtype=bool)
    if n < 2 * k + 1:
        return mask
    p99_with = float(np.nanpercentile(values, 99))
    for t in range(2, n - 2):
        lo, hi = max(0, t - k), min(n, t + k + 1)
        neighborhood = np.concatenate([values[lo:t], values[t + 1 : hi]])
        local_max = float(np.nanmax(neighborhood)) if neighborhood.size else 0.0
        local_floor = max(local_max, 1e-12)
        locally_isolated = values[t] > config.spike_ratio * local_floor
        globally_extreme = False
        if values[t] > 2.0 * local_floor and values[t] >= p99_with:
            # Only now pay for the leave-one-out p99 (the spike itself drags
            # the with-point p99 upward, masking its own extremity).
            p99_excl = float(np.nanpercentile(np.delete(values, t), 99))
            globally_extreme = values[t] > config.normalization_damage_ratio * max(
                p99_excl, 1e-12
            )
        mask[t] = locally_isolated or globally_extreme
    return mask


def spend_drop_outliers(series: pd.Series, config: OutlierConfig) -> np.ndarray:
    """Isolated drops to ~zero in an ALWAYS-ON media series (missed data load).

    Fires only when the local context is clearly always-on (neighborhood
    minimum well above zero relative to the series median) AND the point falls
    far below that floor. Flighted channels never satisfy the context test —
    their dark weeks have near-zero neighbors, which is normal flighting, not
    an error.

    The first/last 2 periods are never flagged (one-sided context: the start
    of a flight trough at week 0 looks like a "drop" against its all-burst
    right-side neighbors).
    """
    values = series.to_numpy(dtype=float)
    n = len(values)
    k = config.spike_neighborhood
    mask = np.zeros(n, dtype=bool)
    med = float(np.nanmedian(values))
    if n < 2 * k + 1 or not np.isfinite(med) or med <= 0:
        return mask
    for t in range(2, n - 2):
        lo, hi = max(0, t - k), min(n, t + k + 1)
        neighborhood = np.concatenate([values[lo:t], values[t + 1 : hi]])
        if not neighborhood.size:
            continue
        nb_min = float(np.nanmin(neighborhood))
        always_on_context = nb_min > config.drop_context_ratio * med
        isolated_drop = values[t] < nb_min / config.spike_ratio
        mask[t] = always_on_context and isolated_drop
    return mask


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------


def _regression_detrend(y: pd.Series, X: pd.DataFrame) -> pd.Series:
    """Residual of an OLS fit of y on [1, X] — strips the media/control-driven
    part of the KPI before outlier screening, so media-driven peaks aren't
    mistaken for shocks."""
    Xv = X.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(yv)), Xv])
    try:
        beta, *_ = np.linalg.lstsq(design, yv, rcond=None)
    except np.linalg.LinAlgError:
        return y - float(np.nanmean(yv))
    return pd.Series(yv - design @ beta, index=y.index, name=y.name)


class OutlierDetector:
    """Run all configured detectors over a panel's KPI + media (+ controls)."""

    def __init__(self, panel: EDAPanel, config: OutlierConfig | None = None):
        self.panel = panel
        self.config = config or OutlierConfig()

    def default_variables(self) -> list[str]:
        roles = [v for v in [self.panel.kpi, *self.panel.media] if v]
        return roles or list(self.panel.variables)

    def run(self, variables: list[str] | None = None) -> OutlierReport:
        cfg = self.config
        variables = variables or self.default_variables()
        period = cfg.stl_period or seasonal_period_for_freq(self.panel.freq)

        flags: list[OutlierFlag] = []
        per_variable: dict[str, dict] = {}

        for var in variables:
            if var not in self.panel.variables:
                continue
            var_stats = {
                "n_flags": 0,
                "excess_kurtosis": None,
                "max_over_p99": None,
                "normalization_damaged": False,
            }
            is_media = var in self.panel.media
            is_kpi = var == self.panel.kpi

            for dim_values, series in self.panel.slices(var):
                if len(series) < 8 or series.nunique() <= 1:
                    continue
                var_flags, resid_kurt = self._run_series(
                    var, series, dim_values, period, is_media=is_media, is_kpi=is_kpi
                )
                flags.extend(var_flags)
                var_stats["n_flags"] += len(var_flags)

                values = series.to_numpy(dtype=float)
                kurt = resid_kurt
                p99 = float(np.nanpercentile(values, 99))
                max_over_p99 = float(np.nanmax(values) / p99) if p99 > 0 else None
                var_stats["excess_kurtosis"] = kurt
                var_stats["max_over_p99"] = max_over_p99
                if (
                    is_media
                    and max_over_p99 is not None
                    and max_over_p99 > cfg.normalization_damage_ratio
                ):
                    var_stats["normalization_damaged"] = True

            per_variable[var] = var_stats

        flags = self._classify_heavy_tails(flags, per_variable)

        return OutlierReport(
            flags=flags,
            actions=[],  # filled by remediation.recommend_treatments
            config=cfg,
            per_variable=per_variable,
            dataset_path=self.panel.source_path,
        )

    # -- internals ---------------------------------------------------------

    def _run_series(
        self,
        var: str,
        series: pd.Series,
        dim_values: dict[str, str],
        period: int | None,
        *,
        is_media: bool,
        is_kpi: bool,
    ) -> tuple[list[OutlierFlag], float]:
        cfg = self.config
        series = series.astype(float)

        if is_media:
            return self._run_media_series(var, series, dim_values), 0.0

        # For the KPI, screen the part media/controls can't explain so that
        # media-driven peaks don't read as shocks.
        target = series
        if is_kpi:
            predictors = [v for v in (*self.panel.media, *self.panel.controls) if v]
            if predictors:
                X = pd.DataFrame(
                    {
                        p: self.panel.series(p, dim_values or None).reindex(
                            series.index
                        )
                        for p in predictors
                    }
                ).fillna(0.0)
                target = _regression_detrend(series, X)

        decomp = decompose_series(
            target, period, rolling_window=cfg.rolling_window, variable=var
        )
        resid = decomp.resid
        # The fitted baseline mapped back to the observed scale.
        expected = series - resid
        seasonally_adjusted = (
            target - decomp.seasonal if decomp.seasonal is not None else target
        )

        fired: dict[str, np.ndarray] = {}
        if "robust_z" in cfg.methods:
            fired["robust_z"] = robust_zscore_outliers(target, cfg.robust_z_threshold)
        if "iqr" in cfg.methods:
            fired["iqr"] = iqr_outliers(target, cfg.iqr_multiplier)
        if "stl_residual" in cfg.methods:
            fired["stl_residual"] = (
                np.abs(robust_zscores(resid.to_numpy(dtype=float)))
                > cfg.stl_residual_threshold
            )

        resid_values = resid.to_numpy(dtype=float)
        resid_kurt = float(sps.kurtosis(resid_values, fisher=True, nan_policy="omit"))

        n = len(series)
        scores = np.zeros(n)
        for method, mask in fired.items():
            scores = scores + METHOD_WEIGHTS.get(method, 0.0) * mask.astype(float)

        flags: list[OutlierFlag] = []
        for t in np.flatnonzero(scores >= cfg.consensus_threshold):
            methods = [m for m, mask in fired.items() if mask[t]]
            value = float(series.iloc[t])
            exp = float(expected.iloc[t])
            if is_kpi and value > exp:
                kind = "kpi_shock"
            elif value < exp:
                kind = "low_outlier"
            else:
                kind = "point_outlier"
            flags.append(
                OutlierFlag(
                    variable=var,
                    period=str(pd.Timestamp(series.index[t]).date()),
                    value=value,
                    expected=exp,
                    methods=methods,
                    score=float(min(1.0, scores[t])),
                    kind=kind,
                    dims=dict(dim_values),
                )
            )

        if "level_shift" in cfg.methods:
            mad = float(np.nanmedian(np.abs(resid_values - np.nanmedian(resid_values))))
            sigma = 1.4826 * mad if mad > 0 else float(np.nanstd(resid_values))
            for t in level_shift_points(
                seasonally_adjusted.to_numpy(dtype=float), sigma, cfg
            ):
                flags.append(
                    OutlierFlag(
                        variable=var,
                        period=str(pd.Timestamp(series.index[t]).date()),
                        value=float(series.iloc[t]),
                        expected=float(expected.iloc[t]),
                        methods=["level_shift"],
                        score=0.6,
                        kind="level_shift",
                        dims=dict(dim_values),
                    )
                )

        return flags, resid_kurt

    def _run_media_series(
        self,
        var: str,
        series: pd.Series,
        dim_values: dict[str, str],
    ) -> list[OutlierFlag]:
        """Media spend: spike + drop detectors only.

        Pulsed flighting (deep on/off cycles) makes z/IQR/STL screens
        systematically misfire on media series, and "low" media weeks are
        normal dark weeks, not errors. The damaging error patterns — an
        isolated spike that corrupts max-normalization, and a missed-load
        zero in an always-on channel — get dedicated detectors instead.
        """
        cfg = self.config
        # Spike-immune local baseline for the "expected" value.
        expected = series.rolling(
            cfg.rolling_window, center=True, min_periods=1
        ).median()
        flags: list[OutlierFlag] = []

        def _add(t: int, kind: str, method: str) -> None:
            flags.append(
                OutlierFlag(
                    variable=var,
                    period=str(pd.Timestamp(series.index[t]).date()),
                    value=float(series.iloc[t]),
                    expected=float(expected.iloc[t]),
                    methods=[method],
                    score=1.0 if kind == "isolated_spike" else 0.9,
                    kind=kind,
                    dims=dict(dim_values),
                )
            )

        if "spend_spike" in cfg.methods:
            for t in np.flatnonzero(spend_spike_outliers(series, cfg)):
                _add(t, "isolated_spike", "spend_spike")
        if "spend_drop" in cfg.methods:
            for t in np.flatnonzero(spend_drop_outliers(series, cfg)):
                _add(t, "isolated_drop", "spend_drop")
        return flags

    def _classify_heavy_tails(
        self,
        flags: list[OutlierFlag],
        per_variable: dict[str, dict],
    ) -> list[OutlierFlag]:
        """Reclassify point flags on heavy-tailed NON-media series as a
        heavy-tail cluster (they're a noise property, not individual errors)."""
        cfg = self.config
        for var, stats_ in per_variable.items():
            if var in self.panel.media:
                continue
            kurt = stats_.get("excess_kurtosis")
            var_point_flags = [
                f
                for f in flags
                if f.variable == var
                and f.kind in ("kpi_shock", "point_outlier", "low_outlier")
            ]
            if (
                kurt is not None
                and kurt > cfg.heavy_tail_kurtosis
                and len(var_point_flags) >= cfg.heavy_tail_min_flags
            ):
                for f in var_point_flags:
                    f.kind = "heavy_tail_member"
                stats_["heavy_tailed"] = True
            else:
                stats_["heavy_tailed"] = False
        return flags


def detect_outliers(
    panel: EDAPanel,
    config: OutlierConfig | None = None,
    variables: list[str] | None = None,
) -> OutlierReport:
    """Functional wrapper around :class:`OutlierDetector`."""
    return OutlierDetector(panel, config).run(variables)


__all__ = [
    "OutlierDetector",
    "detect_outliers",
    "robust_zscores",
    "robust_zscore_outliers",
    "iqr_outliers",
    "stl_residual_outliers",
    "level_shift_points",
    "spend_spike_outliers",
    "spend_drop_outliers",
]
