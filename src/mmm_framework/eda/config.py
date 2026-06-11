"""
Configuration for pre-fit data quality: outlier detection, validation, EDA.

Mirrors the frozen-dataclass config pattern of ``mmm_framework.validation``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# Consensus weights per detector. ``level_shift`` is intentionally absent: a
# level shift is a structural changepoint, not a point outlier, and is
# reported as its own kind rather than voted on.
METHOD_WEIGHTS: dict[str, float] = {
    "stl_residual": 0.35,
    "spend_spike": 0.30,
    "spend_drop": 0.30,
    "robust_z": 0.20,
    "iqr": 0.15,
}


@dataclass(frozen=True)
class OutlierConfig:
    """Thresholds for the outlier detectors.

    Defaults are deliberately conservative: in MMM data, seasonal peaks and
    promo flights are real signal, so only points that survive the
    seasonally-adjusted (STL) view or an isolated-spike test should be flagged.
    """

    methods: tuple[str, ...] = (
        "robust_z",
        "iqr",
        "stl_residual",
        "level_shift",
        "spend_spike",
        "spend_drop",
    )
    # Robust z-score (MAD-based): |0.6745 * (x - med) / MAD| > threshold.
    robust_z_threshold: float = 3.5
    # IQR fences: Tukey "far out" multiplier (3.0, not 1.5) so seasonal peaks
    # don't fire the box-plot rule.
    iqr_multiplier: float = 3.0
    # STL residual robust z threshold; STL runs when >= 2 full seasonal cycles
    # exist, otherwise a centered rolling-median detrend is used.
    stl_residual_threshold: float = 3.5
    stl_period: int | None = None  # None -> inferred from frequency (52 weekly)
    rolling_window: int = 13
    # Level shift: |median(next w) - median(prev w)| compared in units of the
    # standard error of that difference (sigma * sqrt(pi/w)), sustained for at
    # least min_run periods. Runs on seasonally-adjusted non-media series only.
    level_shift_window: int = 8
    level_shift_threshold: float = 3.25
    level_shift_min_run: int = 4
    # Spend spike (media only): x_t / max(neighborhood excl. t) > spike_ratio.
    spike_ratio: float = 5.0
    spike_neighborhood: int = 4
    # max/p99 above this means a single point sets the channel's
    # saturation/normalization scale.
    normalization_damage_ratio: float = 3.0
    # Drop (missed-load) detection: the neighborhood minimum must exceed
    # drop_context_ratio * median for the channel to count as always-on there
    # (flighted channels' dark weeks never satisfy this).
    drop_context_ratio: float = 0.25
    # A point is flagged when its weighted vote reaches this score.
    consensus_threshold: float = 0.3
    # Heavy-tail classification: residual excess kurtosis above this AND at
    # least min_flags flags -> flags become a heavy_tail_member cluster.
    # min_flags=5 because even a clean (Gaussian) world yields ~3 structural
    # misses from the linear screen of a saturated response.
    heavy_tail_kurtosis: float = 2.0
    heavy_tail_min_flags: int = 5
    # How many of the largest heavy-tail shocks get dummy recommendations.
    heavy_tail_top_k: int = 5

    @classmethod
    def for_sensitivity(cls, sensitivity: str = "default") -> "OutlierConfig":
        """Preset configs: 'low' flags less, 'high' flags more."""
        base = cls()
        if sensitivity == "low":
            return replace(
                base,
                robust_z_threshold=4.5,
                iqr_multiplier=4.0,
                stl_residual_threshold=4.5,
                spike_ratio=8.0,
            )
        if sensitivity == "high":
            return replace(
                base,
                robust_z_threshold=3.0,
                iqr_multiplier=2.0,
                stl_residual_threshold=3.0,
                spike_ratio=4.0,
            )
        return base


@dataclass(frozen=True)
class DataValidationConfig:
    """Thresholds for pre-fit dataset validation."""

    missing_warn_pct: float = 5.0
    missing_error_pct: float = 30.0
    zero_inflation_warn_pct: float = 60.0
    near_constant_cv: float = 1e-6
    # max/median across variable scales beyond this suggests a unit mismatch.
    scale_ratio_threshold: float = 1e4
    # Warn when n_periods / approx_n_params falls below this.
    min_obs_per_param: float = 4.0


@dataclass(frozen=True)
class EDAConfig:
    """Knobs for the EDA analyses."""

    correlation_threshold: float = 0.8
    vif_threshold: float = 10.0
    top_correlations: int = 10
    # statsmodels robust STL degenerates to near-interpolation on few cycles;
    # see eda.decomposition.decompose_series.
    stl_robust: bool = False
    significance_level: float = 0.05
    max_series_per_chart: int = 12


__all__ = ["OutlierConfig", "DataValidationConfig", "EDAConfig", "METHOD_WEIGHTS"]
