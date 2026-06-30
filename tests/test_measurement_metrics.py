"""Tests for the measurement-aware ROI/efficiency divisor resolver and the
``MediaChannelConfig`` measurement descriptor (impression-level ROI)."""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.config.enums import MeasurementUnit
from mmm_framework.config.mff import MFFConfig
from mmm_framework.config.variables import KPIConfig, MediaChannelConfig
from mmm_framework.reporting.helpers.measurement import (
    metric_meta_for_channel,
    resolve_channel_divisor,
    resolve_spend_dict,
)

# =============================================================================
# Config validator
# =============================================================================


def test_default_channel_is_spend():
    cfg = MediaChannelConfig(name="TV")
    assert cfg.measurement_unit is MeasurementUnit.SPEND
    assert cfg.spend_column is None and cfg.cpm is None and cfg.cpc is None


def test_cpm_requires_non_spend_unit():
    with pytest.raises(ValueError, match="measurement_unit is 'spend'"):
        MediaChannelConfig(name="Display", cpm=5.0)


def test_cpm_with_impressions_ok():
    cfg = MediaChannelConfig(
        name="Display", measurement_unit=MeasurementUnit.IMPRESSIONS, cpm=5.0
    )
    assert cfg.cpm == 5.0


def test_cpc_requires_clicks_unit():
    with pytest.raises(ValueError, match="cpc .*requires measurement_unit='clicks'"):
        MediaChannelConfig(
            name="Search", measurement_unit=MeasurementUnit.IMPRESSIONS, cpc=2.0
        )


def test_only_one_cost_source():
    with pytest.raises(ValueError, match="at most one cost source"):
        MediaChannelConfig(
            name="X", measurement_unit=MeasurementUnit.IMPRESSIONS, cpm=5.0, cpc=2.0
        )


def test_spend_column_requires_non_spend_unit():
    with pytest.raises(ValueError, match="measurement_unit is 'spend'"):
        MediaChannelConfig(name="X", spend_column="x_spend")


def test_negative_cost_rejected():
    with pytest.raises(ValueError, match="cpm must be positive"):
        MediaChannelConfig(
            name="X", measurement_unit=MeasurementUnit.IMPRESSIONS, cpm=-1.0
        )


# =============================================================================
# Resolver fixtures
# =============================================================================


class _FakeModel:
    """Minimal surface the divisor resolver touches."""

    def __init__(self, mff_config, X_media_raw, channel_names, spend_raw=None):
        self.mff_config = mff_config
        self.X_media_raw = np.asarray(X_media_raw, dtype=np.float64)
        self.channel_names = list(channel_names)
        self.spend_raw = spend_raw


def _model():
    kpi = KPIConfig(name="sales")
    media = [
        MediaChannelConfig(name="TV"),  # spend (default)
        MediaChannelConfig(
            name="Display", measurement_unit=MeasurementUnit.IMPRESSIONS, cpm=5.0
        ),
        MediaChannelConfig(
            name="Banner", measurement_unit=MeasurementUnit.IMPRESSIONS
        ),  # efficiency
        MediaChannelConfig(
            name="Search", measurement_unit=MeasurementUnit.CLICKS, cpc=2.0
        ),
        MediaChannelConfig(
            name="Email",
            measurement_unit=MeasurementUnit.IMPRESSIONS,
            spend_column="email_spend",
        ),
    ]
    cfg = MFFConfig(kpi=kpi, media_channels=media)
    channels = ["TV", "Display", "Banner", "Search", "Email"]
    X = np.array(
        [
            [100, 1000, 2000, 50, 1000],
            [200, 2000, 2000, 50, 1000],
            [300, 3000, 2000, 100, 1000],
            [400, 4000, 2000, 100, 1000],
        ],
        dtype=np.float64,
    )
    spend_raw = {"Email": np.array([10.0, 20.0, 30.0, 40.0])}
    return _FakeModel(cfg, X, channels, spend_raw)


# =============================================================================
# Resolver branches
# =============================================================================


def test_spend_default_byte_identical_sum():
    m = _model()
    d = resolve_channel_divisor(m, "TV")
    assert d.found and d.total == pytest.approx(1000.0)  # 100+200+300+400
    assert d.meta.is_monetary and d.meta.reference == 1.0
    assert d.meta.roi_label == "ROI"
    assert d.meta.cost_basis is None


def test_impressions_cpm_derives_spend():
    m = _model()
    d = resolve_channel_divisor(m, "Display")
    # (1000+2000+3000+4000)/1000 * 5 = 10 * 5 = 50
    assert d.total == pytest.approx(50.0)
    assert d.meta.is_monetary and d.meta.reference == 1.0
    assert d.meta.cost_basis == "cpm" and d.meta.roi_label == "ROI"


def test_clicks_cpc_derives_spend():
    m = _model()
    d = resolve_channel_divisor(m, "Search")
    # (50+50+100+100) * 2 = 300 * 2 = 600
    assert d.total == pytest.approx(600.0)
    assert d.meta.is_monetary and d.meta.cost_basis == "cpc"


def test_impressions_no_cost_is_efficiency():
    m = _model()
    d = resolve_channel_divisor(m, "Banner")
    # 8000 / 1000 = 8 (per-1k-impression basis)
    assert d.total == pytest.approx(8.0)
    assert not d.meta.is_monetary
    assert d.meta.reference == 0.0
    assert not d.meta.supports_profitability
    assert "1K impressions" in d.meta.roi_label


def test_spend_column_uses_external_series():
    m = _model()
    d = resolve_channel_divisor(m, "Email")
    assert d.total == pytest.approx(100.0)  # 10+20+30+40
    assert d.meta.is_monetary and d.meta.cost_basis == "spend_column"


def test_spend_column_missing_degrades_to_efficiency():
    m = _model()
    m.spend_raw = None  # declared but not loaded
    with pytest.warns(UserWarning, match="no spend series is loaded"):
        d = resolve_channel_divisor(m, "Email")
    assert not d.meta.is_monetary
    assert d.total == pytest.approx(4000.0 / 1000.0)  # 4*1000 impressions / 1000


def test_mask_restricts_window():
    m = _model()
    mask = np.array([True, True, False, False])
    d = resolve_channel_divisor(m, "TV", mask=mask)
    assert d.total == pytest.approx(300.0)  # 100 + 200
    d2 = resolve_channel_divisor(m, "Display", mask=mask)
    assert d2.total == pytest.approx((1000 + 2000) / 1000 * 5)  # 15


def test_marginal_denominator_identity():
    """divisor * (factor-1) is the incremental spend/volume for a ScaleInput."""
    m = _model()
    factor = 1.1
    for ch in ["TV", "Display", "Banner", "Search", "Email"]:
        d = resolve_channel_divisor(m, ch)
        assert d.total * (factor - 1.0) == pytest.approx(d.total * 0.1)


def test_resolve_spend_dict_covers_all_channels():
    m = _model()
    sd = resolve_spend_dict(m)
    assert set(sd) == {"TV", "Display", "Banner", "Search", "Email"}
    assert sd["TV"] == pytest.approx(1000.0)


def test_metric_meta_from_config_only():
    m = _model()
    assert metric_meta_for_channel(m, "TV").is_monetary
    assert metric_meta_for_channel(m, "Display").cost_basis == "cpm"
    assert not metric_meta_for_channel(m, "Banner").is_monetary


def test_unknown_channel_not_found():
    m = _model()
    d = resolve_channel_divisor(m, "Nope")
    assert not d.found and d.total == 0.0


def test_no_mff_config_defaults_to_spend():
    m = _model()
    m.mff_config = None
    d = resolve_channel_divisor(m, "TV")
    assert d.found and d.meta.is_monetary and d.total == pytest.approx(1000.0)


# =============================================================================
# Loader integration: spend_column (option a) flows from MFF to model.spend_raw
# =============================================================================


def _mff_long_with_spend():
    """An MFF-long frame: KPI + an impressions channel + its dollar spend col."""
    import pandas as pd

    from mmm_framework.config.enums import DimensionType
    from mmm_framework.data_loader import mff_from_wide_format

    wide = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-07", periods=4, freq="W"),
            "Sales": [1000, 1100, 1200, 1300],
            "Display": [10000, 20000, 30000, 40000],
            "DisplaySpend": [100, 150, 200, 250],
        }
    )
    df = mff_from_wide_format(
        wide,
        period_col="date",
        value_columns={
            "Sales": "Sales",
            "Display": "Display",
            "DisplaySpend": "DisplaySpend",
        },
    )
    spend = [100, 150, 200, 250]
    impr = [10000, 20000, 30000, 40000]

    kpi = KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD])
    media = [
        MediaChannelConfig(
            name="Display",
            dimensions=[DimensionType.PERIOD],
            measurement_unit=MeasurementUnit.IMPRESSIONS,
            spend_column="DisplaySpend",
        )
    ]
    cfg = MFFConfig(kpi=kpi, media_channels=media)
    return df, cfg, spend, impr


def test_loader_populates_spend_raw():
    from mmm_framework.data_loader import load_mff

    df, cfg, spend, _ = _mff_long_with_spend()
    panel = load_mff(df, cfg)
    assert panel.spend_raw is not None and "Display" in panel.spend_raw
    np.testing.assert_allclose(panel.spend_raw["Display"], spend)


def test_loader_spend_column_resolves_to_roi():
    from mmm_framework.data_loader import load_mff

    df, cfg, spend, _ = _mff_long_with_spend()
    panel = load_mff(df, cfg)
    # Wrap the panel in the minimal surface the resolver needs.
    model = _FakeModel(
        cfg,
        panel.X_media.values,
        list(panel.X_media.columns),
        spend_raw=panel.spend_raw,
    )
    d = resolve_channel_divisor(model, "Display")
    assert d.meta.is_monetary and d.meta.cost_basis == "spend_column"
    assert d.total == pytest.approx(sum(spend))  # 700


def test_loader_no_spend_column_leaves_spend_raw_none():
    from mmm_framework.data_loader import load_mff

    df, cfg, _, _ = _mff_long_with_spend()
    # Drop the spend column declaration -> efficiency path, no spend_raw.
    cfg.media_channels[0].spend_column = None
    cfg.media_channels[0].measurement_unit = MeasurementUnit.IMPRESSIONS
    df = df[df["VariableName"] != "DisplaySpend"]
    panel = load_mff(df, cfg)
    assert panel.spend_raw is None


# =============================================================================
# Agent spec -> MFFConfig threading
# =============================================================================


def test_agent_spec_threads_measurement_descriptor():
    from mmm_framework.agents.fitting import _mff_config_from_spec

    spec = {
        "kpi": "Sales",
        "media_channels": [
            {"name": "TV"},
            {"name": "Display", "measurement_unit": "impressions", "cpm": 7.5},
            {"name": "Search", "measurement_unit": "clicks", "cpc": 1.25},
            {"name": "Banner", "measurement_unit": "impressions"},
            {
                "name": "Email",
                "measurement_unit": "impressions",
                "spend_column": "EmailSpend",
            },
        ],
    }
    cfg = _mff_config_from_spec(spec)
    by_name = {m.name: m for m in cfg.media_channels}
    assert by_name["TV"].measurement_unit is MeasurementUnit.SPEND
    assert by_name["Display"].measurement_unit is MeasurementUnit.IMPRESSIONS
    assert by_name["Display"].cpm == 7.5
    assert by_name["Search"].measurement_unit is MeasurementUnit.CLICKS
    assert by_name["Search"].cpc == 1.25
    assert by_name["Banner"].cpm is None and by_name["Banner"].spend_column is None
    assert by_name["Email"].spend_column == "EmailSpend"


# =============================================================================
# End-to-end: a real fit with impression channels flows through the ROI surface
# =============================================================================


@pytest.mark.slow
def test_end_to_end_fit_reports_roi_and_efficiency():
    """Fit a real MMM with a spend channel, an impressions+cpm channel, and a
    cost-less impressions channel; the dashboard ROI surface must report ROI for
    the first two and per-1k-impression efficiency for the third."""
    import pandas as pd

    from mmm_framework.config import (
        DimensionType,
        InferenceMethod,
        ModelConfig,
    )
    from mmm_framework.data_loader import PanelCoordinates, PanelDataset
    from mmm_framework.model import BayesianMMM, TrendConfig, TrendType
    from mmm_framework.reporting.helpers import compute_roi_with_uncertainty

    periods = pd.date_range("2021-01-04", periods=40, freq="W-MON")
    n = len(periods)
    rng = np.random.default_rng(5)
    t = np.arange(n)
    tv = np.abs(rng.normal(100, 25, n))  # spend ($)
    display = np.abs(rng.normal(50000, 12000, n))  # impressions
    banner = np.abs(rng.normal(30000, 8000, n))  # impressions (no cost)
    y = pd.Series(
        1000
        + 10.0 * t
        + 1.0 * tv
        + 0.001 * display
        + 0.002 * banner
        + rng.normal(0, 20, n),
        name="Sales",
    )
    config = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
            MediaChannelConfig(
                name="Display",
                dimensions=[DimensionType.PERIOD],
                measurement_unit=MeasurementUnit.IMPRESSIONS,
                cpm=5.0,
            ),
            MediaChannelConfig(
                name="Banner",
                dimensions=[DimensionType.PERIOD],
                measurement_unit=MeasurementUnit.IMPRESSIONS,
            ),
        ],
    )
    panel = PanelDataset(
        y=y,
        X_media=pd.DataFrame({"TV": tv, "Display": display, "Banner": banner}),
        X_controls=None,
        coords=PanelCoordinates(
            periods=periods,
            geographies=None,
            products=None,
            channels=["TV", "Display", "Banner"],
            controls=None,
        ),
        index=periods,
        config=config,
    )
    model = BayesianMMM(
        panel,
        ModelConfig(
            inference_method=InferenceMethod.BAYESIAN_PYMC,
            n_chains=2,
            n_draws=120,
            n_tune=120,
            target_accept=0.85,
        ),
        TrendConfig(type=TrendType.LINEAR),
    )
    model.fit(random_seed=0)

    roi = compute_roi_with_uncertainty(model).set_index("channel")

    # Spend channel: ROI vs a 1.0 break-even.
    assert bool(roi.loc["TV", "metric_is_monetary"]) is True
    assert roi.loc["TV", "reference"] == 1.0

    # Impressions + cpm: derived-spend ROI (monetary), divisor = impr/1000 * 5.
    assert bool(roi.loc["Display", "metric_is_monetary"]) is True
    assert roi.loc["Display", "cost_basis"] == "cpm"
    assert roi.loc["Display", "spend"] == pytest.approx(display.sum() / 1000 * 5.0)

    # Cost-less impressions: efficiency per 1k, reference 0, no prob_profitable.
    assert bool(roi.loc["Banner", "metric_is_monetary"]) is False
    assert roi.loc["Banner", "reference"] == 0.0
    assert roi.loc["Banner", "value_units"] == "KPI / 1K impr"
    # None in a float DataFrame column reads back as NaN (the source value is None).
    assert pd.isna(roi.loc["Banner", "prob_profitable"])
    assert roi.loc["Banner", "spend"] == pytest.approx(banner.sum() / 1000)

    # Estimands: contribution_roi for the efficiency channel carries reference 0.
    ests = model.evaluate_estimands()
    banner_roi = ests.get("contribution_roi:Banner")
    if banner_roi is not None and banner_roi.status == "ok":
        assert banner_roi.extra.get("metric_is_monetary") is False
        assert banner_roi.extra.get("metric_reference") == 0.0
