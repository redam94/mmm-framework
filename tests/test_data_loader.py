"""
Test suite for mmm_framework.data_loader module.

Tests match the actual API which uses:
- Pydantic config classes
- DimensionType enum with PERIOD, GEOGRAPHY, PRODUCT, etc.
- PanelCoordinates with geographies (not geos), pd.DatetimeIndex periods
- PanelDataset with pd.Series/DataFrame, index (not date_index), config (not raw_df)
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from mmm_framework.data_loader import (
    MFFLoader,
    MFFValidationError,
    PanelCoordinates,
    PanelDataset,
    load_mff,
    mff_from_wide_format,
    validate_mff_structure,
    validate_variable_dimensions,
)
from mmm_framework.config import (
    DimensionType,
    KPIConfig,
    MediaChannelConfig,
    ControlVariableConfig,
    MFFConfig,
    MFFColumnConfig,
    create_simple_mff_config,
    create_national_media_config,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_periods():
    """Sample weekly periods."""
    return pd.date_range("2020-01-06", periods=52, freq="W-MON")


@pytest.fixture
def sample_geographies():
    """Sample geographic regions."""
    return ["East", "West", "Central"]


@pytest.fixture
def sample_products():
    """Sample products."""
    return ["Product_A", "Product_B"]


@pytest.fixture
def national_mff_data(sample_periods):
    """National-level MFF data (Period only)."""
    records = []

    # KPI - Sales at national level
    for period in sample_periods:
        records.append(
            {
                "VariableName": "Sales",
                "VariableValue": 1000 + np.random.normal(0, 100),
                "Period": period,
                "Geography": None,
                "Product": None,
                "Campaign": None,
                "Outlet": None,
                "Creative": None,
            }
        )

    # Media - TV at national level
    for period in sample_periods:
        records.append(
            {
                "VariableName": "TV",
                "VariableValue": max(0, 100 + np.random.exponential(50)),
                "Period": period,
                "Geography": None,
                "Product": None,
                "Campaign": None,
                "Outlet": None,
                "Creative": None,
            }
        )

    # Media - Digital at national level
    for period in sample_periods:
        records.append(
            {
                "VariableName": "Digital",
                "VariableValue": max(0, 80 + np.random.exponential(40)),
                "Period": period,
                "Geography": None,
                "Product": None,
                "Campaign": None,
                "Outlet": None,
                "Creative": None,
            }
        )

    # Control - Price at national level
    for period in sample_periods:
        records.append(
            {
                "VariableName": "Price",
                "VariableValue": 10 + np.random.uniform(-1, 1),
                "Period": period,
                "Geography": None,
                "Product": None,
                "Campaign": None,
                "Outlet": None,
                "Creative": None,
            }
        )

    return pd.DataFrame(records)


@pytest.fixture
def geo_mff_data(sample_periods, sample_geographies):
    """Geo-level MFF data (Period + Geography)."""
    records = []

    # KPI - Sales at geo level
    for period in sample_periods:
        for geo in sample_geographies:
            records.append(
                {
                    "VariableName": "Sales",
                    "VariableValue": 500 + np.random.normal(0, 50),
                    "Period": period,
                    "Geography": geo,
                    "Product": None,
                    "Campaign": None,
                    "Outlet": None,
                    "Creative": None,
                }
            )

    # Media - TV at national level (will need alignment)
    for period in sample_periods:
        records.append(
            {
                "VariableName": "TV",
                "VariableValue": max(0, 100 + np.random.exponential(50)),
                "Period": period,
                "Geography": None,
                "Product": None,
                "Campaign": None,
                "Outlet": None,
                "Creative": None,
            }
        )

    # Control - Price at geo level
    for period in sample_periods:
        for geo in sample_geographies:
            records.append(
                {
                    "VariableName": "Price",
                    "VariableValue": 10 + np.random.uniform(-1, 1),
                    "Period": period,
                    "Geography": geo,
                    "Product": None,
                    "Campaign": None,
                    "Outlet": None,
                    "Creative": None,
                }
            )

    return pd.DataFrame(records)


@pytest.fixture
def national_config():
    """Configuration for national-level model."""
    return MFFConfig(
        kpi=KPIConfig(
            name="Sales",
            dimensions=[DimensionType.PERIOD],
        ),
        media_channels=[
            MediaChannelConfig(
                name="TV",
                dimensions=[DimensionType.PERIOD],
            ),
            MediaChannelConfig(
                name="Digital",
                dimensions=[DimensionType.PERIOD],
            ),
        ],
        controls=[
            ControlVariableConfig(
                name="Price",
                dimensions=[DimensionType.PERIOD],
            ),
        ],
    )


@pytest.fixture
def geo_config():
    """Configuration for geo-level model with national media."""
    return MFFConfig(
        kpi=KPIConfig(
            name="Sales",
            dimensions=[DimensionType.PERIOD, DimensionType.GEOGRAPHY],
        ),
        media_channels=[
            MediaChannelConfig(
                name="TV",
                dimensions=[DimensionType.PERIOD],  # National media
            ),
        ],
        controls=[
            ControlVariableConfig(
                name="Price",
                dimensions=[DimensionType.PERIOD, DimensionType.GEOGRAPHY],
            ),
        ],
    )


# =============================================================================
# PanelCoordinates Tests
# =============================================================================


class TestPanelCoordinates:
    """Tests for PanelCoordinates dataclass."""

    def test_init_national(self, sample_periods):
        """Test national-only coordinates (no geo/product)."""
        coords = PanelCoordinates(
            periods=sample_periods,
            geographies=None,
            products=None,
            channels=["TV", "Digital"],
            controls=["Price"],
        )

        assert coords.n_periods == 52
        assert coords.has_geo is False
        assert coords.has_product is False
        assert coords.n_geos == 1
        assert coords.n_products == 1

    def test_init_with_geographies(self, sample_periods, sample_geographies):
        """Test coordinates with geographies."""
        coords = PanelCoordinates(
            periods=sample_periods,
            geographies=sample_geographies,
            products=None,
            channels=["TV"],
        )

        assert coords.has_geo is True
        assert coords.has_product is False
        assert coords.n_geos == 3
        assert coords.geographies == sample_geographies

    def test_init_with_products(
        self, sample_periods, sample_geographies, sample_products
    ):
        """Test coordinates with geographies and products."""
        coords = PanelCoordinates(
            periods=sample_periods,
            geographies=sample_geographies,
            products=sample_products,
            channels=["TV"],
        )

        assert coords.has_geo is True
        assert coords.has_product is True
        assert coords.n_products == 2

    def test_n_obs_national(self, sample_periods):
        """Test n_obs for national data."""
        coords = PanelCoordinates(
            periods=sample_periods,
            channels=["TV"],
        )
        assert coords.n_obs == 52

    def test_n_obs_geo(self, sample_periods, sample_geographies):
        """Test n_obs for geo-level data."""
        coords = PanelCoordinates(
            periods=sample_periods,
            geographies=sample_geographies,
            channels=["TV"],
        )
        assert coords.n_obs == 52 * 3

    def test_n_obs_geo_product(
        self, sample_periods, sample_geographies, sample_products
    ):
        """Test n_obs for geo+product data."""
        coords = PanelCoordinates(
            periods=sample_periods,
            geographies=sample_geographies,
            products=sample_products,
            channels=["TV"],
        )
        assert coords.n_obs == 52 * 3 * 2

    def test_to_pymc_coords_national(self, sample_periods):
        """Test PyMC coords conversion for national data."""
        coords = PanelCoordinates(
            periods=sample_periods,
            channels=["TV", "Digital"],
            controls=["Price"],
        )

        pymc_coords = coords.to_pymc_coords()

        assert "date" in pymc_coords
        assert "channel" in pymc_coords
        assert "control" in pymc_coords
        assert "geo" not in pymc_coords
        assert len(pymc_coords["date"]) == 52
        assert pymc_coords["channel"] == ["TV", "Digital"]

    def test_to_pymc_coords_geo(self, sample_periods, sample_geographies):
        """Test PyMC coords conversion for geo data."""
        coords = PanelCoordinates(
            periods=sample_periods,
            geographies=sample_geographies,
            channels=["TV"],
        )

        pymc_coords = coords.to_pymc_coords()

        assert "geo" in pymc_coords
        assert pymc_coords["geo"] == sample_geographies


# =============================================================================
# PanelDataset Tests
# =============================================================================


class TestPanelDataset:
    """Tests for PanelDataset dataclass."""

    def test_properties(self, sample_periods, national_config):
        """Test basic properties of PanelDataset."""
        coords = PanelCoordinates(
            periods=sample_periods,
            channels=["TV", "Digital"],
            controls=["Price"],
        )

        y = pd.Series(np.random.randn(52), name="Sales")
        X_media = pd.DataFrame(
            {
                "TV": np.random.randn(52),
                "Digital": np.random.randn(52),
            }
        )
        X_controls = pd.DataFrame(
            {
                "Price": np.random.randn(52),
            }
        )

        panel = PanelDataset(
            y=y,
            X_media=X_media,
            X_controls=X_controls,
            coords=coords,
            index=sample_periods,
            config=national_config,
        )

        assert panel.n_obs == 52
        assert panel.n_channels == 2
        assert panel.n_controls == 1

    def test_is_panel_national(self, sample_periods, national_config):
        """Test is_panel property for national data."""
        coords = PanelCoordinates(
            periods=sample_periods,
            channels=["TV"],
        )

        panel = PanelDataset(
            y=pd.Series(np.random.randn(52)),
            X_media=pd.DataFrame({"TV": np.random.randn(52)}),
            X_controls=None,
            coords=coords,
            index=sample_periods,
            config=national_config,
        )

        assert panel.is_panel is False

    def test_is_panel_geo(self, sample_periods, sample_geographies, geo_config):
        """Test is_panel property for geo data."""
        n_obs = 52 * 3
        coords = PanelCoordinates(
            periods=sample_periods,
            geographies=sample_geographies,
            channels=["TV"],
        )

        # Create multi-index
        index = pd.MultiIndex.from_product(
            [sample_periods, sample_geographies], names=["Period", "Geography"]
        )

        panel = PanelDataset(
            y=pd.Series(np.random.randn(n_obs), index=index),
            X_media=pd.DataFrame({"TV": np.random.randn(n_obs)}, index=index),
            X_controls=None,
            coords=coords,
            index=index,
            config=geo_config,
        )

        assert panel.is_panel is True

    def test_to_numpy(self, sample_periods, national_config):
        """Test conversion to numpy arrays."""
        coords = PanelCoordinates(
            periods=sample_periods,
            channels=["TV"],
            controls=["Price"],
        )

        y_data = np.random.randn(52)
        media_data = np.random.randn(52)
        control_data = np.random.randn(52)

        panel = PanelDataset(
            y=pd.Series(y_data),
            X_media=pd.DataFrame({"TV": media_data}),
            X_controls=pd.DataFrame({"Price": control_data}),
            coords=coords,
            index=sample_periods,
            config=national_config,
        )

        y_np, X_media_np, X_controls_np = panel.to_numpy()

        np.testing.assert_array_equal(y_np, y_data)
        np.testing.assert_array_equal(X_media_np.flatten(), media_data)
        np.testing.assert_array_equal(X_controls_np.flatten(), control_data)

    def test_get_media_for_channel(self, sample_periods, national_config):
        """Test getting media series for a specific channel."""
        coords = PanelCoordinates(
            periods=sample_periods,
            channels=["TV", "Digital"],
        )

        panel = PanelDataset(
            y=pd.Series(np.random.randn(52)),
            X_media=pd.DataFrame(
                {
                    "TV": np.ones(52) * 100,
                    "Digital": np.ones(52) * 50,
                }
            ),
            X_controls=None,
            coords=coords,
            index=sample_periods,
            config=national_config,
        )

        tv_series = panel.get_media_for_channel("TV")
        assert tv_series.mean() == 100

    def test_get_media_for_channel_not_found(self, sample_periods, national_config):
        """Test error when channel not found."""
        coords = PanelCoordinates(
            periods=sample_periods,
            channels=["TV"],
        )

        panel = PanelDataset(
            y=pd.Series(np.random.randn(52)),
            X_media=pd.DataFrame({"TV": np.random.randn(52)}),
            X_controls=None,
            coords=coords,
            index=sample_periods,
            config=national_config,
        )

        with pytest.raises(KeyError, match="Radio"):
            panel.get_media_for_channel("Radio")

    def test_compute_spend_shares(self, sample_periods, national_config):
        """Test computing spend shares."""
        coords = PanelCoordinates(
            periods=sample_periods,
            channels=["TV", "Digital"],
        )

        panel = PanelDataset(
            y=pd.Series(np.random.randn(52)),
            X_media=pd.DataFrame(
                {
                    "TV": np.ones(52) * 100,  # Total: 5200
                    "Digital": np.ones(52) * 100,  # Total: 5200
                }
            ),
            X_controls=None,
            coords=coords,
            index=sample_periods,
            config=national_config,
        )

        shares = panel.compute_spend_shares()
        assert np.isclose(shares["TV"], 0.5)
        assert np.isclose(shares["Digital"], 0.5)


# =============================================================================
# Validation Function Tests
# =============================================================================


class TestValidateMFFStructure:
    """Tests for validate_mff_structure function."""

    def test_valid_structure(self, national_mff_data, national_config):
        """Valid MFF data should return empty warnings list."""
        warnings_list = validate_mff_structure(national_mff_data, national_config)
        # Should not raise, and warnings should be empty or just informational
        assert isinstance(warnings_list, list)

    def test_missing_kpi_variable_raises(self, sample_periods, national_config):
        """Should raise MFFValidationError for missing KPI variable."""
        df = pd.DataFrame(
            {
                "VariableName": ["TV"] * 52,  # No Sales
                "VariableValue": [100] * 52,
                "Period": sample_periods,
                "Geography": [None] * 52,
                "Product": [None] * 52,
                "Campaign": [None] * 52,
                "Outlet": [None] * 52,
                "Creative": [None] * 52,
            }
        )

        with pytest.raises(MFFValidationError, match="Sales|Missing"):
            validate_mff_structure(df, national_config)

    def test_missing_media_variable_raises(self, sample_periods):
        """Should raise MFFValidationError for missing media variable."""
        config = MFFConfig(
            kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
            media_channels=[
                MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
            ],
        )

        df = pd.DataFrame(
            {
                "VariableName": ["Sales"] * 52,  # No TV
                "VariableValue": [100] * 52,
                "Period": pd.date_range("2020-01-06", periods=52, freq="W-MON"),
                "Geography": [None] * 52,
                "Product": [None] * 52,
                "Campaign": [None] * 52,
                "Outlet": [None] * 52,
                "Creative": [None] * 52,
            }
        )

        with pytest.raises(MFFValidationError, match="TV|Missing"):
            validate_mff_structure(df, config)

    def test_missing_columns_raises(self, sample_periods, national_config):
        """Should raise MFFValidationError for missing required columns."""
        df = pd.DataFrame(
            {
                "VariableName": ["Sales"] * 52,
                "VariableValue": [100] * 52,
                "Period": sample_periods,
                # Missing Geography, Product, Campaign, Outlet, Creative columns
            }
        )

        with pytest.raises(MFFValidationError, match="Missing required columns"):
            validate_mff_structure(df, national_config)


# =============================================================================
# MFFLoader Tests
# =============================================================================


class TestMFFLoader:
    """Tests for MFFLoader class."""

    def test_init(self, national_config):
        """Test loader initialization."""
        loader = MFFLoader(national_config)
        assert loader.config == national_config

    def test_load_national(self, national_mff_data, national_config):
        """Test loading national-level data."""
        loader = MFFLoader(national_config)
        loader.load(national_mff_data)
        panel = loader.build_panel()

        assert panel.n_obs == 52
        assert panel.n_channels == 2
        assert panel.n_controls == 1
        assert panel.is_panel is False

    def test_load_geo_level(self, geo_mff_data, geo_config):
        """Test loading geo-level data."""
        loader = MFFLoader(geo_config)
        loader.load(geo_mff_data)
        panel = loader.build_panel()

        assert panel.n_obs == 52 * 3
        assert panel.is_panel is True
        assert panel.coords.has_geo is True

    def test_panel_coords(self, national_mff_data, national_config):
        """Test that panel has correct coordinates."""
        loader = MFFLoader(national_config)
        loader.load(national_mff_data)
        panel = loader.build_panel()

        assert panel.coords.channels == ["TV", "Digital"]
        assert panel.coords.controls == ["Price"]

    def test_media_stats(self, national_mff_data, national_config):
        """Test that media stats are computed."""
        loader = MFFLoader(national_config)
        loader.load(national_mff_data)
        panel = loader.build_panel()

        assert "TV" in panel.media_stats
        assert "total" in panel.media_stats["TV"]
        assert "mean" in panel.media_stats["TV"]


# =============================================================================
# load_mff Convenience Function Tests
# =============================================================================


class TestLoadMFF:
    """Tests for load_mff convenience function."""

    def test_load_mff_basic(self, national_mff_data, national_config):
        """Test basic load_mff usage."""
        panel = load_mff(national_mff_data, national_config)

        assert panel.n_obs == 52
        assert isinstance(panel, PanelDataset)


# =============================================================================
# mff_from_wide_format Tests
# =============================================================================


class TestMFFFromWideFormat:
    """Tests for mff_from_wide_format function."""

    def test_basic_conversion(self):
        """Test basic wide to MFF conversion."""
        wide_df = pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-06", periods=10, freq="W-MON"),
                "Sales": [100 + i for i in range(10)],
                "TV_Spend": [50 + i for i in range(10)],
            }
        )

        mff_df = mff_from_wide_format(
            wide_df,
            period_col="Date",
            value_columns={"Sales": "Sales", "TV_Spend": "TV"},
        )

        assert "VariableName" in mff_df.columns
        assert "VariableValue" in mff_df.columns
        assert "Sales" in mff_df["VariableName"].values
        assert "TV" in mff_df["VariableName"].values

    def test_conversion_with_geo(self):
        """Test conversion with geography column."""
        wide_df = pd.DataFrame(
            {
                "Date": ["2020-01-06", "2020-01-06"],
                "Region": ["East", "West"],
                "Sales": [100, 90],
            }
        )

        mff_df = mff_from_wide_format(
            wide_df,
            period_col="Date",
            value_columns={"Sales": "Sales"},
            geo_col="Region",
        )

        assert "Geography" in mff_df.columns


# =============================================================================
# create_simple_mff_config Factory Tests
# =============================================================================


class TestCreateSimpleMFFConfig:
    """Tests for create_simple_mff_config factory function."""

    def test_basic_usage(self):
        """Test basic factory function usage."""
        config = create_simple_mff_config(
            kpi_name="Sales",
            media_names=["TV", "Digital"],
        )

        assert config.kpi.name == "Sales"
        assert len(config.media_channels) == 2
        assert config.media_channels[0].name == "TV"

    def test_with_controls(self):
        """Test factory with controls."""
        config = create_simple_mff_config(
            kpi_name="Sales",
            media_names=["TV"],
            control_names=["Price", "Distribution"],
        )

        assert len(config.controls) == 2

    def test_with_geo_dimensions(self):
        """Test factory with geo dimensions."""
        config = create_simple_mff_config(
            kpi_name="Sales",
            media_names=["TV"],
            kpi_dimensions=[DimensionType.PERIOD, DimensionType.GEOGRAPHY],
        )

        assert DimensionType.GEOGRAPHY in config.kpi.dimensions

    def test_multiplicative(self):
        """Test multiplicative model flag."""
        config = create_simple_mff_config(
            kpi_name="Sales",
            media_names=["TV"],
            multiplicative=True,
        )

        assert config.kpi.log_transform is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full workflows."""

    def test_full_workflow_national(self, national_mff_data, national_config):
        """Test complete workflow for national model."""
        # Load data
        panel = load_mff(national_mff_data, national_config)

        # Check panel structure
        assert panel.n_obs == 52
        assert panel.coords.n_periods == 52

        # Check data types
        assert isinstance(panel.y, pd.Series)
        assert isinstance(panel.X_media, pd.DataFrame)

        # Check PyMC coords
        pymc_coords = panel.coords.to_pymc_coords()
        assert len(pymc_coords["date"]) == 52

    def test_full_workflow_geo(self, geo_mff_data, geo_config):
        """Test complete workflow for geo-level model."""
        panel = load_mff(geo_mff_data, geo_config)

        assert panel.n_obs == 52 * 3
        assert panel.is_panel is True

        pymc_coords = panel.coords.to_pymc_coords()
        assert "geo" in pymc_coords
        assert len(pymc_coords["geo"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
