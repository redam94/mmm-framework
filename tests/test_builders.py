"""
Tests for builder classes.

These tests ensure that builder refactoring doesn't introduce breaking changes.
"""

import pytest

from mmm_framework.builders import (
    AdstockConfigBuilder,
    ControlVariableConfigBuilder,
    KPIConfigBuilder,
    MediaChannelConfigBuilder,
    ModelConfigBuilder,
    PriorConfigBuilder,
    SaturationConfigBuilder,
)
from mmm_framework.config import (
    AdstockType,
    DimensionType,
    PriorType,
    SaturationType,
)


class TestMediaChannelConfigBuilder:
    """Tests for MediaChannelConfigBuilder."""

    def test_basic_build(self):
        """Test basic builder creates valid config."""
        config = MediaChannelConfigBuilder("TV").build()

        assert config.name == "TV"
        assert config.dimensions == [DimensionType.PERIOD]
        assert config.adstock is not None
        assert config.saturation is not None
        assert config.coefficient_prior is not None

    def test_with_display_name(self):
        """Test display name setting."""
        config = MediaChannelConfigBuilder("tv_spend").with_display_name("TV Advertising").build()

        assert config.name == "tv_spend"
        assert config.display_name == "TV Advertising"

    def test_with_unit(self):
        """Test unit setting."""
        config = MediaChannelConfigBuilder("TV").with_unit("USD").build()

        assert config.unit == "USD"

    def test_with_dimensions(self):
        """Test dimension setting."""
        config = (
            MediaChannelConfigBuilder("TV")
            .with_dimensions(DimensionType.PERIOD, DimensionType.GEOGRAPHY)
            .build()
        )

        assert DimensionType.PERIOD in config.dimensions
        assert DimensionType.GEOGRAPHY in config.dimensions

    def test_with_dimensions_adds_period_if_missing(self):
        """Test that PERIOD is automatically added if not specified."""
        config = (
            MediaChannelConfigBuilder("TV")
            .with_dimensions(DimensionType.GEOGRAPHY)
            .build()
        )

        assert DimensionType.PERIOD in config.dimensions
        assert DimensionType.GEOGRAPHY in config.dimensions
        # PERIOD should be first
        assert config.dimensions[0] == DimensionType.PERIOD

    def test_national(self):
        """Test national convenience method."""
        config = MediaChannelConfigBuilder("TV").national().build()

        assert config.dimensions == [DimensionType.PERIOD]

    def test_by_geo(self):
        """Test by_geo convenience method."""
        config = MediaChannelConfigBuilder("TV").by_geo().build()

        assert config.dimensions == [DimensionType.PERIOD, DimensionType.GEOGRAPHY]

    def test_by_product(self):
        """Test by_product convenience method."""
        config = MediaChannelConfigBuilder("TV").by_product().build()

        assert config.dimensions == [DimensionType.PERIOD, DimensionType.PRODUCT]

    def test_by_geo_and_product(self):
        """Test by_geo_and_product convenience method."""
        config = MediaChannelConfigBuilder("TV").by_geo_and_product().build()

        assert config.dimensions == [
            DimensionType.PERIOD,
            DimensionType.GEOGRAPHY,
            DimensionType.PRODUCT,
        ]

    def test_with_geometric_adstock(self):
        """Test geometric adstock convenience method."""
        config = MediaChannelConfigBuilder("TV").with_geometric_adstock(l_max=12).build()

        assert config.adstock.type == AdstockType.GEOMETRIC
        assert config.adstock.l_max == 12

    def test_with_hill_saturation(self):
        """Test hill saturation convenience method."""
        config = MediaChannelConfigBuilder("TV").with_hill_saturation().build()

        assert config.saturation.type == SaturationType.HILL

    def test_with_positive_prior(self):
        """Test positive prior convenience method."""
        config = MediaChannelConfigBuilder("TV").with_positive_prior(sigma=3.0).build()

        assert config.coefficient_prior.distribution == PriorType.HALF_NORMAL
        assert config.coefficient_prior.params["sigma"] == 3.0

    def test_with_parent_channel(self):
        """Test parent channel setting."""
        config = MediaChannelConfigBuilder("Meta").with_parent_channel("Social").build()

        assert config.parent_channel == "Social"

    def test_with_split_dimensions(self):
        """Test split dimensions setting."""
        config = (
            MediaChannelConfigBuilder("TV")
            .with_split_dimensions(DimensionType.OUTLET)
            .build()
        )

        assert DimensionType.OUTLET in config.split_dimensions

    def test_fluent_api_chaining(self):
        """Test that all builder methods can be chained."""
        config = (
            MediaChannelConfigBuilder("TV")
            .with_display_name("TV Advertising")
            .with_unit("USD")
            .by_geo()
            .with_geometric_adstock(l_max=10)
            .with_hill_saturation()
            .with_positive_prior(sigma=2.5)
            .build()
        )

        assert config.name == "TV"
        assert config.display_name == "TV Advertising"
        assert config.unit == "USD"
        assert DimensionType.GEOGRAPHY in config.dimensions
        assert config.adstock.l_max == 10
        assert config.coefficient_prior.params["sigma"] == 2.5


class TestControlVariableConfigBuilder:
    """Tests for ControlVariableConfigBuilder."""

    def test_basic_build(self):
        """Test basic builder creates valid config."""
        config = ControlVariableConfigBuilder("Price").build()

        assert config.name == "Price"
        assert config.dimensions == [DimensionType.PERIOD]
        assert config.coefficient_prior is not None

    def test_with_display_name(self):
        """Test display name setting."""
        config = (
            ControlVariableConfigBuilder("price_idx")
            .with_display_name("Price Index")
            .build()
        )

        assert config.display_name == "Price Index"

    def test_with_unit(self):
        """Test unit setting."""
        config = ControlVariableConfigBuilder("Price").with_unit("Index").build()

        assert config.unit == "Index"

    def test_with_dimensions(self):
        """Test dimension setting."""
        config = (
            ControlVariableConfigBuilder("Price")
            .with_dimensions(DimensionType.PERIOD, DimensionType.GEOGRAPHY)
            .build()
        )

        assert DimensionType.PERIOD in config.dimensions
        assert DimensionType.GEOGRAPHY in config.dimensions

    def test_national(self):
        """Test national convenience method."""
        config = ControlVariableConfigBuilder("Price").national().build()

        assert config.dimensions == [DimensionType.PERIOD]

    def test_by_geo(self):
        """Test by_geo convenience method."""
        config = ControlVariableConfigBuilder("Price").by_geo().build()

        assert config.dimensions == [DimensionType.PERIOD, DimensionType.GEOGRAPHY]

    def test_by_product(self):
        """Test by_product convenience method."""
        config = ControlVariableConfigBuilder("Price").by_product().build()

        assert config.dimensions == [DimensionType.PERIOD, DimensionType.PRODUCT]

    def test_allow_negative(self):
        """Test allow_negative method."""
        config = ControlVariableConfigBuilder("Price").allow_negative().build()

        assert config.allow_negative is True

    def test_positive_only(self):
        """Test positive_only method."""
        config = ControlVariableConfigBuilder("Distribution").positive_only().build()

        assert config.allow_negative is False

    def test_with_shrinkage(self):
        """Test shrinkage prior setting."""
        config = ControlVariableConfigBuilder("Price").with_shrinkage().build()

        assert config.use_shrinkage is True

    def test_with_normal_prior(self):
        """Test normal prior convenience method."""
        config = (
            ControlVariableConfigBuilder("Price")
            .with_normal_prior(mu=-0.5, sigma=0.5)
            .build()
        )

        assert config.coefficient_prior.distribution == PriorType.NORMAL
        assert config.coefficient_prior.params["mu"] == -0.5
        assert config.coefficient_prior.params["sigma"] == 0.5

    def test_default_prior_for_negative_allowed(self):
        """Test default prior when negative is allowed."""
        config = ControlVariableConfigBuilder("Price").allow_negative().build()

        assert config.coefficient_prior.distribution == PriorType.NORMAL

    def test_default_prior_for_positive_only(self):
        """Test default prior when positive only."""
        config = ControlVariableConfigBuilder("Distribution").positive_only().build()

        assert config.coefficient_prior.distribution == PriorType.HALF_NORMAL

    def test_fluent_api_chaining(self):
        """Test that all builder methods can be chained."""
        config = (
            ControlVariableConfigBuilder("Price")
            .with_display_name("Price Index")
            .with_unit("Index")
            .by_geo()
            .allow_negative()
            .with_normal_prior(mu=-0.3, sigma=0.5)
            .build()
        )

        assert config.name == "Price"
        assert config.display_name == "Price Index"
        assert config.unit == "Index"
        assert DimensionType.GEOGRAPHY in config.dimensions
        assert config.allow_negative is True


class TestKPIConfigBuilder:
    """Tests for KPIConfigBuilder."""

    def test_basic_build(self):
        """Test basic builder creates valid config."""
        config = KPIConfigBuilder("Sales").build()

        assert config.name == "Sales"
        assert config.dimensions == [DimensionType.PERIOD]

    def test_with_display_name(self):
        """Test display name setting."""
        config = KPIConfigBuilder("sales_units").with_display_name("Unit Sales").build()

        assert config.display_name == "Unit Sales"

    def test_with_unit(self):
        """Test unit setting."""
        config = KPIConfigBuilder("Sales").with_unit("Units").build()

        assert config.unit == "Units"

    def test_with_dimensions(self):
        """Test dimension setting."""
        config = (
            KPIConfigBuilder("Sales")
            .with_dimensions(DimensionType.PERIOD, DimensionType.GEOGRAPHY)
            .build()
        )

        assert DimensionType.PERIOD in config.dimensions
        assert DimensionType.GEOGRAPHY in config.dimensions

    def test_national(self):
        """Test national convenience method."""
        config = KPIConfigBuilder("Sales").national().build()

        assert config.dimensions == [DimensionType.PERIOD]

    def test_by_geo(self):
        """Test by_geo convenience method."""
        config = KPIConfigBuilder("Sales").by_geo().build()

        assert config.dimensions == [DimensionType.PERIOD, DimensionType.GEOGRAPHY]

    def test_by_product(self):
        """Test by_product convenience method."""
        config = KPIConfigBuilder("Sales").by_product().build()

        assert config.dimensions == [DimensionType.PERIOD, DimensionType.PRODUCT]

    def test_by_geo_and_product(self):
        """Test by_geo_and_product convenience method."""
        config = KPIConfigBuilder("Sales").by_geo_and_product().build()

        assert config.dimensions == [
            DimensionType.PERIOD,
            DimensionType.GEOGRAPHY,
            DimensionType.PRODUCT,
        ]

    def test_additive(self):
        """Test additive model specification."""
        config = KPIConfigBuilder("Sales").additive().build()

        assert config.log_transform is False

    def test_multiplicative(self):
        """Test multiplicative model specification."""
        config = KPIConfigBuilder("Sales").multiplicative().build()

        assert config.log_transform is True

    def test_fluent_api_chaining(self):
        """Test that all builder methods can be chained."""
        config = (
            KPIConfigBuilder("Sales")
            .with_display_name("Unit Sales")
            .with_unit("Units")
            .by_geo_and_product()
            .multiplicative()
            .build()
        )

        assert config.name == "Sales"
        assert config.display_name == "Unit Sales"
        assert config.unit == "Units"
        assert DimensionType.GEOGRAPHY in config.dimensions
        assert DimensionType.PRODUCT in config.dimensions
        assert config.log_transform is True


class TestBuilderDimensionConsistency:
    """Test that dimension methods behave consistently across all builders."""

    @pytest.mark.parametrize(
        "builder_class,name",
        [
            (MediaChannelConfigBuilder, "TV"),
            (ControlVariableConfigBuilder, "Price"),
            (KPIConfigBuilder, "Sales"),
        ],
    )
    def test_national_returns_period_only(self, builder_class, name):
        """Test national() returns PERIOD only for all builders."""
        config = builder_class(name).national().build()
        assert config.dimensions == [DimensionType.PERIOD]

    @pytest.mark.parametrize(
        "builder_class,name",
        [
            (MediaChannelConfigBuilder, "TV"),
            (ControlVariableConfigBuilder, "Price"),
            (KPIConfigBuilder, "Sales"),
        ],
    )
    def test_by_geo_returns_period_and_geo(self, builder_class, name):
        """Test by_geo() returns PERIOD + GEOGRAPHY for all builders."""
        config = builder_class(name).by_geo().build()
        assert config.dimensions == [DimensionType.PERIOD, DimensionType.GEOGRAPHY]

    @pytest.mark.parametrize(
        "builder_class,name",
        [
            (MediaChannelConfigBuilder, "TV"),
            (ControlVariableConfigBuilder, "Price"),
            (KPIConfigBuilder, "Sales"),
        ],
    )
    def test_by_product_returns_period_and_product(self, builder_class, name):
        """Test by_product() returns PERIOD + PRODUCT for all builders."""
        config = builder_class(name).by_product().build()
        assert config.dimensions == [DimensionType.PERIOD, DimensionType.PRODUCT]

    @pytest.mark.parametrize(
        "builder_class,name",
        [
            (MediaChannelConfigBuilder, "TV"),
            (KPIConfigBuilder, "Sales"),
        ],
    )
    def test_by_geo_and_product_returns_all_three(self, builder_class, name):
        """Test by_geo_and_product() returns all three dimensions."""
        config = builder_class(name).by_geo_and_product().build()
        assert config.dimensions == [
            DimensionType.PERIOD,
            DimensionType.GEOGRAPHY,
            DimensionType.PRODUCT,
        ]

    @pytest.mark.parametrize(
        "builder_class,name",
        [
            (MediaChannelConfigBuilder, "TV"),
            (ControlVariableConfigBuilder, "Price"),
            (KPIConfigBuilder, "Sales"),
        ],
    )
    def test_with_display_name_works(self, builder_class, name):
        """Test with_display_name() works for all builders."""
        config = builder_class(name).with_display_name("Test Display").build()
        assert config.display_name == "Test Display"

    @pytest.mark.parametrize(
        "builder_class,name",
        [
            (MediaChannelConfigBuilder, "TV"),
            (ControlVariableConfigBuilder, "Price"),
            (KPIConfigBuilder, "Sales"),
        ],
    )
    def test_with_unit_works(self, builder_class, name):
        """Test with_unit() works for all builders."""
        config = builder_class(name).with_unit("TestUnit").build()
        assert config.unit == "TestUnit"


class TestPriorConfigBuilder:
    """Tests for PriorConfigBuilder."""

    def test_half_normal(self):
        """Test HalfNormal distribution."""
        config = PriorConfigBuilder().half_normal(sigma=2.0).build()

        assert config.distribution == PriorType.HALF_NORMAL
        assert config.params["sigma"] == 2.0

    def test_normal(self):
        """Test Normal distribution."""
        config = PriorConfigBuilder().normal(mu=1.0, sigma=0.5).build()

        assert config.distribution == PriorType.NORMAL
        assert config.params["mu"] == 1.0
        assert config.params["sigma"] == 0.5

    def test_gamma(self):
        """Test Gamma distribution."""
        config = PriorConfigBuilder().gamma(alpha=3.0, beta=2.0).build()

        assert config.distribution == PriorType.GAMMA
        assert config.params["alpha"] == 3.0
        assert config.params["beta"] == 2.0

    def test_beta(self):
        """Test Beta distribution."""
        config = PriorConfigBuilder().beta(alpha=2.0, beta=5.0).build()

        assert config.distribution == PriorType.BETA
        assert config.params["alpha"] == 2.0
        assert config.params["beta"] == 5.0

    def test_with_dims(self):
        """Test dimension setting."""
        config = PriorConfigBuilder().half_normal().with_dims("channel").build()

        assert config.dims == "channel"

    def test_with_dims_list(self):
        """Test dimension setting with list."""
        config = PriorConfigBuilder().half_normal().with_dims(["channel", "geo"]).build()

        assert config.dims == ["channel", "geo"]

    def test_build_without_distribution_raises(self):
        """Test that building without distribution raises error."""
        with pytest.raises(ValueError, match="Distribution not set"):
            PriorConfigBuilder().build()


class TestAdstockConfigBuilder:
    """Tests for AdstockConfigBuilder."""

    def test_geometric(self):
        """Test geometric adstock."""
        config = AdstockConfigBuilder().geometric().build()

        assert config.type == AdstockType.GEOMETRIC

    def test_with_max_lag(self):
        """Test max lag setting."""
        config = AdstockConfigBuilder().geometric().with_max_lag(12).build()

        assert config.l_max == 12


class TestSaturationConfigBuilder:
    """Tests for SaturationConfigBuilder."""

    def test_hill(self):
        """Test Hill saturation."""
        config = SaturationConfigBuilder().hill().build()

        assert config.type == SaturationType.HILL

    def test_logistic(self):
        """Test Logistic saturation."""
        config = SaturationConfigBuilder().logistic().build()

        assert config.type == SaturationType.LOGISTIC
