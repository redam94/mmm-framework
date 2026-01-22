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

    def test_michaelis_menten(self):
        """Test Michaelis-Menten saturation."""
        config = SaturationConfigBuilder().michaelis_menten().build()

        assert config.type == SaturationType.MICHAELIS_MENTEN

    def test_tanh(self):
        """Test tanh saturation."""
        config = SaturationConfigBuilder().tanh().build()

        assert config.type == SaturationType.TANH

    def test_none(self):
        """Test disabling saturation."""
        config = SaturationConfigBuilder().none().build()

        assert config.type == SaturationType.NONE

    def test_with_kappa_prior(self):
        """Test setting kappa prior."""
        prior = PriorConfigBuilder().beta(2.0, 5.0).build()
        config = SaturationConfigBuilder().hill().with_kappa_prior(prior).build()

        assert config.kappa_prior is not None
        assert config.kappa_prior.distribution == PriorType.BETA

    def test_with_slope_prior(self):
        """Test setting slope prior."""
        prior = PriorConfigBuilder().half_normal(1.5).build()
        config = SaturationConfigBuilder().hill().with_slope_prior(prior).build()

        assert config.slope_prior is not None
        assert config.slope_prior.params["sigma"] == 1.5

    def test_with_beta_prior(self):
        """Test setting beta prior."""
        prior = PriorConfigBuilder().half_normal(2.0).build()
        config = SaturationConfigBuilder().hill().with_beta_prior(prior).build()

        assert config.beta_prior is not None

    def test_with_kappa_bounds(self):
        """Test setting kappa bounds."""
        config = SaturationConfigBuilder().hill().with_kappa_bounds(0.2, 0.8).build()

        assert config.kappa_bounds_percentiles == (0.2, 0.8)

    def test_with_kappa_bounds_invalid_raises(self):
        """Test that invalid kappa bounds raise error."""
        with pytest.raises(ValueError, match="0 <= lower < upper <= 1"):
            SaturationConfigBuilder().hill().with_kappa_bounds(0.8, 0.2).build()


class TestDimensionAlignmentConfigBuilder:
    """Tests for DimensionAlignmentConfigBuilder."""

    def test_basic_build(self):
        """Test basic builder creates valid config."""
        from mmm_framework.builders import DimensionAlignmentConfigBuilder
        from mmm_framework.config import AllocationMethod

        config = DimensionAlignmentConfigBuilder().build()

        assert config.geo_allocation == AllocationMethod.POPULATION
        assert config.product_allocation == AllocationMethod.SALES

    def test_geo_equal(self):
        """Test equal geo allocation."""
        from mmm_framework.builders import DimensionAlignmentConfigBuilder
        from mmm_framework.config import AllocationMethod

        config = DimensionAlignmentConfigBuilder().geo_equal().build()

        assert config.geo_allocation == AllocationMethod.EQUAL

    def test_geo_by_population(self):
        """Test population geo allocation."""
        from mmm_framework.builders import DimensionAlignmentConfigBuilder
        from mmm_framework.config import AllocationMethod

        config = DimensionAlignmentConfigBuilder().geo_by_population().build()

        assert config.geo_allocation == AllocationMethod.POPULATION

    def test_geo_by_sales(self):
        """Test sales geo allocation."""
        from mmm_framework.builders import DimensionAlignmentConfigBuilder
        from mmm_framework.config import AllocationMethod

        config = DimensionAlignmentConfigBuilder().geo_by_sales().build()

        assert config.geo_allocation == AllocationMethod.SALES

    def test_geo_by_custom(self):
        """Test custom geo allocation."""
        from mmm_framework.builders import DimensionAlignmentConfigBuilder
        from mmm_framework.config import AllocationMethod

        config = DimensionAlignmentConfigBuilder().geo_by_custom("RegionWeight").build()

        assert config.geo_allocation == AllocationMethod.CUSTOM
        assert config.geo_weight_variable == "RegionWeight"

    def test_product_equal(self):
        """Test equal product allocation."""
        from mmm_framework.builders import DimensionAlignmentConfigBuilder
        from mmm_framework.config import AllocationMethod

        config = DimensionAlignmentConfigBuilder().product_equal().build()

        assert config.product_allocation == AllocationMethod.EQUAL

    def test_product_by_sales(self):
        """Test sales product allocation."""
        from mmm_framework.builders import DimensionAlignmentConfigBuilder
        from mmm_framework.config import AllocationMethod

        config = DimensionAlignmentConfigBuilder().product_by_sales().build()

        assert config.product_allocation == AllocationMethod.SALES

    def test_product_by_custom(self):
        """Test custom product allocation."""
        from mmm_framework.builders import DimensionAlignmentConfigBuilder
        from mmm_framework.config import AllocationMethod

        config = DimensionAlignmentConfigBuilder().product_by_custom("ProductWeight").build()

        assert config.product_allocation == AllocationMethod.CUSTOM
        assert config.product_weight_variable == "ProductWeight"

    def test_prefer_disaggregation(self):
        """Test prefer disaggregation setting."""
        from mmm_framework.builders import DimensionAlignmentConfigBuilder

        config = DimensionAlignmentConfigBuilder().prefer_disaggregation().build()

        assert config.prefer_disaggregation is True

    def test_prefer_aggregation(self):
        """Test prefer aggregation setting."""
        from mmm_framework.builders import DimensionAlignmentConfigBuilder

        config = DimensionAlignmentConfigBuilder().prefer_aggregation().build()

        assert config.prefer_disaggregation is False


class TestMFFColumnConfigBuilder:
    """Tests for MFFColumnConfigBuilder."""

    def test_basic_build(self):
        """Test basic builder creates valid config."""
        from mmm_framework.builders import MFFColumnConfigBuilder

        config = MFFColumnConfigBuilder().build()

        assert config.period == "Period"
        assert config.geography == "Geography"
        assert config.product == "Product"

    def test_with_period_column(self):
        """Test setting period column."""
        from mmm_framework.builders import MFFColumnConfigBuilder

        config = MFFColumnConfigBuilder().with_period_column("Week").build()

        assert config.period == "Week"

    def test_with_geography_column(self):
        """Test setting geography column."""
        from mmm_framework.builders import MFFColumnConfigBuilder

        config = MFFColumnConfigBuilder().with_geography_column("Region").build()

        assert config.geography == "Region"

    def test_with_product_column(self):
        """Test setting product column."""
        from mmm_framework.builders import MFFColumnConfigBuilder

        config = MFFColumnConfigBuilder().with_product_column("SKU").build()

        assert config.product == "SKU"

    def test_with_campaign_column(self):
        """Test setting campaign column."""
        from mmm_framework.builders import MFFColumnConfigBuilder

        config = MFFColumnConfigBuilder().with_campaign_column("CampaignName").build()

        assert config.campaign == "CampaignName"

    def test_with_outlet_column(self):
        """Test setting outlet column."""
        from mmm_framework.builders import MFFColumnConfigBuilder

        config = MFFColumnConfigBuilder().with_outlet_column("Platform").build()

        assert config.outlet == "Platform"

    def test_with_creative_column(self):
        """Test setting creative column."""
        from mmm_framework.builders import MFFColumnConfigBuilder

        config = MFFColumnConfigBuilder().with_creative_column("AdCreative").build()

        assert config.creative == "AdCreative"

    def test_with_variable_name_column(self):
        """Test setting variable name column."""
        from mmm_framework.builders import MFFColumnConfigBuilder

        config = MFFColumnConfigBuilder().with_variable_name_column("Metric").build()

        assert config.variable_name == "Metric"

    def test_with_variable_value_column(self):
        """Test setting variable value column."""
        from mmm_framework.builders import MFFColumnConfigBuilder

        config = MFFColumnConfigBuilder().with_variable_value_column("Value").build()

        assert config.variable_value == "Value"

    def test_fluent_api_chaining(self):
        """Test all methods can be chained."""
        from mmm_framework.builders import MFFColumnConfigBuilder

        config = (
            MFFColumnConfigBuilder()
            .with_period_column("Week")
            .with_geography_column("Region")
            .with_product_column("SKU")
            .build()
        )

        assert config.period == "Week"
        assert config.geography == "Region"
        assert config.product == "SKU"


class TestHierarchicalConfigBuilder:
    """Tests for HierarchicalConfigBuilder."""

    def test_basic_build(self):
        """Test basic builder creates valid config."""
        from mmm_framework.builders import HierarchicalConfigBuilder

        config = HierarchicalConfigBuilder().build()

        assert config.enabled is True
        assert config.pool_across_geo is True
        assert config.pool_across_product is True

    def test_enabled(self):
        """Test enabling hierarchical."""
        from mmm_framework.builders import HierarchicalConfigBuilder

        config = HierarchicalConfigBuilder().enabled().build()

        assert config.enabled is True

    def test_disabled(self):
        """Test disabling hierarchical."""
        from mmm_framework.builders import HierarchicalConfigBuilder

        config = HierarchicalConfigBuilder().disabled().build()

        assert config.enabled is False

    def test_pool_across_geo(self):
        """Test geo pooling."""
        from mmm_framework.builders import HierarchicalConfigBuilder

        config = HierarchicalConfigBuilder().pool_across_geo().build()

        assert config.pool_across_geo is True

    def test_pool_across_product(self):
        """Test product pooling."""
        from mmm_framework.builders import HierarchicalConfigBuilder

        config = HierarchicalConfigBuilder().pool_across_product().build()

        assert config.pool_across_product is True

    def test_no_geo_pooling(self):
        """Test disabling geo pooling."""
        from mmm_framework.builders import HierarchicalConfigBuilder

        config = HierarchicalConfigBuilder().no_geo_pooling().build()

        assert config.pool_across_geo is False

    def test_no_product_pooling(self):
        """Test disabling product pooling."""
        from mmm_framework.builders import HierarchicalConfigBuilder

        config = HierarchicalConfigBuilder().no_product_pooling().build()

        assert config.pool_across_product is False

    def test_use_non_centered(self):
        """Test non-centered parameterization."""
        from mmm_framework.builders import HierarchicalConfigBuilder

        config = HierarchicalConfigBuilder().use_non_centered().build()

        assert config.use_non_centered is True

    def test_use_centered(self):
        """Test centered parameterization."""
        from mmm_framework.builders import HierarchicalConfigBuilder

        config = HierarchicalConfigBuilder().use_centered().build()

        assert config.use_non_centered is False

    def test_with_non_centered_threshold(self):
        """Test non-centered threshold setting."""
        from mmm_framework.builders import HierarchicalConfigBuilder

        config = HierarchicalConfigBuilder().with_non_centered_threshold(30).build()

        assert config.non_centered_threshold == 30

    def test_with_mu_prior(self):
        """Test setting mu prior."""
        from mmm_framework.builders import HierarchicalConfigBuilder

        prior = PriorConfigBuilder().normal(0.5, 1.0).build()
        config = HierarchicalConfigBuilder().with_mu_prior(prior).build()

        assert config.mu_prior.params["mu"] == 0.5

    def test_with_sigma_prior(self):
        """Test setting sigma prior."""
        from mmm_framework.builders import HierarchicalConfigBuilder

        prior = PriorConfigBuilder().half_normal(0.3).build()
        config = HierarchicalConfigBuilder().with_sigma_prior(prior).build()

        assert config.sigma_prior.params["sigma"] == 0.3


class TestSeasonalityConfigBuilder:
    """Tests for SeasonalityConfigBuilder."""

    def test_basic_build(self):
        """Test basic builder creates valid config."""
        from mmm_framework.builders import SeasonalityConfigBuilder

        config = SeasonalityConfigBuilder().build()

        assert config.yearly == 2  # default

    def test_with_yearly(self):
        """Test yearly seasonality."""
        from mmm_framework.builders import SeasonalityConfigBuilder

        config = SeasonalityConfigBuilder().with_yearly(3).build()

        assert config.yearly == 3

    def test_with_monthly(self):
        """Test monthly seasonality."""
        from mmm_framework.builders import SeasonalityConfigBuilder

        config = SeasonalityConfigBuilder().with_monthly(4).build()

        assert config.monthly == 4

    def test_with_weekly(self):
        """Test weekly seasonality."""
        from mmm_framework.builders import SeasonalityConfigBuilder

        config = SeasonalityConfigBuilder().with_weekly(5).build()

        assert config.weekly == 5

    def test_no_yearly(self):
        """Test disabling yearly seasonality."""
        from mmm_framework.builders import SeasonalityConfigBuilder

        config = SeasonalityConfigBuilder().no_yearly().build()

        assert config.yearly is None

    def test_no_seasonality(self):
        """Test disabling all seasonality."""
        from mmm_framework.builders import SeasonalityConfigBuilder

        config = SeasonalityConfigBuilder().no_seasonality().build()

        assert config.yearly is None
        assert config.monthly is None
        assert config.weekly is None


class TestControlSelectionConfigBuilder:
    """Tests for ControlSelectionConfigBuilder."""

    def test_basic_build(self):
        """Test basic builder creates valid config."""
        from mmm_framework.builders import ControlSelectionConfigBuilder

        config = ControlSelectionConfigBuilder().build()

        assert config.method == "none"

    def test_none(self):
        """Test no selection method."""
        from mmm_framework.builders import ControlSelectionConfigBuilder

        config = ControlSelectionConfigBuilder().none().build()

        assert config.method == "none"

    def test_horseshoe(self):
        """Test horseshoe method."""
        from mmm_framework.builders import ControlSelectionConfigBuilder

        config = ControlSelectionConfigBuilder().horseshoe(expected_nonzero=5).build()

        assert config.method == "horseshoe"
        assert config.expected_nonzero == 5

    def test_spike_slab(self):
        """Test spike-slab method."""
        from mmm_framework.builders import ControlSelectionConfigBuilder

        config = ControlSelectionConfigBuilder().spike_slab().build()

        assert config.method == "spike_slab"

    def test_lasso(self):
        """Test lasso method."""
        from mmm_framework.builders import ControlSelectionConfigBuilder

        config = ControlSelectionConfigBuilder().lasso(regularization=0.5).build()

        assert config.method == "lasso"
        assert config.regularization == 0.5

    def test_with_expected_nonzero(self):
        """Test setting expected nonzero."""
        from mmm_framework.builders import ControlSelectionConfigBuilder

        config = ControlSelectionConfigBuilder().horseshoe().with_expected_nonzero(7).build()

        assert config.expected_nonzero == 7

    def test_with_regularization(self):
        """Test setting regularization."""
        from mmm_framework.builders import ControlSelectionConfigBuilder

        config = ControlSelectionConfigBuilder().lasso().with_regularization(0.8).build()

        assert config.regularization == 0.8


class TestModelConfigBuilder:
    """Tests for ModelConfigBuilder."""

    def test_basic_build(self):
        """Test basic builder creates valid config."""
        from mmm_framework.builders import ModelConfigBuilder
        from mmm_framework.config import ModelSpecification, InferenceMethod

        config = ModelConfigBuilder().build()

        assert config.specification == ModelSpecification.ADDITIVE
        assert config.inference_method == InferenceMethod.BAYESIAN_NUMPYRO

    def test_additive(self):
        """Test additive specification."""
        from mmm_framework.builders import ModelConfigBuilder
        from mmm_framework.config import ModelSpecification

        config = ModelConfigBuilder().additive().build()

        assert config.specification == ModelSpecification.ADDITIVE

    def test_multiplicative(self):
        """Test multiplicative specification."""
        from mmm_framework.builders import ModelConfigBuilder
        from mmm_framework.config import ModelSpecification

        config = ModelConfigBuilder().multiplicative().build()

        assert config.specification == ModelSpecification.MULTIPLICATIVE

    def test_bayesian_pymc(self):
        """Test PyMC inference method."""
        from mmm_framework.builders import ModelConfigBuilder
        from mmm_framework.config import InferenceMethod

        config = ModelConfigBuilder().bayesian_pymc().build()

        assert config.inference_method == InferenceMethod.BAYESIAN_PYMC

    def test_bayesian_numpyro(self):
        """Test NumPyro inference method."""
        from mmm_framework.builders import ModelConfigBuilder
        from mmm_framework.config import InferenceMethod

        config = ModelConfigBuilder().bayesian_numpyro().build()

        assert config.inference_method == InferenceMethod.BAYESIAN_NUMPYRO

    def test_frequentist_ridge(self):
        """Test ridge regression inference method."""
        from mmm_framework.builders import ModelConfigBuilder
        from mmm_framework.config import InferenceMethod

        config = ModelConfigBuilder().frequentist_ridge().build()

        assert config.inference_method == InferenceMethod.FREQUENTIST_RIDGE

    def test_frequentist_cvxpy(self):
        """Test CVXPY inference method."""
        from mmm_framework.builders import ModelConfigBuilder
        from mmm_framework.config import InferenceMethod

        config = ModelConfigBuilder().frequentist_cvxpy().build()

        assert config.inference_method == InferenceMethod.FREQUENTIST_CVXPY

    def test_with_chains(self):
        """Test setting chains."""
        from mmm_framework.builders import ModelConfigBuilder

        config = ModelConfigBuilder().with_chains(6).build()

        assert config.n_chains == 6

    def test_with_draws(self):
        """Test setting draws."""
        from mmm_framework.builders import ModelConfigBuilder

        config = ModelConfigBuilder().with_draws(2000).build()

        assert config.n_draws == 2000

    def test_with_tune(self):
        """Test setting tune."""
        from mmm_framework.builders import ModelConfigBuilder

        config = ModelConfigBuilder().with_tune(500).build()

        assert config.n_tune == 500

    def test_with_target_accept(self):
        """Test setting target accept."""
        from mmm_framework.builders import ModelConfigBuilder

        config = ModelConfigBuilder().with_target_accept(0.95).build()

        assert config.target_accept == 0.95

    def test_with_target_accept_invalid_raises(self):
        """Test invalid target accept raises error."""
        from mmm_framework.builders import ModelConfigBuilder

        with pytest.raises(ValueError, match="between 0 and 1"):
            ModelConfigBuilder().with_target_accept(1.5).build()

    def test_with_hierarchical(self):
        """Test setting hierarchical config."""
        from mmm_framework.builders import ModelConfigBuilder, HierarchicalConfigBuilder

        hierarchical = HierarchicalConfigBuilder().disabled().build()
        config = ModelConfigBuilder().with_hierarchical(hierarchical).build()

        assert config.hierarchical.enabled is False

    def test_with_hierarchical_builder(self):
        """Test setting hierarchical from builder."""
        from mmm_framework.builders import ModelConfigBuilder, HierarchicalConfigBuilder

        config = (
            ModelConfigBuilder()
            .with_hierarchical_builder(HierarchicalConfigBuilder().disabled())
            .build()
        )

        assert config.hierarchical.enabled is False

    def test_with_seasonality(self):
        """Test setting seasonality config."""
        from mmm_framework.builders import ModelConfigBuilder, SeasonalityConfigBuilder

        seasonality = SeasonalityConfigBuilder().with_yearly(4).build()
        config = ModelConfigBuilder().with_seasonality(seasonality).build()

        assert config.seasonality.yearly == 4

    def test_with_seasonality_builder(self):
        """Test setting seasonality from builder."""
        from mmm_framework.builders import ModelConfigBuilder, SeasonalityConfigBuilder

        config = (
            ModelConfigBuilder()
            .with_seasonality_builder(SeasonalityConfigBuilder().with_yearly(5))
            .build()
        )

        assert config.seasonality.yearly == 5

    def test_with_control_selection(self):
        """Test setting control selection config."""
        from mmm_framework.builders import ModelConfigBuilder, ControlSelectionConfigBuilder

        selection = ControlSelectionConfigBuilder().horseshoe().build()
        config = ModelConfigBuilder().with_control_selection(selection).build()

        assert config.control_selection.method == "horseshoe"

    def test_with_control_selection_builder(self):
        """Test setting control selection from builder."""
        from mmm_framework.builders import ModelConfigBuilder, ControlSelectionConfigBuilder

        config = (
            ModelConfigBuilder()
            .with_control_selection_builder(ControlSelectionConfigBuilder().lasso())
            .build()
        )

        assert config.control_selection.method == "lasso"

    def test_with_ridge_alpha(self):
        """Test setting ridge alpha."""
        from mmm_framework.builders import ModelConfigBuilder

        config = ModelConfigBuilder().with_ridge_alpha(2.0).build()

        assert config.ridge_alpha == 2.0

    def test_with_bootstrap_samples(self):
        """Test setting bootstrap samples."""
        from mmm_framework.builders import ModelConfigBuilder

        config = ModelConfigBuilder().with_bootstrap_samples(500).build()

        assert config.bootstrap_samples == 500

    def test_with_optim_maxiter(self):
        """Test setting optim maxiter."""
        from mmm_framework.builders import ModelConfigBuilder

        config = ModelConfigBuilder().with_optim_maxiter(1000).build()

        assert config.optim_maxiter == 1000

    def test_with_optim_seed(self):
        """Test setting optim seed."""
        from mmm_framework.builders import ModelConfigBuilder

        config = ModelConfigBuilder().with_optim_seed(123).build()

        assert config.optim_seed == 123


class TestTrendConfigBuilder:
    """Tests for TrendConfigBuilder."""

    def test_basic_build(self):
        """Test basic builder creates valid config."""
        from mmm_framework.builders import TrendConfigBuilder

        config = TrendConfigBuilder().build()

        assert config is not None

    def test_none_trend(self):
        """Test no trend."""
        from mmm_framework.builders import TrendConfigBuilder
        from mmm_framework.model import TrendType

        config = TrendConfigBuilder().none().build()

        assert config.type == TrendType.NONE

    def test_linear_trend(self):
        """Test linear trend."""
        from mmm_framework.builders import TrendConfigBuilder
        from mmm_framework.model import TrendType

        config = TrendConfigBuilder().linear().build()

        assert config.type == TrendType.LINEAR

    def test_piecewise_trend(self):
        """Test piecewise trend."""
        from mmm_framework.builders import TrendConfigBuilder
        from mmm_framework.model import TrendType

        config = TrendConfigBuilder().piecewise().build()

        assert config.type == TrendType.PIECEWISE

    def test_spline_trend(self):
        """Test spline trend."""
        from mmm_framework.builders import TrendConfigBuilder
        from mmm_framework.model import TrendType

        config = TrendConfigBuilder().spline().build()

        assert config.type == TrendType.SPLINE

    def test_gaussian_process_trend(self):
        """Test GP trend."""
        from mmm_framework.builders import TrendConfigBuilder
        from mmm_framework.model import TrendType

        config = TrendConfigBuilder().gaussian_process().build()

        assert config.type == TrendType.GP

    def test_gp_alias(self):
        """Test GP alias."""
        from mmm_framework.builders import TrendConfigBuilder
        from mmm_framework.model import TrendType

        config = TrendConfigBuilder().gp().build()

        assert config.type == TrendType.GP

    def test_with_n_changepoints(self):
        """Test setting changepoints."""
        from mmm_framework.builders import TrendConfigBuilder

        config = TrendConfigBuilder().piecewise().with_n_changepoints(15).build()

        assert config.n_changepoints == 15

    def test_with_n_changepoints_negative_raises(self):
        """Test negative changepoints raises error."""
        from mmm_framework.builders import TrendConfigBuilder

        with pytest.raises(ValueError, match="non-negative"):
            TrendConfigBuilder().piecewise().with_n_changepoints(-1).build()

    def test_with_changepoint_range(self):
        """Test setting changepoint range."""
        from mmm_framework.builders import TrendConfigBuilder

        config = TrendConfigBuilder().piecewise().with_changepoint_range(0.9).build()

        assert config.changepoint_range == 0.9

    def test_with_changepoint_range_invalid_raises(self):
        """Test invalid changepoint range raises error."""
        from mmm_framework.builders import TrendConfigBuilder

        with pytest.raises(ValueError, match="\\(0, 1\\]"):
            TrendConfigBuilder().piecewise().with_changepoint_range(1.5).build()

    def test_with_changepoint_prior_scale(self):
        """Test setting changepoint prior scale."""
        from mmm_framework.builders import TrendConfigBuilder

        config = TrendConfigBuilder().piecewise().with_changepoint_prior_scale(0.1).build()

        assert config.changepoint_prior_scale == 0.1

    def test_with_changepoint_prior_scale_nonpositive_raises(self):
        """Test non-positive prior scale raises error."""
        from mmm_framework.builders import TrendConfigBuilder

        with pytest.raises(ValueError, match="positive"):
            TrendConfigBuilder().piecewise().with_changepoint_prior_scale(0).build()

    def test_with_n_knots(self):
        """Test setting n_knots."""
        from mmm_framework.builders import TrendConfigBuilder

        config = TrendConfigBuilder().spline().with_n_knots(20).build()

        assert config.n_knots == 20

    def test_with_n_knots_less_than_1_raises(self):
        """Test n_knots < 1 raises error."""
        from mmm_framework.builders import TrendConfigBuilder

        with pytest.raises(ValueError, match="at least 1"):
            TrendConfigBuilder().spline().with_n_knots(0).build()

    def test_with_spline_degree(self):
        """Test setting spline degree."""
        from mmm_framework.builders import TrendConfigBuilder

        config = TrendConfigBuilder().spline().with_spline_degree(2).build()

        assert config.spline_degree == 2

    def test_with_spline_prior_sigma(self):
        """Test setting spline prior sigma."""
        from mmm_framework.builders import TrendConfigBuilder

        config = TrendConfigBuilder().spline().with_spline_prior_sigma(2.0).build()

        assert config.spline_prior_sigma == 2.0

    def test_with_gp_lengthscale(self):
        """Test setting GP lengthscale."""
        from mmm_framework.builders import TrendConfigBuilder

        config = TrendConfigBuilder().gp().with_gp_lengthscale(mu=0.4, sigma=0.15).build()

        assert config.gp_lengthscale_prior_mu == 0.4
        assert config.gp_lengthscale_prior_sigma == 0.15

    def test_with_gp_amplitude(self):
        """Test setting GP amplitude."""
        from mmm_framework.builders import TrendConfigBuilder

        config = TrendConfigBuilder().gp().with_gp_amplitude(sigma=0.7).build()

        assert config.gp_amplitude_prior_sigma == 0.7

    def test_with_gp_n_basis(self):
        """Test setting GP n_basis."""
        from mmm_framework.builders import TrendConfigBuilder

        config = TrendConfigBuilder().gp().with_gp_n_basis(30).build()

        assert config.gp_n_basis == 30

    def test_with_gp_n_basis_less_than_5_raises(self):
        """Test n_basis < 5 raises error."""
        from mmm_framework.builders import TrendConfigBuilder

        with pytest.raises(ValueError, match="at least 5"):
            TrendConfigBuilder().gp().with_gp_n_basis(3).build()

    def test_with_gp_boundary_factor(self):
        """Test setting GP boundary factor."""
        from mmm_framework.builders import TrendConfigBuilder

        config = TrendConfigBuilder().gp().with_gp_boundary_factor(2.0).build()

        assert config.gp_c == 2.0

    def test_with_growth_prior(self):
        """Test setting growth prior."""
        from mmm_framework.builders import TrendConfigBuilder

        config = TrendConfigBuilder().linear().with_growth_prior(mu=0.1, sigma=0.05).build()

        assert config.growth_prior_mu == 0.1
        assert config.growth_prior_sigma == 0.05

    def test_smooth_preset(self):
        """Test smooth preset."""
        from mmm_framework.builders import TrendConfigBuilder
        from mmm_framework.model import TrendType

        config = TrendConfigBuilder().smooth().build()

        assert config.type == TrendType.GP
        assert config.gp_lengthscale_prior_mu == 0.5

    def test_flexible_preset(self):
        """Test flexible preset."""
        from mmm_framework.builders import TrendConfigBuilder
        from mmm_framework.model import TrendType

        config = TrendConfigBuilder().flexible().build()

        assert config.type == TrendType.SPLINE
        assert config.n_knots == 15

    def test_changepoint_detection_preset(self):
        """Test changepoint detection preset."""
        from mmm_framework.builders import TrendConfigBuilder
        from mmm_framework.model import TrendType

        config = TrendConfigBuilder().changepoint_detection().build()

        assert config.type == TrendType.PIECEWISE
        assert config.n_changepoints == 15


class TestMFFConfigBuilder:
    """Tests for MFFConfigBuilder."""

    def test_build_without_kpi_raises(self):
        """Test build without KPI raises error."""
        from mmm_framework.builders import MFFConfigBuilder

        with pytest.raises(ValueError, match="KPI configuration is required"):
            MFFConfigBuilder().build()

    def test_build_without_media_raises(self):
        """Test build without media raises error."""
        from mmm_framework.builders import MFFConfigBuilder

        with pytest.raises(ValueError, match="At least one media channel is required"):
            (
                MFFConfigBuilder()
                .with_kpi_name("Sales")
                .build()
            )

    def test_basic_build(self):
        """Test basic builder creates valid config."""
        from mmm_framework.builders import MFFConfigBuilder

        config = (
            MFFConfigBuilder()
            .with_kpi_name("Sales")
            .add_national_media("TV")
            .build()
        )

        assert config.kpi.name == "Sales"
        assert len(config.media_channels) == 1
        assert config.media_channels[0].name == "TV"

    def test_with_kpi(self):
        """Test setting KPI."""
        from mmm_framework.builders import MFFConfigBuilder

        kpi = KPIConfigBuilder("Revenue").by_geo().build()
        config = (
            MFFConfigBuilder()
            .with_kpi(kpi)
            .add_national_media("TV")
            .build()
        )

        assert config.kpi.name == "Revenue"
        assert DimensionType.GEOGRAPHY in config.kpi.dimensions

    def test_with_kpi_builder(self):
        """Test setting KPI from builder."""
        from mmm_framework.builders import MFFConfigBuilder

        config = (
            MFFConfigBuilder()
            .with_kpi_builder(KPIConfigBuilder("Revenue").multiplicative())
            .add_national_media("TV")
            .build()
        )

        assert config.kpi.name == "Revenue"
        assert config.kpi.log_transform is True

    def test_add_media(self):
        """Test adding media."""
        from mmm_framework.builders import MFFConfigBuilder

        media = MediaChannelConfigBuilder("TV").with_geometric_adstock(12).build()
        config = (
            MFFConfigBuilder()
            .with_kpi_name("Sales")
            .add_media(media)
            .build()
        )

        assert config.media_channels[0].adstock.l_max == 12

    def test_add_media_builder(self):
        """Test adding media from builder."""
        from mmm_framework.builders import MFFConfigBuilder

        config = (
            MFFConfigBuilder()
            .with_kpi_name("Sales")
            .add_media_builder(MediaChannelConfigBuilder("TV").by_geo())
            .build()
        )

        assert DimensionType.GEOGRAPHY in config.media_channels[0].dimensions

    def test_add_media_channels(self):
        """Test adding multiple media channels."""
        from mmm_framework.builders import MFFConfigBuilder

        tv = MediaChannelConfigBuilder("TV").build()
        radio = MediaChannelConfigBuilder("Radio").build()
        config = (
            MFFConfigBuilder()
            .with_kpi_name("Sales")
            .add_media_channels(tv, radio)
            .build()
        )

        assert len(config.media_channels) == 2

    def test_add_national_media(self):
        """Test adding national media."""
        from mmm_framework.builders import MFFConfigBuilder

        config = (
            MFFConfigBuilder()
            .with_kpi_name("Sales")
            .add_national_media("TV", adstock_lmax=10)
            .build()
        )

        assert config.media_channels[0].dimensions == [DimensionType.PERIOD]
        assert config.media_channels[0].adstock.l_max == 10

    def test_add_social_platforms(self):
        """Test adding social platforms."""
        from mmm_framework.builders import MFFConfigBuilder

        config = (
            MFFConfigBuilder()
            .with_kpi_name("Sales")
            .add_social_platforms(["Meta", "TikTok"], parent_name="Social")
            .build()
        )

        assert len(config.media_channels) == 2
        assert config.media_channels[0].parent_channel == "Social"
        assert config.media_channels[1].parent_channel == "Social"

    def test_add_control(self):
        """Test adding control."""
        from mmm_framework.builders import MFFConfigBuilder

        control = ControlVariableConfigBuilder("Price").allow_negative().build()
        config = (
            MFFConfigBuilder()
            .with_kpi_name("Sales")
            .add_national_media("TV")
            .add_control(control)
            .build()
        )

        assert len(config.controls) == 1
        assert config.controls[0].allow_negative is True

    def test_add_control_builder(self):
        """Test adding control from builder."""
        from mmm_framework.builders import MFFConfigBuilder

        config = (
            MFFConfigBuilder()
            .with_kpi_name("Sales")
            .add_national_media("TV")
            .add_control_builder(ControlVariableConfigBuilder("Promo").positive_only())
            .build()
        )

        assert config.controls[0].allow_negative is False

    def test_add_controls(self):
        """Test adding multiple controls."""
        from mmm_framework.builders import MFFConfigBuilder

        price = ControlVariableConfigBuilder("Price").build()
        promo = ControlVariableConfigBuilder("Promo").build()
        config = (
            MFFConfigBuilder()
            .with_kpi_name("Sales")
            .add_national_media("TV")
            .add_controls(price, promo)
            .build()
        )

        assert len(config.controls) == 2

    def test_add_price_control(self):
        """Test adding price control convenience method."""
        from mmm_framework.builders import MFFConfigBuilder

        config = (
            MFFConfigBuilder()
            .with_kpi_name("Sales")
            .add_national_media("TV")
            .add_price_control()
            .build()
        )

        assert config.controls[0].name == "Price"
        assert config.controls[0].allow_negative is True

    def test_add_distribution_control(self):
        """Test adding distribution control convenience method."""
        from mmm_framework.builders import MFFConfigBuilder

        config = (
            MFFConfigBuilder()
            .with_kpi_name("Sales")
            .add_national_media("TV")
            .add_distribution_control()
            .build()
        )

        assert config.controls[0].name == "Distribution"
        assert config.controls[0].allow_negative is False

    def test_with_columns(self):
        """Test setting columns config."""
        from mmm_framework.builders import MFFConfigBuilder, MFFColumnConfigBuilder

        columns = MFFColumnConfigBuilder().with_period_column("Week").build()
        config = (
            MFFConfigBuilder()
            .with_kpi_name("Sales")
            .add_national_media("TV")
            .with_columns(columns)
            .build()
        )

        assert config.columns.period == "Week"

    def test_with_columns_builder(self):
        """Test setting columns from builder."""
        from mmm_framework.builders import MFFConfigBuilder, MFFColumnConfigBuilder

        config = (
            MFFConfigBuilder()
            .with_kpi_name("Sales")
            .add_national_media("TV")
            .with_columns_builder(MFFColumnConfigBuilder().with_geography_column("Region"))
            .build()
        )

        assert config.columns.geography == "Region"

    def test_with_alignment(self):
        """Test setting alignment config."""
        from mmm_framework.builders import MFFConfigBuilder, DimensionAlignmentConfigBuilder
        from mmm_framework.config import AllocationMethod

        alignment = DimensionAlignmentConfigBuilder().geo_by_sales().build()
        config = (
            MFFConfigBuilder()
            .with_kpi_name("Sales")
            .add_national_media("TV")
            .with_alignment(alignment)
            .build()
        )

        assert config.alignment.geo_allocation == AllocationMethod.SALES

    def test_with_alignment_builder(self):
        """Test setting alignment from builder."""
        from mmm_framework.builders import MFFConfigBuilder, DimensionAlignmentConfigBuilder
        from mmm_framework.config import AllocationMethod

        config = (
            MFFConfigBuilder()
            .with_kpi_name("Sales")
            .add_national_media("TV")
            .with_alignment_builder(DimensionAlignmentConfigBuilder().geo_equal())
            .build()
        )

        assert config.alignment.geo_allocation == AllocationMethod.EQUAL

    def test_with_date_format(self):
        """Test setting date format."""
        from mmm_framework.builders import MFFConfigBuilder

        config = (
            MFFConfigBuilder()
            .with_kpi_name("Sales")
            .add_national_media("TV")
            .with_date_format("%d/%m/%Y")
            .build()
        )

        assert config.date_format == "%d/%m/%Y"

    def test_weekly(self):
        """Test weekly frequency."""
        from mmm_framework.builders import MFFConfigBuilder

        config = (
            MFFConfigBuilder()
            .with_kpi_name("Sales")
            .add_national_media("TV")
            .weekly()
            .build()
        )

        assert config.frequency == "W"

    def test_daily(self):
        """Test daily frequency."""
        from mmm_framework.builders import MFFConfigBuilder

        config = (
            MFFConfigBuilder()
            .with_kpi_name("Sales")
            .add_national_media("TV")
            .daily()
            .build()
        )

        assert config.frequency == "D"

    def test_monthly(self):
        """Test monthly frequency."""
        from mmm_framework.builders import MFFConfigBuilder

        config = (
            MFFConfigBuilder()
            .with_kpi_name("Sales")
            .add_national_media("TV")
            .monthly()
            .build()
        )

        assert config.frequency == "M"

    def test_with_fill_missing_media(self):
        """Test setting fill missing media."""
        from mmm_framework.builders import MFFConfigBuilder

        config = (
            MFFConfigBuilder()
            .with_kpi_name("Sales")
            .add_national_media("TV")
            .with_fill_missing_media(0.5)
            .build()
        )

        assert config.fill_missing_media == 0.5

    def test_with_fill_missing_controls(self):
        """Test setting fill missing controls."""
        from mmm_framework.builders import MFFConfigBuilder

        config = (
            MFFConfigBuilder()
            .with_kpi_name("Sales")
            .add_national_media("TV")
            .with_fill_missing_controls(0.0)
            .build()
        )

        assert config.fill_missing_controls == 0.0
