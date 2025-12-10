"""
Examples demonstrating the builder pattern for MMM configuration.

The builder pattern provides a fluent API for constructing complex configuration
objects step-by-step with method chaining.
"""

from mmm_framework import (
    # Builders
    PriorConfigBuilder,
    AdstockConfigBuilder,
    SaturationConfigBuilder,
    MediaChannelConfigBuilder,
    ControlVariableConfigBuilder,
    KPIConfigBuilder,
    DimensionAlignmentConfigBuilder,
    HierarchicalConfigBuilder,
    SeasonalityConfigBuilder,
    ControlSelectionConfigBuilder,
    ModelConfigBuilder,
    MFFConfigBuilder,
    # Enums
    DimensionType,
    # Data loading
    load_mff,
)


# =============================================================================
# Example 1: Building Priors
# =============================================================================

def example_prior_builders():
    """Demonstrate prior config builders."""
    
    print("=" * 60)
    print("Example 1: Prior Builders")
    print("=" * 60)
    
    # Half-normal for positive effects
    positive_prior = (PriorConfigBuilder()
        .half_normal(sigma=2.0)
        .with_dims("channel")
        .build())
    print(f"Positive prior: {positive_prior.distribution.value}(sigma={positive_prior.params['sigma']})")
    
    # Beta for decay rates (bounded 0-1)
    decay_prior = (PriorConfigBuilder()
        .beta(alpha=1, beta=3)
        .build())
    print(f"Decay prior: {decay_prior.distribution.value}(α={decay_prior.params['alpha']}, β={decay_prior.params['beta']})")
    
    # Gamma for scale parameters
    scale_prior = (PriorConfigBuilder()
        .gamma(alpha=2, beta=1)
        .with_dims(["channel", "geo"])
        .build())
    print(f"Scale prior: {scale_prior.distribution.value} with dims={scale_prior.dims}")
    
    # Truncated normal for informed priors
    informed_prior = (PriorConfigBuilder()
        .truncated_normal(mu=0.05, sigma=0.02, lower=0)
        .build())
    print(f"Informed prior: {informed_prior.distribution.value}(μ={informed_prior.params['mu']})")
    
    return positive_prior, decay_prior, scale_prior, informed_prior


# =============================================================================
# Example 2: Building Transformations
# =============================================================================

def example_transformation_builders():
    """Demonstrate adstock and saturation builders."""
    
    print("\n" + "=" * 60)
    print("Example 2: Transformation Builders")
    print("=" * 60)
    
    # Geometric adstock with custom prior
    adstock = (AdstockConfigBuilder()
        .geometric()
        .with_max_lag(8)
        .with_alpha_prior(
            PriorConfigBuilder().beta(alpha=1, beta=3).build()
        )
        .build())
    print(f"Adstock: {adstock.type.value}, l_max={adstock.l_max}")
    
    # Hill saturation with constrained kappa
    saturation = (SaturationConfigBuilder()
        .hill()
        .with_kappa_prior(PriorConfigBuilder().beta(2, 2).build())
        .with_slope_prior(PriorConfigBuilder().half_normal(1.5).build())
        .with_kappa_bounds(0.1, 0.9)
        .build())
    print(f"Saturation: {saturation.type.value}, kappa_bounds={saturation.kappa_bounds_percentiles}")
    
    # No transformation
    no_adstock = AdstockConfigBuilder().none().build()
    no_saturation = SaturationConfigBuilder().none().build()
    print(f"No transforms: adstock={no_adstock.type.value}, saturation={no_saturation.type.value}")
    
    return adstock, saturation


# =============================================================================
# Example 3: Building Media Channels
# =============================================================================

def example_media_builders():
    """Demonstrate media channel builders."""
    
    print("\n" + "=" * 60)
    print("Example 3: Media Channel Builders")
    print("=" * 60)
    
    # TV - national, long carryover
    tv = (MediaChannelConfigBuilder("TV")
        .with_display_name("Television")
        .with_unit("USD")
        .national()
        .with_geometric_adstock(l_max=8)
        .with_hill_saturation()
        .with_positive_prior(sigma=2.0)
        .build())
    print(f"TV: dims={tv.dim_names}, adstock_lmax={tv.adstock.l_max}")
    
    # Digital - national, shorter carryover
    digital = (MediaChannelConfigBuilder("Digital")
        .national()
        .with_adstock_builder(
            AdstockConfigBuilder().geometric().with_max_lag(4)
        )
        .with_saturation_builder(
            SaturationConfigBuilder()
            .hill()
            .with_kappa_prior(PriorConfigBuilder().gamma(3, 2).build())
        )
        .build())
    print(f"Digital: adstock_lmax={digital.adstock.l_max}")
    
    # Geo-level media
    local_radio = (MediaChannelConfigBuilder("LocalRadio")
        .by_geo()
        .with_geometric_adstock(6)
        .with_hill_saturation()
        .build())
    print(f"Local Radio: dims={local_radio.dim_names}")
    
    # Social platforms with hierarchy
    meta = (MediaChannelConfigBuilder("Meta")
        .national()
        .with_parent_channel("Social")
        .with_split_dimensions(DimensionType.OUTLET)
        .with_geometric_adstock(4)
        .with_hill_saturation()
        .build())
    
    snapchat = (MediaChannelConfigBuilder("Snapchat")
        .national()
        .with_parent_channel("Social")
        .with_split_dimensions(DimensionType.OUTLET)
        .with_geometric_adstock(4)
        .build())
    
    print(f"Meta: parent={meta.parent_channel}, splits={[d.value for d in meta.split_dimensions]}")
    print(f"Snapchat: parent={snapchat.parent_channel}")
    
    return tv, digital, local_radio, meta, snapchat


# =============================================================================
# Example 4: Building Controls
# =============================================================================

def example_control_builders():
    """Demonstrate control variable builders."""
    
    print("\n" + "=" * 60)
    print("Example 4: Control Variable Builders")
    print("=" * 60)
    
    # Price - allows negative effect
    price = (ControlVariableConfigBuilder("Price")
        .with_display_name("Price Index")
        .national()
        .allow_negative()
        .with_normal_prior(mu=0, sigma=1)
        .build())
    print(f"Price: allow_negative={price.allow_negative}")
    
    # Distribution - positive only
    distribution = (ControlVariableConfigBuilder("Distribution")
        .with_display_name("ACV Distribution")
        .national()
        .positive_only()
        .build())
    print(f"Distribution: allow_negative={distribution.allow_negative}")
    
    # Promotion with shrinkage
    promo = (ControlVariableConfigBuilder("Promotion")
        .national()
        .positive_only()
        .with_shrinkage()
        .build())
    print(f"Promotion: use_shrinkage={promo.use_shrinkage}")
    
    # Geo-level control
    competitor = (ControlVariableConfigBuilder("CompetitorPrice")
        .by_geo()
        .allow_negative()
        .build())
    print(f"Competitor Price: dims={competitor.dim_names}")
    
    return price, distribution, promo, competitor


# =============================================================================
# Example 5: Building KPI
# =============================================================================

def example_kpi_builders():
    """Demonstrate KPI builders."""
    
    print("\n" + "=" * 60)
    print("Example 5: KPI Builders")
    print("=" * 60)
    
    # National additive
    national_sales = (KPIConfigBuilder("Sales")
        .with_display_name("Total Sales")
        .with_unit("Units")
        .national()
        .additive()
        .build())
    print(f"National Sales: dims={national_sales.dim_names}, log={national_sales.log_transform}")
    
    # Geo-level multiplicative
    geo_revenue = (KPIConfigBuilder("Revenue")
        .by_geo()
        .multiplicative()
        .with_floor_value(1.0)
        .build())
    print(f"Geo Revenue: dims={geo_revenue.dim_names}, log={geo_revenue.log_transform}")
    
    # Full granularity
    full_kpi = (KPIConfigBuilder("Sales")
        .by_geo_and_product()
        .additive()
        .build())
    print(f"Full KPI: dims={full_kpi.dim_names}")
    
    return national_sales, geo_revenue, full_kpi


# =============================================================================
# Example 6: Building Model Config
# =============================================================================

def example_model_builders():
    """Demonstrate model configuration builders."""
    
    print("\n" + "=" * 60)
    print("Example 6: Model Config Builders")
    print("=" * 60)
    
    # Hierarchical config
    hierarchical = (HierarchicalConfigBuilder()
        .enabled()
        .pool_across_geo()
        .pool_across_product()
        .use_non_centered()
        .with_non_centered_threshold(20)
        .with_mu_prior(PriorConfigBuilder().normal(0, 1).build())
        .with_sigma_prior(PriorConfigBuilder().half_normal(0.5).build())
        .build())
    print(f"Hierarchical: enabled={hierarchical.enabled}, geo_pooling={hierarchical.pool_across_geo}")
    
    # Seasonality
    seasonality = (SeasonalityConfigBuilder()
        .with_yearly(order=2)
        .with_monthly(order=1)
        .build())
    print(f"Seasonality: yearly={seasonality.yearly}, monthly={seasonality.monthly}")
    
    # Control selection with horseshoe
    control_selection = (ControlSelectionConfigBuilder()
        .horseshoe(expected_nonzero=3)
        .build())
    print(f"Control selection: method={control_selection.method}")
    
    # Full model config - Bayesian
    bayesian_model = (ModelConfigBuilder()
        .additive()
        .bayesian_numpyro()
        .with_chains(4)
        .with_draws(2000)
        .with_tune(1000)
        .with_target_accept(0.9)
        .with_hierarchical(hierarchical)
        .with_seasonality(seasonality)
        .with_control_selection(control_selection)
        .build())
    print(f"Bayesian model: method={bayesian_model.inference_method.value}, "
          f"chains={bayesian_model.n_chains}, draws={bayesian_model.n_draws}")
    
    # Fast frequentist config
    frequentist_model = (ModelConfigBuilder()
        .additive()
        .frequentist_ridge()
        .with_ridge_alpha(1.0)
        .with_bootstrap_samples(1000)
        .with_optim_maxiter(500)
        .build())
    print(f"Frequentist model: method={frequentist_model.inference_method.value}, "
          f"alpha={frequentist_model.ridge_alpha}")
    
    return bayesian_model, frequentist_model


# =============================================================================
# Example 7: Building Complete MFF Config
# =============================================================================

def example_complete_mff_builder():
    """Demonstrate the complete MFF config builder."""
    
    print("\n" + "=" * 60)
    print("Example 7: Complete MFF Config Builder")
    print("=" * 60)
    
    # Build complete config using chained builders
    config = (MFFConfigBuilder()
        # KPI
        .with_kpi_builder(
            KPIConfigBuilder("Sales")
            .by_geo()
            .additive()
        )
        # Media channels
        .add_media_builder(
            MediaChannelConfigBuilder("TV")
            .national()
            .with_geometric_adstock(8)
            .with_hill_saturation()
        )
        .add_media_builder(
            MediaChannelConfigBuilder("Digital")
            .national()
            .with_geometric_adstock(4)
            .with_saturation_builder(
                SaturationConfigBuilder()
                .hill()
                .with_kappa_prior(PriorConfigBuilder().gamma(3, 2).build())
            )
        )
        .add_media_builder(
            MediaChannelConfigBuilder("Radio")
            .national()
            .with_geometric_adstock(6)
            .with_hill_saturation()
        )
        # Social hierarchy
        .add_social_platforms(
            platforms=["Meta", "Snapchat", "Twitter"],
            parent_name="Social",
            adstock_lmax=4
        )
        # Controls
        .add_control_builder(
            ControlVariableConfigBuilder("Price")
            .national()
            .allow_negative()
            .with_normal_prior(mu=-0.5, sigma=0.5)
        )
        .add_control_builder(
            ControlVariableConfigBuilder("Distribution")
            .national()
            .positive_only()
        )
        # Alignment
        .with_alignment_builder(
            DimensionAlignmentConfigBuilder()
            .geo_by_sales()
            .product_by_sales()
        )
        # Data settings
        .with_date_format("%Y-%m-%d")
        .weekly()
        .with_fill_missing_media(0.0)
        .build())
    
    print(f"KPI: {config.kpi.name} with dims {config.kpi.dim_names}")
    print(f"Media channels: {config.media_names}")
    print(f"Controls: {config.control_names}")
    print(f"Hierarchical groups: {config.get_hierarchical_media_groups()}")
    
    return config


# =============================================================================
# Example 8: Shorthand vs Explicit Builder Comparison
# =============================================================================

def example_builder_styles():
    """Compare shorthand convenience methods vs explicit builders."""
    
    print("\n" + "=" * 60)
    print("Example 8: Builder Styles Comparison")
    print("=" * 60)
    
    # Style 1: Maximum convenience (shorthand methods)
    config_shorthand = (MFFConfigBuilder()
        .with_kpi_name("Sales")
        .add_national_media("TV", adstock_lmax=8)
        .add_national_media("Digital", adstock_lmax=4)
        .add_price_control()
        .add_distribution_control()
        .build())
    print("Shorthand style - minimal code:")
    print(f"  Media: {config_shorthand.media_names}")
    print(f"  Controls: {config_shorthand.control_names}")
    
    # Style 2: Explicit control (full builder chain)
    config_explicit = (MFFConfigBuilder()
        .with_kpi_builder(
            KPIConfigBuilder("Sales")
            .with_display_name("Weekly Sales Volume")
            .with_unit("Units")
            .national()
            .additive()
            .with_floor_value(1.0)
        )
        .add_media_builder(
            MediaChannelConfigBuilder("TV")
            .with_display_name("Television Advertising")
            .with_unit("USD")
            .national()
            .with_adstock_builder(
                AdstockConfigBuilder()
                .geometric()
                .with_max_lag(8)
                .with_alpha_prior(
                    PriorConfigBuilder()
                    .beta(alpha=1, beta=3)
                    .with_dims("channel")
                    .build()
                )
            )
            .with_saturation_builder(
                SaturationConfigBuilder()
                .hill()
                .with_kappa_prior(PriorConfigBuilder().beta(2, 2).build())
                .with_slope_prior(PriorConfigBuilder().half_normal(1.5).build())
                .with_beta_prior(PriorConfigBuilder().half_normal(2.0).build())
                .with_kappa_bounds(0.1, 0.9)
            )
            .with_coefficient_prior(
                PriorConfigBuilder()
                .half_normal(sigma=2.0)
                .build()
            )
        )
        .add_control_builder(
            ControlVariableConfigBuilder("Price")
            .with_display_name("Price Index")
            .national()
            .allow_negative()
            .with_coefficient_prior(
                PriorConfigBuilder()
                .normal(mu=-0.5, sigma=0.5)
                .build()
            )
        )
        .with_alignment_builder(
            DimensionAlignmentConfigBuilder()
            .geo_by_population()
        )
        .with_date_format("%Y-%m-%d")
        .weekly()
        .build())
    print("\nExplicit style - full control:")
    print(f"  KPI display: {config_explicit.kpi.display_name}")
    print(f"  TV display: {config_explicit.media_channels[0].display_name}")
    print(f"  TV adstock prior: {config_explicit.media_channels[0].adstock.alpha_prior.distribution.value}")
    
    return config_shorthand, config_explicit


# =============================================================================
# Example 9: Using Builders with Data Loading
# =============================================================================

def example_builder_with_data():
    """Demonstrate using builders end-to-end with data loading."""
    
    print("\n" + "=" * 60)
    print("Example 9: Builders with Data Loading")
    print("=" * 60)
    
    # Import the data generator from examples
    from ex_config import generate_synthetic_mff
    
    # Generate test data with geos
    mff_data = generate_synthetic_mff(
        n_weeks=104,  # 2 years
        geographies=["East", "West", "Central"],
    )
    print(f"Generated {len(mff_data)} MFF records")
    
    # Build config using builder pattern
    config = (MFFConfigBuilder()
        .with_kpi_builder(
            KPIConfigBuilder("Sales")
            .by_geo()
            .additive()
        )
        .add_media_builder(
            MediaChannelConfigBuilder("TV")
            .national()
            .with_geometric_adstock(8)
            .with_hill_saturation()
        )
        .add_media_builder(
            MediaChannelConfigBuilder("Digital")
            .national()
            .with_geometric_adstock(4)
            .with_hill_saturation()
        )
        .add_media_builder(
            MediaChannelConfigBuilder("Radio")
            .national()
            .with_geometric_adstock(6)
            .with_hill_saturation()
        )
        .add_control_builder(
            ControlVariableConfigBuilder("Price")
            .national()
            .allow_negative()
        )
        .add_control_builder(
            ControlVariableConfigBuilder("Distribution")
            .national()
            .positive_only()
        )
        .with_alignment_builder(
            DimensionAlignmentConfigBuilder()
            .geo_by_sales()
        )
        .build())
    
    # Load data
    panel = load_mff(mff_data, config)
    
    print(f"\nLoaded panel:")
    print(f"  Observations: {panel.n_obs}")
    print(f"  Periods: {panel.coords.n_periods}")
    print(f"  Geos: {panel.coords.n_geos}")
    print(f"  Channels: {panel.n_channels}")
    print(f"  Controls: {panel.n_controls}")
    
    # Build model config
    model_config = (ModelConfigBuilder()
        .additive()
        .bayesian_numpyro()
        .with_chains(4)
        .with_draws(1000)
        .with_hierarchical_builder(
            HierarchicalConfigBuilder()
            .enabled()
            .pool_across_geo()
            .use_non_centered()
        )
        .with_seasonality_builder(
            SeasonalityConfigBuilder()
            .with_yearly(order=2)
        )
        .build())
    
    print(f"\nModel config:")
    print(f"  Method: {model_config.inference_method.value}")
    print(f"  Hierarchical: {model_config.hierarchical.enabled}")
    print(f"  Seasonality: yearly={model_config.seasonality.yearly}")
    
    return panel, config, model_config


# =============================================================================
# Run all examples
# =============================================================================

if __name__ == "__main__":
    # Run all examples
    example_prior_builders()
    example_transformation_builders()
    example_media_builders()
    example_control_builders()
    example_kpi_builders()
    example_model_builders()
    example_complete_mff_builder()
    example_builder_styles()
    example_builder_with_data()
    
    print("\n" + "=" * 60)
    print("All builder examples completed successfully!")
    print("=" * 60)