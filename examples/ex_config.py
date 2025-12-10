"""
Example usage of the flexible MMM framework.

Demonstrates:
1. Simple national-level configuration
2. Geo-level panel configuration  
3. Multi-product hierarchical configuration
4. Social media platform splits
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import from our framework
from mmm_framework import (
    # Config classes
    DimensionType,
    MFFConfig,
    KPIConfig,
    MediaChannelConfig,
    ControlVariableConfig,
    AdstockConfig,
    SaturationConfig,
    PriorConfig,
    DimensionAlignmentConfig,
    AllocationMethod,
    ModelConfig,
    InferenceMethod,
    # Factory functions
    create_national_media_config,
    create_geo_media_config,
    create_social_platform_configs,
    create_simple_mff_config,
    # Data loading
    MFFLoader,
    load_mff,
    mff_from_wide_format,
)


# =============================================================================
# Helper: Generate synthetic MFF data for testing
# =============================================================================

def generate_synthetic_mff(
    n_weeks: int = 156,
    geographies: list[str] | None = None,
    products: list[str] | None = None,
    media_channels: list[str] | None = None,
    include_social_splits: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic MFF data for testing."""
    
    np.random.seed(seed)
    
    # Defaults
    if geographies is None:
        geographies = ["National"]
    if products is None:
        products = ["All"]
    if media_channels is None:
        media_channels = ["TV", "Digital", "Radio"]
    
    # Date range
    start_date = datetime(2022, 1, 3)  # Start on Monday
    dates = [start_date + timedelta(weeks=i) for i in range(n_weeks)]
    
    records = []
    
    for date in dates:
        for geo in geographies:
            for product in products:
                # Base sales with trend and seasonality
                week_num = (date - start_date).days // 7
                trend = 1000 + 2 * week_num
                seasonality = 200 * np.sin(2 * np.pi * week_num / 52)
                noise = np.random.normal(0, 100)
                
                # Geo and product effects
                geo_effect = {"East": 1.2, "West": 0.9, "Central": 1.0, "National": 1.0}.get(geo, 1.0)
                prod_effect = {"Premium": 1.3, "Standard": 1.0, "Budget": 0.7, "All": 1.0}.get(product, 1.0)
                
                base_sales = (trend + seasonality + noise) * geo_effect * prod_effect
                
                # KPI record
                records.append({
                    "Period": date.strftime("%Y-%m-%d"),
                    "Geography": geo,
                    "Product": product,
                    "Campaign": "",
                    "Outlet": "",
                    "Creative": "",
                    "VariableName": "Sales",
                    "VariableValue": max(0, base_sales),
                })
                
                # Media records (national level - same for all geos/products)
                if geo == geographies[0] and product == products[0]:
                    for channel in media_channels:
                        # Pulsed media spending
                        spend = 0
                        if channel == "TV":
                            # Heavy in Q4
                            if date.month in [10, 11, 12]:
                                spend = np.random.uniform(50000, 100000)
                            else:
                                spend = np.random.uniform(10000, 30000)
                        elif channel == "Digital":
                            # Always on with variation
                            spend = np.random.uniform(20000, 50000)
                        elif channel == "Radio":
                            # Sporadic
                            if np.random.random() > 0.3:
                                spend = np.random.uniform(5000, 20000)
                        
                        records.append({
                            "Period": date.strftime("%Y-%m-%d"),
                            "Geography": "",  # National
                            "Product": "",
                            "Campaign": "",
                            "Outlet": "",
                            "Creative": "",
                            "VariableName": channel,
                            "VariableValue": spend,
                        })
                    
                    # Social splits if requested
                    if include_social_splits:
                        social_spend = np.random.uniform(15000, 40000)
                        platforms = {"Meta": 0.6, "Snapchat": 0.25, "Twitter": 0.15}
                        for platform, share in platforms.items():
                            records.append({
                                "Period": date.strftime("%Y-%m-%d"),
                                "Geography": "",
                                "Product": "",
                                "Campaign": "",
                                "Outlet": platform,
                                "Creative": "",
                                "VariableName": "Social",
                                "VariableValue": social_spend * share,
                            })
                
                # Control variables
                if geo == geographies[0] and product == products[0]:
                    # Price index
                    records.append({
                        "Period": date.strftime("%Y-%m-%d"),
                        "Geography": "",
                        "Product": "",
                        "Campaign": "",
                        "Outlet": "",
                        "Creative": "",
                        "VariableName": "Price",
                        "VariableValue": 100 + np.random.normal(0, 5),
                    })
                    
                    # Distribution (ACV)
                    records.append({
                        "Period": date.strftime("%Y-%m-%d"),
                        "Geography": "",
                        "Product": "",
                        "Campaign": "",
                        "Outlet": "",
                        "Creative": "",
                        "VariableName": "Distribution",
                        "VariableValue": 0.85 + np.random.normal(0, 0.02),
                    })
    
    return pd.DataFrame(records)


# =============================================================================
# Example 1: Simple National Model
# =============================================================================

def example_national_model():
    """Simple national-level MMM with default settings."""
    
    print("=" * 60)
    print("Example 1: National Model")
    print("=" * 60)
    
    # Generate synthetic data
    mff_data = generate_synthetic_mff(n_weeks=156)
    
    # Simple config using factory function
    config = create_simple_mff_config(
        kpi_name="Sales",
        media_names=["TV", "Digital", "Radio"],
        control_names=["Price", "Distribution"],
        kpi_dimensions=[DimensionType.PERIOD],
        multiplicative=False,
    )
    
    # Load data
    panel = load_mff(mff_data, config)
    
    print(panel.summary())
    print("\nSpend shares:")
    print(panel.compute_spend_shares())
    
    return panel


# =============================================================================
# Example 2: Geo-Level Panel Model
# =============================================================================

def example_geo_panel_model():
    """Geo-level panel model with national media disaggregation."""
    
    print("\n" + "=" * 60)
    print("Example 2: Geo Panel Model")
    print("=" * 60)
    
    # Generate synthetic data with geos
    mff_data = generate_synthetic_mff(
        n_weeks=156,
        geographies=["East", "West", "Central"],
    )
    
    # Build config manually for more control
    config = MFFConfig(
        kpi=KPIConfig(
            name="Sales",
            dimensions=[DimensionType.PERIOD, DimensionType.GEOGRAPHY],
            log_transform=False,
        ),
        media_channels=[
            MediaChannelConfig(
                name="TV",
                dimensions=[DimensionType.PERIOD],  # National
                adstock=AdstockConfig.geometric(l_max=8),
                saturation=SaturationConfig.hill(),
            ),
            MediaChannelConfig(
                name="Digital",
                dimensions=[DimensionType.PERIOD],  # National
                adstock=AdstockConfig.geometric(l_max=4),
                saturation=SaturationConfig.hill(),
            ),
            MediaChannelConfig(
                name="Radio",
                dimensions=[DimensionType.PERIOD],  # National  
                adstock=AdstockConfig.geometric(l_max=6),
                saturation=SaturationConfig.hill(),
            ),
        ],
        controls=[
            ControlVariableConfig(
                name="Price",
                dimensions=[DimensionType.PERIOD],
                allow_negative=True,
            ),
            ControlVariableConfig(
                name="Distribution", 
                dimensions=[DimensionType.PERIOD],
            ),
        ],
        alignment=DimensionAlignmentConfig(
            geo_allocation=AllocationMethod.SALES,  # Use sales to allocate
        ),
    )
    
    # Load with population weights (could also compute from data)
    geo_weights = {"East": 0.4, "West": 0.35, "Central": 0.25}
    
    panel = load_mff(mff_data, config, geo_weights=geo_weights)
    
    print(panel.summary())
    print("\nPyMC coordinates:")
    print(panel.coords.to_pymc_coords())
    
    return panel


# =============================================================================
# Example 3: Multi-Product Hierarchical Model
# =============================================================================

def example_product_hierarchical_model():
    """Multi-product model with hierarchical media effects."""
    
    print("\n" + "=" * 60)
    print("Example 3: Product Hierarchical Model")
    print("=" * 60)
    
    # Generate data with products
    mff_data = generate_synthetic_mff(
        n_weeks=156,
        geographies=["East", "West"],
        products=["Premium", "Standard", "Budget"],
    )
    
    config = MFFConfig(
        kpi=KPIConfig(
            name="Sales",
            dimensions=[
                DimensionType.PERIOD,
                DimensionType.GEOGRAPHY,
                DimensionType.PRODUCT,
            ],
        ),
        media_channels=[
            MediaChannelConfig(
                name="TV",
                dimensions=[DimensionType.PERIOD],
                adstock=AdstockConfig.geometric(l_max=8),
                saturation=SaturationConfig.hill(
                    # Custom priors
                    kappa_prior=PriorConfig.gamma(alpha=3, beta=2),
                    slope_prior=PriorConfig.half_normal(sigma=2.0),
                ),
            ),
            MediaChannelConfig(
                name="Digital",
                dimensions=[DimensionType.PERIOD],
                adstock=AdstockConfig.geometric(l_max=4),
                saturation=SaturationConfig.hill(),
            ),
            MediaChannelConfig(
                name="Radio",
                dimensions=[DimensionType.PERIOD],
                adstock=AdstockConfig.geometric(l_max=6),
                saturation=SaturationConfig.hill(),
            ),
        ],
        controls=[
            ControlVariableConfig(name="Price", dimensions=[DimensionType.PERIOD]),
            ControlVariableConfig(name="Distribution", dimensions=[DimensionType.PERIOD]),
        ],
        alignment=DimensionAlignmentConfig(
            geo_allocation=AllocationMethod.SALES,
            product_allocation=AllocationMethod.SALES,
        ),
    )
    
    panel = load_mff(mff_data, config)
    
    print(panel.summary())
    print(f"\nPanel shape: {panel.n_obs} observations")
    print(f"  {panel.coords.n_periods} periods × {panel.coords.n_geos} geos × {panel.coords.n_products} products")
    
    return panel


# =============================================================================
# Example 4: Social Platform Splits with Hierarchy
# =============================================================================

def example_social_splits():
    """Social media with platform-level splits and hierarchical pooling."""
    
    print("\n" + "=" * 60)
    print("Example 4: Social Platform Splits")
    print("=" * 60)
    
    # Generate data with social splits
    mff_data = generate_synthetic_mff(
        n_weeks=156,
        include_social_splits=True,
    )
    
    # Create social platform configs
    social_configs = create_social_platform_configs(
        platforms=["Meta", "Snapchat", "Twitter"],
        parent_name="Social",
        adstock_lmax=4,
    )
    
    config = MFFConfig(
        kpi=KPIConfig(
            name="Sales",
            dimensions=[DimensionType.PERIOD],
        ),
        media_channels=[
            create_national_media_config("TV", adstock_lmax=8),
            create_national_media_config("Digital", adstock_lmax=4),
            create_national_media_config("Radio", adstock_lmax=6),
        ] + social_configs,
        controls=[
            ControlVariableConfig(name="Price", dimensions=[DimensionType.PERIOD]),
            ControlVariableConfig(name="Distribution", dimensions=[DimensionType.PERIOD]),
        ],
    )
    
    # Note: For social splits, we need special handling in the loader
    # The current loader aggregates by VariableName, so each platform 
    # would need its own VariableName or we handle Outlet dimension
    
    print("Config created with hierarchical social structure:")
    print(f"  Media channels: {config.media_names}")
    print(f"  Hierarchical groups: {config.get_hierarchical_media_groups()}")
    
    return config


# =============================================================================
# Example 5: Full Model Configuration
# =============================================================================

def example_full_model_config():
    """Complete model configuration including inference settings."""
    
    print("\n" + "=" * 60)
    print("Example 5: Full Model Configuration")
    print("=" * 60)
    
    # MFF config
    mff_config = MFFConfig(
        kpi=KPIConfig(
            name="Sales",
            dimensions=[DimensionType.PERIOD, DimensionType.GEOGRAPHY],
            log_transform=False,
        ),
        media_channels=[
            MediaChannelConfig(
                name="TV",
                dimensions=[DimensionType.PERIOD],
                adstock=AdstockConfig.geometric(l_max=8),
                saturation=SaturationConfig.hill(),
                coefficient_prior=PriorConfig.half_normal(sigma=2.0),
            ),
            MediaChannelConfig(
                name="Digital",
                dimensions=[DimensionType.PERIOD],
                adstock=AdstockConfig.geometric(l_max=4),
                saturation=SaturationConfig.hill(),
            ),
        ],
        controls=[
            ControlVariableConfig(
                name="Price",
                dimensions=[DimensionType.PERIOD],
                allow_negative=True,
                use_shrinkage=False,
            ),
        ],
    )
    
    # Model config
    model_config = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_NUMPYRO,
        n_chains=4,
        n_draws=1000,
        n_tune=1000,
        target_accept=0.9,
    )
    
    print("MFF Configuration:")
    print(f"  KPI: {mff_config.kpi.name} with dims {mff_config.kpi.dim_names}")
    print(f"  Media channels: {mff_config.media_names}")
    print(f"  Controls: {mff_config.control_names}")
    
    print("\nModel Configuration:")
    print(f"  Method: {model_config.inference_method.value}")
    print(f"  Chains: {model_config.n_chains}")
    print(f"  Draws: {model_config.n_draws}")
    print(f"  Use NumPyro: {model_config.use_numpyro}")
    
    return mff_config, model_config


# =============================================================================
# Run examples
# =============================================================================

if __name__ == "__main__":
    # Run all examples
    panel1 = example_national_model()
    panel2 = example_geo_panel_model()
    panel3 = example_product_hierarchical_model()
    config4 = example_social_splits()
    configs5 = example_full_model_config()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)