"""
Examples demonstrating the BayesianMMM model class.

Shows:
- Basic model fitting
- Trend specifications
- Geo-level hierarchical models
- Prior predictive checks
- Model diagnostics
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Framework imports
from mmm_framework import (
    # Builders
    MFFConfigBuilder,
    KPIConfigBuilder,
    MediaChannelConfigBuilder,
    ControlVariableConfigBuilder,
    DimensionAlignmentConfigBuilder,
    HierarchicalConfigBuilder,
    SeasonalityConfigBuilder,
    ModelConfigBuilder,
    # Model
    BayesianMMM,
    TrendConfig,
    TrendType,
    # Data
    load_mff,
)

# Reuse synthetic data generator
from ex_config import generate_synthetic_mff


def example_basic_model():
    """Basic national-level model fitting."""
    
    print("=" * 60)
    print("Example 1: Basic National Model")
    print("=" * 60)
    
    # Generate data
    np.random.seed(42)
    mff_data = generate_synthetic_mff(n_weeks=52)
    
    # Build config
    mff_config = (MFFConfigBuilder()
        .with_kpi_name("Sales")
        .add_national_media("TV", adstock_lmax=8)
        .add_national_media("Digital", adstock_lmax=4)
        .add_price_control()
        .build())
    
    panel = load_mff(mff_data, mff_config)
    print(f"Loaded panel: {panel.n_obs} obs, {panel.n_channels} channels")
    
    # Model config
    model_config = (ModelConfigBuilder()
        .bayesian_pymc()
        .with_chains(2)
        .with_draws(500)
        .with_tune(500)
        .with_target_accept(0.9)
        .build())
    
    # Create and fit model
    mmm = BayesianMMM(panel, model_config, TrendConfig())
    
    print(f"\nModel variables: {[v.name for v in mmm.model.free_RVs]}")
    print(f"Data standardization: y_mean={mmm.y_mean:.1f}, y_std={mmm.y_std:.1f}")
    
    print("\nFitting model...")
    results = mmm.fit(random_seed=42)
    
    print(f"\n=== DIAGNOSTICS ===")
    print(f"Divergences: {results.diagnostics['divergences']}")
    print(f"R-hat max: {results.diagnostics['rhat_max']:.4f}")
    print(f"ESS bulk min: {results.diagnostics['ess_bulk_min']:.0f}")
    
    print(f"\n=== POSTERIOR SUMMARY ===")
    summary = results.summary(['beta_TV', 'beta_Digital', 'sigma'])
    print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat']])
    
    return results


def example_geo_panel():
    """Geo-level panel model with hierarchical pooling."""
    
    print("\n" + "=" * 60)
    print("Example 2: Geo Panel with Hierarchical Pooling")
    print("=" * 60)
    
    # Generate geo data
    np.random.seed(42)
    mff_data = generate_synthetic_mff(
        n_weeks=104,
        geographies=["East", "West", "Central"],
    )
    
    # Config with geo KPI and alignment
    mff_config = (MFFConfigBuilder()
        .with_kpi_builder(KPIConfigBuilder("Sales").by_geo())
        .add_national_media("TV", adstock_lmax=8)
        .add_national_media("Digital", adstock_lmax=4)
        .add_price_control()
        .with_alignment_builder(DimensionAlignmentConfigBuilder().geo_equal())
        .build())
    
    panel = load_mff(mff_data, mff_config)
    print(f"Panel: {panel.n_obs} obs, {panel.coords.n_geos} geos")
    
    # Model with hierarchical pooling
    model_config = (ModelConfigBuilder()
        .bayesian_pymc()
        .with_chains(2)
        .with_draws(500)
        .with_tune(500)
        .with_target_accept(0.95)
        .with_hierarchical_builder(
            HierarchicalConfigBuilder()
            .enabled()
            .pool_across_geo()
        )
        .build())
    
    mmm = BayesianMMM(panel, model_config, TrendConfig())
    
    print("\nFitting hierarchical model...")
    results = mmm.fit(random_seed=42)
    
    print(f"\n=== DIAGNOSTICS ===")
    print(f"Divergences: {results.diagnostics['divergences']}")
    print(f"R-hat max: {results.diagnostics['rhat_max']:.4f}")
    
    print(f"\n=== KEY PARAMETERS ===")
    summary = results.summary(['beta_TV', 'beta_Digital', 'geo_sigma', 'sigma'])
    print(summary[['mean', 'sd', 'r_hat']])
    
    print(f"\n=== CHANNEL CONTRIBUTIONS BY GEO ===")
    if results.channel_contributions is not None:
        contrib = results.channel_contributions.groupby(level='Geography').sum()
        print(contrib)
    
    return results


def example_model_structure():
    """Show model structure without fitting."""
    
    print("\n" + "=" * 60)
    print("Example 3: Model Structure Inspection")
    print("=" * 60)
    
    np.random.seed(42)
    mff_data = generate_synthetic_mff(n_weeks=52)
    
    mff_config = (MFFConfigBuilder()
        .with_kpi_name("Sales")
        .add_national_media("TV", adstock_lmax=8)
        .add_national_media("Digital", adstock_lmax=4)
        .add_national_media("Radio", adstock_lmax=6)
        .add_price_control()
        .add_distribution_control()
        .build())
    
    panel = load_mff(mff_data, mff_config)
    
    model_config = (ModelConfigBuilder()
        .bayesian_pymc()
        .with_chains(4)
        .with_draws(1000)
        .build())
    
    mmm = BayesianMMM(panel, model_config, TrendConfig(type=TrendType.LINEAR))
    
    print("Model free random variables:")
    for var in mmm.model.free_RVs:
        print(f"  {var.name}: {var.type}")
    
    print("\nModel deterministics:")
    for var in mmm.model.deterministics:
        print(f"  {var.name}")
    
    print("\nData shapes:")
    print(f"  y: {mmm.y.shape}")
    print(f"  X_media (scaled): {mmm.X_media_adstocked[0.0].shape}")
    print(f"  X_controls: {mmm.X_controls.shape if mmm.X_controls is not None else None}")
    
    return mmm


def example_prior_predictive():
    """Demonstrate prior predictive checks."""
    
    print("\n" + "=" * 60)
    print("Example 4: Prior Predictive Check")
    print("=" * 60)
    
    np.random.seed(42)
    mff_data = generate_synthetic_mff(n_weeks=52)
    
    mff_config = (MFFConfigBuilder()
        .with_kpi_name("Sales")
        .add_national_media("TV", adstock_lmax=8)
        .add_national_media("Digital", adstock_lmax=4)
        .build())
    
    panel = load_mff(mff_data, mff_config)
    model_config = (ModelConfigBuilder().bayesian_pymc().build())
    
    mmm = BayesianMMM(panel, model_config, TrendConfig())
    
    print("Sampling from prior predictive distribution...")
    prior = mmm.sample_prior_predictive(samples=200)
    
    # Get prior predictive samples
    y_prior = prior.prior_predictive["y_obs"].values.flatten()
    
    print(f"\nPrior predictive y statistics:")
    print(f"  Mean: {y_prior.mean():.2f}")
    print(f"  Std: {y_prior.std():.2f}")
    print(f"  Range: [{y_prior.min():.2f}, {y_prior.max():.2f}]")
    
    print(f"\nActual y (standardized) statistics:")
    print(f"  Mean: {mmm.y.mean():.2f}")
    print(f"  Std: {mmm.y.std():.2f}")
    print(f"  Range: [{mmm.y.min():.2f}, {mmm.y.max():.2f}]")
    
    # Check if priors are reasonable
    if abs(y_prior.mean()) < 5 and y_prior.std() < 10:
        print("\n✓ Prior predictive looks reasonable for standardized data")
    else:
        print("\n⚠ Prior predictive may need adjustment")
    
    return prior


def example_full_workflow():
    """Complete workflow from data to analysis."""
    
    print("\n" + "=" * 60)
    print("Example 5: Full Workflow")
    print("=" * 60)
    
    # 1. Generate data
    print("\n1. Generate Data")
    np.random.seed(42)
    mff_data = generate_synthetic_mff(n_weeks=104)
    print(f"   {len(mff_data)} MFF records")
    
    # 2. Build config
    print("\n2. Build Configuration")
    mff_config = (MFFConfigBuilder()
        .with_kpi_name("Sales")
        .add_national_media("TV", adstock_lmax=8)
        .add_national_media("Digital", adstock_lmax=4)
        .add_national_media("Radio", adstock_lmax=6)
        .add_price_control()
        .add_distribution_control()
        .build())
    print(f"   Media: {mff_config.media_names}")
    print(f"   Controls: {mff_config.control_names}")
    
    # 3. Load panel
    print("\n3. Load Panel Data")
    panel = load_mff(mff_data, mff_config)
    print(f"   Shape: {panel.n_obs} obs, {panel.n_channels} channels")
    
    # 4. Configure model
    print("\n4. Configure Model")
    model_config = (ModelConfigBuilder()
        .bayesian_pymc()
        .with_chains(4)
        .with_draws(1000)
        .with_tune(1000)
        .with_target_accept(0.95)
        .build())
    print(f"   Chains: {model_config.n_chains}")
    print(f"   Draws: {model_config.n_draws}")
    
    # 5. Build and fit
    print("\n5. Build and Fit Model")
    mmm = BayesianMMM(panel, model_config, TrendConfig())
    results = mmm.fit(random_seed=42)
    
    # 6. Diagnostics
    print("\n6. Diagnostics")
    print(f"   Divergences: {results.diagnostics['divergences']}")
    print(f"   R-hat max: {results.diagnostics['rhat_max']:.4f}")
    print(f"   ESS bulk min: {results.diagnostics['ess_bulk_min']:.0f}")
    
    # 7. Results
    print("\n7. Results")
    print("\n   Beta coefficients:")
    summary = results.summary(['beta_TV', 'beta_Digital', 'beta_Radio'])
    print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%']])
    
    print("\n   Total channel contributions:")
    if results.channel_contributions is not None:
        print(results.channel_contributions.sum())
    
    return results


# =============================================================================
# Run examples
# =============================================================================

if __name__ == "__main__":
    print("BayesianMMM Model Examples")
    print("=" * 60)
    
    try:
        import pymc as pm
        import arviz as az
        print(f"PyMC version: {pm.__version__}")
        print(f"ArviZ version: {az.__version__}")
    except ImportError as e:
        print(f"Error: {e}")
        print("Install with: pip install pymc arviz")
        exit(1)
    
    # Run examples
    results1 = example_basic_model()
    results2 = example_geo_panel()
    mmm3 = example_model_structure()
    prior4 = example_prior_predictive()
    results5 = example_full_workflow()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)