"""
Complete MMM Workflow Example
=============================

A comprehensive example demonstrating the full Bayesian Marketing Mix Modeling
workflow using mmm_framework's built-in features:

1. Generate or load MFF (Master Flat File) data
2. Configure the model using fluent builder patterns
3. Fit a BayesianMMM with proper uncertainty quantification
4. Compute all reporting metrics with credible intervals
5. Generate a comprehensive HTML report

This example emphasizes honest uncertainty communication throughout,
avoiding specification shopping by pre-specifying all analytical choices.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
logger.enable("mmm_framework")
# =============================================================================
# Framework Imports
# =============================================================================

from mmm_framework import (
    # Builders for configuration
    MFFConfigBuilder,
    KPIConfigBuilder,
    MediaChannelConfigBuilder,
    ControlVariableConfigBuilder,
    DimensionAlignmentConfigBuilder,
    HierarchicalConfigBuilder,
    SeasonalityConfigBuilder,
    ModelConfigBuilder,
    TrendConfigBuilder,
    PriorConfigBuilder,
    
    # Core model classes
    BayesianMMM,
    TrendConfig,
    TrendType,
    
    # Data loading
    load_mff,
    mff_from_wide_format,
)

# Reporting imports
from mmm_framework.reporting import (
    MMMReportGenerator,
    ReportBuilder,
    ReportConfig,
    SectionConfig,
    ColorScheme,
    ColorPalette,
)

from mmm_framework.reporting.helpers import (
    # ROI computation
    compute_roi_with_uncertainty,
    compute_marginal_roi,
    
    # Prior/posterior analysis
    get_prior_posterior_comparison,
    compute_shrinkage_summary,
    
    # Saturation analysis
    compute_saturation_curves_with_uncertainty,
    
    # Adstock analysis
    compute_adstock_weights,
    
    # Decomposition
    compute_component_decomposition,
    compute_decomposition_waterfall,
    
    # Full summary
    generate_model_summary,
)


# =============================================================================
# Step 1: Generate Synthetic MFF Data
# =============================================================================

def generate_synthetic_mff(
    n_weeks: int = 104,
    geographies: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic MFF (Master Flat File) data for demonstration.
    
    The MFF format is a normalized long-format structure with 8 columns:
    - Period: Time period identifier (date)
    - Geography: Geographic unit (DMA, region, etc.)
    - Product: Product or brand identifier
    - Campaign: Campaign or flight identifier
    - Outlet: Media outlet or channel
    - Creative: Creative execution identifier
    - VariableName: Name of the metric (e.g., "Sales", "TV_Spend")
    - VariableValue: Numeric value for that metric
    
    Parameters
    ----------
    n_weeks : int
        Number of weeks of data to generate
    geographies : list[str], optional
        List of geographic regions. If None, generates national-level data
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        MFF-formatted data with correct column structure
    """
    np.random.seed(seed)
    
    # Date range
    start_date = datetime(2022, 1, 3)  # First Monday of 2022
    dates = pd.date_range(start_date, periods=n_weeks, freq='W-MON')
    
    # Handle geographies
    geos = geographies if geographies else None
    is_national = geos is None
    
    records = []
    
    def create_base_record(date, geo=None):
        """Create base record with all required MFF columns."""
        return {
            "Period": date.strftime("%Y-%m-%d"),
            "Geography": geo if geo else None,
            "Product": None,
            "Campaign": None,
            "Outlet": None,
            "Creative": None,
        }
    
    if is_national:
        # National-level data (no geography dimension)
        for i, date in enumerate(dates):
            week_idx = i
            
            # Seasonal pattern
            week_of_year = date.isocalendar()[1]
            seasonality = 1 + 0.3 * np.sin(2 * np.pi * week_of_year / 52)
            
            # Trend (slight growth)
            trend = 1 + 0.001 * week_idx
            
            # Media spend with realistic patterns
            tv_spend = 80000 * (1.5 if week_idx % 8 < 4 else 0.5) * np.random.lognormal(0, 0.2)
            digital_spend = 50000 * (1 + 0.002 * week_idx) * np.random.lognormal(0, 0.15)
            social_spend = 30000 * seasonality * np.random.lognormal(0, 0.2)
            radio_spend = 15000 * np.random.lognormal(0, 0.3) if np.random.random() > 0.2 else 0
            
            # Controls
            price_index = 100 + 5 * np.sin(2 * np.pi * week_idx / 26) + np.random.normal(0, 2)
            distribution = 85 + 10 * (1 - np.exp(-week_idx / 50)) + np.random.normal(0, 2)
            
            # Generate sales
            base_sales = 800000
            tv_contribution = 0.8 * np.sqrt(tv_spend / 1000)
            digital_contribution = 1.2 * np.sqrt(digital_spend / 1000)
            social_contribution = 0.6 * np.sqrt(social_spend / 1000)
            radio_contribution = 0.4 * np.sqrt(radio_spend / 1000) if radio_spend > 0 else 0
            
            media_lift = (tv_contribution + digital_contribution + 
                         social_contribution + radio_contribution) * 1000
            
            price_effect = -2000 * (price_index - 100)
            distribution_effect = 1500 * (distribution - 85)
            
            sales = (base_sales * trend * seasonality + media_lift + 
                    price_effect + distribution_effect + np.random.normal(0, 50000))
            sales = max(sales, 0)
            
            # Create records with correct MFF structure
            base = create_base_record(date)
            
            # KPI
            records.append({**base, "VariableName": "Sales", "VariableValue": sales})
            
            # Media
            records.append({**base, "VariableName": "TV", "VariableValue": tv_spend})
            records.append({**base, "VariableName": "Digital", "VariableValue": digital_spend})
            records.append({**base, "VariableName": "Paid_Social", "VariableValue": social_spend})
            records.append({**base, "VariableName": "Radio", "VariableValue": radio_spend})
            
            # Controls
            records.append({**base, "VariableName": "Price_Index", "VariableValue": price_index})
            records.append({**base, "VariableName": "Distribution", "VariableValue": distribution})
    
    else:
        # Geo-level data
        for i, date in enumerate(dates):
            week_idx = i
            
            # Seasonal pattern
            week_of_year = date.isocalendar()[1]
            seasonality = 1 + 0.3 * np.sin(2 * np.pi * week_of_year / 52)
            
            # Trend
            trend = 1 + 0.001 * week_idx
            
            # National media spend (same for all geos - will be allocated)
            tv_spend = 80000 * (1.5 if week_idx % 8 < 4 else 0.5) * np.random.lognormal(0, 0.2)
            digital_spend = 50000 * (1 + 0.002 * week_idx) * np.random.lognormal(0, 0.15)
            social_spend = 30000 * seasonality * np.random.lognormal(0, 0.2)
            
            # Add national media once (Geography = None for national)
            base_national = create_base_record(date, geo=None)
            records.append({**base_national, "VariableName": "TV", "VariableValue": tv_spend})
            records.append({**base_national, "VariableName": "Digital", "VariableValue": digital_spend})
            records.append({**base_national, "VariableName": "Paid_Social", "VariableValue": social_spend})
            
            # Geo-level KPI and controls
            for geo in geos:
                geo_multiplier = {
                    "East": 0.35,
                    "West": 0.30,
                    "Central": 0.25,
                    "South": 0.10,
                }.get(geo, 1.0 / len(geos))
                
                # Controls at geo level
                price_index = 100 + 5 * np.sin(2 * np.pi * week_idx / 26) + np.random.normal(0, 2)
                distribution = 85 + 10 * (1 - np.exp(-week_idx / 50)) + np.random.normal(0, 2)
                
                # Sales at geo level
                base_sales = 800000 * geo_multiplier
                tv_contribution = 0.8 * np.sqrt(tv_spend * geo_multiplier / 1000)
                digital_contribution = 1.2 * np.sqrt(digital_spend * geo_multiplier / 1000)
                social_contribution = 0.6 * np.sqrt(social_spend * geo_multiplier / 1000)
                
                media_lift = (tv_contribution + digital_contribution + social_contribution) * 1000
                price_effect = -2000 * (price_index - 100) * geo_multiplier
                distribution_effect = 1500 * (distribution - 85) * geo_multiplier
                
                sales = (base_sales * trend * seasonality + media_lift + 
                        price_effect + distribution_effect + np.random.normal(0, 20000))
                sales = max(sales, 0)
                
                base_geo = create_base_record(date, geo=geo)
                
                # KPI at geo level
                records.append({**base_geo, "VariableName": "Sales", "VariableValue": sales})
                
                # Controls at geo level
                records.append({**base_geo, "VariableName": "Price_Index", "VariableValue": price_index})
                records.append({**base_geo, "VariableName": "Distribution", "VariableValue": distribution})
    
    df = pd.DataFrame(records)
    
    # Ensure correct column order for MFF format
    column_order = ["Period", "Geography", "Product", "Campaign", "Outlet", "Creative", "VariableName", "VariableValue"]
    return df[column_order]


# =============================================================================
# Step 2: Configure the Model (Pre-Specification)
# =============================================================================

def create_mff_config_national() -> 'MFFConfig':
    """
    Create MFF configuration for national-level model.
    
    All analytical choices are pre-specified here, NOT adjusted based on
    initial results. This is crucial for maintaining valid statistical inference.
    
    Returns
    -------
    MFFConfig
        Complete configuration for data loading
    """
    config = (
        MFFConfigBuilder()
        
        # KPI Configuration
        .with_kpi_builder(
            KPIConfigBuilder("Sales")
            .national()           # National-level outcome
            .additive()           # Additive model
            .with_display_name("Revenue")
        )
        
        # Media Channel Configurations
        # Each channel specifies adstock and saturation priors BEFORE seeing data
        .add_media_builder(
            MediaChannelConfigBuilder("TV")
            .national()
            .with_geometric_adstock(l_max=8)  # Up to 8 weeks carryover
            .with_hill_saturation()
            .with_display_name("Television")
            .with_coefficient_prior(
                PriorConfigBuilder().half_normal(sigma=2.0).build()
            )
        )
        .add_media_builder(
            MediaChannelConfigBuilder("Digital")
            .national()
            .with_geometric_adstock(l_max=4)  # Shorter carryover
            .with_hill_saturation()
            .with_display_name("Digital Marketing")
        )
        .add_media_builder(
            MediaChannelConfigBuilder("Paid_Social")
            .national()
            .with_geometric_adstock(l_max=4)
            .with_hill_saturation()
            .with_display_name("Paid Social")
        )
        .add_media_builder(
            MediaChannelConfigBuilder("Radio")
            .national()
            .with_geometric_adstock(l_max=6)
            .with_hill_saturation()
            .with_display_name("Radio")
        )
        
        # Control Variables
        # Controls are CONFOUNDERS - never subject to variable selection
        .add_control_builder(
            ControlVariableConfigBuilder("Price_Index")
            .national()
            .allow_negative()  # Price increases should decrease sales
            .with_coefficient_prior(
                PriorConfigBuilder().normal(mu=-0.5, sigma=1.0).build()
            )
            .with_display_name("Price Index")
        )
        .add_control_builder(
            ControlVariableConfigBuilder("Distribution")
            .national()
            .with_coefficient_prior(
                PriorConfigBuilder().half_normal(sigma=1.0).build()
            )
            .with_display_name("Distribution %")
        )
        
        # Time settings
        .weekly()
        .with_date_format("%Y-%m-%d")
        
        .build()
    )
    
    return config


def create_mff_config_geo(geo_names: list[str]) -> 'MFFConfig':
    """
    Create MFF configuration for geo-level panel model.
    
    Parameters
    ----------
    geo_names : list[str]
        List of geography names
        
    Returns
    -------
    MFFConfig
        Configuration with geo-level KPI and alignment settings
    """
    config = (
        MFFConfigBuilder()
        
        # KPI at geo level
        .with_kpi_builder(
            KPIConfigBuilder("Sales")
            .by_geo()             # Geo-level outcome
            .additive()
        )
        
        # National media allocated to geos
        .add_media_builder(
            MediaChannelConfigBuilder("TV")
            .national()           # Spend is national
            .with_geometric_adstock(l_max=8)
            .with_hill_saturation()
        )
        .add_media_builder(
            MediaChannelConfigBuilder("Digital")
            .national()
            .with_geometric_adstock(l_max=4)
            .with_hill_saturation()
        )
        .add_media_builder(
            MediaChannelConfigBuilder("Paid_Social")
            .national()
            .with_geometric_adstock(l_max=4)
            .with_hill_saturation()
        )
        
        # Controls at geo level
        .add_control_builder(
            ControlVariableConfigBuilder("Price_Index")
            .by_geo()
            .allow_negative()
        )
        .add_control_builder(
            ControlVariableConfigBuilder("Distribution")
            .by_geo()
        )
        
        # Allocation method for national media to geos
        .with_alignment_builder(
            DimensionAlignmentConfigBuilder()
            .geo_by_population()  # Allocate by population weights
        )
        
        .weekly()
        .build()
    )
    
    return config


def create_model_config(
    use_numpyro: bool = False,
    hierarchical: bool = False,
    n_chains: int = 4,
    n_draws: int = 1000,
    n_tune: int = 1000,
) -> 'ModelConfig':
    """
    Create model configuration with sampling parameters.
    
    Parameters
    ----------
    use_numpyro : bool
        Use NumPyro backend (faster) vs PyMC
    hierarchical : bool
        Enable hierarchical pooling across geos
    n_chains : int
        Number of MCMC chains
    n_draws : int
        Number of posterior draws per chain
    n_tune : int
        Number of tuning steps
        
    Returns
    -------
    ModelConfig
        Complete model configuration
    """
    builder = ModelConfigBuilder()
    
    # Backend selection
    if use_numpyro:
        builder.bayesian_numpyro()
    else:
        builder.bayesian_pymc()
    
    # Sampling parameters
    builder.with_chains(n_chains)
    builder.with_draws(n_draws)
    builder.with_tune(n_tune)
    builder.with_target_accept(0.95)  # High target accept for robustness
    
    # Seasonality
    builder.with_seasonality_builder(
        SeasonalityConfigBuilder()
        .with_yearly(order=4)     # Fourier terms for yearly seasonality
    )
    
    # Hierarchical pooling (for geo models)
    if hierarchical:
        builder.with_hierarchical_builder(
            HierarchicalConfigBuilder()
            .enabled()
            .pool_across_geo()
        )
    
    return builder.build()


def create_trend_config(
    trend_type: str = "linear",
) -> TrendConfig:
    """
    Create trend configuration.
    
    Parameters
    ----------
    trend_type : str
        One of "none", "linear", "piecewise", "gp" (Gaussian Process)
        
    Returns
    -------
    TrendConfig
        Trend configuration
    """
    builder = TrendConfigBuilder()
    
    if trend_type == "none":
        builder.none()
    elif trend_type == "linear":
        builder.linear()
    elif trend_type == "piecewise":
        builder.piecewise(n_changepoints=5)
    elif trend_type == "gp":
        builder.gaussian_process()
        builder.with_gp_lengthscale(mu=0.3, sigma=0.2)
        builder.with_gp_n_basis(20)
    else:
        builder.linear()  # Default
    
    return builder.build()


# =============================================================================
# Step 3: Fit the Model
# =============================================================================

def fit_model(
    panel,
    model_config,
    trend_config,
    random_seed: int = 42,
) -> tuple['BayesianMMM', 'FitResults']:
    """
    Fit the Bayesian MMM model.
    
    Parameters
    ----------
    panel : Panel
        Loaded panel data
    model_config : ModelConfig
        Model configuration
    trend_config : TrendConfig
        Trend configuration
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    tuple[BayesianMMM, FitResults]
        Fitted model and results
    """
    print("\n" + "=" * 60)
    print("FITTING BAYESIAN MMM")
    print("=" * 60)
    
    # Create model
    mmm = BayesianMMM(panel, model_config, trend_config)
    
    print(f"\nModel Structure:")
    print(f"  Observations: {mmm.n_obs}")
    print(f"  Channels: {mmm.n_channels}")
    print(f"  Free parameters: {len(mmm.model.free_RVs)}")
    print(f"  Parameters: {[v.name for v in mmm.model.free_RVs]}")
    
    # Data scaling info
    print(f"\nData Standardization:")
    print(f"  y_mean: {mmm.y_mean:,.0f}")
    print(f"  y_std: {mmm.y_std:,.0f}")
    
    # Fit model
    print(f"\nStarting MCMC sampling...")
    print(f"  Chains: {model_config.n_chains}")
    print(f"  Draws: {model_config.n_draws}")
    print(f"  Tune: {model_config.n_tune}")
    
    results = mmm.fit(random_seed=random_seed)
    
    print(f"\n✓ Sampling complete!")
    
    return mmm, results


# =============================================================================
# Step 4: Compute All Report Metrics
# =============================================================================

def compute_all_metrics(
    mmm: 'BayesianMMM',
    results: 'FitResults',
    hdi_prob: float = 0.94,
) -> dict:
    """
    Compute all metrics needed for comprehensive reporting.
    
    This function computes:
    - MCMC diagnostics (convergence checks)
    - ROI with credible intervals and probability of profitability
    - Prior vs posterior comparisons (data informativeness)
    - Saturation curves with uncertainty bands
    - Adstock decay weights
    - Component decomposition
    - Model fit statistics
    
    All metrics include proper uncertainty quantification.
    
    Parameters
    ----------
    mmm : BayesianMMM
        Fitted model
    results : FitResults
        Fit results containing trace and diagnostics
    hdi_prob : float
        Probability for HDI (Highest Density Interval)
        
    Returns
    -------
    dict
        Dictionary containing all computed metrics
    """
    print("\n" + "=" * 60)
    print("COMPUTING REPORT METRICS")
    print("=" * 60)
    
    metrics = {}
    
    # -------------------------------------------------------------------------
    # 1. MCMC Diagnostics
    # -------------------------------------------------------------------------
    print("\n[1/7] MCMC Diagnostics...")
    
    metrics['diagnostics'] = {
        'divergences': results.diagnostics.get('divergences', 0),
        'rhat_max': results.diagnostics.get('rhat_max', 1.0),
        'ess_bulk_min': results.diagnostics.get('ess_bulk_min', 0),
        'ess_tail_min': results.diagnostics.get('ess_tail_min', 0),
    }
    
    # Convergence assessment
    converged = (
        metrics['diagnostics']['divergences'] == 0 and
        metrics['diagnostics']['rhat_max'] < 1.01 and
        metrics['diagnostics']['ess_bulk_min'] > 400
    )
    metrics['diagnostics']['converged'] = converged
    
    print(f"  Divergences: {metrics['diagnostics']['divergences']}")
    print(f"  R-hat max: {metrics['diagnostics']['rhat_max']:.4f}")
    print(f"  ESS bulk min: {metrics['diagnostics']['ess_bulk_min']:.0f}")
    print(f"  Converged: {'✓' if converged else '✗'}")
    
    if not converged:
        print("\n  ⚠️  WARNING: Model may not have converged properly!")
        print("     Consider: more tuning steps, higher target_accept, or simpler model")
    
    # -------------------------------------------------------------------------
    # Helper: Extract spend data robustly
    # -------------------------------------------------------------------------
    def _extract_spend_data(model) -> dict:
        """Extract total spend per channel from model's panel data."""
        spend = {}
        channels = model.channel_names
        
        # Try to get X_media from panel
        if hasattr(model, 'panel') and model.panel is not None:
            panel = model.panel
            
            # If X_media is a DataFrame
            if hasattr(panel, 'X_media'):
                X_media = panel.X_media
                if hasattr(X_media, 'columns'):
                    # It's a DataFrame
                    for ch in channels:
                        if ch in X_media.columns:
                            spend[ch] = float(X_media[ch].sum())
                elif hasattr(X_media, 'values'):
                    # It's a DataFrame, access by position
                    for i, ch in enumerate(channels):
                        if i < X_media.shape[1]:
                            spend[ch] = float(X_media.iloc[:, i].sum())
                else:
                    # It's a numpy array
                    for i, ch in enumerate(channels):
                        if i < X_media.shape[1]:
                            spend[ch] = float(X_media[:, i].sum())
        
        # Fallback: try model's X_media_raw
        if not spend and hasattr(model, 'X_media_raw'):
            X_media = model.X_media_raw
            if hasattr(X_media, 'columns'):
                for ch in channels:
                    if ch in X_media.columns:
                        spend[ch] = float(X_media[ch].sum())
            else:
                for i, ch in enumerate(channels):
                    if i < X_media.shape[1]:
                        spend[ch] = float(X_media[:, i].sum())
        
        # Last fallback: estimate from synthetic data assumption
        if not spend:
            print("    Warning: Could not extract spend from model, using estimates")
            for ch in channels:
                spend[ch] = 100000.0  # Default placeholder
        
        return spend
    
    spend_data = _extract_spend_data(mmm)
    
    # -------------------------------------------------------------------------
    # 2. ROI with Uncertainty
    # -------------------------------------------------------------------------
    print("\n[2/7] ROI Computation...")
    
    try:
        roi_df = compute_roi_with_uncertainty(mmm, spend_data=spend_data, hdi_prob=hdi_prob)
        metrics['roi'] = roi_df
        
        print(f"\n  {'Channel':<15} {'ROI':>8} {'95% CI':>18} {'P(ROI>1)':>10}")
        print("  " + "-" * 55)
        for _, row in roi_df.iterrows():
            ci = f"[{row['roi_hdi_low']:.2f}, {row['roi_hdi_high']:.2f}]"
            print(f"  {row['channel']:<15} {row['roi_mean']:>8.2f} {ci:>18} {row['prob_profitable']:>10.0%}")
    except Exception as e:
        print(f"  ⚠️  ROI helper failed: {e}")
        print("     Computing basic ROI from posterior...")
        
        # Manual fallback: compute ROI from beta coefficients
        try:
            posterior = mmm._trace.posterior
            roi_results = []
            
            for ch in mmm.channel_names:
                # Get beta coefficient
                beta_name = f'beta_{ch}'
                if beta_name not in posterior:
                    continue
                
                beta_samples = posterior[beta_name].values.flatten()
                spend = spend_data.get(ch, 1.0)
                
                # Rough contribution estimate: beta * mean_media_effect * y_std
                # This is simplified but gives directional results
                contrib_samples = beta_samples * mmm.y_std * mmm.n_obs
                roi_samples = contrib_samples / spend if spend > 0 else np.zeros_like(contrib_samples)
                
                roi_mean = float(np.mean(roi_samples))
                roi_lower = float(np.percentile(roi_samples, (1 - hdi_prob) / 2 * 100))
                roi_upper = float(np.percentile(roi_samples, (1 + hdi_prob) / 2 * 100))
                
                roi_results.append({
                    'channel': ch,
                    'spend': spend,
                    'roi_mean': roi_mean,
                    'roi_hdi_low': roi_lower,
                    'roi_hdi_high': roi_upper,
                    'prob_positive': float(np.mean(roi_samples > 0)),
                    'prob_profitable': float(np.mean(roi_samples > 1)),
                })
            
            if roi_results:
                roi_df = pd.DataFrame(roi_results)
                metrics['roi'] = roi_df
                
                print(f"\n  {'Channel':<15} {'ROI':>8} {'95% CI':>18} {'P(ROI>1)':>10}")
                print("  " + "-" * 55)
                for _, row in roi_df.iterrows():
                    ci = f"[{row['roi_hdi_low']:.2f}, {row['roi_hdi_high']:.2f}]"
                    print(f"  {row['channel']:<15} {row['roi_mean']:>8.2f} {ci:>18} {row['prob_profitable']:>10.0%}")
            else:
                print("  No beta parameters found")
                metrics['roi'] = None
        except Exception as e2:
            print(f"  Fallback also failed: {e2}")
            metrics['roi'] = None
    
    # -------------------------------------------------------------------------
    # 3. Prior vs Posterior Comparison
    # -------------------------------------------------------------------------
    print("\n[3/7] Prior vs Posterior Analysis...")
    
    try:
        # Get beta parameters for channels
        beta_params = [f"beta_{ch}" for ch in mmm.channel_names]
        comparisons = get_prior_posterior_comparison(
            mmm,
            parameters=beta_params + ['intercept', 'sigma'],
            n_prior_samples=1000,
        )
        
        metrics['prior_posterior'] = comparisons
        
        shrinkage_df = compute_shrinkage_summary(comparisons)
        metrics['shrinkage'] = shrinkage_df
        
        print(f"\n  {'Parameter':<20} {'Prior SD':>10} {'Post SD':>10} {'Shrinkage':>10}")
        print("  " + "-" * 55)
        for c in comparisons[:5]:  # Show first 5
            prior_sd = f"{c.prior_sd:.3f}" if c.prior_sd else "N/A"
            shrink = f"{c.shrinkage:.0%}" if c.shrinkage else "N/A"
            print(f"  {c.parameter:<20} {prior_sd:>10} {c.posterior_sd:>10.3f} {shrink:>10}")
    except Exception as e:
        print(f"  ⚠️  Prior-posterior helper failed: {e}")
        print("     Computing basic posterior summary...")
        
        # Manual fallback: just show posterior summary
        try:
            posterior = mmm._trace.posterior
            post_summary = []
            
            params_to_check = [f"beta_{ch}" for ch in mmm.channel_names] + ['intercept', 'sigma']
            
            for param in params_to_check:
                if param in posterior:
                    samples = posterior[param].values.flatten()
                    post_summary.append({
                        'parameter': param,
                        'posterior_mean': float(np.mean(samples)),
                        'posterior_sd': float(np.std(samples)),
                    })
            
            if post_summary:
                metrics['prior_posterior'] = post_summary
                print(f"\n  {'Parameter':<20} {'Post Mean':>12} {'Post SD':>10}")
                print("  " + "-" * 45)
                for p in post_summary[:5]:
                    print(f"  {p['parameter']:<20} {p['posterior_mean']:>12.4f} {p['posterior_sd']:>10.4f}")
            else:
                metrics['prior_posterior'] = None
            metrics['shrinkage'] = None
        except Exception as e2:
            print(f"  Fallback also failed: {e2}")
            metrics['prior_posterior'] = None
            metrics['shrinkage'] = None
    
    # -------------------------------------------------------------------------
    # 4. Saturation Curves
    # -------------------------------------------------------------------------
    print("\n[4/7] Saturation Analysis...")
    
    try:
        sat_curves = compute_saturation_curves_with_uncertainty(
            mmm,
            n_points=100,
            n_samples=500,
            hdi_prob=hdi_prob,
        )
        metrics['saturation'] = sat_curves
        
        print(f"\n  {'Channel':<15} {'Current Spend':>15} {'Saturation':>12} {'Marginal ROI':>12}")
        print("  " + "-" * 60)
        for ch, curve in sat_curves.items():
            print(f"  {ch:<15} ${curve.current_spend:>14,.0f} {curve.saturation_level:>11.0%} "
                  f"{curve.marginal_response_at_current:>12.4f}")
    except Exception as e:
        print(f"  ⚠️  Saturation helper failed: {e}")
        print("     Computing basic saturation info from posterior...")
        
        # Manual fallback: extract basic saturation parameters from trace
        try:
            posterior = mmm._trace.posterior
            sat_info = {}
            
            for ch in mmm.channel_names:
                # Try to find saturation lambda parameter
                for param_name in [f'sat_lam_{ch}', f'saturation_lam_{ch}', f'lam_{ch}']:
                    if param_name in posterior:
                        lam_samples = posterior[param_name].values.flatten()
                        sat_info[ch] = {
                            'lam_mean': float(np.mean(lam_samples)),
                            'lam_std': float(np.std(lam_samples)),
                        }
                        break
            
            if sat_info:
                metrics['saturation'] = sat_info
                print(f"\n  {'Channel':<15} {'λ (saturation rate)':>20}")
                print("  " + "-" * 40)
                for ch, info in sat_info.items():
                    print(f"  {ch:<15} {info['lam_mean']:>15.3f} ± {info['lam_std']:.3f}")
            else:
                print("  No saturation parameters found in posterior")
                metrics['saturation'] = None
        except Exception as e2:
            print(f"  Fallback also failed: {e2}")
            metrics['saturation'] = None
    
    # -------------------------------------------------------------------------
    # 5. Adstock Weights
    # -------------------------------------------------------------------------
    print("\n[5/7] Adstock Analysis...")
    
    try:
        adstock_weights = compute_adstock_weights(mmm, hdi_prob=hdi_prob)
        metrics['adstock'] = adstock_weights
        
        print(f"\n  {'Channel':<15} {'Alpha':>10} {'Half-life':>12} {'Total Carryover':>15}")
        print("  " + "-" * 55)
        for ch, result in adstock_weights.items():
            print(f"  {ch:<15} {result.alpha_mean:>10.3f} {result.half_life:>11.1f}w "
                  f"{result.total_carryover:>14.1%}")
    except Exception as e:
        print(f"  ⚠️  Adstock helper failed: {e}")
        print("     Computing basic adstock info from posterior...")
        
        # Manual fallback: extract adstock parameters from trace
        try:
            posterior = mmm._trace.posterior
            adstock_info = {}
            
            for ch in mmm.channel_names:
                # Try to find adstock alpha parameter
                for param_name in [f'adstock_{ch}', f'alpha_{ch}', f'decay_{ch}']:
                    if param_name in posterior:
                        alpha_samples = posterior[param_name].values.flatten()
                        alpha_mean = float(np.mean(alpha_samples))
                        
                        # Compute half-life
                        if 0 < alpha_mean < 1:
                            half_life = np.log(0.5) / np.log(alpha_mean)
                        else:
                            half_life = 0.0
                        
                        adstock_info[ch] = {
                            'alpha_mean': alpha_mean,
                            'alpha_std': float(np.std(alpha_samples)),
                            'half_life': float(half_life),
                        }
                        break
            
            if adstock_info:
                metrics['adstock'] = adstock_info
                print(f"\n  {'Channel':<15} {'Alpha':>10} {'Half-life':>12}")
                print("  " + "-" * 40)
                for ch, info in adstock_info.items():
                    print(f"  {ch:<15} {info['alpha_mean']:>10.3f} {info['half_life']:>11.1f}w")
            else:
                print("  No adstock parameters found in posterior")
                metrics['adstock'] = None
        except Exception as e2:
            print(f"  Fallback also failed: {e2}")
            metrics['adstock'] = None
    
    # -------------------------------------------------------------------------
    # 6. Component Decomposition
    # -------------------------------------------------------------------------
    print("\n[6/7] Component Decomposition...")
    
    try:
        decomp = compute_component_decomposition(
            mmm,
            include_time_series=True,
            hdi_prob=hdi_prob,
        )
        metrics['decomposition'] = decomp
        
        # Calculate totals
        total_contribution = sum(d.total_contribution for d in decomp)
        
        print(f"\n  {'Component':<20} {'Contribution':>18} {'% of Total':>12}")
        print("  " + "-" * 55)
        for d in decomp:
            pct = d.total_contribution / total_contribution if total_contribution != 0 else 0
            print(f"  {d.component:<20} ${d.total_contribution:>17,.0f} {pct:>11.1%}")
    except Exception as e:
        print(f"  ⚠️  Decomposition failed: {e}")
        metrics['decomposition'] = None
    
    # -------------------------------------------------------------------------
    # 7. Model Fit Statistics
    # -------------------------------------------------------------------------
    print("\n[7/7] Model Fit Statistics...")
    
    try:
        # Get predictions
        pred = mmm.predict(return_original_scale=True)
        y_obs = mmm.y_raw
        y_pred = pred.y_pred_mean
        
        # R-squared
        ss_res = np.sum((y_obs - y_pred) ** 2)
        ss_tot = np.sum((y_obs - y_obs.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # MAPE
        mape = np.mean(np.abs(y_obs - y_pred) / np.abs(y_obs)) * 100
        
        # RMSE
        rmse = np.sqrt(np.mean((y_obs - y_pred) ** 2))
        
        metrics['fit_stats'] = {
            'r_squared': r_squared,
            'mape': mape,
            'rmse': rmse,
            'n_obs': len(y_obs),
        }
        
        print(f"\n  R-squared: {r_squared:.3f}")
        print(f"  MAPE: {mape:.1f}%")
        print(f"  RMSE: ${rmse:,.0f}")
        
        # Store predictions for visualization
        metrics['predictions'] = {
            'y_obs': y_obs,
            'y_pred_mean': y_pred,
            'y_pred_lower': pred.y_pred_hdi_low if hasattr(pred, 'y_pred_hdi_low') else None,
            'y_pred_upper': pred.y_pred_hdi_high if hasattr(pred, 'y_pred_hdi_high') else None,
        }
    except Exception as e:
        print(f"  ⚠️  Fit statistics failed: {e}")
        metrics['fit_stats'] = None
        metrics['predictions'] = None
    
    # -------------------------------------------------------------------------
    # 8. Full Model Summary (convenience)
    # -------------------------------------------------------------------------
    try:
        metrics['summary'] = generate_model_summary(mmm, hdi_prob=hdi_prob)
    except Exception as e:
        print(f"  Note: Full summary generation failed: {e}")
        metrics['summary'] = None
    
    print("\n" + "=" * 60)
    print("✓ All metrics computed!")
    print("=" * 60)
    
    return metrics


# =============================================================================
# Step 5: Generate Report
# =============================================================================

def generate_report(
    mmm: 'BayesianMMM',
    panel,
    results: 'FitResults',
    metrics: dict,
    output_path: str = "mmm_report.html",
    title: str = "Marketing Mix Model Analysis",
    client: str = "Demo Client",
    analysis_period: str = "2022-2024",
) -> Path:
    """
    Generate a comprehensive HTML report.
    
    Parameters
    ----------
    mmm : BayesianMMM
        Fitted model
    panel : Panel
        Panel data
    results : FitResults
        Fit results
    metrics : dict
        Pre-computed metrics from compute_all_metrics()
    output_path : str
        Output file path
    title : str
        Report title
    client : str
        Client name
    analysis_period : str
        Analysis period description
        
    Returns
    -------
    Path
        Path to generated report
    """
    print("\n" + "=" * 60)
    print("GENERATING REPORT")
    print("=" * 60)
    
    # Configure report using correct parameter names
    config = ReportConfig(
        title=title,
        client=client,
        analysis_period=analysis_period,
        generated_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        
        # Color scheme
        color_scheme=ColorScheme.from_palette(ColorPalette.SAGE),
        
        # Credible interval for uncertainty bands
        default_credible_interval=0.94,
        
        # Enable all sections
        executive_summary=SectionConfig(enabled=True),
        model_fit=SectionConfig(enabled=True),
        channel_roi=SectionConfig(enabled=True),
        decomposition=SectionConfig(enabled=True),
        saturation=SectionConfig(enabled=True),
        methodology=SectionConfig(enabled=True),
        diagnostics=SectionConfig(enabled=True),
    )
    
    # Generate report from model
    report = MMMReportGenerator(
        model=mmm,
        panel=panel,
        results=results,
        config=config,
    )
    
    # Save report
    output = Path(output_path)
    report.to_html(output)
    
    print(f"\n✓ Report saved to: {output}")
    
    return output


def generate_report_minimal(
    mmm: 'BayesianMMM',
    output_path: str = "mmm_report_minimal.html",
) -> Path:
    """
    Generate a minimal stakeholder report.
    
    For quick updates with just the key findings.
    """
    report = (
        ReportBuilder()
        .with_model(mmm)
        .with_title("Marketing Performance Summary")
        .minimal_report()  # Only executive summary, ROI, methodology
        .build()
    )
    
    output = Path(output_path)
    report.to_html(output)
    
    return output


# =============================================================================
# Step 6: Print Key Findings
# =============================================================================

def print_key_findings(metrics: dict):
    """
    Print a summary of key findings with proper uncertainty communication.
    """
    print("\n" + "=" * 70)
    print("KEY FINDINGS (with uncertainty)")
    print("=" * 70)
    
    print("\n📊 MODEL FIT")
    print("-" * 50)
    if metrics.get('fit_stats'):
        fs = metrics['fit_stats']
        print(f"  The model explains {fs['r_squared']:.0%} of variance in sales")
        print(f"  Average prediction error (MAPE): {fs['mape']:.1f}%")
    
    if metrics.get('diagnostics', {}).get('converged'):
        print("  ✓ MCMC sampling converged properly")
    else:
        print("  ⚠️ MCMC convergence issues - interpret with caution")
    
    print("\n💰 CHANNEL ROI")
    print("-" * 50)
    if metrics.get('roi') is not None:
        roi_df = metrics['roi']
        roi_sorted = roi_df.sort_values('roi_mean', ascending=False)
        
        for _, row in roi_sorted.iterrows():
            prob_prof = row['prob_profitable']
            
            if prob_prof > 0.95:
                certainty = "very likely profitable"
            elif prob_prof > 0.80:
                certainty = "likely profitable"
            elif prob_prof > 0.50:
                certainty = "possibly profitable"
            else:
                certainty = "uncertain profitability"
            
            print(f"  {row['channel']}: ROI = {row['roi_mean']:.2f}x "
                  f"(94% CI: [{row['roi_hdi_low']:.2f}, {row['roi_hdi_high']:.2f}])")
            print(f"    → {certainty} ({prob_prof:.0%} probability ROI > 1)")
    
    print("\n📈 SATURATION STATUS")
    print("-" * 50)
    if metrics.get('saturation'):
        for ch, curve in metrics['saturation'].items():
            # Handle both object-style (from helper) and dict-style (from fallback)
            if isinstance(curve, dict):
                # Fallback format - only has lambda parameters
                lam_mean = curve.get('lam_mean', 0)
                lam_std = curve.get('lam_std', 0)
                print(f"  {ch}: λ (saturation rate) = {lam_mean:.3f} ± {lam_std:.3f}")
                print(f"    → Higher λ means faster saturation")
            else:
                # Full saturation curve object
                sat_level = curve.saturation_level
                
                if sat_level > 0.8:
                    status = "highly saturated - diminishing returns"
                elif sat_level > 0.5:
                    status = "moderately saturated"
                else:
                    status = "opportunity for increased spend"
                
                print(f"  {ch}: {sat_level:.0%} saturated")
                print(f"    → {status}")
    
    print("\n⏱️ CARRYOVER EFFECTS")
    print("-" * 50)
    if metrics.get('adstock'):
        for ch, result in metrics['adstock'].items():
            # Handle both object-style and dict-style
            if isinstance(result, dict):
                half_life = result.get('half_life', 0)
                alpha = result.get('alpha_mean', 0)
                print(f"  {ch}: Half-life = {half_life:.1f} weeks, α = {alpha:.3f}")
            else:
                print(f"  {ch}: Half-life = {result.half_life:.1f} weeks, "
                      f"Total carryover = {result.total_carryover:.0%}")
    
    print("\n" + "=" * 70)
    print("IMPORTANT: These estimates represent our uncertainty about true effects.")
    print("Wide credible intervals indicate less certainty, not lower effects.")
    print("=" * 70)

# =============================================================================
# Main Execution
# =============================================================================

def run_complete_workflow(
    n_weeks: int = 104,
    use_numpyro: bool = False,
    n_chains: int = 2,
    n_draws: int = 500,
    n_tune: int = 500,
    output_dir: str = ".",
    random_seed: int = 42,
):
    """
    Run the complete MMM workflow.
    
    Parameters
    ----------
    n_weeks : int
        Number of weeks of data
    use_numpyro : bool
        Use NumPyro backend (faster, requires JAX)
    n_chains : int
        Number of MCMC chains
    n_draws : int
        Number of posterior draws
    n_tune : int
        Number of tuning steps
    output_dir : str
        Output directory for reports
    random_seed : int
        Random seed for reproducibility
    """
    print("=" * 70)
    print("COMPLETE MMM WORKFLOW")
    print("=" * 70)
    print(f"Data: {n_weeks} weeks")
    print(f"Backend: {'NumPyro' if use_numpyro else 'PyMC'}")
    print(f"Sampling: {n_chains} chains × {n_draws} draws")
    print("=" * 70)
    
    # Step 1: Generate data
    print("\n[STEP 1/5] Generating synthetic MFF data...")
    mff_data = generate_synthetic_mff(n_weeks=n_weeks, seed=random_seed)
    print(f"  Generated {len(mff_data)} MFF records")
    
    # Step 2: Create configuration
    print("\n[STEP 2/5] Creating model configuration...")
    mff_config = create_mff_config_national()
    print(f"  KPI: {mff_config.kpi.name}")
    print(f"  Media channels: {mff_config.media_names}")
    print(f"  Controls: {mff_config.control_names}")
    
    # Step 3: Load data
    print("\n[STEP 3/5] Loading panel data...")
    panel = load_mff(mff_data, mff_config)
    print(f"  Observations: {panel.n_obs}")
    print(f"  Channels: {panel.n_channels}")
    
    # Step 4: Configure and fit model
    print("\n[STEP 4/5] Configuring model...")
    model_config = create_model_config(
        use_numpyro=use_numpyro,
        hierarchical=False,
        n_chains=n_chains,
        n_draws=n_draws,
        n_tune=n_tune,
    )
    trend_config = create_trend_config("linear")
    
    mmm, results = fit_model(panel, model_config, trend_config, random_seed=random_seed)
    
    # Step 5: Compute metrics and generate report
    print("\n[STEP 5/5] Computing metrics and generating report...")
    metrics = compute_all_metrics(mmm, results)
    
    # Generate reports
    output_path = Path(output_dir)
    report_path = generate_report(
        mmm=mmm,
        panel=panel,
        results=results,
        metrics=metrics,
        output_path=str(output_path / "mmm_report.html"),
        title="Marketing Mix Model Analysis",
        client="Demo Company",
        analysis_period=f"{n_weeks} weeks (synthetic data)",
    )
    
    # Print key findings
    print_key_findings(metrics)
    
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE!")
    print("=" * 70)
    print(f"Report: {report_path}")
    
    return mmm, results, metrics


# =============================================================================
# Quick Start Functions
# =============================================================================

def quick_national_model(n_weeks: int = 52) -> tuple:
    """
    Quick start: National-level model with minimal configuration.
    
    Good for initial exploration.
    """
    # Generate data
    mff_data = generate_synthetic_mff(n_weeks=n_weeks)
    
    # Simple config - must match variable names in generated data
    config = (
        MFFConfigBuilder()
        .with_kpi_name("Sales")
        .add_national_media("TV", adstock_lmax=8)
        .add_national_media("Digital", adstock_lmax=4)
        .add_national_media("Paid_Social", adstock_lmax=4)
        .add_control_builder(
            ControlVariableConfigBuilder("Price_Index").national().allow_negative()
        )
        .add_control_builder(
            ControlVariableConfigBuilder("Distribution").national()
        )
        .build()
    )
    
    # Load and fit
    panel = load_mff(mff_data, config)
    
    model_config = (
        ModelConfigBuilder()
        .bayesian_pymc()
        .with_chains(2)
        .with_draws(500)
        .with_tune(500)
        .build()
    )
    
    mmm = BayesianMMM(panel, model_config, TrendConfig())
    results = mmm.fit(random_seed=42)
    
    return mmm, results, panel


def quick_geo_model(
    n_weeks: int = 104,
    geos: list[str] = ["East", "West", "Central"],
) -> tuple:
    """
    Quick start: Geo-level panel model with hierarchical pooling.
    """
    # Generate geo data
    mff_data = generate_synthetic_mff(n_weeks=n_weeks, geographies=geos)
    
    # Geo config - must match variable names in generated data
    config = (
        MFFConfigBuilder()
        .with_kpi_builder(KPIConfigBuilder("Sales").by_geo())
        .add_national_media("TV", adstock_lmax=8)
        .add_national_media("Digital", adstock_lmax=4)
        .add_national_media("Paid_Social", adstock_lmax=4)
        .add_control_builder(
            ControlVariableConfigBuilder("Price_Index").by_geo().allow_negative()
        )
        .add_control_builder(
            ControlVariableConfigBuilder("Distribution").by_geo()
        )
        .with_alignment_builder(DimensionAlignmentConfigBuilder().geo_equal())
        .build()
    )
    
    panel = load_mff(mff_data, config)
    
    model_config = (
        ModelConfigBuilder()
        .bayesian_pymc()
        .with_chains(2)
        .with_draws(500)
        .with_tune(500)
        .with_target_accept(0.95)
        .with_hierarchical_builder(
            HierarchicalConfigBuilder().enabled().pool_across_geo()
        )
        .build()
    )
    
    mmm = BayesianMMM(panel, model_config, TrendConfig())
    results = mmm.fit(random_seed=42)
    
    return mmm, results, panel


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MMM Workflow")
    parser.add_argument("--weeks", type=int, default=104, help="Number of weeks")
    parser.add_argument("--chains", type=int, default=2, help="MCMC chains")
    parser.add_argument("--draws", type=int, default=500, help="Posterior draws")
    parser.add_argument("--tune", type=int, default=500, help="Tuning steps")
    parser.add_argument("--numpyro", action="store_true", help="Use NumPyro backend")
    parser.add_argument("--output", type=str, default=".", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    run_complete_workflow(
        n_weeks=args.weeks,
        use_numpyro=args.numpyro,
        n_chains=args.chains,
        n_draws=args.draws,
        n_tune=args.tune,
        output_dir=args.output,
        random_seed=args.seed,
    )