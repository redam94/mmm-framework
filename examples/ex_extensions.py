"""
Example Usage: MMM Extensions

Demonstrates builder patterns for nested and multivariate models.
"""

import numpy as np
import pandas as pd

# Import from the extensions package
from mmm_framework.mmm_extensions import (
    # Builders
    MediatorConfigBuilder,
    OutcomeConfigBuilder,
    CrossEffectConfigBuilder,
    NestedModelConfigBuilder,
    MultivariateModelConfigBuilder,
    CombinedModelConfigBuilder,
    AdstockConfigBuilder,
    SaturationConfigBuilder,
    EffectPriorConfigBuilder,
    # Factory functions
    awareness_mediator,
    foot_traffic_mediator,
    cannibalization_effect,
    halo_effect,
    # Models
    NestedMMM,
    MultivariateMMM,
    CombinedMMM,
)


# =============================================================================
# Example 1: Nested Model (Media → Awareness → Sales)
# =============================================================================

def example_nested_model():
    """
    Build a nested model where TV/Digital build awareness,
    and awareness drives sales.
    """
    print("=" * 60)
    print("Example 1: Nested Model (Media → Awareness → Sales)")
    print("=" * 60)
    
    # --- Method A: Using factory function ---
    awareness = awareness_mediator(
        name="brand_awareness",
        observation_noise=0.15,
    )
    
    # --- Method B: Using builder (more control) ---
    awareness_custom = (
        MediatorConfigBuilder("brand_awareness")
        .partially_observed(observation_noise=0.15)
        .with_positive_media_effect(sigma=1.0)
        .with_slow_adstock(l_max=12)
        .with_direct_effect(sigma=0.3)  # Allow some direct effect
        .build()
    )
    
    # Build nested model config
    nested_config = (
        NestedModelConfigBuilder()
        .add_mediator(awareness_custom)
        .map_channels_to_mediator(
            "brand_awareness",
            ["tv", "digital", "social"]  # These channels build awareness
        )
        .share_adstock(True)
        .build()
    )
    
    print("\nNested Model Configuration:")
    print(f"  Mediators: {[m.name for m in nested_config.mediators]}")
    print(f"  Media-to-Mediator Map: {dict(nested_config.media_to_mediator_map)}")
    
    # Create synthetic data for demonstration
    np.random.seed(42)
    n_obs = 104  # 2 years of weekly data
    n_channels = 3
    
    X_media = np.random.exponential(scale=100, size=(n_obs, n_channels))
    
    # Survey data (only monthly = 8% of weeks)
    survey_periods = np.random.choice(n_obs, size=8, replace=False)
    awareness_obs = np.full(n_obs, np.nan)
    awareness_obs[survey_periods] = np.random.normal(50, 10, len(survey_periods))
    
    # Outcome
    y = np.random.normal(1000, 100, n_obs)
    
    print("\n  Data shapes:")
    print(f"    X_media: {X_media.shape}")
    print(f"    y: {y.shape}")
    print(f"    Survey observations: {np.sum(~np.isnan(awareness_obs))}")
    
    # Create model (don't fit in example)
    # model = NestedMMM(
    #     X_media=X_media,
    #     y=y,
    #     channel_names=["tv", "digital", "social"],
    #     config=nested_config,
    #     mediator_data={"brand_awareness": awareness_obs},
    # )
    
    return nested_config


# =============================================================================
# Example 2: Multivariate Model (Product Cannibalization)
# =============================================================================

def example_multivariate_model():
    """
    Build a multivariate model with two products:
    - Single-pack sales
    - Multipack sales (cannibalizes single-pack when promoted)
    """
    print("\n" + "=" * 60)
    print("Example 2: Multivariate Model (Product Cannibalization)")
    print("=" * 60)
    
    # Define outcomes
    single_pack = (
        OutcomeConfigBuilder("single_pack", column="single_sales")
        .with_positive_media_effects(sigma=0.5)
        .with_trend()
        .with_seasonality()
        .build()
    )
    
    multipack = (
        OutcomeConfigBuilder("multipack", column="multi_sales")
        .with_positive_media_effects(sigma=0.5)
        .with_trend()
        .with_seasonality()
        .build()
    )
    
    # Define cross-effect: multipack promotions cannibalize single-pack
    # Method A: Factory function
    cannib_effect = cannibalization_effect(
        source="multipack",
        target="single_pack",
        promotion_column="multipack_promo",
        lagged=True,  # Use lag for identification
    )
    
    # Method B: Builder (more control)
    cannib_custom = (
        CrossEffectConfigBuilder("multipack", "single_pack")
        .cannibalization()
        .modulated_by_promotion("multipack_promo")
        .lagged()
        .with_prior_sigma(0.3)
        .build()
    )
    
    # Build multivariate config
    mv_config = (
        MultivariateModelConfigBuilder()
        .add_outcome(single_pack)
        .add_outcome(multipack)
        .add_cross_effect(cannib_custom)
        .with_lkj_eta(2.0)
        .share_media_adstock(True)
        .share_media_saturation(False)  # Different saturation per product
        .share_seasonality(True)
        .build()
    )
    
    print("\nMultivariate Model Configuration:")
    print(f"  Outcomes: {[o.name for o in mv_config.outcomes]}")
    print(f"  Cross-effects: {[(c.source_outcome, '→', c.target_outcome, c.effect_type.value) for c in mv_config.cross_effects]}")
    print(f"  LKJ eta: {mv_config.lkj_eta}")
    print(f"  Shared adstock: {mv_config.share_media_adstock}")
    print(f"  Shared saturation: {mv_config.share_media_saturation}")
    
    return mv_config


# =============================================================================
# Example 3: Combined Model (Full C-Store Scenario)
# =============================================================================

def example_combined_model():
    """
    Full c-store scenario:
    - Media builds awareness (nested)
    - Awareness drives both single-pack and multipack sales
    - Multipack promotions cannibalize single-pack (cross-effect)
    - Correlated errors across products (weather, etc.)
    """
    print("\n" + "=" * 60)
    print("Example 3: Combined Model (Full C-Store Scenario)")
    print("=" * 60)
    
    # Build combined config using fluent API
    config = (
        CombinedModelConfigBuilder()
        # Add awareness mediator
        .with_awareness_mediator(name="brand_awareness")
        # Map TV and digital to awareness (search captures existing demand)
        .map_channels_to_mediator("brand_awareness", ["tv", "digital", "social"])
        # Add outcomes
        .with_outcomes("single_pack", "multipack")
        # Map awareness to both outcomes
        .map_mediator_to_outcomes("brand_awareness", ["single_pack", "multipack"])
        # Add cannibalization
        .with_cannibalization(
            source="multipack",
            target="single_pack",
            promotion_column="multipack_promo"
        )
        # Configure correlations
        .with_lkj_eta(2.0)
        .build()
    )
    
    print("\nCombined Model Configuration:")
    print(f"  Mediators: {[m.name for m in config.nested.mediators]}")
    print(f"  Outcomes: {[o.name for o in config.multivariate.outcomes]}")
    print(f"  Cross-effects: {[(c.source_outcome, '→', c.target_outcome) for c in config.multivariate.cross_effects]}")
    print(f"  Mediator-to-Outcome Map: {dict(config.mediator_to_outcome_map)}")
    
    # Show the causal structure
    print("\n  Causal Structure:")
    print("    Media Channels")
    print("         │")
    print("         ├─────────────────────────────┐")
    print("         ▼                             ▼")
    print("     Awareness                    Direct Effect")
    print("         │                             │")
    print("         └──────────┬──────────────────┘")
    print("                    ▼")
    print("               Outcomes")
    print("         ┌──────────┴──────────┐")
    print("         ▼                     ▼")
    print("    Single-pack           Multipack")
    print("         │                     │")
    print("         └────── cannib ───────┘")
    
    return config


# =============================================================================
# Example 4: Custom Configurations
# =============================================================================

def example_custom_configurations():
    """
    Demonstrate advanced customization of configurations.
    """
    print("\n" + "=" * 60)
    print("Example 4: Custom Configurations")
    print("=" * 60)
    
    # Custom adstock: slow decay for brand-building media
    brand_adstock = (
        AdstockConfigBuilder()
        .with_max_lag(16)
        .with_beta_prior(alpha=4, beta=1)  # Favor high alpha (slow decay)
        .build()
    )
    
    # Custom adstock: fast decay for performance media
    perf_adstock = (
        AdstockConfigBuilder()
        .with_max_lag(4)
        .with_beta_prior(alpha=1, beta=4)  # Favor low alpha (fast decay)
        .build()
    )
    
    print("\nCustom Adstock Configurations:")
    print(f"  Brand media: l_max={brand_adstock.l_max}, prior=Beta({brand_adstock.prior_alpha}, {brand_adstock.prior_beta})")
    print(f"  Perf media: l_max={perf_adstock.l_max}, prior=Beta({perf_adstock.prior_alpha}, {perf_adstock.prior_beta})")
    
    # Custom effect prior: positive with tight regularization
    tight_positive = (
        EffectPriorConfigBuilder()
        .positive(sigma=0.3)
        .with_tight_prior()
        .build()
    )
    
    print(f"\n  Tight positive effect: constraint={tight_positive.constraint.value}, sigma={tight_positive.sigma}")
    
    # Custom mediator: foot traffic with special configuration
    traffic = (
        MediatorConfigBuilder("foot_traffic")
        .fully_observed(observation_noise=0.03)  # Very precise counters
        .with_positive_media_effect(sigma=0.8)
        .without_direct_effect()  # All effect flows through traffic
        .with_adstock(perf_adstock)  # Fast decay
        .build()
    )
    
    print(f"\n  Traffic mediator:")
    print(f"    Type: {traffic.mediator_type.value}")
    print(f"    Observation noise: {traffic.observation_noise_sigma}")
    print(f"    Allow direct effect: {traffic.allow_direct_effect}")
    
    return traffic


# =============================================================================
# Example 5: Model Variants Comparison
# =============================================================================

def example_model_variants():
    """
    Show how to set up different model variants for comparison.
    """
    print("\n" + "=" * 60)
    print("Example 5: Model Variants for Comparison")
    print("=" * 60)
    
    # Variant 1: Direct effects only (baseline)
    baseline = (
        MultivariateModelConfigBuilder()
        .with_outcomes("single_pack", "multipack")
        .with_weak_correlations()  # Just correlated errors
        .build()
    )
    
    # Variant 2: Add cross-effects
    with_cross = (
        MultivariateModelConfigBuilder()
        .with_outcomes("single_pack", "multipack")
        .with_cannibalization("multipack", "single_pack", promotion_column="promo")
        .with_weak_correlations()
        .build()
    )
    
    # Variant 3: Add mediation
    with_mediation = (
        CombinedModelConfigBuilder()
        .with_awareness_mediator()
        .with_outcomes("single_pack", "multipack")
        .map_mediator_to_outcomes("brand_awareness", ["single_pack", "multipack"])
        .build()
    )
    
    # Variant 4: Full model
    full_model = (
        CombinedModelConfigBuilder()
        .with_awareness_mediator()
        .with_outcomes("single_pack", "multipack")
        .map_mediator_to_outcomes("brand_awareness", ["single_pack", "multipack"])
        .with_cannibalization("multipack", "single_pack", promotion_column="promo")
        .build()
    )
    
    print("\nModel Variants:")
    print("  1. Baseline: Direct effects + correlated errors")
    print("  2. + Cross-effects: Adds cannibalization")
    print("  3. + Mediation: Adds awareness pathway")
    print("  4. Full model: All of the above")
    
    print("\n  Compare using WAIC/LOO-CV after fitting each variant")
    
    return baseline, with_cross, with_mediation, full_model


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Run all examples
    example_nested_model()
    example_multivariate_model()
    example_combined_model()
    example_custom_configurations()
    example_model_variants()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)