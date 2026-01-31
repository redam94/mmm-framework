"""
Model Validation Workflow Example
=================================

A comprehensive example demonstrating how to use the mmm_framework validation
package to verify the robustness of Marketing Mix Models.

This example covers:
1. Quick validation (PPC, residuals, channel diagnostics)
2. Standard validation (+ LOO-CV, WAIC)
3. Custom validation configurations
4. Interpreting validation results
5. Generating validation reports

The validation package emphasizes honest uncertainty quantification and
helps identify potential issues with model specification.
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
    ModelConfigBuilder,
    SeasonalityConfigBuilder,
    TrendConfigBuilder,
    PriorConfigBuilder,
    # Core model classes
    BayesianMMM,
    TrendConfig,
    TrendType,
    # Data loading
    mff_from_wide_format,
    load_mff,
)

# Validation imports
from mmm_framework.validation import (
    # Main validator
    ModelValidator,
    # Configuration
    ValidationConfig,
    ValidationConfigBuilder,
    ValidationLevel,
    # Results (for type hints and inspection)
    ValidationSummary,
    LiftTestResult,
)

# Reporting (for comparison)
from mmm_framework.reporting.helpers import generate_model_summary


# =============================================================================
# Step 1: Generate Synthetic Data
# =============================================================================


def generate_synthetic_data(
    n_weeks: int = 104,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic wide-format marketing data for demonstration.

    This creates realistic patterns including:
    - Seasonal effects (higher sales in Q4)
    - Media channel effects with diminishing returns
    - Control variable effects (economic indicators)
    - Random noise

    Parameters
    ----------
    n_weeks : int
        Number of weeks of data to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Wide-format marketing data.
    """
    np.random.seed(seed)

    # Date range
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(weeks=i) for i in range(n_weeks)]

    # Base sales with trend and seasonality
    trend = np.linspace(100, 120, n_weeks)
    week_of_year = np.array([d.isocalendar()[1] for d in dates])
    seasonality = 10 * np.sin(2 * np.pi * week_of_year / 52) + 5 * np.sin(
        4 * np.pi * week_of_year / 52
    )

    # Media spend (realistic patterns with correlation)
    tv_base = 50000 + 20000 * np.sin(2 * np.pi * week_of_year / 52)
    tv_spend = np.maximum(0, tv_base + np.random.normal(0, 5000, n_weeks))

    digital_base = 30000 + 10000 * np.cos(2 * np.pi * week_of_year / 26)
    digital_spend = np.maximum(0, digital_base + np.random.normal(0, 3000, n_weeks))

    social_spend = np.maximum(0, 15000 + np.random.normal(0, 2000, n_weeks))

    # Control variable (economic indicator)
    economic_index = 100 + np.cumsum(np.random.normal(0, 0.5, n_weeks))

    # Generate sales with true effects
    # True betas (for reference, not used in fitting)
    # TV: 0.0003 per dollar, Digital: 0.0004 per dollar, Social: 0.0002 per dollar

    def saturation(x, L=1, k=0.00001):
        """Logistic saturation function."""
        return L / (1 + np.exp(-k * x))

    media_effect = (
        15 * saturation(tv_spend, L=1, k=0.00002)
        + 12 * saturation(digital_spend, L=1, k=0.00003)
        + 8 * saturation(social_spend, L=1, k=0.00005)
    )

    # Economic effect
    economic_effect = 0.5 * (economic_index - 100)

    # Noise
    noise = np.random.normal(0, 5, n_weeks)

    # Total sales
    sales = trend + seasonality + media_effect + economic_effect + noise

    # Create DataFrame
    df = pd.DataFrame(
        {
            "date": dates,
            "sales": sales,
            "tv_spend": tv_spend,
            "digital_spend": digital_spend,
            "social_spend": social_spend,
            "economic_index": economic_index,
        }
    )

    return df


# =============================================================================
# Step 2: Configure and Fit Model
# =============================================================================


def configure_and_fit_model(data: pd.DataFrame) -> tuple:
    """
    Configure and fit a BayesianMMM model.

    Returns both the model and results for validation.

    Parameters
    ----------
    data : pd.DataFrame
        Wide-format marketing data.

    Returns
    -------
    tuple
        (model, results, panel) for validation.
    """
    logger.info("Configuring MFF format...")

    # Build MFF configuration
    mff_config = (
        MFFConfigBuilder()
        # KPI Configuration
        .with_kpi_builder(
            KPIConfigBuilder("Sales").national().with_display_name("Weekly Sales")
        )
        # Media Channel Configurations
        .add_media_builder(
            MediaChannelConfigBuilder("TV")
            .national()
            .with_geometric_adstock(l_max=8)
            .with_hill_saturation()
            .with_display_name("Television")
        )
        .add_media_builder(
            MediaChannelConfigBuilder("Digital")
            .national()
            .with_geometric_adstock(l_max=4)
            .with_hill_saturation()
            .with_display_name("Digital Marketing")
        )
        .add_media_builder(
            MediaChannelConfigBuilder("Social")
            .national()
            .with_geometric_adstock(l_max=4)
            .with_hill_saturation()
            .with_display_name("Social Media")
        )
        # Control Variables
        .add_control_builder(
            ControlVariableConfigBuilder("Economic_Index")
            .national()
            .allow_negative()
            .with_display_name("Economic Index")
        )
        .weekly()
        .build()
    )

    # Convert wide-format data to MFF format
    logger.info("Converting to MFF format...")
    value_columns = {
        "sales": "Sales",
        "tv_spend": "TV",
        "digital_spend": "Digital",
        "social_spend": "Social",
        "economic_index": "Economic_Index",
    }
    mff_df = mff_from_wide_format(
        data,
        period_col="date",
        value_columns=value_columns,
    )

    panel = load_mff(mff_df, mff_config)

    # Configure model
    logger.info("Configuring model...")
    model_config = (
        ModelConfigBuilder()
        .bayesian_pymc()
        .with_chains(4)
        .with_draws(1000)
        .with_tune(2000)
        .with_target_accept(0.99)
        .with_seasonality_builder(
            SeasonalityConfigBuilder().with_yearly(order=3)
        )
        .build()
    )

    # Configure trend
    trend_config = TrendConfigBuilder().spline().with_n_knots(5).build()

    # Create and fit model
    logger.info("Creating BayesianMMM...")
    model = BayesianMMM(
        panel=panel,
        model_config=model_config,
        trend_config=trend_config,
    )

    logger.info("Fitting model (this may take a few minutes)...")
    results = model.fit(random_seed=42)

    logger.info("Model fitting complete!")

    return model, results, panel


# =============================================================================
# Step 3: Quick Validation
# =============================================================================


def run_quick_validation(model, results) -> ValidationSummary:
    """
    Run quick validation checks on the fitted model.

    Quick validation includes:
    - Convergence diagnostics (R-hat, ESS, divergences)
    - Posterior predictive checks
    - Residual diagnostics
    - Channel diagnostics (VIF, per-channel convergence)

    Parameters
    ----------
    model : BayesianMMM
        Fitted model.
    results : MMMResults
        Model results.

    Returns
    -------
    ValidationSummary
        Quick validation results.
    """
    logger.info("=" * 60)
    logger.info("QUICK VALIDATION")
    logger.info("=" * 60)

    # Create validator
    validator = ModelValidator(model, results)

    # Run quick validation
    summary = validator.quick_check()

    # Print results
    print("\n" + "=" * 60)
    print("QUICK VALIDATION RESULTS")
    print("=" * 60)

    print(f"\nOverall Quality: {summary.overall_quality.upper()}")
    print(f"Validation Date: {summary.validation_date}")

    # Convergence
    if summary.convergence:
        print("\n--- Convergence Diagnostics ---")
        print(summary.convergence.summary().to_string(index=False))

    # PPC
    if summary.ppc:
        print("\n--- Posterior Predictive Checks ---")
        print(f"Overall Pass: {summary.ppc.overall_pass}")
        if summary.ppc.problematic_checks:
            print(f"Problematic: {', '.join(summary.ppc.problematic_checks)}")
        print(summary.ppc.summary().to_string(index=False))

    # Residuals
    if summary.residuals:
        print("\n--- Residual Diagnostics ---")
        print(f"Overall Adequate: {summary.residuals.overall_adequate}")
        print(summary.residuals.summary().to_string(index=False))

    # Channel diagnostics
    if summary.channel_diagnostics:
        print("\n--- Channel Diagnostics ---")
        print(
            f"Multicollinearity Warning: {summary.channel_diagnostics.multicollinearity_warning}"
        )
        print(f"Convergence Warning: {summary.channel_diagnostics.convergence_warning}")
        print(summary.channel_diagnostics.summary().to_string(index=False))

    # Issues and recommendations
    if summary.critical_issues:
        print("\n--- CRITICAL ISSUES ---")
        for issue in summary.critical_issues:
            print(f"  ❌ {issue}")

    if summary.warnings:
        print("\n--- Warnings ---")
        for warning in summary.warnings:
            print(f"  ⚠️  {warning}")

    if summary.recommendations:
        print("\n--- Recommendations ---")
        for rec in summary.recommendations:
            print(f"  → {rec}")

    return summary


# =============================================================================
# Step 4: Standard Validation (with Model Comparison)
# =============================================================================


def run_standard_validation(model, results) -> ValidationSummary:
    """
    Run standard validation including model comparison metrics.

    Standard validation adds:
    - LOO-CV (Leave-One-Out Cross-Validation via PSIS)
    - WAIC (Widely Applicable Information Criterion)

    Parameters
    ----------
    model : BayesianMMM
        Fitted model.
    results : MMMResults
        Model results.

    Returns
    -------
    ValidationSummary
        Standard validation results.
    """
    logger.info("=" * 60)
    logger.info("STANDARD VALIDATION")
    logger.info("=" * 60)

    validator = ModelValidator(model, results)
    config = ValidationConfig.standard()
    summary = validator.validate(config)

    print("\n" + "=" * 60)
    print("STANDARD VALIDATION RESULTS")
    print("=" * 60)

    print(f"\nOverall Quality: {summary.overall_quality.upper()}")

    # Model comparison results
    if summary.model_comparison and summary.model_comparison.models:
        print("\n--- Model Comparison Metrics ---")
        entry = summary.model_comparison.models[0]
        if entry.loo:
            print(
                f"LOO ELPD: {entry.loo.elpd_loo:.2f} (SE: {entry.loo.se_elpd_loo:.2f})"
            )
            print(f"p_loo (effective parameters): {entry.loo.p_loo:.2f}")
            print(f"Bad Pareto k values (>0.7): {entry.loo.n_bad_k}")
        if entry.waic:
            print(f"WAIC: {entry.waic.waic:.2f} (SE: {entry.waic.se_waic:.2f})")
            print(f"p_waic: {entry.waic.p_waic:.2f}")

    return summary


# =============================================================================
# Step 5: Custom Validation Configuration
# =============================================================================


def run_custom_validation(model, results) -> ValidationSummary:
    """
    Demonstrate custom validation configuration using the builder.

    Parameters
    ----------
    model : BayesianMMM
        Fitted model.
    results : MMMResults
        Model results.

    Returns
    -------
    ValidationSummary
        Custom validation results.
    """
    logger.info("=" * 60)
    logger.info("CUSTOM VALIDATION")
    logger.info("=" * 60)

    # Build custom configuration
    config = (
        ValidationConfigBuilder()
        .quick()  # Start with quick settings
        .with_ppc(
            n_samples=300,
            checks=("mean", "variance", "extremes"),  # Subset of checks
            include_channel_checks=True,
        )
        .with_residual_tests(
            tests=("durbin_watson", "ljung_box"),  # Only autocorrelation tests
            max_lag=15,
            significance_level=0.05,
        )
        .with_channel_diagnostics(
            vif_threshold=10.0,
            correlation_threshold=0.8,
        )
        .with_model_comparison(method="loo")  # Enable LOO-CV
        .without_plots()  # Disable plot generation for speed
        .build()
    )

    validator = ModelValidator(model, results)
    summary = validator.validate(config)

    print("\n" + "=" * 60)
    print("CUSTOM VALIDATION RESULTS")
    print("=" * 60)
    print(f"\nOverall Quality: {summary.overall_quality.upper()}")
    print(summary.summary().to_string(index=False))

    return summary


# =============================================================================
# Step 6: Generate HTML Report
# =============================================================================


def generate_validation_report(
    summary: ValidationSummary,
    output_path: str = "validation_report.html",
) -> str:
    """
    Generate an HTML validation report.

    Parameters
    ----------
    summary : ValidationSummary
        Validation results.
    output_path : str
        Output file path.

    Returns
    -------
    str
        Path to generated report.
    """
    logger.info("Generating HTML validation report...")

    html = summary.to_html_report()

    output_file = Path(output_path)
    output_file.write_text(html)

    logger.info(f"Validation report saved to: {output_file.absolute()}")

    return str(output_file.absolute())


# =============================================================================
# Step 7: Demonstrate Lift Test Calibration Setup
# =============================================================================


def demonstrate_calibration_setup():
    """
    Demonstrate how to set up validation with lift test calibration.

    Note: Calibration check is not yet fully implemented, but this shows
    how external experiment results would be configured.
    """
    print("\n" + "=" * 60)
    print("CALIBRATION SETUP (DEMONSTRATION)")
    print("=" * 60)

    # Example lift test results from experiments
    lift_tests = [
        LiftTestResult(
            channel="tv_spend",
            test_period=("2023-06-01", "2023-08-31"),
            measured_lift=15000,  # Units
            lift_se=3000,  # Standard error
            holdout_regions=["Northeast", "Southeast"],
            confidence_level=0.95,
        ),
        LiftTestResult(
            channel="digital_spend",
            test_period=("2023-09-01", "2023-10-31"),
            measured_lift=8000,
            lift_se=2000,
            confidence_level=0.95,
        ),
    ]

    # Configuration with calibration (would be used when implemented)
    config = (
        ValidationConfigBuilder()
        .thorough()
        .with_calibration(
            lift_tests=lift_tests,
            ci_level=0.94,
            tolerance_multiplier=1.5,
        )
        .build()
    )

    print("\nLift Test Configuration:")
    for i, test in enumerate(lift_tests, 1):
        print(f"\n  Test {i}: {test.channel}")
        print(f"    Period: {test.test_period[0]} to {test.test_period[1]}")
        print(f"    Measured Lift: {test.measured_lift:,.0f} ± {test.lift_se:,.0f}")
        if test.holdout_regions:
            print(f"    Holdout Regions: {', '.join(test.holdout_regions)}")

    print(
        "\n  [Note: Calibration validation is a placeholder for future implementation]"
    )

    return config


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run the complete validation workflow example."""
    print("=" * 70)
    print("MMM FRAMEWORK - MODEL VALIDATION EXAMPLE")
    print("=" * 70)

    # Generate synthetic data
    logger.info("Generating synthetic marketing data...")
    data = generate_synthetic_data(n_weeks=104, seed=42)
    print(f"\nGenerated {len(data)} weeks of data")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"Media channels: TV, Digital, Social")
    print(f"Control: Economic Index")

    # Fit model
    model, results, panel = configure_and_fit_model(data)

    # Run validations
    quick_summary = run_quick_validation(model, results)
    standard_summary = run_standard_validation(model, results)
    custom_summary = run_custom_validation(model, results)

    # Generate report
    report_path = generate_validation_report(
        standard_summary,
        output_path="validation_report.html",
    )

    # Demonstrate calibration setup
    demonstrate_calibration_setup()

    # Final summary
    print("\n" + "=" * 70)
    print("VALIDATION WORKFLOW COMPLETE")
    print("=" * 70)
    print(f"\nQuick Validation Quality: {quick_summary.overall_quality}")
    print(f"Standard Validation Quality: {standard_summary.overall_quality}")
    print(f"Custom Validation Quality: {custom_summary.overall_quality}")
    print(f"\nHTML Report: {report_path}")

    print("\n" + "-" * 70)
    print("KEY TAKEAWAYS:")
    print("-" * 70)
    print(
        """
    1. Quick validation provides fast feedback on model health
       - Convergence diagnostics (R-hat, ESS, divergences)
       - Posterior predictive checks
       - Residual diagnostics
       - Channel-level multicollinearity (VIF)

    2. Standard validation adds model comparison metrics
       - LOO-CV for predictive accuracy
       - WAIC as an alternative information criterion

    3. Custom validation allows fine-tuning
       - Select specific tests
       - Adjust thresholds
       - Control computation time

    4. Validation results guide model improvement
       - Critical issues require immediate attention
       - Warnings suggest areas for investigation
       - Recommendations provide actionable next steps

    5. HTML reports facilitate stakeholder communication
       - Self-contained validation summary
       - Includes all diagnostics and recommendations
    """
    )

    return quick_summary, standard_summary, custom_summary


if __name__ == "__main__":
    main()
