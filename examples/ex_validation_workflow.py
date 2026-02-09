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
6. **NEW** Cross-validation for out-of-sample performance
7. **NEW** Sensitivity analysis for prior robustness
8. **NEW** Stability analysis for model reliability
9. **NEW** Calibration against lift test experiments

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
    CrossValidationResults,
    SensitivityResults,
    StabilityResults,
    CalibrationResults,
)

# Reporting (for comparison)
from mmm_framework.reporting.helpers import generate_model_summary

# =============================================================================
# Step 1: Generate Synthetic Data
# =============================================================================


def generate_synthetic_data(
    n_weeks: int = 156,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic wide-format marketing data that matches the BayesianMMM model.

    This generates data using the EXACT same transformations as BayesianMMM:
    1. Adstock with geometric decay (alpha interpolation between 0.1 and 0.9)
    2. Normalization by max value (to 0-1 range)
    3. Exponential saturation: 1 - exp(-sat_lam * x_normalized)
    4. Linear combination with beta coefficients
    5. Y standardization (z-score)

    TRUE PARAMETERS (for validation):
    - TV: adstock_mix=0.3, sat_lam=2.0, beta=0.6
    - Digital: adstock_mix=0.5, sat_lam=3.0, beta=0.5
    - Social: adstock_mix=0.2, sat_lam=4.0, beta=0.3
    - trend_slope=0.1
    - sigma=0.15 (on standardized scale)

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

    # Import the adstock function used by the model
    from mmm_framework.transforms.adstock import geometric_adstock_2d

    # Date range
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(weeks=i) for i in range(n_weeks)]

    # === TRUE PARAMETERS ===
    # These should be recoverable by the model
    TRUE_PARAMS = {
        "TV": {"adstock_mix": 0.1, "sat_lam": 10.0, "beta": 0.5},
        "Digital": {"adstock_mix": 0.1, "sat_lam": 3.0, "beta": 0.6},
        "Social": {"adstock_mix": 0.1, "sat_lam": 4.0, "beta": 0.7},
    }
    TRUE_TREND_SLOPE = 0.1  # Slope on scaled time (0-1)
    TRUE_SIGMA = 0.01  # Noise on standardized y scale
    TRUE_CONTROL_BETA = 0.4  # Effect of standardized control

    # === Generate media spend patterns ===
    week_of_year = np.array([d.isocalendar()[1] for d in dates])

    # TV: Higher in Q4, some seasonality
    tv_base = 50000 + 15000 * np.sin(2 * np.pi * week_of_year / 52)
    tv_spend = np.maximum(0, tv_base + np.random.normal(0, 40_000, n_weeks))

    # Digital: Bi-annual pattern
    digital_base = 35000 + 8000 * np.cos(2 * np.pi * week_of_year / 26)
    digital_spend = np.maximum(0, digital_base + np.random.normal(0, 20_000, n_weeks))
    # Social: Relatively stable
    social_spend = np.maximum(0, 18000 + np.random.normal(0, 20_000, n_weeks))

    # Stack into matrix (n_weeks, 3)
    X_media_raw = np.column_stack([tv_spend, digital_spend, social_spend])
    channel_names = ["TV", "Digital", "Social"]

    # === Apply EXACT model transformations ===
    # The model uses alphas 0.1 and 0.9, then interpolates based on adstock_mix

    alpha_low, alpha_high = 0.1, 0.9

    # Apply adstock at both alpha values
    X_adstock_low = geometric_adstock_2d(X_media_raw, alpha_low)
    X_adstock_high = geometric_adstock_2d(X_media_raw, alpha_high)

    # Normalize by max (as model does)
    media_max = {}
    for c, ch_name in enumerate(channel_names):
        # Max of the high-alpha version (model uses this for normalization)
        media_max[ch_name] = max(X_adstock_low[:, c].max(), X_adstock_high[:, c].max())
        X_adstock_low[:, c] = X_adstock_low[:, c] / (media_max[ch_name] + 1e-8)
        X_adstock_high[:, c] = X_adstock_high[:, c] / (media_max[ch_name] + 1e-8)

    # Compute channel contributions using true parameters
    channel_contributions = []
    for c, ch_name in enumerate(channel_names):
        params = TRUE_PARAMS[ch_name]
        adstock_mix = params["adstock_mix"]
        sat_lam = params["sat_lam"]
        beta = params["beta"]

        # Interpolate adstocked values
        x_adstocked = (1 - adstock_mix) * X_adstock_low[
            :, c
        ] + adstock_mix * X_adstock_high[:, c]

        # Apply saturation (same formula as model)
        exponent = np.clip(-sat_lam * x_adstocked, -20, 0)
        x_saturated = 1 - np.exp(exponent)

        # Apply beta
        contribution = beta * x_saturated
        channel_contributions.append(contribution)

    media_effect = sum(channel_contributions)
    import matplotlib.pyplot as plt
    plt.plot(channel_contributions[0], label='TV Contribution')
    plt.show();
    plt.scatter(tv_spend, channel_contributions[0])
    plt.show();
    # === Generate baseline components ===
    # Intercept (will be around 0 after standardization)
    intercept = 0.0  # On standardized scale

    # Trend: linear on scaled time
    t_scaled = np.linspace(0, 1, n_weeks)
    trend = TRUE_TREND_SLOPE * t_scaled

    # Seasonality: using yearly Fourier (order 3 like the example)
    yearly_season = (
        0.15 * np.sin(2 * np.pi * t_scaled)
        + 0.08 * np.cos(2 * np.pi * t_scaled)
        + 0.05 * np.sin(4 * np.pi * t_scaled)
        + 0.03 * np.cos(4 * np.pi * t_scaled)
    )

    # Control variable (economic indicator)
    economic_index_raw = 100 + np.cumsum(np.random.normal(0, 0.3, n_weeks))
    # Standardize control (as model does)
    economic_mean = economic_index_raw.mean()
    economic_std = economic_index_raw.std() + 1e-8
    economic_standardized = (economic_index_raw - economic_mean) / economic_std
    control_effect = TRUE_CONTROL_BETA * economic_standardized

    # === Combine on standardized scale ===
    y_standardized = (
        intercept
        + trend
        + yearly_season
        + media_effect
        + control_effect
        + np.random.normal(0, TRUE_SIGMA, n_weeks)
    )

    # === Transform back to raw scale for realistic values ===
    # Choose reasonable mean and std for sales
    y_mean = 150.0  # Average weekly sales
    y_std = 20.0  # Standard deviation of sales
    sales = y_standardized * y_std + y_mean

    # Print diagnostic info
    print("\n=== SYNTHETIC DATA GENERATION ===")
    print(f"Generated {n_weeks} weeks of data")
    print(f"\nTrue Parameters (should be recoverable):")
    for ch_name, params in TRUE_PARAMS.items():
        print(
            f"  {ch_name}: adstock_mix={params['adstock_mix']:.2f}, "
            f"sat_lam={params['sat_lam']:.1f}, beta={params['beta']:.2f}"
        )
    print(f"  trend_slope: {TRUE_TREND_SLOPE}")
    print(f"  sigma (standardized): {TRUE_SIGMA}")
    print(f"  control_beta: {TRUE_CONTROL_BETA}")
    print(f"\nSales stats: mean={sales.mean():.1f}, std={sales.std():.1f}")
    print(
        f"Media contribution range: [{media_effect.min():.3f}, {media_effect.max():.3f}]"
    )
    print(f"Total media effect (sum): {media_effect.sum():.1f}")
    print("=" * 40 + "\n")

    # Create DataFrame
    df = pd.DataFrame(
        {
            "date": dates,
            "sales": sales,
            "tv_spend": tv_spend,
            "digital_spend": digital_spend,
            "social_spend": social_spend,
            "economic_index": economic_index_raw,
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
            .with_logistic_saturation()
            .with_geometric_adstock()
            .with_display_name("Television")
        )
        .add_media_builder(
            MediaChannelConfigBuilder("Digital")
            .national()
            .with_logistic_saturation()
            .with_geometric_adstock()
            .with_display_name("Digital Marketing")
        )
        .add_media_builder(
            MediaChannelConfigBuilder("Social")
            .national()
            .with_logistic_saturation()
            .with_geometric_adstock()
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
        .bayesian_numpyro()
        .with_chains(4)
        .with_draws(1000)
        .with_tune(2000)
        .with_target_accept(0.99)
        .additive()
        .with_seasonality_builder(SeasonalityConfigBuilder().with_yearly(order=3))
        .build()
    )

    # Configure trend
    trend_config = TrendConfigBuilder().linear().build()

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


def generate_comprehensive_validation_report(
    base_summary: ValidationSummary,
    cv_summary: ValidationSummary | None = None,
    sensitivity_summary: ValidationSummary | None = None,
    stability_summary: ValidationSummary | None = None,
    calibration_summary: ValidationSummary | None = None,
    output_path: str = "full_validation_report.html",
) -> str:
    """
    Generate a comprehensive HTML validation report combining all validation results.

    This merges the basic validation summary with advanced validation results
    (cross-validation, sensitivity, stability, calibration) into a single
    comprehensive report.

    Parameters
    ----------
    base_summary : ValidationSummary
        Base validation results (convergence, PPC, residuals, etc.).
    cv_summary : ValidationSummary, optional
        Cross-validation results.
    sensitivity_summary : ValidationSummary, optional
        Sensitivity analysis results.
    stability_summary : ValidationSummary, optional
        Stability analysis results.
    calibration_summary : ValidationSummary, optional
        Calibration results.
    output_path : str
        Output file path.

    Returns
    -------
    str
        Path to generated report.
    """
    logger.info("Generating comprehensive validation report...")

    # Create a combined summary by merging results
    # Get _full_y_actual from cv_summary if available (needed for CV time-series chart)
    full_y_actual = None
    if cv_summary is not None and cv_summary._full_y_actual is not None:
        full_y_actual = cv_summary._full_y_actual
    elif base_summary._full_y_actual is not None:
        full_y_actual = base_summary._full_y_actual

    combined_summary = ValidationSummary(
        model_name=base_summary.model_name,
        validation_date=base_summary.validation_date,
        # Basic validation components
        convergence=base_summary.convergence,
        ppc=base_summary.ppc,
        residuals=base_summary.residuals,
        channel_diagnostics=base_summary.channel_diagnostics,
        model_comparison=base_summary.model_comparison,
        # Advanced validation components
        cross_validation=(cv_summary.cross_validation if cv_summary else None),
        sensitivity=(sensitivity_summary.sensitivity if sensitivity_summary else None),
        stability=(stability_summary.stability if stability_summary else None),
        calibration=(calibration_summary.calibration if calibration_summary else None),
        # Overall assessment
        overall_quality=base_summary.overall_quality,
        critical_issues=base_summary.critical_issues.copy(),
        warnings=base_summary.warnings.copy(),
        recommendations=base_summary.recommendations.copy(),
        # Full y_actual for CV time-series visualization
        _full_y_actual=full_y_actual,
    )

    # Add recommendations based on advanced validation results
    if cv_summary and cv_summary.cross_validation:
        cv = cv_summary.cross_validation
        if cv.mean_r2 < 0.5:
            combined_summary.warnings.append(
                f"Cross-validation R² is low ({cv.mean_r2:.2f}). "
                "Model may not generalize well to unseen data."
            )
        if cv.mean_coverage < 0.8:
            combined_summary.warnings.append(
                f"Cross-validation coverage is below 80% ({cv.mean_coverage:.1%}). "
                "Credible intervals may be underestimating uncertainty."
            )

    if sensitivity_summary and sensitivity_summary.sensitivity:
        sens = sensitivity_summary.sensitivity
        if sens.sensitive_parameters:
            combined_summary.warnings.append(
                f"{len(sens.sensitive_parameters)} parameter(s) are sensitive to prior choice: "
                f"{', '.join(sens.sensitive_parameters[:3])}{'...' if len(sens.sensitive_parameters) > 3 else ''}. "
                "Consider using more informative priors or collecting more data."
            )

    if stability_summary and stability_summary.stability:
        stab = stability_summary.stability
        if stab.stability_score < 0.7:
            combined_summary.warnings.append(
                f"Stability score is below 0.7 ({stab.stability_score:.2f}). "
                "Results may be sensitive to individual observations."
            )
        if len(stab.influential_observations) > 0:
            combined_summary.recommendations.append(
                f"Review {len(stab.influential_observations)} influential observation(s) "
                "that may disproportionately affect model estimates."
            )

    if calibration_summary and calibration_summary.calibration:
        calib = calibration_summary.calibration
        if not calib.calibrated:
            combined_summary.critical_issues.append(
                "Model is NOT CALIBRATED against lift test experiments. "
                f"Coverage rate: {calib.coverage_rate:.1%}. "
                "Model estimates may not reflect true causal effects."
            )
        else:
            combined_summary.recommendations.append(
                f"Model is calibrated against {len(calib.lift_test_comparisons)} lift test(s) "
                f"with {calib.coverage_rate:.0%} coverage."
            )

    # Update overall quality based on advanced validations
    if combined_summary.critical_issues:
        combined_summary.overall_quality = "poor"
    elif len(combined_summary.warnings) > 3:
        combined_summary.overall_quality = "acceptable"

    # Generate HTML report
    html = combined_summary.to_html_report()

    output_file = Path(output_path)
    output_file.write_text(html)

    logger.info(f"Comprehensive validation report saved to: {output_file.absolute()}")

    return str(output_file.absolute())


# =============================================================================
# Step 7: Cross-Validation for Out-of-Sample Performance
# =============================================================================


def run_cross_validation(model, results) -> ValidationSummary:
    """
    Run time-series cross-validation to assess out-of-sample performance.

    Cross-validation is essential for understanding how well the model
    generalizes to unseen data. This uses an expanding window strategy
    appropriate for time-series data.

    Parameters
    ----------
    model : BayesianMMM
        Fitted model.
    results : MMMResults
        Model results.

    Returns
    -------
    ValidationSummary
        Validation summary with CV results.
    """
    logger.info("=" * 60)
    logger.info("CROSS-VALIDATION")
    logger.info("=" * 60)

    # Configure cross-validation
    # Note: This is computationally expensive as it refits the model for each fold
    config = (
        ValidationConfigBuilder()
        .quick()  # Start with quick base
        .with_cross_validation(
            n_folds=3,  # 3 folds for demonstration (more for production)
            strategy="expanding",  # Expanding window respects time ordering
            min_train_size=52,  # At least 1 year of training data
            gap=0,  # No gap between train and test
        )
        .without_ppc()  # Skip other validations for speed
        .without_residuals()
        .without_channel_diagnostics()
        .build()
    )

    validator = ModelValidator(model, results)

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)

    try:
        summary = validator.validate(config)

        if summary.cross_validation:
            cv = summary.cross_validation
            print(f"\nCV Strategy: {cv.strategy}")
            print(f"Number of Folds: {cv.n_folds}")

            print("\n--- Per-Fold Metrics ---")
            print(cv.summary().to_string(index=False))

            print("\n--- Aggregate Metrics ---")
            print(f"Mean RMSE: {cv.mean_rmse:.2f}")
            print(f"Mean MAE: {cv.mean_mae:.2f}")
            print(f"Mean R²: {cv.mean_r2:.3f}")
            print(f"Mean Coverage: {cv.mean_coverage:.1%}")

        return summary

    except NotImplementedError as e:
        print(f"\n[CV requires model refitting: {e}]")
        print("[Skipping for this demonstration]")
        return None
    except Exception as e:
        print(f"\n[CV error: {e}]")
        return None


# =============================================================================
# Step 8: Sensitivity Analysis for Prior Robustness
# =============================================================================


def run_sensitivity_analysis(model, results) -> ValidationSummary:
    """
    Run sensitivity analysis to test robustness to prior specifications.

    Sensitivity analysis helps understand how sensitive the model's
    conclusions are to the choice of priors. Robust parameters show
    consistent estimates across different prior specifications.

    Parameters
    ----------
    model : BayesianMMM
        Fitted model.
    results : MMMResults
        Model results.

    Returns
    -------
    ValidationSummary
        Validation summary with sensitivity results.
    """
    logger.info("=" * 60)
    logger.info("SENSITIVITY ANALYSIS")
    logger.info("=" * 60)

    # Configure sensitivity analysis
    config = (
        ValidationConfigBuilder()
        .quick()
        .with_sensitivity_analysis(
            prior_multipliers=(0.5, 2.0),  # Test tighter and wider priors
            parameters_of_interest=("beta",),  # Focus on channel effects
            include_specification_tests=False,
        )
        .without_ppc()
        .without_residuals()
        .without_channel_diagnostics()
        .build()
    )

    validator = ModelValidator(model, results)

    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS RESULTS")
    print("=" * 60)

    try:
        summary = validator.validate(config)

        if summary.sensitivity:
            sens = summary.sensitivity
            print("\n--- Base Parameter Estimates ---")
            for param, value in sens.base_estimates.items():
                print(f"  {param}: {value:.4f}")

            print("\n--- Sensitivity Indices ---")
            print("(Lower is more robust; <0.3 considered robust)")
            for param, index in sens.sensitivity_indices.items():
                status = "Robust" if index < 0.3 else "SENSITIVE"
                print(f"  {param}: {index:.3f} [{status}]")

            print(f"\nRobust parameters: {len(sens.robust_parameters)}")
            print(f"Sensitive parameters: {len(sens.sensitive_parameters)}")

            if sens.sensitive_parameters:
                print(
                    "\nWarning: The following parameters are sensitive to prior choice:"
                )
                for p in sens.sensitive_parameters:
                    print(f"  - {p}")

        return summary

    except NotImplementedError as e:
        print(f"\n[Sensitivity analysis requires model refitting: {e}]")
        print("[Skipping for this demonstration]")
        return None
    except Exception as e:
        print(f"\n[Sensitivity analysis error: {e}]")
        return None


# =============================================================================
# Step 9: Stability Analysis for Model Reliability
# =============================================================================


def run_stability_analysis(model, results) -> ValidationSummary:
    """
    Run stability analysis to identify influential observations and assess
    parameter stability via bootstrap.

    Stability analysis helps identify:
    - Influential observations that disproportionately affect estimates
    - Parameter variability under resampling

    Parameters
    ----------
    model : BayesianMMM
        Fitted model.
    results : MMMResults
        Model results.

    Returns
    -------
    ValidationSummary
        Validation summary with stability results.
    """
    logger.info("=" * 60)
    logger.info("STABILITY ANALYSIS")
    logger.info("=" * 60)

    # Configure stability analysis
    # Note: Bootstrap is very expensive; we use a small number for demonstration
    config = (
        ValidationConfigBuilder()
        .quick()
        .with_stability_analysis(
            n_bootstrap=0,  # Skip bootstrap for speed (set >0 for production)
            loo_subset_size=None,  # Use all observations for LOO influence
        )
        .without_ppc()
        .without_residuals()
        .without_channel_diagnostics()
        .with_model_comparison(method="loo")  # Need LOO for influence analysis
        .build()
    )

    validator = ModelValidator(model, results)

    print("\n" + "=" * 60)
    print("STABILITY ANALYSIS RESULTS")
    print("=" * 60)

    try:
        summary = validator.validate(config)

        if summary.stability:
            stab = summary.stability
            print(
                f"\nStability Score: {stab.stability_score:.2f} (0-1, higher is better)"
            )

            n_influential = len(stab.influential_observations)
            print(f"Influential Observations: {n_influential}")

            if n_influential > 0:
                print(f"  Indices: {stab.influential_observations[:10]}")
                if n_influential > 10:
                    print(f"  ... and {n_influential - 10} more")

            if stab.influence_results:
                print("\n--- Influence Diagnostics (via LOO Pareto-k) ---")
                pareto_k = stab.influence_results.observation_influence
                print(f"  Threshold: {stab.influence_results.influence_threshold}")
                print(f"  Max Pareto-k: {pareto_k.max():.3f}")
                print(f"  Mean Pareto-k: {pareto_k.mean():.3f}")

            if stab.bootstrap_results:
                print("\n--- Bootstrap Results ---")
                bootstrap = stab.bootstrap_results
                print(f"  Bootstrap iterations: {bootstrap.n_bootstrap}")
                print("  Parameter stability:")
                for param in list(bootstrap.parameter_means.keys())[:5]:
                    mean = bootstrap.parameter_means[param]
                    std = bootstrap.parameter_stds[param]
                    print(f"    {param}: {mean:.4f} ± {std:.4f}")

        return summary

    except NotImplementedError as e:
        print(f"\n[Stability analysis error: {e}]")
        return None
    except Exception as e:
        print(f"\n[Stability analysis error: {e}]")
        return None


# =============================================================================
# Step 10: Calibration Against Lift Tests
# =============================================================================


def run_calibration(model, results, data) -> ValidationSummary:
    """
    Run calibration validation against external lift test results.

    Calibration compares the model's channel contribution estimates to
    experimentally measured lift values. This is the gold standard for
    validating causal claims in MMM.

    Parameters
    ----------
    model : BayesianMMM
        Fitted model.
    results : MMMResults
        Model results.
    data : pd.DataFrame
        Original data (for date reference).

    Returns
    -------
    ValidationSummary
        Validation summary with calibration results.
    """
    logger.info("=" * 60)
    logger.info("CALIBRATION VALIDATION")
    logger.info("=" * 60)

    # Create synthetic lift test results for demonstration
    # In practice, these would come from actual experiments
    dates = data["date"].tolist()
    mid_point = len(dates) // 2

    lift_tests = [
        LiftTestResult(
            channel="TV",  # Must match channel name in model
            test_period=(
                str(dates[mid_point].date()),
                str(dates[mid_point + 12].date()),
            ),
            measured_lift=30.0,  # Synthetic lift value
            lift_se=10.0,  # Standard error
            confidence_level=0.95,
        ),
        LiftTestResult(
            channel="Digital",
            test_period=(
                str(dates[mid_point + 15].date()),
                str(dates[mid_point + 25].date()),
            ),
            measured_lift=35.0,
            lift_se=13.0,
            confidence_level=0.95,
        ),
    ]

    # Configure calibration
    config = (
        ValidationConfigBuilder()
        .quick()
        .with_calibration(
            lift_tests=lift_tests,
            ci_level=0.94,  # 94% credible interval
            tolerance_multiplier=1.5,
        )
        .without_ppc()
        .without_residuals()
        .without_channel_diagnostics()
        .build()
    )

    validator = ModelValidator(model, results)

    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)

    try:
        summary = validator.validate(config)

        if summary.calibration:
            calib = summary.calibration
            print(
                f"\nCalibration Status: {'CALIBRATED' if calib.calibrated else 'NOT CALIBRATED'}"
            )
            print(f"Coverage Rate: {calib.coverage_rate:.1%}")
            print(
                f"Mean Absolute Calibration Error: {calib.mean_absolute_calibration_error:.1%}"
            )

            print("\n--- Lift Test Comparisons ---")
            print(calib.summary().to_string(index=False))

            for comp in calib.lift_test_comparisons:
                status = "PASS" if comp.within_ci else "FAIL"
                print(f"\n  {comp.channel}:")
                print(f"    Model estimate: {comp.model_estimate:,.0f}")
                print(
                    f"    Model 94% CI: [{comp.model_ci_low:,.0f}, {comp.model_ci_high:,.0f}]"
                )
                print(
                    f"    Experimental: {comp.experimental_estimate:,.0f} ± {comp.experimental_se:,.0f}"
                )
                print(f"    Within CI: {status}")
                print(f"    Relative Error: {comp.relative_error:+.1%}")

        return summary

    except Exception as e:
        print(f"\n[Calibration error: {e}]")
        print("[This may occur if channel names don't match or dates are out of range]")
        return None


# =============================================================================
# Step 11: Demonstrate Full Thorough Validation
# =============================================================================


def demonstrate_thorough_validation_setup():
    """
    Demonstrate how to configure a thorough validation that includes all
    advanced validation methods.

    Note: Thorough validation is computationally expensive and may take
    30+ minutes depending on model complexity and data size.
    """
    print("\n" + "=" * 60)
    print("THOROUGH VALIDATION CONFIGURATION")
    print("=" * 60)

    # Example lift test results from experiments
    lift_tests = [
        LiftTestResult(
            channel="TV",
            test_period=("2023-06-01", "2023-08-31"),
            measured_lift=15000,
            lift_se=3000,
            holdout_regions=["Northeast", "Southeast"],
            confidence_level=0.95,
        ),
        LiftTestResult(
            channel="Digital",
            test_period=("2023-09-01", "2023-10-31"),
            measured_lift=8000,
            lift_se=2000,
            confidence_level=0.95,
        ),
    ]

    # Full thorough configuration
    config = (
        ValidationConfigBuilder()
        .thorough()  # Enables CV, sensitivity, and stability
        # Customize cross-validation
        .with_cross_validation(
            n_folds=5,
            strategy="expanding",
            min_train_size=52,
        )
        # Customize sensitivity analysis
        .with_sensitivity_analysis(
            prior_multipliers=(0.5, 1.0, 2.0),
            parameters_of_interest=("beta", "sat_lam", "adstock"),
        )
        # Customize stability analysis
        .with_stability_analysis(
            n_bootstrap=50,  # More iterations for better estimates
            perturbation_level=0.1,
        )
        # Add calibration if lift tests available
        .with_calibration(
            lift_tests=lift_tests,
            ci_level=0.94,
            tolerance_multiplier=1.5,
        )
        .build()
    )

    print("\nThorough Validation includes:")
    print("  [x] Convergence diagnostics")
    print("  [x] Posterior predictive checks")
    print("  [x] Residual diagnostics")
    print("  [x] Channel diagnostics (VIF, correlation)")
    print("  [x] Model comparison (LOO-CV, WAIC)")
    print("  [x] Time-series cross-validation")
    print("  [x] Prior sensitivity analysis")
    print("  [x] Stability analysis (LOO influence + bootstrap)")
    print("  [x] Calibration against lift tests")

    print("\nConfiguration Summary:")
    print(f"  Level: {config.level.value}")
    print(f"  CV Folds: {config.cross_validation.n_folds}")
    print(f"  CV Strategy: {config.cross_validation.strategy}")
    print(f"  Prior Multipliers: {config.sensitivity.prior_multipliers}")
    print(f"  Bootstrap Iterations: {config.stability.n_bootstrap}")
    print(f"  Lift Tests: {len(config.lift_tests)}")

    print("\n  [Note: Running thorough validation requires significant compute time]")
    print("  [Use ValidationConfig.standard() for faster iteration during development]")

    return config


# =============================================================================
# Main Entry Point
# =============================================================================


def main(run_expensive_validations: bool = False):
    """
    Run the complete validation workflow example.

    Parameters
    ----------
    run_expensive_validations : bool
        If True, run cross-validation, sensitivity, and stability analyses.
        These are computationally expensive and require model refitting.
        Default is False for faster demonstration.
    """
    print("=" * 70)
    print("MMM FRAMEWORK - MODEL VALIDATION EXAMPLE")
    print("=" * 70)

    # Generate synthetic data
    logger.info("Generating synthetic marketing data...")
    data = generate_synthetic_data(n_weeks=156, seed=421)
    print(f"\nGenerated {len(data)} weeks of data")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"Media channels: TV, Digital, Social")
    print(f"Control: Economic Index")

    # Fit model
    model, results, panel = configure_and_fit_model(data)

    # Run basic validations (fast)
    quick_summary = run_quick_validation(model, results)
    standard_summary = run_standard_validation(model, results)
    custom_summary = run_custom_validation(model, results)

    # Generate report
    report_path = generate_validation_report(
        standard_summary,
        output_path="validation_report.html",
    )

    # Run advanced validations (these require model refitting)
    cv_summary = None
    sensitivity_summary = None
    stability_summary = None
    calibration_summary = None

    full_report_path = None

    if run_expensive_validations:
        print("\n" + "=" * 70)
        print("RUNNING ADVANCED VALIDATIONS (This may take a while...)")
        print("=" * 70)

        # Cross-validation
        cv_summary = run_cross_validation(model, results)

        # Sensitivity analysis
        sensitivity_summary = run_sensitivity_analysis(model, results)

        # Stability analysis
        stability_summary = run_stability_analysis(model, results)

        # Calibration
        calibration_summary = run_calibration(model, results, data)

        # Generate comprehensive HTML report with all validation results
        print("\n" + "=" * 70)
        print("GENERATING COMPREHENSIVE VALIDATION REPORT")
        print("=" * 70)

        full_report_path = generate_comprehensive_validation_report(
            base_summary=standard_summary,
            cv_summary=cv_summary,
            sensitivity_summary=sensitivity_summary,
            stability_summary=stability_summary,
            calibration_summary=calibration_summary,
            output_path="full_validation_report.html",
        )
        print(f"\nComprehensive report generated: {full_report_path}")
    else:
        print("\n" + "=" * 70)
        print("SKIPPING EXPENSIVE VALIDATIONS")
        print("=" * 70)
        print("\nTo run cross-validation, sensitivity, stability, and calibration:")
        print("  main(run_expensive_validations=True)")
        print("\nOr configure individual validations:")

    # Show thorough validation configuration
    demonstrate_thorough_validation_setup()

    # Final summary
    print("\n" + "=" * 70)
    print("VALIDATION WORKFLOW COMPLETE")
    print("=" * 70)
    print(f"\nQuick Validation Quality: {quick_summary.overall_quality}")
    print(f"Standard Validation Quality: {standard_summary.overall_quality}")
    print(f"Custom Validation Quality: {custom_summary.overall_quality}")

    if run_expensive_validations:
        if cv_summary and cv_summary.cross_validation:
            print(
                f"Cross-Validation Mean R²: {cv_summary.cross_validation.mean_r2:.3f}"
            )
        if sensitivity_summary and sensitivity_summary.sensitivity:
            n_robust = len(sensitivity_summary.sensitivity.robust_parameters)
            n_sens = len(sensitivity_summary.sensitivity.sensitive_parameters)
            print(f"Sensitivity: {n_robust} robust, {n_sens} sensitive parameters")
        if stability_summary and stability_summary.stability:
            print(f"Stability Score: {stability_summary.stability.stability_score:.2f}")
        if calibration_summary and calibration_summary.calibration:
            print(
                f"Calibration Coverage: {calibration_summary.calibration.coverage_rate:.1%}"
            )

    print(f"\nBasic HTML Report: {report_path}")
    if full_report_path:
        print(f"Full HTML Report (with all validations): {full_report_path}")

    print("\n" + "-" * 70)
    print("KEY TAKEAWAYS:")
    print("-" * 70)
    print("""
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

    4. **NEW** Cross-validation assesses out-of-sample performance
       - Time-series CV respects temporal ordering
       - Supports expanding, rolling, and blocked strategies
       - Reports RMSE, MAE, MAPE, R², and coverage per fold

    5. **NEW** Sensitivity analysis tests prior robustness
       - Refits model with scaled prior variances (0.5x, 2x)
       - Computes sensitivity index (coefficient of variation)
       - Classifies parameters as robust or sensitive

    6. **NEW** Stability analysis identifies reliability issues
       - LOO influence diagnostics via Pareto-k values
       - Optional parametric bootstrap for parameter stability
       - Reports overall stability score (0-1)

    7. **NEW** Calibration validates against experiments
       - Compares model estimates to lift test results
       - Checks if experimental values fall within model CI
       - Reports coverage rate and calibration error

    8. Validation results guide model improvement
       - Critical issues require immediate attention
       - Warnings suggest areas for investigation
       - Recommendations provide actionable next steps

    9. HTML reports facilitate stakeholder communication
       - Self-contained validation summary
       - Includes all diagnostics and recommendations
    """)

    return {
        "quick": quick_summary,
        "standard": standard_summary,
        "custom": custom_summary,
        "cross_validation": cv_summary,
        "sensitivity": sensitivity_summary,
        "stability": stability_summary,
        "calibration": calibration_summary,
        "basic_report_path": report_path,
        "full_report_path": full_report_path,
    }


if __name__ == "__main__":
    # Run with basic validations only (fast)
    # main(run_expensive_validations=False)

    # Uncomment to run all validations including expensive ones:
    main(run_expensive_validations=True)
