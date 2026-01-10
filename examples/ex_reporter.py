"""
Example usage of the MMM Reporting module.

This script demonstrates how to generate reports from:
1. BayesianMMM (core framework)
2. Extended MMM models (nested, multivariate)
3. Manual data bundles (for custom integrations)
4. PyMC-Marketing MMM class

Run with:
    python examples/ex_reporting.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import reporting module
from mmm_framework.reporting import (
    MMMReportGenerator,
    ReportConfig,
    SectionConfig,
    MMMDataBundle,
    ReportBuilder,
    ColorScheme,
    ColorPalette,
)
from mmm_framework.reporting.data_extractors import MMMDataBundle


def create_synthetic_data_bundle() -> MMMDataBundle:
    """
    Create a synthetic data bundle for demonstration.
    
    In real usage, this would be extracted automatically from your fitted model.
    """
    np.random.seed(42)
    n_weeks = 156  # 3 years
    
    # Generate dates
    dates = pd.date_range("2023-01-01", periods=n_weeks, freq="W-MON")
    
    # Generate actual revenue with trend, seasonality, and noise
    trend = np.linspace(14, 18, n_weeks)
    seasonality = 2 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
    noise = np.random.normal(0, 0.5, n_weeks)
    actual = trend + seasonality + noise
    
    # Generate predictions with uncertainty
    pred_mean = actual + np.random.normal(0, 0.2, n_weeks)
    pred_std = 0.5 + 0.2 * np.random.random(n_weeks)
    pred_lower = pred_mean - 1.28 * pred_std
    pred_upper = pred_mean + 1.28 * pred_std
    
    # Channel data
    channels = ["TV", "Paid_Search", "Paid_Social", "Display", "Radio"]
    
    # ROI with uncertainty (TV highest, Radio lowest)
    channel_roi = {
        "TV": {"mean": 1.45, "lower": 1.10, "upper": 1.82},
        "Paid_Search": {"mean": 2.12, "lower": 1.85, "upper": 2.41},
        "Paid_Social": {"mean": 1.78, "lower": 1.32, "upper": 2.28},
        "Display": {"mean": 1.15, "lower": 0.65, "upper": 1.68},
        "Radio": {"mean": 0.82, "lower": 0.45, "upper": 1.22},
    }
    
    # Component totals for waterfall
    component_totals = {
        "Baseline": 680.0,
        "TV": 45.2,
        "Paid_Search": 32.1,
        "Paid_Social": 28.5,
        "Display": 12.8,
        "Radio": 5.4,
    }
    
    # Component time series for stacked area
    component_ts = {}
    for comp, total in component_totals.items():
        if comp == "Baseline":
            component_ts[comp] = trend + np.random.normal(0, 0.1, n_weeks)
        else:
            base = total / n_weeks
            component_ts[comp] = base + base * 0.3 * np.sin(2 * np.pi * np.arange(n_weeks) / 52 + np.random.random() * np.pi)
            component_ts[comp] = np.maximum(component_ts[comp], 0)
    
    # Saturation curves
    saturation_curves = {}
    for ch in channels:
        spend_range = np.linspace(0, 100000, 100)
        k = np.random.uniform(30000, 60000)  # Half-saturation point
        s = np.random.uniform(1.2, 2.5)  # Slope
        response = spend_range ** s / (k ** s + spend_range ** s)
        saturation_curves[ch] = {"spend": spend_range, "response": response * 10000}
    
    # Adstock curves
    adstock_curves = {}
    for ch in channels:
        alpha = np.random.uniform(0.3, 0.8)
        l_max = np.random.randint(4, 10)
        lags = np.arange(l_max)
        weights = alpha ** lags
        adstock_curves[ch] = weights / weights.sum()
    
    # Current spend levels
    current_spend = {
        "TV": 45000,
        "Paid_Search": 28000,
        "Paid_Social": 22000,
        "Display": 15000,
        "Radio": 8000,
    }
    
    # Fit statistics
    fit_statistics = {
        "r2": 0.92,
        "rmse": 0.48,
        "mae": 0.35,
        "mape": 0.023,
    }
    
    # Diagnostics
    diagnostics = {
        "divergences": 0,
        "rhat_max": 1.002,
        "ess_bulk_min": 1250,
        "ess_tail_min": 980,
    }
    
    # Summary metrics
    total_revenue = float(actual.sum() * 1e6)  # Scale to millions
    marketing_contribution = sum(v for k, v in component_totals.items() if k != "Baseline")
    
    return MMMDataBundle(
        dates=dates,
        actual=actual * 1e6,
        predicted={
            "mean": pred_mean * 1e6,
            "lower": pred_lower * 1e6,
            "upper": pred_upper * 1e6,
        },
        fit_statistics=fit_statistics,
        total_revenue=total_revenue,
        marketing_attributed_revenue={
            "mean": marketing_contribution * 1e6,
            "lower": marketing_contribution * 0.78 * 1e6,
            "upper": marketing_contribution * 1.23 * 1e6,
        },
        blended_roi={
            "mean": 1.26,
            "lower": 0.96,
            "upper": 1.58,
        },
        marketing_contribution_pct={
            "mean": 0.137,
            "lower": 0.106,
            "upper": 0.169,
        },
        channel_roi=channel_roi,
        channel_names=channels,
        channel_spend=current_spend,
        component_totals={k: v * 1e6 for k, v in component_totals.items()},
        component_time_series={k: v * 1e6 for k, v in component_ts.items()},
        saturation_curves=saturation_curves,
        adstock_curves=adstock_curves,
        current_spend=current_spend,
        diagnostics=diagnostics,
        model_specification={
            "likelihood": "Normal with estimated scale",
            "baseline": "Linear trend + Fourier seasonality (order 3)",
            "media_effects": "Hill saturation × Geometric adstock",
            "controls": "Holidays, weather, promotional indicators",
            "priors": "Weakly informative, documented in technical appendix",
            "chains": 4,
            "draws": 2000,
            "tune": 1000,
        },
    )


def example_basic_report():
    """Generate a basic report with default settings."""
    print("=" * 60)
    print("Example 1: Basic Report")
    print("=" * 60)
    
    # Create synthetic data
    data = create_synthetic_data_bundle()
    
    # Create report generator
    report = MMMReportGenerator(
        data=data,
        config=ReportConfig(
            title="Marketing Mix Model Report",
            client="Acme Consumer Products",
            subtitle="North America",
            analysis_period="Jan 2023 – Dec 2025",
        ),
    )
    
    # Save report
    output_path = Path("mmm_report_basic.html")
    report.to_html(output_path)
    print(f"Report saved to: {output_path}")
    
    return output_path


def example_custom_sections():
    """Generate a report with custom section configuration."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Section Configuration")
    print("=" * 60)
    
    data = create_synthetic_data_bundle()
    
    # Create config with custom sections
    config = ReportConfig(
        title="Executive MMM Summary",
        client="Acme Corp",
        analysis_period="2025",
        # Disable technical sections
        model_fit=SectionConfig(enabled=False),
        diagnostics=SectionConfig(enabled=False),
        # Customize others
        executive_summary=SectionConfig(
            enabled=True,
            credible_interval=0.9,  # Use 90% CI
        ),
        channel_roi=SectionConfig(
            enabled=True,
            title="Channel Performance Analysis",
            subtitle="ROI estimates with uncertainty quantification",
        ),
        methodology=SectionConfig(
            enabled=True,
            custom_notes="Contact the data science team for technical documentation.",
        ),
    )
    
    report = MMMReportGenerator(data=data, config=config)
    
    output_path = Path("mmm_report_executive.html")
    report.to_html(output_path)
    print(f"Report saved to: {output_path}")
    
    return output_path


def example_builder_pattern():
    """Use the builder pattern for fluent configuration."""
    print("\n" + "=" * 60)
    print("Example 3: Fluent Builder Pattern")
    print("=" * 60)
    
    data = create_synthetic_data_bundle()
    
    # Use builder for fluent API
    report = (
        ReportBuilder()
        .with_data(data)
        .with_title("Q4 2025 Marketing Analysis")
        .with_client("Acme Corp")
        .with_analysis_period("October - December 2025")
        .with_credible_interval(0.8)
        .enable_all_sections()
        .disable_section("diagnostics")
        .build()
    )
    
    output_path = Path("mmm_report_builder.html")
    report.to_html(output_path)
    print(f"Report saved to: {output_path}")
    
    return output_path


def example_color_schemes():
    """Generate reports with different color schemes."""
    print("\n" + "=" * 60)
    print("Example 4: Color Scheme Customization")
    print("=" * 60)
    
    data = create_synthetic_data_bundle()
    
    for palette in [ColorPalette.SAGE, ColorPalette.CORPORATE, ColorPalette.WARM]:
        config = ReportConfig(
            title=f"MMM Report ({palette.value.title()} Theme)",
            client="Demo",
            color_scheme=ColorScheme.from_palette(palette),
            # Minimal for quick demo
            model_fit=SectionConfig(enabled=False),
            saturation=SectionConfig(enabled=False),
            sensitivity=SectionConfig(enabled=False),
            diagnostics=SectionConfig(enabled=False),
        )
        
        report = MMMReportGenerator(data=data, config=config)
        output_path = Path(f"mmm_report_{palette.value}.html")
        report.to_html(output_path)
        print(f"  {palette.value}: {output_path}")


def example_minimal_report():
    """Generate a minimal report for quick stakeholder updates."""
    print("\n" + "=" * 60)
    print("Example 5: Minimal Stakeholder Report")
    print("=" * 60)
    
    data = create_synthetic_data_bundle()
    
    report = (
        ReportBuilder()
        .with_data(data)
        .with_title("Marketing Performance Summary")
        .with_client("Acme Corp")
        .minimal_report()  # Only executive summary, ROI, methodology
        .build()
    )
    
    output_path = Path("mmm_report_minimal.html")
    report.to_html(output_path)
    print(f"Report saved to: {output_path}")
    
    return output_path


def example_with_model():
    """
    Example showing integration with BayesianMMM model.
    
    Note: This requires a fitted BayesianMMM instance.
    Uncomment and adapt for your actual model.
    """
    print("\n" + "=" * 60)
    print("Example 6: Integration with BayesianMMM (Template)")
    print("=" * 60)
    
    print("""
    # Template for integrating with your fitted model:
    
    from mmm_framework import BayesianMMM, load_mff
    from mmm_reporting import MMMReportGenerator, ReportConfig
    
    # Load data and fit model
    panel = load_mff("your_data.csv", mff_config)
    mmm = BayesianMMM(panel, model_config, trend_config)
    results = mmm.fit()
    
    # Generate report directly from model
    report = MMMReportGenerator(
        model=mmm,
        panel=panel,
        results=results,
        config=ReportConfig(
            title="Q4 Marketing Analysis",
            client="Your Client",
        ),
    )
    
    report.to_html("mmm_report.html")
    """)


if __name__ == "__main__":
    # Run all examples
    example_basic_report()
    example_custom_sections()
    example_builder_pattern()
    example_color_schemes()
    example_minimal_report()
    example_with_model()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)