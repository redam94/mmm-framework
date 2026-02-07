"""
Extended MMM Report Example

Generates a comprehensive report for a multi-geo nested MMM with:
- 5 geographic regions with differential media response
- Nested mediator effects (Awareness → Consideration → Purchase)
- Cross-product cannibalization between 3 product lines

This demonstrates the full capabilities of the mmm_reporting module
for complex hierarchical marketing mix models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmm_framework.reporting import (
    MMMReportGenerator,
    ReportConfig,
    SectionConfig,
    ColorScheme,
    ColorPalette,
)
from mmm_framework.reporting.data_extractors import MMMDataBundle


def create_extended_mmm_data() -> MMMDataBundle:
    """
    Create synthetic data for a multi-geo nested MMM with cannibalization.

    Scenario: National CPG brand with 3 product lines across 5 regions
    - Products: Core, Premium, Value
    - Regions: Northeast, Southeast, Midwest, Southwest, West
    - Mediators: Awareness, Consideration (nested model)
    - Channels: TV, Paid Search, Paid Social, Display, Radio
    """
    np.random.seed(2025)

    n_weeks = 104  # 2 years

    # =========================================================================
    # Basic time series data
    # =========================================================================
    dates = pd.date_range("2024-01-01", periods=n_weeks, freq="W-MON")

    # National-level actual revenue (sum across products/geos)
    trend = np.linspace(22, 28, n_weeks)
    seasonality = 3 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
    holiday_bump = np.zeros(n_weeks)
    holiday_bump[46:52] = np.array([1, 2, 4, 3, 2, 1])  # Holiday season
    noise = np.random.normal(0, 0.4, n_weeks)
    actual = trend + seasonality + holiday_bump + noise

    # Predictions with uncertainty
    pred_mean = actual + np.random.normal(0, 0.15, n_weeks)
    pred_std = 0.6 + 0.15 * np.random.random(n_weeks)
    pred_lower = pred_mean - 1.28 * pred_std
    pred_upper = pred_mean + 1.28 * pred_std

    # =========================================================================
    # Channel configuration
    # =========================================================================
    channels = ["TV", "Paid_Search", "Paid_Social", "Display", "Radio"]

    # National-level ROI with uncertainty
    channel_roi = {
        "TV": {"mean": 1.82, "lower": 1.45, "upper": 2.21},
        "Paid_Search": {"mean": 2.45, "lower": 2.12, "upper": 2.81},
        "Paid_Social": {"mean": 1.95, "lower": 1.48, "upper": 2.45},
        "Display": {"mean": 1.28, "lower": 0.78, "upper": 1.82},
        "Radio": {"mean": 0.92, "lower": 0.52, "upper": 1.35},
    }

    channel_spend = {
        "TV": 2_800_000,
        "Paid_Search": 1_950_000,
        "Paid_Social": 1_650_000,
        "Display": 980_000,
        "Radio": 420_000,
    }

    # =========================================================================
    # Component decomposition
    # =========================================================================
    component_totals = {
        "Baseline": 1_850_000_000,
        "Trend": 180_000_000,
        "Seasonality": 95_000_000,
        "TV": 245_000_000,
        "Paid_Search": 168_000_000,
        "Paid_Social": 142_000_000,
        "Display": 58_000_000,
        "Radio": 22_000_000,
    }

    # Time series for each component
    component_ts = {}
    for comp, total in component_totals.items():
        weekly_avg = total / n_weeks
        if comp == "Baseline":
            component_ts[comp] = np.full(n_weeks, weekly_avg)
        elif comp == "Trend":
            component_ts[comp] = np.linspace(0.7, 1.3, n_weeks) * weekly_avg
        elif comp == "Seasonality":
            component_ts[comp] = weekly_avg * (
                1 + 0.8 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
            )
        else:
            # Media channels: some seasonality + random variation
            phase = np.random.random() * 2 * np.pi
            component_ts[comp] = weekly_avg * (
                1 + 0.25 * np.sin(2 * np.pi * np.arange(n_weeks) / 52 + phase)
            )
            component_ts[comp] += np.random.normal(0, weekly_avg * 0.1, n_weeks)
            component_ts[comp] = np.maximum(component_ts[comp], 0)

    # =========================================================================
    # Saturation curves
    # =========================================================================
    saturation_curves = {}
    saturation_params = {
        "TV": {"k": 4_500_000, "s": 1.8},
        "Paid_Search": {"k": 2_800_000, "s": 2.2},
        "Paid_Social": {"k": 2_200_000, "s": 1.9},
        "Display": {"k": 1_500_000, "s": 1.5},
        "Radio": {"k": 800_000, "s": 1.4},
    }

    for ch, params in saturation_params.items():
        max_spend = channel_spend[ch] * 2.5
        spend_range = np.linspace(0, max_spend, 100)
        k, s = params["k"], params["s"]
        response = spend_range**s / (k**s + spend_range**s)
        # Scale response to realistic revenue
        max_response = component_totals[ch] * 1.5 / n_weeks
        saturation_curves[ch] = {
            "spend": spend_range,
            "response": response * max_response,
        }

    # Adstock curves
    adstock_curves = {}
    adstock_params = {
        "TV": {"alpha": 0.65, "l_max": 8},
        "Paid_Search": {"alpha": 0.25, "l_max": 3},
        "Paid_Social": {"alpha": 0.45, "l_max": 5},
        "Display": {"alpha": 0.35, "l_max": 4},
        "Radio": {"alpha": 0.55, "l_max": 6},
    }

    for ch, params in adstock_params.items():
        lags = np.arange(params["l_max"])
        weights = params["alpha"] ** lags
        adstock_curves[ch] = weights / weights.sum()

    # =========================================================================
    # Geographic data (5 regions)
    # =========================================================================
    geo_names = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]

    # Regional performance metrics
    # Each metric can be a dict with mean/lower/upper for uncertainty, or simple values
    geo_performance = {
        "Northeast": {
            "revenue": {
                "mean": 580_000_000,
                "lower": 520_000_000,
                "upper": 640_000_000,
            },
            "blended_roi": {"mean": 1.42, "lower": 1.15, "upper": 1.72},
            "marketing_contribution_pct": {"mean": 0.22, "lower": 0.18, "upper": 0.26},
            "yoy_growth": 0.08,
        },
        "Southeast": {
            "revenue": {
                "mean": 520_000_000,
                "lower": 465_000_000,
                "upper": 575_000_000,
            },
            "blended_roi": {"mean": 1.58, "lower": 1.28, "upper": 1.88},
            "marketing_contribution_pct": {"mean": 0.24, "lower": 0.19, "upper": 0.29},
            "yoy_growth": 0.12,
        },
        "Midwest": {
            "revenue": {
                "mean": 490_000_000,
                "lower": 435_000_000,
                "upper": 545_000_000,
            },
            "blended_roi": {"mean": 1.25, "lower": 0.98, "upper": 1.55},
            "marketing_contribution_pct": {"mean": 0.19, "lower": 0.15, "upper": 0.23},
            "yoy_growth": 0.05,
        },
        "Southwest": {
            "revenue": {
                "mean": 580_000_000,
                "lower": 515_000_000,
                "upper": 645_000_000,
            },
            "blended_roi": {"mean": 1.72, "lower": 1.38, "upper": 2.08},
            "marketing_contribution_pct": {"mean": 0.26, "lower": 0.21, "upper": 0.31},
            "yoy_growth": 0.15,
        },
        "West": {
            "revenue": {
                "mean": 590_000_000,
                "lower": 525_000_000,
                "upper": 655_000_000,
            },
            "blended_roi": {"mean": 1.55, "lower": 1.25, "upper": 1.88},
            "marketing_contribution_pct": {"mean": 0.23, "lower": 0.18, "upper": 0.28},
            "yoy_growth": 0.10,
        },
    }

    # Regional ROI by channel (differential media response)
    # Note: These vary by region due to different audience compositions
    geo_roi = {}
    roi_multipliers = {
        "Northeast": {
            "TV": 0.9,
            "Paid_Search": 1.1,
            "Paid_Social": 0.95,
            "Display": 1.0,
            "Radio": 0.8,
        },
        "Southeast": {
            "TV": 1.15,
            "Paid_Search": 0.95,
            "Paid_Social": 1.1,
            "Display": 0.9,
            "Radio": 1.2,
        },
        "Midwest": {
            "TV": 1.2,
            "Paid_Search": 0.85,
            "Paid_Social": 0.8,
            "Display": 0.85,
            "Radio": 1.35,
        },
        "Southwest": {
            "TV": 1.0,
            "Paid_Search": 1.25,
            "Paid_Social": 1.2,
            "Display": 1.15,
            "Radio": 0.9,
        },
        "West": {
            "TV": 0.85,
            "Paid_Search": 1.3,
            "Paid_Social": 1.25,
            "Display": 1.2,
            "Radio": 0.7,
        },
    }

    for geo in geo_names:
        geo_roi[geo] = {}
        for ch in channels:
            mult = roi_multipliers[geo][ch]
            base = channel_roi[ch]
            # Apply multiplier and adjust uncertainty
            geo_roi[geo][ch] = {
                "mean": base["mean"] * mult,
                "lower": base["lower"] * mult * 0.9,
                "upper": base["upper"] * mult * 1.1,
            }

    # Regional decomposition
    geo_contribution = {}
    geo_shares = {
        "Northeast": 0.21,
        "Southeast": 0.19,
        "Midwest": 0.18,
        "Southwest": 0.21,
        "West": 0.21,
    }

    for geo in geo_names:
        share = geo_shares[geo]
        geo_contribution[geo] = {
            comp: total * share * (0.9 + 0.2 * np.random.random())
            for comp, total in component_totals.items()
        }

    # =========================================================================
    # Mediator pathway data (Nested model)
    # =========================================================================
    mediator_names = ["Awareness", "Consideration"]

    # Effect pathways: how each channel affects sales through mediators
    # Structure: {channel: {mediator: indirect_effect, "_direct": direct_effect, "_total": total}}
    mediator_pathways = {}

    # TV has strongest awareness pathway
    mediator_pathways["TV"] = {
        "Awareness": 0.42,  # 42% of TV effect goes through awareness
        "Consideration": 0.18,  # 18% through consideration
        "_direct": 0.40,  # 40% direct to purchase
        "_total": 1.0,
    }

    # Paid Search: mostly direct (high-intent)
    mediator_pathways["Paid_Search"] = {
        "Awareness": 0.08,
        "Consideration": 0.12,
        "_direct": 0.80,  # 80% direct (high intent)
        "_total": 1.0,
    }

    # Paid Social: balanced
    mediator_pathways["Paid_Social"] = {
        "Awareness": 0.35,
        "Consideration": 0.25,
        "_direct": 0.40,
        "_total": 1.0,
    }

    # Display: mostly awareness
    mediator_pathways["Display"] = {
        "Awareness": 0.55,
        "Consideration": 0.15,
        "_direct": 0.30,
        "_total": 1.0,
    }

    # Radio: awareness-focused
    mediator_pathways["Radio"] = {
        "Awareness": 0.48,
        "Consideration": 0.22,
        "_direct": 0.30,
        "_total": 1.0,
    }

    # Mediator time series (weekly awareness and consideration indices)
    mediator_time_series = {
        "Awareness": 45
        + 15 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
        + np.random.normal(0, 3, n_weeks),
        "Consideration": 28
        + 8 * np.sin(2 * np.pi * np.arange(n_weeks) / 52 + 0.5)
        + np.random.normal(0, 2, n_weeks),
    }

    # Ensure values stay in reasonable range
    mediator_time_series["Awareness"] = np.clip(
        mediator_time_series["Awareness"], 20, 80
    )
    mediator_time_series["Consideration"] = np.clip(
        mediator_time_series["Consideration"], 10, 50
    )

    # Total indirect effect through mediators
    total_indirect_effect = {
        "mean": 0.58,  # 58% of total media effect is indirect
        "lower": 0.48,
        "upper": 0.68,
    }

    # =========================================================================
    # Product cannibalization data (3 product lines)
    # =========================================================================
    product_names = ["Core", "Premium", "Value"]

    # Cannibalization matrix: effect of marketing Product A on sales of Product B
    # Negative = cannibalization, Positive = halo effect
    # Diagonal = own-product lift (always positive)
    cannibalization_matrix = {
        "Core": {
            "Core": {
                "mean": 1.00,
                "lower": 0.85,
                "upper": 1.15,
            },  # Own effect (normalized)
            "Premium": {
                "mean": -0.12,
                "lower": -0.22,
                "upper": -0.02,
            },  # Cannibalizes Premium
            "Value": {
                "mean": -0.08,
                "lower": -0.18,
                "upper": 0.02,
            },  # Slight cannib of Value
        },
        "Premium": {
            "Core": {
                "mean": -0.18,
                "lower": -0.28,
                "upper": -0.08,
            },  # Cannibalizes Core strongly
            "Premium": {"mean": 1.00, "lower": 0.82, "upper": 1.18},  # Own effect
            "Value": {
                "mean": 0.05,
                "lower": -0.05,
                "upper": 0.15,
            },  # Slight halo to Value
        },
        "Value": {
            "Core": {
                "mean": 0.08,
                "lower": -0.02,
                "upper": 0.18,
            },  # Slight halo to Core
            "Premium": {
                "mean": 0.02,
                "lower": -0.08,
                "upper": 0.12,
            },  # Negligible effect
            "Value": {"mean": 1.00, "lower": 0.88, "upper": 1.12},  # Own effect
        },
    }

    # Net product effects after cannibalization
    net_product_effects = {
        "Core": {
            "direct": 185_000_000,
            "cannibalization": -28_000_000,  # Lost to Premium/Value marketing
            "net": 157_000_000,
        },
        "Premium": {
            "direct": 165_000_000,
            "cannibalization": -42_000_000,  # Lost to Core marketing
            "net": 123_000_000,
        },
        "Value": {
            "direct": 95_000_000,
            "cannibalization": 12_000_000,  # Net positive from halo effects
            "net": 107_000_000,
        },
    }

    # =========================================================================
    # Model diagnostics
    # =========================================================================
    diagnostics = {
        "divergences": 0,
        "rhat_max": 1.003,
        "ess_bulk_min": 1850,
        "ess_tail_min": 1420,
        "bfmi": 0.92,
        "n_chains": 4,
        "n_draws": 2000,
        "n_tune": 1500,
    }

    # Fit statistics
    fit_statistics = {
        "r2": 0.94,
        "rmse": 0.38,
        "mae": 0.28,
        "mape": 0.018,
    }

    # =========================================================================
    # Summary metrics
    # =========================================================================
    total_revenue = float(actual.sum() * 1e8)  # Scale appropriately
    marketing_total = sum(
        v
        for k, v in component_totals.items()
        if k not in ["Baseline", "Trend", "Seasonality"]
    )

    # =========================================================================
    # Assemble data bundle
    # =========================================================================
    return MMMDataBundle(
        # Time series
        dates=dates,
        actual=actual * 1e8,
        predicted={
            "mean": pred_mean * 1e8,
            "lower": pred_lower * 1e8,
            "upper": pred_upper * 1e8,
        },
        # Fit metrics
        fit_statistics=fit_statistics,
        # Summary metrics
        total_revenue=total_revenue,
        marketing_attributed_revenue={
            "mean": marketing_total,
            "lower": marketing_total * 0.82,
            "upper": marketing_total * 1.18,
        },
        blended_roi={
            "mean": 1.62,
            "lower": 1.28,
            "upper": 1.98,
        },
        marketing_contribution_pct={
            "mean": 0.228,
            "lower": 0.185,
            "upper": 0.272,
        },
        # Channel data
        channel_roi=channel_roi,
        channel_names=channels,
        channel_spend=channel_spend,
        # Decomposition
        component_totals=component_totals,
        component_time_series=component_ts,
        # Saturation and adstock
        saturation_curves=saturation_curves,
        adstock_curves=adstock_curves,
        current_spend=channel_spend,
        # MCMC diagnostics
        diagnostics=diagnostics,
        # Geographic data
        geo_names=geo_names,
        geo_performance=geo_performance,
        geo_roi=geo_roi,
        geo_contribution=geo_contribution,
        # Mediator pathways (nested model)
        mediator_names=mediator_names,
        mediator_pathways=mediator_pathways,
        mediator_time_series=mediator_time_series,
        total_indirect_effect=total_indirect_effect,
        # Cannibalization data
        product_names=product_names,
        cannibalization_matrix=cannibalization_matrix,
        net_product_effects=net_product_effects,
        # Model specification
        model_specification={
            "likelihood": "Student-t with estimated degrees of freedom",
            "baseline": "Hierarchical intercept + linear trend + Fourier seasonality (order 4)",
            "media_effects": "Hill saturation × Geometric adstock with geo-level random effects",
            "mediator_model": "Two-stage: Media → Awareness/Consideration → Purchase",
            "cross_product": "Multivariate outcome with off-diagonal cannibalization parameters",
            "controls": "Holidays, weather, promotional indicators, competitor SOV",
            "hierarchy": "National-level effects with region-level partial pooling",
            "priors": "Weakly informative, regularized horseshoe for variable selection",
            "chains": 4,
            "draws": 2000,
            "tune": 1500,
            "target_accept": 0.95,
        },
    )


def generate_extended_report():
    """Generate the comprehensive extended MMM report."""

    print("=" * 70)
    print("Extended MMM Report: Multi-Geo Nested Model with Cannibalization")
    print("=" * 70)

    # Create synthetic data
    print("\n1. Creating synthetic data bundle...")
    data = create_extended_mmm_data()

    print(f"   - {len(data.channel_names)} media channels")
    print(f"   - {len(data.geo_names)} geographic regions")
    print(f"   - {len(data.mediator_names)} mediator variables")
    print(f"   - {len(data.product_names)} product lines")
    print(f"   - {len(data.dates)} weeks of data")

    # Create report configuration
    print("\n2. Configuring report sections...")

    config = ReportConfig(
        title="Marketing Mix Model Analysis",
        subtitle="Multi-Geo Nested Model with Cross-Product Effects",
        client="National CPG Brand",
        analysis_period="January 2024 – December 2025",
        default_credible_interval=0.80,
        # Color scheme
        color_scheme=ColorScheme.from_palette(ColorPalette.SAGE),
        # Section configurations
        executive_summary=SectionConfig(
            enabled=True,
            credible_interval=0.80,
        ),
        model_fit=SectionConfig(
            enabled=True,
            title="Model Fit & Validation",
        ),
        channel_roi=SectionConfig(
            enabled=True,
            title="Channel Performance",
            subtitle="National-level ROI estimates with uncertainty quantification",
        ),
        geographic=SectionConfig(
            enabled=True,
            title="Regional Performance",
            subtitle="Geographic variation in media response",
        ),
        decomposition=SectionConfig(
            enabled=True,
            title="Revenue Decomposition",
        ),
        mediators=SectionConfig(
            enabled=True,
            title="Mediator Pathway Analysis",
            subtitle="How marketing drives awareness and consideration before purchase",
            custom_notes="""
                <p>This nested model captures the <strong>indirect effects</strong> of marketing 
                through measured mediators (awareness, consideration). Understanding these pathways 
                helps optimize upper-funnel vs. lower-funnel investment.</p>
            """,
        ),
        cannibalization=SectionConfig(
            enabled=True,
            title="Cross-Product Effects",
            subtitle="Marketing spillover and cannibalization between product lines",
        ),
        saturation=SectionConfig(
            enabled=True,
            title="Saturation & Carryover",
        ),
        methodology=SectionConfig(
            enabled=True,
            title="Methodology",
            subtitle="Bayesian hierarchical model with honest uncertainty quantification",
            custom_notes="""
                <p>This model was fit using a <strong>pre-specified</strong> analysis protocol 
                with all modeling choices documented before observing results. No specification 
                shopping was performed. All uncertainty intervals represent genuine epistemic 
                uncertainty in parameter estimates.</p>
            """,
        ),
        diagnostics=SectionConfig(
            enabled=True,
            title="Technical Diagnostics",
        ),
    )

    # Generate report
    print("\n3. Generating HTML report...")
    report = MMMReportGenerator(data=data, config=config)

    output_path = Path("./extended_mmm_report.html")
    report.to_html(output_path)

    # Get file size
    size_kb = output_path.stat().st_size / 1024
    print(f"\n4. Report saved to: {output_path}")
    print(f"   File size: {size_kb:.1f} KB")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Report Summary")
    print("=" * 70)
    print(f"\nKey Findings:")
    print(f"  • Total Revenue: ${data.total_revenue/1e9:.2f}B")
    print(
        f"  • Marketing Contribution: {data.marketing_contribution_pct['mean']*100:.1f}% "
        f"({data.marketing_contribution_pct['lower']*100:.1f}% - "
        f"{data.marketing_contribution_pct['upper']*100:.1f}%)"
    )
    print(
        f"  • Blended ROI: {data.blended_roi['mean']:.2f}x "
        f"({data.blended_roi['lower']:.2f} - {data.blended_roi['upper']:.2f})"
    )

    print(f"\nChannel ROI (80% CI):")
    for ch in data.channel_names:
        roi = data.channel_roi[ch]
        print(f"  • {ch}: {roi['mean']:.2f}x ({roi['lower']:.2f} - {roi['upper']:.2f})")

    print(f"\nRegional Insights:")
    for geo in data.geo_names:
        perf = data.geo_performance[geo]
        rev = (
            perf["revenue"]["mean"]
            if isinstance(perf["revenue"], dict)
            else perf["revenue"]
        )
        roi = (
            perf["blended_roi"]["mean"]
            if isinstance(perf.get("blended_roi", {}), dict)
            else perf.get("blended_roi", 0)
        )
        print(
            f"  • {geo}: ${rev/1e9:.2f}B revenue, "
            f"{perf['yoy_growth']*100:.0f}% YoY, "
            f"{roi:.2f}x ROI"
        )

    print(f"\nMediator Pathways (% of effect through each pathway):")
    for ch in data.channel_names:
        pathways = data.mediator_pathways[ch]
        print(
            f"  • {ch}: Direct {pathways['_direct']*100:.0f}%, "
            f"Awareness {pathways['Awareness']*100:.0f}%, "
            f"Consideration {pathways['Consideration']*100:.0f}%"
        )

    print(f"\nCross-Product Effects:")
    for prod in data.product_names:
        effects = data.net_product_effects[prod]
        print(
            f"  • {prod}: Direct ${effects['direct']/1e6:.0f}M, "
            f"Cannibalization ${effects['cannibalization']/1e6:+.0f}M, "
            f"Net ${effects['net']/1e6:.0f}M"
        )

    print("\n" + "=" * 70)
    print("Report generation complete!")
    print("=" * 70)

    return output_path


if __name__ == "__main__":
    generate_extended_report()
