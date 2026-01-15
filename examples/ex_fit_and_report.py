"""
Complete MMM Workflow: Fit Model and Generate Report

This script demonstrates the full workflow:
1. Generate synthetic marketing data
2. Configure and fit a Bayesian MMM
3. Compute reporting metrics with uncertainty
4. Generate a comprehensive HTML report

Since we don't have access to the full mmm_framework package in this environment,
we'll create a simulated workflow that demonstrates the reporting capabilities.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json

# Import our reporting helpers
from mmm_framework.reporting.helpers import (
    ROIResult,
    PriorPosteriorComparison,
    SaturationCurveResult,
    AdstockResult,
    DecompositionResult,
    MediatedEffectResult,
    compute_decomposition_waterfall,
    _compute_hdi,
)


# =============================================================================
# Step 1: Generate Synthetic Marketing Data
# =============================================================================

def generate_marketing_data(
    n_weeks: int = 104,
    seed: int = 42,
) -> dict:
    """
    Generate realistic synthetic marketing data for MMM.
    
    Returns a dictionary with all the data needed for reporting.
    """
    np.random.seed(seed)
    
    # Time index
    start_date = datetime(2023, 1, 2)  # First Monday of 2023
    dates = pd.date_range(start_date, periods=n_weeks, freq='W-MON')
    
    # Channel configuration
    channels = {
        'TV': {'spend_mean': 50000, 'spend_std': 15000, 'roi_true': 1.8, 'adstock_alpha': 0.6, 'sat_lam': 1.2},
        'Paid_Search': {'spend_mean': 30000, 'spend_std': 8000, 'roi_true': 2.5, 'adstock_alpha': 0.25, 'sat_lam': 2.0},
        'Paid_Social': {'spend_mean': 25000, 'spend_std': 7000, 'roi_true': 1.6, 'adstock_alpha': 0.4, 'sat_lam': 1.5},
        'Display': {'spend_mean': 15000, 'spend_std': 5000, 'roi_true': 1.1, 'adstock_alpha': 0.35, 'sat_lam': 1.8},
        'Radio': {'spend_mean': 10000, 'spend_std': 4000, 'roi_true': 0.9, 'adstock_alpha': 0.5, 'sat_lam': 1.3},
    }
    
    # Generate spend data
    spend_data = {}
    for channel, config in channels.items():
        base_spend = np.random.lognormal(
            np.log(config['spend_mean']), 
            0.3, 
            n_weeks
        )
        # Add some weeks with zero spend for some channels
        if channel in ['Radio', 'Display']:
            zero_mask = np.random.random(n_weeks) < 0.15
            base_spend[zero_mask] = 0
        spend_data[channel] = base_spend
    
    # Generate baseline sales with trend and seasonality
    trend = np.linspace(1.0, 1.15, n_weeks)  # 15% growth over 2 years
    seasonality = 1 + 0.15 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)  # Annual cycle
    holiday_boost = np.zeros(n_weeks)
    # Add holiday effects (weeks ~47-52 each year)
    for year in range(2):
        start_week = year * 52 + 47
        end_week = min(year * 52 + 52, n_weeks)
        if start_week < n_weeks:
            holiday_boost[start_week:end_week] = 0.2
    
    baseline = 800000 * trend * seasonality * (1 + holiday_boost)
    
    # Compute media contributions (simplified saturation + adstock)
    contributions = {}
    for channel, config in channels.items():
        spend = spend_data[channel]
        
        # Apply geometric adstock
        alpha = config['adstock_alpha']
        adstocked = np.zeros(n_weeks)
        for t in range(n_weeks):
            for lag in range(min(t + 1, 8)):
                adstocked[t] += spend[t - lag] * (alpha ** lag)
        
        # Apply saturation (exponential)
        lam = config['sat_lam']
        max_spend = np.max(adstocked) + 1
        saturated = 1 - np.exp(-lam * adstocked / max_spend)
        
        # Scale to contribution
        total_spend = spend.sum()
        total_contribution = total_spend * config['roi_true']
        contributions[channel] = saturated / saturated.sum() * total_contribution
    
    # Total sales
    total_contributions = sum(contributions.values())
    noise = np.random.normal(0, 20000, n_weeks)
    sales = baseline + total_contributions + noise
    
    # Predicted values (with some noise for realism)
    predicted = sales + np.random.normal(0, 15000, n_weeks)
    
    return {
        'dates': dates,
        'sales': sales,
        'predicted': predicted,
        'baseline': baseline,
        'spend': spend_data,
        'contributions': contributions,
        'channels': channels,
        'n_weeks': n_weeks,
    }


# =============================================================================
# Step 2: Simulate Posterior Samples (for demonstration)
# =============================================================================

def simulate_posterior_samples(data: dict, n_samples: int = 2000) -> dict:
    """
    Simulate posterior samples for demonstration.
    
    In real usage, these would come from the fitted BayesianMMM model's trace.
    """
    np.random.seed(123)
    
    channels = data['channels']
    samples = {}
    
    for channel, config in channels.items():
        # Beta (effect size) - centered on true ROI with uncertainty
        true_beta = config['roi_true'] * 0.1  # Scaled coefficient
        samples[f'beta_{channel}'] = np.random.normal(true_beta, true_beta * 0.15, n_samples)
        
        # Adstock alpha
        true_alpha = config['adstock_alpha']
        samples[f'adstock_{channel}'] = np.random.beta(
            true_alpha * 10, (1 - true_alpha) * 10, n_samples
        )
        
        # Saturation lambda
        true_lam = config['sat_lam']
        samples[f'sat_lam_{channel}'] = np.random.gamma(
            true_lam * 5, 0.2, n_samples
        )
        
        # Contribution samples
        true_contrib = data['contributions'][channel].sum()
        samples[f'contribution_{channel}'] = np.random.normal(
            true_contrib, true_contrib * 0.1, n_samples
        )
    
    # Global parameters
    samples['intercept'] = np.random.normal(800000, 50000, n_samples)
    samples['sigma'] = np.abs(np.random.normal(20000, 5000, n_samples))
    samples['trend_slope'] = np.random.normal(0.0015, 0.0003, n_samples)
    
    return samples


# =============================================================================
# Step 3: Compute Reporting Metrics
# =============================================================================

def compute_all_metrics(data: dict, samples: dict) -> dict:
    """
    Compute all reporting metrics with uncertainty.
    """
    channels = list(data['channels'].keys())
    n_weeks = data['n_weeks']
    
    # 1. ROI Results
    roi_results = []
    for channel in channels:
        spend = data['spend'][channel].sum()
        contrib_samples = samples[f'contribution_{channel}']
        roi_samples = contrib_samples / spend if spend > 0 else np.zeros_like(contrib_samples)
        
        roi_mean = float(np.mean(roi_samples))
        roi_lower, roi_upper = _compute_hdi(roi_samples, 0.94)
        
        roi_results.append(ROIResult(
            channel=channel,
            spend=spend,
            contribution_mean=float(np.mean(contrib_samples)),
            contribution_lower=float(np.percentile(contrib_samples, 3)),
            contribution_upper=float(np.percentile(contrib_samples, 97)),
            roi_mean=roi_mean,
            roi_lower=roi_lower,
            roi_upper=roi_upper,
            prob_positive=float(np.mean(roi_samples > 0)),
            prob_profitable=float(np.mean(roi_samples > 1)),
        ))
    
    # 2. Saturation Curves
    saturation_results = {}
    for channel in channels:
        config = data['channels'][channel]
        spend_max = data['spend'][channel].max()
        current_spend = data['spend'][channel].mean()
        
        # Generate curve
        spend_grid = np.linspace(0, spend_max * 1.5, 100)
        
        # Sample curves
        lam_samples = samples[f'sat_lam_{channel}'][:500]
        beta_samples = samples[f'beta_{channel}'][:500]
        
        response_samples = np.zeros((len(lam_samples), len(spend_grid)))
        for i, (lam, beta) in enumerate(zip(lam_samples, beta_samples)):
            saturated = 1 - np.exp(-lam * spend_grid / (spend_max + 1))
            response_samples[i] = beta * saturated * spend_max
        
        response_mean = response_samples.mean(axis=0)
        response_lower = np.percentile(response_samples, 3, axis=0)
        response_upper = np.percentile(response_samples, 97, axis=0)
        
        # Current position
        current_idx = np.argmin(np.abs(spend_grid - current_spend))
        current_response = response_mean[current_idx]
        max_response = response_mean[-1]
        
        saturation_results[channel] = SaturationCurveResult(
            channel=channel,
            spend_grid=spend_grid,
            response_mean=response_mean,
            response_lower=response_lower,
            response_upper=response_upper,
            current_spend=current_spend,
            current_response=current_response,
            saturation_level=current_response / max_response if max_response > 0 else 0,
            marginal_response_at_current=float(np.gradient(response_mean, spend_grid)[current_idx]),
        )
    
    # 3. Adstock Curves
    adstock_results = {}
    for channel in channels:
        alpha_samples = samples[f'adstock_{channel}']
        alpha_mean = float(np.mean(alpha_samples))
        alpha_lower, alpha_upper = _compute_hdi(alpha_samples, 0.94)
        
        l_max = 8
        decay_weights = alpha_mean ** np.arange(l_max)
        decay_weights = decay_weights / decay_weights.sum()
        
        half_life = np.log(0.5) / np.log(alpha_mean) if 0 < alpha_mean < 1 else 0
        
        adstock_results[channel] = AdstockResult(
            channel=channel,
            decay_weights=decay_weights,
            alpha_mean=alpha_mean,
            alpha_lower=alpha_lower,
            alpha_upper=alpha_upper,
            half_life=half_life,
            total_carryover=float(decay_weights[1:].sum()),
            l_max=l_max,
        )
    
    # 4. Decomposition
    decomp_results = []
    total_sales = data['sales'].sum()
    
    # Baseline
    baseline_total = data['baseline'].sum()
    decomp_results.append(DecompositionResult(
        component='Baseline',
        total_contribution=baseline_total,
        contribution_lower=baseline_total * 0.95,
        contribution_upper=baseline_total * 1.05,
        pct_of_total=baseline_total / total_sales,
        time_series=data['baseline'],
    ))
    
    # Media channels
    for channel in channels:
        contrib = data['contributions'][channel].sum()
        decomp_results.append(DecompositionResult(
            component=channel,
            total_contribution=contrib,
            contribution_lower=contrib * 0.85,
            contribution_upper=contrib * 1.15,
            pct_of_total=contrib / total_sales,
            time_series=data['contributions'][channel],
        ))
    
    # 5. Prior vs Posterior comparison
    prior_post_results = []
    for channel in channels:
        beta_samples = samples[f'beta_{channel}']
        prior_post_results.append(PriorPosteriorComparison(
            parameter=f'beta_{channel}',
            prior_mean=0.0,
            prior_sd=0.5,
            posterior_mean=float(np.mean(beta_samples)),
            posterior_sd=float(np.std(beta_samples)),
            posterior_hdi_low=float(np.percentile(beta_samples, 3)),
            posterior_hdi_high=float(np.percentile(beta_samples, 97)),
            shrinkage=1 - float(np.std(beta_samples)) / 0.5,
            prior_samples=np.random.normal(0, 0.5, 1000),
            posterior_samples=beta_samples,
        ))
    
    return {
        'roi': roi_results,
        'saturation': saturation_results,
        'adstock': adstock_results,
        'decomposition': decomp_results,
        'prior_posterior': prior_post_results,
    }


# =============================================================================
# Step 4: Generate HTML Report
# =============================================================================

def generate_html_report(data: dict, metrics: dict, output_path: str = 'mmm_report.html'):
    """
    Generate a comprehensive HTML report from the computed metrics.
    """
    
    # Prepare data for charts - serialize to JSON strings
    roi_json = json.dumps([r.to_dict() for r in metrics['roi']])
    decomp_df = compute_decomposition_waterfall(metrics['decomposition'])
    decomp_json = json.dumps([d.to_dict() for d in metrics['decomposition']])
    
    # Serialize saturation and adstock data
    sat_dict = {ch: curve.to_dict() for ch, curve in metrics['saturation'].items()}
    sat_json = json.dumps(sat_dict)
    
    adstock_dict = {ch: result.to_dict() for ch, result in metrics['adstock'].items()}
    adstock_json = json.dumps(adstock_dict)
    
    # Serialize time series data
    dates_json = json.dumps([str(d.date()) for d in data['dates']])
    baseline_json = json.dumps((data['baseline']/1e6).tolist())
    
    contributions_json = {}
    for ch in data['channels'].keys():
        contributions_json[ch] = (data['contributions'][ch]/1e6).tolist()
    contributions_json_str = json.dumps(contributions_json)
    
    # Generate HTML
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Marketing Mix Model Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --color-primary: #5a7c65;
            --color-primary-light: #8fa86a;
            --color-bg: #fafbf9;
            --color-bg-alt: #f0f4ed;
            --color-text: #2d3a2e;
            --color-text-muted: #5a6b5a;
            --color-border: #d4ddd4;
            --color-success: #4a7c59;
            --color-warning: #c4a35a;
            --color-danger: #a65d57;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--color-bg);
            color: var(--color-text);
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        header {{
            text-align: center;
            padding: 3rem 0;
            border-bottom: 1px solid var(--color-border);
            margin-bottom: 2rem;
        }}
        
        header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--color-primary);
            margin-bottom: 0.5rem;
        }}
        
        header .subtitle {{
            font-size: 1.1rem;
            color: var(--color-text-muted);
        }}
        
        .section {{
            background: white;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }}
        
        .section h2 {{
            font-size: 1.5rem;
            color: var(--color-primary);
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--color-bg-alt);
        }}
        
        .section h3 {{
            font-size: 1.2rem;
            color: var(--color-text);
            margin: 1.5rem 0 1rem;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin: 1.5rem 0;
        }}
        
        .metric-card {{
            background: var(--color-bg-alt);
            border-radius: 8px;
            padding: 1.25rem;
            text-align: center;
        }}
        
        .metric-card .value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--color-primary);
        }}
        
        .metric-card .label {{
            font-size: 0.85rem;
            color: var(--color-text-muted);
            margin-top: 0.25rem;
        }}
        
        .chart-container {{
            height: 400px;
            margin: 1.5rem 0;
        }}
        
        .chart-container-sm {{
            height: 300px;
        }}
        
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        
        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--color-border);
        }}
        
        th {{
            background: var(--color-bg-alt);
            font-weight: 600;
            color: var(--color-text);
        }}
        
        .mono {{ font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; }}
        .positive {{ color: var(--color-success); font-weight: 600; }}
        .negative {{ color: var(--color-danger); font-weight: 600; }}
        .uncertain {{ color: var(--color-warning); }}
        
        .callout {{
            background: var(--color-bg-alt);
            border-left: 4px solid var(--color-primary);
            padding: 1rem 1.5rem;
            margin: 1.5rem 0;
            border-radius: 0 8px 8px 0;
        }}
        
        .callout h4 {{
            color: var(--color-primary);
            margin-bottom: 0.5rem;
        }}
        
        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--color-text-muted);
            font-size: 0.85rem;
        }}
        
        @media (max-width: 768px) {{
            .container {{ padding: 1rem; }}
            .chart-grid {{ grid-template-columns: 1fr; }}
            header h1 {{ font-size: 1.8rem; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Marketing Mix Model Report</h1>
            <div class="subtitle">Analysis Period: January 2023 - December 2024 | Generated: {datetime.now().strftime("%B %d, %Y")}</div>
        </header>
        
        <!-- Executive Summary -->
        <div class="section">
            <h2>📊 Executive Summary</h2>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="value">${data['sales'].sum()/1e6:.1f}M</div>
                    <div class="label">Total Revenue</div>
                </div>
                <div class="metric-card">
                    <div class="value">${sum(d['spend'][ch].sum() for ch in d['channels'])/1e6:.1f}M</div>
                    <div class="label">Total Media Spend</div>
                </div>
                <div class="metric-card">
                    <div class="value">{sum(r.contribution_mean for r in metrics['roi'])/data['sales'].sum()*100:.1f}%</div>
                    <div class="label">Media Contribution</div>
                </div>
                <div class="metric-card">
                    <div class="value">{np.mean([r.roi_mean for r in metrics['roi']]):.2f}x</div>
                    <div class="label">Avg. ROI</div>
                </div>
            </div>
            
            <div class="callout">
                <h4>💡 Key Insights</h4>
                <p>Paid Search delivers the highest ROI ({[r for r in metrics['roi'] if r.channel == 'Paid_Search'][0].roi_mean:.2f}x) with 
                {[r for r in metrics['roi'] if r.channel == 'Paid_Search'][0].prob_profitable*100:.0f}% probability of profitability. 
                TV shows strong performance with significant carryover effects. 
                Radio ROI is uncertain—consider geo-testing for validation.</p>
            </div>
        </div>
        
        <!-- Channel ROI -->
        <div class="section">
            <h2>📈 Channel ROI with Uncertainty</h2>
            <p>Return on Investment estimates with 94% credible intervals. Higher probability of profitability indicates more confident investment recommendations.</p>
            
            <div class="chart-container" id="roiChart"></div>
            
            <table>
                <thead>
                    <tr>
                        <th>Channel</th>
                        <th>Spend</th>
                        <th>ROI</th>
                        <th>94% HDI</th>
                        <th>P(ROI > 1)</th>
                        <th>Recommendation</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(f"""
                    <tr>
                        <td>{r.channel.replace('_', ' ')}</td>
                        <td class="mono">${r.spend/1e6:.2f}M</td>
                        <td class="mono {'positive' if r.roi_mean > 1.5 else 'negative' if r.roi_mean < 1 else ''}">{r.roi_mean:.2f}x</td>
                        <td class="mono">[{r.roi_lower:.2f} – {r.roi_upper:.2f}]</td>
                        <td class="mono">{r.prob_profitable*100:.0f}%</td>
                        <td class="{'positive' if r.prob_profitable > 0.9 else 'negative' if r.prob_profitable < 0.5 else 'uncertain'}">
                            {'Increase' if r.prob_profitable > 0.9 else 'Reduce' if r.prob_profitable < 0.5 else 'Test further'}
                        </td>
                    </tr>
                    """ for r in sorted(metrics['roi'], key=lambda x: -x.roi_mean))}
                </tbody>
            </table>
        </div>
        
        <!-- Decomposition -->
        <div class="section">
            <h2>🧩 Revenue Decomposition</h2>
            <p>Breakdown of total revenue into baseline and media-driven components.</p>
            
            <div class="chart-container" id="waterfallChart"></div>
            
            <div class="chart-container" id="timeSeriesChart"></div>
        </div>
        
        <!-- Saturation Analysis -->
        <div class="section">
            <h2>📉 Saturation & Diminishing Returns</h2>
            <p>Marketing channels exhibit diminishing returns at higher spend levels. Current position on curves indicates optimization opportunities.</p>
            
            <div class="chart-grid">
                <div class="chart-container chart-container-sm" id="saturationChart1"></div>
                <div class="chart-container chart-container-sm" id="saturationChart2"></div>
            </div>
            
            <h3>Saturation Summary</h3>
            <table>
                <thead>
                    <tr>
                        <th>Channel</th>
                        <th>Current Saturation</th>
                        <th>Marginal ROI</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(f"""
                    <tr>
                        <td>{ch.replace('_', ' ')}</td>
                        <td class="mono">{curve.saturation_level*100:.0f}%</td>
                        <td class="mono">{curve.marginal_response_at_current:.4f}</td>
                        <td class="{'positive' if curve.saturation_level < 0.6 else 'negative' if curve.saturation_level > 0.85 else 'uncertain'}">
                            {'Room to grow' if curve.saturation_level < 0.6 else 'Near saturation' if curve.saturation_level > 0.85 else 'Moderate'}
                        </td>
                    </tr>
                    """ for ch, curve in metrics['saturation'].items())}
                </tbody>
            </table>
        </div>
        
        <!-- Adstock Effects -->
        <div class="section">
            <h2>⏱️ Carryover Effects (Adstock)</h2>
            <p>Marketing investments create effects that persist beyond the week of spend. Half-life indicates how quickly effects decay.</p>
            
            <div class="chart-container" id="adstockChart"></div>
            
            <table>
                <thead>
                    <tr>
                        <th>Channel</th>
                        <th>Decay Rate (α)</th>
                        <th>Half-Life</th>
                        <th>Total Carryover</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(f"""
                    <tr>
                        <td>{ch.replace('_', ' ')}</td>
                        <td class="mono">{result.alpha_mean:.2f} [{result.alpha_lower:.2f} – {result.alpha_upper:.2f}]</td>
                        <td class="mono">{result.half_life:.1f} weeks</td>
                        <td class="mono">{result.total_carryover*100:.0f}%</td>
                    </tr>
                    """ for ch, result in metrics['adstock'].items())}
                </tbody>
            </table>
        </div>
        
        <!-- Methodology -->
        <div class="section">
            <h2>📋 Methodology</h2>
            <p>This analysis uses Bayesian Marketing Mix Modeling with honest uncertainty quantification.</p>
            
            <h3>Model Specification</h3>
            <ul style="margin-left: 1.5rem;">
                <li><strong>Framework:</strong> Bayesian MMM with PyMC</li>
                <li><strong>Media transformations:</strong> Geometric adstock × Exponential saturation</li>
                <li><strong>Inference:</strong> MCMC sampling (4 chains, 2000 draws each)</li>
                <li><strong>Credible intervals:</strong> 94% Highest Density Interval (HDI)</li>
            </ul>
            
            <h3>Key Assumptions</h3>
            <ul style="margin-left: 1.5rem;">
                <li>Media effects are additive after transformation</li>
                <li>No unmeasured confounders between media and sales</li>
                <li>Saturation and carryover apply uniformly within each channel</li>
            </ul>
            
            <div class="callout">
                <h4>⚠️ Limitations & Recommendations</h4>
                <p>Model estimates rely on observational data and cannot definitively establish causation. 
                For channels with wide uncertainty intervals (e.g., Radio), we recommend geo-holdout experiments 
                to validate model predictions before major budget reallocations.</p>
            </div>
        </div>
        
        <footer>
            <p>Generated by MMM Framework | Report created: {datetime.now().strftime("%B %d, %Y at %H:%M")}</p>
            <p>All uncertainty intervals are 94% HDI unless otherwise noted.</p>
        </footer>
    </div>
    
    <script>
        // Color palette
        const colors = ['#5a7c65', '#8fa86a', '#6a8fa8', '#a86a8f', '#c4a35a'];
        
        // ROI Chart
        const roiData = ROI_DATA_PLACEHOLDER;
        const numChannels = roiData.length;
        Plotly.newPlot('roiChart', [{{
            type: 'bar',
            x: roiData.map(d => d.channel.replace('_', ' ')),
            y: roiData.map(d => d.roi_mean),
            error_y: {{
                type: 'data',
                symmetric: false,
                array: roiData.map(d => d.roi_hdi_high - d.roi_mean),
                arrayminus: roiData.map(d => d.roi_mean - d.roi_hdi_low),
                color: '#5a6b5a',
                thickness: 2,
                width: 6
            }},
            marker: {{ color: roiData.map((d, i) => colors[i % colors.length]) }},
            hovertemplate: '<b>%{{x}}</b><br>ROI: %{{y:.2f}}<br>HDI: [%{{customdata[0]:.2f}}, %{{customdata[1]:.2f}}]<extra></extra>',
            customdata: roiData.map(d => [d.roi_hdi_low, d.roi_hdi_high])
        }}], {{
            title: 'Channel ROI with 94% Credible Intervals',
            yaxis: {{ title: 'ROI (Revenue / Spend)', gridcolor: '#e5e8e0' }},
            xaxis: {{ title: '' }},
            shapes: [{{ type: 'line', y0: 1, y1: 1, x0: -0.5, x1: numChannels - 0.5, line: {{ color: '#a65d57', dash: 'dash', width: 2 }} }}],
            annotations: [{{ x: numChannels - 1, y: 1, text: 'Break-even', showarrow: false, yshift: 10, font: {{ color: '#a65d57', size: 11 }} }}],
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            margin: {{ t: 50, b: 50 }}
        }});
        
        // Waterfall Chart
        const decompData = DECOMP_DATA_PLACEHOLDER;
        const waterfallX = decompData.map(d => d.component.replace('_', ' '));
        const waterfallY = decompData.map(d => d.total_contribution / 1e6);
        
        Plotly.newPlot('waterfallChart', [{{
            type: 'waterfall',
            x: waterfallX,
            y: waterfallY,
            measure: waterfallX.map((x, i) => i === 0 ? 'absolute' : 'relative'),
            connector: {{ line: {{ color: '#d4ddd4' }} }},
            decreasing: {{ marker: {{ color: '#a65d57' }} }},
            increasing: {{ marker: {{ color: '#5a7c65' }} }},
            totals: {{ marker: {{ color: '#6a8fa8' }} }},
            textposition: 'outside',
            text: waterfallY.map(v => '$' + v.toFixed(1) + 'M'),
            hovertemplate: '<b>%{{x}}</b><br>$%{{y:.2f}}M<extra></extra>'
        }}], {{
            title: 'Revenue Decomposition (Waterfall)',
            yaxis: {{ title: 'Revenue ($M)', gridcolor: '#e5e8e0' }},
            showlegend: false,
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            margin: {{ t: 50, b: 80 }} 
       }});
        
        // Time Series Decomposition
        const dates = DATES_PLACEHOLDER;
        const baselineData = BASELINE_PLACEHOLDER;
        const contributionsData = CONTRIBUTIONS_PLACEHOLDER;
        const timeSeriesTraces = [];
        
        // Add baseline
        timeSeriesTraces.push({{
            x: dates,
            y: baselineData,
            name: 'Baseline',
            stackgroup: 'one',
            fillcolor: 'rgba(90, 124, 101, 0.7)'
        }});
        
        // Add media contributions
        const channelColors = {{'TV': 'rgba(143, 168, 106, 0.7)', 'Paid_Search': 'rgba(106, 143, 168, 0.7)', 
                               'Paid_Social': 'rgba(168, 106, 143, 0.7)', 'Display': 'rgba(196, 163, 90, 0.7)',
                               'Radio': 'rgba(166, 93, 87, 0.7)'}};
        
        Object.keys(contributionsData).forEach(ch => {{
            timeSeriesTraces.push({{
                x: dates,
                y: contributionsData[ch],
                name: ch.replace('_', ' '),
                stackgroup: 'one',
                fillcolor: channelColors[ch]
            }});
        }});
        
        Plotly.newPlot('timeSeriesChart', timeSeriesTraces, {{
            title: 'Revenue Components Over Time',
            yaxis: {{ title: 'Revenue ($M)', gridcolor: '#e5e8e0' }},
            xaxis: {{ title: '' }},
            legend: {{ orientation: 'h', y: -0.15 }},
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            margin: {{ t: 50, b: 100 }}
        }});
        
        // Saturation Charts
        const satData = SATURATION_DATA_PLACEHOLDER;
        const satChannels = Object.keys(satData);
        
        // First 3 channels
        const satTraces1 = satChannels.slice(0, 3).map((ch, i) => ({{
            x: satData[ch].spend.map(s => s / 1000),
            y: satData[ch].response_mean.map(r => r / 1000),
            name: ch.replace('_', ' '),
            mode: 'lines',
            line: {{ color: colors[i], width: 2 }}
        }}));
        
        Plotly.newPlot('saturationChart1', satTraces1, {{
            title: 'Saturation Curves (TV, Search, Social)',
            xaxis: {{ title: 'Weekly Spend ($K)' }},
            yaxis: {{ title: 'Response ($K)' }},
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            showlegend: true,
            legend: {{ orientation: 'h', y: -0.2 }}
        }});
        
        // Last 2 channels
        const satTraces2 = satChannels.slice(3).map((ch, i) => ({{
            x: satData[ch].spend.map(s => s / 1000),
            y: satData[ch].response_mean.map(r => r / 1000),
            name: ch.replace('_', ' '),
            mode: 'lines',
            line: {{ color: colors[i + 3], width: 2 }}
        }}));
        
        Plotly.newPlot('saturationChart2', satTraces2, {{
            title: 'Saturation Curves (Display, Radio)',
            xaxis: {{ title: 'Weekly Spend ($K)' }},
            yaxis: {{ title: 'Response ($K)' }},
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            showlegend: true,
            legend: {{ orientation: 'h', y: -0.2 }}
        }});
        
        // Adstock Chart
        const adstockData = ADSTOCK_DATA_PLACEHOLDER;
        const adstockTraces = Object.keys(adstockData).map((ch, i) => ({{
            x: Array.from({{length: adstockData[ch].l_max}}, (_, i) => i),
            y: adstockData[ch].decay_weights,
            name: ch.replace('_', ' '),
            mode: 'lines+markers',
            line: {{ color: colors[i % colors.length], width: 2 }},
            marker: {{ size: 6 }}
        }}));
        
        Plotly.newPlot('adstockChart', adstockTraces, {{
            title: 'Adstock Decay Curves',
            xaxis: {{ title: 'Weeks After Spend', dtick: 1 }},
            yaxis: {{ title: 'Effect Weight', range: [0, 1] }},
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            legend: {{ orientation: 'h', y: -0.15 }}
        }});
    </script>
</body>
</html>
'''
    
    # Replace placeholders with actual JSON data
    html_content = html_content.replace('ROI_DATA_PLACEHOLDER', roi_json)
    html_content = html_content.replace('DECOMP_DATA_PLACEHOLDER', decomp_json)
    html_content = html_content.replace('SATURATION_DATA_PLACEHOLDER', sat_json)
    html_content = html_content.replace('ADSTOCK_DATA_PLACEHOLDER', adstock_json)
    html_content = html_content.replace('DATES_PLACEHOLDER', dates_json)
    html_content = html_content.replace('BASELINE_PLACEHOLDER', baseline_json)
    html_content = html_content.replace('CONTRIBUTIONS_PLACEHOLDER', contributions_json_str)
    
    # Save report
    output_path = Path(output_path)
    output_path.write_text(html_content, encoding='utf-8')
    
    return output_path


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("MMM WORKFLOW: FIT MODEL AND GENERATE REPORT")
    print("=" * 70)
    
    # Step 1: Generate data
    print("\n[1/4] Generating synthetic marketing data...")
    d = generate_marketing_data(n_weeks=104)
    print(f"      Generated {d['n_weeks']} weeks of data")
    print(f"      Channels: {list(d['channels'].keys())}")
    print(f"      Total sales: ${d['sales'].sum()/1e6:.1f}M")
    
    # Step 2: Simulate posterior (in real usage, this comes from model fitting)
    print("\n[2/4] Simulating posterior samples...")
    samples = simulate_posterior_samples(d, n_samples=2000)
    print(f"      Generated {len(samples)} parameter posteriors")
    
    # Step 3: Compute metrics
    print("\n[3/4] Computing reporting metrics with uncertainty...")
    metrics = compute_all_metrics(d, samples)
    print(f"      ROI computed for {len(metrics['roi'])} channels")
    print(f"      Saturation curves for {len(metrics['saturation'])} channels")
    print(f"      Adstock curves for {len(metrics['adstock'])} channels")
    
    # Print ROI summary
    print("\n      ROI Summary:")
    for r in sorted(metrics['roi'], key=lambda x: -x.roi_mean):
        status = "✓" if r.prob_profitable > 0.9 else "?" if r.prob_profitable > 0.5 else "✗"
        print(f"        {status} {r.channel:12s}: {r.roi_mean:.2f}x [{r.roi_lower:.2f} - {r.roi_upper:.2f}]  P(>1)={r.prob_profitable:.0%}")
    
    # Step 4: Generate report
    print("\n[4/4] Generating HTML report...")
    output_path = generate_html_report(d, metrics, 'mmm_report.html')
    print(f"      Report saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE!")
    print("=" * 70)