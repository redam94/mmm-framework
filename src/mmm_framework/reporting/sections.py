"""
Report section renderers for MMM reports.

Each section is a modular component that can be enabled/disabled
and customized independently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
import numpy as np
import pandas as pd

from .config import ReportConfig, SectionConfig, ChartConfig
from . import charts

if TYPE_CHECKING:
    from .data_extractors import MMMDataBundle


class Section(ABC):
    """Base class for report sections."""
    
    section_id: str = "section"
    default_title: str = "Section"
    
    def __init__(
        self,
        data: MMMDataBundle,
        config: ReportConfig,
        section_config: SectionConfig | None = None,
    ):
        self.data = data
        self.config = config
        self.section_config = section_config or SectionConfig()
    
    @property
    def title(self) -> str:
        """Get section title."""
        return self.section_config.title or self.default_title
    
    @property
    def is_enabled(self) -> bool:
        """Check if section should be rendered."""
        return self.section_config.enabled
    
    @abstractmethod
    def render(self) -> str:
        """Render section HTML."""
        pass
    
    def _render_section_wrapper(self, content: str) -> str:
        """Wrap content in section div with title."""
        subtitle = ""
        if self.section_config.subtitle:
            subtitle = f'<p class="section-subtitle">{self.section_config.subtitle}</p>'
        
        notes = ""
        if self.section_config.custom_notes:
            notes = f'''
            <div class="methodology-note">
                <p>{self.section_config.custom_notes}</p>
            </div>
            '''
        
        return f'''
        <section class="section" id="{self.section_id}">
            <h2>{self.title}</h2>
            {subtitle}
            {content}
            {notes}
        </section>
        '''


class ExecutiveSummarySection(Section):
    """Executive summary with key metrics and uncertainty callouts."""
    
    section_id: str = "executive-summary"
    default_title: str = "Executive Summary"
    
    def render(self) -> str:
        if not self.is_enabled:
            return ""
        
        colors = self.config.color_scheme
        ci_level = int(self.section_config.credible_interval * 100)
        
        # Build metrics grid
        metrics_html = self._render_metrics_grid()
        
        # Key finding callout
        key_finding = self._render_key_finding()
        
        # Uncertainty callout
        uncertainty_callout = ""
        if self.config.uncertainty_callout:
            uncertainty_callout = f'''
            <div class="callout warning">
                <h4>⚠️ Uncertainty Matters</h4>
                <p>
                    All estimates include {ci_level}% credible intervals reflecting genuine uncertainty from limited data.
                    Point estimates alone can be misleading—decisions should account for the full range of plausible values.
                </p>
            </div>
            '''
        
        content = f'''
            {metrics_html}
            {key_finding}
            {uncertainty_callout}
        '''
        
        return self._render_section_wrapper(content)
    
    def _render_metrics_grid(self) -> str:
        """Render key metrics grid."""
        ci_level = int(self.section_config.credible_interval * 100)
        
        metrics = []
        
        # Total revenue
        if self.data.total_revenue is not None:
            metrics.append({
                "value": self.config.format_currency(self.data.total_revenue),
                "label": "Total Revenue",
                "highlight": True,
            })
        
        # Marketing-attributed revenue
        if self.data.marketing_attributed_revenue is not None:
            metrics.append({
                "value": self.config.format_currency(self.data.marketing_attributed_revenue["mean"]),
                "label": "Marketing-Attributed Revenue",
                "ci": f"{ci_level}% CI: [{self.config.format_currency(self.data.marketing_attributed_revenue['lower'])} – {self.config.format_currency(self.data.marketing_attributed_revenue['upper'])}]",
                "highlight": True,
            })
        
        # Blended ROI
        if self.data.blended_roi is not None:
            metrics.append({
                "value": f"{self.data.blended_roi['mean']:.2f}",
                "label": "Blended Marketing ROI",
                "ci": f"{ci_level}% CI: [{self.data.blended_roi['lower']:.2f} – {self.data.blended_roi['upper']:.2f}]",
            })
        
        # Marketing contribution %
        if self.data.marketing_contribution_pct is not None:
            metrics.append({
                "value": f"{self.data.marketing_contribution_pct['mean']:.1%}",
                "label": "Marketing Contribution",
                "ci": f"{ci_level}% CI: [{self.data.marketing_contribution_pct['lower']:.1%} – {self.data.marketing_contribution_pct['upper']:.1%}]",
            })
        
        # Render metrics cards
        cards = []
        for metric in metrics:
            highlight_class = "highlight" if metric.get("highlight") else ""
            ci_html = f'<div class="ci">{metric["ci"]}</div>' if metric.get("ci") else ""
            
            cards.append(f'''
                <div class="metric-card {highlight_class}">
                    <div class="value">{metric["value"]}</div>
                    <div class="label">{metric["label"]}</div>
                    {ci_html}
                </div>
            ''')
        
        return f'''
            <div class="metrics-grid">
                {''.join(cards)}
            </div>
        '''
    
    def _render_key_finding(self) -> str:
        """Render key finding callout based on data."""
        # Generate insight based on available data
        if self.data.channel_roi and len(self.data.channel_roi) > 0:
            # Find highest ROI channels with narrow uncertainty
            high_conf_channels = []
            uncertain_channels = []
            
            for channel, roi_data in self.data.channel_roi.items():
                spread = roi_data["upper"] - roi_data["lower"]
                relative_spread = spread / max(roi_data["mean"], 0.01)
                
                if roi_data["mean"] > 1.0 and relative_spread < 0.5:
                    high_conf_channels.append(channel)
                elif roi_data["mean"] > 1.0 and relative_spread >= 0.5:
                    uncertain_channels.append(channel)
            
            finding_text = ""
            if high_conf_channels:
                channels_str = ", ".join(high_conf_channels[:2])
                finding_text = f'''
                    <strong>{channels_str}</strong> show the highest ROI with relatively narrow uncertainty bands,
                    suggesting strong evidence of effectiveness.
                '''
            if uncertain_channels:
                channels_str = ", ".join(uncertain_channels[:2])
                finding_text += f''' {channels_str} show positive returns but with wider uncertainty—
                    additional experimentation could sharpen these estimates.'''
            
            if finding_text:
                return f'''
                    <div class="callout insight">
                        <h4>📊 Key Finding</h4>
                        <p>{finding_text}</p>
                    </div>
                '''
        
        return ""


class ModelFitSection(Section):
    """Model fit visualization with actual vs predicted."""
    
    section_id: str = "model-fit"
    default_title: str = "Model Fit"
    
    def render(self) -> str:
        if not self.is_enabled:
            return ""
        
        # Check if we have the required data
        if self.data.dates is None or self.data.actual is None or self.data.predicted is None:
            return ""
        
        chart_config = ChartConfig(
            height=self.section_config.chart_height,
            ci_level=self.section_config.credible_interval,
        )
        
        # Model fit chart
        fit_chart = charts.create_model_fit_chart(
            dates=self.data.dates,
            actual=self.data.actual,
            predicted_mean=self.data.predicted["mean"],
            predicted_lower=self.data.predicted["lower"],
            predicted_upper=self.data.predicted["upper"],
            config=self.config,
            chart_config=chart_config,
        )
        
        # Fit statistics
        stats_html = self._render_fit_statistics()
        
        content = f'''
            {fit_chart}
            {stats_html}
        '''
        
        return self._render_section_wrapper(content)
    
    def _render_fit_statistics(self) -> str:
        """Render model fit statistics."""
        if self.data.fit_statistics is None:
            return ""
        
        stats = self.data.fit_statistics
        
        rows = []
        stat_labels = {
            "r2": ("R²", ".3f"),
            "rmse": ("RMSE", ",.0f"),
            "mae": ("MAE", ",.0f"),
            "mape": ("MAPE", ".1%"),
        }
        
        for key, (label, fmt) in stat_labels.items():
            if key in stats:
                rows.append(f'''
                    <tr>
                        <td>{label}</td>
                        <td class="mono">{stats[key]:{fmt}}</td>
                    </tr>
                ''')
        
        if not rows:
            return ""
        
        return f'''
            <h3>Fit Statistics</h3>
            <table class="data-table" style="max-width: 400px;">
                <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        '''


class ChannelROISection(Section):
    """Channel ROI analysis with forest plot."""
    
    section_id: str = "channel-roi"
    default_title: str = "Channel Performance"
    
    def render(self) -> str:
        if not self.is_enabled:
            return ""
        
        if not self.data.channel_roi:
            return ""
        
        channels = list(self.data.channel_roi.keys())
        roi_mean = np.array([self.data.channel_roi[ch]["mean"] for ch in channels])
        roi_lower = np.array([self.data.channel_roi[ch]["lower"] for ch in channels])
        roi_upper = np.array([self.data.channel_roi[ch]["upper"] for ch in channels])
        
        chart_config = ChartConfig(
            height=max(250, 50 * len(channels)),
            ci_level=self.section_config.credible_interval,
        )
        
        # Forest plot
        forest_plot = charts.create_roi_forest_plot(
            channels=channels,
            roi_mean=roi_mean,
            roi_lower=roi_lower,
            roi_upper=roi_upper,
            config=self.config,
            chart_config=chart_config,
        )
        
        # ROI table
        roi_table = self._render_roi_table(channels, roi_mean, roi_lower, roi_upper)
        
        # Channel legend
        legend = self._render_channel_legend(channels)
        
        content = f'''
            {legend}
            {forest_plot}
            <h3>Detailed ROI Estimates</h3>
            {roi_table}
        '''
        
        return self._render_section_wrapper(content)
    
    def _render_roi_table(
        self,
        channels: list[str],
        roi_mean: np.ndarray,
        roi_lower: np.ndarray,
        roi_upper: np.ndarray,
    ) -> str:
        """Render detailed ROI table."""
        ci_level = int(self.section_config.credible_interval * 100)
        
        # Sort by mean ROI
        sort_idx = np.argsort(roi_mean)[::-1]
        
        rows = []
        for i in sort_idx:
            ch = channels[i]
            mean, lower, upper = roi_mean[i], roi_lower[i], roi_upper[i]
            
            # Determine confidence class
            if lower > 1.0:
                conf_class = "positive"
                status = "Strong evidence"
            elif upper < 1.0:
                conf_class = "negative"
                status = "Underperforming"
            else:
                conf_class = "uncertain"
                status = "Uncertain"
            
            rows.append(f'''
                <tr>
                    <td>{ch}</td>
                    <td class="mono">{mean:.2f}</td>
                    <td class="mono">[{lower:.2f}, {upper:.2f}]</td>
                    <td class="{conf_class}">{status}</td>
                </tr>
            ''')
        
        return f'''
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Channel</th>
                        <th>ROI (Mean)</th>
                        <th>{ci_level}% CI</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        '''
    
    def _render_channel_legend(self, channels: list[str]) -> str:
        """Render channel color legend."""
        pills = []
        for ch in channels:
            color = self.config.channel_colors.get(ch)
            pills.append(f'''
                <div class="channel-pill">
                    <span class="dot" style="background: {color}"></span>{ch}
                </div>
            ''')
        
        return f'''
            <div class="channel-legend">{''.join(pills)}</div>
        '''


class DecompositionSection(Section):
    """Revenue decomposition visualizations."""
    
    section_id: str = "decomposition"
    default_title: str = "Revenue Decomposition"
    
    def render(self) -> str:
        if not self.is_enabled:
            return ""
        
        content_parts = []
        
        # Waterfall chart
        if self.data.component_totals:
            waterfall = charts.create_waterfall_chart(
                categories=list(self.data.component_totals.keys()),
                values=np.array(list(self.data.component_totals.values())),
                config=self.config,
                chart_config=ChartConfig(height=400),
            )
            content_parts.append(f'''
                <h3>Total Contribution by Component</h3>
                {waterfall}
            ''')
        
        # Time series decomposition
        if self.data.component_time_series and self.data.dates is not None:
            stacked = charts.create_stacked_area_chart(
                dates=self.data.dates,
                components=self.data.component_time_series,
                config=self.config,
                chart_config=ChartConfig(height=400),
            )
            content_parts.append(f'''
                <h3>Contribution Over Time</h3>
                {stacked}
            ''')
        
        if not content_parts:
            return ""
        
        return self._render_section_wrapper('\n'.join(content_parts))


class SaturationSection(Section):
    """Saturation and adstock analysis."""
    
    section_id: str = "saturation"
    default_title: str = "Saturation & Carryover Effects"
    
    def render(self) -> str:
        if not self.is_enabled:
            return ""
        
        content_parts = []
        
        # Saturation curves
        if self.data.saturation_curves:
            channels = list(self.data.saturation_curves.keys())
            
            sat_chart = charts.create_saturation_curves(
                channels=channels,
                spend_ranges={ch: self.data.saturation_curves[ch]["spend"] for ch in channels},
                response_curves={ch: self.data.saturation_curves[ch]["response"] for ch in channels},
                current_spend=self.data.current_spend or {},
                config=self.config,
                chart_config=ChartConfig(height=280),
            )
            content_parts.append(f'''
                <h3>Saturation Curves</h3>
                <p>The saturation curves show diminishing returns as spend increases. 
                The orange diamond marks current spend levels.</p>
                {sat_chart}
            ''')
        
        # Adstock curves
        if self.data.adstock_curves:
            adstock_chart = charts.create_adstock_chart(
                channels=list(self.data.adstock_curves.keys()),
                lag_weights=self.data.adstock_curves,
                config=self.config,
                chart_config=ChartConfig(height=350),
            )
            content_parts.append(f'''
                <h3>Adstock Decay</h3>
                <p>Carryover effects show how advertising impact persists over time.</p>
                {adstock_chart}
            ''')
        
        if not content_parts:
            return ""
        
        return self._render_section_wrapper('\n'.join(content_parts))


class SensitivitySection(Section):
    """Sensitivity analysis results."""
    
    section_id: str = "sensitivity"
    default_title: str = "Sensitivity Analysis"
    
    def render(self) -> str:
        if not self.is_enabled:
            return ""
        
        if not self.data.sensitivity_results:
            return ""
        
        content_parts = []
        
        # Overview text
        content_parts.append(f'''
            <p>
                Sensitivity analysis explores how results change under alternative model specifications.
                Robust findings should be stable across reasonable specification choices.
            </p>
        ''')
        
        # Sensitivity chart
        if "scenarios" in self.data.sensitivity_results:
            scenarios = self.data.sensitivity_results["scenarios"]
            base_values = self.data.sensitivity_results.get("base_values", np.array([]))
            alternatives = self.data.sensitivity_results.get("alternatives", {})
            
            sens_chart = charts.create_sensitivity_chart(
                scenarios=scenarios,
                base_values=base_values,
                alternative_values=alternatives,
                config=self.config,
                chart_config=ChartConfig(height=400),
            )
            content_parts.append(sens_chart)
        
        # Results table
        if "table" in self.data.sensitivity_results:
            table_data = self.data.sensitivity_results["table"]
            rows = []
            for row in table_data:
                cells = ''.join(f'<td class="mono">{cell}</td>' for cell in row[1:])
                rows.append(f'<tr><td>{row[0]}</td>{cells}</tr>')
            
            content_parts.append(f'''
                <h3>Specification Comparison</h3>
                <table class="data-table">
                    <thead><tr>{''.join(f'<th>{h}</th>' for h in table_data[0])}</tr></thead>
                    <tbody>{''.join(rows[1:])}</tbody>
                </table>
            ''')
        
        return self._render_section_wrapper('\n'.join(content_parts))


class MethodologySection(Section):
    """Model methodology documentation."""
    
    section_id: str = "methodology"
    default_title: str = "Methodology"
    
    def render(self) -> str:
        if not self.is_enabled:
            return ""
        
        content_parts = []
        
        # Model specification
        if self.data.model_specification:
            spec = self.data.model_specification
            content_parts.append(f'''
                <div class="methodology-note">
                    <h4>Model Specification</h4>
                    <p>This analysis uses a Bayesian Marketing Mix Model with the following components:</p>
                    <ul style="margin: 1rem 0 1rem 1.5rem;">
                        <li><strong>Likelihood:</strong> {spec.get("likelihood", "Normal with estimated scale")}</li>
                        <li><strong>Baseline:</strong> {spec.get("baseline", "Linear trend + Fourier seasonality")}</li>
                        <li><strong>Media effects:</strong> {spec.get("media_effects", "Hill saturation × Geometric adstock")}</li>
                        <li><strong>Controls:</strong> {spec.get("controls", "As specified")}</li>
                        <li><strong>Priors:</strong> {spec.get("priors", "Weakly informative, documented in technical appendix")}</li>
                    </ul>
                    <p>
                        Inference via MCMC ({spec.get("chains", 4)} chains, 
                        {spec.get("draws", 2000)} samples each, 
                        {spec.get("tune", 1000)} warmup).
                    </p>
                </div>
            ''')
        
        # Honest uncertainty principles
        if self.config.methodology_note:
            ci_level = int(self.section_config.credible_interval * 100)
            content_parts.append(f'''
                <div class="methodology-note">
                    <h4>Honest Uncertainty Principles</h4>
                    <p>This report follows principles of honest uncertainty quantification:</p>
                    <ul style="margin: 1rem 0 0 1.5rem;">
                        <li>All estimates include {ci_level}% credible intervals, not just point estimates</li>
                        <li>Model was pre-specified before examining results</li>
                        <li>Sensitivity analysis explores reasonable alternative specifications</li>
                        <li>Recommendations explicitly acknowledge uncertainty levels</li>
                        <li>Experimental validation proposed for high-stakes, uncertain estimates</li>
                    </ul>
                </div>
            ''')
        
        return self._render_section_wrapper('\n'.join(content_parts))


class DiagnosticsSection(Section):
    """MCMC diagnostics and convergence checks."""
    
    section_id: str = "diagnostics"
    default_title: str = "Model Diagnostics"
    
    def render(self) -> str:
        if not self.is_enabled:
            return ""
        
        content_parts = []
        
        # Diagnostics summary table
        if self.data.diagnostics:
            diag = self.data.diagnostics
            
            # Determine status
            divergences = diag.get("divergences", 0)
            rhat_max = diag.get("rhat_max", 1.0)
            ess_min = diag.get("ess_bulk_min", 400)
            
            divergence_status = "✅ Pass" if divergences == 0 else f"⚠️ {divergences} divergences"
            rhat_status = "✅ Pass" if rhat_max < 1.01 else f"⚠️ {rhat_max:.3f}"
            ess_status = "✅ Pass" if ess_min > 400 else f"⚠️ {ess_min:.0f}"
            
            content_parts.append(f'''
                <h3>Convergence Diagnostics</h3>
                <table class="data-table" style="max-width: 500px;">
                    <thead><tr><th>Diagnostic</th><th>Value</th><th>Status</th></tr></thead>
                    <tbody>
                        <tr><td>Divergences</td><td class="mono">{divergences}</td><td>{divergence_status}</td></tr>
                        <tr><td>R-hat (max)</td><td class="mono">{rhat_max:.4f}</td><td>{rhat_status}</td></tr>
                        <tr><td>ESS bulk (min)</td><td class="mono">{ess_min:.0f}</td><td>{ess_status}</td></tr>
                    </tbody>
                </table>
            ''')
        
        # Trace plots
        if self.data.trace_data and self.data.trace_parameters:
            trace_chart = charts.create_trace_plot(
                parameter_names=self.data.trace_parameters[:6],  # Limit to 6
                traces_data=self.data.trace_data,
                config=self.config,
                chart_config=ChartConfig(height=180),
            )
            content_parts.append(f'''
                <h3>Trace Plots</h3>
                <p>Visual inspection of MCMC chains for key parameters.</p>
                {trace_chart}
            ''')
        
        # Prior/posterior comparison
        if self.data.prior_samples and self.data.posterior_samples:
            prior_post_chart = charts.create_prior_posterior_chart(
                parameter_names=list(self.data.posterior_samples.keys())[:6],
                prior_samples=self.data.prior_samples,
                posterior_samples=self.data.posterior_samples,
                config=self.config,
                chart_config=ChartConfig(height=250),
            )
            content_parts.append(f'''
                <h3>Prior vs Posterior</h3>
                <p>Comparison shows how data updated prior beliefs.</p>
                {prior_post_chart}
            ''')
        
        if not content_parts:
            return ""
        
        return self._render_section_wrapper('\n'.join(content_parts))


# Registry of available sections
SECTION_REGISTRY: dict[str, type[Section]] = {
    "executive_summary": ExecutiveSummarySection,
    "model_fit": ModelFitSection,
    "channel_roi": ChannelROISection,
    "decomposition": DecompositionSection,
    "saturation": SaturationSection,
    "sensitivity": SensitivitySection,
    "methodology": MethodologySection,
    "diagnostics": DiagnosticsSection,
}