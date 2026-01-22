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


class Section:
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
        return self.section_config.title or self.default_title
    
    @property
    def is_enabled(self) -> bool:
        return self.section_config.enabled
    
    def render(self) -> str:
        raise NotImplementedError
    
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
    """
    Model fit visualization with actual vs predicted.
    
    UPDATED: Now includes geo-level dropdown selector when geo data is available.
    The default view shows aggregated (total) model fit, with option to drill
    down to individual geographies.
    """
    
    section_id: str = "model-fit"
    default_title: str = "Model Fit"
    
    def render(self) -> str:
        if not self.is_enabled:
            return ""

        # Check for required data
        if self.data.dates is None or self.data.actual is None or self.data.predicted is None:
            return ""

        chart_config = ChartConfig(
            height=self.section_config.chart_height or 400,
            ci_level=self.section_config.credible_interval or 0.8,
        )

        # Determine if we have geo-level and/or product-level data
        has_geo = self.data.has_geo_data
        has_product = self.data.has_product_data

        # Use new dimension filter chart if we have geo or product data
        if has_geo or has_product:
            fit_chart = charts.create_model_fit_chart_with_dimension_filter(
                dates=self.data.dates,
                actual_agg=self.data.actual,
                predicted_agg=self.data.predicted,
                actual_by_geo=self.data.actual_by_geo if has_geo else None,
                predicted_by_geo=self.data.predicted_by_geo if has_geo else None,
                actual_by_product=self.data.actual_by_product if has_product else None,
                predicted_by_product=self.data.predicted_by_product if has_product else None,
                geo_names=self.data.geo_names if has_geo else None,
                product_names=self.data.product_names if has_product else None,
                config=self.config,
                chart_config=chart_config,
            )
        else:
            # Fallback to simple geo selector for backwards compatibility
            fit_chart = charts.create_model_fit_chart_with_geo_selector(
                dates=self.data.dates,
                actual_agg=self.data.actual,
                predicted_agg=self.data.predicted,
                actual_by_geo=None,
                predicted_by_geo=None,
                geo_names=None,
                config=self.config,
                chart_config=chart_config,
            )

        # Fit statistics with geo selector
        stats_html = self._render_fit_statistics_with_geo()

        # Methodology note about aggregation
        aggregation_note = ""
        if has_geo or has_product:
            dims = []
            if has_geo:
                dims.append("geographies")
            if has_product:
                dims.append("products")
            dim_str = " and ".join(dims)
            aggregation_note = f'''
            <div class="methodology-note">
                <p><strong>About this view:</strong> The default view shows
                model fit aggregated across all {dim_str}. Use the filters to examine fit for
                individual {dim_str}. Good aggregate fit does not guarantee good fit across all
                segments—check individual cross-sections if segment performance matters.</p>
            </div>
            '''

        content = f'''
            <p>
                The model fit shows observed data against posterior predictions. The shaded band
                represents the {int(chart_config.ci_level * 100)}% credible interval, capturing
                both parameter uncertainty and residual variance.
            </p>
            {fit_chart}
            {stats_html}
            {aggregation_note}
        '''

        return self._render_section_wrapper(content)
    
    def _render_fit_statistics_with_geo(self) -> str:
        """Render model fit statistics with geo selector."""
        if self.data.fit_statistics is None:
            return ""
        
        has_geo = (
            self.data.geo_names is not None 
            and len(self.data.geo_names) > 1
            and self.data.fit_statistics_by_geo is not None
        )
        
        return charts.create_fit_statistics_with_geo_selector(
            fit_stats_agg=self.data.fit_statistics,
            fit_stats_by_geo=self.data.fit_statistics_by_geo if has_geo else None,
            geo_names=self.data.geo_names if has_geo else None,
            config=self.config,
            div_id="modelFitStats",
        )


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
    """
    Revenue decomposition showing contribution of each component.
    
    UPDATED: Now includes geo-level dropdown selector when geo data is available.
    Features both stacked area (time series) and waterfall (total contribution) views,
    each with independent geo selectors.
    """
    
    section_id: str = "decomposition"
    default_title: str = "Revenue Decomposition"
    
    def render(self) -> str:
        if not self.is_enabled:
            return ""
        
        # Check for required data
        if self.data.component_time_series is None or self.data.dates is None:
            return ""
        
        chart_config = ChartConfig(
            height=self.section_config.chart_height or 450,
        )
        
        # Determine if we have geo-level decomposition
        has_geo = self.data.has_geo_decomposition
        
        content_parts = []
        
        # Introduction
        content_parts.append('''
            <p>
                Revenue decomposition breaks down the predicted outcome into component contributions: 
                baseline, trend, seasonality, media channels, and control variables. Each component's 
                contribution sums to the total predicted revenue.
            </p>
        ''')
        
        # Aggregation note for geo models
        if has_geo:
            content_parts.append('''
            <div class="callout insight">
                <p><strong>Multi-geography model:</strong> The default view shows contributions 
                aggregated across all geographies. Use the dropdown selectors to examine 
                contribution patterns for individual regions.</p>
            </div>
            ''')
        
        # =====================================================================
        # STACKED AREA CHART (Time Series View)
        # =====================================================================
        
        content_parts.append('<h3>Component Contributions Over Time</h3>')
        
        stacked_chart = charts.create_stacked_area_chart_with_geo_selector(
            dates=self.data.dates,
            components_agg=self.data.component_time_series,
            components_by_geo=self.data.component_time_series_by_geo if has_geo else None,
            geo_names=self.data.geo_names if has_geo else None,
            config=self.config,
            chart_config=chart_config,
        )
        content_parts.append(stacked_chart)
        
        # =====================================================================
        # WATERFALL CHART (Total Contribution View)
        # =====================================================================
        
        if self.data.component_totals is not None:
            content_parts.append('<h3>Total Contribution Breakdown</h3>')
            
            waterfall_chart = charts.create_waterfall_chart_with_geo_selector(
                component_totals_agg=self.data.component_totals,
                component_totals_by_geo=self.data.component_totals_by_geo if has_geo else None,
                geo_names=self.data.geo_names if has_geo else None,
                config=self.config,
                chart_config=ChartConfig(height=400),
            )
            content_parts.append(waterfall_chart)
        
        # =====================================================================
        # CONTRIBUTION SUMMARY TABLE
        # =====================================================================
        
        content_parts.append(self._render_contribution_summary_with_geo())
        
        return self._render_section_wrapper('\n'.join(content_parts))
    
    def _render_contribution_summary_with_geo(self) -> str:
        """Render contribution summary table with geo selector."""
        if self.data.component_totals is None:
            return ""
        
        has_geo = self.data.has_geo_decomposition
        
        total = sum(abs(v) for v in self.data.component_totals.values())
        if total == 0:
            total = 1.0
        
        # Build aggregated table
        rows = []
        for comp, value in self.data.component_totals.items():
            pct = value / total
            formatted_value = self.config.format_currency(value) if hasattr(self.config, 'format_currency') else f"{value:,.0f}"
            rows.append(f'''
                <tr data-geo="agg">
                    <td>{comp}</td>
                    <td class="mono">{formatted_value}</td>
                    <td class="mono">{pct:.1%}</td>
                </tr>
            ''')
        
        # Build geo-level rows (hidden by default)
        geo_rows = ""
        if has_geo and self.data.component_totals_by_geo:
            for geo in self.data.geo_names:
                geo_totals = self.data.component_totals_by_geo.get(geo, {})
                geo_total = sum(abs(v) for v in geo_totals.values())
                if geo_total == 0:
                    geo_total = 1.0
                
                for comp, value in geo_totals.items():
                    pct = value / geo_total
                    formatted_value = self.config.format_currency(value) if hasattr(self.config, 'format_currency') else f"{value:,.0f}"
                    geo_rows += f'''
                        <tr data-geo="{geo}" style="display: none;">
                            <td>{comp}</td>
                            <td class="mono">{formatted_value}</td>
                            <td class="mono">{pct:.1%}</td>
                        </tr>
                    '''
        
        # Dropdown selector
        dropdown_html = ""
        if has_geo:
            options = '<option value="agg" selected>Aggregated (Total)</option>'
            for geo in self.data.geo_names:
                options += f'<option value="{geo}">{geo}</option>'
            
            dropdown_html = f'''
            <div style="margin-bottom: 1rem;">
                <label style="font-size: 0.85rem; color: var(--color-text-muted);">View: </label>
                <select id="contribSummarySelect" style="padding: 0.25rem 0.5rem; border: 1px solid var(--color-border); border-radius: 4px; font-size: 0.85rem;">
                    {options}
                </select>
            </div>
            <script>
                document.getElementById('contribSummarySelect').addEventListener('change', function() {{
                    var selected = this.value;
                    var rows = document.querySelectorAll('#contribSummaryTable tr[data-geo]');
                    rows.forEach(function(row) {{
                        row.style.display = row.getAttribute('data-geo') === selected ? '' : 'none';
                    }});
                }});
            </script>
            '''
        
        return f'''
            <h3>Contribution Summary</h3>
            {dropdown_html}
            <table class="data-table" id="contribSummaryTable">
                <thead>
                    <tr>
                        <th>Component</th>
                        <th>Total Contribution</th>
                        <th>% of Total</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                    {geo_rows}
                </tbody>
            </table>
        '''


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
            parameter_names = list(self.data.posterior_samples.keys())
            len_params = len(parameter_names)
            prior_post_chart = charts.create_prior_posterior_chart(
                parameter_names=parameter_names,
                prior_samples=self.data.prior_samples,
                posterior_samples=self.data.posterior_samples,
                config=self.config,
                chart_config=ChartConfig(height=int(250*min(len_params/6, 1))),
            )
            content_parts.append(f'''
                <h3>Prior vs Posterior</h3>
                <p>Comparison shows how data updated prior beliefs.</p>
                {prior_post_chart}
            ''')
        
        if not content_parts:
            return ""
        
        return self._render_section_wrapper('\n'.join(content_parts))


class GeographicSection(Section):
    """Geographic performance breakdown for multi-geo models."""
    
    section_id: str = "geographic"
    default_title: str = "Geographic Analysis"
    
    def render(self) -> str:
        if not self.is_enabled:
            return ""
        
        # Check if we have geographic data
        if not self.data.geo_names or not self.data.geo_performance:
            return ""
        
        content_parts = []
        ci_level = int(self.section_config.credible_interval * 100)
        
        # Geographic performance summary table
        content_parts.append(f'''
            <h3>Performance by Geography</h3>
            <p>Regional breakdown with {ci_level}% credible intervals.</p>
        ''')
        
        # Build performance table
        table_rows = []
        for geo in self.data.geo_names:
            perf = self.data.geo_performance.get(geo, {})
            revenue = perf.get("revenue", {})
            roi = perf.get("blended_roi", {})
            contribution = perf.get("marketing_contribution_pct", {})
            
            rev_str = self._format_currency(revenue.get("mean", 0))
            roi_str = f"{roi.get('mean', 0):.2f}x"
            roi_ci = f"[{roi.get('lower', 0):.2f}, {roi.get('upper', 0):.2f}]"
            contrib_str = self._format_percentage(contribution.get("mean", 0))
            
            table_rows.append(f'''
                <tr>
                    <td><strong>{geo}</strong></td>
                    <td class="mono">{rev_str}</td>
                    <td class="mono">{roi_str}</td>
                    <td class="mono muted">{roi_ci}</td>
                    <td class="mono">{contrib_str}</td>
                </tr>
            ''')
        
        content_parts.append(f'''
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Geography</th>
                        <th>Revenue</th>
                        <th>Blended ROI</th>
                        <th>{ci_level}% CI</th>
                        <th>Marketing %</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(table_rows)}
                </tbody>
            </table>
        ''')
        
        # Geographic ROI heatmap/chart
        if self.data.geo_roi:
            geo_chart = charts.create_geo_roi_heatmap(
                geo_names=self.data.geo_names,
                channel_names=self.data.channel_names or [],
                geo_roi=self.data.geo_roi,
                config=self.config,
                chart_config=ChartConfig(height=max(300, len(self.data.geo_names) * 40)),
            )
            content_parts.append(f'''
                <h3>Channel ROI by Geography</h3>
                <p>Heatmap showing channel performance across regions. Darker colors indicate higher ROI.</p>
                {geo_chart}
            ''')
        
        # Geographic contribution breakdown
        if self.data.geo_contribution:
            geo_decomp = charts.create_geo_decomposition_chart(
                geo_names=self.data.geo_names,
                geo_contribution=self.data.geo_contribution,
                config=self.config,
                chart_config=ChartConfig(height=400),
            )
            content_parts.append(f'''
                <h3>Contribution Decomposition by Geography</h3>
                {geo_decomp}
            ''')
        
        return self._render_section_wrapper('\n'.join(content_parts))


class MediatorSection(Section):
    """Mediator pathway analysis for nested models."""
    
    section_id: str = "mediators"
    default_title: str = "Mediator Pathway Analysis"
    
    def render(self) -> str:
        if not self.is_enabled:
            return ""
        
        # Check for mediator data
        if not self.data.mediator_names or not self.data.mediator_pathways:
            return ""
        
        content_parts = []
        ci_level = int(self.section_config.credible_interval * 100)
        
        # Introduction
        content_parts.append(f'''
            <h3>Indirect Effects Through Mediators</h3>
            <p>Marketing affects sales both directly and indirectly through intermediate outcomes 
            (e.g., awareness, consideration). This analysis decomposes total effects into direct 
            and mediated pathways with {ci_level}% credible intervals.</p>
        ''')
        
        # Pathway diagram (visual representation)
        pathway_chart = charts.create_mediator_pathway_chart(
            channel_names=self.data.channel_names or [],
            mediator_names=self.data.mediator_names,
            mediator_pathways=self.data.mediator_pathways,
            config=self.config,
            chart_config=ChartConfig(height=500),
        )
        content_parts.append(pathway_chart)
        
        # Effects table by channel
        content_parts.append('''
            <h3>Effect Decomposition by Channel</h3>
        ''')
        
        table_rows = []
        for channel in (self.data.channel_names or []):
            pathways = self.data.mediator_pathways.get(channel, {})
            
            # Aggregate effects (handle both dict and scalar)
            total_effect = pathways.get("_total", 0)
            direct_effect = pathways.get("_direct", 0)
            indirect_effect = pathways.get("_indirect", 0)
            
            # Helper to extract value
            def get_val(v, key="mean", default=0):
                return v.get(key, default) if isinstance(v, dict) else v
            
            total_mean = get_val(total_effect)
            total_lower = get_val(total_effect, "lower", total_mean * 0.8)
            total_upper = get_val(total_effect, "upper", total_mean * 1.2)
            direct_mean = get_val(direct_effect)
            indirect_mean = get_val(indirect_effect)
            
            total_str = f"{total_mean:.3f}"
            total_ci = f"[{total_lower:.3f}, {total_upper:.3f}]"
            direct_str = f"{direct_mean:.3f}"
            indirect_str = f"{indirect_mean:.3f}"
            
            # Percentage mediated (using mediator sums if _indirect not provided)
            if indirect_mean == 0 and self.data.mediator_names:
                # Sum up mediator effects
                indirect_mean = sum(
                    get_val(pathways.get(m, 0)) for m in self.data.mediator_names
                )
                indirect_str = f"{indirect_mean:.3f}"
            
            pct_mediated = (indirect_mean / total_mean * 100) if total_mean != 0 else 0
            
            table_rows.append(f'''
                <tr>
                    <td><strong>{channel}</strong></td>
                    <td class="mono">{total_str}</td>
                    <td class="mono muted">{total_ci}</td>
                    <td class="mono">{direct_str}</td>
                    <td class="mono">{indirect_str}</td>
                    <td class="mono">{pct_mediated:.1f}%</td>
                </tr>
            ''')
        
        content_parts.append(f'''
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Channel</th>
                        <th>Total Effect</th>
                        <th>{ci_level}% CI</th>
                        <th>Direct</th>
                        <th>Indirect</th>
                        <th>% Mediated</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(table_rows)}
                </tbody>
            </table>
        ''')
        
        # Mediator time series
        if self.data.mediator_time_series and self.data.dates is not None:
            mediator_ts_chart = charts.create_mediator_time_series(
                dates=self.data.dates,
                mediator_names=self.data.mediator_names,
                mediator_time_series=self.data.mediator_time_series,
                config=self.config,
                chart_config=ChartConfig(height=350),
            )
            content_parts.append(f'''
                <h3>Mediator Values Over Time</h3>
                <p>Tracking awareness, consideration, and other intermediate outcomes.</p>
                {mediator_ts_chart}
            ''')
        
        # Interpretation callout
        if self.data.total_indirect_effect:
            indirect = self.data.total_indirect_effect
            content_parts.append(f'''
                <div class="callout">
                    <strong>Key Insight:</strong> Approximately 
                    <strong>{self._format_percentage(indirect.get('mean', 0))}</strong> 
                    [{self._format_percentage(indirect.get('lower', 0))}, 
                    {self._format_percentage(indirect.get('upper', 0))}] of total marketing 
                    effect operates through measured mediators. Direct response accounts for 
                    the remainder.
                </div>
            ''')
        
        return self._render_section_wrapper('\n'.join(content_parts))


class CannibalizationSection(Section):
    """Cross-product cannibalization effects analysis."""
    
    section_id: str = "cannibalization"
    default_title: str = "Product Cannibalization Analysis"
    
    def render(self) -> str:
        if not self.is_enabled:
            return ""
        
        # Check for cannibalization data
        if not self.data.product_names or not self.data.cannibalization_matrix:
            return ""
        
        content_parts = []
        ci_level = int(self.section_config.credible_interval * 100)
        
        # Introduction
        content_parts.append(f'''
            <h3>Cross-Product Effects</h3>
            <p>Marketing for one product may cannibalize sales of another (substitution) or 
            boost them (halo effects). This matrix shows cross-product effects with {ci_level}% 
            credible intervals. Negative values indicate cannibalization; positive values 
            indicate synergy.</p>
        ''')
        
        # Cannibalization heatmap
        cannib_chart = charts.create_cannibalization_heatmap(
            product_names=self.data.product_names,
            cannibalization_matrix=self.data.cannibalization_matrix,
            config=self.config,
            chart_config=ChartConfig(height=max(350, len(self.data.product_names) * 50)),
        )
        content_parts.append(cannib_chart)
        
        # Net effects table
        if self.data.net_product_effects:
            content_parts.append('''
                <h3>Net Product Effects</h3>
                <p>Direct marketing effect minus cannibalization from other products' marketing.</p>
            ''')
            
            table_rows = []
            for product in self.data.product_names:
                effects = self.data.net_product_effects.get(product, {})
                direct = effects.get("direct", 0)
                cannib = effects.get("cannibalization", 0)
                net = effects.get("net", 0)
                
                # Color coding
                cannib_class = "danger" if cannib < -0.05 else ("success" if cannib > 0.05 else "")
                
                table_rows.append(f'''
                    <tr>
                        <td><strong>{product}</strong></td>
                        <td class="mono">{self._format_currency(direct)}</td>
                        <td class="mono {cannib_class}">{self._format_currency(cannib)}</td>
                        <td class="mono"><strong>{self._format_currency(net)}</strong></td>
                        <td class="mono">{(cannib/direct*100) if direct else 0:.1f}%</td>
                    </tr>
                ''')
            
            content_parts.append(f'''
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Product</th>
                            <th>Direct Effect</th>
                            <th>Cross-Product Effect</th>
                            <th>Net Effect</th>
                            <th>Cannib. %</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(table_rows)}
                    </tbody>
                </table>
            ''')
        
        # Detailed cross-effects table
        content_parts.append(f'''
            <h3>Detailed Cross-Product Matrix</h3>
            <p>Effect of row product's marketing on column product's sales.</p>
        ''')
        
        # Build matrix table
        header_cells = '<th>Source \\ Target</th>' + ''.join(
            f'<th>{p}</th>' for p in self.data.product_names
        )
        
        matrix_rows = []
        for source in self.data.product_names:
            row_cells = [f'<td><strong>{source}</strong></td>']
            for target in self.data.product_names:
                if source == target:
                    row_cells.append('<td class="mono muted">—</td>')
                else:
                    effect = self.data.cannibalization_matrix.get(source, {}).get(target, {})
                    mean = effect.get("mean", 0)
                    lower = effect.get("lower", 0)
                    upper = effect.get("upper", 0)
                    
                    # Color based on sign and significance
                    if upper < 0:
                        cell_class = "danger"  # Significantly negative (cannibalization)
                    elif lower > 0:
                        cell_class = "success"  # Significantly positive (synergy)
                    else:
                        cell_class = "muted"  # CI includes zero
                    
                    row_cells.append(
                        f'<td class="mono {cell_class}" title="[{lower:.3f}, {upper:.3f}]">'
                        f'{mean:.3f}</td>'
                    )
            matrix_rows.append(f'<tr>{"".join(row_cells)}</tr>')
        
        content_parts.append(f'''
            <table class="data-table matrix-table">
                <thead><tr>{header_cells}</tr></thead>
                <tbody>{''.join(matrix_rows)}</tbody>
            </table>
            <p class="muted" style="font-size: 0.85rem; margin-top: 0.5rem;">
                <span class="danger">Red</span> = significant cannibalization; 
                <span class="success">Green</span> = significant synergy; 
                <span class="muted">Gray</span> = not significant (CI includes zero).
                Hover for credible intervals.
            </p>
        ''')
        
        # Key insights
        significant_cannib = []
        significant_synergy = []
        
        for source in self.data.product_names:
            for target in self.data.product_names:
                if source != target:
                    effect = self.data.cannibalization_matrix.get(source, {}).get(target, {})
                    upper = effect.get("upper", 0)
                    lower = effect.get("lower", 0)
                    mean = effect.get("mean", 0)
                    
                    if upper < -0.01:
                        significant_cannib.append((source, target, mean))
                    elif lower > 0.01:
                        significant_synergy.append((source, target, mean))
        
        if significant_cannib or significant_synergy:
            insights = []
            if significant_cannib:
                top_cannib = sorted(significant_cannib, key=lambda x: x[2])[:3]
                cannib_items = [f"{s} → {t} ({m:.1%})" for s, t, m in top_cannib]
                insights.append(f"<strong>Strongest cannibalization:</strong> {', '.join(cannib_items)}")
            
            if significant_synergy:
                top_synergy = sorted(significant_synergy, key=lambda x: -x[2])[:3]
                synergy_items = [f"{s} → {t} (+{m:.1%})" for s, t, m in top_synergy]
                insights.append(f"<strong>Strongest synergies:</strong> {', '.join(synergy_items)}")
            
            content_parts.append(f'''
                <div class="callout">
                    {'<br>'.join(insights)}
                </div>
            ''')
        
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
    "geographic": GeographicSection,
    "mediators": MediatorSection,
    "cannibalization": CannibalizationSection,
}