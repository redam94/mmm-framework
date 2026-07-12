"""
Report section renderers for MMM reports.

Each section is a modular component that can be enabled/disabled
and customized independently.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING
import numpy as np

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

    def _format_currency(self, value: float) -> str:
        """Format a value as currency via the report config."""
        if hasattr(self.config, "format_currency"):
            return self.config.format_currency(value)
        return f"{value:,.0f}"

    def _format_percentage(self, value: float) -> str:
        """Format a proportion (0-1) as a percentage."""
        return f"{value:.1%}"

    def _render_section_wrapper(self, content: str) -> str:
        """Wrap content in section div with title."""
        subtitle = ""
        if self.section_config.subtitle:
            subtitle = f'<p class="section-subtitle">{self.section_config.subtitle}</p>'

        notes = ""
        if self.section_config.custom_notes:
            notes = f"""
            <div class="methodology-note">
                <p>{self.section_config.custom_notes}</p>
            </div>
            """

        return f"""
        <section class="section" id="{self.section_id}">
            <h2>{self.title}</h2>
            {subtitle}
            {content}
            {notes}
        </section>
        """


class ExecutiveSummarySection(Section):
    """Executive summary with key metrics and uncertainty callouts."""

    section_id: str = "executive-summary"
    default_title: str = "Executive Summary"

    def render(self) -> str:
        if not self.is_enabled:
            return ""

        ci_level = int(self.section_config.credible_interval * 100)

        # Build metrics grid
        metrics_html = self._render_metrics_grid()

        # Key finding callout
        key_finding = self._render_key_finding()

        # Uncertainty callout
        uncertainty_callout = ""
        if self.config.uncertainty_callout:
            uncertainty_callout = f"""
            <div class="callout warning">
                <h4>⚠️ Uncertainty Matters</h4>
                <p>
                    All estimates include {ci_level}% credible intervals reflecting genuine uncertainty from limited data.
                    Point estimates alone can be misleading—decisions should account for the full range of plausible values.
                </p>
            </div>
            """

        # Non-convergence banner is stamped at the very top: a non-converged
        # posterior makes every headline number unreliable, so it must not be
        # buried in the diagnostics section.
        convergence_banner = self._render_convergence_banner()
        # Approximate-fit banner: a MAP/ADVI/Pathfinder fit gives uncalibrated
        # uncertainty, so its credible intervals must carry a stop sign too.
        approximate_banner = self._render_approximate_banner()

        content = f"""
            {convergence_banner}
            {approximate_banner}
            {metrics_html}
            {key_finding}
            {uncertainty_callout}
        """

        return self._render_section_wrapper(content)

    def _render_convergence_banner(self) -> str:
        """Prominent warning when the MCMC fit did not converge.

        Renders only when the convergence verdict is explicitly ``False``
        (an approximate fit is ``None`` and handled by the uncertainty callout +
        diagnostics section). Closes the gap where a non-converged model still
        produced a clean ROI deck with no stop sign on the headline.
        """
        diag = getattr(self.data, "diagnostics", None)
        if not diag:
            return ""
        try:
            from ..diagnostics.convergence import (
                convergence_warning_message,
                is_converged,
            )
        except Exception:
            return ""
        if is_converged(diag) is not False:
            return ""
        import html as _html

        msg = convergence_warning_message(diag, label="This model") or (
            "This model did not pass standard MCMC convergence checks."
        )
        return f"""
            <div class="callout warning" style="border-left:6px solid #c0392b;background:#fdecea;color:#611a15;">
                <h4>🛑 Model has NOT converged — do not act on these numbers</h4>
                <p>{_html.escape(msg)}</p>
                <p style="margin-top:6px;font-size:0.9em;">
                    The estimates below come from a sampler that failed convergence checks; their
                    credible intervals are not trustworthy. Re-fit with more tuning/draws/chains
                    (or a reparameterization) before using these numbers for decisions.
                </p>
            </div>
        """

    def _render_approximate_banner(self) -> str:
        """Prominent notice when the model was fit with an APPROXIMATE method
        (MAP / ADVI / full-rank ADVI / Pathfinder) rather than full NUTS.

        Approximate fits are fast checks whose uncertainty is **not calibrated**
        — R-hat/ESS are undefined and the credible intervals are unreliable — so
        the headline numbers must carry a stop sign even though the fit did not
        "fail" convergence (there is no convergence to assess).
        """
        diag = getattr(self.data, "diagnostics", None) or {}
        if not diag.get("approximate"):
            return ""
        import html as _html

        method = _html.escape(str(diag.get("fit_method") or "approximate")).upper()
        return f"""
            <div class="callout warning" style="border-left:6px solid #b45309;background:#fef6e7;color:#663c00;">
                <h4>⚠️ Approximate fit ({method}) — uncertainty is not calibrated</h4>
                <p>
                    This model was fit with an approximate method for a fast check, not full
                    MCMC (NUTS). R-hat/ESS are not assessable and the credible intervals below
                    are <strong>not trustworthy</strong>. Re-fit with NUTS before using these
                    numbers for budget or experiment decisions.
                </p>
            </div>
        """

    def _render_metrics_grid(self) -> str:
        """Render key metrics grid."""
        ci_level = int(self.section_config.credible_interval * 100)

        metrics = []

        # Total revenue
        if self.data.total_revenue is not None:
            metrics.append(
                {
                    "value": self.config.format_currency(self.data.total_revenue),
                    "label": "Total Revenue",
                    "highlight": True,
                }
            )

        # Marketing-attributed revenue
        if self.data.marketing_attributed_revenue is not None:
            metrics.append(
                {
                    "value": self.config.format_currency(
                        self.data.marketing_attributed_revenue["mean"]
                    ),
                    "label": "Marketing-Attributed Revenue",
                    "ci": f"{ci_level}% CI: [{self.config.format_currency(self.data.marketing_attributed_revenue['lower'])} – {self.config.format_currency(self.data.marketing_attributed_revenue['upper'])}]",
                    "highlight": True,
                }
            )

        # Blended ROI
        if self.data.blended_roi is not None:
            metrics.append(
                {
                    "value": f"{self.data.blended_roi['mean']:.2f}",
                    "label": "Blended Marketing ROI",
                    "ci": f"{ci_level}% CI: [{self.data.blended_roi['lower']:.2f} – {self.data.blended_roi['upper']:.2f}]",
                }
            )

        # Marketing contribution %
        if self.data.marketing_contribution_pct is not None:
            metrics.append(
                {
                    "value": f"{self.data.marketing_contribution_pct['mean']:.1%}",
                    "label": "Marketing Contribution",
                    "ci": f"{ci_level}% CI: [{self.data.marketing_contribution_pct['lower']:.1%} – {self.data.marketing_contribution_pct['upper']:.1%}]",
                }
            )

        # Render metrics cards
        cards = []
        for metric in metrics:
            highlight_class = "highlight" if metric.get("highlight") else ""
            ci_html = (
                f'<div class="ci">{metric["ci"]}</div>' if metric.get("ci") else ""
            )

            cards.append(f"""
                <div class="metric-card {highlight_class}">
                    <div class="value">{metric["value"]}</div>
                    <div class="label">{metric["label"]}</div>
                    {ci_html}
                </div>
            """)

        return f"""
            <div class="metrics-grid">
                {''.join(cards)}
            </div>
        """

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
                finding_text = f"""
                    <strong>{channels_str}</strong> show the highest ROI with relatively narrow uncertainty bands,
                    suggesting strong evidence of effectiveness.
                """
            if uncertain_channels:
                channels_str = ", ".join(uncertain_channels[:2])
                finding_text += f""" {channels_str} show positive returns but with wider uncertainty—
                    additional experimentation could sharpen these estimates."""

            if finding_text:
                return f"""
                    <div class="callout insight">
                        <h4>📊 Key Finding</h4>
                        <p>{finding_text}</p>
                    </div>
                """

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
        if (
            self.data.dates is None
            or self.data.actual is None
            or self.data.predicted is None
        ):
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
                predicted_by_product=(
                    self.data.predicted_by_product if has_product else None
                ),
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
            aggregation_note = f"""
            <div class="methodology-note">
                <p><strong>About this view:</strong> The default view shows
                model fit aggregated across all {dim_str}. Use the filters to examine fit for
                individual {dim_str}. Good aggregate fit does not guarantee good fit across all
                segments—check individual cross-sections if segment performance matters.</p>
            </div>
            """

        content = f"""
            <p>
                The model fit shows observed data against posterior predictions. The shaded band
                represents the {int(chart_config.ci_level * 100)}% credible interval, capturing
                both parameter uncertainty and residual variance.
            </p>
            {fit_chart}
            {stats_html}
            {aggregation_note}
        """

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

        # Break-even reference for the forest plot: 1.0 when every channel is
        # monetary (the usual case), 0 when every channel is an efficiency
        # metric; a mixed portfolio keeps 1.0 (the per-channel table is
        # authoritative — the plot mixes units).
        roi_meta = self.data.channel_roi or {}
        monetary_flags = [
            roi_meta.get(ch, {}).get("is_monetary", True) for ch in channels
        ]
        reference_line = 0.0 if monetary_flags and not any(monetary_flags) else 1.0

        # Forest plot
        forest_plot = charts.create_roi_forest_plot(
            channels=channels,
            roi_mean=roi_mean,
            roi_lower=roi_lower,
            roi_upper=roi_upper,
            config=self.config,
            chart_config=chart_config,
            reference_line=reference_line,
        )

        # ROI table
        roi_table = self._render_roi_table(channels, roi_mean, roi_lower, roi_upper)

        # Channel legend
        legend = self._render_channel_legend(channels)

        content = f"""
            {legend}
            {forest_plot}
            <h3>Detailed ROI Estimates</h3>
            {roi_table}
        """

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

        # Per-channel measurement metadata (impression-level ROI). Spend channels
        # default to ROI vs a 1.0 break-even, so the table is unchanged;
        # impression channels show "Efficiency / 1K impr" judged against 0.
        roi_meta = self.data.channel_roi or {}
        all_monetary = all(
            roi_meta.get(ch, {}).get("is_monetary", True) for ch in channels
        )
        value_header = "ROI (Mean)" if all_monetary else "Value (Mean)"

        rows = []
        for i in sort_idx:
            ch = channels[i]
            mean, lower, upper = roi_mean[i], roi_lower[i], roi_upper[i]
            meta = roi_meta.get(ch, {})
            ref = float(meta.get("reference", 1.0))
            metric_label = meta.get("value_units", "ROI")

            # Determine confidence class against the metric's break-even
            # reference (1.0 for ROI, 0 for efficiency).
            if lower > ref:
                conf_class = "positive"
                status = "Strong evidence"
            elif upper < ref:
                conf_class = "negative"
                status = "Underperforming"
            else:
                conf_class = "uncertain"
                status = "Uncertain"

            rows.append(f"""
                <tr>
                    <td>{html.escape(ch)}</td>
                    <td>{html.escape(str(metric_label))}</td>
                    <td class="mono">{mean:.2f}</td>
                    <td class="mono">[{lower:.2f}, {upper:.2f}]</td>
                    <td class="{conf_class}">{status}</td>
                </tr>
            """)

        return f"""
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Channel</th>
                        <th>Metric</th>
                        <th>{value_header}</th>
                        <th>{ci_level}% CI</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        """

    def _render_channel_legend(self, channels: list[str]) -> str:
        """Render channel color legend."""
        pills = []
        for ch in channels:
            color = self.config.channel_colors.get(ch)
            pills.append(f"""
                <div class="channel-pill">
                    <span class="dot" style="background: {color}"></span>{html.escape(ch)}
                </div>
            """)

        return f"""
            <div class="channel-legend">{''.join(pills)}</div>
        """


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

        # Determine the selector dimension: geo when available, otherwise
        # product/outcome (multi-product and multi-outcome models reuse the
        # same selector machinery with different labels).
        has_geo = self.data.has_geo_decomposition
        has_product = (
            not has_geo
            and self.data.product_names is not None
            and len(self.data.product_names) > 1
            and self.data.component_time_series_by_product is not None
        )

        if has_geo:
            dim_series = self.data.component_time_series_by_geo
            dim_totals = self.data.component_totals_by_geo
            dim_names = self.data.geo_names
        elif has_product:
            dim_series = self.data.component_time_series_by_product
            dim_totals = self.data.component_totals_by_product
            dim_names = self.data.product_names
        else:
            dim_series = dim_totals = dim_names = None

        content_parts = []

        # Introduction
        content_parts.append("""
            <p>
                Revenue decomposition breaks down the predicted outcome into component contributions:
                baseline, trend, seasonality, media channels, and control variables. Each component's
                contribution sums to the total predicted revenue.
            </p>
        """)

        # Aggregation note for multi-dimensional models
        if has_geo:
            content_parts.append("""
            <div class="callout insight">
                <p><strong>Multi-geography model:</strong> The default view shows contributions
                aggregated across all geographies. Use the dropdown selectors to examine
                contribution patterns for individual regions.</p>
            </div>
            """)
        elif has_product:
            content_parts.append("""
            <div class="callout insight">
                <p><strong>Multi-outcome model:</strong> The default view shows the primary
                outcome. Use the dropdown selectors to examine contribution patterns for each
                outcome.</p>
            </div>
            """)

        # =====================================================================
        # TABBED VIEWS: stacked area (time series) + waterfall (totals)
        # =====================================================================

        stacked_chart = charts.create_stacked_area_chart_with_geo_selector(
            dates=self.data.dates,
            components_agg=self.data.component_time_series,
            components_by_geo=dim_series,
            geo_names=dim_names,
            config=self.config,
            chart_config=chart_config,
        )

        if self.data.component_totals is not None:
            waterfall_chart = charts.create_waterfall_chart_with_geo_selector(
                component_totals_agg=self.data.component_totals,
                component_totals_by_geo=dim_totals,
                geo_names=dim_names,
                config=self.config,
                chart_config=ChartConfig(height=400),
            )
            content_parts.append(f"""
                <div class="tab-container">
                    <div class="tab-buttons">
                        <button class="tab-btn active" onclick="showTab('decomp-tab-area')">Over Time</button>
                        <button class="tab-btn" onclick="showTab('decomp-tab-waterfall')">Total Breakdown</button>
                    </div>
                    <div id="decomp-tab-area" class="tab-content active">
                        <h3>Component Contributions Over Time</h3>
                        {stacked_chart}
                    </div>
                    <div id="decomp-tab-waterfall" class="tab-content">
                        <h3>Total Contribution Breakdown</h3>
                        {waterfall_chart}
                    </div>
                </div>
            """)
        else:
            content_parts.append("<h3>Component Contributions Over Time</h3>")
            content_parts.append(stacked_chart)

        # =====================================================================
        # CONTRIBUTION SUMMARY TABLE
        # =====================================================================

        content_parts.append(self._render_contribution_summary_with_geo())

        return self._render_section_wrapper("\n".join(content_parts))

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
            formatted_value = (
                self.config.format_currency(value)
                if hasattr(self.config, "format_currency")
                else f"{value:,.0f}"
            )
            rows.append(f"""
                <tr data-geo="agg">
                    <td>{html.escape(comp)}</td>
                    <td class="mono">{formatted_value}</td>
                    <td class="mono">{pct:.1%}</td>
                </tr>
            """)

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
                    formatted_value = (
                        self.config.format_currency(value)
                        if hasattr(self.config, "format_currency")
                        else f"{value:,.0f}"
                    )
                    geo_rows += f"""
                        <tr data-geo="{html.escape(geo)}" style="display: none;">
                            <td>{html.escape(comp)}</td>
                            <td class="mono">{formatted_value}</td>
                            <td class="mono">{pct:.1%}</td>
                        </tr>
                    """

        # Dropdown selector
        dropdown_html = ""
        if has_geo:
            options = '<option value="agg" selected>Aggregated (Total)</option>'
            for geo in self.data.geo_names:
                options += (
                    f'<option value="{html.escape(geo)}">{html.escape(geo)}</option>'
                )

            dropdown_html = f"""
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
            """

        return f"""
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
        """


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
                spend_ranges={
                    ch: self.data.saturation_curves[ch]["spend"] for ch in channels
                },
                response_curves={
                    ch: self.data.saturation_curves[ch]["response"] for ch in channels
                },
                current_spend=self.data.current_spend or {},
                config=self.config,
                chart_config=ChartConfig(height=280),
            )
            content_parts.append(f"""
                <h3>Saturation Curves</h3>
                <p>The saturation curves show diminishing returns as spend increases. 
                The orange diamond marks current spend levels.</p>
                {sat_chart}
            """)

        # Adstock curves
        if self.data.adstock_curves:
            adstock_chart = charts.create_adstock_chart(
                channels=list(self.data.adstock_curves.keys()),
                lag_weights=self.data.adstock_curves,
                config=self.config,
                chart_config=ChartConfig(height=350),
            )
            content_parts.append(f"""
                <h3>Adstock Decay</h3>
                <p>Carryover effects show how advertising impact persists over time.</p>
                {adstock_chart}
            """)

        if not content_parts:
            return ""

        return self._render_section_wrapper("\n".join(content_parts))


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
        content_parts.append("""
            <p>
                Sensitivity analysis explores how results change under alternative model specifications.
                Robust findings should be stable across reasonable specification choices.
            </p>
        """)

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
                cells = "".join(f'<td class="mono">{cell}</td>' for cell in row[1:])
                rows.append(f"<tr><td>{row[0]}</td>{cells}</tr>")

            content_parts.append(f"""
                <h3>Specification Comparison</h3>
                <table class="data-table">
                    <thead><tr>{''.join(f'<th>{h}</th>' for h in table_data[0])}</tr></thead>
                    <tbody>{''.join(rows[1:])}</tbody>
                </table>
            """)

        return self._render_section_wrapper("\n".join(content_parts))


class MethodologySection(Section):
    """Model methodology documentation."""

    section_id: str = "methodology"
    default_title: str = "Methodology"

    def render(self) -> str:
        if not self.is_enabled:
            return ""

        content_parts = []

        # Always disclose the load-bearing default-prior assumption: channel
        # effects use a non-negative prior, so a channel cannot be shown to have a
        # negative/zero effect from observational data alone.
        content_parts.append("""
                <div class="methodology-note">
                    <h4>Default prior assumption — read before reallocating</h4>
                    <p>
                        Unless a channel is calibrated against an experiment, its
                        effect uses a <strong>non-negative prior</strong> (Gamma,
                        placing more mass on ROI&nbsp;&gt;&nbsp;1). This regularises
                        noisy observational data, but it means a channel
                        <strong>cannot be shown to have a negative or zero effect
                        from the model alone</strong> — a genuinely under-performing
                        channel is pulled toward a small positive effect rather than
                        flagged as a loser. To let a channel express a negative
                        effect, set a symmetric prior (e.g. Normal) on its
                        <code>roi_prior</code>, or run an incrementality experiment
                        to calibrate it.
                    </p>
                </div>
            """)

        # Model specification
        if self.data.model_specification:
            spec = self.data.model_specification
            content_parts.append(f"""
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
            """)

        # Honest uncertainty principles
        if self.config.methodology_note:
            ci_level = int(self.section_config.credible_interval * 100)
            content_parts.append(f"""
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
            """)

        return self._render_section_wrapper("\n".join(content_parts))


class CausalAssumptionsSection(Section):
    """Causal assumptions, identification strategy and sensitivity to unobserved
    confounding.

    Always renders the no-unobserved-confounding / SUTVA caveat (the honest
    framing a causal-branded tool owes its readers). When the data bundle carries
    ``causal_assumptions`` (identification strategy, assumed confounders, and the
    robustness-value table from
    ``mmm_framework.validation``), those details are rendered too.
    """

    section_id: str = "causal-assumptions"
    default_title: str = "Causal Assumptions"

    def render(self) -> str:
        if not self.is_enabled:
            return ""

        parts: list[str] = []

        # Always-on caveat banner.
        parts.append("""
            <div class="methodology-note" style="border-left: 4px solid #d4a86a;">
                <h4>⚠️ Identification rests on assumptions, not just fit</h4>
                <p>
                    The channel effects below are causal <em>only if</em> every common
                    cause of media spend and the KPI is measured and adjusted for
                    (<strong>no unobserved confounding</strong>), and one unit's spend
                    does not affect another's outcome (<strong>SUTVA</strong>). In
                    marketing the dominant hidden confounder is
                    <strong>unobserved demand</strong> &mdash; budgets rise when demand
                    is expected to rise &mdash; which no adjustment set can remove.
                    Good fit, tight
                    intervals and passing posterior-predictive checks are all
                    compatible with confounding bias. Effects should be anchored with
                    randomized geo-lift / incrementality experiments where the stakes
                    are high.
                </p>
            </div>
            """)

        ca = self.data.causal_assumptions or {}

        strategy = ca.get("identification_strategy")
        if strategy:
            parts.append(f"""
                <div class="methodology-note">
                    <h4>Identification Strategy</h4>
                    <p>{strategy}</p>
                </div>
                """)

        confounders = ca.get("assumed_confounders")
        if confounders:
            items = "".join(f"<li>{html.escape(c)}</li>" for c in confounders)
            parts.append(f"""
                <div class="methodology-note">
                    <h4>Assumed Confounders (adjusted for)</h4>
                    <ul style="margin: 0.5rem 0 0 1.5rem;">{items}</ul>
                </div>
                """)

        robustness = ca.get("robustness")
        if robustness and robustness.get("channels"):
            parts.append(self._render_robustness_table(robustness))

        return self._render_section_wrapper("\n".join(parts))

    def _render_robustness_table(self, robustness: dict) -> str:
        rows = []
        for ch in robustness["channels"]:
            rv = ch.get("robustness_value")
            pr2 = ch.get("partial_r2")
            fragile = ch.get("is_fragile")
            rv_s = f"{rv:.3f}" if isinstance(rv, (int, float)) else "-"
            pr2_s = f"{pr2:.3f}" if isinstance(pr2, (int, float)) else "-"
            cls = "negative" if fragile else "positive"
            status = "Fragile" if fragile else "Robust"
            rows.append(f"""
                <tr>
                    <td>{html.escape(ch.get("channel", "?"))}</td>
                    <td class="mono">{rv_s}</td>
                    <td class="mono">{pr2_s}</td>
                    <td class="{cls}">{status}</td>
                </tr>
                """)
        caveat = robustness.get("caveat", "")
        return f"""
            <h3>Robustness to Unobserved Confounding</h3>
            <p>
                The <strong>robustness value</strong> is the share of residual variance
                an unobserved confounder would need to explain in <em>both</em> a
                channel's spend and the KPI to nullify its estimated effect. Larger is
                more robust; channels flagged "Fragile" could be overturned by a weak
                confounder.
            </p>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Channel</th>
                        <th>Robustness Value</th>
                        <th>Partial R²</th>
                        <th>Assessment</th>
                    </tr>
                </thead>
                <tbody>{"".join(rows)}</tbody>
            </table>
            <p class="section-subtitle" style="margin-top: 0.75rem;">{caveat}</p>
        """


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

            # Determine status. R-hat/ESS are ``None`` for an approximate
            # (MAP/ADVI) fit — render those as N/A rather than crashing.
            divergences = diag.get("divergences", 0)
            rhat_max = diag.get("rhat_max")
            ess_min = diag.get("ess_bulk_min")

            div_val = "N/A" if divergences is None else f"{divergences}"
            divergence_status = (
                "—"
                if divergences is None
                else (
                    "✅ Pass" if divergences == 0 else f"⚠️ {divergences} divergences"
                )
            )
            if rhat_max is None:
                rhat_val, rhat_status = "N/A", "— (approximate fit)"
            else:
                rhat_val = f"{rhat_max:.4f}"
                rhat_status = "✅ Pass" if rhat_max < 1.01 else f"⚠️ {rhat_max:.3f}"
            if ess_min is None:
                ess_val, ess_status = "N/A", "— (approximate fit)"
            else:
                ess_val = f"{ess_min:.0f}"
                ess_status = "✅ Pass" if ess_min > 400 else f"⚠️ {ess_min:.0f}"

            content_parts.append(f"""
                <h3>Convergence Diagnostics</h3>
                <table class="data-table" style="max-width: 500px;">
                    <thead><tr><th>Diagnostic</th><th>Value</th><th>Status</th></tr></thead>
                    <tbody>
                        <tr><td>Divergences</td><td class="mono">{div_val}</td><td>{divergence_status}</td></tr>
                        <tr><td>R-hat (max)</td><td class="mono">{rhat_val}</td><td>{rhat_status}</td></tr>
                        <tr><td>ESS bulk (min)</td><td class="mono">{ess_val}</td><td>{ess_status}</td></tr>
                    </tbody>
                </table>
            """)

        # Trace plots
        if self.data.trace_data and self.data.trace_parameters:
            trace_chart = charts.create_trace_plot(
                parameter_names=self.data.trace_parameters[:6],  # Limit to 6
                traces_data=self.data.trace_data,
                config=self.config,
                chart_config=ChartConfig(height=180),
            )
            content_parts.append(f"""
                <h3>Trace Plots</h3>
                <p>Visual inspection of MCMC chains for key parameters.</p>
                {trace_chart}
            """)

        # Prior/posterior comparison
        if self.data.prior_samples and self.data.posterior_samples:
            parameter_names = list(self.data.posterior_samples.keys())
            len_params = len(parameter_names)
            prior_post_chart = charts.create_prior_posterior_chart(
                parameter_names=parameter_names,
                prior_samples=self.data.prior_samples,
                posterior_samples=self.data.posterior_samples,
                config=self.config,
                chart_config=ChartConfig(height=int(250 * min(len_params / 6, 1))),
            )
            content_parts.append(f"""
                <h3>Prior vs Posterior</h3>
                <p>Comparison shows how data updated prior beliefs.</p>
                {prior_post_chart}
            """)

        if not content_parts:
            return ""

        return self._render_section_wrapper("\n".join(content_parts))


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
        content_parts.append(f"""
            <h3>Performance by Geography</h3>
            <p>Regional breakdown with {ci_level}% credible intervals.</p>
        """)

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

            table_rows.append(f"""
                <tr>
                    <td><strong>{html.escape(geo)}</strong></td>
                    <td class="mono">{rev_str}</td>
                    <td class="mono">{roi_str}</td>
                    <td class="mono muted">{roi_ci}</td>
                    <td class="mono">{contrib_str}</td>
                </tr>
            """)

        content_parts.append(f"""
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
        """)

        # Geographic ROI heatmap/chart
        if self.data.geo_roi:
            geo_chart = charts.create_geo_roi_heatmap(
                geo_names=self.data.geo_names,
                channel_names=self.data.channel_names or [],
                geo_roi=self.data.geo_roi,
                config=self.config,
                chart_config=ChartConfig(
                    height=max(300, len(self.data.geo_names) * 40)
                ),
            )
            content_parts.append(f"""
                <h3>Channel ROI by Geography</h3>
                <p>Heatmap showing channel performance across regions. Darker colors indicate higher ROI.</p>
                {geo_chart}
            """)

        # Geographic contribution breakdown
        if self.data.geo_contribution:
            geo_decomp = charts.create_geo_decomposition_chart(
                geo_names=self.data.geo_names,
                geo_contribution=self.data.geo_contribution,
                config=self.config,
                chart_config=ChartConfig(height=400),
            )
            content_parts.append(f"""
                <h3>Contribution Decomposition by Geography</h3>
                {geo_decomp}
            """)

        return self._render_section_wrapper("\n".join(content_parts))


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
        content_parts.append(f"""
            <h3>Indirect Effects Through Mediators</h3>
            <p>Marketing affects sales both directly and indirectly through intermediate outcomes 
            (e.g., awareness, consideration). This analysis decomposes total effects into direct 
            and mediated pathways with {ci_level}% credible intervals.</p>
        """)

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
        content_parts.append("""
            <h3>Effect Decomposition by Channel</h3>
        """)

        table_rows = []
        for channel in self.data.channel_names or []:
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

            table_rows.append(f"""
                <tr>
                    <td><strong>{html.escape(channel)}</strong></td>
                    <td class="mono">{total_str}</td>
                    <td class="mono muted">{total_ci}</td>
                    <td class="mono">{direct_str}</td>
                    <td class="mono">{indirect_str}</td>
                    <td class="mono">{pct_mediated:.1f}%</td>
                </tr>
            """)

        content_parts.append(f"""
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
        """)

        # Mediator time series
        if self.data.mediator_time_series and self.data.dates is not None:
            mediator_ts_chart = charts.create_mediator_time_series(
                dates=self.data.dates,
                mediator_names=self.data.mediator_names,
                mediator_time_series=self.data.mediator_time_series,
                config=self.config,
                chart_config=ChartConfig(height=350),
            )
            content_parts.append(f"""
                <h3>Mediator Values Over Time</h3>
                <p>Tracking awareness, consideration, and other intermediate outcomes.</p>
                {mediator_ts_chart}
            """)

        # Interpretation callout
        if self.data.total_indirect_effect:
            indirect = self.data.total_indirect_effect
            content_parts.append(f"""
                <div class="callout">
                    <strong>Key Insight:</strong> Approximately 
                    <strong>{self._format_percentage(indirect.get('mean', 0))}</strong> 
                    [{self._format_percentage(indirect.get('lower', 0))}, 
                    {self._format_percentage(indirect.get('upper', 0))}] of total marketing 
                    effect operates through measured mediators. Direct response accounts for 
                    the remainder.
                </div>
            """)

        return self._render_section_wrapper("\n".join(content_parts))


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
        content_parts.append(f"""
            <h3>Cross-Product Effects</h3>
            <p>Marketing for one product may cannibalize sales of another (substitution) or 
            boost them (halo effects). This matrix shows cross-product effects with {ci_level}% 
            credible intervals. Negative values indicate cannibalization; positive values 
            indicate synergy.</p>
        """)

        # Cannibalization heatmap
        cannib_chart = charts.create_cannibalization_heatmap(
            product_names=self.data.product_names,
            cannibalization_matrix=self.data.cannibalization_matrix,
            config=self.config,
            chart_config=ChartConfig(
                height=max(350, len(self.data.product_names) * 50)
            ),
        )
        content_parts.append(cannib_chart)

        # Net effects table
        if self.data.net_product_effects:
            content_parts.append("""
                <h3>Net Product Effects</h3>
                <p>Direct marketing effect minus cannibalization from other products' marketing.</p>
            """)

            table_rows = []
            for product in self.data.product_names:
                effects = self.data.net_product_effects.get(product, {})
                direct = effects.get("direct", 0)
                cannib = effects.get("cannibalization", 0)
                net = effects.get("net", 0)

                # Color coding
                cannib_class = (
                    "danger" if cannib < -0.05 else ("success" if cannib > 0.05 else "")
                )

                table_rows.append(f"""
                    <tr>
                        <td><strong>{html.escape(product)}</strong></td>
                        <td class="mono">{self._format_currency(direct)}</td>
                        <td class="mono {cannib_class}">{self._format_currency(cannib)}</td>
                        <td class="mono"><strong>{self._format_currency(net)}</strong></td>
                        <td class="mono">{(cannib/direct*100) if direct else 0:.1f}%</td>
                    </tr>
                """)

            content_parts.append(f"""
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
            """)

        # Detailed cross-effects table
        content_parts.append("""
            <h3>Detailed Cross-Product Matrix</h3>
            <p>Effect of row product's marketing on column product's sales.</p>
        """)

        # Build matrix table
        header_cells = "<th>Source \\ Target</th>" + "".join(
            f"<th>{html.escape(p)}</th>" for p in self.data.product_names
        )

        matrix_rows = []
        for source in self.data.product_names:
            row_cells = [f"<td><strong>{html.escape(source)}</strong></td>"]
            for target in self.data.product_names:
                if source == target:
                    row_cells.append('<td class="mono muted">—</td>')
                else:
                    effect = self.data.cannibalization_matrix.get(source, {}).get(
                        target, {}
                    )
                    mean = effect.get("mean", 0)
                    lower = effect.get("lower", 0)
                    upper = effect.get("upper", 0)

                    # Color based on sign and significance
                    if upper < 0:
                        cell_class = (
                            "danger"  # Significantly negative (cannibalization)
                        )
                    elif lower > 0:
                        cell_class = "success"  # Significantly positive (synergy)
                    else:
                        cell_class = "muted"  # CI includes zero

                    row_cells.append(
                        f'<td class="mono {cell_class}" title="[{lower:.3f}, {upper:.3f}]">'
                        f"{mean:.3f}</td>"
                    )
            matrix_rows.append(f'<tr>{"".join(row_cells)}</tr>')

        content_parts.append(f"""
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
        """)

        # Key insights
        significant_cannib = []
        significant_synergy = []

        for source in self.data.product_names:
            for target in self.data.product_names:
                if source != target:
                    effect = self.data.cannibalization_matrix.get(source, {}).get(
                        target, {}
                    )
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
                cannib_items = [
                    f"{html.escape(s)} → {html.escape(t)} ({m:.2f})"
                    for s, t, m in top_cannib
                ]
                insights.append(
                    f"<strong>Strongest cannibalization:</strong> {', '.join(cannib_items)}"
                )

            if significant_synergy:
                top_synergy = sorted(significant_synergy, key=lambda x: -x[2])[:3]
                synergy_items = [
                    f"{html.escape(s)} → {html.escape(t)} (+{m:.2f})"
                    for s, t, m in top_synergy
                ]
                insights.append(
                    f"<strong>Strongest synergies:</strong> {', '.join(synergy_items)}"
                )

            content_parts.append(f"""
                <div class="callout">
                    {'<br>'.join(insights)}
                </div>
            """)

        content_parts.append(self._render_outcome_correlations())

        return self._render_section_wrapper("\n".join(content_parts))

    def _render_outcome_correlations(self) -> str:
        """Residual correlation matrix between outcomes (multivariate models)."""
        corr = self.data.outcome_correlations
        names = self.data.product_names
        if corr is None or names is None:
            return ""

        corr = np.asarray(corr, dtype=float)
        if corr.ndim != 2 or corr.shape[0] != len(names):
            return ""

        header_cells = "<th></th>" + "".join(
            f"<th>{html.escape(n)}</th>" for n in names
        )
        rows = []
        for i, source in enumerate(names):
            cells = [f"<td><strong>{html.escape(source)}</strong></td>"]
            for j in range(len(names)):
                value = corr[i, j]
                cls = "muted" if i == j else ("mono")
                cells.append(f'<td class="mono {cls}">{value:.2f}</td>')
            rows.append(f'<tr>{"".join(cells)}</tr>')

        return f"""
            <h3>Residual Correlation Between Outcomes</h3>
            <p>Correlation of the outcomes' unexplained variation (e.g., shared demand
            shocks). High residual correlation means the outcomes move together beyond
            what media and cross-effects explain.</p>
            <table class="data-table matrix-table">
                <thead><tr>{header_cells}</tr></thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        """


# Registry of available sections
class FactorAnalysisSection(Section):
    """Latent-structure section for a non-MMM model — a CFA's loadings, an LCA's
    class profiles, etc. Renders a summary **table** + the model's declared
    **estimands** as cards, with family-specific headings supplied by the bundle
    (``latent_section_title`` / ``latent_table_title`` / ``latent_estimands_title``).
    Empty unless the bundle carries ``factor_loadings`` or ``cfa_fit_indices``."""

    section_id: str = "factor-analysis"
    default_title: str = "Latent Structure"

    @property
    def title(self) -> str:
        return (
            self.section_config.title
            or getattr(self.data, "latent_section_title", None)
            or self.default_title
        )

    def render(self) -> str:
        if not self.is_enabled:
            return ""
        table_rows = self.data.factor_loadings
        estimands = self.data.cfa_fit_indices
        if not table_rows and not estimands:
            return ""

        parts: list[str] = []
        if estimands:
            est_title = (
                getattr(self.data, "latent_estimands_title", None) or "Estimands"
            )
            parts.append(f"<h3>{est_title}</h3>")
            parts.append(self._render_estimand_cards(estimands))
        if table_rows:
            tbl_title = getattr(self.data, "latent_table_title", None) or "Summary"
            parts.append(f"<h3>{tbl_title}</h3>")
            parts.append(self._render_table(table_rows))
        return self._render_section_wrapper("\n".join(parts))

    def _render_estimand_cards(self, estimands: dict[str, dict[str, float]]) -> str:
        cards = []
        for name, v in estimands.items():
            mean = v.get("mean", float("nan"))
            lo, hi = v.get("lower", mean), v.get("upper", mean)
            ci_html = f'<div class="ci">[{lo:.3f}, {hi:.3f}]</div>' if hi != lo else ""
            cards.append(f"""
                <div class="metric-card">
                    <div class="value">{mean:.3f}</div>
                    <div class="label">{name.upper()}</div>
                    {ci_html}
                </div>
                """)
        return f'<div class="metrics-grid">{"".join(cards)}</div>'

    def _render_table(self, rows: list[dict]) -> str:
        """Column-agnostic table: renders whatever columns the summary rows carry
        (so one renderer serves CFA loadings, LCA class profiles, …)."""
        if not rows:
            return ""
        cols = list(rows[0].keys())
        header = "".join(f"<th>{c.replace('_', ' ').title()}</th>" for c in cols)
        body = []
        for r in rows:
            cells = []
            for c in cols:
                v = r.get(c, "")
                cell = f"{v:.3f}" if isinstance(v, float) else html.escape(str(v))
                klass = ' class="mono"' if isinstance(v, float) else ""
                cells.append(f"<td{klass}>{cell}</td>")
            body.append(f"<tr>{''.join(cells)}</tr>")
        return f"""
            <table class="data-table">
                <thead><tr>{header}</tr></thead>
                <tbody>{"".join(body)}</tbody>
            </table>
        """


class EstimandsSection(Section):
    """Declared / default estimand results with credible intervals.

    Renders the model's pre-specified causal quantities (e.g. contribution ROI,
    marginal ROAS, incremental contribution per channel) as a single table of
    point estimate + credible interval. Data-driven: empty unless the bundle
    carries ``estimands`` (populated by the extractor for MMM models)."""

    section_id: str = "estimands"
    default_title: str = "Estimand Results"

    #: Human-friendly labels keyed by the built-in estimand *name* (the result-key
    #: prefix, e.g. "contribution_roi"); unknown names fall back to a title-cased
    #: version. Keyed by name, not the result ``kind`` — several built-ins share
    #: kind "roi" (contribution_roi, counterfactual_roi) but want distinct labels.
    _KIND_LABELS = {
        "contribution_roi": "Contribution ROI",
        "counterfactual_roi": "Counterfactual ROI",
        "marginal_roas": "Marginal ROAS",
        "contribution": "Incremental contribution",
        "awareness_lift": "Awareness lift",
        "cost_per_conversion": "Cost per conversion",
    }

    def render(self) -> str:
        if not self.is_enabled:
            return ""
        estimands = self.data.estimands
        if not estimands:
            return ""

        intro = (
            "<p>Each estimand below is a pre-specified causal quantity realized "
            "from the full posterior, reported as a point estimate with a credible "
            "interval. The interval — not the point value — is the basis for a "
            "decision.</p>"
        )
        table = self._render_estimand_table(estimands)
        if not table:
            return ""
        note = (
            '<div class="methodology-note"><p><strong>Reading this table:</strong> '
            '"Strong" means the credible interval excludes the no-effect reference '
            '(ROI/ROAS against 1.0, contribution against 0); "Uncertain" means it '
            "does not, so the sign of the effect is not resolved by the data.</p></div>"
        )
        return self._render_section_wrapper(f"{intro}{table}{note}")

    def _kind_label(self, name: str) -> str:
        return self._KIND_LABELS.get(
            name, (name or "Estimand").replace("_", " ").title()
        )

    @staticmethod
    def _is_ratio_kind(kind: str, units: str) -> bool:
        k, u = (kind or "").lower(), (units or "").lower()
        return "roi" in k or "roas" in k or u in {"ratio", "x", "multiple"}

    def _fmt_value(self, kind: str, units: str, value: float) -> str:
        if self._is_ratio_kind(kind, units):
            return f"{value:.2f}"
        if "contribution" in (kind or "").lower() or (units or "").lower() in {
            "$",
            "usd",
            "currency",
            "dollars",
            "kpi",
        }:
            return self._format_currency(value)
        return f"{value:.3f}"

    def _render_estimand_table(self, estimands: dict[str, dict]) -> str:
        items = [
            (key, v)
            for key, v in estimands.items()
            if v.get("mean") is not None and np.isfinite(v.get("mean", float("nan")))
        ]
        if not items:
            return ""

        # Header CI label from the modal hdi_prob across rows (estimands usually
        # share one; differing probs still get a sensible single header).
        probs = [v.get("hdi_prob", 0.94) for _, v in items]
        ci_pct = int(round((max(set(probs), key=probs.count) if probs else 0.94) * 100))

        # Group by estimand name (the dict key prefix, e.g. "contribution_roi"),
        # then by descending |mean| within each group. The label comes from the
        # NAME, not the result ``kind`` — several built-ins share kind "roi".
        def _name_of(key: str) -> str:
            return key.split(":", 1)[0]

        items.sort(
            key=lambda kv: (_name_of(kv[0]), -abs(float(kv[1].get("mean", 0.0))))
        )

        rows = []
        for key, v in items:
            kind = str(v.get("kind", ""))
            units = str(v.get("units", ""))
            mean = float(v["mean"])
            lower = v.get("lower")
            upper = v.get("upper")
            name, _, channel = key.partition(":")
            target = channel if channel else "—"

            val_str = self._fmt_value(kind, units, mean)
            if lower is not None and upper is not None:
                ci_str = (
                    f"[{self._fmt_value(kind, units, float(lower))}, "
                    f"{self._fmt_value(kind, units, float(upper))}]"
                )
            else:
                ci_str = "—"

            # Break-even reference: trust the estimand's measurement metadata
            # when present (efficiency metrics carry reference 0 even though their
            # kind is still "roi"); else fall back to the kind/units heuristic.
            if v.get("reference") is not None:
                ref = float(v["reference"])
            else:
                ref = 1.0 if self._is_ratio_kind(kind, units) else 0.0
            if lower is not None and upper is not None and float(lower) > ref:
                conf_class, status = "positive", "Strong"
            elif lower is not None and upper is not None and float(upper) < ref:
                conf_class, status = "negative", "Below reference"
            else:
                conf_class, status = "uncertain", "Uncertain"

            rows.append(f"""
                <tr>
                    <td>{html.escape(self._kind_label(name))}</td>
                    <td>{html.escape(target)}</td>
                    <td class="mono">{val_str}</td>
                    <td class="mono">{ci_str}</td>
                    <td class="{conf_class}">{status}</td>
                </tr>
            """)

        return f"""
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Estimand</th>
                        <th>Target</th>
                        <th>Estimate</th>
                        <th>{ci_pct}% CI</th>
                        <th>Evidence</th>
                    </tr>
                </thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        """


class PosteriorPredictiveSection(Section):
    """Posterior-predictive goodness-of-fit checks for MMM models.

    Default-on for MMM models: shows whether the fitted model can reproduce the
    data it was trained on, via observed-vs-predicted, a density overlay of
    replicated datasets, predictive-interval calibration, and residual structure.
    Data-driven: empty unless the bundle carries ``posterior_predictive``."""

    section_id: str = "posterior-predictive"
    default_title: str = "Posterior Predictive Checks"

    _BAYES_P_LABELS = {
        "mean": "Mean",
        "std": "Std. deviation",
        "min": "Minimum",
        "max": "Maximum",
    }

    def render(self) -> str:
        if not self.is_enabled:
            return ""
        pp = self.data.posterior_predictive
        if not pp or pp.get("observed") is None:
            return ""

        observed = np.asarray(pp["observed"], dtype=float)
        pred_mean = pp.get("pred_mean")
        if pred_mean is None or len(np.asarray(pred_mean)) != len(observed):
            return ""
        pred_mean = np.asarray(pred_mean, dtype=float)

        height = self.section_config.chart_height or 400
        cc = ChartConfig(height=height)

        intro = (
            "<p>These checks ask the honest question of fit: can the model "
            "reproduce the data it was trained on? Each view compares the observed "
            "KPI against the model's posterior-predictive distribution — the band "
            "of outcomes the model considers plausible, including observation "
            "noise.</p>"
        )

        parts = [intro, self._render_summary(pp, observed, pred_mean)]

        # Observed vs predicted (with predictive-interval error bars).
        parts.append(
            charts.create_ppc_observed_vs_predicted(
                observed=observed,
                pred_mean=pred_mean,
                pred_lower=pp.get("pred_lower"),
                pred_upper=pp.get("pred_upper"),
                config=self.config,
                chart_config=cc,
            )
        )

        # Density overlay (observed vs replicated datasets).
        if pp.get("samples") is not None:
            parts.append("<h3>Predictive density overlay</h3>")
            parts.append(
                charts.create_ppc_density_overlay(
                    observed=observed,
                    samples=np.asarray(pp["samples"], dtype=float),
                    config=self.config,
                    chart_config=cc,
                )
            )

        # Interval calibration.
        if pp.get("coverage"):
            parts.append("<h3>Predictive-interval calibration</h3>")
            parts.append(
                "<p>Across nominal interval widths, the share of observations that "
                "actually fell inside the model's predictive interval. A "
                "well-calibrated model tracks the diagonal.</p>"
            )
            parts.append(
                charts.create_ppc_interval_calibration(
                    coverage=pp["coverage"],
                    config=self.config,
                    chart_config=ChartConfig(height=min(height, 380)),
                )
            )

        # Residual structure.
        parts.append("<h3>Residual structure</h3>")
        parts.append(
            charts.create_ppc_residual_plot(
                observed=observed,
                pred_mean=pred_mean,
                config=self.config,
                chart_config=ChartConfig(height=min(height, 380)),
            )
        )

        # Posterior-predictive p-values for summary statistics.
        bayes_p = pp.get("bayes_p")
        if bayes_p:
            parts.append(self._render_bayes_p(bayes_p))

        parts.append(
            '<div class="methodology-note"><p><strong>How to read these:</strong> '
            "observed points should sit near the 45° line and inside their "
            "predictive intervals; the observed density should nest within the "
            "replicated cloud; calibration should track the diagonal; residuals "
            "should scatter structurelessly around zero. Posterior-predictive "
            "p-values near 0 or 1 flag a summary statistic the model cannot "
            "reproduce.</p></div>"
        )

        return self._render_section_wrapper("\n".join(parts))

    def _render_summary(
        self, pp: dict, observed: np.ndarray, pred_mean: np.ndarray
    ) -> str:
        """Headline goodness-of-fit cards: R² and interval coverage."""
        cards = []

        r2 = pp.get("r2")
        if r2 is None and self.data.fit_statistics:
            r2 = self.data.fit_statistics.get("r2")
        if r2 is not None:
            cards.append(
                f'<div class="metric-card"><div class="value">{float(r2):.3f}</div>'
                '<div class="label">R² (observed vs predicted)</div></div>'
            )

        coverage = pp.get("coverage") or []
        if coverage:
            # Report the headline interval (nominal closest to the report CI).
            target = float(
                pp.get("ci_level", self.section_config.credible_interval or 0.8)
            )
            best = min(coverage, key=lambda d: abs(float(d.get("nominal", 0)) - target))
            nominal = float(best.get("nominal", target))
            empirical = float(best.get("empirical", 0.0))
            cards.append(
                f'<div class="metric-card"><div class="value">{empirical:.0%}</div>'
                f'<div class="label">Observed coverage of the {nominal:.0%} '
                "interval</div></div>"
            )

        if not cards:
            return ""
        return f'<div class="metrics-grid">{"".join(cards)}</div>'

    def _render_bayes_p(self, bayes_p: dict[str, float]) -> str:
        rows = []
        for stat, p in bayes_p.items():
            if p is None:
                continue
            p = float(p)
            label = self._BAYES_P_LABELS.get(stat, stat.title())
            ok = 0.05 <= p <= 0.95
            status = "✅ Reproduced" if ok else "⚠️ Poorly reproduced"
            rows.append(
                f'<tr><td>{label}</td><td class="mono">{p:.2f}</td><td>{status}</td></tr>'
            )
        if not rows:
            return ""
        return f"""
            <h3>Predictive p-values (summary statistics)</h3>
            <p>The probability that a replicated dataset is more extreme than the
            observed data on each statistic. Values near 0.5 indicate the model
            reproduces that feature; values near 0 or 1 indicate it does not.</p>
            <table class="data-table" style="max-width: 500px;">
                <thead><tr><th>Statistic</th><th>p-value</th><th>Status</th></tr></thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        """


class AllocationSection(Section):
    """Budget-allocation plan: recommended per-channel (and per-geo) spend plus an
    optional forward flighting calendar.

    Data-driven and default-off: renders only when the bundle carries
    ``allocation_results`` (a plan computed by the ``plan_budget`` op), so a normal
    model report that has no plan attached silently omits the section."""

    section_id: str = "allocation"
    default_title: str = "Budget Allocation Plan"

    def render(self) -> str:
        if not self.is_enabled:
            return ""
        alloc = getattr(self.data, "allocation_results", None)
        if not alloc or not alloc.get("allocation"):
            return ""

        parts = [self._render_headline(alloc), self._render_alloc_table(alloc)]
        if alloc.get("geo_allocation"):
            parts.append(self._render_geo_table(alloc))
        if alloc.get("flighting", {}).get("schedule"):
            parts.append(self._render_flighting_table(alloc["flighting"]))
        note = (
            '<div class="methodology-note"><p>The recommended allocation maximizes '
            "expected KPI contribution within the spend range the model has evidence "
            "for; the uplift interval — not the point estimate — is the basis for a "
            "decision. The flighting calendar distributes each channel's budget across "
            "future periods.</p></div>"
        )
        return self._render_section_wrapper("".join(parts) + note)

    def _render_headline(self, alloc: dict) -> str:
        total = alloc.get("total_budget", 0.0)
        uplift = alloc.get("expected_uplift", 0.0)
        hdi = alloc.get("uplift_hdi", [0.0, 0.0])
        prob = alloc.get("prob_positive_uplift", 0.0)
        regret = alloc.get("expected_regret")
        n_extrap = int(alloc.get("n_extrapolated", 0) or 0)

        # Lead with the DECISION CONFIDENCE, not the allocation (issue #105):
        # planners defend decisions upward and need "how sure are you this beats
        # the current plan?" first.
        conf_cls = (
            "positive" if prob >= 0.8 else ("uncertain" if prob >= 0.6 else "negative")
        )
        regret_txt = (
            f" If the model's uncertainty resolves against it, you'd expect to leave "
            f"about {self._format_currency(regret)} of KPI on the table versus a "
            f"perfectly-informed plan (expected regret)."
            if regret is not None
            else ""
        )
        extrap_txt = (
            f" <strong>{n_extrap} channel(s)</strong> are recommended beyond their "
            f"observed spend range — those moves are extrapolated (flagged below)."
            if n_extrap
            else ""
        )
        lead = (
            f'<div class="callout {conf_cls}">'
            f"<strong>{prob:.0%} chance this plan beats the current allocation.</strong>"
            f"{regret_txt}{extrap_txt}</div>"
        )
        regret_card = (
            f"""<div class="metric-card">
                    <div class="value">{self._format_currency(regret)}</div>
                    <div class="label">Expected regret</div>
                    <div class="ci">vs a perfectly-informed plan</div>
                </div>"""
            if regret is not None
            else ""
        )
        return f"""
            {lead}
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="value">{prob:.0%}</div>
                    <div class="label">Chance it beats today</div>
                </div>
                <div class="metric-card">
                    <div class="value">{self._format_currency(uplift)}</div>
                    <div class="label">Expected KPI uplift</div>
                    <div class="ci">90% [{self._format_currency(hdi[0])}, {self._format_currency(hdi[1])}]</div>
                </div>
                {regret_card}
                <div class="metric-card">
                    <div class="value">{self._format_currency(total)}</div>
                    <div class="label">Budget allocated</div>
                </div>
            </div>
        """

    def _render_alloc_table(self, alloc: dict) -> str:
        # Show the extrapolation column only when the optimizer supplied the flag.
        has_range = any(
            "within_observed_range" in r for r in alloc.get("allocation", [])
        )
        rows = []
        for r in alloc["allocation"]:
            ch = html.escape(str(r.get("channel", "")))
            cur = float(r.get("current_spend", 0.0))
            opt = float(r.get("optimal_spend", 0.0))
            chg = float(r.get("change_pct", 0.0))
            cls = "positive" if chg > 1 else ("negative" if chg < -1 else "uncertain")
            # Recommended spend with its credible interval (issue #105).
            opt_cell = self._format_currency(opt)
            if r.get("optimal_spend_p5") is not None:
                opt_cell += (
                    f'<br><span class="ci">[{self._format_currency(float(r["optimal_spend_p5"]))}, '
                    f'{self._format_currency(float(r["optimal_spend_p95"]))}]</span>'
                )
            range_cell = ""
            if has_range:
                within = r.get("within_observed_range", True)
                if within:
                    range_cell = '<td class="positive">In tested range</td>'
                else:
                    mo = r.get("max_obs_multiplier")
                    tip = (
                        f"Recommended {r.get('recommended_multiplier', 0):.2f}× current "
                        f"spend, but the model has only observed up to "
                        f"~{mo:.2f}× — beyond that the response curve is extrapolated."
                        if mo is not None
                        else "Beyond the observed spend range — extrapolated."
                    )
                    range_cell = (
                        f'<td class="negative" title="{html.escape(tip)}">'
                        f"⚠ Extrapolated</td>"
                    )
            rows.append(
                f"<tr><td>{ch}</td>"
                f'<td class="mono">{self._format_currency(cur)}</td>'
                f'<td class="mono">{opt_cell}</td>'
                f'<td class="{cls}">{chg:+.0f}%</td>{range_cell}</tr>'
            )
        range_head = "<th>Range</th>" if has_range else ""
        return f"""
            <h3>Recommended allocation</h3>
            <table class="data-table">
                <thead><tr><th>Channel</th><th>Current spend</th>
                    <th>Recommended spend</th><th>Change</th>{range_head}</tr></thead>
                <tbody>{"".join(rows)}</tbody>
            </table>
        """

    def _render_geo_table(self, alloc: dict) -> str:
        rows = []
        for r in alloc["geo_allocation"]:
            geo = html.escape(str(r.get("geo", "")))
            ch = html.escape(str(r.get("channel", "")))
            opt = float(r.get("optimal_spend", 0.0))
            chg = float(r.get("change_pct", 0.0))
            rows.append(
                f"<tr><td>{geo}</td><td>{ch}</td>"
                f'<td class="mono">{self._format_currency(opt)}</td>'
                f'<td class="mono">{chg:+.0f}%</td></tr>'
            )
        return f"""
            <h3>Allocation by geography</h3>
            <table class="data-table">
                <thead><tr><th>Geography</th><th>Channel</th>
                    <th>Recommended spend</th><th>Change</th></tr></thead>
                <tbody>{"".join(rows)}</tbody>
            </table>
        """

    def _render_flighting_table(self, fl: dict) -> str:
        channels = list(fl.get("channels", []))
        head = (
            "<tr><th>Period</th>"
            + "".join(f"<th>{html.escape(str(c))}</th>" for c in channels)
            + "<th>Total</th></tr>"
        )
        rows = []
        for r in fl["schedule"]:
            cells = "".join(
                f'<td class="mono">{self._format_currency(float(r.get(c, 0.0)))}</td>'
                for c in channels
            )
            rows.append(
                f"<tr><td>{html.escape(str(r.get('period', '')))}</td>{cells}"
                f'<td class="mono">{self._format_currency(float(r.get("total", 0.0)))}</td></tr>'
            )
        pat = html.escape(str(fl.get("pattern", "")))
        return f"""
            <h3>Flighting calendar ({pat})</h3>
            <table class="data-table">
                <thead>{head}</thead>
                <tbody>{"".join(rows)}</tbody>
            </table>
        """


SECTION_REGISTRY: dict[str, type[Section]] = {
    "executive_summary": ExecutiveSummarySection,
    "allocation": AllocationSection,
    "factor_analysis": FactorAnalysisSection,
    "model_fit": ModelFitSection,
    "posterior_predictive": PosteriorPredictiveSection,
    "estimands": EstimandsSection,
    "channel_roi": ChannelROISection,
    "decomposition": DecompositionSection,
    "saturation": SaturationSection,
    "sensitivity": SensitivitySection,
    "causal_assumptions": CausalAssumptionsSection,
    "methodology": MethodologySection,
    "diagnostics": DiagnosticsSection,
    "geographic": GeographicSection,
    "mediators": MediatorSection,
    "cannibalization": CannibalizationSection,
}
