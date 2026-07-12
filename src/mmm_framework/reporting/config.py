"""
Configuration classes for MMM report generation.

Provides immutable, validated configuration for report customization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Any
from enum import Enum

from .design_tokens import TOKENS


class ColorPalette(Enum):
    """Pre-defined color palettes for reports."""

    SAGE = "sage"  # Default: sage green with blue accents
    CORPORATE = "corporate"  # Blue/gray professional
    WARM = "warm"  # Warm earth tones
    MONOCHROME = "monochrome"  # Grayscale
    AUGUR = "augur"  # Editorial cream/ink with sage/gold/steel/rust evidence tiers


@dataclass(frozen=True)
class ColorScheme:
    """Color scheme for report styling."""

    primary: str = TOKENS.primary
    primary_dark: str = TOKENS.primary_dark
    accent: str = TOKENS.accent
    accent_dark: str = TOKENS.accent_dark
    warning: str = TOKENS.warning
    danger: str = TOKENS.danger
    success: str = TOKENS.success
    text: str = TOKENS.text
    text_muted: str = TOKENS.text_muted
    background: str = TOKENS.background
    background_alt: str = TOKENS.background_alt
    surface: str = TOKENS.surface
    border: str = TOKENS.border
    # Font used for in-chart text (Plotly layout font). Default keeps the
    # historical "Source Sans 3" so existing reports are byte-identical; the
    # Augur palette overrides it to "IBM Plex Sans" so charts match the shell.
    font_sans: str = "Source Sans 3, sans-serif"

    @classmethod
    def from_palette(cls, palette: ColorPalette) -> ColorScheme:
        """Create color scheme from a named palette."""
        palettes = {
            ColorPalette.SAGE: cls(),  # Default
            ColorPalette.CORPORATE: cls(
                primary="#4a6fa5",
                primary_dark="#3a5a8a",
                accent="#5a8fa5",
                accent_dark="#4a7a8a",
                warning="#d4a86a",
                danger="#c97067",
                success="#5abf7a",
                text="#2d3a4d",
                text_muted="#5a6b7a",
                background="#f8f9fa",
                background_alt="#e9ecef",
                surface="#ffffff",
                border="#dee2e6",
            ),
            ColorPalette.WARM: cls(
                primary="#b87c4c",
                primary_dark="#9a6a3a",
                accent="#8fa86a",
                accent_dark="#7a8a5a",
                warning="#d4a86a",
                danger="#c97067",
                success="#6abf8a",
                text="#3d3a2d",
                text_muted="#6b6a5a",
                background="#faf9f7",
                background_alt="#f4f0ed",
                surface="#ffffff",
                border="#ddd4d0",
            ),
            ColorPalette.MONOCHROME: cls(
                primary="#555555",
                primary_dark="#333333",
                accent="#777777",
                accent_dark="#555555",
                warning="#999999",
                danger="#666666",
                success="#888888",
                text="#222222",
                text_muted="#666666",
                background="#fafafa",
                background_alt="#f0f0f0",
                surface="#ffffff",
                border="#dddddd",
            ),
            # Augur "Media Performance Readout" palette: cream page, ink text,
            # sage (scale) primary, steel (hold) accent, gold (test) warning,
            # rust (reduce) danger. Charts inherit these via to_plotly_layout.
            ColorPalette.AUGUR: cls(
                primary="#5a7a3a",  # sage-700 (scale / profitable)
                primary_dark="#4a6d2a",  # sage-800
                accent="#4a6d8a",  # steel-600 (hold / marginal)
                accent_dark="#3a5a75",  # steel-700
                warning="#b8860b",  # gold-600 (test / unproven)
                danger="#a04535",  # rust-600 (reduce / below break-even)
                success="#5a7a3a",  # sage
                text="#3a4838",  # ink-700 (body)
                text_muted="#7a8a78",  # ink-400
                background="#faf8f3",  # cream-50
                background_alt="#f3f0e6",  # cream-100
                surface="#ffffff",
                border="#e8e4d5",  # line-200
                font_sans="IBM Plex Sans, system-ui, sans-serif",
            ),
        }
        return palettes.get(palette, cls())


@dataclass(frozen=True)
class ChannelColors:
    """Color mapping for media channels."""

    colors: dict[str, str] = field(
        default_factory=lambda: {
            "TV": "#6a8fa8",
            "Paid_Search": "#8fa86a",
            "Paid_Social": "#a88f6a",
            "Display": "#8f6aa8",
            "Radio": "#d4a86a",
            "Print": "#c97067",
            "OOH": "#6abf8a",
            "Video": "#bf6a8f",
            "Email": "#6abfbf",
            "Baseline": "#5a6b5a",
        }
    )

    def get(self, channel: str) -> str:
        """Get color for channel, with fallback to hash-based color."""
        if channel in self.colors:
            return self.colors[channel]
        # Generate consistent color from channel name hash
        h = hash(channel) % 360
        return f"hsl({h}, 55%, 55%)"

    def with_channel(self, channel: str, color: str) -> ChannelColors:
        """Return new ChannelColors with added/updated channel color."""
        new_colors = dict(self.colors)
        new_colors[channel] = color
        return ChannelColors(colors=new_colors)


@dataclass(frozen=True)
class SectionConfig:
    """Configuration for individual report sections."""

    enabled: bool = True
    title: str | None = None  # Override default title
    subtitle: str | None = None
    show_uncertainty: bool = True
    credible_interval: float = 0.8  # HDI width (0.8 = 80% CI)
    chart_height: int = 400
    chart_width: int | None = None  # None = responsive
    custom_notes: str | None = None

    def with_updates(self, **kwargs) -> SectionConfig:
        """Return new config with updates."""
        return SectionConfig(
            enabled=kwargs.get("enabled", self.enabled),
            title=kwargs.get("title", self.title),
            subtitle=kwargs.get("subtitle", self.subtitle),
            show_uncertainty=kwargs.get("show_uncertainty", self.show_uncertainty),
            credible_interval=kwargs.get("credible_interval", self.credible_interval),
            chart_height=kwargs.get("chart_height", self.chart_height),
            chart_width=kwargs.get("chart_width", self.chart_width),
            custom_notes=kwargs.get("custom_notes", self.custom_notes),
        )


@dataclass(frozen=True)
class ReportConfig:
    """Complete configuration for report generation."""

    # Metadata
    title: str = "Marketing Mix Model Report"
    subtitle: str | None = None
    client: str | None = None
    analysis_period: str | None = None
    generated_date: str | None = None  # Auto-populated if None

    # Model info
    model_version: str | None = None
    framework_version: str | None = None

    # Styling
    color_scheme: ColorScheme = field(default_factory=ColorScheme)
    channel_colors: ChannelColors = field(default_factory=ChannelColors)
    font_family_serif: str = TOKENS.font_serif
    font_family_sans: str = TOKENS.font_sans
    font_family_mono: str = TOKENS.font_mono

    # Report shell / template aesthetic. "default" = the classic card layout;
    # "augur" = the editorial "Media Performance Readout" (narrative, evidence-
    # coded, with masthead + numbered contents nav). Selected by the section set
    # and CSS in MMMReportGenerator.
    shell: Literal["default", "augur"] = "default"

    # Optional pre-computed CMO / media-planner narrative insights (keyed by slot,
    # e.g. "headline", "standfirst", "channel:<name>", "tests", "next_steps").
    # When empty and shell == "augur", the generator fills a deterministic
    # templated fallback. See reporting/insights.py.
    cmo_insights: dict[str, str] = field(default_factory=dict)

    # Global settings
    default_credible_interval: float = 0.8
    currency_symbol: str = "$"
    currency_format: str = ",.0f"  # Python format string
    percentage_format: str = ".1%"
    decimal_format: str = ".2f"
    large_number_format: Literal["short", "full", "scientific"] = "short"

    # Section configurations
    executive_summary: SectionConfig = field(default_factory=SectionConfig)
    model_fit: SectionConfig = field(default_factory=SectionConfig)
    # Posterior-predictive goodness-of-fit checks (default ON for MMM models).
    posterior_predictive: SectionConfig = field(default_factory=SectionConfig)
    # Declared / default estimand results with credible intervals.
    estimands: SectionConfig = field(default_factory=SectionConfig)
    channel_roi: SectionConfig = field(default_factory=SectionConfig)
    decomposition: SectionConfig = field(default_factory=SectionConfig)
    saturation: SectionConfig = field(default_factory=SectionConfig)
    sensitivity: SectionConfig = field(default_factory=SectionConfig)
    # In-flight pacing — planned vs actual delivery (issue #107). Data-gated on
    # the bundle's pacing payload.
    pacing: SectionConfig = field(default_factory=SectionConfig)
    # Short-term vs long-term / brand effect (issue #106). Data-gated on the
    # bundle's adstock split; ``long_term_multiplier`` (below) opts into the
    # assumption-driven long-term scenario.
    long_term: SectionConfig = field(default_factory=SectionConfig)
    # Triangulation panel — MMM × experiment × platform (issue #104). Data-gated:
    # renders only when a triangulation payload is attached to the bundle.
    triangulation: SectionConfig = field(default_factory=SectionConfig)
    # Spec-curve / model-averaging robustness (issue #103). Data-gated: renders
    # only when a spec-curve payload is attached to the bundle.
    spec_curve: SectionConfig = field(default_factory=SectionConfig)
    causal_assumptions: SectionConfig = field(default_factory=SectionConfig)
    methodology: SectionConfig = field(default_factory=SectionConfig)
    diagnostics: SectionConfig = field(default_factory=SectionConfig)
    # Extended model sections
    geographic: SectionConfig = field(default_factory=SectionConfig)
    mediators: SectionConfig = field(default_factory=SectionConfig)
    cannibalization: SectionConfig = field(default_factory=SectionConfig)
    # Non-MMM family sections
    factor_analysis: SectionConfig = field(default_factory=SectionConfig)
    # Augur ("Media Performance Readout") sections. Each Augur section no-ops on
    # missing bundle data, so these default ON; they are only consulted when
    # shell == "augur".
    headline: SectionConfig = field(default_factory=SectionConfig)
    marginal_returns: SectionConfig = field(default_factory=SectionConfig)
    reallocation: SectionConfig = field(default_factory=SectionConfig)
    flighting: SectionConfig = field(default_factory=SectionConfig)
    deep_dives: SectionConfig = field(default_factory=SectionConfig)
    carryover: SectionConfig = field(default_factory=SectionConfig)
    # Posterior-predictive timeseries fit (observed vs predicted KPI over time
    # with a credible band) — the dedicated PPC "does the model track reality?".
    ppc_timeseries: SectionConfig = field(default_factory=SectionConfig)
    recommended_tests: SectionConfig = field(default_factory=SectionConfig)
    evidence_guide: SectionConfig = field(default_factory=SectionConfig)
    next_steps: SectionConfig = field(default_factory=SectionConfig)

    # Budget-allocation plan (Planner). Default-OFF: AllocationSection is
    # data-gated and renders only when a plan is attached to the bundle.
    allocation: SectionConfig = field(
        default_factory=lambda: SectionConfig(enabled=False)
    )

    # Output settings
    include_plotly_js: bool = True  # Embed Plotly.js (larger file, fully portable)
    plotly_cdn_version: str = "2.27.0"  # Used if not embedding
    include_print_styles: bool = True
    minify_html: bool = False

    # Uncertainty messaging
    uncertainty_callout: bool = True
    methodology_note: bool = True

    # Long-term / brand scenario (issue #106). When set, the LongTermSection adds
    # an ASSUMPTION-driven scenario: total effect = measured short-term ×
    # multiplier (e.g. 2.0 from published brand meta-analyses). ``None`` keeps the
    # section to the estimable within-window carryover split + the honest caveat.
    long_term_multiplier: float | None = None

    # Client presentation options
    show_nav: bool = False  # Sticky side-nav for multi-section reports
    confidential: bool = False  # Add "Confidential" banner to footer
    format_channel_names: bool = False  # Replace underscores → spaces, title-case

    @classmethod
    def minimal(
        cls, title: str = "MMM Report", client: str | None = None
    ) -> ReportConfig:
        """Create minimal report with only essential sections."""
        return cls(
            title=title,
            client=client,
            model_fit=SectionConfig(enabled=False),
            posterior_predictive=SectionConfig(enabled=False),
            saturation=SectionConfig(enabled=False),
            sensitivity=SectionConfig(enabled=False),
            diagnostics=SectionConfig(enabled=False),
        )

    @classmethod
    def full(
        cls, title: str = "Marketing Mix Model Report", client: str | None = None
    ) -> ReportConfig:
        """Create comprehensive report with all sections."""
        return cls(title=title, client=client)

    @classmethod
    def presentation(
        cls, title: str = "MMM Results", client: str | None = None
    ) -> ReportConfig:
        """Create presentation-focused report optimized for stakeholders."""
        return cls(
            title=title,
            client=client,
            large_number_format="short",
            # Technical goodness-of-fit checks are noise for a stakeholder deck;
            # the estimand results (with CI) stay on as the headline numbers.
            posterior_predictive=SectionConfig(enabled=False),
            diagnostics=SectionConfig(enabled=False),
            methodology=SectionConfig(
                enabled=True,
                custom_notes="Contact data science team for technical details.",
            ),
        )

    @classmethod
    def augur_readout(
        cls,
        title: str = "Media Performance Readout",
        client: str | None = None,
        *,
        cmo_insights: dict[str, str] | None = None,
    ) -> ReportConfig:
        """Create the editorial "Media Performance Readout" (Augur) report.

        A narrative, evidence-coded client deliverable: masthead + numbered
        contents nav, a headline with a KPI strip and recommendations, a channel
        scorecard with Scale/Test/Hold/Reduce tiers, ROI-with-uncertainty,
        marginal-vs-average return, saturation, reallocation, per-channel deep
        dives, carryover, a posterior-predictive fit-over-time + checks section,
        and recommended tests / next steps. CMO/planner narrative is filled by
        ``reporting.insights.build_report_insights`` (templated fallback +
        optional LLM enrichment).
        """
        return cls(
            title=title,
            client=client,
            shell="augur",
            color_scheme=ColorScheme.from_palette(ColorPalette.AUGUR),
            large_number_format="short",
            show_nav=True,
            confidential=True,
            format_channel_names=True,
            cmo_insights=dict(cmo_insights or {}),
            # Goodness-of-fit checks are folded into the Augur "does the model
            # hold up?" block; the classic standalone sections stay off.
            sensitivity=SectionConfig(enabled=False),
            diagnostics=SectionConfig(enabled=False),
            methodology=SectionConfig(enabled=False),
            causal_assumptions=SectionConfig(enabled=False),
        )

    def format_currency(self, value: float) -> str:
        """Format value as currency."""
        if self.large_number_format == "short":
            return self._format_short_currency(value)
        return f"{self.currency_symbol}{value:{self.currency_format}}"

    def _format_short_currency(self, value: float) -> str:
        """Format large currency values with K/M/B suffixes."""
        abs_value = abs(value)
        sign = "" if value >= 0 else "-"

        if abs_value >= 1e9:
            return f"{sign}{self.currency_symbol}{abs_value/1e9:.1f}B"
        elif abs_value >= 1e6:
            return f"{sign}{self.currency_symbol}{abs_value/1e6:.1f}M"
        elif abs_value >= 1e3:
            return f"{sign}{self.currency_symbol}{abs_value/1e3:.1f}K"
        else:
            return f"{sign}{self.currency_symbol}{abs_value:.0f}"

    def format_percentage(self, value: float) -> str:
        """Format value as percentage."""
        return f"{value:{self.percentage_format}}"

    def format_decimal(self, value: float) -> str:
        """Format decimal value."""
        return f"{value:{self.decimal_format}}"


@dataclass(frozen=True)
class ChartConfig:
    """Configuration for individual charts."""

    height: int = 400
    width: int | None = None
    margin: dict[str, int] = field(
        default_factory=lambda: {"t": 40, "r": 40, "b": 70, "l": 60}
    )
    show_legend: bool = True
    legend_position: Literal["top", "bottom", "right", "left"] = "top"
    animation: bool = True
    responsive: bool = True

    # Uncertainty visualization
    show_credible_intervals: bool = True
    ci_alpha: float = 0.2  # Transparency for CI bands
    ci_level: float = 0.8  # 80% HDI

    # Axis settings
    x_title: str | None = None
    y_title: str | None = None
    x_tickformat: str | None = None
    y_tickformat: str | None = None

    def to_plotly_layout(self, color_scheme: ColorScheme) -> dict[str, Any]:
        """Convert to Plotly layout dict."""
        layout = {
            "paper_bgcolor": "transparent",
            "plot_bgcolor": "transparent",
            "font": {
                "family": getattr(
                    color_scheme, "font_sans", "Source Sans 3, sans-serif"
                ),
                "color": color_scheme.text,
                "size": 12,
            },
            "margin": self.margin,
            "showlegend": self.show_legend,
            "hovermode": "x unified",
        }

        if self.height:
            layout["height"] = self.height
        if self.width:
            layout["width"] = self.width

        # Configure axes
        layout["xaxis"] = {
            "gridcolor": color_scheme.border,
            "zerolinecolor": color_scheme.border,
        }
        layout["yaxis"] = {
            "gridcolor": color_scheme.border,
            "zerolinecolor": color_scheme.border,
        }

        if self.x_title:
            layout["xaxis"]["title"] = self.x_title
        if self.y_title:
            layout["yaxis"]["title"] = self.y_title
        if self.x_tickformat:
            layout["xaxis"]["tickformat"] = self.x_tickformat
        if self.y_tickformat:
            layout["yaxis"]["tickformat"] = self.y_tickformat

        # Legend position
        if self.show_legend:
            positions = {
                "top": {
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "left",
                    "x": 0,
                },
                "bottom": {
                    "orientation": "h",
                    "yanchor": "top",
                    "y": -0.15,
                    "xanchor": "left",
                    "x": 0,
                },
                "right": {"yanchor": "top", "y": 1, "xanchor": "left", "x": 1.02},
                "left": {"yanchor": "top", "y": 1, "xanchor": "right", "x": -0.02},
            }
            layout["legend"] = positions.get(self.legend_position, positions["top"])

        return layout
