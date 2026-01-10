"""
Configuration classes for MMM report generation.

Provides immutable, validated configuration for report customization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Any
from enum import Enum


class ColorPalette(Enum):
    """Pre-defined color palettes for reports."""
    
    SAGE = "sage"           # Default: sage green with blue accents
    CORPORATE = "corporate" # Blue/gray professional
    WARM = "warm"           # Warm earth tones
    MONOCHROME = "monochrome"  # Grayscale


@dataclass(frozen=True)
class ColorScheme:
    """Color scheme for report styling."""
    
    primary: str = "#8fa86a"
    primary_dark: str = "#6d8a4a"
    accent: str = "#6a8fa8"
    accent_dark: str = "#4a6d8a"
    warning: str = "#d4a86a"
    danger: str = "#c97067"
    success: str = "#6abf8a"
    text: str = "#2d3a2d"
    text_muted: str = "#5a6b5a"
    background: str = "#fafbf9"
    background_alt: str = "#f0f4ed"
    surface: str = "#ffffff"
    border: str = "#d4ddd4"
    
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
        }
        return palettes.get(palette, cls())


@dataclass(frozen=True)
class ChannelColors:
    """Color mapping for media channels."""
    
    colors: dict[str, str] = field(default_factory=lambda: {
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
    })
    
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
    font_family_serif: str = "'DM Serif Display', serif"
    font_family_sans: str = "'Inter', sans-serif"
    font_family_mono: str = "'JetBrains Mono', monospace"
    
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
    channel_roi: SectionConfig = field(default_factory=SectionConfig)
    decomposition: SectionConfig = field(default_factory=SectionConfig)
    saturation: SectionConfig = field(default_factory=SectionConfig)
    sensitivity: SectionConfig = field(default_factory=SectionConfig)
    methodology: SectionConfig = field(default_factory=SectionConfig)
    diagnostics: SectionConfig = field(default_factory=SectionConfig)
    
    # Output settings
    include_plotly_js: bool = True  # Embed Plotly.js (larger file, fully portable)
    plotly_cdn_version: str = "2.27.0"  # Used if not embedding
    include_print_styles: bool = True
    minify_html: bool = False
    
    # Uncertainty messaging
    uncertainty_callout: bool = True
    methodology_note: bool = True
    
    @classmethod
    def minimal(cls, title: str = "MMM Report", client: str | None = None) -> ReportConfig:
        """Create minimal report with only essential sections."""
        return cls(
            title=title,
            client=client,
            model_fit=SectionConfig(enabled=False),
            saturation=SectionConfig(enabled=False),
            sensitivity=SectionConfig(enabled=False),
            diagnostics=SectionConfig(enabled=False),
        )
    
    @classmethod
    def full(cls, title: str = "Marketing Mix Model Report", client: str | None = None) -> ReportConfig:
        """Create comprehensive report with all sections."""
        return cls(title=title, client=client)
    
    @classmethod
    def presentation(cls, title: str = "MMM Results", client: str | None = None) -> ReportConfig:
        """Create presentation-focused report optimized for stakeholders."""
        return cls(
            title=title,
            client=client,
            large_number_format="short",
            diagnostics=SectionConfig(enabled=False),
            methodology=SectionConfig(
                enabled=True,
                custom_notes="Contact data science team for technical details."
            ),
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
    margin: dict[str, int] = field(default_factory=lambda: {
        "t": 40, "r": 40, "b": 70, "l": 60
    })
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
                "family": "Inter, sans-serif",
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
                "top": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
                "bottom": {"orientation": "h", "yanchor": "top", "y": -0.15, "xanchor": "left", "x": 0},
                "right": {"yanchor": "top", "y": 1, "xanchor": "left", "x": 1.02},
                "left": {"yanchor": "top", "y": 1, "xanchor": "right", "x": -0.02},
            }
            layout["legend"] = positions.get(self.legend_position, positions["top"])
        
        return layout