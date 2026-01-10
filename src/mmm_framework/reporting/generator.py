"""
Main report generator for MMM reports.

Assembles sections, applies styling, and produces portable HTML output.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import html

from .config import ReportConfig, SectionConfig, ColorScheme
from .data_extractors import (
    MMMDataBundle,
    DataExtractor,
    BayesianMMMExtractor,
    ExtendedMMMExtractor,
    PyMCMarketingExtractor,
    create_extractor,
)
from .sections import (
    Section,
    ExecutiveSummarySection,
    ModelFitSection,
    ChannelROISection,
    DecompositionSection,
    SaturationSection,
    SensitivitySection,
    MethodologySection,
    DiagnosticsSection,
    SECTION_REGISTRY,
)


class MMMReportGenerator:
    """
    Generate portable HTML reports from Bayesian MMM results.
    
    Parameters
    ----------
    model : Any, optional
        Fitted MMM model (BayesianMMM, NestedMMM, MultivariateMMM, etc.)
    data : MMMDataBundle, optional
        Pre-extracted data bundle (alternative to model)
    config : ReportConfig, optional
        Report configuration
    panel : Any, optional
        Panel dataset used for fitting
    results : Any, optional
        Model fit results
    sensitivity : dict, optional
        Sensitivity analysis results
    
    Examples
    --------
    >>> from mmm_framework import BayesianMMM
    >>> from mmm_reporting import MMMReportGenerator, ReportConfig
    >>> 
    >>> # Fit model
    >>> mmm = BayesianMMM(panel, model_config, trend_config)
    >>> results = mmm.fit()
    >>> 
    >>> # Generate report
    >>> report = MMMReportGenerator(
    ...     model=mmm,
    ...     config=ReportConfig(
    ...         title="Q4 2025 Marketing Analysis",
    ...         client="Acme Corp",
    ...         analysis_period="Jan 2023 - Dec 2025",
    ...     )
    ... )
    >>> report.to_html("mmm_report.html")
    """
    
    def __init__(
        self,
        model: Any | None = None,
        data: MMMDataBundle | None = None,
        config: ReportConfig | None = None,
        panel: Any | None = None,
        results: Any | None = None,
        sensitivity: dict | None = None,
    ):
        self.config = config or ReportConfig()
        
        # Extract data from model or use provided data
        if data is not None:
            self.data = data
        elif model is not None:
            extractor = create_extractor(model, panel=panel, results=results)
            self.data = extractor.extract()
        else:
            self.data = MMMDataBundle()
        
        # Add sensitivity results if provided
        if sensitivity is not None:
            self.data.sensitivity_results = sensitivity
        
        # Initialize sections
        self._sections: list[Section] = []
        self._initialize_sections()
    
    def _initialize_sections(self):
        """Initialize report sections based on configuration."""
        section_configs = [
            ("executive_summary", ExecutiveSummarySection, self.config.executive_summary),
            ("model_fit", ModelFitSection, self.config.model_fit),
            ("channel_roi", ChannelROISection, self.config.channel_roi),
            ("decomposition", DecompositionSection, self.config.decomposition),
            ("saturation", SaturationSection, self.config.saturation),
            ("sensitivity", SensitivitySection, self.config.sensitivity),
            ("methodology", MethodologySection, self.config.methodology),
            ("diagnostics", DiagnosticsSection, self.config.diagnostics),
        ]
        
        for name, section_class, section_config in section_configs:
            section = section_class(
                data=self.data,
                config=self.config,
                section_config=section_config,
            )
            self._sections.append(section)
    
    def add_section(
        self,
        section_type: str,
        section_config: SectionConfig | None = None,
        position: int | None = None,
    ) -> MMMReportGenerator:
        """
        Add a section to the report.
        
        Parameters
        ----------
        section_type : str
            Type of section from SECTION_REGISTRY
        section_config : SectionConfig, optional
            Configuration for the section
        position : int, optional
            Position to insert (None = append)
            
        Returns
        -------
        MMMReportGenerator
            Self for chaining
        """
        if section_type not in SECTION_REGISTRY:
            raise ValueError(f"Unknown section type: {section_type}")
        
        section_class = SECTION_REGISTRY[section_type]
        section = section_class(
            data=self.data,
            config=self.config,
            section_config=section_config or SectionConfig(),
        )
        
        if position is not None:
            self._sections.insert(position, section)
        else:
            self._sections.append(section)
        
        return self
    
    def remove_section(self, section_id: str) -> MMMReportGenerator:
        """
        Remove a section by ID.
        
        Parameters
        ----------
        section_id : str
            Section ID to remove
            
        Returns
        -------
        MMMReportGenerator
            Self for chaining
        """
        self._sections = [s for s in self._sections if s.section_id != section_id]
        return self
    
    def render(self) -> str:
        """
        Render complete HTML report.
        
        Returns
        -------
        str
            Complete HTML document
        """
        # Render all enabled sections
        sections_html = "\n".join(
            section.render() for section in self._sections if section.is_enabled
        )
        
        # Assemble full HTML
        return self._assemble_html(sections_html)
    
    def _assemble_html(self, sections_html: str) -> str:
        """Assemble complete HTML document."""
        generated_date = self.config.generated_date or datetime.now().strftime("%B %Y")
        
        # Build header
        subtitle = ""
        if self.config.client:
            subtitle = f'<div class="subtitle">{html.escape(self.config.client)}'
            if self.config.subtitle:
                subtitle += f" — {html.escape(self.config.subtitle)}"
            subtitle += "</div>"
        
        period = ""
        if self.config.analysis_period:
            period = f'<div class="date">Analysis Period: {html.escape(self.config.analysis_period)} | Generated: {generated_date}</div>'
        
        header = f'''
        <header class="report-header">
            <h1>{html.escape(self.config.title)}</h1>
            {subtitle}
            {period}
        </header>
        '''
        
        # Build footer
        footer = f'''
        <footer class="report-footer">
            <p>Generated by MMM Framework | <a href="https://github.com/redam94/mmm-framework" style="color: var(--color-primary);">Documentation</a></p>
            <p style="margin-top: 0.5rem;">Report generated: {generated_date}</p>
        </footer>
        '''
        
        # Get CSS
        css = self._generate_css()
        
        # Get Plotly script
        plotly_script = self._get_plotly_script()
        
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(self.config.title)}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    {plotly_script}
    <style>
{css}
    </style>
</head>
<body>
    <div class="report-container">
        {header}
        {sections_html}
        {footer}
    </div>
</body>
</html>'''
    
    def _get_plotly_script(self) -> str:
        """Get Plotly.js script tag."""
        if self.config.include_plotly_js:
            return f'<script src="https://cdn.plot.ly/plotly-{self.config.plotly_cdn_version}.min.js"></script>'
        return ""
    
    def _generate_css(self) -> str:
        """Generate CSS styles from configuration."""
        c = self.config.color_scheme
        
        return f'''
        :root {{
            --color-primary: {c.primary};
            --color-primary-dark: {c.primary_dark};
            --color-accent: {c.accent};
            --color-accent-dark: {c.accent_dark};
            --color-warning: {c.warning};
            --color-danger: {c.danger};
            --color-success: {c.success};
            --color-text: {c.text};
            --color-text-muted: {c.text_muted};
            --color-bg: {c.background};
            --color-bg-alt: {c.background_alt};
            --color-surface: {c.surface};
            --color-border: {c.border};
            --shadow-sm: 0 1px 3px rgba(45, 58, 45, 0.08);
            --shadow-md: 0 4px 12px rgba(45, 58, 45, 0.1);
            --shadow-lg: 0 8px 24px rgba(45, 58, 45, 0.12);
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{ 
            font-family: {self.config.font_family_sans}; 
            background: var(--color-bg); 
            color: var(--color-text); 
            line-height: 1.7; 
        }}

        .report-container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}

        /* Header */
        .report-header {{
            text-align: center;
            padding: 3rem 2rem;
            background: linear-gradient(135deg, var(--color-primary-dark) 0%, var(--color-accent-dark) 100%);
            color: white;
            border-radius: 16px;
            margin-bottom: 2rem;
        }}

        .report-header h1 {{
            font-family: {self.config.font_family_serif};
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}

        .report-header .subtitle {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}

        .report-header .date {{
            margin-top: 1rem;
            font-size: 0.9rem;
            opacity: 0.8;
        }}

        /* Section styling */
        .section {{
            background: var(--color-surface);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--color-border);
        }}

        .section h2 {{
            font-family: {self.config.font_family_serif};
            font-size: 1.6rem;
            color: var(--color-text);
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 2px solid var(--color-primary);
        }}

        .section h3 {{
            font-size: 1.2rem;
            color: var(--color-text);
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
        }}

        .section p {{
            margin-bottom: 1rem;
            color: var(--color-text);
        }}

        .section-subtitle {{
            color: var(--color-text-muted);
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
        }}

        /* Key metrics grid */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin: 1.5rem 0;
        }}

        .metric-card {{
            background: var(--color-bg-alt);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid var(--color-border);
        }}

        .metric-card .value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--color-primary-dark);
            font-family: {self.config.font_family_mono};
        }}

        .metric-card .label {{
            font-size: 0.85rem;
            color: var(--color-text-muted);
            margin-top: 0.25rem;
        }}

        .metric-card .ci {{
            font-size: 0.8rem;
            color: var(--color-text-muted);
            margin-top: 0.5rem;
            font-family: {self.config.font_family_mono};
        }}

        .metric-card.highlight {{
            border-left: 4px solid var(--color-primary);
        }}

        .metric-card.warning {{
            border-left: 4px solid var(--color-warning);
        }}

        .metric-card.danger {{
            border-left: 4px solid var(--color-danger);
        }}

        /* Charts */
        .chart-container {{
            width: 100%;
            min-height: 300px;
            margin: 1.5rem 0;
        }}

        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
            margin: 1.5rem 0;
        }}

        .chart-box {{
            background: var(--color-bg-alt);
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid var(--color-border);
        }}

        /* Callout boxes */
        .callout {{
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1.5rem 0;
        }}

        .callout h4 {{
            margin-bottom: 0.75rem;
            font-size: 1rem;
        }}

        .callout p {{
            margin-bottom: 0;
            font-size: 0.95rem;
        }}

        .callout.insight {{
            background: rgba(106, 143, 168, 0.1);
            border: 1px solid rgba(106, 143, 168, 0.3);
            border-left: 4px solid var(--color-accent);
        }}

        .callout.insight h4 {{ color: var(--color-accent-dark); }}

        .callout.warning {{
            background: rgba(212, 168, 106, 0.1);
            border: 1px solid rgba(212, 168, 106, 0.3);
            border-left: 4px solid var(--color-warning);
        }}

        .callout.warning h4 {{ color: #b8860b; }}

        .callout.success {{
            background: rgba(106, 191, 138, 0.1);
            border: 1px solid rgba(106, 191, 138, 0.3);
            border-left: 4px solid var(--color-success);
        }}

        .callout.success h4 {{ color: #3d8b5a; }}

        .callout.danger {{
            background: rgba(201, 112, 103, 0.08);
            border: 1px solid rgba(201, 112, 103, 0.3);
            border-left: 4px solid var(--color-danger);
        }}

        .callout.danger h4 {{ color: var(--color-danger); }}

        /* Channel pills */
        .channel-legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            margin: 1rem 0;
            justify-content: center;
        }}

        .channel-pill {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: var(--color-surface);
            border: 1px solid var(--color-border);
            border-radius: 20px;
            font-size: 0.85rem;
        }}

        .channel-pill .dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}

        /* Tables */
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
            font-size: 0.95rem;
        }}

        .data-table th, .data-table td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--color-border);
        }}

        .data-table th {{
            background: var(--color-bg-alt);
            font-weight: 600;
            color: var(--color-text-muted);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}

        .data-table tr:hover {{
            background: var(--color-bg-alt);
        }}

        .data-table .mono {{
            font-family: {self.config.font_family_mono};
        }}

        .data-table .positive {{ color: var(--color-success); }}
        .data-table .negative {{ color: var(--color-danger); }}
        .data-table .uncertain {{ color: var(--color-warning); }}

        /* Methodology note */
        .methodology-note {{
            background: var(--color-bg-alt);
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 1.5rem;
            font-size: 0.9rem;
            color: var(--color-text-muted);
        }}

        .methodology-note h4 {{
            color: var(--color-text);
            margin-bottom: 0.75rem;
        }}

        .methodology-note ul {{
            margin: 0.5rem 0 0 1.5rem;
        }}

        /* Footer */
        .report-footer {{
            text-align: center;
            padding: 2rem;
            color: var(--color-text-muted);
            font-size: 0.85rem;
        }}

        .report-footer a {{
            color: var(--color-primary);
            text-decoration: none;
        }}

        .report-footer a:hover {{
            text-decoration: underline;
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            .report-container {{ padding: 1rem; }}
            .chart-grid {{ grid-template-columns: 1fr; }}
            .metrics-grid {{ grid-template-columns: 1fr 1fr; }}
            .report-header h1 {{ font-size: 1.8rem; }}
        }}

        /* Print styles */
        @media print {{
            body {{ background: white; }}
            .report-container {{ max-width: none; padding: 0; }}
            .section {{ 
                break-inside: avoid; 
                box-shadow: none;
                border: 1px solid #ddd;
            }}
            .chart-container {{ 
                break-inside: avoid;
                page-break-inside: avoid;
            }}
        }}
'''
    
    def to_html(self, filepath: str | Path) -> Path:
        """
        Save report to HTML file.
        
        Parameters
        ----------
        filepath : str or Path
            Output file path
            
        Returns
        -------
        Path
            Path to saved file
        """
        filepath = Path(filepath)
        html_content = self.render()
        
        filepath.write_text(html_content, encoding="utf-8")
        
        return filepath
    
    def to_string(self) -> str:
        """
        Get report as HTML string.
        
        Returns
        -------
        str
            HTML document string
        """
        return self.render()
    
    def _repr_html_(self) -> str:
        """Jupyter notebook display."""
        return self.render()


class ReportBuilder:
    """
    Fluent builder for creating customized reports.
    
    Examples
    --------
    >>> report = (
    ...     ReportBuilder()
    ...     .with_model(mmm)
    ...     .with_title("Q4 Analysis")
    ...     .with_client("Acme Corp")
    ...     .enable_all_sections()
    ...     .disable_section("diagnostics")
    ...     .with_credible_interval(0.9)
    ...     .build()
    ... )
    """
    
    def __init__(self):
        self._model: Any | None = None
        self._data: MMMDataBundle | None = None
        self._panel: Any | None = None
        self._results: Any | None = None
        self._sensitivity: dict | None = None
        self._config_kwargs: dict = {}
        self._section_configs: dict[str, SectionConfig] = {}
    
    def with_model(self, model: Any, panel: Any | None = None, results: Any | None = None) -> ReportBuilder:
        """Set the MMM model."""
        self._model = model
        self._panel = panel
        self._results = results
        return self
    
    def with_data(self, data: MMMDataBundle) -> ReportBuilder:
        """Set pre-extracted data bundle."""
        self._data = data
        return self
    
    def with_sensitivity(self, results: dict) -> ReportBuilder:
        """Add sensitivity analysis results."""
        self._sensitivity = results
        return self
    
    def with_title(self, title: str) -> ReportBuilder:
        """Set report title."""
        self._config_kwargs["title"] = title
        return self
    
    def with_client(self, client: str) -> ReportBuilder:
        """Set client name."""
        self._config_kwargs["client"] = client
        return self
    
    def with_subtitle(self, subtitle: str) -> ReportBuilder:
        """Set subtitle."""
        self._config_kwargs["subtitle"] = subtitle
        return self
    
    def with_analysis_period(self, period: str) -> ReportBuilder:
        """Set analysis period string."""
        self._config_kwargs["analysis_period"] = period
        return self
    
    def with_color_scheme(self, scheme: ColorScheme) -> ReportBuilder:
        """Set color scheme."""
        self._config_kwargs["color_scheme"] = scheme
        return self
    
    def with_credible_interval(self, prob: float) -> ReportBuilder:
        """Set default credible interval (e.g., 0.8 for 80% CI)."""
        self._config_kwargs["default_credible_interval"] = prob
        return self
    
    def enable_section(self, section_name: str, **kwargs) -> ReportBuilder:
        """Enable a section with optional configuration."""
        config = SectionConfig(enabled=True, **kwargs)
        self._section_configs[section_name] = config
        return self
    
    def disable_section(self, section_name: str) -> ReportBuilder:
        """Disable a section."""
        self._section_configs[section_name] = SectionConfig(enabled=False)
        return self
    
    def enable_all_sections(self) -> ReportBuilder:
        """Enable all sections."""
        for name in SECTION_REGISTRY.keys():
            if name not in self._section_configs:
                self._section_configs[name] = SectionConfig(enabled=True)
        return self
    
    def minimal_report(self) -> ReportBuilder:
        """Configure for minimal report (executive summary + ROI only)."""
        self._section_configs = {
            "executive_summary": SectionConfig(enabled=True),
            "channel_roi": SectionConfig(enabled=True),
            "methodology": SectionConfig(enabled=True),
            "model_fit": SectionConfig(enabled=False),
            "decomposition": SectionConfig(enabled=False),
            "saturation": SectionConfig(enabled=False),
            "sensitivity": SectionConfig(enabled=False),
            "diagnostics": SectionConfig(enabled=False),
        }
        return self
    
    def build(self) -> MMMReportGenerator:
        """Build the report generator."""
        # Apply section configs
        for name, config in self._section_configs.items():
            self._config_kwargs[name] = config
        
        # Create config
        report_config = ReportConfig(**self._config_kwargs)
        
        # Create generator
        return MMMReportGenerator(
            model=self._model,
            data=self._data,
            config=report_config,
            panel=self._panel,
            results=self._results,
            sensitivity=self._sensitivity,
        )