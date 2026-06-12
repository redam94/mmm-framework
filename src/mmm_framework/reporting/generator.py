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
from .design_tokens import TOKENS
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
    CausalAssumptionsSection,
    MethodologySection,
    DiagnosticsSection,
    GeographicSection,
    MediatorSection,
    CannibalizationSection,
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
            (
                "executive_summary",
                ExecutiveSummarySection,
                self.config.executive_summary,
            ),
            ("model_fit", ModelFitSection, self.config.model_fit),
            ("channel_roi", ChannelROISection, self.config.channel_roi),
            ("geographic", GeographicSection, self.config.geographic),
            ("decomposition", DecompositionSection, self.config.decomposition),
            ("mediators", MediatorSection, self.config.mediators),
            ("cannibalization", CannibalizationSection, self.config.cannibalization),
            ("saturation", SaturationSection, self.config.saturation),
            ("sensitivity", SensitivitySection, self.config.sensitivity),
            (
                "causal_assumptions",
                CausalAssumptionsSection,
                self.config.causal_assumptions,
            ),
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

    @staticmethod
    def _fmt_channel(name: str) -> str:
        """Underscore-to-space + title-case for display-friendly channel names."""
        return name.replace("_", " ").title()

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

        # Post-process: replace raw underscore channel names when in client mode
        if self.config.format_channel_names and self.data:
            for ch in self.data.channel_names or []:
                if "_" in ch:
                    sections_html = sections_html.replace(
                        html.escape(ch), html.escape(self._fmt_channel(ch))
                    )

        # Assemble full HTML
        return self._assemble_html(sections_html)

    def _assemble_html(self, sections_html: str) -> str:
        """Assemble complete HTML document."""
        generated_date = self.config.generated_date or datetime.now().strftime("%B %Y")

        # Build header: an editorial masthead — meta row (document kind,
        # confidentiality, dates), serif title, client line, accent rule.
        subtitle = ""
        if self.config.client:
            subtitle = f'<div class="subtitle">{html.escape(self.config.client)}'
            if self.config.subtitle:
                subtitle += f" — {html.escape(self.config.subtitle)}"
            subtitle += "</div>"

        meta_right = f"Generated {generated_date}"
        if self.config.analysis_period:
            meta_right = (
                f"Analysis period {html.escape(self.config.analysis_period)}"
                f'<span class="meta-sep">·</span>{meta_right}'
            )

        conf_tag = ""
        if self.config.confidential:
            conf_tag = '<span class="masthead-tag">Confidential</span>'

        header = f"""
        <header class="report-header">
            <div class="masthead-meta">
                <span class="masthead-kind">Marketing Mix Model Report{conf_tag}</span>
                <span class="masthead-date">{meta_right}</span>
            </div>
            <h1>{html.escape(self.config.title)}</h1>
            {subtitle}
        </header>
        """

        # Build footer (colophon)
        conf_line = ""
        if self.config.confidential:
            client_label = (
                f" — {html.escape(self.config.client)}" if self.config.client else ""
            )
            conf_line = f'<p class="confidential-notice">CONFIDENTIAL{client_label} — not for distribution</p>'
        footer = f"""
        <footer class="report-footer">
            {conf_line}
            <p>Generated {generated_date} · Built with the MMM Framework
               (Bayesian marketing mix modeling — estimates carry the uncertainty
               intervals shown; point values alone are not decisions)</p>
        </footer>
        """

        # Build sticky side-nav from enabled sections
        nav_html = ""
        if self.config.show_nav:
            nav_items = "".join(
                f'<a href="#{s.section_id}" class="nav-item">{html.escape(s.title)}</a>'
                for s in self._sections
                if s.is_enabled
            )
            nav_html = f'<nav class="report-nav" id="reportNav">{nav_items}</nav>'

        # Wrap content + optional nav
        body_class = "has-nav" if self.config.show_nav else ""
        content_wrap = f'<div class="report-body {body_class}">{nav_html}<div class="report-container">{header}{sections_html}{footer}</div></div>'

        css = self._generate_css()
        plotly_script = self._get_plotly_script()

        # Scroll-spy script for nav active state
        scrollspy = ""
        if self.config.show_nav:
            scrollspy = """
<script>
(function() {
  var sections = document.querySelectorAll('section[id]');
  var links = document.querySelectorAll('.report-nav a');
  function onScroll() {
    var scrollY = window.scrollY + 120;
    var active = null;
    sections.forEach(function(s) { if (s.offsetTop <= scrollY) active = s.id; });
    links.forEach(function(a) {
      a.classList.toggle('active', a.getAttribute('href') === '#' + active);
    });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();
</script>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(self.config.title)}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Source+Sans+3:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    {plotly_script}
    <style>
{css}
    </style>
</head>
<body>
    {content_wrap}
    {scrollspy}
</body>
</html>"""

    def _get_plotly_script(self) -> str:
        """Get Plotly.js script tag."""
        if self.config.include_plotly_js:
            return f'<script src="https://cdn.plot.ly/plotly-{self.config.plotly_cdn_version}.min.js"></script>'
        return ""

    def _generate_css(self) -> str:
        """Generate CSS styles from configuration."""
        c = self.config.color_scheme

        return f"""
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
            --shadow-sm: {TOKENS.shadow_sm};
            --shadow-md: {TOKENS.shadow_md};
            --shadow-lg: {TOKENS.shadow_lg};
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

        /* Header — editorial masthead */
        .report-header {{
            border-top: 4px solid var(--color-primary-dark);
            border-bottom: 1px solid var(--color-border);
            padding: 1.5rem 0 2rem;
            margin-bottom: 2.5rem;
        }}

        .masthead-meta {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            flex-wrap: wrap;
            gap: 0.5rem 1.5rem;
            font-family: {self.config.font_family_mono};
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--color-text-muted);
            margin-bottom: 1.75rem;
        }}

        .masthead-meta .meta-sep {{
            margin: 0 0.6em;
            opacity: 0.5;
        }}

        .masthead-tag {{
            display: inline-block;
            margin-left: 0.9em;
            padding: 0.15em 0.7em;
            border: 1px solid var(--color-danger);
            border-radius: 3px;
            color: var(--color-danger);
            font-size: 0.92em;
        }}

        .report-header h1 {{
            font-family: {self.config.font_family_serif};
            font-size: 2.6rem;
            font-weight: 400;
            line-height: 1.15;
            letter-spacing: -0.01em;
            color: var(--color-text);
            margin-bottom: 0.6rem;
            max-width: 18em;
        }}

        .report-header .subtitle {{
            font-size: 1.05rem;
            color: var(--color-text-muted);
        }}

        /* Section styling — numbered, editorial */
        .report-container {{
            counter-reset: report-section;
        }}

        .section {{
            background: var(--color-surface);
            border-radius: 10px;
            padding: 2.25rem 2.5rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--color-border);
            counter-increment: report-section;
        }}

        .section > h2 {{
            font-family: {self.config.font_family_serif};
            font-size: 1.55rem;
            font-weight: 400;
            color: var(--color-text);
            margin-bottom: 1.25rem;
            padding-bottom: 0.85rem;
            border-bottom: 1px solid var(--color-border);
            position: relative;
        }}

        .section > h2::before {{
            content: counter(report-section, decimal-leading-zero);
            font-family: {self.config.font_family_mono};
            font-size: 0.7rem;
            font-weight: 500;
            letter-spacing: 0.14em;
            color: var(--color-primary-dark);
            display: block;
            margin-bottom: 0.4rem;
        }}

        .section > h2::after {{
            content: '';
            position: absolute;
            left: 0;
            bottom: -1px;
            width: 3.5rem;
            height: 2px;
            background: var(--color-primary);
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
            display: flex;
            flex-direction: column;
            background: var(--color-surface);
            border-radius: 8px;
            padding: 1.4rem 1.5rem 1.25rem;
            text-align: left;
            border: 1px solid var(--color-border);
            border-top: 3px solid var(--color-primary);
        }}

        .metric-card .label {{
            order: -1;
            font-family: {self.config.font_family_mono};
            font-size: 0.68rem;
            font-weight: 500;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--color-text-muted);
            margin-bottom: 0.6rem;
        }}

        .metric-card .value {{
            font-size: 2.1rem;
            font-weight: 600;
            line-height: 1.1;
            color: var(--color-text);
            font-family: {self.config.font_family_sans};
            font-variant-numeric: tabular-nums;
        }}

        .metric-card .ci {{
            font-size: 0.78rem;
            color: var(--color-text-muted);
            margin-top: 0.5rem;
            font-family: {self.config.font_family_mono};
            font-variant-numeric: tabular-nums;
        }}

        .metric-card.highlight {{
            border-top-color: var(--color-accent);
            background: var(--color-bg-alt);
        }}

        .metric-card.warning {{
            border-top-color: var(--color-warning);
        }}

        .metric-card.danger {{
            border-top-color: var(--color-danger);
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
            border-radius: 8px;
            padding: 1.25rem 1.5rem;
            margin: 1.5rem 0;
        }}

        .callout h4 {{
            margin-bottom: 0.6rem;
            font-family: {self.config.font_family_mono};
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
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

        /* Tables — editorial: rule-based, tabular numerals */
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
            font-size: 0.92rem;
            font-variant-numeric: tabular-nums;
        }}

        .data-table th, .data-table td {{
            padding: 0.65rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--color-border);
        }}

        .data-table th {{
            background: transparent;
            border-bottom: 2px solid var(--color-text);
            font-family: {self.config.font_family_mono};
            font-weight: 500;
            color: var(--color-text-muted);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}

        .data-table tbody tr:last-child td {{
            border-bottom: 2px solid var(--color-border);
        }}

        .data-table tbody tr:hover {{
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

        /* Footer — colophon */
        .report-footer {{
            text-align: left;
            border-top: 1px solid var(--color-border);
            margin-top: 2.5rem;
            padding: 1.5rem 0 2.5rem;
            color: var(--color-text-muted);
            font-size: 0.82rem;
            line-height: 1.6;
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

        /* Print styles — consultants print these */
        @page {{
            margin: 18mm 16mm;
        }}

        @media print {{
            body {{ background: white; font-size: 11pt; }}
            .report-nav {{ display: none !important; }}
            .report-body {{ display: block !important; }}
            .report-body.has-nav .report-container {{ padding-left: 0; }}
            .report-container {{ max-width: none; padding: 0; margin: 0; }}
            .report-header {{
                border-top-width: 6px;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
            .section {{
                break-inside: avoid;
                box-shadow: none;
                border: none;
                border-top: 1px solid #ddd;
                border-radius: 0;
                padding: 1.5rem 0;
            }}
            .section > h2::before,
            .section > h2::after,
            .metric-card,
            .callout {{
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
            .metric-card {{ break-inside: avoid; }}
            .chart-container {{
                break-inside: avoid;
                page-break-inside: avoid;
            }}
            .data-table tbody tr:hover {{ background: transparent; }}
            .report-footer {{ page-break-inside: avoid; }}
        }}

        /* ── Layout shell ─────────────────────────────────────────────── */
        .report-body {{
            display: contents;
        }}
        .report-body.has-nav {{
            display: flex;
            min-height: 100vh;
            align-items: flex-start;
        }}
        .report-body.has-nav .report-container {{
            flex: 1;
            min-width: 0;
            padding-left: 2rem;
        }}

        /* ── Sticky side nav ─────────────────────────────────────────── */
        .report-nav {{
            position: sticky;
            top: 2rem;
            width: 200px;
            min-width: 200px;
            padding: 1.5rem 0;
            align-self: flex-start;
            background: var(--color-surface);
            border-right: 1px solid var(--color-border);
            font-size: 0.82rem;
        }}
        .report-nav a.nav-item {{
            display: block;
            padding: 0.45rem 1.25rem;
            color: var(--color-text-muted);
            text-decoration: none;
            border-left: 3px solid transparent;
            transition: color 0.15s, border-color 0.15s;
            line-height: 1.35;
        }}
        .report-nav a.nav-item:hover,
        .report-nav a.nav-item.active {{
            color: var(--color-primary-dark);
            border-left-color: var(--color-primary);
            background: var(--color-bg-alt);
        }}

        /* ── Confidential notice ─────────────────────────────────────── */
        .confidential-notice {{
            font-weight: 600;
            color: var(--color-danger);
            letter-spacing: 0.04em;
            font-size: 0.82rem;
            margin-bottom: 0.25rem;
        }}

        @media (max-width: 900px) {{
            .report-nav {{ display: none; }}
            .report-body.has-nav .report-container {{ padding-left: 0; }}
        }}
"""

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

    def with_model(
        self, model: Any, panel: Any | None = None, results: Any | None = None
    ) -> ReportBuilder:
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

    def with_channel_colors(self, channel_colors) -> ReportBuilder:
        """Set per-channel chart colors (a ChannelColors instance)."""
        self._config_kwargs["channel_colors"] = channel_colors
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

    def client_report(self) -> ReportBuilder:
        """Configure a clean, client-ready report.

        Includes: Executive Summary, Channel Performance, Decomposition,
        Saturation & Carryover, and a simplified Methodology note.
        Excludes: Diagnostics (trace plots / prior-posterior) and Sensitivity.

        Also enables sticky side-nav, confidentiality notice, and automatic
        channel-name formatting (underscores → spaces, title-case).
        """
        self._section_configs = {
            "executive_summary": SectionConfig(enabled=True),
            "channel_roi": SectionConfig(enabled=True),
            "decomposition": SectionConfig(enabled=True),
            "saturation": SectionConfig(enabled=True),
            "methodology": SectionConfig(
                enabled=True,
                custom_notes=(
                    "Results are based on a Bayesian statistical model that quantifies "
                    "the contribution of each marketing channel to revenue while accounting "
                    "for seasonality, baseline trends, and external factors. "
                    "All estimates include credible intervals reflecting genuine uncertainty."
                ),
            ),
            "model_fit": SectionConfig(enabled=False),
            "sensitivity": SectionConfig(enabled=False),
            "diagnostics": SectionConfig(enabled=False),
        }
        self._config_kwargs["show_nav"] = True
        self._config_kwargs["confidential"] = True
        self._config_kwargs["format_channel_names"] = True
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
