"""
Main report generator for MMM reports.

Assembles sections, applies styling, and produces portable HTML output.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import html

from .config import ReportConfig, SectionConfig, ColorScheme, ColorPalette
from .design_tokens import TOKENS
from .data_extractors import (
    MMMDataBundle,
    create_extractor,
)
from .sections import (
    Section,
    AllocationSection,
    ExecutiveSummarySection,
    FactorAnalysisSection,
    ModelFitSection,
    PosteriorPredictiveSection,
    EstimandsSection,
    ChannelROISection,
    DecompositionSection,
    SaturationSection,
    SensitivitySection,
    TriangulationSection,
    SpecCurveSection,
    CausalAssumptionsSection,
    MethodologySection,
    DiagnosticsSection,
    GeographicSection,
    MediatorSection,
    CannibalizationSection,
    SECTION_REGISTRY,
)
from .augur_sections import AUGUR_SECTIONS
from .augur_theme import augur_css, MASTHEAD_LOGO_SVG, AUGUR_FONTS_LINK
from .evidence import EVIDENCE_CHIP_CSS


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
        llm: Any | None = None,
        allocation: dict | None = None,
        triangulation: dict | None = None,
        spec_curve: dict | None = None,
    ):
        self.config = config or ReportConfig()
        self._llm = llm

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

        # Triangulation panel — MMM × experiment × platform (issue #104).
        # Data-gated: the TriangulationSection renders only when attached.
        if triangulation is not None:
            self.data.triangulation = triangulation
        # Spec-curve / model-averaging robustness (issue #103). Attach the
        # SpecCurveResult payload; the SpecCurveSection is data-gated so it only
        # appears when a sweep was actually run.
        if spec_curve is not None:
            self.data.spec_curve = spec_curve

        # Budget-allocation plan (a default reallocation, or a saved Planner plan).
        # When attached, expose it on the bundle and turn the allocation section ON
        # — it is default-off and data-gated, so it never appears without a plan.
        # This drives BOTH the classic AllocationSection and the Augur-native
        # AugurAllocationSection (same bundle field, both gate on config.allocation).
        if isinstance(allocation, dict) and allocation.get("allocation"):
            self.data.allocation_results = allocation
            if not self.config.allocation.enabled:
                import dataclasses

                self.config = dataclasses.replace(
                    self.config, allocation=SectionConfig(enabled=True)
                )

        # Augur shell: fill the CMO/planner narrative (templated fallback +
        # optional LLM enrichment) when the caller hasn't supplied one. Build a
        # fresh ReportConfig via dataclasses.replace rather than mutating the
        # frozen config in place — so a config reused across reports never leaks
        # one report's insights into another. Never blocks.
        if self.config.shell == "augur" and not self.config.cmo_insights:
            try:
                import dataclasses

                from .insights import build_report_insights

                self.config = dataclasses.replace(
                    self.config,
                    cmo_insights=build_report_insights(self.data, llm=self._llm),
                )
            except Exception:
                pass

        # Initialize sections
        self._sections: list[Section] = []
        self._initialize_sections()

    def _initialize_sections(self):
        """Initialize report sections based on configuration."""
        # The Augur "Media Performance Readout" uses a dedicated, ordered section
        # set (narrative + evidence-coded) instead of the classic cards. Each
        # Augur section gates on a ReportConfig SectionConfig and no-ops on
        # missing bundle data, so a partial model still renders.
        if self.config.shell == "augur":
            self._initialize_augur_sections()
            return
        # MMM-specific sections (channels/ROI/decomposition/saturation/geo/
        # mediators/cannibalization) gate OFF for a non-MMM family (e.g. a CFA);
        # the factor-analysis section gates ON only for non-MMM. Detected from the
        # bundle's model_kind (set by the extractor).
        is_mmm = getattr(self.data, "model_kind", "mmm") == "mmm"
        _off = SectionConfig(enabled=False)

        # The factor-analysis section is DATA-driven, not MMM-kind-driven: it turns
        # on whenever the bundle carries latent-structure data (a pure CFA/LCA via
        # the FactorAnalysisExtractor, OR a hybrid MMM that also estimates a latent
        # factor — both fill these fields). FactorAnalysisSection.render() no-ops on
        # empty data, so a plain MMM (empty fields) leaves it off.
        has_latent = bool(getattr(self.data, "factor_loadings", None)) or bool(
            getattr(self.data, "cfa_fit_indices", None)
        )

        def _mmm(cfg: SectionConfig) -> SectionConfig:
            return cfg if is_mmm else _off

        def _latent(cfg: SectionConfig) -> SectionConfig:
            return cfg if has_latent else _off

        section_configs = [
            (
                "executive_summary",
                ExecutiveSummarySection,
                self.config.executive_summary,
            ),
            (
                "factor_analysis",
                FactorAnalysisSection,
                _latent(self.config.factor_analysis),
            ),
            ("model_fit", ModelFitSection, self.config.model_fit),
            (
                "posterior_predictive",
                PosteriorPredictiveSection,
                _mmm(self.config.posterior_predictive),
            ),
            ("channel_roi", ChannelROISection, _mmm(self.config.channel_roi)),
            ("allocation", AllocationSection, _mmm(self.config.allocation)),
            ("estimands", EstimandsSection, _mmm(self.config.estimands)),
            ("geographic", GeographicSection, _mmm(self.config.geographic)),
            ("decomposition", DecompositionSection, _mmm(self.config.decomposition)),
            ("mediators", MediatorSection, _mmm(self.config.mediators)),
            (
                "cannibalization",
                CannibalizationSection,
                _mmm(self.config.cannibalization),
            ),
            ("saturation", SaturationSection, _mmm(self.config.saturation)),
            ("sensitivity", SensitivitySection, _mmm(self.config.sensitivity)),
            ("triangulation", TriangulationSection, _mmm(self.config.triangulation)),
            ("spec_curve", SpecCurveSection, _mmm(self.config.spec_curve)),
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

    def _initialize_augur_sections(self):
        """Build the ordered Augur "Media Performance Readout" section set."""
        default = SectionConfig()
        for _section_id, section_class, cfg_attr in AUGUR_SECTIONS:
            section_config = getattr(self.config, cfg_attr, None) or default
            self._sections.append(
                section_class(
                    data=self.data,
                    config=self.config,
                    section_config=section_config,
                )
            )

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
        if self.config.shell == "augur":
            return self._assemble_html_augur(sections_html)
        generated_date = self.config.generated_date or datetime.now().strftime("%B %Y")

        # Build header: gradient hero — serif title, client/subtitle line,
        # analysis-period and generation dates, confidentiality badge.
        subtitle = ""
        if self.config.client:
            subtitle = f'<div class="subtitle">{html.escape(self.config.client)}'
            if self.config.subtitle:
                subtitle += f" — {html.escape(self.config.subtitle)}"
            subtitle += "</div>"

        date_line = f"Generated: {generated_date}"
        if self.config.analysis_period:
            date_line = (
                f"Analysis Period: {html.escape(self.config.analysis_period)}"
                f'<span class="meta-sep">|</span>{date_line}'
            )

        conf_tag = ""
        if self.config.confidential:
            conf_tag = '<span class="masthead-tag">Confidential</span>'

        header = f"""
        <header class="report-header">
            <h1>{html.escape(self.config.title)}</h1>
            {subtitle}
            <div class="date">{date_line}{conf_tag}</div>
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
    <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    {plotly_script}
    <style>
{css}
    </style>
</head>
<body>
    {content_wrap}
    {scrollspy}
    <script>
    function showTab(tabId) {{
        var pane = document.getElementById(tabId);
        if (!pane) return;
        var container = pane.closest('.tab-container') || document;
        container.querySelectorAll('.tab-btn').forEach(function(b) {{ b.classList.remove('active'); }});
        container.querySelectorAll('.tab-content').forEach(function(c) {{ c.classList.remove('active'); }});
        var btn = (typeof event !== 'undefined' && event) ? event.target : null;
        if (btn) btn.classList.add('active');
        pane.classList.add('active');
        if (window.Plotly) {{
            setTimeout(function() {{
                pane.querySelectorAll('.js-plotly-plot').forEach(function(p) {{
                    Plotly.Plots.resize(p);
                }});
            }}, 10);
        }}
    }}
    </script>
</body>
</html>"""

    def _assemble_html_augur(self, sections_html: str) -> str:
        """Assemble the Augur "Media Performance Readout" document.

        Editorial masthead + numbered sticky contents nav + cream/ink shell.
        Preserves the three injection vectors of the default shell
        (``{css}`` / ``{plotly_script}`` / ``{sections_html}``).
        """
        generated_date = self.config.generated_date or datetime.now().strftime("%B %Y")
        enabled = [s for s in self._sections if s.is_enabled]

        # Masthead
        eyebrow = "Marketing mix modeling · Client &amp; planning readout"
        meta_bits = []
        if self.config.client:
            meta_bits.append(f"Prepared for {html.escape(self.config.client)}")
        if self.config.subtitle:
            meta_bits.append(html.escape(self.config.subtitle))
        if self.config.analysis_period:
            meta_bits.append(html.escape(self.config.analysis_period))
        meta_bits.append(generated_date)
        meta_line = '<span class="sep">·</span>'.join(meta_bits)
        conf = (
            '<div class="conf">Confidential</div>' if self.config.confidential else ""
        )
        header = f"""
        <header class="report-header">
            <div class="masthead-logo">{MASTHEAD_LOGO_SVG}</div>
            <div class="masthead-text">
                <div class="masthead-eyebrow">{eyebrow}</div>
                <h1>{html.escape(self.config.title)}</h1>
                <div class="meta">{meta_line}</div>
                {conf}
            </div>
        </header>
        """

        # Numbered sticky contents nav
        nav_items = "".join(
            f'<a class="nav-item" href="#{s.section_id}">'
            f'<span class="nav-num">{i:02d}</span>{html.escape(s.title)}</a>'
            for i, s in enumerate(enabled, start=1)
        )
        nav_html = (
            '<nav class="report-nav" id="reportNav" aria-label="Report contents">'
            '<div class="nav-head">Contents</div>'
            f"{nav_items}</nav>"
        )

        # Footer / colophon
        conf_line = ""
        if self.config.confidential:
            client_label = (
                f" — {html.escape(self.config.client)}" if self.config.client else ""
            )
            conf_line = (
                f'<p class="confidential-notice">CONFIDENTIAL{client_label} — '
                "not for distribution</p>"
            )
        footer = f"""
        <footer class="report-footer">
            {conf_line}
            <p>ROI, credible intervals, saturation curves and carryover rates are
               output from a Bayesian marketing mix model; figures carry
               {int(self.config.default_credible_interval * 100)}% credible
               intervals unless noted. Weekly flighting patterns are illustrative —
               modelled to be consistent with the channel-level estimates — and are
               marked as such where they appear. Point estimates alone are not
               decisions. Generated {generated_date} with the MMM Framework.</p>
        </footer>
        """

        content_wrap = (
            f'<div class="report-body">{nav_html}'
            f'<div class="report-container">{header}{sections_html}{footer}</div></div>'
        )

        css = self._generate_css()
        plotly_script = self._get_plotly_script()
        scrollspy = """
<script>
(function() {
  var items = Array.prototype.slice.call(document.querySelectorAll('.report-nav .nav-item'));
  if (!items.length) return;
  var sections = items.map(function(a){ return document.getElementById(a.getAttribute('href').slice(1)); }).filter(Boolean);
  function setActive(id){ items.forEach(function(a){ a.classList.toggle('active', a.getAttribute('href').slice(1)===id); }); }
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function(entries){
      entries.forEach(function(e){ if (e.isIntersecting) setActive(e.target.id); });
    }, { rootMargin:'-18% 0px -72% 0px', threshold:0 });
    sections.forEach(function(s){ io.observe(s); });
  }
  if (sections[0]) setActive(sections[0].id);
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
    {AUGUR_FONTS_LINK}
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
        if self.config.shell == "augur":
            return augur_css(self.config.color_scheme)

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

        /* Header — gradient hero */
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
            font-weight: 400;
            line-height: 1.2;
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

        .report-header .meta-sep {{
            margin: 0 0.6em;
            opacity: 0.6;
        }}

        .masthead-tag {{
            display: inline-block;
            margin-left: 0.9em;
            padding: 0.15em 0.7em;
            border: 1px solid rgba(255, 255, 255, 0.65);
            border-radius: 4px;
            color: white;
            font-size: 0.85em;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }}

        /* Section styling — modern cards */
        .section {{
            background: var(--color-surface);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--color-border);
        }}

        .section > h2 {{
            font-family: {self.config.font_family_serif};
            font-size: 1.6rem;
            font-weight: 400;
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
            line-height: 1.15;
            color: var(--color-primary-dark);
            font-family: {self.config.font_family_mono};
            font-variant-numeric: tabular-nums;
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
            font-variant-numeric: tabular-nums;
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

        .chart-container-sm {{
            min-height: 250px;
        }}

        .chart-container-lg {{
            min-height: 450px;
        }}

        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 2rem;
            margin: 1.5rem 0;
        }}

        .chart-box {{
            background: var(--color-bg-alt);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid var(--color-border);
        }}

        .chart-box h4 {{
            font-size: 1rem;
            color: var(--color-text);
            margin-bottom: 1rem;
            text-align: center;
        }}

        /* Interactive controls */
        .control-panel {{
            background: var(--color-bg-alt);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            border: 1px solid var(--color-border);
        }}

        .control-panel h4 {{
            font-size: 0.9rem;
            color: var(--color-text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 1rem;
        }}

        .control-row {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }}

        .control-row label {{
            font-weight: 500;
            min-width: 150px;
            font-size: 0.95rem;
        }}

        .control-row select {{
            padding: 0.5rem;
            border-radius: 4px;
            border: 1px solid var(--color-border);
            min-width: 200px;
            background: var(--color-surface);
            color: var(--color-text);
        }}

        /* Tabs */
        .tab-container {{
            margin: 1.5rem 0;
        }}

        .tab-buttons {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
            border-bottom: 2px solid var(--color-border);
            padding-bottom: 0.5rem;
        }}

        .tab-btn {{
            padding: 0.5rem 1.25rem;
            border: none;
            background: transparent;
            color: var(--color-text-muted);
            cursor: pointer;
            font-family: inherit;
            font-size: 0.95rem;
            font-weight: 500;
            border-radius: 6px 6px 0 0;
            transition: all 0.2s;
        }}

        .tab-btn:hover {{
            color: var(--color-primary);
            background: var(--color-bg-alt);
        }}

        .tab-btn.active {{
            color: var(--color-primary-dark);
            background: var(--color-bg-alt);
            border-bottom: 2px solid var(--color-primary);
            margin-bottom: -2px;
        }}

        .tab-content {{
            display: none;
        }}

        .tab-content.active {{
            display: block;
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
            font-variant-numeric: tabular-nums;
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

        .data-table tbody tr:hover {{
            background: var(--color-bg-alt);
        }}

        .data-table .mono {{
            font-family: {self.config.font_family_mono};
        }}

        .data-table .muted {{ color: var(--color-text-muted); }}
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
            margin-top: 2.5rem;
            padding: 2rem;
            color: var(--color-text-muted);
            font-size: 0.85rem;
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
                border-radius: 0;
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

        /* Evidence chips + legend (issue #102) — one visual language for trust. */
{EVIDENCE_CHIP_CSS}
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
        self._allocation: dict | None = None
        self._llm: Any | None = None
        self._config_kwargs: dict = {}
        self._section_configs: dict[str, SectionConfig] = {}

    def with_llm(self, llm: Any) -> ReportBuilder:
        """Attach a LangChain chat model used to enrich CMO/planner narrative
        in the Augur readout (best-effort; templated fallback otherwise)."""
        self._llm = llm
        return self

    def with_insights(self, insights: dict[str, str]) -> ReportBuilder:
        """Pre-supply CMO/planner narrative slots (skips insight generation)."""
        self._config_kwargs["cmo_insights"] = dict(insights or {})
        return self

    def augur_readout(self) -> ReportBuilder:
        """Configure the editorial "Media Performance Readout" (Augur) shell.

        Sets the Augur palette + cream/ink shell, a numbered sticky contents
        nav, a confidentiality notice and channel-name formatting. The Augur
        section set (headline → reallocation → deep dives → posterior-predictive
        fit & checks → tests → next steps) is selected by ``shell == "augur"``.
        """
        self._config_kwargs.update(
            {
                "shell": "augur",
                "color_scheme": ColorScheme.from_palette(ColorPalette.AUGUR),
                "large_number_format": "short",
                "show_nav": True,
                "confidential": True,
                "format_channel_names": True,
            }
        )
        return self

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

    def with_allocation(self, plan: dict | None) -> ReportBuilder:
        """Attach a budget-allocation plan (a default reallocation from
        :func:`planning.default_reallocation`, or a saved Planner plan).

        The plan is exposed on the report bundle as ``allocation_results`` and
        turns the allocation section ON (classic *and* Augur). A falsy/empty plan
        is ignored, so the section stays off. The plan dict shape is the one the
        ``plan_budget`` op / ``AllocationSection`` consume."""
        self._allocation = plan
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
            llm=self._llm,
            allocation=self._allocation,
        )
