"""
Consultant artifacts for the MMM Framework.

Generates the engagement-playbook documents a consultancy practice lead
needs — diagnostic checklist, experiment pre-registration memo, data
onboarding checklist, executive summary template, and engagement
timeline — as polished, standalone, printable HTML.

The artifacts share the editorial design language of
:class:`~mmm_framework.reporting.generator.MMMReportGenerator`
(masthead header, numbered sections, editorial tables, callouts) and
are content-faithful to the static documentation in ``docs/``
(stress-05 gauntlet, measurement-calibration, data-requirements,
interpreting-results, modeling-guide).

Usage
-----
>>> from mmm_framework.reporting import render_artifact, write_all, ARTIFACTS
>>> html = render_artifact("diagnostic_checklist")
>>> paths = write_all("docs/artifacts/")

Or from the command line::

    python -m mmm_framework.reporting.consultant_artifacts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

from .design_tokens import TOKENS

try:  # version stamp for the masthead
    from mmm_framework import __version__ as _FRAMEWORK_VERSION
except Exception:  # pragma: no cover - fallback for unusual import orders
    _FRAMEWORK_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# CSS — mirrors MMMReportGenerator._generate_css() (masthead, numbered
# sections, metric cards, editorial tables, callouts, print rules) plus
# artifact-specific additions: printable checkboxes, fill-in fields,
# signature blocks, and tighter print compaction for 1–2 page output.
# ---------------------------------------------------------------------------

_ARTIFACT_CSS = f"""
        :root {{
            --color-primary: {TOKENS.primary};
            --color-primary-dark: {TOKENS.primary_dark};
            --color-accent: {TOKENS.accent};
            --color-accent-dark: {TOKENS.accent_dark};
            --color-warning: {TOKENS.warning};
            --color-danger: {TOKENS.danger};
            --color-success: {TOKENS.success};
            --color-text: {TOKENS.text};
            --color-text-muted: {TOKENS.text_muted};
            --color-bg: {TOKENS.background};
            --color-bg-alt: {TOKENS.background_alt};
            --color-surface: {TOKENS.surface};
            --color-border: {TOKENS.border};
            --shadow-sm: {TOKENS.shadow_sm};
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: {TOKENS.font_sans};
            background: var(--color-bg);
            color: var(--color-text);
            line-height: 1.6;
        }}

        .report-container {{
            max-width: 960px;
            margin: 0 auto;
            padding: 2rem;
            counter-reset: report-section;
        }}

        /* Header — editorial masthead */
        .report-header {{
            border-top: 4px solid var(--color-primary-dark);
            border-bottom: 1px solid var(--color-border);
            padding: 1.5rem 0 1.75rem;
            margin-bottom: 2rem;
        }}

        .masthead-meta {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            flex-wrap: wrap;
            gap: 0.5rem 1.5rem;
            font-family: {TOKENS.font_mono};
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--color-text-muted);
            margin-bottom: 1.5rem;
        }}

        .report-header h1 {{
            font-family: {TOKENS.font_serif};
            font-size: 2.3rem;
            font-weight: 400;
            line-height: 1.15;
            letter-spacing: -0.01em;
            color: var(--color-text);
            margin-bottom: 0.6rem;
            max-width: 20em;
        }}

        .report-header .subtitle {{
            font-size: 1rem;
            color: var(--color-text-muted);
            max-width: 44em;
        }}

        /* Section styling — numbered, editorial */
        .section {{
            background: var(--color-surface);
            border-radius: 10px;
            padding: 1.75rem 2rem;
            margin-bottom: 1.5rem;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--color-border);
            counter-increment: report-section;
        }}

        .section > h2 {{
            font-family: {TOKENS.font_serif};
            font-size: 1.4rem;
            font-weight: 400;
            color: var(--color-text);
            margin-bottom: 1.1rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--color-border);
            position: relative;
        }}

        .section > h2::before {{
            content: counter(report-section, decimal-leading-zero);
            font-family: {TOKENS.font_mono};
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
            font-size: 1.05rem;
            color: var(--color-text);
            margin-top: 1.25rem;
            margin-bottom: 0.6rem;
        }}

        .section p {{
            margin-bottom: 0.9rem;
            color: var(--color-text);
        }}

        .section-subtitle {{
            color: var(--color-text-muted);
            font-size: 0.92rem;
            margin-bottom: 1.25rem;
        }}

        code {{
            font-family: {TOKENS.font_mono};
            font-size: 0.88em;
            background: var(--color-bg-alt);
            border-radius: 4px;
            padding: 0.1em 0.35em;
        }}

        /* Key metrics grid */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.25rem;
            margin: 1.25rem 0;
        }}

        .metric-card {{
            display: flex;
            flex-direction: column;
            background: var(--color-surface);
            border-radius: 8px;
            padding: 1.2rem 1.3rem 1.1rem;
            text-align: left;
            border: 1px solid var(--color-border);
            border-top: 3px solid var(--color-primary);
            break-inside: avoid;
        }}

        .metric-card .label {{
            order: -1;
            font-family: {TOKENS.font_mono};
            font-size: 0.68rem;
            font-weight: 500;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--color-text-muted);
            margin-bottom: 0.6rem;
        }}

        .metric-card .value {{
            font-size: 1.6rem;
            font-weight: 600;
            line-height: 1.2;
            color: var(--color-text);
            font-variant-numeric: tabular-nums;
        }}

        .metric-card .ci {{
            font-size: 0.78rem;
            color: var(--color-text-muted);
            margin-top: 0.5rem;
            font-family: {TOKENS.font_mono};
            font-variant-numeric: tabular-nums;
        }}

        .metric-card.highlight {{
            border-top-color: var(--color-accent);
            background: var(--color-bg-alt);
        }}

        /* Callout boxes */
        .callout {{
            border-radius: 8px;
            padding: 1.1rem 1.3rem;
            margin: 1.25rem 0;
            break-inside: avoid;
        }}

        .callout h4 {{
            margin-bottom: 0.5rem;
            font-family: {TOKENS.font_mono};
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }}

        .callout p {{
            margin-bottom: 0;
            font-size: 0.93rem;
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

        .callout.danger {{
            background: rgba(201, 112, 103, 0.08);
            border: 1px solid rgba(201, 112, 103, 0.3);
            border-left: 4px solid var(--color-danger);
        }}

        .callout.danger h4 {{ color: var(--color-danger); }}

        /* Tables — editorial: rule-based, tabular numerals */
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.25rem 0;
            font-size: 0.88rem;
            font-variant-numeric: tabular-nums;
        }}

        .data-table th, .data-table td {{
            padding: 0.55rem 0.8rem;
            text-align: left;
            border-bottom: 1px solid var(--color-border);
            vertical-align: top;
        }}

        .data-table th {{
            background: transparent;
            border-bottom: 2px solid var(--color-text);
            font-family: {TOKENS.font_mono};
            font-weight: 500;
            color: var(--color-text-muted);
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}

        .data-table tbody tr {{ break-inside: avoid; }}

        .data-table tbody tr:last-child td {{
            border-bottom: 2px solid var(--color-border);
        }}

        .data-table tbody tr:hover {{
            background: var(--color-bg-alt);
        }}

        .data-table .mono {{
            font-family: {TOKENS.font_mono};
            font-size: 0.82em;
        }}

        /* Checklist rows — printable empty checkbox via bordered span */
        .check-list {{ margin: 0.75rem 0 0.25rem; }}

        .check-item {{
            display: flex;
            gap: 0.85rem;
            align-items: baseline;
            padding: 0.55rem 0.25rem;
            border-bottom: 1px solid var(--color-border);
            break-inside: avoid;
        }}

        .check-item:last-child {{ border-bottom: none; }}

        .check-box {{
            flex: none;
            display: inline-block;
            width: 0.9em;
            height: 0.9em;
            border: 1.5px solid var(--color-text-muted);
            border-radius: 2px;
            background: var(--color-surface);
            transform: translateY(0.12em);
        }}

        .check-body {{ flex: 1; }}

        .check-what {{
            font-weight: 600;
            font-size: 0.93rem;
        }}

        .check-catches {{
            display: block;
            color: var(--color-text-muted);
            font-size: 0.85rem;
            margin-top: 0.1rem;
        }}

        /* Fill-in fields — bottom-bordered blanks with mono placeholder */
        .fill-field {{
            display: inline-block;
            min-width: 11em;
            margin: 0 0.3em;
            padding: 0.05em 0.5em 0;
            border-bottom: 1px solid var(--color-text-muted);
            font-family: {TOKENS.font_mono};
            font-size: 0.68rem;
            font-weight: 500;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--color-text-muted);
            line-height: 2.1;
            vertical-align: baseline;
        }}

        .fill-field.wide {{
            display: block;
            width: 100%;
            margin: 0.65rem 0;
        }}

        .fill-field.tall {{ line-height: 3.2; }}

        .metric-card .ci .fill-field {{ min-width: 4em; margin: 0 0.15em; }}

        .fill-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 0.4rem 1.5rem;
            margin: 0.75rem 0 1rem;
        }}

        .fill-grid .fill-field {{ display: block; margin: 0.35rem 0; }}

        /* Signature block */
        .signature-block {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 2rem;
            margin-top: 2.25rem;
            break-inside: avoid;
        }}

        .signature-line {{
            border-top: 1px solid var(--color-text);
            margin-top: 2.6rem;
            padding-top: 0.4rem;
            font-family: {TOKENS.font_mono};
            font-size: 0.66rem;
            font-weight: 500;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--color-text-muted);
        }}

        /* Index page artifact cards */
        .artifact-card {{
            display: block;
            text-decoration: none;
            color: inherit;
            background: var(--color-surface);
            border: 1px solid var(--color-border);
            border-left: 4px solid var(--color-primary);
            border-radius: 8px;
            padding: 1.1rem 1.4rem;
            margin: 0.9rem 0;
            break-inside: avoid;
        }}

        .artifact-card:hover {{ background: var(--color-bg-alt); }}

        .artifact-card .artifact-title {{
            font-family: {TOKENS.font_serif};
            font-size: 1.15rem;
            color: var(--color-text);
            margin-bottom: 0.25rem;
        }}

        .artifact-card .artifact-desc {{
            color: var(--color-text-muted);
            font-size: 0.92rem;
            margin-bottom: 0.35rem;
        }}

        .artifact-card .artifact-file {{
            font-family: {TOKENS.font_mono};
            font-size: 0.7rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--color-primary-dark);
        }}

        ol.fix-ladder {{ margin: 0.75rem 0 0.5rem 1.4rem; }}

        ol.fix-ladder li {{
            margin-bottom: 0.7rem;
            break-inside: avoid;
        }}

        /* Footer — colophon */
        .report-footer {{
            text-align: left;
            border-top: 1px solid var(--color-border);
            margin-top: 2rem;
            padding: 1.25rem 0 2rem;
            color: var(--color-text-muted);
            font-size: 0.8rem;
            line-height: 1.6;
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            .report-container {{ padding: 1rem; }}
            .metrics-grid {{ grid-template-columns: 1fr 1fr; }}
            .report-header h1 {{ font-size: 1.7rem; }}
        }}

        /* Print styles — these are printed and filled in by hand */
        @page {{
            margin: 14mm 15mm;
        }}

        @media print {{
            /* rem-based sizes scale from the root: shrink it once, everything follows */
            html {{ font-size: 10.5px; }}
            body {{ background: white; font-size: 0.85rem; line-height: 1.4; }}
            .report-container {{ max-width: none; padding: 0; margin: 0; }}
            .report-header {{
                border-top-width: 5px;
                padding: 0.6rem 0 0.9rem;
                margin-bottom: 1.1rem;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
            .report-header h1 {{ font-size: 1.7rem; }}
            .masthead-meta {{ margin-bottom: 0.8rem; }}
            .section {{
                box-shadow: none;
                border: none;
                border-top: 1px solid #ddd;
                border-radius: 0;
                padding: 0.6rem 0;
                margin-bottom: 0.15rem;
            }}
            .section > h2 {{ font-size: 1.15rem; margin-bottom: 0.5rem; padding-bottom: 0.35rem; }}
            .section > h2::before {{ margin-bottom: 0.15rem; }}
            .section > h2::before,
            .section > h2::after,
            .metric-card,
            .check-box,
            .callout {{
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
            .data-table {{ font-size: 0.7rem; margin: 0.7rem 0; }}
            .data-table th, .data-table td {{ padding: 0.28rem 0.4rem; }}
            .data-table tbody tr:hover {{ background: transparent; }}
            .check-item {{ padding: 0.25rem 0.1rem; }}
            .check-catches {{ font-size: 0.78rem; }}
            .callout {{ padding: 0.6rem 0.85rem; margin: 0.7rem 0; }}
            .metrics-grid {{ gap: 0.7rem; margin: 0.8rem 0; }}
            .metric-card {{ padding: 0.7rem 0.85rem 0.6rem; }}
            .fill-field {{ line-height: 1.9; }}
            .fill-field.tall {{ line-height: 2.6; }}
            .fill-grid {{ gap: 0.2rem 1.2rem; margin: 0.5rem 0 0.7rem; }}
            .signature-block {{ margin-top: 1.2rem; }}
            .signature-line {{ margin-top: 1.8rem; }}
            ol.fix-ladder li {{ margin-bottom: 0.35rem; }}
            .section p {{ margin-bottom: 0.55rem; }}
            .section-subtitle {{ margin-bottom: 0.7rem; }}
            .signature-block {{ page-break-inside: avoid; }}
            .artifact-card {{ box-shadow: none; }}
            .report-footer {{ page-break-inside: avoid; padding-bottom: 0; margin-top: 1.25rem; }}
            a {{ color: inherit; text-decoration: none; }}
        }}
"""


# ---------------------------------------------------------------------------
# Small HTML helpers
# ---------------------------------------------------------------------------


def _check(what: str, catches: str = "") -> str:
    """A checklist row: printable checkbox + check + what it catches."""
    catches_html = f'<span class="check-catches">{catches}</span>' if catches else ""
    return (
        '<div class="check-item"><span class="check-box"></span>'
        f'<span class="check-body"><span class="check-what">{what}</span>'
        f"{catches_html}</span></div>"
    )


def _check_list(*items: str) -> str:
    return f'<div class="check-list">{"".join(items)}</div>'


def _fill(label: str, wide: bool = False, tall: bool = False) -> str:
    """A fill-in blank: bottom-bordered span with a mono placeholder label."""
    cls = "fill-field" + (" wide" if wide else "") + (" tall" if tall else "")
    return f'<span class="{cls}">{label}</span>'


def _callout(kind: str, heading: str, body: str) -> str:
    return f'<div class="callout {kind}"><h4>{heading}</h4><p>{body}</p></div>'


def _section(title: str, body: str, subtitle: str = "") -> str:
    sub = f'<p class="section-subtitle">{subtitle}</p>' if subtitle else ""
    return f'<section class="section"><h2>{title}</h2>{sub}{body}</section>'


def _table(headers: list[str], rows: list[list[str]], extra_class: str = "") -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    cls = f"data-table {extra_class}".strip()
    return (
        f'<table class="{cls}"><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table>"
    )


def _signatures(*roles: str) -> str:
    lines = "".join(f'<div class="signature-line">{r}</div>' for r in roles)
    return f'<div class="signature-block">{lines}</div>'


# ---------------------------------------------------------------------------
# Artifact 1 — MMM Diagnostic Checklist
# Sources: docs/stress-05-gauntlet.html (EDA pre-flight, 15-row decision
# table, fix ladder, doctrine), docs/mmm-walkthrough.html (pre-modeling
# checklist).
# ---------------------------------------------------------------------------


def _build_diagnostic_checklist() -> str:
    preflight = _check_list(
        _check(
            "Outlier screen — every channel's spend max/median well under the ~8× danger line.",
            "Catches data-entry spikes and one-off bursts that crush max-based "
            "normalization and silently flatten the response curve.",
        ),
        _check(
            "Identifiability screen — spend CoV high enough (≳ 0.3) for every channel.",
            "Catches nearly always-on channels with almost no variation to learn from: "
            "expect prior-dominated, wide posteriors no matter what you do observationally. "
            "Flag them for experiments now, before any fit.",
        ),
        _check(
            "Collinearity screen — no spend pair with |corr| > 0.9 / VIF > 10 left unhandled.",
            "Catches channel splits the data cannot identify: the combined effect is "
            "identified, the split isn't.",
        ),
        _check(
            "Demand-chasing screen (the mirage table) — corr(spend, KPI) next to "
            "corr(spend, demand proxy) for every channel.",
            "Catches confounded ROAS: when spend chases demand, corr(spend, KPI) measures "
            "the bidding rule, not the ad. If spend is driven by demand, the demand proxy "
            "is a mandatory control.",
        ),
        _check(
            "Causal-role classification — every control classified: confounder (must "
            "include, don't shrink) / precision control / mediator (exclude as control) / irrelevant.",
            "Catches mediators included as controls — controlling for what a channel is "
            "bought to do destroys that channel's effect. The mediator call comes from "
            "institutional knowledge, not a correlation screen.",
        ),
        _check(
            "KPI structure read — trend, seasonality, and level breaks understood before fitting.",
            "Catches regime changes a constant-coefficient MMM can't absorb; never let "
            "media absorb a break.",
        ),
        _check(
            "Pre-specification — adstock, saturation, priors, trend/seasonality "
            "pre-specified now, before seeing results.",
            "Catches researcher degrees of freedom: structural pivots must later be chosen "
            "from symptoms, never from disliked answers.",
        ),
    )

    decision_rows = [
        [
            "1",
            "Divergences / r-hat &gt; 1.01",
            "Model fighting the data (misspecification, conflict-laden priors) — not merely a tuning problem",
            "Where do divergent draws sit? Does the conflict survive <code>target_accept</code> 0.95?",
            "Triage parameters (learning, pair plots) <strong>before</strong> re-tuning; fix the structure the conflict points at",
        ],
        [
            "2",
            "One channel's HDI ≫ its estimate",
            "Weak/flat spend (low CoV) — intercept-confounded",
            "CoV &lt; ~0.3?",
            "Report the interval, or buy an experiment; no model rescues a flat series",
        ],
        [
            "3",
            "Two channels' HDIs huge <em>and</em> their spends correlated",
            "Collinearity — the combined effect is identified, the split isn't",
            "Spend corr &gt; ~0.9 / VIF &gt; 10",
            "Group them, or lift-test one channel to anchor the split",
        ],
        [
            "4",
            "PPC fails (shape / autocorrelation)",
            "Missing time structure: trend break, seasonality, carryover misspecification",
            "Plot residuals over time and by season",
            "Richer trend / seasonality / adstock — then re-check",
        ],
        [
            "5",
            "PPC <strong>passes</strong>",
            "— it certifies fit, not attribution",
            "(none — that's the point)",
            "Never read a green PPC as causal validation",
        ],
        [
            "6",
            "Confounding robustness value (RV) low on a channel",
            "Estimate would be nullified by a mild confounder",
            "Does a candidate confounder that strong exist?",
            "Find/keep the proxy control; calibrate",
        ],
        [
            "7",
            "RV high but a plausible confounder is <em>known</em>",
            "RV measures required strength, not existence",
            "corr(spend, proxy), corr(proxy, KPI)",
            "Control the proxy regardless; RV is no green light",
        ],
        [
            "8",
            "Parameter prior-dominated (contraction ≈ 0, overlap ≈ 1)",
            "The data carries no information (flat, collinear, short window)",
            "CoV, spend correlations, window length",
            "Experiment, or own the prior in the writeup",
        ],
        [
            "9",
            "<strong>Negative contraction</strong> + big shift",
            "Prior–data conflict — often confounding demanding an absurd effect",
            "corr(spend, demand proxy)",
            "Close the back-door; if it persists, calibrate",
        ],
        [
            "10",
            "Estimate piled on the sign constraint (β ≈ 0 under a positive prior)",
            "True effect negative (cannibalization) or mediated-away",
            "Cross-product / holdout correlations; mediator inventory",
            "Allow the sign / model the cross-effect or mediation explicitly",
        ],
        [
            "11",
            "Spend max/median ≫ ~8",
            "Data-entry spike or one-off burst → normalization trap",
            "The 30-second max/median screen",
            "Verify, winsorize/cap, percentile-normalize",
        ],
        [
            "12",
            "KPI level break",
            "Regime change a constant-coefficient MMM can't absorb",
            "Plot the KPI; ask the business what happened",
            "Dummy / piecewise trend; never let media absorb a break",
        ],
        [
            "13",
            "Holiday-shaped residuals",
            "Seasonality misspecification",
            "Residuals by week-of-year",
            "Holiday terms / higher Fourier order",
        ],
        [
            "14",
            "A ROAS that would make the channel the best investment the company owns",
            "Demand confounding — a mirage by construction",
            "corr(spend, demand proxy); benchmark sanity",
            "Treat as confounded until an experiment says otherwise",
        ],
        [
            "15",
            "Naive corr(spend, KPI) ≫ any plausible ROAS",
            "Spend chases demand — the dashboard mirage",
            "The mirage table",
            "The proxy control is mandatory; calibrate the channel",
        ],
    ]
    decision_table = _table(
        ["#", "Symptom", "Likely cause", "Cheap check", "Action"], decision_rows
    )

    fix_ladder = """
    <ol class="fix-ladder">
        <li><strong>Estimand first.</strong> Pin down what "contribution" means
            (counterfactual zero-out) before reading any posterior. If you can't say
            what number would be wrong, no diagnostic can save you.</li>
        <li><strong>EDA pre-flight.</strong> The mirage table, the CoV screen, the
            max/median outlier screen, the causal-role classification — minutes of work
            that can flag every failure a dataset contains before the first sample is drawn.</li>
        <li><strong>Structural pivots — chosen from symptoms, never from disliked
            answers.</strong> Demand controls help but don't identify; functional-form
            pivots matter only when form is the problem. Pivot at the structure the
            symptom points at, then re-check.</li>
        <li><strong>Experiments calibrate what structure can't.</strong> A lift test is
            surgical on the tested channel — and not contagious: untested channels stay
            wrong. That is why the deliverable ends with a costed test plan, not a full
            table of midpoints.</li>
    </ol>
    """

    doctrine = _callout(
        "danger",
        "The doctrine",
        "<strong>Convergence checks the <em>sampler</em>, never the "
        "<em>attribution</em>.</strong> Green diagnostics — clean r-hat, zero "
        "divergences, a passing posterior predictive check — validate the "
        "computation, not the causal claims. Identification comes from design: "
        "declared causal roles, pre-specified structure, and experiments.",
    )

    return (
        _section(
            "EDA pre-flight",
            preflight,
            "Run before the first fit. Minutes of work that flag most failures "
            "before any sample is drawn.",
        )
        + _section(
            "Symptom → diagnosis → action",
            decision_table,
            "The decision table. Find the symptom, run the cheap check, take the pivot.",
        )
        + _section(
            "The fix ladder",
            fix_ladder,
            "Ordered escalation — exhaust each rung before climbing to the next.",
        )
        + _section("What green diagnostics mean", doctrine)
    )


# ---------------------------------------------------------------------------
# Artifact 2 — Experiment Pre-Registration Memo
# Source: docs/measurement-calibration.html (experimental design section:
# matched-pair construction, treatment intensity, duration from serial
# correlation, power ceiling, ITT/stopping discipline).
# ---------------------------------------------------------------------------


def _build_preregistration_memo() -> str:
    header_block = (
        '<div class="fill-grid">'
        + _fill("experiment id")
        + _fill("date drafted")
        + _fill("owner / analyst")
        + "</div>"
    )

    amendment_callout = _callout(
        "warning",
        "Pre-registration discipline",
        "This memo is written <strong>before</strong> launch. Changes after launch "
        "are amendments — dated, justified, and logged, never silently applied. "
        "Analysis is intention-to-treat by default; all deviations from this plan "
        "are logged with the readout.",
    )

    hypothesis = "<p>Channel under test: " + _fill(
        "channel"
    ) + "</p>" "<p>Hypothesis (directional, falsifiable):</p>" + _fill(
        "hypothesis — e.g. incremental ROAS of channel X exceeds 1.0",
        wide=True,
        tall=True,
    )

    estimand = (
        "<p>Exactly one primary estimand, stated as a counterfactual contrast:</p>"
        + _check_list(
            _check(
                "Incremental <strong>contribution</strong> (KPI units) over the window"
            ),
            _check(
                "Incremental <strong>ROAS</strong> (contribution per dollar) over the window"
            ),
            _check(
                "<strong>mROAS</strong> (marginal ROAS at current spend) over the window"
            ),
        )
        + "<p>Measurement window: "
        + _fill("start date")
        + " to "
        + _fill("end date")
        + " &nbsp;·&nbsp; KPI: "
        + _fill("kpi + units")
        + "</p>"
    )

    design = (
        _check_list(
            _check(
                "Randomized matched-pair geo lift",
                "Treatment geos selected for information yield; controls matched on "
                "pre-period behavior (posterior Mahalanobis distance over geo effects).",
            ),
            _check(
                "Matched-market difference-in-differences",
                "When randomization is infeasible; serial correlation must enter the "
                "power calculation, with placebo checks pre-specified.",
            ),
            _check(
                "Budget-neutral randomized flighting (national data)",
                "When no geo split exists; on/off schedule randomized within the window.",
            ),
        )
        + "<p>Why this design for this channel and dataset:</p>"
        + _fill("design rationale", wide=True, tall=True)
    )

    geo_placeholder = '<span class="mono">________</span>'
    geo_rows = [
        [str(i), geo_placeholder, geo_placeholder, geo_placeholder, geo_placeholder]
        for i in range(1, 5)
    ]
    geos = (
        _table(
            [
                "Pair",
                "Treatment geo",
                "Control geo",
                "Matching distance",
                "Spillover screen |r| &lt; 0.5",
            ],
            geo_rows,
        )
        + "<p>Matching basis: "
        + _fill("matching basis — e.g. pre-period KPI + posterior mahalanobis")
        + "</p>"
        + "<p>Control candidates with cross-geo posterior correlation |r| &gt; 0.5 "
        "are excluded from the donor pool (spillover screen).</p>"
    )

    power = (
        '<div class="fill-grid">'
        + _fill("σ_y — residual sd")
        + _fill("ρ — ar(1) of residuals")
        + _fill("mde — Δy to detect")
        + _fill("α — significance level")
        + _fill("target power 1−β")
        + _fill("design effect d(t, ρ)")
        + _fill("Δspend per geo per week")
        + _fill("duration t* (weeks)")
        + "</div>"
        + "<p>Serial correlation inflates the variance of a T-week mean by the design "
        "effect D(T, ρ); ignoring it is the classic way to overstate the precision of "
        "difference-in-differences estimates. Spend delta is set from the saturation "
        "curve so the dose is detectable at the planned duration.</p>"
        + _check_list(
            _check(
                "Power-ceiling check passed: lim<sub>T→∞</sub> Power(T) exceeds the target.",
                "ROI-posterior uncertainty sets an asymptotic power ceiling that no "
                "duration recovers. If the ceiling is below target, the design is "
                "infeasible — add geos or increase the spend delta and re-simulate; "
                "do not simply extend the test.",
            ),
            _check(
                "Pre-experiment simulation run: design recovers the MDE at target "
                "power in forward simulation from the fitted model."
            ),
        )
    )

    analysis = "<p>Primary metric: " + _fill(
        "primary metric"
    ) + "</p>" "<p>Estimator: " + _fill(
        "estimator — matched-pair diff / did"
    ) + " &nbsp;·&nbsp; α = " + _fill(
        "alpha"
    ) + " &nbsp;·&nbsp; Interval: " + _fill(
        "interval — e.g. 90% ci"
    ) + "</p>" + _check_list(
        _check(
            "Intention-to-treat analysis by default; per-protocol only as a logged secondary."
        ),
        _check("Placebo / pre-period falsification check pre-specified."),
        _check(
            "Carryover guard: first 1–2 weeks treated as burn-in for adstocked channels."
        ),
    )

    stopping = (
        "<p>Stopping rule (pre-specified — no unplanned interim looks; any interim "
        "analysis must be written here before launch):</p>"
        + _fill(
            "stopping rule — e.g. fixed horizon at t* weeks, no peeking",
            wide=True,
            tall=True,
        )
    )

    decision_rows = [
        [
            "If " + '<span class="mono">____________________</span>',
            "then " + '<span class="mono">____________________</span>',
        ],
        [
            "If " + '<span class="mono">____________________</span>',
            "then " + '<span class="mono">____________________</span>',
        ],
        [
            "If the interval is too wide to distinguish the above",
            "then " + '<span class="mono">____________________</span>',
        ],
    ]
    decision = (
        "<p>What result updates what decision — agreed before launch:</p>"
        + _table(["Readout", "Decision"], decision_rows)
        + _callout(
            "insight",
            "Calibration path",
            "The readout (estimate + interval) feeds back into the MMM as an "
            "experiment calibration at the next refit, narrowing the tested "
            "channel's posterior. An experiment that cannot change a decision "
            "or a posterior should not run.",
        )
    )

    signoff = _signatures(
        "Analyst — name / date", "Practice lead — name / date", "Client — name / date"
    )

    return (
        header_block
        + amendment_callout
        + _section("Channel &amp; hypothesis", hypothesis)
        + _section(
            "Estimand",
            estimand,
            "Precise: which counterfactual quantity, over which window.",
        )
        + _section("Design", design)
        + _section("Treatment &amp; control geos", geos)
        + _section("Dose, duration &amp; power", power)
        + _section("Primary metric &amp; analysis plan", analysis)
        + _section("Stopping rule", stopping)
        + _section("Decision rule", decision)
        + _section("Sign-off", signoff)
    )


# ---------------------------------------------------------------------------
# Artifact 3 — Client Data Onboarding Checklist
# Sources: docs/data-requirements.html (MFF dictionary, minimums, hygiene),
# docs/mmm-walkthrough.html (pre-modeling checklist).
# ---------------------------------------------------------------------------


def _build_data_onboarding_checklist() -> str:
    inventory = _check_list(
        _check(
            "KPI series — revenue or units, covering the full modeling window.",
            "One series per outcome; weekly is the reference frequency.",
        ),
        _check(
            "Media spend by channel — every channel, full window.",
            "Prefer spend over impressions when ROI and budget-allocation readouts are "
            "wanted directly (contributions-per-dollar need dollars in the denominator); "
            "use impressions/GRPs when CPMs shifted materially, then convert with cost data.",
        ),
        _check(
            "Controls — price, distribution, promotions, and a category-demand proxy.",
            "If spend chases demand, the demand proxy is a mandatory confounder control, "
            "not optional.",
        ),
        _check(
            "Experiment history — past geo lift tests / RCTs with estimates and intervals.",
            "These become calibration anchors; even old tests narrow the right posteriors.",
        ),
    )

    hygiene = _check_list(
        _check(
            "Long-format MFF — eight columns: <code>Period</code>, <code>Geography</code>, "
            "<code>Product</code>, <code>Campaign</code>, <code>Outlet</code>, "
            "<code>Creative</code>, <code>VariableName</code>, <code>VariableValue</code>.",
            "All eight columns must exist; the five dimension columns may be blank for "
            "data that doesn't use them. One numeric value per row.",
        ),
        _check(
            "Dates parse — default format <code>%Y-%m-%d</code>; consistent frequency "
            "(W weekly / D daily / M monthly).",
            "Unparseable dates fail loading. Weekly (weeks ending Sunday) is the "
            "default and what the published benchmarks use.",
        ),
        _check(
            "Deduplicated — one row per period × dimensions × variable.",
            "Duplicate rows silently double values when pivoted.",
        ),
        _check(
            "NaN-vs-0 policy confirmed with the client.",
            "Default: missing media values fill with 0.0 (no record of spend = no "
            "spend); missing control values forward-fill. Confirm this matches reality "
            "before loading.",
        ),
        _check(
            "Currency and units consistent across the window and across channels.",
            "Mixed currencies or a mid-window unit change corrupts ROI denominators.",
        ),
        _check(
            "Sanity — no negative spend, declared variables all present.",
            "Undeclared names are ignored with a warning; missing declared names fail "
            "loading. The loader and EDA module catch most structural errors.",
        ),
    )

    sufficiency = _check_list(
        _check(
            "History length — 104+ weekly observations (2 years) preferred.",
            "Two full seasonal cycles are needed to separate seasonality from trend and "
            "from media timing; one cycle leaves them confounded.",
        ),
        _check(
            "Zero-spend share — under 50% zero-spend weeks per channel.",
            "A channel that's almost always dark carries too little signal to estimate "
            "a response curve.",
        ),
        _check(
            "Spend variation &amp; positivity — channels sometimes go dark or near-dark, "
            "and channels do not all flight together.",
            "ROI is a do(spend = 0) counterfactual: a channel that never goes dark gives "
            "the model no observations near zero, so the answer is extrapolated from the "
            "prior. Synchronized flighting makes the split between channels unidentifiable.",
        ),
        _check(
            "Controls cover the entire series.",
            "Gaps in confounders (price, distribution, promo) become gaps in causal "
            "adjustment exactly where you need it.",
        ),
        _check(
            "Geo counts, if hierarchical — at least 2 geographies, and per-geo spend "
            "variation (CoV across geos ≥ ~0.15 per channel).",
            "The binding constraint is not the count but per-geo spend variation; "
            "channels spent uniformly across geos carry no geo-level identifying "
            "information.",
        ),
    )

    governance = _check_list(
        _check(
            "Data dictionary received and reconciled against the declared variables.",
        ),
        _check(
            "Refresh cadence agreed — who delivers, how often, in what format.",
        ),
        _check(
            "PII confirmation — none needed for MMM.",
            "The model consumes aggregate spend and KPI series only; confirm no "
            "user-level data is in the delivery.",
        ),
        _check(
            "Named contact for data anomalies, with an agreed turnaround.",
        ),
    )

    gate = _callout(
        "insight",
        "The gate",
        "Onboarding ends with the EDA pre-flight (see the MMM Diagnostic "
        "Checklist): outlier screen, CoV screen, collinearity screen, "
        "demand-chasing screen, causal-role classification. Fitting starts only "
        "after this page and that pre-flight are signed.",
    )

    return (
        _section(
            "Data inventory", inventory, "What must arrive before anything is modeled."
        )
        + _section(
            "Format &amp; hygiene",
            hygiene,
            "The Master Flat File (MFF) contract the loader validates.",
        )
        + _section(
            "Sufficiency gates",
            sufficiency,
            "Minimums below which estimates are priors wearing a model.",
        )
        + _section("Governance", governance)
        + gate
    )


# ---------------------------------------------------------------------------
# Artifact 4 — One-Page Executive Summary Template
# Sources: docs/interpreting-results.html (evidence tiers, no-naked-point-
# estimates), docs/mmm-example-report.html (headline + channel-table format).
# ---------------------------------------------------------------------------


def _build_exec_summary_template() -> str:
    headline = (
        '<div class="metrics-grid">'
        '<div class="metric-card"><span class="label">Total KPI (window)</span>'
        f'<span class="value">{_fill("total kpi")}</span></div>'
        '<div class="metric-card highlight"><span class="label">Marketing-attributed KPI</span>'
        f'<span class="value">{_fill("attributed")}</span>'
        f'<span class="ci">80% CI: [ {_fill("low")} – {_fill("high")} ]</span></div>'
        '<div class="metric-card"><span class="label">Blended ROAS [interval]</span>'
        f'<span class="value">{_fill("roas")}</span>'
        f'<span class="ci">80% CI: [ {_fill("low")} – {_fill("high")} ]</span></div>'
        "</div>"
        "<p>Headline, in one sentence — with its interval:</p>"
        + _fill("headline result incl. credible interval", wide=True, tall=True)
    )

    blank = '<span class="mono">________</span>'
    interval = '<span class="mono">____ [ ____ – ____ ]</span>'
    channel_rows = [[blank, interval, blank, blank] for _ in range(6)]
    channel_table = (
        _table(
            ["Channel", "ROAS [80% interval]", "Evidence tier", "Recommended action"],
            channel_rows,
        )
        + "<p><strong>Evidence tiers</strong> — "
        "<strong>Validated</strong>: consistent with experimental results (geo lift, "
        "RCT); act with confidence. "
        "<strong>Narrow, unvalidated</strong>: precise but never tested; reasonable "
        "for moderate budget changes — design validation tests. "
        "<strong>Wide, unvalidated</strong>: uncertain and untested; avoid major "
        "budget moves — prioritize experiments here, where testing has the highest "
        "information value.</p>"
        "<p><strong>Action vocabulary</strong> — Strong, consider increasing · "
        "Maintain current levels · Uncertain, recommend geo-test · Likely "
        "unprofitable, review · Reduce or eliminate.</p>"
    )

    verified = _check_list(
        _check(
            "Convergence diagnostics clean — r-hat ≤ 1.01, zero divergences across chains."
        ),
        _check(
            "Posterior predictive check passed — replicated data matches mean, variance, and autocorrelation of the observed KPI."
        ),
        _check(
            "Out-of-sample backtest — holdout forecast error within the pre-agreed tolerance."
        ),
        _check(
            "Sensitivity analysis — the headline holds across all pre-specified alternative specifications."
        ),
        _check(
            "Data informativeness — posteriors contracted from priors for every channel we recommend acting on."
        ),
    )

    change_mind = (
        "<p>The next experiment, already designed (see its pre-registration memo):</p>"
        '<div class="fill-grid">'
        + _fill("channel to test")
        + _fill("design — geo lift / did / flighting")
        + _fill("cost")
        + _fill("duration (weeks)")
        + "</div>"
        + "<p>Decision it updates: if the test reads "
        + _fill("readout a")
        + " we will "
        + _fill("action a")
        + "; if it reads "
        + _fill("readout b")
        + " we will "
        + _fill("action b")
        + ".</p>"
    )

    footer_rule = _callout(
        "danger",
        "No naked point estimates",
        "Every number on this page carries its interval. A point estimate without "
        "uncertainty is not a decision input: an optimizer fed point estimates "
        "alone will pile budget onto whichever channel is most flattered by noise. "
        "Where an interval is too wide to act on, the action is an experiment, "
        "not a guess.",
    )

    return (
        _section("Headline result", headline)
        + _section(
            "Channel summary",
            channel_table,
            "Every ROAS with its interval and its evidence tier — never a bare number.",
        )
        + _section("What we verified", verified)
        + _section(
            "What would change our mind",
            change_mind,
            "Honest attribution names the evidence that would revise it.",
        )
        + footer_rule
    )


# ---------------------------------------------------------------------------
# Artifact 5 — Engagement Timeline (10–12 week MMM engagement)
# Maps weeks to the measurement loop (T₀ fit → T₁ prioritize → T₂ experiment
# → T₃ calibrate → T₄ allocate → T₅ re-evaluate) and the docs workflow.
# ---------------------------------------------------------------------------


def _build_engagement_timeline() -> str:
    rows = [
        [
            "W1–2",
            "Data onboarding + EDA gate",
            "Data inventory, MFF assembly, hygiene checks; EDA pre-flight "
            "(outlier, CoV, collinearity, demand-chasing screens; causal-role "
            "classification)",
            "Signed Data Onboarding Checklist; EDA gate readout",
            "MFF loader validation; <code>mmm_framework.eda</code>",
        ],
        [
            "W3",
            "Pre-specification",
            "Define estimand and decision questions; declare causal roles; "
            "pre-register adstock, saturation, priors, trend/seasonality; "
            "identification-assumptions sign-off",
            "Pre-registered model specification",
            "Modeling-guide Phase 1 (Plan); the identification contract",
        ],
        [
            "W4–5",
            "Fit + validation battery (T₀)",
            "Fit; convergence diagnostics (non-negotiable); posterior predictive "
            "check; out-of-sample backtest; pre-specified sensitivity analysis",
            "Validation battery report",
            "<code>BayesianMMM</code>; modeling-guide Phase 3 (Validate); "
            "parameter-learning diagnostic",
        ],
        [
            "W6",
            "Readout 1 — attribution with tiers",
            "Channel ROAS with intervals; evidence-tier classification "
            "(validated / narrow-unvalidated / wide-unvalidated); recommended "
            "actions per tier",
            "One-Page Executive Summary",
            "<code>MMMReportGenerator</code>; interpreting-results conventions",
        ],
        [
            "W7",
            "Experiment design + pre-registration (T₁–T₂)",
            "EIG/EVOI prioritization of what to test; geo-lift / matched-market "
            "DiD / flighting design with power calculation and power-ceiling "
            "check; memo signed by analyst, practice lead, client",
            "Signed Pre-Registration Memo",
            "Experiment priorities (EIG/EVOI); design studio "
            "(measurement-calibration)",
        ],
        [
            "W8–10",
            "Experiment in field",
            "Launch per memo; monitor delivery, not results — interim looks only "
            "if pre-specified; deviations logged; intention-to-treat by default",
            "Field log + amendments (if any)",
            "Experiment lifecycle registry (planned → running → completed)",
        ],
        [
            "W11",
            "Calibrated refit + final readout (T₃)",
            "Fold the experiment readout into the model as a calibration; refit; "
            "compare pre/post-calibration intervals on the tested channel",
            "Final readout — calibrated attribution",
            "Experiment calibration at refit (readout → narrowed posterior)",
        ],
        [
            "W12",
            "Allocation + next-cycle plan (T₄–T₅)",
            "Budget-allocation recommendation under uncertainty; re-evaluation: "
            "information decay schedule, next test priorities, refresh cadence",
            "Allocation memo + next-cycle test plan",
            "T₄ allocate / T₅ re-evaluate; information decay triggers re-tests",
        ],
    ]
    timeline = _table(
        ["Week", "Phase", "Activities", "Deliverable", "Framework feature"], rows
    )

    loop_note = _callout(
        "insight",
        "The loop continues",
        "Week 12 is not the end: experimental information decays as markets "
        "drift, so T₅ re-evaluation feeds the next cycle's T₁ priorities — each "
        "engagement leaves the client with narrower intervals and a standing "
        "test plan, not a one-off report.",
    )

    intro = (
        "<p>A 10–12 week engagement is one pass around the measurement loop: "
        "fit (T₀) → prioritize what to test (T₁) → pre-registered experiment "
        "(T₂) → calibrated refit (T₃) → allocate (T₄) → re-evaluate (T₅). "
        "Each row names the deliverable that gates the next week.</p>"
    )

    return _section("The engagement at a glance", intro + timeline) + loop_note


# ---------------------------------------------------------------------------
# Registry + rendering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactSpec:
    """Specification for one consultant artifact."""

    title: str
    description: str
    build: Callable[[], str] = field(repr=False)

    def filename(self, name: str) -> str:
        return name.replace("_", "-") + ".html"


ARTIFACTS: dict[str, ArtifactSpec] = {
    "diagnostic_checklist": ArtifactSpec(
        title="MMM Diagnostic Checklist",
        description=(
            "EDA pre-flight, the symptom → diagnosis → action decision table, "
            "and the fix ladder — the one-pager to keep next to every fit."
        ),
        build=_build_diagnostic_checklist,
    ),
    "preregistration_memo": ArtifactSpec(
        title="Experiment Pre-Registration Memo",
        description=(
            "Fill-in template locking estimand, design, geos, power (with the "
            "power-ceiling check), stopping rule, and the decision rule — "
            "before launch."
        ),
        build=_build_preregistration_memo,
    ),
    "data_onboarding_checklist": ArtifactSpec(
        title="Client Data Onboarding Checklist",
        description=(
            "Data inventory, MFF format and hygiene, sufficiency gates, and "
            "governance — what must be true of the data before anything is fit."
        ),
        build=_build_data_onboarding_checklist,
    ),
    "exec_summary_template": ArtifactSpec(
        title="One-Page Executive Summary Template",
        description=(
            "Fill-in readout: headline with interval, channel table with "
            "evidence tiers, what we verified, what would change our mind — "
            "and no naked point estimates."
        ),
        build=_build_exec_summary_template,
    ),
    "engagement_timeline": ArtifactSpec(
        title="Engagement Timeline — a 10–12 Week MMM Engagement",
        description=(
            "Week-by-week map of one pass around the measurement loop, from "
            "data onboarding to calibrated allocation, with the deliverable "
            "that gates each step."
        ),
        build=_build_engagement_timeline,
    ),
}


def _document(
    title: str,
    subtitle: str,
    content: str,
    kind: str = "MMM Framework — Consultant Artifact",
) -> str:
    """Wrap artifact content in a standalone HTML document."""
    today = date.today().isoformat()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} — MMM Framework</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Source+Sans+3:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
{_ARTIFACT_CSS}
    </style>
</head>
<body>
    <div class="report-container">
        <header class="report-header">
            <div class="masthead-meta">
                <span class="masthead-kind">{kind}</span>
                <span class="masthead-date">v{_FRAMEWORK_VERSION} · {today}</span>
            </div>
            <h1>{title}</h1>
            <div class="subtitle">{subtitle}</div>
        </header>
        {content}
        <footer class="report-footer">
            <p>Built with the MMM Framework · Apache-2.0 · github.com/redam94/mmm-framework
               — re-generate via <code>python -m mmm_framework.reporting.consultant_artifacts</code></p>
        </footer>
    </div>
</body>
</html>"""


def render_artifact(name: str) -> str:
    """Render one consultant artifact as a full standalone HTML document.

    Parameters
    ----------
    name : str
        Key in :data:`ARTIFACTS` (e.g. ``"diagnostic_checklist"``).

    Returns
    -------
    str
        Complete HTML document.
    """
    if name not in ARTIFACTS:
        valid = ", ".join(sorted(ARTIFACTS))
        raise KeyError(f"Unknown artifact {name!r}. Valid names: {valid}")
    spec = ARTIFACTS[name]
    return _document(spec.title, spec.description, spec.build())


def _build_index() -> str:
    """Build the index page listing all artifacts with download links."""
    cards = ""
    for name, spec in ARTIFACTS.items():
        fname = spec.filename(name)
        cards += f"""
        <a class="artifact-card" href="{fname}" download>
            <div class="artifact-title">{spec.title}</div>
            <div class="artifact-desc">{spec.description}</div>
            <div class="artifact-file">{fname} · download</div>
        </a>"""
    body = (
        "<p>Five engagement-playbook documents — print-ready, fill-in where "
        "fill-in belongs, and content-faithful to the framework's "
        "documentation. Each is a standalone HTML file: open, print, or attach.</p>"
        + cards
    )
    return _document(
        "Consultant Artifacts",
        "The engagement playbook: diagnostic checklist, pre-registration memo, "
        "data onboarding checklist, executive summary template, and engagement "
        "timeline.",
        _section("The artifacts", body),
        kind="MMM Framework — Consultant Artifacts",
    )


def write_all(out_dir: str | Path) -> list[Path]:
    """Write all artifacts plus an index page to ``out_dir``.

    Parameters
    ----------
    out_dir : str or Path
        Output directory (created if missing).

    Returns
    -------
    list[Path]
        Paths written: one per artifact (kebab-case filenames) plus
        ``index.html``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, spec in ARTIFACTS.items():
        path = out / spec.filename(name)
        path.write_text(render_artifact(name), encoding="utf-8")
        written.append(path)
    index_path = out / "index.html"
    index_path.write_text(_build_index(), encoding="utf-8")
    written.append(index_path)
    return written


if __name__ == "__main__":
    paths = write_all("consultant_artifacts")
    for p in paths:
        print(p)
