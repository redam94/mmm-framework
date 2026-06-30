"""Augur "Media Performance Readout" theme — the editorial stylesheet,
masthead logo and webfont link for the ``shell="augur"`` report.

The component CSS is ported verbatim from the Augur reference template (cream/ink
palette, Fraunces/IBM Plex Sans/JetBrains Mono, sage/gold/steel/rust evidence
tiers). ``augur_css`` appends a ``--color-*`` compatibility block derived from the
report ``ColorScheme`` so the reused chart kit and any legacy component classes
theme to the same palette.
"""

from __future__ import annotations

# Masthead mark — the Augur "sprout" (sage stem + gold seed).
MASTHEAD_LOGO_SVG = (
    '<svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<path d="M 25 53 L 25 32 C 25 18 36 11 43 16 C 48.5 20 47 29 40 29.5 C 35 29.8 33.5 24.5 37 22.5" '
    'fill="none" stroke="#a8c485" stroke-width="5.5" stroke-linecap="round"></path>'
    '<circle cx="47.5" cy="43.5" r="2.8" fill="#d4a86a"></circle></svg>'
)

# Google Fonts for the Augur typeface stack.
AUGUR_FONTS_LINK = (
    '<link href="https://fonts.googleapis.com/css2?'
    "family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&"
    "family=IBM+Plex+Sans:wght@400;500;600;700&"
    'family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">'
)


AUGUR_COMPONENT_CSS = r""":root{
  --cream-50:#faf8f3;--cream-100:#f3f0e6;--cream-200:#f0ede0;--cream-300:#e9e5d4;
  --ink-900:#2a3528;--ink-700:#3a4838;--ink-600:#4a5a48;--ink-400:#7a8a78;--ink-300:#9aa498;
  --sage-100:#eef2e7;--sage-300:#a8c485;--sage-600:#6d8a4a;--sage-700:#5a7a3a;--sage-800:#4a6d2a;
  --gold-100:#f5ecd8;--gold-300:#d4a86a;--gold-600:#b8860b;--gold-700:#8a6408;
  --steel-100:#e7eef2;--steel-300:#9db8c9;--steel-600:#4a6d8a;--steel-700:#3a5a75;
  --rust-100:#f5e7e2;--rust-600:#a04535;--rust-700:#7a3525;
  --line-200:#e8e4d5;--line-300:#d8d4c5;
  --shadow-sm:0 1px 2px 0 rgb(42 53 40 / .05);
  --font-display:"Fraunces",Georgia,serif;--font-sans:"IBM Plex Sans",system-ui,sans-serif;--font-mono:"JetBrains Mono",ui-monospace,monospace;
}
*{margin:0;padding:0;box-sizing:border-box;}
html{scroll-behavior:smooth;}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto;}}
body{font-family:var(--font-sans);background:var(--cream-50);color:var(--ink-700);line-height:1.6;font-size:15px;-webkit-font-smoothing:antialiased;}
.report-body{display:flex;align-items:flex-start;max-width:1240px;margin:0 auto;min-height:100vh;}
.report-container{flex:1;min-width:0;max-width:1020px;padding:2.5rem 2rem 3rem 2rem;}
.section{scroll-margin-top:1.25rem;}

/* Contents nav */
.report-nav{position:sticky;top:1.5rem;width:238px;min-width:238px;padding:.25rem 1.25rem .25rem 0;align-self:flex-start;border-right:1px solid var(--line-200);font-size:.82rem;max-height:calc(100vh - 3rem);overflow-y:auto;}
.report-nav .nav-head{font-size:.68rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-400);padding:0 1rem .7rem;margin-bottom:.35rem;border-bottom:1px solid var(--line-200);}
.report-nav a.nav-item{display:flex;align-items:baseline;gap:.6rem;padding:.45rem 1rem;color:var(--ink-600);text-decoration:none;border-left:2px solid transparent;transition:color .15s,border-color .15s,background .15s;line-height:1.3;}
.report-nav a.nav-item .nav-num{font-family:var(--font-mono);font-size:.7rem;color:var(--ink-300);font-variant-numeric:tabular-nums;flex:none;}
.report-nav a.nav-item:hover{color:var(--ink-900);background:var(--cream-100);}
.report-nav a.nav-item.active{color:var(--sage-800);border-left-color:var(--sage-700);background:var(--cream-100);font-weight:500;}
.report-nav a.nav-item.active .nav-num{color:var(--sage-700);}

/* Masthead */
.report-header{display:flex;align-items:center;gap:1.5rem;background:var(--ink-900);color:var(--cream-50);border-radius:12px;padding:2.25rem 2.5rem;margin-bottom:2rem;}
.masthead-logo{flex:none;width:60px;height:60px;border-radius:14px;border:1px solid rgba(168,196,133,.30);background:rgba(255,255,255,.04);display:grid;place-items:center;}
.masthead-logo svg{width:38px;height:38px;}
.masthead-text{min-width:0;}
.masthead-eyebrow{font-size:.72rem;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--sage-300);margin-bottom:.5rem;}
.report-header h1{font-family:var(--font-display);font-weight:600;font-size:2.15rem;line-height:1.08;letter-spacing:-.02em;color:#fff;margin-bottom:.55rem;}
.report-header .meta{font-family:var(--font-mono);font-size:.82rem;color:rgba(250,248,243,.66);}
.report-header .meta .sep{margin:0 .55em;color:rgba(250,248,243,.4);}
.report-header .conf{display:inline-block;margin-top:.85rem;font-size:.68rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--sage-300);border:1px solid rgba(168,196,133,.45);border-radius:4px;padding:.15em .65em;}

/* Sections */
.section{background:#fff;border:1px solid var(--line-200);border-radius:10px;padding:2rem 2.25rem;margin-bottom:1.5rem;box-shadow:var(--shadow-sm);}
.section-eyebrow{font-size:.7rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--sage-700);margin-bottom:.55rem;}
.section>h2{font-family:var(--font-display);font-size:1.5rem;font-weight:600;letter-spacing:-.02em;color:var(--ink-900);margin-bottom:1.1rem;padding-bottom:.8rem;border-bottom:1px solid var(--line-200);position:relative;}
.section>h2::after{content:"";position:absolute;left:0;bottom:-1px;width:44px;height:2px;background:var(--sage-700);}
.section p{margin-bottom:1rem;color:var(--ink-700);max-width:72ch;}
.section p:last-child{margin-bottom:0;}
.section h3{font-family:var(--font-display);font-size:1.12rem;font-weight:600;color:var(--ink-900);margin:1.75rem 0 .35rem;letter-spacing:-.01em;}
.lede{font-size:1.12rem;line-height:1.55;color:var(--ink-700);max-width:64ch;}
.lede strong{color:var(--ink-900);font-weight:600;}
.note{font-size:.82rem;color:var(--ink-400);max-width:74ch;}

/* KPI strip */
.kpi-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1.5rem 0;}
.kpi{display:flex;flex-direction:column;background:var(--cream-100);border:1px solid var(--line-200);border-radius:10px;padding:1.3rem 1.4rem;border-left:3px solid var(--sage-700);}
.kpi .label{order:-1;font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:var(--ink-400);margin-bottom:.5rem;}
.kpi .value{font-family:var(--font-display);font-size:1.95rem;font-weight:600;line-height:1.02;letter-spacing:-.02em;color:var(--ink-900);font-variant-numeric:tabular-nums;}
.kpi .ci{font-size:.74rem;color:var(--ink-400);margin-top:.5rem;font-family:var(--font-mono);}

/* Recommendation callout */
.rec{background:var(--sage-100);border:1px solid rgba(90,122,58,.22);border-left:3px solid var(--sage-700);border-radius:10px;padding:1.25rem 1.4rem;margin-top:1.5rem;}
.rec h4{font-size:.72rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--sage-800);margin-bottom:.75rem;}
.rec ul{list-style:none;display:flex;flex-direction:column;gap:.6rem;}
.rec li{display:flex;gap:.7rem;font-size:.93rem;color:var(--ink-700);align-items:flex-start;}
.rec li b{color:var(--ink-900);font-weight:600;}
.rec .marker{flex:none;margin-top:.42em;width:7px;height:7px;border-radius:50%;background:var(--sage-700);}

/* Table */
.data-table{width:100%;border-collapse:collapse;margin:.5rem 0 0;font-size:.9rem;border:1px solid var(--line-200);}
.data-table th,.data-table td{padding:.7rem 1rem;text-align:left;}
.data-table thead tr{background:var(--cream-100);}
.data-table th{font-weight:600;color:var(--ink-400);font-size:.7rem;text-transform:uppercase;letter-spacing:.07em;}
.data-table tbody tr{border-top:1px solid var(--line-200);}
.data-table td{color:var(--ink-700);vertical-align:middle;}
.data-table .mono{font-family:var(--font-mono);font-variant-numeric:tabular-nums;letter-spacing:-.01em;}
.data-table .chname{font-weight:600;color:var(--ink-900);}
.data-table .chname .swatch{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:.5rem;vertical-align:baseline;}
.data-table tbody tr:hover{background:rgba(243,240,230,.55);}

/* Tier chips */
.tier-chip{display:inline-flex;align-items:center;gap:.4rem;padding:.22rem .6rem;border-radius:999px;font-size:.72rem;font-weight:600;white-space:nowrap;}
.tier-chip::before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor;flex:none;}
.t-scale{background:var(--sage-100);color:var(--sage-800);}
.t-test{background:var(--gold-100);color:var(--gold-700);}
.t-hold{background:var(--steel-100);color:var(--steel-700);}
.t-reduce{background:var(--rust-100);color:var(--rust-700);}
.action-cell{font-weight:600;font-size:.84rem;}
.action-cell.t-scale{color:var(--sage-800);}.action-cell.t-test{color:var(--gold-700);}
.action-cell.t-hold{color:var(--steel-700);}.action-cell.t-reduce{color:var(--rust-700);}

/* Charts */
.chart-card{border:1px solid var(--line-200);border-radius:10px;background:#fff;padding:1.1rem 1.15rem .7rem;margin:1rem 0;}
.chart-card.cream{background:var(--cream-50);}
.chart-container{width:100%;}
.chart-caption,.caption{font-size:.84rem;color:var(--ink-400);max-width:78ch;margin-top:.4rem;}
.chart-grid-2{display:grid;grid-template-columns:1fr 1fr;gap:1rem;}
.sat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.85rem;margin-top:.5rem;}
.sat-cell{border:1px solid var(--line-200);border-radius:10px;background:var(--cream-50);padding:.5rem .5rem .3rem;}
.sat-key{display:flex;flex-wrap:wrap;gap:.4rem 1.4rem;margin:1rem 0 .25rem;padding:.85rem 1.1rem;background:var(--cream-100);border:1px solid var(--line-200);border-radius:10px;}
.sat-key .k{display:flex;align-items:center;gap:.5rem;font-size:.84rem;color:var(--ink-600);}
.sat-key .k b{color:var(--ink-900);font-weight:600;}
.sat-key .km{flex:none;width:13px;height:13px;display:inline-block;}
.sat-key .km.bt{border-radius:50%;background:#4a6d8a;}
.sat-key .km.op{background:#5a7a3a;clip-path:polygon(50% 0,61% 35%,98% 35%,68% 57%,79% 91%,50% 70%,21% 91%,32% 57%,2% 35%,39% 35%);}
.sat-key .km.sa{background:#a04535;border-radius:2px;}
.sat-key .km.cu{background:#b8860b;transform:rotate(45deg);border-radius:2px;width:11px;height:11px;}
.illus-tag{display:inline-flex;align-items:center;gap:.4rem;font-size:.68rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--gold-700);background:var(--gold-100);border:1px solid rgba(184,134,11,.28);border-radius:999px;padding:.18rem .6rem;margin-bottom:.9rem;}
.illus-tag::before{content:"";width:6px;height:6px;border-radius:50%;background:var(--gold-600);}

/* Reallocation cards */
.realloc-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:1rem;margin-top:.5rem;}
.realloc{border:1px solid var(--line-200);border-top:3px solid;border-radius:10px;padding:1.15rem 1.3rem;background:#fff;}
.realloc.t-scale{border-top-color:var(--sage-700);}
.realloc.t-test{border-top-color:var(--gold-600);}
.realloc.t-hold{border-top-color:var(--steel-600);}
.realloc.t-reduce{border-top-color:var(--rust-600);}
.realloc .rl-head{display:flex;align-items:center;justify-content:space-between;gap:.5rem;margin-bottom:.5rem;}
.realloc .rl-action{font-family:var(--font-display);font-size:1.15rem;font-weight:600;color:var(--ink-900);}
.realloc .rl-chs{font-size:.82rem;color:var(--ink-400);margin-bottom:.6rem;font-weight:500;}
.realloc p{font-size:.88rem;color:var(--ink-700);margin:0;max-width:none;}

/* Legend */
.legend{display:grid;grid-template-columns:repeat(2,1fr);gap:.85rem 1.5rem;margin:.5rem 0 1rem;}
.legend .row{display:flex;gap:.7rem;align-items:flex-start;}
.legend .sw{flex:none;width:14px;height:14px;border-radius:4px;margin-top:.15rem;}
.legend .sw.scale{background:var(--sage-700);}.legend .sw.test{background:var(--gold-600);}
.legend .sw.hold{background:var(--steel-600);}.legend .sw.reduce{background:var(--rust-600);}
.legend .lg-name{font-weight:600;color:var(--ink-900);font-size:.9rem;}
.legend .lg-desc{font-size:.84rem;color:var(--ink-600);}

/* Carryover */
.carry-stat{font-family:var(--font-display);font-size:1.9rem;font-weight:600;color:var(--ink-900);letter-spacing:-.02em;}
.carry-stat span{font-size:.95rem;color:var(--ink-400);font-family:var(--font-sans);font-weight:400;}

/* Deep dives */
.dd{border-top:1px solid var(--line-200);padding:2rem 0 .5rem;scroll-margin-top:1.25rem;}
.dd:first-of-type{border-top:none;padding-top:.5rem;}
.dd-head{display:flex;align-items:flex-start;justify-content:space-between;gap:1.25rem;flex-wrap:wrap;}
.dd-title{font-family:var(--font-display);font-size:1.45rem;font-weight:600;color:var(--ink-900);letter-spacing:-.01em;display:flex;align-items:center;gap:.6rem;}
.dd-title .dot{width:13px;height:13px;border-radius:3px;flex:none;}
.dd-read{color:var(--ink-600);font-size:.95rem;max-width:62ch;margin-top:.4rem;}
.dd-kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:.7rem;margin:1.25rem 0 1.1rem;}
.dd-kpi{background:var(--cream-100);border:1px solid var(--line-200);border-radius:8px;padding:.65rem .75rem;}
.dd-kpi .l{font-size:.62rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-400);margin-bottom:.35rem;line-height:1.25;}
.dd-kpi .v{font-family:var(--font-mono);font-size:1.18rem;font-weight:500;color:var(--ink-900);font-variant-numeric:tabular-nums;letter-spacing:-.02em;}
.dd-kpi .sub{font-size:.66rem;color:var(--ink-400);font-family:var(--font-mono);margin-top:.2rem;}
.dd-charts{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:.25rem;}
.dd-chart h5{font-size:.7rem;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:var(--ink-400);margin-bottom:.1rem;}
.dd-lower{display:grid;grid-template-columns:300px 1fr;gap:1.1rem;margin-top:1rem;align-items:stretch;}
.dd-adstock-box{border:1px solid var(--line-200);border-radius:10px;background:var(--cream-50);padding:.7rem .8rem .3rem;}
.dd-adstock-box h5{font-size:.7rem;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:var(--ink-400);margin-bottom:.1rem;}
.dd-rec{border-radius:10px;padding:1rem 1.2rem;border:1px solid;border-left-width:3px;}
.dd-rec.t-scale{background:var(--sage-100);border-color:rgba(90,122,58,.28);border-left-color:var(--sage-700);}
.dd-rec.t-test{background:var(--gold-100);border-color:rgba(184,134,11,.28);border-left-color:var(--gold-600);}
.dd-rec.t-hold{background:var(--steel-100);border-color:rgba(74,109,138,.25);border-left-color:var(--steel-600);}
.dd-rec.t-reduce{background:var(--rust-100);border-color:rgba(160,69,53,.25);border-left-color:var(--rust-600);}
.dd-rec h5{font-size:.7rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;margin-bottom:.5rem;}
.dd-rec.t-scale h5{color:var(--sage-800);}.dd-rec.t-test h5{color:var(--gold-700);}
.dd-rec.t-hold h5{color:var(--steel-700);}.dd-rec.t-reduce h5{color:var(--rust-700);}
.dd-rec p{font-size:.9rem;color:var(--ink-700);margin:0;max-width:none;}

/* Loop */
.loop{display:flex;flex-wrap:wrap;align-items:center;gap:.4rem;margin:.5rem 0 1.25rem;}
.loop .step{font-size:.82rem;padding:.4rem .8rem;border-radius:999px;border:1px solid var(--line-200);background:var(--cream-100);color:var(--ink-400);font-weight:500;}
.loop .step.current{background:var(--sage-700);border-color:var(--sage-700);color:#fff;}
.loop .arrow{color:var(--ink-300);font-family:var(--font-mono);}
.next-steps{list-style:none;display:flex;flex-direction:column;gap:.55rem;}
.next-steps li{display:flex;gap:.7rem;font-size:.92rem;color:var(--ink-700);align-items:flex-start;}
.next-steps .who{flex:none;font-size:.68rem;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:var(--ink-400);width:78px;padding-top:.18em;}

/* Tests */
.tests{display:flex;flex-direction:column;gap:.85rem;margin-top:.5rem;}
.test-item{display:flex;gap:1rem;padding:1rem 1.2rem;background:var(--cream-100);border:1px solid var(--line-200);border-radius:10px;}
.test-item .tnum{font-family:var(--font-mono);font-size:.95rem;font-weight:500;color:var(--sage-700);flex:none;padding-top:.1rem;}
.test-item .tbody b{display:block;color:var(--ink-900);font-size:.95rem;margin-bottom:.2rem;}
.test-item .tbody span{font-size:.86rem;color:var(--ink-600);}

/* Footer */
.report-footer{margin-top:1.5rem;padding:1.5rem 0 .5rem;color:var(--ink-400);font-size:.8rem;line-height:1.6;border-top:1px solid var(--line-200);}
.report-footer a{color:var(--sage-700);text-decoration:none;}
.report-footer a:hover{text-decoration:underline;}

@media (max-width:900px){
  .report-nav{display:none;}
  .kpi-grid,.realloc-grid,.legend,.chart-grid-2,.dd-charts{grid-template-columns:1fr;}
  .sat-grid{grid-template-columns:1fr 1fr;}
  .dd-kpis{grid-template-columns:repeat(2,1fr);}
  .dd-lower{grid-template-columns:1fr;}
}

@page{margin:14mm 13mm;}
@media print{
  body{background:#fff;font-size:10pt;color:var(--ink-900);}
  .report-nav{display:none !important;}
  .report-body{display:block;max-width:none;}
  .report-container{max-width:none;padding:0;}
  .report-header{border-radius:0;-webkit-print-color-adjust:exact;print-color-adjust:exact;}
  .section{break-inside:avoid;box-shadow:none;border:none;border-top:1px solid var(--line-200);border-radius:0;padding:1.1rem 0;}
  .section>h2::after{display:none;}
  .kpi,.rec,.realloc,.tier-chip,.data-table thead tr,.test-item,.loop .step,.chart-card,.sat-cell,.dd-kpi,.dd-rec,.dd-adstock-box{-webkit-print-color-adjust:exact;print-color-adjust:exact;}
  .realloc-grid,.dd,.dd-charts,.chart-card{break-inside:avoid;}
  .report-footer{page-break-inside:avoid;}
}"""


def augur_css(color_scheme) -> str:
    """Augur stylesheet + a ``--color-*`` compatibility block from the scheme."""
    c = color_scheme
    compat = (
        ":root{"
        f"--color-primary:{c.primary};--color-primary-dark:{c.primary_dark};"
        f"--color-accent:{c.accent};--color-accent-dark:{c.accent_dark};"
        f"--color-warning:{c.warning};--color-danger:{c.danger};"
        f"--color-success:{c.success};--color-text:{c.text};"
        f"--color-text-muted:{c.text_muted};--color-bg:{c.background};"
        f"--color-bg-alt:{c.background_alt};--color-surface:{c.surface};"
        f"--color-border:{c.border};"
        "--shadow-sm:0 1px 2px 0 rgb(42 53 40 / .05);"
        "}"
    )
    return AUGUR_COMPONENT_CSS + "\n" + compat


__all__ = ["AUGUR_COMPONENT_CSS", "augur_css", "MASTHEAD_LOGO_SVG", "AUGUR_FONTS_LINK"]
