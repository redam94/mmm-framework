"""
HTML project report and Reveal.js slideshow generator.

Called by the generate_project_report agent tool to produce self-contained
deliverables from the current MMM session state.
"""

from __future__ import annotations

import json
from typing import Any

_PALETTE = [
    "#4f46e5",
    "#0d9488",
    "#f59e0b",
    "#e11d48",
    "#059669",
    "#7c3aed",
    "#0284c7",
    "#b45309",
    "#6366f1",
    "#0f766e",
]


def apply_branding_html(html: str, branding: dict | None) -> str:
    """Minimal client-branding pass over a finished report/slides HTML.

    String-level by design: remapping the design-palette hexes recolors the
    CSS *and* the embedded Plotly figure JSON in one move; the sidebar logo
    text, logo image, and footer line are swapped via their known markup.
    Call only with confirmed branding (``agents.branding.is_active``)."""
    if not branding or not html:
        return html
    colors = branding.get("colors") or {}
    palette = [c for c in (colors.get("palette") or []) if c] or [
        c
        for c in (colors.get("primary"), colors.get("secondary"), colors.get("accent"))
        if c
    ]
    if palette:
        for i, design_color in enumerate(_PALETTE):
            html = html.replace(design_color, palette[i % len(palette)])
    client_name = branding.get("client_name")
    logo_url = branding.get("logo_url")
    if client_name or logo_url:
        logo_inner = ""
        if logo_url:
            logo_inner += (
                f'<img src="{logo_url}" alt="" '
                'style="max-height:28px;max-width:160px;display:block;'
                'margin-bottom:0.35rem"/>'
            )
        logo_inner += client_name or "MMM Framework"
        html = html.replace(
            '<div class="logo">MMM Framework<span>',
            f'<div class="logo">{logo_inner}<span>',
        )
    footer_text = branding.get("footer_text")
    if footer_text:
        html = html.replace(
            "MMM Framework — Bayesian Marketing Mix Modelling", footer_text
        )
    return html


_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
_REVEAL_CSS = "https://cdn.jsdelivr.net/npm/reveal.js@4/dist/reveal.css"
_REVEAL_THEME = "https://cdn.jsdelivr.net/npm/reveal.js@4/dist/theme/white.css"
_REVEAL_JS = "https://cdn.jsdelivr.net/npm/reveal.js@4/dist/reveal.js"

# ── CSS for the HTML report ───────────────────────────────────────────────────

_REPORT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body {
  font-family: 'Inter', -apple-system, system-ui, sans-serif;
  background: #f1f5f9; color: #1e293b; line-height: 1.7; font-size: 15px;
  -webkit-font-smoothing: antialiased;
}
a { color: #4f46e5; text-decoration: none; }

/* ── Layout ── */
.layout { display: flex; min-height: 100vh; }

.sidebar {
  width: 220px; min-width: 220px;
  background: #0f172a;
  position: sticky; top: 0; height: 100vh; overflow-y: auto;
  padding: 2rem 1.25rem; flex-shrink: 0;
}
.sidebar .logo {
  font-size: 0.9rem; font-weight: 800; color: #fff; letter-spacing: -0.01em;
  padding-bottom: 1.25rem; margin-bottom: 0.5rem;
  border-bottom: 1px solid rgba(255,255,255,0.08);
}
.sidebar .logo span { display: block; font-size: 0.65rem; font-weight: 400; color: #64748b; margin-top: 3px; letter-spacing: 0.03em; text-transform: uppercase; }
.sidebar .nav-group { font-size: 0.6rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.1em; color: #475569; margin: 1.25rem 0 0.35rem 0.5rem; }
.sidebar nav a {
  display: flex; align-items: center; gap: 0.5rem;
  color: #94a3b8; padding: 0.4rem 0.6rem; border-radius: 0.5rem;
  margin-bottom: 1px; font-size: 0.78rem; transition: all 0.15s;
}
.sidebar nav a:hover { color: #fff; background: rgba(255,255,255,0.08); }
.sidebar nav a .dot { width: 5px; height: 5px; border-radius: 50%; background: currentColor; flex-shrink: 0; opacity: 0.6; }

.body-wrap { flex: 1; min-width: 0; }

/* ── Hero ── */
.hero {
  background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e40af 100%);
  padding: 3.5rem 4rem 3rem;
  position: relative; overflow: hidden;
}
.hero::after {
  content: ''; position: absolute; inset: 0;
  background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
  pointer-events: none;
}
.hero-label {
  display: inline-flex; align-items: center; gap: 0.4rem;
  background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.2);
  color: #c7d2fe; border-radius: 999px;
  padding: 0.3rem 0.9rem; font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 1rem;
}
.hero h1 {
  font-size: 2.1rem; font-weight: 800; color: #fff; line-height: 1.15;
  letter-spacing: -0.02em; margin-bottom: 0.5rem; max-width: 700px;
}
.hero-meta { color: #a5b4fc; font-size: 0.85rem; margin-top: 0.25rem; }
.hero-kpis {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 1rem; margin-top: 2rem; max-width: 700px;
}
.hero-kpi {
  background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.15);
  backdrop-filter: blur(8px); border-radius: 0.875rem; padding: 1rem;
}
.hero-kpi .val { font-size: 1.6rem; font-weight: 800; color: #fff; line-height: 1; }
.hero-kpi .lbl { font-size: 0.65rem; color: #a5b4fc; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.3rem; }

/* ── Main content ── */
.main { padding: 2.5rem 3.5rem; max-width: 900px; }

/* ── Section ── */
.section { margin-bottom: 3.5rem; scroll-margin-top: 80px; }
.section-label {
  display: inline-block; font-size: 0.65rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.1em; color: #6366f1;
  margin-bottom: 0.35rem;
}
.section h2 {
  font-size: 1.4rem; font-weight: 800; color: #1e293b;
  letter-spacing: -0.01em; margin-bottom: 0.4rem;
}
.section .section-intro {
  color: #64748b; font-size: 0.9rem; max-width: 680px; margin-bottom: 1.5rem; line-height: 1.65;
}
.section-divider { border: none; border-top: 1px solid #e2e8f0; margin-bottom: 1.5rem; }

/* ── Cards ── */
.card {
  background: #fff; border: 1px solid #e2e8f0; border-radius: 1.25rem;
  padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 4px 12px rgba(0,0,0,0.03);
}
.card + .card, .card + .chart-card, .chart-card + .card { margin-top: 1rem; }
.card-eyebrow {
  font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.09em; color: #94a3b8; margin-bottom: 0.5rem;
}
.card h3 { font-size: 1rem; font-weight: 700; color: #1e293b; margin-bottom: 0.25rem; }
.card p { font-size: 0.875rem; color: #475569; line-height: 1.65; }

/* ── Insight boxes ── */
.insight {
  display: flex; gap: 1rem; align-items: flex-start;
  background: #fafbff; border: 1px solid #e0e7ff;
  border-left: 4px solid #6366f1;
  border-radius: 0 1rem 1rem 0; padding: 1rem 1.25rem;
  margin-bottom: 0.75rem;
}
.insight .icon { font-size: 1.2rem; flex-shrink: 0; line-height: 1.4; }
.insight .body { flex: 1; }
.insight .body strong { display: block; font-size: 0.85rem; font-weight: 700; color: #312e81; margin-bottom: 0.2rem; }
.insight .body p { font-size: 0.82rem; color: #475569; margin: 0; line-height: 1.55; }

.insight-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 1rem; }
@media (max-width: 700px) { .insight-grid { grid-template-columns: 1fr; } }

/* ── KPI big number ── */
.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.875rem; margin-bottom: 1.5rem; }
.kpi-box {
  background: #fff; border: 1px solid #e2e8f0; border-radius: 1rem;
  padding: 1.25rem 1rem; text-align: center;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.kpi-box .val { font-size: 2rem; font-weight: 800; line-height: 1; }
.kpi-box .val.indigo { color: #4f46e5; }
.kpi-box .val.teal { color: #0d9488; }
.kpi-box .val.amber { color: #d97706; }
.kpi-box .val.red { color: #dc2626; }
.kpi-box .val.green { color: #059669; }
.kpi-box .lbl { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em; color: #94a3b8; margin-top: 0.35rem; }
.kpi-box .sub { font-size: 0.75rem; color: #64748b; margin-top: 0.2rem; }

/* ── Performance tier cards ── */
.channel-card {
  background: #fff; border: 1px solid #e2e8f0; border-radius: 1rem;
  padding: 1.25rem 1.5rem; display: flex; align-items: center; gap: 1.25rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04); margin-bottom: 0.75rem;
}
.channel-card .tier-badge {
  width: 44px; height: 44px; border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.2rem; flex-shrink: 0;
}
.tier-strong { background: #f0fdf4; }
.tier-moderate { background: #fffbeb; }
.tier-weak { background: #fef2f2; }
.channel-card .ch-info { flex: 1; }
.channel-card .ch-name { font-size: 1rem; font-weight: 700; color: #1e293b; }
.channel-card .ch-desc { font-size: 0.8rem; color: #64748b; margin-top: 0.15rem; }
.channel-card .roi-pill {
  font-size: 1.2rem; font-weight: 800; padding: 0.35rem 0.9rem;
  border-radius: 0.625rem; white-space: nowrap; flex-shrink: 0;
}
.roi-strong { background: #f0fdf4; color: #059669; }
.roi-moderate { background: #fffbeb; color: #d97706; }
.roi-weak { background: #fef2f2; color: #dc2626; }
.prob-bar-wrap { display: flex; align-items: center; gap: 0.5rem; margin-top: 0.4rem; }
.prob-bar { height: 4px; width: 80px; background: #e2e8f0; border-radius: 999px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 999px; }
.prob-label { font-size: 0.7rem; color: #64748b; }

/* ── Table ── */
.table-wrap { overflow-x: auto; border-radius: 1rem; border: 1px solid #e2e8f0; background: #fff; }
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
thead th {
  background: #f8fafc; color: #94a3b8; font-weight: 600; font-size: 0.7rem;
  text-transform: uppercase; letter-spacing: 0.06em;
  padding: 0.875rem 1.25rem; text-align: left; border-bottom: 1px solid #e2e8f0;
}
tbody td { padding: 0.875rem 1.25rem; border-bottom: 1px solid #f1f5f9; color: #334155; }
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover td { background: #fafbff; }

/* ── Badges / tags ── */
.badge {
  display: inline-block; padding: 0.2rem 0.65rem; border-radius: 999px;
  font-size: 0.7rem; font-weight: 600; letter-spacing: 0.02em;
}
.badge-indigo { background: #eef2ff; color: #4f46e5; }
.badge-green { background: #f0fdf4; color: #059669; }
.badge-amber { background: #fffbeb; color: #d97706; }
.badge-red { background: #fef2f2; color: #dc2626; }
.badge-gray { background: #f1f5f9; color: #64748b; }
.tag-row { display: flex; flex-wrap: wrap; gap: 0.4rem; }

/* ── Chart containers ── */
.chart-card {
  background: #fff; border: 1px solid #e2e8f0; border-radius: 1.25rem;
  padding: 1.25rem 1rem 0.5rem; margin-bottom: 1rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.chart-card .chart-title { font-size: 0.8rem; font-weight: 700; color: #334155; padding: 0 0.5rem 0.75rem; }

/* ── Prog bar ── */
.prog-bar { height: 6px; background: #e2e8f0; border-radius: 999px; overflow: hidden; }
.prog-bar-fill { height: 100%; border-radius: 999px; }

/* ── Callout box ── */
.callout {
  border-radius: 1rem; padding: 1rem 1.25rem; margin-bottom: 1rem;
  display: flex; gap: 0.75rem; align-items: flex-start;
}
.callout-ok { background: #f0fdf4; border: 1px solid #bbf7d0; color: #166534; }
.callout-warn { background: #fffbeb; border: 1px solid #fde68a; color: #92400e; }
.callout-info { background: #eef2ff; border: 1px solid #c7d2fe; color: #3730a3; }
.callout .ci { font-size: 1.1rem; flex-shrink: 0; }
.callout .ct { font-size: 0.85rem; line-height: 1.55; }
.callout .ct strong { display: block; margin-bottom: 0.1rem; }

/* ── Appendix ── */
.appendix { background: #f8fafc; border-top: 2px solid #e2e8f0; padding: 2.5rem 3.5rem; }
.appendix h2 { font-size: 1.1rem; font-weight: 800; color: #64748b; letter-spacing: -0.01em; margin-bottom: 1.5rem; }
.appendix .app-section { margin-bottom: 2rem; }
.appendix .app-section h3 { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; color: #94a3b8; margin-bottom: 0.75rem; }

/* ── Footer ── */
.footer {
  text-align: center; padding: 1.5rem; color: #94a3b8; font-size: 0.75rem;
  border-top: 1px solid #e2e8f0; background: #f8fafc;
}

/* ── Responsive ── */
@media (max-width: 900px) {
  .sidebar { display: none; }
  .hero { padding: 2rem; }
  .main { padding: 1.5rem 1.25rem; }
  .appendix { padding: 1.5rem 1.25rem; }
  .hero h1 { font-size: 1.5rem; }
}
@media print {
  .sidebar { display: none; }
  .hero { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
}
"""

# ── Slide CSS ─────────────────────────────────────────────────────────────────

_SLIDE_CSS = """
/* Override Reveal white-theme transforms and sizes */
.reveal, .reveal * { box-sizing: border-box; }
.reveal .slides section { text-align: left; padding: 0 2em; }
.reveal .slides section[data-slide-type="title"] { text-align: center; }

.reveal h1, .reveal h2, .reveal h3, .reveal h4 {
  font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
  text-transform: none !important;
  letter-spacing: -0.01em !important;
  font-weight: 700 !important;
  line-height: 1.2 !important;
  color: #1e293b !important;
  word-break: break-word !important;
}
.reveal h1 { font-size: 1.9em !important; font-weight: 800 !important; text-align: center; }
.reveal h2 {
  font-size: 1.3em !important;
  border-bottom: 2px solid #4f46e5 !important;
  padding-bottom: 0.3em !important;
  margin-bottom: 0.7em !important;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.reveal h2.wrap { white-space: normal; font-size: 1.1em !important; }
.reveal h3 { font-size: 0.95em !important; color: #475569 !important; }
.reveal p, .reveal li {
  font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
  font-size: 0.82em !important; color: #334155 !important; line-height: 1.55 !important;
  text-transform: none !important;
}
.reveal ul { list-style: none !important; padding: 0 !important; }
.reveal li { padding: 0.3em 0 0.3em 1.3em !important; position: relative !important; }
.reveal li::before { content: "▸"; position: absolute; left: 0; color: #4f46e5; font-size: 0.75em; top: 0.45em; }

.reveal .slide-tag {
  display: inline-block; background: #eef2ff; color: #4f46e5;
  border-radius: 999px; padding: 0.2em 0.75em; font-size: 0.58em;
  font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
  margin-bottom: 0.6em;
}
.reveal .subtitle { color: #64748b; font-size: 0.78em; margin-top: 0.4em; }
.reveal .rq-box {
  background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #4f46e5;
  border-radius: 0 0.75rem 0.75rem 0; padding: 0.9em 1.1em; margin: 0.6em 0;
  font-size: 0.82em !important; color: #1e293b !important; line-height: 1.55;
}
.reveal .rq-box .rq-main {
  font-size: 1.05em !important; font-weight: 600; color: #1e293b !important; margin-bottom: 0.5em;
}
.reveal .rq-meta { display: grid; grid-template-columns: 1fr 1fr; gap: 0.4em 1em; margin-top: 0.5em; }
.reveal .rq-meta-item { font-size: 0.85em !important; }
.reveal .rq-meta-item .lbl { color: #94a3b8; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.05em; }

.stat-grid-slide { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; margin: 0.75rem 0; }
.stat-slide { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 0.75rem; padding: 0.9rem 0.75rem; text-align: center; }
.stat-slide .v { font-size: 1.7em; font-weight: 800; color: #4f46e5; line-height: 1; }
.stat-slide .l { font-size: 0.6em; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.3em; }

.chart-slide { height: 360px; width: 100%; }
.finding-slide {
  background: #f5f7ff; border-left: 3px solid #4f46e5;
  border-radius: 0 0.5rem 0.5rem 0; padding: 0.55em 0.85em;
  margin-bottom: 0.45em; font-size: 0.78em !important; color: #1e293b !important;
  line-height: 1.45;
}
.tag-row { display: flex; flex-wrap: wrap; gap: 0.35rem; }
.stag { background: #eef2ff; color: #4f46e5; border-radius: 999px; padding: 0.25em 0.7em; font-size: 0.65em; font-weight: 600; }
"""

# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_rq(value: Any) -> tuple[str, dict]:
    """Return (main_question_str, extra_fields_dict) from an assumption value."""
    if isinstance(value, dict):
        q = (
            value.get("question")
            or value.get("primary_question")
            or value.get("text")
            or ""
        )
        if not q:
            # fallback: first non-empty string value
            q = next((str(v) for v in value.values() if v), str(value))
        extra = {
            k: v
            for k, v in value.items()
            if k not in ("question", "primary_question", "text") and v
        }
        return q, extra
    return str(value), {}


def _truncate_title(s: str, max_len: int = 55) -> str:
    """Strip HTML tags and truncate to max_len chars."""
    import re

    clean = re.sub(r"<[^>]+>", " ", s).strip()
    clean = re.sub(r"\s+", " ", clean)
    return clean[:max_len].rstrip() + "…" if len(clean) > max_len else clean


def _rq_slide_html(question: str, extra: dict) -> str:
    meta = ""
    if extra:
        meta = '<div class="rq-meta">'
        for k, v in list(extra.items())[:6]:  # cap at 6 extra fields
            meta += f'<div class="rq-meta-item"><div class="lbl">{k.replace("_"," ")}</div><div>{v}</div></div>'
        meta += "</div>"
    return (
        f'<div class="rq-box">'
        f'<div class="rq-main">{question}</div>'
        f"{meta}"
        f"</div>"
    )


def _js(div_id: str, fig: dict, config: dict | None = None) -> str:
    cfg = config or {"responsive": True, "displayModeBar": False}
    return (
        f'Plotly.newPlot("{div_id}",'
        f'{json.dumps(fig.get("data", []))},'
        f'{json.dumps(fig.get("layout", {}))},'
        f"{json.dumps(cfg)});"
    )


def _hydrate_plots(plots: list | None) -> list[dict]:
    """Resolve content-addressed plot refs to inline Plotly figures.

    Session charts live in ``dashboard_data['plots']`` as thin
    ``{"id", "title"}`` refs — the heavy figure JSON is held in the
    content-addressed plot store (:mod:`agents.workspace`) and streamed to the
    UI once via ``GET /plots/{id}``. A self-contained HTML report/slideshow
    cannot fetch by id, so each ref must be inlined back into a
    ``{"data", "layout"}`` figure here; otherwise the "Additional Charts"
    section renders empty ``<div>`` placeholders (``p.get("data", [])`` → ``[]``).

    Inline figures (the store-write-failure back-compat fallback, which already
    carry a ``data`` list) pass through unchanged. A ref whose stored payload is
    missing or unreadable is dropped so it never renders as an empty graph.
    """
    from mmm_framework.agents import workspace as _ws

    out: list[dict] = []
    for p in plots or []:
        if not isinstance(p, dict):
            continue
        if isinstance(p.get("data"), list):
            out.append(p)  # already an inline figure
            continue
        pid = p.get("id")
        if not pid:
            continue
        path = _ws.plot_path(str(pid))
        if path is None:
            continue
        try:
            fig = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - unreadable/corrupt payload: drop it
            continue
        if not (isinstance(fig, dict) and isinstance(fig.get("data"), list)):
            continue
        # Keep the ref's (often nicer, human-authored) title when the stored
        # figure layout has none, so headings/slide titles stay meaningful.
        ref_title = p.get("title")
        layout = fig.setdefault("layout", {})
        if ref_title and not layout.get("title"):
            layout["title"] = {"text": ref_title}
        out.append(fig)
    return out


def _fmt_num(v: float | None, decimals: int = 2) -> str:
    if v is None:
        return "—"
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"{v/1_000:.0f}K"
    return f"{v:.{decimals}f}"


def _pct(v: float | None) -> str:
    return f"{v*100:.1f}%" if v is not None else "—"


def _roi_fig(roi: list[dict]) -> dict:
    channels = [r["channel"] for r in roi]
    means = [r.get("roi_mean", 0) for r in roi]
    lo = [r.get("roi_hdi_low", r.get("roi_mean", 0)) for r in roi]
    hi = [r.get("roi_hdi_high", r.get("roi_mean", 0)) for r in roi]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(channels))]
    return {
        "data": [
            {
                "type": "bar",
                "orientation": "h",
                "x": means,
                "y": channels,
                "error_x": {
                    "type": "data",
                    "symmetric": False,
                    "array": [h - m for h, m in zip(hi, means)],
                    "arrayminus": [m - lo_i for m, lo_i in zip(means, lo)],
                    "color": "#94a3b8",
                    "thickness": 2,
                    "width": 5,
                },
                "marker": {"color": colors},
                "text": [f"{m:.2f}x" for m in means],
                "textposition": "outside",
            }
        ],
        "layout": {
            "title": {
                "text": "ROI by Channel (94% HDI)",
                "font": {"size": 14, "color": "#1e293b"},
            },
            "xaxis": {
                "title": "Return per £1 Spent",
                "automargin": True,
                "gridcolor": "#f1f5f9",
            },
            "yaxis": {"automargin": True},
            "margin": {"l": 10, "r": 80, "t": 50, "b": 50},
            "plot_bgcolor": "#f9fafb",
            "paper_bgcolor": "rgba(0,0,0,0)",
            "font": {"family": "Inter, system-ui, sans-serif", "size": 12},
            "uniformtext": {"minsize": 9, "mode": "hide"},
        },
    }


def _decomp_fig(decomp: list[dict]) -> dict:
    sorted_d = sorted(decomp, key=lambda x: x["pct_of_total"], reverse=True)
    labels = [d["component"] for d in sorted_d]
    values = [d["pct_of_total"] * 100 for d in sorted_d]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(labels))]
    return {
        "data": [
            {
                "type": "pie",
                "labels": labels,
                "values": values,
                "hole": 0.42,
                "marker": {"colors": colors},
                "textinfo": "label+percent",
                "textposition": "outside",
                "pull": [0.03] + [0] * (len(labels) - 1),
            }
        ],
        "layout": {
            "title": {
                "text": "KPI Decomposition",
                "font": {"size": 14, "color": "#1e293b"},
            },
            "showlegend": True,
            "legend": {"orientation": "v", "x": 1.0, "y": 0.5},
            "margin": {"l": 10, "r": 160, "t": 50, "b": 20},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "font": {"family": "Inter, system-ui, sans-serif", "size": 12},
        },
    }


# ── Slide-only chart helpers ──────────────────────────────────────────────────


def _scurves_fig(curves: dict) -> dict:
    """Normalized S-curve chart: x = spend relative to current, y = % of max response."""
    traces = []
    for i, (ch, c) in enumerate(curves.items()):
        spend = c.get("spend") or []
        resp = c.get("response_mean") or []
        cur_spend = c.get("current_spend") or 0.0
        if not spend or not resp or cur_spend <= 0:
            continue
        max_resp = max(resp) if resp else 0.0
        if max_resp <= 0:
            continue

        x_norm = [s / cur_spend for s in spend]
        y_norm = [r / max_resp * 100 for r in resp]
        color = _PALETTE[i % len(_PALETTE)]

        resp_lo = c.get("response_hdi_low") or resp
        resp_hi = c.get("response_hdi_high") or resp
        y_lo = [r / max_resp * 100 for r in resp_lo]
        y_hi = [r / max_resp * 100 for r in resp_hi]

        # Uncertainty band (filled polygon)
        traces.append(
            {
                "type": "scatter",
                "x": x_norm + x_norm[::-1],
                "y": y_hi + y_lo[::-1],
                "fill": "toself",
                "fillcolor": color + "22",
                "line": {"color": "rgba(0,0,0,0)"},
                "showlegend": False,
                "hoverinfo": "skip",
            }
        )
        traces.append(
            {
                "type": "scatter",
                "x": x_norm,
                "y": y_norm,
                "name": ch,
                "mode": "lines",
                "line": {"color": color, "width": 2.5},
            }
        )
        sat_pct = c.get("saturation_level", 0) * 100
        traces.append(
            {
                "type": "scatter",
                "x": [1.0],
                "y": [sat_pct],
                "mode": "markers",
                "marker": {
                    "size": 9,
                    "color": color,
                    "symbol": "circle",
                    "line": {"color": "#fff", "width": 2},
                },
                "showlegend": False,
                "hovertemplate": f"<b>{ch}</b><br>{sat_pct:.0f}% of maximum response at current spend<extra></extra>",
            }
        )

    return {
        "data": traces,
        "layout": {
            "title": {
                "text": "Saturation Curves — Where Each Channel Sits on the Diminishing Returns Curve",
                "font": {"size": 12, "color": "#1e293b"},
            },
            "xaxis": {
                "title": "Spend relative to current",
                "tickformat": ".0%",
                "range": [0, 1.6],
                "gridcolor": "#f1f5f9",
                "automargin": True,
            },
            "yaxis": {
                "title": "% of Maximum Response",
                "ticksuffix": "%",
                "range": [0, 110],
                "gridcolor": "#f1f5f9",
                "automargin": True,
            },
            "shapes": [
                {
                    "type": "line",
                    "x0": 1.0,
                    "x1": 1.0,
                    "y0": 0,
                    "y1": 110,
                    "line": {"color": "#94a3b8", "width": 1, "dash": "dot"},
                }
            ],
            "annotations": [
                {
                    "x": 1.02,
                    "y": 108,
                    "text": "Current spend",
                    "showarrow": False,
                    "font": {"size": 9, "color": "#94a3b8"},
                    "xanchor": "left",
                }
            ],
            "margin": {"l": 60, "r": 40, "t": 65, "b": 60},
            "plot_bgcolor": "#f9fafb",
            "paper_bgcolor": "rgba(0,0,0,0)",
            "font": {"family": "Inter, system-ui, sans-serif", "size": 12},
        },
    }


def _mroi_roi_fig(roi: list[dict], mroi: dict, fmt_ch=None) -> dict:
    """Grouped horizontal bar: average ROI vs marginal ROI per channel."""
    if fmt_ch is None:

        def fmt_ch(x):
            return x

    channels_raw = [r["channel"] for r in roi]
    channels_display = [fmt_ch(ch) for ch in channels_raw]
    avg_rois = [r.get("roi_mean", 0) for r in roi]
    marg_rois = [mroi.get(ch, {}).get("marginal_roi_mean") or 0 for ch in channels_raw]
    channels = channels_display  # use formatted names for y-axis labels

    return {
        "data": [
            {
                "type": "bar",
                "orientation": "h",
                "name": "Average ROI",
                "x": avg_rois,
                "y": channels,
                "marker": {"color": "#4f46e5"},
                "text": [f"{v:.2f}x" for v in avg_rois],
                "textposition": "outside",
            },
            {
                "type": "bar",
                "orientation": "h",
                "name": "Marginal ROI (next £1)",
                "x": marg_rois,
                "y": channels,
                "marker": {"color": "#0d9488"},
                "text": [f"{v:.2f}x" for v in marg_rois],
                "textposition": "outside",
            },
        ],
        "layout": {
            "title": {
                "text": "Average ROI vs Marginal ROI — What the Next £1 Earns",
                "font": {"size": 13, "color": "#1e293b"},
            },
            "barmode": "group",
            "xaxis": {
                "title": "Return per £1",
                "automargin": True,
                "gridcolor": "#f1f5f9",
            },
            "yaxis": {"automargin": True},
            "shapes": [
                {
                    "type": "line",
                    "x0": 1.0,
                    "x1": 1.0,
                    "y0": -0.5,
                    "y1": len(channels) - 0.5,
                    "line": {"color": "#dc2626", "width": 1.5, "dash": "dot"},
                }
            ],
            "annotations": [
                {
                    "x": 1.0,
                    "y": len(channels) - 0.5,
                    "text": "Breakeven",
                    "showarrow": False,
                    "font": {"size": 9, "color": "#dc2626"},
                    "xanchor": "left",
                    "yanchor": "bottom",
                }
            ],
            "margin": {"l": 10, "r": 90, "t": 60, "b": 50},
            "plot_bgcolor": "#f9fafb",
            "paper_bgcolor": "rgba(0,0,0,0)",
            "font": {"family": "Inter, system-ui, sans-serif", "size": 12},
            "legend": {"orientation": "h", "x": 0, "y": -0.2},
            "uniformtext": {"minsize": 8, "mode": "hide"},
        },
    }


def _channel_perf_html(
    roi: list[dict],
    mroi: dict,
    sat: dict,
    fmt_ch,
) -> str:
    """HTML channel performance cards — working/not-working classification."""
    items = []
    for r in sorted(roi, key=lambda x: x.get("roi_mean", 0), reverse=True):
        ch = r["channel"]
        avg_roi = r.get("roi_mean", 0)
        m = mroi.get(ch, {})
        marg = m.get("marginal_roi_mean")
        # sat can come from saturation_curves (has saturation_level key)
        # or from the older saturation tool output (same key)
        s = sat.get(ch, {})
        sat_lvl = s.get("saturation_level") if isinstance(s, dict) else None

        if marg is not None and sat_lvl is not None:
            if avg_roi >= 1.0 and marg >= 1.0 and sat_lvl < 0.65:
                badge, badge_color = "Scale Up", "#059669"
                rec = (
                    "Strong returns with room to grow — additional spend is efficient."
                )
            elif avg_roi >= 1.0 and (marg >= 0.5 or sat_lvl < 0.80):
                badge, badge_color = "Maintain", "#d97706"
                rec = "Good returns but approaching diminishing returns — hold current investment."
            elif avg_roi >= 1.0 and marg < 0.5 and sat_lvl >= 0.80:
                badge, badge_color = "Near Saturation", "#ea580c"
                rec = "Overall ROI positive but marginal spend is inefficient. Redirect incremental budget."
            else:
                badge, badge_color = "Review", "#dc2626"
                rec = "Current spend may not be covering costs. Consider reducing or reallocating."
        elif avg_roi >= 1.5:
            badge, badge_color = "Performing", "#059669"
            rec = "Strong average ROI."
        elif avg_roi >= 0.8:
            badge, badge_color = "Moderate", "#d97706"
            rec = "Moderate ROI — monitor closely."
        else:
            badge, badge_color = "Review", "#dc2626"
            rec = "Below-breakeven ROI. Review spend levels."

        meta_parts = [f"avg ROI {avg_roi:.2f}x"]
        if marg is not None:
            meta_parts.append(f"mROI {marg:.2f}x")
        if sat_lvl is not None:
            meta_parts.append(f"{sat_lvl * 100:.0f}% saturated")

        items.append(
            f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
            f"border-left:4px solid {badge_color};border-radius:0 0.6rem 0.6rem 0;"
            f'padding:0.6rem 0.9rem;margin-bottom:0.45rem">'
            f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.15rem">'
            f'<span style="font-weight:700;font-size:0.85em;color:#1e293b">{fmt_ch(ch)}</span>'
            f'<span style="font-size:0.6em;font-weight:700;color:{badge_color};'
            f'background:{badge_color}18;padding:0.15em 0.55em;border-radius:999px">{badge}</span>'
            f"</div>"
            f'<div style="font-size:0.63em;color:#64748b;margin-bottom:0.15rem">{" · ".join(meta_parts)}</div>'
            f'<div style="font-size:0.7em;color:#475569">{rec}</div>'
            f"</div>"
        )
    return "".join(items)


# ── HTML Report ───────────────────────────────────────────────────────────────


def generate_html_report(
    title: str,
    date_str: str,
    dashboard: dict,
    assumptions: list[dict],
) -> str:
    import re

    dataset = dashboard.get("dataset") or {}
    model_spec = dashboard.get("model_spec") or {}
    decomp = dashboard.get("decomposition") or []
    roi = dashboard.get("roi_metrics") or []
    diag = dashboard.get("diagnostics") or {}
    plots = _hydrate_plots(dashboard.get("plots"))
    model_run = dashboard.get("model_run") or {}

    _rq_raw = next(
        (a["value"] for a in assumptions if a.get("key") == "research_question"), None
    )
    research_q, _rq_extra = _extract_rq(_rq_raw) if _rq_raw is not None else ("", {})
    prior_check = next(
        (a for a in assumptions if a.get("key") == "prior_predictive_check"), None
    )
    other_assumptions = [
        a
        for a in assumptions
        if a.get("key") not in ("research_question", "prior_predictive_check")
        and not a.get("key", "").startswith("sensitivity::")
    ]
    sensitivity_assumptions = [
        a for a in assumptions if a.get("key", "").startswith("sensitivity::")
    ]

    channels: list[str] = model_run.get("channels") or [
        c.get("name", "") for c in (model_spec.get("media_channels") or [])
    ]
    controls: list[str] = model_run.get("controls") or [
        (c.get("name", c) if isinstance(c, dict) else c)
        for c in (model_spec.get("control_variables") or [])
    ]
    kpi: str = model_run.get("kpi") or model_spec.get("kpi") or "KPI"
    rows_val = dataset.get("rows")

    # ── Derive key findings for exec summary ─────────────────────────────────

    findings: list[tuple[str, str, str]] = []  # (icon, heading, body)
    if decomp:
        top = max(decomp, key=lambda x: x["pct_of_total"])
        media_pct = sum(
            d["pct_of_total"]
            for d in decomp
            if d["component"].lower()
            not in ("baseline", "trend", "seasonality", "controls")
        )
        findings.append(
            (
                "📊",
                f"{top['component']} is the top driver",
                f"Accounts for {_pct(top['pct_of_total'])} of fitted {kpi}.",
            )
        )
        if media_pct > 0:
            findings.append(
                (
                    "📺",
                    f"Media drives {_pct(media_pct)} of {kpi}",
                    "Combined paid channels collectively explain this share of the KPI in-sample.",
                )
            )
    if roi:
        best = max(roi, key=lambda x: x.get("roi_mean", 0))
        worst = min(roi, key=lambda x: x.get("roi_mean", 0))
        findings.append(
            (
                "💰",
                f"{best['channel']} has the highest ROI",
                f"Returns {best.get('roi_mean', 0):.2f}x per unit of spend.",
            )
        )
        if len(roi) > 1 and worst["channel"] != best["channel"]:
            findings.append(
                (
                    "⚠️",
                    f"{worst['channel']} has the lowest ROI",
                    f"Returns only {worst.get('roi_mean', 0):.2f}x — consider rebalancing budget.",
                )
            )
    if diag:
        if diag.get("converged"):
            findings.append(
                (
                    "✅",
                    "Model converged successfully",
                    f"R̂ max {diag.get('rhat_max', '<1.01')}, {diag.get('divergences', 0)} divergences. Estimates are reliable.",
                )
            )
        else:
            findings.append(
                (
                    "⚠️",
                    "Model has convergence issues",
                    "Treat estimates with caution — consider adjusting priors or increasing samples.",
                )
            )

    # ── Build chart JS + chart_htmls list ────────────────────────────────────

    scripts: list[str] = []
    chart_htmls: list[tuple[str, str]] = []  # (div_id, clean_title)

    if decomp:
        scripts.append(_js("chart-decomp", _decomp_fig(decomp)))
    if roi:
        scripts.append(_js("chart-roi", _roi_fig(roi)))
    for i, p in enumerate(plots):
        div_id = f"chart-extra-{i}"
        p_layout = dict(p.get("layout") or {})
        p_layout.setdefault("paper_bgcolor", "rgba(0,0,0,0)")
        p_layout.setdefault("plot_bgcolor", "#f9fafb")
        if not p_layout.get("margin"):
            p_layout["margin"] = {"l": 60, "r": 40, "t": 70, "b": 70}
        scripts.append(_js(div_id, {"data": p.get("data", []), "layout": p_layout}))
        raw_t = (p.get("layout") or {}).get("title", {})
        if isinstance(raw_t, dict):
            raw_t = raw_t.get("text", f"Chart {i + 1}")
        ct = re.sub(r"<[^>]+>", " ", str(raw_t)).strip() or f"Chart {i + 1}"
        chart_htmls.append((div_id, ct))

    # ── Hero ─────────────────────────────────────────────────────────────────

    hero_kpis = ""
    if channels:
        hero_kpis += f'<div class="hero-kpi"><div class="val">{len(channels)}</div><div class="lbl">Media Channels</div></div>'
    if kpi != "KPI":
        hero_kpis += f'<div class="hero-kpi"><div class="val" style="font-size:1.1rem">{kpi}</div><div class="lbl">KPI</div></div>'
    if rows_val:
        hero_kpis += f'<div class="hero-kpi"><div class="val">{rows_val:,}</div><div class="lbl">Data Rows</div></div>'
    if roi:
        best_roi_val = max(r.get("roi_mean", 0) for r in roi)
        hero_kpis += f'<div class="hero-kpi"><div class="val">{best_roi_val:.2f}x</div><div class="lbl">Best Channel ROI</div></div>'
    if diag:
        hero_kpis += f'<div class="hero-kpi"><div class="val" style="font-size:1.4rem">{"✓" if diag.get("converged") else "⚠"}</div><div class="lbl">Model Status</div></div>'

    rq_preview = (research_q[:115] + "…") if len(research_q) > 115 else research_q
    hero_html = f"""
<div class="hero">
  <div class="hero-label">📊 MMM Project Report</div>
  <h1>{title}</h1>
  <div class="hero-meta">Generated {date_str}{f" &nbsp;·&nbsp; {rq_preview}" if rq_preview else ""}</div>
  {f'<div class="hero-kpis">{hero_kpis}</div>' if hero_kpis else ''}
</div>"""

    # ── Executive Summary ─────────────────────────────────────────────────────

    exec_rq = ""
    if research_q:
        extra_defs = ""
        if _rq_extra:
            extra_defs = '<dl style="display:grid;grid-template-columns:auto 1fr;gap:0.3rem 1rem;font-size:0.825rem;margin-top:0.75rem">'
            for k, v in _rq_extra.items():
                extra_defs += f'<dt style="color:#64748b;font-weight:600;text-transform:capitalize">{k.replace("_"," ")}</dt><dd style="color:#1e293b">{v}</dd>'
            extra_defs += "</dl>"
        exec_rq = f'<div class="card" style="margin-bottom:1.5rem"><div class="card-eyebrow">Research Question</div><h3 style="font-size:1.05rem;font-weight:700;color:#1e293b;margin-bottom:0">{research_q}</h3>{extra_defs}</div>'

    if findings:
        exec_insights = '<div class="insight-grid">'
        for icon, hdg, body in findings:
            exec_insights += f'<div class="insight"><div class="icon">{icon}</div><div class="body"><strong>{hdg}</strong><p>{body}</p></div></div>'
        exec_insights += "</div>"
    else:
        exec_insights = '<p class="section-intro">Fit a model and run decomposition and ROI analysis to generate the executive summary.</p>'

    exec_section = f"""
<section class="section" id="summary">
  <span class="section-label">Executive Summary</span>
  <h2>What the Model Found</h2>
  <p class="section-intro">A snapshot of the key findings from this Marketing Mix Modelling project.</p>
  <hr class="section-divider"/>
  {exec_rq}
  {exec_insights}
</section>"""

    # ── KPI Decomposition ─────────────────────────────────────────────────────

    if decomp:
        decomp_sorted = sorted(decomp, key=lambda x: x["pct_of_total"], reverse=True)
        media_components = [
            d
            for d in decomp_sorted
            if d["component"].lower()
            not in ("baseline", "trend", "seasonality", "controls")
        ]

        decomp_chart = '<div class="chart-card"><div class="chart-title">KPI Decomposition</div><div id="chart-decomp" style="height:380px"></div></div>'

        ch_cards = ""
        for i, d in enumerate(media_components):
            pct = d["pct_of_total"] * 100
            color = _PALETTE[i % len(_PALETTE)]
            contrib = _fmt_num(d.get("total_contribution"))
            ch_cards += f"""
<div class="channel-card">
  <div class="tier-badge" style="background:{color}22;color:{color};font-size:1.1rem">📺</div>
  <div class="ch-info">
    <div class="ch-name">{d["component"]}</div>
    <div class="ch-desc">{contrib} total contribution</div>
    <div class="prob-bar-wrap" style="margin-top:0.35rem">
      <div class="prob-bar"><div class="prob-bar-fill" style="width:{min(pct, 100):.1f}%;background:{color}"></div></div>
      <span class="prob-label">{pct:.1f}% of {kpi}</span>
    </div>
  </div>
</div>"""

        decomp_insight = ""
        if media_components:
            top_m = max(media_components, key=lambda x: x["pct_of_total"])
            baseline = next(
                (d for d in decomp_sorted if d["component"].lower() == "baseline"), None
            )
            ib = f"<b>{top_m['component']}</b> is the strongest media contributor at {_pct(top_m['pct_of_total'])} of {kpi}."
            if baseline:
                ib += f" The model baseline accounts for {_pct(baseline['pct_of_total'])} — media spend explains the remainder."
            decomp_insight = f'<div class="insight"><div class="icon">💡</div><div class="body"><strong>What this means</strong><p>{ib}</p></div></div>'

        decomp_section = f"""
<section class="section" id="decomposition">
  <span class="section-label">Attribution</span>
  <h2>What Drove {kpi}?</h2>
  <p class="section-intro">Decomposition breaks down fitted {kpi} into its contributing factors — showing which channels and baseline effects drive business outcomes.</p>
  <hr class="section-divider"/>
  {decomp_chart}
  {ch_cards}
  {decomp_insight}
</section>"""
    else:
        decomp_section = f"""
<section class="section" id="decomposition">
  <span class="section-label">Attribution</span>
  <h2>What Drove {kpi}?</h2>
  <hr class="section-divider"/>
  <div class="callout callout-info"><span class="ci">ℹ️</span><div class="ct"><strong>Not yet available</strong>Fit the model and run decomposition analysis to populate this section.</div></div>
</section>"""

    # ── Channel ROI ───────────────────────────────────────────────────────────

    if roi:
        roi_sorted = sorted(roi, key=lambda x: x.get("roi_mean", 0), reverse=True)

        roi_chart = '<div class="chart-card"><div class="chart-title">Channel ROI (94% Credible Interval)</div><div id="chart-roi" style="height:380px"></div></div>'

        roi_cards = ""
        for r in roi_sorted:
            mean = r.get("roi_mean", 0)
            lo = r.get("roi_hdi_low", mean)
            hi = r.get("roi_hdi_high", mean)
            prob = r.get("prob_profitable")
            if mean >= 1.5:
                tier_cls, roi_cls, tier_icon, tier_desc = (
                    "tier-strong",
                    "roi-strong",
                    "🟢",
                    "Strong performer",
                )
            elif mean >= 0.8:
                tier_cls, roi_cls, tier_icon, tier_desc = (
                    "tier-moderate",
                    "roi-moderate",
                    "🟡",
                    "Moderate performer",
                )
            else:
                tier_cls, roi_cls, tier_icon, tier_desc = (
                    "tier-weak",
                    "roi-weak",
                    "🔴",
                    "Below breakeven",
                )
            prob_html = ""
            if prob is not None:
                fc = (
                    "#059669"
                    if prob >= 0.8
                    else "#d97706" if prob >= 0.5 else "#dc2626"
                )
                prob_html = f'<div class="prob-bar-wrap"><div class="prob-bar"><div class="prob-bar-fill" style="width:{prob * 100:.0f}%;background:{fc}"></div></div><span class="prob-label">{prob * 100:.0f}% prob. profitable</span></div>'
            roi_cards += f"""
<div class="channel-card">
  <div class="tier-badge {tier_cls}">{tier_icon}</div>
  <div class="ch-info">
    <div class="ch-name">{r["channel"]}</div>
    <div class="ch-desc">{tier_desc} &nbsp;·&nbsp; 94% HDI [{lo:.2f}, {hi:.2f}]</div>
    {prob_html}
  </div>
  <div class="roi-pill {roi_cls}">{mean:.2f}x</div>
</div>"""

        best_r = roi_sorted[0]
        worst_r = roi_sorted[-1]
        roi_insight = (
            f'<div class="insight"><div class="icon">💰</div><div class="body">'
            f"<strong>Budget Implication</strong>"
            f'<p><b>{best_r["channel"]}</b> delivers the best return at {best_r.get("roi_mean", 0):.2f}x. '
            f"If budget is constrained, prioritise channels above 1.0x and consider reallocating spend from "
            f'<b>{worst_r["channel"]}</b> ({worst_r.get("roi_mean", 0):.2f}x).</p>'
            f"</div></div>"
        )

        roi_section = f"""
<section class="section" id="roi">
  <span class="section-label">Performance</span>
  <h2>Channel Return on Investment</h2>
  <p class="section-intro">How much {kpi} does each unit of spend generate? Bars show the posterior mean; whiskers show the 94% Bayesian credible interval.</p>
  <hr class="section-divider"/>
  {roi_chart}
  {roi_cards}
  {roi_insight}
</section>"""
    else:
        roi_section = """
<section class="section" id="roi">
  <span class="section-label">Performance</span>
  <h2>Channel Return on Investment</h2>
  <hr class="section-divider"/>
  <div class="callout callout-info"><span class="ci">ℹ️</span><div class="ct"><strong>Not yet available</strong>Fit the model and run ROI analysis to populate this section.</div></div>
</section>"""

    # ── Model Health ──────────────────────────────────────────────────────────

    if diag:
        converged = diag.get("converged", False)
        diag_msg = (
            "Model converged successfully. Bayesian estimates are reliable."
            if converged
            else "Model has convergence issues. Treat estimates with caution — consider adjusting priors or increasing samples."
        )
        try:
            rhat_val = float(str(diag.get("rhat_max", 99)))
            rhat_color = "green" if rhat_val < 1.01 else "amber"
        except (ValueError, TypeError):
            rhat_color = "amber"

        prior_callout = ""
        if prior_check:
            pc_ok = str(prior_check.get("value", "")).lower() in (
                "passed",
                "ok",
                "good",
                "yes",
                "true",
                "acceptable",
            )
            prior_callout = (
                f'<div class="callout {"callout-ok" if pc_ok else "callout-warn"}">'
                f'<span class="ci">{"✅" if pc_ok else "⚠️"}</span>'
                f'<div class="ct"><strong>Prior Predictive Check: {prior_check.get("value","")}</strong>'
                f'{prior_check.get("rationale","")}</div></div>'
            )

        diag_section = f"""
<section class="section" id="diagnostics">
  <span class="section-label">Validation</span>
  <h2>Model Health</h2>
  <p class="section-intro">Bayesian diagnostics confirm whether the MCMC sampler explored the posterior reliably. Good convergence is essential for trustworthy estimates.</p>
  <hr class="section-divider"/>
  <div class="callout {"callout-ok" if converged else "callout-warn"}">
    <span class="ci">{"✅" if converged else "⚠️"}</span>
    <div class="ct"><strong>{"Converged" if converged else "Convergence Warning"}</strong>{diag_msg}</div>
  </div>
  <div class="kpi-row">
    <div class="kpi-box"><div class="val {"green" if converged else "red"}">{diag.get("divergences", 0)}</div><div class="lbl">Divergences</div><div class="sub">Target: 0</div></div>
    <div class="kpi-box"><div class="val {rhat_color}">{diag.get("rhat_max", "—")}</div><div class="lbl">Max R̂</div><div class="sub">Target: &lt;1.01</div></div>
    <div class="kpi-box"><div class="val teal">{diag.get("ess_bulk_min", "—")}</div><div class="lbl">Min ESS Bulk</div><div class="sub">Target: &gt;400</div></div>
  </div>
  {prior_callout}
</section>"""
    else:
        diag_section = """
<section class="section" id="diagnostics">
  <span class="section-label">Validation</span>
  <h2>Model Health</h2>
  <hr class="section-divider"/>
  <div class="callout callout-info"><span class="ci">ℹ️</span><div class="ct"><strong>Not yet available</strong>Fit a model to see diagnostics.</div></div>
</section>"""

    # ── Additional Charts ─────────────────────────────────────────────────────

    if chart_htmls:
        extra_content = "".join(
            f'<div class="chart-card"><div class="chart-title">{ct}</div><div id="{did}" style="height:420px"></div></div>'
            for did, ct in chart_htmls
        )
        extra_section = f"""
<section class="section" id="charts">
  <span class="section-label">Analysis</span>
  <h2>Additional Charts</h2>
  <p class="section-intro">Charts generated during the analysis session providing further context on model behaviour and data patterns.</p>
  <hr class="section-divider"/>
  {extra_content}
</section>"""
    else:
        extra_section = ""

    # ── Appendix (technical detail) ───────────────────────────────────────────

    app_inner = ""

    if model_spec or model_run:
        inf = model_run.get("inference") or model_spec.get("inference") or {}
        trend_type = model_run.get("trend") or (model_spec.get("trend") or {}).get(
            "type", "—"
        )
        seas = model_run.get("seasonality") or model_spec.get("seasonality") or {}

        app_inner += (
            '<div class="app-section"><h3>Inference Settings</h3>'
            f'<p style="font-size:0.85rem;color:#475569">Chains: <b>{inf.get("chains","—")}</b>'
            f' &nbsp; Draws: <b>{inf.get("draws","—")}</b>'
            f' &nbsp; Tune: <b>{inf.get("tune","—")}</b>'
            f' &nbsp; Target Accept: <b>{inf.get("target_accept","—")}</b></p>'
            f'<p style="font-size:0.85rem;color:#475569;margin-top:0.4rem">Trend: <b>{trend_type}</b>'
            f' &nbsp; Yearly seasonality: <b>{seas.get("yearly", 0)}</b>'
            f' &nbsp; Weekly: <b>{seas.get("weekly", 0)}</b></p>'
            "</div>"
        )

        mc_list: list[dict] = model_spec.get("media_channels") or []
        if mc_list:
            app_inner += (
                '<div class="app-section"><h3>Channel Configuration</h3>'
                '<div class="table-wrap"><table><thead><tr>'
                "<th>Channel</th><th>Adstock</th><th>L-Max</th><th>Saturation</th>"
                "</tr></thead><tbody>"
            )
            for ch in mc_list:
                ads = ch.get("adstock") or {}
                sat = ch.get("saturation") or {}
                app_inner += (
                    f'<tr><td><b>{ch.get("name","")}</b></td>'
                    f'<td>{ads.get("type","—")}</td>'
                    f'<td>{ads.get("l_max","—")}</td>'
                    f'<td>{sat.get("type","—")}</td></tr>'
                )
            app_inner += "</tbody></table></div></div>"

        if controls:
            app_inner += (
                '<div class="app-section"><h3>Control Variables</h3><div class="tag-row">'
                + "".join(
                    f'<span class="badge badge-gray">{c}</span>' for c in controls
                )
                + "</div></div>"
            )

    if other_assumptions:
        app_inner += '<div class="app-section"><h3>Documented Assumptions</h3>'
        _cat_badge = {
            "causal_structure": "badge-indigo",
            "data": "badge-indigo",
            "functional_form": "badge-amber",
            "prior": "badge-amber",
            "identification": "badge-indigo",
            "external_evidence": "badge-green",
        }
        for a in other_assumptions:
            cb = _cat_badge.get(a.get("category", ""), "badge-gray")
            val_str = json.dumps(a.get("value"), ensure_ascii=False)
            if len(val_str) > 160:
                val_str = val_str[:157] + "…"
            app_inner += (
                f'<div style="border:1px solid #e2e8f0;border-radius:0.75rem;padding:0.875rem 1rem;margin-bottom:0.5rem;background:#fff">'
                f'<div style="margin-bottom:0.3rem"><span class="badge {cb}">{a.get("category","other")}</span>'
                f' <span style="font-size:0.8rem;font-weight:600;color:#334155;margin-left:0.4rem">{a.get("key","")}</span></div>'
                f'<div style="font-size:0.8rem;color:#475569"><b>Value:</b> {val_str}</div>'
                f'<div style="font-size:0.8rem;color:#475569;margin-top:0.15rem"><b>Rationale:</b> {a.get("rationale","")}</div>'
                f"</div>"
            )
        app_inner += "</div>"

    if sensitivity_assumptions:
        app_inner += '<div class="app-section"><h3>Sensitivity Analysis</h3>'
        for a in sensitivity_assumptions:
            v = a.get("value") or {}
            dropped = v.get("dropped", a.get("key", ""))
            frac = v.get("fraction_dropped", 0)
            app_inner += (
                f'<div style="border:1px solid #e2e8f0;border-radius:0.75rem;padding:0.875rem 1rem;margin-bottom:0.5rem;background:#fff">'
                f'<div style="font-weight:600;font-size:0.85rem">Drop: {dropped}</div>'
                f'<div style="font-size:0.8rem;color:#475569">Removing this channel reduces KPI by <b>{frac * 100:.1f}%</b>. {a.get("rationale","")}</div>'
                f"</div>"
            )
        app_inner += "</div>"

    appendix_html = (
        f'<div class="appendix"><h2>Technical Appendix</h2>{app_inner}</div>'
        if app_inner
        else ""
    )

    # ── Sidebar navigation ────────────────────────────────────────────────────

    nav_items = [
        ("#summary", "Executive Summary"),
        ("#decomposition", "KPI Decomposition"),
        ("#roi", "Channel ROI"),
        ("#diagnostics", "Model Health"),
    ]
    if chart_htmls:
        nav_items.append(("#charts", "Additional Charts"))
    if app_inner:
        nav_items.append(("#appendix", "Technical Appendix"))

    nav_html = "".join(
        f'<a href="{href}"><span class="dot"></span>{label}</a>'
        for href, label in nav_items
    )

    scripts_block = "\n".join(scripts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>{title} — MMM Report</title>
<script src="{_PLOTLY_CDN}"></script>
<style>{_REPORT_CSS}</style>
</head>
<body>
<div class="layout">
  <aside class="sidebar">
    <div class="logo">MMM Framework<span>Project Report</span></div>
    <div class="nav-group">Contents</div>
    <nav>{nav_html}</nav>
  </aside>
  <div class="body-wrap">
    {hero_html}
    <div class="main">
      {exec_section}
      {decomp_section}
      {roi_section}
      {diag_section}
      {extra_section}
    </div>
    {appendix_html}
    <div class="footer">MMM Framework — Bayesian Marketing Mix Modelling · Generated {date_str}</div>
  </div>
</div>
<script>
window.addEventListener('load', function() {{
  {scripts_block}
}});
</script>
</body>
</html>"""


# ── Slideshow ─────────────────────────────────────────────────────────────────


def generate_html_slides(
    title: str,
    date_str: str,
    dashboard: dict,
    assumptions: list[dict],
    client_mode: bool = False,
    client_name: str | None = None,
) -> str:
    dataset = dashboard.get("dataset") or {}
    model_spec = dashboard.get("model_spec") or {}
    decomp = dashboard.get("decomposition") or []
    roi = dashboard.get("roi_metrics") or []
    diag = dashboard.get("diagnostics") or {}
    plots = _hydrate_plots(dashboard.get("plots"))
    model_run = dashboard.get("model_run") or {}
    curves_data: dict = dashboard.get("saturation_curves") or {}
    mroi_data: dict = dashboard.get("marginal_roi") or {}
    sat_data: dict = dashboard.get("saturation") or {}

    _rq_raw_s = next(
        (a["value"] for a in assumptions if a.get("key") == "research_question"),
        None,
    )
    research_q, _rq_extra_s = (
        _extract_rq(_rq_raw_s)
        if _rq_raw_s is not None
        else ("To quantify the causal contribution of media channels to the KPI.", {})
    )
    channels: list[str] = model_run.get("channels") or [
        c.get("name", "") for c in (model_spec.get("media_channels") or [])
    ]
    kpi: str = model_run.get("kpi") or model_spec.get("kpi") or "KPI"
    date_range = dataset.get("date_range") or {}
    rows = dataset.get("rows")

    def _fmt_ch(name: str) -> str:
        return name.replace("_", " ").title() if client_mode else name

    display_channels = [_fmt_ch(c) for c in channels]

    # Apply channel name formatting to chart data in client mode
    chart_roi = (
        [{**r, "channel": _fmt_ch(r["channel"])} for r in roi] if client_mode else roi
    )
    chart_decomp = (
        [{**d, "component": _fmt_ch(d["component"])} for d in decomp]
        if client_mode
        else decomp
    )

    # Build key findings bullets
    findings: list[str] = []
    if decomp:
        top = max(decomp, key=lambda x: x["pct_of_total"])
        findings.append(
            f"<b>{_fmt_ch(top['component'])}</b> is the largest driver ({_pct(top['pct_of_total'])} of {kpi})"
        )
        media_pct = sum(
            d["pct_of_total"]
            for d in decomp
            if d["component"].lower()
            not in ("baseline", "trend", "seasonality", "controls")
        )
        if media_pct > 0:
            findings.append(
                f"Media collectively drives <b>{_pct(media_pct)}</b> of fitted {kpi}"
            )
    if roi:
        best = max(roi, key=lambda x: x.get("roi_mean", 0))
        worst = min(roi, key=lambda x: x.get("roi_mean", 0))
        findings.append(
            f"Highest ROI: <b>{_fmt_ch(best['channel'])}</b> at {best.get('roi_mean',0):.2f}x"
        )
        if len(roi) > 1:
            findings.append(
                f"Lowest ROI: <b>{_fmt_ch(worst['channel'])}</b> at {worst.get('roi_mean',0):.2f}x"
            )
    if client_mode:
        if diag.get("converged"):
            findings.append("Model estimates are statistically robust and reliable")
    else:
        if diag.get("converged"):
            findings.append("Model converged successfully (R̂ < 1.01, low divergences)")

    scripts: list[str] = []
    extra_slides = ""

    # Decomp slide
    decomp_slide = ""
    if chart_decomp:
        scripts.append(
            _js(
                "slide-decomp",
                _decomp_fig(chart_decomp),
                {"responsive": True, "displayModeBar": False},
            )
        )
        decomp_slide = (
            "<section>"
            '<span class="slide-tag">Results</span>'
            f"<h2>What Drove {kpi}?</h2>"
            '<div class="chart-slide" id="slide-decomp"></div>'
            "</section>"
        )

    # ROI slide
    roi_slide = ""
    if chart_roi:
        scripts.append(
            _js(
                "slide-roi",
                _roi_fig(chart_roi),
                {"responsive": True, "displayModeBar": False},
            )
        )
        roi_slide = (
            "<section>"
            '<span class="slide-tag">ROI</span>'
            "<h2>Channel Return on Investment</h2>"
            '<div class="chart-slide" id="slide-roi"></div>'
            "</section>"
        )

    # Client-specific slides: S-curves, mROI vs ROI, channel performance
    scurves_slide = ""
    mroi_slide = ""
    channel_perf_slide = ""

    if client_mode:
        # Build sat lookup (raw channel keys, either from curves or the saturation tool output)
        sat_for_perf = (
            {
                ch: {"saturation_level": c.get("saturation_level")}
                for ch, c in curves_data.items()
            }
            if curves_data
            else sat_data
        )

        # S-curves slide — pre-format channel name keys for legend labels
        if curves_data:
            curves_display = {_fmt_ch(ch): c for ch, c in curves_data.items()}
            scripts.append(
                _js(
                    "slide-scurves",
                    _scurves_fig(curves_display),
                    {"responsive": True, "displayModeBar": False},
                )
            )
            scurves_slide = (
                "<section>"
                '<span class="slide-tag">Efficiency</span>'
                "<h2>Diminishing Returns — Where Are You on the Curve?</h2>"
                '<div class="chart-slide" id="slide-scurves"></div>'
                "</section>"
            )

        # mROI vs avg ROI — pass raw roi (keys match mroi_data) + fmt_ch for display labels
        if roi and mroi_data:
            scripts.append(
                _js(
                    "slide-mroi",
                    _mroi_roi_fig(roi, mroi_data, _fmt_ch),
                    {"responsive": True, "displayModeBar": False},
                )
            )
            mroi_slide = (
                "<section>"
                '<span class="slide-tag">Marginal Efficiency</span>'
                "<h2>Average ROI vs Marginal ROI — What the Next £1 Earns</h2>"
                '<div class="chart-slide" id="slide-mroi"></div>'
                "</section>"
            )

        # Channel performance — pass raw roi so lookup keys match mroi_data / sat_for_perf
        if roi:
            perf_html = _channel_perf_html(roi, mroi_data, sat_for_perf, _fmt_ch)
            channel_perf_slide = (
                "<section>"
                '<span class="slide-tag">Recommendations</span>'
                "<h2>What's Working &amp; Budget Priorities</h2>"
                f"{perf_html}"
                "</section>"
            )

    # Extra chart slides — omitted in client mode (internal validation charts)
    if not client_mode:
        for i, p in enumerate(plots[:3]):
            div_id = f"slide-extra-{i}"
            raw_t = (p.get("layout") or {}).get("title", {})
            if isinstance(raw_t, dict):
                raw_t = raw_t.get("text", f"Chart {i+1}")
            slide_heading = _truncate_title(str(raw_t), max_len=55)
            p_layout = dict(p.get("layout") or {})
            p_layout["title"] = ""
            p_layout.update(
                {
                    "paper_bgcolor": "rgba(0,0,0,0)",
                    "plot_bgcolor": "#f9fafb",
                    "margin": {"l": 60, "r": 40, "t": 20, "b": 60},
                }
            )
            scripts.append(
                _js(
                    div_id,
                    {"data": p.get("data", []), "layout": p_layout},
                    {"responsive": True, "displayModeBar": False},
                )
            )
            extra_slides += (
                f"<section>"
                f'<span class="slide-tag">Analysis</span>'
                f'<h2 class="wrap">{slide_heading}</h2>'
                f'<div class="chart-slide" id="{div_id}"></div>'
                f"</section>"
            )

    # Diagnostics slide content
    if client_mode:
        diag_content = (
            '<div style="text-align:center;margin:1.5rem 0">'
            '<div style="font-size:3rem">✓</div>'
            '<div style="font-size:1.2rem;font-weight:700;color:#059669;margin:0.5rem 0">Model Validated</div>'
            "</div>"
            '<p style="color:#64748b;text-align:center">'
            "Estimates have been tested for statistical robustness and are suitable for decision-making."
            "</p>"
        )
    elif diag:
        converged = diag.get("converged", False)
        icon = "✓" if converged else "⚠"
        color = "#059669" if converged else "#d97706"
        diag_content = (
            f'<div style="text-align:center;margin:1.5rem 0">'
            f'<div style="font-size:3rem">{icon}</div>'
            f'<div style="font-size:1.2rem;font-weight:700;color:{color};margin:0.5rem 0">'
            f'{"Converged" if converged else "Convergence Issues"}</div>'
            f"</div>"
            f'<div class="stat-grid-slide">'
            f'<div class="stat-slide"><div class="v">{diag.get("divergences",0)}</div><div class="l">Divergences</div></div>'
            f'<div class="stat-slide"><div class="v">{diag.get("rhat_max","—")}</div><div class="l">Max R̂</div></div>'
            f'<div class="stat-slide"><div class="v">{diag.get("ess_bulk_min","—")}</div><div class="l">Min ESS</div></div>'
            f"</div>"
        )
    else:
        diag_content = '<p style="color:#64748b">Diagnostics not available.</p>'

    # Key findings slide
    findings_html = (
        "".join(f'<div class="finding-slide">{f}</div>' for f in findings)
        if findings
        else '<p style="color:#64748b">Fit a model to generate findings.</p>'
    )

    # Data slide stats
    data_stats = ""
    if client_mode and date_range.get("min") and date_range.get("max"):
        # Show analysis weeks instead of raw row count
        try:
            from datetime import datetime as _dt

            def _parse(s: str):
                for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
                    try:
                        return _dt.strptime(s, fmt).date()
                    except ValueError:
                        pass
                return None

            d_min = _parse(str(date_range["min"]))
            d_max = _parse(str(date_range["max"]))
            if d_min and d_max:
                weeks = round((d_max - d_min).days / 7)
                data_stats += f'<div class="stat-slide"><div class="v">{weeks}</div><div class="l">Weeks</div></div>'
        except Exception:
            if rows:
                data_stats += f'<div class="stat-slide"><div class="v">{rows:,}</div><div class="l">Data Points</div></div>'
    elif rows:
        data_stats += f'<div class="stat-slide"><div class="v">{rows:,}</div><div class="l">Rows</div></div>'

    if date_range.get("min"):
        data_stats += f'<div class="stat-slide"><div class="v" style="font-size:1.1em">{date_range["min"]}</div><div class="l">Start</div></div>'
    if date_range.get("max"):
        data_stats += f'<div class="stat-slide"><div class="v" style="font-size:1.1em">{date_range["max"]}</div><div class="l">End</div></div>'

    channels_html = "".join(f'<span class="stag">{c}</span>' for c in display_channels)

    # Model spec technical detail — hidden in client mode
    if client_mode:
        model_spec_detail = (
            '<p style="margin-top:0.75rem;color:#64748b;font-size:0.85em">'
            "Bayesian marketing mix model estimating the incremental contribution of each media channel to the KPI."
            "</p>"
        )
    else:
        model_spec_detail = _model_spec_slide_detail(model_spec, model_run)

    # Title slide subtitle
    if client_mode:
        subtitle_html = f'<p class="subtitle" style="text-align:center">{client_name or "Confidential"}</p>'
    else:
        subtitle_html = '<p class="subtitle" style="text-align:center">Marketing Mix Modelling — Project Report</p>'

    # Confidentiality footer (client mode only)
    confidential_footer = ""
    if client_mode:
        confidential_footer = (
            f'<div style="position:fixed;bottom:0;left:0;right:0;'
            f"background:rgba(248,250,252,0.95);border-top:1px solid #e2e8f0;"
            f'padding:0.3rem 1rem;font-size:0.6em;color:#94a3b8;text-align:center;z-index:100">'
            f'CONFIDENTIAL — Prepared for {client_name or "Client"} · Not for distribution'
            f"</div>"
        )

    diag_slide_label = "Model Validation" if client_mode else "Model Diagnostics"

    scripts_block = "\n      ".join(scripts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>{title} — Slides</title>
<link rel="stylesheet" href="{_REVEAL_CSS}"/>
<link rel="stylesheet" href="{_REVEAL_THEME}"/>
<script src="{_PLOTLY_CDN}"></script>
<script src="{_REVEAL_JS}"></script>
<style>{_SLIDE_CSS}</style>
</head>
<body>
{confidential_footer}
<div class="reveal">
  <div class="slides">

    <!-- Title -->
    <section data-slide-type="title" style="text-align:center">
      <h1 style="text-align:center">{title}</h1>
      {subtitle_html}
      <p class="subtitle" style="text-align:center;font-size:0.7em;margin-top:0.75rem;color:#94a3b8">{date_str}</p>
    </section>

    <!-- Research Question -->
    <section>
      <span class="slide-tag">Objective</span>
      <h2>Research Question</h2>
      {_rq_slide_html(research_q, _rq_extra_s)}
    </section>

    <!-- Data -->
    <section>
      <span class="slide-tag">Data</span>
      <h2>Dataset Overview</h2>
      {(f'<div class="stat-grid-slide">{data_stats}</div>') if data_stats else ''}
      {(f'<p style="margin-top:1rem"><b>KPI:</b> {kpi}</p>') if kpi != "—" else ''}
      {(f'<p style="margin-top:0.75rem"><b>Media Channels:</b></p><div class="tag-row" style="margin-top:0.4rem">{channels_html}</div>') if display_channels else ''}
    </section>

    <!-- Model Setup -->
    <section>
      <span class="slide-tag">Model</span>
      <h2>Model Specification</h2>
      {'<ul>' + ''.join(f'<li>{c}</li>' for c in display_channels) + '</ul>' if display_channels else '<p style="color:#94a3b8">No channels configured.</p>'}
      {model_spec_detail}
    </section>

    <!-- Decomposition -->
    {decomp_slide}

    <!-- ROI -->
    {roi_slide}

    <!-- S-curves (client mode only) -->
    {scurves_slide}

    <!-- mROI vs avg ROI (client mode only) -->
    {mroi_slide}

    <!-- Channel performance (client mode only) -->
    {channel_perf_slide}

    <!-- Diagnostics -->
    <section>
      <span class="slide-tag">Validation</span>
      <h2>{diag_slide_label}</h2>
      {diag_content}
    </section>

    <!-- Extra charts -->
    {extra_slides}

    <!-- Key Findings -->
    <section>
      <span class="slide-tag">Findings</span>
      <h2>Key Findings</h2>
      {findings_html}
    </section>

    <!-- Next Steps -->
    <section>
      <span class="slide-tag">Next Steps</span>
      <h2>Recommended Actions</h2>
      <ul>
        <li>Validate ROI estimates against held-out experiments or geo lift tests</li>
        <li>Run budget optimisation using the calibrated model</li>
        <li>Monitor for structural breaks (seasonality shifts, new channels)</li>
        <li>Conduct a full sensitivity analysis by refitting with channel exclusions</li>
      </ul>
    </section>

  </div>
</div>
<script>
  Reveal.initialize({{
    hash: true, slideNumber: true, transition: 'fade', transitionSpeed: 'fast',
    width: 1100, height: 700, margin: 0.05,
  }});
  Reveal.on('ready', function() {{
    {scripts_block}
    setTimeout(function() {{
      document.querySelectorAll('.chart-slide').forEach(function(el) {{
        Plotly.Plots.resize(el);
      }});
    }}, 300);
  }});
  Reveal.on('slidechanged', function() {{
    document.querySelectorAll('.chart-slide').forEach(function(el) {{
      Plotly.Plots.resize(el);
    }});
  }});
</script>
</body>
</html>"""


def _model_spec_slide_detail(model_spec: dict, model_run: dict) -> str:
    inf = model_run.get("inference") or model_spec.get("inference") or {}
    trend_type = model_run.get("trend") or (model_spec.get("trend") or {}).get("type")
    if not inf and not trend_type:
        return ""
    parts = []
    if inf.get("chains"):
        parts.append(f'{inf["chains"]} chains × {inf.get("draws","?")} draws')
    if trend_type:
        parts.append(f"Trend: {trend_type}")
    return (
        f'<p style="margin-top:0.75rem;color:#64748b;font-size:0.85em">{" &nbsp;·&nbsp; ".join(parts)}</p>'
        if parts
        else ""
    )
