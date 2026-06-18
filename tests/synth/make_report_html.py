"""Render a self-contained HTML robustness report (data + results).

Recomputes the *violated* data-generating curves and series from
:mod:`tests.synth.dgp` (deliberately NOT the model's clean generative family),
reads ``results/stress_matrix.{json,md}`` and ``results/bench.json``, and writes
one offline HTML file with inline SVG charts — no JS, no CDN, no network.

    uv run python -m tests.synth.make_report_html
    # -> tests/synth/results/robustness_report.html
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import numpy as np

from . import dgp

RESULTS = Path("tests/synth/results")

# Aurora brand palette (shared with the showcase) -------------------------------
INK = "#2b2118"
ACCENT = "#b5651d"
CREMA = "#c8a26a"
LEAF = "#3f7d5e"
BERRY = "#a63a50"
SKY = "#3b6ea5"
AMBER = "#d98a2b"
MUTED = "#8a8079"
PAPER = "#fbf7f0"
CARD = "#ffffff"
CH_COLOR = {"TV": ACCENT, "Search": SKY, "Social": BERRY, "Display": LEAF}


def esc(s) -> str:
    return html.escape(str(s))


# ---------------------------------------------------------------------------
# inline-SVG chart primitives (dependency-free)
# ---------------------------------------------------------------------------


def _ticks(lo, hi, n=4):
    if hi <= lo:
        hi = lo + 1
    step = (hi - lo) / n
    return [lo + i * step for i in range(n + 1)]


def line_chart(
    curves,
    *,
    w=540,
    h=300,
    pad=46,
    xlabel="",
    ylabel="",
    xdomain=None,
    ydomain=None,
    xfmt="{:.0f}",
    yfmt="{:.2f}",
):
    """curves: list of dict(label,color,points=[(x,y)],dash=bool,width=float)."""
    xs = [p[0] for c in curves for p in c["points"]]
    ys = [p[1] for c in curves for p in c["points"]]
    x0, x1 = xdomain or (min(xs), max(xs))
    y0, y1 = ydomain or (min(ys), max(ys))
    if y1 == y0:
        y1 = y0 + 1
    if x1 == x0:
        x1 = x0 + 1
    sx = lambda x: pad + (x - x0) / (x1 - x0) * (w - 2 * pad)  # noqa: E731
    sy = lambda y: h - pad - (y - y0) / (y1 - y0) * (h - 2 * pad)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    # gridlines + y ticks
    for ty in _ticks(y0, y1):
        yy = sy(ty)
        out.append(
            f'<line x1="{pad}" y1="{yy:.1f}" x2="{w - pad}" y2="{yy:.1f}" '
            f'stroke="#eadfce" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{pad - 6}" y="{yy + 3:.1f}" text-anchor="end" '
            f'class="tick">{yfmt.format(ty)}</text>'
        )
    for tx in _ticks(x0, x1):
        xx = sx(tx)
        out.append(
            f'<text x="{xx:.1f}" y="{h - pad + 16}" text-anchor="middle" '
            f'class="tick">{xfmt.format(tx)}</text>'
        )
    # axes
    out.append(
        f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" '
        f'stroke="{INK}" stroke-width="1.5"/>'
    )
    out.append(
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h - pad}" '
        f'stroke="{INK}" stroke-width="1.5"/>'
    )
    if xlabel:
        out.append(
            f'<text x="{w / 2:.0f}" y="{h - 6}" text-anchor="middle" '
            f'class="axlab">{esc(xlabel)}</text>'
        )
    if ylabel:
        out.append(
            f'<text x="14" y="{h / 2:.0f}" text-anchor="middle" '
            f'transform="rotate(-90 14 {h / 2:.0f})" class="axlab">{esc(ylabel)}</text>'
        )
    # lines
    for c in curves:
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in c["points"])
        dash = ' stroke-dasharray="6 4"' if c.get("dash") else ""
        out.append(
            f'<polyline points="{pts}" fill="none" stroke="{c["color"]}" '
            f'stroke-width="{c.get("width", 2.4)}"{dash} '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
    # legend
    lx, ly = pad + 8, pad + 4
    for c in curves:
        out.append(
            f'<rect x="{lx}" y="{ly - 8}" width="14" height="3.5" rx="1" '
            f'fill="{c["color"]}"/>'
        )
        out.append(
            f'<text x="{lx + 19}" y="{ly - 4}" class="leg">{esc(c["label"])}</text>'
        )
        ly += 16
    out.append("</svg>")
    return "".join(out)


def grouped_bars(
    categories,
    series,
    *,
    w=540,
    h=300,
    pad=46,
    ylabel="",
    yfmt="{:.0f}",
    ydomain=None,
    zero_line=True,
):
    """series: list of dict(label,color,values aligned to categories)."""
    allv = [v for s in series for v in s["values"]]
    y0, y1 = ydomain or (min(0, min(allv)), max(allv))
    if y1 == y0:
        y1 = y0 + 1
    sy = lambda y: h - pad - (y - y0) / (y1 - y0) * (h - 2 * pad)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    for ty in _ticks(y0, y1):
        yy = sy(ty)
        out.append(
            f'<line x1="{pad}" y1="{yy:.1f}" x2="{w - pad}" y2="{yy:.1f}" '
            f'stroke="#eadfce"/>'
        )
        out.append(
            f'<text x="{pad - 6}" y="{yy + 3:.1f}" text-anchor="end" '
            f'class="tick">{yfmt.format(ty)}</text>'
        )
    n_cat = len(categories)
    n_ser = len(series)
    band = (w - 2 * pad) / max(n_cat, 1)
    bw = band * 0.8 / max(n_ser, 1)
    for ci, cat in enumerate(categories):
        cx = pad + band * ci + band * 0.1
        for si, s in enumerate(series):
            v = s["values"][ci]
            x = cx + si * bw
            yv = sy(v)
            y0p = sy(0)
            top = min(yv, y0p)
            hh = abs(yv - y0p)
            out.append(
                f'<rect x="{x:.1f}" y="{top:.1f}" width="{bw - 2:.1f}" '
                f'height="{hh:.1f}" rx="2" fill="{s["color"]}"/>'
            )
        out.append(
            f'<text x="{cx + (n_ser * bw) / 2:.1f}" y="{h - pad + 16}" '
            f'text-anchor="middle" class="tick">{esc(cat)}</text>'
        )
    if zero_line:
        out.append(
            f'<line x1="{pad}" y1="{sy(0):.1f}" x2="{w - pad}" y2="{sy(0):.1f}" '
            f'stroke="{INK}" stroke-width="1.5"/>'
        )
    if ylabel:
        out.append(
            f'<text x="14" y="{h / 2:.0f}" text-anchor="middle" '
            f'transform="rotate(-90 14 {h / 2:.0f})" class="axlab">{esc(ylabel)}</text>'
        )
    lx, ly = pad + 8, pad + 4
    for s in series:
        out.append(
            f'<rect x="{lx}" y="{ly - 8}" width="12" height="10" rx="2" '
            f'fill="{s["color"]}"/>'
        )
        out.append(f'<text x="{lx + 17}" y="{ly}" class="leg">{esc(s["label"])}</text>')
        ly += 16
    out.append("</svg>")
    return "".join(out)


def hbar_chart(
    items,
    *,
    w=540,
    rowh=26,
    pad_l=150,
    pad_r=60,
    color=ACCENT,
    vfmt="{:.1f}",
    highlight=None,
):
    """items: list of (label, value). Horizontal bars, value labels at the end."""
    vmax = max(v for _, v in items) or 1
    h = rowh * len(items) + 16
    sx = lambda v: pad_l + v / vmax * (w - pad_l - pad_r)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    for i, (lab, v) in enumerate(items):
        y = 8 + i * rowh
        col = highlight.get(lab, color) if highlight else color
        out.append(
            f'<text x="{pad_l - 8}" y="{y + rowh / 2 + 3:.0f}" text-anchor="end" '
            f'class="tick">{esc(lab)}</text>'
        )
        out.append(
            f'<rect x="{pad_l}" y="{y + 3:.0f}" width="{sx(v) - pad_l:.1f}" '
            f'height="{rowh - 10}" rx="3" fill="{col}"/>'
        )
        out.append(
            f'<text x="{sx(v) + 6:.1f}" y="{y + rowh / 2 + 3:.0f}" '
            f'class="tick">{vfmt.format(v)}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def scatter(points, *, w=420, h=300, pad=46, color=SKY, xlabel="", ylabel=""):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    sx = lambda x: pad + (x - x0) / (x1 - x0) * (w - 2 * pad)  # noqa: E731
    sy = lambda y: h - pad - (y - y0) / (y1 - y0) * (h - 2 * pad)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    out.append(
        f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" '
        f'stroke="{INK}"/>'
    )
    out.append(
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h - pad}" stroke="{INK}"/>'
    )
    for x, y in points:
        out.append(
            f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" '
            f'fill="{color}" fill-opacity="0.55"/>'
        )
    out.append(
        f'<text x="{w / 2:.0f}" y="{h - 6}" text-anchor="middle" '
        f'class="axlab">{esc(xlabel)}</text>'
    )
    out.append(
        f'<text x="14" y="{h / 2:.0f}" text-anchor="middle" '
        f'transform="rotate(-90 14 {h / 2:.0f})" class="axlab">{esc(ylabel)}</text>'
    )
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------------------
# data: recompute the *violated* generating processes
# ---------------------------------------------------------------------------


def data_panels() -> list[dict]:
    """Each panel proves a test dataset deviates from the model's assumptions."""
    panels = []
    x = np.linspace(0, 1, 120)

    # 1. Saturation form: model's concave 1-exp vs the true S-shaped Hill.
    model_sat = dgp._logistic_sat(x, dgp._LAM["TV"])
    true_sat = dgp._hill_sat(x, 0.45, 3.0)
    panels.append(
        {
            "id": "saturation_misspec",
            "title": "Saturation shape — the model can't bend like the data",
            "assumption": "Model assumes a strictly concave 1−exp(−λx) curve (diminishing returns from the first dollar).",
            "reality": "saturation_misspec data is S-shaped (Hill, coef 3): a low-spend threshold, then a steep rise.",
            "svg": line_chart(
                [
                    {
                        "label": "model: 1−exp(−λx)",
                        "color": MUTED,
                        "points": list(zip(x, model_sat)),
                        "dash": True,
                    },
                    {
                        "label": "truth: S-curve (Hill)",
                        "color": BERRY,
                        "points": list(zip(x, true_sat)),
                    },
                ],
                xlabel="normalized adstocked spend",
                ylabel="saturated response",
                yfmt="{:.1f}",
            ),
            "result": "46% median error, 25% coverage — and PPC + refutation both pass. Silent.",
        }
    )

    # 2. Adstock kernel: model's geometric (l_max 8, peak 0) vs delayed Weibull tail.
    lags = np.arange(26)
    geo = dgp._geom_adstock(
        np.eye(26)[0], dgp._ALPHA["TV"], l_max=8
    )  # impulse response
    wei = dgp._weibull_adstock(np.eye(26)[0], 2.6, 9.0, 26)
    panels.append(
        {
            "id": "adstock_misspec",
            "title": "Carryover shape & window — truncated and mis-peaked",
            "assumption": "Model uses geometric decay truncated at l_max = 8 weeks, peaking at lag 0.",
            "reality": "adstock_misspec data is a delayed Weibull peaking ~6–8 weeks out with mass past 26 weeks.",
            "svg": line_chart(
                [
                    {
                        "label": "model: geometric (8-wk)",
                        "color": MUTED,
                        "points": list(zip(lags, geo)),
                        "dash": True,
                    },
                    {
                        "label": "truth: delayed Weibull (26-wk)",
                        "color": ACCENT,
                        "points": list(zip(lags, wei)),
                    },
                ],
                xlabel="lag (weeks)",
                ylabel="carryover weight",
                yfmt="{:.2f}",
            ),
            "result": "77% median error — caught here only by the robustness value (Search flagged fragile).",
        }
    )

    # 3. Unobserved confounding: spend chases the latent (hidden) demand.
    sc = dgp.build("unobserved_confounding")
    demand = sc.notes["latent_demand"]
    corr = float(np.corrcoef(sc.spend["Search"], demand)[0, 1])
    pts = list(zip((demand - demand.mean()).tolist(), sc.spend["Search"].tolist()))
    panels.append(
        {
            "id": "unobserved_confounding",
            "title": "Unobserved confounding — spend is not exogenous",
            "assumption": "Model assumes media spend is exogenous (no hidden driver of both spend and sales).",
            "reality": f"Search spend chases a hidden demand signal (corr {corr:.2f}); that signal also lifts sales.",
            "svg": scatter(
                pts,
                color=SKY,
                xlabel="latent demand (hidden, centered)",
                ylabel="Search spend",
            ),
            "result": "41% median error; Search over-credited +153%; truth outside the CI. Silent.",
        }
    )

    # 4. Multicollinearity: all channels move on one shared flighting calendar.
    mc = dgp.build("multicollinearity")
    t = np.arange(len(mc.y))
    curves = []
    for c in dgp.CHANNELS:
        s = mc.spend[c].to_numpy()
        curves.append(
            {
                "label": c,
                "color": CH_COLOR[c],
                "points": list(zip(t[:60], (s / s.mean())[:60])),
                "width": 1.8,
            }
        )
    panels.append(
        {
            "id": "multicollinearity",
            "title": "Multicollinearity — channels rise and fall together",
            "assumption": "Model assumes enough independent spend variation to separate each channel's effect.",
            "reality": f"All four channels share one flighting calendar (mean pairwise corr {mc.notes['mean_pairwise_corr']:.2f}).",
            "svg": line_chart(
                curves, xlabel="week", ylabel="spend / channel mean", yfmt="{:.1f}"
            ),
            "result": "29% median error, total media right (−2%) but the per-channel split is scrambled. Silent.",
        }
    )

    # 5. Spend outliers: one data-entry spike inflates the max-normalizer.
    so = dgp.build("spend_outliers")
    tv = so.spend["TV"].to_numpy()
    t2 = np.arange(len(tv))
    panels.append(
        {
            "id": "spend_outliers",
            "title": "Spend outlier — one spike compresses the normalizer",
            "assumption": "Model normalizes each channel by its training max, assuming the max is representative.",
            "reality": "A 15× data-entry spike sets the max; every real week collapses toward 0 on the curve.",
            "svg": line_chart(
                [
                    {
                        "label": "TV spend (with spike)",
                        "color": ACCENT,
                        "points": list(zip(t2, tv)),
                    },
                ],
                xlabel="week",
                ylabel="spend",
                yfmt="{:.0f}",
            ),
            "result": "46% median error, 0% coverage — no channel's truth lands in its interval. Silent.",
        }
    )

    # 6. Time-varying beta: effectiveness drifts + a structural break.
    n = len(dgp.build("clean").y)
    brk = n // 2
    tv_mult = np.linspace(1.4, 0.6, n)
    se_mult = np.where(np.arange(n) < brk, 0.6, 1.5)
    tt = np.arange(n)
    panels.append(
        {
            "id": "time_varying_beta",
            "title": "Time-varying effectiveness — not one constant β",
            "assumption": "Model fits one constant coefficient per channel for the whole window.",
            "reality": "TV fatigues over two years; Search jumps at a mid-series break (algo/creative change).",
            "svg": line_chart(
                [
                    {
                        "label": "TV effectiveness (fatigue)",
                        "color": ACCENT,
                        "points": list(zip(tt, tv_mult)),
                    },
                    {
                        "label": "Search effectiveness (break)",
                        "color": SKY,
                        "points": list(zip(tt, se_mult)),
                    },
                ],
                xlabel="week",
                ylabel="true β multiplier",
                yfmt="{:.1f}",
            ),
            "result": "11% median error — the window total absorbs the drift (a time-resolved estimand would not).",
        }
    )
    return panels


# ---------------------------------------------------------------------------
# results: stress matrix + per-channel + benchmark
# ---------------------------------------------------------------------------


def parse_per_channel(md: str) -> dict:
    """Parse the '### name — ...' per-channel tables from the matrix markdown."""
    out = {}
    blocks = md.split("\n### ")[1:]
    for b in blocks:
        name = b.split(" — ")[0].strip()
        rows = []
        for line in b.splitlines():
            m = re.match(
                r"\|\s*([A-Za-z]\w*)\s*\|\s*([-\d,]+)\s*\|\s*([-\d,]+)\s*\|"
                r"\s*([+\-]?\d+%)\s*\|\s*(✓|✗)\s*\|",
                line,
            )
            if m:
                rows.append(
                    {
                        "channel": m.group(1),
                        "true": float(m.group(2).replace(",", "")),
                        "est": float(m.group(3).replace(",", "")),
                        "rel": m.group(4),
                        "covered": m.group(5) == "✓",
                    }
                )
        if rows:
            out[name] = rows
    return out


def verdict_meta(r: dict):
    if r.get("silent_failure"):
        return ("SILENT FAILURE", BERRY, "verdict-silent")
    if not r.get("representable"):
        return ("expected (unrepresentable)", MUTED, "verdict-exp")
    return ("ok", LEAF, "verdict-ok")


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------


def build_html() -> str:
    rows = json.loads((RESULTS / "stress_matrix.json").read_text())
    bench = json.loads((RESULTS / "bench.json").read_text())
    per_ch = parse_per_channel((RESULTS / "stress_matrix.md").read_text())
    panels = data_panels()

    n_silent = sum(r["silent_failure"] for r in rows)
    clean = next(r for r in rows if r["name"] == "clean")

    # --- data-not-clean panels --------------------------------------------
    panel_html = []
    for p in panels:
        panel_html.append(
            f"""
        <div class="panel">
          <div class="panel-text">
            <h3>{esc(p['title'])}</h3>
            <p class="assume"><span class="tag tag-model">model assumes</span> {esc(p['assumption'])}</p>
            <p class="real"><span class="tag tag-data">the data does</span> {esc(p['reality'])}</p>
            <p class="res">&rarr; {esc(p['result'])}</p>
          </div>
          <div class="panel-chart">{p['svg']}</div>
        </div>"""
        )

    # --- the matrix table --------------------------------------------------
    trows = []
    for r in rows:
        label, col, cls = verdict_meta(r)
        med = f"{r['median_abs_rel_error'] * 100:.0f}%"
        mx = f"{r['max_abs_rel_error'] * 100:.0f}%"
        cov = f"{r['coverage_rate'] * 100:.0f}%"
        cov_cls = "bad" if r["coverage_rate"] < 0.75 else "good"
        med_cls = "bad" if r["median_abs_rel_error"] > 0.25 else "good"
        div = r["divergences"]
        ppc = "✓" if r["ppc_pass"] else "✗"
        trows.append(
            f"""<tr>
          <td class="mono">{esc(r['name'])}</td>
          <td class="muted small">{esc(r['violates'] or '— (positive control)')}</td>
          <td class="num {med_cls}">{med}</td><td class="num">{mx}</td>
          <td class="num {cov_cls}">{cov}</td>
          <td class="num">{r['rhat_max']:.2f}</td>
          <td class="num">{div}</td>
          <td class="num">{ppc}</td>
          <td><span class="badge" style="background:{col}1a;color:{col};border-color:{col}55">{esc(label)}</span></td>
        </tr>"""
        )

    # --- silent-failure spotlights (true vs est bars) ---------------------
    spot = []
    for name in [
        "unobserved_confounding",
        "multicollinearity",
        "saturation_misspec",
        "spend_outliers",
        "confounding_controlled",
    ]:
        if name not in per_ch:
            continue
        chs = per_ch[name]
        cats = [c["channel"] for c in chs]
        bars = grouped_bars(
            cats,
            [
                {
                    "label": "true (causal)",
                    "color": INK,
                    "values": [c["true"] for c in chs],
                },
                {
                    "label": "model estimate",
                    "color": ACCENT,
                    "values": [c["est"] for c in chs],
                },
            ],
            ylabel="total contribution",
            h=260,
        )
        miss = [c["channel"] for c in chs if not c["covered"]]
        miss_txt = (
            f"Truth outside the 90% CI for: <b>{', '.join(miss)}</b>."
            if miss
            else "All channels covered."
        )
        spot.append(
            f"""
        <div class="card">
          <h4 class="mono">{esc(name)}</h4>
          {bars}
          <p class="small muted">{miss_txt}</p>
        </div>"""
        )

    # --- benchmark: fit time + sampler ------------------------------------
    grid = {g["label"]: g for g in bench["grid"]}
    order = [
        "base(600d/4c/param/pymc/156w/4ch)",
        "numpyro",
        "legacy_adstock",
        "chains=2",
        "draws=300",
        "draws=1200",
        "weeks=104",
        "weeks=260",
        "channels=8",
    ]
    nice = {
        "base(600d/4c/param/pymc/156w/4ch)": "baseline (pyMC, 4ch)",
        "numpyro": "numpyro backend",
        "legacy_adstock": "legacy adstock",
        "chains=2": "2 chains",
        "draws=300": "300 draws",
        "draws=1200": "1200 draws",
        "weeks=104": "104 weeks",
        "weeks=260": "260 weeks",
        "channels=8": "8 channels",
    }
    time_items = [(nice[k], grid[k]["sample_s"]) for k in order if k in grid]
    hl = {"numpyro backend": LEAF, "8 channels": BERRY}
    time_chart = hbar_chart(time_items, color=ACCENT, vfmt="{:.0f}s", highlight=hl)
    ops = bench["ops"]
    ops_items = [
        ("fit (baseline)", grid[order[0]]["sample_s"]),
        ("counterfactual contrib", ops["counterfactual_contributions_s"]),
        ("parameter learning", ops["parameter_learning_s"]),
        ("refutation suite (4 refits)", ops["refutation_suite_s"]),
    ]
    ops_chart = hbar_chart(
        ops_items,
        color=SKY,
        vfmt="{:.1f}s",
        highlight={"refutation suite (4 refits)": BERRY},
    )

    # --- convergence per scenario -----------------------------------------
    conv_rows = []
    for r in rows:
        bad = r["divergences"] > 0 or r["rhat_max"] >= 1.05
        conv_rows.append(
            f"""<tr>
          <td class="mono small">{esc(r['name'])}</td>
          <td class="num">{r['rhat_max']:.3f}</td>
          <td class="num {'bad' if r['divergences'] else ''}">{r['divergences']}</td>
          <td>{'⚠ pathological' if bad else '✓ clean'}</td></tr>"""
        )

    worst = grid[order[0]]["worst_ess_bulk"][:4]
    worst_txt = ", ".join(
        f"<span class='mono'>{esc(w[0])}</span> ({w[1]:.0f})" for w in worst
    )

    # --- failure-mode catalog ---------------------------------------------
    fm = [
        (
            "A",
            "Convergence &amp; geometry",
            LEAF,
            [
                (
                    "β·λ identifiability ridge",
                    "high",
                    "confirmed",
                    "Contribution ≈ β·λ·x in the linear regime — only the product is identified; a curved ridge.",
                ),
                (
                    "Diagonal mass matrix can't rotate ridges",
                    "high",
                    "confirmed",
                    "adapt_diag tunes per-parameter scales only (and overrides PyMC's jitter); r̂ floors at ~1.01.",
                ),
                (
                    "Collinear channels → divergences",
                    "high",
                    "confirmed",
                    "Parallel regressors: sum identified, difference not; sign-swapping attribution.",
                ),
                (
                    "Hill + latent mediation (extensions)",
                    "high",
                    "confirmed",
                    "Stiff Hill × bilinear mediator → Aurora kitchen-sink r̂ 1.11 / 108 divergences.",
                ),
            ],
        ),
        (
            "B",
            "Fit-time &amp; scaling",
            SKY,
            [
                (
                    "Trace memory blows up with n_obs",
                    "high",
                    "latent",
                    "~8 obs-sized Deterministics + channel_contributions stored every draw; ~4 GB at 156wk×50geo.",
                ),
                (
                    "Parametric adstock inside every leapfrog step",
                    "high",
                    "latent",
                    "O(n_obs·l_max) convolution per channel per gradient — why channels scale super-linearly.",
                ),
                (
                    "Contributions quadratic in channels",
                    "high",
                    "latent",
                    "N+1 predictive passes, each re-evaluating all N adstock kernels.",
                ),
                (
                    "Validation refits multiply the fit",
                    "medium",
                    "latent",
                    "Refutation ≤4×, CV ~5×, bootstrap up to ~20× — each re-incurs the trace cost.",
                ),
            ],
        ),
        (
            "C",
            "Numerical &amp; data edge cases",
            AMBER,
            [
                (
                    "Delayed/Weibull adstock NaN kernel (0/0)",
                    "high",
                    "latent (verified)",
                    "Unguarded w/w.sum() (adstock_pt.py:68); large θ / tiny scale underflows all weights → NaN.",
                ),
                (
                    "No NaN/inf input validation",
                    "high",
                    "latent",
                    "One bad cell → NaN y_std → cryptic 'Bad initial energy'. No finiteness check in _prepare_data.",
                ),
                (
                    "Geo cross-boundary adstock bleed",
                    "high",
                    "latent (verified)",
                    "Adstock convolves the whole stacked column — carryover bleeds across geo boundaries.",
                ),
                (
                    "Near-constant column leverage",
                    "medium",
                    "latent",
                    "+1e-8 floor; a 0/…/1 flag standardizes to z≈12.4; a lone spike makes sat_lam unidentifiable.",
                ),
            ],
        ),
        (
            "D",
            "Statistical &amp; operational",
            BERRY,
            [
                (
                    "adstock·saturation·β equifinality",
                    "high",
                    "confirmed",
                    "Many (decay, λ, β) triples fit equally — non-identified even with infinite data.",
                ),
                (
                    "Prior–data conflict (downward shrink)",
                    "medium",
                    "confirmed",
                    "Gamma(μ=1.5) β prior pulls large true effects down — the clean control's ~7% under-recovery.",
                ),
                (
                    "Counterfactual HDI inflation",
                    "medium",
                    "latent (verified)",
                    "Default random_seed=None unpairs noise → CI ~√2 too wide. The harness passes a seed, so its coverage is honest.",
                ),
                (
                    "Strict r̂<1.01 gate false-alarms",
                    "medium",
                    "confirmed",
                    "The ridge parks r̂ at ~1.01 → fast fits flagged 'not converged' despite 0 divergences.",
                ),
            ],
        ),
    ]
    fm_html = []
    for letter, title, col, items in fm:
        cards = "".join(
            f"""
          <div class="fm">
            <div class="fm-head"><span class="sev sev-{it[1].split()[0]}">{esc(it[1])}</span>
              <span class="status">{esc(it[2])}</span></div>
            <div class="fm-title">{esc(it[0])}</div>
            <div class="fm-body small muted">{esc(it[3])}</div>
          </div>"""
            for it in items
        )
        fm_html.append(
            f"""
        <div class="fm-group">
          <h3 style="color:{col}">{letter}. {title}</h3>
          <div class="fm-grid">{cards}</div>
        </div>"""
        )

    # ---------------------------------------------------------------- HTML
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MMM Robustness Report</title>
<style>
:root{{--ink:{INK};--accent:{ACCENT};--paper:{PAPER};--card:{CARD};--muted:{MUTED};
--leaf:{LEAF};--berry:{BERRY};--sky:{SKY};--crema:{CREMA};}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);
font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}}
.wrap{{max-width:1080px;margin:0 auto;padding:0 24px 80px}}
header{{background:linear-gradient(135deg,{INK} 0%,#3a2c1f 60%,{ACCENT} 160%);color:#fbf7f0;
padding:56px 24px 44px;}}
header .wrap{{padding-bottom:0}}
h1{{font-size:34px;margin:0 0 8px;letter-spacing:-.5px}}
.sub{{font-size:18px;opacity:.9;max-width:760px;margin:0 0 22px}}
.kpis{{display:flex;gap:14px;flex-wrap:wrap;margin-top:8px}}
.kpi{{background:#ffffff18;border:1px solid #ffffff33;border-radius:12px;padding:12px 16px;min-width:120px}}
.kpi b{{display:block;font-size:26px;line-height:1.1}}
.kpi span{{font-size:12.5px;opacity:.85}}
h2{{font-size:24px;margin:54px 0 6px;letter-spacing:-.3px}}
h2 .n{{color:var(--accent);font-variant-numeric:tabular-nums;margin-right:8px}}
.lead{{color:#5b5048;margin:0 0 22px;max-width:820px}}
section{{scroll-margin-top:20px}}
.panel{{display:grid;grid-template-columns:1fr 540px;gap:22px;align-items:center;
background:var(--card);border:1px solid #ece2d2;border-radius:16px;padding:20px 22px;margin:16px 0;
box-shadow:0 1px 2px #0000000a}}
.panel-text h3{{margin:0 0 10px;font-size:18px}}
.tag{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.4px;
padding:2px 8px;border-radius:6px;margin-right:6px;white-space:nowrap}}
.tag-model{{background:#8a807922;color:var(--muted)}}
.tag-data{{background:{ACCENT}22;color:var(--accent)}}
.panel p{{margin:8px 0;font-size:14.5px}}
.assume{{color:#6b6058}} .real{{color:var(--ink)}}
.res{{margin-top:12px!important;font-weight:600;color:var(--berry);font-size:14px}}
.chart{{width:100%;height:auto;display:block}}
.tick{{font-size:11px;fill:#7a6f64}} .axlab{{font-size:12px;fill:#5b5048;font-weight:600}}
.leg{{font-size:12px;fill:var(--ink)}}
table{{width:100%;border-collapse:collapse;background:var(--card);border-radius:12px;overflow:hidden;
border:1px solid #ece2d2;font-size:14px;margin:8px 0}}
th,td{{padding:9px 12px;text-align:left;border-bottom:1px solid #f0e8da}}
th{{background:{INK};color:#f3ece0;font-size:12px;text-transform:uppercase;letter-spacing:.4px}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}
.mono{{font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace;font-size:13px}}
.small{{font-size:12.5px}} .muted{{color:#8a8079}}
.bad{{color:var(--berry);font-weight:700}} .good{{color:var(--leaf);font-weight:600}}
.badge{{font-size:11.5px;font-weight:700;padding:3px 9px;border-radius:999px;border:1px solid}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
.grid3{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}
.card{{background:var(--card);border:1px solid #ece2d2;border-radius:14px;padding:16px;box-shadow:0 1px 2px #0000000a}}
.card h4{{margin:0 0 6px}}
.note{{background:#fff;border-left:4px solid var(--accent);border-radius:0 10px 10px 0;
padding:14px 18px;margin:16px 0;font-size:14.5px}}
.fm-group{{margin:20px 0}}
.fm-group h3{{font-size:17px;margin:0 0 10px}}
.fm-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.fm{{background:var(--card);border:1px solid #ece2d2;border-radius:12px;padding:12px 14px}}
.fm-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}}
.fm-title{{font-weight:700;font-size:14px;margin-bottom:3px}}
.sev{{font-size:10.5px;font-weight:800;text-transform:uppercase;padding:2px 7px;border-radius:5px}}
.sev-high{{background:{BERRY}22;color:{BERRY}}} .sev-medium{{background:{AMBER}22;color:#9a6212}}
.status{{font-size:11px;color:#8a8079;font-style:italic}}
.recs{{counter-reset:r}}
.recs li{{margin:7px 0}}
footer{{margin-top:50px;padding-top:20px;border-top:1px solid #e6dccb;color:#8a8079;font-size:13px}}
code{{background:#f1e9db;padding:1px 6px;border-radius:5px;font-size:12.5px}}
.toc{{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 0}}
.toc a{{font-size:12.5px;color:#f3ece0;background:#ffffff18;border:1px solid #ffffff33;
padding:5px 11px;border-radius:8px;text-decoration:none}}
@media(max-width:840px){{.panel{{grid-template-columns:1fr}}.grid2,.grid3,.fm-grid{{grid-template-columns:1fr}}}}
</style></head>
<body>
<header><div class="wrap">
  <h1>MMM Robustness Report</h1>
  <p class="sub">Marketing-Mix Model stress-tested on data that <b>breaks its structural
  assumptions</b> — confounding, wrong functional forms, collinearity, outliers — not the
  clean data the model was built to fit. Recovery, convergence, fit-time and failure modes.</p>
  <div class="kpis">
    <div class="kpi"><b>13</b><span>structural-violation scenarios</span></div>
    <div class="kpi"><b style="color:#ffd9a8">{n_silent}</b><span>silent failures (wrong, all-green)</span></div>
    <div class="kpi"><b>{clean['median_abs_rel_error'] * 100:.0f}%</b><span>error on the clean control</span></div>
    <div class="kpi"><b>~17s</b><span>fit · 25s refutation</span></div>
  </div>
  <div class="toc">
    <a href="#data">1 · The data isn't clean</a>
    <a href="#recovery">2 · Recovery</a>
    <a href="#convergence">3 · Convergence</a>
    <a href="#time">4 · Fit-time</a>
    <a href="#failures">5 · Failure modes</a>
  </div>
</div></header>
<div class="wrap">

<section id="data">
<h2><span class="n">1</span>The test data is not the model's generative process</h2>
<p class="lead">The model's existing fixtures draw data from (essentially) the same family it
assumes, so a passing test only proves the sampler works when reality matches the model. Every
panel below shows a test dataset that <b>deliberately deviates</b> from a model assumption. The
one dataset that <i>does</i> match — <span class="mono">clean</span> — is the positive control,
recovered to {clean['median_abs_rel_error'] * 100:.0f}% with {clean['coverage_rate'] * 100:.0f}% coverage.</p>
{''.join(panel_html)}
</section>

<section id="recovery">
<h2><span class="n">2</span>Recovery &amp; coverage — the matrix</h2>
<p class="lead">For each scenario: median / worst per-channel error on total contribution, 90%-CI
coverage, and the routine diagnostics. A <b style="color:{BERRY}">silent failure</b> is a
representable scenario that is badly wrong yet passes every check an analyst acts on
(convergence + robustness value). <code>ppc</code> is shown but excluded from the gate (it fires
on the clean control too).</p>
<table>
<thead><tr><th>scenario</th><th>assumption broken</th><th>med|err|</th><th>max|err|</th>
<th>cover</th><th>r̂</th><th>div</th><th>ppc</th><th>verdict</th></tr></thead>
<tbody>{''.join(trows)}</tbody></table>

<h3 style="margin-top:30px">Silent-failure spotlight — true vs. estimated contribution</h3>
<p class="lead">Where the bars diverge, the model is confidently misattributing budget while every
routine diagnostic stays green.</p>
<div class="grid2">{''.join(spot)}</div>
</section>

<section id="convergence">
<h2><span class="n">3</span>Convergence</h2>
<p class="lead">National single-KPI fits converge cleanly — r̂ ≈ 1.005–1.01, 0 divergences
everywhere except the Aurora kitchen-sink (extension model). But the production gate
(<code>r̂ &lt; 1.01</code>, strict) false-alarms because an inherent <b>β·λ identifiability
ridge</b> parks r̂ right at 1.01.</p>
<div class="grid2">
  <div class="card">
    <h4>Per-scenario convergence</h4>
    <table style="margin:0"><thead><tr><th>scenario</th><th>r̂ max</th><th>div</th><th></th></tr></thead>
    <tbody>{''.join(conv_rows)}</tbody></table>
  </div>
  <div>
    <div class="note"><b>Why r̂ sticks at ~1.01.</b> A channel's contribution is ≈ β·λ·x in the
    linear regime, so only the <i>product</i> β·λ is identified — a curved ridge a diagonal NUTS
    mass matrix can't rotate onto. The lowest-ESS parameters are always the media/saturation
    terms: {worst_txt}. More draws help (300→1200 lifts min-ESS 790→3,112).</div>
    <div class="note" style="border-color:{BERRY}"><b>Where divergences appear.</b> Collinear
    spend, the legacy 2-point adstock blend, single-group hierarchies, and Hill + latent
    mediation (Aurora: r̂ 1.11, 108 divergences).</div>
  </div>
</div>
</section>

<section id="time">
<h2><span class="n">4</span>Fit-time &amp; scaling</h2>
<p class="lead">A 156-week / 4-channel fit is fast; the analyses <i>around</i> it dominate.
<b>numpyro is ~2× faster</b> than pyMC; channels scale super-linearly; the
<b>refutation suite costs more than the fit itself</b>.</p>
<div class="grid2">
  <div class="card"><h4>Pure fit() wall-clock by config</h4>{time_chart}
    <p class="small muted">One knob varied at a time from the baseline. <span style="color:{LEAF}">numpyro</span>
    halves the time; <span style="color:{BERRY}">8 channels</span> is ~2.7× the 4-channel fit.</p></div>
  <div class="card"><h4>Cost of a scored scenario</h4>{ops_chart}
    <p class="small muted">The <span style="color:{BERRY}">refutation suite</span> (4 refits) is the
    cost center — bigger than the fit. Contributions are O(channels) predictive passes.</p></div>
</div>
<div class="note"><b>Scaling laws.</b> Time ∝ n_obs (≈ weeks × geos) and ∝ draws; super-linear in
channels (parametric adstock runs an O(n_obs·l_max) convolution inside every leapfrog step, per
channel). At geo scale the obs-sized deterministics in the trace can reach multiple GB — compute
the decomposition post-hoc instead of storing it every draw.</div>
</section>

<section id="failures">
<h2><span class="n">5</span>Failure-mode catalog</h2>
<p class="lead">Beyond recovery bias: convergence/geometry, scaling, numerical edge cases, and
statistical/operational risks. <i>confirmed</i> = observed empirically here; <i>latent</i> =
code-confirmed but off the default path.</p>
{''.join(fm_html)}
<div class="note" style="border-color:{LEAF}"><b>The one real fix.</b> The ridge, collinearity,
and confounding are <b>identification</b> problems no sampler setting solves. Anchoring effects
with randomized geo-lift / incrementality experiments (<code>calibration/</code>) pins β,
collapses the ridge, and is the one signal these silent failures cannot fake.</div>
</section>

<footer>
<p>Generated from <code>tests/synth/results/stress_matrix.json</code> + <code>bench.json</code>;
data curves recomputed from <code>tests/synth/dgp.py</code>. Full write-up:
<code>technical-docs/mmm-robustness-report.md</code>.</p>
<p>Reproduce: <code>uv run python -m tests.synth.run_stress_matrix --all</code> ·
<code>uv run python -m tests.synth.bench</code> ·
<code>uv run python -m tests.synth.make_report_html</code></p>
</footer>
</div></body></html>"""


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "robustness_report.html"
    out.write_text(build_html())
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
