"""Animate the MONOTONE SPLINE activation converging on an unknown curve shape.

    uv run --with plotly --with kaleido --with pillow python build_spline_animation.py

A sibling of ``build_misspecification_animation.py``. That one dramatizes fitting
the WRONG parametric family; this one shows the way out: the **shape-agnostic
monotone I-spline** activation (``activation="monotone_spline"``) learning the
true curve with NO family assumption at all. The true per-channel response is the
two-Hill mixture of notebook §14 — a two-phase shape no single parametric family
in the registry matches — and we run TWO real accumulating trust-region loops on
the same world: one fitting the **monotone spline**, one fitting a parametric
reference family (default **logistic** — §14's SEVERE misspecification: concave,
no S-shape, so it *cannot* represent the truth at any sample size; ``SPL_REF=hill``
swaps in the mild case, where the contrast is decision-only).

A 2×2 GIF (all Scatter traces):

* Top row — *the curve you fit*, wave by wave. Left: the spline's Pulse
  response-recovery, purple mean + a 90% band that starts WIDE (an honest "any
  monotone saturating shape") and **tightens onto the fixed black truth** — the
  convergence property being demonstrated. Right: the parametric reference on
  the SAME fixed y-range, tightening onto the *wrong* curve, with a red overlay
  where the truth escapes its band. Corner badges: profit gap, curve coverage
  (the % of the sweep where the truth is inside the 90% band), band width, R̂.
* Bottom row — summary trackers. Left: profit-gap vs truth-optimal for both
  families (both converge — the decision was never the problem, §14's lesson).
  Right: curve coverage per wave — the spline's stays far above the wrong
  family's as both bands shrink: the spline converges toward the truth, the
  wrong family converges toward its own bias.

The single idea: **flexibility + designed local waves = convergence to the true
curve**, where a structurally wrong family converges only in decision, not in
shape. (Both families stay honestly uncertain only where the design probed —
neither band should be read as a promise about far extrapolation.)

Env knobs: SPL_N_WAVES(6), SPL_WARM_SPLINE(400), SPL_WARM_REF(400),
SPL_REF("logistic"|"hill"), SPL_N_GEO(72), SPL_T_PRE(6), SPL_T_TEST(10),
SPL_NOISE(0.4), SPL_DELTA(0.6).
"""

from __future__ import annotations

import json
import os
import warnings

import jax
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image
from plotly.subplots import make_subplots

import mmm_framework.continuous_learning as cl
from mmm_framework.continuous_learning.surface import ACTIVATIONS

warnings.filterwarnings("ignore")
jax.config.update("jax_platform_name", "cpu")
pio.templates.default = "plotly_white"

OUTDIR = os.path.join(os.path.dirname(__file__), "artifacts")


def _f(name, default):
    return float(os.environ.get(name, default))


def _i(name, default):
    return int(os.environ.get(name, default))


# ── decision + design constants (match notebook §14/§15) ─────────────────────
CENTER = np.full(4, 0.7)
B, VALUE = 3.2, 5.0
PROBE_LO, PROBE_HI = CENTER[0] * 0.4, CENTER[0] * 1.6  # ~[0.28, 1.12]
FOCAL = 1  # Pulse — the strongly two-phase channel

N_WAVES = _i("SPL_N_WAVES", 6)
N_GEO = _i("SPL_N_GEO", 72)
T_PRE = _i("SPL_T_PRE", 6)
T_TEST = _i("SPL_T_TEST", 10)
NOISE = _f("SPL_NOISE", 0.4)
DELTA = _f("SPL_DELTA", 0.6)
WARM_SPLINE = _i("SPL_WARM_SPLINE", 400)
WARM_REF = _i("SPL_WARM_REF", 400)
REF = os.environ.get("SPL_REF", "logistic")  # the parametric reference family
REF_LABEL = "logistic" if REF == "logistic" else "single Hill"

CACHE = f"/tmp/_spline_anim_cache_{REF}.npz"

# ── the true world (a real analyst never sees this) ──────────────────────────
world_mix = cl.make_world_hill_mixture(seed=0)
GRID = np.linspace(0.0, 1.5, 80)

COL = {
    "spline": "#7b5aa6",
    "ref": "#c9962f" if REF == "logistic" else "#3a6ea5",
    "truth": "#111111",
    "red": "#d1495b",
}
RGB = {
    "spline": "123,90,166",
    "ref": "201,150,47" if REF == "logistic" else "58,110,165",
}


# ── §14 helpers (defined locally; the standalone builder has no notebook scope) ─
def anchored_slice(post, c, s_grid, x0):
    """Posterior draws of the total incremental response as channel ``c`` is
    swept (others at ``x0``), anchored at channel-off so it isolates shape."""
    names = ACTIVATIONS[post.activation][0]
    fn = ACTIVATIONS[post.activation][1]
    S = post.samples
    D = S["beta"].shape[0]
    beta = S["beta"]
    shp = [S[nm] for nm in names]
    gv = {(i, j): S[cl.pair_name(post.channels, (i, j))] for (i, j) in post.pairs}
    out = np.empty((D, s_grid.size))
    for t, v in enumerate(s_grid):
        x = np.tile(x0, (D, 1))
        x[:, c] = v
        f = np.asarray(fn(x, *shp))
        tot = (beta * f).sum(1)
        for (i, j), g in gv.items():
            tot = tot + g * f[:, i] * f[:, j]
        out[:, t] = tot
    return out - out[:, :1]


def true_anchored(c, s_grid, x0):
    X = np.tile(x0, (s_grid.size, 1))
    X[:, c] = s_grid
    v = world_mix.response_mean(X)
    return v - v[0]


def prof_true(a):
    return VALUE * float(world_mix.response_mean(np.asarray(a, float)[None, :])[0]) - B


# ── the two accumulating loops ───────────────────────────────────────────────
TRUE_CURVE = true_anchored(FOCAL, GRID, CENTER)
_TRUE_ALLOC, TRUE_PROFIT = cl.world_optimal_allocation(
    world_mix, B, VALUE, mode="fixed"
)


def _run_family(activation: str, nw: int) -> dict:
    """One trust-region loop, capturing the Pulse band + convergence metrics per
    wave. Shares wave 0 (seed=1) across families so early frames compare."""
    state = cl.LearningState(
        channels=world_mix.channels,
        center=CENTER.copy(),
        B=B,
        value=VALUE,
        pairs=world_mix.pairs,
        pair_signs=cl.PAIR_SIGNS_EXAMPLE,
        activation=activation,
        mode="fixed",
    )
    w0 = cl.simulate_panel(
        world_mix,
        CENTER,
        n_geo=N_GEO,
        t_pre=T_PRE,
        t_test=T_TEST,
        delta=DELTA,
        noise=NOISE,
        seed=1,
    )
    a_geo = np.asarray(w0["a_geo"])
    state.ingest(w0)

    mean, lo, hi, gap, cover, bandw, rhat, rows = [], [], [], [], [], [], [], []
    for wave in range(N_WAVES):
        state.fit(num_warmup=nw, num_samples=nw, num_chains=2, seed=wave)
        post = state.posterior
        rec = state.recommend(q=200)
        gap.append(100.0 * (TRUE_PROFIT - prof_true(rec)) / abs(TRUE_PROFIT))
        cur = anchored_slice(post, FOCAL, GRID, CENTER)
        m = cur.mean(0)
        l5 = np.percentile(cur, 5, 0)
        h95 = np.percentile(cur, 95, 0)
        mean.append(m)
        lo.append(l5)
        hi.append(h95)
        # curve-convergence metrics over the sweep (skip the anchored 0 point)
        inside = (TRUE_CURVE[1:] >= l5[1:]) & (TRUE_CURVE[1:] <= h95[1:])
        cover.append(100.0 * float(np.mean(inside)))
        bandw.append(float(np.mean(h95 - l5)))
        rh = post.diagnostics.get("max_rhat")
        rhat.append(float(rh) if rh is not None else np.nan)
        rows.append(int(state.data["spend"].shape[0]))
        if wave < N_WAVES - 1:
            design = cl.central_composite(rec, DELTA, world_mix.pairs)
            wn = cl.simulate_wave(
                world_mix,
                design,
                a_geo,
                t_test=T_TEST,
                center=rec,
                noise=NOISE,
                seed=2 + wave,
            )
            state.recenter(rec)
            state.ingest(wn)
    return {
        "mean": np.array(mean),
        "lo": np.array(lo),
        "hi": np.array(hi),
        "gap": np.array(gap),
        "cover": np.array(cover),
        "bandw": np.array(bandw),
        "rhat": np.array(rhat),
        "rows": np.array(rows),
    }


def compute() -> dict:
    spline = _run_family("monotone_spline", WARM_SPLINE)
    ref = _run_family(REF, WARM_REF)
    # red mask: where the true curve leaves the REFERENCE family's band, per wave
    red = np.where(
        (TRUE_CURVE[None, :] < ref["lo"]) | (TRUE_CURVE[None, :] > ref["hi"]),
        np.tile(TRUE_CURVE, (N_WAVES, 1)),
        np.nan,
    )
    # global fixed ranges (learning must read as band-shrink, never axis rescale)
    all_lo = np.concatenate(
        [spline["lo"].ravel(), ref["lo"].ravel(), [0.0], TRUE_CURVE]
    )
    all_hi = np.concatenate(
        [spline["hi"].ravel(), ref["hi"].ravel(), [0.0], TRUE_CURVE]
    )
    pad = 0.05 * (all_hi.max() - all_lo.min())
    y_curve = (float(all_lo.min() - pad), float(all_hi.max() + pad))
    gap_max = float(max(spline["gap"].max(), ref["gap"].max()))
    return {
        "grid": GRID,
        "truth_curve": TRUE_CURVE,
        "red": red,
        "s_mean": spline["mean"],
        "s_lo": spline["lo"],
        "s_hi": spline["hi"],
        "h_mean": ref["mean"],
        "h_lo": ref["lo"],
        "h_hi": ref["hi"],
        "s_gap": spline["gap"],
        "h_gap": ref["gap"],
        "s_cover": spline["cover"],
        "h_cover": ref["cover"],
        "s_bandw": spline["bandw"],
        "h_bandw": ref["bandw"],
        "s_rhat": spline["rhat"],
        "h_rhat": ref["rhat"],
        "s_rows": spline["rows"],
        "h_rows": ref["rows"],
        "y_curve": np.array(y_curve),
        "gap_max": np.array([gap_max]),
    }


def _load() -> dict:
    if not os.path.exists(CACHE):
        np.savez(CACHE, **compute())
    return dict(np.load(CACHE, allow_pickle=True))


# ── one frame ────────────────────────────────────────────────────────────────
def _rhat_str(v):
    return "—" if not np.isfinite(v) else f"{v:.2f}"


def _band(fig, grid, lo, hi, rgb, row, col):
    fig.add_trace(
        go.Scatter(
            x=np.r_[grid, grid[::-1]],
            y=np.r_[hi, lo[::-1]],
            fill="toself",
            fillcolor=f"rgba({rgb},0.16)",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def frame_figure(d: dict, w: int) -> go.Figure:
    grid = d["grid"]
    truth = d["truth_curve"]
    y_curve = tuple(d["y_curve"])
    gap_max = float(d["gap_max"][0])
    waves = np.arange(w + 1)

    fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=[0.60, 0.40],
        vertical_spacing=0.16,
        horizontal_spacing=0.12,
        subplot_titles=(
            "Shape-agnostic — monotone spline (no family assumed)",
            f"Parametric — {REF_LABEL} ({'severe misspecification' if REF == 'logistic' else 'mild misspecification'})",
            "Decision — profit gap vs truth-optimal (both)",
            "Convergence — % of curve inside the 90% band",
        ),
    )
    # resize ONLY the four subplot titles, before any manual annotation
    for a in fig.layout.annotations[:4]:
        a.font.size = 12.5

    # ---- top-left: monotone spline ----
    fig.add_vrect(
        x0=PROBE_LO,
        x1=PROBE_HI,
        fillcolor="rgba(120,120,120,0.07)",
        line_width=0,
        row=1,
        col=1,
    )
    _band(fig, grid, d["s_lo"][w], d["s_hi"][w], RGB["spline"], 1, 1)
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=truth,
            mode="lines",
            line=dict(color=COL["truth"], width=3.3),
            name="true curve (two-Hill mixture)",
            legendgroup="truth",
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=d["s_mean"][w],
            mode="lines",
            line=dict(color=COL["spline"], width=2.2),
            name="fit: monotone spline",
            legendgroup="spline",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # ---- top-right: single Hill ----
    fig.add_vrect(
        x0=PROBE_LO,
        x1=PROBE_HI,
        fillcolor="rgba(120,120,120,0.07)",
        line_width=0,
        row=1,
        col=2,
    )
    _band(fig, grid, d["h_lo"][w], d["h_hi"][w], RGB["ref"], 1, 2)
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=truth,
            mode="lines",
            line=dict(color=COL["truth"], width=3.3),
            legendgroup="truth",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=d["h_mean"][w],
            mode="lines",
            line=dict(color=COL["ref"], width=2.2),
            name=f"fit: {REF_LABEL} (wrong family)",
            legendgroup="hill",
            showlegend=True,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=d["red"][w],
            mode="lines",
            line=dict(color=COL["red"], width=3.2),
            connectgaps=False,
            name="true curve outside fitted band",
            legendgroup="red",
            showlegend=True,
        ),
        row=1,
        col=2,
    )

    # dummy legend proxy for the probed-range shading (a vrect cannot legend)
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=11, symbol="square", color="rgba(120,120,120,0.35)"),
            name="probed range",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # ---- bottom-left: profit gap, both families (the decision converges) ----
    for key, fam in [("s_gap", "spline"), ("h_gap", "ref")]:
        fig.add_trace(
            go.Scatter(
                x=waves,
                y=d[key][: w + 1],
                mode="lines+markers",
                line=dict(color=COL[fam], width=2.4),
                marker=dict(size=7),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    # ---- bottom-right: curve coverage (calibrated convergence vs decay) ----
    for key, fam in [("s_cover", "spline"), ("h_cover", "ref")]:
        fig.add_trace(
            go.Scatter(
                x=waves,
                y=d[key][: w + 1],
                mode="lines+markers",
                line=dict(color=COL[fam], width=2.4),
                marker=dict(size=7),
                showlegend=False,
            ),
            row=2,
            col=2,
        )
    fig.add_hline(y=90.0, line=dict(color="gray", dash="dash"), row=2, col=2)

    # ---- corner badges (top-left interior, provably empty) ----
    for col, pre in [(1, "s"), (2, "h")]:
        xr = "x domain" if col == 1 else "x2 domain"
        yr = "y domain" if col == 1 else "y2 domain"
        fam = "spline" if col == 1 else "ref"
        fig.add_annotation(
            xref=xr,
            yref=yr,
            x=0.03,
            y=0.97,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            align="left",
            font=dict(size=10.5),
            text=(
                f"gap {d[pre + '_gap'][w]:.1f}%<br>"
                f"curve cover {d[pre + '_cover'][w]:.0f}%<br>"
                f"band width {d[pre + '_bandw'][w]:.2f} · "
                f"R̂ {_rhat_str(float(d[pre + '_rhat'][w]))}"
            ),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=COL[fam],
            borderwidth=1,
            borderpad=3,
        )

    # ---- axes (all ranges FIXED across frames) ----
    for col in (1, 2):
        fig.update_xaxes(
            title_text="Pulse spend (scaled)", range=[0, 1.5], row=1, col=col
        )
        fig.update_yaxes(
            title_text="Δ response vs. off" if col == 1 else "",
            range=list(y_curve),
            row=1,
            col=col,
        )
    fig.update_xaxes(
        title_text="wave (experiment round)",
        range=[-0.4, N_WAVES - 0.6],
        dtick=1,
        row=2,
        col=1,
    )
    fig.update_xaxes(
        title_text="wave (experiment round)",
        range=[-0.4, N_WAVES - 0.6],
        dtick=1,
        row=2,
        col=2,
    )
    fig.update_yaxes(
        title_text="profit gap (%)", range=[0, gap_max * 1.1], row=2, col=1
    )
    fig.update_yaxes(title_text="truth inside band (%)", range=[0, 105], row=2, col=2)

    gs, gh = float(d["s_gap"][w]), float(d["h_gap"][w])
    cs, ch = float(d["s_cover"][w]), float(d["h_cover"][w])
    rows = int(d["s_rows"][w])
    fig.update_layout(
        width=1240,
        height=820,
        margin=dict(l=72, r=30, t=128, b=122),
        title=dict(
            text=(
                "<b>The monotone spline converges on a curve no parametric "
                "family matches</b><br>"
                f"<sup>Wave {w} · {rows} geo-weeks · two-Hill-mixture truth · "
                f"gap spline {gs:.1f}% vs {REF_LABEL} {gh:.1f}% · curve cover "
                f"{cs:.0f}% vs {ch:.0f}%</sup>"
            ),
            x=0.5,
            xanchor="center",
            y=0.975,
            font=dict(size=16),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.13,
            xanchor="center",
            x=0.5,
            font=dict(size=10.5),
        ),
    )
    return fig


def build_gif(d: dict) -> str:
    os.makedirs(OUTDIR, exist_ok=True)
    frames_dir = os.path.join(OUTDIR, f"_spline_frames_{REF}")
    os.makedirs(frames_dir, exist_ok=True)
    imgs = []
    for w in range(N_WAVES):
        fig = frame_figure(d, w)
        p = os.path.join(frames_dir, f"f{w:02d}.png")
        fig.write_image(p, width=1240, height=820, scale=2)
        imgs.append(Image.open(p).convert("RGB"))
    suf = "" if REF == "logistic" else f"_{REF}"
    gif = os.path.join(OUTDIR, f"continuous_learning_spline{suf}.gif")
    per = max(700, int(11000 / N_WAVES))
    durations = [per] * (N_WAVES - 1) + [2800]
    imgs[0].save(
        gif,
        save_all=True,
        append_images=imgs[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    return gif


def _metrics(d: dict) -> dict:
    return {
        "n_waves": int(N_WAVES),
        "n_geo": int(N_GEO),
        "t_test": int(T_TEST),
        "noise": float(NOISE),
        "gap_spline": [round(float(x), 2) for x in d["s_gap"]],
        "ref_family": REF,
        "gap_ref": [round(float(x), 2) for x in d["h_gap"]],
        "cover_spline": [round(float(x), 1) for x in d["s_cover"]],
        "cover_ref": [round(float(x), 1) for x in d["h_cover"]],
        "bandw_spline": [round(float(x), 3) for x in d["s_bandw"]],
        "bandw_ref": [round(float(x), 3) for x in d["h_bandw"]],
        "rhat_spline": [round(float(x), 2) for x in d["s_rhat"]],
        "rhat_ref": [round(float(x), 2) for x in d["h_rhat"]],
    }


def main() -> None:
    d = _load()
    gif = build_gif(d)
    m = _metrics(d)
    with open(os.path.join(OUTDIR, f"_spline_metrics_{REF}.json"), "w") as fh:
        json.dump(m, fh, indent=2)
    print(f"ref family = {REF_LABEL} | waves={N_WAVES} n_geo={N_GEO} noise={NOISE}")
    print(f"profit gap spline : {m['gap_spline']}")
    print(f"profit gap ref    : {m['gap_ref']}")
    print(f"curve cover spline: {m['cover_spline']}")
    print(f"curve cover ref   : {m['cover_ref']}")
    print(f"band width spline : {m['bandw_spline']}")
    print(f"band width ref    : {m['bandw_ref']}")
    print("wrote", gif)


if __name__ == "__main__":
    main()
