"""Animate the continuous-learning loop LEARNING WITH THE WRONG response family.

    uv run --with plotly --with kaleido --with pillow python builders/build_misspecification_animation.py

A sibling of ``build_acquisition_animation.py`` (that one animates the acquisition
*surfaces* while fitting the world's own correct activation). This one dramatizes
notebook §14 "When the response family is wrong": the true per-channel response is
a **two-Hill mixture**, and we run TWO real accumulating trust-region loops on the
same world — one fitting the **correct** mixture family, one fitting the **wrong**
single Hill — wave by wave.

A 2×2 GIF (all Scatter traces — no contours/colorbars, so no-overlap holds
trivially):

* Top row — *the curve you fit* (calibration). Left: the correct family's Pulse
  response-recovery, green mean + shrinking 90% band hugging and covering the
  fixed black truth. Right: the WRONG family's recovery on the SAME fixed y-range,
  blue mean + a shrinking-but-biased band, with a **red overlay** where the true
  curve falls OUTSIDE the wrong band (beyond the probed range a single Hill cannot
  bend two-phase). Corner badges carry gap%, mROAS-CI coverage, and R̂.
* Bottom row — summary trackers. Left: profit-gap vs truth-optimal for BOTH
  families; they descend and **overlap** — the DECISION converges. Right: the mean
  marginal-ROAS 90% CI width for both; the wrong (blue) line stays **below** the
  correct (green) and the gap does not close — narrow-AND-wrong = overconfident.

The single, honest idea: the loop reaches the RIGHT DECISION on a WRONG model while
the curve and the interval it would report stay confidently wrong.

Env knobs: MIS_N_WAVES(7), MIS_WRONG("hill"|"logistic", default "hill" — the mild
headline case with the clean R̂≈1.5 tell; logistic gives a more dramatic curve bias
but no clean R-hat tell), MIS_WARM_WRONG(300), MIS_WARM_MIX(500), MIS_N_GEO(72),
MIS_T_TEST(10), MIS_NOISE(0.4), MIS_DELTA(0.6), MIS_RED_OVERLAY(1).
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

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def _f(name, default):
    return float(os.environ.get(name, default))


def _i(name, default):
    return int(os.environ.get(name, default))


# ── decision + design constants (match notebook §14) ─────────────────────────
CENTER = np.full(4, 0.7)
B, VALUE = 3.2, 5.0
PROBE_LO, PROBE_HI = CENTER[0] * 0.4, CENTER[0] * 1.6  # ~[0.28, 1.12]
FOCAL = 1  # Pulse — the strongly two-phase channel

N_WAVES = _i("MIS_N_WAVES", 7)
N_GEO = _i("MIS_N_GEO", 72)
T_PRE = _i("MIS_T_PRE", 6)
T_TEST = _i("MIS_T_TEST", 10)
NOISE = _f("MIS_NOISE", 0.4)
DELTA = _f("MIS_DELTA", 0.6)
WARM_MIX = _i("MIS_WARM_MIX", 500)
WARM_WRONG = _i("MIS_WARM_WRONG", 300)
WRONG = os.environ.get("MIS_WRONG", "hill")  # the misspecified family
RED_OVERLAY = _i("MIS_RED_OVERLAY", 1)

CACHE = f"/tmp/_mis_anim_cache_{WRONG}.npz"

# ── the true world (a real analyst never sees this) ──────────────────────────
world_mix = cl.make_world_hill_mixture(seed=0)
GRID = np.linspace(0.0, 1.5, 80)

COL = {"correct": "#5a8a5a", "wrong": "#3a6ea5", "truth": "#111111", "red": "#d1495b"}
RGB = {"correct": "90,138,90", "wrong": "58,110,165"}
WRONG_LABEL = "logistic" if WRONG == "logistic" else "single Hill"


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


def true_mroas(a):
    a = np.asarray(a, float)
    eps = 1e-3
    r0 = float(world_mix.response_mean(a[None, :])[0])
    out = np.empty(a.size)
    for c in range(a.size):
        ap = a.copy()
        ap[c] += eps
        out[c] = VALUE * (float(world_mix.response_mean(ap[None, :])[0]) - r0) / eps
    return out


# ── the two accumulating loops ───────────────────────────────────────────────
TRUE_CURVE = true_anchored(FOCAL, GRID, CENTER)
_TRUE_ALLOC, TRUE_PROFIT = cl.world_optimal_allocation(
    world_mix, B, VALUE, mode="fixed"
)


def _run_family(activation: str, nw: int) -> dict:
    """One trust-region loop, capturing the Pulse band + decision/honesty metrics
    per wave. Shares wave 0 (seed=1) across families so early frames compare."""
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

    mean, lo, hi, gap, ciw, cov, rhat, rows = [], [], [], [], [], [], [], []
    for wave in range(N_WAVES):
        state.fit(num_warmup=nw, num_samples=nw, num_chains=2, seed=wave)
        post = state.posterior
        rec = state.recommend(q=200)
        gap.append(100.0 * (TRUE_PROFIT - prof_true(rec)) / abs(TRUE_PROFIT))
        cur = anchored_slice(post, FOCAL, GRID, CENTER)
        mean.append(cur.mean(0))
        lo.append(np.percentile(cur, 5, 0))
        hi.append(np.percentile(cur, 95, 0))
        _, _, mr = cl.marginal_roas(post, rec, VALUE, q=200)
        mlo, mhi = np.percentile(mr, 5, 0), np.percentile(mr, 95, 0)
        tmr = true_mroas(rec)
        width = mhi - mlo
        fin = np.isfinite(width)
        ciw.append(float(np.mean(width[fin])) if fin.any() else np.nan)
        cov.append(int(np.sum((mlo <= tmr) & (tmr <= mhi) & fin)))
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
        "ciw": np.array(ciw),
        "cov": np.array(cov),
        "rhat": np.array(rhat),
        "rows": np.array(rows),
    }


def compute() -> dict:
    correct = _run_family("hill_mixture", WARM_MIX)
    wrong = _run_family(WRONG, WARM_WRONG)
    # red mask: where the true curve leaves the WRONG family's band, per wave
    red = np.where(
        (TRUE_CURVE[None, :] < wrong["lo"]) | (TRUE_CURVE[None, :] > wrong["hi"]),
        np.tile(TRUE_CURVE, (N_WAVES, 1)),
        np.nan,
    )
    # global fixed ranges (learning must read as band-shrink, never axis rescale)
    all_lo = np.concatenate(
        [correct["lo"].ravel(), wrong["lo"].ravel(), [0.0], TRUE_CURVE]
    )
    all_hi = np.concatenate(
        [correct["hi"].ravel(), wrong["hi"].ravel(), [0.0], TRUE_CURVE]
    )
    pad = 0.05 * (all_hi.max() - all_lo.min())
    y_curve = (float(all_lo.min() - pad), float(all_hi.max() + pad))
    gap_max = float(max(correct["gap"].max(), wrong["gap"].max()))
    ciw_max = float(max(np.nanmax(correct["ciw"]), np.nanmax(wrong["ciw"])))
    return {
        "grid": GRID,
        "truth_curve": TRUE_CURVE,
        "red": red,
        "c_mean": correct["mean"],
        "c_lo": correct["lo"],
        "c_hi": correct["hi"],
        "w_mean": wrong["mean"],
        "w_lo": wrong["lo"],
        "w_hi": wrong["hi"],
        "c_gap": correct["gap"],
        "w_gap": wrong["gap"],
        "c_ciw": correct["ciw"],
        "w_ciw": wrong["ciw"],
        "c_cov": correct["cov"],
        "w_cov": wrong["cov"],
        "c_rhat": correct["rhat"],
        "w_rhat": wrong["rhat"],
        "c_rows": correct["rows"],
        "w_rows": wrong["rows"],
        "y_curve": np.array(y_curve),
        "gap_max": np.array([gap_max]),
        "ciw_max": np.array([ciw_max]),
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
            fillcolor=f"rgba({rgb},0.14)",
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
    ciw_max = float(d["ciw_max"][0])
    waves = np.arange(w + 1)

    fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=[0.60, 0.40],
        vertical_spacing=0.16,
        horizontal_spacing=0.12,
        subplot_titles=(
            "Curve you fit — correct family (mixture)",
            f"Curve you fit — WRONG family ({WRONG_LABEL})",
            "Decision — profit gap vs truth-optimal (both families)",
            "Honesty — marginal-ROAS 90% CI width (both families)",
        ),
    )
    # resize ONLY the four subplot titles, before any manual annotation
    for a in fig.layout.annotations[:4]:
        a.font.size = 12.5

    # ---- top-left: correct family ----
    fig.add_vrect(
        x0=PROBE_LO,
        x1=PROBE_HI,
        fillcolor="rgba(120,120,120,0.07)",
        line_width=0,
        row=1,
        col=1,
    )
    _band(fig, grid, d["c_lo"][w], d["c_hi"][w], RGB["correct"], 1, 1)
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=truth,
            mode="lines",
            line=dict(color=COL["truth"], width=3.3),
            name="true (mixture)",
            legendgroup="truth",
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=d["c_mean"][w],
            mode="lines",
            line=dict(color=COL["correct"], width=2.2),
            name="fit: mixture (correct)",
            legendgroup="correct",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # ---- top-right: wrong family ----
    fig.add_vrect(
        x0=PROBE_LO,
        x1=PROBE_HI,
        fillcolor="rgba(120,120,120,0.07)",
        line_width=0,
        row=1,
        col=2,
    )
    _band(fig, grid, d["w_lo"][w], d["w_hi"][w], RGB["wrong"], 1, 2)
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
            y=d["w_mean"][w],
            mode="lines",
            line=dict(color=COL["wrong"], width=2.2),
            name=f"fit: {WRONG_LABEL} (wrong)",
            legendgroup="wrong",
            showlegend=True,
        ),
        row=1,
        col=2,
    )
    if RED_OVERLAY:
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

    # ---- bottom-left: profit gap, both families (decision converges) ----
    for key, fam, rgbkey in [
        ("c_gap", "correct", "correct"),
        ("w_gap", "wrong", "wrong"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=waves,
                y=d[key][: w + 1],
                mode="lines+markers",
                line=dict(color=COL[rgbkey], width=2.4),
                marker=dict(size=7),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    # ---- bottom-right: CI width, both families (honesty diverges) ----
    for key, rgbkey in [("c_ciw", "correct"), ("w_ciw", "wrong")]:
        fig.add_trace(
            go.Scatter(
                x=waves,
                y=d[key][: w + 1],
                mode="lines+markers",
                line=dict(color=COL[rgbkey], width=2.4),
                marker=dict(size=7),
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    # ---- corner honesty badges (top-left interior, provably empty) ----
    for col, pre in [(1, "c"), (2, "w")]:
        xr = "x domain" if col == 1 else "x2 domain"
        yr = "y domain" if col == 1 else "y2 domain"
        fam = "correct" if col == 1 else "wrong"
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
                f"mROAS cov {int(d[pre + '_cov'][w])}/4<br>"
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
    fig.update_yaxes(
        title_text="mROAS 90% CI width", range=[0, ciw_max * 1.1], row=2, col=2
    )

    gw, gc = float(d["w_gap"][w]), float(d["c_gap"][w])
    rw, rc = float(d["w_rhat"][w]), float(d["c_rhat"][w])
    rows = int(d["w_rows"][w])
    fig.update_layout(
        width=1240,
        height=820,
        margin=dict(l=72, r=30, t=128, b=122),
        title=dict(
            text=(
                "<b>The loop learns even though the response family is wrong</b><br>"
                f"<sup>Wave {w} · {rows} geo-weeks · fitting a {WRONG_LABEL} to two-Hill "
                f"truth · gap wrong {gw:.1f}% vs correct {gc:.1f}% · "
                f"R̂ {_rhat_str(rw)} (wrong) / {_rhat_str(rc)} (correct)</sup>"
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
    frames_dir = os.path.join(OUTDIR, f"_mis_frames_{WRONG}")
    os.makedirs(frames_dir, exist_ok=True)
    imgs = []
    for w in range(N_WAVES):
        fig = frame_figure(d, w)
        p = os.path.join(frames_dir, f"f{w:02d}.png")
        fig.write_image(p, width=1240, height=820, scale=2)
        imgs.append(Image.open(p).convert("RGB"))
    suf = "" if WRONG == "hill" else f"_{WRONG}"
    gif = os.path.join(OUTDIR, f"continuous_learning_misspecification{suf}.gif")
    per = max(650, int(11000 / N_WAVES))
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
        "wrong_family": WRONG,
        "n_waves": int(N_WAVES),
        "n_geo": int(N_GEO),
        "t_test": int(T_TEST),
        "noise": float(NOISE),
        "gap_wrong": [round(float(x), 2) for x in d["w_gap"]],
        "gap_correct": [round(float(x), 2) for x in d["c_gap"]],
        "ciw_wrong": [round(float(x), 3) for x in d["w_ciw"]],
        "ciw_correct": [round(float(x), 3) for x in d["c_ciw"]],
        "cov_wrong": [int(x) for x in d["w_cov"]],
        "cov_correct": [int(x) for x in d["c_cov"]],
        "rhat_wrong": [round(float(x), 2) for x in d["w_rhat"]],
        "rhat_correct": [round(float(x), 2) for x in d["c_rhat"]],
    }


def main() -> None:
    d = _load()
    gif = build_gif(d)
    m = _metrics(d)
    with open(os.path.join(OUTDIR, f"_mis_metrics_{WRONG}.json"), "w") as fh:
        json.dump(m, fh, indent=2)
    print(f"wrong family = {WRONG_LABEL} | waves={N_WAVES} n_geo={N_GEO}")
    print(f"profit gap wrong  : {m['gap_wrong']}")
    print(f"profit gap correct: {m['gap_correct']}")
    print(f"CI width wrong    : {m['ciw_wrong']}")
    print(f"CI width correct  : {m['ciw_correct']}")
    print(f"R-hat wrong       : {m['rhat_wrong']}")
    print("wrote", gif)


if __name__ == "__main__":
    main()
