"""Animate INFORMATION DISCOUNTING on a world whose media behaviour DRIFTS.

    uv run --with plotly --with kaleido --with pillow python builders/build_discount_animation.py

A sibling of ``build_spline_animation.py``. That one held the truth FIXED and
showed the shape-agnostic spline converging onto it; this one moves the truth —
``drift_world`` applies a geometric random walk to the response surface between
waves (audiences shift, creative fatigues) — and shows why the static
accumulate-everything fit goes confidently stale while the **information
discount** (``discount_half_life``) tracks the moving target.

Both loops fit the SAME accumulating data stream with the monotone I-spline
under the **P-spline shrinkage prior** (``spline_prior="pspline"`` — the
learned per-channel smoothness that lets the basis earn its flexibility), and
differ in exactly one knob:

* **discounted** — ``discount_half_life=DISC_H`` weeks: every row's likelihood
  decays on the Eq. 22 clock (old rows' noise inflated by
  ``exp(0.5 * lambda * age)``), so effective sample size SATURATES and the
  band keeps an honest floor.
* **static** — ``discount_half_life=None``: the historical fit; every row
  weighs the same forever, so the band shrinks like 1/sqrt(rows) while the
  truth walks away — narrow and wrong, the temporal twin of the
  misspecification failure.

A 2×2 GIF (all Scatter traces):

* Top row — *the curve you believe today* vs *the curve that is true today*
  (black, moving every frame; the dashed grey curve is where behaviour
  started). Left: the discounted fit's 90% band tracking the moving truth.
  Right: the static fit tightening onto a stale average, red wherever
  today's truth escapes its band.
* Bottom row — trackers. Left: profit gap vs TODAY's optimum for both.
  Right: coverage of today's curve by each band.

Coverage is judged over the PROBED spend window (where the designed cells
earned the fit an opinion) — both fits extrapolate outside it and neither
band is a promise there.

Env knobs: DIS_N_WAVES(6), DIS_WARM(400), DIS_N_GEO(72), DIS_T_PRE(6),
DIS_T_TEST(10), DIS_NOISE(0.4), DIS_DELTA(0.6), DIS_DRIFT(0.09),
DIS_HALF_LIFE(6).
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


# ── decision + design constants (match notebook §14/§15/§16) ─────────────────
CENTER = np.full(4, 0.7)
B, VALUE = 3.2, 5.0
PROBE_LO, PROBE_HI = CENTER[0] * 0.4, CENTER[0] * 1.6
FOCAL = 1  # Pulse

N_WAVES = _i("DIS_N_WAVES", 6)
N_GEO = _i("DIS_N_GEO", 72)
T_PRE = _i("DIS_T_PRE", 6)
T_TEST = _i("DIS_T_TEST", 10)
NOISE = _f("DIS_NOISE", 0.4)
DELTA = _f("DIS_DELTA", 0.6)
WARM = _i("DIS_WARM", 400)
DRIFT = _f("DIS_DRIFT", 0.09)  # per-wave geometric drift sd on beta + shape
DISC_H = _f("DIS_HALF_LIFE", 6.0)  # weeks: old regime ~gone one wave later

CACHE = f"/tmp/_discount_anim_cache_{N_WAVES}_{DRIFT:g}_{DISC_H:g}.npz"

COL = {
    "disc": "#2e7d6b",  # discounted — teal
    "stat": "#3a6ea5",  # static — blue
    "truth": "#111111",
    "start": "#9a9a9a",
    "red": "#d1495b",
}
RGB = {"disc": "46,125,107", "stat": "58,110,165"}

GRID = np.linspace(0.0, 1.5, 80)


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


def true_anchored(world, c, s_grid, x0):
    X = np.tile(x0, (s_grid.size, 1))
    X[:, c] = s_grid
    v = world.response_mean(X)
    return v - v[0]


def compute() -> dict:
    # the drifting truth: one geometric-random-walk step per wave
    worlds = [cl.make_world_hill_mixture(seed=0)]
    for w in range(1, N_WAVES):
        worlds.append(cl.drift_world(worlds[-1], rate=DRIFT, seed=100 + w))

    def new_state(h):
        return cl.LearningState(
            channels=worlds[0].channels,
            center=CENTER.copy(),
            B=B,
            value=VALUE,
            pairs=worlds[0].pairs,
            pair_signs=cl.PAIR_SIGNS_EXAMPLE,
            activation="monotone_spline",
            spline_prior="pspline",
            discount_half_life=h,
            mode="fixed",
        )

    st_disc, st_stat = new_state(DISC_H), new_state(None)

    w0 = cl.simulate_panel(
        worlds[0],
        CENTER,
        n_geo=N_GEO,
        t_pre=T_PRE,
        t_test=T_TEST,
        delta=DELTA,
        noise=NOISE,
        seed=1,
    )
    a_geo = np.asarray(w0["a_geo"])
    st_disc.ingest(w0)
    st_stat.ingest(w0)

    rec_disc = CENTER.copy()
    out = {
        nm: {
            "mean": [],
            "lo": [],
            "hi": [],
            "gap": [],
            "cover": [],
            "bandw": [],
            "rhat": [],
            "eff_rows": [],
        }
        for nm in ("disc", "stat")
    }
    truths, starts, rows_per_wave = [], [], []
    start_curve = true_anchored(worlds[0], FOCAL, GRID, CENTER)

    for wave in range(N_WAVES):
        world_now = worlds[wave]
        truth_now = true_anchored(world_now, FOCAL, GRID, CENTER)
        truths.append(truth_now)
        starts.append(start_curve)
        _opt, opt_profit = cl.world_optimal_allocation(
            world_now, B, VALUE, mode="fixed"
        )

        for nm, st in (("disc", st_disc), ("stat", st_stat)):
            st.fit(num_warmup=WARM, num_samples=WARM, num_chains=2, seed=wave)
            post = st.posterior
            rec = st.recommend(q=200)
            if nm == "disc":
                rec_disc = rec
            achieved = (
                VALUE * float(world_now.response_mean(np.asarray(rec)[None, :])[0]) - B
            )
            out[nm]["gap"].append(100.0 * (opt_profit - achieved) / abs(opt_profit))
            cur = anchored_slice(post, FOCAL, GRID, CENTER)
            m, l5, h95 = (
                cur.mean(0),
                np.percentile(cur, 5, 0),
                np.percentile(cur, 95, 0),
            )
            out[nm]["mean"].append(m)
            out[nm]["lo"].append(l5)
            out[nm]["hi"].append(h95)
            # coverage judged where the design gave the fit an opinion — the
            # probed spend window (both fits extrapolate outside it)
            pm = (GRID >= PROBE_LO) & (GRID <= PROBE_HI) & (GRID > GRID[0])
            inside = (truth_now[pm] >= l5[pm]) & (truth_now[pm] <= h95[pm])
            out[nm]["cover"].append(100.0 * float(np.mean(inside)))
            out[nm]["bandw"].append(float(np.mean(h95 - l5)))
            rh = post.diagnostics.get("max_rhat")
            out[nm]["rhat"].append(float(rh) if rh is not None else np.nan)
            d = post.diagnostics.get("discount")
            out[nm]["eff_rows"].append(
                float(d["effective_rows"]) if d else float(st.data["y"].shape[0])
            )
        rows_per_wave.append(int(st_disc.data["y"].shape[0]))

        if wave < N_WAVES - 1:
            # ONE shared data stream: the trust region follows the discounted
            # loop (the loop you would actually run); the next wave's outcomes
            # come from the NEXT (drifted) world over the SAME geos.
            design = cl.central_composite(rec_disc, DELTA, worlds[0].pairs)
            wn = cl.simulate_wave(
                worlds[wave + 1],
                design,
                a_geo,
                t_test=T_TEST,
                center=rec_disc,
                noise=NOISE,
                seed=2 + wave,
            )
            st_disc.recenter(rec_disc)
            st_stat.recenter(rec_disc)
            st_disc.ingest(wn)
            st_stat.ingest(wn)

    truths = np.array(truths)
    red = {
        nm: np.where(
            (truths < np.array(out[nm]["lo"])) | (truths > np.array(out[nm]["hi"])),
            truths,
            np.nan,
        )
        for nm in ("disc", "stat")
    }
    all_vals = np.concatenate(
        [
            np.array(out[nm][key]).ravel()
            for nm in ("disc", "stat")
            for key in ("lo", "hi")
        ]
        + [truths.ravel(), start_curve, [0.0]]
    )
    pad = 0.05 * (all_vals.max() - all_vals.min())
    y_curve = (float(all_vals.min() - pad), float(all_vals.max() + pad))
    gap_max = float(max(np.max(out["disc"]["gap"]), np.max(out["stat"]["gap"])))

    flat: dict[str, np.ndarray] = {
        "grid": GRID,
        "truths": truths,
        "start_curve": start_curve,
        "rows": np.array(rows_per_wave),
        "y_curve": np.array(y_curve),
        "gap_max": np.array([gap_max]),
        "red_disc": red["disc"],
        "red_stat": red["stat"],
    }
    for nm in ("disc", "stat"):
        for key, val in out[nm].items():
            flat[f"{nm}_{key}"] = np.array(val)
    return flat


def _load() -> dict:
    if not os.path.exists(CACHE):
        np.savez(CACHE, **compute())
    return dict(np.load(CACHE, allow_pickle=True))


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
    truth = d["truths"][w]
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
            f"Discounted — half-life {DISC_H:g} wk (old waves fade)",
            "Static — every wave weighs the same forever",
            "Decision — profit gap vs TODAY's optimum",
            "Honesty — % of TODAY's curve inside the band (probed range)",
        ),
    )
    for a in fig.layout.annotations[:4]:
        a.font.size = 12.5

    panels = [("disc", 1, "red_disc"), ("stat", 2, "red_stat")]
    for nm, col, red_key in panels:
        fig.add_vrect(
            x0=PROBE_LO,
            x1=PROBE_HI,
            fillcolor="rgba(120,120,120,0.07)",
            line_width=0,
            row=1,
            col=col,
        )
        _band(fig, grid, d[f"{nm}_lo"][w], d[f"{nm}_hi"][w], RGB[nm], 1, col)
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=d["start_curve"],
                mode="lines",
                line=dict(color=COL["start"], width=1.6, dash="dash"),
                name="where behaviour started (wave 0)",
                legendgroup="start",
                showlegend=(col == 1),
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=truth,
                mode="lines",
                line=dict(color=COL["truth"], width=3.3),
                name="true curve TODAY (drifting)",
                legendgroup="truth",
                showlegend=(col == 1),
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=d[f"{nm}_mean"][w],
                mode="lines",
                line=dict(color=COL[nm], width=2.2),
                name=(
                    "fit: P-spline + information discount"
                    if nm == "disc"
                    else "fit: P-spline, static (no discount)"
                ),
                legendgroup=nm,
                showlegend=True,
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=d[red_key][w],
                mode="lines",
                line=dict(color=COL["red"], width=3.2),
                connectgaps=False,
                name="today's truth outside fitted band",
                legendgroup="red",
                showlegend=(nm == "stat"),
            ),
            row=1,
            col=col,
        )

    # bottom trackers
    for key, nm in [("disc_gap", "disc"), ("stat_gap", "stat")]:
        fig.add_trace(
            go.Scatter(
                x=waves,
                y=d[key][: w + 1],
                mode="lines+markers",
                line=dict(color=COL[nm], width=2.4),
                marker=dict(size=7),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    for key, nm in [("disc_cover", "disc"), ("stat_cover", "stat")]:
        fig.add_trace(
            go.Scatter(
                x=waves,
                y=d[key][: w + 1],
                mode="lines+markers",
                line=dict(color=COL[nm], width=2.4),
                marker=dict(size=7),
                showlegend=False,
            ),
            row=2,
            col=2,
        )
    fig.add_hline(y=90.0, line=dict(color="gray", dash="dash"), row=2, col=2)

    # corner badges
    total_rows = int(d["rows"][w])
    for col, nm in [(1, "disc"), (2, "stat")]:
        xr = "x domain" if col == 1 else "x2 domain"
        yr = "y domain" if col == 1 else "y2 domain"
        eff = float(d[f"{nm}_eff_rows"][w])
        eff_line = (
            f"effective rows {eff:,.0f} / {total_rows:,}"
            if nm == "disc"
            else f"rows {total_rows:,} (all full weight)"
        )
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
                f"gap {d[nm + '_gap'][w]:.1f}%<br>"
                f"curve cover {d[nm + '_cover'][w]:.0f}%<br>"
                f"band width {d[nm + '_bandw'][w]:.2f} · "
                f"R̂ {_rhat_str(float(d[nm + '_rhat'][w]))}<br>"
                f"{eff_line}"
            ),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=COL[nm],
            borderwidth=1,
            borderpad=3,
        )

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
    for col in (1, 2):
        fig.update_xaxes(
            title_text="wave (experiment round)",
            range=[-0.4, N_WAVES - 0.6],
            dtick=1,
            row=2,
            col=col,
        )
    fig.update_yaxes(
        title_text="profit gap (%)", range=[0, gap_max * 1.1], row=2, col=1
    )
    fig.update_yaxes(
        title_text="today's truth inside band (%)", range=[0, 105], row=2, col=2
    )

    gd, gs = float(d["disc_gap"][w]), float(d["stat_gap"][w])
    cd, cs = float(d["disc_cover"][w]), float(d["stat_cover"][w])
    fig.update_layout(
        width=1240,
        height=820,
        margin=dict(l=72, r=30, t=128, b=122),
        title=dict(
            text=(
                "<b>When media behaviour drifts, discount old information — "
                "the static fit goes confidently stale</b><br>"
                f"<sup>Wave {w} · truth drifting ~{100 * DRIFT:.0f}%/wave · "
                f"{total_rows:,} geo-weeks accumulated · gap discounted "
                f"{gd:.1f}% vs static {gs:.1f}% · today's-curve cover "
                f"{cd:.0f}% vs {cs:.0f}%</sup>"
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
    frames_dir = os.path.join(OUTDIR, "_discount_frames")
    os.makedirs(frames_dir, exist_ok=True)
    imgs = []
    for w in range(N_WAVES):
        fig = frame_figure(d, w)
        p = os.path.join(frames_dir, f"f{w:02d}.png")
        fig.write_image(p, width=1240, height=820, scale=2)
        imgs.append(Image.open(p).convert("RGB"))
    gif = os.path.join(OUTDIR, "continuous_learning_discount.gif")
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
        "drift_rate": float(DRIFT),
        "half_life_weeks": float(DISC_H),
        "gap_discounted": [round(float(x), 2) for x in d["disc_gap"]],
        "gap_static": [round(float(x), 2) for x in d["stat_gap"]],
        "cover_discounted": [round(float(x), 1) for x in d["disc_cover"]],
        "cover_static": [round(float(x), 1) for x in d["stat_cover"]],
        "bandw_discounted": [round(float(x), 3) for x in d["disc_bandw"]],
        "bandw_static": [round(float(x), 3) for x in d["stat_bandw"]],
        "rhat_discounted": [round(float(x), 2) for x in d["disc_rhat"]],
        "rhat_static": [round(float(x), 2) for x in d["stat_rhat"]],
        "eff_rows_discounted": [round(float(x), 1) for x in d["disc_eff_rows"]],
        "rows": [int(x) for x in d["rows"]],
    }


def main() -> None:
    d = _load()
    gif = build_gif(d)
    m = _metrics(d)
    with open(os.path.join(OUTDIR, "_discount_metrics.json"), "w") as fh:
        json.dump(m, fh, indent=2)
    print(
        f"drift={DRIFT}/wave half_life={DISC_H}wk | waves={N_WAVES} "
        f"n_geo={N_GEO} noise={NOISE}"
    )
    print(f"gap discounted : {m['gap_discounted']}")
    print(f"gap static     : {m['gap_static']}")
    print(f"cover discounted: {m['cover_discounted']}")
    print(f"cover static    : {m['cover_static']}")
    print(f"band discounted : {m['bandw_discounted']}")
    print(f"band static     : {m['bandw_static']}")
    print(f"effective rows  : {m['eff_rows_discounted']} of {m['rows']}")
    print("wrote", gif)


if __name__ == "__main__":
    main()
