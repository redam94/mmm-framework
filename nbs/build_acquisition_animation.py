"""Animate how the acquisition & uncertainty surfaces update with each test.

    PYTHONPATH=.. uv run --with kaleido python build_acquisition_animation.py

Runs the continuous-learning loop wave by wave. After each designed wave is
folded in and the surface refit, it re-evaluates — over a fixed 2-D allocation
slice (Pulse × Orbit, the other channels held at the true optimum) — the
posterior-mean profit, the uncertainty ``value·SD[R]``, and the UCB acquisition,
and it **tracks the experiment history**:

* **which parameters were searched** — the probed central-composite cells (in the
  slice) are overlaid on the uncertainty panel, together with the recommendation
  trajectory (how the trust region moved), and
* **what each experiment measured** — the observed incremental KPI per wave with
  its observation uncertainty (±SE), accumulated in a readout panel.

The story: as tests accumulate the **uncertainty collapses** where the cells were
probed, the mean profit sharpens toward the truth, and the **acquisition optimum
(◆) converges onto the exploit optimum (★)** and the true optimum (gold ★) — while
the noisy per-wave readouts pin the surface down.

Many small, noisy waves make the learning visible. Tunable via env vars
(defaults below) so several cadences can be compared:

    ACQ_N_WAVES ACQ_T_TEST ACQ_N_GEO ACQ_NOISE ACQ_T_PRE ACQ_KAPPA ACQ_GRID_N
    ACQ_GRID_HI ACQ_VALUE ACQ_TAG   (ACQ_TAG names a side-by-side variant)

Colour ranges are fixed across frames (so the shrinking uncertainty is visible,
not renormalised away). Layout is deliberate — gutter colorbars *without titles*
(a title juts right into the next panel's y-axis label), the y-axis labelled only
on the leftmost panel, one horizontal legend below the panels, contour labels off
— so nothing overlaps in any frame.
"""

from __future__ import annotations

import json
import os

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image
from plotly.subplots import make_subplots

import mmm_framework.continuous_learning as cl
from build_acquisition_viz import _argmax2d

pio.templates.default = "plotly_white"

OUTDIR = os.path.join(os.path.dirname(__file__), "artifacts")


def _f(name, default):
    return float(os.environ.get(name, default))


def _i(name, default):
    return int(os.environ.get(name, default))


# Defaults: many small, noisy waves so the update-by-update learning is visible.
VALUE = _f("ACQ_VALUE", 2.5)
KAPPA = _f("ACQ_KAPPA", 1.5)
CAP = 3.4
SLICE = (1, 2)  # Pulse × Orbit
SCOLS = list(SLICE)
GRID_N = _i("ACQ_GRID_N", 60)
GRID_HI = _f("ACQ_GRID_HI", 3.4)
N_WAVES = _i("ACQ_N_WAVES", 18)  # many frames -> many small updates
N_GEO = _i("ACQ_N_GEO", 34)  # few geos/wave -> slow, gentle convergence
T_PRE = _i("ACQ_T_PRE", 4)
T_TEST = _i("ACQ_T_TEST", 2)  # 2-week window -> a small data increment per wave
NOISE = _f("ACQ_NOISE", 1.0)  # high observation noise -> more variance / jitter
DELTA = _f("ACQ_DELTA", 0.6)
# Fewer probed cells per wave -> less identification per wave -> the trust region
# keeps hunting (more movement, less of a settled/converged state).
#   "slice" = probe only the slice pair (15 cells); "none" = 13; "all" = 25.
PROBE = os.environ.get("ACQ_PROBE", "slice")
WORLD = os.environ.get("ACQ_WORLD", "easy")  # "easy" | "hard"
TAG = os.environ.get("ACQ_TAG", "")

_SUF = f"_{TAG}" if TAG else ""
CACHE = f"/tmp/_acq_anim_cache_v3{_SUF}.npz"


def _hard_world():
    """A deliberately harder world: strong audience-overlap cannibalization
    between the two slice channels (Pulse × Orbit) gives the response surface a
    ridge — you cannot fund both — so the optimum is a narrow trade-off, the
    main effects are close (hard to rank), and saturation is slow. Learning it
    from small noisy waves is genuinely hard."""
    ch = ["Chatter", "Pulse", "Orbit", "Vibe"]
    pairs = cl.default_pairs(4)
    beta = np.array([1.7, 1.55, 1.45, 1.2])  # weak, close -> hard to rank
    kappa = np.array([1.0, 1.1, 1.0, 1.2])  # slow saturation
    alpha = np.array([2.4, 2.2, 2.6, 1.6])
    lookup = {
        (1, 2): -1.3,
        (0, 1): -0.5,
        (2, 3): 0.35,
    }  # strong Pulse×Orbit cannibalization
    gp = np.array([lookup.get(p, 0.0) for p in pairs])
    return cl.TrueWorld(
        beta=beta,
        kappa=kappa,
        alpha=alpha,
        gamma_pairs=gp,
        channels=ch,
        pairs=pairs,
        a_level=4.0,
        sigma_a=1.0,
    )


def _make_world():
    if WORLD == "hard":
        return _hard_world()
    if WORLD == "logistic":
        return cl.make_world_logistic(seed=0)  # exponential-saturation activation
    return cl.make_world(seed=0)


def _probe_pairs(world):
    if PROBE == "all":
        return world.pairs
    if PROBE == "none":
        return []
    return [(SLICE[0], SLICE[1])]  # the slice pair only -> a leaner design


def _surfaces(post, held, gg):
    """Mean-profit, uncertainty and UCB grids + the exploit/acquisition optima."""
    i, j = SLICE
    gi, gj = np.meshgrid(gg, gg)
    s = np.repeat(np.asarray(held, float)[None, :], gi.size, axis=0)
    s[:, i] = gi.ravel()
    s[:, j] = gj.ravel()
    d = min(250, post.n_draws)
    idx = np.linspace(0, post.n_draws - 1, d).astype(int)
    # activation-agnostic: works for a Hill posterior or a logistic one
    r = cl.response_grid(post, s, idx)  # (G, D)
    mu = (VALUE * r.mean(1) - s.sum(1)).reshape(GRID_N, GRID_N)
    sg = (VALUE * r.std(1)).reshape(GRID_N, GRID_N)
    ucb = mu + KAPPA * sg
    return mu, sg, ucb, _argmax2d(mu, gg), _argmax2d(ucb, gg)


def _readout(y, geo_idx, baselines):
    """Observed incremental KPI (vs each geo's pre-period baseline) + its SE."""
    inc = np.asarray(y, float) - baselines[np.asarray(geo_idx, int)]
    return float(inc.mean()), float(inc.std(ddof=1) / np.sqrt(len(inc)))


def compute() -> dict:
    """Run the loop, capturing the surfaces AND the experiment history each wave."""
    world = _make_world()
    center = np.full(4, 0.7)
    true_alloc, _ = cl.world_optimal_allocation(world, 3.2, VALUE, mode="free", cap=CAP)
    held = true_alloc.copy()  # hold the off-slice channels at the truth
    gg = np.linspace(0.02, GRID_HI, GRID_N)

    state = cl.LearningState(
        channels=world.channels,
        center=center,
        B=3.2,
        value=VALUE,
        pairs=world.pairs,
        pair_signs=cl.PAIR_SIGNS_EXAMPLE,
        activation=world.activation,  # fit the same family the world uses
        mode="free",
        cap=CAP,
    )
    probe = _probe_pairs(world)
    wave0 = cl.simulate_panel(
        world,
        center,
        n_geo=N_GEO,
        t_pre=T_PRE,
        t_test=T_TEST,
        delta=DELTA,
        probe_pairs=probe,
        noise=NOISE,
        seed=0,
    )
    a_geo = wave0["a_geo"]
    # per-geo pre-period baseline for the observed-incremental readout
    n_pre = T_PRE * N_GEO
    y0, g0 = wave0["y"], wave0["geo_idx"]
    baselines = np.array([y0[:n_pre][g0[:n_pre] == g].mean() for g in range(N_GEO)])

    state.ingest(wave0)
    state.fit(num_warmup=300, num_samples=300, num_chains=2, seed=0)

    # experiment history, one entry per data-wave (aligned to the frame index)
    designs = [np.asarray(wave0["design"])[:, SCOLS]]
    centers = [center[SCOLS]]
    rmean, rse = [], []
    rm, rs = _readout(y0[n_pre:], g0[n_pre:], baselines)
    rmean.append(rm)
    rse.append(rs)

    pis, uncs, ucbs, exs, acs, rows = [], [], [], [], [], []
    for w in range(N_WAVES):
        mu, sg, ucb, ex, ac = _surfaces(state.posterior, held, gg)
        pis.append(mu)
        uncs.append(sg)
        ucbs.append(ucb)
        exs.append(ex)
        acs.append(ac)
        rows.append(int(state.data["spend"].shape[0]))
        if w < N_WAVES - 1:
            rec = state.recommend(q=150)
            design = cl.central_composite(rec, DELTA, probe)
            wn = cl.simulate_wave(
                world, design, a_geo, t_test=T_TEST, center=rec, noise=NOISE, seed=w + 1
            )
            designs.append(np.asarray(design)[:, SCOLS])
            centers.append(np.asarray(rec)[SCOLS])
            rm, rs = _readout(wn["y"], wn["geo_idx"], baselines)
            rmean.append(rm)
            rse.append(rs)
            state.recenter(rec)
            state.ingest(wn)
            state.fit(num_warmup=300, num_samples=300, num_chains=2, seed=0)

    return {
        "gg": gg,
        "pi": np.array(pis),
        "unc": np.array(uncs),
        "ucb": np.array(ucbs),
        "ex": np.array(exs),
        "ac": np.array(acs),
        "rows": np.array(rows),
        "true_opt": np.array([true_alloc[SLICE[0]], true_alloc[SLICE[1]]]),
        "channels": np.array(world.channels),
        "designs": np.array(designs),
        "centers": np.array(centers),
        "readout_mean": np.array(rmean),
        "readout_se": np.array(rse),
    }


def _load() -> dict:
    if not os.path.exists(CACHE):
        np.savez(CACHE, **compute())
    return dict(np.load(CACHE, allow_pickle=True))


def _marker(x, y, name, symbol, fill, edge, show, size=15):
    return go.Scatter(
        x=[x] if np.isscalar(x) else x,
        y=[y] if np.isscalar(y) else y,
        mode="markers",
        name=name,
        legendgroup=name,
        showlegend=show,
        marker=dict(
            symbol=symbol, size=size, color=fill, line=dict(color=edge, width=2.2)
        ),
    )


def frame_figure(d: dict, w: int, ranges: dict) -> go.Figure:
    gg = d["gg"]
    ch = [str(x) for x in d["channels"]]
    xlab, ylab = ch[SLICE[0]], ch[SLICE[1]]
    tx, ty = d["true_opt"]
    ex, ac = d["ex"][w], d["ac"][w]
    dist = float(np.hypot(ac[0] - tx, ac[1] - ty))

    fig = make_subplots(
        rows=2,
        cols=3,
        row_heights=[0.64, 0.36],
        horizontal_spacing=0.12,
        vertical_spacing=0.17,
        specs=[[{}, {}, {}], [{"colspan": 3}, None, None]],
        subplot_titles=(
            "Posterior-mean profit  π",
            "Uncertainty  σ  +  parameters searched",
            "Acquisition  UCB = π + κσ",
            "Experiment readouts — observed incremental KPI ± SE (per wave)",
        ),
    )

    def contour(z, scale, zr, cbx):
        return go.Contour(
            x=gg,
            y=gg,
            z=z,
            colorscale=scale,
            zmin=zr[0],
            zmax=zr[1],
            ncontours=18,
            line=dict(width=0),
            contours=dict(showlabels=False),
            colorbar=dict(x=cbx, y=0.80, len=0.40, thickness=10, tickfont=dict(size=8)),
        )

    fig.add_trace(contour(d["pi"][w], "Viridis", ranges["pi"], 0.263), row=1, col=1)
    fig.add_trace(contour(d["unc"][w], "Blues", ranges["unc"], 0.637), row=1, col=2)
    fig.add_trace(contour(d["ucb"][w], "Plasma", ranges["ucb"], 1.005), row=1, col=3)

    # --- parameters searched (uncertainty panel): trajectory + current cells ---
    ctr = d["centers"][: w + 1]
    fig.add_trace(
        go.Scatter(
            x=ctr[:, 0],
            y=ctr[:, 1],
            mode="lines+markers",
            name="search trajectory",
            legendgroup="traj",
            showlegend=True,
            line=dict(color="rgba(40,40,40,0.55)", width=1.4),
            marker=dict(color="rgba(40,40,40,0.55)", size=5),
        ),
        row=1,
        col=2,
    )
    cells = d["designs"][w]
    fig.add_trace(
        go.Scatter(
            x=cells[:, 0],
            y=cells[:, 1],
            mode="markers",
            name="cells probed (this wave)",
            legendgroup="cells",
            showlegend=True,
            marker=dict(
                symbol="circle-open",
                color="#ff8c1a",
                size=8,
                line=dict(width=2, color="#ff8c1a"),
            ),
        ),
        row=1,
        col=2,
    )

    # --- optima on every surface panel ---
    for c in (1, 2, 3):
        first = c == 1
        fig.add_trace(
            _marker(tx, ty, "true optimum", "star", "#ffd21e", "black", first),
            row=1,
            col=c,
        )
        fig.add_trace(
            _marker(
                ex[0], ex[1], "exploit optimum (mean)", "star", "white", "black", first
            ),
            row=1,
            col=c,
        )
        fig.add_trace(
            _marker(
                ac[0],
                ac[1],
                "acquisition optimum (UCB)",
                "diamond",
                "black",
                "white",
                first,
            ),
            row=1,
            col=c,
        )
        fig.update_xaxes(title_text=xlab, row=1, col=c, range=[gg[0], gg[-1]])
        fig.update_yaxes(
            title_text=(ylab if c == 1 else ""), row=1, col=c, range=[gg[0], gg[-1]]
        )

    # --- experiment readouts (observed incremental ± SE), accumulating ---
    waves = np.arange(w + 1)
    rm = d["readout_mean"][: w + 1]
    rs = d["readout_se"][: w + 1]
    fig.add_trace(
        go.Scatter(
            x=waves,
            y=rm,
            error_y=dict(
                type="data",
                array=rs,
                visible=True,
                color="rgba(59,111,182,0.55)",
                thickness=1.4,
                width=4,
            ),
            mode="lines+markers",
            name="observed incremental ± SE",
            legendgroup="readout",
            showlegend=True,
            line=dict(color="#3b6fb6", width=1.8),
            marker=dict(color="#3b6fb6", size=7),
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(
        title_text="wave (experiment round)",
        row=2,
        col=1,
        range=[-0.5, len(d["rows"]) - 0.5],
        dtick=2,
    )
    fig.update_yaxes(title_text="observed incremental KPI", row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        width=1180,
        height=760,
        margin=dict(l=70, r=80, t=110, b=95),
        title=dict(
            text="<b>Acquisition, uncertainty &amp; experiment history — updating with each test"
            + (" · HARD problem" if WORLD == "hard" else "")
            + (" · logistic saturation" if WORLD == "logistic" else "")
            + "</b>"
            f"<br><sup>Wave {w} · {int(d['rows'][w])} geo-weeks · "
            f"acquisition→truth distance {dist:.2f}"
            + (" · strong Pulse×Orbit cannibalization" if WORLD == "hard" else "")
            + (" · f(s)=1−exp(−λs), not Hill" if WORLD == "logistic" else "")
            + "</sup>",
            x=0.5,
            xanchor="center",
            y=0.975,
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.09,
            xanchor="center",
            x=0.5,
            font=dict(size=10.5),
        ),
    )
    for ann in fig.layout.annotations:
        ann.font.size = 12.5
    return fig


def _metrics(d: dict) -> dict:
    ac, topt = d["ac"], d["true_opt"]
    acq_truth = [
        float(np.hypot(ac[w][0] - topt[0], ac[w][1] - topt[1])) for w in range(len(ac))
    ]
    jitter = [0.0] + [float(np.hypot(*(ac[w] - ac[w - 1]))) for w in range(1, len(ac))]
    mean_sigma = [float(d["unc"][w].mean()) for w in range(len(ac))]
    rows = [int(r) for r in d["rows"]]
    return {
        "tag": TAG,
        "n_waves": int(N_WAVES),
        "n_geo": int(N_GEO),
        "t_test": int(T_TEST),
        "noise": float(NOISE),
        "rows_per_wave": [int(rows[0])] + [int(x) for x in np.diff(rows)],
        "rows": rows,
        "acq_truth": acq_truth,
        "opt_jitter": jitter,
        "mean_sigma": mean_sigma,
        "readout_mean": [float(x) for x in d["readout_mean"]],
        "readout_se": [float(x) for x in d["readout_se"]],
        "sigma_shrink": float(mean_sigma[0] - mean_sigma[-1]),
        "mean_jitter": float(np.mean(jitter[1:])) if len(jitter) > 1 else 0.0,
        "final_acq_truth": acq_truth[-1],
    }


def build_gif(d: dict) -> str:
    ranges = {
        "pi": (float(d["pi"].min()), float(d["pi"].max())),
        "unc": (
            0.0,
            float(d["unc"][0].max()),
        ),  # frame-0 is the widest -> shrinkage shows
        "ucb": (float(d["ucb"].min()), float(d["ucb"].max())),
    }
    os.makedirs(OUTDIR, exist_ok=True)
    frames_dir = os.path.join(OUTDIR, f"_frames{_SUF}")
    os.makedirs(frames_dir, exist_ok=True)
    imgs = []
    n = len(d["rows"])
    for w in range(n):
        fig = frame_figure(d, w, ranges)
        p = os.path.join(frames_dir, f"f{w:02d}.png")
        fig.write_image(p, width=1180, height=760, scale=2)
        imgs.append(Image.open(p).convert("RGB"))
    gif = os.path.join(OUTDIR, f"continuous_learning_acquisition{_SUF}.gif")
    per = max(500, int(10000 / n))  # keep the whole loop ~10s
    durations = [per] * (n - 1) + [2800]  # hold the converged frame
    imgs[0].save(
        gif,
        save_all=True,
        append_images=imgs[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    return gif


def main() -> None:
    d = _load()
    gif = build_gif(d)
    m = _metrics(d)
    with open(os.path.join(OUTDIR, f"_acq_metrics{_SUF}.json"), "w") as fh:
        json.dump(m, fh, indent=2)
    print(
        f"tag={TAG or '(default)'} waves={m['n_waves']} n_geo={m['n_geo']} "
        f"t_test={m['t_test']} noise={m['noise']}"
    )
    print(
        f"rows/wave (increment): first {m['rows_per_wave'][0]}, "
        f"then ~{int(np.median(m['rows_per_wave'][1:]))}"
    )
    print(
        f"σ shrink {m['mean_sigma'][0]:.2f} -> {m['mean_sigma'][-1]:.2f} "
        f"| mean optimum jitter {m['mean_jitter']:.3f} "
        f"| final acq→truth {m['final_acq_truth']:.2f}"
    )
    print(
        f"readouts: {np.round(m['readout_mean'], 2)}  ±  {np.round(m['readout_se'], 2)}"
    )
    print("wrote", gif)


if __name__ == "__main__":
    main()
