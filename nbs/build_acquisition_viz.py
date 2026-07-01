"""Render the acquisition-function + uncertainty-surface visualization.

    PYTHONPATH=.. uv run --with kaleido python build_acquisition_viz.py

Over a 2-D slice of the allocation space (Pulse × Orbit, the other channels
held at the recommendation), the continuous-learning posterior gives, at every
candidate spend point ``s``:

* the **posterior-mean profit**  ``pi(s) = value * E[R(s)] - 1^T s``,
* the **uncertainty surface**    ``sigma(s) = value * SD[R(s)]``  (where the
  model is least sure — highest away from the probed cells), and
* an **acquisition surface**     ``UCB(s) = pi(s) + kappa * sigma(s)`` that trades
  exploitation (high mean profit) against exploration (high uncertainty).

The exploit optimum (argmax of the mean) and the acquisition optimum (argmax of
UCB) differ: the acquisition point is pulled toward the higher-uncertainty
region — the next allocation worth probing. Writes a PNG and a standalone HTML
to ``nbs/artifacts/``.

Layout is deliberate: contour labels off (the colorbars carry the scale), one
horizontal legend below the panels (never over a title), and each colorbar
parked in a panel gutter so it never overlaps a neighbouring panel.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import mmm_framework.continuous_learning as cl
from mmm_framework.continuous_learning import surface

jax.config.update("jax_platform_name", "cpu")
pio.templates.default = "plotly_white"

CACHE = "/tmp/_acq_cache.npz"
OUTDIR = os.path.join(os.path.dirname(__file__), "artifacts")
VALUE = 2.5  # $/unit: low enough that the profit peaks in the interior
KAPPA = 1.5  # exploration weight in the UCB acquisition
SLICE = (1, 2)  # Pulse × Orbit (a positive-synergy pair)
GRID_N = 80
GRID_HI = 3.4  # wide enough that both π and UCB peak in the interior

_inc2 = jax.vmap(
    jax.vmap(surface.incremental, in_axes=(None, 0, 0, 0, 0)),
    in_axes=(0, None, None, None, None),
)


def compute() -> dict:
    """Fit a posterior and evaluate the three surfaces over the slice."""
    world = cl.make_world(seed=0)
    center = np.full(4, 0.7)
    data = cl.simulate_panel(
        world, center, n_geo=60, t_pre=6, t_test=8, delta=0.6, noise=0.5, seed=1
    )
    post = cl.fit(
        data,
        channels=world.channels,
        pair_signs=cl.PAIR_SIGNS_EXAMPLE,
        num_warmup=400,
        num_samples=400,
        num_chains=2,
        seed=0,
    )
    rec = cl.recommend_allocation(post, 3.2, VALUE, q=300, mode="fixed")

    i, j = SLICE
    g = np.linspace(0.02, GRID_HI, GRID_N)
    gi, gj = np.meshgrid(g, g)
    spend = np.repeat(rec[None, :], gi.size, axis=0)  # others held at rec
    spend[:, i] = gi.ravel()
    spend[:, j] = gj.ravel()

    d = min(300, post.n_draws)
    idx = np.linspace(0, post.n_draws - 1, d).astype(int)
    betas = post.samples["beta"][idx]
    kappas = post.samples["kappa"][idx]
    alphas = post.samples["alpha"][idx]
    gammas = np.stack([post.gamma_matrix(int(k)) for k in idx])

    r = np.asarray(
        _inc2(
            jnp.asarray(spend, float),
            jnp.asarray(betas, float),
            jnp.asarray(kappas, float),
            jnp.asarray(alphas, float),
            jnp.asarray(gammas, float),
        )
    )  # (G, D)
    mean_r = r.mean(1).reshape(GRID_N, GRID_N)
    std_r = r.std(1).reshape(GRID_N, GRID_N)
    spend_total = spend.sum(1).reshape(GRID_N, GRID_N)

    pi = VALUE * mean_r - spend_total  # posterior-mean profit
    unc = VALUE * std_r  # uncertainty surface
    ucb = pi + KAPPA * unc  # acquisition

    return {
        "g": g,
        "pi": pi,
        "unc": unc,
        "ucb": ucb,
        "channels": np.array(world.channels),
        "rec": rec,
        "i": i,
        "j": j,
    }


def _load() -> dict:
    if not os.path.exists(CACHE):
        np.savez(CACHE, **compute())
    return dict(np.load(CACHE, allow_pickle=True))


def _argmax2d(z, g):
    r, c = np.unravel_index(int(np.argmax(z)), z.shape)
    return g[c], g[r]  # (x = slice-i spend, y = slice-j spend)


def build_figure(d: dict) -> go.Figure:
    g = d["g"]
    channels = [str(x) for x in d["channels"]]
    xi, yi = int(d["i"]), int(d["j"])
    xlab, ylab = channels[xi], channels[yi]
    pi, unc, ucb = d["pi"], d["unc"], d["ucb"]

    ex_x, ex_y = _argmax2d(pi, g)  # exploit optimum (mean profit)
    ac_x, ac_y = _argmax2d(ucb, g)  # acquisition optimum (UCB)
    print(
        f"exploit optimum ({xlab},{ylab}) = ({ex_x:.2f}, {ex_y:.2f}) "
        f"[interior={g[0] < ex_x < g[-1] and g[0] < ex_y < g[-1]}]"
    )
    print(
        f"acquisition optimum       = ({ac_x:.2f}, {ac_y:.2f})  "
        f"shift = ({ac_x - ex_x:+.2f}, {ac_y - ex_y:+.2f})"
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.14,
        vertical_spacing=0.15,
        subplot_titles=(
            "Posterior-mean profit  π(s)",
            "Uncertainty surface  σ(s) = value·SD[R]",
            "Acquisition  UCB(s) = π + κ·σ",
            f"Slice at {xlab} = {ac_x:.2f}: explore vs exploit",
        ),
    )

    def contour(z, scale, cbx, cby, cbtitle):
        return go.Contour(
            x=g,
            y=g,
            z=z,
            colorscale=scale,
            ncontours=18,
            line=dict(width=0),
            contours=dict(showlabels=False),
            colorbar=dict(
                x=cbx,
                y=cby,
                len=0.34,
                thickness=11,
                title=dict(text=cbtitle, side="right"),
                tickfont=dict(size=9),
            ),
        )

    # colorbars parked in the gutters / right margin so they never overlap panels
    fig.add_trace(contour(pi, "Viridis", 0.455, 0.815, "$"), row=1, col=1)
    fig.add_trace(contour(unc, "Blues", 1.02, 0.815, "σ ($)"), row=1, col=2)
    fig.add_trace(contour(ucb, "Plasma", 0.455, 0.185, "UCB"), row=2, col=1)

    def marker(x, y, name, symbol, fill, edge, show):
        # white/black fill with a contrasting outline stays visible on every
        # colorscale (Viridis, Blues, Plasma) and in the legend.
        return go.Scatter(
            x=[x],
            y=[y],
            mode="markers",
            name=name,
            legendgroup=name,
            showlegend=show,
            marker=dict(
                symbol=symbol, size=16, color=fill, line=dict(color=edge, width=2.2)
            ),
        )

    # exploit (star) and explore (diamond) markers on every surface panel
    for (r, c), first_star, first_diamond in [
        ((1, 1), True, False),
        ((1, 2), False, True),
        ((2, 1), False, False),
    ]:
        fig.add_trace(
            marker(
                ex_x,
                ex_y,
                "exploit optimum (mean profit)",
                "star",
                "white",
                "black",
                first_star,
            ),
            row=r,
            col=c,
        )
        fig.add_trace(
            marker(
                ac_x,
                ac_y,
                "acquisition optimum (UCB)",
                "diamond",
                "black",
                "white",
                first_diamond,
            ),
            row=r,
            col=c,
        )
        fig.update_xaxes(title_text=xlab, row=r, col=c)
        fig.update_yaxes(title_text=ylab, row=r, col=c)

    # 1-D slice: fix slice-i channel at the acquisition x, vary slice-j
    col = int(np.argmin(np.abs(g - ac_x)))
    mu = pi[:, col]
    sd = unc[:, col]
    band = ucb[:, col]
    fig.add_trace(
        go.Scatter(
            x=np.r_[g, g[::-1]],
            y=np.r_[mu + 2 * sd, (mu - 2 * sd)[::-1]],
            fill="toself",
            fillcolor="rgba(90,158,111,0.18)",
            line=dict(width=0),
            name="±2σ (profit)",
            legendgroup="band",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=g,
            y=mu,
            mode="lines",
            line=dict(color="#5a9e6f", width=2.5),
            name="mean profit π",
            legendgroup="mean",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=g,
            y=band,
            mode="lines",
            line=dict(color="#b15a7a", width=2.5, dash="dash"),
            name="acquisition UCB",
            legendgroup="ucb",
        ),
        row=2,
        col=2,
    )
    fig.add_vline(
        x=ex_y, line=dict(color="#7f7f7f", width=1.5, dash="dot"), row=2, col=2
    )
    fig.add_vline(
        x=ac_y, line=dict(color="#00b0a4", width=1.5, dash="dot"), row=2, col=2
    )
    fig.update_xaxes(title_text=ylab, row=2, col=2)
    fig.update_yaxes(title_text="profit ($)", row=2, col=2)

    fig.update_layout(
        template="plotly_white",
        width=1040,
        height=880,
        margin=dict(l=75, r=95, t=110, b=95),
        title=dict(
            text="<b>Acquisition function & uncertainty surface</b>"
            f"<br><sup>{xlab} × {ylab} slice; the acquisition point is "
            "pulled toward higher uncertainty</sup>",
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
            font=dict(size=11),
        ),
    )
    # give the subplot titles a little breathing room from the panels
    for ann in fig.layout.annotations:
        ann.font.size = 12.5
    return fig


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    fig = build_figure(_load())
    png = os.path.join(OUTDIR, "continuous_learning_acquisition.png")
    html = os.path.join(OUTDIR, "continuous_learning_acquisition.html")
    fig.write_image(png, width=1040, height=880, scale=2)
    fig.write_html(html, include_plotlyjs="cdn")
    print("wrote", png, "and", html)


if __name__ == "__main__":
    main()
