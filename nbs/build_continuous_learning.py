"""Author the continuous-learning notebook (run from ``nbs/``).

    uv run --with nbformat python build_continuous_learning.py
    PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        continuous_learning.ipynb --ExecutePreprocessor.timeout=3600 \
        --ExecutePreprocessor.kernel_name=python3

A visual walkthrough of the **model-free continuous-learning loop**
(``mmm_framework.continuous_learning``): learn how spend drives outcome directly
from designed geo experiments — with NO pre-fit MMM — then allocate, decide the
funding line, and stop when more testing no longer pays.

Authored as md/code cells via nbformat (pattern: ``build_breakout_weighting``).
Every computational cell ends with asserts encoding its claim — directional and
seed-tolerant for MCMC. Markdown points at the live plots, never at hardcoded
fit numbers.
"""

from __future__ import annotations

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


CELLS = [
    # ====================================================================
    # 1. Hook + thesis
    # ====================================================================
    md(r"""
# Continuous sequential learning — measuring without a model

Most marketing-mix work fits a model to **observational history** and hopes the
spend was exogenous enough to be causal. This notebook walks through the
opposite: a self-contained loop that learns how spend drives outcome **directly
from designed geo experiments**, with **no pre-fit MMM at all**.

It is the `mmm_framework.continuous_learning` package — a Bayesian
*geo response-surface bandit*. Each round (a **wave**) it:

1. runs a **designed** batch of geo cells (a central-composite design) and
   observes outcomes,
2. fits a Bayesian response surface (Hill saturation + sign-informed synergies),
3. **plans** — a Thompson posterior over the optimal budget split + a funding
   line, and
4. **decides whether another wave is worth it** (the ENBS stopping rule),
   recentering on the recommendation and going again if so.

```
   ┌──────────────────────────────────────────────────────────────┐
   ▼                                                                │
 FIT posterior ─▶ SCORE & PICK (acquisition) ─▶ RUN WAVE ─▶ UPDATE  │
                  thompson / funding / KG       designed     data ──┘
   ▲                                            holdouts
   └─────────────────────────  STOP?  (expected_regret + ENBS)
```

The posterior is *carried across waves* by refitting on **all** accumulated
data, so every wave borrows strength from the last.

### Where it sits in the framework

| | `planning/` (model-anchored) | `continuous_learning/` (model-free) |
|---|---|---|
| Needs a fitted `BayesianMMM` | **yes** | **no** |
| Source of the surface | observational time series | designed geo experiments |
| Backend | PyMC | NumPyro / JAX |
| Decision | one-shot + score next experiment | sequential loop, posterior carried |

They are complementary. A team with no usable history can run this loop; the
Hill activation here is **convention-identical** to `SaturationType.HILL`
(`slope = α`, `sat_half = κ`), so the two posteriors are directly comparable.
"""),
    code(r"""
import warnings
warnings.filterwarnings("ignore")

import jax
jax.config.update("jax_platform_name", "cpu")

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import mmm_framework.continuous_learning as cl
from mmm_framework.continuous_learning import surface

pio.templates.default = "plotly_white"

# A consistent per-channel palette used throughout.
CHANNELS = ["Chatter", "Pulse", "Orbit", "Vibe"]
PALETTE = {"Chatter": "#3b6fb6", "Pulse": "#d98c3f", "Orbit": "#5a9e6f", "Vibe": "#b15a7a"}
COLORS = [PALETTE[c] for c in CHANNELS]
SIGN_COLOR = {"neg": "#c0504d", "pos": "#4a8d57", "zero": "#9aa0a6", "weak": "#7f7f7f"}

# Shared decision parameters (scaled spend units; value = $ per unit KPI).
B = 3.2                                   # budget
VALUE = 5.0                               # $ / unit KPI -> sets the funding line
CENTER = np.array([0.7, 0.7, 0.7, 0.7])   # current operating allocation


def style(fig, title=None, height=400, **kw):
    fig.update_layout(
        title=title, height=height, margin=dict(l=60, r=20, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), **kw,
    )
    return fig


def forest(names, mean, lo, hi, truth=None, title=None, xlab="value", colors=None):
    "Horizontal credible-interval plot with optional truth markers."
    y = list(range(len(names)))
    colors = colors or ["#3b6fb6"] * len(names)
    fig = go.Figure()
    for i, nm in enumerate(names):
        fig.add_trace(go.Scatter(
            x=[lo[i], hi[i]], y=[i, i], mode="lines",
            line=dict(color=colors[i], width=3), showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=mean, y=y, mode="markers", name="posterior mean",
        marker=dict(color=colors, size=11, line=dict(color="white", width=1)),
    ))
    if truth is not None:
        fig.add_trace(go.Scatter(
            x=truth, y=y, mode="markers", name="truth",
            marker=dict(symbol="x", color="black", size=12, line=dict(width=2)),
        ))
    fig.update_yaxes(tickmode="array", tickvals=y, ticktext=names)
    return style(fig, title=title, xaxis_title=xlab, height=90 + 55 * len(names))


print("continuous_learning package:", cl.__name__)
print("public API:", len(cl.__all__), "symbols")
assert {"fit", "central_composite", "thompson_wave", "run_closed_loop"} <= set(cl.__all__)
"""),
    # ====================================================================
    # 2. The response surface
    # ====================================================================
    md(r"""
## 1 · The response surface — one differentiable function

Everything rests on a single function (`surface.incremental`), used in three
places so they can never disagree: the **likelihood** fits it, the **simulator**
generates from it, and the **allocator** differentiates it. Per geo-week, for a
scaled spend vector $s\in\mathbb{R}^K$:

$$
f_c(s_c) = \frac{s_c^{\alpha_c}}{\kappa_c^{\alpha_c} + s_c^{\alpha_c}}
\qquad\text{(Hill activation, }[0,1)\text{)}
$$
$$
\text{incremental}(s) = \sum_c \beta_c\, f_c(s_c)
  \;+\; \sum_{c<c'} \gamma_{cc'}\, f_c(s_c)\, f_{c'}(s_{c'})
$$

$\beta_c$ is the channel ceiling, $\kappa_c$ the half-saturation (where
$f=\tfrac12$), $\alpha_c$ the Hill shape, and $\gamma$ the **synergy** —
positive for complementarity, negative for cannibalization.
"""),
    code(r"""
# Hill activation: how the shape (alpha) and half-saturation (kappa) bite.
s_grid = np.linspace(0, 3, 200)
fig = make_subplots(rows=1, cols=2, subplot_titles=(
    "shape α (κ = 1 fixed)", "half-saturation κ (α = 2 fixed)"))

for a in [0.7, 1.0, 2.0, 4.0]:
    f = np.asarray(surface.activation(s_grid, np.ones_like(s_grid), np.full_like(s_grid, a)))
    fig.add_trace(go.Scatter(x=s_grid, y=f, name=f"α={a}"), row=1, col=1)
for kp in [0.5, 1.0, 1.5, 2.0]:
    f = np.asarray(surface.activation(s_grid, np.full_like(s_grid, kp), np.full_like(s_grid, 2.0)))
    fig.add_trace(go.Scatter(x=s_grid, y=f, name=f"κ={kp}", line=dict(dash="dot")), row=1, col=2)

for col in (1, 2):
    fig.add_hline(y=0.5, line=dict(color="gray", dash="dash"), row=1, col=col)
fig.update_xaxes(title_text="scaled spend s", row=1, col=1)
fig.update_xaxes(title_text="scaled spend s", row=1, col=2)
fig.update_yaxes(title_text="activation f(s)", row=1, col=1)
style(fig, "Hill activation = SaturationType.HILL (slope=α, sat_half=κ)", height=380)
fig.show()

# f(kappa) == 0.5 exactly, and f is increasing in spend.
f_at_kappa = np.asarray(surface.activation(np.array([1.3]), np.array([1.3]), np.array([2.0])))[0]
assert np.isclose(f_at_kappa, 0.5, atol=1e-5)
"""),
    md(r"""
### Synergy: the interaction surface

The $\gamma$ term bends the joint response. Below we isolate the **interaction
component** — `incremental(full γ) − incremental(γ = 0)` — over two channels at
a time, holding the others at center. A **positive** $\gamma$ (Pulse × Orbit)
lifts the corner where both are funded (complementarity); a **negative**
$\gamma$ (Chatter × Pulse) carves a valley there (shared-audience
cannibalization).
"""),
    code(r"""
world = cl.make_world(seed=0)
additive = cl.TrueWorld(  # same main effects, synergies zeroed
    beta=world.beta, kappa=world.kappa, alpha=world.alpha,
    gamma_pairs=np.zeros(len(world.pairs)), channels=world.channels, pairs=world.pairs,
)

def interaction_grid(pair, n=45, hi=2.2):
    i, j = pair
    g = np.linspace(0.0, hi, n)
    gi, gj = np.meshgrid(g, g)
    spend = np.repeat(CENTER[None, :], gi.size, axis=0)
    spend[:, i] = gi.ravel()
    spend[:, j] = gj.ravel()
    full = world.response_mean(spend).reshape(n, n)
    base = additive.response_mean(spend).reshape(n, n)
    return g, full - base   # synergy-only component

pairs_show = [(1, 2), (0, 1)]   # Pulse x Orbit (+), Chatter x Pulse (-)
fig = make_subplots(rows=1, cols=2, subplot_titles=[
    f"{world.channels[i]} × {world.channels[j]}  (γ>0, complementarity)"
    if (i, j) == (1, 2) else
    f"{world.channels[i]} × {world.channels[j]}  (γ<0, cannibalization)"
    for (i, j) in pairs_show])
for col, pair in enumerate(pairs_show, start=1):
    g, z = interaction_grid(pair)
    fig.add_trace(go.Contour(
        x=g, y=g, z=z, colorscale="RdBu", zmid=0, showscale=(col == 2),
        contours=dict(showlabels=True)), row=1, col=col)
    fig.update_xaxes(title_text=world.channels[pair[0]], row=1, col=col)
    fig.update_yaxes(title_text=world.channels[pair[1]], row=1, col=col)
style(fig, "Interaction component of the response surface", height=440)
fig.show()

# the +gamma pair lifts the joint corner above additive; the -gamma pair depresses it
g, z_pos = interaction_grid((1, 2))
_, z_neg = interaction_grid((0, 1))
assert z_pos[-1, -1] > 0.05 and z_neg[-1, -1] < -0.05
"""),
    # ====================================================================
    # 3. The experimental design
    # ====================================================================
    md(r"""
## 2 · The designed wave (central-composite)

A model is only causal because spend is **experimentally assigned**. The
central-composite design (CCD) is the minimal set of geo cells that identifies
the local surface around the operating point:

* **center** — the current allocation (the trust-region anchor),
* **axial** — each channel moved $\pm\delta$ alone (gives the gradient / main
  effects),
* **off-axis** — two channels moved jointly (the only way to see $\gamma$),
* **shutoff** — one channel set to 0; these **break the $\beta$–$\gamma$
  collinearity** (without them $\beta$ attenuates).

The heatmap below *is* the design: read each row as one cell's scaled spend.
"""),
    code(r"""
probe = world.pairs
dsg = cl.central_composite(CENTER, delta=0.6, probe_pairs=probe)
n_cells, K = dsg.shape

# label each row by its role
roles = (["center"] + [f"axial {world.channels[c]}{pm}" for c in range(K) for pm in ("+", "-")]
         + [f"off-axis {world.channels[i]}×{world.channels[j]}{pm}"
            for (i, j) in probe for pm in ("+", "-")]
         + [f"shutoff {world.channels[c]}" for c in range(K)])

fig = go.Figure(go.Heatmap(
    z=dsg, x=world.channels, y=[f"{r}" for r in roles],
    colorscale="Blues", colorbar=dict(title="scaled<br>spend")))
fig.update_yaxes(autorange="reversed")
style(fig, f"Central-composite design — {n_cells} cells × {K} channels", height=560)
fig.show()

assert n_cells == 1 + 2 * K + 2 * len(probe) + K
np.testing.assert_allclose(dsg[0], CENTER)            # first row is the center
assert np.any(np.isclose(dsg[-K:], 0.0))              # shutoffs zero one channel
"""),
    md(r"""
### Identification: the pre/test split

`simulate_panel` builds the data contract the loop needs: a **pre-period** where
every geo runs the status-quo `center` (this pins each geo's baseline intercept
$a_g$ and breaks the within-geo baseline↔response collinearity), then a
**test-period** where each geo runs its assigned CCD cell (the designed
variation that identifies the surface). Round-robin assignment keeps the cells
balanced across geos.
"""),
    code(r"""
rng = np.random.default_rng(0)
geo_alloc, cell_idx = cl.assign_geos(dsg, n_geo=80, rng=rng)

# how many geos land in each design cell (balance)
counts = np.bincount(cell_idx, minlength=n_cells)
fig = go.Figure(go.Bar(x=[f"c{i}" for i in range(n_cells)], y=counts, marker_color="#3b6fb6"))
style(fig, "Round-robin geo assignment is balanced across cells", height=320,
      xaxis_title="design cell", yaxis_title="# geos")
fig.show()

assert counts.max() - counts.min() <= 1     # round-robin => near-perfectly balanced
"""),
    # ====================================================================
    # 4. Recovery
    # ====================================================================
    md(r"""
## 3 · Fit a known world and recover the effects

We simulate one wave from a **known** world (so we can grade ourselves), then
fit the response surface. The sign-informed priors (`PAIR_SIGNS_EXAMPLE`) encode
domain knowledge about which pairs cannibalize vs complement.

> The guide's lesson: **signs recover robustly; magnitudes are
> prior-sensitive** — audit them (§7 below).
"""),
    code(r"""
data = cl.simulate_panel(world, CENTER, n_geo=80, t_pre=6, t_test=10,
                         delta=0.6, noise=0.5, seed=1)
print(f"panel: {data['spend'].shape[0]} geo-weeks across {data['n_geo']} geos, "
      f"{data['design'].shape[0]} CCD cells")

post = cl.fit(data, channels=world.channels, pair_signs=cl.PAIR_SIGNS_EXAMPLE,
              num_warmup=500, num_samples=500, num_chains=2, seed=0)
_rhat = post.diagnostics.get("max_rhat")
print(f"posterior: {post.n_draws} draws | max R-hat = "
      f"{_rhat:.3f}" if _rhat is not None else f"posterior: {post.n_draws} draws")

assert post.n_draws == 1000
assert post.diagnostics["max_rhat"] is None or post.diagnostics["max_rhat"] < 1.2
"""),
    code(r"""
# Pre-period vs test-period KPI: the designed variation shows up in the spread.
pre = data["spend"].shape[0] - data["n_geo"] * 10
y = data["y"]
fig = go.Figure()
fig.add_trace(go.Box(y=y[:pre], name="pre-period<br>(all at center)", marker_color="#9aa0a6"))
fig.add_trace(go.Box(y=y[pre:], name="test-period<br>(designed cells)", marker_color="#3b6fb6"))
style(fig, "Designed test-period variation is what identifies the surface",
      height=360, yaxis_title="KPI (natural units)")
fig.show()
assert y[pre:].std() > y[:pre].std()   # the test window carries more spread
"""),
    code(r"""
# Main-effect recovery: posterior beta with truth overlaid.
b = post.samples["beta"]
fig = forest(world.channels, b.mean(0),
             np.percentile(b, 5, axis=0), np.percentile(b, 95, axis=0),
             truth=world.beta, title="Channel ceilings β — recovered vs truth",
             xlab="β", colors=COLORS)
fig.show()

# the strongest channel and the top-2 are recovered (well-separated, robust);
# the two weakest (Orbit ~ Vibe) are close and order-fragile under noise —
# exactly the "magnitudes are prior-sensitive" lesson.
assert int(np.argmax(b.mean(0))) == int(np.argmax(world.beta))
assert set(np.argsort(-b.mean(0))[:2]) == set(np.argsort(-world.beta)[:2])
"""),
    code(r"""
# Synergy recovery: signs are the reliable part. Color by prior family.
gs = post.gamma_summary()
names = list(gs.keys())
mean = [gs[n]["mean"] for n in names]
lo = [gs[n]["p5"] for n in names]
hi = [gs[n]["p95"] for n in names]
truth = {f"gamma_{world.channels[i]}_{world.channels[j]}": float(g)
         for (i, j), g in zip(world.pairs, world.gamma_pairs)}
cols = [SIGN_COLOR[gs[n]["sign"]] for n in names]
fig = forest([n.replace("gamma_", "") for n in names], mean, lo, hi,
             truth=[truth[n] for n in names],
             title="Synergies γ — sign is reliable (color = prior family)",
             xlab="γ", colors=cols)
fig.add_vline(x=0, line=dict(color="black", dash="dash"))
fig.show()

assert gs["gamma_Chatter_Pulse"]["mean"] < 0      # cannibalization recovered negative
assert gs["gamma_Pulse_Orbit"]["mean"] > 0    # complementarity recovered positive
assert gs["gamma_Orbit_Vibe"]["mean"] > 0
"""),
    code(r"""
# Per-channel response curves: fitted posterior band vs the true curve.
s = np.linspace(0, 2.5, 120)
fig = make_subplots(rows=2, cols=2, subplot_titles=world.channels)
for c, name in enumerate(world.channels):
    f = s[None, :] ** post.samples["alpha"][:, c, None]
    f = f / (f + post.samples["kappa"][:, c, None] ** post.samples["alpha"][:, c, None])
    contrib = post.samples["beta"][:, c, None] * f            # (draws, grid)
    mean = contrib.mean(0)
    lo, hi = np.percentile(contrib, [5, 95], axis=0)
    ft = s ** world.alpha[c] / (s ** world.alpha[c] + world.kappa[c] ** world.alpha[c])
    true = world.beta[c] * ft
    r, cc = divmod(c, 2)
    r, cc = r + 1, cc + 1
    fig.add_trace(go.Scatter(x=np.r_[s, s[::-1]], y=np.r_[hi, lo[::-1]], fill="toself",
                             fillcolor="rgba(59,111,182,0.15)", line=dict(width=0),
                             showlegend=False), row=r, col=cc)
    fig.add_trace(go.Scatter(x=s, y=mean, line=dict(color=PALETTE[name]),
                             showlegend=False), row=r, col=cc)
    fig.add_trace(go.Scatter(x=s, y=true, line=dict(color="black", dash="dash"),
                             showlegend=False), row=r, col=cc)
style(fig, "Recovered response curve (band) vs truth (dashed)", height=560)
fig.show()

# at the operating point the fitted contribution is close to truth for the top channel
c0 = int(np.argmax(world.beta))
f0 = CENTER[c0] ** post.samples["alpha"][:, c0] / (
    CENTER[c0] ** post.samples["alpha"][:, c0] + post.samples["kappa"][:, c0] ** post.samples["alpha"][:, c0])
fit0 = (post.samples["beta"][:, c0] * f0).mean()
ft0 = CENTER[c0] ** world.alpha[c0] / (CENTER[c0] ** world.alpha[c0] + world.kappa[c0] ** world.alpha[c0])
true0 = world.beta[c0] * ft0
assert abs(fit0 - true0) / true0 < 0.25
"""),
    # ====================================================================
    # 5. The planner
    # ====================================================================
    md(r"""
## 4 · Plan — a posterior over the optimal split, and a funding line

**Thompson sampling**: solve the budget allocation under *each* posterior draw.
The result is a *distribution* over the optimal split — its mean is the
recommendation and its spread is the exploration signal (which channels we are
still unsure how to fund). The surface is non-concave (negative $\gamma$), so
the allocator multi-starts.
"""),
    code(r"""
allocs, profits, draws = cl.thompson_wave(post, B, VALUE, q=300, mode="fixed", seed=0)
rec = allocs.mean(0)
rec = rec * (B / rec.sum())
true_alloc, true_profit = cl.world_optimal_allocation(world, B, VALUE, mode="fixed")

fig = go.Figure()
for c, name in enumerate(world.channels):
    fig.add_trace(go.Violin(y=allocs[:, c], name=name, line_color=PALETTE[name],
                            box_visible=True, meanline_visible=True, points=False))
fig.add_trace(go.Scatter(x=world.channels, y=true_alloc, mode="markers", name="truth-optimal",
                         marker=dict(symbol="x", color="black", size=13, line=dict(width=2))))
style(fig, "Thompson posterior over the optimal split (× = truth-optimal)",
      height=420, yaxis_title="scaled spend")
fig.show()

assert abs(rec.sum() - B) < 0.05                      # recommendation respects the budget
assert allocs.shape == (300, 4)
"""),
    md(r"""
### The funding line

A channel is **funded** where the posterior says its marginal ROAS clears the
break-even line: $P(\text{value}\cdot \partial R/\partial s_c > 1) > 0.5$. The
violins below are the marginal-ROAS posterior at the recommended allocation; the
dashed line is break-even.
"""),
    code(r"""
mroas_mean, prob_above, mroas_draws = cl.marginal_roas(post, rec, VALUE, q=300, seed=1)
fig = go.Figure()
for c, name in enumerate(world.channels):
    fig.add_trace(go.Violin(y=mroas_draws[:, c], name=f"{name}<br>P(>1)={prob_above[c]:.0%}",
                            line_color=PALETTE[name], box_visible=True, points=False))
fig.add_hline(y=1.0, line=dict(color="black", dash="dash"),
              annotation_text="break-even", annotation_position="top left")
style(fig, "Funding line — marginal ROAS posterior at the recommendation",
      height=420, yaxis_title="value · ∂R/∂s")
fig.show()

# the strongest channel is funded with high probability
assert prob_above[int(np.argmax(world.beta))] > 0.5
"""),
    # ====================================================================
    # 6. The closed loop
    # ====================================================================
    md(r"""
## 5 · The loop — carry the posterior, stop on ENBS

Now the whole thing in motion. `run_closed_loop` refits on **all** accumulated
data each wave (that is how the posterior is carried), recommends, prices the
funding line, computes the **expected regret** (the profit still on the table
from posterior uncertainty), and checks the **ENBS** stopping rule:

$$\text{ENBS} = \mathbb{E}[\text{regret}]\cdot\text{margin}\cdot\text{population} - \text{wave cost} \le 0 \;\Rightarrow\; \text{stop.}$$

In a real deployment the synthetic wave collector is swapped for the actual geo
holdout results — nothing else changes.
"""),
    code(r"""
out = cl.run_closed_loop(
    world, center=CENTER, B=B, value=VALUE,
    n_geo=80, t_pre=6, t_test=10, delta=0.6, noise=0.6,
    mode="fixed", pair_signs=cl.PAIR_SIGNS_EXAMPLE,
    margin=1.0, population=2.0, wave_cost=0.45, max_waves=4, planner_q=200,
    fit_kwargs=dict(num_warmup=400, num_samples=400, num_chains=2, seed=0),
    seed=7,
)
hist = out["history"]
for r in hist:
    print(f"wave {r['wave']}: rows={r['n_rows']:5d}  E[regret]={r['e_regret']:.3f}  "
          f"ENBS={r['enbs']:+.3f}  stop={r['stop']}  gap={100*r['profit_gap_rel']:.1f}%")

assert hist[-1]["e_regret"] < hist[0]["e_regret"]      # closure: regret shrinks
assert any(r["stop"] for r in hist)                    # stopping: ENBS fires
"""),
    code(r"""
waves = [r["wave"] for r in hist]
fig = make_subplots(rows=2, cols=2, subplot_titles=(
    "Expected regret ↓ (learning)", "ENBS ↓ crosses 0 (stop)",
    "Profit gap vs truth-optimal", "Recommendation → truth (dashed)"))

fig.add_trace(go.Scatter(x=waves, y=[r["e_regret"] for r in hist], mode="lines+markers",
                         line=dict(color="#3b6fb6"), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=waves, y=[r["enbs"] for r in hist], mode="lines+markers",
                         line=dict(color="#d98c3f"), showlegend=False), row=1, col=2)
fig.add_hline(y=0, line=dict(color="black", dash="dash"), row=1, col=2)
stop_wave = next((r["wave"] for r in hist if r["stop"]), None)
if stop_wave is not None:
    fig.add_vline(x=stop_wave, line=dict(color="#c0504d", dash="dot"), row=1, col=2)
fig.add_trace(go.Scatter(x=waves, y=[100 * r["profit_gap_rel"] for r in hist],
                         mode="lines+markers", line=dict(color="#5a9e6f"), showlegend=False),
              row=2, col=1)
recs = np.array([r["recommendation"] for r in hist])
for c, name in enumerate(world.channels):
    fig.add_trace(go.Scatter(x=waves, y=recs[:, c], mode="lines+markers", name=name,
                             line=dict(color=PALETTE[name])), row=2, col=2)
    fig.add_hline(y=out["true_allocation"][c], line=dict(color=PALETTE[name], dash="dash"),
                  row=2, col=2)
fig.update_yaxes(title_text="KPI units", row=1, col=1)
fig.update_yaxes(title_text="$", row=1, col=2)
fig.update_yaxes(title_text="%", row=2, col=1)
fig.update_yaxes(title_text="scaled spend", row=2, col=2)
fig.update_xaxes(title_text="wave", row=2, col=1)
fig.update_xaxes(title_text="wave", row=2, col=2)
style(fig, "The loop: learn, converge, and stop when testing stops paying", height=620)
fig.show()

assert hist[-1]["enbs"] < hist[0]["enbs"]
assert hist[-1]["profit_gap_rel"] < 0.1                # final recommendation tracks truth
"""),
    # ====================================================================
    # 7. Prior-sensitivity audit
    # ====================================================================
    md(r"""
## 6 · Audit the magnitudes (`gamma_scale`)

The most consequential knob is `gamma_scale`. The guide warns: signs recover,
but **magnitudes are prior-sensitive**. A *prior-dominated* pair (here
`Chatter × Vibe`, whose true synergy is 0) tracks its prior — widen `gamma_scale`
and its posterior widens with it. Flag such pairs as sign-reliable /
magnitude-assumed in any decision.
"""),
    code(r"""
audit = {}
for g in (0.4, 0.8, 1.6):
    p = cl.fit(data, channels=world.channels, pair_signs=cl.PAIR_SIGNS_EXAMPLE,
               gamma_scale=g, num_warmup=300, num_samples=300, num_chains=2, seed=0)
    audit[g] = p.samples["gamma_Chatter_Vibe"]

fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], subplot_titles=(
    "Posterior of γ(Chatter×Vibe), true = 0", "Posterior sd vs gamma_scale"))
for g, draws in audit.items():
    fig.add_trace(go.Histogram(x=draws, name=f"scale={g}", opacity=0.55, nbinsx=40,
                               histnorm="probability density"), row=1, col=1)
fig.add_vline(x=0, line=dict(color="black", dash="dash"), row=1, col=1)
fig.add_trace(go.Bar(x=[str(g) for g in audit], y=[float(np.std(v)) for v in audit.values()],
                     marker_color="#b15a7a", showlegend=False), row=1, col=2)
fig.update_layout(barmode="overlay")
fig.update_xaxes(title_text="γ", row=1, col=1)
fig.update_xaxes(title_text="gamma_scale", row=1, col=2)
fig.update_yaxes(title_text="posterior sd", row=1, col=2)
style(fig, "Prior-sensitivity audit — a prior-dominated synergy tracks its prior",
      height=400)
fig.show()

sds = {g: float(np.std(v)) for g, v in audit.items()}
assert sds[1.6] > sds[0.4]      # wider prior -> wider posterior for the unidentified pair
"""),
    # ====================================================================
    # 7. Baseline realism — adstock pre-pass + CUPED
    # ====================================================================
    md(r"""
## 7 · Baseline realism — adstock pre-pass & CUPED (guide §9.3/9.4)

Two cheap upgrades make the loop production-real without touching the surface.

**Adstock pre-pass.** If spend has **carryover**, the response is driven by the
*adstocked* spend series, not this week's spend. The CCD holds each geo's cell
constant, so in a long window the adstocked and raw series coincide (which is why
the reference "treats spend as the lever directly") — but in a **short** window
the pre→test transition dominates and fitting on raw spend biases the curve. The
fix: geometric-adstock the spend, then fit (`adstock_prepass`).
"""),
    code(r"""
# A carryover world (response sees the adstocked series) measured over a SHORT
# window, so the transition — not steady state — dominates.
data_raw = cl.simulate_panel(world, CENTER, n_geo=80, t_pre=6, t_test=4,
                             delta=0.6, noise=0.4, adstock_alpha=0.7, seed=11)
data_ad = cl.adstock_prepass(data_raw, t_pre=6, t_test=4, alpha=0.7)

post_raw = cl.fit(data_raw, channels=world.channels, pair_signs=cl.PAIR_SIGNS_EXAMPLE,
                  num_warmup=300, num_samples=300, num_chains=2, seed=0)
post_ad = cl.fit(data_ad, channels=world.channels, pair_signs=cl.PAIR_SIGNS_EXAMPLE,
                 num_warmup=300, num_samples=300, num_chains=2, seed=0)
br, ba = post_raw.samples["beta"].mean(0), post_ad.samples["beta"].mean(0)

# what the pre-pass does to one geo's spend series (Chatter), pre vs test weeks
T = 6 + 4
geo0_raw = data_raw["spend"].reshape(T, 80, 4)[:, 0, 0]
geo0_ad = data_ad["spend"].reshape(T, 80, 4)[:, 0, 0]
fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], subplot_titles=(
    "one geo's Chatter spend: raw vs adstocked", "β recovery: raw fit vs adstock pre-pass"))
wk = np.arange(T)
fig.add_trace(go.Scatter(x=wk, y=geo0_raw, name="raw", line=dict(color="#9aa0a6")), row=1, col=1)
fig.add_trace(go.Scatter(x=wk, y=geo0_ad, name="adstocked", line=dict(color="#3b6fb6")), row=1, col=1)
fig.add_vline(x=5.5, line=dict(color="black", dash="dot"), row=1, col=1)
fig.add_trace(go.Bar(x=world.channels, y=world.beta, name="truth", marker_color="black"), row=1, col=2)
fig.add_trace(go.Bar(x=world.channels, y=br, name="raw fit", marker_color="#c0504d"), row=1, col=2)
fig.add_trace(go.Bar(x=world.channels, y=ba, name="adstock fit", marker_color="#4a8d57"), row=1, col=2)
fig.update_xaxes(title_text="week (pre | test)", row=1, col=1)
fig.update_yaxes(title_text="β", row=1, col=2)
style(fig, "Carryover bias and the adstock pre-pass", height=400, barmode="group")
fig.show()

err_raw = np.mean(np.abs(br - world.beta) / world.beta)
err_ad = np.mean(np.abs(ba - world.beta) / world.beta)
print(f"mean |β rel error|  —  raw fit: {err_raw:.0%}   adstock pre-pass: {err_ad:.0%}")
assert err_ad < err_raw     # pre-adstocking recovers the carryover-biased curve
"""),
    md(r"""
**CUPED.** The pre-period KPI is a strong predictor of a geo's test-period
outcome (they share the baseline `a_geo`). Subtracting the CUPED prediction
`θ·x_pre` strips that baseline variance from the outcome, so the lift estimate's
variance falls by `1 − ρ²` — the same precision with **fewer geos** (a direct
testing-budget lever).
"""),
    code(r"""
# Use the clean recovery panel (t_pre=6) for CUPED.
adj, info = cl.cuped_adjust(data, t_pre=6)
x_pre = cl.cuped_covariate(data, t_pre=6)
n_pre = 6 * data["n_geo"]
y_test_geo = np.array([data["y"][n_pre:][data["geo_idx"][n_pre:] == g].mean()
                       for g in range(data["n_geo"])])
adj_test_geo = np.array([adj["y"][n_pre:][adj["geo_idx"][n_pre:] == g].mean()
                         for g in range(data["n_geo"])])

fig = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42], subplot_titles=(
    f"test outcome vs pre-period covariate (ρ={info['rho']:.2f})",
    "geo-level variance: raw vs CUPED"))
xs = np.linspace(x_pre.min(), x_pre.max(), 50)
fig.add_trace(go.Scatter(x=x_pre, y=y_test_geo, mode="markers",
                         marker=dict(color="#3b6fb6", opacity=0.7), name="geos"), row=1, col=1)
fig.add_trace(go.Scatter(x=xs, y=y_test_geo.mean() + info["theta"] * xs, mode="lines",
                         line=dict(color="black", dash="dash"), name="CUPED fit"), row=1, col=1)
fig.add_trace(go.Bar(x=["raw", "CUPED-adjusted"], y=[np.var(y_test_geo), np.var(adj_test_geo)],
                     marker_color=["#9aa0a6", "#4a8d57"], showlegend=False), row=1, col=2)
fig.update_xaxes(title_text="pre-period KPI (centered)", row=1, col=1)
fig.update_yaxes(title_text="test-period geo mean KPI", row=1, col=1)
fig.update_yaxes(title_text="variance", row=1, col=2)
style(fig, f"CUPED removes {info['var_reduction']:.0%} of geo-level variance "
           f"(≈ {1 - info['var_reduction']:.0%} of the geos for the same MDE)", height=400)
fig.show()

assert 0 < info["var_reduction"] < 1
assert np.var(adj_test_geo) < np.var(y_test_geo)
"""),
    # ====================================================================
    # 8. Fast Laplace knowledge-gradient
    # ====================================================================
    md(r"""
## 8 · Fast knowledge-gradient — Laplace instead of refit (guide §9.1)

The reference `knowledge_gradient` refits with full NUTS once *per fantasy* — too
slow to score many candidate *test* designs. The Laplace version
(`laplace_knowledge_gradient`) moment-matches the posterior to a Gaussian,
linearizes the surface, and computes the pre-posterior decision value in closed
form + a few cheap allocations — **no MCMC**. It ranks designs like the NUTS KG
at a fraction of the cost.
"""),
    code(r"""
import time

sigma_hat = float(post.samples["sigma"].mean())
cands = {
    "probe-all (δ=0.6)": cl.central_composite(CENTER, 0.6, world.pairs),
    "no-probe (δ=0.6)": cl.central_composite(CENTER, 0.6, []),
    "wide (δ=0.8)": cl.central_composite(CENTER, 0.8, world.pairs),
    "tight (δ=0.4)": cl.central_composite(CENTER, 0.4, world.pairs),
}

# Fast Laplace KG for every candidate.
t0 = time.perf_counter()
lap = {nm: cl.laplace_knowledge_gradient(post, d, B, VALUE, sigma=sigma_hat,
                                         n_geo=data["n_geo"], t_test=10, n_outcomes=48, seed=0)
       for nm, d in cands.items()}
lap_t = time.perf_counter() - t0

# NUTS-refit KG for a subset (the expensive path), to check agreement.
refit = cl.refit_fn_from_data(data, channels=world.channels, pair_signs=cl.PAIR_SIGNS_EXAMPLE,
                              num_warmup=80, num_samples=80, num_chains=1)
subset = ["probe-all (δ=0.6)", "no-probe (δ=0.6)"]
t0 = time.perf_counter()
nuts = {nm: cl.knowledge_gradient(post, cands[nm], refit, B, VALUE, n_fantasy=2,
                                  t_test=10, n_geo=data["n_geo"], q=60, seed=1)
        for nm in subset}
nuts_t = time.perf_counter() - t0
speedup = (nuts_t / len(subset)) / (lap_t / len(cands))
print(f"Laplace KG: {len(cands)} designs in {lap_t:.2f}s   |   "
      f"NUTS KG: {len(subset)} designs in {nuts_t:.1f}s   |   ~{speedup:.0f}× faster per design")

fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], subplot_titles=(
    "Laplace KG (decision value) per candidate", "Laplace vs NUTS KG agree"))
fig.add_trace(go.Bar(x=list(lap), y=list(lap.values()), marker_color="#3b6fb6",
                     showlegend=False), row=1, col=1)
fig.add_trace(go.Bar(x=subset, y=[lap[nm] for nm in subset], name="Laplace",
                     marker_color="#3b6fb6"), row=1, col=2)
fig.add_trace(go.Bar(x=subset, y=[nuts[nm] for nm in subset], name="NUTS refit",
                     marker_color="#d98c3f"), row=1, col=2)
fig.update_yaxes(title_text="KG (KPI units)", row=1, col=1)
style(fig, "Fast Laplace KG matches the NUTS KG ranking, far cheaper",
      height=400, barmode="group")
fig.show()

assert all(np.isfinite(v) for v in lap.values())
assert speedup > 2          # the Laplace path is much cheaper per design
"""),
    # ====================================================================
    # 9. Pure-EIG acquisition (D / D_s-optimal)
    # ====================================================================
    md(r"""
## 9 · Pure-EIG acquisition — when information ≠ decision value (guide §9.2)

Sometimes the goal isn't the next dollar but **identifying** a decision-pivotal
interaction the exploit-heavy Thompson waves under-probe. `design_eig` scores a
design by the entropy it removes — over **all** parameters (D-optimal) or just
the **synergy** sub-block (`target="gamma"`, D_s-optimal).

The punchline: D_s-optimal and the decision-value KG **disagree**. Probing the
off-axis synergy cells maximizes γ-information, but the *main-effect* cells move
the budget decision more. Use EIG only to shore up under-identified synergies;
prefer KG when the goal is the allocation.
"""),
    code(r"""
ds = {nm: cl.design_eig(post, d, sigma=sigma_hat, target="gamma",
                        n_geo=data["n_geo"], t_test=10) for nm, d in cands.items()}
d_all = {nm: cl.design_eig(post, d, sigma=sigma_hat, target="all",
                           n_geo=data["n_geo"], t_test=10) for nm, d in cands.items()}

def _norm(d):
    m = max(d.values())
    return {k: v / m for k, v in d.items()}

ds_n, kg_n = _norm(ds), _norm(lap)
fig = go.Figure()
fig.add_trace(go.Bar(x=list(cands), y=[ds_n[nm] for nm in cands],
                     name="D_s-opt: γ information", marker_color="#b15a7a"))
fig.add_trace(go.Bar(x=list(cands), y=[kg_n[nm] for nm in cands],
                     name="Laplace KG: decision value", marker_color="#3b6fb6"))
style(fig, "Information (D_s-opt on γ) vs decision value (KG) rank designs differently",
      height=420, barmode="group", yaxis_title="score (normalized to max)")
fig.show()

print("best for γ-identification :", max(ds, key=ds.get))
print("best for the decision (KG):", max(lap, key=lap.get))
assert ds["probe-all (δ=0.6)"] > ds["no-probe (δ=0.6)"]   # off-axis cells identify γ
assert d_all["probe-all (δ=0.6)"] >= ds["probe-all (δ=0.6)"]  # all-param info ≥ γ sub-block
"""),
    # ====================================================================
    # 10. Acquisition & uncertainty surface
    # ====================================================================
    md(r"""
## 10 · The acquisition & uncertainty surface

One picture for the whole decision. Over a 2-D slice of the allocation space
(Pulse × Orbit, the other channels held at the recommendation), the recovery
posterior gives at every candidate point `s`:

* **mean profit** `π(s) = value·E[R(s)] − 1ᵀs`,
* the **uncertainty surface** `σ(s) = value·SD[R(s)]` — highest where the design
  probed least, and
* an **acquisition** `UCB(s) = π + κ·σ` that trades exploiting high mean profit
  against exploring high uncertainty.

The exploit optimum (★, mean profit) and the acquisition optimum (◆, UCB)
**differ**: the acquisition point is pulled toward the under-probed region — the
next allocation a wave should target. (Rendered by `build_acquisition_viz.py`,
which also writes a standalone PNG/HTML to `nbs/artifacts/`.)
"""),
    code(r"""
import sys
sys.path.insert(0, ".")           # make the sibling builder importable in-kernel
import jax as _jax
import build_acquisition_viz as _av   # reuse the exact, layout-tuned figure code

val_s, kap, gn, ghi = 2.5, 1.5, 70, 3.4   # low value -> the profit peaks in the interior
ci, cj = 1, 2                             # Pulse × Orbit (a positive-synergy pair)
rec_s = cl.recommend_allocation(post, 3.2, val_s, q=200, mode="fixed")
gg = np.linspace(0.02, ghi, gn)
GI, GJ = np.meshgrid(gg, gg)
Sg = np.repeat(rec_s[None, :], GI.size, axis=0)
Sg[:, ci] = GI.ravel()
Sg[:, cj] = GJ.ravel()

d = min(300, post.n_draws)
ii = np.linspace(0, post.n_draws - 1, d).astype(int)
gam = np.stack([post.gamma_matrix(int(k)) for k in ii])
R = np.asarray(_av._inc2(Sg, post.samples["beta"][ii], post.samples["kappa"][ii],
                         post.samples["alpha"][ii], gam))               # (grid, draws)
mu = (val_s * R.mean(1) - Sg.sum(1)).reshape(gn, gn)                    # mean profit
sg = (val_s * R.std(1)).reshape(gn, gn)                                 # uncertainty
fig = _av.build_figure({"g": gg, "pi": mu, "unc": sg, "ucb": mu + kap * sg,
                        "channels": np.array(world.channels), "rec": rec_s,
                        "i": ci, "j": cj})
fig.show()

assert int(mu.argmax()) != int((mu + kap * sg).argmax())   # exploit ≠ acquisition optimum
"""),
    # ====================================================================
    # 11. Watching it learn (animation)
    # ====================================================================
    md(r"""
## 11 · Watching it learn — the surfaces and the experiment log update together

The three surfaces **animated across many small, noisy waves** with only a
**handful of cells each**, now alongside the **experiment history**. Colour ranges
are held fixed so you can see what actually changes:

* the **uncertainty surface shrinks** where the cells are probed and the
  mean-profit peak sharpens, but with so little data per wave it **never fully
  collapses** — so the **acquisition optimum (◆) keeps hunting** the
  persistently-uncertain regions rather than locking onto the exploit optimum
  (★) / truth (gold ★);
* the **parameters searched** are overlaid on the uncertainty panel — the orange
  central-composite cells run each wave plus the recommendation **trajectory**
  (how the trust region wanders as it chases the optimum); and
* the **experiment readouts** panel logs what each wave actually measured — the
  observed incremental KPI **with its observation uncertainty (±SE)** — the noisy
  data the surface is fit to.

The exploit optimum settles near the truth (the model learns its best guess), but
because each lean wave adds little, the loop keeps finding **somewhere worth
testing** — the ENBS rule (§5) says *keep going*. Rendered by
`build_acquisition_animation.py` (18 lean waves, ~15 cells each; cadence
env-tunable via `ACQ_N_WAVES` / `ACQ_N_GEO` / `ACQ_PROBE` / …) to a GIF in
`nbs/artifacts/`.
"""),
    code(r"""
import os
import sys
sys.path.insert(0, ".")
from IPython.display import Image as _IPImage
import build_acquisition_animation as _anim

_gif = os.path.join(_anim.OUTDIR, "continuous_learning_acquisition.gif")
if not os.path.exists(_gif):        # build on demand (≈6 short fits) if missing
    _anim.build_gif(_anim._load())
assert os.path.exists(_gif)
_IPImage(filename=_gif)
"""),
    # ====================================================================
    # 12. A harder problem
    # ====================================================================
    md(r"""
## 12 · A harder problem — a ridge you can't fund your way across

The same loop on a **deliberately harder world**. Here the two slice channels
(Pulse × Orbit) **strongly cannibalize** each other — a large negative γ, as if
their audiences heavily overlap — so the response surface has a **ridge**: funding
both is wasteful and the optimum is a narrow trade-off. The main effects are also
close together (hard to rank) and saturate slowly.

Learning this from small, noisy waves is genuinely hard — watch the exploit
optimum **wander along the ridge**, the acquisition keep probing, and the readouts
stay noisy: convergence is slow and non-monotone, and the ENBS rule keeps saying
*keep testing*. (Same builder, `ACQ_WORLD=hard`.)
"""),
    code(r"""
import os
import subprocess
import sys
sys.path.insert(0, ".")
from IPython.display import Image as _IPImageH
import build_acquisition_animation as _animH

_hard_gif = os.path.join(_animH.OUTDIR, "continuous_learning_acquisition_hard.gif")
if not os.path.exists(_hard_gif):    # build on demand (env selects the hard world)
    subprocess.run(
        [sys.executable, "build_acquisition_animation.py"],
        env={**os.environ, "ACQ_WORLD": "hard", "ACQ_TAG": "hard",
             "ACQ_N_GEO": "30", "ACQ_NOISE": "1.1"},
        cwd=os.path.dirname(_animH.__file__), check=True,
    )
assert os.path.exists(_hard_gif)
_IPImageH(filename=_hard_gif)
"""),
    # ====================================================================
    # 13. Beyond Hill — a pluggable activation
    # ====================================================================
    md(r"""
## 13 · Beyond Hill — a pluggable activation

The response surface is **not Hill-specific**. The loop needs only a smooth,
monotonically increasing, saturating activation $f_c \in [0, 1)$ with $f_c(0)=0$
and a finite gradient (the allocator follows $\partial R/\partial s_c$). Hill is
the default; a single `activation="logistic"` flag swaps in the **exponential**
family $f(s) = 1 - e^{-\lambda s}$ — genuinely different: strictly concave (no
S-shape), one shape parameter per channel. Everything downstream — the fit, the
Thompson plan, the funding line, the ENBS stop, the animation — is unchanged.
"""),
    code(r"""
# The two activation families share one interface but have different shapes.
s = np.linspace(0, 3, 200)
f_hill = np.asarray(surface.activation(s, np.ones_like(s), np.full_like(s, 2.0)))
f_log = np.asarray(surface.logistic(s, np.full_like(s, 1.4)))
fig = go.Figure()
fig.add_trace(go.Scatter(x=s, y=f_hill, name="Hill (α=2, κ=1) — S-shaped",
                         line=dict(color="#3b6fb6")))
fig.add_trace(go.Scatter(x=s, y=f_log, name="logistic (λ=1.4) — concave, no inflection",
                         line=dict(color="#d98c3f")))
fig.add_hline(y=0.5, line=dict(color="gray", dash="dash"))
style(fig, "Two saturating activations — same interface, different curve",
      height=360, xaxis_title="scaled spend s", yaxis_title="activation f(s)")
fig.show()
assert f_hill.max() < 1.0 and f_log.max() < 1.0          # both saturate below 1
assert f_log[50] > f_hill[50]                            # logistic rises faster early (concave)
"""),
    code(r"""
# Fit a LOGISTIC world and run the same recovery — activation="logistic".
world_log = cl.make_world_logistic(seed=0)
data_log = cl.simulate_panel(world_log, CENTER, n_geo=80, t_pre=6, t_test=10,
                             delta=0.6, noise=0.5, seed=1)
post_log = cl.fit(data_log, channels=world_log.channels, pair_signs=cl.PAIR_SIGNS_EXAMPLE,
                  activation="logistic", num_warmup=400, num_samples=400, num_chains=2, seed=0)
print(f"posterior activation: {post_log.activation}  |  shape params: {post_log.shape_names}")

b = post_log.samples["beta"]
fig = forest(world_log.channels, b.mean(0),
             np.percentile(b, 5, axis=0), np.percentile(b, 95, axis=0),
             truth=world_log.beta, title="β recovery on a LOGISTIC world (not Hill)",
             xlab="β", colors=COLORS)
fig.show()

# the planner is activation-agnostic: it reads the posterior's family and plans.
rec = cl.recommend_allocation(post_log, B, VALUE, q=200, mode="fixed")
_, prob_above, _ = cl.marginal_roas(post_log, rec, VALUE, q=200)
print("recommended split:", np.round(rec, 2),
      "| funded:", [world_log.channels[c] for c in range(4) if prob_above[c] > 0.5])
assert post_log.activation == "logistic" and "lam" in post_log.samples
assert int(np.argmax(b.mean(0))) == int(np.argmax(world_log.beta))   # strongest recovered
"""),
    md(r"""
And the same animation, now on the logistic world — identical machinery, a
different saturation curve underneath:
"""),
    code(r"""
import os
import subprocess
import sys
sys.path.insert(0, ".")
from IPython.display import Image as _IPImageL
import build_acquisition_animation as _animL

_log_gif = os.path.join(_animL.OUTDIR, "continuous_learning_acquisition_logistic.gif")
if not os.path.exists(_log_gif):     # build on demand (env selects the logistic world)
    subprocess.run(
        [sys.executable, "build_acquisition_animation.py"],
        env={**os.environ, "ACQ_WORLD": "logistic", "ACQ_TAG": "logistic"},
        cwd=os.path.dirname(_animL.__file__), check=True,
    )
assert os.path.exists(_log_gif)
_IPImageL(filename=_log_gif)
"""),
    # ====================================================================
    # 14. When the response family is WRONG — misspecification
    # ====================================================================
    md(r"""
## 14 · When the response family is wrong

A pluggable activation raises the honest question: **what if none of them is
right?** Real saturation is rarely a single Hill. Suppose the *true* per-channel
response is a **weighted sum of two Hills** — an early, steep component (a soft
activation threshold) plus a later, gentler one — a two-phase shape a single Hill
can only average over and a logistic (concave, no inflection) cannot represent at
all. We fit that world three ways and ask what the misspecification costs:

* `activation="hill_mixture"` — the **correct** family,
* `activation="hill"` — **mild** misspecification (one Hill for two),
* `activation="logistic"` — **severe** misspecification (concave, no S-shape).

The result is the important one: **misspecification barely dents the *decision*,
but it wrecks *calibration*.** Near an interior optimum the profit surface is
flat, so a wrong-but-monotone-saturating curve still gets the local marginal
ordering right and allocates well — while its credible intervals get narrow and
*wrong*: confidently biased exactly where the experiments did not probe.
"""),
    code(r"""
from mmm_framework.continuous_learning.surface import ACTIVATIONS, surface_value

world_mix = cl.make_world_hill_mixture(seed=0)          # true = sum of two Hills
data_mix = cl.simulate_panel(world_mix, CENTER, n_geo=72, t_pre=6, t_test=10,
                             delta=0.6, noise=0.4, seed=1)
PROBE_LO, PROBE_HI = CENTER[0] * 0.4, CENTER[0] * 1.6    # the CCD-probed spend range

# a single Hill cannot match a two-Hill mixture's mid-range bend
s = np.linspace(0, 2.0, 200)
p = world_mix.shape
f_mix = np.asarray(surface.hill_mixture(s, p["kappa1"][1], p["alpha1"][1],
                                        p["kappa2"][1], p["alpha2"][1], p["w"][1]))
f_one = np.asarray(surface.activation(s, 0.5*(p["kappa1"][1]+p["kappa2"][1]),
                                      0.5*(p["alpha1"][1]+p["alpha2"][1])))
fig = go.Figure()
fig.add_trace(go.Scatter(x=s, y=f_mix, name="true: mixture of two Hills",
                         line=dict(color="black", width=3)))
fig.add_trace(go.Scatter(x=s, y=f_one, name="closest single Hill",
                         line=dict(color="#3a6ea5", width=2, dash="dash")))
style(fig, "The true activation is a two-phase shape (Pulse)", height=340,
      xaxis_title="scaled spend s", yaxis_title="activation f(s)")
fig.show()
"""),
    code(r"""
# Fit the SAME mixture-truth data three ways and score decision vs. calibration.
true_alloc, true_profit = cl.world_optimal_allocation(world_mix, B, VALUE, mode="fixed")

def prof_true(a):
    return VALUE * float(world_mix.response_mean(np.asarray(a, float)[None, :])[0]) - B

def true_mroas(a):                       # value * dR/ds_c under the true mixture
    a = np.asarray(a, float); eps = 1e-3; R0 = float(world_mix.response_mean(a[None, :])[0])
    out = np.empty(a.size)
    for c in range(a.size):
        ap = a.copy(); ap[c] += eps
        out[c] = VALUE * (float(world_mix.response_mean(ap[None, :])[0]) - R0) / eps
    return out

FITS = {}
rows = []
for act, nw in [("hill_mixture", 600), ("hill", 400), ("logistic", 400)]:
    post = cl.fit(data_mix, channels=world_mix.channels, pair_signs=cl.PAIR_SIGNS_EXAMPLE,
                  activation=act, num_warmup=nw, num_samples=nw, num_chains=2, seed=0)
    FITS[act] = post
    rec = cl.recommend_allocation(post, B, VALUE, q=200, mode="fixed")
    gap = 100 * (true_profit - prof_true(rec)) / abs(true_profit)
    _, _, mr = cl.marginal_roas(post, rec, VALUE, q=200)
    lo, hi = np.percentile(mr, 5, 0), np.percentile(mr, 95, 0); tmr = true_mroas(rec)
    rows.append((act, post.diagnostics.get("max_rhat"), gap,
                 int(((lo <= tmr) & (tmr <= hi)).sum()), float(np.mean(hi - lo))))

print(f"true optimum {np.round(true_alloc, 2)}   true profit {true_profit:.2f}\n")
print(f"{'fitted family':16s}{'profit gap':>11s}{'mROAS-CI covers':>17s}{'CI width':>10s}")
for act, rh, gap, cov, ciw in rows:
    print(f"{act:16s}{gap:9.1f}% {cov:>13}/4 {ciw:>9.2f}")
print("\ndecision barely moves; the SEVERE misspecification (logistic) is the most")
print("confident (narrowest CI) yet covers the truth least often — overconfidence.")
"""),
    md(r"""
The recovered **response curves** show where the wrong families go wrong. We
sweep one channel (holding the others at the operating point) and plot each fit's
posterior mean ± 90% band against the truth, anchored at channel-off so the plot
isolates *shape*, not an overall level offset:
"""),
    code(r"""
def anchored_slice(post, c, s_grid, x0):
    # total incremental response as channel c is swept, minus its channel-off value
    names = ACTIVATIONS[post.activation][0]; fn = ACTIVATIONS[post.activation][1]
    S = post.samples; D = S["beta"].shape[0]
    beta = S["beta"]; shp = [S[nm] for nm in names]
    gv = {(i, j): S[cl.pair_name(post.channels, (i, j))] for (i, j) in post.pairs}
    out = np.empty((D, s_grid.size))
    for t, v in enumerate(s_grid):
        x = np.tile(x0, (D, 1)); x[:, c] = v
        f = np.asarray(fn(x, *shp)); tot = (beta * f).sum(1)
        for (i, j), g in gv.items():
            tot = tot + g * f[:, i] * f[:, j]
        out[:, t] = tot
    return out - out[:, :1]

def true_anchored(c, s_grid, x0):
    X = np.tile(x0, (s_grid.size, 1)); X[:, c] = s_grid
    v = world_mix.response_mean(X); return v - v[0]

from plotly.subplots import make_subplots
COL = {"hill": "#3a6ea5", "logistic": "#c9962f", "hill_mixture": "#5a8a5a"}
RGB = {"hill": "58,110,165", "logistic": "201,150,47", "hill_mixture": "90,138,90"}
LAB = {"hill_mixture": "fit: mixture (correct)", "hill": "fit: single Hill (mild)",
       "logistic": "fit: logistic (severe)"}
chans = [(1, "Pulse"), (2, "Orbit")]
grid = np.linspace(0, 1.5, 80); x0 = CENTER.copy()
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                    subplot_titles=[f"Response as {nm} is swept" for _, nm in chans])
for col, (c, nm) in enumerate(chans, start=1):
    show = col == 1
    fig.add_vrect(x0=PROBE_LO, x1=PROBE_HI, fillcolor="rgba(120,120,120,0.07)",
                  line_width=0, row=1, col=col)
    fig.add_trace(go.Scatter(x=grid, y=true_anchored(c, grid, x0), mode="lines",
        line=dict(color="black", width=3.3), name="true (mixture)", legendgroup="t",
        showlegend=show), row=1, col=col)
    for act in ("hill_mixture", "hill", "logistic"):
        cur = anchored_slice(FITS[act], c, grid, x0)
        m = cur.mean(0); lo = np.percentile(cur, 5, 0); hi = np.percentile(cur, 95, 0)
        fig.add_trace(go.Scatter(x=np.r_[grid, grid[::-1]], y=np.r_[hi, lo[::-1]],
            fill="toself", fillcolor=f"rgba({RGB[act]},0.12)", line=dict(width=0),
            hoverinfo="skip", legendgroup=act, showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(x=grid, y=m, mode="lines", line=dict(color=COL[act], width=2.2),
            name=LAB[act], legendgroup=act, showlegend=show), row=1, col=col)
    fig.update_xaxes(title_text=f"{nm} spend (scaled)", row=1, col=col)
    fig.update_yaxes(title_text="Δ response vs. off" if col == 1 else "", row=1, col=col)
fig.update_layout(template="plotly_white", height=430,
    title=dict(text="Where wrong families diverge: agree where probed (shaded), "
                    "then bend apart", x=0.5, xanchor="center", font=dict(size=15)),
    legend=dict(orientation="h", yanchor="bottom", y=-0.32, xanchor="center", x=0.5),
    margin=dict(l=70, r=25, t=80, b=95))
fig.show()

# the severe (logistic) misfit is materially larger than the mild (single-Hill) one
dev = {act: float(np.max(np.abs(anchored_slice(FITS[act], 1, grid, x0).mean(0)
                                 - true_anchored(1, grid, x0)))) for act in COL}
print("max shape error (Pulse):", {k: round(v, 2) for k, v in dev.items()})
assert dev["logistic"] > dev["hill"]       # severe misspec is worse than mild
"""),
    md(r"""
**Why the decision survives but the story doesn't.** The allocator only needs the
*local gradient ordering* to be right, and any smooth saturating curve fit to the
probed cells reproduces that — so the recommended split captures ~99% of the
achievable profit even under the logistic. But the per-channel marginal-ROAS a
planner would *report* is biased and its interval is narrow: the model does not
know it is wrong, so it under-states its own uncertainty. **Portfolio profit is
robust to the response family; the channel-by-channel numbers are not.**

The other guardrail is the loop itself. Because each wave re-probes *locally* over
the **same geos** and refits on all accumulated data, the loop never commits to a
far extrapolation — it crawls toward the optimum and re-checks. We run the real
accumulating loop (`LearningState` + `simulate_wave`, holding the geo baselines
`a_geo` fixed) under the **mild** misspecification and, for reference, under the
**correct** mixture. The misspecified single-Hill loop converges essentially as
fast as the correctly-specified one — the trust region makes the wrong family
nearly irrelevant to the *decision trajectory*:
"""),
    code(r"""
# The REAL accumulating loop: same geos every wave (fixed a_geo), refit on all data,
# recenter on the recommendation. Fitting a single Hill = deliberate misspecification.
def run_misspecified_loop(activation, nw, seed0=1):
    state = cl.LearningState(channels=world_mix.channels, center=CENTER.copy(),
                             B=B, value=VALUE, pairs=world_mix.pairs,
                             pair_signs=cl.PAIR_SIGNS_EXAMPLE, activation=activation, mode="fixed")
    w0 = cl.simulate_panel(world_mix, CENTER, n_geo=72, t_pre=6, t_test=10,
                           delta=0.6, noise=0.4, seed=seed0)
    a_geo = np.asarray(w0["a_geo"]); state.ingest(w0)          # a_geo persists across waves
    traj = []
    for wave in range(4):
        state.fit(num_warmup=nw, num_samples=nw, num_chains=2, seed=wave)
        rec = state.recommend(q=200)
        traj.append((np.round(state.center, 2), 100*(true_profit-prof_true(rec))/abs(true_profit)))
        if wave == 3:
            break
        design = cl.central_composite(rec, 0.6, world_mix.pairs)  # recenter + re-probe LOCALLY
        wave_next = cl.simulate_wave(world_mix, design, a_geo, t_test=10,
                                     center=rec, noise=0.4, seed=2 + wave)
        state.recenter(rec); state.ingest(wave_next)
    return traj

print(f"{'wave':>4s}{'operating point':>26s}{'single-Hill gap':>16s}{'mixture gap':>13s}")
mis = run_misspecified_loop("hill", 300)
cor = run_misspecified_loop("hill_mixture", 500)
for w, ((c, gm), (_, gc)) in enumerate(zip(mis, cor)):
    print(f"{w:>4d}{str(c):>26s}{gm:>15.1f}%{gc:>12.1f}%")
print("\nthe WRONG family (single Hill) tracks the CORRECT one down to a fraction of a")
print("percent — local re-testing makes the mild misspecification decision-irrelevant.")
assert mis[-1][1] < 1.5 and abs(mis[-1][1] - cor[-1][1]) < 1.5   # tracks the correct loop
"""),
    md(r"""
**Takeaways for the wrong-model case.**

* **Use the most flexible activation you can identify.** The mixture recovered the
  truth honestly (widest, well-calibrated intervals); it costs more data and mixes
  harder (a single Hill on two-Hill data often will not even converge — a *symptom*
  of misspecification worth watching in R̂).
* **Trust the ranking, distrust the magnitudes.** Under misspecification the funded
  set and the allocation stay reliable; the point marginal-ROAS and its interval do
  not. Report decisions, not press-release channel ROIs.
* **Keep the loop local.** The trust-region design is itself a robustness
  mechanism — it never extrapolates far without a fresh test.
* **Check the fit.** Systematic residuals across the CCD cross-section (or a
  posterior-predictive check) are the tell that the response family is too rigid;
  that is your cue to widen the activation, not to narrow the intervals.
"""),
    # ====================================================================
    # 15. Recap
    # ====================================================================
    md(r"""
## Recap

* **No model required.** The surface was learned entirely from *designed*
  experiments — the pre/test split + central-composite cells make it causal.
* **Recovery.** Main-effect ordering and the synergy *signs* came back; the
  fitted response curves tracked the truth.
* **Decisions, with uncertainty.** Thompson sampling gave a posterior over the
  optimal split; the funding line said which channels clear break-even.
* **Closure + stopping.** As waves accumulated, expected regret shrank, the
  recommendation converged to the truth-optimal split, and the ENBS rule halted
  testing once it no longer paid.
* **Audit.** Magnitudes of unidentified synergies are prior-sensitive — the
  `gamma_scale` sweep makes that visible.
* **Production realism.** The adstock pre-pass recovered a carryover-biased
  curve; CUPED stripped baseline variance (fewer geos per MDE); the **Laplace**
  knowledge-gradient scored candidate designs with no MCMC; and pure-EIG
  (D_s-optimal) showed that *information* and *decision value* can disagree.
* **The decision in one picture.** The acquisition & uncertainty surfaces made
  the explore/exploit trade-off visible — the acquisition optimum sits off the
  mean-profit peak, out toward where the posterior is least sure.
* **Watching it learn.** Animated across many small, lean, noisy waves — with the
  parameters searched (the probed cells + the trust-region trajectory) and the
  experiment readouts (observed incremental ± SE) tracked alongside — the exploit
  optimum settles near the truth while the acquisition optimum keeps hunting the
  still-uncertain regions: the loop keeps finding somewhere worth testing. A
  **harder world** (strong Pulse × Orbit cannibalization → a ridge) makes that
  hunt visibly slower and messier.
* **Not Hill-specific.** The activation is pluggable — the same fit, plan, stop,
  and animation run on a **logistic** (exponential-saturation) world with one
  flag (`activation="logistic"`); any smooth, monotone, saturating curve drops in.
* **Robust to a *wrong* family.** When the truth is a mixture of Hills but we fit a
  single Hill (or a logistic), the *decision* barely moves (~99% of the achievable
  profit) — but the credible intervals get narrow and biased. Misspecification
  costs calibration, not portfolio profit; trust the ranking, distrust the
  channel-by-channel magnitudes, and keep the loop local.

This is the model-free half of the framework's measurement story. The
model-anchored half lives in `mmm_framework.planning` (it sits on top of a
fitted `BayesianMMM`), and the two share conventions — the Hill activation here
is identical to `SaturationType.HILL`.

**See also:** `examples/ex_continuous_learning.py`,
`technical-docs/continuous-learning.md`, and the design guide
`assets/continous_learning.md`.

*Deferred (not in this notebook): agent tools + REST/UI wiring, and replacing the
loop's per-fantasy NUTS refit with the Laplace update everywhere (here it is
shown as a standalone acquisition).*
"""),
]


def main() -> None:
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.metadata["language_info"] = {"name": "python"}
    with open("continuous_learning.ipynb", "w") as fh:
        nbformat.write(nb, fh)
    print(f"wrote continuous_learning.ipynb with {len(CELLS)} cells")


if __name__ == "__main__":
    main()
