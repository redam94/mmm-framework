"""Author workshop notebook 02 — Sampling: How the Machine Works (run from ``nbs/``).

    uv run python builders/build_workshop_02_sampling.py
    PYTHONPATH=.. uv run jupyter nbconvert --to notebook --execute --inplace \
        workshop/workshop_02_sampling.ipynb --ExecutePreprocessor.timeout=2400

Notebook 02 of the 6-part *workshop* series: the Bayesian causal workflow for
marketing analysts with no Bayesian background. This one answers "how does the
computer actually get the posterior?": the curse of dimensionality, Monte Carlo,
a hand-built Metropolis sampler (animated), multiple chains and R-hat, effective
sample size, NUTS, divergences, and a trace-plot reading clinic — ending with
the crucial caveat that green diagnostics validate the computation, not the model.

Authored as md/code cells via nbformat (pattern: ``build_stress_00_rosy_picture.py``).
Main charts are plotly (animation frames for the darts + walker demos); three
``ipywidgets.interact`` live-exploration cells, each paired with a static
multi-panel figure. Every computational cell ends with asserts encoding its
claim — seeded, and directional rather than tight for anything MCMC.
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
    # 1. Hook: the curse of dimensionality
    # ====================================================================
    md(r"""
# Workshop 02 — Sampling: How the Machine Works

In [workshop 00](workshop_00_thinking_in_distributions.ipynb) you met the
posterior, and you computed it the honest way: lay a fine grid over every
possible value of the conversion rate, evaluate prior × likelihood at each
point, normalize. Done. For **one** unknown, that works beautifully.

Now look ahead to [workshop 03](workshop_03_first_mmm.ipynb), where you'll fit
your first marketing mix model. Even a small MMM has **dozens of unknowns**:
a coefficient, a carryover rate, and a saturation parameter *per channel*, plus
trend, seasonality, and noise terms — call it 40 parameters. Try the grid trick
there: with just 100 grid points per parameter, you need $100^{40} = 10^{80}$
evaluations. That is roughly the number of **atoms in the observable universe**.
No computer will ever do it. This explosion is called the **curse of
dimensionality** — *every grid-style method becomes impossible as the number of
parameters grows, because the number of grid cells multiplies with every new
parameter*.

So how does PyMC hand you a posterior over 40 parameters in a minute or two?
**It never maps the whole posterior. It draws samples from it.** This notebook
opens the hood:

1. **Monte Carlo** — answering questions with random samples (the dart-throwing π demo).
2. **MCMC** — how to sample from a posterior you can only *evaluate*, not draw from.
3. **Metropolis in ~20 lines** — build a real sampler, watch it walk, break it with bad step sizes.
4. **Multiple chains and R-hat** — how we *check* the walker did its job.
5. **Effective sample size** — why 20,000 correlated draws may be worth only a few hundred.
6. **NUTS** — the modern engine PyMC (and the MMM framework) actually uses, and its **divergences** warning light.
7. **A trace-plot reading clinic** — four patterns you'll meet in the wild.

No new probability theory today — just the machinery. By the end, the
diagnostics table that a fitted MMM prints will read like a story, not a
checklist.
"""),
    code(r"""
import warnings, logging, time, contextlib, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import arviz as az
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
# pymc (imported later) emits sampler warnings at ERROR level -> CRITICAL keeps outputs clean
for _n in ("pymc", "numpyro", "jax", "arviz", "pytensor"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

@contextlib.contextmanager
def quiet():
    'Hide the samplers progress bars / chatter; our own prints stay visible.'
    with open(os.devnull, "w") as _dn, contextlib.redirect_stdout(_dn), \
            contextlib.redirect_stderr(_dn):
        yield

pio.renderers.default = "notebook_connected"
plt.rcParams.update({
    "figure.figsize": (9, 4), "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110,
})
INK, SKY, BERRY, LEAF, AMBER, MUTED = (
    "#2b2118", "#3b6ea5", "#a63a50", "#3f7d5e", "#d98a2b", "#8a8079")

def rhat_of(arr):
    'Rank-normalized split R-hat of a (chains, draws) array, via arviz.'
    return float(az.rhat(az.convert_to_dataset(np.asarray(arr)))["x"])

def ess_of(arr):
    'Bulk effective sample size of a (chains, draws) array, via arviz.'
    return float(az.ess(az.convert_to_dataset(np.asarray(arr)))["x"])

assert pio.renderers.default == "notebook_connected"
assert callable(quiet) and callable(rhat_of) and callable(ess_of)
print("setup ok — plotly renderer:", pio.renderers.default)
"""),
    # ====================================================================
    # 2. Grid blow-up
    # ====================================================================
    md(r"""
## 1 — Why the grid dies: count the cells

Let's actually count. Suppose evaluating prior × likelihood at one grid point
takes a nanosecond — a generous **one billion evaluations per second**. Here is
how long a 100-points-per-axis grid takes as the model grows:
"""),
    code(r"""
GRID_PTS = 100
EVALS_PER_SEC = 1e9
AGE_OF_UNIVERSE_S = 4.35e17     # ~13.8 billion years, in seconds
ATOMS_IN_UNIVERSE = 1e80        # standard order-of-magnitude estimate

def fmt_duration(seconds):
    'Human-friendly duration, up to multiples of the age of the universe.'
    if seconds < 1:
        return f"{seconds * 1000:.3g} ms"
    if seconds < 3600:
        return f"{seconds:.3g} s"
    years = seconds / 3.154e7
    if years < 1e4:
        return f"{years:,.3g} years"
    if seconds < AGE_OF_UNIVERSE_S:
        return f"{years:.2e} years"
    return f"{seconds / AGE_OF_UNIVERSE_S:.2e} x the age of the universe"

rows = []
for d in [1, 2, 3, 5, 10, 20, 40]:
    evals = float(GRID_PTS) ** d
    rows.append({
        "parameters": d,
        "grid cells to evaluate": f"10^{2 * d}",
        "time at 1 billion evals/sec": fmt_duration(evals / EVALS_PER_SEC),
    })
grid_cost = pd.DataFrame(rows).set_index("parameters")
display(grid_cost)

# CLAIM: a 40-parameter grid needs more evaluations than there are atoms in the
# observable universe, while the 1-parameter grid from workshop_00 is instant.
assert float(GRID_PTS) ** 40 >= ATOMS_IN_UNIVERSE
assert float(GRID_PTS) ** 1 / EVALS_PER_SEC < 1e-3
print("✓ the grid that took microseconds in workshop_00 is physically impossible at MMM scale")
"""),
    # ====================================================================
    # 3. Monte Carlo refresher: darts
    # ====================================================================
    md(r"""
## 2 — Monte Carlo: answer questions by sampling

Here is the escape hatch. Instead of *mapping* a quantity everywhere, you can
often *estimate* it from random samples. This is the **Monte Carlo method** —
*replace an exact calculation with the average of many random draws* — and it
has a magical property: **its cost doesn't depend on the number of dimensions**,
only on how many samples you take.

The classic demo: estimate π by throwing darts. Throw darts uniformly at a
1×1 square. The quarter-circle of radius 1 inside it has area π/4, so the
fraction of darts landing inside it ≈ π/4, and

$$\hat{\pi} = 4 \times \frac{\text{darts inside}}{\text{darts thrown}}.$$

No geometry, no calculus — just counting. Press **play** (or drag the slider)
and watch the estimate sharpen as darts accumulate:
"""),
    code(r"""
rng = np.random.default_rng(0)
N_DARTS = 5000
darts = rng.random((N_DARTS, 2))
inside_mask = (darts ** 2).sum(axis=1) <= 1.0
running_est = 4 * np.cumsum(inside_mask) / np.arange(1, N_DARTS + 1)

arc_t = np.linspace(0, np.pi / 2, 200)
arc = go.Scatter(x=np.cos(arc_t), y=np.sin(arc_t), mode="lines",
                 line=dict(color=INK, width=2), name="quarter circle")

frame_ns = [25, 50, 100, 250, 500, 1000, 2500, 5000]
frames = []
for n in frame_ns:
    pin, pout = darts[:n][inside_mask[:n]], darts[:n][~inside_mask[:n]]
    frames.append(go.Frame(
        name=str(n), traces=[1, 2],
        data=[go.Scatter(x=pin[:, 0], y=pin[:, 1], mode="markers",
                         marker=dict(color=LEAF, size=4, opacity=0.6), name="inside"),
              go.Scatter(x=pout[:, 0], y=pout[:, 1], mode="markers",
                         marker=dict(color=BERRY, size=4, opacity=0.6), name="outside")],
        layout=go.Layout(title_text=(
            f"n = {n} darts   →   π̂ = {running_est[n-1]:.4f}"
            f"   (truth 3.1416)"))))

fig = go.Figure(
    data=[arc, frames[0].data[0], frames[0].data[1]],
    frames=frames,
    layout=go.Layout(
        title=f"n = {frame_ns[0]} darts   →   π̂ = {running_est[frame_ns[0]-1]:.4f}   (truth 3.1416)",
        width=560, height=580, xaxis=dict(range=[0, 1], constrain="domain"),
        yaxis=dict(range=[0, 1], scaleanchor="x"), showlegend=False,
        updatemenus=[dict(type="buttons", x=0, y=1.18, xanchor="left", showactive=False,
            buttons=[dict(label="▶ Play", method="animate",
                          args=[None, dict(frame=dict(duration=700, redraw=False),
                                           transition=dict(duration=0), fromcurrent=True)]),
                     dict(label="⏸ Pause", method="animate",
                          args=[[None], dict(frame=dict(duration=0, redraw=False),
                                             mode="immediate")])])],
        sliders=[dict(currentvalue=dict(prefix="darts: "), pad=dict(t=30),
            steps=[dict(method="animate", label=f.name,
                        args=[[f.name], dict(mode="immediate",
                                             frame=dict(duration=0, redraw=False))])
                   for f in frames])]))
fig.show()

# CLAIM: with 5000 seeded darts the estimate lands close to pi, and each frame
# partitions exactly its n darts into inside + outside.
assert abs(running_est[-1] - np.pi) < 0.05, running_est[-1]
for n, f in zip(frame_ns, fig.frames):
    assert len(f.data[0].x) + len(f.data[1].x) == n
print(f"✓ final estimate {running_est[-1]:.4f} vs π = {np.pi:.4f}")
"""),
    md(r"""
How fast does the dart answer improve? Watch the error as we vary the number of
darts — averaging over 200 repeat experiments per size so we see the *typical*
error, not one lucky run. On a log–log plot the error falls on a straight line
of slope ≈ −½: Monte Carlo error shrinks like $1/\sqrt{n}$. That's the
**root-n rate** — *to cut the error in half, you need four times the samples*.
Slow, but it never gets slower in higher dimensions — which is the whole trade.
"""),
    code(r"""
ns = np.array([100, 300, 1000, 3000, 10000, 30000])
B = 200
rng = np.random.default_rng(42)
rmse = []
for n in ns:
    pts = rng.random((B, n, 2))
    est = 4 * ((pts ** 2).sum(axis=2) <= 1.0).mean(axis=1)
    rmse.append(float(np.sqrt(((est - np.pi) ** 2).mean())))
rmse = np.array(rmse)
slope = float(np.polyfit(np.log(ns), np.log(rmse), 1)[0])

# Static multi-panel companion to the interactive demos: three dart boards + the rate.
show_ns = [100, 1000, 10000]
fig = make_subplots(rows=1, cols=4, column_widths=[0.22, 0.22, 0.22, 0.34],
                    subplot_titles=[f"n = {n}" for n in show_ns]
                    + [f"typical error vs n (slope ≈ {slope:.2f})"])
rngp = np.random.default_rng(7)
for j, n in enumerate(show_ns, start=1):
    p = rngp.random((n, 2)); m = (p ** 2).sum(axis=1) <= 1
    fig.add_trace(go.Scatter(x=p[m, 0], y=p[m, 1], mode="markers",
                             marker=dict(color=LEAF, size=2.5, opacity=0.5)), 1, j)
    fig.add_trace(go.Scatter(x=p[~m, 0], y=p[~m, 1], mode="markers",
                             marker=dict(color=BERRY, size=2.5, opacity=0.5)), 1, j)
    fig.add_trace(go.Scatter(x=np.cos(arc_t), y=np.sin(arc_t), mode="lines",
                             line=dict(color=INK, width=1.5)), 1, j)
    fig.update_xaxes(range=[0, 1], row=1, col=j, showticklabels=False)
    fig.update_yaxes(range=[0, 1], row=1, col=j, showticklabels=False)
fig.add_trace(go.Scatter(x=ns, y=rmse, mode="lines+markers",
                         line=dict(color=SKY, width=2), name="measured error"), 1, 4)
ref = rmse[0] * np.sqrt(ns[0] / ns)
fig.add_trace(go.Scatter(x=ns, y=ref, mode="lines",
                         line=dict(color=AMBER, dash="dash"), name="1/√n reference"), 1, 4)
fig.update_xaxes(type="log", title_text="darts n", row=1, col=4)
fig.update_yaxes(type="log", title_text="RMS error of π̂", row=1, col=4)
fig.update_layout(width=1050, height=330, showlegend=False,
                  title="Monte Carlo error falls like 1/√n")
fig.show()

# CLAIM: the measured convergence rate is the root-n rate (slope ~ -0.5 on
# log-log), and going 100 -> 30,000 darts cuts typical error at least 5-fold.
assert -0.65 < slope < -0.35, slope
assert np.all(np.diff(rmse) < 0), "typical error should fall monotonically with n"
assert rmse[-1] < rmse[0] / 5
print(f"✓ log-log slope {slope:.2f} ≈ -0.5; error shrank {rmse[0]/rmse[-1]:.0f}x from n=100 to n=30,000")
"""),
    md(r"""
### 🎛️ Live exploration (run me!) — throw your own darts

Drag `n` and watch the estimate stabilize; change `seed` to see that *which*
random darts you get doesn't matter once `n` is large. (Static reference:
the panel above shows representative settings if you're reading a saved copy.)
"""),
    code(r"""
from ipywidgets import interact, IntSlider, FloatSlider, FloatLogSlider

def throw_darts(n=500, seed=0):
    'Throw n uniform darts, plot them, return the pi estimate.'
    rng = np.random.default_rng(seed)
    p = rng.random((n, 2)); m = (p ** 2).sum(axis=1) <= 1
    est = 4 * m.mean()
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.scatter(*p[m].T, s=4, color=LEAF, alpha=0.5)
    ax.scatter(*p[~m].T, s=4, color=BERRY, alpha=0.5)
    ax.plot(np.cos(arc_t), np.sin(arc_t), color=INK, lw=1.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_title(f"n={n}, seed={seed}:  π̂ = {est:.4f}   (error {abs(est-np.pi):.4f})")
    plt.show()
    return est

interact(throw_darts,
         n=IntSlider(value=500, min=50, max=20000, step=50),
         seed=IntSlider(value=0, min=0, max=10));

# CLAIM (headless check at a representative setting): plenty of darts -> close to pi.
assert abs(throw_darts(20000, seed=0) - np.pi) < 0.1
print("✓ interactive dart thrower works; large n lands near π")
"""),
    # ====================================================================
    # 4. The problem: we can't sample the posterior directly
    # ====================================================================
    md(r"""
## 3 — One catch: you can't throw darts *at a posterior*

The dart trick worked because we know how to draw uniform points in a square.
But the thing we want to sample in Bayesian inference is the **posterior** —
and for any real model, *nobody knows how to draw from it directly*.

Here's the asymmetry that defines the whole problem. From workshop_00, Bayes'
rule says

$$p(\theta \mid \text{data}) = \frac{p(\text{data} \mid \theta)\, p(\theta)}{p(\text{data})}.$$

The numerator — likelihood × prior — is easy: pick any candidate $\theta$,
plug in, get a number. The denominator $p(\text{data})$ is the killer: it's the
sum of the numerator over **all possible** $\theta$ — exactly the
universe-sized grid we just gave up on. So our situation is:

- ✅ we **can** compute a function *proportional to* the posterior at any point we choose,
- ❌ we **cannot** compute the normalizing constant, and we cannot draw samples directly.

The solution, and the central idea of this notebook, is **MCMC — Markov chain
Monte Carlo**: *send a walker wandering through parameter space, with movement
rules cleverly rigged so that, in the long run, the walker spends time in each
region in exact proportion to its posterior probability*. Then the walker's
visit log **is** a sample from the posterior — and the rules only ever need
the computable numerator. The unknowable denominator cancels out.

Let's make this concrete with a target we know the answer to. Recycle the
workshop_00 story: an email campaign got **12 signups from 100 visitors**, and
your prior on the conversion rate θ was Beta(2, 8). The grid (or conjugate
algebra) tells us the exact posterior — so we can grade our sampler against
truth:
"""),
    code(r"""
A_PRIOR, B_PRIOR = 2, 8          # prior: Beta(2, 8) — "most plausibly ~10-25%"
N_VIS, CONV = 100, 12            # data: 12 signups from 100 visitors
A_POST, B_POST = A_PRIOR + CONV, B_PRIOR + N_VIS - CONV   # exact posterior: Beta(14, 96)
TRUE_MEAN = A_POST / (A_POST + B_POST)
TRUE_SD = float(stats.beta(A_POST, B_POST).std())

def log_post(theta):
    'log(prior x likelihood) up to a constant — the only thing MCMC ever needs.'
    if theta <= 0.0 or theta >= 1.0:
        return -np.inf
    return (A_POST - 1) * np.log(theta) + (B_POST - 1) * np.log(1 - theta)

grid = np.linspace(1e-4, 1 - 1e-4, 600)
unnorm = np.exp([log_post(t) for t in grid])
exact_pdf = stats.beta(A_POST, B_POST).pdf(grid)

fig = make_subplots(rows=1, cols=2, subplot_titles=(
    "what we CAN compute: prior × likelihood (unnormalized)",
    "what we WANT: the normalized posterior"))
fig.add_trace(go.Scatter(x=grid, y=unnorm, line=dict(color=AMBER, width=2.5)), 1, 1)
fig.add_trace(go.Scatter(x=grid, y=exact_pdf, line=dict(color=SKY, width=2.5)), 1, 2)
fig.update_xaxes(title_text="conversion rate θ", range=[0, 0.4])
fig.update_yaxes(title_text="unnormalized height", row=1, col=1)
fig.update_yaxes(title_text="posterior density", row=1, col=2)
fig.update_layout(width=950, height=340, showlegend=False,
                  title="Same shape, unknown scale — sampling only needs the shape")
fig.show()

# CLAIM: the unnormalized curve is exactly proportional to the true posterior —
# dividing one by the other gives a constant (where there is any mass).
ratio = unnorm / exact_pdf
core = (grid > 0.03) & (grid < 0.35)
assert np.allclose(ratio[core], ratio[core].mean(), rtol=1e-6)
assert abs(np.trapezoid(exact_pdf, grid) - 1) < 1e-3
print(f"✓ unnormalized/normalized ratio is constant; exact posterior mean = {TRUE_MEAN:.4f}, sd = {TRUE_SD:.4f}")
"""),
    # ====================================================================
    # 5. Metropolis in ~20 lines
    # ====================================================================
    md(r"""
## 4 — Build a sampler in ~20 lines: the Metropolis algorithm

The oldest MCMC recipe (Metropolis et al., 1953 — it predates the moon landing)
is short enough to memorize. The walker stands at position $x$ and repeats:

1. **Propose** a move: pick a candidate $x' = x + \text{noise}$, e.g. a Gaussian
   step. The candidate-generating rule is the **proposal** — *how the walker
   suggests its next location*.
2. **Compare heights**: compute the ratio of (unnormalized!) posterior heights,
   $r = \tilde p(x') / \tilde p(x)$. The unknown normalizing constant divides out — this is the trick.
3. **Accept or reject**: if the candidate is uphill ($r \ge 1$), always move.
   If downhill, move with probability $r$ — flip a biased coin. If rejected,
   *stay put and record the current position again*. The fraction of proposals
   taken is the **acceptance rate**.

Rule 3 is the genius part: always rising would find the peak and stop (that's
optimization); accepting downhill moves *in proportion to relative height*
makes the time spent at each location proportional to posterior height. The
recorded list of positions is called a **chain** — *the walker's visit log,
which becomes our bag of posterior samples*.
"""),
    code(r"""
def metropolis(log_target, start, step, n, seed, details=False):
    'Random-walk Metropolis: n steps from start, Gaussian proposals of sd=step.'
    rng = np.random.default_rng(seed)
    chain = np.empty(n)
    props = np.empty(n)
    accepted = np.zeros(n, dtype=bool)
    x, lx, n_acc = float(start), log_target(start), 0
    for i in range(n):
        prop = x + rng.normal(0.0, step)            # 1. propose
        lp = log_target(prop)                       # 2. compare heights (in logs)
        if np.log(rng.uniform()) < lp - lx:         # 3. accept downhill with prob r
            x, lx, n_acc, accepted[i] = prop, lp, n_acc + 1, True
        chain[i], props[i] = x, prop                # rejected -> record x again
    return (chain, n_acc / n, props, accepted) if details else (chain, n_acc / n)

STEP_GOLD = 0.075        # ~2.4x the posterior sd — a classic good choice
t0 = time.perf_counter()
chain0, acc0 = metropolis(log_post, start=0.5, step=STEP_GOLD, n=20000, seed=0)
T_METRO = time.perf_counter() - t0
BURN0 = 2000
kept0 = chain0[BURN0:]
print(f"20,000 steps in {T_METRO:.2f}s   acceptance rate {acc0:.0%}")
print(f"sample mean {kept0.mean():.4f}  vs exact posterior mean {TRUE_MEAN:.4f}")
print(f"sample sd   {kept0.std():.4f}  vs exact posterior sd   {TRUE_SD:.4f}")

# CLAIM: ~20 lines of numpy, knowing ONLY the unnormalized log-height, recover
# the exact Beta(14, 96) posterior's mean and sd (seeded, generous tolerance).
assert abs(kept0.mean() - TRUE_MEAN) < 0.01
assert abs(kept0.std() - TRUE_SD) < 0.01
assert 0.15 < acc0 < 0.70, acc0
print("✓ a 20-line walker reproduced the exact posterior — no normalizing constant needed")
"""),
    md(r"""
### Watch the walker think

Each frame below is one step of the *same* algorithm you just read. The
<span style="color:#3f7d5e">**green diamond**</span> is an accepted proposal,
the <span style="color:#a63a50">**red diamond**</span> a rejected one (the
walker stays put). Blue dots pile up where the walker has been — directly
under the bulk of the target curve. The walker starts at θ = 0.5, in the
posterior's wasteland, and you can watch it *find* the high-probability region
in the first dozen moves:
"""),
    code(r"""
START_ANIM, N_ANIM = 0.5, 120
chain_a, acc_a, props_a, accs_a = metropolis(
    log_post, start=START_ANIM, step=STEP_GOLD, n=N_ANIM, seed=1, details=True)
before = np.concatenate([[START_ANIM], chain_a[:-1]])     # position before each step

pdf_scale = exact_pdf.max()
Y_WALK, Y_HIST = -0.9, -2.2
rngj = np.random.default_rng(2)
jitter = rngj.uniform(-0.6, 0.6, N_ANIM)

frames = []
for i in range(N_ANIM):
    col = LEAF if accs_a[i] else BERRY
    verdict = "accepted ✓" if accs_a[i] else "rejected ✗"
    frames.append(go.Frame(
        name=str(i + 1), traces=[1, 2, 3, 4],
        data=[
            go.Scatter(x=chain_a[:i], y=Y_HIST + jitter[:i], mode="markers",
                       marker=dict(color=SKY, size=4, opacity=0.45)),
            go.Scatter(x=[before[i]], y=[Y_WALK], mode="markers",
                       marker=dict(color=INK, size=12, symbol="circle")),
            go.Scatter(x=[before[i], props_a[i]], y=[Y_WALK, Y_WALK], mode="lines",
                       line=dict(color=col, width=1.5, dash="dot")),
            go.Scatter(x=[props_a[i]], y=[Y_WALK], mode="markers",
                       marker=dict(color=col, size=10, symbol="diamond")),
        ],
        layout=go.Layout(title_text=(
            f"step {i + 1}: propose {props_a[i]:.3f} → {verdict}"
            f"   (acceptance so far {accs_a[:i+1].mean():.0%})"))))

fig = go.Figure(
    data=[go.Scatter(x=grid, y=exact_pdf, mode="lines",
                     line=dict(color=AMBER, width=2.5), name="target (shape only)"),
          frames[0].data[0], frames[0].data[1], frames[0].data[2], frames[0].data[3]],
    frames=frames,
    layout=go.Layout(
        width=820, height=460, showlegend=False,
        title=f"step 1: the walker starts at θ = {START_ANIM}",
        xaxis=dict(title="conversion rate θ", range=[0, 0.62]),
        yaxis=dict(range=[Y_HIST - 1.2, pdf_scale * 1.06], showticklabels=False,
                   title="target height  /  walker lane below"),
        updatemenus=[dict(type="buttons", x=0, y=1.15, xanchor="left", showactive=False,
            buttons=[dict(label="▶ Play", method="animate",
                          args=[None, dict(frame=dict(duration=150, redraw=False),
                                           transition=dict(duration=0), fromcurrent=True)]),
                     dict(label="⏸ Pause", method="animate",
                          args=[[None], dict(frame=dict(duration=0, redraw=False),
                                             mode="immediate")])])],
        sliders=[dict(currentvalue=dict(prefix="step: "), pad=dict(t=25),
            steps=[dict(method="animate", label=f.name,
                        args=[[f.name], dict(mode="immediate",
                                             frame=dict(duration=0, redraw=False))])
                   for f in frames])]))
fig.add_hline(y=0, line_width=0.5, line_color=MUTED)
fig.show()

# CLAIM: the animation covers all steps; the seeded walker leaves the bad start
# behind (late positions sit near the mode, far below the 0.5 start).
assert len(fig.frames) == N_ANIM
assert abs(np.median(chain_a[60:]) - TRUE_MEAN) < 4 * TRUE_SD
assert chain_a[60:].max() < START_ANIM - 0.2
print(f"✓ walker found the mode: median of late steps {np.median(chain_a[60:]):.3f} ≈ {TRUE_MEAN:.3f}")
"""),
    md(r"""
Two things the animation makes obvious:

- **The first steps are garbage.** The walker began at θ = 0.5 because *we* put
  it there; those early positions reflect our arbitrary start, not the
  posterior. The convention is to discard the early portion — the **burn-in**
  (PyMC calls it **warm-up/tune**) — *the initial stretch of the chain thrown
  away because the walker hadn't yet forgotten where it started*.
- **Piling up = sampling.** Once settled, the walker revisits high regions often
  and low regions rarely. Histogram its visit log and you should recover the
  target curve. Let's check that on the long run from before:
"""),
    code(r"""
fig = go.Figure()
fig.add_trace(go.Histogram(x=kept0, histnorm="probability density", nbinsx=60,
                           marker_color=SKY, opacity=0.65, name="walker's visit log"))
fig.add_trace(go.Scatter(x=grid, y=exact_pdf, mode="lines",
                         line=dict(color=AMBER, width=3), name="exact posterior"))
fig.update_layout(width=820, height=400, barmode="overlay",
                  title=f"18,000 post-burn-in visits vs the exact Beta({A_POST}, {B_POST}) posterior",
                  xaxis=dict(title="conversion rate θ", range=[0.02, 0.28]),
                  yaxis_title="density")
fig.show()

# CLAIM: the visit-log histogram matches the exact posterior — including the
# tails (central 90% interval endpoints land near the exact quantiles).
q_exact = stats.beta(A_POST, B_POST).ppf([0.05, 0.95])
q_chain = np.quantile(kept0, [0.05, 0.95])
assert np.all(np.abs(q_chain - q_exact) < 0.012), (q_chain, q_exact)
print(f"✓ 90% interval from the chain [{q_chain[0]:.3f}, {q_chain[1]:.3f}] "
      f"vs exact [{q_exact[0]:.3f}, {q_exact[1]:.3f}]")
"""),
    md(r"""
### The one knob that can ruin everything: step size

Our proposal was `Gaussian(0, 0.075)`. Why 0.075? Try the extremes:

- **Tiny steps** — almost every proposal is accepted (nearby points have nearly
  the same height), but the walker inches along like a snail. Thousands of
  steps explore a sliver of the posterior.
- **Huge steps** — proposals usually leap into the wasteland and get rejected,
  so the walker mostly *stands still*, recording the same value over and over.
- **Goldilocks** — moderate rejection is the *price of real movement*. For a
  1-D target the sweet spot is an acceptance rate around 30–50%; high-dimensional
  theory says ~23%. **A high acceptance rate is not a good sign by itself.**
"""),
    code(r"""
STEP_TINY, STEP_HUGE = 0.002, 1.5
N_STEPS, BURN = 6000, 1000
STEPS = {"tiny (0.002)": STEP_TINY, "goldilocks (0.075)": STEP_GOLD, "huge (1.5)": STEP_HUGE}
CH, ACC = {}, {}
for lbl, s in STEPS.items():
    runs = [metropolis(log_post, TRUE_MEAN, s, N_STEPS, seed=100 + 10 * k)
            for k in range(4)]                      # 4 chains, started at the mode
    CH[lbl] = np.array([r[0] for r in runs])
    ACC[lbl] = float(np.mean([r[1] for r in runs]))

fig = make_subplots(rows=2, cols=3, vertical_spacing=0.14,
                    subplot_titles=[f"{l} — acceptance {ACC[l]:.0%}" for l in STEPS]
                    + [f"visits vs target — {l.split(' ')[0]}" for l in STEPS])
for j, lbl in enumerate(STEPS, start=1):
    tr = CH[lbl][0, BURN:BURN + 2000]
    fig.add_trace(go.Scatter(y=tr, mode="lines", line=dict(color=SKY, width=0.8)), 1, j)
    fig.add_trace(go.Histogram(x=CH[lbl][:, BURN:].ravel(), histnorm="probability density",
                               nbinsx=50, marker_color=SKY, opacity=0.6), 2, j)
    fig.add_trace(go.Scatter(x=grid, y=exact_pdf, mode="lines",
                             line=dict(color=AMBER, width=2)), 2, j)
    fig.update_yaxes(range=[0.02, 0.28], row=1, col=j)
    fig.update_xaxes(range=[0.02, 0.28], row=2, col=j, title_text="θ")
fig.update_yaxes(title_text="θ (2000 steps)", row=1, col=1)
fig.update_yaxes(title_text="density", row=2, col=1)
fig.update_layout(width=1050, height=600, showlegend=False,
                  title="Step size: tiny = snail, huge = statue, goldilocks = sampler")
fig.show()

# CLAIM: acceptance ordering tiny > goldilocks > huge; tiny accepts almost
# everything (and that is BAD), huge rejects almost everything.
assert ACC["tiny (0.002)"] > 0.90
assert ACC["huge (1.5)"] < 0.15
assert ACC["tiny (0.002)"] > ACC["goldilocks (0.075)"] > ACC["huge (1.5)"]
# Only goldilocks nails the posterior sd in this budget; tiny under-explores.
sd_tiny = CH["tiny (0.002)"][0, BURN:].std()
assert sd_tiny < 0.8 * TRUE_SD, sd_tiny
assert abs(CH["goldilocks (0.075)"][:, BURN:].std() - TRUE_SD) < 0.01
print("✓ acceptance: tiny > goldilocks > huge — and the 99%-acceptance chain explored the least")
"""),
    md(r"""
### 🎛️ Live exploration (run me!) — drive the step size yourself

Drag `step` across three orders of magnitude. Watch the trace (left) change
character: fuzzy caterpillar (good), smooth wandering river (too small), or
flat staircase (too big) — and watch the histogram's match to the target
degrade at both extremes. The static panel above shows the three regimes if
you're reading a non-live copy.
"""),
    code(r"""
def explore_step(step=0.075, n=4000):
    'Run a fresh seeded chain at this step size; plot trace + histogram.'
    ch, acc = metropolis(log_post, 0.5, step, n, seed=0)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    axes[0].plot(ch, lw=0.6, color=SKY)
    axes[0].set_title(f"trace — step={step:.4g}, acceptance {acc:.0%}")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("θ")
    axes[1].hist(ch[n // 5:], bins=45, density=True, color=SKY, alpha=0.65)
    axes[1].plot(grid, exact_pdf, color=AMBER, lw=2)
    axes[1].set_xlim(0, 0.45); axes[1].set_title("visits vs exact posterior")
    axes[1].set_xlabel("θ")
    plt.tight_layout(); plt.show()
    return acc

interact(explore_step,
         step=FloatLogSlider(value=0.075, base=10, min=-3, max=0.3, step=0.1),
         n=IntSlider(value=4000, min=500, max=20000, step=500));

# CLAIM (headless, representative settings): the acceptance-rate ordering holds
# for the exact chains the slider produces at its extremes.
a_tiny = metropolis(log_post, 0.5, 0.002, 4000, seed=0)[1]
a_gold = metropolis(log_post, 0.5, 0.075, 4000, seed=0)[1]
a_huge = metropolis(log_post, 0.5, 1.5, 4000, seed=0)[1]
assert a_tiny > a_gold > a_huge
print(f"✓ slider chains: acceptance {a_tiny:.0%} (tiny) > {a_gold:.0%} (goldilocks) > {a_huge:.0%} (huge)")
"""),
    # ====================================================================
    # 6. Multiple chains + R-hat
    # ====================================================================
    md(r"""
## 5 — Trust, but verify: multiple chains and R-hat

Everything above worked because *we could overlay the exact answer*. In real
life there is no orange curve — the posterior is unknown (that's why we're
sampling it!). So how do you know your walker explored the whole thing and not
just one corner?

The standard trick: **run several chains from deliberately different starting
points**. If they all wander to the *same* distribution and forget where they
began, that's strong evidence they found the real posterior. If different
starts lead to different answers, the chains are lying to you — at least some
of them haven't **mixed** (*mixing* = the chain moving freely through the full
posterior rather than lingering in one region).

Four walkers, four corners:
"""),
    code(r"""
STARTS = [0.05, 0.35, 0.65, 0.95]
chains_disp = np.array([metropolis(log_post, s, STEP_GOLD, N_STEPS, seed=200 + 10 * k)[0]
                        for k, s in enumerate(STARTS)])
COLS4 = [SKY, BERRY, LEAF, AMBER]

fig = go.Figure()
for k, s in enumerate(STARTS):
    fig.add_trace(go.Scatter(y=chains_disp[k, :800], mode="lines", name=f"chain {k+1} (start {s})",
                             line=dict(color=COLS4[k], width=1)))
fig.add_vrect(x0=0, x1=BURN, fillcolor=MUTED, opacity=0.15, line_width=0,
              annotation_text="burn-in / warm-up", annotation_position="top left")
fig.add_hline(y=TRUE_MEAN, line=dict(color=INK, dash="dash", width=1),
              annotation_text="exact posterior mean")
fig.update_layout(width=900, height=420, title="Four chains from four corners forget their starts",
                  xaxis_title="step", yaxis_title="θ")
fig.show()

post_means = chains_disp[:, BURN:].mean(axis=1)
print("post-burn-in means by chain:", np.round(post_means, 4))

# CLAIM: the starts span the space, yet all four post-burn-in chains agree with
# each other and with the exact mean.
assert max(STARTS) - min(STARTS) > 0.5
assert post_means.max() - post_means.min() < 0.01
assert np.all(np.abs(post_means - TRUE_MEAN) < 0.01)
print("✓ four dispersed starts → one answer: the chains mixed")
"""),
    md(r"""
### R-hat: "do independent chains agree?" as a single number

Eyeballing four traces works for one parameter; an MMM has dozens. **R-hat**
(also written $\hat R$, "Gelman–Rubin statistic") automates the eyeball test:
*it compares the spread **between** chains to the spread **within** chains*.

- If the chains have mixed, between-chain spread ≈ within-chain spread, and
  R-hat ≈ **1.0**.
- If chains disagree (stuck in different places), between-spread exceeds
  within-spread and R-hat rises above 1.

The rule of thumb you'll use forever: **R-hat below ~1.01 is healthy; above
that, do not trust the draws for that parameter.** Let's compute it by hand —
it's one formula — then confirm with `arviz` (which uses a stricter
rank-normalized, split-chain version of the same idea):
"""),
    code(r"""
def split_rhat_by_hand(chains):
    'Classic split-R-hat: split each chain in half, compare between- vs within-spread.'
    chains = np.asarray(chains)
    half = chains.shape[1] // 2
    seqs = np.concatenate([chains[:, :half], chains[:, half:2 * half]], axis=0)
    m, n = seqs.shape
    W = seqs.var(axis=1, ddof=1).mean()             # avg within-sequence variance
    B = n * seqs.mean(axis=1).var(ddof=1)           # between-sequence variance
    var_plus = (n - 1) / n * W + B / n              # pooled variance estimate
    return float(np.sqrt(var_plus / W))

good = chains_disp[:, BURN:]
rhat_hand = split_rhat_by_hand(good)
rhat_az = rhat_of(good)
print(f"R-hat by hand:  {rhat_hand:.4f}")
print(f"R-hat (arviz):  {rhat_az:.4f}")

# CLAIM: healthy, mixed chains earn R-hat below the 1.01 bar on both versions.
assert rhat_hand < 1.01, rhat_hand
assert rhat_az < 1.01, rhat_az
print("✓ good chains: R-hat < 1.01 — between-chain spread ≈ within-chain spread")
"""),
    md(r"""
### Now make R-hat scream

R-hat earns its keep when things go wrong. The classic killer is a **bimodal**
posterior — *two separate islands of probability* (in MMMs this happens when
two explanations of the data are both locally plausible). A random-walk
sampler can't cross the ocean of near-zero probability between islands: chains
that start on different islands stay there, each one *locally* looking
perfectly healthy.
"""),
    code(r"""
def make_bimodal_logp(d, sd=0.5):
    'Unnormalized log-density of two equal Gaussian islands at -d and +d.'
    def logp(x):
        return float(np.logaddexp(-0.5 * ((x - d) / sd) ** 2,
                                  -0.5 * ((x + d) / sd) ** 2))
    return logp

def bimodal_chains(d, n=3000, step=0.5, seed_base=300):
    'Four chains, two started on each island.'
    logp = make_bimodal_logp(d)
    starts = [-d - 0.3, -d + 0.3, d - 0.3, d + 0.3]
    ch = np.array([metropolis(logp, s, step, n, seed_base + 10 * k)[0]
                   for k, s in enumerate(starts)])
    return ch[:, 300:]          # drop a short burn-in

D_BAD = 3.0
bad = bimodal_chains(D_BAD)
rhat_bad_hand = split_rhat_by_hand(bad)
rhat_bad_az = rhat_of(bad)

xg = np.linspace(-D_BAD - 2, D_BAD + 2, 400)
target_bad = np.exp([make_bimodal_logp(D_BAD)(x) for x in xg])

fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38],
                    subplot_titles=(f"4 chains, R-hat = {rhat_bad_az:.2f} — RED ALERT",
                                    "the target they were meant to sample"))
for k in range(4):
    fig.add_trace(go.Scatter(y=bad[k], mode="lines",
                             line=dict(color=COLS4[k], width=0.8), name=f"chain {k+1}"), 1, 1)
fig.add_trace(go.Scatter(x=xg, y=target_bad, mode="lines",
                         line=dict(color=INK, width=2), showlegend=False), 1, 2)
fig.update_xaxes(title_text="step", row=1, col=1)
fig.update_yaxes(title_text="x", row=1, col=1)
fig.update_xaxes(title_text="x", row=1, col=2)
fig.update_layout(width=1000, height=380,
                  title="Each chain looks healthy alone — only comparing chains reveals the lie")
fig.show()

# CLAIM: every chain is stuck on its island (no chain ever visits the far mode),
# so R-hat blows far past the 1.05 alarm threshold — by hand and by arviz.
assert all(np.all(bad[k] < 0) or np.all(bad[k] > 0) for k in range(4))
assert rhat_bad_hand > 1.05 and rhat_bad_az > 1.05
assert rhat_bad_az > 1.5            # not borderline — flagrant
print(f"✓ stuck chains: R-hat by hand {rhat_bad_hand:.2f}, arviz {rhat_bad_az:.2f} — both far above 1.05")
"""),
    md(r"""
This is why every PyMC fit runs **2–4 chains by default**, and why a single
chain is never trustworthy no matter how nice it looks: **a stuck chain has no
idea it is stuck.** Only the disagreement *between* chains exposes the problem.

### 🎛️ Live exploration (run me!) — how far apart can the islands be?

Drag `d` (half the distance between islands). Small `d`: the islands merge into
one blob and the walker crosses freely — R-hat is happy. Large `d`: the chains
strand, and R-hat explodes. Somewhere in between is the cliff edge. The static
panel below fixes three representative separations.
"""),
    code(r"""
def explore_modes(d=3.0):
    'Four dispersed chains on a bimodal target; plot traces and report R-hat.'
    ch = bimodal_chains(d)
    r = rhat_of(ch)
    fig, ax = plt.subplots(figsize=(9, 3))
    for k in range(4):
        ax.plot(ch[k], lw=0.6, color=COLS4[k])
    verdict = "OK (mixed)" if r < 1.05 else "BROKEN (stuck)"
    ax.set_title(f"mode separation d={d:.1f}  →  R-hat = {r:.2f}  →  {verdict}")
    ax.set_xlabel("step"); ax.set_ylabel("x")
    plt.tight_layout(); plt.show()
    return r

interact(explore_modes, d=FloatSlider(value=3.0, min=0.5, max=6.0, step=0.5));

# CLAIM (headless, representative settings): R-hat rises with island distance.
r_near, r_far = rhat_of(bimodal_chains(1.0)), rhat_of(bimodal_chains(6.0))
assert r_near < r_far
assert r_far > 1.5
print(f"✓ R-hat grows with separation: d=1 → {r_near:.2f},  d=6 → {r_far:.2f}")
"""),
    code(r"""
# Static companion: three representative separations side by side.
fig = make_subplots(rows=1, cols=3, shared_yaxes=False,
                    subplot_titles=[f"d = {d}" for d in (1.0, 3.0, 6.0)])
rhats_d = {}
for j, d in enumerate((1.0, 3.0, 6.0), start=1):
    ch = bimodal_chains(d)
    rhats_d[d] = rhat_of(ch)
    for k in range(4):
        fig.add_trace(go.Scatter(y=ch[k], mode="lines",
                                 line=dict(color=COLS4[k], width=0.7)), 1, j)
    fig.layout.annotations[j - 1].text = f"d = {d}  —  R-hat {rhats_d[d]:.2f}"
fig.update_layout(width=1050, height=340, showlegend=False,
                  title="One blob (mixes) → two islands (strands): R-hat catches the transition")
fig.update_xaxes(title_text="step")
fig.show()

# CLAIM: with overlapping islands the chains mix (R-hat < 1.05); once the
# islands separate, R-hat blows up, and more separation only makes it worse.
assert rhats_d[1.0] < 1.05, rhats_d
assert rhats_d[3.0] > 1.05 and rhats_d[6.0] > 1.05
assert rhats_d[6.0] >= rhats_d[1.0]
print("✓ R-hat: ", {d: round(r, 2) for d, r in rhats_d.items()})
"""),
    # ====================================================================
    # 7. Effective sample size
    # ====================================================================
    md(r"""
## 6 — Effective sample size: 20,000 draws that count as 40

One more accounting subtlety. Monte Carlo's √n guarantee assumed *independent*
draws — fresh darts. A Metropolis chain's draws are **not** independent: each
position is one small step from the last, so consecutive draws are highly
**autocorrelated** (*correlated with their own recent past — knowing draw 500
tells you a lot about draw 501*). Correlated draws repeat information, so
20,000 of them can carry far less evidence than 20,000 darts.

**Effective sample size (ESS)** is the honest exchange rate: *the number of
independent draws that would estimate the posterior mean equally well*. Slowly
decaying autocorrelation ⇒ low ESS. Compare our tiny-step chain (each draw
nearly a copy of the last) against goldilocks:
"""),
    code(r"""
def acf(x, max_lag):
    'Autocorrelation of a 1-D series at lags 0..max_lag-1.'
    x = np.asarray(x, float) - np.mean(x)
    c0 = float(np.mean(x * x))
    return np.array([1.0] + [float(np.mean(x[:-k] * x[k:])) / c0
                             for k in range(1, max_lag)])

LAGS = 80
acf_tiny = acf(CH["tiny (0.002)"][0, BURN:], LAGS)
acf_gold = acf(CH["goldilocks (0.075)"][0, BURN:], LAGS)
ess_tiny = ess_of(CH["tiny (0.002)"][:, BURN:])
ess_gold = ess_of(CH["goldilocks (0.075)"][:, BURN:])
n_kept = CH["goldilocks (0.075)"][:, BURN:].size

fig = make_subplots(rows=1, cols=2, subplot_titles=(
    f"tiny step — ESS {ess_tiny:.0f} of {n_kept:,} draws",
    f"goldilocks — ESS {ess_gold:.0f} of {n_kept:,} draws"))
for j, (a, col) in enumerate([(acf_tiny, BERRY), (acf_gold, LEAF)], start=1):
    fig.add_trace(go.Bar(x=np.arange(LAGS), y=a, marker_color=col, opacity=0.8), 1, j)
    fig.update_xaxes(title_text="lag (steps apart)", row=1, col=j)
    fig.update_yaxes(range=[-0.1, 1.05], row=1, col=j)
fig.update_yaxes(title_text="autocorrelation", row=1, col=1)
fig.update_layout(width=950, height=360, showlegend=False,
                  title="How fast does the chain forget itself? Slow forgetting = few effective draws")
fig.show()

print(f"exchange rate: tiny step keeps {ess_tiny / n_kept:.1%} of its draws' value, "
      f"goldilocks keeps {ess_gold / n_kept:.1%}")

# CLAIM: the tiny-step chain stays heavily self-correlated at long lags while
# goldilocks forgets fast; ESS ranks them accordingly, and even the good chain's
# ESS is well below the raw draw count (correlation always costs something).
assert acf_tiny[40] > 0.5 and acf_gold[40] < 0.30
assert ess_tiny < ess_gold / 5
assert ess_gold < n_kept
print(f"✓ ESS: tiny {ess_tiny:.0f} ≪ goldilocks {ess_gold:.0f} < raw draws {n_kept:,}")
"""),
    md(r"""
Practical rules you'll carry into the MMM notebooks:

- ESS is reported **per parameter** (arviz: `ess_bulk` for the center,
  `ess_tail` for intervals). You want it comfortably in the **hundreds** for
  anything you report — quantiles need more than means.
- ESS, not the raw draw count, sets your Monte Carlo error: it's √ESS, not √n,
  in the denominator.
- A sampler that produces *less correlated* draws gives you more ESS per draw
  — and that's exactly the sales pitch for the engine in the next section.
"""),
    # ====================================================================
    # 8. NUTS
    # ====================================================================
    md(r"""
## 7 — NUTS: the engine PyMC actually uses

Random-walk Metropolis is a blindfolded walker: every proposal is a guess. In
40 dimensions, almost every direction is "downhill into the wasteland", so the
walker must take minute steps to keep any acceptance at all — autocorrelation
explodes and ESS collapses. The curse of dimensionality, back for revenge.

Modern samplers remove the blindfold. The posterior's log-height has a
**gradient** — *the local slope, telling you uphill from downhill* — and
computers can get it nearly free (the same automatic differentiation that
trains neural networks). **Hamiltonian Monte Carlo (HMC)** uses it with a
physics trick: turn the posterior into a landscape (high probability = low
valley), place a puck at the current position, flick it in a random direction,
and let it **glide** along the surface — momentum carrying it through long,
sweeping, curve-hugging paths instead of dithering. The end of the glide is
the proposal: typically *far* from the start yet still in a high-probability
region, so it's usually accepted. Distant, nearly-uncorrelated draws — huge
ESS per draw. **NUTS (the No-U-Turn Sampler)** is HMC with the last knob
automated: it lets the puck glide until the path starts to **U-turn** back
toward where it began, then stops — no hand-tuning of the glide length. NUTS
is the default in PyMC and in `mmm_framework`.

One picture instead of equations — same compute budget, blindfold vs puck:
"""),
    code(r"""
# A 2-D correlated Gaussian: the long diagonal ridge that kills random walks.
Sigma = np.array([[1.0, 0.9], [0.9, 1.0]])
Sinv = np.linalg.inv(Sigma)
U = lambda q: 0.5 * q @ Sinv @ q          # "potential energy" = -log density
gradU = lambda q: Sinv @ q

# Puck: one leapfrog glide (the inside of an HMC/NUTS step), 60 gradient steps.
rng = np.random.default_rng(3)
q = np.array([-1.7, -1.9])
p = rng.normal(size=2)
H0 = U(q) + 0.5 * p @ p
eps, L = 0.1, 60
traj = [q.copy()]
for _ in range(L):
    p = p - 0.5 * eps * gradU(q)
    q = q + eps * p
    p = p - 0.5 * eps * gradU(q)
    traj.append(q.copy())
traj = np.array(traj)
H1 = U(traj[-1]) + 0.5 * p @ p

# Blindfold: random-walk Metropolis, 60 proposals from the same start.
log_target_2d = lambda x: float(-U(np.asarray(x)))
rng2 = np.random.default_rng(4)
x, lx = np.array([-1.7, -1.9]), log_target_2d([-1.7, -1.9])
rw = [x.copy()]
for _ in range(L):
    prop = x + rng2.normal(0, 0.25, size=2)
    if np.log(rng2.uniform()) < log_target_2d(prop) - lx:
        x, lx = prop, log_target_2d(prop)
    rw.append(x.copy())
rw = np.array(rw)

gx = np.linspace(-3, 3, 160)
G1, G2 = np.meshgrid(gx, gx)
Z = np.exp(-0.5 * (Sinv[0, 0] * G1**2 + 2 * Sinv[0, 1] * G1 * G2 + Sinv[1, 1] * G2**2))
fig, ax = plt.subplots(figsize=(7.5, 6))
ax.contour(G1, G2, Z, levels=6, colors=MUTED, linewidths=0.8)
ax.plot(rw[:, 0], rw[:, 1], "-o", color=BERRY, ms=2.5, lw=0.9,
        label=f"blindfolded walker ({L} proposals)")
ax.plot(traj[:, 0], traj[:, 1], "-", color=LEAF, lw=2.2,
        label=f"gliding puck ({L} gradient steps)")
ax.plot(*traj[0], "s", color=INK, ms=8, label="shared start")
ax.plot(*traj[-1], "*", color=LEAF, ms=16, label="puck's proposal")
ax.set_title("Same budget, no contest: gradients turn dithering into gliding\n"
             "(NUTS stops the glide when the path begins to U-turn)")
ax.set_xlabel("parameter 1"); ax.set_ylabel("parameter 2")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout(); plt.show()

dist_puck = np.linalg.norm(traj - traj[0], axis=1).max()
dist_rw = np.linalg.norm(rw - rw[0], axis=1).max()
print(f"farthest reach from start — puck: {dist_puck:.1f}, walker: {dist_rw:.1f}")
print(f"energy drift along the glide |ΔH| = {abs(H1 - H0):.3f} (small ⇒ proposal almost always accepted)")

# CLAIM: in the same number of moves the gradient-guided glide travels much
# farther than the random walk while staying on-ridge (near-conserved energy).
assert dist_puck > 2 * dist_rw
assert abs(H1 - H0) < 0.5
assert np.exp(-U(traj[-1])) > 0.01 * Z.max()
print("✓ the puck out-explores the walker and lands in a high-probability region")
"""),
    md(r"""
Now the real thing. We hand our conversion-rate model to PyMC — same prior,
same data, so it is sampling the *same posterior* our 20-line walker did — and
let NUTS at it. Watch three numbers in the comparison: R-hat, ESS **per draw
kept** (the fair efficiency metric), and **divergences** (explained below).
"""),
    code(r"""
import pymc as pm

# pymc installs its own log handlers on import; re-quiet them (CRITICAL, not ERROR).
for _n in ("pymc", "pymc.sampling", "pymc.stats.convergence", "numpyro", "jax",
           "arviz", "pytensor"):
    _lg = logging.getLogger(_n); _lg.setLevel(logging.CRITICAL); _lg.propagate = False

t0 = time.perf_counter()
with pm.Model() as conversion_model:
    theta = pm.Beta("theta", alpha=A_PRIOR, beta=B_PRIOR)
    pm.Binomial("signups", n=N_VIS, p=theta, observed=CONV)
    with quiet():
        idata = pm.sample(draws=800, tune=800, chains=2, cores=1, random_seed=0,
                          progressbar=False, compute_convergence_checks=False)
T_NUTS = time.perf_counter() - t0

nuts_draws = idata.posterior["theta"].values            # (2 chains, 800 draws)
ess_nuts = float(az.ess(idata, var_names=["theta"])["theta"])
rhat_nuts = float(az.rhat(idata, var_names=["theta"])["theta"])
n_div = int(idata.sample_stats["diverging"].sum())
n_nuts_kept = nuts_draws.size

compare = pd.DataFrame({
    "our Metropolis (goldilocks)": {
        "draws kept": f"{n_kept:,}",
        "ESS (bulk)": f"{ess_gold:.0f}",
        "ESS per draw kept": f"{ess_gold / n_kept:.2f}",
        "posterior mean": f"{CH['goldilocks (0.075)'][:, BURN:].mean():.4f}",
        "R-hat": f"{rhat_of(CH['goldilocks (0.075)'][:, BURN:]):.3f}",
        "divergences": "n/a (not gradient-based)",
        "wall time": f"{T_METRO:.1f}s (pure-python loop)",
    },
    "PyMC NUTS": {
        "draws kept": f"{n_nuts_kept:,}",
        "ESS (bulk)": f"{ess_nuts:.0f}",
        "ESS per draw kept": f"{ess_nuts / n_nuts_kept:.2f}",
        "posterior mean": f"{nuts_draws.mean():.4f}",
        "R-hat": f"{rhat_nuts:.3f}",
        "divergences": str(n_div),
        "wall time": f"{T_NUTS:.1f}s (incl. model compile)",
    },
})
compare.index.name = f"exact posterior mean = {TRUE_MEAN:.4f}"
display(compare)

# CLAIM: both engines agree with the exact answer; NUTS converges cleanly with
# zero divergences and beats a WELL-TUNED Metropolis on information per draw —
# even here, on the easiest possible target (1-D, where random walks are at
# their best and the gap is smallest).
assert abs(nuts_draws.mean() - TRUE_MEAN) < 0.01
assert rhat_nuts < 1.01 and n_div == 0
assert ess_nuts / n_nuts_kept > ess_gold / n_kept
print(f"✓ NUTS: {ess_nuts / n_nuts_kept:.2f} effective draws per draw kept vs "
      f"Metropolis {ess_gold / n_kept:.2f} — gradients buy independence")
"""),
    md(r"""
Read the table honestly: on this *one-parameter* toy, our hand-rolled walker is
a perfectly respectable engine, and its pure-python loop is even quick. A 1-D
target is random walk's best case — and NUTS still wins on **information per
draw**, the column that matters. The gap becomes a chasm as dimensions grow:
random-walk efficiency *collapses* with dimension (tiny steps, sky-high
autocorrelation) while gradient-guided gliding barely notices. Your
40-parameter MMM in workshop_03 is only feasible because of this.

### Divergences: the engine warning light

The table has a column Metropolis can't have: **divergences** — *moments when
the gliding puck's simulated physics broke down: the trajectory shot off the
landscape and the step had to be abandoned*. They happen in regions of extreme
curvature, where the glide step is too coarse to follow the surface — think a
canyon whose walls bend sharper than the puck's stride. Three things to burn in:

1. **A divergence is a location, not a nuisance.** It marks a *specific region*
   the sampler tried and failed to enter — the draws you got may be blind to
   exactly that region, so the posterior you see can be **biased**, not just noisy.
2. **Don't ignore them; don't just silence them.** A handful in tens of
   thousands of draws may be tolerable; dozens are a model problem. The real
   fixes change the *geometry*: gentler/tighter priors that don't carve cliffs,
   or **reparameterization** — *rewriting the model in equivalent variables
   whose landscape is smoother* (the famous "non-centered" trick for
   hierarchical models).
3. **You mostly inherit the fixes.** `mmm_framework`'s default priors and its
   hierarchical structures already encode these reparameterizations — one of
   several reasons workshop_03 starts from the framework's defaults instead of
   hand-rolled PyMC.
"""),
    # ====================================================================
    # 9. Trace-plot reading clinic
    # ====================================================================
    md(r"""
## 8 — The trace-plot reading clinic

A **trace plot** — *parameter value vs step number, one line per chain* — is
the sampler's EKG, and reading one is a 10-second skill that catches most
problems before any statistic does. Four patterns cover ~90% of what you'll
ever see. All four below are real chains generated in this notebook:
"""),
    code(r"""
WIN = 1500
healthy = chains_disp[:, BURN:BURN + WIN]                       # goldilocks, mixed
trending = np.array([metropolis(log_post, s, 0.0015, WIN, seed=400 + 10 * k)[0]
                     for k, s in enumerate(STARTS)])            # tiny step, far starts
stuck = bad[:, :WIN]                                            # bimodal islands
sticky = np.array([metropolis(log_post, TRUE_MEAN, 0.008, WIN, seed=500 + 10 * k)[0]
                   for k in range(4)])                          # arrived, but slow

panels = [
    ("HEALTHY: 'fuzzy caterpillar'", healthy,
     "chains overlap in one stationary band.\nDIAGNOSIS: converged & mixing. REMEDY: none — ship it."),
    ("TRENDING: still drifting", trending,
     "each chain still travelling from its start.\nDIAGNOSIS: not converged. REMEDY: longer warm-up /\nbetter inits / bigger steps."),
    ("STUCK: parallel flat bands", stuck,
     "chains level but DISAGREE on the level.\nDIAGNOSIS: multimodal or pathological posterior.\nREMEDY: reparameterize or rethink the model."),
    ("STICKY: slow random waves", sticky,
     "one band, but long lazy swings (high autocorrelation).\nDIAGNOSIS: converged, tiny ESS. REMEDY: more draws /\nbetter sampler (NUTS)."),
]
fig, axes = plt.subplots(2, 2, figsize=(11.5, 7))
for ax, (title, ch, note) in zip(axes.ravel(), panels):
    for k in range(4):
        ax.plot(ch[k], lw=0.55, color=COLS4[k], alpha=0.9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=7.2,
            va="bottom", bbox=dict(boxstyle="round", fc="white", ec=MUTED, alpha=0.85))
    ax.set_xlabel("step", fontsize=8)
fig.suptitle("Trace-plot clinic: four patterns, four diagnoses", fontsize=12)
plt.tight_layout(); plt.show()

# CLAIM: the diagnostics agree with the eyeball: healthy mixes (low R-hat),
# trending and stuck fail R-hat badly, sticky passes R-hat but pays in ESS.
r_healthy, r_trend, r_stuck = rhat_of(healthy), rhat_of(trending), rhat_of(stuck)
ess_healthy, ess_sticky = ess_of(healthy), ess_of(sticky)
assert r_healthy < 1.05 and r_healthy < r_trend and r_healthy < r_stuck
assert r_trend > 1.1 and r_stuck > 1.5
assert ess_sticky < ess_healthy / 3
print(f"✓ R-hat — healthy {r_healthy:.2f}, trending {r_trend:.2f}, stuck {r_stuck:.2f}; "
      f"ESS — healthy {ess_healthy:.0f} vs sticky {ess_sticky:.0f}")
"""),
    # ====================================================================
    # 10. The crucial caveat
    # ====================================================================
    md(r"""
## 9 — The crucial caveat: green lights validate the *computation*, not the *model*

Everything in this notebook — R-hat, ESS, divergences, trace plots — answers
exactly one question: **did the sampler faithfully explore the posterior of the
model you wrote down?** None of it asks whether the model you wrote down is
*right*. A model that confuses correlation with causation — say, an MMM where a
hidden demand surge drives both ad spend and sales — will converge **beautifully**:
R-hat 1.00, zero divergences, gorgeous fuzzy caterpillars… to a confidently
*wrong* answer. The chains agree with each other; they're just agreeing on a
biased posterior.

So treat today's diagnostics as **necessary, never sufficient**. They are the
pre-flight checklist, not the navigation. The full treatment of this trap —
two worlds, identical green dashboards, one badly wrong ROAS table — is
[`../stress/stress_00_the_rosy_picture.ipynb`](../stress/stress_00_the_rosy_picture.ipynb), worth
reading once you finish this series. And it's why
[workshop 03](workshop_03_first_mmm.ipynb) fits our first MMM on a **synthetic
world whose true answer we know**: the only way to learn what the posterior
*should* look like is to practice where truth is checkable.
"""),
    # ====================================================================
    # 11. Glossary + what's next
    # ====================================================================
    md(r"""
## 10 — Glossary and what's next

| term | plain English |
|---|---|
| **Monte Carlo** | estimating a quantity from random samples instead of exact calculation; error shrinks like 1/√n |
| **curse of dimensionality** | grid-style methods explode exponentially with the number of parameters — why sampling exists |
| **MCMC** | Markov chain Monte Carlo: a walker whose movement rules make time-spent proportional to posterior probability |
| **proposal** | the rule for suggesting the walker's next position (e.g. a Gaussian step) |
| **acceptance rate** | fraction of proposals taken; near-100% usually means steps are too timid, not that things are great |
| **chain** | one walker's recorded visit log = one stream of posterior draws |
| **burn-in / warm-up** | early draws discarded because the walker hadn't yet forgotten its arbitrary start (PyMC: `tune`) |
| **mixing** | the chain roaming freely across the whole posterior rather than lingering |
| **R-hat** | "do independently-started chains agree?" — between-chain vs within-chain spread; want < 1.01 |
| **autocorrelation** | a draw's correlation with its recent predecessors; slow decay = repeated information |
| **ESS** | effective sample size: how many *independent* draws your correlated draws are worth; want hundreds+ |
| **gradient** | the local slope of log-probability — the information that lets modern samplers glide instead of guess |
| **NUTS** | No-U-Turn Sampler: gradient-guided HMC that auto-stops each glide when it starts curling back; PyMC's default |
| **divergence** | the engine light: a spot where NUTS's simulated glide broke down — possible bias near that region; fix geometry, don't mute |
| **trace plot** | parameter vs step per chain — the sampler's EKG (want fuzzy caterpillars) |

**The arc so far:** workshop_00 gave you the posterior; workshop_01 taught you
to choose and *check* priors before seeing data; today you learned how the
machine actually delivers posteriors at scale, and how to audit its work.

**Next — [workshop 03: your first MMM](workshop_03_first_mmm.ipynb).** We
finally meet the model this series is for: media channels, adstock, saturation,
a real `BayesianMMM` fit on a synthetic market where the true ROAS is known —
and your new diagnostic reflexes get their first real workout.
"""),
]


def main():
    nb = new_notebook(cells=CELLS)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "title": "Workshop 02 — Sampling: How the Machine Works",
    })
    path = "workshop/workshop_02_sampling.ipynb"
    with open(path, "w") as fh:
        nbformat.write(nb, fh)
    print(f"wrote {path} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
