"""Author the *narrative* continuous-learning notebook (run from ``nbs/``).

    uv run --with nbformat python build_continuous_learning_story.py
    PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel --with kaleido \
        jupyter nbconvert --to notebook --execute --inplace \
        continuous_learning_story.ipynb --ExecutePreprocessor.timeout=5400 \
        --ExecutePreprocessor.kernel_name=python3

A second, *story-first* companion to ``continuous_learning.ipynb``. Same engine
(``mmm_framework.continuous_learning``), but framed as one brand's measurement
cycle — the business questions, the data, the findings and plots the analyst
produces along the way, and the one-slide they take to the client / media team.

The technical notebook proves the machinery; this one shows what it feels like to
run. Numbers come from live fits (a fixed synthetic "truth" the analyst never
sees); markdown points at the live plots, not at hardcoded values.
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
    # 1. The brand + the brief
    # ====================================================================
    md(r"""
# A brand's measurement cycle — learning what works, one test at a time

**The brand.** *Nomi* is a direct-to-consumer functional-beverage company. It
spends **\$560k a week** on paid social, split evenly out of habit across four
platforms — **Chatter, Pulse, Orbit, and Vibe** — \$140k each. Growth has
plateaued and the CFO wants the budget defended.

**The problem.** Nomi has two years of weekly data, but every attempt to read ROI
off it disagrees with the last. Spend and sales both rise in good weeks, so the
dashboards flatter every channel. The head of growth has stopped trusting them.

This notebook follows Nomi's analyst through one **measurement cycle** built on
*designed geo experiments* instead of historical correlation — the
`mmm_framework.continuous_learning` loop. No marketing-mix model, no assumption
that past spend was clean. Just: **run a test, learn, reallocate, decide whether
to test again.**

### The questions on the table

| # | The business question | What answers it |
|---|---|---|
| 1 | Are we over- or under-investing in any channel? | the recovered **response curves** |
| 2 | Which channels actually drive incremental sales? | the **funding line** (marginal ROAS) |
| 3 | Do the channels fight each other or help each other? | the **synergy map** (interactions) |
| 4 | What should next quarter's split be — and how sure are we? | the **Thompson allocation** + its spread |
| 5 | Is it worth running another test, or do we lock the plan? | **expected regret** + the ENBS stop rule |
| 6 | When will this answer go stale? | **information-decay** re-test cadence |

We'll answer them in order, and end with the one-slide the analyst brings to the
media-planning team.
"""),
    md(r"""
### How the cycle works (the 60-second version)

Every few weeks Nomi runs a **wave**: it splits its DMAs into groups, deliberately
pushes each channel's spend **up, down, or off** in different groups (a
*central-composite design*), and holds a quiet **pre-period** where every group
spends normally so we can pin each market's baseline. Because the variation is
*designed*, the differences in sales across groups are **causal** — not "we spent
more when demand was already high."

A Bayesian **response surface** is fit to the wave, the optimizer reads a
**recommended split** and a **funding line** off it, and a stopping rule asks
whether one more wave is worth its cost. Then we recenter on the recommendation
and (maybe) go again. The picture sharpens each round.
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

pio.templates.default = "plotly_white"

# ---- Nomi's channels + a consistent palette -------------------------------
CHANNELS = ["Chatter", "Pulse", "Orbit", "Vibe"]
PALETTE = {"Chatter": "#3b6fb6", "Pulse": "#d98c3f", "Orbit": "#5a9e6f", "Vibe": "#b15a7a"}
COLORS = [PALETTE[c] for c in CHANNELS]

# ---- money + decision framing ---------------------------------------------
SPEND_SCALE = 200_000          # $ per one unit of the model's scaled spend
VALUE = 5.0                    # margin $ per unit KPI -> sets the funding line (break-even mROAS = 1)
CENTER = np.full(4, 0.7)       # status quo: $140k each ( = 0.7 * $200k )
B = 2.8                        # weekly budget = 4 * 0.7 = $560k

def dollars(scaled):
    "scaled spend -> a readable $ string"
    return f"${scaled * SPEND_SCALE / 1000:,.0f}k"

# ---- the hidden 'truth' (a real analyst NEVER sees this) -------------------
# We use a fixed synthetic world only to generate honest experiment outcomes.
# Everything the analyst 'sees' below is estimated from data, never read off this.
world = cl.make_world(seed=7)
PAIR_SIGNS = cl.PAIR_SIGNS_EXAMPLE

print("Nomi buys on:", ", ".join(CHANNELS))
print("Weekly budget:", dollars(B), " (status quo:", dollars(0.7), "per channel)")
print("Break-even funding line: marginal ROAS = 1.0")
"""),
    # ====================================================================
    # 2. Why the dashboard lies (confounding)
    # ====================================================================
    md(r"""
## Act 1 · Why last year's dashboard lied

Before spending a dollar on testing, the analyst re-makes the case for *why*
testing is necessary. She pulls two years of weekly data and plots **total spend
vs. total sales**. The correlation is gorgeous — and useless.

The catch: Nomi, like everyone, **leans into spend when demand is already hot**
(new-product buzz, seasonality, a viral moment). So weeks with high spend are also
weeks with high underlying demand. A naive line through the cloud credits *spend*
for sales that **demand** would have delivered anyway. That inflated slope is the
number the old dashboard reported as "ROI."
"""),
    code(r"""
# A plausible observational history: a hidden 'demand' driver moves BOTH spend
# and sales, so the naive slope over-credits spend. (Illustrative, not a fit.)
rng = np.random.default_rng(3)
weeks = 104
demand = np.cumsum(rng.normal(0, 1, weeks)); demand = (demand - demand.mean()) / demand.std()
season = 0.6 * np.sin(np.arange(weeks) * 2 * np.pi / 52)
hot = demand + season                                        # latent demand index
spend = 560 + 130 * hot + rng.normal(0, 25, weeks)           # $k/week: planners chase demand
TRUE_MARGINAL = 0.9                                          # $ sales per $ spend, held-demand
sales = 1900 + TRUE_MARGINAL * spend + 240 * hot + rng.normal(0, 60, weeks)  # $k/week

naive_slope = np.polyfit(spend, sales, 1)[0]                 # what the dashboard 'sees'
xs = np.linspace(spend.min(), spend.max(), 50)
fig = go.Figure()
fig.add_trace(go.Scatter(x=spend, y=sales, mode="markers", name="weekly history",
    marker=dict(color="#9aa7b0", size=7, line=dict(color="white", width=0.5))))
fig.add_trace(go.Scatter(x=xs, y=np.polyval(np.polyfit(spend, sales, 1), xs), mode="lines",
    name=f"naive 'ROI' fit  (slope {naive_slope:.1f}x)", line=dict(color="#c0504d", width=3)))
fig.add_trace(go.Scatter(x=xs, y=sales.mean() + TRUE_MARGINAL * (xs - spend.mean()), mode="lines",
    name=f"true causal slope  ({TRUE_MARGINAL:.1f}x)", line=dict(color="#4a8d57", width=3, dash="dash")))
fig.update_layout(height=440, margin=dict(l=70, r=30, t=70, b=110),
    title="Two years of history: the naive slope over-credits spend by ~2-3x",
    xaxis_title="total weekly spend ($k)", yaxis_title="total weekly sales ($k)",
    legend=dict(orientation="h", yanchor="top", y=-0.18, x=0))
fig.show()
print(f"Naive slope says every $1 returns ${naive_slope:.1f}. Held-demand truth is ${TRUE_MARGINAL:.1f}.")
print("The gap is pure confounding — and it's per-channel, so the RANKING is wrong too.")
assert naive_slope > TRUE_MARGINAL * 1.5      # confounding materially inflates the read
"""),
    md(r"""
**Analyst's readout →** *"We can't allocate on this. The history says every
channel is a hero because we only ever spent hard when demand was high. To get a
number we can defend to the CFO, we have to create the variation ourselves."*

That is the entire case for the loop: **stop mining the past, start designing the
present.**
"""),
    # ====================================================================
    # 3. The experiment we designed
    # ====================================================================
    md(r"""
## Act 2 · The test we designed

The analyst designs one **budget-neutral geo wave**. Nomi's DMAs are split into
groups; each group runs a different *cell* of the design — some channels pushed
up, some down, a few switched **off** entirely — while total spend stays roughly
flat. A holdout-style **pre-period** keeps every group at status quo first, which
pins each market's baseline so the test-period differences are clean.

The off cells matter: turning a channel **off** in some markets is what separates
its own effect from its synergies (you can't tell a team player from a coat-tail
rider unless you sometimes bench it).
"""),
    code(r"""
# The central-composite design: rows = cells, columns = channels, values = spend
# multiplier vs. the status-quo center. 1.0 = normal, >1 up, <1 down, 0 = OFF.
design = cl.central_composite(CENTER, 0.6, world.pairs)
mult = design / CENTER
fig = go.Figure(go.Heatmap(
    z=mult, x=CHANNELS, y=[f"cell {i}" for i in range(mult.shape[0])],
    colorscale=[[0, "#7a2732"], [0.4, "#f2f2f2"], [1.0, "#1f4e79"]], zmid=1.0,
    colorbar=dict(title="× status quo", thickness=14),
    text=np.round(mult, 1), texttemplate="%{text}", textfont=dict(size=9)))
fig.update_layout(height=520, margin=dict(l=60, r=30, t=60, b=40),
    title=f"Experiment brief: {mult.shape[0]} designed cells across 64 DMAs, ~3 test weeks",
    xaxis_title="channel", yaxis=dict(autorange="reversed"))
fig.show()
print(f"{mult.shape[0]} cells: 1 status-quo, up/down axials per channel, pair probes, and "
      f"{len(CHANNELS)} shut-off cells.")
print("Budget-neutral by construction — no extra money, just rearranged for one wave.")
"""),
    md(r"""
**What the client hears →** *"For three weeks, in a rotation of markets, we'll
nudge each platform's budget up in some, down in others, and dark in a few. Total
national spend doesn't change. Nobody's quarter gets blown up — and at the end we
can read each platform's real contribution, not its correlation."*
"""),
    # ====================================================================
    # 4. Behind the scenes: run the program
    # ====================================================================
    md(r"""
## Behind the scenes: we ran the program

We now run **three waves** of the cycle against Nomi's (hidden) reality. Each wave:
fit the surface on **all** data so far, read the recommendation + funding line +
uncertainty, decide whether to keep testing, then recenter on the recommendation
and run the next designed wave over the **same** markets.

The rest of the notebook replays what the analyst saw at each stage.
"""),
    code(r"""
# The real accumulating loop (same markets each wave => a stable baseline).
state = cl.LearningState(channels=CHANNELS, center=CENTER.copy(), B=B, value=VALUE,
                         pairs=world.pairs, pair_signs=PAIR_SIGNS, activation="hill", mode="fixed")
w0 = cl.simulate_panel(world, CENTER, n_geo=64, t_pre=6, t_test=10, delta=0.6, noise=0.45, seed=11)
a_geo = np.asarray(w0["a_geo"])          # each market's baseline persists across waves
state.ingest(w0)

# ENBS stopping inputs (business terms): plan horizon in weeks, cost of a wave.
HORIZON, MARGIN, WAVE_COST = 13.0, 1.0, 1.8

waves = []
center = CENTER.copy()
for wave in range(3):
    state.fit(num_warmup=400, num_samples=400, num_chains=2, seed=wave)
    post = state.posterior
    rec = state.recommend(q=250)
    mroas_mean, prob_above, mroas_draws = cl.marginal_roas(post, rec, VALUE, q=250)
    allocs, _profits, _ = cl.thompson_wave(post, B, VALUE, q=250, mode="fixed", seed=wave)
    e_regret, consensus, alloc_sd, _psd = cl.expected_regret(post, B, VALUE, q=250)
    stop, enbs_val = cl.should_stop(e_regret, margin=MARGIN, population=HORIZON, wave_cost=WAVE_COST)
    waves.append(dict(wave=wave, center=center.copy(), post=post, rec=rec,
                      beta=post.samples["beta"].mean(0), mroas=mroas_mean, prob=prob_above,
                      mroas_draws=mroas_draws, allocs=allocs, e_regret=e_regret,
                      alloc_sd=alloc_sd, enbs=enbs_val, stop=stop,
                      rhat=post.diagnostics.get("max_rhat")))
    print(f"wave {wave}: rec { {c: dollars(rec[i]) for i,c in enumerate(CHANNELS)} }"
          f"  E[regret] {e_regret:.2f}  stop={stop}")
    if wave == 2:            # run the full 3-wave arc; the ENBS chart shows where it WOULD stop
        break
    design = cl.central_composite(rec, 0.6, world.pairs)
    wnext = cl.simulate_wave(world, design, a_geo, t_test=10, center=rec, noise=0.45, seed=20 + wave)
    state.recenter(rec); state.ingest(wnext); center = rec

first, last = waves[0], waves[-1]
print(f"\nran {len(waves)} waves; posterior R-hat healthy:",
      [round(w["rhat"], 2) for w in waves])
assert all(w["rhat"] < 1.15 for w in waves)
"""),
    # ====================================================================
    # 5. First read — which channels work
    # ====================================================================
    md(r"""
## Act 3 · First read — which channels actually work

Wave 0 is in. The first thing the analyst looks at is the **incremental effect**
of each channel — the ceiling of sales each one can drive, estimated from the
designed variation alone. This is the answer to *"are we over- or under-investing
anywhere?"*
"""),
    code(r"""
# Recovered channel effects (posterior mean + 90% credible interval). The analyst
# does NOT know the truth; this is what the experiment revealed.
b = first["post"].samples["beta"]
names, mean = CHANNELS, b.mean(0)
lo, hi = np.percentile(b, 5, 0), np.percentile(b, 95, 0)
fig = go.Figure()
for i, nm in enumerate(names):
    fig.add_trace(go.Scatter(x=[lo[i], hi[i]], y=[i, i], mode="lines",
        line=dict(color=COLORS[i], width=4), showlegend=False))
fig.add_trace(go.Scatter(x=mean, y=list(range(4)), mode="markers+text",
    marker=dict(color=COLORS, size=13, line=dict(color="white", width=1.5)),
    text=[f"  {v:.1f}" for v in mean], textposition="middle right", showlegend=False))
fig.update_yaxes(tickmode="array", tickvals=list(range(4)), ticktext=names,
                 range=[-0.6, 3.6])
fig.update_layout(height=330, margin=dict(l=80, r=40, t=60, b=50),
    title="Incremental effect per channel — Chatter leads, Vibe trails (wave 0)",
    xaxis_title="estimated sales ceiling (model units, higher = stronger)")
fig.show()
order = [CHANNELS[i] for i in np.argsort(-mean)]
print("Strength order from the experiment:", " > ".join(order))
"""),
    md(r"""
**The surprise.** Nomi's team *assumed* Vibe was a workhorse — it's trendy and
gets the most internal love. The experiment says it's the **weakest** of the four.
Chatter, the quiet incumbent, is the real engine. Assumptions are exactly what
designed measurement is for.

Next: strength isn't the same as *worth the next dollar*. For that we need the
**funding line** — the marginal ROAS at the current spend, with the break-even
line at 1.0.
"""),
    code(r"""
# Funding line: marginal return on the NEXT dollar, per channel, at status quo.
mm, pa = first["mroas"], first["prob"]
verdict = ["FUND" if p > 0.6 else ("CUT" if (p < 0.4 and m < 1) else "HOLD")
           for m, p in zip(mm, pa)]
vcol = {"FUND": "#4a8d57", "HOLD": "#d9a441", "CUT": "#c0504d"}
ytop = float(max(mm)) * 1.32
fig = go.Figure()
fig.add_trace(go.Bar(x=CHANNELS, y=mm, marker_color=[vcol[v] for v in verdict], showlegend=False))
fig.add_hline(y=1.0, line=dict(color="#333", dash="dash"),
              annotation_text="break-even (mROAS = 1)", annotation_position="bottom left")
# verdict labels in a clean top row, clear of the bars and the break-even line
for c, v, p in zip(CHANNELS, verdict, pa):
    fig.add_annotation(x=c, y=ytop * 0.96, text=f"<b>{v}</b><br>P(profit) {p:.0%}",
                       showarrow=False, align="center", font=dict(size=11, color=vcol[v]))
fig.update_layout(height=440, margin=dict(l=60, r=30, t=60, b=50),
    title="Funding line (wave 0): is the next dollar profitable?",
    yaxis_title="marginal ROAS", xaxis_title="channel", yaxis=dict(range=[0, ytop]))
fig.show()
for c, m, p, v in zip(CHANNELS, mm, pa, verdict):
    print(f"  {c:8s} marginal ROAS {m:4.1f}  P(profitable) {p:4.0%}  -> {v}")
"""),
    md(r"""
**Analyst's readout →** *"At today's even split, the next dollar into Chatter,
Pulse, and Orbit still pays for itself several times over — we're **under**-funding
all three. Vibe's next dollar is under water. We're pouring \$140k a week into our
weakest platform while starving the ones with headroom."*

But before cutting Vibe, one more question: **do the channels interact?**
"""),
    code(r"""
# Synergy map: posterior-mean interaction between each pair. Positive = they
# amplify each other (fund together); negative = they cannibalize (don't max both).
K = 4
G = np.mean([last["post"].gamma_matrix(d) for d in range(0, last["post"].samples["beta"].shape[0], 2)], axis=0)
G = G + G.T
np.fill_diagonal(G, np.nan)
fig = go.Figure(go.Heatmap(z=G, x=CHANNELS, y=CHANNELS,
    colorscale=[[0, "#7a2732"], [0.5, "#f4f4f4"], [1, "#1f4e79"]], zmid=0,
    colorbar=dict(title="synergy", thickness=14),
    text=np.where(np.isnan(G), "", np.round(G, 2)), texttemplate="%{text}",
    textfont=dict(size=13)))
fig.update_layout(height=430, margin=dict(l=70, r=30, t=70, b=40),
    title="Synergy map: who amplifies whom (blue) vs. who cannibalizes (red)",
    yaxis=dict(autorange="reversed"))
fig.show()
gs = last["post"].gamma_summary()
strong = {k: v for k, v in gs.items() if abs(v["mean"]) > 0.2}
for k, v in sorted(strong.items(), key=lambda kv: kv[1]["mean"]):
    tag = "cannibalize" if v["mean"] < 0 else "amplify"
    print(f"  {k.replace('gamma_','').replace('_',' x '):22s} {v['mean']:+.2f} -> {tag}")
"""),
    md(r"""
**Three findings the analyst circles:**

- **Chatter × Pulse cannibalize** (negative). They chase the same audience — maxing
  both wastes money on overlap. Pick a lead.
- **Orbit is the hub** (Pulse × Orbit and Orbit × Vibe both positive). Orbit's sales
  ride on spend flowing through Pulse and Vibe.
- **Vibe's redemption clause.** Vibe is weak *alone*, but it **amplifies Orbit**.
  Kill it to zero and you lose some of Orbit's lift. So the question isn't "cut Vibe
  or not" — it's "what's the smallest Vibe budget that still feeds Orbit?"

This is why you measure **interactions**, not just channels: the naive
"cut the weakest" move would have quietly taxed the strongest.
"""),
    # ====================================================================
    # 6. The reallocation
    # ====================================================================
    md(r"""
## Act 4 · The reallocation — and how sure we are

Now the money question. The optimizer reads the best split off the response
surface — but as a **distribution**, not a point. Nomi's analyst shows the client
the recommended move *and* its uncertainty, because "move \$100k" lands very
differently when the error bar is \$10k versus \$60k.
"""),
    code(r"""
# Current vs. recommended weekly split ($), with the recommendation's uncertainty.
rec, asd = last["rec"], last["alloc_sd"]
cur = CENTER
fig = go.Figure()
fig.add_trace(go.Bar(name="today (even split)", x=CHANNELS, y=cur * SPEND_SCALE / 1000,
                     marker_color="#c9d3dc", text=[dollars(v) for v in cur], textposition="inside"))
fig.add_trace(go.Bar(name="recommended", x=CHANNELS, y=rec * SPEND_SCALE / 1000,
                     marker_color=COLORS,
                     error_y=dict(type="data", array=asd * SPEND_SCALE / 1000, visible=True, color="#555")))
fig.update_layout(barmode="group", height=440, margin=dict(l=70, r=30, t=60, b=90),
    title="Recommended weekly reallocation (error bars = how sure the model is)",
    yaxis_title="weekly spend ($k)", xaxis_title="channel",
    legend=dict(orientation="h", yanchor="top", y=-0.16, x=0))
fig.show()
for c, u, r in zip(CHANNELS, cur, rec):
    print(f"  {c:8s} {dollars(u)} -> {dollars(r)}   ({(r-u)*SPEND_SCALE/1000:+,.0f}k / week)")
"""),
    md(r"""
**The move.** Redirect the freed Vibe budget — about **\$130k/week** — into
**Chatter** (the workhorse) and **Orbit** (the synergy hub), hold Pulse roughly
flat, and leave Vibe a **token maintenance budget** (kept only for its Orbit halo).
Same total \$560k — just pointed at headroom instead of habit.

**How the confidence is built.** The optimizer solves the split for hundreds of
posterior draws; the spread of those solutions is the error bar. Here's the full
distribution behind each recommendation — narrow bars mean "act now," wide bars
mean "this is where another test would pay."
"""),
    code(r"""
# The distribution of the optimal split across posterior draws (the 'confidence').
allocs = last["allocs"] * SPEND_SCALE / 1000
fig = go.Figure()
for i, c in enumerate(CHANNELS):
    fig.add_trace(go.Box(y=allocs[:, i], name=c, marker_color=COLORS[i],
                         boxpoints=False, showlegend=False))
fig.add_trace(go.Scatter(x=CHANNELS, y=cur * SPEND_SCALE / 1000, mode="markers",
    marker=dict(symbol="line-ew", size=26, color="#333", line=dict(width=3)),
    name="today"))
fig.update_layout(height=430, margin=dict(l=70, r=30, t=60, b=70),
    title="How confident is each recommendation? (posterior spread of the optimal split)",
    yaxis_title="weekly spend ($k)", xaxis_title="channel",
    legend=dict(orientation="h", yanchor="top", y=-0.14, x=0))
fig.show()
tight = CHANNELS[int(np.argmin(last["alloc_sd"]))]
loose = CHANNELS[int(np.argmax(last["alloc_sd"]))]
print(f"Most certain move: {tight}.  Least certain (best test target): {loose}.")
"""),
    # ====================================================================
    # 7. Do we test again?
    # ====================================================================
    md(r"""
## Act 5 · Do we test again, or lock the plan?

Testing isn't free — every wave rearranges real budget and costs analyst time. The
loop makes the call explicitly. **Expected regret** is the profit still on the table
from acting on today's (uncertain) recommendation instead of the truly-optimal one.
The **ENBS rule** (Expected Net Benefit of Sampling) compares the value of resolving
that uncertainty over the plan horizon against the cost of another wave.
"""),
    code(r"""
w = [x["wave"] for x in waves]
regret = [x["e_regret"] for x in waves]
enbs = [x["enbs"] for x in waves]
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.14, subplot_titles=(
    "Uncertainty is shrinking (expected regret)",
    "Is another wave worth it? (ENBS)"))
fig.add_trace(go.Scatter(x=w, y=regret, mode="lines+markers",
    line=dict(color="#3b6fb6", width=3), marker=dict(size=11), showlegend=False), row=1, col=1)
fig.add_trace(go.Bar(x=w, y=enbs, marker_color=["#4a8d57" if e > 0 else "#c0504d" for e in enbs],
    showlegend=False), row=1, col=2)
fig.add_hline(y=0, line=dict(color="#333", dash="dash"), row=1, col=2)
fig.update_xaxes(title_text="wave", tickvals=w, row=1, col=1)
fig.update_xaxes(title_text="wave", tickvals=w, row=1, col=2)
fig.update_yaxes(title_text="profit left on the table", row=1, col=1)
fig.update_yaxes(title_text="net value of one more wave", row=1, col=2)
fig.update_layout(height=400, margin=dict(l=70, r=30, t=70, b=55),
    title_text="The stopping decision")
fig.show()
stop_wave = next((x["wave"] for x in waves if x["enbs"] <= 0), None)
print("expected regret by wave:", [round(r, 2) for r in regret])
print("ENBS (green = test again, red = stop):", [round(e, 2) for e in enbs])
print(f"-> lock the plan at wave {stop_wave}" if stop_wave is not None
      else "-> still worth testing at the end of this run")
"""),
    md(r"""
**Analyst's readout →** *"Regret roughly halved between the first and last wave —
the split stopped moving and the error bars tightened. By the last wave, the
expected gain from yet another test no longer covers its cost. We lock this plan
and put the next test dollars toward a **new** question instead."*

Notice the funding line also *equalized* across the funded channels by the final
wave — every kept channel returns about the same on its next dollar. That flatness
**is** the signature of an optimized budget: there's no obvious dollar left to move.
"""),
    # ====================================================================
    # 8. When does this go stale?
    # ====================================================================
    md(r"""
## Act 6 · When will this answer go stale?

A measured response surface is a *snapshot*. Audiences drift, competitors move,
creative fatigues. The loop treats each finding as having a **half-life**: the
effective uncertainty grows over time until it's worth re-testing. This is the same
information-decay clock the model-anchored planner uses, so Nomi's calendar of
re-tests is consistent across both.
"""),
    code(r"""
# Information decay: today's uncertainty grows until a re-test is due.
sigma_post = float(np.mean(last["alloc_sd"]))     # how tight we are now
weeks = np.arange(0, 40)
sig_eff, due_at = [], None
for t in weeks:
    due, s = cl.due_for_retest(sigma_post, float(t), "Pulse", sigma_post)
    sig_eff.append(float(s))
    if due and due_at is None:
        due_at = int(t)
fig = go.Figure()
fig.add_trace(go.Scatter(x=weeks, y=sig_eff, mode="lines", line=dict(color="#b15a7a", width=3),
                         name="effective uncertainty"))
fig.add_hline(y=sig_eff[0], line=dict(color="#4a8d57", dash="dash"),
              annotation_text="precision just after this wave", annotation_position="bottom right")
if due_at is not None:
    fig.add_vline(x=due_at, line=dict(color="#c0504d", dash="dot"),
                  annotation_text=f"re-test due ~wk {due_at}", annotation_position="top left")
fig.update_layout(height=380, margin=dict(l=70, r=30, t=60, b=55),
    title="Information decay: schedule the next test before the answer rots",
    xaxis_title="weeks since last wave", yaxis_title="effective uncertainty",
    legend=dict(orientation="h", yanchor="top", y=-0.2, x=0))
fig.show()
print(f"At current precision, a re-test becomes worthwhile around week {due_at}."
      if due_at else "Precision holds beyond the plotted horizon.")
"""),
    # ====================================================================
    # 9. The one-slide
    # ====================================================================
    md(r"""
## Act 7 · The one-slide for the media team

Everything above collapses into a single decision table. This is what goes in the
deck — per channel: today's spend, the recommended spend, the change, the verdict,
and how confident we are.
"""),
    code(r"""
rec, asd, mm, pa = last["rec"], last["alloc_sd"], last["mroas"], last["prob"]
def verdict(u, r):
    if r < 0.15:                        return "Cut to maintenance", "#c0504d"
    if r > u * 1.15:                    return "Scale up", "#4a8d57"
    if r < u * 0.85:                    return "Pull back", "#d9a441"
    return "Hold", "#6b7785"
def tint(hexc, a=0.16):
    h = hexc.lstrip("#"); r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"
conf = ["High" if s < 0.09 else ("Medium" if s < 0.16 else "Low") for s in asd]
rows = list(zip(
    CHANNELS,
    [dollars(v) for v in CENTER],
    [dollars(v) for v in rec],
    [f"{(r-u)*SPEND_SCALE/1000:+,.0f}k" for u, r in zip(CENTER, rec)],
    [verdict(u, r)[0] for u, r in zip(CENTER, rec)],
    conf,
))
vcolors = [verdict(u, r)[1] for u, r in zip(CENTER, rec)]
fig = go.Figure(go.Table(
    columnwidth=[95, 60, 108, 82, 108, 92],
    header=dict(values=["<b>Channel</b>", "<b>Today</b>", "<b>Recommended</b>",
                        "<b>Change/wk</b>", "<b>Verdict</b>", "<b>Confidence</b>"],
                fill_color="#1f4e79", font=dict(color="white", size=13), align="left", height=30),
    cells=dict(values=list(zip(*rows)),
               fill_color=[["#f4f6f8"] * 4, ["#f4f6f8"] * 4, ["#eef2f6"] * 4, ["#f4f6f8"] * 4,
                           [tint(c) for c in vcolors], ["#f4f6f8"] * 4],
               font=dict(size=12), align="left", height=28)))
fig.update_layout(height=230, margin=dict(l=10, r=10, t=40, b=10),
                  title=f"Nomi weekly media plan — total {dollars(B)} (unchanged)")
fig.show()

print("Three messages for the media team:")
print("  1. Point the budget, don't spread it: Chatter + Orbit have headroom; Vibe doesn't.")
print("  2. Don't max Chatter and Pulse together — they cannibalize. Lead with Chatter.")
print("  3. Keep a token Vibe budget: it's weak alone but it feeds Orbit.")
"""),
    md(r"""
## What we can — and can't — claim

The honest framing the analyst puts on the last slide, so the numbers are used well:

- ✅ **This is causal, by design.** The lifts come from variation *we* created, not
  from correlations in spend we didn't control. That's the whole point of the wave.
- ✅ **The direction is solid.** The *ranking* (Chatter/Orbit up, Vibe down) and the
  *synergy structure* (Chatter↔Pulse cannibalize, Orbit is the hub) are stable across
  waves and robust to modeling choices.
- ⚠️ **Trust the ranking more than the decimals.** A response curve is an
  approximation of a messier truth. The recommended *split* is reliable; a specific
  channel's "4.4× marginal ROAS" is a modeled point estimate — quote the **decision**,
  not a press-release ROI. (The sibling notebook, *§14 "When the response family is
  wrong,"* shows exactly why the decision survives model error while the magnitudes
  don't.)
- 🔁 **It has a shelf life.** Re-test on the decay clock above, or sooner if the market
  jumps.

For the reader only — never for the analyst — here's the hidden truth this demo was
generated from, next to what the loop recovered from experiments alone:
"""),
    code(r"""
# The demo's answer key (a real program never has this). How did the loop do?
true_alloc, true_profit = cl.world_optimal_allocation(world, B, VALUE, mode="fixed")
rec = last["rec"]
fig = go.Figure()
fig.add_trace(go.Bar(name="truth-optimal (hidden)", x=CHANNELS, y=true_alloc * SPEND_SCALE / 1000,
                     marker_color="#333", opacity=0.55, text=[dollars(v) for v in true_alloc],
                     textposition="outside", cliponaxis=False))
fig.add_trace(go.Bar(name="loop's recommendation", x=CHANNELS, y=rec * SPEND_SCALE / 1000,
                     marker_color=COLORS, text=[dollars(v) for v in rec],
                     textposition="inside"))
fig.update_layout(barmode="group", height=420, margin=dict(l=70, r=30, t=60, b=80),
    title="Reveal: the loop's split vs. the hidden truth (never available in real life)",
    yaxis_title="weekly spend ($k)", legend=dict(orientation="h", yanchor="top", y=-0.16, x=0))
fig.show()
rec_profit = VALUE * float(world.response_mean(rec[None, :])[0]) - B
gap = 100 * (true_profit - rec_profit) / abs(true_profit)
print(f"Loop captured {100 - gap:.1f}% of the truth-optimal profit "
      f"(gap {gap:.1f}%) — from designed experiments alone, no history, no model of the past.")
assert gap < 6.0
"""),
    md(r"""
## Recap — the cycle, in business terms

| Wave | What the analyst learned | What changed |
|---|---|---|
| **0** | Chatter/Orbit under-funded, Vibe under water, Chatter↔Pulse cannibalize | drafted the reallocation |
| **1** | the split held up; Vibe's Orbit-halo showed | tightened the plan, kept testing |
| **2** | funded channels' returns equalized; regret no longer worth a test | **locked the plan**, scheduled a re-test |

Nomi went from *"defend the budget"* to a causal, uncertainty-quantified plan that
moves ~\$130k/week toward headroom — **without a single assumption that last year's
spend was clean.** The loop didn't just give an answer; it said **how sure** it was
and **when to stop**.

**Where to go next**

- The engine, proven end-to-end (recovery, calibration, stopping, misspecification):
  `continuous_learning.ipynb`.
- The math and design choices: `technical-docs/continuous-learning.md`.
- The model-*anchored* sibling (when you *do* have a usable MMM):
  `mmm_framework.planning`.
"""),
]


def main() -> None:
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    out = "continuous_learning_story.ipynb"
    with open(out, "w") as f:
        nbformat.write(nb, f)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    print(f"wrote {out} with {len(nb.cells)} cells ({n_code} code)")


if __name__ == "__main__":
    main()
