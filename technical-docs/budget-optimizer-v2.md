# Budget optimizer v2 — #139

The v1 optimizer answered one question: *"what's the optimal allocation at exactly
this budget?"* (greedy marginal allocation on the posterior-mean response curves,
re-run per draw for decision stability). Planners ask four more:

1. **How does return scale as I add or cut budget?** → the **efficient frontier**.
2. **What budget hits my target KPI?** → **goal-seek**.
3. **What's the safe allocation if I'm risk-averse?** → **risk-aware objectives**.
4. **Fund each channel to breakeven** → **free-budget mode** + shadow prices.

Plus richer **constraints** (absolute-$ bounds, portfolio groups, a keep-on
floor). All of it runs on the same posterior-sampled `ResponseCurves` — the same
saturation/adstock-aware curves v1 used — so it is a decision layer, not a new
model.

## Where it lives

- `planning/budget.py` — the engine: `objective_curves` (risk curves),
  `_solve_allocation` (the constrained SLSQP solve), and the extended
  `optimize_budget` (advanced knobs + shadow price + marginal ROAS). The historical
  **mean / fixed / no-constraint path is byte-for-byte unchanged** (greedy); any v2
  feature routes through the solver.
- `planning/frontier.py` — `budget_frontier` and `goal_seek` (+ their result
  dataclasses), thin wrappers over the solver.
- Wiring: model-op `plan_budget` / `optimize_budget` (`agents/model_ops.py`), agent
  tool `run_budget_optimizer` (`agents/tools.py`), REST `POST
  /projects/{id}/planner/optimize` (`api/main.py`), React `PlannerStudio` +
  `AllocationResult`.

## Objectives (risk-aware)

`objective` selects the response curve the allocator maximizes:

| value | meaning |
|-------|---------|
| `mean` (default) | posterior-mean KPI — risk-neutral |
| `p<q>` (e.g. `p10`) | the per-channel q-th percentile across draws — a **downside** curve |
| `cvar<a>` (e.g. `cvar5`) | the per-channel mean of the worst a% of draws — **expected shortfall** |

**Approximation note.** The downside/CVaR curves are built *per channel* (the
q-th percentile of each channel's contribution independently). This is a
conservative proxy for the joint-portfolio quantile — it ignores the cross-channel
correlation of the posterior draws, so it slightly over-weights each channel's own
downside. It is the standard robust-optimization simplification and keeps the
solver structure (and speed) identical to the mean case; the result reports which
objective was used so the plan is never silently risk-shifted.

## Modes: fixed vs free (breakeven)

- `mode="fixed"` — spend exactly `total_budget` (the default). At the optimum the
  marginal return per dollar is equalized across interior funded channels — that
  common level is the **budget shadow price** (`shadow_price`), the marginal KPI
  value of the next dollar. Invest more while it exceeds your breakeven.
- `mode="free"` — the total is itself a decision: fund each channel until its
  marginal return hits the breakeven line, maximizing `value_per_kpi·KPI − spend`.
  The total budget becomes an **output**. `value_per_kpi` converts a non-revenue
  KPI to dollars (default 1.0 for a revenue KPI).

Both modes report per-channel **marginal ROAS** (`marginal_roas`) at the plan —
the funding line: fund a channel while its marginal return exceeds 1.

## Constraints

- **Absolute-$ bounds** — `abs_bounds={"TV": [50000, 120000]}` overrides the
  multiplier bounds for that channel.
- **Portfolio groups** — `groups=[{"name": "digital", "channels": [...],
  "min_share": 0.4}]` ("digital ≥ 40% of budget"). `min_share`/`max_share` are
  budget fractions; `min_spend`/`max_spend` are dollars.
- **Keep-on floor** — `min_channel_spend` (a scalar or per-channel dict) raises
  every channel's lower bound so no channel is turned off.

All are validated up front (unknown channels, malformed ranges → a clear error,
so a constraint is never silently dropped and a planner never gets an unexecutable
plan).

## Efficient frontier

`budget_frontier(mmm, budget_min, budget_max, n_points, objective, ...)` sweeps a
budget range (default 40%–200% of current spend) and, at each budget, optimizes
the objective curve, evaluates the plan across all posterior draws for a return
band, and records the discrete slope as the **marginal ROI** at that budget. The
frontier is monotone increasing and concave (diminishing returns), so it doubles
as the goal-seek search space. `FrontierResult.to_dict()` feeds the React frontier
table.

## Goal-seek

`goal_seek(mmm, target_kpi, objective, ...)` bisects the monotone frontier for the
**minimum budget** whose optimized plan reaches `target_kpi` total contribution,
returning the budget, the mix, and **`prob_hit_target`** — the posterior
probability the plan actually clears the target (the honest answer to "will this
budget hit my number?"). An unreachable target returns `feasible=False` with the
max attainable return, rather than a fictitious budget.

## Usage

```python
from mmm_framework import planning as pl

# risk-averse, digital ≥ 40%, keep every channel on, plus the frontier
res = pl.optimize_budget(
    mmm, total_budget=1_000_000, objective="cvar5",
    groups=[{"name": "digital", "channels": ["Search", "Social"], "min_share": 0.4}],
    min_channel_spend=10_000,
)
print(res.shadow_price, res.marginal_roas)

frontier = pl.budget_frontier(mmm, objective="mean")
gs = pl.goal_seek(mmm, target_kpi=5_000_000)      # min budget to hit 5M KPI

# breakeven: fund each channel to marginal ROAS = 1
be = pl.optimize_budget(mmm, mode="free", value_per_kpi=1.0)
```

Agent: `run_budget_optimizer(objective="cvar5", mode="free", groups=[...],
frontier=True, target_kpi=...)`. REST: `POST /projects/{id}/planner/optimize` with
the matching body fields. UI: the Planner studio's **Decision mode** panel.

## Scope & limits

- The v2 features run on the **national** response curves; `by_geo` keeps the
  historical per-geo greedy path (geo-aware advanced constraints are a follow-up).
- The response curves are **additive across channels** (no cross-channel synergy
  term), so the allocation problem is convex and separable — every constraint and
  the breakeven mode apply cleanly. A model with `channel_interactions` (#142) is
  optimized on its per-channel curves; the synergy is in the fit, not the optimizer.
- Recommendations past observed spend are flagged (issue #105) and their intervals
  widened — the frontier's far right is curve extrapolation; confirm large
  scale-ups with an experiment.

## Tests

`tests/test_budget_optimizer_v2.py` — objective curves + risk ordering, the
constrained solver (fixed budget, binding group, absolute bounds, breakeven),
`optimize_budget` advanced knobs + shadow price/marginal ROAS, the frontier
(monotone + diminishing marginal ROI + bands), goal-seek (reachable + infeasible),
and the op layer (validation errors + v2 payload).
