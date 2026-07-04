"""
Continuous Sequential Learning — model-free geo response-surface bandit
======================================================================

Learn how spend drives outcome **directly from designed experiments**, with no
pre-fit MMM. The loop:

1. Run a designed wave (central-composite geo cells) and observe outcomes.
2. Fit a Bayesian response surface (Hill activation + sign-informed synergies).
3. Recover the channel effects and audit what the data identified.
4. Plan: a Thompson posterior over the optimal split + a funding line.
5. Decide whether another wave is worth it (the ENBS stopping rule), and if so,
   recenter on the recommendation and run again.

This mirrors ``assets/continous_learning.md`` (the implementation guide) and the
``mmm_framework.continuous_learning`` package. The Hill activation matches the
framework's ``SaturationType.HILL`` (``slope = alpha``, ``sat_half = kappa``), so
this posterior is directly comparable to a BayesianMMM Hill fit on the same data.

Run::

    uv run python examples/ex_continuous_learning.py
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np

import mmm_framework.continuous_learning as cl


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    rng_seed = 0
    B = 3.2  # budget in scaled units
    value = 5.0  # $ per unit KPI -> sets the funding line (value * dR/ds = 1)
    center = np.array([0.7, 0.7, 0.7, 0.7])  # current operating allocation (scaled)

    # ── A known world (so we can check recovery) ──────────────────────────────
    section("1. A known world")
    world = cl.make_world(seed=rng_seed)
    print("channels      :", world.channels)
    print("true beta     :", np.round(world.beta, 2))
    print("true kappa    :", np.round(world.kappa, 2))
    print("true alpha    :", np.round(world.alpha, 2))
    print(
        "true synergies:",
        {
            f"{world.channels[i]}x{world.channels[j]}": round(float(g), 2)
            for (i, j), g in zip(world.pairs, world.gamma_pairs)
            if abs(g) > 1e-9
        },
    )

    # ── One designed wave + a fit (RECOVERY) ──────────────────────────────────
    section("2. Run a wave, fit the surface, recover the effects")
    data = cl.simulate_panel(
        world, center, n_geo=80, t_pre=6, t_test=10, delta=0.6, noise=0.6, seed=1
    )
    print(
        f"panel: {data['spend'].shape[0]} geo-weeks, "
        f"{data['design'].shape[0]} CCD cells (center, axial, off-axis, shutoff)"
    )

    post = cl.fit(
        data,
        channels=world.channels,
        pair_signs=cl.PAIR_SIGNS_EXAMPLE,  # domain knowledge: which pairs synergize
        num_warmup=500,
        num_samples=500,
        num_chains=2,
        seed=rng_seed,
    )
    print(
        f"posterior: {post.n_draws} draws  (max R-hat "
        f"{post.diagnostics.get('max_rhat')})"
    )
    print(
        "beta  true ->",
        np.round(world.beta, 2),
        " hat ->",
        np.round(post.samples["beta"].mean(0), 2),
    )
    print("\nsynergy recovery (sign is reliable, magnitude is prior-sensitive):")
    for name, s in post.gamma_summary().items():
        print(
            f"  {name:24s} {s['sign']:>5s}  mean {s['mean']:+.2f}  "
            f"[{s['p5']:+.2f}, {s['p95']:+.2f}]"
        )

    # ── Plan: recommendation + funding line ───────────────────────────────────
    section("3. Plan — Thompson recommendation + funding line")
    rec = cl.recommend_allocation(post, B, value, q=300, mode="fixed")
    mroas_mean, prob_above, _ = cl.marginal_roas(post, rec, value, q=300)
    print(f"recommended split (sum={rec.sum():.2f} of B={B}):")
    for c, name in enumerate(world.channels):
        funded = "FUND" if prob_above[c] > 0.5 else "hold"
        print(
            f"  {name:8s} spend {rec[c]:.2f}   mROAS {mroas_mean[c]:.2f}   "
            f"P(>1)={prob_above[c]:.0%}  [{funded}]"
        )

    true_alloc, true_profit = cl.world_optimal_allocation(world, B, value, mode="fixed")
    print(
        f"\ntruth-optimal split: {np.round(true_alloc, 2)}  (profit {true_profit:.2f})"
    )

    # ── The full sequential loop (CLOSURE + STOPPING) ─────────────────────────
    section("4. The continuous-learning loop (carry posterior, stop on ENBS)")
    out = cl.run_closed_loop(
        world,
        center=center,
        B=B,
        value=value,
        n_geo=80,
        t_pre=6,
        t_test=10,
        delta=0.6,
        noise=0.6,
        mode="fixed",
        pair_signs=cl.PAIR_SIGNS_EXAMPLE,
        margin=1.0,
        population=2.0,
        wave_cost=0.5,
        max_waves=4,
        planner_q=200,
        fit_kwargs=dict(num_warmup=400, num_samples=400, num_chains=2, seed=rng_seed),
        seed=7,
    )
    print(
        f"{'wave':>4} {'rows':>6} {'E[regret]':>10} {'ENBS':>8} {'stop':>5} "
        f"{'gap%':>6}  recommendation"
    )
    for r in out["history"]:
        print(
            f"{r['wave']:>4} {r['n_rows']:>6} {r['e_regret']:>10.3f} "
            f"{r['enbs']:>+8.3f} {str(r['stop']):>5} {100 * r['profit_gap_rel']:>5.1f}%  "
            f"{np.round(r['recommendation'], 2)}"
        )
    print(
        "\nE[regret] shrinks as evidence accumulates; the loop halts when no "
        "wave's\nexpected reallocation value clears its cost (ENBS <= 0)."
    )
    print("final recommendation:", np.round(out["final_recommendation"], 2))
    print("truth-optimal split :", np.round(out["true_allocation"], 2))

    # ── Shape-agnostic activation: the monotone spline ────────────────────────
    section("5. No family assumption — the monotone-spline activation")
    # The true response here is a two-Hill mixture (a two-phase shape no single
    # parametric family in the registry matches). activation="monotone_spline"
    # fits a normalized monotone I-spline instead of guessing a family: the
    # only assumptions are monotone + saturating.
    world_mix = cl.make_world_hill_mixture(seed=rng_seed)
    data_mix = cl.simulate_panel(
        world_mix, center, n_geo=72, t_pre=6, t_test=10, delta=0.6, noise=0.4, seed=1
    )
    post_spl = cl.fit(
        data_mix,
        channels=world_mix.channels,
        pair_signs=cl.PAIR_SIGNS_EXAMPLE,
        activation="monotone_spline",  # positive I-spline weights w1..w6
        num_warmup=400,
        num_samples=400,
        num_chains=2,
        seed=rng_seed,
    )
    print(
        f"activation: {post_spl.activation}  shape sites: {post_spl.shape_names}  "
        f"(max R-hat {post_spl.diagnostics.get('max_rhat')})"
    )
    rec_spl = cl.recommend_allocation(post_spl, B, value, q=200, mode="fixed")
    achieved = value * float(world_mix.response_mean(rec_spl[None, :])[0]) - B
    _, mix_profit = cl.world_optimal_allocation(world_mix, B, value, mode="fixed")
    print(
        f"recommended split {np.round(rec_spl, 2)} captures "
        f"{100 * achieved / mix_profit:.1f}% of the truth-optimal profit —\n"
        "the true curve is a two-Hill mixture the spline never had to know."
    )


if __name__ == "__main__":
    main()
