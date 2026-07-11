"""Multi-seed re-run of the misspecification study, with Monte Carlo SEs.

The single-cycle numbers quoted by ``docs/continuous-learning-math.html``
(§Misspecification), the narrative page and ``nbs/continuous_learning/continuous_learning.ipynb``
§14 — profit gap 0.9% (single Hill) / 1.4% (logistic) / 0.9% (correct
mixture), marginal-ROAS 90%-CI coverage 3/4 / 2/4 / 4/4, and the sequential
convergence 0.9 -> 0.5 -> 0.2 -> 0.3% — were one-seed point estimates. Per
Morris, White & Crowther (2019, *Statistics in Medicine*), a simulation-study
headline needs a Monte Carlo standard error, and a coverage read out of 4
Bernoulli draws is essentially uninformative (worst-case MCSE = 25pp).

This script repeats the exact study over many independent replications —
same known two-Hill-mixture world, same fit settings as the notebook; each
replication redraws the geo panel, the observation noise and the sampler
seed — and reports every headline as estimate ± MCSE, with coverage as a
proportion with a Wilson 95% interval.

Usage (from the repo root; takes ~15-25 min with 4 workers):

    uv run python scripts/run_misspec_mcse.py \
        [--one-wave-seeds 24] [--seq-seeds 10] [--workers 4] \
        [--out nbs/artifacts/misspec_mcse.json]

The JSON artifact is what the docs pages quote; re-run this script before
changing those numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

# ── study constants: MUST match nbs/builders/build_continuous_learning.py §14 ─────────
CENTER = np.full(4, 0.7)
B = 3.2
VALUE = 5.0
N_GEO = 72
T_PRE = 6
T_TEST = 10
DELTA = 0.6
NOISE = 0.4
FAMILIES = {"hill_mixture": 600, "hill": 400, "logistic": 400}  # num_warmup/samples
SEQ_FAMILIES = {"hill": 300, "hill_mixture": 500}
SEQ_WAVES = 4


def _world():
    import mmm_framework.continuous_learning as cl

    return cl, cl.make_world_hill_mixture(seed=0)


def _true_profit(cl, world):
    _, true_profit = cl.world_optimal_allocation(world, B, VALUE, mode="fixed")
    return float(true_profit)


def _prof_true(world, a):
    return VALUE * float(world.response_mean(np.asarray(a, float)[None, :])[0]) - B


def _true_mroas(world, a):
    a = np.asarray(a, float)
    eps = 1e-3
    r0 = float(world.response_mean(a[None, :])[0])
    out = np.empty(a.size)
    for c in range(a.size):
        ap = a.copy()
        ap[c] += eps
        out[c] = VALUE * (float(world.response_mean(ap[None, :])[0]) - r0) / eps
    return out


def one_wave_task(args: tuple[str, int]) -> dict:
    """One (family, seed) replication of the one-wave study."""
    family, seed = args
    cl, world = _world()
    true_profit = _true_profit(cl, world)
    data = cl.simulate_panel(
        world,
        CENTER,
        n_geo=N_GEO,
        t_pre=T_PRE,
        t_test=T_TEST,
        delta=DELTA,
        noise=NOISE,
        seed=seed,
    )
    nw = FAMILIES[family]
    post = cl.fit(
        data,
        channels=world.channels,
        pair_signs=cl.PAIR_SIGNS_EXAMPLE,
        activation=family,
        num_warmup=nw,
        num_samples=nw,
        num_chains=2,
        seed=seed,
    )
    rec = cl.recommend_allocation(post, B, VALUE, q=200, mode="fixed")
    gap = 100.0 * (true_profit - _prof_true(world, rec)) / abs(true_profit)
    _, _, mr = cl.marginal_roas(post, rec, VALUE, q=200)
    lo, hi = np.percentile(mr, 5, 0), np.percentile(mr, 95, 0)
    tmr = _true_mroas(world, rec)
    covers = ((lo <= tmr) & (tmr <= hi)).astype(int)
    rhat = post.diagnostics.get("max_rhat")
    return {
        "part": "one_wave",
        "family": family,
        "seed": int(seed),
        "profit_gap_pct": float(gap),
        "covers": [int(c) for c in covers],
        "n_channels": int(covers.size),
        "ci_width_mean": float(np.mean(hi - lo)),
        "max_rhat": None if rhat is None else float(rhat),
    }


def sequential_task(args: tuple[str, int]) -> dict:
    """One (family, seed) replication of the 4-wave accumulating loop."""
    family, seed0 = args
    cl, world = _world()
    true_profit = _true_profit(cl, world)
    nw = SEQ_FAMILIES[family]
    state = cl.LearningState(
        channels=world.channels,
        center=CENTER.copy(),
        B=B,
        value=VALUE,
        pairs=world.pairs,
        pair_signs=cl.PAIR_SIGNS_EXAMPLE,
        activation=family,
        mode="fixed",
    )
    w0 = cl.simulate_panel(
        world,
        CENTER,
        n_geo=N_GEO,
        t_pre=T_PRE,
        t_test=T_TEST,
        delta=DELTA,
        noise=NOISE,
        seed=seed0,
    )
    a_geo = np.asarray(w0["a_geo"])
    state.ingest(w0)
    gaps = []
    for wave in range(SEQ_WAVES):
        state.fit(num_warmup=nw, num_samples=nw, num_chains=2, seed=seed0 * 10 + wave)
        rec = state.recommend(q=200)
        gaps.append(100.0 * (true_profit - _prof_true(world, rec)) / abs(true_profit))
        if wave == SEQ_WAVES - 1:
            break
        design = cl.central_composite(rec, DELTA, world.pairs)
        wave_next = cl.simulate_wave(
            world,
            design,
            a_geo,
            t_test=T_TEST,
            center=rec,
            noise=NOISE,
            seed=seed0 * 10 + 50 + wave,
        )
        state.recenter(rec)
        state.ingest(wave_next)
    return {
        "part": "sequential",
        "family": family,
        "seed": int(seed0),
        "gaps_pct": [float(g) for g in gaps],
    }


# ── aggregation (Morris, White & Crowther 2019 MCSEs; Wilson interval) ───────


def _mean_mcse(x: list[float]) -> dict:
    a = np.asarray(x, dtype=float)
    n = a.size
    return {
        "n": int(n),
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if n > 1 else None,
        "mcse": float(a.std(ddof=1) / math.sqrt(n)) if n > 1 else None,
    }


def _wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def aggregate(rows: list[dict]) -> dict:
    out: dict = {"one_wave": {}, "sequential": {}}
    for family in FAMILIES:
        fr = [r for r in rows if r["part"] == "one_wave" and r["family"] == family]
        if not fr:
            continue
        seed_cov = [float(np.mean(r["covers"])) for r in fr]
        pooled_hits = int(sum(sum(r["covers"]) for r in fr))
        pooled_n = int(sum(r["n_channels"] for r in fr))
        rhats = [r["max_rhat"] for r in fr if r["max_rhat"] is not None]
        out["one_wave"][family] = {
            "profit_gap_pct": _mean_mcse([r["profit_gap_pct"] for r in fr]),
            "coverage": {
                **_mean_mcse(seed_cov),
                "pooled": pooled_hits / pooled_n,
                "pooled_hits": pooled_hits,
                "pooled_n": pooled_n,
                "wilson95": list(_wilson(pooled_hits, pooled_n)),
            },
            "ci_width": _mean_mcse([r["ci_width_mean"] for r in fr]),
            "rhat_gt_1.1_share": (
                float(np.mean([r > 1.1 for r in rhats])) if rhats else None
            ),
            "rhat_gt_1.1_n": int(sum(r > 1.1 for r in rhats)),
            "rhat_n": len(rhats),
        }
    for family in SEQ_FAMILIES:
        fr = [r for r in rows if r["part"] == "sequential" and r["family"] == family]
        if not fr:
            continue
        out["sequential"][family] = [
            _mean_mcse([r["gaps_pct"][w] for r in fr]) for w in range(SEQ_WAVES)
        ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one-wave-seeds", type=int, default=24)
    ap.add_argument("--seq-seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="nbs/artifacts/misspec_mcse.json")
    args = ap.parse_args()

    tasks_one = [
        (fam, 1000 + i) for fam in FAMILIES for i in range(args.one_wave_seeds)
    ]
    tasks_seq = [(fam, 2000 + i) for fam in SEQ_FAMILIES for i in range(args.seq_seeds)]
    t0 = time.time()
    rows: list[dict] = []
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futs = [(ex.submit(one_wave_task, t), t) for t in tasks_one] + [
            (ex.submit(sequential_task, t), t) for t in tasks_seq
        ]
        for i, (fut, t) in enumerate(futs):
            rows.append(fut.result())
            print(
                f"[{i + 1}/{len(futs)}] done {t} " f"({time.time() - t0:.0f}s elapsed)",
                flush=True,
            )

    result = {
        "config": {
            "world": "make_world_hill_mixture(seed=0)",
            "center": [float(x) for x in CENTER],
            "B": B,
            "value": VALUE,
            "n_geo": N_GEO,
            "t_pre": T_PRE,
            "t_test": T_TEST,
            "delta": DELTA,
            "noise": NOISE,
            "one_wave_seeds": args.one_wave_seeds,
            "seq_seeds": args.seq_seeds,
            "fit": {
                "families": FAMILIES,
                "seq_families": SEQ_FAMILIES,
                "num_chains": 2,
            },
            "ci": "90% equal-tailed (percentile 5/95), as in nbs §14",
            "runtime_seconds": round(time.time() - t0, 1),
        },
        "aggregates": aggregate(rows),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out} in {time.time() - t0:.0f}s")
    agg = result["aggregates"]
    for fam, a in agg["one_wave"].items():
        g, c = a["profit_gap_pct"], a["coverage"]
        print(
            f"{fam:14s} gap {g['mean']:.2f}% ± {g['mcse']:.2f} | "
            f"coverage {c['pooled']:.2f} "
            f"[{c['wilson95'][0]:.2f}, {c['wilson95'][1]:.2f}] | "
            f"CI width {a['ci_width']['mean']:.2f} | "
            f"Rhat>1.1 {a['rhat_gt_1.1_n']}/{a['rhat_n']}"
        )
    for fam, waves in agg["sequential"].items():
        print(
            f"seq {fam:14s} "
            + " -> ".join(f"{w['mean']:.2f}±{w['mcse']:.2f}%" for w in waves)
        )


if __name__ == "__main__":
    main()
