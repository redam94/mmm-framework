"""Tests for the model-free continuous-learning loop.

Fast unit tests cover the surface math, the central-composite design, the
allocator, and the stopping arithmetic (no MCMC). The slow tests are the three
feasibility gates from the guide (``assets/continous_learning.md`` §8): recovery
(fit a known world and get the effects back), the prior-sensitivity audit (the
``gamma_scale`` knob), and closure/stopping (the loop carries the posterior, the
recommendation tracks truth, and the ENBS rule fires before testing forever).
"""

from __future__ import annotations

import numpy as np
import pytest

import mmm_framework.continuous_learning as cl
from mmm_framework.continuous_learning import (
    acquisition,
    design,
    model,
    planner,
    preprocess,
    surface,
)

# ── fast: surface ─────────────────────────────────────────────────────────────


def test_activation_half_saturation_and_monotone():
    kappa = np.array([0.5, 1.0, 2.0])
    alpha = np.array([1.0, 2.0, 3.0])
    # at s == kappa the Hill fraction is exactly 0.5
    f = np.asarray(surface.activation(kappa, kappa, alpha))
    np.testing.assert_allclose(f, 0.5, atol=1e-6)
    # strictly increasing in spend
    lo = np.asarray(surface.activation(kappa * 0.5, kappa, alpha))
    hi = np.asarray(surface.activation(kappa * 1.5, kappa, alpha))
    assert np.all(lo < f) and np.all(f < hi)
    # zero spend -> ~0 activation (shutoff cell), finite (no nan)
    z = np.asarray(surface.activation(np.zeros(3), kappa, alpha))
    assert np.all(np.isfinite(z)) and np.all(z < 1e-3)


def test_incremental_no_interaction_is_sum_of_main_effects():
    k = 3
    spend = np.array([0.4, 0.8, 1.2])
    beta = np.array([1.0, 2.0, 0.5])
    kappa = np.array([0.6, 0.9, 1.1])
    alpha = np.array([1.5, 2.0, 1.2])
    gamma0 = np.zeros((k, k))
    f = np.asarray(surface.activation(spend, kappa, alpha))
    expected = float(np.sum(beta * f))
    got = float(surface.incremental(spend, beta, kappa, alpha, gamma0))
    assert got == pytest.approx(expected, rel=1e-6)


def test_grad_incremental_matches_finite_difference():
    spend = np.array([0.5, 0.7, 0.3, 0.9])
    beta = np.array([2.0, 1.5, 1.0, 0.8])
    kappa = np.array([0.8, 0.6, 1.0, 0.7])
    alpha = np.array([2.0, 1.5, 2.5, 1.3])
    gamma = np.zeros((4, 4))
    gamma[0, 1] = -0.4
    gamma[1, 2] = 0.3
    import jax.numpy as jnp

    g = np.asarray(
        surface.grad_incremental(jnp.asarray(spend), beta, kappa, alpha, gamma)
    )
    # JAX runs float32 by default (as in production), so use a step large
    # enough to avoid catastrophic cancellation and a float32-appropriate
    # tolerance — still tight enough to catch a wrong/missing term (those move
    # the gradient by O(0.1+)).
    eps = 1e-3
    fd = np.empty(4)
    for c in range(4):
        sp, sm = spend.copy(), spend.copy()
        sp[c] += eps
        sm[c] -= eps
        fd[c] = (
            float(surface.incremental(sp, beta, kappa, alpha, gamma))
            - float(surface.incremental(sm, beta, kappa, alpha, gamma))
        ) / (2 * eps)
    np.testing.assert_allclose(g, fd, atol=5e-3)


# ── fast: design ──────────────────────────────────────────────────────────────


def test_central_composite_structure():
    center = np.array([1.0, 1.0, 1.0, 1.0])
    probe = [(0, 1), (1, 2)]
    d = design.central_composite(center, 0.6, probe)
    k = 4
    assert d.shape == (1 + 2 * k + 2 * len(probe) + k, k)
    np.testing.assert_allclose(d[0], center)  # center cell
    assert np.all(d >= 0)  # non-negative
    # the last K rows are shutoffs: each has exactly one zeroed channel
    shutoffs = d[-k:]
    for c in range(k):
        assert shutoffs[c, c] == pytest.approx(0.0)


def test_central_composite_rejects_bad_delta():
    with pytest.raises(ValueError):
        design.central_composite(np.ones(3), 0.0, [])
    with pytest.raises(ValueError):
        design.central_composite(np.ones(3), 1.5, [])


def test_assign_geos_balanced_with_holdouts():
    center = np.array([0.8, 0.8, 0.8])
    d = design.central_composite(center, 0.5, [(0, 1)])
    rng = np.random.default_rng(0)
    geo_alloc, cell_idx = design.assign_geos(d, 30, rng, n_holdout=4, center=center)
    assert geo_alloc.shape == (30, 3)
    assert np.all(cell_idx[:4] == -1)  # first 4 are holdouts
    assert np.allclose(geo_alloc[:4], center)  # ... held at status quo
    assert cell_idx[4:].min() >= 0 and cell_idx[4:].max() < d.shape[0]


def test_assign_geos_stratified_balances_baseline():
    """Blocked randomization on a strong baseline gradient: per-cell counts
    match round-robin's (as a multiset) while the between-cell baseline-mean
    spread shrinks strictly below the shuffled round-robin's."""
    center = np.array([0.8, 0.8, 0.8])
    d = design.central_composite(center, 0.5, [(0, 1)])
    n_geo = 36
    baseline = np.linspace(0.0, 10.0, n_geo)  # strong monotone gradient

    _, rr_idx = design.assign_geos(d, n_geo, np.random.default_rng(0))
    _, st_idx = design.assign_geos(
        d, n_geo, np.random.default_rng(0), baseline=baseline
    )
    assert sorted(np.bincount(rr_idx, minlength=d.shape[0]).tolist()) == sorted(
        np.bincount(st_idx, minlength=d.shape[0]).tolist()
    )

    def spread(idx):
        means = [
            baseline[idx == c].mean() for c in range(d.shape[0]) if np.any(idx == c)
        ]
        return float(np.ptp(means))

    assert spread(st_idx) < spread(rr_idx)


def test_assign_geos_stratified_holdouts_span_baseline_range():
    center = np.array([0.8, 0.8, 0.8])
    d = design.central_composite(center, 0.5, [(0, 1)])
    baseline = np.arange(30, dtype=float)
    rng = np.random.default_rng(1)
    geo_alloc, cell_idx = design.assign_geos(
        d, 30, rng, n_holdout=4, center=center, baseline=baseline
    )
    hold = np.nonzero(cell_idx == -1)[0]
    assert hold.size == 4
    assert np.allclose(geo_alloc[hold], center)
    # 4 evenly spaced positions in baseline-sorted order -> spans the range
    assert set(hold.tolist()) == {0, 10, 19, 29}
    assert np.ptp(baseline[hold]) >= 0.5 * np.ptp(baseline)
    assert cell_idx[cell_idx >= 0].max() < d.shape[0]


@pytest.mark.parametrize(
    "n_geo,n_holdout",
    [(10, 6), (10, 7), (9, 4), (12, 8), (12, 5), (30, 4), (10, 10)],
)
def test_assign_geos_stratified_holdout_count_is_exact(n_geo, n_holdout):
    """The stratified path must hold out EXACTLY the requested count (the old
    strided pick returned only ceil(n_geo/step) geos — e.g. 10/6 -> 5) while
    still spanning the baseline range."""
    center = np.array([0.8, 0.8, 0.8])
    d = design.central_composite(center, 0.5, [(0, 1)])
    baseline = np.linspace(0.0, 100.0, n_geo)
    geo_alloc, cell_idx = design.assign_geos(
        d,
        n_geo,
        np.random.default_rng(0),
        n_holdout=n_holdout,
        center=center,
        baseline=baseline,
    )
    hold = np.nonzero(cell_idx == -1)[0]
    assert hold.size == n_holdout  # exact count, never fewer
    assert np.allclose(geo_alloc[hold], center)
    # spans the baseline range: extremes are always included
    assert baseline[hold].min() == pytest.approx(baseline.min())
    assert baseline[hold].max() == pytest.approx(baseline.max())
    # non-holdout geos still carry valid design cells
    if n_holdout < n_geo:
        rest = cell_idx[cell_idx >= 0]
        assert rest.min() >= 0 and rest.max() < d.shape[0]


def test_assign_geos_stratified_requires_center_and_aligned_baseline():
    center = np.array([0.8, 0.8, 0.8])
    d = design.central_composite(center, 0.5, [(0, 1)])
    with pytest.raises(ValueError, match="baseline"):
        design.assign_geos(d, 30, np.random.default_rng(0), baseline=np.arange(7.0))
    with pytest.raises(ValueError, match="center"):
        design.assign_geos(
            d, 30, np.random.default_rng(0), n_holdout=2, baseline=np.arange(30.0)
        )


# ── fast: planner ─────────────────────────────────────────────────────────────


def _concave_params(k):
    # alpha == 1 -> Hill is strictly concave (s / (k + s)), so the budget
    # allocation problem has a unique optimum we can reason about.
    return {
        "beta": np.ones(k),
        "kappa": np.ones(k),
        "alpha": np.ones(k),
        "gamma": np.zeros((k, k)),
    }


def test_allocator_symmetric_split_fixed_budget():
    params = _concave_params(4)
    alloc, _ = cl.allocate_under_sample(
        params, B=4.0, value=5.0, mode="fixed", n_starts=4
    )
    assert alloc.sum() == pytest.approx(4.0, abs=1e-2)
    np.testing.assert_allclose(alloc, 1.0, atol=0.05)  # identical channels -> equal


def test_allocator_free_mode_drops_dead_channel():
    params = _concave_params(2)
    params["beta"] = np.array([3.0, 1e-6])  # second channel is worthless
    alloc, _ = cl.allocate_under_sample(
        params, B=5.0, value=2.0, mode="free", cap=5.0, n_starts=4
    )
    assert alloc[1] < 1e-2  # free budget: don't fund the dead channel
    assert alloc[0] > 0.1


def test_allocator_rejects_infeasible_fixed_budget_cap():
    # fixed mode with cap < B/k makes the budget simplex unreachable
    # (k*cap < B) -> fail loudly rather than return an off-simplex fallback.
    params = _concave_params(4)
    with pytest.raises(ValueError, match="infeasible"):
        cl.allocate_under_sample(params, B=10.0, value=1.0, mode="fixed", cap=1.0)
    # free mode has no such constraint -> fine
    alloc, _ = cl.allocate_under_sample(params, B=10.0, value=1.0, mode="free", cap=1.0)
    assert np.all(alloc <= 1.0 + 1e-6)


def test_enbs_and_stop_arithmetic():
    # E[regret]=0.2, margin=1, population=10 -> value 2.0; cost 1.5 -> ENBS 0.5
    val = planner.enbs(0.2, margin=1.0, population=10.0, wave_cost=1.5)
    assert val == pytest.approx(0.5)
    stop, v = planner.should_stop(0.05, margin=1.0, population=10.0, wave_cost=1.5)
    assert stop is True and v < 0


# ── fast: model helpers ───────────────────────────────────────────────────────


def test_demote_channel_zeros_its_pairs():
    channels = ["Chatter", "Pulse", "Orbit", "Vibe"]
    signs = model.demote_channel(channels, "Vibe")
    # every pair touching Vibe (idx 3) -> "zero"
    assert (
        signs[(0, 3)] == "zero" and signs[(1, 3)] == "zero" and signs[(2, 3)] == "zero"
    )
    probe = model.probe_pairs_excluding(channels, "Vibe")
    assert all(3 not in p for p in probe)


def test_true_world_response_matches_surface():
    world = cl.make_world(seed=3)
    spend = np.array([[0.5, 0.6, 0.7, 0.4], [1.0, 0.2, 0.8, 0.9]])
    got = world.response_mean(spend)
    g = world.gamma_matrix()
    want = np.array(
        [
            float(surface.incremental(row, world.beta, world.kappa, world.alpha, g))
            for row in spend
        ]
    )
    np.testing.assert_allclose(got, want, atol=1e-6)


# ── fast: adstock pre-pass + CUPED (preprocess) ───────────────────────────────


def test_adstock_panel_convolves_within_geo():
    # n_geo=2, t_pre=1, t_test=3 -> T=4 weeks; impulse in geo 0 at week 0.
    n_geo, t_pre, t_test = 2, 1, 3
    spend = np.zeros((8, 1))  # rows: pre(w0:g0,g1) then test(w0,w1,w2 × g0,g1)
    spend[0, 0] = 1.0  # geo 0, overall week 0
    out = preprocess.adstock_panel(spend, n_geo, t_pre, t_test, alpha=0.5, l_max=6)
    from mmm_framework.config.enums import AdstockType
    from mmm_framework.transforms.adstock import adstock_weights

    w = adstock_weights(AdstockType.GEOMETRIC, 6, alpha=0.5, normalize=True)
    # geo-0 rows in overall-week order: 0 (pre w0), 2, 4, 6 (test w0..w2)
    geo0 = out[[0, 2, 4, 6], 0]
    np.testing.assert_allclose(geo0, w[:4], atol=1e-6)
    assert np.allclose(out[[1, 3, 5, 7], 0], 0.0)  # geo 1 untouched (no spend)


def test_cuped_reduces_geo_variance():
    world = cl.make_world(seed=0)
    center = np.array([0.8, 0.8, 0.8, 0.8])
    data = cl.simulate_panel(
        world, center, n_geo=60, t_pre=5, t_test=8, delta=0.6, noise=0.5, seed=1
    )
    adj, info = cl.cuped_adjust(data, t_pre=5)
    assert 0.0 < info["var_reduction"] < 1.0
    assert info["var_reduction"] == pytest.approx(1 - info["rho"] ** 2, abs=1e-9)
    # the adjusted test-period geo means have lower variance than the raw ones
    n_pre = 5 * 60
    raw = np.array(
        [data["y"][n_pre:][data["geo_idx"][n_pre:] == g].mean() for g in range(60)]
    )
    new = np.array(
        [adj["y"][n_pre:][adj["geo_idx"][n_pre:] == g].mean() for g in range(60)]
    )
    assert np.var(new) < np.var(raw)


# ── fast: acquisition (pure-EIG + Laplace KG) on a fabricated posterior ────────


def _fake_posterior(world, n_geo=40, n=300, seed=0):
    """A Gaussian-ish posterior around a known world — no MCMC needed."""
    rng = np.random.default_rng(seed)
    k = world.n_channels
    s = {
        "beta": np.abs(world.beta + 0.15 * rng.standard_normal((n, k))),
        "kappa": np.abs(world.kappa + 0.08 * rng.standard_normal((n, k))),
        "alpha": np.clip(world.alpha + 0.15 * rng.standard_normal((n, k)), 0.5, 5),
        "A": rng.normal(4, 0.3, n),
        "sigma_a": np.abs(rng.normal(1, 0.1, n)),
        "sigma": np.abs(rng.normal(0.5, 0.05, n)),
        "a_geo": rng.normal(4, 1, (n, n_geo)),
    }
    for idx, (i, j) in enumerate(world.pairs):
        s[model.pair_name(world.channels, (i, j))] = world.gamma_pairs[
            idx
        ] + 0.15 * rng.standard_normal(n)
    return cl.Posterior(
        samples=s, channels=world.channels, pairs=world.pairs, pair_signs={}
    )


def test_theta_map_layout_and_roundtrip():
    """The unconstrained packing keeps the historical [beta, kappa, alpha,
    gamma] Hill layout, and constrain(unconstrain(draw)) round-trips."""
    world = cl.make_world(seed=0)
    post = _fake_posterior(world)
    tmap = acquisition.theta_map(post)
    k, p = world.n_channels, len(world.pairs)
    assert tmap.dim == 3 * k + p
    assert tmap.gamma_idx == list(range(3 * k, 3 * k + p))
    m = tmap.unconstrain_draws(post.samples)
    assert m.shape == (300, tmap.dim)
    assert np.all(np.isfinite(m))
    params = tmap.constrain_params(m[0])
    # jnp is float32 by default -> ~1e-6 relative round-trip error
    np.testing.assert_allclose(params["beta"], post.samples["beta"][0], rtol=1e-4)
    np.testing.assert_allclose(params["shape"][0], post.samples["kappa"][0], rtol=1e-4)
    np.testing.assert_allclose(params["shape"][1], post.samples["alpha"][0], rtol=1e-3)
    np.testing.assert_allclose(params["gamma"], post.gamma_matrix(0), atol=1e-5)


def test_theta_map_sign_constrained_gammas_stay_signed():
    """neg/pos pair signs get sign-aware log transforms, so EVERY fantasy
    sample maps back to a gamma with the prior's sign — no clipping."""
    world = cl.make_world(seed=0)
    post = _fake_posterior(world)
    rng = np.random.default_rng(3)
    nm_neg = model.pair_name(world.channels, (0, 1))
    nm_pos = model.pair_name(world.channels, (1, 2))
    post.samples[nm_neg] = -np.abs(post.samples[nm_neg]) - 1e-3
    post.samples[nm_pos] = np.abs(post.samples[nm_pos]) + 1e-3
    post.pair_signs = {(0, 1): "neg", (1, 2): "pos"}
    tmap = acquisition.theta_map(post)
    mu, sigma0 = acquisition.theta_moments(post, tmap=tmap)
    etas = rng.multivariate_normal(mu, sigma0, size=50)
    for eta in etas:
        params = tmap.constrain_params(eta)
        assert params["gamma"][0, 1] < 0  # "neg" pair keeps its sign
        assert params["gamma"][1, 2] > 0  # "pos" pair keeps its sign
        assert np.all(params["beta"] > 0)  # log space -> positive
        assert np.all(params["shape"][0] > 0)  # kappa positive
        assert np.all((params["shape"][1] >= 0.5) & (params["shape"][1] <= 5.0))


def test_design_information_is_psd():
    world = cl.make_world(seed=0)
    post = _fake_posterior(world)
    mu, sigma0 = acquisition.theta_moments(post)
    dsg = cl.central_composite(np.array([0.7, 0.7, 0.7, 0.7]), 0.6, world.pairs)
    lam = acquisition.design_information(
        dsg, mu, sigma=0.4, k=world.n_channels, pairs=world.pairs, n_geo=40, t_test=8
    )
    assert np.allclose(lam, lam.T, atol=1e-8)
    assert np.linalg.eigvalsh(lam).min() > -1e-8  # PSD


def test_pure_eig_probing_lifts_synergy_identification():
    world = cl.make_world(seed=0)
    post = _fake_posterior(world)
    center = np.array([0.7, 0.7, 0.7, 0.7])
    full = cl.central_composite(center, 0.6, world.pairs)  # probes synergies
    noprobe = cl.central_composite(center, 0.6, [])  # axial + shutoff only
    ds_full = cl.design_eig(post, full, sigma=0.4, target="gamma", n_geo=40, t_test=8)
    ds_no = cl.design_eig(post, noprobe, sigma=0.4, target="gamma", n_geo=40, t_test=8)
    assert ds_full > ds_no >= 0.0  # off-axis cells identify gamma
    # D-optimal (all params) >= D_s-optimal (gamma sub-block) for the same design
    d_all = cl.design_eig(post, full, sigma=0.4, target="all", n_geo=40, t_test=8)
    assert d_all >= ds_full


def test_laplace_kg_finite_and_fast():
    world = cl.make_world(seed=0)
    post = _fake_posterior(world)
    dsg = cl.central_composite(np.array([0.7, 0.7, 0.7, 0.7]), 0.6, world.pairs)
    kg = cl.laplace_knowledge_gradient(
        post,
        dsg,
        B=3.2,
        value=5.0,
        sigma=0.4,
        n_geo=40,
        t_test=8,
        n_outcomes=32,
        seed=0,
    )
    assert np.isfinite(kg)


# ── fast: KG-driven design selection (loop.select_next_design) ────────────────


def test_select_next_design_returns_the_best_scored_candidate():
    world = cl.make_world(seed=0)
    post = _fake_posterior(world)
    center = np.array([0.7, 0.7, 0.7, 0.7])
    cells, meta = cl.select_next_design(
        post,
        center,
        world.pairs,
        B=3.2,
        value=5.0,
        candidate_deltas=(0.3, 0.6, 0.9),
        n_geo=40,
        t_test=8,
        n_outcomes=16,
        seed=0,
    )
    assert meta["kg_used"] is True
    assert meta["chosen_delta"] in (0.3, 0.6, 0.9)
    assert len(meta["kg_scores"]) == 3
    # the returned cells are exactly the chosen candidate's CCD
    np.testing.assert_allclose(
        cells, cl.central_composite(center, meta["chosen_delta"], world.pairs)
    )
    # ... and it carries the max score (common random numbers -> comparable)
    best = max(meta["kg_scores"], key=lambda s: s["score"])
    assert best["delta"] == meta["chosen_delta"]
    assert meta["sigma"] == pytest.approx(float(np.mean(post.samples["sigma"])))
    # the scores are finite and actually discriminate between candidates
    # (constant/NaN scores would make the argmax vacuous)
    vals = [float(s["score"]) for s in meta["kg_scores"]]
    assert all(np.isfinite(v) for v in vals)
    assert len(set(vals)) >= 2


def test_select_next_design_prefers_probes_when_synergy_is_decision_pivotal():
    # Rigged setup (the EIG-test pattern): tiny main-effect uncertainty, huge
    # synergy uncertainty — the remaining decision value is in learning gamma,
    # so the probe-pairs candidate beats the no-probe candidate. NB the KG gap
    # is genuinely small here (shutoff cells also identify gamma, and
    # main-effect cells move the allocation more — the documented KG≠EIG
    # finding), so a large ``n_outcomes`` under common random numbers is what
    # makes the argmax deterministic. The expected winner is deliberately
    # placed SECOND in the candidate list: ties (and NaN comparisons) resolve
    # to the FIRST candidate scored, so a scorer that collapses to a constant
    # would otherwise keep this test green without discriminating anything.
    world = cl.make_world(seed=0)
    rng = np.random.default_rng(0)
    n, k = 300, world.n_channels
    s = {
        "beta": np.abs(world.beta + 0.02 * rng.standard_normal((n, k))),
        "kappa": np.abs(world.kappa + 0.02 * rng.standard_normal((n, k))),
        "alpha": np.clip(world.alpha + 0.02 * rng.standard_normal((n, k)), 0.5, 5),
        "sigma": np.abs(rng.normal(0.5, 0.05, n)),
    }
    for idx, (i, j) in enumerate(world.pairs):
        s[model.pair_name(world.channels, (i, j))] = world.gamma_pairs[
            idx
        ] + 1.2 * rng.standard_normal(n)
    post = cl.Posterior(
        samples=s, channels=world.channels, pairs=world.pairs, pair_signs={}
    )
    _, meta = cl.select_next_design(
        post,
        np.full(4, 0.7),
        world.pairs,
        B=3.2,
        value=5.0,
        candidate_deltas=(0.6,),
        candidate_probe_sets=[[], world.pairs],  # winner NOT first-scored
        n_geo=40,
        t_test=8,
        n_outcomes=256,
        seed=0,
    )
    assert meta["kg_used"] is True
    assert meta["chosen_probe_pairs"] == [[int(i), int(j)] for i, j in world.pairs]
    # the scores genuinely discriminate: all finite, >= 2 distinct values, and
    # the probes candidate STRICTLY beats the no-probes candidate
    vals = [float(sc["score"]) for sc in meta["kg_scores"]]
    assert all(np.isfinite(v) for v in vals)
    assert len(set(vals)) >= 2
    by_probes = {
        tuple(map(tuple, sc["probe_pairs"])): float(sc["score"])
        for sc in meta["kg_scores"]
    }
    probes_key = tuple((int(i), int(j)) for i, j in world.pairs)
    assert by_probes[probes_key] > by_probes[()]


def test_select_next_design_falls_back_on_nan_scores(monkeypatch):
    """A scorer degenerating to NaN must NOT be reported as kg_used=True with
    the argmax silently landing on the first candidate (nan > nan is False)."""
    from mmm_framework.continuous_learning import acquisition

    world = cl.make_world(seed=0)
    post = _fake_posterior(world)
    monkeypatch.setattr(
        acquisition, "laplace_knowledge_gradient", lambda *a, **k: float("nan")
    )
    center = np.full(4, 0.7)
    cells, meta = cl.select_next_design(
        post,
        center,
        world.pairs,
        B=3.2,
        value=5.0,
        candidate_deltas=(0.3, 0.6),
        fallback_delta=0.6,
    )
    assert meta["kg_used"] is False
    assert "non-finite" in meta["reason"]
    np.testing.assert_allclose(cells, cl.central_composite(center, 0.6, world.pairs))


def test_select_next_design_falls_back_on_unsupported_posteriors():
    center = np.full(4, 0.7)
    # (a) an activation with no transform spec -> clean fallback
    world = cl.make_world(seed=0)
    post = _fake_posterior(world)
    post.activation = "not_registered"
    cells, meta = cl.select_next_design(
        post, center, world.pairs, B=3.2, value=5.0, fallback_delta=0.6
    )
    assert meta["kg_used"] is False and "transform spec" in meta["reason"]
    np.testing.assert_allclose(cells, cl.central_composite(center, 0.6, world.pairs))

    # (b) a count posterior MISSING its observation sites (no 'phi') -> fallback
    hill_world = cl.make_world(seed=0)
    post_nb = _fake_posterior(hill_world)
    del post_nb.samples["sigma"]
    post_nb.likelihood = "negbinomial"
    cells2, meta2 = cl.select_next_design(
        post_nb, center, hill_world.pairs, B=3.2, value=5.0, fallback_delta=0.4
    )
    assert meta2["kg_used"] is False and "negbinomial" in meta2["reason"]
    np.testing.assert_allclose(
        cells2, cl.central_composite(center, 0.4, hill_world.pairs)
    )

    # (b2) an unknown observation family -> fallback
    post_uk = _fake_posterior(hill_world)
    post_uk.likelihood = "poisson"
    _, meta_uk = cl.select_next_design(
        post_uk, center, hill_world.pairs, B=3.2, value=5.0, fallback_delta=0.4
    )
    assert meta_uk["kg_used"] is False and "poisson" in meta_uk["reason"]

    # (c) a Gaussian summaries-only posterior (no 'sigma' site) -> fallback,
    # NEVER a hard-coded noise guess: under prior_scaling="auto" its theta
    # lives on the KPI's natural scale, so a fixed sigma would make every
    # candidate score identically while reporting kg_used=True.
    post_summ = _fake_posterior(hill_world)
    del post_summ.samples["sigma"]
    assert post_summ.likelihood == "normal"
    cells3, meta3 = cl.select_next_design(
        post_summ, center, hill_world.pairs, B=3.2, value=5.0, fallback_delta=0.5
    )
    assert meta3["kg_used"] is False
    assert "observation-noise" in meta3["reason"]
    assert "summaries-only" in meta3["reason"]
    np.testing.assert_allclose(
        cells3, cl.central_composite(center, 0.5, hill_world.pairs)
    )


# ── fast: pluggable activations (not Hill-specific) ───────────────────────────


def test_activation_registry_and_logistic_properties():
    assert set(cl.ACTIVATIONS) >= {"hill", "logistic"}
    lam = np.array([1.0, 2.0])
    # logistic: f(0)=0, strictly increasing, saturating toward 1
    z = np.asarray(cl.logistic(np.zeros(2), lam))
    lo = np.asarray(cl.logistic(np.full(2, 0.5), lam))
    hi = np.asarray(cl.logistic(np.full(2, 5.0), lam))
    assert np.allclose(z, 0.0, atol=1e-6)
    assert np.all(z < lo) and np.all(lo < hi) and np.all(hi < 1.0)
    assert np.all(hi > 0.9)  # nearly saturated by 5 half-lives


def test_surface_value_hill_matches_incremental():
    # the general surface with the Hill activation reproduces `incremental`
    spend = np.array([0.4, 0.9, 1.3])
    beta = np.array([1.0, 2.0, 0.5])
    kappa = np.array([0.6, 0.9, 1.1])
    alpha = np.array([1.5, 2.0, 1.2])
    gamma = np.zeros((3, 3))
    gamma[0, 1] = -0.4
    general = float(cl.surface_value(spend, beta, gamma, cl.activation, (kappa, alpha)))
    direct = float(surface.incremental(spend, beta, kappa, alpha, gamma))
    assert general == pytest.approx(direct, rel=1e-6)


def test_true_world_logistic_response_matches_surface():
    world = cl.make_world_logistic(seed=1)
    assert world.activation == "logistic" and "lam" in world.shape
    spend = np.array([[0.5, 0.6, 0.7, 0.4], [1.0, 0.2, 0.8, 0.9]])
    got = world.response_mean(spend)
    want = np.asarray(
        cl.surface_over_rows(
            spend, world.beta, world.gamma_matrix(), world.act_fn(), world.shape_tuple()
        )
    )
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_hill_mixture_activation_properties():
    """The mixture activation is registered, well-behaved, and strictly more
    expressive than a single Hill (it can bend where one Hill cannot)."""
    assert "hill_mixture" in cl.ACTIVATIONS
    names, fn = cl.ACTIVATIONS["hill_mixture"]
    assert names == ("kappa1", "alpha1", "kappa2", "alpha2", "w")
    s = np.linspace(0, 2.0, 50)
    k1, a1, k2, a2, w = 0.35, 4.0, 1.5, 2.0, 0.5
    f = np.asarray(fn(s, k1, a1, k2, a2, w))
    assert f[0] == pytest.approx(0.0, abs=1e-6)  # f(0)=0
    assert np.all(np.diff(f) >= -1e-9)  # monotone non-decreasing
    assert f[-1] < 1.0  # saturating below 1
    # equals the explicit weighted sum of two Hills
    want = w * np.asarray(cl.activation(s, k1, a1)) + (1 - w) * np.asarray(
        cl.activation(s, k2, a2)
    )
    np.testing.assert_allclose(f, want, atol=1e-6)
    # a two-phase mixture reaches a DIFFERENT shape than either single Hill —
    # its mid-range value is not reproducible by averaging the components' kappas
    single = np.asarray(cl.activation(s, 0.5 * (k1 + k2), 0.5 * (a1 + a2)))
    assert np.max(np.abs(f - single)) > 0.03


def test_true_world_hill_mixture_response_matches_surface():
    world = cl.make_world_hill_mixture(seed=1)
    assert world.activation == "hill_mixture"
    assert {"kappa1", "alpha1", "kappa2", "alpha2", "w"} <= set(world.shape)
    spend = np.array([[0.5, 0.6, 0.7, 0.4], [1.0, 0.2, 0.8, 0.9]])
    got = world.response_mean(spend)
    want = np.asarray(
        cl.surface_over_rows(
            spend, world.beta, world.gamma_matrix(), world.act_fn(), world.shape_tuple()
        )
    )
    np.testing.assert_allclose(got, want, atol=1e-6)


def _fake_logistic_posterior(world, n=100, seed=0, with_sigma=True):
    rng = np.random.default_rng(seed)
    k = world.n_channels
    s = {
        "beta": np.abs(world.beta + 0.1 * rng.standard_normal((n, k))),
        "lam": np.abs(world.shape["lam"] + 0.1 * rng.standard_normal((n, k))),
        "a_geo": rng.normal(4, 1, (n, 30)),
    }
    if with_sigma:
        s["sigma"] = np.abs(rng.normal(0.5, 0.05, n))
    for idx, (i, j) in enumerate(world.pairs):
        s[model.pair_name(world.channels, (i, j))] = world.gamma_pairs[
            idx
        ] + 0.1 * rng.standard_normal(n)
    return cl.Posterior(
        samples=s, channels=world.channels, pairs=world.pairs, activation="logistic"
    )


def test_acquisition_supports_registered_non_hill_activations():
    # the registry-driven ThetaMap packs ANY activation with a transform spec:
    # a logistic posterior gets real moments, EIG and Laplace-KG scores.
    world = cl.make_world_logistic(seed=0)
    post = _fake_logistic_posterior(world)
    tmap = acquisition.theta_map(post)
    assert tmap.dim == 2 * world.n_channels + len(world.pairs)  # beta + lam + gammas
    mu, sigma0 = acquisition.theta_moments(post, tmap=tmap)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma0))
    dsg = cl.central_composite(np.full(world.n_channels, 0.7), 0.6, world.pairs)
    eig = cl.design_eig(post, dsg, sigma=0.5, n_geo=40, t_test=8)
    assert np.isfinite(eig) and eig > 0
    kg = cl.laplace_knowledge_gradient(
        post, dsg, B=3.2, value=5.0, n_geo=40, t_test=8, n_outcomes=8, seed=0
    )
    assert np.isfinite(kg)
    # an activation with NO transform spec still fails loudly
    post.activation = "not_registered"
    with pytest.raises(NotImplementedError, match="transform spec"):
        acquisition.theta_map(post)


def test_acquisition_negbinomial_glm_weights():
    """A count posterior scores via the softplus-link GLM Fisher weights."""
    world = cl.make_world(seed=0)
    rng = np.random.default_rng(5)
    post = _fake_posterior(world)
    del post.samples["sigma"]
    post.samples["phi"] = np.abs(rng.normal(30.0, 3.0, 300))
    post.likelihood = "negbinomial"
    center = np.full(4, 0.7)
    dsg = cl.central_composite(center, 0.6, world.pairs)
    tmap = acquisition.theta_map(post)
    mu, _ = acquisition.theta_moments(post, tmap=tmap)
    ui = acquisition.observation_unit_info(post, dsg, tmap, mu)
    assert ui.shape == (dsg.shape[0],)
    assert np.all(np.isfinite(ui)) and np.all(ui > 0)
    # counts carry per-cell information that varies with the cell mean
    assert len(np.unique(np.round(ui, 12))) > 1
    eig = cl.design_eig(post, dsg, n_geo=40, t_test=8)
    assert np.isfinite(eig) and eig > 0
    cells, meta = cl.select_next_design(
        post,
        center,
        world.pairs,
        B=3.2,
        value=5.0,
        candidate_deltas=(0.4, 0.7),
        n_geo=40,
        t_test=8,
        n_outcomes=8,
        seed=0,
    )
    assert meta["kg_used"] is True
    assert meta["sigma"] is None  # a count family has no Gaussian noise scale
    np.testing.assert_allclose(
        cells, cl.central_composite(center, meta["chosen_delta"], world.pairs)
    )


def test_acquisition_studentt_discounts_the_gaussian_information():
    """Student-t unit info is the Gaussian's times (nu+1)/(nu+3) < 1, so the
    same design buys strictly less EIG under heavy tails."""
    world = cl.make_world(seed=0)
    post_n = _fake_posterior(world)
    post_t = _fake_posterior(world)
    post_t.samples["nu"] = np.full(300, 5.0)
    post_t.likelihood = "studentt"
    dsg = cl.central_composite(np.full(4, 0.7), 0.6, world.pairs)
    tmap = acquisition.theta_map(post_n)
    mu, _ = acquisition.theta_moments(post_n, tmap=tmap)
    ui_n = acquisition.observation_unit_info(post_n, dsg, tmap, mu)
    ui_t = acquisition.observation_unit_info(post_t, dsg, tmap, mu)
    np.testing.assert_allclose(ui_t, ui_n * (5.0 + 1.0) / (5.0 + 3.0), rtol=1e-9)
    eig_n = cl.design_eig(post_n, dsg, n_geo=40, t_test=8)
    eig_t = cl.design_eig(post_t, dsg, n_geo=40, t_test=8)
    assert 0 < eig_t < eig_n


# ── slow: the three feasibility gates ─────────────────────────────────────────


@pytest.fixture(scope="module")
def recovered():
    """Fit a known world once; reused by the recovery assertions."""
    world = cl.make_world(seed=0)
    center = np.array([0.8, 0.8, 0.8, 0.8])
    data = cl.simulate_panel(
        world, center, n_geo=80, t_pre=6, t_test=10, delta=0.6, noise=0.5, seed=1
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
    return world, post


@pytest.mark.slow
def test_recovery_main_effects_and_synergy_signs(recovered):
    world, post = recovered
    beta_hat = post.samples["beta"].mean(0)
    # the strongest channel (Chatter) is recovered as strongest
    assert int(np.argmax(beta_hat)) == int(np.argmax(world.beta))
    # main-effect ordering tracks truth (rank correlation)
    from scipy.stats import spearmanr

    rho = spearmanr(beta_hat, world.beta).correlation
    assert rho >= 0.8
    # sign-informed synergies: cannibalization negative, complementarities positive
    gs = post.gamma_summary()
    assert gs["gamma_Chatter_Pulse"]["mean"] < 0.0  # neg pair
    assert gs["gamma_Pulse_Orbit"]["mean"] > 0.0  # pos pair
    assert gs["gamma_Orbit_Vibe"]["mean"] > 0.0  # pos pair
    # and a healthy fit
    assert post.diagnostics["max_rhat"] is None or post.diagnostics["max_rhat"] < 1.2


@pytest.mark.slow
def test_recovered_posterior_plans_a_sensible_funding_line(recovered):
    world, post = recovered
    rec = cl.recommend_allocation(post, B=3.2, value=5.0, q=200, mode="fixed")
    assert rec.sum() == pytest.approx(3.2, abs=0.05)
    _, prob_above, _ = cl.marginal_roas(post, rec, value=5.0, q=200)
    # the strongest channel is funded with high probability
    assert prob_above[int(np.argmax(world.beta))] > 0.5


@pytest.mark.slow
def test_prior_sensitivity_audit_gamma_scale():
    # A prior-dominated (weak, true ~0) synergy tracks its prior: widen the
    # gamma_scale and the posterior spread of that pair grows (guide §8.2).
    world = cl.make_world(seed=2)
    center = np.array([0.8, 0.8, 0.8, 0.8])
    data = cl.simulate_panel(
        world, center, n_geo=70, t_pre=5, t_test=8, delta=0.6, noise=0.5, seed=4
    )
    weak_pair = "gamma_Chatter_Vibe"  # true gamma == 0
    sds = {}
    for gs in (0.3, 1.5):
        post = cl.fit(
            data,
            channels=world.channels,
            pair_signs=cl.PAIR_SIGNS_EXAMPLE,
            gamma_scale=gs,
            num_warmup=300,
            num_samples=300,
            num_chains=2,
            seed=0,
        )
        sds[gs] = float(np.std(post.samples[weak_pair]))
    assert sds[1.5] > sds[0.3]


@pytest.mark.slow
def test_closure_and_stopping():
    world = cl.make_world(seed=0)
    center = np.array([0.7, 0.7, 0.7, 0.7])
    out = cl.run_closed_loop(
        world,
        center=center,
        B=3.2,
        value=5.0,
        n_geo=64,
        t_pre=5,
        t_test=8,
        delta=0.6,
        noise=0.5,
        mode="fixed",
        pair_signs=cl.PAIR_SIGNS_EXAMPLE,
        margin=1.0,
        population=2.0,
        wave_cost=0.3,
        max_waves=4,
        planner_q=120,
        fit_kwargs=dict(num_warmup=300, num_samples=300, num_chains=2, seed=0),
        seed=2,
    )
    hist = out["history"]
    assert len(hist) >= 2
    # closure: expected regret shrinks as the posterior is carried across waves
    assert hist[-1]["e_regret"] < hist[0]["e_regret"]
    # stopping: ENBS decreases and the rule fires before max_waves
    assert hist[-1]["enbs"] < hist[0]["enbs"]
    assert any(r["stop"] for r in hist)
    # recovery: the final recommendation is close to the truth-optimal profit
    assert hist[-1]["profit_gap_rel"] < 0.1


@pytest.mark.slow
def test_closure_with_laplace_kg():
    """The closure invariants hold when the Laplace KG picks each wave's design
    (use_laplace_kg=True): E[regret] shrinks, the ENBS rule fires (or the loop
    exhausts max_waves), and the final plan stays near the truth optimum."""
    world = cl.make_world(seed=0)
    center = np.array([0.7, 0.7, 0.7, 0.7])
    out = cl.run_closed_loop(
        world,
        center=center,
        B=3.2,
        value=5.0,
        n_geo=64,
        t_pre=5,
        t_test=8,
        delta=0.6,
        noise=0.5,
        mode="fixed",
        pair_signs=cl.PAIR_SIGNS_EXAMPLE,
        margin=1.0,
        population=2.0,
        wave_cost=0.3,
        max_waves=4,
        planner_q=120,
        fit_kwargs=dict(num_warmup=300, num_samples=300, num_chains=2, seed=0),
        use_laplace_kg=True,
        candidate_deltas=(0.4, 0.6, 0.8),
        kg_n_outcomes=32,
        seed=2,
    )
    hist = out["history"]
    assert len(hist) >= 2
    assert hist[-1]["e_regret"] < hist[0]["e_regret"]
    assert any(r["stop"] for r in hist) or len(hist) == 4
    assert hist[-1]["profit_gap_rel"] < 0.15
    # the KG actually chose the later designs (Hill/Gaussian -> never falls back)
    assert all(r["kg_used"] for r in hist[1:])
    assert all(r["chosen_delta"] in (0.4, 0.6, 0.8) for r in hist[1:])
    assert hist[0]["kg_used"] is False  # wave 0 is the fixed initial CCD


@pytest.mark.slow
def test_knowledge_gradient_runs_and_is_finite():
    world = cl.make_world(seed=1)
    center = np.array([0.8, 0.8, 0.8, 0.8])
    data = cl.simulate_panel(
        world, center, n_geo=48, t_pre=4, t_test=6, delta=0.6, noise=0.5, seed=3
    )
    post = cl.fit(
        data,
        channels=world.channels,
        pair_signs=cl.PAIR_SIGNS_EXAMPLE,
        num_warmup=150,
        num_samples=150,
        num_chains=1,
        seed=0,
    )
    refit = cl.refit_fn_from_data(
        data,
        channels=world.channels,
        pair_signs=cl.PAIR_SIGNS_EXAMPLE,
        num_warmup=100,
        num_samples=100,
        num_chains=1,
        seed=5,
    )
    candidate = cl.central_composite(center, 0.6, world.pairs)
    kg = cl.knowledge_gradient(
        post,
        candidate,
        refit,
        B=3.2,
        value=5.0,
        n_fantasy=2,
        t_test=6,
        n_geo=48,
        q=40,
        seed=4,
    )
    assert np.isfinite(kg)


@pytest.mark.slow
def test_logistic_activation_recovers_and_plans():
    """The whole loop is activation-agnostic: fit a logistic (exponential-
    saturation) world and recover the effects + plan, exactly as for Hill."""
    world = cl.make_world_logistic(seed=0)
    center = np.array([0.8, 0.8, 0.8, 0.8])
    data = cl.simulate_panel(
        world, center, n_geo=72, t_pre=6, t_test=10, delta=0.6, noise=0.5, seed=1
    )
    post = cl.fit(
        data,
        channels=world.channels,
        pair_signs=cl.PAIR_SIGNS_EXAMPLE,
        activation="logistic",
        num_warmup=400,
        num_samples=400,
        num_chains=2,
        seed=0,
    )
    assert post.activation == "logistic"
    assert "lam" in post.samples and "kappa" not in post.samples  # its own params
    beta_hat = post.samples["beta"].mean(0)
    # strongest channel + top-2 recovered (as in the Hill recovery test)
    assert int(np.argmax(beta_hat)) == int(np.argmax(world.beta))
    assert set(np.argsort(-beta_hat)[:2]) == set(np.argsort(-world.beta)[:2])
    # the activation-agnostic planner produces a valid funded split
    rec = cl.recommend_allocation(post, B=3.2, value=5.0, q=200, mode="fixed")
    assert rec.sum() == pytest.approx(3.2, abs=0.05)
    _, prob_above, _ = cl.marginal_roas(post, rec, value=5.0, q=200)
    assert prob_above[int(np.argmax(world.beta))] > 0.5


@pytest.mark.slow
def test_misspecified_single_hill_still_makes_a_near_optimal_decision():
    """Headline of the misspecification study: when the TRUE response is a
    weighted sum of Hills but we fit a single Hill, the *decision* barely
    suffers — the recommended allocation still captures the vast majority of the
    true optimum's profit, because near an interior optimum the profit surface is
    flat and a wrong-but-monotone-saturating curve gets the local marginal
    ordering right. (Calibration is what degrades, not portfolio profit.)"""
    world = cl.make_world_hill_mixture(seed=0)
    B, value = 3.2, 5.0
    _, true_profit = cl.world_optimal_allocation(world, B, value, mode="fixed")
    data = cl.simulate_panel(
        world,
        np.full(4, 0.7),
        n_geo=72,
        t_pre=6,
        t_test=10,
        delta=0.6,
        noise=0.4,
        seed=1,
    )
    post = cl.fit(
        data,
        channels=world.channels,
        pair_signs=cl.PAIR_SIGNS_EXAMPLE,
        activation="hill",
        num_warmup=400,
        num_samples=400,
        num_chains=2,
        seed=0,
    )
    rec = cl.recommend_allocation(post, B, value, q=200, mode="fixed")
    achieved = value * float(world.response_mean(np.asarray(rec)[None, :])[0]) - B
    # the misspecified plan captures >=95% of the achievable profit
    assert achieved >= 0.95 * true_profit


@pytest.mark.slow
def test_negbinomial_recovery_counts_world():
    """NB feasibility gate: designed count data (gamma-Poisson DGP, counts in
    the hundreds) fit with ``likelihood='negbinomial'`` — the beta ordering is
    recovered and the chains mix, mirroring the Gaussian recovery gate."""
    channels = ["Chatter", "Pulse", "Orbit", "Vibe"]
    world = cl.TrueWorld(
        beta=np.array([150.0, 100.0, 60.0, 40.0]),
        kappa=np.array([0.8, 0.6, 1.0, 0.7]),
        alpha=np.array([2.2, 1.6, 2.6, 1.3]),
        gamma_pairs=np.zeros(6),
        channels=channels,
        a_level=250.0,
        sigma_a=25.0,
        phi_true=25.0,
    )
    center = np.array([0.8, 0.8, 0.8, 0.8])
    data = cl.simulate_panel(
        world,
        center,
        n_geo=60,
        t_pre=4,
        t_test=6,
        delta=0.6,
        noise_family="negbinomial",
        seed=1,
    )
    assert np.all(data["y"] >= 0)
    np.testing.assert_allclose(data["y"], np.round(data["y"]))
    post = cl.fit(
        data,
        channels=channels,
        pairs=[],  # main effects only: keep the count gate small
        likelihood="negbinomial",
        beta_scale=2.0,
        # the scale-free LogNormal(log 30, 2) dispersion prior (finding [4])
        # mixes phi more slowly than the old exponential-tailed Gamma —
        # a longer adaptation keeps the R-hat gate honest without touching
        # the prior back.
        num_warmup=800,
        num_samples=600,
        num_chains=2,
        seed=0,
    )
    assert post.likelihood == "negbinomial"
    assert "phi" in post.samples and "sigma" not in post.samples
    beta_hat = post.samples["beta"].mean(0)
    assert int(np.argmax(beta_hat)) == int(np.argmax(world.beta))
    from scipy.stats import spearmanr

    rho = spearmanr(beta_hat, world.beta).correlation
    assert rho >= 0.8
    assert post.diagnostics["max_rhat"] is None or post.diagnostics["max_rhat"] < 1.2


@pytest.mark.slow
def test_studentt_recovery_heavy_tailed_world():
    """Student-t feasibility gate: a heavy-tailed world (t(3) residuals, so a
    few geo-weeks land far off the surface) fit with ``likelihood='studentt'``
    — beta ordering recovered, the tail df concentrates well below Gaussian,
    and the chains mix. Mirrors the NB count gate."""
    world = cl.make_world(seed=0)
    world.nu_true = 3.0
    center = np.array([0.8, 0.8, 0.8, 0.8])
    data = cl.simulate_panel(
        world,
        center,
        n_geo=60,
        t_pre=4,
        t_test=6,
        delta=0.6,
        noise=0.5,
        noise_family="studentt",
        seed=1,
    )
    post = cl.fit(
        data,
        channels=world.channels,
        pairs=[],  # main effects only: keep the gate small
        likelihood="studentt",
        num_warmup=500,
        num_samples=500,
        num_chains=2,
        seed=0,
    )
    assert post.likelihood == "studentt"
    assert "nu" in post.samples and "sigma" in post.samples
    beta_hat = post.samples["beta"].mean(0)
    assert int(np.argmax(beta_hat)) == int(np.argmax(world.beta))
    from scipy.stats import spearmanr

    rho = spearmanr(beta_hat, world.beta).correlation
    assert rho >= 0.8
    # planted t(3) residuals: the posterior tail df should sit well below the
    # near-Gaussian prior mass (Gamma(2, 0.1) has mean 20)
    assert float(np.median(post.samples["nu"])) < 10.0
    assert post.diagnostics["max_rhat"] is None or post.diagnostics["max_rhat"] < 1.2


@pytest.mark.slow
def test_national_time_effect_recovery():
    """tau gate: planted national per-period shocks are recovered
    (corr(tau_hat, tau_true) > 0.7) while the beta recovery stays intact."""
    world = cl.make_world(seed=0)
    center = np.array([0.8, 0.8, 0.8, 0.8])
    data = cl.simulate_panel(
        world,
        center,
        n_geo=60,
        t_pre=4,
        t_test=6,
        delta=0.6,
        noise=0.5,
        tau_scale=1.0,
        seed=3,
    )
    post = cl.fit(
        data,
        channels=world.channels,
        pair_signs=cl.PAIR_SIGNS_EXAMPLE,
        time_effect="national",
        num_warmup=400,
        num_samples=400,
        num_chains=2,
        seed=0,
    )
    assert post.time_effect == "national"
    tau_hat = post.samples["tau"].mean(0)
    assert tau_hat.shape == (10,)  # t_pre + t_test national shocks
    corr = float(np.corrcoef(tau_hat, np.asarray(data["tau_true"]))[0, 1])
    assert corr > 0.7
    beta_hat = post.samples["beta"].mean(0)
    assert int(np.argmax(beta_hat)) == int(np.argmax(world.beta))
    assert post.diagnostics["max_rhat"] is None or post.diagnostics["max_rhat"] < 1.2
