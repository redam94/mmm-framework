"""Continuous sequential learning — a model-free geo response-surface bandit.

A self-contained Bayesian sequential-experimentation loop that allocates
continuous budget across channels by repeatedly (1) fitting a response surface
**directly from designed experiment data**, (2) choosing the most valuable next
experiment, and (3) stopping when further testing no longer pays — *without
requiring a pre-fit MMM*. The experiment's designed cross-sectional variation
identifies the surface, so the priors inform but the data dominates.

This complements (does not replace) the model-anchored planning layer in
:mod:`mmm_framework.planning`, which sits on top of a fitted
:class:`~mmm_framework.model.BayesianMMM`. Here there is no observational model:
the loop learns from experiments alone.

Layers
------
* :mod:`.surface` — the differentiable JAX response surface (single source of
  truth, shared by the likelihood, the DGP, and the allocator).
* :mod:`.model` — the NumPyro generative model + priors + the :class:`Posterior`
  container and :func:`fit`.
* :mod:`.design` — central-composite geo cells (:func:`central_composite`) and
  geo assignment (:func:`assign_geos`).
* :mod:`.dgp` — a synthetic :class:`TrueWorld` with causal ground truth, the
  recovery harness :func:`simulate_panel`, and the fantasy engine.
* :mod:`.planner` — the allocator (:func:`allocate_under_sample`,
  :func:`thompson_wave`), the funding line (:func:`marginal_roas`), the stopping
  rule (:func:`expected_regret`, :func:`enbs`), and decision-aware EVSI
  (:func:`knowledge_gradient`).
* :mod:`.loop` — :class:`LearningState` (carry the posterior across waves) and
  :func:`run_closed_loop` (the end-to-end demo / closure test).

The Hill activation matches ``SaturationType.HILL`` (``slope = alpha``,
``sat_half = kappa``), so a continuous-learning posterior is directly comparable
to a :class:`~mmm_framework.model.BayesianMMM` Hill fit on the same channel.

See ``technical-docs/continuous-learning.md`` and ``assets/continous_learning.md``.
"""

from __future__ import annotations

from .arms import (
    ARM_SEP,
    ArmSpec,
    arm_shares,
    cross_parent_pairs,
    default_arm_pair_signs,
    expand_arms,
    within_parent_pairs,
)
from .design import assign_geos, central_composite
from .dgp import (
    TrueWorld,
    make_world,
    make_world_hill_mixture,
    make_world_logistic,
    simulate_panel,
    simulate_wave,
)
from .evidence import experiments_to_summaries
from .loop import (
    LearningState,
    WaveRecord,
    due_for_retest,
    run_closed_loop,
    select_next_design,
    world_optimal_allocation,
)
from .model import (
    PAIR_SIGNS_EXAMPLE,
    Posterior,
    default_pairs,
    demote_channel,
    fit,
    pair_name,
    probe_pairs_excluding,
    refit_fn_from_data,
)
from .planner import (
    PlanResult,
    allocate_under_sample,
    enbs,
    expected_regret,
    knowledge_gradient,
    marginal_roas,
    plan_from_posterior,
    posterior_optimal_allocation,
    recommend_allocation,
    response_grid,
    should_stop,
    thompson_wave,
)
from .preprocess import adstock_panel, adstock_prepass, cuped_adjust, cuped_covariate
from .acquisition import (
    design_eig,
    design_information,
    gaussian_eig,
    laplace_knowledge_gradient,
    observation_unit_info,
    theta_map,
    theta_moments,
)
from .scaling import to_dollars, to_scaled
from .serialize import (
    posterior_from_payload,
    posterior_to_payload,
    state_from_npz,
    state_to_npz,
)
from .surface import (
    ACTIVATIONS,
    MSPLINE_J,
    MSPLINE_S_MAX,
    activation,
    hill_mixture,
    incremental,
    logistic,
    monotone_spline,
    monotone_spline_basis,
    response_curve,
    surface_over_rows,
    surface_value,
)

__all__ = [
    # surface (pluggable activations)
    "activation",
    "logistic",
    "hill_mixture",
    "monotone_spline",
    "monotone_spline_basis",
    "MSPLINE_J",
    "MSPLINE_S_MAX",
    "incremental",
    "response_curve",
    "surface_value",
    "surface_over_rows",
    "ACTIVATIONS",
    # model
    "fit",
    "Posterior",
    "default_pairs",
    "pair_name",
    "demote_channel",
    "probe_pairs_excluding",
    "refit_fn_from_data",
    "PAIR_SIGNS_EXAMPLE",
    # design
    "central_composite",
    "assign_geos",
    # dgp
    "TrueWorld",
    "make_world",
    "make_world_logistic",
    "make_world_hill_mixture",
    "simulate_panel",
    "simulate_wave",
    # planner
    "allocate_under_sample",
    "thompson_wave",
    "recommend_allocation",
    "marginal_roas",
    "expected_regret",
    "plan_from_posterior",
    "PlanResult",
    "posterior_optimal_allocation",
    "knowledge_gradient",
    "response_grid",
    "enbs",
    "should_stop",
    # preprocess (adstock pre-pass + CUPED)
    "adstock_panel",
    "adstock_prepass",
    "cuped_covariate",
    "cuped_adjust",
    # acquisition (fast Laplace KG + pure EIG)
    "laplace_knowledge_gradient",
    "design_eig",
    "design_information",
    "gaussian_eig",
    "observation_unit_info",
    "theta_map",
    "theta_moments",
    # scaling (dollars <-> scaled units)
    "to_scaled",
    "to_dollars",
    # evidence (experiment registry -> summary observations)
    "experiments_to_summaries",
    # serialize (Posterior payloads + LearningState .npz persistence)
    "posterior_to_payload",
    "posterior_from_payload",
    "state_to_npz",
    "state_from_npz",
    # arms (sub-channel measurement)
    "ARM_SEP",
    "ArmSpec",
    "arm_shares",
    "expand_arms",
    "within_parent_pairs",
    "cross_parent_pairs",
    "default_arm_pair_signs",
    # loop
    "LearningState",
    "WaveRecord",
    "run_closed_loop",
    "select_next_design",
    "world_optimal_allocation",
    "due_for_retest",
]
