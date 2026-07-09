"""Spec-settable extension priors + trend / seasonality / likelihood wiring.

Covers the two deferred features now shipped:
  (A) NestedMMM / MultivariateMMM / CombinedMMM honor the spec's trend,
      seasonality, and (Nested) outcome-likelihood family, and
  (B) mediator / outcome / cross-effect priors are settable through the spec
      (`priors.mediator|outcome|cross_effect.*`) and actually reach the graph,
      with the consumed-paths registry gating them.

Invariant throughout: a model built WITHOUT a model_config/trend_config (the
array-level path) is byte-identical to before — new components/priors activate
only from an explicit config.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mmm_framework.mmm_extensions.config import (
    CombinedModelConfig,
    CrossEffectConfig,
    MediatorConfig,
    MediatorType,
    MultivariateModelConfig,
    NestedModelConfig,
    OutcomeConfig,
)
from mmm_framework.mmm_extensions.models.combined import CombinedMMM
from mmm_framework.mmm_extensions.models.multivariate import MultivariateMMM
from mmm_framework.mmm_extensions.models.nested import NestedMMM
from mmm_framework.config import LikelihoodConfig, ModelConfig, SeasonalityConfig
from mmm_framework.model import TrendConfig, TrendType


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def idx():
    return pd.date_range("2022-01-03", periods=60, freq="W-MON")


@pytest.fixture
def media():
    rng = np.random.default_rng(0)
    return np.abs(rng.normal(100, 20, (60, 2)))


@pytest.fixture
def nested_cfg():
    return NestedModelConfig(
        mediators=(
            MediatorConfig(name="Awareness", mediator_type=MediatorType.FULLY_LATENT),
        )
    )


def _y(media):
    rng = np.random.default_rng(1)
    aware = 40 + 0.3 * media[:, 0] + rng.normal(0, 4, 60)
    return 1000 + 4 * aware + 2 * media[:, 1] + rng.normal(0, 40, 60)


def _rv_names(model) -> set:
    return set(model.model.named_vars.keys())


# --------------------------------------------------------------------------- #
# (A) trend / seasonality / likelihood — NestedMMM
# --------------------------------------------------------------------------- #
def test_nested_baseline_is_byte_identical(media, idx, nested_cfg):
    """No model_config/trend_config → no trend/season RVs, Normal outcome."""
    m = NestedMMM(media, _y(media), ["TV", "Digital"], nested_cfg, index=idx)
    rvs = _rv_names(m)
    assert "sigma_y" in rvs
    assert not any(("trend" in k or "seasonality" in k or k == "nu_y") for k in rvs)


def test_nested_linear_trend_adds_rv(media, idx, nested_cfg):
    m = NestedMMM(
        media,
        _y(media),
        ["TV", "Digital"],
        nested_cfg,
        index=idx,
        trend_config=TrendConfig(type=TrendType.LINEAR),
    )
    rvs = _rv_names(m)
    assert "trend_slope" in rvs and "trend_component" in rvs


def test_nested_seasonality_adds_rv(media, idx, nested_cfg):
    mc = ModelConfig()
    mc.seasonality = SeasonalityConfig(yearly=2, monthly=0, weekly=0)
    m = NestedMMM(
        media, _y(media), ["TV", "Digital"], nested_cfg, index=idx, model_config=mc
    )
    rvs = _rv_names(m)
    assert "seasonality_coefs" in rvs and "seasonality_component" in rvs


def test_nested_student_t_likelihood(media, idx, nested_cfg):
    mc = ModelConfig()
    mc.likelihood = LikelihoodConfig(family="student_t")
    m = NestedMMM(
        media, _y(media), ["TV", "Digital"], nested_cfg, index=idx, model_config=mc
    )
    rvs = _rv_names(m)
    assert "nu_y" in rvs and "sigma_y" in rvs


def test_nested_unsupported_likelihood_raises(media, idx, nested_cfg):
    mc = ModelConfig()
    mc.likelihood = LikelihoodConfig(family="poisson")
    m = NestedMMM(
        media, _y(media), ["TV", "Digital"], nested_cfg, index=idx, model_config=mc
    )
    with pytest.raises(NotImplementedError, match="poisson|not supported"):
        _ = m.model


def test_nested_seasonality_no_datetime_index_skips(media, nested_cfg):
    """A RangeIndex can't fix a period — seasonality is skipped with a warning."""
    mc = ModelConfig()
    mc.seasonality = SeasonalityConfig(yearly=2)
    # no index → RangeIndex; model_config carries the seasonality request
    m = NestedMMM(media, _y(media), ["TV", "Digital"], nested_cfg, model_config=mc)
    with pytest.warns(UserWarning, match="no datetime index"):
        rvs = _rv_names(m)
    assert "seasonality_coefs" not in rvs


# --------------------------------------------------------------------------- #
# (B) spec-settable priors reach the graph — NestedMMM
# --------------------------------------------------------------------------- #
def _prior_std(model, var, draws=400):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idata = model.sample_prior_predictive(samples=draws, random_seed=0)
    return float(idata.prior[var].std())


def test_mediator_media_effect_prior_scales_the_rv(media, idx):
    """A larger media_effect sigma widens the media->mediator prior."""
    tight = NestedModelConfig(
        mediators=(
            MediatorConfig(
                name="Awareness",
                mediator_type=MediatorType.FULLY_LATENT,
                media_effect=__import__(
                    "mmm_framework.mmm_extensions.config",
                    fromlist=["EffectPriorConfig"],
                ).EffectPriorConfig(constraint=_pos(), sigma=0.3),
            ),
        )
    )
    wide = NestedModelConfig(
        mediators=(
            MediatorConfig(
                name="Awareness",
                mediator_type=MediatorType.FULLY_LATENT,
                media_effect=__import__(
                    "mmm_framework.mmm_extensions.config",
                    fromlist=["EffectPriorConfig"],
                ).EffectPriorConfig(constraint=_pos(), sigma=3.0),
            ),
        )
    )
    y = _y(media)
    s_tight = _prior_std(
        NestedMMM(media, y, ["TV", "Digital"], tight, index=idx), "beta_TV_to_Awareness"
    )
    s_wide = _prior_std(
        NestedMMM(media, y, ["TV", "Digital"], wide, index=idx), "beta_TV_to_Awareness"
    )
    assert s_wide > 3 * s_tight  # sigma 3.0 vs 0.3 → ~10x wider prior


def _pos():
    from mmm_framework.mmm_extensions.config import EffectConstraint

    return EffectConstraint.POSITIVE


def test_direct_effect_config_now_reaches_the_graph(media, idx):
    """The direct-effect prior sigma was hard-coded (dead config); it now scales
    the delta_direct RV."""
    from mmm_framework.mmm_extensions.config import EffectPriorConfig

    def cfg(sigma):
        return NestedModelConfig(
            mediators=(
                MediatorConfig(
                    name="Awareness",
                    mediator_type=MediatorType.FULLY_LATENT,
                    allow_direct_effect=True,
                    direct_effect=EffectPriorConfig(sigma=sigma),
                ),
            )
        )

    y = _y(media)
    s_narrow = _prior_std(
        NestedMMM(media, y, ["TV", "Digital"], cfg(0.1), index=idx), "delta_direct_TV"
    )
    s_wide = _prior_std(
        NestedMMM(media, y, ["TV", "Digital"], cfg(2.0), index=idx), "delta_direct_TV"
    )
    assert s_wide > 5 * s_narrow


# --------------------------------------------------------------------------- #
# (B) spec bridge: priors.* → DAG node config → MediatorConfig
# --------------------------------------------------------------------------- #
def test_spec_injection_overrides_mediator_config():
    from mmm_framework.agents.fitting import _inject_extension_priors
    from mmm_framework.dag_model_builder.config_translator import dag_to_nested_config
    from mmm_framework.dag_model_builder.dag_spec import DAGSpec
    from mmm_framework.dag_model_builder.frontend_adapter import create_mediation_dag

    dag = create_mediation_dag(
        kpi_name="Sales", media_names=["TV", "Digital"], mediator_name="Awareness"
    )
    spec = {
        "dag_model_type": "nested_mmm",
        "priors": {
            "mediator": {
                "Awareness": {
                    "media_effect_sigma": 3.0,
                    "outcome_effect_sigma": 2.5,
                    "direct_effect_sigma": 0.9,
                    "media_effect_constraint": "positive",
                }
            }
        },
    }
    ncfg = dag_to_nested_config(
        DAGSpec.model_validate(_inject_extension_priors(dag.model_dump(), spec))
    )
    med = ncfg.mediators[0]
    assert med.media_effect.sigma == 3.0
    assert med.outcome_effect.sigma == 2.5
    assert med.direct_effect.sigma == 0.9
    # default (no override) is unchanged
    d0 = dag_to_nested_config(DAGSpec.model_validate(dag.model_dump())).mediators[0]
    assert (d0.media_effect.sigma, d0.outcome_effect.sigma, d0.direct_effect.sigma) == (
        1.0,
        1.0,
        0.5,
    )


# --------------------------------------------------------------------------- #
# (B) registry gate
# --------------------------------------------------------------------------- #
@pytest.fixture
def ext_spec():
    return {
        "dag_model_type": "nested_mmm",
        "media_channels": [{"name": "TV"}, {"name": "Digital"}],
        "control_variables": [],
    }


def test_registry_accepts_mediator_prior(ext_spec):
    from mmm_framework.agents.fitting import unconsumed_prior_path

    assert (
        unconsumed_prior_path(
            ["priors", "mediator", "Awareness", "media_effect_sigma"], 3.0, ext_spec
        )
        is None
    )


def test_registry_accepts_seasonality_for_extension(ext_spec):
    from mmm_framework.agents.fitting import unconsumed_prior_path

    assert (
        unconsumed_prior_path(["priors", "seasonality", "prior_sigma"], 0.5, ext_spec)
        is None
    )


def test_registry_rejects_plain_media_prior_for_extension(ext_spec):
    from mmm_framework.agents.fitting import unconsumed_prior_path

    err = unconsumed_prior_path(["priors", "media", "TV", "coefficient"], {}, ext_spec)
    assert err and "won't apply" in err and "nested_mmm" in err


def test_registry_rejects_unknown_mediator_key(ext_spec):
    from mmm_framework.agents.fitting import unconsumed_prior_path

    err = unconsumed_prior_path(
        ["priors", "mediator", "Awareness", "bogus"], 1.0, ext_spec
    )
    assert err and "bogus" in err


def test_registry_still_accepts_plain_prior_on_non_extension():
    from mmm_framework.agents.fitting import unconsumed_prior_path

    plain = {"media_channels": [{"name": "TV"}], "control_variables": []}
    assert (
        unconsumed_prior_path(
            ["priors", "media", "TV", "roi"], {"median": 2.0, "sigma": 0.5}, plain
        )
        is None
    )


# --------------------------------------------------------------------------- #
# (A) MV / Combined — trend/seasonality + per-outcome priors live
# --------------------------------------------------------------------------- #
def _outcome_data(media):
    rng = np.random.default_rng(2)
    return {
        "A": 1000 + 2 * media[:, 0] + rng.normal(0, 30, 60),
        "B": 800 + 1.5 * media[:, 1] + rng.normal(0, 25, 60),
    }


def test_mv_baseline_byte_identical(media, idx):
    cfg = MultivariateModelConfig(
        outcomes=(
            OutcomeConfig(name="A", column="A"),
            OutcomeConfig(name="B", column="B"),
        )
    )
    m = MultivariateMMM(media, _outcome_data(media), ["TV", "Digital"], cfg, index=idx)
    rvs = _rv_names(m)
    assert {"alpha", "beta_media"} <= rvs
    assert not any(("trend" in k or "seasonality" in k) for k in rvs)


def test_mv_trend_and_seasonality(media, idx):
    cfg = MultivariateModelConfig(
        outcomes=(
            OutcomeConfig(name="A", column="A"),
            OutcomeConfig(name="B", column="B"),
        )
    )
    mc = ModelConfig()
    mc.seasonality = SeasonalityConfig(yearly=2)
    m = MultivariateMMM(
        media,
        _outcome_data(media),
        ["TV", "Digital"],
        cfg,
        index=idx,
        model_config=mc,
        trend_config=TrendConfig(type=TrendType.LINEAR),
    )
    rvs = _rv_names(m)
    # share_trend defaults False -> per-outcome; share_seasonality True -> shared
    assert "A_trend_slope" in rvs and "B_trend_slope" in rvs
    assert "seasonality_coefs" in rvs


def test_mv_per_outcome_intercept_sigma_live(media, idx):
    """OutcomeConfig.intercept_prior_sigma now scales the intercept prior."""
    cfg = MultivariateModelConfig(
        outcomes=(
            OutcomeConfig(name="A", column="A", intercept_prior_sigma=0.1),
            OutcomeConfig(name="B", column="B", intercept_prior_sigma=5.0),
        )
    )
    m = MultivariateMMM(media, _outcome_data(media), ["TV", "Digital"], cfg, index=idx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idata = m.sample_prior_predictive(samples=400, random_seed=0)
    alpha = idata.prior["alpha"]
    std_a = float(alpha.sel(outcome="A").std())
    std_b = float(alpha.sel(outcome="B").std())
    assert std_b > 5 * std_a  # 5.0 vs 0.1


@pytest.mark.parametrize(
    "ttype,expected",
    [
        (TrendType.LINEAR, {"trend_slope"}),
        # piecewise has NO trend_m: the scalar offset would be an unidentified
        # dead parameter once the trend is zero-centered (level -> intercept).
        (TrendType.PIECEWISE, {"trend_k", "trend_delta"}),
        (TrendType.SPLINE, {"spline_coef_raw", "spline_scale"}),
        (TrendType.GP, {"gp_lengthscale", "gp_amplitude"}),
    ],
)
def test_nested_trend_types_build_their_rvs(media, idx, nested_cfg, ttype, expected):
    """All four trend families (not just linear) build in an extension graph."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = NestedMMM(
            media,
            _y(media),
            ["TV", "Digital"],
            nested_cfg,
            index=idx,
            trend_config=TrendConfig(type=ttype),
        )
        rvs = _rv_names(m)
    assert expected <= rvs
    assert "trend_component" in rvs  # original-unit deterministic


def test_piecewise_trend_has_no_dead_offset(media, idx, nested_cfg):
    """The Prophet scalar offset was dropped — zero-centering makes it a fully
    unidentified (dead) parameter, so it must not be in the graph at all."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = NestedMMM(
            media,
            _y(media),
            ["TV", "Digital"],
            nested_cfg,
            index=idx,
            trend_config=TrendConfig(type=TrendType.PIECEWISE),
        )
        rvs = _rv_names(m)
    assert "trend_m" not in rvs


@pytest.mark.parametrize("ttype", [TrendType.PIECEWISE, TrendType.SPLINE, TrendType.GP])
def test_nonlinear_trends_have_no_parameter_independent_deterministics(
    media, idx, nested_cfg, ttype
):
    """Every trend free RV is a graph ancestor of the observed RV (no
    parameter-independent Deterministic → Pathfinder ``_anchored_det`` safe).

    NB this graph-ancestry check is the Pathfinder-relevant property; it cannot
    see an *algebraically* cancelled direction (e.g. the RW-spline's constant
    shift ``spline_coef_raw[0]``, a benign unidentified direction the extension
    inherits from the core spline — it samples from its prior and never touches
    mu/ROI)."""
    from pytensor.graph.traversal import ancestors

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = NestedMMM(
            media,
            _y(media),
            ["TV", "Digital"],
            nested_cfg,
            index=idx,
            trend_config=TrendConfig(type=ttype),
        )
        model = m.model
    reachable = set(ancestors(model.observed_RVs))
    dead = {rv.name for rv in model.free_RVs if rv not in reachable}
    assert dead == set()


def test_gp_trend_warns_about_map_instability(media, idx, nested_cfg):
    with pytest.warns(UserWarning, match="HSGP|NUTS|weakly identified"):
        _rv_names(
            NestedMMM(
                media,
                _y(media),
                ["TV", "Digital"],
                nested_cfg,
                index=idx,
                trend_config=TrendConfig(type=TrendType.GP),
            )
        )


def _mv_dag_with_cross_effect():
    from mmm_framework.dag_model_builder.dag_spec import (
        DAGEdge,
        DAGNode,
        DAGSpec,
        EdgeType,
        NodeType,
    )

    nodes = [
        DAGNode(id="o1", variable_name="revenue", node_type=NodeType.OUTCOME),
        DAGNode(id="o2", variable_name="volume", node_type=NodeType.OUTCOME),
        DAGNode(id="m1", variable_name="marketing", node_type=NodeType.MEDIA),
    ]
    edges = [
        DAGEdge(source="m1", target="o1", edge_type=EdgeType.DIRECT),
        DAGEdge(source="m1", target="o2", edge_type=EdgeType.DIRECT),
        DAGEdge(source="o1", target="o2", edge_type=EdgeType.CROSS_EFFECT),
    ]
    return DAGSpec(nodes=nodes, edges=edges)


def test_cross_effect_prior_reaches_config_and_does_not_crash():
    """priors.cross_effect.<src>__<tgt> is folded into edge.metadata (a real
    DAGEdge field — validation must not crash) and reaches the CrossEffectConfig."""
    from mmm_framework.agents.fitting import _inject_extension_priors
    from mmm_framework.dag_model_builder.config_translator import (
        dag_to_multivariate_config,
    )
    from mmm_framework.dag_model_builder.dag_spec import DAGSpec

    dag = _mv_dag_with_cross_effect()
    spec = {
        "dag_model_type": "multivariate_mmm",
        "priors": {
            "cross_effect": {
                "revenue__volume": {"effect_type": "halo", "prior_sigma": 0.7}
            }
        },
    }
    ds = DAGSpec.model_validate(_inject_extension_priors(dag.model_dump(), spec))
    ce = [
        c
        for c in dag_to_multivariate_config(ds).cross_effects
        if c.target_outcome == "volume"
    ][0]
    assert ce.effect_type.value == "halo"
    assert ce.prior_sigma == 0.7
    # the caller's DAG is untouched (deepcopy), and the default is unchanged
    assert dag.edges[2].metadata == {}
    ce0 = [
        c
        for c in dag_to_multivariate_config(dag).cross_effects
        if c.target_outcome == "volume"
    ][0]
    assert ce0.effect_type.value == "cannibalization" and ce0.prior_sigma == 0.3


def test_registry_rejects_typoed_mediator_name():
    from mmm_framework.agents.fitting import unconsumed_prior_path
    from mmm_framework.dag_model_builder.frontend_adapter import create_mediation_dag

    dag = create_mediation_dag(
        kpi_name="Sales", media_names=["TV", "Digital"], mediator_name="Awareness"
    )
    spec = {"dag_model_type": "nested_mmm", "dag_spec": dag.model_dump()}
    # correct name accepted; a typo is rejected (not accept-then-no-op)
    assert (
        unconsumed_prior_path(
            ["priors", "mediator", "Awareness", "media_effect_sigma"], 2.0, spec
        )
        is None
    )
    err = unconsumed_prior_path(
        ["priors", "mediator", "Awarenesss", "media_effect_sigma"], 2.0, spec
    )
    assert err and "unknown mediator" in err


def test_mv_per_outcome_spline_trend(media, idx):
    """Non-linear trends work per-outcome in the multi-outcome models too."""
    cfg = MultivariateModelConfig(
        outcomes=(
            OutcomeConfig(name="A", column="A"),
            OutcomeConfig(name="B", column="B"),
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MultivariateMMM(
            media,
            _outcome_data(media),
            ["TV", "Digital"],
            cfg,
            index=idx,
            trend_config=TrendConfig(type=TrendType.SPLINE),
        )
        rvs = _rv_names(m)
    # share_trend defaults False → per-outcome spline coefficients
    assert "A_spline_coef_raw" in rvs and "B_spline_coef_raw" in rvs


def test_combined_builds_with_trend_seasonality(media, idx):
    ncfg = NestedModelConfig(
        mediators=(
            MediatorConfig(name="Awareness", mediator_type=MediatorType.FULLY_LATENT),
        )
    )
    mvcfg = MultivariateModelConfig(
        outcomes=(
            OutcomeConfig(name="A", column="A"),
            OutcomeConfig(name="B", column="B"),
        )
    )
    cfg = CombinedModelConfig(nested=ncfg, multivariate=mvcfg)
    mc = ModelConfig()
    mc.seasonality = SeasonalityConfig(yearly=2)
    m = CombinedMMM(
        media,
        _outcome_data(media),
        ["TV", "Digital"],
        cfg,
        index=idx,
        model_config=mc,
        trend_config=TrendConfig(type=TrendType.LINEAR),
    )
    rvs = _rv_names(m)
    assert {"alpha_y", "beta_direct", "gamma"} <= rvs
    assert "seasonality_coefs" in rvs
