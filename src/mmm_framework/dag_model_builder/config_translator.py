"""
Config Translator

Translates DAG specifications to framework configuration objects.
"""

from __future__ import annotations

import warnings

from mmm_framework.config import (
    AdstockConfig,
    CausalControlRole,
    ControlVariableConfig,
    DimensionType,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
    PriorConfig,
    PriorType,
    SaturationConfig,
)

from .dag_spec import DAGSpec, NodeType
from .identification import DagRoleClassification, classify_dag_roles
from .node_configs import (
    ControlNodeConfig,
    KPINodeConfig,
    MediaNodeConfig,
    MediatorNodeConfig,
    OutcomeNodeConfig,
)


def _parse_dimensions(dims: list[str]) -> list[DimensionType]:
    """Convert string dimension names to DimensionType enums."""
    dim_map = {
        "period": DimensionType.PERIOD,
        "geography": DimensionType.GEOGRAPHY,
        "geo": DimensionType.GEOGRAPHY,
        "product": DimensionType.PRODUCT,
        "campaign": DimensionType.CAMPAIGN,
        "outlet": DimensionType.OUTLET,
        "creative": DimensionType.CREATIVE,
    }

    result = []
    for d in dims:
        lower_d = d.lower()
        if lower_d in dim_map:
            result.append(dim_map[lower_d])
        else:
            # Try exact match with DimensionType values
            for dt in DimensionType:
                if dt.value.lower() == lower_d:
                    result.append(dt)
                    break

    # Default to Period if empty
    if not result:
        result = [DimensionType.PERIOD]

    return result


def _build_media_config(
    node_id: str,
    variable_name: str,
    dimensions: list[str],
    config: dict,
    label: str | None = None,
) -> MediaChannelConfig:
    """Build MediaChannelConfig from DAG node."""
    # Parse node config
    media_config = MediaNodeConfig(
        **{k: v for k, v in config.items() if k in MediaNodeConfig.model_fields}
    )

    # Build adstock config
    adstock = AdstockConfig(
        type=media_config.adstock_type,
        l_max=media_config.adstock_lmax,
        normalize=media_config.adstock_normalize,
        alpha_prior=PriorConfig.beta(
            alpha=media_config.adstock_alpha_prior_alpha,
            beta=media_config.adstock_alpha_prior_beta,
        ),
    )

    # Build saturation config
    saturation = SaturationConfig(
        type=media_config.saturation_type,
        kappa_prior=PriorConfig.beta(
            alpha=media_config.saturation_kappa_prior_alpha,
            beta=media_config.saturation_kappa_prior_beta,
        ),
        slope_prior=PriorConfig.half_normal(
            sigma=media_config.saturation_slope_prior_sigma
        ),
        beta_prior=PriorConfig.half_normal(
            sigma=media_config.saturation_beta_prior_sigma
        ),
    )

    # Build coefficient prior
    coefficient_prior = PriorConfig.half_normal(
        sigma=media_config.coefficient_prior_sigma
    )

    return MediaChannelConfig(
        name=variable_name,
        display_name=label or variable_name,
        dimensions=_parse_dimensions(dimensions),
        adstock=adstock,
        saturation=saturation,
        coefficient_prior=coefficient_prior,
        parent_channel=media_config.parent_channel,
    )


def _build_control_config(
    node_id: str,
    variable_name: str,
    dimensions: list[str],
    config: dict,
    label: str | None = None,
    causal_role: CausalControlRole | None = None,
    causal_role_reason: str | None = None,
) -> ControlVariableConfig:
    """Build ControlVariableConfig from DAG node.

    ``causal_role`` (and its human-readable ``causal_role_reason``) are inferred
    from the identified adjustment set by :func:`dag_to_mff_config` so the model
    can prevent bad-control bias.
    """
    # Parse node config
    control_config = ControlNodeConfig(
        **{k: v for k, v in config.items() if k in ControlNodeConfig.model_fields}
    )

    # Build coefficient prior
    if control_config.allow_negative:
        coefficient_prior = PriorConfig(
            distribution=PriorType.NORMAL,
            params={
                "mu": control_config.coefficient_prior_mu,
                "sigma": control_config.coefficient_prior_sigma,
            },
        )
    else:
        coefficient_prior = PriorConfig.half_normal(
            sigma=control_config.coefficient_prior_sigma
        )

    return ControlVariableConfig(
        name=variable_name,
        display_name=label or variable_name,
        dimensions=_parse_dimensions(dimensions),
        allow_negative=control_config.allow_negative,
        coefficient_prior=coefficient_prior,
        use_shrinkage=control_config.use_shrinkage,
        causal_role=causal_role,
        causal_role_reason=causal_role_reason,
    )


def _build_kpi_config(
    node_id: str,
    variable_name: str,
    dimensions: list[str],
    config: dict,
    label: str | None = None,
) -> KPIConfig:
    """Build KPIConfig from DAG node."""
    # Parse node config
    kpi_config = KPINodeConfig(
        **{k: v for k, v in config.items() if k in KPINodeConfig.model_fields}
    )

    return KPIConfig(
        name=variable_name,
        display_name=label or variable_name,
        dimensions=_parse_dimensions(dimensions),
        log_transform=kpi_config.log_transform,
        floor_value=kpi_config.floor_value,
    )


def _classify_dag_controls(dag: DAGSpec) -> DagRoleClassification | None:
    """Run backdoor identification to classify control roles, or ``None``.

    Returns ``None`` when there is nothing to identify (no media treatments or no
    KPI), in which case all controls keep an unknown/precision role.
    """
    kpi_nodes = dag.kpi_nodes
    media_nodes = dag.media_nodes
    if not kpi_nodes or not media_nodes:
        return None
    # Single primary outcome; multi-KPI DAGs classify against the first KPI.
    outcome_id = kpi_nodes[0].id
    treatment_ids = [n.id for n in media_nodes]
    # Pass the actual control set so collider detection reflects what the model
    # really conditions on (collider danger depends on the full conditioning set).
    control_ids = [n.id for n in dag.control_nodes]
    return classify_dag_roles(dag, treatment_ids, outcome_id, control_ids)


def _warn_on_identification(
    dag: DAGSpec, classification: DagRoleClassification, control_var_names: set[str]
) -> None:
    """Surface the identification verdict the fitting path would otherwise ignore.

    Two distinct warnings, matching critique.md §3.1 ("``identifiable`` is
    reported as if it characterizes the fitted model"):

    1. an identified confounder (adjustment-set node) that is *not* among the
       model's controls is an open backdoor the user cannot see; and
    2. the effect is not identified by the proposed adjustment set at all.
    """
    missing = []
    for node_id in classification.adjustment_set:
        node = dag.get_node(node_id)
        if node is None:
            continue
        if node.variable_name not in control_var_names:
            missing.append(node.variable_name)
    if missing:
        warnings.warn(
            "Identified confounder(s) "
            f"{missing} are in the backdoor adjustment set but are NOT included "
            "as controls. The corresponding backdoor path is left open, so media "
            "effects may be confounded. Add them as controls, or anchor the "
            "channel with an experiment-calibrated prior "
            "(mmm_framework.calibration).",
            UserWarning,
            stacklevel=3,
        )
    if not classification.identifiable:
        warnings.warn(
            "The treatment effect is NOT identified by backdoor adjustment for "
            "at least one media channel: open backdoor path(s) remain after "
            "conditioning on the available controls. Treat reported effects as "
            "associational unless anchored by experimental calibration "
            "(mmm_framework.calibration). See the identification notes for "
            "details.",
            UserWarning,
            stacklevel=3,
        )


def dag_to_mff_config(
    dag: DAGSpec,
    date_format: str = "%Y-%m-%d",
    frequency: str = "W",
    enforce_identification: bool = True,
) -> MFFConfig:
    """
    Translate DAG specification to MFFConfig.

    Parameters
    ----------
    dag : DAGSpec
        The DAG specification.
    date_format : str
        Date format string for parsing.
    frequency : str
        Data frequency ("W", "D", "M").
    enforce_identification : bool
        When True (default), run backdoor identification and (a) tag each control
        with the causal role inferred from the adjustment set, so the model can
        refuse bad controls, and (b) warn when the effect is unidentified or when
        an identified confounder is missing from the controls. Set False to skip
        identification entirely (controls keep an unknown role).

    Returns
    -------
    MFFConfig
        The generated MFF configuration.

    Raises
    ------
    ValueError
        If no KPI node is found in the DAG.

    Notes
    -----
    INSTRUMENT nodes are intentionally NOT emitted into the MFFConfig: they are
    exogenous variation used only for graph-based IV identification checks
    (:func:`..identification.iv_criterion`), not model regressors. IV *estimation*
    is a separate, not-yet-implemented feature.
    """
    # Find KPI node (first one found)
    kpi_nodes = dag.kpi_nodes
    if not kpi_nodes:
        raise ValueError("DAG must have at least one KPI node")

    kpi_node = kpi_nodes[0]
    kpi_config = _build_kpi_config(
        kpi_node.id,
        kpi_node.variable_name,
        kpi_node.dimensions,
        kpi_node.config,
        kpi_node.label,
    )

    # Build media configs
    media_configs = []
    for node in dag.media_nodes:
        media_config = _build_media_config(
            node.id,
            node.variable_name,
            node.dimensions,
            node.config,
            node.label,
        )
        media_configs.append(media_config)

    # Classify control causal roles from the identified adjustment set so the
    # model can prevent bad-control bias (mediators/colliders refused;
    # confounders given an un-shrunk prior).
    classification = _classify_dag_controls(dag) if enforce_identification else None

    # Build control configs
    control_configs = []
    for node in dag.control_nodes:
        causal_role: CausalControlRole | None = None
        causal_role_reason: str | None = None
        if classification is not None:
            role_str, reason = classification.role_for(node.id)
            causal_role = CausalControlRole(role_str)
            causal_role_reason = reason
        control_config = _build_control_config(
            node.id,
            node.variable_name,
            node.dimensions,
            node.config,
            node.label,
            causal_role=causal_role,
            causal_role_reason=causal_role_reason,
        )
        control_configs.append(control_config)

    if classification is not None:
        _warn_on_identification(dag, classification, {c.name for c in control_configs})

    return MFFConfig(
        kpi=kpi_config,
        media_channels=media_configs,
        controls=control_configs,
        date_format=date_format,
        frequency=frequency,
    )


def dag_to_nested_config(dag: DAGSpec):
    """
    Extract nested model configuration from DAG.

    Builds:
    - MediatorConfig for each mediator node
    - media_to_mediator_map from edges

    Parameters
    ----------
    dag : DAGSpec
        The DAG specification.

    Returns
    -------
    NestedModelConfig
        The nested model configuration.
    """
    from mmm_framework.mmm_extensions.config import (
        EffectConstraint,
        EffectPriorConfig,
        MediatorConfig,
        MediatorType,
        NestedModelConfig,
    )

    mediator_configs = []
    media_to_mediator_map: dict[str, list[str]] = {}

    for mediator_node in dag.mediator_nodes:
        # Parse mediator config
        config = mediator_node.config
        med_config = MediatorNodeConfig(
            **{k: v for k, v in config.items() if k in MediatorNodeConfig.model_fields}
        )

        # Map mediator type string to enum
        mediator_type_map = {
            "fully_observed": MediatorType.FULLY_OBSERVED,
            "partially_observed": MediatorType.PARTIALLY_OBSERVED,
            "aggregated_survey": MediatorType.AGGREGATED_SURVEY,
            "fully_latent": MediatorType.FULLY_LATENT,
        }
        mediator_type = mediator_type_map.get(
            med_config.mediator_type, MediatorType.PARTIALLY_OBSERVED
        )

        # Map effect constraint
        constraint_map = {
            "none": EffectConstraint.NONE,
            "positive": EffectConstraint.POSITIVE,
            "negative": EffectConstraint.NEGATIVE,
        }
        media_effect_constraint = constraint_map.get(
            med_config.media_effect_constraint, EffectConstraint.POSITIVE
        )

        mediator_config = MediatorConfig(
            name=mediator_node.variable_name,
            mediator_type=mediator_type,
            media_effect=EffectPriorConfig(
                constraint=media_effect_constraint,
                sigma=med_config.media_effect_sigma,
            ),
            outcome_effect=EffectPriorConfig(
                constraint=EffectConstraint.NONE,
                sigma=med_config.outcome_effect_sigma,
            ),
            observation_noise_sigma=med_config.observation_noise_sigma,
            allow_direct_effect=med_config.allow_direct_effect,
            direct_effect=EffectPriorConfig(sigma=med_config.direct_effect_sigma),
            apply_adstock=med_config.apply_adstock,
            apply_saturation=med_config.apply_saturation,
        )
        mediator_configs.append(mediator_config)

        # Build media_to_mediator_map from edges
        incoming_edges = dag.get_incoming_edges(mediator_node.id)
        for edge in incoming_edges:
            source_node = dag.get_node(edge.source)
            if source_node and source_node.node_type == NodeType.MEDIA:
                if source_node.variable_name not in media_to_mediator_map:
                    media_to_mediator_map[source_node.variable_name] = []
                media_to_mediator_map[source_node.variable_name].append(
                    mediator_node.variable_name
                )

    # Convert lists to tuples for frozen dataclass
    media_to_mediator_map_tuple = {
        k: tuple(v) for k, v in media_to_mediator_map.items()
    }

    return NestedModelConfig(
        mediators=tuple(mediator_configs),
        media_to_mediator_map=media_to_mediator_map_tuple,
    )


def dag_to_multivariate_config(dag: DAGSpec):
    """
    Extract multivariate model configuration from DAG.

    Builds:
    - OutcomeConfig for each outcome node
    - CrossEffectConfig for cross-effect edges

    Parameters
    ----------
    dag : DAGSpec
        The DAG specification.

    Returns
    -------
    MultivariateModelConfig
        The multivariate model configuration.
    """
    from mmm_framework.mmm_extensions.config import (
        CrossEffectConfig,
        CrossEffectType,
        EffectPriorConfig,
        MultivariateModelConfig,
        OutcomeConfig,
    )

    from .dag_spec import EdgeType

    outcome_configs = []

    # Process all outcome nodes (KPI + OUTCOME)
    for outcome_node in dag.outcome_nodes:
        config = outcome_node.config
        out_config = OutcomeNodeConfig(
            **{k: v for k, v in config.items() if k in OutcomeNodeConfig.model_fields}
        )

        outcome_config = OutcomeConfig(
            name=outcome_node.variable_name,
            column=outcome_node.variable_name,
            intercept_prior_sigma=out_config.intercept_prior_sigma,
            media_effect=EffectPriorConfig(sigma=out_config.media_effect_sigma),
            include_trend=out_config.include_trend,
            include_seasonality=out_config.include_seasonality,
        )
        outcome_configs.append(outcome_config)

    # Build cross-effect configs
    cross_effect_configs = []
    for edge in dag.edges:
        if edge.edge_type == EdgeType.CROSS_EFFECT:
            source_node = dag.get_node(edge.source)
            target_node = dag.get_node(edge.target)

            if source_node and target_node:
                # Per-edge overrides from spec.priors.cross_effect (folded into
                # edge.metadata upstream): effect_type (halo/cannibalization/…)
                # and prior_sigma. Default stays CANNIBALIZATION.
                meta = edge.metadata or {}
                try:
                    effect_type = CrossEffectType(
                        str(meta.get("effect_type", "cannibalization")).lower()
                    )
                except ValueError:
                    effect_type = CrossEffectType.CANNIBALIZATION
                kwargs = {
                    "source_outcome": source_node.variable_name,
                    "target_outcome": target_node.variable_name,
                    "effect_type": effect_type,
                }
                if meta.get("prior_sigma") is not None:
                    kwargs["prior_sigma"] = float(meta["prior_sigma"])
                cross_effect_configs.append(CrossEffectConfig(**kwargs))

    return MultivariateModelConfig(
        outcomes=tuple(outcome_configs),
        cross_effects=tuple(cross_effect_configs),
    )


def dag_to_combined_config(dag: DAGSpec):
    """
    Build combined nested + multivariate config from DAG.

    Parameters
    ----------
    dag : DAGSpec
        The DAG specification.

    Returns
    -------
    CombinedModelConfig
        The combined model configuration.
    """
    from mmm_framework.mmm_extensions.config import CombinedModelConfig

    nested_config = dag_to_nested_config(dag)
    multivariate_config = dag_to_multivariate_config(dag)

    # Build mediator_to_outcome_map from edges
    mediator_to_outcome_map: dict[str, list[str]] = {}

    for mediator_node in dag.mediator_nodes:
        outgoing_edges = dag.get_outgoing_edges(mediator_node.id)
        for edge in outgoing_edges:
            target_node = dag.get_node(edge.target)
            if target_node and target_node.is_target:
                if mediator_node.variable_name not in mediator_to_outcome_map:
                    mediator_to_outcome_map[mediator_node.variable_name] = []
                mediator_to_outcome_map[mediator_node.variable_name].append(
                    target_node.variable_name
                )

    # Convert to tuples for frozen dataclass
    mediator_to_outcome_map_tuple = {
        k: tuple(v) for k, v in mediator_to_outcome_map.items()
    }

    return CombinedModelConfig(
        nested=nested_config,
        multivariate=multivariate_config,
        mediator_to_outcome_map=mediator_to_outcome_map_tuple,
    )


# Valid LatentFactorSpec construction keys for dag.metadata["latent_factors"]
# entries (kept in sync with mmm_extensions.config.LatentFactorSpec).
_LATENT_FACTOR_KEYS = {
    "name",
    "dynamics",
    "rho_prior_alpha",
    "rho_prior_beta",
    "affects_outcome",
    "outcome_effect_sigma",
    "mediator_effect_sigma",
    "anchor",
}


def dag_to_structural_config(dag: DAGSpec):
    """Extract a :class:`StructuralNestedConfig` + per-mediator data
    requirements from a structural DAG.

    Per mediator node: channels from MEDIA->mediator edges, parents from
    mediator->mediator edges, controls from CONTROL->mediator edges, latent
    factors + dynamics/measurement/priors from the node config -- mapping ONLY
    keys present in the RAW config dict so ``MediatorSpec`` defaults (e.g. the
    tight direct-effect prior, dynamics-resolved adstock) hold when unset.
    Latent factors come from ``dag.metadata["latent_factors"]`` (a list of
    LatentFactorSpec-shaped dicts). ``outcome_controls`` is the set of
    CONTROL->KPI edges, so a control driving only a mediator (price ->
    consideration) stays out of the outcome equation.

    Returns
    -------
    (StructuralNestedConfig, list[dict])
        The model config and, per non-latent mediator, the MFF data
        requirement: ``{"name", "likelihood", "variable_name",
        "trials_variable", "category_variables"}``.
    """
    from mmm_framework.mmm_extensions.config import (
        EffectConstraint,
        EffectPriorConfig,
        LatentFactorSpec,
        MediatorDynamics,
        MediatorLikelihood,
        MediatorMeasurement,
        MediatorSpec,
        StructuralNestedConfig,
    )

    constraint_map = {
        "none": EffectConstraint.NONE,
        "positive": EffectConstraint.POSITIVE,
        "negative": EffectConstraint.NEGATIVE,
    }
    id_to_node = {n.id: n for n in dag.nodes}
    # KPI + OUTCOME nodes: a mediator's default affects_outcome comes from
    # whether the DAG actually draws an edge into one of these.
    outcome_ids = {n.id for n in dag.outcome_nodes}

    mediator_specs = []
    data_requirements: list[dict] = []

    for mediator_node in dag.mediator_nodes:
        raw = dict(mediator_node.config or {})
        mc = MediatorNodeConfig(
            **{k: v for k, v in raw.items() if k in MediatorNodeConfig.model_fields}
        )
        name = mediator_node.variable_name

        channels: list[str] = []
        parents: list[str] = []
        controls: list[str] = []
        for edge in dag.get_incoming_edges(mediator_node.id):
            source = id_to_node.get(edge.source)
            if source is None:
                continue
            # Dedupe (order-preserving): duplicate edges would otherwise
            # produce duplicate driver terms and a PyMC duplicate-RV crash.
            if source.node_type == NodeType.MEDIA:
                if source.variable_name not in channels:
                    channels.append(source.variable_name)
            elif source.node_type == NodeType.MEDIATOR:
                if source.variable_name not in parents:
                    parents.append(source.variable_name)
            elif source.node_type == NodeType.CONTROL:
                if source.variable_name not in controls:
                    controls.append(source.variable_name)

        # Measurement family: explicit `likelihood` wins; else derive from the
        # plain-nested `mediator_type` (fully_latent -> LATENT, else GAUSSIAN).
        if mc.likelihood is not None:
            likelihood = MediatorLikelihood(str(mc.likelihood).lower())
        elif mc.trials_variable or mc.category_variables:
            # Survey columns without the matching family would silently derive
            # GAUSSIAN and fail later with a misleading shape error.
            raise ValueError(
                f"Structural mediator '{mediator_node.variable_name}' sets "
                "trials_variable/category_variables but no `likelihood` -- "
                "set likelihood='binomial' (trials) or 'ordered' (categories)"
            )
        elif mc.mediator_type == "fully_latent":
            likelihood = MediatorLikelihood.LATENT
        else:
            likelihood = MediatorLikelihood.GAUSSIAN

        if likelihood == MediatorLikelihood.BINOMIAL and not mc.trials_variable:
            raise ValueError(
                f"Structural mediator '{name}' has a binomial likelihood but "
                "no `trials_variable` (the weekly survey sample-size column)"
            )
        if likelihood == MediatorLikelihood.ORDERED and not mc.category_variables:
            raise ValueError(
                f"Structural mediator '{name}' has an ordered likelihood but "
                "no `category_variables` (the per-category count columns, "
                "ordered low -> high)"
            )

        meas_kwargs: dict = {"likelihood": likelihood}
        if "observation_noise_sigma" in raw:
            meas_kwargs["noise_sigma"] = mc.observation_noise_sigma
        if "design_effect" in raw:
            meas_kwargs["design_effect"] = mc.design_effect
        if mc.category_variables:
            meas_kwargs["n_categories"] = len(mc.category_variables)
        if "cutpoint_prior_sigma" in raw:
            meas_kwargs["cutpoint_prior_sigma"] = mc.cutpoint_prior_sigma

        spec_kwargs: dict = {
            "name": name,
            "channels": tuple(channels),
            "parents": tuple(parents),
            "controls": tuple(controls),
            "latent_factors": tuple(mc.latent_factors or ()),
            "measurement": MediatorMeasurement(**meas_kwargs),
        }
        if "dynamics" in raw:
            spec_kwargs["dynamics"] = MediatorDynamics(str(mc.dynamics).lower())
        if "rho_prior_alpha" in raw:
            spec_kwargs["rho_prior_alpha"] = mc.rho_prior_alpha
        if "rho_prior_beta" in raw:
            spec_kwargs["rho_prior_beta"] = mc.rho_prior_beta
        if "innovation_sigma" in raw:
            spec_kwargs["innovation_sigma"] = mc.innovation_sigma
        if "state_parameterization" in raw:
            spec_kwargs["state_parameterization"] = mc.state_parameterization
        if "affects_outcome" in raw:
            spec_kwargs["affects_outcome"] = mc.affects_outcome
        else:
            # DAG-faithful default: the mediator feeds the outcome equation
            # only when the DAG draws that edge (a pure upstream funnel stage
            # like awareness -> consideration must not get a gamma path the
            # DAG does not contain).
            spec_kwargs["affects_outcome"] = any(
                e.source == mediator_node.id and e.target in outcome_ids
                for e in dag.edges
            )
        if "allow_direct_effect" in raw:
            spec_kwargs["allow_direct_effect"] = mc.allow_direct_effect
        if "direct_effect_sigma" in raw:
            spec_kwargs["direct_effect"] = EffectPriorConfig(
                sigma=mc.direct_effect_sigma
            )
        if "apply_adstock" in raw:
            spec_kwargs["apply_adstock"] = mc.apply_adstock
        if "media_effect_sigma" in raw or "media_effect_constraint" in raw:
            spec_kwargs["media_effect"] = EffectPriorConfig(
                constraint=constraint_map.get(
                    mc.media_effect_constraint, EffectConstraint.POSITIVE
                ),
                sigma=mc.media_effect_sigma,
            )
        if "outcome_effect_sigma" in raw:
            # Positive (the MediatorSpec default constraint) with the DAG's
            # scale -- funnel gammas are sign-constrained, unlike plain nested.
            spec_kwargs["outcome_effect"] = EffectPriorConfig(
                constraint=EffectConstraint.POSITIVE,
                sigma=mc.outcome_effect_sigma,
            )
        if "parent_effect_sigma" in raw:
            spec_kwargs["parent_effect"] = EffectPriorConfig(
                constraint=EffectConstraint.POSITIVE, sigma=mc.parent_effect_sigma
            )
        if "control_effect_sigma" in raw:
            spec_kwargs["control_effect"] = EffectPriorConfig(
                sigma=mc.control_effect_sigma
            )

        mediator_specs.append(MediatorSpec(**spec_kwargs))
        if likelihood != MediatorLikelihood.LATENT:
            data_requirements.append(
                {
                    "name": name,
                    "likelihood": likelihood.value,
                    "variable_name": name,
                    "trials_variable": mc.trials_variable,
                    "category_variables": list(mc.category_variables or []),
                }
            )

    # Latent factors ride the DAG metadata (list of LatentFactorSpec dicts).
    factors = []
    for entry in (dag.metadata or {}).get("latent_factors", []) or []:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError(
                "Each dag.metadata['latent_factors'] entry must be a dict with "
                f"a 'name' (got {entry!r})"
            )
        unknown = set(entry) - _LATENT_FACTOR_KEYS
        if unknown:
            raise ValueError(
                f"latent_factors entry {entry.get('name')!r} has unknown keys "
                f"{sorted(unknown)}; valid: {sorted(_LATENT_FACTOR_KEYS)}"
            )
        factors.append(LatentFactorSpec(**entry))

    # Outcome controls: only controls with an edge INTO the KPI.
    kpi_ids = {n.id for n in dag.nodes if n.node_type == NodeType.KPI}
    outcome_controls = tuple(
        id_to_node[e.source].variable_name
        for e in dag.edges
        if e.target in kpi_ids
        and e.source in id_to_node
        and id_to_node[e.source].node_type == NodeType.CONTROL
    )

    config = StructuralNestedConfig(
        mediators=tuple(mediator_specs),
        latent_factors=tuple(factors),
        outcome_controls=outcome_controls,
    )
    return config, data_requirements
