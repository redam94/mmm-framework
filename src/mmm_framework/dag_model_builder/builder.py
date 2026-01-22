"""
DAG Model Builder

Main builder class for constructing MMM models from DAG specifications.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import pandas as pd

from mmm_framework.config import InferenceMethod, MFFConfig, ModelConfig

from .config_translator import (
    dag_to_combined_config,
    dag_to_mff_config,
    dag_to_multivariate_config,
    dag_to_nested_config,
)
from .dag_spec import DAGEdge, DAGNode, DAGSpec, EdgeType, NodeType
from .model_type_resolver import ModelType, get_model_class, resolve_model_type
from .validation import ValidationResult, validate_complete, validate_dag

if TYPE_CHECKING:
    from mmm_framework.data_loader import PanelDataset
    from mmm_framework.model import BayesianMMM, TrendConfig
    from mmm_framework.mmm_extensions.models import (
        CombinedMMM,
        MultivariateMMM,
        NestedMMM,
    )


class DAGBuildError(Exception):
    """Exception raised when model building fails."""

    pass


class DAGModelBuilder:
    """
    Builder for constructing MMM models from DAG specifications.

    This is the primary interface for the DAG-based model building workflow.
    It supports fluent API for configuration and automatic model type selection.

    Examples
    --------
    Basic usage:

    >>> from mmm_framework.dag_model_builder import (
    ...     DAGModelBuilder, DAGSpec, DAGNode, DAGEdge, NodeType
    ... )
    >>> dag = DAGSpec(
    ...     nodes=[
    ...         DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
    ...         DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA),
    ...     ],
    ...     edges=[DAGEdge(source="tv", target="sales")]
    ... )
    >>> model = (
    ...     DAGModelBuilder()
    ...     .with_dag(dag)
    ...     .with_mff_data("data.csv")
    ...     .bayesian_numpyro()
    ...     .build()
    ... )

    With mediation:

    >>> dag = DAGSpec(
    ...     nodes=[
    ...         DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
    ...         DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA),
    ...         DAGNode(id="awareness", variable_name="Awareness", node_type=NodeType.MEDIATOR),
    ...     ],
    ...     edges=[
    ...         DAGEdge(source="tv", target="awareness"),
    ...         DAGEdge(source="awareness", target="sales"),
    ...         DAGEdge(source="tv", target="sales"),
    ...     ]
    ... )
    >>> # Automatically uses NestedMMM
    >>> model = DAGModelBuilder().with_dag(dag).with_mff_data(df).build()
    """

    def __init__(self) -> None:
        self._dag: DAGSpec | None = None
        self._panel: PanelDataset | None = None
        self._mff_config: MFFConfig | None = None
        self._model_config: ModelConfig | None = None
        self._trend_config: TrendConfig | None = None
        self._node_config_overrides: dict[str, dict[str, Any]] = {}
        self._date_format: str = "%Y-%m-%d"
        self._frequency: str = "W"

    # =========================================================================
    # DAG Specification
    # =========================================================================

    def with_dag(self, dag: DAGSpec) -> Self:
        """
        Set the DAG specification.

        Parameters
        ----------
        dag : DAGSpec
            The DAG specification.

        Returns
        -------
        Self
            The builder instance for chaining.
        """
        self._dag = dag
        return self

    def with_dag_dict(self, dag_dict: dict) -> Self:
        """
        Set DAG from a dictionary.

        Parameters
        ----------
        dag_dict : dict
            Dictionary with "nodes" and "edges" keys.

        Returns
        -------
        Self
            The builder instance for chaining.
        """
        nodes = [DAGNode(**n) for n in dag_dict.get("nodes", [])]
        edges = [DAGEdge(**e) for e in dag_dict.get("edges", [])]
        metadata = dag_dict.get("metadata", {})

        self._dag = DAGSpec(nodes=nodes, edges=edges, metadata=metadata)
        return self

    @classmethod
    def from_dag(cls, dag: DAGSpec) -> Self:
        """
        Create a builder with a DAG specification.

        Parameters
        ----------
        dag : DAGSpec
            The DAG specification.

        Returns
        -------
        DAGModelBuilder
            A new builder instance with the DAG set.
        """
        builder = cls()
        builder._dag = dag
        return builder

    @classmethod
    def from_frontend_json(
        cls,
        json_data: dict | str,
        panel: "PanelDataset | None" = None,
    ) -> Self:
        """
        Create builder from React Flow frontend JSON format.

        Parameters
        ----------
        json_data : dict | str
            React Flow JSON data or JSON string.
        panel : PanelDataset | None
            Optional panel dataset.

        Returns
        -------
        DAGModelBuilder
            A new builder instance.
        """
        from .frontend_adapter import react_flow_to_dag_spec
        import json

        if isinstance(json_data, str):
            json_data = json.loads(json_data)

        nodes = json_data.get("nodes", [])
        edges = json_data.get("edges", [])

        dag = react_flow_to_dag_spec(nodes, edges)

        builder = cls()
        builder._dag = dag
        if panel is not None:
            builder._panel = panel

        return builder

    # =========================================================================
    # Data
    # =========================================================================

    def with_panel(self, panel: "PanelDataset") -> Self:
        """
        Set the panel dataset.

        Parameters
        ----------
        panel : PanelDataset
            The panel dataset.

        Returns
        -------
        Self
            The builder instance for chaining.
        """
        self._panel = panel
        return self

    def with_mff_data(
        self,
        data: pd.DataFrame | str,
        mff_config: MFFConfig | None = None,
    ) -> Self:
        """
        Load data from MFF format.

        If mff_config is not provided, it will be auto-generated from the DAG.

        Parameters
        ----------
        data : pd.DataFrame | str
            MFF data or path to CSV file.
        mff_config : MFFConfig | None
            Optional MFF configuration. If None, generated from DAG.

        Returns
        -------
        Self
            The builder instance for chaining.
        """
        from mmm_framework.data_loader import MFFLoader

        if mff_config is not None:
            self._mff_config = mff_config
        elif self._dag is not None:
            self._mff_config = dag_to_mff_config(
                self._dag,
                date_format=self._date_format,
                frequency=self._frequency,
            )
        else:
            raise DAGBuildError(
                "Cannot load MFF data without DAG or MFF config. "
                "Call with_dag() first or provide mff_config."
            )

        # Load data
        if isinstance(data, str):
            data = pd.read_csv(data)

        loader = MFFLoader(self._mff_config)
        self._panel = loader.load(data).build_panel()

        return self

    def with_date_format(self, date_format: str) -> Self:
        """
        Set the date format for parsing MFF data.

        Parameters
        ----------
        date_format : str
            Date format string (e.g., "%Y-%m-%d").

        Returns
        -------
        Self
            The builder instance for chaining.
        """
        self._date_format = date_format
        return self

    def with_frequency(self, frequency: str) -> Self:
        """
        Set the data frequency.

        Parameters
        ----------
        frequency : str
            Data frequency ("W" for weekly, "D" for daily, "M" for monthly).

        Returns
        -------
        Self
            The builder instance for chaining.
        """
        self._frequency = frequency
        return self

    # =========================================================================
    # Model Configuration
    # =========================================================================

    def with_model_config(self, config: ModelConfig) -> Self:
        """
        Set model-level configuration.

        Parameters
        ----------
        config : ModelConfig
            The model configuration.

        Returns
        -------
        Self
            The builder instance for chaining.
        """
        self._model_config = config
        return self

    def with_trend_config(self, config: "TrendConfig") -> Self:
        """
        Set trend configuration.

        Parameters
        ----------
        config : TrendConfig
            The trend configuration.

        Returns
        -------
        Self
            The builder instance for chaining.
        """
        self._trend_config = config
        return self

    def bayesian_pymc(self) -> Self:
        """
        Use PyMC backend for inference.

        Returns
        -------
        Self
            The builder instance for chaining.
        """
        if self._model_config is None:
            self._model_config = ModelConfig()

        self._model_config = self._model_config.model_copy(
            update={"inference_method": InferenceMethod.BAYESIAN_PYMC}
        )
        return self

    def bayesian_numpyro(self) -> Self:
        """
        Use NumPyro backend for inference (faster).

        Returns
        -------
        Self
            The builder instance for chaining.
        """
        if self._model_config is None:
            self._model_config = ModelConfig()

        self._model_config = self._model_config.model_copy(
            update={"inference_method": InferenceMethod.BAYESIAN_NUMPYRO}
        )
        return self

    def with_draws(self, n_draws: int) -> Self:
        """
        Set number of posterior draws.

        Parameters
        ----------
        n_draws : int
            Number of draws per chain.

        Returns
        -------
        Self
            The builder instance for chaining.
        """
        if self._model_config is None:
            self._model_config = ModelConfig()

        self._model_config = self._model_config.model_copy(update={"n_draws": n_draws})
        return self

    def with_tune(self, n_tune: int) -> Self:
        """
        Set number of tuning samples.

        Parameters
        ----------
        n_tune : int
            Number of tuning samples per chain.

        Returns
        -------
        Self
            The builder instance for chaining.
        """
        if self._model_config is None:
            self._model_config = ModelConfig()

        self._model_config = self._model_config.model_copy(update={"n_tune": n_tune})
        return self

    def with_chains(self, n_chains: int) -> Self:
        """
        Set number of chains.

        Parameters
        ----------
        n_chains : int
            Number of MCMC chains.

        Returns
        -------
        Self
            The builder instance for chaining.
        """
        if self._model_config is None:
            self._model_config = ModelConfig()

        self._model_config = self._model_config.model_copy(update={"n_chains": n_chains})
        return self

    # =========================================================================
    # Node-Level Configuration Overrides
    # =========================================================================

    def configure_media(
        self,
        variable_name: str,
        adstock_lmax: int | None = None,
        adstock_type: str | None = None,
        saturation_type: str | None = None,
        coefficient_prior_sigma: float | None = None,
        parent_channel: str | None = None,
    ) -> Self:
        """
        Override configuration for a specific media channel.

        Parameters
        ----------
        variable_name : str
            Name of the media variable.
        adstock_lmax : int | None
            Maximum lag for adstock.
        adstock_type : str | None
            Type of adstock ("geometric", "weibull", etc.).
        saturation_type : str | None
            Type of saturation ("hill", "logistic", etc.).
        coefficient_prior_sigma : float | None
            Sigma for coefficient prior.
        parent_channel : str | None
            Parent channel for hierarchical grouping.

        Returns
        -------
        Self
            The builder instance for chaining.
        """
        if variable_name not in self._node_config_overrides:
            self._node_config_overrides[variable_name] = {}

        overrides = self._node_config_overrides[variable_name]

        if adstock_lmax is not None:
            overrides["adstock_lmax"] = adstock_lmax
        if adstock_type is not None:
            from mmm_framework.config import AdstockType
            overrides["adstock_type"] = AdstockType(adstock_type)
        if saturation_type is not None:
            from mmm_framework.config import SaturationType
            overrides["saturation_type"] = SaturationType(saturation_type)
        if coefficient_prior_sigma is not None:
            overrides["coefficient_prior_sigma"] = coefficient_prior_sigma
        if parent_channel is not None:
            overrides["parent_channel"] = parent_channel

        return self

    def configure_control(
        self,
        variable_name: str,
        allow_negative: bool | None = None,
        coefficient_prior_mu: float | None = None,
        coefficient_prior_sigma: float | None = None,
        use_shrinkage: bool | None = None,
    ) -> Self:
        """
        Override configuration for a specific control variable.

        Parameters
        ----------
        variable_name : str
            Name of the control variable.
        allow_negative : bool | None
            Whether to allow negative coefficient.
        coefficient_prior_mu : float | None
            Mean for coefficient prior.
        coefficient_prior_sigma : float | None
            Sigma for coefficient prior.
        use_shrinkage : bool | None
            Whether to apply shrinkage prior.

        Returns
        -------
        Self
            The builder instance for chaining.
        """
        if variable_name not in self._node_config_overrides:
            self._node_config_overrides[variable_name] = {}

        overrides = self._node_config_overrides[variable_name]

        if allow_negative is not None:
            overrides["allow_negative"] = allow_negative
        if coefficient_prior_mu is not None:
            overrides["coefficient_prior_mu"] = coefficient_prior_mu
        if coefficient_prior_sigma is not None:
            overrides["coefficient_prior_sigma"] = coefficient_prior_sigma
        if use_shrinkage is not None:
            overrides["use_shrinkage"] = use_shrinkage

        return self

    def configure_mediator(
        self,
        variable_name: str,
        mediator_type: str | None = None,
        observation_noise_sigma: float | None = None,
        allow_direct_effect: bool | None = None,
        media_effect_constraint: str | None = None,
    ) -> Self:
        """
        Override configuration for a specific mediator.

        Parameters
        ----------
        variable_name : str
            Name of the mediator variable.
        mediator_type : str | None
            Type of mediator observation model.
        observation_noise_sigma : float | None
            Observation noise sigma.
        allow_direct_effect : bool | None
            Whether to allow direct media -> outcome effect.
        media_effect_constraint : str | None
            Constraint on media -> mediator effect.

        Returns
        -------
        Self
            The builder instance for chaining.
        """
        if variable_name not in self._node_config_overrides:
            self._node_config_overrides[variable_name] = {}

        overrides = self._node_config_overrides[variable_name]

        if mediator_type is not None:
            overrides["mediator_type"] = mediator_type
        if observation_noise_sigma is not None:
            overrides["observation_noise_sigma"] = observation_noise_sigma
        if allow_direct_effect is not None:
            overrides["allow_direct_effect"] = allow_direct_effect
        if media_effect_constraint is not None:
            overrides["media_effect_constraint"] = media_effect_constraint

        return self

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(self) -> ValidationResult:
        """
        Validate DAG structure and data compatibility.

        Returns
        -------
        ValidationResult
            Validation result with errors and warnings.

        Raises
        ------
        DAGBuildError
            If no DAG has been set.
        """
        if self._dag is None:
            raise DAGBuildError("No DAG set. Call with_dag() first.")

        return validate_complete(self._dag, self._panel)

    # =========================================================================
    # Build
    # =========================================================================

    def get_model_type(self) -> ModelType:
        """
        Get the resolved model type based on DAG structure.

        Returns
        -------
        ModelType
            The model type that will be built.

        Raises
        ------
        DAGBuildError
            If no DAG has been set.
        """
        if self._dag is None:
            raise DAGBuildError("No DAG set. Call with_dag() first.")

        return resolve_model_type(self._dag)

    def build_mff_config(self) -> MFFConfig:
        """
        Build MFFConfig from DAG without building the full model.

        Returns
        -------
        MFFConfig
            The generated MFF configuration.

        Raises
        ------
        DAGBuildError
            If no DAG has been set.
        """
        if self._dag is None:
            raise DAGBuildError("No DAG set. Call with_dag() first.")

        # Apply overrides to DAG nodes
        dag = self._apply_overrides()

        return dag_to_mff_config(
            dag,
            date_format=self._date_format,
            frequency=self._frequency,
        )

    def _apply_overrides(self) -> DAGSpec:
        """Apply node config overrides to DAG."""
        if self._dag is None:
            raise DAGBuildError("No DAG set.")

        if not self._node_config_overrides:
            return self._dag

        # Create new nodes with overrides applied
        new_nodes = []
        for node in self._dag.nodes:
            if node.variable_name in self._node_config_overrides:
                # Merge overrides into node config
                new_config = {**node.config, **self._node_config_overrides[node.variable_name]}
                new_node = node.model_copy(update={"config": new_config})
                new_nodes.append(new_node)
            else:
                new_nodes.append(node)

        return DAGSpec(
            nodes=new_nodes,
            edges=self._dag.edges,
            metadata=self._dag.metadata,
        )

    def build(
        self,
    ) -> "BayesianMMM | NestedMMM | MultivariateMMM | CombinedMMM":
        """
        Build the appropriate model based on DAG structure.

        Returns
        -------
        BayesianMMM | NestedMMM | MultivariateMMM | CombinedMMM
            The constructed model.

        Raises
        ------
        DAGBuildError
            If DAG or data is not set, or validation fails.
        """
        # Validate DAG is set
        if self._dag is None:
            raise DAGBuildError("No DAG set. Call with_dag() first.")

        # Apply overrides
        dag = self._apply_overrides()

        # Validate DAG structure
        validation = validate_dag(dag)
        if not validation.valid:
            raise DAGBuildError(
                "DAG validation failed:\n" + "\n".join(f"  - {e}" for e in validation.errors)
            )

        # Ensure panel data is available
        if self._panel is None:
            raise DAGBuildError(
                "No data set. Call with_panel() or with_mff_data() first."
            )

        # Get model type and class
        model_type = resolve_model_type(dag)
        model_class = get_model_class(model_type)

        # Get or create model config
        model_config = self._model_config or ModelConfig()

        # Build the appropriate model
        if model_type == ModelType.BAYESIAN_MMM:
            return model_class(
                panel=self._panel,
                model_config=model_config,
                trend_config=self._trend_config,
            )

        elif model_type == ModelType.NESTED_MMM:
            nested_config = dag_to_nested_config(dag)
            return model_class(
                panel=self._panel,
                model_config=model_config,
                nested_config=nested_config,
                trend_config=self._trend_config,
            )

        elif model_type == ModelType.MULTIVARIATE_MMM:
            multivariate_config = dag_to_multivariate_config(dag)
            return model_class(
                panel=self._panel,
                model_config=model_config,
                multivariate_config=multivariate_config,
                trend_config=self._trend_config,
            )

        elif model_type == ModelType.COMBINED_MMM:
            combined_config = dag_to_combined_config(dag)
            return model_class(
                panel=self._panel,
                model_config=model_config,
                combined_config=combined_config,
                trend_config=self._trend_config,
            )

        else:
            raise DAGBuildError(f"Unknown model type: {model_type}")
