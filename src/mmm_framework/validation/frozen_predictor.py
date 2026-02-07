"""
Frozen Predictor for efficient posterior predictions.

This module provides a compiled PyTensor function for generating posterior
predictions that reuses the actual model computation graph with frozen
posterior samples, avoiding the need to manually reconstruct model logic.

Key features:
- Uses ancestors() to find only the RVs needed for outputs (efficient compilation)
- Supports dynamic input sizing for cross-validation
- Handles observed intermediate RVs using ICDF replacement (for mediation models)
- Vectorized across posterior samples for efficiency
- Reproducible sampling via seed parameter
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import arviz as az
    import pymc as pm

logger = logging.getLogger(__name__)


class FrozenPredictorError(Exception):
    """Raised when frozen predictor cannot be created or used."""

    pass


@dataclass
class FrozenPredictorConfig:
    """Configuration for frozen predictor creation.

    Parameters
    ----------
    output_var : str
        Name of the output variable to predict (e.g., "y_obs", "mu").
        Default is "y_obs" which gives the full likelihood output.
    include_noise : bool
        Whether to include observation noise (sigma) in predictions.
        If True, samples from Normal(mu, sigma). Default is True.
    seed : int | None
        Random seed for reproducibility. Affects both posterior sample
        selection and observation noise generation.
    num_samples : int | None
        Number of posterior samples to use. If None, uses all available.
    """

    output_var: str = "y_obs"
    include_noise: bool = True
    seed: int | None = None
    num_samples: int | None = None


@dataclass
class FrozenPredictorOutput:
    """Output from frozen predictor evaluation.

    Parameters
    ----------
    mu : np.ndarray
        Deterministic predictions (mean), shape (n_samples, n_points).
    y_samples : np.ndarray | None
        Predictions with observation noise (if include_noise=True),
        shape (n_samples, n_points).
    sigma : np.ndarray | None
        Sigma samples used for noise generation.
    """

    mu: npt.NDArray[np.floating[Any]]
    y_samples: npt.NDArray[np.floating[Any]] | None = None
    sigma: npt.NDArray[np.floating[Any]] | None = None


@runtime_checkable
class HasPyMCModel(Protocol):
    """Protocol for objects with a PyMC model."""

    @property
    def model(self) -> "pm.Model": ...

    @property
    def _trace(self) -> "az.InferenceData | None": ...


class FrozenPredictor:
    """
    Compiled predictor using frozen posterior samples.

    Reuses the model's actual computation graph with:
    - Free RVs replaced by frozen posterior samples
    - Input data replaced by new symbolic inputs
    - SpecifyShape ops removed for dynamic sizing

    This avoids manually reconstructing model logic and ensures predictions
    match exactly what the model would produce.

    Parameters
    ----------
    predict_fn : callable
        Compiled PyTensor function for predictions.
    input_names : list[str]
        Names of input variables (pm.Data nodes).
    input_shapes : dict[str, tuple]
        Expected shapes for each input (for validation).
    posterior_samples : dict[str, np.ndarray]
        Flattened posterior samples for all needed RVs.
    sigma_samples : np.ndarray | None
        Flattened sigma samples for observation noise.
    config : FrozenPredictorConfig
        Configuration used to create this predictor.
    """

    def __init__(
        self,
        predict_fn: Any,
        input_names: list[str],
        input_shapes: dict[str, tuple[int, ...]],
        posterior_samples: dict[str, npt.NDArray[np.floating[Any]]],
        sigma_samples: npt.NDArray[np.floating[Any]] | None,
        config: FrozenPredictorConfig,
    ):
        self._predict_fn = predict_fn
        self._input_names = input_names
        self._input_shapes = input_shapes
        self._posterior_samples = posterior_samples
        self._sigma_samples = sigma_samples
        self._config = config
        self._n_samples = next(iter(posterior_samples.values())).shape[0]

    @property
    def n_samples(self) -> int:
        """Number of posterior samples available."""
        return self._n_samples

    @property
    def input_names(self) -> list[str]:
        """Names of required input variables."""
        return self._input_names.copy()

    def predict(
        self,
        inputs: dict[str, npt.NDArray[np.floating[Any]]],
        include_noise: bool | None = None,
        seed: int | None = None,
    ) -> FrozenPredictorOutput:
        """
        Generate predictions for new inputs.

        Parameters
        ----------
        inputs : dict[str, np.ndarray]
            New input data. Keys must match input_names.
            Values should have shape (n_points, ...) matching original dims.
        include_noise : bool | None
            Whether to add observation noise. If None, uses config setting.
        seed : int | None
            Random seed for noise generation. If None, uses config seed.

        Returns
        -------
        FrozenPredictorOutput
            Contains mu (deterministic) and optionally y_samples (with noise).
        """
        # Validate inputs
        missing = set(self._input_names) - set(inputs.keys())
        if missing:
            raise ValueError(f"Missing required inputs: {missing}")

        # Prepare input arrays in correct order
        input_arrays = []
        for name in self._input_names:
            arr = np.asarray(inputs[name], dtype=np.float64)
            input_arrays.append(arr)

        # Run the compiled function
        try:
            mu = self._predict_fn(*input_arrays)
        except Exception as e:
            raise FrozenPredictorError(
                f"Prediction failed: {e}. "
                "This may indicate shape mismatch or graph compilation issues."
            ) from e

        # Add observation noise if requested
        if include_noise is None:
            include_noise = self._config.include_noise

        y_samples = None
        sigma = None
        if include_noise and self._sigma_samples is not None:
            sigma = self._sigma_samples
            rng = np.random.default_rng(seed or self._config.seed)
            # mu shape: (n_samples, n_points)
            # sigma shape: (n_samples,) or (n_samples, 1)
            sigma_broadcast = sigma[:, np.newaxis] if sigma.ndim == 1 else sigma
            noise = rng.normal(0, sigma_broadcast, size=mu.shape)
            y_samples = mu + noise

        return FrozenPredictorOutput(
            mu=mu,
            y_samples=y_samples,
            sigma=sigma,
        )


def _flatten_posterior_samples(
    trace: "az.InferenceData",
    rv_names: list[str],
    num_samples: int | None = None,
    seed: int | None = None,
) -> dict[str, npt.NDArray[np.floating[Any]]]:
    """
    Flatten chain and draw dimensions of posterior samples.

    Parameters
    ----------
    trace : az.InferenceData
        ArviZ inference data with posterior group.
    rv_names : list[str]
        Names of RVs to extract.
    num_samples : int | None
        Number of samples to use. If None, uses all.
    seed : int | None
        Random seed for sample selection.

    Returns
    -------
    dict[str, np.ndarray]
        Flattened samples with shape (n_samples, *param_dims).
    """
    posterior = trace.posterior

    samples = {}
    for name in rv_names:
        if name not in posterior:
            logger.warning(f"RV '{name}' not found in posterior, skipping")
            continue

        arr = posterior[name].values
        # Shape: (n_chains, n_draws, *param_dims) -> (n_samples, *param_dims)
        n_chains, n_draws = arr.shape[:2]
        flattened = arr.reshape(n_chains * n_draws, *arr.shape[2:])

        if num_samples is not None and num_samples < flattened.shape[0]:
            rng = np.random.default_rng(seed)
            indices = rng.choice(flattened.shape[0], size=num_samples, replace=False)
            flattened = flattened[indices]

        samples[name] = flattened

    return samples


def _find_data_nodes(model: "pm.Model") -> dict[str, Any]:
    """
    Find all pm.Data nodes in the model.

    Parameters
    ----------
    model : pm.Model
        PyMC model to analyze.

    Returns
    -------
    dict[str, TensorVariable]
        Mapping from variable name to pm.Data node.
    """
    import pytensor.tensor as pt

    data_nodes = {}
    for name, var in model.named_vars.items():
        # pm.Data creates a SharedVariable wrapper
        if hasattr(var, "get_value"):
            data_nodes[name] = var
        # Also check for constant tensors that were created from data
        elif hasattr(var, "owner") and var.owner is None:
            # Could be a constant - check if it has a value
            if hasattr(var, "data"):
                data_nodes[name] = var

    return data_nodes


def _find_needed_rvs(
    model: "pm.Model",
    output_var: str,
) -> tuple[list[Any], list[Any], set[Any]]:
    """
    Find all RVs needed to compute the output variable.

    Uses graph traversal via ancestors() to find only the RVs that
    are actually required, avoiding unnecessary computation.

    Parameters
    ----------
    model : pm.Model
        PyMC model.
    output_var : str
        Name of output variable.

    Returns
    -------
    tuple[list, list, set]
        - needed_free_rvs: Free RVs that are ancestors of output
        - needed_observed_rvs: Observed RVs that are ancestors of output
        - all_ancestors: All ancestor nodes
    """
    from pytensor.graph.traversal import ancestors

    if output_var not in model.named_vars:
        raise FrozenPredictorError(f"Output variable '{output_var}' not found in model")

    output = model.named_vars[output_var]
    all_ancestors = set(ancestors([output]))

    # Find which RVs are in the ancestry
    free_rvs_set = set(model.free_RVs)
    observed_rvs_set = set(model.observed_RVs)

    needed_free_rvs = [rv for rv in model.free_RVs if rv in all_ancestors]
    needed_observed_rvs = [rv for rv in model.observed_RVs if rv in all_ancestors]

    return needed_free_rvs, needed_observed_rvs, all_ancestors


def _remove_specify_shape(
    outputs: list[Any],
    model: "pm.Model",
) -> list[Any]:
    """
    Remove SpecifyShape ops to allow dynamic input sizes.

    Parameters
    ----------
    outputs : list
        PyTensor output expressions.
    model : pm.Model
        PyMC model (for dim info).

    Returns
    -------
    list
        Outputs with SpecifyShape ops removed.
    """
    from pytensor.graph import clone_replace
    from pytensor.graph.traversal import ancestors
    from pytensor.tensor.shape import SpecifyShape

    replacements = {}
    for node in list(ancestors(outputs)) + list(outputs):
        if (
            hasattr(node, "owner")
            and node.owner
            and isinstance(node.owner.op, SpecifyShape)
        ):
            # Replace SpecifyShape output with its input
            tensor = node.owner.inputs[0]
            replacements[node] = tensor

    if not replacements:
        return outputs

    return clone_replace(outputs, replace=replacements, rebuild_strict=False)


def _replace_observed_rvs_with_icdf(
    model: "pm.Model",
    observed_rvs: list[Any],
    data_nodes: dict[str, Any],
    all_ancestors: set[Any],
    n_samples: int,
    seed: int | None = None,
) -> tuple[dict[Any, Any], npt.NDArray[np.floating[Any]]]:
    """
    Replace observed RVs that depend on inputs with ICDF expressions.

    For mediation models where intermediate observed RVs depend on inputs,
    we can't simply use the observed data. Instead, we:
    1. Sample uniform quantiles U ~ Uniform(0,1) once
    2. Replace the observed RV with ICDF(distribution, U)

    This makes predictions deterministic given the frozen quantiles.

    Parameters
    ----------
    model : pm.Model
        PyMC model.
    observed_rvs : list
        Observed RVs that are ancestors of output.
    data_nodes : dict
        Data nodes in the model.
    all_ancestors : set
        All ancestors of the output.
    n_samples : int
        Number of posterior samples.
    seed : int | None
        Random seed for uniform sampling.

    Returns
    -------
    tuple[dict, np.ndarray]
        - replacements: Mapping from observed RVs to replacement expressions
        - uniform_samples: 2D array (n_samples, n_observed_rvs) of frozen quantiles
    """
    import pymc as pm
    import pytensor.tensor as pt
    from pytensor.graph.traversal import ancestors

    replacements = {}
    uniform_samples_list = []
    data_vars = set(data_nodes.values())

    rng = np.random.default_rng(seed)

    for rv in observed_rvs:
        # Check if this observed RV depends on any input data nodes
        rv_ancestors = set(ancestors([rv]))
        depends_on_input = bool(rv_ancestors & data_vars)

        if not depends_on_input:
            # Input-independent: use observed data as constant
            observed_data = model.rvs_to_values[rv]
            if hasattr(observed_data, "get_value"):
                data_values = observed_data.get_value()
            else:
                data_values = np.asarray(observed_data)
            replacements[rv] = pt.constant(data_values.astype(rv.dtype), name=rv.name)
        else:
            # Input-dependent: use ICDF with frozen uniform quantiles
            logger.info(
                f"Observed RV '{rv.name}' depends on inputs. "
                "Using ICDF with frozen uniform quantiles."
            )

            # Sample uniform quantiles (one per posterior sample)
            uniform_vals = rng.uniform(0, 1, size=n_samples)
            uniform_samples_list.append(uniform_vals)

            # Create a constant tensor of uniform values
            uniform_tensor = pt.constant(
                uniform_vals.astype(rv.dtype), name=f"{rv.name}_uniform"
            )

            try:
                # Use ICDF to convert uniform to the target distribution
                # Note: pm.icdf returns the inverse CDF evaluated at the quantile
                icdf_expr = pm.icdf(rv, uniform_tensor, warn_rvs=False)
                replacements[rv] = icdf_expr
            except NotImplementedError as e:
                raise FrozenPredictorError(
                    f"ICDF not available for observed RV '{rv.name}' "
                    f"(distribution: {rv.owner.op}). "
                    f"This distribution doesn't support inverse CDF. "
                    f"Consider using sample_posterior_predictive instead."
                ) from e

    uniform_samples = (
        np.column_stack(uniform_samples_list) if uniform_samples_list else np.array([])
    )
    return replacements, uniform_samples


def create_frozen_predictor(
    model: "pm.Model",
    trace: "az.InferenceData",
    config: FrozenPredictorConfig | None = None,
) -> FrozenPredictor:
    """
    Create a frozen predictor from a fitted PyMC model.

    This function creates a compiled PyTensor function that evaluates model
    expressions at new input values, vectorized over posterior samples.
    It reuses the actual model graph rather than reconstructing it manually.

    Parameters
    ----------
    model : pm.Model
        The fitted PyMC model.
    trace : az.InferenceData
        ArviZ InferenceData with posterior samples.
    config : FrozenPredictorConfig | None
        Configuration options. If None, uses defaults.

    Returns
    -------
    FrozenPredictor
        Compiled predictor ready for inference.

    Raises
    ------
    FrozenPredictorError
        If the predictor cannot be created (e.g., missing output var,
        ICDF not available for observed RV).

    Examples
    --------
    >>> predictor = create_frozen_predictor(model.model, model._trace)
    >>> output = predictor.predict({
    ...     "X_media_low": X_new_low,
    ...     "X_media_high": X_new_high,
    ...     "X_controls": X_controls_new,
    ...     "time_idx": time_idx_new,
    ...     "geo_idx": geo_idx_new,
    ...     "product_idx": product_idx_new,
    ... })
    >>> y_pred = output.y_samples.mean(axis=0)
    """
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt
    from pytensor.graph import clone_replace
    from pytensor.graph.replace import vectorize_graph

    if config is None:
        config = FrozenPredictorConfig()

    # Find what we need to compute the output
    needed_free_rvs, needed_observed_rvs, all_ancestors = _find_needed_rvs(
        model, config.output_var
    )

    logger.debug(
        f"Found {len(needed_free_rvs)} free RVs needed for {config.output_var}"
    )

    # Find data nodes (inputs we'll replace)
    data_nodes = _find_data_nodes(model)
    logger.debug(f"Found {len(data_nodes)} data nodes: {list(data_nodes.keys())}")

    # Flatten posterior samples
    rv_names = [rv.name for rv in needed_free_rvs]
    posterior_samples = _flatten_posterior_samples(
        trace, rv_names, config.num_samples, config.seed
    )

    n_samples = next(iter(posterior_samples.values())).shape[0]
    logger.debug(f"Using {n_samples} posterior samples")

    # Get sigma samples for observation noise
    sigma_samples = None
    if config.include_noise and "sigma" in posterior_samples:
        sigma_samples = posterior_samples["sigma"]
    elif config.include_noise:
        # Try to find sigma in the trace
        sigma_samples_dict = _flatten_posterior_samples(
            trace, ["sigma"], config.num_samples, config.seed
        )
        if "sigma" in sigma_samples_dict:
            sigma_samples = sigma_samples_dict["sigma"]

    # Build replacement dictionaries

    # Step 1: Create PLACEHOLDER tensors for RVs (same shape as originals)
    # These will be vectorized later with actual samples
    rv_placeholders = {}
    for rv in needed_free_rvs:
        rv_placeholders[rv] = pt.tensor(
            name=rv.name,
            shape=rv.type.shape,
            dtype=rv.dtype,
        )

    # Step 2: Create symbolic inputs for data nodes
    symbolic_inputs = {}
    input_shapes = {}
    for name, data_var in data_nodes.items():
        if data_var in all_ancestors:
            # Get the shape from the current data
            if hasattr(data_var, "get_value"):
                current_shape = data_var.get_value().shape
            else:
                current_shape = np.asarray(data_var).shape

            input_shapes[name] = current_shape

            # Create appropriate symbolic variable based on dimensionality
            if len(current_shape) == 0:
                symbolic_inputs[name] = pt.scalar(f"{name}_in", dtype="float64")
            elif len(current_shape) == 1:
                symbolic_inputs[name] = pt.vector(f"{name}_in", dtype="float64")
            elif len(current_shape) == 2:
                symbolic_inputs[name] = pt.matrix(f"{name}_in", dtype="float64")
            else:
                symbolic_inputs[name] = pt.tensor(
                    f"{name}_in", dtype="float64", shape=(None,) * len(current_shape)
                )

    data_replacements = {
        data_nodes[name]: symbolic_inputs[name]
        for name in symbolic_inputs
        if name in data_nodes
    }

    # Step 3: Handle observed RVs (with ICDF for input-dependent ones)
    # Note: Pass rv_placeholders to ensure ICDF expressions use placeholders
    observed_rv_replacements, _ = _replace_observed_rvs_with_icdf(
        model, needed_observed_rvs, data_nodes, all_ancestors, n_samples, config.seed
    )

    # Step 4: Clone graph with placeholders and symbolic inputs (not samples yet)
    base_replacements = {
        **rv_placeholders,
        **data_replacements,
        **observed_rv_replacements,
    }

    # Get the output variable
    output = model.named_vars[config.output_var]

    # Clone the graph with placeholders
    try:
        cloned_outputs = clone_replace([output], replace=base_replacements)
    except Exception as e:
        raise FrozenPredictorError(
            f"Failed to clone graph: {e}. "
            "This may indicate incompatible replacements or graph structure issues."
        ) from e

    # Step 5: Vectorize with frozen samples using vectorize_graph
    # This properly broadcasts sample dimension across the computation
    sample_replacements = {
        placeholder: pt.constant(
            posterior_samples[placeholder.name],
            name=placeholder.name,
        )
        for placeholder in rv_placeholders.values()
        if placeholder.name in posterior_samples
    }

    try:
        vectorized_outputs = vectorize_graph(
            cloned_outputs, replace=sample_replacements
        )
    except Exception as e:
        raise FrozenPredictorError(
            f"Failed to vectorize graph: {e}. "
            "This may indicate shape inference issues with posterior samples."
        ) from e

    # Remove SpecifyShape ops for dynamic sizing
    final_outputs = _remove_specify_shape(vectorized_outputs, model)

    # Compile the function
    ordered_input_names = list(symbolic_inputs.keys())
    ordered_inputs = [symbolic_inputs[name] for name in ordered_input_names]

    try:
        compiled_fn = pytensor.function(
            inputs=ordered_inputs,
            outputs=final_outputs[0],  # Single output
            on_unused_input="ignore",
        )
    except Exception as e:
        raise FrozenPredictorError(
            f"Failed to compile function: {e}. "
            "This may indicate shape inference issues."
        ) from e

    logger.info(
        f"Created frozen predictor with {n_samples} samples, "
        f"{len(ordered_input_names)} inputs: {ordered_input_names}"
    )

    return FrozenPredictor(
        predict_fn=compiled_fn,
        input_names=ordered_input_names,
        input_shapes=input_shapes,
        posterior_samples=posterior_samples,
        sigma_samples=sigma_samples,
        config=config,
    )


def create_frozen_predictor_from_model(
    mmm_model: HasPyMCModel,
    config: FrozenPredictorConfig | None = None,
) -> FrozenPredictor:
    """
    Create a frozen predictor from an MMM model object.

    Convenience wrapper that extracts the PyMC model and trace
    from a BayesianMMM or extended model object.

    Parameters
    ----------
    mmm_model : HasPyMCModel
        MMM model with .model and ._trace attributes.
    config : FrozenPredictorConfig | None
        Configuration options.

    Returns
    -------
    FrozenPredictor
        Compiled predictor.

    Raises
    ------
    FrozenPredictorError
        If model hasn't been fitted or predictor creation fails.
    """
    if mmm_model._trace is None:
        raise FrozenPredictorError(
            "Model has no trace. Fit the model before creating a predictor."
        )

    return create_frozen_predictor(mmm_model.model, mmm_model._trace, config)


__all__ = [
    "FrozenPredictor",
    "FrozenPredictorConfig",
    "FrozenPredictorError",
    "FrozenPredictorOutput",
    "create_frozen_predictor",
    "create_frozen_predictor_from_model",
]
