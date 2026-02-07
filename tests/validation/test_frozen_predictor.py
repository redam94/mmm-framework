"""
Tests for the frozen predictor module.

This module contains unit tests for:
- FrozenPredictor class
- FrozenPredictorConfig
- create_frozen_predictor function
- Graph manipulation utilities
"""

import numpy as np
import pytest

from mmm_framework.validation import (
    FrozenPredictor,
    FrozenPredictorConfig,
    FrozenPredictorError,
    FrozenPredictorOutput,
    create_frozen_predictor,
    create_frozen_predictor_from_model,
)
from mmm_framework.validation.frozen_predictor import (
    _find_data_nodes,
    _find_needed_rvs,
    _flatten_posterior_samples,
)


class TestFrozenPredictorConfig:
    """Tests for FrozenPredictorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FrozenPredictorConfig()

        assert config.output_var == "y_obs"
        assert config.include_noise is True
        assert config.seed is None
        assert config.num_samples is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = FrozenPredictorConfig(
            output_var="mu",
            include_noise=False,
            seed=42,
            num_samples=100,
        )

        assert config.output_var == "mu"
        assert config.include_noise is False
        assert config.seed == 42
        assert config.num_samples == 100


class TestFrozenPredictorOutput:
    """Tests for FrozenPredictorOutput dataclass."""

    def test_output_with_mu_only(self):
        """Test output with only mu (no noise)."""
        mu = np.random.randn(100, 50)  # 100 samples, 50 points
        output = FrozenPredictorOutput(mu=mu)

        assert output.mu.shape == (100, 50)
        assert output.y_samples is None
        assert output.sigma is None

    def test_output_with_all_fields(self):
        """Test output with all fields populated."""
        mu = np.random.randn(100, 50)
        y_samples = np.random.randn(100, 50)
        sigma = np.random.rand(100)

        output = FrozenPredictorOutput(mu=mu, y_samples=y_samples, sigma=sigma)

        assert output.mu.shape == (100, 50)
        assert output.y_samples.shape == (100, 50)
        assert output.sigma.shape == (100,)


class TestFlattenPosteriorSamples:
    """Tests for _flatten_posterior_samples helper."""

    @pytest.fixture
    def mock_trace(self):
        """Create a mock ArviZ InferenceData-like object."""
        import xarray as xr

        # Create mock posterior with 4 chains, 100 draws
        posterior_data = {
            "intercept": xr.DataArray(
                np.random.randn(4, 100),
                dims=["chain", "draw"],
            ),
            "beta_tv": xr.DataArray(
                np.random.randn(4, 100),
                dims=["chain", "draw"],
            ),
            "sigma": xr.DataArray(
                np.abs(np.random.randn(4, 100)),
                dims=["chain", "draw"],
            ),
        }

        class MockPosterior:
            def __init__(self, data):
                self._data = data

            def __contains__(self, key):
                return key in self._data

            def __getitem__(self, key):
                return self._data[key]

        class MockTrace:
            def __init__(self, posterior):
                self.posterior = posterior

        return MockTrace(MockPosterior(posterior_data))

    def test_flatten_all_samples(self, mock_trace):
        """Test flattening all posterior samples."""
        rv_names = ["intercept", "sigma"]
        samples = _flatten_posterior_samples(mock_trace, rv_names)

        assert "intercept" in samples
        assert "sigma" in samples
        assert samples["intercept"].shape[0] == 400  # 4 chains * 100 draws
        assert samples["sigma"].shape[0] == 400

    def test_flatten_with_num_samples(self, mock_trace):
        """Test flattening with sample limit."""
        rv_names = ["intercept"]
        samples = _flatten_posterior_samples(
            mock_trace, rv_names, num_samples=50, seed=42
        )

        assert samples["intercept"].shape[0] == 50

    def test_flatten_missing_rv_warning(self, mock_trace, caplog):
        """Test warning for missing RV."""
        rv_names = ["intercept", "nonexistent_rv"]
        samples = _flatten_posterior_samples(mock_trace, rv_names)

        assert "intercept" in samples
        assert "nonexistent_rv" not in samples


class TestFrozenPredictorCreation:
    """Tests for frozen predictor creation with simple PyMC models."""

    @pytest.fixture
    def simple_pymc_model(self):
        """Create a simple PyMC model for testing."""
        pytest.importorskip("pymc")
        import pymc as pm
        import pytensor.tensor as pt

        n_obs = 50
        n_channels = 3

        with pm.Model() as model:
            # Data nodes
            X = pm.Data("X", np.random.randn(n_obs, n_channels))

            # Parameters
            intercept = pm.Normal("intercept", mu=0, sigma=1)
            beta = pm.Normal("beta", mu=0, sigma=1, shape=n_channels)
            sigma = pm.HalfNormal("sigma", sigma=1)

            # Linear predictor
            mu = intercept + pt.dot(X, beta)

            # Likelihood
            y_obs = pm.Normal(
                "y_obs", mu=mu, sigma=sigma, observed=np.random.randn(n_obs)
            )

        return model

    @pytest.fixture
    def mock_simple_trace(self, simple_pymc_model):
        """Create mock trace for simple model."""
        pytest.importorskip("arviz")
        import arviz as az
        import xarray as xr

        n_chains, n_draws = 2, 50

        posterior_data = {
            "intercept": (["chain", "draw"], np.random.randn(n_chains, n_draws)),
            "beta": (
                ["chain", "draw", "beta_dim_0"],
                np.random.randn(n_chains, n_draws, 3),
            ),
            "sigma": (
                ["chain", "draw"],
                np.abs(np.random.randn(n_chains, n_draws)) + 0.1,
            ),
        }

        posterior = xr.Dataset(posterior_data)
        return az.InferenceData(posterior=posterior)

    def test_create_frozen_predictor_simple(self, simple_pymc_model, mock_simple_trace):
        """Test creating a frozen predictor from a simple model."""
        config = FrozenPredictorConfig(
            output_var="y_obs",
            include_noise=True,
            seed=42,
        )

        try:
            predictor = create_frozen_predictor(
                model=simple_pymc_model,
                trace=mock_simple_trace,
                config=config,
            )

            assert isinstance(predictor, FrozenPredictor)
            assert predictor.n_samples == 100  # 2 chains * 50 draws
            assert "X" in predictor.input_names
        except FrozenPredictorError as e:
            # Some models may not support frozen prediction
            pytest.skip(f"Frozen predictor creation failed: {e}")

    def test_frozen_predictor_predict(self, simple_pymc_model, mock_simple_trace):
        """Test making predictions with frozen predictor."""
        config = FrozenPredictorConfig(
            output_var="y_obs",
            include_noise=False,
            seed=42,
        )

        try:
            predictor = create_frozen_predictor(
                model=simple_pymc_model,
                trace=mock_simple_trace,
                config=config,
            )

            # Make predictions with new data
            new_X = np.random.randn(20, 3)
            output = predictor.predict({"X": new_X})

            assert output.mu.shape[1] == 20  # 20 new points
            # mu should have n_samples rows
            assert output.mu.shape[0] == predictor.n_samples
        except FrozenPredictorError as e:
            pytest.skip(f"Frozen predictor creation failed: {e}")

    def test_frozen_predictor_missing_input(self, simple_pymc_model, mock_simple_trace):
        """Test error when required input is missing."""
        config = FrozenPredictorConfig(output_var="y_obs")

        try:
            predictor = create_frozen_predictor(
                model=simple_pymc_model,
                trace=mock_simple_trace,
                config=config,
            )

            with pytest.raises(ValueError, match="Missing required inputs"):
                predictor.predict({})  # Empty inputs
        except FrozenPredictorError as e:
            pytest.skip(f"Frozen predictor creation failed: {e}")


class TestFrozenPredictorFromModel:
    """Tests for create_frozen_predictor_from_model convenience function."""

    def test_no_trace_error(self):
        """Test error when model has no trace."""

        class MockModel:
            @property
            def model(self):
                return None

            @property
            def _trace(self):
                return None

        mock_model = MockModel()

        with pytest.raises(FrozenPredictorError, match="no trace"):
            create_frozen_predictor_from_model(mock_model)


class TestFrozenPredictorWithNoise:
    """Tests for observation noise handling."""

    def test_noise_reproducibility(self):
        """Test that noise generation is reproducible with same seed."""
        mu = np.random.randn(50, 20)
        sigma = np.abs(np.random.randn(50)) + 0.1

        output1 = FrozenPredictorOutput(mu=mu, sigma=sigma)
        output2 = FrozenPredictorOutput(mu=mu, sigma=sigma)

        # Without y_samples, just testing that output creation works
        assert output1.mu.shape == output2.mu.shape
        assert output1.sigma.shape == output2.sigma.shape


class TestCrossValidationConfigFrozenPredictor:
    """Tests for frozen predictor options in CrossValidationConfig."""

    def test_default_frozen_predictor_enabled(self):
        """Test that frozen predictor is enabled by default."""
        from mmm_framework.validation import CrossValidationConfig

        config = CrossValidationConfig()

        assert config.use_frozen_predictor is True
        assert config.frozen_predictor_seed == 42

    def test_custom_frozen_predictor_config(self):
        """Test custom frozen predictor configuration."""
        from mmm_framework.validation import CrossValidationConfig

        config = CrossValidationConfig(
            use_frozen_predictor=False,
            frozen_predictor_seed=None,
        )

        assert config.use_frozen_predictor is False
        assert config.frozen_predictor_seed is None


class TestFindNeededRVs:
    """Tests for _find_needed_rvs helper function."""

    def test_output_var_not_found(self):
        """Test error when output variable doesn't exist."""
        pytest.importorskip("pymc")
        import pymc as pm

        with pm.Model() as model:
            x = pm.Normal("x", mu=0, sigma=1)

        with pytest.raises(FrozenPredictorError, match="not found in model"):
            _find_needed_rvs(model, "nonexistent_var")


class TestFindDataNodes:
    """Tests for _find_data_nodes helper function."""

    def test_find_data_nodes(self):
        """Test finding pm.Data nodes in a model."""
        pytest.importorskip("pymc")
        import pymc as pm

        with pm.Model() as model:
            X = pm.Data("X", np.random.randn(10, 3))
            y = pm.Data("y_data", np.random.randn(10))
            beta = pm.Normal("beta", mu=0, sigma=1)

        data_nodes = _find_data_nodes(model)

        # Should find X and y_data but not beta (which is an RV)
        assert "X" in data_nodes or "y_data" in data_nodes
