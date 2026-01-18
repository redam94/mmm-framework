"""
Tests for serialization module.

These tests ensure that the MMMSerializer class correctly handles
save/load operations and maintains backward compatibility with
the BayesianMMM.save() and BayesianMMM.load() methods.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestMMMSerializerImports:
    """Tests for serialization module imports."""

    def test_can_import_serializer(self):
        """Test that MMMSerializer can be imported."""
        from mmm_framework.serialization import MMMSerializer

        assert MMMSerializer is not None

    def test_serializer_has_required_methods(self):
        """Test that MMMSerializer has all required class methods."""
        from mmm_framework.serialization import MMMSerializer

        assert hasattr(MMMSerializer, "save")
        assert hasattr(MMMSerializer, "load")
        assert hasattr(MMMSerializer, "save_trace_only")
        assert hasattr(MMMSerializer, "load_trace_only")

        # Verify they are class methods
        assert callable(MMMSerializer.save)
        assert callable(MMMSerializer.load)
        assert callable(MMMSerializer.save_trace_only)
        assert callable(MMMSerializer.load_trace_only)

    def test_serializer_has_helper_methods(self):
        """Test that MMMSerializer has private helper methods."""
        from mmm_framework.serialization import MMMSerializer

        assert hasattr(MMMSerializer, "_collect_metadata")
        assert hasattr(MMMSerializer, "_collect_configs")
        assert hasattr(MMMSerializer, "_collect_scaling_params")
        assert hasattr(MMMSerializer, "_save_trace")
        assert hasattr(MMMSerializer, "_load_trace")
        assert hasattr(MMMSerializer, "_check_version")
        assert hasattr(MMMSerializer, "_validate_panel_compatibility")
        assert hasattr(MMMSerializer, "_restore_scaling_params")


class TestMMMSerializerMetadataCollection:
    """Tests for metadata collection helper methods."""

    def test_collect_metadata_structure(self):
        """Test that _collect_metadata returns expected structure."""
        from mmm_framework.serialization import MMMSerializer

        # Create a mock model object with required attributes
        class MockModel:
            _VERSION = "1.0.0"
            n_obs = 100
            n_channels = 3
            n_controls = 2
            n_time_periods = 52
            channel_names = ["TV", "Radio", "Digital"]
            control_names = ["Price", "Promo"]
            has_geo = False
            has_product = False
            adstock_alphas = [0.0, 0.5, 0.9]

        model = MockModel()
        metadata = MMMSerializer._collect_metadata(model)

        assert "version" in metadata
        assert "format_version" in metadata
        assert "n_obs" in metadata
        assert "n_channels" in metadata
        assert "n_controls" in metadata
        assert "channel_names" in metadata
        assert "control_names" in metadata
        assert "has_geo" in metadata
        assert "has_product" in metadata
        assert "adstock_alphas" in metadata

        assert metadata["n_obs"] == 100
        assert metadata["n_channels"] == 3
        assert metadata["channel_names"] == ["TV", "Radio", "Digital"]

    def test_collect_metadata_with_geo(self):
        """Test metadata collection with geo dimension."""
        from mmm_framework.serialization import MMMSerializer

        class MockModelWithGeo:
            _VERSION = "1.0.0"
            n_obs = 100
            n_channels = 3
            n_controls = 2
            n_time_periods = 52
            channel_names = ["TV", "Radio", "Digital"]
            control_names = ["Price", "Promo"]
            has_geo = True
            has_product = False
            adstock_alphas = [0.0, 0.5]
            geo_names = ["East", "West", "Central"]

        model = MockModelWithGeo()
        metadata = MMMSerializer._collect_metadata(model)

        assert metadata["has_geo"] is True
        assert "geo_names" in metadata
        assert metadata["geo_names"] == ["East", "West", "Central"]

    def test_collect_metadata_with_product(self):
        """Test metadata collection with product dimension."""
        from mmm_framework.serialization import MMMSerializer

        class MockModelWithProduct:
            _VERSION = "1.0.0"
            n_obs = 100
            n_channels = 3
            n_controls = 2
            n_time_periods = 52
            channel_names = ["TV", "Radio", "Digital"]
            control_names = ["Price", "Promo"]
            has_geo = False
            has_product = True
            adstock_alphas = [0.0, 0.5]
            product_names = ["Product A", "Product B"]

        model = MockModelWithProduct()
        metadata = MMMSerializer._collect_metadata(model)

        assert metadata["has_product"] is True
        assert "product_names" in metadata
        assert metadata["product_names"] == ["Product A", "Product B"]


class TestMMMSerializerScalingParams:
    """Tests for scaling parameter collection."""

    def test_collect_scaling_params_basic(self):
        """Test basic scaling parameter collection."""
        from mmm_framework.serialization import MMMSerializer

        class MockModel:
            y_mean = 100.0
            y_std = 25.0
            _media_max = {"TV": 1000.0, "Radio": 500.0}
            X_controls_raw = None

        model = MockModel()
        params = MMMSerializer._collect_scaling_params(model)

        assert params["y_mean"] == 100.0
        assert params["y_std"] == 25.0
        assert params["media_max"] == {"TV": 1000.0, "Radio": 500.0}
        assert "control_mean" not in params
        assert "control_std" not in params

    def test_collect_scaling_params_with_controls(self):
        """Test scaling parameter collection with controls."""
        from mmm_framework.serialization import MMMSerializer

        class MockModel:
            y_mean = 100.0
            y_std = 25.0
            _media_max = {"TV": 1000.0}
            X_controls_raw = np.array([[1, 2], [3, 4]])
            control_mean = np.array([2.0, 3.0])
            control_std = np.array([1.0, 1.0])

        model = MockModel()
        params = MMMSerializer._collect_scaling_params(model)

        assert "control_mean" in params
        assert "control_std" in params
        assert params["control_mean"] == [2.0, 3.0]
        assert params["control_std"] == [1.0, 1.0]


class TestMMMSerializerVersionCheck:
    """Tests for version compatibility checking."""

    def test_version_check_matching(self):
        """Test version check with matching versions (no warning)."""
        from mmm_framework.serialization import MMMSerializer
        from mmm_framework.model import BayesianMMM

        metadata = {"version": BayesianMMM._VERSION}

        # Should not raise or warn
        MMMSerializer._check_version(metadata)

    def test_version_check_mismatch_warns(self):
        """Test version check warns on mismatch."""
        from mmm_framework.serialization import MMMSerializer
        import warnings

        metadata = {"version": "0.0.0"}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MMMSerializer._check_version(metadata)
            assert len(w) == 1
            assert "compatibility" in str(w[0].message).lower()

    def test_version_check_missing_version(self):
        """Test version check with missing version."""
        from mmm_framework.serialization import MMMSerializer
        import warnings

        metadata = {}  # No version

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MMMSerializer._check_version(metadata)
            assert len(w) == 1


class TestMMMSerializerPanelValidation:
    """Tests for panel compatibility validation."""

    def test_validate_panel_matching(self):
        """Test validation with matching panel."""
        from mmm_framework.serialization import MMMSerializer

        class MockCoords:
            channels = ["TV", "Radio"]
            controls = ["Price"]

        class MockPanel:
            coords = MockCoords()

        metadata = {
            "channel_names": ["TV", "Radio"],
            "control_names": ["Price"],
        }

        # Should not raise
        MMMSerializer._validate_panel_compatibility(MockPanel(), metadata)

    def test_validate_panel_channel_mismatch(self):
        """Test validation raises on channel mismatch."""
        from mmm_framework.serialization import MMMSerializer

        class MockCoords:
            channels = ["TV", "Radio"]
            controls = ["Price"]

        class MockPanel:
            coords = MockCoords()

        metadata = {
            "channel_names": ["TV", "Digital"],  # Different!
            "control_names": ["Price"],
        }

        with pytest.raises(ValueError, match="channels"):
            MMMSerializer._validate_panel_compatibility(MockPanel(), metadata)

    def test_validate_panel_control_mismatch(self):
        """Test validation raises on control mismatch."""
        from mmm_framework.serialization import MMMSerializer

        class MockCoords:
            channels = ["TV", "Radio"]
            controls = ["Price"]

        class MockPanel:
            coords = MockCoords()

        metadata = {
            "channel_names": ["TV", "Radio"],
            "control_names": ["Promo"],  # Different!
        }

        with pytest.raises(ValueError, match="controls"):
            MMMSerializer._validate_panel_compatibility(MockPanel(), metadata)


class TestMMMSerializerRestoreScalingParams:
    """Tests for restoring scaling parameters."""

    def test_restore_basic_scaling_params(self):
        """Test restoring basic scaling parameters."""
        from mmm_framework.serialization import MMMSerializer

        class MockInstance:
            y_mean = None
            y_std = None
            _media_max = None
            _scaling_params = {}

        instance = MockInstance()
        scaling_params = {
            "y_mean": 100.0,
            "y_std": 25.0,
            "media_max": {"TV": 1000.0},
        }

        MMMSerializer._restore_scaling_params(instance, scaling_params)

        assert instance.y_mean == 100.0
        assert instance.y_std == 25.0
        assert instance._media_max == {"TV": 1000.0}
        assert instance._scaling_params["y_mean"] == 100.0
        assert instance._scaling_params["y_std"] == 25.0

    def test_restore_scaling_params_with_controls(self):
        """Test restoring scaling parameters with control stats."""
        from mmm_framework.serialization import MMMSerializer

        class MockInstance:
            y_mean = None
            y_std = None
            _media_max = None
            _scaling_params = {}
            control_mean = None
            control_std = None

        instance = MockInstance()
        scaling_params = {
            "y_mean": 100.0,
            "y_std": 25.0,
            "media_max": {"TV": 1000.0},
            "control_mean": [2.0, 3.0],
            "control_std": [1.0, 1.5],
        }

        MMMSerializer._restore_scaling_params(instance, scaling_params)

        np.testing.assert_array_equal(instance.control_mean, [2.0, 3.0])
        np.testing.assert_array_equal(instance.control_std, [1.0, 1.5])


class TestSerializerFormatVersion:
    """Tests for format version handling."""

    def test_format_version_exists(self):
        """Test that format version constant exists."""
        from mmm_framework.serialization import MMMSerializer

        assert hasattr(MMMSerializer, "_FORMAT_VERSION")
        assert isinstance(MMMSerializer._FORMAT_VERSION, str)

    def test_format_version_in_metadata(self):
        """Test that format version is included in metadata."""
        from mmm_framework.serialization import MMMSerializer

        class MockModel:
            _VERSION = "1.0.0"
            n_obs = 100
            n_channels = 3
            n_controls = 2
            n_time_periods = 52
            channel_names = ["TV", "Radio", "Digital"]
            control_names = ["Price", "Promo"]
            has_geo = False
            has_product = False
            adstock_alphas = [0.0, 0.5]

        metadata = MMMSerializer._collect_metadata(MockModel())
        assert "format_version" in metadata
        assert metadata["format_version"] == MMMSerializer._FORMAT_VERSION


class TestBayesianMMMSerializationIntegration:
    """Tests for BayesianMMM integration with serialization."""

    def test_save_method_uses_serializer(self):
        """Test that BayesianMMM.save uses MMMSerializer."""
        from mmm_framework.model import BayesianMMM

        # Verify the method imports from serialization
        import inspect

        source = inspect.getsource(BayesianMMM.save)
        assert "MMMSerializer" in source
        # Check that it imports from serialization module (relative import depth may vary)
        assert "serialization import MMMSerializer" in source

    def test_load_method_uses_serializer(self):
        """Test that BayesianMMM.load uses MMMSerializer."""
        from mmm_framework.model import BayesianMMM

        import inspect

        source = inspect.getsource(BayesianMMM.load)
        assert "MMMSerializer" in source

    def test_save_trace_only_uses_serializer(self):
        """Test that save_trace_only uses MMMSerializer."""
        from mmm_framework.model import BayesianMMM

        import inspect

        source = inspect.getsource(BayesianMMM.save_trace_only)
        assert "MMMSerializer" in source

    def test_load_trace_only_uses_serializer(self):
        """Test that load_trace_only uses MMMSerializer."""
        from mmm_framework.model import BayesianMMM

        import inspect

        source = inspect.getsource(BayesianMMM.load_trace_only)
        assert "MMMSerializer" in source


class TestMMMSerializerLoadTrace:
    """Tests for trace loading."""

    def test_load_trace_missing_file(self, tmp_path):
        """Test loading trace when file doesn't exist."""
        from mmm_framework.serialization import MMMSerializer

        path = tmp_path / "test_model"
        path.mkdir()

        result = MMMSerializer._load_trace(path)

        assert result is None


class TestMMMSerializerSaveTraceOnly:
    """Tests for save_trace_only method."""

    def test_save_trace_only_method_exists(self):
        """Test that save_trace_only method exists on class."""
        from mmm_framework.serialization import MMMSerializer

        assert hasattr(MMMSerializer, "save_trace_only")
        assert callable(MMMSerializer.save_trace_only)


class TestMMMSerializerLoadTraceOnly:
    """Tests for load_trace_only method."""

    def test_load_trace_only_method_exists(self):
        """Test that load_trace_only method exists on class."""
        from mmm_framework.serialization import MMMSerializer

        assert hasattr(MMMSerializer, "load_trace_only")
        assert callable(MMMSerializer.load_trace_only)

    def test_load_trace_only_missing_file_raises(self, tmp_path):
        """Test that load_trace_only raises on missing file."""
        from mmm_framework.serialization import MMMSerializer

        nonexistent_path = tmp_path / "nonexistent.nc"

        with pytest.raises(FileNotFoundError):
            MMMSerializer.load_trace_only(nonexistent_path)


class TestMMMSerializerEdgeCases:
    """Edge case tests for serializer."""

    def test_collect_metadata_with_minimal_model(self):
        """Test metadata collection with minimal model attributes."""
        from mmm_framework.serialization import MMMSerializer

        class MinimalModel:
            _VERSION = "1.0.0"
            n_obs = 10
            n_channels = 1
            n_controls = 0
            n_time_periods = 10
            channel_names = ["TV"]
            control_names = []
            has_geo = False
            has_product = False
            adstock_alphas = None

        metadata = MMMSerializer._collect_metadata(MinimalModel())

        assert metadata["n_controls"] == 0
        assert metadata["control_names"] == []
        assert metadata["adstock_alphas"] is None

    def test_format_version_format(self):
        """Test format version is correctly formatted."""
        from mmm_framework.serialization import MMMSerializer

        # Format version should be a simple string
        assert isinstance(MMMSerializer._FORMAT_VERSION, str)
        assert len(MMMSerializer._FORMAT_VERSION) > 0


class TestMMMSerializerSaveLoadPath:
    """Tests for path handling in save/load."""

    def test_save_creates_directory(self, tmp_path):
        """Test that save creates directory if it doesn't exist."""
        from mmm_framework.serialization import MMMSerializer
        from pathlib import Path

        # Create minimal mock model
        class MockConfig:
            def model_dump(self):
                return {}

        class MockTrendConfig:
            def to_dict(self):
                return {}

        class MockModel:
            _VERSION = "1.0.0"
            n_obs = 10
            n_channels = 1
            n_controls = 0
            n_time_periods = 10
            channel_names = ["TV"]
            control_names = []
            has_geo = False
            has_product = False
            adstock_alphas = [0.5]
            _trace = None
            model_config = MockConfig()
            trend_config = MockTrendConfig()
            mff_config = MockConfig()
            y_mean = 100.0
            y_std = 25.0
            _media_max = {"TV": 1000.0}
            X_controls_raw = None
            trend_features = {}
            seasonality_features = {}

        new_dir = tmp_path / "new_model_dir"
        assert not new_dir.exists()

        MMMSerializer.save(MockModel(), new_dir)

        assert new_dir.exists()
        assert (new_dir / "metadata.json").exists()
        assert (new_dir / "configs.json").exists()
        assert (new_dir / "scaling_params.json").exists()

    def test_load_raises_on_missing_directory(self, tmp_path):
        """Test that load raises on missing directory."""
        from mmm_framework.serialization import MMMSerializer

        class MockPanel:
            pass

        nonexistent_path = tmp_path / "nonexistent_model"

        with pytest.raises(FileNotFoundError):
            MMMSerializer.load(nonexistent_path, MockPanel())
