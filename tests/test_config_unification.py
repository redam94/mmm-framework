"""
Tests for config unification between main config and mmm_extensions config.

These tests ensure that:
1. Shared enums are imported correctly from the main config
2. Extension-specific configs remain independent
3. Both configs can be used together without conflicts
4. Backward compatibility is maintained
"""

import pytest


class TestSaturationTypeUnification:
    """Tests for SaturationType enum unification."""

    def test_main_config_saturation_type_exists(self):
        """Test that SaturationType exists in main config."""
        from mmm_framework.config import SaturationType

        assert SaturationType is not None
        assert hasattr(SaturationType, "HILL")
        assert hasattr(SaturationType, "LOGISTIC")

    def test_extension_config_saturation_type_exists(self):
        """Test that SaturationType is available in extension config."""
        from mmm_framework.mmm_extensions.config import SaturationType

        assert SaturationType is not None
        assert hasattr(SaturationType, "HILL")
        assert hasattr(SaturationType, "LOGISTIC")

    def test_saturation_types_are_same_class(self):
        """Test that both modules use the same SaturationType class."""
        from mmm_framework.config import SaturationType as MainSaturationType
        from mmm_framework.mmm_extensions.config import (
            SaturationType as ExtSaturationType,
        )

        # They should be the exact same class object
        assert MainSaturationType is ExtSaturationType

    def test_saturation_type_values_identical(self):
        """Test that enum values are identical across modules."""
        from mmm_framework.config import SaturationType as MainSaturationType
        from mmm_framework.mmm_extensions.config import (
            SaturationType as ExtSaturationType,
        )

        # Compare HILL values
        assert MainSaturationType.HILL == ExtSaturationType.HILL
        assert MainSaturationType.HILL.value == ExtSaturationType.HILL.value

        # Compare LOGISTIC values
        assert MainSaturationType.LOGISTIC == ExtSaturationType.LOGISTIC
        assert MainSaturationType.LOGISTIC.value == ExtSaturationType.LOGISTIC.value

    def test_extension_has_access_to_all_saturation_types(self):
        """Test that extension can access all saturation types from main config."""
        from mmm_framework.mmm_extensions.config import SaturationType

        # All main config saturation types should be available
        expected_types = ["HILL", "LOGISTIC", "MICHAELIS_MENTEN", "TANH", "NONE"]
        for type_name in expected_types:
            assert hasattr(SaturationType, type_name), f"Missing {type_name}"

    def test_saturation_type_in_extension_saturation_config(self):
        """Test that extension SaturationConfig uses the unified SaturationType."""
        from mmm_framework.config import SaturationType as MainSaturationType
        from mmm_framework.mmm_extensions.config import SaturationConfig

        # Create a SaturationConfig with default type
        config = SaturationConfig()

        # The type should be from the unified SaturationType
        assert config.type in list(MainSaturationType)

    def test_can_use_main_saturation_type_in_extension_config(self):
        """Test that main config SaturationType works in extension configs."""
        from mmm_framework.config import SaturationType
        from mmm_framework.mmm_extensions.config import SaturationConfig

        # Create config with explicit type from main config
        config = SaturationConfig(type=SaturationType.HILL)
        assert config.type == SaturationType.HILL

        config2 = SaturationConfig(type=SaturationType.LOGISTIC)
        assert config2.type == SaturationType.LOGISTIC


class TestAdstockConfigSeparation:
    """Tests for AdstockConfig separation between main and extension."""

    def test_main_adstock_config_is_pydantic(self):
        """Test that main AdstockConfig is a Pydantic model."""
        from mmm_framework.config import AdstockConfig

        # Check it's a Pydantic model (has model_config)
        assert hasattr(AdstockConfig, "model_config")

    def test_extension_adstock_config_is_dataclass(self):
        """Test that extension AdstockConfig is a frozen dataclass."""
        from dataclasses import is_dataclass

        from mmm_framework.mmm_extensions.config import AdstockConfig

        assert is_dataclass(AdstockConfig)

    def test_adstock_configs_are_different_classes(self):
        """Test that main and extension AdstockConfig are different classes."""
        from mmm_framework.config import AdstockConfig as MainAdstockConfig
        from mmm_framework.mmm_extensions.config import (
            AdstockConfig as ExtAdstockConfig,
        )

        # They should be different classes
        assert MainAdstockConfig is not ExtAdstockConfig

    def test_main_adstock_config_has_prior_config(self):
        """Test that main AdstockConfig uses PriorConfig for priors."""
        from mmm_framework.config import AdstockConfig

        config = AdstockConfig.geometric(l_max=8)
        assert hasattr(config, "alpha_prior")

    def test_extension_adstock_config_has_flat_priors(self):
        """Test that extension AdstockConfig uses flat prior parameters."""
        from mmm_framework.mmm_extensions.config import AdstockConfig

        config = AdstockConfig()
        assert hasattr(config, "prior_alpha")
        assert hasattr(config, "prior_beta")
        assert isinstance(config.prior_alpha, float)


class TestSaturationConfigSeparation:
    """Tests for SaturationConfig separation between main and extension."""

    def test_main_saturation_config_is_pydantic(self):
        """Test that main SaturationConfig is a Pydantic model."""
        from mmm_framework.config import SaturationConfig

        assert hasattr(SaturationConfig, "model_config")

    def test_extension_saturation_config_is_dataclass(self):
        """Test that extension SaturationConfig is a frozen dataclass."""
        from dataclasses import is_dataclass

        from mmm_framework.mmm_extensions.config import SaturationConfig

        assert is_dataclass(SaturationConfig)

    def test_saturation_configs_are_different_classes(self):
        """Test that main and extension SaturationConfig are different classes."""
        from mmm_framework.config import SaturationConfig as MainSaturationConfig
        from mmm_framework.mmm_extensions.config import (
            SaturationConfig as ExtSaturationConfig,
        )

        assert MainSaturationConfig is not ExtSaturationConfig


class TestExtensionSpecificEnums:
    """Tests for enums that only exist in mmm_extensions."""

    def test_mediator_type_only_in_extension(self):
        """Test that MediatorType is only in extension."""
        from mmm_framework.mmm_extensions.config import MediatorType

        assert MediatorType is not None
        assert hasattr(MediatorType, "FULLY_OBSERVED")
        assert hasattr(MediatorType, "PARTIALLY_OBSERVED")

    def test_cross_effect_type_only_in_extension(self):
        """Test that CrossEffectType is only in extension."""
        from mmm_framework.mmm_extensions.config import CrossEffectType

        assert CrossEffectType is not None
        assert hasattr(CrossEffectType, "CANNIBALIZATION")
        assert hasattr(CrossEffectType, "HALO")

    def test_effect_constraint_only_in_extension(self):
        """Test that EffectConstraint is only in extension."""
        from mmm_framework.mmm_extensions.config import EffectConstraint

        assert EffectConstraint is not None
        assert hasattr(EffectConstraint, "NONE")
        assert hasattr(EffectConstraint, "POSITIVE")
        assert hasattr(EffectConstraint, "NEGATIVE")


class TestExtensionPackageExports:
    """Tests for mmm_extensions package exports."""

    def test_saturation_type_exported_from_package(self):
        """Test that SaturationType is exported from mmm_extensions package."""
        from mmm_framework.mmm_extensions import SaturationType

        assert SaturationType is not None

    def test_exported_saturation_type_is_unified(self):
        """Test that exported SaturationType is the unified version."""
        from mmm_framework.config import SaturationType as MainSaturationType
        from mmm_framework.mmm_extensions import SaturationType as ExportedType

        assert MainSaturationType is ExportedType

    def test_all_extension_configs_exported(self):
        """Test that all extension configs are properly exported."""
        from mmm_framework.mmm_extensions import (
            AdstockConfig,
            CrossEffectConfig,
            EffectConstraint,
            EffectPriorConfig,
            MediatorConfig,
            MediatorType,
            OutcomeConfig,
            SaturationConfig,
            SaturationType,
        )

        # All should be importable
        assert AdstockConfig is not None
        assert SaturationConfig is not None
        assert EffectPriorConfig is not None
        assert MediatorConfig is not None
        assert OutcomeConfig is not None
        assert CrossEffectConfig is not None
        assert MediatorType is not None
        assert EffectConstraint is not None
        assert SaturationType is not None


class TestBackwardCompatibility:
    """Tests for backward compatibility after unification."""

    def test_old_extension_saturation_type_usage(self):
        """Test that old code using extension SaturationType still works."""
        from mmm_framework.mmm_extensions.config import (
            SaturationConfig,
            SaturationType,
        )

        # This is how old code might have been written
        config = SaturationConfig(type=SaturationType.LOGISTIC)
        assert config.type == SaturationType.LOGISTIC

    def test_cross_module_enum_comparison(self):
        """Test that enum values can be compared across modules."""
        from mmm_framework.config import SaturationType as MainType
        from mmm_framework.mmm_extensions.config import SaturationType as ExtType

        # Comparisons should work
        assert MainType.HILL == ExtType.HILL
        assert MainType.LOGISTIC == ExtType.LOGISTIC

    def test_main_config_unchanged(self):
        """Test that main config functionality is unchanged."""
        from mmm_framework.config import (
            AdstockConfig,
            MediaChannelConfig,
            PriorConfig,
            SaturationConfig,
        )

        # Create configs as before
        adstock = AdstockConfig.geometric(l_max=8)
        saturation = SaturationConfig.hill()

        # Check structure
        assert adstock.l_max == 8
        assert saturation.type.value == "hill"

    def test_mediator_config_with_saturation(self):
        """Test MediatorConfig uses the unified SaturationType."""
        from mmm_framework.config import SaturationType as MainType
        from mmm_framework.mmm_extensions.config import (
            MediatorConfig,
            SaturationConfig,
        )

        sat_config = SaturationConfig(type=MainType.HILL)
        mediator = MediatorConfig(name="awareness", saturation=sat_config)

        assert mediator.saturation.type == MainType.HILL


class TestConfigCreation:
    """Tests for creating configs with the unified types."""

    def test_create_extension_saturation_config_with_hill(self):
        """Test creating extension SaturationConfig with HILL type."""
        from mmm_framework.mmm_extensions.config import (
            SaturationConfig,
            SaturationType,
        )

        config = SaturationConfig(type=SaturationType.HILL)
        assert config.type == SaturationType.HILL
        assert config.type.value == "hill"

    def test_create_extension_saturation_config_with_logistic(self):
        """Test creating extension SaturationConfig with LOGISTIC type."""
        from mmm_framework.mmm_extensions.config import (
            SaturationConfig,
            SaturationType,
        )

        config = SaturationConfig(type=SaturationType.LOGISTIC)
        assert config.type == SaturationType.LOGISTIC
        assert config.type.value == "logistic"

    def test_saturation_config_default_type(self):
        """Test default SaturationType in extension config."""
        from mmm_framework.mmm_extensions.config import (
            SaturationConfig,
            SaturationType,
        )

        config = SaturationConfig()
        assert config.type == SaturationType.LOGISTIC


class TestModuleImports:
    """Tests for import behavior across modules."""

    def test_circular_import_prevention(self):
        """Test that there are no circular import issues."""
        # These imports should all work without circular import errors
        from mmm_framework.config import SaturationType

        from mmm_framework.mmm_extensions.config import SaturationType as ExtType

        assert SaturationType is ExtType

    def test_extension_components_use_unified_type(self):
        """Test that extension components can use the unified type."""
        # This tests that the components module can import SaturationType
        from mmm_framework.mmm_extensions.config import SaturationType

        # Access a value to ensure it's properly imported
        assert SaturationType.HILL.value == "hill"

    def test_extension_builders_use_unified_type(self):
        """Test that extension builders work with unified type."""
        from mmm_framework.mmm_extensions.builders import SaturationConfigBuilder

        builder = SaturationConfigBuilder()
        config = builder.hill().build()

        from mmm_framework.config import SaturationType

        assert config.type == SaturationType.HILL
