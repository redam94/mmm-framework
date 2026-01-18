# MMM Framework Refactoring Guide

This document tracks code quality improvements for the MMM Framework. Each section contains specific refactoring tasks with implementation details and progress tracking.

---

## Table of Contents

1. [Builder Method Consolidation](#1-builder-method-consolidation)
2. [Data Standardization Utility](#2-data-standardization-utility)
3. [Time-Period Masking Helper](#3-time-period-masking-helper)
4. [HDI Calculation Utility](#4-hdi-calculation-utility)
5. [Config Lookup Consolidation](#5-config-lookup-consolidation)
6. [Naming Consistency Fixes](#6-naming-consistency-fixes)
7. [BayesianMMM Decomposition](#7-bayesianmmm-decomposition)
8. [Transform Utilities Module](#8-transform-utilities-module)
9. [Reporting Extractor Abstraction](#9-reporting-extractor-abstraction)
10. [Config Hierarchy Unification](#10-config-hierarchy-unification)

---

## 1. Builder Method Consolidation

**Priority**: HIGH
**Status**: [x] **COMPLETED** (2026-01-17)
**Estimated Impact**: ~90 lines of duplicated code removed

### Problem

Three builder classes in `builders.py` implement identical methods:
- `MediaChannelConfigBuilder` (lines 340-379)
- `ControlVariableConfigBuilder` (lines 487-517)
- `KPIConfigBuilder` (lines 589-628)

Duplicated methods:
- `with_display_name()`
- `with_unit()`
- `with_dimensions()`
- `national()`
- `by_geo()`
- `by_product()`
- `by_geo_and_product()`

### Solution

Create a `VariableConfigBuilderMixin` class that provides shared functionality.

### Implementation Steps

- [ ] **Step 1.1**: Create mixin class at top of `builders.py`
  ```python
  class VariableConfigBuilderMixin:
      """Shared methods for variable configuration builders."""

      _display_name: str | None
      _unit: str | None
      _dimensions: list[DimensionType]

      def _init_variable_fields(self) -> None:
          """Initialize common variable fields. Call in subclass __init__."""
          self._display_name = None
          self._unit = None
          self._dimensions = [DimensionType.PERIOD]

      def with_display_name(self, name: str) -> Self:
          """Set human-readable display name."""
          self._display_name = name
          return self

      def with_unit(self, unit: str) -> Self:
          """Set unit of measurement."""
          self._unit = unit
          return self

      def with_dimensions(self, *dims: DimensionType) -> Self:
          """Set dimensions this variable is defined over."""
          self._dimensions = list(dims)
          if DimensionType.PERIOD not in self._dimensions:
              self._dimensions.insert(0, DimensionType.PERIOD)
          return self

      def national(self) -> Self:
          """Set as national-level (Period only)."""
          self._dimensions = [DimensionType.PERIOD]
          return self

      def by_geo(self) -> Self:
          """Set as geo-level (Period + Geography)."""
          self._dimensions = [DimensionType.PERIOD, DimensionType.GEOGRAPHY]
          return self

      def by_product(self) -> Self:
          """Set as product-level (Period + Product)."""
          self._dimensions = [DimensionType.PERIOD, DimensionType.PRODUCT]
          return self

      def by_geo_and_product(self) -> Self:
          """Set as geo+product level."""
          self._dimensions = [
              DimensionType.PERIOD,
              DimensionType.GEOGRAPHY,
              DimensionType.PRODUCT,
          ]
          return self
  ```

- [ ] **Step 1.2**: Update `MediaChannelConfigBuilder` to use mixin
  ```python
  class MediaChannelConfigBuilder(VariableConfigBuilderMixin):
      def __init__(self, name: str) -> None:
          self._name = name
          self._init_variable_fields()  # From mixin
          # ... rest of init (adstock, saturation, etc.)
  ```

- [ ] **Step 1.3**: Update `ControlVariableConfigBuilder` to use mixin

- [ ] **Step 1.4**: Update `KPIConfigBuilder` to use mixin

- [ ] **Step 1.5**: Remove duplicated methods from all three classes

- [ ] **Step 1.6**: Run tests to verify: `make tests`

### Files Modified
- `src/mmm_framework/builders.py`

### Testing
```bash
make tests
uv run python -c "from mmm_framework.builders import MediaChannelConfigBuilder; print(MediaChannelConfigBuilder('test').by_geo().build())"
```

---

## 2. Data Standardization Utility

**Priority**: HIGH
**Status**: [x] **COMPLETED** (2026-01-17) - Utility created, model.py not yet updated to use it
**Estimated Impact**: Cleaner data handling, ~30 lines consolidated

### Problem

Standardization logic repeated in `model.py`:
- Lines 539-545 (y standardization)
- Lines 574-581 (controls standardization)
- Lines 2167+ (reload standardization)

### Solution

Create a `DataStandardizer` utility class.

### Implementation Steps

- [ ] **Step 2.1**: Create new file `src/mmm_framework/utils/standardization.py`
  ```python
  """Data standardization utilities."""
  from __future__ import annotations

  from dataclasses import dataclass
  from typing import TYPE_CHECKING

  import numpy as np

  if TYPE_CHECKING:
      from numpy.typing import NDArray


  @dataclass
  class StandardizationParams:
      """Parameters from standardization fit."""
      mean: float | NDArray
      std: float | NDArray

      def to_dict(self) -> dict:
          """Convert to serializable dict."""
          return {
              "mean": self.mean if isinstance(self.mean, float) else self.mean.tolist(),
              "std": self.std if isinstance(self.std, float) else self.std.tolist(),
          }

      @classmethod
      def from_dict(cls, d: dict) -> StandardizationParams:
          """Create from dict."""
          return cls(
              mean=np.array(d["mean"]) if isinstance(d["mean"], list) else d["mean"],
              std=np.array(d["std"]) if isinstance(d["std"], list) else d["std"],
          )


  class DataStandardizer:
      """Standardize data with zero mean and unit variance."""

      def __init__(self, epsilon: float = 1e-8):
          self.epsilon = epsilon
          self._params: StandardizationParams | None = None

      def fit(self, data: NDArray) -> StandardizationParams:
          """Compute standardization parameters."""
          mean = data.mean(axis=0)
          std = data.std(axis=0) + self.epsilon
          self._params = StandardizationParams(mean=mean, std=std)
          return self._params

      def transform(self, data: NDArray, params: StandardizationParams | None = None) -> NDArray:
          """Apply standardization."""
          p = params or self._params
          if p is None:
              raise ValueError("Must call fit() first or provide params")
          return (data - p.mean) / p.std

      def fit_transform(self, data: NDArray) -> tuple[NDArray, StandardizationParams]:
          """Fit and transform in one step."""
          params = self.fit(data)
          return self.transform(data, params), params

      def inverse_transform(self, data: NDArray, params: StandardizationParams | None = None) -> NDArray:
          """Reverse standardization."""
          p = params or self._params
          if p is None:
              raise ValueError("Must call fit() first or provide params")
          return data * p.std + p.mean
  ```

- [ ] **Step 2.2**: Create `src/mmm_framework/utils/__init__.py`
  ```python
  """Utility modules for MMM Framework."""
  from .standardization import DataStandardizer, StandardizationParams

  __all__ = ["DataStandardizer", "StandardizationParams"]
  ```

- [ ] **Step 2.3**: Update `model.py` to use `DataStandardizer`
  - Replace y standardization (lines 539-545)
  - Replace controls standardization (lines 574-581)
  - Update `_scaling_params` storage to use `StandardizationParams.to_dict()`

- [ ] **Step 2.4**: Update `load()` method to use `StandardizationParams.from_dict()`

- [ ] **Step 2.5**: Run tests: `make tests`

### Files Modified
- `src/mmm_framework/utils/standardization.py` (new)
- `src/mmm_framework/utils/__init__.py` (new)
- `src/mmm_framework/model.py`

### Testing
```bash
make tests
uv run python -c "from mmm_framework.utils import DataStandardizer; import numpy as np; s = DataStandardizer(); data, params = s.fit_transform(np.random.randn(100)); print(params)"
```

---

## 3. Time-Period Masking Helper

**Priority**: MEDIUM
**Status**: [x] **COMPLETED** (2026-01-17)
**Estimated Impact**: ~15 lines consolidated, improved readability

### Problem

Same masking logic in 3 methods in `model.py`:
- `compute_counterfactual_contributions()` (lines 1620-1625)
- `compute_marginal_contributions()` (lines 1771-1775)
- `what_if_scenario()` (lines 1849-1853)

```python
if time_period is not None:
    start_idx, end_idx = time_period
    time_mask = (self.time_idx >= start_idx) & (self.time_idx <= end_idx)
else:
    time_mask = np.ones(self.n_obs, dtype=bool)
```

### Solution

Extract to private helper method.

### Implementation Steps

- [ ] **Step 3.1**: Add helper method to `BayesianMMM` class
  ```python
  def _get_time_mask(self, time_period: tuple[int, int] | None) -> NDArray[np.bool_]:
      """Create boolean mask for time period filtering.

      Parameters
      ----------
      time_period : tuple[int, int] | None
          (start_idx, end_idx) inclusive range, or None for all observations.

      Returns
      -------
      NDArray[np.bool_]
          Boolean mask array of shape (n_obs,).
      """
      if time_period is not None:
          start_idx, end_idx = time_period
          return (self.time_idx >= start_idx) & (self.time_idx <= end_idx)
      return np.ones(self.n_obs, dtype=bool)
  ```

- [ ] **Step 3.2**: Update `compute_counterfactual_contributions()` to use helper

- [ ] **Step 3.3**: Update `compute_marginal_contributions()` to use helper

- [ ] **Step 3.4**: Update `what_if_scenario()` to use helper

- [ ] **Step 3.5**: Run tests: `make tests`

### Files Modified
- `src/mmm_framework/model.py`

---

## 4. HDI Calculation Utility

**Priority**: MEDIUM
**Status**: [x] **COMPLETED** (2026-01-17)
**Estimated Impact**: ~10 lines consolidated

### Problem

HDI percentile calculation repeated in `model.py`:
- `predict()` (lines 1426-1429)
- `compute_counterfactual_contributions()` (lines 1699-1700)

```python
hdi_low_pct = (1 - hdi_prob) / 2 * 100
hdi_high_pct = (1 + hdi_prob) / 2 * 100
```

### Solution

Create utility function for HDI bounds computation.

### Implementation Steps

- [ ] **Step 4.1**: Add to `src/mmm_framework/utils/statistics.py`
  ```python
  """Statistical utility functions."""
  from __future__ import annotations

  from typing import TYPE_CHECKING

  import numpy as np

  if TYPE_CHECKING:
      from numpy.typing import NDArray


  def compute_hdi_bounds(
      samples: NDArray,
      hdi_prob: float = 0.94,
      axis: int = 0,
  ) -> tuple[NDArray, NDArray]:
      """Compute highest density interval bounds.

      Parameters
      ----------
      samples : NDArray
          Sample array.
      hdi_prob : float
          Probability mass for HDI (default 0.94).
      axis : int
          Axis along which to compute percentiles.

      Returns
      -------
      tuple[NDArray, NDArray]
          (lower_bound, upper_bound) arrays.
      """
      hdi_low_pct = (1 - hdi_prob) / 2 * 100
      hdi_high_pct = (1 + hdi_prob) / 2 * 100
      return (
          np.percentile(samples, hdi_low_pct, axis=axis),
          np.percentile(samples, hdi_high_pct, axis=axis),
      )
  ```

- [ ] **Step 4.2**: Update `utils/__init__.py` to export

- [ ] **Step 4.3**: Update `predict()` method to use `compute_hdi_bounds()`

- [ ] **Step 4.4**: Update `compute_counterfactual_contributions()` to use utility

- [ ] **Step 4.5**: Run tests: `make tests`

### Files Modified
- `src/mmm_framework/utils/statistics.py` (new)
- `src/mmm_framework/utils/__init__.py`
- `src/mmm_framework/model.py`

---

## 5. Config Lookup Consolidation

**Priority**: LOW
**Status**: [x] **COMPLETED** (2026-01-18)
**Estimated Impact**: Cleaner config access pattern, reduced code duplication

### Problem

Similar lookup methods in `config.py` (lines 409-421):
- `get_media_config()`
- `get_control_config()`

Both implemented identical loop-and-compare logic.

### Solution

Created generic `_get_config_by_name()` helper method with TypeVar for type safety.
Also added new `get_variable_config()` method for unified lookup across all variable types.

### Implementation Steps

- [x] **Step 5.1**: Add TypeVar import for generic typing
  ```python
  from typing import Any, Literal, TypeVar
  T = TypeVar("T", bound="VariableConfig")
  ```

- [x] **Step 5.2**: Add generic lookup helper to `MFFConfig`
  ```python
  def _get_config_by_name(self, configs: list[T], name: str) -> T | None:
      """Generic config lookup by name."""
      for config in configs:
          if config.name == name:
              return config
      return None
  ```

- [x] **Step 5.3**: Update `get_media_config()` to use helper
  ```python
  def get_media_config(self, name: str) -> MediaChannelConfig | None:
      return self._get_config_by_name(self.media_channels, name)
  ```

- [x] **Step 5.4**: Update `get_control_config()` to use helper
  ```python
  def get_control_config(self, name: str) -> ControlVariableConfig | None:
      return self._get_config_by_name(self.controls, name)
  ```

- [x] **Step 5.5**: Add new `get_variable_config()` method for unified lookup
  ```python
  def get_variable_config(self, name: str) -> VariableConfig | None:
      """Get any variable config by name (media, control, or KPI)."""
      if self.kpi.name == name:
          return self.kpi
      return self._get_config_by_name(self.media_channels, name) or \
             self._get_config_by_name(self.controls, name)
  ```

- [x] **Step 5.6**: Write comprehensive tests (`tests/test_config_lookup.py`)
  - 26 tests covering:
    - MFFConfig lookup (6 tests)
    - New get_variable_config method (5 tests)
    - Generic helper (5 tests)
    - Backward compatibility (4 tests)
    - Edge cases (3 tests)
    - Type annotations (3 tests)
  - All tests passing

### Files Modified
- `src/mmm_framework/config.py` (added generic helper, updated lookup methods)
- `tests/test_config_lookup.py` (new - 26 tests)

---

## 6. Naming Consistency Fixes

**Priority**: MEDIUM
**Status**: [x] **COMPLETED** (2026-01-17)
**Estimated Impact**: Improved code clarity

### Problem

Several naming inconsistencies identified:

1. **`_std` suffix overloaded** - Sometimes "standard deviation", sometimes "standardized data"
2. **Builder method patterns** - Mixed `with_*`, adjective, and state patterns
3. **Contribution naming** - Inconsistent singular/plural

### Solution

Standardize naming conventions across the codebase.

### Implementation Steps

#### 6A. Fix `_std` Suffix Ambiguity

- [x] **Step 6A.1**: In `model.py`, rename standardized data variables:
  - `X_controls_std` → `X_controls_scaled` (standardized data uses `_scaled`)
  - `intercept_std`, `trend_std`, etc. → `intercept_scaled`, `trend_scaled`, etc.
  - Keep `y_std`, `control_std` for standard deviation values (correct naming)

- [x] **Step 6A.2**: Update all references to renamed variables

#### 6B. Standardize Builder Methods (Documentation Only)

- [x] **Step 6B.1**: Document current conventions in module docstring
  - Added comprehensive documentation of `with_*`, convenience, and action patterns
  - See `builders.py` module docstring for details

#### 6C. Contribution Naming

- [x] **Step 6C.1**: Standardize on plural `contributions` for collections
  - Renamed `channel_contrib_std` → `channel_contributions_scaled`
  - Renamed `control_contrib_std` → `control_contributions_scaled`

### Files Modified
- `src/mmm_framework/model.py` - Variable renaming
- `src/mmm_framework/builders.py` - Added method naming conventions documentation

---

## 7. BayesianMMM Decomposition

**Priority**: HIGH (Long-term)
**Status**: [x] **COMPLETED** (2026-01-17) - All 3 phases done
**Estimated Impact**: Major maintainability improvement

### Problem

`BayesianMMM` class is ~2300 lines with mixed responsibilities:
- Data preparation
- PyMC model building
- Prediction logic
- Analysis methods
- Serialization

### Solution

Extract focused helper classes while maintaining the public API.

### Implementation Steps

This is a large refactoring effort. Recommended approach:

#### Phase 1: Extract Serialization ✓ COMPLETED

- [x] **Step 7.1**: Create `src/mmm_framework/serialization.py`
  ```python
  class MMMSerializer:
      """Handle save/load for BayesianMMM models."""

      @classmethod
      def save(cls, model: BayesianMMM, path: str | Path, ...) -> None: ...

      @classmethod
      def load(cls, path: str | Path, panel: PanelDataset, ...) -> BayesianMMM: ...

      @classmethod
      def save_trace_only(cls, trace: az.InferenceData, path: str | Path) -> None: ...

      @classmethod
      def load_trace_only(cls, path: str | Path) -> az.InferenceData: ...
  ```

- [x] **Step 7.2**: Move `save()` and `load()` logic to `MMMSerializer`
  - Extracted metadata collection: `_collect_metadata()`
  - Extracted config collection: `_collect_configs()`
  - Extracted scaling params: `_collect_scaling_params()`, `_restore_scaling_params()`
  - Extracted trace handling: `_save_trace()`, `_load_trace()`
  - Extracted validation: `_check_version()`, `_validate_panel_compatibility()`
  - Extracted feature saving: `_save_trend_features()`, `_save_seasonality_features()`
  - Extracted feature loading: `_load_trend_features()`, `_load_seasonality_features()`

- [x] **Step 7.3**: Keep `save()` and `load()` on `BayesianMMM` as thin wrappers
  - `save()` delegates to `MMMSerializer.save()`
  - `load()` delegates to `MMMSerializer.load()`
  - `save_trace_only()` delegates to `MMMSerializer.save_trace_only()`
  - `load_trace_only()` delegates to `MMMSerializer.load_trace_only()`

#### Phase 2: Extract Data Preparation ✓ COMPLETED

- [x] **Step 7.4**: Create `src/mmm_framework/data_preparation.py`
  - `ScalingParameters` dataclass for storing standardization parameters
  - `PreparedData` dataclass for storing all prepared model data
  - `DataPreparator` class for data preparation logic
  - `standardize_array()` and `unstandardize_array()` utility functions

- [x] **Step 7.5**: Create comprehensive tests in `tests/test_data_preparation.py`
  - 17 tests covering all functionality
  - All tests passing

#### Phase 3: Extract Analysis Methods ✓ COMPLETED

- [x] **Step 7.6**: Create `src/mmm_framework/analysis.py`
  - `MarginalAnalysisResult` dataclass for marginal analysis results
  - `ScenarioResult` dataclass for what-if scenario results
  - `MMMAnalyzer` class for post-fitting analysis
  - `compute_contribution_summary()` and `compute_period_contributions()` helper functions

- [x] **Step 7.7**: Create comprehensive tests in `tests/test_analysis.py`
  - 11 tests covering all functionality
  - All tests passing

### Files Modified
- `src/mmm_framework/serialization.py` (new - ~485 lines)
- `src/mmm_framework/data_preparation.py` (new - ~400 lines)
- `src/mmm_framework/analysis.py` (new - ~300 lines)
- `src/mmm_framework/model.py` (reduced by ~260 lines)
- `tests/test_serialization.py` (new - 22 tests)
- `tests/test_data_preparation.py` (new - 17 tests)
- `tests/test_analysis.py` (new - 11 tests)

---

## 8. Transform Utilities Module

**Priority**: MEDIUM
**Status**: [x] **COMPLETED** (2026-01-17)
**Estimated Impact**: Better code organization, reusability

### Problem

Utility functions scattered in `model.py` (lines 159-272):
- `create_fourier_features()`
- `geometric_adstock_np()` / `geometric_adstock_2d()`
- `logistic_saturation_np()`
- `create_bspline_basis()`
- `create_piecewise_trend_matrix()`

### Solution

Create dedicated transform modules.

### Implementation Steps

- [x] **Step 8.1**: Create `src/mmm_framework/transforms/__init__.py`

- [x] **Step 8.2**: Create `src/mmm_framework/transforms/adstock.py`
  ```python
  """Adstock transformation functions."""

  def geometric_adstock(x: NDArray, alpha: float) -> NDArray: ...
  def geometric_adstock_2d(X: NDArray, alpha: float) -> NDArray: ...
  ```

- [x] **Step 8.3**: Create `src/mmm_framework/transforms/saturation.py`
  ```python
  """Saturation transformation functions."""

  def logistic_saturation(x: NDArray, lam: float) -> NDArray: ...
  ```

- [x] **Step 8.4**: Create `src/mmm_framework/transforms/seasonality.py`
  ```python
  """Seasonality feature creation."""

  def create_fourier_features(t: NDArray, period: float, order: int) -> NDArray: ...
  ```

- [x] **Step 8.5**: Create `src/mmm_framework/transforms/trend.py`
  ```python
  """Trend feature creation."""

  def create_bspline_basis(t: NDArray, n_knots: int, degree: int) -> NDArray: ...
  def create_piecewise_trend_matrix(t: NDArray, n_changepoints: int, changepoint_range: float) -> tuple: ...
  ```

- [x] **Step 8.6**: Update `model.py` to import from transforms module (with backward compatibility aliases)

- [x] **Step 8.7**: Run tests: `make tests` - All 146 tests passing

### Files Modified
- `src/mmm_framework/transforms/` (new directory)
- `src/mmm_framework/model.py`

---

## 9. Reporting Extractor Abstraction

**Priority**: HIGH
**Status**: [x] **COMPLETED** (2026-01-18)
**Estimated Impact**: ~500+ lines of potential deduplication

### Problem

`reporting/data_extractors.py` has 3 parallel classes (~2894 lines):
- `BayesianMMMExtractor`
- `ExtendedMMMExtractor`
- `PyMCMarketingExtractor`

Each implements similar extraction methods independently.

### Solution

Enhanced the existing `DataExtractor` ABC with shared utility methods and created
an `AggregationMixin` class for data aggregation utilities.

### Implementation Steps

- [x] **Step 9.1**: Analyze common methods across all three extractors
  - Identified: `_compute_fit_statistics`, `_compute_hdi`, aggregation methods
  - Identified: `ci_prob` property pattern needed across all extractors

- [x] **Step 9.2**: Enhance `DataExtractor` ABC
  ```python
  class DataExtractor(ABC):
      """Base class for model data extractors."""

      @property
      def ci_prob(self) -> float:
          """Credible interval probability. Override in subclass."""
          return getattr(self, '_ci_prob', 0.8)

      @abstractmethod
      def extract(self) -> MMMDataBundle: ...

      def _compute_hdi(self, samples, prob=None) -> tuple[float, float]: ...
      def _compute_percentile_bounds(self, samples, prob=None, axis=0) -> tuple: ...
      def _compute_fit_statistics(self, actual, predicted) -> dict | None: ...
      def _extract_diagnostics(self, trace) -> dict: ...
  ```

- [x] **Step 9.3**: Create `AggregationMixin` class
  ```python
  class AggregationMixin:
      """Mixin providing data aggregation utilities for extractors."""

      def _aggregate_by_period_simple(self, values, periods, unique_periods) -> np.ndarray: ...
      def _aggregate_samples_by_period(self, samples, periods, unique_periods, ci_prob) -> dict: ...
      def _aggregate_by_group(self, values, group_idx, n_groups) -> np.ndarray: ...
  ```

- [x] **Step 9.4**: Update `BayesianMMMExtractor` to inherit from both
  - Now inherits from `DataExtractor` and `AggregationMixin`
  - Removed duplicate `_compute_fit_statistics` (uses inherited version)
  - Uses `_ci_prob` property pattern

- [x] **Step 9.5**: Update `ExtendedMMMExtractor` to use enhanced base
  - Uses `_ci_prob` property pattern
  - Inherits shared methods from `DataExtractor`

- [x] **Step 9.6**: Update `PyMCMarketingExtractor` to use enhanced base
  - Uses `_ci_prob` property pattern
  - Inherits shared methods from `DataExtractor`

- [x] **Step 9.7**: Update `reporting/__init__.py` exports
  - Added: `DataExtractor`, `AggregationMixin`, `BayesianMMMExtractor`,
    `ExtendedMMMExtractor`, `PyMCMarketingExtractor`, `create_extractor`

- [x] **Step 9.8**: Write comprehensive tests (`tests/test_extractors.py`)
  - 34 tests covering all functionality
  - All tests passing

### Files Modified
- `src/mmm_framework/reporting/data_extractors.py` (enhanced base classes)
- `src/mmm_framework/reporting/__init__.py` (added exports)
- `tests/test_extractors.py` (new - 34 tests)

---

## 10. Config Hierarchy Unification

**Priority**: MEDIUM
**Status**: [x] **COMPLETED** (2026-01-18)
**Estimated Impact**: Reduced maintenance burden, single source of truth for shared enums

### Problem

Overlapping config definitions in:
- `config.py` (main) - Uses Pydantic `BaseModel` with nested `PriorConfig` objects
- `mmm_extensions/config.py` - Uses frozen `dataclass` with flat prior parameters

Both define `SaturationType`, `AdstockConfig`, `SaturationConfig`.

### Analysis

After auditing the differences:

**Can be shared (enum values identical):**
- `SaturationType` - Main config has superset (HILL, LOGISTIC, MICHAELIS_MENTEN, TANH, NONE),
  extension only used (LOGISTIC, HILL). Extension can import from main and use directly.

**Must remain separate (different structure/purpose):**
- `AdstockConfig` - Main uses Pydantic with `PriorConfig` nesting, extension uses frozen dataclass
  with flat prior params (`prior_alpha`, `prior_beta`)
- `SaturationConfig` - Same pattern: different class structures for different use cases

### Solution

Import shared enums from main config to establish single source of truth.
Keep separate dataclass-based configs in extension for their specific use cases.

### Implementation Steps

- [x] **Step 10.1**: Audit differences between main and extension configs
  - Identified: `SaturationType` can be shared (enum)
  - Identified: `AdstockConfig` and `SaturationConfig` need to stay separate
    (Pydantic vs dataclass, different field structures)

- [x] **Step 10.2**: Determine which configs can be shared vs need extension
  - Shared: `SaturationType` (enum)
  - Separate: `AdstockConfig`, `SaturationConfig` (different class systems)

- [x] **Step 10.3**: Update `mmm_extensions/config.py` to import from main config
  ```python
  # Import shared enum from main config to avoid duplication
  from mmm_framework.config import SaturationType

  # Extension-specific enums (not in main config)
  class MediatorType(str, Enum): ...
  class CrossEffectType(str, Enum): ...
  class EffectConstraint(str, Enum): ...
  ```

- [x] **Step 10.4**: Verify imports work correctly in all `mmm_extensions/` modules
  - `__init__.py`, `components.py`, `builders.py`, `models.py` all import from `.config`
  - All correctly get the unified `SaturationType` through re-export

- [x] **Step 10.5**: Write comprehensive tests (`tests/test_config_unification.py`)
  - 31 tests covering:
    - SaturationType unification (7 tests)
    - AdstockConfig separation (5 tests)
    - SaturationConfig separation (3 tests)
    - Extension-specific enums (3 tests)
    - Package exports (3 tests)
    - Backward compatibility (4 tests)
    - Config creation (3 tests)
    - Module imports (3 tests)
  - All tests passing

### Files Modified
- `src/mmm_framework/mmm_extensions/config.py` (import SaturationType from main config)
- `tests/test_config_unification.py` (new - 31 tests)

---

## Progress Summary

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| 1. Builder Method Consolidation | HIGH | [x] **COMPLETED** | `VariableConfigBuilderMixin` created |
| 2. Data Standardization Utility | HIGH | [x] **COMPLETED** | `DataStandardizer` in `utils/` |
| 3. Time-Period Masking Helper | MEDIUM | [x] **COMPLETED** | `_get_time_mask()` added |
| 4. HDI Calculation Utility | MEDIUM | [x] **COMPLETED** | `compute_hdi_bounds()` in `utils/` |
| 5. Config Lookup Consolidation | LOW | [x] **COMPLETED** | Generic `_get_config_by_name()` helper |
| 6. Naming Consistency Fixes | MEDIUM | [x] **COMPLETED** | `_scaled` suffix, builder docs |
| 7. BayesianMMM Decomposition | HIGH | [x] **COMPLETED** | All 3 phases: serialization, data_preparation, analysis |
| 8. Transform Utilities Module | MEDIUM | [x] **COMPLETED** | `transforms/` module with 4 submodules |
| 9. Reporting Extractor Abstraction | HIGH | [x] **COMPLETED** | Enhanced `DataExtractor`, created `AggregationMixin` |
| 10. Config Hierarchy Unification | MEDIUM | [x] **COMPLETED** | Shared `SaturationType` from main config |

---

## Recommended Order of Implementation

1. **Phase 1 - Quick Wins** (can be done independently)
   - Task 1: Builder Method Consolidation
   - Task 3: Time-Period Masking Helper
   - Task 4: HDI Calculation Utility

2. **Phase 2 - Utilities Infrastructure**
   - Task 2: Data Standardization Utility
   - Task 8: Transform Utilities Module

3. **Phase 3 - Naming & Consistency**
   - Task 5: Config Lookup Consolidation
   - Task 6: Naming Consistency Fixes
   - Task 10: Config Hierarchy Unification

4. **Phase 4 - Major Refactoring**
   - Task 9: Reporting Extractor Abstraction
   - Task 7: BayesianMMM Decomposition (phased)

---

## Testing Strategy

After each task:

```bash
# Run all tests
make tests

# Run fast tests during development
make fast_tests

# Format code
make format

# Verify example still works
uv run python examples/ex_model_workflow.py
```

---

## Change Log

| Date | Task | Status | Notes |
|------|------|--------|-------|
| 2026-01-17 | Initial guide created | Complete | All tasks documented |
| 2026-01-17 | Task 1: Builder Method Consolidation | Complete | Created `VariableConfigBuilderMixin`, updated 3 builders |
| 2026-01-17 | Task 2: Data Standardization Utility | Complete | Created `utils/standardization.py` with `DataStandardizer` |
| 2026-01-17 | Task 3: Time-Period Masking Helper | Complete | Added `_get_time_mask()` to BayesianMMM, updated 3 methods |
| 2026-01-17 | Task 4: HDI Calculation Utility | Complete | Created `utils/statistics.py` with `compute_hdi_bounds()`, updated model.py |
| 2026-01-17 | Tests created | Complete | 68 builder tests + 17 utility tests, all passing |
| 2026-01-17 | Task 8: Transform Utilities Module | Complete | Created `transforms/` module with adstock.py, saturation.py, seasonality.py, trend.py. Updated model.py with backward-compatible imports. 28 new tests added, all 146 tests passing |
| 2026-01-17 | Task 6: Naming Consistency Fixes | Complete | Renamed `_std` variables to `_scaled` for standardized data, documented builder conventions in module docstring |
| 2026-01-17 | Task 7: BayesianMMM Decomposition (Phase 1) | Complete | Created `serialization.py` with `MMMSerializer` class. Extracted 260+ lines from model.py. Updated save/load methods to use thin wrappers. 22 new serialization tests, all 168 tests passing |
| 2026-01-17 | Task 7: BayesianMMM Decomposition (Phase 2) | Complete | Created `data_preparation.py` with `DataPreparator`, `ScalingParameters`, `PreparedData` classes and standardization utilities. 17 new tests, all passing |
| 2026-01-17 | Task 7: BayesianMMM Decomposition (Phase 3) | Complete | Created `analysis.py` with `MMMAnalyzer`, `MarginalAnalysisResult`, `ScenarioResult` classes and helper functions. 11 new tests, all 196 refactoring tests passing |
| 2026-01-18 | Task 9: Reporting Extractor Abstraction | Complete | Enhanced `DataExtractor` base class with `_compute_fit_statistics`, `_compute_percentile_bounds`, and `ci_prob` property. Created `AggregationMixin` for data aggregation utilities. Updated all 3 extractor classes. 34 new tests, all 230 refactoring tests passing |
| 2026-01-18 | Task 10: Config Hierarchy Unification | Complete | Unified `SaturationType` enum by importing from main config into extension config. Kept `AdstockConfig` and `SaturationConfig` separate (different class systems: Pydantic vs dataclass). 31 new tests verifying unification and backward compatibility, all 310 refactoring tests passing |
| 2026-01-18 | Task 5: Config Lookup Consolidation | Complete | Created generic `_get_config_by_name()` helper with TypeVar. Updated `get_media_config()` and `get_control_config()` to use helper. Added new `get_variable_config()` for unified lookup. 26 new tests, all 331 refactoring tests passing |

