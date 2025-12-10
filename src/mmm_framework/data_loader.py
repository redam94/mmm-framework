"""
Data loading and transformation utilities for MFF format.

Handles variable-dimension data, dimension alignment, and panel construction.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .config import (
    AllocationMethod,
    DimensionType,
    MediaChannelConfig,
    MFFConfig,
    VariableConfig,
    VariableRole,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Data validation
# =============================================================================

class MFFValidationError(Exception):
    """Raised when MFF data fails validation."""
    pass


def validate_mff_structure(df: pd.DataFrame, config: MFFConfig) -> list[str]:
    """
    Validate MFF dataframe structure against config.
    
    Returns list of warnings (empty if clean).
    Raises MFFValidationError for critical issues.
    """
    warnings_list = []
    cols = config.columns
    
    # Check required columns exist
    missing_cols = set(cols.all_columns) - set(df.columns)
    if missing_cols:
        raise MFFValidationError(f"Missing required columns: {missing_cols}")
    
    # Check for expected variables
    actual_vars = set(df[cols.variable_name].unique())
    expected_vars = set(config.variable_names)
    
    missing_vars = expected_vars - actual_vars
    if missing_vars:
        raise MFFValidationError(f"Missing expected variables: {missing_vars}")
    
    extra_vars = actual_vars - expected_vars
    if extra_vars:
        warnings_list.append(f"Extra variables in data (will be ignored): {extra_vars}")
    
    # Check for nulls in key columns
    null_counts = df[cols.all_columns].isnull().sum()
    if null_counts[cols.variable_value] > 0:
        warnings_list.append(
            f"Found {null_counts[cols.variable_value]} null values in {cols.variable_value}"
        )
    
    # Validate date parsing
    try:
        pd.to_datetime(df[cols.period].iloc[0], format=config.date_format)
    except Exception as e:
        raise MFFValidationError(
            f"Cannot parse dates with format '{config.date_format}': {e}"
        )
    
    return warnings_list


def validate_variable_dimensions(
    df: pd.DataFrame,
    var_config: VariableConfig,
    config: MFFConfig,
) -> tuple[bool, str]:
    """
    Validate that a variable's data matches its configured dimensions.
    
    Returns (is_valid, message).
    """
    cols = config.columns
    var_data = df[df[cols.variable_name] == var_config.name]
    
    if var_data.empty:
        return False, f"No data found for variable '{var_config.name}'"
    
    # Map dimension types to column names
    dim_col_map = {
        DimensionType.PERIOD: cols.period,
        DimensionType.GEOGRAPHY: cols.geography,
        DimensionType.PRODUCT: cols.product,
        DimensionType.CAMPAIGN: cols.campaign,
        DimensionType.OUTLET: cols.outlet,
        DimensionType.CREATIVE: cols.creative,
    }
    
    # Check which dimensions have variation
    active_dims = []
    for dim_type in DimensionType:
        col = dim_col_map[dim_type]
        unique_vals = var_data[col].dropna().unique()
        # Consider dimension active if more than one unique non-null value
        # or if it's Period (always active)
        if len(unique_vals) > 1 or dim_type == DimensionType.PERIOD:
            active_dims.append(dim_type)
    
    configured_dims = set(var_config.dimensions)
    actual_dims = set(active_dims)
    
    # Period should always be present
    if DimensionType.PERIOD not in actual_dims:
        return False, f"Variable '{var_config.name}' has no time variation"
    
    # Check for unexpected dimensions
    extra_dims = actual_dims - configured_dims - {DimensionType.PERIOD}
    if extra_dims and DimensionType.PERIOD in configured_dims:
        # Only warn if we expected Period-only but got more
        if configured_dims == {DimensionType.PERIOD}:
            return False, (
                f"Variable '{var_config.name}' has unexpected dimensions: {extra_dims}. "
                f"Configured for: {configured_dims}"
            )
    
    return True, "OK"


# =============================================================================
# Panel dataset container
# =============================================================================

@dataclass
class PanelCoordinates:
    """Coordinate labels for panel dimensions."""
    
    periods: pd.DatetimeIndex
    geographies: list[str] | None = None
    products: list[str] | None = None
    channels: list[str] = field(default_factory=list)
    controls: list[str] = field(default_factory=list)
    
    @property
    def has_geo(self) -> bool:
        return self.geographies is not None and len(self.geographies) > 0
    
    @property
    def has_product(self) -> bool:
        return self.products is not None and len(self.products) > 0
    
    @property
    def n_periods(self) -> int:
        return len(self.periods)
    
    @property
    def n_geos(self) -> int:
        return len(self.geographies) if self.geographies else 1
    
    @property
    def n_products(self) -> int:
        return len(self.products) if self.products else 1
    
    @property
    def n_obs(self) -> int:
        """Total number of observations."""
        return self.n_periods * self.n_geos * self.n_products
    
    def to_pymc_coords(self) -> dict[str, list]:
        """Convert to PyMC coordinate dictionary."""
        coords = {
            "date": self.periods.tolist(),
            "channel": self.channels,
        }
        if self.has_geo:
            coords["geo"] = self.geographies
        if self.has_product:
            coords["product"] = self.products
        if self.controls:
            coords["control"] = self.controls
        return coords


@dataclass
class PanelDataset:
    """
    Structured panel data ready for modeling.
    
    All arrays are aligned to the same index structure.
    """
    
    # Target variable
    y: pd.Series
    
    # Feature matrices
    X_media: pd.DataFrame
    X_controls: pd.DataFrame | None
    
    # Coordinate information
    coords: PanelCoordinates
    
    # Index for reconstruction
    index: pd.MultiIndex | pd.DatetimeIndex
    
    # Original config reference
    config: MFFConfig
    
    # Metadata
    media_stats: dict[str, dict] = field(default_factory=dict)
    
    @property
    def n_obs(self) -> int:
        return len(self.y)
    
    @property
    def n_channels(self) -> int:
        return self.X_media.shape[1] if self.X_media is not None else 0
    
    @property
    def n_controls(self) -> int:
        return self.X_controls.shape[1] if self.X_controls is not None else 0
    
    @property
    def is_panel(self) -> bool:
        """True if data has geo or product dimensions."""
        return self.coords.has_geo or self.coords.has_product
    
    def to_numpy(self) -> tuple[NDArray, NDArray, NDArray | None]:
        """Convert to numpy arrays for modeling."""
        y = self.y.values
        X_media = self.X_media.values
        X_controls = self.X_controls.values if self.X_controls is not None else None
        return y, X_media, X_controls
    
    def get_media_for_channel(self, channel: str) -> pd.Series:
        """Get media series for a specific channel."""
        if channel not in self.X_media.columns:
            raise KeyError(f"Channel '{channel}' not found. Available: {list(self.X_media.columns)}")
        return self.X_media[channel]
    
    def compute_spend_shares(self) -> pd.Series:
        """Compute share of total spend for each channel."""
        totals = self.X_media.sum()
        return totals / totals.sum()
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "PanelDataset Summary",
            "=" * 40,
            f"Observations: {self.n_obs}",
            f"Time periods: {self.coords.n_periods}",
            f"Geographies: {self.coords.n_geos}",
            f"Products: {self.coords.n_products}",
            f"Media channels: {self.n_channels}",
            f"Control variables: {self.n_controls}",
            "",
            "Target (y) stats:",
            f"  Mean: {self.y.mean():.2f}",
            f"  Std:  {self.y.std():.2f}",
            f"  Min:  {self.y.min():.2f}",
            f"  Max:  {self.y.max():.2f}",
        ]
        
        if self.n_channels > 0:
            lines.append("")
            lines.append("Media channel totals:")
            for col in self.X_media.columns:
                lines.append(f"  {col}: {self.X_media[col].sum():,.0f}")
        
        return "\n".join(lines)


# =============================================================================
# MFF Loader and Parser
# =============================================================================

class MFFLoader:
    """
    Loads and parses MFF format data into panel structure.
    
    Handles:
    - Variable extraction by name
    - Dimension alignment (national to geo, etc.)
    - Missing value handling
    - Date parsing and frequency alignment
    """
    
    def __init__(self, config: MFFConfig):
        self.config = config
        self._raw_data: pd.DataFrame | None = None
        self._allocation_weights: dict[str, pd.Series] = {}
    
    def load(self, data: pd.DataFrame | str) -> MFFLoader:
        """
        Load MFF data from DataFrame or file path.
        
        Parameters
        ----------
        data : pd.DataFrame or str
            Either a DataFrame in MFF format or path to CSV/parquet file.
        
        Returns
        -------
        self : MFFLoader
            For method chaining.
        """
        if isinstance(data, str):
            if data.endswith('.parquet'):
                self._raw_data = pd.read_parquet(data)
            else:
                self._raw_data = pd.read_csv(data)
        else:
            self._raw_data = data.copy()
        
        # Validate structure
        warnings_list = validate_mff_structure(self._raw_data, self.config)
        for w in warnings_list:
            warnings.warn(w)
        
        # Parse dates
        cols = self.config.columns
        self._raw_data[cols.period] = pd.to_datetime(
            self._raw_data[cols.period],
            format=self.config.date_format
        )
        
        return self
    
    def set_allocation_weights(
        self,
        dimension: DimensionType,
        weights: pd.Series | dict[str, float],
    ) -> MFFLoader:
        """
        Set custom allocation weights for dimension disaggregation.
        
        Parameters
        ----------
        dimension : DimensionType
            The dimension to set weights for (GEOGRAPHY or PRODUCT).
        weights : pd.Series or dict
            Weights indexed by dimension level (e.g., geo names).
            Will be normalized to sum to 1.
        
        Returns
        -------
        self : MFFLoader
            For method chaining.
        """
        if isinstance(weights, dict):
            weights = pd.Series(weights)
        
        # Normalize
        weights = weights / weights.sum()
        self._allocation_weights[dimension.value] = weights
        
        return self
    
    def _extract_variable(
        self,
        var_config: VariableConfig,
    ) -> pd.DataFrame:
        """Extract a single variable from MFF data."""
        cols = self.config.columns
        
        # Filter to this variable
        mask = self._raw_data[cols.variable_name] == var_config.name
        var_data = self._raw_data[mask].copy()
        
        if var_data.empty:
            raise MFFValidationError(f"No data found for variable '{var_config.name}'")
        
        # Build dimension columns based on config
        dim_cols = [cols.period]  # Always include period
        
        if DimensionType.GEOGRAPHY in var_config.dimensions:
            dim_cols.append(cols.geography)
        if DimensionType.PRODUCT in var_config.dimensions:
            dim_cols.append(cols.product)
        
        # Handle split dimensions for media
        if isinstance(var_config, MediaChannelConfig):
            if DimensionType.OUTLET in var_config.split_dimensions:
                dim_cols.append(cols.outlet)
            if DimensionType.CAMPAIGN in var_config.split_dimensions:
                dim_cols.append(cols.campaign)
        
        # Aggregate to configured dimensions
        result = var_data.groupby(dim_cols, as_index=False)[cols.variable_value].sum()
        
        return result
    
    def _get_allocation_weights(
        self,
        dimension: DimensionType,
        levels: list[str],
    ) -> pd.Series:
        """Get allocation weights for a dimension."""
        dim_name = dimension.value
        
        # Check for custom weights
        if dim_name in self._allocation_weights:
            weights = self._allocation_weights[dim_name]
            # Ensure all levels are present
            missing = set(levels) - set(weights.index)
            if missing:
                warnings.warn(
                    f"Missing allocation weights for {missing}. Using equal weights."
                )
                return pd.Series(1.0 / len(levels), index=levels)
            return weights.loc[levels]
        
        # Check config for method
        if dimension == DimensionType.GEOGRAPHY:
            method = self.config.alignment.geo_allocation
            weight_var = self.config.alignment.geo_weight_variable
        elif dimension == DimensionType.PRODUCT:
            method = self.config.alignment.product_allocation
            weight_var = self.config.alignment.product_weight_variable
        else:
            method = AllocationMethod.EQUAL
            weight_var = None
        
        if method == AllocationMethod.EQUAL:
            return pd.Series(1.0 / len(levels), index=levels)
        
        elif method == AllocationMethod.SALES and self._raw_data is not None:
            # Use KPI totals as weights
            cols = self.config.columns
            kpi_data = self._raw_data[
                self._raw_data[cols.variable_name] == self.config.kpi.name
            ]
            dim_col = cols.geography if dimension == DimensionType.GEOGRAPHY else cols.product
            totals = kpi_data.groupby(dim_col)[cols.variable_value].sum()
            weights = totals / totals.sum()
            return weights.reindex(levels).fillna(1.0 / len(levels))
        
        elif method == AllocationMethod.CUSTOM and weight_var:
            # Extract weight variable from data
            cols = self.config.columns
            weight_data = self._raw_data[
                self._raw_data[cols.variable_name] == weight_var
            ]
            dim_col = cols.geography if dimension == DimensionType.GEOGRAPHY else cols.product
            weights = weight_data.groupby(dim_col)[cols.variable_value].sum()
            weights = weights / weights.sum()
            return weights.reindex(levels).fillna(1.0 / len(levels))
        
        # Default to equal
        return pd.Series(1.0 / len(levels), index=levels)
    
    def _align_to_target_dimensions(
        self,
        var_data: pd.DataFrame,
        var_config: VariableConfig,
        target_index: pd.MultiIndex | pd.DatetimeIndex,
    ) -> pd.Series:
        """
        Align variable data to target KPI dimensions.
        
        Handles:
        - Disaggregation (national → geo, national → geo+product)
        - Aggregation (product-geo → geo)
        - Direct mapping (same dimensions)
        """
        cols = self.config.columns
        target_dims = self.config.kpi.dimensions
        var_dims = var_config.dimensions
        
        # Get dimension names for indexing
        target_has_geo = DimensionType.GEOGRAPHY in target_dims
        target_has_product = DimensionType.PRODUCT in target_dims
        var_has_geo = DimensionType.GEOGRAPHY in var_dims
        var_has_product = DimensionType.PRODUCT in var_dims
        
        # Build index columns
        var_index_cols = [cols.period]
        if var_has_geo:
            var_index_cols.append(cols.geography)
        if var_has_product:
            var_index_cols.append(cols.product)
        
        # Set index on variable data
        var_series = var_data.set_index(var_index_cols)[cols.variable_value]
        
        # Case 1: Same dimensions - direct reindex
        if set(var_dims) == set(target_dims):
            return var_series.reindex(target_index).fillna(0)
        
        # Case 2: Disaggregation needed - build full cross-product expansion
        expand_geos = target_has_geo and not var_has_geo
        expand_products = target_has_product and not var_has_product
        
        if expand_geos or expand_products:
            # Get the periods from original data
            if isinstance(var_series.index, pd.MultiIndex):
                periods = var_series.index.get_level_values(cols.period).unique()
            else:
                periods = var_series.index.unique()
            
            # Build expansion dimensions and weights
            expand_dims = []
            expand_weights = []
            
            if expand_geos:
                geo_levels = target_index.get_level_values(cols.geography).unique().tolist()
                geo_weights = self._get_allocation_weights(DimensionType.GEOGRAPHY, geo_levels)
                expand_dims.append((cols.geography, geo_levels))
                expand_weights.append(geo_weights)
            
            if expand_products:
                product_levels = target_index.get_level_values(cols.product).unique().tolist()
                product_weights = self._get_allocation_weights(DimensionType.PRODUCT, product_levels)
                expand_dims.append((cols.product, product_levels))
                expand_weights.append(product_weights)
            
            # Create expanded data
            expanded_records = []
            
            for period in periods:
                # Get base value for this period
                if isinstance(var_series.index, pd.MultiIndex):
                    base_val = var_series.xs(period, level=cols.period)
                    if isinstance(base_val, pd.Series):
                        base_val = base_val.iloc[0]
                else:
                    base_val = var_series.loc[period]
                
                # Generate all combinations
                if len(expand_dims) == 1:
                    dim_name, levels = expand_dims[0]
                    weights = expand_weights[0]
                    for level in levels:
                        record = {cols.period: period, dim_name: level}
                        record["value"] = base_val * weights[level]
                        expanded_records.append(record)
                elif len(expand_dims) == 2:
                    dim1_name, levels1 = expand_dims[0]
                    dim2_name, levels2 = expand_dims[1]
                    weights1, weights2 = expand_weights
                    for level1 in levels1:
                        for level2 in levels2:
                            record = {
                                cols.period: period,
                                dim1_name: level1,
                                dim2_name: level2,
                            }
                            record["value"] = base_val * weights1[level1] * weights2[level2]
                            expanded_records.append(record)
            
            # Convert to series with proper index
            expanded_df = pd.DataFrame(expanded_records)
            
            # Build index in same order as target
            index_cols = []
            for name in target_index.names:
                if name in expanded_df.columns:
                    index_cols.append(name)
            
            var_series = expanded_df.set_index(index_cols)["value"]
        
        # Case 3: Aggregation needed (less common)
        if var_has_geo and not target_has_geo:
            # Aggregate across geo
            var_series = var_series.groupby(level=cols.period).sum()
        
        if var_has_product and not target_has_product:
            # Aggregate across product
            group_levels = [n for n in var_series.index.names if n != cols.product]
            var_series = var_series.groupby(level=group_levels).sum()
        
        # Final reindex to target
        return var_series.reindex(target_index).fillna(0)
    
    def build_panel(self) -> PanelDataset:
        """
        Build complete panel dataset from loaded MFF data.
        
        Returns
        -------
        PanelDataset
            Structured data ready for modeling.
        """
        if self._raw_data is None:
            raise RuntimeError("No data loaded. Call load() first.")
        
        cols = self.config.columns
        
        # 1. Extract and build KPI (target) index
        kpi_data = self._extract_variable(self.config.kpi)
        
        # Build index based on KPI dimensions
        index_cols = [cols.period]
        if self.config.kpi.has_geo:
            index_cols.append(cols.geography)
        if self.config.kpi.has_product:
            index_cols.append(cols.product)
        
        kpi_data = kpi_data.sort_values(index_cols)
        
        if len(index_cols) > 1:
            target_index = pd.MultiIndex.from_frame(kpi_data[index_cols])
        else:
            target_index = pd.DatetimeIndex(kpi_data[cols.period])
        
        y = pd.Series(
            kpi_data[cols.variable_value].values,
            index=target_index,
            name=self.config.kpi.name
        )
        
        # Apply log transform if multiplicative
        if self.config.kpi.log_transform:
            y = np.log(y.clip(lower=self.config.kpi.floor_value))
        
        # 2. Build coordinates
        periods = kpi_data[cols.period].unique()
        periods = pd.DatetimeIndex(sorted(periods))
        
        geos = None
        if self.config.kpi.has_geo:
            geos = sorted(kpi_data[cols.geography].unique().tolist())
        
        products = None
        if self.config.kpi.has_product:
            products = sorted(kpi_data[cols.product].unique().tolist())
        
        coords = PanelCoordinates(
            periods=periods,
            geographies=geos,
            products=products,
            channels=self.config.media_names,
            controls=self.config.control_names,
        )
        
        # 3. Build media matrix
        media_series = {}
        media_stats = {}
        
        for media_config in self.config.media_channels:
            var_data = self._extract_variable(media_config)
            aligned = self._align_to_target_dimensions(
                var_data, media_config, target_index
            )
            
            # Fill missing media with configured value
            aligned = aligned.fillna(self.config.fill_missing_media)
            
            media_series[media_config.name] = aligned
            media_stats[media_config.name] = {
                "total": float(aligned.sum()),
                "mean": float(aligned.mean()),
                "std": float(aligned.std()),
                "nonzero_pct": float((aligned > 0).mean()),
            }
        
        X_media = pd.DataFrame(media_series)
        X_media.index = target_index
        
        # 4. Build control matrix
        X_controls = None
        if self.config.controls:
            control_series = {}
            
            for control_config in self.config.controls:
                var_data = self._extract_variable(control_config)
                aligned = self._align_to_target_dimensions(
                    var_data, control_config, target_index
                )
                
                # Fill missing controls
                if self.config.fill_missing_controls is not None:
                    aligned = aligned.fillna(self.config.fill_missing_controls)
                else:
                    aligned = aligned.ffill().bfill()
                
                control_series[control_config.name] = aligned
            
            X_controls = pd.DataFrame(control_series)
            X_controls.index = target_index
        
        # 5. Construct panel dataset
        panel = PanelDataset(
            y=y,
            X_media=X_media,
            X_controls=X_controls,
            coords=coords,
            index=target_index,
            config=self.config,
            media_stats=media_stats,
        )
        
        return panel


# =============================================================================
# Convenience functions
# =============================================================================

def load_mff(
    data: pd.DataFrame | str,
    config: MFFConfig,
    geo_weights: pd.Series | dict | None = None,
    product_weights: pd.Series | dict | None = None,
) -> PanelDataset:
    """
    Convenience function to load MFF data in one call.
    
    Parameters
    ----------
    data : pd.DataFrame or str
        MFF data or path to file.
    config : MFFConfig
        Configuration specifying variables and dimensions.
    geo_weights : pd.Series or dict, optional
        Custom weights for geo allocation.
    product_weights : pd.Series or dict, optional
        Custom weights for product allocation.
    
    Returns
    -------
    PanelDataset
        Ready-to-use panel data.
    
    Examples
    --------
    >>> config = create_simple_mff_config(
    ...     kpi_name="Sales",
    ...     media_names=["TV", "Digital", "Social"],
    ...     control_names=["Price", "Distribution"],
    ...     kpi_dimensions=[DimensionType.PERIOD, DimensionType.GEOGRAPHY],
    ... )
    >>> panel = load_mff("data.csv", config)
    >>> print(panel.summary())
    """
    loader = MFFLoader(config)
    loader.load(data)
    
    if geo_weights is not None:
        loader.set_allocation_weights(DimensionType.GEOGRAPHY, geo_weights)
    
    if product_weights is not None:
        loader.set_allocation_weights(DimensionType.PRODUCT, product_weights)
    
    return loader.build_panel()


def mff_from_wide_format(
    df: pd.DataFrame,
    period_col: str,
    value_columns: dict[str, str],
    geo_col: str | None = None,
    product_col: str | None = None,
) -> pd.DataFrame:
    """
    Convert wide-format data to MFF format.
    
    Parameters
    ----------
    df : pd.DataFrame
        Wide format dataframe with one column per variable.
    period_col : str
        Name of the date/period column.
    value_columns : dict
        Mapping of column names to variable names.
        E.g., {"sales": "Sales", "tv_spend": "TV"}
    geo_col : str, optional
        Name of geography column.
    product_col : str, optional
        Name of product column.
    
    Returns
    -------
    pd.DataFrame
        Data in MFF format.
    """
    records = []
    
    for _, row in df.iterrows():
        base_record = {
            "Period": row[period_col],
            "Geography": row.get(geo_col, "") if geo_col else "",
            "Product": row.get(product_col, "") if product_col else "",
            "Campaign": "",
            "Outlet": "",
            "Creative": "",
        }
        
        for col_name, var_name in value_columns.items():
            record = base_record.copy()
            record["VariableName"] = var_name
            record["VariableValue"] = row[col_name]
            records.append(record)
    
    return pd.DataFrame(records)