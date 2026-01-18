"""
Base chart utilities and common functions for MMM reporting.

Contains JSON encoding, color conversion, date formatting, and Plotly div creation.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Handle datetime64 arrays
            if np.issubdtype(obj.dtype, np.datetime64):
                return [str(d) for d in obj]
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        # Handle numpy datetime64 scalar
        if isinstance(obj, np.datetime64):
            return str(obj)
        # Handle numpy generic types (includes datetime64)
        if isinstance(obj, np.generic):
            return obj.item() if hasattr(obj, "item") else str(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        # Catch-all for datetime-like types
        type_name = type(obj).__name__
        if "datetime" in type_name.lower():
            return str(obj)
        return super().default(obj)


def _to_json(data: Any) -> str:
    """Convert data to JSON string for Plotly."""
    return json.dumps(data, cls=NumpyEncoder)


def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex color to RGB string for Plotly rgba()."""
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"{r}, {g}, {b}"


def _dates_to_strings(dates: list | np.ndarray | pd.DatetimeIndex) -> list[str]:
    """Convert dates to string format for JSON serialization."""
    if len(dates) == 0:
        return []
    if hasattr(dates, "strftime"):
        # pandas DatetimeIndex
        return [d.strftime("%Y-%m-%d") for d in dates]
    elif hasattr(dates[0], "strftime"):
        # List of Timestamps/datetime objects
        return [d.strftime("%Y-%m-%d") for d in dates]
    elif isinstance(dates[0], np.datetime64):
        # numpy datetime64
        return [str(d)[:10] for d in dates]
    else:
        # Already strings or other format
        return [str(d) for d in dates]


def create_plotly_div(
    traces: list[dict],
    layout: dict,
    div_id: str,
    config: dict | None = None,
) -> str:
    """Create an HTML div with embedded Plotly chart."""
    config = config or {"displayModeBar": False, "responsive": True}

    return f"""
    <div id="{div_id}" class="chart-container"></div>
    <script>
        Plotly.newPlot(
            "{div_id}",
            {_to_json(traces)},
            {_to_json(layout)},
            {_to_json(config)}
        );
    </script>
    """


def _generate_dimension_colors(
    names: list[str] | None,
    colors: Any,
    offset: int = 0,
) -> dict[str, str]:
    """Generate distinct colors for dimension values."""
    if not names:
        return {}

    # Color palette for dimensions
    palette = [
        "#3498db",
        "#e74c3c",
        "#2ecc71",
        "#9b59b6",
        "#f39c12",
        "#1abc9c",
        "#e67e22",
        "#34495e",
        "#16a085",
        "#c0392b",
        "#8e44ad",
        "#27ae60",
        "#d35400",
        "#2980b9",
        "#7f8c8d",
    ]

    result = {}
    for i, name in enumerate(names):
        result[name] = palette[(i + offset) % len(palette)]
    return result


def _build_dimension_filter_html(
    div_id: str,
    geo_names: list[str] | None,
    product_names: list[str] | None,
    geo_colors: dict[str, str],
    product_colors: dict[str, str],
) -> str:
    """Build HTML for dropdown dimension filters."""
    if not geo_names and not product_names:
        return ""

    filter_groups = []

    # Geo dropdown
    if geo_names:
        geo_options = '<option value="agg" selected>Aggregated (Total)</option>'
        for geo in geo_names:
            geo_options += f'<option value="{geo}">{geo}</option>'
        filter_groups.append(
            f"""
        <div class="filter-group">
            <label for="{div_id}_geo_select">View:</label>
            <select id="{div_id}_geo_select" onchange="updateDimensionFilter_{div_id}()">
                {geo_options}
            </select>
        </div>
        """
        )

    # Product dropdown
    if product_names:
        prod_options = '<option value="agg" selected>Aggregated (Total)</option>'
        for product in product_names:
            prod_options += f'<option value="{product}">{product}</option>'
        filter_groups.append(
            f"""
        <div class="filter-group">
            <label for="{div_id}_product_select">Product:</label>
            <select id="{div_id}_product_select" onchange="updateDimensionFilter_{div_id}()">
                {prod_options}
            </select>
        </div>
        """
        )

    return f"""
    <style>
        .dimension-filter-container {{ margin-bottom: 1rem; }}
        .dimension-filters {{ display: flex; gap: 1.5rem; margin-bottom: 1rem; flex-wrap: wrap; align-items: center; }}
        .filter-group {{ display: flex; align-items: center; gap: 0.5rem; }}
        .filter-group label {{ font-size: 0.9rem; font-weight: 500; color: var(--color-text-muted, #666); }}
        .filter-group select {{
            padding: 0.4rem 0.75rem;
            border: 1px solid var(--color-border, #e0e0e0);
            border-radius: 4px;
            font-size: 0.9rem;
            background: white;
            cursor: pointer;
            min-width: 180px;
        }}
        .filter-group select:focus {{ outline: none; border-color: var(--color-primary, #3498db); }}
    </style>
    <div class="dimension-filters">
        {''.join(filter_groups)}
    </div>
    """


def _build_dimension_filter_js(
    div_id: str,
    trace_metadata: list[dict],
    has_geo: bool,
    has_product: bool,
) -> str:
    """Build JavaScript for dropdown dimension filter functionality."""
    metadata_json = _to_json(trace_metadata)

    return f"""
    <script>
    (function() {{
        var traceMetadata_{div_id} = {metadata_json};

        window.updateDimensionFilter_{div_id} = function() {{
            var visibility = [];
            var metadata = traceMetadata_{div_id};

            // Get selected values from dropdowns
            var geoSelect = document.getElementById('{div_id}_geo_select');
            var productSelect = document.getElementById('{div_id}_product_select');

            var selectedGeo = geoSelect ? geoSelect.value : 'agg';
            var selectedProduct = productSelect ? productSelect.value : 'agg';

            // Determine visibility for each trace
            for (var i = 0; i < metadata.length; i++) {{
                var m = metadata[i];
                var show = false;

                if (m.type === 'agg') {{
                    // Show aggregated only when both dropdowns are set to "agg"
                    show = (selectedGeo === 'agg') && (selectedProduct === 'agg');
                }} else if (m.dim === 'geo') {{
                    // Show this geo's traces if it's selected
                    show = (m.value === selectedGeo);
                }} else if (m.dim === 'product') {{
                    // Show this product's traces if it's selected
                    show = (m.value === selectedProduct);
                }}

                visibility.push(show);
            }}

            Plotly.restyle('{div_id}', {{'visible': visibility}});
        }};
    }})();
    </script>
    """


__all__ = [
    "NumpyEncoder",
    "_to_json",
    "_hex_to_rgb",
    "_dates_to_strings",
    "create_plotly_div",
    "_generate_dimension_colors",
    "_build_dimension_filter_html",
    "_build_dimension_filter_js",
]
