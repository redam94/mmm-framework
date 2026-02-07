"""
Model fit chart functions for MMM reporting.

Contains actual vs predicted visualizations with geo/product selectors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ChartConfig, ColorScheme, ReportConfig
from .base import (
    _build_dimension_filter_html,
    _build_dimension_filter_js,
    _dates_to_strings,
    _generate_dimension_colors,
    _hex_to_rgb,
    _to_json,
    create_plotly_div,
)


def create_model_fit_chart(
    dates: list | np.ndarray | pd.DatetimeIndex,
    actual: np.ndarray,
    predicted_mean: np.ndarray,
    predicted_lower: np.ndarray,
    predicted_upper: np.ndarray,
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "modelFitChart",
) -> str:
    """
    Create model fit visualization showing actual vs predicted with uncertainty.

    Parameters
    ----------
    dates : array-like
        Time index for observations
    actual : ndarray
        Observed KPI values
    predicted_mean : ndarray
        Posterior mean predictions
    predicted_lower : ndarray
        Lower bound of credible interval
    predicted_upper : ndarray
        Upper bound of credible interval
    config : ReportConfig
        Report configuration
    chart_config : ChartConfig, optional
        Chart-specific configuration
    div_id : str
        HTML div ID for the chart

    Returns
    -------
    str
        HTML string with embedded Plotly chart
    """
    chart_config = chart_config or ChartConfig(
        height=400,
        y_title="Revenue",
        show_credible_intervals=True,
    )

    colors = config.color_scheme

    # Convert dates to strings for JSON serialization
    date_strings = [str(d) for d in dates]

    traces = [
        # Credible interval band
        {
            "type": "scatter",
            "x": date_strings + date_strings[::-1],
            "y": list(predicted_upper) + list(predicted_lower[::-1]),
            "fill": "toself",
            "fillcolor": f"rgba({_hex_to_rgb(colors.accent)}, {chart_config.ci_alpha})",
            "line": {"color": "transparent"},
            "name": f"{int(chart_config.ci_level * 100)}% CI",
            "hoverinfo": "skip",
        },
        # Predicted mean line
        {
            "type": "scatter",
            "x": date_strings,
            "y": list(predicted_mean),
            "mode": "lines",
            "name": "Predicted",
            "line": {"color": colors.accent, "width": 2},
            "hovertemplate": "Predicted: %{y:,.0f}<extra></extra>",
        },
        # Actual values
        {
            "type": "scatter",
            "x": date_strings,
            "y": list(actual),
            "mode": "markers",
            "name": "Actual",
            "marker": {"color": colors.primary_dark, "size": 6},
            "hovertemplate": "Actual: %{y:,.0f}<extra></extra>",
        },
    ]

    layout = chart_config.to_plotly_layout(colors)
    layout["title"] = {"text": "Model Fit: Actual vs Predicted", "font": {"size": 16}}

    return create_plotly_div(traces, layout, div_id)


def create_model_fit_chart_with_geo_selector(
    dates: list | np.ndarray | pd.DatetimeIndex,
    actual_agg: np.ndarray,
    predicted_agg: dict[str, np.ndarray],  # {"mean", "lower", "upper"}
    actual_by_geo: dict[str, np.ndarray] | None = None,
    predicted_by_geo: dict[str, dict[str, np.ndarray]] | None = None,
    geo_names: list[str] | None = None,
    config: ReportConfig = None,
    chart_config: ChartConfig | None = None,
    div_id: str = "modelFitChart",
) -> str:
    """
    Create model fit visualization with geo selector dropdown.

    Parameters
    ----------
    dates : array-like
        Time index for x-axis
    actual_agg : ndarray
        Aggregated observed values (sum over all geos)
    predicted_agg : dict
        Aggregated predictions with keys "mean", "lower", "upper"
    actual_by_geo : dict, optional
        Per-geo observed values: {geo_name: ndarray}
    predicted_by_geo : dict, optional
        Per-geo predictions: {geo_name: {"mean", "lower", "upper"}}
    geo_names : list, optional
        List of geography names
    config : ReportConfig
        Report configuration
    chart_config : ChartConfig, optional
        Chart-specific configuration
    div_id : str
        HTML div ID for the chart

    Returns
    -------
    str
        HTML string with embedded Plotly chart and dropdown
    """
    chart_config = chart_config or ChartConfig(height=400, ci_level=0.8)
    colors = config.color_scheme if config else ColorScheme()

    # Convert dates to string format for JSON
    dates_str = _dates_to_strings(dates)

    traces = []

    # =========================================================================
    # AGGREGATED TRACES (visible by default) - 3 traces
    # =========================================================================

    # Trace 0: Uncertainty band (fill)
    traces.append(
        {
            "type": "scatter",
            "x": dates_str + dates_str[::-1],
            "y": list(predicted_agg["upper"]) + list(predicted_agg["lower"])[::-1],
            "fill": "toself",
            "fillcolor": f"rgba({_hex_to_rgb(colors.primary)}, 0.2)",
            "line": {"width": 0},
            "name": f"{int(chart_config.ci_level * 100)}% CI",
            "showlegend": True,
            "hoverinfo": "skip",
            "visible": True,
        }
    )

    # Trace 1: Predicted mean
    traces.append(
        {
            "type": "scatter",
            "x": dates_str,
            "y": list(predicted_agg["mean"]),
            "mode": "lines",
            "name": "Predicted",
            "line": {"color": colors.primary, "width": 2},
            "hovertemplate": "Predicted: %{y:,.0f}<extra></extra>",
            "visible": True,
        }
    )

    # Trace 2: Actual values
    traces.append(
        {
            "type": "scatter",
            "x": dates_str,
            "y": list(actual_agg),
            "mode": "lines+markers",
            "name": "Actual",
            "line": {"color": colors.text, "width": 1.5, "dash": "dot"},
            "marker": {"color": colors.text, "size": 4},
            "hovertemplate": "Actual: %{y:,.0f}<extra></extra>",
            "visible": True,
        }
    )

    n_agg_traces = 3

    # =========================================================================
    # GEO-LEVEL TRACES (hidden by default) - 3 traces per geo
    # =========================================================================

    has_geo = (
        geo_names is not None
        and len(geo_names) > 1
        and actual_by_geo is not None
        and predicted_by_geo is not None
    )

    n_geo_traces = 3  # Same structure: band, predicted, actual
    n_geos = len(geo_names) if has_geo else 0

    if has_geo:
        for geo in geo_names:
            geo_actual = actual_by_geo.get(geo, [])
            geo_pred = predicted_by_geo.get(geo, {})

            geo_pred_mean = geo_pred.get("mean", [])
            geo_pred_lower = geo_pred.get("lower", [])
            geo_pred_upper = geo_pred.get("upper", [])

            # Skip if no data for this geo
            if len(geo_pred_mean) == 0:
                # Add placeholder traces to maintain indexing
                for _ in range(n_geo_traces):
                    traces.append(
                        {
                            "type": "scatter",
                            "x": [],
                            "y": [],
                            "visible": False,
                        }
                    )
                continue

            # Trace: Uncertainty band
            traces.append(
                {
                    "type": "scatter",
                    "x": dates_str + dates_str[::-1],
                    "y": list(geo_pred_upper) + list(geo_pred_lower)[::-1],
                    "fill": "toself",
                    "fillcolor": f"rgba({_hex_to_rgb(colors.primary)}, 0.2)",
                    "line": {"width": 0},
                    "name": f"{int(chart_config.ci_level * 100)}% CI",
                    "showlegend": True,
                    "hoverinfo": "skip",
                    "visible": False,
                }
            )

            # Trace: Predicted mean
            traces.append(
                {
                    "type": "scatter",
                    "x": dates_str,
                    "y": list(geo_pred_mean),
                    "mode": "lines",
                    "name": "Predicted",
                    "line": {"color": colors.primary, "width": 2},
                    "hovertemplate": f"{geo} Predicted: %{{y:,.0f}}<extra></extra>",
                    "visible": False,
                }
            )

            # Trace: Actual
            traces.append(
                {
                    "type": "scatter",
                    "x": dates_str,
                    "y": list(geo_actual),
                    "mode": "lines+markers",
                    "name": "Actual",
                    "line": {"color": colors.text, "width": 1.5, "dash": "dot"},
                    "marker": {"color": colors.text, "size": 4},
                    "hovertemplate": f"{geo} Actual: %{{y:,.0f}}<extra></extra>",
                    "visible": False,
                }
            )

    # =========================================================================
    # BUILD DROPDOWN MENU
    # =========================================================================

    buttons = []

    # Button 1: "Aggregated (Total)" - show aggregate traces, hide geo traces
    visible_agg = [True] * n_agg_traces + [False] * (n_geo_traces * n_geos)
    buttons.append(
        {
            "label": "Aggregated (Total)",
            "method": "update",
            "args": [
                {"visible": visible_agg},
                {"title": {"text": "Model Fit: Aggregated (Total)"}},
            ],
        }
    )

    # Buttons for each geo
    if has_geo:
        for i, geo in enumerate(geo_names):
            # Hide aggregate traces, show only this geo's traces
            visible_geo = [False] * n_agg_traces
            for j in range(n_geos):
                visible_geo.extend([j == i] * n_geo_traces)

            buttons.append(
                {
                    "label": geo,
                    "method": "update",
                    "args": [
                        {"visible": visible_geo},
                        {"title": {"text": f"Model Fit: {geo}"}},
                    ],
                }
            )

    # =========================================================================
    # LAYOUT
    # =========================================================================

    layout = {
        "title": {"text": "Model Fit: Aggregated (Total)", "font": {"size": 16}},
        "paper_bgcolor": "transparent",
        "plot_bgcolor": "transparent",
        "font": {"family": "Inter, sans-serif", "color": colors.text, "size": 12},
        "margin": {"t": 80, "r": 30, "b": 60, "l": 70},
        "height": chart_config.height,
        "xaxis": {
            "title": "Period",
            "gridcolor": colors.border,
            "showgrid": True,
            "zeroline": False,
        },
        "yaxis": {
            "title": chart_config.y_title or "Revenue",
            "gridcolor": colors.border,
            "showgrid": True,
            "zeroline": False,
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        "hovermode": "x unified",
    }

    # Add dropdown menu if we have geo data
    if has_geo and len(buttons) > 1:
        layout["updatemenus"] = [
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "active": 0,
                "x": 0.0,
                "xanchor": "left",
                "y": 1.18,
                "yanchor": "top",
                "bgcolor": colors.surface,
                "bordercolor": colors.border,
                "borderwidth": 1,
                "font": {"size": 11},
                "pad": {"r": 10, "t": 10},
            }
        ]

    return create_plotly_div(traces, layout, div_id)


def create_model_fit_chart_with_dimension_filter(
    dates: list | np.ndarray | pd.DatetimeIndex,
    actual_agg: np.ndarray,
    predicted_agg: dict[str, np.ndarray],
    actual_by_geo: dict[str, np.ndarray] | None = None,
    predicted_by_geo: dict[str, dict[str, np.ndarray]] | None = None,
    actual_by_product: dict[str, np.ndarray] | None = None,
    predicted_by_product: dict[str, dict[str, np.ndarray]] | None = None,
    geo_names: list[str] | None = None,
    product_names: list[str] | None = None,
    config: ReportConfig = None,
    chart_config: ChartConfig | None = None,
    div_id: str = "modelFitChartFiltered",
) -> str:
    """
    Create model fit visualization with multi-select dimension filters.

    Supports filtering by geography and/or product with checkbox-based UI.
    Default view shows aggregated data; users can select multiple
    specific geos/products to compare.

    Parameters
    ----------
    dates : array-like
        Time index for x-axis
    actual_agg : ndarray
        Aggregated observed values (sum over all dimensions)
    predicted_agg : dict
        Aggregated predictions with keys "mean", "lower", "upper"
    actual_by_geo : dict, optional
        Per-geo observed values: {geo_name: ndarray}
    predicted_by_geo : dict, optional
        Per-geo predictions: {geo_name: {"mean", "lower", "upper"}}
    actual_by_product : dict, optional
        Per-product observed values: {product_name: ndarray}
    predicted_by_product : dict, optional
        Per-product predictions: {product_name: {"mean", "lower", "upper"}}
    geo_names : list, optional
        List of geography names
    product_names : list, optional
        List of product names
    config : ReportConfig
        Report configuration
    chart_config : ChartConfig, optional
        Chart-specific configuration
    div_id : str
        HTML div ID for the chart

    Returns
    -------
    str
        HTML string with embedded Plotly chart and multi-select filters
    """
    chart_config = chart_config or ChartConfig(height=400, ci_level=0.8)
    colors = config.color_scheme if config else ColorScheme()

    # Convert dates to string format for JSON
    dates_str = _dates_to_strings(dates)

    traces = []
    trace_metadata = []  # Track what each trace represents

    # =========================================================================
    # AGGREGATED TRACES (visible by default) - 3 traces
    # =========================================================================

    # Trace 0: Uncertainty band (fill)
    traces.append(
        {
            "type": "scatter",
            "x": dates_str + dates_str[::-1],
            "y": list(predicted_agg["upper"]) + list(predicted_agg["lower"])[::-1],
            "fill": "toself",
            "fillcolor": f"rgba({_hex_to_rgb(colors.primary)}, 0.2)",
            "line": {"width": 0},
            "name": f"{int(chart_config.ci_level * 100)}% CI",
            "showlegend": True,
            "hoverinfo": "skip",
            "visible": True,
        }
    )
    trace_metadata.append({"type": "agg", "dim": None, "value": None})

    # Trace 1: Predicted mean
    traces.append(
        {
            "type": "scatter",
            "x": dates_str,
            "y": list(predicted_agg["mean"]),
            "mode": "lines",
            "name": "Predicted",
            "line": {"color": colors.primary, "width": 2},
            "hovertemplate": "Predicted: %{y:,.0f}<extra></extra>",
            "visible": True,
        }
    )
    trace_metadata.append({"type": "agg", "dim": None, "value": None})

    # Trace 2: Actual values
    traces.append(
        {
            "type": "scatter",
            "x": dates_str,
            "y": list(actual_agg),
            "mode": "lines+markers",
            "name": "Actual",
            "line": {"color": colors.text, "width": 1.5, "dash": "dot"},
            "marker": {"color": colors.text, "size": 4},
            "hovertemplate": "Actual: %{y:,.0f}<extra></extra>",
            "visible": True,
        }
    )
    trace_metadata.append({"type": "agg", "dim": None, "value": None})

    # =========================================================================
    # GEO-LEVEL TRACES (hidden by default) - 3 traces per geo
    # =========================================================================

    has_geo = (
        geo_names is not None
        and len(geo_names) > 1
        and actual_by_geo is not None
        and predicted_by_geo is not None
    )

    geo_colors = _generate_dimension_colors(geo_names, colors) if has_geo else {}

    if has_geo:
        for geo in geo_names:
            geo_actual = actual_by_geo.get(geo, [])
            geo_pred = predicted_by_geo.get(geo, {})

            geo_pred_mean = geo_pred.get("mean", [])
            geo_pred_lower = geo_pred.get("lower", [])
            geo_pred_upper = geo_pred.get("upper", [])
            geo_color = geo_colors.get(geo, colors.primary)

            if len(geo_pred_mean) == 0:
                for _ in range(3):
                    traces.append(
                        {"type": "scatter", "x": [], "y": [], "visible": False}
                    )
                    trace_metadata.append({"type": "geo", "dim": "geo", "value": geo})
                continue

            # CI band
            traces.append(
                {
                    "type": "scatter",
                    "x": dates_str + dates_str[::-1],
                    "y": list(geo_pred_upper) + list(geo_pred_lower)[::-1],
                    "fill": "toself",
                    "fillcolor": f"rgba({_hex_to_rgb(geo_color)}, 0.15)",
                    "line": {"width": 0},
                    "name": f"{geo} CI",
                    "showlegend": False,
                    "hoverinfo": "skip",
                    "visible": False,
                }
            )
            trace_metadata.append({"type": "geo", "dim": "geo", "value": geo})

            # Predicted
            traces.append(
                {
                    "type": "scatter",
                    "x": dates_str,
                    "y": list(geo_pred_mean),
                    "mode": "lines",
                    "name": f"{geo}",
                    "line": {"color": geo_color, "width": 2},
                    "hovertemplate": f"{geo} Pred: %{{y:,.0f}}<extra></extra>",
                    "visible": False,
                }
            )
            trace_metadata.append({"type": "geo", "dim": "geo", "value": geo})

            # Actual
            traces.append(
                {
                    "type": "scatter",
                    "x": dates_str,
                    "y": list(geo_actual),
                    "mode": "markers",
                    "name": f"{geo} Actual",
                    "marker": {"color": geo_color, "size": 5, "symbol": "circle-open"},
                    "hovertemplate": f"{geo} Actual: %{{y:,.0f}}<extra></extra>",
                    "visible": False,
                    "showlegend": False,
                }
            )
            trace_metadata.append({"type": "geo", "dim": "geo", "value": geo})

    # =========================================================================
    # PRODUCT-LEVEL TRACES (hidden by default) - 3 traces per product
    # =========================================================================

    has_product = (
        product_names is not None
        and len(product_names) > 1
        and actual_by_product is not None
        and predicted_by_product is not None
    )

    product_colors = (
        _generate_dimension_colors(product_names, colors, offset=len(geo_names or []))
        if has_product
        else {}
    )

    if has_product:
        for product in product_names:
            prod_actual = actual_by_product.get(product, [])
            prod_pred = predicted_by_product.get(product, {})

            prod_pred_mean = prod_pred.get("mean", [])
            prod_pred_lower = prod_pred.get("lower", [])
            prod_pred_upper = prod_pred.get("upper", [])
            prod_color = product_colors.get(product, colors.secondary)

            if len(prod_pred_mean) == 0:
                for _ in range(3):
                    traces.append(
                        {"type": "scatter", "x": [], "y": [], "visible": False}
                    )
                    trace_metadata.append(
                        {"type": "product", "dim": "product", "value": product}
                    )
                continue

            # CI band
            traces.append(
                {
                    "type": "scatter",
                    "x": dates_str + dates_str[::-1],
                    "y": list(prod_pred_upper) + list(prod_pred_lower)[::-1],
                    "fill": "toself",
                    "fillcolor": f"rgba({_hex_to_rgb(prod_color)}, 0.15)",
                    "line": {"width": 0},
                    "name": f"{product} CI",
                    "showlegend": False,
                    "hoverinfo": "skip",
                    "visible": False,
                }
            )
            trace_metadata.append(
                {"type": "product", "dim": "product", "value": product}
            )

            # Predicted
            traces.append(
                {
                    "type": "scatter",
                    "x": dates_str,
                    "y": list(prod_pred_mean),
                    "mode": "lines",
                    "name": f"{product}",
                    "line": {"color": prod_color, "width": 2, "dash": "dash"},
                    "hovertemplate": f"{product} Pred: %{{y:,.0f}}<extra></extra>",
                    "visible": False,
                }
            )
            trace_metadata.append(
                {"type": "product", "dim": "product", "value": product}
            )

            # Actual
            traces.append(
                {
                    "type": "scatter",
                    "x": dates_str,
                    "y": list(prod_actual),
                    "mode": "markers",
                    "name": f"{product} Actual",
                    "marker": {
                        "color": prod_color,
                        "size": 5,
                        "symbol": "diamond-open",
                    },
                    "hovertemplate": f"{product} Actual: %{{y:,.0f}}<extra></extra>",
                    "visible": False,
                    "showlegend": False,
                }
            )
            trace_metadata.append(
                {"type": "product", "dim": "product", "value": product}
            )

    # =========================================================================
    # LAYOUT
    # =========================================================================

    layout = {
        "title": {"text": "Model Fit", "font": {"size": 16}},
        "paper_bgcolor": "transparent",
        "plot_bgcolor": "transparent",
        "font": {"family": "Inter, sans-serif", "color": colors.text, "size": 12},
        "margin": {"t": 60, "r": 30, "b": 60, "l": 70},
        "height": chart_config.height,
        "xaxis": {
            "title": "Period",
            "gridcolor": colors.border,
            "showgrid": True,
            "zeroline": False,
        },
        "yaxis": {
            "title": chart_config.y_title or "Revenue",
            "gridcolor": colors.border,
            "showgrid": True,
            "zeroline": False,
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        "hovermode": "x unified",
    }

    # =========================================================================
    # BUILD HTML WITH MULTI-SELECT FILTERS
    # =========================================================================

    filter_html = _build_dimension_filter_html(
        div_id=div_id,
        geo_names=geo_names if has_geo else None,
        product_names=product_names if has_product else None,
        geo_colors=geo_colors,
        product_colors=product_colors,
    )

    js_code = _build_dimension_filter_js(
        div_id=div_id,
        trace_metadata=trace_metadata,
        has_geo=has_geo,
        has_product=has_product,
    )

    chart_html = f"""
    <div id="{div_id}" class="chart-container"></div>
    <script>
        Plotly.newPlot(
            "{div_id}",
            {_to_json(traces)},
            {_to_json(layout)},
            {{"displayModeBar": false, "responsive": true}}
        );
    </script>
    """

    return f"""
    <div class="dimension-filter-container">
        {filter_html}
        {chart_html}
        {js_code}
    </div>
    """


def create_fit_statistics_with_geo_selector(
    fit_stats_agg: dict[str, float],
    fit_stats_by_geo: dict[str, dict[str, float]] | None = None,
    geo_names: list[str] | None = None,
    config: ReportConfig = None,
    div_id: str = "fitStatsTable",
) -> str:
    """
    Create fit statistics display with geo selector.

    This uses JavaScript to show/hide table rows based on selection.

    Parameters
    ----------
    fit_stats_agg : dict
        Aggregated fit statistics: {"r2", "rmse", "mape"}
    fit_stats_by_geo : dict, optional
        Per-geo stats: {geo_name: {"r2", "rmse", "mape"}}
    geo_names : list, optional
        Geography names
    config : ReportConfig
        Report configuration
    div_id : str
        HTML div ID

    Returns
    -------
    str
        HTML string with statistics table and selector
    """
    colors = config.color_scheme if config else ColorScheme()

    has_geo = (
        geo_names is not None and len(geo_names) > 1 and fit_stats_by_geo is not None
    )

    # Build options for dropdown
    options_html = '<option value="agg" selected>Aggregated (Total)</option>'
    if has_geo:
        for geo in geo_names:
            options_html += f'<option value="{geo}">{geo}</option>'

    # Build table rows for aggregated stats
    def format_stat(key: str, value: float) -> str:
        if key == "r2":
            return f"{value:.4f}"
        elif key == "rmse":
            return f"{value:,.2f}"
        elif key == "mape":
            return f"{value:.2f}%"
        else:
            return f"{value:.4f}"

    stat_labels = {
        "r2": "R²",
        "rmse": "RMSE",
        "mae": "MAE",
        "mape": "MAPE",
    }

    # Aggregated stats row
    agg_rows = ""
    for key, label in stat_labels.items():
        if key in fit_stats_agg:
            val = format_stat(key, fit_stats_agg[key])
            agg_rows += (
                f'<tr data-geo="agg"><td>{label}</td>'
                f'<td class="mono">{val}</td></tr>'
            )

    # Geo-level stats rows (hidden by default)
    geo_rows = ""
    if has_geo:
        for geo in geo_names:
            geo_stats = fit_stats_by_geo.get(geo, {})
            for key, label in stat_labels.items():
                if key in geo_stats:
                    val = format_stat(key, geo_stats[key])
                    geo_rows += (
                        f'<tr data-geo="{geo}" style="display: none;">'
                        f"<td>{label}</td>"
                        f'<td class="mono">{val}</td></tr>'
                    )

    # JavaScript for toggling visibility
    js_code = f"""
    <script>
        document.getElementById('{div_id}_select').addEventListener('change', function() {{
            var selected = this.value;
            var table = document.getElementById('{div_id}_tbody');
            var rows = table.querySelectorAll('tr');
            rows.forEach(function(row) {{
                if (row.getAttribute('data-geo') === selected) {{
                    row.style.display = '';
                }} else {{
                    row.style.display = 'none';
                }}
            }});
        }});
    </script>
    """

    # Build HTML
    dropdown_html = ""
    if has_geo:
        dropdown_html = f"""
        <div style="margin-bottom: 1rem;">
            <label style="font-size: 0.85rem; color: {colors.text_muted};">View: </label>
            <select id="{div_id}_select" style="padding: 0.25rem 0.5rem; border: 1px solid {colors.border}; border-radius: 4px; font-size: 0.85rem;">
                {options_html}
            </select>
        </div>
        """

    html = f"""
    <div id="{div_id}">
        {dropdown_html}
        <table class="data-table" style="max-width: 400px;">
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody id="{div_id}_tbody">
                {agg_rows}
                {geo_rows}
            </tbody>
        </table>
    </div>
    {js_code if has_geo else ''}
    """

    return html


__all__ = [
    "create_model_fit_chart",
    "create_model_fit_chart_with_geo_selector",
    "create_model_fit_chart_with_dimension_filter",
    "create_fit_statistics_with_geo_selector",
]
