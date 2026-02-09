"""
Scenarios Page.

Run what-if scenarios and budget optimization.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Any

from api_client import (
    get_api_client,
    fetch_models,
    fetch_model_results,
    APIError,
    JobStatus,
)
from components import (
    apply_custom_css,
    page_header,
    format_datetime,
    format_number,
    format_percent,
    display_api_error,
    init_session_state,
    status_icon,
    CHART_COLORS,
    plot_scenario_comparison,
    plot_budget_optimization,
)

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Scenarios | MMM Framework",
    page_icon="🔮",
    layout="wide",
)

apply_custom_css()

init_session_state(
    selected_model_id=None,
    scenario_results=None,
    scenario_history=[],
)


# =============================================================================
# Model Selector
# =============================================================================


@st.fragment
def render_model_selector():
    """Render the model selection dropdown."""
    try:
        client = get_api_client()
        models = fetch_models(client, status_filter="completed")

        if not models:
            st.warning("No completed models found. Fit a model first.")
            if st.button("Go to Model Fitting"):
                st.switch_page("pages/3_🔬_Model_Fitting.py")
            return False

        # Build options
        model_options = {
            f"{m.name or m.model_id[:8]} ({format_datetime(m.created_at)})": m.model_id
            for m in models
        }

        # Find current selection
        current_idx = None
        if st.session_state.selected_model_id:
            for i, (name, mid) in enumerate(model_options.items()):
                if mid == st.session_state.selected_model_id:
                    current_idx = i
                    break

        col1, col2 = st.columns([4, 1])

        with col1:
            selected_name = st.selectbox(
                "Select Model",
                options=list(model_options.keys()),
                index=current_idx,
                key="scenario_model_selector",
            )

        with col2:
            if st.button("🔄", key="refresh_scenario_selector", help="Refresh"):
                st.cache_data.clear()
                st.rerun()

        if selected_name:
            new_model_id = model_options[selected_name]
            if new_model_id != st.session_state.selected_model_id:
                st.session_state.selected_model_id = new_model_id
                st.session_state.scenario_results = None
                st.rerun()

        return True

    except APIError as e:
        display_api_error(e)
        return False
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return False


# =============================================================================
# Scenario Form
# =============================================================================


@st.fragment
def render_scenario_form():
    """Render the scenario configuration form."""
    st.markdown("### Define Scenario")

    model_id = st.session_state.selected_model_id
    if not model_id:
        st.info("Select a model first.")
        return

    # Get channel info from model
    try:
        client = get_api_client()
        results = fetch_model_results(client, model_id)

        # Extract channel names
        channels = results.get("channel_names", [])
        if not channels:
            # Try to extract from other fields
            contrib = results.get("contributions", {})
            if contrib:
                channels = list(contrib.get("total_contributions", {}).keys())

        if not channels:
            st.warning("Could not determine channel names from model.")
            return

    except Exception as e:
        st.error(f"Could not load model info: {e}")
        return

    # Scenario type
    scenario_type = st.radio(
        "Scenario Type",
        options=["Percentage Change", "Absolute Budget", "Budget Reallocation"],
        horizontal=True,
    )

    st.markdown("---")

    spend_changes = {}

    if scenario_type == "Percentage Change":
        st.markdown("#### Adjust spend by percentage")

        # Quick presets
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("-20% All", use_container_width=True):
                for ch in channels:
                    st.session_state[f"pct_{ch}"] = -20.0
        with col2:
            if st.button("-10% All", use_container_width=True):
                for ch in channels:
                    st.session_state[f"pct_{ch}"] = -10.0
        with col3:
            if st.button("+10% All", use_container_width=True):
                for ch in channels:
                    st.session_state[f"pct_{ch}"] = 10.0
        with col4:
            if st.button("+20% All", use_container_width=True):
                for ch in channels:
                    st.session_state[f"pct_{ch}"] = 20.0

        st.markdown("---")

        # Individual channel controls
        n_cols = min(4, len(channels))
        cols = st.columns(n_cols)

        for i, channel in enumerate(channels):
            with cols[i % n_cols]:
                pct = st.slider(
                    channel,
                    min_value=-100.0,
                    max_value=100.0,
                    value=st.session_state.get(f"pct_{channel}", 0.0),
                    step=5.0,
                    key=f"slider_pct_{channel}",
                    format="%.0f%%",
                )
                spend_changes[channel] = pct / 100.0  # Convert to decimal

    elif scenario_type == "Absolute Budget":
        st.markdown("#### Set absolute budget values")

        # Get current spend from results if available
        current_spend = results.get("current_spend", {ch: 100000 for ch in channels})

        n_cols = min(3, len(channels))
        cols = st.columns(n_cols)

        for i, channel in enumerate(channels):
            with cols[i % n_cols]:
                current = current_spend.get(channel, 100000)
                new_budget = st.number_input(
                    channel,
                    min_value=0.0,
                    value=float(current),
                    step=1000.0,
                    key=f"abs_{channel}",
                )
                # Calculate percentage change
                if current > 0:
                    spend_changes[channel] = (new_budget - current) / current
                else:
                    spend_changes[channel] = 0.0

    else:  # Budget Reallocation
        st.markdown("#### Reallocate budget between channels")

        current_spend = results.get("current_spend", {ch: 100000 for ch in channels})
        total_budget = sum(current_spend.values())

        st.metric("Total Budget", f"${total_budget:,.0f}")

        st.markdown("Adjust allocation percentages (must sum to 100%):")

        # Calculate current percentages
        current_pcts = {
            ch: (v / total_budget * 100) if total_budget > 0 else (100 / len(channels))
            for ch, v in current_spend.items()
        }

        new_pcts = {}
        n_cols = min(4, len(channels))
        cols = st.columns(n_cols)

        for i, channel in enumerate(channels):
            with cols[i % n_cols]:
                new_pcts[channel] = st.number_input(
                    channel,
                    min_value=0.0,
                    max_value=100.0,
                    value=current_pcts.get(channel, 100 / len(channels)),
                    step=1.0,
                    key=f"alloc_{channel}",
                    format="%.1f%%",
                )

        # Check if percentages sum to 100
        total_pct = sum(new_pcts.values())
        if abs(total_pct - 100) > 0.1:
            st.warning(
                f"⚠️ Allocations sum to {total_pct:.1f}%. Please adjust to 100%."
            )

        # Calculate spend changes
        for channel in channels:
            new_spend = total_budget * new_pcts[channel] / 100
            current = current_spend.get(channel, 0)
            if current > 0:
                spend_changes[channel] = (new_spend - current) / current
            else:
                spend_changes[channel] = 0.0

    # Time period selection
    st.markdown("---")
    st.markdown("#### Time Period (Optional)")

    use_time_period = st.checkbox("Limit to specific time period")
    time_period = None

    if use_time_period:
        col1, col2 = st.columns(2)
        with col1:
            start_idx = st.number_input("Start Index", min_value=0, value=0)
        with col2:
            end_idx = st.number_input(
                "End Index", min_value=1, value=results.get("n_obs", 100)
            )
        time_period = (int(start_idx), int(end_idx))

    # Run scenario button
    st.markdown("---")

    if st.button("🚀 Run Scenario", type="primary", use_container_width=True):
        with st.spinner("Running scenario analysis..."):
            try:
                scenario_results = client.run_scenario(
                    model_id=model_id,
                    spend_changes=spend_changes,
                    time_period=time_period,
                )

                st.session_state.scenario_results = scenario_results

                # Add to history
                history_entry = {
                    "spend_changes": spend_changes,
                    "results": scenario_results,
                    "type": scenario_type,
                }
                st.session_state.scenario_history.append(history_entry)

                st.success("✅ Scenario completed!")
                st.rerun()

            except APIError as e:
                display_api_error(e)
            except Exception as e:
                st.error(f"Error running scenario: {e}")


# =============================================================================
# Scenario Results
# =============================================================================


@st.fragment
def render_scenario_results():
    """Render scenario results."""
    st.markdown("### Scenario Results")

    results = st.session_state.scenario_results

    if not results:
        st.info("Run a scenario to see results here.")
        return

    # Check for error
    if results.get("error"):
        st.error(f"Scenario failed: {results['error']}")
        return

    # Key metrics
    baseline = results.get("baseline_outcome", 0)
    scenario = results.get("scenario_outcome", 0)
    change = results.get("outcome_change", 0)
    change_pct = results.get("outcome_change_pct", 0)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Baseline", format_number(baseline))
    with col2:
        st.metric("Scenario", format_number(scenario))
    with col3:
        delta_color = "normal" if change >= 0 else "inverse"
        st.metric(
            "Change",
            format_number(change),
            delta=f"{change_pct:+.1f}%",
            delta_color=delta_color,
        )
    with col4:
        roi = (
            change
            / sum(abs(v * baseline) for v in results.get("spend_changes", {}).values())
            if results.get("spend_changes")
            else 0
        )
        st.metric("Scenario ROI", f"{roi:.2f}x" if abs(roi) < 100 else "N/A")

    # Visualization
    st.markdown("---")

    spend_changes = results.get("spend_changes", {})
    channel_effects = results.get("channel_effects", {})

    plot_scenario_comparison(
        baseline=baseline,
        scenario=scenario,
        channel_effects=spend_changes,
        title="Baseline vs Scenario Outcome",
    )

    # Spend changes breakdown
    st.markdown("### Spend Changes Applied")

    changes_df = pd.DataFrame(
        [
            {
                "Channel": k,
                "Change (%)": f"{v*100:+.1f}%",
            }
            for k, v in spend_changes.items()
        ]
    )

    st.dataframe(changes_df, use_container_width=True, hide_index=True)


# =============================================================================
# Scenario History
# =============================================================================


@st.fragment
def render_scenario_history():
    """Render scenario history."""
    st.markdown("### Scenario History")

    history = st.session_state.scenario_history

    if not history:
        st.info("No scenarios run yet.")
        return

    # Clear history button
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🗑️ Clear", key="clear_history"):
            st.session_state.scenario_history = []
            st.session_state.scenario_results = None
            st.rerun()

    # Display history
    for i, entry in enumerate(reversed(history)):
        results = entry["results"]

        with st.expander(
            f"Scenario {len(history) - i}: {entry['type']}", expanded=i == 0
        ):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Baseline", format_number(results.get("baseline_outcome", 0)))
            with col2:
                st.metric("Scenario", format_number(results.get("scenario_outcome", 0)))
            with col3:
                change_pct = results.get("outcome_change_pct", 0)
                st.metric("Change", f"{change_pct:+.1f}%")

            # Spend changes
            st.caption("Spend Changes:")
            for ch, change in entry["spend_changes"].items():
                st.caption(f"  • {ch}: {change*100:+.1f}%")

            # Load this scenario
            if st.button(f"📋 Load", key=f"load_scenario_{i}"):
                st.session_state.scenario_results = results
                st.rerun()


# =============================================================================
# Optimization Section
# =============================================================================


@st.fragment
def render_optimization():
    """Render budget optimization section."""
    st.markdown("### Budget Optimization")

    model_id = st.session_state.selected_model_id
    if not model_id:
        st.info("Select a model first.")
        return

    st.info("🚧 Budget optimization is a premium feature. Contact us for access.")

    st.markdown("""
    Budget optimization would include:
    - **Objective Function**: Maximize KPI, maximize ROAS, or target specific outcome
    - **Constraints**: Total budget, min/max per channel, change limits
    - **Algorithm**: Sequential Least Squares Programming (SLSQP) or Bayesian optimization
    """)

    # Placeholder UI
    with st.expander("⚙️ Optimization Settings (Preview)", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.selectbox(
                "Objective",
                options=["Maximize KPI", "Maximize ROAS", "Target KPI Value"],
                disabled=True,
            )

            st.number_input(
                "Total Budget",
                value=1000000.0,
                disabled=True,
            )

        with col2:
            st.slider(
                "Max Change per Channel",
                min_value=10,
                max_value=100,
                value=50,
                disabled=True,
            )

            st.number_input(
                "Target KPI (if applicable)",
                value=0.0,
                disabled=True,
            )

        st.button("🎯 Optimize", disabled=True, use_container_width=True)


# =============================================================================
# Main
# =============================================================================


def main():
    """Main page function."""
    page_header(
        "🔮 Scenarios", "Run what-if scenarios and optimize your marketing budget."
    )

    # Model selector
    has_model = render_model_selector()

    if not has_model:
        return

    if not st.session_state.selected_model_id:
        st.info("Select a model to run scenarios.")
        return

    st.markdown("---")

    # Two-column layout
    col_left, col_right = st.columns([1, 1])

    with col_left:
        render_scenario_form()

    with col_right:
        render_scenario_results()

    st.markdown("---")

    # Additional sections
    tab1, tab2 = st.tabs(["📜 History", "🎯 Optimization"])

    with tab1:
        render_scenario_history()

    with tab2:
        render_optimization()


if __name__ == "__main__":
    main()
