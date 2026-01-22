"""
Model Fitting Page.

Start and monitor model fitting jobs.
"""

import streamlit as st
import time
from datetime import datetime

from api_client import (
    get_api_client,
    fetch_datasets,
    fetch_configs,
    fetch_models,
    clear_model_cache,
    APIError,
    JobStatus,
    ModelInfo,
)
from components import (
    apply_custom_css,
    page_header,
    format_datetime,
    format_duration,
    display_api_error,
    init_session_state,
    status_icon,
    job_progress_display,
)


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Model Fitting | MMM Framework",
    page_icon="🔬",
    layout="wide",
)

apply_custom_css()

init_session_state(
    selected_data_id=None,
    selected_config_id=None,
    selected_model_id=None,
    monitoring_model_id=None,
)


# =============================================================================
# Job Submission Form
# =============================================================================


@st.fragment
def render_job_form():
    """Render the job submission form."""
    st.markdown("### Start New Model Fitting Job")

    try:
        client = get_api_client()
        datasets = fetch_datasets(client)
        configs = fetch_configs(client)
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return

    if not datasets:
        st.warning("Please upload a dataset first.")
        if st.button("Go to Data Management"):
            st.switch_page("pages/1_📁_Data_Management.py")
        return

    if not configs:
        st.warning("Please create a configuration first.")
        if st.button("Go to Configuration"):
            st.switch_page("pages/2_⚙️_Configuration.py")
        return

    # Build options
    dataset_options = {d.filename: d.data_id for d in datasets}
    config_options = {c.name: c.config_id for c in configs}

    with st.form("fit_form"):
        col1, col2 = st.columns(2)

        with col1:
            # Data selection
            st.markdown("#### Data")
            selected_dataset_name = st.selectbox(
                "Select Dataset",
                options=list(dataset_options.keys()),
                index=None,
                placeholder="Choose a dataset...",
            )

            # Config selection
            st.markdown("#### Configuration")
            selected_config_name = st.selectbox(
                "Select Configuration",
                options=list(config_options.keys()),
                index=None,
                placeholder="Choose a configuration...",
            )

        with col2:
            # Job info
            st.markdown("#### Job Settings")

            job_name = st.text_input("Job Name (optional)", placeholder="My Model")
            job_description = st.text_area("Description (optional)", height=68)

            # Override MCMC settings
            st.markdown("#### MCMC Overrides (optional)")

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                override_chains = st.number_input(
                    "Chains",
                    min_value=1,
                    max_value=8,
                    value=None,
                    placeholder="Default",
                )
            with col_b:
                override_draws = st.number_input(
                    "Draws",
                    min_value=100,
                    max_value=10000,
                    value=None,
                    placeholder="Default",
                )
            with col_c:
                override_tune = st.number_input(
                    "Tune",
                    min_value=100,
                    max_value=5000,
                    value=None,
                    placeholder="Default",
                )

            random_seed = st.number_input(
                "Random Seed (optional)", min_value=0, value=None, placeholder="Random"
            )

        # Submit button
        submitted = st.form_submit_button(
            "🚀 Start Fitting", type="primary", use_container_width=True
        )

        if submitted:
            if not selected_dataset_name:
                st.error("Please select a dataset.")
                return
            if not selected_config_name:
                st.error("Please select a configuration.")
                return

            data_id = dataset_options[selected_dataset_name]
            config_id = config_options[selected_config_name]

            with st.spinner("Submitting job..."):
                try:
                    result = client.submit_fit_job(
                        data_id=data_id,
                        config_id=config_id,
                        name=job_name or None,
                        description=job_description or None,
                        n_chains=override_chains if override_chains else None,
                        n_draws=override_draws if override_draws else None,
                        n_tune=override_tune if override_tune else None,
                        random_seed=random_seed if random_seed else None,
                    )

                    st.success(f"✅ Job submitted! Model ID: {result.model_id}")
                    st.session_state.selected_model_id = result.model_id
                    st.session_state.monitoring_model_id = result.model_id

                    clear_model_cache()
                    st.rerun()

                except APIError as e:
                    display_api_error(e)
                except Exception as e:
                    st.error(f"Failed to submit job: {e}")


# =============================================================================
# Job Monitor
# =============================================================================


@st.fragment
def render_job_monitor():
    """Render the job monitoring section."""
    st.markdown("### Job Monitor")

    model_id = st.session_state.monitoring_model_id

    if not model_id:
        st.info("Submit a job or select one from the list below to monitor.")
        return

    try:
        client = get_api_client()
        model = client.get_model(model_id)

        # Header
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{model.name or model.model_id}**")
            st.caption(f"ID: {model.model_id}")
        with col2:
            if st.button("🔄 Refresh", key="refresh_monitor"):
                st.rerun()

        # Status display
        job_progress_display(
            progress=model.progress,
            message=model.progress_message,
        )

        # Timing info
        col1, col2, col3 = st.columns(3)

        with col1:
            st.caption(f"Created: {format_datetime(model.created_at)}")
        with col2:
            if model.started_at:
                st.caption(f"Started: {format_datetime(model.started_at)}")
        with col3:
            if model.completed_at:
                st.caption(f"Completed: {format_datetime(model.completed_at)}")
                duration = (
                    (model.completed_at - model.started_at).total_seconds()
                    if model.started_at
                    else None
                )
                if duration:
                    st.caption(f"Duration: {format_duration(duration)}")

        # Error message
        if model.error_message:
            st.error(f"❌ Error: {model.error_message}")

        # Diagnostics preview (if completed)
        if model.progress == "completed" and model.diagnostics:
            st.markdown("#### Quick Diagnostics")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Divergences", model.diagnostics.get("divergences", "N/A"))
            with col2:
                rhat = model.diagnostics.get("rhat_max")
                st.metric("R-hat Max", f"{rhat:.4f}" if rhat else "N/A")
            with col3:
                ess = model.diagnostics.get("ess_bulk_min")
                st.metric("ESS Min", f"{ess:.0f}" if ess else "N/A")

        # Actions
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if model.progress == "completed":
                if st.button(
                    "📈 View Results", type="primary", use_container_width=True
                ):
                    st.session_state.selected_model_id = model.model_id
                    st.switch_page("pages/4_Results.py")

        with col2:
            if model.progress in {"pending", "queued", "running"}:
                if st.button("⏹️ Cancel", use_container_width=True):
                    st.session_state[f"confirm_cancel_{model.model_id}"] = True
                    st.rerun()

        # Cancel confirmation
        if st.session_state.get(f"confirm_cancel_{model.model_id}", False):
            st.warning("⚠️ Cancel this job?")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Yes, Cancel", key="confirm_cancel_yes", type="primary"):
                    try:
                        client.cancel_job(model.model_id)
                        st.success("Job cancelled!")
                        clear_model_cache()
                        del st.session_state[f"confirm_cancel_{model.model_id}"]
                        st.rerun()
                    except APIError as e:
                        display_api_error(e)
            with col_no:
                if st.button("No, Keep Running", key="confirm_cancel_no"):
                    del st.session_state[f"confirm_cancel_{model.model_id}"]
                    st.rerun()

        with col3:
            if st.button("🗑️ Delete", use_container_width=True):
                st.session_state[f"confirm_delete_model_{model.model_id}"] = True
                st.rerun()

        # Delete confirmation
        if st.session_state.get(f"confirm_delete_model_{model.model_id}", False):
            st.warning("⚠️ Delete this model and all results?")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Yes", key="confirm_del_yes", type="primary"):
                    try:
                        client.delete_model(model.model_id)
                        st.success("Deleted!")
                        st.session_state.monitoring_model_id = None
                        st.session_state.selected_model_id = None
                        clear_model_cache()
                        del st.session_state[f"confirm_delete_model_{model.model_id}"]
                        st.rerun()
                    except APIError as e:
                        display_api_error(e)
            with col_no:
                if st.button("Cancel", key="confirm_del_no"):
                    del st.session_state[f"confirm_delete_model_{model.model_id}"]
                    st.rerun()

        # Auto-refresh for active jobs
        if getattr(model, "is_active", False):
            st.caption("⏳ Auto-refreshing every 5 seconds...")
            time.sleep(5)
            st.rerun()

    except APIError as e:
        display_api_error(e)
    except Exception as e:
        st.error(f"Error loading model: {e}")


# =============================================================================
# Job List
# =============================================================================


@st.fragment
def render_job_list():
    """Render the list of all jobs."""
    st.markdown("### All Jobs")

    # Filters
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        status_filter = st.selectbox(
            "Status Filter",
            options=["All", "Running", "Completed", "Failed", "Queued", "Pending"],
            index=0,
        )

    with col2:
        sort_order = st.selectbox(
            "Sort By",
            options=["Newest First", "Oldest First"],
            index=0,
        )

    with col3:
        if st.button("🔄 Refresh", use_container_width=True):
            clear_model_cache()
            st.rerun()

    try:
        client = get_api_client()

        # Get models with filter
        filter_status = None if status_filter == "All" else status_filter.lower()
        models = fetch_models(client, status_filter=filter_status)

        if not models:
            st.info("No jobs found.")
            return

        # Sort
        if sort_order == "Oldest First":
            models = list(reversed(models))

        # Display jobs
        for model in models:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])

                with col1:
                    is_monitoring = (
                        st.session_state.monitoring_model_id == model.model_id
                    )
                    icon = "👁️" if is_monitoring else status_icon(model.status)
                    st.markdown(f"**{icon} {model.name or model.model_id[:8]}**")

                with col2:
                    st.caption(format_datetime(model.created_at))

                with col3:
                    st.markdown(
                        f"{status_icon(model.status)} {model.status}"
                    )

                with col4:
                    if model.progress == "running":
                        st.progress(model.progress / 100)
                    elif model.progress == "completed":
                        st.success("Complete", icon="✅")
                    elif model.progress == "failed":
                        st.error("Failed", icon="❌")
                    else:
                        st.caption(model.status)

                with col5:
                    if st.button("👁️", key=f"monitor_{model.model_id}", help="Monitor"):
                        st.session_state.monitoring_model_id = model.model_id
                        st.rerun()

                st.markdown("---")

    except APIError as e:
        display_api_error(e)
    except Exception as e:
        st.error(f"Error loading jobs: {e}")


# =============================================================================
# Active Jobs Summary
# =============================================================================


@st.fragment
def render_active_jobs_summary():
    """Render summary of active jobs."""
    try:
        client = get_api_client()

        # Get all models
        models = client.list_models(limit=100)

        # Count by status
        status_counts = {}
        for model in models:
            status = model.status
            status_counts[status] = status_counts.get(status, 0) + 1

        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total", len(models))
        with col2:
            st.metric("Running", status_counts.get("running", 0))
        with col3:
            st.metric("Queued", status_counts.get("queued", 0))
        with col4:
            st.metric("Completed", status_counts.get("completed", 0))
        with col5:
            st.metric("Failed", status_counts.get("failed", 0))

    except Exception as e:
        st.warning(f"Could not load summary: {e}")


# =============================================================================
# Main
# =============================================================================


def main():
    """Main page function."""
    page_header("🔬 Model Fitting", "Submit and monitor Bayesian MMM fitting jobs.")

    # Summary
    render_active_jobs_summary()

    st.markdown("---")

    # Two-column layout
    col_left, col_right = st.columns([1, 1])

    with col_left:
        with st.expander(
            "➕ Submit New Job", expanded=not st.session_state.monitoring_model_id
        ):
            render_job_form()

    with col_right:
        render_job_monitor()

    st.markdown("---")
    render_job_list()


if __name__ == "__main__":
    main()
