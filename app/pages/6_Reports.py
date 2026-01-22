"""
Report Generation Page.

Generate and download HTML reports for fitted models.
"""

import streamlit as st
import streamlit.components.v1 as components

import time
from datetime import datetime

from api_client import (
    get_api_client,
    fetch_models,
    clear_model_cache,
    APIError,
)
from components import (
    apply_custom_css,
    page_header,
    format_datetime,
    format_bytes,
    display_api_error,
    init_session_state,
    status_icon,
)


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Reports | MMM Framework",
    page_icon="📄",
    layout="wide",
)

apply_custom_css()

init_session_state(
    selected_model_id=None,
    report_id=None,
    report_status=None,
    generating=False,
)


# =============================================================================
# Model Selector
# =============================================================================


@st.fragment
def render_model_selector():
    """Render the model selection dropdown."""
    try:
        api_client = get_api_client()
        models = fetch_models(api_client, status_filter="completed")

        if not models:
            st.warning("No completed models found. Fit a model first.")
            if st.button("Go to Model Fitting"):
                st.switch_page("pages/3_Model_Fitting.py")
            return False

        # Build options
        model_options = {}
        for m in models:
            label = f"{m.name or m.model_id[:8]}"
            if m.completed_at:
                label += f" ({format_datetime(m.completed_at)})"
            model_options[label] = m.model_id

        # Find current selection
        current_idx = 0
        if st.session_state.selected_model_id:
            for i, (name, mid) in enumerate(model_options.items()):
                if mid == st.session_state.selected_model_id:
                    current_idx = i
                    break

        col1, col2 = st.columns([4, 1])

        with col1:
            selected_name = st.selectbox(
                "Select Completed Model",
                options=list(model_options.keys()),
                index=current_idx,
                key="report_model_selector",
            )

        with col2:
            if st.button("🔄", key="refresh_model_selector", help="Refresh"):
                clear_model_cache()
                st.rerun()

        if selected_name:
            new_model_id = model_options[selected_name]
            if new_model_id != st.session_state.selected_model_id:
                st.session_state.selected_model_id = new_model_id
                st.session_state.report_id = None
                st.session_state.report_status = None
                st.rerun()

        return True

    except APIError as e:
        display_api_error(e)
        return False
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return False


# =============================================================================
# Report Configuration Form
# =============================================================================


@st.fragment
def render_report_form():
    """Render the report configuration form."""
    st.markdown("### Report Configuration")

    model_id = st.session_state.selected_model_id
    if not model_id:
        st.info("Select a model first.")
        return

    # Basic info
    with st.expander("📝 Basic Information", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            report_title = st.text_input(
                "Report Title",
                value="Marketing Mix Model Report",
                help="Main title for the report",
            )
            client_name = st.text_input(
                "Client Name",
                value="",
                help="Client or company name (optional)",
            )

        with col2:
            report_subtitle = st.text_input(
                "Subtitle",
                value="",
                help="Additional subtitle (optional)",
            )
            analysis_period = st.text_input(
                "Analysis Period",
                value="",
                placeholder="e.g., Jan 2023 - Dec 2025",
                help="Time period covered by the analysis",
            )

    # Sections
    with st.expander("📑 Report Sections", expanded=True):
        st.markdown("Select which sections to include in the report:")

        col1, col2 = st.columns(2)

        with col1:
            include_executive_summary = st.checkbox(
                "Executive Summary",
                value=True,
                help="Key metrics and high-level findings",
            )
            include_model_fit = st.checkbox(
                "Model Fit",
                value=True,
                help="Actual vs predicted comparison",
            )
            include_channel_roi = st.checkbox(
                "Channel ROI",
                value=True,
                help="Return on investment by channel with uncertainty",
            )
            include_decomposition = st.checkbox(
                "Decomposition",
                value=True,
                help="Contribution breakdown over time",
            )

        with col2:
            include_saturation = st.checkbox(
                "Saturation Curves",
                value=True,
                help="Response curves showing diminishing returns",
            )
            include_diagnostics = st.checkbox(
                "Model Diagnostics",
                value=True,
                help="MCMC convergence and model quality metrics",
            )
            include_methodology = st.checkbox(
                "Methodology",
                value=True,
                help="Technical methodology explanation",
            )

    # Formatting options
    with st.expander("🎨 Formatting Options", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            credible_interval = st.slider(
                "Credible Interval",
                min_value=0.5,
                max_value=0.99,
                value=0.80,
                step=0.01,
                format="%.2f%%",
                help="Width of uncertainty bands (e.g., 80% = 80% CI)",
            )

        with col2:
            currency_symbol = st.selectbox(
                "Currency Symbol",
                options=["$", "€", "£", "¥", "₹"],
                index=0,
                help="Currency symbol for monetary values",
            )

        with col3:
            currency_scale_options = [
                ("Units", 1.0),
                ("Thousands (K)", 1000.0),
                ("Millions (M)", 1000000.0),
            ]
            currency_scale_selection = st.selectbox(
                "Currency Scale",
                options=currency_scale_options,
                format_func=lambda x: x[0],
                index=0,
                help="Scale for displaying monetary values",
            )
            currency_scale = currency_scale_selection[1]

    # Generate button
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        generate_clicked = st.button(
            "📄 Generate Report",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.generating,
        )

    if generate_clicked:
        try:
            api_client = get_api_client()

            with st.spinner("Submitting report generation request..."):
                result = api_client.generate_report(
                    model_id=model_id,
                    title=report_title or None,
                    client=client_name or None,
                    subtitle=report_subtitle or None,
                    analysis_period=analysis_period or None,
                    include_executive_summary=include_executive_summary,
                    include_model_fit=include_model_fit,
                    include_channel_roi=include_channel_roi,
                    include_decomposition=include_decomposition,
                    include_saturation=include_saturation,
                    include_diagnostics=include_diagnostics,
                    include_methodology=include_methodology,
                    credible_interval=credible_interval,
                    currency_symbol=currency_symbol,
                )

            st.session_state.report_id = result.get("report_id")
            st.session_state.report_status = "generating"
            st.session_state.generating = True
            st.success(f"Report generation started! ID: {result.get('report_id')}")
            st.rerun()

        except APIError as e:
            display_api_error(e)
        except Exception as e:
            st.error(f"Failed to start report generation: {e}")


# =============================================================================
# Report Status Monitor
# =============================================================================


@st.fragment
def render_report_monitor():
    """Render the report generation status monitor."""
    model_id = st.session_state.selected_model_id
    report_id = st.session_state.report_id

    if not report_id:
        return

    st.markdown("### Report Generation Status")

    try:
        api_client = get_api_client()
        status_data = api_client.get_report_status(model_id, report_id)

        status = status_data.get("status", "unknown")
        message = status_data.get("message", "")
        filename = status_data.get("filename")

        # Status display
        if status == "generating":
            st.info(f"⏳ **Generating...** {message}")

            # Auto-refresh
            progress_placeholder = st.empty()
            with progress_placeholder:
                st.caption("Auto-refreshing in 3 seconds...")

            time.sleep(3)
            st.rerun()

        elif status == "completed":
            st.success(f"✅ **Report Ready!** {message}")
            st.session_state.generating = False

            # Download button
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                try:
                    report_content = api_client.download_report(model_id, report_id)

                    st.download_button(
                        label="📥 Download Report",
                        data=report_content,
                        file_name=filename or f"mmm_report_{report_id}.html",
                        mime="text/html",
                        type="primary",
                        use_container_width=True,
                    )

                    st.caption(f"File: {filename} ({format_bytes(len(report_content))})")

                except Exception as e:
                    st.error(f"Failed to download: {e}")
            if report_content:
                st.markdown("---")
                st.markdown("### 📄 Report Preview")
                
                # Render the HTML report in an iframe-like component
                components.html(
                    report_content.decode("utf-8"),
                    height=800,
                    scrolling=True,
                )
            # Clear button
            if st.button("Generate Another Report", use_container_width=False):
                st.session_state.report_id = None
                st.session_state.report_status = None
                st.rerun()

        elif status == "failed":
            st.error(f"❌ **Generation Failed:** {message}")
            st.session_state.generating = False

            if st.button("Try Again"):
                st.session_state.report_id = None
                st.session_state.report_status = None
                st.rerun()

        else:
            st.warning(f"Unknown status: {status}")

    except APIError as e:
        display_api_error(e)
        st.session_state.generating = False
    except Exception as e:
        st.error(f"Error checking status: {e}")
        st.session_state.generating = False


# =============================================================================
# Existing Reports List
# =============================================================================


@st.fragment
def render_existing_reports():
    """Render list of existing reports for the model."""
    model_id = st.session_state.selected_model_id

    if not model_id:
        return

    st.markdown("### Existing Reports")

    try:
        api_client = get_api_client()
        result = api_client.list_reports(model_id)
        reports = result.get("reports", [])

        if not reports:
            st.info("No reports generated yet for this model.")
            return

        # Report selector + preview layout
        col_list, col_preview = st.columns([1, 2])
        
        with col_list:
            st.markdown("#### Available Reports")
            
            for i, report in enumerate(reports):
                rid = report.get("report_id")
                filename = report.get("filename", "Unknown")
                created_at = report.get("created_at", "")
                size = report.get("size_bytes", 0)
                
                # Format datetime
                date_str = ""
                if created_at:
                    try:
                        dt = datetime.fromisoformat(created_at)
                        date_str = format_datetime(dt)
                    except:
                        date_str = created_at

                # Report card
                with st.container():
                    st.markdown(f"**{filename}**")
                    st.caption(f"{date_str} • {format_bytes(size)}")
                    
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button("👁️", key=f"view_{rid}", help="Preview"):
                            st.session_state.preview_report_id = rid
                    
                    with btn_col2:
                        try:
                            content = api_client.download_report(model_id, rid)
                            st.download_button(
                                "📥",
                                data=content,
                                file_name=filename,
                                mime="text/html",
                                key=f"dl_{rid}",
                                help="Download",
                            )
                        except:
                            st.button("📥", disabled=True, key=f"dl_{rid}")
                    
                    st.markdown("---")

        with col_preview:
            st.markdown("#### Preview")
            
            preview_rid = st.session_state.get("preview_report_id")
            
            if preview_rid:
                try:
                    report_content = api_client.download_report(model_id, preview_rid)
                    components.html(
                        report_content.decode("utf-8"),
                        height=700,
                        scrolling=True,
                    )
                except Exception as e:
                    st.error(f"Failed to load preview: {e}")
            else:
                st.info("Select a report to preview")

    except APIError as e:
        display_api_error(e)
    except Exception as e:
        st.error(f"Error loading reports: {e}")

# =============================================================================
# Main
# =============================================================================


def main():
    """Main page function."""
    page_header(
        "📄 Report Generation",
        "Generate professional HTML reports from fitted models.",
    )

    # Model selector
    if not render_model_selector():
        return

    st.markdown("---")

    # Check if we're monitoring a report
    if st.session_state.report_id:
        render_report_monitor()
    else:
        # Show form
        render_report_form()

    st.markdown("---")

    # Show existing reports
    render_existing_reports()


if __name__ == "__main__":
    main()