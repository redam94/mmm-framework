"""
Data Management Page.

Upload, view, and manage MFF datasets.
"""

import streamlit as st
import pandas as pd
import io

from api_client import (
    get_api_client,
    fetch_datasets,
    clear_data_cache,
    APIError,
    DatasetInfo,
)
from components import (
    apply_custom_css,
    page_header,
    format_bytes,
    format_datetime,
    display_api_error,
    data_preview_table,
    summary_statistics,
    init_session_state,
    confirm_dialog,
)


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Data Management | MMM Framework",
    page_icon="📁",
    layout="wide",
)

apply_custom_css()

init_session_state(
    selected_data_id=None,
    upload_success=False,
)


# =============================================================================
# Data Upload Section
# =============================================================================

@st.fragment
def render_upload_section():
    """Render the data upload section."""
    st.markdown("### Upload New Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "parquet", "xlsx", "xls"],
        help="Upload data in MFF format. Supported formats: CSV, Parquet, Excel",
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info(f"📄 {uploaded_file.name} ({format_bytes(uploaded_file.size)})")
        
        with col2:
            if st.button("Upload", type="primary", use_container_width=True):
                with st.spinner("Uploading..."):
                    try:
                        client = get_api_client()
                        content = uploaded_file.getvalue()
                        result = client.upload_data(content, uploaded_file.name)
                        
                        st.success(f"✅ Uploaded successfully! ID: {result.data_id}")
                        st.session_state.selected_data_id = result.data_id
                        st.session_state.upload_success = True
                        
                        # Clear cache to show new data
                        clear_data_cache()
                        st.rerun()
                        
                    except APIError as e:
                        display_api_error(e)
                    except Exception as e:
                        st.error(f"Upload failed: {e}")
        
        # Preview uploaded file
        with st.expander("Preview File Contents", expanded=False):
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
                elif uploaded_file.name.endswith('.parquet'):
                    df = pd.read_parquet(io.BytesIO(uploaded_file.getvalue()))
                else:
                    df = pd.read_excel(io.BytesIO(uploaded_file.getvalue()))
                
                st.dataframe(df.head(20), use_container_width=True)
                st.caption(f"Showing first 20 of {len(df)} rows, {len(df.columns)} columns")
                
            except Exception as e:
                st.error(f"Could not preview file: {e}")


# =============================================================================
# Dataset List Section
# =============================================================================

@st.fragment
def render_dataset_list():
    """Render the list of existing datasets."""
    st.markdown("### Existing Datasets")
    
    try:
        client = get_api_client()
        datasets = fetch_datasets(client)
        
        if not datasets:
            st.info("No datasets found. Upload a file to get started.")
            return
        
        # Display datasets in a table-like format
        for dataset in datasets:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])
                
                with col1:
                    is_selected = st.session_state.selected_data_id == dataset.data_id
                    icon = "✅" if is_selected else "📄"
                    st.markdown(f"**{icon} {dataset.filename}**")
                
                with col2:
                    st.caption(f"ID: {dataset.data_id[:8]}...")
                
                with col3:
                    st.caption(f"{dataset.rows:,} rows × {dataset.columns} cols")
                
                with col4:
                    st.caption(format_bytes(dataset.size_bytes))
                
                with col5:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("📋", key=f"select_{dataset.data_id}", help="Select"):
                            st.session_state.selected_data_id = dataset.data_id
                            st.rerun()
                    with col_b:
                        if st.button("🗑️", key=f"delete_{dataset.data_id}", help="Delete"):
                            st.session_state[f"confirm_delete_{dataset.data_id}"] = True
                            st.rerun()
                
                # Confirm delete dialog
                if st.session_state.get(f"confirm_delete_{dataset.data_id}", False):
                    st.warning(f"⚠️ Delete '{dataset.filename}'? This cannot be undone.")
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("Yes, delete", key=f"confirm_yes_{dataset.data_id}", type="primary"):
                            try:
                                client.delete_dataset(dataset.data_id)
                                st.success("Deleted!")
                                if st.session_state.selected_data_id == dataset.data_id:
                                    st.session_state.selected_data_id = None
                                clear_data_cache()
                                del st.session_state[f"confirm_delete_{dataset.data_id}"]
                                st.rerun()
                            except APIError as e:
                                display_api_error(e)
                    with col_no:
                        if st.button("Cancel", key=f"confirm_no_{dataset.data_id}"):
                            del st.session_state[f"confirm_delete_{dataset.data_id}"]
                            st.rerun()
                
                st.markdown("---")
                
    except APIError as e:
        display_api_error(e)
    except Exception as e:
        st.error(f"Error loading datasets: {e}")


# =============================================================================
# Dataset Details Section
# =============================================================================

@st.fragment
def render_dataset_details():
    """Render details for the selected dataset."""
    st.markdown("### Dataset Details")
    
    data_id = st.session_state.selected_data_id
    
    if not data_id:
        st.info("Select a dataset to view details.")
        return
    
    try:
        client = get_api_client()
        dataset = client.get_dataset(data_id, include_preview=True, preview_rows=50)
        
        # Basic info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", f"{dataset.rows:,}")
        with col2:
            st.metric("Columns", dataset.columns)
        with col3:
            st.metric("Size", format_bytes(dataset.size_bytes))
        with col4:
            st.metric("Variables", len(dataset.variables))
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 Preview", "📊 Variables", "🏷️ Dimensions"])
        
        with tab1:
            if dataset.preview:
                df = pd.DataFrame(dataset.preview)
                st.dataframe(df, use_container_width=True)
                st.caption(f"Showing first {len(df)} rows")
            else:
                st.info("No preview available")
        
        with tab2:
            # Get variable summary
            try:
                var_info = client.get_dataset_variables(data_id)
                if var_info.get("variables"):
                    var_df = pd.DataFrame(var_info["variables"])
                    st.dataframe(var_df, use_container_width=True)
                else:
                    st.info("Variable summary not available")
            except Exception as e:
                st.warning(f"Could not load variable summary: {e}")
        
        with tab3:
            if dataset.dimensions:
                for dim_name, values in dataset.dimensions.items():
                    with st.expander(f"**{dim_name}** ({len(values)} values)"):
                        # Show as tags
                        st.markdown(", ".join([f"`{v}`" for v in values[:20]]))
                        if len(values) > 20:
                            st.caption(f"...and {len(values) - 20} more")
            else:
                st.info("No dimension information available")
        
        # Actions
        st.markdown("---")
        st.markdown("### Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔬 Use for Model", use_container_width=True, type="primary"):
                st.switch_page("pages/2_Configuration.py")
        
        with col2:
            if st.button("📥 Download", use_container_width=True):
                # Create download link
                st.info("Download functionality would go here")
        
        with col3:
            if st.button("🗑️ Delete", use_container_width=True):
                st.session_state[f"confirm_delete_{data_id}"] = True
                st.rerun()
        
    except APIError as e:
        display_api_error(e)
    except Exception as e:
        st.error(f"Error loading dataset details: {e}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main page function."""
    page_header(
        "📁 Data Management",
        "Upload, view, and manage your MFF datasets."
    )
    
    # Two-column layout
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        render_upload_section()
        st.markdown("---")
        render_dataset_list()
    
    with col_right:
        render_dataset_details()


if __name__ == "__main__":
    main()