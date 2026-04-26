"""
Streamlit page: Upload and Validation
Handles file uploads, data quality checks, and column mapping display.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from src import (
    state, config, io_utils, column_mapper,
    validators, forecasting
)

# Page config
st.set_page_config(
    page_title="Upload & Validation",
    page_icon="📤",
    layout="wide"
)

# Initialize session state
state.initialize_session_state()

st.markdown(
    """
<style>
    :root {
        --primary: #1f4e79;
        --muted: #6b7280;
        --card: #f8fafc;
        --border: #e5e7eb;
    }
    h1, h2, h3 { color: var(--primary); }
    div[data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 10px 12px;
    }
    div[data-testid="stMetric"] label { color: var(--muted); }
</style>
""",
    unsafe_allow_html=True,
)

st.title("📤 Upload & Validation")
st.markdown("""
Upload inventory and sales files to start forecasting.
The app will normalize column names and validate data quality.
""")

# Data readiness guidance
st.info(
    "Data readiness guide: minimum 3 months to run, 6–12 months recommended, "
    "12+ months best for seasonality. Sparse SKUs may still use fallback."
)

# ============================================================================
# Demo Data Loader
# ============================================================================
def _run_pipeline(inventory_data, sales_data, calendar_data, use_simple_3m, show_dates: bool = False):
    if show_dates and (inventory_data or sales_data):
        st.subheader("📅 Detected File Dates")
        date_rows = []
        for _, parsed_date, filename in inventory_data:
            date_rows.append({
                "file": filename,
                "type": "inventory",
                "detected_date": parsed_date.date().isoformat() if parsed_date else "NOT DETECTED"
            })
        for _, parsed_date, filename in sales_data:
            date_rows.append({
                "file": filename,
                "type": "sales",
                "detected_date": parsed_date.date().isoformat() if parsed_date else "NOT DETECTED"
            })
        st.dataframe(date_rows, use_container_width=True, height=220)

    pipeline_results = forecasting.end_to_end_forecast_pipeline(
        inventory_data,
        sales_data,
        calendar_df=calendar_data,
        use_simple_3m=use_simple_3m,
        verbose=True
    )

    # Store results in session state
    st.session_state[config.STATE_INVENTORY_FILES] = inventory_data
    st.session_state[config.STATE_SALES_FILES] = sales_data
    st.session_state[config.STATE_CALENDAR_FILE] = calendar_data
    st.session_state[config.STATE_FORECAST_RESULTS] = pipeline_results

    if pipeline_results['success']:
        st.success("✅ Files processed successfully!")
        st.balloons()
        st.session_state[config.STATE_MASTER_DATA] = pipeline_results['master_data']
        st.session_state[config.STATE_MODEL] = pipeline_results['model']
        st.session_state[config.STATE_MODEL_METRICS] = pipeline_results['model_results']
    else:
        st.error("❌ Pipeline failed")
        for error in pipeline_results['errors']:
            st.error(f"  • {error}")

    if pipeline_results['warnings']:
        for warning in pipeline_results['warnings']:
            st.warning(f"  ⚠️ {warning}")


st.subheader("🧪 Demo Data (Simulated)")
st.caption("Use a clean 12‑month simulated dataset to test the app end‑to‑end.")
if st.button("Load Simulated 1‑Year Data"):
    with st.spinner("🔄 Loading simulated data..."):
        inventory_data = []
        sales_data = []
        calendar_data = None

        inv_dir = "data/simulated/Inventory"
        sales_dir = "data/simulated/Sales"

        inv_files = sorted(Path(inv_dir).glob("*.csv"))
        sales_files = sorted(Path(sales_dir).glob("*.csv"))

        if not inv_files or not sales_files:
            st.error("❌ Simulated data not found. Please generate it first.")
        else:
            for p in inv_files:
                df, parsed_date = io_utils.read_csv_file(str(p))
                inventory_data.append((df, parsed_date, p.name))

            for p in sales_files:
                df, parsed_date = io_utils.read_csv_file(str(p))
                sales_data.append((df, parsed_date, p.name))

            _run_pipeline(
                inventory_data=inventory_data,
                sales_data=sales_data,
                calendar_data=calendar_data,
                use_simple_3m=False,
                show_dates=True,
            )

# ============================================================================
# File Upload Section
# ============================================================================
with st.form("upload_form", clear_on_submit=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📦 Inventory Files")
        inventory_files = st.file_uploader(
            "Upload one or more inventory files (.xls, .xlsx, .csv)",
            accept_multiple_files=True,
            type=['xls', 'xlsx', 'csv'],
            key="inventory_uploader"
        )
    
    with col2:
        st.subheader("📊 Sales Files")
        sales_files = st.file_uploader(
            "Upload one or more sales files (.xls, .xlsx, .csv)",
            accept_multiple_files=True,
            type=['xls', 'xlsx', 'csv'],
            key="sales_uploader"
        )
    
    st.subheader("📅 Event Calendar (Optional)")
    calendar_file = st.file_uploader(
        "Upload event calendar for holiday/campaign adjustments (.csv, .xlsx). "
        "Optional scope columns: vendor (brand), manufacturer.",
        type=['csv', 'xlsx'],
        key="calendar_uploader"
    )
    
    use_simple_3m = st.checkbox(
        "Use simple 3×M1 estimation for 3-month demand",
        value=False,
        help="If unchecked, will estimate M2 and M3 separately (experimental)"
    )
    
    submitted = st.form_submit_button("🚀 Process Files & Build Forecast")

# ============================================================================
# Process Files
# ============================================================================
if submitted:
    if not inventory_files or not sales_files:
        st.error("❌ Please upload at least one inventory file and one sales file")
    else:
        with st.spinner("🔄 Processing files..."):
            # Read uploaded files
            inventory_data = []
            sales_data = []
            calendar_data = None
            
            errors = []
            
            # Read inventory files
            for inv_file in inventory_files:
                try:
                    df, parsed_date, filename = io_utils.read_uploaded_file(inv_file)
                    inventory_data.append((df, parsed_date, filename))
                    if parsed_date is None:
                        errors.append(f"Inventory {filename}: No date detected in filename; using today's date.")
                except Exception as e:
                    errors.append(f"Inventory {inv_file.name}: {str(e)}")
            
            # Read sales files
            for sales_file in sales_files:
                try:
                    df, parsed_date, filename = io_utils.read_uploaded_file(sales_file)
                    sales_data.append((df, parsed_date, filename))
                    if parsed_date is None:
                        errors.append(f"Sales {filename}: No date detected in filename; using today's date.")
                except Exception as e:
                    errors.append(f"Sales {sales_file.name}: {str(e)}")
            
            # Read calendar if provided
            if calendar_file:
                try:
                    df, _, _ = io_utils.read_uploaded_file(calendar_file)
                    calendar_data = df
                except Exception as e:
                    errors.append(f"Calendar {calendar_file.name}: {str(e)}")
            
            # Show any read errors
            if errors:
                for error in errors:
                    st.warning(f"⚠️ {error}")

            # Show detected dates summary
            if inventory_data or sales_data:
                st.subheader("📅 Detected File Dates")
                date_rows = []
                for _, parsed_date, filename in inventory_data:
                    date_rows.append({
                        "file": filename,
                        "type": "inventory",
                        "detected_date": parsed_date.date().isoformat() if parsed_date else "NOT DETECTED"
                    })
                for _, parsed_date, filename in sales_data:
                    date_rows.append({
                        "file": filename,
                        "type": "sales",
                        "detected_date": parsed_date.date().isoformat() if parsed_date else "NOT DETECTED"
                    })
                st.dataframe(date_rows, use_container_width=True, height=220)
            
            # Run full pipeline
            if inventory_data and sales_data:
                pipeline_results = forecasting.end_to_end_forecast_pipeline(
                    inventory_data,
                    sales_data,
                    calendar_df=calendar_data,
                    use_simple_3m=use_simple_3m,
                    verbose=True
                )
                
                # Store results in session state
                st.session_state[config.STATE_INVENTORY_FILES] = inventory_data
                st.session_state[config.STATE_SALES_FILES] = sales_data
                st.session_state[config.STATE_CALENDAR_FILE] = calendar_data
                st.session_state[config.STATE_FORECAST_RESULTS] = pipeline_results
                
                # Show results
                if pipeline_results['success']:
                    st.success("✅ Files processed successfully!")
                    st.balloons()
                    
                    # Store in session state
                    st.session_state[config.STATE_MASTER_DATA] = pipeline_results['master_data']
                    st.session_state[config.STATE_MODEL] = pipeline_results['model']
                    st.session_state[config.STATE_MODEL_METRICS] = pipeline_results['model_results']
                    
                else:
                    st.error("❌ Pipeline failed")
                    for error in pipeline_results['errors']:
                        st.error(f"  • {error}")
                
                # Show warnings
                if pipeline_results['warnings']:
                    for warning in pipeline_results['warnings']:
                        st.warning(f"  ⚠️ {warning}")

# ============================================================================
# Display Summary if Data Loaded
# ============================================================================
forecast_results = st.session_state.get(config.STATE_FORECAST_RESULTS)
if forecast_results is not None:
    st.divider()
    st.subheader("📊 Processing Summary")
    
    results = forecast_results
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Inventory Rows",
            results.get('inventory_processed', 0),
            delta_color="off"
        )
    
    with col2:
        st.metric(
            "Sales Rows",
            results.get('sales_processed', 0),
            delta_color="off"
        )
    
    with col3:
        master = results.get('master_data')
        if master is not None:
            st.metric(
                "Unique SKUs",
                master['item_no'].nunique(),
                delta_color="off"
            )
    
    with col4:
        summary = results.get('planning_summary') or {}
        st.metric(
            "Active SKUs",
            summary.get('total_active_skus', 0),
            delta_color="off"
        )
    
    # Show data samples
    st.subheader("📋 Column Mapping Summary")
    
    inv_files = st.session_state.get(config.STATE_INVENTORY_FILES)
    if inv_files and len(inv_files) > 0:
        inv_df, _, inv_filename = inv_files[0]
        mapping = column_mapper.get_column_mapping_summary(list(inv_df.columns))
        
        with st.expander("Inventory columns detected"):
            for orig, standard in sorted(mapping.items()):
                st.text(f"  {orig} → {standard}")
        
        unmapped = column_mapper.get_unmapped_columns(list(inv_df.columns))
        if unmapped:
            st.warning(f"⚠️ Some inventory columns could not be mapped: {', '.join(unmapped)}")
    
    # Navigation
    st.divider()
    st.info("✨ Files are ready! Navigate to **Executive Dashboard** to see insights.")

from src import rag as _rag
_rag.render_sidebar_chat()
