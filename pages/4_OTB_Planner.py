"""
Streamlit page: OTB Planner
Download and review the final Open-to-Buy recommendations.
"""

import streamlit as st
import pandas as pd
import math
from datetime import datetime
from src import state, config

# Page config
st.set_page_config(
    page_title="OTB Planner",
    page_icon="📋",
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

st.title("📋 OTB Planner")
st.markdown("Review, filter, and download Open-to-Buy recommendations.")

# Check if data is loaded
forecast_results = st.session_state.get(config.STATE_FORECAST_RESULTS)
if forecast_results is None:
    st.warning("⚠️ No data loaded yet. Please go to **Upload & Validation** first.")
else:
    planner_output = forecast_results.get('planner_output')
    
    if planner_output is None or (hasattr(planner_output, 'empty') and planner_output.empty):
        st.error("❌ No OTB output available")
    else:
        # ====================================================================
        # Filters and Scenario Options
        # ====================================================================
        with st.form("planner_filter_form"):
            if 'vendor' in planner_output.columns:
                col1, col2, col3, col4 = st.columns(4)
            else:
                col1, col2, col4 = st.columns(3)
            
            with col1:
                min_reorder = st.number_input(
                    "Minimum Reorder Qty",
                    min_value=0,
                    value=0,
                    step=1
                )
            
            with col2:
                status_filter = st.multiselect(
                    "Stock Health Status",
                    options=planner_output['stock_health'].unique(),
                    default=planner_output['stock_health'].unique()
                )
            
            if 'vendor' in planner_output.columns:
                with col3:
                    brand_filter = st.multiselect(
                        "Brand",
                        options=['All'] + sorted(planner_output['vendor'].dropna().unique()),
                        default=['All']
                    )
            else:
                brand_filter = ['All']
        
            with col4:
                item_status_filter = ['All']

            st.markdown("**Scenario Settings**")
            col5, col6, col7 = st.columns(3)
            with col5:
                horizon_months = st.selectbox(
                    "Forward Horizon (Months)",
                    options=[3, 6, 9],
                    index=0
                )
            with col6:
                sales_uplift = st.slider(
                    "Sales Increase (%)",
                    min_value=0,
                    max_value=100,
                    value=0,
                    step=5
                )
            with col7:
                promo_uplift = st.slider(
                    "Promotion Uplift (%)",
                    min_value=0,
                    max_value=100,
                    value=0,
                    step=5
                )
            markdown_effect = st.slider(
                "Markdown Effect (%)",
                min_value=0,
                max_value=100,
                value=0,
                step=5
            )
            
            include_zero_forecast = st.checkbox(
                "Include items with zero forecast",
                value=False
            )
            
            apply_filter = st.form_submit_button("🔍 Apply Filters")
        
        # ====================================================================
        # Apply Filters
        # ====================================================================
        filtered_planner = planner_output.copy()
        
        filtered_planner = filtered_planner[
            filtered_planner['stock_health'].isin(status_filter)
        ]
        
        if 'All' not in (brand_filter or ['All']):
            filtered_planner = filtered_planner[
                filtered_planner['vendor'].isin(brand_filter)
            ]

        # Item status filter removed from UI
        
        filtered_planner = filtered_planner[
            filtered_planner['reorder_qty'] >= min_reorder
        ]
        
        if not include_zero_forecast:
            filtered_planner = filtered_planner[
                filtered_planner['forecast_m1'] > 0
            ]

        # ====================================================================
        # Scenario Projection
        # ====================================================================
        net_multiplier = max(
            0.0,
            1 + (sales_uplift / 100.0) + (promo_uplift / 100.0) - (markdown_effect / 100.0)
        )
        filtered_planner = filtered_planner.copy()
        filtered_planner['scenario_multiplier'] = net_multiplier
        filtered_planner['scenario_monthly_forecast'] = filtered_planner['forecast_m1'] * net_multiplier
        filtered_planner['scenario_horizon_demand'] = filtered_planner['scenario_monthly_forecast'] * horizon_months
        filtered_planner['scenario_reorder_qty'] = (
            filtered_planner['scenario_horizon_demand'] - filtered_planner['total_stock']
        ).clip(lower=0)
        filtered_planner['scenario_overstock_qty'] = (
            filtered_planner['total_stock'] - filtered_planner['scenario_horizon_demand']
        ).clip(lower=0)
        
        # ====================================================================
        # Summary
        # ====================================================================
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total SKUs", len(filtered_planner), delta_color="off")
        
        with col2:
            total_reorder = filtered_planner['scenario_reorder_qty'].sum()
            st.metric("Total Reorder Qty", f"{total_reorder:.0f}", delta_color="off")
        
        with col3:
            total_1m = filtered_planner['scenario_monthly_forecast'].sum()
            st.metric("Scenario 1M Forecast", f"{total_1m:.0f}", delta_color="off")
        
        with col4:
            total_h = filtered_planner['scenario_horizon_demand'].sum()
            st.metric(f"Scenario {horizon_months}M Demand", f"{total_h:.0f}", delta_color="off")
        
        with col5:
            current_stock = filtered_planner['total_stock'].sum()
            st.metric("Current Stock", f"{current_stock:.0f}", delta_color="off")
        
        # ====================================================================
        # Main OTB Table
        # ====================================================================
        st.subheader("📊 Open-to-Buy Planning Table")

        method_labels = {
            'ml_model': 'ML Model',
            'fallback_recent_avg': 'Recent Avg',
            'fallback_category_vendor': 'Brand Avg',
            'fallback_category': 'Overall Avg',
            'fallback_existing_forecast': 'Existing Forecast',
            'fallback_zero': 'Zero',
        }

        filtered_planner = filtered_planner.copy()
        filtered_planner['forecast_method'] = filtered_planner['forecast_method'].map(method_labels).fillna(
            filtered_planner['forecast_method']
        )
        
        # Select columns to display
        display_cols = [
            'item_no',
            'item_description',
            'vendor',
            'manufacturer',
            'total_stock',
            'warehouse_stock',
            'latest_monthly_sales',
            'recent_3m_avg',
            'overstock_qty',
            'latest_monthly_forecast',
            'forecast_m1',
            'forecast_m2',
            'forecast_m3',
            'projected_3m_demand',
            'reorder_qty',
            'scenario_monthly_forecast',
            'scenario_horizon_demand',
            'scenario_reorder_qty',
            'scenario_overstock_qty',
            'stock_health',
            'forecast_method',
            'remark',
            'event_applied_any',
            'base_price',
            'rrp',
        ]
        
        # Filter to available columns
        available_cols = [col for col in display_cols if col in filtered_planner.columns]
        
        base_df = filtered_planner[available_cols].reset_index(drop=True).copy()
        display_df = base_df.copy()
        for col in [
            'latest_monthly_sales', 'total_stock', 'warehouse_stock', 'latest_monthly_forecast',
            'scenario_monthly_forecast', 'scenario_horizon_demand', 'scenario_reorder_qty',
            'scenario_overstock_qty'
        ]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda v: math.ceil(v) if pd.notna(v) else v
                )

        for col in [
            'latest_monthly_sales', 'total_stock', 'warehouse_stock', 'latest_monthly_forecast',
            'scenario_monthly_forecast', 'scenario_horizon_demand', 'scenario_reorder_qty',
            'scenario_overstock_qty'
        ]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda v: pd.NA if pd.notna(v) and float(v) == 0 else v
                )

        def highlight_otb(row):
            styles = [''] * len(row)
            idx = row.name
            if idx is not None:
                if 'scenario_reorder_qty' in base_df.columns:
                    if pd.notna(base_df.at[idx, 'scenario_reorder_qty']) and base_df.at[idx, 'scenario_reorder_qty'] > 0:
                        col_idx = display_df.columns.get_loc('scenario_reorder_qty')
                        styles[col_idx] = 'background-color: #ffe6e6'
                if 'scenario_overstock_qty' in base_df.columns:
                    if pd.notna(base_df.at[idx, 'scenario_overstock_qty']) and base_df.at[idx, 'scenario_overstock_qty'] > 0:
                        col_idx = display_df.columns.get_loc('scenario_overstock_qty')
                        styles[col_idx] = 'background-color: #fff4cc'
            return styles

        styled = display_df.style.apply(highlight_otb, axis=1)
        st.dataframe(
            styled,
            use_container_width=True,
            height=600,
            column_config={
                'reorder_qty': st.column_config.NumberColumn(format="%.0f"),
                'forecast_m1': st.column_config.NumberColumn(format="%.0f"),
                'forecast_m2': st.column_config.NumberColumn(format="%.0f"),
                'forecast_m3': st.column_config.NumberColumn(format="%.0f"),
                'latest_monthly_sales': st.column_config.NumberColumn(format="%.0f"),
                'overstock_qty': st.column_config.NumberColumn(format="%.0f"),
                'projected_3m_demand': st.column_config.NumberColumn(format="%.0f"),
                'scenario_monthly_forecast': st.column_config.NumberColumn(format="%.0f"),
                'scenario_horizon_demand': st.column_config.NumberColumn(format="%.0f"),
                'scenario_reorder_qty': st.column_config.NumberColumn(format="%.0f"),
                'scenario_overstock_qty': st.column_config.NumberColumn(format="%.0f"),
                'total_stock': st.column_config.NumberColumn(format="%.0f"),
                'base_price': st.column_config.NumberColumn(format="%.2f"),
                'rrp': st.column_config.NumberColumn(format="%.2f"),
            }
        )
        
        # ====================================================================
        # Download Options
        # ====================================================================
        st.divider()
        st.subheader("📥 Download OTB Plan")
        
        col1, col2, col3 = st.columns(3)
        
        # CSV Download
        csv_data = filtered_planner[available_cols].to_csv(index=False)
        with col1:
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name=f"company_otb_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Excel Download
        with col2:
            try:
                excel_buffer = pd.ExcelWriter.__enter__
                from io import BytesIO
                
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    filtered_planner[available_cols].to_excel(
                        writer,
                        index=False,
                        sheet_name='OTB_Plan'
                    )
                
                buffer.seek(0)
                st.download_button(
                    label="📥 Download as Excel",
                    data=buffer.getvalue(),
                    file_name=f"company_otb_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.warning(f"Excel export not available: {str(e)}")
        
        # JSON Download
        with col3:
            json_data = filtered_planner[available_cols].to_json(orient='records')
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name=f"company_otb_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # ====================================================================
        # Statistics
        # ====================================================================
        with st.expander("📊 Planning Statistics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Stock Health Breakdown")
                health_counts = filtered_planner['stock_health'].value_counts()
                for status, count in health_counts.items():
                    st.text(f"  {status.replace('_', ' ').title()}: {count}")
            
            with col2:
                st.subheader("Forecast Method Breakdown")
                method_counts = filtered_planner['forecast_method'].value_counts()
                for method, count in method_counts.items():
                    st.text(f"  {method.replace('_', ' ').title()}: {count}")
