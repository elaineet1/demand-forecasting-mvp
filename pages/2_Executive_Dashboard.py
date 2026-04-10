"""
Streamlit page: Executive Dashboard
High-level overview of planning metrics and visualizations.
"""

import streamlit as st
import pandas as pd
import math
from src import state, config, charts

# Page config
st.set_page_config(
    page_title="Executive Dashboard",
    page_icon="📈",
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

st.title("📈 Executive Dashboard")

# Check if data is loaded
forecast_results = st.session_state.get(config.STATE_FORECAST_RESULTS)
if forecast_results is None:
    st.warning("⚠️ No data loaded yet. Please go to **Upload & Validation** first.")
else:
    planner_output = forecast_results.get('planner_output')
    planning_summary = forecast_results.get('planning_summary', {})
    master_data = forecast_results.get('master_data')
    
    if planner_output is None or (hasattr(planner_output, 'empty') and planner_output.empty):
        st.error("❌ No forecast output available")
    else:
        # ====================================================================
        # Key Metrics
        # ====================================================================
        st.subheader("📊 Key Planning Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_active = planning_summary.get('total_active_skus', 0)
            st.metric("Active SKUs", total_active, delta_color="off")
        
        with col2:
            proj_1m = planner_output['forecast_m1'].sum()
            st.metric("Projected Next-Month Demand", f"{proj_1m:.0f} units", delta_color="off")
        
        with col3:
            proj_3m = planner_output['projected_3m_demand'].sum()
            st.metric("Projected 3-Month Demand", f"{proj_3m:.0f} units", delta_color="off")
        
        with col4:
            total_reorder = math.ceil(planner_output['reorder_qty'].sum())
            st.metric("Total Recommended Reorder", f"{total_reorder:.0f} units", delta_color="off")
        
        with col5:
            current_stock = planner_output['total_stock'].sum()
            st.metric("Current Total Stock", f"{current_stock:.0f} units", delta_color="off")
        
        # ====================================================================
        # Stock Health Overview
        # ====================================================================
        col1, col2 = st.columns([2, 2])
        
        with col1:
            st.subheader("📊 Stock Health Distribution")
            health_fig = charts.chart_active_skus_by_health(planner_output)
            st.plotly_chart(health_fig, use_container_width=True)
        
        with col2:
            st.subheader("📋 Health Details")
            health_stats = planner_output['stock_health'].value_counts()
            for status in ['understock_risk', 'healthy_stock', 'overstock_risk']:
                count = health_stats.get(status, 0)
                if status == 'understock_risk':
                    icon = '⚠️'
                elif status == 'healthy_stock':
                    icon = '✓'
                else:
                    icon = '📦'
                st.text(f"{icon} {status.replace('_', ' ').title()}: {count} SKUs")
        
        # ====================================================================
        # Forecast Method Distribution
        # ====================================================================
        col1, col2 = st.columns([2, 2])
        
        with col1:
            st.subheader("🎯 Forecast Method Distribution")
            method_fig = charts.chart_forecast_method_distribution(planner_output)
            st.plotly_chart(method_fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Reorder by Top Brands")
            vendor_fig = charts.chart_vendor_reorder_totals(planner_output, top_n=10)
            st.plotly_chart(vendor_fig, use_container_width=True)
        
        # ====================================================================
        # Top Reorder Items
        # ====================================================================
        st.subheader("🛒 Top Items: Reorder and Overstock")

        if 'overstock_qty' not in planner_output.columns:
            planner_output = planner_output.copy()
            planner_output['overstock_qty'] = (
                planner_output.get('total_stock', 0) - planner_output.get('projected_3m_demand', 0)
            ).clip(lower=0)

        top_reorder = planner_output[planner_output['reorder_qty'] > 0].nlargest(10, 'reorder_qty')
        top_overstock = planner_output[planner_output['stock_health'] == 'overstock_risk'].nlargest(
            10,
            'overstock_qty'
        )

        top_reorder = top_reorder.assign(action='Reorder')
        top_overstock = top_overstock.assign(action='Overstock')

        combined = pd.concat([top_reorder, top_overstock], ignore_index=True)

        if len(combined) > 0:
            method_labels = {
                'ml_model': 'ML Model',
                'fallback_recent_avg': 'Recent Avg',
                'fallback_category_vendor': 'Brand Avg',
                'fallback_category': 'Overall Avg',
                'fallback_existing_forecast': 'Existing Forecast',
                'fallback_zero': 'Zero',
            }
            combined = combined.copy()
            combined['forecast_method'] = combined['forecast_method'].map(method_labels).fillna(combined['forecast_method'])
            display_cols = [
                'action', 'item_no', 'item_description', 'total_stock',
                'latest_monthly_sales', 'recent_3m_avg', 'forecast_m1', 'projected_3m_demand',
                'reorder_qty', 'overstock_qty', 'stock_health', 'forecast_method', 'remark'
            ]
            available_cols = [col for col in display_cols if col in combined.columns]

            combined_display = combined[available_cols].reset_index(drop=True).copy()
            for col in [
                'latest_monthly_sales', 'total_stock', 'forecast_m1',
                'projected_3m_demand', 'reorder_qty', 'overstock_qty'
            ]:
                if col in combined_display.columns:
                    combined_display[col] = combined_display[col].apply(
                        lambda v: math.ceil(v) if pd.notna(v) else v
                    )

            zero_hide_cols = ['latest_monthly_sales', 'total_stock', 'reorder_qty', 'overstock_qty']
            for col in zero_hide_cols:
                if col in combined_display.columns:
                    combined_display[col] = combined_display[col].apply(
                        lambda v: "" if pd.notna(v) and float(v) == 0 else v
                    )

            def highlight_actions(row):
                if row.get('action') == 'Reorder' and row.get('reorder_qty', 0) > 0:
                    return ['background-color: #ffe6e6'] * len(row)
                if row.get('action') == 'Overstock' and row.get('overstock_qty', 0) > 0:
                    return ['background-color: #fff4cc'] * len(row)
                return [''] * len(row)

            styled = combined_display.style.apply(highlight_actions, axis=1)
            st.dataframe(styled, use_container_width=True, height=400)
        else:
            st.info("✓ No items require reorder or overstock action at this time")
        
        # ====================================================================
        # Vendor Analysis
        # ====================================================================
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📦 Reorder by Top Brands")
            vendor_fig = charts.chart_vendor_reorder_totals(planner_output, top_n=8)
            st.plotly_chart(vendor_fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Stock vs. 3-Month Demand")
            demand_fig = charts.chart_forecast_vs_stock(planner_output, max_items=10)
            st.plotly_chart(demand_fig, use_container_width=True)
        
        # ====================================================================
        # Data Quality Notes
        # ====================================================================
        with st.expander("ℹ️ Data Quality & Warnings"):
            if forecast_results.get('warnings'):
                for warning in forecast_results['warnings']:
                    st.warning(f"⚠️ {warning}")
            else:
                st.success("✓ No data quality warnings")
            
            if master_data is not None:
                st.text(f"Data includes {master_data['snapshot_date'].nunique()} unique months")
                st.text(f"Date range: {master_data['snapshot_date'].min().date()} to {master_data['snapshot_date'].max().date()}")
                st.info(
                    "Data readiness guide: minimum 3 months to run, 6–12 months recommended, "
                    "12+ months best for seasonality. Sparse SKUs may still use fallback."
                )
