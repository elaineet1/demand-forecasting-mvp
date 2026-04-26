"""
Streamlit page: Forecast Explorer
SKU-level drill-down with filtering and detailed forecasts.
"""

import streamlit as st
import pandas as pd
from src import state, config, charts, explainability

# Page config
st.set_page_config(
    page_title="Forecast Explorer",
    page_icon="🔍",
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

st.title("🔍 Forecast Explorer")
st.markdown("Drill down into SKU-level forecasts and sales history.")

# Check if data is loaded
forecast_results = st.session_state.get(config.STATE_FORECAST_RESULTS)
if forecast_results is None:
    st.warning("⚠️ No data loaded yet. Please go to **Upload & Validation** first.")
else:
    planner_output = forecast_results.get('planner_output')
    master_data = forecast_results.get('master_data')
    model_results = forecast_results.get('model_results')
    
    if planner_output is None or (hasattr(planner_output, 'empty') and planner_output.empty):
        st.error("❌ No forecast output available")
    else:
        # ====================================================================
        # Filters
        # ====================================================================
        with st.form("filter_form"):
            col1, col2 = st.columns(2)
            
            # Filter by health status
            with col1:
                health_options = ['All'] + list(planner_output['stock_health'].unique())
                selected_health = st.selectbox(
                    "Filter by Stock Health",
                    health_options,
                    key="health_filter"
                )
            
            # Filter by brand (vendor)
            with col2:
                vendor_options = ['All'] + sorted(planner_output['vendor'].dropna().unique())
                selected_vendor = st.selectbox(
                    "Filter by Brand",
                    vendor_options,
                    key="vendor_filter"
                )
            
            apply_filters = st.form_submit_button("🔍 Apply Filters")
        
        # ====================================================================
        # Apply Filters
        # ====================================================================
        filtered_df = planner_output.copy()

        if selected_health != 'All':
            filtered_df = filtered_df[filtered_df['stock_health'] == selected_health]

        if selected_vendor != 'All':
            filtered_df = filtered_df[filtered_df['vendor'] == selected_vendor]

        # ====================================================================
        # Summary
        # ====================================================================
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("SKUs Found", len(filtered_df), delta_color="off")
        
        with col2:
            st.metric(
                "Total Reorder Qty",
                f"{filtered_df['reorder_qty'].sum():.0f}",
                delta_color="off"
            )
        
        with col3:
            st.metric(
                "Avg Unit Stock",
                f"{filtered_df['total_stock'].mean():.0f}",
                delta_color="off"
            )
        
        with col4:
            st.metric(
                "Avg Forecast M1",
                f"{filtered_df['forecast_m1'].mean():.0f}",
                delta_color="off"
            )
        
        # ====================================================================
        # SKU Selection for Detailed View
        # ====================================================================
        st.subheader("📍 Select SKU for Detailed Analysis")
        
        if len(filtered_df) > 0:
            filtered_df = filtered_df.reset_index(drop=True)
            sku_options = filtered_df.apply(
                lambda x: f"{x['item_no']} - {str(x['item_description'])[:30]}",
                axis=1
            ).tolist()
            
            selected_sku_label = st.selectbox(
                "Choose an SKU",
                sku_options,
                key="sku_selector"
            )
            
            selected_matches = filtered_df.loc[
                sku_options.index(selected_sku_label):sku_options.index(selected_sku_label)
            ]

            if selected_matches.empty:
                st.warning(
                    "The selected SKU is no longer available for the current filters. "
                    "Please choose another SKU."
                )
            else:
                selected_row = selected_matches.iloc[0]
                selected_item_no = selected_row['item_no']
                
                # ============================================================
                # Detailed View
                # ============================================================
                st.divider()
                st.subheader(f"📦 {selected_item_no} - {selected_row['item_description']}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Brand", selected_row.get('vendor', 'N/A'))
                    st.metric("Item No", selected_row.get('item_no', 'N/A'))
                
                with col2:
                    st.metric("Current Stock", f"{selected_row['total_stock']:.0f}")
                    cover = selected_row.get('stock_cover_months')
                    st.metric(
                        "Stock Cover (months)",
                        f"{cover:.1f}" if pd.notna(cover) else "N/A"
                    )
                    latest_sales = selected_row.get('latest_monthly_sales')
                    if pd.notna(latest_sales):
                        st.metric("Latest Monthly Sales", f"{latest_sales:.0f}")
                    else:
                        st.metric("Latest Monthly Sales", "N/A")
                    recent_avg = selected_row.get('recent_3m_avg')
                    if pd.notna(recent_avg):
                        st.metric("Recent 3M Avg", f"{recent_avg:.0f}")
                    else:
                        st.metric("Recent 3M Avg", "N/A")
                    st.metric("Forecast M1", f"{selected_row['forecast_m1']:.0f}")
                
                with col3:
                    st.metric("3-M Demand", f"{selected_row['projected_3m_demand']:.0f}")
                    st.metric("Reorder Qty", f"{selected_row['reorder_qty']:.0f}")
                    event_applied = selected_row.get('event_applied_any', False)
                    st.metric("Event Calendar Applied", "Yes" if event_applied else "No")
                
                # Show explanation
                explanation = explainability.generate_sku_explanation(selected_row)
                with st.expander("📝 Forecast Explanation", expanded=True):
                    st.text(explanation)
                
                # Show sales history if available
                if master_data is not None:
                    sku_history = master_data[master_data['item_no'] == selected_item_no].copy()
                    
                    if len(sku_history) > 0:
                        st.subheader("📈 Sales History")
                        
                        # Time series chart
                        history_fig = charts.chart_daily_sales_trend(
                            sku_history,
                            item_no=selected_item_no
                        )
                        st.plotly_chart(history_fig, use_container_width=True)
                        
                        # Data table
                        with st.expander("📋 Historical Data"):
                            display_cols = [
                                'snapshot_date', 'quantity', 'sales_amt', 'gross_profit_pct'
                            ]
                            available_cols = [col for col in display_cols if col in sku_history.columns]
                            st.dataframe(
                                sku_history[available_cols].sort_values(
                                    'snapshot_date',
                                    ascending=False
                                ),
                                use_container_width=True
                            )
        
        else:
            st.info("No SKUs match the selected filters")
        
        # ====================================================================
        # Forecast Method Explanation
        # ====================================================================
        with st.expander("ℹ️ Forecast Methods Explained"):
            st.markdown("""
            - **ML Model**: Based on historical sales pattern using gradient boosting
            - **Fallback (Recent Avg)**: Uses recent 3-month average for low-volume SKUs
            - **Fallback (Brand Avg)**: Average of similar products from the same brand
            - **Fallback (All Brands Avg)**: Overall average when brand data is insufficient
            - **Fallback (Existing)**: Using existing forecast_qty from inventory file
            - **Fallback (Zero)**: No sales history available
            """)
        
        # ====================================================================
        # Full Table
        # ====================================================================
        st.subheader("📊 All Filtered SKUs")
        
        method_labels = {
            'ml_model': 'ML Model',
            'fallback_recent_avg': 'Recent Avg',
            'fallback_category_vendor': 'Brand Avg',
            'fallback_category': 'Overall Avg',
            'fallback_existing_forecast': 'Existing Forecast',
            'fallback_zero': 'Zero',
        }

        filtered_df = filtered_df.copy()
        filtered_df['forecast_method'] = filtered_df['forecast_method'].map(method_labels).fillna(filtered_df['forecast_method'])

        display_cols = [
            'item_no', 'item_description', 'vendor',
            'total_stock', 'stock_cover_months',
            'latest_monthly_sales', 'recent_3m_avg', 'forecast_m1', 'forecast_m2', 'forecast_m3',
            'projected_3m_demand', 'reorder_qty', 'stock_health',
            'forecast_method', 'remark', 'event_applied_any'
        ]
        available_cols = [col for col in display_cols if col in filtered_df.columns]

        st.dataframe(
            filtered_df[available_cols].reset_index(drop=True),
            use_container_width=True,
            height=600,
            column_config={
                'stock_cover_months': st.column_config.NumberColumn(
                    label='Stock Cover (months)', format='%.1f'
                ),
            }
        )

from src import rag as _rag
_rag.render_sidebar_chat()
