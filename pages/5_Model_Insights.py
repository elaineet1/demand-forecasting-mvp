"""
Streamlit page: Model Insights
Model performance metrics, feature importance, and transparency.
"""

import streamlit as st
import pandas as pd
from src import state, config, charts, explainability

# Page config
st.set_page_config(
    page_title="Model Insights",
    page_icon="🧠",
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

st.title("🧠 Model Insights & Transparency")
st.markdown("""
Model performance metrics, feature importance, and insights into forecast accuracy.
""")

# Check if data is loaded
forecast_results = st.session_state.get(config.STATE_FORECAST_RESULTS)
if forecast_results is None:
    st.warning("⚠️ No data loaded yet. Please go to **Upload & Validation** first.")
else:
    model_results = forecast_results.get('model_results')
    planner_output = forecast_results.get('planner_output')
    master_data = forecast_results.get('master_data')
    
    # ========================================================================
    # Model Info
    # ========================================================================
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if model_results:
            st.subheader("🤖 Model Details")
            model_info = model_results.get('training_info', {})
            st.text(f"Model Type: {model_results.get('model_name', 'Unknown')}")
            st.text(f"Training Rows: {model_info.get('training_rows', 'N/A')}")
            st.text(f"Features Used: {model_info.get('features_used', 'N/A')}")
        else:
            st.info("ℹ️ Using fallback forecasting (no ML model trained)")
    
    with col2:
        st.subheader("📊 Forecast Method Coverage")
        if planner_output is not None and not planner_output.empty:
            coverage = explainability.get_fallback_coverage_stats(planner_output)
            st.text(f"ML Coverage: {coverage['ml_coverage']}")
            st.text(f"Fallback Coverage: {coverage['fallback_coverage']}")
            st.text(f"Total SKUs: {coverage['total_skus']}")
    
    # ========================================================================
    # Performance Metrics
    # ========================================================================
    if model_results and 'validation_metrics' in model_results:
        st.divider()
        st.subheader("📈 Model Performance (Holdout Test Set)")
        
        metrics_dict = model_results['validation_metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "MAE",
                f"{metrics_dict.get('mae', 0):.2f}",
                help="Mean Absolute Error - average prediction error in units"
            )
        
        with col2:
            st.metric(
                "RMSE",
                f"{metrics_dict.get('rmse', 0):.2f}",
                help="Root Mean Squared Error - penalizes large errors"
            )
        
        with col3:
            wape = metrics_dict.get('wape', 0)
            accuracy = explainability.get_model_transparency_notes()
            st.metric(
                "WAPE",
                f"{wape:.1f}%",
                help="Weighted Absolute Percentage Error - robust to zero values"
            )
        
        with col4:
            mape = metrics_dict.get('mape')
            if mape is not None:
                st.metric(
                    "MAPE",
                    f"{mape:.1f}%",
                    help="Mean Absolute Percentage Error"
                )
            else:
                st.metric("MAPE", "N/A", help="Not available due to sparsity")
        
        st.caption(
            "Note: MAPE can look high when many SKUs have very small or zero actual sales. "
            "WAPE is typically more stable for retail demand forecasting."
        )

        test_start = metrics_dict.get('test_date_start')
        test_end = metrics_dict.get('test_date_end')
        test_months = metrics_dict.get('test_unique_months')
        if test_start and test_end:
            st.text(f"Test Period: {test_start} to {test_end} ({test_months} months)")
        
        # Interpretation
        performance_text = explainability.get_model_performance_summary(metrics_dict)
        with st.expander("📝 Performance Interpretation", expanded=False):
            st.markdown(performance_text)
    
    else:
        st.info("⚠️ Model validation metrics not available (fallback mode active)")
    
    # ========================================================================
    # Feature Importance
    # ========================================================================
    if model_results and 'feature_importance' in model_results:
        st.divider()
        st.subheader("🎯 Feature Importance")
        
        feature_importance_df = model_results['feature_importance']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = charts.chart_feature_importance(feature_importance_df, top_n=15)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Top 10 Drivers**")
            top_10 = feature_importance_df.head(10)
            for idx, (_, row) in enumerate(top_10.iterrows(), 1):
                st.text(f"{idx}. {row['feature']}")
    
    else:
        st.info("ℹ️ Feature importance data not available")
    
    # ========================================================================
    # Data Quality
    # ========================================================================
    st.divider()
    st.subheader("📊 Data Quality Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Dataset Overview")
        if master_data is not None:
            st.text(f"Total Aggregated Records: {len(master_data)}")
            st.text(f"Unique SKUs: {master_data['item_no'].nunique()}")
            st.text(f"Unique Months: {master_data['snapshot_date'].nunique()}")
            date_range = (
                f"{master_data['snapshot_date'].min().date()} to "
                f"{master_data['snapshot_date'].max().date()}"
            )
            st.text(f"Date Range: {date_range}")
    
    with col2:
        st.subheader("⚠️ Data Observations")
        if master_data is not None:
            month_count = master_data['snapshot_date'].nunique()
        else:
            month_count = "N/A"
        st.markdown(f"""
        - **History Depth**: {month_count} months of sales available
        - **Seasonal Learning**: Limited when history is short or uneven
        - **New Products**: Frequent launches require fallback rules
        - **Sparse Items**: Low-volume SKUs use brand/vendor averages
        """)
    
    # ========================================================================
    # Assumptions and Limitations
    # ========================================================================
    st.divider()
    
    with st.expander("📋 Complete Assumptions & Limitations", expanded=False):
        if master_data is not None:
            month_count = master_data['snapshot_date'].nunique()
            st.markdown(f"**History Depth in This Run**: {month_count} months of sales data")
        assumptions_text = explainability.get_model_assumptions_text()
        st.markdown(assumptions_text)
    
    # ========================================================================
    # Transparency Notes
    # ========================================================================
    st.divider()
    st.subheader("🔍 Model Transparency")
    
    transparency_notes = explainability.get_model_transparency_notes()
    
    for topic, note in transparency_notes.items():
        st.text(f"{topic.replace('_', ' ').upper()}: {note}")
    
    # ========================================================================
    # Recommendations
    # ========================================================================
    st.divider()
    st.subheader("💡 Recommendations for Use")
    
    st.markdown("""
    1. **Decision Support Tool**: Use forecasts as a starting point, not sole truth
    2. **Manual Review**: Prioritize reviewing edge cases and high-reorder items
    3. **Seasonal Adjustment**: Apply manual adjustments for known holidays/campaigns
    4. **Feedback Loop**: Monitor actual sales vs. forecast for continuous improvement
    5. **Data Quality**: Ensure consistent file formats and date parsing for accuracy
    6. **SKU Consolidation**: Consider consolidating obsolete or duplicate SKUs
    7. **Grouping Updates**: Periodically review brand/vendor groupings for accuracy
    """)
    
    # ========================================================================
    # Warnings and Issues
    # ========================================================================
    if forecast_results.get('warnings') or forecast_results.get('errors'):
        st.divider()
        st.subheader("⚠️ Issues Detected")
        
        if forecast_results.get('errors'):
            for error in forecast_results['errors']:
                st.error(error)
        
        if forecast_results.get('warnings'):
            for warning in forecast_results['warnings']:
                st.warning(warning)
