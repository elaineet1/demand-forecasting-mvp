"""
Main Streamlit app entry point for ML-Assisted Demand & OTB Forecasting MVP.
"""

import streamlit as st
from src import state, config

# Configure page
st.set_page_config(
    page_title="Demand & OTB Forecasting MVP",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 0rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
state.initialize_session_state()

# Global styling
st.markdown(
    """
<style>
    :root {
        --primary: #1f4e79;
        --accent: #0f6cbd;
        --muted: #6b7280;
        --card: #f8fafc;
        --border: #e5e7eb;
    }
    .block-container { padding-top: 2.25rem; }
    h1, h2, h3 { color: var(--primary); }
    .hero {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2f7 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px 22px;
        margin-bottom: 14px;
    }
    .hero-title { font-size: 28px; font-weight: 700; margin: 0; color: var(--primary); }
    .hero-sub { margin: 4px 0 0; color: var(--muted); font-weight: 500; }
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

# Header
st.markdown(
    """
<div class="hero">
  <div class="hero-title">ML-Assisted Demand & OTB Forecasting</div>
  <div class="hero-sub">Forecasting MVP v0.1 • Theme v1</div>
</div>
""",
    unsafe_allow_html=True,
)

st.divider()

# Introduction
st.markdown("""
Welcome to the **ML-Assisted Demand & OTB Forecasting MVP**!

This application combines machine learning with business rules to help management
and purchasing teams make better inventory decisions.

### 🎯 What This App Does

1. **Ingests** your inventory and sales files
2. **Validates** data quality and normalizes column names
3. **Trains** ML models on historical sales data
4. **Forecasts** next-month, 2-month, and 3-month demand by SKU
5. **Recommends** purchase quantities based on stock targets
6. **Evaluates** stock health (understock, healthy, overstock)
7. **Provides** dashboards and downloadable planning tables

### 📄 How to Use

1. **Upload & Validation** (Step 1)
   - Upload your inventory and sales files
   - The app will automatically normalize column names and validate data
   - Review warnings and summaries

2. **Executive Dashboard** (Step 2)
   - View high-level planning metrics
   - See stock health distribution
   - Identify top reorder items

3. **Forecast Explorer** (Step 3)
   - Drill into SKU-level details
   - View sales history and forecasts
   - Understand forecast drivers

4. **OTB Planner** (Step 4)
   - Review detailed recommendations
   - Filter and sort by various criteria
   - Download planning table as CSV/Excel

5. **Model Insights** (Step 5)
   - Review model performance metrics
   - Understand feature importance
   - Learn about assumptions and limitations

6. **Narrative Co-Pilot** (Step 6)
   - Draft executive and buyer-ready summaries
   - Turn forecast outputs into stakeholder language
   - Highlight priority risks and reorder actions

### ⚙️ Key Features

✅ **Hybrid ML + Fallback Logic**
- Uses gradient boosting (LightGBM) where data exists
- Falls back to brand/vendor averages for sparse items
- Explicit forecast method tracking

✅ **Time-Aware Validation**
- No random shuffling of sales history
- Respects time-series structure

✅ **Stock Health Assessment**
- Understock Risk: < 2 months of projected demand
- Healthy Stock: 2-3 months of projected demand
- Overstock Risk: > 3 months of projected demand

✅ **Transparency & Explainability**
- Feature importance charts
- Per-SKU forecast explanations
- Detailed assumptions documentation

### ⚠️ Important Limitations (MVP)

- **~2 Months Data**: Insufficient for seasonal modeling
- **Sparse History**: Many SKUs will use fallback rules
- **No Trend Breaking**: Assumes stable demand patterns
- **Local Deployment**: Deploy on server for team access

### 📊 Recommended Data Format

**Inventory Files**:
- Date in filename (e.g., `Item List_18.12.2021.xls`)
- Columns: Item No, Description, Vendor, Total Stock, Item Status, Active (Y/N)

**Sales Files**:
- Date or date range in filename
- Columns: Item No, Item Description, Quantity, Sales Amount

### 🚀 Getting Started

Click **"Upload & Validation"** in the sidebar to begin, or contact your analyst for help.
""")

# Sidebar
with st.sidebar:
    st.markdown("## 📚 Navigation")
    st.markdown("""
    The app has 6 main pages:
    
    1. **📤 Upload & Validation**
       - Upload files and start pipeline
    
    2. **📈 Executive Dashboard**
       - High-level metrics and visualizations
    
    3. **🔍 Forecast Explorer**
       - SKU-level detail and drill-down
    
4. **📋 OTB Planner**
       - Download recommendations
    
    5. **🧠 Model Insights**
       - Performance and transparency

    6. **📝 Narrative Co-Pilot**
       - Draft stakeholder-ready summaries
    """)
    
    st.divider()
    
    st.markdown("## ⚙️ Quick Actions")
    
    if st.button("🔄 Clear All Data", use_container_width=True):
        state.clear_all_state()
        st.success("✓ Data cleared")
        st.rerun()
    
    st.divider()
    
    # Status indicator
    st.markdown("## 📊 Status")
    
    if st.session_state.get(config.STATE_FORECAST_RESULTS):
        results = st.session_state[config.STATE_FORECAST_RESULTS]
        if results.get('success'):
            st.success("✅ Forecast ready")
            summary = results.get('planning_summary', {})
            st.text(f"Active SKUs: {summary.get('total_active_skus', 0)}")
            st.text(f"Reorder Qty: {summary.get('total_reorder_qty', 0):.0f}")
        else:
            st.error("❌ Forecast failed")
    else:
        st.info("⏳ Awaiting data upload")

# Footer
st.divider()
st.markdown("""
---
**Version**: 0.1.0 MVP  
**Last Updated**: 2024  
For support or feedback, contact your analytics team.
""")
