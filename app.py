"""
Main Streamlit app entry point for ML-Assisted Demand & OTB Forecasting MVP.
"""

import streamlit as st
from src import state, config, rag, persistence

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

# Restore persisted forecast into session state if this is a fresh session
if not st.session_state.get(config.STATE_FORECAST_RESULTS) and persistence.has_saved_run():
    saved = persistence.load_run()
    if saved:
        st.session_state[config.STATE_FORECAST_RESULTS] = saved

# Global styling
st.markdown(
    """
<style>
    :root {
        --primary: #0f1728;
        --accent: #4f8cff;
        --accent-2: #9ac7ff;
        --muted: #667085;
        --card: #ffffff;
        --border: rgba(15, 23, 40, 0.08);
        --ink: #0f1728;
        --bg: #f5f7fb;
        --panel: #eef4ff;
    }
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(79, 140, 255, 0.16), transparent 24%),
            radial-gradient(circle at top left, rgba(198, 219, 255, 0.45), transparent 28%),
            linear-gradient(180deg, #fbfcfe 0%, var(--bg) 100%);
    }
    .block-container { padding-top: 2.1rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: var(--primary); letter-spacing: -0.03em; }
    .hero {
        background:
            linear-gradient(135deg, rgba(255,255,255,0.92) 0%, rgba(246,249,255,0.96) 55%, rgba(237,244,255,0.98) 100%);
        border: 1px solid rgba(255,255,255,0.65);
        box-shadow: 0 24px 64px rgba(15, 23, 40, 0.08);
        backdrop-filter: blur(12px);
        border-radius: 30px;
        padding: 22px 28px 20px;
        margin-bottom: 14px;
        position: relative;
        overflow: hidden;
    }
    .hero:before {
        content: "";
        position: absolute;
        width: 360px;
        height: 360px;
        right: -90px;
        top: -130px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(79,140,255,0.18) 0%, rgba(79,140,255,0.04) 56%, transparent 70%);
    }
    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin: 0;
        color: var(--ink);
        letter-spacing: -0.03em;
        max-width: 580px;
        line-height: 1.02;
    }
    .hero-sub {
        margin: 12px 0 0;
        color: var(--muted);
        font-weight: 500;
        max-width: 560px;
        font-size: 16px;
        line-height: 1.45;
    }
    .hero-kicker {
        font-size: 12px;
        font-weight: 700;
        color: var(--accent);
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 14px;
    }
    .hero-shell {
        display: grid;
        grid-template-columns: 1fr;
        gap: 16px;
        align-items: start;
    }
    .hero-copy {
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        padding-top: 2px;
        max-width: 780px;
    }
    .hero-actions {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 14px;
    }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        background: rgba(79,140,255,0.10);
        color: var(--primary);
        border: 1px solid rgba(79,140,255,0.16);
        padding: 9px 13px;
        font-size: 12px;
        font-weight: 700;
    }
    .hero-visual {
        position: relative;
        min-height: 220px;
        border-radius: 26px;
        background:
            linear-gradient(180deg, rgba(255,255,255,0.82) 0%, rgba(250,252,255,0.98) 100%);
        border: 1px solid rgba(15,23,40,0.06);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.7), 0 18px 40px rgba(79,140,255,0.10);
        overflow: hidden;
        max-width: 960px;
    }
    .hero-visual svg {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
    }
    .hero-visual-inner {
        position: relative;
        z-index: 2;
        padding: 18px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .visual-kicker {
        color: var(--accent);
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .visual-title {
        color: var(--ink);
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 6px;
    }
    .visual-copy {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.5;
        max-width: 360px;
    }
    .visual-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin-top: 14px;
    }
    .visual-card {
        border-radius: 18px;
        background: rgba(255,255,255,0.84);
        border: 1px solid rgba(15,23,40,0.08);
        box-shadow: 0 10px 24px rgba(15,23,40,0.05);
        padding: 12px 14px;
    }
    .visual-card-label {
        color: var(--muted);
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .visual-card-value {
        color: var(--ink);
        font-size: 18px;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 4px;
    }
    .visual-card-copy {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.45;
    }
    .visual-footer {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 12px;
    }
    .visual-badge {
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(79,140,255,0.10);
        border: 1px solid rgba(79,140,255,0.16);
        color: var(--primary);
        font-size: 12px;
        font-weight: 700;
    }
    .hero-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin-top: 10px;
    }
    .hero-card {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.14);
        backdrop-filter: blur(8px);
        border-radius: 18px;
        padding: 16px 18px;
    }
    .hero-card h4 {
        margin: 0 0 8px;
        color: var(--ink);
        font-size: 15px;
    }
    .hero-card p {
        margin: 0;
        color: var(--muted);
        font-size: 14px;
        line-height: 1.5;
    }
    .mini-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin: 10px 0 18px;
    }
    .mini-card {
        border: 1px solid var(--border);
        border-radius: 22px;
        background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(248,251,254,0.98) 100%);
        box-shadow: 0 20px 40px rgba(15, 23, 40, 0.06);
        padding: 18px 20px;
        position: relative;
        overflow: hidden;
    }
    .mini-card:after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, var(--primary) 0%, #5a8dff 60%, var(--accent) 100%);
    }
    .mini-kicker {
        font-size: 12px;
        font-weight: 700;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }
    .mini-title {
        font-size: 20px;
        font-weight: 800;
        color: var(--ink);
        margin-bottom: 8px;
    }
    .mini-copy {
        font-size: 14px;
        color: var(--muted);
        line-height: 1.5;
    }
    .workflow {
        border: 1px solid var(--border);
        border-radius: 22px;
        background: linear-gradient(135deg, #ffffff 0%, var(--panel) 100%);
        box-shadow: 0 20px 40px rgba(15, 23, 40, 0.06);
        padding: 18px 20px;
        margin: 12px 0 18px;
    }
    .workflow-title {
        font-size: 19px;
        font-weight: 800;
        color: var(--ink);
        margin-bottom: 12px;
    }
    .workflow-step {
        margin: 8px 0;
        color: #334155;
        font-size: 14px;
        padding: 12px 14px;
        border-radius: 14px;
        background: rgba(255,255,255,0.84);
        border: 1px solid rgba(15,39,68,0.08);
    }
    .section-card {
        border: 1px solid var(--border);
        border-radius: 22px;
        background: rgba(255,255,255,0.96);
        box-shadow: 0 20px 40px rgba(15, 23, 40, 0.06);
        padding: 18px 20px;
        height: 100%;
    }
    .section-card h3 {
        margin-top: 0;
        margin-bottom: 12px;
        font-size: 20px;
        color: var(--ink);
    }
    .section-card ul {
        margin: 0;
        padding-left: 1.1rem;
        color: var(--muted);
    }
    .section-card li { margin: 8px 0; }
    div[data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: 18px;
        background: rgba(255,255,255,0.96);
        box-shadow: 0 12px 24px rgba(22, 50, 79, 0.04);
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.92);
        border: 1px solid var(--border);
        border-radius: 22px;
        box-shadow: 0 18px 36px rgba(15, 23, 40, 0.06);
        padding: 14px 16px;
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
  <div class="hero-shell">
    <div class="hero-copy">
      <div class="hero-kicker">Retail Planning • AI-assisted</div>
      <div class="hero-title">ML-Assisted Demand & OTB Forecasting</div>
      <div class="hero-sub">A decision-ready retail planning workspace for forecasting demand, spotting stock risk, and generating buyer or leadership summaries.</div>
      <div class="hero-actions">
        <div class="hero-pill">Forecast demand</div>
        <div class="hero-pill">Prioritize reorders</div>
        <div class="hero-pill">Generate narratives</div>
      </div>
    </div>
    <div class="hero-visual">
      <svg viewBox="0 0 520 360" role="img" aria-label="Product overview background">
        <defs>
          <filter id="blur" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="18"/>
          </filter>
        </defs>
        <circle cx="420" cy="78" r="70" fill="#d9ebff" filter="url(#blur)" opacity="0.9"/>
        <circle cx="100" cy="270" r="82" fill="#ffe4bf" filter="url(#blur)" opacity="0.85"/>
        <circle cx="250" cy="170" r="120" fill="#eef5ff" opacity="0.82"/>
      </svg>
      <div class="hero-visual-inner">
        <div>
          <div class="visual-kicker">What You Get In One Run</div>
          <div class="visual-title">Planning-ready outputs</div>
          <div class="visual-copy">This app is not just a chart. It gives a complete planning package: forecast quantities, stock risk, reorder actions, and stakeholder-ready narrative summaries.</div>
        </div>
        <div class="visual-grid">
          <div class="visual-card">
            <div class="visual-card-label">Forecast</div>
            <div class="visual-card-value">SKU-level</div>
            <div class="visual-card-copy">Next-month and 3-month demand projections by item.</div>
          </div>
          <div class="visual-card">
            <div class="visual-card-label">Planning</div>
            <div class="visual-card-value">OTB</div>
            <div class="visual-card-copy">Recommended reorder actions based on projected stock cover.</div>
          </div>
          <div class="visual-card">
            <div class="visual-card-label">Risk</div>
            <div class="visual-card-value">Stock Health</div>
            <div class="visual-card-copy">Understock and overstock exposure surfaced at a glance.</div>
          </div>
          <div class="visual-card">
            <div class="visual-card-label">Narrative</div>
            <div class="visual-card-value">Ready</div>
            <div class="visual-card-copy">Executive, buyer, and risk summaries generated from the run.</div>
          </div>
        </div>
        <div class="visual-footer">
          <div class="visual-badge">Faster planning</div>
          <div class="visual-badge">Clearer actions</div>
          <div class="visual-badge">Better communication</div>
        </div>
      </div>
    </div>
  </div>
  <div class="hero-grid">
    <div class="hero-card">
      <h4>Who It Serves</h4>
      <p>Management, planners, and buyers who need faster monthly inventory decisions.</p>
    </div>
    <div class="hero-card">
      <h4>What It Solves</h4>
      <p>Combines forecast output, stock health, and reorder recommendations in one workflow.</p>
    </div>
    <div class="hero-card">
      <h4>Why It Matters</h4>
      <p>Turns raw files into decision-ready views, scenario planning, and narrative summaries.</p>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="mini-grid">
  <div class="mini-card">
    <div class="mini-kicker">Decision Support</div>
    <div class="mini-title">Forecast Demand</div>
    <div class="mini-copy">Estimate next-month and 3-month demand with ML plus fallback logic.</div>
  </div>
  <div class="mini-card">
    <div class="mini-kicker">Inventory Health</div>
    <div class="mini-title">Spot Stock Risk</div>
    <div class="mini-copy">Identify understock and overstock exposure before purchases are locked in.</div>
  </div>
  <div class="mini-card">
    <div class="mini-kicker">Communication</div>
    <div class="mini-title">Generate Narratives</div>
    <div class="mini-copy">Turn forecast outputs into executive, buyer, and risk-ready summaries.</div>
  </div>
</div>

<div class="workflow">
  <div class="workflow-title">Fastest Demo Path</div>
  <div class="workflow-step"><strong>1.</strong> Open <strong>Upload & Validation</strong> and load simulated 1-year data.</div>
  <div class="workflow-step"><strong>2.</strong> Review <strong>Executive Dashboard</strong> for top actions and stock exposure.</div>
  <div class="workflow-step"><strong>3.</strong> Use <strong>Narrative Co-Pilot</strong> to generate a pitch-ready summary.</div>
</div>
""",
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1.25, 1])
with col1:
    st.markdown(
        """
<div class="section-card">
  <h3>What This App Helps You Decide</h3>
  <ul>
    <li>Which SKUs need urgent reorder attention</li>
    <li>Where stock is too lean or too heavy</li>
    <li>How much demand to plan for over the next 1 to 3 months</li>
    <li>Which brands or items deserve manual review first</li>
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """
<div class="section-card">
  <h3>Version 1 Value</h3>
  <ul>
    <li>Faster planning cycles</li>
    <li>Clearer reorder prioritization</li>
    <li>More confidence in management discussions</li>
    <li>Better communication across planning teams</li>
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )

with st.expander("View Product Details", expanded=False):
    st.markdown(
        """
**Main workflow**

1. Upload inventory and sales files
2. Validate and normalize the data
3. Forecast demand and generate reorder recommendations
4. Review executive metrics, SKU detail, and planning scenarios
5. Generate buyer or leadership-ready narrative summaries

**Key capabilities**

- Hybrid ML plus fallback logic
- Time-aware validation
- Stock health assessment
- Explainability and model transparency

**Current limitations**

- Short history reduces seasonal learning
- Sparse items may rely on fallback rules
- Best used as decision support, not as a sole source of truth
        """
    )

# Sidebar
with st.sidebar:
    st.markdown("## 📚 Navigation")
    st.markdown(
        """
`1.` **Upload & Validation**  
`2.` **Executive Dashboard**  
`3.` **Forecast Explorer**  
`4.` **OTB Planner**  
`5.` **Model Insights**  
`6.` **Insights & Report Generator**
`7.` **Forecast Chat**
        """
    )
    
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

rag.render_sidebar_chat()

# Footer
import datetime as _dt
st.divider()
st.markdown(f"""
---
**Version**: 0.1.0 MVP
**Last Updated**: {_dt.date.today().year}
For support or feedback, contact your analytics team.
""")
