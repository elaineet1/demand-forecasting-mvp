"""
Streamlit page: Narrative Co-Pilot
Drafts stakeholder-ready summaries from forecast outputs.
"""

import streamlit as st

import src.copilot as copilot
from src import config, state


st.set_page_config(
    page_title="Narrative Co-Pilot",
    page_icon="📝",
    layout="wide",
)

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

st.title("📝 Narrative Co-Pilot")
st.markdown(
    "Draft executive, buyer, and risk summaries from the current forecast run. "
    "This page can use either the local grounded draft generator or OpenAI for richer wording."
)

forecast_results = st.session_state.get(config.STATE_FORECAST_RESULTS)
if forecast_results is None:
    st.warning("⚠️ No data loaded yet. Please go to **Upload & Validation** first.")
else:
    planner_output = forecast_results.get("planner_output")

    if planner_output is None or (hasattr(planner_output, "empty") and planner_output.empty):
        st.error("❌ No forecast output available")
    else:
        vendor_options = ["All"]
        if "vendor" in planner_output.columns:
            vendor_options += sorted(planner_output["vendor"].dropna().unique().tolist())

        health_options = ["All"]
        if "stock_health" in planner_output.columns:
            health_options += sorted(planner_output["stock_health"].dropna().unique().tolist())

        openai_config = copilot.get_openai_config()

        mode_label = "OpenAI connected" if openai_config["configured"] else "Local draft mode"
        mode_help = (
            f"Using model `{openai_config['model']}`."
            if openai_config["configured"]
            else "Set `OPENAI_API_KEY` in your environment or `.streamlit/secrets.toml` to enable OpenAI."
        )
        st.caption(f"{mode_label}. {mode_help}")

        with st.form("copilot_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                narrative_type = st.selectbox(
                    "Narrative Type",
                    options=["executive", "buyer", "risk"],
                    format_func=lambda value: {
                        "executive": "Executive Summary",
                        "buyer": "Buyer Brief",
                        "risk": "Risk Summary",
                    }[value],
                )

            with col2:
                vendor_filter = st.selectbox("Brand Scope", options=vendor_options)

            with col3:
                health_filter = st.selectbox("Health Scope", options=health_options)

            col4, col5 = st.columns([1, 2])
            with col4:
                use_openai = st.checkbox(
                    "Use OpenAI",
                    value=openai_config["configured"],
                    disabled=not openai_config["configured"],
                    help="Uses the OpenAI API when configured; otherwise the local grounded draft stays available.",
                )
            with col5:
                openai_model = st.text_input(
                    "OpenAI Model",
                    value=openai_config["model"] or "gpt-4.1-mini",
                    disabled=not openai_config["configured"],
                    help="Override the default model from OPENAI_MODEL if needed.",
                )

            focus_note = st.text_area(
                "Optional Focus Note",
                placeholder="Example: emphasize understock risk for board review",
                height=100,
            )

            submitted = st.form_submit_button("Generate Narrative")

        context = copilot.get_copilot_context(
            forecast_results,
            vendor_filter=vendor_filter,
            health_filter=health_filter,
        )

        if context.get("has_data"):
            metrics = context["metrics"]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Scoped SKUs", metrics["total_skus"], delta_color="off")
            with col2:
                st.metric("Reorder Qty", metrics["total_reorder_qty"], delta_color="off")
            with col3:
                st.metric("Understock SKUs", metrics["understock_count"], delta_color="off")
            with col4:
                st.metric("ML Coverage", metrics["ml_coverage"], delta_color="off")

        if submitted:
            try:
                if use_openai and openai_config["configured"]:
                    narrative = copilot.generate_narrative_with_openai(
                        forecast_results,
                        narrative_type=narrative_type,
                        vendor_filter=vendor_filter,
                        health_filter=health_filter,
                        focus_note=focus_note,
                        model=openai_model,
                    )
                    st.success("Generated with OpenAI.")
                else:
                    narrative = copilot.generate_narrative(
                        forecast_results,
                        narrative_type=narrative_type,
                        vendor_filter=vendor_filter,
                        health_filter=health_filter,
                        focus_note=focus_note,
                    )
                    st.info("Generated with the local grounded draft mode.")
            except Exception as exc:
                st.warning(f"OpenAI generation failed, falling back to local draft mode: {exc}")
                narrative = copilot.generate_narrative(
                    forecast_results,
                    narrative_type=narrative_type,
                    vendor_filter=vendor_filter,
                    health_filter=health_filter,
                    focus_note=focus_note,
                )

            st.subheader("Drafted Narrative")
            st.markdown(narrative)
            st.download_button(
                label="Download Narrative",
                data=narrative,
                file_name=f"narrative_{narrative_type}.md",
                mime="text/markdown",
            )

        with st.expander("How To Use This Page", expanded=False):
            st.markdown(
                """
Use this page when you want a management-friendly readout without manually stitching together charts and tables.

- **Executive Summary** is best for leadership updates.
- **Buyer Brief** is best for replenishment and trading conversations.
- **Risk Summary** is best for surfacing understock, overstock, and forecast-reliability concerns.
- To enable OpenAI, set `OPENAI_API_KEY` and optionally `OPENAI_MODEL` in your shell or `.streamlit/secrets.toml`.
                """
            )
