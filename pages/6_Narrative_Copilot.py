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
        --primary: #0f1728;
        --accent: #4f8cff;
        --muted: #667085;
        --card: #ffffff;
        --border: rgba(15, 23, 40, 0.08);
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(79, 140, 255, 0.12), transparent 24%),
            linear-gradient(180deg, #f9fbfe 0%, #f2f6fb 100%);
    }
    h1, h2, h3 { color: var(--primary); letter-spacing: -0.02em; }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.94);
        border: 1px solid var(--border);
        border-radius: 18px;
        box-shadow: 0 18px 36px rgba(15, 23, 40, 0.06);
        padding: 14px 16px;
    }
    div[data-testid="stMetric"] label { color: var(--muted); }
    .copilot-banner {
        border: 1px solid var(--border);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(255,255,255,0.96) 0%, rgba(237,244,255,0.98) 100%);
        box-shadow: 0 24px 48px rgba(15, 23, 40, 0.08);
        padding: 22px 24px;
        margin-bottom: 14px;
        color: #334155;
    }
    .copilot-kicker {
        color: var(--accent);
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("📝 Narrative Co-Pilot")
st.markdown('<div class="copilot-kicker">Narrative workspace • Executive, buyer, and risk modes</div>', unsafe_allow_html=True)
st.markdown(
    """
<div class="copilot-banner">
Draft executive, buyer, and risk summaries from the current forecast run. Start with a prompt suggestion,
then refine the focus note only if you need a more specific narrative.
</div>
""",
    unsafe_allow_html=True,
)

PROMPT_SUGGESTIONS = {
    "executive": [
        "Summarize this forecast run for senior management, focusing on top risks, reorder priorities, and forecast confidence.",
        "Write a board-style update highlighting the most important inventory actions this month.",
        "Explain the overall demand outlook and stock exposure in plain business language.",
    ],
    "buyer": [
        "Write a buyer brief focused on urgent reorder actions and items needing manual review.",
        "Summarize the top replenishment priorities for this run and the brands needing fastest action.",
        "Focus on understock-risk SKUs that could affect near-term sales.",
    ],
    "risk": [
        "Write a risk summary covering understock, overstock, fallback usage, and data quality caveats.",
        "Explain the main inventory risks if no action is taken this month.",
        "Focus on where forecast reliability is limited and why decision-makers should care.",
    ],
}

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

        if "copilot_focus_note" not in st.session_state:
            st.session_state["copilot_focus_note"] = ""

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

            selected_suggestion = st.selectbox(
                "Prompt Suggestion",
                options=["Custom"] + PROMPT_SUGGESTIONS[narrative_type],
                help="Choose a starter prompt, or leave it on Custom and write your own focus note.",
            )

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

            default_focus_note = (
                selected_suggestion if selected_suggestion != "Custom" else st.session_state["copilot_focus_note"]
            )
            focus_note = st.text_area(
                "Optional Focus Note",
                placeholder="Example: emphasize understock risk for board review",
                value=default_focus_note,
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
            use_tab1, use_tab2, use_tab3 = st.tabs(["Executive", "Buyer", "Risk"])
            with use_tab1:
                st.caption("Best for leadership updates, pitch meetings, and top-line planning reviews.")
            with use_tab2:
                st.caption("Best for replenishment planning, vendor reviews, and manual action lists.")
            with use_tab3:
                st.caption("Best for stock imbalance, fallback reliance, and forecast-confidence discussions.")

        if submitted:
            st.session_state["copilot_focus_note"] = focus_note
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
            download_col, help_col = st.columns([1, 1.2])
            with download_col:
                st.download_button(
                    label="Download Narrative",
                    data=narrative,
                    file_name=f"narrative_{narrative_type}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with help_col:
                st.caption(
                    "Tip: use the downloaded markdown as a draft for slides, emails, or meeting notes."
                )

        with st.expander("How To Use This Page", expanded=False):
            st.markdown(
                """
Use this page when you want a management-friendly readout without manually stitching together charts and tables.

- **Executive Summary** is best for leadership updates.
- **Buyer Brief** is best for replenishment and trading conversations.
- **Risk Summary** is best for surfacing understock, overstock, and forecast-reliability concerns.
- Start with a built-in prompt suggestion, then refine the focus note if you want a narrower narrative.
- To enable OpenAI, set `OPENAI_API_KEY` and optionally `OPENAI_MODEL` in your shell or `.streamlit/secrets.toml`.
                """
            )
