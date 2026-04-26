"""
Streamlit page: Forecast Chat
Interactive Q&A interface for exploring forecast data using RAG (Retrieval-Augmented Generation).
"""

import streamlit as st

from src import config, rag, state

st.set_page_config(
    page_title="Forecast Chat",
    page_icon="💬",
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
    .chat-banner {
        border: 1px solid var(--border);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(255,255,255,0.96) 0%, rgba(237,244,255,0.98) 100%);
        box-shadow: 0 24px 48px rgba(15, 23, 40, 0.08);
        padding: 22px 24px;
        margin-bottom: 14px;
        color: #334155;
    }
    .chat-kicker {
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

st.title("💬 Forecast Chat")
st.markdown('<div class="chat-kicker">Interactive Q&A • Ask questions about your forecast</div>', unsafe_allow_html=True)
st.markdown(
    """
<div class="chat-banner">
Ask questions about forecast data, stock health, reorder recommendations, and model insights.
The assistant retrieves relevant forecast context to ground all answers in your data.
</div>
""",
    unsafe_allow_html=True,
)

# Initialize chat history in session state
if config.STATE_RAG_CHAT_HISTORY not in st.session_state:
    st.session_state[config.STATE_RAG_CHAT_HISTORY] = []

# Get forecast results
forecast_results = st.session_state.get(config.STATE_FORECAST_RESULTS)

if forecast_results is None:
    st.warning("⚠️ No data loaded yet. Please go to **Upload & Validation** first.")
else:
    planner_output = forecast_results.get("planner_output")

    if planner_output is None or (hasattr(planner_output, "empty") and planner_output.empty):
        st.error("❌ No forecast output available. Please generate a forecast first.")
    else:
        # Display chat history
        st.subheader("Chat History")
        chat_container = st.container()

        with chat_container:
            for message in st.session_state[config.STATE_RAG_CHAT_HISTORY]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Chat input
        st.subheader("Ask a Question")
        user_input = st.chat_input("What would you like to know about the forecast?")

        if user_input:
            # Add user message to history
            st.session_state[config.STATE_RAG_CHAT_HISTORY].append({
                "role": "user",
                "content": user_input,
            })

            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)

            # Generate response
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Retrieving forecast context..."):
                        response = rag.chat_completion(
                            query=user_input,
                            forecast_results=forecast_results,
                            chat_history=st.session_state[config.STATE_RAG_CHAT_HISTORY][:-1],  # Exclude current user message
                        )

                    st.markdown(response)

                    # Add assistant message to history
                    st.session_state[config.STATE_RAG_CHAT_HISTORY].append({
                        "role": "assistant",
                        "content": response,
                    })

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    # Remove the user message if assistant fails
                    st.session_state[config.STATE_RAG_CHAT_HISTORY].pop()

        # Clear chat button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Clear Chat History", use_container_width=True):
                st.session_state[config.STATE_RAG_CHAT_HISTORY] = []
                st.rerun()

        # Help section
        with st.expander("How to Use This Page", expanded=False):
            st.markdown("""
**Ask about:**
- **Specific SKUs**: "What's the forecast for item XYZ?"
- **Stock Health**: "Which items are understocked?"
- **Reorder Actions**: "What are the top reorder items?"
- **Forecast Methods**: "Why was the fallback method used for item XYZ?"
- **Model Insights**: "What's the forecast accuracy?"
- **Categories/Vendors**: "Show me all overstocked items from Vendor ABC"

**Tips:**
- Be specific: mention SKU numbers, vendors, or categories when relevant
- The assistant grounds all answers in your forecast data
- Chat history persists while you navigate other pages
- Refresh the page to clear all data and start fresh
""")
