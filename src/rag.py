"""
RAG (Retrieval-Augmented Generation) module for the Forecast Chat.
Handles document indexing, retrieval, and chat completions grounded in forecast data.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src import copilot


class Document:
    """Simple document wrapper for RAG retrieval."""

    def __init__(self, content: str, metadata: Dict = None):
        self.content = content
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Document(content={self.content[:80]}..., metadata={self.metadata})"


def build_documents_from_forecast(forecast_results: Dict) -> List[Document]:
    """
    Convert forecast data into retrievable documents.

    Creates documents from:
    1. Planner output (one document per SKU)
    2. Planning summary and model insights
    3. Feature importance
    """
    documents = []

    planner_output = forecast_results.get("planner_output")
    planning_summary = forecast_results.get("planning_summary", {})
    model_results = forecast_results.get("model_results", {})

    if planner_output is not None and not planner_output.empty:
        # Create one document per SKU with all relevant fields
        for _, row in planner_output.iterrows():
            item_no = row.get("item_no", "Unknown")
            desc = row.get("item_description", "N/A")
            vendor = row.get("vendor", "N/A")
            category = row.get("category", "N/A")

            current_stock = row.get("total_stock", 0)
            forecast_m1 = row.get("forecast_m1", 0)
            forecast_m2 = row.get("forecast_m2", 0)
            forecast_m3 = row.get("forecast_m3", 0)
            projected_demand = row.get("projected_3m_demand", 0)

            reorder_qty = row.get("reorder_qty", 0)
            stock_health = row.get("stock_health", "unknown")
            forecast_method = row.get("forecast_method", "unknown")
            remark = row.get("remark", "")
            stock_cover_months = row.get("stock_cover_months", None)

            avg_monthly_demand = projected_demand / 3 if projected_demand else 0
            if stock_cover_months is None or pd.isna(stock_cover_months):
                cover_str = (
                    f"{current_stock / avg_monthly_demand:.1f}" if avg_monthly_demand > 0 else "N/A"
                )
            else:
                cover_str = f"{stock_cover_months:.1f}"

            # Build SKU document
            sku_doc = f"""SKU: {item_no}
Product: {desc}
Vendor: {vendor}
Category: {category}
Current Stock: {current_stock:.1f}
Average Monthly Demand (forecast): {avg_monthly_demand:.1f}
Stock Cover Months: {cover_str}
Forecast M1 (next month): {forecast_m1:.1f}
Forecast M2: {forecast_m2:.1f}
Forecast M3: {forecast_m3:.1f}
Projected 3-Month Demand: {projected_demand:.1f}
Recommended Reorder Quantity: {reorder_qty:.1f}
Stock Health Status: {stock_health}
Forecast Method Used: {forecast_method}
Notes: {remark}"""

            documents.append(
                Document(
                    content=sku_doc,
                    metadata={
                        "type": "sku",
                        "item_no": item_no,
                        "vendor": vendor,
                        "category": category,
                        "stock_health": stock_health,
                    },
                )
            )

    # Add planning summary document
    if planning_summary:
        summary_text = f"""Forecast Run Summary:
Total Active SKUs: {planning_summary.get('total_active_skus', 0)}
Total Current Stock: {planning_summary.get('total_current_stock', 0):.1f}
Total Projected 3-Month Demand: {planning_summary.get('total_projected_3m_demand', 0):.1f}
Total Recommended Reorder Qty: {planning_summary.get('total_reorder_qty', 0):.1f}
Average Monthly Demand: {planning_summary.get('avg_projected_monthly_demand', 0):.1f}
ML Model Coverage: {planning_summary.get('ml_coverage', 'N/A')}
Fallback Forecast Coverage: {planning_summary.get('fallback_coverage', 'N/A')}"""

        documents.append(
            Document(
                content=summary_text,
                metadata={"type": "summary"},
            )
        )

    # Add model insights document
    if model_results:
        validation = model_results.get("validation_metrics", {})
        model_doc = f"""Model Performance and Insights:
Model Type: {model_results.get('model_name', 'Unknown')}
Validation WAPE: {validation.get('wape', 0):.1f}%
Mean Absolute Error: {validation.get('mae', 0):.2f}
Root Mean Squared Error: {validation.get('rmse', 0):.2f}
Test Samples: {validation.get('n_test_samples', 0)}
Test Period: {validation.get('test_date_start', 'N/A')} to {validation.get('test_date_end', 'N/A')}
Unique Test Months: {validation.get('test_unique_months', 0)}

Interpretation: WAPE (Weighted Absolute Percentage Error) around 30-50% is typical for retail forecasting with sparse data.
The model was trained on 80% of historical data and tested on the remaining 20% in chronological order."""

        documents.append(
            Document(
                content=model_doc,
                metadata={"type": "model_insights"},
            )
        )

    # Add forecast method explanation
    forecast_methods_doc = """Forecast Methods Hierarchy:
1. ML Model: LightGBM regression trained on historical sales with time-series validation.
   - Used when: SKU has 2+ months of sales history.
   - Advantages: Data-driven, captures trends and seasonality.

2. Fallback - Recent Average: 3-month rolling average of recent sales.
   - Used when: ML model insufficient or as guardrail for low-volume SKUs.
   - Advantages: Stable, not overly sensitive to single outliers.

3. Fallback - Category + Vendor Average: Average sales for similar products from the same brand.
   - Used when: Not enough recent data for individual SKU.
   - Advantages: Leverages market trends by brand/category.

4. Fallback - Category Average: Overall category benchmark.
   - Used when: Vendor-specific data unavailable.

5. Fallback - Existing Forecast: Pre-existing forecast from inventory file.
   - Used when: Uploaded inventory already contains forecast_qty column.

6. Fallback - Zero: No sales history available (new SKUs).
   - Result: Forecast set to zero; requires manual input."""

    documents.append(
        Document(
            content=forecast_methods_doc,
            metadata={"type": "methodology"},
        )
    )

    return documents


def get_or_create_embeddings(
    documents: List[Document],
    api_key: Optional[str] = None,
) -> Tuple[np.ndarray, List[Document]]:
    """
    Create embeddings for documents using OpenAI's embedding API.

    Returns:
        Tuple of (embeddings_array, documents_list)
    """
    if not documents:
        return np.array([]), []

    config = copilot.get_openai_config()
    api_key = api_key or config["api_key"]

    if not api_key:
        raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY in environment or .streamlit/secrets.toml")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("OpenAI package required. Install with: pip install openai") from exc

    client = OpenAI(api_key=api_key)

    # Batch embed all documents
    texts = [doc.content for doc in documents]

    try:
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small",
        )
        embeddings = np.array([item.embedding for item in response.data])
        return embeddings, documents
    except Exception as e:
        raise RuntimeError(f"Failed to create embeddings: {e}") from e


def retrieve_relevant_docs(
    query: str,
    embeddings: np.ndarray,
    documents: List[Document],
    k: int = 5,
    api_key: Optional[str] = None,
) -> List[Document]:
    """
    Retrieve top-k documents most similar to the query using cosine similarity.
    """
    if not documents or len(embeddings) == 0:
        return []

    config = copilot.get_openai_config()
    api_key = api_key or config["api_key"]

    if not api_key:
        raise ValueError("OpenAI API key not configured.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("OpenAI package required.") from exc

    client = OpenAI(api_key=api_key)

    # Embed the query
    response = client.embeddings.create(
        input=[query],
        model="text-embedding-3-small",
    )
    query_embedding = np.array(response.data[0].embedding)

    # Cosine similarity
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-10
    )

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:k]

    return [documents[i] for i in top_indices]


def build_rag_prompt(
    query: str,
    retrieved_docs: List[Document],
    planning_summary: Dict,
) -> str:
    """
    Build a system prompt with retrieved context for RAG chat.
    """
    context_blocks = [
        "You are a helpful demand forecasting and inventory planning assistant.",
        "Answer user questions based ONLY on the provided forecast data.",
        "Be concise, cite specific numbers and SKUs, and ground all claims in the data.",
        "If the data cannot answer the question, say so clearly.",
        "",
        "--- FORECAST DATA CONTEXT ---",
    ]

    for i, doc in enumerate(retrieved_docs, 1):
        context_blocks.append(f"\n[Document {i}]")
        context_blocks.append(doc.content)

    if planning_summary:
        context_blocks.append("\n--- FORECAST RUN SUMMARY ---")
        for key, value in planning_summary.items():
            context_blocks.append(f"{key}: {value}")

    context_blocks.extend([
        "",
        "--- USER QUESTION ---",
        query,
    ])

    return "\n".join(context_blocks)


def chat_completion(
    query: str,
    forecast_results: Dict,
    chat_history: List[Dict] = None,
    api_key: Optional[str] = None,
    _extra_docs: Optional[List[Dict]] = None,
) -> str:
    """
    End-to-end RAG chat pipeline.

    Args:
        query: User's question
        forecast_results: Forecast output from the pipeline
        chat_history: Previous messages for context (not used in current version but reserved for future)
        api_key: OpenAI API key (uses config if not provided)

    Returns:
        Assistant's response
    """
    if chat_history is None:
        chat_history = []

    # Get documents and embeddings (cached per forecast run)
    documents = build_documents_from_forecast(forecast_results)
    if not documents:
        return "No forecast data available. Please upload files and generate a forecast first."

    cfg_obj = copilot.get_openai_config()
    api_key = api_key or cfg_obj["api_key"]
    model = cfg_obj["model"] or "gpt-4.1-mini"

    embeddings, docs = _get_cached_embeddings(forecast_results, documents, api_key)

    # Retrieve relevant forecast documents
    retrieved = retrieve_relevant_docs(query, embeddings, docs, k=5, api_key=api_key)

    # Merge in any user-document chunks
    if _extra_docs:
        for d in _extra_docs:
            retrieved.append(Document(
                content=f"[{d['source']}] {d['text']}",
                metadata={"type": "user_doc", "source": d["source"]},
            ))

    # Build RAG prompt
    planning_summary = forecast_results.get("planning_summary", {})
    rag_prompt = build_rag_prompt(query, retrieved, planning_summary)

    if not api_key:
        raise ValueError("OpenAI API key not configured.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("OpenAI package required.") from exc

    client = OpenAI(api_key=api_key)

    # Build messages with history
    messages = []

    # Add chat history (simplified - just last few messages)
    if chat_history:
        for msg in chat_history[-6:]:  # Keep last 6 messages (3 exchanges)
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

    # Add current query
    messages.append({
        "role": "user",
        "content": rag_prompt,
    })

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Chat completion failed: {e}") from e


# ==============================================================================
# USER DOCUMENT STORE
# ==============================================================================

USER_DOC_STORE_KEY = "rag_user_docs"
_RAG_EMB_CACHE_KEY = "rag_emb_cache"


def _chunk_text(text: str, size: int = 400, overlap: int = 60) -> List[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + size]))
        start += size - overlap
    return [c for c in chunks if c.strip()]


def _extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(uploaded_file)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            return ""
    return ""


def index_user_document(name: str, text: str, api_key: str) -> int:
    """Chunk, embed, and store a user-uploaded document in session state."""
    try:
        import streamlit as st
    except ImportError:
        return 0

    if USER_DOC_STORE_KEY not in st.session_state:
        st.session_state[USER_DOC_STORE_KEY] = []

    chunks = _chunk_text(text)
    if not chunks:
        return 0

    try:
        from openai import OpenAI
    except ImportError:
        return 0

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(input=chunks, model="text-embedding-3-small")
    embeddings = np.array([item.embedding for item in response.data])

    for chunk, emb in zip(chunks, embeddings):
        st.session_state[USER_DOC_STORE_KEY].append(
            {"source": name, "text": chunk, "embedding": emb.tolist()}
        )
    return len(chunks)


def _retrieve_user_docs(query: str, api_key: str, k: int = 3) -> List[Dict]:
    """Retrieve top-k user document chunks by cosine similarity."""
    try:
        import streamlit as st
    except ImportError:
        return []

    store = st.session_state.get(USER_DOC_STORE_KEY, [])
    if not store:
        return []

    try:
        from openai import OpenAI
    except ImportError:
        return []

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(input=[query], model="text-embedding-3-small")
    q_emb = np.array(response.data[0].embedding)

    scored = []
    for doc in store:
        emb = np.array(doc["embedding"])
        denom = np.linalg.norm(q_emb) * np.linalg.norm(emb)
        sim = float(np.dot(q_emb, emb) / denom) if denom > 0 else 0.0
        scored.append((sim, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:k]]


def _get_cached_embeddings(
    forecast_results: Dict, documents: List[Document], api_key: str
) -> Tuple[np.ndarray, List[Document]]:
    """Return cached forecast embeddings, recomputing only when the forecast changes."""
    try:
        import streamlit as st
        planner_output = forecast_results.get("planner_output")
        cache_key = len(planner_output) if planner_output is not None else 0
        cached = st.session_state.get(_RAG_EMB_CACHE_KEY)
        if cached and cached.get("key") == cache_key:
            return cached["embeddings"], cached["documents"]
        embeddings, docs = get_or_create_embeddings(documents, api_key=api_key)
        st.session_state[_RAG_EMB_CACHE_KEY] = {
            "key": cache_key,
            "embeddings": embeddings,
            "documents": docs,
        }
        return embeddings, docs
    except Exception:
        return get_or_create_embeddings(documents, api_key=api_key)


# ==============================================================================
# SIDEBAR CHAT COMPONENT
# ==============================================================================


def render_sidebar_chat() -> None:
    """Render a persistent RAG chatbot in the sidebar. Call at the end of each page."""
    try:
        import streamlit as st
    except ImportError:
        return

    from src import config

    cfg = copilot.get_openai_config()

    if config.STATE_RAG_CHAT_HISTORY not in st.session_state:
        st.session_state[config.STATE_RAG_CHAT_HISTORY] = []

    history = st.session_state[config.STATE_RAG_CHAT_HISTORY]
    forecast_results = st.session_state.get(config.STATE_FORECAST_RESULTS)

    with st.sidebar:
        st.divider()
        with st.expander("💬 Ask Your Data", expanded=False):
            if not cfg["configured"]:
                st.caption("Add `OPENAI_API_KEY` to Streamlit secrets to enable.")
                return

            # Document upload
            uploaded = st.file_uploader(
                "Upload PDF or TXT for extra context",
                type=["pdf", "txt"],
                key="rag_sidebar_uploader",
                label_visibility="collapsed",
            )
            if uploaded:
                store = st.session_state.get(USER_DOC_STORE_KEY, [])
                already = any(d["source"] == uploaded.name for d in store)
                if already:
                    st.caption(f"✓ '{uploaded.name}' already indexed.")
                elif st.button("Index document", key="rag_index_btn"):
                    with st.spinner("Indexing..."):
                        text = _extract_text(uploaded)
                        if text.strip():
                            n = index_user_document(uploaded.name, text, cfg["api_key"])
                            st.success(f"Indexed {n} chunks from '{uploaded.name}'.")
                        else:
                            st.warning("Could not extract text from this file.")

            user_store = st.session_state.get(USER_DOC_STORE_KEY, [])
            if user_store:
                sources = sorted({d["source"] for d in user_store})
                st.caption(f"📎 Docs: {', '.join(sources)}")
                if st.button("Clear docs", key="rag_clear_docs_btn"):
                    st.session_state[USER_DOC_STORE_KEY] = []
                    st.rerun()

            st.divider()

            # Chat history (last 4 messages)
            for msg in history[-4:]:
                prefix = "**You:** " if msg["role"] == "user" else "**AI:** "
                content = msg["content"]
                st.markdown(prefix + (content[:160] + "…" if len(content) > 160 else content))

            if history:
                if st.button("Clear chat", key="rag_clear_chat_btn"):
                    st.session_state[config.STATE_RAG_CHAT_HISTORY] = []
                    st.rerun()

            # Input
            with st.form("rag_sidebar_form", clear_on_submit=True):
                q = st.text_input(
                    "Ask",
                    placeholder="e.g. Which SKUs need urgent reorder?",
                    label_visibility="collapsed",
                )
                send = st.form_submit_button("Ask ▶", use_container_width=True)

            if send and q.strip():
                if forecast_results is None:
                    st.warning("Load data first on the Upload page.")
                else:
                    with st.spinner("Thinking..."):
                        try:
                            # Retrieve user docs for augmentation
                            user_docs = _retrieve_user_docs(q.strip(), cfg["api_key"])
                            reply = chat_completion(
                                query=q.strip(),
                                forecast_results=forecast_results,
                                chat_history=history,
                                api_key=cfg["api_key"],
                                _extra_docs=user_docs,
                            )
                            history.append({"role": "user", "content": q.strip()})
                            history.append({"role": "assistant", "content": reply})
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Error: {exc}")
