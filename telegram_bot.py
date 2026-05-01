# telegram_bot.py
# Supervisor-ready Telegram bot with direct RAG forecast chat.
#
# Structured commands:
#   /start      — welcome + command list
#   /dashboard  — clickable app menu
#   /summary    — portfolio snapshot
#   /reorders   — top 5 SKUs needing reorder
#   /health     — stock health breakdown
#   /sku ITEM   — detail for a specific SKU
#   /clearchat  — reset your conversation history
#
# Free-text messages → RAG chat grounded in the latest forecast data.

import os
import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters,
)

from src import persistence
from src.rag import build_documents_from_forecast, get_or_create_embeddings, retrieve_relevant_docs, build_rag_prompt

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
STREAMLIT_APP_URL = os.getenv("STREAMLIT_APP_URL", "https://demand-forecasting-mvp.streamlit.app")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# ==============================================================================
# In-process embedding cache (shared across all users, invalidated by mtime)
# ==============================================================================

_emb_cache: dict = {"mtime": None, "embeddings": None, "documents": None}


def _get_embeddings(forecast_results: dict) -> tuple:
    """Return cached embeddings, recomputing when planner data changes."""
    mtime = persistence.planner_mtime()
    if _emb_cache["mtime"] == mtime and _emb_cache["embeddings"] is not None:
        return _emb_cache["embeddings"], _emb_cache["documents"]

    documents = build_documents_from_forecast(forecast_results)
    embeddings, docs = get_or_create_embeddings(documents, api_key=OPENAI_API_KEY)
    _emb_cache["mtime"] = mtime
    _emb_cache["embeddings"] = embeddings
    _emb_cache["documents"] = docs
    return embeddings, docs


# ==============================================================================
# Helpers
# ==============================================================================

def _fmt_int(val) -> str:
    try:
        return f"{int(round(float(val))):,}"
    except Exception:
        return "—"


def _no_data_msg() -> str:
    return (
        "No forecast data found.\n"
        "Please run a forecast in the Streamlit app first, then come back here."
    )


def _load_run() -> dict | None:
    return persistence.load_run()


# ==============================================================================
# Dashboard menu
# ==============================================================================

def _dashboard_text() -> str:
    meta = persistence.get_metadata()
    last_run = meta.get("last_run", "never")
    if last_run != "never":
        last_run = last_run[:16].replace("T", " ")
    skus = meta.get("total_skus", "—")
    reorder = meta.get("total_reorder_qty", "—")
    try:
        reorder = f"{int(round(float(reorder))):,}"
    except Exception:
        pass
    return (
        f"📊 *Demand & OTB Forecasting*\n"
        f"Last run: {last_run}\n"
        f"Active SKUs: {skus} | Reorder: {reorder} units\n\n"
        "Choose an action or type any question to chat with your forecast data."
    )


def _dashboard_buttons() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📤 Upload & Run Forecast", url=STREAMLIT_APP_URL)],
        [
            InlineKeyboardButton("📊 Executive Dashboard", url=STREAMLIT_APP_URL),
            InlineKeyboardButton("🛒 OTB Planner", url=STREAMLIT_APP_URL),
        ],
        [
            InlineKeyboardButton("🔍 Forecast Explorer", url=STREAMLIT_APP_URL),
            InlineKeyboardButton("📈 Charts & Reports", url=STREAMLIT_APP_URL),
        ],
        [
            InlineKeyboardButton("📋 File Format Help", callback_data="file_help"),
            InlineKeyboardButton("ℹ️ How To Use", callback_data="how_to_use"),
        ],
    ])


def _back_button() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("⬅️ Back", callback_data="back")]]
    )


# ==============================================================================
# Structured command handlers
# ==============================================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hello! I'm your Forecasting Dashboard Bot.\n\n"
        "*Structured commands:*\n"
        "/dashboard — app menu\n"
        "/summary   — portfolio snapshot\n"
        "/reorders  — top 5 reorder SKUs\n"
        "/health    — stock health breakdown\n"
        "/sku ITEM  — detail for one SKU\n"
        "/clearchat — reset chat history\n\n"
        "💬 Or just *type any question* to chat with your forecast data — "
        "e.g. _Which brand has the most overstock?_ or _What is the reorder for item ABC123?_",
        parse_mode="Markdown"
    )


async def dashboard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        _dashboard_text(),
        reply_markup=_dashboard_buttons(),
        parse_mode="Markdown"
    )


async def summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    run = _load_run()
    if run is None:
        await update.message.reply_text(_no_data_msg())
        return

    s = run.get("planning_summary", {})
    meta = run.get("metadata", {})
    last_run = meta.get("last_run", "unknown")[:16].replace("T", " ")
    health = s.get("avg_stock_health", {})

    method_breakdown = s.get("forecast_method_breakdown", {})
    ml_count = method_breakdown.get("ml_model", 0)
    total_skus = s.get("total_active_skus", 1) or 1
    ml_pct = f"{ml_count / total_skus * 100:.0f}%"

    text = (
        f"📊 *Forecast Summary*\n"
        f"Last run: {last_run}\n\n"
        f"Active SKUs: *{_fmt_int(s.get('total_active_skus', 0))}*\n"
        f"Current stock: {_fmt_int(s.get('total_current_stock', 0))} units\n"
        f"3-month demand: {_fmt_int(s.get('total_projected_3m_demand', 0))} units\n"
        f"Reorder needed: *{_fmt_int(s.get('total_reorder_qty', 0))} units*\n\n"
        f"*Stock health*\n"
        f"🔴 Understock: {health.get('understock_risk', 0)} SKUs\n"
        f"🟢 Healthy:    {health.get('healthy_stock', 0)} SKUs\n"
        f"🟡 Overstock:  {health.get('overstock_risk', 0)} SKUs\n\n"
        f"*Forecast coverage*\n"
        f"ML model: {ml_pct} of SKUs\n\n"
        f"_Type a question to dig deeper._"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def reorders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    run = _load_run()
    if run is None:
        await update.message.reply_text(_no_data_msg())
        return

    df: pd.DataFrame = run["planner_output"]
    top = df[df["reorder_qty"] > 0].nlargest(5, "reorder_qty")
    if top.empty:
        await update.message.reply_text("✅ No reorders needed in the current forecast.")
        return

    lines = ["🛒 *Top 5 Reorder Items*\n"]
    for _, row in top.iterrows():
        health_emoji = {"understock_risk": "🔴", "healthy_stock": "🟢", "overstock_risk": "🟡"}.get(
            row.get("stock_health", ""), "⚪"
        )
        lines.append(
            f"{health_emoji} `{row.get('item_no', '?')}` "
            f"{str(row.get('item_description', ''))[:35]}\n"
            f"  Vendor: {row.get('vendor', '—')} | "
            f"Stock: {_fmt_int(row.get('total_stock', 0))} | "
            f"3m demand: {_fmt_int(row.get('projected_3m_demand', 0))} | "
            f"*Reorder: {_fmt_int(row.get('reorder_qty', 0))}*"
        )

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    run = _load_run()
    if run is None:
        await update.message.reply_text(_no_data_msg())
        return

    df: pd.DataFrame = run["planner_output"]
    counts = df["stock_health"].value_counts().to_dict() if "stock_health" in df.columns else {}
    under_df = df[df["stock_health"] == "understock_risk"]
    over_df = df[df["stock_health"] == "overstock_risk"]

    lines = [
        "📦 *Stock Health Breakdown*\n",
        f"🔴 Understock risk: *{counts.get('understock_risk', 0)}* SKUs",
        f"🟢 Healthy:         *{counts.get('healthy_stock', 0)}* SKUs",
        f"🟡 Overstock risk:  *{counts.get('overstock_risk', 0)}* SKUs",
    ]

    if not under_df.empty:
        top_under = under_df.nlargest(3, "reorder_qty")
        lines.append("\n*Most urgent understocked:*")
        for _, row in top_under.iterrows():
            lines.append(
                f"• `{row.get('item_no', '?')}` "
                f"{str(row.get('item_description', ''))[:32]} "
                f"— reorder *{_fmt_int(row.get('reorder_qty', 0))}*"
            )

    if not over_df.empty:
        top_over = over_df.nlargest(3, "overstock_qty") if "overstock_qty" in over_df.columns else over_df.head(3)
        lines.append("\n*Most overstocked:*")
        for _, row in top_over.iterrows():
            excess = _fmt_int(row.get("overstock_qty", 0)) if "overstock_qty" in row else "—"
            lines.append(
                f"• `{row.get('item_no', '?')}` "
                f"{str(row.get('item_description', ''))[:32]} "
                f"— excess *{excess}*"
            )

    lines.append("\n_Type a question to explore further._")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def sku_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /sku ITEM\\_NO\nExample: /sku ABC123", parse_mode="Markdown")
        return

    item_no = " ".join(context.args).strip().upper()
    run = _load_run()
    if run is None:
        await update.message.reply_text(_no_data_msg())
        return

    df: pd.DataFrame = run["planner_output"]
    match = df[df["item_no"].astype(str).str.upper() == item_no]
    if match.empty:
        await update.message.reply_text(
            f"SKU `{item_no}` not found in the last forecast.\n"
            "Check the item number and try again.",
            parse_mode="Markdown"
        )
        return

    row = match.iloc[0]
    health_emoji = {"understock_risk": "🔴", "healthy_stock": "🟢", "overstock_risk": "🟡"}.get(
        row.get("stock_health", ""), "⚪"
    )
    cover = row.get("stock_cover_months")
    cover_str = f"{cover:.1f} months" if pd.notna(cover) else "—"

    text = (
        f"🔍 *{item_no}*\n"
        f"{str(row.get('item_description', ''))}\n"
        f"Vendor: {row.get('vendor', '—')} | Category: {row.get('category', '—')}\n\n"
        f"*Inventory*\n"
        f"Total stock: {_fmt_int(row.get('total_stock', 0))} units\n"
        f"Stock cover: {cover_str}\n"
        f"Health: {health_emoji} {row.get('stock_health', '—')}\n\n"
        f"*Forecast*\n"
        f"M1: {_fmt_int(row.get('forecast_m1', 0))} | "
        f"M2: {_fmt_int(row.get('forecast_m2', 0))} | "
        f"M3: {_fmt_int(row.get('forecast_m3', 0))}\n"
        f"3-month demand: {_fmt_int(row.get('projected_3m_demand', 0))}\n"
        f"Method: `{row.get('forecast_method', '—')}`\n\n"
        f"*Recommended action*\n"
        f"Reorder: *{_fmt_int(row.get('reorder_qty', 0))} units*\n"
        f"{str(row.get('remark', ''))}"
    )
    await update.message.reply_text(text[:4096], parse_mode="Markdown")


async def clearchat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["history"] = []
    await update.message.reply_text("✅ Chat history cleared.")


# ==============================================================================
# RAG chat — triggered by any plain text message
# ==============================================================================

async def chat_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.strip()
    if not query:
        return

    if not OPENAI_API_KEY:
        await update.message.reply_text(
            "OpenAI API key not configured. Add OPENAI_API_KEY to your .env file to enable chat."
        )
        return

    run = _load_run()
    if run is None:
        await update.message.reply_text(_no_data_msg())
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        from openai import OpenAI
    except ImportError:
        await update.message.reply_text("openai package not installed. Run: pip install openai")
        return

    try:
        embeddings, docs = _get_embeddings(run)
        retrieved = retrieve_relevant_docs(query, embeddings, docs, k=6, api_key=OPENAI_API_KEY)
        planning_summary = run.get("planning_summary", {})
        system_prompt = build_rag_prompt(query, retrieved, planning_summary)

        history: list = context.user_data.get("history", [])

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history[-8:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": query})

        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=800,
        )
        reply = response.choices[0].message.content.strip()

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": reply})
        context.user_data["history"] = history[-20:]

        # Telegram Markdown can choke on certain chars — send safely
        await update.message.reply_text(reply[:4096])

    except Exception as e:
        logger.exception("RAG chat error")
        await update.message.reply_text(f"Sorry, something went wrong: {e}")


# ==============================================================================
# Inline button callbacks
# ==============================================================================

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "file_help":
        await query.edit_message_text(
            "📋 *File Format Help*\n\n"
            "*Inventory file* (Excel or CSV)\n"
            "Columns: Item No, Description, Total Stock, Item Status, Active\n\n"
            "*Sales file* (Excel or CSV)\n"
            "Columns: Item No, Description, Quantity\n\n"
            "*Optional:* Event calendar, Brand mapping",
            reply_markup=_back_button(),
            parse_mode="Markdown"
        )
    elif query.data == "how_to_use":
        await query.edit_message_text(
            "ℹ️ *How To Use*\n\n"
            "1. Tap Upload & Run Forecast in the app\n"
            "2. Upload inventory and sales files\n"
            "3. Run the forecast\n"
            "4. Come back here — no need to open the app again:\n\n"
            "   📊 /summary — portfolio snapshot\n"
            "   🛒 /reorders — what to buy\n"
            "   📦 /health — stock health\n"
            "   🔍 /sku ITEM — one SKU detail\n\n"
            "   💬 *Or just type any question* and I will answer\n"
            "   directly from your forecast data.",
            reply_markup=_back_button(),
            parse_mode="Markdown"
        )
    elif query.data == "back":
        await query.edit_message_text(
            _dashboard_text(),
            reply_markup=_dashboard_buttons(),
            parse_mode="Markdown"
        )


# ==============================================================================
# Entry point
# ==============================================================================

def main():
    if not BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN is missing. Add it to your .env file.")

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("dashboard", dashboard))
    app.add_handler(CommandHandler("summary", summary))
    app.add_handler(CommandHandler("reorders", reorders))
    app.add_handler(CommandHandler("health", health))
    app.add_handler(CommandHandler("sku", sku_cmd))
    app.add_handler(CommandHandler("clearchat", clearchat))
    app.add_handler(CallbackQueryHandler(button_handler))

    # Free-text messages → RAG chat
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_handler))

    logger.info("Telegram bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
