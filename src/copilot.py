"""
Narrative co-pilot helpers for turning forecast outputs into business summaries.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import pandas as pd

from src import explainability


def _to_int(value) -> int:
    """Safely convert values to rounded integers for narrative display."""
    try:
        if pd.isna(value):
            return 0
        return int(round(float(value)))
    except Exception:
        return 0


def _pct(part: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{(part / total) * 100:.1f}%"


def _build_item_lines(df: pd.DataFrame, metric_col: str, limit: int = 5) -> List[str]:
    """Create compact bullet lines for top SKUs by a chosen metric."""
    if df is None or df.empty or metric_col not in df.columns:
        return []

    top_df = df[df[metric_col] > 0].nlargest(limit, metric_col)
    lines = []
    for _, row in top_df.iterrows():
        item_no = row.get("item_no", "Unknown")
        desc = str(row.get("item_description", "N/A"))[:50]
        vendor = row.get("vendor", "N/A")
        metric_value = _to_int(row.get(metric_col, 0))
        forecast = _to_int(row.get("forecast_m1", 0))
        stock = _to_int(row.get("total_stock", 0))
        lines.append(
            f"- `{item_no}` {desc} ({vendor}): {metric_col.replace('_', ' ')} {metric_value}, "
            f"forecast M1 {forecast}, stock {stock}"
        )
    return lines


def get_copilot_context(
    forecast_results: Dict,
    vendor_filter: str = "All",
    health_filter: str = "All",
) -> Dict:
    """
    Build a compact, narrative-friendly context from forecast results.
    """
    planner_output = forecast_results.get("planner_output")
    model_results = forecast_results.get("model_results")
    planning_summary = forecast_results.get("planning_summary", {})
    warnings = forecast_results.get("warnings", [])
    errors = forecast_results.get("errors", [])

    if planner_output is None or (hasattr(planner_output, "empty") and planner_output.empty):
        return {
            "has_data": False,
            "message": "No planner output is available yet.",
        }

    filtered_df = planner_output.copy()
    if vendor_filter != "All" and "vendor" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["vendor"] == vendor_filter]
    if health_filter != "All" and "stock_health" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["stock_health"] == health_filter]

    if filtered_df.empty:
        return {
            "has_data": False,
            "message": "No SKUs match the selected filters.",
        }

    if "overstock_qty" not in filtered_df.columns:
        filtered_df = filtered_df.copy()
        filtered_df["overstock_qty"] = (
            filtered_df.get("total_stock", 0) - filtered_df.get("projected_3m_demand", 0)
        ).clip(lower=0)

    total_skus = len(filtered_df)
    reorder_skus = int((filtered_df["reorder_qty"] > 0).sum()) if "reorder_qty" in filtered_df.columns else 0
    health_counts = filtered_df["stock_health"].value_counts().to_dict() if "stock_health" in filtered_df.columns else {}
    method_counts = filtered_df["forecast_method"].value_counts().to_dict() if "forecast_method" in filtered_df.columns else {}

    coverage = explainability.get_fallback_coverage_stats(filtered_df) if "forecast_method" in filtered_df.columns else {
        "ml_coverage": "0%",
        "fallback_coverage": "0%",
        "method_breakdown": {},
    }

    metric_block = {
        "total_skus": total_skus,
        "reorder_skus": reorder_skus,
        "reorder_sku_pct": _pct(reorder_skus, total_skus),
        "total_reorder_qty": _to_int(filtered_df["reorder_qty"].sum()) if "reorder_qty" in filtered_df.columns else 0,
        "total_stock": _to_int(filtered_df["total_stock"].sum()) if "total_stock" in filtered_df.columns else 0,
        "forecast_m1": _to_int(filtered_df["forecast_m1"].sum()) if "forecast_m1" in filtered_df.columns else 0,
        "projected_3m_demand": _to_int(filtered_df["projected_3m_demand"].sum()) if "projected_3m_demand" in filtered_df.columns else 0,
        "understock_count": int(health_counts.get("understock_risk", 0)),
        "healthy_count": int(health_counts.get("healthy_stock", 0)),
        "overstock_count": int(health_counts.get("overstock_risk", 0)),
        "ml_coverage": coverage.get("ml_coverage", "0%"),
        "fallback_coverage": coverage.get("fallback_coverage", "0%"),
    }

    validation_metrics = model_results.get("validation_metrics", {}) if model_results else {}

    return {
        "has_data": True,
        "vendor_filter": vendor_filter,
        "health_filter": health_filter,
        "metrics": metric_block,
        "planning_summary": planning_summary,
        "warnings": warnings,
        "errors": errors,
        "method_counts": method_counts,
        "top_reorder_lines": _build_item_lines(filtered_df, "reorder_qty"),
        "top_overstock_lines": _build_item_lines(filtered_df, "overstock_qty"),
        "top_forecast_lines": _build_item_lines(filtered_df, "forecast_m1"),
        "validation_metrics": validation_metrics,
    }


def _scope_line(context: Dict) -> str:
    vendor = context.get("vendor_filter", "All")
    health = context.get("health_filter", "All")
    scope_bits = []
    if vendor != "All":
        scope_bits.append(f"brand `{vendor}`")
    if health != "All":
        scope_bits.append(f"health `{health}`")
    if not scope_bits:
        return "Scope: all SKUs in the current forecast run."
    return "Scope: " + ", ".join(scope_bits) + "."


def _json_safe(value):
    """
    Convert pandas/numpy scalars to plain Python values for JSON serialization.
    """
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def get_openai_config() -> Dict[str, Optional[str]]:
    """
    Read OpenAI settings from environment variables or Streamlit secrets.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    try:
        import streamlit as st

        api_key = api_key or st.secrets.get("OPENAI_API_KEY")
        model = st.secrets.get("OPENAI_MODEL", model)
    except Exception:
        pass

    return {
        "api_key": api_key,
        "model": model,
        "configured": bool(api_key),
    }


def _build_openai_prompt(context: Dict, narrative_type: str, focus_note: str = "") -> str:
    """
    Build a grounded prompt from the structured forecast context.
    """
    prompt_parts = [
        "You are a retail demand planning narrative assistant.",
        "Write a concise, business-ready summary grounded only in the supplied context.",
        "Do not invent facts, SKUs, vendors, causes, or numbers.",
        "If the context shows uncertainty, mention it plainly.",
        "Use short markdown headings and bullets where helpful.",
        f"Narrative type: {narrative_type}",
    ]
    if focus_note.strip():
        prompt_parts.append(f"Requested focus: {focus_note.strip()}")

    prompt_parts.extend(
        [
            "Context JSON:",
            json.dumps(context, indent=2, default=_json_safe),
        ]
    )
    return "\n".join(prompt_parts)


def generate_narrative_with_openai(
    forecast_results: Dict,
    narrative_type: str = "executive",
    vendor_filter: str = "All",
    health_filter: str = "All",
    focus_note: str = "",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Generate a narrative using the OpenAI Responses API.
    """
    context = get_copilot_context(
        forecast_results,
        vendor_filter=vendor_filter,
        health_filter=health_filter,
    )

    if not context["has_data"]:
        return context["message"]

    config = get_openai_config()
    api_key = api_key or config["api_key"]
    model = model or config["model"] or "gpt-4.1-mini"

    if not api_key:
        raise ValueError("OpenAI API key not configured.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "OpenAI package not installed. Add `openai` to requirements and install dependencies."
        ) from exc

    client = OpenAI(api_key=api_key)
    prompt = _build_openai_prompt(context, narrative_type=narrative_type, focus_note=focus_note)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1200,
    )

    text = response.choices[0].message.content
    if text:
        return text.strip()

    raise RuntimeError("OpenAI response did not include output text.")


def generate_narrative(
    forecast_results: Dict,
    narrative_type: str = "executive",
    vendor_filter: str = "All",
    health_filter: str = "All",
    focus_note: str = "",
) -> str:
    """
    Generate a stakeholder-ready narrative from the forecast outputs.
    """
    context = get_copilot_context(
        forecast_results,
        vendor_filter=vendor_filter,
        health_filter=health_filter,
    )

    if not context["has_data"]:
        return context["message"]

    m = context["metrics"]
    warnings = context.get("warnings", [])
    validation_metrics = context.get("validation_metrics", {})
    method_counts = context.get("method_counts", {})

    note_block = f"\nRequested focus: {focus_note.strip()}\n" if focus_note.strip() else ""

    if narrative_type == "buyer":
        headline = "## Buyer Brief"
        opening = (
            f"This view covers {m['total_skus']} SKUs with {m['total_reorder_qty']} units of recommended reorder. "
            f"{m['understock_count']} SKUs look understocked while {m['overstock_count']} appear overstocked."
        )
        action = (
            f"Immediate buying attention should go to the {m['reorder_skus']} SKUs that require replenishment "
            f"({m['reorder_sku_pct']} of the scoped range)."
        )
        top_lines = context["top_reorder_lines"] or ["- No positive reorder quantities in the current scope."]
        risk_lines = context["top_overstock_lines"] or ["- No material overstock positions in the current scope."]
        sections = [
            headline,
            "",
            opening,
            _scope_line(context),
            note_block.strip(),
            "",
            "### Buying Priorities",
            action,
            *top_lines,
            "",
            "### Inventory Caution List",
            *risk_lines,
        ]
    elif narrative_type == "risk":
        headline = "## Risk Summary"
        opening = (
            f"The main risk profile is split across {m['understock_count']} understocked SKUs and "
            f"{m['overstock_count']} overstocked SKUs. Total projected 3-month demand is "
            f"{m['projected_3m_demand']} units against current stock of {m['total_stock']} units."
        )
        coverage = (
            f"Forecast coverage is {m['ml_coverage']} ML-driven and {m['fallback_coverage']} fallback-driven."
        )
        top_under = context["top_reorder_lines"] or ["- No understock-driven reorder actions in the current scope."]
        top_over = context["top_overstock_lines"] or ["- No overstock actions in the current scope."]
        sections = [
            headline,
            "",
            opening,
            _scope_line(context),
            note_block.strip(),
            "",
            "### Forecast Reliability",
            coverage,
            f"Method mix: {method_counts}" if method_counts else "Method mix not available.",
            "",
            "### Understock Watchlist",
            *top_under,
            "",
            "### Overstock Watchlist",
            *top_over,
        ]
    else:
        headline = "## Executive Summary"
        opening = (
            f"This forecast run covers {m['total_skus']} SKUs. Next-month demand is projected at "
            f"{m['forecast_m1']} units and 3-month demand at {m['projected_3m_demand']} units, with "
            f"{m['total_reorder_qty']} units of recommended reorder."
        )
        stock_view = (
            f"The assortment currently includes {m['understock_count']} understock-risk SKUs, "
            f"{m['healthy_count']} healthy SKUs, and {m['overstock_count']} overstock-risk SKUs."
        )
        coverage = (
            f"Model coverage is {m['ml_coverage']} ML-based and {m['fallback_coverage']} fallback-based."
        )
        top_lines = context["top_reorder_lines"] or ["- No major reorder actions identified in the current scope."]
        sections = [
            headline,
            "",
            opening,
            stock_view,
            coverage,
            _scope_line(context),
            note_block.strip(),
            "",
            "### Priority Reorder Items",
            *top_lines,
        ]

    if validation_metrics:
        wape = validation_metrics.get("wape")
        n_test = validation_metrics.get("n_test_samples")
        if wape is not None:
            sections.extend(
                [
                    "",
                    "### Model Signal",
                    f"Holdout validation WAPE is {wape:.1f}% across {n_test or 0} test observations.",
                ]
            )

    if warnings:
        warning_lines = [f"- {warning}" for warning in warnings[:5]]
        sections.extend(["", "### Data and Modeling Notes", *warning_lines])

    cleaned_sections = [section for section in sections if section != ""]
    return "\n".join(cleaned_sections).strip()
