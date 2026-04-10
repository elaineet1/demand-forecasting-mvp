"""
Model explainability utilities.
Feature importance summary and model transparency.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def get_model_transparency_notes() -> Dict[str, str]:
    """
    Get human-readable model transparency notes for UI.
    
    Returns:
        Dictionary with explanation sections
    """
    return {
        'model_type': 'Gradient Boosting (LightGBM, HistGradientBoosting, or RandomForest)',
        'target': 'Monthly Quantity Sold (SKU-level)',
        'validation_approach': 'Time-Series Split (no random shuffle)',
        'data_requirements': 'Minimum 2 months of sales history per SKU for ML',
        'fallback_strategy': 'Brand avg → Overall avg → Existing forecast → Zero',
        'key_limitation_1': 'Limited historical depth reduces seasonal learning; more months improve accuracy',
        'key_limitation_2': 'New SKUs and sparse items rely on brand/vendor averages',
        'key_limitation_3': 'Holidays/campaigns not modeled unless calendar provided',
        'accuracy_caveat': 'Model performance limited by data sparsity; treat as estimate only',
    }


def get_model_assumptions_text() -> str:
    """
    Get text describing all model assumptions.
    
    Returns:
        Formatted assumptions text
    """
    assumptions = """
## Model Assumptions & Limitations

### Data Assumptions
- **Active Filter**: Only "Active = Y" SKUs are included in recommendations
- **Stock Definition**: TotalStock is end-of-period inventory (not dynamic)
- **Sales Aggregation**: Daily transactions aggregated to monthly level
- **Item Status**: Not filtered; included as-is for transparency

### Forecasting Constraints
1. **Limited Historical Data**: Depth of history limits seasonal learning
2. **New Product Launches**: Frequent new SKUs have sparse history; fallback rules used
3. **Sparse SKUs**: Products with <2 months of sales use brand/vendor averages
4. **Holiday Effects**: Not explicitly modeled unless event calendar provided

### Planning Assumptions
- **Stock Coverage Target**: 2-3 months of inventory considered healthy
- **Lead Time**: 3-month reorder lead time built into planning logic
- **Demand Stability**: Assumes next 3 months similar to historical pattern (MVP limitation)
- **No Cannibalization**: Model treats each SKU independently

### Model Limitations
- Accuracy limited by sparse data; WAPE may be 30-50%+
- Does not handle demand shocks or trend breaks
- Brand-level averages used for tier-1 forecasts are rough estimates
- Event calendar adjustments are multiplier-based, not probabilistic

### Recommendations
- Use as a decision support tool, not sole source of truth
- Review high-reorder-qty items manually
- Monitor actual vs. predicted monthly for recalibration
- Increase model confidence as data accumulates
"""
    return assumptions.strip()


def generate_sku_explanation(sku_row: pd.Series,
                           feature_importance_df: pd.DataFrame = None,
                           top_features: int = 3) -> str:
    """
    Generate human-readable explanation for a single SKU forecast.
    
    Args:
        sku_row: Row from final planner output
        feature_importance_df: Optional DataFrame with feature importance
        top_features: Number of top drivers to mention
    
    Returns:
        Explanation text
    """
    item_no = sku_row.get('item_no', 'Unknown')
    item_desc = sku_row.get('item_description', 'N/A')
    method = sku_row.get('forecast_method', 'unknown')
    health = sku_row.get('stock_health', 'unknown')
    forecast_m1 = sku_row.get('forecast_m1', 0)
    reorder = sku_row.get('reorder_qty', 0)
    stock = sku_row.get('total_stock', 0)
    demand_3m = sku_row.get('projected_3m_demand', 0)
    
    # Base explanation
    explanation = f"""
SKU {item_no} ({item_desc}):
- Forecast for next month: {forecast_m1:.0f} units
- Current stock: {stock:.0f} units
- 3-month projected demand: {demand_3m:.0f} units
- Recommended reorder: {reorder:.0f} units
- Stock health: {health.replace('_', ' ').title()}
"""
    
    # Method-specific explanation
    if method == 'ml_model':
        explanation += "- Forecast based on historical sales pattern (ML model)\n"
    elif method == 'fallback_category_vendor':
        vendor = sku_row.get('vendor', 'N/A')
        explanation += f"- Forecast based on {vendor} brand average\n"
    elif method == 'fallback_category':
        explanation += "- Forecast based on overall average (limited history)\n"
    elif method == 'fallback_existing_forecast':
        explanation += "- Using existing forecast quantity from inventory file\n"
    elif method == 'fallback_zero':
        explanation += "- No sales history; assuming zero demand\n"
    
    # Health-specific advice
    if health == 'understock_risk':
        explanation += "- ⚠️ ACTION: Current stock may be insufficient; prioritize reorder\n"
    elif health == 'healthy_stock':
        explanation += "- ✓ Stock level appropriate; standard reorder sufficient\n"
    elif health == 'overstock_risk':
        explanation += "- ⚠️ CAUTION: Excess stock; prioritize sales/clearance over reorder\n"
    
    return explanation.strip()


def get_model_performance_summary(metrics_dict: Dict) -> str:
    """
    Create human-readable model performance summary.
    
    Args:
        metrics_dict: Dictionary from validation (MAE, RMSE, WAPE, etc.)
    
    Returns:
        Summary text
    """
    mae = metrics_dict.get('mae', 0)
    rmse = metrics_dict.get('rmse', 0)
    wape = metrics_dict.get('wape', 0)
    mape = metrics_dict.get('mape')
    n_test = metrics_dict.get('n_test_samples', 0)
    
    summary = f"""
## Model Performance Summary

**Test Set Size**: {n_test} observations

**Accuracy Metrics**:
- Mean Absolute Error (MAE): {mae:.2f} units
- Root Mean Squared Error (RMSE): {rmse:.2f} units
- Weighted Absolute Percentage Error (WAPE): {wape:.1f}%
"""
    
    if mape is not None:
        summary += f"- Mean Absolute Percentage Error (MAPE): {mape:.1f}%\n"
    
    # Interpretation
    if wape < 20:
        interpretation = "**Accuracy Assessment**: Good (< 20% WAPE)"
    elif wape < 50:
        interpretation = "**Accuracy Assessment**: Acceptable but watch (20-50% WAPE)"
    else:
        interpretation = "**Accuracy Assessment**: Limited due to sparse data (> 50% WAPE)"
    
    summary += f"\n{interpretation}\n"
    test_start = metrics_dict.get('test_date_start')
    test_end = metrics_dict.get('test_date_end')
    test_months = metrics_dict.get('test_unique_months')
    if test_start and test_end:
        summary += f"\n**Test Period**: {test_start} to {test_end} ({test_months} months)\n"
    summary += "\n**Caveat**: Forecast accuracy is limited by available sales history depth.\n"
    summary += "As more data accumulates, model accuracy should improve."
    
    return summary.strip()


def get_fallback_coverage_stats(forecast_df: pd.DataFrame) -> Dict:
    """
    Calculate how many SKUs relied on fallback vs. ML.
    
    Args:
        forecast_df: Forecast output with forecast_method column
    
    Returns:
        Dictionary with coverage statistics
    """
    total = len(forecast_df)
    method_counts = forecast_df['forecast_method'].value_counts().to_dict()
    
    ml_count = method_counts.get('ml_model', 0)
    fallback_count = total - ml_count
    
    stats = {
        'total_skus': total,
        'ml_coverage': f"{ml_count / total * 100:.1f}%" if total > 0 else "0%",
        'fallback_coverage': f"{fallback_count / total * 100:.1f}%" if total > 0 else "0%",
        'method_breakdown': method_counts,
    }
    
    return stats


def get_feature_importance_text(importance_df: pd.DataFrame,
                               top_n: int = 10) -> str:
    """
    Generate text summary of feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
    
    Returns:
        Formatted text
    """
    if importance_df is None or importance_df.empty:
        return "Feature importance data not available."
    
    top_features = importance_df.head(top_n)
    
    text = f"## Top {top_n} Forecast Drivers\n\n"
    
    for idx, (_, row) in enumerate(top_features.iterrows(), 1):
        feature = row['feature']
        importance = row['importance']
        text += f"{idx}. **{feature}**: {importance:.4f}\n"
    
    return text.strip()
