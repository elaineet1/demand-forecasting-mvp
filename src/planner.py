"""
OTB (Open-to-Buy) planning calculations and stock health assessment.
Combines forecasts with inventory to recommend purchase quantities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from src import config


def calculate_3_month_demand(row: pd.Series,
                            use_average: bool = False) -> float:
    """
    Calculate 3-month projected demand from individual month forecasts.
    
    Args:
        row: Series with forecast_m1, forecast_m2, forecast_m3 columns
        use_average: If True, use simpler 3*forecast_m1 approach
    
    Returns:
        Total 3-month demand
    """
    if use_average:
        # Simple fallback: assume next 3 months similar to M1
        return row.get('forecast_m1', 0) * 3
    else:
        # Sum individual months
        m1 = row.get('forecast_m1', 0) or 0
        m2 = row.get('forecast_m2', 0) or 0
        m3 = row.get('forecast_m3', 0) or 0
        return float(m1 + m2 + m3)


def calculate_reorder_qty(total_stock: float,
                         projected_3m_demand: float,
                         lead_time_months: float = config.REQUEST_LEAD_TIME_MONTHS) -> float:
    """
    Calculate recommended reorder quantity.
    
    Logic: Need enough stock to cover lead time + planned period
    reorder_qty = max(projected_3m_demand - current_stock, 0)
    
    Args:
        total_stock: Current total stock
        projected_3m_demand: 3-month projected demand
        lead_time_months: Lead time for orders
    
    Returns:
        Recommended reorder quantity (non-negative)
    """
    target_stock = projected_3m_demand
    reorder = max(target_stock - total_stock, 0)
    return float(reorder)


def assess_stock_health(total_stock: float,
                       projected_monthly_demand: float,
                       healthy_min: float = config.STOCK_COVER_MONTHS_HEALTHY_MIN,
                       healthy_max: float = config.STOCK_COVER_MONTHS_HEALTHY_MAX) -> str:
    """
    Assess stock health based on months of coverage.
    
    Args:
        total_stock: Current stock quantity
        projected_monthly_demand: Forecasted monthly demand
        healthy_min: Minimum months of stock (e.g., 2)
        healthy_max: Maximum months of stock (e.g., 3)
    
    Returns:
        Stock health label: 'understock_risk', 'healthy_stock', or 'overstock_risk'
    """
    if projected_monthly_demand <= 0:
        return 'unknown'
    
    months_of_supply = total_stock / projected_monthly_demand
    
    if months_of_supply < healthy_min:
        return 'understock_risk'
    elif months_of_supply <= healthy_max:
        return 'healthy_stock'
    else:
        return 'overstock_risk'


def generate_reorder_remark(row: pd.Series) -> str:
    """
    Create a short human-readable explanation for the reorder recommendation.

    Args:
        row: Planner output row

    Returns:
        Plain-language remark
    """
    reorder_qty = float(row.get('reorder_qty', 0) or 0)
    stock = float(row.get('total_stock', 0) or 0)
    demand_3m = float(row.get('projected_3m_demand', 0) or 0)
    monthly_sales = float(row.get('latest_monthly_sales', 0) or 0)
    recent_avg = float(row.get('recent_3m_avg', 0) or 0)
    forecast_m1 = float(row.get('forecast_m1', 0) or 0)
    method = str(row.get('forecast_method', 'unknown') or 'unknown')
    health = str(row.get('stock_health', 'unknown') or 'unknown')

    method_labels = {
        'ml_model': 'ML forecast',
        'fallback_recent_avg': 'recent average fallback',
        'fallback_category_vendor': 'brand average fallback',
        'fallback_category': 'overall average fallback',
        'fallback_existing_forecast': 'existing forecast fallback',
        'fallback_zero': 'zero-demand fallback',
    }
    method_text = method_labels.get(method, method.replace('_', ' '))

    if reorder_qty > 0:
        base = (
            f"Reorder {int(np.ceil(reorder_qty))} units because current stock "
            f"({int(round(stock))}) is below projected 3-month demand "
            f"({int(np.ceil(demand_3m))}) based on {method_text}."
        )
        if health == 'understock_risk':
            return base + " Stock is in understock risk."
        return base

    if forecast_m1 <= 0 and demand_3m <= 0:
        return f"No reorder recommended because forecast demand is zero under {method_text}."

    if stock >= demand_3m and demand_3m > 0:
        if health == 'overstock_risk':
            return (
                f"No reorder recommended because current stock ({int(round(stock))}) already exceeds "
                f"projected 3-month demand ({int(np.ceil(demand_3m))}); item is overstocked."
            )
        return (
            f"No reorder recommended because current stock ({int(round(stock))}) already covers "
            f"projected 3-month demand ({int(np.ceil(demand_3m))}) from {method_text}."
        )

    if recent_avg > 0 or monthly_sales > 0:
        driver = int(round(recent_avg)) if recent_avg > 0 else int(round(monthly_sales))
        driver_name = "recent 3-month average" if recent_avg > 0 else "latest monthly sales"
        return (
            f"No reorder recommended after comparing stock ({int(round(stock))}) against forecast demand; "
            f"{driver_name} is {driver} units and the recommendation is driven by {method_text}."
        )

    return f"No reorder recommended based on {method_text}."


def create_otp_planner_output(forecast_df: pd.DataFrame,
                             inventory_latest: pd.DataFrame,
                             use_simple_3m: bool = False) -> pd.DataFrame:
    """
    Create the final OTB planner output table.
    
    Args:
        forecast_df: DataFrame with forecasts_m1, m2, m3
        inventory_latest: Latest inventory snapshot
        use_simple_3m: If True, use 3*M1 for 3-month demand
    
    Returns:
        Comprehensive planning DataFrame
    """
    result = forecast_df.copy()
    
    # Join with latest inventory data
    inv_cols = [
        'item_no', 'item_description', 'vendor', 'manufacturer',
        'category', 'subcategory', 'item_status', 'total_stock',
        'warehouse_stock', 'base_price', 'rrp', 'active'
    ]
    available_inv_cols = [col for col in inv_cols if col in inventory_latest.columns]
    
    result = result.merge(
        inventory_latest[available_inv_cols],
        on='item_no',
        how='left',
        suffixes=('', '_inv')
    )
    
    # Handle duplicate columns (take from inventory if available)
    for col in inv_cols:
        if col != 'item_no' and col + '_inv' in result.columns:
            result[col] = result[col + '_inv'].fillna(result[col])
            result = result.drop(columns=[col + '_inv'])
    
    # Calculate 3-month demand
    result['projected_3m_demand'] = result.apply(
        lambda row: calculate_3_month_demand(row, use_average=use_simple_3m),
        axis=1
    )
    
    # Capture latest observed monthly sales (units) if present in forecast data
    if 'quantity' in result.columns:
        result['latest_monthly_sales'] = pd.to_numeric(
            result['quantity'],
            errors='coerce'
        ).fillna(0)

    # Extract monthly demands for health check
    result['latest_monthly_forecast'] = result.get('forecast_m1', 0)
    
    # Calculate reorder quantity
    result['reorder_qty'] = result.apply(
        lambda row: calculate_reorder_qty(
            row.get('total_stock', 0) or 0,
            row.get('projected_3m_demand', 0) or 0
        ),
        axis=1
    )

    # Overstock quantity (if current stock exceeds projected demand)
    result['overstock_qty'] = (result.get('total_stock', 0) - result['projected_3m_demand']).clip(lower=0)
    
    # Assess stock health
    result['stock_health'] = result.apply(
        lambda row: assess_stock_health(
            row.get('total_stock', 0) or 0,
            row.get('latest_monthly_forecast', 0) or 1  # avoid div by zero
        ),
        axis=1
    )
    result['remark'] = result.apply(generate_reorder_remark, axis=1)

    # Stock cover months: current stock ÷ latest monthly sales (fallback to recent 3m avg)
    monthly_demand = pd.to_numeric(result.get('latest_monthly_sales', 0), errors='coerce').fillna(0)
    if 'recent_3m_avg' in result.columns:
        monthly_demand = monthly_demand.where(
            monthly_demand > 0,
            pd.to_numeric(result['recent_3m_avg'], errors='coerce').fillna(0)
        )
    result['stock_cover_months'] = np.where(
        monthly_demand > 0,
        (result['total_stock'] / monthly_demand).round(1),
        np.nan
    )
    
    # Keep only active SKUs
    if 'active' in result.columns:
        result = result[result['active'] == config.ACTIVE_SKU_FILTER].copy()
    
    # Filter by item_status if needed (keep only relevant statuses)
    if 'item_status' in result.columns:
        # Could add filtering here, but spec says "use ItemStatus as-is"
        pass
    
    # Round up forecasted and planning quantities
    for col in ['forecast_m1', 'forecast_m2', 'forecast_m3', 'projected_3m_demand', 'reorder_qty']:
        if col in result.columns:
            result[col] = np.ceil(pd.to_numeric(result[col], errors='coerce').fillna(0))

    # Round display-only quantities to whole units
    for col in ['latest_monthly_sales', 'total_stock', 'warehouse_stock', 'latest_monthly_forecast']:
        if col in result.columns:
            result[col] = np.round(pd.to_numeric(result[col], errors='coerce').fillna(0))

    # Force recent_3m_avg to whole numbers for display
    if 'recent_3m_avg' in result.columns:
        result['recent_3m_avg'] = np.round(
            pd.to_numeric(result['recent_3m_avg'], errors='coerce').fillna(0)
        ).astype(int)

    # Select and order final columns
    final_cols = [
        'item_no', 'item_description', 'vendor', 'manufacturer',
        'category', 'subcategory', 'item_status', 'active',
        'total_stock', 'warehouse_stock',
        'latest_monthly_sales', 'recent_3m_avg', 'overstock_qty',
        'latest_monthly_forecast', 'forecast_m1', 'forecast_m2', 'forecast_m3',
        'projected_3m_demand', 'reorder_qty', 'stock_cover_months', 'stock_health',
        'forecast_method', 'remark', 'event_applied_any', 'base_price', 'rrp'
    ]
    
    # Keep only columns that exist
    final_cols = [col for col in final_cols if col in result.columns]
    result = result[final_cols]
    
    # Fill NaN values in numeric columns
    numeric_cols = [
        'total_stock', 'latest_monthly_sales', 'latest_monthly_forecast', 'overstock_qty',
        'forecast_m1', 'forecast_m2',
        'forecast_m3', 'projected_3m_demand', 'reorder_qty'
    ]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors='coerce').fillna(0)
    
    return result


def get_planning_summary(planner_df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics from OTB planner output.
    
    Args:
        planner_df: Output from create_otp_planner_output
    
    Returns:
        Dictionary with planning summary statistics
    """
    summary = {
        'total_active_skus': len(planner_df),
        'total_current_stock': planner_df['total_stock'].sum(),
        'total_projected_3m_demand': planner_df['projected_3m_demand'].sum(),
        'total_reorder_qty': planner_df['reorder_qty'].sum(),
        'avg_projected_monthly_demand': planner_df['forecast_m1'].mean(),
        'avg_stock_health': planner_df.groupby('stock_health').size().to_dict(),
    }
    
    # Count by forecast method
    method_counts = planner_df['forecast_method'].value_counts().to_dict()
    summary['forecast_method_breakdown'] = method_counts
    
    # Top reorder candidates
    summary['top_reorder_items'] = len(planner_df[planner_df['reorder_qty'] > 0])
    
    return summary


def get_stock_health_summary(planner_df: pd.DataFrame) -> Dict:
    """
    Get detailed summary by stock health category.
    
    Args:
        planner_df: OTB planner output
    
    Returns:
        Dictionary with counts and totals by health status
    """
    health_statuses = ['understock_risk', 'healthy_stock', 'overstock_risk']
    summary = {}
    
    for status in health_statuses:
        subset = planner_df[planner_df['stock_health'] == status]
        summary[status] = {
            'count': len(subset),
            'total_stock': subset['total_stock'].sum(),
            'total_reorder_qty': subset['reorder_qty'].sum(),
            'avg_stock': subset['total_stock'].mean() if len(subset) > 0 else 0,
        }
    
    return summary
