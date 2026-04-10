"""
Plotly charts and visualizations for dashboards.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Dict, List


def chart_active_skus_by_health(planner_df: pd.DataFrame) -> go.Figure:
    """
    Bar chart of SKU counts by stock health status.
    
    Args:
        planner_df: OTB planner output DataFrame
    
    Returns:
        Plotly figure
    """
    health_counts = planner_df['stock_health'].value_counts()
    
    colors = {
        'understock_risk': '#ef553b',
        'healthy_stock': '#00cc96',
        'overstock_risk': '#ffa15a',
        'unknown': '#636efa'
    }
    
    fig = go.Figure(
        data=[go.Bar(
            x=health_counts.index,
            y=health_counts.values,
            marker_color=[colors.get(x, '#636efa') for x in health_counts.index],
            text=health_counts.values,
            textposition='auto',
        )]
    )
    
    fig.update_layout(
        title="SKUs by Stock Health Status",
        xaxis_title="Status",
        yaxis_title="Count",
        hovermode='x unified',
        height=400,
    )
    
    return fig


def chart_reorder_by_category(planner_df: pd.DataFrame) -> go.Figure:
    """
    Stacked bar chart of reorder qty by category and health status.
    
    Args:
        planner_df: OTB planner output
    
    Returns:
        Plotly figure
    """
    category_health = planner_df.groupby(['category', 'stock_health'])['reorder_qty'].sum().unstack(fill_value=0)
    
    colors = {
        'understock_risk': '#ef553b',
        'healthy_stock': '#00cc96',
        'overstock_risk': '#ffa15a',
    }
    
    fig = go.Figure()
    
    for health_status in category_health.columns:
        fig.add_trace(go.Bar(
            x=category_health.index,
            y=category_health[health_status],
            name=health_status.replace('_', ' ').title(),
            marker_color=colors.get(health_status, '#636efa'),
        ))
    
    fig.update_layout(
        title="Reorder Quantity by Category",
        xaxis_title="Category",
        yaxis_title="Reorder Qty",
        barmode='stack',
        hovermode='x',
        height=400,
    )
    
    return fig


def chart_forecast_vs_stock(planner_df: pd.DataFrame,
                           max_items: int = 15) -> go.Figure:
    """
    Grouped bar chart comparing current stock vs. 3-month demand.
    
    Args:
        planner_df: OTB planner output
        max_items: Max items to display (top by reorder qty)
    
    Returns:
        Plotly figure
    """
    # Get top items by reorder qty
    top_items = planner_df.nlargest(max_items, 'reorder_qty').copy()
    top_items['label'] = top_items['item_no'].astype(str) + ' - ' + top_items['item_description'].astype(str).str[:20]
    
    fig = go.Figure(data=[
        go.Bar(name='Current Stock',
               x=top_items['label'],
               y=top_items['total_stock'],
               marker_color='#636efa'),
        go.Bar(name='3-Month Demand',
               x=top_items['label'],
               y=top_items['projected_3m_demand'],
               marker_color='#ef553b'),
    ])
    
    fig.update_layout(
        title=f"Top {max_items} Items: Current Stock vs. 3-Month Demand",
        xaxis_title="Item",
        yaxis_title="Quantity",
        barmode='group',
        hovermode='x unified',
        height=500,
        xaxis_tickangle=-45,
    )
    
    return fig


def chart_forecast_method_distribution(planner_df: pd.DataFrame) -> go.Figure:
    """
    Pie chart showing distribution of forecast methods.
    
    Args:
        planner_df: OTB planner output
    
    Returns:
        Plotly figure
    """
    method_labels = {
        'ml_model': 'ML Model',
        'fallback_recent_avg': 'Recent Avg',
        'fallback_category_vendor': 'Brand Avg',
        'fallback_category': 'Overall Avg',
        'fallback_existing_forecast': 'Existing Forecast',
        'fallback_zero': 'Zero',
    }

    method_counts = planner_df['forecast_method'].value_counts()
    labels = [method_labels.get(x, x) for x in method_counts.index]
    
    colors = {
        'ml_model': '#636efa',
        'fallback_recent_avg': '#19d3f3',
        'fallback_category_vendor': '#00cc96',
        'fallback_category': '#ffa15a',
        'fallback_existing_forecast': '#ab63fa',
        'fallback_zero': '#ef553b',
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=method_counts.values,
        marker_colors=[colors.get(x, '#999') for x in method_counts.index],
    )])
    
    fig.update_layout(
        title="Forecast Method Distribution",
        height=400,
    )
    
    return fig


def chart_vendor_reorder_totals(planner_df: pd.DataFrame,
                               top_n: int = 10) -> go.Figure:
    """
    Horizontal bar chart of total reorder by vendor.
    
    Args:
        planner_df: OTB planner output
        top_n: Top vendors to show
    
    Returns:
        Plotly figure
    """
    vendor_totals = planner_df.groupby('vendor')['reorder_qty'].sum().nlargest(top_n)
    
    fig = go.Figure(data=[go.Bar(
        x=vendor_totals.values,
        y=vendor_totals.index,
        orientation='h',
        marker_color='#636efa',
        text=vendor_totals.values.astype(int),
        textposition='auto',
    )])
    
    fig.update_layout(
        title=f"Top {top_n} Vendors by Reorder Qty",
        xaxis_title="Total Reorder Qty",
        yaxis_title="Vendor",
        hovermode='y unified',
        height=400,
    )
    
    return fig


def chart_daily_sales_trend(master_df: pd.DataFrame,
                           item_no: str = None) -> go.Figure:
    """
    Line chart of monthly sales for a specific SKU or aggregate.
    
    Args:
        master_df: Master dataset
        item_no: Specific SKU to chart (None = all items aggregate)
    
    Returns:
        Plotly figure
    """
    if item_no:
        data = master_df[master_df['item_no'] == item_no].copy()
        title = f"Monthly Sales Trend - SKU {item_no}"
    else:
        data = master_df.copy()
        title = "Aggregate Monthly Sales Trend"
    
    data = data.sort_values('snapshot_date')
    
    if item_no:
        # Individual SKU
        fig = go.Figure(data=[go.Scatter(
            x=data['snapshot_date'],
            y=data['quantity'],
            mode='lines+markers',
            marker=dict(size=8, color='#636efa'),
            line=dict(width=2, color='#636efa'),
        )])
    else:
        # Aggregate by month
        agg_data = data.groupby('snapshot_date')['quantity'].sum().reset_index()
        agg_data = agg_data.sort_values('snapshot_date')
        fig = go.Figure(data=[go.Scatter(
            x=agg_data['snapshot_date'],
            y=agg_data['quantity'],
            mode='lines+markers',
            marker=dict(size=8, color='#636efa'),
            line=dict(width=2, color='#636efa'),
            fill='tozeroy',
        )])
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Quantity Sold",
        hovermode='x unified',
        height=400,
    )
    
    return fig


def chart_feature_importance(importance_df: pd.DataFrame,
                            top_n: int = 15) -> go.Figure:
    """
    Horizontal bar chart of feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of features to show
    
    Returns:
        Plotly figure
    """
    top_features = importance_df.head(top_n)
    
    fig = go.Figure(data=[go.Bar(
        x=top_features['importance'],
        y=top_features['feature'],
        orientation='h',
        marker_color='#636efa',
        text=top_features['importance'].round(4),
        textposition='auto',
    )])
    
    fig.update_layout(
        title=f"Top {top_n} Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        hovermode='y unified',
        height=500,
    )
    
    return fig


def chart_forecast_accuracy_scatter(test_df: pd.DataFrame,
                                   y_pred: np.ndarray) -> go.Figure:
    """
    Scatter plot of actual vs. predicted values.
    
    Args:
        test_df: Test dataset
        y_pred: Predicted values
    
    Returns:
        Plotly figure
    """
    y_true = test_df['quantity'].values
    
    fig = go.Figure()
    
    # Scatter of actual vs. predicted
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        marker=dict(size=6, color='#636efa', opacity=0.6),
        name='Predictions',
    ))
    
    # Perfect prediction line
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(dash='dash', color='red', width=2),
        name='Perfect Fit',
    ))
    
    fig.update_layout(
        title="Forecast Accuracy: Actual vs. Predicted",
        xaxis_title="Actual Quantity",
        yaxis_title="Predicted Quantity",
        hovermode='closest',
        height=400,
    )
    
    return fig
