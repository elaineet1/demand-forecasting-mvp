"""
Fallback forecasting logic for SKUs with sparse sales history.
Hierarchical fallback: ML -> Category+Vendor -> Category -> Existing Forecast -> Zero
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable
from src import config


def calculate_category_vendor_average(df: pd.DataFrame,
                                     category_col: str = 'category',
                                     vendor_col: str = 'vendor',
                                     value_col: str = 'quantity') -> Dict:
    """
    Calculate average sales by category-vendor combination.
    
    Args:
        df: Master dataset with aggregated sales
        category_col: Category column name
        vendor_col: Vendor column name
        value_col: Value to average (typically quantity)
    
    Returns:
        Dictionary keyed by (category, vendor) with mean value
    """
    if category_col not in df.columns or vendor_col not in df.columns:
        return {}
    grouped = df.groupby([category_col, vendor_col])[value_col].mean()
    return grouped.to_dict()


def calculate_category_average(df: pd.DataFrame,
                              category_col: str = 'category',
                              value_col: str = 'quantity') -> Dict:
    """
    Calculate average sales by category.
    
    Args:
        df: Master dataset
        category_col: Category column name
        value_col: Value to average
    
    Returns:
        Dictionary keyed by category with mean value
    """
    if category_col not in df.columns:
        return {}
    grouped = df.groupby(category_col)[value_col].mean()
    return grouped.to_dict()


def get_sku_history_length(df: pd.DataFrame,
                          sku_col: str = 'item_no',
                          time_col: str = 'snapshot_date') -> Dict:
    """
    Count number of monthly observations per SKU.
    
    Args:
        df: Master dataset
        sku_col: SKU column name
        time_col: Time column name
    
    Returns:
        Dictionary keyed by SKU with count of observations
    """
    counts = df.groupby(sku_col)[time_col].nunique()
    return counts.to_dict()


def get_latest_forecast_qty(df: pd.DataFrame,
                           sku_col: str = 'item_no',
                           forecast_col: str = 'forecast_qty') -> Dict:
    """
    Get most recent forecast_qty from inventory file for each SKU.
    
    Args:
        df: Dataframe with forecast_qty column
        sku_col: SKU column name
        forecast_col: Forecast quantity column name
    
    Returns:
        Dictionary keyed by SKU with latest forecast_qty
    """
    if forecast_col not in df.columns:
        return {}
    latest = df.sort_values('snapshot_date').groupby(sku_col)[forecast_col].last()
    return latest.to_dict()


def get_recent_average(df: pd.DataFrame,
                       sku_col: str = 'item_no',
                       date_col: str = 'snapshot_date',
                       value_col: str = 'quantity',
                       window: int = 3) -> Dict:
    """
    Compute recent rolling average demand for each SKU using the latest N months.
    """
    if df.empty or sku_col not in df.columns or date_col not in df.columns or value_col not in df.columns:
        return {}

    recent = (
        df[[sku_col, date_col, value_col]]
        .dropna(subset=[sku_col, date_col])
        .copy()
    )

    recent[date_col] = pd.to_datetime(recent[date_col], errors='coerce')
    recent = recent.dropna(subset=[date_col])

    if recent.empty:
        return {}

    # Use the latest month in the dataset as the reference window
    latest_month = recent[date_col].max().to_period('M')
    last_months = [(latest_month - (window - 1 - i)).to_timestamp(how='end') for i in range(window)]

    recent['month_end'] = recent[date_col].dt.to_period('M').dt.to_timestamp(how='end')

    avg_map: Dict = {}
    for sku, sku_df in recent.groupby(sku_col):
        month_values = sku_df.groupby('month_end')[value_col].sum().to_dict()
        values = [month_values.get(m, 0) for m in last_months]
        avg_map[sku] = float(np.mean(values))

    return avg_map


def apply_fallback_logic(row: pd.Series,
                        ml_predictor: Callable,
                        recent_avg: Dict,
                        cat_vendor_avg: Dict,
                        cat_avg: Dict,
                        existing_forecasts: Dict,
                        min_history_for_ml: int = config.MIN_HISTORY_FOR_ML) -> tuple:
    """
    Apply hierarchical fallback logic to determine forecast for a single SKU.
    
    Args:
        row: Row from forecasting dataframe
        ml_predictor: Function that takes features and returns (prediction, success)
        cat_vendor_avg: Category-vendor average dictionary
        cat_avg: Category average dictionary
        existing_forecasts: Dictionary of existing forecast_qty values
        min_history_for_ml: Minimum observations to use ML
    
    Returns:
        Tuple of (forecast_value, method_used)
    """
    sku = row.get('item_no')
    category = row.get('category')
    vendor = row.get('vendor') if 'vendor' in row else None
    history_length = row.get('history_length', 0)
    recent_value = (recent_avg.get(sku) if recent_avg else None)
    if recent_value is None:
        recent_value = 0

    # Method 0: Low-volume guardrail (use recent average)
    if recent_value <= config.LOW_VOLUME_THRESHOLD:
        return recent_value, 'fallback_recent_avg'

    # If recent average is zero, do not force a reorder from brand/overall averages
    if recent_value == 0:
        return 0, 'fallback_recent_avg'
    
    # Method 1: ML Model
    if history_length >= min_history_for_ml:
        try:
            pred, success = ml_predictor(row)
            if success:
                return pred, 'ml_model'
        except:
            pass
    
    # Method 2: Category + Vendor Average
    if category and vendor:
        key = (category, vendor)
        if key in cat_vendor_avg:
            return cat_vendor_avg[key], 'fallback_category_vendor'
    
    # Method 3: Category Average
    if category and category in cat_avg:
        return cat_avg[category], 'fallback_category'
    
    # Method 4: Existing Forecast Quantity from inventory file
    if sku in existing_forecasts:
        existing_val = existing_forecasts[sku]
        if pd.notna(existing_val) and existing_val > 0:
            return existing_val, 'fallback_existing_forecast'
    
    # Method 5: Default to zero
    return 0, 'fallback_zero'


def create_fallback_forecast_dataframe(skus_to_forecast: pd.DataFrame,
                                       master_data: pd.DataFrame,
                                       ml_predictor: Callable = None) -> pd.DataFrame:
    """
    Create forecast for all SKUs using fallback logic.
    
    Args:
        skus_to_forecast: DataFrame with unique SKUs and latest data
        master_data: Master dataset (for calculating averages)
        ml_predictor: Optional ML predictor function
    
    Returns:
        DataFrame with forecast_m1 and forecast_method columns
    """
    result = skus_to_forecast.copy()
    
    # Calculate history lengths
    history_lengths = get_sku_history_length(master_data)
    result['history_length'] = result['item_no'].map(history_lengths).fillna(0)
    
    # Calculate fallback averages
    recent_avg = get_recent_average(master_data)
    cat_vendor_avg = calculate_category_vendor_average(master_data)
    cat_avg = calculate_category_average(master_data)
    existing_forecasts = get_latest_forecast_qty(master_data)
    
    # Apply fallback logic to each row
    forecasts = []
    methods = []
    
    for idx, row in result.iterrows():
        pred, method = apply_fallback_logic(
            row,
            ml_predictor=ml_predictor or (lambda x: (0, False)),
            recent_avg=recent_avg,
            cat_vendor_avg=cat_vendor_avg,
            cat_avg=cat_avg,
            existing_forecasts=existing_forecasts,
        )
        forecasts.append(pred)
        methods.append(method)

    result['forecast_m1'] = forecasts
    result['forecast_method'] = methods
    result['recent_3m_avg'] = result['item_no'].map(recent_avg).fillna(0)
    
    return result


def estimate_forward_months(df: pd.DataFrame,
                           forecast_col: str = 'forecast_m1',
                           volatility_factor: float = 1.0,
                           history_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Estimate M2 and M3 forecasts from M1.
    Uses recent SKU trend when available, otherwise falls back to carry-forward.
    
    Args:
        df: DataFrame with M1 forecasts
        forecast_col: Name of M1 forecast column
        volatility_factor: Factor to apply (1.0 = no change, <1.0 = conservative)
    
    Returns:
        DataFrame with forecast_m2 and forecast_m3 added
    """
    df = df.copy()
    
    trend_map = {}
    category_trend_map = {}
    if history_df is not None and not history_df.empty:
        history_cols = ['item_no', 'snapshot_date', 'quantity']
        if 'category' in history_df.columns:
            history_cols.append('category')
        history = history_df[history_cols].dropna(subset=['item_no', 'snapshot_date', 'quantity'])
        history = history.sort_values(['item_no', 'snapshot_date'])

        def _trend_ratio_from_series(series: pd.Series) -> float | None:
            tail = series.tail(3)
            if len(tail) < 2:
                return None
            avg = tail.mean()
            last = tail.iloc[-1]
            if avg <= 0:
                return None
            ratio = float(last / avg)
            return float(np.clip(ratio, config.TREND_FACTOR_MIN, config.TREND_FACTOR_MAX))

        trend_series = history.groupby('item_no')['quantity'].apply(_trend_ratio_from_series)
        trend_map = trend_series.dropna().to_dict()

        # Category-level trend (sum of monthly sales across SKUs)
        if 'category' in history.columns:
            cat_monthly = (
                history.groupby(['category', 'snapshot_date'])['quantity']
                .sum()
                .reset_index()
                .sort_values(['category', 'snapshot_date'])
            )
            category_trend = (
                cat_monthly.groupby('category')['quantity']
                .apply(_trend_ratio_from_series)
                .dropna()
            )
            category_trend_map = category_trend.to_dict()

    # Simple carry-forward with optional scaling + trend factor
    if trend_map or category_trend_map:
        df['trend_factor'] = df['item_no'].map(trend_map)
        if 'category' in df.columns and category_trend_map:
            df['trend_factor'] = df['trend_factor'].fillna(df['category'].map(category_trend_map))
        df['trend_factor'] = df['trend_factor'].fillna(1.0)
    else:
        df['trend_factor'] = 1.0

    df['forecast_m2'] = df[forecast_col] * df['trend_factor'] * volatility_factor
    df['forecast_m3'] = df['forecast_m2'] * df['trend_factor']
    
    # Could add seasonality adjustment here if event calendar provided
    # For now, keep simple for MVP
    
    return df


def add_event_adjustments(df: pd.DataFrame,
                         calendar: pd.DataFrame,
                         forecast_col: str = 'forecast_m1',
                         event_threshold: float = 0.2,
                         scope_cols: List[str] = None) -> pd.DataFrame:
    """
    Apply event-based adjustments to forecasts if calendar provided.
    For example, boost forecast for Christmas month.
    
    Args:
        df: Forecast dataframe (with snapshot_date)
        calendar: Event calendar dataframe
        forecast_col: Forecast column to adjust
        event_threshold: Multiplier for event months (e.g., 1.2 = 20% boost)
    
    Returns:
        DataFrame with adjusted forecasts
    """
    df = df.copy()
    
    if 'snapshot_date' not in df.columns:
        return df
    
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    
    # Check for event flags
    event_flags = [
        'children_day', 'christmas', 'school_holiday',
        'year_end_holiday', 'summer_holiday', 'campaign_flag', 'launch_flag'
    ]

    if scope_cols is None:
        scope_cols = ['vendor', 'category', 'manufacturer']
    
    # Track event application flags per SKU/month
    event_flags_by_month = {}

    # For each future month, check if event is coming and adjust
    for month_offset, m_col in [(1, 'forecast_m1'), (2, 'forecast_m2'), (3, 'forecast_m3')]:
        if m_col not in df.columns:
            continue
        
        # Get upcoming month
        df['future_date'] = df['snapshot_date'] + pd.DateOffset(months=month_offset)
        
        # Check for events in future month
        has_event = pd.Series(False, index=df.index)
        for _, cal_row in calendar.iterrows():
            cal_date = pd.to_datetime(cal_row.get('date'))
            month_match = (
                (df['future_date'].dt.year == cal_date.year) &
                (df['future_date'].dt.month == cal_date.month)
            )

            scope_match = pd.Series(True, index=df.index)
            for scope_col in scope_cols:
                if scope_col in cal_row and pd.notna(cal_row.get(scope_col)):
                    if scope_col in df.columns:
                        scope_match = scope_match & (df[scope_col] == cal_row.get(scope_col))
                    else:
                        scope_match = False

            # Check if any event flag is true
            for flag in event_flags:
                if flag in cal_row and cal_row[flag] == 1:
                    has_event = has_event | (month_match & scope_match)
        
        # Apply multiplier to forecasts with events
        df[m_col] = df[m_col] * (1 + event_threshold * has_event.astype(int))
        event_flags_by_month[m_col] = has_event.astype(bool)

    # Add a per-SKU indicator so the UI can show whether calendar was applied
    if event_flags_by_month:
        for m_col, flag_series in event_flags_by_month.items():
            df[f'event_applied_{m_col}'] = flag_series.values
        any_event = pd.Series(False, index=df.index)
        for flag_series in event_flags_by_month.values():
            any_event = any_event | flag_series
        df['event_applied_any'] = any_event.values
    
    return df
