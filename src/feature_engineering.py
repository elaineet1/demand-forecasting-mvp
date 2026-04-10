"""
Feature engineering for ML forecasting.
Creates lag features, rolling statistics, event flags, and encoded categoricals.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from src import config


def create_lag_features(df: pd.DataFrame,
                       group_col: str = 'item_no',
                       value_col: str = 'quantity',
                       lags: List[int] = None) -> pd.DataFrame:
    """
    Create lagged features for time-series prediction.
    
    Args:
        df: DataFrame sorted by item_no and snapshot_date
        group_col: Column to group by (default item_no)
        value_col: Column to create lags for
        lags: List of lag periods (default [1, 2])
    
    Returns:
        DataFrame with lag features added
    """
    if lags is None:
        lags = config.LAG_WINDOWS
    
    df = df.copy()
    
    for lag in lags:
        lag_col_name = f'{value_col}_lag_{lag}'
        # Group by item_no and shift within each group
        df[lag_col_name] = df.groupby(group_col)[value_col].shift(lag)
    
    return df


def create_rolling_features(df: pd.DataFrame,
                           group_col: str = 'item_no',
                           value_col: str = 'quantity',
                           window: int = None) -> pd.DataFrame:
    """
    Create rolling window statistics.
    
    Args:
        df: DataFrame
        group_col: Column to group by
        value_col: Column to calculate rolling stats for
        window: Window size (default from config)
    
    Returns:
        DataFrame with rolling features
    """
    if window is None:
        window = config.ROLLING_WINDOW
    
    df = df.copy()
    
    # Rolling mean
    rolling_col_name = f'{value_col}_rolling_mean_{window}'
    df[rolling_col_name] = df.groupby(group_col)[value_col].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    
    # Rolling std (can indicate volatility)
    rolling_std_col_name = f'{value_col}_rolling_std_{window}'
    df[rolling_std_col_name] = df.groupby(group_col)[value_col].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
    )
    
    return df


def create_temporal_features(df: pd.DataFrame,
                            date_col: str = 'snapshot_date') -> pd.DataFrame:
    """
    Create temporal features (month, quarter, year).
    
    Args:
        df: DataFrame
        date_col: Name of date column
    
    Returns:
        DataFrame with temporal features
    """
    df = df.copy()
    
    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Month, quarter, year already present in preprocess.py output
    # Add day of week, week of year for finer temporal patterns
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    df['day_of_month'] = df[date_col].dt.day
    
    return df


def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features related to pricing and margins.
    
    Args:
        df: DataFrame
    
    Returns:
        DataFrame with price features
    """
    df = df.copy()
    
    # Price delta: RRP vs Base Price (as a margin proxy)
    if 'rrp' in df.columns and 'base_price' in df.columns:
        df['price_margin'] = df['rrp'] - df['base_price']
        df['price_margin_pct'] = (df['price_margin'] / df['base_price'] * 100).replace([np.inf, -np.inf], np.nan)
    
    # Price change from last purchase
    if 'base_price' in df.columns and 'last_purchase_price' in df.columns:
        df['price_change'] = df['base_price'] - df['last_purchase_price']
    
    # Average cost (evaluated price)
    if 'evaluated_price' in df.columns and 'quantity' in df.columns:
        df['stock_value'] = df['evaluated_price'] * df['total_stock']
    
    return df


def create_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features related to inventory levels.
    
    Args:
        df: DataFrame
    
    Returns:
        DataFrame with stock features
    """
    df = df.copy()
    
    # Stock cover: how many months the current stock covers
    # Based on prior-month sales quantity to avoid leaking the current target.
    lagged_qty_col = 'quantity_lag_1' if 'quantity_lag_1' in df.columns else None
    if 'total_stock' in df.columns and lagged_qty_col is not None:
        monthly_sales = df[lagged_qty_col].replace(0, np.nan)
        df['stock_cover_months'] = (df['total_stock'] / monthly_sales).fillna(0)
        df['stock_cover_months'] = df['stock_cover_months'].replace([np.inf, -np.inf], 0)
    
    # Warehouse stock ratio
    if 'warehouse_stock' in df.columns and 'total_stock' in df.columns:
        df['warehouse_stock_ratio'] = (df['warehouse_stock'] / df['total_stock']).replace([np.inf, -np.inf], 0)
    
    return df


def encode_categorical_features(df: pd.DataFrame,
                                categorical_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, dict]:
    """
    One-hot or label encode categorical features.
    
    Args:
        df: DataFrame
        categorical_cols: List of columns to encode (auto-detect if None)
    
    Returns:
        Tuple of (encoded DataFrame, encoding mapping)
    """
    df = df.copy()
    
    if categorical_cols is None:
        # Auto-detect categorical columns from standard feature names
        categorical_cols = [
            'vendor', 'manufacturer', 'category', 'subcategory',
            'item_status'
        ]
        categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    # Create encoding mapping for reference
    encoding_map = {}
    
    # Use one-hot encoding for categorical features
    for col in categorical_cols:
        if col in df.columns:
            # Create dummy variables with column name prefix
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            
            # Store mapping
            encoding_map[col] = list(dummies.columns)
    
    return df, encoding_map


def select_features_for_model(df: pd.DataFrame,
                             target_col: str = 'quantity',
                             exclude_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select features for ML model training.
    Excludes IDs, dates, target, and other non-predictive columns.
    
    Args:
        df: Full featured DataFrame
        target_col: Name of target column
        exclude_cols: Additional columns to exclude
    
    Returns:
        Tuple of (feature DataFrame, list of feature column names)
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Base exclusions
    base_exclusions = [
        'item_no', 'item_description',  # IDs
        'snapshot_date',  # Dates (temporal info extracted separately)
        target_col,  # Target variable
        'forecast_method',  # Output column (we'll add this later)
    ]
    
    exclude_cols = list(set(base_exclusions + exclude_cols))
    
    # Select features
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Filter to numeric columns (ML models use numeric features)
    feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
    
    feature_df = df[feature_cols].copy()
    
    # Handle any remaining NaN values
    feature_df = feature_df.fillna(0)
    
    return feature_df, feature_cols


def create_full_feature_engineering_pipeline(df: pd.DataFrame,
                                            target_col: str = 'quantity') -> Tuple[pd.DataFrame, List[str]]:
    """
    Complete feature engineering pipeline.
    
    Args:
        df: Master dataset
        target_col: Target variable
    
    Returns:
        Tuple of (featured DataFrame, list of feature names used)
    """
    df = df.copy()
    
    # Sort by item_no and date for lag/rolling features
    if 'snapshot_date' not in df.columns:
        raise ValueError("DataFrame must have 'snapshot_date' column")
    
    df = df.sort_values(['item_no', 'snapshot_date']).reset_index(drop=True)
    
    # Create lag features
    df = create_lag_features(df, value_col=target_col, lags=config.LAG_WINDOWS)
    
    # Create rolling features
    df = create_rolling_features(df, value_col=target_col, window=config.ROLLING_WINDOW)
    
    # Create temporal features
    df = create_temporal_features(df)
    
    # Create price features
    df = create_price_features(df)
    
    # Create stock features
    df = create_stock_features(df)
    
    # Encode categorical features
    df, _ = encode_categorical_features(df)
    
    # Select features for model
    feature_df, feature_cols = select_features_for_model(df, target_col=target_col)
    
    # Add back non-feature columns that we'll need
    for col in ['item_no', 'snapshot_date', target_col]:
        if col in df.columns:
            feature_df[col] = df[col]
    
    return feature_df, feature_cols
