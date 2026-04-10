"""
Data preprocessing and cleaning utilities.
Handles data type conversion, missing values, outliers, and aggregation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from datetime import datetime
from typing import Tuple, Optional, List
import warnings
import re
import numpy as np
from src import config


def clean_numeric_column(series: pd.Series, col_name: str) -> pd.Series:
    """
    Convert column to numeric, replacing invalid values with NaN.
    
    Args:
        series: Series to clean
        col_name: Column name (for logging)
    
    Returns:
        Cleaned numeric series
    """
    try:
        # Try direct conversion first
        return pd.to_numeric(series, errors='coerce')
    except Exception as e:
        warnings.warn(f"Error converting {col_name} to numeric: {str(e)}")
        return pd.Series(np.nan, index=series.index)


def aggregate_sales_to_sku_month(sales_df: pd.DataFrame,
                                  snapshot_date_col: str = 'snapshot_date') -> pd.DataFrame:
    """
    Aggregate daily/transaction sales to SKU-month level.
    
    Args:
        sales_df: Sales DataFrame with transactions
        snapshot_date_col: Name of date column
    
    Returns:
        Aggregated DataFrame at SKU-month level
    """
    df = sales_df.copy()
    
    # Ensure required columns exist
    required = ['item_no', 'quantity', snapshot_date_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Convert numeric columns
    df['quantity'] = clean_numeric_column(df['quantity'], 'quantity')
    
    if 'sales_amt' in df.columns:
        df['sales_amt'] = clean_numeric_column(df['sales_amt'], 'sales_amt')
    else:
        df['sales_amt'] = np.nan
    
    if 'gross_profit' in df.columns:
        df['gross_profit'] = clean_numeric_column(df['gross_profit'], 'gross_profit')
    else:
        df['gross_profit'] = np.nan
    
    if 'gross_profit_pct' in df.columns:
        df['gross_profit_pct'] = clean_numeric_column(df['gross_profit_pct'], 'gross_profit_pct')
    else:
        df['gross_profit_pct'] = np.nan
    
    # Convert snapshot_date to datetime
    df[snapshot_date_col] = pd.to_datetime(df[snapshot_date_col], errors='coerce')
    
    # Extract year-month from snapshot_date
    df['year'] = df[snapshot_date_col].dt.year
    df['month'] = df[snapshot_date_col].dt.month
    df['year_month'] = df[snapshot_date_col].dt.to_period('M')
    
    # Group by item_no and year_month
    agg_dict = {
        'quantity': 'sum',
        'sales_amt': 'sum',
        'gross_profit': 'sum',
    }
    
    grouped = df.groupby(['item_no', 'year_month'], as_index=False).agg(agg_dict)
    
    # Convert year_month back to timestamp (end of month)
    grouped[snapshot_date_col] = grouped['year_month'].dt.to_timestamp(freq='M')
    grouped['year'] = grouped[snapshot_date_col].dt.year
    grouped['month'] = grouped[snapshot_date_col].dt.month
    
    # Keep only useful columns
    result = grouped[['item_no', snapshot_date_col, 'year', 'month', 'quantity', 'sales_amt', 'gross_profit']].copy()

    # Clamp negative monthly sales to zero (returns/corrections can cause negatives)
    negative_monthly_sales = int((result['quantity'] < 0).sum())
    result['quantity'] = result['quantity'].clip(lower=0)

    # Calculate gross profit percentage where possible
    result['gross_profit_pct'] = (result['gross_profit'] / result['sales_amt'] * 100).where(
        result['sales_amt'] > 0, np.nan
    )
    
    # Drop completely null rows
    result = result.dropna(subset=['quantity'])
    
    result.attrs['negative_monthly_sales_corrected'] = negative_monthly_sales
    return result


def prepare_inventory_latest_snapshot(inventory_df: pd.DataFrame,
                                      snapshot_col: str = 'snapshot_date') -> pd.DataFrame:
    """
    Get latest inventory snapshot per SKU (item_no).
    
    Args:
        inventory_df: Inventory DataFrame (may have multiple dates)
        snapshot_col: Name of snapshot date column
    
    Returns:
        Latest snapshot per item_no
    """
    df = inventory_df.copy()
    
    # Ensure we have snapshot column
    if snapshot_col not in df.columns:
        df[snapshot_col] = datetime.now()
    
    df[snapshot_col] = pd.to_datetime(df[snapshot_col], errors='coerce')
    
    # Convert numeric columns
    numeric_cols = [
        'total_stock', 'warehouse_stock', 'base_price', 'rrp',
        'last_purchase_price', 'evaluated_price', 'forecast_qty'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col], col)

    for stock_col in ['warehouse_stock', 'total_stock']:
        if stock_col in df.columns:
            df.loc[df[stock_col] < 0, stock_col] = 0
    
    # Get latest snapshot per item_no
    latest = df.sort_values(snapshot_col).groupby('item_no', as_index=False).last()
    
    # Ensure snapshot_date column is present in output
    if snapshot_col not in latest.columns:
        latest[snapshot_col] = datetime.now()
    
    return latest


def clean_event_calendar(calendar_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize event calendar.
    
    Args:
        calendar_df: Event calendar DataFrame
    
    Returns:
        Cleaned calendar DataFrame
    """
    df = calendar_df.copy()
    
    # Parse date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        raise ValueError("Event calendar must have 'date' column")
    
    # For boolean-like columns, convert to int (0/1)
    bool_cols = [
        'children_day', 'christmas', 'school_holiday', 'year_end_holiday',
        'summer_holiday', 'campaign_flag', 'launch_flag'
    ]
    
    for col in bool_cols:
        if col in df.columns:
            # Handle various true/false representations
            df[col] = df[col].astype(str).str.lower().isin(['y', 'yes', '1', 'true']).astype(int)
        else:
            # If column doesn't exist, create it with all zeros
            df[col] = 0

    # Normalize brand/vendor scope columns if provided
    if 'brand' in df.columns and 'vendor' not in df.columns:
        df['vendor'] = df['brand']
    for scope_col in ['vendor', 'brand', 'manufacturer', 'category']:
        if scope_col in df.columns:
            df[scope_col] = df[scope_col].astype(str).str.strip()
            df.loc[df[scope_col].isin(['', 'nan', 'None']), scope_col] = np.nan
    
    return df


def load_brand_mapping(mapping_path: str) -> Dict[str, str]:
    """
    Load brand mapping from CSV with columns: vendor_raw, brand.
    Returns a dict mapping vendor_raw -> brand.
    """
    if mapping_path is None or not mapping_path:
        return {}
    if not Path(mapping_path).exists():
        return {}

    try:
        mapping_df = pd.read_csv(mapping_path)
    except Exception:
        return {}

    required_cols = {'vendor_raw', 'brand'}
    if not required_cols.issubset(set(mapping_df.columns)):
        return {}

    mapping_df['vendor_raw'] = mapping_df['vendor_raw'].astype(str).str.strip()
    mapping_df['brand'] = mapping_df['brand'].astype(str).str.strip()
    mapping_df = mapping_df[(mapping_df['vendor_raw'] != '') & (mapping_df['brand'] != '')]

    return dict(zip(mapping_df['vendor_raw'], mapping_df['brand']))


def apply_brand_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Normalize vendor to brand using a mapping dict. Leaves unmapped values unchanged.
    """
    if df is None or df.empty or not mapping:
        return df
    if 'vendor' not in df.columns:
        return df

    df = df.copy()
    df['vendor'] = df['vendor'].astype(str).str.strip()
    df['vendor'] = df['vendor'].replace(mapping)
    return df


def create_master_dataset(inventory_history: pd.DataFrame,
                         sales_agg: pd.DataFrame,
                         calendar: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Create master dataset combining inventory and sales data.
    Matches sales to latest inventory snapshot per SKU.
    
    Args:
        inventory_history: Inventory snapshots across time
        sales_agg: Aggregated monthly sales
        calendar: Optional event calendar
    
    Returns:
        Master dataset ready for feature engineering
    """
    # Start with aggregated sales
    master = sales_agg.copy()
    
    inventory_history_df = inventory_history.copy()
    inventory_history_df['snapshot_date'] = pd.to_datetime(
        inventory_history_df['snapshot_date'],
        errors='coerce'
    )
    master['snapshot_date'] = pd.to_datetime(master['snapshot_date'], errors='coerce')

    # Join each sales month to the most recent inventory snapshot on or before that month.
    inventory_history_df = inventory_history_df.sort_values(['item_no', 'snapshot_date']).reset_index(drop=True)
    master = master.sort_values(['item_no', 'snapshot_date']).reset_index(drop=True)
    
    # Ensure consistent datetime64[ns] dtype BEFORE the merge loop
    inventory_history_df['snapshot_date'] = pd.to_datetime(inventory_history_df['snapshot_date'], utc=False).astype('datetime64[ns]')
    master['snapshot_date'] = pd.to_datetime(master['snapshot_date'], utc=False).astype('datetime64[ns]')

    merged_groups = []
    inventory_cols = [col for col in inventory_history_df.columns if col != 'item_no']

    for item_no, sales_group in master.groupby('item_no', sort=False):
        sales_group = sales_group.copy()  # Make explicit copy from groupby view
        sales_group = sales_group.sort_values('snapshot_date').reset_index(drop=True)
        inv_group = inventory_history_df[inventory_history_df['item_no'] == item_no].copy()

        if inv_group.empty:
            for col in inventory_cols:
                if col not in sales_group.columns:
                    sales_group[col] = np.nan
            merged_groups.append(sales_group)
            continue

        inv_group = inv_group.sort_values('snapshot_date').reset_index(drop=True)
        
        merged_group = pd.merge_asof(
            sales_group,
            inv_group.drop(columns=['item_no']),
            on='snapshot_date',
            direction='backward',
            suffixes=('', '_inv')
        )
        merged_groups.append(merged_group)

    master = pd.concat(merged_groups, ignore_index=True)
    
    # Normalize datetime dtype to ensure consistency for downstream merges
    if 'snapshot_date' in master.columns:
        master['snapshot_date'] = pd.to_datetime(master['snapshot_date'], utc=False).astype('datetime64[ns]')
    
    # Add calendar features if provided
    if calendar is not None:
        # Rename date column for clarity
        calendar_copy = calendar.copy().rename(columns={'date': 'snapshot_date'})
        
        # Normalize calendar datetime dtype to match
        if 'snapshot_date' in calendar_copy.columns:
            calendar_copy['snapshot_date'] = pd.to_datetime(calendar_copy['snapshot_date'], utc=False).astype('datetime64[ns]')
        
        # Join calendar by date (many-to-one, as calendar has one entry per date)
        master = master.merge(
            calendar_copy,
            on='snapshot_date',
            how='left'
        )
        
        # Fill NaN boolean columns with 0
        bool_cols = [
            'children_day', 'christmas', 'school_holiday', 'year_end_holiday',
            'summer_holiday', 'campaign_flag', 'launch_flag'
        ]
        for col in bool_cols:
            if col in master.columns:
                master[col] = master[col].fillna(0).astype(int)
    
    # Sort by item_no and date
    master = master.sort_values(['item_no', 'snapshot_date']).reset_index(drop=True)
    
    return master


def handle_missing_values(df: pd.DataFrame, strategy: str = 'forward_fill') -> pd.DataFrame:
    """
    Handle missing values in master dataset.
    
    Args:
        df: DataFrame with possible missing values
        strategy: 'forward_fill', 'backward_fill', or 'drop'
    
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    # Numeric columns to forward fill (inventory stays constant between sales snapshots)
    numeric_cols = [
        'total_stock', 'warehouse_stock', 'base_price', 'rrp',
        'last_purchase_price', 'evaluated_price', 'forecast_qty',
        'item_description', 'vendor', 'manufacturer', 'category',
        'subcategory', 'item_status', 'active'
    ]
    
    if strategy == 'forward_fill':
        # Forward fill within each item_no group
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df.groupby('item_no')[col].transform(lambda x: x.ffill())
    
    elif strategy == 'backward_fill':
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df.groupby('item_no')[col].transform(lambda x: x.bfill())
    
    # Drop rows with critical missing values
    critical_cols = ['item_no', 'quantity', 'snapshot_date']
    df = df.dropna(subset=critical_cols)
    
    return df


def remove_outliers_iqr(df: pd.DataFrame,
                        column: str,
                        multiplier: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers using IQR method.
    
    Args:
        df: DataFrame
        column: Column to check for outliers
        multiplier: IQR multiplier (default 1.5 for standard IQR)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in df.columns:
        return df
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def clean_for_modeling(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Clean the master dataset before feature engineering and ML training.

    This keeps the cleaning conservative: remove clearly invalid rows, standardize
    key fields, and clip extreme quantity outliers instead of dropping them.

    Args:
        df: Master dataset

    Returns:
        Tuple of (cleaned_df, cleaning_report)
    """
    cleaned = df.copy()
    report = {
        'rows_in': len(cleaned),
        'rows_out': 0,
        'duplicate_rows_removed': 0,
        'rows_missing_keys_removed': 0,
        'rows_negative_qty_removed': 0,
        'rows_bad_dates_removed': 0,
        'rows_missing_item_removed': 0,
        'quantity_outliers_clipped': 0,
        'skus_with_short_history': 0,
        'non_data_columns_removed': 0,
        'negative_stock_clipped': 0,
    }

    if cleaned.empty:
        return cleaned, report

    # Drop index-like or non-data columns that sometimes appear in Excel exports.
    non_data_cols = []
    for col in cleaned.columns:
        col_str = str(col).strip()
        if col_str == '#':
            non_data_cols.append(col)
            continue
        if col_str.lower().startswith('unnamed'):
            non_data_cols.append(col)
            continue
        normalized = re.sub(r'[^\w]', '', col_str.lower())
        if normalized == '':
            non_data_cols.append(col)

    if non_data_cols:
        cleaned = cleaned.drop(columns=list(set(non_data_cols)), errors='ignore')
        report['non_data_columns_removed'] = len(set(non_data_cols))

    # Normalize common text columns used downstream by grouping / fallback logic.
    text_cols = [
        'item_no', 'item_description', 'vendor', 'manufacturer',
        'category', 'subcategory', 'item_status', 'active'
    ]
    for col in text_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.strip()
            cleaned.loc[cleaned[col].isin(['', 'nan', 'None']), col] = np.nan

    if 'snapshot_date' in cleaned.columns:
        cleaned['snapshot_date'] = pd.to_datetime(cleaned['snapshot_date'], errors='coerce')

    numeric_cols = [
        'quantity', 'sales_amt', 'gross_profit', 'gross_profit_pct',
        'total_stock', 'warehouse_stock', 'base_price', 'rrp',
        'last_purchase_price', 'evaluated_price', 'forecast_qty'
    ]
    for col in numeric_cols:
        if col in cleaned.columns:
            cleaned[col] = clean_numeric_column(cleaned[col], col)

    before = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    report['duplicate_rows_removed'] = before - len(cleaned)

    if 'item_no' in cleaned.columns:
        before = len(cleaned)
        cleaned = cleaned.dropna(subset=['item_no'])
        report['rows_missing_item_removed'] = before - len(cleaned)

    if 'snapshot_date' in cleaned.columns:
        before = len(cleaned)
        cleaned = cleaned.dropna(subset=['snapshot_date'])
        report['rows_bad_dates_removed'] = before - len(cleaned)

    critical_cols = [col for col in ['item_no', 'snapshot_date', 'quantity'] if col in cleaned.columns]
    if critical_cols:
        before = len(cleaned)
        cleaned = cleaned.dropna(subset=critical_cols)
        report['rows_missing_keys_removed'] = before - len(cleaned)

    if 'quantity' in cleaned.columns:
        before = len(cleaned)
        cleaned = cleaned[cleaned['quantity'] >= config.MODEL_MAX_NEGATIVE_QUANTITY].copy()
        report['rows_negative_qty_removed'] = before - len(cleaned)

        # Clip extreme quantity outliers per SKU (avoid global caps that
        # suppress legitimately high-volume items).
        def _clip_series_iqr(series: pd.Series) -> pd.Series:
            non_null = series.dropna()
            if len(non_null) < 4:
                return series
            q1 = non_null.quantile(0.25)
            q3 = non_null.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr <= 0:
                return series
            upper_bound = q3 + config.MODEL_OUTLIER_IQR_MULTIPLIER * iqr
            return series.clip(upper=upper_bound)

        before_qty = cleaned['quantity'].copy()
        if 'item_no' in cleaned.columns:
            cleaned['quantity'] = cleaned.groupby('item_no')['quantity'].transform(_clip_series_iqr)
        else:
            cleaned['quantity'] = _clip_series_iqr(cleaned['quantity'])

        report['quantity_outliers_clipped'] = int((before_qty > cleaned['quantity']).sum())

    # Clamp negative inventory values to zero (warehouse and total stock).
    for stock_col in ['warehouse_stock', 'total_stock']:
        if stock_col in cleaned.columns:
            neg_mask = cleaned[stock_col] < 0
            neg_count = int(neg_mask.sum())
            if neg_count > 0:
                cleaned.loc[neg_mask, stock_col] = 0
                report['negative_stock_clipped'] += neg_count

    if 'snapshot_date' in cleaned.columns and 'item_no' in cleaned.columns:
        cleaned = cleaned.sort_values(['item_no', 'snapshot_date']).reset_index(drop=True)
        history_counts = cleaned.groupby('item_no')['snapshot_date'].nunique()
        report['skus_with_short_history'] = int(
            (history_counts < config.MODEL_MIN_SKU_HISTORY).sum()
        )

    report['rows_out'] = len(cleaned)
    return cleaned, report
