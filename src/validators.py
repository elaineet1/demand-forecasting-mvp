"""
Data validation utilities.
Validates file format, required columns, and data quality issues.
"""

import pandas as pd
from typing import List, Dict, Tuple
from src import config


def validate_inventory_file(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that an inventory DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Tuple (is_valid, list_of_error_messages)
    """
    errors = []
    
    # Check for required columns
    missing_cols = [col for col in config.INVENTORY_REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required inventory columns: {', '.join(missing_cols)}")
    
    # Check for empty DataFrame
    if df.empty:
        errors.append("Inventory DataFrame is empty")
    
    # Check for total_stock column and data types
    if 'total_stock' in df.columns:
        try:
            pd.to_numeric(df['total_stock'], errors='coerce')
        except:
            errors.append("total_stock column contains non-numeric data")
    
    # Check for item_status column - should have valid values
    if 'item_status' in df.columns:
        unique_statuses = df['item_status'].unique()
        if len(unique_statuses) > 20:
            errors.append(f"Warning: item_status has many unique values ({len(unique_statuses)})")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def validate_sales_file(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that a sales DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Tuple (is_valid, list_of_error_messages)
    """
    errors = []
    
    # Check for required columns
    missing_cols = [col for col in config.SALES_REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required sales columns: {', '.join(missing_cols)}")
    
    # Check for empty DataFrame
    if df.empty:
        errors.append("Sales DataFrame is empty")
    
    # Check for quantity column and data types
    if 'quantity' in df.columns:
        try:
            pd.to_numeric(df['quantity'], errors='coerce')
        except:
            errors.append("quantity column contains non-numeric data")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def validate_event_calendar_file(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate event calendar file has reasonable structure.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Tuple (is_valid, list_of_error_messages)
    """
    errors = []
    
    # Check for at least date column
    if 'date' not in df.columns:
        errors.append("Event calendar must have a 'date' column")
    
    if df.empty:
        errors.append("Event calendar DataFrame is empty")
    
    return len(errors) == 0, errors


def check_data_quality(df: pd.DataFrame, file_type: str = 'inventory') -> Dict[str, any]:
    """
    Generate data quality report for a DataFrame.
    
    Args:
        df: DataFrame to check
        file_type: Type of file ('inventory', 'sales', or 'calendar')
    
    Returns:
        Dictionary with quality metrics
    """
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'columns': list(df.columns),
        'duplicates': df.duplicated().sum(),
        'null_counts': df.isnull().sum().to_dict(),
        'warnings': []
    }
    
    # File-specific checks
    if file_type == 'inventory':
        # Check active SKU count
        if 'active' in df.columns:
            active_count = (df['active'] == config.ACTIVE_SKU_FILTER).sum()
            report['active_sku_count'] = active_count
            if active_count == 0:
                report['warnings'].append("No active SKUs found (Active='Y')")
        
        # Check total_stock
        if 'total_stock' in df.columns:
            try:
                stock_numeric = pd.to_numeric(df['total_stock'], errors='coerce')
                report['avg_stock'] = stock_numeric.mean()
                report['null_stock_count'] = stock_numeric.isnull().sum()
            except:
                report['warnings'].append("Could not parse total_stock as numeric")
    
    elif file_type == 'sales':
        # Check sales quantity
        if 'quantity' in df.columns:
            try:
                qty_numeric = pd.to_numeric(df['quantity'], errors='coerce')
                report['total_quantity_sold'] = qty_numeric.sum()
                report['avg_qty_per_transaction'] = qty_numeric.mean()
            except:
                report['warnings'].append("Could not parse quantity as numeric")
        
        # Check sales amount
        if 'sales_amt' in df.columns:
            try:
                sales_numeric = pd.to_numeric(df['sales_amt'], errors='coerce')
                report['total_sales_amt'] = sales_numeric.sum()
            except:
                pass
    
    return report


def summarize_files_processed(inventory_dfs: List[Dict], sales_dfs: List[Dict]) -> Dict:
    """
    Generate summary of all files processed.
    
    Args:
        inventory_dfs: List of dicts with {filename, date, df, errors}
        sales_dfs: List of dicts with {filename, date, df, errors}
    
    Returns:
        Summary dictionary
    """
    summary = {
        'inventory_files_processed': len(inventory_dfs),
        'sales_files_processed': len(sales_dfs),
        'inventory_errors': sum(1 for d in inventory_dfs if d.get('errors')),
        'sales_errors': sum(1 for d in sales_dfs if d.get('errors')),
        'total_inventory_rows': sum(len(d.get('df', [])) for d in inventory_dfs),
        'total_sales_rows': sum(len(d.get('df', [])) for d in sales_dfs),
    }
    return summary
