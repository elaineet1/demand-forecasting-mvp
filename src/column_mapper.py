"""
Column name normalization utilities.
Handles the various column naming conventions in Company inventory and sales files.
"""

import pandas as pd
from typing import Dict, List
import re


# Mapping of common column variants to standardized snake_case names
COLUMN_VARIANTS = {
    'item_no': [
        'item no.', 'item no', 'itemno', 'item_no', 'item number',
        'article no', 'article_no', 'skuno', 'sku no'
    ],
    'item_description': [
        'item description', 'item desc', 'description', 'item_description',
        'product description', 'product_description', 'desc'
    ],
    'vendor': [
        'vendor', 'vendor name', 'vendor_name', 'supplier'
    ],
    'manufacturer': [
        'manufacturer', 'mfr', 'mfr name', 'manufacturer_name', 'make'
    ],
    'total_stock': [
        'total stock', 'total_stock', 'totalstock', 'on hand',
        'on_hand', 'stock quantity', 'qty on hand', 'inventory'
    ],
    'warehouse_stock': [
        'warehouse stock', 'warehouse_stock', 'warehsestock', 'warehse stock',
        '$warehsestock', 'wh stock', 'warehouse qty'
    ],
    'base_price': [
        'base price', 'base_price', 'unit price', 'unit_price',
        'cost price', 'cost_price', 'purchase price'
    ],
    'rrp': [
        'rrp', 'rrp(incl.gst)', 'rrp (incl.gst)', 'rrp incl.gst',
        'rrp(incl. gst)', 'recommended retail price',
        'selling price', 'retail price'
    ],
    'last_purchase_price': [
        'last purchase price', 'last_purchase_price', 'last purchase amt',
        'last purchase amount', 'last_purchase_amount'
    ],
    'evaluated_price': [
        'evaluated price', 'evaluated_price', 'weighted avg price',
        'average cost price', 'valuation price'
    ],
    'forecast_qty': [
        'forecast qty', 'forecast_qty', 'forecastqty', 'forecast quantity',
        'forecast_quantity', 'projected qty'
    ],
    'active': [
        'active', 'active status', 'is_active', 'status',
        'active flag', 'is active'
    ],
    'item_status': [
        'item status', 'item_status', 'status', 'itemstatus',
        'product status', 'product_status'
    ],
    'category': [
        'category', 'category name', 'category_name', 'product category',
        'product_category', 'class'
    ],
    'subcategory': [
        'subcategory', 'sub category', 'sub_category', 'subcat',
        'sub_cat', 'product subcategory', 'type'
    ],
    'barcode': [
        'barcode', 'ean', 'upc', 'code', 'bar code'
    ],
    'vendor_item_code': [
        'vendor itemcode', 'vendor item code', 'vendor_itemcode',
        'vendor_item_code', 'supplier code', 'vendor sku'
    ],
    'sku_number': [
        'sku number', 'sku_number', 'skunumber', 'sku', 'skuno'
    ],
    'inventory_uom': [
        'inventory uom', 'inventory_uom', 'uom', 'unit of measure',
        'unit_of_measure', 'units'
    ],
    'quantity': [
        'quantity', 'qty', 'qty sold', 'qty_sold', 'units sold',
        'units_sold', 'sales qty', 'sales_qty', 'quantity sold'
    ],
    'sales_amt': [
        'sales amt', 'sales_amt', 'sales amount', 'sales_amount',
        'sales value', 'sales revenue', 'revenue'
    ],
    'customer_code': [
        'customer code', 'customer_code', 'cust code', 'cust_code',
        'customer id', 'customer_id', 'customer no'
    ],
    'customer_name': [
        'customer name', 'customer_name', 'customer', 'cust name',
        'cust_name', 'account name'
    ],
    'gross_profit': [
        'gross profit', 'gross_profit', 'gp', 'profit',
        'gross margin', 'gross_margin'
    ],
    'gross_profit_pct': [
        'gross profit %', 'gross_profit_%', 'gross profit pct',
        'gross_profit_pct', 'gp %', 'margin %', 'margin_pct'
    ],
    'total_cost': [
        'total cost', 'total_cost', 'cost', 'total_cost_of_goods',
        'cogs', 'cost amount'
    ],
}


def normalize_column_name(col_name: str) -> str:
    """
    Convert any column name to standardized snake_case.
    Handles spaces, dots, parens, and other special characters.
    
    Args:
        col_name: Original column name
    
    Returns:
        Normalized name in snake_case
    """
    # Strip whitespace
    col = col_name.strip()
    
    # Convert to lowercase
    col = col.lower()
    
    # Replace common separators with underscore
    col = re.sub(r'[.\s\-()]+', '_', col)
    
    # Remove duplicate underscores
    col = re.sub(r'_+', '_', col)
    
    # Remove leading/trailing underscores
    col = col.strip('_')
    
    # Replace special characters and currency symbols
    col = col.replace('$', '')
    col = re.sub(r'[^\w]', '', col)
    
    return col


def map_columns(df: pd.DataFrame, variant_map: Dict[str, List[str]] = None) -> pd.DataFrame:
    """
    Intelligently map DataFrame columns to standardized names.
    
    Args:
        df: Input DataFrame with inconsistent column names
        variant_map: Dictionary mapping standard names to variants.
                    If None, uses COLUMN_VARIANTS.
    
    Returns:
        DataFrame with renamed columns
    """
    if variant_map is None:
        variant_map = COLUMN_VARIANTS
    
    df = df.copy()
    
    # Normalize all existing column names to snake_case for matching
    normalized_cols = {col: normalize_column_name(col) for col in df.columns}
    
    # Build reverse mapping: normalized name -> original name
    reverse_normalized = {v: k for k, v in normalized_cols.items()}
    
    # Build mapping from original columns to standard names
    rename_map = {}
    for standard_name, variants in variant_map.items():
        # Normalize all variants
        norm_variants = [normalize_column_name(v) for v in variants]
        
        # Check which variant matches in our data
        for norm_variant in norm_variants:
            if norm_variant in reverse_normalized:
                original_col = reverse_normalized[norm_variant]
                rename_map[original_col] = standard_name
                break  # Use first match
    
    # Rename columns
    df = df.rename(columns=rename_map)
    
    return df


def detect_missing_required_columns(df: pd.DataFrame, required_cols: List[str]) -> List[str]:
    """
    Check if required columns are present in DataFrame.
    
    Args:
        df: DataFrame to check
        required_cols: List of required column names (in standard snake_case)
    
    Returns:
        List of missing column names (empty if all present)
    """
    present_cols = set(df.columns)
    missing = [col for col in required_cols if col not in present_cols]
    return missing


def get_column_mapping_summary(original_cols: List[str]) -> Dict[str, str]:
    """
    Generate a summary of how original columns map to standard names.
    Useful for showing users what was detected.
    
    Args:
        original_cols: List of column names from input file
    
    Returns:
        Dictionary mapping original -> standard names
    """
    summary = {}
    normalized = {normalize_column_name(col): col for col in original_cols}
    
    for standard_name, variants in COLUMN_VARIANTS.items():
        norm_variants = [normalize_column_name(v) for v in variants]
        for norm_variant in norm_variants:
            if norm_variant in normalized:
                original = normalized[norm_variant]
                summary[original] = standard_name
                break
    
    return summary


def get_unmapped_columns(original_cols: List[str]) -> List[str]:
    """
    Identify columns that don't map to any standard name.
    
    Args:
        original_cols: List of column names
    
    Returns:
        List of unmapped column names
    """
    mapping = get_column_mapping_summary(original_cols)
    return [col for col in original_cols if col not in mapping]
