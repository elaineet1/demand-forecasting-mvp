"""
I/O utilities for reading and writing data files.
Handles Excel, CSV files with robust error handling and date parsing from filenames.
"""

import pandas as pd
import os
from datetime import datetime
from typing import Optional, Tuple, List
import re


def parse_date_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract date from common filename patterns used in Company exports.
    Examples:
      - Item List_18.12.2021.xls -> 2021-12-18
      - Item List_1.9.2021_LOCAL.xls -> 2021-09-01
      - Sales Analysis by Items_1.9.2021 - 30.9.2021_CLOSING.xls -> 2021-09-30 (end date)
    
    Returns:
        datetime or None if no date found
    """
    # Pattern 1: DD.MM.YYYY format (most common in Company files)
    pattern1 = r'(\d{1,2})\.(\d{1,2})\.(\d{4})'
    matches = re.findall(pattern1, filename)
    
    if matches:
        # Use the last date found (in case of range like "1.9.2021 - 30.9.2021")
        last_match = matches[-1]
        day, month, year = last_match
        try:
            return datetime(int(year), int(month), int(day))
        except ValueError:
            pass
    
    # Pattern 2: YYYY-MM-DD format
    pattern2 = r'(\d{4})-(\d{1,2})-(\d{1,2})'
    matches = re.findall(pattern2, filename)
    if matches:
        year, month, day = matches[-1]
        try:
            return datetime(int(year), int(month), int(day))
        except ValueError:
            pass
    
    return None


def read_excel_file(file_path: str, sheet_name: int = 0) -> Tuple[pd.DataFrame, Optional[datetime]]:
    """
    Read an Excel file and parse date from filename.
    Automatically detects .xls (xlrd) vs .xlsx (openpyxl) format.
    
    Args:
        file_path: Path to Excel file
        sheet_name: Sheet index (default 0 for first sheet)
    
    Returns:
        Tuple of (DataFrame, parsed_date_from_filename)
    """
    try:
        # Determine correct engine based on file extension
        if file_path.endswith('.xls'):
            engine = 'xlrd'
        else:  # .xlsx or other
            engine = 'openpyxl'
        
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine=engine)
        filename = os.path.basename(file_path)
        parsed_date = parse_date_from_filename(filename)
        return df, parsed_date
    except Exception as e:
        raise ValueError(f"Failed to read Excel file {file_path}: {str(e)}")


def read_csv_file(file_path: str) -> Tuple[pd.DataFrame, Optional[datetime]]:
    """
    Read a CSV file and parse date from filename.
    
    Args:
        file_path: Path to CSV file
    
    Returns:
        Tuple of (DataFrame, parsed_date_from_filename)
    """
    try:
        df = pd.read_csv(file_path, dtype=str)  # Read as string initially for safety
        filename = os.path.basename(file_path)
        parsed_date = parse_date_from_filename(filename)
        return df, parsed_date
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {file_path}: {str(e)}")


def read_uploaded_file(uploaded_file) -> Tuple[pd.DataFrame, Optional[datetime], str]:
    """
    Read an uploaded Streamlit file (Excel or CSV).
    Returns the DataFrame, parsed date, and file name.
    Automatically detects .xls vs .xlsx for correct Excel engine.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        Tuple of (DataFrame, parsed_date, filename)
    """
    filename = uploaded_file.name
    
    try:
        if filename.endswith(('.xls', '.xlsx')):
            # Determine correct engine based on file extension
            engine = 'xlrd' if filename.endswith('.xls') else 'openpyxl'
            df = pd.read_excel(uploaded_file, sheet_name=0, engine=engine)
        elif filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file, dtype=str)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        
        parsed_date = parse_date_from_filename(filename)
        return df, parsed_date, filename
    
    except Exception as e:
        raise ValueError(f"Failed to read file {filename}: {str(e)}")


def save_dataframe_to_csv(df: pd.DataFrame, output_path: str) -> bool:
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        output_path: Path to save CSV
    
    Returns:
        True if successful, False otherwise
    """
    try:
        df.to_csv(output_path, index=False)
        return True
    except Exception as e:
        print(f"Failed to save CSV to {output_path}: {str(e)}")
        return False


def save_dataframe_to_excel(df: pd.DataFrame, output_path: str) -> bool:
    """
    Save DataFrame to Excel file.
    
    Args:
        df: DataFrame to save
        output_path: Path to save Excel
    
    Returns:
        True if successful, False otherwise
    """
    try:
        df.to_excel(output_path, index=False, engine='openpyxl')
        return True
    except Exception as e:
        print(f"Failed to save Excel to {output_path}: {str(e)}")
        return False


def merge_dataframes_with_date_column(dfs_with_dates: List[Tuple[pd.DataFrame, Optional[datetime]]],
                                      file_names: List[str],
                                      snapshot_col_name: str = 'snapshot_date') -> pd.DataFrame:
    """
    Merge multiple DataFrames and add a date column to each.
    
    Args:
        dfs_with_dates: List of tuples (DataFrame, parsed_date)
        file_names: List of filenames for reference
        snapshot_col_name: Name of the date column to add
    
    Returns:
        Combined DataFrame with date column
    """
    combined_dfs = []
    
    for (df, parsed_date), filename in zip(dfs_with_dates, file_names):
        df_copy = df.copy()
        
        # Add date column
        if parsed_date is not None:
            df_copy[snapshot_col_name] = parsed_date
        else:
            # If no date parsed, use today's date with a warning
            df_copy[snapshot_col_name] = datetime.now()
        
        combined_dfs.append(df_copy)
    
    # Concatenate all dataframes
    result = pd.concat(combined_dfs, ignore_index=True)
    
    # Ensure consistent datetime64[ns] dtype for snapshot_date to avoid merge conflicts
    if snapshot_col_name in result.columns:
        result[snapshot_col_name] = pd.to_datetime(result[snapshot_col_name], utc=False).astype('datetime64[ns]')
    
    return result
