"""
Main forecasting pipeline orchestrator.
Connects all preprocessing, feature engineering, and modeling steps.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
from datetime import datetime

from src import (
    config, io_utils, column_mapper, validators,
    preprocess, feature_engineering, model_train,
    fallback, planner, metrics, persistence
)


def end_to_end_forecast_pipeline(
    inventory_dfs: List[Tuple[pd.DataFrame, Optional[datetime], str]],
    sales_dfs: List[Tuple[pd.DataFrame, Optional[datetime], str]],
    calendar_df: Optional[pd.DataFrame] = None,
    use_simple_3m: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Complete forecast pipeline from raw files to OTB recommendations.
    
    Args:
        inventory_dfs: List of (df, date, filename) tuples
        sales_dfs: List of (df, date, filename) tuples
        calendar_df: Optional event calendar
        use_simple_3m: If True, use 3*M1 for 3-month demand
        verbose: Print progress messages
    
    Returns:
        Dictionary with results and artifacts
    """
    
    results = {
        'success': False,
        'errors': [],
        'warnings': [],
        'inventory_processed': None,
        'sales_processed': None,
        'calendar_processed': None,
        'master_data': None,
        'modeling_data': None,
        'cleaning_report': None,
        'model': None,
        'model_results': None,
        'forecast_output': None,
        'planner_output': None,
        'planning_summary': None,
    }
    
    try:
        brand_mapping = preprocess.load_brand_mapping("sample_data/vendor_brand_mapping.csv")
        # =====================================================================
        # Step 1: Normalize and Validate Data
        # =====================================================================
        if verbose:
            print("🔄 Step 1: Normalizing and validating input files...")
        
        # Process inventory files
        inventory_list = []
        for df, parsed_date, filename in inventory_dfs:
            df_norm = column_mapper.map_columns(df)
            df_norm = preprocess.apply_brand_mapping(df_norm, brand_mapping)
            is_valid, errors = validators.validate_inventory_file(df_norm)
            if not is_valid:
                for error in errors:
                    results['errors'].append(f"{filename}: {error}")
            inventory_list.append((df_norm, parsed_date, filename))
        
        if not inventory_list:
            results['errors'].append("No valid inventory files to process")
            return results
        
        # Process sales files
        sales_list = []
        for df, parsed_date, filename in sales_dfs:
            df_norm = column_mapper.map_columns(df)
            df_norm = preprocess.apply_brand_mapping(df_norm, brand_mapping)
            is_valid, errors = validators.validate_sales_file(df_norm)
            if not is_valid:
                for error in errors:
                    results['errors'].append(f"{filename}: {error}")
            sales_list.append((df_norm, parsed_date, filename))
        
        if not sales_list:
            results['errors'].append("No valid sales files to process")
            return results
        
        # Process calendar if provided
        if calendar_df is not None:
            calendar_df = column_mapper.map_columns(calendar_df)
            calendar_df = preprocess.apply_brand_mapping(calendar_df, brand_mapping)
            is_valid, errors = validators.validate_event_calendar_file(calendar_df)
            if not is_valid:
                results['warnings'].append("Calendar validation issues - will proceed without event flags")
                calendar_df = None
            else:
                calendar_df = preprocess.clean_event_calendar(calendar_df)
        
        results['calendar_processed'] = calendar_df is not None
        
        # =====================================================================
        # Step 2: Merge and Clean Data
        # =====================================================================
        if verbose:
            print("🔄 Step 2: Merging and cleaning data...")
        
        # Merge inventory files with dates
        inv_dfs_with_dates = [(df, date) for df, date, _ in inventory_list]
        inv_filenames = [name for _, _, name in inventory_list]
        inventory_combined = io_utils.merge_dataframes_with_date_column(
            inv_dfs_with_dates, inv_filenames, snapshot_col_name='snapshot_date'
        )
        
        # Merge sales files with dates
        sales_dfs_with_dates = [(df, date) for df, date, _ in sales_list]
        sales_filenames = [name for _, _, name in sales_list]
        sales_combined = io_utils.merge_dataframes_with_date_column(
            sales_dfs_with_dates, sales_filenames, snapshot_col_name='snapshot_date'
        )
        
        # Aggregate sales to SKU-month
        sales_agg = preprocess.aggregate_sales_to_sku_month(sales_combined)
        negative_monthly_sales = int(sales_agg.attrs.get('negative_monthly_sales_corrected', 0))
        if negative_monthly_sales > 0:
            results['warnings'].append(
                f"Corrected {negative_monthly_sales} negative monthly sales values to zero."
            )
        
        # Get latest inventory snapshot
        inventory_latest = preprocess.prepare_inventory_latest_snapshot(inventory_combined)
        
        results['inventory_processed'] = inventory_combined.shape[0]
        results['sales_processed'] = sales_agg.shape[0]
        
        # =====================================================================
        # Step 3: Create Master Dataset
        # =====================================================================
        if verbose:
            print("🔄 Step 3: Creating master dataset...")
        
        master_data = preprocess.create_master_dataset(
            inventory_combined, sales_agg, calendar_df
        )
        
        # Handle missing values
        master_data = preprocess.handle_missing_values(master_data, strategy='forward_fill')
        results['master_data'] = master_data

        modeling_data, cleaning_report = preprocess.clean_for_modeling(master_data)
        results['modeling_data'] = modeling_data
        results['cleaning_report'] = cleaning_report

        if cleaning_report['duplicate_rows_removed'] > 0:
            results['warnings'].append(
                f"Pre-ML cleaning removed {cleaning_report['duplicate_rows_removed']} duplicate rows."
            )
        if cleaning_report['rows_missing_item_removed'] > 0:
            results['warnings'].append(
                f"Pre-ML cleaning removed {cleaning_report['rows_missing_item_removed']} rows with missing item numbers."
            )
        if cleaning_report['rows_bad_dates_removed'] > 0:
            results['warnings'].append(
                f"Pre-ML cleaning removed {cleaning_report['rows_bad_dates_removed']} rows with invalid dates."
            )
        if cleaning_report['rows_missing_keys_removed'] > 0:
            results['warnings'].append(
                f"Pre-ML cleaning removed {cleaning_report['rows_missing_keys_removed']} rows missing critical modeling fields."
            )
        if cleaning_report['rows_negative_qty_removed'] > 0:
            results['warnings'].append(
                f"Pre-ML cleaning removed {cleaning_report['rows_negative_qty_removed']} rows with negative quantity."
            )
        if cleaning_report['quantity_outliers_clipped'] > 0:
            results['warnings'].append(
                f"Pre-ML cleaning clipped {cleaning_report['quantity_outliers_clipped']} extreme quantity outliers."
            )
        
        # Check for minimum data
        unique_skus = modeling_data['item_no'].nunique()
        unique_months = modeling_data['snapshot_date'].nunique()
        
        if unique_skus < 5 or unique_months < 2:
            results['warnings'].append(
                f"Limited data: {unique_skus} SKUs, {unique_months} months. "
                "Reliance on fallback logic may be high."
            )
        
        # =====================================================================
        # Step 4: Feature Engineering
        # =====================================================================
        if verbose:
            print("🔄 Step 4: Engineering features...")
        
        featured_data, feature_cols = feature_engineering.create_full_feature_engineering_pipeline(
            modeling_data, target_col='quantity'
        )
        
        # Remove any rows with NaN target or critical features
        featured_data = featured_data.dropna(subset=['quantity'])
        featured_data = featured_data.fillna(0)
        
        if len(featured_data) < config.MIN_HISTORY_FOR_ML:
            results['warnings'].append(
                f"Insufficient featured data ({len(featured_data)} rows) for ML training. "
                "Will use fallback rules only."
            )
            use_ml = False
        else:
            use_ml = True
        
        # =====================================================================
        # Step 5: Train Model (if enough data)
        # =====================================================================
        model_results = None
        model_obj = None
        ml_predictor = None
        
        if use_ml:
            if verbose:
                print("🔄 Step 5: Training ML model...")
            
            try:
                model_obj, model_results = model_train.train_with_validation(
                    featured_data, feature_cols, target_col='quantity'
                )
                
                # Create ML predictor function
                def ml_predictor(row):
                    try:
                        feature_values = row.reindex(feature_cols, fill_value=0)
                        X = feature_values.to_numpy(dtype=np.float32).reshape(1, -1)
                        pred = model_train.make_predictions(model_obj, X)[0]
                        return pred, True
                    except:
                        return 0, False
                
                results['model'] = model_obj
                results['model_results'] = model_results
                
                if verbose:
                    print(f"✓ Model trained: {model_results['model_name']}")
                    print(f"  Train rows: {model_results['training_info']['training_rows']}")
                    print(f"  WAPE: {model_results['validation_metrics']['wape']:.1f}%")
            
            except Exception as e:
                results['warnings'].append(f"ML training failed: {str(e)}. Using fallback only.")
                use_ml = False
        
        # =====================================================================
        # Step 6: Apply Forecasting and Fallback Logic
        # =====================================================================
        if verbose:
            print("🔄 Step 6: Generating forecasts...")
        
        # Get latest unique SKU snapshot for forecasting
        skus_to_forecast = master_data[
            master_data['snapshot_date'] == master_data['snapshot_date'].max()
        ].drop_duplicates('item_no')

        if use_ml:
            latest_featured = featured_data[
                featured_data['snapshot_date'] == featured_data['snapshot_date'].max()
            ].drop_duplicates('item_no')

            feature_frame = latest_featured[['item_no'] + feature_cols].copy()
            skus_to_forecast = skus_to_forecast.merge(
                feature_frame,
                on='item_no',
                how='left',
                suffixes=('', '_feature')
            )
        
        forecast_df = fallback.create_fallback_forecast_dataframe(
            skus_to_forecast,
            modeling_data,
            ml_predictor=ml_predictor if use_ml else None
        )
        
        # Estimate M2 and M3
        forecast_df = fallback.estimate_forward_months(
            forecast_df,
            volatility_factor=1.0,
            history_df=modeling_data
        )
        
        # Apply event adjustments if calendar available
        if calendar_df is not None:
            forecast_df = fallback.add_event_adjustments(forecast_df, calendar_df)
        
        results['forecast_output'] = forecast_df
        
        # =====================================================================
        # Step 7: Create OTB Planning Output
        # =====================================================================
        if verbose:
                print("🔄 Step 7: Creating OTB planner output...")
        
        planner_output = planner.create_otp_planner_output(
            forecast_df, inventory_latest, use_simple_3m=use_simple_3m
        )
        
        planner_summary = planner.get_planning_summary(planner_output)
        
        results['planner_output'] = planner_output
        results['planning_summary'] = planner_summary
        results['feature_cols'] = feature_cols
        results['success'] = True

        persistence.save_run(results, feature_cols)

        if verbose:
            print("✓ Pipeline complete!")
            print(f"  Total active SKUs: {planner_summary['total_active_skus']}")
            print(f"  Total reorder qty: {planner_summary['total_reorder_qty']:.0f} units")
    
    except Exception as e:
        results['errors'].append(f"Pipeline error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return results
