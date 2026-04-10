"""
Machine learning model training for sales quantity forecasting.
Implements time-aware validation, ensemble methods, and fallback model selection.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple, Optional, Dict, List
from src import config
import warnings

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def select_model_class(prefer_ensemble: bool = True):
    """
    Select the best available model class based on installed packages.
    
    Args:
        prefer_ensemble: If True, try ensemble methods first
    
    Returns:
        (model_class, model_name) tuple
    """
    if prefer_ensemble:
        # Try ensemble of available models
        available_models = []
        
        if HAS_LIGHTGBM:
            available_models.append(('lgbm', LGBMRegressor(**config.LIGHTGBM_PARAMS)))
        
        if HAS_XGBOOST:
            available_models.append(('xgb', XGBRegressor(**config.XGB_PARAMS)))
        
        try:
            available_models.append(('hgb', HistGradientBoostingRegressor(**config.HGB_PARAMS)))
        except:
            pass
        
        if len(available_models) >= 2:
            return VotingRegressor(estimators=available_models), 'ensemble'
        elif len(available_models) == 1:
            return available_models[0][1], available_models[0][0]
    
    # Single model fallback
    if HAS_LIGHTGBM:
        return LGBMRegressor, 'lightgbm'
    
    if HAS_XGBOOST:
        return XGBRegressor, 'xgboost'
    
    try:
        return HistGradientBoostingRegressor, 'histgradientboosting'
    except:
        pass
    
    try:
        return RandomForestRegressor, 'randomforest'
    except:
        raise RuntimeError("No suitable model library available. Install LightGBM, XGBoost, or scikit-learn.")


def prepare_training_data(df: pd.DataFrame,
                         feature_cols: List[str],
                         target_col: str = 'quantity',
                         min_rows: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare X and y arrays for model training.
    
    Args:
        df: Featured DataFrame
        feature_cols: List of feature column names
        target_col: Target column name
        min_rows: Minimum rows required
    
    Returns:
        Tuple of (X, y) as numpy arrays
    """
    if len(df) < min_rows:
        raise ValueError(f"Insufficient training data: {len(df)} rows < {min_rows} minimum")
    
    # Select features and target
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Ensure numeric
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    
    return X, y


def time_series_split_data(df: pd.DataFrame,
                           feature_cols: List[str],
                           target_col: str = 'quantity',
                           test_ratio: float = None) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                                               Tuple[np.ndarray, np.ndarray],
                                                               pd.DataFrame]:
    """
    Split data using time-aware validation (no random shuffle).
    Later observations go to test set.
    
    Args:
        df: Featured DataFrame (must be sorted by time)
        feature_cols: List of feature column names
        target_col: Target column name
        test_ratio: Ratio of test set (default from config)
    
    Returns:
        Tuple of ((X_train, y_train), (X_test, y_test), test_indices_df)
    """
    if test_ratio is None:
        test_ratio = config.TEST_RATIO
    
    df = df.copy()
    if 'snapshot_date' not in df.columns:
        raise ValueError("Featured data must include 'snapshot_date' for time-based validation")

    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'], errors='coerce')
    df = df.dropna(subset=['snapshot_date']).sort_values(['snapshot_date', 'item_no']).reset_index(drop=True)

    unique_dates = df['snapshot_date'].sort_values().drop_duplicates().tolist()
    if len(unique_dates) < 2:
        raise ValueError("Need at least 2 unique dates for time-based validation")

    test_date_count = max(int(len(unique_dates) * test_ratio), config.MIN_TEST_MONTHS)
    test_date_count = min(test_date_count, len(unique_dates) - 1)
    test_dates = set(unique_dates[-test_date_count:])

    train_df = df[~df['snapshot_date'].isin(test_dates)].copy()
    test_df = df[df['snapshot_date'].isin(test_dates)].copy()

    if len(train_df) < config.MIN_HISTORY_FOR_ML:
        raise ValueError(
            f"Insufficient training data after split: {len(train_df)} rows "
            f"< {config.MIN_HISTORY_FOR_ML} minimum"
        )
    if len(test_df) < config.MIN_TEST_SIZE:
        raise ValueError(
            f"Insufficient test data after split: {len(test_df)} rows "
            f"< {config.MIN_TEST_SIZE} minimum"
        )
    
    X_train, y_train = prepare_training_data(train_df, feature_cols, target_col)
    X_test, y_test = prepare_training_data(test_df, feature_cols, target_col)
    
    return (X_train, y_train), (X_test, y_test), test_df


def train_model(X: np.ndarray,
               y: np.ndarray,
               model_class=None,
               model_params: Dict = None) -> Tuple:
    """
    Train a single model.
    
    Args:
        X: Feature array
        y: Target array
        model_class: Model class to use (default selected automatically)
        model_params: Model parameters (default from config)
    
    Returns:
        (model, model_name) tuple
    """
    if model_class is None:
        model_class, _ = select_model_class(prefer_ensemble=False)
    
    # Get default parameters based on model type
    model_name = getattr(model_class, '__name__', model_class.__class__.__name__)
    
    if model_params is None and isinstance(model_class, type):
        if model_name == 'LGBMRegressor':
            model_params = config.LIGHTGBM_PARAMS.copy()
        elif model_name == 'HistGradientBoostingRegressor':
            model_params = config.HGB_PARAMS.copy()
        elif model_name == 'RandomForestRegressor':
            model_params = config.RF_PARAMS.copy()
        else:
            model_params = {}
    
    # Create and train model
    if isinstance(model_class, type):
        model = model_class(**(model_params or {}))
    else:
        model = model_class

    model.fit(X, y)
    
    return model, model_name.replace('Regressor', '').lower()


def get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
    
    Returns:
        DataFrame with feature importance
    """
    try:
        # Try attribute-based importance first (LightGBM, RandomForest)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'feature_importance_'):
            importances = model.feature_importance_()
        else:
            # Fallback for models without direct importances
            importances = np.ones(len(feature_names)) / len(feature_names)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    except Exception as e:
        warnings.warn(f"Could not extract feature importance: {str(e)}")
        return pd.DataFrame({
            'feature': feature_names,
            'importance': np.ones(len(feature_names)) / len(feature_names)
        })


def make_predictions(model,
                    X: np.ndarray,
                    ensure_positive: bool = True) -> np.ndarray:
    """
    Make predictions and ensure reasonable values.
    
    Args:
        model: Trained model
        X: Feature array
        ensure_positive: If True, clip negative predictions to 0
    
    Returns:
        Prediction array
    """
    predictions = model.predict(X)
    
    if ensure_positive:
        predictions = np.maximum(predictions, 0)
    
    return predictions


def train_full_model(df: pd.DataFrame,
                    feature_cols: List[str],
                    target_col: str = 'quantity') -> Tuple[object, Dict]:
    """
    Train a model on full dataset (no holdout).
    Used for final production model after validation.
    
    Args:
        df: Featured DataFrame
        feature_cols: List of feature column names
        target_col: Target column name
    
    Returns:
        (model, training_info_dict)
    """
    X, y = prepare_training_data(df, feature_cols, target_col, min_rows=config.MIN_HISTORY_FOR_ML)
    
    model, model_name = train_model(X, y)
    
    info = {
        'model_type': model_name,
        'training_rows': len(df),
        'features_used': len(feature_cols),
        'mean_target': y.mean(),
        'std_target': y.std(),
    }
    
    return model, info


def validate_model(model,
                  X_test: np.ndarray,
                  y_test: np.ndarray) -> Dict:
    """
    Validate model on test set and compute metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
    
    Returns:
        Dictionary with validation metrics
    """
    from src import metrics
    
    y_pred = make_predictions(model, X_test)
    
    mae = metrics.mean_absolute_error(y_test, y_pred)
    rmse = metrics.root_mean_squared_error(y_test, y_pred)
    wape = metrics.weighted_absolute_percentage_error(y_test, y_pred)
    mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'wape': wape,
        'mape': mape,
        'n_test_samples': len(y_test),
        'y_pred_mean': y_pred.mean(),
        'y_test_mean': y_test.mean(),
    }


def train_with_validation(df: pd.DataFrame,
                         feature_cols: List[str],
                         target_col: str = 'quantity') -> Tuple[object, Dict]:
    """
    Train model with time-aware validation.
    
    Args:
        df: Featured DataFrame
        feature_cols: List of feature column names
        target_col: Target column name
    
    Returns:
        (model_trained_on_full_data, validation_results_dict)
    """
    # Split data
    (X_train, y_train), (X_test, y_test), test_df = time_series_split_data(
        df, feature_cols, target_col
    )
    
    # Train on training set
    train_df = df.iloc[:-len(test_df)]
    model_val, model_name = train_model(X_train, y_train)
    
    # Validate
    val_metrics = validate_model(model_val, X_test, y_test)

    # Add test date context if available
    if 'snapshot_date' in test_df.columns:
        test_dates = pd.to_datetime(test_df['snapshot_date'], errors='coerce').dropna()
        if not test_dates.empty:
            val_metrics['test_date_start'] = test_dates.min().date().isoformat()
            val_metrics['test_date_end'] = test_dates.max().date().isoformat()
            val_metrics['test_unique_months'] = test_dates.dt.to_period('M').nunique()
    
    # Train final model on all data
    final_model, final_info = train_full_model(df, feature_cols, target_col)
    
    # Combine info
    results = {
        'model': final_model,
        'model_name': model_name,
        'validation_metrics': val_metrics,
        'training_info': final_info,
        'feature_importance': get_feature_importance(final_model, feature_cols),
    }
    
    return final_model, results
