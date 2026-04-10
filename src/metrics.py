"""
Metrics calculation for model evaluation and performance tracking.
Includes MAE, RMSE, WAPE, MAPE.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAE score
    """
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        RMSE score
    """
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)


def weighted_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Weighted Absolute Percentage Error (WAPE).
    More robust than MAPE as it's not sensitive to individual large percentage errors.
    
    WAPE = sum(|y_true - y_pred|) / sum(|y_true|)
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        WAPE as percentage (0-100)
    """
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true))
    
    if denominator == 0:
        warnings.warn("WAPE denominator is zero")
        return 0
    
    return (numerator / denominator) * 100


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    WARNING: Can be misleading for small values. Use WAPE instead.
    
    Returns None if many zero values exist.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAPE as percentage or None if undefined
    """
    # Check for zero or near-zero values
    if np.sum(np.abs(y_true) < 0.01) > len(y_true) * 0.1:
        warnings.warn("MAPE unreliable: >10% of actuals near zero")
        return None
    
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return None
    
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape


def mean_absolute_scaled_error(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_train: np.ndarray = None) -> float:
    """
    Calculate Mean Absolute Scaled Error (MASE).
    Scales error by the moving average error of a baseline model.
    
    Args:
        y_true: True test values
        y_pred: Predicted values
        y_train: Training values (for baseline calculation)
    
    Returns:
        MASE score
    """
    mae = mean_absolute_error(y_true, y_pred)
    
    if y_train is None or len(y_train) < 2:
        # Use a naive forecast baseline (previous value)
        naive_forecast_error = np.mean(np.abs(np.diff(y_true)))
    else:
        # Use training data baseline
        naive_forecast_error = np.mean(np.abs(np.diff(y_train)))
    
    if naive_forecast_error == 0:
        return 0
    
    return mae / naive_forecast_error


def calculate_all_metrics(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         y_train: np.ndarray = None) -> dict:
    """
    Calculate all available metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_train: Training values (optional, for MASE)
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'wape': weighted_absolute_percentage_error(y_true, y_pred),
    }
    
    mape = mean_absolute_percentage_error(y_true, y_pred)
    if mape is not None:
        metrics['mape'] = mape
    
    if y_train is not None:
        metrics['mase'] = mean_absolute_scaled_error(y_true, y_pred, y_train)
    
    return metrics


def forecast_accuracy_interpretation(wape: float) -> str:
    """
    Provide qualitative interpretation of forecast accuracy.
    
    Args:
        wape: WAPE percentage
    
    Returns:
        Interpretation string
    """
    if wape < 10:
        return "Excellent (< 10%)"
    elif wape < 20:
        return "Good (10-20%)"
    elif wape < 50:
        return "Acceptable (20-50%)"
    else:
        return "Poor (> 50%)"
