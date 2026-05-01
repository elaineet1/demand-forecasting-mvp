"""
Disk persistence for model artifacts and forecast results.
Survives Streamlit session resets and can be read by external processes (e.g. Telegram bot).
"""

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

ARTIFACTS_DIR = Path("artifacts")
_MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
_FEATURES_PATH = ARTIFACTS_DIR / "feature_cols.pkl"
_PLANNER_PATH = ARTIFACTS_DIR / "planner_output.parquet"
_SUMMARY_PATH = ARTIFACTS_DIR / "planning_summary.json"
_METADATA_PATH = ARTIFACTS_DIR / "run_metadata.json"
_MODEL_META_PATH = ARTIFACTS_DIR / "model_meta.json"


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def save_run(forecast_results: Dict, feature_cols: List[str]) -> bool:
    """Persist model, feature list, planner output, and summary to disk."""
    try:
        ARTIFACTS_DIR.mkdir(exist_ok=True)

        model = forecast_results.get("model")
        if model is not None:
            joblib.dump(model, _MODEL_PATH)

        if feature_cols:
            joblib.dump(feature_cols, _FEATURES_PATH)

        planner_output = forecast_results.get("planner_output")
        if planner_output is not None and not planner_output.empty:
            planner_output.to_parquet(_PLANNER_PATH, index=False)

        summary = forecast_results.get("planning_summary", {})
        with open(_SUMMARY_PATH, "w") as f:
            json.dump(summary, f, default=_json_default, indent=2)

        # Save model metadata separately so RAG can describe model performance
        model_results = forecast_results.get("model_results")
        if model_results:
            model_meta = {
                "model_name": model_results.get("model_name", "unknown"),
                "validation_metrics": model_results.get("validation_metrics", {}),
                "training_info": model_results.get("training_info", {}),
            }
            with open(_MODEL_META_PATH, "w") as f:
                json.dump(model_meta, f, default=_json_default, indent=2)

        metadata = {
            "last_run": datetime.now().isoformat(),
            "total_skus": summary.get("total_active_skus", 0),
            "total_reorder_qty": float(summary.get("total_reorder_qty", 0)),
            "model_available": model is not None,
        }
        with open(_METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)

        return True
    except Exception as e:
        print(f"Warning: Could not save artifacts: {e}")
        return False


def load_run() -> Optional[Dict]:
    """
    Load persisted forecast results from disk.
    Returns None if no artifacts exist or loading fails.
    """
    if not _PLANNER_PATH.exists() or not _SUMMARY_PATH.exists():
        return None

    try:
        planner_output = pd.read_parquet(_PLANNER_PATH)

        with open(_SUMMARY_PATH) as f:
            planning_summary = json.load(f)

        model = joblib.load(_MODEL_PATH) if _MODEL_PATH.exists() else None
        feature_cols = joblib.load(_FEATURES_PATH) if _FEATURES_PATH.exists() else []

        metadata = {}
        if _METADATA_PATH.exists():
            with open(_METADATA_PATH) as f:
                metadata = json.load(f)

        model_results = None
        if _MODEL_META_PATH.exists():
            with open(_MODEL_META_PATH) as f:
                model_results = json.load(f)

        return {
            "success": True,
            "planner_output": planner_output,
            "planning_summary": planning_summary,
            "model": model,
            "feature_cols": feature_cols,
            "metadata": metadata,
            "model_results": model_results,
            "errors": [],
            "warnings": [],
            "master_data": None,
            "forecast_output": None,
        }
    except Exception as e:
        print(f"Warning: Could not load artifacts: {e}")
        return None


def has_saved_run() -> bool:
    return _PLANNER_PATH.exists() and _SUMMARY_PATH.exists()


def planner_mtime() -> float:
    """Return mtime of planner parquet — used as embedding cache key."""
    try:
        return _PLANNER_PATH.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def get_metadata() -> Dict:
    if _METADATA_PATH.exists():
        with open(_METADATA_PATH) as f:
            return json.load(f)
    return {}
