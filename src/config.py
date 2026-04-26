"""
Configuration settings for the Company forecasting MVP.
Centralized place for constants, thresholds, and defaults.
"""

# ==============================================================================
# PLANNING PARAMETERS
# ==============================================================================
STOCK_COVER_MONTHS_HEALTHY_MIN = 2.0  # Minimum months of stock considered healthy
STOCK_COVER_MONTHS_HEALTHY_MAX = 3.0  # Maximum months of stock before overstock
REQUEST_LEAD_TIME_MONTHS = 3  # Lead time for reorder planning

# ==============================================================================
# ML MODEL PARAMETERS
# ==============================================================================
MIN_HISTORY_FOR_ML = 2  # Minimum monthly observations to use ML model
MIN_TEST_SIZE = 1  # Minimum test set size for time-aware validation
TEST_RATIO = 0.2  # Test set ratio from time-aware split
MIN_TEST_MONTHS = 2  # Minimum number of months in test period
LOW_VOLUME_THRESHOLD = 20  # Units/month below which we avoid ML over-forecasting

# Model selection (in order of preference)
MODEL_PREFERENCE = ['lightgbm', 'histgradientboosting', 'randomforest']

# LightGBM hyperparameters
LIGHTGBM_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'num_leaves': 31,
    'random_state': 42,
    'verbose': -1,
}

# HistGradientBoosting hyperparameters (fallback)
HGB_PARAMS = {
    'max_iter': 100,
    'max_depth': 3,
    'learning_rate': 0.1,
    'random_state': 42,
}

# RandomForest hyperparameters (second fallback)
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'random_state': 42,
    'n_jobs': -1,
}

# ==============================================================================
# FEATURE ENGINEERING PARAMETERS
# ==============================================================================
LAG_WINDOWS = [1, 2]  # Lag features to create
ROLLING_WINDOW = 2  # Rolling mean window
TREND_FACTOR_MIN = 0.7  # Min month-to-month trend multiplier
TREND_FACTOR_MAX = 1.3  # Max month-to-month trend multiplier

# ==============================================================================
# VALIDATION AND FILTERING
# ==============================================================================
# Column requirements for inventory files
INVENTORY_REQUIRED_COLS = [
    'item_no',
    'item_description',
    'total_stock',
    'item_status',
    'active',
]

# Column requirements for sales files
SALES_REQUIRED_COLS = [
    'item_no',
    'item_description',
    'quantity',
]

# Active SKU filter
ACTIVE_SKU_FILTER = 'Y'

# ==============================================================================
# FILE UPLOAD SETTINGS
# ==============================================================================
MAX_FILE_SIZE_MB = 200
ALLOWED_FILE_EXTENSIONS = ['.xls', '.xlsx', '.csv']

# ==============================================================================
# FALLBACK METHOD PRIORITIES
# ==============================================================================
FALLBACK_METHODS = [
    'ml_model',
    'fallback_recent_avg',
    'fallback_category_vendor',
    'fallback_category',
    'fallback_existing_forecast',
    'fallback_zero',
]

# ==============================================================================
# EVENT CALENDAR COLUMNS
# ==============================================================================
EVENT_CALENDAR_COLUMNS = [
    'date',
    'event_name',
    'event_type',
    'children_day',
    'christmas',
    'school_holiday',
    'year_end_holiday',
    'summer_holiday',
    'campaign_flag',
    'launch_flag',
]

# ==============================================================================
# CACHE AND STATE
# ==============================================================================
# Session state keys for Streamlit
STATE_INVENTORY_FILES = 'uploaded_inventory_files'
STATE_SALES_FILES = 'uploaded_sales_files'
STATE_CALENDAR_FILE = 'uploaded_calendar_file'
STATE_INVENTORY_DF = 'processed_inventory_df'
STATE_SALES_DF = 'processed_sales_df'
STATE_CALENDAR_DF = 'processed_calendar_df'
STATE_MASTER_DATA = 'master_data'
STATE_MODEL = 'fitted_model'
STATE_FEATURES = 'feature_list'
STATE_FORECAST_RESULTS = 'forecast_results'
STATE_MODEL_METRICS = 'model_metrics'
STATE_FEATURE_IMPORTANCE = 'feature_importance'

# RAG chat state keys
STATE_RAG_DOCUMENTS = 'rag_documents'
STATE_RAG_EMBEDDINGS = 'rag_embeddings'
STATE_RAG_CHAT_HISTORY = 'rag_chat_history'

# ==============================================================================
# METRICS THRESHOLDS AND DISPLAY
# ==============================================================================
# Warning thresholds for data quality
MIN_SALES_MONTHS_FOR_SEASONAL = 4
RECOMMENDED_SALES_MONTHS = 12

# ==============================================================================
# MODELING DATA CLEANING
# ==============================================================================
MODEL_MAX_NEGATIVE_QUANTITY = 0
MODEL_OUTLIER_IQR_MULTIPLIER = 3.0
MODEL_MIN_SKU_HISTORY = 2
