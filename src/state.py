"""
Session state management for Streamlit app.
Handles persistent data across app reruns using st.session_state.
"""

import streamlit as st
from src import config


def initialize_session_state():
    """Initialize all required session state variables if they don't exist."""
    
    # File uploads tracking
    if config.STATE_INVENTORY_FILES not in st.session_state:
        st.session_state[config.STATE_INVENTORY_FILES] = []
    
    if config.STATE_SALES_FILES not in st.session_state:
        st.session_state[config.STATE_SALES_FILES] = []
    
    if config.STATE_CALENDAR_FILE not in st.session_state:
        st.session_state[config.STATE_CALENDAR_FILE] = None
    
    # Processed data
    if config.STATE_INVENTORY_DF not in st.session_state:
        st.session_state[config.STATE_INVENTORY_DF] = None
    
    if config.STATE_SALES_DF not in st.session_state:
        st.session_state[config.STATE_SALES_DF] = None
    
    if config.STATE_CALENDAR_DF not in st.session_state:
        st.session_state[config.STATE_CALENDAR_DF] = None
    
    # Master data and modeling artifacts
    if config.STATE_MASTER_DATA not in st.session_state:
        st.session_state[config.STATE_MASTER_DATA] = None
    
    if config.STATE_MODEL not in st.session_state:
        st.session_state[config.STATE_MODEL] = None
    
    if config.STATE_FEATURES not in st.session_state:
        st.session_state[config.STATE_FEATURES] = None
    
    if config.STATE_FORECAST_RESULTS not in st.session_state:
        st.session_state[config.STATE_FORECAST_RESULTS] = None
    
    if config.STATE_MODEL_METRICS not in st.session_state:
        st.session_state[config.STATE_MODEL_METRICS] = None
    
    if config.STATE_FEATURE_IMPORTANCE not in st.session_state:
        st.session_state[config.STATE_FEATURE_IMPORTANCE] = None


def clear_all_state():
    """Clear all session state - useful for starting fresh."""
    keys = [
        config.STATE_INVENTORY_FILES,
        config.STATE_SALES_FILES,
        config.STATE_CALENDAR_FILE,
        config.STATE_INVENTORY_DF,
        config.STATE_SALES_DF,
        config.STATE_CALENDAR_DF,
        config.STATE_MASTER_DATA,
        config.STATE_MODEL,
        config.STATE_FEATURES,
        config.STATE_FORECAST_RESULTS,
        config.STATE_MODEL_METRICS,
        config.STATE_FEATURE_IMPORTANCE,
    ]
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]


def clear_model_state():
    """Clear model-related state (but keep input data)."""
    keys = [
        config.STATE_MASTER_DATA,
        config.STATE_MODEL,
        config.STATE_FEATURES,
        config.STATE_FORECAST_RESULTS,
        config.STATE_MODEL_METRICS,
        config.STATE_FEATURE_IMPORTANCE,
    ]
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]
