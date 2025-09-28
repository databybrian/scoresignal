# load_clean_historical_data.py

import pandas as pd
import numpy as np
from pathlib import Path

# Get project root (parent of current file's directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load dtype mapping using PROJECT_ROOT
DTYPE_FILE = PROJECT_ROOT / "raw_data" / "data_type_mapping.csv"
dtype_df = pd.read_csv(DTYPE_FILE)
DTYPE_MAPPING = dict(zip(dtype_df['column_name'], dtype_df['data_type']))

def clean_numeric_string(value):
    """Convert problematic strings to NaN for numeric columns"""
    if isinstance(value, str):
        value = value.strip()
        if value in ['', '#', 'NA', 'N/A', 'NULL', 'NaN', 'nan']:
            return np.nan
        cleaned = ''.join(char for char in value if char.isdigit() or char in '.-')
        return cleaned if cleaned else np.nan
    return value

# Identify numeric columns
numeric_columns = [col for col, dtype in DTYPE_MAPPING.items() 
                   if dtype in ['float32', 'float64', 'int8', 'int16', 'int32', 'int64']]
converters = {col: clean_numeric_string for col in numeric_columns}

def load_clean_historical_data(cache=True, file_path="cleaned_historical_data.csv"):
    """
    Efficiently load, clean, and cache historical football data.
    """
    CACHE_ATTR = '_cached_df'
    
    # Return cached data if available and caching is enabled
    if cache and hasattr(load_clean_historical_data, CACHE_ATTR):
        return getattr(load_clean_historical_data, CACHE_ATTR).copy()
    
    try:
        # Load data with optimized parameters
        df = pd.read_csv(
            file_path,
            low_memory=False,
            dtype=DTYPE_MAPPING,
            converters=converters,
            na_values=['', '#', 'NA', 'N/A', 'NULL', 'NaN', 'nan'],
            parse_dates=['Date']
        )
        
        # Sort by date
        df = df.sort_values('Date', ignore_index=True)
        
        # Cache the result
        if cache:
            setattr(load_clean_historical_data, CACHE_ATTR, df.copy())
        
        return df
        
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        raise

# Remove the automatic execution line - let callers decide when to load
# modelling_df = load_clean_historical_data()  # ← DELETED