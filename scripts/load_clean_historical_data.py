import pandas as pd
import numpy as np
import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from bot.config import SCRIPT_DIR

DTYPE_FILE = SCRIPT_DIR / "data_type_mapping.csv"
dtype_df = pd.read_csv(DTYPE_FILE)
DTYPE_MAPPING = dict(zip(dtype_df['column_name'], dtype_df['data_type']))
# Load data with the mapping

def clean_numeric_string(value):
    """Convert problematic strings to NaN for numeric columns"""
    if isinstance(value, str):
        # Remove common non-numeric characters and handle empty values
        value = value.strip()
        if value in ['', '#', 'NA', 'N/A', 'NULL', 'NaN', 'nan']:
            return np.nan
        # Remove any remaining non-numeric characters except decimal point and negative sign
        cleaned = ''.join(char for char in value if char.isdigit() or char in '.-')
        return cleaned if cleaned else np.nan
    return value

# Identify which columns are numeric in your DTYPE_MAPPING
numeric_columns = [col for col, dtype in DTYPE_MAPPING.items() 
                   if dtype in ['float32', 'float64', 'int8', 'int16', 'int32', 'int64']]

# Create converters dictionary for numeric columns only
converters = {col: clean_numeric_string for col in numeric_columns}

# data_loader.py
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
    
    # Final processing
    df = df.sort_values('Date', ignore_index=True)
    
    # Cache with copy to avoid external modifications affecting cache
    if cache:
        setattr(load_clean_historical_data, cache_attr, df.copy())
    
    return df

modelling_df = load_clean_historical_data()
