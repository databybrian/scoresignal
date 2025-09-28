# scripts/process_historical_data.py

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Get project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import format_date_column

# Load dtype mapping - use PROJECT_ROOT instead of SCRIPT_DIR
DTYPE_FILE = PROJECT_ROOT / "data_type_mapping.csv"
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

def process_raw_to_cleaned():
    """
    Process raw combined_historical_data.csv into cleaned_historical_data.csv
    """
    RAW_FILE = PROJECT_ROOT / "combined_historical_data.csv"
    CLEANED_FILE = PROJECT_ROOT / "data" / "cleaned_historical_data.csv"
    
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw historical data not found: {RAW_FILE}")
    
    print(f"📥 Loading raw historical data from {RAW_FILE}")
    df = pd.read_csv(
        RAW_FILE,
        low_memory=False,
        dtype=DTYPE_MAPPING,
        converters=converters,
        na_values=['', '#', 'NA', 'N/A', 'NULL', 'NaN', 'nan']
    )
    
    print(f"📊 Loaded {len(df)} raw matches")
    
    # Apply date formatting
    try:
        df = format_date_column(df)
    except Exception as e:
        print(f"⚠️  Date formatting warning: {e}")
    
    # Sort by date
    df = df.sort_values('Date', ignore_index=True)
    
    # Ensure data directory exists
    CLEANED_FILE.parent.mkdir(exist_ok=True)
    
    # Save cleaned version
    df.to_csv(CLEANED_FILE, index=False, encoding='utf-8')
    print(f"✅ Saved cleaned historical data to {CLEANED_FILE}")
    print(f"📊 Final cleaned matches: {len(df)}")
    
    return df

if __name__ == "__main__":
    process_raw_to_cleaned()