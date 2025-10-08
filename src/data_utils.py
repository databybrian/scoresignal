import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

# Keep only essential columns
current_season_int: int = 2526  
essential_columns = [
    'country', 
    'league_name',
    'league_code',    # from footballdata.co.uk (e.g. E0, SP1)
    'season',         # Season identifier (e.g. 23-24)
    'Date',           # Match date
    'Time',           # Match time (if available)
    'HomeTeam',       # standardize later
    'AwayTeam',       # standardize later
    'FTHG',           # Full Time Home Goals
    'FTAG',           # Full Time Away Goals  
    'FTR',            # Full Time Result (H/D/A)
    'HTHG',           # Half Time Home Goals 
    'HTAG',           # Half Time Away Goals 
    'B365H', 'B365D', 'B365A','PSH', 'PSD', 'PSA','WHH', 'WHD', 'WHA',  # Bet365 & Pinnacle and Willian Hill odds 
    'BbAvH', 'BbAvD', 'BbAvA', # Average Bookmaker Odds
    'B365CH', 'B365CD', 'B365CA','PSCH', 'PSCD', 'PSCA', # Closing odds 
    'B365>2.5', 'B365<2.5', 'P>2.5', 'P<2.5', # Over/Under 2.5 goals odds
]

def detect_date_format(date_series: pd.Series) -> str:
    """
    Intelligently detect whether dates are in DD/MM/YYYY or MM/DD/YYYY format.
    
    Strategy:
    1. Sample dates and look for values > 12 in first position (must be day)
    2. Check for consistency across the dataset
    3. Default to DD/MM/YYYY for European football data
    """
    sample_dates = date_series.dropna().head(100).astype(str)
    
    dd_mm_count = 0
    mm_dd_count = 0
    ambiguous_count = 0
    
    for date_str in sample_dates:
        # Clean the date string
        date_str = date_str.strip()
        
        # Try to parse the first two parts
        parts = date_str.replace('-', '/').split('/')
        if len(parts) < 2:
            continue
            
        try:
            first_num = int(parts[0])
            second_num = int(parts[1])
            
            # If first number > 12, must be DD/MM/YYYY
            if first_num > 12:
                dd_mm_count += 1
            # If second number > 12, must be MM/DD/YYYY
            elif second_num > 12:
                mm_dd_count += 1
            # Both <= 12, ambiguous
            else:
                ambiguous_count += 1
                
        except (ValueError, IndexError):
            continue
    
    # Decision logic
    if dd_mm_count > mm_dd_count:
        return 'DD/MM/YYYY'
    elif mm_dd_count > dd_mm_count:
        return 'MM/DD/YYYY'
    else:
        # Default to DD/MM/YYYY for European football data
        print("⚠️  Ambiguous date format detected. Defaulting to DD/MM/YYYY (European format)")
        return 'DD/MM/YYYY'


def parse_date_safe(date_str: str, format_type: str = 'DD/MM/YYYY') -> Optional[pd.Timestamp]:
    """
    Safely parse a single date string with known format.
    
    Args:
        date_str: Date string to parse
        format_type: Either 'DD/MM/YYYY' or 'MM/DD/YYYY'
    
    Returns:
        Parsed datetime or None if parsing fails
    """
    if pd.isna(date_str) or date_str == '':
        return None
    
    date_str = str(date_str).strip()
    
    # Define format strings to try
    if format_type == 'DD/MM/YYYY':
        formats = ['%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y', '%d/%m/%y', '%d-%m-%y']
    else:  # MM/DD/YYYY
        formats = ['%m/%d/%Y', '%m-%d-%Y', '%m.%d.%Y', '%m/%d/%y', '%m-%d-%y']
    
    # Also try ISO format which is unambiguous
    formats.extend(['%Y-%m-%d', '%Y/%m/%d'])
    
    for fmt in formats:
        try:
            return pd.Timestamp(datetime.strptime(date_str, fmt))
        except (ValueError, TypeError):
            continue
    
    return None


def format_date_column(df: pd.DataFrame, date_format: str = 'auto') -> pd.DataFrame:
    """
    Convert the date column to proper datetime format with intelligent format detection.
    
    Args:
        df: DataFrame with date column
        date_format: 'auto', 'DD/MM/YYYY', 'MM/DD/YYYY', or 'YYYY-MM-DD'
                    Default 'auto' will intelligently detect the format
    
    Returns:
        DataFrame with properly formatted Date column
    
    Examples:
        >>> df = format_date_column(df)  # Auto-detect format
        >>> df = format_date_column(df, date_format='DD/MM/YYYY')  # Explicit European format
        >>> df = format_date_column(df, date_format='YYYY-MM-DD')  # ISO format
    """
    # Safety check: ensure df is not None or empty
    if df is None or len(df) == 0:
        print("⚠️  Empty or None DataFrame provided")
        return df if df is not None else pd.DataFrame()
    
    # Safety check: ensure columns exist and are valid
    if not hasattr(df, 'columns') or len(df.columns) == 0:
        print("⚠️  DataFrame has no columns")
        return df
    
    # Standardize column name to 'Date' - SAFELY handle non-string column names
    try:
        # Convert all column names to strings first (handles numeric columns)
        df.columns = df.columns.astype(str)
        
        # Find date column case-insensitively
        date_col_candidates = [col for col in df.columns if col.lower() == 'date']
        
        if date_col_candidates:
            date_col = date_col_candidates[0]
            if date_col != 'Date':
                df = df.rename(columns={date_col: 'Date'})
        
    except Exception as e:
        print(f"⚠️  Error standardizing column names: {e}")
        # Continue anyway - maybe Date column already exists
    
    # Check if Date column exists
    if 'Date' not in df.columns:
        print("⚠️  No date column found in DataFrame")
        print(f"    Available columns: {list(df.columns)[:10]}")  # Show first 10 columns
        return df
    
    # Safety check: ensure Date column has valid data
    if df['Date'].isna().all():
        print("⚠️  Date column contains only null values")
        return df
    
    # Detect format if auto
    if date_format == 'auto':
        try:
            detected_format = detect_date_format(df['Date'])
            date_format = detected_format
        except Exception as e:
            print(f"⚠️  Format detection failed: {e}, defaulting to DD/MM/YYYY")
            date_format = 'DD/MM/YYYY'
    else:
        print(f"✓ Using specified date format: {date_format}")
    
    # Parse dates
    original_count = len(df)
    
    try:
        if date_format == 'YYYY-MM-DD':
            # ISO format - unambiguous, use pandas directly
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
        else:
            # Use our safe parser
            df['Date'] = df['Date'].apply(lambda x: parse_date_safe(x, date_format))
    except Exception as e:
        print(f"⚠️  Error parsing dates: {e}")
        # Try fallback parsing
        try:
            print("    Attempting fallback date parsing...")
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        except Exception as e2:
            print(f"❌ Fallback parsing also failed: {e2}")
            return df
    
    # Remove invalid dates
    df = df.dropna(subset=['Date'])
    removed_count = original_count - len(df)
    
    if removed_count > 0:
        print(f"⚠️  Removed {removed_count} rows with invalid dates ({removed_count/original_count*100:.1f}%)")
    
    # Validate date range (football data should be reasonable)
    if len(df) > 0:
        try:
            min_date = df['Date'].min()
            max_date = df['Date'].max()
            
            if min_date.year < 1990 or max_date.year > datetime.now().year + 1:
                print(f"⚠️  WARNING: Unusual date range detected: {min_date.date()} to {max_date.date()}")
                print(f"    This might indicate incorrect date parsing!")
            else:
                print(f"Date range validated: {min_date.date()} to {max_date.date()}")
        except Exception as e:
            print(f"⚠️  Could not validate date range: {e}")
    
    return df


# team_mapping_utils.py
def validate_team_mapping(fixtures_df: pd.DataFrame, mapping_df: pd.DataFrame):
    """
    Identify fixture teams that don't have mappings
    """
    # Get unique teams from fixtures
    fixture_teams = set(fixtures_df['home_team'].unique()) | set(fixtures_df['away_team'].unique())
    
    # Get mapped teams
    mapped_teams = set(mapping_df['fixture_name'].unique())  # Adjust column name as needed
    
    # Find unmapped teams
    unmapped_teams = fixture_teams - mapped_teams
    
    if unmapped_teams:
        print(f"⚠️  Found {len(unmapped_teams)} unmapped teams:")
        for team in sorted(unmapped_teams):
            print(f"  - '{team}'")
        
        # Suggest possible matches
        print("\n🔍 Suggested mappings (check for typos):")
        for unmapped in sorted(unmapped_teams):
            # Find similar team names in mapping
            similar = mapping_df[mapping_df['fixture_team'].str.contains(unmapped[:3], na=False, case=False)]
            if not similar.empty:
                print(f"  '{unmapped}' might match: {similar['fixture_team'].tolist()[:3]}")
    else:
        print("✅ All fixture teams have mappings!")
    
    return unmapped_teams