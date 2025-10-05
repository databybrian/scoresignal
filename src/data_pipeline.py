# src/data_pipeline.py - FIXED VERSION
"""
Shared data pipeline functions used by both main.py (worker) and Streamlit (dashboard).
No side effects: no Telegram, no locks, no print-to-stdout (minimal).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_FILE = PROJECT_ROOT / "combined_historical_data.csv"
CLEANED_FILE = DATA_DIR / "cleaned_historical_data.csv"

# Ensure data dir exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -------------------
# Helpers
# -------------------
def get_current_season() -> int:
    """
    Return current season in football-data.co.uk format as integer.
    Examples:
      - Aug 2024 – May 2025 → 2425
      - June 2025 → still 2425 (season hasn't changed)
      - Aug 2025 → 2526
    """
    now = datetime.now()
    year = now.year
    month = now.month

    # European season starts in August
    if month < 8:
        # Before August: current season is (year-1) to (year)
        start_year = year - 1
    else:
        # August or later: current season is (year) to (year+1)
        start_year = year

    end_year = start_year + 1

    # Format as YYZZ → e.g., 2425
    season_str = f"{start_year % 100:02d}{end_year % 100:02d}"
    return int(season_str)

def file_age_days(path: Path) -> float:
    if not path.exists():
        return float("inf")
    return (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds() / 86400.0

def needs_refresh(path: Path, days: int = 7) -> bool:
    return not path.exists() or (file_age_days(path) >= days)

def clean_numeric_string(value):
    if isinstance(value, str):
        value = value.strip()
        if value in ['', '#', 'NA', 'N/A', 'NULL', 'NaN', 'nan']:
            return np.nan
        cleaned = ''.join(char for char in value if char.isdigit() or char in '.-')
        return cleaned if cleaned else np.nan
    return value

# -------------------
# Core Functions
# -------------------
def process_raw_to_cleaned():
    """Process raw combined_historical_data.csv into cleaned_historical_data.csv"""
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw historical data not found: {RAW_FILE}")

    DTYPE_FILE = PROJECT_ROOT / "raw_data" / "data_type_mapping.csv"
    if not DTYPE_FILE.exists():
        raise FileNotFoundError(f"Data type mapping not found: {DTYPE_FILE}")

    dtype_df = pd.read_csv(DTYPE_FILE)
    DTYPE_MAPPING = dict(zip(dtype_df['column_name'], dtype_df['data_type']))

    numeric_columns = [col for col, dtype in DTYPE_MAPPING.items()
                       if dtype in ['float32', 'float64', 'int8', 'int16', 'int32', 'int64']]
    converters = {col: clean_numeric_string for col in numeric_columns}

    df = pd.read_csv(
        RAW_FILE,
        low_memory=False,
        dtype=DTYPE_MAPPING,
        converters=converters,
        na_values=['', '#', 'NA', 'N/A', 'NULL', 'NaN', 'nan']
    )

    # Apply date formatting if available
    try:
        from src.data_utils import format_date_column
        df = format_date_column(df)
    except Exception:
        pass  # Fallback: keep as-is

    df = df.sort_values('Date', ignore_index=True)
    df.to_csv(CLEANED_FILE, index=False, encoding='utf-8')
    return df

def ensure_historical_data_exists(days: int = 7):
    """Ensure cleaned historical data exists and is fresh (within `days`)."""
    if CLEANED_FILE.exists() and not needs_refresh(CLEANED_FILE, days=days):
        return True

    from scripts.download_historical_data import download_and_combine_all_historical_data
    download_and_combine_all_historical_data()
    process_raw_to_cleaned()
    return True

# -------------------
# League Table Builder (from table_builder.py) - FIXED
# -------------------
def build_league_table(
    df: pd.DataFrame,
    league_code: str,
    season: int = None,
    as_of_date: Optional[str] = None,
    save_current: bool = False
) -> pd.DataFrame:
    """
    Build league table for a specific league and season up to a given date.
    
    FIX: If no data exists for the specified season, fall back to the most recent season.
    """
    if season is None:
        season = get_current_season()

    # Ensure Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Try the specified season first
    league_df = df[
        (df['league_code'] == league_code) &
        (df['season'] == season)
    ].copy()

    # If no data for current season, fall back to most recent available
    if league_df.empty:
        # Get all data for this league
        league_all = df[df['league_code'] == league_code].copy()
        
        if league_all.empty:
            return pd.DataFrame(columns=['Team','P','W','D','L','GF','GA','GD','Pts','Form'])
        
        # Find most recent season with data
        available_seasons = league_all['season'].dropna().unique()
        if len(available_seasons) == 0:
            return pd.DataFrame(columns=['Team','P','W','D','L','GF','GA','GD','Pts','Form'])
        
        most_recent_season = max(available_seasons)
        league_df = league_all[league_all['season'] == most_recent_season].copy()
        
        # Log fallback (but don't print to avoid Railway logs)
        # You could add logging here if needed

    # Handle as_of_date
    if as_of_date:
        as_of_date = pd.to_datetime(as_of_date)
    else:
        as_of_date = league_df['Date'].max()
    league_df = league_df[league_df['Date'] <= as_of_date]

    # Remove matches without full-time results
    league_df = league_df.dropna(subset=['FTHG','FTAG'])
    if league_df.empty:
        return pd.DataFrame(columns=['Team','P','W','D','L','GF','GA','GD','Pts','Form'])

    teams = set(league_df['HomeTeam']).union(league_df['AwayTeam'])
    table = []

    for team in teams:
        # Home
        home = league_df[league_df['HomeTeam'] == team]
        home_w = (home['FTHG'] > home['FTAG']).sum()
        home_d = (home['FTHG'] == home['FTAG']).sum()
        home_l = (home['FTHG'] < home['FTAG']).sum()
        home_gf, home_ga = home['FTHG'].sum(), home['FTAG'].sum()

        # Away
        away = league_df[league_df['AwayTeam'] == team]
        away_w = (away['FTAG'] > away['FTHG']).sum()
        away_d = (away['FTAG'] == away['FTHG']).sum()
        away_l = (away['FTAG'] < away['FTHG']).sum()
        away_gf, away_ga = away['FTAG'].sum(), away['FTHG'].sum()

        # Totals
        w, d, l = home_w+away_w, home_d+away_d, home_l+away_l
        gf, ga = home_gf+away_gf, home_ga+away_ga
        pts, gd = w*3 + d, gf - ga

        # Last 5 form
        recent = pd.concat([home, away]).sort_values('Date', ascending=False).head(5)
        form_pts = sum(
            3 if (g['HomeTeam'] == team and g['FTHG'] > g['FTAG']) or
                 (g['AwayTeam'] == team and g['FTAG'] > g['FTHG'])
            else 1 if g['FTHG'] == g['FTAG']
            else 0
            for _, g in recent.iterrows()
        )

        table.append({
            'Team': team, 'P': w+d+l, 'W': w, 'D': d, 'L': l,
            'GF': gf, 'GA': ga, 'GD': gd, 'Pts': pts, 'Form': form_pts
        })

    # Sort and index
    table_df = pd.DataFrame(table).sort_values(['Pts','GD','GF'], ascending=False).reset_index(drop=True)
    table_df.index = table_df.index + 1
    table_df.index.name = 'Pos'
    return table_df

# -------------------
# Save All Current Tables - FIXED
# -------------------
def save_all_current_tables():
    """
    Generate and save current_season_leagues_table.csv from cleaned historical data.
    
    FIX: Better error handling and debugging information.
    """
    if not CLEANED_FILE.exists():
        raise FileNotFoundError(f"Cleaned historical data not found: {CLEANED_FILE}")

    #  Add validation
    df = pd.read_csv(CLEANED_FILE, parse_dates=['Date'])
    
    if df.empty:
        raise RuntimeError("Cleaned historical data file is empty")
    
    # Check required columns
    required_cols = ['country', 'league_code', 'league_name', 'season', 'Date', 
                     'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise RuntimeError(f"Missing required columns in historical data: {missing_cols}")

    # Get unique leagues
    league_info = (
        df[['country', 'league_code', 'league_name']]
        .dropna(subset=['country', 'league_code', 'league_name'])
        .drop_duplicates()
    )
    
    if league_info.empty:
        raise RuntimeError("No league information found in historical data")

    season = get_current_season()
    all_tables = []
    
    # Track failures for debugging
    failed_leagues = []

    for _, row in league_info.iterrows():
        league_code = row['league_code']
        league_name = row['league_name']
        country = row['country']

        try:
            table_df = build_league_table(df, league_code, season)

            if not table_df.empty:
                table_df = table_df.copy()
                table_df['country'] = country
                table_df['league_code'] = league_code
                table_df['league_name'] = league_name
                table_df = table_df.reset_index()  # exposes 'Pos'

                cols = ['country', 'league_code', 'league_name', 'Pos', 'Team', 'P', 'W', 'D',
                        'L', 'GF', 'GA', 'GD', 'Pts', 'Form']
                cols = [c for c in cols if c in table_df.columns]
                table_df = table_df[cols]
                all_tables.append(table_df)
            else:
                failed_leagues.append(f"{league_name} ({league_code})")
        except Exception as e:
            failed_leagues.append(f"{league_name} ({league_code}): {str(e)}")

    if all_tables:
        master_table = pd.concat(all_tables, ignore_index=True)
        master_table = master_table.sort_values(['league_name', 'Pos']).reset_index(drop=True)
        output_path = DATA_DIR / "current_season_leagues_table.csv"
        master_table.to_csv(output_path, index=False)
        
        # Return success info for debugging
        return {
            'success': True,
            'leagues_generated': len(all_tables),
            'failed_leagues': failed_leagues,
            'output_path': str(output_path)
        }
    else:
        # Provide detailed error message
        error_msg = f"No league tables were generated. Failed leagues: {', '.join(failed_leagues) if failed_leagues else 'None found'}"
        raise RuntimeError(error_msg)