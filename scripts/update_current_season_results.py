# update_current_season_results.py

import pandas as pd
import sys
from pathlib import Path
import requests
from tqdm import tqdm
from io import StringIO

# Get project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from config (without SCRIPT_DIR)
from bot.config import current_season
from src.data_utils import format_date_column

# Configuration - use correct paths
LEAGUE_CONFIG_FILE = PROJECT_ROOT / "raw_data" / "footballdata_league_list.csv"
HISTORICAL_FILE = PROJECT_ROOT / "raw_data" / "combined_historical_data.csv"
CURRENT_SEASON = "2526"
BASE_URL = "https://www.football-data.co.uk"  # Fixed extra spaces

def load_league_config(filepath: str = None) -> pd.DataFrame:
    """Load league configuration"""
    if filepath is None:
        filepath = LEAGUE_CONFIG_FILE
    return pd.read_csv(filepath)

def download_current_season_data() -> pd.DataFrame:
    """
    Download current season data and update historical file efficiently.
    Returns: Updated DataFrame with all historical + current season data
    """
    print(f"🔄 Updating current season: {current_season}")
    
    # Load existing historical data
    historical_df = pd.DataFrame()
    if HISTORICAL_FILE.exists():
        historical_df = pd.read_csv(HISTORICAL_FILE, encoding='utf-8')
        print(f"📊 Loaded existing data: {len(historical_df):,} matches")
        
        # Remove existing current season data to avoid duplicates
        pre_filter_count = len(historical_df)
        historical_df = historical_df[historical_df['season'] != current_season]
        if pre_filter_count > len(historical_df):
            print(f"🗑️  Removed {pre_filter_count - len(historical_df):,} old {current_season} matches")
    
    # Load league config and filter valid leagues
    league_df = load_league_config()
    # Make sure you're using the correct column name from your CSV
    valid_leagues = league_df.dropna(subset=['odds_league_code'])[['country', 'league_name', 'odds_league_code']].drop_duplicates()
    
    print(f"⚽ Processing {len(valid_leagues)} leagues for season {current_season}")
    
    current_season_data = []
    required_columns = ['Date', 'HomeTeam', 'AwayTeam']
    
    for _, row in tqdm(valid_leagues.iterrows(), total=len(valid_leagues), desc="Downloading leagues"):
        country = row['country']
        league_code = row['odds_league_code']  # Updated column name
        league_name = row['league_name']
        
        url = f"{BASE_URL}/{current_season}/{league_code}.csv"
        
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 404:
                continue  # Season not available yet
            
            response.raise_for_status()
            
            # Read CSV and validate required columns
            df = pd.read_csv(StringIO(response.text))
            
            # Check if required columns exist
            if not all(col in df.columns for col in required_columns):
                print(f"⚠️  Missing columns in {league_code}, skipping")
                continue
            
            # Filter valid matches and add metadata
            df = df.dropna(subset=required_columns).copy()
            if df.empty:
                continue
                
            df['country'] = country
            df['league_name'] = league_name
            df['league_code'] = league_code
            df['season'] = current_season
            
            # Format date if function exists, otherwise keep original
            try:
                df = format_date_column(df)
            except (ImportError, AttributeError):
                pass  # Continue without date formatting if function not available
            
            current_season_data.append(df)
            print(f"✅ {league_code}: {len(df):,} matches")
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error for {league_code}: {e}")
            continue
        except Exception as e:
            print(f"❌ Processing error for {league_code}: {e}")
            continue
    
    # Combine and deduplicate data
    if current_season_data:
        current_df = pd.concat(current_season_data, ignore_index=True)
        print(f"✅ Downloaded {len(current_df):,} current season matches")
        
        # Ensure consistent columns before concatenation
        if not historical_df.empty:
            all_columns = set(historical_df.columns) | set(current_df.columns)
            for col in all_columns:
                if col not in historical_df.columns:
                    historical_df[col] = None
                if col not in current_df.columns:
                    current_df[col] = None
            
            # Reorder columns to match
            current_df = current_df[historical_df.columns]
        
        # Combine datasets
        updated_df = pd.concat([historical_df, current_df], ignore_index=True, sort=False)
        
        # Remove duplicates based on key match identifiers
        duplicate_cols = ['Date', 'HomeTeam', 'AwayTeam', 'season', 'league_code']
        pre_dedup_count = len(updated_df)
        updated_df = updated_df.drop_duplicates(subset=duplicate_cols, keep='last')
        
        if pre_dedup_count > len(updated_df):
            print(f"🧹 Removed {pre_dedup_count - len(updated_df):,} duplicates")
        
        # Save updated file
        updated_df.to_csv(HISTORICAL_FILE, index=False, encoding='utf-8')
        print(f"💾 Saved {len(updated_df):,} total matches to {HISTORICAL_FILE}")
        
        return updated_df
    else:
        print("⚠️  No current season data available yet")
        # Save historical data back (in case we removed old current season data)
        if not historical_df.empty:
            historical_df.to_csv(HISTORICAL_FILE, index=False, encoding='utf-8')
        return historical_df

if __name__ == "__main__":
    download_current_season_data()