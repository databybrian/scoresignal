# scripts/download_historical_data.py

import pandas as pd
import requests
from pathlib import Path
import sys
from io import StringIO

# Get project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import format_date_column, essential_columns
from bot.config import current_season  # Only import what you need

# Configuration
BASE_URL = "https://www.football-data.co.uk/mmz4281"  # Fixed extra spaces

# Use current_season from config
current_start_year = int(current_season[:2])  # 25
start_year = current_start_year - 15  # 25 - 15 = 10 (2010-11)
end_year = current_start_year         # 25 (2025-26)

SEASONS = [f"{year:02d}{(year+1):02d}" for year in range(start_year, end_year + 1)]

# Load league mapping from CSV - use correct path
LEAGUE_CONFIG_FILE = PROJECT_ROOT / "raw_data" / "footballdata_league_list.csv"
OUTPUT_FILE = PROJECT_ROOT / "combined_historical_data.csv"

def load_league_config(config_file: Path) -> pd.DataFrame:
    """Load league configuration from CSV"""
    if not config_file.exists():
        raise FileNotFoundError(f"League config file not found: {config_file}")
    
    df = pd.read_csv(config_file)
    required_cols = ['country', 'odds_league_code', 'league_name']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {config_file}: {missing_cols}")
    
    return df

def download_and_process_league_season(league_code: str, season: str, country: str, league_name: str) -> pd.DataFrame:
    """
    Download a single league-season CSV and add metadata
    Returns DataFrame with added country and league info, or empty DataFrame if failed
    """
    url = f"{BASE_URL}/{season}/{league_code}.csv"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 404:
            return pd.DataFrame()  # Return empty for missing seasons
        
        response.raise_for_status()
        
        # Read CSV directly from response content
        df = pd.read_csv(StringIO(response.text))
        
        # Add metadata columns
        df['country'] = country
        df['league_name'] = league_name
        df['league_code'] = league_code
        df['season'] = season
        
        # Keep only rows that have at least date and teams
        if 'Date' in df.columns and 'HomeTeam' in df.columns and 'AwayTeam' in df.columns:
            df = df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam'])
            return df
        else:
            print(f"⚠️  Warning: {league_code}_{season} missing required columns")
            return pd.DataFrame()
            
    except Exception as e:
        if "404" not in str(e):
            print(f"❌ Error downloading {league_code}_{season}: {e}")
        return pd.DataFrame()

def download_and_combine_all_historical_data():
    """Download all historical data and combine into single CSV"""
    # Load league config
    league_df = load_league_config(LEAGUE_CONFIG_FILE)
    
    all_data = []
    total_matches = 0
    
    for _, row in league_df.iterrows():
        country = row['country']
        league_code = row['odds_league_code']
        league_name = row['league_name']
        
        # Skip leagues with missing odds_league_code
        if pd.isna(league_code) or league_code == "":
            print(f"\n{'='*60}")
            print(f"⚠️  Skipping {country} - {league_name}: Missing odds_league_code")
            print(f"{'='*60}")
            continue
            
        print(f"\n{'='*60}")
        print(f"Processing {country} - {league_name} ({league_code})")
        print(f"{'='*60}")
        
        season_count = 0
        for season in SEASONS:
            print(f"  → {league_code}_{season}...", end=" ", flush=True)
            
            df = download_and_process_league_season(league_code, season, country, league_name)
            
            if not df.empty:
                all_data.append(df)
                season_count += 1
                total_matches += len(df)
                print(f"✅ ({len(df)} matches)")
            else:
                print("❌")
        
        print(f"    Total seasons processed: {season_count}")
    
    if not all_data:
        print("❌ No data was downloaded!")
        return
    
    # Combine all dataframes
    print(f"\n{'='*60}")
    print(f"🔄 Combining all data...")
    combined_df = pd.concat(all_data, ignore_index=True)
    try:
        combined_df = format_date_column(combined_df)
    except (ImportError, AttributeError):
        pass  # Continue without date formatting if function not available
    
    # Reorder columns to put metadata first and keep only essential columns
    available_columns = [col for col in essential_columns if col in combined_df.columns]
    combined_df = combined_df[available_columns]
    
    metadata_cols = ['country', 'league_name', 'league_code', 'season', 'Date', 'HomeTeam', 'AwayTeam']
    other_cols = [col for col in combined_df.columns if col not in metadata_cols]
    combined_df = combined_df[metadata_cols + other_cols]
    
    # Save to CSV
    combined_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    
    print(f"✅ COMBINED DATA SAVED")
    print(f"📁 File: {OUTPUT_FILE.absolute()}")
    print(f"📊 Total matches: {len(combined_df):,}")
    print(f"🌍 Countries: {combined_df['country'].nunique()}")
    print(f"🏆 Leagues: {combined_df['league_name'].nunique()}")
    print(f"📅 Seasons: {combined_df['season'].nunique()}")
    print(f"{'='*60}")

if __name__ == "__main__":
    download_and_combine_all_historical_data()