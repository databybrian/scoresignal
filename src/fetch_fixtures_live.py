# src/fetch_fixtures_live.py

import requests
import json
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path
import warnings
from typing import List, Tuple

# Handle imports gracefully with fallbacks
try:
    from src.data_utils import format_date_column
    from src.league_data import LEAGUE_PATHS 
except ImportError:
    # Try relative imports
    try:
        from .data_utils import format_date_column
        from .league_data import LEAGUE_PATHS
    except ImportError:
        # Fallback definitions
        print("⚠️  src modules not found, using fallbacks")
        
        def format_date_column(df):
            """Fallback date formatting function"""
            if 'date' in df.columns:
                try:
                    df['date'] = pd.to_datetime(df['date'])
                except:
                    pass
            return df
        
        # Minimal league paths for testing
        LEAGUE_PATHS = {
            "premier-league": {
                "path": "2025-26/en.1.json", 
                "name": "Premier League"
            },
            "championship": {
                "path": "2025-26/en.2.json",
                "name": "Championship" 
            }
        }

last_tier = None

def fetch_fixtures(league_key: str, season: str = "2025-26") -> pd.DataFrame:
    """
    Fetch fixtures for one league using openfootball.
    """
    if league_key not in LEAGUE_PATHS:
        print(f"⚠️  {league_key} not found in LEAGUE_PATHS, using fallback")
        # Try a fallback URL
        url = f"https://raw.githubusercontent.com/openfootball/football.json/master/2025-26/en.1.json"
    else:
        url = f"https://raw.githubusercontent.com/openfootball/football.json/master/{LEAGUE_PATHS[league_key]['path']}"
    
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        
        raw_text = resp.content.decode('utf-8')
        data = json.loads(raw_text)

        fixtures = []
        for match in data.get("matches", []):
            fixtures.append({
                "round": match.get("round"),
                "date": match.get("date"),
                "time": match.get("time"),
                "home_team": match.get("team1"),
                "away_team": match.get("team2"),
                "home_score": match.get("score", {}).get("ft", [None, None])[0],
                "away_score": match.get("score", {}).get("ft", [None, None])[1],
                "league_key": league_key,
                "league_name": LEAGUE_PATHS.get(league_key, {}).get('name', league_key),
                "season": season
            })

        return pd.DataFrame(fixtures)
    
    except Exception as e:
        print(f"❌ Error fetching {league_key}: {e}")
        return pd.DataFrame()

def fetch_all_fixtures(season: str = "2025-26") -> pd.DataFrame:
    """Fetch all leagues with error handling"""
    all_fixtures = []
    
    for league_key in tqdm(LEAGUE_PATHS.keys(), desc="Fetching Fixtures", unit="league"):
        df = fetch_fixtures(league_key, season=season)
        if df is not None and len(df) > 0:
            all_fixtures.append(df)
    
    if all_fixtures:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return pd.concat(all_fixtures, ignore_index=True)
    else:
        return pd.DataFrame()

def fetch_and_save_fixtures(filepath: str = None, season: str = "2025-26"):
    """
    Fetch fixtures and save to specified path.
    """
    # Use current directory as base if SCRIPT_DIR fails
    try:
        SCRIPT_DIR = Path(__file__).resolve().parent.parent
    except:
        SCRIPT_DIR = Path.cwd()
    
    if filepath is None:
        filepath = SCRIPT_DIR / "data" / "fixtures_data.csv"
    else:
        filepath = Path(filepath)
    
    # Ensure data directory exists
    filepath.parent.mkdir(exist_ok=True)
    
    # Fetch fixtures
    print(f"📥 Fetching fixtures for season {season}...")
    df = fetch_all_fixtures(season=season)
    
    if df.empty:
        print("❌ No fixtures fetched - creating empty DataFrame with required columns")
        # Create empty DataFrame with required columns to prevent crashes
        df = pd.DataFrame(columns=[
            'round', 'date', 'time', 'home_team', 'away_team', 
            'home_score', 'away_score', 'league_key', 'league_name', 'season'
        ])
    
    # Format date if possible
    try:
        df = format_date_column(df)
        print("✅ Dates formatted")
    except (NameError, AttributeError) as e:
        print(f"⚠️  Date formatting skipped: {e}")
    
    # Save to CSV
    try:
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"✅ Saved {len(df)} fixtures to {filepath}")
    except Exception as e:
        print(f"❌ Failed to save fixtures: {e}")
    
    return df

if __name__ == "__main__":
    # Test the function
    df = fetch_and_save_fixtures()
    print(f"Test completed. Fixtures shape: {df.shape}")