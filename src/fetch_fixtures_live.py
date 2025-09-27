# fixture_data.py

import requests
import json
import pandas as pd
from tqdm import tqdm
import warnings
from typing import List, Tuple
from data_utils import format_date_column
from league_data import LEAGUE_PATHS 

def fetch_fixtures(league_key: str, season: str = "2025-26") -> pd.DataFrame:
    """
    Fetch fixtures for one league using openfootball.
    """
    if league_key not in LEAGUE_PATHS:
        raise ValueError(f"{league_key} not found in LEAGUE_PATHS")

    url = f"https://raw.githubusercontent.com/openfootball/football.json/master/{LEAGUE_PATHS[league_key]['path']}"
    
    # Get the raw bytes and decode as UTF-8 explicitly
    resp = requests.get(url)
    resp.raise_for_status()
    
    # Force UTF-8 decoding
    raw_text = resp.content.decode('utf-8')
    data = json.loads(raw_text)  # Use json.loads instead of resp.json()

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
            "league_name": LEAGUE_PATHS[league_key]['name'],
            "season": season
        })

    return pd.DataFrame(fixtures)

def fetch_all_fixtures(season: str = "2025-26") -> pd.DataFrame:
    """Your existing function - fetches all leagues"""
    all_fixtures = []
    failed_fixtures = []
    
    for league_key in tqdm(LEAGUE_PATHS.keys(), desc="Fetching Fixtures", unit="league"):
        try:
            df = fetch_fixtures(league_key, season=season)
            if df is not None and len(df) > 0:
                all_fixtures.append(df)
        except Exception as e:
            print(f"Failed {league_key}: {e}")
            continue
    
    if all_fixtures:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return pd.concat(all_fixtures, ignore_index=True)
    else:
        return pd.DataFrame()

def save_fixtures_to_csv(filepath: str = "data/fixtures.csv", season: str = "2025-26"):
    """Fetch all fixtures and save to CSV"""
    print("🔄 Fetching fresh fixtures from openfootball...")
    df = fetch_all_fixtures(season)
    try:
        df = format_date_column(df)
    except (ImportError, AttributeError):
        pass  # Continue without date formatting if function not available
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"✅ Saved {len(df)} fixtures to {filepath}")
    return df

# Optional: Global variable (use sparingly)
# ALL_FIXTURES, FAILED_LEAGUES = fetch_all_fixtures("2025-26")