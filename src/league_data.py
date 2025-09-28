# Import your config
import requests
import os
import pandas as pd
from typing import Dict, Any
import logging
from bot.config import TODAY, HEADERS
from bot.config import current_season

logger = logging.getLogger(__name__)
# League key to name mapping dictionary

def get_current_season_leagues(season: str = "2023-24") -> pd.DataFrame:
    """
    Get league paths for a specific season.
    """
    url = f"https://api.github.com/repos/openfootball/football.json/contents/{season}"
    
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        files = resp.json()
        
        leagues = []
        for f in files:
            if isinstance(f, dict) and f.get("name", "").endswith(".json"):
                league_key = f["name"].replace(".json", "")
                leagues.append({
                    'season': season,
                    'league_key': league_key,
                    'path': f"{season}/{f['name']}",
                    'download_url': f.get('download_url', '')
                })
        
        df = pd.DataFrame(leagues)
        print(f"Found {len(df)} leagues for season {season}")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return pd.DataFrame()

# Quick usage for current season
# df = get_current_season_leagues("2023-24")
"""
Use league fixtures from openfootball to confirm the list of leagues you want to fetch."""
def load_league_names(filepath: str | None = None) -> Dict[str, str]:
    """
    Load league names from CSV file.
    Expected columns: 'fixtures_league_key', 'odds_league_code', 'league_name'
    Only includes rows where both fixtures_league_key and odds_league_code are non-empty.
    """
    if filepath is None:
        # Get the project root (parent of src/) to access raw_data/
        project_root = Path(__file__).resolve().parent.parent
        filepath = project_root / "raw_data" / "footballdata_league_list.csv"
    
    try:
        df = pd.read_csv(filepath)
        
        # Remove rows where either fixtures_league_key or odds_league_code is blank/NaN
        df = df.replace('', pd.NA)
        valid_rows = df.dropna(subset=['fixtures_league_key', 'odds_league_code'])
        
        # Create dictionary: {fixtures_league_key: league_name}
        league_dict = dict(zip(valid_rows['fixtures_league_key'], valid_rows['league_name']))
        
        print(f"Loaded {len(league_dict)} valid league mappings from {len(df)} total rows")
        return league_dict
        
    except FileNotFoundError:
        raise FileNotFoundError(f"League mapping file not found: {filepath}")
    except KeyError as e:
        raise KeyError(f"Required column missing in {filepath}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading league names: {e}")

# Load league names from CSV with proper error handling
try:
    LEAGUE_NAMES = load_league_names()
    print(f"✅ Successfully loaded {len(LEAGUE_NAMES)} league mappings")
except Exception as e:
    print(f"❌ Failed to load league names: {e}")
    LEAGUE_NAMES = {}  # Fallback to empty dict

_cached_league_paths = None

def fetch_league_paths(season: str = "2025-26") -> Dict[str, Dict[str, str]]:
    """
    Fetch all available league paths for a given season from openfootball repo.
    Returns a dict with league info {league_key: {'path': path, 'name': name}}.
    """
    # Fix the URL - remove extra spaces!
    url = f"https://api.github.com/repos/openfootball/football.json/contents/{season}"
    resp = requests.get(url)
    resp.raise_for_status()
    files = resp.json()

    league_paths = {}
    for f in files:
        if isinstance(f, dict) and f.get("name", "").endswith(".json"):
            league_key = f["name"].replace(".json", "")  # e.g. "en.1"
            league_paths[league_key] = {
                'path': f"{season}/{f['name']}",
                'name': LEAGUE_NAMES.get(league_key, f"Unknown League ({league_key})")
            }
    
    return league_paths

def get_league_paths(season: str = "2025-26") -> Dict[str, Dict[str, str]]:
    """Cached version to avoid repeated API calls"""
    global _cached_league_paths
    if _cached_league_paths is None:
        _cached_league_paths = fetch_league_paths(season)
    return _cached_league_paths

# Global variable ready for import
_all_league_paths = get_league_paths(current_season)
# Filter to only leagues that have valid mappings in our CSV
LEAGUE_PATHS = {
    key: value 
    for key, value in _all_league_paths.items() 
    if key in LEAGUE_NAMES
}