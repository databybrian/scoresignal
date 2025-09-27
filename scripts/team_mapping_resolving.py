import pandas as pd
import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from src.data_utils import essential_columns
from bot.config import SCRIPT_DIR

HISTORICAL_FILE = SCRIPT_DIR / "combined_historical_data.csv"
NEW_HISTORICAL_FILE = SCRIPT_DIR / "cleaned_historical_data.csv"
FIXTURES_FILE = SCRIPT_DIR / "fixtures_data.csv"
MAPPING_FILE = SCRIPT_DIR / "team_mapping.csv"
OUTPUT_FILE = SCRIPT_DIR / "historical_data_clean.csv"

def resolve_team_name(
    league_code: str, 
    odds_name: str, 
    team_name_map: dict, 
    fixture_teams_set: set
) -> str:
    """
    Resolve team names using enhanced mapping with fallbacks.
    """
    if pd.isna(odds_name) or not isinstance(odds_name, str):
        return odds_name
        
    # Primary: Use enhanced mapping (historical_league_code + odds_name -> fixture_name)
    if (league_code, odds_name) in team_name_map:
        return team_name_map[(league_code, odds_name)]
    
    # Fallback 1: If team is in your fixture teams, keep original name
    elif odds_name in fixture_teams_set:
        return odds_name
    
    # Fallback 2: Keep original (for historical context)
    else:
        return odds_name

def resolve_column(team_series: pd.Series, league_series: pd.Series, team_name_map: dict, fixture_teams_set: set) -> pd.Series:
    """
    Vectorized team name resolution.
    """
    resolved = []
    for team, league in zip(team_series, league_series):
        if pd.isna(team) or pd.isna(league):
            resolved.append(team)
        else:
            resolved.append(resolve_team_name(league, team, team_name_map, fixture_teams_set))
    return pd.Series(resolved, index=team_series.index)

def load_and_resolve_teams() -> pd.DataFrame:
    """
    Load data and resolve team names in historical data.
    """
    # Load fixtures data
    print("Loading fixtures data...")
    fixtures_data = pd.read_csv(FIXTURES_FILE)
    
    # Build fixture teams set
    fixture_teams_set = set(fixtures_data['home_team'].dropna().unique()) | \
                       set(fixtures_data['away_team'].dropna().unique())
    print(f"Loaded {len(fixture_teams_set)} unique fixture teams")

    # Load historical data
    print("Loading historical data...")
    historical_data = pd.read_csv(HISTORICAL_FILE)
    
    # Load enhanced team mapping
    print("Loading team mapping...")
    mapping_df = pd.read_csv(MAPPING_FILE)
    
    # Build team name mapping dict: (historical_league_code, odds_name) -> fixture_name
    team_name_map = {}
    for _, row in mapping_df.iterrows():
        if pd.notna(row['league_code']) and pd.notna(row['odds_name']):
            team_name_map[(row['league_code'], row['odds_name'])] = row['fixture_name']
    
    print(f"Loaded {len(team_name_map)} team mapping entries")

    # Ensure historical data has 'league_code' column
    if 'league_code' not in historical_data.columns:
        raise ValueError("Historical data must have 'league_code' column (e.g., 'E0', 'SP1')")

    # Apply resolution
    historical_data['HomeTeam_clean'] = resolve_column(
        historical_data['HomeTeam'], 
        historical_data['league_code'],
        team_name_map,
        fixture_teams_set
    )
    historical_data['AwayTeam_clean'] = resolve_column(
        historical_data['AwayTeam'], 
        historical_data['league_code'],
        team_name_map,
        fixture_teams_set
    )
    
    return historical_data

def create_production_historical_file(historical_data_clean: pd.DataFrame, output_path: Path):
    """
    Create lean production-ready historical data file.
    """
    # Select only essential columns
    essential_cols = essential_columns        # Already defined in download_historical_data.py
    
    # Use CLEANED names as the new standard
    production_df = historical_data_clean.copy()
    production_df['HomeTeam'] = production_df['HomeTeam_clean']
    production_df['AwayTeam'] = production_df['AwayTeam_clean']
    
    # Keep only essential columns (drop originals and temp columns)
    production_df = production_df[essential_cols]
    
    # Remove any rows with missing critical data
    production_df = production_df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'])
    
    # Sort by date for efficient lookups
    production_df = production_df.sort_values('Date').reset_index(drop=True)
    
    # Save as the new canonical file
    production_df.to_csv(output_path, index=False)
    print(f"✅ Production file created: {output_path}")
    print(f"   Final size: {len(production_df)} matches")
    print(f"   Memory usage: {production_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# In your main block:
if __name__ == "__main__":
    try:
        historical_data_clean = load_and_resolve_teams()
        
        # Create lean production file (REPLACES original)
        create_production_historical_file(historical_data_clean, NEW_HISTORICAL_FILE)
        
        print(f"\n🎉 Historical data cleaned and optimized!")
        print(f"   New file: {NEW_HISTORICAL_FILE}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise