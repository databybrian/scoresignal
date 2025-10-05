import pandas as pd
from pathlib import Path

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

def format_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the date column to proper datetime format.
    Handles both 'Date' and 'date' column names.
    """
    # Standardize column name to 'Date'
    if 'date' in df.columns.str.lower():
        date_col = df.columns[df.columns.str.lower() == 'date'][0]
        if date_col != 'Date':
            df = df.rename(columns={date_col: 'Date'})
    
    if 'Date' not in df.columns:
        return df
    
    # Convert to datetime
    original_count = len(df)
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
    df = df.dropna(subset=['Date'])
    
    removed_count = original_count - len(df)
    if removed_count > 0:
        print(f"⚠️  Removed {removed_count} rows with invalid dates")
    
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