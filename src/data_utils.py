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
    Handles both DD/MM/YYYY and MM/DD/YYYY formats robustly,
    while avoiding pandas 'Could not infer format' warnings.
    """
    # Identify and standardize the 'Date' column ---
    date_col_candidates = [c for c in df.columns if c.lower() == 'date']
    if not date_col_candidates:
        return df  # No date column found
    date_col = date_col_candidates[0]
    if date_col != 'Date':
        df = df.rename(columns={date_col: 'Date'})

    # Clean up the date strings ---
    df['Date'] = df['Date'].astype(str).str.replace(r"[-._]", "/", regex=True).str.strip()

    # Infer format (dayfirst or monthfirst) ---
    sample_dates = df['Date'].dropna().head(50)
    ddmm_count, mmdd_count = 0, 0

    for d in sample_dates:
        parts = d.split('/')
        if len(parts) == 3 and all(p.isdigit() for p in parts[:2]):
            day, month = int(parts[0]), int(parts[1])
            if day > 12:
                ddmm_count += 1
            elif month > 12:
                mmdd_count += 1

    dayfirst = ddmm_count > mmdd_count

    # Apply explicit format to suppress warnings
    # Choose appropriate format string
    fmt = "%d/%m/%Y" if dayfirst else "%m/%d/%Y"

    # Try parsing with inferred format; fallback safely if mixed formats exist
    parsed = pd.to_datetime(df['Date'], format=fmt, errors='coerce')

    # If more than 30% NaT, try the opposite format
    if parsed.isna().mean() > 0.3:
        alt_fmt = "%m/%d/%Y" if dayfirst else "%d/%m/%Y"
        parsed_alt = pd.to_datetime(df['Date'], format=alt_fmt, errors='coerce')
        if parsed_alt.notna().sum() > parsed.notna().sum():
            parsed = parsed_alt

    df['Date'] = parsed
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