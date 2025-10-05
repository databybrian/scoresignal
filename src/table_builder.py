# src/table_builder.py
import pandas as pd
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
# TABLES_DIR = DATA_DIR / "current_season_tables"
# TABLES_DIR.mkdir(exist_ok=True)
from src.data_utils import current_season_int as current_season


def build_league_table(
    df: pd.DataFrame,
    league_code: str,
    season = current_season,  # football-data.co.uk format: YYZZ
    as_of_date: Optional[str] = None,
    save_current: bool = False
) -> pd.DataFrame:
    """
    Build league table for a specific league and season up to a given date.
    """

    # Ensure Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Filter by BOTH league_code AND season
    league_df = df[
        (df['league_code'] == league_code) &
        (df['season'] == season)
    ].copy()

    if league_df.empty:
        print(f"⚠️ No matches found for {league_code} in season {season}")
        return pd.DataFrame(columns=['Team','P','W','D','L','GF','GA','GD','Pts','Form'])

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
    # Save current table if needed for streamlit dashboard
    # if save_current:
    #     filename = f"{league_code}_{season}.csv"
    #     table_df.to_csv(TABLES_DIR / filename)
    #     print(f"✅ Saved current table: {filename}")

    return table_df

def get_team_position(table_df: pd.DataFrame, team_name: str) -> int:
    """Return the position of a team from a league table.
    
    Args:
        table_df: DataFrame with at least ['Team', 'Pos'].
        team_name: Name of the team to look up.
    
    Returns:
        Team's position (int), or -1 if not found.
    """
    if 'Pos' in table_df.columns:
        row = table_df.loc[table_df['Team'] == team_name]
        if not row.empty:
            return int(row['Pos'].iloc[0])
    else:
        # fallback: if Pos is index
        try:
            return int(table_df.index[table_df['Team'] == team_name][0])
        except IndexError:
            pass
    return -1  # not found
        