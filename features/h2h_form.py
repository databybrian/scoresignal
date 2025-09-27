# features/h2h_form.py
"""
Time-safe feature engineering for football match predictions.
All features use only historical data prior to match date.
"""
import pandas as pd
import numpy as np
from typing import Dict
from typing import Optional

def compute_h2h_features(
    historical_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp
) -> Dict[str, float]:
    """
    Compute Head-to-Head features using matches BEFORE match_date.
    """
    # Get all matches before the target match
    past = historical_df[historical_df['Date'] < match_date]
    
    # Get all H2H matches (both home/away directions)
    h2h = past[
        ((past['HomeTeam'] == home_team) & (past['AwayTeam'] == away_team)) |
        ((past['HomeTeam'] == away_team) & (past['AwayTeam'] == home_team))
    ].copy()
    
    if len(h2h) == 0:
        return {
            'h2h_matches': 0,
            'h2h_home_win_rate': 0.5,
            'h2h_draw_rate': 0.2,
            'h2h_avg_goals_home': 1.5,
            'h2h_avg_goals_away': 1.5,
            'h2h_btts_rate': 0.5,
            'h2h_over_25_rate': 0.5
        }
    
    # Normalize to current home/away perspective
    h2h['goals_home'] = np.where(h2h['HomeTeam'] == home_team, h2h['FTHG'], h2h['FTAG'])
    h2h['goals_away'] = np.where(h2h['HomeTeam'] == home_team, h2h['FTAG'], h2h['FTHG'])
    h2h['home_win'] = h2h['goals_home'] > h2h['goals_away']
    h2h['draw'] = h2h['goals_home'] == h2h['goals_away']
    h2h['btts'] = (h2h['goals_home'] > 0) & (h2h['goals_away'] > 0)
    h2h['over_25'] = (h2h['goals_home'] + h2h['goals_away']) > 2.5
    
    return {
        'h2h_matches': len(h2h),
        'h2h_home_win_rate': h2h['home_win'].mean(),
        'h2h_draw_rate': h2h['draw'].mean(),
        'h2h_avg_goals_home': h2h['goals_home'].mean(),
        'h2h_avg_goals_away': h2h['goals_away'].mean(),
        'h2h_btts_rate': h2h['btts'].mean(),
        'h2h_over_25_rate': h2h['over_25'].mean()
    }

def compute_team_form(
    historical_df: pd.DataFrame,
    team: str,
    match_date: pd.Timestamp,
    is_home: bool = True,
    n_games: int = 5
) -> Dict[str, float]:
    """
    Compute recent form for a team (home or away).
    """
    past = historical_df[historical_df['Date'] < match_date]
    
    if is_home:
        team_matches = past[past['HomeTeam'] == team].copy()
        if team_matches.empty:
            return {
                'avg_goals_scored': 1.5,
                'avg_goals_conceded': 1.5,
                'points_per_game': 1.0,
                'clean_sheet_rate': 0.3,
                'failed_to_score_rate': 0.3
            }
        recent = team_matches.sort_values('Date', ascending=False).head(n_games)
        goals_scored = recent['FTHG']
        goals_conceded = recent['FTAG']
        wins = (recent['FTHG'] > recent['FTAG']).sum()
        draws = (recent['FTHG'] == recent['FTAG']).sum()
        clean_sheets = (recent['FTAG'] == 0).sum()
        failed_to_score = (recent['FTHG'] == 0).sum()
    else:
        team_matches = past[past['AwayTeam'] == team].copy()
        if team_matches.empty:
            return {
                'avg_goals_scored': 1.0,
                'avg_goals_conceded': 1.8,
                'points_per_game': 0.8,
                'clean_sheet_rate': 0.2,
                'failed_to_score_rate': 0.4
            }
        recent = team_matches.sort_values('Date', ascending=False).head(n_games)
        goals_scored = recent['FTAG']
        goals_conceded = recent['FTHG']
        wins = (recent['FTAG'] > recent['FTHG']).sum()
        draws = (recent['FTAG'] == recent['FTHG']).sum()
        clean_sheets = (recent['FTHG'] == 0).sum()
        failed_to_score = (recent['FTAG'] == 0).sum()
    
    total_games = len(recent)
    points = wins * 3 + draws
    
    return {
        'avg_goals_scored': goals_scored.mean() if total_games > 0 else 1.5,
        'avg_goals_conceded': goals_conceded.mean() if total_games > 0 else 1.5,
        'points_per_game': points / total_games if total_games > 0 else 1.0,
        'clean_sheet_rate': clean_sheets / total_games if total_games > 0 else 0.3,
        'failed_to_score_rate': failed_to_score / total_games if total_games > 0 else 0.3
    }

def compute_match_features(
    historical_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp,
    league_code:  Optional[str] = None
) -> Dict[str, float]:
    """
    Compute all features for a single match.
    """
    # H2H features
    h2h_feat = compute_h2h_features(historical_df, home_team, away_team, match_date)
    
    # Home team home form
    home_form = compute_team_form(historical_df, home_team, match_date, is_home=True)
    home_form = {f'home_{k}': v for k, v in home_form.items()}
    
    # Away team away form
    away_form = compute_team_form(historical_df, away_team, match_date, is_home=False)
    away_form = {f'away_{k}': v for k, v in away_form.items()}
    
    # Combine all features
    features = {}
    features.update(h2h_feat)
    features.update(home_form)
    features.update(away_form)
    
    return features