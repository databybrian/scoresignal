# features/h2h_form.py
"""
Time-safe feature engineering for football match predictions.
All features use only historical data prior to match date.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

def compute_elo_ratings(
    historical_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp,
    k_factor: float = 20
) -> Dict[str, float]:
    """
    Compute ELO ratings for teams up to match_date.
    """
    past = historical_df[historical_df['Date'] < match_date].sort_values('Date')
    
    elo_ratings = {}
    default_elo = 1500
    
    # Initialize all teams with default ELO
    all_teams = pd.concat([past['HomeTeam'], past['AwayTeam']]).unique()
    for team in all_teams:
        elo_ratings[team] = default_elo
    
    # Update ELO through history
    for _, match in past.iterrows():
        home = match['HomeTeam']
        away = match['AwayTeam']
        
        if home not in elo_ratings:
            elo_ratings[home] = default_elo
        if away not in elo_ratings:
            elo_ratings[away] = default_elo
        
        home_elo = elo_ratings[home]
        away_elo = elo_ratings[away]
        
        if pd.notna(match['FTR']):
            expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
            
            if match['FTR'] == 'H':
                actual_home = 1.0
            elif match['FTR'] == 'D':
                actual_home = 0.5
            else:
                actual_home = 0.0
            
            elo_ratings[home] += k_factor * (actual_home - expected_home)
            elo_ratings[away] += k_factor * ((1 - actual_home) - (1 - expected_home))
    
    # Get current ELO ratings
    home_elo = elo_ratings.get(home_team, default_elo)
    away_elo = elo_ratings.get(away_team, default_elo)
    
    return {
        'home_elo_before': home_elo,
        'away_elo_before': away_elo,
        'elo_diff': home_elo - away_elo
    }

def compute_advanced_form_features(
    historical_df: pd.DataFrame,
    team: str,
    match_date: pd.Timestamp,
    is_home: bool = True,
    windows: list = [3, 5, 10]
) -> Dict[str, float]:
    """
    Compute advanced form features with multiple time windows.
    """
    past = historical_df[historical_df['Date'] < match_date]
    
    if is_home:
        team_matches = past[past['HomeTeam'] == team].copy()
        team_matches['team_goals'] = team_matches['FTHG']
        team_matches['opp_goals'] = team_matches['FTAG']
        team_matches['points'] = team_matches['FTR'].map({'H': 3, 'D': 1, 'A': 0})
    else:
        team_matches = past[past['AwayTeam'] == team].copy()
        team_matches['team_goals'] = team_matches['FTAG']
        team_matches['opp_goals'] = team_matches['FTHG']
        team_matches['points'] = team_matches['FTR'].map({'H': 0, 'D': 1, 'A': 3})
    
    if team_matches.empty:
        return _get_default_form_features(windows, is_home)
    
    team_matches = team_matches.sort_values('Date', ascending=False)
    
    features = {}
    prefix = 'home' if is_home else 'away'
    
    for window in windows:
        recent = team_matches.head(window)
        
        if len(recent) > 0:
            features[f'{prefix}_form_{window}'] = recent['points'].sum()
            features[f'{prefix}_goals_scored_{window}'] = recent['team_goals'].mean()
            features[f'{prefix}_goals_conceded_{window}'] = recent['opp_goals'].mean()
            features[f'{prefix}_goal_diff_{window}'] = (recent['team_goals'] - recent['opp_goals']).mean()
            features[f'{prefix}_ppg_{window}'] = recent['points'].mean()
            features[f'{prefix}_win_rate_{window}'] = (recent['points'] == 3).mean()
            features[f'{prefix}_draw_rate_{window}'] = (recent['points'] == 1).mean()
            features[f'{prefix}_loss_rate_{window}'] = (recent['points'] == 0).mean()
        else:
            features.update(_get_default_window_features(prefix, window))
    
    # Momentum features
    if len(team_matches) >= 5:
        recent_5 = team_matches.head(5)
        weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
        features[f'{prefix}_recent_momentum'] = np.average(recent_5['points'].values, weights=weights)
        features[f'{prefix}_consistency'] = recent_5['points'].std()
        
        if len(recent_5) >= 3:
            goals = recent_5['team_goals'].values
            features[f'{prefix}_scoring_trend'] = np.polyfit(range(len(goals)), goals, 1)[0]
        else:
            features[f'{prefix}_scoring_trend'] = 0.0
    else:
        features[f'{prefix}_recent_momentum'] = 1.0
        features[f'{prefix}_consistency'] = 1.0
        features[f'{prefix}_scoring_trend'] = 0.0
    
    return features

def _get_default_form_features(windows: list, is_home: bool) -> Dict[str, float]:
    """Get default form features when no historical data exists."""
    features = {}
    prefix = 'home' if is_home else 'away'
    
    for window in windows:
        features.update(_get_default_window_features(prefix, window))
    
    features[f'{prefix}_recent_momentum'] = 1.0
    features[f'{prefix}_consistency'] = 1.0
    features[f'{prefix}_scoring_trend'] = 0.0
    
    return features

def _get_default_window_features(prefix: str, window: int) -> Dict[str, float]:
    """Get default features for a specific window."""
    if prefix == 'home':
        default_goals = 1.5
        default_conceded = 1.2
        default_ppg = 1.6
        default_win_rate = 0.4
    else:
        default_goals = 1.2
        default_conceded = 1.5
        default_ppg = 1.2
        default_win_rate = 0.3
    
    return {
        f'{prefix}_form_{window}': default_ppg * window,
        f'{prefix}_goals_scored_{window}': default_goals,
        f'{prefix}_goals_conceded_{window}': default_conceded,
        f'{prefix}_goal_diff_{window}': default_goals - default_conceded,
        f'{prefix}_ppg_{window}': default_ppg,
        f'{prefix}_win_rate_{window}': default_win_rate,
        f'{prefix}_draw_rate_{window}': 0.25,
        f'{prefix}_loss_rate_{window}': 1 - default_win_rate - 0.25
    }

def compute_h2h_features(
    historical_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp
) -> Dict[str, float]:
    """
    Compute comprehensive Head-to-Head features.
    """
    past = historical_df[historical_df['Date'] < match_date]
    
    h2h = past[
        ((past['HomeTeam'] == home_team) & (past['AwayTeam'] == away_team)) |
        ((past['HomeTeam'] == away_team) & (past['AwayTeam'] == home_team))
    ].copy()
    
    if len(h2h) == 0:
        return {
            'h2h_matches': 0,
            'h2h_home_wins': 0,
            'h2h_draws': 0,
            'h2h_away_wins': 0,
            'h2h_avg_goals_home': 1.5,
            'h2h_avg_goals_away': 1.5,
            'h2h_btts_rate': 0.5,
            'h2h_over25_rate': 0.5,
            'h2h_home_win_rate': 0.4,
            'h2h_recent_trend': 1.5
        }
    
    # Normalize to current home team perspective
    h2h['goals_for'] = np.where(h2h['HomeTeam'] == home_team, h2h['FTHG'], h2h['FTAG'])
    h2h['goals_against'] = np.where(h2h['HomeTeam'] == home_team, h2h['FTAG'], h2h['FTHG'])
    h2h['result'] = np.where(h2h['goals_for'] > h2h['goals_against'], 'W',
                            np.where(h2h['goals_for'] == h2h['goals_against'], 'D', 'L'))
    
    h2h_sorted = h2h.sort_values('Date', ascending=False)
    
    features = {
        'h2h_matches': len(h2h),
        'h2h_home_wins': (h2h['result'] == 'W').sum(),
        'h2h_draws': (h2h['result'] == 'D').sum(),
        'h2h_away_wins': (h2h['result'] == 'L').sum(),
        'h2h_avg_goals_home': h2h['goals_for'].mean(),
        'h2h_avg_goals_away': h2h['goals_against'].mean(),
        'h2h_btts_rate': ((h2h['goals_for'] > 0) & (h2h['goals_against'] > 0)).mean(),
        'h2h_over25_rate': ((h2h['goals_for'] + h2h['goals_against']) > 2.5).mean(),
        'h2h_home_win_rate': (h2h['result'] == 'W').mean()
    }
    
    # Recent trend (last 3 H2H)
    if len(h2h_sorted) >= 3:
        recent = h2h_sorted.head(3)
        trend_points = recent['result'].map({'W': 3, 'D': 1, 'L': 0}).mean()
        features['h2h_recent_trend'] = trend_points
    else:
        features['h2h_recent_trend'] = 1.5
    
    return features

def compute_league_table_features(
    historical_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp,
    league_code: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute league table position features (simplified version).
    This would typically come from external league table data.
    """
    # For now, return default values - in practice you'd integrate with league table data
    return {
        'home_league_position': 8.0,
        'home_league_points': 45.0,
        'home_league_gd': 5.0,
        'home_league_form_pts': 7.0,
        'away_league_position': 12.0,
        'away_league_points': 35.0,
        'away_league_gd': -3.0,
        'away_league_form_pts': 5.0,
        'position_diff': -4.0,  # away_pos - home_pos
        'points_diff': 10.0,    # home_pts - away_pts
        'gd_diff': 8.0          # home_gd - away_gd
    }

def extract_odds_features(
    historical_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp
) -> Dict[str, float]:
    """
    Extract odds-related features from historical data.
    In practice, you'd get current odds from an external source.
    """
    # For new matches, you'd get current odds from betting APIs
    # Here we return neutral values
    return {
        'b365_prob_h': 0.45,
        'b365_prob_d': 0.25,
        'b365_prob_a': 0.30,
        'odds_ratio_h_a': 1.5,
        'odds_ratio_h_d': 1.8,
        'b365_expected_home_goals': 1.6,
        'b365_expected_away_goals': 1.2,
        'odds_entropy': 1.0,
        'b365_overround': 1.05,
        'b365_prob_over25': 0.52
    }

def compute_match_features(
    historical_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp,
    league_code: Optional[str] = None,
    include_odds: bool = False
) -> Dict[str, float]:
    """
    Compute all features for a single match, aligned with training script.
    """
    features = {}
    
    # 1. ELO features
    elo_features = compute_elo_ratings(historical_df, home_team, away_team, match_date)
    features.update(elo_features)
    
    # 2. Advanced form features for home team (home matches)
    home_form = compute_advanced_form_features(
        historical_df, home_team, match_date, is_home=True
    )
    features.update(home_form)
    
    # 3. Advanced form features for away team (away matches)
    away_form = compute_advanced_form_features(
        historical_df, away_team, match_date, is_home=False
    )
    features.update(away_form)
    
    # 4. H2H features
    h2h_features = compute_h2h_features(historical_df, home_team, away_team, match_date)
    features.update(h2h_features)
    
    # 5. League table features
    league_features = compute_league_table_features(
        historical_df, home_team, away_team, match_date, league_code
    )
    features.update(league_features)
    
    # 6. Odds features (optional - for BTTS and Over/Under models)
    if include_odds:
        odds_features = extract_odds_features(historical_df, home_team, away_team, match_date)
        features.update(odds_features)
    
    return features

def get_feature_columns(include_odds: bool = False) -> list:
    """
    Get the list of feature columns in the same order as training.
    This MUST match exactly with the features computed in compute_match_features.
    """
    base_features = [
        # ELO (3 features)
        'home_elo_before', 'away_elo_before', 'elo_diff',
        
        # Form features - 3 windows × 8 metrics × 2 teams = 48 features
        'home_form_3', 'home_form_5', 'home_form_10',
        'away_form_3', 'away_form_5', 'away_form_10',
        'home_goals_scored_3', 'home_goals_scored_5', 'home_goals_scored_10',
        'away_goals_scored_3', 'away_goals_scored_5', 'away_goals_scored_10',
        'home_goals_conceded_3', 'home_goals_conceded_5', 'home_goals_conceded_10',
        'away_goals_conceded_3', 'away_goals_conceded_5', 'away_goals_conceded_10',
        'home_goal_diff_3', 'home_goal_diff_5', 'home_goal_diff_10',
        'away_goal_diff_3', 'away_goal_diff_5', 'away_goal_diff_10',
        'home_ppg_3', 'home_ppg_5', 'home_ppg_10',
        'away_ppg_3', 'away_ppg_5', 'away_ppg_10',
        'home_win_rate_3', 'home_win_rate_5', 'home_win_rate_10',
        'away_win_rate_3', 'away_win_rate_5', 'away_win_rate_10',
        'home_draw_rate_3', 'home_draw_rate_5', 'home_draw_rate_10',
        'away_draw_rate_3', 'away_draw_rate_5', 'away_draw_rate_10',
        'home_loss_rate_3', 'home_loss_rate_5', 'home_loss_rate_10',
        'away_loss_rate_3', 'away_loss_rate_5', 'away_loss_rate_10',
        
        # Momentum (6 features)
        'home_recent_momentum', 'away_recent_momentum',
        'home_consistency', 'away_consistency',
        'home_scoring_trend', 'away_scoring_trend',
        
        # H2H (10 features)
        'h2h_matches', 'h2h_home_wins', 'h2h_draws', 'h2h_away_wins',
        'h2h_avg_goals_home', 'h2h_avg_goals_away', 'h2h_btts_rate',
        'h2h_over25_rate', 'h2h_home_win_rate', 'h2h_recent_trend',
        
        # League table (11 features)
        'home_league_position', 'home_league_points', 'home_league_gd',
        'home_league_form_pts', 'away_league_position', 'away_league_points',
        'away_league_gd', 'away_league_form_pts', 'position_diff',
        'points_diff', 'gd_diff'
    ]
    
    # Verify count: 3(ELO) + 48(form) + 6(momentum) + 10(H2H) + 11(league) = 78
    expected_base_count = 3 + 48 + 6 + 10 + 11
    actual_base_count = len(base_features)
    
    if actual_base_count != expected_base_count:
        print(f"⚠️  WARNING: Base feature count mismatch. Expected: {expected_base_count}, Got: {actual_base_count}")
    
    if include_odds:
        odds_features = [
            'b365_prob_h', 'b365_prob_d', 'b365_prob_a',
            'odds_ratio_h_a', 'odds_ratio_h_d',
            'b365_expected_home_goals', 'b365_expected_away_goals',
            'odds_entropy', 'b365_overround', 'b365_prob_over25'
        ]
        # 78 base + 10 odds = 88 total
        expected_total = expected_base_count + len(odds_features)
        actual_total = len(base_features) + len(odds_features)
        
        if actual_total != expected_total:
            print(f"⚠️  WARNING: Total feature count mismatch. Expected: {expected_total}, Got: {actual_total}")
            
        return base_features + odds_features
    
    return base_features

def create_feature_vector(
    features_dict: Dict[str, float],
    feature_columns: list
) -> np.ndarray:
    """
    Create a feature vector in the correct order for model prediction.
    """
    return np.array([features_dict.get(col, 0.0) for col in feature_columns])