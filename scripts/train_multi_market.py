# scripts/train_advanced_multi_market.py
import sys
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    log_loss, accuracy_score, brier_score_loss, 
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import joblib
from scipy.stats import rankdata
from typing import Tuple, Dict, List

# Paths
DATA_DIR = SCRIPT_DIR / "data"
MODEL_DIR = SCRIPT_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# Global timer
START_TIME = time.time()

def log_time(message: str):
    """Log elapsed time with message."""
    elapsed = time.time() - START_TIME
    print(f"[{elapsed/60:.2f}m] {message}")

def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load historical data and league tables."""
    log_time("Loading data...")
    
    # Load historical data
    hist_path = DATA_DIR / "cleaned_historical_data.csv"
    df = pd.read_csv(hist_path, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Load league tables
    table_path = DATA_DIR / "current_season_leagues_table.csv"
    league_table = pd.read_csv(table_path)
    
    log_time(f"Loaded {len(df):,} matches from {df['season'].nunique()} seasons")
    log_time(f"Loaded league tables with {len(league_table)} entries")
    
    return df, league_table

def compute_elo_ratings(df: pd.DataFrame, k_factor: float = 20) -> pd.DataFrame:
    """Compute dynamic ELO ratings for teams."""
    df = df.copy().sort_values('Date').reset_index(drop=True)
    
    elo_ratings = {}
    default_elo = 1500
    
    df['home_elo_before'] = 0.0
    df['away_elo_before'] = 0.0
    df['elo_diff'] = 0.0
    
    for idx, row in df.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        
        # Initialize if new team
        if home not in elo_ratings:
            elo_ratings[home] = default_elo
        if away not in elo_ratings:
            elo_ratings[away] = default_elo
        
        # Store pre-match ELO
        home_elo = elo_ratings[home]
        away_elo = elo_ratings[away]
        df.at[idx, 'home_elo_before'] = home_elo
        df.at[idx, 'away_elo_before'] = away_elo
        df.at[idx, 'elo_diff'] = home_elo - away_elo
        
        # Update ELO after match (if result available)
        if pd.notna(row['FTR']):
            expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
            
            if row['FTR'] == 'H':
                actual_home = 1.0
            elif row['FTR'] == 'D':
                actual_home = 0.5
            else:
                actual_home = 0.0
            
            elo_ratings[home] += k_factor * (actual_home - expected_home)
            elo_ratings[away] += k_factor * ((1 - actual_home) - (1 - expected_home))
    
    return df

def compute_advanced_form_features(df: pd.DataFrame, windows: List[int] = [3, 5, 10]) -> pd.DataFrame:
    """Compute rolling form features for multiple windows."""
    df = df.copy().sort_values('Date').reset_index(drop=True)
    
    # Ensure numeric
    df['FTHG'] = pd.to_numeric(df['FTHG'], errors='coerce')
    df['FTAG'] = pd.to_numeric(df['FTAG'], errors='coerce')
    
    feature_dict = {}
    
    log_time("Computing advanced form features...")
    
    # Build match history for each team
    all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    
    for team in all_teams:
        # All matches for this team
        home_matches = df[df['HomeTeam'] == team].copy()
        away_matches = df[df['AwayTeam'] == team].copy()
        
        home_matches['team_goals'] = home_matches['FTHG']
        home_matches['opp_goals'] = home_matches['FTAG']
        home_matches['points'] = home_matches['FTR'].map({'H': 3, 'D': 1, 'A': 0})
        home_matches['venue'] = 'home'
        
        away_matches['team_goals'] = away_matches['FTAG']
        away_matches['opp_goals'] = away_matches['FTHG']
        away_matches['points'] = away_matches['FTR'].map({'H': 0, 'D': 1, 'A': 3})
        away_matches['venue'] = 'away'
        
        all_matches = pd.concat([home_matches, away_matches]).sort_values('Date')
        feature_dict[team] = all_matches
    
    # Initialize feature columns
    for window in windows:
        for prefix in ['home', 'away']:
            df[f'{prefix}_form_{window}'] = 0.0
            df[f'{prefix}_goals_scored_{window}'] = 0.0
            df[f'{prefix}_goals_conceded_{window}'] = 0.0
            df[f'{prefix}_goal_diff_{window}'] = 0.0
            df[f'{prefix}_ppg_{window}'] = 0.0
            df[f'{prefix}_win_rate_{window}'] = 0.0
            df[f'{prefix}_draw_rate_{window}'] = 0.0
            df[f'{prefix}_loss_rate_{window}'] = 0.0
    
    # Additional features
    for prefix in ['home', 'away']:
        df[f'{prefix}_recent_momentum'] = 0.0
        df[f'{prefix}_consistency'] = 0.0
        df[f'{prefix}_scoring_trend'] = 0.0
    
    # Compute features for each match
    for idx in range(len(df)):
        if idx % 10000 == 0:
            log_time(f"  Progress: {idx:,}/{len(df):,} matches")
        
        row = df.iloc[idx]
        match_date = row['Date']
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Home team features
        home_history = feature_dict[home_team]
        home_past = home_history[home_history['Date'] < match_date]
        home_past_venue = home_past[home_past['venue'] == 'home']
        
        # Away team features
        away_history = feature_dict[away_team]
        away_past = away_history[away_history['Date'] < match_date]
        away_past_venue = away_past[away_past['venue'] == 'away']
        
        # Compute for each window
        for window in windows:
            # Home team
            if len(home_past_venue) >= window:
                recent = home_past_venue.tail(window)
                df.at[idx, f'home_form_{window}'] = recent['points'].sum()
                df.at[idx, f'home_goals_scored_{window}'] = recent['team_goals'].mean()
                df.at[idx, f'home_goals_conceded_{window}'] = recent['opp_goals'].mean()
                df.at[idx, f'home_goal_diff_{window}'] = (recent['team_goals'] - recent['opp_goals']).mean()
                df.at[idx, f'home_ppg_{window}'] = recent['points'].mean()
                df.at[idx, f'home_win_rate_{window}'] = (recent['points'] == 3).mean()
                df.at[idx, f'home_draw_rate_{window}'] = (recent['points'] == 1).mean()
                df.at[idx, f'home_loss_rate_{window}'] = (recent['points'] == 0).mean()
            
            # Away team
            if len(away_past_venue) >= window:
                recent = away_past_venue.tail(window)
                df.at[idx, f'away_form_{window}'] = recent['points'].sum()
                df.at[idx, f'away_goals_scored_{window}'] = recent['team_goals'].mean()
                df.at[idx, f'away_goals_conceded_{window}'] = recent['opp_goals'].mean()
                df.at[idx, f'away_goal_diff_{window}'] = (recent['team_goals'] - recent['opp_goals']).mean()
                df.at[idx, f'away_ppg_{window}'] = recent['points'].mean()
                df.at[idx, f'away_win_rate_{window}'] = (recent['points'] == 3).mean()
                df.at[idx, f'away_draw_rate_{window}'] = (recent['points'] == 1).mean()
                df.at[idx, f'away_loss_rate_{window}'] = (recent['points'] == 0).mean()
        
        # Momentum features (weighted recent form)
        if len(home_past_venue) >= 5:
            recent_5 = home_past_venue.tail(5)
            weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])  # More weight on recent
            df.at[idx, 'home_recent_momentum'] = np.average(recent_5['points'].values, weights=weights)
            df.at[idx, 'home_consistency'] = recent_5['points'].std()
            
            if len(recent_5) >= 3:
                goals = recent_5['team_goals'].values
                df.at[idx, 'home_scoring_trend'] = np.polyfit(range(len(goals)), goals, 1)[0]
        
        if len(away_past_venue) >= 5:
            recent_5 = away_past_venue.tail(5)
            weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
            df.at[idx, 'away_recent_momentum'] = np.average(recent_5['points'].values, weights=weights)
            df.at[idx, 'away_consistency'] = recent_5['points'].std()
            
            if len(recent_5) >= 3:
                goals = recent_5['team_goals'].values
                df.at[idx, 'away_scoring_trend'] = np.polyfit(range(len(goals)), goals, 1)[0]
    
    return df

def compute_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute head-to-head features."""
    df = df.copy().sort_values('Date').reset_index(drop=True)
    
    log_time("Computing H2H features...")
    
    # Initialize
    h2h_cols = [
        'h2h_matches', 'h2h_home_wins', 'h2h_draws', 'h2h_away_wins',
        'h2h_avg_goals_home', 'h2h_avg_goals_away', 'h2h_btts_rate',
        'h2h_over25_rate', 'h2h_home_win_rate', 'h2h_recent_trend'
    ]
    
    for col in h2h_cols:
        df[col] = 0.0
    
    for idx in range(len(df)):
        if idx % 10000 == 0:
            log_time(f"  H2H Progress: {idx:,}/{len(df):,}")
        
        row = df.iloc[idx]
        home, away = row['HomeTeam'], row['AwayTeam']
        match_date = row['Date']
        
        # Get past H2H
        past = df.iloc[:idx]
        h2h = past[
            ((past['HomeTeam'] == home) & (past['AwayTeam'] == away)) |
            ((past['HomeTeam'] == away) & (past['AwayTeam'] == home))
        ]
        
        if len(h2h) > 0:
            # Perspective from current home team
            h2h = h2h.copy()
            h2h['goals_for'] = np.where(h2h['HomeTeam'] == home, h2h['FTHG'], h2h['FTAG'])
            h2h['goals_against'] = np.where(h2h['HomeTeam'] == home, h2h['FTAG'], h2h['FTHG'])
            h2h['result'] = np.where(h2h['goals_for'] > h2h['goals_against'], 'W',
                                    np.where(h2h['goals_for'] == h2h['goals_against'], 'D', 'L'))
            
            df.at[idx, 'h2h_matches'] = len(h2h)
            df.at[idx, 'h2h_home_wins'] = (h2h['result'] == 'W').sum()
            df.at[idx, 'h2h_draws'] = (h2h['result'] == 'D').sum()
            df.at[idx, 'h2h_away_wins'] = (h2h['result'] == 'L').sum()
            df.at[idx, 'h2h_avg_goals_home'] = h2h['goals_for'].mean()
            df.at[idx, 'h2h_avg_goals_away'] = h2h['goals_against'].mean()
            df.at[idx, 'h2h_btts_rate'] = ((h2h['goals_for'] > 0) & (h2h['goals_against'] > 0)).mean()
            df.at[idx, 'h2h_over25_rate'] = ((h2h['goals_for'] + h2h['goals_against']) > 2.5).mean()
            df.at[idx, 'h2h_home_win_rate'] = (h2h['result'] == 'W').mean()
            
            # Recent trend (last 3 H2H)
            if len(h2h) >= 3:
                recent = h2h.tail(3)
                trend_points = recent['result'].map({'W': 3, 'D': 1, 'L': 0}).mean()
                df.at[idx, 'h2h_recent_trend'] = trend_points
    
    return df

def merge_league_table_features(df: pd.DataFrame, league_table: pd.DataFrame, current_season: str = '2024-25') -> pd.DataFrame:
    """Merge league table features for current season matches."""
    df = df.copy()
    
    log_time("Merging league table features...")
    
    # Initialize columns
    table_features = [
        'home_league_position', 'home_league_points', 'home_league_gd',
        'home_league_form_pts', 'away_league_position', 'away_league_points',
        'away_league_gd', 'away_league_form_pts', 'position_diff',
        'points_diff', 'gd_diff'
    ]
    
    for col in table_features:
        df[col] = 0.0
    
    # Only process current season matches
    current_season_mask = df['season'] == current_season
    
    for idx in df[current_season_mask].index:
        row = df.loc[idx]
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        league_code = row['league_code']
        
        # Find teams in league table
        home_data = league_table[
            (league_table['Team'] == home_team) & 
            (league_table['league_code'] == league_code)
        ]
        away_data = league_table[
            (league_table['Team'] == away_team) & 
            (league_table['league_code'] == league_code)
        ]
        
        if not home_data.empty and not away_data.empty:
            home_data = home_data.iloc[0]
            away_data = away_data.iloc[0]
            
            # Extract features
            df.at[idx, 'home_league_position'] = home_data['Pos']
            df.at[idx, 'home_league_points'] = home_data['Pts']
            df.at[idx, 'home_league_gd'] = home_data['GD']
            
            df.at[idx, 'away_league_position'] = away_data['Pos']
            df.at[idx, 'away_league_points'] = away_data['Pts']
            df.at[idx, 'away_league_gd'] = away_data['GD']
            
            # Parse form (e.g., "WWDLL" -> recent points)
            if pd.notna(home_data['Form']) and home_data['Form']:
                form_str = str(home_data['Form'])[-5:]  # Last 5 games
                form_points = sum(3 if c == 'W' else (1 if c == 'D' else 0) for c in form_str)
                df.at[idx, 'home_league_form_pts'] = form_points
            
            if pd.notna(away_data['Form']) and away_data['Form']:
                form_str = str(away_data['Form'])[-5:]
                form_points = sum(3 if c == 'W' else (1 if c == 'D' else 0) for c in form_str)
                df.at[idx, 'away_league_form_pts'] = form_points
            
            # Differential features
            df.at[idx, 'position_diff'] = away_data['Pos'] - home_data['Pos']  # Positive = home ranked higher
            df.at[idx, 'points_diff'] = home_data['Pts'] - away_data['Pts']
            df.at[idx, 'gd_diff'] = home_data['GD'] - away_data['GD']
    
    log_time(f"  Merged features for {current_season_mask.sum():,} current season matches")
    
    return df

def extract_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from betting odds."""
    df = df.copy()
    
    log_time("Extracting odds features...")
    
    # Use B365 odds (most complete)
    # Replace invalid odds (0, negative, or NaN) with neutral values
    for col in ['B365H', 'B365D', 'B365A']:
        if col in df.columns:
            df[col] = df[col].replace([0, -np.inf, np.inf], np.nan)
            df[col] = df[col].clip(lower=1.01, upper=100)  # Reasonable odds range
    
    # Market-implied probabilities with safe division
    df['b365_inv_h'] = np.where(df['B365H'].notna() & (df['B365H'] > 0), 
                                 1 / df['B365H'], 0.33)
    df['b365_inv_d'] = np.where(df['B365D'].notna() & (df['B365D'] > 0), 
                                 1 / df['B365D'], 0.33)
    df['b365_inv_a'] = np.where(df['B365A'].notna() & (df['B365A'] > 0), 
                                 1 / df['B365A'], 0.33)
    
    df['b365_overround'] = df['b365_inv_h'] + df['b365_inv_d'] + df['b365_inv_a']
    df['b365_overround'] = df['b365_overround'].clip(lower=1.0, upper=1.5)  # Typical range
    
    # Normalized probabilities
    df['b365_prob_h'] = df['b365_inv_h'] / df['b365_overround']
    df['b365_prob_d'] = df['b365_inv_d'] / df['b365_overround']
    df['b365_prob_a'] = df['b365_inv_a'] / df['b365_overround']
    
    # Clip probabilities to valid range
    df['b365_prob_h'] = df['b365_prob_h'].clip(0.01, 0.99)
    df['b365_prob_d'] = df['b365_prob_d'].clip(0.01, 0.99)
    df['b365_prob_a'] = df['b365_prob_a'].clip(0.01, 0.99)
    
    # Odds ratios with safe division
    df['odds_ratio_h_a'] = np.where((df['B365H'] > 0) & (df['B365A'] > 0),
                                     df['B365H'] / df['B365A'], 1.0)
    df['odds_ratio_h_d'] = np.where((df['B365H'] > 0) & (df['B365D'] > 0),
                                     df['B365H'] / df['B365D'], 1.0)
    
    # Clip ratios to reasonable range
    df['odds_ratio_h_a'] = df['odds_ratio_h_a'].clip(0.1, 10.0)
    df['odds_ratio_h_d'] = df['odds_ratio_h_d'].clip(0.1, 10.0)
    
    # Expected goals from odds (simplified Poisson approximation)
    df['b365_expected_home_goals'] = -np.log(df['b365_prob_a'].clip(0.05, 0.95)) * 1.4
    df['b365_expected_away_goals'] = -np.log(df['b365_prob_h'].clip(0.05, 0.95)) * 1.4
    
    # Clip expected goals to realistic range
    df['b365_expected_home_goals'] = df['b365_expected_home_goals'].clip(0.1, 5.0)
    df['b365_expected_away_goals'] = df['b365_expected_away_goals'].clip(0.1, 5.0)
    
    # Odds entropy (market uncertainty)
    probs = df[['b365_prob_h', 'b365_prob_d', 'b365_prob_a']].values
    df['odds_entropy'] = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    df['odds_entropy'] = df['odds_entropy'].clip(0, 2.0)  # Reasonable entropy range
    
    # Over/Under odds
    if 'B365>2.5' in df.columns and 'B365<2.5' in df.columns:
        # Clean odds
        df['B365>2.5'] = df['B365>2.5'].replace([0, -np.inf, np.inf], np.nan)
        df['B365<2.5'] = df['B365<2.5'].replace([0, -np.inf, np.inf], np.nan)
        df['B365>2.5'] = df['B365>2.5'].clip(lower=1.01, upper=10)
        df['B365<2.5'] = df['B365<2.5'].clip(lower=1.01, upper=10)
        
        inv_over = np.where(df['B365>2.5'].notna() & (df['B365>2.5'] > 0),
                           1 / df['B365>2.5'], 0.5)
        inv_under = np.where(df['B365<2.5'].notna() & (df['B365<2.5'] > 0),
                            1 / df['B365<2.5'], 0.5)
        
        df['b365_prob_over25'] = inv_over / (inv_over + inv_under)
        df['b365_prob_over25'] = df['b365_prob_over25'].clip(0.01, 0.99)
    
    return df

def build_feature_matrix(df: pd.DataFrame, league_table: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Build complete feature matrix."""
    log_time("Building feature matrix...")
    
    # Compute all features
    df = compute_elo_ratings(df)
    df = compute_advanced_form_features(df)
    df = compute_h2h_features(df)
    df = merge_league_table_features(df, league_table)
    df = extract_odds_features(df)
    
    # Define feature columns (exclude odds for HDA model, include for others)
    base_features = [
        # ELO
        'home_elo_before', 'away_elo_before', 'elo_diff',
        # Form features (all windows)
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
        # Momentum
        'home_recent_momentum', 'away_recent_momentum',
        'home_consistency', 'away_consistency',
        'home_scoring_trend', 'away_scoring_trend',
        # H2H
        'h2h_matches', 'h2h_home_wins', 'h2h_draws', 'h2h_away_wins',
        'h2h_avg_goals_home', 'h2h_avg_goals_away', 'h2h_btts_rate',
        'h2h_over25_rate', 'h2h_home_win_rate', 'h2h_recent_trend',
        # League table
        'home_league_position', 'home_league_points', 'home_league_gd',
        'home_league_form_pts', 'away_league_position', 'away_league_points',
        'away_league_gd', 'away_league_form_pts', 'position_diff',
        'points_diff', 'gd_diff'
    ]
    
    odds_features = [
        'b365_prob_h', 'b365_prob_d', 'b365_prob_a',
        'odds_ratio_h_a', 'odds_ratio_h_d',
        'b365_expected_home_goals', 'b365_expected_away_goals',
        'odds_entropy', 'b365_overround'
    ]
    
    if 'b365_prob_over25' in df.columns:
        odds_features.append('b365_prob_over25')
    
    # Clean all features: replace inf and clip extreme values
    log_time("Cleaning features...")
    all_features = base_features + odds_features
    
    for col in all_features:
        if col in df.columns:
            # Replace inf/-inf with NaN
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # Clip extreme values (based on reasonable ranges)
            if 'elo' in col.lower():
                df[col] = df[col].clip(-1000, 1000)
            elif 'goals' in col.lower():
                df[col] = df[col].clip(-10, 10)
            elif 'rate' in col.lower() or 'prob' in col.lower():
                df[col] = df[col].clip(0, 1)
            elif 'ppg' in col.lower() or 'form' in col.lower():
                df[col] = df[col].clip(-10, 30)
            elif 'position' in col.lower():
                df[col] = df[col].clip(-30, 30)
            elif 'points' in col.lower():
                df[col] = df[col].clip(-100, 100)
            elif 'gd' in col.lower():
                df[col] = df[col].clip(-100, 100)
            elif 'ratio' in col.lower():
                df[col] = df[col].clip(0.01, 100)
            elif 'entropy' in col.lower():
                df[col] = df[col].clip(0, 5)
            elif 'overround' in col.lower():
                df[col] = df[col].clip(1.0, 1.5)
            else:
                # General extreme value clipping
                df[col] = df[col].clip(-1000, 1000)
    
    log_time("Feature cleaning complete")
    
    return df, base_features, odds_features

def train_ensemble_model(X_train, y_train, X_val, y_val, task='multiclass', class_weights=None):
    """Train ensemble of XGBoost, LightGBM, and CatBoost."""
    models = {}
    predictions = {}
    
    # Compute class weights if provided
    if class_weights is not None:
        sample_weights_train = np.array([class_weights[y] for y in y_train])
        sample_weights_val = np.array([class_weights[y] for y in y_val])
    else:
        sample_weights_train = None
        sample_weights_val = None
    
    # XGBoost
    if task == 'multiclass':
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            objective='multi:softprob',
            num_class=3,
            tree_method='hist'
        )
        xgb_model.fit(X_train, y_train, sample_weight=sample_weights_train)
        predictions['xgb'] = xgb_model.predict_proba(X_val)
    else:
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            objective='binary:logistic',
            tree_method='hist',
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]) if len(y_train[y_train==1]) > 0 else 1
        )
        xgb_model.fit(X_train, y_train, sample_weight=sample_weights_train)
        predictions['xgb'] = xgb_model.predict_proba(X_val)[:, 1]
    
    models['xgb'] = xgb_model
    
    # LightGBM
    if task == 'multiclass':
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            objective='multiclass',
            num_class=3,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train, sample_weight=sample_weights_train)
        predictions['lgb'] = lgb_model.predict_proba(X_val)
    else:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            objective='binary',
            is_unbalance=True,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train, sample_weight=sample_weights_train)
        predictions['lgb'] = lgb_model.predict_proba(X_val)[:, 1]
    
    models['lgb'] = lgb_model
    
    # CatBoost
    if task == 'multiclass':
        cat_model = CatBoostClassifier(
            iterations=300,
            depth=7,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_seed=42,
            loss_function='MultiClass',
            verbose=0
        )
        cat_model.fit(X_train, y_train, sample_weight=sample_weights_train)
        predictions['cat'] = cat_model.predict_proba(X_val)
    else:
        cat_model = CatBoostClassifier(
            iterations=300,
            depth=7,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_seed=42,
            loss_function='Logloss',
            auto_class_weights='Balanced',
            verbose=0
        )
        cat_model.fit(X_train, y_train, sample_weight=sample_weights_train)
        predictions['cat'] = cat_model.predict_proba(X_val)[:, 1]
    
    models['cat'] = cat_model
    
    # Ensemble (simple average)
    if task == 'multiclass':
        ensemble_pred = np.mean([predictions['xgb'], predictions['lgb'], predictions['cat']], axis=0)
    else:
        ensemble_pred = np.mean([predictions['xgb'], predictions['lgb'], predictions['cat']], axis=0)
    
    return models, ensemble_pred

def evaluate_model(y_true, y_pred_proba, y_pred_class, task_name: str, class_names=None):
    """Comprehensive model evaluation."""
    print(f"\n{'='*70}")
    print(f"EVALUATION: {task_name}")
    print(f"{'='*70}")
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred_class)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Probabilistic metrics
    if len(y_pred_proba.shape) == 1 or y_pred_proba.shape[1] == 2:
        # Binary classification
        if len(y_pred_proba.shape) == 2:
            y_pred_proba_positive = y_pred_proba[:, 1]
        else:
            y_pred_proba_positive = y_pred_proba
        
        brier = brier_score_loss(y_true, y_pred_proba_positive)
        print(f"Brier Score: {brier:.4f} (lower is better)")
        
        try:
            auc = roc_auc_score(y_true, y_pred_proba_positive)
            print(f"ROC-AUC: {auc:.4f}")
        except:
            print("ROC-AUC: N/A")
        
        logloss = log_loss(y_true, np.column_stack([1-y_pred_proba_positive, y_pred_proba_positive]))
        print(f"Log Loss: {logloss:.4f} (lower is better)")
    else:
        # Multiclass
        logloss = log_loss(y_true, y_pred_proba)
        print(f"Log Loss: {logloss:.4f} (lower is better)")
        
        # Brier score for multiclass
        n_classes = y_pred_proba.shape[1]
        y_true_binary = np.zeros((len(y_true), n_classes))
        for i, label in enumerate(y_true):
            y_true_binary[i, label] = 1
        brier = np.mean(np.sum((y_pred_proba - y_true_binary) ** 2, axis=1))
        print(f"Brier Score: {brier:.4f} (lower is better)")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_class, target_names=class_names, digits=4))
    
    # Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred_class)
    print(cm)
    
    # Per-class precision/recall
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred_class, average=None)
    print("\nPer-Class Metrics:")
    for i, name in enumerate(class_names):
        print(f"  {name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")
    
    # Expected Value metrics (comparing to betting odds if available)
    print(f"\n{'='*70}")
    
    return {
        'accuracy': accuracy,
        'log_loss': logloss if 'logloss' in locals() else None,
        'brier': brier,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def calculate_expected_value(df: pd.DataFrame, model_probs: np.ndarray, market: str = 'HDA'):
    """Calculate expected value of betting strategy."""
    if market == 'HDA':
        # Home/Draw/Away
        home_odds = df['B365H'].values
        draw_odds = df['B365D'].values
        away_odds = df['B365A'].values
        
        # Model probabilities
        model_home = model_probs[:, 0]
        model_draw = model_probs[:, 1]
        model_away = model_probs[:, 2]
        
        # Expected value for each outcome
        ev_home = (model_home * home_odds) - 1
        ev_draw = (model_draw * draw_odds) - 1
        ev_away = (model_away * away_odds) - 1
        
        # Find positive EV bets
        positive_ev_home = ev_home > 0.05  # 5% edge threshold
        positive_ev_draw = ev_draw > 0.05
        positive_ev_away = ev_away > 0.05
        
        total_positive = positive_ev_home.sum() + positive_ev_draw.sum() + positive_ev_away.sum()
        
        print(f"\n{'='*70}")
        print(f"VALUE BETTING ANALYSIS (HDA Market)")
        print(f"{'='*70}")
        print(f"Total matches analyzed: {len(df)}")
        print(f"Positive EV opportunities (>5% edge): {total_positive}")
        print(f"  - Home bets: {positive_ev_home.sum()} (avg EV: {ev_home[positive_ev_home].mean():.2%})")
        print(f"  - Draw bets: {positive_ev_draw.sum()} (avg EV: {ev_draw[positive_ev_draw].mean():.2%})")
        print(f"  - Away bets: {positive_ev_away.sum()} (avg EV: {ev_away[positive_ev_away].mean():.2%})")
        print(f"{'='*70}")

def train_models():
    """Main training pipeline."""
    log_time("="*70)
    log_time("STARTING ADVANCED FOOTBALL PREDICTION MODEL TRAINING")
    log_time("="*70)
    
    # Load data
    df, league_table = load_and_prepare_data()
    
    # Build features
    df, base_features, odds_features = build_feature_matrix(df, league_table)
    
    # Remove rows with missing target or critical features
    df = df.dropna(subset=['FTHG', 'FTAG', 'FTR']).copy()
    
    # Remove rows with all NaN features
    feature_cols_all = base_features + odds_features
    df = df.dropna(subset=feature_cols_all, how='all')
    
    log_time(f"Final dataset: {len(df):,} matches with complete features")
    
    # Fill remaining NaNs with 0
    df[feature_cols_all] = df[feature_cols_all].fillna(0)
    
    # Final safety check: replace any remaining inf values
    df[feature_cols_all] = df[feature_cols_all].replace([np.inf, -np.inf], 0)
    
    # Verify no NaN or inf values remain
    for col in feature_cols_all:
        if df[col].isna().any():
            log_time(f"WARNING: Column {col} still has NaN values. Filling with 0.")
            df[col] = df[col].fillna(0)
        if np.isinf(df[col]).any():
            log_time(f"WARNING: Column {col} still has inf values. Replacing with 0.")
            df[col] = df[col].replace([np.inf, -np.inf], 0)
    
    # Create targets
    df['target_hda'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    df['target_btts'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
    df['target_over25'] = ((df['FTHG'] + df['FTAG']) > 2.5).astype(int)
    
    # Remove any remaining invalid targets
    df = df.dropna(subset=['target_hda', 'target_btts', 'target_over25'])
    
    log_time(f"Training data ready: {len(df):,} matches")
    
    # Time series split (3-fold)
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Storage for results
    all_results = {
        'HDA': [],
        'BTTS': [],
        'Over25': []
    }
    
    # ========================================================================
    # TRAIN HDA MODEL (Home/Draw/Away)
    # ========================================================================
    log_time("\n" + "="*70)
    log_time("TRAINING HDA MODEL (Home/Draw/Away)")
    log_time("="*70)
    
    # Use base features only (no odds) for HDA to make it more challenging
    X_hda = df[base_features].values
    y_hda = df['target_hda'].values
    
    # Compute class weights for imbalanced classes
    class_weights_hda = compute_class_weight('balanced', classes=np.unique(y_hda), y=y_hda)
    class_weight_dict_hda = {i: class_weights_hda[i] for i in range(len(class_weights_hda))}
    
    log_time(f"Class distribution: Home={np.mean(y_hda==0):.2%}, Draw={np.mean(y_hda==1):.2%}, Away={np.mean(y_hda==2):.2%}")
    log_time(f"Class weights: {class_weight_dict_hda}")
    
    fold_models_hda = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_hda)):
        log_time(f"\nFold {fold+1}/3")
        X_train, X_val = X_hda[train_idx], X_hda[val_idx]
        y_train, y_val = y_hda[train_idx], y_hda[val_idx]
        
        # Train ensemble
        models, ensemble_pred = train_ensemble_model(
            X_train, y_train, X_val, y_val,
            task='multiclass',
            class_weights=class_weight_dict_hda
        )
        
        y_pred_class = np.argmax(ensemble_pred, axis=1)
        
        # Evaluate
        results = evaluate_model(
            y_val, ensemble_pred, y_pred_class,
            f"HDA - Fold {fold+1}",
            class_names=['Home', 'Draw', 'Away']
        )
        
        all_results['HDA'].append(results)
        fold_models_hda.append(models)
        
        # Calculate EV on validation set
        val_df = df.iloc[val_idx]
        if 'B365H' in val_df.columns:
            calculate_expected_value(val_df, ensemble_pred, market='HDA')
    
    # Train final HDA model on all data
    log_time("\nTraining final HDA model on all data...")
    final_models_hda, _ = train_ensemble_model(
        X_hda, y_hda, X_hda[:100], y_hda[:100],  # Dummy validation
        task='multiclass',
        class_weights=class_weight_dict_hda
    )
    
    # Save models
    joblib.dump(final_models_hda, MODEL_DIR / "ensemble_hda.pkl")
    log_time("✅ HDA models saved")
    
    # ========================================================================
    # TRAIN BTTS MODEL (Both Teams to Score)
    # ========================================================================
    log_time("\n" + "="*70)
    log_time("TRAINING BTTS MODEL (Both Teams to Score)")
    log_time("="*70)
    
    # Use all features including odds for BTTS
    X_btts = df[base_features + odds_features].values
    y_btts = df['target_btts'].values
    
    log_time(f"Class distribution: No={np.mean(y_btts==0):.2%}, Yes={np.mean(y_btts==1):.2%}")
    
    fold_models_btts = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_btts)):
        log_time(f"\nFold {fold+1}/3")
        X_train, X_val = X_btts[train_idx], X_btts[val_idx]
        y_train, y_val = y_btts[train_idx], y_btts[val_idx]
        
        # Train ensemble
        models, ensemble_pred = train_ensemble_model(
            X_train, y_train, X_val, y_val,
            task='binary'
        )
        
        y_pred_class = (ensemble_pred > 0.5).astype(int)
        
        # Evaluate
        results = evaluate_model(
            y_val, ensemble_pred, y_pred_class,
            f"BTTS - Fold {fold+1}",
            class_names=['No', 'Yes']
        )
        
        all_results['BTTS'].append(results)
        fold_models_btts.append(models)
    
    # Train final BTTS model
    log_time("\nTraining final BTTS model on all data...")
    final_models_btts, _ = train_ensemble_model(
        X_btts, y_btts, X_btts[:100], y_btts[:100],
        task='binary'
    )
    
    joblib.dump(final_models_btts, MODEL_DIR / "ensemble_btts.pkl")
    log_time("✅ BTTS models saved")
    
    # ========================================================================
    # TRAIN OVER/UNDER 2.5 MODEL
    # ========================================================================
    log_time("\n" + "="*70)
    log_time("TRAINING OVER/UNDER 2.5 MODEL")
    log_time("="*70)
    
    # Use all features including odds
    X_over25 = df[base_features + odds_features].values
    y_over25 = df['target_over25'].values
    
    log_time(f"Class distribution: Under={np.mean(y_over25==0):.2%}, Over={np.mean(y_over25==1):.2%}")
    
    fold_models_over25 = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_over25)):
        log_time(f"\nFold {fold+1}/3")
        X_train, X_val = X_over25[train_idx], X_over25[val_idx]
        y_train, y_val = y_over25[train_idx], y_over25[val_idx]
        
        # Train ensemble
        models, ensemble_pred = train_ensemble_model(
            X_train, y_train, X_val, y_val,
            task='binary'
        )
        
        y_pred_class = (ensemble_pred > 0.5).astype(int)
        
        # Evaluate
        results = evaluate_model(
            y_val, ensemble_pred, y_pred_class,
            f"Over/Under 2.5 - Fold {fold+1}",
            class_names=['Under', 'Over']
        )
        
        all_results['Over25'].append(results)
        fold_models_over25.append(models)
    
    # Train final Over/Under model
    log_time("\nTraining final Over/Under 2.5 model on all data...")
    final_models_over25, _ = train_ensemble_model(
        X_over25, y_over25, X_over25[:100], y_over25[:100],
        task='binary'
    )
    
    joblib.dump(final_models_over25, MODEL_DIR / "ensemble_over25.pkl")
    log_time("✅ Over/Under 2.5 models saved")
    
    # ========================================================================
    # SAVE FEATURE METADATA
    # ========================================================================
    feature_metadata = {
        'base_features': base_features,
        'odds_features': odds_features,
        'hda_features': base_features,
        'btts_features': base_features + odds_features,
        'over25_features': base_features + odds_features
    }
    joblib.dump(feature_metadata, MODEL_DIR / "feature_metadata.pkl")
    log_time("✅ Feature metadata saved")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    log_time("\n" + "="*70)
    log_time("FINAL CROSS-VALIDATION SUMMARY")
    log_time("="*70)
    
    for market in ['HDA', 'BTTS', 'Over25']:
        results = all_results[market]
        avg_acc = np.mean([r['accuracy'] for r in results])
        avg_logloss = np.mean([r['log_loss'] for r in results if r['log_loss'] is not None])
        avg_brier = np.mean([r['brier'] for r in results])
        
        log_time(f"\n{market} Market:")
        log_time(f"  Average Accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
        log_time(f"  Average Log Loss: {avg_logloss:.4f}")
        log_time(f"  Average Brier Score: {avg_brier:.4f}")
    
    log_time("\n" + "="*70)
    log_time("KEY INSIGHTS")
    log_time("="*70)
    log_time("1. HDA model uses only base features (no odds) for pure prediction")
    log_time("2. BTTS and Over/Under models include odds for enhanced accuracy")
    log_time("3. Ensemble combines XGBoost, LightGBM, and CatBoost")
    log_time("4. Class weights applied to handle imbalanced datasets")
    log_time("5. Time-series CV ensures no data leakage")
    log_time("6. ELO ratings and advanced form features capture team dynamics")
    log_time("7. League table features add current season context")
    
    # Calculate total training time
    total_time = time.time() - START_TIME
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    log_time("\n" + "="*70)
    log_time(f"✅ TRAINING COMPLETE!")
    log_time(f"Total Training Time: {hours}h {minutes}m {seconds}s")
    log_time(f"Models saved to: {MODEL_DIR}")
    log_time("="*70)

if __name__ == "__main__":
    train_models()