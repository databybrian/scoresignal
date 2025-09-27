# scripts/train_multi_market.py
import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
import xgboost as xgb
import joblib

# Paths
DATA_DIR = SCRIPT_DIR / "data"
MODEL_DIR = SCRIPT_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

def compute_vectorized_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute time-safe features for ALL matches using vectorized operations."""
    df = df.copy().sort_values('Date').reset_index(drop=True)
    
    # Ensure numeric goals
    df['FTHG'] = pd.to_numeric(df['FTHG'], errors='coerce')
    df['FTAG'] = pd.to_numeric(df['FTAG'], errors='coerce')
    
    # Initialize feature columns
    feature_cols = [
        'h2h_matches', 'h2h_home_win_rate', 'h2h_draw_rate',
        'h2h_avg_goals_home', 'h2h_avg_goals_away',
        'h2h_btts_rate', 'h2h_over_25_rate',
        'home_avg_goals_scored', 'home_avg_goals_conceded',
        'home_points_per_game', 'home_clean_sheet_rate',
        'home_failed_to_score_rate',
        'away_avg_goals_scored', 'away_avg_goals_conceded',
        'away_points_per_game', 'away_clean_sheet_rate',
        'away_failed_to_score_rate'
    ]
    
    for col in feature_cols:
        df[col] = np.nan
    
    print("Precomputing team histories...")
    team_histories = {}
    all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    for team in all_teams:
        home_games = df[df['HomeTeam'] == team].copy()
        away_games = df[df['AwayTeam'] == team].copy()
        
        if not home_games.empty:
            home_games['team_goals'] = home_games['FTHG']
            home_games['opp_goals'] = home_games['FTAG']
            home_games['is_home'] = True
        if not away_games.empty:
            away_games['team_goals'] = away_games['FTAG']
            away_games['opp_goals'] = away_games['FTHG']
            away_games['is_home'] = False
            
        all_games = pd.concat([home_games, away_games])
        all_games = all_games.sort_values('Date').reset_index(drop=True)
        team_histories[team] = all_games
    
    print("Computing features...")
    for idx in range(len(df)):
        if idx % 10000 == 0:
            print(f"  Processed {idx}/{len(df)} matches")
            
        row = df.iloc[idx]
        home, away = row['HomeTeam'], row['AwayTeam']
        match_date = row['Date']
        past_df = df.iloc[:idx]
        
        # --- H2H Features ---
        h2h = past_df[
            ((past_df['HomeTeam'] == home) & (past_df['AwayTeam'] == away)) |
            ((past_df['HomeTeam'] == away) & (past_df['AwayTeam'] == home))
        ]
        
        if len(h2h) > 0:
            h2h = h2h.copy()
            h2h['goals_home'] = np.where(h2h['HomeTeam'] == home, h2h['FTHG'], h2h['FTAG'])
            h2h['goals_away'] = np.where(h2h['HomeTeam'] == home, h2h['FTAG'], h2h['FTHG'])
            h2h['home_win'] = h2h['goals_home'] > h2h['goals_away']
            h2h['draw'] = h2h['goals_home'] == h2h['goals_away']
            h2h['btts'] = (h2h['goals_home'] > 0) & (h2h['goals_away'] > 0)
            h2h['over_25'] = (h2h['goals_home'] + h2h['goals_away']) > 2.5
            
            df.loc[idx, 'h2h_matches'] = len(h2h)
            df.loc[idx, 'h2h_home_win_rate'] = h2h['home_win'].mean()
            df.loc[idx, 'h2h_draw_rate'] = h2h['draw'].mean()
            df.loc[idx, 'h2h_avg_goals_home'] = h2h['goals_home'].mean()
            df.loc[idx, 'h2h_avg_goals_away'] = h2h['goals_away'].mean()
            df.loc[idx, 'h2h_btts_rate'] = h2h['btts'].mean()
            df.loc[idx, 'h2h_over_25_rate'] = h2h['over_25'].mean()
        else:
            df.loc[idx, 'h2h_matches'] = 0
            df.loc[idx, 'h2h_home_win_rate'] = 0.5
            df.loc[idx, 'h2h_draw_rate'] = 0.2
            df.loc[idx, 'h2h_avg_goals_home'] = 1.5
            df.loc[idx, 'h2h_avg_goals_away'] = 1.5
            df.loc[idx, 'h2h_btts_rate'] = 0.5
            df.loc[idx, 'h2h_over_25_rate'] = 0.5
        
        # --- Home Team Form ---
        home_hist = team_histories[home]
        home_past = home_hist[(home_hist['Date'] < match_date) & (home_hist['is_home'] == True)]
        if len(home_past) > 0:
            recent = home_past.tail(5)
            df.loc[idx, 'home_avg_goals_scored'] = recent['team_goals'].mean()
            df.loc[idx, 'home_avg_goals_conceded'] = recent['opp_goals'].mean()
            wins = (recent['team_goals'] > recent['opp_goals']).sum()
            draws = (recent['team_goals'] == recent['opp_goals']).sum()
            df.loc[idx, 'home_points_per_game'] = (wins * 3 + draws) / len(recent)
            df.loc[idx, 'home_clean_sheet_rate'] = (recent['opp_goals'] == 0).mean()
            df.loc[idx, 'home_failed_to_score_rate'] = (recent['team_goals'] == 0).mean()
        else:
            df.loc[idx, 'home_avg_goals_scored'] = 1.5
            df.loc[idx, 'home_avg_goals_conceded'] = 1.5
            df.loc[idx, 'home_points_per_game'] = 1.0
            df.loc[idx, 'home_clean_sheet_rate'] = 0.3
            df.loc[idx, 'home_failed_to_score_rate'] = 0.3
        
        # --- Away Team Form ---
        away_hist = team_histories[away]
        away_past = away_hist[(away_hist['Date'] < match_date) & (away_hist['is_home'] == False)]
        if len(away_past) > 0:
            recent = away_past.tail(5)
            df.loc[idx, 'away_avg_goals_scored'] = recent['team_goals'].mean()
            df.loc[idx, 'away_avg_goals_conceded'] = recent['opp_goals'].mean()
            wins = (recent['team_goals'] > recent['opp_goals']).sum()
            draws = (recent['team_goals'] == recent['opp_goals']).sum()
            df.loc[idx, 'away_points_per_game'] = (wins * 3 + draws) / len(recent)
            df.loc[idx, 'away_clean_sheet_rate'] = (recent['opp_goals'] == 0).mean()
            df.loc[idx, 'away_failed_to_score_rate'] = (recent['team_goals'] == 0).mean()
        else:
            df.loc[idx, 'away_avg_goals_scored'] = 1.0
            df.loc[idx, 'away_avg_goals_conceded'] = 1.8
            df.loc[idx, 'away_points_per_game'] = 0.8
            df.loc[idx, 'away_clean_sheet_rate'] = 0.2
            df.loc[idx, 'away_failed_to_score_rate'] = 0.4
    
    return df, feature_cols

def build_value_alert_map(df, model_proba_home, feature_cols):
    """Build calibration map: model probability → historical market probability."""
    # Calculate market-implied home win probability (with overround adjustment)
    inv_h = 1 / df['B365H']
    inv_d = 1 / df['B365D']
    inv_a = 1 / df['B365A']
    overround = inv_h + inv_d + inv_a
    market_home_prob = inv_h / overround
    
    # Bin model probabilities
    bins = np.arange(0, 1.05, 0.05)  # 0.00-0.05, 0.05-0.10, ..., 0.95-1.00
    prob_bins = pd.cut(model_proba_home, bins=bins, include_lowest=True)
    
    # Create calibration map
    calibration_df = pd.DataFrame({
        'model_prob': model_proba_home,
        'market_prob': market_home_prob,
        'prob_bin': prob_bins
    }).dropna()
    
    value_map = calibration_df.groupby('prob_bin')['market_prob'].mean()
    return value_map

def train_multi_market_models():
    """Train HDA, GG, Over/Under 2.5 models + value alert map."""
    # Load data
    data_path = DATA_DIR / "cleaned_historical_data.csv"
    df = pd.read_csv(
        data_path,
        low_memory=False,
        dtype={'FTHG': 'Int64', 'FTAG': 'Int64', 'FTR': 'string'}
    )
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A']).copy()
    
    print(f"Loaded {len(df)} matches. Computing features...")
    df, feature_cols = compute_vectorized_features(df)
    
    # Prepare features
    X = df[feature_cols].fillna(0)
    
    # Time-series CV
    tscv = TimeSeriesSplit(n_splits=3)
    
    # --- 1. Train HDA Model ---
    print("\n" + "="*50)
    print("TRAINING HDA MODEL (Home/Draw/Away)")
    print("="*50)
    y_hda = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    hda_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_hda.iloc[train_idx], y_hda.iloc[val_idx]
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='multi:softprob',
            num_class=3
        )
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)
        y_pred = model.predict(X_val)
        
        logloss = log_loss(y_val, y_pred_proba)
        accuracy = accuracy_score(y_val, y_pred)
        hda_scores.append((logloss, accuracy))
        print(f"Fold {fold+1}: LogLoss={logloss:.4f}, Accuracy={accuracy:.2%}")
    
    final_hda = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='multi:softprob',
        num_class=3
    )
    final_hda.fit(X, y_hda)
    joblib.dump(final_hda, MODEL_DIR / "football_model_hda.pkl")
    
    # --- Build Value Alert Map ---
    print("\n" + "="*50)
    print("BUILDING VALUE ALERT CALIBRATION MAP")
    print("="*50)
    model_proba_home = final_hda.predict_proba(X)[:, 0]  # Home win probabilities
    value_alert_map = build_value_alert_map(df, model_proba_home, feature_cols)
    joblib.dump(value_alert_map, MODEL_DIR / "value_alert_map.pkl")
    print("✅ Value alert map saved!")
    
    # --- 2. Train GG Model ---
    print("\n" + "="*50)
    print("TRAINING GG MODEL (Both Teams to Score)")
    print("="*50)
    y_gg = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
    gg_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_gg.iloc[train_idx], y_gg.iloc[val_idx]
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='binary:logistic'
        )
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
        
        brier = brier_score_loss(y_val, y_pred_proba)
        accuracy = accuracy_score(y_val, y_pred)
        gg_scores.append((brier, accuracy))
        print(f"Fold {fold+1}: Brier={brier:.4f}, Accuracy={accuracy:.2%}")
    
    final_gg = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='binary:logistic'
    )
    final_gg.fit(X, y_gg)
    joblib.dump(final_gg, MODEL_DIR / "football_model_gg.pkl")
    
    # --- 3. Train Over/Under 2.5 Model ---
    print("\n" + "="*50)
    print("TRAINING OVER/UNDER 2.5 MODEL")
    print("="*50)
    y_over25 = ((df['FTHG'] + df['FTAG']) > 2.5).astype(int)
    over25_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_over25.iloc[train_idx], y_over25.iloc[val_idx]
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='binary:logistic'
        )
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
        
        brier = brier_score_loss(y_val, y_pred_proba)
        accuracy = accuracy_score(y_val, y_pred)
        over25_scores.append((brier, accuracy))
        print(f"Fold {fold+1}: Brier={brier:.4f}, Accuracy={accuracy:.2%}")
    
    final_over25 = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='binary:logistic'
    )
    final_over25.fit(X, y_over25)
    joblib.dump(final_over25, MODEL_DIR / "football_model_over25.pkl")
    
    # Save feature columns
    joblib.dump(feature_cols, MODEL_DIR / "feature_columns.pkl")
    
    # Final summary
    print("\n" + "="*60)
    print("✅ ALL MODELS + VALUE ALERT MAP TRAINED AND SAVED")
    print("="*60)
    print(f"HDA - Avg Accuracy: {np.mean([s[1] for s in hda_scores]):.2%}")
    print(f"GG  - Avg Accuracy: {np.mean([s[1] for s in gg_scores]):.2%}")
    print(f"O/U - Avg Accuracy: {np.mean([s[1] for s in over25_scores]):.2%}")
    print(f"\nModels saved to: {MODEL_DIR}")

if __name__ == "__main__":
    train_multi_market_models()