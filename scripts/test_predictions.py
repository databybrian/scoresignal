# scripts/test_predictions.py
import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import pandas as pd
import numpy as np
import joblib
from features.h2h_form import compute_match_features

# Paths
DATA_DIR = SCRIPT_DIR / "data"
MODEL_DIR = SCRIPT_DIR / "model"

def load_models():
    """Load all trained models and calibration map."""
    models = {
        'hda': joblib.load(MODEL_DIR / "football_model_hda.pkl"),
        'gg': joblib.load(MODEL_DIR / "football_model_gg.pkl"),
        'over25': joblib.load(MODEL_DIR / "football_model_over25.pkl"),
        'value_map': joblib.load(MODEL_DIR / "value_alert_map.pkl"),
        'feature_cols': joblib.load(MODEL_DIR / "feature_columns.pkl")
    }
    return models

def get_market_baseline(my_prob, value_map):
    """Get historical market probability for given model probability."""
    bins = np.arange(0, 1.05, 0.05)
    bin_idx = np.digitize([my_prob], bins) - 1
    if bin_idx[0] >= len(bins) - 1:
        bin_idx[0] = len(bins) - 2
    bin_key = pd.Interval(bins[bin_idx[0]], bins[bin_idx[0] + 1], closed='right')
    return value_map.get(bin_key, my_prob)

def test_recent_matches(n_matches=5):
    """Test predictions on n most recent completed matches."""
    # Load data
    df = pd.read_csv(
        DATA_DIR / "cleaned_historical_data.csv",
        low_memory=False,
        dtype={'FTHG': 'Int64', 'FTAG': 'Int64', 'FTR': 'string'}
    )
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['FTHG', 'FTAG', 'FTR']).sort_values('Date', ascending=False)
    
    # Get recent completed matches
    recent = df.head(n_matches).copy()
    
    # Load models
    models = load_models()
    
    print("🔍 VALIDATING PREDICTIONS ON RECENT MATCHES")
    print("=" * 80)
    
    for idx, row in recent.iterrows():
        print(f"\nMatch: {row['HomeTeam']} vs {row['AwayTeam']} | {row['Date'].date()} | League: {row['league_code']}")
        print(f"Actual: {row['FTHG']}-{row['FTAG']} ({row['FTR']}) | GG: {'Yes' if (row['FTHG']>0 and row['FTAG']>0) else 'No'} | O/U 2.5: {'Over' if (row['FTHG']+row['FTAG'])>2.5 else 'Under'}")
        
        # Compute features (as if predicting before match)
        features = compute_match_features(
            historical_df=df,
            home_team=row['HomeTeam'],
            away_team=row['AwayTeam'],
            match_date=row['Date'],
            league_code=row['league_code']
        )
        X = pd.DataFrame([features])[models['feature_cols']]
        
        # HDA Prediction
        hda_proba = models['hda'].predict_proba(X)[0]
        hda_pred = models['hda'].classes_[np.argmax(hda_proba)]
        hda_map = {0: 'H', 1: 'D', 2: 'A'}
        hda_pred_str = hda_map[hda_pred]
        
        # GG Prediction
        gg_proba = models['gg'].predict_proba(X)[0][1]  # Prob of GG=Yes
        gg_pred = "Yes" if gg_proba > 0.5 else "No"
        
        # Over/Under 2.5 Prediction
        over25_proba = models['over25'].predict_proba(X)[0][1]
        over25_pred = "Over" if over25_proba > 0.5 else "Under"
        
        # Value Alert (HDA only)
        home_win_prob = hda_proba[0]  # Probability of Home win
        market_baseline = get_market_baseline(home_win_prob, models['value_map'])
        edge = home_win_prob - market_baseline
        
        print(f"Predicted HDA: H({hda_proba[0]:.1%}) D({hda_proba[1]:.1%}) A({hda_proba[2]:.1%}) → {hda_pred_str}")
        print(f"Predicted GG: {gg_proba:.1%} → {gg_pred}")
        print(f"Predicted O/U 2.5: {over25_proba:.1%} → {over25_pred}")
        print(f"Value Alert: Model {home_win_prob:.0%} vs Market ~{market_baseline:.0%} → Edge: {edge:+.0%}")
        
        # Accuracy check
        hda_correct = (hda_pred_str == row['FTR'])
        gg_correct = (gg_pred == ('Yes' if (row['FTHG']>0 and row['FTAG']>0) else 'No'))
        ou_correct = (over25_pred == ('Over' if (row['FTHG']+row['FTAG'])>2.5 else 'Under'))
        print(f"✅ Correct: HDA={hda_correct} | GG={gg_correct} | O/U={ou_correct}")

if __name__ == "__main__":
    test_recent_matches(n_matches=5)