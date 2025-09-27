# scripts/test_features.py
import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import pandas as pd
from features.h2h_form import compute_match_features

def test_feature_extraction():
    """Test feature extraction on a real match from your dataset."""
    # Load clean historical data
    data_path = SCRIPT_DIR / "data" / "cleaned_historical_data.csv"
    df = pd.read_csv(
        data_path,
        low_memory=False,
        dtype={'FTHG': 'Int64', 'FTAG': 'Int64'}
    )
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Pick a match from the middle of the dataset (not too early, not too recent)
    test_match = df.iloc[len(df) // 2]
    
    print(f"🧪 Testing features for:")
    print(f"   {test_match['HomeTeam']} vs {test_match['AwayTeam']}")
    print(f"   Date: {test_match['Date'].date()}")
    print(f"   League: {test_match['league_code']}")
    print(f"   Actual result: {test_match['FTHG']}-{test_match['FTAG']} ({test_match['FTR']})")
    
    # Compute features using ONLY prior matches
    features = compute_match_features(
        historical_df=df,
        home_team=test_match['HomeTeam'],
        away_team=test_match['AwayTeam'],
        match_date=test_match['Date'],
        league_code=test_match['league_code']
    )
    
    print(f"\n✅ Extracted {len(features)} features:")
    for key, value in sorted(features.items()):
        print(f"   {key}: {value:.3f}")
    
    # Basic validation
    assert 0 <= features['h2h_home_win_rate'] <= 1, "H2H win rate out of bounds"
    assert features['home_avg_goals_scored'] >= 0, "Negative goals"
    assert features['away_points_per_game'] >= 0, "Negative points"
    
    print(f"\n🎉 All validations passed!")

if __name__ == "__main__":
    test_feature_extraction()