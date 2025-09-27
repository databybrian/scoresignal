# fixture_data.py

import pandas as pd
import os
from typing import Optional

# Path to your static fixtures file
FIXTURES_FILE = "fixtures_data.csv"  

def load_fixtures(filepath: str = "fixtures_data.csv") -> pd.DataFrame:
    """
    Load fixtures from the pre-downloaded CSV file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Fixtures file not found: {filepath}\n"
            "💡 To generate this file:\n"
            "1. Run: python update_fixtures.py\n"
            "2. This will fetch fresh data from openfootball and save it locally."
        )
    return pd.read_csv(filepath)
    
    # Ensure essential columns exist
    required_cols = ['league_key', 'home_team', 'away_team', 'date']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in fixtures file: {missing_cols}")
    
    print(f"✅ Loaded {len(df)} fixtures from {filepath}")
    return df

if __name__ == "__main__":
    # Keep this for testing - very useful!
    try:
        df = load_fixtures()
        print(f"✅ Loaded {len(df)} fixtures")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Leagues: {df['league_key'].nunique()}")
        print(f"With times: {df['time'].notna().sum()}/{len(df)}")
        print("\nSample:")
        print(df.head())
    except FileNotFoundError as e:
        print(f"❌ {e}")
