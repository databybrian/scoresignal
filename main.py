# main.py
import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import joblib

# Add project root to Python path FIRST
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

print(f"🔍 Project root: {PROJECT_ROOT}")
print(f"🔍 Python path: {sys.path}")

# Now import other modules with error handling
try:
    from features.h2h_form import compute_match_features
    from bot.telegram_bot import send_telegram_message
    from bot.llm_summary import generate_daily_summary, create_summary_message
    from bot.time_utils import get_next_tips_time, format_time, get_now_nairobi
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    # Create minimal fallbacks to prevent crashes
    def compute_match_features(*args, **kwargs):
        return {}
    def send_telegram_message(*args, **kwargs):
        print("📱 Telegram message (simulated):", kwargs.get('message', 'No message'))
    def get_now_nairobi():
        return datetime.now(pytz.timezone('Africa/Nairobi'))
    # Add other necessary fallbacks...

# Paths
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
PREDICTION_LOG = DATA_DIR / "prediction_log.csv"

def safe_load_models():
    """Safely load models with error handling"""
    models = {}
    try:
        print("🚀 Loading models...")
        models['hda'] = joblib.load(MODEL_DIR / "football_model_hda.pkl")
        models['gg'] = joblib.load(MODEL_DIR / "football_model_gg.pkl")
        models['over25'] = joblib.load(MODEL_DIR / "football_model_over25.pkl")
        models['value_map'] = joblib.load(MODEL_DIR / "value_alert_map.pkl")
        models['feature_cols'] = joblib.load(MODEL_DIR / "feature_columns.pkl")
        print("✅ Models loaded!")
        return models
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return None

MODELS = safe_load_models()
if MODELS is None:
    print("🆘 Continuing without models - predictions will be disabled")
    MODELS = {}

def load_historical_data():
    """Load clean historical data for feature computation."""
    try:
        df = pd.read_csv(
            DATA_DIR / "cleaned_historical_data.csv",
            low_memory=False,
            dtype={'FTHG': 'Int64', 'FTAG': 'Int64'}
        )
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        print(f"✅ Loaded historical data: {len(df)} matches")
        return df
    except Exception as e:
        print(f"❌ Failed to load historical data: {e}")
        return pd.DataFrame()

def load_todays_fixtures():
    """Load today's fixtures with error handling."""
    try:
        fixtures = pd.read_csv(DATA_DIR / "fixtures_data.csv")
        fixtures['Date'] = pd.to_datetime(fixtures['date'], errors='coerce')  # Use 'date' column
        today = pd.Timestamp.now(tz='UTC').normalize()
        todays_fixtures = fixtures[fixtures['Date'].dt.date == today.date()].copy()
        print(f"✅ Loaded {len(todays_fixtures)} fixtures for today")
        return todays_fixtures
    except Exception as e:
        print(f"❌ Failed to load fixtures: {e}")
        return pd.DataFrame()

def safe_fetch_fixtures():
    """Safely fetch fixtures with multiple fallback strategies."""
    fixtures_file = DATA_DIR / "fixtures_data.csv"
    
    if fixtures_file.exists():
        print("✅ Fixtures file exists")
        return True
        
    print("🔄 fixtures_data.csv not found - fetching live fixtures...")
    
    # Try multiple import strategies
    import_strategies = [
        lambda: __import__('src.fetch_fixtures_live', fromlist=['fetch_and_save_fixtures']).fetch_and_save_fixtures,
        lambda: __import__('fetch_fixtures_live', fromlist=['fetch_and_save_fixtures']).fetch_and_save_fixtures,
    ]
    
    for i, strategy in enumerate(import_strategies):
        try:
            fetch_function = strategy()
            fetch_function(str(fixtures_file))
            print("✅ Fixtures fetched successfully")
            return True
        except Exception as e:
            print(f"❌ Strategy {i+1} failed: {e}")
            continue
    
    # Final fallback: create empty fixtures file
    print("🆘 All import strategies failed - creating empty fixtures file")
    empty_df = pd.DataFrame(columns=[
        'round', 'date', 'time', 'home_team', 'away_team',
        'home_score', 'away_score', 'league_key', 'league_name', 'season'
    ])
    empty_df.to_csv(fixtures_file, index=False)
    return False

# ... (keep all your existing functions like parse_match_datetime, format_times_for_message, 
# has_been_predicted, log_prediction, get_market_baseline, should_send_tip, etc.)
# They can remain exactly the same since they're well-written

def build_prediction_message(match_row, historical_df):
    """Build prediction message with model availability check."""
    if not MODELS:
        return "🔧 Models not available - predictions disabled"
    
    # Your existing build_prediction_message logic here
    # ... (keep the same implementation)

def run_prediction_tier(tier_name, fixtures_subset, historical_df, collect_for_summary=False):
    """Run prediction tier with model availability check."""
    if not MODELS:
        print("⏭️  Skipping predictions - models not loaded")
        return []
    
    if fixtures_subset.empty:
        print(f"⏭️  No matches for {tier_name}")
        return []

    # Your existing run_prediction_tier logic here
    # ... (keep the same implementation)

last_tier = None

def main():
    global last_tier
    
    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Safely fetch fixtures
    if not safe_fetch_fixtures():
        print("❌ Could not fetch fixtures - exiting")
        return
    
    # Send header only if models are available
    if MODELS:
        header = (
            "scoresignal uses 10+ years of historical data and machine learning "
            "to curate probabilistic football tips.\n\n"
            "⚠️ Disclaimer: Predictions are probabilistic from our model, not guarantees. "
            "For support MPESA Till: 9105695. \n\n"
            "scoresignal || Data-driven football tips"
        )
        send_telegram_message(header)
    else:
        send_telegram_message("🔧 System starting up - models loading...")
    
    # Load data
    historical_df = load_historical_data()
    fixtures = load_todays_fixtures()

    if fixtures.empty:
        print("😴 No fixtures for today")
        send_telegram_message("📭 No fixtures found for today")
        return

    # Your existing tier logic here
    # ... (keep the same implementation)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"💥 Critical error in main: {e}")
        import traceback
        traceback.print_exc()
        # Try to send error notification
        try:
            send_telegram_message(f"🚨 Bot crashed: {str(e)[:100]}...")
        except:
            pass