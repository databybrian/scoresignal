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
        print("⚠️  Using fallback compute_match_features")
        return {}
    
    def send_telegram_message(message):
        print(f"📱 Telegram message (simulated): {message}")
    
    def generate_daily_summary(tips):
        return "Summary unavailable - fallback mode"
    
    def create_summary_message(summary):
        return "Daily summary unavailable"
    
    def get_next_tips_time(now):
        return now + timedelta(hours=1)
    
    def format_time(dt):
        return dt.strftime("%H:%M")
    
    def get_now_nairobi():
        return datetime.now(pytz.timezone('Africa/Nairobi'))

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
        # Try both 'Date' and 'date' columns
        if 'Date' in fixtures.columns:
            fixtures['Date'] = pd.to_datetime(fixtures['Date'], errors='coerce')
        elif 'date' in fixtures.columns:
            fixtures['Date'] = pd.to_datetime(fixtures['date'], errors='coerce')
        else:
            fixtures['Date'] = pd.NaT
            
        today = pd.Timestamp.now(tz='UTC').normalize()
        todays_fixtures = fixtures[fixtures['Date'].dt.date == today.date()].copy()
        print(f"✅ Loaded {len(todays_fixtures)} fixtures for today")
        return todays_fixtures
    except Exception as e:
        print(f"❌ Failed to load fixtures: {e}")
        return pd.DataFrame()

def parse_match_datetime(row):
    """Parse match date + time into UTC datetime."""
    if pd.isna(row['time']) or row['time'] == '' or row['time'] == 'TBD':
        return pd.NaT
    try:
        dt_str = f"{row['Date'].strftime('%Y-%m-%d')} {row['time']}"
        return pd.to_datetime(dt_str, format='%Y-%m-%d %H:%M', utc=True)
    except:
        return pd.NaT

def format_times_for_message(match_utc):
    """Return formatted UTC and EAT times."""
    if pd.isna(match_utc):
        return "TBD", "TBD"
    utc_time = match_utc.tz_convert('UTC').strftime('%H:%M UTC')
    local_time = match_utc.tz_convert('Etc/GMT-3').strftime('%H:%M EAT')
    return utc_time, local_time

def has_been_predicted(home, away, date_str):
    """Check if match already predicted — handles missing/wrong columns."""
    if not PREDICTION_LOG.exists():
        return False
    
    try:
        log_df = pd.read_csv(PREDICTION_LOG)
        
        # If log is empty or missing required columns, treat as "not predicted"
        required_cols = {'home_team', 'away_team', 'date'}
        if not required_cols.issubset(log_df.columns):
            print("⚠️  Prediction log missing required columns — resetting log")
            PREDICTION_LOG.unlink()  # Delete malformed log
            return False
        
        return ((log_df['home_team'] == home) & 
                (log_df['away_team'] == away) & 
                (log_df['date'] == date_str)).any()
                
    except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError):
        print("⚠️  Prediction log corrupted or empty — resetting")
        if PREDICTION_LOG.exists():
            PREDICTION_LOG.unlink()
        return False

def log_prediction(home, away, date_str):
    """Log predicted match — ensures correct header."""
    log_entry = pd.DataFrame([{
        'home_team': home,
        'away_team': away,
        'date': date_str
    }])
    
    # Always write with header if file doesn't exist
    if not PREDICTION_LOG.exists():
        log_entry.to_csv(PREDICTION_LOG, index=False)
    else:
        # Append without header
        log_entry.to_csv(PREDICTION_LOG, mode='a', header=False, index=False)

def get_market_baseline(my_prob, value_map):
    """Get historical market probability."""
    bins = np.arange(0, 1.05, 0.05)
    bin_idx = np.digitize([my_prob], bins) - 1
    if bin_idx[0] >= len(bins) - 1:
        bin_idx[0] = len(bins) - 2
    bin_key = pd.Interval(bins[bin_idx[0]], bins[bin_idx[0] + 1], closed='right')
    return value_map.get(bin_key, my_prob)

def should_send_tip(hda_proba, gg_proba, over25_proba, edge):
    """
    Determine if match qualifies as a high-confidence tip.
    Only send if at least one market shows strong signal.
    """
    home_win, draw, away_win = hda_proba
    
    # HDA: Clear favorite
    hda_clear = (home_win >= 0.55) or (away_win >= 0.55) or (draw >= 0.40)
    
    # GG: Decisive probability
    gg_clear = (gg_proba >= 0.65) or (gg_proba <= 0.35)
    
    # Over/Under: Decisive probability
    ou_clear = (over25_proba >= 0.65) or (over25_proba <= 0.35)
    
    # Value edge is strong
    strong_edge = edge >= 0.02
    
    # Send if any condition is met
    return hda_clear or gg_clear or ou_clear or strong_edge

def build_prediction_message(match_row, historical_df):
    """Build a rich, high-confidence Telegram prediction message. Returns None if low-confidence."""
    if not MODELS:
        return "🔧 Models not available - predictions temporarily disabled"
    
    home = match_row['home_team']
    away = match_row['away_team']
    match_date = match_row['Date']
    league_name = match_row.get('league_name', 'Unknown League')
    
    # Compute features using only historical data
    features = compute_match_features(
        historical_df=historical_df,
        home_team=home,
        away_team=away,
        match_date=match_date,
        league_code=match_row.get('league_code', 'E0')
    )
    X = pd.DataFrame([features])[MODELS['feature_cols']]
    
    # Get predictions
    hda_proba = MODELS['hda'].predict_proba(X)[0]
    gg_proba = MODELS['gg'].predict_proba(X)[0][1]
    over25_proba = MODELS['over25'].predict_proba(X)[0][1]
    
    # Calculate value edge vs historical market baseline
    home_win_prob = hda_proba[0]
    market_baseline = get_market_baseline(home_win_prob, MODELS['value_map'])
    edge = home_win_prob - market_baseline
    
    # Apply tip filtering: only send high-confidence signals
    if not should_send_tip(hda_proba, gg_proba, over25_proba, edge):
        return None
    
    # Determine best tip label
    home_win, draw, away_win = hda_proba
    best_tip = ""
    
    if home_win >= 0.55:
        best_tip = "🟢 HOME (Strong Home Favorite)"
    elif away_win >= 0.55:
        best_tip = "🔵 AWAY (Strong Away Win)"
    elif draw >= 0.40:
        best_tip = "🟡 DRAW (High Draw Probability)"
    elif gg_proba >= 0.65:
        best_tip = "🟢 GG (Both Teams to Score)"
    elif gg_proba <= 0.35:
        best_tip = "🔴 NG (No Goals Expected)"
    elif over25_proba >= 0.65:
        best_tip = "🟢 OVER 2.5 (High-Scoring Game)"
    elif over25_proba <= 0.35:
        best_tip = "🔴 UNDER 2.5 (Low-Scoring Game)"
    else:
        best_tip = "💡 Mixed Signals"

    # Format kickoff time
    match_utc = parse_match_datetime(match_row)
    utc_str, local_str = format_times_for_message(match_utc)
    time_str = f"🕒 Kickoff: {utc_str} | {local_str}" if not pd.isna(match_utc) else "🕒 Kickoff: TBD (today | tomorrow)"
    
    # Build final message
    separator = "─" * 30
    message = (
        "TIP\n"
        f"🏆{league_name}\n"
        f"⏰{time_str}\n"
        f"🏠{home} vs {away}\n"
        f"💡 {best_tip}\n\n"
        f"🏠 Home: {hda_proba[0]:.0%} | 🤝 Draw: {hda_proba[1]:.0%} | 🚌 Away: {hda_proba[2]:.0%}\n"
        f"⚽⚽ Both Teams to Score: {gg_proba:.0%}\n"
        f"⚽⚽⚽ Over 2.5 Goals: {over25_proba:.0%}\n"
    )
    
    # Add value alert if edge is strong (≥2%)
    if edge > 0.02:
        message += (
            f"\n\n⚠️ HIGH VALUE ALERT!\n"
            f"Model: {home_win_prob:.0%} Home | Market Baseline: ~{market_baseline:.0%}\n"
            f"→ {edge:+.0%} edge (Our Model vs Historical Average)"
        )
    
    # Responsible gambling footer
    message += (f"{separator}\n" 
                f"💡 Bet responsibly || scoresignal")
    
    return message

def extract_hda_probabilities(match_row, historical_df):
    """Safely extract HDA probabilities for LLM summary."""
    try:
        home = match_row['home_team']
        away = match_row['away_team']
        match_date = match_row['Date']
        
        features = compute_match_features(
            historical_df=historical_df,
            home_team=home,
            away_team=away,
            match_date=match_date,
            league_code=match_row.get('league_code', 'E0')
        )
        X = pd.DataFrame([features])[MODELS['feature_cols']]
        return MODELS['hda'].predict_proba(X)[0]
    except Exception as e:
        print(f"⚠️  Could not extract probabilities: {e}")
        return [0.33, 0.33, 0.33]

def extract_best_tip_label(match_row, historical_df):
    """Extract best tip label for LLM summary."""
    try:
        hda_proba = extract_hda_probabilities(match_row, historical_df)
        home_win, draw, away_win = hda_proba
        
        if home_win >= 0.55:
            return "Strong Home Favorite"
        elif away_win >= 0.55:
            return "Strong Away Win"
        elif draw >= 0.40:
            return "High Draw Probability"
        else:
            # Get secondary markets
            home = match_row['home_team']
            away = match_row['away_team']
            match_date = match_row['Date']
            
            features = compute_match_features(
                historical_df=historical_df,
                home_team=home,
                away_team=away,
                match_date=match_date,
                league_code=match_row.get('league_code', 'E0')
            )
            X = pd.DataFrame([features])[MODELS['feature_cols']]
            gg_proba = MODELS['gg'].predict_proba(X)[0][1]
            over25_proba = MODELS['over25'].predict_proba(X)[0][1]
            
            if gg_proba >= 0.65:
                return "Both Teams to Score"
            elif gg_proba <= 0.35:
                return "No Goals Expected"
            elif over25_proba >= 0.65:
                return "Over 2.5 Goals"
            elif over25_proba <= 0.35:
                return "Under 2.5 Goals"
            else:
                return "Mixed Signals"
    except Exception as e:
        print(f"⚠️  Could not extract best tip: {e}")
        return "Mixed Signals"

def run_prediction_tier(tier_name, fixtures_subset, historical_df, collect_for_summary=False):
    """
    Run prediction for a specific tier.
    """
    if not MODELS:
        print("⏭️  Skipping predictions - models not loaded")
        return []
        
    if fixtures_subset.empty:
        print(f"⏭️  No matches for {tier_name}")
        return []

    sent_tips = []
    print(f"🎯 Processing {len(fixtures_subset)} matches for {tier_name}...")

    for _, match in fixtures_subset.iterrows():
        home = match['home_team']
        away = match['away_team']
        date_str = match['Date'].strftime('%Y-%m-%d')

        # Skip if already predicted
        if has_been_predicted(home, away, date_str):
            print(f"⏭️  Skipped (already predicted): {home} vs {away}")
            continue

        try:
            message = build_prediction_message(match, historical_df)

            # Skip low-confidence tips or model errors
            if message is None or "Models not available" in message:
                print(f"⏭️  Skipped (low confidence or model issue): {home} vs {away}")
                continue

            # Send to Telegram
            try:
                send_telegram_message(message)
                log_prediction(home, away, date_str)
                print(f"✅ Sent prediction: {home} vs {away}")

                # Collect tip data for LLM summary if requested
                if collect_for_summary:
                    hda_proba = extract_hda_probabilities(match, historical_df)
                    if hda_proba is not None:
                        sent_tips.append({
                            'home': home,
                            'away': away,
                            'league': match.get('league_name', 'Unknown League'),
                            'best_tip': extract_best_tip_label(match, historical_df),
                            'home_prob': hda_proba[0],
                            'draw_prob': hda_proba[1],
                            'away_prob': hda_proba[2]
                        })

            except Exception as telegram_error:
                print(f"❌ Telegram API error for {home} vs {away}: {telegram_error}")

            # Be kind to APIs
            time.sleep(1)

        except Exception as e:
            print(f"❌ Error predicting {home} vs {away}: {e}")
            continue

    print(f"📊 {len(sent_tips)} tips collected for {tier_name}")
    return sent_tips

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
            "🔐 For advanced modelling tips, support via MPESA:\n"
            "📌 TILL: 9105695\n\n"
            "Thank you for your support! 💙\n"
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

    # Categorize matches
    no_time = fixtures[pd.isna(fixtures['time']) | (fixtures['time'] == '') | (fixtures['time'] == 'TBD')]
    daytime = fixtures[
        (fixtures['time'].notna()) &
        (fixtures['time'] != '') &
        (fixtures['time'] != 'TBD') &
        (pd.to_datetime(fixtures['time'], format='%H:%M', errors='coerce').dt.hour.between(12, 18))
    ]
    evening = fixtures[
        (fixtures['time'].notna()) &
        (fixtures['time'] != '') &
        (fixtures['time'] != 'TBD') &
        (pd.to_datetime(fixtures['time'], format='%H:%M', errors='coerce').dt.hour >= 19)
    ]

    all_sent_tips = []
    now = get_now_nairobi()
    current_hour = now.hour
    print(f"🕒 Current Nairobi time: {now.strftime('%Y-%m-%d %H:%M')}")

    # Tier logic (using Nairobi time)
    if 7 <= current_hour <= 9:  # Tier 1
        tier_name = "Tier 1 (No-time)"
        print("🕐 Running Tier 1: No-time matches")
        all_sent_tips.extend(run_prediction_tier(tier_name, no_time, historical_df, collect_for_summary=True))
        last_tier = tier_name

    elif 12 <= current_hour <= 14:  # Tier 2 (shifted to 12–14 Nairobi)
        tier_name = "Tier 2 (Daytime)"
        print("🕐 Running Tier 2: Daytime matches (12–18)")
        all_sent_tips.extend(run_prediction_tier(tier_name, daytime, historical_df, collect_for_summary=True))
        last_tier = tier_name

    elif 16 <= current_hour <= 18:  # Tier 3
        tier_name = "Tier 3 (Evening)"
        print("🕐 Running Tier 3: Evening matches (19+)")
        all_sent_tips.extend(run_prediction_tier(tier_name, evening, historical_df, collect_for_summary=True))
        last_tier = tier_name

    else:
        print(f"🕒 Outside prediction windows (current hour: {current_hour})")

        # Only send "No tips" once per tier
        if last_tier != "Outside":
            next_time = get_next_tips_time(now)
            msg = (
                f"📭 No high-confidence tips to summarize.\n\n"
                f"⏭️ Next predictions will be sent at: {format_time(next_time)} Nairobi time"
            )
            send_telegram_message(msg)
            print(msg)
            last_tier = "Outside"

    # Generate and send LLM summary if tips exist
    if all_sent_tips:
        try:
            print("🧠 Generating LLM summary...")
            summary = generate_daily_summary(all_sent_tips)
            summary_message = create_summary_message(summary)
            send_telegram_message(summary_message)
            print("✅ LLM summary sent!")
        except Exception as e:
            print(f"❌ LLM summary failed (continuing): {e}")
    else:
        print("📭 No high-confidence tips to summarize")

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