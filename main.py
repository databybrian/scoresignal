# main.py
import argparse
import sys
from pathlib import Path

# Add project root to Python path FIRST
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import time
import os
import shutil
import traceback
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import joblib

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"src path: {PROJECT_ROOT / 'src'}")

# -------------------
# Import project modules (with safe fallbacks)
# -------------------
try:
    from features.h2h_form import compute_match_features
    from bot.telegram_bot import send_telegram_message
    from bot.llm_summary import generate_daily_summary, create_summary_message
    from bot.time_utils import get_next_tips_time, format_time, get_now_nairobi
    from bot.chat_manager import ensure_table

except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    # Minimal fallbacks (safe)
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

# -------------------
# Paths and globals
# -------------------
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
PREDICTION_LOG = DATA_DIR / "prediction_log.csv"
PREDICTION_LOG_BACKUP_DIR = DATA_DIR / "prediction_log_backups"
EXECUTION_LOCK = PROJECT_ROOT / ".execution.lock"

# ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_LOG_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# Track daily notification state
DAILY_FLAG = PROJECT_ROOT / ".daily_notified"

# -------------------
# Import shared data pipeline functions
# -------------------
from src.data_pipeline import ensure_historical_data_exists, save_all_current_tables

# -------------------
# Cross-Platform Execution Lock
# -------------------
def acquire_execution_lock():
    """Prevent multiple instances from running simultaneously - cross-platform"""
    try:
        # Check if lock file exists and is recent (less than 30 minutes old)
        if EXECUTION_LOCK.exists():
            lock_age = (datetime.now() - datetime.fromtimestamp(EXECUTION_LOCK.stat().st_mtime)).total_seconds() / 60
            if lock_age < 30:  # Lock is less than 30 minutes old
                print("⏭️ Another instance is already running (lock file exists) - skipping")
                return False
            else:
                # Stale lock - remove it
                print("🔄 Removing stale execution lock")
                EXECUTION_LOCK.unlink(missing_ok=True)
        
        # Create new lock file
        EXECUTION_LOCK.write_text(f"Locked at: {datetime.now().isoformat()}")
        return True
        
    except Exception as e:
        print(f"⚠️ Lock acquisition failed: {e}")
        return False

def release_execution_lock():
    """Release the execution lock"""
    try:
        if EXECUTION_LOCK.exists():
            EXECUTION_LOCK.unlink()
    except Exception as e:
        print(f"⚠️ Lock release failed: {e}")

# -------------------
# New Scheduling Logic
# -------------------
def should_run_tips():
    """
    Use 15-minute windows around exact target times
    """
    nairobi_tz = pytz.timezone('Africa/Nairobi')
    now = datetime.now(nairobi_tz)
    current_time = now.time()
    current_weekday = now.weekday()  # 0=Monday, 6=Sunday
    
    # Define time windows (15-minute windows)
    morning_window_start = datetime.time(10, 0)   # 10:00 AM
    morning_window_end = datetime.time(10, 15)    # 10:15 AM
    
    afternoon_window_start = datetime.time(16, 0) # 4:00 PM  
    afternoon_window_end = datetime.time(16, 15)  # 4:15 PM
    
    # Weekday schedule
    if current_weekday < 5:  # Monday-Friday
        if morning_window_start <= current_time <= morning_window_end:
            return True, 'morning'
    
    # Weekend schedule
    else:  # Saturday-Sunday
        if morning_window_start <= current_time <= morning_window_end:
            return True, 'morning'
        elif afternoon_window_start <= current_time <= afternoon_window_end:
            return True, 'afternoon'
    
    return False, None

def reset_daily_flag_if_new_day():
    """Remove DAILY_FLAG if its date is older than today."""
    if DAILY_FLAG.exists():
        try:
            ts = datetime.fromtimestamp(DAILY_FLAG.stat().st_mtime).date()
            if ts < datetime.now().date():
                DAILY_FLAG.unlink()
                print("🔄 Daily flag reset for new day")
        except Exception:
            try:
                DAILY_FLAG.unlink()
            except Exception:
                pass

# -------------------
# Ensure historical data exists
# -------------------
historical_ready = ensure_historical_data_exists()

# -------------------
# Load Historical Data
# -------------------
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
        traceback.print_exc()
        return pd.DataFrame()  # Return empty, don't try to generate

# -------------------
# Model and Data Loading
# -------------------
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
        traceback.print_exc()
        return None

MODELS = safe_load_models()
if MODELS is None:
    print("🆘 Continuing without models - predictions will be disabled")
    MODELS = {}

def backup_prediction_log(max_keep: int = 7):
    """Backup prediction log with rotation"""
    try:
        if PREDICTION_LOG.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = PREDICTION_LOG_BACKUP_DIR / f"prediction_log_{ts}.csv"
            shutil.copy(PREDICTION_LOG, dest)
            backups = sorted(
                PREDICTION_LOG_BACKUP_DIR.glob("prediction_log_*.csv"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            for old in backups[max_keep:]:
                try:
                    old.unlink()
                except Exception:
                    pass
            print(f"📦 Backed up prediction log -> {dest}")
    except Exception as e:
        print(f"⚠️ Failed to backup prediction log: {e}")


def safe_fetch_fixtures():
    """Safely fetch fixtures; refresh weekly"""
    fixtures_file = DATA_DIR / "fixtures_data.csv"

    if fixtures_file.exists() and not needs_refresh(fixtures_file, days=7):
        print("✅ Fixtures file is recent - no refresh needed")
        return True

    print("🔄 fixtures_data.csv missing/old - fetching live fixtures...")
    try:
        from src.fetch_fixtures_live import fetch_and_save_fixtures
        fetch_and_save_fixtures(str(fixtures_file))
        print("✅ Fixtures fetched successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to fetch fixtures: {e}")
        empty_df = pd.DataFrame(columns=[
            'round', 'date', 'time', 'home_team', 'away_team',
            'home_score', 'away_score', 'league_key', 'league_name', 'season'
        ])
        empty_df.to_csv(fixtures_file, index=False)
        return False

def load_todays_fixtures():
    """Load today's fixtures with error handling."""
    try:
        fixtures = pd.read_csv(DATA_DIR / "fixtures_data.csv")
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

# -------------------
# Prediction Functions
# -------------------
def parse_match_datetime(row):
    if pd.isna(row.get('time')) or row.get('time') in ['', 'TBD', None]:
        return pd.NaT
    try:
        dt_str = f"{row['Date'].strftime('%Y-%m-%d')} {row['time']}"
        return pd.to_datetime(dt_str, format='%Y-%m-%d %H:%M', utc=True)
    except Exception:
        return pd.NaT

def format_times_for_message(match_utc):
    if pd.isna(match_utc):
        return "TBD", "TBD"
    utc_time = match_utc.tz_convert('UTC').strftime('%H:%M UTC')
    local_time = match_utc.tz_convert('Africa/Nairobi').strftime('%H:%M EAT')
    return utc_time, local_time

def has_been_predicted(home, away, date_str):
    if not PREDICTION_LOG.exists():
        return False
    try:
        log_df = pd.read_csv(PREDICTION_LOG)
        required_cols = {'home_team', 'away_team', 'date'}
        if not required_cols.issubset(set(log_df.columns)):
            backup_prediction_log()
            try:
                PREDICTION_LOG.unlink()
            except Exception:
                pass
            return False
        return ((log_df['home_team'] == home) &
                (log_df['away_team'] == away) &
                (log_df['date'] == date_str)).any()
    except Exception as e:
        print(f"⚠️  Prediction log corrupted: {e}")
        backup_prediction_log()
        try:
            PREDICTION_LOG.unlink()
        except Exception:
            pass
        return False

def log_prediction(home, away, date_str):
    log_entry = pd.DataFrame([{
        'home_team': home,
        'away_team': away,
        'date': date_str
    }])
    if not PREDICTION_LOG.exists():
        log_entry.to_csv(PREDICTION_LOG, index=False)
    else:
        log_entry.to_csv(PREDICTION_LOG, mode='a', header=False, index=False)

def get_market_baseline(my_prob, value_map):
    bins = np.arange(0, 1.05, 0.05)
    bin_idx = np.digitize([my_prob], bins) - 1
    if bin_idx[0] >= len(bins) - 1:
        bin_idx[0] = len(bins) - 2
    bin_key = pd.Interval(bins[bin_idx[0]], bins[bin_idx[0] + 1], closed='right')
    return value_map.get(bin_key, my_prob)

def should_send_tip(hda_proba, gg_proba, over25_proba, edge):
    home_win, draw, away_win = hda_proba
    hda_clear = (home_win >= 0.48) or (away_win >= 0.48) or (draw >= 0.35)
    gg_clear = (gg_proba >= 0.52) or (gg_proba <= 0.32)
    ou_clear = (over25_proba >= 0.52) or (over25_proba <= 0.32)
    strong_edge = edge >= 0.01
    return hda_clear or gg_clear or ou_clear or strong_edge

def build_prediction_message(match_row, historical_df):
    if not MODELS:
        return "🔧 Models not available - predictions temporarily disabled"

    home = match_row['home_team']
    away = match_row['away_team']
    match_date = match_row['Date']
    league_name = match_row.get('league_name', 'Unknown League')

    features = compute_match_features(
        historical_df=historical_df,
        home_team=home,
        away_team=away,
        match_date=match_date,
        league_code=match_row.get('league_code', 'E0')
    )

    X_raw = pd.DataFrame([features]) if isinstance(features, dict) else pd.DataFrame([features])
    try:
        X = X_raw.reindex(columns=MODELS['feature_cols'], fill_value=0)
    except Exception:
        X = X_raw.copy()

    try:
        hda_proba = MODELS['hda'].predict_proba(X)[0]
        gg_proba = MODELS['gg'].predict_proba(X)[0][1]
        over25_proba = MODELS['over25'].predict_proba(X)[0][1]
    except Exception as e:
        print(f"❌ Model prediction error: {e}")
        return None

    home_win_prob = hda_proba[0]
    market_baseline = get_market_baseline(home_win_prob, MODELS.get('value_map', {}))
    edge = home_win_prob - market_baseline

    if not should_send_tip(hda_proba, gg_proba, over25_proba, edge):
        return None

    home_win, draw, away_win = hda_proba
    best_tip = ""

    if home_win >= 0.52:
        best_tip = "🟢 HOME (Strong Home Favorite)"
    elif away_win >= 0.52:
        best_tip = "🔵 AWAY (Strong Away Win)"
    elif draw >= 0.35:
        best_tip = "🟡 DRAW (High Draw Probability)"
    elif gg_proba >= 0.52:
        best_tip = "🟢 GG (Both Teams to Score)"
    elif gg_proba <= 0.35:
        best_tip = "🔴 NG (No Goals Expected)"
    elif over25_proba >= 0.52:
        best_tip = "🟢 OVER 2.5 (High-Scoring Game)"
    elif over25_proba <= 0.35:
        best_tip = "🔴 UNDER 2.5 (Low-Scoring Game)"
    else:
        best_tip = "💡 Mixed Signals"

    match_utc = parse_match_datetime(match_row)
    utc_str, local_str = format_times_for_message(match_utc)
    time_str = f"🕒Kickoff: {utc_str} | {local_str}" if not pd.isna(match_utc) else "🕒 Kickoff: TBD (today | tomorrow)"

    separator = "─" * 30
    message = (
        f"📌*TIP ALERT*\n"
        f"🏆League: {league_name}\n"
        f"🏠{home} vs {away}\n"
        f"{time_str}\n"
        f"*TIP*:{best_tip}\n\n"
        f"🏠 Home: {hda_proba[0]:.0%} *|* 🤝 Draw: {hda_proba[1]:.0%} *|* 🚌 Away: {hda_proba[2]:.0%}\n"
        f"⚽Both Teams to Score: {gg_proba:.0%}\n"
        f"📈Over 2.5 Goals: {over25_proba:.0%}\n"
    )

    if edge > 0.02:
        message += (
            f"\n`" + "─" * 20 + "`\n"
            f"⚠️ *HIGH VALUE ALERT!*\n"
            f"📊 *Model*: {home_win_prob:.0%} Home | *Market Baseline*: ~{market_baseline:.0%}\n"
            f"➡️ {edge:+.0%} *edge (Our Model vs Historical Average)*\n"
        )

    message += (f"{separator}\n"
                f"💡 Bet responsibly || scoresignal")

    return message

# -------------------
# Weekday vs Weekend Logic
# ------------------
def _get_target_fixtures_for_window(fixtures, run_type, is_weekday):
    """
    Determine which fixtures to process based on run type and day type
    """
    if run_type == 'morning':
        if is_weekday:
            # WEEKDAYS: Process ALL fixtures in morning run
            target_fixtures = fixtures.copy()
            print(f"🌅 Weekday morning run: processing {len(target_fixtures)} fixtures (ALL matches)")
        else:
            # WEEKENDS: Process fixtures up to 5 PM + all no-time matches
            target_fixtures = _get_weekend_morning_fixtures(fixtures)
    else:  # afternoon (weekends only)
        # WEEKENDS ONLY: Process fixtures from 5 PM onwards
        target_fixtures = _get_weekend_afternoon_fixtures(fixtures)
    
    return target_fixtures

# -------------------
# Weekend morning fixture filtering helper
# ------------------
def _get_weekend_morning_fixtures(fixtures):
    """Get fixtures for weekend morning run (before 5 PM + no-time matches)"""
    target_fixtures = fixtures[
        # Include matches with valid times before 5 PM
        ((fixtures['time'].notna()) & 
         (fixtures['time'] != '') & 
         (fixtures['time'] != 'TBD') &
         (pd.to_datetime(fixtures['time'], format='%H:%M', errors='coerce').dt.hour <= 17)) |
        # OR include matches with no time data
        (fixtures['time'].isna() | 
         (fixtures['time'] == '') | 
         (fixtures['time'] == 'TBD'))
    ]
    print(f"🌅 Weekend morning run: processing {len(target_fixtures)} fixtures (before 5 PM + no-time matches)")
    return target_fixtures

# -------------------
# Weekend afternoon fixture filtering helper
# ------------------
def _get_weekend_afternoon_fixtures(fixtures):
    """Get fixtures for weekend afternoon run (after 5 PM only)"""
    target_fixtures = fixtures[
        (fixtures['time'].notna()) & 
        (fixtures['time'] != '') & 
        (fixtures['time'] != 'TBD') &
        (pd.to_datetime(fixtures['time'], format='%H:%M', errors='coerce').dt.hour > 17)
    ]
    print(f"🌇 Weekend afternoon run: processing {len(target_fixtures)} fixtures (5 PM and later only)")
    return target_fixtures

# -------------------
# Single match processing helper
# ------------------
def _process_single_match(match, historical_df):
    """
    Process a single match and return tip data if successful
    """
    home = match['home_team']
    away = match['away_team']
    date_str = match['Date'].strftime('%Y-%m-%d')

    if has_been_predicted(home, away, date_str):
        print(f"⏭️  Skipped (already predicted): {home} vs {away}")
        return None

    try:
        message = build_prediction_message(match, historical_df)
        if message is None or "Models not available" in message:
            print(f"⏭️  Skipped (low confidence): {home} vs {away}")
            return None

        send_telegram_message(message)
        backup_prediction_log()
        log_prediction(home, away, date_str)
        print(f"✅ Sent prediction: {home} vs {away}")
        
        return {
            'home': home,
            'away': away,
            'league': match.get('league_name', 'Unknown League')
        }

    except Exception as e:
        print(f"❌ Error predicting {home} vs {away}: {e}")
        return None

# -------------------
# Run predictions for time window
# ------------------
def run_predictions_for_time_window(fixtures, historical_df, run_type):
    """
    Run predictions for specific time windows based on run_type and day type
    - morning: all fixtures up to 5 PM + all no-time matches
               (on weekdays: ALL fixtures regardless of time)
    - afternoon: all fixtures after 5 PM only (weekends only)
    """
    # Determine day type
    nairobi_tz = pytz.timezone('Africa/Nairobi')
    now = datetime.now(nairobi_tz)
    current_weekday = now.weekday()  # 0=Monday, 6=Sunday
    is_weekday = current_weekday < 5  # Monday-Friday
    
    # Get appropriate fixtures for this run
    target_fixtures = _get_target_fixtures_for_window(fixtures, run_type, is_weekday)
    
    if target_fixtures.empty:
        print(f"⏭️  No fixtures for {run_type} run")
        return []

    # Process each match
    sent_tips = []
    for _, match in target_fixtures.iterrows():
        tip_data = _process_single_match(match, historical_df)
        if tip_data:
            sent_tips.append(tip_data)
            time.sleep(0.5)  # Rate limiting

    print(f"📊 {len(sent_tips)} tips sent for {run_type} run")
    return sent_tips

# -------------------
# Dynamic Header Creation
# -------------------
def create_dynamic_header(sent_tips, fixtures, run_type):
    """
    Create a dynamic header message based on prediction results
    """
    nairobi_tz = pytz.timezone('Africa/Nairobi')
    now = datetime.now(nairobi_tz)
    current_weekday = now.weekday()
    is_weekday = current_weekday < 5
    
    # Determine the prediction context
    if fixtures.empty:
        result_context = "found *no fixtures* scheduled for today, stay tuned"
    elif not sent_tips:
        result_context = "found *no high-confidence tips*, hang tight"
    else:
        tip_word = "tip" if len(sent_tips) == 1 else "tips"
        result_context = f"found *{len(sent_tips)} high-confidence {tip_word}*, check them out below"
    
    # Determine run context
    if is_weekday:
        run_context = "single daily analysis"
    else:
        if run_type == 'morning':
            run_context = "morning analysis (matches until 5 PM)"
        else:
            run_context = "afternoon analysis (matches after 5 PM)"
    
    header = (
        "*SCORESIGNAL FOOTBALL PREDICTIONS*\n\n"
        
        "*How It Works:*\n"
        "We analyze fixtures from *15+ European leagues* using machine learning "
        "models trained on a *decade of historical data*.\n\n"
        
        "*Today's Results:*\n"
        f"In our {run_context}, we {result_context}.\n\n"
        
        "*Our Approach:*\n"
        "• Evidence-based probabilistic insights\n"
        "• High-confidence tips only\n" 
        "• Rigorous, data-driven analysis\n"
        "• No noise, no unrealistic promises\n\n"
        
        "`──────────────────────────────`\n"
        "*Support Our Work:* MPESA TILL *9105695*\n"
        "*scoresignal* •  *ML-Powered Tips* •  *Bet Responsibly*"
    )
    
    return header
# -------------------
# Main Execution Logic
# -------------------
def main():
    # Acquire execution lock first to prevent duplicate runs
    if not acquire_execution_lock():
        return
    
    try:
        # Check if we should run based on schedule
        should_run, run_type = should_run_tips()
        if not should_run:
            print(f"⏸️ Not in scheduled run window (current Nairobi time: {get_now_nairobi().strftime('%H:%M')})")
            return

        print(f"🎯 Starting {run_type} run at {get_now_nairobi().strftime('%Y-%m-%d %H:%M')} Nairobi time")

        # Initialize database with robust error handling
        try:
            ensure_table()
            print("✅ Database initialized successfully")
        except Exception as e:
            print(f"⚠️  Database initialization failed: {e}")
            # Don't crash - continue without database functionality
            # Database is only for chat management, not critical for predictions

        # Reset daily flag for new day
        reset_daily_flag_if_new_day()

        # Backup prediction log
        backup_prediction_log()

        # Ensure historical data is available (weekly)
        historical_ready = ensure_historical_data_exists()

        # Fetch fixtures if needed
        fixtures_fetched = safe_fetch_fixtures()
        if not fixtures_fetched:
            print("❌ Could not fetch fixtures - continuing with fallback")

        # Load today's fixtures
        fixtures = load_todays_fixtures()

        # Handle no fixtures case FIRST (before sending header)
        if fixtures.empty:
            if not DAILY_FLAG.exists():
                print("📭 No fixtures today — sending notification")
                # Create header for no fixtures scenario
                header = create_dynamic_header([], fixtures, run_type)
                send_telegram_message(header)
                
                # Additional no fixtures message
                no_fixtures_msg = (
                    "😴 *No Matches Today*\n\n"
                    "There are no fixtures scheduled for today in our curated leagues. "
                    "We'll be back tomorrow with fresh predictions!\n\n"
                    "🏃‍♂️ *See you tomorrow!*"
                )
                send_telegram_message(no_fixtures_msg)
                DAILY_FLAG.touch()
            else:
                print("📭 No fixtures today (already notified)")
            return

        # Send initial header (before predictions, so we don't know results yet)
        initial_header = create_dynamic_header([], fixtures, run_type)
        send_telegram_message(initial_header)

        # Load historical data for predictions
        historical_df = load_historical_data()

        # Run predictions for the appropriate time window
        sent_tips = run_predictions_for_time_window(fixtures, historical_df, run_type)

        # Handle results with appropriate messaging
        if sent_tips:
            try:
                print("🧠 Generating LLM summary...")
                summary = generate_daily_summary(sent_tips)
                summary_message = create_summary_message(summary)
                send_telegram_message(summary_message)
                print("✅ LLM summary sent!")
            except Exception as e:
                print(f"❌ LLM summary failed: {e}")
        else:
            # No tips found - send appropriate message
            if not fixtures.empty:  # Only send if there were fixtures but no tips
                no_tips_message = (
                    "🔍 *Analysis Complete*\n\n"
                    "We've analyzed today's fixtures but didn't find any matches "
                    "meeting our *high-confidence thresholds*.\n\n"
                    "💡 *Why this happens:*\n"
                    "• Matches are too evenly balanced\n" 
                    "• Insufficient historical data\n"
                    "• Probabilities below our confidence levels\n\n"
                    
                    "🔄 *What's Next:*\n"
                )
                
                # Add next run time based on day and current run
                if run_type == 'morning':
                    next_run = "4 PM today" if datetime.now().weekday() >= 5 else "tomorrow at 10 AM"
                else:  # afternoon
                    next_run = "tomorrow at 10 AM"
                
                no_tips_message += f"Next predictions will be sent at *{next_run}* Nairobi time"
                
                send_telegram_message(no_tips_message)
                print("📭 No high-confidence tips found")

        # Mark as notified for the day
        if run_type == 'morning':
            DAILY_FLAG.touch()

    except Exception as e:
        print(f"💥 Critical error in main: {e}")
        traceback.print_exc()
        try:
            send_telegram_message(f"🚨 Bot crashed: {str(e)[:100]}...")
        except Exception:
            pass
    finally:
        # Release lock
        release_execution_lock()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"💥 Critical error in main: {e}")
        traceback.print_exc()
        # Release lock even on crash
        release_execution_lock()
        try:
            send_telegram_message(f"🚨 Bot crashed: {str(e)[:100]}...")
        except Exception:
            pass