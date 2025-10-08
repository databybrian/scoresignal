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
from datetime import datetime, timedelta, time as dt_time  # Rename to avoid conflict
import pytz
import joblib

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"src path: {PROJECT_ROOT / 'src'}")

# -------------------
# Import project modules (with safe fallbacks)
# -------------------
try:
    from src.h2h_form import compute_match_features, get_feature_columns, create_feature_vector
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

    def get_feature_columns(include_odds=False):
        return []

    def create_feature_vector(features_dict, feature_columns):
        return np.array([])

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
from src.data_pipeline import ensure_historical_data_exists, save_all_current_tables, needs_refresh

# -------------------
# Parse time safely
# -------------------
def _parse_time_safely(time_str):
    """Parse time string safely, return None if invalid"""
    try:
        if pd.isna(time_str) or time_str in ['', 'TBD', None]:
            return None
        parsed = pd.to_datetime(time_str, format='%H:%M', errors='coerce')
        return parsed if pd.notna(parsed) else None
    except:
        return None

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
    current_weekday = now.weekday()
    
    morning_window_start = dt_time(10, 0)  # Fixed: use dt_time instead of time
    morning_window_end = dt_time(10, 15)
    
    afternoon_window_start = dt_time(16, 0)
    afternoon_window_end = dt_time(16, 15)
    
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
        return pd.DataFrame()

# -------------------
# Model and Data Loading
# -------------------
def safe_load_models():
    """Safely load ensemble models with error handling"""
    models = {}
    try:
        print("🚀 Loading ensemble models...")
        
        # Load ensemble models (dict with 'xgb', 'lgb', 'cat' keys)
        models['hda'] = joblib.load(MODEL_DIR / "ensemble_hda.pkl")
        models['btts'] = joblib.load(MODEL_DIR / "ensemble_btts.pkl")
        models['over25'] = joblib.load(MODEL_DIR / "ensemble_over25.pkl")
        
        # Load feature metadata
        feature_metadata = joblib.load(MODEL_DIR / "feature_metadata.pkl")
        models['hda_features'] = feature_metadata['hda_features']
        models['btts_features'] = feature_metadata['btts_features']
        models['over25_features'] = feature_metadata['over25_features']
        
        print("✅ Ensemble models loaded successfully!")
        print(f"   - HDA features: {len(models['hda_features'])}")
        print(f"   - BTTS features: {len(models['btts_features'])}")
        print(f"   - Over/Under features: {len(models['over25_features'])}")
        
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
    """Load today's fixtures with robust error handling."""
    try:
        fixtures = pd.read_csv(DATA_DIR / "fixtures_data.csv")
        
        # Normalize column names
        if 'date' in fixtures.columns and 'Date' not in fixtures.columns:
            fixtures.rename(columns={'date': 'Date'}, inplace=True)
        
        if 'Date' not in fixtures.columns:
            print("❌ No Date column found in fixtures")
            return pd.DataFrame()
        
        fixtures['Date'] = pd.to_datetime(fixtures['Date'], errors='coerce')
        
        if 'league_code' not in fixtures.columns:
            fixtures['league_code'] = 'E0'
        
        today = pd.Timestamp.now(tz='UTC').normalize()
        todays_fixtures = fixtures[fixtures['Date'].dt.date == today.date()].copy()
        print(f"✅ Loaded {len(todays_fixtures)} fixtures for today")
        return todays_fixtures
    except Exception as e:
        print(f"❌ Failed to load fixtures: {e}")
        traceback.print_exc()
        return pd.DataFrame()

# -------------------
# Ensemble Prediction Functions
# -------------------
def predict_with_ensemble(ensemble_models, X, task='multiclass'):
    """
    Get ensemble predictions by averaging XGBoost, LightGBM, and CatBoost
    
    Args:
        ensemble_models: dict with keys 'xgb', 'lgb', 'cat'
        X: feature matrix
        task: 'multiclass' or 'binary'
    
    Returns:
        averaged probabilities
    """
    predictions = []
    
    # XGBoost prediction
    if task == 'multiclass':
        predictions.append(ensemble_models['xgb'].predict_proba(X))
    else:
        predictions.append(ensemble_models['xgb'].predict_proba(X)[:, 1])
    
    # LightGBM prediction
    if task == 'multiclass':
        predictions.append(ensemble_models['lgb'].predict_proba(X))
    else:
        predictions.append(ensemble_models['lgb'].predict_proba(X)[:, 1])
    
    # CatBoost prediction
    if task == 'multiclass':
        predictions.append(ensemble_models['cat'].predict_proba(X))
    else:
        predictions.append(ensemble_models['cat'].predict_proba(X)[:, 1])
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    
    return ensemble_pred

# -------------------
# Advanced Tip Selection Logic
# -------------------
def select_best_tip(hda_proba, btts_proba, over25_proba):
    """
    Intelligent tip selection based on confidence levels and edge detection.
    
    Strategy:
    1. Calculate confidence scores for each market
    2. Select the market with highest confidence above threshold
    3. Return the specific tip with reasoning
    
    Thresholds (calibrated to professional standards):
    - HDA: Home/Away ≥ 55%, Draw ≥ 32%
    - BTTS: Yes/No ≥ 56%
    - Over/Under: ≥ 56%
    
    Returns:
        tuple: (tip_type, tip_text, confidence, should_send, secondary_tips)
    """
    home_prob, draw_prob, away_prob = hda_proba
    btts_yes = btts_proba
    btts_no = 1 - btts_proba
    over_prob = over25_proba
    under_prob = 1 - over25_proba
    
    # Calculate confidence scores (how far from 50% baseline)
    hda_home_confidence = max(0, home_prob - 0.53)
    hda_away_confidence = max(0, away_prob - 0.53)
    hda_draw_confidence = max(0, draw_prob - 0.32)
    
    btts_yes_confidence = max(0, btts_yes - 0.56)
    btts_no_confidence = max(0, btts_no - 0.56)
    
    over_confidence = max(0, over_prob - 0.56)
    under_confidence = max(0, under_prob - 0.56)
    
    # Collect all viable tips with their confidence
    tips = []
    
    # HDA tips
    if hda_home_confidence > 0:
        tips.append(('HDA_HOME', f"🟢 HOME WIN", home_prob, hda_home_confidence, 'primary'))
    if hda_away_confidence > 0:
        tips.append(('HDA_AWAY', f"🔵 AWAY WIN", away_prob, hda_away_confidence, 'primary'))
    if hda_draw_confidence > 0:
        tips.append(('HDA_DRAW', f"🟡 DRAW", draw_prob, hda_draw_confidence, 'secondary'))
    
    # BTTS tips
    if btts_yes_confidence > 0:
        tips.append(('BTTS_YES', f"⚽ BOTH TEAMS TO SCORE (Yes)", btts_yes, btts_yes_confidence, 'primary'))
    if btts_no_confidence > 0:
        tips.append(('BTTS_NO', f"🚫 BOTH TEAMS TO SCORE (No)", btts_no, btts_no_confidence, 'secondary'))
    
    # Over/Under tips
    if over_confidence > 0:
        tips.append(('OVER25', f"📈 OVER 2.5 GOALS", over_prob, over_confidence, 'primary'))
    if under_confidence > 0:
        tips.append(('UNDER25', f"📉 UNDER 2.5 GOALS", under_prob, under_confidence, 'secondary'))
    
    # If no tips meet threshold
    if not tips:
        return None, None, 0, False, []
    
    # Sort by confidence (descending)
    tips.sort(key=lambda x: x[3], reverse=True)
    
    # Select best tip
    best_tip = tips[0]
    tip_type, tip_text, probability, confidence, priority = best_tip
    
    # Secondary tips (complementary, lower priority)
    secondary_tips = [t for t in tips[1:3] if t[4] == 'secondary' or t[3] > 0.05]
    
    return tip_type, tip_text, probability, True, secondary_tips

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

def build_prediction_message(match_row, historical_df):
    """
    Build prediction message using ensemble models and advanced tip selection
    """
    if not MODELS:
        return "🔧 Models not available - predictions temporarily disabled"

    # Validate inputs
    try:
        home = str(match_row['home_team'])
        away = str(match_row['away_team'])
        match_date = pd.to_datetime(match_row['Date'])
        league_name = str(match_row.get('league_name', 'Unknown League'))
        league_code = str(match_row.get('league_code', 'E0'))
    except Exception as e:
        print(f"❌ Invalid match data: {e}")
        return None

    # Validate historical data
    if historical_df.empty:
        print("⚠️ No historical data available")
        return None

    # Compute features for all models
    try:
        # Base features (for HDA)
        base_features_dict = compute_match_features(
            historical_df=historical_df,
            home_team=home,
            away_team=away,
            match_date=match_date,
            league_code=league_code,
            include_odds=False
        )
        
        # All features (for BTTS and Over/Under)
        all_features_dict = compute_match_features(
            historical_df=historical_df,
            home_team=home,
            away_team=away,
            match_date=match_date,
            league_code=league_code,
            include_odds=True
        )
    except Exception as e:
        print(f"❌ Feature computation failed: {e}")
        traceback.print_exc()
        return None

    # Create feature vectors
    try:
        # HDA prediction (base features only)
        X_hda = pd.DataFrame([base_features_dict])[MODELS['hda_features']].fillna(0).values
        hda_proba = predict_with_ensemble(MODELS['hda'], X_hda, task='multiclass')[0]
        
        # BTTS prediction (all features)
        X_btts = pd.DataFrame([all_features_dict])[MODELS['btts_features']].fillna(0).values
        btts_proba = predict_with_ensemble(MODELS['btts'], X_btts, task='binary')[0]
        
        # Over/Under prediction (all features)
        X_over25 = pd.DataFrame([all_features_dict])[MODELS['over25_features']].fillna(0).values
        over25_proba = predict_with_ensemble(MODELS['over25'], X_over25, task='binary')[0]
        
    except Exception as e:
        print(f"❌ Model prediction error: {e}")
        traceback.print_exc()
        return None

    # Select best tip using advanced logic
    tip_type, tip_text, confidence, should_send, secondary_tips = select_best_tip(
        hda_proba, btts_proba, over25_proba
    )

    if not should_send:
        return None

    # Format match time
    match_utc = parse_match_datetime(match_row)
    utc_str, local_str = format_times_for_message(match_utc)
    time_str = f"🕒 Kickoff: {utc_str} | {local_str}" if not pd.isna(match_utc) else "🕒 Kickoff: TBD"

    # Build message
    message = (
        f"📌 *TIP ALERT*\n"
        f"🏆 League: {league_name}\n"
        f"⚽ {home} vs {away}\n"
        f"{time_str}\n\n"
        f"*PRIMARY TIP:* {tip_text}\n"
        f"💪 Confidence: {confidence:.0%}\n\n"
    )

    # Add secondary tips if available
    if secondary_tips:
        message += "*Alternative Options:*\n"
        for _, sec_text, sec_prob, sec_conf, _ in secondary_tips:
            message += f"  • {sec_text}: {sec_prob:.0%}\n"
        message += "\n"

    # Add full probabilities breakdown
    message += (
        f"📊 *Full Analysis:*\n"
        f"🏠 Home Win: {hda_proba[0]:.0%} | 🤝 Draw: {hda_proba[1]:.0%} | 🚌 Away Win: {hda_proba[2]:.0%}\n"
        f"⚽ Both Score: {btts_proba:.0%}\n"
        f"📈 Over 2.5: {over25_proba:.0%}\n\n"
    )

    # Add confidence explanation
    if confidence >= 0.10:
        message += "✨ *HIGH CONFIDENCE* - Strong statistical edge detected\n"
    elif confidence >= 0.05:
        message += "💡 *GOOD VALUE* - Solid opportunity identified\n"
    else:
        message += "📍 *MODERATE* - Acceptable edge over baseline\n"

    return message

# -------------------
# Weekday vs Weekend Logic
# -------------------
def _get_target_fixtures_for_window(fixtures, run_type, is_weekday):
    """
    Determine which fixtures to process based on run type and day type
    """
    if run_type == 'morning':
        if is_weekday:
            target_fixtures = fixtures.copy()
            print(f"🌅 Weekday morning run: processing {len(target_fixtures)} fixtures (ALL matches)")
        else:
            target_fixtures = _get_weekend_morning_fixtures(fixtures)
    else:  # afternoon (weekends only)
        target_fixtures = _get_weekend_afternoon_fixtures(fixtures)
    
    return target_fixtures

def _get_weekend_morning_fixtures(fixtures):
    """Get fixtures for weekend morning run (before 5 PM + no-time matches)"""
    fixtures = fixtures.copy()
    fixtures['parsed_time'] = fixtures['time'].apply(_parse_time_safely)
    
    target_fixtures = fixtures[
        ((fixtures['parsed_time'].notna()) & 
         (fixtures['parsed_time'].dt.hour < 17)) |
        (fixtures['parsed_time'].isna())
    ]
    print(f"🌅 Weekend morning run: processing {len(target_fixtures)} fixtures (before 5 PM + no-time matches)")
    return target_fixtures

def _get_weekend_afternoon_fixtures(fixtures):
    """Get fixtures for weekend afternoon run (after 5 PM only)"""
    fixtures = fixtures.copy()
    fixtures['parsed_time'] = fixtures['time'].apply(_parse_time_safely)
    
    target_fixtures = fixtures[
        (fixtures['parsed_time'].notna()) & 
        (fixtures['parsed_time'].dt.hour >= 17)
    ]
    print(f"🌇 Weekend afternoon run: processing {len(target_fixtures)} fixtures (5 PM and later only)")
    return target_fixtures

# -------------------
# Batch message sending with Telegram limits
# -------------------
def send_batched_tips(tips_messages):
    """
    Send tips in batched messages respecting Telegram's 4096 character limit.
    Each tip separated by horizontal line.
    """
    if not tips_messages:
        return
    
    TELEGRAM_LIMIT = 4096
    SEPARATOR = "\n\n" + "─" * 40 + "\n\n"
    
    current_batch = []
    current_length = 0
    
    for tip_msg in tips_messages:
        tip_length = len(tip_msg)
        separator_length = len(SEPARATOR) if current_batch else 0
        
        if current_length + separator_length + tip_length > TELEGRAM_LIMIT - 100:
            if current_batch:
                batched_message = SEPARATOR.join(current_batch)
                send_telegram_message(batched_message)
                time.sleep(1)
            
            current_batch = [tip_msg]
            current_length = tip_length
        else:
            current_batch.append(tip_msg)
            current_length += separator_length + tip_length
    
    if current_batch:
        batched_message = SEPARATOR.join(current_batch)
        send_telegram_message(batched_message)

# -------------------
# Single match processing helper
# -------------------
def _process_single_match(match, historical_df):
    """Process a single match and return tip data if successful"""
    if pd.isna(match.get('home_team')) or pd.isna(match.get('away_team')):
        print("⚠️ Skipped: Missing team names")
        return None
    
    home = match['home_team']
    away = match['away_team']
    
    if pd.isna(match.get('Date')):
        print(f"⚠️ Skipped: Missing date for {home} vs {away}")
        return None
    
    date_str = match['Date'].strftime('%Y-%m-%d')

    if has_been_predicted(home, away, date_str):
        print(f"⏭️  Skipped (already predicted): {home} vs {away}")
        return None

    try:
        message = build_prediction_message(match, historical_df)
        if message is None or "Models not available" in message:
            print(f"⏭️  Skipped (low confidence): {home} vs {away}")
            return None

        backup_prediction_log()
        log_prediction(home, away, date_str)
        print(f"✅ Generated prediction: {home} vs {away}")
        
        return {
            'home': home,
            'away': away,
            'league': match.get('league_name', 'Unknown League'),
            'message': message
        }

    except Exception as e:
        print(f"❌ Error predicting {home} vs {away}: {e}")
        traceback.print_exc()
        return None

# -------------------
# Run predictions for time window
# -------------------
def run_predictions_for_time_window(fixtures, historical_df, run_type):
    """Run predictions for specific time windows"""
    nairobi_tz = pytz.timezone('Africa/Nairobi')
    now = datetime.now(nairobi_tz)
    current_weekday = now.weekday()
    is_weekday = current_weekday < 5
    
    target_fixtures = _get_target_fixtures_for_window(fixtures, run_type, is_weekday)
    
    if target_fixtures.empty:
        print(f"⏭️  No fixtures for {run_type} run")
        return []

    sent_tips = []
    tip_messages = []
    
    for _, match in target_fixtures.iterrows():
        tip_data = _process_single_match(match, historical_df)
        if tip_data:
            sent_tips.append({
                'home': tip_data['home'],
                'away': tip_data['away'],
                'league': tip_data['league']
            })
            tip_messages.append(tip_data['message'])
            time.sleep(0.3)

    if tip_messages:
        send_batched_tips(tip_messages)
        
    print(f"📊 {len(sent_tips)} tips sent for {run_type} run")
    return sent_tips

# -------------------
# Dynamic Header Creation
# -------------------
def create_dynamic_header(sent_tips, fixtures, run_type):
    """Create a dynamic header message based on prediction results"""
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
        "🎯 *SCORESIGNAL FOOTBALL PREDICTIONS*\n\n"
        
        "🤖 *How It Works:*\n"
        "We analyze fixtures from *15+ European leagues* using advanced ensemble ML "
        "models (XGBoost + LightGBM + CatBoost) trained on *100,000+ historical matches*.\n\n"
        
        "📊 *Today's Results:*\n"
        f"In our {run_context}, we {result_context}.\n\n"
        
        "⚡ *Our Approach:*\n"
        "• Evidence-based probabilistic insights\n"
        "• High-confidence tips only (calibrated thresholds)\n" 
        "• Rigorous, data-driven ensemble predictions\n"
        "• ELO ratings, form analysis, H2H stats\n"
        "• No noise, no unrealistic promises\n\n"
        
        "`─────────────────────────────────`\n"
        "💰 *Support Our Work:* MPESA TILL *9105695*\n"
        "🔬 *scoresignal* • *ML-Powered Tips* • *Bet Responsibly*"
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
                print("🔭 No fixtures today – sending notification")
                header = create_dynamic_header([], fixtures, run_type)
                send_telegram_message(header)
                
                no_fixtures_msg = (
                    "😴 *No Matches Today*\n\n"
                    "There are no fixtures scheduled for today in our curated leagues. "
                    "We'll be back tomorrow with fresh predictions!\n\n"
                    "🏃‍♂️ *See you tomorrow!*"
                )
                send_telegram_message(no_fixtures_msg)
                DAILY_FLAG.touch()
            else:
                print("🔭 No fixtures today (already notified)")
            return

        # Send initial header
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
            if not fixtures.empty:
                no_tips_message = (
                    "🔍 *Analysis Complete*\n\n"
                    "We've analyzed today's fixtures using our ensemble models "
                    "(XGBoost + LightGBM + CatBoost) but didn't find any matches "
                    "meeting our *high-confidence thresholds*.\n\n"
                    "💡 *Why this happens:*\n"
                    "• Matches are too evenly balanced\n" 
                    "• Model predictions below calibrated thresholds:\n"
                    "  - HDA: Home/Away <53%, Draw <32%\n"
                    "  - BTTS: <56% confidence\n"
                    "  - Over/Under: <56% confidence\n"
                    "• Insufficient historical data for reliable features\n\n"
                    
                    "🔄 *What's Next:*\n"
                )
                
                # Add next run time based on day and current run
                if run_type == 'morning':
                    next_run = "4 PM today" if datetime.now().weekday() >= 5 else "tomorrow at 10 AM"
                else:
                    next_run = "tomorrow at 10 AM"
                
                no_tips_message += f"Next predictions will be sent at *{next_run}* Nairobi time"
                
                send_telegram_message(no_tips_message)
                print("🔭 No high-confidence tips found")

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
        release_execution_lock()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"💥 Critical error in main: {e}")
        traceback.print_exc()
        release_execution_lock()
        try:
            send_telegram_message(f"🚨 Bot crashed: {str(e)[:100]}...")
        except Exception:
            pass