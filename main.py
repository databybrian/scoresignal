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
print("Current Python path:")
for i, path in enumerate(sys.path):
    print(f"  {i}: {path}")

# -------------------
# Run scheduler behavior
# -------------------
def should_run_now():
    """
    We keep the hourly cron schedule. This helper logs whether we are inside
    the strict tip windows (for info). The script will run on every cron trigger.
    """
    nairobi_tz = pytz.timezone('Africa/Nairobi')
    now = datetime.now(nairobi_tz := nairobi_tz)

    # tip windows (Nairobi time) - informational only
    tip_windows = [
        (8, 0),   # 8:00 AM
        (12, 0),  # 12:00 PM
        (16, 0)   # 4:00 PM
    ]

    for hour, minute in tip_windows:
        target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        time_diff = abs((now - target_time).total_seconds())
        if time_diff <= 300:
            print(f"⏰ Within a tip window (near {hour}:{minute:02d} Nairobi time)")
            return True

    # Not in a strict window — but we still run because cron calls hourly.
    print("ℹ️ Not inside strict tip window — performing scheduled hourly scan (no Telegram messages outside tiers)")
    return True

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

# ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_LOG_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# Track first-tier "no matches today" message per day (prevents spamming when no fixtures)
DAILY_FLAG = PROJECT_ROOT / ".first_tier_notified"

# load models once
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

# -------------------
# Backup utilities (prediction_log) with rotation
# -------------------
def backup_prediction_log(max_keep: int = 7):
    """
    Copy current prediction log to backups with timestamp, then keep only
    `max_keep` newest backups.
    """
    try:
        if PREDICTION_LOG.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = PREDICTION_LOG_BACKUP_DIR / f"prediction_log_{ts}.csv"
            shutil.copy(PREDICTION_LOG, dest)
            # rotate
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
            print(f"📦 Backed up prediction log -> {dest} (kept {min(len(backups), max_keep)} backups)")
    except Exception as e:
        print(f"⚠️ Failed to backup prediction log: {e}")
        traceback.print_exc()

# -------------------
# Freshness helpers (weekly refresh)
# -------------------
def file_age_days(path: Path) -> float:
    if not path.exists():
        return float("inf")
    return (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds() / 86400.0

def needs_refresh(path: Path, days: int = 7) -> bool:
    """Return True if file is missing or older than `days` days."""
    return not path.exists() or (file_age_days(path) >= days)

# -------------------
# Data processing functions (cleaning historical -> cleaned_historical_data.csv)
# -------------------
def process_raw_to_cleaned():
    """Process raw combined_historical_data.csv into cleaned_historical_data.csv"""
    RAW_FILE = PROJECT_ROOT / "combined_historical_data.csv"
    CLEANED_FILE = DATA_DIR / "cleaned_historical_data.csv"

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw historical data not found: {RAW_FILE}")

    # Load dtype mapping
    DTYPE_FILE = PROJECT_ROOT / "raw_data" / "data_type_mapping.csv"
    if not DTYPE_FILE.exists():
        raise FileNotFoundError(f"Data type mapping not found: {DTYPE_FILE}")

    dtype_df = pd.read_csv(DTYPE_FILE)
    DTYPE_MAPPING = dict(zip(dtype_df['column_name'], dtype_df['data_type']))

    def clean_numeric_string(value):
        if isinstance(value, str):
            value = value.strip()
            if value in ['', '#', 'NA', 'N/A', 'NULL', 'NaN', 'nan']:
                return np.nan
            cleaned = ''.join(char for char in value if char.isdigit() or char in '.-')
            return cleaned if cleaned else np.nan
        return value

    numeric_columns = [col for col, dtype in DTYPE_MAPPING.items()
                       if dtype in ['float32', 'float64', 'int8', 'int16', 'int32', 'int64']]
    converters = {col: clean_numeric_string for col in numeric_columns}

    print(f"📥 Loading raw historical data from {RAW_FILE}")
    df = pd.read_csv(
        RAW_FILE,
        low_memory=False,
        dtype=DTYPE_MAPPING,
        converters=converters,
        na_values=['', '#', 'NA', 'N/A', 'NULL', 'NaN', 'nan']
    )

    print(f"📊 Loaded {len(df)} raw matches")

    # Apply date formatting
    try:
        from src.data_utils import format_date_column
        df = format_date_column(df)
    except Exception as e:
        print(f"⚠️  Date formatting warning: {e}")

    # Sort by date
    df = df.sort_values('Date', ignore_index=True)

    # Ensure data directory exists
    CLEANED_FILE.parent.mkdir(exist_ok=True)

    # Save cleaned version
    df.to_csv(CLEANED_FILE, index=False, encoding='utf-8')
    print(f"✅ Saved cleaned historical data to {CLEANED_FILE}")
    return df

# -------------------
# Ensure historical data (weekly)
# -------------------
def ensure_historical_data_exists():
    """Ensure cleaned historical data exists (refresh weekly)"""
    CLEANED_FILE = DATA_DIR / "cleaned_historical_data.csv"

    if CLEANED_FILE.exists() and not needs_refresh(CLEANED_FILE, days=7):
        print("✅ Cleaned historical data is recent - no refresh needed")
        return True

    print("🔄 Setting up/refeshing historical data (weekly)...")
    try:
        # Step 1: Download raw data
        print("📥 Downloading raw historical data...")
        from scripts.download_historical_data import download_and_combine_all_historical_data
        download_and_combine_all_historical_data()

        # Step 2: Process into cleaned format
        print("🧹 Processing raw data into cleaned format...")
        process_raw_to_cleaned()

        print("✅ Historical data setup complete!")
        return True

    except Exception as e:
        print(f"❌ Failed to setup historical data: {e}")
        traceback.print_exc()
        return False

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
# Fixtures fetching (weekly)
# -------------------
def safe_fetch_fixtures():
    """Safely fetch fixtures; refresh weekly to avoid repeated downloads."""
    fixtures_file = DATA_DIR / "fixtures_data.csv"

    if fixtures_file.exists() and not needs_refresh(fixtures_file, days=7):
        print("✅ Fixtures file is recent - no refresh needed")
        return True

    print("🔄 fixtures_data.csv missing/old - fetching live fixtures (weekly)...")
    try:
        from src.fetch_fixtures_live import fetch_and_save_fixtures
        fetch_and_save_fixtures(str(fixtures_file))
        print("✅ Fixtures fetched successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to fetch fixtures: {e}")
        traceback.print_exc()
        # Create empty fixtures file as fallback
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
        # Normalize date column name to 'Date'
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
        traceback.print_exc()
        return pd.DataFrame()

# -------------------
# Time parsing & formatting
# -------------------
def parse_match_datetime(row):
    """Parse match date + time into UTC datetime."""
    if pd.isna(row.get('time')) or row.get('time') in ['', 'TBD', None]:
        return pd.NaT
    try:
        dt_str = f"{row['Date'].strftime('%Y-%m-%d')} {row['time']}"
        return pd.to_datetime(dt_str, format='%Y-%m-%d %H:%M', utc=True)
    except Exception:
        return pd.NaT

def format_times_for_message(match_utc):
    """Return formatted UTC and EAT times."""
    if pd.isna(match_utc):
        return "TBD", "TBD"
    utc_time = match_utc.tz_convert('UTC').strftime('%H:%M UTC')
    local_time = match_utc.tz_convert('Africa/Nairobi').strftime('%H:%M EAT')
    return utc_time, local_time

# -------------------
# Next-tier helper (ADDED)
# -------------------
def next_tier_start(now: datetime):
    """
    Given a Nairobi-localized datetime `now`, return (next_start_dt, tier_label).
    next_start_dt is timezone-aware (Africa/Nairobi).
    Tiers start at 08:00, 12:00, 16:00 Nairobi.
    If `now` is before a tier start the same day, return that; otherwise return next day's Tier 1 start.
    """
    tz = pytz.timezone('Africa/Nairobi')
    if now.tzinfo is None:
        now = tz.localize(now)
    else:
        now = now.astimezone(tz)

    today = now.date()

    tier_starts = [
        (datetime.combine(today, datetime.min.time()).replace(hour=8, minute=0, tzinfo=tz), "Tier 1 (No-time)"),
        (datetime.combine(today, datetime.min.time()).replace(hour=12, minute=0, tzinfo=tz), "Tier 2 (Daytime)"),
        (datetime.combine(today, datetime.min.time()).replace(hour=16, minute=0, tzinfo=tz), "Tier 3 (Evening)"),
    ]

    for start_dt, label in tier_starts:
        if now < start_dt:
            return start_dt, label

    # If we're past today's last tier, return tomorrow's Tier 1
    tomorrow = today + timedelta(days=1)
    next_start = datetime.combine(tomorrow, datetime.min.time()).replace(hour=8, minute=0, tzinfo=tz)
    return next_start, "Tier 1 (No-time)"

# -------------------
# Prediction log helpers (robust)
# -------------------
def has_been_predicted(home, away, date_str):
    """Check if match already predicted — handles missing/wrong columns."""
    if not PREDICTION_LOG.exists():
        return False

    try:
        log_df = pd.read_csv(PREDICTION_LOG)

        required_cols = {'home_team', 'away_team', 'date'}
        if not required_cols.issubset(set(log_df.columns)):
            print("⚠️  Prediction log missing required columns — backing up and resetting log")
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
        print(f"⚠️  Prediction log corrupted or unreadable — backing up and resetting: {e}")
        traceback.print_exc()
        backup_prediction_log()
        try:
            PREDICTION_LOG.unlink()
        except Exception:
            pass
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

# -------------------
# Model helpers and predictions
# -------------------
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
    hda_clear = (home_win >= 0.58) or (away_win >= 0.58) or (draw >= 0.38)

    # GG: Decisive probability
    gg_clear = (gg_proba >= 0.68) or (gg_proba <= 0.32)

    # Over/Under: Decisive probability
    ou_clear = (over25_proba >= 0.68) or (over25_proba <= 0.32)

    # Value edge is strong
    strong_edge = edge >= 0.04

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

    # Create X and ensure expected columns are present (fill missing with 0)
    X_raw = pd.DataFrame([features]) if isinstance(features, dict) else pd.DataFrame([features])
    try:
        X = X_raw.reindex(columns=MODELS['feature_cols'], fill_value=0)
    except Exception:
        # If feature_cols missing or any issue, try to use X_raw directly (fall back)
        X = X_raw.copy()

    # Get predictions
    try:
        hda_proba = MODELS['hda'].predict_proba(X)[0]
        gg_proba = MODELS['gg'].predict_proba(X)[0][1]
        over25_proba = MODELS['over25'].predict_proba(X)[0][1]
    except Exception as e:
        print(f"❌ Model prediction error: {e}")
        traceback.print_exc()
        return None

    # Calculate value edge vs historical market baseline
    home_win_prob = hda_proba[0]
    market_baseline = get_market_baseline(home_win_prob, MODELS.get('value_map', {}))
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
    time_str = f"🕒Kickoff: {utc_str} | {local_str}" if not pd.isna(match_utc) else "🕒 Kickoff: TBD (today | tomorrow)"

    # Build final message
    separator = "─" * 30
    message = (
        f"📌*TIP ALERT*\n"
        f"🏆League: {league_name}\n"
        f"🏠{home} vs {away}\n"
        f"{time_str}\n"
        f"*→*:{best_tip}\n\n"
        f"🏠 Home: {hda_proba[0]:.0%} *|* 🤝 Draw: {hda_proba[1]:.0%} *|* 🚌 Away: {hda_proba[2]:.0%}\n"
        f"⚽Both Teams to Score: {gg_proba:.0%}\n"
        f"📈Over 2.5 Goals: {over25_proba:.0%}\n"
    )

    # Add value alert if edge is strong (≥2%)
    if edge > 0.02:
        message += (
            f"\n`" + "─" * 20 + "`\n"
            f"⚠️ *HIGH VALUE ALERT!*\n"
            f"📊 *Model*: {home_win_prob:.0%} Home | *Market Baseline*: ~{market_baseline:.0%}\n"
            f"➡️ {edge:+.0%} *edge (Our Model vs Historical Average)*\n"
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
        X_raw = pd.DataFrame([features])
        X = X_raw.reindex(columns=MODELS['feature_cols'], fill_value=0)
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
            features = compute_match_features(
                historical_df=historical_df,
                home_team=match_row['home_team'],
                away_team=match_row['away_team'],
                match_date=match_row['Date'],
                league_code=match_row.get('league_code', 'E0')
            )
            X_raw = pd.DataFrame([features])
            X = X_raw.reindex(columns=MODELS['feature_cols'], fill_value=0)
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

# -------------------
# Run predictions per tier
# -------------------
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
                # Backup prediction log before modifying (not on every append to reduce I/O,
                # but do it here to ensure a copy exists before we add new line)
                backup_prediction_log()
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
            time.sleep(0.5)

        except Exception as e:
            print(f"❌ Error predicting {home} vs {away}: {e}")
            traceback.print_exc()
            continue

    print(f"📊 {len(sent_tips)} tips collected for {tier_name}")
    return sent_tips

# -------------------
# Main entrypoint
# -------------------
last_tier = None

def is_tier_window(current_hour: int):
    """Return the tier name if current_hour falls into a tier window, else None."""
    if 8 <= current_hour <= 10:
        return "Tier 1 (No-time)"
    if 12 <= current_hour <= 14:
        return "Tier 2 (Daytime)"
    if 16 <= current_hour <= 18:
        return "Tier 3 (Evening)"
    return None

def reset_daily_flag_if_new_day():
    """Remove DAILY_FLAG if its date is older than today (so it resets each day)."""
    if DAILY_FLAG.exists():
        try:
            ts = datetime.fromtimestamp(DAILY_FLAG.stat().st_mtime).date()
            if ts < datetime.now().date():
                DAILY_FLAG.unlink()
        except Exception:
            try:
                DAILY_FLAG.unlink()
            except Exception:
                pass

def main():
    # Ensure DB table exists before anything else
    try:
        ensure_table()
    except Exception as e:
        print(f"⚠️  Database initialization failed (continuing): {e}")
        # Don't crash — predictions can still run, just won't send to chats
        # (Telegram messages will fail gracefully in send_telegram_message)
    global last_tier

    # Informational run check
    should_run_now()

    # Ensure directories exist (safe)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_LOG_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # Reset daily flag at new day
    reset_daily_flag_if_new_day()

    # Backup prediction log at start (light operation)
    backup_prediction_log()

    # Ensure historical data is available (weekly)
    historical_ready = ensure_historical_data_exists()

    # Safely fetch fixtures (weekly)
    fixtures_fetched = safe_fetch_fixtures()
    if not fixtures_fetched:
        print("❌ Could not fetch fixtures - continuing with fallback (may be empty)")

    # Determine current Nairobi time and tier
    now = get_now_nairobi()
    current_hour = now.hour
    print(f"🕒 Current Nairobi time: {now.strftime('%Y-%m-%d %H:%M')}")
    tier_name = is_tier_window(current_hour)

    # If not in tier window, do nothing (no Telegram messages)
    if not tier_name:
        print("⏸ Outside tier window — no Telegram messages will be sent.")
        return

    # Load data (historical and today's fixtures)
    if historical_ready:
        historical_df = load_historical_data()
        if historical_df.empty:
            print("⚠️  Historical data loaded but empty - predictions may be limited")
            historical_df = pd.DataFrame()
    else:
        print("⚠️  Historical data not available - using empty DataFrame for predictions")
        historical_df = pd.DataFrame()

    fixtures = load_todays_fixtures()

    # If there are absolutely no fixtures for today:
    if fixtures.empty:
        # Only the first tier of the day should send the "see you tomorrow" message
        if not DAILY_FLAG.exists() and tier_name.startswith("Tier 1"):
            print("📭 No fixtures today — sending single 'see you tomorrow' message (first tier only).")
            # Build header and message
            header = (
                "*scoresignal* curates fixtures from over *15 major European leagues*, leveraging over a decade of data "
                "and advanced machine learning models to deliver probabilistic football insights.\n\n"
                "`" + ("─" * 30) + "`\n\n"
                "🙏 Thank you for your support. *MPESA TILL:* `9105695`\n"
                "*scoresignal* • *Data-driven football tips* • *Bet responsibly*"
            )
            try:
                send_telegram_message(header)
                send_telegram_message("📭 No matches scheduled for today — see you tomorrow!")
                # mark that we've notified for the day
                DAILY_FLAG.touch()
            except Exception as e:
                print(f"⚠️ Could not send 'no matches today' message: {e}")
            return
        else:
            # Either not first tier or already notified; do nothing
            print("📭 No fixtures today and already notified (or this is not first tier) — skipping messaging.")
            return

    # There are fixtures today — proceed to categorize and run predictions for the active tier
    no_time = fixtures[pd.isna(fixtures.get('time')) | (fixtures.get('time') == '') | (fixtures.get('time') == 'TBD')]
    daytime = fixtures[
        (fixtures.get('time').notna()) &
        (fixtures.get('time') != '') &
        (fixtures.get('time') != 'TBD') &
        (pd.to_datetime(fixtures['time'], format='%H:%M', errors='coerce').dt.hour.between(12, 18))
    ]
    evening = fixtures[
        (fixtures.get('time').notna()) &
        (fixtures.get('time') != '') &
        (fixtures.get('time') != 'TBD') &
        (pd.to_datetime(fixtures['time'], format='%H:%M', errors='coerce').dt.hour >= 19)
    ]

    # Send header for this tier (always during tier window)
    header = (
        "*scoresignal* curates fixtures from over *15 major European leagues*, leveraging over a decade of data "
        "and advanced machine learning models to deliver probabilistic football insights.\n\n"
        "`" + ("─" * 30) + "`\n\n"
        "🙏 Thank you for your support. *MPESA TILL:* `9105695`\n"
        "*scoresignal* • *Data-driven football tips*• *Bet responsibly*"
    )
    try:
        send_telegram_message(header)
    except Exception as e:
        print(f"⚠️ Could not send header message: {e}")

    all_sent_tips = []

    # Tier-specific execution
    if tier_name.startswith("Tier 1"):
        print("🕐 Running Tier 1: No-time matches")
        all_sent_tips.extend(run_prediction_tier(tier_name, no_time, historical_df, collect_for_summary=True))
        last_tier = tier_name

    elif tier_name.startswith("Tier 2"):
        print("🕐 Running Tier 2: Daytime matches (12–18)")
        all_sent_tips.extend(run_prediction_tier(tier_name, daytime, historical_df, collect_for_summary=True))
        last_tier = tier_name

    elif tier_name.startswith("Tier 3"):
        print("🕐 Running Tier 3: Evening matches (19+)")
        all_sent_tips.extend(run_prediction_tier(tier_name, evening, historical_df, collect_for_summary=True))
        last_tier = tier_name

    # If we have tips, send summary
    if all_sent_tips:
        try:
            print("🧠 Generating LLM summary...")
            summary = generate_daily_summary(all_sent_tips)
            summary_message = create_summary_message(summary)
            send_telegram_message(summary_message)
            print("✅ LLM summary sent!")
        except Exception as e:
            print(f"❌ LLM summary failed (continuing): {e}")
            traceback.print_exc()
    else:
        # No tips found for this tier — inform subscribers (we already sent header)
        try:
            # Use our next_tier_start helper to give a specific next tier start/time
            next_start_dt, next_tier_label = next_tier_start(now)
            msg = (
                f"📭 In our {tier_name} scan we didn't find high-confidence tips.\n\n"
                f"⏭️ Next predictions will be sent at: {format_time(next_start_dt)} ({next_tier_label})"
            )
            send_telegram_message(msg)
            print(msg)
        except Exception as e:
            print(f"⚠️ Could not send 'no tips in tier' message: {e}")
            traceback.print_exc()

    # If we reached here and we've completed Tier 1 (whether tips or not) and there were fixtures,
    # mark DAILY_FLAG so that the special 'no matches today' message won't be sent later again.
    try:
        if tier_name.startswith("Tier 1"):
            DAILY_FLAG.touch()
    except Exception:
        pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"💥 Critical error in main: {e}")
        traceback.print_exc()
        # Try to send error notification
        try:
            send_telegram_message(f"🚨 Bot crashed: {str(e)[:100]}...")
        except Exception:
            pass
