# main.py
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import pandas as pd
import numpy as np
import time
import pytz
from datetime import datetime, timedelta
import joblib
from features.h2h_form import compute_match_features
from bot.telegram_bot import send_telegram_message
from bot.llm_summary import generate_daily_summary, create_summary_message
from bot.time_utils import get_next_tips_time, format_time, get_now_nairobi


# Paths
DATA_DIR = SCRIPT_DIR / "data"
MODEL_DIR = SCRIPT_DIR / "model"
PREDICTION_LOG = DATA_DIR / "prediction_log.csv"

# Load models once at startup
print("🚀 Loading models...")
MODELS = {
    'hda': joblib.load(MODEL_DIR / "football_model_hda.pkl"),
    'gg': joblib.load(MODEL_DIR / "football_model_gg.pkl"),
    'over25': joblib.load(MODEL_DIR / "football_model_over25.pkl"),
    'value_map': joblib.load(MODEL_DIR / "value_alert_map.pkl"),
    'feature_cols': joblib.load(MODEL_DIR / "feature_columns.pkl")
}
print("✅ Models loaded!")

def load_historical_data():
    """Load clean historical data for feature computation."""
    df = pd.read_csv(
        DATA_DIR / "cleaned_historical_data.csv",
        low_memory=False,
        dtype={'FTHG': 'Int64', 'FTAG': 'Int64'}
    )
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def load_todays_fixtures():
    """Load today's fixtures."""
    fixtures = pd.read_csv(DATA_DIR / "fixtures_data.csv")
    fixtures['Date'] = pd.to_datetime(fixtures['Date'])
    today = pd.Timestamp.now(tz='UTC').normalize()
    return fixtures[fixtures['Date'].dt.date == today.date()].copy()

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
        return False # Handle empty/corrupted log

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
    strong_edge = edge >= 0.02 # 2% edge threshold
    
    # Send if any condition is met
    return hda_clear or gg_clear or ou_clear or strong_edge

def build_prediction_message(match_row, historical_df):
    """Build a rich, high-confidence Telegram prediction message. Returns None if low-confidence."""
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
        #f"{separator}\n"
        f"{league_name}\n"
        f"{time_str}\n"
        f"{home} vs {away}\n"
        f"{best_tip}\n\n"
        f"🏠 Home: {hda_proba[0]:.0%} | 🤝 Draw: {hda_proba[1]:.0%} | 🚌 Away: {hda_proba[2]:.0%}\n"
        f"⚽⚽ Both Teams to Score: {gg_proba:.0%}\n"
        f"⚽⚽⚽ Over 2.5 Goals: {over25_proba:.0%}\n"
    )
    
    # Add value alert if edge is strong (≥8%)
    if edge > 0.02:
        message += (
            f"\n\n⚠️ HIGH VALUE ALERT!\n"
            f"Model: {home_win_prob:.0%} Home | Market Baseline: ~{market_baseline:.0%}\n"
            f"→ {edge:+.0%} edge (Our Model vs Historical Average"
        )
    
    # Responsible gambling footer
    message += (f"{separator}\n" 
                f"💡 Bet responsibly || scoresignal")
    
    return message

def run_prediction_tier(tier_name, fixtures_subset, historical_df, collect_for_summary=False):
    """
    Run prediction for a specific tier.
    
    Args:
        tier_name (str): Name of the prediction tier
        fixtures_subset (pd.DataFrame): Subset of fixtures to process
        historical_df (pd.DataFrame): Historical match data for feature computation
        collect_for_summary (bool): Whether to collect tip data for LLM summary

    Returns:
        list: Sent tips (each tip is a dict) — empty if none
    """
    if fixtures_subset.empty:
        print(f"⏭️  No matches for {tier_name}")
        return []  # Always return a list

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

            # Skip low-confidence tips
            if message is None:
                print(f"⏭️  Skipped (low confidence): {home} vs {away}")
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
                # Don’t break loop

            # Be kind to APIs
            time.sleep(1)

        except Exception as e:
            print(f"❌ Error predicting {home} vs {away}: {e}")
            continue  # Continue with next match

    print(f"📊 {len(sent_tips)} tips collected for {tier_name}")
    return sent_tips  # Always a list


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
        return [0.33, 0.33, 0.33]  # Default fallback

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


last_tier = None

def main():
    # Ensure data directory exists
    global last_tier  # ← Must be first line in function
    
    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)
    
    # Generate fixtures if missing
    if not (DATA_DIR / "fixtures_data.csv").exists():
        print("🔄 fixtures_data.csv not found - fetching live fixtures...")
        try:
            from src.fetch_fixtures_live import fetch_and_save_fixtures
            fetch_and_save_fixtures()
            print("✅ Fixtures fetched successfully")
        except Exception as e:
            print(f"❌ Failed to fetch fixtures: {e}")
            return

    # Send header
    header = (
        "scoresignal uses 10+ years of historical data and machine learning "
        "to curate probabilistic football tips.\n\n"
        "⚠️ Disclaimer: Predictions are probabilistic from our model, not guarantees. "
        "For support MPESA Till: 9105695. \n\n"
        "scoresignal || Data-driven football tips"
    )
    send_telegram_message(header)

    # Load data
    historical_df = load_historical_data()
    fixtures = load_todays_fixtures()

    if fixtures.empty:
        print("😴 No fixtures for today")
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
            from bot.llm_summary import generate_daily_summary, create_summary_message
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
    main()