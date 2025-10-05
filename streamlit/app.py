# streamlit_app/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import joblib
import sys
import numpy as np
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="⚽ Football League Standings",
    page_icon="⚽",
    layout="wide"
)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_FILE = PROJECT_ROOT / "data" / "current_season_leagues_table.csv"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src")) 

from src.data_pipeline import save_all_current_tables

def file_age_days(path: Path) -> float:
    if not path.exists():
        return float("inf")
    return (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds() / 86400.0

def ensure_league_data_fresh():
    """Ensure current_season_leagues_table.csv exists and is <=7 days old."""
    DATA_FILE = PROJECT_ROOT / "data" / "current_season_leagues_table.csv"
    DATA_DIR = PROJECT_ROOT / "data"
    DATA_DIR.mkdir(exist_ok=True)

    if not DATA_FILE.exists() or file_age_days(DATA_FILE) >= 7:
        st.info("📊 League data missing or stale. Refreshing...")
        try:
            # Ensure historical data is ready (needed for league table)
            from src.data_pipeline import ensure_historical_data_exists
            ensure_historical_data_exists(days=7)
            # Generate league table
            save_all_current_tables()
            st.success("✅ League data refreshed!")
        except Exception as e:
            st.error(f"❌ Failed to refresh league data: {e}")
            st.stop()

@st.cache_data(ttl=3600)
def load_data():
    """Load and validate league table data, ensuring it's fresh."""
    ensure_league_data_fresh()  # ← this now works
    df = pd.read_csv(DATA_FILE)
    required_cols = {'league_name', 'Pos', 'Team', 'P', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts', 'Form'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        st.error(f"Missing columns in data: {missing}")
        st.stop()
    return df

    # Check if file is older than 7 days
    file_mod_time = datetime.fromtimestamp(os.path.getmtime(DATA_FILE))
    if datetime.now() - file_mod_time > timedelta(days=7):
        st.info("🔄 League data is older than 7 days. Refreshing...")
        try:
            from src.save_current_tables import save_all_current_tables
            save_all_current_tables()
            st.success("✅ League data refreshed!")
        except Exception as e:
            st.warning(f"⚠️ Failed to refresh data (using stale data): {e}")
        # Don't stop — allow stale data as fallback

def extract_country_from_league(league_name):
    """Extract country name from league name."""
    league_country_map = {
        'Premier League': 'England',
        'Championship': 'England',
        'La Liga': 'Spain',
        'Serie A': 'Italy',
        'Bundesliga': 'Germany',
        'Ligue 1': 'France',
        'Eredivisie': 'Netherlands',
        'Primeira Liga': 'Portugal',
        'Scottish Premiership': 'Scotland'
    }
    
    if league_name in league_country_map:
        return league_country_map[league_name]
    
    for country in ['England', 'Spain', 'Italy', 'Germany', 'France', 'Netherlands', 'Portugal', 'Scotland']:
        if country.lower() in league_name.lower():
            return country
    
    return 'Other'

def style_table(df: pd.DataFrame) -> pd.DataFrame:
    """Apply conditional formatting to league table with improved colors."""
    max_pos = df['Pos'].max()
    relegation_start = max_pos - 2  # Bottom 3
    
    def highlight_rows(row):
        styles = [''] * len(row)
        if row['Pos'] <= 6:
            styles = ['background-color: #e8f5e8; color: #000000'] * len(row)
        elif row['Pos'] >= relegation_start:
            styles = ['background-color: #ffeaea; color: #000000'] * len(row)
        return styles
    
    styled_df = df.style.apply(highlight_rows, axis=1)
    styled_df = styled_df.set_properties(**{
        'font-size': '14px',
        'font-family': 'Arial, sans-serif',
        'text-align': 'center'
    })
    
    styled_df = styled_df.format({
        'Pos': '{:.0f}',
        'Pts': '{:.0f}',
        'GD': '{:+.0f}'
    })
    
    return styled_df

def create_metrics_cards(league_df):
    """Create key metrics cards for the league."""
    total_teams = len(league_df)
    matches_per_team = league_df['P'].iloc[0]
    avg_goals_per_game = (league_df['GF'].sum() + league_df['GA'].sum()) / (league_df['P'].sum() / 2)
    avg_points = league_df['Pts'].mean()
    
    cols = st.columns(5)
    with cols[0]:
        st.metric("Total Teams", total_teams)
    with cols[1]:
        st.metric("Matches Played/Team", matches_per_team)
    with cols[2]:
        relegation_positions = f"{total_teams-2}-{total_teams}"
        st.metric("Relegation Positions", relegation_positions)
    with cols[3]:
        st.metric("Avg Goals/Game", f"{avg_goals_per_game:.2f}")
    with cols[4]:
        st.metric("Avg Points/Team", f"{avg_points:.1f}")

def create_goal_difference_bars(league_df):
    """Create goal difference bar chart."""
    league_df = league_df.sort_values('GD', ascending=True)
    fig = px.bar(league_df, y='Team', x='GD', 
                 color='GD',
                 color_continuous_scale='RdYlGn')
    fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
    return fig

def create_form_analysis(league_df):
    """Analyze and display recent form from Form column."""
    form_stats = []
    
    for _, team in league_df.iterrows():
        form_string = str(team['Form']).strip()
        
        try:
            form_points = int(float(form_string))
        except (ValueError, TypeError):
            form_points = 0
        
        remaining_points = form_points
        wins = min(remaining_points // 3, 5)
        remaining_points -= wins * 3
        draws = min(remaining_points, 5 - wins)
        losses = 5 - wins - draws
        
        form_stats.append({
            'Team': team['Team'],
            'Wins': wins,
            'Draws': draws,
            'Losses': losses,
            'Form_Points': form_points,
            'Form_String': form_string
        })
    
    form_df = pd.DataFrame(form_stats)
    return form_df.sort_values('Form_Points', ascending=False)

# =============================================================================
# PREDICTION TAB FUNCTIONS - REFACTORED TO MATCH MAIN.PY
# =============================================================================

def load_models():
    """Load trained models and feature columns using main.py structure."""
    MODEL_DIR = PROJECT_ROOT / "model"
    
    if not MODEL_DIR.exists():
        st.error(f"Model directory not found: {MODEL_DIR}")
        return None
    
    required_files = {
        'hda': "football_model_hda.pkl",
        'gg': "football_model_gg.pkl", 
        'over25': "football_model_over25.pkl",
        'feature_cols': "feature_columns.pkl",
        'value_map': "value_alert_map.pkl"
    }
    
    # Check for missing files
    missing_files = []
    for key, filename in required_files.items():
        if not (MODEL_DIR / filename).exists():
            missing_files.append(filename)
    
    if missing_files:
        st.error(f"Missing model files: {', '.join(missing_files)}")
        return None
    
    try:
        models = {}
        for key, filename in required_files.items():
            models[key] = joblib.load(MODEL_DIR / filename)
        
        st.sidebar.success(f"✅ Loaded {len(models)} models")
        return models
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None

def get_market_baseline(my_prob, value_map):
    """Get market baseline for value detection - matches main.py logic."""
    bins = np.arange(0, 1.05, 0.05)
    bin_idx = np.digitize([my_prob], bins) - 1
    if bin_idx[0] >= len(bins) - 1:
        bin_idx[0] = len(bins) - 2
    bin_key = pd.Interval(bins[bin_idx[0]], bins[bin_idx[0] + 1], closed='right')
    return value_map.get(bin_key, my_prob)

def should_send_tip(hda_proba, gg_proba, over25_proba, edge):
    """Use the exact same logic as main.py for signal detection."""
    home_win, draw, away_win = hda_proba
    
    hda_clear = (home_win >= 0.52) or (away_win >= 0.52) or (draw >= 0.35)
    gg_clear = (gg_proba >= 0.54) or (gg_proba <= 0.32)
    ou_clear = (over25_proba >= 0.54) or (over25_proba <= 0.32)
    strong_edge = edge >= 0.02
    
    result = hda_clear or gg_clear or ou_clear or strong_edge
    
    # Log why signal was triggered
    reasons = []
    if result:
        if home_win >= 0.48: reasons.append(f"Home win {home_win:.1%} >= 48%")
        if away_win >= 0.48: reasons.append(f"Away win {away_win:.1%} >= 48%") 
        if draw >= 0.35: reasons.append(f"Draw {draw:.1%} >= 35%")
        if gg_proba >= 0.52: reasons.append(f"BTTS Yes {gg_proba:.1%} >= 52%")
        if gg_proba <= 0.32: reasons.append(f"BTTS No {gg_proba:.1%} <= 32%")
        if over25_proba >= 0.52: reasons.append(f"Over 2.5 {over25_proba:.1%} >= 52%")
        if over25_proba <= 0.32: reasons.append(f"Under 2.5 {over25_proba:.1%} <= 32%")
        if edge >= 0.01: reasons.append(f"Edge {edge:.1%} >= 1%")
    else:
        reasons = ["No criteria met"]
    
    return result, reasons

def get_best_signal(hda_proba, gg_proba, over25_proba, edge):
    """Determine the best signal based on main.py criteria."""
    home_win, draw, away_win = hda_proba
    
    signals = []
    
    # HDA signals (using main.py thresholds)
    if home_win >= 0.48:
        signals.append(('🟢 HOME (Strong Home Favorite)', home_win, 'hda'))
    if away_win >= 0.48:
        signals.append(('🔵 AWAY (Strong Away Win)', away_win, 'hda'))
    if draw >= 0.35:
        signals.append(('🟡 DRAW (High Draw Probability)', draw, 'hda'))
    
    # GG signals
    if gg_proba >= 0.52:
        signals.append(('🟢 GG (Both Teams to Score)', gg_proba, 'gg'))
    elif gg_proba <= 0.32:
        signals.append(('🔴 NG (No Goals Expected)', 1-gg_proba, 'gg'))
    
    # Over/Under signals
    if over25_proba >= 0.52:
        signals.append(('🟢 OVER 2.5 (High-Scoring Game)', over25_proba, 'ou'))
    elif over25_proba <= 0.32:
        signals.append(('🔴 UNDER 2.5 (Low-Scoring Game)', 1-over25_proba, 'ou'))
    
    # Edge-based signals
    if edge >= 0.01:
        best_hda = max([('🏠 Home', home_win), ('🤝 Draw', draw), ('🚌 Away', away_win)], key=lambda x: x[1])
        signals.append((f'💰 {best_hda[0]} (Value)', best_hda[1], 'value'))
    
    if signals:
        best_signal = max(signals, key=lambda x: x[1])
        return best_signal[0]
    else:
        return "📊 No Clear Signal"

def load_prediction_data():
    """Load historical and fixtures data for predictions using main.py paths."""
    DATA_DIR = PROJECT_ROOT / "data"
    
    historical_file = DATA_DIR / "cleaned_historical_data.csv"
    fixtures_file = DATA_DIR / "fixtures_data.csv"
    
    if not historical_file.exists():
        st.error(f"❌ Historical data file not found: {historical_file}")
        return None, None
    
    if not fixtures_file.exists():
        st.error(f"❌ Fixtures data file not found: {fixtures_file}")
        return None, None
    
    try:
        df_historical = pd.read_csv(historical_file)
        fixtures_df = pd.read_csv(fixtures_file)
        
        # Handle date conversion like main.py
        df_historical['Date'] = pd.to_datetime(df_historical['Date'], errors='coerce')
        
        # Handle fixtures date columns
        if 'Date' in fixtures_df.columns:
            fixtures_df['Date'] = pd.to_datetime(fixtures_df['Date'], errors='coerce')
        elif 'date' in fixtures_df.columns:
            fixtures_df['Date'] = pd.to_datetime(fixtures_df['date'], errors='coerce')
        else:
            fixtures_df['Date'] = pd.NaT
        
        st.sidebar.success(f"✅ Loaded {len(df_historical)} historical records")
        st.sidebar.success(f"✅ Loaded {len(fixtures_df)} fixtures")
        return df_historical, fixtures_df
        
    except Exception as e:
        st.error(f"Failed to load prediction data: {e}")
        return None, None

def safe_feature_computation(historical_df, home_team, away_team, match_date, league_code='E0'):
    """Safely compute match features using the same function as main.py."""
    try:
        # Import the exact same function used in main.py
        from features.h2h_form import compute_match_features
        
        # Use the same function signature as main.py
        features = compute_match_features(
            historical_df=historical_df,
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            league_code=league_code
        )
        
        # Convert features dict to DataFrame for model input
        feature_df = pd.DataFrame([features])
        
        return features, feature_df
    except Exception as e:
        st.sidebar.warning(f"❌ Feature computation failed for {home_team} vs {away_team}: {e}")
        # Return empty features and dataframe
        return {}, pd.DataFrame()

def create_predictions_table(predictions):
    """Create a comprehensive table format for predictions."""
    if not predictions:
        return None
    
    df = pd.DataFrame(predictions)
    
    # Format for display
    display_df = df.copy()
    display_df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%m/%d %H:%M')
    display_df['Match'] = df['home_team'] + ' vs ' + df['away_team']
    display_df['League'] = df['league']
    
    # Format probabilities as percentages
    prob_columns = ['hda_home', 'hda_draw', 'hda_away', 'gg_yes', 'over25']
    for col in prob_columns:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}")
    
    # Add edge information
    display_df['Edge'] = df['edge'].apply(lambda x: f"{x:+.1%}")
    
    # Select and order columns for display
    final_columns = [
        'Date', 'Match', 'League', 'hda_home', 'hda_draw', 'hda_away', 
        'gg_yes', 'over25', 'Edge', 'Signal', 'Signal_Reasons'
    ]
    
    display_df = display_df[final_columns]
    
    # Rename columns for better readability
    column_names = {
        'Date': '📅 Date',
        'Match': '⚽ Match',
        'League': '🏆 League',
        'hda_home': '🏠 Home %',
        'hda_draw': '🤝 Draw %', 
        'hda_away': '🚌 Away %',
        'gg_yes': '⚽ BTTS %',
        'over25': '📈 Over 2.5 %',
        'Edge': '💰 Edge',
        'Signal': '🎯 Signal',
        'Signal_Reasons': '🔍 Why?'
    }
    
    display_df = display_df.rename(columns=column_names)
    return display_df

def style_predictions_table(df):
    """Apply styling to the predictions table with better signal highlighting."""
    def color_signal(val):
        if 'No Clear Signal' in val:
            return 'background-color: #f8f9fa; color: #6c757d;'
        elif '🟢' in val or '💰' in val:
            return 'background-color: #e8f5e8; color: #000000; font-weight: bold'
        elif '🔵' in val or '🟡' in val:
            return 'background-color: #fff3cd; color: #000000; font-weight: bold'
        elif '🔴' in val:
            return 'background-color: #ffeaea; color: #000000; font-weight: bold'
        else:
            return ''
    
    def color_edge(val):
        try:
            edge_val = float(val.strip('%'))
            if edge_val > 0:
                return 'background-color: #e8f5e8; color: #000000; font-weight: bold'
            elif edge_val < 0:
                return 'background-color: #ffeaea; color: #000000;'
            else:
                return ''
        except:
            return ''
    
    styled_df = df.style.applymap(color_signal, subset=['🎯 Signal'])
    styled_df = styled_df.applymap(color_edge, subset=['💰 Edge'])
    
    styled_df = styled_df.set_properties(**{
        'font-size': '12px',
        'font-family': 'Arial, sans-serif',
        'text-align': 'center'
    })
    
    return styled_df

def prediction_tab():
    """Refactored prediction tab using main.py logic."""
    st.subheader("🔮 AI-Powered Match Predictions")
    st.markdown("Using the **exact same signal logic** as the Telegram bot")
    
    # Configuration - removed confidence slider, fixed at 48%
    MIN_CONFIDENCE = 48  # Fixed minimum confidence
    
    col1, col2 = st.columns(2)
    with col1:
        days_ahead = st.selectbox("Days ahead to predict", [1, 2, 3, 4, 5, 7], index=2)
    with col2:
        show_all = st.checkbox("Show all matches", value=True, help="Show all matches, not just signals")
    
    # Load prediction data
    with st.spinner("Loading prediction data..."):
        df_historical, fixtures_df = load_prediction_data()
    
    if df_historical is None or fixtures_df is None:
        st.error("Please ensure both historical data and fixtures data are available.")
        return
    
    # Load models
    with st.spinner("Loading AI models..."):
        models = load_models()
    
    if models is None:
        st.error("Failed to load prediction models. Please ensure models are trained and available.")
        return
    
    # Filter fixtures for today + specified days ahead
    today = pd.Timestamp.now(tz='UTC').normalize()
    end_date = today + timedelta(days=days_ahead)
    
    upcoming = fixtures_df[
        (fixtures_df['Date'].dt.date >= today.date()) & 
        (fixtures_df['Date'].dt.date <= end_date.date())
    ].copy()
    
    if upcoming.empty:
        st.info(f"🎉 No upcoming matches found in the next {days_ahead} days.")
        return
    
    upcoming = upcoming.sort_values('Date').reset_index(drop=True)
    
    st.success(f"📊 Found **{len(upcoming)} matches** in the next {days_ahead} days")
    
    # Generate predictions
    if st.button("🎯 Generate Predictions", type="primary", use_container_width=True):
        predictions = []
        signal_count = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, match in upcoming.iterrows():
            home_team = match.get('home_team', 'Unknown')
            away_team = match.get('away_team', 'Unknown')
            
            status_text.text(f"Processing {idx+1}/{len(upcoming)}: {home_team} vs {away_team}")
            
            try:
                # Compute features using the same function as main.py
                features, feature_df = safe_feature_computation(
                    historical_df=df_historical,
                    home_team=home_team,
                    away_team=away_team,
                    match_date=match['Date'],
                    league_code=match.get('league_code', 'E0')
                )
                
                # Skip if feature computation failed
                if feature_df.empty:
                    continue
                
                # Ensure all feature columns are present (like main.py)
                if 'feature_cols' in models:
                    X = feature_df.reindex(columns=models['feature_cols'], fill_value=0)
                else:
                    X = feature_df.copy()
                
                # Get predictions (same as main.py)
                hda_proba = models['hda'].predict_proba(X)[0]
                gg_proba = models['gg'].predict_proba(X)[0][1]
                over25_proba = models['over25'].predict_proba(X)[0][1]
                
                # Calculate edge (same as main.py)
                home_win_prob = hda_proba[0]
                market_baseline = get_market_baseline(home_win_prob, models.get('value_map', {}))
                edge = home_win_prob - market_baseline
                
                # Determine if we should send tip (same logic as main.py)
                should_tip, signal_reasons = should_send_tip(hda_proba, gg_proba, over25_proba, edge)
                
                # Get best signal
                signal = get_best_signal(hda_proba, gg_proba, over25_proba, edge)
                
                if should_tip:
                    signal_count += 1
                
                # Apply fixed confidence filter (52%)
                max_prob = max(hda_proba[0], hda_proba[1], hda_proba[2], gg_proba, over25_proba)
                meets_confidence = (max_prob * 100) >= MIN_CONFIDENCE
                
                prediction_data = {
                    'Date': match['Date'],
                    'home_team': home_team,
                    'away_team': away_team,
                    'league': match.get('league_name', match.get('league', 'Unknown')),
                    'hda_home': hda_proba[0],
                    'hda_draw': hda_proba[1],
                    'hda_away': hda_proba[2],
                    'gg_yes': gg_proba,
                    'over25': over25_proba,
                    'edge': edge,
                    'should_tip': should_tip,
                    'Signal': signal,
                    'Signal_Reasons': ", ".join(signal_reasons),
                    'Meets_Confidence': meets_confidence
                }
                
                # Only add to predictions based on filters
                if show_all or (should_tip and meets_confidence):
                    predictions.append(prediction_data)
                
            except Exception as e:
                st.sidebar.error(f"❌ Prediction failed for {home_team} vs {away_team}: {e}")
                continue
            
            progress_bar.progress((idx + 1) / len(upcoming))
        
        progress_bar.empty()
        status_text.empty()
        
        if predictions:
            # Enhanced Summary Statistics
            st.markdown("### 📊 Prediction Summary")
            
            summary_col1, summary_col2, summary_col3, summary_col4, summary_col5 = st.columns(5)
            
            with summary_col1:
                st.metric(
                    "Total Analyzed", 
                    len(upcoming),
                    help="Number of matches processed"
                )
            with summary_col2:
                st.metric(
                    "Signals Found", 
                    signal_count,
                    delta=f"{signal_count/len(upcoming):.1%}" if len(upcoming) > 0 else "0%",
                    delta_color="normal" if signal_count > 0 else "off",
                    help="High-confidence betting signals detected"
                )
            with summary_col3:
                st.metric(
                    "Displayed", 
                    len(predictions),
                    help="Matches shown in the table below"
                )
            with summary_col4:
                signal_rate = signal_count/len(upcoming) if len(upcoming) > 0 else 0
                st.metric(
                    "Signal Rate", 
                    f"{signal_rate:.1%}",
                    help="Percentage of matches with betting signals"
                )
            with summary_col5:
                avg_confidence = np.mean([max(p['hda_home'], p['hda_draw'], p['hda_away']) for p in predictions]) if predictions else 0
                st.metric(
                    "Avg Confidence", 
                    f"{avg_confidence:.1%}",
                    help="Average maximum probability across displayed matches"
                )
            
            st.markdown("---")
            
            # Quick insights based on the results
            if signal_count == 0:
                st.info("🔍 **Insight**: No strong signals detected. Consider adjusting confidence thresholds or check back later.")
            elif signal_count / len(upcoming) < 0.1:
                st.warning("⚠️ **Insight**: Low signal rate detected. Markets may be efficient today.")
            elif signal_count / len(upcoming) > 0.3:
                st.success("**Insight**: High signal rate! Multiple betting opportunities identified.")
            
            # Display the predictions table
            st.markdown("### Match Predictions & Signals")
            
            predictions_table = create_predictions_table(predictions)
            if predictions_table is not None:
                styled_table = style_predictions_table(predictions_table)
                
                st.dataframe(
                    styled_table,
                    use_container_width=True,
                    height=min(600, (len(predictions_table) + 1) * 35 + 3)
                )

def main():
    st.markdown("""
    <style>
        .dataframe {
            font-size: 14px !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("⚽ Scoresignal Football Analytics Dashboard")
    
    # Header
    header_col1, header_col2 = st.columns([3, 2])
    
    with header_col1:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 8px;">
                <a href="https://github.com/databybryan" target="_blank">
                    <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" 
                         width="22" height="22" style="vertical-align:middle;">
                </a>
                <span style="font-size:14px;">
                    Original project by <a href="https://github.com/databybrian/scoresignal" target="_blank">
                    <b>databybryan</b></a>
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <p style="font-size:15px; line-height:1.5; color:#444;">
            Live standings and predictions powered by <b>machine learning</b> for top European leagues.<br>
            <span style="font-size:13px; color:#666;">
            *Using the same signal logic as our Telegram bot*
            </span>
            </p>
            """,
            unsafe_allow_html=True
        )
    
    # Create tabs
    tab1, tab2 = st.tabs(["League Standings", "Match Predictions"])
    
    # TAB 1: LEAGUE STANDINGS (UNCHANGED)
    with tab1:
        df = load_data()
        
        if 'country' not in df.columns:
            df['country'] = df['league_name'].apply(extract_country_from_league)
        
        col1, col2 = st.columns(2)
        
        with col1:
            countries = sorted(df['country'].unique())
            selected_country = st.selectbox("Select Country", ["All"] + countries, index=0, key="standings_country")
        
        with col2:
            if selected_country == "All":
                available_leagues = sorted(df['league_name'].unique())
            else:
                available_leagues = sorted(df[df['country'] == selected_country]['league_name'].unique())
            
            selected_league = st.selectbox("Select League", available_leagues, index=0, key="standings_league")
            if st.button("🔄 Force Refresh League Data", key="refresh_data"):
                if DATA_FILE.exists():
                    DATA_FILE.unlink()  # Delete to force regeneration
                st.cache_data.clear()  # Clear Streamlit cache
                st.rerun()
        
        league_df = df[df['league_name'] == selected_league].copy()
        
        league_df = df[df['league_name'] == selected_league].copy()
        league_df = league_df.sort_values('Pos').reset_index(drop=True)
        
        if not league_df.empty:
            country_name = league_df['country'].iloc[0]
            st.subheader(f"{selected_league} - {country_name}")
            
            create_metrics_cards(league_df)
            
            st.markdown("### League Standings")
            display_cols = ['Pos', 'Team', 'P', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts', 'Form']
            styled_df = style_table(league_df[display_cols])
            num_teams = len(league_df)
            table_height = min(600, (num_teams + 1) * 35 + 3)
            st.dataframe(
                styled_df, 
                use_container_width=True,
                hide_index=True,
                height=table_height
            )

            st.markdown("---")
            
            col_analytics1, col_analytics2 = st.columns(2)
            
            with col_analytics1:
                st.markdown("### Goal Difference by Team")
                st.plotly_chart(create_goal_difference_bars(league_df), use_container_width=True)
            
            with col_analytics2:
                st.markdown("### Recent Form (Last 5 Games)")
                form_df = create_form_analysis(league_df)
                
                top_form_teams = form_df.head(3)
                bottom_form_teams = form_df.tail(3)
                
                st.markdown("**Best Form:**")
                form_cols1 = st.columns(3)
                for idx, (_, team) in enumerate(top_form_teams.iterrows()):
                    with form_cols1[idx]:
                        st.metric(
                            team['Team'], 
                            f"{team['Form_Points']} pts",
                            delta=f"{team['Wins']}W-{team['Draws']}D-{team['Losses']}L"
                        )
                
                st.markdown("**Worst Form:**")
                form_cols2 = st.columns(3)
                for idx, (_, team) in enumerate(bottom_form_teams.iterrows()):
                    with form_cols2[idx]:
                        st.metric(
                            team['Team'], 
                            f"{team['Form_Points']} pts",
                            delta=f"{team['Wins']}W-{team['Draws']}D-{team['Losses']}L",
                            delta_color="inverse"
                        )
    
    # TAB 2: MATCH PREDICTIONS (REFACTORED)
    with tab2:
        prediction_tab()
    
    # Footer
    st.markdown("---")
    st.caption("Data updated weekly | Source: football-data.co.uk | Predictions with machine learning")

if __name__ == "__main__":
    main()