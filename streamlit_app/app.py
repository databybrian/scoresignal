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
    page_title="Scoresignal Football Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for compact design
st.markdown("""
    <style>
        /* Reduce top padding */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            max-width: 100%;
        }
        /* Compact header */
        h1 {
            font-size: 1.8rem !important;
            margin-bottom: 0.5rem !important;
            padding-top: 0 !important;
        }
        h2 {
            font-size: 1.4rem !important;
            margin-top: 0.5rem !important;
            margin-bottom: 0.5rem !important;
        }
        h3 {
            font-size: 1.2rem !important;
            margin-top: 0.5rem !important;
            margin-bottom: 0.5rem !important;
        }
        /* Compact metrics */
        [data-testid="stMetricValue"] {
            font-size: 1.2rem;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.8rem;
        }
        /* Compact tables */
        .dataframe {
            font-size: 13px !important;
        }
        /* Remove extra spacing */
        .element-container {
            margin-bottom: 0.5rem;
        }
        /* Compact tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            padding-top: 0;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem 1rem;
            font-size: 0.95rem;
        }
        /* Compact buttons */
        .stButton > button {
            padding: 0.4rem 1rem;
            font-size: 0.9rem;
        }
        /* Compact selectbox */
        .stSelectbox {
            margin-bottom: 0.5rem;
        }
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_FILE = PROJECT_ROOT / "data" / "current_season_leagues_table.csv"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


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
            from src.data_pipeline import ensure_historical_data_exists, save_all_current_tables
            ensure_historical_data_exists(days=7)
            save_all_current_tables()
            st.success("✅ League data refreshed!")
        except Exception as e:
            st.error(f"❌ Failed to refresh league data: {e}")
            st.stop()


# =============================================================================
# PREDICTION LOGIC (SYNCED WITH main.py)
# =============================================================================

def load_models():
    """Load ensemble models exactly as in main.py"""
    MODEL_DIR = PROJECT_ROOT / "model"
    if not MODEL_DIR.exists():
        st.error(f"Model directory not found: {MODEL_DIR}")
        return None

    required_files = [
        "ensemble_hda.pkl",
        "ensemble_btts.pkl",
        "ensemble_over25.pkl",
        "feature_metadata.pkl"
    ]
    
    for f in required_files:
        if not (MODEL_DIR / f).exists():
            st.error(f"Missing model file: {f}")
            return None

    try:
        models = {}
        models['hda'] = joblib.load(MODEL_DIR / "ensemble_hda.pkl")
        models['btts'] = joblib.load(MODEL_DIR / "ensemble_btts.pkl")
        models['over25'] = joblib.load(MODEL_DIR / "ensemble_over25.pkl")
        feature_metadata = joblib.load(MODEL_DIR / "feature_metadata.pkl")
        models.update(feature_metadata)
        st.sidebar.success("✅ Loaded ensemble models (XGBoost + LightGBM + CatBoost)")
        return models
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None


def select_best_tip(hda_proba, btts_proba, over25_proba):
    """
    Exact copy from main.py
    """
    home_prob, draw_prob, away_prob = hda_proba
    btts_yes = btts_proba
    btts_no = 1 - btts_proba
    over_prob = over25_proba
    under_prob = 1 - over25_proba

    hda_home_confidence = max(0, home_prob - 0.53)
    hda_away_confidence = max(0, away_prob - 0.53)
    hda_draw_confidence = max(0, draw_prob - 0.32)
    btts_yes_confidence = max(0, btts_yes - 0.56)
    btts_no_confidence = max(0, btts_no - 0.56)
    over_confidence = max(0, over_prob - 0.56)
    under_confidence = max(0, under_prob - 0.56)

    tips = []
    if hda_home_confidence > 0:
        tips.append(('HDA_HOME', f"🟢 HOME WIN", home_prob, hda_home_confidence, 'primary'))
    if hda_away_confidence > 0:
        tips.append(('HDA_AWAY', f"🔵 AWAY WIN", away_prob, hda_away_confidence, 'primary'))
    if hda_draw_confidence > 0:
        tips.append(('HDA_DRAW', f"🟡 DRAW", draw_prob, hda_draw_confidence, 'secondary'))
    if btts_yes_confidence > 0:
        tips.append(('BTTS_YES', f"⚽ BOTH TEAMS TO SCORE (Yes)", btts_yes, btts_yes_confidence, 'primary'))
    if btts_no_confidence > 0:
        tips.append(('BTTS_NO', f"🚫 BOTH TEAMS TO SCORE (No)", btts_no, btts_no_confidence, 'secondary'))
    if over_confidence > 0:
        tips.append(('OVER25', f"📈 OVER 2.5 GOALS", over_prob, over_confidence, 'primary'))
    if under_confidence > 0:
        tips.append(('UNDER25', f"📉 UNDER 2.5 GOALS", under_prob, under_confidence, 'secondary'))

    if not tips:
        return None, None, 0, False, []

    tips.sort(key=lambda x: x[3], reverse=True)
    best_tip = tips[0]
    tip_type, tip_text, probability, confidence, priority = best_tip
    secondary_tips = [t for t in tips[1:3] if t[4] == 'secondary' or t[3] > 0.05]
    return tip_type, tip_text, probability, True, secondary_tips


def predict_with_ensemble(ensemble_models, X, task='multiclass'):
    """Same as main.py"""
    predictions = []
    if task == 'multiclass':
        predictions.append(ensemble_models['xgb'].predict_proba(X))
        predictions.append(ensemble_models['lgb'].predict_proba(X))
        predictions.append(ensemble_models['cat'].predict_proba(X))
    else:
        predictions.append(ensemble_models['xgb'].predict_proba(X)[:, 1])
        predictions.append(ensemble_models['lgb'].predict_proba(X)[:, 1])
        predictions.append(ensemble_models['cat'].predict_proba(X)[:, 1])
    return np.mean(predictions, axis=0)


def load_prediction_data():
    """Load data, auto-generate if missing (like main.py)"""
    DATA_DIR = PROJECT_ROOT / "data"
    DATA_DIR.mkdir(exist_ok=True)

    # Ensure historical data exists
    if not (DATA_DIR / "cleaned_historical_data.csv").exists():
        st.info("🔄 Generating historical data...")
        from src.data_pipeline import ensure_historical_data_exists
        ensure_historical_data_exists()

    # Ensure fixtures exist
    fixtures_file = DATA_DIR / "fixtures_data.csv"
    from src.data_pipeline import needs_refresh
    if not fixtures_file.exists() or needs_refresh(fixtures_file, days=7):
        st.info("🔄 Fetching live fixtures...")
        try:
            from src.fetch_fixtures_live import fetch_and_save_fixtures
            fetch_and_save_fixtures(str(fixtures_file))
        except Exception as e:
            st.warning(f"⚠️ Failed to fetch fixtures: {e}")
            empty_df = pd.DataFrame(columns=[
                'round', 'date', 'time', 'home_team', 'away_team',
                'home_score', 'away_score', 'league_key', 'league_name', 'season'
            ])
            empty_df.to_csv(fixtures_file, index=False)

    # Load data
    try:
        df_historical = pd.read_csv(DATA_DIR / "cleaned_historical_data.csv")
        df_historical['Date'] = pd.to_datetime(df_historical['Date'], errors='coerce')
        fixtures_df = pd.read_csv(fixtures_file)
        if 'date' in fixtures_df.columns and 'Date' not in fixtures_df.columns:
            fixtures_df.rename(columns={'date': 'Date'}, inplace=True)
        fixtures_df['Date'] = pd.to_datetime(fixtures_df['Date'], errors='coerce')
        return df_historical, fixtures_df
    except Exception as e:
        st.error(f"❌ Failed to load prediction data: {e}")
        return None, None


def safe_feature_computation(historical_df, home_team, away_team, match_date, league_code='E0'):
    """Use the exact same function as main.py"""
    try:
        from src.h2h_form import compute_match_features
        base_features = compute_match_features(
            historical_df=historical_df,
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            league_code=league_code,
            include_odds=False
        )
        all_features = compute_match_features(
            historical_df=historical_df,
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            league_code=league_code,
            include_odds=True
        )
        return base_features, all_features
    except Exception as e:
        st.sidebar.warning(f"❌ Feature computation failed for {home_team} vs {away_team}: {e}")
        return {}, {}


# =============================================================================
# EXISTING UI FUNCTIONS (UNCHANGED)
# =============================================================================

def extract_country_from_league(league_name):
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
    max_pos = df['Pos'].max()
    relegation_start = max_pos - 2
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
        'P': '{:.0f}',
        'W': '{:.0f}',
        'D': '{:.0f}',
        'L': '{:.0f}',
        'GF': '{:.0f}',
        'GA': '{:.0f}',
        'GD': '{:+.0f}',
        'Pts': '{:.0f}',
        'Form': '{:.0f}'
    })
    return styled_df


def create_metrics_cards(league_df):
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
    league_df = league_df.sort_values('GD', ascending=True)
    fig = px.bar(league_df, y='Team', x='GD', color='GD', color_continuous_scale='RdYlGn')
    fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
    return fig


def create_form_analysis(league_df):
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


def create_predictions_table(predictions):
    if not predictions:
        return None
    df = pd.DataFrame(predictions)
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%m/%d %H:%M')
    df['Match'] = df['home_team'] + ' vs ' + df['away_team']
    df['League'] = df['league']
    prob_columns = ['hda_home', 'hda_draw', 'hda_away', 'gg_yes', 'over25']
    for col in prob_columns:
        df[col] = df[col].apply(lambda x: f"{x:.1%}")
    df['Edge'] = df['confidence'].apply(lambda x: f"{x:+.1%}")
    final_columns = [
        'Date', 'Match', 'League', 'hda_home', 'hda_draw', 'hda_away', 
        'gg_yes', 'over25', 'Edge', 'Signal', 'Signal_Reasons'
    ]
    display_df = df[final_columns]
    column_names = {
        'Date': '📅 Date',
        'Match': '⚽ Match',
        'League': '🏆 League',
        'hda_home': '🏠 Home %',
        'hda_draw': '🤝 Draw %', 
        'hda_away': '🚌 Away %',
        'gg_yes': '⚽ BTTS %',
        'over25': '📈 Over 2.5 %',
        'Edge': '💪 Confidence',
        'Signal': '🎯 Signal',
        'Signal_Reasons': '🔍 Why?'
    }
    display_df = display_df.rename(columns=column_names)
    return display_df


def style_predictions_table(df):
    def color_signal(val):
        if 'No Clear Signal' in val:
            return 'background-color: #f8f9fa; color: #6c757d;'
        elif '🟢' in val or '🔵' in val or '🟡' in val:
            return 'background-color: #e8f5e8; color: #000000; font-weight: bold'
        elif '⚽' in val or '📈' in val:
            return 'background-color: #fff3cd; color: #000000; font-weight: bold'
        elif '🚫' in val or '📉' in val:
            return 'background-color: #ffeaea; color: #000000; font-weight: bold'
        else:
            return ''
    def color_edge(val):
        try:
            edge_val = float(val.strip('%'))
            if edge_val > 0:
                return 'background-color: #e8f5e8; color: #000000; font-weight: bold'
            else:
                return ''
        except:
            return ''
    styled_df = df.style.applymap(color_signal, subset=['🎯 Signal'])
    styled_df = styled_df.applymap(color_edge, subset=['💪 Confidence'])
    styled_df = styled_df.set_properties(**{
        'font-size': '12px',
        'font-family': 'Arial, sans-serif',
        'text-align': 'center'
    })
    return styled_df


def prediction_tab():
    st.subheader("AI-Powered Match Predictions")
    st.markdown("Using the **exact same signal logic** as the Telegram bot")

    col1, col2 = st.columns(2)
    with col1:
        days_ahead = st.selectbox("Days ahead to predict", [1, 2, 3, 4, 5, 7], index=2)
    with col2:
        show_all = st.checkbox("Show all matches", value=True)

    with st.spinner("Loading prediction data..."):
        df_historical, fixtures_df = load_prediction_data()
    if df_historical is None or fixtures_df is None:
        st.error("Please ensure both historical data and fixtures data are available.")
        return

    with st.spinner("Loading AI models..."):
        models = load_models()
    if models is None:
        st.error("Failed to load prediction models.")
        return

    today = pd.Timestamp.now().normalize()
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
                base_features, all_features = safe_feature_computation(
                    historical_df=df_historical,
                    home_team=home_team,
                    away_team=away_team,
                    match_date=match['Date'],
                    league_code=match.get('league_code', 'E0')
                )
                if not base_features or not all_features:
                    continue

                X_hda = pd.DataFrame([base_features])[models['hda_features']].fillna(0).values
                X_btts = pd.DataFrame([all_features])[models['btts_features']].fillna(0).values
                X_over25 = pd.DataFrame([all_features])[models['over25_features']].fillna(0).values

                hda_proba = predict_with_ensemble(models['hda'], X_hda, task='multiclass')[0]
                btts_proba = predict_with_ensemble(models['btts'], X_btts, task='binary')[0]
                over25_proba = predict_with_ensemble(models['over25'], X_over25, task='binary')[0]

                tip_type, tip_text, confidence, should_send, secondary_tips = select_best_tip(
                    hda_proba, btts_proba, over25_proba
                )

                signal = tip_text if should_send else "📊 No Clear Signal"
                reasons = [f"Confidence: {confidence:.1%}"] if should_send else ["Below threshold"]

                if should_send:
                    signal_count += 1

                predictions.append({
                    'Date': match['Date'],
                    'home_team': home_team,
                    'away_team': away_team,
                    'league': match.get('league_name', 'Unknown'),
                    'hda_home': hda_proba[0],
                    'hda_draw': hda_proba[1],
                    'hda_away': hda_proba[2],
                    'gg_yes': btts_proba,
                    'over25': over25_proba,
                    'confidence': confidence,
                    'Signal': signal,
                    'Signal_Reasons': ", ".join(reasons)
                })

            except Exception as e:
                st.sidebar.error(f"❌ Prediction failed for {home_team} vs {away_team}: {e}")
                continue

            progress_bar.progress((idx + 1) / len(upcoming))

        progress_bar.empty()
        status_text.empty()

        if predictions:
            st.markdown("### 📊 Prediction Summary")
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            with summary_col1:
                st.metric("Total Analyzed", len(upcoming))
            with summary_col2:
                st.metric("Signals Found", signal_count)
            with summary_col3:
                signal_rate = signal_count / len(upcoming) if len(upcoming) > 0 else 0
                st.metric("Signal Rate", f"{signal_rate:.1%}")

            st.markdown("---")
            predictions_table = create_predictions_table(predictions)
            if predictions_table is not None:
                styled_table = style_predictions_table(predictions_table)
                st.dataframe(styled_table, use_container_width=True)


def load_data():
    ensure_league_data_fresh()
    df = pd.read_csv(DATA_FILE)
    required_cols = {'league_name', 'Pos', 'Team', 'P', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts', 'Form'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        st.error(f"Missing columns in data: {missing}")
        st.stop()
    return df


def main():
    st.markdown("""
    <style>
        .dataframe {
            font-size: 14px !important;
        }
    </style>
    """, unsafe_allow_html=True)
    st.title("Scoresignal Football Analytics Dashboard")
    
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

    tab1, tab2 = st.tabs(["League Standings", "Match Predictions"])

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
                    DATA_FILE.unlink()
                st.cache_data.clear()
                st.rerun()

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
            st.dataframe(styled_df, use_container_width=True, hide_index=True, height=table_height)

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

    with tab2:
        prediction_tab()

    st.markdown("---")
    st.caption("Data updated weekly | Source: football-data.co.uk | Predictions with machine learning")


if __name__ == "__main__":
    main()