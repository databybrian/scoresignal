# streamlit_app/app.py (Modified for compactness & Railway compatibility)
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Scoresignal Football Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for compact design - MINIMAL PADDING
st.markdown("""
    <style>
        /* Drastically reduce top/bottom padding */
        .block-container {
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
            max-width: 100%;
        }
        /* Compact header */
        h1 {
            font-size: 1.6rem !important;
            margin-bottom: 0.3rem !important;
            padding-top: 0 !important;
        }
        h2 {
            font-size: 1.2rem !important;
            margin-top: 0.3rem !important;
            margin-bottom: 0.3rem !important;
        }
        h3 {
            font-size: 1.1rem !important;
            margin-top: 0.3rem !important;
            margin-bottom: 0.3rem !important;
        }
        /* Compact metrics */
        [data-testid="stMetricValue"] {
            font-size: 1.0rem;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.7rem;
        }
        /* Compact tables */
        .dataframe {
            font-size: 12px !important;
        }
        /* Minimize spacing between elements */
        .element-container {
            margin-bottom: 0.3rem !important;
        }
        /* Compact tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.8rem;
            padding-top: 0;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.4rem 0.8rem;
            font-size: 0.9rem;
        }
        /* Compact buttons */
        .stButton > button {
            padding: 0.3rem 0.8rem;
            font-size: 0.85rem;
        }
        /* Compact selectbox */
        .stSelectbox {
            margin-bottom: 0.3rem;
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
        'font-size': '12px', # Smaller font
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
        st.metric("Teams", total_teams, help="Total Teams in League") # Shorter label
    with cols[1]:
        st.metric("Matches/Team", matches_per_team, help="Matches Played Per Team") # Shorter label
    with cols[2]:
        relegation_positions = f"{total_teams-2}-{total_teams}"
        st.metric("Relegation", relegation_positions, help="Positions 18-20 (or similar)") # Shorter label
    with cols[3]:
        st.metric("Avg Goals/Game", f"{avg_goals_per_game:.2f}", help="Average Goals Scored Per Game") # Shorter label
    with cols[4]:
        st.metric("Avg Points", f"{avg_points:.1f}", help="Average Points Per Team") # Shorter label

def create_goal_difference_bars(league_df):
    league_df = league_df.sort_values('GD', ascending=True)
    fig = px.bar(league_df, y='Team', x='GD', color='GD', color_continuous_scale='RdYlGn')
    fig.update_layout(height=350, yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=30, b=0)) # Tighter margins
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

def load_data():
    ensure_league_data_fresh()
    df = pd.read_csv(DATA_FILE)
    required_cols = {'league_name', 'Pos', 'Team', 'P', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts', 'Form'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        st.error(f"Missing columns in data: {missing}")
        st.stop()
    return df

def load_fixtures_data():
    """Load fixtures data without prediction logic"""
    DATA_DIR = PROJECT_ROOT / "data"
    fixtures_file = DATA_DIR / "fixtures_data.csv"
    if not fixtures_file.exists():
        st.warning("No fixtures data found.")
        return pd.DataFrame()
    try:
        fixtures_df = pd.read_csv(fixtures_file)
        if 'date' in fixtures_df.columns and 'Date' not in fixtures_df.columns:
            fixtures_df.rename(columns={'date': 'Date'}, inplace=True)
        fixtures_df['Date'] = pd.to_datetime(fixtures_df['Date'], errors='coerce')
        return fixtures_df
    except Exception as e:
        st.error(f"❌ Failed to load fixtures data: {e}")
        return pd.DataFrame()

def create_upcoming_fixtures_table(fixtures_df):
    """Create a simple table of upcoming fixtures"""
    if fixtures_df.empty:
        return None
    today = pd.Timestamp.now().normalize()
    upcoming = fixtures_df[
        (fixtures_df['Date'].dt.date >= today.date())
    ].copy()
    if upcoming.empty:
        return None
    upcoming = upcoming.sort_values('Date').reset_index(drop=True)
    # Select relevant columns
    display_df = upcoming[['Date', 'home_team', 'away_team', 'league_name']].copy()
    display_df['Match'] = display_df['home_team'] + ' vs ' + display_df['away_team']
    display_df = display_df.rename(columns={
        'Date': '📅 Date',
        'Match': '⚽ Match',
        'league_name': '🏆 League'
    })
    display_df['📅 Date'] = display_df['📅 Date'].dt.strftime('%m/%d %H:%M') # Format date
    return display_df

def create_fixtures_visualizations(fixtures_df):
    """Create simple visualizations for upcoming fixtures"""
    if fixtures_df.empty:
        return

    today = pd.Timestamp.now().normalize()
    upcoming = fixtures_df[
        (fixtures_df['Date'].dt.date >= today.date())
    ].copy()
    if upcoming.empty:
        st.info("No upcoming fixtures found.")
        return

    # Visualization 1: Matches by League
    league_counts = upcoming['league_name'].value_counts().reset_index()
    league_counts.columns = ['League', 'Number of Matches']
    fig_league = px.bar(league_counts, x='League', y='Number of Matches', title="Upcoming Matches by League")
    fig_league.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))

    # Visualization 2: Matches by Date (Distribution)
    upcoming['Date_only'] = upcoming['Date'].dt.date
    date_counts = upcoming['Date_only'].value_counts().reset_index()
    date_counts.columns = ['Date', 'Number of Matches']
    date_counts = date_counts.sort_values('Date')
    fig_date = px.line(date_counts, x='Date', y='Number of Matches', title="Upcoming Matches Distribution by Date")
    fig_date.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))

    # Visualization 3: Matches by Country (if possible)
    # Assuming you have a way to map league to country, otherwise skip this part.
    # For simplicity, let's assume we can add a 'country' column based on league name.
    def get_country(league_name):
        mapping = {
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
        return mapping.get(league_name, 'Other')

    upcoming['country'] = upcoming['league_name'].apply(get_country)
    country_counts = upcoming['country'].value_counts().reset_index()
    country_counts.columns = ['Country', 'Number of Matches']
    fig_country = px.pie(country_counts, names='Country', values='Number of Matches', title="Upcoming Matches by Country")
    fig_country.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_league, use_container_width=True)
    with col2:
        st.plotly_chart(fig_date, use_container_width=True)
    st.plotly_chart(fig_country, use_container_width=True)

def main():
    st.title("Scoresignal Football Analytics Dashboard")

    # Header with GitHub link and description
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
            <p style="font-size:13px; line-height:1.5; color:#444;">
            Live standings and fixtures for top European leagues.<br>
            <span style="font-size:12px; color:#666;">
            *Predictions temporarily disabled for Railway compatibility*
            </span>
            </p>
            """,
            unsafe_allow_html=True
        )

    tab1, tab2 = st.tabs(["League Standings", "Upcoming Fixtures"])

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
            # Adjust table height dynamically but keep it compact
            table_height = min(500, (num_teams + 1) * 30 + 3) # Reduced row height
            st.dataframe(styled_df, use_container_width=True, hide_index=True, height=table_height)
            st.markdown("---")
            col_analytics1, col_analytics2 = st.columns(2)
            with col_analytics1:
                st.markdown("### Goal Difference")
                st.plotly_chart(create_goal_difference_bars(league_df), use_container_width=True)
            with col_analytics2:
                st.markdown("### Recent Form")
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
        st.subheader("Upcoming Fixtures")
        st.markdown("*Simple view of upcoming matches without AI predictions.*")

        fixtures_df = load_fixtures_data()
        if fixtures_df.empty:
            st.info("No fixture data available.")
        else:
            # Display Upcoming Fixtures Table
            upcoming_table = create_upcoming_fixtures_table(fixtures_df)
            if upcoming_table is not None:
                st.markdown("### 📅 Upcoming Matches")
                st.dataframe(upcoming_table, use_container_width=True, height=300)
            else:
                st.info("No upcoming matches found.")

            # Display Visualizations
            st.markdown("---")
            st.markdown("### 📊 Fixture Insights")
            create_fixtures_visualizations(fixtures_df)

    st.markdown("---")
    st.caption("Data updated weekly | Source: football-data.co.uk | Predictions temporarily disabled for Railway")

if __name__ == "__main__":
    main()