# scripts/save_current_tables.py
import sys
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from src.data_utils import current_season_int

from src.table_builder import build_league_table

def save_all_current_tables():
    # Load data with Date parsing
    df = pd.read_csv(
        SCRIPT_DIR / "data" / "cleaned_historical_data.csv",
        parse_dates=['Date']
    )

    # Get unique leagues with names
    league_info = (
    df[['country', 'league_code', 'league_name']]
    .dropna(subset=['country', 'league_code', 'league_name'])
    .drop_duplicates()
    )

    # ⚠️ ensure consistent season type
    season = current_season_int

    all_tables = []

    for _, row in league_info.iterrows():
        league_code = row['league_code']
        league_name = row['league_name']
        country = row['country']

        print(f"\n Building table for {league_name} ({league_code})...")

        table_df = build_league_table(
            df=df,
            league_code=league_code,
            season=season,
            save_current=True  # Saves individual CSV
        )

        if not table_df.empty:
            table_df['league_code'] = league_code
            table_df['league_name'] = league_name
            table_df['country'] = country

            # Reset index to expose Pos column
            table_df = table_df.reset_index()

            # Reorder columns
            cols = ['country', 'league_code','league_name', 'Pos','Team','P','W','D',
                    'L','GF','GA','GD','Pts','Form']
            table_df = table_df[cols]

            print(f"   ✅ {league_name} table built ({len(table_df)} teams)")

            all_tables.append(table_df)
        else:
            print(f"   ⚠️ No valid data for {league_name} ({league_code}) in season {season}")

    # Save master
    if all_tables:
        master_table = pd.concat(all_tables, ignore_index=True)
        master_table = master_table.sort_values(['league_name','Pos']).reset_index(drop=True)

        master_path = SCRIPT_DIR / "data" / "current_season_leagues_table.csv"
        master_path.parent.mkdir(exist_ok=True, parents=True)

        master_table.to_csv(master_path, index=False)
        print(f"\n✅ Master table saved: {master_path}")
        print(f"📊 Total: {len(master_table)} teams across {len(all_tables)} leagues")
    else:
        print("❌ No tables generated!")

if __name__ == "__main__":
    save_all_current_tables()
