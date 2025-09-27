# update_fixtures.py
import os
import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from src.fetch_fixtures_live import save_fixtures_to_csv
from src.league_data import current_season

#if __name__ == "__main__":
    # Get the directory where this script is located
    # This updates your main fixtures.csv file
#    save_fixtures_to_csv("fixtures_data.csv", current_season)
if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    print(f"This script location: {os.path.abspath(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    
    output_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures_data.csv")
    print(f"Saving to: {output_filepath}")
    save_fixtures_to_csv(output_filepath, current_season)