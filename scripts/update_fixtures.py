# update_fixtures.py
import os
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from src.fetch_fixtures_live import fetch_and_save_fixtures

#if __name__ == "__main__":
    # Get the directory where this script is located
    # This updates your main fixtures.csv file
#    save_fixtures_to_csv("fixtures_data.csv", current_season)
def main():
    current_season = "2025-26"  # Update as needed
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"This script location: {os.path.abspath(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    
    # Save to project root (or adjust as needed)
    output_filepath = SCRIPT_DIR.parent / "data" / "fixtures_data.csv"
    print(f"Saving to: {output_filepath}")
    
    # Use the enhanced function
    fetch_and_save_fixtures(filepath=str(output_filepath), season=current_season)

if __name__ == "__main__":
    main()