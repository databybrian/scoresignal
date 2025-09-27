# bot/scheduler.py
import schedule
import time
import subprocess
import os
import sys
from datetime import datetime

def run_predictions():
    """Run the main prediction script from project root."""
    try:
        print(f"⚽ Running predictions at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
        
        # Get project root (where main.py is located)
        project_root = os.path.dirname(os.path.abspath(__file__))  # This gives bot directory
        project_root = os.path.dirname(project_root)  # Go up one level to project root
        
        main_script_path = os.path.join(project_root, "main.py")
        
        print(f"📁 Project root: {project_root}")
        print(f"🔎 Main script: {main_script_path}")
        
        # Check if main.py exists
        if not os.path.exists(main_script_path):
            print(f"❌ Error: main.py not found at {main_script_path}")
            return
        
        # Run the script from project root directory
        print("🚀 Starting prediction script...")
        result = subprocess.run(
            [sys.executable, main_script_path],
            cwd=project_root,  # Run from project root so imports work correctly
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Predictions completed successfully")
            if result.stdout:
                print(f"📋 Output: {result.stdout}")
        else:
            print(f"❌ Predictions failed with error code: {result.returncode}")
            if result.stderr:
                print(f"💥 Error: {result.stderr}")
                
    except Exception as e:
        print(f"❌ Unexpected error in scheduler: {e}")

# For testing - run every 2 minutes
# schedule.every(2).minutes.do(run_predictions)

# Production schedules (comment these in when testing is done)
schedule.every().day.at("08:00").do(run_predictions)  # Tier 1
schedule.every().day.at("12:00").do(run_predictions)  # Tier 2
schedule.every().day.at("17:00").do(run_predictions)  # Tier 3

print("✅ Scheduler started. Waiting for jobs...")

while True:
    schedule.run_pending()
    time.sleep(60)