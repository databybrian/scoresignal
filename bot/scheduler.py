# bot/scheduler.py
import schedule
import time
import subprocess

def run_predictions():
    """Run the main prediction script."""
    print("⚽ Running predictions...")
    subprocess.run(["python", "bot/prediction_main.py"])

# Schedules (Nairobi time, local testing you can adjust to every few minutes)
schedule.every().day.at("08:00").do(run_predictions)  # Tier 1
schedule.every().day.at("12:00").do(run_predictions)  # Tier 2
schedule.every().day.at("17:00").do(run_predictions)  # Tier 3

# 🔹 For testing, uncomment this instead of the above fixed times:
# schedule.every(2).minutes.do(run_predictions)

print("✅ Scheduler started. Waiting for jobs...")

while True:
    schedule.run_pending()
    time.sleep(60)
