# bot/time_utils.py
import time
from datetime import datetime, time as dt_time, timedelta
from zoneinfo import ZoneInfo

NAIROBI_TZ = ZoneInfo("Africa/Nairobi")


def get_now_nairobi():
    """Return current Nairobi time (timezone-aware)."""
    return datetime.now(NAIROBI_TZ)

def format_time(dt):
    """Format datetime for user-friendly output (with date)."""
    return dt.strftime("%Y-%m-%d %H:%M")

def get_next_tips_time(now=None):
    """
    Get the next tier prediction time in Nairobi timezone.
    Tiers:
      - Tier 1: 07:00–09:00
      - Tier 2: 12:00–14:00 (shifted window)
      - Tier 3: 16:00–18:00
    """
    if now is None:
        now = get_now_nairobi()
    
    today = now.date()

    # Define tier start times using proper datetime construction
    tier_times = [
        datetime(today.year, today.month, today.day, 7, 0, tzinfo=NAIROBI_TZ),
        datetime(today.year, today.month, today.day, 12, 0, tzinfo=NAIROBI_TZ),
        datetime(today.year, today.month, today.day, 16, 0, tzinfo=NAIROBI_TZ),
    ]

    # Check for next tier time today
    for tier_start in tier_times:
        if now < tier_start:
            return tier_start

    # If past all tiers today → tomorrow 07:00
    tomorrow = today + timedelta(days=1)
    return datetime(tomorrow.year, tomorrow.month, tomorrow.day, 7, 0, tzinfo=NAIROBI_TZ)

def format_time(dt):
    """Pretty format Nairobi datetime for messages."""
    return dt.strftime("%Y-%m-%d %H:%M Nairobi time")