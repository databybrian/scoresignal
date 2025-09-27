# bot/polling.py
import requests
from pathlib import Path
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from bot.time_utils import get_now_nairobi, format_time, get_next_tips_time
from bot.chat_manager import add_chat_id
from bot.telegram_bot import send_telegram_message
from bot.config import TELEGRAM_BOT_TOKEN

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
NAIROBI_TZ = ZoneInfo("Africa/Nairobi")


def poll_updates(offset=None):
    url = f"{BASE_URL}/getUpdates"
    params = {"timeout": 5, "offset": offset}
    resp = requests.get(url, params=params, timeout=10).json()
    return resp.get("result", [])


def run_polling():
    print("🚀 Starting polling loop...")
    offset = None
    while True:
        now = get_now_nairobi()
        print(f"🕒 Current Nairobi time: {format_time(now)}")

        updates = poll_updates(offset)
        for update in updates:
            offset = update["update_id"] + 1
            if "message" in update:
                chat_id = str(update["message"]["chat"]["id"])
                text = update["message"].get("text", "").strip()

                if text == "/start":
                    add_chat_id(chat_id)
                    send_telegram_message(
                        "🤖 Welcome to scoresignal!\n\n"
                        "You'll receive daily football predictions with:\n"
                        "• High-confidence tips\n"
                        "• Value alerts\n"
                        "• Daily summaries\n\n"
                        "💡 Bet responsibly",
                        chat_id=chat_id
                    )

                elif text == "/nexttips":
                    next_time = get_next_tips_time(now)
                    send_telegram_message(
                        f"⏭️ The next tips will be sent at {format_time(next_time)} Nairobi time.",
                        chat_id=chat_id
                    )


if __name__ == "__main__":
    run_polling()
