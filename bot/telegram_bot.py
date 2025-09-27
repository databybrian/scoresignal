# bot/telegram_bot.py
"""
Telegram bot messaging module.
Handles sending predictions or other messages to Telegram chats.
"""

import requests
from .config import TELEGRAM_BOT_TOKEN
from .chat_manager import get_active_chat_ids

# Reuse one session for efficiency
session = requests.Session()
API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


def send_telegram_message(message: str, chat_id: str = None) -> None:
    """
    Send a Telegram message.
    
    If `chat_id` is provided, sends to that single chat.
    Otherwise, broadcasts to all active chats.
    """
    if not TELEGRAM_BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not configured")
        return

    # Decide recipients
    target_chats = [chat_id] if chat_id else get_active_chat_ids()

    if not target_chats:
        print("⚠️ No active chats to send to")
        return

    for cid in target_chats:
        payload = {
            "chat_id": cid,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }

        try:
            response = session.post(f"{API_URL}/sendMessage", json=payload, timeout=10)
            data = response.json()

            if response.status_code == 200 and data.get("ok"):
                print(f"✅ Message sent to chat {cid}")
            else:
                err_desc = data.get("description", "Unknown error")
                print(f"❌ Failed to send to chat {cid}: {err_desc}")

        except requests.RequestException as e:
            print(f"❌ Network error sending to chat {cid}: {e}")
