# bot/telegram_bot.py
"""
Telegram bot messaging module.
Handles sending predictions or other messages to Telegram chats.
"""

import requests
from typing import Optional
from .config import TELEGRAM_BOT_TOKEN
from .chat_manager import get_active_chat_ids

# Reuse one session for efficiency
session = requests.Session()
API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


def send_telegram_message(message: str, chat_id: Optional[str] = None):
    """
    Send message to specific chat ID or all active chats.
    """
    if not TELEGRAM_BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not configured")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    # If specific chat_id provided, send only to that chat
    if chat_id:
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                print(f"✅ Message sent to chat {chat_id}")
            else:
                print(f"❌ Failed to send to chat {chat_id}: {response.status_code}")
        except Exception as e:
            print(f"❌ Error sending to chat {chat_id}: {e}")
        return
    
    # Otherwise, send to all active chats (existing logic)
    active_chats = get_active_chat_ids()
    if not active_chats:
        print("⚠️  No active chats to send to")
        return
    
    for cid in active_chats:
        payload = {
            "chat_id": cid,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                print(f"✅ Message sent to chat {cid}")
            else:
                print(f"❌ Failed to send to chat {cid}: {response.status_code}")
        except Exception as e:
            print(f"❌ Error sending to chat {cid}: {e}")