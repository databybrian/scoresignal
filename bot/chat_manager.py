# bot/chat_manager.py
"""
Chat manager for handling active Telegram chat IDs.
Stores IDs in a local JSON file under /data.
"""

import json
from pathlib import Path
from typing import Set

CHAT_IDS_FILE = Path(__file__).parent.parent / "data" / "active_chats.json"


def load_chat_ids() -> Set[str]:
    """Load active chat IDs from file (returns empty set if missing or invalid)."""
    if not CHAT_IDS_FILE.exists():
        return set()

    try:
        with open(CHAT_IDS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            chats = data.get("chats", [])
            return set(str(chat_id) for chat_id in chats)
    except (json.JSONDecodeError, OSError, KeyError) as e:
        print(f"⚠️ Failed to load chat IDs ({e}), resetting file")
        return set()


def save_chat_ids(chat_ids: Set[str]) -> None:
    """Save chat IDs to file (atomic write)."""
    CHAT_IDS_FILE.parent.mkdir(exist_ok=True)

    tmp_file = CHAT_IDS_FILE.with_suffix(".tmp")
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump({"chats": sorted(chat_ids)}, f, indent=2)
        tmp_file.replace(CHAT_IDS_FILE)
    except OSError as e:
        print(f"❌ Failed to save chat IDs: {e}")


def add_chat_id(chat_id: str) -> None:
    """Add a new chat ID to active chats."""
    chat_ids = load_chat_ids()
    if str(chat_id) not in chat_ids:
        chat_ids.add(str(chat_id))
        save_chat_ids(chat_ids)
        print(f"✅ Added new chat ID: {chat_id}")
    else:
        print(f"ℹ️ Chat ID already active: {chat_id}")


def remove_chat_id(chat_id: str) -> None:
    """Remove a chat ID from active chats."""
    chat_ids = load_chat_ids()
    if str(chat_id) in chat_ids:
        chat_ids.remove(str(chat_id))
        save_chat_ids(chat_ids)
        print(f"🗑️ Removed chat ID: {chat_id}")
    else:
        print(f"ℹ️ Chat ID not found: {chat_id}")


def get_active_chat_ids() -> Set[str]:
    """Return all active chat IDs."""
    return load_chat_ids()
