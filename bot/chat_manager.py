# bot/chat_manager.py
"""
Chat manager for handling active Telegram chat IDs using PostgreSQL (Railway DB).
"""

import os
import psycopg2
from typing import Set

# Railway provides this automatically; ensure it is set in your environment
DB_URL = os.getenv("DATABASE_URL")
conn = psycopg2.connect(DB_URL, sslmode="require")


def _get_conn():
    """Get a new database connection with SSL required for Railway."""
    if not DB_URL:
        raise RuntimeError("❌ DATABASE_URL not configured")
    return psycopg2.connect(DB_URL, sslmode="require")


def add_chat_id(chat_id: int, username: str = None) -> None:
    """Add a new chat ID if not already present."""
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO telegram_chats (chat_id, username)
                    VALUES (%s, %s)
                    ON CONFLICT (chat_id) DO NOTHING
                    """,
                    (chat_id, username),
                )
        print(f"✅ Added chat ID {chat_id}")
    finally:
        conn.close()


def remove_chat_id(chat_id: int) -> None:
    """Remove a chat ID."""
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM telegram_chats WHERE chat_id=%s", (chat_id,))
        print(f"🗑️ Removed chat ID {chat_id}")
    finally:
        conn.close()


def get_active_chat_ids() -> Set[int]:
    """Return all currently active chat IDs."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT chat_id FROM telegram_chats")
            rows = cur.fetchall()
        return set(row[0] for row in rows)
    finally:
        conn.close()
